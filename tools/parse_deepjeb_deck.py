#!/usr/bin/env python3
"""Read exact interface node IDs and loads from original DeepJEB OptiStruct FEM.

parse_deck(path) returns a dictionary with a full `grid` ID -> XYZ mapping and
`summary`, a JSON-serializable description of constraints, interfaces and loads.
Do not assume GRID ID - 1 is the corresponding H5 vertex index: map coordinates.
The parser validates the one-weight-group RBE3 form present in this release and
fails explicitly for unsupported forms rather than guessing interface membership.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def nastran_float(value: str) -> float:
    value = value.strip().replace("D", "E").replace("d", "e")
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        value = re.sub(r"(?<=[0-9.])([+-][0-9]+)$", r"e\1", value)
        return float(value)


def fixed_cards(path: Path):
    """Yield small-field cards (8 chars), preserving significant blank fields."""
    name, values = None, []
    bulk = False
    with path.open(encoding="utf-8", errors="strict") as source:
        for raw in source:
            line = raw.rstrip("\r\n")
            if line.startswith("BEGIN BULK"):
                bulk = True
                continue
            if not bulk or not line.strip() or line.startswith("$"):
                continue
            if "," in line:
                # Release includes unrelated PARAM/GAPPRM free-field controls.
                if name:
                    yield name, values
                    name, values = None, []
                continue
            first = line[:8].strip()
            if first.startswith("*"):
                raise ValueError("Large-field continuation is unsupported")
            if first.startswith("+") or not first:
                if name:
                    values.extend(line[start:start + 8].strip() for start in range(8, 72, 8))
                continue
            if name:
                yield name, values
            if first.endswith("*"):
                raise ValueError("Large-field card is unsupported")
            name = first
            values = [line[start:start + 8].strip() for start in range(8, 72, 8)]
        if name:
            yield name, values


def parse_deck(path: str | Path) -> dict:
    path = Path(path)
    subcases = {}
    current = None
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.startswith("BEGIN BULK"):
                break
            match = re.match(r"SUBCASE\s+(\d+)", line)
            if match:
                current = int(match[1])
                subcases[current] = {}
            match = re.match(r"(LABEL|ANALYSIS|SPC|LOAD)\s*=\s*(.*?)\s*$", line)
            if current is not None and match:
                key, value = match.groups()
                subcases[current][key.lower()] = int(value) if key in {"SPC", "LOAD"} else value
    grid, rbe2, rbe3, spc, spcadd, actions, loadadd, materials = {}, {}, {}, {}, {}, {}, {}, {}
    for name, values in fixed_cards(path):
        while values and not values[-1]:
            values.pop()
        if name == "GRID":
            if values[1] and int(values[1]) != 0:
                raise ValueError("GRID uses non-global coordinate system")
            grid[int(values[0])] = tuple(nastran_float(item) for item in values[2:5])
        elif name == "RBE2":
            eid, master, dofs = int(values[0]), int(values[1]), values[2]
            nodes = [int(item) for item in values[3:] if item]
            rbe2[eid] = {"element_id": eid, "reference_node_id": master, "dofs": dofs, "node_ids": nodes}
        elif name == "RBE3":
            eid, master, dofs = int(values[0]), int(values[2]), values[3]
            weight, node_dofs = nastran_float(values[4]), values[5]
            if weight != 1.0 or node_dofs != "123":
                raise ValueError(f"Unsupported RBE3 weighting in element {eid}")
            raw_nodes = [item for item in values[6:] if item]
            if any(not item.isdigit() for item in raw_nodes):
                raise ValueError(f"Unsupported additional RBE3 groups in element {eid}")
            rbe3[eid] = {"element_id": eid, "reference_node_id": master, "reference_dofs": dofs,
                         "node_dofs": node_dofs, "weight": weight, "node_ids": [int(item) for item in raw_nodes]}
        elif name == "SPC":
            sid = int(values[0])
            for offset in (1, 4):
                if offset < len(values) and values[offset]:
                    spc.setdefault(sid, []).append({"node_id": int(values[offset]), "dofs": values[offset + 1],
                                                  "value": nastran_float(values[offset + 2])})
        elif name == "SPCADD":
            spcadd[int(values[0])] = [int(item) for item in values[1:] if item]
        elif name in {"FORCE", "MOMENT"}:
            sid, node = int(values[0]), int(values[1])
            if values[2] and int(values[2]) != 0:
                raise ValueError("Load uses non-global coordinate system")
            magnitude = nastran_float(values[3])
            direction = [nastran_float(item) for item in values[4:7]]
            actions.setdefault(sid, []).append({"kind": name.lower(), "node_id": node, "magnitude": magnitude,
                                               "direction": direction, "vector": [magnitude * value for value in direction],
                                               "units": "N" if name == "FORCE" else "N mm"})
        elif name == "LOAD":
            sid, scale = int(values[0]), nastran_float(values[1])
            pairs = [item for item in values[2:] if item]
            if len(pairs) % 2:
                raise ValueError(f"Incomplete LOAD combination {sid}")
            loadadd[sid] = [(scale * nastran_float(pairs[index]), int(pairs[index + 1])) for index in range(0, len(pairs), 2)]
        elif name == "MAT1":
            mid = int(values[0])
            materials[mid] = {"material_id": mid, "E": nastran_float(values[1]), "nu": nastran_float(values[3]),
                              "rho": nastran_float(values[4]), "E_units": "MPa", "rho_units": "t/mm3",
                              "E_MPa": nastran_float(values[1]), "rho_kg_per_mm3": nastran_float(values[4]) * 1000}

    def constraints(sid):
        if sid in spcadd:
            return [item for child in spcadd[sid] for item in constraints(child)]
        return spc.get(sid, [])

    def expanded_loads(sid, scale=1.0):
        if sid in loadadd:
            return [item for child_scale, child in loadadd[sid] for item in expanded_loads(child, scale * child_scale)]
        result = []
        for item in actions.get(sid, []):
            result.append({**item, "magnitude": item["magnitude"] * scale,
                           "vector": [value * scale for value in item["vector"]], "source_sid": sid})
        return result

    for info in subcases.values():
        if "spc" in info:
            info["constraints"] = constraints(info["spc"])
        if "load" in info:
            info["actions"] = expanded_loads(info["load"])
    constrained = {item["node_id"] for info in subcases.values() for item in info.get("constraints", [])}
    loaded = {item["node_id"] for info in subcases.values() for item in info.get("actions", [])}
    supports = [item for item in rbe2.values() if item["reference_node_id"] in constrained]
    loads = [item for item in rbe3.values() if item["reference_node_id"] in loaded]
    for interface in supports + loads:
        interface["reference_coordinates_mm"] = grid[interface["reference_node_id"]]
        interface["num_nodes"] = len(interface["node_ids"])
        if any(node not in grid for node in interface["node_ids"]):
            raise ValueError("RBE interface contains undefined GRID node")
    summary = {"case": path.stem, "source_deck": str(path.resolve()), "units": "MPa (mm t N s)",
               "grid_count": len(grid), "subcases": subcases, "support_interfaces": supports, "load_interfaces": loads,
               "fixed_mesh_node_ids": sorted({node for interface in supports for node in interface["node_ids"]}),
               "loaded_mesh_node_ids": sorted({node for interface in loads for node in interface["node_ids"]}),
               "coordinate_mapping_note": "Match FEM GRID coordinates to H5 vertices; do not use node ID - 1 as H5 index."}
    for interface in supports + loads:
        interface["master_node_id"] = interface["reference_node_id"]
        interface["master_coords"] = interface["reference_coordinates_mm"]
        interface["member_node_ids"] = interface["node_ids"]
    label_to_case = {"Vertical": "ver", "Horizontal": "hor", "Diagonal": "dia", "Torsion": "tor"}
    cases = {}
    for info in subcases.values():
        if info.get("label") in label_to_case:
            force, moment = [0.0] * 3, [0.0] * 3
            for action in info.get("actions", []):
                accumulator = force if action["kind"] == "force" else moment
                for index, value in enumerate(action["vector"]):
                    accumulator[index] += value
            cases[label_to_case[info["label"]]] = {"force": force, "moment": moment,
                                                   "force_units": "N", "moment_units": "N mm",
                                                   "actions": info.get("actions", [])}
    if len(loads) != 1 or len(materials) != 1:
        raise ValueError(f"Expected one RBE3 loaded interface and one MAT1; got {len(loads)}, {len(materials)}")
    summary["cases"] = cases
    summary["material"] = next(iter(materials.values()))
    return {"grid": grid, "node_coordinates": grid, "summary": summary, "fixed_groups": supports,
            "loaded_group": loads[0], "cases": cases, "material": summary["material"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deck", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/raw/BoundaryConditions"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for deck in args.deck:
        parsed = parse_deck(deck)
        summary = parsed["summary"]
        target = args.output / (deck.stem + ".json")
        target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"{deck.stem}: {summary['grid_count']} GRID, "
              f"{len(summary['fixed_mesh_node_ids'])} support nodes, {len(summary['loaded_mesh_node_ids'])} load nodes -> {target}", flush=True)


if __name__ == "__main__":
    main()
