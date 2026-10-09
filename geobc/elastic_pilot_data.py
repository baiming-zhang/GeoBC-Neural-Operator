"""Single TRAIN101_479 target-free physical-input specialization.

Metadata APIs import only the standard library. Actual preparation opens only
the four declared input sources; labels remain deferred to the separate solve.
This is an explicit approximate physical model, not an OptiStruct replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "geobc/elastic_pilot_data.py"
DEFAULT_PLAN = "notes/geobc_elastic_pilot_plan.json"
GEOMETRY_PLAN = "notes/geobc_continuous_geometry_plan.json"
GEOMETRY_PLAN_SHA = "16169bbd20fc1d746cc8ccf88fa9d4c287fa410744d4ee8b6f0bb210543bb71f"
SCHEMA = "elastic_pilot_fixed_support_input_v1"
CAD, CASE = "101_479", "101_479/ver"
PATHS = {"deck":"data/raw/SolverDeck/101_479.fem", "geometry_h5":"data/raw/FieldMesh/101_479.h5",
    "cells_npz":"data/geobc_gradient_supervision/101_479.npz",
    "canonical_npz":"data/prepared_geobc_multiload/101_479_ver.npz", "geometry_metadata":GEOMETRY_PLAN}
MEMBERS = {"geometry_h5":["vertices"], "cells_npz":["cells10"], "canonical_npz":["xyz","grid_node_ids"]}
EDGES = ((0,1),(1,2),(2,0),(0,3),(1,3),(2,3))


class UnsupportedInput(RuntimeError):
    """A scientific/input-model negative, rather than a corrupt-source exception."""


def _path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", value):
        value = value[5].upper()+":/"+value[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", value):
        value = "/mnt/"+value[0].lower()+"/"+value[3:]
    result = Path(value).expanduser()
    return (result if result.is_absolute() else ROOT/result).resolve()


def _json(path):
    return json.loads(_path(path).read_text(encoding="utf-8-sig"))


def _sha(path):
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _identity(path, sha=False):
    path = _path(path); stat = path.stat()
    result = {"path":path.relative_to(ROOT).as_posix(),"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns}
    if sha:
        result["sha256"] = _sha(path)
    return result


def _match(path, identity):
    actual = _identity(path)
    if actual != {key:identity[key] for key in ("path","bytes","mtime_ns")}:
        raise ValueError(f"Source/cache identity changed: {actual['path']}")


def _write(path, value):
    path = _path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def _require(condition, message, error=ValueError):
    if not condition:
        raise error(message)


def request_contract(plan_path=DEFAULT_PLAN):
    """Pure JSON, source SHA and source-file stat; no deck/array content reads."""
    plan = _json(plan_path)
    _require(plan.get("cad") == CAD and plan.get("case") == CASE,"Only the frozen single TRAIN/ver input is allowed")
    _require(plan.get("input_members") == MEMBERS,"Input member allowlist changed")
    _require(plan.get("post_solve_label_members") == {"canonical_npz":["displacement"]},"Deferred label scope changed")
    _require(_path(plan["data_output"]) == _path("data/geobc_elastic_input101479"),"Input cache destination changed")
    for name,expected in PATHS.items():
        _require(_path(plan["sources"][name]) == _path(expected),f"Input path changed: {name}")
    _require(_sha(GEOMETRY_PLAN) == GEOMETRY_PLAN_SHA,"Inherited geometry metadata changed")
    geometry = _json(GEOMETRY_PLAN)["physical_XYZ_path_closed_at_metadata_level"]
    inherited = geometry["inherited_array_identity"]
    expected = plan["expected_geometry"]
    names = {"vertices_float32_sha256":"vertices_float32_le_Corder_sha256",
        "cells10_int64_sha256":"cells10_int64_le_Corder_sha256",
        "canonical_xyz_float32_sha256":"canonical_normalized_xyz_float32_sha256",
        "grid_node_ids_int64_sha256":"grid_node_ids_int64_le_Corder_sha256"}
    _require(expected["nodes"] == 179316 and expected["cells"] == 111754
        and all(expected[a] == inherited[b] for a,b in names.items())
        and expected["characteristic_length_mm"] == geometry["normalization"]["characteristic_length_mm"]
        and expected["bbox_min_mm"] == geometry["normalization"]["bbox_min_mm"],"Canonical geometry identity differs")
    split = _json(plan["existing_split"])
    _require(CAD in split["groups"]["train"] and _path(split["case_paths"][CASE]) == _path(PATHS["canonical_npz"]),"Case is not the fixed TRAIN source")
    recipe = plan["recipe"]
    _require(recipe["node_dofs"] == 3 and recipe["dtype"] == "float64"
        and recipe["quadrature_points_per_cell"] == 4 and recipe["load_transfer"]["minimum_rank"] == 6
        and recipe["load_transfer"]["maximum_normal_condition"] == 1e12
        and recipe["post_solve"]["stress_scalar_comparison"] is False,"Physical input model changed")
    for key in ("old_final_reads","DEV_dia_reads","sealed_new_test_reads","training_updates","scaler_fits"):
        _require(recipe["post_solve"][key] == 0,f"Forbidden scope changed: {key}")
    source_sha = _sha(SOURCE)
    _require(plan.get("source_sha256",{}).get(SOURCE,source_sha) == source_sha,"Pinned input source changed")
    contract = {"kind":"elastic_pilot_input","schema":SCHEMA,"cad":CAD,"case":CASE,
        "source_sha256":{SOURCE:source_sha},"design_sha256":plan["design_sha256"],
        "split_sha256":_sha(plan["existing_split"]),"geometry_metadata_sha256":GEOMETRY_PLAN_SHA,
        "sources":{name:_identity(path) for name,path in PATHS.items() if name != "geometry_metadata"},
        "input_members":MEMBERS,"expected_geometry":expected,"recipe":recipe,
        "cache":_path(plan["data_output"]).relative_to(ROOT).as_posix(),
        "deferred_displacement_source":_identity(PATHS["canonical_npz"]),"labels_read":0,"stress_reads":0,
        "scientific_array_reads_this_request":0,"raw_deck_read_this_request":False}
    _require(_sha(plan["design"]) == plan["design_sha256"],"Physical design identity changed")
    return contract,plan


def read_cache_metadata(plan_path=DEFAULT_PLAN, *, contract=None):
    """Reject negative/incomplete input before operator CUDA initialization."""
    if contract is None:
        contract,plan = request_contract(plan_path)
    else:
        plan = _json(plan_path)
    manifest = _json(_path(plan["data_output"])/"manifest.json")
    _require(manifest.get("status") == "completed" and manifest.get("contract") == contract,"Input cache incomplete or contract changed")
    _require(manifest.get("feasibility_pass") is True,"Physical input specialization is unsupported; do not initialize operator/GPU")
    for name,identity in contract["sources"].items():
        _match(PATHS[name],identity)
    _match(manifest["cache_file"],manifest["file_identity"])
    return manifest


def _number(text, default=None):
    if not text.strip():
        return default
    text = text.strip().replace("D","E").replace("d","e")
    text = re.sub(r"(?<=[0-9.])([+-][0-9]+)$",r"e\1",text) if "e" not in text.lower() else text
    value = float(text)
    _require(math.isfinite(value),"Nonfinite deck number")
    return value


def _integer(text, default=None):
    if not text.strip():
        return default
    _require(re.fullmatch(r"[+-]?\d+",text.strip()) is not None,f"Invalid integer field: {text!r}")
    return int(text)


def _dofs(text, allow_zero=False):
    if text.strip() in ("","0") and allow_zero:
        return set()
    _require(text and all(c in "123456" for c in text) and len(set(text)) == len(text),f"Invalid DOFs: {text!r}")
    return set(text)


def _deadline(began, budget):
    _require(time.monotonic()-began <= budget,"combined_input_operator_budget_exhausted_during_input",UnsupportedInput)


def _cards(path, scope, began, budget):
    """Read small/free cards with significant empty fields and explicit controls."""
    bulk = False; name = None; fields = []; card_line = None; current = None
    controls = {}; defaults = {}; ignored = []
    outputs = set(scope["ignored_output_cards"])
    with _path(path).open(encoding="utf-8",errors="strict") as handle:
        for line_number,raw in enumerate(handle,1):
            if line_number % 16384 == 0:
                _deadline(began,budget)
            line = raw.split("$",1)[0].rstrip("\r\n")
            stripped = line.strip()
            if not stripped:
                continue
            if not bulk:
                if re.fullmatch(r"BEGIN\s+BULK",stripped,re.I):
                    bulk = True; continue
                match = re.fullmatch(r"SUBCASE\s+(\d+)",stripped,re.I)
                if match:
                    current = int(match[1]); _require(current not in controls,"Duplicate SUBCASE")
                    controls[current] = dict(defaults); continue
                match = re.match(r"([A-Z][A-Z0-9_]*)(?:\([^)]*\))?\s*=\s*(.*?)\s*$",stripped,re.I)
                if match:
                    key,value = match[1].upper(),match[2]
                    target = defaults if current is None else controls[current]
                    if key in {"SPC","LOAD"}:
                        target[key] = _integer(value)
                    elif key == "METHOD":
                        option = re.search(r"\(([^)]*)\)",stripped.split("=",1)[0])
                        _require(current is not None and (option is None or option[1].upper() == "STRUCTURE"),
                            "Only subcase structural METHOD is supported",UnsupportedInput)
                        _require("METHOD" not in target,"Repeated METHOD control",UnsupportedInput)
                        target[key] = _integer(value); target["METHOD_line"] = line_number
                    elif key in {"LABEL","ANALYSIS"}:
                        target[key] = value.strip()
                    elif key in outputs or key in {"TITLE","SUBTITLE"}:
                        ignored.append({"line":line_number,"card":key,"fields":[value],"reason":"output_only"})
                    else:
                        raise UnsupportedInput(f"Unsupported case-control {key} at line {line_number}")
                    continue
                if stripped.upper() == "CEND" or re.fullmatch(r"SOL[\s,]+101",stripped,re.I):
                    continue
                key = re.split(r"[\s,(]",stripped.upper(),1)[0]
                if key == "DEBUG":
                    parts = [part for part in re.split(r"[\s,]+",stripped.upper()) if part]
                    _require(parts == ["DEBUG","LGSHPSNS","0.0"],
                        f"Unsupported DEBUG setting at line {line_number}",UnsupportedInput)
                    ignored.append({"line":line_number,"card":"DEBUG","fields":parts[1:],
                        "reason":"definition_unverified_explicitly_not_reproduced",
                        "original_setting_effect":"unknown; not asserted inactive or output-only"})
                    continue
                if key in outputs:
                    ignored.append({"line":line_number,"card":key,"fields":[stripped],"reason":"output_only"}); continue
                raise UnsupportedInput(f"Unsupported pre-BULK entry at line {line_number}: {stripped[:100]}")
            if stripped.upper() == "ENDDATA":
                if name is not None:
                    yield name,fields,card_line,controls,ignored
                return
            if "," in line:
                parts = [part.strip() for part in line.split(",")]
                first,values = parts[0],parts[1:]
                if values and values[-1].startswith("+"):
                    values.pop()
            else:
                first = line[:8].strip(); values = [line[start:start+8].strip() for start in range(8,72,8)]
            _require(not first.endswith("*") and not first.startswith("*"),"Large-field cards are unsupported",UnsupportedInput)
            if not first or first.startswith("+"):
                _require(name is not None,f"Orphan continuation at line {line_number}")
                fields.extend(values); continue
            if name is not None:
                yield name,fields,card_line,controls,ignored
            name,fields,card_line = first.upper(),values,line_number
        _require(bulk,"Missing BEGIN BULK")
        if name is not None:
            yield name,fields,card_line,controls,ignored


def _parse_deck(path, recipe, began, budget):
    scope = recipe["reader_scope"]
    data = {name:{} for name in ("grid","rbe2","rbe3","spc","spcadd","actions","loadadd","materials","psolid","ctetra","eigrl")}
    defaults = None; counts = Counter(); ignored_bulk = []; controls = {}; ignored_header = []
    def put(table,key,value):
        _require(key not in data[table],f"Duplicate {table} ID {key}")
        data[table][key] = value
    for name,values,line,controls,ignored_header in _cards(path,scope,began,budget):
        counts[name] += 1
        while values and not values[-1]:
            values.pop()
        v = values+[""]*16
        if name == "GRID":
            put("grid",_integer(v[0]),{"xyz":[_number(s) for s in v[2:5]],"CP":v[1],"CD":v[5],"PS":v[6],"SEID":v[7]})
        elif name == "GRDSET":
            _require(defaults is None and not v[0],"Multiple/nonstandard GRDSET unsupported",UnsupportedInput)
            defaults = {"CP":v[1],"CD":v[5],"PS":v[6],"SEID":v[7]}
        elif name == "RBE2":
            dofs = _dofs(v[2]); _require(set("123") <= dofs,"RBE2 CM must include123",UnsupportedInput)
            members = [_integer(s) for s in values[3:] if s]
            _require(members and len(set(members)) == len(members) and all(n>0 for n in members),"RBE2 duplicate/thermal/non-node fields unsupported",UnsupportedInput)
            put("rbe2",_integer(v[0]),{"reference":_integer(v[1]),"dofs":"".join(sorted(dofs)),"members":members})
        elif name == "RBE3":
            _require(not v[1] and v[5] == "123" and not any(s.upper() in {"UM","ALPHA"} for s in values),"Only one-group Ci123 RBE3 without UM/ALPHA is supported",UnsupportedInput)
            weight = _number(v[4]); _require(weight is not None and weight>0,"RBE3 weight must be positive",UnsupportedInput)
            refc = _dofs(v[3]); members = [_integer(s) for s in values[6:] if s]
            _require(members and len(set(members)) == len(members) and all(n>0 for n in members),"RBE3 member list unsupported",UnsupportedInput)
            put("rbe3",_integer(v[0]),{"reference":_integer(v[2]),"REFC":"".join(sorted(refc)),"weight":weight,"members":members})
        elif name in {"SPC","SPC1"}:
            _require(name == "SPC","SPC1 is not supported in this fixed specialization",UnsupportedInput)
            _require(len(values) in {4,7},"Malformed SPC field count")
            sid = _integer(v[0])
            for offset in (1,4):
                if v[offset]:
                    data["spc"].setdefault(sid,[]).append((_integer(v[offset]),_dofs(v[offset+1]),_number(v[offset+2],0.)))
        elif name == "SPCADD":
            put("spcadd",_integer(v[0]),[_integer(s) for s in values[1:] if s])
        elif name in {"FORCE","MOMENT"}:
            _require(_integer(v[2],0) == 0,"Nonbasic load coordinate unsupported",UnsupportedInput)
            magnitude = _number(v[3]); direction = [_number(s) for s in v[4:7]]
            _require(magnitude is not None and None not in direction,"Incomplete load vector")
            data["actions"].setdefault(_integer(v[0]),[]).append({"kind":name,"node":_integer(v[1]),"vector":[magnitude*s for s in direction]})
        elif name == "LOAD":
            rest = values[2:]; _require(len(rest)%2 == 0 and rest and all(rest),"Malformed LOAD pairs")
            scale = _number(v[1]); _require(scale is not None,"Missing LOAD scale")
            put("loadadd",_integer(v[0]),[(scale*_number(rest[i]),_integer(rest[i+1])) for i in range(0,len(rest),2)])
        elif name == "MAT1":
            put("materials",_integer(v[0]),{"E":_number(v[1]),"G":_number(v[2]),"nu":_number(v[3]),"remaining_fields":values[4:]})
        elif name == "PSOLID":
            _require(v[6].upper() in {"","SMECH"},"PSOLID nonstructural FCTN unsupported",UnsupportedInput)
            put("psolid",_integer(v[0]),{"MID":_integer(v[1]),"original_controls_not_replicated":values[2:]})
        elif name == "CTETRA":
            _require(len(values) == 12 and all(values),"Only complete CTETRA10 is supported",UnsupportedInput)
            put("ctetra",_integer(v[0]),{"PID":_integer(v[1]),"nodes":[_integer(s) for s in values[2:]]})
        elif name == "EIGRL":
            sid = _integer(v[0]); _require(sid is not None and sid>0,"Invalid EIGRL SID")
            put("eigrl",sid,{"line":line,"fields":list(values)})
        elif name == "PARAM":
            key = v[0].upper(); allowed = scope["allowed_static_params"].get(key)
            _require(allowed is not None and v[1].upper() in {str(s).upper() for s in allowed}
                and not any(values[2:]),f"Unsupported PARAM {values} at line {line}",UnsupportedInput)
            reasons = {"CHECKEL":"original_quality_check_control_not_reproduced_own_mathematical_guards_retained",
                "RENUMOK":"original_reordering_not_reproduced_canonical_connectivity_unchanged",
                "PRINFACC":"inertia_relief_output_control_not_reproduced",
                "CONTFEL":"contact_friendly_control_excluded_no_contact_cards_supported"}
            ignored_bulk.append({"line":line,"card":name,"fields":values,
                "reason":reasons.get(key,"explicit_static_control_not_replicated")})
        elif name == "GAPPRM":
            _require(v[0].upper() in scope["allowed_gapprm_names"] and v[0].upper() == "GAPCMPL"
                and v[1].upper() == "NO" and not any(values[2:]),
                f"Unsupported GAPPRM at line {line}",UnsupportedInput)
            ignored_bulk.append({"line":line,"card":name,"fields":values,
                "reason":"gap_contact_compliance_control_excluded_no_contact_cards_supported",
                "does_not_disable_contact":True})
        elif name in scope["ignored_output_cards"]:
            ignored_bulk.append({"line":line,"card":name,"fields":values,"reason":"output_only"})
        else:
            raise UnsupportedInput(f"Unsupported BULK card {name} at line {line}")
    defaults = defaults or {"CP":"","CD":"","PS":"","SEID":""}
    for node,row in data["grid"].items():
        _require(node>0 and all(value is not None for value in row["xyz"]),"Malformed GRID")
        for field in defaults:
            row[field] = row[field] if row[field] else defaults[field]
        _require(_integer(row["CP"],0) == _integer(row["CD"],0) == _integer(row["SEID"],0) == 0,
            f"Nonbasic/superelement GRID unsupported: {node}",UnsupportedInput)
        row["PS"] = _dofs(row["PS"],True)
    matches = [row for row in controls.values() if row.get("LABEL","").lower() == "vertical"]
    _require(len(matches) == 1 and matches[0].get("ANALYSIS","").upper() == "STATICS","Require one labelled Vertical STATICS subcase",UnsupportedInput)
    selected = matches[0]; _require({"SPC","LOAD"} <= selected.keys(),"Vertical subcase missing SPC/LOAD")
    _require("METHOD" not in selected,"Selected Vertical static subcase has METHOD",UnsupportedInput)
    unselected_modes = {}
    for subcase,row in controls.items():
        if "METHOD" not in row:
            continue
        _require(row is not selected and row.get("ANALYSIS","").upper() in {"","MODES"},
            f"METHOD in non-modal subcase {subcase} is unsupported",UnsupportedInput)
        sid = row["METHOD"]
        _require(sid in data["eigrl"],f"Unselected METHOD {sid} does not reference EIGRL",UnsupportedInput)
        unselected_modes.setdefault(sid,[]).append(subcase)
        ignored_header.append({"line":row["METHOD_line"],"card":"METHOD","fields":[sid],
            "subcase":subcase,"reason":"unselected_normal_modes_not_part_of_static_operator"})
    _require(set(data["eigrl"]) <= set(unselected_modes),
        "EIGRL not exclusively referenced by an unselected modal METHOD",UnsupportedInput)
    for sid,row in data["eigrl"].items():
        ignored_bulk.append({"line":row["line"],"card":"EIGRL","fields":row["fields"],
            "unselected_subcases":unselected_modes[sid],"reason":"unselected_normal_modes_not_part_of_static_operator"})
    def constraints(sid, stack=()):
        _require(sid not in stack,"SPCADD dependency cycle",UnsupportedInput)
        if sid in data["spcadd"]:
            _require(sid not in data["spc"],"SPC/SPCADD SID collision",UnsupportedInput)
            return [r for child in data["spcadd"][sid] for r in constraints(child,stack+(sid,))]
        _require(sid in data["spc"],f"Undefined active SPC set {sid}")
        return data["spc"][sid]
    def actions(sid, scale=1., stack=()):
        _require(sid not in stack,"LOAD dependency cycle",UnsupportedInput)
        if sid in data["loadadd"]:
            _require(sid not in data["actions"],"LOAD/action SID collision",UnsupportedInput)
            return [r for factor,child in data["loadadd"][sid] for r in actions(child,scale*factor,stack+(sid,))]
        _require(sid in data["actions"],f"Undefined active LOAD set {sid}")
        return [{**r,"vector":[scale*x for x in r["vector"]]} for r in data["actions"][sid]]
    data.update(selected_controls=selected,active_constraints=constraints(selected["SPC"]),active_actions=actions(selected["LOAD"]),
        cards=dict(counts),ignored_controls=ignored_header+ignored_bulk,GRDSET=defaults)
    # Check the complete input graphs without using any other load as a solve.
    for sid in data["spcadd"]:
        constraints(sid)
    for sid in data["loadadd"]:
        actions(sid)
    return data


def _array_sha(value, dtype):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value,dtype=dtype).tobytes()).hexdigest()


def _inputs(contract, receipt):
    import h5py
    import numpy as np
    for name,identity in contract["sources"].items():
        _match(PATHS[name],identity)
    expected = contract["expected_geometry"]; n,t = expected["nodes"],expected["cells"]
    with h5py.File(_path(PATHS["geometry_h5"]),"r") as source:
        dataset = source["vertices"]
        _require(dataset.shape == (n,3) and dataset.dtype == np.dtype("float32"),"H5 vertices schema differs")
        vertices = dataset[...]
        receipt["accessed_members"][PATHS["geometry_h5"]] = ["vertices"]
    _require(_array_sha(vertices,"<f4") == expected["vertices_float32_sha256"],"Source vertices SHA differs")
    with np.load(_path(PATHS["cells_npz"]),allow_pickle=False) as source:
        cells = source["cells10"]
        receipt["accessed_members"][PATHS["cells_npz"]] = ["cells10"]
    with np.load(_path(PATHS["canonical_npz"]),allow_pickle=False) as source:
        xyz,ids = source["xyz"],source["grid_node_ids"]
        receipt["accessed_members"][PATHS["canonical_npz"]] = ["xyz","grid_node_ids"]
    _require(cells.shape == (t,10) and cells.dtype == np.dtype("int64") and ids.shape == (n,)
        and ids.dtype == np.dtype("int64") and xyz.shape == (n,3) and xyz.dtype == np.dtype("float32"),"Canonical input schema differs")
    _require(_array_sha(cells,"<i8") == expected["cells10_int64_sha256"]
        and _array_sha(ids,"<i8") == expected["grid_node_ids_int64_sha256"]
        and _array_sha(xyz,"<f4") == expected["canonical_xyz_float32_sha256"],"Canonical array SHA differs")
    reproduced = ((vertices-np.asarray(expected["bbox_min_mm"],np.float32))/expected["characteristic_length_mm"]).astype(np.float32)
    _require(np.array_equal(reproduced,xyz) and np.isfinite(vertices).all() and np.isfinite(xyz).all()
        and (cells>=0).all() and (cells<n).all() and (np.diff(np.sort(cells,axis=1),axis=1)>0).all()
        and (ids>0).all() and len(np.unique(ids)) == n,"Corrupt source geometry/order")
    return vertices.astype(np.float64),cells,ids


def _build(deck,vertices,cells,ids,contract,began,budget):
    import numpy as np
    n = len(vertices); recipe = contract["recipe"]; expected = contract["expected_geometry"]
    length = expected["characteristic_length_mm"]; rows = {int(g):i for i,g in enumerate(ids)}
    _require(set(rows) <= deck["grid"].keys(),"Canonical GRID IDs missing from deck")
    deck_xyz = np.asarray([deck["grid"][int(g)]["xyz"] for g in ids],np.float64)
    coordinate_error = float(np.max(np.abs(deck_xyz-vertices))/length)
    _require(coordinate_error <= recipe["reader_scope"]["deck_coordinate_relative_tolerance"],"Deck/canonical coordinate disagreement",UnsupportedInput)
    used = np.zeros(n,bool); used[cells.ravel()] = True
    corners = {tuple(sorted(int(g) for g in ids[cell[:4]])):i for i,cell in enumerate(cells)}
    _require(len(corners) == len(cells),"Duplicate cached tetra corner set")
    element_ids = np.full(len(cells),-1,np.int64); material_ids = set()
    _require(len(deck["ctetra"]) == len(cells),"Deck/cached TET10 coverage differs",UnsupportedInput)
    for count,(eid,element) in enumerate(deck["ctetra"].items()):
        if count%8192 == 0:
            _deadline(began,budget)
        nodes = element["nodes"]; key = tuple(sorted(nodes[:4]))
        _require(key in corners and len(set(nodes)) == 10 and set(nodes) <= rows.keys(),"Deck CTETRA has no canonical cell or invalid GRID mapping")
        index = corners[key]; _require(element_ids[index] == -1,"Duplicate deck CTETRA corner set")
        supplied = {tuple(sorted((nodes[a],nodes[b]))):nodes[4+j] for j,(a,b) in enumerate(EDGES)}
        cached = [int(g) for g in ids[cells[index]]]
        associated = {tuple(sorted((cached[a],cached[b]))):cached[4+j] for j,(a,b) in enumerate(EDGES)}
        _require(supplied == associated,"CTETRA/cached corner-midpoint edge association differs")
        _require(element["PID"] in deck["psolid"],"CTETRA missing PSOLID")
        mid = deck["psolid"][element["PID"]]["MID"]
        _require(mid in deck["materials"],"PSOLID missing MAT1"); material_ids.add(mid); element_ids[index] = eid
    _require((element_ids>=0).all() and len(material_ids) == 1,"One material must cover every cached cell",UnsupportedInput)
    material_id = next(iter(material_ids)); material = deck["materials"][material_id]
    E,G,nu = material["E"],material["G"],material["nu"]
    if E is None and G is not None and nu is not None:
        E = 2*G*(1+nu)
    if nu is None and E is not None and G is not None:
        _require(G != 0,"Zero explicit MAT1 G",UnsupportedInput); nu = E/(2*G)-1
    _require(E is not None and nu is not None and E>0 and -1<nu<.5,"Invalid/incomplete isotropic MAT1",UnsupportedInput)
    mu = E/(2*(1+nu)); lam = E*nu/((1+nu)*(1-2*nu))
    mismatch = 0. if G is None else abs(G-mu)/max(abs(G),abs(mu))
    _require(mismatch <= recipe["reader_scope"]["material_relative_tolerance"],"Explicit MAT1 G inconsistent with E/nu",UnsupportedInput)
    imposed = {}; direct_fixed = set()
    for node,dofs,value in deck["active_constraints"]:
        _require(node in deck["grid"] and value == 0.,"Undefined/nonzero active SPC unsupported",UnsupportedInput)
        for dof in dofs:
            key = (node,dof); _require(key not in imposed,"Repeated active constrained DOF",UnsupportedInput); imposed[key] = 0.
    refs = sorted(row["reference"] for row in deck["rbe2"].values())
    _require(len(refs) == 4 and len(set(refs)) == 4,"Exactly four distinct supported RBE2 references required",UnsupportedInput)
    refset = set(refs)
    _require(all(all((node,d) in imposed for d in "123456") for node in refs),"RBE2 master is not SPC123456=0",UnsupportedInput)
    for node,row in deck["grid"].items():
        for dof in row["PS"]:
            _require((node,dof) not in imposed,"SPC/PS overlapping constrained DOF",UnsupportedInput)
            imposed[(node,dof)] = 0.
    for node in {node for node,_ in imposed}-refset:
        dofs = {d for g,d in imposed if g == node}
        _require(dofs == set("123") and node in rows,"Only direct physical123 zero SPC/PS supported",UnsupportedInput)
        direct_fixed.add(rows[node])
    fixed = set(direct_fixed); regions = {}; dependent = set()
    for region,row in enumerate(sorted(deck["rbe2"].values(),key=lambda r:r["reference"])):
        _require(row["reference"] in deck["grid"] and row["reference"] not in row["members"],"RBE2 reference/member conflict",UnsupportedInput)
        members = row["members"]
        _require(set(members) <= rows.keys(),"RBE2 member missing canonical physical row",UnsupportedInput)
        _require(not (set(members)&refset) and not (set(members)&dependent),"Dependent RBE2 overlap/cycle",UnsupportedInput)
        _require(not any((g,d) in imposed for g in members for d in row["dofs"]),"RBE2 dependent DOF also constrained",UnsupportedInput)
        dependent.update(members); indices = np.asarray([rows[g] for g in members],np.int64)
        _require(used[indices].all(),"RBE2 physical member unused by all cells",UnsupportedInput)
        fixed.update(indices.tolist()); regions[f"region_support_{region}"] = np.sort(indices)
        if row["reference"] in rows:
            fixed.add(rows[row["reference"]])
    _require(len(deck["rbe3"]) == 1,"Exactly one loaded RBE3 required",UnsupportedInput)
    rbe = next(iter(deck["rbe3"].values())); reference = rbe["reference"]
    _require(reference in deck["grid"] and reference not in refset and reference not in rbe["members"]
        and not any((reference,d) in imposed for d in "123456"),"Loaded reference undefined/constrained/dependent conflict",UnsupportedInput)
    _require(set(rbe["members"]) <= rows.keys(),"RBE3 member missing canonical physical row",UnsupportedInput)
    member_ids = np.asarray([rows[g] for g in rbe["members"]],np.int64)
    _require(used[member_ids].all(),"RBE3 member unused by cells",UnsupportedInput)
    ref_xyz = np.asarray(deck["grid"][reference]["xyz"],np.float64); offsets = vertices[member_ids]-ref_xyz
    fit_length = float(np.linalg.norm(offsets,axis=1).mean()); _require(fit_length>0,"Zero RBE3 characteristic length",UnsupportedInput)
    r = offsets/fit_length; skew = np.zeros((len(r),3,3),np.float64)
    skew[:,0,1]=-r[:,2]; skew[:,0,2]=r[:,1]; skew[:,1,0]=r[:,2]
    skew[:,1,2]=-r[:,0]; skew[:,2,0]=-r[:,1]; skew[:,2,1]=r[:,0]
    B = np.concatenate((np.broadcast_to(np.eye(3),(len(r),3,3)),-skew),axis=2).reshape(-1,6)
    weights = np.full(3*len(r),rbe["weight"],np.float64); normal = B.T@(weights[:,None]*B)
    rank = int(np.linalg.matrix_rank(normal)); condition = float(np.linalg.cond(normal))
    _require(rank == 6 and math.isfinite(condition) and condition <= recipe["load_transfer"]["maximum_normal_condition"],"RBE3 weighted-LS normal matrix rank/condition fails",UnsupportedInput)
    H = np.linalg.solve(normal,B.T*weights[None,:]); reproduction = float(np.max(np.abs(H@B-np.eye(6))))
    reference_load = np.zeros(6,np.float64)
    for action in deck["active_actions"]:
        _require(action["node"] == reference,"All active loads must act at the loaded RBE3 reference",UnsupportedInput)
        offset = 0 if action["kind"] == "FORCE" else 3
        reference_load[offset:offset+3] += action["vector"]
    _require(np.isfinite(reference_load).all() and np.linalg.norm(reference_load)>0,"Zero/nonfinite ver load",UnsupportedInput)
    _require(all(d in rbe["REFC"] for d,x in zip("123456",reference_load) if x != 0),"Applied load on omitted RBE3 REFC",UnsupportedInput)
    scaled = np.concatenate((reference_load[:3],reference_load[3:]/fit_length)); transferred = (H.T@scaled).reshape(-1,3)
    resultant = transferred.sum(axis=0); moment = np.cross(offsets,transferred).sum(axis=0)
    force_error = float(np.max(np.abs(resultant-reference_load[:3]))/max(1.,np.linalg.norm(scaled)))
    moment_error = float(np.max(np.abs(moment-reference_load[3:]))/max(1.,fit_length*np.linalg.norm(scaled)))
    witness = np.random.default_rng(recipe["seed"]).standard_normal(3*len(r))
    work_error = float(abs(witness@(H.T@scaled)-(H@witness)@scaled)/max(1.,np.linalg.norm(witness)*np.linalg.norm(H.T@scaled)))
    tolerance = recipe["load_transfer"]["relative_guard_tolerance"]
    _require(max(reproduction,force_error,moment_error,work_error) <= tolerance,"RBE3 reproduction/virtual-work/force-moment guard failed",UnsupportedInput)
    force = np.zeros((n,3),np.float64); np.add.at(force,member_ids,transferred)
    derived = []
    if reference in rows:
        _require(not used[rows[reference]] and rows[reference] not in fixed,"Loaded reference inside domain/fixed unsupported",UnsupportedInput)
        _require(set("123") <= set(rbe["REFC"]),"Derived physical reference translation not defined by REFC",UnsupportedInput)
        derived = [rows[reference]]
    fixed_ids = np.asarray(sorted(fixed),np.int64); free_ids = np.flatnonzero(used & ~np.isin(np.arange(n),fixed_ids)).astype(np.int64)
    covered = np.zeros(n,np.int8); covered[free_ids]+=1; covered[fixed_ids]+=1; covered[derived]+=1
    _require((covered == 1).all(),f"Unused/unclassified physical nodes unsupported: {np.flatnonzero(covered!=1)[:8].tolist()}",UnsupportedInput)
    regions["region_load"] = np.sort(member_ids); regions["region_interior"] = np.flatnonzero(used & ~np.isin(np.arange(n),np.concatenate(list(regions.values())))).astype(np.int64)
    arrays = {"vertices_mm":vertices,"cells10":cells,"grid_node_ids":ids,"free_node_ids":free_ids,"fixed_node_ids":fixed_ids,
        "force":force,"lame":np.asarray([lam,mu],np.float64),"ctetra_element_ids":element_ids,
        "derived_node_ids":np.asarray(derived,np.int64),"derived_source_ids":member_ids if derived else np.empty(0,np.int64),
        "derived_rowptr":np.asarray([0,len(member_ids)] if derived else [0],np.int64),
        "derived_blocks":H[:3].reshape(3,len(member_ids),3).transpose(1,0,2).copy() if derived else np.empty((0,3,3),np.float64),
        "rbe3_member_node_ids":member_ids,"rbe3_Hbar":H,"rbe3_length_mm":np.asarray(fit_length),
        "reference_xyz_mm":ref_xyz,"reference_load6":reference_load,"characteristic_length_mm":np.asarray(length),**regions}
    facts = {"node_count":n,"cell_count":len(cells),"used_cell_nodes":int(used.sum()),"free_nodes":len(free_ids),"fixed_nodes":len(fixed_ids),"derived_nodes":len(derived),
        "allN_partition_exact":True,"deck_coordinate_max_error_over_length":coordinate_error,"ctetra_all_cells_material_edge_association":True,
        "material":{"MID":material_id,"E_MPa":E,"nu":nu,"G_MPa":mu,"explicit_G_relative_error":mismatch,"lambda_MPa":lam,"mu_MPa":mu},
        "constraints":{"support_reference_GRID_IDs":refs,"RBE2_count":4,"actual_card_CM":[r["dofs"] for r in sorted(deck["rbe2"].values(),key=lambda r:r["reference"])],
            "active_SPC_DOF_count":len(imposed),"direct_fixed_physical_nodes":len(direct_fixed),"derived_output_only":True,"role_flags_used":False},
        "load_guard":{"reference_GRID_ID":reference,"member_count":len(member_ids),"REFC":rbe["REFC"],"weight":rbe["weight"],"length_mm":fit_length,
            "normal_rank":rank,"normal_condition":condition,"HB_max_error":reproduction,"virtual_work_relative_error":work_error,
            "force_relative_error":force_error,"moment_relative_error":moment_error,"reference_load6":reference_load.tolist(),"passed":True},
        "cards":deck["cards"],"selected_controls":deck["selected_controls"],"ignored_controls":deck["ignored_controls"],"GRDSET":deck["GRDSET"],
        "PSOLID_original_controls_not_replicated":deck["psolid"],"K_applications":0,"labels_read":0,"stress_reads":0,"GPU_calls":0}
    return arrays,facts


def prepare_data(plan_path=DEFAULT_PLAN):
    """Actual single input job. No stiffness/geometry-GPU guard or label access."""
    began = time.monotonic(); contract,plan = request_contract(plan_path)
    _require(plan.get("scientific_rules_frozen") is True and plan.get("actual_execution_released") is True,"Root must release the frozen input stage")
    cache = _path(plan["data_output"]); cache.mkdir(parents=True,exist_ok=True)
    if (cache/"manifest.json").exists():
        existing = _json(cache/"manifest.json")
        _require(existing.get("contract") == contract,"Existing input receipt belongs to another contract")
        if existing.get("status") == "completed":
            if existing.get("feasibility_pass"):
                return read_cache_metadata(plan_path,contract=contract)
            return existing
    config = {"contract":contract}; _write(cache/"config.json",config)
    budget = plan["recipe"]["input_operator"]["maximum_combined_data_operator_wall_seconds"]
    manifest = {"kind":"elastic_pilot_input","status":"running","feasibility_pass":False,"schema":SCHEMA,"contract":contract,"config":config,
        "labels_read":0,"stress_reads":0,"K_applications":0,"GPU_calls":0,"training_updates":0,"scaler_fits":0,
        "read_receipt":{"accessed_members":{},"raw_deck_open_attempted":False,"labels_read":0,"stress_reads":0,"H5_other_datasets_read":0,"K_applications":0,"GPU_calls":0}}
    _write(cache/"status.json",{"status":"running","phase":"input_sources","elapsed_seconds":time.monotonic()-began})
    try:
        decode_began = time.monotonic(); vertices,cells,ids = _inputs(contract,manifest["read_receipt"])
        source_array_seconds = time.monotonic()-decode_began; _deadline(began,budget)
        manifest["read_receipt"].update(raw_deck_open_attempted=True,raw_deck=PATHS["deck"])
        parse_began = time.monotonic(); deck = _parse_deck(PATHS["deck"],plan["recipe"],began,budget)
        deck_parse_seconds = time.monotonic()-parse_began; _deadline(began,budget)
        build_began = time.monotonic(); arrays,facts = _build(deck,vertices,cells,ids,contract,began,budget)
        build_seconds = time.monotonic()-build_began; _deadline(began,budget)
        import numpy as np
        output = cache/"input.npz"; temporary = cache/"input.npz.tmp"
        save_began = time.monotonic()
        with temporary.open("wb") as handle:
            np.savez(handle,**arrays)
        temporary.replace(output); identity = _identity(output,True); save_seconds = time.monotonic()-save_began
        _deadline(began,budget)
        manifest.update(status="completed",feasibility_pass=True,cache_file=output.relative_to(ROOT).as_posix(),file_identity=identity,
            preparation_seconds=time.monotonic()-began,elapsed_seconds=time.monotonic()-began,
            costs={"source_array_seconds":source_array_seconds,"deck_parse_seconds":deck_parse_seconds,"input_algebra_seconds":build_seconds,"save_seconds":save_seconds},
            facts=facts,node_count=len(vertices),cell_count=len(cells),region_array_keys=[k for k in arrays if k.startswith("region_")],
            array_schema={k:{"shape":list(v.shape),"dtype":str(v.dtype),"sha256":_array_sha(v,v.dtype)} for k,v in arrays.items()},
            deferred_displacement_source=contract["deferred_displacement_source"],deferred_displacement_member="displacement",
            total_cache_bytes=identity["bytes"],budget_seconds=budget)
        manifest["preparation_seconds"] = manifest["elapsed_seconds"] = time.monotonic()-began
        _deadline(began,budget)
    except UnsupportedInput as error:
        manifest.update(status="completed",feasibility_pass=False,scientific_outcome="unsupported_input_or_budget",
            failure={"type":type(error).__name__,"message":str(error)},preparation_seconds=time.monotonic()-began,elapsed_seconds=time.monotonic()-began)
    except Exception as error:
        manifest.update(status="failed",feasibility_pass=False,failure={"type":type(error).__name__,"message":str(error)},
            preparation_seconds=time.monotonic()-began,elapsed_seconds=time.monotonic()-began)
        _write(cache/"manifest.json",manifest); _write(cache/"summary.json",manifest); _write(cache/"status.json",manifest)
        raise
    _write(cache/"manifest.json",manifest); _write(cache/"summary.json",manifest)
    _write(cache/"status.json",{"status":manifest["status"],"feasibility_pass":manifest["feasibility_pass"],"elapsed_seconds":manifest["elapsed_seconds"]})
    return manifest


def load_input(manifest, device):
    """Load target-free cache only after metadata negative rejection."""
    _require(manifest.get("status") == "completed" and manifest.get("feasibility_pass") is True,"Input negative/incomplete; no CUDA initialization")
    _match(manifest["cache_file"],manifest["file_identity"])
    _require(_sha(manifest["cache_file"]) == manifest["file_identity"]["sha256"],"Input cache byte SHA changed")
    import numpy as np
    import torch
    result = {}; regions = {}
    with np.load(_path(manifest["cache_file"]),allow_pickle=False) as source:
        _require(set(source.files) == set(manifest["array_schema"]),"Input cache member schema changed")
        for key,schema in manifest["array_schema"].items():
            value = source[key]
            _require(list(value.shape) == schema["shape"] and str(value.dtype) == schema["dtype"]
                and _array_sha(value,value.dtype) == schema["sha256"],f"Input cache array identity differs: {key}")
            tensor = torch.from_numpy(value.copy()).to(device=device)
            if key in manifest["region_array_keys"]:
                regions[key.removeprefix("region_")] = tensor
            else:
                result[key] = tensor
    result["regions"] = regions
    result["manifest"] = manifest
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",default=DEFAULT_PLAN)
    args = parser.parse_args()
    result = prepare_data(args.plan)
    print(json.dumps({"status":result["status"],"feasibility_pass":result["feasibility_pass"],"preparation_seconds":result["preparation_seconds"]}),flush=True)


if __name__ == "__main__":
    main()
