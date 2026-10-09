#!/usr/bin/env python3
"""Guard the physical load basis of local train/validation DeepJEB CADs.

Reads SolverDeck metadata and ONLY canonical ver features/sample_key from
prepared NPZs. No field labels, predictions, dia NPZs, or final-test NPZs are
accessed. Incompatible decks are recorded and cause a nonzero exit status.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
if os.name == "nt" and (ROOT / ".local_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".local_deps"))
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.parse_deepjeb_deck import parse_deck

LOADS = ("ver", "hor", "dia", "tor")
BASIS_LOADS = ("ver", "hor", "tor")
LABELS = {"Vertical": "ver", "Horizontal": "hor", "Diagonal": "dia", "Torsion": "tor"}
PHYSICAL_SCALE = np.asarray([42300.0] * 3 + [565000.0] * 3)


def exact_load(actions: list[dict]) -> list[Fraction]:
    result = [Fraction(0)] * 6
    for action in actions:
        start = 0 if action["kind"] == "force" else 3
        for i, direction in enumerate(action["direction"]):
            result[start + i] += Fraction(str(action["magnitude"])) * Fraction(str(direction))
    return result


def solve_exact(basis: list[list[Fraction]], target: list[Fraction]) -> tuple[int, list[Fraction] | None]:
    """Exact RREF of six physical equations and three dimensionless unknowns."""
    rows = [[basis[column][row] for column in range(3)] + [target[row]] for row in range(6)]
    pivot_rows = []
    next_row = 0
    for column in range(3):
        pivot = next((i for i in range(next_row, 6) if rows[i][column] != 0), None)
        if pivot is None:
            continue
        rows[next_row], rows[pivot] = rows[pivot], rows[next_row]
        divisor = rows[next_row][column]
        rows[next_row] = [x / divisor for x in rows[next_row]]
        for i in range(6):
            if i != next_row:
                factor = rows[i][column]
                rows[i] = [x - factor*y for x, y in zip(rows[i], rows[next_row])]
        pivot_rows.append((column, next_row))
        next_row += 1
    inconsistent = any(all(v == 0 for v in row[:3]) and row[3] != 0 for row in rows)
    if inconsistent or len(pivot_rows) != 3:
        return len(pivot_rows), None
    answer = [Fraction(0)] * 3
    for column, row in pivot_rows:
        answer[column] = rows[row][3]
    return 3, answer


def guard_one(task: tuple) -> dict:
    cad, group, raw_s, prepared_s = task
    started = time.monotonic()
    deck_path = Path(raw_s) / "SolverDeck" / f"{cad}.fem"
    record = {"cad_id": cad, "group": group, "deck": str(deck_path), "canonical_prepared_case": f"{cad}/ver", "reasons": []}
    try:
        deck = parse_deck(deck_path)
        static = {LABELS[x["label"]]: x for x in deck["summary"]["subcases"].values() if x.get("label") in LABELS}
        if set(static) != set(LOADS) or set(deck["cases"]) != set(LOADS):
            raise ValueError("Missing one of the four recognized static subcases")
        constraints = {key: sorted((x["node_id"], x["dofs"], x["value"]) for x in item.get("constraints", [])) for key, item in static.items()}
        same_constraints = all(x == constraints["ver"] for x in constraints.values())
        homogeneous_fixed = (len(constraints["ver"]) == 4 and all(dofs == "123456" and value == 0 for _, dofs, value in constraints["ver"]))
        same_interface = all(item.get("actions") and all(a["node_id"] == deck["loaded_group"]["master_node_id"] for a in item["actions"]) for item in static.values())
        if not same_constraints:
            record["reasons"].append("Constraints differ across static subcases")
        if not homogeneous_fixed:
            record["reasons"].append("Expected four homogeneous zero-DOF123456 support references")
        if not same_interface:
            record["reasons"].append("Static loads use a different or missing loaded reference interface")
        if any(x.get("analysis") != "STATICS" for x in static.values()):
            record["reasons"].append("A load subcase is not ANALYSIS=STATICS")
        cards = Counter()
        nonlinear_or_unsupported = []
        evidence = []
        solid_materials = set()
        with deck_path.open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                card = line[:8].strip()
                upper = line.upper()
                if re.fullmatch("[A-Z][A-Z0-9]*", card):
                    cards[card] += 1
                unsupported = (card in {"NLPARM", "NLPCI", "CONTACT", "PCONT", "CGAP", "CGAPG", "GRAV", "RFORCE", "DEFORM", "TEMP", "TEMPD"}
                               or card.startswith("MAT") and card != "MAT1"
                               or card.startswith("PLOAD")
                               or card.startswith("FORCE") and card != "FORCE"
                               or card.startswith("MOMENT") and card != "MOMENT"
                               or any(token in upper for token in ("LGDISP", "NLSTAT", "STATSUB", "CNTNLSUB"))
                               or re.match(r"\s*(MATERIAL|MATSEL)\s*=", upper))
                if unsupported:
                    nonlinear_or_unsupported.append({"line": number, "text": line.rstrip()})
                if card in {"FORCE", "MOMENT", "SPC", "SPCADD", "LOAD", "MAT1"}:
                    evidence.append({"line": number, "text": line.rstrip()})
                if card == "PSOLID":
                    solid_materials.add(int(line[16:24].strip()))
                if card == "MAT1" and "SELECT" in upper:
                    nonlinear_or_unsupported.append({"line": number, "text": line.rstrip()})
        if nonlinear_or_unsupported:
            record["reasons"].append("Nonlinear, contact, prestress, varying-material or unsupported-load metadata found")
        material = deck["material"]
        common_material = (cards["MAT1"] == 1 and solid_materials == {material["material_id"]}
                           and material["E"] > 0 and -1 < material["nu"] < .5 and material["rho"] > 0)
        if not common_material:
            record["reasons"].append("A single positive linear MAT1 is not shared by solid properties")
        load_vectors = {key: exact_load(static[key]["actions"]) for key in LOADS}
        with np.load(prepared_s, allow_pickle=False) as stored:
            features = stored["features"]
            sample_key = str(stored["sample_key"])
        if sample_key != f"{cad}/ver" or features.ndim != 2 or features.shape[1] != 25:
            raise ValueError("Canonical prepared sample_key/features[N,25] mismatch")
        expected = np.asarray([float(x) for x in load_vectors["ver"]]) / PHYSICAL_SCALE
        normalized_error = float(np.max(np.abs(features[:, 7:13].astype(np.float64) - expected)))
        feature_compatible = np.isfinite(features[:, 7:13]).all() and normalized_error <= 1e-7
        if not feature_compatible:
            record["reasons"].append("Canonical ver prepared force/moment slots disagree with real deck")
        record.update({"subcase_ids_and_controls": {key: {k: item[k] for k in ("label", "analysis", "spc", "load")} for key, item in static.items()},
                       "loads": {key: {"physical_vector": [float(x) for x in vector], "exact_rational_vector": [str(x) for x in vector]} for key, vector in load_vectors.items()},
                       "constraints_identical": same_constraints, "zero_fixed_constraints": homogeneous_fixed,
                       "SPC_constraints": static["ver"]["constraints"], "material_identical_within_CAD": common_material,
                       "material": material, "load_interface_identical": same_interface,
                       "loaded_reference_node_id": deck["loaded_group"]["master_node_id"],
                       "loaded_interface_member_count": deck["loaded_group"]["num_nodes"],
                       "support_reference_node_ids": [g["master_node_id"] for g in deck["fixed_groups"]],
                       "STATICS_without_nonlinear_entries": all(x["analysis"] == "STATICS" for x in static.values()) and not nonlinear_or_unsupported,
                       "nonlinear_or_unsupported_entries": nonlinear_or_unsupported,
                       "prepared_slot_mapping_compatible": bool(feature_compatible), "prepared_load_slots_max_normalized_error": normalized_error,
                       "prepared_nodes": len(features), "source_card_evidence": evidence,
                       "target_arrays_accessed": False, "elapsed_seconds": time.monotonic() - started})
    except Exception as exc:
        record["reasons"].append(f"Metadata parse/check failed: {exc}")
        record["elapsed_seconds"] = time.monotonic() - started
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / "notes/geobc_split.json")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "notes/geobc_load_basis_45.json")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    groups = {key: split["groups"][key] for key in ("train", "validation")}
    if len(groups["train"]) != 40 or len(groups["validation"]) != 5 or set(groups["train"]) & set(groups["validation"]):
        raise ValueError("Expected the fixed, disjoint 40/5 training/validation CAD split")
    tasks = [(cad, group, str(args.raw), str(ROOT / split["case_paths"][f"{cad}/ver"])) for group, cads in groups.items() for cad in cads]
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(guard_one, tasks))
    reference = next(x for x in records if x["cad_id"] == "101_479")
    if "loads" not in reference:
        for record in records:
            record["compatible"] = False
            record["reasons"].append("Shared-basis compatibility unavailable because reference train CAD metadata failed")
        failed = {"schema_version": 1, "status": "incompatible", "compatible": False, "compatible_all": False,
                  "CADs_checked": len(records), "compatible_CADs": 0, "groups": groups, "cases": records,
                  "badcases": [{"cad_id": x["cad_id"], "reasons": x["reasons"]} for x in records],
                  "elapsed_seconds": time.monotonic()-start, "error": "Canonical reference metadata could not be parsed"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(failed, indent=2)+"\n", encoding="utf-8")
        os.replace(temporary, args.output)
        print(json.dumps({"status": "incompatible", "error": failed["error"]}))
        raise SystemExit(2)
    basis = [[Fraction(x) for x in reference["loads"][key]["exact_rational_vector"]] for key in BASIS_LOADS]
    common_rank, _ = solve_exact(basis, [Fraction(0)] * 6)
    for record in records:
        if "loads" not in record:
            record["compatible"] = False
            continue
        local = {key: [Fraction(x) for x in record["loads"][key]["exact_rational_vector"]] for key in LOADS}
        rank, _ = solve_exact([local[key] for key in BASIS_LOADS], [Fraction(0)] * 6)
        same_basis = all(local[key] == basis[i] for i, key in enumerate(BASIS_LOADS))
        if rank != 3:
            record["reasons"].append("Actual local ver/hor/tor load basis is rank-deficient")
        if not same_basis:
            record["reasons"].append("Local normative basis differs from shared reference basis")
        record.update({"basis_rank": rank, "basis_matches_shared_reference": same_basis, "alpha_by_load": {}})
        for key in LOADS:
            _, alpha = solve_exact(basis, local[key])
            if alpha is None:
                record["reasons"].append(f"{key} is outside the common physical load span")
                continue
            residual = [sum(alpha[j] * basis[j][i] for j in range(3)) - local[key][i] for i in range(6)]
            record["alpha_by_load"][key] = {"alpha": [float(x) for x in alpha], "exact_rational_alpha": [str(x) for x in alpha],
                                                 "force_residual_N": [float(x) for x in residual[:3]],
                                                 "moment_residual_Nmm": [float(x) for x in residual[3:]], "exact_residual_zero": all(x == 0 for x in residual)}
        record["compatible"] = not record["reasons"] and common_rank == 3 and len(record["alpha_by_load"]) == 4
    bad = [{"cad_id": x["cad_id"], "reasons": x["reasons"]} for x in records if not x["compatible"]]
    physical_basis = np.asarray([[float(x) for x in column] for column in basis]).T
    normalized_basis = physical_basis / PHYSICAL_SCALE[:, None]
    feature_to_alpha = np.linalg.pinv(normalized_basis)
    feature_to_alpha[np.abs(feature_to_alpha) < 1e-14] = 0
    report = {"schema_version": 1, "status": "compatible" if not bad else "incompatible", "compatible": not bad, "compatible_all": not bad,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_split": str(args.split), "groups": groups,
              "CADs_checked": len(records), "compatible_CADs": len(records)-len(bad), "badcases": bad,
              "elapsed_seconds": time.monotonic()-start, "workers": args.workers,
              "scope": "45 train/validation CADs only; real four-subcase SolverDeck metadata, canonical ver prepared features/sample_key only; no field-label/prediction/test/dia-NPZ access",
              "canonical_basis": {"reference_CAD": "101_479", "order": list(BASIS_LOADS), "physical_vector_order": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
                                  "component_units": ["N"]*3 + ["N mm"]*3, "physical_matrix_6x3": physical_basis.tolist(), "exact_rank": common_rank},
              "prepared_feature_mapping": {"slots": [7, 8, 9, 10, 11, 12], "physical_scale": PHYSICAL_SCALE.tolist(),
                                            "alpha_order": list(BASIS_LOADS), "normalized_basis_6x3": normalized_basis.tolist(),
                                            "feature_to_alpha_matrix_3x6": feature_to_alpha.tolist(),
                                            "formula": "alpha[N,3] = features[N,7:13] @ feature_to_alpha_matrix_3x6.T; actual physical loads = features[:,7:13] * physical_scale",
                                            "tolerance_normalized": 1e-7},
              "cases": records,
              "interpretation": "For unchanged linear small-displacement stiffness and homogeneous constraints, displacement has a ver/hor/tor response-basis expectation. This metadata guard does not numerically validate held-out dia fields.",
              "limitations": ["Do not linearly combine source signed scalar stress; keep its learned branch/targets independent.",
                              "STATICS may include gap analysis in the solver; this guard rejects gap/contact/nonlinear/prestress/unsupported loading entries rather than silently accepting them.",
                              "No archived solver execution flags/logs or physical small-deformation adequacy were audited.",
                              "Dia is within the load linear span, not a new independent mode or necessarily a convex combination.",
                              "This does not provide independent Fy/Mx/My responses or arbitrary six-component loading generalization.",
                              "No target scaler is fitted and held-out dia/test fields remain inaccessible."],
              "primary_semantics": [{"title": "Altair OptiStruct ANALYSIS", "url": "https://help.altair.com/hwsolvers/os/topics/solvers/os/analysis_sub_r.htm"},
                                    {"title": "Altair OptiStruct MAT1", "url": "https://help.altair.com/hwsolvers/os/topics/solvers/os/mat1_bulk_r.htm"},
                                    {"title": "Altair OptiStruct PARAM LGDISP", "url": "https://help.altair.com/hwsolvers/os/topics/solvers/os/param_lgdisp_bulk_r.htm"}]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({key: report[key] for key in ("status", "CADs_checked", "compatible_CADs", "badcases", "elapsed_seconds")}))
    print(json.dumps({"canonical_basis": report["canonical_basis"], "prepared_feature_mapping": report["prepared_feature_mapping"],
                      "reference_alpha_by_load": reference.get("alpha_by_load")}))
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
