#!/usr/bin/env python3
"""Prepare local original-DeepJEB load transfer cases and a CAD-disjoint split.

Only CPU preprocessing. Existing geometry, node IDs, boundary faces, and the
explicit 25-feature schema are reused. Labels are read from official CSV and
aligned by GRID ID, with coordinates checked against the preserved H5 order.
No label scaler is fitted here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if os.name == "nt" and (ROOT / ".local_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".local_deps"))
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
from tools.parse_deepjeb_deck import parse_deck

LOADS = ("ver", "hor", "dia", "tor")
TRANSFER_CADS = ("101_479", "107_320", "114_202")
EXPECTED_TEST = {"107_95", "113_345", "113_472", "114_319", "116_61"}
ALIGNMENT = "official CSV GRID coordinates -> H5 vertices"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def file_name(value: str) -> str:
    """Accept manifests saved on either Windows or POSIX hosts."""
    return Path(value.replace("\\", "/")).name


def make_split(raw: Path, source: dict, path: Path, seed: int) -> dict:
    cads = sorted(key.split("/")[0] for key in source["selection"])
    if len(cads) != 50 or len(set(cads)) != 50:
        raise ValueError("Expected the preserved 50 unique CADs, one source condition each")
    official_train = set(json.loads((raw / "Metadata/train_split_random.json").read_text()))
    official_test = set(json.loads((raw / "Metadata/test_split_random.json").read_text()))
    test = sorted(set(cads) & official_test)
    eligible = sorted(set(cads) & official_train)
    if set(test) != EXPECTED_TEST or len(eligible) != 45 or set(test) & set(eligible):
        raise ValueError("Local CADs do not match the expected official random train/test intersections")
    validation = sorted(random.Random(seed).sample(eligible, 5))
    train = sorted(set(eligible) - set(validation))
    groups = {"train": train, "validation": validation, "test": test}
    nodes = {x["sample_key"].split("/")[0]: x["nodes"] for x in source["cases"]}
    report = {
        "dataset": source["dataset"], "source_revision": source["source_revision"],
        "seed": seed, "rng": "Python random.Random(seed).sample(sorted(official_train_intersection), 5)",
        "official_split": "data/raw/Metadata/{train,test}_split_random.json",
        "source_manifest": "data/prepared50_multifield/manifest.json",
        "prepared_manifest": "data/prepared_geobc_multiload/manifest.json",
        "grouping": "Whole CAD, including all nodes, faces, and all four loads; no CAD crosses groups",
        "geometry_population_limit": "Official archive-prefix replacement 50; design-family lineage is not available in this manifest",
        "loads": list(LOADS), "splits": {}, "groups": groups,
        "case_paths": {f"{cad}/{load}": f"data/prepared_geobc_multiload/{cad}_{load}.npz" for cad in cads for load in LOADS},
        "train_loads": ["ver", "hor", "tor"], "heldout_load": "dia",
        "checkpoint_validation_loads": ["ver", "hor", "tor"],
        "optimization_case_keys": [f"{cad}/{load}" for cad in train for load in ("ver", "hor", "tor")],
        "checkpoint_validation_case_keys": [f"{cad}/{load}" for cad in validation for load in ("ver", "hor", "tor")],
        "heldout_direction_policy": "No dia labels in optimization, target-scaler fitting, or checkpoint/hyperparameter selection. Final-test CADs remain unevaluated until the final protocol is fixed.",
        "scaler_policy": "Fit all target transforms/scalers exclusively on optimization_case_keys: train CADs x train_loads. Never fit on all200, validation, test, or held-out dia labels.",
        "evaluation_policy": "Validation selects checkpoints/hyperparameters. Final test remains held out. New models must not initialize from old all50-fit checkpoints.",
        "checks": {"pairwise_CAD_disjoint": True, "CAD_union_matches_source50": True,
                   "test_matches_official_random_intersection": True, "all_loads_in_same_CAD_group": True},
    }
    for name, cad_list in groups.items():
        report["splits"][name] = {"cad_ids": cad_list, "case_keys": [f"{cad}/{load}" for cad in cad_list for load in LOADS],
                                  "CADs": len(cad_list), "cases": 4 * len(cad_list),
                                  "nodes_per_load_sum": sum(nodes[cad] for cad in cad_list),
                                  "nodes_all_four_loads": 4 * sum(nodes[cad] for cad in cad_list)}
    all_sets = [set(x) for x in groups.values()]
    if set.union(*all_sets) != set(cads) or any(all_sets[i] & all_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("CAD split disjointness failed")
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        immutable = ("dataset", "source_revision", "seed", "rng", "official_split", "source_manifest", "splits")
        if any(existing.get(k) != report[k] for k in immutable):
            raise FileExistsError(f"Refusing to change an existing different split: {path}")
        if existing != report:
            write_json(path, report)
    else:
        write_json(path, report)
    return report


def copy_exact(source: Path, target: Path) -> str:
    if target.exists():
        if os.path.samefile(source, target):
            return "existing byte-identical hardlink"
        raise FileExistsError(f"Refusing to replace existing uncached file {target}")
    try:
        os.link(source, target)
        return "byte-identical hardlink, not a symlink"
    except OSError:
        with target.open("xb") as dst, source.open("rb") as src:
            shutil.copyfileobj(src, dst, 2**20)
        return "byte-identical file copy"


def build_cad(task: tuple) -> dict:
    raw_s, output_s, base_s, base_info, loads, reusable = task
    raw, output, base = Path(raw_s), Path(output_s), Path(base_s)
    cad, old_load = base_info["sample_key"].split("/")
    cache = output / ".cad_manifests" / f"{cad}.json"
    fingerprint = {"source_h5_sha256": base_info["source_h5_sha256"],
                   "source_csv_sha256": base_info["source_csv_sha256"],
                   "source_deck_sha256": base_info["source_deck_sha256"], "loads": list(loads), "alignment": ALIGNMENT}
    if cache.is_file():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint and all((output / file_name(x["npz"])).is_file() for x in cached["cases"]):
            return cached
        raise ValueError(f"{cad}: existing cache is incompatible or incomplete")
    start = time.monotonic()
    with np.load(base, allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}
    if str(data["sample_key"]) != base_info["sample_key"] or str(data["label_alignment"]) != ALIGNMENT:
        raise ValueError(f"{cad}: base inputs are not correctly aligned")
    n = len(data["xyz_mm"])
    if data["features"].shape != (n, 25) or len(data["grid_node_ids"]) != n:
        raise ValueError(f"{cad}: invalid preserved input dimensions")
    deck = parse_deck(raw / "SolverDeck" / f"{cad}.fem")
    static = [x for x in deck["summary"]["subcases"].values() if x.get("label") in ("Vertical", "Horizontal", "Diagonal", "Torsion")]
    constraints = [sorted((x["node_id"], x["dofs"], x["value"]) for x in item["constraints"]) for item in static]
    if len(static) != 4 or any(x != constraints[0] for x in constraints) or set(deck["cases"]) != set(LOADS):
        raise ValueError(f"{cad}: boundary conditions differ across loads; reusing static features would be invalid")
    if any(action["node_id"] != deck["loaded_group"]["master_node_id"] for item in static for action in item.get("actions", [])):
        raise ValueError(f"{cad}: varying loaded reference interface is unsupported")
    old_condition = deck["cases"][old_load]
    expected = np.asarray(old_condition["force"] + old_condition["moment"], dtype=np.float32) / np.asarray([42300] * 3 + [565000] * 3, dtype=np.float32)
    if not np.allclose(data["features"][:, 7:13], expected, rtol=0, atol=1e-7):
        raise ValueError(f"{cad}: preserved load features disagree with the deck")
    material = deck["material"]
    expected_material = [material["E"] / 113800, material["nu"], material["rho"] / 4.47e-9]
    if not np.allclose(data["features"][:, 13:16], expected_material, rtol=0, atol=1e-7):
        raise ValueError(f"{cad}: preserved material features disagree with the deck")
    h5_missing = {}
    with h5py.File(raw / "FieldMesh" / f"{cad}.h5", "r") as h5:
        if h5["vertices"].shape != (n, 3):
            raise ValueError(f"{cad}: source geometry shape mismatch")
        volume_tets = h5["cells"].shape[0]
        for load in loads:
            if h5[f"nodal_variables/{load}_stress(MPa)"].shape != (n,):
                raise ValueError(f"{cad}/{load}: missing or invalid source H5 stress shape")
            h5_missing[load] = [f"nodal_variables/{load}_{axis}_disp(mm)" for axis in "xyz" if f"nodal_variables/{load}_{axis}_disp(mm)" not in h5]
    to_build = [load for load in loads if load != old_load and f"{cad}/{load}" not in reusable]
    max_coord_error = float(base_info["csv_to_prepared_max_coordinate_error_mm"])
    csv_cols = None
    if to_build:
        csv_path = raw / "Field" / f"{cad}.csv"
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f))
        names = ["nodeID"] + [f"coord_{axis}(mm)" for axis in "xyz"]
        names += [f"{load}_{axis}_disp(mm)" for load in to_build for axis in "xyz"]
        names += [f"{load}_stress(MPa)" for load in to_build]
        csv_cols = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=[header.index(k) for k in names], dtype=np.float64)
        ids = csv_cols[:, 0].astype(np.int64)
        if len(ids) != n or len(np.unique(ids)) != n:
            raise ValueError(f"{cad}: source CSV GRID IDs are not unique or counts differ")
        order = np.argsort(ids)
        positions = np.searchsorted(ids[order], data["grid_node_ids"])
        if np.any(positions >= n):
            raise ValueError(f"{cad}: prepared GRID ID missing from source CSV")
        rows = order[positions]
        if not np.array_equal(ids[rows], data["grid_node_ids"]):
            raise ValueError(f"{cad}: prepared GRID ID missing from source CSV")
        max_coord_error = float(np.linalg.norm(csv_cols[rows, 1:4] - data["xyz_mm"].astype(np.float64), axis=1).max())
        if max_coord_error > 2e-4:
            raise ValueError(f"{cad}: GRID-to-H5 coordinate error exceeds tolerance: {max_coord_error}")
    records = []
    for load in loads:
        key = f"{cad}/{load}"
        target = output / f"{cad}_{load}.npz"
        if load == old_load or key in reusable:
            origin = base_info if load == old_load else reusable[key]
            source = base if load == old_load else Path(origin["npz"])
            mode = copy_exact(source, target)
            info = {**origin, "npz": str(target.resolve()), "reused_prepared_file": str(source.resolve()), "file_reuse": mode}
        else:
            i = to_build.index(load)
            u = csv_cols[rows, 4 + 3*i:7 + 3*i].astype(np.float32)
            stress = csv_cols[rows, 4 + 3*len(to_build) + i].astype(np.float32)
            if not np.isfinite(u).all() or not np.isfinite(stress).all():
                raise ValueError(f"{key}: nonfinite raw target")
            features = data["features"].copy()
            condition = deck["cases"][load]
            features[:, 7:10] = np.asarray(condition["force"]) / 42300
            features[:, 10:13] = np.asarray(condition["moment"]) / 565000
            result = {**data, "features": features, "u": u, "displacement": u, "stress": stress,
                      "sample_key": np.asarray(key), "label_alignment": np.asarray(ALIGNMENT)}
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite uncached {target}")
            fd, temporary_s = tempfile.mkstemp(prefix=target.stem + ".", suffix=".npz", dir=output)
            os.close(fd)
            temporary = Path(temporary_s)
            try:
                np.savez(temporary, **result)
                os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            info = {**base_info, "sample_key": key, "npz": str(target.resolve()),
                    "force_N": condition["force"], "moment_Nmm": condition["moment"],
                    "signed_stress_min_MPa": float(stress.min()), "signed_stress_max_MPa": float(stress.max()),
                    "signed_stress_RMS_MPa": float(np.sqrt(np.mean(stress.astype(np.float64)**2))),
                    "u_component_source_columns": [f"{load}_{axis}_disp(mm)" for axis in "xyz"],
                    "u_shape": [n, 3], "u_units": "mm", "u_min_mm": u.min(0).tolist(), "u_max_mm": u.max(0).tolist(),
                    "u_RMS_mm": np.sqrt(np.mean(u.astype(np.float64)**2, axis=0)).tolist(),
                    "csv_to_prepared_max_coordinate_error_mm": max_coord_error,
                    "missing_source_h5_displacement_fields": h5_missing[load],
                    "preserved_geometry_input": str(base.resolve()),
                    "geometry_input_builder": "Preserved aligned H5 geometry/boundary/GRID IDs; deck-derived load slots 7:13 replaced; all other 25-feature slots unchanged",
                    "label_alignment": ALIGNMENT}
            for stale in ("old_prepared_wrong_alignment_stress_max_difference_MPa", "h5_raw_field_vs_csv_GRID_row_stress_max_difference_MPa", "csv_vs_available_h5_displacement_max_difference_mm"):
                info.pop(stale, None)
        info.update({"cad_id": cad, "load": load, "volume_tetrahedra": volume_tets,
                     "all_four_loads_have_identical_constraints": True,
                     "all_four_loads_use_same_loaded_interface": True,
                     "source_field_alignment_check": "CSV nodeID permutation -> preserved grid_node_ids, coordinates checked against xyz_mm"})
        records.append(info)
    report = {"cad_id": cad, "fingerprint": fingerprint, "cases": records,
              "elapsed_seconds": time.monotonic() - start, "csv_coordinate_max_error_mm": max_coord_error}
    write_json(cache, report)
    return report


def prepare_stage(raw: Path, source_dir: Path, output: Path, source: dict, cad_loads: dict,
                  workers: int, reusable: dict, purpose: str, progress) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    info = {x["sample_key"].split("/")[0]: x for x in source["cases"]}
    tasks = [(str(raw), str(output), str(source_dir / file_name(info[cad]["npz"])), info[cad], loads, reusable) for cad, loads in cad_loads.items()]
    start = time.monotonic()
    reports = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_cad, task): task[3]["sample_key"].split("/")[0] for task in tasks}
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            elapsed = time.monotonic() - start
            progress({"completed_CADs": len(reports), "total_CADs": len(tasks), "elapsed_seconds": elapsed,
                      "estimated_remaining_seconds": elapsed * (len(tasks) - len(reports)) / len(reports), "last_CAD": report["cad_id"]})
    by_key = {case["sample_key"]: case for report in reports for case in report["cases"]}
    selection = [f"{cad}/{load}" for cad, loads in cad_loads.items() for load in loads]
    manifest = {key: value for key, value in source.items() if key not in ("cases", "selection", "selection_policy", "nodes_total", "load_counts")}
    manifest.update({"selection": selection, "cases": [by_key[k] for k in selection], "selection_policy": purpose,
                     "nodes_total": sum(by_key[k]["nodes"] for k in selection),
                     "load_counts": {load: sum(k.endswith("/" + load) for k in selection) for load in LOADS},
                     "source_manifest": "data/prepared50_multifield/manifest.json", "scaler_fitted": False,
                     "preservation": "Preserved CSV-aligned geometry, boundary faces, GRID IDs, and static features; selected existing NPZs reused byte-identically; new loads change only feature slots 7:13 and raw signed targets.",
                     "CAD_input_policy": "CAD ID is metadata/grouping only, never an input feature",
                     "prep_elapsed_seconds": time.monotonic() - start})
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--source", type=Path, default=ROOT / "data/prepared50_multifield")
    parser.add_argument("--transfer-output", type=Path, default=ROOT / "data/prepared_geobc_transfer")
    parser.add_argument("--all-output", type=Path, default=ROOT / "data/prepared_geobc_multiload")
    parser.add_argument("--split", type=Path, default=ROOT / "notes/geobc_split.json")
    parser.add_argument("--plan", type=Path, default=ROOT / "notes/geobc_data_plan.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--stage", choices=("plan", "transfer", "all", "both"), default="both")
    args = parser.parse_args()
    source = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    for entry in source["cases"]:
        cad = entry["sample_key"].split("/")[0]
        for component, suffix in (("FieldMesh", ".h5"), ("Field", ".csv"), ("SolverDeck", ".fem")):
            if not (args.raw / component / (cad + suffix)).is_file():
                raise FileNotFoundError(f"{cad}: missing local {component} source")
    by_cad = {x["sample_key"].split("/")[0]: x for x in source["cases"]}
    if any(by_cad[cad]["sample_key"] != f"{cad}/tor" for cad in TRANSFER_CADS):
        raise ValueError("Expected all three migration-source CADs to have been trained only under tor")
    split = make_split(args.raw, source, args.split, args.seed)
    plan = {"created_at_utc": utc(), "stage": "planned", "workers": args.workers, "split": rel(args.split),
            "split_counts": {k: {"CADs": v["CADs"], "cases": v["cases"]} for k, v in split["splits"].items()},
            "transfer": {"output": rel(args.transfer_output), "case_keys": [f"{cad}/{load}" for cad in TRANSFER_CADS for load in LOADS[:3]],
                         "interpretation": "Same CAD, held-out CAD x load combinations; all four load types occurred in original shared50 training. Not globally unseen load types."},
            "all200": {"output": rel(args.all_output), "CADs": 50, "cases": 200},
            "label_scaler_fitted": False, "network_download": False,
            "source_policy": "Official original DeepJEB raw already local; preserve CSV-aligned signed labels and current explicit25 schema; no resampling"}
    write_json(args.plan, plan)
    print(json.dumps({"event": "split_ready", "path": rel(args.split), "groups": plan["split_counts"]}), flush=True)
    def progress(stage):
        def update(status):
            plan["stage"] = stage
            plan["progress"] = status
            plan["updated_at_utc"] = utc()
            write_json(args.plan, plan)
            print(json.dumps({"event": stage, **status}), flush=True)
        return update
    if args.stage in ("transfer", "both"):
        transfer = prepare_stage(args.raw, args.source, args.transfer_output, source,
                                 {cad: LOADS[:3] for cad in TRANSFER_CADS}, args.workers, {},
                                 plan["transfer"]["interpretation"], progress("preparing_transfer9"))
        plan["transfer"].update({"status": "ready", "nodes_total": transfer["nodes_total"], "elapsed_seconds": transfer["prep_elapsed_seconds"]})
        write_json(args.plan, plan)
        print(json.dumps({"event": "transfer9_ready", **plan["transfer"]}), flush=True)
    if args.stage in ("all", "both"):
        reusable = {}
        if (args.transfer_output / "manifest.json").is_file():
            reusable = {x["sample_key"]: {**x, "npz": str((args.transfer_output / file_name(x["npz"])).resolve())}
                        for x in json.loads((args.transfer_output / "manifest.json").read_text(encoding="utf-8"))["cases"]}
        full = prepare_stage(args.raw, args.source, args.all_output, source,
                             {k.split("/")[0]: LOADS for k in source["selection"]}, args.workers, reusable,
                             "All four original loads of the preserved first50 CADs; use notes/geobc_split.json for strict 40/5/5 grouping", progress("preparing_all200"))
        plan["all200"].update({"status": "ready", "nodes_total": full["nodes_total"], "elapsed_seconds": full["prep_elapsed_seconds"]})
    plan["stage"] = "complete" if args.stage != "plan" else "planned"
    plan["updated_at_utc"] = utc()
    write_json(args.plan, plan)
    print(json.dumps({"event": "preprocessing_finished", "stage": args.stage}), flush=True)


if __name__ == "__main__":
    main()
