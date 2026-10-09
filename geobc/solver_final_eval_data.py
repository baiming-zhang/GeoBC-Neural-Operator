"""Gated, input-only final-CAD cache for the frozen solver evaluation.

Metadata entry points use only the standard library. Scientific dependencies
are imported after both the final-rule and six-confirmation gates pass. The
model loader never opens the source NPZ or returns target/physical fields;
the evaluator separately reads aligned truth only after its prediction.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "geobc_solver_final_input_v1"
ALIGNMENT = "official CSV GRID coordinates -> H5 vertices"
INPUT_ARRAYS = ("xyz", "features", "cloud", "mask", "grid", "query_indices", "query_weights")
MAX_JSON_BYTES = 32 * 1024 * 1024
CONFIRMATION_SOURCES = (
    "geobc/solver_selection_train.py", "geobc/solver_selection_models.py",
    "geobc/solver_selection_data.py", "geobc/benchmark.py",
    "geobc/conditioned_field.py", "geobc/tcnn_fields.py",
    "geobc/transformers.py", "geobc/fno.py",
    "third_party/transformer_sources.json", "geobc/solver_confirmation_train.py",
)
RULE_SCOPE_KEYS = ("existing_split", "confirmation_plan", "selection_cache", "prepared_data",
                   "eval_cache", "known_loads", "combination_loads", "routes", "input_recipe", "rules")


def _path(value):
    text = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", text):
        text = text[5].upper() + ":/" + text[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", text):
        text = "/mnt/" + text[0].lower() + "/" + text[3:]
    result = Path(text)
    return (result if result.is_absolute() else ROOT / result).resolve()


def _project_path(value):
    result = _path(value)
    if not result.is_relative_to(ROOT):
        raise ValueError(f"Final-evaluation receipt/input leaves project: {result}")
    return result


def _constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _json(filename, optional=False):
    filename = Path(filename)
    if optional and not filename.is_file():
        return {}, {"path": str(filename), "present": False}
    if filename.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"Small metadata limit exceeded: {filename}")
    data = filename.read_bytes()
    result = json.loads(data.decode("utf-8-sig"), parse_constant=_constant)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {filename}")
    return result, {"path": str(filename), "present": True, "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest()}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _stat(filename):
    filename = Path(filename)
    stat = filename.stat()
    if not filename.is_file():
        raise ValueError(f"Expected regular file: {filename}")
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _recipe(plan):
    recipe = dict(plan["input_recipe"])
    expected = {"cloud_seed": 2026, "cloud_points": 1024, "geometry_dim": 32,
                "grid_resolution": 64, "features": 25, "channels": [0, 1, 2, 3], "refit_scaler": False}
    if any(recipe.get(key, value) != value for key, value in expected.items()):
        raise ValueError("Final inputs cannot change shared cloud/grid/feature/scaler recipe")
    for key, value in expected.items():
        recipe.setdefault(key, value)
    recipe.setdefault("device", "cuda")
    if plan["known_loads"] != ["ver", "hor", "tor"] or plan["combination_loads"] != ["dia"]:
        raise ValueError("Final evaluation uses exactly final5 x known3 and dia")
    return recipe


def _confirmation_gate(plan, confirmation, selection, source_hashes):
    jobs = confirmation.get("jobs", [])
    recipe = dict(confirmation["recipe"])
    methods, seeds = recipe.get("methods", []), recipe.get("seeds", [])
    expected_pairs = {(method, seed) for method in methods for seed in seeds}
    if (confirmation.get("candidates_frozen") is not True or len(jobs) != 6
            or len(methods) != 3 or seeds != [2026, 2027]
            or {(job.get("method"), job.get("seed")) for job in jobs} != expected_pairs
            or len(expected_pairs) != 6):
        raise ValueError("Expected six distinct jobs from the frozen three-route/two-seed confirmation")
    outputs = [_project_path(job["output"]) for job in jobs]
    if len(set(outputs)) != 6:
        raise ValueError("Confirmation jobs share an output directory")
    if "confirmation_runs" in plan and {_project_path(p) for p in plan["confirmation_runs"]} != set(outputs):
        raise ValueError("Final plan confirmation_runs differ from confirmation jobs")
    identity = {"recipe": recipe, **{key: confirmation[key] for key in ("existing_split", "prepared_data", "data_cache")}}
    recipe_sha = _digest(identity)
    runs, invalid = [], []
    for job, directory in zip(jobs, outputs):
        summary, summary_id = _json(directory / "summary.json", optional=True)
        config, config_id = _json(directory / "config.json", optional=True)
        state = summary.get("status", "pending")
        valid, errors = None, []
        if state == "completed":
            contract = config.get("contract", {})
            if not config or summary.get("config") != config:
                errors.append("summary and saved config identity differ")
            if any(obj.get("method") != job["method"] or obj.get("seed") != job["seed"]
                   for obj in (summary, config, contract)):
                errors.append("method/seed identity differs from frozen job")
            if contract.get("recipe") != recipe or contract.get("recipe_sha256") != recipe_sha:
                errors.append("requested confirmation recipe identity differs")
            for key in ("scaler", "cache_contract", "train_keys", "validation_keys", "training_readout_keys"):
                expected = selection["contract"] if key == "cache_contract" else selection[key]
                if config.get(key) != expected or contract.get(key) != expected:
                    errors.append(f"shared {key} identity differs")
            source_ids = contract.get("source_sha256", {})
            if not isinstance(source_ids, dict) or set(source_ids) != set(CONFIRMATION_SOURCES):
                errors.append("requested source identities differ from the small-source allowlist")
            else:
                for relative in CONFIRMATION_SOURCES:
                    filename = _project_path(relative)
                    if filename.stat().st_size > MAX_JSON_BYTES:
                        raise ValueError(f"Small source limit exceeded: {filename}")
                    if relative not in source_hashes:
                        source_hashes[relative] = hashlib.sha256(filename.read_bytes()).hexdigest()
                    if source_hashes[relative] != source_ids[relative]:
                        errors.append(f"frozen source identity differs: {relative}")
            endpoint = summary.get("endpoint_step")
            if (not isinstance(endpoint, int) or isinstance(endpoint, bool) or endpoint <= 0
                    or summary.get("final_metrics", {}).get("step") != endpoint):
                errors.append("committed confirmation endpoint identity differs")
            valid = not errors
            if errors:
                invalid.append({"method": job["method"], "seed": job["seed"], "errors": errors})
        runs.append({"method": job["method"], "seed": job["seed"], "output": str(directory),
                     "status": state, "identity_valid": valid, "summary": summary_id, "config": config_id,
                     "errors": errors})
    return {"all_six_completed": all(run["status"] == "completed" and run["identity_valid"] is True for run in runs),
            "invalid_runs": invalid, "recipe_sha256": recipe_sha, "runs": runs}


def request_contract(plan_path):
    """Read JSON/source identities/path stats only; pending rules are allowed."""
    plan, _ = _json(_project_path(plan_path))
    recipe = _recipe(plan)
    split, split_id = _json(_project_path(plan["existing_split"]))
    groups = split["groups"]
    if [len(groups[key]) for key in ("train", "validation", "test")] != [40, 5, 5]:
        raise ValueError("Expected the existing whole-CAD40/5/5 split")
    if any(len(set(groups[key])) != len(groups[key]) for key in groups):
        raise ValueError("Repeated CAD within split")
    if any(set(groups[a]) & set(groups[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("CAD groups overlap")
    selection, selection_id = _json(_project_path(plan["selection_cache"]) / "manifest.json")
    if selection.get("status") != "completed" or selection.get("contract", {}).get("schema") != "geobc_solver_selection_pointgrid_v1":
        raise ValueError("Shared selection input/scaler metadata are incomplete")
    shared = selection["contract"]
    if shared.get("split_sha256") != split_id["sha256"]:
        raise ValueError("Final split differs from shared scaler/input split")
    for key in ("cloud_seed", "cloud_points", "grid_resolution"):
        if shared.get(key) != recipe[key]:
            raise ValueError(f"Final {key} differs from shared selection recipe")
    for group in ("train", "validation"):
        expected = [f"{cad}/{load}" for cad in groups[group] for load in plan["known_loads"]]
        if selection.get(group + "_keys") != expected:
            raise ValueError(f"Shared {group} membership differs from whole-CAD split")
    scaler = selection["scaler"]
    if (scaler.get("channels") != [0, 1, 2, 3] or len(scaler.get("mean", [])) != 4
            or len(scaler.get("std", [])) != 4 or not all(_finite(x) for x in scaler["mean"])
            or not all(_finite(x) and x > 0 for x in scaler["std"])
            or not _finite(scaler.get("stress_scale")) or scaler["stress_scale"] <= 0):
        raise ValueError("Expected existing finite TRAIN-only signed4/asinh scaler")
    if recipe.get("train_scaler_source") and _project_path(recipe["train_scaler_source"]) != _project_path(plan["selection_cache"]) / "manifest.json":
        raise ValueError("Final scaler source must be the existing shared manifest")
    confirmation, _ = _json(_project_path(plan["confirmation_plan"]))
    if (_project_path(confirmation["existing_split"]) != _project_path(plan["existing_split"])
            or _project_path(confirmation["data_cache"]) != _project_path(plan["selection_cache"])
            or _project_path(confirmation["prepared_data"]) != _project_path(plan["prepared_data"])):
        raise ValueError("Final data metadata differ from confirmation")
    source_hashes = {relative: hashlib.sha256(_project_path(relative).read_bytes()).hexdigest()
                     for relative in ("geobc/solver_final_eval_data.py", "geobc/solver_selection_data.py", "geobc/conditioned_field.py")}
    gate = _confirmation_gate(plan, confirmation, selection, source_hashes)
    known = [f"{cad}/{load}" for cad in groups["test"] for load in plan["known_loads"]]
    combination = [f"{cad}/{load}" for cad in groups["test"] for load in plan["combination_loads"]]
    base = _project_path(plan["prepared_data"])
    source_records = []
    for key in known + combination:
        filename = _project_path(split["case_paths"][key])
        if not filename.is_relative_to(base):
            raise ValueError("Final source leaves the declared prepared dataset")
        source_records.append({"key": key, "source_npz_path": str(filename), "stat": _stat(filename)})
    rule_scope = {key: plan.get(key) for key in RULE_SCOPE_KEYS}
    contract = {"schema": SCHEMA, "plan_path": str(_project_path(plan_path)),
                "final_rules_frozen": plan.get("final_rules_frozen") is True,
                "final_rules_sha256": _digest(rule_scope), "routes": plan.get("routes", []),
                "input_recipe": recipe, "split": {**split_id, "test_CADs": groups["test"]},
                "selection_cache": {"manifest": selection_id, "contract_sha256": _digest(shared), "scaler_sha256": _digest(scaler)},
                "scaler": scaler, "known_keys": known, "combination_keys": combination,
                "source_npz_metadata": source_records, "confirmation_gate": gate,
                "source_sha256": source_hashes,
                "truth_access": "Input APIs never read displacement/stress members; evaluator may read aligned truth only after prediction"}
    return contract, plan


def _require_actual(contract):
    if contract["final_rules_frozen"] is not True:
        raise ValueError("Final input access blocked: final_rules_frozen must be true")
    if contract["confirmation_gate"]["all_six_completed"] is not True:
        raise ValueError("Final input access blocked: all six confirmation summaries/config identities must be completed and matching")
    if not contract["routes"]:
        raise ValueError("Final input access blocked: no final routes frozen")


def _record_paths(cache, key, cad):
    geometry = cache / "geometry" / cad
    case = cache / "cases" / key.replace("/", "_")
    return {name + "_path": str((case / f"{name}.npy" if name in ("features", "grid") else geometry / f"{name}.npy").resolve()) for name in INPUT_ARRAYS}


def read_cache_metadata(plan_path, *, contract=None):
    """Validate input-cache metadata, optionally reusing this call's contract.

    A reused contract must come from request_contract for the same current
    frozen plan. Its six-run identities are not reread; the complete snapshot
    must still match the immutable cache manifest.
    """
    if contract is None:
        contract, plan = request_contract(plan_path)
    else:
        plan, _ = _json(_project_path(plan_path))
        if (contract.get("schema") != SCHEMA
                or _project_path(contract["plan_path"]) != _project_path(plan_path)
                or plan.get("final_rules_frozen") is not True
                or contract.get("final_rules_sha256") != _digest({key: plan.get(key) for key in RULE_SCOPE_KEYS})
                or contract.get("input_recipe") != _recipe(plan)):
            raise ValueError("Reused final-input contract does not match the current frozen plan")
    _require_actual(contract)
    cache = _project_path(plan["eval_cache"])
    manifest, _ = _json(cache / "manifest.json")
    keys = contract["known_keys"] + contract["combination_keys"]
    if (manifest.get("status") != "completed" or manifest.get("cache_schema") != SCHEMA
            or manifest.get("contract") != contract or manifest.get("scaler") != contract["scaler"]
            or set(manifest.get("records", {})) != set(keys)):
        raise ValueError("Final input cache is incomplete or its frozen metadata contract differs")
    sources = {item["key"]: item for item in contract["source_npz_metadata"]}
    for key, record in manifest["records"].items():
        expected = _record_paths(cache, key, key.split("/")[0])
        if (record.get("key") != key or record.get("input_only") is not True
                or record.get("source_npz_path") != sources[key]["source_npz_path"]
                or record.get("label_alignment") != ALIGNMENT
                or any(record.get(name) != value for name, value in expected.items())
                or any(name in record for name in ("target_path", "physical_path", "displacement_path", "stress_path"))):
            raise ValueError(f"Final record violates input-only path/schema contract: {key}")
        for name in INPUT_ARRAYS:
            if _stat(_project_path(record[name + "_path"])) != record["input_file_stats"][name]:
                raise ValueError(f"Final input cache file metadata changed: {key}/{name}")
    return manifest


def prepare_data(plan_path):
    """Create only final5 input arrays after the strict metadata gates pass."""
    contract, plan = request_contract(plan_path)
    _require_actual(contract)
    cache = _project_path(plan["eval_cache"])
    if (cache / "manifest.json").is_file():
        return read_cache_metadata(plan_path, contract=contract)
    import numpy as np
    import torch
    from .solver_selection_data import _save, _write, _stencil, _splat
    from .conditioned_field import load_geometry_cloud
    began = time.monotonic()
    cache.mkdir(parents=True, exist_ok=True)
    recipe, resolution = contract["input_recipe"], contract["input_recipe"]["grid_resolution"]
    device = torch.device(recipe["device"])
    axis = torch.linspace(0, 1, resolution, device=device)
    coords = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"))
    records, geometry = {}, {}
    for index, source in enumerate(contract["source_npz_metadata"]):
        key, filename = source["key"], _project_path(source["source_npz_path"])
        cad, load = key.split("/")
        if _stat(filename) != source["stat"]:
            raise ValueError(f"Source metadata changed after gate: {key}")
        with np.load(filename, allow_pickle=False) as stored:
            if str(stored["sample_key"].item()) != key or str(stored["label_alignment"].item()) != ALIGNMENT:
                raise ValueError(f"Final source identity/alignment differs: {key}")
            xyz = np.asarray(stored["xyz"], dtype=np.float32)
            features = np.asarray(stored["features"], dtype=np.float32)
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz)
                or features.shape != (len(xyz), 25)
                or not np.isfinite(xyz).all() or not np.isfinite(features).all()):
            raise ValueError(f"Malformed/nonfinite final geometry/condition inputs: {key}")
        paths = _record_paths(cache, key, cad)
        if cad not in geometry:
            cloud = load_geometry_cloud(filename, cloud_points=recipe["cloud_points"], seed=recipe["cloud_seed"], cad_key=cad)
            indices, weights = _stencil(torch.as_tensor(xyz, device=device), resolution)
            for name, value in (("xyz", xyz), ("cloud", cloud.points), ("mask", cloud.mask),
                                ("query_indices", indices.to(torch.int32).cpu().numpy()), ("query_weights", weights.cpu().numpy())):
                _save(paths[name + "_path"], value)
            geometry[cad] = {"xyz": xyz, "indices": indices, "weights": weights,
                             "xyz_sha256": hashlib.sha256(xyz.tobytes(order="C")).hexdigest()}
            _write(cache / "geometry" / cad / "cloud_metadata.json", cloud.record())
        elif not np.array_equal(xyz, geometry[cad]["xyz"]):
            raise ValueError(f"Final geometry/node order changed across loads: {cad}")
        indices, weights = geometry[cad]["indices"], geometry[cad]["weights"]
        raster, active = _splat(torch.as_tensor(features, device=device), indices, weights, resolution)
        grid = torch.cat((active.float().reshape(1, resolution, resolution, resolution), coords,
                          raster.T.reshape(25, resolution, resolution, resolution)), dim=0)
        if not bool(torch.isfinite(grid).all()):
            raise ValueError(f"Nonfinite target-free final grid: {key}")
        _save(paths["features_path"], features)
        _save(paths["grid_path"], grid.cpu().numpy())
        record = {"key": key, "cad_id": cad, "load": load,
                  "scope": "known" if key in contract["known_keys"] else "combination",
                  "num_nodes": len(xyz), "source_npz_path": str(filename), "source_stat": source["stat"],
                  "label_alignment": ALIGNMENT, "input_only": True, "cache_schema": SCHEMA,
                  "cache_root": str(cache), "xyz_sha256": geometry[cad]["xyz_sha256"], **paths,
                  "grid_occupied_fraction": float(active.float().mean()),
                  "input_file_stats": {name: _stat(Path(paths[name + "_path"])) for name in INPUT_ARRAYS}}
        records[key] = record
        _write(cache / "status.json", {"phase": "preparing_inputs", "completed_cases": index+1,
                                      "total_cases": 20, "elapsed_seconds": time.monotonic()-began,
                                      "target_members_read": 0, "current_key": key})
        del features, raster, active, grid
    manifest = {"status": "completed", "cache_schema": SCHEMA, "contract": contract,
                "records": records, "known_keys": contract["known_keys"], "combination_keys": contract["combination_keys"],
                "scaler": contract["scaler"], "array_keys": list(INPUT_ARRAYS), "target_members_read": 0,
                "input_members_read": ["xyz", "features", "faces", "sample_key", "label_alignment"],
                "scaler_fitted": False, "learned_global_context_cached": False,
                "grid_schema": ["node_occupancy", "x", "y", "z"] + [f"original_feature_{i}" for i in range(25)],
                "grid_definition": "Same target-free trilinear node-splat input means; no GT roundtrip/label raster, live global32 appended by model",
                "elapsed_seconds": time.monotonic()-began, "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    _write(cache / "manifest.json", manifest)
    _write(cache / "status.json", {"phase": "completed", "completed_cases": 20, "target_members_read": 0,
                                  "elapsed_seconds": manifest["elapsed_seconds"]})
    return manifest


def load_evaluation_case(record, device, for_fno=False):
    """Load cached inputs only; source_npz_path is deliberately never opened."""
    if record.get("input_only") is not True or record.get("cache_schema") != SCHEMA:
        raise ValueError("Expected a verified final input-only cache record")
    cache = _project_path(record["cache_root"])
    expected = _record_paths(cache, record["key"], record["cad_id"])
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Input record path layout differs from final cache")
    import numpy as np
    import torch
    def tensor(name, dtype=torch.float32, batch=False):
        filename = _project_path(record[name + "_path"])
        if _stat(filename) != record["input_file_stats"][name]:
            raise ValueError(f"Input cache metadata changed: {record['key']}/{name}")
        value = np.array(np.load(filename, mmap_mode="r", allow_pickle=False), copy=True)
        result = torch.as_tensor(value, dtype=dtype, device=device)
        return result.unsqueeze(0) if batch else result
    result = {"key": record["key"], "xyz": tensor("xyz"), "features": tensor("features"),
              "cloud": tensor("cloud", batch=True), "mask": tensor("mask", torch.bool, batch=True)}
    if result["xyz"].shape != (record["num_nodes"], 3) or result["features"].shape != (record["num_nodes"], 25):
        raise ValueError("Final cached geometry/feature shape differs")
    if result["cloud"].shape != (1, 1024, 3) or result["mask"].shape != (1, 1024):
        raise ValueError("Final cached geometry-cloud schema differs")
    if for_fno:
        result.update(grid=tensor("grid", batch=True), query_indices=tensor("query_indices", torch.int64),
                      query_weights=tensor("query_weights"))
        if (result["grid"].shape != (1, 29, 64, 64, 64)
                or result["query_indices"].shape != (record["num_nodes"], 8)
                or result["query_weights"].shape != (record["num_nodes"], 8)):
            raise ValueError("Final cached point-to-grid query schema differs")
    return result


def source_check(plan_path):
    """Pending-safe stdlib-only AST/metadata check, never preparing inputs."""
    ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=__file__)
    contract, _ = request_contract(plan_path)
    return {"status": "source_ready", "schema": SCHEMA,
            "source_sha256": contract["source_sha256"]["geobc/solver_final_eval_data.py"],
            "contract_sha256": _digest(contract), "final_rules_frozen": contract["final_rules_frozen"],
            "confirmation_completed": sum(run["status"] == "completed" for run in contract["confirmation_gate"]["runs"]),
            "confirmation_metadata_gate_passed": contract["confirmation_gate"]["all_six_completed"],
            "known_case_count": len(contract["known_keys"]), "combination_case_count": len(contract["combination_keys"]),
            "scientific_arrays_read": 0, "truth_members_read": 0, "weights_read": 0, "CUDA_initialized": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("notes/geobc_solver_final_eval_plan.json"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--source-check", action="store_true")
    args = parser.parse_args(argv)
    result = source_check(args.plan) if args.source_check else prepare_data(args.plan)
    if args.prepare:
        result = {"status": result["status"], "cases": len(result["records"]), "target_members_read": 0,
                  "elapsed_seconds": result["elapsed_seconds"]}
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
