"""One input-only TRAIN-CAD radius-neighborhood feasibility pilot.

The metadata/AST source check imports only the standard library. Actual CUDA
execution is a separate root-coordinator release, not performed by this check.
No feature, label, weight, model, or neuraloperator dependency is accessed.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "geobc_gino_regular_grid_radius_pilot_v1"
CASE_KEY = "101_479/ver"
ALIGNMENT = "official CSV GRID coordinates -> H5 vertices"
MAX_METADATA_BYTES = 32 * 1024 * 1024


def host_path(value):
    text = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", text):
        text = text[5].upper() + ":/" + text[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", text):
        text = "/mnt/" + text[0].lower() + "/" + text[3:]
    result = Path(text)
    return (result if result.is_absolute() else ROOT / result).resolve()


def project_path(value):
    result = host_path(value)
    if not result.is_relative_to(ROOT):
        raise ValueError(f"Pilot file leaves this project: {result}")
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def reject_constant(value):
    raise ValueError(f"Nonfinite metadata: {value}")


def read_json(filename):
    filename = project_path(filename)
    if filename.suffix != ".json" or not filename.is_file() or filename.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError(f"Expected bounded JSON metadata: {filename}")
    raw = filename.read_bytes()
    obj = json.loads(raw.decode("utf-8-sig"), parse_constant=reject_constant)
    if not isinstance(obj, dict):
        raise ValueError("Expected metadata object")
    return obj, {"path": str(filename), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def stat_record(filename):
    filename = project_path(filename)
    info = filename.stat()
    if not filename.is_file():
        raise ValueError("Expected regular file")
    return {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(filename, obj):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(filename)


def request_contract(plan_path):
    """Stdlib JSON/source identities and source stat only, never an NPY read."""
    plan, plan_receipt = read_json(plan_path)
    recipe = plan.get("recipe", {})
    fixed = {"case_key": CASE_KEY, "resolution": 64, "in_radius": .033, "out_radius": .033,
             "chunk_points": 8192, "device": "cuda", "expected_points": 179316,
             "coordinate_dtype": "float32", "distance_dtype": "float64", "distance_predicate": "d2 <= radius**2"}
    if plan.get("kind") != "gino_radius_input_pilot" or plan.get("status") != "predeclared" or recipe != fixed:
        raise ValueError("Pilot scope/recipe must equal the single predeclared full-N R64/.033 TRAIN case")
    if plan.get("access", {}).get("allowed_scientific_members") != ["xyz.npy"] or plan.get("access", {}).get("labels_features_weights_models") != "forbidden":
        raise ValueError("Expected geometry-only scientific allowlist")
    expected = plan["frozen_input_identity"]
    manifest, manifest_receipt = read_json(plan["selection_manifest"])
    split, split_receipt = read_json(plan["existing_split"])
    if manifest.get("status") != "completed" or manifest.get("contract", {}).get("schema") != "geobc_solver_selection_pointgrid_v1":
        raise ValueError("Expected the existing completed canonical point/grid input producer")
    if (manifest_receipt["sha256"] != expected["manifest_sha256"]
            or digest(manifest["contract"]) != expected["selection_contract_sha256"]
            or split_receipt["sha256"] != expected["split_sha256"]
            or manifest["contract"].get("split_sha256") != split_receipt["sha256"]):
        raise ValueError("Frozen manifest/contract/split identity changed")
    record = manifest["records"][CASE_KEY]
    if (record.get("key") != CASE_KEY or record.get("group") != "train" or record.get("cad_id") != "101_479"
            or record.get("num_nodes") != 179316 or record.get("label_alignment") != ALIGNMENT
            or CASE_KEY not in manifest["train_keys"] or "101_479" not in split["groups"]["train"]
            or any("101_479" in split["groups"][group] for group in ("validation", "test"))):
        raise ValueError("Source is not the fixed aligned TRAIN101_479 canonical geometry")
    filename = project_path(record["xyz_path"])
    if filename != project_path(expected["xyz_path"]) or filename.suffix != ".npy" or stat_record(filename) != expected["xyz_stat"]:
        raise ValueError("Canonical xyz.npy source path/stat changed")
    sources = {}
    for relative in ("geobc/regular_grid_radius.py", "tools/prepare_gino_radius_pilot.py", "geobc/solver_selection_data.py"):
        raw = project_path(relative).read_bytes()
        if len(raw) > MAX_METADATA_BYTES:
            raise ValueError("Small source identity limit exceeded")
        sources[relative] = hashlib.sha256(raw).hexdigest()
    if sources["geobc/solver_selection_data.py"] != expected["producer_sha256"]:
        raise ValueError("Frozen canonical producer source changed")
    return {"schema": SCHEMA, "plan": plan_receipt, "recipe": recipe,
            "selection_manifest": manifest_receipt, "selection_contract_sha256": digest(manifest["contract"]),
            "split": split_receipt,
            "source": {"case_key": CASE_KEY, "cad_id": "101_479", "group": "train", "num_points": 179316,
                       "xyz_path": str(filename), "stat": stat_record(filename), "label_alignment": ALIGNMENT},
            "source_sha256": sources, "access": plan["access"], "limitations": plan["limitations"],
            "cache_schema": {"grid_queries": "R^3 row_ptr; neighbors_index indexes original points",
                             "point_queries": "N row_ptr; neighbors_index indexes x-major unit lattice",
                             "array_dtypes": {"row_ptr": "int64", "neighbors_index": "int64", "squared_distances": "float64", "grid_axis": "float32", "grid_xyz": "float32"},
                             "same_edges": "Two CSR orientations from one enumerated pair set, permuted for grid rows",
                             "no_kernel_weights": True, "input_only": True}}


def source_check(plan_path):
    for relative in ("geobc/regular_grid_radius.py", "tools/prepare_gino_radius_pilot.py"):
        ast.parse(project_path(relative).read_text(encoding="utf-8"), filename=relative)
    contract = request_contract(plan_path)
    return {"status": "source_ready", "contract_sha256": digest(contract), "contract": contract,
            "scientific_array_files_read": 0, "labels_read": 0, "features_read": 0,
            "checkpoint_weights_read": 0, "models_constructed": 0, "GPU_work": False,
            "scientific_modules_imported": [name for name in ("numpy", "torch", "neuralop", "open3d") if name in sys.modules]}


def save_tensor(np, output, name, tensor):
    value = tensor.detach().cpu().numpy()
    filename = output / (name + ".npy")
    temporary = filename.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(filename)
    return {"path": str(filename), "shape": list(value.shape), "dtype": str(value.dtype),
            "array_values_sha256": hashlib.sha256(memoryview(value).cast("B")).hexdigest(),
            "array_payload_bytes": value.nbytes, **stat_record(filename)}


def run(plan_path, output_path):
    contract = request_contract(plan_path)
    output = project_path(output_path)
    if (output / "summary.json").is_file():
        previous, _ = read_json(output / "summary.json")
        if previous.get("status") == "completed":
            if previous.get("config", {}).get("contract") != contract:
                raise ValueError("Completed input pilot contract differs; immutable output is preserved")
            for item in previous["inventory"].values():
                if stat_record(item["path"]) != {key: item[key] for key in ("bytes", "mtime_ns")}:
                    raise ValueError("Completed input cache file stat differs")
            return previous
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Nonempty unfinished pilot output is preserved; choose a new explicit output")
    output.mkdir(parents=True, exist_ok=True)
    config = {"contract": contract, "training_updates": 0, "scaler_fits": 0, "models_constructed": 0,
              "purpose": "Input-radius feasibility only; not GINO training, FEM integration, or continuous-query acceptance"}
    write_json(output / "config.json", config)
    began = time.monotonic()
    state = {"started_at_utc": utc_now(), "scientific_array_files_read": 0, "labels_read": 0, "features_read": 0,
             "checkpoint_weights_read": 0, "models_constructed": 0, "training_updates": 0, "backwards": 0}
    def status(phase, **details):
        state.update(details, phase=phase, updated_at_utc=utc_now(), elapsed_seconds=time.monotonic()-began)
        write_json(output / "status.json", state)
    status("metadata_contract_passed")
    receipt = {"allowed_scientific_files": [contract["source"]["xyz_path"]], "scientific_files_opened": [],
               "array_members": ["xyz.npy"], "labels_read": 0, "features_read": 0, "weights_read": 0,
               "models_constructed": 0, "training_updates": 0, "backwards": 0, "CUDA_initialized": False}
    try:
        import numpy as np
        import torch
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from geobc.regular_grid_radius import build_regular_grid_radius, degree_statistics
        source = project_path(contract["source"]["xyz_path"])
        if stat_record(source) != contract["source"]["stat"]:
            raise ValueError("Canonical source stat changed between contract and input access")
        status("reading_allowed_xyz_only", scientific_array_files_read=1)
        stage = time.monotonic()
        receipt["scientific_files_opened"].append(str(source))
        points_cpu = np.load(source, allow_pickle=False)
        if (not isinstance(points_cpu, np.ndarray) or points_cpu.shape != (179316, 3)
                or points_cpu.dtype != np.float32 or not points_cpu.flags.c_contiguous):
            raise ValueError("Canonical point shape/dtype/C-order differs; no cast or reordering allowed")
        if stat_record(source) != contract["source"]["stat"]:
            raise ValueError("Canonical source stat changed during its sole read")
        input_read_seconds = time.monotonic()-stage
        receipt["loaded_xyz_values_sha256"] = hashlib.sha256(memoryview(points_cpu).cast("B")).hexdigest()
        stage = time.monotonic()
        if not torch.cuda.is_available():
            raise RuntimeError("This fixed pilot requires CUDA; no CPU or sampling fallback")
        device = torch.device("cuda")
        torch.cuda.init(); receipt["CUDA_initialized"] = True
        CUDA_initialization_seconds = time.monotonic()-stage
        torch.cuda.reset_peak_memory_stats(device)
        stage = time.monotonic()
        points = torch.as_tensor(points_cpu, device=device)
        torch.cuda.synchronize(device)
        H2D_seconds = time.monotonic()-stage
        def progress(details):
            phase = details.pop("phase")
            status(phase, **details)
        stage = time.monotonic()
        result = build_regular_grid_radius(points, resolution=64, radius=.033, chunk_points=8192, progress=progress)
        torch.cuda.synchronize(device)
        build_seconds = time.monotonic()-stage
        status("query_degree_statistics")
        stage = time.monotonic()
        grid_stats = degree_statistics(result["grid_queries"]["row_ptr"])
        point_stats = degree_statistics(result["point_queries"]["row_ptr"])
        distances = result["point_queries"]["squared_distances"]
        distance_stats = {"min_squared_distance": float(distances.min()) if len(distances) else None,
                          "max_squared_distance": float(distances.max()) if len(distances) else None,
                          "pairs_exactly_at_radius_squared": int((distances == result["construction"]["radius_squared"]).sum()),
                          "storage_dtype": "float64", "units": "normalized length squared"}
        torch.cuda.synchronize(device)
        statistics_seconds = time.monotonic()-stage
        memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                  "scope": "Measured Torch allocator for coordinates, enumeration/sort/CSR/statistics; not GNO/FNO/kernel activation/backward or total driver memory"}
        status("saving_input_only_cache")
        stage = time.monotonic()
        inventory = {}
        for orientation in ("grid_queries", "point_queries"):
            for member, tensor in result[orientation].items():
                inventory[orientation + "_" + member] = save_tensor(np, output, orientation + "_" + member, tensor)
        for member in ("grid_axis", "grid_xyz"):
            inventory[member] = save_tensor(np, output, member, result[member])
        cache_save_seconds = time.monotonic()-stage
        feasible = all(result["checks"].values()) and point_stats["empty_rows"] == 0
        summary = {"status": "completed", "config": config, "feasible": feasible,
                   "stats": {"grid_queries": grid_stats, "point_queries": point_stats,
                             "distances": distance_stats, "construction": result["construction"],
                             "empty_latent_rows_legal": True, "point_queries_all_covered": point_stats["empty_rows"] == 0},
                   "checks": result["checks"], "inventory": inventory,
                   "cache_bytes": sum(item["bytes"] for item in inventory.values()),
                   "cache_payload_bytes": sum(item["array_payload_bytes"] for item in inventory.values()),
                   "costs": {"input_read_seconds": input_read_seconds, "CUDA_initialization_seconds": CUDA_initialization_seconds,
                             "H2D_seconds": H2D_seconds,
                             "build_seconds": build_seconds, **result["costs"],
                             "statistics_seconds": statistics_seconds, "cache_save_seconds": cache_save_seconds,
                             "elapsed_seconds": time.monotonic()-began},
                   "memory": memory, "receipt": receipt,
                   "environment": {"torch_version": str(torch.__version__), "CUDA_runtime": torch.version.cuda,
                                   "GPU_name": torch.cuda.get_device_name(device)},
                   "candidate_model_memory": {"measured": False, "reason": "No model/kernel learned activations or backward exists in this pilot"},
                   "limitations": contract["limitations"], "completed_at_utc": utc_now()}
        write_json(output / "summary.json", summary)
        status("completed", feasible=feasible, cache_bytes=summary["cache_bytes"], edges=result["construction"]["edges_per_orientation"])
        return summary
    except Exception as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "summary.json", {"status": "failed", "config": config, "receipt": receipt,
                   "error": f"{type(error).__name__}: {error}", "costs": {"elapsed_seconds": time.monotonic()-began}})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-check", action="store_true")
    args = parser.parse_args(argv)
    if args.source_check:
        result = source_check(args.plan)
    else:
        if args.output is None:
            raise ValueError("Actual pilot requires --output")
        summary = run(args.plan, args.output)
        result = {"status": summary["status"], "feasible": summary.get("feasible"),
                  "elapsed_seconds": summary["costs"]["elapsed_seconds"], "cache_bytes": summary.get("cache_bytes"),
                  "summary": str(project_path(args.output) / "summary.json")}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
