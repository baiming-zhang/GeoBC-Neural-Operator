"""Input-only BC-role replay for one frozen TRAIN101_479/ver cached case.

Source/metadata check (no scientific imports, array reads, or CUDA):
  python tools/diagnose_solver_bc_representation.py --source-check
Actual sole-queue invocation (not run by the author):
  python tools/diagnose_solver_bc_representation.py --output \
    runs/geobc_exploration/solver_bc_representation_train101_479_ver

Only cached features[:,0:7], grid[4:11], query_indices and query_weights are
accessed. No coordinates, targets, labels, checkpoint, model or rerasterization.
The 0.5 cutoff describes role reconstruction; it never decodes or repairs BCs.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/geobc_solver_selection"
CASE = "101_479/ver"
CAD = "101_479"
RESOLUTION = 64
CHUNK = 32768
TOLERANCE = 2e-6
ROLE_NAMES = [f"support_reference_constrained_dof_{index}" for index in range(1, 7)] + ["loaded_interface_membership"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def host_path(value):
    text = str(value).replace("\\", "/")
    if os.name == "nt" and text.startswith("/mnt/") and len(text) > 7:
        text = text[5].upper() + ":/" + text[7:]
    elif os.name != "nt" and len(text) > 2 and text[1] == ":":
        text = "/mnt/" + text[0].lower() + "/" + text[3:]
    path = Path(text)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def identity(path, hash_contents=False):
    stat = path.stat()
    result = {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if hash_contents:
        result["sha256"] = digest(path.read_bytes())
    return result


def request_contract():
    """Small standard-library metadata only. No NPY header/value reads."""
    path = CACHE / "manifest.json"
    data = path.read_bytes()
    manifest = json.loads(data)
    cache_contract = manifest.get("contract", {})
    expected_schema = ["node_occupancy", "x", "y", "z"] + [f"original_feature_{index}" for index in range(25)]
    if (manifest.get("status") != "completed" or manifest.get("grid_schema") != expected_schema
            or cache_contract.get("schema") != "geobc_solver_selection_pointgrid_v1"
            or cache_contract.get("grid_resolution") != RESOLUTION
            or cache_contract.get("cloud_points") != 1024
            or CASE not in manifest.get("train_keys", [])
            or CASE not in cache_contract.get("train_keys", [])):
        raise ValueError("Fixed TRAIN cache identity/schema does not match BC-role replay")
    record = manifest["records"][CASE]
    nodes = record.get("num_nodes")
    if (record.get("key") != CASE or record.get("cad_id") != CAD or record.get("group") != "train"
            or isinstance(nodes, bool) or not isinstance(nodes, int) or nodes <= 0):
        raise ValueError("Fixed case record identity/node count mismatch")
    expected = {"features": CACHE / "cases/101_479_ver/features.npy",
                "grid": CACHE / "cases/101_479_ver/grid.npy",
                "query_indices": CACHE / "geometry/101_479/query_indices.npy",
                "query_weights": CACHE / "geometry/101_479/query_weights.npy"}
    files = {}
    for name, target in expected.items():
        actual = host_path(record[name + "_path"])
        if actual != target.resolve() or not actual.is_file():
            raise ValueError(f"Fixed input path missing or changed: {name}")
        files[name] = identity(actual)  # Stat only; no binary-array hash/read.
    producer = ROOT / "geobc/solver_selection_data.py"
    return {"schema": "geobc_solver_bc_representation_v1", "case": CASE, "nodes": nodes,
            "script_sha256": digest(Path(__file__).read_bytes()),
            "cache_manifest": {**identity(path), "sha256": digest(data)},
            "cache_contract_sha256": digest(json.dumps(cache_contract, sort_keys=True, separators=(",", ":")).encode()),
            "cache_contract_schema": cache_contract["schema"],
            "grid_schema": expected_schema, "producer_source": identity(producer, True),
            "input_files": files, "input_slices": {"features": "all N rows, channels [0:7]",
                "grid": "channels [4:11], all R64 grid entries", "query_indices": "all N x8", "query_weights": "all N x8"},
            "query_formula": "flat=grid[4:11].flatten(1).T; (flat[indices]*weights[...,None]).sum(1), float32, eight corners",
            "device": "cuda:0", "chunk_nodes": CHUNK, "grid_resolution": RESOLUTION,
            "range_tolerance": TOLERANCE, "positive_source_definition": "exact source role ==1; negative ==0; other values are reported as source anomalies",
            "description_threshold": 0.5, "clip_or_repair": False,
            "scientific_imports_in_contract": 0, "array_reads_in_contract": 0, "CUDA_initialized_in_contract": False}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def analyze(contract, output, started):
    import numpy as np
    import torch

    files = contract["input_files"]
    arrays = {name: np.load(item["path"], mmap_mode="r", allow_pickle=False) for name, item in files.items()}
    count = contract["nodes"]
    expected_shapes = {"features": (count, 25), "grid": (29, RESOLUTION, RESOLUTION, RESOLUTION),
                       "query_indices": (count, 8), "query_weights": (count, 8)}
    for name, array in arrays.items():
        dtype = np.int32 if name == "query_indices" else np.float32
        if array.shape != expected_shapes[name] or array.dtype != np.dtype(dtype):
            raise ValueError(f"Corrupt input shape/dtype: {name} {array.shape}/{array.dtype}")
    if not torch.cuda.is_available():
        raise RuntimeError("Fixed cuda:0 device is unavailable")
    device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    gpu_started = time.monotonic()
    # Materialize ONLY seven role channels, then use the existing grid ordering.
    grid = torch.as_tensor(np.array(arrays["grid"][4:11], copy=True), device=device)
    flat = grid.flatten(1).T
    source = torch.empty((count, 7), dtype=torch.float32, device=device)
    replay = torch.empty_like(source)
    stencil = {"nonfinite_weights": 0, "weights_outside_unit_interval": 0,
               "partition_of_unity_bad_rows": 0}
    with torch.no_grad():
        for start in range(0, count, CHUNK):
            end = min(start + CHUNK, count)
            indices = torch.as_tensor(np.array(arrays["query_indices"][start:end], copy=True), dtype=torch.int64, device=device)
            if bool(((indices < 0) | (indices >= RESOLUTION**3)).any().item()):
                raise ValueError("Corrupt cached query index outside fixed R64 grid")
            weights = torch.as_tensor(np.array(arrays["query_weights"][start:end], copy=True), device=device)
            source[start:end] = torch.as_tensor(np.array(arrays["features"][start:end, :7], copy=True), device=device)
            replay[start:end] = (flat[indices] * weights[..., None]).sum(1)
            stencil["nonfinite_weights"] += int((~torch.isfinite(weights)).sum().item())
            stencil["weights_outside_unit_interval"] += int(((weights < -TOLERANCE) | (weights > 1 + TOLERANCE)).sum().item())
            sums = weights.sum(1)
            stencil["partition_of_unity_bad_rows"] += int(((~torch.isfinite(sums)) | ((sums - 1).abs() > TOLERANCE)).sum().item())
            atomic_json(output / "status.json", {"status": "running", "phase": "role_replay", "case": CASE,
                        "nodes_processed": end, "nodes": count, "elapsed_seconds": time.monotonic() - started,
                        "updated_at_utc": utc_now()})

        def scalar(value):
            return float(value.item())

        def anomalies(values):
            return {"nonfinite": int((~torch.isfinite(values)).sum().item()),
                    "below_zero_beyond_tolerance": int((values < -TOLERANCE).sum().item()),
                    "above_one_beyond_tolerance": int((values > 1 + TOLERANCE).sum().item())}

        roles = []
        for index, name in enumerate(ROLE_NAMES):
            src, predicted = source[:, index], replay[:, index]
            positive, negative = src == 1, src == 0
            npos, nneg = int(positive.sum().item()), int(negative.sum().item())
            finite_all = bool(torch.isfinite(src).all().item() and torch.isfinite(predicted).all().item())
            stats_defined = finite_all and not stencil["nonfinite_weights"]
            pos_stats = {"mean": None, "min": None, "median": None, "fraction_ge_0_5": None,
                         "reason": "no_source_positive_nodes" if not npos else "nonfinite_all_node_replay" if not stats_defined else None}
            neg_stats = {"mean_absolute_leakage": None, "max_absolute_leakage": None, "sum_absolute_leakage": None,
                         "reason": "no_source_negative_nodes" if not nneg else "nonfinite_all_node_replay" if not stats_defined else None}
            if stats_defined and npos:
                values = predicted[positive].to(torch.float64)
                pos_stats.update(mean=scalar(values.mean()), min=scalar(values.min()),
                                 median=scalar(torch.quantile(values, 0.5)),
                                 fraction_ge_0_5=scalar((values >= 0.5).to(torch.float64).mean()))
            if stats_defined and nneg:
                values = predicted[negative].to(torch.float64).abs()
                neg_stats.update(mean_absolute_leakage=scalar(values.mean()),
                                 max_absolute_leakage=scalar(values.max()), sum_absolute_leakage=scalar(values.sum()))
            roles.append({"role_index": index, "grid_channel": index + 4, "role_name": name,
                          "nodes": count, "source_positive_count": npos, "source_negative_count": nneg,
                          "source_other_or_nonfinite_count": count - npos - nneg,
                          "overall_MAE": scalar((predicted.to(torch.float64) - src.to(torch.float64)).abs().mean()) if stats_defined else None,
                          "statistics_defined_for_all_nodes": stats_defined,
                          "positive_reconstruction": pos_stats, "negative_leakage": neg_stats,
                          "anomalies": {"source": anomalies(src), "cached_grid_role": anomalies(grid[index]),
                                        "reconstructed": anomalies(predicted)}})
        maes = [row["overall_MAE"] for row in roles]
        anomalous = any(row["source_other_or_nonfinite_count"] or any(number for group in row["anomalies"].values() for number in group.values()) for row in roles) or any(stencil.values())
        results = {"nodes": count, "role_count": 7, "chunks": (count + CHUNK - 1) // CHUNK,
                   "all_node_all_7_role_MAE": sum(maes) / 7 if all(value is not None for value in maes) else None,
                   "roles": roles, "stencil_anomalies": stencil,
                   "representation_anomalies_present": bool(anomalous),
                   "statistics_policy": "All N nodes; no invalid-point filtering, clipping or repair. Nonfinite replay makes affected all-node statistics undefined, not a partial-node result. Source membership is exact 0/1; threshold0.5 describes reconstruction only."}
    torch.cuda.synchronize(device)
    gpu_seconds = time.monotonic() - gpu_started
    peak = {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}
    if request_contract() != contract:
        raise ValueError("Input file stat/manifest/source identity changed during read-only diagnostic")
    return {"status": "completed", "case": CASE, "contract": contract,
            "elapsed_seconds": time.monotonic() - started, "device": str(device),
            "GPU_section_seconds": gpu_seconds, "GPU_section_scope": "Seven-role H2D, eight-corner replay and scalar statistics with host status I/O; not pure kernel timing",
            "peak": peak, "results": results, "completed_at_utc": utc_now(),
            "versions": {"torch": torch.__version__, "numpy": np.__version__},
            "input_read_receipt": {"input_only": True, "allowed_arrays": list(files),
                "actual_value_slices": contract["input_slices"], "nodes_processed": count,
                "xyz_values_read": 0, "physical_labels_read": 0, "target_values_read": 0,
                "checkpoints_read": 0, "models_constructed": 0, "FNO_rerasterizations": 0,
                "array_files_written": 0, "GPU_initialized": True},
            "limits": "One fixed TRAIN101_479/ver input representation diagnostic. Channels0:6 are support-reference DOF roles, channel6 is loaded membership; neither is a node Dirichlet rule or stress GT. Fraction0.5 is descriptive, not BC decoding/repair. Role distortion does not establish the cause of model error."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-check", action="store_true")
    args = parser.parse_args(argv)
    if args.source_check:
        ast.parse(Path(__file__).read_bytes(), filename=__file__)
        contract = request_contract()
        print(json.dumps({"status": "source_ready", "AST": "passed", "contract": contract,
                          "array_reads": 0, "scientific_imports": 0, "GPU_initialized": False}, ensure_ascii=False))
        return contract
    if args.output is None:
        parser.error("--output is required for the actual sole-queue diagnostic")
    output = host_path(args.output)
    if not output.is_relative_to(ROOT):
        raise ValueError("Diagnostic output must remain inside the project")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        contract = request_contract()
        summary_path = output / "summary.json"
        if summary_path.is_file():
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            if (previous.get("status") != "completed" or previous.get("contract") != contract
                    or previous.get("case") != CASE or previous.get("results", {}).get("nodes") != contract["nodes"]
                    or previous.get("input_read_receipt", {}).get("input_only") is not True):
                raise ValueError("Existing completed receipt does not match the exact BC-role contract")
            atomic_json(output / "status.json", {"status": "completed", "phase": "completed", "case": CASE,
                        "nodes_processed": contract["nodes"], "elapsed_seconds": previous["elapsed_seconds"],
                        "peak": previous["peak"], "reused": True,
                        "representation_anomalies_present": previous["results"]["representation_anomalies_present"],
                        "updated_at_utc": utc_now()})
            print(json.dumps({"status": "completed", "reused": True, "case": CASE, "output": str(output)}, ensure_ascii=False))
            return previous  # Strict reuse precedes any NumPy/Torch import.
        atomic_json(output / "config.json", {"status": "configured", "contract": contract})
        atomic_json(output / "status.json", {"status": "running", "phase": "input_replay", "case": CASE,
                    "elapsed_seconds": time.monotonic() - started, "updated_at_utc": utc_now()})
        summary = analyze(contract, output, started)
        atomic_json(summary_path, summary)
        atomic_json(output / "status.json", {"status": "completed", "phase": "completed", "case": CASE,
                    "nodes_processed": contract["nodes"], "elapsed_seconds": summary["elapsed_seconds"],
                    "peak": summary["peak"], "representation_anomalies_present": summary["results"]["representation_anomalies_present"],
                    "updated_at_utc": utc_now()})
        print(json.dumps({"status": "completed", "case": CASE, "output": str(output),
                          "elapsed_seconds": summary["elapsed_seconds"],
                          "role_MAE": [row["overall_MAE"] for row in summary["results"]["roles"]]}, ensure_ascii=False))
        return summary
    except Exception as error:
        atomic_json(output / "status.json", {"status": "failed", "phase": "failed", "case": CASE,
                    "elapsed_seconds": time.monotonic() - started, "updated_at_utc": utc_now(),
                    "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
