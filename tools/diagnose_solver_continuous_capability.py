"""Bounded TRAIN solver capability, with separate native FNO and nodal P2.

Metadata only: python tools/diagnose_solver_continuous_capability.py --source-check
Sole released GPU run: python tools/diagnose_solver_continuous_capability.py \
  --plan notes/geobc_solver_continuous_capability_plan.json --output \
  runs/geobc_exploration/solver_continuous_capability_train101_479_ver

No preparation, training, array export, final case, feature extension or repair.
Scientific imports and checkpoint/array reads occur only in actual().
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = "notes/geobc_solver_continuous_capability_plan.json"


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path(value):
    text = str(value).replace("\\", "/")
    if os.name == "nt" and text.startswith("/mnt/"):
        text = text[5].upper() + ":/" + text[7:]
    elif os.name != "nt" and len(text) > 2 and text[1] == ":":
        text = "/mnt/" + text[0].lower() + "/" + text[3:]
    result = Path(text)
    return result.resolve() if result.is_absolute() else (ROOT / result).resolve()


def read(value):
    return json.loads(path(value).read_text(encoding="utf-8-sig"))


def stat(value):
    item = path(value).stat()
    return {"bytes": item.st_size, "mtime_ns": item.st_mtime_ns}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(target, value):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def request_contract(plan_path=DEFAULT_PLAN):
    """Pure standard-library identity/stat/JSON; no scientific imports/reads."""
    plan = read(plan_path)
    require(plan["kind"] == "solver_continuous_capability" and plan["status"] == "frozen_source_ready", "Plan is not frozen")
    scope = plan["scope"]
    require(scope["case"] == "101_479/ver" and scope["logical_queries"] == 504
            and len(scope["selected_rows"]) == 32 and len(scope["shared_faces"]) == 8, "Fixed query scope changed")
    require({(r["method"], r["seed"]) for r in plan["routes"]}
            == {(m, s) for m in ("fno", "mlp") for s in (2026, 2027)}
            and len(plan["routes"]) == 4, "Only four locked primary checkpoints are allowed")
    for name, expected in plan["source_sha256"].items():
        require(sha(path(name)) == expected, f"Frozen source changed: {name}")
    for record in plan["parents"].values():
        require(sha(path(record["path"])) == record["sha256"], "Parent metadata changed")
    prior = read(plan["parents"]["geometry_result"]["path"])
    gate = prior["geometry_gate"]
    require(prior["status"] == "completed" and prior["gate_pass"] is True
            and gate["selected_original_rows"] == scope["selected_rows"]
            and gate["selected_shared_faces"] == scope["shared_faces"], "Passed geometry receipt differs")
    for name, record in plan["files"].items():
        require(stat(record["path"]) == record["stat"], f"Scientific file identity changed: {name}")
    final = read(plan["parents"]["final_plan"]["path"])
    primary = {r["route_id"]: r for r in final["routes"] if r["checkpoint_role"] == "endpoint_primary"}
    for route in plan["routes"]:
        original = primary[route["route_id"]]
        require(route["checkpoint_role"] == "endpoint_primary" and
                all(route[k] == original[k] for k in ("method", "seed", "selected_step", "checkpoint", "checkpoint_stat", "expected_model_source_config", "parameter_counts")), "Locked primary route differs")
        require(stat(route["checkpoint"]) == route["checkpoint_stat"], "Checkpoint stat differs")
        config = read(route["config_path"])
        require(sha(path(route["config_path"])) == route["config_file_sha256"]
                and digest(config) == route["checkpoint_config_sha256"], "Checkpoint origin config differs")
    manifest = read(plan["cache_manifest"]["path"])
    require(sha(path(plan["cache_manifest"]["path"])) == plan["cache_manifest"]["sha256"]
            and manifest["status"] == "completed" and scope["case"] in manifest["train_keys"]
            and manifest["contract"]["grid_resolution"] == 64
            and manifest["scaler"] == plan["scaler"], "TRAIN cache/scaler identity differs")
    return {"schema": "geobc_solver_continuous_capability_v1", "plan_sha256": sha(path(plan_path)),
            "script_sha256": sha(Path(__file__)), "source_sha256": plan["source_sha256"],
            "parents": plan["parents"], "files": plan["files"], "cache_manifest": plan["cache_manifest"],
            "routes": [{k: r[k] for k in ("route_id", "method", "seed", "selected_step", "checkpoint", "checkpoint_stat", "checkpoint_config_sha256")} for r in plan["routes"]],
            "scope": scope, "normalization": plan["normalization"], "scaler": plan["scaler"],
            "numeric_contract": plan["numeric_contract"], "native_physical_coordinates": "(X_mm-bbox_min_mm)/L_mm; derivative std_u/L_mm",
            "device": "cuda:0", "precision": "inherited model FP32/TF32 high; physical maps/stencils/gradients/metrics FP64; common FNO node-query FP32 cache",
            "scientific_imports_in_contract": 0, "scientific_array_reads_in_contract": 0,
            "CUDA_initialized_in_contract": False}


def source_check(plan_path):
    began = time.monotonic()
    receipt = path("notes/solver_continuous_capability_source_check.json")
    require(not receipt.exists(), "Unique author source check already exists; do not repeat")
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    contract = request_contract(plan_path)
    result = {"status": "passed", "plan_sha256": contract["plan_sha256"], "tool_sha256": contract["script_sha256"],
              "contract_sha256": digest(contract), "AST_checks": 1, "fixture_runs": 0,
              "scientific_array_reads": 0, "labels_read": 0, "checkpoints_read": 0, "model_calls": 0,
              "CUDA_initialized": False, "elapsed_seconds": time.monotonic()-began, "created_at_utc": utc()}
    atomic_json(receipt, result)
    print(json.dumps(result, allow_nan=False))
    return 0


def load_math():
    spec = importlib.util.spec_from_file_location("frozen_quadratic_kinematics", path("tools/diagnose_quadratic_kinematics.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def norm_error(np, a, b):
    return float(np.max(np.abs(a-b)) / max(1., float(np.max(np.abs(a))), float(np.max(np.abs(b)))))


def value_metrics(np, predicted, truth):
    result = {}
    for label, section, unit in (("u", slice(0, 3), "mm"), ("stress", slice(3, 4), "MPa")):
        p, t = predicted[..., section], truth[..., section]
        error, ref = float(np.sum((p-t)**2)), float(np.sum(t*t))
        result[label] = {"relative_L2": None if ref == 0 else math.sqrt(error/ref),
                         "reason": "exact_zero_reference_energy" if ref == 0 else None,
                         "RMS_error_"+unit: float(np.sqrt(np.mean(np.sum((p-t)**2, axis=-1)))),
                         "sum_squared_error": error, "sum_squared_reference": ref}
    return result


def aggregate_kinematics(np, helper, weights, truth, predicted, rows):
    # Across cells, constant per-cell P1 strains need an explicit Gauss axis.
    combined = helper.metrics(np, weights,
        {k: v[:, None] if k == "eps1" else v for k, v in truth.items()},
        {k: v[:, None] if k == "eps1" else v for k, v in predicted.items()})
    per_cell = [helper.metrics(np, weights[i], {k: v[i] for k, v in truth.items()},
                               {k: v[i] for k, v in predicted.items()}) for i in range(len(rows))]
    worst = {}
    for name in ("P1", "P2"):
        available = [(m[name]["relative_L2_to_truth"]["value"], row) for m, row in zip(per_cell, rows)
                     if m[name]["relative_L2_to_truth"]["value"] is not None]
        value, row = max(available) if available else (None, None)
        worst[name] = {"relative_L2_to_truth": value, "original_row": row,
                       "zero_reference_cells": len(rows)-len(available)}
    combined["worst_cell"] = worst
    return combined


def native_stencil(torch, canonical, resolution):
    """Exact bounded-cube test, no coordinate clip. FP64 trilinear derivative."""
    valid = torch.isfinite(canonical).all(1) & (canonical >= 0).all(1) & (canonical <= 1).all(1)
    if not bool(valid.all().item()):
        return None, {"available": False, "reason": "physical_query_outside_normalized_cube_or_nonfinite",
                      "invalid_query_count": int((~valid).sum().item()),
                      "first_invalid_logical_indices": torch.nonzero(~valid).flatten()[:8].cpu().tolist()}
    position = canonical * (resolution-1)
    low = position.floor().to(torch.int64)
    # At exactly coordinate1 the last valid voxel is used; the coordinate is unchanged.
    low = torch.where(low == resolution-1, low-1, low)
    frac = position-low
    bits = torch.tensor([[i&1, (i>>1)&1, (i>>2)&1] for i in range(8)], device=canonical.device)
    factors = torch.where(bits.bool(), frac[:, None, :], 1-frac[:, None, :])
    weights = factors.prod(-1)
    nodes = low[:, None, :]+bits
    indices = nodes[..., 0]*resolution**2+nodes[..., 1]*resolution+nodes[..., 2]
    derivatives = []
    for axis in range(3):
        others = [j for j in range(3) if j != axis]
        derivatives.append((2*bits[:, axis]-1)[None, :]*factors[..., others].prod(-1)*(resolution-1))
    derivative = torch.stack(derivatives, -1)
    require(bool(torch.isfinite(weights).all().item()) and
            float((weights.sum(1)-1).abs().max().item()) <= 1e-12, "Native stencil partition failure")
    planes = position == position.round()
    return (indices, weights, derivative), {"available": True, "invalid_query_count": 0,
            "grid_plane_query_count": int(planes.any(1).sum().item()),
            "grid_plane_axis_incidence_count": int(planes.sum().item()),
            "derivative_convention": "floor/right voxel on interior planes; left voxel at coordinate1; piecewise C0 value, derivative may jump"}


def actual(plan_path, contract, output, began):
    import numpy as np
    import h5py
    import torch
    sys.path.insert(0, str(ROOT))
    from geobc.continuous_query_geometry import REFERENCE_BARYCENTRIC, map_points, quadratic_shape, shape_gradient
    from geobc.solver_selection_models import build_selection_model
    from geobc.solver_selection_data import query_grid
    from geobc.benchmark import FieldScaler

    plan = read(plan_path)
    helper = load_math()
    files, rows = contract["files"], contract["scope"]["selected_rows"]
    receipt = {"input_members": {"H5": ["vertices"], "connectivity_NPZ": ["cells10"]},
               "cache_inputs": ["xyz", "features", "cloud", "mask", "grid", "query_indices", "query_weights"],
               "physical_labels": "selected original-node rows of physical.npy only, after all four predictions",
               "target_npy_read": False, "labels_read": False, "checkpoint_loads": 0,
               "model_forwards": 0, "optimizer_updates": 0, "array_writes": 0,
               "geometry_audit_repeated": False, "locator_calls": 0, "fixtures_run": 0}
    def phase(name, **extra):
        atomic_json(output/"status.json", {"status": "running", "phase": name,
                    "elapsed_seconds": time.monotonic()-began, "updated_at_utc": utc(), **extra})
    phase("loading_fixed_input_geometry")
    with h5py.File(path(files["vertices"]["path"]), "r") as handle:
        ds = handle["vertices"]
        require(ds.shape == (179316, 3) and ds.dtype == np.dtype("float32"), "Corrupt vertices schema")
        vertices = ds[...]
    with np.load(path(files["cells10"]["path"]), allow_pickle=False) as archive:
        cells = archive["cells10"]
    require(cells.shape == (111754, 10) and cells.dtype == np.dtype("int64"), "Corrupt cells10 schema")
    for name, array in (("vertices_float32_le_Corder_sha256", vertices), ("cells10_int64_le_Corder_sha256", cells)):
        require(hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() == plan["array_hashes"][name], "Original geometry/node order SHA differs")
    chosen = cells[rows]
    require(int(chosen.min()) >= 0 and int(chosen.max()) < len(vertices)
            and bool(np.all(np.diff(np.sort(chosen, axis=1), axis=1) != 0)), "Corrupt selected node IDs")
    node_ids = np.unique(chosen)
    local_cells = np.searchsorted(node_ids, chosen)
    X = vertices[chosen].astype(np.float64)
    require(bool(np.isfinite(X).all()), "Corrupt nonfinite physical geometry")
    length = plan["normalization"]["characteristic_length_mm"]
    origin = np.asarray(plan["normalization"]["bbox_min_mm"], np.float64)
    bary = helper.quadrature(np)
    D = shape_gradient(bary)
    prior = read(plan["parents"]["geometry_result"]["path"])
    orientation = {str(r["original_row"]): r["orientation_sign"] for r in prior["geometry_gate"]["regularity_selected_cells_11_samples"]}
    J, regularity, failures = helper.gauss_geometry(np, D, X, rows, length, orientation)
    checks = {"all128_Gauss_J_valid": not failures, "Gauss_invalid_count": len(failures),
              "first8_Gauss_failures": failures[:8],
              "min_abs_det_over_L3": min((abs(p["det_J_over_L3"]) for r in regularity for p in r["points"] if p["det_J_over_L3"] is not None), default=None),
              "min_sigma_min_over_L": min((p["sigma_min_J_over_L"] for r in regularity for p in r["points"] if p["sigma_min_J_over_L"] is not None), default=None),
              "max_condition": max((p["condition"] for r in regularity for p in r["points"] if p["condition"] is not None), default=None)}
    corner = (X[:, 1:4]-X[:, :1]).transpose(0, 2, 1)/length
    corner_sv, corner_det = np.linalg.svd(corner, compute_uv=False), np.linalg.det(corner)
    corner_valid = ((np.abs(corner_det) > 1e-18) & (corner_sv[:, -1] > 1e-12)
                    & (corner_sv[:, 0]/corner_sv[:, -1] < 1e12))
    checks["all32_corner_P1_matrices_regular"] = bool(corner_valid.all())
    if failures or not corner_valid.all():
        return {"status": "completed", "scientific_outcome": "negative_invalid_Gauss_geometry", "results": None,
                "checks": checks, "receipt": receipt, "counts": {"logical_queries": 504, "executed_model_queries": 0},
                "unexecuted_reason": "No invalid Gauss/corner-P1 geometry is skipped; no checkpoint/model/GT processing"}
    weights = np.abs(np.linalg.det(J))/24
    query_rows, query_bary = [], []
    for ri in range(32):
        query_rows.extend([ri]*10); query_bary.extend(REFERENCE_BARYCENTRIC.tolist())
    for ri in range(32):
        query_rows.extend([ri]*4); query_bary.extend(bary.tolist())
    face_pairs = []
    for face in contract["scope"]["shared_faces"]:
        pair = []
        for row in face["cell_rows"]:
            ri = rows.index(row)
            positions = [list(cells[row, :4]).index(corner) for corner in face["corner_ids"]]
            values = np.zeros((7, 4), np.float64)
            values[:, positions] = np.asarray(contract["scope"]["face_barycentric"])
            pair.append((ri, values))
        face_pairs.append(pair)
        query_rows.extend([pair[0][0]]*7); query_bary.extend(pair[0][1].tolist())
    query_rows, query_bary = np.asarray(query_rows), np.asarray(query_bary, np.float64)
    require(len(query_rows) == 504, "Frozen query count differs")
    physical_query = map_points(X[query_rows], query_bary)
    canonical = (physical_query-origin)/length
    Qshape = quadratic_shape(query_bary)
    def mmap(name, shape, dtype):
        array = np.load(path(files[name]["path"]), mmap_mode="r", allow_pickle=False)
        require(array.shape == shape and array.dtype == np.dtype(dtype), f"Corrupt cache schema: {name}")
        return array
    n = contract["scope"]["nodes"]
    cachedxyz = np.array(mmap("xyz", (n, 3), "float32")[node_ids], copy=True)
    features = np.array(mmap("features", (n, 25), "float32")[node_ids], copy=True)
    cloud = np.array(mmap("cloud", (1024, 3), "float32"), copy=True)
    mask = np.array(mmap("mask", (1024,), "bool"), copy=True)
    indices = np.array(mmap("query_indices", (n, 8), "int32")[node_ids], copy=True)
    nodeweights = np.array(mmap("query_weights", (n, 8), "float32")[node_ids], copy=True)
    require(all(np.isfinite(a).all() for a in (cachedxyz, features, cloud, nodeweights))
            and mask.any() and indices.min() >= 0 and indices.max() < 64**3
            and np.max(np.abs(nodeweights.sum(1)-1)) <= 2e-6, "Corrupt cached model inputs/stencil")
    checks["cached_xyz_vs_direct_physical_normalization_max_abs"] = float(np.max(np.abs(cachedxyz-(vertices[node_ids].astype(np.float64)-origin)/length)))
    require(torch.cuda.is_available(), "cuda:0 unavailable")
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    tensor = lambda a, dtype=torch.float32: torch.as_tensor(a, dtype=dtype, device=device)
    gpu_xyz, gpu_features = tensor(cachedxyz), tensor(features)
    gpu_cloud, gpu_mask = tensor(cloud).unsqueeze(0), tensor(mask, torch.bool).unsqueeze(0)
    nodecase = {"query_indices": tensor(indices, torch.int64), "query_weights": tensor(nodeweights)}
    stencil, native_receipt = native_stencil(torch, tensor(canonical, torch.float64), 64)
    scaler = FieldScaler(**plan["scaler"])
    predictions, native_outputs, route_costs, results = {}, {}, {}, []
    grid = tensor(np.array(mmap("grid", (29, 64, 64, 64), "float32"), copy=True)).unsqueeze(0)
    require(bool(torch.isfinite(grid).all().item()), "Nonfinite cached grid")
    torch.cuda.synchronize(device)
    input_geometry_stencil_seconds = time.monotonic()-began
    for route in plan["routes"]:
        route_id = route["route_id"]
        phase("predicting_fixed_TRAIN_inputs", route_id=route_id)
        started = time.monotonic()
        saved = torch.load(path(route["checkpoint"]), map_location="cpu", weights_only=False)
        require(saved["step"] == route["selected_step"] and digest(saved["config"]) == route["checkpoint_config_sha256"], "Actual checkpoint origin differs")
        receipt["checkpoint_loads"] += 1
        model = build_selection_model(route["method"], device=device, seed=route["seed"])
        require(model.source_config == route["expected_model_source_config"] and model.parameter_counts == route["parameter_counts"], "Actual model/source configuration differs")
        model.load_state_dict(saved["model"]); del saved
        model.eval(); torch.cuda.synchronize(device)
        load_seconds = time.monotonic()-started
        started = time.monotonic()
        with torch.no_grad():
            context = model.encode_geometry(gpu_cloud, gpu_mask)
            if route["method"] == "fno":
                prediction = model.forward_grid(grid, context)
                encoded = query_grid(prediction, nodecase).cpu().numpy()
                if stencil is not None:
                    si, sw, sd = stencil
                    sampled = prediction[0].flatten(1).T[si].to(torch.float64)
                    native_encoded = (sampled*sw[..., None]).sum(1)
                    derivative = torch.einsum("qkc,qkr->qcr", sampled[..., :3], sd)
                    derivative *= tensor(np.asarray(scaler.std[:3]), torch.float64)[None, :, None]/length
                    native_outputs[route_id] = (native_encoded.cpu().numpy(), derivative.cpu().numpy())
                del prediction
            else:
                encoded = model.forward_points(gpu_xyz, gpu_features, context).cpu().numpy()
        torch.cuda.synchronize(device)
        route_costs[route_id] = {"checkpoint_model_load_seconds": load_seconds,
                               "encode_forward_query_transfer_seconds": time.monotonic()-started,
                               "queried_unique_original_nodes": len(node_ids),
                               "native_logical_queries": 504 if route["method"] == "fno" and stencil is not None else 0}
        predictions[route_id] = encoded
        receipt["model_forwards"] += 1
        del model, context
    phase("reading_bounded_node_GT_after_all_predictions")
    truth_nodes = np.array(mmap("physical", (n, 4), "float32")[node_ids], dtype=np.float64, copy=True)
    receipt["labels_read"] = True
    receipt["physical_label_unique_nodes"] = len(node_ids)
    require(bool(np.isfinite(truth_nodes).all()), "Corrupt nonfinite physical labels")
    truth_cells = truth_nodes[local_cells]
    truth_kin_list = [helper.kinematics(np, D, X[i], truth_cells[i, :, :3]) for i in range(32)]
    truth_kin = {k: np.stack([value[k] for value in truth_kin_list])
                 for k in ("G1", "G2", "eps1", "eps2", "epsmid", "delta")}
    truth_values = np.einsum("qi,qic->qc", Qshape, truth_cells[query_rows])
    for route in plan["routes"]:
        route_id = route["route_id"]
        try:
            decoded = scaler.decode(predictions[route_id])
            cells_pred = decoded[local_cells]
            pred_kin_list = [helper.kinematics(np, D, X[i], cells_pred[i, :, :3]) for i in range(32)]
            pred_kin = {k: np.stack([m[k] for m in pred_kin_list]) for k in truth_kin}
            if not all(np.isfinite(v).all() for v in pred_kin.values()):
                raise FloatingPointError("Nonfinite full-cell kinematics; no point filtering")
            values = np.einsum("qi,qic->qc", Qshape, cells_pred[query_rows])
            strain = aggregate_kinematics(np, helper, weights, truth_kin, pred_kin, rows)
            guards = {"energy_closure_le_1e_10": strain["error_decomposition"]["identity_relative_residual"] <= 1e-10,
                      "direct_eps_decomposition_le_1e_10": norm_error(np, pred_kin["eps2"], pred_kin["eps1"][:, None]+pred_kin["epsmid"]) <= 1e-10}
            jumps = []
            for pair in face_pairs:
                traces = [quadratic_shape(b)@cells_pred[ri] for ri, b in pair]
                jumps.append(norm_error(np, traces[0], traces[1]))
            replay_error = norm_error(np, values[:320].reshape(32, 10, 4), cells_pred)
            guards.update(nodal_replay_le_1e_12=replay_error <= 1e-12,
                          shared_face_C0_le_1e_12=max(jumps) <= 1e-12)
            common = {"node_value_errors": value_metrics(np, decoded, truth_nodes),
                      "all504_GT_node_full10_proxy_value_errors": value_metrics(np, values, truth_values),
                      "kinematics_proxy": strain, "numeric_guards": guards,
                      "nodal_replay_normalized_max": replay_error,
                      "shared_face_normalized_max_jump": max(jumps),
                      "reconstruction": "physical decoded nodal all10 signed interpolation; structural C0 is interface sanity"}
            native = {"status": "notimplemented", "reason": "MLP original25 offnode continuation undefined"}
            if route["method"] == "fno":
                native = {"status": "unavailable", **native_receipt}
                if route_id in native_outputs:
                    ne, derivative = native_outputs[route_id]
                    if not np.isfinite(derivative).all():
                        raise FloatingPointError("Nonfinite native derivative; no point filtering")
                    native_values = scaler.decode(ne)
                    native_eps = helper.symmetric(derivative[320:448].reshape(32, 4, 3, 3))
                    reference = helper.energy(np, weights, truth_kin["eps2"])
                    error = helper.energy(np, weights, native_eps-truth_kin["eps2"])
                    worst = [(helper.ratio(helper.energy(np, weights[i], native_eps[i]-truth_kin["eps2"][i]),
                                           helper.energy(np, weights[i], truth_kin["eps2"][i]))["value"], rows[i]) for i in range(32)]
                    defined = [(v, row) for v, row in worst if v is not None]
                    native.update(status="completed", all504_GT_node_full10_proxy_value_errors=value_metrics(np, native_values, truth_values),
                        node_replay_vs_cached_query=value_metrics(np, native_values[:320], cells_pred.reshape(320, 4)),
                        native_vs_common_full10_value=value_metrics(np, native_values, values),
                        P2_reference_strain_proxy={"truth_reference_energy_mm3": reference, "error_energy_mm3": error,
                            "relative_L2": helper.ratio(error, reference),
                            "error_Frobenius_RMS": math.sqrt(error/float(weights.sum())),
                            "worst_cell": {"relative_L2": max(defined)[0] if defined else None,
                                           "original_row": max(defined)[1] if defined else None,
                                           "zero_reference_cells": 32-len(defined)}},
                        native_vs_common_P2_strain_proxy_RMS=float(math.sqrt(helper.energy(np, weights, native_eps-pred_kin["eps2"])/weights.sum())))
            results.append({"route_id": route_id, "method": route["method"], "seed": route["seed"], "step": route["selected_step"],
                            "status": "completed" if all(guards.values()) else "scientific_negative_numeric_guard", "common_nodal_full10": common, "native": native,
                            "costs": route_costs[route_id], "parameter_counts": route["parameter_counts"]})
        except FloatingPointError as error:
            results.append({"route_id": route_id, "method": route["method"], "seed": route["seed"], "step": route["selected_step"],
                            "status": "scientific_negative_nonfinite_prediction", "common_nodal_full10": None,
                            "native": None, "reason": str(error), "costs": route_costs[route_id]})
    torch.cuda.synchronize(device)
    require(request_contract(plan_path) == contract, "Sources/metadata/stat changed during execution")
    checks["all_routes_numeric_guards_passed"] = all(r["status"] == "completed" for r in results)
    return {"status": "completed", "scientific_outcome": "bounded_capability_measurement" if checks["all_routes_numeric_guards_passed"] else "scientific_negative_numeric_guard_or_nonfinite", "results": results,
            "checks": checks, "counts": {"logical_queries": 504, "node_incidences": 320, "Gauss_points": 128,
            "shared_face_points": 56, "unique_original_nodes": len(node_ids),
            "unique_physical_queries_exact_float64": len(np.unique(physical_query, axis=0)),
            "midside_RMS_local_incidences": 192}, "receipt": receipt,
            "native_FNO_stencil": native_receipt,
            "costs": {"shared_input_geometry_stencil_seconds": input_geometry_stencil_seconds,
                      "checkpoint_model_load_seconds": sum(v["checkpoint_model_load_seconds"] for v in route_costs.values()),
                      "encode_forward_query_transfer_seconds": sum(v["encode_forward_query_transfer_seconds"] for v in route_costs.values())},
            "memory": {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                       "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))},
            "limits": ["One fixed TRAIN case,32 cells,8 faces; no final/wholeCAD continuous/strain truth claim",
                       "P1/P2/node interpolations are GT-node-derived proxies, not independent offnode or strain GT",
                       "FNO native piecewise trilinear value C0; voxel-face derivative discontinuities allowed",
                       "MLP native offnode original25 feature extension notimplemented;25 roles are not declared Dirichlet",
                       "No solver/checkpoint selection, training, input reencoding, BC repair or causal attribution"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--output")
    parser.add_argument("--source-check", action="store_true")
    args = parser.parse_args()
    if args.source_check:
        require(args.output is None, "Source check has no actual output directory")
        return source_check(args.plan)
    require(args.output is not None, "Actual run requires --output")
    output = path(args.output); output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    try:
        contract = request_contract(args.plan)
        config = {"kind": "solver_continuous_capability", "contract": contract, "plan": args.plan}
        if (output/"summary.json").exists():
            previous = read(output/"summary.json")
            require(previous.get("status") == "completed" and previous.get("config") == config, "Completed reuse contract mismatch")
            print(json.dumps({"status": "completed_reuse", "elapsed_seconds": previous["elapsed_seconds"], "counts": previous["counts"]}))
            return 0
        if (output/"config.json").exists():
            require(read(output/"config.json") == config, "Pending output contract mismatch")
        atomic_json(output/"config.json", config)
        result = actual(args.plan, contract, output, began)
        result.update(kind="solver_continuous_capability", config=config, elapsed_seconds=time.monotonic()-began, completed_at_utc=utc())
        atomic_json(output/"summary.json", result)
        atomic_json(output/"status.json", {"status": "completed", "elapsed_seconds": result["elapsed_seconds"],
                    "scientific_outcome": result["scientific_outcome"], "updated_at_utc": utc()})
        compact = {"status": "completed", "elapsed_seconds": result["elapsed_seconds"], "counts": result["counts"],
                   "P2_relative_L2": {r["route_id"]: r["common_nodal_full10"]["kinematics_proxy"]["P2"]["relative_L2_to_truth"]["value"]
                    for r in result.get("results") or [] if r.get("common_nodal_full10")}}
        print(json.dumps(compact, allow_nan=False))
        return 0
    except Exception as error:
        failure = {"status": "failed", "error_type": type(error).__name__, "reason": str(error),
                   "elapsed_seconds": time.monotonic()-began, "updated_at_utc": utc()}
        atomic_json(output/"status.json", failure)
        print(json.dumps(failure, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
