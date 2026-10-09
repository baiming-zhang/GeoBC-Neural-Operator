"""One frozen TRAIN/ver full-CAD CUDA FP64 saved-field diagnostic.

Only request_contract and CLI parsing are import-time/pure-stdlib operations.
No author toy, inference, model, optimizer, checkpoint or array output path.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_fullcad_quadratic_kinematics_plan.json"
PLAN_SHA = "40b3b9b30f1de0f4afc1f94788881778af40871e784dd0f1694f86ab66579688"
OLD_TOOL = "tools/diagnose_quadratic_kinematics.py"
OLD_SHA = "b2d5ca8b3a2abbca4818aaa34f845e5034156e80b111d1e7fd3bf8f34e058865"
OUTPUT = "runs/geobc_exploration/fullcad_quadratic_kinematics_specialist101_479_ver"
RESULT = "notes/geobc_fullcad_quadratic_kinematics_result.json"
CHUNK = 8192


def bounded_helpers():
    if hashlib.sha256((ROOT / OLD_TOOL).read_bytes()).hexdigest() != OLD_SHA:
        raise ValueError("Frozen bounded tool source SHA mismatch before import")
    spec = importlib.util.spec_from_file_location("frozen_bounded_kinematics", ROOT / OLD_TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Frozen top-level imports are standard library only.
    if module.sha_file(ROOT / OLD_TOOL) != OLD_SHA:
        raise ValueError("Frozen bounded tool source SHA mismatch")
    return module


def request_contract(planPath=PLAN):
    """Strict source/stat/hash/JSON/ZIP-header metadata only; no scientific imports."""
    old = bounded_helpers()
    plan_path = Path(planPath)
    plan_path = plan_path.resolve() if plan_path.is_absolute() else old.local_path(str(plan_path))
    old.require(plan_path == old.local_path(PLAN) and old.sha_file(plan_path) == PLAN_SHA,
                "Full-CAD frozen plan path/SHA mismatch")
    plan = old.read_json(plan_path)
    scope = plan["fixed_scope"]
    old.require(scope["CAD"] == "101_479" and scope["case"] == "101_479/ver" and
                scope["endpoint"] == 10000 and scope["cells"] == 111754 and
                scope["quadrature_points_per_cell"] == 4 and scope["quadrature_points"] == 447016 and
                scope["regularity_center_points"] == 111754 and scope["GPU_device"] == "cuda:0" and
                scope["dtype"] == "float64 for geometry/strain/energies" and
                scope["maximum_chunk_cells"] == CHUNK, "Frozen full-CAD scope mismatch")
    identities = {"plan": old.identity(PLAN, PLAN_SHA)}
    for key in ("bounded_plan", "bounded_result", "geometry_result"):
        identities[key] = old.identity(plan["parent"][key], plan["parent"][key + "_sha256"])
    for relative, digest in plan["source_pins"].items():
        identities[relative] = old.identity(relative, digest)
    inherited = old.request_contract(plan["parent"]["bounded_plan"])
    previous = old.read_json(old.local_path(plan["parent"]["bounded_result"]))
    old.require(previous["status"] == "completed" and previous["gate_pass"] is True and
                previous["kinematics"]["selected_original_rows"] == old.SELECTED_ROWS and
                previous["kinematics"]["target_numerical_points_processed"] == 128,
                "Frozen bounded parity reference mismatch")
    tool_sha = old.sha_file(Path(__file__))
    ready_path = old.local_path(plan["execution"]["ready"])
    ready = old.read_json(ready_path)
    old.require(ready["status"] == "ready_for_source_review" and ready["plan_sha256"] == PLAN_SHA and
                ready["tool_sha256"] == tool_sha and ready["author_scientific_executions"] == 0 and
                ready["author_GPU_initialized"] is False and ready["author_array_reads"] == 0,
                "Full-CAD source readiness mismatch")
    identities["ready"] = old.identity(plan["execution"]["ready"], old.sha_file(ready_path))
    return {"schema_version": 1, "stage": plan["stage"], "plan_sha256": PLAN_SHA,
            "tool_sha256": tool_sha, "identities": identities, "inherited_contract": inherited,
            "fixed_scope": scope, "chunk_cells": CHUNK, "maximum_chunks": 14,
            "geometry_thresholds": inherited["geometry_thresholds"],
            "scalar_parity_relative_tolerance": 1e-10,
            "direct_math_relative_tolerance": 1e-10,
            "signed_energy_identity_relative_tolerance": 1e-10,
            "parity_scope": "All physical aggregate scalars and128 Gauss numerical geometry values; cancellation-residual diagnostics use frozen guards",
            "scientific_imports_in_request_contract": 0, "models": 0, "optimizer_updates": 0}


def predicate(torch, matrices):
    """Same normalized predicate for center, corner matrix and Gauss Jacobians."""
    finite = torch.isfinite(matrices).all(dim=-1).all(dim=-1)
    # Invalid finite-input arithmetic is a scientific geometry failure, not an SVD crash.
    safe = torch.where(finite[..., None, None], matrices,
                       torch.eye(3, dtype=matrices.dtype, device=matrices.device))
    det = torch.linalg.det(safe)
    sigma = torch.linalg.svdvals(safe)
    condition = sigma[..., 0] / sigma[..., -1]
    valid = (finite & torch.isfinite(det) & torch.isfinite(sigma).all(dim=-1) &
             torch.isfinite(condition) & (det.abs() > 1e-18) &
             (sigma[..., -1] > 1e-12) & (condition < 1e12))
    return {"valid": valid, "det": det, "sigma_min": sigma[..., -1],
            "condition": condition, "sign": torch.sign(det).to(torch.int64), "finite": finite}


def finite_number(value):
    return float(value) if math.isfinite(float(value)) else None


def geometry_pass(torch, vertices, cells, D, Dc, scale, selected):
    counts = {"center_invalid": 0, "corner_P1_invalid": 0, "Gauss_invalid": 0,
              "Gauss_center_sign_disagreement": 0}
    first = []
    subset = {}
    minima = {"minimum_abs_det_J_over_L3": math.inf, "minimum_sigma_min_J_over_L": math.inf,
              "maximum_condition": 0.0}
    for start in range(0, cells.shape[0], CHUNK):
        stop = min(start + CHUNK, cells.shape[0])
        X = vertices[cells[start:stop]]
        center = predicate(torch, torch.einsum("nic,ir->ncr", X, Dc) / scale)
        corner = predicate(torch, (X[:, 1:4] - X[:, :1]).transpose(-1, -2) / scale)
        gauss = predicate(torch, torch.einsum("nic,qir->nqcr", X, D) / scale)
        sign_mismatch = gauss["sign"] != center["sign"][:, None]
        gp_valid = gauss["valid"] & ~sign_mismatch
        counts["center_invalid"] += int((~center["valid"]).sum().item())
        counts["corner_P1_invalid"] += int((~corner["valid"]).sum().item())
        counts["Gauss_invalid"] += int((~gp_valid).sum().item())
        counts["Gauss_center_sign_disagreement"] += int(sign_mismatch.sum().item())
        # Fixed point order: center0, cornerP1matrix1, Gauss2..5; first8 ascending row/pointID.
        bad = torch.cat((~center["valid"][:, None], ~corner["valid"][:, None], ~gp_valid), dim=1)
        if len(first) < 8:
            for ci, point in bad.nonzero()[:8 - len(first)].cpu().tolist():
                data = center if point == 0 else corner if point == 1 else gauss
                qi = () if point < 2 else (point - 2,)
                idx = (ci,) + qi
                first.append({"original_row": start + ci, "point_ID": point,
                              "point_kind": "center" if point == 0 else "corner_P1_matrix" if point == 1 else "Gauss",
                              "Gauss_index": None if point < 2 else point - 2,
                              "finite": bool(data["finite"][idx].item()),
                              "det_J_over_L3": finite_number(data["det"][idx].item()),
                              "sigma_min_J_over_L": finite_number(data["sigma_min"][idx].item()),
                              "condition": finite_number(data["condition"][idx].item()),
                              "sign": int(data["sign"][idx].item()),
                              "center_orientation_sign": int(center["sign"][ci].item())})
        for data in (center, corner, gauss):
            mask = data["valid"]
            if bool(mask.any().item()):
                minima["minimum_abs_det_J_over_L3"] = min(minima["minimum_abs_det_J_over_L3"], float(data["det"][mask].abs().min().item()))
                minima["minimum_sigma_min_J_over_L"] = min(minima["minimum_sigma_min_J_over_L"], float(data["sigma_min"][mask].min().item()))
                minima["maximum_condition"] = max(minima["maximum_condition"], float(data["condition"][mask].max().item()))
        for row in selected:
            if start <= row < stop:
                ci = row - start
                subset[row] = {"original_row": row, "passed": bool(gp_valid[ci].all().item()),
                               "center_orientation_sign": int(center["sign"][ci].item()),
                               "points": [{"quadrature_index": qi, "valid": bool(gp_valid[ci, qi].item()),
                                           "det_J_over_L3": float(gauss["det"][ci, qi].item()),
                                           "sigma_min_J_over_L": float(gauss["sigma_min"][ci, qi].item()),
                                           "condition": float(gauss["condition"][ci, qi].item()),
                                           "orientation_sign": int(gauss["sign"][ci, qi].item()),
                                           "prior_orientation_sign": int(center["sign"][ci].item())} for qi in range(4)]}
    passed = not any(counts[key] for key in ("center_invalid", "corner_P1_invalid", "Gauss_invalid"))
    return {"passed": passed, "counts": counts, "first8_failures": first,
            "centers_checked": 111754, "corner_P1_matrices_checked": 111754, "Gauss_points_checked": 447016,
            "extrema_valid_samples_only": {key: finite_number(value) for key, value in minima.items()},
            "consistent_negative_orientation_allowed": True, "corner_sign_replaces_center": False,
            "global_curved_J_validity_proven": False}, [subset[row] for row in selected]


def sym(tensor):
    return (tensor + tensor.transpose(-1, -2)) * .5


def kinematics(torch, X, u, D):
    A = (X[:, 1:4] - X[:, :1]).transpose(-1, -2)
    U = (u[:, 1:4] - u[:, :1]).transpose(-1, -2)
    G1 = torch.linalg.solve(A.transpose(-1, -2), U.transpose(-1, -2)).transpose(-1, -2)
    J = torch.einsum("nic,qir->nqcr", X, D)
    U2 = torch.einsum("nic,qir->nqcr", u, D)
    G2 = torch.linalg.solve(J.transpose(-1, -2), U2.transpose(-1, -2)).transpose(-1, -2)
    affine = u[:, :1] + torch.matmul(X - X[:, :1], G1.transpose(-1, -2))
    delta = u - affine  # Preserve floating corner residuals; never overwrite with zero.
    Umid = torch.einsum("nic,qir->nqcr", delta, D)
    Gmid = torch.linalg.solve(J.transpose(-1, -2), Umid.transpose(-1, -2)).transpose(-1, -2)
    return {"eps1": sym(G1)[:, None], "eps2": sym(G2), "epsmid": sym(Gmid), "delta": delta}, J, A


def energy_per_cell(w, tensor):
    # Full3x3 Frobenius sums both symmetric off-diagonal entries (shear weight2).
    return (w * (tensor * tensor).sum(dim=(-2, -1))).sum(dim=-1)


def contributions(torch, w, truth, pred, A):
    values = {"volume": w.sum(dim=-1), "corner_volume": torch.linalg.det(A).abs() / 6}
    for label, data in (("truth", truth), ("prediction", pred)):
        for kind, key in (("P1", "eps1"), ("P2", "eps2")):
            values[label + "_" + kind] = energy_per_cell(w, data[key])
        values[label + "_mismatch"] = energy_per_cell(w, data["eps2"] - data["eps1"])
        values[label + "_mid_displacement_squared"] = (data["delta"][:, 4:] ** 2).sum(dim=(-2, -1))
    e1, emid, e2 = pred["eps1"] - truth["eps1"], pred["epsmid"] - truth["epsmid"], pred["eps2"] - truth["eps2"]
    values["E1"] = energy_per_cell(w, e1)
    values["Emid"] = energy_per_cell(w, emid)
    values["E2"] = energy_per_cell(w, e2)
    values["signed_cross"] = (2 * w * (e1 * emid).sum(dim=(-2, -1))).sum(dim=-1)
    denominator = values["E1"].abs() + values["Emid"].abs() + values["E2"].abs() + values["signed_cross"].abs()
    numerator = (values["E2"] - values["E1"] - values["Emid"] - values["signed_cross"]).abs()
    # Exact all-zero branch, no arbitrary denominator floor.
    closure = torch.where(denominator == 0, torch.zeros_like(denominator), numerator / denominator)
    direct = torch.zeros(w.shape[0], dtype=w.dtype, device=w.device)
    for data in (truth, pred):
        expected = data["eps1"] + data["epsmid"]
        norm = torch.maximum(torch.ones_like(direct), expected.abs().amax(dim=(-3, -2, -1)))
        direct = torch.maximum(direct, (data["eps2"] - expected).abs().amax(dim=(-3, -2, -1)) / norm)
        direct = torch.maximum(direct, data["delta"][:, :4].abs().amax(dim=(-2, -1)))
    finite = torch.ones(w.shape[0], dtype=torch.bool, device=w.device)
    for data in (truth, pred):
        for tensor in data.values():
            finite &= torch.isfinite(tensor).flatten(start_dim=1).all(dim=1)
    for tensor in values.values():
        finite &= torch.isfinite(tensor)
    finite &= torch.isfinite(direct) & torch.isfinite(closure)
    return values, {"direct": direct, "closure": closure, "finite": finite}


def add_totals(totals, values, mask=None):
    for key, tensor in values.items():
        value = tensor if mask is None else tensor[mask]
        totals[key] = totals.get(key, 0.0) + float(value.sum().item())


def finalized_metrics(old, totals, cells):
    volume = totals["volume"]
    denominator = abs(totals["E1"]) + abs(totals["Emid"]) + abs(totals["E2"]) + abs(totals["signed_cross"])
    closure = 0.0 if denominator == 0 else abs(totals["E2"] - totals["E1"] - totals["Emid"] - totals["signed_cross"]) / denominator
    result = {"error_decomposition": {key: totals[key] for key in ("E1", "Emid", "signed_cross", "E2")}}
    result["error_decomposition"]["identity_relative_residual"] = closure
    for kind, error in (("P1", "E1"), ("P2", "E2")):
        truth, prediction = totals["truth_" + kind], totals["prediction_" + kind]
        result[kind] = {"truth_reference_energy_mm3": truth, "prediction_reference_energy_mm3": prediction,
                        "error_energy_mm3": totals[error], "truth_reference_Frobenius_RMS": math.sqrt(truth / volume),
                        "prediction_reference_Frobenius_RMS": math.sqrt(prediction / volume),
                        "error_Frobenius_RMS": math.sqrt(totals[error] / volume),
                        "relative_L2_to_truth": old.ratio(totals[error], truth)}
    for label in ("truth", "prediction"):
        mismatch = totals[label + "_mismatch"]
        result[label + "_P2_minus_P1"] = {"energy_mm3": mismatch, "Frobenius_RMS": math.sqrt(mismatch / volume),
                                          "relative_L2_to_same_P2": old.ratio(mismatch, totals[label + "_P2"])}
        result[label + "_midside_residual_displacement_vector_RMS_mm"] = math.sqrt(totals[label + "_mid_displacement_squared"] / (6 * cells))
    return result


def scalar_parity(actual, expected):
    comparisons, failures = [], []
    def walk(a, b, path):
        if isinstance(b, dict):
            if not isinstance(a, dict) or set(a) != set(b):
                failures.append({"path": path, "reason": "dictionary_schema_mismatch"})
                return
            for key in b:
                if key != "identity_relative_residual":
                    walk(a[key], b[key], path + "." + key)
        elif isinstance(b, list):
            if not isinstance(a, list) or len(a) != len(b):
                failures.append({"path": path, "reason": "list_schema_mismatch"})
                return
            for i, (aa, bb) in enumerate(zip(a, b)):
                walk(aa, bb, path + f"[{i}]")
        elif isinstance(b, (int, float)) and not isinstance(b, bool):
            if b == 0 or a == 0:
                passed, relative = a == b, 0.0 if a == b else None
            else:
                relative = abs(a - b) / abs(b)
                passed = math.isfinite(relative) and relative <= 1e-10
            record = {"path": path, "GPU": a, "frozen_CPU": b, "relative_error": relative, "passed": passed}
            comparisons.append(record)
            if not passed:
                failures.append(record)
        elif a != b:
            failures.append({"path": path, "reason": "null_reason_or_discrete_value_mismatch", "GPU": a, "frozen_CPU": b})
    walk(actual, expected, "parity")
    return {"passed": not failures, "scalar_comparisons": len(comparisons),
            "maximum_relative_error": max((r["relative_error"] for r in comparisons if r["relative_error"] is not None), default=0.0),
            "failed_count": len(failures), "first8_failures": failures[:8],
            "relative_tolerance": 1e-10, "both_zero_equal_one_zero_differs": True,
            "cancellation_residuals_compared": False}


def runtime(contract, receipt):
    old = bounded_helpers()
    plan = old.read_json(old.local_path(PLAN))
    inherited_plan = old.read_json(old.local_path(plan["parent"]["bounded_plan"]))
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import h5py
    import torch
    from geobc.continuous_query_geometry import shape_gradient
    old.require(torch.cuda.is_available(), "Frozen cuda:0 device unavailable; no CPU fallback")
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    receipt.update({"CUDA_initialized": True, "backend": "PyTorch CUDA", "device": str(device),
                    "GPU_name": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
                    "dtype": "float64", "chunk_cells": CHUNK})
    inputs = inherited_plan["inputs"]
    with h5py.File(old.local_path(inputs["geometry"]["path"]), "r") as h:
        old.require(h["vertices"].shape == (179316, 3) and h["vertices"].dtype == np.dtype("float32"), "Vertices schema mismatch")
        vertices_np = h["vertices"][...]
        receipt["accessed_members"]["H5"].append("vertices")
    old.require(old.array_sha(np, vertices_np) == inputs["geometry"]["array_float32_sha256"] and
                bool(np.isfinite(vertices_np).all()), "Physical vertices identity/finite mismatch")
    with np.load(old.local_path(inputs["connectivity"]["path"]), allow_pickle=False) as z:
        cells_np = z["cells10"]
        receipt["accessed_members"]["connectivity_NPZ"].append("cells10")
    old.require(cells_np.shape == (111754, 10) and cells_np.dtype == np.dtype("int64") and
                old.array_sha(np, cells_np) == inputs["connectivity"]["array_int64_sha256"] and
                int(cells_np.min()) >= 0 and int(cells_np.max()) < 179316 and
                bool((np.diff(np.sort(cells_np, axis=1), axis=1) != 0).all()), "Connectivity schema/identity/index mismatch")
    with np.load(old.local_path(inputs["endpoint"]["path"]), allow_pickle=False) as z:
        xyz = z["xyz"]
        receipt["accessed_members"]["endpoint_NPZ"].append("xyz")
        normalization = inputs["normalization"]
        canonical = (vertices_np - np.asarray(normalization["bbox_min_mm"], dtype=np.float32)) / np.float32(normalization["characteristic_length_mm"])
        old.require(old.array_sha(np, xyz) == inputs["endpoint"]["xyz_float32_sha256"] and np.array_equal(canonical, xyz), "Physical/canonical coordinate identity mismatch")
        for key, expected in (("alpha", inputs["endpoint"]["alpha"]), ("sample_key", "101_479/ver"),
                              ("step", 10000), ("channels", inputs["endpoint"]["channels"]),
                              ("u_scale_mm", contract["inherited_contract"]["u_scale_mm"])):
            value = z[key]
            receipt["accessed_members"]["endpoint_NPZ"].append(key)
            old.require((value.tolist() if value.ndim else value.item()) == expected, f"Saved endpoint metadata mismatch: {key}")
        bary = old.quadrature(np)
        D = torch.as_tensor(shape_gradient(bary), dtype=torch.float64, device=device)
        Dc = torch.as_tensor(shape_gradient(np.full(4, .25, dtype=np.float64)), dtype=torch.float64, device=device)
        vertices = torch.as_tensor(vertices_np.astype(np.float64), device=device)
        cells = torch.as_tensor(cells_np, dtype=torch.int64, device=device)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        geometry, subset_geometry = geometry_pass(torch, vertices, cells, D, Dc,
                                                  normalization["characteristic_length_mm"], old.SELECTED_ROWS)
        end.record()
        end.synchronize()
        geometry_GPU_span = begin.elapsed_time(end) / 1000
        math_begin = None
        base = {"geometry": geometry, "checks": {"source_scope_schema_coordinate_identity": True,
                                                   "all_geometry_before_target_reads": geometry["passed"]},
                "metrics": None, "subset32": {"metrics": None, "parity": None},
                "target_quadrature_points_processed": 0, "target_cells_processed": 0,
                "scientific_failures": [], "original_rows": "0..111753 ascending exactly once"}
        def gpu_finish():
            math_span = 0.0
            if math_begin is not None:
                end.record()
                end.synchronize()
                math_span = math_begin.elapsed_time(end) / 1000
            receipt.update({"geometry_GPU_event_span_seconds": geometry_GPU_span,
                            "kinematics_GPU_event_span_seconds": math_span,
                            "GPU_event_elapsed_seconds": geometry_GPU_span + math_span,
                            "GPU_time_definition": "Sum of device event phase spans; excludes intervening CPU saved-field loading, includes phase host synchronization overhead",
                            "GPU_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                            "GPU_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
        if not geometry["passed"]:
            gpu_finish()
            return {**base, "gate_pass": False,
                    "scientific_failures": [{"reason": "fullCAD_geometry", "counts": geometry["counts"], "first8": geometry["first8_failures"]}],
                    "unexecuted_reason": "Any invalid center/corner/Gauss blocks ALL saved-field reads and target reports"}
        # ALL centers, corner matrices and447016 Gauss points passed before either label/prediction member.
        targets = {}
        for key in ("u_true_mm", "u_pred_mm"):
            value = z[key]
            receipt["accessed_members"]["endpoint_NPZ"].append(key)
            receipt["labels_read" if key == "u_true_mm" else "prediction_arrays_read"] += 1
            old.require(value.shape == (179316, 3) and value.dtype == np.dtype("float64") and
                        bool(np.isfinite(value).all()), f"Saved physical field schema/finite mismatch: {key}")
            targets[key] = torch.as_tensor(value, dtype=torch.float64, device=device)
        math_begin = torch.cuda.Event(enable_timing=True)
        math_begin.record()
    full, subset = {}, {}
    max_direct = max_closure = 0.0
    subset_max_direct = subset_max_closure = 0.0
    failures = {"nonfinite_cells": 0, "direct_math_cells": 0, "energy_identity_cells": 0}
    first = []
    chunks = 0
    with torch.no_grad():
        for start in range(0, 111754, CHUNK):
            stop = min(start + CHUNK, 111754)
            ids = cells[start:stop]
            X = vertices[ids]
            truth, J, A = kinematics(torch, X, targets["u_true_mm"][ids], D)
            pred, _, _ = kinematics(torch, X, targets["u_pred_mm"][ids], D)
            w = torch.linalg.det(J).abs() / 24
            values, guards = contributions(torch, w, truth, pred, A)
            bad_finite, bad_direct, bad_closure = ~guards["finite"], guards["direct"] > 1e-10, guards["closure"] > 1e-10
            failures["nonfinite_cells"] += int(bad_finite.sum().item())
            failures["direct_math_cells"] += int(bad_direct.sum().item())
            failures["energy_identity_cells"] += int(bad_closure.sum().item())
            if len(first) < 8:
                for ci in (bad_finite | bad_direct | bad_closure).nonzero()[:8 - len(first), 0].cpu().tolist():
                    first.append({"original_row": start + ci, "finite": bool(guards["finite"][ci].item()),
                                  "direct_block_error": finite_number(guards["direct"][ci].item()),
                                  "energy_identity_relative_residual": finite_number(guards["closure"][ci].item())})
            chunks += 1
            base["target_cells_processed"] += stop - start
            base["target_quadrature_points_processed"] += 4 * (stop - start)
            if bool(bad_finite.any().item()):
                gpu_finish()
                return {**base, "gate_pass": False, "numerical_failure_counts": failures,
                        "scientific_failures": [{"reason": "Nonfinite runtime gradient/energy", "first8": first}],
                        "unexecuted_reason": "Full numerical result withheld; no partial target metric reported"}
            max_direct = max(max_direct, float(guards["direct"].max().item()))
            max_closure = max(max_closure, float(guards["closure"].max().item()))
            add_totals(full, values)
            local_rows = [row - start for row in old.SELECTED_ROWS if start <= row < stop]
            if local_rows:
                mask = torch.as_tensor(local_rows, dtype=torch.int64, device=device)
                add_totals(subset, values, mask)
                subset_max_direct = max(subset_max_direct, float(guards["direct"][mask].max().item()))
                subset_max_closure = max(subset_max_closure, float(guards["closure"][mask].max().item()))
    full_metrics = finalized_metrics(old, full, 111754)
    subset_metrics = finalized_metrics(old, subset, 32)
    previous = old.read_json(old.local_path(plan["parent"]["bounded_result"]))["kinematics"]
    # Retain the prior point-schema exactly; center sign is the new orientation anchor.
    comparable_geometry = [{key: value for key, value in record.items() if key != "center_orientation_sign"} for record in subset_geometry]
    parity = scalar_parity({"metrics": subset_metrics, "Gauss_geometry": comparable_geometry,
                            "quadrature_volume_mm3": subset["volume"], "selected_corner_volume_mm3": subset["corner_volume"]},
                           {"metrics": previous["metrics"], "Gauss_geometry": previous["Gauss_geometry"],
                            "quadrature_volume_mm3": previous["quadrature_volume_mm3"],
                            "selected_corner_volume_mm3": previous["selected_corner_volume_mm3"]})
    # Cancellation diagnostics are guarded, not matched relative to a ~machine-zero CPU residual.
    numerical_pass = not any(failures.values()) and full_metrics["error_decomposition"]["identity_relative_residual"] <= 1e-10 and subset_metrics["error_decomposition"]["identity_relative_residual"] <= 1e-10
    base["checks"].update({"finite_physical_affine_and_signed_energy_guards": numerical_pass,
                           "unique_runtime_GPU_subset32_parity": parity["passed"]})
    passed = all(base["checks"].values())
    gpu_finish()
    return {**base, "gate_pass": passed, "metrics": full_metrics if passed else None,
            "subset32": {"selected_original_rows": old.SELECTED_ROWS, "cells": 32, "Gauss_points": 128,
                         "metrics": subset_metrics if passed else None, "parity": parity,
                         "max_direct_block_error": subset_max_direct, "max_per_cell_energy_identity_relative_residual": subset_max_closure},
            "numerical_failure_counts": failures, "first8_numerical_failures": first,
            "max_direct_block_error": max_direct, "max_per_cell_energy_identity_relative_residual": max_closure,
            "aggregate_energy_identity_relative_residual": full_metrics["error_decomposition"]["identity_relative_residual"],
            "scientific_failures": [] if passed else [{"reason": "Numerical_guard_or_GPU_parity", "numerical_counts": failures, "parity_failed_count": parity["failed_count"]}],
            "ascending_chunks": chunks, "maximum_chunk_cells": CHUNK,
            "quadrature_volume_mm3": full["volume"] if passed else None,
            "corner_volume_mm3": full["corner_volume"] if passed else None,
            "midside_residual_RMS_definition": "Uniform vector RMS over670524 local midside incidences; repeated node IDs retained",
            "energy_units": "mm^3; dimensionless full symmetric3x3 strain squared times physical quadrature volume",
            "accuracy_gate_applied": False, "historical_allnode_P1_replaced": False,
            "limits": plan["limits"]}


def execute(planPath, output):
    old = bounded_helpers()
    started = time.perf_counter()
    destination = Path(output).resolve() if Path(output).is_absolute() else old.local_path(output)
    old.require(destination == old.local_path(OUTPUT), "Only the frozen output directory is permitted")
    contract = request_contract(planPath)
    plan = old.read_json(old.local_path(PLAN))
    config_path, status_path, summary_path = destination / "config.json", destination / "status.json", destination / "summary.json"
    result_path = old.local_path(RESULT)
    if destination.exists() and any(destination.iterdir()):
        old.require(all(p.is_file() for p in (config_path, status_path, summary_path, result_path)), "Incomplete output; automatic rerun forbidden")
        previous = old.read_json(summary_path)
        old.require(previous["status"] == "completed" and old.read_json(status_path)["status"] == "completed" and
                    previous["config"]["contract"] == contract and old.read_json(config_path)["contract"] == contract and
                    old.read_json(result_path) == previous, "Strict completed reuse contract mismatch")
        print(json.dumps({"status": "completed", "reused": True, "gate_pass": previous["gate_pass"], "result": str(result_path)}))
        return 0
    old.require(not result_path.exists(), "Formal result exists; no automatic replacement")
    config = {"contract": contract, "started_at_utc": old.utc(), "geometry_gate": plan["geometry_gate"],
              "mathematics": plan["mathematics"], "parity_and_numerics": plan["parity_and_numerics"],
              "limits": plan["limits"], "no_array_writes": True}
    receipt = {"accessed_members": {"H5": [], "connectivity_NPZ": [], "endpoint_NPZ": []},
               "labels_read": 0, "prediction_arrays_read": 0, "other_labels_read": 0,
               "weights_read": 0, "model_calls": 0, "inference_calls": 0, "optimizer_updates": 0,
               "CUDA_initialized": False, "scientific_array_writes": 0, "CPU_loader_workers": 1, "BLAS_threads": 1}
    destination.mkdir(parents=True, exist_ok=True)
    old.write_json(config_path, config)
    old.write_json(status_path, {"status": "running", "started_at_utc": config["started_at_utc"]})
    try:
        gate = runtime(contract, receipt)
        old.require(request_contract(planPath) == contract, "Source identities changed during the unique runtime")
        result = {"schema_version": 1, "status": "completed", "gate_pass": gate["gate_pass"],
                  "scientific_outcome": "passed" if gate["gate_pass"] else "negative", "config": config,
                  "kinematics": gate, "receipt": receipt, "elapsed_seconds": time.perf_counter() - started,
                  "peak_RSS": old.peak_rss(), "completed_at_utc": old.utc()}
        old.write_json(summary_path, result)
        old.write_json(result_path, result)
        old.write_json(status_path, {"status": "completed", "gate_pass": result["gate_pass"],
                                     "scientific_outcome": result["scientific_outcome"], "completed_at_utc": result["completed_at_utc"]})
        print(json.dumps({"status": "completed", "gate_pass": result["gate_pass"], "scientific_outcome": result["scientific_outcome"],
                          "elapsed_seconds": result["elapsed_seconds"], "result": str(result_path)}))
        return 0
    except Exception as exc:
        failure = {"schema_version": 1, "status": "failed", "gate_pass": False, "config": config,
                   "receipt": receipt, "error_type": type(exc).__name__, "error": str(exc),
                   "elapsed_seconds": time.perf_counter() - started, "peak_RSS": old.peak_rss(), "failed_at_utc": old.utc()}
        for path in (summary_path, result_path, status_path):
            old.write_json(path, failure)
        print(json.dumps(failure, allow_nan=False))
        return 1


def preflight_failure(planPath, output, exc):
    """Persist operational/source failure without replacing any existing output."""
    failure = {"schema_version": 1, "status": "failed", "gate_pass": False,
               "phase": "purestdlib_preflight", "requested_plan": str(planPath),
               "error_type": type(exc).__name__, "error": str(exc), "failed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "receipt": {"scientific_array_reads": 0, "CUDA_initialized": False, "model_calls": 0,
                           "optimizer_updates": 0, "scientific_array_writes": 0}}
    destination = Path(output).resolve() if Path(output).is_absolute() else (ROOT / output).resolve()
    if destination == (ROOT / OUTPUT).resolve() and not (ROOT / RESULT).exists() and not (destination.exists() and any(destination.iterdir())):
        destination.mkdir(parents=True, exist_ok=True)
        records = {destination / "config.json": {"requested_plan": str(planPath), "contract": None},
                   destination / "status.json": failure, destination / "summary.json": failure, ROOT / RESULT: failure}
        for path, record in records.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            os.replace(tmp, path)
    print(json.dumps(failure, allow_nan=False))
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN)
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()
    try:
        return execute(args.plan, args.output)
    except Exception as exc:
        return preflight_failure(args.plan, args.output, exc)


if __name__ == "__main__":
    raise SystemExit(main())
