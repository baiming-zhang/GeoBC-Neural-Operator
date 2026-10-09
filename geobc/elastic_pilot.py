"""One fixed TRAIN/ver approximate TET10 elastic operator and bounded PCG.

Import-time/request_contract work is standard-library metadata only.  Root
releases the actual operator preparation and solve separately.  No global or
cell stiffness matrix, target-based stop, stress label, model or optimizer is
used.  Four-point curved-P2 integration and weighted-LS RBE3 are explicit
approximations, not an exact replay of the original commercial solver.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_elastic_pilot_plan.json"
SOURCE = "geobc/elastic_pilot.py"
CHUNK = 8192
EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


def host_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path):
    return json.loads(host_path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_sha(path):
    return hashlib.sha256(host_path(path).read_bytes()).hexdigest()


def stat_record(path):
    info = host_path(path).stat()
    return {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def write_json(path, value):
    path = host_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _rules(plan):
    if (plan.get("kind") != "elastic_physical_pilot" or plan.get("scientific_rules_frozen") is not True
            or plan.get("case") != "101_479/ver" or plan.get("cad") != "101_479"):
        raise ValueError("Only the fixed, scientifically frozen TRAIN101_479/ver physical pilot is supported")
    recipe = plan["recipe"]
    fixed = {"device": "cuda:0", "dtype": "float64", "seed": 2026, "node_dofs": 3,
             "cell_chunk": CHUNK, "quadrature_points_per_cell": 4}
    if any(recipe.get(key) != value for key, value in fixed.items()):
        raise ValueError("Frozen elastic geometry/device/quadrature scope differs")
    solve = recipe["solve"]
    expected = {"method": "PCG", "initial_guess": "zero", "maximum_iterations": 2000,
                "maximum_cumulative_solve_wall_seconds": 600, "true_relative_residual_tolerance": 1e-8,
                "true_residual_interval": 25, "checkpoint_interval": 25, "resume_budget_reset": False}
    if any(solve.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen PCG budget or residual policy differs")
    operator = recipe["input_operator"]
    if (operator["maximum_combined_data_operator_wall_seconds"] != 300
            or operator["maximum_guard_K_applications"] != 8
            or operator["relative_algebra_guard_tolerance"] != 1e-9
            or operator["all_inputs_target_free"] is not True
            or operator["repeat_setup_or_guards_in_solve"] is not False):
        raise ValueError("Frozen input/operator guard policy differs")
    if (plan["post_solve_label_members"] != {"canonical_npz": ["displacement"]}
            or plan["recipe"]["post_solve"]["stress_scalar_comparison"] is not False
            or plan["expected_geometry"]["nodes"] != 179316
            or plan["expected_geometry"]["cells"] != 111754):
        raise ValueError("Only full original nodes and this one deferred displacement label are allowed")


def request_contract(plan_path=PLAN):
    """(contract, plan), strictly small JSON/source/stat metadata; no arrays."""
    from .elastic_pilot_data import request_contract as input_contract
    inherited, plan = input_contract(plan_path)
    _rules(plan)
    science = {key: plan[key] for key in ("kind", "case", "cad", "sources", "input_members",
               "post_solve_label_members", "expected_geometry", "recipe", "data_output", "operator_output", "solve_output")}
    return {"schema": "geobc_elastic_operator_solve_v1", "case": "101_479/ver",
            "science_sha256": digest(science), "science": science,
            "input_contract": inherited, "source": SOURCE, "source_sha256": source_sha(SOURCE),
            "shape_formula_source": "geobc/continuous_query_geometry.py",
            "shape_formula_source_sha256": source_sha("geobc/continuous_query_geometry.py"),
            "array_access_in_request_contract": 0, "training_updates": 0, "scaler_fits": 0}, plan


def _released(plan):
    if plan.get("actual_execution_released") is not True:
        raise ValueError("Root actual-execution release is required before scientific imports/input arrays/CUDA")


class ScientificNegative(RuntimeError):
    """Reached numerical/geometry/budget negative; no automatic repair."""


def _sync(torch):
    torch.cuda.synchronize()


def _save_array(np, path, tensor):
    path = host_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, tensor.detach().cpu().numpy(), allow_pickle=False)
    os.replace(temporary, path)
    return {"path": str(path), **stat_record(path), "shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def _load_array(np, torch, record, device):
    if stat_record(record["path"]) != {key: record[key] for key in ("bytes", "mtime_ns")}:
        raise ValueError("Committed input-only operator array stat changed")
    array = np.load(host_path(record["path"]), allow_pickle=False)
    result = torch.as_tensor(array, device=device)
    if list(result.shape) != record["shape"] or str(result.dtype) != record["dtype"]:
        raise ValueError("Committed operator array shape/dtype differs")
    return result


def _torch_shape_gradient(torch, bary, device):
    """Exact source-equivalent dN/dr in VTK TET10 edge order, FP64."""
    bary = torch.as_tensor(bary, dtype=torch.float64, device=device)
    derivative = torch.tensor([[-1., -1., -1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                              dtype=torch.float64, device=device)
    edges = torch.tensor(EDGES, dtype=torch.int64, device=device)
    corner = (4 * bary - 1)[..., :, None] * derivative
    mid = 4 * (bary[..., edges[:, 1], None] * derivative[edges[:, 0]]
               + bary[..., edges[:, 0], None] * derivative[edges[:, 1]])
    return torch.cat((corner, mid), dim=-2)


def _quadrature(torch, device):
    a, b = (5 + 3 * math.sqrt(5)) / 20, (5 - math.sqrt(5)) / 20
    bary = [[a if i == j else b for i in range(4)] for j in range(4)]
    return _torch_shape_gradient(torch, bary, device), _torch_shape_gradient(torch, [[.25] * 4], device)


def _deadline(torch, began, prior_seconds, maximum=300):
    _sync(torch)
    if prior_seconds + time.monotonic() - began > maximum:
        raise ScientificNegative("Combined data/operator setup wall exceeded the frozen300s budget; no method/budget change")


def _validate_inputs(torch, inputs, plan):
    n, t = plan["expected_geometry"]["nodes"], plan["expected_geometry"]["cells"]
    x, cells, free, fixed, force = (inputs[name] for name in ("vertices_mm", "cells10", "free_node_ids", "fixed_node_ids", "force"))
    if (x.shape != (n, 3) or cells.shape != (t, 10) or force.shape != (n, 3)
            or x.dtype != torch.float64 or force.dtype != torch.float64 or cells.dtype != torch.int64
            or free.ndim != 1 or fixed.ndim != 1 or free.dtype != torch.int64 or fixed.dtype != torch.int64
            or inputs["lame"].shape != (2,) or inputs["lame"].dtype != torch.float64):
        raise ValueError("Actual elastic input shape/dtype differs from frozen schema")
    if not torch.isfinite(x).all() or not torch.isfinite(force).all() or not torch.isfinite(inputs["lame"]).all():
        raise ScientificNegative("Nonfinite geometry/material/force input")
    lam, mu = (float(value) for value in inputs["lame"].tolist())
    if mu <= 0 or 3 * lam + 2 * mu <= 0:
        raise ScientificNegative("Nonpositive isotropic elastic material energy")
    # The producer proves actual deck-derived zero constraints, coverage and mapping.
    # These are schema guards, never role flags or an independently guessed T.
    if free.numel() == 0 or inputs["derived_node_ids"].ndim != 1:
        raise ScientificNegative("No supported free identity DOFs or malformed output-only reference map")
    return lam, mu


class MatrixFreeElastic:
    def __init__(self, torch, cells, gradients, weights, node_count, lam, mu):
        self.torch, self.cells, self.gradients, self.weights = torch, cells, gradients, weights
        self.node_count, self.lam, self.mu, self.calls = node_count, lam, mu, 0
        self.apply_seconds = 0.

    def apply(self, displacement):
        torch = self.torch
        _sync(torch)
        began = time.monotonic()
        result = torch.zeros_like(displacement)
        eye = torch.eye(3, dtype=torch.float64, device=displacement.device)
        for start in range(0, len(self.cells), CHUNK):
            stop = min(start + CHUNK, len(self.cells))
            cells, gradient = self.cells[start:stop], self.gradients[start:stop]
            derivative = torch.einsum("nia,nqib->nqab", displacement[cells], gradient)
            strain = .5 * (derivative + derivative.transpose(-1, -2))
            trace = strain.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            stress = 2 * self.mu * strain + self.lam * trace[..., None, None] * eye
            local = torch.einsum("nqab,nqib,nq->nia", stress, gradient, self.weights[start:stop])
            result.index_add_(0, cells.reshape(-1), local.reshape(-1, 3))
        _sync(torch)
        self.calls += 1
        self.apply_seconds += time.monotonic() - began
        return result


def _build_geometry(torch, inputs, plan, began, prior_seconds):
    x, cells = inputs["vertices_mm"], inputs["cells10"]
    device = x.device
    D, Dc = _quadrature(torch, device)
    n, t = len(x), len(cells)
    gradients = torch.empty((t, 4, 10, 3), dtype=torch.float64, device=device)
    weights = torch.empty((t, 4), dtype=torch.float64, device=device)
    diagonal = torch.zeros((n, 3, 3), dtype=torch.float64, device=device)
    lam, mu = _validate_inputs(torch, inputs, plan)
    length = plan["expected_geometry"]["characteristic_length_mm"]
    minimum_det, maximum_condition = math.inf, 0.
    geometry_seconds, preconditioner_seconds = 0., 0.
    eye = torch.eye(3, dtype=torch.float64, device=device)
    guard = plan["recipe"]["geometry_guard"]
    for start in range(0, t, CHUNK):
        stop = min(start + CHUNK, t)
        _sync(torch); tick = time.monotonic()
        Xe = x[cells[start:stop]]
        J = torch.einsum("nic,qir->nqcr", Xe, D)
        center = torch.einsum("nic,qir->nqcr", Xe, Dc)
        matrices = torch.cat((center, J), dim=1)
        if not torch.isfinite(matrices).all():
            raise ScientificNegative(f"Nonfinite full geometry at chunk{start}")
        normalized = matrices / length
        try:
            determinant = torch.linalg.det(normalized)
            singular = torch.linalg.svdvals(normalized)
        except torch.linalg.LinAlgError as error:
            raise ScientificNegative(f"Geometry determinant/SVD failed at chunk{start}: {error}") from error
        condition = singular[..., 0] / singular[..., -1]
        valid = (torch.isfinite(condition) & torch.isfinite(determinant)
                 & (determinant.abs() > guard["minimum_abs_detJ_over_length_cubed"])
                 & (condition < guard["maximum_J_condition"]))
        consistent = (torch.sign(determinant) == torch.sign(determinant[:, :1])).all(dim=1)
        if not valid.all() or not consistent.all():
            raise ScientificNegative(f"Full center/fourQ geometry invalid or orientation inconsistent at chunk{start}")
        minimum_det = min(minimum_det, float(determinant.abs().min()))
        maximum_condition = max(maximum_condition, float(condition.max()))
        # J[c,r], so grad_x N = J^-T grad_r N. No normalized-coordinate gradient is used.
        try:
            gradient = torch.linalg.solve(J.transpose(-1, -2), D.transpose(-1, -2)[None].expand(stop-start, -1, -1, -1)).transpose(-1, -2)
        except torch.linalg.LinAlgError as error:
            raise ScientificNegative(f"Physical gradient solve failed at chunk{start}: {error}") from error
        weight = torch.linalg.det(J).abs() / 24.
        if not torch.isfinite(gradient).all() or not torch.isfinite(weight).all() or not (weight > 0).all():
            raise ScientificNegative("Nonfinite physical gradients/integration weights")
        gradients[start:stop], weights[start:stop] = gradient, weight
        _sync(torch); geometry_seconds += time.monotonic() - tick
        tick = time.monotonic()
        gg = gradient.square().sum(dim=-1)
        local = (mu * gg[..., None, None] * eye
                 + (lam + mu) * gradient[..., :, None] * gradient[..., None, :])
        block = (weight[..., None, None, None] * local).sum(dim=1)
        diagonal.index_add_(0, cells[start:stop].reshape(-1), block.reshape(-1, 3, 3))
        _sync(torch); preconditioner_seconds += time.monotonic() - tick
        _deadline(torch, began, prior_seconds)
    tick = time.monotonic()
    free_block = diagonal[inputs["free_node_ids"]]
    chol, info = torch.linalg.cholesky_ex(free_block, check_errors=False)
    if (info != 0).any() or not torch.isfinite(chol).all():
        raise ScientificNegative("Reduced3x3 block Jacobi Cholesky failed; no shift/fallback")
    _sync(torch); preconditioner_seconds += time.monotonic() - tick
    operator = MatrixFreeElastic(torch, cells, gradients, weights, n, lam, mu)
    return operator, chol, diagonal, {"all_cells": t, "center_points": t, "fourQ_points": t*4,
            "minimum_abs_detJ_over_length_cubed": minimum_det, "maximum_J_condition": maximum_condition,
            "orientation": "Per-cell consistent center/fourQ; consistently negative cells allowed",
            "geometry_seconds": geometry_seconds, "preconditioner_seconds": preconditioner_seconds}


def _rigid_columns(torch, x, length):
    centered = (x - x.mean(dim=0)) / length
    eye = torch.eye(3, dtype=torch.float64, device=x.device)
    return [eye[j].expand_as(x).clone() for j in range(3)] + [torch.cross(eye[j].expand_as(x), centered, dim=-1) for j in range(3)]


def _algebra_guards(torch, operator, diagonal, inputs, plan, began, prior_seconds):
    tolerance = plan["recipe"]["input_operator"]["relative_algebra_guard_tolerance"]
    vectors = _rigid_columns(torch, inputs["vertices_mm"], plan["expected_geometry"]["characteristic_length_mm"])
    rigid_records = []
    for index, vector in enumerate(vectors):
        applied = operator.apply(vector)
        scale = torch.linalg.vector_norm(torch.einsum("nij,nj->ni", diagonal, vector))
        if not torch.isfinite(applied).all() or not torch.isfinite(scale) or scale <= 0:
            raise ScientificNegative("Invalid rigid-null scale/operator")
        relative = float(torch.linalg.vector_norm(applied) / scale)
        rigid_records.append({"column": index, "relative_null": relative,
                              "denominator": float(scale), "denominator_definition": "norm(full nodal diagonal times rigid column)"})
        if relative > tolerance:
            raise ScientificNegative(f"Rigid-null relative guard failed at column{index}: {relative}")
        _deadline(torch, began, prior_seconds)
    constrained = torch.stack(vectors, dim=-1)[inputs["fixed_node_ids"]].reshape(-1, 6)
    singular = torch.linalg.svdvals(constrained)
    rank_ratio = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.
    if not math.isfinite(rank_ratio) or rank_ratio <= tolerance:
        raise ScientificNegative("Actual fixed translations do not eliminate all six physical rigid motions")
    generator = torch.Generator(device=inputs["vertices_mm"].device).manual_seed(2026)
    u = torch.zeros_like(inputs["vertices_mm"]); v = torch.zeros_like(u)
    free = inputs["free_node_ids"]
    u[free] = torch.randn((len(free), 3), dtype=torch.float64, device=u.device, generator=generator)
    v[free] = torch.randn((len(free), 3), dtype=torch.float64, device=u.device, generator=generator)
    Ku, Kv = operator.apply(u), operator.apply(v)
    denominator = torch.linalg.vector_norm(u) * torch.linalg.vector_norm(Kv) + torch.linalg.vector_norm(v) * torch.linalg.vector_norm(Ku)
    energy_u, energy_v = (u * Ku).sum(), (v * Kv).sum()
    if (not torch.isfinite(Ku).all() or not torch.isfinite(Kv).all() or not torch.isfinite(denominator)
            or denominator <= 0 or not torch.isfinite(energy_u) or not torch.isfinite(energy_v)):
        raise ScientificNegative("Nonfinite random operator algebra")
    symmetry = float(((u * Kv).sum() - (v * Ku).sum()).abs() / denominator)
    eu = float(energy_u / (torch.linalg.vector_norm(u) * torch.linalg.vector_norm(Ku)))
    ev = float(energy_v / (torch.linalg.vector_norm(v) * torch.linalg.vector_norm(Kv)))
    if symmetry > tolerance or eu < -tolerance or ev < -tolerance or energy_u <= 0 or energy_v <= 0:
        raise ScientificNegative("Reduced random symmetry/positive-energy guard failed")
    if operator.calls != 8:
        raise RuntimeError("Exactly eight full K applications are required for operator guards")
    _deadline(torch, began, prior_seconds)
    return {"passed": True, "K_applications": 8, "rigid_null": rigid_records,
            "constraint_rigid_rank": 6, "constraint_smallest_over_largest_singular_value": rank_ratio,
            "symmetry_relative": symmetry, "symmetry_denominator": float(denominator),
            "random_normalized_energies": [eu, ev], "seed": 2026,
            "scope": "Sampled full-operator/reduced random algebra; no proof of complete spectrum or engineering validity"}


def _memory(torch):
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def prepare_operator(plan_path=PLAN, output_path=None):
    contract, plan = request_contract(plan_path); _released(plan)
    output = host_path(output_path or plan["operator_output"])
    config = {"contract": contract, "stage": "elastic_input_operator"}
    previous_path = output / "summary.json"
    if (output / "config.json").is_file() and read_json(output / "config.json") != config:
        raise ValueError("Existing operator output belongs to a different immutable contract")
    if previous_path.is_file():
        previous = read_json(previous_path)
        if previous.get("status") == "completed":
            if previous.get("config") != config:
                raise ValueError("Completed operator identity differs; never overwrite a different cache")
            return previous
    output.mkdir(parents=True, exist_ok=True); write_json(output / "config.json", config)
    began = time.monotonic(); status = {"phase": "preparing_operator", "config": config}
    write_json(output / "status.json", status)
    try:
        import numpy as np
        import torch
        from .elastic_pilot_data import read_cache_metadata, load_input
        manifest = read_cache_metadata(plan_path, contract=contract["input_contract"])
        prior_seconds = float(manifest["preparation_seconds"])
        if not math.isfinite(prior_seconds) or prior_seconds < 0 or prior_seconds >= 300:
            raise ScientificNegative("Recorded input-preparation time exhausts combined300s budget")
        device = torch.device(plan["recipe"]["device"])
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        inputs = load_input(manifest, device)
        operator, chol, diagonal, geometry = _build_geometry(torch, inputs, plan, began, prior_seconds)
        guards = _algebra_guards(torch, operator, diagonal, inputs, plan, began, prior_seconds)
        arrays = {"gradients": _save_array(np, output / "physical_gradients.npy", operator.gradients),
                  "weights": _save_array(np, output / "quadrature_weights.npy", operator.weights),
                  "free_cholesky": _save_array(np, output / "free_cholesky.npy", chol)}
        _deadline(torch, began, prior_seconds)
        elapsed = time.monotonic() - began
        summary = {"status": "completed", "kind": "elastic_input_operator", "config": config,
                   "feasibility_pass": True, "numerical_state": "operator_guards_passed", "arrays": arrays,
                   "input_manifest_path": str(host_path(plan["data_output"]) / "manifest.json"),
                   "input_manifest_sha256": source_sha(host_path(plan["data_output"]) / "manifest.json"),
                   "geometry": geometry, "algebra_guards": guards, "node_count": len(inputs["vertices_mm"]),
                   "cell_count": len(inputs["cells10"]), "free_nodes": len(inputs["free_node_ids"]),
                   "fixed_nodes": len(inputs["fixed_node_ids"]), "derived_output_nodes": len(inputs["derived_node_ids"]),
                   "costs": {"data_preparation_seconds": prior_seconds, "operator_wall_seconds": elapsed,
                             "combined_data_operator_wall_seconds": prior_seconds + elapsed,
                             "geometry_seconds": geometry["geometry_seconds"], "preconditioner_seconds": geometry["preconditioner_seconds"],
                             "guard_K_seconds": operator.apply_seconds, "guard_K_count": operator.calls},
                   "memory": _memory(torch), "read_receipt": {"displacement_reads": 0, "stress_reads": 0,
                      "model_or_weight_reads": 0, "optimizer_steps": 0, "scaler_fits": 0},
                   "scope": "Pure input-only approximate fourQ TET10 operator; solve reuses gradients/cholesky without rebuilding guards"}
        write_json(previous_path, summary); write_json(output / "status.json", {"phase": "completed", "feasibility_pass": True})
        return summary
    except ScientificNegative as error:
        summary = {"status": "completed", "kind": "elastic_input_operator", "config": config,
                   "feasibility_pass": False, "numerical_state": "scientifically_infeasible", "reason": str(error),
                   "elapsed_seconds": time.monotonic()-began, "displacement_reads": 0, "stress_reads": 0}
        write_json(previous_path, summary); write_json(output / "status.json", {"phase": "completed", "feasibility_pass": False, "reason": str(error)})
        return summary
    except Exception as error:
        write_json(previous_path, {"status": "failed", "kind": "elastic_input_operator", "config": config,
                                 "error": f"{type(error).__name__}: {error}", "elapsed_seconds": time.monotonic()-began})
        write_json(output / "status.json", {"phase": "failed", "error": f"{type(error).__name__}: {error}"})
        raise


def _operator_metadata(contract, plan):
    path = host_path(plan["operator_output"]) / "summary.json"
    summary = read_json(path)
    if (summary.get("status") != "completed" or summary.get("feasibility_pass") is not True
            or summary.get("config") != {"contract": contract, "stage": "elastic_input_operator"}
            or summary.get("algebra_guards", {}).get("K_applications") != 8):
        raise ValueError("Solve requires the strict completed input operator; never rebuild/repair it")
    if source_sha(summary["input_manifest_path"]) != summary["input_manifest_sha256"]:
        raise ValueError("Input manifest changed after operator construction")
    for record in summary["arrays"].values():
        if stat_record(record["path"]) != {key: record[key] for key in ("bytes", "mtime_ns")}:
            raise ValueError("Input operator cache stat changed")
    return summary


def _restore_full(torch, z, inputs):
    result = torch.zeros_like(inputs["vertices_mm"])
    result[inputs["free_node_ids"]] = z
    nodes = inputs["derived_node_ids"]
    for index in range(len(nodes)):
        start, stop = (int(value) for value in inputs["derived_rowptr"][index:index+2].tolist())
        sources = inputs["derived_source_ids"][start:stop]
        result[nodes[index]] = torch.einsum("lij,lj->i", inputs["derived_blocks"][start:stop], result[sources])
    return result


def _relative(torch, residual, norm_b):
    value = float(torch.linalg.vector_norm(residual))
    if norm_b == 0:
        return 0. if value == 0 else None
    return value / norm_b


def _save_pcg(np, output, state, z, r, p):
    filename = f"pcg_state_{state['iteration']:06d}.npz"
    target = output / filename; temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, z=z.detach().cpu().numpy(), r=r.detach().cpu().numpy(), p=p.detach().cpu().numpy())
    os.replace(temporary, target)
    state = {**state, "vectors_path": str(target), "vectors_stat": stat_record(target)}
    write_json(output / "checkpoint.json", state)
    return state


def _metrics(np, truth, prediction, regions):
    def one(ids):
        actual, predicted = truth[ids], prediction[ids]
        difference = predicted-actual
        energy = float(np.square(actual).sum()); error = float(np.square(difference).sum())
        return {"nodes": len(actual), "u_vector_relative_l2": (math.sqrt(error/energy) if energy > 0 else (0. if error == 0 else None)) if len(actual) else None,
                "zero_reference": energy == 0, "vector_RMS_mm": math.sqrt(error/len(actual)) if len(actual) else None,
                "max_nodal_vector_error_mm": float(np.linalg.norm(difference, axis=1).max()) if len(actual) else None,
                "reference_energy_mm2": energy, "error_energy_mm2": error}
    full = one(np.arange(len(truth)))
    full["regions"] = {name: one(np.asarray(ids, dtype=np.int64)) for name, ids in regions.items()}
    return full


def solve(plan_path=PLAN, output_path=None):
    contract, plan = request_contract(plan_path); _released(plan)
    operator_record = _operator_metadata(contract, plan)  # Metadata-only negative gate precedes CUDA/array loads.
    output = host_path(output_path or plan["solve_output"])
    config = {"contract": contract, "stage": "elastic_pcg_solve",
              "operator_summary_sha256": source_sha(host_path(plan["operator_output"]) / "summary.json")}
    if (output / "config.json").is_file() and read_json(output / "config.json") != config:
        raise ValueError("Existing solve output belongs to a different immutable contract")
    if (output / "summary.json").is_file():
        previous = read_json(output / "summary.json")
        if previous.get("status") == "completed":
            if previous.get("config") != config: raise ValueError("Completed elastic solve identity differs")
            return previous
    output.mkdir(parents=True, exist_ok=True); write_json(output / "config.json", config)
    began = time.monotonic()
    try:
        import numpy as np
        import torch
        from .elastic_pilot_data import read_cache_metadata, load_input
        manifest = read_cache_metadata(plan_path, contract=contract["input_contract"])
        device = torch.device(plan["recipe"]["device"])
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        inputs = load_input(manifest, device); lam, mu = _validate_inputs(torch, inputs, plan)
        arrays = {name: _load_array(np, torch, record, device) for name, record in operator_record["arrays"].items()}
        if (arrays["gradients"].shape != (len(inputs["cells10"]), 4, 10, 3)
                or arrays["weights"].shape != (len(inputs["cells10"]), 4)
                or arrays["free_cholesky"].shape != (len(inputs["free_node_ids"]), 3, 3)
                or any(value.dtype != torch.float64 for value in arrays.values())):
            raise ValueError("Strict operator cache dimensions/dtype differ; no rebuilding")
        operator = MatrixFreeElastic(torch, inputs["cells10"], arrays["gradients"], arrays["weights"], len(inputs["vertices_mm"]), lam, mu)
        chol, free = arrays["free_cholesky"], inputs["free_node_ids"]
        b = inputs["force"][free]; norm_b = float(torch.linalg.vector_norm(b))
        setup_seconds = time.monotonic() - began
        state = {"config": config, "iteration": 0, "iterations_executed_total": 0,
                 "cumulative_solve_wall_seconds": 0., "K_applications": 0, "trace": []}
        z = torch.zeros_like(b); r = b.clone(); p = torch.zeros_like(b)
        if (output / "checkpoint.json").is_file():
            state = read_json(output / "checkpoint.json")
            if state["config"] != config or stat_record(state["vectors_path"]) != state["vectors_stat"]:
                raise ValueError("PCG resume vector/config identity differs; budget must not reset")
            with np.load(host_path(state["vectors_path"]), allow_pickle=False) as archive:
                z, r, p = (torch.as_tensor(np.array(archive[name]), device=device) for name in ("z", "r", "p"))
            if any(value.shape != b.shape or value.dtype != torch.float64 or not torch.isfinite(value).all() for value in (z, r, p)):
                raise ValueError("Malformed saved PCG vectors")
        # Carry live accounting beyond the last committed vectors: a crash/replay cannot erase spent cost/iterations.
        if (output / "status.json").is_file():
            journal = read_json(output / "status.json")
            if journal.get("config") != config: raise ValueError("Resume accounting identity differs")
            state["cumulative_solve_wall_seconds"] = max(state["cumulative_solve_wall_seconds"], journal.get("cumulative_solve_wall_seconds", 0.))
            state["iterations_executed_total"] = max(state["iterations_executed_total"], journal.get("iterations_executed_total", 0))
            state["K_applications"] = max(state["K_applications"], journal.get("K_applications", 0))
        base_seconds, base_K = state["cumulative_solve_wall_seconds"], state["K_applications"]
        _sync(torch); solve_began = time.monotonic(); preconditioner_seconds = 0.; checkpoint_seconds = 0.

        def elapsed():
            _sync(torch)
            return base_seconds + time.monotonic() - solve_began

        def reduced_apply(value):
            physical = torch.zeros_like(inputs["vertices_mm"]); physical[free] = value
            return operator.apply(physical)[free]

        def precondition(value):
            nonlocal preconditioner_seconds
            _sync(torch); tick = time.monotonic()
            result = torch.cholesky_solve(value[..., None], chol).squeeze(-1)
            _sync(torch); preconditioner_seconds += time.monotonic()-tick
            return result

        def account(phase):
            state.update(cumulative_solve_wall_seconds=elapsed(), K_applications=base_K + operator.calls)
            write_json(output / "status.json", {**state, "phase": phase, "trace": None})

        def save():
            nonlocal state, checkpoint_seconds
            tick = time.monotonic(); account("solving")
            state = _save_pcg(np, output, state, z, r, p)
            checkpoint_seconds += time.monotonic()-tick
            account("solving")

        true_r = b - reduced_apply(z)
        true_relative = _relative(torch, true_r, norm_b)
        if not torch.isfinite(true_r).all(): raise ScientificNegative("Nonfinite initial/resumed true residual")
        state["trace"].append({"iteration": state["iteration"], "true_relative_residual": true_relative,
                               "cumulative_solve_wall_seconds": elapsed(), "reason": "initial_or_resume_true_verification"})
        if state["iteration"] == 0:
            r = true_r.clone(); preconditioned = precondition(r); rz = float((r*preconditioned).sum()); p = preconditioned.clone()
        else:
            preconditioned = precondition(r); rz = float((r*preconditioned).sum())
        numerical_state = "converged" if true_relative is not None and true_relative <= 1e-8 else "solving"
        if numerical_state == "solving" and (not math.isfinite(rz) or rz <= 0):
            raise ScientificNegative("Initial/resumed rTMinvr is nonfinite or nonpositive")
        while numerical_state == "solving":
            if state["iterations_executed_total"] >= 2000 or elapsed() >= 600:
                numerical_state = "iteration_budget_limit" if state["iterations_executed_total"] >= 2000 else "wall_budget_limit"
                break
            state["iterations_executed_total"] += 1
            account("solving")  # Commit the attempted iteration/accounting before its potentially interrupted K call.
            Ap = reduced_apply(p); pAp = float((p*Ap).sum())
            if not math.isfinite(pAp) or pAp <= 0 or not torch.isfinite(Ap).all():
                raise ScientificNegative("PCG pTAp is nonfinite/nonpositive; no alternate solver or shift")
            alpha = rz/pAp; z = z + alpha*p; r = r-alpha*Ap
            state["iteration"] += 1
            if not torch.isfinite(z).all() or not torch.isfinite(r).all():
                raise ScientificNegative("Nonfinite PCG solution/recursive residual")
            recursive_relative = _relative(torch, r, norm_b)
            # Every25steps, and an apparent recursive convergence, use a true residual for stopping.
            verify_now = state["iteration"] % 25 == 0 or (recursive_relative is not None and recursive_relative <= 1e-8)
            if verify_now:
                true_r = b-reduced_apply(z); true_relative = _relative(torch, true_r, norm_b)
                if not torch.isfinite(true_r).all(): raise ScientificNegative("Nonfinite true residual")
                state["trace"].append({"iteration": state["iteration"], "true_relative_residual": true_relative,
                                       "recursive_relative_residual": recursive_relative, "cumulative_solve_wall_seconds": elapsed()})
                if true_relative is not None and true_relative <= 1e-8: numerical_state = "converged"
            if numerical_state == "solving":
                preconditioned = precondition(r); next_rz = float((r*preconditioned).sum())
                if not math.isfinite(next_rz) or next_rz <= 0: raise ScientificNegative("PCG rTMinvr is nonfinite/nonpositive")
                p = preconditioned+(next_rz/rz)*p; rz = next_rz
            account("solving")
            if state["iteration"] % 25 == 0: save()
        # This final true residual is charged to the same cumulative solve budget.
        true_r = b-reduced_apply(z); true_relative = _relative(torch, true_r, norm_b)
        if not torch.isfinite(true_r).all() or true_relative is None:
            raise ScientificNegative("Nonfinite/undefined endpoint true residual")
        if true_relative <= 1e-8: numerical_state = "converged"
        state["trace"].append({"iteration": state["iteration"], "true_relative_residual": true_relative,
                               "cumulative_solve_wall_seconds": elapsed(), "reason": "endpoint_true_verification"})
        save(); cumulative_seconds = elapsed(); solve_K_count = operator.calls
        solve_K_seconds = operator.apply_seconds
        # Freeze/persist the complete physical field and equilibrium before the one target read.
        verify_began = time.monotonic(); physical = _restore_full(torch, z, inputs)
        Ku = operator.apply(physical); residual = Ku-inputs["force"]
        reaction = torch.zeros_like(residual); reaction[inputs["fixed_node_ids"]] = residual[inputs["fixed_node_ids"]]
        external = inputs["force"] + reaction
        reference = inputs["reference_xyz_mm"]
        net_force = external.sum(dim=0); net_moment = torch.cross(inputs["vertices_mm"]-reference, external, dim=-1).sum(dim=0)
        if not torch.isfinite(physical).all() or not torch.isfinite(residual).all() or not torch.isfinite(net_force).all() or not torch.isfinite(net_moment).all():
            raise ScientificNegative("Nonfinite frozen endpoint displacement/equilibrium before GT")
        prediction_record = _save_array(np, output / "physical_u_mm.npy", physical)
        reaction_record = _save_array(np, output / "constraint_reaction_N.npy", reaction)
        grid_ids_record = _save_array(np, output / "source_grid_node_ids.npy", inputs["grid_node_ids"])
        equilibrium = {"true_free_residual_relative": _relative(torch, residual[free], norm_b),
                       "true_free_residual_norm_N": float(torch.linalg.vector_norm(residual[free])),
                       "constraint_reaction_sum_N": reaction.sum(dim=0).tolist(), "net_force_with_reaction_N": net_force.tolist(),
                       "net_moment_with_reaction_Nmm": net_moment.tolist(),
                       "fixed_translation_max_abs_mm": float(physical[inputs["fixed_node_ids"]].abs().max()),
                       "reference_xyz_mm": reference.tolist(), "full_K_verification_calls": 1}
        _sync(torch); verification_seconds = time.monotonic()-verify_began
        write_json(output / "endpoint_before_GT.json", {"config": config, "numerical_state": numerical_state,
            "iteration": state["iteration"], "true_relative_residual": true_relative,
            "prediction": prediction_record, "reaction": reaction_record,
            "source_grid_node_ids": grid_ids_record, "equilibrium": equilibrium})
        # The sole allowed field: signed uXYZ for this fixed TRAIN case, never stress.
        label_path = host_path(plan["sources"]["canonical_npz"])
        label_identity = contract["input_contract"]["sources"]["canonical_npz"]
        if stat_record(label_path) != {key: label_identity[key] for key in ("bytes", "mtime_ns")}:
            raise ValueError("Deferred label source stat changed")
        gt_began = time.monotonic()
        with np.load(label_path, allow_pickle=False) as archive:
            truth = np.array(archive["displacement"], dtype=np.float64)
        prediction = physical.cpu().numpy()
        if truth.shape != prediction.shape or not np.isfinite(truth).all(): raise ValueError("Deferred signed displacement shape/finite invalid")
        regions = {key: value.cpu().numpy() for key, value in inputs["regions"].items()}
        metrics = _metrics(np, truth, prediction, regions)
        gt_seconds = time.monotonic()-gt_began
        summary = {"status": "completed", "kind": "elastic_physical_solve", "config": config,
                   "numerical_state": numerical_state, "numerically_converged": numerical_state == "converged",
                   "iteration": state["iteration"], "iterations_executed_total": state["iterations_executed_total"],
                   "true_relative_residual": true_relative, "trace": state["trace"], "metrics": metrics,
                   "prediction": prediction_record, "constraint_reaction": reaction_record,
                   "source_grid_node_ids": grid_ids_record, "equilibrium": equilibrium,
                   "costs": {"input_operator": operator_record["costs"], "solve_setup_read_seconds": setup_seconds,
                             "cumulative_solve_wall_seconds": cumulative_seconds, "this_call_solve_wall_seconds": cumulative_seconds-base_seconds,
                             "this_call_K_apply_seconds": solve_K_seconds, "preconditioner_seconds": preconditioner_seconds,
                             "checkpoint_seconds": checkpoint_seconds, "endpoint_verification_seconds": verification_seconds,
                             "GT_read_and_metrics_seconds": gt_seconds, "solve_K_count_this_call": solve_K_count,
                             "solve_K_count_cumulative": base_K+solve_K_count, "verification_K_count": 1,
                             "total_this_call_wall_seconds": time.monotonic()-began, "budget_overshoot_seconds": max(0., cumulative_seconds-600)},
                   "memory": _memory(torch), "read_receipt": {"signed_displacement_member_reads": 1, "stress_reads": 0,
                     "prediction_and_equilibrium_persisted_before_GT": True, "old_final_reads": 0, "DEV_dia_reads": 0,
                     "sealed_new_test_reads": 0, "optimizer_steps": 0, "scaler_fits": 0},
                   "scope": "One TRAIN/ver approximate physical model; finite budget endpoint is not a convergence or engineering/generalization success"}
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {**state, "trace": None, "phase": "completed", "numerical_state": numerical_state})
        return summary
    except ScientificNegative as error:
        summary = {"status": "completed", "kind": "elastic_physical_solve", "config": config,
                   "numerical_state": "breakdown", "numerically_converged": False, "reason": str(error),
                   "elapsed_seconds": time.monotonic()-began, "signed_displacement_reads": 0, "stress_reads": 0}
        write_json(output / "summary.json", summary); write_json(output / "status.json", {"phase": "completed", "config": config, "numerical_state": "breakdown", "reason": str(error)})
        return summary
    except Exception as error:
        write_json(output / "summary.json", {"status": "failed", "kind": "elastic_physical_solve", "config": config,
                   "error": f"{type(error).__name__}: {error}", "elapsed_seconds": time.monotonic()-began})
        # Preserve the live accounting journal/checkpoint for a strict resume.
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, default=Path(PLAN))
    result.add_argument("--output", type=Path)
    result.add_argument("--prepare-operator", action="store_true")
    return result


def run(args):
    return prepare_operator(args.plan, args.output) if args.prepare_operator else solve(args.plan, args.output)


if __name__ == "__main__":
    run(parser().parse_args())
