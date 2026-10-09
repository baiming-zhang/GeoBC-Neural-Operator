"""Two frozen CADs: P1-derived Galerkin additive coarse PCG, GT free.

The fine equation/curved-P2 quadrature and exact supported constraints remain
the accepted crossCAD equation. Only its numerical preconditioner changes.
All preparation/contract APIs are stdlib metadata; scientific imports occur
only after authentic root source release. A coarse polynomial always executes
32 zero-start Jacobi updates; there is no variable or warmed inner solve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_elastic_coarse_plan.json"
SOURCE = "geobc/elastic_coarse.py"
KIND = "elastic_p1_derived_coarse_screen"
CASE_KIND = "elastic_p1_derived_coarse_case"
CASES = ("101_428/ver", "102_249/ver")
SOURCES = (SOURCE, "geobc/elastic_crosscad.py", "geobc/elastic_pilot.py",
           "geobc/elastic_pilot_data.py", "geobc/continuous_query_geometry.py")
EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
SOLVE = {"maximum_iterations": 10000, "maximum_cumulative_solve_wall_seconds": 300,
         "true_relative_residual_tolerance": 1e-8, "true_residual_interval": 25,
         "checkpoint_interval": 25, "resume_budget_reset": False}
ZERO_KEYS = ("GT_reads", "displacement_reads", "stress_reads", "old_final_reads",
             "DEV_dia_reads", "sealed_new_test_reads", "optimizer_steps", "scaler_fits")


def helpers():
    from . import elastic_crosscad as cross
    from . import elastic_pilot as base
    return cross, base


def host_path(path):
    return helpers()[0].host_path(path)


def read_json(path):
    return json.loads(host_path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(host_path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_cost(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def zero_receipt():
    return {**{key: 0 for key in ZERO_KEYS}, "learned_weight_reads": 0,
            "fine_operator_rebuilds": 0, "inherited_fine_K_guards_repeated": 0, "setup_K_guards_repeated": 0,
            "numerical_reference_reads": 0}


def request_contract(plan_path=PLAN):
    """Source/small-JSON/stat only. Never call the old single-CAD contract."""
    cross, base = helpers()
    plan = read_json(plan_path)
    require(plan.get("kind") == KIND and tuple(plan.get("case_keys", ())) == CASES,
            "Frozen two-CAD coarse scope/order differs")
    recipe, comparison, gate = plan["recipe"], plan["comparison"], plan["selection_gate"]
    require(recipe["device"] == "cuda:0" and recipe["dtype"] == "float64" and recipe["seed"] == 2026
            and recipe["initial_guess"] == "zero" and recipe["assembly_cell_chunk"] == 8192
            and tuple(tuple(edge) for edge in recipe["mid_edge_order"]) == EDGES
            and recipe["coarse_inverse"]["steps"] == 32
            and recipe["coarse_inverse"]["kind"] == "fixed_linear_zero_start_damped_Jacobi"
            and recipe["coarse_inverse"]["omega"] == "1/gamma"
            and recipe["coarse_inverse"]["changing_inner_iterations"] is False,
            "Frozen P1-derived mapping/32-step coarse inverse differs")
    require(recipe["solve"] == SOLVE and recipe["maximum_coarse_setup_wall_seconds_per_case"] == 60
            and recipe["maximum_stage_wall_seconds"] == 900
            and recipe["maximum_actual_peak_GPU_bytes"] == 24 * 1024**3
            and recipe["no_geometry_or_operator_rebuild"] is True
            and recipe["no_shift_pseudoinverse_or_precision_fallback"] is True,
            "Frozen solve/budget/no-fallback policy differs")
    guards = recipe["algebra_guards"]
    require(guards["seed"] == 2026 and guards["relative_tolerance"] == 1e-9
            and guards["true_fine_K_guards_repeated"] == 0 and guards["new_coarse_guard_vectors"] == 2
            and guards["galerkin_action_vectors"] == 2,
            "Frozen one-construction coarse guard policy differs")
    require(comparison["same_solution_relative_l2_maximum"] == 1e-5
            and comparison["zero_reference_RMS_mm_maximum"] == 1e-12
            and comparison["all_original_nodes"] is True and comparison["new_GT_metrics"] is False
            and gate["inventory_denominator"] == 2 and gate["both_methods_true_converged_required"] == 2
            and gate["allN_same_solution_gate_required"] == 2
            and gate["minimum_CADs_with_at_least25percent_total_cold_reduction"] == 2
            and gate["minimum_total_cold_reduction_fraction"] == .25
            and gate["complete_cost_components_required"] is True,
            "Frozen same-solution/full-cold macro gate differs")
    require(all(plan["read_policy"].get(key) == 0 for key in ZERO_KEYS)
            and plan["read_policy"]["learned_weight_reads"] == 0,
            "Coarse stage must remain target/weight/train free")
    require(sha(plan["design"]) == plan["design_sha256"]
            and sha(plan["physical_plan"]) == plan["physical_plan_sha256"], "Pinned parent/design metadata changed")
    parent = read_json(plan["physical_plan"])
    require(parent["kind"] == "elastic_crosscad_screen" and parent["actual_execution_released"] is True,
            "Parent accepted crossCAD plan differs")
    overall = plan["physical_summary"]
    require(sha(overall["path"]) == overall["sha256"], "Pinned parent screen summary changed")
    screen = read_json(overall["path"])
    require(screen["status"] == "completed" and screen["kind"] == "elastic_crosscad_screen",
            "Completed accepted parent screen required")
    records = {}
    for key in CASES:
        frozen = plan["cases"][key]
        record = frozen["physical_record"]
        require(record["key"] == key and record["cad"] == key.split("/")[0]
                and record["recipe"]["dtype"] == "float64"
                and record["recipe"]["quadrature_points_per_cell"] == 4,
                "Accepted fine physical record differs")
        for member in ("input_manifest", "operator_summary", "base_summary"):
            require(sha(frozen[member]["path"]) == frozen[member]["sha256"], "Pinned cache/endpoint metadata changed: " + key + "/" + member)
        manifest = read_json(frozen["input_manifest"]["path"])
        operator = read_json(frozen["operator_summary"]["path"])
        reference = read_json(frozen["base_summary"]["path"])
        require(manifest["status"] == "completed" and manifest["feasibility_pass"] is True
                and manifest["contract"] == record and manifest["file_identity"] == frozen["input_manifest"]["file_identity"]
                and manifest["cache_file"] == frozen["input_manifest"]["cache_file"], "Strict accepted target-free input cache differs")
        require(operator["status"] == "completed" and operator["feasibility_pass"] is True
                and operator["config"] == {"contract": record, "stage": "elastic_crosscad_operator"}
                and operator["input_manifest_sha256"] == frozen["input_manifest"]["sha256"]
                and operator["algebra_guards"]["K_applications"] == 8
                and operator["arrays"] == frozen["operator_summary"]["arrays"],
                "Strict accepted four-Q/eight-guard fine operator cache differs")
        require(reference["status"] == "completed"
                and reference["config"] == {"contract": record, "stage": "elastic_crosscad_pcg"}
                and reference["input_compatible"] is True and reference["operator_compatible"] is True
                and reference["numerically_converged"] is True and reference["numerical_state"] == "converged"
                and finite_cost(reference["true_relative_residual"]) and reference["true_relative_residual"] <= 1e-8
                and reference["prediction"] == frozen["base_summary"]["prediction"], "Frozen zero-start true-converged numerical reference differs")
        costs = reference["costs"]
        zero = costs["cold_case_accounted_wall_seconds"] - costs["GT_read_and_metrics_seconds"]
        inherited = costs["input_preparation_seconds"] + costs["operator_wall_seconds"]
        require(finite_cost(zero) and zero > 0 and finite_cost(inherited)
                and zero == frozen["base_summary"]["zero_cold_without_GT_seconds"]
                and inherited == frozen["base_summary"]["inherited_input_operator_seconds"], "Frozen zero/fine setup costs differ")
        for value in [manifest["file_identity"], *operator["arrays"].values(), reference["prediction"]]:
            require(base.stat_record(value.get("path", manifest["cache_file"])) == {part: value[part] for part in ("bytes", "mtime_ns")},
                    "Accepted target-free array/reference stat changed")
        records[key] = frozen
    pins = {name: sha(name) for name in SOURCES}
    require(set(plan.get("source_sha256", {})) == set(SOURCES), "Exactly five source pins required")
    for name, value in plan["source_sha256"].items():
        require(value == pins[name], "Source pin differs: " + name)
    science = {"kind": KIND, "case_keys": list(CASES), "plan_science": {name: plan[name] for name in
               ("recipe", "comparison", "cost", "selection_gate", "read_policy")}, "records": records,
               "design_sha256": plan["design_sha256"], "physical_plan_sha256": plan["physical_plan_sha256"],
               "physical_summary": overall, "source_sha256": pins, "output": str(host_path(plan["output"]))}
    return {"schema": "geobc_elastic_p1_derived_coarse_v1", **science, "science_sha256": digest(science),
            "metadata_only_array_reads": 0, "base_gate": {"eligible": True, "true_converged_cases": 2}}, plan


COUNTERS = ("iterations_executed_total", "fine_K_applications", "setup_K_applications", "solve_K_applications",
            "endpoint_K_applications", "coarse_matvec_calls", "P_calls", "PT_calls", "preconditioner_calls",
            "coarse_construct_attempts", "numerical_reference_reads")
TIMES = ("cumulative_setup_read_seconds", "cumulative_coarse_setup_seconds", "cumulative_solve_wall_seconds",
         "cumulative_endpoint_seconds", "cumulative_reference_seconds", "cumulative_persistence_seconds",
         "block_preconditioner_seconds", "coarse_inverse_seconds", "transfer_seconds",
         "preconditioner_wall_seconds", "checkpoint_seconds")
PHASE_TIME = {"setup_read": "cumulative_setup_read_seconds", "coarse_setup": "cumulative_coarse_setup_seconds",
              "solving": "cumulative_solve_wall_seconds", "endpoint": "cumulative_endpoint_seconds",
              "reference": "cumulative_reference_seconds", "persistence": "cumulative_persistence_seconds"}


class CaseLedger:
    """Separate immutable vector checkpoint and maximal cumulative cost journal."""
    def __init__(self, output, config, clock, key):
        self.output, self.config, self.clock, self.key = output, config, clock, key
        self.config_sha256 = digest(config)
        self.data = {"config_sha256": self.config_sha256, **{name: 0 for name in COUNTERS},
                     **{name: 0. for name in TIMES}, "cumulative_case_wall_seconds": 0., "read_receipt": zero_receipt()}
        old = output / "ledger.json"
        if old.exists():
            self.data = read_json(old)
            require(self.data["config_sha256"] == self.config_sha256, "Coarse cost journal belongs to another immutable case")
            require(all(finite_cost(self.data.get(name)) for name in (*COUNTERS, *TIMES, "cumulative_case_wall_seconds")), "Invalid preserved coarse accounting")
        self.prior = self.data["cumulative_case_wall_seconds"]
        self.began = self.last_tick = time.monotonic()
        self.phase = "setup_read"
        self.torch = None

    def sync(self):
        if self.torch is not None:
            self.torch.cuda.synchronize()

    def update(self):
        self.sync()
        now = time.monotonic()
        self.data[PHASE_TIME[self.phase]] += now - self.last_tick
        self.last_tick = now
        self.data["cumulative_case_wall_seconds"] = self.prior + now - self.began

    def account(self, phase=None):
        self.update()
        if phase is not None:
            self.phase = phase
        self.data["active_phase"] = self.phase
        self.data["read_receipt"]["numerical_reference_reads"] = self.data["numerical_reference_reads"]
        _, base = helpers()
        base.write_json(self.output / "ledger.json", self.data)
        base.write_json(self.output / "status.json", {"config_sha256": self.config_sha256, "phase": self.phase,
            "case": self.key, "cumulative_case_wall_seconds": self.data["cumulative_case_wall_seconds"],
            "cumulative_coarse_setup_seconds": self.data["cumulative_coarse_setup_seconds"],
            "cumulative_solve_wall_seconds": self.data["cumulative_solve_wall_seconds"],
            "iterations_executed_total": self.data["iterations_executed_total"]})
        self.clock.account(self.phase, self.key)

    def check_memory(self):
        _, base = helpers()
        if self.torch is not None:
            memory = base._memory(self.torch)
            if max(memory.values()) > 24 * 1024**3:
                raise base.ScientificNegative("actual_GPU_peak_cap_24GiB_exceeded")

    def check(self, phase_budget=True):
        self.update()
        _, base = helpers()
        self.check_memory()
        if self.clock.remaining() <= 0:
            raise base.ScientificNegative("stage_wall_budget_exhausted")
        if phase_budget and self.phase == "coarse_setup" and self.data["cumulative_coarse_setup_seconds"] >= 60:
            raise base.ScientificNegative("coarse_setup_cumulative_60s_budget_exhausted")


def negative(condition, message):
    if not bool(condition):
        raise helpers()[1].ScientificNegative(message)


def make_mapping(torch, inputs, ledger):
    """Actual topology only: no coordinate midpoint or GRID-ID arithmetic guess."""
    cells, free = inputs["cells10"], inputs["free_node_ids"]
    n = len(inputs["vertices_mm"])
    device = cells.device
    corner_ids = torch.unique(cells[:, :4].reshape(-1), sorted=True)
    corner_mask = torch.zeros(n, dtype=torch.bool, device=device)
    corner_mask[corner_ids] = True
    mids = cells[:, 4:].reshape(-1)
    negative(not corner_mask[mids].any(), "Global node is both CTETRA corner and midpoint")
    edge_index = torch.tensor(EDGES, dtype=torch.int64, device=device)
    endpoints = torch.sort(cells[:, :4][:, edge_index].reshape(-1, 2), dim=1).values
    edge_keys = endpoints[:, 0] * n + endpoints[:, 1]
    order = torch.argsort(mids)
    sm, sk, se = mids[order], edge_keys[order], endpoints[order]
    negative(not ((sm[1:] == sm[:-1]) & (sk[1:] != sk[:-1])).any(), "One midpoint has multiple unordered global corner edges")
    order_edge = torch.argsort(edge_keys)
    ek, em = edge_keys[order_edge], mids[order_edge]
    negative(not ((ek[1:] == ek[:-1]) & (em[1:] != em[:-1])).any(), "One unordered corner edge has multiple midpoint IDs")
    keep = torch.ones(len(sm), dtype=torch.bool, device=device)
    keep[1:] = sm[1:] != sm[:-1]
    unique_mid, unique_ends = sm[keep], se[keep]
    free_map = torch.full((n,), -1, dtype=torch.int64, device=device)
    free_map[free] = torch.arange(len(free), dtype=torch.int64, device=device)
    coarse_corners = corner_ids[free_map[corner_ids] >= 0]
    negative(len(coarse_corners) > 0, "No actual free CTETRA corner coarse nodes")
    coarse_map = torch.full((n,), -1, dtype=torch.int64, device=device)
    coarse_map[coarse_corners] = torch.arange(len(coarse_corners), dtype=torch.int64, device=device)
    rows = [free_map[coarse_corners]]
    cols = [coarse_map[coarse_corners]]
    values = [torch.ones(len(coarse_corners), dtype=torch.float64, device=device)]
    mid_rows = free_map[unique_mid]
    endpoint_cols = coarse_map[unique_ends]
    valid = (mid_rows[:, None] >= 0) & (endpoint_cols >= 0)
    rows.append(mid_rows[:, None].expand(-1, 2)[valid])
    cols.append(endpoint_cols[valid])
    values.append(torch.full((int(valid.sum()),), .5, dtype=torch.float64, device=device))
    indices = torch.stack((torch.cat(rows), torch.cat(cols)))
    Pcoo = torch.sparse_coo_tensor(indices, torch.cat(values), (len(free), len(coarse_corners)),
                                  dtype=torch.float64, device=device).coalesce()
    P = Pcoo.to_sparse_csr()
    PT = Pcoo.transpose(0, 1).coalesce().to_sparse_csr()
    fixed_mid_free_corner = (mid_rows < 0) & (endpoint_cols >= 0).any(dim=1)
    facts = {"all_original_nodes": n, "all_cells": len(cells), "original_corner_nodes": len(corner_ids),
        "unique_midpoint_nodes": len(unique_mid), "free_corner_nodes": len(coarse_corners), "coarse_vector_dofs": 3*len(coarse_corners),
        "fine_free_nodes": len(free), "P_scalar_nnz": Pcoo._nnz(), "corner_mid_disjoint": True,
        "mid_to_unique_unordered_edge": True, "unordered_edge_to_unique_mid": True,
        "free_corner_injection_rows": len(coarse_corners), "coarse_full_column_rank_reason": "Independent actual free-corner identity rows",
        "fixed_mid_with_any_free_corner_count": int(fixed_mid_free_corner.sum()),
        "strict_nested_P1_at_fixed_rows": not bool(fixed_mid_free_corner.any()),
        "coarse_space_name": "constraint-masked P1-derived algebraic coarse",
        "global_mapping_source": "Accepted cached cells10 and actual free_node_ids; no coordinates, remeshing or GRIDID-1"}
    ledger.check(); ledger.account()
    return P, PT, coarse_corners, coarse_map, free_map, facts


def assemble_coarse(torch, inputs, arrays, coarse_map, free_map, ledger):
    """Ac=P^T A P; fixed fine-row masks are applied BEFORE local gradients."""
    cells, gradient, weight = inputs["cells10"], arrays["gradients"], arrays["weights"]
    lam, mu = (float(value) for value in inputs["lame"].tolist())
    device = cells.device
    nc = int((coarse_map >= 0).sum())
    local_P = torch.zeros((10, 4), dtype=torch.float64, device=device)
    local_P[:4] = torch.eye(4, dtype=torch.float64, device=device)
    for i, (a, b) in enumerate(EDGES):
        local_P[4+i, a] = local_P[4+i, b] = .5
    eye = torch.eye(3, dtype=torch.float64, device=device)
    components = torch.arange(3, dtype=torch.int64, device=device)
    row_chunks, col_chunks, value_chunks = [], [], []
    for start in range(0, len(cells), 8192):
        ledger.check(); ledger.account()
        stop = min(start+8192, len(cells))
        local = cells[start:stop]
        free_mask = (free_map[local] >= 0).to(torch.float64)
        gc = torch.einsum("nqia,ij,ni->nqja", gradient[start:stop], local_P, free_mask)
        dot = torch.einsum("nqia,nqja->nqij", gc, gc)
        blocks = mu*torch.einsum("nqij,ab->nqiajb", dot, eye)
        blocks = blocks + mu*torch.einsum("nqib,nqja->nqiajb", gc, gc)
        blocks = blocks + lam*torch.einsum("nqia,nqjb->nqiajb", gc, gc)
        local_K = torch.einsum("nqiajb,nq->niajb", blocks, weight[start:stop]).reshape(-1, 12, 12)
        negative(torch.isfinite(local_K).all(), "Nonfinite Galerkin local stiffness")
        corner_cols = coarse_map[local[:, :4]]
        dofs = (corner_cols[..., None]*3+components).reshape(-1, 12)
        valid_dofs = (corner_cols >= 0)[..., None].expand(-1, -1, 3).reshape(-1, 12)
        valid_pairs = valid_dofs[:, :, None] & valid_dofs[:, None, :]
        row_chunks.append(dofs[:, :, None].expand(-1, 12, 12)[valid_pairs])
        col_chunks.append(dofs[:, None, :].expand(-1, 12, 12)[valid_pairs])
        value_chunks.append(local_K[valid_pairs])
        del gc, dot, blocks, local_K, valid_pairs
        ledger.check(); ledger.account()
    indices = torch.stack((torch.cat(row_chunks), torch.cat(col_chunks)))
    values = torch.cat(value_chunks)
    negative(torch.isfinite(values).all(), "Nonfinite Galerkin COO values")
    raw_nnz = len(values)
    ledger.check(); ledger.account()
    Ac = torch.sparse_coo_tensor(indices, values, (3*nc, 3*nc), dtype=torch.float64, device=device).coalesce()
    del row_chunks, col_chunks, value_chunks, indices, values
    ledger.check(); ledger.account()
    delta = (Ac-Ac.transpose(0, 1)).coalesce()
    symmetry = float(torch.linalg.vector_norm(delta.values())/torch.linalg.vector_norm(Ac.values()))
    negative(math.isfinite(symmetry) and symmetry <= 1e-9, "Galerkin COO symmetry guard failed")
    # Roundoff symmetrization only AFTER the tolerance gate, not a matrix repair.
    Ac = (.5*(Ac+Ac.transpose(0, 1))).coalesce()
    ids, values = Ac.indices(), Ac.values()
    diagonal = torch.zeros(3*nc, dtype=torch.float64, device=device)
    diagonal_mask = ids[0] == ids[1]
    diagonal.index_add_(0, ids[0, diagonal_mask], values[diagonal_mask])
    negative(torch.isfinite(diagonal).all() and (diagonal > 0).all(), "Coarse scalar diagonal is nonpositive/nonfinite")
    scaled_abs = values.abs()/torch.sqrt(diagonal[ids[0]]*diagonal[ids[1]])
    row_sum = torch.zeros_like(diagonal)
    row_sum.index_add_(0, ids[0], scaled_abs)
    gamma = float(row_sum.max())
    negative(math.isfinite(gamma) and gamma > 0, "Nonfinite/nonpositive symmetric-scaled Gershgorin Gamma")
    result = Ac.to_sparse_csr()
    facts = {"raw_COO_entries": raw_nnz, "coalesced_CSR_nnz": Ac._nnz(), "vector_shape": [3*nc, 3*nc],
        "raw_COO_symmetry_relative": symmetry, "roundoff_symmetrization_after_guard": True,
        "coarse_minimum_scalar_diagonal": float(diagonal.min()), "symmetric_scaled_Gershgorin_Gamma": gamma,
        "omega": 1/gamma, "Ac_definition": "P^T A P using accepted curved-P2/4Q gradients with fine fixed rows masked before gc",
        "straight_P1_stiffness_used": False}
    ledger.check(); ledger.account()
    return result, diagonal, gamma, facts


class AdditiveCoarse:
    def __init__(self, torch, P, PT, Ac, diagonal, gamma, fine_cholesky, ledger):
        self.torch, self.P, self.PT, self.Ac = torch, P, PT, Ac
        self.diagonal, self.omega, self.chol, self.ledger = diagonal, 1/gamma, fine_cholesky, ledger

    def prolong(self, value):
        self.ledger.data["P_calls"] += 1
        return self.torch.sparse.mm(self.P, value)

    def restrict(self, value):
        self.ledger.data["PT_calls"] += 1
        return self.torch.sparse.mm(self.PT, value)

    def coarse_apply(self, value):
        self.ledger.data["coarse_matvec_calls"] += 1
        return self.torch.sparse.mm(self.Ac, value[:, None]).squeeze(1)

    def apply(self, value):
        torch, ledger = self.torch, self.ledger
        ledger.sync(); began = time.monotonic()
        ledger.data["preconditioner_calls"] += 1
        tick = time.monotonic()
        fine = torch.cholesky_solve(value[..., None], self.chol).squeeze(-1)
        ledger.sync(); ledger.data["block_preconditioner_seconds"] += time.monotonic()-tick
        tick = time.monotonic()
        rhs = self.restrict(value).reshape(-1)
        ledger.sync(); ledger.data["transfer_seconds"] += time.monotonic()-tick
        tick = time.monotonic()
        zc = torch.zeros_like(rhs)  # Always zero: never warm or reuse a previous coarse solution.
        for _ in range(32):
            zc = zc + self.omega*(rhs-self.coarse_apply(zc))/self.diagonal
        ledger.sync(); ledger.data["coarse_inverse_seconds"] += time.monotonic()-tick
        tick = time.monotonic()
        result = fine+self.prolong(zc.reshape(-1, 3))
        ledger.sync(); ledger.data["transfer_seconds"] += time.monotonic()-tick
        ledger.data["preconditioner_wall_seconds"] += time.monotonic()-began
        ledger.check_memory()
        return result


def algebra_guards(torch, coarse, operator, inputs, ledger):
    """Two real fixed random vectors once; necessary guards, not a spectrum proof."""
    device, free = inputs["cells10"].device, inputs["free_node_ids"]
    generator = torch.Generator(device=device).manual_seed(2026)
    witnesses = [torch.randn((len(free), 3), dtype=torch.float64, device=device, generator=generator) for _ in range(2)]
    energies, discrepancy, fine_W = [], [], []
    before = dict(ledger.data)
    for value in witnesses:
        ledger.check()
        yc = coarse.restrict(value)
        projected = coarse.prolong(yc)
        full = torch.zeros_like(inputs["vertices_mm"]); full[free] = projected
        ledger.data["fine_K_applications"] += 1; ledger.data["setup_K_applications"] += 1
        exact = coarse.restrict(operator.apply(full)[free]).reshape(-1)
        actual = coarse.coarse_apply(yc.reshape(-1))
        denom = float(torch.linalg.vector_norm(exact))
        error = float(torch.linalg.vector_norm(actual-exact)/denom) if denom > 0 else None
        energy = float((yc.reshape(-1)*actual).sum())
        negative(error is not None and math.isfinite(error) and error <= 1e-9 and math.isfinite(energy) and energy > 0,
                 "Real Galerkin action/positive coarse energy guard failed")
        discrepancy.append(error); energies.append(energy)
        result = coarse.apply(value)
        negative(torch.isfinite(result).all(), "Nonfinite fixed additive-W guard result")
        fine_W.append(result); ledger.check(); ledger.account()
    xy, yx = float((witnesses[0]*fine_W[1]).sum()), float((witnesses[1]*fine_W[0]).sum())
    denom = float(torch.linalg.vector_norm(witnesses[0])*torch.linalg.vector_norm(fine_W[1])
                  + torch.linalg.vector_norm(witnesses[1])*torch.linalg.vector_norm(fine_W[0]))
    symmetry = abs(xy-yx)/max(denom, 1e-300)
    w_energy = [float((x*y).sum()) for x, y in zip(witnesses, fine_W)]
    negative(math.isfinite(symmetry) and symmetry <= 1e-9 and all(math.isfinite(x) and x > 0 for x in w_energy),
             "Fixed additive-W symmetry/positive energy guard failed")
    return {"passed": True, "seed": 2026, "vectors": 2, "Galerkin_action_relative_errors": discrepancy,
        "coarse_positive_energies": energies, "fixed_W_positive_energies": w_energy,
        "fixed_W_symmetry_relative": symmetry, "new_fine_K_calls": 2,
        "inherited_eight_fine_K_guards_repeated": 0,
        "coarse_matvec_calls": ledger.data["coarse_matvec_calls"]-before["coarse_matvec_calls"],
        "P_calls": ledger.data["P_calls"]-before["P_calls"], "PT_calls": ledger.data["PT_calls"]-before["PT_calls"],
        "claim": "Necessary guards plus conditional Galerkin/fixed-polynomial SPD argument; not exhaustive spectrum/engineering proof"}


def save_sparse(np, base, output, name, matrix):
    return {"shape": list(matrix.shape), "rowptr": base._save_array(np, output/(name+"_rowptr.npy"), matrix.crow_indices()),
            "columns": base._save_array(np, output/(name+"_columns.npy"), matrix.col_indices()),
            "values": base._save_array(np, output/(name+"_values.npy"), matrix.values())}


def load_sparse(np, torch, base, record, device):
    rows = base._load_array(np, torch, record["rowptr"], device)
    cols = base._load_array(np, torch, record["columns"], device)
    values = base._load_array(np, torch, record["values"], device)
    nr, nc = record["shape"]
    require(rows.dtype == cols.dtype == torch.int64 and values.dtype == torch.float64
            and rows.shape == (nr+1,) and cols.shape == values.shape and int(rows[0]) == 0 and int(rows[-1]) == len(cols)
            and (rows[1:] >= rows[:-1]).all() and (cols >= 0).all() and (cols < nc).all()
            and torch.isfinite(values).all(), "Committed coarse CSR schema differs")
    return torch.sparse_csr_tensor(rows, cols, values, size=(nr, nc), dtype=torch.float64, device=device)


def prepare_coarse(np, torch, base, inputs, arrays, operator, ledger, config):
    cache = ledger.output/"coarse_cache"; path = cache/"manifest.json"
    if path.exists():
        manifest = read_json(path)
        require(manifest["config"] == config and manifest["status"] == "completed" and manifest["algebra_guards"]["passed"] is True,
                "Committed coarse construction identity/guard differs")
        require(manifest["maximum_coarse_setup_wall_seconds"] == 60 and manifest["coarse_setup_seconds"] < 60,
                "Committed coarse setup exceeded frozen budget")
        ledger.account("setup_read")
        P = load_sparse(np, torch, base, manifest["arrays"]["P"], inputs["cells10"].device)
        PT = load_sparse(np, torch, base, manifest["arrays"]["PT"], inputs["cells10"].device)
        Ac = load_sparse(np, torch, base, manifest["arrays"]["Ac"], inputs["cells10"].device)
        diagonal = base._load_array(np, torch, manifest["arrays"]["diagonal"], inputs["cells10"].device)
        require(tuple(P.shape) == (len(inputs["free_node_ids"]), manifest["mapping"]["free_corner_nodes"])
                and tuple(PT.shape) == (P.shape[1], P.shape[0]) and tuple(Ac.shape) == (3*P.shape[1], 3*P.shape[1])
                and diagonal.shape == (Ac.shape[0],) and diagonal.dtype == torch.float64
                and torch.isfinite(diagonal).all() and (diagonal > 0).all(), "Coarse cache dimensions/diagonal differ")
        coarse = AdditiveCoarse(torch, P, PT, Ac, diagonal, manifest["assembly"]["symmetric_scaled_Gershgorin_Gamma"], arrays["free_cholesky"], ledger)
        alias = ledger.output/"coarse_manifest.json"
        if alias.exists():
            require(read_json(alias) == manifest, "Coarse cache export metadata differs from committed manifest")
        else:
            base.write_json(alias, manifest)
        ledger.check(); ledger.account()
        return coarse, manifest
    ledger.account("coarse_setup"); ledger.check()
    ledger.data["coarse_construct_attempts"] += 1; ledger.account()
    P, PT, corners, coarse_map, free_map, mapping = make_mapping(torch, inputs, ledger)
    Ac, diagonal, gamma, assembly = assemble_coarse(torch, inputs, arrays, coarse_map, free_map, ledger)
    coarse = AdditiveCoarse(torch, P, PT, Ac, diagonal, gamma, arrays["free_cholesky"], ledger)
    guards = algebra_guards(torch, coarse, operator, inputs, ledger)
    ledger.check(); cache.mkdir(parents=True, exist_ok=True)
    array_records = {"P": save_sparse(np, base, cache, "P", P), "PT": save_sparse(np, base, cache, "PT", PT),
                     "Ac": save_sparse(np, base, cache, "Ac", Ac), "diagonal": base._save_array(np, cache/"diagonal.npy", diagonal),
                     "coarse_original_node_ids": base._save_array(np, cache/"coarse_original_node_ids.npy", corners)}
    ledger.check(); ledger.account()
    manifest = {"status": "completed", "kind": "elastic_p1_derived_coarse_cache", "config": config,
        "mapping": mapping, "assembly": assembly, "algebra_guards": guards, "arrays": array_records,
        "coarse_setup_seconds": ledger.data["cumulative_coarse_setup_seconds"], "maximum_coarse_setup_wall_seconds": 60,
        "memory": base._memory(torch), "read_receipt": zero_receipt(), "checks_repeated_on_cache_load": False}
    # Commit only a completed real construction; partial array files never constitute a cache.
    base.write_json(path, manifest)
    base.write_json(ledger.output/"coarse_manifest.json", manifest)
    ledger.check(); ledger.account("setup_read")
    return coarse, manifest


def numerical_difference(np, prediction, reference):
    difference = prediction-reference
    reference_energy = float(np.square(reference).sum())
    error_energy = float(np.square(difference).sum())
    rms = math.sqrt(error_energy/len(reference))
    relative = math.sqrt(error_energy/reference_energy) if reference_energy > 0 else None
    passed = relative <= 1e-5 if relative is not None else rms <= 1e-12
    return {"nodes": len(reference), "reference_role": "Same-system converged approximate numerical displacement, not GT",
            "relative_l2": relative, "vector_RMS_mm": rms,
            "maximum_nodal_vector_difference_mm": float(np.linalg.norm(difference, axis=1).max()),
            "reference_energy_mm2": reference_energy, "difference_energy_mm2": error_energy,
            "zero_numerical_reference": reference_energy == 0, "same_solution_passed": bool(passed),
            "relative_l2_tolerance": 1e-5, "zero_reference_RMS_tolerance_mm": 1e-12,
            "old_GT_accuracy_assigned_to_new_endpoint": False}


def case_config(record, contract):
    return {"stage": CASE_KIND, "contract": {"stage_contract_sha256": digest(contract),
            "case_key": record["physical_record"]["key"], "physical_record": record["physical_record"],
            "source_sha256": contract["source_sha256"][SOURCE],
            "numerical_reference_summary_sha256": record["base_summary"]["sha256"]}}


def case_base(record, config):
    physical = record["physical_record"]
    return {"key": physical["key"], "cad": physical["cad"], "group": physical["group"],
            "node_count": physical["expected_geometry"]["nodes"], "cell_count": physical["expected_geometry"]["cells"],
            "kind": CASE_KIND, "config": config, "status": "completed", "attempted": False,
            "numerically_converged": False, "same_solution_passed": False, "read_receipt": zero_receipt(),
            "metrics": None, "costs": {"zero_cold_without_GT_seconds": record["base_summary"]["zero_cold_without_GT_seconds"],
                "coarse_cold_accounted_wall_seconds": None, "cold_cost_reduction_fraction": None,
                "coarse_to_zero_cold_cost_ratio": None, "components_complete": False}}


def costs_for_case(record, ledger, metadata_seconds):
    ledger.update()
    data = ledger.data
    zero = record["base_summary"]["zero_cold_without_GT_seconds"]
    inherited = record["base_summary"]["inherited_input_operator_seconds"]
    new = data["cumulative_case_wall_seconds"]
    complete = all(finite_cost(value) for value in (zero, inherited, metadata_seconds, new)) and zero > 0
    cold = inherited+metadata_seconds+new if complete else None
    return {"zero_cold_without_GT_seconds": zero, "inherited_fine_input_operator_seconds": inherited,
        "stage_metadata_single_query_seconds": metadata_seconds, "new_case_cumulative_wall_seconds": new,
        "coarse_cold_accounted_wall_seconds": cold, "coarse_to_zero_cold_cost_ratio": cold/zero if complete else None,
        "cold_cost_reduction_fraction": 1-cold/zero if complete else None, "components_complete": complete,
        **{name: data[name] for name in TIMES},
        "cached_solve_verification_reference_seconds": sum(data[name] for name in
            ("cumulative_solve_wall_seconds", "cumulative_endpoint_seconds", "cumulative_reference_seconds")),
        "fresh_combined_cold_remeasured": False,
        "definition": "Inherited accepted original fine input/operator setup + actual cumulative new case wall + common stage metadata per single query; numerical reference/verification/checkpoints included",
        "fine_setup_reused_but_charged": True, "old_zero_iterations_charged_to_new_budget": False,
        "coarse_setup_budget_overshoot_seconds": max(0., data["cumulative_coarse_setup_seconds"]-60),
        "solve_budget_overshoot_seconds": max(0., data["cumulative_solve_wall_seconds"]-300)}


def finish_case(record, endpoint, ledger, metadata_seconds):
    _, base = helpers()
    summary = endpoint["summary"]
    require(summary["config"] == ledger.config, "Persisted numerical endpoint belongs to another immutable case")
    for name in ("prediction", "constraint_reaction", "source_grid_node_ids"):
        value = summary[name]
        require(base.stat_record(value["path"]) == {part: value[part] for part in ("bytes", "mtime_ns")},
                "Persisted original-node endpoint array changed")
    ledger.account("persistence")
    summary["costs"] = costs_for_case(record, ledger, metadata_seconds)
    summary["counts"] = {name: ledger.data[name] for name in COUNTERS}
    summary["read_receipt"] = ledger.data["read_receipt"]
    summary["budget_finalization"] = "No new PCG iteration after limits; finite endpoint/comparison persisted with explicit wall overshoot"
    base.write_json(ledger.output/"summary.json", summary)
    ledger.account("persistence")
    # Include the actual preceding result persistence in the final reported cost.
    summary["costs"] = costs_for_case(record, ledger, metadata_seconds)
    base.write_json(ledger.output/"summary.json", summary)
    ledger.account("persistence")
    base.write_json(ledger.output/"status.json", {"config_sha256": ledger.config_sha256, "phase": "completed",
        "numerical_state": summary["numerical_state"], "cumulative_case_wall_seconds": ledger.data["cumulative_case_wall_seconds"],
        "cumulative_solve_wall_seconds": ledger.data["cumulative_solve_wall_seconds"],
        "iterations_executed_total": ledger.data["iterations_executed_total"]})
    return summary


def coarse_case(record, contract, output, clock, metadata_seconds):
    """Actual accepted cache → fixed coarse construction → same-system PCG."""
    cross, base = helpers()
    config = case_config(record, contract)
    if (output/"config.json").exists():
        require(read_json(output/"config.json") == config, "Existing coarse case config differs")
    output.mkdir(parents=True, exist_ok=True); base.write_json(output/"config.json", config)
    ledger = CaseLedger(output, config, clock, record["physical_record"]["key"])
    ledger.account()
    if (output/"endpoint_no_GT.json").exists():
        return finish_case(record, read_json(output/"endpoint_no_GT.json"), ledger, metadata_seconds)
    state, operator, coarse, coarse_manifest = None, None, None, None
    try:
        ledger.check()
        import numpy as np
        import torch
        from .elastic_pilot_data import load_input
        ledger.torch = torch
        device = torch.device("cuda:0")
        torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        physical = record["physical_record"]
        manifest = read_json(record["input_manifest"]["path"])
        accepted = read_json(record["operator_summary"]["path"])
        inputs = load_input(manifest, device)
        lam, mu = base._validate_inputs(torch, inputs, physical)
        arrays = {name: base._load_array(np, torch, value, device) for name, value in accepted["arrays"].items()}
        require(arrays["gradients"].shape == (len(inputs["cells10"]), 4, 10, 3)
                and arrays["weights"].shape == (len(inputs["cells10"]), 4)
                and arrays["free_cholesky"].shape == (len(inputs["free_node_ids"]), 3, 3)
                and all(value.dtype == torch.float64 for value in arrays.values()), "Accepted fine operator actual array schema differs")
        operator = base.MatrixFreeElastic(torch, inputs["cells10"], arrays["gradients"], arrays["weights"],
                                         len(inputs["vertices_mm"]), lam, mu)
        free = inputs["free_node_ids"]
        b = inputs["force"][free]
        norm_b = float(torch.linalg.vector_norm(b))
        negative(math.isfinite(norm_b) and norm_b > 0, "Accepted reduced RHS is zero/nonfinite")
        ledger.check(); ledger.account()
        coarse, coarse_manifest = prepare_coarse(np, torch, base, inputs, arrays, operator, ledger, config)
        z, r, p = torch.zeros_like(b), b.clone(), torch.zeros_like(b)
        state = {"config": config, "iteration": 0, "rz": None, "trace": []}
        checkpoint = output/"checkpoint.json"
        if checkpoint.exists():
            state = read_json(checkpoint)
            require(state["config"] == config and base.stat_record(state["vectors_path"]) == state["vectors_stat"],
                    "Committed same-coarse PCG vector identity differs")
            with np.load(host_path(state["vectors_path"]), allow_pickle=False) as source:
                require(set(source.files) == {"z", "r", "p"}, "Unexpected coarse PCG checkpoint members")
                z, r, p = (torch.as_tensor(np.array(source[name]), device=device) for name in ("z", "r", "p"))
            require(all(value.shape == b.shape and value.dtype == torch.float64 and torch.isfinite(value).all() for value in (z, r, p)),
                    "Committed coarse PCG state is nonfinite/malformed")
            require(state["iteration"] <= ledger.data["iterations_executed_total"], "Cumulative attempted iterations cannot reset below checkpoint")
        ledger.account("solving")

        def reduced_apply(value, count="solve_K_applications"):
            full = torch.zeros_like(inputs["vertices_mm"]); full[free] = value
            ledger.data["fine_K_applications"] += 1; ledger.data[count] += 1
            result = operator.apply(full)[free]
            ledger.check_memory()
            return result

        def save():
            nonlocal state
            tick = time.monotonic(); ledger.account()
            state["cumulative_solve_wall_seconds"] = ledger.data["cumulative_solve_wall_seconds"]
            state["iterations_executed_total"] = ledger.data["iterations_executed_total"]
            state["fine_K_applications"] = ledger.data["fine_K_applications"]
            state = base._save_pcg(np, output, state, z, r, p)
            ledger.data["checkpoint_seconds"] += time.monotonic()-tick
            ledger.account()

        true_r = b-reduced_apply(z)
        true_relative = base._relative(torch, true_r, norm_b)
        negative(torch.isfinite(true_r).all() and true_relative is not None, "Nonfinite/undefined initial or resumed coarse PCG residual")
        state["trace"].append({"iteration": state["iteration"], "true_relative_residual": true_relative,
                               "reason": "initial_or_resume_true_verification"})
        numerical_state = "converged" if true_relative <= 1e-8 else "solving"
        if state.get("checkpoint_recurrence_state") == "terminal_true_converged":
            negative(numerical_state == "converged", "Committed terminal endpoint no longer passes true residual; no stale-PCG continuation")
        ledger.update()
        if state["iteration"] == 0 and numerical_state == "solving" and ledger.data["cumulative_solve_wall_seconds"] < 300 and clock.remaining() > 0:
            r = true_r.clone()
            preconditioned = coarse.apply(r)
            rz = float((r*preconditioned).sum()); p = preconditioned.clone()
            state["rz"] = rz
        else:
            rz = state["rz"]
        if numerical_state == "solving" and state["iteration"] > 0:
            negative(rz is not None and math.isfinite(rz) and rz > 0, "Committed coarse PCG rTWr is nonpositive/nonfinite")
        save()
        while numerical_state == "solving":
            ledger.update()
            if clock.remaining() <= 0:
                numerical_state = "stage_wall_budget_limit"; break
            if ledger.data["cumulative_solve_wall_seconds"] >= 300:
                numerical_state = "solve_wall_budget_limit"; break
            if ledger.data["iterations_executed_total"] >= 10000:
                numerical_state = "attempted_iteration_limit"; break
            ledger.check(phase_budget=False)
            negative(rz is not None and math.isfinite(rz) and rz > 0, "Coarse PCG rTWr is nonpositive/nonfinite")
            ledger.data["iterations_executed_total"] += 1; ledger.account()
            Ap = reduced_apply(p)
            pAp = float((p*Ap).sum())
            negative(math.isfinite(pAp) and pAp > 0, "Fine pTAp is nonpositive/nonfinite; no shift/repair")
            alpha = rz/pAp
            negative(math.isfinite(alpha), "Nonfinite PCG alpha")
            z = z+alpha*p; r = r-alpha*Ap
            state["iteration"] += 1
            negative(torch.isfinite(z).all() and torch.isfinite(r).all(), "Nonfinite coarse PCG vectors")
            recurrence = base._relative(torch, r, norm_b)
            verify = state["iteration"] % 25 == 0 or (recurrence is not None and recurrence <= 1e-8)
            if verify:
                true_r = b-reduced_apply(z); true_relative = base._relative(torch, true_r, norm_b)
                negative(torch.isfinite(true_r).all() and true_relative is not None, "Nonfinite true fine residual")
                ledger.update()
                state["trace"].append({"iteration": state["iteration"], "recurrence_relative_residual": recurrence,
                    "true_relative_residual": true_relative, "cumulative_solve_wall_seconds": ledger.data["cumulative_solve_wall_seconds"],
                    "reason": "periodic_or_candidate_true_verification"})
                if true_relative <= 1e-8:
                    numerical_state = "converged"
            if numerical_state == "solving":
                preconditioned = coarse.apply(r)
                next_rz = float((r*preconditioned).sum())
                negative(math.isfinite(next_rz) and next_rz > 0, "New rTWr is nonpositive/nonfinite")
                p = preconditioned+(next_rz/rz)*p; rz = next_rz; state["rz"] = rz
                state["checkpoint_recurrence_state"] = "consistent_continuation"
            else:
                state["checkpoint_recurrence_state"] = "terminal_true_converged"
            if state["iteration"] % 25 == 0 or numerical_state == "converged":
                save()
            else:
                ledger.account()
        # The true endpoint verification is mandatory and separately exposes budget overshoot.
        true_r = b-reduced_apply(z); true_relative = base._relative(torch, true_r, norm_b)
        negative(torch.isfinite(true_r).all() and true_relative is not None, "Nonfinite endpoint true fine residual")
        if true_relative <= 1e-8:
            numerical_state = "converged"
            state["checkpoint_recurrence_state"] = "terminal_true_converged"
        ledger.update()
        state["trace"].append({"iteration": state["iteration"], "true_relative_residual": true_relative,
            "cumulative_solve_wall_seconds": ledger.data["cumulative_solve_wall_seconds"], "reason": "endpoint_true_verification"})
        save(); ledger.account("endpoint")
        full_u = base._restore_full(torch, z, inputs)
        ledger.data["fine_K_applications"] += 1; ledger.data["endpoint_K_applications"] += 1
        residual = operator.apply(full_u)-inputs["force"]
        ledger.check_memory()
        reaction = torch.zeros_like(residual); reaction[inputs["fixed_node_ids"]] = residual[inputs["fixed_node_ids"]]
        external = inputs["force"]+reaction
        reference_xyz = inputs["reference_xyz_mm"]
        net_force = external.sum(dim=0)
        net_moment = torch.cross(inputs["vertices_mm"]-reference_xyz, external, dim=-1).sum(dim=0)
        negative(all(torch.isfinite(value).all() for value in (full_u, residual, net_force, net_moment)),
                 "Nonfinite original-node endpoint or equilibrium")
        prediction = base._save_array(np, output/"physical_u_mm.npy", full_u)
        reaction_record = base._save_array(np, output/"constraint_reaction_N.npy", reaction)
        grid_record = base._save_array(np, output/"source_grid_node_ids.npy", inputs["grid_node_ids"])
        equilibrium = {"true_free_residual_relative": base._relative(torch, residual[free], norm_b),
            "true_free_residual_norm_N": float(torch.linalg.vector_norm(residual[free])),
            "fixed_translation_max_abs_mm": float(full_u[inputs["fixed_node_ids"]].abs().max()),
            "constraint_reaction_sum_N": reaction.sum(dim=0).tolist(), "net_force_with_reaction_N": net_force.tolist(),
            "net_moment_with_reaction_Nmm": net_moment.tolist(), "reference_xyz_mm": reference_xyz.tolist()}
        ledger.account("reference")
        old_prediction = record["base_summary"]["prediction"]
        require(base.stat_record(old_prediction["path"]) == {part: old_prediction[part] for part in ("bytes", "mtime_ns")},
                "Frozen numerical reference stat changed after contract")
        ledger.data["numerical_reference_reads"] += 1; ledger.account()
        reference = np.load(host_path(old_prediction["path"]), allow_pickle=False)
        require(reference.shape == (physical["expected_geometry"]["nodes"], 3) and reference.dtype == np.float64
                and np.isfinite(reference).all(), "Same-system numerical reference original-node schema differs")
        comparison = numerical_difference(np, full_u.detach().cpu().numpy(), reference)
        base.write_json(output/"numerical_comparison.json", {"config": config, **comparison, "read_receipt": zero_receipt()})
        ledger.account("persistence")
        summary = {**case_base(record, config), "attempted": True, "input_compatible": True, "operator_compatible": True,
            "numerical_state": numerical_state, "numerically_converged": numerical_state == "converged",
            "true_relative_residual": true_relative, "iteration": state["iteration"], "trace": state["trace"],
            "same_solution_passed": comparison["same_solution_passed"], "numerical_comparison": comparison,
            "prediction": prediction, "physical_u_path": prediction["path"], "constraint_reaction": reaction_record,
            "source_grid_node_ids": grid_record, "equilibrium": equilibrium, "memory": base._memory(torch),
            "coarse_manifest_path": str(output/"coarse_manifest.json"), "coarse_mapping": coarse_manifest["mapping"],
            "coarse_assembly": coarse_manifest["assembly"], "coarse_algebra_guards": coarse_manifest["algebra_guards"],
            "input_manifest_path": record["input_manifest"]["path"], "operator_summary_path": record["operator_summary"]["path"],
            "old_GT_accuracy_assigned_to_new_endpoint": False}
        endpoint = {"summary": summary, "read_receipt": zero_receipt(),
                    "numerical_reference_role": "Existing converged same-equation solution, not GT"}
        base.write_json(output/"endpoint_no_GT.json", endpoint); ledger.account()
        return finish_case(record, endpoint, ledger, metadata_seconds)
    except base.ScientificNegative as error:
        ledger.account()
        summary = {**case_base(record, config), "attempted": True, "numerical_state": "frozen_candidate_negative",
                   "reason": str(error), "costs": costs_for_case(record, ledger, metadata_seconds),
                   "counts": {name: ledger.data[name] for name in COUNTERS}, "read_receipt": ledger.data["read_receipt"]}
        if ledger.torch is not None:
            summary["memory"] = base._memory(ledger.torch)
        base.write_json(output/"summary.json", summary); ledger.account()
        return summary
    except (NotImplementedError, RuntimeError) as error:
        # Backend/OOM negatives terminate this frozen candidate, never trigger CPU/precision fallback.
        message = str(error).lower()
        unsupported = isinstance(error, NotImplementedError) or any(token in message for token in
                      ("not implemented", "not supported", "unsupported", "cusparse", "out of memory"))
        ledger.account()
        if unsupported:
            summary = {**case_base(record, config), "attempted": True, "numerical_state": "backend_or_memory_negative",
                "reason": f"{type(error).__name__}: {error}", "costs": costs_for_case(record, ledger, metadata_seconds),
                "counts": {name: ledger.data[name] for name in COUNTERS}, "read_receipt": ledger.data["read_receipt"],
                "fallbacks": 0}
            if ledger.torch is not None:
                summary["memory"] = base._memory(ledger.torch)
            base.write_json(output/"summary.json", summary); return summary
        raise
    except Exception:
        ledger.account()
        raise


def selection_gate(cases):
    converged = sum(item.get("numerically_converged") is True and finite_cost(item.get("true_relative_residual"))
                    and item["true_relative_residual"] <= 1e-8 for item in cases.values())
    same = sum(item.get("same_solution_passed") is True for item in cases.values())
    complete = len(cases) == 2 and all(item.get("costs", {}).get("components_complete") is True for item in cases.values())
    gains = [item.get("costs", {}).get("cold_cost_reduction_fraction") for item in cases.values()]
    beneficial = sum(isinstance(value, (int, float)) and math.isfinite(value) and value >= .25 for value in gains)
    no_gt = len(cases) == 2 and all(all(item["read_receipt"].get(name) == 0 for name in (*ZERO_KEYS, "learned_weight_reads", "setup_K_guards_repeated"))
                                 for item in cases.values())
    passed = converged == same == beneficial == 2 and complete and no_gt
    return {"passed": passed, "inventory_denominator": 2, "numerically_converged_cases": converged,
        "same_solution_passed_cases": same, "cases_with_at_least25percent_cold_cost_reduction": beneficial,
        "complete_cost_components": complete, "target_and_weight_reads_zero": no_gt,
        "minimum_required_gain_fraction": .25,
        "decision": "expand_fixed_candidate_to_existing_five_CAD_only_after_root_decision" if passed else "close_fixed32_P1_derived_coarse_candidate_no_rescue"}


def write_report(output, summary):
    """Small output JSON/Markdown only; no array/recomputed scientific metrics."""
    _, base = helpers()
    metrics = {"kind": KIND, "case_keys": list(CASES), "selection_gate": summary["selection_gate"],
               "read_receipt": summary["read_receipt"], "costs": summary["costs"],
               "cases": {key: {name: summary["cases"][key].get(name) for name in
                   ("numerical_state", "true_relative_residual", "same_solution_passed", "numerical_comparison", "costs", "counts", "memory")}
                   for key in CASES}}
    base.write_json(output/"metrics.json", metrics)
    verdict = "通过两例完整cold各至少25%收益门，可由root决定扩五例。" if summary["selection_gate"]["passed"] else "未通过两例完整cold各至少25%收益门，关闭这个固定32步候选。"
    lines = ["# P1-derived 两层粗空间成本粗筛", "", verdict, "", "| CAD | 原zero cold秒 | coarse cold秒 | 成本降低 |",
             "|---|---:|---:|---:|"]
    for key in CASES:
        values = summary["cases"][key]["costs"]
        def fmt(value, percent=False):
            return "—" if value is None else (f"{100*value:.2f}%" if percent else f"{value:.3f}")
        lines.append(f"| {key} | {fmt(values.get('zero_cold_without_GT_seconds'))} | {fmt(values.get('coarse_cold_accounted_wall_seconds'))} | {fmt(values.get('cold_cost_reduction_fraction'), True)} |")
    lines += ["", "完整cold为账本成本：复用但计价的原fine输入/算子setup，加新case全部wall与共同元数据；不是重新测得的fresh组合cold。",
        "同一原P2方程、约束和RBE3载荷；只改数值预条件。GT、stress、旧final、新sealed test和神经权重读取均为0。",
        "固定中点掩码可能令空间不再严格nested P1；P2→P1不是h网格粗化。负结果仅关闭此固定候选，不否定所有AMG。",
        "当前物理接口与点云/25特征学习接口信息不同，本结果不得与旧final神经误差拼成最终solver排名。"]
    path = output/"BRIEF.md"; temporary = path.with_name(path.name+".tmp")
    temporary.write_text("\n".join(lines)+"\n", encoding="utf-8"); os.replace(temporary, path)


def run(plan_path=PLAN, output_path=None):
    began = time.monotonic(); contract, plan = request_contract(plan_path)
    require(plan.get("scientific_rules_frozen") is True and plan.get("actual_execution_released") is True,
            "Root actual release required before scientific imports/array/CUDA reads")
    review = read_json(plan["root_review"])
    require(review.get("passed") is True and review.get("source_sha256") == contract["source_sha256"][SOURCE]
            and review.get("science_recipe_sha256") == digest(contract["plan_science"])
            and review.get("base_evidence_decision") == "proceed_frozen_P1_derived_coarse",
            "Authentic precise root source review and base-evidence decision required")
    output = host_path(output_path or plan["output"])
    require(output == host_path(plan["output"]), "Coarse stage output differs from frozen destination")
    config = {"contract": contract, "stage": KIND}
    _, base = helpers()
    if (output/"config.json").exists():
        require(read_json(output/"config.json") == config, "Existing coarse stage config differs")
    if (output/"summary.json").exists():
        previous = read_json(output/"summary.json")
        require(previous.get("config") == config, "Saved coarse stage identity differs")
        if previous.get("status") == "completed":
            return previous
    output.mkdir(parents=True, exist_ok=True); base.write_json(output/"config.json", config)
    metadata_path = output/"stage_metadata_receipt.json"
    old_metadata = read_json(metadata_path) if metadata_path.exists() else {}
    if old_metadata:
        require(old_metadata["config_sha256"] == digest(config), "Cumulative metadata identity differs")
    actual_metadata = time.monotonic()-began
    metadata_seconds = old_metadata.get("first_metadata_seconds", actual_metadata)
    base.write_json(metadata_path, {"config_sha256": digest(config), "first_metadata_seconds": metadata_seconds,
                    "cumulative_actual_metadata_seconds": old_metadata.get("cumulative_actual_metadata_seconds", 0.)+actual_metadata})
    cross, _ = helpers()
    clock = cross.StageClock(output, config, began); clock.account("running")
    cases = {}
    try:
        for key in CASES:
            record = contract["records"][key]; case_output = host_path(record["case_output"])
            require(case_output == output/key.replace("/", "_"), "Fixed coarse case output differs")
            if (case_output/"summary.json").exists():
                item = read_json(case_output/"summary.json")
                require(item.get("config") == case_config(record, contract), "Saved coarse case contract differs")
                if item.get("status") == "completed":
                    cases[key] = item; clock.account("case_reused", key); continue
            if clock.remaining() <= 0 and not (case_output/"checkpoint.json").exists() and not (case_output/"endpoint_no_GT.json").exists():
                case_output.mkdir(parents=True, exist_ok=True)
                item = {**case_base(record, case_config(record, contract)), "numerical_state": "stage_budget_unattempted_negative",
                        "reason": "No new case after900s; fixed failure denominator retained, cold cost remains null"}
                base.write_json(case_output/"config.json", item["config"]); base.write_json(case_output/"summary.json", item)
                cases[key] = item; clock.account("case_not_attempted", key); continue
            item = coarse_case(record, contract, case_output, clock, metadata_seconds)
            cases[key] = item; clock.account("case_completed", key)
        gate = selection_gate(cases)
        elapsed = clock.elapsed()
        receipt = zero_receipt()
        receipt["numerical_reference_reads"] = sum(item["read_receipt"]["numerical_reference_reads"] for item in cases.values())
        summary = {"status": "completed", "kind": KIND, "config": config, "case_keys": list(CASES), "cases": cases,
            "selection_gate": gate, "read_receipt": receipt, "elapsed_seconds": elapsed,
            "costs": {"cumulative_stage_wall_seconds": elapsed, "maximum_stage_wall_seconds": 900,
                "this_call_stage_wall_seconds": time.monotonic()-began, "stage_budget_overshoot_seconds": max(0., elapsed-900),
                "metadata_seconds_per_single_query": metadata_seconds,
                "actual_metadata_seconds_including_resumes": read_json(metadata_path)["cumulative_actual_metadata_seconds"]},
            "counts": {"fixed_cases": 2, "attempted": sum(item["attempted"] for item in cases.values()),
                "numerically_converged": gate["numerically_converged_cases"], "same_solution": gate["same_solution_passed_cases"],
                "new_setup_Galerkin_fine_K_calls": sum(item.get("counts", {}).get("setup_K_applications", 0) for item in cases.values()),
                "fine_operator_eight_guard_repeats": 0},
            "scientific_outcome": "frozen_macro_speed_gate_passed" if gate["passed"] else "frozen_macro_speed_gate_failed",
            "scope": "Two fixed existing TRAIN/DEV ver CADs; same approximate fine P2 equation, no GT/stress/final selection claim",
            "interface_comparison_limit": "FullTET10/realconstraints physics and pointcloud25 learning interfaces differ; no combined finalsolver ranking",
            "old_GT_accuracy_assigned_to_new_endpoint": False, "new_training_updates": 0,
            "failure_policy": "Close this frozen32 P1-derived candidate on scientific/budget/backend negative; no fallback/rescue/replacement"}
        base.write_json(output/"summary.json", summary); write_report(output, summary); clock.account("completed")
        return summary
    except Exception as error:
        clock.account("failed")
        base.write_json(output/"summary.json", {"status": "failed", "kind": KIND, "config": config,
            "case_keys": list(CASES), "cases": cases, "read_receipt": zero_receipt(),
            "reason": f"{type(error).__name__}: {error}", "costs": {"cumulative_stage_wall_seconds": clock.elapsed()}})
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, default=Path(PLAN))
    result.add_argument("--output", type=Path)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    run(args.plan, args.output)
