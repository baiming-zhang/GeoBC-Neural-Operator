"""One frozen U-only blind full-FE service and frozen FNO reference.

Only stdlib metadata is read before the authentic root release.  The parent
launches a fresh process per request; labels require ten committed node fields
and five committed native physical four-Q fields.  No stress readout exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_blind_solver_plan.json"
KIND = "blind_full_FE_U_with_frozen_FNO_reference"
OWN = "geobc/blind_solver_service.py"
DECISION = "proceed_frozen_blind_full_FE_U_with_FNO_reference"
CASES = ("489_12/ver", "616_205/ver", "416_282/ver", "247_50/ver", "142_454/ver")
ARMS = ("physics_full_FE", "FNO_frozen2880")
ZERO = ("stress_target_reads", "H5_nodal_variable_reads", "strain_target_reads",
    "hor_tor_dia_target_reads", "training_updates", "scaler_fits", "test_endpoint_selection",
    "test_seed_selection", "old_final_reads", "old_development_target_reads",
    "GT_U_reads_before_prediction_barrier", "source_scalar_formula_trials",
    "case_replacements", "mesh_or_geometry_repairs")
ACTUAL = ("raw_h5_input_reads", "raw_deck_input_reads", "raw_CSV_coordinate_reads",
    "input_cache_reads", "input_member_reads", "new_operator_builds",
    "new_fine_guard_K_applications", "new_coarse_builds", "new_coarse_guard_K_applications",
    "fine_K_applications", "solve_iterations", "preconditioner_applications",
    "FNO_checkpoint_read_attempts", "FNO_checkpoint_reads", "FNO_checkpoint_byte_hash_reads",
    "FNO_inference_requests", "native_query_cases", "own_PCG_checkpoint_reads",
    "own_PCG_checkpoint_writes", "source_U_target_parse_attempts", "source_U_target_parses",
    "owned_GT_cache_reads", "own_prediction_array_reads")


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def read(value):
    return json.loads(path(value).read_text(encoding="utf-8"))


def write(value, data):
    p = path(value); p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, p)


def sha(value):
    h = hashlib.sha256()
    with path(value).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Negative(RuntimeError):
    """Terminal scientific, backend or fixed-budget negative; no rescue."""


def request_contract(plan_path=PLAN):
    """Metadata/source identity only; never parse raw data or load weights."""
    plan = read(plan_path)
    require(plan["kind"] == KIND and plan["case_keys"] == list(CASES) and plan["arms"] == list(ARMS), "Fixed inventory differs")
    require(plan["scientific_rules_frozen"] is True and plan["actual_execution_released"] is False, "Source plan release/science flags differ")
    require(digest(plan["science"]) == plan["scientific_digest"], "Scientific digest differs")
    require(plan["science"]["root_recipe"] == read(plan["root_recipe"]["path"]), "Authoritative frozen recipe differs")
    for name, expected in plan["source_sha256"].items():
        require(sha(name) == expected, "Pinned source changed: " + name)
    require(OWN in plan["source_sha256"], "Driver source pin missing")
    for record in plan["metadata_pins"].values():
        require(sha(record["path"]) == record["sha256"], "Pinned metadata changed: " + record["path"])
    # Input helper contract is also metadata-only.  Its module contains no
    # scientific top-level imports and is never imported by source validation.
    from . import blind_solver_inputs as data
    input_contract, _ = data.request_contract(plan_path)
    contract = {"schema": "blind_full_FE_U_service_v1", "kind": KIND,
        "case_keys": list(CASES), "arms": list(ARMS), "science": plan["science"],
        "scientific_digest": plan["scientific_digest"], "source_sha256": plan["source_sha256"],
        "metadata_pins": plan["metadata_pins"], "input_contract": input_contract,
        "input_contract_sha256": digest(input_contract), "output": str(path(plan["output"]))}
    return contract, plan


def release(contract, plan):
    marker = read(plan["root_review"])
    require(marker.get("passed") is True and marker.get("decision") == DECISION
        and marker.get("source_sha256") == contract["source_sha256"][OWN]
        and marker.get("source_pins") == contract["source_sha256"]
        and marker.get("scientific_digest") == contract["scientific_digest"], "Authentic root release missing or differs")


def receipt():
    return {k: 0 for k in ZERO + ACTUAL}


def merge_receipts(values):
    result = receipt()
    for value in values:
        require(all(value.get(k, 0) == 0 for k in ("source_S_target_reads", "source_strain_target_reads",
            "DEV_dia_reads", "model_or_weight_reads", "optimizer_steps", "new_physical_solves", "K_applications")), "Helper forbidden read/action observed")
        for k in result:
            v = value.get(k, 0)
            require(type(v) is int and v >= 0, "Native receipt counter differs: " + k)
            result[k] += v
    require(all(result[k] == 0 for k in ZERO), "Forbidden read/action observed")
    return result


class Clock:
    """Compact cumulative wall journal, including unclosed attempt gaps."""
    def __init__(self, output, maximum, allowance=None):
        self.output = path(output); self.output.mkdir(parents=True, exist_ok=True)
        self.file = self.output / "clock.json"
        self.data = read(self.file) if self.file.exists() else {"cumulative_wall_seconds": 0.0,
            "phase_wall_seconds": {}, "attempts": 0, "active": False, "last_epoch": None,
            "phase": "startup", "unknown_gap_charged_seconds": 0.0, "peak_GPU_bytes": 0}
        now = time.time()
        if self.data["active"]:
            gap = max(0., now - self.data["last_epoch"])
            self.data["cumulative_wall_seconds"] += gap
            phase = self.data["phase"]
            self.data["phase_wall_seconds"][phase] = self.data["phase_wall_seconds"].get(phase, 0.) + gap
            self.data["unknown_gap_charged_seconds"] += gap
        self.prior = self.data["cumulative_wall_seconds"]
        self.maximum = float(maximum)
        self.allowance = float(allowance) if allowance is not None else self.maximum - self.prior
        self.began = self.last = time.monotonic(); self.phase_limit = None
        self.data["attempts"] += 1; self.data["active"] = True
        self.persist()

    def account(self):
        now = time.monotonic(); delta = now - self.last; self.last = now
        self.data["cumulative_wall_seconds"] += delta
        name = self.data["phase"]
        self.data["phase_wall_seconds"][name] = self.data["phase_wall_seconds"].get(name, 0.) + delta

    def persist(self):
        self.account(); self.data["last_epoch"] = time.time(); write(self.file, self.data)

    def phase(self, name, limit=None):
        self.account(); self.data["phase"] = name; self.phase_limit = limit; self.persist()

    def guard(self):
        self.account()
        if self.data["cumulative_wall_seconds"] >= self.maximum or time.monotonic() - self.began >= self.allowance:
            raise Negative("Fixed campaign/request wall budget exhausted")
        if self.phase_limit is not None and self.data["phase_wall_seconds"].get(self.data["phase"], 0.) >= self.phase_limit:
            raise Negative("Fixed phase wall budget exhausted: " + self.data["phase"])

    def memory(self, torch):
        torch.cuda.synchronize()
        peak = max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
        self.data["peak_GPU_bytes"] = max(self.data["peak_GPU_bytes"], int(peak))
        if peak > 24 * 1024 ** 3:
            raise Negative("Fixed 24GiB peak GPU budget exhausted")
        self.guard()

    def finish(self):
        self.account(); self.data["active"] = False; self.persist()


class Counter:
    def __init__(self, output):
        self.output = path(output); self.file = self.output / "read_receipt.json"
        self.data = read(self.file) if self.file.exists() else receipt()

    def bump(self, name, count=1, persist=False):
        require(name in ACTUAL and type(count) is int and count >= 0, "Counter event differs")
        self.data[name] += count
        if persist:
            self.persist()

    def persist(self):
        require(all(self.data[k] == 0 for k in ZERO), "Forbidden counter nonzero")
        write(self.file, self.data)

    def observer(self, event, source, member=None):
        # The helper durably journals attempted/completed events itself before
        # returning; native completed counters are absorbed without inventing
        # a second counter from its differently named observer events.
        return None

    def absorb(self, value, before):
        require(all(value.get(k, 0) == 0 for k in ("source_S_target_reads", "source_strain_target_reads",
            "DEV_dia_reads", "model_or_weight_reads", "optimizer_steps", "new_physical_solves", "K_applications")), "Helper forbidden read/action observed")
        for k in ACTUAL:
            if k in value:
                require(type(value[k]) is int and value[k] >= 0, "Helper actual counter differs")
                delta = value[k] - before.get(k, 0)
                require(delta >= 0, "Helper cumulative counter reset")
                self.data[k] += delta
        for k in ZERO:
            require(value.get(k, 0) == 0, "Helper forbidden counter nonzero")
        self.persist()


class CoarseLedger:
    """Accepted coarse math adapter; no old scope/GT entrypoint."""
    def __init__(self, torch, coarse, timer, counters, prior=None):
        self.torch, self.timer, self.counters = torch, timer, counters
        self.data = {k: 0 for k in coarse.COUNTERS}
        self.data.update({k: 0.0 for k in coarse.TIMES})
        if prior:
            self.data.update(prior)

    def check(self):
        self.timer.memory(self.torch)

    def check_memory(self):
        self.timer.memory(self.torch)

    def sync(self):
        self.torch.cuda.synchronize()

    def account(self, *args):
        self.timer.account()


def array_record(np, destination, array):
    destination = path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    array = np.ascontiguousarray(array)
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, destination); info = destination.stat()
    return {"path": str(destination), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
        "shape": list(array.shape), "dtype": str(array.dtype),
        "values_sha256": hashlib.sha256(array.tobytes()).hexdigest(), "file_sha256": sha(destination)}


def load_owned(np, record, counters, GT=False):
    counters.bump("owned_GT_cache_reads" if GT else "own_prediction_array_reads", persist=True)
    info = path(record["path"]).stat()
    require(info.st_size == record["bytes"] and info.st_mtime_ns == record["mtime_ns"] and sha(record["path"]) == record["file_sha256"], "Owned array file identity differs")
    value = np.load(path(record["path"]), allow_pickle=False)
    require(list(value.shape) == record["shape"] and str(value.dtype) == record["dtype"]
        and hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == record["values_sha256"], "Owned array value identity differs")
    return value


def child_dir(output, arm, key):
    return path(output) / "requests" / arm / key.replace("/", "_")


def save_config(folder, config):
    target = folder / "config.json"
    if target.exists():
        require(read(target) == config, "Own resumed config differs")
    else:
        write(target, config)


def helper_counts(folder):
    target = path(folder) / "read_receipt.json"
    return read(target) if target.exists() else {}


def physical_prepare(np, torch, base, coarse, inputs, plan, folder, timer, counters, config):
    timer.phase("input_operator_coarse", 300)
    operator_file = folder / "operator_manifest.json"
    if operator_file.exists():
        manifest = read(operator_file)
        require(manifest["config"] == config and manifest["status"] == "completed" and manifest["algebra_guards"]["passed"], "Own operator cache differs")
        arrays = {name: base._load_array(np, torch, record, inputs["vertices_mm"].device) for name, record in manifest["arrays"].items()}
        lam, mu = base._validate_inputs(torch, inputs, plan)
        operator = base.MatrixFreeElastic(torch, inputs["cells10"], arrays["gradients"], arrays["weights"], len(inputs["vertices_mm"]), lam, mu)
    else:
        counters.bump("new_operator_builds", persist=True)
        # The accepted helper's internal 300s deadline includes already charged
        # raw input and prior failed setup, never starts a fresh scientific cap.
        prior = timer.data["phase_wall_seconds"].get("input_operator_coarse", 0.)
        began = time.monotonic()
        operator, chol, diagonal, geometry = base._build_geometry(torch, inputs, plan, began, prior)
        before = operator.calls
        try:
            guards = base._algebra_guards(torch, operator, diagonal, inputs, plan, began, prior)
        finally:
            counters.bump("new_fine_guard_K_applications", operator.calls - before)
            counters.bump("fine_K_applications", operator.calls - before, persist=True)
        # Necessary fresh guards already accepted, not duplicated by queue.
        require(guards["K_applications"] == 8, "Fresh eight-guard count differs")
        guards["passed"] = True
        arrays = {"gradients": operator.gradients, "weights": operator.weights, "free_cholesky": chol, "diagonal_blocks": diagonal}
        cache = folder / "operator_cache"; cache.mkdir(parents=True, exist_ok=True)
        records = {name: base._save_array(np, cache / (name + ".npy"), value) for name, value in arrays.items()}
        timer.memory(torch)
        manifest = {"status": "completed", "config": config, "geometry": geometry,
            "algebra_guards": guards, "arrays": records, "checks_repeated_on_cache_load": False}
        write(operator_file, manifest)
    ledger = CoarseLedger(torch, coarse, timer, counters)
    coarse_file = folder / "coarse_manifest.json"
    if coarse_file.exists():
        committed = read(coarse_file)
        require(committed["config"] == config and committed["status"] == "completed" and committed["algebra_guards"]["passed"], "Own coarse cache differs")
        P, PT, Ac = [coarse.load_sparse(np, torch, base, committed["arrays"][name], inputs["vertices_mm"].device) for name in ("P", "PT", "Ac")]
        D = base._load_array(np, torch, committed["arrays"]["diagonal"], inputs["vertices_mm"].device)
        gamma = committed["assembly"]["symmetric_scaled_Gershgorin_Gamma"]
        W = coarse.AdditiveCoarse(torch, P, PT, Ac, D, gamma, arrays["free_cholesky"], ledger)
    else:
        counters.bump("new_coarse_builds", persist=True)
        P, PT, corners, coarse_map, free_map, mapping = coarse.make_mapping(torch, inputs, ledger)
        Ac, D, gamma, assembly = coarse.assemble_coarse(torch, inputs, arrays, coarse_map, free_map, ledger)
        W = coarse.AdditiveCoarse(torch, P, PT, Ac, D, gamma, arrays["free_cholesky"], ledger)
        before = operator.calls; W_before = ledger.data["preconditioner_calls"]
        try:
            guards = coarse.algebra_guards(torch, W, operator, inputs, ledger)
        finally:
            counters.bump("new_coarse_guard_K_applications", operator.calls - before)
            counters.bump("fine_K_applications", operator.calls - before, persist=True)
            counters.bump("preconditioner_applications", ledger.data["preconditioner_calls"] - W_before, persist=True)
        cache = folder / "coarse_cache"; cache.mkdir(parents=True, exist_ok=True)
        records = {name: coarse.save_sparse(np, base, cache, name, value) for name, value in (("P", P), ("PT", PT), ("Ac", Ac))}
        records["diagonal"] = base._save_array(np, cache / "diagonal.npy", D)
        records["coarse_original_node_ids"] = base._save_array(np, cache / "corners.npy", corners)
        timer.memory(torch)
        committed = {"status": "completed", "config": config, "mapping": mapping, "assembly": assembly,
            "algebra_guards": guards, "arrays": records, "checks_repeated_on_cache_load": False}
        write(coarse_file, committed)
    return operator, W, arrays, manifest, ledger


def physical_solve(np, torch, base, operator, W, inputs, folder, timer, counters, config):
    timer.phase("solve", 300)
    free = inputs["free_node_ids"]; b = inputs["force"][free]; norm_b = float(torch.linalg.vector_norm(b))
    if not math.isfinite(norm_b) or norm_b <= 0:
        raise Negative("Zero or nonfinite force norm; no automatic accuracy pass")
    checkpoint_file = folder / "checkpoint.json"
    def apply(value):
        timer.memory(torch); counters.bump("fine_K_applications")
        full = torch.zeros_like(inputs["vertices_mm"]); full[free] = value
        result = operator.apply(full)[free]; timer.guard(); return result
    if checkpoint_file.exists():
        state = read(checkpoint_file); require(state["config"] == config, "Own PCG resume identity differs")
        counters.bump("own_PCG_checkpoint_reads", persist=True)
        vector_file = path(state["vectors_path"]); require(sha(vector_file) == state["vectors_sha256"], "Own PCG vectors changed")
        with np.load(vector_file, allow_pickle=False) as values:
            z, r, p = [torch.as_tensor(values[name], device=b.device, dtype=torch.float64) for name in ("z", "r", "p")]
        iteration = state["iteration"]; rz = state["rz"]; true = state["true_relative_residual"]
        require(all(v.shape == b.shape and torch.isfinite(v).all() for v in (z, r, p)), "Own PCG vector schema differs")
        # Resume uses the committed p/r recurrence; no warm guess or reset.
    else:
        z, r = torch.zeros_like(b), b.clone()
        counters.bump("preconditioner_applications"); p = W.apply(r)
        rz = float((r * p).sum()); iteration = 0; true = 1.
    def checkpoint(converged=False):
        timer.account(); counters.bump("own_PCG_checkpoint_writes", persist=True)
        name = folder / ("pcg_vectors_%06d.npz" % iteration)
        temporary = name.with_name(name.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, z=z.detach().cpu().numpy(), r=r.detach().cpu().numpy(), p=p.detach().cpu().numpy())
        os.replace(temporary, name)
        state = {"config": config, "iteration": iteration, "rz": rz, "true_relative_residual": true,
            "converged": converged, "vectors_path": str(name), "vectors_sha256": sha(name)}
        write(checkpoint_file, state); timer.persist(); return state
    converged = bool(true is not None and true <= 1e-8 and iteration > 0)
    while not converged and iteration < 10000 and counters.data["solve_iterations"] < 10000:
        timer.guard()
        if not math.isfinite(rz) or rz <= 0:
            raise Negative("PCG nonpositive rTWr; no shift/fallback")
        Ap = apply(p); pAp = float((p * Ap).sum())
        if not math.isfinite(pAp) or pAp <= 0:
            raise Negative("PCG nonpositive pTAp; no method change")
        alpha = rz / pAp; z = z + alpha * p; r = r - alpha * Ap
        iteration += 1; counters.bump("solve_iterations")
        recurrence = float(torch.linalg.vector_norm(r)) / norm_b
        if iteration % 25 == 0 or recurrence <= 1e-8 or iteration == 10000:
            verified = b - apply(z); true = float(torch.linalg.vector_norm(verified)) / norm_b
            if not math.isfinite(true):
                raise Negative("PCG true residual nonfinite")
            converged = true <= 1e-8
        if not converged:
            counters.bump("preconditioner_applications"); w = W.apply(r)
            next_rz = float((r * w).sum())
            if not math.isfinite(next_rz) or next_rz <= 0:
                raise Negative("PCG next rTWr nonpositive")
            p = w + (next_rz / rz) * p; rz = next_rz
        if iteration % 25 == 0 or converged or iteration == 10000:
            checkpoint(converged)
    if not converged:
        raise Negative("Fixed maximum iteration budget, true convergence not attained")
    # Endpoint true fine residual is real and charged, not borrowed from state.
    true = float(torch.linalg.vector_norm(b - apply(z))) / norm_b
    if not math.isfinite(true) or true > 1e-8:
        raise Negative("Endpoint true fine residual failed")
    full = base._restore_full(torch, z, inputs)
    if not torch.isfinite(full).all():
        raise Negative("Full original-node endpoint nonfinite")
    checkpoint(True); counters.persist(); return full.detach().cpu().numpy(), true, iteration


def physical_request(np, torch, data, contract, plan, key, folder, timer, counters, config):
    from . import elastic_pilot as base, elastic_coarse as coarse
    from . import elastic_continuous_capability as continuous, continuous_query_geometry as geometry
    timer.phase("input_operator_coarse", 300); before = helper_counts(folder / "input")
    record = data.prepare_case(PLAN if config["plan"] is None else config["plan"], key,
        contract=contract["input_contract"], guard=timer.guard, budget_seconds=300,
        output=folder / "input")
    counters.absorb(helper_counts(folder / "input"), before); write(folder / "input_manifest.json", record)
    if record.get("status") != "completed" or record.get("feasibility_pass") is not True:
        raise Negative("Target-free physical input unsupported: " + str(record.get("reason", record.get("negative_reason"))))
    before = helper_counts(folder / "input")
    inputs = data.load_input(record, "cuda:0", guard=timer.guard, observer=counters.observer)
    counters.absorb(inputs.get("_actual_read_receipt", {}), before)
    length = float(inputs["characteristic_length_mm"])
    local_plan = {"recipe": plan["science"]["physical_recipe"], "expected_geometry": {
        "nodes": len(inputs["vertices_mm"]), "cells": len(inputs["cells10"]), "characteristic_length_mm": length}}
    operator, W, arrays, operator_meta, coarse_ledger = physical_prepare(np, torch, base, coarse, inputs,
        local_plan, folder, timer, counters, config)
    nodal, true, iterations = physical_solve(np, torch, base, operator, W, inputs, folder, timer, counters, config)
    timer.phase("queries_and_metrics", 60)
    Urecord = array_record(np, folder / "node_U_mm.npy", nodal)
    node_manifest = {"status": "completed", "config": config, "arm": ARMS[0], "case_key": key,
        "all_original_nodes": True, "node_count": len(nodal), "prediction": Urecord,
        "true_fine_relative_residual": true, "true_converged": True, "iterations": iterations,
        "units": "mm", "GT_opened": False}
    write(folder / "prediction_manifest.json", node_manifest)
    query_inputs = {"cells": inputs["cells10"], "gradients": arrays["gradients"], "weights": arrays["weights"],
        "X": inputs["vertices_mm"]}
    counters.bump("native_query_cases", persist=True)
    full = continuous.query_full(np, torch, geometry, query_inputs, nodal, timer, include_X=True)
    records = {name: array_record(np, folder / ("fourQ_" + name + ".npy"), full[name]) for name in ("X_mm", "U_mm", "gradient_dimensionless")}
    cpu_inputs = {"vertices_mm": inputs["vertices_mm"].detach().cpu().numpy(),
        "cells10": inputs["cells10"].detach().cpu().numpy(), "L_mm": length}
    small, diagnostics = continuous.small_queries(np, geometry, cpu_inputs, nodal, timer)
    if not diagnostics["C0_passed"] or not diagnostics["inverse_replay_passed"]:
        raise Negative("Fixed native query geometry/inverse/C0 policy failed")
    small_record = continuous.save_small(np, folder / "small_queries.npz", small)
    write(folder / "geometry_diagnostics.json", {"status": "completed", "config": config,
        "operator_geometry": operator_meta["geometry"], "native_queries": diagnostics,
        "orientation": "Accepted coherent nonzero signed orientation; absdet quadrature weights"})
    query_manifest = {"status": "completed", "config": config, "case_key": key, "all_cells": len(inputs["cells10"]),
        "quadrature_points": 4 * len(inputs["cells10"]), "all_native_complete": True,
        "arrays": records, "small_queries": small_record, "weights": operator_meta["arrays"]["weights"],
        "gradients": operator_meta["arrays"]["gradients"], "input_record": record, "GT_opened": False,
        "units": {"X": "mm", "U": "mm", "G": "dimensionless"}, "proxy_is_source_strain_GT": False}
    write(folder / "query_manifest.json", query_manifest); timer.memory(torch); counters.persist()
    return {"node_prediction": node_manifest, "native_prediction": query_manifest,
        "coarse_work": coarse_ledger.data,
        "coarse_work_scope": "accepted coarse-helper calls only; complete fine/iteration counts are native read_receipt",
        "true_fine_relative_residual": true}


def frozen_FNO_request(np, torch, data, contract, plan, key, folder, timer, counters, config):
    from .solver_selection_models import build_selection_model
    from .solver_selection_data import query_grid
    timer.phase("FNO_request", 120); before = helper_counts(folder / "input")
    record = data.prepare_case(config["plan"], key, contract=contract["input_contract"], guard=timer.guard,
        budget_seconds=120, output=folder / "input")
    counters.absorb(helper_counts(folder / "input"), before); write(folder / "input_manifest.json", record)
    if record.get("status") != "completed" or record.get("feasibility_pass") is not True:
        raise Negative("Exact target-free25 FNO input unsupported: " + str(record.get("reason", record.get("negative_reason"))))
    before = helper_counts(folder / "input")
    inputs = data.load_input(record, "cuda:0", for_fno=True, guard=timer.guard, observer=counters.observer)
    counters.absorb(inputs.get("_actual_read_receipt", {}), before)
    endpoint = plan["science"]["root_recipe"]["FNO_reference"]; checkpoint = path(endpoint["checkpoint"]["path"])
    info = checkpoint.stat(); require(info.st_size == endpoint["checkpoint"]["bytes"] and info.st_mtime_ns == endpoint["checkpoint"]["mtime_ns"], "Frozen FNO checkpoint stat differs")
    counters.bump("FNO_checkpoint_byte_hash_reads", persist=True); checkpoint_sha = sha(checkpoint); timer.guard()
    counters.bump("FNO_checkpoint_read_attempts", persist=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    counters.bump("FNO_checkpoint_reads", persist=True)
    require(payload["step"] == 2880 and payload["config"] == read(plan["science"]["FNO_config"]["path"]), "Frozen FNO endpoint/config differs")
    model = build_selection_model("fno", "cuda:0", 2026)
    require(model.source_config == plan["science"]["FNO_model_source_config"]
        and model.parameter_counts == plan["science"]["FNO_parameter_counts"], "Frozen FNO source/schema/counts differ")
    state = payload["model"]
    require(isinstance(state, dict), "Frozen checkpoint model payload absent")
    model.load_state_dict(state, strict=True); model.eval(); del payload
    timer.memory(torch); counters.bump("FNO_inference_requests", persist=True)
    with torch.inference_mode():
        context = model.encode_geometry(inputs["cloud"], inputs["mask"])
        grid = model.forward_grid(inputs["grid"], context)
        # The model internally emits four encoded channels.  Only U is decoded.
        encoded_U = query_grid(grid, inputs)[:, :3].double()
        mean = torch.tensor(endpoint["frozen_scaler"]["mean"][:3], dtype=torch.float64, device="cuda")
        std = torch.tensor(endpoint["frozen_scaler"]["std"][:3], dtype=torch.float64, device="cuda")
        U = (encoded_U * std + mean).cpu().numpy()
    if U.shape != (len(inputs["xyz"]), 3) or not np.isfinite(U).all():
        raise Negative("Frozen FNO full original-node U nonfinite/incomplete")
    prediction = array_record(np, folder / "node_U_mm.npy", U)
    manifest = {"status": "completed", "config": config, "arm": ARMS[1], "case_key": key,
        "all_original_nodes": True, "node_count": len(U), "prediction": prediction,
        "checkpoint_sha256": checkpoint_sha, "checkpoint_step": 2880, "units": "mm", "GT_opened": False,
        "native_gradient_available": False, "S_decoded_or_persisted": False}
    write(folder / "prediction_manifest.json", manifest); timer.memory(torch)
    return {"node_prediction": manifest, "checkpoint_sha256": checkpoint_sha}


def request_child(plan_path, output, arm, key, allowance):
    contract, plan = request_contract(plan_path); release(contract, plan)
    require(arm in ARMS and key in CASES, "Child inventory differs")
    folder = child_dir(output, arm, key); folder.mkdir(parents=True, exist_ok=True)
    config = {"contract": contract, "stage": "blind_U_request", "arm": arm, "case_key": key, "plan": str(path(plan_path))}
    save_config(folder, config)
    counters = Counter(folder); maximum = 120 if arm == ARMS[1] else 2400
    timer = Clock(folder, maximum, allowance)
    status, negative, result = "completed", None, None
    try:
        timer.guard()
        import numpy as np
        import torch
        from . import blind_solver_inputs as data
        torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True; torch.cuda.init(); torch.cuda.reset_peak_memory_stats()
        timer.memory(torch)
        function = physical_request if arm == ARMS[0] else frozen_FNO_request
        result = function(np, torch, data, contract, plan, key, folder, timer, counters, config)
    except Exception as error:
        # Accepted scientific math helpers expose ScientificNegative; budgets,
        # unsupported and CUDA backend failures terminate this fixed candidate.
        scientific = isinstance(error, Negative) or error.__class__.__name__ in ("ScientificNegative", "UnsupportedInput", "UncommittedTargetRead")
        backend = "cuda" in str(error).lower() or "out of memory" in str(error).lower()
        status = "completed" if scientific or backend else "failed"
        negative = type(error).__name__ + ": " + str(error)
    finally:
        # Preparation failures retain the helper's attempted/completed ledger.
        input_receipt = helper_counts(folder / "input")
        for name in ("raw_h5_input_reads", "raw_deck_input_reads", "raw_CSV_coordinate_reads", "input_cache_reads", "input_member_reads"):
            counters.data[name] = max(counters.data[name], input_receipt.get(name, 0))
        counters.persist(); timer.finish()
    summary = {"status": status, "kind": KIND, "config": config, "arm": arm, "case_key": key,
        "prediction_complete": result is not None, "terminal_negative_reason": negative,
        "result": result, "read_receipt": counters.data,
        "input_read_events": {name: input_receipt.get(name, {}) for name in ("attempted_reads", "completed_reads")},
        "costs": {"child_cumulative_wall_seconds": timer.data["cumulative_wall_seconds"],
            "phase_wall_seconds": timer.data["phase_wall_seconds"], "peak_GPU_bytes": timer.data["peak_GPU_bytes"],
            "raw_to_field_request_start": "parent_launch", "parent_observed_exit_required": True,
            "acquisition_and_upstream_meshing_excluded": True}}
    write(folder / "costs.json", summary["costs"]); write(folder / "summary.json", summary)
    write(folder / "status.json", {"status": status, "prediction_complete": summary["prediction_complete"], "terminal_negative_reason": negative})
    return summary


def metric_values(np, prediction, reference, weights=None):
    require(prediction.shape == reference.shape and np.isfinite(prediction).all() and np.isfinite(reference).all(), "Metric field schema differs")
    error = prediction - reference
    axes = tuple(range(1 if weights is None else 2, prediction.ndim))
    e2 = np.square(error).sum(axis=axes); y2 = np.square(reference).sum(axis=axes)
    if weights is None:
        numerator, denominator = float(e2.sum()), float(y2.sum())
        rms = math.sqrt(float(e2.mean())); maximum = math.sqrt(float(e2.max()))
    else:
        require(weights.shape == e2.shape and np.isfinite(weights).all() and (weights > 0).all(), "Metric positive volume weights differ")
        numerator, denominator = float((e2 * weights).sum()), float((y2 * weights).sum())
        rms = math.sqrt(numerator / float(weights.sum())); maximum = math.sqrt(float(e2.max()))
    relative = math.sqrt(numerator / denominator) if denominator > 0 else None
    return {"relative_l2": relative, "absolute_RMS": rms, "absolute_max": maximum,
        "zero_reference_relative_is_null": denominator == 0}


def metric_child(plan_path, output, key, allowance):
    contract, plan = request_contract(plan_path); release(contract, plan)
    folder = path(output) / "metrics" / key.replace("/", "_"); folder.mkdir(parents=True, exist_ok=True)
    config = {"contract": contract, "stage": "blind_U_postbarrier_metrics", "case_key": key}
    save_config(folder, config); counters = Counter(folder)
    physical_dir = child_dir(output, ARMS[0], key)
    physical_cost = read(physical_dir / "summary.json")["costs"]["phase_wall_seconds"].get("queries_and_metrics", 0.)
    label_meta = read(path(output) / "label_cache" / (key.replace("/", "_") + ".json"))
    GT_cost = label_meta["shared_source_U_read_and_cache_wall_seconds"]
    timer = Clock(folder, max(0., 60 - physical_cost - GT_cost), allowance)
    status, negative, result = "completed", None, None
    input_helper_before, input_helper_folder, helper_absorbed = None, None, False
    try:
        timer.phase("metrics", max(0., 60 - physical_cost - GT_cost)); timer.guard()
        barrier = read(path(output) / "prediction_barrier.json")
        require(barrier["passed"] is True and barrier["input_contract_sha256"] == contract["input_contract_sha256"], "Metric barrier differs")
        import numpy as np
        import torch
        from . import elastic_pilot as base, elastic_continuous_capability as continuous, continuous_query_geometry as geometry
        torch.cuda.init(); torch.cuda.reset_peak_memory_stats(); timer.memory(torch)
        GTmeta = read(path(output) / "label_cache" / (key.replace("/", "_") + ".json"))
        truth = load_owned(np, GTmeta["prediction"], counters, GT=True)
        physical = read(physical_dir / "prediction_manifest.json")
        fno = read(child_dir(output, ARMS[1], key) / "prediction_manifest.json")
        pu = load_owned(np, physical["prediction"], counters); fu = load_owned(np, fno["prediction"], counters)
        native = read(physical_dir / "query_manifest.json")
        pU = load_owned(np, native["arrays"]["U_mm"], counters)
        pG = load_owned(np, native["arrays"]["gradient_dimensionless"], counters)
        # Only committed candidate outputs are loaded.  Derive the reference
        # on the identical basis/gradients; no new candidate solve/recovery.
        from . import blind_solver_inputs as data
        input_helper_folder = path(native["input_record"]["path"]).parent
        before = helper_counts(input_helper_folder); input_helper_before = before
        inputs = data.load_input(native["input_record"], "cuda:0", guard=timer.guard, observer=counters.observer)
        counters.absorb(inputs.get("_actual_read_receipt", {}), before)
        helper_absorbed = True
        gradients = base._load_array(np, torch, native["gradients"], torch.device("cuda:0"))
        weights = base._load_array(np, torch, native["weights"], torch.device("cuda:0"))
        ref = continuous.query_full(np, torch, geometry, {"cells": inputs["cells10"], "gradients": gradients,
            "weights": weights, "X": inputs["vertices_mm"]}, truth, timer, include_X=False)
        w = weights.cpu().numpy()
        result = {"physics_full_FE": {"node_U": metric_values(np, pu, truth),
            "native_P2_U": metric_values(np, pU, ref["U_mm"], w),
            "native_P2_G": metric_values(np, pG, ref["gradient_dimensionless"], w),
            "reference": "same TET10 P2 proxy derived from original source node U, not source strain GT"},
            "FNO_frozen2880": {"node_U": metric_values(np, fu, truth), "native_gradient_available": False}}
        write(folder / "reference_manifest.json", {"status": "completed", "config": config,
            "GT_node_U": GTmeta["prediction"], "derivation": "same frozen physical fourQ basis/gradients",
            "units": {"U": "mm", "G": "dimensionless"}, "source_strain_GT": False})
        timer.memory(torch)
    except Exception as error:
        scientific = isinstance(error, Negative) or error.__class__.__name__ in ("ScientificNegative", "UnsupportedInput", "UncommittedTargetRead")
        backend = "cuda" in str(error).lower() or "out of memory" in str(error).lower()
        status = "completed" if scientific or backend else "failed"
        negative = type(error).__name__ + ": " + str(error)
    finally:
        if input_helper_before is not None and not helper_absorbed:
            counters.absorb(helper_counts(input_helper_folder), input_helper_before)
        counters.persist(); timer.finish()
    summary = {"status": status, "kind": KIND, "config": config, "case_key": key, "metrics": result,
        "terminal_negative_reason": negative, "read_receipt": counters.data,
        "costs": {"metric_cumulative_wall_seconds": timer.data["cumulative_wall_seconds"],
            "candidate_query_seconds": physical_cost, "shared_source_U_read_and_cache_wall_seconds": GT_cost,
            "combined_query_metric_seconds": physical_cost + GT_cost + timer.data["cumulative_wall_seconds"],
            "peak_GPU_bytes": timer.data["peak_GPU_bytes"]}}
    write(folder / "metrics.json", result); write(folder / "costs.json", summary["costs"])
    write(folder / "summary.json", summary); write(folder / "status.json", {"status": status, "terminal_negative_reason": negative})
    return summary


def launch(plan_path, output, timer, arguments, directory, process_cap):
    """Real process cold wall; incomplete observed exit is never fabricated."""
    directory.mkdir(parents=True, exist_ok=True); marker = directory / "launch_receipt.json"
    previous = read(marker) if marker.exists() else {"attempts": [], "completed_observed_wall_seconds": 0.0}
    # An unclosed process attempt has already been charged by the campaign
    # active wall journal; retain unknown cold cost rather than invent an exit.
    attempt = {"launch_epoch": time.time(), "exit_epoch": None, "observed_wall_seconds": None,
        "pid": None, "returncode": None, "complete_observed_exit": False}
    previous["attempts"].append(attempt); write(marker, previous); timer.persist()
    command = [sys.executable, "-m", "geobc.blind_solver_service", "--plan", str(path(plan_path)),
        "--output", str(path(output)), "--allowance", str(min(process_cap, timer.maximum - timer.data["cumulative_wall_seconds"])), *arguments]
    start = time.monotonic()
    with (directory / "runtime.log").open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        attempt["pid"] = process.pid; write(marker, previous)
        expired = False
        while process.poll() is None:
            time.sleep(.2); timer.persist()
            remaining = timer.maximum - timer.data["cumulative_wall_seconds"]
            if time.monotonic() - start >= process_cap or remaining <= 0:
                expired = True; process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=5)
                break
        wall = time.monotonic() - start; timer.persist()
    attempt.update(exit_epoch=time.time(), observed_wall_seconds=wall, returncode=process.returncode,
        complete_observed_exit=True, fixed_cap_terminated=expired)
    previous["completed_observed_wall_seconds"] += wall; write(marker, previous)
    if expired:
        raise Negative("Fixed child process/campaign cap exhausted; retained partial attempt")
    if process.returncode != 0 or not (directory / "summary.json").exists():
        raise RuntimeError("Independent child failed before committed summary: " + str(directory))
    result = read(directory / "summary.json")
    if result["status"] == "failed":
        raise RuntimeError("Child technical failure retained: " + str(result.get("terminal_negative_reason")))
    result["costs"]["fresh_request_wall_seconds"] = previous["completed_observed_wall_seconds"] if all(x["complete_observed_exit"] for x in previous["attempts"]) else None
    result["costs"]["parent_launch_to_observed_exit_seconds"] = wall
    result["costs"]["complete_cold_cost_accounted"] = result["costs"]["fresh_request_wall_seconds"] is not None
    write(directory / "summary.json", result); write(directory / "costs.json", result["costs"])
    return result


def commit_barrier(output, contract, requests):
    nodes, native, weight_sha = {}, {}, None
    for arm in ARMS:
        for key in CASES:
            token = arm + ":" + key; request = requests[token]
            require(request["prediction_complete"] is True, "Uncommitted request cannot open GT barrier")
            folder = child_dir(output, arm, key); node_path = folder / "prediction_manifest.json"
            node = read(node_path); require(node["status"] == "completed" and node["all_original_nodes"] is True
                and node["config"]["contract"] == contract and node["arm"] == arm and node["case_key"] == key, "Node commit differs")
            nodes[token] = {"path": str(node_path), "sha256": sha(node_path)}
            if arm == ARMS[0]:
                require(node["true_converged"] is True and node["true_fine_relative_residual"] <= 1e-8, "Physical fine convergence missing")
                native_path = folder / "query_manifest.json"; query = read(native_path)
                require(query["all_native_complete"] is True and query["quadrature_points"] == 4 * query["all_cells"]
                    and query["config"]["contract"] == contract and query["case_key"] == key, "Native full4Q commit missing")
                native[key] = {"path": str(native_path), "sha256": sha(native_path)}
            else:
                physical_dir = child_dir(output, ARMS[0], key)
                physical_node = read(physical_dir / "prediction_manifest.json")
                require(node["node_count"] == physical_node["node_count"], "Both-arm fullN node count differs")
                physical_input = read(physical_dir / "input_manifest.json")
                FNO_input = read(folder / "input_manifest.json")
                for name in ("vertices_mm", "cells10", "grid_node_ids", "h5_row_to_csv"):
                    require(physical_input["arrays"][name] == FNO_input["arrays"][name], "Both-arm original node/grid identity differs: " + name)
                actual = node["checkpoint_sha256"]
                require(weight_sha is None or actual == weight_sha, "Frozen FNO checkpoint bytes changed between requests")
                weight_sha = actual
    barrier = {"status": "completed", "passed": True, "case_keys": list(CASES), "arms": list(ARMS),
        "all10_node_U_predictions_committed": True, "all5_native_complete": True,
        "input_contract_sha256": contract["input_contract_sha256"], "node_prediction_manifests": nodes,
        "native_prediction_manifests": native, "FNO_checkpoint_sha256": weight_sha,
        "source_U_target_parses_before_barrier": 0, "stress_target_reads": 0}
    write(path(output) / "prediction_barrier.json", barrier); return barrier


def gates(cases, barrier, negative):
    def below(value, limit):
        return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 and value < limit
    readout = barrier and all(cases[key].get("metrics") is not None for key in CASES)
    physical = readout and all(below(cases[key]["metrics"][ARMS[0]][name]["relative_l2"], limit)
        for key in CASES for name, limit in (("node_U", .01), ("native_P2_U", .01), ("native_P2_G", .05)))
    fno = readout and all(below(cases[key]["metrics"][ARMS[1]]["node_U"]["relative_l2"], .01) for key in CASES)
    return {"passed": bool(physical), "prediction_barrier_complete": bool(barrier), "GT_U_readout_complete": bool(readout),
        "physical_accuracy_gate": bool(physical), "FNO_node_U_gate": bool(fno), "U_only": True,
        "final_solver_selection": False, "fixed_denominator": 5, "terminal_negative_reason": negative}


def write_report(output, summary):
    gate = summary["completion_gate"]
    lines = ["# Blind U-only service comparison", "", "Five fixed new CAD IDs; original DeepJEB/ver. S labels remain sealed.",
        "Full FE and point-cloud FNO interfaces differ; this is service evidence, not architecture causality or a full GeoBC selection.",
        "Fresh request cost is independent parent launch-to-observed-exit; acquisition and upstream meshing excluded.",
        "Fresh process/app state; operating-system disk page cache is not reset.",
        "", "| Route | Node U mean / worst | Fresh cold total s |", "|---|---:|---:|"]
    for arm in ARMS:
        a = summary["arms"][arm]; aggregate = a["aggregate"]
        values = [aggregate["node_U_relative_l2_mean"], aggregate["node_U_relative_l2_worst"]]
        accuracy = "unavailable" if any(v is None for v in values) else " / ".join("%.3f%%" % (100 * v) for v in values)
        cold = a["costs"]["fresh_request_wall_seconds_sum"]
        lines.append("| %s | %s | %s |" % (arm, accuracy, "unavailable" if cold is None else "%.3f" % cold))
    lines += ["", "Physical node/P2-U <1%, P2-G <5% each-case gate: " + str(gate["physical_accuracy_gate"]),
        "FNO node-U <1% each-case descriptive gate: " + str(gate["FNO_node_U_gate"]),
        "Full prediction barrier / readout: %s / %s" % (gate["prediction_barrier_complete"], gate["GT_U_readout_complete"]),
        "P2-G reference is node-U-derived proxy, not source strain GT. FNO has no native gradient result.",
        "Terminal reason: " + str(gate["terminal_negative_reason"])]
    (path(output) / "BRIEF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    table_start = next(i for i, line in enumerate(lines) if line.startswith("| Route"))
    (path(output) / "metrics.md").write_text("\n".join(lines[table_start:table_start + 4]) + "\n", encoding="utf-8")


def run(plan_path=PLAN, output=None):
    contract, plan = request_contract(plan_path); release(contract, plan)
    output = path(output or plan["output"]); output.mkdir(parents=True, exist_ok=True)
    config = {"contract": contract}; save_config(output, config)
    if (output / "summary.json").exists():
        saved = read(output / "summary.json")
        if saved["status"] == "completed":
            require(saved["config"] == config, "Completed own campaign identity differs"); return saved
    timer = Clock(output, 2400); requests = {}; cases = {key: {"status": "not_attempted", "metrics": None} for key in CASES}
    negative, status, barrier, labels = None, "completed", False, receipt()
    try:
        for arm in ARMS:
            for key in CASES:
                timer.guard(); folder = child_dir(output, arm, key); token = arm + ":" + key
                if (folder / "summary.json").exists() and read(folder / "summary.json")["status"] == "completed":
                    result = read(folder / "summary.json")
                    require(result["config"]["contract"] == contract, "Completed own request identity differs")
                else:
                    maximum = 120 if arm == ARMS[1] else timer.maximum - timer.data["cumulative_wall_seconds"]
                    result = launch(plan_path, output, timer, ["--child-arm", arm, "--case", key], folder, maximum)
                requests[token] = result
                write(output / "status.json", {"status": "running", "phase": "requests", "completed_requests": len(requests),
                    "request": token, "cumulative_wall_seconds": timer.data["cumulative_wall_seconds"]})
                if not result["prediction_complete"]:
                    raise Negative(result["terminal_negative_reason"] or "Incomplete fixed request")
        timer.guard(); proof = commit_barrier(output, contract, requests); barrier = True
        from . import blind_solver_inputs as data
        import numpy as np
        label_dir = output / "label_cache"; label_dir.mkdir(parents=True, exist_ok=True)
        # All five candidates exist before shared original U reads.  The helper
        # claims source reads first and selectively parses exactly U columns.
        for key in CASES:
            timer.guard(); record = read(child_dir(output, ARMS[0], key) / "input_manifest.json")
            meta_file = label_dir / (key.replace("/", "_") + ".json")
            if meta_file.exists():
                saved_label = read(meta_file)
                require(saved_label["case_key"] == key and saved_label["source_U_only"] is True,
                    "Committed shared U metadata differs")
                continue
            case_labels = label_dir / key.replace("/", "_")
            query_cost = requests[ARMS[0] + ":" + key]["costs"]["phase_wall_seconds"].get("queries_and_metrics", 0.)
            label_clock = Clock(case_labels, max(0., 60 - query_cost), timer.maximum - timer.data["cumulative_wall_seconds"])
            def label_guard():
                timer.guard(); label_clock.guard()
            try:
                label_guard()
                truth = data.read_U(record, proof, contract=contract["input_contract"], output=case_labels, guard=label_guard)
                own = array_record(np, label_dir / (key.replace("/", "_") + ".npy"), truth)
                label_guard()
            finally:
                label_clock.finish()
            helper_meta = {"helper_manifest": {"path": str(case_labels / "manifest.json"), "sha256": sha(case_labels / "manifest.json")},
                "prediction": own, "case_key": key, "source_U_only": True, "source_strain_GT": False,
                "shared_source_U_read_and_cache_wall_seconds": label_clock.data["cumulative_wall_seconds"]}
            write(meta_file, helper_meta)
        labels = merge_receipts([helper_counts(label_dir / key.replace("/", "_")) for key in CASES])
        for key in CASES:
            timer.guard(); folder = output / "metrics" / key.replace("/", "_")
            if (folder / "summary.json").exists() and read(folder / "summary.json")["status"] == "completed":
                metric = read(folder / "summary.json"); require(metric["config"]["contract"] == contract, "Completed own metric identity differs")
            else:
                metric = launch(plan_path, output, timer, ["--metric-case", key], folder,
                    min(60., timer.maximum - timer.data["cumulative_wall_seconds"]))
            cases[key] = metric
            if metric["metrics"] is None:
                raise Negative(metric["terminal_negative_reason"] or "Missing fixed metric")
    except Exception as error:
        status = "completed" if isinstance(error, Negative) or error.__class__.__name__ in ("UnsupportedInput", "UncommittedTargetRead") else "failed"
        negative = type(error).__name__ + ": " + str(error)
    finally:
        # Recover partial request/metric receipts even when an attempt failed.
        for arm in ARMS:
            for key in CASES:
                p = child_dir(output, arm, key) / "summary.json"
                if p.exists() and arm + ":" + key not in requests:
                    requests[arm + ":" + key] = read(p)
        labels = merge_receipts([helper_counts(output / "label_cache" / key.replace("/", "_")) for key in CASES])
        for key in CASES:
            claim = output / "label_cache" / key.replace("/", "_") / "source_U_read_started.json"
            if claim.exists():
                write(output / "label_cache" / (key.replace("/", "_") + "_read_started.json"), read(claim))
        timer.persist()
    complete_receipts = [labels]
    for arm in ARMS:
        for key in CASES:
            folder = child_dir(output, arm, key); p = folder / "read_receipt.json"
            if p.exists():
                value = read(p)
                child = requests.get(arm + ":" + key)
                if child is None or child.get("status") != "completed":
                    helper = helper_counts(folder / "input")
                    for name in ("raw_h5_input_reads", "raw_deck_input_reads", "raw_CSV_coordinate_reads", "input_cache_reads", "input_member_reads"):
                        value[name] = max(value[name], helper.get(name, 0))
                complete_receipts.append(value)
    for key in CASES:
        folder = output / "metrics" / key.replace("/", "_"); p = folder / "read_receipt.json"
        if p.exists():
            complete_receipts.append(read(p))
    total_receipt = merge_receipts(complete_receipts)
    all_requests = {}
    arms = {}
    for arm in ARMS:
        members = {}; observed = []; accuracies = []; observed_count = 0
        for key in CASES:
            token = arm + ":" + key; value = requests.get(token, {"status": "not_attempted", "prediction_complete": False,
                "costs": {"fresh_request_wall_seconds": None}, "read_receipt": receipt()})
            all_requests[token] = value; members[key] = value
            cold = value["costs"].get("fresh_request_wall_seconds"); observed.append(cold)
            observed_count += int(cold is not None and value.get("prediction_complete", False))
            if cases[key].get("metrics") is not None:
                accuracies.append(cases[key]["metrics"][arm]["node_U"]["relative_l2"])
        aggregate = {"node_U_relative_l2_mean": sum(accuracies) / 5 if len(accuracies) == 5 and all(x is not None for x in accuracies) else None,
            "node_U_relative_l2_worst": max(accuracies) if len(accuracies) == 5 and all(x is not None for x in accuracies) else None,
            "fixed_denominator": 5}
        arms[arm] = {"cases": members, "aggregate": aggregate, "costs": {
            "fresh_request_wall_seconds_sum": sum(observed) if all(x is not None for x in observed) else None,
            "observed_completed_requests": observed_count, "fresh_cold_scope": "raw input to persisted node U and required physical native queries; independent process",
            "offline_training_seconds": plan["science"]["FNO_offline_training_seconds"] if arm == ARMS[1] else 0,
            "acquisition_and_upstream_meshing_excluded": True}, "information_interface": "full TET10/material/real constraints/load" if arm == ARMS[0] else "canonical xyz/original25/whole-CAD cloud"}
    gate = gates(cases, barrier, negative)
    summary = {"status": status, "kind": KIND, "config": config, "case_keys": list(CASES), "cases": cases,
        "arms": arms, "requests": all_requests, "completion_gate": gate, "read_receipt": total_receipt,
        "costs": {"cumulative_campaign_wall_seconds": timer.data["cumulative_wall_seconds"],
            "campaign_cap_seconds": 2400, "unknown_gap_charged_seconds": timer.data["unknown_gap_charged_seconds"],
            "finalization_overshoot_seconds": max(0., timer.data["cumulative_wall_seconds"] - 2400),
            "fresh_combined_measurement": True}, "interpretation": "U-only unseen-ID service evidence; information interfaces differ; all new S sealed; final solver not selected"}
    write(output / "read_receipt.json", total_receipt); write(output / "metrics.json", {k: v.get("metrics") for k, v in cases.items()})
    write(output / "summary.json", summary); write(output / "status.json", {"status": status, "completion_gate": gate})
    if not (output / "prediction_barrier.json").exists():
        write(output / "prediction_barrier.json", {"status": "completed", "passed": False, "case_keys": list(CASES),
            "arms": list(ARMS), "all10_node_U_predictions_committed": False, "all5_native_complete": False,
            "source_U_target_parses_before_barrier": 0, "terminal_negative_reason": negative})
    write_report(output, summary)
    timer.finish()
    summary["costs"]["cumulative_campaign_wall_seconds"] = timer.data["cumulative_wall_seconds"]
    summary["costs"]["finalization_overshoot_seconds"] = max(0., timer.data["cumulative_wall_seconds"] - 2400)
    write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--plan", default=PLAN); parser.add_argument("--output")
    parser.add_argument("--child-arm", choices=ARMS); parser.add_argument("--case", choices=CASES)
    parser.add_argument("--metric-case", choices=CASES); parser.add_argument("--allowance", type=float, default=2400)
    args = parser.parse_args()
    if args.child_arm:
        result = request_child(args.plan, args.output, args.child_arm, args.case, args.allowance)
    elif args.metric_case:
        result = metric_child(args.plan, args.output, args.metric_case, args.allowance)
    else:
        result = run(args.plan, args.output)
    print(json.dumps({"status": result["status"], "kind": KIND, "terminal_negative_reason": result.get("terminal_negative_reason")}, ensure_ascii=False))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
