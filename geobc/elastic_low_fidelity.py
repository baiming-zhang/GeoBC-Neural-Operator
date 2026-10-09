"""Frozen direct masked-Galerkin coarse displacement screen, numerical GT free.

This changes the trial space to range(P). It does not change the accepted fine
equation, tune its preconditioner, recover scalar stress, or transfer GT errors.
Contracts read source/small metadata/stat only; scientific imports follow root
release. Immutable accepted Ac/P/PT are never reconstructed or guard-replayed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_low_fidelity_plan.json"
SOURCE = "geobc/elastic_low_fidelity.py"
KIND = "elastic_direct_coarse_screen"
CASE_KIND = "elastic_direct_coarse_case"
CASES = ("101_428/ver", "102_249/ver")
ANCHORS = (42.107864124, 36.492053894)
SOURCES = (SOURCE, "geobc/elastic_coarse.py", "geobc/elastic_crosscad.py",
           "geobc/elastic_pilot.py", "geobc/elastic_pilot_data.py",
           "geobc/continuous_query_geometry.py")
ZERO_KEYS = ("GT_reads", "displacement_reads", "stress_reads", "old_final_reads",
             "DEV_dia_reads", "sealed_new_test_reads", "optimizer_steps", "scaler_fits",
             "learned_weight_reads", "setup_K_guards_repeated",
             "inherited_fine_K_guards_repeated", "fine_operator_rebuilds")
SCIENCE_FIELDS = ("recipe", "comparison", "cost", "completion_gate", "read_policy")


def host_path(value):
    value = str(value).replace("\\", "/")
    if len(value) > 2 and value[1:3] == ":/":
        value = "/mnt/" + value[0].lower() + value[2:]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path):
    return json.loads(host_path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(host_path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def finite_cost(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def helpers():
    from . import elastic_coarse as core
    from . import elastic_pilot as base
    return core, base


def zero_receipt():
    return {**{key: 0 for key in ZERO_KEYS}, "numerical_reference_reads": 0,
            "accepted_coarse_cache_reads": 0, "fine_endpoint_K_applications": 0}


def verify_pin(record):
    require(sha(record["path"]) == record["sha256"], "Frozen metadata/source identity changed: " + record["path"])
    return read_json(record["path"])


def request_contract(plan_path=PLAN):
    """Metadata contracts/stat only; never execute old physical solve/GT APIs."""
    plan = read_json(plan_path)
    require(plan.get("kind") == KIND and tuple(plan.get("case_keys", ())) == CASES,
            "Direct coarse fixed TRAIN/DEV scope/order differs")
    recipe = plan["recipe"]
    require(recipe["device"] == "cuda:0" and recipe["dtype"] == "float64"
            and recipe["initial_guess"] == "zero"
            and recipe["preconditioner"] == "positive accepted scalar diagonal Jacobi"
            and recipe["maximum_iterations"] == 10000
            and recipe["maximum_cumulative_solve_wall_seconds_per_case"] == 60
            and recipe["maximum_cumulative_campaign_accounted_wall_seconds"] == 240
            and recipe["true_coarse_relative_residual_tolerance"] == 1e-8
            and recipe["true_residual_interval"] == recipe["checkpoint_interval"] == 25
            and recipe["maximum_actual_peak_GPU_bytes"] == 24 * 1024**3
            and recipe["resume_budget_reset"] is False
            and recipe["no_fallback_or_tuning"] is True
            and recipe["old_fine_eight_K_guards_repeated"] == 0
            and recipe["fine_apply_endpoint_only"] is True
            and recipe["persist_full_original_node_U_before_reference"] is True,
            "Frozen direct coarse solver/budget/read order differs")
    require(plan["comparison"]["numerical_U_relative_l2_maximum"] == .05
            and plan["comparison"]["full_original_nodes"] is True
            and plan["comparison"]["fine_true1e8_required"] is False
            and plan["comparison"]["old_GT_accuracy_assigned_to_new_endpoint"] is False
            and plan["completion_gate"]["inventory_denominator"] == 2
            and all(plan["completion_gate"][name] is True for name in
                ("both_coarse_true_converged", "both_fullN_numerical_U_l2_le_5percent",
                 "both_accounted_cold_le_half_accepted_coarse2", "cost_components_complete_required",
                 "missing_budget_backend_terminal_negative"))
            and all(plan["read_policy"].get(key) == 0 for key in ZERO_KEYS),
            "Frozen numerical comparison/gate/target-free policy differs")
    science = {name: plan[name] for name in SCIENCE_FIELDS}
    require(digest(science) == plan["science_recipe_sha256"], "Frozen scientific recipe digest differs")
    require(sha(plan["design"]["path"]) == plan["design"]["sha256"], "Frozen design source changed")
    parent_plan = verify_pin(plan["parent_plan"])
    review = verify_pin(plan["parent_root_review"])
    accepted = verify_pin(plan["parent_summary"])
    core, base = helpers()
    parent_contract, _ = core.request_contract(plan["parent_plan"]["path"])
    require(parent_plan["actual_execution_released"] is True and review["passed"] is True
            and review["source_sha256"] == parent_contract["source_sha256"]["geobc/elastic_coarse.py"]
            and review["science_recipe_sha256"] == digest(parent_contract["plan_science"])
            and review["base_evidence_decision"] == "proceed_frozen_P1_derived_coarse"
            and accepted["status"] == "completed" and accepted["kind"] == "elastic_p1_derived_coarse_screen"
            and accepted["config"]["contract"] == parent_contract
            and accepted["selection_gate"]["passed"] is True,
            "Accepted immutable coarse2 parent contract/root/math gates required")
    for key, anchor in zip(CASES, ANCHORS):
        record = plan["cases"][key]
        require(record["parent_record"] == parent_contract["records"][key], "Frozen original fine reference/input contract differs")
        child = verify_pin(record["accepted_case_summary"])
        cache = verify_pin(record["coarse_manifest"])
        require(child["status"] == cache["status"] == "completed"
                and child["config"] == cache["config"] == core.case_config(record["parent_record"], parent_contract)
                and child["numerically_converged"] is True and child["same_solution_passed"] is True
                and cache["kind"] == "elastic_p1_derived_coarse_cache"
                and cache["algebra_guards"]["passed"] is True
                and cache["algebra_guards"]["inherited_eight_fine_K_guards_repeated"] == 0
                and cache["arrays"] == record["coarse_arrays"] and cache["mapping"] == record["mapping"],
                "Accepted Galerkin/P/cache/true fine endpoint identities differ")
        fine_cost = record["parent_record"]["base_summary"]["inherited_input_operator_seconds"]
        setup = record["attributable_setup"]
        require(setup["fine_input_operator_seconds"] == fine_cost
                and setup["accepted_coarse_construction_seconds"] == child["costs"]["cumulative_coarse_setup_seconds"]
                and setup["total_seconds"] == fine_cost + setup["accepted_coarse_construction_seconds"]
                and setup["old_fine_iterative_solve_seconds_charged"] == 0
                and setup["old_numerical_reference_IO_seconds_charged"] == 0
                and finite_cost(setup["total_seconds"])
                and record["accepted_coarse_cold_seconds"] == child["costs"]["coarse_cold_accounted_wall_seconds"]
                and abs(record["accepted_coarse_cold_seconds"] - anchor) < 1e-8
                and record["gate_anchor_coarse_cold_seconds"] == anchor
                and record["maximum_candidate_accounted_cold_seconds"] == .5 * anchor,
                "Attributable setup/cold anchors differ or old solve/reference was charged")
        def stat_array(array):
            require(base.stat_record(array["path"]) == {name: array[name] for name in ("bytes", "mtime_ns")},
                    "Immutable accepted coarse array stat changed")
        for name, array in cache["arrays"].items():
            if name in ("P", "PT", "Ac"):
                for part in ("rowptr", "columns", "values"):
                    stat_array(array[part])
            else:
                stat_array(array)
    pins = {name: sha(name) for name in SOURCES}
    require(plan["source_sha256"] == pins, "Exactly six frozen API/source identities required")
    identity = {"kind": KIND, "case_keys": list(CASES), "plan_science": science,
                "source_sha256": pins, "records": plan["cases"], "parent_plan": plan["parent_plan"],
                "parent_root_review": plan["parent_root_review"], "parent_summary": plan["parent_summary"],
                "design": plan["design"], "output": str(host_path(plan["output"]))}
    return {"schema": "geobc_elastic_direct_coarse_v1", **identity,
            "source_hashes": pins, "scientific_digest": digest(identity),
            "science_recipe_sha256": digest(science), "metadata_only_array_reads": 0}, plan


class CampaignClock:
    """240s composed budget: old attributable setup + all actual resumed walls."""
    def __init__(self, output, config, began, inherited):
        self.output, self.config, self.began = output, config, began
        self.inherited, self.prior = inherited, 0.
        self.config_sha256 = digest(config)
        for name in ("status.json", "summary.json"):
            path = output/name
            if path.exists():
                old = read_json(path)
                require(old.get("config_sha256", digest(old.get("config"))) == self.config_sha256,
                        "Preserved campaign accounting belongs to another contract")
                costs = old.get("costs", old)
                self.prior = max(self.prior, costs.get("cumulative_actual_stage_wall_seconds", 0.))
        require(finite_cost(self.prior), "Invalid preserved failed-attempt wall accounting")

    def actual(self):
        return self.prior + time.monotonic() - self.began

    def elapsed(self):
        return self.inherited + self.actual()

    def remaining(self):
        return max(0., 240. - self.elapsed())

    def account(self, phase, key=None):
        _, base = helpers()
        base.write_json(self.output/"status.json", {"config_sha256": self.config_sha256,
            "phase": phase, "case": key, "cumulative_actual_stage_wall_seconds": self.actual(),
            "inherited_attributable_setup_seconds": self.inherited,
            "cumulative_campaign_accounted_wall_seconds": self.elapsed(), "maximum_accounted_wall_seconds": 240})


def case_config(record, contract):
    return {"stage": CASE_KIND, "contract": {"stage_contract_sha256": digest(contract),
        "case_key": record["parent_record"]["physical_record"]["key"],
        "source_sha256": contract["source_sha256"][SOURCE],
        "coarse_manifest_sha256": record["coarse_manifest"]["sha256"],
        "original_fine_reference_summary_sha256": record["parent_record"]["base_summary"]["sha256"]}}


def case_base(record, contract):
    physical = record["parent_record"]["physical_record"]
    return {"status": "completed", "kind": CASE_KIND, "config": case_config(record, contract),
            "key": physical["key"], "cad": physical["cad"], "group": physical["group"],
            "attempted": False, "coarse_true_converged": False, "numerical_U_difference_passed": False,
            "cold_cost_gate_passed": False, "metrics": None, "read_receipt": zero_receipt(),
            "old_GT_accuracy_assigned_to_new_endpoint": False, "costs": {"components_complete": False,
                "accounted_cold_wall_seconds": None, "fresh_combined_cold_remeasured": False}}


def case_costs(record, ledger, metadata_seconds):
    ledger.update()
    inherited = record["attributable_setup"]["total_seconds"]
    new = ledger.data["cumulative_case_wall_seconds"]
    complete = all(finite_cost(value) for value in (inherited, new, metadata_seconds))
    cold = inherited + new + metadata_seconds if complete else None
    return {"components_complete": complete, "accounted_cold_wall_seconds": cold,
        "inherited_attributable_setup": record["attributable_setup"],
        "actual_new_case_cumulative_wall_seconds": new, "actual_metadata_per_single_query_seconds": metadata_seconds,
        "maximum_candidate_accounted_cold_seconds": record["maximum_candidate_accounted_cold_seconds"],
        "accepted_coarse_cold_seconds": record["accepted_coarse_cold_seconds"],
        "cold_to_accepted_coarse_ratio": cold/record["accepted_coarse_cold_seconds"] if complete else None,
        "cached_actual_latency_seconds": new, "fresh_combined_cold_remeasured": False,
        "new_reference_IO_and_comparison_seconds": ledger.data["cumulative_reference_seconds"],
        "new_reference_IO_already_in_actual_case_wall": True,
        "old_fine_iterative_solve_and_reference_IO_charged": False,
        "cumulative_solve_wall_seconds": ledger.data["cumulative_solve_wall_seconds"],
        "cumulative_setup_read_seconds": ledger.data["cumulative_setup_read_seconds"],
        "cumulative_endpoint_seconds": ledger.data["cumulative_endpoint_seconds"],
        "cumulative_persistence_seconds": ledger.data["cumulative_persistence_seconds"],
        "checkpoint_seconds": ledger.data["checkpoint_seconds"],
        "solve_budget_overshoot_seconds": max(0., ledger.data["cumulative_solve_wall_seconds"] - 60.),
        "definition": "Inherited accepted fine input/operator + coarse construction + actual cumulative case wall + actual stage metadata; reference/endpoint/checkpoint I/O inside wall once"}


def finalize_case(record, summary, ledger, metadata_seconds):
    _, base = helpers()
    ledger.account("persistence")
    receipt = {**zero_receipt(), **ledger.data["read_receipt"],
               "fine_endpoint_K_applications": ledger.data["endpoint_K_applications"]}
    summary.update({"costs": case_costs(record, ledger, metadata_seconds), "read_receipt": receipt,
                    "counts": {name: ledger.data[name] for name in helpers()[0].COUNTERS}})
    summary["cold_cost_gate_passed"] = bool(summary["costs"]["components_complete"]
        and summary["costs"]["accounted_cold_wall_seconds"] <= record["maximum_candidate_accounted_cold_seconds"])
    base.write_json(ledger.output/"summary.json", summary)
    ledger.account("persistence")
    summary["costs"] = case_costs(record, ledger, metadata_seconds)
    summary["cold_cost_gate_passed"] = bool(summary["costs"]["components_complete"]
        and summary["costs"]["accounted_cold_wall_seconds"] <= record["maximum_candidate_accounted_cold_seconds"])
    base.write_json(ledger.output/"summary.json", summary)
    ledger.account("completed")
    return summary


def solve_case(record, contract, output, clock, metadata_seconds):
    core, base = helpers()
    config = case_config(record, contract)
    if (output/"config.json").exists():
        require(read_json(output/"config.json") == config, "Saved direct coarse case identity differs")
    output.mkdir(parents=True, exist_ok=True)
    base.write_json(output/"config.json", config)
    ledger = core.CaseLedger(output, config, clock, record["parent_record"]["physical_record"]["key"])
    ledger.data["read_receipt"] = {**zero_receipt(), **ledger.data["read_receipt"]}
    ledger.account()
    endpoint_path = output/"endpoint_no_GT.json"
    if endpoint_path.exists():
        endpoint = read_json(endpoint_path)
        require(endpoint["config"] == config, "Committed direct coarse endpoint differs")
        require(base.stat_record(endpoint["prediction"]["path"]) == {name: endpoint["prediction"][name] for name in ("bytes", "mtime_ns")},
                "Persisted direct coarse prediction changed")
        return finalize_case(record, endpoint, ledger, metadata_seconds)
    result = case_base(record, contract)
    result["attempted"] = True
    try:
        ledger.check()
        import numpy as np
        import torch
        from .elastic_pilot_data import load_input
        ledger.torch = torch
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        parent = record["parent_record"]
        physical = parent["physical_record"]
        inputs = load_input(read_json(parent["input_manifest"]["path"]), device)
        lam, mu = base._validate_inputs(torch, inputs, physical)
        cache = read_json(record["coarse_manifest"]["path"])
        arrays = cache["arrays"]
        P = core.load_sparse(np, torch, base, arrays["P"], device)
        PT = core.load_sparse(np, torch, base, arrays["PT"], device)
        Ac = core.load_sparse(np, torch, base, arrays["Ac"], device)
        diagonal = base._load_array(np, torch, arrays["diagonal"], device)
        free = inputs["free_node_ids"]
        require(P.shape == (len(free), record["mapping"]["free_corner_nodes"])
                and PT.shape == (P.shape[1], P.shape[0]) and Ac.shape == (3*P.shape[1], 3*P.shape[1])
                and diagonal.shape == (Ac.shape[0],) and diagonal.dtype == torch.float64
                and torch.isfinite(diagonal).all() and (diagonal > 0).all(), "Actual accepted direct coarse dimensions/diagonal differ")
        coarse = core.AdditiveCoarse(torch, P, PT, Ac, diagonal,
            cache["assembly"]["symmetric_scaled_Gershgorin_Gamma"], None, ledger)
        ledger.data["read_receipt"]["accepted_coarse_cache_reads"] += 1
        b_f = inputs["force"][free]
        b = coarse.restrict(b_f).reshape(-1)
        norm_b = float(torch.linalg.vector_norm(b))
        core.negative(math.isfinite(norm_b) and norm_b > 0, "Coarse projected RHS is zero or nonfinite")
        ledger.check(); ledger.account()
        y, r, p = torch.zeros_like(b), b.clone(), torch.zeros_like(b)
        state = {"config": config, "iteration": 0, "rz": None, "trace": []}
        checkpoint = output/"checkpoint.json"
        if checkpoint.exists():
            state = read_json(checkpoint)
            require(state["config"] == config and base.stat_record(state["vectors_path"]) == state["vectors_stat"],
                    "Preserved direct coarse checkpoint identity differs")
            with np.load(host_path(state["vectors_path"]), allow_pickle=False) as source:
                require(set(source.files) == {"z", "r", "p"}, "Direct coarse checkpoint vector inventory differs")
                y, r, p = (torch.as_tensor(np.array(source[name]), device=device) for name in ("z", "r", "p"))
            require(all(value.shape == b.shape and value.dtype == torch.float64 and torch.isfinite(value).all() for value in (y, r, p))
                    and state["iteration"] <= ledger.data["iterations_executed_total"], "Preserved direct coarse state/counters differ")
        ledger.account("solving")
        def save():
            nonlocal state
            tick = time.monotonic(); ledger.account()
            state.update({"cumulative_solve_wall_seconds": ledger.data["cumulative_solve_wall_seconds"],
                          "iterations_executed_total": ledger.data["iterations_executed_total"]})
            state = base._save_pcg(np, output, state, y, r, p)
            ledger.data["checkpoint_seconds"] += time.monotonic() - tick
            ledger.account()
        def true_residual(reason):
            actual = b - coarse.coarse_apply(y)
            value = base._relative(torch, actual, norm_b)
            core.negative(torch.isfinite(actual).all() and value is not None and math.isfinite(value), "Undefined true coarse residual")
            state["trace"].append({"iteration": state["iteration"], "true_coarse_relative_residual": value, "reason": reason})
            ledger.check_memory()
            return actual, value
        true_r, true_relative = true_residual("initial_or_resume")
        numerical_state = "converged" if true_relative <= 1e-8 else "solving"
        if state.get("checkpoint_recurrence_state") == "terminal_true_converged":
            core.negative(numerical_state == "converged", "Saved terminal coarse state failed true residual")
        if state["iteration"] == 0 and numerical_state == "solving":
            r = true_r.clone(); p = r/diagonal; state["rz"] = float((r*p).sum())
        if numerical_state == "solving":
            core.negative(finite_cost(state["rz"]) and state["rz"] > 0, "Nonpositive coarse rTMinvr")
        while numerical_state == "solving":
            ledger.update(); ledger.check_memory()
            if ledger.data["iterations_executed_total"] >= 10000:
                numerical_state = "maximum_iterations_negative"; break
            if ledger.data["cumulative_solve_wall_seconds"] >= 60 or clock.remaining() <= 0:
                numerical_state = "cumulative_budget_negative"; break
            ledger.data["iterations_executed_total"] += 1
            ledger.account()
            Ap = coarse.coarse_apply(p)
            curvature = float((p*Ap).sum())
            core.negative(math.isfinite(curvature) and curvature > 0, "Nonpositive coarse pTAcp or nonfinite PCG")
            alpha = state["rz"]/curvature
            y = y + alpha*p; r = r - alpha*Ap
            core.negative(torch.isfinite(y).all() and torch.isfinite(r).all(), "Nonfinite direct coarse iterate")
            state["iteration"] += 1
            recursive = base._relative(torch, r, norm_b)
            verify = state["iteration"] % 25 == 0 or recursive is None or recursive <= 1e-8
            if verify:
                true_r, true_relative = true_residual("periodic_or_convergence_candidate")
                if true_relative <= 1e-8:
                    numerical_state = "converged"
                    state["checkpoint_recurrence_state"] = "terminal_true_converged"
                    break
            s = r/diagonal
            new_rz = float((r*s).sum())
            core.negative(math.isfinite(new_rz) and new_rz > 0, "Nonpositive coarse recursive preconditioned energy")
            p = s + (new_rz/state["rz"])*p; state["rz"] = new_rz
            if state["iteration"] % 25 == 0:
                save()
        _, true_relative = true_residual("terminal_endpoint")
        numerical_state = "converged" if true_relative <= 1e-8 else numerical_state
        if numerical_state == "converged":
            state["checkpoint_recurrence_state"] = "terminal_true_converged"
        save(); ledger.account("endpoint")
        full_u = base._restore_full(torch, coarse.prolong(y.reshape(-1, 3)), inputs)
        core.negative(torch.isfinite(full_u).all(), "Nonfinite restored direct coarse full-node displacement")
        # Commit full original-node U before any immutable numerical reference I/O.
        prediction = base._save_array(np, output/"physical_u_mm.npy", full_u)
        base.write_json(output/"prediction_before_reference.json", {"config": config, "prediction": prediction,
            "coarse_true_relative_residual": true_relative, "numerical_reference_reads": ledger.data["numerical_reference_reads"]})
        ledger.account("endpoint")
        operator_record = read_json(parent["operator_summary"]["path"])
        grad = base._load_array(np, torch, operator_record["arrays"]["gradients"], device)
        weights = base._load_array(np, torch, operator_record["arrays"]["weights"], device)
        require(grad.shape == (len(inputs["cells10"]), 4, 10, 3) and weights.shape == (len(inputs["cells10"]), 4),
                "Accepted fine endpoint operator cache shape differs")
        operator = base.MatrixFreeElastic(torch, inputs["cells10"], grad, weights, len(full_u), lam, mu)
        ledger.data["fine_K_applications"] += 1; ledger.data["endpoint_K_applications"] += 1
        residual = inputs["force"] - operator.apply(full_u)
        norm_f = float(torch.linalg.vector_norm(b_f))
        projected = coarse.restrict(residual[free]).reshape(-1)
        fine_relative = base._relative(torch, residual[free], norm_f)
        projected_relative = base._relative(torch, projected, norm_b)
        core.negative(torch.isfinite(residual).all() and fine_relative is not None and projected_relative is not None,
                      "Nonfinite fine/projected endpoint residual")
        ledger.check_memory(); ledger.account()
        fine_report = {"fine_free_relative_residual": fine_relative,
            "fine_free_residual_norm_N": float(torch.linalg.vector_norm(residual[free])),
            "PT_fine_residual_relative_to_coarse_rhs": projected_relative,
            "PT_fine_residual_norm_N": float(torch.linalg.vector_norm(projected)),
            "projected_orthogonality_le1e8": projected_relative <= 1e-8,
            "fixed_translation_max_abs_mm": float(full_u[inputs["fixed_node_ids"]].abs().max()),
            "fine_true1e8_required": False, "full_fine_equilibrium_claimed": False,
            "fine_operator_applications": 1, "old_fine_K_guard_repeats": 0}
        ledger.account("reference")
        reference_record = parent["base_summary"]["prediction"]
        require(base.stat_record(reference_record["path"]) == {name: reference_record[name] for name in ("bytes", "mtime_ns")},
                "Original zero-start fine numerical reference changed")
        ledger.data["numerical_reference_reads"] += 1; ledger.account()
        reference = np.load(host_path(reference_record["path"]), allow_pickle=False)
        require(reference.shape == (physical["expected_geometry"]["nodes"], 3) and reference.dtype == np.float64
                and np.isfinite(reference).all(), "Original full-node numerical reference schema differs")
        prediction_cpu = full_u.detach().cpu().numpy()
        regions = {name: value.detach().cpu().numpy() for name, value in inputs["regions"].items()}
        metrics = base._metrics(np, reference, prediction_cpu, regions)
        difference = metrics["u_vector_relative_l2"]
        passed_difference = difference is not None and difference <= .05
        metrics.update({"reference_role": "Immutable original zero-start fine converged numerical U, not GT",
                        "numerical_difference_relative_l2_maximum": .05,
                        "old_GT_accuracy_assigned_to_new_endpoint": False})
        ledger.account("persistence")
        result.update({"numerical_state": numerical_state, "coarse_true_converged": numerical_state == "converged",
            "numerical_U_difference_passed": passed_difference, "true_coarse_relative_residual": true_relative,
            "iteration": state["iteration"], "trace": state["trace"], "prediction": prediction,
            "metrics": metrics, "endpoint_fine_residual": fine_report, "memory": base._memory(torch),
            "coarse_mapping": record["mapping"], "coarse_manifest": record["coarse_manifest"],
            "numerical_reference_summary": parent["base_summary"], "read_receipt": zero_receipt(),
            "trial_space": "range(P); a lower-dimensional approximate solution, not the full fine solution"})
        base.write_json(endpoint_path, result); ledger.account()
        return finalize_case(record, result, ledger, metadata_seconds)
    except (base.ScientificNegative, RuntimeError, MemoryError) as error:
        ledger.account()
        result.update({"numerical_state": "frozen_direct_coarse_negative", "reason": f"{type(error).__name__}: {error}",
                       "fallbacks": 0})
        if ledger.torch is not None:
            result["memory"] = base._memory(ledger.torch)
        return finalize_case(record, result, ledger, metadata_seconds)
    except Exception:
        ledger.account()
        raise


def completion_gate(cases):
    present = all(key in cases for key in CASES)
    converged = sum(bool(cases.get(key, {}).get("coarse_true_converged")) for key in CASES)
    differences = sum(bool(cases.get(key, {}).get("numerical_U_difference_passed")) for key in CASES)
    cold = sum(bool(cases.get(key, {}).get("cold_cost_gate_passed")) for key in CASES)
    complete = present and all(cases[key]["costs"].get("components_complete") is True for key in CASES)
    target_free = present and all(cases[key]["read_receipt"].get(name) == 0 for key in CASES for name in ZERO_KEYS)
    return {"passed": bool(present and complete and target_free and converged == differences == cold == 2),
            "inventory_denominator": 2, "coarse_true_converged_cases": converged,
            "fullN_numerical_U_difference_le5percent_cases": differences,
            "accounted_cold_le_half_accepted_coarse_cases": cold, "cost_components_complete": complete,
            "target_and_weight_reads_zero": target_free,
            "decision": "eligible_for_root_scope_decision" if present and complete and target_free and converged == differences == cold == 2
                        else "close_frozen_low_fidelity_candidate_no_rescue"}


def write_report(output, summary):
    _, base = helpers()
    lines = ["# Direct masked-Galerkin coarse：两例低保真筛选", "",
        f"固定门：{'PASS' if summary['completion_gate']['passed'] else 'FAIL'}；零GT，旧fine精度未转移。",
        "| CAD/ver | coarse真实残差 | 相对fine数值U差 | RMS/max mm | accounted cold / 上限s |",
        "|---|---:|---:|---:|---:|"]
    def number(value):
        return f"{value:.6g}" if finite_cost(value) else "missing"
    for key in CASES:
        item = summary["cases"][key]; metrics = item.get("metrics") or {}; costs = item["costs"]
        lines.append(f"| {key} | {number(item.get('true_coarse_relative_residual'))} | {number(metrics.get('u_vector_relative_l2'))} | "
            f"{number(metrics.get('vector_RMS_mm'))}/{number(metrics.get('max_nodal_vector_error_mm'))} | "
            f"{number(costs.get('accounted_cold_wall_seconds'))}/{number(summary['config']['contract']['records'][key]['maximum_candidate_accounted_cold_seconds'])} |")
    lines += ["", "5%为对原fine近似模型的数值差距，非CSV GT误差。fine残差与PT投影残差分别在metrics.json；不要求fine1e-8。",
              "计费cold组成账本包含继承setup与新IO/求解/恢复/验证/保存/数值参考，不是fresh联合cold；cached实际延迟另列。",
              "完整TET10/约束物理接口不同于学习点云25接口；本候选没有source signed-S，不作完整solver排名。",
              "缺失/预算/后端失败关闭固定候选，不换CAD、救容差或二次回放。"]
    require(len(lines) <= 25, "Small report exceeded frozen line budget")
    (output/"BRIEF.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    base.write_json(output/"metrics.json", {"status": "completed", "kind": KIND,
        "completion_gate": summary["completion_gate"], "cases": {key: {name: summary["cases"][key].get(name)
        for name in ("true_coarse_relative_residual", "metrics", "endpoint_fine_residual", "costs", "memory", "reason")}
        for key in CASES}, "GT_reads": 0, "old_GT_accuracy_assigned_to_new_endpoint": False})


def run(plan_path=PLAN, output=None):
    began = time.monotonic()
    contract, plan = request_contract(plan_path)
    require(plan.get("scientific_rules_frozen") is True and plan.get("actual_execution_released") is True,
            "Authentic root actual release required before scientific imports/array reads")
    review = read_json(plan["root_review"])
    require(review.get("passed") is True and review.get("source_sha256") == contract["source_sha256"][SOURCE]
            and review.get("science_recipe_sha256") == contract["science_recipe_sha256"]
            and review.get("base_evidence_decision") == "proceed_one_frozen_direct_Galerkin_coarse_candidate",
            "Authentic root source/recipe/base-evidence decision required")
    destination = host_path(output or plan["output"])
    require(destination == host_path(plan["output"]), "Frozen direct coarse output differs")
    config = {"contract": contract, "stage": KIND}
    core, base = helpers()
    if (destination/"config.json").exists():
        require(read_json(destination/"config.json") == config, "Preserved direct coarse campaign identity differs")
    if (destination/"summary.json").exists():
        previous = read_json(destination/"summary.json")
        require(previous.get("config") == config, "Preserved direct coarse summary differs")
        if previous.get("status") == "completed":
            return previous
    destination.mkdir(parents=True, exist_ok=True)
    base.write_json(destination/"config.json", config)
    inherited = sum(record["attributable_setup"]["total_seconds"] for record in contract["records"].values())
    clock = CampaignClock(destination, config, began, inherited)
    metadata_path = destination/"stage_metadata_receipt.json"
    old = read_json(metadata_path) if metadata_path.exists() else {}
    require(not old or old["config_sha256"] == digest(config), "Preserved actual metadata costs differ")
    metadata_seconds = old.get("cumulative_actual_metadata_seconds", 0.) + time.monotonic() - began
    base.write_json(metadata_path, {"config_sha256": digest(config), "cumulative_actual_metadata_seconds": metadata_seconds})
    clock.account("running")
    cases = {}
    try:
        for key in CASES:
            record = contract["records"][key]
            case_output = host_path(record["case_output"])
            require(case_output == destination/key.replace("/", "_"), "Frozen direct coarse case destination differs")
            if (case_output/"summary.json").exists():
                item = read_json(case_output/"summary.json")
                require(item.get("config") == case_config(record, contract), "Preserved direct coarse child identity differs")
                if item.get("status") == "completed":
                    costs = item["costs"]
                    if costs.get("components_complete") is True:
                        # Reused scientific endpoints need no array replay. Actual
                        # additional metadata attempts remain in their cold ledger.
                        costs["actual_metadata_per_single_query_seconds"] = metadata_seconds
                        costs["accounted_cold_wall_seconds"] = (record["attributable_setup"]["total_seconds"]
                            + costs["actual_new_case_cumulative_wall_seconds"] + metadata_seconds)
                        costs["cold_to_accepted_coarse_ratio"] = costs["accounted_cold_wall_seconds"]/record["accepted_coarse_cold_seconds"]
                        item["cold_cost_gate_passed"] = costs["accounted_cold_wall_seconds"] <= record["maximum_candidate_accounted_cold_seconds"]
                        base.write_json(case_output/"summary.json", item)
                    cases[key] = item; clock.account("case_reused", key); continue
            if clock.remaining() <= 0 and not (case_output/"checkpoint.json").exists() and not (case_output/"endpoint_no_GT.json").exists():
                item = {**case_base(record, contract), "numerical_state": "campaign_budget_unattempted_negative",
                        "reason": "240s campaign accounted budget exhausted; fixed denominator retained"}
                case_output.mkdir(parents=True, exist_ok=True)
                base.write_json(case_output/"config.json", item["config"])
                base.write_json(case_output/"summary.json", item)
            else:
                item = solve_case(record, contract, case_output, clock, metadata_seconds)
            cases[key] = item; clock.account("case_completed", key)
        receipt = zero_receipt()
        for name in ("numerical_reference_reads", "accepted_coarse_cache_reads", "fine_endpoint_K_applications"):
            receipt[name] = sum(item["read_receipt"].get(name, 0) for item in cases.values())
        gate = completion_gate(cases)
        if clock.elapsed() > 240:
            gate.update({"passed": False, "decision": "close_frozen_low_fidelity_candidate_no_rescue",
                         "campaign_accounted_budget_passed": False})
        else:
            gate["campaign_accounted_budget_passed"] = True
        summary = {"status": "completed", "kind": KIND, "config": config, "case_keys": list(CASES),
            "cases": cases, "completion_gate": gate, "read_receipt": receipt,
            "costs": {"cumulative_actual_stage_wall_seconds": clock.actual(),
                "inherited_attributable_setup_seconds": inherited,
                "cumulative_campaign_accounted_wall_seconds": clock.elapsed(), "maximum_accounted_wall_seconds": 240,
                "budget_overshoot_seconds": max(0., clock.elapsed()-240.), "this_call_actual_stage_wall_seconds": time.monotonic()-began,
                "actual_metadata_seconds_including_resumes": metadata_seconds, "fresh_combined_cold_remeasured": False},
            "scope": "Two fixed TRAIN/DEV ver, range(P) displacement approximation; no GT/S/final claim",
            "old_GT_accuracy_assigned_to_new_endpoint": False,
            "failure_policy": "Terminal negative closes fixed direct-coarse candidate; no fallback/tuning/second replay"}
        base.write_json(destination/"summary.json", summary)
        write_report(destination, summary)
        clock.account("completed")
        summary["costs"].update({"cumulative_actual_stage_wall_seconds": clock.actual(),
            "cumulative_campaign_accounted_wall_seconds": clock.elapsed(), "budget_overshoot_seconds": max(0., clock.elapsed()-240.)})
        if clock.elapsed() > 240:
            summary["completion_gate"].update({"passed": False, "campaign_accounted_budget_passed": False,
                                              "decision": "close_frozen_low_fidelity_candidate_no_rescue"})
        base.write_json(destination/"summary.json", summary)
        return summary
    except Exception as error:
        clock.account("failed")
        base.write_json(destination/"summary.json", {"status": "failed", "kind": KIND, "config": config,
            "case_keys": list(CASES), "cases": cases, "reason": f"{type(error).__name__}: {error}",
            "read_receipt": zero_receipt(), "costs": {"cumulative_actual_stage_wall_seconds": clock.actual(),
                "inherited_attributable_setup_seconds": inherited,
                "cumulative_campaign_accounted_wall_seconds": clock.elapsed()}})
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", default=PLAN)
    result.add_argument("--output", default=None)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    run(args.plan, args.output)
