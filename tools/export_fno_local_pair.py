"""CPU export of saved FNO local25 pair JSON/loss scalars after actual exit.

Default API/CLI returns immediately with zero IO.  ``--export`` requires an
independent root export release and authentic terminal FNO summary, closed
inactive clock, and exact saved guard/child process exit proof.  No data array,
checkpoint body, model construction, inference, or target parsing is used.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
OWN = "tools/export_fno_local_pair.py"
PLAN = "notes/geobc_fno_local_condition_pair_plan.json"
REVIEW = "notes/fno_local_pair_exporter_root_review.json"
OUTPUT = "results/geobc_fno_local_pair"
KIND = "FNO_TRAIN120_local25_representation_pair"
EXPORT_KIND = "FNO_local25_pair_saved_scalar_export_v1"
SCIENCE = "192ccffe2a79bf63cdd2e8199dd2b9fab79dd29ad8eb10ed689c78dd10e26b51"
ARMS = ("FNO_latent_query_grid25", "FNO_latent_query_native25")
OLD = "fno_cad40_fresh4800"
OLD_SUMMARY = "results/geobc_train_fit/summary.json"
OLD_SHA = "f20f2d68d1a277e05c42b05a9440572a8dd32b051b6445417c04d7d69e75e161"
GUARD = "tools/run_geobc_fno_pair_guard.py"
DRIVER = "geobc/fno_local_condition_pair.py"
RELATIVE = ("u_relative_l2", "stress_relative_l2", "stress_hotspot_relative_l2")
AGGREGATES = tuple(name for prefix in ("u", "stress", "stress_hotspot")
                   for name in ("case_mean_" + prefix + "_relative_l2", "worst_case_" + prefix + "_relative_l2"))
CASE_SCALARS = RELATIVE + ("u_vector_error_RMS_mm", "u_max_vector_error_mm", "stress_RMSE_MPa",
                          "stress_MAE_MPa", "stress_max_abs_error_MPa", "stress_hotspot_nodes", "nodes")
ZERO = ("DEV_target_reads", "held5_target_reads", "raw_source_reads", "old_final_reads",
        "sealed_test_reads", "DEV_dia_reads", "scaler_fits", "pretrained_weight_reads",
        "new_data_preparations", "model_selection_updates", "physical_U_or_tensor_input_reads")


class NotReady(RuntimeError):
    pass


def require(ok, message):
    if not bool(ok):
        raise NotReady(message)


def path(value):
    text = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", text):
        text = text[5].upper() + ":/" + text[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", text):
        text = "/mnt/" + text[0].lower() + text[2:]
    filename = Path(text).expanduser()
    return (filename if filename.is_absolute() else ROOT / filename).resolve()


def relative(value):
    return path(value).relative_to(ROOT.resolve()).as_posix()


def sha(filename):
    result = hashlib.sha256()
    with path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_json(filename, receipt=None):
    filename = path(filename)
    require(filename.suffix == ".json", "Exporter JSON input must be a project JSON file")
    name = relative(filename)
    value = json.loads(filename.read_text(encoding="utf-8-sig"))
    if receipt is not None:
        receipt.append({"path": name, "bytes": filename.stat().st_size,
                        "sha256": sha(filename), "read_kind": "saved_JSON_metadata_or_scalars"})
    return value


def write_json(filename, value):
    filename = path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(filename.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, filename)


def finite(value, nonnegative=False):
    return type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0)


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def process_identity(pid):
    require(os.name != "nt" and Path("/proc").is_dir(), "Export exit proof uses the common WSL process namespace")
    require(type(pid) is int and pid > 0, "Actual Linux PID is required")
    try:
        folder = Path("/proc") / str(pid)
        raw = (folder / "stat").read_text(encoding="utf-8")
        tail = raw[raw.rfind(")") + 2:].split()
        if tail[0] in ("Z", "X"):
            return None
        argv = [item.decode("utf-8", errors="strict") for item in
                (folder / "cmdline").read_bytes().split(b"\0") if item]
        return {"pid": pid, "ppid": int(tail[1]), "start_ticks": int(tail[19]), "argv": argv}
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (OSError, ValueError, IndexError, UnicodeError) as error:
        raise NotReady("Actual process identity is unknown; export remains blocked") from error


def exited(record):
    require(isinstance(record, dict) and type(record.get("pid")) is int and record["pid"] > 0
            and type(record.get("start_ticks")) is int and record["start_ticks"] > 0
            and isinstance(record.get("argv"), list) and bool(record["argv"]),
            "Genuine saved PID/start_ticks/argv is required; never infer identity from an old PID")
    current = process_identity(record["pid"])
    return current is None or current["start_ticks"] != record["start_ticks"]


def terminal_inputs(plan_path=PLAN):
    """Saved scalar JSON + exact process proof; no scientific data/module API."""
    receipt = []
    review = read_json(REVIEW, receipt)
    require(review.get("passed") is True and review.get("actual_export_released") is True
            and review.get("source_sha256") == sha(OWN)
            and review.get("scientific_digest") == SCIENCE,
            "Separate genuine root exporter release is required")
    plan = read_json(plan_path, receipt)
    science = plan["scientific"]
    require(plan.get("kind") == KIND and plan.get("scientific_rules_frozen") is True
            and digest(science) == plan.get("scientific_digest") == SCIENCE
            and science["arms"] == list(ARMS) and science["fixed_updates"] == 4800
            and science["TRAIN_readout_updates"] == [0, 4800]
            and len(science["TRAIN_keys"]) == len(set(science["TRAIN_keys"])) == 120,
            "Frozen FNO pair/TRAIN120 scope differs")
    root_review = read_json(plan["root_review"], receipt)
    require(root_review.get("passed") is True and root_review.get("actual_execution_released") is True
            and root_review.get("fno_pair_released") is True
            and root_review.get("scientific_digest") == SCIENCE
            and root_review.get("source_pins") == plan["source_sha256"]
            and root_review.get("source_sha256") == plan["source_sha256"].get(DRIVER),
            "FNO actual science/source root binding differs")
    require(sha(DRIVER) == plan["source_sha256"][DRIVER]
            and sha(GUARD) == plan["source_sha256"][GUARD], "FNO driver or actual external guard source changed")
    folder = path(plan["output"])
    require(relative(folder) == "runs/geobc_coverage_comparison/FNO_LOCAL25_PAIR", "Frozen FNO output differs")
    summary = read_json(folder / "summary.json", receipt)
    status = read_json(folder / "status.json", receipt)
    clock = read_json(folder / "clock.json", receipt)
    launch = read_json(folder / "external_launch.json", receipt)
    contract = summary.get("config", {}).get("contract", {})
    require(summary.get("status") in ("completed", "failed") and summary.get("kind") == KIND
            and status.get("phase") == "terminal" and status.get("status") == summary["status"]
            and clock.get("closed") is True and clock.get("active") is False
            and contract.get("scientific_digest") == SCIENCE
            and contract.get("source_sha256") == root_review["source_pins"]
            and contract.get("plan_science") == science
            and summary.get("TRAIN_keys") == contract.get("TRAIN_keys") == science["TRAIN_keys"]
            and summary.get("fixed_TRAIN_denominator") == 120 and summary.get("fixed_arm_denominator") == 2,
            "Authentic terminal FNO summary and closed inactive clock are required")
    require(launch.get("status") == "terminal" and launch.get("summary_present") is True
            and launch.get("automatic_replay") is False and launch.get("external_budget_seconds") == 9000
            and launch.get("guard_source_sha256") == plan["source_sha256"][GUARD]
            and type(launch.get("exit_code")) is int and type(launch.get("external_timeout")) is bool
            and type(launch.get("within_external_budget")) is bool
            and finite(launch.get("elapsed_launch_to_exit_seconds"), True)
            and isinstance(launch.get("ended_at_utc"), str), "Actual FNO external terminal receipt is required")
    guard_process, child_process = launch.get("guard_process"), launch.get("child_process")
    require(exited(guard_process) and exited(child_process), "Actual FNO guard or child is still running")
    require(child_process.get("ppid") == guard_process["pid"]
            and child_process["pid"] == launch.get("child_pid") == status.get("pid") == clock.get("pid")
            and child_process["argv"] == launch.get("child_argv")
            and "geobc.fno_local_condition_pair" in child_process["argv"]
            and "--start" in child_process["argv"], "Genuine FNO child identity/argv differs")
    guard_argv = guard_process["argv"]
    correct_guard = any(path(token) == path(GUARD) for token in guard_argv if token.endswith(".py"))
    correct_guard = correct_guard or any(token == "tools.run_geobc_fno_pair_guard" and index > 0
                                         and guard_argv[index - 1] == "-m" for index, token in enumerate(guard_argv))
    require(correct_guard, "Saved real guard argv does not name the reviewed FNO guard")
    require(all(summary.get("read_receipt", {}).get(name) == 0 for name in ZERO), "Saved FNO forbidden read receipt is nonzero")
    # Failure summaries may omit embedded endpoints; use their genuine saved
    # fixed readout JSON when it exists, without recreating any prediction.
    for name in ARMS:
        record = summary.setdefault("arms", {}).setdefault(name, {})
        if record.get("config"):
            require(record["config"].get("contract") == contract
                    and record["config"].get("arm", name) == name,
                    "Saved arm scientific/source/config binding differs")
        require(record.get("arm", name) == name, "Saved arm identity differs")
        endpoints = dict(record.get("endpoints") or {})
        for step in (0, 4800):
            readout_file = folder / name / ("TRAIN120_step" + str(step) + ".json")
            if readout_file.is_file():
                saved = read_json(readout_file, receipt)
                require(saved.get("status") == "completed" and saved.get("arm") == name and saved.get("step") == step
                        and saved.get("fixed_denominator") == 120, "Saved fixed FNO readout identity differs")
                if str(step) in endpoints:
                    require(endpoints[str(step)] == saved, "Embedded and saved fixed FNO readout differ")
                endpoints[str(step)] = saved
        record["endpoints"] = endpoints
    reference = read_json(OLD_SUMMARY, receipt)
    require(sha(OLD_SUMMARY) == OLD_SHA and reference.get("status") == "completed"
            and reference.get("case_keys") == science["TRAIN_keys"]
            and reference.get("fixed_case_denominator_each_arm") == 120
            and reference.get("config", {}).get("contract", {}).get("scaler") == science["scaler"],
            "Saved immutable old FNO fullTRAIN120 diagnostic reference changed")
    old = reference["arms"][OLD]
    old_training = None
    old_training_status = "missing_saved_original_training_metadata"
    source = old.get("original_training_summary", {})
    if source.get("path"):
        original_file = path(source["path"])
        require(relative(original_file) == "runs/geobc_coverage_comparison/fno_cad40_fresh4800/summary.json"
                and source.get("sha256") == reference["config"]["contract"]["endpoint_inventory"][OLD]["summary"]["sha256"],
                "Saved old FNO training-summary identity differs")
        if original_file.is_file():
            old_training = read_json(original_file, receipt)
            require(sha(original_file) == source["sha256"], "Frozen old FNO original training summary changed")
            old_training_status = "saved_original_training_metadata_verified"
    return {"plan": plan, "summary": summary, "clock": clock, "launch": launch,
            "old": old, "old_training": old_training, "old_training_status": old_training_status,
            "input_receipt": receipt, "case_keys": science["TRAIN_keys"]}


def full120_metrics(readout, case_keys, eligible=True, reason=None, old=False, expected_step=4800):
    """Derive only full fixed120 scalar aggregates; never subset statistics."""
    readout = readout or {}
    stored = readout.get("aggregate", {})
    per = readout.get("per_case", {}) if old else stored.get("per_case", {})
    if not isinstance(per, dict):
        per = {}
    normalized = {key: clean(per.get(key)) for key in case_keys}
    exact_inventory = set(per) == set(case_keys)
    schema_complete = (readout.get("all120_metrics_available") is True and readout.get("checkpoint_step") == expected_step if old
                       else readout.get("status") == "completed" and readout.get("step") == expected_step
                       and readout.get("complete120") is True and readout.get("fixed_denominator") == 120
                       and readout.get("completed_cases") == 120)
    reasons = []
    if not eligible:
        reasons.append(reason or "fixed4800 endpoint was not completed")
    if not exact_inventory or not schema_complete:
        reasons.append("saved full120 inventory/readout is incomplete")
    values = {}
    for metric, prefix in zip(RELATIVE, ("u", "stress", "stress_hotspot")):
        column = [normalized[key].get(metric) if isinstance(normalized[key], dict) else None for key in case_keys]
        complete = eligible and exact_inventory and schema_complete and all(finite(item, True) for item in column)
        mean_name, worst_name = "case_mean_" + prefix + "_relative_l2", "worst_case_" + prefix + "_relative_l2"
        values[mean_name] = sum(column) / 120 if complete else None
        values[worst_name] = max(column) if complete else None
        if complete:
            for name in (mean_name, worst_name):
                saved = stored.get(name)
                if not finite(saved, True) or not math.isclose(saved, values[name], rel_tol=1e-12, abs_tol=1e-14):
                    values[name] = None
                    reasons.append("saved aggregate/per-case scalar identity differs: " + name)
        elif eligible and schema_complete and exact_inventory:
            reasons.append("nonfinite/missing/zero-reference scalar prevents full120 " + metric)
    complete = all(finite(values.get(name), True) for name in AGGREGATES)
    return {"fixed_denominator": 120, "all120_metrics_complete": complete,
            "metrics_available_cases": sum(isinstance(item, dict) for item in normalized.values()),
            "aggregate": values, "per_case": normalized,
            "reason": "; ".join(dict.fromkeys(reasons)) or None,
            "subset_mean_used": False, "source_step": readout.get("checkpoint_step") if old else readout.get("step")}


def parameter_count(value):
    value = value or {}
    count = value.get("total")
    return count if type(count) is int and count > 0 else None


def build_metrics(inputs):
    summary, clock, launch = inputs["summary"], inputs["clock"], inputs["launch"]
    keys = inputs["case_keys"]
    arms = {}
    for name in ARMS:
        record = summary.get("arms", {}).get(name, {})
        fixed = record.get("completed4800") is True and record.get("endpoint_step") == 4800
        exposure = record.get("case_exposure_counts") or {}
        balanced = set(exposure) == set(keys) and all(exposure[key] == 40 for key in keys)
        gradient = clean(record.get("first_actual_gradient_check"))
        endpoints = record.get("endpoints") or {}
        fixed_readout = endpoints.get("4800")
        if fixed_readout is None and (record.get("final_TRAIN120") or {}).get("step") == 4800:
            fixed_readout = record["final_TRAIN120"]
        reason = record.get("stop_reason") or (record.get("error") or {}).get("reason") or "fixed4800 endpoint unavailable"
        costs = clean(record.get("costs") or {})
        costs["training_seconds"] = clock.get("training_seconds", {}).get(name)
        costs["arm_wall_seconds"] = clock.get("arm_wall_seconds", {}).get(name)
        costs["authoritative_closed_clock"] = True
        costs["native_clock_axis"] = "monotonic; any unknown interruption charge is disclosed separately"
        costs["training_epoch_UTC_seconds"] = None
        costs["arm_wall_epoch_UTC_seconds"] = None
        costs["phase_epoch_UTC_seconds"] = None
        costs["epoch_UTC_phase_status"] = "not_recorded"
        arms[name] = {"status": record.get("status", "missing"), "actual_endpoint_step": record.get("endpoint_step"),
            "completed4800": fixed, "balanced40_exposures_each_case": balanced,
            "fixed0": full120_metrics(endpoints.get("0"), keys, expected_step=0),
            "fixed4800": full120_metrics(fixed_readout, keys, fixed, reason),
            "parameter_counts": clean(record.get("parameter_counts")),
            "parameters": parameter_count(record.get("parameter_counts")),
            "costs": costs, "memory": clean(record.get("memory")),
            "first_actual_gradient_check": gradient,
            "first_actual_gradient_passed": isinstance(gradient, dict) and gradient.get("passed") is True,
            "named_initial_state": clean(record.get("initial_parameters")), "stop_reason": reason,
            "case_exposure_counts": clean(exposure), "node_exposures": record.get("node_exposures")}
    old_record = inputs["old"]
    old_fixed = old_record.get("status") == "completed" and old_record.get("checkpoint_step") == 4800
    old_training = inputs["old_training"] or {}
    old_counts = old_training.get("parameter_counts", old_training.get("config", {}).get("parameter_counts"))
    old = {"fixed4800": full120_metrics(old_record, keys, old_fixed, "old fixed4800 reference incomplete", old=True),
           "parameters": parameter_count(old_counts), "parameter_counts": clean(old_counts),
           "training_seconds": old_record.get("original_training_costs", {}).get("train_seconds"),
           "training_attempt_wall_seconds": old_record.get("original_training_elapsed_this_attempt_seconds"),
           "diagnostic_readout_costs": clean(old_record.get("costs", {})),
           "original_training_metadata_status": inputs["old_training_status"],
           "checkpoint_step": old_record.get("checkpoint_step"), "summary_sha256": OLD_SHA}
    left, right = arms[ARMS[0]], arms[ARMS[1]]
    named = left["named_initial_state"]
    named_equal = (isinstance(named, dict) and bool(named.get("parameters"))
                   and named == right["named_initial_state"])
    ratios, old_ratios = {}, {}
    for metric in AGGREGATES:
        a, b = left["fixed4800"]["aggregate"][metric], right["fixed4800"]["aggregate"][metric]
        ratios[metric] = b / a if finite(a, True) and a > 0 and finite(b, True) else None
        reference = old["fixed4800"]["aggregate"][metric]
        old_ratios[metric] = {name: arms[name]["fixed4800"]["aggregate"][metric] / reference
                             if finite(reference, True) and reference > 0
                             and finite(arms[name]["fixed4800"]["aggregate"][metric], True) else None for name in ARMS}
    inner_wall = clock.get("cumulative_wall_seconds")
    external_wall = launch["elapsed_launch_to_exit_seconds"]
    external_epoch_wall = launch.get("elapsed_epoch_launch_to_exit_seconds")
    axis_difference = launch.get("clock_axis_difference_seconds")
    axes_consistent = (finite(external_wall, True) and finite(external_epoch_wall, True)
                       and finite(axis_difference)
                       and abs(axis_difference - (external_epoch_wall - external_wall)) <= 1e-6)
    external_axes_within = (axes_consistent and external_wall <= 9000 and external_epoch_wall <= 9000)
    unknown_charge = clock.get("unknown_interruption_charge_seconds")
    wall_ordered = (finite(inner_wall, True) and finite(external_wall, True)
                    and external_wall >= inner_wall)
    cost_exact = (launch.get("in_flight_costs_exact") is True and launch["external_timeout"] is False
                  and launch["exit_code"] == 0 and launch["within_external_budget"] is True
                  and wall_ordered and external_axes_within
                  and finite(unknown_charge, True) and unknown_charge == 0.)
    budgets = {"each_training_seconds_max": 3600, "whole_inner_wall_seconds_max": 9000,
               "actual_inner_wall_seconds": inner_wall,
               "actual_guard_launch_to_child_exit_seconds": external_wall,
               "actual_guard_epoch_launch_to_child_exit_seconds": external_epoch_wall,
               "clock_axis_difference_seconds": axis_difference,
               "clock_axis_difference_consistent": axes_consistent,
               "both_guard_clock_axes_within_9000_seconds": external_axes_within,
               "conservative_guard_launch_to_child_exit_seconds": max(external_wall, external_epoch_wall)
                   if finite(external_wall, True) and finite(external_epoch_wall, True) else None,
               "external_timeout": launch["external_timeout"], "within_external_budget": launch["within_external_budget"],
               "external_exit_code": launch["exit_code"],
               "in_flight_costs_exact": launch.get("in_flight_costs_exact") is True,
               "cost_accounting_exact": cost_exact,
               "external_wall_covers_inner_closed_clock": wall_ordered,
               "unknown_interruption_charge_seconds": unknown_charge,
               "arm_training_seconds": clean(clock.get("training_seconds", {})),
               "arm_wall_seconds": clean(clock.get("arm_wall_seconds", {})),
               "phase_seconds": clean(clock.get("phase_seconds", {})),
               "native_clock_axis": "monotonic; unknown interruption epoch charge remains separate",
               "guard_elapsed_clock_axis": "monotonic",
               "guard_epoch_elapsed_clock_axis": "epoch/UTC",
               "clock_axis_difference_definition": "epoch elapsed minus monotonic elapsed",
               "phase_epoch_UTC_seconds": None, "arm_epoch_UTC_seconds": None,
               "training_epoch_UTC_seconds": None, "epoch_UTC_phase_status": "not_recorded"}
    within = (finite(budgets["actual_inner_wall_seconds"], True) and budgets["actual_inner_wall_seconds"] < 9000
              and launch["within_external_budget"] is True and launch["external_timeout"] is False
              and budgets["cost_accounting_exact"] and budgets["external_wall_covers_inner_closed_clock"]
              and finite(external_wall, True) and external_wall <= 9000
              and finite(external_epoch_wall, True) and external_epoch_wall <= 9000
              and budgets["both_guard_clock_axes_within_9000_seconds"]
              and finite(budgets["unknown_interruption_charge_seconds"], True)
              and budgets["unknown_interruption_charge_seconds"] == 0.
              and all(finite(clock.get("training_seconds", {}).get(name), True)
                      and clock["training_seconds"][name] <= 3600 for name in ARMS))
    budgets["within_frozen_training_and_both_wall_axes_budget"] = bool(within)
    qualified = (summary["status"] == "completed" and named_equal and within
                 and left["parameters"] is not None and left["parameters"] == right["parameters"]
                 and all(arms[name]["completed4800"] and arms[name]["balanced40_exposures_each_case"]
                         and arms[name]["first_actual_gradient_passed"]
                         and arms[name]["fixed4800"]["all120_metrics_complete"] for name in ARMS))
    limits = {"case_mean_stress_relative_l2": .8, "case_mean_u_relative_l2": 1.05,
              "worst_case_stress_relative_l2": 1., "worst_case_stress_hotspot_relative_l2": 1.}
    reproduced_gate = qualified and all(finite(ratios[name], True) and ratios[name] <= maximum
                                        for name, maximum in limits.items())
    metric_labels = {"case_mean_u_relative_l2": "U mean", "worst_case_u_relative_l2": "U worst",
                     "case_mean_stress_relative_l2": "signed S mean", "worst_case_stress_relative_l2": "signed S worst",
                     "case_mean_stress_hotspot_relative_l2": "hotspot mean", "worst_case_stress_hotspot_relative_l2": "hotspot worst"}
    return {"kind": EXPORT_KIND, "scientific_digest": SCIENCE, "fixed_case_keys": keys,
            "fixed_TRAIN_denominator": 120, "arms": arms, "old_FNO4800": old,
            "native_to_grid_ratios": ratios, "new_to_old_ratios": old_ratios,
            "new_metrics_worse_than_old": {name: [metric for metric in AGGREGATES
                 if finite(old_ratios[metric][name], True) and old_ratios[metric][name] > 1.] for name in ARMS},
            "parameters_vs_old": {name: {"new_minus_old": arms[name]["parameters"] - old["parameters"]
                  if arms[name]["parameters"] is not None and old["parameters"] is not None else None,
                  "new_to_old_ratio": arms[name]["parameters"] / old["parameters"]
                  if arms[name]["parameters"] is not None and old["parameters"] is not None else None} for name in ARMS},
            "named_initial_pair": {"exact_named_tensor_identity_equal": named_equal,
                "summary_claim": summary.get("same_initial_state_proven"),
                "named_tensor_count": len(named.get("parameters", {})) if isinstance(named, dict) else 0,
                "all_named_trainable_sha256": named.get("all_named_trainable_sha256") if isinstance(named, dict) else None},
            "costs_and_budgets": budgets, "both_fixed4800_qualified": bool(qualified),
            "protocol_qualification": bool(qualified),
            "scientific_qualification": bool(reproduced_gate),
            "saved_completion_gate": clean(summary.get("completion_gate")),
            "reproduced_TRAIN_representation_gate_passed": bool(reproduced_gate),
            "gate_matches_saved": summary.get("completion_gate", {}).get("passed") == bool(reproduced_gate),
            "metric_labels": metric_labels, "summary_status": summary["status"],
            "terminal_error": clean(summary.get("error")),
            "interpretation": "native25 versus grid25 changes only local25 origin within the new decoder; compare absolute old FNO errors/cost/parameters before any further experiment. TRAIN120 is not held20/generalization evidence; pair gate does not choose a solver.",
            "held20_results_used": False, "inference_rerun": False, "solver_selected": False}


def saved_loss(filename, receipt):
    filename = path(filename)
    if not filename.is_file():
        return {"status": "missing", "reason": "No saved loss.jsonl", "records": []}
    require(filename.suffix == ".jsonl", "Only saved scalar loss JSONL is allowed")
    member = {"path": relative(filename), "bytes": filename.stat().st_size,
              "sha256": sha(filename), "read_kind": "saved_loss_scalars", "records": 0}
    receipt.append(member)
    records = []
    try:
        with filename.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                item = json.loads(line)
                require(type(item.get("step")) is int and 1 <= item["step"] <= 4800
                        and finite(item.get("encoded4_case_mean_loss"), True)
                        and finite(item.get("training_seconds"), True), "Saved loss scalar schema is invalid")
                records.append({name: item.get(name) for name in
                                ("step", "case", "nodes", "encoded4_case_mean_loss", "training_seconds", "attempt")})
                member["records"] = len(records)
    except (ValueError, NotReady) as error:
        return {"status": "invalid", "reason": str(error), "invalid_line": line_number, "records": []}
    return {"status": "saved" if records else "empty", "records": records,
            "duplicate_step_records": len(records) - len({item["step"] for item in records}),
            "reason": None if records else "No completed-update loss records"}


def plots(output, metrics, losses):
    """Lazy Agg CPU figures from saved scalars; missing points remain missing."""
    result = {"backend": "MatplotlibAgg", "loss": {"status": "pending"}, "aggregate": {"status": "pending"}}
    os.environ["MPLCONFIGDIR"] = str(output / ".mpl-cache")
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as error:
        return {"backend": "MatplotlibAgg", "loss": {"status": "unavailable", "reason": str(error)},
                "aggregate": {"status": "unavailable", "reason": str(error)}}
    colors = {ARMS[0]: "#245E9B", ARMS[1]: "#D66B24"}
    labels = {ARMS[0]: "New grid25", ARMS[1]: "New native25"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    def save(fig, stem):
        names = []
        for extension in ("png", "svg"):
            filename = output / (stem + "." + extension)
            temporary = output / (stem + ".tmp." + extension)
            fig.savefig(temporary, dpi=180, bbox_inches="tight")
            os.replace(temporary, filename)
            names.append(relative(filename))
        plt.close(fig)
        return names
    try:
        present = [name for name in ARMS if losses[name]["status"] == "saved"]
        if not present:
            result["loss"] = {"status": "missing", "reason": "Neither arm has valid saved loss scalars"}
        else:
            fig, ax = plt.subplots(figsize=(10.5, 4.0))
            for name in present:
                records = losses[name]["records"]
                attempts = sorted({item.get("attempt") or 1 for item in records})
                for index, attempt in enumerate(attempts):
                    selected = [item for item in records if (item.get("attempt") or 1) == attempt]
                    label = labels[name] + ("; attempt " + str(attempt) if len(attempts) > 1 else "")
                    ax.plot([item["step"] for item in selected], [item["encoded4_case_mean_loss"] for item in selected],
                            color=colors[name], linewidth=.65, alpha=.8, label=label)
            ax.set(xlabel="Completed full-case optimizer step", ylabel="Encoded4 complete-case MSE",
                   title="Saved TRAIN120 losses; raw completed-update scalars", xlim=(0, 4800))
            all_losses = [item["encoded4_case_mean_loss"] for name in present for item in losses[name]["records"]]
            if min(all_losses) > 0 and max(all_losses) / min(all_losses) > 20:
                ax.set_yscale("log")
                ax.set_ylabel("Encoded4 complete-case MSE (log scale)")
            ax.grid(alpha=.2); ax.legend(frameon=False)
            result["loss"] = {"status": "saved", "files": save(fig, "loss_comparison"),
                              "smoothing": False, "scalar_records": {name: len(losses[name]["records"]) for name in ARMS}}
        fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0))
        availability = {}
        plot_metrics = tuple(prefix + field + "_relative_l2" for prefix in ("case_mean_", "worst_case_")
                             for field in ("u", "stress", "stress_hotspot"))
        for ax, metric in zip(axes.flat, plot_metrics):
            positions, tick_labels, values = [], [], []
            for arm_index, name in enumerate(ARMS):
                for step_index, step in enumerate((0, 4800)):
                    x = arm_index * 2 + step_index
                    value = metrics["arms"][name]["fixed" + str(step)]["aggregate"][metric]
                    positions.append(x); tick_labels.append(labels[name].replace("New ", "") + "\n" + str(step)); values.append(value)
                    if finite(value, True):
                        ax.bar(x, value, color=colors[name], alpha=.65 if step == 0 else 1., width=.68)
                    else:
                        ax.text(x, .02, "missing", rotation=90, ha="center", va="bottom", transform=ax.get_xaxis_transform(), color="#777777", fontsize=8)
            old = metrics["old_FNO4800"]["fixed4800"]["aggregate"][metric]
            if finite(old, True):
                ax.axhline(old, color="#333333", linestyle="--", linewidth=1., label="Old FNO4800")
            ax.set(title=metrics["metric_labels"][metric], ylabel="Full120 case-relative L2", xticks=positions, xticklabels=tick_labels)
            actual_values = [value for value in values + [old] if finite(value, True)]
            if actual_values and min(actual_values) > 0 and max(actual_values) / min(actual_values) > 20:
                ax.set_yscale("log")
                ax.set_ylabel("Full120 case-relative L2 (log scale)")
            ax.grid(axis="y", alpha=.2)
            availability[metric] = {"new0_4800_values": values, "old4800_reference": old}
        axes.flat[0].legend(frameon=False, fontsize=9)
        fig.suptitle("Saved fixed0/fixed4800 fullTRAIN120 metrics; dashed old FNO4800 reference", y=1.02)
        fig.tight_layout()
        result["aggregate"] = {"status": "saved", "files": save(fig, "aggregate_comparison"),
                               "fixed_denominator": 120, "missing_values_plotted_as_zero": False, "values": availability}
    except Exception as error:
        plt.close("all")
        if result["loss"]["status"] == "pending":
            result["loss"] = {"status": "failed", "reason": str(error)}
        if result["aggregate"]["status"] == "pending":
            result["aggregate"] = {"status": "failed", "reason": str(error)}
    return result


def format_value(value, digits=4):
    return format(value, "." + str(digits) + "g") if finite(value) else "null"


def brief(metrics, plot_status):
    left, right = (metrics["arms"][name] for name in ARMS)
    if metrics["both_fixed4800_qualified"]:
        opening = "两臂完成固定4800更新及全部TRAIN120读出。"
    else:
        opening = "此次真实终态未满足两臂固定4800及完整TRAIN120资格；缺失的固定端点指标记为null，分母保持120。"
    gate = "通过" if metrics["reproduced_TRAIN_representation_gate_passed"] else "未通过"
    lines = ["# FNO局部条件表示配对：保存标量导出", "", opening + " native25与grid25的唯一局部因子是原节点features25或同一输入网格的roundtrip features25；新decoder其余结构、初始张量、曝光规则和scaler相同。", "",
             "TRAIN表示推进门" + gate + "，该相对结果不自动选择新decoder。应同时对照旧FNO4800的绝对U、signed S、top1%abs原S hotspot误差、最坏病例、成本与参数数。TRAIN120不是held20结果，也不提供总体泛化、家族上限或工程资格。", "",
             "| 固定4800参考 | U均值 / 最坏 | signed S均值 / 最坏 | hotspot均值 / 最坏 | 参数 | monotonic训练秒 / arm秒 |",
             "|---|---:|---:|---:|---:|---:|"]
    rows = [("新grid25", left["fixed4800"], left["parameters"], left["costs"].get("training_seconds"), left["costs"].get("arm_wall_seconds")),
            ("新native25", right["fixed4800"], right["parameters"], right["costs"].get("training_seconds"), right["costs"].get("arm_wall_seconds")),
            ("旧FNO4800", metrics["old_FNO4800"]["fixed4800"], metrics["old_FNO4800"]["parameters"],
             metrics["old_FNO4800"]["training_seconds"], metrics["old_FNO4800"]["training_attempt_wall_seconds"])]
    for label, endpoint, parameters, training, wall in rows:
        a = endpoint["aggregate"]
        pairs = [format_value(a["case_mean_" + prefix + "_relative_l2"]) + " / "
                 + format_value(a["worst_case_" + prefix + "_relative_l2"]) for prefix in ("u", "stress", "stress_hotspot")]
        lines.append("| " + " | ".join([label, *pairs, str(parameters) if parameters is not None else "null",
                     format_value(training, 6) + " / " + format_value(wall, 6)]) + " |")
    costs = metrics["costs_and_budgets"]
    worse = metrics["new_metrics_worse_than_old"][ARMS[1]]
    if worse:
        lines += ["", "新native25仍高于旧FNO的完整120指标：" + "、".join(metrics["metric_labels"][name] for name in worse)
                  + "；即使配对门通过，也须保留旧FNO作为绝对参考，不能据此自动替换decoder。"]
    lines += ["", "新pair保存计时：native闭合monotonic累计 " + format_value(costs["actual_inner_wall_seconds"], 7)
              + " 秒；外guard launch至child退出monotonic " + format_value(costs["actual_guard_launch_to_child_exit_seconds"], 7)
              + " 秒，真实epoch/UTC " + format_value(costs["actual_guard_epoch_launch_to_child_exit_seconds"], 7)
              + " 秒，两轴差(epoch−monotonic) " + format_value(costs["clock_axis_difference_seconds"], 7)
              + " 秒。预算要求两条外轴各≤9000秒、native累计<9000秒，各臂原monotonic训练≤3600秒；外timeout=" + str(costs["external_timeout"])
              + "，exit=" + str(costs["external_exit_code"]) + "，cost accounting exact=" + str(costs["cost_accounting_exact"])
              + "。未知区间保留为nonexact成本。native各phase/arm/训练保留原monotonic值；phase/arm/训练的epoch/UTC耗时未记录，均为null，不按总轴差缩放。旧FNO表中wall是其保存的原训练attempt，读出成本单列于metrics，二者不冒充相同总计算预算。", "",
              "同名初始张量完全一致=" + str(metrics["named_initial_pair"]["exact_named_tensor_identity_equal"])
              + "；真实首个完整TRAIN反传通过（grid/native）=" + str(left["first_actual_gradient_passed"])
              + "/" + str(right["first_actual_gradient_passed"]) + "。具体梯度、参数组、曝光、停止原因、预算和缺失原因见[metrics.json](metrics.json)。",
              "", "图仅使用保存的loss及fixed0/4800 full120聚合；不重跑推理。loss图状态=" + plot_status["loss"]["status"]
              + "，聚合图状态=" + plot_status["aggregate"]["status"] + "。"]
    artifacts = metrics["export_artifacts"]
    lines += ["", "导出产物状态=" + artifacts["status"]
              + "；protocol qualification=" + str(metrics["protocol_qualification"])
              + "；TRAIN scientific qualification=" + str(metrics["scientific_qualification"])
              + "。产物完整性不代表实验PASS。"]
    if artifacts["incomplete_reasons"]:
        lines.append("缺失产物原因：" + "；".join(artifacts["incomplete_reasons"]) + "。")
    for name in ARMS:
        reason = metrics["arms"][name]["fixed4800"]["reason"]
        if reason:
            lines.append("- " + name + "：actual step=" + str(metrics["arms"][name]["actual_endpoint_step"]) + "；" + reason)
    links = ["[scalar CSV](scalars.csv)"]
    for group in ("loss", "aggregate"):
        for filename in plot_status[group].get("files", []):
            name = Path(filename).name
            links.append("[" + name + "](" + name + ")")
    lines += ["", " · ".join(links), ""]
    return "\n".join(lines)


def write_scalars(output, metrics, losses):
    filename = output / "scalars.csv"
    temporary = output / "scalars.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "fixed_update", "case", "metric", "value", "fixed_denominator", "complete_full120", "reason"])
        endpoints = [(name, step, metrics["arms"][name]["fixed" + str(step)]) for name in ARMS for step in (0, 4800)]
        endpoints.append((OLD, 4800, metrics["old_FNO4800"]["fixed4800"]))
        for name, step, endpoint in endpoints:
            for metric in AGGREGATES:
                value = endpoint["aggregate"][metric]
                writer.writerow([name, step, "ALL120", metric, value if finite(value) else "null", 120,
                                 endpoint["all120_metrics_complete"], endpoint["reason"] or ""])
            for key, record in endpoint["per_case"].items():
                for metric in CASE_SCALARS:
                    value = record.get(metric) if isinstance(record, dict) else None
                    writer.writerow([name, step, key, metric, value if finite(value, True) else "null", 120,
                                     endpoint["all120_metrics_complete"], endpoint["reason"] or ""])
        for name in ARMS:
            for item in losses[name]["records"]:
                writer.writerow([name, item["step"], item.get("case"), "encoded4_case_mean_loss",
                                 item["encoded4_case_mean_loss"], 120, "saved_update_only", ""])
    os.replace(temporary, filename)


def export_saved(plan_path=PLAN, export=False):
    """Default zero IO; actual mode exports only genuine saved terminal scalars."""
    if not export:
        return {"kind": EXPORT_KIND, "status": "source_only", "actual_export": False,
                "reads": 0, "writes": 0, "scientific_array_reads": 0, "GPU_calls": 0}
    began = time.monotonic()
    inputs = terminal_inputs(plan_path)
    losses = {name: saved_loss(path(inputs["plan"]["output"]) / name / "loss.jsonl", inputs["input_receipt"]) for name in ARMS}
    output = path(OUTPUT)
    output.relative_to(ROOT.resolve())
    binding = {"exporter_source_sha256": sha(OWN), "scientific_digest": SCIENCE,
               "inputs": inputs["input_receipt"], "output": OUTPUT}
    existing = output / "manifest.json"
    if existing.is_file():
        prior = read_json(existing)
        require(prior.get("status") == "completed",
                "Incomplete owned export is preserved; root review is required before replacement")
        require(prior.get("binding") == binding,
                "Owned export cache identity differs; root review is required before replacement")
        for name, item in prior["files"].items():
            require((output / name).is_file() and sha(output / name) == item["sha256"], "Owned cached export artifact changed")
        return prior
    require(not output.exists() or not any(output.iterdir()), "Interrupted owned export exists; preserve it for root review")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "ownership.json", {"kind": EXPORT_KIND, "binding": binding, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    metrics = build_metrics(inputs)
    plot_status = plots(output, metrics, losses)
    metrics["plot_status"] = plot_status
    incomplete_reasons = [group + " plot " + plot_status[group]["status"] + ": "
                          + plot_status[group].get("reason", "Required plot was not saved")
                          for group in ("loss", "aggregate") if plot_status[group]["status"] != "saved"]
    artifact_status = "completed_incomplete" if incomplete_reasons else "completed"
    metrics["export_artifacts"] = {"status": artifact_status, "complete": not incomplete_reasons,
                                   "incomplete_reasons": incomplete_reasons}
    metrics["loss_status"] = {name: {key: value for key, value in losses[name].items() if key != "records"} for name in ARMS}
    metrics["export_read_receipt"] = {"members": inputs["input_receipt"], "scientific_array_reads": 0,
                                     "checkpoint_body_reads": 0, "raw_source_reads": 0, "numeric_GT_parses": 0,
                                     "model_constructions": 0, "inference_calls": 0, "GPU_calls": 0}
    write_json(output / "metrics.json", clean(metrics))
    (output / "BRIEF.md").write_text(brief(metrics, plot_status), encoding="utf-8")
    write_scalars(output, metrics, losses)
    files = {}
    for filename in output.iterdir():
        if filename.is_file() and filename.name not in ("manifest.json", "ownership.json"):
            files[filename.name] = {"bytes": filename.stat().st_size, "sha256": sha(filename)}
    # Loss-file identities belong to the cache binding as well as the receipt.
    binding["inputs"] = inputs["input_receipt"]
    manifest = {"kind": EXPORT_KIND, "status": artifact_status, "binding": binding, "files": files,
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "CPU_export_seconds": time.monotonic() - began,
                "CPU_export_clock_axis": "monotonic",
                "actual_export": True, "fixed_TRAIN_denominator": 120, "fixed_update": 4800,
                "export_artifacts_complete": not incomplete_reasons, "incomplete_reasons": incomplete_reasons,
                "both_fixed4800_qualified": metrics["both_fixed4800_qualified"],
                "protocol_qualification": metrics["protocol_qualification"],
                "scientific_qualification": metrics["scientific_qualification"],
                "TRAIN_representation_gate_passed": metrics["reproduced_TRAIN_representation_gate_passed"],
                "plot_status": plot_status, "solver_selected": False, "held20_results_used": False,
                "read_receipt": metrics["export_read_receipt"],
                "actual_terminal_exit_proof": {"guard_process": inputs["launch"]["guard_process"],
                    "child_process": inputs["launch"]["child_process"], "clock_closed_inactive": True,
                    "both_exact_processes_exited": True}}
    write_json(output / "manifest.json", clean(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    result = export_saved(args.plan, args.export)
    print(json.dumps({"status": result["status"], "kind": EXPORT_KIND,
                      "actual_export": bool(args.export)}, ensure_ascii=False))
    return 1 if result["status"] == "completed_incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
