"""Bounded, TRAIN-only readout of five immutable coverage-study endpoints.

Imports are standard-library only. No model, array or target is opened until
the separate root release, live coordinator proof and shared GPU lock agree.
This is a final-endpoint fit diagnostic, never a DEV selection experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
OWN = "geobc/coverage_train_fit_diagnostic.py"
PLAN = "notes/geobc_train_fit_diagnostic_plan.json"
KIND = "coverage_TRAIN120_fixed4800_fit_diagnostic"
DECISION = "proceed_frozen_TRAIN120_fixed4800_diagnostic"
METHODS = ("transolver", "transolver_pp", "ts3", "fno", "mlp")
ZERO = ("training_updates", "backward_calls", "optimizer_steps", "optimizer_state_restores",
        "scaler_fits", "DEV_target_reads", "old_DEV_target_reads", "raw_source_reads",
        "old_final_reads", "sealed_test_reads", "DEV_dia_reads", "model_selection_updates")
ACTUAL = ("model_initializations", "fixed_checkpoint_read_attempts", "fixed_checkpoint_reads",
          "checkpoint_sha256_reads", "model_state_loads", "input_load_attempts",
          "input_load_completed", "input_array_reads", "cached_TRAIN_target_array_reads",
          "fullN_prediction_attempts", "fullN_predictions", "physical_metric_cases")
STAGE_COSTS = ("model_initialization_seconds", "checkpoint_CPU_load_identity_H2D_seconds",
               "input_and_H2D_seconds", "fullN_forward_seconds", "physical_decode_seconds",
               "physical_metrics_seconds", "persistence_seconds", "model_cleanup_seconds")


class DiagnosticNotReady(RuntimeError):
    pass


class DiagnosticBudget(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", value):
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else ROOT / candidate).resolve()


def read(filename):
    return json.loads(path(filename).read_text(encoding="utf-8-sig"))


def write(filename, value):
    filename = path(filename); filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(filename.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, filename)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def sha(filename):
    value = hashlib.sha256()
    with path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message, error=ValueError):
    if not condition:
        raise error(message)


def stat_identity(filename):
    filename = path(filename); info = filename.stat()
    require(filename.is_file() and info.st_size > 0, "Required endpoint file is empty")
    return {"path": str(filename), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def request_contract(plan_path=PLAN):
    """Source/small-JSON/stat only; checkpoint bodies and targets are deferred."""
    plan = read(plan_path); science = plan["scientific"]
    require(plan["kind"] == KIND and digest(science) == plan["scientific_digest"], "Diagnostic science differs")
    require(plan.get("scientific_rules_frozen") is True, "Diagnostic protocol has not been frozen")
    require(science["fixed_step"] == 4800 and science["seed"] == 2026
            and science["budget_seconds"] == 600 and science["fixed_arm_denominator"] == 5
            and science["fixed_case_denominator_each_arm"] == 120
            and science["fixed_prediction_denominator"] == 600, "Fixed diagnostic scope differs")
    keys = science["TRAIN_keys"]
    require(len(keys) == 120 and len(set(keys)) == 120 and len({k.split("/")[0] for k in keys}) == 40
            and all({cad + "/" + load for load in ("ver", "hor", "tor")} <= set(keys)
                    for cad in {k.split("/")[0] for k in keys}), "TRAIN40 three-load inventory differs")
    parent = read(plan["parent_plan"])
    require(digest(parent["scientific"]) == science["parent_scientific_digest"] == parent["scientific_digest"], "Parent science changed")
    require(parent["source_sha256"] == plan["parent_source_sha256"], "The frozen twenty-source map changed")
    pins = {name: sha(name) for name in plan["parent_source_sha256"]}
    require(pins == plan["parent_source_sha256"], "Frozen dependency source changed")
    pins[OWN] = sha(OWN)
    from . import coverage_comparison_train as train
    from . import coverage_comparison_data as data
    manifest = data.read_bootstrap_TRAIN40(plan["parent_plan"])
    require(manifest["train_keys"] == {"40": keys} and set(manifest["records"]) == set(keys)
            and manifest["scaler"] == science["scaler"] and manifest["complete_shared660"] is False,
            "Only the immutable original TRAIN120 bootstrap is allowed")
    selected = [item for item in train.arms(parent) if item["train_CADs"] == 40]
    require([item["method"] for item in selected] == list(METHODS)
            and selected == science["arms"], "Five fixed TRAIN40 arms differ")
    endpoint_inventory = {}
    for arm in selected:
        folder = train.arm_output(parent, arm)
        try:
            summary = read(folder / "summary.json"); endpoint = read(folder / "step4800.json")
            checkpoint_stat = stat_identity(folder / "step4800.pt")
        except FileNotFoundError as error:
            raise DiagnosticNotReady("All five fixed4800 endpoints must finish before this diagnostic") from error
        original_contract, _ = train.request_contract(plan["parent_plan"], arm["method"], 40, "train")
        require(summary["status"] == "completed" and summary.get("completed4800") is True
                and summary.get("endpoint_step") == 4800 and 4800 in summary.get("available_checkpoints", [])
                and summary["config"]["contract"] == original_contract
                and summary.get("method") == arm["method"] and summary.get("train_CADs") == 40,
                "A fixed endpoint is incomplete or has a different original contract", DiagnosticNotReady)
        require(endpoint.get("status") == "completed" and endpoint.get("step") == 4800
                and digest(endpoint["config"]) == digest(summary["config"])
                and path(endpoint["checkpoint"]) == path(checkpoint_stat["path"])
                and endpoint["case_exposure_counts"] == {key: 40 for key in keys}, "Endpoint metadata/configuration differs")
        endpoint_inventory[arm["id"]] = {
            "arm": arm, "summary": {"path": str(folder / "summary.json"), "sha256": sha(folder / "summary.json")},
            "checkpoint_metadata": {"path": str(folder / "step4800.json"), "sha256": sha(folder / "step4800.json")},
            "checkpoint_file_stat": checkpoint_stat, "original_contract_sha256": digest(original_contract),
            "original_config_sha256": digest(summary["config"])}
    contract = {"kind": KIND, "schema": 1, "scientific_digest": plan["scientific_digest"],
                "parent_scientific_digest": science["parent_scientific_digest"], "source_sha256": pins,
                "TRAIN_keys": keys, "arms": selected, "fixed_step": 4800, "fixed_denominator_each_arm": 120,
                "input_contract_sha256": digest(manifest["contract"]), "scaler": manifest["scaler"],
                "endpoint_inventory": endpoint_inventory, "read_scope": "original TRAIN120 only"}
    return contract, plan


def release(contract, plan):
    marker = read(plan["root_review"])
    require(marker.get("passed") is True and marker.get("diagnostic_released") is True
            and marker.get("decision") == DECISION and marker.get("source_sha256") == contract["source_sha256"][OWN]
            and marker.get("source_pins") == contract["source_sha256"]
            and marker.get("scientific_digest") == contract["scientific_digest"]
            and marker.get("parent_scientific_digest") == contract["parent_scientific_digest"],
            "A genuine, separate root release is required")


def live_process(pid):
    if type(pid) is not int or pid <= 0:
        return None
    try:
        folder = Path("/proc") / str(pid)
        raw = (folder / "stat").read_text(encoding="utf-8")
        tail = raw[raw.rfind(")") + 2:].split()
        if tail[0] in ("Z", "X"):
            return None
        argv = (folder / "cmdline").read_bytes().split(b"\0")
        return {"pid": pid, "start_ticks": int(tail[19]),
                "argv": [item.decode("utf-8", errors="strict") for item in argv if item]}
    except (OSError, ValueError, UnicodeError, IndexError):
        return None


def waiting_coordinator(plan):
    """Real Linux PID/argv/starttime and no live child; no GPU resource probe."""
    require(os.name != "nt" and Path("/proc").is_dir(), "Actual diagnostic requires the common WSL process namespace", DiagnosticNotReady)
    filename = path(plan["coordinator_status"])
    record = read(filename)
    require(record.get("phase") == "waiting_for_acquisition", "Main coordinator is not CPU-waiting for acquisition", DiagnosticNotReady)
    process = live_process(record.get("pid"))
    require(process is not None, "Main coordinator is no longer alive", DiagnosticNotReady)
    tokens = process["argv"]
    correct_script = any(Path(token).name == "run_geobc_coverage_comparison.py"
                         and path(token) == path("tools/run_geobc_coverage_comparison.py") for token in tokens)
    correct_module = any(token == "tools.run_geobc_coverage_comparison" and i > 0 and tokens[i - 1] == "-m"
                         for i, token in enumerate(tokens))
    require(correct_script or correct_module, "Coordinator PID belongs to a different program", DiagnosticNotReady)
    try:
        process_cwd = Path(os.readlink(Path("/proc") / str(process["pid"]) / "cwd")).resolve()
    except OSError as error:
        raise DiagnosticNotReady("Cannot prove the coordinator workspace") from error
    require(process_cwd == ROOT, "Live coordinator belongs to a different workspace", DiagnosticNotReady)
    require("--start" in tokens and "--plan" in tokens
            and path(tokens[tokens.index("--plan") + 1]) == path(plan["parent_plan"]), "Live coordinator arguments differ", DiagnosticNotReady)
    child_file = Path("/proc") / str(process["pid"]) / "task" / str(process["pid"]) / "children"
    try:
        children = [int(item) for item in child_file.read_text(encoding="utf-8").split()]
    except OSError as error:
        raise DiagnosticNotReady("Cannot prove that the coordinator has no active child") from error
    # Treat even an unreadable/not-yet-reaped child as active; uncertainty must
    # never be interpreted as permission to read a scientific input.
    active = [child for child in children if (Path("/proc") / str(child)).exists()]
    require(not active and live_process(record.get("child_pid")) is None,
            "A coordinator child is still active; diagnostic must defer", DiagnosticNotReady)
    return {"status_path": str(filename), "phase": record["phase"], "pid": process["pid"],
            "start_ticks": process["start_ticks"], "argv": tokens, "cwd": str(process_cwd), "live_children": [],
            "observed_at_utc": utc(), "updated_at_utc": record.get("updated_at_utc")}


def without_selection(value):
    value = dict(value); value.pop("selection_score", None)
    return value


def CAD_aggregate(per_case, train_keys):
    result = {}
    for cad in dict.fromkeys(key.split("/")[0] for key in train_keys):
        keys = [cad + "/" + load for load in ("ver", "hor", "tor")]
        item = {"fixed_load_denominator": 3, "case_keys": keys,
                "metrics_available_cases": sum(per_case[key] is not None for key in keys)}
        for metric in ("u_relative_l2", "stress_relative_l2", "stress_hotspot_relative_l2"):
            values = [None if per_case[key] is None else per_case[key].get(metric) for key in keys]
            valid = all(isinstance(v, (float, int)) and math.isfinite(v) and v >= 0 for v in values)
            item["three_load_mean_" + metric] = sum(values) / 3 if valid else None
        result[cad] = item
    return result


def brief(summary, filename):
    def percent(value):
        return "缺失" if value is None else f"{100 * value:.2f}%"
    lines = ["仅固定4800端点的原TRAIN120拟合诊断；不包含DEV，也不作方法上限或最终求解器选择。",
             f"完成全节点病例 {summary['completed_predictions']}/600；预算状态：{summary['stop_reason']}。",
             "", "| 方法 | TRAIN均值 U / S | 热点均值 / 最坏S | 本次诊断秒 |", "|---|---:|---:|---:|"]
    for arm in summary["arms"].values():
        metrics = arm["aggregate"]
        lines.append(f"| {arm['method']} | {percent(metrics['case_mean_u_relative_l2'])} / {percent(metrics['case_mean_stress_relative_l2'])} | "
                     f"{percent(metrics['case_mean_stress_hotspot_relative_l2'])} / {percent(metrics['worst_case_stress_relative_l2'])} | {arm['costs']['observed_arm_wall_seconds']:.2f} |")
    filename = path(filename); temporary = filename.with_name(filename.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8"); os.replace(temporary, filename)


def run(plan_path=PLAN, output=None, start=False):
    plan = read(plan_path)
    if not start:
        return {"status": "source_only_ready", "kind": KIND, "scientific_execution": False,
                "actual_execution": "Only --start after the authentic root diagnostic release"}
    output = path(output or plan["output"])
    require(output == path(plan["output"]) and ROOT in output.parents,
            "The diagnostic may only write its independently owned project output")
    contract, plan = request_contract(plan_path); release(contract, plan)
    if (output / "summary.json").exists():
        previous = read(output / "summary.json")
        require(previous.get("status") == "completed" and previous["config"]["contract"] == contract,
                "Existing terminal diagnostic identity differs")
        return previous
    require(not (output / "clock.json").exists(), "An interrupted diagnostic cannot automatically reset or replay")
    before = waiting_coordinator(plan)
    from . import coverage_comparison_train as train
    try:
        held = train.lock(plan["shared_GPU_lock"]); held.__enter__()
    except (OSError, RuntimeError) as error:
        raise DiagnosticNotReady("Shared GPU lock is busy; no scientific input was read") from error
    began = time.monotonic(); start_epoch = time.time(); receipt = {name: 0 for name in ZERO + ACTUAL}
    receipt.update(input_read_receipt_exact=True, serialized_optimizer_payload_loaded_on_CPU=False,
                   optimizer_state_used=False, read_scope="cached original TRAIN120 only")
    costs = {name: 0. for name in STAGE_COSTS}; results = {}; torch = None; reason = "fixed_final_readout_complete"
    cleanup = None; global_stop = False
    try:
        after = waiting_coordinator(plan)
        require((before["pid"], before["start_ticks"]) == (after["pid"], after["start_ticks"]),
                "Coordinator identity changed while acquiring the GPU lock", DiagnosticNotReady)
        refreshed, _ = request_contract(plan_path)
        require(refreshed == contract, "Endpoint/input/source identities changed before scientific reads")
        output.mkdir(parents=True, exist_ok=True)
        def elapsed():
            return time.monotonic() - began
        def persist(filename, value):
            t = time.monotonic(); write(filename, value); costs["persistence_seconds"] += time.monotonic() - t
        def clock(active=True):
            persist(output / "clock.json", {"kind": KIND, "active": active,
                    "started_at_utc": datetime.fromtimestamp(start_epoch, timezone.utc).isoformat(),
                    "elapsed_from_lock_acquisition_seconds": elapsed(), "budget_seconds": 600,
                    "last_updated_at_utc": utc(), "automatic_resume_or_reset": False})
        def guard():
            if elapsed() >= 600:
                raise DiagnosticBudget("Cumulative lock-held diagnostic wall reached600s")
        def checkpoint_identity(entry):
            require(stat_identity(entry["checkpoint_file_stat"]["path"]) == entry["checkpoint_file_stat"], "Fixed checkpoint stat changed")
            require(sha(entry["summary"]["path"]) == entry["summary"]["sha256"]
                    and sha(entry["checkpoint_metadata"]["path"]) == entry["checkpoint_metadata"]["sha256"], "Fixed endpoint metadata changed")
        persist(output / "config.json", {"contract": contract, "scope": "TRAIN-diagnostic only", "selection_score_used": False})
        persist(output / "coordinator_wait_proof.json", {"before_lock": before, "after_lock": after,
                "shared_GPU_lock": str(path(plan["shared_GPU_lock"])), "lock_acquired": True})
        clock()
        from . import coverage_comparison_data as data
        manifest = data.read_bootstrap_TRAIN40(plan["parent_plan"])
        keys = contract["TRAIN_keys"]
        for arm in contract["arms"]:
            per = {key: None for key in keys}; errors = {key: "not_attempted" for key in keys}
            arm_costs = {name: 0. for name in STAGE_COSTS}; arm_started = time.monotonic()
            entry = contract["endpoint_inventory"][arm["id"]]
            parent_summary = read(entry["summary"]["path"])
            item = {"method": arm["method"], "train_CADs": 40, "checkpoint_step": 4800,
                    "per_case": per, "case_failures": errors, "costs": arm_costs,
                    "per_case_costs": {},
                    "original_training_costs": parent_summary["costs"],
                    "original_training_elapsed_this_attempt_seconds": parent_summary.get("elapsed_this_attempt_seconds"),
                    "original_training_summary": entry["summary"], "checkpoint": dict(entry["checkpoint_file_stat"])}
            results[arm["id"]] = item
            model = None; cleanup = None; np = None
            def charge(name, started):
                seconds = time.monotonic() - started; costs[name] += seconds; arm_costs[name] += seconds
            try:
                if global_stop:
                    raise DiagnosticBudget("Remaining fixed cases were not attempted after terminal stage failure")
                guard(); checkpoint_identity(entry)
                t = time.monotonic(); receipt["model_initializations"] += 1
                import torch as diagnostic_torch
                diagnostic_torch.set_grad_enabled(False)
                np, torch, model, scaler, cleanup, _, _, _, predict, query_grid = train.model_environment(manifest, arm["method"])
                torch.set_grad_enabled(False); torch.cuda.synchronize(); charge("model_initialization_seconds", t); guard()
                require(digest(model.source_config) == digest(parent_summary["config"]["model"])
                        and digest(model.parameter_counts) == digest(parent_summary["config"]["parameter_counts"]), "Original model constructor/configuration differs")
                t = time.monotonic(); receipt["fixed_checkpoint_read_attempts"] += 1
                saved = torch.load(path(entry["checkpoint_file_stat"]["path"]), map_location="cpu", weights_only=False)
                receipt["fixed_checkpoint_reads"] += 1
                receipt["serialized_optimizer_payload_loaded_on_CPU"] = "optimizer" in saved
                require(saved.get("step") == 4800 and saved.get("next_case_position") == 4800
                        and digest(saved["config"]) == entry["original_config_sha256"]
                        and saved["case_exposure_counts"] == {key: 40 for key in keys}, "Fixed checkpoint contents/configuration differ")
                model.load_state_dict(saved["model"], strict=True); receipt["model_state_loads"] += 1
                for parameter in model.parameters(): parameter.requires_grad_(False)
                model.eval(); torch.cuda.synchronize()
                receipt["checkpoint_sha256_reads"] += 1
                item["checkpoint"]["sha256"] = sha(entry["checkpoint_file_stat"]["path"])
                checkpoint_identity(entry); del saved
                charge("checkpoint_CPU_load_identity_H2D_seconds", t); guard()
                with torch.no_grad():
                    for index, key in enumerate(keys):
                        case = None; physical = None; encoded = None
                        case_started = time.monotonic(); case_cost_before = dict(arm_costs)
                        try:
                            guard(); require(key in keys and key in manifest["records"], "TRAIN whitelist violated")
                            persist(output / "status.json", {"status": "running", "kind": KIND,
                                    "arm_id": arm["id"], "case_key": key, "case_index": index,
                                    "elapsed_from_lock_acquisition_seconds": elapsed(), "TRAIN_only": True})
                            t = time.monotonic(); receipt["input_load_attempts"] += 1
                            try:
                                case = data.load_case(manifest, key, with_targets=True, device="cuda", for_fno=arm["method"] == "fno")
                            except Exception:
                                receipt["input_read_receipt_exact"] = False; raise
                            receipt["input_load_completed"] += 1
                            actual = case["_actual_read_receipt"]
                            require(actual["DEV_target_reads"] == 0 and actual["source_archive_reads"] == 0 and actual["scaler_fits"] == 0,
                                    "Unexpected scientific read scope")
                            receipt["input_array_reads"] += actual["input_array_reads"]
                            receipt["cached_TRAIN_target_array_reads"] += actual["cached_TRAIN_target_array_reads"]
                            del case["target"]  # This diagnostic never computes a training loss.
                            torch.cuda.synchronize(); charge("input_and_H2D_seconds", t); guard()
                            torch.manual_seed(2026 + int(hashlib.sha256(f"TRAIN:4800:{key}".encode()).hexdigest()[:8], 16))
                            t = time.monotonic(); receipt["fullN_prediction_attempts"] += 1
                            encoded = predict(torch, model, arm["method"], case, query_grid)
                            torch.cuda.synchronize(); charge("fullN_forward_seconds", t); guard()
                            t = time.monotonic(); physical = scaler.decode(encoded)
                            require(physical.shape == (len(case["xyz"]), 4) and bool(np.isfinite(physical).all()), "Prediction is not finite fullN physical4")
                            receipt["fullN_predictions"] += 1; charge("physical_decode_seconds", t); guard()
                            t = time.monotonic(); metrics = train.physical_metrics(np, case["physical"], physical)
                            receipt["physical_metric_cases"] += 1; charge("physical_metrics_seconds", t)
                            per[key] = metrics; errors[key] = None
                        except DiagnosticBudget:
                            errors[key] = "cumulative600s_budget"; global_stop = True; reason = "terminal_budget_exhausted"; break
                        except Exception as error:
                            errors[key] = type(error).__name__
                            if isinstance(error, RuntimeError):
                                global_stop = True; reason = "terminal_backend_failure"
                        finally:
                            loader_receipt_available = case is not None
                            del case, physical, encoded
                            t = time.monotonic()
                            with (output / "case_records.jsonl").open("a", encoding="utf-8") as stream:
                                stream.write(json.dumps({"arm_id": arm["id"], "key": key, "step": 4800,
                                      "role": "TRAIN", "metrics": per[key], "failure": errors[key],
                                      "elapsed_from_lock_acquisition_seconds": elapsed()}, ensure_ascii=False, allow_nan=False) + "\n")
                            charge("persistence_seconds", t)
                            persist(output / "read_receipt.json", receipt); clock()
                            item["per_case_costs"][key] = {
                                **{name: arm_costs[name] - case_cost_before[name] for name in STAGE_COSTS},
                                "observed_case_wall_seconds": time.monotonic() - case_started,
                                "successful_loader_receipt_available": loader_receipt_available}
                        if global_stop:
                            break
            except DiagnosticBudget:
                global_stop = True
                if reason == "fixed_final_readout_complete": reason = "terminal_budget_exhausted"
            except Exception as error:
                item["arm_failure"] = type(error).__name__
                if isinstance(error, RuntimeError): global_stop = True; reason = "terminal_backend_failure"
            finally:
                t = time.monotonic()
                if cleanup is not None:
                    try: cleanup()
                    except Exception as error:
                        item["cleanup_failure"] = type(error).__name__; global_stop = True; reason = "terminal_cleanup_failure"
                model = None; cleanup = None
                if torch is not None:
                    try: torch.cuda.synchronize()
                    except Exception: global_stop = True; reason = "terminal_backend_failure"
                charge("model_cleanup_seconds", t)
                arm_costs["observed_arm_wall_seconds"] = time.monotonic() - arm_started
                arm_costs["unattributed_arm_wall_seconds"] = max(0., arm_costs["observed_arm_wall_seconds"]
                                                                  - sum(arm_costs[name] for name in STAGE_COSTS))
                for key in keys:
                    if errors[key] == "not_attempted":
                        errors[key] = item.get("arm_failure", reason if global_stop else "arm_endpoint_failure")
                item["aggregate"] = without_selection(train.aggregate(per, 120))
                item["CAD_three_load_aggregate"] = CAD_aggregate(per, keys)
                item["status"] = "completed"
                item["all120_metrics_available"] = item["aggregate"]["all_cases_complete"]
                persist(output / arm["id"] / "summary.json", item)
                clock()
        completed = sum(item["aggregate"]["metrics_available_cases"] for item in results.values())
        all_complete = len(results) == 5 and all(item["all120_metrics_available"] for item in results.values())
        if not all_complete and reason == "fixed_final_readout_complete": reason = "terminal_incomplete_or_nonfinite_readout"
        summary = {"status": "completed", "kind": KIND, "stage": "TRAIN_fit_diagnostic", "config": {"contract": contract},
                   "arms": results, "case_keys": keys, "fixed_arm_denominator": 5,
                   "fixed_case_denominator_each_arm": 120, "fixed_prediction_denominator": 600,
                   "completed_predictions": completed, "stop_reason": reason, "read_receipt": receipt,
                   "costs": costs, "seed": 2026, "read_scope": "original TRAIN120 cached targets only",
                   "metrics_units": {"uXYZ": "mm", "signed_stress": "MPa", "relative_metrics": "dimensionless"},
                   "completion_gate": {"all600_metrics_complete": all_complete, "TRAIN_only": True,
                        "model_selection_performed": False, "final_solver_selection": False},
                   "limitations": ["Final4800 endpoint only; not a full TRAIN learning curve",
                        "No DEV/generalization conclusion or architecture capacity upper bound",
                        "Families have different capacity; fixed recipe fit comparison only",
                        "Original training costs and new diagnostic costs remain separate",
                        "Serialized checkpoint optimizer payload was loaded on CPU but never restored or used"],
                   "coordinator_wait_proof": {"before_lock": before, "after_lock": after}}
        t = time.monotonic(); brief(summary, output / "BRIEF.md"); costs["persistence_seconds"] += time.monotonic() - t
        costs["elapsed_from_lock_acquisition_seconds"] = elapsed()
        costs["elapsed_measurement_endpoint"] = "before_terminal_summary_persistence; closed clock includes later metadata persistence"
        costs["unattributed_wall_seconds_before_precommit"] = max(0., elapsed() - sum(costs[name] for name in STAGE_COSTS))
        costs["budget_overshoot_seconds"] = max(0., elapsed() - 600)
        summary["completion_gate"]["within600s_at_precommit"] = elapsed() <= 600
        summary["completion_gate"]["passed"] = all_complete and elapsed() <= 600
        persist(output / "summary.json", summary)
        persist(output / "status.json", {"status": "completed", "kind": KIND, "phase": "terminal",
                "stop_reason": reason, "completed_predictions": completed, "TRAIN_only": True})
        clock(False)
        # The clock closes after summary/status persistence. Parent launch->exit
        # remains the authoritative external600s cap; no fresh cold-service claim.
        return summary
    except BaseException as error:
        failure = {"status": "failed", "kind": KIND, "error_type": type(error).__name__,
                   "automatic_replay": False, "elapsed_from_lock_acquisition_seconds": time.monotonic() - began,
                   "read_receipt": receipt, "costs": costs, "scientific_inputs_may_have_been_read": bool(receipt["model_initializations"])}
        if output.exists(): write(output / "failure.json", failure)
        raise
    finally:
        if cleanup is not None:
            try: cleanup()
            except Exception: pass
        held.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN)
    parser.add_argument("--output")
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.plan, args.output, args.start)
        print(json.dumps({"status": result["status"], "kind": KIND}, ensure_ascii=False), flush=True)
    except DiagnosticNotReady as error:
        print(json.dumps({"status": "not_started", "kind": KIND, "reason": str(error), "scientific_execution": False}), flush=True)
    except Exception as error:
        print(json.dumps({"status": "failed", "kind": KIND, "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
