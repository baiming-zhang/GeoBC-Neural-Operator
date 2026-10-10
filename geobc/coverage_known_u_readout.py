"""CPU-only known-U readout after the independent signed-S experiment closes.

Only immutable saved prediction U views and five previously opened U caches
are permitted. No source CSV, S labels, model, scaler or GPU is used here.
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
OWN = "geobc/coverage_known_u_readout.py"
PLAN = "notes/geobc_known_u_readout_plan.json"
KIND = "coverage_fixed4800_known_U5_CPU_readout"
DECISION = "proceed_known_U_readout_after_S_terminal"
METHODS = ("transolver", "transolver_pp", "ts3", "fno", "mlp")
CASES = ("489_12/ver", "616_205/ver", "416_282/ver", "247_50/ver", "142_454/ver")
NODES = (166790, 238303, 302979, 162864, 209499)
ZERO = ("source_U_reads", "source_S_reads", "cached_S_label_reads", "prediction_S_evaluations",
        "raw_CSV_reads", "H5_reads", "source_strain_reads",
        "weight_reads", "model_predictions", "training_updates", "scaler_fits", "GPU_calls")
ACTUAL = ("known_U_cache_read_attempts", "known_U_cache_reads", "known_U_file_hash_read_attempts", "known_U_file_hash_reads",
          "prediction_file_hash_read_attempts", "prediction_file_hash_reads", "prediction_physical4_read_attempts",
          "prediction_physical4_reads", "prediction_U_view_reads", "metric_cases")


class NotReady(RuntimeError):
    pass


class BudgetLimit(RuntimeError):
    pass


def path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", value):
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    filename = Path(value)
    return (filename if filename.is_absolute() else ROOT / filename).resolve()


def read(filename):
    return json.loads(path(filename).read_text(encoding="utf-8-sig"))


def write(filename, value):
    filename = path(filename); filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(filename.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, filename)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(filename, guard=None):
    value = hashlib.sha256()
    with path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if guard: guard()
            value.update(block)
    return value.hexdigest()


def require(condition, message, error=ValueError):
    if not condition: raise error(message)


def process_still_alive(record):
    require(os.name != "nt" and Path("/proc").is_dir(), "Readout must use the same WSL process namespace", NotReady)
    require(type(record.get("pid")) is int and type(record.get("start_ticks")) is int
            and isinstance(record.get("argv"), list), "S actual-process evidence is missing", NotReady)
    try:
        raw = (Path("/proc") / str(record["pid"]) / "stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    except OSError as error:
        raise NotReady("Cannot prove that the S process has exited") from error
    fields = raw[raw.rfind(")") + 2:].split()
    return int(fields[19]) == record["start_ticks"] and fields[0] not in ("Z", "X")


def request_contract(plan_path=PLAN):
    """Only small JSON and source hashes; no scientific array or weight reads."""
    plan = read(plan_path); science = plan["scientific"]
    require(plan["kind"] == KIND and plan.get("scientific_rules_frozen") is True
            and digest(science) == plan["scientific_digest"], "Known-U protocol differs")
    require(science["methods"] == list(METHODS) and science["case_keys"] == list(CASES)
            and science["node_counts"] == list(NODES) and science["fixed_step"] == 4800
            and science["seed"] == 2026 and science["budget_seconds"] == 120
            and science["fixed_prediction_denominator"] == 25 and science["U_is_blind"] is False,
            "Fixed known-U scope differs")
    require(plan["known_U_metadata"] == science["known_U_metadata"]
            and plan["known_U_cache_root"] == science["known_U_cache_root"]
            and plan["S_output"] == science["S_output"], "Frozen cache/campaign paths or identities changed")
    summary = read(plan["S_output"] + "/summary.json")
    clock = read(plan["S_output"] + "/clock.json")
    barrier = read(plan["S_output"] + "/prediction_barrier.json")
    require(summary.get("status") in ("completed", "failed") and summary.get("terminal") is True
            and clock.get("closed") is True and clock.get("active_phase") is None,
            "The S-only experiment has not truly closed", NotReady)
    require(not process_still_alive(summary.get("readout_process", {})), "The S process is still alive", NotReady)
    S_contract = summary["config"]["contract"]
    require(S_contract["scientific_digest"] == science["S_scientific_digest"], "Frozen S experiment identity differs")
    expected = {method + ":" + key for method in METHODS for key in CASES}
    references = barrier.get("prediction_manifests", {})
    require(barrier.get("passed") is True and barrier.get("required_entries") == 25
            and barrier.get("finite_fullN_entries") == 25 and barrier.get("contract_sha256") == digest(S_contract)
            and set(references) == expected, "The immutable fullN25 prediction barrier is incomplete", NotReady)
    predictions = {}
    for method in METHODS:
        for key, n in zip(CASES, NODES):
            token = method + ":" + key; entry = references[token]
            require(sha(entry["path"]) == entry["sha256"], "Committed prediction manifest changed")
            item = read(entry["path"]); physical = item["physical"]
            require(item.get("method") == method and item.get("key") == key
                    and item.get("checkpoint", {}).get("step") == 4800
                    and physical.get("shape") == [n, 4] and physical.get("dtype") == "float64"
                    and len(physical.get("sha256", "")) == 64, "FullN fixed4800 physical prediction identity differs")
            predictions[token] = {"manifest": entry, "physical": physical, "checkpoint": item["checkpoint"]}
    known = {}
    for key, n in zip(CASES, NODES):
        reference = plan["known_U_metadata"][key]
        require(sha(reference["path"]) == reference["sha256"], "Historical known-U metadata changed")
        item = read(reference["path"]); helper_ref = item["helper_manifest"]
        require(sha(helper_ref["path"]) == helper_ref["sha256"], "Historical U-only helper manifest changed")
        helper = read(helper_ref["path"]); identity = item["prediction"]
        require(item["case_key"] == key and item["source_U_only"] is True and item["source_strain_GT"] is False
                and identity["shape"] == [n, 3] and identity["dtype"] == "float64"
                and path(identity["path"]) == path(plan["known_U_cache_root"]) / (key.replace("/", "_") + ".npy")
                and helper["status"] == "completed" and helper["key"] == key
                and helper["alignment"] == "official CSV GRID coordinates -> H5 vertices"
                and helper["source"]["columns"] == ["ver_x_disp(mm)", "ver_y_disp(mm)", "ver_z_disp(mm)"]
                and helper["S_reads"] == 0 and helper["source"]["S_reads"] == 0 and helper["original_source_U_parse_count"] == 1
                and helper["array"]["shape"] == [n, 3] and helper["array"]["dtype"] == "float64"
                and helper["file_identity"]["sha256"] == identity["file_sha256"]
                and helper["array"]["sha256"] == identity["values_sha256"], "Known-U original-row/unit/source proof differs")
        known[key] = {"metadata": reference, "helper_manifest": helper_ref, "array": identity,
                      "historical_source_U_parse_count": 1, "original_row_order": helper["alignment"]}
    return {"kind": KIND, "scientific_digest": plan["scientific_digest"], "source_sha256": {OWN: sha(OWN)},
            "S_contract_sha256": digest(S_contract), "S_summary": {"path": plan["S_output"] + "/summary.json", "sha256": sha(plan["S_output"] + "/summary.json")},
            "S_barrier": {"path": plan["S_output"] + "/prediction_barrier.json", "sha256": sha(plan["S_output"] + "/prediction_barrier.json")},
            "methods": list(METHODS), "case_keys": list(CASES), "predictions": predictions,
            "known_U": known, "fixed_prediction_denominator": 25, "U_is_blind": False}, plan


def verify_file(identity, expected_sha, guard, receipt, counter):
    filename = path(identity["path"]); stat = filename.stat()
    require(stat.st_size == identity["bytes"] and stat.st_mtime_ns == identity["mtime_ns"], "Saved scientific file stat changed")
    receipt[counter[:-1] + "_attempts"] += 1
    actual = sha(filename, guard); receipt[counter] += 1
    require(actual == expected_sha, "Saved scientific file byte identity changed")


def aggregate(per_case):
    result = {"fixed_denominator": 5, "available_cases": sum(value is not None for value in per_case.values()), "per_case": per_case}
    for metric in ("u_relative_l2", "u_vector_error_RMS_mm", "u_max_vector_error_mm"):
        values = [None if per_case[key] is None else per_case[key][metric] for key in CASES]
        complete = all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
        result["case_mean_" + metric] = sum(values) / 5 if complete else None
        result["worst_case_" + metric] = max(values) if complete else None
    return result


def run(plan_path=PLAN, output=None, start=False):
    plan = read(plan_path)
    if not start: return {"status": "source_only_ready", "kind": KIND, "actual_execution": False}
    require(digest(plan["scientific"]) == plan["scientific_digest"], "Known-U science changed")
    marker = read(plan["root_review"])
    require(marker.get("passed") is True and marker.get("known_U_readout_released") is True
            and marker.get("decision") == DECISION and marker.get("scientific_digest") == plan["scientific_digest"]
            and marker.get("source_sha256") == sha(OWN), "Authentic root CPU release is required")
    began = time.perf_counter(); contract, plan = request_contract(plan_path)
    output = path(output or plan["output"])
    require(output == path(plan["output"]) and ROOT in output.parents, "Only the independent project output is allowed")
    if (output / "summary.json").exists():
        old = read(output / "summary.json"); require(old["config"]["contract"] == contract, "Completed known-U identity changed"); return old
    output.mkdir(parents=True, exist_ok=True)
    try:
        with (output / "once_claim.json").open("x", encoding="utf-8") as stream:
            json.dump({"contract_sha256": digest(contract), "started_at_utc": datetime.now(timezone.utc).isoformat(),
                       "automatic_replay": False}, stream)
    except FileExistsError as error:
        raise NotReady("An existing attempted readout cannot automatically reset or replay") from error
    receipt = {name: 0 for name in ZERO + ACTUAL}; truths = {}; truth_errors = {}
    results = {method: {"per_case": {key: None for key in CASES}, "failures": {key: "not_attempted" for key in CASES}} for method in METHODS}
    costs = {"metadata_gate_seconds": time.perf_counter() - began, "known_U_read_identity_seconds": 0.,
             "prediction_U_read_identity_seconds": 0., "metrics_seconds": 0., "persistence_seconds": 0.}
    stop_reason = "fixed_known_U_readout_complete"
    def guard():
        if time.perf_counter() - began >= 120: raise BudgetLimit("Cumulative CPU readout wall reached120s")
    def persist():
        t = time.perf_counter()
        write(output / "read_receipt.json", receipt)
        write(output / "clock.json", {"closed": False, "elapsed_seconds": time.perf_counter() - began,
                                      "budget_seconds": 120, "automatic_replay": False})
        costs["persistence_seconds"] += time.perf_counter() - t
    write(output / "config.json", {"contract": contract, "known_U_not_blind": True})
    persist()
    try:
        import numpy as np
        for key, n in zip(CASES, NODES):
            t = time.perf_counter()
            try:
                guard(); identity = contract["known_U"][key]["array"]
                verify_file(identity, identity["file_sha256"], guard, receipt, "known_U_file_hash_reads")
                receipt["known_U_cache_read_attempts"] += 1
                value = np.load(path(identity["path"]), allow_pickle=False)
                receipt["known_U_cache_reads"] += 1
                require(value.shape == (n, 3) and value.dtype == np.float64 and bool(np.isfinite(value).all()), "Known-U finite fullN schema differs")
                require(hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == identity["values_sha256"], "Known-U values changed")
                truths[key] = value; guard()
            except BudgetLimit:
                truth_errors[key] = "budget120s"; stop_reason = "terminal_budget_exhausted"; break
            except Exception as error:
                truth_errors[key] = type(error).__name__
            finally:
                costs["known_U_read_identity_seconds"] += time.perf_counter() - t; persist()
        for method in METHODS:
            for key, n in zip(CASES, NODES):
                target = results[method]; prediction = None
                try:
                    guard()
                    if key not in truths:
                        target["failures"][key] = truth_errors.get(key, "known_U_unavailable"); continue
                    identity = contract["predictions"][method + ":" + key]["physical"]
                    t = time.perf_counter()
                    verify_file(identity, identity["sha256"], guard, receipt, "prediction_file_hash_reads")
                    receipt["prediction_physical4_read_attempts"] += 1
                    saved = np.load(path(identity["path"]), mmap_mode="r", allow_pickle=False)
                    receipt["prediction_physical4_reads"] += 1
                    require(saved.shape == (n, 4) and saved.dtype == np.float64, "Saved fullN physical4 schema differs")
                    prediction = np.array(saved[:, :3], dtype=np.float64, copy=True)
                    receipt["prediction_U_view_reads"] += 1; del saved
                    require(bool(np.isfinite(prediction).all()), "Nonfinite candidate U")
                    costs["prediction_U_read_identity_seconds"] += time.perf_counter() - t; guard()
                    t = time.perf_counter(); truth = truths[key]; error = prediction - truth
                    reference2 = float(np.square(truth).sum()); squared_error = float(np.square(error).sum())
                    require(math.isfinite(reference2) and math.isfinite(squared_error), "Nonfinite U metric accumulation")
                    target["per_case"][key] = {"nodes": n, "u_relative_l2": math.sqrt(squared_error / reference2) if reference2 > 0 else None,
                            "u_vector_error_RMS_mm": math.sqrt(squared_error / n),
                            "u_max_vector_error_mm": float(np.linalg.norm(error, axis=1).max()),
                            "u_zero_reference": reference2 <= 0, "all_original_nodes": True, "U_is_blind": False}
                    require(math.isfinite(target["per_case"][key]["u_max_vector_error_mm"]), "Nonfinite maximum U error")
                    receipt["metric_cases"] += 1
                    target["failures"][key] = None if reference2 > 0 else "zero_U_reference_relative_L2_undefined"
                    costs["metrics_seconds"] += time.perf_counter() - t
                except BudgetLimit:
                    target["failures"][key] = "budget120s"; stop_reason = "terminal_budget_exhausted"
                except Exception as error:
                    target["per_case"][key] = None; target["failures"][key] = type(error).__name__
                finally:
                    del prediction; persist()
            target["aggregate"] = aggregate(target["per_case"])
        completed = sum(value is not None for target in results.values() for value in target["per_case"].values())
        relative_complete = all(target["per_case"][key] is not None
                and isinstance(target["per_case"][key]["u_relative_l2"], (int, float))
                and math.isfinite(target["per_case"][key]["u_relative_l2"]) for target in results.values() for key in CASES)
        if completed != 25 and stop_reason == "fixed_known_U_readout_complete": stop_reason = "terminal_incomplete_or_nonfinite"
        summary = {"status": "completed", "terminal": True, "kind": KIND, "config": {"contract": contract},
                   "methods": list(METHODS), "case_keys": list(CASES), "fixed_prediction_denominator": 25,
                   "completed_metric_cases": completed, "results": results, "stop_reason": stop_reason,
                   "known_U_failures": truth_errors, "read_receipt": receipt, "costs": costs,
                   "units": {"U": "mm", "relative_l2": "dimensionless"}, "U_is_blind": False,
                   "S_labels_read_by_this_stage": False, "final_solver_selection": False,
                   "meaning": "Known-U generalization interpretation paired with the separately field-blind signed-S stage; no new source read or prediction"}
        costs["elapsed_seconds_before_terminal_persist"] = time.perf_counter() - began
        costs["unattributed_wall_seconds_before_terminal_persist"] = max(0., time.perf_counter() - began - sum(value for name, value in costs.items() if name.endswith("_seconds") and name != "elapsed_seconds_before_terminal_persist"))
        summary["completion_gate"] = {"all25_metrics_available": completed == 25, "all25_relative_L2_defined": relative_complete,
                                     "known_U_not_blind": True, "within120s_precommit": time.perf_counter() - began <= 120,
                                     "passed": relative_complete and time.perf_counter() - began <= 120}
        write(output / "summary.json", summary)
        write(output / "clock.json", {"closed": True, "elapsed_seconds": time.perf_counter() - began,
                                     "budget_seconds": 120, "automatic_replay": False, "cost_exact_at_process_exit": False})
        return summary
    except BaseException as error:
        write(output / "failure.json", {"kind": KIND, "error_type": type(error).__name__, "read_receipt": receipt,
                                        "elapsed_seconds": time.perf_counter() - began, "automatic_replay": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN); parser.add_argument("--output"); parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.plan, args.output, args.start)
        print(json.dumps({"status": result["status"], "kind": KIND}), flush=True)
    except NotReady as error:
        print(json.dumps({"status": "not_started", "kind": KIND, "reason": str(error)}), flush=True)
    except Exception as error:
        print(json.dumps({"status": "failed", "kind": KIND, "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
