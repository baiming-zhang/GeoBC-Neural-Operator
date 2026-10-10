"""CPU-only scalar export of an authentic, exited available-CAD20 pilot.

The default returns immediately without reading a plan or creating any file.
The exporter never imports experiment modules or opens arrays, PT or raw/GT
files. Matplotlib is imported only by an explicitly released actual export.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time

PLAN = "notes/geobc_available_cad_pilot_plan.json"
OWN = "tools/export_available_cad_pilot.py"
REVIEW = "notes/available_cad_pilot_exporter_root_review.json"
SCI = "c7c994a44d462b94426ecfa034f2d535bbf7751331c1e16ab8925485e5a67770"
KIND = "GeoBC_available_CAD20_candidate9_scalar_export"
EVAL_KIND = "GeoBC_available_CAD20_candidate9_independent_readout"
GUARD_KIND = "GeoBC_available20_one_shot_guard"
GUARD_SOURCE = "tools/run_geobc_available_pilot_guard.py"
LOADS = ("ver", "hor", "tor")
CANDIDATES = (
    ("transolver_cad40_fresh4800", "transolver", 4800, "Old TS"),
    ("transolver_pp_cad40_fresh4800", "transolver_pp", 4800, "Old PP"),
    ("ts3_cad40_fresh4800", "ts3", 4800, "Old TS3"),
    ("fno_cad40_fresh4800", "fno", 4800, "Old FNO"),
    ("mlp_cad40_fresh4800", "mlp", 4800, "Old MLP"),
    ("PP_regime_step4800", "transolver_pp", 4800, "PP 4800"),
    ("PP_regime_step24000", "transolver_pp", 24000, "PP 24000"),
    ("FNO_latent_query_grid25_step4800", "FNO_latent_query_grid25", 4800, "FNO grid25"),
    ("FNO_latent_query_native25_step4800", "FNO_latent_query_native25", 4800, "FNO native25"),
)
FIELDS = (("u_relative_l2", "u"), ("stress_relative_l2", "stress"),
          ("stress_hotspot_relative_l2", "stress_hotspot"))
FORBIDDEN_COUNTERS = ("optimizer_restores", "training_updates", "scaler_fits", "old_DEV_reads", "old_final_reads",
    "original_new_DEV20_reads", "dia_reads", "strain_reads", "H5_nodal_reads")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def root():
    return Path(__file__).resolve().parents[1]


def path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", value):
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    filename = Path(value)
    filename = (filename if filename.is_absolute() else root() / filename).resolve()
    filename.relative_to(root()); return filename


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def finite(value, nonnegative=False):
    return type(value) in (float, int) and math.isfinite(value) and (not nonnegative or value >= 0)


def _json(filename, index):
    filename = path(filename)
    require(filename.suffix.lower() == ".json", "Only small JSON metadata can be opened")
    require(filename.stat().st_size <= 64 * 1024**2, "Metadata JSON exceeds the declared size bound")
    value = filename.read_bytes()
    def reject_constant(token):
        raise ValueError("Nonfinite JSON constant: " + token)
    result = json.loads(value.decode("utf-8-sig"), parse_constant=reject_constant)
    index[str(filename.relative_to(root()))] = {"path": str(filename), "bytes": len(value),
                                              "sha256": hashlib.sha256(value).hexdigest()}
    require(isinstance(result, dict), "Metadata JSON must be an object")
    return result


def _source_hash(name):
    filename = path(name)
    require(filename.suffix.lower() in (".py", ".ps1", ".json"), "Source identity must be a pinned source/text file")
    require(filename.stat().st_size <= 64 * 1024**2, "Pinned text source exceeds size bound")
    return hashlib.sha256(filename.read_bytes()).hexdigest()


def _process(record):
    require(isinstance(record, dict) and type(record.get("pid")) is int and record["pid"] > 0
        and type(record.get("start_ticks")) is int and record["start_ticks"] > 0
        and isinstance(record.get("argv"), list) and record["argv"]
        and all(isinstance(token, str) and token for token in record["argv"]), "Exact PID/startticks/argv identity missing")
    require(os.name != "nt" and Path("/proc").is_dir(), "Use the actual experiment WSL process namespace")
    folder = Path("/proc") / str(record["pid"])
    try:
        stat = (folder / "stat").read_text(encoding="utf-8")
        tail = stat[stat.rfind(")") + 2:].split()
        if tail[0] in ("Z", "X"):
            return True
        ticks = int(tail[19])
        argv = [v.decode("utf-8") for v in (folder / "cmdline").read_bytes().split(b"\0") if v]
    except (FileNotFoundError, ProcessLookupError):
        return True
    except (OSError, ValueError, UnicodeError, IndexError) as error:
        raise RuntimeError("Actual process exit cannot be established") from error
    if ticks != record["start_ticks"]:
        return True
    require(argv == record["argv"], "A live exact PID/startticks changed argv; exit is not established")
    return False


def _module_argv(record, module, plan_path):
    argv = record["argv"]
    require("-m" in argv and argv[argv.index("-m") + 1] == module and "--start" in argv
        and "--plan" in argv and path(argv[argv.index("--plan") + 1]) == path(plan_path), "Recorded child argv differs")


def _stage(receipt, native, execution, purpose, plan_path, source_pins):
    require(receipt.get("schema") == 1 and receipt.get("kind") == GUARD_KIND
        and receipt.get("purpose") == purpose and receipt.get("scientific_digest") == SCI
        and receipt.get("status") == "terminal" and receipt.get("automatic_replay") is False
        and receipt.get("actual_child_exited") is True, "Authentic external stage terminal receipt missing")
    child = receipt["child_process"]; guard = receipt["guard"]
    require(child == native.get("run_process") == execution.get("run_process")
        and native.get("closed") is True and native.get("active") is False and native.get("attempts") == 1,
        "Actual native terminal/execution/child identity differs")
    require(guard.get("schema") == 1 and guard.get("kind") == GUARD_KIND and guard.get("purpose") == purpose
        and guard.get("scientific_digest") == SCI and guard.get("guard_source") == GUARD_SOURCE
        and guard.get("guard_source_sha256") == source_pins[GUARD_SOURCE]
        and guard.get("budget_seconds") == 3600 and guard.get("automatic_replay") is False
        and execution.get("guard") == guard, "Actual reviewed guard/science binding differs")
    require(child.get("ppid") == guard["guard_process"]["pid"]
        and _process(child) and _process(guard["guard_process"]), "Exact child or campaign guard is still active")
    _module_argv(child, "geobc.available_cad_pilot_" + ("data" if purpose == "preparation" else "eval"), plan_path)
    guard_argv = guard["guard_process"]["argv"]
    require(any(token.endswith(".py") and path(token) == path(GUARD_SOURCE) for token in guard_argv)
        or any(token == "tools.run_geobc_available_pilot_guard" and i > 0 and guard_argv[i-1] == "-m"
               for i, token in enumerate(guard_argv)), "Actual campaign guard argv differs")
    fields = ("entry_epoch", "entry_monotonic", "deadline_epoch", "deadline_monotonic",
              "campaign_entry_epoch", "campaign_entry_monotonic", "campaign_deadline_epoch", "campaign_deadline_monotonic")
    require(all(finite(guard.get(name)) for name in fields)
        and abs(guard["deadline_epoch"] - guard["entry_epoch"] - 3600) < .001
        and abs(guard["deadline_monotonic"] - guard["entry_monotonic"] - 3600) < .001
        and abs(guard["campaign_deadline_epoch"] - guard["campaign_entry_epoch"] - 7200) < .001
        and abs(guard["campaign_deadline_monotonic"] - guard["campaign_entry_monotonic"] - 7200) < .001,
        "Recorded stage/campaign guard clock schema differs")
    elapsed = receipt.get("elapsed_seconds"); inner = native.get("cumulative_wall_seconds")
    exact = finite(elapsed, True) and finite(inner, True) and elapsed + 1e-6 >= inner
    normal = (receipt.get("exit_code") == 0 and receipt.get("external_timeout") is False
        and receipt.get("interrupted") is False and receipt.get("within_external_budget") is True
        and "termination_error_class" not in receipt and exact and elapsed <= 3600)
    return {"authoritative_seconds": elapsed if finite(elapsed, True) else None,
        "native_seconds": inner if finite(inner, True) else None, "cost_exact": exact,
        "normal_within_budget": normal, "exit_code": receipt.get("exit_code"),
        "external_timeout": receipt.get("external_timeout"), "within_external_budget": receipt.get("within_external_budget"),
        "phase_seconds": native.get("phase_seconds", {}), "guard": guard, "child_process": child}


def _gate(plan_path, index):
    require(path(plan_path) == path(PLAN), "Only the formal frozen pilot plan may be exported")
    plan = _json(plan_path, index)
    require(plan.get("scientific_rules_frozen") is True and digest(plan["scientific"]) == plan["scientific_digest"] == SCI,
            "Frozen pilot science differs")
    science = plan["scientific"]; selection = science["CAD_selection"]
    cads = selection["selected_CADs"]; keys = selection["case_keys"]
    require(len(cads) == len(set(cads)) == 20 and cads == sorted(selection["eligible_CADs"])[:20]
        and keys == [cad + "/" + load for cad in cads for load in LOADS]
        and not set(cads).intersection(selection["excluded_old_TRAIN40"] + selection["excluded_original_DEV20"]),
        "Frozen availability-conditioned CAD20/known3 selection differs")
    candidates = science["candidate_inventory"]["candidates"]
    require([(c["candidate_id"], c["method"], c["step"]) for c in candidates] == [c[:3] for c in CANDIDATES]
        and all(c["TRAIN_CADs"] == 40 for c in candidates), "Fixed nine old40 candidates differ")
    source_pins = dict(plan["parent_source_sha256"])
    require(source_pins == science["immutable_parent"]["original_twenty_source_pins"] and len(source_pins) == 20,
            "Original twenty source pins differ")
    for mapping in (plan["additional_source_sha256"], plan["new_source_sha256"]):
        for name, value in mapping.items():
            require(name not in source_pins or source_pins[name] == value, "Conflicting experiment source pins")
            source_pins[name] = value
    for name, expected in source_pins.items():
        require(_source_hash(name) == expected, "Frozen experiment source changed: " + name)
    exporter_sha = _source_hash(OWN); review = _json(REVIEW, index)
    require(review.get("passed") is True and review.get("actual_export_released") is True
        and review.get("scientific_digest") == SCI and review.get("source_sha256") == exporter_sha,
        "Separate authentic exporter source release is required")
    pilot_review = _json(plan["root_review"], index)
    require(pilot_review.get("passed") is True and pilot_review.get("actual_execution_released") is True
        and pilot_review.get("scientific_digest") == SCI and pilot_review.get("source_pins") == source_pins,
        "Authentic experiment release/source review differs")
    outputs = science["outputs"]; base = path(outputs["readout"]); prepared = path(outputs["input_cache"])
    summary = _json(base / "summary.json", index); status = _json(base / "status.json", index)
    native = _json(base / "clock.json", index); execution = _json(base / "execution.json", index)
    evaluation_external = _json(base / "evaluation_external_terminal.json", index)
    input_manifest = _json(prepared / "manifest.json", index); preparation_native = _json(prepared / "clock.json", index)
    preparation_execution = _json(prepared / "execution.json", index)
    preparation_external = _json(prepared / "preparation_external_terminal.json", index)
    campaign = _json(base / "campaign_external_terminal.json", index)
    require(summary.get("kind") == EVAL_KIND and summary.get("status") in ("completed", "completed_negative")
        and summary.get("terminal") is True and status.get("status") == summary["status"]
        and status.get("phase") == "terminal" and summary.get("run_process") == status.get("run_process")
        == native.get("run_process") and summary.get("scientific_digest") == SCI
        and summary.get("source_sha256") == execution.get("source_sha256") == source_pins
        and execution.get("scientific_digest") == SCI and execution.get("contract_sha256") == summary.get("contract_sha256")
        and summary.get("candidate_denominator") == 9 and summary.get("case_denominator") == 60
        and summary.get("CAD_denominator") == 20, "Saved pilot summary/status/source identity differs")
    contract = input_manifest["contract"]
    require(input_manifest.get("terminal") is True and input_manifest.get("status") in ("completed", "failed")
        and type(input_manifest.get("feasibility_pass")) is bool and contract.get("scientific_digest") == SCI
        and contract.get("source_sha256") == source_pins and digest(contract) == summary["contract_sha256"]
        == preparation_execution.get("input_contract_sha256") and input_manifest.get("case_keys") == keys
        and input_manifest.get("run_process") == preparation_native.get("run_process"), "Saved target-free preparation binding differs")
    require(set(input_manifest.get("records", {})) <= set(keys)
        and set(input_manifest.get("CADs", {})) <= set(cads), "Saved input metadata contains substituted CADs/cases")
    if input_manifest["feasibility_pass"]:
        require(set(input_manifest["records"]) == set(keys) and set(input_manifest["CADs"]) == set(cads)
            and set(input_manifest.get("GT_header_metadata", {})) == set(cads)
            and all(v.get("feasibility_pass") is True and v.get("status") == "completed"
                and "target_path" not in v and "physical_path" not in v for v in input_manifest["records"].values()),
            "Positive input metadata is not complete fixed60/20 target-free input")
    stages = {"preparation": _stage(preparation_external, preparation_native, preparation_execution, "preparation", plan_path, source_pins),
              "evaluation": _stage(evaluation_external, native, execution, "evaluation", plan_path, source_pins)}
    require(campaign.get("schema") == 1 and campaign.get("kind") == GUARD_KIND and campaign.get("status") == "terminal"
        and campaign.get("closed") is True and campaign.get("active") is False and campaign.get("actual_execution") is True
        and campaign.get("attempts") == 1 and campaign.get("automatic_replay") is False
        and campaign.get("scientific_digest") == SCI and campaign.get("source_sha256") == source_pins
        and campaign.get("guard_process") == evaluation_external["guard"]["guard_process"]
        == preparation_external["guard"]["guard_process"] and _process(campaign["guard_process"])
        and campaign.get("stages") == {"preparation": preparation_external, "evaluation": evaluation_external},
        "Authentic complete campaign receipts/exact exited guard differ")
    require(campaign.get("entry_epoch") == evaluation_external["guard"]["campaign_entry_epoch"]
        == preparation_external["guard"]["campaign_entry_epoch"]
        and campaign.get("entry_monotonic") == evaluation_external["guard"]["campaign_entry_monotonic"]
        == preparation_external["guard"]["campaign_entry_monotonic"]
        and campaign.get("deadline_epoch") == evaluation_external["guard"]["campaign_deadline_epoch"]
        == preparation_external["guard"]["campaign_deadline_epoch"]
        and campaign.get("deadline_monotonic") == evaluation_external["guard"]["campaign_deadline_monotonic"]
        == preparation_external["guard"]["campaign_deadline_monotonic"] and campaign.get("budget_seconds") == 7200,
        "External campaign/stage clock identity differs")
    elapsed = campaign.get("elapsed_seconds")
    exact = finite(elapsed, True) and all(s["cost_exact"] for s in stages.values()) and elapsed + 1e-6 >= sum(
        s["authoritative_seconds"] for s in stages.values())
    normal = (exact and elapsed <= 7200 and campaign.get("within_external_budget") is True
        and campaign.get("all_requested_stages_exited_normally") is True
        and all(s["normal_within_budget"] for s in stages.values()))
    costs = {"stages": stages, "campaign_authoritative_seconds": elapsed if finite(elapsed, True) else None,
        "campaign_cost_exact": exact, "campaign_normal_within_budget": normal,
        "guard_overhead_seconds": campaign.get("between_stage_and_guard_overhead_seconds"),
        "campaign_within_external_budget": campaign.get("within_external_budget")}
    if review.get("experiment_summary_sha256") is not None:
        require(review["experiment_summary_sha256"] == index[str((base / "summary.json").relative_to(root()))]["sha256"],
                "Exporter review binds a different experiment summary")
    return plan, summary, input_manifest, costs, source_pins, exporter_sha


def _close_number(stored, expected):
    return stored is None if expected is None else finite(stored) and math.isclose(stored, expected, rel_tol=1e-12, abs_tol=1e-15)


def _aggregate(saved, expected_keys, denominator):
    per = saved.get("per_case")
    require(saved.get("fixed_denominator") == denominator and isinstance(per, dict) and set(per) == set(expected_keys),
            "Saved scalar aggregate has a different fixed denominator/case set")
    result = {"fixed_denominator": denominator, "metrics_available_cases": sum(v is not None for v in per.values()), "per_case": per}
    require(saved.get("metrics_available_cases") == result["metrics_available_cases"], "Saved metric-case count differs")
    for item in per.values():
        if item is not None:
            require(isinstance(item, dict) and type(item.get("nodes")) is int and item["nodes"] > 0, "Saved original-node scalar schema differs")
            for metric, _ in FIELDS:
                require(metric in item and (item[metric] is None or finite(item[metric], True)), "Saved relative error is absent/nonfinite/negative")
            for metric, flag in (("u_relative_l2", "u_zero_reference"), ("stress_relative_l2", "stress_zero_reference"),
                                 ("stress_hotspot_relative_l2", "hotspot_zero_reference")):
                require(type(item.get(flag)) is bool and (item[metric] is None) == item[flag], "Saved zero-reference scalar flag differs")
            require(all(finite(item.get(name), True) for name in ("u_vector_error_RMS_mm", "u_max_vector_error_mm",
                "stress_RMSE_MPa", "stress_MAE_MPa", "stress_max_abs_error_MPa")), "Saved physical error scalars are absent/nonfinite")
    for metric, prefix in FIELDS:
        values = [per[key].get(metric) if per[key] else None for key in expected_keys]
        complete = len(values) == denominator and all(finite(v, True) for v in values)
        mean = sum(values) / denominator if complete else None; worst = max(values) if complete else None
        mean_name = "case_mean_" + prefix + "_relative_l2"; worst_name = "worst_case_" + prefix + "_relative_l2"
        require(_close_number(saved.get(mean_name), mean) and _close_number(saved.get(worst_name), worst),
                "Saved full-denominator scalar mean/worst differs")
        result[mean_name] = mean; result[worst_name] = worst
    u = result["case_mean_u_relative_l2"]; s = result["case_mean_stress_relative_l2"]
    score = .5 * (u + s) if u is not None and s is not None else None
    require(_close_number(saved.get("selection_score"), score), "Saved primary-score arithmetic differs")
    complete = all(result["case_mean_" + prefix + "_relative_l2"] is not None for _, prefix in FIELDS)
    require(_close_number(saved.get("primary_score"), score if complete else None), "Saved complete-U/S/hot formal score differs")
    result.update(selection_score=score, complete_U_S_hot=complete,
                  descriptive_score=score if complete else None, primary_score=None, ranking_eligible=False)
    return result


def _barrier_and_gt(plan, summary, inputs, index):
    keys = plan["scientific"]["CAD_selection"]["case_keys"]
    cads = plan["scientific"]["CAD_selection"]["selected_CADs"]
    required = {cid + ":" + key for cid, _, _, _ in CANDIDATES for key in keys}
    inventory = summary.get("prediction_inventory", {})
    require(isinstance(inventory, dict) and set(inventory) <= required, "Saved prediction inventory has foreign cases/candidates")
    base = path(plan["scientific"]["outputs"]["readout"])
    for token, identity in inventory.items():
        cid, key = token.split(":", 1); cad, load = key.split("/")
        require(path(identity["path"]) == base / "predictions" / cid / cad / load / "manifest.json"
            and re.fullmatch(r"[0-9a-f]{64}", identity.get("sha256", "")), "Saved manifest metadata index path/SHA differs")
    proof = summary.get("prediction_barrier"); receipt = summary["read_receipt"]
    require(all(receipt.get(name) == 0 for name in FORBIDDEN_COUNTERS)
        and receipt.get("source_GT_parses_before_barrier") == 0, "Forbidden/source-before-barrier read receipt differs")
    for name in ("source_GT_parse_attempts", "source_GT_parses", "U_target_case_reads", "S_target_case_reads"):
        require(type(receipt.get(name)) is int and receipt[name] >= 0, "GT receipt count missing")
    require(receipt["source_GT_parses"] <= receipt["source_GT_parse_attempts"] <= 20
        and receipt["U_target_case_reads"] == receipt["S_target_case_reads"] == 3*receipt["source_GT_parses"],
        "GT once-perCAD/three-load actual counters differ")
    barrier_complete = False
    if proof is not None:
        require(proof.get("kind") == EVAL_KIND and proof.get("scientific_digest") == SCI
            and proof.get("source_sha256") == summary["source_sha256"] and proof.get("contract_sha256") == summary["contract_sha256"]
            and proof.get("entries") == inventory and proof.get("endpoints_sha256") == digest(summary["endpoints"]),
            "Saved immutable barrier binding differs")
        barrier_complete = (set(inventory) == required and proof.get("terminal_entries") == proof.get("required_entries") == 540
            and proof.get("exact_cartesian_product_verified") is True and proof.get("owned_allN_arrays_verified") is True
            and type(proof.get("finite_predictions")) is int and type(proof.get("real_failures")) is int
            and proof["finite_predictions"] + proof["real_failures"] == 540)
    if receipt["source_GT_parse_attempts"]:
        require(barrier_complete and proof.get("all_fixed20_inputs_complete") is True,
                "GT attempt lacks a genuine complete saved prediction barrier")
    shared = None; identity = summary.get("shared_GT_manifest")
    if identity is not None:
        require(path(identity["path"]) == base / "shared_GT" / "manifest.json", "Saved GT manifest index path differs")
        shared = _json(identity["path"], index)
        require(index[str(path(identity["path"]).relative_to(root()))]["sha256"] == identity["sha256"]
            and shared.get("contract_sha256") == summary["contract_sha256"] and shared.get("barrier_sha256") == digest(proof),
            "Shared GT small-metadata identity differs")
    gt_complete = (shared is not None and shared.get("status") == "completed" and shared.get("terminal") is True
        and set(shared.get("records", {})) == set(keys) and set(shared.get("source_reads", {})) == set(cads)
        and receipt["source_GT_parse_attempts"] == receipt["source_GT_parses"] == 20
        and receipt["U_target_case_reads"] == receipt["S_target_case_reads"] == 60
        and set(receipt.get("GT_source_attempts", {})) == set(cads)
        and all(v.get("status") == "committed_owned_cache" for v in receipt["GT_source_attempts"].values()))
    if gt_complete:
        expected_names = [load + suffix for load in LOADS for suffix in (
            "_x_disp(mm)", "_y_disp(mm)", "_z_disp(mm)", "_stress(MPa)")]
        require(all(shared["source_reads"][cad].get("parse_once") is True
            and shared["source_reads"][cad].get("columns") == expected_names
            and shared["source_reads"][cad].get("source") == inputs["contract"]["raw_sources"][cad]["Field"] for cad in cads)
            and all(v.get("all_original_nodes") is True and v.get("signed_stress_unchanged") is True
                    and v.get("units") == ["mm", "mm", "mm", "MPa"] for v in shared["records"].values()),
            "Shared once12 original-column/source/sign metadata differs")
    if inputs["feasibility_pass"] is False:
        require(receipt["source_GT_parse_attempts"] == receipt["source_GT_parses"] == 0 and shared is None,
                "Input-negative must have zero numeric GT")
    return {"prediction_terminal_entries": len(inventory), "required_prediction_entries": 540,
        "barrier_complete": barrier_complete, "input_negative": inputs["feasibility_pass"] is False,
        "input_negative_reason": inputs.get("reason"), "GT_complete_once20": gt_complete,
        "GT_expected": {"source_parses": 20, "U_case_reads": 60, "S_case_reads": 60},
        "GT_actual": {name: receipt[name] for name in ("source_GT_parse_attempts", "source_GT_parses", "U_target_case_reads", "S_target_case_reads")},
        "forbidden_reads": {name: receipt[name] for name in FORBIDDEN_COUNTERS},
        "shared_GT_manifest_identity": identity, "proof": proof}


def _paired(rows, cads):
    pairs = {}; ids = list(rows)
    for i, left in enumerate(ids):
        for right in ids[i+1:]:
            per = {}
            for cad in cads:
                item = {}
                for metric, _ in FIELDS:
                    deltas = []
                    for load in LOADS:
                        a = rows[left]["full60"]["per_case"][cad + "/" + load]
                        b = rows[right]["full60"]["per_case"][cad + "/" + load]
                        av = a.get(metric) if a else None; bv = b.get(metric) if b else None
                        deltas.append(bv-av if finite(av) and finite(bv) else None)
                    item[metric + "_right_minus_left_mean3"] = sum(deltas)/3 if all(v is not None for v in deltas) else None
                per[cad] = item
            names = list(next(iter(per.values())))
            pairs[left + "__vs__" + right] = {"left": left, "right": right, "fixed_CAD_denominator": 20, "per_CAD": per,
                "mean20": {name: sum(item[name] for item in per.values())/20
                    if all(item[name] is not None for item in per.values()) else None for name in names}}
    return pairs


def _metrics(plan, summary, inputs, costs, receipt):
    selection = plan["scientific"]["CAD_selection"]; keys = selection["case_keys"]; cads = selection["selected_CADs"]
    require(set(summary.get("results", {})) == {c[0] for c in CANDIDATES}, "Saved scalar rows must retain all nine candidates")
    qualification = (summary["status"] == "completed" and summary.get("terminal_negative_reason") is None
        and costs["campaign_normal_within_budget"] and receipt["barrier_complete"]
        and receipt["GT_complete_once20"] and not receipt["input_negative"])
    rows = {}
    for candidate, (_, _, _, label) in zip(plan["scientific"]["candidate_inventory"]["candidates"], CANDIDATES):
        cid = candidate["candidate_id"]; saved = summary["results"][cid]
        overall = _aggregate(saved["full60"], keys, 60)
        per_load = {load: _aggregate(saved["per_load20"][load], [cad + "/" + load for cad in cads], 20) for load in LOADS}
        for load in LOADS:
            require(per_load[load]["per_case"] == {cad + "/" + load: overall["per_case"][cad + "/" + load] for cad in cads},
                    "Saved per-load20 scalar cases disagree with full60")
        for aggregate in (overall, *per_load.values()):
            aggregate["ranking_eligible"] = qualification and aggregate["complete_U_S_hot"]
            aggregate["primary_score"] = aggregate["descriptive_score"] if aggregate["ranking_eligible"] else None
        endpoint = summary.get("endpoints", {}).get(cid)
        if endpoint:
            require(endpoint.get("candidate") == candidate, "Saved endpoint belongs to a different fixed candidate")
            if endpoint.get("status") == "valid":
                require(endpoint.get("payload_step") == candidate["step"]
                    and path(endpoint["checkpoint"]["path"]) == path(candidate["checkpoint_path"])
                    and re.fullmatch(r"[0-9a-f]{64}", endpoint["checkpoint"].get("sha256", ""))
                    and endpoint.get("optimizer_restored") is False and endpoint.get("model_eval") is True,
                    "Saved fixed endpoint payload identity differs")
        metadata = endpoint.get("metadata", {}) if endpoint else {}
        exposures = endpoint.get("case_exposure_counts") if endpoint else None
        parameter_counts = metadata.get("parameter_counts")
        count = parameter_counts.get("total") if isinstance(parameter_counts, dict) else None
        require(count is None or type(count) is int and count >= 0, "Inherited parameter count differs")
        require(exposures is None or isinstance(exposures, dict) and all(type(v) is int and v >= 0 for v in exposures.values()),
                "Inherited endpoint exposure metadata differs")
        success_seconds = saved.get("actual_inference_seconds")
        failed_seconds = saved.get("failed_inference_and_persistence_seconds")
        train_seconds = metadata.get("inherited_training_seconds")
        require(all(v is None or finite(v, True) for v in (success_seconds, failed_seconds, train_seconds)), "Saved actual cost scalar differs")
        failures = saved.get("true_failures", {})
        require(isinstance(failures, dict) and set(failures) <= set(keys)
            and all(isinstance(v, dict) and v.get("real_terminal_outcome") is True and bool(v.get("reason")) for v in failures.values()),
            "True failure scalar inventory differs")
        present = sum(cid + ":" + key in summary.get("prediction_inventory", {}) for key in keys)
        rows[cid] = {"candidate_id": cid, "label": label, "method": candidate["method"], "fixed_step": candidate["step"],
            "TRAIN_CADs": 40, "planned_complete_TRAIN120_passes": candidate["complete_TRAIN120_passes"],
            "full60": overall, "per_load20": per_load, "true_failures": failures, "true_failure_cases": len(failures),
            "missing_terminal_entries": 60-present, "endpoint_status": endpoint.get("status") if endpoint else "not_bound",
            "actual_inference_seconds": success_seconds, "failed_inference_and_persistence_seconds": failed_seconds,
            "inherited_training_seconds": train_seconds, "inherited_costs": metadata.get("inherited_costs"),
            "parameter_counts": parameter_counts, "total_parameters": count, "case_exposure_counts": exposures,
            "exposure_min": min(exposures.values()) if exposures else None, "exposure_max": max(exposures.values()) if exposures else None,
            "exposure_total": sum(exposures.values()) if exposures else None,
            "training_qualification_context_only": metadata.get("training_qualification"),
            "checkpoint_metadata_index": {"checkpoint": endpoint.get("checkpoint"), "endpoint_metadata": endpoint.get("endpoint_metadata"),
                "payload_step": endpoint.get("payload_step"), "payload_config_sha256": endpoint.get("payload_config_sha256"),
                "training_source_sha256": endpoint.get("training_source_sha256"), "training_summary": metadata.get("training_summary")}
                if endpoint else None, "nonqualified_descriptive_source_score": saved["full60"].get("selection_score")}
    pairs = _paired(rows, cads)
    require(summary.get("paired_CAD20_differences") == pairs, "Saved paired20 scalar differences disagree with complete case scalars")
    return {"kind": KIND, "scientific_digest": SCI, "scientific_qualification": qualification,
        "candidate_denominator": 9, "case_denominator": 60, "CAD_denominator": 20,
        "selected_CADs": cads, "case_keys": keys, "rows": rows, "paired_CAD20_differences": pairs,
        "costs": costs, "receipt": receipt, "source_status": summary["status"],
        "terminal_negative_reason": summary.get("terminal_negative_reason"),
        "score_definition": "0.5*(full60 case-mean U vector relative L2+full60 case-mean signedS relative L2)",
        "no_subset_means": True, "nonqualified_descriptive_scores_are_not_selections": True,
        "interpretation": "Single-seed availability-conditioned official-TRAIN20 pilot; candidates trained only old40. "
            "Original sealed20DEV and finaltest remain unread. Future200TRAIN inclusion invalidates held qualification. "
            "Different capacity/recipe/exposure/train budgets cannot establish an architecture ceiling or significance."}


def _best(rows, ts=False):
    available = [r for r in rows.values() if r["full60"]["ranking_eligible"]
        and (not ts or r["method"] in ("transolver", "transolver_pp", "ts3"))]
    if not available:
        return []
    score = min(r["full60"]["primary_score"] for r in available)
    return [r for r in available if r["full60"]["primary_score"] == score]


def _number(value, digits=4):
    return f"{value:.{digits}g}" if finite(value) else "null"


def _percent(value):
    return f"{100*value:.3g}%" if finite(value) else "null"


def _brief(metrics, plot):
    rows = metrics["rows"]; best = _best(rows); best_ts = _best(rows, True)
    def labels(items):
        return "、".join(row["label"] for row in items) + ("（同分全部保留）" if len(items) > 1 else "") if items else "null"
    def span(values, percent=False, missing="null"):
        if not values or not all(finite(v) for v in values):
            return missing
        render = _percent if percent else _number
        low, high = min(values), max(values)
        return render(low) if low == high else render(low) + "–" + render(high)
    lines = ["最优 eligible：" + labels(best) + "。最佳 TS：" + labels(best_ts) + "。"]
    if best and best_ts:
        u_delta = [100*(a["full60"]["case_mean_u_relative_l2"]-b["full60"]["case_mean_u_relative_l2"]) for a in best for b in best_ts]
        s_delta = [100*(a["full60"]["case_mean_stress_relative_l2"]-b["full60"]["case_mean_stress_relative_l2"]) for a in best for b in best_ts]
        lines.append("最佳相对最佳 TS：均U差 " + span(u_delta) + "、均S差 " + span(s_delta) + " 个百分点（负值表示误差更小）。")
        inference_delta = [a["actual_inference_seconds"]-b["actual_inference_seconds"]
            if finite(a["actual_inference_seconds"]) and finite(b["actual_inference_seconds"]) else None for a in best for b in best_ts]
        train_delta = [(a["inherited_training_seconds"]-b["inherited_training_seconds"])/60
            if finite(a["inherited_training_seconds"]) and finite(b["inherited_training_seconds"]) else None for a in best for b in best_ts]
        lines.append("推理成本差 " + span(inference_delta, missing="unknown/nonexact") + " 秒，继承训练成本差 "
            + span(train_delta, missing="unknown/nonexact") + " 分钟。")
    elif best:
        lines.append("最佳 TS 为null，最佳相对最佳 TS的U/S百分点差与成本差均null。")
    else:
        lines.append("没有满足完整60例及协议资格的候选，U/S差与成本差均null；不按子集排名。")
    if best:
        groups = [("最佳 eligible", best), ("最佳 TS", best_ts), ("旧 FNO", [rows["fno_cad40_fresh4800"]])]
        covered = set(); selected = []
        for role, items in groups:
            fresh = [row for row in items if row["candidate_id"] not in covered]
            if fresh:
                selected.append((role, fresh)); covered.update(row["candidate_id"] for row in fresh)
        lines.extend(["", "| 方法 | 均U | 均S | 训练分钟 | 推理秒 |", "|---|---:|---:|---:|---:|"])
        for role, items in selected:
            lines.append("| " + role + "：" + labels(items) + " | "
                + span([row["full60"]["case_mean_u_relative_l2"] for row in items], True) + " | "
                + span([row["full60"]["case_mean_stress_relative_l2"] for row in items], True) + " | "
                + span([row["inherited_training_seconds"]/60 if finite(row["inherited_training_seconds"]) else None for row in items], missing="unknown/nonexact") + " | "
                + span([row["actual_inference_seconds"] for row in items], missing="unknown/nonexact") + " |")
        if len(best) > 1 or len(best_ts) > 1:
            lines.append("同分组的范围来自各候选完整60例指标；保留全部同分候选，单种子不作显著性声明。")
    costs = metrics["costs"]; receipt = metrics["receipt"]
    lines.extend(["", f"真实准备/独立readout/整campaign外成本：{_number(costs['stages']['preparation']['authoritative_seconds'])}/"
        f"{_number(costs['stages']['evaluation']['authoritative_seconds'])}/{_number(costs['campaign_authoritative_seconds'])}秒；"
        f"成本exact={costs['campaign_cost_exact']}，协议scientific qualification={metrics['scientific_qualification']}。",
        f"输入negative={receipt['input_negative']}；真实终态清单{receipt['prediction_terminal_entries']}/540；"
        f"共享GT原CSV解析{receipt['GT_actual']['source_GT_parses']}/20次，U/S病例读取"
        f"{receipt['GT_actual']['U_target_case_reads']}/{receipt['GT_actual']['S_target_case_reads']}（期望各60）；"
        "oldDEV/final/originalnewDEV20/dia/strain/H5 nodal读取均0。",
        "终态原因：" + (metrics["terminal_negative_reason"] or receipt["input_negative_reason"] or "固定流程终态；详见真实失败与外成本记录") + "。",
        "完整九候选、hot/worst、失败、参数、曝光和每载荷20/20CAD配对差："
        "[九行 CSV](metrics.csv) · [详细 JSON](metrics.json) · [标量比较图](comparison.png)。图状态：" + plot["status"] + "。",
        "这些20 CAD按缓存可用性与字符串排序选自官方TRAIN池，仅对old40训练候选提供独立候选pilot；不是原封存20DEV或无偏最终test。"
        "以后进入200TRAIN即失去held资格。TRAIN曝光、容量、优化制度及成本不同，不能据此宣称架构上限。",
        "原source540/input660/prediction2400、20源码pins及317/540 acquisition input-negative保持原样。"])
    if not costs["campaign_cost_exact"]:
        lines.append("未能证明的耗时记为unknown/nonexact（null），不零填，也不重置实验技术成本。")
    if not metrics["scientific_qualification"]:
        lines.append("protocol/cost negative：表中完整分母标量仅作nonqualified descriptive，正式选择score与leader/bestTS均null。")
    if plot["status"] != "completed":
        lines.append("图未完成：" + plot.get("reason", "unknown") + "；导出不能宣称全部PASS。")
    return "\n".join(lines) + "\n"


def _plot(metrics, output):
    os.environ["MPLCONFIGDIR"] = str(output / ".mplconfig")
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except Exception as error:
        return {"status": "incomplete", "reason": type(error).__name__ + ": " + str(error), "png": None, "svg": None}
    fig = None
    try:
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
        rows = list(metrics["rows"].values()); fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for axis, (metric, title) in zip(axes.flat, [("case_mean_u_relative_l2", "Mean U"),
            ("case_mean_stress_relative_l2", "Mean signed S"), ("case_mean_stress_hotspot_relative_l2", "Mean hotspot S"),
            ("worst_case_u_relative_l2", "Worst U"), ("worst_case_stress_relative_l2", "Worst signed S"),
            ("worst_case_stress_hotspot_relative_l2", "Worst hotspot S")]):
            for i, row in enumerate(rows):
                value = row["full60"][metric]
                if finite(value, True):
                    axis.bar(i, 100*value, color="#176B87" if row["method"] in ("transolver", "transolver_pp", "ts3") else "#729B79")
                else:
                    axis.text(i, .02, "NA", ha="center", va="bottom", transform=axis.get_xaxis_transform(), color="#777777")
            axis.set_xticks(range(9), [r["label"] for r in rows], rotation=45, ha="right")
            axis.set_title(title + " / fixed60 cases"); axis.set_ylabel("Relative L2 (%)"); axis.set_ylim(bottom=0)
            axis.grid(axis="y", alpha=.15); axis.set_axisbelow(True)
        subtitle = "Qualified complete-denominator scalar comparison" if metrics["scientific_qualification"] else "Protocol/cost negative: nonqualified descriptive scalars only"
        fig.suptitle("Available TRAIN-pool CAD20 pilot\n" + subtitle, fontsize=12)
        for extension in ("png", "svg"):
            temporary = output / ("comparison." + extension + ".tmp")
            fig.savefig(temporary, format=extension, dpi=180, facecolor="white")
            temporary.replace(output / ("comparison." + extension))
        require(all((output / ("comparison." + ext)).stat().st_size > 0 for ext in ("png", "svg")), "Aggregate figure output is empty")
        return {"status": "completed", "png": "comparison.png", "svg": "comparison.svg", "scalar_only": True}
    except Exception as error:
        return {"status": "incomplete", "reason": type(error).__name__ + ": " + str(error), "png": None, "svg": None}
    finally:
        if fig is not None:
            plt.close(fig)


def _write(filename, content):
    temporary = filename.with_name(filename.name + ".tmp")
    temporary.write_text(content, encoding="utf-8"); temporary.replace(filename)


def _csv(metrics):
    fields = ["candidate_id", "method", "fixed_step", "TRAIN_CADs", "planned_complete_TRAIN120_passes", "ranking_eligible", "primary_score",
        "nonqualified_descriptive_source_score", "mean_U", "mean_S", "mean_hot", "worst_U", "worst_S", "worst_hot",
        "true_failure_cases", "missing_terminal_entries", "actual_inference_seconds", "failed_inference_and_persistence_seconds",
        "inherited_training_seconds", "total_parameters", "exposure_min", "exposure_max", "exposure_total", "endpoint_status"]
    stream = io.StringIO(newline=""); writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader()
    for row in metrics["rows"].values():
        value = {name: row.get(name) for name in fields}; a = row["full60"]
        value.update(ranking_eligible=a["ranking_eligible"], primary_score=a["primary_score"],
            mean_U=a["case_mean_u_relative_l2"], mean_S=a["case_mean_stress_relative_l2"], mean_hot=a["case_mean_stress_hotspot_relative_l2"],
            worst_U=a["worst_case_u_relative_l2"], worst_S=a["worst_case_stress_relative_l2"], worst_hot=a["worst_case_stress_hotspot_relative_l2"])
        writer.writerow(value)
    return stream.getvalue()


def run(plan_path=PLAN, export=False):
    if not export:
        return {"status": "source_only_ready", "kind": KIND, "actual_export": False, "reads": 0, "writes": 0}
    began = time.monotonic(); index = {}
    plan, summary, inputs, costs, pins, exporter_sha = _gate(plan_path, index)
    receipt = _barrier_and_gt(plan, summary, inputs, index)
    metrics = _metrics(plan, summary, inputs, costs, receipt)
    output = path(plan["scientific"]["outputs"]["report"])
    identity = index[str(path(plan["scientific"]["outputs"]["readout"] + "/summary.json").relative_to(root()))]
    if (output / "manifest.json").exists():
        previous = _json(output / "manifest.json", {})
        require(previous.get("exporter_source_sha256") == exporter_sha and previous.get("experiment_summary_identity") == identity,
                "Report destination belongs to a different immutable export/experiment")
    output.mkdir(parents=True, exist_ok=True)
    plot = _plot(metrics, output)
    metrics["plot"] = plot
    metrics["leader_candidate_ids"] = [r["candidate_id"] for r in _best(metrics["rows"])]
    metrics["best_TS_candidate_ids"] = [r["candidate_id"] for r in _best(metrics["rows"], True)]
    metrics["equal_score_candidates_retained"] = True
    _write(output / "metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    _write(output / "metrics.csv", _csv(metrics)); _write(output / "BRIEF.md", _brief(metrics, plot))
    artifacts = {}
    for name in ("BRIEF.md", "metrics.json", "metrics.csv", "comparison.png", "comparison.svg"):
        filename = output / name
        if filename.is_file() and (not name.startswith("comparison") or plot["status"] == "completed"):
            value = filename.read_bytes(); artifacts[name] = {"path": str(filename), "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    manifest = {"kind": KIND, "schema": 1, "status": "completed" if plot["status"] == "completed" else "completed_incomplete",
        "actual_export": True, "scientific_digest": SCI, "exporter_source_sha256": exporter_sha, "experiment_source_pins": pins,
        "experiment_summary_identity": identity, "metadata_index": index, "artifacts": artifacts, "plot": plot,
        "candidate_rows": 9, "scientific_qualification": metrics["scientific_qualification"],
        "leader_candidate_ids": metrics["leader_candidate_ids"], "best_TS_candidate_ids": metrics["best_TS_candidate_ids"],
        "read_scope": "Saved JSON/scalars, frozen source text and /proc identities only; no arrays/PT/raw/GT/GPU/model/contract",
        "original_protocol_and_negative_preserved": True,
        "export_CPU_seconds_before_manifest_persistence": time.monotonic() - began, "exported_at_utc": utc(),
        "exporter_numeric_array_validation_performed": False,
        "saved_evaluator_numeric_barrier_is_the_array_proof": receipt["barrier_complete"]}
    _write(output / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return {"status": manifest["status"], "kind": KIND, "actual_export": True, "scientific_qualification": metrics["scientific_qualification"],
        "output": str(output), "plot": plot["status"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN); parser.add_argument("--export", action="store_true")
    args = parser.parse_args(); result = run(args.plan, export=args.export)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    if result["status"] == "completed_incomplete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
