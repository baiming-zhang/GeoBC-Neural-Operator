"""Small-JSON CPU report of the frozen signed-scalar stress readout screen.

No array, checkpoint, solver, model or target imports/reads. The optional figure
uses completed scalar observations only. Nothing launches or resumes a job.
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = "notes/geobc_stress_readout_plan.json"
DEFAULT_OUTPUT = "results/elastic_stress_readout"
KIND = "elastic_signed_stress_readout_screen"
ARMS = ("direct_zero_slots", "physical_tensor_slots")
TRAIN = ("101_428/ver", "101_56/ver", "101_78/ver")
DEV = ("102_249/ver", "102_38/ver")
ZERO_READ_KEYS = ("displacement_reads", "old_final_reads", "DEV_dia_reads",
                  "sealed_new_test_reads")


def path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", value):
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    result = Path(value).expanduser()
    return (result if result.is_absolute() else ROOT / result).resolve()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def nonnegative(value):
    return finite(value) and value >= 0


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def fmt(value, digits=4):
    return f"{value:.{digits}g}" if finite(value) else "—"


def percent(value):
    return fmt(100 * value) if finite(value) else "—"


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    return value


def read_json(filename, diagnostics):
    filename = path(filename)
    if not filename.is_file():
        return None, {"path": str(filename), "present": False}
    try:
        info = filename.stat()
        if info.st_size > 16 * 1024 * 1024:
            raise ValueError("Input exceeds the small-JSON size limit")
        raw = filename.read_bytes()
        value = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value, {"path": str(filename), "present": True, "bytes": len(raw),
                       "mtime_ns": info.st_mtime_ns, "sha256": hashlib.sha256(raw).hexdigest()}
    except (OSError, UnicodeError, ValueError) as error:
        diagnostics.append(f"{filename}: {type(error).__name__}: {error}")
        return None, {"path": str(filename), "present": True, "invalid": True}


def write_json(filename, value):
    filename = path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(filename)


def validate_plan(plan):
    gate = plan.get("comparison", {}).get("readout_pass_gate", {})
    if (plan.get("kind") != KIND or tuple(plan.get("train_keys", ())) != TRAIN
            or tuple(plan.get("validation_keys", ())) != DEV
            or tuple(plan.get("neural", {}).get("arms", ())) != ARMS
            or plan["neural"].get("seed") != 2026
            or plan["neural"].get("full_case_updates_per_arm") != 120
            or plan["neural"].get("passes_per_arm") != 40
            or plan["comparison"].get("fixed_DEV_denominator") != 2
            or gate.get("both_arms_complete120_and_all_DEV_predictions") is not True
            or gate.get("each_DEV_tensor_arm_stress_relativeL2_strictly_below") != .25
            or gate.get("mean_stress_relativeL2_reduction_at_least") != .20
            or gate.get("tensor_arm_worst_to_direct_worst_at_most") != 1.05
            or gate.get("mean_hotspot_relativeL2_reduction_at_least") != .10):
        raise ValueError("Requires the frozen TRAIN3/DEV2, seed2026,120-update and four numeric stress gates")
    return gate


def aggregate_cases(per_case):
    if (not isinstance(per_case, dict) or set(per_case) != set(DEV)
            or any(not isinstance(item, dict) for item in per_case.values())):
        return None
    errors = [per_case[key].get("stress_relative_l2") for key in DEV]
    hotspots = [per_case[key].get("hotspot_relative_l2") for key in DEV]
    complete_errors = all(nonnegative(value) for value in errors)
    complete_hotspots = all(nonnegative(value) for value in hotspots)
    return {"DEV_denominator": 2, "relative_metric_cases": sum(nonnegative(value) for value in errors),
            "case_mean_stress_relative_l2": sum(errors) / 2 if complete_errors else None,
            "worst_case_stress_relative_l2": max(errors) if complete_errors else None,
            "case_mean_stress_hotspot_relative_l2": sum(hotspots) / 2 if complete_hotspots else None,
            "worst_case_stress_hotspot_relative_l2": max(hotspots) if complete_hotspots else None,
            "worst_stress_case_keys": [key for key, error in zip(DEV, errors) if complete_errors and error == max(errors)],
            "worst_hotspot_case_keys": [key for key, error in zip(DEV, hotspots) if complete_hotspots and error == max(hotspots)],
            "per_case": per_case}


def relative_reduction(control, candidate):
    return 1 - candidate / control if finite(control) and control > 0 and nonnegative(candidate) else None


def arm_record(name, raw):
    raw = raw if isinstance(raw, dict) else {}
    completed = (raw.get("status") == "completed" and raw.get("completed120") is True
                 and raw.get("endpoint_step") == 120)
    validation = raw.get("validation")
    validation = validation if completed and isinstance(validation, dict) else {}
    aggregate = aggregate_cases(validation.get("per_case"))
    cold = raw.get("costs", {}).get("cold_deployment_per_case", {})
    values = {}
    complete_costs = completed and isinstance(cold, dict) and set(cold) == set(DEV)
    for key in DEV:
        entry = cold.get(key) if isinstance(cold, dict) else None
        entry = entry if isinstance(entry, dict) else {}
        value = entry.get("total_seconds")
        components_valid = (entry.get("components_complete") is True and nonnegative(value)
                            and all(nonnegative(entry.get(component)) for component in
                                    ("physical_GT_free_cold_seconds", "tensor_recovery_seconds",
                                     "incidence_mask_seconds", "neural_input_preparation_seconds",
                                     "model_load_seconds", "inference_decode_persistence_seconds")))
        if name == ARMS[0]:
            components_valid = (components_valid and entry.get("physical_GT_free_cold_seconds") == 0
                                and entry.get("tensor_recovery_seconds") == 0)
        else:
            components_valid = components_valid and finite(entry.get("physical_GT_free_cold_seconds")) and entry["physical_GT_free_cold_seconds"] > 0
        complete_costs = complete_costs and components_valid
        values[key] = value if components_valid else None
    cold_summary = {"DEV_denominator": 2, "complete": complete_costs,
                    "per_case_seconds": values, "components": cold,
                    "minimum_seconds": min(values.values()) if complete_costs else None,
                    "maximum_seconds": max(values.values()) if complete_costs else None,
                    "mean_seconds": sum(values.values()) / 2 if complete_costs else None,
                    "fresh_joint_measurement": False}
    return {"arm": name, "status": raw.get("status", "missing"), "completed120": completed,
            "endpoint_step": raw.get("endpoint_step"), "validation": aggregate,
            "validation_saved": validation or None, "training_readout": raw.get("training_readout"),
            "parameter_counts": raw.get("parameter_counts"),
            "initial_parameters_sha256": raw.get("initial_parameters_sha256"),
            "cold_cost_summary": cold_summary, "costs": raw.get("costs"),
            "first_actual_gradient_check": raw.get("first_actual_gradient_check"), "read_receipt": raw.get("read_receipt"),
            "memory": raw.get("memory"), "record": raw}


def build_report(plan_path=DEFAULT_PLAN, run_path=None):
    diagnostics = []
    plan, plan_identity = read_json(plan_path, diagnostics)
    empty_arms = {name: arm_record(name, None) for name in ARMS}
    if not plan:
        return {"status": "pending", "kind": "elastic_signed_stress_readout_report",
                "gate": {"passed": None, "state": "not_evaluated"}, "arms": empty_arms,
                "sources": {"plan": plan_identity}, "diagnostics": diagnostics}
    frozen_gate = validate_plan(plan)
    summary, summary_identity = read_json(path(run_path or plan["output"])/"summary.json", diagnostics)
    complete = bool(summary and summary.get("status") == "completed")
    raw_arms = summary.get("arms", {}) if summary else {}
    arms = {name: arm_record(name, raw_arms.get(name) if isinstance(raw_arms, dict) else None) for name in ARMS}
    receipt = summary.get("read_receipt", {}) if summary else {}
    contract = summary.get("config", {}).get("contract", {}) if summary else {}
    checks = {}
    if summary:
        checks = {"kind": summary.get("kind") == KIND,
                  "train_keys": tuple(summary.get("train_keys", ())) == TRAIN,
                  "validation_keys": tuple(summary.get("validation_keys", ())) == DEV,
                  "contract_present": isinstance(contract, dict) and bool(contract),
                  "contract_train_keys": tuple(contract.get("train_keys", ())) == TRAIN,
                  "contract_validation_keys": tuple(contract.get("validation_keys", ())) == DEV,
                  "frozen_plan_science_identity": contract.get("plan_science") == {
                      key: plan[key] for key in ("physics_features", "neural", "label_access", "budgets", "comparison")},
                  "exact_two_arms": isinstance(raw_arms, dict) and set(raw_arms) == set(ARMS),
                  "source_case_inventory": isinstance(summary.get("cases"), dict) and set(summary["cases"]) == set(TRAIN+DEV)}
    if complete:
        checks["both_arms_complete120"] = all(arm["completed120"] for arm in arms.values())
        checks["both_parameters_recorded_equal"] = (bool(arms[ARMS[0]]["parameter_counts"])
            and arms[ARMS[0]]["parameter_counts"] == arms[ARMS[1]]["parameter_counts"])
        initial = [arms[name]["initial_parameters_sha256"] for name in ARMS]
        checks["full_initial_trainable_identity"] = (all(isinstance(value, str) and re.fullmatch("[0-9a-fA-F]{64}", value) for value in initial)
                                                     and initial[0] == initial[1])
        for name in ARMS:
            aggregate = arms[name]["validation"]
            per_case = (aggregate or {}).get("per_case", {})
            checks[name+"_two_DEV_allN_metrics"] = (isinstance(aggregate, dict) and len(per_case) == 2
                and arms[name]["validation_saved"].get("denominator") == 2
                and all(integer(item.get("num_nodes")) and item["num_nodes"] > 0
                        and all(nonnegative(item.get(key)) for key in ("RMSE_MPa", "MAE_MPa", "max_abs_error_MPa"))
                        for item in per_case.values()))
            checks[name+"_zero_reference_semantics"] = bool(per_case) and all(
                (item.get("zero_reference") is False and nonnegative(item.get("stress_relative_l2"))
                 or item.get("zero_reference") is True and item.get("stress_relative_l2") is None)
                and (item.get("hotspot_zero_reference") is False and nonnegative(item.get("hotspot_relative_l2"))
                     or item.get("hotspot_zero_reference") is True and item.get("hotspot_relative_l2") is None)
                for item in per_case.values())
            first = arms[name]["first_actual_gradient_check"] or {}
            checks[name+"_first_actual_gradient"] = (first.get("passed") is True
                and first.get("all_trainable_present_and_finite") is True
                and all(finite(first.get(key)) and first[key] > 0 for key in ("encoder_norm", "head_norm", "context_leaf_norm"))
                and first.get("geometry_encodes") == 1 and first.get("original_context_backwards") == 1)
        checks["both_same_DEV_node_counts"] = all(
            (arms[ARMS[0]]["validation"] or {}).get("per_case", {}).get(key, {}).get("num_nodes") is not None
            and (arms[ARMS[0]]["validation"] or {}).get("per_case", {}).get(key, {}).get("num_nodes")
            == (arms[ARMS[1]]["validation"] or {}).get("per_case", {}).get(key, {}).get("num_nodes") for key in DEV)
        for key in ZERO_READ_KEYS:
            checks["zero_"+key] = receipt.get(key) == 0
        train_reads, dev_reads = receipt.get("TRAIN_stress_reads"), receipt.get("DEV_stress_reads")
        checks["actual_stress_read_receipt"] = (integer(train_reads) and train_reads >= 3 and integer(dev_reads) and dev_reads >= 2
            and receipt.get("stress_reads") == train_reads+dev_reads and receipt.get("GT_reads") == receipt.get("stress_reads"))
        checks["two_TRAIN_only_scaler_fits"] = receipt.get("scaler_fits") == 2
        checks["actual_optimizer_updates_240"] = receipt.get("optimizer_steps") == 240
    control = arms[ARMS[0]]["validation"] or {}
    tensor = arms[ARMS[1]]["validation"] or {}
    mean_gain = relative_reduction(control.get("case_mean_stress_relative_l2"), tensor.get("case_mean_stress_relative_l2"))
    hot_gain = relative_reduction(control.get("case_mean_stress_hotspot_relative_l2"), tensor.get("case_mean_stress_hotspot_relative_l2"))
    control_worst, tensor_worst = control.get("worst_case_stress_relative_l2"), tensor.get("worst_case_stress_relative_l2")
    worst_ratio = tensor_worst/control_worst if finite(control_worst) and control_worst > 0 and nonnegative(tensor_worst) else None
    per_case = tensor.get("per_case", {})
    criteria = {"both_arms_complete120": all(arm["completed120"] for arm in arms.values()),
                "each_DEV_below_25_percent": set(per_case) == set(DEV) and all(
                    nonnegative(item.get("stress_relative_l2")) and item["stress_relative_l2"] < .25 for item in per_case.values()),
                "mean_relative_reduction_at_least_20_percent": finite(mean_gain) and mean_gain >= .20,
                "worst_ratio_at_most_1_05": finite(worst_ratio) and worst_ratio <= 1.05,
                "hotspot_mean_reduction_at_least_10_percent": finite(hot_gain) and hot_gain >= .10}
    metadata_pass = complete and bool(checks) and all(checks.values())
    passed = bool(metadata_pass and all(criteria.values())) if complete else None
    saved_gate = summary.get("selection_gate") if summary else None
    saved_matches = saved_gate.get("passed") == passed if complete and isinstance(saved_gate, dict) else None
    if complete and not saved_matches:
        diagnostics.append("Saved selection_gate.passed does not match the independently recomputed complete-endpoint gate")
        passed = False
    gate = {"passed": passed, "state": "evaluated" if complete else "not_evaluated",
            "required_DEV_denominator": 2, "criteria": criteria, "frozen_thresholds": frozen_gate,
            "mean_relative_reduction": mean_gain, "worst_ratio": worst_ratio,
            "hotspot_mean_relative_reduction": hot_gain, "metadata_checks_passed": metadata_pass,
            "saved_gate": saved_gate, "saved_gate_passed_matches": saved_matches}
    plot_ready = complete and metadata_pass and saved_matches and all(
        arm["cold_cost_summary"]["complete"] and arm["validation"] is not None
        and all(nonnegative(arm["validation"].get(key)) for key in
                ("case_mean_stress_relative_l2", "worst_case_stress_relative_l2", "case_mean_stress_hotspot_relative_l2"))
        for arm in arms.values())
    if complete and not metadata_pass:
        diagnostics += ["Metadata check failed: "+name for name, passed_check in checks.items() if not passed_check]
    return clean({"status": "completed" if complete else "pending", "kind": "elastic_signed_stress_readout_report",
                  "run_status": summary.get("status") if summary else "missing",
                  "scientific_rules_frozen": plan.get("scientific_rules_frozen"),
                  "actual_execution_released": plan.get("actual_execution_released"),
                  "seed": 2026, "endpoint_step": 120, "train_keys": list(TRAIN), "validation_keys": list(DEV),
                  "gate": gate, "checks": checks, "arms": arms,
                  "source_cases": summary.get("cases") if summary else None,
                  "read_receipt": receipt or None, "scalers": summary.get("scalers") if summary else None,
                  "costs": summary.get("costs") if summary else None,
                  "shared_stage_memory": summary.get("memory") if summary else None,
                  "contract": contract or None, "sources": {"plan": plan_identity, "summary": summary_identity},
                  "plot_observations_complete": bool(plot_ready), "diagnostics": diagnostics,
                  "units": {"stress": "Original signed scalar MPa", "relative_l2": "fraction; plot percent", "tensor": "Approximate physical tensor MPa input, not solver scalar GT"},
                  "limitations": ["Single seed,TRAIN3/DEV2/ver and fixed120 endpoints; no final solver or statistical-significance claim",
                                  "GT displacement never a feature; source scalar sign/export/recovery convention remains unknown",
                                  "Cold accounting combines inherited measured physical phases and current measured phases; not a fresh joint latency measurement",
                                  "Direct deployment excludes physical solve/tensor recovery, but retains mask/topology and its own input/model/inference cost",
                                  "No old displacement accuracy is transferred to stress predictions"],
                  "created_at_utc": datetime.now(timezone.utc).isoformat()})


def brief(report):
    gate = report["gate"]
    verdict = "PASS" if gate.get("passed") is True else "FAIL" if gate.get("passed") is False else "尚未评估"
    lines = ["# 物理张量＋signed stress读出", "", f"固定粗筛门：**{verdict}**。"]
    if report.get("status") != "completed" or not gate.get("criteria", {}).get("both_arms_complete120"):
        lines += ["", "当前没有两臂完整120更新与完整DEV端点结果；不使用中途指标或补零。"]
    lines += ["", "| 读出臂 | DEV mean / worst S L2% | 热点 mean L2% | 单请求冷成本s |",
              "|---|---:|---:|---:|"]
    for name in ARMS:
        arm = report["arms"].get(name, {})
        metrics = arm.get("validation") or {}
        costs = arm.get("cold_cost_summary") or {}
        low, high = costs.get("minimum_seconds"), costs.get("maximum_seconds")
        cost_text = f"{fmt(low)}–{fmt(high)}" if nonnegative(low) and nonnegative(high) else "—"
        label = "直接 MLP／六零槽" if name == ARMS[0] else "物理 σ6／同 validity"
        lines.append(f"| {label} | {percent(metrics.get('case_mean_stress_relative_l2'))} / {percent(metrics.get('worst_case_stress_relative_l2'))} | {percent(metrics.get('case_mean_stress_hotspot_relative_l2'))} | {cost_text} |")
    lines += ["", "TRAIN3/DEV2、仅ver、单seed2026、固定120个完整case更新。比较原始signed MPa；两臂相同布局/初始状态，σ6来自物理预测位移，validity共同。热点为每case最高|真S|的1%原节点；完整MPa误差、最差ID、实际读取次数与节点曝光见metrics.json。"]
    if report.get("status") == "completed":
        lines += ["", f"mean相对改善{percent(gate.get('mean_relative_reduction'))}%、worst比值{fmt(gate.get('worst_ratio'))}、热点mean相对改善{percent(gate.get('hotspot_mean_relative_reduction'))}%；每DEV<25%的检查{gate.get('criteria', {}).get('each_DEV_below_25_percent')}。"]
    receipts = report.get("read_receipt") or {}
    lines += ["", f"真实累计：stress读取{receipts.get('stress_reads', '未提供')}次，displacement读取{receipts.get('displacement_reads', '未提供')}次；scaler fit记录{receipts.get('scaler_fits', '未提供')}。resume增加的实际读取按receipt保留，不固定写成5。"]
    lines += ["", "冷成本是已测baseline物理费用＋本阶段恢复/读出的组成核算，并非一次fresh联合计时。直接MLP不计物理解算或σ恢复；共同拓扑validity与自身输入/加载/推理仍计费。训练、实验共用准备、缓存增量延迟分别保存，不与推理冷成本混合。",
              "", "该门只决定是否值得扩大这个粗筛；未通过则结束此固定recipe。源scalar符号/恢复仍未知，张量proxy不是源stress GT；单seed/两个DEV不证明工程资格、泛化或整个solver族失败。没有把旧位移GT精度移到新应力端点。"]
    if report.get("diagnostics"):
        lines += ["", "元数据问题：" + "；".join(report["diagnostics"][:4])]
    return "\n".join(lines) + "\n"


def plot(report, filename):
    """Only completed physical scalar observations; no interpolation or fields."""
    if report.get("status") != "completed" or not report.get("plot_observations_complete"):
        return {"status": "not_generated", "reason": "Requires two completed arms, full DEV scalar metrics and both cold-cost components"}
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    colors = ("#486B8A", "#CE7B43")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.25))
    metric_names = ("case_mean_stress_relative_l2", "worst_case_stress_relative_l2", "case_mean_stress_hotspot_relative_l2")
    for index, name in enumerate(ARMS):
        values = [100 * report["arms"][name]["validation"][key] for key in metric_names]
        positions = [j + (index - .5) * .32 for j in range(3)]
        bars = axes[0].bar(positions, values, .30, label="Direct zero" if index == 0 else "Physical tensor", color=colors[index])
        for bar, value in zip(bars, values):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom", fontsize=8)
        cold = report["arms"][name]["cold_cost_summary"]["per_case_seconds"]
        y = [cold[key] for key in DEV]
        x = [j+(index-.5)*.20 for j in range(2)]
        axes[1].scatter(x, y, color=colors[index], s=35, label="Direct zero" if index == 0 else "Physical tensor")
        for px, value in zip(x, y):
            axes[1].annotate(f"{value:.2f}", (px, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    axes[0].set_xticks(range(3), ("Mean", "Worst", "Hotspot mean")); axes[0].set_ylabel("Signed stress relative L2 (%)")
    axes[1].set_xticks(range(2), tuple(key.split("/")[0] for key in DEV)); axes[1].set_ylabel("Accounted single-request cold cost (s)")
    for axis in axes:
        # Setting a lower limit disables autoscale; reserve actual headroom
        # explicitly so labels above the largest observation stay clear.
        axis.set_ylim(0, 1.22 * max(float(axis.dataLim.ymax), 1e-12))
        axis.grid(axis="y", alpha=.2); axis.set_axisbelow(True)
        axis.margins(x=.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.02))
    fig.suptitle("Fixed120 endpoint | TRAIN3 / DEV2 ver | Seed2026", fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .96))
    filename = path(filename); filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=180, bbox_inches="tight"); plt.close(fig)
    return {"status": "generated", "path": str(filename), "observations": "Actual completed metrics and cost points; no field rendering or interpolation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--run", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    report = build_report(args.plan, args.run)
    output = path(args.output); output.mkdir(parents=True, exist_ok=True)
    report["figure"] = {"status": "disabled"} if args.no_plot else plot(report, output/"comparison.png")
    write_json(output/"metrics.json", report)
    (output/"BRIEF.md").write_text(brief(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "gate": report["gate"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
