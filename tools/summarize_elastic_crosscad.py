"""Small-JSON CPU report of the fixed five-CAD zero-start elastic screen.

No solver/model, raw-deck, array, target or checkpoint loader imports. An
optional Matplotlib plot consumes only already recorded scalar JSON metrics.
This file does not prepare inputs, run a solve or recompute target metrics.
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
DEFAULT_PLAN = "notes/geobc_elastic_crosscad_plan.json"
DEFAULT_OUTPUT = "results/elastic_crosscad"
CASE_KEYS = ("101_428/ver", "101_56/ver", "101_78/ver", "102_249/ver", "102_38/ver")
KIND = "elastic_crosscad_screen"
MAX_JSON_BYTES = 16 * 1024 * 1024


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


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(filename, diagnostics):
    filename = path(filename)
    if not filename.is_file():
        return None, {"path": str(filename), "present": False}
    try:
        info = filename.stat()
        if info.st_size > MAX_JSON_BYTES:
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
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(filename)


def fmt(value, digits=5):
    return f"{value:.{digits}g}" if finite(value) else "—"


def state_flag(value):
    return "通过" if value is True else "否" if value is False else "未记录"


def _clean(value):
    """Preserve missing values; reject nonfinite scalars rather than invent zeros."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _metric_aggregate(rows, field):
    values = [(row["case_key"], row.get(field)) for row in rows if finite(row.get(field))]
    if not values:
        return {"metric_cases": 0, "inventory_denominator": 5, "mean": None, "worst": None, "worst_case": None}
    worst = max(values, key=lambda item: item[1])
    return {"metric_cases": len(values), "inventory_denominator": 5,
            "mean": sum(value for _, value in values) / len(values), "worst": worst[1], "worst_case": worst[0]}


def _row(key, record, planned, tolerance, diagnostics):
    if not isinstance(record, dict):
        return {"case_key": key, "group": planned.get("group"), "record_present": False,
                "expected_nodes": planned.get("expected_geometry", {}).get("nodes"),
                "numerical_state": "missing", "validated_precision": False}
    metric = record.get("metrics")
    metric = metric if isinstance(metric, dict) else None
    costs = record.get("costs", {})
    memory = record.get("memory", {})
    receipt = record.get("read_receipt", {})
    metric_nodes = metric.get("nodes") if metric else None
    nodes = record.get("node_count")
    expected_nodes = planned.get("expected_geometry", {}).get("nodes")
    relative = metric.get("u_vector_relative_l2") if metric else None
    rms = metric.get("vector_RMS_mm") if metric else None
    maximum = metric.get("max_nodal_vector_error_mm") if metric else None
    zero_reference = metric.get("zero_reference") if metric else None
    reads = receipt.get("displacement_reads")
    if not isinstance(reads, int) or isinstance(reads, bool) or not 0 <= reads <= 1:
        diagnostics.append(f"{key}: displacement read count missing or outside [0,1]")
    numeric_pass = (record.get("numerically_converged") is True and record.get("numerical_state") == "converged"
                    and nonnegative(record.get("true_relative_residual")) and record["true_relative_residual"] <= tolerance)
    all_nodes = (isinstance(nodes, int) and nodes == expected_nodes and (metric_nodes is None or metric_nodes == nodes))
    absolute_finite = nonnegative(rms) and nonnegative(maximum)
    relative_valid = nonnegative(relative) or (relative is None and zero_reference is True)
    frozen_before_GT = receipt.get("finite_allN_prediction_and_equilibrium_persisted_before_GT") is True
    metrics_valid = metric is not None and reads == 1 and all_nodes and absolute_finite and relative_valid and frozen_before_GT
    compatibility_flags = (record.get("input_compatible"), record.get("operator_compatible"))
    compatible = True if all(value is True for value in compatibility_flags) else False if any(value is False for value in compatibility_flags) else None
    if metric and not metrics_valid:
        diagnostics.append(f"{key}: physical metrics are present but all-node/finite/read-count evidence is incomplete")
    if record.get("numerically_converged") is True and not numeric_pass:
        diagnostics.append(f"{key}: claimed convergence disagrees with state or the frozen true-residual threshold")
    return {"case_key": key, "group": record.get("group", planned.get("group")), "record_present": True,
            "attempted": record.get("attempted"), "input_compatible": record.get("input_compatible"),
            "operator_compatible": record.get("operator_compatible"), "compatible": compatible,
            "node_count": nodes, "cell_count": record.get("cell_count"), "expected_nodes": expected_nodes,
            "numerical_state": record.get("numerical_state"), "claimed_numerically_converged": record.get("numerically_converged"),
            "numerical_threshold_passed": numeric_pass, "true_relative_residual": record.get("true_relative_residual"),
            "iterations_executed_total": record.get("iterations_executed_total", record.get("iteration")),
            "metrics_available": metric is not None, "metrics_evidence_passed": metrics_valid,
            "all_original_nodes": all_nodes if metric else None, "finite_endpoint_and_equilibrium_before_GT": frozen_before_GT if metric else None,
            "u_vector_relative_l2": relative,
            "u_vector_relative_l2_percent": 100 * relative if finite(relative) else None,
            "vector_RMS_mm": rms, "max_nodal_vector_error_mm": maximum, "zero_reference": zero_reference,
            "region_metrics": metric.get("regions") if metric else None,
            "validated_precision": compatible is True and numeric_pass and metrics_valid,
            "cold_start_stage_seconds": costs.get("cold_case_accounted_wall_seconds"),
            "peak_allocated_bytes": memory.get("peak_allocated_bytes"), "peak_reserved_bytes": memory.get("peak_reserved_bytes"),
            "displacement_reads": reads, "costs_by_phase": costs, "memory": memory, "equilibrium": record.get("equilibrium"),
            "read_receipt": receipt, "reason": record.get("reason"),
            "input_manifest": record.get("input_manifest"), "operator_summary": record.get("operator_summary"),
            "physical_u_path": record.get("physical_u_path"), "prediction": record.get("prediction"),
            "constraint_reaction": record.get("constraint_reaction"), "source_grid_node_ids": record.get("source_grid_node_ids")}


def build_report(plan_path=DEFAULT_PLAN, run_path=None):
    diagnostics = []
    plan, plan_identity = read_json(plan_path, diagnostics)
    if not plan:
        return {"status": "pending", "kind": "elastic_crosscad_report", "outcome": "plan_unavailable",
                "sources": {"plan": plan_identity}, "rows": [], "diagnostics": diagnostics}
    if plan.get("kind") != KIND or tuple(plan.get("case_keys", ())) != CASE_KEYS:
        raise ValueError("Report requires the fixed five-CAD physical screen plan")
    filename = path(run_path if run_path is not None else plan["output"]) / "summary.json"
    summary, summary_identity = read_json(filename, diagnostics)
    completed = bool(summary and summary.get("status") == "completed")
    record_cases = summary.get("cases", {}) if summary else {}
    contract = summary.get("config", {}).get("contract", {}) if summary else {}
    tolerance = plan["recipe"]["solve"]["true_relative_residual_tolerance"]
    checks = {}
    if summary:
        checks.update(summary_kind=summary.get("kind") == KIND,
                      summary_case_order=tuple(summary.get("case_keys", ())) == CASE_KEYS,
                      contract_kind=contract.get("kind") == KIND,
                      contract_case_order=tuple(contract.get("case_keys", ())) == CASE_KEYS,
                      recipe_matches_plan=contract.get("recipe") == plan.get("recipe"),
                      no_extra_case_records=isinstance(record_cases, dict) and set(record_cases).issubset(CASE_KEYS))
        if completed:
            checks["all_five_case_records_present"] = isinstance(record_cases, dict) and set(record_cases) == set(CASE_KEYS)
        science = {key: value for key, value in contract.items()
                   if key not in ("schema", "science_sha256", "array_access_in_request_contract", "raw_deck_read_in_request_contract")}
        checks["contract_internal_digest"] = bool(contract.get("science_sha256")) and digest(science) == contract["science_sha256"]
    rows = [_row(key, record_cases.get(key) if isinstance(record_cases, dict) else None,
                 plan["cases"][key], tolerance, diagnostics) for key in CASE_KEYS]
    receipt = summary.get("read_receipt", {}) if summary else {}
    if completed:
        checks["global_displacement_reads_in_range"] = (isinstance(receipt.get("displacement_reads"), int)
                and not isinstance(receipt.get("displacement_reads"), bool) and 0 <= receipt["displacement_reads"] <= 5)
        row_reads = [row.get("displacement_reads") for row in rows]
        checks["global_displacement_reads_match_cases"] = all(isinstance(value, int) for value in row_reads) and sum(row_reads) == receipt.get("displacement_reads")
        for key in ("stress_reads", "old_final_reads", "DEV_dia_reads", "sealed_new_test_reads", "model_or_weight_reads", "optimizer_steps", "scaler_fits"):
            checks["zero_" + key] = receipt.get(key) == 0
            checks["all_cases_zero_" + key] = all(row.get("read_receipt", {}).get(key) == 0 for row in rows)
        operator_rows = [row for row in rows if row.get("operator_compatible") is True]
        checks["accepted_operators_eight_guards_once"] = all(row.get("read_receipt", {}).get("operator_K_guards_this_CAD") == 8
                                                           and row.get("read_receipt", {}).get("operator_rebuilds_in_solve") == 0 for row in operator_rows)
    passed = completed and bool(checks) and all(checks.values()) and not diagnostics
    validated = [row for row in rows if row.get("validated_precision")] if passed else []
    reported = [row for row in rows if row.get("metrics_evidence_passed")]
    counts = {"inventory_denominator": 5, "records_present": sum(row.get("record_present") is True for row in rows),
              "attempted": sum(row.get("attempted") is True for row in rows),
              "attempt_unknown": sum(row.get("attempted") is None for row in rows),
              "input_compatible": sum(row.get("input_compatible") is True for row in rows),
              "operator_compatible": sum(row.get("operator_compatible") is True for row in rows),
              "numerically_converged": sum(row.get("numerical_threshold_passed") is True for row in rows),
              "physical_metrics_evidence_passed": len(reported), "validated_precision_cases": len(validated),
              "missing_records": sum(row.get("record_present") is False for row in rows),
              "stage_budget_not_attempted": sum(row.get("attempted") is False and row.get("numerical_state") == "stage_budget_exhausted_negative" for row in rows)}
    validated_aggregates = {name: _metric_aggregate(validated, name) for name in
                           ("u_vector_relative_l2_percent", "vector_RMS_mm", "max_nodal_vector_error_mm")}
    return _clean({"status": "completed" if passed else "review_required" if completed else "pending",
                   "kind": "elastic_crosscad_report", "run_status": summary.get("status") if summary else "not_started",
                   "plan_rules_frozen": plan.get("scientific_rules_frozen"), "actual_execution_released": plan.get("actual_execution_released"),
                   "case_keys": list(CASE_KEYS), "sources": {"plan": plan_identity, "summary": summary_identity},
                   "contract": contract or None, "checks": checks, "checks_passed": passed,
                   "rows": rows, "counts": counts, "recorded_counts": summary.get("counts") if summary else None,
                   "validated_precision_aggregates": validated_aggregates,
                   "reported_metric_scope": "Individual finite budget endpoints may have metrics; only compatible, true-converged, allN metric-evidenced cases enter validated aggregates. Aggregate metric_cases is explicit, never an implied five-CAD success.",
                   "costs": summary.get("costs") if summary else None, "read_receipt": receipt or None,
                   "scope": summary.get("scope") if summary else None, "diagnostics": diagnostics,
                   "units": {"displacement": "mm", "relative_L2_display": "%", "force": "N", "stress_reads": 0},
                   "interpretation": ["Traditional zero-start approximate per-CAD physical solve; not neural-operator generalization ranking.",
                       "Compatibility, numerical convergence, physical error and cold-start cost are separate evidence.",
                       "Curved-P2 four-point quadrature, weighted-LS RBE3 transfer and proprietary-control exclusions do not establish OptiStruct exactness or engineering qualification.",
                       "No stress accuracy, DEV dia, old final, sealed test, model-weight or optimizer evidence is produced.",
                       "Historical FNO accuracy and learned training/inference costs use a different operator/input pipeline and are not a same-budget ranking here."],
                   "reporter_read_receipt": {"input_type": "plan and one summary small JSON only", "raw_array_reads": 0,
                       "raw_deck_reads": 0, "GT_reads": 0, "weight_reads": 0, "GPU_runs": 0, "metrics_recomputed_from_fields": 0}})


def brief(report):
    lines = ["# 五CAD物理基线", "", "固定3 TRAIN＋2 DEV、ver、零初值；这是近似传统弹性解算，分母始终为5。", ""]
    if report["status"] != "completed":
        lines += [f"当前 `{report['status']}` / 原运行 `{report.get('run_status', '未记录')}`：结果尚未完整或metadata/scope需复核，不宣称精度成功。", ""]
    counts = report.get("counts", {})
    if counts:
        lines += [f"输入兼容 {counts['input_compatible']}/5；算子兼容 {counts['operator_compatible']}/5；真残差≤1e−8 {counts['numerically_converged']}/5；有完整物理误差读出 {counts['physical_metrics_evidence_passed']}/5。", ""]
    lines += ["| CAD | 数值状态 | U L2 % | 冷启动秒 |",
              "|---|---|---:|---:|"]
    for row in report.get("rows", []):
        numerical = row.get("numerical_state") or "未记录"
        lines.append(f"| {row['case_key'].split('/')[0]} | {numerical} | {fmt(row.get('u_vector_relative_l2_percent'))} | {fmt(row.get('cold_start_stage_seconds'))} |")
    lines += ["", "逐案例真残差、RMS/max mm、峰值显存、GT读数及兼容细项见 metrics.json；缺失指标保留为空。"]
    aggregate = report.get("validated_precision_aggregates", {}).get("u_vector_relative_l2_percent", {})
    if aggregate.get("metric_cases"):
        lines += ["", f"仅数值通过且allN读出合规的 {aggregate['metric_cases']}/5 案例：U均值 {fmt(aggregate['mean'])}%，最差 {fmt(aggregate['worst'])}%（{aggregate['worst_case']}）；这是子集误差摘要，没有预设工程精度通过线。"]
    costs = report.get("costs") or {}
    if finite(costs.get("cumulative_stage_wall_seconds")):
        lines += ["", f"全阶段累计wall账本 {fmt(costs['cumulative_stage_wall_seconds'])} s，预算900 s，finalization overshoot {fmt(costs.get('budget_overshoot_seconds'))} s。data/operator/solve/验证/写盘/GT子计时保留JSON，不重复相加；未触及或未收敛案例不补零。"]
    lines += ["", "预算端点可有误差读出，但不算数值收敛。四点积分/RBE3/控制设置有明确近似；尚非OptiStruct精确重放或工程资格。与FNO的输入、算子及训练/推理成本不同，不作为同预算神经泛化排名；本工具不重新读取GT或计算字段指标。"]
    if report.get("diagnostics"):
        lines += ["", "需复核：" + "；".join(report["diagnostics"][:4])]
    return "\n".join(lines) + "\n"


def make_plot(report, filename):
    if report["status"] != "completed":
        return {"generated": False, "reason": "No complete validated JSON report; no placeholder plot"}
    rows = report["rows"]
    if not any(finite(row.get("u_vector_relative_l2_percent")) or finite(row.get("cold_start_stage_seconds")) for row in rows):
        return {"generated": False, "reason": "No actual finite scalar observations"}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), constrained_layout=True)
    colors = {True: "#246b4f", False: "#bd7b25"}
    for index, row in enumerate(rows):
        value = row.get("u_vector_relative_l2_percent")
        cost = row.get("cold_start_stage_seconds")
        color = colors[row.get("validated_precision") is True]
        if finite(value):
            axes[0].scatter(index, value, color=color, s=46, zorder=3)
            axes[0].annotate(f"{value:.3g}", (index, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        if finite(cost):
            axes[1].scatter(index, cost, color=color, s=46, zorder=3)
            axes[1].annotate(f"{cost:.3g}", (index, cost), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    labels = [row["case_key"].split("/")[0] for row in rows]
    for ax, label in zip(axes, ("All-node displacement relative L2 (%)", "Cold-start accounted wall (s)")):
        ax.set_xticks(range(5), labels, rotation=28)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.22)
        ax.margins(x=.12, y=.22)
    for ax, key in zip(axes, ("u_vector_relative_l2_percent", "cold_start_stage_seconds")):
        values = [row[key] for row in rows if finite(row.get(key))]
        if values and max(values) > 0:
            ax.set_ylim(0, max(values) * 1.25)
    fig.suptitle("Fixed five-CAD zero-start elastic screen — vertical load", fontsize=11)
    fig.text(.5, -.035, "Green: compatible + true convergence + metric evidence. Amber: other finite readouts. Missing cases have no invented points.", ha="center", fontsize=8)
    filename = path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {"generated": True, "path": str(filename), "source": "Only actual scalar summary JSON observations; no interpolation or field visualization"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--run", help="Run directory; default is the formal plan output")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--no-plot", action="store_true", help="Keep execution standard-library JSON/Markdown only")
    args = parser.parse_args()
    report = build_report(args.plan, args.run)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    output = path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report["figure"] = {"generated": False, "reason": "--no-plot"} if args.no_plot else make_plot(report, output / "comparison.png")
    write_json(output / "metrics.json", report)
    (output / "BRIEF.md").write_text(brief(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "figure": report["figure"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
