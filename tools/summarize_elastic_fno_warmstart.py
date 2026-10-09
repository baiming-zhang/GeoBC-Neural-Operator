"""Small-JSON CPU report of the frozen FNO initial-guess/PCG comparison.

No solver, model, raw-array, checkpoint or target imports/reads. Matplotlib is
lazy and receives only five complete actual scalar cold-cost observations.
This tool never launches a job or transfers old GT metrics to new endpoints.
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
DEFAULT_PLAN = "notes/geobc_elastic_hybrid_plan.json"
DEFAULT_OUTPUT = "results/elastic_fno_warmstart"
KIND = "elastic_frozen_fno_warmstart_screen"
CASE_KEYS = ("101_428/ver", "101_56/ver", "101_78/ver", "102_249/ver", "102_38/ver")
ZERO_READ_KEYS = ("GT_reads", "displacement_reads", "stress_reads", "old_final_reads", "DEV_dia_reads",
                  "sealed_new_test_reads", "optimizer_steps", "scaler_fits")


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


def fmt(value):
    return f"{value:.6g}" if finite(value) else "—"


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
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
        return value, {"path": str(filename), "bytes": len(raw), "mtime_ns": info.st_mtime_ns,
                       "sha256": hashlib.sha256(raw).hexdigest(), "present": True}
    except (OSError, UnicodeError, ValueError) as error:
        diagnostics.append(f"{filename}: {type(error).__name__}: {error}")
        return None, {"path": str(filename), "present": True, "invalid": True}


def write_json(filename, value):
    filename = path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(filename)


def base_pass(record, tolerance):
    return (isinstance(record, dict) and record.get("status") == "completed"
            and record.get("input_compatible") is True and record.get("operator_compatible") is True
            and record.get("numerically_converged") is True and record.get("numerical_state") == "converged"
            and nonnegative(record.get("true_relative_residual")) and record["true_relative_residual"] <= tolerance)


def solution_pass(value, relative_tolerance):
    if not isinstance(value, dict):
        return False
    if value.get("zero_reference") is True:
        return value.get("relative_l2") is None and nonnegative(value.get("vector_RMS_mm")) and value["vector_RMS_mm"] <= 1e-12
    return (value.get("zero_reference") is False and nonnegative(value.get("relative_l2"))
            and value["relative_l2"] <= relative_tolerance)


def row_record(key, record, base, tolerance, same_tolerance):
    if not isinstance(record, dict):
        return {"case_key": key, "record_present": False, "numerical_state": "missing",
                "zero_cold_cost_seconds": None, "warm_cold_cost_seconds": None, "reduction_percent": None,
                "warm_true_converged": False, "same_solution_passed": False, "complete_cold_components": False}
    costs = record.get("costs", {})
    zero = costs.get("zero_cold_cost_seconds")
    warm = costs.get("warm_cold_cost_seconds")
    components = costs.get("components_complete") is True
    cost_available = components and finite(zero) and zero > 0 and nonnegative(warm)
    ratio = warm / zero if cost_available else None
    reduction = 1 - ratio if cost_available else None
    converged = (record.get("numerically_converged") is True and record.get("numerical_state") == "converged"
                 and nonnegative(record.get("true_relative_residual")) and record["true_relative_residual"] <= tolerance)
    same = record.get("numerical_same_solution")
    same = same if isinstance(same, dict) else None
    return {"case_key": key, "record_present": True, "numerical_state": record.get("numerical_state"),
            "zero_cold_cost_seconds": zero, "warm_cold_cost_seconds": warm,
            "complete_cold_components": components, "complete_cold_cost_pair": cost_available,
            "warm_to_zero_cost_ratio": ratio, "reduction_fraction": reduction,
            "reduction_percent": 100 * reduction if finite(reduction) else None,
            "zero_true_converged": base_pass(base, tolerance), "warm_true_converged": converged,
            "same_solution_passed": solution_pass(same, same_tolerance), "numerical_same_solution": same,
            "raw_FNO_to_physical_reference": record.get("raw_FNO_to_physical_reference"),
            "feasible_initial_to_physical_reference": record.get("feasible_initial_to_physical_reference"),
            "true_relative_residual": record.get("true_relative_residual"),
            "initial_true_relative_residual": record.get("initial_true_relative_residual"),
            "iterations_executed_total": record.get("iterations_executed_total", record.get("iteration")),
            "costs": costs, "read_receipt": record.get("read_receipt"), "memory": record.get("memory"),
            "equilibrium": record.get("equilibrium"), "base_reference": base,
            "K_call_fields": {name: value for name, value in record.items() if "K" in name and ("call" in name.lower() or "application" in name.lower())},
            "reason": record.get("reason"), "record": record}


def build_report(plan_path=DEFAULT_PLAN, run_path=None):
    diagnostics = []
    plan, plan_identity = read_json(plan_path, diagnostics)
    if not plan:
        return {"status": "pending", "kind": "elastic_fno_warmstart_report", "sources": {"plan": plan_identity},
                "gate": {"passed": None, "state": "not_evaluated"}, "rows": [], "diagnostics": diagnostics}
    selection = plan.get("selection_gate", {})
    if (plan.get("kind") != KIND or tuple(plan.get("case_keys", ())) != CASE_KEYS
            or selection.get("inventory_denominator") != 5
            or selection.get("minimum_CADs_with_at_least20percent_total_cold_cost_reduction") != 4
            or selection.get("maximum_worst_case_warm_to_zero_cost_ratio") != 1.10
            or selection.get("both_methods_true_converged_required") != 5
            or selection.get("allN_same_solution_gate_required") != 5
            or plan["physics"]["true_relative_residual_tolerance"] != 1e-8
            or plan["comparison"]["fullN_vector_difference_relative_l2_maximum_for_same_solution_gate"] != 1e-5):
        raise ValueError("The reporter requires the frozen five-CAD 20%/4, 1.10, true-residual and same-solution gate")
    filename = path(run_path if run_path is not None else plan["output"]) / "summary.json"
    summary, summary_identity = read_json(filename, diagnostics)
    complete = bool(summary and summary.get("status") == "completed")
    contract = summary.get("config", {}).get("contract", {}) if summary else {}
    base_gate = contract.get("base_gate", {})
    base_records = base_gate.get("records", {})
    cases = summary.get("cases", {}) if summary else {}
    rows = [row_record(key, cases.get(key) if isinstance(cases, dict) else None,
                       base_records.get(key), 1e-8, 1e-5) for key in CASE_KEYS]
    checks = {}
    if summary:
        checks.update(summary_kind=summary.get("kind") == KIND,
                      fixed_case_order=tuple(summary.get("case_keys", ())) == CASE_KEYS,
                      contract_present=isinstance(contract, dict) and bool(contract),
                      no_extra_case_records=isinstance(cases, dict) and set(cases).issubset(CASE_KEYS))
    receipt = summary.get("read_receipt", {}) if summary else {}
    if complete:
        checks["all_five_case_records"] = isinstance(cases, dict) and set(cases) == set(CASE_KEYS)
        checks["saved_base_eligible"] = base_gate.get("eligible") is True
        checks["saved_base_all_five_complete"] = base_gate.get("all_five_completed") is True
        checks["saved_base_all_five_true_converged"] = base_gate.get("all_five_true_converged") is True
        checks["saved_base_all_five_prior_metrics_available"] = base_gate.get("all_five_original_metrics_available") is True
        checks["base_proof_five_records"] = isinstance(base_records, dict) and set(base_records) == set(CASE_KEYS)
        checks["base_true_residual_proof"] = all(base_pass(base_records.get(key), 1e-8) for key in CASE_KEYS)
        for key in ZERO_READ_KEYS:
            checks["zero_" + key] = receipt.get(key) == 0
            checks["all_cases_zero_" + key] = all((row.get("read_receipt") or {}).get(key) == 0 for row in rows)
    ratios = [row["warm_to_zero_cost_ratio"] for row in rows if finite(row.get("warm_to_zero_cost_ratio"))]
    numerical = sum(row.get("zero_true_converged") is True and row.get("warm_true_converged") is True for row in rows)
    same = sum(row.get("same_solution_passed") is True for row in rows)
    reductions = sum(finite(row.get("warm_to_zero_cost_ratio")) and row["warm_to_zero_cost_ratio"] <= .80 for row in rows)
    worst = max(ratios) if ratios else None
    criteria = {"both_true_converged_5_of_5": numerical == 5, "same_solution_5_of_5": same == 5,
                "at_least20percent_cold_gain_4_of_5": reductions >= 4,
                "worst_cost_ratio_at_most1_10": len(ratios) == 5 and worst <= 1.10}
    metadata_passed = complete and bool(checks) and all(checks.values()) and not diagnostics
    calculated_pass = metadata_passed and all(criteria.values())
    recorded_gate = summary.get("selection_gate", {}) if summary else {}
    if complete:
        checks["recorded_gate_consistent"] = recorded_gate.get("passed") is calculated_pass
        if not checks["recorded_gate_consistent"]:
            diagnostics.append("Recorded selection gate disagrees with the frozen scalar/read-scope recomputation")
            metadata_passed = False
            calculated_pass = False
    return clean({"status": "completed" if metadata_passed else "review_required" if complete else "pending",
                  "kind": "elastic_fno_warmstart_report", "run_status": summary.get("status") if summary else "not_started",
                  "sources": {"plan": plan_identity, "summary": summary_identity}, "case_keys": list(CASE_KEYS),
                  "contract": contract or None, "rows": rows, "checks": checks, "checks_passed": metadata_passed,
                  "gate": {"passed": calculated_pass if complete else None, "state": "PASS" if calculated_pass else "FAIL" if complete else "not_evaluated",
                           "criteria": criteria, "inventory_denominator": 5, "numerically_converged_cases": numerical,
                           "same_solution_passed_cases": same, "at_least20percent_reduction_cases": reductions,
                           "complete_cold_cost_pairs": len(ratios), "worst_warm_to_zero_cost_ratio": worst,
                           "recorded": recorded_gate or None},
                  "costs": summary.get("costs") if summary else None,
                  "training_cost_context": summary.get("training_cost_context") if summary else None,
                  "read_receipt": receipt or None, "diagnostics": diagnostics,
                  "units": {"cold_cost": "seconds", "gain_display": "%", "displacement_difference": "mm"},
                  "interpretation": ["Primary cost is accounted single-request cold cost; cached incremental and five-query model-load amortization remain separate secondary readouts.",
                      "The inherited input/operator/guard setup stays in cold accounting; historical zero PCG solve is never a warm solve budget component.",
                      "Differences to persisted zero-start physical_u are numerical-reference differences, not GT error. No old endpoint accuracy is assigned to new FNO/warm endpoints.",
                      "A failed macro gate ends this frozen-initial-guess candidate, without seed/width/loss/budget rescue or automatic job dispatch.",
                      "Five existing TRAIN/DEV geometries and approximate physical displacement do not establish sealed generalization, stress accuracy, OptiStruct exactness or engineering qualification."],
                  "reporter_read_receipt": {"small_JSON_only": True, "GT_reads": 0, "weight_reads": 0,
                      "array_reads": 0, "solver_imports": 0, "GPU_runs": 0, "field_metrics_recomputed": 0}})


def brief(report):
    gate = report["gate"]
    result = gate["state"] if gate["passed"] is not None else "尚未评估"
    lines = ["# 冻结FNO初值＋物理PCG", "", f"**总门：{result}。** 固定五CAD/ver，GT读0；不重训或选择seed。", ""]
    if gate["passed"] is not None:
        lines += [f"共同真残差≤1e−8：{gate['numerically_converged_cases']}/5；全N同解门：{gate['same_solution_passed_cases']}/5；完整冷成本收益≥20%：{gate['at_least20percent_reduction_cases']}/5；最坏warm/zero成本比 {fmt(gate['worst_warm_to_zero_cost_ratio'])}（上限1.10）。", ""]
    else:
        lines += ["尚无完整结果，不填造预算、误差或收益。", ""]
    lines += ["| CAD | zero冷成本 s | warm冷成本 s | 收益 % |", "|---|---:|---:|---:|"]
    for row in report.get("rows", []):
        lines.append(f"| {row['case_key'].split('/')[0]} | {fmt(row.get('zero_cold_cost_seconds'))} | {fmt(row.get('warm_cold_cost_seconds'))} | {fmt(row.get('reduction_percent'))} |")
    lines += ["", "主表为单请求accounted cold cost，包含神经几何/grid准备、模型/scaler加载、推理、FEM input/operator及warm solve；缓存实际延迟和batch摊销另存JSON，不把跳过setup计为0。真残差、初始残差、K调用、同解mm差异、详细成本与读数见metrics.json。", "",
              "同解参考为旧冻结收敛physical_u，不是GT；旧端点精度不转给FNO或warm端点。预算失败/缺失不补零。仅满足4/5明显成本收益、5/5共同真收敛/同解及最坏成本门才值得后续；失败不救微参数。尚非神经泛化排名、stress资格、OptiStruct精确重放或工程合格。"]
    if report["status"] == "review_required":
        lines += ["", "metadata/read-scope不一致，需复核后使用；不生成图。"]
    if report.get("diagnostics"):
        lines += ["", "复核项：" + "；".join(report["diagnostics"][:3])]
    return "\n".join(lines) + "\n"


def make_plot(report, filename):
    if report["status"] != "completed" or report["gate"]["complete_cold_cost_pairs"] != 5:
        return {"generated": False, "reason": "Only five complete actual scalar cold-cost pairs can be plotted; no placeholder"}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["rows"]
    fig, ax = plt.subplots(figsize=(8, 4.1), constrained_layout=True)
    zero = [row["zero_cold_cost_seconds"] for row in rows]
    warm = [row["warm_cold_cost_seconds"] for row in rows]
    ax.bar([index - .18 for index in range(5)], zero, width=.36, color="#65788e", label="Zero-start PCG")
    ax.bar([index + .18 for index in range(5)], warm, width=.36, color="#287c75", label="Frozen FNO + same PCG")
    for index, row in enumerate(rows):
        ax.annotate(f"{row['reduction_percent']:+.1f}%", (index, max(zero[index], warm[index])),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(range(5), [row["case_key"].split("/")[0] for row in rows])
    ax.set_ylabel("Accounted single-request cold cost (s)")
    ax.set_title(f"Fixed five-CAD macro gate: {report['gate']['state']}")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.14), ncol=2, frameon=False)
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)
    ax.margins(y=.22)
    filename = path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {"generated": True, "path": str(filename), "scope": "Actual complete JSON cold costs only; no interpolation or GT-field plot"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--run", help="Override the formal plan's output directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    report = build_report(args.plan, args.run)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    output = path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report["figure"] = {"generated": False, "reason": "--no-plot"} if args.no_plot else make_plot(report, output / "comparison.png")
    write_json(output / "metrics.json", report)
    (output / "BRIEF.md").write_text(brief(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "gate": report["gate"]["state"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
