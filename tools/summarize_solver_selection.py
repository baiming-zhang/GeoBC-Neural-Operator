"""Summarize the six solver screens from small JSON receipts only.

Example (run on CPU after measured endpoints exist):
  /home/research/.venvs/geobc/bin/python tools/summarize_solver_selection.py \
      --plan notes/geobc_exploration_plan.json --output results/solver_selection

Reads only the exploration plan and each solver_screen directory's config.json,
status.json, metrics.jsonl and summary.json. Does not import a trainer/model,
read arrays/checkpoints, launch GPU work, or wait for unfinished jobs. Matplotlib
is imported lazily only when the report is actually generated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("transolver", "mlp", "fno", "transolver_pp", "ts3", "hashgrid")
LABELS = {"transolver": "Transolver", "mlp": "MLP", "fno": "FNO",
          "transolver_pp": "Transolver++", "ts3": "Transolver3", "hashgrid": "Hash"}
ENDPOINTS = (0, 120, 240, 480)
MAX_JSON_BYTES = 32 * 1024 * 1024
FIELDS = ("case_mean_u_relative_l2", "worst_case_u_relative_l2",
          "case_mean_stress_relative_l2", "worst_case_stress_relative_l2")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def reject_nonfinite(token):
    raise ValueError(f"nonfinite JSON token: {token}")


def read_small(path, diagnostics, jsonl=False):
    """Snapshot a live receipt; an unfinished final JSONL line is not an endpoint."""
    if not path.is_file():
        return ([] if jsonl else {}), {"path": str(path), "present": False}
    size = path.stat().st_size
    if size > MAX_JSON_BYTES:
        diagnostics.append(f"{path.name}: exceeds small-JSON limit ({size} bytes)")
        return ([] if jsonl else {}), {"path": str(path), "present": True, "bytes": size}
    data = path.read_bytes()
    receipt = {"path": str(path), "present": True, "bytes": len(data), "sha256": sha256(data)}
    try:
        text = data.decode("utf-8-sig")
        if not jsonl:
            result = json.loads(text, parse_constant=reject_nonfinite)
            if not isinstance(result, dict):
                raise ValueError("JSON object required")
            return result, receipt
        lines = text.splitlines()
        if text and not text.endswith(("\n", "\r")):
            diagnostics.append(f"{path.name}: incomplete final line ignored")
            lines = lines[:-1]
        result = []
        for index, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=reject_nonfinite)
                if not isinstance(record, dict):
                    raise ValueError("JSON object required")
                result.append(record)
            except (ValueError, TypeError) as error:
                diagnostics.append(f"{path.name}:{index}: {error}")
        return result, receipt
    except (UnicodeError, ValueError, TypeError) as error:
        diagnostics.append(f"{path.name}: {error}")
        return ([] if jsonl else {}), receipt


def point(record, diagnostics, label):
    if not isinstance(record, dict):
        return None
    step = record.get("step")
    validation = record.get("validation", {})
    score = record.get("selection_score")
    if (step not in ENDPOINTS or not isinstance(validation, dict)
            or validation.get("cases") != 15
            or not all(finite(validation.get(key)) and validation[key] >= 0 for key in FIELDS)
            or not finite(score) or score < 0):
        diagnostics.append(f"{label}: invalid endpoint/VAL15 physical metric schema")
        return None
    expected = 0.5 * (validation[FIELDS[0]] + validation[FIELDS[2]])
    if abs(score - expected) > 1e-12 * max(1.0, abs(expected)):
        diagnostics.append(f"{label}: joint VAL score does not match its U/stress checkpoint")
        return None
    # Retain complete physical endpoint groups, including per-case metrics; no
    # checkpoint or scientific array is accessed to reconstruct these values.
    return {"step": step, "selection_score": score,
            **{key: record.get(key) for key in ("elapsed_seconds", "training_loss",
               "validation", "train_readout", "costs", "memory")}}


def costs_view(costs):
    costs = costs if isinstance(costs, dict) else {}
    seconds = costs.get("inference_seconds")
    cases = costs.get("inference_case_exposures")
    nodes = costs.get("inference_node_exposures")
    valid = finite(seconds) and seconds >= 0
    return {"measured": costs,
            "all_node_inference_seconds_per_case_exposure":
                seconds / cases if valid and finite(cases) and cases > 0 else None,
            "all_node_inference_seconds_per_million_node_exposures":
                seconds * 1e6 / nodes if valid and finite(nodes) and nodes > 0 else None,
            "inference_scope": "Cumulative measured native encode/forward/CPU output conversion for TRAIN3+VAL15 at observed endpoints; excludes case loading and physical metric calculation. Per-case average mixes actual case sizes.",
            "train_scope": "Measured complete case updates, including case loading; cumulative actual seconds, not wall-time estimates."}


def summarize_row(job):
    method = job["method"]
    directory = (ROOT / job["output"]).resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError(f"Screen output leaves project: {directory}")
    diagnostics, sources = [], {}
    objects = {}
    for name in ("config.json", "status.json", "metrics.jsonl", "summary.json"):
        objects[name], sources[name] = read_small(directory / name, diagnostics, name.endswith(".jsonl"))
    config, status, summary = (objects[name] for name in ("config.json", "status.json", "summary.json"))
    records = {}
    for index, record in enumerate(objects["metrics.jsonl"]):
        parsed = point(record, diagnostics, f"metrics.jsonl record {index + 1}")
        if parsed is not None:
            # Resume receipts can repeat an endpoint; the last observed receipt
            # is retained, with the completed summary authoritative below.
            records[parsed["step"]] = parsed
    for index, record in enumerate(summary.get("endpoint_metrics", [])):
        parsed = point(record, diagnostics, f"summary endpoint {index}")
        if parsed is not None:
            records[parsed["step"]] = parsed
    initial = point(summary.get("initial_metrics"), diagnostics, "summary initial")
    final = point(summary.get("final_metrics"), diagnostics, "summary final")
    best = point(summary.get("best_metrics"), diagnostics, "summary best")
    completed = summary.get("status") == "completed"
    if completed:
        valid = (summary.get("method") == method and summary.get("endpoint_step") == 480
                 and initial is not None and initial["step"] == 0
                 and final is not None and final["step"] == 480 and best is not None
                 and summary.get("best_step") == best["step"]
                 and all(step in records for step in ENDPOINTS))
        if valid:
            minimum = min(records.values(), key=lambda item: (item["selection_score"], item["step"]))
            valid = (best["step"] == minimum["step"]
                     and best["validation"] == records[best["step"]]["validation"]
                     and best["selection_score"] == minimum["selection_score"])
        if not valid:
            diagnostics.append("completed summary is inconsistent with its best/final/four-endpoint receipts")
            completed = False
    state = ("completed" if completed else "invalid_completed_summary" if summary.get("status") == "completed"
             else status.get("phase", status.get("status", "pending")))
    if state == "completed" and not completed:
        state = "awaiting_summary"
    observed = [records[key] for key in sorted(records)]
    latest = observed[-1] if observed else None
    best_so_far = min(observed, key=lambda item: (item["selection_score"], item["step"])) if observed else None
    costs = summary.get("costs") if completed else status.get("costs")
    costs = costs if isinstance(costs, dict) else (latest or {}).get("costs", {})
    config = config or summary.get("config", {})
    parameters = summary.get("parameter_counts", config.get("parameter_counts"))
    memory = summary.get("memory") if completed else status.get("memory", (latest or {}).get("memory"))
    selected = best if completed else None
    trained = selected is not None and selected["step"] > 0
    eligible = trained and selected["validation"][FIELDS[0]] < 1 and selected["validation"][FIELDS[2]] < 1
    positive = [item for item in observed if item["step"] > 0
                and finite((item.get("costs") or {}).get("train_seconds"))
                and item["costs"]["train_seconds"] > 0]
    declining = len(positive) >= 2 and positive[-1]["selection_score"] < positive[0]["selection_score"]
    trend = {"first_trained_endpoint": positive[0]["step"] if positive else None,
             "last_trained_endpoint": positive[-1]["step"] if positive else None,
             "selection_score_change": positive[-1]["selection_score"] - positive[0]["selection_score"] if len(positive) >= 2 else None,
             "observed_decrease": declining,
             "retained_unqualified_route": bool(declining and not eligible),
             "interpretation": "Observed endpoint trend only; no interpolation, convergence or elimination claim."}
    curve = [{"step": item["step"], "train_seconds": item["costs"]["train_seconds"],
              "joint_validation_score": item["selection_score"]} for item in positive]
    return {"method": method, "label": LABELS[method], "output_directory": str(directory),
            "status": state, "steps": 480 if completed else status.get("step", (latest or {}).get("step", 0)),
            "planned_steps": 480, "selected_checkpoint": selected,
            "best_step": selected["step"] if selected else None,
            "best_label": ("未训练基线" if selected and selected["step"] == 0 else "共同VAL分数best" if selected else "尚无完成的checkpoint选择"),
            "best": best, "final": final, "initial": initial or records.get(0),
            "fixed_endpoint_480": final if final and final["step"] == 480 else records.get(480),
            "best_so_far_provisional": best_so_far if not completed else None,
            "observed_endpoints": observed, "curve_points": curve,
            "costs": costs_view(costs), "parameter_counts": parameters, "memory": memory,
            "source_config": config.get("model"), "precision": config.get("precision"),
            "case_order_sha256": summary.get("case_order_sha256", config.get("case_order_sha256")),
            "qualification": {"qualified": bool(eligible), "trained_checkpoint": bool(trained),
                "both_case_means_below_zero_baseline": bool(eligible),
                "both_worst_cases_below_zero_baseline": bool(selected and all(selected["validation"][key] < 1 for key in (FIELDS[1], FIELDS[3]))),
                "zero_field_relative_L2_baseline": 1.0,
                "rule": "completed AND best_step>0 AND VAL case-mean physical U<1 AND signed stress<1; worst-case flags are separate diagnostics, not additional selection rules."},
            "baseline_proximity": {"both_means_near_zero_baseline": bool(selected and all(selected["validation"][key] >= 0.95 for key in (FIELDS[0], FIELDS[2]))),
                "descriptive_band_lower_relative_L2": 0.95,
                "interpretation": "Display diagnostic only, not a selection gate or statistical significance test; small improvement over zero does not establish clear effectiveness."},
            "trained_endpoint_trend": trend, "diagnostics": diagnostics, "sources": sources}


def brief(report):
    def metric(item, keys):
        return " / ".join(f"{100 * item['validation'][key]:.2f}%" for key in keys) if item else "—"
    lines = ["# GeoBC solver 筛选", "", "| 方法 | status / steps | best checkpoint | U mean / worst | signed stress mean / worst | 训练秒 |",
             "|---|---|---|---|---|---|"]
    for row in report["methods"]:
        selected = row["selected_checkpoint"]
        checkpoint = f"{row['best_step']}（{row['best_label']}）" if selected else "—"
        seconds = row["costs"]["measured"].get("train_seconds")
        timing = f"{seconds:.2f}" if finite(seconds) else "—"
        lines.append(f"| {row['label']} | {row['status']} / {row['steps']}/480 | {checkpoint} | {metric(selected, FIELDS[:2])} | {metric(selected, FIELDS[2:])} | {timing} |")
    names = [LABELS[method] for method in report["qualified_shortlist"]]
    selection = "满足本轮双均值门槛的已训练候选：" + "、".join(names) + "（不据四轮结果宣称最终solver胜出）。" if names else "未形成合格候选。"
    near = [row["label"] for row in report["methods"] if row["qualification"]["qualified"] and row["baseline_proximity"]["both_means_near_zero_baseline"]]
    if near:
        selection += " " + "、".join(near) + "两均值均接近零场基线，尚无明显有效性证据（≥95%仅为文字诊断区间）。"
    slow = [LABELS[method] for method in report["retained_declining_unqualified_routes"]]
    if slow:
        selection += " 保留已观测训练端点下降但尚不合格的路线：" + "、".join(slow) + "。"
    lines += ["", selection, "",
              "固定 strict TRAIN40 / VAL5 CAD × ver/hor/tor 三个已知载荷；各家共享 PointNet 1024点→global32 与原25维 BC/material，四次完整TRAIN遍历、共同仅TRAIN拟合的含stress asinh四通道scaler。表中U与signed stress来自同一共同物理VAL分数best checkpoint；step 0是未训练基线，不作为已训练候选。完整480步与全部观测端点保留于 metrics.json。原生preset容量不同；FNO使用point→grid特征适配及grid→node查询，有分辨率与插值限制。本轮预算尚不能证明收敛，未达门槛或下降较慢的家族不作淘汰。", "",
              "[实测端点图](error_cost.png)只绘制正累计训练秒的joint VAL score散点，不作插值；全节点推理耗时、参数及峰值显存见 [metrics.json](metrics.json)。"]
    if any(row["diagnostics"] for row in report["methods"]):
        lines += ["", "部分运行收据尚未完整或存在schema诊断，详见 metrics.json；未以不完整摘要发布checkpoint选择。"]
    return "\n".join(lines) + "\n"


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    observed = 0
    for row in report["methods"]:
        points = row["curve_points"]
        if not points:
            continue
        ax.scatter([item["train_seconds"] for item in points],
                   [item["joint_validation_score"] for item in points], label=row["label"], s=36)
        for item in points:
            ax.annotate(str(item["step"]), (item["train_seconds"], item["joint_validation_score"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
        observed += len(points)
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1, label="Zero-field joint baseline")
    if not observed:
        ax.text(0.5, 0.5, "Awaiting measured positive-time endpoints", transform=ax.transAxes, ha="center")
    ax.set(xlabel="Measured cumulative training seconds", ylabel="Joint physical VAL relative L2 score",
           title="GeoBC solver screen: observed endpoints (step labels)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("notes/geobc_exploration_plan.json"))
    parser.add_argument("--output", type=Path, default=Path("results/solver_selection"))
    args = parser.parse_args(argv)
    plan_path = (ROOT / args.plan).resolve()
    diagnostics = []
    plan, plan_receipt = read_small(plan_path, diagnostics)
    if diagnostics:
        raise ValueError("Exploration plan could not be read: " + "; ".join(diagnostics))
    jobs = [job for job in plan.get("jobs", []) if job.get("kind") == "solver_screen"]
    if len(jobs) != 6 or {job.get("method") for job in jobs} != set(METHODS):
        raise ValueError("Exploration plan must identify exactly the six solver_screen families")
    rows = [summarize_row(job) for job in jobs]
    qualified = sorted((row for row in rows if row["qualification"]["qualified"]),
                       key=lambda row: row["selected_checkpoint"]["selection_score"])
    report = {"schema_version": "geobc_solver_selection_report_v1",
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "plan": plan_receipt, "report_tool_sha256": sha256(Path(__file__).read_bytes()),
              "selection_policy": "Completed summary.best_metrics, same checkpoint for U/stress; 0.5*(VAL15 case-mean physical U-vector relative L2 + signed-stress relative L2). Initial step0 may win this score but is explicitly untrained and ineligible for the trained shortlist.",
              "curve_policy": "Only observed step>0 endpoints with finite positive measured costs.train_seconds; scatter only, no interpolation or invented elapsed time. Step0 remains in JSON.",
              "methods": rows, "qualified_shortlist": [row["method"] for row in qualified],
              "shortlist_status": "qualified_trained_candidates" if qualified else "未形成合格候选",
              "retained_declining_unqualified_routes": [row["method"] for row in rows if row["trained_endpoint_trend"]["retained_unqualified_route"]],
              "limits": "Strict TRAIN40/VAL5 x known3, four TRAIN passes; common PointNet/global32+25 BC/material and TRAIN-only signed4/asinh scaler. Native capacities differ, FNO point-grid resolution/interpolation differs, no final-test/dia evaluation. This screen does not establish convergence or eliminate a solver family."}
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    (output / "BRIEF.md").write_text(brief(report), encoding="utf-8")
    plot(report, output / "error_cost.png")
    print(json.dumps({"output": str(output), "completed": sum(row["status"] == "completed" for row in rows),
                      "qualified_shortlist": report["qualified_shortlist"], "shortlist_status": report["shortlist_status"]}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
