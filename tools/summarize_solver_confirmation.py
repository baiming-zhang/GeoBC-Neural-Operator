"""CPU-only confirmation report from small JSON receipts; never read weights.

  python tools/summarize_solver_confirmation.py --check
  python tools/summarize_solver_confirmation.py --output results/solver_confirmation

Reads a pending/frozen confirmation plan and its declared run directories.
Produces BRIEF.md, metrics.json, and at most one measured-point scatter plot.
No trainer, torch, arrays, checkpoint, GPU work, waiting, or solver selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"mlp": "MLP", "hashgrid": "Hash", "fno": "FNO", "transolver": "Transolver",
          "transolver_pp": "Transolver++", "ts3": "Transolver-3"}
FIELDS = ("case_mean_u_relative_l2", "worst_case_u_relative_l2",
          "case_mean_stress_relative_l2", "worst_case_stress_relative_l2",
          "case_mean_stress_hotspot_relative_l2")
MAX_JSON_BYTES = 32 * 1024 * 1024


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(a, b):
    return finite(a) and finite(b) and abs(a-b) <= 1e-9 * max(1., abs(a), abs(b))


def path(value):
    result = (ROOT / value).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError(f"Receipt/output leaves project: {result}")
    return result


def invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read_small(filename, diagnostics, jsonl=False):
    if not filename.is_file():
        return ([] if jsonl else {}), {"path": str(filename), "present": False}
    size = filename.stat().st_size
    if size > MAX_JSON_BYTES:
        diagnostics.append(f"{filename.name}: exceeds small JSON limit")
        return ([] if jsonl else {}), {"path": str(filename), "present": True, "bytes": size}
    data = filename.read_bytes()
    receipt = {"path": str(filename), "present": True, "bytes": len(data),
               "sha256": hashlib.sha256(data).hexdigest()}
    try:
        text = data.decode("utf-8-sig")
        if not jsonl:
            result = json.loads(text, parse_constant=invalid_constant)
            if not isinstance(result, dict):
                raise ValueError("Expected JSON object")
            return result, receipt
        lines = text.splitlines()
        if text and not text.endswith(("\r", "\n")):
            diagnostics.append(f"{filename.name}: unfinished final line ignored")
            lines = lines[:-1]
        result = []
        for index, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line, parse_constant=invalid_constant)
                if not isinstance(item, dict):
                    raise ValueError("Expected JSON object")
                result.append(item)
            except (ValueError, TypeError) as error:
                diagnostics.append(f"{filename.name}:{index}: {error}")
        return result, receipt
    except (ValueError, TypeError, UnicodeError) as error:
        diagnostics.append(f"{filename.name}: {error}")
        return ([] if jsonl else {}), receipt


def metric_point(record, diagnostics, origin):
    if not isinstance(record, dict):
        return None
    step, validation, score = record.get("step"), record.get("validation", {}), record.get("selection_score")
    valid = (isinstance(step, int) and not isinstance(step, bool) and step >= 0
             and isinstance(validation, dict) and validation.get("cases") == 15
             and all(finite(validation.get(key)) and validation[key] >= 0 for key in FIELDS)
             and finite(score) and score >= 0)
    if not valid:
        diagnostics.append(f"{origin}: incomplete VAL15 physical metric receipt")
        return None
    expected = .5 * (validation[FIELDS[0]] + validation[FIELDS[2]])
    if not close(score, expected):
        diagnostics.append(f"{origin}: score differs from common U/stress checkpoint score")
        return None
    costs = record.get("costs", {})
    measured_seconds = costs.get("train_seconds") if isinstance(costs, dict) else None
    if not finite(measured_seconds) or measured_seconds < 0:
        diagnostics.append(f"{origin}: missing/nonfinite measured train_seconds")
        return None
    return {"step": step, "selection_score": score, "train_seconds": measured_seconds,
            "validation": validation, "train_readout": record.get("train_readout"),
            "costs": costs, "budget": record.get("budget"), "memory": record.get("memory"),
            "elapsed_seconds": record.get("elapsed_seconds"), "training_loss": record.get("training_loss"),
            "u_hotspot_relative_l2": None, "u_hotspot_status": "not measured by shared trainer"}


def budget_view(budget, costs, config, recipe):
    budget = budget if isinstance(budget, dict) else {}
    costs = costs if isinstance(costs, dict) else {}
    actual = budget.get("total_train_seconds", budget.get("actual_seconds", costs.get("train_seconds")))
    parent = budget.get("parent_train_seconds")
    if parent is None and config.get("contract", {}).get("initialization", {}).get("mode") == "fresh":
        parent = 0.
    incremental = budget.get("incremental_train_seconds")
    if incremental is None and finite(actual) and finite(parent):
        incremental = actual-parent
    requested = budget.get("requested_seconds", recipe.get("train_budget_seconds"))
    overshoot = budget.get("overshoot_seconds")
    if overshoot is None and finite(actual) and finite(requested):
        overshoot = max(0., actual-requested)
    return {"requested_train_seconds": requested, "actual_train_seconds": actual,
            "parent_train_seconds": parent, "incremental_train_seconds": incremental,
            "overshoot_seconds": overshoot, "budget_crossing_step": budget.get("crossing_step"),
            "overshoot_updates": budget.get("overshoot_updates"),
            "complete_pass": budget.get("complete_pass"), "max_steps": budget.get("max_steps"),
            "evaluation_seconds": costs.get("evaluation_seconds"),
            "inference_seconds": costs.get("inference_seconds"),
            "scope": "Cumulative complete-case update seconds, including case loading/H2D and synchronized forward/backward/step; evaluation/checkpoint/cache time separate."}


def run_row(method, seed, job, recipe):
    row = {"method": method, "seed": seed, "status": "pending_job_definition",
           "endpoint_primary": None, "validation_best_secondary": None, "observed_endpoints": [],
           "latest_measured_provisional": None, "steps": None, "stop_reason": None,
           "budget": budget_view({}, {}, {}, recipe), "diagnostics": [], "sources": {}}
    if job is None:
        return row
    directory = path(job["output"])
    row["output_directory"] = str(directory)
    objects = {}
    for name in ("config.json", "status.json", "metrics.jsonl", "summary.json"):
        objects[name], row["sources"][name] = read_small(directory / name, row["diagnostics"], name.endswith(".jsonl"))
    config, status, summary = (objects[name] for name in ("config.json", "status.json", "summary.json"))
    config = config or summary.get("config", {})
    points = {}
    for origin, records in (("metrics.jsonl", objects["metrics.jsonl"]),
                            ("summary.endpoint_metrics", summary.get("endpoint_metrics", []))):
        for index, record in enumerate(records):
            item = metric_point(record, row["diagnostics"], f"{origin}[{index}]")
            if item is not None:
                points[item["step"]] = item
    final = metric_point(summary.get("final_metrics"), row["diagnostics"], "summary.final_metrics")
    best = metric_point(summary.get("best_metrics"), row["diagnostics"], "summary.best_metrics")
    if final is not None:
        points[final["step"]] = final
    ordered = [points[step] for step in sorted(points)]
    latest = ordered[-1] if ordered else None
    completed = summary.get("status") == "completed"
    budget = budget_view(summary.get("budget") if completed else status.get("budget"),
                         summary.get("costs") if completed else status.get("costs", (latest or {}).get("costs")),
                         config, recipe)
    if completed:
        valid = (summary.get("method") == method and summary.get("seed") == seed
                 and final is not None and final["step"] == summary.get("endpoint_step")
                 and best is not None and best["step"] == summary.get("best_step")
                 and summary.get("stop_reason") in ("train_budget_seconds", "max_steps")
                 and close(budget["actual_train_seconds"], final["train_seconds"])
                 and finite(budget["parent_train_seconds"]) and budget["parent_train_seconds"] >= 0
                 and finite(budget["incremental_train_seconds"]) and budget["incremental_train_seconds"] >= 0
                 and close(budget["parent_train_seconds"] + budget["incremental_train_seconds"], budget["actual_train_seconds"]))
        if best is not None and ordered:
            minimum = min(ordered, key=lambda item: (item["selection_score"], item["step"]))
            valid = (valid and close(best["selection_score"], minimum["selection_score"])
                     and best["validation"] == points.get(best["step"], {}).get("validation"))
        if summary.get("stop_reason") == "train_budget_seconds":
            valid = (valid and finite(budget["requested_train_seconds"])
                     and budget["actual_train_seconds"] >= budget["requested_train_seconds"]
                     and budget["complete_pass"] is True)
        if not valid:
            row["diagnostics"].append("Completed receipt is inconsistent with method/seed, endpoint/best, or cumulative budget; not published as completed.")
            completed = False
    phase = status.get("phase", status.get("status", "pending"))
    state = "completed" if completed else "invalid_completed_summary" if summary.get("status") == "completed" else phase
    if state == "completed" and not completed:
        state = "awaiting_summary"
    row.update(status=state, steps=summary.get("endpoint_step") if completed else status.get("step", (latest or {}).get("step")),
               stop_reason=summary.get("stop_reason") if completed else None,
               endpoint_primary=final if completed else None,
               validation_best_secondary=best if completed else None,
               latest_measured_provisional=None if completed else latest,
               best_so_far_provisional=None if completed or not ordered else min(ordered, key=lambda item: item["selection_score"]),
               observed_endpoints=ordered, budget=budget,
               best_step=summary.get("best_step") if completed else None,
               best_checkpoint_origin=summary.get("best_checkpoint_origin") if completed else None,
               best_checkpoint_trained=bool(completed and best["step"] > 0),
               parent_elapsed_seconds=summary.get("parent_elapsed_seconds"),
               elapsed_seconds=summary.get("elapsed_seconds", status.get("elapsed_seconds")),
               phase_elapsed_seconds=summary.get("phase_elapsed_seconds"),
               parameter_counts=summary.get("parameter_counts", config.get("parameter_counts")),
               memory=summary.get("memory", status.get("memory", (latest or {}).get("memory"))),
               model=config.get("model"), precision=config.get("precision"),
               first_update_gradient_check=summary.get("first_update_gradient_check"),
               case_order_sha256=summary.get("case_order_sha256", config.get("case_order_sha256")),
               common_input_identity={key: config.get(key) for key in ("scaler", "cache_contract", "train_keys", "validation_keys", "training_readout_keys")})
    return row


def build_report(plan_path, jobs_path=None):
    diagnostics = []
    plan, receipt = read_small(path(plan_path), diagnostics)
    if diagnostics or not plan:
        raise ValueError("Confirmation plan missing or invalid: " + "; ".join(diagnostics))
    recipe = plan.get("recipe", {})
    methods, seeds = recipe.get("methods", []), recipe.get("seeds", plan.get("seeds", [2026, 2027]))
    if len(methods) > 3 or len(set(methods)) != len(methods) or any(method not in LABELS for method in methods):
        raise ValueError("Report supports at most three distinct declared solver candidates")
    if len(set(seeds)) != len(seeds) or any(seed not in (2026, 2027) for seed in seeds):
        raise ValueError("Only declared seeds2026/2027 are supported")
    jobs = list(plan.get("jobs", []))
    extra_receipt = None
    if jobs_path is not None:
        extra, extra_receipt = read_small(path(jobs_path), diagnostics)
        jobs += [job for job in extra.get("jobs", []) if job.get("kind") == "solver_confirmation"]
    lookup = {}
    for job in jobs:
        if job.get("method") not in methods or job.get("seed") not in seeds:
            continue
        key = (job["method"], job["seed"])
        if key in lookup and lookup[key]["output"] != job["output"]:
            raise ValueError(f"Conflicting declared output directories for {key}")
        lookup[key] = job
    selection = plan.get("candidate_selection", "")
    rows = [{"method": method, "label": LABELS[method],
             "priority": "lower-priority slow structural route" if method == "transolver_pp" and "lower-priority" in selection else "declared candidate",
             "runs": [run_row(method, seed, lookup.get((method, seed)), recipe) for seed in seeds]}
            for method in methods]
    identities = [run["common_input_identity"] for row in rows for run in row["runs"]
                  if run.get("common_input_identity", {}).get("scaler") is not None]
    common_match = all(item == identities[0] for item in identities[1:]) if identities else None
    return {"schema": "geobc_solver_confirmation_report_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "candidates_pending" if plan.get("candidates_frozen") is not True else "snapshot",
            "plan": receipt, "jobs_plan": extra_receipt, "candidates_frozen": plan.get("candidates_frozen") is True,
            "candidate_selection": selection, "recipe": recipe, "methods": rows,
            "completed_runs": sum(run["status"] == "completed" for row in rows for run in row["runs"]),
            "planned_runs": len(methods)*len(seeds), "common_input_and_scaler_match": common_match,
            "diagnostics": diagnostics, "report_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "primary_readout": "Measured cumulative-training-budget endpoint; completed only. Overshoot and safety-cap stop remain explicit.",
            "secondary_readout": "One common VAL15 score best checkpoint, including inherited screen history for seed2026; U/stress/hotspot from same checkpoint. Step0 best is explicitly untrained.",
            "curve_policy": "Measured positive cumulative costs.train_seconds at committed evaluations; scatter only, no interpolation. Inherited screen points retained with their actual costs.",
            "limits": "Development-validation5 reused; final5/dia untouched by this report. Two seeds do not establish statistical significance. No solver winner/qualification is inferred from errors below100%. FNO is a node-splat/trilinear geometry-conditioned candidate, with adapter/resolution limits. Stress hotspot is the source-|stress| highest1% per case; no U hotspot receipt exists."}


def fmt(value, percent=False):
    return f"{value*100 if percent else value:.2f}" if finite(value) else "—"


def metric(item, fields):
    return " / ".join(fmt(item["validation"].get(key), True) for key in fields) if item else "—"


def brief(report):
    lines = ["# GeoBC 求解器确认轮", "",
             f"已完成 {report['completed_runs']}/{report['planned_runs']}；候选{'已冻结' if report['candidates_frozen'] else '尚未冻结'}。固定累计训练成本端点为主结果，共同VAL分数best为次结果。", ""]
    if report["candidate_selection"]:
        lines += ["冻结的候选依据：" + report["candidate_selection"], ""]
    lines += ["| 候选 / seed | 状态 / step / stop | 预算端点 U mean / worst % | S mean / worst / hotspot % | actual / parent / incremental训练秒 | 超额秒 |",
              "|---|---|---:|---:|---:|---:|"]
    for row in report["methods"]:
        for run in row["runs"]:
            budget = run["budget"]
            timing = " / ".join(fmt(budget[key]) for key in ("actual_train_seconds", "parent_train_seconds", "incremental_train_seconds"))
            state = f"{run['status']} / {run['steps'] if run['steps'] is not None else '—'} / {run['stop_reason'] or '—'}"
            lines.append(f"| {row['label']} / {run['seed']} | {state} | {metric(run['endpoint_primary'], FIELDS[:2])} | {metric(run['endpoint_primary'], FIELDS[2:])} | {timing} | {fmt(budget['overshoot_seconds'])} |")
    lines += ["", "| 同一候选 / seed | 次结果：共同VAL best step | U mean / worst % | S mean / worst / hotspot % | best累计训练秒 |",
              "|---|---|---:|---:|---:|"]
    for row in report["methods"]:
        for run in row["runs"]:
            best = run["validation_best_secondary"]
            label = str(best["step"]) + ("（未训练基线）" if best["step"] == 0 else "") if best else "尚未完成"
            lines.append(f"| {row['label']} / {run['seed']} | {label} | {metric(best, FIELDS[:2])} | {metric(best, FIELDS[2:])} | {fmt(best['train_seconds']) if best else '—'} |")
    lines += ["", "未完成组的主/次结果留空；最新已测端点仅在metrics.json标为provisional。实际600秒附近按完整pass停，max_steps安全上限未达预算时单列；不能把超额或父成本抹去。", "",
              "相同40/5 CAD、三known-load、TRAIN-only共同scaler/global32/input与固定配方；seed2026沿screen last480轨迹及Adam/RNG接续，seed2027 fresh，cloud_seed2026不变。热点仅指源应力最高1%节点，U热点未测；表中误差均为物理解码后全原节点指标。", "",
              "两个seed逐一报告，不宣称统计显著，不按均值低于100%自动宣布胜出。FNO适配/分辨率限制、慢路线尚未充分收敛和反复使用开发验证的限制保留；final5/dia继续隔离。完整历史、每case指标、参数/显存/推断成本见[metrics.json](metrics.json)。"]
    if report["common_input_and_scaler_match"] is False:
        lines += ["", "**输入/scaler收据不一致，结果不能作为共同配方排名。**"]
    if report.get("figure", {}).get("generated"):
        lines += ["", "![实测成本与物理验证误差](precision_vs_time.png)", "散点为真实累计训练秒/已提交评估，无插值；大空心点标完成的预算端点。"]
    if any(run["diagnostics"] for row in report["methods"] for run in row["runs"]) or report["diagnostics"]:
        lines += ["", "部分收据存在schema诊断，详见metrics.json；未将不完整summary作为完成结果。"]
    return "\n".join(lines) + "\n"


def plot(report, filename):
    if not any(point["train_seconds"] > 0 for row in report["methods"] for run in row["runs"] for point in run["observed_endpoints"]):
        return {"generated": False, "reason": "No measured positive-time evaluations yet"}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for index, row in enumerate(report["methods"]):
        for run in row["runs"]:
            points = [point for point in run["observed_endpoints"] if point["train_seconds"] > 0]
            if not points:
                continue
            marker = "o" if run["seed"] == 2026 else "^"
            color = f"C{index}"
            ax.scatter([p["train_seconds"] for p in points], [100*p["selection_score"] for p in points],
                       marker=marker, color=color, s=25, alpha=.65, label=f"{row['label']} / {run['seed']}")
            endpoint = run["endpoint_primary"]
            if endpoint and endpoint["train_seconds"] > 0:
                ax.scatter([endpoint["train_seconds"]], [100*endpoint["selection_score"]],
                           marker=marker, facecolors="none", edgecolors=color, s=120, linewidths=1.7)
    requested = report["recipe"].get("train_budget_seconds")
    if finite(requested):
        ax.axvline(requested, color="0.7", linestyle=":", linewidth=1)
    ax.axhline(100, color="0.65", linestyle="--", linewidth=1)
    ax.set(xlabel="Measured cumulative training seconds", ylabel="Joint physical VAL L2 score (%)",
           title="Solver confirmation: observed checkpoints (lower is better)")
    ax.grid(alpha=.18)
    ax.legend(fontsize=8)
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    return {"generated": True, "file": filename.name, "policy": "Measured scatter points only; no interpolation"}


def write(filename, text):
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(filename)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("notes/geobc_solver_confirmation_plan.json"))
    parser.add_argument("--jobs-plan", type=Path, help="Optional explicit coordinator job mapping; never infer run directories")
    parser.add_argument("--output", type=Path, default=Path("results/solver_confirmation"))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--check", action="store_true", help="Build a JSON-only snapshot without writing artifacts or importing matplotlib")
    args = parser.parse_args(argv)
    report = build_report(args.plan, args.jobs_plan)
    if args.check:
        print(json.dumps({"schema": report["schema"], "completed_runs": report["completed_runs"],
                          "planned_runs": report["planned_runs"], "candidates_frozen": report["candidates_frozen"],
                          "common_input_and_scaler_match": report["common_input_and_scaler_match"],
                          "scientific_arrays_read": 0, "weights_read": 0, "GPU_work": False}, ensure_ascii=False))
        return report
    output = path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report["figure"] = {"generated": False, "reason": "--no-plot"}
    if not args.no_plot:
        try:
            report["figure"] = plot(report, output / "precision_vs_time.png")
        except ImportError as error:
            report["figure"] = {"generated": False, "reason": f"CPU matplotlib unavailable: {error}"}
    write(output / "metrics.json", json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    write(output / "BRIEF.md", brief(report))
    print(json.dumps({"output": str(output), "completed_runs": report["completed_runs"],
                      "planned_runs": report["planned_runs"], "figure": report["figure"]}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
