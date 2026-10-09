"""Summarize completed frozen final-CAD receipts without loading field arrays.

Primary results are the two predetermined budget-endpoint checkpoints. The
validation-selected checkpoints remain secondary JSON readouts. Pending runs
never borrow intermediate development-validation or partial final metrics.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MAX_JSON_BYTES = 32 * 1024 * 1024
ROLES = ("endpoint_primary", "validation_best_secondary")
SEEDS = (2026, 2027)
LABELS = {"fno": "FNO", "mlp": "MLP", "hashgrid": "HashGrid", "transolver": "Transolver",
          "transolver_pp": "Transolver++", "ts3": "Transolver3"}
RULE_SCOPE_KEYS = ("existing_split", "confirmation_plan", "selection_cache", "prepared_data",
                   "eval_cache", "known_loads", "combination_loads", "routes", "input_recipe", "rules")
RELATIVE_FIELDS = ("u_relative_l2", "stress_relative_l2", "stress_hotspot_relative_l2")
PHYSICAL_FIELDS = ("u_vector_error_RMS_mm", "u_max_vector_error_mm", "stress_RMSE_MPa", "stress_MAE_MPa")


def host_path(value):
    text = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", text):
        text = text[5].upper() + ":/" + text[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", text):
        text = "/mnt/" + text[0].lower() + "/" + text[3:]
    result = Path(text)
    return (result if result.is_absolute() else ROOT / result).resolve()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read_json(filename, optional=False):
    filename = host_path(filename)
    if optional and not filename.is_file():
        return {}, {"path": str(filename), "present": False}
    if not filename.is_file() or filename.suffix != ".json" or filename.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"Expected a bounded small JSON receipt: {filename}")
    raw = filename.read_bytes()
    value = json.loads(raw.decode("utf-8-sig"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {filename}")
    return value, {"path": str(filename), "present": True, "bytes": len(raw),
                   "sha256": hashlib.sha256(raw).hexdigest()}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def nonnegative(value):
    return finite(value) and value >= 0


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def canonical_method(value):
    return {"transolverpp": "transolver_pp", "transolver++": "transolver_pp",
            "transolver3": "ts3", "hash": "hashgrid"}.get(value, value)


def route_key(route):
    return canonical_method(route["method"]), route["seed"], route["checkpoint_role"]


def declared_routes(plan):
    if plan.get("known_loads") != ["ver", "hor", "tor"] or plan.get("combination_loads") != ["dia"]:
        raise ValueError("Expected known3 and separate dia final-evaluation scopes")
    routes = plan.get("routes", [])
    keys = [route_key(route) for route in routes]
    methods = list(dict.fromkeys(key[0] for key in keys))
    if len(methods) > 2 or len(keys) != len(set(keys)) or len(keys) > 8:
        raise ValueError("Final report is bounded by two methods x two seeds x two roles")
    if any(method not in LABELS or seed not in SEEDS or role not in ROLES for method, seed, role in keys):
        raise ValueError("Unknown declared solver, seed, or checkpoint role")
    if plan.get("final_rules_frozen") is True and set(keys) != {(method, seed, role) for method in methods for seed in SEEDS for role in ROLES}:
        raise ValueError("Frozen methods must retain both seeds and both checkpoint roles")
    return routes, methods


def validate_group(group, keys, scope):
    per_case = group.get("per_case", {})
    if set(per_case) != set(keys) or group.get("cases") != len(keys):
        raise ValueError(f"Completed {scope} case membership/count differs")
    for key, item in per_case.items():
        nodes = item.get("nodes")
        if not isinstance(nodes, int) or isinstance(nodes, bool) or nodes <= 0:
            raise ValueError(f"Invalid all-node count: {scope}/{key}")
        if not all(nonnegative(item.get(field)) for field in RELATIVE_FIELDS + PHYSICAL_FIELDS):
            raise ValueError(f"Missing/nonfinite physical metric: {scope}/{key}")
        if item.get("stress_hotspot_nodes") != max(1, math.ceil(.01 * nodes)):
            raise ValueError(f"Hotspot count differs from source-|stress| top ceil(1% N): {key}")
    if group.get("nodes") != sum(item["nodes"] for item in per_case.values()):
        raise ValueError(f"Completed {scope} node totals differ")
    checks = {"case_mean_u_relative_l2": sum(item["u_relative_l2"] for item in per_case.values()) / len(keys),
              "case_mean_stress_relative_l2": sum(item["stress_relative_l2"] for item in per_case.values()) / len(keys),
              "case_mean_stress_hotspot_relative_l2": sum(item["stress_hotspot_relative_l2"] for item in per_case.values()) / len(keys),
              "worst_case_u_relative_l2": max(item["u_relative_l2"] for item in per_case.values()),
              "worst_case_stress_relative_l2": max(item["stress_relative_l2"] for item in per_case.values()),
              "worst_case_stress_hotspot_relative_l2": max(item["stress_hotspot_relative_l2"] for item in per_case.values())}
    for field, expected in checks.items():
        if not close(group.get(field), expected):
            raise ValueError(f"Stored and per-case {scope}/{field} disagree")
    for field, metric in (("worst_case_u_key", "u_relative_l2"),
                          ("worst_case_stress_key", "stress_relative_l2"),
                          ("worst_case_stress_hotspot_key", "stress_hotspot_relative_l2")):
        if group.get(field) not in per_case or not close(per_case[group[field]][metric], max(item[metric] for item in per_case.values())):
            raise ValueError(f"Worst-case identity differs: {scope}/{field}")
    return dict(group)


def primary_range(routes, scope):
    if {row["seed"] for row in routes} != set(SEEDS) or len(routes) != 2:
        raise ValueError("Two predetermined seeds are required for descriptive ranges")
    result = {"seeds": list(SEEDS), "per_seed": {}, "case_mean_ranges": {}, "two_seed_worst_cases": {}}
    for row in routes:
        result["per_seed"][str(row["seed"])] = row[scope]
    for field in ("case_mean_u_relative_l2", "case_mean_stress_relative_l2", "case_mean_stress_hotspot_relative_l2"):
        values = [row[scope][field] for row in routes]
        result["case_mean_ranges"][field] = {"min": min(values), "max": max(values)}
    for metric, value_field, id_field in (("u", "worst_case_u_relative_l2", "worst_case_u_key"),
                                         ("stress", "worst_case_stress_relative_l2", "worst_case_stress_key"),
                                         ("stress_hotspot", "worst_case_stress_hotspot_relative_l2", "worst_case_stress_hotspot_key")):
        largest = max(row[scope][value_field] for row in routes)
        result["two_seed_worst_cases"][metric] = {"relative_l2": largest,
            "identities": [{"seed": row["seed"], "case_key": row[scope][id_field]}
                           for row in routes if close(row[scope][value_field], largest)]}
    result["interpretation"] = "Two fixed seeds and the same CAD cases; descriptive ranges, not independent-sample significance or final-data selection"
    return result


def validate_completed(plan, summary, saved_config):
    if plan.get("final_rules_frozen") is not True or not plan.get("routes"):
        raise ValueError("Completed final metrics require a nonempty frozen final plan")
    config = summary.get("config", {})
    if saved_config != config:
        raise ValueError("Completed summary/config identity differs")
    contract = config.get("contract", {})
    inputs = contract.get("input_contract", {})
    rule_sha = digest({key: plan.get(key) for key in RULE_SCOPE_KEYS})
    gate = inputs.get("confirmation_gate", {})
    if (contract.get("schema") != "geobc_solver_final_eval_v1" or inputs.get("final_rules_frozen") is not True
            or inputs.get("final_rules_sha256") != rule_sha or gate.get("all_six_completed") is not True
            or gate.get("invalid_runs") or len(gate.get("runs", [])) != 6):
        raise ValueError("Completed final rule/confirmation contract differs")
    if any(summary.get(key) != expected for key, expected in (("training_updates", 0), ("scaler_fits", 0),
            ("label_read_after_prediction", True), ("final_selection_performed", False))):
        raise ValueError("Final receipt must record zero training/fits, prediction-first labels, and no final-data selection")
    known, combination = inputs.get("known_keys", []), inputs.get("combination_keys", [])
    if (len(known) != 15 or len(combination) != 5 or len(set(known + combination)) != 20
            or summary.get("known_keys") != known or summary.get("combination_keys") != combination
            or any(key.split("/")[-1] not in ("ver", "hor", "tor") for key in known)
            or any(key.split("/")[-1] != "dia" for key in combination)
            or {key.split("/")[0] for key in known} != {key.split("/")[0] for key in combination}):
        raise ValueError("Completed final15 known/final5 dia scopes differ")
    declared = {route_key(route): route for route in plan["routes"]}
    expected = {route_key(route): route for route in contract.get("routes", [])}
    rows = summary.get("routes", [])
    if (set(expected) != set(declared) or len(rows) != len(expected)
            or {route_key(row) for row in rows} != set(expected)):
        raise ValueError("Completed route/seed/role inventory differs")
    for row in rows:
        identity = expected[route_key(row)]
        if any(row.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Completed route metadata differs: {row.get('route_id')}")
        specified = declared[route_key(row)]
        if (host_path(row["checkpoint"]) != host_path(specified["checkpoint"])
                or host_path(row["confirmation_run"]) != host_path(specified["confirmation_run"])):
            raise ValueError("Completed checkpoint/confirmation path differs from frozen route")
        for field in ("selected_step", "checkpoint_stat", "budget_origin", "parameter_counts"):
            if field in specified and specified[field] != row[field]:
                # Paths embedded in origin may be host-specific; compare its path separately.
                if field != "budget_origin" or specified[field].get("phase") != row[field].get("phase") or specified[field].get("step") != row[field].get("step") or host_path(specified[field]["checkpoint"]) != host_path(row[field]["checkpoint"]):
                    raise ValueError(f"Completed frozen {field} differs")
        validate_group(row["known"], known, "known")
        validate_group(row["combination"], combination, "combination")
        if not isinstance(row.get("deduplicated"), bool):
            raise ValueError("Explicit checkpoint deduplication receipt missing")
    unique_ids = {row["evaluation_id"] for row in rows}
    if (summary.get("requested_routes") != len(rows) or summary.get("unique_evaluations") != len(unique_ids)
            or summary.get("unique_case_predictions") != 20 * len(unique_ids)
            or summary.get("deduplicated_case_predictions") != 20 * (len(rows) - len(unique_ids))):
        raise ValueError("Unique/deduplicated checkpoint and prediction counts differ")
    return contract, rows


def build_report(plan_path, run_path):
    plan, plan_id = read_json(plan_path)
    declared, methods = declared_routes(plan)
    run = host_path(run_path)
    summary, summary_id = read_json(run / "summary.json", optional=True)
    config, config_id = read_json(run / "config.json", optional=True)
    state, state_id = read_json(run / "status.json", optional=True)
    diagnostics, completed, rows, contract = [], False, [], None
    if summary.get("status") == "completed":
        try:
            contract, rows = validate_completed(plan, summary, config)
            completed = True
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as error:
            diagnostics.append(f"Completed summary not published: {type(error).__name__}: {error}")
            rows, contract = [], None
    status = ("completed" if completed else "invalid_completed_receipt" if summary.get("status") == "completed"
              else "not_frozen_not_executed" if plan.get("final_rules_frozen") is not True or not declared
              else "failed" if state.get("phase") == "failed" else "not_completed")
    candidates = []
    for method in methods:
        primary = sorted([row for row in rows if canonical_method(row["method"]) == method and row["checkpoint_role"] == ROLES[0]], key=lambda row: row["seed"])
        secondary = sorted([row for row in rows if canonical_method(row["method"]) == method and row["checkpoint_role"] == ROLES[1]], key=lambda row: row["seed"])
        candidates.append({"method": method, "label": LABELS[method], "endpoint_primary_routes": primary,
            "validation_best_secondary_routes": secondary,
            "known_primary": primary_range(primary, "known") if completed else None,
            "combination_primary": primary_range(primary, "combination") if completed else None})
    return {"schema": "geobc_solver_final_report_v1", "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "final_rules_frozen": plan.get("final_rules_frozen") is True,
        "sources": {"plan": plan_id, "summary": summary_id, "config": config_id, "status": state_id},
        "run_directory": str(run), "planned_routes": len(declared), "published_completed_routes": len(rows),
        "progress_only": {key: state.get(key) for key in ("phase", "elapsed_seconds", "method", "seed", "key", "committed_routes", "completed_cases", "error")},
        "methods": candidates, "routes": rows, "seed_summary_source": summary.get("seed_summary") if completed else None,
        "evaluation_contract": contract, "elapsed_seconds": summary.get("elapsed_seconds") if completed else None,
        "evaluation_costs": summary.get("costs") if completed else None, "peak_memory": summary.get("memory") if completed else None,
        "checkpoint_deduplication": {key: summary.get(key) for key in ("requested_routes", "unique_evaluations", "unique_case_predictions", "deduplicated_case_predictions")} if completed else None,
        "training_budgets_by_method_seed": [{"method": row["method"], "seed": row["seed"],
            "confirmation_budget": row["confirmation_budget"], "selected_step": row["selected_step"],
            "selected_checkpoint_costs": row["selected_checkpoint_costs"], "budget_origin": row["budget_origin"]}
            for row in rows if row["checkpoint_role"] == ROLES[0]],
        "scope": {"known": "Final5 CAD x ver/hor/tor (15 cases); primary nodal readout",
                  "combination": "Same final5 CAD x dia (5 cases); separate load-combination transfer readout",
                  "point_scope": "All original FEM nodes, physical decoded signed uXYZ and source signed scalar stress"},
        "metric_definitions": {"relative_l2": "sqrt(sum squared physical error / sum squared physical reference), without denominator floor; nonpositive reference energy fails evaluator",
            "units": {"displacement": "mm", "source_signed_stress": "MPa", "relative_l2": "fraction (figures/table multiply by100)"},
            "displacement_RMS": "sqrt(sum over nodes and xyz components of squared error / N), vector RMS in mm; not component RMSE",
            "stress_hotspot": "Source |stress| largest ceil(.01*N) nodes per case, then signed stress relative L2 within that fixed subset; no U hotspot measured",
            "case_mean": "Equal-case mean of relative L2; not pooled-node L2 or a mean signed field",
            "raw_zero": "A zero signed physical prediction has U/S/hotspot relative L2=1 (100%) for positive reference energy; this benchmark is not a qualification or selection threshold",
            "semantics": "Reported scalar is source signed stress; no tensor/constitutive/Von-Mises identity is inferred"},
        "cost_policy": "Confirmation cumulative training costs include measured parent+incremental costs and complete-pass overshoot; selected checkpoint cost and origin retained. Final cache/H2D, inference, decode, label read, metrics, checkpoint initialization and memory are separate JSON receipts. Summary evaluation costs deduplicate shared checkpoints.",
        "limitations": ["Two predetermined seeds on the same five CADs provide descriptive ranges, not statistical significance.",
            "Final cases do not select checkpoints, architecture, hyperparameters, or retraining; frozen routes are reported even when outcomes are negative.",
            "Known3 and dia remain separate; they do not establish unseen boundary types, continuous-query accuracy, boundary enforcement or elastic-kinematic consistency.",
            "FNO is the fixed node-splat/trilinear geometry-conditioned candidate, with its adapter/resolution limits; no field/BC capability follows from a nodal error plot.",
            "Secondary checkpoints were chosen by prior development-validation score, potentially including seed2026 screen history; they are not selected on final results."],
        "diagnostics": diagnostics, "report_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "access_receipt": {"small_JSON_only": True, "scientific_arrays_read": 0, "truth_members_read": 0,
                           "checkpoint_weights_read": 0, "models_constructed": 0, "GPU_work": False}}


def fmt(value, percent=False):
    return f"{value * (100 if percent else 1):.2f}" if finite(value) else "—"


def range_text(group, field):
    if group is None:
        return "—"
    bounds = group["case_mean_ranges"][field]
    return f"{fmt(bounds['min'], True)}–{fmt(bounds['max'], True)}"


def table(report, scope):
    lines = ["| 方法 | 两seed U case-mean范围 % | 两seed S case-mean范围 % | 两seed最差case U % | 两seed最差case S % |",
             "|---|---:|---:|---:|---:|"]
    for row in report["methods"]:
        group = row[scope]
        worst = group["two_seed_worst_cases"] if group else {}
        lines.append(f"| {row['label']} | {range_text(group, 'case_mean_u_relative_l2')} | {range_text(group, 'case_mean_stress_relative_l2')} | {fmt(worst.get('u', {}).get('relative_l2'), True)} | {fmt(worst.get('stress', {}).get('relative_l2'), True)} |")
    return lines


def brief(report):
    lines = ["# GeoBC 冻结最终评估", ""]
    if report["status"] != "completed":
        state = "规则尚未冻结、未执行最终评估" if report["status"] == "not_frozen_not_executed" else f"最终评估尚无有效完成收据（{report['status']}）"
        lines += [state + "。主/次指标留空，不引用中途评估或开发验证结果。", "",
                  f"规则冻结：{'是' if report['final_rules_frozen'] else '否'}；声明 routes：{report['planned_routes']}。"]
        if report["methods"]:
            lines += [""] + table(report, "known_primary")
    else:
        lines += ["主结果：预先固定的600秒累计训练预算端点，seed2026与2027分别报告。以下范围来自两个seed的各自case-mean；最差值跨两seed和所有该组case。", "",
                  "**已知载荷：未见final5 CAD × ver/hor/tor，共15case。**", ""] + table(report, "known_primary")
        lines += ["", "**组合载荷：同final5 CAD × dia，共5case，独立于上表。**", ""] + table(report, "combination_primary")
        lines += ["", "两seed不构成统计显著性证据，最终评估不用于选赢家或反馈训练。U保留signed XYZ，S保留源signed scalar；误差在全原始节点物理解码后计算，单位为mm / MPa。热点仅源|stress|最高1%节点，worst case ID、物理RMS/MAE/max、每case指标见[metrics.json](metrics.json)。", "",
                  "共同VAL-best作为次结果仅列入JSON，保留checkpoint角色、selected step与screen/confirmation来源。实际累计训练秒（parent / incremental / overshoot）、初始化、缓存加载、推断/评估时间、显存/参数和共享checkpoint去重计数均分开记录；不将训练预算和评估开销混为同一成本。"]
        if report.get("figure", {}).get("generated"):
            lines += ["", "![冻结端点的全节点signed-field误差](signed_field_error_comparison.png)",
                      "每点是一个case和固定seed，箱体仅描述这些值；100%虚线是raw-zero基准，不表示连续场或边界条件性能。"]
    if report["diagnostics"]:
        lines += ["", "收据身份或指标检查未通过，未发布完成数值；原因见metrics.json。"]
    return "\n".join(lines) + "\n"


def plot(report, filename):
    if report["status"] != "completed" or not report["methods"]:
        return {"generated": False, "reason": "No valid completed frozen final evaluation; no synthetic or partial plot"}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.3), constrained_layout=True)
    for row_index, (scope, description) in enumerate((("known", "Known loads (15 cases / seed)"), ("combination", "Diagonal load (5 cases / seed)"))):
        for column, (field, label) in enumerate((("u_relative_l2", "Signed displacement XYZ"), ("stress_relative_l2", "Source signed stress"))):
            ax = axes[row_index][column]
            values = []
            for index, candidate in enumerate(report["methods"]):
                routes = candidate["endpoint_primary_routes"]
                all_values = [100 * item[field] for route in routes for item in route[scope]["per_case"].values()]
                values.append(all_values)
                for route in routes:
                    ordered = sorted(route[scope]["per_case"].items())
                    offset = -.10 if route["seed"] == 2026 else .10
                    positions = [index + 1 + offset + .055 * (2 * rank / max(1, len(ordered)-1) - 1) for rank in range(len(ordered))]
                    ax.scatter(positions, [100 * item[field] for _, item in ordered], s=18, alpha=.65,
                               marker="o" if route["seed"] == 2026 else "^", color=f"C{index}")
            ax.boxplot(values, positions=list(range(1, len(values)+1)), widths=.5, showfliers=False,
                       medianprops={"color": "black"}, boxprops={"color": "0.45"},
                       whiskerprops={"color": "0.55"}, capprops={"color": "0.55"})
            ax.axhline(100, color="0.6", linestyle="--", linewidth=.9)
            ax.set_xticks(list(range(1, len(values)+1)), [item["label"] for item in report["methods"]])
            ax.set(title=f"{description}\n{label}", ylabel="Physical relative L2 error (%)")
            ax.grid(axis="y", alpha=.16)
            ax.set_ylim(bottom=0)
    fig.suptitle("Frozen budget endpoints: errors of signed nodal fields", fontsize=12)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", color="0.3", label="Seed2026"),
                        Line2D([], [], marker="^", linestyle="none", color="0.3", label="Seed2027")],
               loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(filename, dpi=170)
    plt.close(fig)
    return {"generated": True, "file": filename.name,
            "scope": "Primary endpoint; known and dia separate; per-case error scalars only, no field render",
            "policy": "Fixed two-seed cases, no final-data ranking, interpolation, or significance claim"}


def write(filename, text):
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(filename)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("notes/geobc_solver_final_eval_plan.json"))
    parser.add_argument("--run", type=Path, default=Path("runs/geobc_exploration/solver_final_validation"))
    parser.add_argument("--output", type=Path, default=Path("results/solver_final_eval"))
    parser.add_argument("--check", "--source-check", dest="check", action="store_true", help="Stdlib-only AST/JSON check; no report writes or plotting imports")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=__file__)
    report = build_report(args.plan, args.run)
    if args.check:
        print(json.dumps({"status": report["status"], "final_rules_frozen": report["final_rules_frozen"],
                          "planned_routes": report["planned_routes"], "published_completed_routes": report["published_completed_routes"],
                          "diagnostics": report["diagnostics"], "source_sha256": report["report_tool_sha256"],
                          **report["access_receipt"]}, ensure_ascii=False, allow_nan=False))
        return report
    output = host_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report["figure"] = {"generated": False, "reason": "--no-plot"}
    if not args.no_plot:
        try:
            report["figure"] = plot(report, output / "signed_field_error_comparison.png")
        except ImportError as error:
            report["figure"] = {"generated": False, "reason": f"CPU matplotlib unavailable: {error}"}
    write(output / "metrics.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write(output / "BRIEF.md", brief(report))
    print(json.dumps({"status": report["status"], "output": str(output), "figure": report["figure"]}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
