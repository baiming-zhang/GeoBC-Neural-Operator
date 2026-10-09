"""CPU report for the completed GINO and historical six-family 480 screens.

Author metadata-free syntax receipt (once):
  python tools/summarize_gino_screen.py --source-check
After the actual GINO summary is completed (root runs once, then views figure):
  python tools/summarize_gino_screen.py --plan notes/geobc_gino_training_screen_plan.json

Reads small plan/config/summary JSON only. No trainer/model/array/checkpoint
imports or GPU work. Matplotlib is lazy and plots real positive-time dots only.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_gino_training_screen_plan.json"
FIELDS = ("case_mean_u_relative_l2", "worst_case_u_relative_l2",
          "case_mean_stress_relative_l2", "worst_case_stress_relative_l2")
METHODS = ("mlp", "hashgrid", "transolver", "transolver_pp", "ts3", "fno")
LABELS = {"mlp": "MLP", "hashgrid": "Hash", "transolver": "Transolver",
          "transolver_pp": "Transolver++", "ts3": "Transolver3", "fno": "FNO", "gino": "GINO"}


def local(value):
    item = Path(value)
    return item.resolve() if item.is_absolute() else (ROOT/item).resolve()


def sha(item):
    return hashlib.sha256(item.read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_small(item):
    require(item.is_file(), f"Small receipt not yet available: {item}")
    require(item.stat().st_size <= 32*1024*1024, f"Not a small JSON receipt: {item}")
    data = item.read_bytes()
    def invalid(value):
        raise ValueError(f"Nonfinite JSON token {value}")
    result = json.loads(data.decode("utf-8-sig"), parse_constant=invalid)
    require(isinstance(result, dict), "Expected a JSON object")
    return result, {"path": str(item), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def atomic(item, text):
    item.parent.mkdir(parents=True, exist_ok=True)
    temporary = item.with_suffix(item.suffix+".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(item)


def checked_point(record, step):
    require(isinstance(record, dict) and record.get("step") == step, f"Endpoint {step} missing")
    val = record["validation"]
    require(val.get("cases") == 15 and all(finite(val.get(k)) and val[k] >= 0 for k in FIELDS), "Physical VAL15 schema differs")
    score = .5*(val[FIELDS[0]]+val[FIELDS[2]])
    require(finite(record.get("selection_score")) and abs(record["selection_score"]-score) <= 1e-12*max(1., score), "Joint score differs from its same-checkpoint U/S")
    seconds = record["costs"]["train_seconds"]
    require(finite(seconds) and seconds >= 0 and (step == 0 or seconds > 0), "Endpoint has no measured positive train time")
    return {k: record[k] for k in ("step", "validation", "train_readout", "selection_score", "costs", "elapsed_seconds", "memory") if k in record}


def row(method, summary_path, require_passes=False):
    summary, identity = read_small(summary_path)
    require(summary.get("status") == "completed" and summary.get("method") == method
            and summary.get("endpoint_step") == 480, f"{method}: formal report requires completed endpoint480")
    config, config_identity = read_small(summary_path.parent/"config.json")
    require(config == summary["config"], f"{method}: summary/config mismatch")
    exposure = summary["case_exposure_counts"]
    require(len(exposure) == 120 and all(isinstance(v, int) and not isinstance(v, bool) and v == 4 for v in exposure.values())
            and summary["costs"]["train_case_exposures"] == 480, f"{method}: four complete TRAIN120 exposures not reached")
    expected = (0, 120, 240, 360, 480) if method == "gino" else (0, 120, 240, 480)
    points = {p["step"]: p for p in summary["endpoint_metrics"]}
    require(set(points) == set(expected), f"{method}: expected real endpoint receipts differ")
    observed = [checked_point(points[step], step) for step in expected]
    final = checked_point(summary["final_metrics"], 480)
    best = checked_point(summary["best_metrics"], summary["best_step"])
    require(final == observed[-1] and best == next(p for p in observed if p["step"] == summary["best_step"]), "Best/final endpoint values disagree")
    minimum = min(observed, key=lambda p: (p["selection_score"], p["step"]))
    require(best["step"] == minimum["step"], "Declared joint VAL best differs from observed endpoints")
    require(all(b["costs"]["train_seconds"] > a["costs"]["train_seconds"] for a, b in zip(observed, observed[1:])), "Actual train times must increase")
    passes = summary.get("pass_costs", [])
    if require_passes:
        require(len(passes) == 4 and [p["step"] for p in passes] == [120, 240, 360, 480]
                and [p["pass"] for p in passes] == [1, 2, 3, 4]
                and all(p["case_updates"] == 120 for p in passes), "Four GINO pass cost receipts required")
        for item in passes:
            point = next(p for p in observed if p["step"] == item["step"])
            require(abs(item["cumulative_train_seconds"]-point["costs"]["train_seconds"]) <= 1e-9,
                    "Pass cost differs from its measured physical endpoint")
    params = summary["parameter_counts"]
    parameter_view = {"raw_numel_counts": params, "real_scalar_DOF": None, "complex_entry_count": None,
                      "definition": "Stored parameter numel receipt; no model or weights inspected by this report"}
    if method == "gino":
        model = config["model"]
        require(model["FNO_modes"] == [24, 24, 22] and model["FNO_hidden"] == 16 and model["FNO_layers"] == 4, "Single frozen matched-frequency candidate changed")
        complex_entries = 4*16*16*24*24*(22//2+1)
        parameter_view.update(complex_entry_count=complex_entries, real_scalar_DOF=params["total"]+complex_entries,
            definition="Actual raw total from completed summary; official four dense16x16x24x24x12 cfloat tensors add one real scalar each for real-DOF conversion; source arithmetic, not another model execution")
    elif method == "fno":
        parameter_view.update(complex_entry_count=0, real_scalar_DOF=params["total"],
            definition="Old FNO stores spectral real/imag as separate real Parameters; raw numel already counts real scalar DOF")
    return {"method": method, "label": LABELS[method], "status": "completed", "endpoint_step": 480,
            "source": identity, "config_source": config_identity, "case_order_sha256": summary["case_order_sha256"],
            "primary": final, "best_secondary": best, "best_step": summary["best_step"],
            "best_label": "未训练基线" if summary["best_step"] == 0 else "共同VAL-score best（次结果）",
            "initial": observed[0], "observed_endpoints": observed, "pass_costs": passes,
            "curve_points": [{"step": p["step"], "train_seconds": p["costs"]["train_seconds"], "joint_validation_score": p["selection_score"]} for p in observed if p["step"] > 0],
            "costs": summary["costs"], "elapsed_seconds": summary["elapsed_seconds"], "memory": summary["memory"],
            "parameter_counts": parameter_view, "optimizer": "Adam foreach=True/fused=False" if method == "gino" else "Prior screen Adam fused=True",
            "config": config, "radius_cache": summary.get("radius_cache"),
            "first_update_gradient_check": summary.get("first_update_gradient_check"),
            "case_exposure_receipt": {"case_count": len(exposure), "updates": 480, "per_case": 4},
            "formal_gate_passed": True}


def conclusion(report):
    rows = {r["method"]: r for r in report["methods"]}
    g = rows["gino"]["primary"]["validation"]
    f = rows["fno"]["primary"]["validation"]
    seconds = rows["gino"]["costs"]["train_seconds"]
    f_seconds = rows["fno"]["costs"]["train_seconds"]
    both = g[FIELDS[0]] < f[FIELDS[0]] and g[FIELDS[2]] < f[FIELDS[2]]
    neither = g[FIELDS[0]] >= f[FIELDS[0]] and g[FIELDS[2]] >= f[FIELDS[2]]
    evidence = ("固定480端点，GINO的U/S均值均低于旧FNO；" if both else
                "固定480端点，GINO的U/S均值均未低于旧FNO；" if neither else
                "固定480端点，GINO与旧FNO的U/S均值互有高低；")
    evidence += f"实测可比训练时间为{seconds:.1f} vs {f_seconds:.1f}秒。"
    context = report['confirmation_cost_context']
    if seconds > max(r['primary']['costs']['train_seconds'] for r in context):
        evidence += "GINO480耗时已超过四个600秒确认端点；不能把多耗时后的更新数比较称作成本效率胜出。"
    if g[FIELDS[0]] >= 1 or g[FIELDS[2]] >= 1:
        evidence += "GINO至少一个均值未优于零场100%基准，尚未形成双场合格候选。"
    elif g[FIELDS[0]] >= .95 and g[FIELDS[2]] >= .95:
        evidence += "两均值接近零场基准，尚无明显有效性证据。"
    else:
        evidence += "单seed短筛只提供后续候选依据，不能据此选定最终solver。"
    scores = rows["gino"]["observed_endpoints"]
    if scores[-1]["selection_score"] < scores[1]["selection_score"]:
        evidence += "已观察训练端点仍下降，不将慢路线直接淘汰。"
    return evidence


def brief(report):
    lines = ["# GINO 与既有六家族：固定480更新初筛", "", conclusion(report), "",
             "| 方法 | 480端点 U mean / worst (%) | signed S mean / worst (%) | 可比训练秒 |",
             "|---|---:|---:|---:|"]
    for r in report["methods"]:
        v = r["primary"]["validation"]
        lines.append(f"| {r['label']} | {100*v[FIELDS[0]]:.2f} / {100*v[FIELDS[1]]:.2f} | {100*v[FIELDS[2]]:.2f} / {100*v[FIELDS[3]]:.2f} | {r['costs']['train_seconds']:.1f} |")
    g = next(r for r in report["methods"] if r["method"] == "gino")
    best = g["best_secondary"]; v = best["validation"]
    lines.extend(["", f"主结果固定480；GINO共同VAL-score best次结果是step{best['step']}（{g['best_label']}），同点U/S均值={100*v[FIELDS[0]]:.2f}%/{100*v[FIELDS[2]]:.2f}%，不替换端点或以某一场单独选模。初始0、GINO120/240/360/480与旧族真实端点均完整保留于[metrics.json](metrics.json)。", "",
        f"共同strict40TRAIN/5VAL×known3、PointNet32＋原25、TRAIN-only signed4/asinh scaler、seed2026和四次完整case遍历；零final/dia/新test使用。GINO train_seconds包括case/缓存读取与H2D，首建CSR/save另报，不加进图横轴；总wall={g['elapsed_seconds']:.1f}s。GINO原参数numel={g['parameter_counts']['raw_numel_counts']['total']:,}，complex条目按1计；源形状换算实DOF={g['parameter_counts']['real_scalar_DOF']:,}。旧FNO14,177,284按实数存储计；频带相当但lifting/skip/projection/GNO不同，非严格容量或单组件对照。GINO Adam foreach/非fused，旧族Adam fused，成本差异不能只归因网络。", "",
        "[实际精度-时间点图](error_cost.png)仅画正累计训练秒与joint VAL score散点，横轴log；不插值、不跨时间范围外推或捏造等时精度。每场最差case、hotspot、推理/显存、逐pass与CSR build/read/save成本见JSON。本轮不证明收敛、连续场/梯度、硬BC/PDE满足或工程可用；角色region是描述符而非节点Dirichlet。", ""])
    context = []
    for r in report['confirmation_cost_context']:
        p = r['primary']; val = p['validation']
        context.append(f"{r['label']}/{r['seed']}：{p['costs']['train_seconds']:.1f}s，U/S {100*val[FIELDS[0]]:.2f}/{100*val[FIELDS[2]]:.2f}%")
    lines.insert(-2, "成本背景为已完成FNO/MLP两seed的600累计秒预算端点（含真实超预算）："+'；'.join(context)+'。图中空心点单列，不称GINO480等600或按它们外推等时精度；两种预算视角分开。')
    return "\n".join(lines)


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.1), constrained_layout=True)
    colors = {}
    for r in report["methods"]:
        points = r["curve_points"]
        dots = ax.scatter([p["train_seconds"] for p in points], [p["joint_validation_score"] for p in points],
                          s=55 if r["method"] == "gino" else 30, marker="D" if r["method"] == "gino" else "o", label=r["label"])
        if r["method"] == "gino":
            for p in points:
                ax.annotate(str(p["step"]), (p["train_seconds"], p["joint_validation_score"]),
                            xytext=(5, -12), textcoords="offset points", fontsize=7,
                            color=dots.get_facecolors()[0])
        final = points[-1]
        colors[r['method']] = dots.get_facecolors()[0]
        ax.scatter([final["train_seconds"]], [final["joint_validation_score"]], marker="*", s=130,
                   color=dots.get_facecolors()[0], edgecolors="black", linewidths=.5)
    for r in report['confirmation_cost_context']:
        p = r['primary']; x, y = p['costs']['train_seconds'], p['selection_score']
        ax.scatter([x], [y], marker='^' if r['seed']==2026 else 'v', s=70,
                   facecolors='none', edgecolors=colors[r['method']], linewidths=1.4,
                   label=f"{r['label']}600s seed{r['seed']}")
    ax.axhline(1., color=".5", linestyle="--", linewidth=1, label="Zero-field joint baseline")
    ax.set(xscale="log", xlabel="Measured cumulative comparable training seconds (log)",
           ylabel="Joint physical VAL relative L2", title="Fixed 480 updates: observed points; stars = endpoints")
    ax.grid(alpha=.2); ax.legend(fontsize=8, ncol=2)
    fig.savefig(output/"error_cost.png", dpi=160); plt.close(fig)


def confirmation_context(scaler):
    result = []
    for method in ('fno', 'mlp'):
        for seed in (2026, 2027):
            summary, identity = read_small(local(f'runs/geobc_exploration/solver_confirm_{method}_seed{seed}_600s/summary.json'))
            require(summary.get('status')=='completed' and summary['method']==method and summary['seed']==seed
                    and summary['config']['scaler']==scaler, 'Completed shared-scaler confirmation required')
            budget = summary['budget']
            require(budget['mode']=='total_training_seconds' and budget['requested_seconds']==600
                    and budget['complete_pass'] is True, 'Expected already completed600s endpoint budget')
            point = checked_point(summary['final_metrics'], summary['endpoint_step'])
            require(abs(point['costs']['train_seconds']-budget['total_train_seconds']) <= 1e-9,
                    'Actual confirmation cost differs from budget receipt')
            result.append({'method':method,'label':LABELS[method],'seed':seed,'source':identity,
                           'primary':point,'budget':budget,'role':'Cost context only; no equal-time interpolation/extrapolation or checkpoint selection'})
    return result


def source_check(plan_path):
    receipt = local("notes/gino_screen_report_source_ready.json")
    require(not receipt.exists(), "Unique author AST receipt already exists")
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    result = {"status": "source_ready", "tool_sha256": sha(Path(__file__)), "plan_sha256": sha(local(plan_path)),
              "global_summarizer_sha256": sha(local('tools/summarize_geobc_exploration.py')),
              "AST_checks": 1, "scientific_array_reads": 0, "weights_or_models_read": 0, "GPU_initialized": False,
              "report_executed": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic(receipt, json.dumps(result, indent=2)+"\n")
    print(json.dumps(result)); return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN)
    parser.add_argument("--output", default="results/gino_screen")
    parser.add_argument("--source-check", action="store_true")
    args = parser.parse_args()
    if args.source_check:
        return source_check(args.plan)
    plan, identity = read_small(local(args.plan))
    require(plan["kind"] == "gino_training_screen" and plan["training_rules_frozen"] is True, "Frozen single screen plan required")
    rows = [row("gino", local(plan["suggested_output"])/"summary.json", True)]
    comparisons = plan["existing_screen_comparisons"]
    require(len(comparisons) == 6 and {p["method"] for p in comparisons} == set(METHODS), "Exact historical six screens required")
    rows.extend(row(p["method"], local(p["summary"])) for p in comparisons)
    require(len({r["case_order_sha256"] for r in rows}) == 1, "Same480 case-order witness differs")
    g = rows[0]
    for r in rows[1:]:
        require(r["config"]["scaler"] == g["config"]["scaler"] and r["config"]["case_order"] == g["config"]["case_order"], "Shared scaler/schedule differs")
    report = {"schema": "geobc_gino_screen_report_v1", "status": "completed", "plan": identity,
              "tool_sha256": sha(Path(__file__)), "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "formal_gate": "completed endpoint480;480 train_case_exposures;120 cases each4;five GINO endpoints/four pass costs;old six exact480;common case-order/scaler",
              "primary_rule": "Every method fixed480; U/S same checkpoint. Joint-VAL best secondary only, step0 explicitly untrained",
              "curve_rule": "Measured step>0 train_seconds dots only, log x; no interpolation/extrapolation/equal-cost claims",
              "frontend_cost_rule": "GINO first CSR build/save separate from comparable train time; cached CSR IO/H2D included. Totalwall/cache byte inventory retained",
              "methods": rows, "conclusion": None, "new_scientific_execution": False,
              "limits": plan["limitations"]}
    report['confirmation_cost_context'] = confirmation_context(g['config']['scaler'])
    cache, cache_identity = read_small(local('data/geobc_solver_selection/manifest.json'))
    require(cache['scaler']==g['config']['scaler'], 'Shared frontend metadata/scaler differs')
    report['frontend_context'] = {'shared_input_cache':{'source':cache_identity,
        'prepare_wall_seconds':cache['elapsed_seconds'],'definition':'Recorded one-time shared input/grid/scaler cache preparation; not added to any comparable train_seconds'},
        'GINO_radius_cache':g['radius_cache'], 'GINO_first_CSR_build_save_definition':'Separate recorded costs.radius_build_seconds/radius_save_seconds; cached read/H2D already included in train_seconds'}
    report["conclusion"] = conclusion(report)
    output = local(args.output); output.mkdir(parents=True, exist_ok=True)
    atomic(output/"metrics.json", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    atomic(output/"BRIEF.md", brief(report))
    plot(report, output)
    print(json.dumps({"status": "completed", "methods": 7, "output": str(output),
                      "gino_train_seconds": g["costs"]["train_seconds"], "gino_endpoint": 480}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
