"""CPU small-JSON report for one fixed480 all-node mesh solver screen.

No data arrays, checkpoints, trainers or model imports. Historical fixed480
metrics are context, not strict capacity or topology-only causal controls.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

if __package__:
    from .summarize_deeponet_screen import training_view
    from .summarize_fno_response import finite, fmt, metric_view, path, read_json, write
else:
    from summarize_deeponet_screen import training_view
    from summarize_fno_response import finite, fmt, metric_view, path, read_json, write

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = "runs/geobc_exploration/mesh_multiscale_seed2026_480"
DEFAULT_DIA = "runs/geobc_exploration/mesh_multiscale_development_dia480"
DEFAULT_CACHE = "data/geobc_multiscale_mesh_inputs45"
HISTORY = {
    "fno": ("FNO direct4", "runs/geobc_exploration/solver_screen_fno_480"),
    "mlp": ("MLP direct4", "runs/geobc_exploration/solver_screen_mlp_480"),
    "gino": ("GINO", "runs/geobc_exploration/gino_screen_seed2026_480"),
    "fno_response": ("FNO force-free u9", "runs/geobc_exploration/fno_response_linear_basis_seed2026_480"),
    "deeponet": ("BC DeepONet u9", "runs/geobc_exploration/deeponet_bc_seed2026_480"),
}
U = "case_mean_u_relative_l2"
S = "case_mean_stress_relative_l2"


def build_report(args):
    mesh = training_view("mesh", args.run, historical=True, label="Multiscale mesh")
    summary, summary_source = read_json(path(args.run)/"summary.json", mesh["diagnostics"])
    if mesh["status"] == "completed" and summary.get("kind") != "multiscale_mesh_screen":
        mesh["status"] = "invalid_completed_kind"
    cache_diagnostics = []
    cache, cache_source = read_json(path(args.cache)/"manifest.json", cache_diagnostics)
    dia_diagnostics = []
    dia, dia_source = read_json(path(args.dia_run)/"summary.json", dia_diagnostics)
    route = dia.get("routes", {}).get("mesh", {})
    dia_metrics = metric_view(route.get("development_dia"), 5, dia_diagnostics, "mesh DEV5 dia")
    dia_complete = (dia.get("status") == "completed"
                    and dia.get("kind") == "multiscale_mesh_development_dia_eval"
                    and dia.get("training_updates") == dia.get("scaler_fits") == 0
                    and dia_metrics is not None)
    cache_complete = (cache.get("status") == "completed" and cache.get("kind") == "multiscale_mesh_input")
    first = summary.get("first_actual_check", {})
    mesh_complete = mesh["status"] == "completed" and first.get("passed") is True
    historical = {name: training_view(name, directory, historical=True, label=label)
                  for name, (label, directory) in HISTORY.items()}
    fno = historical["fno"]
    comparison = None
    if mesh_complete and fno["status"] == "completed":
        current, reference = mesh["validation"], fno["validation"]
        comparison = {
            "reference": "Historical direct4 FNO at fixed480, same case order/labels; different architecture/counts/input graph",
            "mean_U_gain_percentage_points": 100*(reference[U]-current[U]),
            "mean_S_gain_percentage_points": 100*(reference[S]-current[S]),
            "train_seconds_ratio": mesh["costs"]["train_seconds"]/fno["costs"]["train_seconds"],
            "both_field_means_lower": current[U] < reference[U] and current[S] < reference[S],
            "some_field_mean_lower": current[U] < reference[U] or current[S] < reference[S],
            "both_worst_fields_lower": current["worst_case_u_relative_l2"] < reference["worst_case_u_relative_l2"]
                and current["worst_case_stress_relative_l2"] < reference["worst_case_stress_relative_l2"],
            "capacity_or_topology_only_causal_control": False,
        }
    return {
        "status": "completed" if mesh_complete and dia_complete and cache_complete else "incomplete",
        "mesh": mesh, "development_dia": {"status": "completed" if dia_complete else "incomplete",
            "metrics": dia_metrics, "summary": dia, "source": dia_source, "diagnostics": dia_diagnostics},
        "input_cache": {"status": "completed" if cache_complete else "incomplete",
            "manifest": cache, "source": cache_source, "diagnostics": cache_diagnostics},
        "historical480_context": historical, "comparison": comparison,
        "summary_source": summary_source,
        "scope": {
            "primary": "One fresh seed2026 endpoint480 / four complete passes / TRAIN40 known3 / previously explored DEV5",
            "physical_units": {"U": "mm signed vector", "S": "MPa original signed scalar"},
            "selection": "Endpoint480 primary; no dia/best/seed/budget selection in this reporter",
            "historical_context": "No oldfinal/600-second checkpoint mixture; parameter layouts and native input interaction differ",
            "limits": "No statistical significance, constitutive stress recovery, native continuous derivative, PDE or strict BC qualification",
            "old_final_or_new_sealed_test_access": False,
            "reporter_reads": "Small completed JSON only; no scientific arrays, weights, solvers or trainers",
        },
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def brief(report):
    mesh, dia = report["mesh"], report["development_dia"]["metrics"]
    known = mesh["validation"]
    lines = ["# 多尺度网格求解器：固定480初筛", "",
        "一个新mesh家族，原25含实际载荷、direct4、宽度32及四个learned V-cycle。完整真实节点/边与粗到细支持，不接Hash。", "",
        "| 固定端点 | 位移 mean / worst % | signed stress mean / worst / hotspot % |", "|---|---:|---:|"]
    for label, row in (("known15", known), ("DEV5 dia", dia)):
        lines.append(f"| {label} | {fmt(100*row[U])} / {fmt(100*row['worst_case_u_relative_l2'])} | "
            f"{fmt(100*row[S])} / {fmt(100*row['worst_case_stress_relative_l2'])} / {fmt(100*row['case_mean_stress_hotspot_relative_l2'])} |")
    lines += ["", f"实际训练{mesh['costs']['train_seconds']:.2f}秒，raw wall {mesh['elapsed_seconds']:.2f}秒；参数{mesh['parameter_counts'].get('total')}。缓存准备、读入/H2D、推断与peak分列于JSON，重叠计时不相加。", ""]
    comparison = report["comparison"]
    if comparison:
        if comparison["both_field_means_lower"]:
            judgment = "两场均值均低于历史FNO480，具有继续固定成本／第二seed确认的精度机会。"
        elif comparison["some_field_mean_lower"]:
            judgment = "仅一场均值改善，需要同时权衡另一场、最差样本和实际成本。"
        else:
            judgment = "两场均值未优于历史FNO480，本配置当前没有精度选型优势。"
        lines += [judgment + " 四pass不是收敛或整个家族有效性的结论。", ""]
    lines += ["历史同480 known15，仅作上下文：", "",
        "| 方法 | U / S mean % | 实际训练秒 |", "|---|---:|---:|"]
    for row in report["historical480_context"].values():
        if row["status"] == "completed":
            lines.append(f"| {row['label']} | {fmt(100*row['validation'][U])} / {fmt(100*row['validation'][S])} | {row['costs']['train_seconds']:.2f} |")
    lines += ["", "这些结构的参数、局部交互与输入图不同，不是同容量或拓扑单因素因果实验。完整曲线、最差ID、首正式全案例梯度收据、实际层级节点数/边数/P-R支持与物理单位见[metrics.json](metrics.json)。", "",
        "support-reference角色不等于节点Dirichlet，图/mass不是刚度矩阵。DEVdia已探索；旧final和新sealed test保持关闭。本初筛不新增P2监督或应变GT声明。"]
    if report.get("figure", {}).get("generated"):
        lines += ["", "![实际固定端点精度与成本](comparison.png)"]
    return "\n".join(lines)+"\n"


def plot(report, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    active = [name for name in ("fno", "mlp", "gino")
              if report["historical480_context"][name]["status"] == "completed"]
    rows = [report["mesh"]] + [report["historical480_context"][name] for name in active]
    labels = ["Mesh"] + [name.upper() for name in active]
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.8), constrained_layout=True)
    for axis, field, title in ((axes[0], U, "Known15 U mean"), (axes[1], S, "Known15 signed S mean")):
        bars = axis.bar(labels, [100*row["validation"][field] for row in rows], color=["#66aa99"]+["#4477aa"]*(len(rows)-1))
        for bar in bars:
            axis.annotate(f"{bar.get_height():.1f}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                ha="center", va="bottom", fontsize=8, xytext=(0, 3), textcoords="offset points")
        axis.set(title=title, ylabel="Physical relative L2 (%)"); axis.margins(y=.18); axis.grid(axis="y", alpha=.16)
    bars = axes[2].bar(labels, [row["costs"]["train_seconds"] for row in rows], color=["#66aa99"]+["#4477aa"]*(len(rows)-1))
    for bar in bars:
        axes[2].annotate(f"{bar.get_height():.1f}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
            ha="center", va="bottom", fontsize=8, xytext=(0, 3), textcoords="offset points")
    axes[2].set(title="480 complete-case cost", ylabel="Actual training seconds"); axes[2].margins(y=.18); axes[2].grid(axis="y", alpha=.16)
    fig.suptitle("Actual endpoint480; one seed; different architectures/counts/native interactions", fontsize=10)
    fig.savefig(filename, dpi=170, bbox_inches="tight"); plt.close(fig)
    return {"generated": True, "file": filename.name, "policy": "Actual completed bars only; no interpolation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--dia-run", default=DEFAULT_DIA)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--output", default="results/multiscale_mesh")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    report = build_report(args)
    if report["status"] != "completed":
        raise SystemExit("Formal mesh screen/cache/DEVdia incomplete; no report or figure exported")
    output = path(args.output); output.mkdir(parents=True, exist_ok=True)
    report["figure"] = {"generated": False, "reason": "--no-plot"}
    if not args.no_plot:
        report["figure"] = plot(report, output/"comparison.png")
    write(output/"metrics.json", json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    (output/"BRIEF.md").write_text(brief(report), encoding="utf8")
    print(json.dumps({"status": report["status"], "output": str(output), "figure": report["figure"]}))


if __name__ == "__main__":
    main()
