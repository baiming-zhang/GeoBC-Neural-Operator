"""Read one small completed capability JSON and write a one-page report.

  python tools/summarize_solver_capability.py
  python tools/summarize_solver_capability.py --source PATH --output DIRECTORY

Standard library only. No scientific arrays, weights, model, GPU, or new test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = "runs/geobc_exploration/solver_continuous_capability_train101_479_ver/summary.json"
DEFAULT_OUTPUT = "results/solver_capability"


def local(value):
    result = Path(value)
    return result.resolve() if result.is_absolute() else (ROOT / result).resolve()


def atomic_write(target, text):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)


def percent(value):
    return "null" if value is None else f"{value*100:.3f}"


def summarize(source, output):
    content = source.read_bytes()
    summary = json.loads(content)
    if summary.get("status") != "completed" or summary.get("kind") != "solver_continuous_capability":
        raise ValueError("Expected the completed bounded capability summary")
    contract = summary["config"]["contract"]
    scope = contract["scope"]
    if (scope["case"] != "101_479/ver" or len(scope["selected_rows"]) != 32
            or len(scope["shared_faces"]) != 8 or scope["logical_queries"] != 504):
        raise ValueError("This report keeps the fixed TRAIN32/8/504 scope")
    results = summary.get("results") or []
    if results and {(r["method"], r["seed"]) for r in results} != {
        (m, s) for m in ("fno", "mlp") for s in (2026, 2027)
    }:
        raise ValueError("Both preselected methods and both seeds must remain in the report")
    # Copy scalar diagnostics faithfully, including signed cross and zero-ref nulls.
    routes = [{k: r[k] for k in ("route_id", "method", "seed", "step", "status", "common_nodal_full10", "native", "costs") if k in r}
              for r in sorted(results, key=lambda r: (r["method"], r["seed"]))]
    metrics = {"schema": "geobc_solver_capability_report_v1", "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "sha256": hashlib.sha256(content).hexdigest(),
                   "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "diagnostic_tool_sha256": contract["script_sha256"],
                   "plan_sha256": contract["plan_sha256"], "checkpoint_routes": contract["routes"]},
        "scope": {"case": scope["case"], "original_cell_rows": scope["selected_rows"],
                  "selected_cells": 32, "shared_faces": 8, "counts": summary["counts"]},
        "diagnostic": {k: summary[k] for k in ("status", "scientific_outcome", "elapsed_seconds", "checks", "costs", "memory", "native_FNO_stencil", "receipt") if k in summary},
        "routes": routes,
        "definitions": {
            "checkpoint_rule": "Four already locked endpoint_primary checkpoints; both seeds reported, no seed/checkpoint selection",
            "table_node_errors": "U vector/stress scalar relative L2 over248 unique original nodes; physical units mm/MPa",
            "table_P2": "Same128 positive Gauss points and same GT-node P2 strain energy denominator for common full10 and native FNO",
            "strain_RMS": "Quadrature weighted full symmetric3x3 Frobenius RMS; strain dimensionless; both shears count twice",
            "energy": "Physical mm^3 times dimensionless squared strain; E2=E1+Emid+signed_cross, sign retained",
            "P1_P2_denominators": "P1 uses GT-P1 energy, P2 uses GT-P2; exact zero reference remains null with reason",
            "midside_RMS": "Uniform displacement vector RMS over192 local midside incidences, repeated IDs retained",
            "native_value_errors": "All504 logical physical queries versus GT-node full10 values; differs from248 unique-node table scope",
            "cost_interpretation": "One ordered run, no warmup/repeat; first FNO timing includes cold execution. These timings are measured, not a controlled steady-state speed ranking",
            "C0": "Compatible nodal full10 and native trilinear grid values have structural C0 interfaces; this is not learned accuracy. Grid-face derivatives may jump",
            "native_MLP": "notimplemented because original25 offnode continuation is undefined; no invented BC flags",
            "scope_limits": "Fixed TRAIN101_479/ver local32 cells, not wholeCAD/final/independent offnode or solver strain GT; P1/P2 are GT-node-derived kinematic proxies",
            "BC_roles": "Support-reference DOF/load membership are not automatically nodal Dirichlet constraints; no BC repair or causal error claim"},
        "new_scientific_execution": False}
    lines = ["# 固定 TRAIN 局部 solver 能力诊断", "",
             f"唯一实际诊断 completed，{summary['elapsed_seconds']:.3f} 秒；固定 `101_479/ver` 的 32 cells / 8 faces / 504 logical queries（248 unique 原节点、128 Gauss）。四个已锁 endpoint 全部保留，两 seed 均报告。", "",
             "| Solver / seed（endpoint step） | 原节点 U / signed stress relative L2 (%) | P2 proxy relative L2：共同 full10 / 原生 (%) |",
             "|---|---:|---:|"]
    for r in routes:
        common = r.get("common_nodal_full10")
        if common is None:
            lines.append(f"| {r['method'].upper()} / {r['seed']} ({r['step']}) | 未执行或未定义 | 未执行或未定义 |")
            continue
        node = common["node_value_errors"]
        p2 = common["kinematics_proxy"]["P2"]["relative_L2_to_truth"]["value"]
        native = r.get("native") or {}
        native_p2 = native.get("P2_reference_strain_proxy", {}).get("relative_L2", {}).get("value")
        native_text = percent(native_p2) if native.get("status") == "completed" else "未实现" if native.get("status") == "notimplemented" else "不可用"
        lines.append(f"| {r['method'].upper()} / {r['seed']} ({r['step']}) | {percent(node['u']['relative_L2'])} / {percent(node['stress']['relative_L2'])} | {percent(p2)} / {native_text} |")
    if results:
        all_defined = all(r.get("common_nodal_full10") is not None for r in routes)
        if all_defined:
            by = {(r["method"], r["seed"]): r for r in routes}
            fno_node_better = all(by[("fno", s)]["common_nodal_full10"]["node_value_errors"][f]["relative_L2"]
                                  < by[("mlp", s)]["common_nodal_full10"]["node_value_errors"][f]["relative_L2"]
                                  for s in (2026, 2027) for f in ("u", "stress"))
            sentence = "此固定局部，两 seed 的 FNO 节点 U/S 误差均低于 MLP；" if fno_node_better else "此固定局部的节点 U/S 与运动学指标需分开判断；"
            sentence += "节点拟合优势没有转化为准确的 P2 运动学 proxy，FNO 原生查询也未消除这一差距。" if all(r["common_nodal_full10"]["kinematics_proxy"]["P2"]["relative_L2_to_truth"]["value"] > 1 for r in routes) else "P2 数值只反映这128点的节点派生运动学参考。"
            f26 = by[("fno", 2026)]["common_nodal_full10"]["kinematics_proxy"]["P2"]
            lines.extend(["", sentence+f" 例如 FNO2026 的 GT-P2 Frobenius RMS={f26['truth_reference_Frobenius_RMS']:.5g}，误差 RMS={f26['error_Frobenius_RMS']:.5g}（无量纲）；差距不只表现为相对百分比。"])
    else:
        lines.extend(["", "几何科学负结果已完成；没有跳过坏点或报告部分目标精度。"])
    checks = summary["checks"]
    passed = checks.get("all128_Gauss_J_valid") and checks.get("all32_corner_P1_matrices_regular") and checks.get("all_routes_numeric_guards_passed")
    lines.extend(["", f"几何与数值 guards {'PASS' if passed else '未全部通过'}；详见 [metrics.json](metrics.json) 的绝对 RMS、signed cross、最差 cell、源身份、查询可用性、成本与显存。共同 full10 的 C0 是结构接口检查，不能当学习成功；FNO 网格面导数允许跳变。MLP 原生 offnode 因原25输入延拓未定义而未实现。", "",
                  "这是固定 TRAIN 局部诊断，不能外推全 CAD 或 final，也没有 solver strain/offnode GT；P1/P2 是 GT 节点派生 proxy。原25角色不自动定义节点 Dirichlet，此结果不归因于 BC 模糊、不修复输入、不再选 seed 或 checkpoint。", ""])
    atomic_write(output/"metrics.json", json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    atomic_write(output/"BRIEF.md", "\n".join(lines))
    print(json.dumps({"status": "completed", "routes": len(routes), "output": str(output),
                      "source_elapsed_seconds": summary["elapsed_seconds"], "arrays_or_models_read": 0}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summarize(local(args.source), local(args.output))


if __name__ == "__main__":
    main()
