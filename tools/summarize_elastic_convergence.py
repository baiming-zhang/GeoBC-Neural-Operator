"""Export small residual-only receipts; never open fields, weights or labels."""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]


def main(plan_path, output):
    plan = json.loads((ROOT / plan_path).read_text(encoding="utf-8-sig"))
    source = ROOT / plan["output"]
    summary = json.loads((source / "summary.json").read_text())
    if summary.get("status") != "completed" or summary.get("kind") != "elastic_residual_continuation":
        raise ValueError("Require a completed residual-only stage")
    receipt = summary["read_receipt"]
    for key in ("displacement_reads", "stress_reads", "old_final_reads", "DEV_dia_reads",
                "sealed_new_test_reads", "model_or_weight_reads", "optimizer_steps", "scaler_fits",
                "K_guards_repeated", "setup_repeated"):
        if receipt.get(key) != 0:
            raise ValueError(f"Residual-only scope differs: {key}")
    if summary.get("node_count") != 179316 or receipt.get("parent_state_reused") is not True:
        raise ValueError("Original full-node/state coverage differs")
    dest = ROOT / output
    dest.mkdir(parents=True, exist_ok=True)
    result = {key: summary.get(key) for key in (
        "status", "kind", "node_count", "numerical_state", "numerically_converged", "iteration",
        "iterations_executed_total", "true_relative_residual", "costs", "equilibrium", "memory", "read_receipt")}
    result.update(case=plan["case"], source_summary=str(source.relative_to(ROOT) / "summary.json"),
                  original_accuracy_is_not_new_endpoint_accuracy=True,
                  scope="Single TRAIN approximate elasticity; no GT or cross-CAD accuracy claim")
    (dest / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    costs = summary["costs"]
    residual = summary.get("true_relative_residual")
    residual_text = "未产生有限终点" if residual is None else f"{residual:.6g}"
    converged = summary.get("numerically_converged") is True
    cumulative = costs.get("cumulative_solve_wall_seconds")
    time_text = "未知" if cumulative is None else f"{cumulative:.2f} 秒"
    text = (
        "# 物理基线：残差确认\n\n"
        f"固定TRAIN101_479/ver，179,316原节点；状态 `{summary['numerical_state']}`。"
        f"总迭代 {summary.get('iteration')}，真相对残差 {residual_text}，门限1e−8；"
        f"数值收敛 **{'通过' if converged else '未通过'}**。\n\n"
        f"累计求解成本 {time_text}，包含原2000步的27.98秒；新阶段setup/读取与终点平衡成本单列于metrics.json。"
        "原算子、材料、约束、载荷与预条件器复用，未重复8次算子检查。\n\n"
        "这次只根据残差停机，未读GT、未计算新的位移误差。原0.0273%属于先前2000步未收敛端点，"
        "不能移作新端点的精度。4Q、RBE3及未知DEBUG排除仍是原明确近似，数值收敛不证明OptiStruct等价、"
        "跨几何可靠性或工程资格。\n\n"
        "下一步：若收敛通过，固定不同TRAIN/DEV几何比较物理误差、兼容失败率与冷启动成本；"
        "再判断冻结学习预测作为初值是否降低总求解成本。旧final与新封存test保持关闭。\n"
    )
    (dest / "BRIEF.md").write_text(text, encoding="utf-8")
    trace = [item for item in summary.get("trace", []) if item.get("true_relative_residual", 0) > 0]
    if len(trace) >= 2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter([item["cumulative_solve_wall_seconds"] for item in trace],
                   [item["true_relative_residual"] for item in trace], s=10, label="Actual true residual")
        ax.axhline(1e-8, color="tab:orange", linestyle="--", label="Frozen tolerance 1e-8")
        ax.set(yscale="log", xlabel="Cumulative solve wall seconds (including parent)",
               ylabel="True relative residual", title="TRAIN101_479/ver: residual-only continuation")
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(dest / "residual_cost.png", dpi=160)
        plt.close(fig)
    print(json.dumps({"status": "completed", "numerically_converged": converged, "output": str(dest)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="notes/geobc_elastic_convergence_plan.json")
    parser.add_argument("--output", default="results/elastic_convergence")
    args = parser.parse_args()
    main(args.plan, args.output)
