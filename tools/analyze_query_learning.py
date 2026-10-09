"""Summarize the completed frozen TRAIN query-learning diagnostic on CPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
CADS = ("101_479", "101_428", "101_56", "101_78")
LOADS = ("ver", "hor", "tor")
LOSS_NAMES = ("relative_Lu", "normalized_volume_P1_Frobenius_loss", "joint")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def finite(value):
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError("Diagnostic contains a nonfinite numeric readout")


def block_readout(block):
    pairs = block["pairs"]
    direction = block["joint_direction"]
    return {
        "negative_pairs": sum(pair["cosine"] is not None and pair["cosine"] < 0 for pair in pairs),
        "total_pairs": len(pairs),
        "undefined_zero_norm_pairs": sum(pair["zero_norm"] for pair in pairs),
        "pairs": pairs,
        "CAD_gradient_norms": block["CAD_gradient_norms"],
        "mean4_gradient_norm": block["mean4_norm"],
        "dot_with_mean4": {cad: value["dot_with_mean4"] for cad, value in direction.items()},
        "dot_with_mean_other3": {cad: value["dot_with_mean_other3"] for cad, value in direction.items()},
        "CADs_with_negative_dot_mean4": [cad for cad, value in direction.items() if value["dot_with_mean4"] < 0],
        "CADs_with_negative_dot_mean_other3": [cad for cad, value in direction.items() if value["dot_with_mean_other3"] < 0],
    }


def analyze(summary_path, plan_path):
    began = time.monotonic()
    summary, plan = read(summary_path), read(plan_path)
    contract = summary["config"]["contract"]
    keys = [f"{cad}/{load}" for cad in CADS for load in LOADS]
    if (summary["status"] != "completed" or summary["fixed_step"] != 10000
            or summary["case_backwards"] != 24 or summary["optimizer_updates"] != 0
            or contract["CADs"] != list(CADS) or contract["cases"] != keys
            or contract["normalizers_refitted"] is not False
            or plan["data_contract"]["cases"] != keys
            or hashlib.sha256(plan_path.read_bytes()).hexdigest() != contract["plan"]["sha256"]):
        raise ValueError("Expected the completed predeclared fixed-last10k TRAIN4x3 diagnostic")
    finite(summary["models"])
    models = summary["models"]
    alignment, contexts = {}, {}
    for mode in ("zero", "actual"):
        model = models[mode]
        if (model["fixed_step"] != 10000 or not model["parameter_gradient_finite_and_present"]
                or list(model["case_losses"]) != list(CADS)
                or any(set(model["case_losses"][cad]) != set(LOADS) for cad in CADS)):
            raise ValueError("Frozen model readouts must cover only the fixed training cases")
        alignment[mode] = {name: block_readout(block) for name, block in model["CAD_joint_gradients"].items()}
        contexts[mode] = {
            "query_centered_RMS_mean4": mean(model["context_statistics"][cad]["query"]["query_centered_RMS"] for cad in CADS),
            "query_magnitude_RMS_mean4": mean(model["context_statistics"][cad]["query"]["total_RMS"] for cad in CADS),
            "per_CAD": {cad: {name: model["context_statistics"][cad]["query"][name] for name in
                ("query_centered_RMS", "total_RMS", "structural_exact_zero", "FP16_cast_centered_variation_ratio")} for cad in CADS},
            "CAD_context_distances": model["context_CAD_pair_distances"],
        }
    rows = models["actual"]["actual_final_query_zero_intervention"]
    if set(rows) != set(keys) or any(row["query_rows"] != 12800 for row in rows.values()):
        raise ValueError("Actual mask readout must use the same twelve 12800-row witnesses")
    intact = {name: mean(row["intact"][name] for row in rows.values()) for name in LOSS_NAMES}
    masked = {name: mean(row["final_query32_masked"][name] for row in rows.values()) for name in LOSS_NAMES}
    if any(value <= 0 for value in intact.values()):
        raise ValueError("Positive intact loss required for the reported loss multipliers")
    counts = {name: sum(row["delta_masked_minus_intact"][name] > 0 for row in rows.values()) for name in LOSS_NAMES}
    mask = {
        "cases": len(rows), "query_rows_per_case": 12800,
        "aggregation": "Equal-case arithmetic means over twelve duplicate-preserving mixed witnesses; multipliers are ratios of these means",
        "mean_intact_losses": intact,
        "mean_masked_losses": masked,
        "masked_to_intact_loss_multiplier": {name: masked[name] / intact[name] for name in LOSS_NAMES},
        "mean_delta_masked_minus_intact": {name: masked[name] - intact[name] for name in LOSS_NAMES},
        "cases_masking_increases_loss": counts,
        "mean_delta_u_over_true_u_norm": mean(row["delta_u_over_true_u_norm"] for row in rows.values()),
        "per_case": {key: {"delta_u_over_true_u_norm": row["delta_u_over_true_u_norm"],
            "intact": row["intact"], "final_query32_masked": row["final_query32_masked"]} for key, row in rows.items()},
        "interpretation": "Masking final query32 raises each prescribed sampled loss in all twelve cases, supporting local learned use. The zero latent can be off distribution; this is not a causal source-information or generalization test.",
    }
    actual_full = alignment["actual"]["full_model"]
    result = {
        "status": "completed", "stage": "frozen_TRAIN_query_learning_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_CPU_only": True, "analysis_arrays_weights_or_GPU_opened": False,
        "source_summary": {"path": str(summary_path.relative_to(ROOT)), "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest()},
        "source_plan": {"path": str(plan_path.relative_to(ROOT)), "sha256": contract["plan"]["sha256"]},
        "scope": {"CADs": list(CADS), "loads": list(LOADS), "cases": keys,
            "checkpoint_step": 10000, "checkpoint_selection": "Both fixed last.pt at10000; no validation-based selection in this diagnostic",
            "validation_dia_finaltest_stress_fields_read_by_diagnostic": False,
            "normalizers": "Original train40 common zero-center displacement scale and per-case u/P1 loss denominators reused; no refit",
            "gradient_weight": contract["gradient_weight"], "case_backwards": 24, "optimizer_updates": 0,
            "witness": "One portable witness per CAD shared across three loads, both models and mask;6400 uniform nodes +1600 uniform tets x4corners, duplicates retained",
            "gradient": "Arithmetic mean of the three unnormalized joint-objective load gradients per CAD; cosines only within the same frozen model state"},
        "loss_definitions": {
            "relative_Lu": "Relative-case scaled vector MSE on sampled mixed rows; a squared objective, not an all-node displacement relativeL2 score",
            "normalized_volume_P1_Frobenius_loss": "Uniformly sampled cell mean of (V/fullcaseMeanV)*FrobeniusErrorSquared/fullcaseReferenceSquaredRMS; tensor shear squared weight2; physical cornerP1 recovered from signed displacement",
            "P1_limit": "Sampled normalized squared volume loss; neither all-node volume-averaged P1 relativeL2 nor solver strain tensor accuracy",
            "joint": "relative_Lu + fixed_gradient_weight*normalized_volume_P1_Frobenius_loss"},
        "gradient_alignment": alignment, "learned_query_context": contexts,
        "actual_same_weights_query_mask": mask,
        "interpretation": {
            "actual_negative_full_model_pairs": actual_full["negative_pairs"],
            "actual_full_model_pairs": actual_full["total_pairs"],
            "actual_CADs_with_negative_dot_mean4": actual_full["CADs_with_negative_dot_mean4"],
            "all_actual_CADs_have_positive_dot_mean4": all(value > 0 for value in actual_full["dot_with_mean4"].values()),
            "gradient_readout": "Actual has4/6 negative CAD-pair cosines, but all four full-model CAD gradients have positive dot with mean4. Negative pairs alone do not show that the shared mean direction harms a CAD; these fixed-state TRAIN readings do not establish gradient interference as the cause of weak validation.",
            "block_limit": "Encoder-only or mean-other3 opposition can coexist with a positive complete mean4 direction; disjoint block norms and signs are reported without treating every small block cosine as the main mechanism.",
            "optimizer_limit": "Infinitesimal plain-gradient reasoning is not actual Adam momentum/preconditioning or finite-step behavior; no optimizer update was performed.",
            "current_hypothesis": "The query branch is learned and locally used, yet shared validation and displacement-derived P1 quality remain limited. How geometry conditions the coordinate/hash representation is a bounded next hypothesis, not a conclusion that information is absent or optimization alone explains the gap.",
            "generalization_claim": False},
        "report_sentence_zh": "冻结训练诊断表明query分支已被局部使用：掩码使12/12采样case的位移和cell P1目标增大，但这是可能离分布的latent干预。actual虽有4/6负CAD对夹角，完整梯度对四CAD平均方向的点积均为正，尚不能把验证不足归因于梯度冲突；下一单一假设可考察几何条件如何作用于坐标/hash表示。",
        "cost": {"diagnostic_elapsed_seconds": summary["elapsed_seconds"], "memory": summary["memory"],
            "GPU_tensors_released_before_completion": summary["GPU_tensors_released_before_completion"]},
        "analysis_elapsed_seconds": time.monotonic() - began,
    }
    finite(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "runs/geobc_exploration/query_learning_frozen_10k/summary.json")
    parser.add_argument("--plan", type=Path, default=ROOT / "notes/geobc_query_learning_plan.json")
    parser.add_argument("--output", type=Path, default=ROOT / "notes/geobc_query_learning_result.json")
    args = parser.parse_args()
    result = analyze(args.summary.resolve(), args.plan.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")
    mask = result["actual_same_weights_query_mask"]
    print(json.dumps({"status": result["status"], "actual_negative_pairs": result["interpretation"]["actual_negative_full_model_pairs"],
        "actual_negative_dot_mean4_CADs": result["interpretation"]["actual_CADs_with_negative_dot_mean4"],
        "masked_to_intact_loss_multiplier": mask["masked_to_intact_loss_multiplier"],
        "mean_query_centered_RMS": result["learned_query_context"]["actual"]["query_centered_RMS_mean4"],
        "mean_delta_u_over_true_u_norm": mask["mean_delta_u_over_true_u_norm"]}))


if __name__ == "__main__":
    main()
