"""Apply the predeclared gradient-teacher gate to one completed matched pair."""
from pathlib import Path
import argparse
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def differences(left, right, path=()):
    if type(left) is not type(right):
        return [path]
    if isinstance(left, dict):
        if set(left) != set(right):
            return [path + ("<keys>",)]
        return [item for key in left for item in differences(left[key], right[key], path + (key,))]
    if isinstance(left, list):
        if len(left) != len(right):
            return [path + ("<length>",)]
        return [item for index, (a, b) in enumerate(zip(left, right))
                for item in differences(a, b, path + (str(index),))]
    return [] if left == right else [path]


def reduction(zero, actual):
    if not math.isfinite(zero) or not math.isfinite(actual) or zero <= 0:
        raise ValueError("Relative improvement requires finite positive control error")
    return 1 - actual / zero


def fixed_cad_mean(group):
    per = group["strain"]["per_case"]
    return sum(per[f"101_479/{load}"]["relative_l2"] for load in ("ver", "hor", "tor")) / 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mode", choices=("pair", "calibrated"), default="pair")
    args = parser.parse_args()
    suffix = "" if args.seed == 2026 else f"_seed{args.seed}"
    folders = {mode: ROOT / f"runs/geobc_exploration/displacement_gradient_{mode}{suffix}_10k"
               for mode in ("zero", "actual")}
    if args.mode == "calibrated":
        if args.seed != 2026:
            raise ValueError("Declare the second-seed pair filenames before evaluating it")
        folders["actual"] = ROOT / "runs/geobc_exploration/displacement_gradient_calibrated_10k"
    summaries = {mode: read(folder / "summary.json") for mode, folder in folders.items()}
    if any(value.get("status") != "completed" for value in summaries.values()):
        raise ValueError("Both branches must be completed")
    z, a = (summaries[mode] for mode in ("zero", "actual"))
    zc, ac = z["config"], a["config"]
    for key in ("model", "scaler", "initial_parameters", "cases", "geometry", "u_loss"):
        if zc[key] != ac[key]:
            raise ValueError(f"Paired {key} differs")
    if args.mode == "pair":
        changed = differences(zc["contract"], ac["contract"])
        if not changed or any(path[-1] != "gradient_weight" for path in changed):
            raise ValueError(f"Nonmatched contract differences: {changed}")
        expected = 1
    else:
        from tools.review_gradient_balance_setup import common_contract
        gate = read(ROOT / "notes/gradient_balance_parent_review.json")
        if not gate["baseline_reuse_accepted"] or gate["source_sha256"] != ac["contract"]["code_sha256"]:
            raise ValueError("Calibrated computation-body compatibility proof differs")
        changed = differences(common_contract(zc["contract"]), common_contract(ac["contract"], calibrated=True))
        if changed:
            raise ValueError(f"Calibrated common controls differ: {changed}")
        expected = read(ROOT / "notes/geobc_gradient_calibration.json")["gradient_weight"]
    if [c["contract"]["requested"]["gradient_weight"] for c in (zc, ac)] != [0, expected]:
        raise ValueError("Unexpected frozen gradient coefficient")
    witnesses = [read(folders[mode] / "sampling_first_batch.json") for mode in ("zero", "actual")]
    if witnesses[0] != witnesses[1]:
        raise ValueError("Node/cell/corner sampling witnesses differ")
    for summary in (z, a):
        if summary["encoder_gradient_check"]["encoder_gradient_l1"] <= 0:
            raise ValueError("Geometry encoder did not receive native decoder gradients")
    zv, av = (s["best_metrics"]["validation"] for s in (z, a))
    zt, at = (s["best_metrics"]["train"] for s in (z, a))
    val_reduction = reduction(zv["strain"]["case_mean_relative_l2"], av["strain"]["case_mean_relative_l2"])
    fixed_reduction = reduction(fixed_cad_mean(zt), fixed_cad_mean(at))
    mean_delta = 100 * (av["case_mean_relative_l2"] - zv["case_mean_relative_l2"])
    worst_delta = 100 * (av["worst_case_relative_l2"] - zv["worst_case_relative_l2"])
    gate_passed = val_reduction >= .30 and fixed_reduction >= .50 and mean_delta <= 2 and worst_delta <= 2
    result = {
        "status": "completed", "stage": "matched_displacement_gradient_supervision" if args.mode == "pair" else "single_train_calibrated_gradient_weight", "seed": args.seed,
        "pair_controls_passed": True, "contract_difference_paths": [".".join(p) for p in changed],
        "validation_strain_relative_error_reduction": val_reduction,
        "fixed_train_101_479_strain_relative_error_reduction": fixed_reduction,
        "validation_u_mean_deterioration_percentage_points": mean_delta,
        "validation_u_worst_deterioration_percentage_points": worst_delta,
        "predeclared_gate": read(ROOT / "notes/geobc_gradient_teacher_plan.json")["predeclared_next_gate"],
        "clear_gain_gate_passed": gate_passed,
        "better_than_zero_strain_baseline": av["strain"]["case_mean_relative_l2"] <= 1,
        "next_stage": "seed2027_replication" if gate_passed else "targeted_gradient_optimization_or_transfer_diagnosis",
        "sampling_witness_sha256": witnesses[0]["witness_sha256"],
        "checkpoint_selection": "known-load validation mean physical u only; strain reports same selected checkpoint",
        "final_test_and_dia_opened": False,
        "interpretation": "Development validation and training-u-derived P1 tensor proxy; no solver strain label, significance or final generalization claim.",
        "runs": {}
    }
    for mode, summary in summaries.items():
        train, val = summary["best_metrics"]["train"], summary["best_metrics"]["validation"]
        result["runs"][mode] = {
            "best_step": summary["best_step"], "elapsed_seconds": summary["elapsed_seconds"],
            "train_u_mean_relative_l2": train["case_mean_relative_l2"],
            "train_strain_mean_relative_l2": train["strain"]["case_mean_relative_l2"],
            "validation_u_mean_relative_l2": val["case_mean_relative_l2"],
            "validation_u_worst_relative_l2": val["worst_case_relative_l2"],
            "validation_strain_mean_relative_l2": val["strain"]["case_mean_relative_l2"],
            "validation_strain_worst_relative_l2": val["strain"]["worst_case_relative_l2"],
            "fixed_train_101_479_strain_mean_relative_l2": fixed_cad_mean(train),
        }
    filename = f"geobc_gradient_teacher_pair_result{suffix}.json" if args.mode == "pair" else "geobc_gradient_calibrated_result.json"
    path = ROOT / "notes" / filename
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({key: result[key] for key in ("validation_strain_relative_error_reduction",
          "fixed_train_101_479_strain_relative_error_reduction", "validation_u_mean_deterioration_percentage_points",
          "validation_u_worst_deterioration_percentage_points", "next_stage")}))


if __name__ == "__main__":
    main()
