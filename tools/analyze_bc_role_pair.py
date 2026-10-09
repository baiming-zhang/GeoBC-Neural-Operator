"""Assess one frozen matched global BC-role displacement experiment pair."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_gradient_teacher_pair import differences, reduction


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def main():
    folders = {mode: ROOT / f"runs/geobc_exploration/displacement_bc_role_{mode}_10k"
               for mode in ("zero", "actual")}
    summaries = {mode: read(folder / "summary.json") for mode, folder in folders.items()}
    z, a = (summaries[mode] for mode in ("zero", "actual"))
    if any(summary.get("status") != "completed" for summary in (z, a)):
        raise ValueError("Both BC-role runs must finish")
    zc, ac = z["config"], a["config"]
    allowed_contract = {("requested", "bc_role_mode"), ("bc_roles", "bc_role_mode")}
    changed = differences(zc["contract"], ac["contract"])
    if set(changed) != allowed_contract:
        raise ValueError(f"Pair contract differences are not matched: {changed}")
    for key in ("scaler", "initial_parameters", "initial_decoder_parameter_sha256", "cases", "u_loss",
                "local_geometry", "gradient_supervision", "optimizer", "precision", "sampling_rng",
                "parameter_count", "decoder_parameter_count", "checkpoint_selection"):
        if zc[key] != ac[key]:
            raise ValueError(f"Paired {key} differs")
    model_changes = differences(zc["model"], ac["model"])
    if not model_changes or any(path[-1] not in ("bc_role_mode", "role_input_policy") for path in model_changes):
        raise ValueError(f"Model changes exceed the declared role policy: {model_changes}")
    geometry_changes = differences(zc["geometry"], ac["geometry"])
    if set(geometry_changes) != {("bc_role_mode",), ("encoder_roles_sha256",)}:
        raise ValueError(f"Geometry coordinates/indices/raw roles changed: {geometry_changes}")
    witnesses = [read(folders[mode] / "sampling_first_batch.json") for mode in ("zero", "actual")]
    if witnesses[0] != witnesses[1]:
        raise ValueError("Case/node/cell sampling witnesses differ")
    for summary in (z, a):
        check = summary["encoder_gradient_check"]
        if check["encoder_gradient_l1"] <= 0 or check["decoder_gradient_l1"] <= 0:
            raise ValueError("Encoder/native decoder did not receive gradients")
        if summary["config"]["model"]["encoder_parameters"] != 16800:
            raise ValueError("Both encoders must include the same64 extra parameters")
        if not summary["config"]["geometry"]["reference_xyz_mask_node_indices_exact_equal"]:
            raise ValueError("The original clouds were not reused exactly")
    zv, av = (s["best_metrics"]["validation"] for s in (z, a))
    zt, at = (s["best_metrics"]["train"] for s in (z, a))
    mean_gain = 100 * (zv["case_mean_relative_l2"] - av["case_mean_relative_l2"])
    worst_delta = 100 * (av["worst_case_relative_l2"] - zv["worst_case_relative_l2"])
    strain_gain = reduction(zv["strain"]["case_mean_relative_l2"], av["strain"]["case_mean_relative_l2"])
    gate = read(ROOT / "notes/geobc_bc_role_plan.json")["comparison"]["clear_gain_gate"]
    passed = (mean_gain >= gate["validation_u_mean_improvement_percentage_points_at_least"]
              and worst_delta <= gate["validation_u_worst_deterioration_percentage_points_at_most"]
              and strain_gain >= gate["validation_P1_relative_error_reduction_at_least"]
              and av["strain"]["case_mean_relative_l2"] <= gate["validation_P1_case_mean_relative_L2_at_most"])
    result = {"status": "completed", "stage": "global_BC_role_displacement_control", "seed": 2026,
              "pair_controls_passed": True, "matched_contract_difference_paths": [".".join(p) for p in changed],
              "validation_u_mean_improvement_percentage_points": mean_gain,
              "validation_u_worst_deterioration_percentage_points": worst_delta,
              "validation_P1_relative_error_reduction": strain_gain,
              "train_u_mean_improvement_percentage_points": 100 * (zt["case_mean_relative_l2"] - at["case_mean_relative_l2"]),
              "train_P1_relative_error_reduction": reduction(zt["strain"]["case_mean_relative_l2"], at["strain"]["case_mean_relative_l2"]),
              "better_than_zero_strain_baseline": av["strain"]["case_mean_relative_l2"] <= 1,
              "predeclared_gate": gate, "clear_gain_gate_passed": passed,
              "next_stage": "matched_seed2027_replication" if passed else "one_fitting_vs_shared_conditioning_diagnostic",
              "sampling_witness_sha256": witnesses[0]["witness_sha256"],
              "selection": "Each checkpoint selected only by known-load validation physical displacement mean",
              "interpretation": "Development validation only; P1 is training-u-derived proxy, not solver strain; role union is coarse and sparse",
              "final_test_and_dia_opened": False, "runs": {}}
    for mode, summary in summaries.items():
        train, val = summary["best_metrics"]["train"], summary["best_metrics"]["validation"]
        result["runs"][mode] = {"best_step": summary["best_step"], "elapsed_seconds": summary["elapsed_seconds"],
            "train_u_mean_relative_l2": train["case_mean_relative_l2"],
            "train_P1_mean_relative_l2": train["strain"]["case_mean_relative_l2"],
            "validation_u_mean_relative_l2": val["case_mean_relative_l2"],
            "validation_u_worst_relative_l2": val["worst_case_relative_l2"],
            "validation_P1_mean_relative_l2": val["strain"]["case_mean_relative_l2"],
            "validation_P1_worst_relative_l2": val["strain"]["worst_case_relative_l2"]}
    path = ROOT / "notes/geobc_bc_role_pair_result.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({key: result[key] for key in ("pair_controls_passed", "validation_u_mean_improvement_percentage_points",
        "validation_u_worst_deterioration_percentage_points", "validation_P1_relative_error_reduction", "clear_gain_gate_passed", "next_stage")}))


if __name__ == "__main__":
    main()
