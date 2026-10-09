"""Summarize fixed single-CAD endpoints and their explicit exposure caveats."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def draws(case_keys, steps, replace, seed):
    rng = np.random.default_rng(seed + 23)
    counts = {key: 0 for key in ("101_479/ver", "101_479/hor", "101_479/tor")}
    first = None
    for _ in range(steps):
        selected = [case_keys[index] for index in rng.choice(len(case_keys), 4, replace=replace)]
        if first is None:
            first = selected
        for key in selected:
            if key in counts:
                counts[key] += 1
    return counts, first


def main():
    folder = ROOT / "runs/geobc_exploration/specialist_101_479_joint_u_P1_10k"
    summary = read(folder / "summary.json")
    if summary["status"] != "completed" or summary["checkpoint_selection"] != "fixed10000_endpoint_no_selection":
        raise ValueError("Fixed specialist endpoints have not completed")
    config = summary["config"]
    reference_folder = ROOT / "runs/geobc_exploration/displacement_gradient_calibrated_10k"
    reference = read(reference_folder / "config.json")
    for key in ("model", "scaler", "initial_parameters", "initial_decoder_parameter_sha256", "optimizer"):
        if config[key] != reference[key]:
            raise ValueError(f"Frozen specialist control changed: {key}")
    if summary["encoder_gradient_check"]["encoder_gradient_l1"] <= 0 or summary["encoder_gradient_check"]["decoder_gradient_l1"] <= 0:
        raise ValueError("Native parameter gradients were missing")
    cases = [record["sample_key"] for record in config["cases"]["train"]]
    if cases != ["101_479/ver", "101_479/hor", "101_479/tor"]:
        raise ValueError("Unexpected diagnostic field scope")
    for key in cases:
        if config["u_loss"]["case_records"][key] != reference["u_loss"]["case_records"][key]:
            raise ValueError("Original per-case u normalizers were refit")
    full = [record["sample_key"] for record in reference["cases"]["train"]]
    shared_counts, shared_first = draws(full, 10000, False, 2026)
    short_counts, specialist_first = draws(cases, 250, True, 2026)
    final_counts, _ = draws(cases, 10000, True, 2026)
    if shared_first != read(reference_folder / "sampling_first_batch.json")["case_keys"]:
        raise ValueError("Shared exposure replay first draw differs from saved witness")
    if specialist_first != read(folder / "sampling_first_batch.json")["case_keys"]:
        raise ValueError("Specialist exposure replay first draw differs from saved witness")
    baseline = summary["reference_final_10000"]
    endpoints = {}
    for step in (250, 10000):
        metrics = summary["endpoint_metrics"][str(step)]
        if metrics["step"] != step:
            raise ValueError("Fixed endpoint changed")
        group = metrics["train"]
        endpoints[str(step)] = {
            "step": step, "u_mean_relative_l2": group["case_mean_relative_l2"],
            "u_worst_relative_l2": group["worst_case_relative_l2"],
            "P1_mean_relative_l2": group["strain"]["case_mean_relative_l2"],
            "P1_worst_relative_l2": group["strain"]["worst_case_relative_l2"],
            "u_relative_error_reduction_vs_shared_final": 1-group["case_mean_relative_l2"]/baseline["case_mean_relative_l2"],
            "P1_relative_error_reduction_vs_shared_final": 1-group["strain"]["case_mean_relative_l2"]/baseline["strain"]["case_mean_relative_l2"],
            "per_case": {key: {"u_relative_l2": group["per_case"][key]["relative_l2"],
                               "P1_relative_l2": group["strain"]["per_case"][key]["relative_l2"]} for key in cases}}
    result = {"status": "completed", "stage": "fixed_training_CAD_joint_u_P1_fitting_diagnostic",
              "elapsed_seconds": summary["elapsed_seconds"], "control_checks_passed": True,
              "endpoint_selection": "Fixed250/10000, no validation or strain-based model selection",
              "reference_step": 10000, "reference_u_mean_relative_l2": baseline["case_mean_relative_l2"],
              "reference_P1_mean_relative_l2": baseline["strain"]["case_mean_relative_l2"],
              "endpoints": endpoints, "case_slot_draw_counts": {"shared10000": shared_counts,
                  "specialist250": short_counts, "specialist10000": final_counts},
              "exposure_replay": "Only NumPy case RNG replay validated against saved first-case witnesses; no fields/checkpoints/GPU",
              "expected_exposure_ratio_final": 40,
              "conclusion": "On this fixed training CAD, the same native field and calibrated loss fit both displacement and P1 gradients under dedicated optimization. Sharing/conditioning/optimization competition remain candidates; cause and transfer success are not established.",
              "limitations": "OneCAD; separate model adds totalparameters and40x case exposure at10k.250 matches expectedslotexposure only, not optimization trajectory or per-case counts. P1 proxy is not solverGTtensor.",
              "next_stage": "matched_query_dependent_geometry_BC_context", "validation_dia_finaltest_fields_opened": False}
    path = ROOT / "notes/geobc_specialist_result.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({"fixed250": endpoints["250"], "fixed10000": endpoints["10000"],
                      "case_slot_draw_counts": result["case_slot_draw_counts"]}))


if __name__ == "__main__":
    main()
