"""Read completed paired runs and apply the predeclared next-stage rule."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def main():
    summaries = {}
    for mode in ("zero", "actual"):
        folder = ROOT / f"runs/geobc_exploration/stress_u_condition_{mode}_width128_10k"
        s = read(folder / "summary.json")
        if s["status"] != "completed":
            raise ValueError("Both runs must be completed before selecting the next stage")
        summaries[mode] = s
    z, a = (summaries[k] for k in ("zero", "actual"))
    zc, ac = z["config"], a["config"]
    for key in ("model", "scaler", "initial_parameters", "cases", "geometry"):
        if zc[key] != ac[key]:
            raise ValueError(f"Paired runs differ in {key}")
    for key in ("split_sha256", "code_sha256", "u_response"):
        if zc["contract"][key] != ac["contract"][key]:
            raise ValueError(f"Paired provenance differs in {key}")
    for mode, s in summaries.items():
        if not s["encoder_gradient_check"]["encoder_gradient_l1"] > 0:
            raise ValueError("Student geometry encoder did not receive gradients")
    witnesses = [read(ROOT / f"runs/geobc_exploration/stress_u_condition_{m}_width128_10k/sampling_first_batch.json") for m in ("zero", "actual")]
    if witnesses[0] != witnesses[1]:
        raise ValueError("Pair sampling witness differs")
    requested = [dict(s["config"]["contract"]["requested"]) for s in (z, a)]
    for item in requested:
        item.pop("u_condition")
    if requested[0] != requested[1]:
        raise ValueError("Pair sampling/budget/optimizer settings differ")
    zv, av = (s["best_metrics"]["validation"] for s in (z, a))
    gain = 100 * (zv["case_mean_relative_l2"] - av["case_mean_relative_l2"])
    worst_delta = 100 * (av["worst_case_relative_l2"] - zv["worst_case_relative_l2"])
    hotspot_delta = 100 * (av["case_mean_hotspot_relative_l2"] - zv["case_mean_hotspot_relative_l2"])
    gate = read(ROOT / "notes/geobc_u_conditioned_stress_plan.json")["next_decision_gate"]
    clear = gain >= 2 and worst_delta <= 2 and hotspot_delta <= 0
    result = {"stage": "frozen_u_conditioned_stress", "status": "completed",
              "pair_control_checks_passed": True,
              "validation_mean_gain_percentage_points": gain,
              "worst_deterioration_percentage_points": worst_delta,
              "hotspot_deterioration_percentage_points": hotspot_delta,
              "predeclared_gate": gate, "clear_gain_gate_passed": clear,
              "next_stage": "seed2027_replication" if clear else "mesh_recovered_strain_control",
              "common_u3_for_strain_pair_if_needed": "actual" if gain > 0 else "zero",
              "teacher_fingerprint": zc["u_condition"]["teacher_fingerprint"],
              "initial_parameters": zc["initial_parameters"],
              "sampling_witness_sha256": witnesses[0]["witness_sha256"],
              "final_test_and_dia_opened": False,
              "interpretation": "Known-load development validation only; heuristic research decision, no statistical significance or final generalization claim",
              "runs": {}}
    for mode, s in summaries.items():
        v = s["best_metrics"]["validation"]
        result["runs"][mode] = {"best_step": s["best_step"], "elapsed_seconds": s["elapsed_seconds"],
            "train_mean_relative_l2": s["best_metrics"]["train"]["case_mean_relative_l2"],
            "validation_mean_relative_l2": v["case_mean_relative_l2"],
            "validation_worst_relative_l2": v["worst_case_relative_l2"],
            "validation_hotspot_mean_relative_l2": v["case_mean_hotspot_relative_l2"],
            "per_load_validation_mean_relative_l2": {load: sum(x["relative_l2"] for key, x in v["per_case"].items() if key.endswith("/" + load)) / 5 for load in ("ver", "hor", "tor")}}
    path = ROOT / "notes/geobc_u_stress_pair_result.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({key: result[key] for key in ("validation_mean_gain_percentage_points", "worst_deterioration_percentage_points", "hotspot_deterioration_percentage_points", "next_stage")}))


if __name__ == "__main__":
    main()
