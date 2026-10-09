"""Formal JSON-only readout of one completed frozen TRAIN hash-gate probe."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_query_context_pair import finite_numbers
from tools.diagnose_hash_modulation_use import CADS, KEYS, ROWS, STEP, LAMBDA, brief_readout


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def analyze(folder, plan_path, ready_path, prior_witness_path):
    folder, plan_path = Path(folder).resolve(), Path(plan_path).resolve()
    ready_path, prior_witness_path = Path(ready_path).resolve(), Path(prior_witness_path).resolve()
    summary_path, config_path = folder / "summary.json", folder / "config.json"
    summary, config = read(summary_path), read(config_path)
    ready, plan = read(ready_path), read(plan_path)
    contract = config["contract"]
    require(summary.get("status") == "completed" and summary["config"] == config
            and ready["status"] == "ready" and contract == ready["metadata_dry_run"]["contract"]
            and ready["plan_sha256"] == contract["plan"]["sha256"] == sha(plan_path)
            and ready["source_sha256"] == sha(ROOT / "tools/diagnose_hash_modulation_use.py")
            and plan["status"] == "selected_after_failed_hash_modulation_gate",
            "Completed summary/config/frozen ready-plan-source contract differs")
    require(summary["fixed_step"] == STEP and summary["basis_forward_calls"] == 16
            and summary["backwards"] == 0 and summary["optimizer_updates"] == 0
            and summary["GPU_tensors_released_before_completion"] is True
            and contract["CADs"] == list(CADS) and contract["cases"] == list(KEYS)
            and contract["gradient_weight"] == LAMBDA and contract["normalizers_refitted"] is False
            and math.isfinite(summary["elapsed_seconds"]) and summary["elapsed_seconds"] > 0,
            "Probe cap/frozen endpoint/scope/no-update contract differs")
    prior, witnesses = read(prior_witness_path), summary["witnesses"]
    require(set(witnesses) == set(prior) == set(CADS), "Witness must cover only the fixed TRAIN4 CADs")
    same_fields = ("node_entropy", "tet_entropy", "node_count", "cell_count", "rows", "indices_sha256",
                   "hash_encoding", "normalized_xyz_sha256")
    for rank, cad in enumerate(CADS):
        witness = witnesses[cad]
        require(all(witness[k] == prior[cad][k] for k in same_fields)
                and witness["node_entropy"] == [2026, 1701, rank]
                and witness["tet_entropy"] == [2026, 1702, rank] and witness["rows"] == ROWS
                and witness["cloud_hash_verified_against_both_configs"] is True,
                f"{cad}: sampler/order/node/corner hashes differ from the prior frozen witness")
    require(set(summary["models"]) == {"zero", "actual"}, "Only the two frozen models are allowed")
    for mode, model in summary["models"].items():
        finite_numbers(model, f"{mode}.diagnostic")
        require(model["fixed_step"] == STEP and set(model["cases"]) == set(KEYS)
                and set(model["gamma_by_CAD"]) == set(CADS) and model["parameters_unchanged"] is True
                and model["gradients_absent"] is True and model["gate_mode_restored"] is True,
                f"{mode}: cases/frozen parameters/inference-only evidence differs")
        for key, row in model["cases"].items():
            require(row["query_rows"] == ROWS, f"{key}: mixed-query count differs")
            natural, masked = row["natural"], row["gamma_masked"]
            for quantity in (natural, masked):
                require(all(quantity[k] >= 0 for k in quantity)
                        and close(quantity["joint"], quantity["relative_Lu"] + LAMBDA * quantity["normalized_volume_P1_Frobenius_loss"]),
                        f"{key}: finite squared Lu/P1/joint definition differs")
            for name, value in natural.items():
                require(close(row["delta_masked_minus_natural"][name], masked[name] - value), f"{key}: delta differs")
                ratio = row["masked_to_natural_loss_ratio"][name]
                require((ratio is None if value == 0 else ratio is not None and close(ratio, masked[name] / value)),
                        f"{key}: loss ratio differs")
            if mode == "zero":
                require(natural == masked and row["delta_u_over_true_u_norm"] == 0
                        and all(v == 0 for v in row["delta_masked_minus_natural"].values()), "Zero mask must be identical")
        for cad, gamma in model["gamma_by_CAD"].items():
            require(gamma["queries"] == ROWS and gamma["channels"] == 24 and gamma["finite_fraction"] == 1
                    and gamma["delta_RMS"] >= 0 and 0 <= gamma["spatial_centered_RMS"] <= gamma["delta_RMS"] + 1e-12
                    and 0 <= gamma["exact_either_endpoint_fraction"] <= gamma["near_either_endpoint_fraction"] <= 1
                    and len(gamma["gamma_channel_mean"]) == 24
                    and 0 <= gamma["gamma_min"] <= gamma["gamma_max"] <= 2, f"{cad}: gamma definition/range differs")
            if mode == "zero":
                require(gamma["delta_RMS"] == gamma["spatial_centered_RMS"] == 0
                        and gamma["gamma_min"] == gamma["gamma_max"] == 1, "Zero gate must be identity")
        if mode == "zero":
            require(model["trained_gate_head"]["weight_and_bias_exact_zero"] is True
                    and set(model["zero_control_checks"]) == set(CADS)
                    and all(v["natural_masked_basis_exact_equal"] is True and v["max_abs_scaled_basis_delta"] == 0
                            for v in model["zero_control_checks"].values()), "Zero head/basis exact-identity evidence differs")
    require(summary["readout"] == brief_readout(summary["models"]), "Aggregate readout differs from its 12-case records")
    readout = summary["readout"]
    actual = readout["actual"]
    natural, masked = actual["mean_natural_losses"], actual["mean_masked_losses"]
    result = {"status": "completed", "stage": "frozen_hash_modulation_use_and_saturation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "formal_checks_passed": True, "CPU_only_analysis": True,
        "fixed_step": STEP, "basis_forward_calls": 16, "backwards": 0, "optimizer_updates": 0,
        "scientific_training_runs": 0, "frozen_diagnostic_runs": 1, "elapsed_seconds": summary["elapsed_seconds"],
        "scope": "fixed TRAIN4 CADs x ver/hor/tor, same mixed12800 queries; no validation/dia/finaltest/stress arrays",
        "prior_frozen_sampler_exact_match": True, "zero_gate_identity_passed": True,
        "actual": {"Lu_natural": natural["relative_Lu"], "Lu_gamma_masked": masked["relative_Lu"],
            "Lu_masked_to_natural_ratio": masked["relative_Lu"] / natural["relative_Lu"],
            "P1_cell_squared_loss_natural": natural["normalized_volume_P1_Frobenius_loss"],
            "P1_cell_squared_loss_gamma_masked": masked["normalized_volume_P1_Frobenius_loss"],
            "P1_cell_squared_loss_masked_to_natural_ratio": masked["normalized_volume_P1_Frobenius_loss"] / natural["normalized_volume_P1_Frobenius_loss"],
            "cases_Lu_masking_harms": actual["cases_masking_increases_loss"]["relative_Lu"],
            "cases_P1_cell_loss_masking_improves": actual["cases_masking_decreases_loss"]["normalized_volume_P1_Frobenius_loss"],
            "sampled_cases": 12, "gamma_delta_RMS_mean4": actual["gamma_delta_RMS_mean4"],
            "gamma_spatial_centered_RMS_mean4": actual["gamma_spatial_centered_RMS_mean4"],
            "near_endpoint_fraction_mean4": actual["near_either_endpoint_fraction_mean4"],
            "exact_endpoint_fraction_mean4": actual["exact_either_endpoint_fraction_mean4"],
            "mean_delta_u_over_true_u_norm": actual["mean_delta_u_over_true_u_norm"]},
        "readout": readout,
        "trained_gate_heads": {mode: model["trained_gate_head"] for mode, model in summary["models"].items()},
        "gamma_by_CAD": summary["models"]["actual"]["gamma_by_CAD"],
        "interpretation": "The learned gate is used by the fitted TRAIN displacement field. Turning it off worsens sampled signed-u loss while lowering sampled P1 cell loss; this is a within-checkpoint tradeoff, not a retrained-model comparison or a cause of generalization failure.",
        "limitations": ["Masking is an out-of-distribution intervention on fixed last10000 weights, not u-only-selected best checkpoints.",
            "Near endpoint means gamma<=0.05 or>=1.95; it is not hard numerical saturation. Exact gamma0/2 frequency is reported separately.",
            "P1 is normalized volume-weighted squared cell Frobenius loss, not the all-node recovered P1 relativeL2 percentage or solver GT strain.",
            "One fixed TRAIN witness with duplicate rows cannot establish whole-node performance, validation causes or statistical/generalization claims.",
            "Equal stored parameters does not remove the active gate's effective-capacity difference; prior Composite controls are not matched."],
        "next_stage": "root_selected_geometry_connectivity_input_pilot_pending_result",
        "next_stage_scope": "Root-authorized single TRAIN101_479 label-free mesh-edge distance5 input pilot; no training/GT yet. Graph distance is not stiffness. No gate-parameter sweep is implied.",
        "source_artifacts": {name: {"path": str(path), "sha256": sha(path)} for name, path in
            {"summary": summary_path, "config": config_path, "plan": plan_path, "ready": ready_path, "prior_witnesses": prior_witness_path}.items()}}
    finite_numbers(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/geobc_exploration/hash_modulation_use_frozen_10k")
    parser.add_argument("--plan", type=Path, default=ROOT / "notes/geobc_hash_modulation_use_plan.json")
    parser.add_argument("--ready", type=Path, default=ROOT / "notes/hash_modulation_use_diagnostic_ready.json")
    parser.add_argument("--prior-witnesses", type=Path, default=ROOT / "runs/geobc_exploration/query_learning_frozen_10k/witnesses.json")
    parser.add_argument("--output", type=Path, default=ROOT / "notes/geobc_hash_modulation_use_result.json")
    args = parser.parse_args()
    result = analyze(args.run, args.plan, args.ready, args.prior_witnesses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({"status": result["status"], "formal_checks_passed": result["formal_checks_passed"],
                      "actual": result["actual"], "next_stage": result["next_stage"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
