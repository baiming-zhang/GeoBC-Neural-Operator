"""JSON-only assessment of the predeclared four-TRAIN message pilot.

No scientific arrays, checkpoints or GPU are opened. A negative utility gate
is a completed scientific result; incomplete or unmatched runs are rejected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA = "05fd5a5afba10871c58d18fde2e31997b85b1c1aa695b828e4ee416c43e43e68"
MODES = ("self_only", "neighbors")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    value = float(value)
    require(math.isfinite(value) and value >= 0, f"Nonfinite/negative {name}")
    return value


def contract_without_mode(contract):
    # Source/data/recipe hashes are not removed. Only the declared mode and its
    # optional aggregate request digest can differ in the otherwise same scope.
    return {k: v for k, v in contract.items()
            if k not in ("message_mode", "request_digest", "contract_digest")}


def load_arm(output, mode, plan):
    summary = read(output / "summary.json")
    require(summary.get("status") == "completed", f"{mode} is not completed")
    config = summary["config"]
    require(summary.get("message_mode") == mode and summary.get("endpoint_step") == 1000,
            f"{mode}: wrong mode/fixed endpoint")
    require(config["contract"]["plan_sha256"] == PLAN_SHA,
            f"{mode}: formal plan identity differs")
    initial = config["initial_parameters"]
    require(initial.get("recorded_before_optimizer_or_resume") is True,
            f"{mode}: initialization witness not before fitting")
    hashes = initial["named_parameter_sha256"]
    require(bool(hashes) and all(isinstance(v, str) and len(v) == 64 for v in hashes.values()),
            f"{mode}: missing complete named initialization hashes")
    require(config["actual_parameter_counts"]["total"] == 7797152,
            f"{mode}: actual parameter count differs")
    require(summary["first_backward_check"]["passed"] is True,
            f"{mode}: first backward failed")
    witnesses = summary["sampling_witnesses"]
    require(isinstance(witnesses, list) and witnesses,
            f"{mode}: ordered sampling witness missing")
    train = summary["final_metrics"]["train"]
    strain = train["strain"]
    expected = plan["scope"]["case_order"]
    require(set(train["per_case"]) == set(expected) and set(strain["per_case"]) == set(expected),
            f"{mode}: all12 TRAIN case inventory differs")
    physical = []
    for key in expected:
        s = strain["per_case"][key]
        rms = finite(s["physical_tensor_error_RMS"], key + " P1 RMS")
        reconstructed = math.sqrt(finite(s["error_energy"], key + " P1 energy") / int(s["nodes"]))
        require(math.isclose(rms, reconstructed, rel_tol=1e-12, abs_tol=1e-15),
                f"{mode}: physical P1 RMS is not sqrt(Frobenius energy/Nnodes)")
        physical.append(rms)
    physical_mean = sum(physical) / len(physical)
    require(math.isclose(physical_mean, strain["mean_case_physical_tensor_error_RMS"],
                         rel_tol=1e-12, abs_tol=1e-15), f"{mode}: physical P1 mean differs")
    elapsed = finite(summary["elapsed_seconds"], mode + " elapsed")
    metrics = {
        "train_u_mean_relative_l2": finite(train["case_mean_relative_l2"], mode + " U mean"),
        "train_u_worst_relative_l2": finite(train["worst_case_relative_l2"], mode + " U worst"),
        "train_P1_mean_relative_l2": finite(strain["case_mean_relative_l2"], mode + " P1 mean"),
        "train_P1_worst_relative_l2": finite(strain["worst_case_relative_l2"], mode + " P1 worst"),
        "train_P1_mean_case_physical_tensor_error_RMS": physical_mean,
        "elapsed_seconds": elapsed,
    }
    return {"summary": summary, "config": config, "metrics": metrics,
            "summary_source": str((output / "summary.json").relative_to(ROOT)),
            "summary_sha256": digest(output / "summary.json")}


def reduction(control, candidate):
    # A perfect control makes the gain undefined, not an epsilon-based pass.
    return None if control == 0 else (control - candidate) / control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("notes/geobc_topology_message_plan.json"))
    parser.add_argument("--self-output", type=Path,
                        default=Path("runs/geobc_exploration/topology_message_self_only_train4_1k"))
    parser.add_argument("--neighbor-output", type=Path,
                        default=Path("runs/geobc_exploration/topology_message_neighbors_train4_1k"))
    parser.add_argument("--output", type=Path, default=Path("notes/geobc_topology_message_pair_result.json"))
    args = parser.parse_args()
    started = time.monotonic()
    path = (ROOT / args.plan).resolve()
    require(digest(path) == PLAN_SHA, "Formal plan changed after predeclaration")
    plan = read(path)
    arms = {mode: load_arm((ROOT / output).resolve(), mode, plan)
            for mode, output in zip(MODES, (args.self_output, args.neighbor_output))}
    a, b = (arms[m] for m in MODES)
    require(contract_without_mode(a["config"]["contract"]) ==
            contract_without_mode(b["config"]["contract"]), "Common source/data/recipe contract differs")
    for key in ("named_parameter_sha256", "all_parameter_sha256", "native_public_parameter_sha256"):
        require(a["config"]["initial_parameters"][key] == b["config"]["initial_parameters"][key],
                f"Initial trainable parameters differ: {key}")
    require(a["config"]["actual_parameter_counts"] == b["config"]["actual_parameter_counts"],
            "Actual stored parameter counts differ")
    require(a["summary"]["sampling_witnesses"] == b["summary"]["sampling_witnesses"],
            "Ordered case/node/cell sampling witnesses differ")
    for key in ("scaler", "u_loss", "optimizer", "cases"):
        require(a["config"][key] == b["config"][key], f"Common fitting configuration differs: {key}")
    x, y = a["metrics"], b["metrics"]
    u_gain = reduction(x["train_u_mean_relative_l2"], y["train_u_mean_relative_l2"])
    p_gain = reduction(x["train_P1_mean_relative_l2"], y["train_P1_mean_relative_l2"])
    deterioration = y["train_u_worst_relative_l2"] - x["train_u_worst_relative_l2"]
    t = plan["clear_gain_gate"]["thresholds"]
    assessments = {
        "u_mean": {"relative_reduction": u_gain, "minimum": t["u_mean_relative_reduction_minimum"],
                   "passed": u_gain is not None and u_gain >= t["u_mean_relative_reduction_minimum"]},
        "P1_mean": {"relative_reduction": p_gain, "minimum": t["recovered_P1_mean_relative_reduction_minimum"],
                    "passed": p_gain is not None and p_gain >= t["recovered_P1_mean_relative_reduction_minimum"]},
        "u_worst": {"deterioration": deterioration, "maximum": t["u_worst_deterioration_maximum"],
                    "passed": deterioration <= t["u_worst_deterioration_maximum"]},
        "physical_P1": {"self_only_RMS": x["train_P1_mean_case_physical_tensor_error_RMS"],
                        "neighbors_RMS": y["train_P1_mean_case_physical_tensor_error_RMS"],
                        "passed": y["train_P1_mean_case_physical_tensor_error_RMS"] <=
                                  x["train_P1_mean_case_physical_tensor_error_RMS"]},
    }
    result = {"status": "completed", "stage": plan["stage"],
              "plan_source": str(path.relative_to(ROOT)), "plan_sha256": PLAN_SHA,
              "analyzer_sha256": digest(Path(__file__)), "matched_controls_passed": True,
              "controls": {"complete_initial_trainables_equal": True, "native_initial_equal": True,
                           "ordered_sampler_witnesses_equal": True, "common_contract_equal": True,
                           "first_backward_passed_both": True, "fixed_endpoint": 1000},
              "arms": {m: {k: v for k, v in arm.items() if k not in ("summary", "config")}
                       for m, arm in arms.items()},
              "clear_gain_gate": assessments,
              "clear_gain_gate_passed": all(g["passed"] for g in assessments.values()),
              "scope": "12 TRAIN cases, fitting utility only; no validation/DIA/finaltest",
              "scientific_limits": plan["scientific_limits"],
              "no_automatic_scaling_or_sweep": True,
              "elapsed_seconds": time.monotonic() - started,
              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "matched_controls_passed": True,
                      "clear_gain_gate_passed": result["clear_gain_gate_passed"],
                      "metrics": {m: arms[m]["metrics"] for m in MODES}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
