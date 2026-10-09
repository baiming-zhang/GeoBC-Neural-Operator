"""Release the selected single-TRAIN-CAD diagnostic without opening fields."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def main():
    from geobc import specialist_train as specialist
    from geobc import gradient_balance_train as shared
    ready = read(ROOT / "notes/specialist_train_ready.json")
    if ready.get("status") != "ready":
        raise ValueError("Specialist implementation is not ready")
    for relative, expected in ready["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Ready source changed: {relative}")
    design = read(ROOT / "notes/geobc_specialist_plan.json")
    result = read(ROOT / "notes/geobc_bc_role_pair_result.json")
    if design["status"] != "selected_after_failed_BC_gate" or result["clear_gain_gate_passed"]:
        raise ValueError("The prerequisite failed-BC decision is not recorded")
    plan = read(ROOT / "notes/geobc_exploration_plan.json")
    if len(plan["jobs"]) != 1 or plan["jobs"][0]["name"] != "specialist_101_479_joint_u_P1_10k":
        raise ValueError("Exactly one fixed CAD diagnostic is released")
    job = plan["jobs"][0]
    command = job["command"]
    args = specialist.parser().parse_args(command[command.index("geobc.specialist_train") + 1:])
    split, split_path, keys, files, contract = specialist._scope(args)
    expected_keys = ["101_479/ver", "101_479/hor", "101_479/tor"]
    if keys != {"train": expected_keys, "validation": []} or set(files) != set(expected_keys):
        raise ValueError("Actual field scope is not exactly the three original TRAIN labels")
    for name in ("mixed_query_indices", "gradient_batch_loss", "_evaluate_group"):
        if getattr(specialist, name) is not getattr(shared, name):
            raise ValueError(f"Frozen computation not directly reused: {name}")
    controls = contract["specialist"]
    if (controls["fixed_endpoints"] != [250, 10000] or controls["refit_label_scaler_or_weights"]
            or controls["checkpoint_selection"] != "fixed10000_endpoint_no_selection"
            or controls["reference_final_10000"]["step"] != 10000):
        raise ValueError("Fixed endpoint/normalizer/reference recipe changed")
    record = {"status": "ready", "job": job["name"], "source_sha256": contract["code_sha256"],
              "actual_label_scope": expected_keys, "validation_dia_finaltest_labels_opened": False,
              "scaler_case_weights_refit": False, "fixed_endpoints": [250, 10000],
              "reference_step": 10000, "baseline_best8000_not_primary": True,
              "computation_reuse": "Same imported mixed sampler, gradient objective and all-node physical u/P1 evaluator",
              "declared_changes": "One CAD, four case slots with replacement, fixed endpoint evaluations; no validation selection",
              "runtime_checks": "Exact old cloud coordinates/masks/indices and complete fresh initial parameters; native encoder/all-decoder gradients",
              "interpretation": "Dedicated model and40x finalcaseexposure; fitting diagnostic, not operator generalization",
              "review_scope": "Small source/ready hashes and metadata/stat only; no arrays/weights/GPU"}
    path = ROOT / "notes/specialist_parent_review.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": "ready", "actual_label_scope": expected_keys,
                      "fixed_endpoints": [250, 10000], "scaler_case_weights_refit": False}))


if __name__ == "__main__":
    main()
