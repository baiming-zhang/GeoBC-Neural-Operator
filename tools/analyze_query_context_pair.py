"""CPU-only assessment of the predeclared matched query-context pair."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_gradient_teacher_pair import differences, reduction


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def finite_numbers(value, path="metrics"):
    if isinstance(value, dict):
        for key, item in value.items():
            finite_numbers(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_numbers(item, f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError(f"Nonfinite {path}")


def paired_controls(configs, witnesses, gradients):
    """Check small actual artifacts without loading models, arrays or CUDA."""
    zc, ac = configs["zero"], configs["actual"]
    changed = differences(zc["contract"], ac["contract"])
    allowed = {("requested", "query_context_mode"), ("query_context", "query_context_mode")}
    if set(changed) != allowed:
        raise ValueError(f"Unmatched contract differences: {changed}")
    model_changes = differences(zc["model"], ac["model"])
    allowed_model = {("query_mode",), ("conditioned_field", "query_context", "mode"),
                     ("query_context", "mode")}
    if set(model_changes) != allowed_model:
        raise ValueError(f"Model changes exceed the declared query mode: {model_changes}")
    query_changes = differences(zc["query_context"], ac["query_context"])
    if set(query_changes) != {("query_context_mode",)}:
        raise ValueError(f"Query configuration differences: {query_changes}")
    argument_changes = differences(zc["arguments"], ac["arguments"])
    if set(argument_changes) != {("output",), ("query_context_mode",)}:
        raise ValueError(f"Unmatched CLI arguments: {argument_changes}")
    for key in ("scaler", "initial_parameters", "initial_decoder_parameter_sha256", "cases", "u_loss",
                "geometry", "local_geometry", "gradient_supervision", "optimizer", "precision",
                "sampling_rng", "sampling", "parameter_count", "decoder_parameter_count",
                "checkpoint_selection", "total_steps"):
        if zc[key] != ac[key]:
            raise ValueError(f"Paired {key} differs")
    if witnesses["zero"] != witnesses["actual"]:
        raise ValueError("Case/node/tetra-corner sampling witnesses differ")
    for mode, config in configs.items():
        query = config["query_context"]
        geometry = config["geometry"]
        model = config["model"]
        if (config["arguments"]["query_context_mode"] != mode
                or config["arguments"]["bc_role_mode"] != "actual"
                or query["query_context_mode"] != mode
                or query["sigma"] != .15 or query["dimension"] != 32 or query["query_chunk"] != 4096
                or model["native_identity_condition_dim"] != 102
                or model["query_projection_parameters"] != 2080
                or model["encoder_parameters"] != 16800
                or not geometry["reference_xyz_mask_node_indices_exact_equal"]
                or geometry["bc_role_mode"] != "actual"
                or geometry["raw_roles_sha256"] != geometry["encoder_roles_sha256"]
                or geometry["cloud_features5_sha256"] != query["reference_role_cloud"]["raw_features5_sha256"]
                or geometry["raw_roles_sha256"] != query["reference_role_cloud"]["raw_roles_sha256"]
                or geometry["node_indices_sha256"] != query["reference_role_cloud"]["node_indices_sha256"]):
            raise ValueError(f"{mode}: fixed layout or original actual-role cloud changed")
        check = gradients[mode]
        finite_numbers(check, f"{mode}.first_gradient")
        if (check["step"] != 1 or check["query_context_mode"] != mode
                or not check["projection_and_tokens_gradients_present_and_finite"]
                or not check["decoder_gradients_present_and_finite"]
                or check["encoder_gradient_l1"] <= 0 or check["decoder_gradient_l1"] <= 0
                or not check["context_gradient_l1_per_cad"]
                or sum(check["context_gradient_l1_per_cad"]) <= 0
                or check["query_projection_parameter_count"] != 2080
                or check["projected_tokens_shape"][-2:] != [1024, 32]):
            raise ValueError(f"{mode}: missing native/global/query first-gradient evidence")
        projection = check["query_projection_gradient_l1"]
        tokens = check["projected_tokens_gradient_l1"]
        if mode == "zero":
            if projection != 0 or tokens != 0 or check["expected_gradient"] != "exact_zero":
                raise ValueError("Zero query branch has active projection/token gradients")
        elif projection <= 0 or tokens <= 0 or check["expected_gradient"] != "nonzero":
            raise ValueError("Actual query branch lacks positive projection/token gradients")
    return {"matched_controls_passed": True, "pair_controls_passed": True,
            "matched_contract_difference_paths": [".".join(path) for path in changed],
            "matched_model_difference_paths": [".".join(path) for path in model_changes],
            "sampling_witness_sha256": witnesses["zero"]["witness_sha256"],
            "initial_parameters": zc["initial_parameters"], "raw_geometry_equal": True,
            "first_gradients": gradients}


def gate_assessment(zero, actual, gate):
    zv, av = zero["validation"], actual["validation"]
    zt, at = zero["train"], actual["train"]
    mean_gain = 100 * (zv["case_mean_relative_l2"] - av["case_mean_relative_l2"])
    worst_delta = 100 * (av["worst_case_relative_l2"] - zv["worst_case_relative_l2"])
    strain_gain = reduction(zv["strain"]["case_mean_relative_l2"], av["strain"]["case_mean_relative_l2"])
    checks = {
        "validation_u_mean_gain": mean_gain >= gate["validation_u_mean_improvement_percentage_points_at_least"],
        "validation_u_worst_preserved": worst_delta <= gate["validation_u_worst_deterioration_percentage_points_at_most"],
        "validation_P1_relative_gain": strain_gain >= gate["validation_P1_relative_error_reduction_at_least"],
        "validation_P1_at_most_zero_strain_baseline": av["strain"]["case_mean_relative_l2"] <= gate["validation_P1_case_mean_relative_L2_at_most"],
    }
    return {"validation_u_mean_improvement_percentage_points": mean_gain,
            "validation_u_worst_deterioration_percentage_points": worst_delta,
            "validation_P1_relative_error_reduction": strain_gain,
            "train_u_mean_improvement_percentage_points": 100 * (zt["case_mean_relative_l2"] - at["case_mean_relative_l2"]),
            "train_P1_relative_error_reduction": reduction(zt["strain"]["case_mean_relative_l2"], at["strain"]["case_mean_relative_l2"]),
            "better_than_zero_strain_baseline": av["strain"]["case_mean_relative_l2"] <= 1,
            "predeclared_gate": gate, "gate_checks": checks,
            "clear_gain_gate_passed": all(checks.values())}


def analyze(folders, plan_path):
    began = time.monotonic()
    plan = read(plan_path)
    configs, summaries, metrics, witnesses, gradients, sources = {}, {}, {}, {}, {}, {}
    for mode, folder in folders.items():
        paths = {name: folder / filename for name, filename in {
            "config": "config.json", "summary": "summary.json", "best": "metrics_best.json",
            "latest": "metrics_latest.json", "sampling": "sampling_first_batch.json",
            "gradient": "encoder_gradient_check.json"}.items()}
        records = {name: read(path) for name, path in paths.items()}
        summary, config = records["summary"], records["config"]
        if (summary.get("status") != "completed" or summary["step"] != plan["recipe"]["steps"]
                or summary["config"] != config or summary["best_metrics"] != records["best"]
                or summary["final_metrics"] != records["latest"]
                or summary["encoder_gradient_check"] != records["gradient"]
                or summary["best_step"] != records["best"]["step"]
                or summary["best_score"] != records["best"]["validation"]["case_mean_relative_l2"]
                or records["best"]["selection_score"] != summary["best_score"]
                or records["latest"]["step"] != plan["recipe"]["steps"]):
            raise ValueError(f"{mode}: incomplete or inconsistent summary/metric/config artifacts")
        expected_args = {"seed": plan["recipe"]["seed"], "steps": plan["recipe"]["steps"],
                         "batch_size": plan["recipe"]["batch_size"], "eval_every": plan["recipe"]["eval_every"],
                         "learning_rate": .003, "cases_per_batch": 4, "gradient_weight": plan["recipe"]["gradient_weight"]}
        if any(config["arguments"][name] != value for name, value in expected_args.items()):
            raise ValueError(f"{mode}: actual budget differs from predeclared recipe")
        for epoch in (records["best"], records["latest"]):
            finite_numbers(epoch, f"{mode}.metrics")
            for group in ("train", "validation"):
                declared = {case["sample_key"]: case["nodes"] for case in config["cases"][group]}
                measured = epoch[group]
                if set(measured["per_case"]) != set(declared) or set(measured["strain"]["per_case"]) != set(declared):
                    raise ValueError(f"{mode}: metric case scope differs from config")
                if any(measured["per_case"][key]["nodes"] != count
                       or measured["strain"]["per_case"][key]["nodes"] != count for key, count in declared.items()):
                    raise ValueError(f"{mode}: metrics must cover all original nodes")
        configs[mode], summaries[mode], metrics[mode] = config, summary, records["best"]
        witnesses[mode], gradients[mode] = records["sampling"], records["gradient"]
        sources[mode] = {name: {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in paths.items()}
    controls = paired_controls(configs, witnesses, gradients)
    assessment = gate_assessment(metrics["zero"], metrics["actual"], plan["clear_gain_gate"])
    result = {"status": "completed", "stage": "matched_query_dependent_geometry_BC_context",
              "seed": plan["recipe"]["seed"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
              **controls, **assessment,
              "next_stage": "matched_seed2027_replication" if assessment["clear_gain_gate_passed"] else "one_bounded_query_branch_learning_diagnostic",
              "selection": "Each checkpoint selected only by known-load validation all-node physical displacement mean; P1 assessed at that same checkpoint",
              "interpretation": "The active spatial context increases effective capacity despite identical stored parameters. This tests branch activation, not only missing information. Gaussian geometry distance is not a solved elastic load path; sparse clouds and displacement-derived P1 proxy remain limitations. Development validation is adaptively reused; no novelty/generalization success is implied.",
              "final_test_and_dia_opened": False, "CPU_only": True,
              "analysis_elapsed_seconds": time.monotonic() - began,
              "plan": {"path": str(plan_path.relative_to(ROOT)), "sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest()},
              "source_artifacts": sources, "runs": {}, "cost": {}}
    for mode, summary in summaries.items():
        train, val = summary["best_metrics"]["train"], summary["best_metrics"]["validation"]
        result["runs"][mode] = {"best_step": summary["best_step"], "elapsed_seconds": summary["elapsed_seconds"],
            "train_u_mean_relative_l2": train["case_mean_relative_l2"], "train_u_worst_relative_l2": train["worst_case_relative_l2"],
            "train_P1_mean_relative_l2": train["strain"]["case_mean_relative_l2"], "train_P1_worst_relative_l2": train["strain"]["worst_case_relative_l2"],
            "validation_u_mean_relative_l2": val["case_mean_relative_l2"], "validation_u_worst_relative_l2": val["worst_case_relative_l2"],
            "validation_P1_mean_relative_l2": val["strain"]["case_mean_relative_l2"], "validation_P1_worst_relative_l2": val["strain"]["worst_case_relative_l2"],
            "train": train, "validation": val, "final_metrics": summary["final_metrics"],
            "memory": summary.get("memory"), "parameter_count": configs[mode]["parameter_count"],
            "decoder_parameter_count": configs[mode]["decoder_parameter_count"]}
        result["cost"][mode] = {"elapsed_seconds": summary["elapsed_seconds"], "memory": summary.get("memory")}
    result["cost"]["total_pair_elapsed_seconds"] = sum(s["elapsed_seconds"] for s in summaries.values())
    result["cost"]["actual_to_zero_elapsed_ratio"] = summaries["actual"]["elapsed_seconds"] / summaries["zero"]["elapsed_seconds"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "notes/geobc_query_context_plan.json")
    parser.add_argument("--output", type=Path, default=ROOT / "notes/geobc_query_context_pair_result.json")
    args = parser.parse_args()
    folders = {mode: ROOT / f"runs/geobc_exploration/displacement_query_context_{mode}_10k" for mode in ("zero", "actual")}
    incomplete = [mode for mode, folder in folders.items() if not (folder / "summary.json").is_file()
                  or read(folder / "summary.json").get("status") != "completed"]
    if incomplete:
        print(json.dumps({"status": "waiting", "incomplete": incomplete, "result_written": False}))
        return 2
    result = analyze(folders, args.plan.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({key: result[key] for key in ("matched_controls_passed", "validation_u_mean_improvement_percentage_points",
        "validation_u_worst_deterioration_percentage_points", "validation_P1_relative_error_reduction", "clear_gain_gate_passed", "next_stage")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
