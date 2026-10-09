"""CPU/JSON-only assessment of the predeclared active interface-distance pair.

Graph is compared only with the fresh matched Euclidean-interface control.
No model, checkpoint, NPZ, field array or GPU is opened. Incomplete runs do
not produce a result. A valid failed scientific gate is a completed analysis.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_hash_modulation_pair import (
    FILES, read, sha, display_path, recorded_path, require, exact_changes,
    valid_digest, check_scope_and_metrics,
)
from tools.analyze_query_context_pair import finite_numbers, gate_assessment

MODES = ("euclidean", "graph")
PLAN_SHA = "cf2a68f0fd09b487bb05092443c01e94ea808a8e9dd366d2b7d519425f5ed273"
NEW_SOURCE_SHA = {
    "graph_context_field.py": "8e88a11b360a4a8d937bdaeec0111d9132aa1c7c05c384fe7fc0ab60449d3194",
    "graph_context_train.py": "52c7f6cc53a7f46fcb250d2815b092d472f94fe411cf68f66ea04560db26e35f",
    "tools/prepare_graph_context.py": "a445c202a01956604254a7410c3e0d96aea7318073e1855ce79582fc4fdedf0a",
}


def check_native_layout(layout):
    expected = {"raw_public_input_dims": 110, "raw_native_input_dims": 110,
        "public_output_dims": 9, "native_padded_output_dims": 16,
        "combined_public_parameter_count": 7770496,
        "combined_public_parameter": "field.decoder.native.params",
        "master_dtype": "torch.float32", "compute_dtype": "torch.float16",
        "Identity_conditions": 107, "nominal_encoded_channels": 131,
        "source_derived_padded_encoded_channels": 144,
        "Identity_source_derived_padding_count": 13,
        "Identity_source_derived_padding_value": 1, "flat_buffer_offsets_inferred": False}
    require(all(layout.get(key) == value for key, value in expected.items()),
            "Actual public combined native layout differs from the frozen43-input architecture")


def native_acceptance(plan):
    """Read existing synthetic CUDA evidence; this analysis runs no native code."""
    ready_path = ROOT / "notes/graph_context_native_ready.json"
    ready = read(ready_path)
    summary_path = recorded_path(ready["summary"])
    summary = read(summary_path)
    contract = summary["config"]["contract"]
    require(ready.get("status") == "ready" and ready["actual_CUDA_passed"] is True
            and ready["plan_sha256"] == contract["plan"]["sha256"] == PLAN_SHA
            and ready["summary_sha256"] == sha(summary_path)
            and ready["source_sha256"] == contract["sources_sha256"]
            and summary.get("status") == "completed"
            and summary["basis_forward_calls"] == 6 and summary["backwards"] == 2
            and summary["optimizer_updates"] == ready["optimizer_updates"] == 0
            and summary["no_GT"] is ready["no_GT"] is True
            and summary["dataset_or_checkpoint_files_opened"] == 0
            and summary["paired_initial_state_exact_equal"] is True
            and ready["paired_complete_parameters_exact_equal"] is True
            and summary["common_identical_distance_fixture_prediction_exact_equal"] is True
            and ready["common_fixture_forward_exact_equal"] is True
            and summary["natural_step0_prediction_equality_required"] is False
            and ready["natural_forward_equality_not_required"] is True
            and ready["force_mask_and_native_encoder_query_gradients_passed"] is True,
            "Missing/inconsistent genuine synthetic native acceptance")
    for mode in MODES:
        check_native_layout(ready["native_layout"][mode])
        require(summary["native_checks"][mode]["native_layout"] == ready["native_layout"][mode]
                and summary["native_checks"][mode]["actual_counts"] == {
                    key: plan["native_source_derivation"]["source_derived_counts"][key]
                    for key in ("native_total", "PointNet", "query_projection", "total_expected_each_model")},
                "Native synthetic count/layout evidence disagrees")
    return {"kind": "synthetic_native_acceptance; separate from the two scientific training runs",
        "status": "completed", "actual_CUDA_passed": True,
        "ready": {"path": display_path(ready_path), "sha256": sha(ready_path)},
        "summary": {"path": display_path(summary_path), "sha256": sha(summary_path)},
        "source_sha256": ready["source_sha256"], "basis_forwards": 6, "backwards": 2,
        "optimizer_updates": 0, "real_dataset_or_checkpoint_files_opened": 0,
        "common_identical_distance_fixture_exact_equal": True,
        "natural_step0_prediction_equality_required": False,
        "layout": ready["native_layout"]["euclidean"]}


def check_initial(config):
    initial = config["initial_parameters"]
    require(initial["recorded_before_optimizer_or_resume"] is True
            and initial["paired_modes"] == list(MODES)
            and initial["paired_architecture_and_seed_required"] is True
            and valid_digest(initial["all_parameter_sha256"])
            and valid_digest(initial["decoder_parameter_sha256"])
            and initial["decoder_parameter_sha256"] == config["initial_decoder_parameter_sha256"],
            "Complete-parameter/public combined-native initial witnesses disagree")


def check_layout(config, mode, plan):
    model, geometry, query = config["model"], config["geometry"], config["query_context"]
    decoder = model["conditioned_field"]["decoder"]
    counts = plan["native_source_derivation"]["source_derived_counts"]
    require(model["distance_mode"] == model["interface_distance"]["mode"] == mode
            and model["query_mode"] == model["bc_role_mode"] == "actual"
            and model["mode"] == "linear_basis" and model["condition_dim"] == 43
            and model["native_condition_dim"] == model["native_identity_condition_dim"] == 107
            and model["native_raw_input_dim"] == 110
            and model["native_nominal_encoded_channels"] == 131
            and model["native_source_derived_padded_encoded_channels"] == 144
            and model["actual_decoder_backend"] == "tcnn_fully_fused"
            and model["native_parameter_buffers"] == ["field.decoder.native.params"]
            and model["hash_modulation_enabled"] is False and model["new_learned_head_parameters"] == 0
            and model["load_feature_policy"] == "zero original25 columns7:13"
            and config["force_moment_slots"] == list(range(7, 13)),
            "Fixed fused basis/active43-input layout changed")
    require(model["interface_distance"]["both_modes_active"] is True
            and model["interface_distance"]["outer_slice"] == [38, 43]
            and model["interface_distance"]["zero_branch"] is False
            and model["conditioned_field"]["condition_dim"] == 75
            and decoder["condition_dim"] == 107 and decoder["input_mode"] == "both"
            and decoder["out_channels"] == 9 and decoder["jit_fusion"] is False
            and decoder["source_commit"] == plan["native_source_derivation"]["official_commit"]
            and decoder["network"] == plan["architecture"]["network"],
            "Both distance5 branches must be active with unchanged official fused decoder")
    hash_config = {"otype": "HashGrid", **plan["architecture"]["hash"], "n_dims_to_encode": 3}
    require(decoder["encoding"] == {"otype": "Composite", "nested": [hash_config,
            {"otype": "Identity", "n_dims_to_encode": 107, "scale": 1.0, "offset": 0.0}]},
            "Fixed Composite HashGrid+Identity107 encoding changed")
    require(config["actual_parameter_counts"] == {"native_combined": counts["native_total"],
            "PointNet": counts["PointNet"], "query_projection": counts["query_projection"],
            "total": counts["total_expected_each_model"]}
            and config["parameter_count"] == model["full_model_parameters"] == counts["total_expected_each_model"]
            and config["decoder_parameter_count"] == model["decoder_parameters"] == counts["native_total"]
            and model["encoder_parameters"] == counts["PointNet"]
            and model["query_projection_parameters"] == counts["query_projection"],
            "Measured public combined/encoder/query parameter counts differ")
    check_native_layout(config["native_layout"])
    require(model["native_layout"] == config["native_layout"], "Two native layout records disagree")
    reference = query["reference_role_cloud"]
    require(query["query_context_mode"] == "actual" and query["sigma"] == .15
            and query["dimension"] == 32 and query["query_chunk"] == 4096
            and geometry["reference_xyz_mask_node_indices_exact_equal"] is True
            and geometry["bc_role_mode"] == "actual" and geometry["cloud_channels"] == 5
            and geometry["cloud_points"] == 1024
            and geometry["raw_roles_sha256"] == geometry["encoder_roles_sha256"] == reference["raw_roles_sha256"]
            and geometry["cloud_features5_sha256"] == reference["raw_features5_sha256"]
            and geometry["node_indices_sha256"] == reference["node_indices_sha256"],
            "Original actual-role/query/global cloud must be shared exactly")
    graph = config["graph_context"]
    require(graph["distance_mode"] == mode and graph["both_distance_modes_active"] is True
            and graph["outer_slice"] == [38, 43] and graph["target_independent"] is True
            and graph["fresh_pair"] is True and graph["historical_control_reused"] is False
            and graph["input_dtype"] == "float32" and graph["fit_or_extra_transform"] is False
            and graph["capacity_limit"] == plan["pair"]["capacity_limit"],
            "Fresh shared active distance-cache contract changed")
    check_initial(config)


def check_gradient(check, mode):
    finite_numbers(check, f"{mode}.first_gradient")
    require(check["step"] == 1 and check["query_context_mode"] == "actual"
            and check["expected_gradient"] == "nonzero"
            and check["checked_after_backward_before_optimizer_step"] is True
            and check["projection_and_tokens_gradients_present_and_finite"] is True
            and check["decoder_gradients_present_and_finite"] is True
            and check["decoder_parameter_tensors"] == 1
            and check["native_combined_gradient_l1"] == check["decoder_gradient_l1"]
            and check["query_projection_parameter_tensors"] == 2
            and check["query_projection_parameter_count"] == 2080
            and check["projected_tokens_shape"][-2:] == [1024, 32]
            and all(check[key] > 0 for key in ("encoder_gradient_l1", "decoder_gradient_l1",
                        "query_projection_gradient_l1", "projected_tokens_gradient_l1"))
            and check["context_gradient_l1_per_cad"] and sum(check["context_gradient_l1_per_cad"]) > 0,
            f"{mode}: missing finite positive whole-native/global/query first gradients")


def paired_controls(configs, witnesses, gradients, plan):
    ec, gc = configs["euclidean"], configs["graph"]
    changed = exact_changes(ec["contract"], gc["contract"],
        {("requested", "distance_mode"), ("graph_context", "distance_mode")}, "contract")
    model_changes = exact_changes(ec["model"], gc["model"],
        {("distance_mode",), ("interface_distance", "mode")}, "model")
    exact_changes(ec["arguments"], gc["arguments"], {("output",), ("distance_mode",)}, "CLI arguments")
    exact_changes(ec["graph_context"], gc["graph_context"], {("distance_mode",)}, "graph configuration")
    exempt = {"contract", "model", "arguments", "graph_context", "started_at_utc"}
    require(set(ec) == set(gc), "Config key sets differ")
    for key in set(ec) - exempt:
        require(ec[key] == gc[key], f"Paired {key} differs")
    require(witnesses["euclidean"] == witnesses["graph"], "Case/node/tetra-corner sampling witnesses differ")
    witness = witnesses["euclidean"]
    require(witness["step"] == 1 and witness["nodes_per_case"] == 12800
            and witness["uniform_nodes_per_case"] == 6400 and witness["uniform_cells_per_case"] == 1600
            and witness["corner_queries_per_case"] == 6400 and len(witness["case_keys"]) == 4
            and len(set(witness["case_keys"])) == 4 and witness["sampler_streams_independent_of_model"] is True
            and valid_digest(witness["witness_sha256"]), "Wrong frozen mixed-query sampling witness")
    train_keys = {item["sample_key"] for item in ec["cases"]["train"]}
    require(set(witness["case_keys"]) <= train_keys, "First batch contains non-TRAIN cases")
    for mode, config in configs.items():
        check_layout(config, mode, plan)
        check_gradient(gradients[mode], mode)
    return {"matched_controls_passed": True, "pair_controls_passed": True,
        "matched_contract_difference_paths": changed, "matched_model_difference_paths": model_changes,
        "sampling_witness_sha256": witness["witness_sha256"], "initial_parameters": ec["initial_parameters"],
        "raw_geometry_equal": True, "first_gradients": gradients, "fresh_fused_pair": True,
        "both_distance_modes_active": True, "historical_control_reused": False,
        "natural_step0_prediction_equality_required": False}


def analyze(folders, plan_path):
    began = time.monotonic()
    plan_path = Path(plan_path).resolve()
    plan, plan_sha = read(plan_path), sha(plan_path)
    require(plan_sha == PLAN_SHA and plan["status"] == "predeclared", "Frozen predeclared plan SHA differs")
    gate = plan["clear_gain_gate"]
    require(gate["validation_u_mean_improvement_percentage_points_at_least"] == 2.0
            and gate["validation_u_worst_deterioration_percentage_points_at_most"] == 2.0
            and gate["validation_P1_relative_error_reduction_at_least"] == .3
            and gate["validation_P1_case_mean_relative_L2_at_most"] == 1.0
            and gate["all_conditions_required"] is True, "Frozen four-way gate changed")
    native = native_acceptance(plan)
    norm_dir = ROOT / "runs/geobc_exploration/displacement_query_context_actual_10k"
    norm_paths = {name: norm_dir / f"{name}.json" for name in ("config", "summary")}
    norm = {name: read(path) for name, path in norm_paths.items()}
    require(norm["summary"].get("status") == "completed" and norm["summary"]["config"] == norm["config"],
            "Direct query-actual TRAIN-only normalizer reference is incomplete/inconsistent")
    configs, summaries, metrics, witnesses, gradients, sources = {}, {}, {}, {}, {}, {}
    for mode in MODES:
        paths = {name: Path(folders[mode]) / filename for name, filename in FILES.items()}
        records = {name: read(path) for name, path in paths.items()}
        summary, config, recipe = records["summary"], records["config"], plan["recipe"]
        require(summary.get("status") == "completed" and summary["step"] == recipe["steps"]
                and summary["distance_mode"] == mode and summary["config"] == config
                and summary["best_metrics"] == records["best"] and summary["final_metrics"] == records["latest"]
                and summary["encoder_gradient_check"] == records["gradient"]
                and summary["best_step"] == records["best"]["step"]
                and summary["best_score"] == records["best"]["validation"]["case_mean_relative_l2"]
                == records["best"]["selection_score"] and records["latest"]["step"] == recipe["steps"]
                and 0 < summary["best_step"] <= recipe["steps"] and summary["best_step"] % recipe["eval_every"] == 0
                and math.isfinite(summary["elapsed_seconds"]) and summary["elapsed_seconds"] > 0,
                f"{mode}: incomplete/inconsistent completed artifacts or u-only selection")
        expected = {"seed": recipe["seed"], "steps": recipe["steps"], "batch_size": recipe["batch_size"],
            "eval_every": recipe["eval_every"], "learning_rate": recipe["optimizer"]["learning_rate"],
            "cases_per_batch": recipe["case_slots"], "gradient_weight": recipe["gradient_weight"],
            "distance_mode": mode, "query_context_mode": "actual", "bc_role_mode": "actual",
            "physics_mode": "linear_basis", "u_loss": "relative_case_mse", "local_mode": "actual", "resume": None}
        require(all(config["arguments"][key] == value for key, value in expected.items())
                and config["total_steps"] == recipe["steps"] and config["optimizer"] == recipe["optimizer"]
                and config["sampling_rng"] == recipe["sampling_rng"]
                and config["gradient_supervision"]["gradient_weight"] == recipe["gradient_weight"]
                and config["gradient_supervision"]["GT_input_features"] is False
                and config["gradient_supervision"]["GT_stress_in_optimizer"] is False,
                f"{mode}: fixed recipe, sampler or target scope differs")
        contract, graph = config["contract"], config["contract"]["graph_context"]
        require({key: config["graph_context"][key] for key in graph} == graph
                and graph["plan"]["sha256"] == plan_sha and graph["plan"]["metadata"] == plan,
                f"{mode}: frozen plan/cache contract differs")
        for name, expected_sha in NEW_SOURCE_SHA.items():
            source_path = ROOT / name if "/" in name else ROOT / "geobc" / name
            require(contract["code_sha256"][name] == expected_sha == sha(source_path),
                    f"{mode}: frozen new source {name} differs")
        require(graph["field_ready"]["source_sha256"] == NEW_SOURCE_SHA["graph_context_field.py"]
                == native["source_sha256"]["geobc/graph_context_field.py"], "Native/field/trainer source evidence differs")
        fixed = contract["fixed_normalizers"]
        for name, path in norm_paths.items():
            require(recorded_path(fixed[f"{name}_path"]) == path.resolve()
                    and fixed[f"{name}_sha256"] == sha(path), f"{mode}: normalizer source {name} differs")
        for name in ("scaler", "u_loss"):
            require(config[name] == fixed[name] == norm["config"][name], f"{mode}: {name} was not directly reused")
        train_keys = {item["sample_key"] for item in config["cases"]["train"]}
        require(config["scaler"]["scale_mm"] == recipe["u_scale_mm"]
                and config["scaler"]["center_mm"] == recipe["u_center_mm"]
                and set(config["scaler"]["fit_case_keys"]) == set(config["u_loss"]["fit_case_keys"]) == train_keys
                and config["groups"] == norm["config"]["groups"], "Train-only normalizer scope changed")
        for epoch in (records["best"], records["latest"]):
            finite_numbers(epoch, f"{mode}.metrics")
        check_scope_and_metrics(config, records, {"scope": {"train_loads": ["ver", "hor", "tor"]}})
        configs[mode], summaries[mode], metrics[mode] = config, summary, records["best"]
        witnesses[mode], gradients[mode] = records["sampling"], records["gradient"]
        sources[mode] = {name: {"path": display_path(path), "sha256": sha(path)} for name, path in paths.items()}
    controls = paired_controls(configs, witnesses, gradients, plan)
    assessment = gate_assessment(metrics["euclidean"], metrics["graph"], gate)
    result = {"status": "completed", "stage": "matched_label_free_mesh_graph_distance_context",
        "seed": plan["recipe"]["seed"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **controls, **assessment, "next_stage": "parent_decision_pending",
        "comparator": {"control": "euclidean", "candidate": "graph"},
        "selection": "Each own checkpoint uses known-load validation15 full-node physical signed-u vector mean only; P1 is measured at the same checkpoint",
        "metric_units": {"u_relative_l2": "dimensionless vector error in physical mm",
            "P1_relative_l2": "dimensionless incident-volume-averaged cornerP1 symmetric Frobenius tensor error; squared shear weight2", "display_percent": "100*relative_l2"},
        "zero_strain_baseline_definition": gate["P1_zero_tensor_baseline"],
        "distance_capacity_limit": plan["pair"]["capacity_limit"],
        "scientific_limits": ["Only the two fresh enlarged fused models are matched; historical smaller query or split-gate models are not controls.",
            "Both five-channel branches are active and use the same actual source groups/L. Distance distributions and conditioning differ; gain would establish representation utility, not a stiffness/topology causal mechanism.",
            "Mesh-edge distance is not elastic stiffness, exact continuous geodesic or remeshing invariance.",
            "P1 is recovered from u, not solver-GT strain or sampled optimization cell squared loss.",
            "Development CAD-validation is adaptively reused; no final-test/dia evidence, statistical significance or algorithm novelty is claimed."],
        "final_test_and_dia_opened": False, "CPU_only": True, "scientific_training_runs": 2,
        "synthetic_native_acceptance": native, "synthetic_native_acceptance_runs_in_this_analysis": 0,
        "plan": {"path": display_path(plan_path), "sha256": plan_sha}, "source_artifacts": sources,
        "fixed_normalizers": {name: {"path": display_path(path), "sha256": sha(path)} for name, path in norm_paths.items()},
        "runs": {}, "cost": {}}
    for mode, summary in summaries.items():
        train, val, config = summary["best_metrics"]["train"], summary["best_metrics"]["validation"], configs[mode]
        result["runs"][mode] = {"best_step": summary["best_step"], "completed_steps": summary["step"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "train_u_mean_relative_l2": train["case_mean_relative_l2"], "train_u_worst_relative_l2": train["worst_case_relative_l2"],
            "train_P1_mean_relative_l2": train["strain"]["case_mean_relative_l2"], "train_P1_worst_relative_l2": train["strain"]["worst_case_relative_l2"],
            "validation_u_mean_relative_l2": val["case_mean_relative_l2"], "validation_u_worst_relative_l2": val["worst_case_relative_l2"],
            "validation_P1_mean_relative_l2": val["strain"]["case_mean_relative_l2"], "validation_P1_worst_relative_l2": val["strain"]["worst_case_relative_l2"],
            "case_counts": {g: len(config["cases"][g]) for g in ("train", "validation")},
            "all_node_counts": {g: sum(v["nodes"] for v in config["cases"][g]) for g in ("train", "validation")},
            "train": train, "validation": val, "final_metrics": summary["final_metrics"],
            "memory": summary.get("memory"), "parameter_count": config["parameter_count"],
            "decoder_parameter_count": config["decoder_parameter_count"], "actual_parameter_counts": config["actual_parameter_counts"]}
        result["cost"][mode] = {"elapsed_seconds": summary["elapsed_seconds"], "memory": summary.get("memory")}
    result["cost"]["total_pair_elapsed_seconds"] = sum(s["elapsed_seconds"] for s in summaries.values())
    result["cost"]["graph_to_euclidean_elapsed_ratio"] = summaries["graph"]["elapsed_seconds"] / summaries["euclidean"]["elapsed_seconds"]
    result["analysis_elapsed_seconds"] = time.monotonic() - began
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "notes/geobc_graph_context_candidate_plan.json")
    parser.add_argument("--euclidean", type=Path, default=ROOT / "runs/geobc_exploration/displacement_interface_euclidean_10k")
    parser.add_argument("--graph", type=Path, default=ROOT / "runs/geobc_exploration/displacement_interface_graph_10k")
    parser.add_argument("--output", type=Path, default=ROOT / "notes/geobc_graph_context_pair_result.json")
    args = parser.parse_args()
    folders = {mode: getattr(args, mode).resolve() for mode in MODES}
    incomplete = [mode for mode, folder in folders.items() if not (folder / "summary.json").is_file()
                  or read(folder / "summary.json").get("status") != "completed"]
    if incomplete:
        print(json.dumps({"status": "waiting", "incomplete": incomplete, "result_written": False}))
        return 2
    try:
        result = analyze(folders, args.plan.resolve())
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({"status": "rejected", "error": f"{type(error).__name__}: {error}", "result_written": False}))
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in ("matched_controls_passed", "validation_u_mean_improvement_percentage_points",
        "validation_u_worst_deterioration_percentage_points", "validation_P1_relative_error_reduction", "clear_gain_gate_passed", "next_stage")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
