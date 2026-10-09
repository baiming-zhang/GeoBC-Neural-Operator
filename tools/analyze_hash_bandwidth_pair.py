"""CPU/JSON-only assessment of the frozen full12 versus fixed coarse8 pair.

No arrays, checkpoints, models or GPU are opened. Only completed fresh split
models can be compared. Fixed mask buffers may differ; named trainable
parameters and both public native buffers must have identical initial hashes.
Natural initial predictions need not be equal. A failed scientific gate is a
completed result; incomplete or unmatched runs never write a result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import struct
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

MODES = ("full12", "coarse8")
PLAN_SHA = "8fc4e3355c8892f143428383049e39fa6f0fe380968cbefd59bb3b3058e42f12"
COMMIT = "3daa6e5b8b8e0fa17971a1a20acfbb3754494b38"


def check_mask(mask, mode):
    active = 12 if mode == "full12" else 8
    values = [1.0] * (2 * active) + [0.0] * (24 - 2 * active)
    require(mask["mode"] == mode and mask["values"] == values
            and mask["active_levels"] == active
            and mask["buffer"] == "feature_mask" and mask["learned"] is False,
            f"{mode}: only the predeclared nonlearned final8-channel mask may differ")
    if "sha256" in mask:
        import hashlib
        require(mask["sha256"] == hashlib.sha256(struct.pack("<24f", *values)).hexdigest(),
                f"{mode}: fixed mask value/hash record differs")


def check_layout(config, mode, plan):
    model, graph, query, geometry = (config[k] for k in ("model", "graph_context", "query_context", "geometry"))
    decoder = model["decoder_implementation"]
    require(decoder == model["conditioned_field"]["decoder"], "Duplicate decoder implementation records disagree")
    require(model["hash_bandwidth_mode"] == model["hash_bandwidth"]["mode"] == mode
            and model["query_mode"] == model["bc_role_mode"] == "actual"
            and model["mode"] == "linear_basis" and model["condition_dim"] == 43
            and model["distance_mode"] == graph["distance_mode"] == "euclidean"
            and model["load_feature_policy"] == "zero original25 columns7:13"
            and config["force_moment_slots"] == list(range(7, 13))
            and model["actual_decoder_backend"] == decoder["backend"] == "tcnn_split_encoding_network_hash_bandwidth"
            and model["native_condition_dim"] == model["native_identity_condition_dim"] == 107
            and model["native_public_network_input_dim"] == 144
            and model["native_parameter_buffers"] == ["field.decoder.encoding.params", "field.decoder.network.params"]
            and model["hash_modulation_enabled"] is False, "Fixed split basis/Euclidean43 input layout changed")
    require(decoder["source_commit"] == COMMIT and decoder["jit_fusion"] is False
            and decoder["learned_gate"] is False and decoder["LoD_used"] is False
            and decoder["new_learned_parameters"] == 0 and decoder["native_layout_verified"] is True
            and decoder["flat_buffer_offsets_inferred"] is False
            and decoder["encoding"]["class"] == "tinycudann.Encoding"
            and decoder["encoding"]["n_input_dims"] == 3 and decoder["encoding"]["n_output_dims"] == 24
            and decoder["encoding"]["encoding_config"] == {"otype": "HashGrid", **plan["architecture"]["hash"]}
            and decoder["network"]["class"] == "tinycudann.Network" and decoder["network"]["n_input_dims"] == 144
            and decoder["network"]["n_output_dims"] == 9 and decoder["network"]["network_config"] == plan["architecture"]["network"],
            "Official public split HashGrid/Network144 configuration changed")
    check_mask(decoder["mask"], mode)
    check_mask(config["bandwidth_mask"], mode)
    require(config["bandwidth_mask"]["shape"] == [24] and config["bandwidth_mask"]["trainable"] is False
            and config["bandwidth_mask"]["excluded_from_trainable_parameter_equality"] is True
            and config["bandwidth_mask"]["full_state_equality_claimed"] is False
            and config["initial_decoder_parameters"]["fixed_feature_mask"] == config["bandwidth_mask"],
            "Mode-specific nonlearned mask state must be recorded outside trainable parameter equality")
    counts = plan["architecture"]["source_derived_counts"]
    expected_counts = {"encoding": counts["encoding"], "network": counts["network"],
        "native_total": counts["total_native"], "PointNet": counts["PointNet"],
        "query_projection": counts["query_projection"], "total": counts["total_model"]}
    require(config["actual_parameter_counts"] == expected_counts
            and config["parameter_count"] == model["full_model_parameters"] == counts["total_model"]
            and config["decoder_parameter_count"] == counts["total_native"]
            and model["encoder_parameters"] == counts["PointNet"]
            and model["query_projection_parameters"] == counts["query_projection"], "Actual stored native/model parameter counts changed")
    layout = config["native_layout"]
    require(model["native_layout"] == layout and layout["encoding_parameters"] == counts["encoding"]
            and layout["network_parameters"] == counts["network"]
            and layout["native_total_parameters"] == counts["total_native"]
            and layout["full_model_parameters"] == counts["total_model"]
            and layout["encoding_public_output_dims"] == 24 and layout["network_public_input_dims"] == 144
            and layout["network_native_padded_output_dims"] == 16
            and layout["public_parameter_buffers"] == ["encoding.params", "network.params"]
            and layout["explicit_one_padding_channels"] == 13 and layout["master_dtype"] == "torch.float32"
            and layout["compute_dtype"] == "torch.float16" and layout["flat_buffer_offsets_inferred"] is False,
            "Public native split dimensions/counts/padding differ")
    reference = query["reference_role_cloud"]
    require(query["query_context_mode"] == "actual" and query["sigma"] == .15
            and query["dimension"] == 32 and query["query_chunk"] == 4096
            and geometry["reference_xyz_mask_node_indices_exact_equal"] is True
            and geometry["bc_role_mode"] == "actual" and geometry["cloud_channels"] == 5
            and geometry["cloud_points"] == 1024
            and geometry["raw_roles_sha256"] == geometry["encoder_roles_sha256"] == reference["raw_roles_sha256"]
            and geometry["cloud_features5_sha256"] == reference["raw_features5_sha256"]
            and geometry["node_indices_sha256"] == reference["node_indices_sha256"]
            and graph["manifest_sha256"] == plan["data"]["manifest_sha256"]
            and graph["fit_or_extra_transform"] is False, "Same actual Euclidean/role/query/global inputs must be reused")
    check_initial(config, counts)


def paired_controls(configs, witnesses, gradients, plan):
    fc, cc = configs["full12"], configs["coarse8"]
    changed = exact_changes(fc["contract"], cc["contract"],
        {("requested", "bandwidth_mode"), ("hash_bandwidth", "bandwidth_mode")}, "contract")
    model_allowed = {("hash_bandwidth_mode",), ("hash_bandwidth", "mode"),
                     ("hash_bandwidth", "active_levels"), ("hash_bandwidth", "nonlearned_mask")}
    for prefix in (("conditioned_field", "decoder", "mask"), ("decoder_implementation", "mask")):
        model_allowed |= {prefix + ("mode",), prefix + ("active_levels",)}
        model_allowed |= {prefix + ("values", str(index)) for index in range(16, 24)}
    model_changes = exact_changes(fc["model"], cc["model"], model_allowed, "model")
    exact_changes(fc["arguments"], cc["arguments"], {("output",), ("bandwidth_mode",)}, "CLI arguments")
    exact_changes(fc["hash_bandwidth"], cc["hash_bandwidth"], {("bandwidth_mode",)}, "bandwidth configuration")
    mask_allowed = {("mode",), ("active_levels",)} | {("values", str(index)) for index in range(16, 24)}
    if "sha256" in fc["bandwidth_mask"]:
        mask_allowed.add(("sha256",))
    exact_changes(fc["bandwidth_mask"], cc["bandwidth_mask"], mask_allowed, "nonlearned mask records")
    require({key: value for key, value in fc["initial_decoder_parameters"].items() if key != "fixed_feature_mask"}
            == {key: value for key, value in cc["initial_decoder_parameters"].items() if key != "fixed_feature_mask"},
            "Initial native/decoder trainable witnesses differ beyond the separately recorded fixed mask")
    exempt = {"contract", "model", "arguments", "hash_bandwidth", "bandwidth_mask", "initial_decoder_parameters", "started_at_utc"}
    require(set(fc) == set(cc), "Config key sets differ")
    for key in set(fc) - exempt:
        require(fc[key] == cc[key], f"Paired {key} differs")
    require(witnesses["full12"] == witnesses["coarse8"], "Case/node/tetra-corner sampling witnesses differ")
    witness = witnesses["full12"]
    require(witness["step"] == 1 and witness["nodes_per_case"] == 12800
            and witness["uniform_nodes_per_case"] == 6400 and witness["uniform_cells_per_case"] == 1600
            and witness["corner_queries_per_case"] == 6400 and len(witness["case_keys"]) == 4
            and len(set(witness["case_keys"])) == 4 and witness["sampler_streams_independent_of_model"] is True
            and valid_digest(witness["witness_sha256"]), "Wrong fixed mixed-query sampler witness")
    require(set(witness["case_keys"]) <= {item["sample_key"] for item in fc["cases"]["train"]}, "Non-TRAIN first-batch case")
    counts = plan["architecture"]["source_derived_counts"]
    for mode, config in configs.items():
        contract_bandwidth = config["contract"]["hash_bandwidth"]
        require(config["hash_bandwidth"] == contract_bandwidth
                and contract_bandwidth["plan"]["sha256"] == PLAN_SHA
                and contract_bandwidth["plan"]["metadata"] == plan,
                f"{mode}: frozen bandwidth source contract differs")
        for filename, ready_key in (("geobc/hash_bandwidth_field.py", "source_sha256"),
                                    ("tools/validate_hash_bandwidth_native.py", "smoke_sha256")):
            ready = read(ROOT / "notes/hash_bandwidth_field_ready.json")
            require(ready["status"] == "ready" and ready["plan_sha256"] == PLAN_SHA
                    and sha(ROOT / filename) == ready[ready_key], "Frozen field/smoke ready source changed")
        check_layout(config, mode, plan)
        check_gradient(gradients[mode], mode, counts)
    return {"matched_controls_passed": True, "pair_controls_passed": True,
        "matched_contract_difference_paths": changed, "matched_model_difference_paths": model_changes,
        "sampling_witness_sha256": witness["witness_sha256"], "initial_parameters": fc["initial_parameters"],
        "initial_decoder_parameters": fc["initial_decoder_parameters"], "nonlearned_masks": {mode: configs[mode]["bandwidth_mask"] for mode in MODES},
        "nonlearned_mask_buffers_equal": False, "raw_geometry_and_Euclidean_inputs_equal": True,
        "first_gradients": gradients, "fresh_split_pair": True, "historical_control_reused": False,
        "natural_step0_prediction_equality_required": False}


def native_acceptance(configs, plan):
    """Only reads the root-produced real-CUDA synthetic acceptance JSONs."""
    ready_path = ROOT / "notes/hash_bandwidth_native_ready.json"
    ready = read(ready_path)
    summary_path = recorded_path(ready["summary"])
    summary = read(summary_path)
    contract = summary["config"]["contract"]
    require(ready.get("status") == "ready" and ready["plan_sha256"] == contract["plan"]["sha256"] == PLAN_SHA
            and ready["summary_sha256"] == sha(summary_path)
            and ready["source_sha256"] == summary["source_sha256"] == contract["sources_sha256"]
            and summary.get("status") == "completed" and summary["backwards"] == 2
            and summary["optimizer_updates"] == ready["optimizer_updates"] == 0
            and summary["GT_fields_read"] == ready["GT_fields_read"] == 0
            and summary["dataset_or_checkpoint_files_opened"] == 0
            and summary["paired_named_trainable_initial_exact_equal"] is True
            and ready["paired_trainable_init_exact"] is True and ready["mask_buffer_difference_declared"] is True
            and ready["public_native_encoder_query_and_inactive_activation_gradients_passed"] is True
            and summary["nonlearned_mask_buffers_equal"] is False
            and summary["natural_step0_prediction_equality_required"] is False
            and summary["initial_parameters"]["full12"] == summary["initial_parameters"]["coarse8"]
            and summary["nonlearned_mask_buffers"] == {"full12": [1.0] * 24, "coarse8": [1.0] * 16 + [0.0] * 8},
            "Missing/inconsistent genuine synthetic split-native acceptance")
    counts = plan["architecture"]["source_derived_counts"]
    initial = summary["initial_parameters"]["full12"]
    require(valid_digest(initial["complete_named_trainable_sha256"]) and initial["total_parameters"] == counts["total_model"],
            "Synthetic complete-trainable witness/count differs")
    for mode in MODES:
        code = configs[mode]["contract"]["code_sha256"]
        for filename in ("hash_bandwidth_field.py", "hash_bandwidth_train.py"):
            expected = code[filename]
            require(sha(ROOT / "geobc" / filename) == expected, f"Frozen {filename} source changed")
        require(code["hash_bandwidth_field.py"] == ready["source_sha256"]["geobc/hash_bandwidth_field.py"],
                "Native/training field source identity differs")
        record = summary["native_checks"][mode]
        require(record["layout"] == ready["native_layout"][mode] == configs[mode]["native_layout"]
                and record["force_slot_invariant"] is True, "Native/training public layout/force-mask checks differ")
        finite_numbers(record, f"native.{mode}")
        for name in ("encoding", "network"):
            group = record[f"native_{name}_gradient"]
            require(group["parameter_tensors"] == 1 and group["parameters"] == counts[name]
                    and group["present_finite"] is True and group["gradient_L1"] > 0,
                    "Synthetic public native group lacks finite positive first gradient")
        activation = record["activation_gradient"]
        require(activation["all_present_and_finite"] is True and activation["no_per_level_parameter_offsets"] is True
                and activation["active_first16_gradient_L1"] > 0 and activation["pre_mask_fine_last8_value_L1"] > 0
                and ((activation["fine_last8_gradient_L1"] == 0 and activation["post_mask_fine_last8_value_L1"] == 0)
                     if mode == "coarse8" else activation["fine_last8_gradient_L1"] > 0),
                "Synthetic pre-mask fine activation gradient/value checks differ")
    return {"kind": "genuine CUDA synthetic public-API/gradient acceptance, separate from two scientific training runs",
        "status": "completed", "ready": {"path": display_path(ready_path), "sha256": sha(ready_path)},
        "summary": {"path": display_path(summary_path), "sha256": sha(summary_path)},
        "source_sha256": ready["source_sha256"], "backwards": 2, "optimizer_updates": 0,
        "real_dataset_or_checkpoint_files_opened": 0, "GT_fields_read": 0,
        "complete_named_trainable_initial_exact_equal": True, "nonlearned_mask_buffers_equal": False,
        "natural_step0_prediction_equality_required": False,
        "layout": ready["native_layout"]["full12"]}


def check_initial(config, counts):
    initial, decoder = config["initial_parameters"], config["initial_decoder_parameters"]
    groups = decoder["native_groups"]
    require(set(groups) == {"encoding", "network"}, "Exactly two public native parameter buffers are required")
    for name in ("encoding", "network"):
        require(len(groups[name]) == 1, f"Expected one public {name} buffer")
        item = groups[name][0]
        require(item["count"] == counts[name] and math.prod(item["shape"]) == item["count"]
                and item["dtype"] == "torch.float32" and valid_digest(item["sha256"]),
                f"Invalid initial public {name} buffer witness")
    require(initial["recorded_before_optimizer_or_resume"] is True
            and valid_digest(initial["all_parameter_sha256"])
            and initial["decoder_parameter_sha256"] == decoder["all_decoder_parameter_sha256"]
            == config["initial_decoder_parameter_sha256"]
            and valid_digest(initial["decoder_parameter_sha256"])
            and initial["native_public_parameter_sha256"] == {key: groups[key][0]["sha256"] for key in groups}
            and initial["paired_modes"] == list(MODES) and initial["paired_architecture_and_seed_required"] is True
            and initial["nonlearned_mask_buffer_excluded"] is True and initial["full_state_equality_claimed"] is False
            and decoder["flat_buffer_offsets_inferred"] is False,
            "Complete named-trainable/native initialization records disagree")


def check_gradient(check, mode, counts):
    finite_numbers(check, f"{mode}.first_gradient")
    require(check["step"] == 1 and check["query_context_mode"] == "actual"
            and check["expected_gradient"] == "nonzero"
            and check["checked_after_backward_before_optimizer_step"] is True
            and check["projection_and_tokens_gradients_present_and_finite"] is True
            and check["decoder_gradients_present_and_finite"] is True
            and check["query_projection_parameter_tensors"] == 2
            and check["query_projection_parameter_count"] == 2080
            and check["projected_tokens_shape"][-2:] == [1024, 32]
            and all(check[key] > 0 for key in ("encoder_gradient_l1", "decoder_gradient_l1",
                        "query_projection_gradient_l1", "projected_tokens_gradient_l1"))
            and check["context_gradient_l1_per_cad"] and sum(check["context_gradient_l1_per_cad"]) > 0,
            f"{mode}: missing finite positive shared native/global/query first-gradient evidence")
    require(set(check["native_parameter_groups"]) == {"encoding", "network"}, "Native gradient group scope changed")
    for name in ("encoding", "network"):
        group = check["native_parameter_groups"][name]
        require(group["gradients_present_and_finite"] is True and group["gradient_l1"] > 0
                and group["parameter_tensors"] == 1 and group["parameters"] == counts[name],
                f"{mode}: public {name} gradients must be finite and positive")
    activation = check["bandwidth_activation_gradient"]
    require(check["bandwidth_mode"] == mode and activation["pre_mask_first16_gradient_l1"] > 0
            and activation["pre_mask_gradient_not_post_mask_gradient"] is True
            and activation["checked_after_backward_before_optimizer_step"] is True,
            "Mask-gradient scope must be retained pre-mask output, with active first16 positive")
    require((activation["pre_mask_last8_gradient_l1"] == 0
             and activation["post_mask_last8_activation_max_abs"] == 0) if mode == "coarse8"
            else activation["pre_mask_last8_gradient_l1"] > 0,
            f"{mode}: last8 retained activation gradients/values violate the fixed mask policy")


def analyze(folders, plan_path):
    began = time.monotonic()
    plan_path = Path(plan_path).resolve()
    plan, plan_sha = read(plan_path), sha(plan_path)
    require(plan_sha == PLAN_SHA and plan["status"] == "predeclared", "Frozen predeclared plan SHA differs")
    gate, recipe = plan["clear_gain_gate"], plan["recipe"]
    require(gate["validation_u_mean_improvement_percentage_points_at_least"] == 2.0
            and gate["validation_u_worst_deterioration_percentage_points_at_most"] == 2.0
            and gate["validation_P1_relative_error_reduction_at_least"] == .3
            and gate["validation_P1_case_mean_relative_L2_at_most"] == 1.0
            and gate["all_conditions_required"] is True, "Frozen four-way gate changed")
    norm_dir = ROOT / plan["data"]["fixed_normalizers_directory"]
    norm_paths = {name: norm_dir / f"{name}.json" for name in ("config", "summary")}
    norm = {name: read(path) for name, path in norm_paths.items()}
    require(norm["summary"].get("status") == "completed" and norm["summary"]["config"] == norm["config"],
            "Direct query-actual TRAIN-only normalizer reference is incomplete/inconsistent")
    configs, summaries, metrics, witnesses, gradients, sources = {}, {}, {}, {}, {}, {}
    for mode in MODES:
        paths = {name: Path(folders[mode]) / filename for name, filename in FILES.items()}
        records = {name: read(path) for name, path in paths.items()}
        summary, config = records["summary"], records["config"]
        require(summary.get("status") == "completed" and summary["step"] == recipe["steps"]
                and summary["bandwidth_mode"] == mode and summary["config"] == config
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
            "bandwidth_mode": mode, "distance_mode": "euclidean", "query_context_mode": "actual", "bc_role_mode": "actual",
            "physics_mode": "linear_basis", "u_loss": "relative_case_mse", "local_mode": "actual", "resume": None}
        require(all(config["arguments"][key] == value for key, value in expected.items())
                and config["total_steps"] == recipe["steps"] and config["optimizer"] == recipe["optimizer"]
                and config["sampling_rng"] == recipe["sampling_rng"]
                and config["gradient_supervision"]["gradient_weight"] == recipe["gradient_weight"]
                and config["gradient_supervision"]["GT_input_features"] is False
                and config["gradient_supervision"]["GT_stress_in_optimizer"] is False,
                f"{mode}: fixed recipe, sampler or target scope differs")
        fixed = config["contract"]["fixed_normalizers"]
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
    native = native_acceptance(configs, plan)
    assessment = gate_assessment(metrics["full12"], metrics["coarse8"], gate)
    result = {"status": "completed", "stage": "matched_fixed_hash_bandwidth", "seed": recipe["seed"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(), **controls, **assessment,
        "next_stage": "parent_decision_pending", "comparator": {"control": "full12", "candidate": "coarse8"},
        "selection": "Each own checkpoint uses known-load validation15 full-node physical signed-u vector mean only; P1 is measured at the same checkpoint",
        "metric_units": {"u_relative_l2": "dimensionless vector error in physical mm",
            "P1_relative_l2": "dimensionless incident-volume-averaged cornerP1 symmetric Frobenius tensor error; squared shear weight2", "display_percent": "100*relative_l2"},
        "zero_strain_baseline_definition": gate["P1_zero_tensor_baseline"],
        "effective_capacity_limit": plan["pair"]["capacity_limit"],
        "scientific_limits": plan["scientific_limits"] + [
            "Only the two fresh split-native models are matched; historical fused or learned-gate models are contextual only.",
            "Equal stored trainable parameters do not imply equal usable capacity. The fixed post-encoding mask also changes gradient paths and optimization; it is not an encoding speedup.",
            "Natural initial predictions may differ. Mode-specific nonlearned mask buffers are not claimed equal.",
            "Activation-gradient masking is measured on the public encoded output, without guessing Hash parameter partitions."],
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
            "case_counts": {group: len(config["cases"][group]) for group in ("train", "validation")},
            "all_node_counts": {group: sum(item["nodes"] for item in config["cases"][group]) for group in ("train", "validation")},
            "train": train, "validation": val, "final_metrics": summary["final_metrics"], "memory": summary.get("memory"),
            "parameter_count": config["parameter_count"], "decoder_parameter_count": config["decoder_parameter_count"],
            "actual_parameter_counts": config["actual_parameter_counts"]}
        result["cost"][mode] = {"elapsed_seconds": summary["elapsed_seconds"], "memory": summary.get("memory")}
    result["cost"]["total_pair_elapsed_seconds"] = sum(summary["elapsed_seconds"] for summary in summaries.values())
    result["cost"]["coarse8_to_full12_elapsed_ratio"] = summaries["coarse8"]["elapsed_seconds"] / summaries["full12"]["elapsed_seconds"]
    result["analysis_elapsed_seconds"] = time.monotonic() - began
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "notes/geobc_hash_bandwidth_plan.json")
    parser.add_argument("--full12", type=Path, default=ROOT / "runs/geobc_exploration/displacement_hash_full12_10k")
    parser.add_argument("--coarse8", type=Path, default=ROOT / "runs/geobc_exploration/displacement_hash_coarse8_10k")
    parser.add_argument("--output", type=Path, default=ROOT / "notes/geobc_hash_bandwidth_pair_result.json")
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
