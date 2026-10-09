"""CPU/JSON-only assessment of the frozen fresh split-native hash-gate pair.

No model, weight, NPZ, target array, or GPU is opened. Incomplete runs produce
a waiting response and do not write a result. A valid failed scientific gate
is still a completed analysis; an unmatched pair is an error.
"""
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
from tools.analyze_gradient_teacher_pair import differences
from tools.analyze_query_context_pair import finite_numbers, gate_assessment

MODES = ("zero", "actual")
FILES = {"config": "config.json", "summary": "summary.json", "best": "metrics_best.json",
         "latest": "metrics_latest.json", "sampling": "sampling_first_batch.json",
         "gradient": "encoder_gradient_check.json"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def display_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def recorded_path(value):
    """Resolve metadata paths emitted by either host; never open array paths."""
    value = str(value)
    if sys.platform == "win32" and value.startswith("/mnt/") and len(value) > 7 and value[6] == "/":
        value = value[5].upper() + ":/" + value[7:]
    elif sys.platform != "win32" and len(value) > 3 and value[1:3] in (":/", ":\\"):
        value = "/mnt/" + value[0].lower() + "/" + value[3:].replace("\\", "/")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_changes(left, right, allowed, name):
    changed = set(differences(left, right))
    require(changed == allowed, f"Unmatched {name} differences: {sorted(changed)}")
    return [".".join(path) for path in sorted(changed)]


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def check_initial(config, counts):
    initial, decoder = config["initial_parameters"], config["initial_decoder_parameters"]
    groups = decoder["native_groups"]
    require(set(groups) == {"encoding", "network"}, "Expected exactly two independent native buffers")
    for name, count_key in (("encoding", "hash"), ("network", "network")):
        values = groups[name]
        require(len(values) == 1, f"Expected one public native {name} tensor")
        item = values[0]
        require(item["count"] == counts[count_key] and math.prod(item["shape"]) == item["count"]
                and item["dtype"] == "torch.float32" and valid_digest(item["sha256"]),
                f"Invalid initial {name} buffer witness")
    gate = decoder["gate_head"]
    require(gate["parameters"] == counts["gate_head"] and gate["W_and_bias_exact_zero"] is True
            and gate["sha256"] == hashlib.sha256(bytes(4 * counts["gate_head"])).hexdigest(),
            "Gate weight/bias must be exact FP32 zeros before training")
    require(initial["recorded_before_optimizer_or_resume"] is True
            and initial["paired_zero_actual_require_same_architecture_and_seed"] is True
            and valid_digest(initial["all_parameter_sha256"])
            and initial["decoder_parameter_sha256"] == decoder["all_decoder_parameter_sha256"]
            == config["initial_decoder_parameter_sha256"]
            and valid_digest(initial["decoder_parameter_sha256"])
            and initial["native_buffer_sha256"] == {k: v[0]["sha256"] for k, v in groups.items()}
            and initial["gate_head_parameter_sha256"] == gate["sha256"],
            "Complete/native/gate initial hash records are inconsistent")


def check_layout(config, mode, plan):
    model, geometry, query = config["model"], config["geometry"], config["query_context"]
    counts = plan["native_backend"]["source_derived_counts"]
    decoder = model["decoder_implementation"]
    require(decoder == model["conditioned_field"]["decoder"], "Decoder source records disagree")
    require(model["hash_modulation_mode"] == mode and model["hash_modulation"]["mode"] == mode
            and model["query_mode"] == "actual" and model["bc_role_mode"] == "actual"
            and model["mode"] == "linear_basis" and model["condition_dim"] == 38
            and model["native_condition_dim"] == 102 and model["native_identity_condition_dim"] == 126
            and model["full_native_network_input_dim"] == 126
            and model["load_feature_policy"] == "zero original25 columns7:13"
            and config["force_moment_slots"] == list(range(7, 13)), "Fixed basis/condition layout changed")
    require(decoder["backend"] == model["actual_decoder_backend"] == "tcnn_split_encoding_network_hash_gate"
            and decoder["native_buffers"] == ["encoding.params", "network.params"]
            and model["native_parameter_buffers"] == ["field.decoder.encoding.params", "field.decoder.network.params"]
            and decoder["fake_flat_buffer"] is False and decoder["native_layout_verified"] is True
            and decoder["source_commit"] == plan["native_backend"]["official_commit"],
            "Actual source_config is not the fresh official split-native backend")
    encoding, network, gate = decoder["encoding"], decoder["network"], decoder["gate"]
    require(encoding["class"] == "tinycudann.Encoding" and encoding["n_input_dims"] == 3
            and encoding["n_output_dims"] == 24 and encoding["parameters"] == counts["hash"]
            and encoding["encoding_config"]["otype"] == "HashGrid"
            and network["class"] == "tinycudann.Network" and network["n_input_dims"] == 126
            and network["internal_Identity_padded_input"] == 128
            and network["input_padding_count"] == 2 and network["input_padding_value"] == 1
            and network["requested_output_dims"] == 9 and network["native_padded_output_dims"] == 16
            and network["parameters"] == counts["network"], "Native buffers/dimensions/padding changed")
    require(gate["mode"] == mode and gate["head"] == [64, 24] and gate["parameters"] == 1560
            and gate["context_slice"] == [38, 102] and gate["context_order"] == ["query32", "global32"]
            and gate["formula"] == "hash * (1 + tanh(gate(context)))"
            and gate["weight_bias_initialization"] == "exact0" and gate["scale_open_bounds"] == [0, 2],
            "Predeclared unique gamma=1+tanh gate changed")
    require(config["actual_parameter_counts"] == counts
            and config["parameter_count"] == model["full_model_parameters"] == counts["total_expected_each_model"]
            and config["decoder_parameter_count"] == counts["native_total"] + counts["gate_head"]
            and model["encoder_parameters"] == counts["PointNet"]
            and model["query_projection_parameters"] == counts["query_projection"], "Runtime parameter counts changed")
    reference = query["reference_role_cloud"]
    require(query["query_context_mode"] == "actual" and query["sigma"] == .15
            and query["dimension"] == 32 and query["query_chunk"] == 4096
            and geometry["reference_xyz_mask_node_indices_exact_equal"] is True
            and geometry["bc_role_mode"] == "actual" and geometry["cloud_channels"] == 5
            and geometry["cloud_points"] == 1024
            and geometry["raw_roles_sha256"] == geometry["encoder_roles_sha256"] == reference["raw_roles_sha256"]
            and geometry["cloud_features5_sha256"] == reference["raw_features5_sha256"]
            and geometry["node_indices_sha256"] == reference["node_indices_sha256"],
            "Both branches must reuse the same actual-role/query cloud")
    check_initial(config, counts)


def check_gradient(check, mode, counts):
    finite_numbers(check, f"{mode}.first_gradient")
    require(check["step"] == 1 and check["hash_modulation_mode"] == mode
            and check["query_context_mode"] == "actual" and check["expected_gradient"] == "nonzero"
            and check["checked_after_backward_before_optimizer_step"] is True
            and check["projection_and_tokens_gradients_present_and_finite"] is True
            and check["decoder_gradients_present_and_finite"] is True
            and check["gate_head_gradients_present_and_finite"] is True
            and check["query_projection_parameter_count"] == 2080
            and check["gate_head_parameter_count"] == 1560
            and check["projected_tokens_shape"][-2:] == [1024, 32]
            and all(check[key] > 0 for key in ("encoder_gradient_l1", "decoder_gradient_l1",
                                               "query_projection_gradient_l1", "projected_tokens_gradient_l1"))
            and check["context_gradient_l1_per_cad"] and sum(check["context_gradient_l1_per_cad"]) > 0,
            f"{mode}: missing finite positive common native/global/query first gradients")
    require(set(check["native_parameter_groups"]) == {"encoding", "network"}, "Native gradient group scope changed")
    for name, count_key in (("encoding", "hash"), ("network", "network")):
        value = check["native_parameter_groups"][name]
        require(value["present_and_finite"] is True and value["gradient_l1"] > 0
                and value["parameter_tensors"] == 1 and value["parameters"] == counts[count_key],
                f"{mode}: {name} buffer does not have a finite positive gradient")
    require((check["gate_head_gradient_l1"] == 0 and check["expected_gate_gradient"] == "exact_zero") if mode == "zero"
            else (check["gate_head_gradient_l1"] > 0 and check["expected_gate_gradient"] == "nonzero"),
            f"{mode}: gate first gradient violates zero/positive policy")


def paired_controls(configs, witnesses, gradients, plan):
    zc, ac = configs["zero"], configs["actual"]
    changed = exact_changes(zc["contract"], ac["contract"],
        {("requested", "hash_modulation_mode"), ("hash_modulation", "hash_modulation_mode")}, "contract")
    model_changes = exact_changes(zc["model"], ac["model"],
        {("hash_modulation_mode",), ("hash_modulation", "mode"),
         ("conditioned_field", "decoder", "gate", "mode"), ("decoder_implementation", "gate", "mode")}, "model")
    exact_changes(zc["arguments"], ac["arguments"], {("output",), ("hash_modulation_mode",)}, "CLI arguments")
    exact_changes(zc["hash_modulation"], ac["hash_modulation"], {("hash_modulation_mode",)}, "hash configuration")
    exempt = {"contract", "model", "arguments", "hash_modulation", "started_at_utc"}
    require(set(zc) == set(ac), "Config key sets differ")
    for key in set(zc) - exempt:
        require(zc[key] == ac[key], f"Paired {key} differs")
    require(witnesses["zero"] == witnesses["actual"], "Case/node/tetra-corner sampling witnesses differ")
    witness = witnesses["zero"]
    require(witness["step"] == 1 and witness["nodes_per_case"] == 12800
            and witness["uniform_nodes_per_case"] == 6400 and witness["uniform_cells_per_case"] == 1600
            and witness["corner_queries_per_case"] == 6400 and len(set(witness["case_keys"])) == 4
            and witness["sampler_streams_independent_of_model"] is True and valid_digest(witness["witness_sha256"]),
            "Wrong frozen 4-case node/tetra-corner sampling witness")
    train_keys = {item["sample_key"] for item in zc["cases"]["train"]}
    require(set(witness["case_keys"]) <= train_keys, "First batch contains non-TRAIN cases")
    for mode, config in configs.items():
        check_layout(config, mode, plan)
        check_gradient(gradients[mode], mode, plan["native_backend"]["source_derived_counts"])
    return {"matched_controls_passed": True, "pair_controls_passed": True,
            "matched_contract_difference_paths": changed, "matched_model_difference_paths": model_changes,
            "sampling_witness_sha256": witness["witness_sha256"], "initial_parameters": zc["initial_parameters"],
            "initial_decoder_parameters": zc["initial_decoder_parameters"], "raw_geometry_equal": True,
            "first_gradients": gradients, "fresh_split_pair": True, "historical_control_reused": False}


def check_scope_and_metrics(config, records, plan):
    groups, loads = config["groups"], plan["scope"]["train_loads"]
    require(config["train_loads"] == loads == ["ver", "hor", "tor"] and config["heldout_load"] == "dia"
            and len(groups["train"]) == 40 and len(groups["validation"]) == 5
            and not (set(groups["train"]) & set(groups["validation"]))
            and not ((set(groups["train"]) | set(groups["validation"])) & set(groups["test"])),
            "CAD-disjoint known-load scope changed")
    for group in ("train", "validation"):
        cases = config["cases"][group]
        declared = {case["sample_key"]: case["nodes"] for case in cases}
        expected = {f"{cad}/{load}" for cad in groups[group] for load in loads}
        require(set(declared) == expected and len(cases) == len(expected) and all(n > 0 for n in declared.values()),
                f"{group}: declared cases are not the full allowed CAD/load product")
        for epoch in (records["best"], records["latest"]):
            measured = epoch[group]
            for quantity in (measured, measured["strain"]):
                per = quantity["per_case"]
                require(set(per) == set(declared) and all(per[key]["nodes"] == n for key, n in declared.items()),
                        f"{group}: U/P1 metrics must cover every original node")
                values = [value["relative_l2"] for value in per.values()]
                require(all(v >= 0 for v in values)
                        and math.isclose(quantity["case_mean_relative_l2"], sum(values) / len(values), rel_tol=1e-10, abs_tol=1e-12)
                        and math.isclose(quantity["worst_case_relative_l2"], max(values), rel_tol=1e-10, abs_tol=1e-12),
                        f"{group}: recorded mean/worst U/P1 does not match per-case values")


def analyze(folders, plan_path):
    began = time.monotonic()
    plan_path = Path(plan_path).resolve()
    plan, plan_sha = read(plan_path), sha(plan_path)
    require(plan["status"] == "predeclared", "Only the frozen predeclared plan can be assessed")
    normalizer_dir = ROOT / "runs/geobc_exploration/displacement_query_context_actual_10k"
    norm_paths = {"config": normalizer_dir / "config.json", "summary": normalizer_dir / "summary.json"}
    norm = {name: read(path) for name, path in norm_paths.items()}
    require(norm["summary"].get("status") == "completed" and norm["summary"]["config"] == norm["config"],
            "Direct query-actual normalizer reference is incomplete/inconsistent")
    configs, summaries, metrics, witnesses, gradients, sources = {}, {}, {}, {}, {}, {}
    for mode in MODES:
        paths = {name: Path(folders[mode]) / filename for name, filename in FILES.items()}
        records = {name: read(path) for name, path in paths.items()}
        summary, config = records["summary"], records["config"]
        recipe = plan["recipe"]
        require(summary.get("status") == "completed" and summary["step"] == recipe["steps"]
                and summary["hash_modulation_mode"] == mode and summary["config"] == config
                and summary["best_metrics"] == records["best"] and summary["final_metrics"] == records["latest"]
                and summary["encoder_gradient_check"] == records["gradient"]
                and summary["best_step"] == records["best"]["step"]
                and summary["best_score"] == records["best"]["validation"]["case_mean_relative_l2"]
                == records["best"]["selection_score"] and records["latest"]["step"] == recipe["steps"]
                and 0 < summary["best_step"] <= recipe["steps"] and summary["best_step"] % recipe["eval_every"] == 0
                and math.isfinite(summary["elapsed_seconds"]) and summary["elapsed_seconds"] > 0,
                f"{mode}: incomplete/inconsistent completed artifacts or selection")
        expected = {"seed": recipe["seed"], "steps": recipe["steps"], "batch_size": recipe["batch_size"],
            "eval_every": recipe["eval_every"], "learning_rate": recipe["optimizer"]["learning_rate"],
            "cases_per_batch": recipe["case_slots"], "gradient_weight": recipe["gradient_weight"],
            "hash_modulation_mode": mode, "query_context_mode": "actual", "bc_role_mode": "actual",
            "physics_mode": "linear_basis", "u_loss": "relative_case_mse", "local_mode": "actual", "resume": None}
        require(all(config["arguments"][key] == value for key, value in expected.items())
                and config["total_steps"] == recipe["steps"] and config["optimizer"] == recipe["optimizer"],
                f"{mode}: actual fixed recipe differs from plan")
        contract = config["contract"]
        require(config["hash_modulation"] == contract["hash_modulation"]
                and contract["hash_modulation"]["plan"]["sha256"] == plan_sha
                and contract["hash_modulation"]["plan"]["metadata"] == plan
                and contract["hash_modulation"]["fresh_pair"] is True
                and contract["hash_modulation"]["historical_control_reused"] is False,
                f"{mode}: frozen plan/fresh-pair evidence differs")
        fixed = contract["fixed_normalizers"]
        for name, path in norm_paths.items():
            require(recorded_path(fixed[f"{name}_path"]) == path.resolve()
                    and fixed[f"{name}_sha256"] == sha(path), f"{mode}: normalizer source {name} differs")
        for name in ("scaler", "u_loss"):
            require(config[name] == fixed[name] == norm["config"][name], f"{mode}: {name} was not directly reused")
        require(config["scaler"]["scale_mm"] == recipe["u_scale_mm"]
                and config["scaler"]["center_mm"] == recipe["u_center_mm"]
                and set(config["scaler"]["fit_case_keys"]) == {v["sample_key"] for v in config["cases"]["train"]}
                and config["groups"] == norm["config"]["groups"], "Train-only normalizer scope changed")
        for epoch in (records["best"], records["latest"]):
            finite_numbers(epoch, f"{mode}.metrics")
        check_scope_and_metrics(config, records, plan)
        configs[mode], summaries[mode], metrics[mode] = config, summary, records["best"]
        witnesses[mode], gradients[mode] = records["sampling"], records["gradient"]
        sources[mode] = {name: {"path": display_path(path), "sha256": sha(path)} for name, path in paths.items()}
    controls = paired_controls(configs, witnesses, gradients, plan)
    assessment = gate_assessment(metrics["zero"], metrics["actual"], plan["clear_gain_gate"])
    result = {"status": "completed", "stage": "matched_geometry_conditioned_hash_feature_modulation",
        "seed": plan["recipe"]["seed"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **controls, **assessment, "next_stage": "parent_decision_pending",
        "selection": "Each own checkpoint uses only known-load validation15 full-node physical signed-u vector mean; P1 is measured at that same checkpoint",
        "metric_units": {"u_relative_l2": "dimensionless vector error in physical mm", "P1_relative_l2": "dimensionless symmetric Frobenius tensor error; shear squared weight2", "display_percent": "100*relative_l2"},
        "P1_reference_limit": plan["clear_gain_gate"]["P1_limit"],
        "effective_capacity_limit": plan["pair"]["effective_capacity_limit"],
        "interpretation": "Only the two new split-native models are matched. Actual activates extra multiplicative function capacity with identical stored parameters and common query/global inputs. Historical Composite results are contextual, not this control. A gate pass is developmental CAD-validation evidence, not final-test generalization, significance, a new modulation algorithm or a constitutive law.",
        "final_test_and_dia_opened": False, "CPU_only": True,
        "scientific_training_runs": 2, "synthetic_native_acceptance_runs_in_this_analysis": 0,
        "analysis_elapsed_seconds": time.monotonic() - began,
        "plan": {"path": display_path(plan_path), "sha256": plan_sha}, "source_artifacts": sources,
        "fixed_normalizers": {name: {"path": display_path(path), "sha256": sha(path)} for name, path in norm_paths.items()},
        "runs": {}, "cost": {}}
    for mode, summary in summaries.items():
        train, val = summary["best_metrics"]["train"], summary["best_metrics"]["validation"]
        config = configs[mode]
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
    result["cost"]["actual_to_zero_elapsed_ratio"] = summaries["actual"]["elapsed_seconds"] / summaries["zero"]["elapsed_seconds"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "notes/geobc_hash_modulation_plan.json")
    parser.add_argument("--zero", type=Path, default=ROOT / "runs/geobc_exploration/displacement_hash_modulation_zero_10k")
    parser.add_argument("--actual", type=Path, default=ROOT / "runs/geobc_exploration/displacement_hash_modulation_actual_10k")
    parser.add_argument("--output", type=Path, default=ROOT / "notes/geobc_hash_modulation_pair_result.json")
    args = parser.parse_args()
    folders = {mode: getattr(args, mode).resolve() for mode in MODES}
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
