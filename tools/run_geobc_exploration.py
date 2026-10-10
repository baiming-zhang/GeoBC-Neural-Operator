"""Run the current GeoBC plan through one recoverable serial GPU queue.

The JSON plan is reloaded between stages and while waiting for inputs, allowing
new jobs to be appended during a running stage. Completed outputs are reused.
Dry runs read local artifacts only and do not acquire locks or create outputs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geobc.exploration import exclusive_lock, identity_digest, request_identity, parser as exploration_parser
from geobc.train import utc_now, write_json


def load_jobs(plan: Path, only: list[str] | None = None) -> list[dict[str, Any]]:
    record = json.loads(plan.read_text(encoding="utf-8"))
    jobs = record["jobs"]
    names = [job["name"] for job in jobs]
    outputs = [str((ROOT / job["output"]).resolve()) for job in jobs]
    if len(names) != len(set(names)) or len(outputs) != len(set(outputs)):
        raise ValueError("Plan job names and output directories must be unique")
    for job in jobs:
        if (not isinstance(job["command"], list) or not job["command"]
                or any(not isinstance(item, str) for item in job["command"])):
            raise ValueError(f"Invalid argv for {job['name']}")
        if not isinstance(job.get("requires", []), list):
            raise ValueError(f"Invalid requires for {job['name']}")
    if only:
        missing = set(only).difference(names)
        if missing:
            raise ValueError(f"Unknown --only jobs: {sorted(missing)}")
        jobs = [job for job in jobs if job["name"] in only]
    return jobs


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # Status writers replace files atomically. WSL's mounted Windows
        # filesystem can briefly lose the directory entry during replacement.
        # Retry on the next poll instead of aborting the live child process.
        return None


def completed_summary(job: dict[str, Any]) -> dict[str, Any] | None:
    filename = "manifest.json" if any(module in job["command"] for module in
        ("geobc.solver_final_eval_data", "geobc.deeponet_data", "geobc.multiscale_mesh_data", "geobc.pointconv_data", "geobc.elastic_pilot_data")) else "summary.json"
    summary = read_json_if_exists(ROOT / job["output"] / filename)
    if not summary or summary.get("status") != "completed":
        return None
    command = job["command"]
    # BEGIN NODAL_TENSOR_PAIR_INTEGRATION
    if "geobc.elastic_nodal_tensor_pair" in command:
        from geobc.elastic_nodal_tensor_pair import request_contract
        contract, requested = request_contract(command[command.index("--plan") + 1])
        fixed_train = ["101_428/ver", "101_56/ver", "101_78/ver"]
        fixed_validation = ["102_249/ver", "102_38/ver"]
        fixed_arms = {"fourQ_cellmean", "element_node"}
        source_pins = contract.get("source_sha256")
        own_source = source_pins.get("geobc/elastic_nodal_tensor_pair.py") if isinstance(source_pins, dict) else None
        gate = summary.get("completion_gate", {})
        if (job.get("kind") != "physical_nodal_tensor_signed_S_representation_pair"
                or summary.get("kind") != "physical_nodal_tensor_signed_S_representation_pair"
                or summary.get("config", {}).get("contract") != contract
                or requested.get("train_keys") != fixed_train or requested.get("validation_keys") != fixed_validation
                or summary.get("train_keys") != fixed_train or summary.get("validation_keys") != fixed_validation
                or summary.get("case_keys") != fixed_train + fixed_validation
                or set(summary.get("cases", {})) != set(fixed_train + fixed_validation)
                or set(summary.get("arms", {})) != fixed_arms
                or not isinstance(source_pins, dict) or not isinstance(own_source, str) or len(own_source) != 64
                or not isinstance(contract.get("scientific_digest"), str) or not contract["scientific_digest"]
                or type(gate.get("fixed_DEV_denominator")) is not int or gate["fixed_DEV_denominator"] != 2
                or gate.get("final_solver_selection") is not False
                or any(type(gate.get(name)) is not bool for name in
                       ("passed", "both_arms_complete120", "all_four_DEV_predictions_persisted",
                        "full5_nodal_recovery_complete", "complete_service_accuracy_gate", "representation_gain_gate"))
                or (gate.get("terminal_negative_reason") is not None
                    and not isinstance(gate["terminal_negative_reason"], str))):
            raise ValueError(f"Nodal tensor representation pair differs from frozen arms/cases/source or terminal gate schema: {job['name']}")
        receipt = summary.get("read_receipt", {})
        for key in ("displacement_target_reads", "source_canonical_stress_reads", "source_canonical_strain_reads",
                    "source_canonical_CSV_reads", "external_learned_weight_reads", "new_physical_solves",
                    "K_applications", "old_K_guard_repeats", "old_final_reads", "sealed_new_test_reads",
                    "DEV_dia_reads", "DEV_reads_before_prediction_barrier", "DEV_scaler_fits", "source_formula_or_sign_trials"):
            if type(receipt.get(key)) is not int or receipt[key] != 0:
                raise ValueError(f"Nodal tensor representation pair unexpectedly accessed {key}: {job['name']}")
        for key in ("physical_U_reads", "physical_U_byte_hash_reads", "input_cache_reads", "input_member_reads",
                    "operator_weight_array_reads", "old_tensor_array_reads", "valid_array_reads", "new_nodal_tensor_recoveries",
                    "TRAIN_cached_S_reads", "DEV_cached_S_reads", "GT_reads", "tensor_scaler_fits", "training_updates",
                    "optimizer_updates", "own_checkpoint_reads", "own_checkpoint_writes", "own_prediction_array_reads", "original_U_byte_copies"):
            if type(receipt.get(key)) is not int or receipt[key] < 0:
                raise ValueError(f"Nodal tensor representation pair lacks actual nonnegative {key}: {job['name']}")
        # Nodal recovery, supervised cached-S reads, updates and own checkpoints
        # are allowed work. Completed budget/performance negatives are terminal.
        return summary
    # END NODAL_TENSOR_PAIR_INTEGRATION
    if "geobc.elastic_continuous_capability" in command:
        from geobc.elastic_continuous_capability import request_contract
        contract, requested = request_contract(command[command.index("--plan") + 1])
        fixed_cases = ["101_428/ver", "101_56/ver", "101_78/ver", "102_249/ver", "102_38/ver"]
        source_pins = contract.get("source_sha256")
        own_source = source_pins.get("geobc/elastic_continuous_capability.py") if isinstance(source_pins, dict) else None
        if (job.get("kind") != "physical_TET10_continuous_U_gradient_capability"
                or summary.get("kind") != "physical_TET10_continuous_U_gradient_capability"
                or summary.get("config", {}).get("contract") != contract
                or requested.get("case_keys") != fixed_cases
                or summary.get("case_keys") != fixed_cases
                or set(summary.get("cases", {})) != set(fixed_cases)
                or not isinstance(source_pins, dict) or not isinstance(own_source, str) or len(own_source) != 64
                or not isinstance(contract.get("scientific_digest"), str) or not contract["scientific_digest"]):
            raise ValueError(f"Physical continuous capability differs from frozen five-case/input/source contract: {job['name']}")
        receipt = summary.get("read_receipt", {})
        for key in ("stress_source_label_reads", "strain_source_label_reads", "learned_weight_reads",
                    "training_updates", "scaler_fits", "new_physical_solves", "new_tensor_recoveries",
                    "K_applications", "old_K_guard_repeats", "old_final_reads", "sealed_new_test_reads",
                    "DEV_dia_reads", "GT_reads_before_all_five_prediction_barrier"):
            if type(receipt.get(key)) is not int or receipt[key] != 0:
                raise ValueError(f"Physical continuous capability unexpectedly accessed {key}: {job['name']}")
        for key in ("physical_U_prediction_reads", "physical_U_byte_hash_reads", "GT_U_reads",
                    "input_cache_reads", "input_cache_byte_hash_reads", "input_member_reads", "operator_array_reads",
                    "own_prediction_array_reads", "own_reference_array_reads", "source_GT_U_read_attempts", "source_GT_U_read_completed"):
            if type(receipt.get(key)) is not int or receipt[key] < 0:
                raise ValueError(f"Physical continuous capability lacks actual nonnegative {key}: {job['name']}")
        # Node-U-derived TET10 gradients are proxies, not source strain GT.
        # Terminal budget/numerical capability negatives never restart solves.
    elif "geobc.elastic_spatial_budget_readout" in command:
        from geobc.elastic_spatial_budget_readout import request_contract
        contract, requested = request_contract(command[command.index("--plan") + 1])
        fixed_validation = ["102_249/ver", "102_38/ver"]
        fixed_arms = {"local_physical_zero_latent", "global_FNO_physical_latent"}
        source_pins = contract.get("source_sha256")
        own_source = source_pins.get("geobc/elastic_spatial_budget_readout.py") if isinstance(source_pins, dict) else None
        gate = summary.get("completion_gate", {})
        if (job.get("kind") != "budget_limited_spatial_pair_fixed_endpoint_readout"
                or summary.get("kind") != "budget_limited_spatial_pair_fixed_endpoint_readout"
                or summary.get("config", {}).get("contract") != contract
                or summary.get("validation_keys") != fixed_validation
                or set(summary.get("arms", {})) != fixed_arms
                or set(summary.get("cases", {})) != set(fixed_validation)
                or not isinstance(source_pins, dict) or not isinstance(own_source, str) or len(own_source) != 64
                or any(not isinstance(contract.get(name), str) or not contract[name]
                       for name in ("scientific_digest", "science_recipe_sha256"))
                or gate.get("parent_pair_pass_claimed") is not False or gate.get("capacity_matched") is not False):
            raise ValueError(f"Fixed spatial budget readout differs from endpoints/DEV scope/source or parent disclosure: {job['name']}")
        receipt = summary.get("read_receipt", {})
        for key in ("displacement_target_reads", "TRAIN_displacement_target_reads", "DEV_displacement_target_reads",
                    "source_csv_or_archive_reads", "canonical_target_member_reads", "old_final_reads",
                    "sealed_new_test_reads", "DEV_dia_reads", "new_physical_solves", "new_tensor_recoveries",
                    "K_guards_repeated", "external_model_or_weight_reads", "optimizer_steps", "training_updates",
                    "scaler_fits", "DEV_scaler_fits", "TRAIN_stress_reads"):
            if type(receipt.get(key)) is not int or receipt[key] != 0:
                raise ValueError(f"Spatial budget readout unexpectedly accessed {key}: {job['name']}")
        for key in ("GT_reads", "stress_reads", "DEV_stress_reads", "cached_stress_array_reads",
                    "parent_evaluation_cache_reads", "own_evaluation_cache_reads", "cached_input_array_reads",
                    "physical_U_input_reads", "physical_U_copy_reads", "parent_checkpoint_reads",
                    "parent_checkpoint_hash_reads", "own_prediction_array_reads"):
            if type(receipt.get(key)) is not int or receipt[key] < 0:
                raise ValueError(f"Spatial budget readout lacks actual nonnegative {key}: {job['name']}")
        # Parent budget/gate remains negative. Fixed readout, skipped existing
        # metrics and terminal budget negatives never rescue/replay training.
    elif "geobc.elastic_spatial_recovery" in command:
        from geobc.elastic_spatial_recovery import request_contract
        contract, requested = request_contract(command[command.index("--plan") + 1])
        fixed_train = ["101_428/ver", "101_56/ver", "101_78/ver"]
        fixed_validation = ["102_249/ver", "102_38/ver"]
        fixed_arms = {"local_physical_zero_latent", "global_FNO_physical_latent"}
        source_pins = contract.get("source_sha256")
        own_source = source_pins.get("geobc/elastic_spatial_recovery.py") if isinstance(source_pins, dict) else None
        if (job.get("kind") != "spatial_physical_signed_recovery_pair"
                or summary.get("kind") != "spatial_physical_signed_recovery_pair"
                or summary.get("config", {}).get("contract") != contract
                or requested.get("train_keys") != fixed_train or requested.get("validation_keys") != fixed_validation
                or summary.get("train_keys") != fixed_train or summary.get("validation_keys") != fixed_validation
                or summary.get("case_keys") != fixed_train + fixed_validation
                or set(summary.get("arms", {})) != fixed_arms
                or set(summary.get("cases", {})) != set(fixed_train + fixed_validation)
                or not isinstance(source_pins, dict) or not isinstance(own_source, str) or len(own_source) != 64
                or any(not isinstance(contract.get(name), str) or not contract[name]
                       for name in ("scientific_digest", "science_recipe_sha256"))):
            raise ValueError(f"Spatial physical signed-recovery pair differs from frozen arms/cases/source contract: {job['name']}")
        receipt = summary.get("read_receipt", {})
        for key in ("displacement_target_reads", "TRAIN_displacement_target_reads", "DEV_displacement_target_reads",
                    "source_csv_or_archive_reads", "canonical_target_member_reads", "old_final_reads",
                    "sealed_new_test_reads", "DEV_dia_reads", "new_physical_solves", "new_tensor_recoveries",
                    "K_guards_repeated", "external_model_or_weight_reads", "DEV_scaler_fits"):
            if type(receipt.get(key)) is not int or receipt[key] != 0:
                raise ValueError(f"Spatial recovery pair unexpectedly accessed {key}: {job['name']}")
        for key in ("GT_reads", "stress_reads", "TRAIN_stress_reads", "DEV_stress_reads", "physical_U_input_reads",
                    "cached_input_array_reads", "scaler_fits", "optimizer_steps", "own_checkpoint_reads", "own_checkpoint_writes"):
            if type(receipt.get(key)) is not int or receipt[key] < 0:
                raise ValueError(f"Spatial recovery pair lacks actual nonnegative {key}: {job['name']}")
        # Supervised stress/optimizer/own-checkpoint accesses are real allowed
        # work. Completed budget or performance negatives end this fixed pair.
    elif "geobc.elastic_low_fidelity" in command:
        from geobc.elastic_low_fidelity import request_contract
        contract, requested = request_contract(command[command.index("--plan") + 1])
        fixed_cases = ["101_428/ver", "102_249/ver"]
        source_pins = contract.get("source_sha256")
        source_hashes = contract.get("source_hashes")
        own_source = source_pins.get("geobc/elastic_low_fidelity.py") if isinstance(source_pins, dict) else None
        if (job.get("kind") != "elastic_direct_coarse_screen"
                or summary.get("kind") != "elastic_direct_coarse_screen"
                or summary.get("config", {}).get("contract") != contract
                or requested.get("case_keys") != fixed_cases
                or summary.get("case_keys") != fixed_cases
                or set(summary.get("cases", {})) != set(fixed_cases)
                or not isinstance(source_pins, dict) or not isinstance(source_hashes, dict)
                or not isinstance(own_source, str) or len(own_source) != 64
                or source_hashes.get("geobc/elastic_low_fidelity.py") != own_source
                or any(not isinstance(contract.get(name), str) or not contract[name]
                       for name in ("scientific_digest", "science_recipe_sha256"))):
            raise ValueError(f"Direct coarse screen differs from frozen two-case/source/equation contract: {job['name']}")
        receipts = [summary.get("read_receipt", {})]
        for case_key in fixed_cases:
            record = summary["cases"][case_key]
            receipts.append(record.get("read_receipt", {}))
        for receipt in receipts:
            for key in ("GT_reads", "displacement_reads", "stress_reads", "old_final_reads",
                        "DEV_dia_reads", "sealed_new_test_reads", "optimizer_steps", "scaler_fits",
                        "learned_weight_reads", "setup_K_guards_repeated",
                        "inherited_fine_K_guards_repeated", "fine_operator_rebuilds"):
                if type(receipt.get(key)) is not int or receipt[key] != 0:
                    raise ValueError(f"Direct coarse screen unexpectedly accessed {key}: {job['name']}")
            for key in ("numerical_reference_reads", "accepted_coarse_cache_reads", "fine_endpoint_K_applications"):
                if type(receipt.get(key)) is not int or receipt[key] < 0:
                    raise ValueError(f"Direct coarse screen lacks actual nonnegative {key}: {job['name']}")
        # False numerical/cold-cost gates and budget negatives are terminal
        # completed screens; numerical U references are required reads, not GT.
    elif "geobc.solver_service_benchmark" in command:
        from geobc.solver_service_benchmark import request_contract
        contract, requested = request_contract(command[command.index("--plan") + 1])
        fixed_requests = ["fno:102_249/ver", "fno:102_38/ver",
                          "mlp:102_249/ver", "mlp:102_38/ver"]
        source_pins = contract.get("source_sha256", {})
        own_source = source_pins.get("geobc/solver_service_benchmark.py")
        if (job.get("kind") != "solver_service_cost_screen"
                or summary.get("kind") != "solver_service_cost_screen"
                or summary.get("config", {}).get("contract") != contract
                or summary.get("request_keys") != fixed_requests
                or set(summary.get("requests", {})) != set(fixed_requests)
                or not isinstance(own_source, str) or len(own_source) != 64):
            raise ValueError(f"Service costs differ from frozen endpoint/input/source/request contract: {job['name']}")
        receipts = [summary.get("read_receipt", {})]
        for request_key in fixed_requests:
            record = summary["requests"][request_key]
            method, case_key = request_key.split(":", 1)
            if record.get("method") != method or record.get("case_key") != case_key:
                raise ValueError(f"Service request endpoint/case differs: {job['name']}:{request_key}")
            receipts.append(record.get("read_receipt", {}))
        for receipt in receipts:
            for key in ("zero_GT_reads", "displacement_target_reads", "stress_target_reads",
                        "training_updates", "scaler_fits", "old_final_reads",
                        "sealed_new_test_reads", "DEV_dia_reads"):
                if type(receipt.get(key)) is not int or receipt[key] != 0:
                    raise ValueError(f"Service costs unexpectedly accessed {key}: {job['name']}")
            for key in ("model_checkpoint_reads", "prepared_member_reads"):
                if type(receipt.get(key)) is not int or receipt[key] < 0:
                    raise ValueError(f"Service costs lack actual nonnegative {key}: {job['name']}")
        # Completed budget/backend negatives terminate this frozen cost screen.
        # Endpoint weights and input members are required reads, not forbidden GT.
    elif "geobc.exploration" in command:
        cli_args = command[command.index("geobc.exploration") + 1:]
        parsed = exploration_parser().parse_args(cli_args)
        if parsed.steps is None and parsed.epochs is None:
            parsed.steps = 2000
        if summary["config"].get("request_digest") != identity_digest(request_identity(parsed)):
            raise ValueError(f"Completed output for {job['name']} differs from current plan")
    elif "geobc.solver_final_eval_data" in command:
        from geobc.solver_final_eval_data import request_contract, read_cache_metadata
        requested_plan = command[command.index("--plan") + 1]
        contract, requested = request_contract(requested_plan)
        if (ROOT / job["output"]).resolve() != (ROOT / requested["eval_cache"]).resolve():
            raise ValueError(f"Final cache output differs from frozen plan: {job['name']}")
        verified = read_cache_metadata(requested_plan, contract=contract)
        if summary != verified:
            raise ValueError(f"Completed final input cache differs from frozen metadata: {job['name']}")
    elif "geobc.solver_final_eval" in command:
        from geobc.solver_final_eval import request_contract
        requested_plan = command[command.index("--plan") + 1]
        contract, _ = request_contract(requested_plan)
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed final evaluation differs from frozen routes/input/checkpoint contract: {job['name']}")
        if summary.get("training_updates") != 0 or summary.get("scaler_fits") != 0:
            raise ValueError(f"Final evaluation unexpectedly trained or refit scalers: {job['name']}")
    elif "geobc.solver_confirmation_train" in command:
        from geobc.solver_confirmation_train import request_contract
        requested_plan = command[command.index("--plan") + 1]
        requested_method = command[command.index("--method") + 1]
        requested_seed = int(command[command.index("--seed") + 1])
        warmstart = command[command.index("--warmstart") + 1] if "--warmstart" in command else None
        contract, _ = request_contract(requested_plan, requested_method, requested_seed, warmstart)
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed solver confirmation {job['name']} differs from requested protocol/data")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed solver confirmation {job['name']} lacks {filename}")
    elif "geobc.solver_selection_train" in command:
        from geobc.solver_selection_train import request_contract, contracts_compatible
        requested_plan = command[command.index("--plan") + 1]
        requested_method = command[command.index("--method") + 1]
        contract, _ = request_contract(requested_plan, requested_method)
        previous = summary.get("config", {}).get("contract")
        if not contracts_compatible(previous, contract):
            raise ValueError(f"Completed solver screen {job['name']} differs from requested protocol/data")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed solver screen {job['name']} lacks {filename}")
    elif "geobc.conditional_train" in command:
        # The trainer's own strict contract uses only small source/config
        # metadata here; no labels, checkpoints, models or CUDA are loaded.
        from geobc.conditional_train import parser as conditional_parser, _scope
        parsed = conditional_parser().parse_args(command[command.index("geobc.conditional_train") + 1:])
        contract = _scope(parsed)[4]
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed output for {job['name']} differs from current strict configuration/data contract")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "geobc.physics_train" in command:
        from geobc.physics_train import parser as physics_parser, _scope
        parsed = physics_parser().parse_args(command[command.index("geobc.physics_train") + 1:])
        contract = _scope(parsed)[4]
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed output for {job['name']} differs from its displacement contract")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "geobc.gradient_train" in command:
        from geobc.gradient_train import parser as gradient_parser, _scope
        parsed = gradient_parser().parse_args(command[command.index("geobc.gradient_train") + 1:])
        contract = _scope(parsed)[4]
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed output for {job['name']} differs from its gradient-supervision contract")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "geobc.gradient_balance_train" in command:
        from geobc.gradient_balance_train import parser as balance_parser, _scope
        parsed = balance_parser().parse_args(command[command.index("geobc.gradient_balance_train") + 1:])
        contract = _scope(parsed)[4]
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed output for {job['name']} differs from its calibrated-gradient contract")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "geobc.bc_role_train" in command:
        from geobc.bc_role_train import parser as role_parser, _scope
        parsed = role_parser().parse_args(command[command.index("geobc.bc_role_train") + 1:])
        contract = _scope(parsed)[4]
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed output for {job['name']} differs from its BC-role displacement contract")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "geobc.specialist_train" in command:
        from geobc.specialist_train import parser as specialist_parser, _scope
        parsed = specialist_parser().parse_args(command[command.index("geobc.specialist_train") + 1:])
        contract = _scope(parsed)[4]
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed output for {job['name']} differs from its fixed TRAIN-CAD contract")
        for filename in ("last.pt", "endpoint_250.pt", "endpoint_10000.pt", "metrics_endpoint_250.json",
                         "metrics_endpoint_10000.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "geobc.query_context_train" in command:
        from geobc.query_context_train import parser as query_parser, _scope
        parsed = query_parser().parse_args(command[command.index("geobc.query_context_train") + 1:])
        contract = _scope(parsed)[4]
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed output for {job['name']} differs from its query-context displacement contract")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "tools.fetch_next_cycle_test" in command:
        from tools.fetch_next_cycle_test import request_contract, host_path
        requested_plan = command[command.index("--plan") + 1]
        received_contract = dict(summary.get("config", {}).get("contract", {}))
        # The CPU transport may use Windows' configured proxy while this
        # coordinator stays in WSL. Normalize only the three host paths;
        # source hashes, frozen IDs/URLs and all access rules still match.
        for path_key in ("plan", "source_index", "output"):
            if path_key in received_contract:
                received_contract[path_key] = str(host_path(received_contract[path_key]))
        if received_contract != request_contract(requested_plan):
            raise ValueError(f"Completed sealed acquisition differs from frozen source/ID contract: {job['name']}")
        if (summary.get("completed_files") != 15 or summary.get("scientific_array_reads") != 0
                or summary.get("Field_CSV_values_parsed") is not False or summary.get("labels_sealed") is not True):
            raise ValueError(f"Sealed acquisition completeness or label-access contract failed: {job['name']}")
        for record in summary["files"]:
            target = host_path(ROOT / job["output"] / record["relative_path"])
            actual_stat = target.stat()
            if {"bytes": actual_stat.st_size, "mtime_ns": actual_stat.st_mtime_ns} != record["file_stat"]:
                raise ValueError(f"Completed sealed source file missing or changed: {job['name']}")
    elif "geobc.elastic_pilot_data" in command:
        from geobc.elastic_pilot_data import request_contract, read_cache_metadata
        requested_plan = command[command.index("--plan") + 1]
        contract, _ = request_contract(requested_plan)
        if (summary.get("kind") != "elastic_pilot_input"
                or summary.get("contract") != contract
                or summary.get("labels_read") != 0 or summary.get("stress_reads") != 0):
            raise ValueError(f"Physical input metadata or label-access contract differs: {job['name']}")
        if summary != read_cache_metadata(requested_plan, contract=contract):
            raise ValueError(f"Physical input cache differs from its frozen source identities: {job['name']}")
    elif "geobc.elastic_coarse_extension" in command:
        from geobc.elastic_coarse_extension import request_contract
        contract, requested = request_contract(command[command.index("--plan") + 1])
        if (summary.get("kind") != "elastic_p1_derived_coarse_extension"
                or summary.get("config", {}).get("contract") != contract
                or summary.get("case_keys") != requested["case_keys"]
                or set(summary.get("cases", {})) != set(requested["case_keys"])):
            raise ValueError(f"Coarse extension differs from frozen five-case campaign: {job['name']}")
        receipt = summary.get("read_receipt", {})
        for key in ("GT_reads", "displacement_reads", "stress_reads", "old_final_reads",
                    "DEV_dia_reads", "sealed_new_test_reads", "optimizer_steps", "scaler_fits",
                    "learned_weight_reads", "setup_K_guards_repeated"):
            if receipt.get(key) != 0:
                raise ValueError(f"Coarse extension unexpectedly accessed {key}: {job['name']}")
    elif "geobc.elastic_factored_stress" in command:
        from geobc.elastic_factored_stress import request_contract
        contract, requested = request_contract(command[command.index("--plan") + 1])
        if (summary.get("kind") != "elastic_factored_signed_stress_screen"
                or summary.get("config", {}).get("contract") != contract
                or summary.get("train_keys") != requested["train_keys"]
                or summary.get("validation_keys") != requested["validation_keys"]
                or set(summary.get("cases", {})) != set(requested["train_keys"] + requested["validation_keys"])):
            raise ValueError(f"Factored signed-stress candidate differs from its frozen cache/recipe: {job['name']}")
        receipt = summary.get("read_receipt", {})
        for key in ("source_stress_member_reads", "canonical_archive_reads", "displacement_reads",
                    "old_final_reads", "DEV_dia_reads", "sealed_new_test_reads", "external_model_or_weight_reads",
                    "new_physical_solves", "K_guards_repeated", "scaler_fits"):
            if receipt.get(key) != 0:
                raise ValueError(f"Factored signed-stress candidate unexpectedly accessed {key}: {job['name']}")
        for key in ("GT_reads", "stress_reads", "TRAIN_stress_reads", "DEV_stress_reads", "optimizer_steps"):
            if type(receipt.get(key)) is not int or receipt[key] < 0:
                raise ValueError(f"Factored candidate lacks actual nonnegative {key}: {job['name']}")
        if (receipt["GT_reads"] != receipt["stress_reads"]
                or receipt["stress_reads"] != receipt["TRAIN_stress_reads"] + receipt["DEV_stress_reads"]):
            raise ValueError(f"Factored signed-stress label accounting is inconsistent: {job['name']}")
        # Fixed60 budget/accuracy negatives close the representation recipe.
    elif "geobc.elastic_coarse" in command:
        from geobc.elastic_coarse import request_contract
        contract, requested = request_contract(command[command.index("--plan") + 1])
        if (summary.get("kind") != "elastic_p1_derived_coarse_screen"
                or summary.get("config", {}).get("contract") != contract
                or summary.get("case_keys") != requested["case_keys"]
                or set(summary.get("cases", {})) != set(requested["case_keys"])):
            raise ValueError(f"Coarse solver screen differs from its frozen equation or two cases: {job['name']}")
        receipt = summary.get("read_receipt", {})
        for key in ("GT_reads", "displacement_reads", "stress_reads", "old_final_reads",
                    "DEV_dia_reads", "sealed_new_test_reads", "optimizer_steps", "scaler_fits",
                    "learned_weight_reads", "setup_K_guards_repeated"):
            if receipt.get(key) != 0:
                raise ValueError(f"Coarse solver screen unexpectedly accessed {key}: {job['name']}")
        # Sparse-backend, budget and performance negatives close this fixed
        # candidate; the scientific same-solution/cost gate is kept separate.
    elif "geobc.elastic_stress_magnitude" in command:
        from geobc.elastic_stress_magnitude import request_contract
        contract, requested = request_contract(command[command.index("--plan") + 1])
        if (summary.get("kind") != "elastic_stress_magnitude_diagnostic"
                or summary.get("config", {}).get("contract") != contract
                or summary.get("train_keys") != requested["train_keys"]
                or summary.get("validation_keys") != requested["validation_keys"]
                or set(summary.get("cases", {})) != set(requested["train_keys"] + requested["validation_keys"])):
            raise ValueError(f"Magnitude diagnostic differs from its frozen cases or cache identities: {job['name']}")
        receipt = summary.get("read_receipt", {})
        for key in requested["science"]["read_policy"]["forbidden_zero"]:
            if receipt.get(key) != 0:
                raise ValueError(f"Magnitude diagnostic unexpectedly accessed {key}: {job['name']}")
        for key in requested["science"]["read_policy"]["normal_completed_reads"]:
            if type(receipt.get(key)) is not int or receipt[key] < 0:
                raise ValueError(f"Magnitude diagnostic lacks actual nonnegative {key}: {job['name']}")
        if (receipt["GT_reads"] != receipt["stress_reads"]
                or receipt["stress_reads"] != receipt["cached_signed_stress_reads"]
                or receipt["stress_reads"] != receipt["TRAIN_stress_reads"] + receipt["DEV_stress_reads"]):
            raise ValueError(f"Cached GT stress accounting is inconsistent: {job['name']}")
        # Invalid/budget-limited scientific outcomes still finish this fixed
        # diagnostic; they cannot become accuracy successes or retry loops.
    elif "geobc.elastic_stress_readout" in command:
        from geobc.elastic_stress_readout import request_contract
        requested_plan = command[command.index("--plan") + 1]
        contract, requested = request_contract(requested_plan)
        if (summary.get("kind") != "elastic_signed_stress_readout_screen"
                or summary.get("config", {}).get("contract") != contract
                or summary.get("train_keys") != requested["train_keys"]
                or summary.get("validation_keys") != requested["validation_keys"]
                or set(summary.get("cases", {})) != set(requested["train_keys"] + requested["validation_keys"])
                or set(summary.get("arms", {})) != set(requested["neural"]["arms"])):
            raise ValueError(f"Stress readout screen differs from its frozen cases, arms or source recipe: {job['name']}")
        receipt = summary.get("read_receipt", {})
        for key in ("displacement_reads", "old_final_reads", "DEV_dia_reads", "sealed_new_test_reads"):
            if receipt.get(key) != 0:
                raise ValueError(f"Stress readout screen unexpectedly accessed {key}: {job['name']}")
        for key in ("GT_reads", "stress_reads", "TRAIN_stress_reads", "DEV_stress_reads", "optimizer_steps", "scaler_fits"):
            if type(receipt.get(key)) is not int or receipt[key] < 0:
                raise ValueError(f"Stress readout screen has no actual nonnegative {key} receipt: {job['name']}")
        if (receipt["GT_reads"] != receipt["stress_reads"]
                or receipt["stress_reads"] != receipt["TRAIN_stress_reads"] + receipt["DEV_stress_reads"]):
            raise ValueError(f"Stress-only label receipt is inconsistent: {job['name']}")
        # Scientific budget/performance negatives still complete this frozen
        # stage; exact120/finite DEV2 and accuracy gates are the reporter's job.
    elif "geobc.elastic_fno_warmstart" in command:
        from geobc.elastic_fno_warmstart import request_contract
        requested_plan = command[command.index("--plan") + 1]
        contract, requested = request_contract(requested_plan)
        if (summary.get("kind") != "elastic_frozen_fno_warmstart_screen"
                or summary.get("config", {}).get("contract") != contract
                or summary.get("case_keys") != requested["case_keys"]
                or set(summary.get("cases", {})) != set(requested["case_keys"])):
            raise ValueError(f"Frozen FNO/PCG comparison identity differs: {job['name']}")
        receipt = summary.get("read_receipt", {})
        for key in ("GT_reads", "displacement_reads", "stress_reads", "old_final_reads",
                    "DEV_dia_reads", "sealed_new_test_reads", "optimizer_steps", "scaler_fits"):
            if receipt.get(key) != 0:
                raise ValueError(f"Frozen FNO/PCG comparison unexpectedly accessed {key}: {job['name']}")
    elif "geobc.elastic_crosscad" in command:
        from geobc.elastic_crosscad import request_contract
        requested_plan = command[command.index("--plan") + 1]
        contract, requested = request_contract(requested_plan)
        if (summary.get("kind") != "elastic_crosscad_screen"
                or summary.get("config", {}).get("contract") != contract
                or summary.get("case_keys") != requested["case_keys"]
                or set(summary.get("cases", {})) != set(requested["case_keys"])):
            raise ValueError(f"Physical screen differs from its five fixed cases/source/budgets: {job['name']}")
        receipt = summary.get("read_receipt", {})
        reads = receipt.get("displacement_reads")
        if not isinstance(reads, int) or not 0 <= reads <= len(requested["case_keys"]):
            raise ValueError(f"Physical screen displacement-read budget differs: {job['name']}")
        for key in ("stress_reads", "old_final_reads", "DEV_dia_reads", "sealed_new_test_reads",
                    "model_or_weight_reads", "optimizer_steps", "scaler_fits"):
            if receipt.get(key) != 0:
                raise ValueError(f"Physical screen unexpectedly accessed {key}: {job['name']}")
    elif "geobc.elastic_pilot_convergence" in command:
        from geobc.elastic_pilot_convergence import request_contract
        requested_plan = command[command.index("--plan") + 1]
        contract, _ = request_contract(requested_plan)
        if (summary.get("kind") != "elastic_residual_continuation"
                or summary.get("config", {}).get("contract") != contract
                or summary.get("node_count") != 179316):
            raise ValueError(f"Residual continuation differs from its frozen input/state/budget: {job['name']}")
        receipt = summary.get("read_receipt", {})
        for key in ("displacement_reads", "stress_reads", "old_final_reads", "DEV_dia_reads",
                    "sealed_new_test_reads", "model_or_weight_reads", "optimizer_steps", "scaler_fits",
                    "K_guards_repeated", "setup_repeated"):
            if receipt.get(key) != 0:
                raise ValueError(f"Residual-only continuation unexpectedly accessed or repeated {key}: {job['name']}")
        if receipt.get("parent_state_reused") is not True:
            raise ValueError(f"Residual continuation did not retain its parent recurrence: {job['name']}")
    elif "geobc.elastic_pilot" in command:
        from geobc.elastic_pilot import request_contract, _operator_metadata, source_sha, host_path
        requested_plan = command[command.index("--plan") + 1]
        contract, requested = request_contract(requested_plan)
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Physical operator/solve differs from its frozen mathematical model: {job['name']}")
        if "--prepare-operator" in command:
            if summary.get("kind") != "elastic_input_operator":
                raise ValueError(f"Physical operator kind differs: {job['name']}")
            _operator_metadata(contract, requested)
            receipt = summary.get("read_receipt", {})
            if receipt.get("displacement_reads") != 0 or receipt.get("stress_reads") != 0:
                raise ValueError(f"Physical input operator unexpectedly accessed labels: {job['name']}")
        else:
            if (summary.get("kind") != "elastic_physical_solve"
                    or summary["config"].get("operator_summary_sha256")
                        != source_sha(host_path(requested["operator_output"]) / "summary.json")):
                raise ValueError(f"Physical solve lacks its fixed input operator identity: {job['name']}")
            if summary.get("numerical_state") == "breakdown":
                if summary.get("signed_displacement_reads") != 0 or summary.get("stress_reads") != 0:
                    raise ValueError(f"Broken physical solve unexpectedly accessed labels: {job['name']}")
            else:
                receipt = summary.get("read_receipt", {})
                if (summary.get("metrics", {}).get("nodes") != 179316
                        or receipt.get("signed_displacement_member_reads") != 1
                        or receipt.get("stress_reads") != 0
                        or receipt.get("prediction_and_equilibrium_persisted_before_GT") is not True
                        or any(receipt.get(key) != 0 for key in
                               ("old_final_reads", "DEV_dia_reads", "sealed_new_test_reads", "optimizer_steps", "scaler_fits"))):
                    raise ValueError(f"Physical solve full-node/deferred-label contract failed: {job['name']}")
    elif "geobc.pointconv_data" in command:
        from geobc.pointconv_data import request_contract, read_cache_metadata
        requested_plan = command[command.index("--plan") + 1]
        contract, _ = request_contract(requested_plan)
        if summary != read_cache_metadata(requested_plan, contract=contract):
            raise ValueError(f"Completed PointConv geometry cache differs from its frozen identities: {job['name']}")
    elif ("geobc.pointconv_eval_recovery" in command
          or ("geobc.pointconv_train" in command and "--evaluate-dia" in command)):
        if "geobc.pointconv_eval_recovery" in command:
            from geobc.pointconv_eval_recovery import request_dia_contract
        else:
            from geobc.pointconv_train import request_dia_contract
        requested_plan = command[command.index("--plan") + 1]
        contract, _ = request_dia_contract(requested_plan)
        if (summary.get("kind") != "pointconv_development_dia_eval"
                or summary.get("config", {}).get("contract") != contract):
            raise ValueError(f"Completed PointConv DEV dia differs from its fixed endpoint contract: {job['name']}")
        if summary.get("training_updates") != 0 or summary.get("scaler_fits") != 0:
            raise ValueError(f"PointConv DEV dia unexpectedly trained or refit scalers: {job['name']}")
    elif "geobc.pointconv_train" in command:
        from geobc.pointconv_train import request_contract
        requested_plan = command[command.index("--plan") + 1]
        contract, _ = request_contract(requested_plan)
        if (summary.get("kind") != "pointconv_screen"
                or summary.get("config", {}).get("contract") != contract):
            raise ValueError(f"Completed PointConv screen differs from its frozen whole-case recipe: {job['name']}")
        counts = summary.get("case_exposure_counts", {})
        if (summary.get("endpoint_step") != 480
                or summary.get("costs", {}).get("train_case_exposures") != 480
                or len(counts) != 120 or any(count != 4 for count in counts.values())
                or summary.get("first_actual_check", {}).get("passed") is not True):
            raise ValueError(f"PointConv screen lacks checked complete TRAIN120 four-pass exposure: {job['name']}")
        for filename in ("config.json", "last.pt", "best.pt", "step480.pt", "metrics_latest.json", "metrics_best.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed PointConv screen lacks {filename}: {job['name']}")
    elif "geobc.multiscale_mesh_data" in command:
        from geobc.multiscale_mesh_data import request_contract, read_cache_metadata
        requested_plan = command[command.index("--plan") + 1]
        contract, _ = request_contract(requested_plan)
        if summary != read_cache_metadata(requested_plan, contract=contract):
            raise ValueError(f"Completed multiscale mesh input cache differs from frozen geometry: {job['name']}")
    elif "geobc.multiscale_mesh_train" in command and "--evaluate-dia" in command:
        from geobc.multiscale_mesh_train import request_dia_contract
        requested_plan = command[command.index("--plan") + 1]
        contract, _ = request_dia_contract(requested_plan)
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed mesh DEV dia differs from its fixed endpoint contract: {job['name']}")
        if summary.get("training_updates") != 0 or summary.get("scaler_fits") != 0:
            raise ValueError(f"Mesh DEV dia unexpectedly trained or refit scalers: {job['name']}")
    elif "geobc.multiscale_mesh_train" in command:
        from geobc.multiscale_mesh_train import request_contract
        requested_plan = command[command.index("--plan") + 1]
        contract, _ = request_contract(requested_plan)
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed mesh screen differs from its frozen whole-case recipe: {job['name']}")
        counts = summary.get("case_exposure_counts", {})
        if (summary.get("endpoint_step") != 480
                or summary.get("costs", {}).get("train_case_exposures") != 480
                or len(counts) != 120 or any(count != 4 for count in counts.values())):
            raise ValueError(f"Mesh screen lacks four exposures of all TRAIN120 cases: {job['name']}")
        for filename in ("config.json", "last.pt", "best.pt", "step480.pt", "metrics_latest.json", "metrics_best.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed mesh screen lacks {filename}: {job['name']}")
    elif "geobc.deeponet_data" in command:
        from geobc.deeponet_data import request_contract, read_cache_metadata
        requested_plan = command[command.index("--plan") + 1]
        contract, _ = request_contract(requested_plan)
        verified = read_cache_metadata(requested_plan, contract=contract)
        if summary != verified:
            raise ValueError(f"Completed geometry/BC cloud cache differs from frozen input identities: {job['name']}")
    elif "geobc.deeponet_train" in command and "--evaluate-dia" in command:
        from geobc.deeponet_train import request_dia_contract
        requested_plan = command[command.index("--plan") + 1]
        contract, _ = request_dia_contract(requested_plan)
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed DeepONet/concat DEV dia differs from its fixed endpoint contract: {job['name']}")
        if summary.get("training_updates") != 0 or summary.get("scaler_fits") != 0:
            raise ValueError(f"DeepONet/concat DEV dia unexpectedly trained or refit scalers: {job['name']}")
    elif "geobc.deeponet_train" in command:
        from geobc.deeponet_train import request_contract
        requested_plan = command[command.index("--plan") + 1]
        requested_mode = command[command.index("--mode") + 1]
        contract, _ = request_contract(requested_plan, requested_mode)
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed geometry/BC field screen differs from its shared-input recipe: {job['name']}")
        counts = summary.get("case_exposure_counts", {})
        if (summary.get("endpoint_step") != 480
                or summary.get("costs", {}).get("train_case_exposures") != 480
                or len(counts) != 120 or any(count != 4 for count in counts.values())):
            raise ValueError(f"Geometry/BC field screen lacks four exposures of TRAIN120: {job['name']}")
        for filename in ("config.json", "last.pt", "best.pt", "step480.pt", "metrics_latest.json", "metrics_best.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed geometry/BC field screen lacks {filename}: {job['name']}")
    elif "geobc.fno_response_train" in command:
        from geobc.fno_response_train import request_contract
        requested_plan = command[command.index("--plan") + 1]
        requested_mode = command[command.index("--mode") + 1]
        contract, _ = request_contract(requested_plan, requested_mode)
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed FNO response screen differs from its frozen paired protocol: {job['name']}")
        counts = summary.get("case_exposure_counts", {})
        if (summary.get("endpoint_step") != 480
                or summary.get("costs", {}).get("train_case_exposures") != 480
                or len(counts) != 120 or any(count != 4 for count in counts.values())):
            raise ValueError(f"FNO response screen lacks four exposures of every TRAIN120 case: {job['name']}")
        for filename in ("config.json", "last.pt", "best.pt", "step480.pt", "metrics_latest.json", "metrics_best.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed FNO response screen lacks {filename}: {job['name']}")
    elif "geobc.fno_response_eval" in command:
        from geobc.fno_response_eval import request_contract
        requested_plan = command[command.index("--plan") + 1]
        contract, _ = request_contract(requested_plan)
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed FNO response dia evaluation differs from its frozen DEV routes: {job['name']}")
        if summary.get("training_updates") != 0 or summary.get("scaler_fits") != 0:
            raise ValueError(f"FNO response dia evaluation unexpectedly trained or refit scalers: {job['name']}")
    elif "geobc.gino_train" in command:
        from geobc.gino_train import request_contract, _resource_gate
        requested_plan = command[command.index("--plan") + 1]
        _resource_gate(read_json_if_exists(ROOT / requested_plan))
        contract, _ = request_contract(requested_plan)
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed GINO screen differs from its frozen common-data contract: {job['name']}")
        if summary.get("endpoint_step") != 480 or summary.get("costs", {}).get("train_case_exposures") != 480:
            raise ValueError(f"Completed GINO screen lacks full480 updates: {job['name']}")
        if any(count != 4 for count in summary.get("case_exposure_counts", {}).values()):
            raise ValueError(f"GINO screen did not expose every TRAIN case four times: {job['name']}")
        for filename in ("config.json", "last.pt", "best.pt", "step480.pt", "metrics_latest.json", "first_update_gradient_check.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed GINO screen lacks {filename}: {job['name']}")
    elif "tools/validate_gino_native_resource.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("gino_native_resource", ROOT / "tools/validate_gino_native_resource.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = command[command.index("--plan") + 1]
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed GINO model resource differs from its input-only contract: {job['name']}")
        if summary.get("feasible") is not True or not all(summary.get("checks", {}).values()):
            raise ValueError(f"Completed GINO resource did not pass its bounded resource/backend checks: {job['name']}")
        receipt = summary.get("receipt", {})
        if any(receipt.get(key) != 0 for key in ("labels_read", "weights_read", "scaler_fits", "optimizer_steps", "training_updates")):
            raise ValueError(f"GINO resource unexpectedly trained or accessed forbidden inputs: {job['name']}")
    elif "tools/prepare_gino_radius_pilot.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("gino_radius_pilot", ROOT / "tools/prepare_gino_radius_pilot.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = command[command.index("--plan") + 1]
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed GINO radius pilot differs from fixed input/radius contract: {job['name']}")
        for record in summary.get("inventory", {}).values():
            if module.stat_record(module.host_path(record["path"])) != {key: record[key] for key in ("bytes", "mtime_ns")}:
                raise ValueError(f"Completed GINO radius input cache is missing or changed: {job['name']}")
        if summary.get("receipt", {}).get("labels_read") != 0:
            raise ValueError(f"Input-radius pilot unexpectedly read labels: {job['name']}")
    elif "tools/diagnose_solver_continuous_capability.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("solver_continuous_capability", ROOT / "tools/diagnose_solver_continuous_capability.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = command[command.index("--plan") + 1]
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed TRAIN capability differs from fixed solver/input contract: {job['name']}")
        if summary.get("receipt", {}).get("optimizer_updates") != 0:
            raise ValueError(f"TRAIN capability unexpectedly trained: {job['name']}")
    elif "tools/diagnose_solver_bc_representation.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("solver_bc_representation", ROOT / "tools/diagnose_solver_bc_representation.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if summary.get("contract") != module.request_contract():
            raise ValueError(f"Completed BC-role replay differs from fixed TRAIN input contract: {job['name']}")
        if not (ROOT / job["output"] / "config.json").is_file():
            raise ValueError(f"Completed BC-role replay lacks config.json: {job['name']}")
        if summary.get("input_read_receipt", {}).get("input_only") is not True:
            raise ValueError(f"BC-role replay unexpectedly read non-input data: {job['name']}")
    elif "tools/diagnose_query_learning.py" in command:
        # Frozen diagnostics have no optimizer/checkpoint outputs. Compare their
        # small request/source/last-checkpoint metadata before allowing reuse.
        import importlib.util
        spec = importlib.util.spec_from_file_location("query_learning_diagnostic", ROOT / "tools/diagnose_query_learning.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        contract = module.request_contract(requested_plan)
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed output for {job['name']} differs from its frozen diagnostic contract")
        if not (ROOT / job["output"] / "config.json").is_file():
            raise ValueError(f"Completed output for {job['name']} lacks config.json")
    elif "tools/diagnose_hash_modulation_use.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("hash_modulation_use_diagnostic", ROOT / "tools/diagnose_hash_modulation_use.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its frozen hash gate-use diagnostic contract")
        if not (ROOT / job["output"] / "config.json").is_file():
            raise ValueError(f"Completed output for {job['name']} lacks config.json")
    elif "tools/diagnose_field_error_gradient.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("field_error_gradient_diagnostic", ROOT / "tools/diagnose_field_error_gradient.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its frozen field-error contract")
        for path in (ROOT / job["output"] / "config.json", ROOT / "notes/geobc_error_gradient_result.json"):
            if not path.is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {path}")
    elif "tools/validate_hash_modulation_native.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("hash_modulation_native_check", ROOT / "tools/validate_hash_modulation_native.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its native interface check")
        if not (ROOT / "notes/hash_modulation_native_ready.json").is_file():
            raise ValueError(f"Completed native check for {job['name']} lacks readiness record")
    elif "tools/validate_graph_context_native.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("graph_context_native_check", ROOT / "tools/validate_graph_context_native.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its graph-context native check")
        if not (ROOT / "notes/graph_context_native_ready.json").is_file():
            raise ValueError(f"Completed native check for {job['name']} lacks readiness record")
    elif "tools/prepare_topology_context_balanced.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("balanced_topology_context_preparation", ROOT / "tools/prepare_topology_context_balanced.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its balanced topology contract")
        result_path = ROOT / "notes/geobc_topology_balanced_feasibility_result.json"
        if not result_path.is_file() or read_json_if_exists(result_path).get("status") != "completed":
            raise ValueError(f"Completed input diagnostic for {job['name']} lacks its formal result")
    elif "tools/prepare_topology_context.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("topology_context_preparation", ROOT / "tools/prepare_topology_context.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its topology input contract")
        result_path = ROOT / "notes/geobc_topology_context_feasibility_result.json"
        if not result_path.is_file() or read_json_if_exists(result_path).get("status") != "completed":
            raise ValueError(f"Completed input diagnostic for {job['name']} lacks its formal result")
    elif "tools/diagnose_hash_condition_paths.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("hash_condition_path_diagnostic", ROOT / "tools/diagnose_hash_condition_paths.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its frozen H/C path contract")
        result_path = ROOT / "notes/geobc_hash_condition_path_result.json"
        if not result_path.is_file() or read_json_if_exists(result_path).get("status") != "completed":
            raise ValueError(f"Completed diagnostic for {job['name']} lacks its formal result")
    elif "tools/validate_hash_bandwidth_native.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("hash_bandwidth_native_check", ROOT / "tools/validate_hash_bandwidth_native.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its bandwidth native check")
        if not (ROOT / "notes/hash_bandwidth_native_ready.json").is_file():
            raise ValueError(f"Completed native check for {job['name']} lacks readiness record")
    elif "tools/validate_continuous_query_geometry.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("continuous_geometry_acceptance_check", ROOT / "tools/validate_continuous_query_geometry.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its continuous-geometry acceptance contract")
        result = read_json_if_exists(ROOT / "notes/geobc_continuous_geometry_result.json")
        if not result or result.get("status") != "completed":
            raise ValueError(f"Completed output for {job['name']} lacks its formal continuous-geometry result")
    elif "tools/diagnose_quadratic_kinematics.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("quadratic_kinematic_diagnostic", ROOT / "tools/diagnose_quadratic_kinematics.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its frozen quadratic-kinematic contract")
        if (read_json_if_exists(ROOT / "notes/geobc_quadratic_kinematics_result.json") or {}).get("status") != "completed":
            raise ValueError(f"Completed output for {job['name']} lacks its formal quadratic-kinematic result")
    elif "tools/diagnose_fullcad_quadratic_kinematics.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("fullcad_quadratic_diagnostic", ROOT / "tools/diagnose_fullcad_quadratic_kinematics.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its fullCAD quadratic-kinematic contract")
        if (read_json_if_exists(ROOT / "notes/geobc_fullcad_quadratic_kinematics_result.json") or {}).get("status") != "completed":
            raise ValueError(f"Completed output for {job['name']} lacks its formal fullCAD result")
    elif "geobc.topology_message_train" in command:
        from geobc.topology_message_train import parser as topology_parser, _scope
        parsed = topology_parser().parse_args(command[command.index("geobc.topology_message_train") + 1:])
        if summary.get("config", {}).get("contract") != _scope(parsed)[3]:
            raise ValueError(f"Completed output for {job['name']} differs from its fixed topology-message contract")
        if summary.get("endpoint_step") != 1000 or summary.get("message_mode") != parsed.message_mode:
            raise ValueError(f"Completed output for {job['name']} differs from its fixed endpoint/mode")
        for filename in ("last.pt", "endpoint_1000.pt", "metrics_step0000.json", "metrics_step1000.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "geobc.hash_bandwidth_train" in command:
        from geobc.hash_bandwidth_train import parser as bandwidth_parser, _scope
        parsed = bandwidth_parser().parse_args(command[command.index("geobc.hash_bandwidth_train") + 1:])
        if summary.get("config", {}).get("contract") != _scope(parsed)[4]:
            raise ValueError(f"Completed output for {job['name']} differs from its fixed bandwidth contract")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "geobc.graph_context_train" in command:
        from geobc.graph_context_train import parser as graph_parser, _scope
        parsed = graph_parser().parse_args(command[command.index("geobc.graph_context_train") + 1:])
        if summary.get("config", {}).get("contract") != _scope(parsed)[4]:
            raise ValueError(f"Completed output for {job['name']} differs from its graph-context contract")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "geobc.hash_modulation_train" in command:
        from geobc.hash_modulation_train import parser as modulation_parser, _scope
        parsed = modulation_parser().parse_args(command[command.index("geobc.hash_modulation_train") + 1:])
        if summary.get("config", {}).get("contract") != _scope(parsed)[4]:
            raise ValueError(f"Completed output for {job['name']} differs from its hash modulation contract")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    return summary


def stage_command(job: dict[str, Any]) -> list[str]:
    command = job["command"].copy()
    output = ROOT / job["output"]
    if "geobc.solver_final_eval" in command and "--resume" not in command and (output / "config.json").is_file():
        command.append("--resume")
    fitting = ("geobc.exploration" in command and "--mode" in command
               and command[command.index("--mode") + 1] == "fit")
    if ((fitting or "geobc.conditional_train" in command or "geobc.physics_train" in command
         or "geobc.gradient_train" in command or "geobc.gradient_balance_train" in command
         or "geobc.bc_role_train" in command or "geobc.specialist_train" in command
         or "geobc.query_context_train" in command or "geobc.hash_modulation_train" in command
         or "geobc.graph_context_train" in command or "geobc.hash_bandwidth_train" in command
         or "geobc.topology_message_train" in command or "geobc.solver_selection_train" in command
         or "geobc.solver_confirmation_train" in command or "geobc.fno_response_train" in command
         or "geobc.gino_train" in command
         or ("geobc.deeponet_train" in command and "--evaluate-dia" not in command)
         or ("geobc.multiscale_mesh_train" in command and "--evaluate-dia" not in command)
         or ("geobc.pointconv_train" in command and "--evaluate-dia" not in command))
            and "--resume" not in command and (output / "last.pt").exists()):
        command.extend(["--resume", str(output / "last.pt")])
    return command


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.chdir(ROOT)
    plan = args.plan.resolve()
    state_dir = args.state_dir.resolve()
    if args.dry_run:
        records = []
        for job in load_jobs(plan, args.only):
            summary = completed_summary(job)
            records.append({"name": job["name"], "completed_reuse": summary is not None,
                            "command": stage_command(job), "output": job["output"],
                            "missing_inputs": [path for path in job.get("requires", []) if not (ROOT / path).exists()]})
        result = {"dry_run": True, "plan": str(plan), "serial_GPU_queue": True,
                  "state_directory": str(state_dir), "dynamic_plan_reload": True, "jobs": records}
        print(json.dumps(result, indent=2), flush=True)
        return result
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / ".coordinator.lock"
    lock = exclusive_lock(lock_path)
    try:
        lock.__enter__()
    except RuntimeError:
        print(f"A live coordinator already holds {lock_path}; no duplicate jobs started", flush=True)
        return {"status": "already_running", "lock": str(lock_path)}
    started = time.monotonic()
    state = {"phase": "starting", "plan": str(plan), "pid": os.getpid(),
             "started_at_utc": utc_now(), "completed": [], "failures": [], "child_pid": None}
    def update(**extra):
        state.update(extra, elapsed_seconds=time.monotonic() - started, updated_at_utc=utc_now())
        write_json(state_dir / "coordinator_status.json", state)
    completed_names = set()
    try:
        update()
        while True:
            # Always reload the plan: never retain a stale list after a long stage.
            jobs = load_jobs(plan, args.only)
            state["total_jobs"] = len(jobs)
            pending = []
            for job in jobs:
                if job["name"] in completed_names:
                    continue
                summary = completed_summary(job)
                if summary:
                    completed_names.add(job["name"])
                    state["completed"].append({"name": job["name"], "reused": True,
                                               "elapsed_seconds": summary.get("elapsed_seconds")})
                    print(f"Reuse completed {job['name']}", flush=True)
                else:
                    pending.append(job)
            if not pending:
                update(phase="completed", current=None, child_pid=None, completed_at_utc=utc_now())
                print(f"GeoBC queue completed {len(completed_names)} stages", flush=True)
                return state
            job = pending[0]
            wait_started = time.monotonic()
            while True:
                refreshed = {entry["name"]: entry for entry in load_jobs(plan, args.only)}
                if job["name"] not in refreshed:
                    raise ValueError(f"Waiting job was removed from plan: {job['name']}")
                job = refreshed[job["name"]]
                missing = [path for path in job.get("requires", []) if not (ROOT / path).exists()]
                if not missing:
                    break
                update(phase="waiting_for_inputs", current=job["name"], missing_inputs=missing,
                       waiting_seconds=time.monotonic() - wait_started)
                if time.monotonic() - wait_started > args.input_timeout:
                    raise TimeoutError(f"Missing inputs for {job['name']}: {missing}")
                time.sleep(5)
            command = stage_command(job)
            update(phase="running", current=job["name"], current_stage=job.get("stage"),
                   command=command, missing_inputs=[], current_started_at_utc=utc_now())
            log_path = state_dir / (job["name"] + ".log")
            stage_started = time.monotonic()
            print(f"Start {job['name']}; log {log_path}", flush=True)
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
                update(child_pid=child.pid)
                while child.poll() is None:
                    time.sleep(5)
                    status = read_json_if_exists(ROOT / job["output"] / "status.json")
                    update(current_elapsed_seconds=time.monotonic() - stage_started,
                           child_status=status, eta_seconds=status.get("eta_seconds") if status else None,
                           estimated_completion_utc=status.get("estimated_completion_utc") if status else None)
            if child.returncode:
                failure = {"name": job["name"], "exit_code": child.returncode,
                           "log": str(log_path), "failed_at_utc": utc_now(),
                           "recovery": "rerun coordinator; completed stages reuse, fitting resumes last.pt"}
                state["failures"].append(failure)
                update(phase="failed", child_pid=None)
                raise RuntimeError(f"Stage failed: {job['name']}; see {log_path}")
            summary = completed_summary(job)
            if not summary:
                raise RuntimeError(f"Stage exited successfully without a completed summary: {job['name']}")
            completed_names.add(job["name"])
            state["completed"].append({"name": job["name"], "reused": False,
                                       "elapsed_seconds": time.monotonic() - stage_started})
            update(child_pid=None, child_status=None)
    except BaseException as error:
        update(phase="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        lock.__exit__(None, None, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("notes/geobc_exploration_plan.json"))
    parser.add_argument("--state-dir", type=Path, default=Path("runs/geobc_exploration"))
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--input-timeout", type=float, default=3600)
    run(parser.parse_args())
