"""Independent, conditional fixed-checkpoint readout of a budget-ended pair.

No parent training/clock/contract runner is called. Source checks and contracts
use only stdlib source/JSON/stat; scientific access requires a genuine release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "geobc/elastic_spatial_budget_readout.py"
PLAN = "notes/geobc_spatial_budget_readout_plan.json"
KIND = "budget_limited_spatial_pair_fixed_endpoint_readout"
SCIENCE = "d39d0af8a686fed87dbcfe5931f356b9311862890c798dbf2ceea4e2786132f4"
TRAIN = ("101_428/ver", "101_56/ver", "101_78/ver")
DEV = ("102_249/ver", "102_38/ver")
ARMS = ("local_physical_zero_latent", "global_FNO_physical_latent")
ZERO_KEYS = ("displacement_target_reads", "TRAIN_displacement_target_reads",
    "DEV_displacement_target_reads", "source_csv_or_archive_reads", "canonical_target_member_reads",
    "old_final_reads", "sealed_new_test_reads", "DEV_dia_reads", "new_physical_solves",
    "new_tensor_recoveries", "K_guards_repeated", "external_model_or_weight_reads",
    "optimizer_steps", "training_updates", "scaler_fits", "DEV_scaler_fits", "TRAIN_stress_reads")
ACTUAL_KEYS = ("GT_reads", "stress_reads", "DEV_stress_reads", "cached_stress_array_reads",
    "parent_evaluation_cache_reads", "own_evaluation_cache_reads", "cached_input_array_reads",
    "physical_U_input_reads", "physical_U_copy_reads", "parent_checkpoint_reads",
    "parent_checkpoint_hash_reads", "own_prediction_array_reads")


def path(value):
    value = str(value)
    if os.name == "nt" and value.startswith("/mnt/"):
        value = value[5].upper() + ":/" + value[7:]
    item = Path(value)
    return item if item.is_absolute() else ROOT / item


def read(value):
    return json.loads(path(value).read_text(encoding="utf-8"))


def write(value, data):
    destination = path(value); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def sha(value):
    h = hashlib.sha256()
    with path(value).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stat_match(record):
    info = path(record["path"]).stat()
    return info.st_size == record["bytes"] and info.st_mtime_ns == record["mtime_ns"]


def pinned(record):
    require(sha(record["path"]) == record["sha256"], "Frozen small-JSON identity differs: " + record["path"])
    return read(record["path"])


def metric_complete(metric):
    if not all(finite(metric.get(name)) for name in ("RMSE_MPa", "MAE_MPa", "max_abs_error_MPa")):
        return False
    return all(finite(metric.get(name)) or (metric.get(zero) is True and metric.get(name) is None)
        for name, zero in (("stress_relative_l2", "zero_reference"), ("hotspot_relative_l2", "hotspot_zero_reference")))


def complete_parent_metrics(summary):
    return all(metric_complete(summary.get("metrics", {}).get(arm, {}).get("per_case", {}).get(key, {}))
        for arm in ARMS for key in DEV)


def source_check(plan_path=PLAN):
    """Pending plans are legal here; never inspect live checkpoint bytes."""
    plan = read(plan_path)
    require(plan["kind"] == KIND and tuple(plan["arms"]) == ARMS and tuple(plan["validation_keys"]) == DEV,
        "Conditional readout scope differs")
    require(plan["scientific_digest"] == SCIENCE and plan["recipe"]["maximum_cumulative_readout_wall_seconds"] == 180
        and plan["recipe"]["maximum_GPU_peak_bytes"] == 24 * 1024**3
        and plan["recipe"]["checkpoint_selection"] == "last_committed_at_parent_terminal_snapshot"
        and plan["recipe"]["training_updates"] == 0 and plan["recipe"]["scaler_fits"] == 0,
        "Frozen readout recipe differs")
    actual = {name: sha(name) for name in plan["source_sha256"]}
    require(actual == plan["source_sha256"] and SOURCE in actual, "Frozen source identity differs")
    return {"status": "source_check_passed", "kind": KIND, "source_sha256": actual,
        "scientific_digest": SCIENCE, "pending_parent_binding_allowed": True,
        "array_target_checkpoint_import_GPU_accesses": 0}


def request_contract(plan_path=PLAN):
    """Strict metadata-only contract, including terminal eligibility and stats."""
    source_check(plan_path); plan = read(plan_path)
    require(plan["scientific_rules_frozen"] is True and plan["actual_execution_released"] is False,
        "Only frozen science with an independent root marker may execute")
    bindings = plan["parent_terminal_bindings"]
    require(isinstance(bindings, dict), "Parent terminal snapshot has not been bound")
    parent = pinned(bindings["summary"]); parent_status = pinned(bindings["status"])
    parent_clock = pinned(bindings["clock"]); parent_receipt = pinned(bindings["read_receipt"])
    require(parent["kind"] == "spatial_physical_signed_recovery_pair"
        and parent["status"] in ("completed", "failed") and parent_status["status"] == parent["status"]
        and parent_clock["active"] is False, "Parent is not a frozen terminal stage")
    parent_contract = parent["config"]["contract"]
    require(parent_contract["scientific_digest"] == SCIENCE
        and tuple(parent_contract["validation_keys"]) == DEV and tuple(parent_contract["arms"]) == ARMS
        and parent_contract["source_sha256"]["geobc/elastic_spatial_recovery.py"] == plan["source_sha256"]["geobc/elastic_spatial_recovery.py"],
        "Parent model/science/source differs")
    skipped = complete_parent_metrics(parent)
    if not skipped:
        reason = parent.get("completion_gate", {}).get("terminal_negative_reason") or ""
        require("campaign wall budget exhausted" in reason.lower() and parent.get("technical_error") is None
            and parent["completion_gate"]["passed"] is False, "Conditional readout requires an actual campaign-budget terminal negative")
        local = pinned(bindings["arms"][ARMS[0]]["training_summary"])
        require(local["complete"] is True and local["offset"] == 0 and local["steps"] == 3 * local["pass_index"]
            and local["training_seconds"] >= 600 and local["first_gradient"]["passed"] is True
            and set(local["case_updates"]) == set(TRAIN)
            and all(value == local["pass_index"] for value in local["case_updates"].values())
            and local["config"] == {"contract": parent_contract, "arm": ARMS[0]}, "Local600 endpoint is not complete")
        for arm in ARMS:
            record = bindings["arms"][arm]
            require(stat_match(record["checkpoint"]), "Last committed terminal checkpoint stat differs")
            require(path(record["checkpoint"]["path"]) == path(parent_contract["output"]) / arm / "last.pt",
                "Endpoint is not the last committed parent checkpoint")
            journal = pinned(record["status"])
            require(journal.get("active") is False, "Parent arm journal remains active")
        require(stat_match(bindings["initial_common"]), "Parent common checkpoint stat differs")
    scalers = pinned(bindings["scalers"])
    require(scalers["parent_scalers"] == parent_contract["parent_scalers"] and tuple(scalers["fit_keys"]) == TRAIN,
        "Frozen TRAIN-only normalizers differ")
    records = {key: parent_contract["records"][key] for key in DEV}
    for key, record in records.items():
        for name in ("xyz", "features", "cloud", "mask", "valid", "sigma_MPa"):
            require(stat_match(record["input_arrays"][name]), "Frozen input cache stat differs")
        require(stat_match(record["original_zero_physical_U"]) and stat_match(record["cached_original_signed_stress"]),
            "Original U or signed-S cache stat differs")
    return {"schema": "geobc_budget_fixed_endpoint_readout_v1", "kind": KIND,
        "source_sha256": plan["source_sha256"], "scientific_digest": SCIENCE, "science_recipe_sha256": SCIENCE,
        "recipe": plan["recipe"], "validation_keys": list(DEV), "case_keys": list(DEV), "arms": list(ARMS),
        "parent_bindings": bindings, "parent_contract": parent_contract,
        "parent_terminal_summary": parent, "parent_terminal_clock": parent_clock,
        "parent_read_receipt_context": parent_receipt, "scalers": scalers, "records": records,
        "skipped_parent_already_has_metrics": skipped, "output": str(path(plan["output"])),
        "array_or_checkpoint_reads_in_request_contract": 0}, plan


def released(contract, plan):
    marker = read(plan["root_review_marker"])
    require(marker["status"] == "passed" and marker["decision"] == "proceed_conditional_fixed_endpoint_readout"
        and marker["contract_sha256"] == digest(contract) and marker["source_sha256"] == contract["source_sha256"],
        "Genuine root release is missing or differs")


class Negative(RuntimeError):
    pass


class ReadoutClock:
    def __init__(self, output, invocation_start):
        self.file = output / "clock.json"; old = read(self.file) if self.file.exists() else {}
        self.prior = float(old.get("cumulative_readout_wall_seconds", 0)); self.gap = float(old.get("unobserved_exit_gap_seconds", 0))
        if old.get("active"):
            extra = max(0.0, time.time() - old["observed_unix_seconds"])
            self.prior += extra; self.gap += extra
        self.start = invocation_start; self.torch = None; self.attempts = int(old.get("attempts", 0)) + 1
        self.allocated = int(old.get("peak_GPU_allocated_bytes", 0)); self.reserved = int(old.get("peak_GPU_reserved_bytes", 0))
        self.persist()

    def sync(self):
        if self.torch is not None:
            self.torch.cuda.synchronize()

    def seconds(self):
        self.sync(); return self.prior + time.perf_counter() - self.start

    def persist(self, active=True):
        if self.torch is not None:
            self.allocated = max(self.allocated, self.torch.cuda.max_memory_allocated())
            self.reserved = max(self.reserved, self.torch.cuda.max_memory_reserved())
        write(self.file, {"cumulative_readout_wall_seconds": self.seconds(), "maximum_wall_seconds": 180,
            "parent_budget_reset": False, "parent_costs_context_only": True, "attempts": self.attempts,
            "unobserved_exit_gap_seconds": self.gap, "active": active, "observed_unix_seconds": time.time(),
            "peak_GPU_allocated_bytes": self.allocated, "peak_GPU_reserved_bytes": self.reserved})

    def guard(self):
        self.persist()
        if self.seconds() >= 180:
            raise Negative("Independent cumulative180s readout budget exhausted")
        if max(self.allocated, self.reserved) > 24 * 1024**3:
            raise Negative("Independent24GiB readout memory budget exceeded")


class Ledger:
    def __init__(self, output):
        self.file = output / "read_receipt.json"
        self.data = read(self.file) if self.file.exists() else {name: 0 for name in ZERO_KEYS + ACTUAL_KEYS}
        require(all(type(self.data.get(name)) is int and self.data[name] == 0 for name in ZERO_KEYS), "Forbidden read or update receipt differs")
        require(all(type(self.data.get(name)) is int and self.data[name] >= 0 for name in ACTUAL_KEYS), "Actual read receipt differs")
        self.data.setdefault("attempted_reads", {}); self.data.setdefault("completed_reads", {}); self.persist()

    def persist(self):
        write(self.file, self.data)

    def bump(self, name, token=None):
        self.data[name] += 1
        if token is not None:
            self.data["attempted_reads"][token] = self.data["attempted_reads"].get(token, 0) + 1
        self.persist()

    def complete(self, token):
        self.data["completed_reads"][token] = self.data["completed_reads"].get(token, 0) + 1; self.persist()


def checkpoint_load(torch, record, ledger, token, clock):
    clock.guard(); require(stat_match(record), "Parent checkpoint stat differs before read")
    ledger.bump("parent_checkpoint_reads", token)
    payload = torch.load(path(record["path"]), map_location="cpu", weights_only=False)
    ledger.complete(token); ledger.bump("parent_checkpoint_hash_reads", token + ":sha")
    actual_sha = sha(record["path"]); ledger.complete(token + ":sha")
    require(stat_match(record), "Parent checkpoint stat differs after read")
    clock.guard(); return payload, actual_sha


def prepare_inputs(np, parent, contract, ledger, output, clock):
    items = {}; scalers = contract["scalers"]; sigma = scalers["parent_scalers"]["tensor6"]
    write(output / "scalers.json", {**scalers, "new_scaler_fits": 0, "source": "immutable parent serialized scaler"})
    for key in DEV:
        clock.guard(); record = contract["records"][key]
        item = {name: parent.array_load(np, record["input_arrays"][name], ledger, "cached_input_array_reads", key + ":" + name)
            for name in ("xyz", "features", "cloud", "mask", "valid", "sigma_MPa")}
        u = parent.array_load(np, record["original_zero_physical_U"], ledger, "physical_U_input_reads", key + ":U", record["physical_U_file_sha256"])
        n = record["node_count"]
        require(item["xyz"].shape == (n, 3) and item["features"].shape == (n, 25) and u.shape == (n, 3)
            and item["sigma_MPa"].shape == (n, 6) and item["valid"].shape == (n,)
            and item["valid"].dtype == np.bool_, "Parent original-node input schema differs")
        item["u_normalized"] = (u / np.asarray(scalers["physical_U_zero_center_component_RMS_mm"])).astype(np.float32)
        item["sigma_normalized"] = ((item["sigma_MPa"] - np.asarray(sigma["mean"])) / np.asarray(sigma["std"])).astype(np.float32)
        item["sigma_normalized"][~item["valid"]] = 0
        require(np.isfinite(item["u_normalized"]).all() and np.isfinite(item["sigma_normalized"]).all(), "Nonfinite frozen normalization")
        item["grid_cpu"] = None; items[key] = item
    return items


def prediction_location(output, arm, key):
    directory = output / arm / "predictions" / key.replace("/", "_")
    return directory, directory / "prediction_metadata.json"


def validate_prediction(meta, contract, arm, key, endpoint):
    require(meta["status"] == "completed" and meta["config"] == {"contract": contract, "arm": arm, "case_key": key}
        and meta["endpoint_sha256"] == endpoint["checkpoint_sha256"] and meta["full_original_node_count"] == contract["records"][key]["node_count"]
        and stat_match(meta["signedS"]) and stat_match(meta["physical_U"])
        and meta["physical_U"]["file_sha256"] == contract["records"][key]["physical_U_file_sha256"], "Committed fixed-endpoint prediction differs")


def predict_arm(np, torch, parent, cls, arm, common, items, contract, ledger, output, clock):
    directory = output / arm; directory.mkdir(parents=True, exist_ok=True)
    endpoint_file = directory / "endpoint_metadata.json"; selected = contract["parent_bindings"]["arms"][arm]
    existing = read(endpoint_file) if endpoint_file.exists() else None
    predictions = {}; missing = []
    for key in DEV:
        _, file = prediction_location(output, arm, key)
        if file.exists():
            require(existing is not None, "Prediction exists without frozen endpoint metadata")
            meta = read(file); validate_prediction(meta, contract, arm, key, existing); predictions[key] = meta
        else:
            missing.append(key)
    if not missing:
        return {**existing, "predictions": predictions}
    clock.guard(); loaded_start = time.perf_counter()
    payload, checkpoint_sha = checkpoint_load(torch, selected["checkpoint"], ledger, arm + ":last", clock)
    require(payload["config"] == {"contract": contract["parent_contract"], "arm": arm}, "Actual selected checkpoint arm/config differs")
    state = payload["state"]
    require(type(state["steps"]) is int and state["steps"] > 0 and state["steps"] == 3 * state["pass_index"] + state["offset"]
        and 0 <= state["offset"] < 3 and set(state["case_updates"]) == set(TRAIN)
        and sum(state["case_updates"].values()) == state["steps"] and state["first_gradient"]["passed"] is True,
        "Actual selected checkpoint has invalid exposure/gradient state")
    if arm == ARMS[0]:
        summary = pinned(selected["training_summary"])
        require(state["steps"] == summary["steps"] and state["offset"] == 0 and summary["complete"] is True,
            "Selected local checkpoint is not its complete600 endpoint")
    journal = pinned(selected["status"])
    model, initial_identity = parent.create_model(np, torch, cls, arm, common)
    require(initial_identity == pinned(contract["parent_bindings"]["initial_common_metadata"])["identity"], "Parent common initialization differs")
    model.load_state_dict(payload["model"], strict=True); model.eval(); clock.sync()
    endpoint = {"status": "selected", "arm": arm, "config": {"contract": contract, "arm": arm},
        "checkpoint": selected["checkpoint"], "checkpoint_sha256": checkpoint_sha,
        "state_disclosure": {name: state.get(name) for name in ("steps", "pass_index", "offset", "case_updates", "case_order_prefix", "training_seconds", "first_gradient", "uncommitted_updates_replayed_after_restore")},
        "parent_journal_disclosure": {name: journal.get(name) for name in ("steps", "pass_index", "offset", "training_seconds", "failed_actual_training_seconds", "uncommitted_updates_replayed_after_restore")},
        "committed_training_seconds": state["training_seconds"], "journal_charged_training_seconds": journal["training_seconds"],
        "uncommitted_or_cost_only_training_seconds": max(0.0, journal["training_seconds"] - state["training_seconds"]),
        "parameter_counts": {"PointNet": sum(p.numel() for p in model.encoder.parameters()), "head": sum(p.numel() for p in model.head.parameters()),
            "FNO_global": 0 if model.global_fno is None else sum(p.numel() for p in model.global_fno.parameters()), "total": sum(p.numel() for p in model.parameters())},
        "model_create_checkpoint_load_seconds_this_invocation": time.perf_counter() - loaded_start,
        "parent_pair_pass_claimed": False, "capacity_matched": False}
    if existing is not None:
        require(existing["checkpoint_sha256"] == checkpoint_sha and existing["state_disclosure"] == endpoint["state_disclosure"], "Selected endpoint changed on resume")
    else:
        write(endpoint_file, endpoint)
    sscale = contract["scalers"]["parent_scalers"]["signed_stress"]
    for key in missing:
        clock.guard(); folder, meta_file = prediction_location(output, arm, key); folder.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter(); gpu, input_seconds = parent.gpu_item(np, torch, items[key], arm == ARMS[1], clock)
        values = np.empty(contract["records"][key]["node_count"], dtype=np.float64)
        with torch.no_grad():
            context, latent = model.fields(gpu)
            for start in range(0, len(values), 32768):
                clock.guard(); end = min(start + 32768, len(values))
                encoded = model.points(gpu, start, end, context, latent).double().cpu().numpy()
                values[start:end] = np.sinh(encoded * sscale["std"] + sscale["mean"]) * sscale["scale_MPa"]
        if not np.isfinite(values).all():
            raise Negative("Fixed-checkpoint allN signed-S prediction is nonfinite")
        clock.guard()
        from .elastic_stress_readout import save_array
        stress = save_array(np, folder / "signedS.npy", values)
        original = contract["records"][key]
        ledger.bump("physical_U_copy_reads", arm + ":" + key + ":U_copy")
        temporary = folder / "physical_u_mm.npy.tmp"; shutil.copyfile(path(original["original_zero_physical_U"]["path"]), temporary)
        require(sha(temporary) == original["physical_U_file_sha256"], "Copied original U bytes differ")
        os.replace(temporary, folder / "physical_u_mm.npy"); ledger.complete(arm + ":" + key + ":U_copy")
        info = (folder / "physical_u_mm.npy").stat()
        u = {"path": str(folder / "physical_u_mm.npy"), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
            "shape": [len(values), 3], "dtype": "float64", "file_sha256": original["physical_U_file_sha256"],
            "new_U_GT_metrics": None, "preserved_original_byte_identity": True}
        meta = {"status": "completed", "config": {"contract": contract, "arm": arm, "case_key": key},
            "endpoint_sha256": checkpoint_sha, "selected_step": state["steps"], "full_original_node_count": len(values),
            "signedS": stress, "physical_U": u, "costs": {"new_prediction_wall_seconds": time.perf_counter() - started,
                "input_H2D_targetfree_grid_seconds": input_seconds, "new_cost_not_parent_budget_refill": True},
            "prediction_persisted_unix_seconds": time.time(), "read_receipt": dict(ledger.data)}
        write(meta_file, meta); write(folder / "read_receipt.json", ledger.data); predictions[key] = meta
        del gpu, context, latent, encoded, values; clock.guard()
    del model, payload; torch.cuda.empty_cache()
    return {**endpoint, "predictions": predictions}


def read_DEV_label(np, parent, contract, key, ledger, output, clock):
    file = output / "evaluation_labels" / (key.replace("/", "_") + ".json")
    if file.exists():
        return parent.array_load(np, read(file), ledger, "own_evaluation_cache_reads", "own_DEV:" + key)
    token = "DEV:" + key
    require(ledger.data["attempted_reads"].get(token, 0) == 0, "Uncommitted readout DEV label attempt cannot silently repeat")
    parent_copy = path(contract["parent_contract"]["output"]) / "evaluation_labels" / (key.replace("/", "_") + ".json")
    record = contract["records"][key]["cached_original_signed_stress"]
    counter = "cached_stress_array_reads"
    if parent_copy.exists():
        saved = read(parent_copy)
        require(all(saved.get(name) == record.get(name) for name in ("shape", "dtype", "values_sha256")), "Committed parent label copy identity differs")
        record = saved; counter = "parent_evaluation_cache_reads"
    else:
        prior_token = "DEV:" + key
        receipt = contract["parent_read_receipt_context"]
        require(receipt.get("attempted_reads", {}).get(prior_token, 0) == 0,
            "Parent attempted a DEV source read without a committed copy; explicit recovery is required")
    clock.guard(); ledger.bump("GT_reads"); ledger.bump("stress_reads"); ledger.bump("DEV_stress_reads")
    raw = parent.array_load(np, record, ledger, counter, token)
    from .elastic_stress_readout import save_array
    saved = save_array(np, file.with_suffix(".npy"), raw); write(file, saved)
    return raw


def evaluate(np, parent, contract, arms, ledger, output, clock):
    identities = {}
    for arm in ARMS:
        identities[arm] = {}
        for key in DEV:
            validate_prediction(arms[arm]["predictions"][key], contract, arm, key, arms[arm])
            identities[arm][key] = sha(prediction_location(output, arm, key)[1])
    barrier = {"both_fixed_arms_allDEV_predictions_persisted_before_labels": True, "validation_keys": list(DEV),
        "prediction_manifest_sha256": identities, "endpoint_sha256": {arm: arms[arm]["checkpoint_sha256"] for arm in ARMS}}
    barrier_file = output / "prediction_barrier.json"
    if barrier_file.exists():
        require(read(barrier_file) == barrier, "Frozen readout prediction barrier differs")
    else:
        require(ledger.data["DEV_stress_reads"] == 0, "Readout DEV labels precede complete prediction barrier")
        write(barrier_file, barrier)
    truth = {key: read_DEV_label(np, parent, contract, key, ledger, output, clock) for key in DEV}
    from .elastic_stress_readout import signed_metrics, aggregate
    for arm in ARMS:
        per_case = {}
        for key in DEV:
            folder, _ = prediction_location(output, arm, key); clock.guard()
            pred = parent.array_load(np, arms[arm]["predictions"][key]["signedS"], ledger, "own_prediction_array_reads", arm + ":" + key + ":S")
            metric = signed_metrics(np, truth[key], pred); per_case[key] = metric
            write(folder / "summary.json", {"status": "completed", "config": {"contract": contract, "arm": arm, "case_key": key},
                "metrics": metric, "new_U_GT_metrics": None, "parent_pair_pass_claimed": False})
        arms[arm]["metrics"] = aggregate(per_case, DEV)
    write(output / "evaluation_receipt.json", {"barrier": barrier, "read_receipt": ledger.data,
        "GT_reads_are_new_stage_cached_label_reads": True, "original_source_stress_member_reads": 0})


def run(plan_path=PLAN, output=None):
    began = time.perf_counter(); contract, plan = request_contract(plan_path); released(contract, plan)
    output = path(output or plan["output"]); require(output == path(contract["output"]), "Readout output differs")
    output.mkdir(parents=True, exist_ok=True); config = {"contract": contract, "kind": KIND}
    summary_file = output / "summary.json"
    if summary_file.exists():
        previous = read(summary_file); require(previous["config"] == config, "Saved terminal readout identity differs")
        if previous["status"] == "completed":
            return previous
    write(output / "config.json", config); ledger = Ledger(output); clock = ReadoutClock(output, began)
    arms = {arm: {"arm": arm, "metrics": None, "predictions": {}} for arm in ARMS}
    negative = None; technical = None; skipped = contract["skipped_parent_already_has_metrics"]
    try:
        if not skipped:
            clock.guard()
            import numpy as np
            import torch
            from . import elastic_spatial_recovery as parent
            require(torch.cuda.is_available(), "Frozen readout CUDA backend unavailable")
            torch.cuda.init(); torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
            torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.set_num_threads(8)
            clock.torch = torch
            common_payload, common_sha = checkpoint_load(torch, contract["parent_bindings"]["initial_common"], ledger, "initial_common", clock)
            require(common_payload["config"] == contract["parent_terminal_summary"]["config"], "Parent initial common config differs")
            common = common_payload["common"]
            require(parent.common_identity(torch, common) == pinned(contract["parent_bindings"]["initial_common_metadata"])["identity"], "Parent initial common values differ")
            write(output / "endpoint_selection.json", {"selection": "last_committed_at_parent_terminal_snapshot",
                "bindings": contract["parent_bindings"], "initial_common_checkpoint_sha256": common_sha,
                "selection_precedes_DEV_label_access": True, "parent_pair_pass_claimed": False})
            items = prepare_inputs(np, parent, contract, ledger, output, clock); cls = parent.model_class(torch)
            for arm in ARMS:
                arms[arm] = predict_arm(np, torch, parent, cls, arm, common, items, contract, ledger, output, clock)
            evaluate(np, parent, contract, arms, ledger, output, clock); clock.guard()
    except Negative as error:
        negative = str(error)
    except Exception as error:
        technical = type(error).__name__ + ": " + str(error)
    finally:
        clock.persist(False); ledger.persist()
    finite_metrics = all(metric_complete((arms[arm].get("metrics") or {}).get("per_case", {}).get(key, {}))
        for arm in ARMS for key in DEV)
    evidence = not skipped and negative is None and technical is None and finite_metrics
    gates = {"complete_finite_evidence": evidence, "skipped_parent_already_has_metrics": skipped,
        "terminal_negative_reason": negative or technical, "parent_pair_pass_claimed": False, "capacity_matched": False,
        "fair600s600s_ranking_claimed": False, "parent_gate_permanently_negative": not skipped}
    descriptive = {}
    if evidence:
        local = arms[ARMS[0]]["metrics"]["stress_relative_l2_mean"]
        value = arms[ARMS[1]]["metrics"]["stress_relative_l2_mean"]
        descriptive = {"global_vs_local_mean_relative_difference": (local - value) / local if finite(local) and local > 0 and finite(value) else None,
            "each_fixed_endpoint_DEV_S_and_hot_below25pct": {arm: all(finite(arms[arm]["metrics"]["per_case"][key][name]) and arms[arm]["metrics"]["per_case"][key][name] < .25
                for key in DEV for name in ("stress_relative_l2", "hotspot_relative_l2")) for arm in ARMS},
            "thresholds_descriptive_only_not_parent_gate": True}
    summary = {"status": "failed" if technical else "completed", "kind": KIND, "config": config,
        "validation_keys": list(DEV), "case_keys": list(DEV), "arms": arms,
        "cases": {key: {"node_count": contract["records"][key]["node_count"], "new_U_GT_metrics": None,
            "original_U_byte_identity_expected": contract["records"][key]["physical_U_file_sha256"]} for key in DEV},
        "completion_gate": gates, "descriptive_readout": descriptive, "costs": read(clock.file), "read_receipt": ledger.data,
        "parent_costs_context": contract["parent_terminal_clock"], "parent_completion_gate_preserved": contract["parent_terminal_summary"]["completion_gate"],
        "technical_error": technical, "training_updates": 0, "scaler_fits": 0,
        "limitations": ["Unequal fixed training endpoints; not a600s/600s or capacity-matched comparison", "No final solver/generalization or exact source-exporter claim", "Original U byte copy only; no new U accuracy", "Independent readout costs cannot repair the parent budget or gate"]}
    write(summary_file, summary); write(output / "status.json", {"status": summary["status"], "completion_gate": gates, "costs": summary["costs"]})
    clock.persist(False); summary["costs"] = read(clock.file)
    if summary["costs"]["cumulative_readout_wall_seconds"] >= 180 and evidence:
        summary["completion_gate"]["complete_finite_evidence"] = False
        summary["completion_gate"]["terminal_negative_reason"] = "Independent readout budget exhausted including final receipt persistence"
    write(summary_file, summary)
    write(output / "status.json", {"status": summary["status"], "completion_gate": summary["completion_gate"], "costs": summary["costs"]})
    if technical:
        raise RuntimeError(technical)
    return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--plan", default=PLAN); parser.add_argument("--output")
    parser.add_argument("--source-check", action="store_true"); args = parser.parse_args()
    if args.source_check:
        print(json.dumps(source_check(args.plan), ensure_ascii=False))
    else:
        run(args.plan, args.output)


if __name__ == "__main__":
    main()
