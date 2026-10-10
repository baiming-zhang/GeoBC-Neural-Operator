"""Frozen full-FE native-node versus four-Q tensor representation pair.

No scientific dependencies, arrays, labels, weights or CUDA are accessed at
import time. The first released run is the runtime acceptance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_nodal_tensor_pair_plan.json"
KIND = "physical_nodal_tensor_signed_S_representation_pair"
OWN = "geobc/elastic_nodal_tensor_pair.py"
SCIENCE = "0cdfd4ae71d2de1ce26d6bb6d5fe517d1920d7cd66d49eb7001b4ad49c8fa5b8"
TRAIN = ("101_428/ver", "101_56/ver", "101_78/ver")
DEV = ("102_249/ver", "102_38/ver")
CASES = TRAIN + DEV
ARMS = ("fourQ_cellmean", "element_node")
FORBIDDEN = ("displacement_target_reads", "source_canonical_stress_reads", "source_canonical_strain_reads",
    "source_canonical_CSV_reads", "external_learned_weight_reads", "new_physical_solves", "K_applications",
    "old_K_guard_repeats", "old_final_reads", "sealed_new_test_reads", "DEV_dia_reads",
    "DEV_reads_before_prediction_barrier", "DEV_scaler_fits", "source_formula_or_sign_trials")
ACTUAL = ("physical_U_reads", "physical_U_byte_hash_reads", "input_cache_reads", "input_member_reads",
    "operator_weight_array_reads", "old_tensor_array_reads", "valid_array_reads", "new_nodal_tensor_recoveries",
    "TRAIN_cached_S_reads", "DEV_cached_S_reads", "GT_reads", "tensor_scaler_fits", "training_updates",
    "optimizer_updates", "own_checkpoint_reads", "own_checkpoint_writes", "own_prediction_array_reads",
    "original_U_byte_copies")


class ScientificNegative(RuntimeError):
    pass


def path(value):
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def read(value):
    return json.loads(path(value).read_text(encoding="utf-8"))


def write(value, data):
    value = path(value); value.parent.mkdir(parents=True, exist_ok=True)
    temporary = value.with_name(value.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, value)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(value):
    h = hashlib.sha256()
    with path(value).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stat_guard(record):
    p = path(record["path"]); stat = p.stat()
    require("bytes" not in record or stat.st_size == record["bytes"], "Pinned file size differs: " + str(p))
    require("mtime_ns" not in record or stat.st_mtime_ns == record["mtime_ns"], "Pinned file mtime differs: " + str(p))


def request_contract(plan_path=PLAN):
    """Metadata only: never call a parent scientific entrypoint or load arrays."""
    plan = read(plan_path)
    require(plan["kind"] == KIND and plan["scientific_digest"] == SCIENCE
        and digest(plan["science"]) == SCIENCE, "Frozen nodal pair science differs")
    science = plan["science"]
    require(tuple(science["case_keys"]) == CASES and tuple(science["train_keys"]) == TRAIN
        and tuple(science["validation_keys"]) == DEV and tuple(science["arms"]) == ARMS, "Frozen case/arm inventory differs")
    require(science["neural"]["full_case_updates_per_arm"] == 120 and science["neural"]["seed"] == 2026
        and science["neural"]["point_chunk"] == 32768 and science["representation"]["cell_chunk"] == 8192,
        "Frozen exposure or chunks differ")
    schedule = science["neural"]["case_order"]
    require(len(schedule) == 120 and digest(schedule) == science["neural"]["case_order_sha256"]
        and all(sorted(schedule[i:i+3]) == sorted(TRAIN) for i in range(0, 120, 3)), "Fixed balanced schedule differs")
    sources = {name: file_hash(name) for name in plan["source_sha256"]}
    require(sources == plan["source_sha256"] and sources.get(OWN) is not None, "Source pins differ")
    for record in plan["evidence_pins"].values():
        require(file_hash(record["path"]) == record["sha256"], "Parent metadata pin differs")
    frozen_design = read(plan["design"])
    require(frozen_design["science"] == science and frozen_design["records"] == plan["records"],
        "Frozen representation design or case/cache identities differ")
    parent = read(plan["parent_summary"])
    require(parent["status"] == "completed" and parent["kind"] == "elastic_signed_stress_readout_screen"
        and tuple(parent["train_keys"]) == TRAIN and tuple(parent["validation_keys"]) == DEV,
        "Accepted parent scope differs")
    require(parent["scalers"]["signed_stress"] == science["normalization"]["scaler_record"], "Parent signed TRAIN scaler differs")
    for key in CASES:
        record = plan["records"][key]
        for name in ("input_manifest", "operator_summary", "original_zero_summary", "old_feature_manifest", "label_committed"):
            require(file_hash(record[name]["path"]) == record[name]["sha256"], "Per-case parent metadata pin differs: " + key)
        manifest = read(record["input_manifest"]["path"])
        require(manifest["status"] == "completed" and manifest["node_count"] == record["nodes"]
            and manifest["cell_count"] == record["cells"] and manifest["labels_read"] == 0,
            "Accepted target-free physical cache differs: " + key)
        for name in ("input_cache", "original_physical_U", "accepted_positive_fourQ_weights", "old_mean_tensor", "same_valid", "cached_original_signed_S"):
            stat_guard(record[name])
        require(read(record["label_committed"]["path"])["identity"] == record["label_identity"], "Committed source S identity differs")
    inherited = sum(science["budgets"][k] for k in ("inherited_original_GT_free_physics_seconds_once",
        "inherited_accepted_fourQ_recovery_seconds_once", "inherited_accepted_validity_seconds_once"))
    require(abs(inherited + 300 - science["budgets"]["maximum_campaign_composed_wall_seconds"]) <= 1e-8,
        "Frozen inherited campaign arithmetic differs")
    contract = {"schema": "geobc_nodal_tensor_pair_v1", "kind": KIND, "source_sha256": sources,
        "scientific_digest": SCIENCE, "case_keys": list(CASES), "train_keys": list(TRAIN),
        "validation_keys": list(DEV), "arms": list(ARMS), "science": science, "records": plan["records"],
        "evidence_pins": plan["evidence_pins"], "output": plan["output"],
        "array_GT_weight_reads_in_request_contract": 0}
    return contract, plan


class Clock:
    def __init__(self, output, inherited, started=None):
        self.output = output; self.file = output / "clock.json"
        old = read(self.file) if self.file.exists() else {}
        self.prior = float(old.get("new_actual_wall_seconds", 0.0)); self.gap = float(old.get("unobserved_exit_gap_seconds", 0.0))
        if old.get("active"):
            gap = max(0., time.time() - old["observed_unix_seconds"])
            self.prior += gap; self.gap += gap
        self.inherited = inherited; self.start = started if started is not None else time.perf_counter(); self.last_persist = self.start
        self.attempts = int(old.get("attempts", 0)) + 1; self.torch = None
        self.peak_allocated = int(old.get("peak_GPU_allocated_bytes", 0)); self.peak_reserved = int(old.get("peak_GPU_reserved_bytes", 0))
        rp = output / "read_receipt.json"
        self.receipt = read(rp) if rp.exists() else {name: 0 for name in FORBIDDEN + ACTUAL}
        self.receipt.setdefault("attempted_members", {}); self.receipt.setdefault("completed_members", {})
        journals = [read(output / arm / "training_state.json") for arm in ARMS
            if (output / arm / "training_state.json").exists()]
        self.receipt["training_updates"] = max(self.receipt["training_updates"],
            sum(v.get("actual_attempted_updates", 0) for v in journals))
        self.receipt["optimizer_updates"] = max(self.receipt["optimizer_updates"],
            sum(v.get("actual_optimizer_updates", 0) for v in journals))
        self.persist("starting")

    def elapsed(self):
        return self.prior + time.perf_counter() - self.start

    def memory(self):
        if self.torch is not None:
            self.peak_allocated = max(self.peak_allocated, self.torch.cuda.max_memory_allocated())
            self.peak_reserved = max(self.peak_reserved, self.torch.cuda.max_memory_reserved())
        return {"peak_GPU_allocated_bytes": self.peak_allocated, "peak_GPU_reserved_bytes": self.peak_reserved}

    def persist(self, phase="active", active=True):
        write(self.file, {"new_actual_wall_seconds": self.elapsed(), "inherited_seconds_once": self.inherited,
            "campaign_composed_wall_seconds": self.inherited + self.elapsed(), "active": active,
            "observed_unix_seconds": time.time(), "attempts": self.attempts,
            "unobserved_exit_gap_seconds": self.gap, "phase": phase, **self.memory()})
        write(self.output / "read_receipt.json", self.receipt); self.last_persist = time.perf_counter()

    def count(self, name, member=None):
        self.receipt[name] += 1
        if member is not None:
            entries = self.receipt["attempted_members"]; entries[member] = entries.get(member, 0) + 1

    def completed(self, member):
        entries = self.receipt["completed_members"]; entries[member] = entries.get(member, 0) + 1

    def guard(self):
        if self.elapsed() >= 300:
            raise ScientificNegative("Fixed cumulative new300s stage budget exhausted")
        if max(self.memory().values()) > 24 * 1024 ** 3:
            raise ScientificNegative("Fixed cumulative24GiB GPU peak exceeded")
        if time.perf_counter() - self.last_persist >= 60:
            self.persist()


def arraysave(np, value, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, destination)
    return {"path": str(destination), "shape": list(value.shape), "dtype": str(value.dtype),
        "bytes": destination.stat().st_size, "file_sha256": file_hash(destination),
        "values_sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def arrayload(np, record, clock, counter, member):
    clock.guard(); clock.count(counter, member)
    value = np.load(path(record["path"]), allow_pickle=False)
    require(list(value.shape) == record["shape"], "Array shape differs: " + member)
    if "values_sha256" in record:
        require(hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == record["values_sha256"], "Array values differ: " + member)
    if "file_sha256" in record:
        if counter == "physical_U_reads":
            clock.count("physical_U_byte_hash_reads", member + ":filehash")
        require(file_hash(record["path"]) == record["file_sha256"], "Array file differs: " + member)
        if counter == "physical_U_reads":
            clock.completed(member + ":filehash")
    require(str(value.dtype) == record["dtype"].replace("torch.", ""), "Array dtype differs: " + member)
    clock.completed(member); return value


def sync(torch):
    torch.cuda.synchronize()


def build_case(np, torch, geometry, helper, key, contract, output, clock):
    record = contract["records"][key]; folder = output / "input_cache" / key.replace("/", "_")
    manifest_file = folder / "manifest.json"
    config = {"contract_sha256": digest(contract), "key": key, "science": SCIENCE}
    if manifest_file.exists():
        manifest = read(manifest_file)
        require(manifest["config"] == config and manifest["status"] == "completed", "Own committed representation differs")
        return manifest
    folder.mkdir(parents=True, exist_ok=True); write(folder / "config.json", config)
    costfile = folder / "costs.json"; old = read(costfile) if costfile.exists() else {}
    prior = float(old.get("cumulative_nodal_case_seconds", 0.0))
    if old.get("active"):
        prior += max(0., time.time() - old["observed_unix_seconds"])
    began = time.perf_counter(); start_receipt = copy.deepcopy(clock.receipt)
    costs = {"cumulative_nodal_case_seconds": prior, "active": True, "observed_unix_seconds": time.time(),
        "attempts": int(old.get("attempts", 0)) + 1}
    write(costfile, costs); write(folder / "status.json", {"status": "building", "key": key})
    def guard():
        clock.guard()
        if prior + time.perf_counter() - began >= 30:
            raise ScientificNegative("Fixed cumulative nodal30s case budget exhausted: " + key)
    try:
        guard(); clock.count("input_cache_reads", key + ":physical_cache_archive")
        require(file_hash(record["input_cache"]["path"]) == record["input_cache"]["sha256"], "Physical cache bytes differ")
        members = {}
        with np.load(path(record["input_cache"]["path"]), allow_pickle=False) as archive:
            for name in ("vertices_mm", "cells10", "grid_node_ids", "ctetra_element_ids", "lame"):
                token = key + ":physical_cache:" + name; clock.count("input_member_reads", token)
                value = archive[name]; schema = record["input_members"][name]
                require(list(value.shape) == schema["shape"] and str(value.dtype) == schema["dtype"]
                    and hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == schema["sha256"], "Accepted physical member identity differs")
                members[name] = value; clock.completed(token)
        clock.completed(key + ":physical_cache_archive"); guard()
        U = arrayload(np, record["original_physical_U"], clock, "physical_U_reads", key + ":original_zero_U")
        valid = arrayload(np, record["same_valid"], clock, "valid_array_reads", key + ":old_valid")
        weights = arrayload(np, record["accepted_positive_fourQ_weights"], clock, "operator_weight_array_reads", key + ":accepted_Qweights")
        baseline_began = time.perf_counter()
        old_sigma = arrayload(np, record["old_mean_tensor"], clock, "old_tensor_array_reads", key + ":old_fourQ_sigma")
        old_sigma_record = arraysave(np, old_sigma, folder / "fourQ_sigma_MPa.npy")
        baseline_only_seconds = time.perf_counter() - baseline_began
        require(np.isfinite(U).all() and np.isfinite(old_sigma).all(), "Nonfinite accepted physical input")
        tensors = {name: torch.as_tensor(value, device="cuda", dtype=torch.int64 if name == "cells10" else torch.float64)
            for name, value in members.items() if name in ("vertices_mm", "cells10", "lame")}
        sync(torch); recovery_began = time.perf_counter(); clock.count("new_nodal_tensor_recoveries", key + ":native10_recovery")
        sigma_gpu, valid_gpu, diagnostics = helper.recover_tensor(torch, np, geometry,
            tensors["vertices_mm"], tensors["cells10"], torch.as_tensor(U, device="cuda", dtype=torch.float64),
            tensors["lame"], torch.as_tensor(weights, device="cuda", dtype=torch.float64),
            torch.as_tensor(valid, device="cuda", dtype=torch.bool), guard, chunk=8192)
        sync(torch); new_recovery_seconds = time.perf_counter() - recovery_began
        clock.completed(key + ":native10_recovery"); guard()
        nodal = sigma_gpu.cpu().numpy(); actual_valid = valid_gpu.cpu().numpy()
        require(np.array_equal(actual_valid, valid), "New/old original-node coverage differs")
        arrays = {"fourQ_cellmean": old_sigma_record,
            "element_node": arraysave(np, nodal, folder / "node_sigma_MPa.npy"),
            "valid": arraysave(np, valid, folder / "valid.npy")}
        copied_U = folder / "physical_u_mm.npy"; clock.count("original_U_byte_copies", key + ":shared_U_copy")
        shutil.copyfile(path(record["original_physical_U"]["path"]), copied_U)
        clock.count("physical_U_byte_hash_reads", key + ":shared_U_copy_hash")
        require(file_hash(copied_U) == record["original_physical_U"]["file_sha256"], "Shared original U byte copy differs")
        clock.completed(key + ":shared_U_copy_hash")
        clock.completed(key + ":shared_U_copy")
        arrays["original_U"] = {**record["original_physical_U"], "path": str(copied_U)}
        write(folder / "nodal_recovery_diagnostics.json", diagnostics); guard()
        elapsed = prior + time.perf_counter() - began
        costs.update(cumulative_nodal_case_seconds=elapsed, active=False, observed_unix_seconds=time.time(),
            new_native_tensor_recovery_seconds=new_recovery_seconds,
            baseline_fourQ_cache_IO_persist_seconds=baseline_only_seconds,
            actual_input_recovery_persistence_seconds=elapsed-baseline_only_seconds, new_recovery_is_cached_actual=True,
            fresh_joint_cold_measurement=False)
        write(costfile, costs); guard()
        child_receipt = {k: clock.receipt[k] - start_receipt[k] for k in FORBIDDEN + ACTUAL}
        manifest = {"config": config, "status": "completed", "key": key, "node_count": record["nodes"],
            "cell_count": record["cells"], "arrays": arrays, "costs": costs, "diagnostics": diagnostics,
            "read_receipt": child_receipt, "exact_original_U_byte_identity": True, "source_U": record["original_physical_U"],
            "old_node_GT_U_metric_recomputed": False}
        write(folder / "read_receipt.json", child_receipt)
    finally:
        costs.update(cumulative_nodal_case_seconds=prior + time.perf_counter() - began, active=False, observed_unix_seconds=time.time())
        write(costfile, costs); clock.persist("nodal_case_exit")
    # An immutable completed manifest is committed only after final cost journal.
    guard(); manifest["costs"] = dict(costs)
    manifest["costs"]["actual_input_recovery_persistence_seconds"] = costs["cumulative_nodal_case_seconds"]-baseline_only_seconds
    write(manifest_file, manifest); write(folder / "status.json", {"status": "completed", "key": key})
    clock.persist("nodal_case_committed"); guard(); return manifest


def label_once(np, key, contract, output, clock, barrier=False):
    if key in DEV:
        require(barrier and (output / "prediction_barrier.json").exists(), "DEV cached S read lacks both fixed endpoints barrier")
    folder = output / "label_cache"; meta = folder / (key.replace("/", "_") + ".json")
    claim = folder / (key.replace("/", "_") + "_read_started.json")
    scope = "TRAIN" if key in TRAIN else "DEV"
    counter = scope + "_cached_S_reads"
    if meta.exists():
        record = read(meta); require(record["key"] == key and record["contract_sha256"] == digest(contract), "Own S cache differs")
        clock.count("GT_reads"); return arrayload(np, record["array"], clock, counter, key + ":own_cached_S")
    require(not claim.exists(), "Uncommitted prior cached source-S read cannot be repeated")
    write(claim, {"key": key, "scope": scope, "contract_sha256": digest(contract), "source": contract["records"][key]["cached_original_signed_S"],
        "pre_DEV_barrier": key in DEV and not barrier, "started_unix_seconds": time.time()})
    clock.count("GT_reads"); clock.persist("cached_S_read_started")
    value = arrayload(np, contract["records"][key]["cached_original_signed_S"], clock, counter, key + ":original_cached_S")
    if not np.isfinite(value).all():
        raise ScientificNegative("Nonfinite original signed S cache")
    array = arraysave(np, value, folder / (key.replace("/", "_") + ".npy"))
    write(meta, {"key": key, "scope": scope, "contract_sha256": digest(contract), "array": array,
        "original_label_identity": contract["records"][key]["label_identity"], "original_signed_units": "MPa"})
    clock.persist("cached_S_committed"); return value


def fit_scalers(np, features, contract, output, clock):
    destination = output / "scalers.json"
    if destination.exists():
        saved = read(destination)
        require(saved["contract_sha256"] == digest(contract), "Own TRAIN scalers differ")
        return saved
    saved = {"contract_sha256": digest(contract), "signed_stress": contract["science"]["normalization"]["scaler_record"],
        "tensor6": {}, "output_scaler_fits": 0, "input_only_fit_keys": list(TRAIN)}
    for arm in ARMS:
        values = []
        for key in TRAIN:
            valid = arrayload(np, features[key]["arrays"]["valid"], clock, "valid_array_reads", key + ":fit_valid:" + arm)
            sigma = arrayload(np, features[key]["arrays"][arm], clock, "input_member_reads", key + ":fit_tensor:" + arm)
            values.append(sigma[valid])
        clock.guard(); clock.count("tensor_scaler_fits", arm + ":TRAIN_tensor_fit")
        tensor = np.concatenate(values, axis=0)
        require(len(tensor) > 0, "Empty valid TRAIN tensor representation")
        mean = tensor.mean(axis=0, dtype=np.float64); raw_std = tensor.std(axis=0, dtype=np.float64)
        std = np.maximum(raw_std, 1e-8)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ScientificNegative("Nonfinite TRAIN tensor normalizer")
        saved["tensor6"][arm] = {"mean_MPa": mean.tolist(), "std_MPa": std.tolist(),
            "raw_std_MPa": raw_std.tolist(), "std_floor_MPa": 1e-8, "valid_TRAIN_nodes": len(tensor),
            "fit_keys": list(TRAIN), "representation": arm}
        clock.completed(arm + ":TRAIN_tensor_fit"); del tensor, values
    write(destination, saved); clock.persist("both_TRAIN_scalers_committed"); clock.guard()
    return saved


def model_new(torch):
    return torch.nn.Sequential(torch.nn.Linear(7, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
        torch.nn.ReLU(), torch.nn.Linear(64, 1))


def parameter_hash(np, model):
    hashes = {name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
        for name, value in model.state_dict().items()}
    return digest(hashes)


def rng_save(np, torch):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all()}


def rng_restore(np, torch, saved):
    random.setstate(saved["python"]); np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch"].cpu()); torch.cuda.set_rng_state_all([item.cpu() for item in saved["cuda"]])


def checkpoint(torch, model, optimizer, state, config, folder, np, clock):
    clock.count("own_checkpoint_writes", folder.name + ":checkpoint_write")
    destination = folder / "last.pt"; temporary = folder / "last.pt.tmp"
    torch.save({"config": config, "state": state, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "rng": rng_save(np, torch)}, temporary)
    os.replace(temporary, destination)
    record = {"path": str(destination), "bytes": destination.stat().st_size, "file_sha256": file_hash(destination),
        "step": state["step"], "config_sha256": digest(config), "science": SCIENCE}
    write(folder / "checkpoint.json", record); clock.completed(folder.name + ":checkpoint_write")
    return record


def case_tensor(np, torch, key, arm, features, scalers, clock, labels=None):
    sigma = arrayload(np, features[key]["arrays"][arm], clock, "input_member_reads", key + ":neural_tensor:" + arm)
    valid = arrayload(np, features[key]["arrays"]["valid"], clock, "valid_array_reads", key + ":neural_valid:" + arm)
    norm = scalers["tensor6"][arm]
    normalized = (sigma - np.asarray(norm["mean_MPa"])) / np.asarray(norm["std_MPa"])
    normalized[~valid] = 0.
    inputs = torch.as_tensor(np.concatenate((normalized, valid[:, None]), axis=1), device="cuda", dtype=torch.float32)
    target = None
    if labels is not None:
        scale = scalers["signed_stress"]
        encoded = (np.arcsinh(labels / scale["scale_MPa"]) - scale["mean"]) / scale["std"]
        target = torch.as_tensor(encoded[:, None], device="cuda", dtype=torch.float32)
    sync(torch); clock.guard(); return inputs, target


def train_arm(np, torch, arm, common, initial_sha, features, scalers, labels, contract, output, clock):
    folder = output / arm; folder.mkdir(parents=True, exist_ok=True)
    config = {"contract_sha256": digest(contract), "arm": arm, "initial_parameters_sha256": initial_sha,
        "scaler_sha256": digest(scalers), "case_order_sha256": contract["science"]["neural"]["case_order_sha256"]}
    summary_file = folder / "summary.json"
    if summary_file.exists():
        saved = read(summary_file)
        require(saved["config"] == config, "Own training arm contract differs")
        if saved["completed120"]:
            return saved
    write(folder / "config.json", config)
    state_file = folder / "training_state.json"; journal = read(state_file) if state_file.exists() else {}
    prior = float(journal.get("cumulative_training_seconds", 0.0))
    if journal.get("active"):
        prior += max(0., time.time() - journal["observed_unix_seconds"])
    start = time.perf_counter(); last_checkpoint = start
    state = {"step": 0, "first_actual_gradient_check": None, "initial_parameters_sha256": initial_sha,
        "actual_attempted_updates": 0, "actual_optimizer_updates": 0, "case_exposure_counts": {key: 0 for key in TRAIN},
        "cumulative_training_seconds": prior, "history": []}
    model = copy.deepcopy(common).to("cuda")
    require(parameter_hash(np, model) == initial_sha, "Actual cloned head initialization differs")
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0., fused=True)
    own = folder / "last.pt"; record = None; began_receipt = copy.deepcopy(clock.receipt)
    if own.exists():
        index = read(folder / "checkpoint.json")
        require(file_hash(own) == index["file_sha256"], "Own training checkpoint file differs")
        clock.count("own_checkpoint_reads", arm + ":resume_checkpoint")
        saved = torch.load(own, map_location="cuda", weights_only=False)
        require(saved["config"] == config and 0 <= saved["state"]["step"] <= 120, "Own checkpoint science differs")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); state = saved["state"]
        rng_restore(np, torch, saved["rng"]); clock.completed(arm + ":resume_checkpoint")
        state["actual_attempted_updates"] = max(state["actual_attempted_updates"], journal.get("actual_attempted_updates", 0))
        state["actual_optimizer_updates"] = max(state["actual_optimizer_updates"], journal.get("actual_optimizer_updates", 0))
    else:
        require(not journal or journal.get("step", 0) == 0, "Uncommitted model progress cannot silently restart")
    # Costs survive checkpoint rollback; only committed model/RNG define step.
    state["cumulative_training_seconds"] = prior
    training_inputs = {}
    def elapsed():
        return prior + time.perf_counter() - start
    def status(phase, active=True):
        state.update(cumulative_training_seconds=elapsed(), active=active, observed_unix_seconds=time.time())
        compact = {name: state[name] for name in ("step", "cumulative_training_seconds", "active", "observed_unix_seconds",
            "actual_attempted_updates", "actual_optimizer_updates", "initial_parameters_sha256")}
        write(state_file, compact)
        write(folder / "status.json", {"status": phase, "arm": arm, "step": state["step"],
            "training_seconds": state["cumulative_training_seconds"], "complete120": state["step"] == 120})
    def guard():
        clock.guard()
        if elapsed() >= 60:
            raise ScientificNegative("Fixed cumulative training60s arm budget exhausted: " + arm)
    error = None
    try:
        guard(); status("loading_TRAIN")
        for key in TRAIN:
            training_inputs[key] = case_tensor(np, torch, key, arm, features, scalers, clock, labels[key])
            guard()
        model.train()
        while state["step"] < 120:
            guard(); key = contract["science"]["neural"]["case_order"][state["step"]]
            inputs, target = training_inputs[key]; nodes = len(inputs); loss_sum = 0.
            state["actual_attempted_updates"] += 1; clock.count("training_updates")
            optimizer.zero_grad(set_to_none=True)
            for begin in range(0, nodes, 32768):
                guard(); end = min(begin + 32768, nodes)
                loss = (model(inputs[begin:end]) - target[begin:end]).square().sum() / nodes
                if not bool(torch.isfinite(loss)):
                    raise ScientificNegative("Nonfinite full-CAD encoded signed-S loss")
                loss_sum += float(loss.detach()); loss.backward()
            finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            norm = math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in model.parameters())) if finite else None
            if not finite or norm is None or not math.isfinite(norm):
                raise ScientificNegative("Missing/nonfinite active head gradient")
            if state["first_actual_gradient_check"] is None:
                check = {"passed": norm > 0., "full_case_loss_finite": math.isfinite(loss_sum),
                    "all_active_parameter_gradients_present_finite": finite, "total_gradient_norm": norm,
                    "actual_first_TRAIN_key": key, "standalone_preflight": False}
                write(folder / "first_actual_gradient_check.json", check)
                if not check["passed"]:
                    raise ScientificNegative("First actual full-CAD head gradient is zero")
                state["first_actual_gradient_check"] = check
            optimizer.step(); sync(torch); clock.count("optimizer_updates"); state["actual_optimizer_updates"] += 1
            state["step"] += 1; state["case_exposure_counts"][key] += 1
            state["history"].append({"step": state["step"], "key": key, "encoded_S_case_mean_MSE": loss_sum})
            status("update_committed")
            if time.perf_counter() - last_checkpoint >= 60:
                record = checkpoint(torch, model, optimizer, state, config, folder, np, clock)
                last_checkpoint = time.perf_counter(); clock.persist("fullCAD_checkpoint")
        status("endpoint120", active=False)
        record = checkpoint(torch, model, optimizer, state, config, folder, np, clock)
        guard()
    except Exception as caught:
        error = caught; status("failed_or_budget", active=False)
        # Current finite model, Adam, RNG and committed cursor survive a repair.
        record = checkpoint(torch, model, optimizer, state, config, folder, np, clock)
    finally:
        sync(torch); status("completed" if state["step"] == 120 and error is None else "incomplete", active=False)
        clock.persist("arm_exit")
    child_receipt = {k: clock.receipt[k] - began_receipt[k] for k in FORBIDDEN + ACTUAL}
    result = {"config": config, "status": "completed" if state["step"] == 120 and error is None else "incomplete",
        "completed120": state["step"] == 120 and error is None, "endpoint_step": state["step"],
        "initial_parameters_sha256": initial_sha, "parameter_count": sum(p.numel() for p in model.parameters()),
        "case_exposure_counts": state["case_exposure_counts"], "case_order_sha256": config["case_order_sha256"],
        "first_actual_gradient_check": state["first_actual_gradient_check"], "checkpoint": record,
        "costs": {"cumulative_training_seconds": elapsed(), "training_oneoff_separate": True,
            "actual_attempted_updates": state["actual_attempted_updates"], "actual_optimizer_updates": state["actual_optimizer_updates"]},
        "history": state["history"], "read_receipt": child_receipt,
        "reason": str(error) if error is not None else None}
    write(summary_file, result); write(folder / "read_receipt.json", child_receipt)
    del model, optimizer, training_inputs; torch.cuda.empty_cache()
    if error is not None:
        raise error
    return result


def predict_arm(np, torch, arm, training, features, scalers, contract, output, clock):
    require(training["completed120"] and training["endpoint_step"] == 120, "No partial/unpaired DEV readout")
    folder = output / arm; model = model_new(torch).to("cuda")
    source = training["checkpoint"]; require(file_hash(source["path"]) == source["file_sha256"], "Fixed endpoint checkpoint bytes differ")
    clock.count("own_checkpoint_reads", arm + ":endpoint_checkpoint")
    began_model = time.perf_counter(); saved = torch.load(path(source["path"]), map_location="cuda", weights_only=False)
    require(saved["config"] == training["config"] and saved["state"]["step"] == 120, "Fixed endpoint checkpoint scope differs")
    model.load_state_dict(saved["model"]); model.eval(); sync(torch)
    model_load_seconds = time.perf_counter() - began_model; clock.completed(arm + ":endpoint_checkpoint")
    predictions = {}
    for key in DEV:
        destination = folder / "predictions" / key.replace("/", "_"); meta = destination / "prediction_manifest.json"
        if meta.exists():
            result = read(meta)
            require(result["config_sha256"] == digest(training["config"]) and result["endpoint_step"] == 120, "Own prediction endpoint differs")
            require(file_hash(result["original_U"]["path"]) == contract["records"][key]["original_physical_U"]["file_sha256"], "Committed U copy differs")
            predictions[key] = result; continue
        clock.guard(); began = time.perf_counter(); before = copy.deepcopy(clock.receipt)
        inputs, _ = case_tensor(np, torch, key, arm, features, scalers, clock)
        input_seconds = time.perf_counter() - began; begin_infer = time.perf_counter(); values = []
        with torch.no_grad():
            for begin in range(0, len(inputs), 32768):
                clock.guard(); encoded = model(inputs[begin:begin+32768]).double().flatten()
                scale = scalers["signed_stress"]
                physical = torch.sinh(encoded * scale["std"] + scale["mean"]) * scale["scale_MPa"]
                if not bool(torch.isfinite(physical).all()):
                    raise ScientificNegative("Nonfinite direct unbounded signed-S endpoint decoding")
                values.append(physical.cpu().numpy())
        prediction = arraysave(np, np.concatenate(values), destination / "signed_S_MPa.npy")
        clock.count("original_U_byte_copies", arm + ":" + key + ":U_output")
        target_U = destination / "physical_u_mm.npy"; shutil.copyfile(path(features[key]["arrays"]["original_U"]["path"]), target_U)
        clock.count("physical_U_byte_hash_reads", arm + ":" + key + ":U_output_hash")
        require(file_hash(target_U) == contract["records"][key]["original_physical_U"]["file_sha256"], "Prediction U bytes differ")
        clock.completed(arm + ":" + key + ":U_output_hash")
        clock.completed(arm + ":" + key + ":U_output"); sync(torch)
        inference_seconds = time.perf_counter() - begin_infer
        old = contract["records"][key]["inherited_cost_components"]
        recovery = old["sigma_recovery_seconds"] + old["validity_incidence_seconds"] if arm == ARMS[0] else features[key]["costs"]["actual_input_recovery_persistence_seconds"]
        components = {"original_GT_free_physical_cold_seconds": old["physical_GT_free_cold_seconds"],
            "representation_recovery_input_persist_seconds": recovery, "actual_head_input_H2D_seconds": input_seconds,
            "model_load_seconds": model_load_seconds, "fullN_inference_decode_Ucopy_prediction_persist_seconds": inference_seconds}
        costs = {"components": components, "campaign_accounted_cold_seconds": sum(components.values()),
            "cached_neural_request_seconds": time.perf_counter() - began + model_load_seconds,
            "fresh_joint_cold_measurement": False, "training_and_scaler_oneoff_excluded": True,
            "new_coarse_solver_cost_spliced": False}
        child_receipt = {k: clock.receipt[k] - before[k] for k in FORBIDDEN + ACTUAL}
        result = {"key": key, "arm": arm, "endpoint_step": 120, "config_sha256": digest(training["config"]),
            "allN_prediction_persisted": True, "node_count": len(inputs), "prediction": prediction,
            "original_U": {**contract["records"][key]["original_physical_U"], "path": str(target_U)},
            "original_U_exact_byte_identity": True, "new_GT_U_metric_computed": False, "costs": costs,
            "read_receipt": child_receipt, "signed_S_units": "MPa", "input_interface": "full_FE_tensor_proxy"}
        write(destination / "costs.json", costs); write(destination / "read_receipt.json", child_receipt)
        clock.guard(); write(meta, result); predictions[key] = result; clock.persist("allN_prediction_committed")
        del inputs, values
    del model; torch.cuda.empty_cache(); return predictions


def prediction_barrier(predictions, training, contract, output):
    require(all(training.get(arm, {}).get("completed120") is True for arm in ARMS), "Both fixed120 arms required")
    for arm in ARMS:
        require(set(predictions.get(arm, {})) == set(DEV), "Both fullN DEV requests required")
        for key in DEV:
            record = predictions[arm][key]
            require(record["allN_prediction_persisted"] is True and record["endpoint_step"] == 120
                and record["node_count"] == contract["records"][key]["nodes"]
                and record["original_U_exact_byte_identity"] is True, "AllN prediction barrier coverage differs")
            require(file_hash(record["prediction"]["path"]) == record["prediction"]["file_sha256"]
                and file_hash(record["original_U"]["path"]) == record["original_U"]["file_sha256"], "Committed barrier prediction bytes differ")
    proof = {"passed": True, "contract_sha256": digest(contract), "both_arms_complete120": True,
        "all_four_DEV_predictions_persisted": True, "original_U_exact_byte_copy": True,
        "predictions": {arm: {key: {name: predictions[arm][key][name] for name in ("prediction", "original_U")}
            for key in DEV} for arm in ARMS}, "DEV_S_reads_before_this_barrier": 0}
    write(output / "prediction_barrier.json", proof); return proof


def empty_gate(reason):
    return {"passed": False, "both_arms_complete120": False, "all_four_DEV_predictions_persisted": False,
        "full5_nodal_recovery_complete": False, "fixed_DEV_denominator": 2, "complete_service_accuracy_gate": False,
        "representation_gain_gate": False, "terminal_negative_reason": reason, "final_solver_selection": False}


def comparison_gate(training, predictions, features, validations, receipt, reason=None):
    gate = empty_gate(reason)
    gate["both_arms_complete120"] = all(training.get(a, {}).get("completed120") is True for a in ARMS)
    gate["all_four_DEV_predictions_persisted"] = all(set(predictions.get(a, {})) == set(DEV) for a in ARMS)
    gate["full5_nodal_recovery_complete"] = set(features) == set(CASES) and all(v.get("status") == "completed" for v in features.values())
    complete = gate["both_arms_complete120"] and gate["all_four_DEV_predictions_persisted"] and gate["full5_nodal_recovery_complete"]
    complete = complete and all(receipt[k] == 0 for k in FORBIDDEN)
    baseline = validations.get(ARMS[0], {}).get("per_case", {}); native = validations.get(ARMS[1], {}).get("per_case", {})
    available = set(baseline) == set(DEV) and set(native) == set(DEV)
    def number(v):
        return isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)
    keys = ("stress_relative_l2", "hotspot_relative_l2")
    finite = available and all(number(data[key][name]) for data in (baseline, native) for key in DEV for name in keys)
    service = complete and finite and all(native[key][name] < .25 for key in DEV for name in keys)
    gain = None; no_worse = False
    if finite:
        base_mean = sum(baseline[k]["stress_relative_l2"] for k in DEV)/2
        native_mean = sum(native[k]["stress_relative_l2"] for k in DEV)/2
        gain = (base_mean-native_mean)/base_mean if base_mean > 0 else None
        no_worse = all(native[key][name] <= 1.05*baseline[key][name] for key in DEV for name in keys)
    representation = complete and finite and gain is not None and gain >= .2 and no_worse
    gate.update(complete_service_accuracy_gate=bool(service), representation_gain_gate=bool(representation),
        passed=bool(service and representation and reason is None), mean_signed_S_relative_improvement=gain,
        each_DEV_S_and_hotspot_within_1p05_baseline=bool(no_worse), matched_baseline_has_same_fresh7_head=True,
        historical67_head_spliced_into_gate=False, metrics_available_DEV_cases=len(native),
        partial_representation_improvement_only=bool(representation and not service))
    if not gate["passed"] and gate["terminal_negative_reason"] is None:
        gate["terminal_negative_reason"] = "fixed_representation_or_complete_signed_service_gate_failed"
    return gate


def report(output, summary):
    gate = summary["completion_gate"]
    rows = ["| Representation | DEV mean S L2 | Worst hot1% L2 | Outcome |", "|---|---:|---:|---|"]
    for arm in ARMS:
        metrics = summary["arms"].get(arm, {}).get("validation", {})
        def percent(value):
            return "missing" if value is None else f"{100*value:.2f}%"
        endpoint = "complete120" if summary["arms"].get(arm, {}).get("completed120") else "incomplete"
        rows.append(f"| {arm} | {percent(metrics.get('stress_relative_l2_mean'))} | {percent(metrics.get('hotspot_relative_l2_worst'))} | {endpoint} |")
    (output / "metrics.md").write_text("\n".join(rows)+"\n", encoding="utf-8")
    lines = ["Frozen native-node tensor representation pair", f"Completion gate: {'PASS' if gate['passed'] else 'NEGATIVE'}",
        f"Reason: {gate['terminal_negative_reason'] or 'none'}", "Both arms: fresh tensor6+valid7 ->64->64->1, seed2026, fixed120 fullCAD updates.",
        "Inputs: same original zero-start physical U and full TET10 geometry; no new physical solve or U target read.",
        "Node representation: curved full10 physical gradients and one fixed incident positive-volume average.",
        "Historical67-dimensional readout is context only; it is not the matched baseline.",
        "Original signed S MPa supervised directly; no hand scalar/sign/exporter rule.",
        "DEV cached S reads occur after both fullN endpoint predictions and byte-identical original U copies persist.",
        "Recovered tensor is a constitutive proxy, not asserted original H3D recovery.",
        f"New cumulative stage wall: {summary['costs']['new_actual_wall_seconds']:.6f}s; inherited paid once: {summary['costs']['inherited_seconds_once']:.6f}s.",
        "Cold costs are composed accounting; cached latency and training are separate, not fresh combined cold.",
        "Geometry, budget or accuracy negatives close this frozen candidate without recipe rescue.",
        "TRAIN3/DEV2/ver representation screen; final GeoBC solver remains undecided."]
    (output / "BRIEF.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(plan_path=PLAN, output=None):
    started = time.perf_counter(); minimal = read(plan_path)
    output = path(output or minimal["output"]); output.mkdir(parents=True, exist_ok=True)
    inherited = sum(minimal["science"]["budgets"][name] for name in ("inherited_original_GT_free_physics_seconds_once",
        "inherited_accepted_fourQ_recovery_seconds_once", "inherited_accepted_validity_seconds_once"))
    clock = Clock(output, inherited, started); contract = None; config = None; helper = None
    features = {}; training = {}; predictions = {}; validations = {}; technical = None; negative = None; scalers = None
    try:
        contract, plan = request_contract(plan_path); config = {"contract": contract}
        config_file = output / "config.json"
        if config_file.exists():
            require(read(config_file) == config, "Existing frozen campaign differs")
        else:
            write(config_file, config)
        marker = read(plan["root_review"])
        require(marker.get("passed") is True and marker.get("source_sha256") == contract["source_sha256"][OWN]
            and marker.get("scientific_digest") == SCIENCE
            and marker.get("decision") == "proceed_frozen_nodal_tensor_representation_pair", "Actual root release marker differs or missing")
        # All scientific dependencies are lazy and reachable only after release.
        import numpy as np
        import torch
        from geobc import continuous_query_geometry as geometry
        from geobc import elastic_nodal_tensor as native_helper
        from geobc.elastic_stress_readout import signed_metrics, aggregate
        helper = native_helper
        require(helper.SCIENTIFIC_DIGEST == SCIENCE and helper.GEOMETRY_SOURCE_SHA256 == contract["source_sha256"]["geobc/continuous_query_geometry.py"],
            "Native helper frozen science/geometry differs")
        random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        clock.torch = torch; torch.cuda.reset_peak_memory_stats(); clock.guard()
        common = model_new(torch); initial_sha = parameter_hash(np, common)
        initial = {"contract_sha256": digest(contract), "seed": 2026, "initial_parameters_sha256": initial_sha,
            "parameter_count": sum(p.numel() for p in common.parameters()), "common_state_cloned_before_any_update": True,
            "external_or_old_endpoint_weights_loaded": False}
        initial_file = output / "initialization.json"
        if initial_file.exists():
            require(read(initial_file) == initial, "Shared fresh initialization differs on resume")
        else:
            write(initial_file, initial)
        write(output / "status.json", {"status": "preparing_nodal_inputs", "kind": KIND})
        for key in CASES:
            features[key] = build_case(np, torch, geometry, helper, key, contract, output, clock)
        labels = {key: label_once(np, key, contract, output, clock) for key in TRAIN}
        scalers = fit_scalers(np, features, contract, output, clock)
        for arm in ARMS:
            write(output / "status.json", {"status": "training", "arm": arm})
            training[arm] = train_arm(np, torch, arm, common, initial_sha, features, scalers, labels, contract, output, clock)
        for arm in ARMS:
            predictions[arm] = predict_arm(np, torch, arm, training[arm], features, scalers, contract, output, clock)
        prediction_barrier(predictions, training, contract, output); clock.persist("all_four_DEV_prediction_barrier")
        truth = {key: label_once(np, key, contract, output, clock, barrier=True) for key in DEV}
        for arm in ARMS:
            per_case = {}
            for key in DEV:
                clock.guard(); pred = arrayload(np, predictions[arm][key]["prediction"], clock,
                    "own_prediction_array_reads", arm + ":" + key + ":metric_prediction")
                per_case[key] = signed_metrics(np, truth[key], pred)
            validations[arm] = aggregate(per_case, DEV); training[arm]["validation"] = validations[arm]
        del truth, labels, common
    except Exception as error:
        if isinstance(error, ScientificNegative) or (helper is not None and isinstance(error, helper.ScientificNegative)):
            negative = str(error)
        elif "torch" in locals() and isinstance(error, torch.cuda.OutOfMemoryError):
            negative = "Fixed native CUDA backend allocation failure; no fallback: " + str(error)
        else:
            technical = {"type": type(error).__name__, "message": str(error)}
    finally:
        # Recover small immutable completed evidence only, never run a parent.
        if contract is not None:
            for key in CASES:
                meta = output / "input_cache" / key.replace("/", "_") / "manifest.json"
                if key not in features and meta.exists():
                    cached = read(meta)
                    if cached.get("config", {}).get("contract_sha256") == digest(contract) and cached.get("status") == "completed":
                        features[key] = cached
            for arm in ARMS:
                meta = output / arm / "summary.json"
                if arm not in training and meta.exists():
                    training[arm] = read(meta)
        if clock.elapsed() >= 300 and technical is None:
            negative = negative or "Fixed cumulative new300s stage budget exhausted including finalization"
        gate = comparison_gate(training, predictions, features, validations, clock.receipt, negative or (technical or {}).get("message"))
        costs = {"new_actual_wall_seconds": clock.elapsed(), "inherited_seconds_once": inherited,
            "campaign_composed_wall_seconds": clock.elapsed() + inherited, "maximum_new_actual_wall_seconds": 300,
            "maximum_campaign_composed_wall_seconds": inherited + 300, "failed_resume_wall_included": True,
            "new_actual_overshoot_seconds": max(0., clock.elapsed()-300), "fresh_joint_cold_measurement": False,
            "original_physics_not_new_coarse": True}
        summary = {"status": "failed" if technical else "completed", "kind": KIND, "config": config,
            "case_keys": list(CASES), "train_keys": list(TRAIN), "validation_keys": list(DEV),
            "cases": {key: {"group": "TRAIN" if key in TRAIN else "DEV", "status": "completed" if key in features else "missing",
                "node_count": minimal["records"][key]["nodes"], "input_manifest": features.get(key),
                "original_U_byte_identity_preserved": features.get(key, {}).get("exact_original_U_byte_identity", False)} for key in CASES},
            "arms": {arm: {**training.get(arm, {"status": "missing", "completed120": False}),
                "predictions": predictions.get(arm, {}), "validation": validations.get(arm, {})} for arm in ARMS},
            "completion_gate": gate, "costs": costs, "memory": clock.memory(), "read_receipt": copy.deepcopy(clock.receipt),
            "technical_error": technical, "old_GT_U_accuracy_recomputed": False, "source_signed_scalar_formula_assumed": False,
            "input_interface": "full_TET10_FE_geometry_original_physical_U", "scalers": scalers}
        write(output / "checkpoint_index.json", {"arms": {arm: training.get(arm, {}).get("checkpoint") for arm in ARMS},
            "weights_exported": False})
        report(output, summary); write(output / "summary.json", summary)
        if clock.elapsed() >= 300 and gate["passed"]:
            gate.update(passed=False, terminal_negative_reason="Cumulative new300s expired during report/persistence")
            summary["completion_gate"] = gate; write(output / "summary.json", summary); report(output, summary)
        clock.persist("failed" if technical else "completed", active=False)
        costs.update(new_actual_wall_seconds=clock.elapsed(), campaign_composed_wall_seconds=clock.elapsed()+inherited,
            new_actual_overshoot_seconds=max(0., clock.elapsed()-300))
        summary["costs"] = costs; write(output / "summary.json", summary)
        write(output / "status.json", {"status": summary["status"], "kind": KIND, "completion_gate": gate,
            "new_actual_wall_seconds": costs["new_actual_wall_seconds"]})
        clock.persist("process_finalized", active=False)
    if technical is not None:
        raise RuntimeError(technical["type"] + ": " + technical["message"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN); parser.add_argument("--output")
    args = parser.parse_args(); summary = run(args.plan, args.output)
    print(json.dumps({"status": summary["status"], "kind": KIND, "completion_gate": summary["completion_gate"], "costs": summary["costs"]}))


if __name__ == "__main__":
    main()
