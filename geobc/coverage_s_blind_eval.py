"""Independent fixed4800, U-known/S-field-blind readout of five CADs.

Import/request_contract use stdlib and metadata only. Actual scientific imports,
checkpoint/NPZ reads and GPU work require the genuine root release plus the
parent wait/diagnostic gates and the existing common GPU lock. No training API
is called. A failed or interrupted invocation is never automatically replayed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
OWN = "geobc/coverage_s_blind_eval.py"
PLAN = "notes/geobc_s_field_blind_plan.json"
KIND = "coverage_cad40_fixed4800_U_known_S_field_blind5"
CASES = ("489_12/ver", "616_205/ver", "416_282/ver", "247_50/ver", "142_454/ver")
METHODS = ("transolver", "transolver_pp", "ts3", "fno", "mlp")
ZERO = ("U_target_reads", "H5_nodal_variable_reads", "strain_target_reads",
        "hor_tor_dia_target_reads", "new_DEV20_target_reads", "old_DEV_target_reads",
        "old_final_target_reads", "training_updates", "optimizer_steps", "backward_calls",
        "optimizer_state_restores", "scaler_fits", "endpoint_selection", "seed_selection", "case_replacements")
ACTUAL = ("model_initializations", "checkpoint_read_attempts", "checkpoint_reads",
          "serialized_optimizer_payloads_loaded_on_CPU",
          "checkpoint_byte_hash_reads", "input_cache_byte_hash_reads", "input_cache_read_attempts",
          "input_cache_reads", "input_member_reads", "grid_builds", "prediction_attempts",
          "fullN_predictions", "prediction_array_writes", "prediction_array_hash_reads",
          "source_CSV_byte_hash_reads", "source_S_target_parse_attempts", "source_S_target_parses",
          "GT_read_attempts", "GT_reads", "stress_target_reads", "S_target_values_read",
          "label_cache_writes", "permutation_cache_reads",
          "prediction_array_reads", "physical_metric_cases")


class NotReady(RuntimeError):
    pass


class BudgetLimit(RuntimeError):
    pass


class ScientificNegative(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", value):
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    p = Path(value).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def read(filename):
    return json.loads(path(filename).read_text(encoding="utf-8-sig"))


def write(filename, value):
    filename = path(filename); filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(filename.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, filename)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def require(condition, message, error=ValueError):
    if not condition:
        raise error(message)


def sha(filename):
    h = hashlib.sha256()
    with path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(filename, with_sha=False):
    filename = path(filename); st = filename.stat()
    result = {"path": str(filename), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
    if with_sha:
        result["sha256"] = sha(filename)
    return result


def stat_matches(record):
    current = identity(record["path"])
    require(all(current[k] == record[k] for k in ("bytes", "mtime_ns")), "Stat identity changed: " + record["path"])


def request_contract(plan_path=PLAN):
    """No scientific import/header/array/checkpoint-value reads."""
    plan = read(plan_path); science = plan["scientific"]
    require(plan["kind"] == KIND and digest(science) == plan["scientific_digest"], "Own frozen science differs")
    require(science["case_keys"] == list(CASES) and science["methods"] == list(METHODS)
            and science["checkpoint_step"] == 4800 and science["seed"] == 2026,
            "Fixed five-family/five-case endpoint scope changed")
    require(science["prediction_budget_seconds"] == 600 and science["readout_budget_seconds"] == 120
            and science["stress_column"] == "ver_stress(MPa)" and science["fixed_denominator"] == 5,
            "Fixed budgets/label scope changed")
    parent = read(science["coverage_plan"])
    require(parent["scientific_digest"] == science["coverage_scientific_digest"]
            and digest(parent["scientific"]) == parent["scientific_digest"], "Coverage science changed")
    require(parent["source_sha256"] == science["original_twenty_source_pins"], "Original twenty pins changed")
    for group in ("train40_CADs", "additional_train_CADs", "train200_CADs", "DEV_CADs"):
        require(not (set(k.split("/")[0] for k in CASES) & set(parent["scientific"][group])), "Blind CAD overlaps " + group)
    require(parent["scientific"]["scaler"]["values"] == science["scaler"], "Frozen TRAIN40 scaler changed")
    historical = read(science["historical_U_summary"])
    require(historical["status"] == "completed" and historical["case_keys"] == list(CASES)
            and historical["read_receipt"]["stress_target_reads"] == 0
            and historical["read_receipt"]["H5_nodal_variable_reads"] == 0,
            "Historical controlled pipeline does not prove S-field sealing")
    source_pins = {**science["original_twenty_source_pins"], **science["additional_source_pins"]}
    sources = {name: sha(name) for name in source_pins}
    require(all(sources[k] == v for k, v in source_pins.items()), "Frozen dependency source differs")
    sources[OWN] = sha(OWN)
    inputs = {}
    for key in CASES:
        filename = science["input_manifests"][key]; record = read(filename); n = record["facts"]["node_count"]
        require(record["status"] == "completed" and record["feasibility_pass"] is True and record["key"] == key
                and type(n) is int and n > 0 and record["read_receipt"]["source_S_target_reads"] == 0
                and record["read_receipt"]["H5_nodal_variable_reads"] == 0, "Target-free input identity differs")
        arrays = record["arrays"]
        require(not (set(arrays) & {"target", "physical", "stress", "displacement", "u_true"}), "Input cache contains a target")
        for member, shape, dtype in (("xyz", [n, 3], "float32"), ("features", [n, 25], "float32"),
                                     ("cloud", [1, 1024, 3], "float32"), ("mask", [1, 1024], "bool"),
                                     ("h5_row_to_csv", [n], "int64")):
            require(arrays[member]["shape"] == shape and arrays[member]["dtype"] == dtype, "Original-node input layout differs: " + member)
        require(record["facts"]["coordinate_maps"]["one_to_one"] is True
                and record["facts"]["coordinate_maps"]["tolerance_mm"] == .0002,
                "Existing GRID/H5 alignment evidence differs")
        stat_matches(record["file_identity"]); stat_matches(record["source"]["Field"])
        inputs[key] = {"manifest": identity(filename, True), "cache": record["file_identity"],
                       "node_count": n, "arrays": arrays, "CSV_source": record["source"]["Field"]}
    routes = {}
    for row in science["routes"]:
        method = row["method"]; folder = path(row["training_output"])
        require(method in METHODS and row["arm_id"] == method + "_cad40_fresh4800", "Wrong arm")
        cfg_path = folder / "config.json"; summary_path = folder / "summary.json"; metadata_path = folder / "step4800.json"
        available = all(p.is_file() for p in (cfg_path, summary_path, metadata_path, folder / "step4800.pt"))
        route = {"method": method, "arm_id": row["arm_id"], "available": available,
                 "checkpoint_path": str(folder / "step4800.pt")}
        if available:
            config = read(cfg_path); summary = read(summary_path); endpoint = read(metadata_path)
            require(summary["status"] == "completed" and summary["completed4800"] is True
                    and config["method"] == method and config["train_CADs"] == 40
                    and config["scaler"] == science["scaler"] and endpoint["step"] == 4800
                    and digest(endpoint["config"]) == digest(config) == digest(summary["config"]),
                    "Fixed4800 training metadata differs")
            require(config["contract"]["scientific_digest"] == science["coverage_scientific_digest"]
                    and config["contract"]["input_phase"] == "train40_bootstrap"
                    and config["model"]["seed"] == 2026, "Wrong training source/scale/seed")
            route.update(config=identity(cfg_path, True), summary=identity(summary_path, True),
                         endpoint_metadata=identity(metadata_path, True), checkpoint=identity(folder / "step4800.pt"),
                         source_model=config["model"], parameter_counts=config["parameter_counts"],
                         training_cost_context=summary.get("costs"),
                         training_exposure_context=summary.get("case_exposure_counts"))
        routes[method] = route
    require(set(routes) == set(METHODS), "Route inventory differs")
    contract = {"schema": "S_field_blind_fixed4800_v1", "kind": KIND,
                "scientific_digest": plan["scientific_digest"], "source_sha256": sources,
                "historical_U_summary": identity(science["historical_U_summary"], True),
                "coverage_plan_science": science["coverage_scientific_digest"],
                "case_keys": list(CASES), "methods": list(METHODS), "scaler": science["scaler"],
                "inputs": inputs, "routes": routes, "all_fixed_endpoints_ready": all(v["available"] for v in routes.values())}
    return contract, plan


def parent_idle(plan, marker):
    require(os.name != "nt" and Path("/proc").is_dir(),
            "Actual readout must share the coordinator's WSL /proc and GPU lock namespace", NotReady)
    from .coverage_train_fit_diagnostic import waiting_coordinator
    conditions = plan["execution_gates"]
    try:
        proof = waiting_coordinator({"coordinator_status": conditions["coordinator_status"],
                                     "parent_plan": plan["scientific"]["coverage_plan"]})
    except RuntimeError as error:
        raise NotReady(str(error)) from error
    expected = marker.get("parent_wait_proof", {})
    require(all(proof.get(name) == expected.get(name) for name in ("pid", "start_ticks", "argv", "cwd"))
            and proof["pid"] == marker.get("parent_coordinator_pid") and proof["live_children"] == [],
            "Exact parent process identity/live-child proof changed", NotReady)
    age = time.time() - datetime.fromisoformat(proof["updated_at_utc"]).timestamp()
    require(0 <= age <= 90, "Parent wait heartbeat is stale", NotReady)
    summary = read(conditions["TRAIN_diagnostic_summary"]); clock = read(conditions["TRAIN_diagnostic_clock"])
    require(summary.get("status") == "completed" and clock.get("active") is False,
            "TRAIN diagnostic is not terminal with a closed clock", NotReady)
    require(identity(conditions["TRAIN_diagnostic_summary"], True) == marker["TRAIN_diagnostic_summary_identity"],
            "Diagnostic terminal proof changed", NotReady)
    return proof


def released(contract, plan):
    require(plan.get("scientific_rules_frozen") is True and plan.get("actual_execution_released") is True,
            "Own plan is source-only/unreleased", NotReady)
    marker = read(plan["root_review"])
    require(marker.get("passed") is True and marker.get("actual_execution_released") is True
            and marker.get("decision") == "release_independent_fixed4800_S_field_blind5"
            and marker.get("scientific_digest") == contract["scientific_digest"]
            and marker.get("source_sha256") == contract["source_sha256"]
            and marker.get("contract_sha256") == digest(contract), "Genuine root release/identities missing", NotReady)
    require(contract["all_fixed_endpoints_ready"], "All five committed4800 endpoints required", NotReady)
    for method in METHODS:
        require(re.fullmatch(r"[0-9a-f]{64}", marker.get("checkpoint_sha256", {}).get(method, "")),
                "Root must bind each checkpoint's already committed byte SHA", NotReady)
    parent_idle(plan, marker)
    return marker


@contextmanager
def lock(filename):
    filename = path(filename); filename.parent.mkdir(parents=True, exist_ok=True); handle = filename.open("a+")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(" "); handle.flush()
            handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        handle.close()


class Clock:
    def __init__(self, output, plan, marker):
        self.filename = output / "clock.json"; self.plan = plan; self.marker = marker
        self.costs = {"prediction_seconds": 0., "readout_seconds": 0.}; self.phase = None; self.began = None

    def start(self, phase):
        require(self.phase is None, "Overlapping clocks")
        self.phase = phase; self.began = time.monotonic(); self.persist()

    def elapsed(self):
        result = dict(self.costs)
        if self.phase:
            result[self.phase + "_seconds"] += time.monotonic() - self.began
        return result

    def persist(self):
        costs = self.elapsed()
        write(self.filename, {"kind": KIND, "active_phase": self.phase, "closed": self.phase is None,
              "updated_at_utc": utc(), "costs": costs, "limits_seconds": {"prediction": 600, "readout": 120},
              "overshoot_seconds": {name: max(0., costs[name + "_seconds"] - limit) for name, limit in (("prediction", 600), ("readout", 120))},
              "no_automatic_resume_or_reset": True})

    def guard(self):
        if self.phase:
            self.persist()
            limit = 600 if self.phase == "prediction" else 120
            if self.elapsed()[self.phase + "_seconds"] >= limit:
                raise BudgetLimit(self.phase + " cumulative wall limit")

    def close(self):
        if self.phase:
            self.costs[self.phase + "_seconds"] += time.monotonic() - self.began
            self.phase = None; self.began = None
        self.persist()


def array_schema(np, value):
    value = np.ascontiguousarray(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def save_array(np, filename, value, counts):
    filename = path(filename); filename.parent.mkdir(parents=True, exist_ok=True)
    require(not filename.exists(), "Owned array already exists; never replay/overwrite")
    temporary = filename.with_name(filename.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, filename); counts["prediction_array_writes"] += 1
    values = array_schema(np, value)
    result = {**identity(filename, True), "shape": values["shape"], "dtype": values["dtype"],
              "values_sha256": values["sha256"]}
    counts["prediction_array_hash_reads"] += 1
    return result


def load_case(np, torch, record, for_fno, counts, clock, checked):
    clock.guard(); stat_matches(record["file_identity"])
    if record["path"] not in checked:
        counts["input_cache_byte_hash_reads"] += 1
        require(sha(record["path"]) == record["file_identity"]["sha256"], "Prepared input file changed")
        checked.add(record["path"])
    result = {}; counts["input_cache_read_attempts"] += 1
    with np.load(path(record["path"]), allow_pickle=False) as stored:
        require(set(stored.files) == set(record["arrays"]), "Prepared member inventory differs")
        counts["input_cache_reads"] += 1
        for name in ("xyz", "features", "cloud", "mask"):
            value = stored[name]; counts["input_member_reads"] += 1
            require(array_schema(np, value) == record["arrays"][name], "Input value identity differs: " + name)
            require(np.isfinite(value).all(), "Input is nonfinite", ScientificNegative)
            result[name] = torch.from_numpy(value.copy()).to(device="cuda")
    if for_fno:
        from .solver_selection_data import _stencil, _splat
        indices, weights = _stencil(result["xyz"], 64)
        raster, active = _splat(result["features"], indices, weights, 64)
        axis = torch.linspace(0, 1, 64, device="cuda")
        xyz_grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"))
        result.update(query_indices=indices, query_weights=weights,
                      grid=torch.cat((active.float().reshape(1, 64, 64, 64), xyz_grid,
                                      raster.T.reshape(25, 64, 64, 64)), dim=0).unsqueeze(0))
        require(tuple(result["grid"].shape) == (1, 29, 64, 64, 64), "FNO29 input layout differs")
        counts["grid_builds"] += 1
    require("target" not in result and "physical" not in result, "Target leaked to model input")
    clock.guard(); return result


def prediction_path(output, method, key):
    return output / "predictions" / method / key.replace("/", "_")


def commit_barrier(output, contract, counts):
    require(counts["source_S_target_parse_attempts"] == 0 and counts["GT_read_attempts"] == 0,
            "Prediction barrier cannot be created after any S target parse attempt")
    inventory = {}; good = 0
    for method in METHODS:
        for key in CASES:
            filename = prediction_path(output, method, key) / "manifest.json"; value = read(filename)
            require(value["status"] == "completed" and value["terminal"] is True
                    and value["contract_sha256"] == digest(contract) and value["method"] == method and value["key"] == key,
                    "Prediction terminal metadata differs")
            if value["prediction_complete"]:
                n = contract["inputs"][key]["node_count"]
                require(value["finite_allN"] is True and value["all_original_nodes"] is True
                        and value["encoded"]["shape"] == value["physical"]["shape"] == [n, 4]
                        and value["encoded"]["dtype"] == "float32" and value["physical"]["dtype"] == "float64",
                        "Incomplete fullN field")
                stat_matches(value["encoded"]); stat_matches(value["physical"]); good += 1
            else:
                require(bool(value["reason"]), "Terminal failure lacks reason")
            inventory[method + ":" + key] = identity(filename, True)
    proof = {"status": "completed", "passed": good == 25, "required_entries": 25, "finite_fullN_entries": good,
             "fixed_methods": list(METHODS), "case_keys": list(CASES), "contract_sha256": digest(contract),
             "prediction_manifests": inventory, "source_S_parses_before_barrier": 0,
             "checkpoint_selection_performed": False, "new_training_performed": False}
    write(output / "prediction_barrier.json", proof); return proof


def shared_S(np, output, contract, proof, counts, clock):
    require(proof["passed"] and proof["finite_fullN_entries"] == 25, "No S read before all25 finite commits")
    import csv
    result = {}
    for key in CASES:
        clock.guard(); record = read(contract["inputs"][key]["manifest"]["path"])
        source = record["source"]["Field"]; stat_matches(source)
        counts["source_CSV_byte_hash_reads"] += 1
        require(sha(source["path"]) == source["sha256"], "Original source CSV byte identity changed")
        folder = output / "S_labels" / key.replace("/", "_"); claim = folder / "read_started.json"
        require(not claim.exists() and not (folder / "manifest.json").exists(), "S read already claimed; never replay")
        with path(source["path"]).open(encoding="utf-8-sig", newline="") as handle:
            columns = next(csv.reader(handle))
        require(len(columns) == len(set(columns)) and "ver_stress(MPa)" in columns, "Original ver signed-S column missing/ambiguous", ScientificNegative)
        clock.guard()
        counts["source_S_target_parse_attempts"] += 1; counts["GT_read_attempts"] += 1
        write(claim, {"key": key, "source": source, "column": "ver_stress(MPa)", "started_at_utc": utc(),
              "contract_sha256": digest(contract), "prediction_barrier_sha256": digest(proof), "completed": False})
        write(output / "read_receipt.json", counts)
        raw = np.loadtxt(path(source["path"]), delimiter=",", skiprows=1,
                         usecols=[columns.index("ver_stress(MPa)")], dtype=np.float64, ndmin=2)
        counts["source_S_target_parses"] += 1; counts["GT_reads"] += 1; counts["stress_target_reads"] += 1
        counts["S_target_values_read"] += raw.size
        write(output / "read_receipt.json", counts)
        n = contract["inputs"][key]["node_count"]
        require(raw.shape == (n, 1) and np.isfinite(raw).all(), "Signed S shape/nonfinite differs", ScientificNegative)
        stat_matches(record["file_identity"]); counts["permutation_cache_reads"] += 1
        with np.load(path(record["path"]), allow_pickle=False) as stored:
            permutation = stored["h5_row_to_csv"]; counts["input_member_reads"] += 1
        require(array_schema(np, permutation) == record["arrays"]["h5_row_to_csv"], "Frozen GRID/H5 permutation changed")
        require(np.array_equal(np.sort(permutation), np.arange(n)), "Alignment is not a complete original-row permutation")
        aligned = np.ascontiguousarray(raw[permutation, 0], dtype=np.float64)
        filename = folder / "signed_S_MPa.npy"; folder.mkdir(parents=True, exist_ok=True)
        temporary = filename.with_name(filename.name + ".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, aligned, allow_pickle=False)
        os.replace(temporary, filename); counts["label_cache_writes"] += 1
        values = array_schema(np, aligned)
        item = {"status": "completed", "key": key, "source_column": "ver_stress(MPa)", "units": "MPa",
                "original_sign_preserved": True, "array": {**identity(filename, True),
                "shape": values["shape"], "dtype": values["dtype"], "values_sha256": values["sha256"]},
                "contract_sha256": digest(contract), "prediction_barrier_sha256": digest(proof)}
        write(folder / "manifest.json", item)
        write(claim, {**read(claim), "completed": True, "committed_at_utc": utc()})
        result[key] = aligned; del raw, permutation; clock.guard()
    require(counts["source_S_target_parses"] == counts["source_S_target_parse_attempts"] == 5, "Shared once-only S read count differs")
    return result


def metrics(np, truth, predicted):
    require(truth.ndim == 1 and predicted.shape == truth.shape and len(truth) > 0
            and np.isfinite(truth).all() and np.isfinite(predicted).all(), "Physical signed-S metric schema/nonfinite", ScientificNegative)
    error = predicted - truth; denom = float(np.square(truth).sum()); count = max(1, math.ceil(.01 * len(truth)))
    hot = np.argpartition(np.abs(truth), len(truth) - count)[-count:]
    hotdenom = float(np.square(truth[hot]).sum())
    return {"num_nodes": len(truth), "stress_relative_l2": math.sqrt(float(np.square(error).sum()) / denom) if denom > 0 else None,
            "stress_hotspot_relative_l2": math.sqrt(float(np.square(error[hot]).sum()) / hotdenom) if hotdenom > 0 else None,
            "stress_RMSE_MPa": math.sqrt(float(np.square(error).mean())), "stress_MAE_MPa": float(np.abs(error).mean()),
            "stress_max_abs_error_MPa": float(np.abs(error).max()), "stress_hotspot_nodes": count,
            "stress_zero_reference": denom == 0, "hotspot_zero_reference": hotdenom == 0,
            "hotspot_definition": "ceil1% largest abs(original signed S), identical existing argpartition convention"}


def aggregate(per):
    value = {"fixed_denominator": 5, "per_case": per, "metrics_available_cases": sum(x is not None for x in per.values())}
    for name in ("stress_relative_l2", "stress_hotspot_relative_l2", "stress_RMSE_MPa", "stress_MAE_MPa", "stress_max_abs_error_MPa"):
        rows = [item.get(name) if item else None for item in per.values()]
        complete = len(rows) == 5 and all(type(x) in (float, int) and math.isfinite(x) and x >= 0 for x in rows)
        value["case_mean_" + name] = sum(rows) / 5 if complete else None
        value["worst_case_" + name] = max(rows) if complete else None
    return value


def run(plan_path=PLAN, output=None):
    contract, plan = request_contract(plan_path); marker = released(contract, plan)
    output = path(output or plan["output"])
    require(output == path(plan["output"]), "Unfrozen output override", NotReady)
    with lock(plan["execution_gates"]["common_GPU_lock"]), lock(str(output) + ".lock"):
        parent_idle(plan, marker)
        if (output / "summary.json").exists():
            previous = read(output / "summary.json")
            require(previous["config"]["contract"] == contract, "Completed readout contract changed", NotReady)
            return previous
        require(not (output / "config.json").exists() and not (output / "clock.json").exists(),
                "Interrupted invocation has no automatic replay/reset; root must preserve/recover its real ledger", NotReady)
        output.mkdir(parents=True, exist_ok=True)
        from .coverage_train_fit_diagnostic import live_process
        process = live_process(os.getpid())
        require(process is not None, "Cannot persist actual readout process identity", NotReady)
        counts = {name: 0 for name in ZERO + ACTUAL}
        counts.update(serialized_optimizer_payload_loaded_on_CPU=False, optimizer_state_used=False)
        config = {"contract": contract, "plan_science": plan["scientific"], "readout_process": process}
        write(output / "config.json", config); write(output / "read_receipt.json", counts)
        write(output / "status.json", {"status": "running", "terminal": False, "kind": KIND, "readout_process": process})
        clock = Clock(output, plan, marker); clock.start("prediction")
        failure = None; proof = None; aggregates = {}; costs = {"per_method": {}}; memory = None
        np = torch = None; cleanup = None
        try:
            import numpy as np
            import torch
            import random
            from .benchmark import FieldScaler
            from .solver_selection_models import build_selection_model, close_selection_process_group
            from .solver_selection_train import _predict_case
            from .solver_selection_data import query_grid
            cleanup = close_selection_process_group
            random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
            torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True; torch.cuda.init(); torch.cuda.reset_peak_memory_stats()
            scaler = FieldScaler(**contract["scaler"]); checked = set()
            for method in METHODS:
                clock.guard(); route = contract["routes"][method]; model_started = time.monotonic()
                model = build_selection_model(method, "cuda", 2026); counts["model_initializations"] += 1
                try:
                    require(digest(model.source_config) == digest(route["source_model"]), "Live model source/preset differs")
                    stat_matches(route["checkpoint"]); counts["checkpoint_byte_hash_reads"] += 1
                    actual_sha = sha(route["checkpoint_path"])
                    require(actual_sha == marker["checkpoint_sha256"][method], "Fixed checkpoint byte SHA differs")
                    clock.guard(); counts["checkpoint_read_attempts"] += 1
                    saved = torch.load(route["checkpoint_path"], map_location="cpu", weights_only=False); counts["checkpoint_reads"] += 1
                    if "optimizer" in saved:
                        counts["serialized_optimizer_payloads_loaded_on_CPU"] += 1
                        counts["serialized_optimizer_payload_loaded_on_CPU"] = True
                    train_config = read(route["config"]["path"])
                    require(saved["step"] == saved["next_case_position"] == 4800
                            and digest(saved["config"]) == digest(train_config)
                            and digest(saved["config"]["model"]) == digest(model.source_config),
                            "Checkpoint/config/source metadata differs")
                    model.load_state_dict(saved["model"], strict=True); del saved; model.eval()
                    method_cost = {"model_checkpoint_seconds": time.monotonic() - model_started,
                                   "input_and_grid_seconds": 0., "forward_decode_seconds": 0., "persistence_seconds": 0.}
                    costs["per_method"][method] = method_cost
                    for key in CASES:
                        clock.guard(); record = read(contract["inputs"][key]["manifest"]["path"])
                        t = time.monotonic(); case = load_case(np, torch, record, method == "fno", counts, clock, checked)
                        method_cost["input_and_grid_seconds"] += time.monotonic() - t
                        try:
                            counts["prediction_attempts"] += 1; t = time.monotonic()
                            with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
                                seed = 2026 + int(hashlib.sha256(f"DEV:4800:{key}".encode()).hexdigest()[:8], 16)
                                torch.manual_seed(seed)
                                encoded = _predict_case(torch, model, method, case, query_grid)
                            torch.cuda.synchronize(); n = contract["inputs"][key]["node_count"]
                            require(encoded.shape == (n, 4) and np.isfinite(encoded).all(), "Incomplete/nonfinite encoded fullN", ScientificNegative)
                            physical = np.asarray(scaler.decode(encoded), dtype=np.float64)
                            require(physical.shape == (n, 4) and np.isfinite(physical).all(), "Nonfinite physical fullN", ScientificNegative)
                            method_cost["forward_decode_seconds"] += time.monotonic() - t; clock.guard()
                            t = time.monotonic(); folder = prediction_path(output, method, key)
                            e = save_array(np, folder / "encoded4.npy", np.asarray(encoded, dtype=np.float32), counts)
                            p = save_array(np, folder / "physical4.npy", physical, counts)
                            value = {"status": "completed", "terminal": True, "method": method, "key": key,
                                     "contract_sha256": digest(contract), "prediction_complete": True, "reason": None,
                                     "finite_allN": True, "all_original_nodes": True, "node_count": n,
                                     "encoded": e, "physical": p, "units": ["mm", "mm", "mm", "MPa"],
                                     "checkpoint": {**route["checkpoint"], "sha256": actual_sha, "step": 4800},
                                     "isolated_eval_seed": seed, "target_reads": 0}
                            write(folder / "manifest.json", value); counts["fullN_predictions"] += 1
                            method_cost["persistence_seconds"] += time.monotonic() - t
                            write(output / "read_receipt.json", counts); clock.guard()
                            del encoded, physical
                        finally:
                            del case
                finally:
                    del model; cleanup(); torch.cuda.empty_cache()
            proof = commit_barrier(output, contract, counts); clock.guard(); clock.close()
            require(clock.costs["prediction_seconds"] <= 600, "Prediction commit exceeded600s", BudgetLimit)
            clock.start("readout"); labels = shared_S(np, output, contract, proof, counts, clock)
            for method in METHODS:
                per = {key: None for key in CASES}
                aggregates[method] = aggregate(per)
                for key in CASES:
                    clock.guard(); info = read(prediction_path(output, method, key) / "manifest.json")["physical"]
                    stat_matches(info); counts["prediction_array_hash_reads"] += 1
                    require(sha(info["path"]) == info["sha256"], "Committed prediction file bytes changed")
                    predicted = np.load(path(info["path"]), mmap_mode="r", allow_pickle=False)
                    counts["prediction_array_reads"] += 1
                    require(array_schema(np, predicted) == {"shape": info["shape"], "dtype": info["dtype"],
                            "sha256": info["values_sha256"]},
                            "Committed physical array values changed")
                    per[key] = metrics(np, labels[key], np.asarray(predicted[:, 3], dtype=np.float64))
                    counts["physical_metric_cases"] += 1; del predicted
                    aggregates[method] = aggregate(per)
                    write(output / "metrics_progress.json", {"arms": aggregates}); clock.guard()
            memory = {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                      "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
        except BaseException as error:
            failure = {"type": type(error).__name__, "reason": str(error), "phase": clock.phase,
                       "technical_failure": not isinstance(error, (BudgetLimit, ScientificNegative))}
            write(output / "failure.json", failure)
        finally:
            if cleanup is not None:
                try:
                    cleanup()
                except Exception as error:
                    failure = failure or {"type": type(error).__name__, "reason": str(error), "phase": clock.phase, "technical_failure": True}
            for method in METHODS:
                for key in CASES:
                    filename = prediction_path(output, method, key) / "manifest.json"
                    if not filename.exists():
                        write(filename, {"status": "completed", "terminal": True, "method": method, "key": key,
                              "contract_sha256": digest(contract), "prediction_complete": False, "finite_allN": False,
                              "reason": failure["reason"] if failure else "uncompleted fixed request"})
                aggregates.setdefault(method, aggregate({key: None for key in CASES}))
            if proof is None:
                try:
                    proof = commit_barrier(output, contract, counts)
                except Exception as error:
                    proof = {"status": "completed", "passed": False, "required_entries": 25, "reason": str(error)}
                    write(output / "prediction_barrier.json", proof)
            require(all(counts[k] == 0 for k in ZERO), "Forbidden read/work receipt became nonzero")
            closed = clock.elapsed()
            complete = (failure is None and proof["passed"] and counts["source_S_target_parses"] == 5
                        and counts["physical_metric_cases"] == 25 and closed["prediction_seconds"] <= 600
                        and closed["readout_seconds"] <= 120
                        and all(aggregates[m]["case_mean_stress_relative_l2"] is not None for m in METHODS))
            uncommitted = []
            for key in CASES:
                claim = output / "S_labels" / key.replace("/", "_") / "read_started.json"
                if claim.exists() and read(claim).get("completed") is not True:
                    uncommitted.append(key)
            result = {"status": "failed" if failure and failure["technical_failure"] else "completed", "terminal": True,
                      "kind": KIND, "config": config, "case_keys": list(CASES), "fixed_denominator": 5,
                      "readout_process": process,
                      "methods": list(METHODS), "arms": {m: {"validation": aggregates[m], "checkpoint_step": 4800,
                      "parameter_counts": contract["routes"][m].get("parameter_counts"), "costs": costs["per_method"].get(m),
                      "training_cost_context": contract["routes"][m].get("training_cost_context"),
                      "training_exposure_context": contract["routes"][m].get("training_exposure_context")} for m in METHODS},
                      "completion_gate": {"passed": complete, "all25_finite_prediction_barrier": proof["passed"],
                      "shared_S_reads_complete": counts["source_S_target_parses"] == 5, "fixed_5_denominator": True,
                      "parent_20DEV_or_10arm_gate_claimed": False, "final_solver_selection": False},
                      "read_receipt": counts, "uncommitted_S_read_claims": uncommitted,
                      "uncommitted_read_scope": "A claimed parse that did not return may have consumed unknown partial S values; never infer zero or replay",
                      "costs": {**costs, **closed}, "memory": memory, "failure": failure,
                      "completed_at_utc": utc(), "interpretation": ["U-known/S-field-blind pilot; S values were not semantically materialized in prior controlled pipeline",
                      "CSV bytes/header/coordinates/U were previously read; this is not untouched-file blindness",
                      "Fixed4800 single-seed current presets, unequal capacity/cost; no family ceiling or engineering qualification",
                      "No substitute for frozen20DEV or ten-arm coverage comparison; no U GT/metric readout or endpoint choice",
                      "Missing/nonfinite/incomplete outcomes remain null with all five CADs; no automatic replay"]}
            write(output / "summary.json", result); write(output / "metrics.json", result["arms"])
            write(output / "read_receipt.json", counts); write(output / "status.json", {
                "status": result["status"], "terminal": True, "kind": KIND, "readout_process": process})
            # Phase charges include scientific imports, input/weight IO, prediction/GT/metrics,
            # cleanup and the first full terminal publication. The final closed-clock/summary
            # identity amendment is separately measured metadata publication, not hidden zero.
            clock.close(); closed = clock.elapsed(); publication_started = time.monotonic()
            result["costs"].update(closed)
            result["costs"]["phase_overshoot_seconds"] = {
                "prediction": max(0., closed["prediction_seconds"] - 600),
                "readout": max(0., closed["readout_seconds"] - 120)}
            result["completion_gate"]["passed"] = bool(complete and closed["prediction_seconds"] <= 600
                                                       and closed["readout_seconds"] <= 120)
            result["costs"]["closed_clock_definition"] = "Phases include first complete terminal publication; subsequent final identity/closed-cost amendment is explicit metadata-only tail"
            write(output / "summary.json", result)
            publication_seconds = time.monotonic() - publication_started
            write(output / "publication_receipt.json", {"after_phase_close_publication_seconds": publication_seconds,
                  "final_clock_closed": True, "phase_costs": closed,
                  "measurement_endpoint": "After closed-cost summary amendment; this receipt's own write excluded",
                  "external_full_invocation_wall_must_be_reported_separately": True})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN); parser.add_argument("--output"); parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    if not args.start:
        print(json.dumps({"status": "source_only", "kind": KIND, "scientific_execution": False,
                          "note": "--start still requires genuine root release, terminal diagnostic, idle parent and GPU lock"}))
        return
    try:
        value = run(args.plan, args.output)
        print(json.dumps({"status": value["status"], "kind": KIND, "completion_gate": value["completion_gate"]}, ensure_ascii=False))
    except NotReady as error:
        print(json.dumps({"status": "not_started", "reason": str(error), "scientific_execution": False}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
