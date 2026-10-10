"""Target-free inputs for the frozen availability-conditional CAD20 pilot.

Import and ``prepare(start=False)`` are standard-library/source-only.  The
metadata contract reads small JSON/source hashes and file stats, never raw
headers, array bodies, or checkpoint bodies.  Actual preparation is one shot
under a genuine root release, exact predecessor-exit proof, external whole-
stage wall guard, and the common nonblocking GPU lock.  The original input
math is reused without calling any original protocol wrapper.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
OWN = "geobc/available_cad_pilot_data.py"
PLAN = "notes/geobc_available_cad_pilot_plan.json"
KIND = "GeoBC_available_original_TRAIN_CAD20_candidate9_source_design"
SCIENCE = "c7c994a44d462b94426ecfa034f2d535bbf7751331c1e16ab8925485e5a67770"
SCHEMA = "GeoBC_available20_target_free_inputs_v1"
LOADS = ["ver", "hor", "tor"]
COORDS = ["nodeID", "coord_x(mm)", "coord_y(mm)", "coord_z(mm)"]
TARGET_COLUMNS = [load + suffix for load in LOADS for suffix in
                  ("_x_disp(mm)", "_y_disp(mm)", "_z_disp(mm)", "_stress(MPa)")]
MEMBERS = ("xyz_path", "cloud_path", "mask_path", "features_path", "grid_path",
           "query_indices_path", "query_weights_path")
GUARD_SOURCE = "tools/run_geobc_available_pilot_guard.py"
GUARD_KIND = "GeoBC_available20_one_shot_guard"
PREDECESSOR_PLANS = {"PP": "notes/geobc_transolver_pp_regime_plan.json",
                     "FNO": "notes/geobc_fno_local_condition_pair_plan.json"}
INPUT_COUNTERS = ("raw_h5_input_reads", "raw_deck_input_reads", "raw_deck_header_reads",
                  "CSV_coordinate_reads", "input_member_reads", "CSV_header_only_reads")
ZERO_COUNTERS = ("TRAIN_source_target_parse_attempts", "TRAIN_source_target_parses",
                 "DEV_source_target_parse_attempts", "DEV_source_target_parses",
                 "TRAIN_U_target_case_reads", "TRAIN_stress_target_case_reads",
                 "DEV_U_target_case_reads", "DEV_stress_target_case_reads",
                 "owned_DEV_target_cache_reads", "H5_nodal_variable_reads",
                 "source_strain_reads", "dia_target_reads", "old_DEV_target_reads",
                 "old_final_reads", "sealed_new_test_reads", "scaler_fits",
                 "numeric_GT_parse_attempts", "numeric_GT_parses")


class NotReady(RuntimeError):
    pass


class Budget(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(value, message, error=ValueError):
    if not bool(value):
        raise error(message)


def path(value):
    text = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", text):
        text = text[5].upper() + ":/" + text[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", text):
        text = "/mnt/" + text[0].lower() + text[2:]
    result = Path(text).expanduser()
    return (result if result.is_absolute() else ROOT / result).resolve()


def relative(value):
    return path(value).relative_to(ROOT.resolve()).as_posix()


def shared_gpu_lock(plan):
    """Resolve the frozen prose-valued lock declaration to its sole file path."""
    declared = plan["scientific"]["execution_device_and_lock"]["single_shared_GPU_lock"]
    filename = "runs/geobc_exploration/.coordinator.lock"
    require(declared == filename + " for actual preparation/grid and full independent readout; nonblocking",
            "Frozen common GPU lock declaration changed")
    require(relative(filename) == filename, "Common GPU lock resolves outside its declared path")
    return filename


def read(value):
    return json.loads(path(value).read_text(encoding="utf-8-sig"))


def write(value, record):
    filename = path(value)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(filename.name + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, filename)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(value):
    result = hashlib.sha256()
    with path(value).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def identity(value, hashed=False):
    filename = path(value)
    stat = filename.stat()
    result = {"path": relative(filename), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if hashed:
        result["sha256"] = sha(filename)
    return result


def stat_matches(record):
    actual = identity(record["path"])
    require(all(actual[name] == record[name] for name in ("bytes", "mtime_ns")),
            "File stat differs: " + record["path"])


def keys(cads):
    return [cad + "/" + load for cad in cads for load in LOADS]


def request_contract(plan_path=PLAN):
    """Small JSON, source SHA and stats only; no raw header/array/GT reads."""
    plan = read(plan_path)
    science = plan["scientific"]
    require(plan.get("kind") == KIND and plan.get("scientific_rules_frozen") is True
            and digest(science) == plan["scientific_digest"] == SCIENCE,
            "Available CAD pilot science is not the frozen design")
    parent_spec = science["immutable_parent"]
    parent_file = parent_spec["plan"]
    parent = read(parent_file)
    require(sha(parent_file) == parent_spec["plan_sha256_at_design"]
            and digest(parent["scientific"]) == parent["scientific_digest"]
            == parent_spec["scientific_digest"]
            and parent["source_sha256"] == parent_spec["original_twenty_source_pins"]
            == plan["parent_source_sha256"], "Original frozen parent changed")
    selection = science["CAD_selection"]
    selected = selection["selected_CADs"]
    eligible = selection["eligible_CADs"]
    require(len(eligible) == len(set(eligible)) == 102 and eligible == sorted(eligible)
            and selected == eligible[:20] and len(set(selected)) == 20
            and selection["loads"] == LOADS and selection["case_keys"] == keys(selected)
            and selection["fixed_CAD_denominator"] == 20
            and selection["fixed_case_denominator"] == 60,
            "Fixed lexicographic CAD20/case60 inventory differs")
    pscience = parent["scientific"]
    require(selection["excluded_old_TRAIN40"] == pscience["train40_CADs"]
            and selection["excluded_original_DEV20"] == pscience["DEV_CADs"]
            and set(eligible) <= set(pscience["additional_train_CADs"])
            and not set(selected).intersection(pscience["train40_CADs"] + pscience["DEV_CADs"]),
            "Original TRAIN/DEV exclusion differs")
    require(science["budgets"] == {"entire_preparation_wall_seconds_max": 3600,
            "all9x60_readout_wall_seconds_max": 3600, "whole_campaign_wall_seconds_max": 7200,
            "peak_allocated_and_reserved_GPU_GiB_max": 24,
            "all_initialization_IO_transfer_persistence_failure_costs_included": True,
            "no_automatic_resume_or_budget_reset": True}, "Frozen budgets differ")
    policy = science["input_policy"]
    recipe = plan["recipe_reference"]
    require(recipe == pscience["recipe"] and recipe["device"] == "cuda"
            and recipe["cloud_points"] == 1024 and recipe["cloud_seed"] == 2026
            and recipe["grid_resolution"] == 64 and recipe["channels"] == [0, 1, 2, 3]
            and policy["features"] == 25 and policy["GRID_to_H5_bijective_tolerance_mm"] == .0002
            and policy["scaler"] == pscience["scaler"]["values"]
            and policy["new_scaler_fit"] is False and policy["targets_in_input_cache"] is False,
            "Original input mathematics/scaler changed")
    scaler_file = pscience["scaler"]["source_manifest"]
    scaler_manifest = read(scaler_file)
    require(sha(scaler_file) == pscience["scaler"]["source_manifest_sha256"]
            and scaler_manifest.get("status") == "completed"
            and scaler_manifest["scaler"] == policy["scaler"], "Frozen TRAIN40 scaler metadata changed")
    sources = {}
    expected = dict(plan["parent_source_sha256"])
    for name, value in plan["additional_source_sha256"].items():
        require(name not in expected or expected[name] == value, "Conflicting original source pins")
        expected[name] = value
    for name, value in plan.get("new_source_sha256", {}).items():
        require(name not in expected or expected[name] == value, "Conflicting new source pins")
        expected[name] = value
    for name, value in expected.items():
        sources[name] = sha(name)
        require(sources[name] == value, "Pinned source differs: " + name)
    sources[OWN] = sha(OWN)
    summary_file = parent_spec["raw_summary"]
    summary = read(summary_file)
    require(sha(summary_file) == parent_spec["raw_summary_sha256_at_design"]
            and summary.get("status") == parent_spec["raw_status"] == "budget_exhausted"
            and summary.get("terminal_reason") == parent_spec["raw_terminal_reason"],
            "Original acquisition-negative summary changed")
    received = summary.get("files", [])
    if isinstance(received, dict):
        received = list(received.values())
    acquisition = parent["acquisition"]
    require(len(acquisition["files"]) == 540 and len(received) == 540
            and sum(item.get("status") == "completed" for item in received)
            == parent_spec["raw_completed_files"] == 317,
            "Original540/317 acquisition inventory changed")
    by_relative = {}
    for item in received:
        name = item["relative_path"]
        require(name not in by_relative, "Duplicate original acquisition receipt")
        by_relative[name] = item
    inventory = {(item["cad_id"], item["component"]): item for item in acquisition["files"]}
    require(len(inventory) == 540, "Original acquisition entries repeat")
    complete = sorted(cad for cad in pscience["additional_train_CADs"] if all(
        by_relative[inventory[cad, component]["relative_path"]].get("status") == "completed"
        for component in ("FieldMesh", "Field", "SolverDeck")))
    require(complete == eligible, "Availability-conditioned eligible102 inventory differs")
    index = science["selected_source_receipt_index"]
    require([item["cad_id"] for item in index] == selected, "Frozen selected receipt order differs")
    raw_sources = {}
    for selected_receipt in index:
        cad = selected_receipt["cad_id"]
        require(set(selected_receipt["components"]) == {"FieldMesh", "Field", "SolverDeck"},
                "Selected canonical source triplet differs")
        raw_sources[cad] = {}
        for component, frozen in selected_receipt["components"].items():
            official = inventory[cad, component]
            name = official["relative_path"]
            entry = by_relative[name]
            require(official["role"] == frozen["original_role"] == "additional_train"
                    and frozen["relative_path"] == name
                    and relative(path(acquisition["output"]) / name) == frozen["path"]
                    and entry.get("status") == frozen["status"] == "completed"
                    and entry.get("source_url") == official["source_url"]
                    and entry.get("actual_bytes") == frozen["actual_bytes"]
                    and entry.get("observed_sha256") == frozen["observed_sha256"]
                    and entry.get("file_stat", entry.get("stat")) == frozen["recorded_file_stat"],
                    "Frozen selected official receipt differs: " + name)
            canonical_file = path(acquisition["output"]) / ".receipts" / (name + ".json")
            require(relative(canonical_file) == frozen["canonical_receipt_path"], "Canonical receipt path differs")
            canonical = read(canonical_file)
            require(canonical.get("status") == "completed" and all(canonical.get(member) == entry.get(member)
                    for member in ("relative_path", "source_url", "actual_bytes", "observed_sha256", "file_stat")),
                    "Canonical completed receipt differs: " + name)
            stat = frozen["recorded_file_stat"]
            require(stat["bytes"] == frozen["actual_bytes"] and type(stat["bytes"]) is int
                    and stat["bytes"] > 0 and re.fullmatch(r"[0-9a-f]{64}", frozen["observed_sha256"]),
                    "Incomplete honest source byte identity")
            source = {"path": frozen["path"], "bytes": stat["bytes"], "mtime_ns": stat["mtime_ns"],
                      "sha256": frozen["observed_sha256"], "source_url": official["source_url"],
                      "role": "additional_train", "source_role": "additional_train",
                      "pilot_role": "evaluation_only", "available": True,
                      "checksum_origin": frozen["checksum_origin"],
                      "canonical_receipt": identity(canonical_file, True)}
            stat_matches(source)
            raw_sources[cad][component] = source
    reader_file = plan["input_reader_policy"]
    reader = read(reader_file)
    input_recipe = {"H5_members": ["vertices", "cells"], "coordinate_columns": COORDS,
        "known_loads": LOADS, "features": 25, "GRID_H5_tolerance_mm": .0002,
        "cloud_points": 1024, "cloud_seed": 2026, "grid_resolution": 64, "grid_channels": 29,
        "normalization": "original FP32 xyz_mm; (xyz_mm-bbox_min)/max_bbox_extent",
        "cloud": "unique original quadratic-boundary FEM IDs; BLAKE2b seed2026:CAD without replacement; zero padding/bool mask",
        "support_flags": "actual RBE2 CM on mapped members and actual SPC; no role-implied clamping",
        "material": "explicit real MAT1 E/nu/rho; no defaults",
        "reader_scope": reader["recipe"]["reader_scope"],
        "load_transfer": reader["recipe"]["load_transfer"], "seed": 2026,
        "scaler_fit": "none; exact original TRAIN40-only frozen scaler", "DEV_input_targets": False,
        "target_columns_presence_check": TARGET_COLUMNS, "numeric_targets_before_barrier": False}
    contract = {"schema": SCHEMA, "plan": relative(plan_path), "scientific_digest": SCIENCE,
        "plan_science": science, "source_sha256": sources,
        "groups": {"train_new160": [], "pilot_evaluation_CADs": selected, "validation20": selected},
        "selected_CADs": selected, "case_keys": selection["case_keys"],
        "validation_keys": selection["case_keys"], "train_keys": {},
        "source_role": "additional_train", "pilot_role": "evaluation_only",
        "helper_validation_is_technical_branch_only": True,
        "scaler": policy["scaler"], "scaler_source": identity(scaler_file, True),
        "input_recipe": input_recipe, "reader_policy": identity(reader_file, True),
        "raw_sources": raw_sources, "selected_source_files": 60,
        "original_acquisition_summary": identity(summary_file, True),
        "original_acquisition_status": "budget_exhausted", "original540_complete": False,
        "original660_complete": False, "original2400_barrier_complete": False,
        "original_negative_is_pilot_case_failure": False,
        "fixed_CAD_denominator": 20, "fixed_case_denominator": 60}
    return contract, plan


def process_snapshot(pid):
    """Read Linux process identity only; no GPU or data access."""
    if type(pid) is not int or pid <= 0:
        return None
    try:
        folder = Path("/proc") / str(pid)
        raw = (folder / "stat").read_text(encoding="utf-8")
        tail = raw[raw.rfind(")") + 2:].split()
        if tail[0] in ("Z", "X"):
            return None
        argv = (folder / "cmdline").read_bytes().split(b"\0")
        return {"pid": pid, "ppid": int(tail[1]), "start_ticks": int(tail[19]),
                "argv": [item.decode("utf-8", errors="strict") for item in argv if item]}
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (OSError, ValueError, UnicodeError, IndexError) as error:
        raise NotReady("Linux process identity could not be established") from error


def exact_process_exited(record):
    require(type(record.get("pid")) is int and record["pid"] > 0
            and type(record.get("start_ticks")) is int and record["start_ticks"] > 0,
            "Exact PID/start-ticks identity is required", NotReady)
    current = process_snapshot(record["pid"])
    return current is None or current["start_ticks"] != record["start_ticks"]


def release(contract, plan, purpose="preparation"):
    require(purpose in ("preparation", "evaluation"), "Unknown pilot release purpose")
    marker = read(plan["root_review"])
    require(marker.get("passed") is True and marker.get("scientific_digest") == SCIENCE
            and marker.get(purpose + "_released") is True
            and marker.get("actual_execution_released") is True
            and marker.get("source_pins") == contract["source_sha256"]
            and marker.get("source_pins", {}).get(OWN) == sha(OWN),
            "Separate genuine available-CAD pilot root release is required", NotReady)
    return marker


def _small_identity_matches(record):
    stat_matches(record)
    require(sha(record["path"]) == record["sha256"], "Frozen small JSON byte identity differs")


def predecessor_gate(plan):
    """Require both authentic terminal/closed clocks and exact process exits."""
    require(os.name != "nt" and Path("/proc").is_dir(), "Use the common WSL process namespace", NotReady)
    marker = read(plan["root_review"])
    bindings = marker.get("upstream_terminal_bindings", {})
    require(set(bindings) == set(PREDECESSOR_PLANS), "Both PP/FNO terminal bindings are required", NotReady)
    proof = {}
    for name, expected_plan in PREDECESSOR_PLANS.items():
        binding = bindings[name]
        upstream = read(expected_plan)
        require(binding.get("plan") == expected_plan and upstream.get("scientific_rules_frozen") is True
                and digest(upstream["scientific"]) == upstream["scientific_digest"],
                "Upstream frozen science binding differs", NotReady)
        folder = path(upstream["output"])
        for member, filename in (("summary_identity", "summary.json"), ("clock_identity", "clock.json"),
                                 ("status_identity", "status.json")):
            require(path(binding[member]["path"]) == folder / filename, "Upstream proof path differs", NotReady)
            _small_identity_matches(binding[member])
        summary = read(folder / "summary.json")
        clock = read(folder / "clock.json")
        status = read(folder / "status.json")
        upstream_contract = summary.get("config", {}).get("contract", {})
        upstream_marker = read(upstream["root_review"])
        require(summary.get("status") in ("completed", "failed") and summary.get("kind") == upstream["kind"]
                and status.get("phase") == "terminal" and status.get("status") == summary["status"]
                and clock.get("closed") is True and clock.get("active") is False
                and upstream_contract.get("scientific_digest") == upstream["scientific_digest"]
                and upstream_contract.get("source_sha256") == upstream_marker.get("source_pins")
                and upstream_marker.get("passed") is True and upstream_marker.get("actual_execution_released") is True,
                "Upstream is not an authentic terminal with a closed inactive clock", NotReady)
        process = binding["run_process"]
        recorded = summary.get("run_process", status.get("run_process"))
        if recorded is not None:
            require(all(recorded.get(key) == process.get(key) for key in ("pid", "start_ticks", "argv")),
                    "PP actual process identity differs", NotReady)
        else:
            require(clock.get("pid") == status.get("pid") == process.get("pid"),
                    "FNO root-bound actual PID differs", NotReady)
        require(exact_process_exited(process), "PP/FNO actual process is still live", NotReady)
        for related in binding.get("related_processes", []):
            require(exact_process_exited(related), "PP/FNO related process is still live", NotReady)
        proof[name] = {"binding": binding, "process_exited": True, "clock_closed_inactive": True}
    return {"PP_and_FNO": proof, "observed_at_utc": utc()}


def validate_external_guard(contract, purpose="preparation"):
    """The child must be directly owned by the reviewed one-shot wall guard."""
    require(purpose in ("preparation", "evaluation"), "Unknown guard purpose")
    require(os.name != "nt" and Path("/proc").is_dir(), "Guard requires common WSL", NotReady)
    variable = "GEOBC_AVAILABLE20_" + ("PREP" if purpose == "preparation" else "EVAL") + "_GUARD"
    require(variable in os.environ, "One-shot external pilot wall guard is required", NotReady)
    guard = json.loads(os.environ[variable])
    require(guard.get("schema") == 1 and guard.get("kind") == GUARD_KIND
            and guard.get("purpose") == purpose and guard.get("scientific_digest") == SCIENCE
            and guard.get("guard_source") == GUARD_SOURCE
            and guard.get("guard_source_sha256") == contract["source_sha256"].get(GUARD_SOURCE)
            and guard["guard_source_sha256"] == sha(GUARD_SOURCE)
            and guard.get("budget_seconds") == 3600 and guard.get("automatic_replay") is False,
            "Reviewed guard identity/budget differs", NotReady)
    for field in ("entry_epoch", "entry_monotonic", "deadline_epoch", "deadline_monotonic",
                  "campaign_entry_epoch", "campaign_entry_monotonic", "campaign_deadline_epoch",
                  "campaign_deadline_monotonic"):
        require(type(guard.get(field)) in (int, float) and math.isfinite(guard[field]), "Guard clock is invalid", NotReady)
    require(abs(guard["deadline_epoch"] - guard["entry_epoch"] - 3600) < .001
            and abs(guard["deadline_monotonic"] - guard["entry_monotonic"] - 3600) < .001
            and abs(guard["campaign_deadline_epoch"] - guard["campaign_entry_epoch"] - 7200) < .001
            and abs(guard["campaign_deadline_monotonic"] - guard["campaign_entry_monotonic"] - 7200) < .001
            and guard["campaign_entry_epoch"] <= guard["entry_epoch"] <= time.time()
            and guard["campaign_entry_monotonic"] <= guard["entry_monotonic"] <= time.monotonic(),
            "Guard whole-stage/campaign clocks differ", NotReady)
    recorded = guard["guard_process"]
    actual = process_snapshot(recorded.get("pid"))
    own = process_snapshot(os.getpid())
    require(actual is not None and own is not None and own["ppid"] == os.getppid() == actual["pid"]
            and all(actual.get(key) == recorded.get(key) for key in ("pid", "start_ticks", "argv")),
            "Actual direct parent is not the exact reviewed guard process", NotReady)
    argv = actual["argv"]
    correct = any(path(token) == path(GUARD_SOURCE) for token in argv if token.endswith(".py"))
    correct = correct or any(token == "tools.run_geobc_available_pilot_guard" and index > 0
                             and argv[index - 1] == "-m" for index, token in enumerate(argv))
    require(correct, "Direct guard argv does not name the reviewed tool", NotReady)
    require(time.monotonic() < min(guard["deadline_monotonic"], guard["campaign_deadline_monotonic"]),
            "External pilot wall budget already exhausted", Budget)
    return guard


@contextmanager
def gpu_lock(filename):
    """The same flock inode as the existing exploration workers; no waiting."""
    require(os.name != "nt", "Actual pilot uses WSL flock", NotReady)
    import fcntl
    filename = path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    handle = filename.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise NotReady("Shared GPU lock is busy") from error
        yield
    finally:
        handle.close()


class PreparationClock:
    def __init__(self, output, guard, run_process):
        self.filename = output / "clock.json"
        require(not self.filename.exists(), "Prior preparation clock exists; root technical review is required", NotReady)
        self.guard = guard
        self.value = {"schema": 1, "purpose": "preparation", "attempts": 1,
            "entry_epoch": guard["entry_epoch"], "entry_monotonic": guard["entry_monotonic"],
            "run_process": run_process, "pid": run_process["pid"], "start_ticks": run_process["start_ticks"],
            "active": True, "closed": False, "phase": "metadata_gate_and_lock",
            "cumulative_wall_seconds": max(0., time.monotonic() - guard["entry_monotonic"]),
            "phase_seconds": {}, "automatic_resume": False, "budget_seconds": 3600,
            "external_guard_is_authoritative": True}
        self.last = time.monotonic()
        self.value["phase_seconds"][self.value["phase"]] = self.value["cumulative_wall_seconds"]
        self.persist()

    def tick(self):
        now = time.monotonic()
        delta = max(0., now - self.last)
        self.value["cumulative_wall_seconds"] += delta
        name = self.value["phase"]
        self.value["phase_seconds"][name] = self.value["phase_seconds"].get(name, 0.) + delta
        self.last = now
        self.value.update(last_epoch=time.time(), last_monotonic=now)

    def persist(self):
        write(self.filename, self.value)

    def phase(self, name):
        self.tick()
        self.value["phase"] = name
        self.persist()

    def guard_budget(self):
        self.tick()
        if self.value["cumulative_wall_seconds"] >= 3600 or time.monotonic() >= min(
                self.guard["deadline_monotonic"], self.guard["campaign_deadline_monotonic"]):
            self.persist()
            raise Budget("Entire preparation wall budget exhausted; fixed20 remains input-negative")

    def close(self):
        self.tick()
        self.value.update(active=False, closed=True, phase="terminal", closed_at_utc=utc())
        self.persist()


class InputLedger:
    """The original helper's attempt/complete interface with persistent zeros."""
    def __init__(self, output, contract, clock):
        self.filename = output / "read_receipt.json"
        require(not self.filename.exists(), "Prior input read ledger exists; no automatic reparse", NotReady)
        self.clock = clock
        self.value = {"input_contract_sha256": digest(contract),
                      **{name: 0 for name in INPUT_COUNTERS + ZERO_COUNTERS},
                      "attempted_reads": {}, "completed_reads": {}}
        write(self.filename, self.value)

    def attempt(self, filename, member=""):
        self.clock.guard_budget()
        require(member != "known3_U_S", "Numeric targets are forbidden in the input producer")
        name = str(filename) + (":" + member if member else "")
        self.value["attempted_reads"][name] = self.value["attempted_reads"].get(name, 0) + 1
        write(self.filename, self.value)

    def complete(self, filename, counter=None, member=""):
        require(counter is None or counter in INPUT_COUNTERS, "Forbidden numeric label read counter")
        name = str(filename) + (":" + member if member else "")
        self.value["completed_reads"][name] = self.value["completed_reads"].get(name, 0) + 1
        if counter:
            self.value[counter] += 1
        require(all(self.value[name] == 0 for name in ZERO_COUNTERS), "Producer forbidden read recorded")
        write(self.filename, self.value)
        self.clock.guard_budget()


def _record_is_target_free(record, contract):
    require(record.get("status") == "completed" and record.get("feasibility_pass") is True
            and record.get("key") in contract["case_keys"]
            and record.get("cad_id") in contract["selected_CADs"]
            and record.get("group") == "validation" and record.get("reused_TRAIN40") is False
            and type(record.get("num_nodes")) is int and record["num_nodes"] > 0
            and set(record.get("file_identities", {})) == set(MEMBERS)
            and "target_path" not in record and "physical_path" not in record,
            "Required complete target-free original-helper input record differs")
    for name in MEMBERS:
        member = record["file_identities"][name]
        require(record.get(name) == member["path"] and re.fullmatch(r"[0-9a-f]{64}", member.get("sha256", "")),
                "Complete input member identity is required")
        stat_matches(member)
    count = record["num_nodes"]
    shapes = {"xyz_path": [count, 3], "cloud_path": [1024, 3], "mask_path": [1024],
              "features_path": [count, 25], "grid_path": [29, 64, 64, 64],
              "query_indices_path": [count, 8], "query_weights_path": [count, 8]}
    for name, shape in shapes.items():
        require(record["file_identities"][name].get("shape") == shape,
                "Original input member shape metadata differs: " + name)
    require(record["deferred_source"] == contract["raw_sources"][record["cad_id"]]["Field"],
            "Deferred source identity differs")
    stat_matches(record["permutation"])
    require(record["permutation"].get("shape") == [record["num_nodes"]]
            and record["permutation"].get("dtype") == "int64", "CSV/H5 permutation metadata differs")


def prepare(plan_path=PLAN, start=False):
    """Default returns immediately, including zero plan/source metadata reads."""
    if not start:
        return {"status": "source_only", "kind": SCHEMA, "actual_execution": False,
                "read_receipt": {"metadata_reads": 0, "raw_reads": 0, "array_reads": 0,
                                 "numeric_GT_reads": 0, "writes": 0}}
    contract, plan = request_contract(plan_path)
    release(contract, plan, "preparation")
    guard = validate_external_guard(contract, "preparation")
    before = predecessor_gate(plan)
    output = path(plan["scientific"]["outputs"]["input_cache"])
    output.relative_to(ROOT.resolve())
    lockfile = shared_gpu_lock(plan)
    with gpu_lock(lockfile):
        after = predecessor_gate(plan)
        require(before["PP_and_FNO"] == after["PP_and_FNO"], "Upstream binding changed while acquiring GPU", NotReady)
        refreshed, _ = request_contract(plan_path)
        require(refreshed == contract, "Input/source metadata changed before actual preparation")
        run_process = process_snapshot(os.getpid())
        require(run_process is not None, "Cannot prove actual preparation process", NotReady)
        for filename in ("manifest.json", "clock.json", "read_receipt.json", "execution.json"):
            require(not (output / filename).exists(), "Prior preparation exists; no automatic replay or budget reset", NotReady)
        require(not (output / "geometry").exists() and not (output / "cases").exists(),
                "Prior owned arrays exist; root technical review is required", NotReady)
        output.mkdir(parents=True, exist_ok=True)
        clock = PreparationClock(output, guard, run_process)
        ledger = InputLedger(output, contract, clock)
        write(output / "execution.json", {"run_process": run_process, "guard": guard,
              "scientific_digest": SCIENCE, "input_contract_sha256": digest(contract), "started_at_utc": utc()})
        manifest = {"kind": SCHEMA, "status": "running", "terminal": False, "contract": contract,
            "case_keys": contract["case_keys"], "validation_keys": contract["case_keys"],
            "train_keys": {}, "scaler": contract["scaler"], "records": {}, "CADs": {},
            "feasibility_pass": False, "fixed_CAD_denominator": 20, "fixed_case_denominator": 60,
            "source_role": "additional_train", "pilot_role": "evaluation_only",
            "helper_validation_is_technical_branch_only": True,
            "run_process": run_process, "guard": guard, "predecessor_proof": after,
            "output": relative(output), "GT_header_metadata": {},
            "read_receipt": ledger.value, "original540_complete": False,
            "original660_complete": False, "original2400_barrier_complete": False}
        write(output / "manifest.json", manifest)
        torch = None
        memory = {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0, "cap_bytes": 24 * 2**30,
                  "measurement_available": False}
        try:
            clock.guard_budget()
            minimum_grids = 60 * 29 * 64**3 * 4
            require(shutil.disk_usage(output).free > minimum_grids,
                    "Insufficient storage for all fixed60 R64 grids", NotReady)
            clock.phase("imports_and_CUDA_initialization")
            import torch
            from geobc import coverage_comparison_data as original
            torch.cuda.init()
            device = plan["recipe_reference"]["device"]
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1., memory["cap_bytes"] / total), device)
            def memory_guard():
                memory.update(peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                              peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)), measurement_available=True)
                write(output / "memory.json", memory)
                require(memory["peak_allocated_bytes"] <= memory["cap_bytes"]
                        and memory["peak_reserved_bytes"] <= memory["cap_bytes"],
                        "Allocated or reserved GPU memory exceeds frozen24 GiB cap", Budget)
                clock.guard_budget()
            memory_guard()
            helper_plan = {"scientific": {"recipe": plan["recipe_reference"]}}
            for cad in contract["selected_CADs"]:
                clock.phase("header_schema_" + cad)
                source = contract["raw_sources"][cad]["Field"]
                stat_matches(source)
                ledger.attempt(source["path"], "header_only")
                with path(source["path"]).open(encoding="utf-8-sig", newline="") as stream:
                    columns = next(csv.reader(stream))
                ledger.complete(source["path"], "CSV_header_only_reads", "header_only")
                require(len(columns) == len(set(columns)) and all(name in columns for name in COORDS + TARGET_COLUMNS),
                        "Original three-load12 label header names or coordinate schema absent", original.UnsupportedInput)
                header = {"columns": columns, "columns_sha256": digest(columns),
                          "known3_target_columns_present": True, "numeric_target_reads": 0}
                manifest["GT_header_metadata"][cad] = header
                clock.phase("original_target_free_prepare_" + cad)
                item = original._prepare_CAD(contract, helper_plan, cad, output, ledger,
                                             lambda status: clock.guard_budget())
                item["header_schema"] = header
                item["source_role"] = "additional_train"
                item["pilot_role"] = "evaluation_only"
                manifest["CADs"][cad] = item
                if not item.get("feasibility_pass"):
                    raise original.UnsupportedInput(item.get("reason", "Selected CAD input incomplete"))
                require(set(item["records"]) == set(keys([cad])), "Selected CAD must supply all three input cases")
                for record in item["records"].values():
                    _record_is_target_free(record, contract)
                    record.update(source_role="additional_train", pilot_role="evaluation_only",
                                  helper_validation_is_technical_branch_only=True)
                manifest["records"].update(item["records"])
                manifest.update(read_receipt=ledger.value, elapsed_seconds=clock.value["cumulative_wall_seconds"])
                write(output / "manifest.json", manifest)
                memory_guard()
            require(set(manifest["records"]) == set(contract["case_keys"])
                    and set(manifest["CADs"]) == set(contract["selected_CADs"]), "Fixed20/60 input inventory incomplete")
            require(all(ledger.value[name] == 0 for name in ZERO_COUNTERS), "Forbidden prebarrier GT read")
            memory_guard()
            manifest.update(status="completed", terminal=True, feasibility_pass=True,
                            scientific_outcome="all_fixed60_target_free_inputs_complete")
        except BaseException as error:
            manifest.update(status="completed" if isinstance(error, Budget) or type(error).__name__ == "UnsupportedInput" else "failed",
                            terminal=True, feasibility_pass=False, scientific_outcome="input_negative",
                            reason=f"{type(error).__name__}: {error}", no_case_substitution=True,
                            numeric_GT_reads=0)
        finally:
            clock.phase("terminal_persistence_and_CUDA_cleanup")
            if torch is not None:
                try:
                    device = plan["recipe_reference"]["device"]
                    torch.cuda.synchronize(device)
                    memory.update(peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                                  peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)), measurement_available=True)
                    torch.cuda.empty_cache()
                except Exception as error:
                    manifest.update(status="failed", feasibility_pass=False, reason=f"CUDA cleanup: {error}")
            if (memory["peak_allocated_bytes"] > memory["cap_bytes"]
                    or memory["peak_reserved_bytes"] > memory["cap_bytes"]):
                manifest.update(status="completed", feasibility_pass=False, scientific_outcome="input_negative",
                                reason="Actual allocated/reserved GPU cap exceeded")
            clock.tick()
            if clock.value["cumulative_wall_seconds"] >= 3600:
                manifest.update(status="completed", feasibility_pass=False, scientific_outcome="input_negative",
                                reason="Entire preparation wall budget exhausted including terminal costs")
            manifest.update(read_receipt=ledger.value, memory=memory,
                            elapsed_seconds=clock.value["cumulative_wall_seconds"],
                            clock_path=relative(clock.filename), prepare_clock_path=relative(clock.filename),
                            external_terminal_path=relative(output / "preparation_external_terminal.json"))
            write(output / "read_receipt.json", ledger.value)
            write(output / "memory.json", memory)
            write(output / "manifest.json", manifest)
            clock.close()
        return manifest


def read_completed_inputs(plan_path=PLAN):
    """Metadata only; actual child exit and authoritative external cost required."""
    contract, plan = request_contract(plan_path)
    output = path(plan["scientific"]["outputs"]["input_cache"])
    manifest = read(output / "manifest.json")
    clock = read(output / "clock.json")
    receipt = read(output / "read_receipt.json")
    execution = read(output / "execution.json")
    external = read(output / "preparation_external_terminal.json")
    require(manifest.get("kind") == SCHEMA and manifest.get("status") in ("completed", "failed")
            and manifest.get("terminal") is True and type(manifest.get("feasibility_pass")) is bool
            and manifest.get("contract") == contract
            and manifest.get("case_keys") == contract["case_keys"]
            and set(manifest.get("records", {})) <= set(contract["case_keys"])
            and set(manifest.get("CADs", {})) <= set(contract["selected_CADs"]),
            "Authentic fixed20/60 input terminal metadata is required")
    require(clock.get("closed") is True and clock.get("active") is False
            and clock.get("attempts") == 1 and 0 <= clock.get("cumulative_wall_seconds", math.inf) <= 3600
            and execution.get("run_process") == manifest.get("run_process") == clock.get("run_process")
            and execution.get("guard") == manifest.get("guard")
            and exact_process_exited(manifest["run_process"]),
            "Actual preparation child has not exited with a closed inactive clock", NotReady)
    require(external.get("schema") == 1 and external.get("kind") == GUARD_KIND
            and external.get("purpose") == "preparation" and external.get("status") == "terminal"
            and external.get("scientific_digest") == SCIENCE
            and external.get("guard") == manifest["guard"]
            and external.get("child_process") == manifest["run_process"]
            and external.get("external_timeout") is False
            and external.get("exit_code") == (1 if manifest["status"] == "failed" else 0)
            and external.get("within_external_budget") is True
            and clock["cumulative_wall_seconds"] <= external.get("elapsed_seconds", -1) <= 3600,
            "Authoritative full-stage external preparation cost/exit proof missing", NotReady)
    require(receipt == manifest.get("read_receipt") and receipt.get("input_contract_sha256") == digest(contract)
            and all(receipt.get(name) == 0 for name in ZERO_COUNTERS)
            and all(type(receipt.get(name)) is int and receipt[name] >= 0 for name in INPUT_COUNTERS),
            "Persistent target-free input receipt differs")
    for record in manifest["records"].values():
        _record_is_target_free(record, contract)
    if manifest["feasibility_pass"]:
        require(set(manifest["records"]) == set(contract["case_keys"])
                and set(manifest["CADs"]) == set(contract["selected_CADs"])
                and set(manifest.get("GT_header_metadata", {})) == set(contract["selected_CADs"])
                and all(receipt.get(name) == count for name, count in
                        (("raw_h5_input_reads", 20), ("raw_deck_input_reads", 20), ("raw_deck_header_reads", 20),
                         ("CSV_coordinate_reads", 20), ("input_member_reads", 40), ("CSV_header_only_reads", 20))),
                "Positive input cache requires every original fixed20/60 input and receipt")
        require(manifest["memory"]["measurement_available"] is True
                and manifest["memory"]["peak_allocated_bytes"] <= 24 * 2**30
                and manifest["memory"]["peak_reserved_bytes"] <= 24 * 2**30,
                "Actual allocated/reserved preparation memory exceeds frozen cap")
        for cad, item in manifest["CADs"].items():
            require(item.get("feasibility_pass") is True
                    and item.get("header_schema", {}).get("known3_target_columns_present") is True
                    and item["header_schema"].get("numeric_target_reads") == 0
                    and item["header_schema"] == manifest["GT_header_metadata"][cad],
                    "Real input/header validation metadata incomplete")
    else:
        require(isinstance(manifest.get("reason"), str) and bool(manifest["reason"]),
                "Real input-negative reason is required; never invent missing-case success")
        manifest["missing_case_keys"] = [key for key in contract["case_keys"] if key not in manifest["records"]]
        manifest["numeric_GT_reads"] = 0
    manifest["authoritative_preparation_wall_seconds"] = external["elapsed_seconds"]
    manifest["preparation_clock"] = clock
    manifest["prepare_clock"] = clock
    manifest["prep_read_receipt"] = receipt
    manifest["preparation_external_terminal"] = external
    return manifest, contract, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    result = prepare(args.plan, args.start)
    print(json.dumps({"status": result["status"], "kind": SCHEMA,
                      "actual_execution": bool(args.start)}, ensure_ascii=False))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
