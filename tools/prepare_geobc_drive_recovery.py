"""Metadata-only coordinator for same-Drive-ID binary transport recovery.

The original acquisition configuration, clock and attempts remain historical.
This module never fetches a URL or parses an H5/CSV/deck member.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_coverage_comparison_plan.json"
DECISION = "notes/geobc_drive_quota_recovery_decision.json"
MARKER = "notes/geobc_drive_recovery_root_review.json"
HELPERS = ("tools/prepare_geobc_drive_recovery.py", "tools/materialize_geobc_coverage_drive_ref.py")
DEADLINE = "2026-10-10T08:15:02+00:00"
START = "2026-10-10T02:15:02+00:00"
LIMIT = 150 * 2**30
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def now():
    return datetime.now(timezone.utc).isoformat()


def path(value):
    text = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", text):
        text = text[5].upper() + ":/" + text[7:]
    target = Path(text)
    return (target if target.is_absolute() else ROOT / target).resolve()


def read(value):
    return json.loads(path(value).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def sha(value):
    result = hashlib.sha256()
    with path(value).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(value, payload):
    target = path(value); target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=target.parent, encoding="utf-8",
        prefix=target.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    try:
        for attempt in range(5):
            try:
                os.replace(temporary, target); return
            except PermissionError:
                if attempt == 4: raise
                time.sleep(.04 * (attempt + 1))
    finally:
        if temporary.exists(): temporary.unlink()


def require(condition, message):
    if not condition: raise ValueError(message)


def identity(value):
    target = path(value); stat = target.stat()
    return {"path": str(target.relative_to(ROOT)).replace("\\", "/"),
        "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha(target)}


def request_contract(plan_path=PLAN):
    plan = read(plan_path); science = plan["scientific"]; acquisition = plan["acquisition"]
    require(digest(science) == plan["scientific_digest"], "Frozen scientific digest differs")
    decision = read(DECISION)
    require(decision["scientific_digest"] == plan["scientific_digest"]
        and decision["scientific_protocol_unchanged"] is True, "Recovery decision differs")
    old = decision["old_download"]; extension = decision["execution_extension"]
    require(old["completed_files"] == 196 and old["open_attempts_retained"] == 2
        and old["known_network_bytes_lower_bound"] == 8924647897
        and decision["old_unknown_payload_handling"]["declared_conditional_source_size_reservation_bytes"] == 87421990
        and extension["maximum_total_declared_charged_bytes"] == LIMIT
        and extension["minimum_free_bytes"] == LIMIT, "Accepted recovery accounting changed")
    probe = decision["actual_connector_probe"]
    require(probe["actual_bytes"] == 24178652 and probe["elapsed_seconds"] == 8.375,
        "Observed probe receipt differs")
    sources = plan["source_sha256"]
    require(len(sources) == 20, "Original scientific twenty-source map must remain unchanged")
    for name, expected in sources.items(): require(sha(name) == expected, "Original pinned source differs: " + name)
    output = path(acquisition["output"])
    require(output == path("data/raw_geobc_coverage"), "Unfrozen recovery output")
    original = read(output / "config.json")
    entries = original["contract"]["files"]
    require(len(entries) == 540 and len({item["relative_path"] for item in entries}) == 540,
        "Original inventory is not the fixed540")
    by_name = {item["relative_path"]: item for item in entries}
    require(len(acquisition["files"]) == 540, "Frozen acquisition inventory differs")
    for item in acquisition["files"]:
        require(item["relative_path"] in by_name and all(by_name[item["relative_path"]][key] == item[key]
            for key in ("cad_id", "component", "name", "relative_path", "source_url", "role")),
            "Same official Drive inventory changed")
    require(original["contract"]["scientific_digest"] == plan["scientific_digest"], "Original science differs")
    return {"schema": "geobc_Drive_connector_recovery_v1", "output": "data/raw_geobc_coverage",
        "scientific_digest": plan["scientific_digest"], "scientific_source_pins_sha256": digest(sources),
        "acquisition_inventory_sha256": digest(acquisition["files"]), "files": entries,
        "helper_source_sha256": {name: sha(name) for name in HELPERS},
        "original_config": identity(output / "config.json"), "original_clock": identity(output / "clock.json"),
        "budget_decision": identity(DECISION), "original_known_network_bytes_lower_bound": 8924647897,
        "old_conditional_source_size_reservation_bytes": 87421990,
        "inherited_probe_two_hop_charge_bytes": 48357304, "probe": probe,
        "original_persisted_wall_seconds": old["persisted_cumulative_wall_seconds"],
        "absolute_start_utc": START, "absolute_deadline_utc": DEADLINE,
        "maximum_declared_charged_bytes": LIMIT, "minimum_free_bytes": LIMIT,
        "network_bytes_exact": False, "cost_exact": False, "old_open_attempts_preserved": 2}


def context(plan_path=PLAN):
    contract = request_contract(plan_path); marker = read(MARKER)
    require(marker.get("passed") is True and marker.get("acquisition_recovery_released") is True
        and marker.get("scientific_digest") == contract["scientific_digest"]
        and marker.get("scientific_source_pins_sha256") == contract["scientific_source_pins_sha256"]
        and marker.get("acquisition_inventory_sha256") == contract["acquisition_inventory_sha256"]
        and marker.get("helper_source_sha256") == contract["helper_source_sha256"]
        and marker.get("decision") == "release_same_Drive_ID_binary_connector_recovery"
        and marker.get("budget_decision") == "explicit_recovery_accounting", "Genuine recovery release missing")
    return contract, marker, path(contract["output"]) / ".connector_recovery"


def _process_tick(pid):
    require(os.name == "nt", "Recovery ledger uses the accepted Windows process identity")
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        error = ctypes.get_last_error()
        if error == 87: return None
        raise OSError(error, "Prior recovery process identity is unknown")
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)): raise ctypes.WinError(ctypes.get_last_error())
        if code.value != 259: return None
        created, exited, cpu, user = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *[ctypes.byref(item) for item in (created, exited, cpu, user)]):
            raise ctypes.WinError(ctypes.get_last_error())
        return (created.dwHighDateTime << 32) | created.dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


class LedgerLock:
    def __init__(self, directory):
        self.directory = path(directory); self.target = self.directory / ".ledger.lock"
        self.record = {"pid": os.getpid(), "creation_tick": _process_tick(os.getpid()),
            "host": socket.gethostname(), "created_at_utc": now(), "token": uuid.uuid4().hex}

    def __enter__(self):
        # The OS mutex protects lease creation/reclamation as well as every
        # ledger mutation: a second process never observes half-written JSON.
        import ctypes
        from ctypes import wintypes
        self.directory.mkdir(parents=True, exist_ok=True)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        self.kernel.CreateMutexW.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        name = "Local\\GeoBCDriveRecovery_" + hashlib.sha256(str(self.directory).casefold().encode()).hexdigest()
        self.handle = self.kernel.CreateMutexW(None, False, name)
        if not self.handle: raise ctypes.WinError(ctypes.get_last_error())
        result = self.kernel.WaitForSingleObject(self.handle, 45000)
        if result not in (0, 128):
            self.kernel.CloseHandle(self.handle)
            raise TimeoutError("Recovery ledger mutex unavailable; no duplicate mutation")
        try:
            if self.target.exists():
                prior = read(self.target)
                require(prior.get("host") == self.record["host"], "Foreign ledger-lock owner is unknown")
                live = _process_tick(prior["pid"])
                require(live is None or live != prior["creation_tick"], "Live prior ledger lease is inconsistent")
                os.replace(self.target, self.directory / ("stale_lock_" + str(time.time_ns()) + ".json"))
            write(self.target, self.record)
            return self
        except BaseException:
            self.kernel.ReleaseMutex(self.handle); self.kernel.CloseHandle(self.handle); raise

    def __exit__(self, *unused):
        try:
            if self.target.exists() and read(self.target) == self.record: self.target.unlink()
        finally:
            self.kernel.ReleaseMutex(self.handle); self.kernel.CloseHandle(self.handle)


def entry_for(contract, key):
    found = [item for item in contract["files"] if item["relative_path"] == key]
    require(len(found) == 1, "File outside the fixed540 inventory")
    return found[0]


def target_path(contract, entry):
    result = path(contract["output"]) / entry["relative_path"]
    require(result.resolve().is_relative_to(path(contract["output"])), "Unsafe raw target")
    return result


def receipt_path(contract, entry):
    return path(contract["output"]) / ".receipts" / (entry["relative_path"] + ".json")


def load_attempt(directory, attempt_id):
    require(re.fullmatch(r"[0-9a-f]{32}", attempt_id) is not None, "Invalid attempt ID")
    return read(path(directory) / "attempts" / (attempt_id + ".json"))


def update_attempt(directory, attempt_id, changes):
    """Caller holds LedgerLock; never acquire a nested process lock."""
    value = load_attempt(directory, attempt_id)
    require(not {"attempt_id", "relative_path", "drive_file_id", "source_metadata", "size_bytes",
        "two_hop_charge_bytes", "reserved_at_utc"} & set(changes), "Reservation identity/accounting is immutable")
    value.update(changes); value["updated_at_utc"] = now()
    write(path(directory) / "attempts" / (attempt_id + ".json"), value)
    return value


def recompute_clock(contract, directory):
    """Caller holds LedgerLock; failed/open reservations stay charged forever."""
    directory = path(directory); config = read(directory / "config.json")
    require(config["contract"] == contract, "Recovery execution contract differs")
    attempts = [read(item) for item in sorted((directory / "attempts").glob("*.json"))]
    recovery_wall = max(0., time.time() - datetime.fromisoformat(config["created_at_utc"]).timestamp())
    charged = contract["original_known_network_bytes_lower_bound"] + contract["old_conditional_source_size_reservation_bytes"]
    charged += contract["inherited_probe_two_hop_charge_bytes"] + sum(item["two_hop_charge_bytes"] for item in attempts)
    elapsed_values = [item.get("fetch_elapsed_seconds") for item in attempts if not item.get("reuse_probe")]
    value = {"contract_sha256": digest(contract), "updated_at_utc": now(), "recovery_wall_seconds": recovery_wall,
        "cumulative_wall_seconds": contract["original_persisted_wall_seconds"] + contract["probe"]["elapsed_seconds"] + recovery_wall,
        "original_persisted_wall_seconds": contract["original_persisted_wall_seconds"],
        "probe_local_materialization_seconds": 8.375, "probe_API_elapsed_seconds": None,
        "absolute_elapsed_seconds": max(0., time.time() - datetime.fromisoformat(START).timestamp()),
        "conditional_byte_charge": charged, "original_known_network_bytes_lower_bound": 8924647897,
        "old_unknown_payload_conditional_reservation_bytes": 87421990, "old_open_attempts_preserved": 2,
        "inherited_probe_two_hop_charge_bytes": 48357304,
        "new_reserved_two_hop_charge_bytes": sum(item["two_hop_charge_bytes"] for item in attempts),
        "downstream_bytes_observed": sum(item.get("downstream_bytes_observed", 0) for item in attempts),
        "fetch_elapsed_seconds": None,
        "new_attempt_fetch_elapsed_seconds": sum(elapsed_values) if all(type(v) in (int, float) for v in elapsed_values) else None,
        "known_fetch_elapsed_seconds": sum(v for v in elapsed_values if type(v) in (int, float)),
        "local_elapsed_seconds": sum(item.get("local_elapsed_seconds", 0.) for item in attempts),
        "attempts": len(attempts), "open_attempts": sum(item.get("open_attempt") is True for item in attempts),
        "failed_attempts": sum(item.get("status") == "failed" for item in attempts),
        "network_bytes_exact": False, "cost_exact": False, "API_duration_missing_or_unverified": True}
    write(directory / "clock.json", value); return value


def budget_check(contract, directory, next_charge_bytes=0):
    require(type(next_charge_bytes) is int and next_charge_bytes >= 0, "Invalid new reservation charge")
    clock = recompute_clock(contract, directory)
    if time.time() >= datetime.fromisoformat(contract["absolute_deadline_utc"]).timestamp(): return "absolute_six_hour_deadline"
    if clock["conditional_byte_charge"] + next_charge_bytes > contract["maximum_declared_charged_bytes"]: return "declared_byte_budget"
    if shutil.disk_usage(path(contract["output"])).free < contract["minimum_free_bytes"]: return "minimum_free_disk_guard"
    return None


def _canonical(contract, entry):
    filename = receipt_path(contract, entry)
    if not filename.exists(): return None
    value = read(filename); target = target_path(contract, entry)
    require(value.get("status") == "completed" and all(value.get(k) == entry[k]
        for k in ("relative_path", "source_url", "drive_file_id")), "Canonical source identity differs")
    require(type(value.get("actual_bytes")) is int and value["actual_bytes"] > 0
        and HEX64.fullmatch(value.get("observed_sha256", "")), "Canonical byte identity missing")
    stat = target.stat()
    require(value["file_stat"] == {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        and stat.st_size == value["actual_bytes"], "Canonical target stat differs")
    return value


def prepare(plan_path=PLAN):
    contract, marker, directory = context(plan_path)
    with LedgerLock(directory):
        if (directory / "config.json").exists():
            require(read(directory / "config.json")["contract"] == contract, "Existing recovery config differs")
        else:
            old = []
            for entry in contract["files"]:
                value = _canonical(contract, entry)
                if value is not None: old.append({"relative_path": entry["relative_path"],
                    "receipt_identity": identity(receipt_path(contract, entry)), "receipt": value})
            require(len(old) == 196, "Original completed196 receipts differ")
            original_attempts = [identity(item) for item in sorted((path(contract["output"]) / ".attempts").glob("*.json"))]
            write(directory / "config.json", {"contract": contract, "created_at_utc": now(),
                "producer": "mixed_transport_recovery", "credentials": "managed_connector_not_extracted_or_saved",
                "original_completed_receipts": old, "original_attempts": original_attempts,
                "original_history_untouched": True, "old_unknown_records_not_closed": True})
        clock = recompute_clock(contract, directory)
        return {"status": "prepared", "contract_sha256": digest(contract), "recovery_dir": str(directory), "clock": clock}


def pending(plan_path=PLAN):
    contract, marker, directory = context(plan_path)
    with LedgerLock(directory):
        require((directory / "config.json").exists(), "Prepare recovery metadata first")
        attempts = [read(item) for item in (directory / "attempts").glob("*.json")]
        active = {item["relative_path"] for item in attempts if item.get("open_attempt")}
        rows = [{**entry, "reserved_open": entry["relative_path"] in active} for entry in contract["files"]
            if _canonical(contract, entry) is None]
        return {"status": "pending", "pending": rows, "files": rows, "pending_files": len(rows), "clock": recompute_clock(contract, directory)}


def reserve(plan_path, payload):
    contract, marker, directory = context(plan_path); entry = entry_for(contract, payload["relative_path"])
    with LedgerLock(directory):
        require((directory / "config.json").exists(), "Prepare recovery metadata first")
        if _canonical(contract, entry) is not None: return {"status": "already_completed", "relative_path": entry["relative_path"]}
        active_count = 0
        for item in (directory / "attempts").glob("*.json"):
            prior = read(item)
            active_count += prior.get("open_attempt") is True
            require(prior["relative_path"] != entry["relative_path"] or not prior.get("open_attempt"), "Same file has an unresolved reservation")
        require(active_count < 2, "Maximum two active recovery reservations")
        probe = payload.get("reuse_probe") is True
        if probe:
            value = contract["probe"]
            metadata = {"id": value["official_Drive_id"], "name": value["official_name"], "size": value["actual_bytes"],
                "modified_time": value["source_modified_UTC"], "artifact_file_id": value["artifact_file_id"]}
            require(entry["relative_path"] == "SolverDeck/275_400.fem", "Probe reuse is only the observed exact file")
        else:
            source = payload["source_metadata"]
            require(set(source) <= {"id", "name", "size", "modified_time", "artifact_file_id"}, "Unapproved source metadata field")
            metadata = dict(source)
        require(metadata.get("id") == entry["drive_file_id"] and metadata.get("name") == entry["name"]
            and type(metadata.get("size")) is int and metadata["size"] > 0
            and isinstance(metadata.get("modified_time"), str), "Same-ID source metadata differs")
        datetime.fromisoformat(metadata["modified_time"].replace("Z", "+00:00"))
        charge = 0 if probe else 2 * metadata["size"]
        reason = budget_check(contract, directory, charge)
        require(reason is None, "Reservation blocked: " + str(reason))
        attempt_id = uuid.uuid4().hex
        value = {"attempt_id": attempt_id, "relative_path": entry["relative_path"], "drive_file_id": entry["drive_file_id"],
            "source_metadata": metadata, "size_bytes": metadata["size"], "two_hop_charge_bytes": charge,
            "reuse_probe": probe, "open_attempt": True, "status": "reserved", "downstream_bytes_observed": 0,
            "fetch_elapsed_seconds": None, "local_elapsed_seconds": 0., "reserved_at_utc": now()}
        if payload.get("connector_fetch_started_at_utc") is not None:
            datetime.fromisoformat(payload["connector_fetch_started_at_utc"].replace("Z", "+00:00"))
            value["connector_fetch_started_at_utc"] = payload["connector_fetch_started_at_utc"]
        write(directory / "attempts" / (attempt_id + ".json"), value); recompute_clock(contract, directory)
        return {**value, "reservation_id": attempt_id}


def record_fetch_failure(plan_path, payload):
    contract, marker, directory = context(plan_path)
    with LedgerLock(directory):
        prior = load_attempt(directory, payload.get("reservation_id", payload.get("attempt_id")))
        require(prior.get("open_attempt") is True, "Attempt already terminal")
        # Deliberately exclude raw connector errors, signed URLs and credentials.
        reason = str(payload.get("reason", "connector_fetch_failed"))
        require(re.fullmatch(r"[A-Za-z0-9_. -]{1,160}", reason) is not None, "Use a generic failure reason")
        elapsed = payload.get("fetch_elapsed_seconds")
        require(elapsed is None or type(elapsed) in (int, float) and elapsed >= 0, "Invalid actual fetch elapsed")
        changes = {"status": "failed", "open_attempt": False,
            "fetch_elapsed_seconds": elapsed, "reason": reason, "completed_at_utc": now()}
        for name in ("fetch_started_at_utc", "fetched_at_utc"):
            if payload.get(name) is not None:
                datetime.fromisoformat(payload[name].replace("Z", "+00:00")); changes[name] = payload[name]
        value = update_attempt(directory, prior["attempt_id"], changes)
        recompute_clock(contract, directory); return value


def finalize(plan_path=PLAN):
    contract, marker, directory = context(plan_path)
    with LedgerLock(directory):
        config = read(directory / "config.json"); rows = []; missing = []
        for entry in contract["files"]:
            value = _canonical(contract, entry)
            if value is None:
                missing.append(entry["relative_path"]); rows.append({**entry, "status": "not_completed"})
            else: rows.append(value)
        clock = recompute_clock(contract, directory); complete = not missing
        reason = budget_check(contract, directory)
        original = {item["relative_path"]: item for item in config["original_completed_receipts"]}
        for entry in contract["files"]:
            if entry["relative_path"] in original:
                require(identity(receipt_path(contract, entry)) == original[entry["relative_path"]]["receipt_identity"], "Historical196 canonical receipt changed")
        result = {"kind": "geobc_coverage_mixed_transport_binary_acquisition", "status": "completed" if complete else "budget_exhausted" if reason else "incomplete",
            "producer": "mixed_transport_recovery", "config": {"original_config": contract["original_config"],
                "execution_extension": {"contract": contract, "recovery_config_identity": identity(directory / "config.json")}},
            "scientific_digest": contract["scientific_digest"], "files": rows, "total_files": 540,
            "completed_files": 540-len(missing), "missing_files": missing, "costs": clock, "terminal_reason": reason if not complete else None,
            "completion_gate": {"passed": complete, "binary_completeness": complete, "exact_transfer_cost_proven": False,
                "fixed_540_inventory": True, "no_case_substitution": True},
            "read_receipt": {"scientific_array_reads": 0, "Field_CSV_values_parsed": False, "label_array_reads": 0,
                "weight_reads": 0, "CUDA_initialized": False}, "completed_at_utc": now(),
            "transport_disclosure": "196 original anonymous receipts + authenticated managed-connector recovery; no credentials or signed URL saved",
            "SHA_disclosure": "Locally observed binary SHA, not upstream checksum/revision equality; original receipts checked by immutable metadata/stat, subsequent data preparation verifies raw byte SHA"}
        write(directory / "finalization.json", result)
        if complete or reason: write(path(contract["output"]) / "summary.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--plan", default=PLAN)
    parser.add_argument("--action", choices=("prepare", "pending", "reserve", "finalize", "fail", "record-fetch-failure"), required=True)
    args = parser.parse_args()
    if args.action in ("reserve", "fail", "record-fetch-failure"):
        import sys
        payload = json.load(sys.stdin)
        result = reserve(args.plan, payload) if args.action == "reserve" else record_fetch_failure(args.plan, payload)
    else: result = {"prepare": prepare, "pending": pending, "finalize": finalize}[args.action](args.plan)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__": main()
