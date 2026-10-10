#!/usr/bin/env python3
"""Released binary acquisition for the frozen 180-CAD GeoBC coverage screen.

No H5, CSV or deck fields are parsed. Public transport and its atomic observed
byte/SHA receipts are reused; two requests at most are in flight. A failed file
is attempted once per invocation and never replaced by a different CAD.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import tempfile
import threading
import time
import urllib.parse
import uuid

ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_coverage_comparison_plan.json"
KIND = "geobc_solver_coverage40vs200_transolver_all_versions"
SCHEMA = "geobc_coverage_binary_acquisition_v1"
SOURCES = ("tools/fetch_geobc_coverage.py", "tools/start_geobc_coverage_fetch.ps1",
           "tools/fetch_next_cycle_test.py", "tools/fetch_exact_deepjeb.py")
COMPONENTS = {"FieldMesh": ".h5", "Field": ".csv", "SolverDeck": ".fem"}
BLOCK = 1024 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def path(value):
    text = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", text):
        text = text[5].upper() + ":/" + text[7:]
    p = Path(text)
    return (p if p.is_absolute() else ROOT / p).resolve()


def read(value):
    return json.loads(path(value).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def sha(value):
    h = hashlib.sha256()
    with path(value).open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def write(value, data):
    target = path(value); target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                     prefix=target.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    try:
        for retry in range(4):
            try:
                os.replace(temporary, target); return
            except PermissionError:
                if retry == 3:
                    raise
                time.sleep(.04 * (retry + 1))
    finally:
        if temporary.exists():
            temporary.unlink()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def drive_id(url):
    p = urllib.parse.urlparse(url)
    found = re.fullmatch(r"/file/d/([A-Za-z0-9_-]+)/view", p.path)
    require(p.scheme == "https" and p.netloc == "drive.google.com" and found,
            "A directly indexed public Drive file URL is required")
    return found.group(1)


def request_contract(plan_path=PLAN):
    """Only frozen JSON and source identities; no raw files or network."""
    plan = read(plan_path)
    require(plan.get("kind") == KIND, "Wrong coverage plan kind")
    science = plan["scientific"]; science_hash = plan["scientific_digest"]
    require(digest(science) == science_hash, "Frozen scientific digest differs")
    acq = plan["acquisition"]
    require(acq["binary_only"] is True and acq["parse_fields"] is False
            and acq["no_case_substitution"] is True and acq["concurrency"] == 2,
            "Only the frozen two-worker binary acquisition is supported")
    CADs = acq["CADs"]
    require(len(CADs) == 180 and len(set(CADs)) == 180, "Exactly 180 unique frozen CADs required")
    additional = science["additional_train_CADs"]; development = science["DEV_CADs"]
    require(len(additional) == 160 and len(development) == 20 and CADs == additional + development,
            "Acquisition must preserve the frozen 160 TRAIN + 20 DEV order")
    require(not set(CADs) & set(science["selection"]["excluded_seen_ids"]), "Already seen CAD in new acquisition")
    index_record = acq["source_index"]
    require(sha(index_record["path"]) == index_record["sha256"], "Official index identity differs")
    index = read(index_record["path"])
    lookup = {c: {e["name"]: e["url"] for e in index[c]["entries"]} for c in COMPONENTS}
    entries = []
    for e in acq["files"]:
        rel = PurePosixPath(e["relative_path"]); component = e["component"]; cad = e["cad_id"]
        require(component in COMPONENTS and rel.parts == (component, e["name"])
                and not rel.is_absolute() and ":" not in str(rel) and "\\" not in e["relative_path"]
                and rel.stem == cad and cad in CADs and rel.suffix == COMPONENTS[component],
                "File is outside the exact frozen component inventory")
        require(e.get("expected_bytes") is None and e.get("expected_sha256") is None,
                "Upstream sizes/checksums must remain explicitly unknown")
        require(lookup[component].get(e["name"]) == e["source_url"], "Source differs from the official indexed URL")
        role = "additional_train" if cad in additional else "new_DEV"
        require(e["role"] == role, "Frozen CAD role differs")
        entries.append({"cad_id": cad, "role": role, "component": component, "name": e["name"],
                        "relative_path": str(rel), "source_url": e["source_url"],
                        "drive_file_id": drive_id(e["source_url"])})
    expected = [f"{component}/{cad}{suffix}" for cad in CADs for component, suffix in COMPONENTS.items()]
    require(len(entries) == 540 and [e["relative_path"] for e in entries] == expected
            and len({e["drive_file_id"] for e in entries}) == 540, "Exact ordered 540-file inventory required")
    output = path(acq["output"])
    require(output == (ROOT / "data/raw_geobc_coverage").resolve(), "Only the isolated coverage output is allowed")
    budgets = acq["budgets"]
    require(budgets["maximum_wall_seconds"] == 21600 and budgets["maximum_received_bytes"] == 150 * 2**30
            and budgets["minimum_initial_free_bytes"] == 150 * 2**30, "Frozen acquisition budgets differ")
    sources = {name: sha(name) for name in SOURCES}
    for name, expected_sha in plan.get("source_sha256", {}).items():
        if name in sources:
            require(sources[name] == expected_sha, "Pinned fetch source differs: " + name)
    # Whole scientific-source maps and mutable release fields are deliberately
    # excluded, so later data/training source registration cannot change reuse.
    return {"schema": SCHEMA, "scientific_digest": science_hash, "acquisition_digest": digest(acq),
            "CADs": CADs, "files": entries, "output": "data/raw_geobc_coverage", "concurrency": 2,
            "budgets": budgets, "source_index": index_record, "source_sha256": sources,
            "checksum_origin": "Observed received bytes; not upstream checksum or HF revision identity",
            "labels_sealed": True, "scientific_array_reads": 0, "Field_CSV_values_parsed": False}


def release(plan_path, contract):
    plan = read(plan_path); record = plan["root_review"]
    marker_path = record["path"] if isinstance(record, dict) else record
    require(path(marker_path) == (ROOT / "notes/geobc_coverage_root_review.json").resolve(), "Unexpected root review path")
    marker = read(marker_path)
    require(marker.get("passed") is True and marker.get("acquisition_released") is True
            and marker.get("scientific_digest") == contract["scientific_digest"], "Authentic acquisition release missing")
    require(marker.get("source_pins") == plan["source_sha256"], "Root marker source map differs from current plan")
    for name, observed in contract["source_sha256"].items():
        require(plan["source_sha256"].get(name) == observed, "Root did not accept the actual fetch dependency: " + name)


def process_identity(pid):
    """Windows process creation tick; None is proven absent, errors stay unknown."""
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
        if error == 87:
            return None
        raise OSError(error, "Cannot establish the prior download process identity")
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        if code.value != 259:
            return None
        created, exited, kernel_time, user_time = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *[ctypes.byref(x) for x in (created, exited, kernel_time, user_time)]):
            raise ctypes.WinError(ctypes.get_last_error())
        return (created.dwHighDateTime << 32) | created.dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


class AcquisitionLock:
    def __init__(self, output, contract):
        self.target = output / ".fetch.lock"
        self.record = {"pid": os.getpid(), "creation_tick": process_identity(os.getpid()), "host": socket.gethostname(),
                       "contract_sha256": digest(contract), "created_utc": now()}

    def __enter__(self):
        for _ in range(2):
            try:
                fd = os.open(self.target, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                prior = read(self.target)
                require(prior.get("host") == socket.gethostname(), "Prior lock host cannot be established locally")
                live = process_identity(prior["pid"])
                require(live is None or live != prior["creation_tick"], "An existing acquisition process is alive; no duplicate launched")
                archived = self.target.with_name(".fetch.lock.stale." + str(time.time_ns()) + ".json")
                os.replace(self.target, archived)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self.record, stream); stream.flush(); os.fsync(stream.fileno())
            return self
        raise RuntimeError("Concurrent lock creation; no duplicate download")

    def __exit__(self, *unused):
        if self.target.exists() and read(self.target) == self.record:
            self.target.unlink()


def run(plan_path=PLAN, output=None):
    entered = time.monotonic(); contract = request_contract(plan_path); release(plan_path, contract)
    require(os.name == "nt", "Actual acquisition uses the Windows bundled Python/system proxy")
    destination = path(output or contract["output"])
    require(destination == path(contract["output"]), "Output override differs from the isolated frozen output")
    destination.mkdir(parents=True, exist_ok=True)
    with AcquisitionLock(destination, contract):
        from tools.fetch_next_cycle_test import _one_file
        config = {"contract": contract, "binary_only": True, "GPU_work": False}
        config_path = destination / "config.json"
        if config_path.exists():
            require(read(config_path) == config, "Existing acquisition config differs; no raw files changed")
        else:
            write(config_path, config)
        clock_path = destination / "clock.json"; previous = read(clock_path) if clock_path.exists() else {}
        prior_wall = float(previous.get("cumulative_wall_seconds", 0.))
        if previous.get("open_run"):
            prior_wall += max(0., time.time() - previous["last_persist_epoch"])
        began = time.monotonic(); initial_wall = began - entered
        attempts_dir = destination / ".attempts"; attempts_dir.mkdir(exist_ok=True)
        prior_attempts = [read(p) for p in attempts_dir.glob("*.json")]
        prior_network_attempts = sum(bool(a.get("network_phase_started")) for a in prior_attempts)
        prior_bytes = sum(int(a.get("network_bytes_received", 0)) for a in prior_attempts)
        unresolved = [a for a in prior_attempts if a.get("open_attempt")]
        uncertain_bytes = any(a.get("network_phase_started") for a in unresolved)
        # A killed transfer lacks its final return receipt. Preserve the lower
        # bound and reserve the rest of its known HTTP payload (or the entire
        # remaining byte budget if no size exists), rather than inventing bytes.
        prior_reserve = sum(max(0, int(a["content_length"]) - int(a.get("network_bytes_received", 0)))
                            if a.get("content_length") is not None else contract["budgets"]["maximum_received_bytes"]
                            for a in unresolved if a.get("network_phase_started"))
        budget = contract["budgets"]; lock = threading.RLock(); active = {}; finished = {}
        results = {}; stopped = None; initial_free = shutil.disk_usage(destination).free
        run_id = uuid.uuid4().hex; start_utc = now()

        def elapsed():
            return prior_wall + initial_wall + time.monotonic() - began

        def byte_counts():
            return prior_bytes + sum(a["network_bytes_received"] for a in finished.values()) + sum(a["network_bytes_received"] for a in active.values())

        def cap_reason():
            if elapsed() >= budget["maximum_wall_seconds"]:
                return "cumulative_wall_budget"
            if byte_counts() + prior_reserve >= budget["maximum_received_bytes"]:
                return "cumulative_received_byte_budget_or_interrupted_transfer_reserve"
            if initial_free < budget["minimum_initial_free_bytes"]:
                return "minimum_initial_free_space"
            return None

        def persist(phase="running"):
            write(clock_path, {"schema": 1, "contract_sha256": digest(contract), "open_run": phase == "running",
                "cumulative_wall_seconds": elapsed(), "last_persist_epoch": time.time(), "run_id": run_id,
                "observed_network_bytes_lower_bound": byte_counts(), "unclosed_transfer_byte_reserve": prior_reserve,
                "network_bytes_exact": not uncertain_bytes, "failure_and_resume_wall_preserved": True})
            completed = sum(r.get("status") == "completed" for r in results.values())
            failures = sum(r.get("status") == "failed" for r in results.values())
            write(destination / "status.json", {"phase": phase, "pid": os.getpid(), "run_id": run_id,
                "started_at_utc": start_utc, "updated_at_utc": now(), "total_files": 540,
                "completed_files_this_invocation": completed, "failed_files_this_invocation": failures,
                "remaining_files_this_invocation": 540 - len(results), "active_files": list(active.values()),
                "cumulative_wall_seconds": elapsed(), "network_bytes_received_lower_bound": byte_counts(),
                "network_bytes_exact": not uncertain_bytes, "byte_budget_reserve": prior_reserve,
                "network_request_attempts_this_invocation": sum(bool(a["network_phase_started"]) for a in active.values())
                    + sum(bool(a["network_phase_started"]) for a in finished.values()),
                "stop_issuing_reason": stopped or cap_reason(), "initial_free_bytes": initial_free,
                "ETA_seconds": None, "ETA_basis": "Unknown HTTP content lengths, quota and latency",
                "scientific_array_reads": 0, "Field_CSV_values_parsed": False, "labels_sealed": True})

        def progress(entry, phase, received, content_length):
            with lock:
                a = active[entry["relative_path"]]
                a["phase"] = phase; a["content_length"] = content_length; a["updated_at_utc"] = now()
                if phase == "requesting_public_source":
                    a["network_phase_started"] = True
                if phase in ("downloading", "downloaded", "failed"):
                    a["network_bytes_received"] = max(a["network_bytes_received"], int(received))
                write(attempts_dir / (a["attempt_id"] + ".json"), a)
                persist()

        def invoke(entry):
            try:
                return _one_file(entry, destination, progress)
            except Exception as error:
                # Source/size/hash conflicts raised before HTTP are retained as
                # failures, with no overwrite or response-dependent replacement.
                return {**entry, "status": "failed", "run_status": "failed", "error": f"{type(error).__name__}: {error}",
                        "network_bytes_received_this_run": active[entry["relative_path"]]["network_bytes_received"],
                        "scientific_array_reads": 0, "Field_CSV_values_parsed": False, "labels_sealed": True}

        iterator = iter(contract["files"]); futures = {}; exhausted = False
        persist()
        try:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="coverage_binary") as pool:
                while futures or not exhausted:
                    with lock:
                        stopped = stopped or cap_reason()
                        while not exhausted and not stopped and len(futures) < 2:
                            entry = next(iterator, None)
                            if entry is None:
                                exhausted = True; break
                            attempt_id = run_id + "_" + str(len(results) + len(active)).zfill(4)
                            a = {"attempt_id": attempt_id, "relative_path": entry["relative_path"], "open_attempt": True,
                                 "phase": "scheduled", "network_phase_started": False, "network_bytes_received": 0,
                                 "content_length": None, "started_at_utc": now()}
                            active[entry["relative_path"]] = a; write(attempts_dir / (attempt_id + ".json"), a)
                            futures[pool.submit(invoke, entry)] = entry
                        if stopped:
                            exhausted = True
                        persist()
                    if not futures:
                        break
                    ready, _ = wait(futures, timeout=3, return_when=FIRST_COMPLETED)
                    with lock:
                        for future in ready:
                            entry = futures.pop(future); result = future.result(); key = entry["relative_path"]
                            a = active.pop(key); a.update({"open_attempt": False, "phase": result["status"],
                                "network_bytes_received": int(result["network_bytes_received_this_run"]), "result": result,
                                "completed_at_utc": now()})
                            finished[a["attempt_id"]] = a; results[key] = result
                            write(attempts_dir / (a["attempt_id"] + ".json"), a)
                        persist()
        finally:
            # ThreadPoolExecutor waits for already-issued requests. Closing the
            # ledger never discards their observed bytes or failed attempts.
            with lock:
                persist("stopped")
        files = []
        for entry in contract["files"]:
            result = results.get(entry["relative_path"])
            if result is None:
                prior_receipt = destination / ".receipts" / entry["component"] / (entry["name"] + ".json")
                canonical = read(prior_receipt) if prior_receipt.exists() else None
                result = {**entry, "status": "not_attempted_budget", "prior_completed_receipt": canonical,
                          "verified_this_invocation": False, "scientific_array_reads": 0}
            files.append(result)
        complete = all(f["status"] == "completed" for f in files)
        phase = "completed" if complete else "budget_exhausted" if stopped else "failed"
        summary = {"schema": SCHEMA, "kind": KIND, "status": phase, "config": config, "contract": contract,
            "scientific_digest": contract["scientific_digest"], "files": files, "total_files": 540,
            "completed_files": sum(f["status"] == "completed" for f in files),
            "failed_files": sum(f["status"] == "failed" for f in files), "completed_at_utc": now(),
            "costs": {"cumulative_wall_seconds": elapsed(), "actual_network_bytes_lower_bound": byte_counts(),
                "network_bytes_exact": not uncertain_bytes, "unclosed_transfer_byte_reserve": prior_reserve,
                "wall_overshoot_seconds": max(0., elapsed() - budget["maximum_wall_seconds"]),
                "received_byte_overshoot": max(0, byte_counts() - budget["maximum_received_bytes"]),
                "initial_free_bytes": initial_free}, "terminal_reason": stopped,
            "completion_gate": {"passed": complete, "fixed_540_inventory": True, "no_case_substitution": True},
            "read_receipt": {"scientific_array_reads": 0, "Field_CSV_values_parsed": False,
                "label_array_reads": 0, "weight_reads": 0, "CUDA_initialized": False,
                "network_request_attempts_this_invocation": sum(bool(a["network_phase_started"]) for a in finished.values()),
                "cumulative_network_request_attempts": prior_network_attempts + sum(bool(a["network_phase_started"]) for a in finished.values()),
                "completed_file_byte_SHA_reverifications_this_invocation": sum(f.get("run_status") == "verified_existing" for f in files)},
            "resume_policy": "Same source, completed-file size/SHA reverify; unfinished requests restart, no byte-range claim"}
        write(destination / "summary.json", summary); persist(phase)
        print(json.dumps({"status": phase, "completed_files": summary["completed_files"], "total_files": 540,
                          "failed_files": summary["failed_files"], "cumulative_wall_seconds": elapsed(),
                          "summary": str(destination / "summary.json")}, ensure_ascii=False), flush=True)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN); parser.add_argument("--output")
    args = parser.parse_args(); result = run(args.plan, args.output)
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
