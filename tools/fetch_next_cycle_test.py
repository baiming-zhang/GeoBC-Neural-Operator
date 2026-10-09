#!/usr/bin/env python3
"""Acquire the fixed fifteen public Drive files; keep all field values sealed.

Only binary transfer, byte counts and SHA256 are performed. No H5/CSV/FEM
contents are parsed. First-observation checksums are not upstream checksums.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import threading
import time
import urllib.parse
import urllib.request

try:
    from .fetch_exact_deepjeb import public_drive_id, looks_like_html, write_json, now, USER_AGENT, BLOCK_SIZE
except ImportError:
    from fetch_exact_deepjeb import public_drive_id, looks_like_html, write_json, now, USER_AGENT, BLOCK_SIZE

ROOT = Path(__file__).resolve().parents[1]
CADS = ("489_12", "616_205", "416_282", "247_50", "142_454")
COMPONENTS = {"FieldMesh": ".h5", "Field": ".csv", "SolverDeck": ".fem"}
CONCURRENCY = 2
TIMEOUT_SECONDS = 60


def host_path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", value):
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def request_contract(plan_path):
    """JSON/source identities only; no file-content verification or network."""
    path = host_path(plan_path); payload = path.read_bytes(); plan = json.loads(payload)
    if plan.get("kind") != "next_cycle_independent_test_acquisition" or plan.get("status") != "ids_frozen_raw_labels_sealed":
        raise ValueError("Expected the frozen, label-sealed next-cycle acquisition plan")
    if plan["test_CADs"] != list(CADS) or plan["loads"] != ["ver", "hor", "tor", "dia"]:
        raise ValueError("The fixed five CAD identities/load declaration changed")
    selection = plan["selection"]
    if selection["seed"] != 2028 or selection["pool_count"] != 2086 or len(set(selection["excluded_seen_ids"])) != 52:
        raise ValueError("Frozen next-cycle metadata selection differs")
    if set(CADS) & set(selection["excluded_seen_ids"]):
        raise ValueError("A proposed independent CAD was already seen")
    expected_ranks = [{"id": cad, "rank": hashlib.sha256(("GeoBC-next-cycle-v1;seed=2028;CAD=" + cad).encode()).hexdigest()} for cad in CADS]
    if selection["chosen"] != expected_ranks:
        raise ValueError("Chosen CAD rank witnesses differ from the frozen selection")
    acquisition = plan["acquisition"]
    if (acquisition["read_labels_as_arrays"] is not False or acquisition["parse_Field_CSV_values"] is not False
            or acquisition["existing_original_data_modified"] is not False or acquisition["no_case_substitution"] is not True):
        raise ValueError("Acquisition must keep labels sealed and original data unchanged")
    if plan["future_eval_gate"]["final_candidates_seeds_checkpoints_budget_metrics_must_be_frozen_before_Field_value_parse"] is not True:
        raise ValueError("The future field-value evaluation gate must remain explicit")
    index_path = host_path(plan["source_index"]); index_payload = index_path.read_bytes()
    index_sha = hashlib.sha256(index_payload).hexdigest()
    if index_sha != plan["source_index_sha256"]:
        raise ValueError("Official source index metadata changed")
    index = json.loads(index_payload)
    lookup = {component: {item["name"]: item["url"] for item in index[component]["entries"]} for component in COMPONENTS}
    entries = []
    for item in plan["files"]:
        relative = PurePosixPath(item["relative_path"]); component = item["component"]
        if (component not in COMPONENTS or relative.is_absolute() or len(relative.parts) != 2
                or ".." in relative.parts or "\\" in item["relative_path"] or ":" in item["relative_path"]
                or relative.parts != (component, item["name"]) or relative.suffix != COMPONENTS[component]
                or relative.stem not in CADS):
            raise ValueError("Unexpected file outside the fifteen frozen component paths")
        if item.get("expected_bytes") is not None or item.get("expected_sha256") is not None:
            raise ValueError("First-acquisition upstream sizes/checksums are explicitly unknown")
        if lookup[component].get(item["name"]) != item["source_url"]:
            raise ValueError("Official indexed file URL differs from the plan")
        entries.append({"component": component, "name": item["name"], "relative_path": str(relative),
                        "source_url": item["source_url"], "drive_file_id": public_drive_id(item["source_url"])})
    expected_files = {f"{component}/{cad}{suffix}" for cad in CADS for component, suffix in COMPONENTS.items()}
    if len(entries) != 15 or {item["relative_path"] for item in entries} != expected_files or len({item["drive_file_id"] for item in entries}) != 15:
        raise ValueError("Exactly fifteen unique frozen component files are required")
    output = host_path(plan["output"])
    if output != (ROOT / "data/raw_next_cycle").resolve():
        raise ValueError("Only the isolated data/raw_next_cycle output is allowed")
    return {"schema": "geobc_next_cycle_drive_acquisition_v1", "plan": str(path),
            "plan_sha256": hashlib.sha256(payload).hexdigest(), "source_index": str(index_path),
            "source_index_sha256": index_sha, "dataset": plan["dataset"], "test_CADs": list(CADS),
            "output": str(output), "files": entries, "concurrency": CONCURRENCY,
            "timeout_seconds": TIMEOUT_SECONDS, "upstream_expected_checksums": None,
            "checksum_policy": "Observed received-byte SHA256; no independent upstream checksum or HuggingFace revision mapping",
            "future_eval_gate": plan["future_eval_gate"], "scientific_array_reads": 0,
            "source_sha256": {"tools/fetch_next_cycle_test.py": file_sha256(Path(__file__)),
                              "tools/fetch_exact_deepjeb.py": file_sha256(Path(__file__).with_name("fetch_exact_deepjeb.py"))}}


def source_check(plan_path):
    ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=__file__)
    contract = request_contract(plan_path)
    return {"status": "source_ready", "contract": contract, "network_requests": 0,
            "scientific_array_reads": 0, "Field_CSV_values_parsed": False, "CUDA_initialized": False,
            "existing_raw_files_read": 0, "existing_raw_or_cache_modified": False}


def _same_source(receipt, entry):
    return all(receipt.get(key) == entry[key] for key in ("relative_path", "source_url", "drive_file_id"))


def _one_file(entry, output, progress):
    began = time.monotonic()
    target = output.joinpath(*PurePosixPath(entry["relative_path"]).parts)
    receipt_path = output / ".receipts" / entry["component"] / (entry["name"] + ".json")
    prior = read_json(receipt_path) if receipt_path.is_file() else None
    if prior is not None and (prior.get("status") != "completed" or not _same_source(prior, entry)):
        raise ValueError("Existing receipt does not identify this exact completed public source")
    if target.exists():
        if not target.is_file() or prior is None:
            raise FileExistsError("Existing file has no matching source receipt; left untouched: " + str(target))
        if target.stat().st_size != prior["actual_bytes"] or file_sha256(target) != prior["observed_sha256"]:
            raise ValueError("Existing file byte size/hash differs from its source receipt; left untouched: " + str(target))
        progress(entry, "verified_existing", prior["actual_bytes"], prior.get("content_length"))
        return {**prior, "run_status": "verified_existing", "verification_seconds": time.monotonic() - began,
                "file_stat": {"bytes": target.stat().st_size, "mtime_ns": target.stat().st_mtime_ns},
                "network_downloads_this_run": 0, "network_bytes_this_run": 0, "network_bytes_received_this_run": 0}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    received = 0
    network_received = 0; content_length = None; content_type = None; final_host = None
    query = urllib.parse.urlencode({"export": "download", "id": entry["drive_file_id"]})
    request = urllib.request.Request("https://drive.google.com/uc?" + query,
                                     headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    progress(entry, "requesting_public_source", 0, None)
    try:
        # No login, cookie jar, browser profile, credential, or HTML form parsing.
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            final_host = urllib.parse.urlparse(response.geturl()).hostname or ""
            if final_host not in ("drive.google.com", "drive.usercontent.google.com") and not final_host.endswith(".googleusercontent.com"):
                raise RuntimeError("Public Drive redirected outside its permitted public file hosts")
            content_type = response.headers.get("Content-Type", "")
            encoding = response.headers.get("Content-Encoding", "identity").lower()
            if encoding not in ("", "identity"):
                raise RuntimeError("Unexpected encoded HTTP payload; byte identity is not assumed")
            length_header = response.headers.get("Content-Length")
            content_length = None if length_header is None else int(length_header)
            if content_length is not None and content_length <= 0:
                raise RuntimeError("Public download reports an empty or invalid Content-Length")
            first = response.read(BLOCK_SIZE)
            network_received += len(first)
            if "text/html" in content_type.lower() or looks_like_html(first):
                raise RuntimeError("Public Drive returned HTML/login/confirmation; no cookies or confirmation automation used")
            if not first:
                raise RuntimeError("Public Drive returned no file bytes")
            digest = hashlib.sha256(); last_report = time.monotonic()
            with tempfile.NamedTemporaryFile("wb", dir=target.parent, prefix=target.name + ".", suffix=".part", delete=False) as handle:
                temporary = Path(handle.name)
                block = first
                while block:
                    handle.write(block); digest.update(block); received += len(block)
                    if content_length is not None and received > content_length:
                        raise ValueError("Received bytes exceed the HTTP Content-Length")
                    if time.monotonic() - last_report >= 3:
                        progress(entry, "downloading", received, content_length); last_report = time.monotonic()
                    block = response.read(BLOCK_SIZE)
                    network_received += len(block)
                handle.flush(); os.fsync(handle.fileno())
        if content_length is not None and received != content_length:
            raise ValueError("Download ended before its HTTP Content-Length")
        observed_sha = digest.hexdigest()
        if prior is not None and (received != prior["actual_bytes"] or observed_sha != prior["observed_sha256"]):
            raise ValueError("Redownload differs from the prior observed source bytes; no file replaced")
        if target.exists():
            raise FileExistsError("Another writer created the destination; no file replaced: " + str(target))
        receipt = {**entry, "status": "completed", "path": str(target), "actual_bytes": received,
                   "observed_sha256": observed_sha, "content_length": content_length,
                   "content_length_matched": received == content_length if content_length is not None else None,
                   "content_type": content_type, "response_file_host": final_host,
                   "checksum_origin": "Locally calculated from received bytes, not an independent upstream checksum",
                   "upstream_expected_sha256": None, "upstream_revision_mapping": None,
                   "file_stat": {"bytes": temporary.stat().st_size, "mtime_ns": temporary.stat().st_mtime_ns},
                   "completed_at_utc": now(), "download_elapsed_seconds": time.monotonic() - began,
                   "scientific_array_reads": 0, "Field_CSV_values_parsed": False, "labels_sealed": True}
        # Commit the complete-file source receipt first. A missing target can then
        # be restored only by reproducing these exact observed bytes after failure.
        write_json(receipt_path, receipt)
        os.replace(temporary, target); temporary = None
        progress(entry, "downloaded", received, content_length)
        return {**receipt, "run_status": "downloaded", "verification_seconds": 0.,
                "network_downloads_this_run": 1, "network_bytes_this_run": received,
                "network_bytes_received_this_run": network_received}
    except Exception as error:
        progress(entry, "failed", received, content_length)
        return {**entry, "status": "failed", "run_status": "failed", "error": f"{type(error).__name__}: {error}",
                "file_bytes_received_before_failure": received, "network_bytes_received_this_run": network_received,
                "content_length": content_length, "content_type": content_type, "response_file_host": final_host,
                "attempt_elapsed_seconds": time.monotonic() - began, "scientific_array_reads": 0,
                "Field_CSV_values_parsed": False, "labels_sealed": True}
    finally:
        if temporary is not None and temporary.exists():
            # Only this process's randomly named unfinished file is removed.
            temporary.unlink()


def run(args):
    if args.source_check:
        result = source_check(args.plan); print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return result
    contract = request_contract(args.plan); output = host_path(args.output or contract["output"])
    if output != host_path(contract["output"]):
        raise ValueError("--output must be the frozen isolated data/raw_next_cycle path")
    config = {"contract": contract, "labels_sealed": True, "scientific_array_reads": 0,
              "Field_CSV_values_parsed": False, "native_or_GPU_work": False}
    config_path = output / "config.json"
    if config_path.is_file() and read_json(config_path) != config:
        raise ValueError("Existing acquisition config contract differs; files left untouched")
    output.mkdir(parents=True, exist_ok=True); write_json(config_path, config)
    began = time.monotonic(); lock = threading.Lock()
    status_path = output / "status.json"
    previous = read_json(status_path) if status_path.is_file() else {}
    previous_elapsed = previous.get("elapsed_seconds", 0.)
    previous_requests = previous.get("cumulative_network_requests", 0)
    state = {"phase": "running", "started_at_utc": now(), "files": {}, "completed_files": 0,
             "total_files": 15, "network_requests_this_run": 0, "cumulative_network_requests": previous_requests,
             "scientific_array_reads": 0, "Field_CSV_values_parsed": False, "labels_sealed": True,
             "eta_seconds": None, "eta_basis": "Remaining content lengths and public endpoint latency are unknown"}
    def progress(entry, phase, received, content_length):
        with lock:
            if phase == "requesting_public_source":
                state["network_requests_this_run"] += 1; state["cumulative_network_requests"] += 1
            state["files"][entry["relative_path"]] = {"phase": phase, "bytes_received": received, "content_length": content_length}
            state["completed_files"] = sum(row["phase"] in ("downloaded", "verified_existing") for row in state["files"].values())
            state.update(updated_at_utc=now(), elapsed_seconds=previous_elapsed + time.monotonic() - began)
            write_json(status_path, state)
    write_json(status_path, state)
    results = {}
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(_one_file, entry, output, progress): entry for entry in contract["files"]}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                result = future.result()
                if result["status"] == "completed":
                    print(f"{result['run_status']}: {entry['relative_path']} ({result['actual_bytes']} bytes)", flush=True)
                else:
                    print(f"failed: {entry['relative_path']}: {result['error']}", flush=True)
            except Exception as error:
                result = {**entry, "status": "failed", "run_status": "failed", "error": str(error),
                          "scientific_array_reads": 0, "Field_CSV_values_parsed": False}
                print(f"failed: {entry['relative_path']}: {error}", flush=True)
            results[entry["relative_path"]] = result
    rows = [results[entry["relative_path"]] for entry in contract["files"]]
    failed = sum(row["status"] != "completed" for row in rows)
    elapsed_this_run = time.monotonic() - began
    summary = {"status": "failed" if failed else "completed", "config": config,
               "elapsed_seconds": previous_elapsed + elapsed_this_run, "elapsed_seconds_this_run": elapsed_this_run,
               "completed_at_utc": now(), "completed_files": 15 - failed, "failed_files": failed,
               "files": rows, "download_count": sum(row.get("network_downloads_this_run", 0) for row in rows),
               "verified_existing_count": sum(row["run_status"] == "verified_existing" for row in rows),
               "network_requests_this_run": state["network_requests_this_run"],
               "cumulative_network_requests": state["cumulative_network_requests"],
               "request_count_definition": "Initial public-source urlopen attempts; urllib redirect hops are not counted separately",
               "network_bytes_successful_this_run": sum(row.get("network_bytes_this_run", 0) for row in rows),
               "network_bytes_received_this_run": sum(row.get("network_bytes_received_this_run", 0) for row in rows),
               "completed_file_bytes": sum(row.get("actual_bytes", 0) for row in rows if row["status"] == "completed"),
               "verification_seconds_this_run": sum(row.get("verification_seconds", 0.) for row in rows),
               "concurrency": CONCURRENCY, "scientific_array_reads": 0, "Field_CSV_values_parsed": False,
               "labels_sealed": True, "CAD_substitutions": 0, "original_raw_cache_provenance_modified": False,
               "checksum_limit": "Received-byte integrity only; independent upstream checksum/revision mapping absent"}
    write_json(output / "summary.json", summary)
    state.update(phase=summary["status"], elapsed_seconds=summary["elapsed_seconds"], updated_at_utc=now(), failed_files=failed)
    write_json(status_path, state)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, required=True)
    result.add_argument("--output", type=Path)
    result.add_argument("--source-check", action="store_true")
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    raise SystemExit(0 if result["status"] in ("completed", "source_ready") else 1)
