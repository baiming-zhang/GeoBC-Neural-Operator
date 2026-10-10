#!/usr/bin/env python3
"""Materialize one reserved, same-Drive-ID managed connector file reference.

Read the ephemeral signed HTTPS file reference only from stdin. No URI, query,
credential, response body or unsanitized exception is saved or printed. This
binary transport does not parse H5/CSV/FEM content. The companion metadata tool
owns fixed inventory, authentic root release, reservations, locks and budgets.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from tools import prepare_geobc_drive_recovery as recovery

BLOCK = 1024 * 1024
MAX_STDIN_BYTES = 128 * 1024
TIMEOUT_SECONDS = 60
DEADLINE_EPOCH = datetime.fromisoformat("2026-10-10T08:15:02+00:00").timestamp()
SAFE_ARTIFACT = re.compile(r"^sediment://file_[A-Za-z0-9_]+$")


class MaterializationError(RuntimeError):
    """Only fixed, non-sensitive error codes cross the CLI boundary."""


def require(condition, code):
    if not condition:
        raise MaterializationError(code)


def finite_seconds(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "fetch_elapsed_seconds_required")
    return float(value)


def artifact(value):
    if value is None:
        return None
    require(isinstance(value, str) and SAFE_ARTIFACT.fullmatch(value),
            "artifact_reference_must_be_unsigned_file_identifier")
    return value


def safe_timestamp(value, required=False):
    if value is None and not required:
        return None
    require(isinstance(value, str) and len(value) <= 64, "timestamp_metadata_required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        raise MaterializationError("invalid_timestamp_metadata") from None
    require(parsed.tzinfo is not None, "timestamp_timezone_required")
    return value


def safe_source_metadata(value, entry):
    require(isinstance(value, dict), "source_metadata_required")
    size = value.get("size")
    if isinstance(size, str) and size.isdecimal():
        size = int(size)
    require(type(size) is int and size > 0, "positive_API_size_required")
    require(value.get("id") == entry["drive_file_id"] and value.get("name") == entry["name"],
            "connector_original_Drive_identity_mismatch")
    modified = safe_timestamp(value.get("modified_time"), required=True)
    # Only explicitly permitted metadata keys are persisted. A nested API
    # download URL or SAS cannot leak through an unrestricted dict copy.
    return {"id": entry["drive_file_id"], "name": entry["name"], "size": size,
            "modified_time": modified,
            "artifact_file_id": artifact(value.get("artifact_file_id")),
            "upstream_expected_sha256": None,
            "wrapper_version_returned": False}


def source_matches_reservation(source, reserved):
    return all(source.get(key) == reserved.get(key) for key in ("id", "name", "size", "modified_time"))


def permitted_url(value, marker):
    require(isinstance(value, str) and 0 < len(value) <= 65536, "HTTPS_reference_required")
    try:
        parsed = urllib.parse.urlsplit(value)
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except Exception:
        raise MaterializationError("invalid_ephemeral_reference") from None
    exact = marker.get("allowed_materialization_hosts", [])
    suffixes = marker.get("allowed_materialization_host_suffixes", [])
    require(isinstance(exact, list) and isinstance(suffixes, list), "root_host_allowlist_required")
    allowed = host in {str(item).lower() for item in exact}
    allowed = allowed or any(host == str(item).lower().lstrip(".")
                             or host.endswith("." + str(item).lower().lstrip("."))
                             for item in suffixes if item)
    require(parsed.scheme == "https" and not parsed.username and not parsed.password
            and port in (None, 443) and host and allowed,
            "reference_host_outside_actual_root_allowlist")
    return value


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, marker):
        self.marker = marker

    def redirect_request(self, request, file_pointer, code, message, headers, newurl):
        permitted_url(newurl, self.marker)
        return super().redirect_request(request, file_pointer, code, message, headers, newurl)


def looks_like_error_body(block):
    prefix = block[:2048].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    return (prefix.startswith((b"<!doctype html", b"<html", b"<head", b"<body", b"<?xml"))
            or b"<title>" in prefix or prefix.startswith(b"<error>"))


def error_code(error):
    if isinstance(error, MaterializationError):
        return str(error)
    if isinstance(error, urllib.error.HTTPError):
        return "HTTP_status_" + str(int(error.code))
    # In particular str(URLError/OSError) can embed a signed reference. Never
    # persist it, a traceback, response headers or request/response objects.
    return type(error).__name__


def update(recovery_dir, contract, attempt_id, fields):
    with recovery.LedgerLock(recovery_dir):
        recovery.update_attempt(recovery_dir, attempt_id, fields)
        return recovery.recompute_clock(contract, recovery_dir)


def guard(contract, recovery_dir):
    with recovery.LedgerLock(recovery_dir):
        reason = recovery.budget_check(contract, recovery_dir, next_charge_bytes=0)
    require(reason is None, "operational_budget_" + str(reason))


def read_request():
    raw = sys.stdin.buffer.read(MAX_STDIN_BYTES + 1)
    require(len(raw) <= MAX_STDIN_BYTES, "stdin_limit_exceeded")
    try:
        result = json.loads(raw)
    except Exception:
        raise MaterializationError("invalid_stdin_JSON") from None
    require(isinstance(result, dict), "stdin_JSON_object_required")
    return result


def materialize(request, plan_path=recovery.PLAN):
    entered = time.monotonic()
    contract, marker, recovery_dir = recovery.context(plan_path)
    key = request.get("relative_path")
    entry = recovery.entry_for(contract, key)
    attempt_id = request.get("reservation_id", request.get("attempt_id"))
    require(request.get("attempt_id", attempt_id) == attempt_id, "reservation_alias_conflict")
    require(isinstance(attempt_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{8,96}", attempt_id),
            "valid_pre_API_reservation_id_required")
    with recovery.LedgerLock(recovery_dir):
        attempt = recovery.load_attempt(recovery_dir, attempt_id)
        require(attempt.get("relative_path") == key and attempt.get("drive_file_id") == entry["drive_file_id"],
                "reservation_inventory_identity_mismatch")
        if attempt.get("status") == "completed":
            receipt = recovery.read(recovery.receipt_path(contract, entry))
            target = recovery.target_path(contract, entry)
            require(target.is_file() and target.stat().st_size == receipt.get("actual_bytes"),
                    "completed_target_stat_mismatch")
            return {"status": "completed", "relative_path": key, "attempt_id": attempt_id, "reservation_id": attempt_id,
                    "run_status": "completed_reservation_reused", "receipt": str(recovery.receipt_path(contract, entry)),
                    "actual_bytes": receipt["actual_bytes"], "observed_sha256": receipt["observed_sha256"],
                    "network_downloads_this_invocation": 0, "signed_reference_persisted": False}
        require(attempt.get("status") == "reserved" and attempt.get("open_attempt") is True,
                "reservation_is_not_available_for_one_materialization")
        probe = attempt.get("reuse_probe") is True
        fetch_seconds = None if probe else finite_seconds(request.get("fetch_elapsed_seconds"))
        source = attempt["source_metadata"] if probe else safe_source_metadata(request.get("source_metadata"), entry)
        require(source_matches_reservation(source, attempt["source_metadata"]),
                "postfetch_metadata_differs_from_reserved_original_source")
        recovery.update_attempt(recovery_dir, attempt_id, {
            "status": "materializing", "phase": "materializing", "materializing_pid": os.getpid(),
            "materialization_started_at_utc": recovery.now(), "materialized_source_metadata": source,
            "artifact_file_id": source.get("artifact_file_id"),
            "fetch_elapsed_seconds": fetch_seconds,
            "fetch_elapsed_precision": "unknown_inherited_probe_API" if probe else "caller_observed_actual",
            "API_fetch_started_at_utc": None if probe else safe_timestamp(request.get("fetch_started_at_utc")),
            "API_fetch_completed_at_utc": None if probe else safe_timestamp(request.get("fetched_at_utc")),
            "local_elapsed_seconds": max(float(attempt.get("local_elapsed_seconds", 0)), time.monotonic() - entered),
            "signed_reference_persisted": False})
        recovery.recompute_clock(contract, recovery_dir)
    expected = source["size"]
    received = 0
    local_received = 0
    content_type = None
    content_length = None
    temporary = recovery_dir / "partials" / (attempt_id + ".bin.part")
    target = recovery.target_path(contract, entry)
    canonical_path = recovery.receipt_path(contract, entry)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    response = None
    stream = None
    digest = hashlib.sha256()
    began = time.monotonic()
    last_report = began
    prior = None
    expected_probe_sha = None
    try:
        guard(contract, recovery_dir)
        if canonical_path.is_file():
            prior = recovery.read(canonical_path)
            require(prior.get("status") == "completed"
                    and all(prior.get(field) == entry[field] for field in ("relative_path", "source_url", "drive_file_id")),
                    "canonical_source_receipt_conflict")
        if target.exists():
            # Never overwrite a former successful download, even if a caller
            # accidentally fetched a duplicate managed reference.
            require(target.is_file() and prior is not None and target.stat().st_size == prior["actual_bytes"],
                    "existing_target_has_no_matching_complete_receipt")
            update(recovery_dir, contract, attempt_id, {
                "status": "completed", "phase": "completed_existing_target", "open_attempt": False,
                "local_elapsed_seconds": time.monotonic() - entered,
                "downstream_bytes_observed": 0, "completed_at_utc": recovery.now(),
                "source_reference_fetch_already_charged": not probe})
            return {"status": "completed", "relative_path": key, "attempt_id": attempt_id, "reservation_id": attempt_id,
                    "run_status": "existing_canonical_file_preserved", "actual_bytes": prior["actual_bytes"],
                    "observed_sha256": prior["observed_sha256"], "network_downloads_this_invocation": 0,
                    "signed_reference_persisted": False}
        require(not temporary.exists(), "owned_partial_already_exists_no_silent_restart")
        if probe:
            metadata = recovery.read(recovery_dir / "probe_receipt.json")
            require(metadata.get("status") == "completed_binary_materialization"
                    and metadata.get("official_Drive_id") == entry["drive_file_id"]
                    and metadata.get("official_name") == entry["name"]
                    and metadata.get("actual_bytes") == expected,
                    "inherited_probe_metadata_identity_mismatch")
            probe_path = recovery.path(metadata["path"])
            require(probe_path == recovery_dir / "275_400.fem.bin" and probe_path.is_file()
                    and probe_path.stat().st_size == expected, "inherited_probe_binary_stat_mismatch")
            expected_probe_sha = metadata["sha256"]
            stream = probe_path.open("rb")
            content_type = "application/octet-stream"
            content_length = expected
        else:
            reference = permitted_url(request.get("file_uri"), marker)
            opener = urllib.request.build_opener(SafeRedirect(marker))
            response = opener.open(urllib.request.Request(reference, headers={
                "User-Agent": "GeoBC-fixed-coverage-connector-materialization/1.0",
                "Accept-Encoding": "identity"}), timeout=TIMEOUT_SECONDS)
            # Every redirected location was already checked without logging it.
            permitted_url(response.geturl(), marker)
            require(response.status == 200, "materialization_requires_complete_HTTP_200")
            declared_content_type = response.headers.get("Content-Type", "")
            require("html" not in declared_content_type.lower() and "json" not in declared_content_type.lower(),
                    "managed_reference_returned_nonbinary_error_content")
            content_type = declared_content_type.split(";", 1)[0].strip().lower()
            require(not content_type or re.fullmatch(r"[a-z0-9.+_-]+/[a-z0-9.+_-]+", content_type),
                    "invalid_content_type_metadata")
            encoding = response.headers.get("Content-Encoding", "identity").lower()
            require(encoding in ("", "identity"), "unexpected_encoded_payload")
            length = response.headers.get("Content-Length")
            content_length = None if length is None else int(length)
            require(content_length is None or content_length == expected,
                    "HTTP_length_differs_from_original_API_size")
            stream = response
        with temporary.open("xb") as output:
            first = True
            while True:
                # The full two-hop source-size charge is reserved before API
                # fetch. Do not rescan/write the growing metadata ledger once
                # per MiB; only the absolute deadline needs a per-chunk check.
                require(time.time() < DEADLINE_EPOCH, "operational_budget_absolute_deadline")
                block = stream.read(min(BLOCK, max(1, expected - received + 1)))
                if not probe:
                    local_received += len(block)
                if not block:
                    break
                received += len(block)
                require(received <= expected, "payload_exceeds_reserved_source_size")
                if first:
                    require(not looks_like_error_body(block), "HTML_or_XML_error_body_rejected")
                    first = False
                output.write(block)
                digest.update(block)
                if time.monotonic() - last_report >= 3:
                    update(recovery_dir, contract, attempt_id, {
                        "downstream_bytes_observed": local_received, "copied_file_bytes": received,
                        "local_elapsed_seconds": time.monotonic() - entered, "updated_at_utc": recovery.now()})
                    guard(contract, recovery_dir)
                    last_report = time.monotonic()
            output.flush()
            os.fsync(output.fileno())
        require(received == expected and received > 0, "payload_truncated_or_empty")
        observed_sha = digest.hexdigest()
        require(expected_probe_sha is None or observed_sha == expected_probe_sha,
                "inherited_probe_observed_SHA_mismatch")
        require(prior is None or (prior["actual_bytes"] == received and prior["observed_sha256"] == observed_sha),
                "restored_bytes_differ_from_prior_canonical_receipt")
        guard(contract, recovery_dir)
        stat = temporary.stat()
        receipt = {**entry, "status": "completed", "path": str(target),
            "actual_bytes": received, "observed_sha256": observed_sha,
            "file_stat": {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            "content_length": content_length, "content_length_matched": content_length == received if content_length is not None else None,
            "content_type": content_type, "completed_at_utc": recovery.now(),
            "download_elapsed_seconds": time.monotonic() - entered,
            "producer": "GoogleDriveConnector_same_original_ID_binary_materialization",
            "transport": "authenticated Google Drive connector raw file reference",
            "source_metadata": source, "artifact_file_id": source.get("artifact_file_id"),
            "source_modified_UTC": source["modified_time"],
            "connector_recovery_attempt_id": attempt_id,
            "helper_source_sha256": contract["helper_source_sha256"],
            "scientific_digest": contract["scientific_digest"],
            "checksum_origin": "Observed materialized binary SHA256; no independent upstream checksum returned",
            "upstream_expected_sha256": None,
            "signed_reference_persisted": False, "download_url_saved": False,
            "reused_existing_probe_without_network": probe,
            "fetch_elapsed_seconds": fetch_seconds,
            "fetch_elapsed_precision": "unknown_inherited_probe_API" if probe else "caller_observed_actual",
            "scientific_array_reads": 0, "Field_CSV_values_parsed": False, "labels_sealed": True}
        with recovery.LedgerLock(recovery_dir):
            require(not target.exists(), "concurrent_target_created_no_overwrite")
            recovery.write(canonical_path, receipt)
            os.replace(temporary, target)
            recovery.update_attempt(recovery_dir, attempt_id, {
                "status": "completed", "phase": "completed", "open_attempt": False,
                "downstream_bytes_observed": local_received, "copied_file_bytes": received,
                "local_elapsed_seconds": time.monotonic() - entered,
                "completed_at_utc": recovery.now(), "actual_bytes": received,
                "observed_sha256": observed_sha, "canonical_receipt": str(canonical_path),
                "signed_reference_persisted": False})
            recovery.recompute_clock(contract, recovery_dir)
        return {"status": "completed", "relative_path": key, "attempt_id": attempt_id, "reservation_id": attempt_id,
                "actual_bytes": received, "observed_sha256": observed_sha,
                "network_downloads_this_invocation": 0 if probe else 1,
                "signed_reference_persisted": False, "receipt": str(canonical_path)}
    except Exception as error:
        code = error_code(error)
        update(recovery_dir, contract, attempt_id, {
            "status": "failed", "phase": "failed", "open_attempt": False,
            "downstream_bytes_observed": local_received, "copied_file_bytes": received,
            "local_elapsed_seconds": time.monotonic() - entered,
            "completed_at_utc": recovery.now(), "error_code": code,
            "owned_partial_retained": temporary.exists(), "signed_reference_persisted": False})
        return {"status": "failed", "relative_path": key, "attempt_id": attempt_id, "reservation_id": attempt_id,
                "error_code": code, "downstream_bytes_observed": local_received,
                "signed_reference_persisted": False}
    finally:
        if stream is not None and (probe or stream is not response):
            stream.close()
        if response is not None:
            response.close()
        # Drop the secret-bearing local binding; nothing serializes request.
        request.pop("file_uri", None)


def main():
    class SafeParser(argparse.ArgumentParser):
        def error(self, unused_message):
            print(json.dumps({"status": "failed", "error_code": "invalid_cli_arguments",
                              "signed_reference_persisted": False}), flush=True)
            raise SystemExit(2)

    parser = SafeParser(description=__doc__)
    parser.add_argument("--plan", default=recovery.PLAN)
    args = parser.parse_args()
    request = None
    try:
        request = read_request()
        result = materialize(request, args.plan)
    except Exception as error:
        code = error_code(error)
        # Rejected postfetch metadata still consumed a real API request. If an
        # authenticated reservation is identifiable and belongs to this caller,
        # retain its observed duration and terminal failure without leaking URI.
        try:
            if isinstance(request, dict):
                contract, unused_marker, recovery_dir = recovery.context(args.plan)
                attempt_id = request.get("reservation_id", request.get("attempt_id"))
                with recovery.LedgerLock(recovery_dir):
                    attempt = recovery.load_attempt(recovery_dir, attempt_id)
                    owned = attempt.get("status") == "reserved" or (
                        attempt.get("status") == "materializing" and attempt.get("materializing_pid") == os.getpid())
                    if owned and attempt.get("relative_path") == request.get("relative_path"):
                        duration = request.get("fetch_elapsed_seconds")
                        duration = float(duration) if type(duration) in (int, float) and math.isfinite(duration) and duration >= 0 else None
                        recovery.update_attempt(recovery_dir, attempt_id, {
                            "status": "failed", "phase": "failed_before_body", "open_attempt": False,
                            "fetch_elapsed_seconds": duration, "error_code": code,
                            "completed_at_utc": recovery.now(), "signed_reference_persisted": False})
                        recovery.recompute_clock(contract, recovery_dir)
        except Exception:
            # Its reservation remains visible/open and charged if identity or
            # ledger access could not be proven; never forge successful closure.
            pass
        result = {"status": "failed", "error_code": code,
                  "signed_reference_persisted": False}
    finally:
        if isinstance(request, dict):
            request.pop("file_uri", None)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
