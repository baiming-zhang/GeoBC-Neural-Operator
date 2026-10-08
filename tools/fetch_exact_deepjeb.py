#!/usr/bin/env python3
"""Verify or fetch the six original DeepJEB files for the slide-exact CADs.

Uses only Python's standard library and public Google Drive download endpoints.
There is no login, credential/cookie extraction, or confirmation-page automation.
Existing files are hashed and skipped only if both size and SHA256 match.
The original acquisition provenance is never changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROVENANCE = ROOT / "data/raw_exact/deepjeb_drive_provenance.json"
DEFAULT_MANIFEST = ROOT / "references/deepjeb_exact_files.json"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/140.0.0.0 Safari/537.36"
BLOCK_SIZE = 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def public_drive_id(source_url: str) -> str:
    parsed = urllib.parse.urlparse(source_url)
    if parsed.scheme != "https" or parsed.netloc != "drive.google.com":
        raise ValueError("Expected an official public https://drive.google.com file URL")
    matched = re.fullmatch(r"/file/d/([A-Za-z0-9_-]+)/view", parsed.path)
    if not matched:
        raise ValueError("Source URL must contain a directly observed Drive file ID")
    return matched.group(1)


def validate_entry(entry: dict) -> PurePosixPath:
    relative = PurePosixPath(entry["relative_path"])
    if (relative.is_absolute() or len(relative.parts) != 2 or ".." in relative.parts
            or relative.parts[0] not in {"FieldMesh", "Field", "SolverDeck"}
            or "\\" in entry["relative_path"] or ":" in entry["relative_path"]):
        raise ValueError(f"Invalid portable component/name: {entry['relative_path']!r}")
    if not isinstance(entry["size_bytes"], int) or entry["size_bytes"] <= 0:
        raise ValueError("Manifest file size must be a positive integer")
    if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
        raise ValueError("Manifest SHA256 must contain 64 lowercase hex characters")
    if public_drive_id(entry["source_url"]) != entry["drive_file_id"]:
        raise ValueError("Manifest Drive ID and official source URL disagree")
    return relative


def manifest_from_provenance(source: Path, target: Path) -> dict:
    original = json.loads(source.read_text(encoding="utf-8"))
    files = []
    for item in original["files"]:
        if Path(item["name"]).name != item["name"] or "/" in item["name"]:
            raise ValueError("Provenance contains an invalid file name")
        entry = {"relative_path": f"{item['component']}/{item['name']}",
                 "drive_file_id": public_drive_id(item["source_url"]),
                 "source_url": item["source_url"], "size_bytes": item["size"],
                 "sha256": item["sha256"]}
        validate_entry(entry)
        files.append(entry)
    if len(files) != 6 or len({item["relative_path"] for item in files}) != 6:
        raise ValueError("Expected exactly six unique slide-exact component files")
    manifest = {"schema_version": 1, "dataset": original["dataset"],
                "source_page": original["source_page"], "source_folder": original["source_folder"],
                "provenance_source_name": source.name,
                "manifest_policy": "Official Drive IDs, sizes and SHA256 copied programmatically from the original local acquisition provenance; no machine-specific paths or temporary connector URLs.",
                "CAD_ids": sorted({PurePosixPath(item["relative_path"]).stem for item in files}),
                "expected_total_bytes": sum(item["size_bytes"] for item in files),
                "files": files}
    if target.resolve() == source.resolve():
        raise ValueError("Portable manifest must not overwrite acquisition provenance")
    write_json(target, manifest)
    return manifest


def verify(path: Path, entry: dict) -> dict:
    actual_size = path.stat().st_size
    if actual_size != entry["size_bytes"]:
        raise ValueError(f"{path}: size mismatch, expected {entry['size_bytes']}, found {actual_size}; existing file is left untouched")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_SIZE), b""):
            digest.update(block)
    actual_hash = digest.hexdigest()
    if actual_hash != entry["sha256"]:
        raise ValueError(f"{path}: SHA256 mismatch; existing file is left untouched")
    return {"verified_size_bytes": actual_size, "verified_sha256": actual_hash}


def looks_like_html(block: bytes) -> bool:
    prefix = block[:1024].lstrip().lower()
    return prefix.startswith((b"<!doctype html", b"<html", b"<head", b"<body"))


def download(path: Path, entry: dict, timeout: float) -> dict:
    query = urllib.parse.urlencode({"export": "download", "id": entry["drive_file_id"]})
    url = "https://drive.google.com/uc?" + query
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # No cookie jar, browser profile, OAuth token, or connector credential.
        with urllib.request.urlopen(request, timeout=timeout) as response:
            final_host = urllib.parse.urlparse(response.geturl()).netloc
            content_type = response.headers.get("Content-Type", "").lower()
            block = response.read(BLOCK_SIZE)
            if (final_host == "accounts.google.com" or "text/html" in content_type
                    or looks_like_html(block)):
                raise RuntimeError(
                    f"Public Drive endpoint returned an HTML/login/confirmation page for {entry['relative_path']}. "
                    "This downloader does not log in, extract cookies, or follow confirmation forms. "
                    f"Open the official file URL {entry['source_url']} and place the downloaded file at {path}, then rerun verification.")
            with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=path.name + ".",
                                             suffix=".part", delete=False) as handle:
                temporary = Path(handle.name)
                count = 0
                while block:
                    count += len(block)
                    if count > entry["size_bytes"]:
                        raise ValueError("Public download exceeded the manifest's expected size")
                    handle.write(block)
                    block = response.read(BLOCK_SIZE)
                handle.flush()
                os.fsync(handle.fileno())
        verified = verify(temporary, entry)
        # Preserve an independently created destination if another process won.
        if path.exists():
            verified = verify(path, entry)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        temporary = None
        return verified
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> int:
    if args.build_manifest_from:
        manifest = manifest_from_provenance(args.build_manifest_from, args.manifest)
    else:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    receipt_path = args.receipt or args.output / "exact_fetch_receipt.json"
    protected = {args.manifest.resolve(), DEFAULT_PROVENANCE.resolve()}
    if args.build_manifest_from:
        protected.add(args.build_manifest_from.resolve())
    if receipt_path.resolve() in protected:
        raise ValueError("Receipt must not overwrite the manifest or acquisition provenance")
    receipt = {"dataset": manifest["dataset"], "manifest": str(args.manifest.resolve()),
               "output": str(args.output.resolve()), "started_at_utc": now(),
               "verify_only": args.verify_only, "network_downloads": 0, "files": []}
    failures = 0
    for entry in manifest["files"]:
        relative = validate_entry(entry)
        path = args.output.joinpath(*relative.parts)
        item = {"relative_path": str(relative), "official_source_url": entry["source_url"]}
        try:
            if path.is_file():
                item.update(status="verified_existing", **verify(path, entry))
            elif args.verify_only:
                raise FileNotFoundError(f"Missing {path}; --verify-only disables all downloads")
            else:
                receipt["network_downloads"] += 1
                item.update(status="downloaded_verified", **download(path, entry, args.timeout))
            print(f"{item['status']}: {relative}", flush=True)
        except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
            item.update(status="failed", error=str(error))
            failures += 1
            print(f"failed: {relative}: {error}", flush=True)
        receipt["files"].append(item)
    receipt.update(completed_at_utc=now(), status="completed" if not failures else "failed",
                   verified_files=len(receipt["files"]) - failures, failed_files=failures)
    write_json(receipt_path, receipt)
    print(f"Receipt: {receipt_path}", flush=True)
    return 1 if failures else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument("--output", type=Path, default=ROOT / "data/raw_exact")
    result.add_argument("--receipt", type=Path, help="Separate receipt; default <output>/exact_fetch_receipt.json")
    result.add_argument("--verify-only", action="store_true", help="Hash existing files; never access the network")
    result.add_argument("--timeout", type=float, default=60, help="Public download request timeout in seconds")
    result.add_argument("--build-manifest-from", type=Path, help="Programmatically copy portable IDs/sizes/SHA256 from the original acquisition provenance")
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
