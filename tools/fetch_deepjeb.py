#!/usr/bin/env python3
"""Fetch a small, resumable prefix of the original official DeepJEB release.

This deliberately uses KAIST-SmartDesignLab/DeepJEB, never DeepJEB-PP.
The official release is a gzip tar archive; arbitrary late cases require scanning
earlier compressed bytes. The default takes the first five complete H5 members.
Cached compressed bytes let interrupted runs resume through HTTP Range requests.
No external Python packages are required.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import time
import urllib.error
import urllib.request

REPO = "KAIST-SmartDesignLab/DeepJEB"
REVISION = "160424ca2a5095315ff9ead8a5fc36d2667fd3c6"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
SOURCE = f"https://huggingface.co/datasets/{REPO}"
PAPER = "https://arxiv.org/html/2406.09047v2"
BLOCK = 128 * 1024


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def dump_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


class CachedHTTPStream(io.RawIOBase):
    """Read existing prefix, append remote suffix, and stop at a bounded budget."""

    def __init__(self, url: str, cache: Path, limit: int, retries: int = 4):
        self.url, self.cache, self.limit, self.retries = url, cache, limit, retries
        cache.parent.mkdir(parents=True, exist_ok=True)
        self.file = cache.open("r+b" if cache.exists() else "w+b")
        self.cached = cache.stat().st_size
        self.position = 0
        self.response = None
        self.downloaded = 0
        self.attempts = 0

    def readable(self) -> bool:
        return True

    def _open(self) -> None:
        headers = {"User-Agent": "GeoBC-DeepJEB-small-sample/1.0", "Accept-Encoding": "identity"}
        if self.cached:
            headers["Range"] = f"bytes={self.cached}-"
        self.response = urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=45)
        if self.cached:
            content_range = self.response.headers.get("Content-Range", "")
            if self.response.status != 206 or not content_range.startswith(f"bytes {self.cached}-"):
                self.response.close()
                self.response = None
                raise RuntimeError("Server ignored requested byte range; refusing to redownload the archive.")

    def read(self, size: int = -1) -> bytes:
        size = BLOCK if size < 0 else min(size, BLOCK)
        if size == 0:
            return b""
        if self.position < self.cached:
            self.file.seek(self.position)
            value = self.file.read(min(size, self.cached - self.position))
            self.position += len(value)
            return value
        if self.cached >= self.limit:
            raise RuntimeError(f"Compressed prefix reached {self.limit / 2**20:.0f} MiB budget. "
                               "Increase --max-download-mib to continue the cached stream.")
        while True:
            try:
                if self.response is None:
                    self._open()
                value = self.response.read(min(size, self.limit - self.cached))
                break
            except (OSError, urllib.error.URLError) as exc:
                self.attempts += 1
                if self.response:
                    self.response.close()
                self.response = None
                if self.attempts > self.retries:
                    raise RuntimeError(f"HTTP stream failed after {self.retries} retries: {exc}") from exc
                time.sleep(min(2 ** self.attempts, 8))
        if value:
            self.file.seek(self.cached)
            self.file.write(value)
            self.file.flush()
            self.cached += len(value)
            self.position += len(value)
            self.downloaded += len(value)
        return value

    def close(self) -> None:
        if self.response:
            self.response.close()
        self.file.close()
        super().close()


def safe_destination(root: Path, member: tarfile.TarInfo, component: str) -> Path:
    name = PurePosixPath(member.name)
    if (name.is_absolute() or ".." in name.parts or not name.parts or name.parts[0] != component
            or "\\" in member.name or ":" in member.name):
        raise RuntimeError(f"Unexpected archive path: {member.name}")
    if member.issym() or member.islnk():
        raise RuntimeError(f"Links are not accepted from archive: {member.name}")
    return root.joinpath(*name.parts)


def save_member(tar: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    source = tar.extractfile(member)
    if source is None:
        raise RuntimeError(f"No data for {member.name}")
    with temporary.open("wb") as output:
        while value := source.read(BLOCK):
            output.write(value)
            digest.update(value)
            size += len(value)
    if size != member.size:
        raise RuntimeError(f"Incomplete member {member.name}: {size} / {member.size}")
    temporary.replace(target)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--cases", nargs="*", help="Exact original case IDs. Late IDs may exceed the bounded prefix budget.")
    parser.add_argument("--max-download-mib", type=int, default=512,
                        help="Maximum compressed prefix per archive, including bytes cached from earlier runs.")
    parser.add_argument("--components", nargs="+", default=["FieldMesh", "Scalar", "Metadata"],
                        choices=["FieldMesh", "Field", "SurfaceMesh", "VolumeMesh", "SolverDeck", "Scalar", "Metadata", "BRep", "Image"])
    args = parser.parse_args()
    if args.count <= 0 or args.max_download_mib <= 0:
        parser.error("--count and --max-download-mib must be positive")
    if args.cases and any(not re.fullmatch(r"[A-Za-z0-9_-]+", case) for case in args.cases):
        parser.error("Case IDs may contain letters, digits, underscores, and hyphens only")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    provenance_path = root / "deepjeb_provenance.json"
    if provenance_path.exists():
        state = json.loads(provenance_path.read_text(encoding="utf-8"))
        if state.get("repository") != REPO or state.get("revision") != REVISION:
            raise RuntimeError("Existing provenance belongs to a different dataset or revision.")
    else:
        state = {"dataset": "DeepJEB original v1.0", "repository": REPO, "revision": REVISION,
                 "source_url": SOURCE, "paper_url": PAPER, "license": "ODC-By v1.0",
                 "created_at": now(), "files": {}, "archives": {}, "selected_cases": [],
                 "selection_note": "First complete H5 members in official archive order; cases differ from slide illustrations.",
                 "units": {"coordinates": "mm", "displacement": "mm", "source_signed_stress": "MPa"},
                 "stress_semantics": "Raw source stress(MPa), retaining sign. Paper calls it signed von Mises stress; source field does not expose a stress tensor. Do not take abs or reconstruct a different target.",
                 "load_conditions_paper": {"ver": "+z 35.6 kN", "hor": "+x 37.8 kN",
                                            "dia": "42.3 kN, 42 degrees from vertical", "tor": "-z 565 kN mm"}}
    wanted = set(args.cases or [])
    state["units"].pop("signed_von_mises_stress", None)
    state["units"]["source_signed_stress"] = "MPa"
    state["stress_semantics"] = "Raw source stress(MPa), retaining sign. Paper calls it signed von Mises stress; source field does not expose a stress tensor. Do not take abs or reconstruct a different target."
    if wanted:
        state["selection_note"] = "Explicit original DeepJEB case IDs; gzip stream scanned within user-selected budget."
    state["request"] = {"cases": args.cases, "count": args.count, "components": args.components,
                        "max_download_mib": args.max_download_mib}
    state["status"] = "running"
    state["updated_at"] = now()
    dump_json(provenance_path, state)
    start = time.monotonic()
    try:
        for component in args.components:
            archive_name = component + ".tar.gz"
            previous = state["archives"].get(archive_name, {})
            archive_files = {name: record for name, record in state["files"].items() if record["archive"] == archive_name}
            files_present = bool(archive_files) and all(
                root.joinpath(*PurePosixPath(name).parts).is_file()
                and root.joinpath(*PurePosixPath(name).parts).stat().st_size == record["size"]
                for name, record in archive_files.items())
            if (previous.get("complete_selection") and files_present
                    and previous.get("request_cases") == sorted(wanted) and previous.get("request_count") == args.count):
                print(f"Already complete: {archive_name}", flush=True)
                continue
            stream = CachedHTTPStream(f"{BASE}/{archive_name}", root / ".download_cache" / f"{REVISION[:12]}_{archive_name}.part",
                                      args.max_download_mib * 2**20)
            selected = []
            archive_state = {"source_url": stream.url, "request_cases": sorted(wanted), "request_count": args.count,
                             "complete_selection": False, "started_at": now()}
            state["archives"][archive_name] = archive_state
            try:
                with tarfile.open(fileobj=stream, mode="r|gz") as tar:
                    for member in tar:
                        if not member.isfile():
                            continue
                        target = safe_destination(root, member, component)
                        case = target.stem
                        small_component = component in {"Scalar", "Metadata"}
                        choose = small_component or (case in wanted if wanted else len(selected) < args.count)
                        if not choose:
                            continue
                        if not small_component:
                            selected.append(case)
                        record = state["files"].get(member.name)
                        if record and target.exists() and target.stat().st_size == record["size"]:
                            print(f"Cached extracted member: {member.name}", flush=True)
                        else:
                            print(f"Extracting {member.name} ({member.size / 2**20:.1f} MiB)", flush=True)
                            digest = save_member(tar, member, target)
                            state["files"][member.name] = {"size": member.size, "sha256": digest,
                                                          "archive": archive_name, "extracted_at": now()}
                        if component == "FieldMesh":
                            state["selected_cases"] = selected.copy()
                        archive_state.update({"cached_compressed_bytes": stream.cached,
                                              "downloaded_bytes_this_run": stream.downloaded, "selected_cases": selected.copy()})
                        state["updated_at"] = now()
                        dump_json(provenance_path, state)
                        if not small_component and len(selected) >= (len(wanted) if wanted else args.count):
                            break
                    if not small_component and len(selected) < (len(wanted) if wanted else args.count):
                        raise RuntimeError(f"Archive ended before enough requested cases were found: {selected}")
                archive_state["complete_selection"] = True
            finally:
                archive_state.update({"cached_compressed_bytes": stream.cached,
                                      "downloaded_bytes_this_run": stream.downloaded, "finished_at": now()})
                stream.close()
                dump_json(provenance_path, state)
        license_path = root / "LICENSE.txt"
        if not license_path.exists():
            with urllib.request.urlopen(f"{BASE}/LICENSE.txt", timeout=45) as response:
                license_path.write_bytes(response.read(128 * 1024))
        state["status"] = "complete"
        state["elapsed_seconds_this_run"] = round(time.monotonic() - start, 2)
        state["updated_at"] = now()
        dump_json(provenance_path, state)
        print(f"Complete in {time.monotonic() - start:.1f}s. Cases: {state['selected_cases']}", flush=True)
        print(f"Provenance: {provenance_path}", flush=True)
    except BaseException as exc:
        state.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                      "last_error": str(exc), "updated_at": now(),
                      "elapsed_seconds_this_run": round(time.monotonic() - start, 2)})
        dump_json(provenance_path, state)
        raise


if __name__ == "__main__":
    main()
