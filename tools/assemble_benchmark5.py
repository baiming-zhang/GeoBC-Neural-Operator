#!/usr/bin/env python3
"""Assemble the fixed five-case benchmark from aligned original DeepJEB inputs.

The first two cases come from --exact and the remaining three from --pool.
Dry-run checks manifests, scalar NPZ metadata, coordinate array headers, and
existing destination hashes, without creating or changing any file.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if os.name == "nt" and (ROOT / ".local_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".local_deps"))
import numpy as np

SELECTION = ["506_533/ver", "506_25/tor", "10_30/ver", "10_421/hor", "10_460/dia"]
ALIGNMENT = "official CSV GRID coordinates -> H5 vertices"


def reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("label_alignment") != ALIGNMENT:
        raise ValueError(f"{directory}: manifest does not declare verified CSV/H5 coordinate alignment")
    return manifest


def npz_metadata(path: Path, expected_key: str) -> int:
    with np.load(path, allow_pickle=False) as data:
        for key in ["sample_key", "label_alignment", "xyz"]:
            if key not in data.files:
                raise ValueError(f"{path}: missing NPZ field {key}")
        if data["sample_key"].shape != () or str(data["sample_key"].item()) != expected_key:
            raise ValueError(f"{path}: sample_key disagrees with requested {expected_key}")
        if data["label_alignment"].shape != () or str(data["label_alignment"].item()) != ALIGNMENT:
            raise ValueError(f"{path}: missing or incorrect node-field alignment marker")
    # Read only the coordinate .npy header to establish the actual node count.
    with zipfile.ZipFile(path) as archive, archive.open("xyz.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"{path}: unsupported coordinate NPY header version {version}")
    if len(shape) != 2 or shape[1] != 3 or shape[0] <= 0 or dtype.kind != "f":
        raise ValueError(f"{path}: expected floating point xyz with shape [N,3]")
    return int(shape[0])


def plan(exact: Path, pool: Path, output: Path) -> tuple[dict, list[dict]]:
    if output.resolve() in {exact.resolve(), pool.resolve()}:
        raise ValueError("Assembly output must differ from both preserved input directories")
    manifests = [read_manifest(exact), read_manifest(pool)]
    for key in ["dataset", "source_revision", "feature_names", "feature_schema_note",
                "normalization", "target_policy", "label_alignment"]:
        if manifests[0].get(key) != manifests[1].get(key):
            raise ValueError(f"Input manifests disagree on {key}")
    if len(manifests[0].get("feature_names", [])) != 25:
        raise ValueError("Expected the shared 25-channel conditioning schema")
    report = deepcopy(manifests[0])
    report.update(selection=list(SELECTION), cases=[],
                  selection_policy="Slide-exact two CAD/load cases plus three fixed original-archive extension cases; no claim that this is a historical reference five-case split.",
                  preservation="Source NPZs are copied byte-for-byte. Existing destination NPZs with different SHA256 are never overwritten.",
                  assembly_sources=[{"directory": reference(exact), "manifest": reference(exact / "manifest.json")},
                                    {"directory": reference(pool), "manifest": reference(pool / "manifest.json")}],
                  assembly_tool="tools/assemble_benchmark5.py")
    operations = []
    for number, key in enumerate(SELECTION):
        directory, manifest = (exact, manifests[0]) if number < 2 else (pool, manifests[1])
        matching = [item for item in manifest["cases"] if item.get("sample_key") == key]
        if len(matching) != 1:
            raise ValueError(f"{directory}: expected exactly one manifest record for {key}")
        original = matching[0]
        filename = key.replace("/", "_") + ".npz"
        source, target = directory / filename, output / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing source NPZ {source}")
        nodes = npz_metadata(source, key)
        if original.get("nodes") != nodes:
            raise ValueError(f"{key}: manifest node count differs from the actual NPZ coordinate shape")
        source_hash = sha256(source)
        if target.exists():
            if not target.is_file() or sha256(target) != source_hash:
                raise FileExistsError(f"{target}: existing NPZ differs from the selected source SHA256; assembly will not overwrite it. Use a new --output directory.")
            action = "skip_identical"
        else:
            action = "copy"
        record = deepcopy(original)
        record.update(npz=reference(target), nodes=nodes, npz_sha256=source_hash,
                      copied_from=reference(source),
                      assembly_source_manifest=reference(directory / "manifest.json"),
                      assembly_source_record=deepcopy(original))
        report["cases"].append(record)
        operations.append({"sample_key": key, "source": source, "target": target,
                           "sha256": source_hash, "nodes": nodes, "action": action})
    report["nodes_total"] = sum(item["nodes"] for item in report["cases"])
    report["load_counts"] = {load: sum(key.endswith("/" + load) for key in SELECTION)
                             for load in ["ver", "hor", "dia", "tor"]}
    return report, operations


def execute(report: dict, operations: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for item in operations:
        source, target = item["source"], item["target"]
        if target.exists():
            if not target.is_file() or sha256(target) != item["sha256"]:
                raise FileExistsError(f"{target}: destination changed since preflight; refusing to overwrite")
            continue
        # Exclusive creation also prevents a concurrently created destination
        # from being overwritten. No source NPZ or source manifest is edited.
        created = False
        try:
            with source.open("rb") as reader, target.open("xb") as writer:
                created = True
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
            if sha256(target) != item["sha256"]:
                raise ValueError(f"{target}: copied file failed SHA256 verification")
        except BaseException:
            if created:
                target.unlink(missing_ok=True)
            raise
    manifest_path = output / "manifest.json"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output,
                                     prefix="manifest.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", type=Path, default=ROOT / "data/prepared_exact_multifield")
    parser.add_argument("--pool", type=Path, default=ROOT / "data/prepared50_multifield")
    parser.add_argument("--output", type=Path, default=ROOT / "data/prepared_benchmark5")
    parser.add_argument("--dry-run", action="store_true", help="Validate the complete plan without writing files or directories")
    args = parser.parse_args()
    try:
        report, operations = plan(args.exact, args.pool, args.output)
        for item in operations:
            print(json.dumps({"sample_key": item["sample_key"], "nodes": item["nodes"],
                              "action": item["action"], "source": reference(item["source"]),
                              "target": reference(item["target"])}), flush=True)
        print(json.dumps({"dry_run": args.dry_run, "nodes_total": report["nodes_total"],
                          "cases": len(operations), "load_counts": report["load_counts"]}), flush=True)
        if not args.dry_run:
            execute(report, operations, args.output)
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"Assembly stopped before manifest publication: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
