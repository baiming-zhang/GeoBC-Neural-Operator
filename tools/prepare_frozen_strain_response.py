"""Recover one fixed corner-P1 predicted strain basis for 45 allowed CADs.

Reads only raw H5 vertices/cells and the frozen predicted-u cache. No source
u/stress labels, dia archives, test CADs, model weights or new downloads.
Completed files can be resumed only with an identical metadata/source contract.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.name == "nt" and (ROOT / ".local_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".local_deps"))
import numpy as np

from geobc.displacement_condition import (BASIS_ORDER, file_identity, host_path, json_sha256,
                                          load_response_cache, load_response_manifest, verify_file_identity)
from geobc.local_geometry import xyz_sha256
from geobc.strain_condition import (COMPONENT_ORDER, FEATURE_NAMES, RECOVERY, SCHEMA_VERSION,
                                    load_strain_manifest, recover_corner_p1_strain)

PARENT_SOURCE_COMMIT = "8af33985bdbde5d68a4a4b26f2be7ce9dc0aa2a3"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _preflight(args):
    """JSON/stat only; no torch, h5py, NPZ arrays or model deserialization."""
    directory, raw = host_path(args.u_response_cache), host_path(args.raw_meshes)
    parent = load_response_manifest(directory)
    parent_path = directory / "manifest.json"
    parent_identity = file_identity(parent_path, small_sha=True)
    gate = _json(host_path(parent["gate"]["path"]))
    split = _json(host_path(gate["split"]["path"]))
    cads = split["groups"]["train"] + split["groups"]["validation"]
    if len(cads) != 45 or set(cads) != set(parent["CADs"]) or set(cads) & set(split["groups"]["test"]):
        raise ValueError("Strain recovery scope must be exactly frozen train40+validation5")
    guard = _json(host_path(gate["load_basis"]["path"]))
    coefficient_records = {r["cad_id"]: r for r in guard["cases"]}
    for cad in cads:
        for row, load in enumerate(BASIS_ORDER):
            alpha = coefficient_records[cad]["alpha_by_load"][load]["alpha"]
            if not np.array_equal(np.asarray(alpha, np.float64), np.eye(3)[row]):
                raise ValueError("Common RMS formula requires the existing unit ver/hor/tor reference coefficients")
    geometry = {}
    for cad in cads:
        record = parent["CADs"][cad]
        verify_file_identity(directory / record["file"], record)
        normalization = record["normalization"]
        if normalization.get("status") != "known" or normalization["characteristic_length_mm"] <= 0 or len(normalization["bbox_min_mm"]) != 3:
            raise ValueError("Exact existing isotropic mm normalization metadata is required; do not guess it")
        geometry[cad] = file_identity(raw / f"{cad}.h5")
    code = {"strain_condition.py": _sha(ROOT / "geobc/strain_condition.py"),
            "prepare_frozen_strain_response.py": _sha(Path(__file__))}
    config = {"schema_version": SCHEMA_VERSION, "parent_source_commit": PARENT_SOURCE_COMMIT,
              "parent_u_manifest": parent_identity, "parent_response_fingerprint": parent["teacher_fingerprint"],
              "parent_teacher_fingerprint": parent["teacher_fingerprint"], "groups": split["groups"],
              "split_sha256": gate["split"]["sha256"], "load_basis_sha256": gate["load_basis"]["sha256"],
              "CAD_order": cads, "raw_geometry_sources": geometry, "u_cache_sources": parent["CADs"],
              "recovery": RECOVERY, "basis_units": "raw physical dimensionless tensor strain", "engineering_shear": False,
              "basis_order": list(BASIS_ORDER), "component_order": list(COMPONENT_ORDER),
              "formula": "G=(teacher_scale_mm/characteristic_length_mm)*gradient_normalized_xyz(basis_scaled); epsilon=sym(G)",
              "scale_formula": "sqrt(sum_train40_raw_basis_strain_squared/(train_unique_nodes*3loads*6tensor_components)); center0; no floor/clipping",
              "scale_fit_scope": "train40 predicted ver/hor/tor basis only; validation/dia/test excluded",
              "requested": {"u_response_cache": str(directory), "raw_meshes": str(raw), "output": str(host_path(args.output)),
                            "batch_size": args.batch_size, "device": args.device}, "source_sha256": code,
              "source_members_read": {"raw_H5": ["vertices", "cells"], "u_cache": ["basis_scaled", "identity/normalization metadata"]},
              "GT_field_members_read": [], "labels_or_weights_fit": False,
              "limitations": "Four-corner P1 proxy ignores midside displacement values and curved TETRA10 geometry in gradients, then smears constant cell values onto all10 nodes. Not exact quadratic strain, solver stress recovery or a constitutive scalar claim."}
    config["recovery_fingerprint"] = json_sha256(config)
    return {"config": config, "parent": parent, "split": split, "directory": directory,
            "raw": raw, "CADs": cads, "output": host_path(args.output)}


def read_geometry(path, normalization, expected_xyz_sha256, expected_nodes):
    """Only two H5 geometry datasets; reproduce original float32 xyz node hash."""
    import h5py
    with h5py.File(path, "r") as stored:
        vertices = np.asarray(stored["vertices"], np.float32)
        cells = np.asarray(stored["cells"], np.int64)
    if vertices.shape != (expected_nodes, 3) or not np.isfinite(vertices).all() or cells.ndim != 2 or cells.shape[1] != 10 or not len(cells):
        raise ValueError("Raw H5 geometry no longer matches the original TETRA10 node layout")
    origin = np.asarray(normalization["bbox_min_mm"], np.float32)
    length = float(normalization["characteristic_length_mm"])
    original_xyz = ((vertices - origin) / length).astype(np.float32)
    if xyz_sha256(original_xyz) != expected_xyz_sha256:
        raise ValueError("Raw H5 and parent u-cache normalized xyz/node order differ")
    # Solve using original float32 mm geometry promoted before normalization,
    # rather than the rounded normalized float32 input coordinates.
    coordinates = (vertices.astype(np.float64) - origin.astype(np.float64)) / length
    return coordinates, cells


def _cached_record(path, config, parent_record, raw_source):
    """Read only small cache metadata; arrays were finite-checked before atomic write."""
    with np.load(path, allow_pickle=False) as stored:
        record = json.loads(str(stored["record_json"].item()))
        if (str(stored["recovery_fingerprint"].item()) != config["recovery_fingerprint"]
                or str(stored["parent_response_fingerprint"].item()) != config["parent_response_fingerprint"]
                or str(stored["xyz_sha256"].item()) != parent_record["xyz_sha256"]
                or int(stored["node_count"].item()) != parent_record["node_count"]
                or record["raw_geometry_source"] != raw_source or record["normalization"] != parent_record["normalization"]
                or record["finite"] is not True):
            raise ValueError("Existing partial strain cache differs from the fixed resume/source contract")
    return {"file": path.name, **{k: v for k, v in file_identity(path).items() if k != "path"}, **record}


def generate(args, metadata):
    import torch
    output, config, parent = metadata["output"], metadata["config"], metadata["parent"]
    if (output / "manifest.json").is_file():
        existing = load_strain_manifest(output, expected_parent=config["parent_response_fingerprint"])
        if existing["recovery_fingerprint"] != config["recovery_fingerprint"]:
            raise ValueError("Completed strain cache differs from requested source/recovery contract")
        for cad, record in existing["CADs"].items():
            verify_file_identity(output / f"{cad}.npz", record)
        print(json.dumps({"status": "completed", "reused": True, "output": str(output)}), flush=True)
        return existing
    if output.exists() and any(output.iterdir()):
        if not args.resume or not (output / "config.json").is_file() or _json(output / "config.json") != config:
            raise FileExistsError("Nonempty incomplete output needs --resume with exactly matching source/config contract")
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "config.json").exists():
        _write(output / "config.json", config)
    started = time.monotonic()
    earlier_elapsed = _json(output / "status.json").get("elapsed_seconds", 0) if args.resume and (output / "status.json").exists() else 0
    records = {}
    status = {"phase": "preparing_recovery", "completed_CADs": 0, "total_CADs": 45}
    def update(phase, **extra):
        status.update(extra, phase=phase, elapsed_seconds=earlier_elapsed + time.monotonic() - started,
                      updated_at_utc=datetime.now(timezone.utc).isoformat())
        done = status["completed_CADs"]
        status["eta_seconds"] = status["elapsed_seconds"] / done * (45 - done) if done else None
        _write(output / "status.json", status)
    update("preparing_recovery")
    try:
        if args.device != "cuda" and not args.device.startswith("cuda:"):
            raise ValueError("Formal recovery uses root's CUDA queue; CPU is reserved for synthetic acceptance")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; metadata --dry-run remains usable")
        for cad in metadata["CADs"]:
            update("recovering_corner_P1", current_CAD=cad)
            parent_record = parent["CADs"][cad]
            source = config["raw_geometry_sources"][cad]
            path = output / f"{cad}.npz"
            if path.exists():
                if not args.resume:
                    raise FileExistsError("Existing CAD strain file requires explicit --resume")
                records[cad] = _cached_record(path, config, parent_record, source)
            else:
                verify_file_identity(host_path(source["path"]), source)
                coordinates, cells = read_geometry(host_path(source["path"]), parent_record["normalization"],
                                                    parent_record["xyz_sha256"], parent_record["node_count"])
                cache = load_response_cache(metadata["directory"], cad, manifest=parent,
                                             expected_teacher=config["parent_response_fingerprint"])
                strain, diagnostics = recover_corner_p1_strain(coordinates, cells, cache.basis_scaled,
                    u_scale_mm=cache.scale_mm, characteristic_length_mm=parent_record["normalization"]["characteristic_length_mm"],
                    batch_size=args.batch_size, device=args.device)
                verify_file_identity(host_path(source["path"]), source)
                strain64 = strain.astype(np.float64)
                sum_squared = float(np.einsum("nlc,nlc->", strain64, strain64))
                del strain64
                record = {"cad_id": cad, "node_count": len(strain), "xyz_sha256": parent_record["xyz_sha256"],
                          "parent_response_fingerprint": config["parent_response_fingerprint"],
                          "recovery_fingerprint": config["recovery_fingerprint"], "normalization": parent_record["normalization"],
                          "raw_geometry_source": source, "parent_u_cache_source": file_identity(metadata["directory"] / parent_record["file"]),
                          "sum_squared_raw_basis_strain": sum_squared, "shape": list(strain.shape), "dtype": "float32", "finite": True,
                          "diagnostics": diagnostics, "basis_units": "raw physical dimensionless tensor strain"}
                temporary = path.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez(handle, basis_strain=strain, cad_id=np.asarray(cad), node_count=np.asarray(len(strain)),
                             xyz_sha256=np.asarray(parent_record["xyz_sha256"]), basis_order=np.asarray(BASIS_ORDER),
                             component_order=np.asarray(COMPONENT_ORDER), schema_version=np.asarray(SCHEMA_VERSION),
                             parent_response_fingerprint=np.asarray(config["parent_response_fingerprint"]),
                             recovery_fingerprint=np.asarray(config["recovery_fingerprint"]), record_json=np.asarray(json.dumps(record, sort_keys=True)))
                temporary.replace(path)
                records[cad] = {"file": path.name, **{k: v for k, v in file_identity(path).items() if k != "path"}, **record}
                del cache, strain, coordinates, cells
            status["completed_CADs"] += 1
            _write(output / "progress.json", {"recovery_fingerprint": config["recovery_fingerprint"], "CADs": records})
            update("recovering_corner_P1")
        train = config["groups"]["train"]
        fit_nodes = sum(records[c]["node_count"] for c in train)
        sum_squared = sum(records[c]["sum_squared_raw_basis_strain"] for c in train)
        scale = float(np.sqrt(sum_squared / (fit_nodes * 3 * 6)))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Predicted train strain RMS is nonpositive/nonfinite; no artificial scale or clipping is introduced")
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        verify_file_identity(host_path(config["parent_u_manifest"]["path"]), config["parent_u_manifest"], small_sha=True)
        manifest = {"status": "completed", "complete": True, "schema_version": SCHEMA_VERSION,
                    "parent_source_commit": PARENT_SOURCE_COMMIT, "parent_u_manifest": config["parent_u_manifest"],
                    "parent_response_fingerprint": config["parent_response_fingerprint"],
                    "parent_teacher_fingerprint": config["parent_teacher_fingerprint"], "parent_teacher": parent["teacher"],
                    "recovery_fingerprint": config["recovery_fingerprint"], "source_config": config,
                    "groups": config["groups"], "split_sha256": config["split_sha256"], "load_basis_sha256": config["load_basis_sha256"],
                    "recovery": RECOVERY, "basis_units": config["basis_units"], "engineering_shear": False,
                    "basis_order": list(BASIS_ORDER), "component_order": list(COMPONENT_ORDER), "feature_names": list(FEATURE_NAMES),
                    "scale": scale, "scaler": {"scale": scale, "center": 0., "fit_CADs": train, "fit_loads": list(BASIS_ORDER),
                        "fit_nodes": fit_nodes, "channels": list(COMPONENT_ORDER), "sum_squared_raw_strain": sum_squared,
                        "denominator": fit_nodes * 3 * 6, "definition": config["scale_formula"], "floor": None, "clipping": None,
                        "fit_data": "only train40 frozen predicted physical ver/hor/tor strain; no GT labels or validation statistics"},
                    "CADs": records, "CAD_count": 45, "node_count": sum(r["node_count"] for r in records.values()),
                    "total_bytes": sum(r["bytes"] for r in records.values()), "elapsed_seconds": earlier_elapsed + time.monotonic() - started,
                    "GT_field_members_read": [], "dia_sources_opened": [], "final_test_sources_opened": [],
                    "GPU_tensors_released_before_manifest": True, "teacher_weights_loaded": False,
                    "cost": {"parent_u_cost": parent["cost"], "recovery_elapsed_seconds": earlier_elapsed + time.monotonic() - started},
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(), "limitations": config["limitations"]}
        _write(output / "manifest.json", manifest)
        update("completed", GPU_tensors_released=True)
        print(json.dumps({"status": "completed", "CADs": 45, "nodes": manifest["node_count"], "scale": scale,
                          "elapsed_seconds": manifest["elapsed_seconds"], "cache_bytes": manifest["total_bytes"]}), flush=True)
        return manifest
    except Exception as exc:
        update("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--u-response-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--raw-meshes", type=Path, default=Path("data/raw/FieldMesh"))
    p.add_argument("--batch-size", type=int, default=262144, help="Tetrahedra per FP64 geometry solve/scatter chunk")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="JSON/stat only; no H5/NPZ arrays, torch, weights or CUDA")
    args = p.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    metadata = _preflight(args)
    if args.dry_run:
        print(json.dumps({"status": "metadata_ready", "CADs": 45,
                          "parent_response_fingerprint": metadata["config"]["parent_response_fingerprint"],
                          "raw_geometry_arrays_read": 0, "u_basis_arrays_read": 0, "GT_fields_read": 0,
                          "GPU_used": False}, indent=2))
        return
    generate(args, metadata)


if __name__ == "__main__":
    main()
