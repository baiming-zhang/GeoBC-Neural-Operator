"""Shared original-node data and a target-free point-to-grid FNO frontend.

This is a solver-selection adapter, not the historical TETRA10 volume grid.
All six solvers use the same TRAIN-only four-field scaler and case schedule.
No final-test CAD or dia archive is opened. Grid values contain inputs only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .benchmark import FieldScaler, field_metrics
from .conditioned_field import load_geometry_cloud

ROOT = Path(__file__).resolve().parents[1]
ALIGNMENT = "official CSV GRID coordinates -> H5 vertices"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def _selected(plan):
    split_path = _path(plan["existing_split"])
    split = _json(split_path)
    groups = split["groups"]
    if [len(groups[k]) for k in ("train", "validation", "test")] != [40, 5, 5]:
        raise ValueError("Expected existing 40/5/5 whole-CAD split")
    if any(set(groups[a]) & set(groups[b]) for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("CAD groups overlap")
    loads = plan["recipe"]["train_loads"]
    if loads != ["ver", "hor", "tor"]:
        raise ValueError("This screen keeps dia reserved")
    keys = {group: [f"{cad}/{load}" for cad in groups[group] for load in loads]
            for group in ("train", "validation")}
    paths = {key: _path(split["case_paths"][key]).resolve()
             for group in keys.values() for key in group}
    base = _path(plan["prepared_data"]).resolve()
    if any(not path.is_relative_to(base) or not path.is_file() for path in paths.values()):
        raise ValueError("Selected prepared input missing or outside dataset directory")
    return split_path, split, keys, paths


def cache_contract(plan):
    """Small metadata only; no array, model, CUDA initialization, or test read."""
    split_path, _, keys, paths = _selected(plan)
    return {"schema": "geobc_solver_selection_pointgrid_v1",
            "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "train_keys": keys["train"], "validation_keys": keys["validation"],
            "cloud_seed": plan["recipe"].get("cloud_seed", plan["recipe"]["seed"]),
            "grid_resolution": plan["recipe"]["grid_resolution"],
            "cloud_points": plan["recipe"]["cloud_points"],
            "stress_transform": plan["recipe"]["stress_transform"],
            "sources": [{"key": key, "path": str(path), "bytes": path.stat().st_size,
                         "mtime_ns": path.stat().st_mtime_ns} for key, path in paths.items()]}


def read_cache_metadata(plan):
    manifest = _json(_path(plan["data_cache"]) / "manifest.json")
    if manifest.get("status") != "completed" or manifest.get("contract") != cache_contract(plan):
        raise ValueError("Shared cache is incomplete or differs from requested data recipe")
    return manifest


def query_grid(prediction, case):
    """Query encoded channels on original nodes, before physical decoding."""
    if prediction.ndim != 5 or prediction.shape[0] != 1 or prediction.shape[1] != 4:
        raise ValueError("Expected [1,4,R,R,R] encoded grid prediction")
    flat = prediction[0].flatten(1).T
    return (flat[case["query_indices"]] * case["query_weights"][..., None]).sum(1)


def load_case(record, device, for_fno=False):
    def tensor(name, dtype=torch.float32, batch=False):
        # Materialize a writable host array: torch must not mutate read-only mmap.
        value = np.array(np.load(record[name + "_path"], mmap_mode="r", allow_pickle=False), copy=True)
        result = torch.as_tensor(value, dtype=dtype, device=device)
        return result.unsqueeze(0) if batch else result
    result = {"key": record["key"], "xyz": tensor("xyz"), "features": tensor("features"),
              "target": tensor("target"), "physical": np.load(record["physical_path"], mmap_mode="r"),
              "cloud": tensor("cloud", batch=True), "mask": tensor("mask", torch.bool, batch=True)}
    if for_fno:
        result.update(grid=tensor("grid", batch=True),
                      query_indices=tensor("query_indices", torch.int64),
                      query_weights=tensor("query_weights"))
    return result


def _stencil(xyz, resolution):
    if not torch.isfinite(xyz).all() or xyz.min() < -2e-6 or xyz.max() > 1 + 2e-6:
        raise ValueError("Geometry outside normalized cube")
    # Only normalized-coordinate rounding tolerance is clipped, never labels.
    position = xyz.clamp(0, 1) * (resolution - 1)
    low = position.floor().to(torch.int64).clamp(max=resolution - 2)
    frac = position - low
    corners = torch.tensor([[i & 1, (i >> 1) & 1, (i >> 2) & 1]
                            for i in range(8)], device=xyz.device)
    nodes = low[:, None, :] + corners
    indices = nodes[..., 0] * resolution**2 + nodes[..., 1] * resolution + nodes[..., 2]
    weights = torch.where(corners.bool(), frac[:, None, :], 1-frac[:, None, :]).prod(-1)
    if not torch.allclose(weights.sum(1), torch.ones(len(xyz), device=xyz.device), atol=2e-6, rtol=0):
        raise ValueError("Trilinear stencil does not form a partition of unity")
    return indices, weights


def _splat(values, indices, weights, resolution):
    size = resolution**3
    sums = torch.zeros((size, values.shape[1]), dtype=torch.float32, device=values.device)
    counts = torch.zeros(size, dtype=torch.float32, device=values.device)
    for corner in range(8):
        count = weights[:, corner]
        counts.index_add_(0, indices[:, corner], count)
        sums.index_add_(0, indices[:, corner], values.float() * count[:, None])
    active = counts > 0
    sums[active] /= counts[active, None]
    return sums, active


def prepare_data(plan):
    began = time.monotonic()
    cache = _path(plan["data_cache"])
    if (cache / "manifest.json").is_file():
        return read_cache_metadata(plan)
    contract = cache_contract(plan)
    _, split, keys, paths = _selected(plan)
    cache.mkdir(parents=True, exist_ok=True)
    status = {"phase": "preparing", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "total_cases": len(paths), "prepared_cases": 0, "grid_cases": 0}
    def update(phase, **extra):
        elapsed = time.monotonic()-began
        status.update(phase=phase, elapsed_seconds=elapsed, **extra)
        count = status.get("grid_cases", 0) if phase == "grids" else status.get("prepared_cases", 0)
        status["eta_seconds"] = elapsed/count*(len(paths)-count) if count else None
        _write(cache / "status.json", status)
    update("inputs")
    records, cad_paths = {}, {}
    for index, (key, path) in enumerate(paths.items()):
        cad = key.split("/")[0]
        group = "train" if key in keys["train"] else "validation"
        with np.load(path, allow_pickle=False) as f:
            if str(f["label_alignment"].item()) != ALIGNMENT or str(f["sample_key"].item()) != key:
                raise ValueError(f"Unverified alignment or case identity: {key}")
            xyz = np.asarray(f["xyz"], np.float32)
            features = np.asarray(f["features"], np.float32)
            physical = np.column_stack((f["displacement"], f["stress"])).astype(np.float32)
        if features.shape != (len(xyz), 25) or physical.shape != (len(xyz), 4):
            raise ValueError(f"Incorrect four-field or condition shape: {key}")
        if not all(np.isfinite(value).all() for value in (xyz, features, physical)):
            raise ValueError(f"Nonfinite prepared inputs/labels: {key}")
        directory = cache / "cases" / key.replace("/", "_")
        record = {"key": key, "cad_id": cad, "group": group, "num_nodes": len(xyz),
                  "label_alignment": ALIGNMENT}
        if cad not in cad_paths:
            geometry = cache / "geometry" / cad
            cloud = load_geometry_cloud(path, cloud_points=plan["recipe"]["cloud_points"],
                                        seed=plan["recipe"].get("cloud_seed", plan["recipe"]["seed"]), cad_key=cad)
            for name, value in (("xyz", xyz), ("cloud", cloud.points), ("mask", cloud.mask)):
                target = geometry / f"{name}.npy"
                _save(target, value)
            cad_paths[cad] = {name+"_path": str((geometry / f"{name}.npy").resolve())
                              for name in ("xyz", "cloud", "mask")}
            _write(geometry / "cloud_metadata.json", cloud.record())
        elif not np.array_equal(xyz, np.load(cad_paths[cad]["xyz_path"], mmap_mode="r")):
            raise ValueError(f"Geometry/node order changed across loads: {cad}")
        record.update(cad_paths[cad])
        for name, value in (("features", features), ("physical", physical)):
            target = directory / f"{name}.npy"
            _save(target, value)
            record[name+"_path"] = str(target.resolve())
        record["target_path"] = str((directory / "target.npy").resolve())
        record["grid_path"] = str((directory / "grid.npy").resolve())
        geometry = cache / "geometry" / cad
        for name in ("query_indices", "query_weights"):
            record[name+"_path"] = str((geometry / f"{name}.npy").resolve())
        records[key] = record
        update("inputs", prepared_cases=index+1, current_key=key)
    update("scaler")
    train_targets = [np.load(records[key]["physical_path"], mmap_mode="r") for key in keys["train"]]
    scaler = FieldScaler.fit(train_targets, [0, 1, 2, 3], plan["recipe"]["stress_transform"])
    del train_targets
    _write(cache / "scaler.json", {**asdict(scaler), "fit": "TRAIN120 original nodes only"})
    for record in records.values():
        target = scaler.encode(np.load(record["physical_path"], mmap_mode="r"))
        _save(record["target_path"], target)
    # One actual GPU rasterization preparation, not an additional model probe.
    resolution = plan["recipe"]["grid_resolution"]
    device = torch.device(plan["recipe"]["device"])
    axis = torch.linspace(0, 1, resolution, device=device)
    coords = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"))
    stencil_cad, roundtrips = {}, {}
    selected_roundtrip = set(keys["validation"] + plan["recipe"]["training_readout_keys"])
    grid_began = time.monotonic()
    for index, (key, record) in enumerate(records.items()):
        xyz = torch.as_tensor(np.array(np.load(record["xyz_path"], mmap_mode="r")), device=device)
        features = torch.as_tensor(np.array(np.load(record["features_path"], mmap_mode="r")), device=device)
        if record["cad_id"] not in stencil_cad:
            indices, weights = _stencil(xyz, resolution)
            _save(record["query_indices_path"], indices.to(torch.int32).cpu().numpy())
            _save(record["query_weights_path"], weights.cpu().numpy())
            stencil_cad[record["cad_id"]] = True
        else:
            indices = torch.as_tensor(np.array(np.load(record["query_indices_path"], mmap_mode="r")),
                                      dtype=torch.int64, device=device)
            weights = torch.as_tensor(np.array(np.load(record["query_weights_path"], mmap_mode="r")), device=device)
        raster, active = _splat(features, indices, weights, resolution)
        grid = torch.cat((active.float().reshape(1,resolution,resolution,resolution), coords,
                          raster.T.reshape(25,resolution,resolution,resolution)), dim=0)
        _save(record["grid_path"], grid.cpu().numpy())
        record.update(grid_occupied_fraction=float(active.float().mean()), query_coverage=1.0,
                      grid_frontend="target-free trilinear node-splat feature means, not TETRA volume occupancy")
        if key in selected_roundtrip:
            encoded = torch.as_tensor(np.array(np.load(record["target_path"], mmap_mode="r")), device=device)
            diagnostic_grid, _ = _splat(encoded, indices, weights, resolution)
            node_encoded = (diagnostic_grid[indices]*weights[...,None]).sum(1).cpu().numpy()
            physical = np.load(record["physical_path"], mmap_mode="r")
            roundtrips[key] = field_metrics(physical, scaler.decode(node_encoded), [0,1,2,3])
        del xyz, features, indices, weights, raster, grid
        update("grids", grid_cases=index+1, current_key=key,
               grid_elapsed_seconds=time.monotonic()-grid_began)
        if (index+1) % 15 == 0:
            print(json.dumps({"phase":"grids", "completed":index+1, "total":len(records),
                              "elapsed_seconds":time.monotonic()-began}), flush=True)
    schedule_rng = random.Random(plan["recipe"]["seed"])
    order = []
    while len(order) < plan["recipe"]["steps"]:
        epoch = list(keys["train"])
        schedule_rng.shuffle(epoch)
        order.extend(epoch)
    order = order[:plan["recipe"]["steps"]]
    manifest = {"status":"completed", "contract":contract, "records":records,
                "train_keys":keys["train"], "validation_keys":keys["validation"],
                "training_readout_keys":plan["recipe"]["training_readout_keys"],
                "scaler":asdict(scaler), "case_order":order,
                "case_order_sha256":hashlib.sha256(json.dumps(order,separators=(",",":")).encode()).hexdigest(),
                "grid_roundtrip":roundtrips,
                "grid_roundtrip_definition":"TRAIN3/VAL15 encode -> label splat -> node query -> decode; diagnostic only, not model supervision or error lower bound",
                "grid_schema":["node_occupancy","x","y","z"]+[f"original_feature_{i}" for i in range(25)],
                "learned_context":"All methods re-encode same geometry cloud each update; global32 never cached as an input array",
                "elapsed_seconds":time.monotonic()-began,
                "completed_at_utc":datetime.now(timezone.utc).isoformat()}
    _write(cache / "manifest.json", manifest)
    update("completed", eta_seconds=0)
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    args=parser.parse_args()
    plan=_json(args.plan)
    if args.metadata_only:
        print(json.dumps(cache_contract(plan)), flush=True)
    else:
        result=prepare_data(plan)
        print(json.dumps({"status":result["status"], "train_cases":len(result["train_keys"]),
                          "validation_cases":len(result["validation_keys"]), "elapsed_seconds":result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
