#!/usr/bin/env python3
"""Cache target-independent boundary-node KNN/PCA features for train/validation.

Reads only xyz, faces, and sample_key from each CAD's canonical ver NPZ. The
descriptors characterize a discrete boundary mesh; they are neither an exact
signed-distance function nor features of a uniformly sampled surface.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
ROOT = Path(__file__).resolve().parents[1]
if os.name == "nt" and (ROOT / ".local_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".local_deps"))
import numpy as np
from scipy.spatial import cKDTree

FEATURE_NAMES = [
    "nearest_boundary_offset_x_over_0.05", "nearest_boundary_offset_y_over_0.05",
    "nearest_boundary_offset_z_over_0.05", "nearest_boundary_distance_over_0.05",
    "neighbor_cov_xx_over_trace", "neighbor_cov_yy_over_trace", "neighbor_cov_zz_over_trace",
    "neighbor_cov_xy_over_trace", "neighbor_cov_xz_over_trace", "neighbor_cov_yz_over_trace",
    "neighbor_eigen_min_over_trace", "neighbor_eigen_mid_over_trace", "kth_boundary_distance_over_0.05",
]
SCHEMA = {
    "version": 1, "feature_names": FEATURE_NAMES, "features": "float32[N,13] in unchanged canonical FEM-node order",
    "K": 32, "geometry_scale": 0.05,
    "coordinate_space": "Existing per-CAD isotropic normalized xyz, preserving aspect ratio",
    "boundary_definition": "Unique global FEM node indices referenced by the real reconstructed TETRA10 boundary faces; quadratic midside nodes retained",
    "nearest_offset": "nearest_boundary_node_xyz - query_xyz, divided by 0.05",
    "covariance": "Population covariance of 32 nearest boundary-node coordinates, centered on their neighborhood mean; components [xx,yy,zz,xy,xz,yz] divided by trace",
    "eigenvalues": "Ascending eigenvalues of the same covariance; store min/trace and middle/trace",
    "neighbors": "32 includes the query node itself when it is a boundary node; tree query workers=1",
    "xyz_hash": "SHA256 of contiguous little-endian float32 xyz raw bytes, without NPY header; node count/shape recorded separately",
    "data_access": ["xyz", "faces", "sample_key"], "target_fit": False,
    "limitations": "Discrete KNN/PCA depends on mesh-node density. Distance is to boundary nodes, not the continuous triangle surface; no sign or exact SDF is inferred. No uniform surface sampling is used.",
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_one(task: tuple[str, str, str, str, int]) -> dict:
    cad, group, source_s, output_s, chunk_size = task
    started = time.monotonic()
    source, output = Path(source_s), Path(output_s)
    target = output / f"{cad}.npz"
    # np.load's ZIP directory is inspected, but no target arrays are accessed.
    with np.load(source, allow_pickle=False) as stored:
        xyz = np.ascontiguousarray(stored["xyz"], dtype="<f4")
        faces = stored["faces"]
        key = str(stored["sample_key"])
    if key != f"{cad}/ver":
        raise ValueError(f"{cad}: expected canonical ver, found {key}")
    n = len(xyz)
    if xyz.shape != (n, 3) or n == 0 or not np.isfinite(xyz).all():
        raise ValueError(f"{cad}: invalid finite xyz[N,3]")
    if xyz.min() < -1e-6 or xyz.max() > 1 + 1e-6:
        raise ValueError(f"{cad}: xyz is outside the recorded normalized coordinate range")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError(f"{cad}: empty or invalid integer boundary faces[M,3]")
    if faces.min() < 0 or faces.max() >= n:
        raise ValueError(f"{cad}: boundary face has an invalid global node index")
    boundary_ids = np.unique(faces.reshape(-1))
    if len(boundary_ids) < SCHEMA["K"]:
        raise ValueError(f"{cad}: fewer than {SCHEMA['K']} real boundary nodes")
    xyz_hash = hashlib.sha256(xyz.tobytes(order="C")).hexdigest()
    cached = False
    if target.exists():
        with np.load(target, allow_pickle=False) as stored:
            cached_features = stored["features"]
            valid = (str(stored["xyz_sha256"]) == xyz_hash and int(stored["node_count"]) == n
                     and str(stored["canonical_case"]) == key and int(stored["schema_version"]) == SCHEMA["version"]
                     and int(stored["K"]) == 32 and float(stored["geometry_scale"]) == 0.05
                     and stored["feature_names"].tolist() == FEATURE_NAMES and cached_features.shape == (n, 13)
                     and cached_features.dtype == np.float32 and np.isfinite(cached_features).all())
        if not valid:
            raise FileExistsError(f"{cad}: refusing to overwrite incompatible/nonfinite cache")
        cached = True
    else:
        boundary_xyz = xyz[boundary_ids].astype(np.float64)
        tree = cKDTree(boundary_xyz)
        features = np.empty((n, 13), dtype=np.float32)
        for first in range(0, n, chunk_size):
            last = min(n, first + chunk_size)
            query = xyz[first:last].astype(np.float64)
            distance, indices = tree.query(query, k=SCHEMA["K"], workers=1)
            neighborhood = boundary_xyz[indices]
            centered = neighborhood - neighborhood.mean(axis=1, keepdims=True)
            covariance = np.einsum("nki,nkj->nij", centered, centered, optimize=True) / SCHEMA["K"]
            trace = np.trace(covariance, axis1=1, axis2=2)
            if not np.isfinite(trace).all() or np.any(trace <= 0):
                raise ValueError(f"{cad}: zero/nonfinite boundary-neighborhood covariance trace")
            block = np.empty((last-first, 13), dtype=np.float64)
            block[:, :3] = (neighborhood[:, 0, :] - query) / SCHEMA["geometry_scale"]
            block[:, 3] = distance[:, 0] / SCHEMA["geometry_scale"]
            block[:, 4:10] = np.column_stack([covariance[:, 0, 0], covariance[:, 1, 1], covariance[:, 2, 2],
                                            covariance[:, 0, 1], covariance[:, 0, 2], covariance[:, 1, 2]]) / trace[:, None]
            eigenvalues = np.linalg.eigvalsh(covariance)
            block[:, 10:12] = eigenvalues[:, :2] / trace[:, None]
            block[:, 12] = distance[:, -1] / SCHEMA["geometry_scale"]
            if not np.isfinite(block).all():
                raise ValueError(f"{cad}: nonfinite local descriptors in nodes {first}:{last}")
            features[first:last] = block
        if features.shape != (n, 13) or not np.isfinite(features).all():
            raise ValueError(f"{cad}: cache feature shape/finite check failed")
        fd, temp_s = tempfile.mkstemp(prefix=cad + ".", suffix=".npz", dir=output)
        os.close(fd)
        temporary = Path(temp_s)
        try:
            np.savez(temporary, features=features, xyz_sha256=np.asarray(xyz_hash), node_count=np.asarray(n),
                     canonical_case=np.asarray(key), cad_id=np.asarray(cad), feature_names=np.asarray(FEATURE_NAMES),
                     K=np.asarray(SCHEMA["K"]), geometry_scale=np.asarray(SCHEMA["geometry_scale"]),
                     schema_version=np.asarray(SCHEMA["version"]), boundary_node_count=np.asarray(len(boundary_ids)),
                     schema_json=np.asarray(json.dumps(SCHEMA)), source_canonical_npz=np.asarray(source_s),
                     target_independent=np.asarray(True))
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return {"cad_id": cad, "split_group": group, "canonical_case": key,
            "source_canonical_npz": str(source), "npz": str(target), "node_count": n,
            "features_shape": [n, 13], "features_dtype": "float32", "finite": True,
            "xyz_sha256": xyz_hash, "surface_triangles": len(faces), "boundary_nodes": len(boundary_ids),
            "elapsed_seconds": time.monotonic() - started, "reused_existing_cache": cached,
            "targets_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / "notes/geobc_split.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/geobc_local_geometry")
    parser.add_argument("--ready", type=Path, default=ROOT / "notes/geobc_local_geometry_ready.json")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    groups = {k: split["groups"][k] for k in ("train", "validation")}
    if len(groups["train"]) != 40 or len(groups["validation"]) != 5 or set(groups["train"]) & set(groups["validation"]):
        raise ValueError("Expected 40/5 disjoint train/validation CAD groups")
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [(cad, group, str(ROOT / split["case_paths"][f"{cad}/ver"]), str(args.output), args.chunk_size)
             for group, cads in groups.items() for cad in cads]
    report = {"status": "preparing", "started_at_utc": utc(), "prepared_groups": groups,
              "source_split": str(args.split), "schema": SCHEMA, "workers": args.workers,
              "BLAS_threads_per_worker": 1, "tree_query_workers": 1, "chunk_size": args.chunk_size,
              "total_CADs": len(tasks), "completed_CADs": 0, "cases": [],
              "scope": "Train40 + validation5 only; canonical ver geometry only; final-test fields and dia not accessed",
              "reuse_contract": "Cache features align with canonical xyz raw-byte hash; all loads of the same CAD may reuse them only after confirming identical xyz/node order",
              "target_fit": False, "label_arrays_accessed": False}
    started = time.monotonic()
    write_json(args.ready, report)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(build_one, task): task[0] for task in tasks}
            for future in as_completed(futures):
                record = future.result()
                report["cases"].append(record)
                report["completed_CADs"] = len(report["cases"])
                elapsed = time.monotonic() - started
                report["elapsed_seconds"] = elapsed
                report["estimated_remaining_seconds"] = elapsed * (len(tasks) - len(report["cases"])) / len(report["cases"])
                write_json(args.ready, report)
                print(json.dumps({"completed_CADs": len(report["cases"]), "total_CADs": len(tasks),
                                  "last_CAD": record["cad_id"], "elapsed_seconds": elapsed,
                                  "estimated_remaining_seconds": report["estimated_remaining_seconds"]}), flush=True)
        if report["completed_CADs"] != 45 or not all(x["finite"] and x["features_shape"] == [x["node_count"], 13] for x in report["cases"]):
            raise ValueError("Not all 45 finite feature caches were completed")
        report["cases"].sort(key=lambda x: (x["split_group"], x["cad_id"]))
        report.update({"status": "ready", "completed_at_utc": utc(), "elapsed_seconds": time.monotonic() - started,
                       "nodes_total": sum(x["node_count"] for x in report["cases"]),
                       "features_data_bytes": sum(x["node_count"] * 13 * 4 for x in report["cases"]),
                       "estimated_remaining_seconds": 0})
        write_json(args.output / "manifest.json", report)
        write_json(args.ready, report)
        print(json.dumps({"status": "ready", "CADs": len(tasks), "nodes_total": report["nodes_total"],
                          "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    except Exception as exc:
        report.update({"status": "failed", "error": str(exc), "elapsed_seconds": time.monotonic() - started})
        write_json(args.ready, report)
        raise


if __name__ == "__main__":
    main()
