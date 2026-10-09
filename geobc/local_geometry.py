"""Strict reader for target-independent, node-aligned local geometry caches.

The 13 channels describe KNN/PCA of discrete boundary FEM nodes. They are not
an exact SDF or features of a uniformly sampled continuous surface. A cache
is shared by every load of a CAD after checking each load's normalized xyz.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np


FEATURE_NAMES = (
    "nearest_boundary_offset_x_over_0.05", "nearest_boundary_offset_y_over_0.05",
    "nearest_boundary_offset_z_over_0.05", "nearest_boundary_distance_over_0.05",
    "neighbor_cov_xx_over_trace", "neighbor_cov_yy_over_trace", "neighbor_cov_zz_over_trace",
    "neighbor_cov_xy_over_trace", "neighbor_cov_xz_over_trace", "neighbor_cov_yz_over_trace",
    "neighbor_eigen_min_over_trace", "neighbor_eigen_mid_over_trace", "kth_boundary_distance_over_0.05",
)
FEATURE_DIM = len(FEATURE_NAMES)


def normalized_xyz_array(xyz) -> np.ndarray:
    """Canonical little-endian float32 xyz, without a NPY header or targets."""
    if hasattr(xyz, "detach"):
        xyz = xyz.detach().cpu().numpy()
    values = np.ascontiguousarray(xyz, dtype="<f4")
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Normalized geometry must be finite nonempty xyz[N,3]")
    if values.min() < -1e-6 or values.max() > 1 + 1e-6:
        raise ValueError("Local geometry uses prepared normalized xyz in [0,1]")
    return values


def xyz_sha256(xyz) -> str:
    return hashlib.sha256(normalized_xyz_array(xyz).tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class LocalGeometryCache:
    features: np.ndarray
    cad_id: str
    node_count: int
    xyz_sha256: str
    path: Path
    canonical_case: str
    source_canonical_npz: str
    boundary_node_count: int
    schema: dict

    def verify_xyz(self, xyz) -> None:
        values = normalized_xyz_array(xyz)
        if len(values) != self.node_count:
            raise ValueError(f"{self.cad_id}: local cache and query node counts differ")
        if hashlib.sha256(values.tobytes(order="C")).hexdigest() != self.xyz_sha256:
            raise ValueError(f"{self.cad_id}: normalized xyz/node-order SHA mismatch; never remap by guessing")

    def tensor(self, *, mode: str, device="cpu"):
        """Actual and zero controls have identical channels and use no RNG."""
        import torch
        if mode == "zero":
            return torch.zeros((self.node_count, FEATURE_DIM), dtype=torch.float32, device=device)
        if mode != "actual":
            raise ValueError("A local cache tensor requires mode zero or actual")
        return torch.as_tensor(self.features, device=device, dtype=torch.float32)

    def record(self) -> dict:
        stat = self.path.stat()
        return {"cad_id": self.cad_id, "cache": str(self.path), "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "node_count": self.node_count,
                "xyz_sha256": self.xyz_sha256, "canonical_case": self.canonical_case,
                "source_canonical_npz": self.source_canonical_npz,
                "boundary_node_count": self.boundary_node_count, "feature_names": list(FEATURE_NAMES),
                "feature_dim": FEATURE_DIM, "schema_version": 1, "K": 32, "geometry_scale": .05,
                "schema_sha256": hashlib.sha256(json.dumps(self.schema, sort_keys=True).encode()).hexdigest(),
                "target_independent": True, "all_loads_share_cache": True,
                "limitations": self.schema.get("limitations")}


def load_local_geometry(directory: str | Path, cad_id: str, *, xyz=None) -> LocalGeometryCache:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", cad_id) or cad_id in (".", ".."):
        raise ValueError("Invalid CAD cache key")
    path = Path(directory).resolve() / f"{cad_id}.npz"
    required = {"features", "xyz_sha256", "node_count", "cad_id", "canonical_case",
                "feature_names", "K", "geometry_scale", "schema_version", "boundary_node_count",
                "schema_json", "source_canonical_npz", "target_independent"}
    with np.load(path, allow_pickle=False) as stored:
        if required - set(stored.files):
            raise ValueError(f"{path}: missing cache metadata {sorted(required-set(stored.files))}")
        scalar = lambda key: np.asarray(stored[key]).item()
        features = stored["features"]
        actual_cad, count = str(scalar("cad_id")), int(scalar("node_count"))
        sha, canonical = str(scalar("xyz_sha256")), str(scalar("canonical_case"))
        source, boundary_count = str(scalar("source_canonical_npz")), int(scalar("boundary_node_count"))
        schema = json.loads(str(scalar("schema_json")))
        if actual_cad != cad_id or canonical != f"{cad_id}/ver":
            raise ValueError(f"{path}: CAD/canonical source mismatch")
        if int(scalar("schema_version")) != 1 or int(scalar("K")) != 32 or float(scalar("geometry_scale")) != .05:
            raise ValueError(f"{path}: local geometry schema/version/K/scale mismatch")
        if stored["feature_names"].tolist() != list(FEATURE_NAMES):
            raise ValueError(f"{path}: local feature order differs from the decoder contract")
        if not bool(scalar("target_independent")):
            raise ValueError(f"{path}: target-independent geometry provenance is required")
    if schema.get("version") != 1 or schema.get("feature_names") != list(FEATURE_NAMES) or schema.get("target_fit") is not False:
        raise ValueError(f"{path}: schema JSON is inconsistent or target-fitted")
    if set(schema.get("data_access", [])) != {"xyz", "faces", "sample_key"}:
        raise ValueError(f"{path}: cache preparation accessed unapproved target/field members")
    if not re.fullmatch(r"[0-9a-f]{64}", sha) or boundary_count < 32 or count < 1:
        raise ValueError(f"{path}: invalid node/hash metadata")
    if features.dtype != np.dtype("float32") or features.shape != (count, FEATURE_DIM) or not np.isfinite(features).all():
        raise ValueError(f"{path}: expected finite float32[N,13] features")
    result = LocalGeometryCache(np.ascontiguousarray(features), cad_id, count, sha, path,
                                canonical, source, boundary_count, schema)
    if xyz is not None:
        result.verify_xyz(xyz)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True, help="One allowed train/validation prepared case; only xyz/key are read")
    args = p.parse_args()
    with np.load(args.prepared, allow_pickle=False) as stored:
        xyz = stored["xyz"]
        key = str(stored["sample_key"].item())
    cache = load_local_geometry(args.cache, key.split("/", 1)[0], xyz=xyz)
    print(json.dumps(cache.record(), indent=2))


if __name__ == "__main__":
    main()
