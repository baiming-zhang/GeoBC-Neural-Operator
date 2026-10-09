"""Frozen predicted displacement conditions, not ground-truth field inputs.

One CAD cache stores [node, ver/hor/tor, ux/uy/uz] response bases. The student
receives only their three-component combination in the teacher's fixed common
zero-centered scale. A zero control keeps the same three input slots.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from .local_geometry import normalized_xyz_array, xyz_sha256

ROOT = Path(__file__).resolve().parents[1]
BASIS_ORDER = ("ver", "hor", "tor")
COMPONENT_ORDER = ("ux", "uy", "uz")
FEATURE_NAMES = tuple(f"predicted_{c}_over_teacher_train_common_u_RMS" for c in COMPONENT_ORDER)
FEATURE_DIM = 3
SCHEMA_VERSION = 1


def host_path(value) -> Path:
    value = str(value).replace("\\", "/")
    if os.name == "nt" and value.startswith("/mnt/") and len(value) > 7:
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and len(value) > 2 and value[1:3] == ":/":
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def json_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_identity(path, *, small_sha=False) -> dict:
    path = host_path(path)
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if small_sha else None}


def verify_file_identity(path, expected, *, small_sha=False) -> dict:
    actual = file_identity(path, small_sha=small_sha)
    if any(actual[k] != expected[k] for k in ("bytes", "mtime_ns")):
        raise ValueError(f"{path}: frozen source size/mtime changed")
    if small_sha and actual["sha256"] != expected["sha256"]:
        raise ValueError(f"{path}: frozen small-file SHA changed")
    return actual


def teacher_fingerprint(gate) -> str:
    """Stable across host path conventions; no checkpoint contents are hashed."""
    t = gate["teacher"]
    return json_sha256({
        "schema_version": SCHEMA_VERSION, "best_step": t["best_step"],
        "checkpoint": {k: t["checkpoint"][k] for k in ("bytes", "mtime_ns")},
        "config_sha256": t["config"]["sha256"], "summary_sha256": t["summary"]["sha256"],
        "metrics_best_sha256": t["metrics_best"]["sha256"], "scaler": t["scaler"],
        "model": t["model"], "geometry": t["geometry"], "source_code_sha256": t["source_code_sha256"],
        "split_sha256": gate["split"]["sha256"], "load_basis_sha256": gate["load_basis"]["sha256"],
        "CADs": gate["cache_CADs"],
        "local_geometry": {c: {k: t["local_geometry"]["CADs"][c][k]
                                for k in ("bytes", "mtime_ns", "xyz_sha256", "schema_sha256")}
                           for c in gate["cache_CADs"]}})


def load_response_manifest(directory, *, expected_teacher=None) -> dict:
    path = host_path(directory) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("complete") is not True or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Frozen response cache must be complete with the supported schema")
    gate_path = host_path(manifest["gate"]["path"])
    if hashlib.sha256(gate_path.read_bytes()).hexdigest() != manifest["gate"]["sha256"]:
        raise ValueError("Teacher gate changed after response generation")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    for name in ("config", "summary", "metrics_best", "checkpoint", "geometry_clouds"):
        source = gate["teacher"][name]
        path_source = host_path(source["path"])
        if path_source.exists():
            verify_file_identity(path_source, source, small_sha=name in ("config", "summary", "metrics_best"))
    verify_file_identity(host_path(gate["load_basis"]["path"]), gate["load_basis"], small_sha=True)
    split_path = host_path(gate["split"]["path"])
    verify_file_identity(split_path, gate["split"], small_sha=True)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    allowed = split["groups"]["train"] + split["groups"]["validation"]
    if (gate.get("status") != "frozen" or gate.get("final_test_excluded") is not True
            or gate.get("dia_excluded_from_this_stage") is not True
            or len(allowed) != 45 or gate["cache_CADs"] != allowed
            or set(manifest["CADs"]) != set(allowed) or set(allowed) & set(split["groups"]["test"])):
        raise ValueError("Cache allowlist must contain exactly train40+validation5 CADs, excluding final test")
    fingerprint = teacher_fingerprint(gate)
    if fingerprint != manifest["teacher_fingerprint"] or expected_teacher is not None and expected_teacher != fingerprint:
        raise ValueError("Frozen response teacher identity differs")
    if manifest["basis_order"] != list(BASIS_ORDER) or manifest["component_order"] != list(COMPONENT_ORDER):
        raise ValueError("Frozen response axes differ")
    scale = float(manifest["scale_mm"])
    if not math.isfinite(scale) or scale <= 0 or scale != gate["teacher"]["scaler"]["scale_mm"]:
        raise ValueError("Response cache must use the teacher's fixed positive common u scale")
    return manifest


@dataclass(frozen=True)
class FrozenResponseCache:
    basis_scaled: np.ndarray
    scale_mm: float
    cad_id: str
    node_count: int
    xyz_sha256: str
    path: Path
    teacher_fingerprint: str
    gate_sha256: str
    normalization: dict

    def verify_xyz(self, xyz):
        values = normalized_xyz_array(xyz)
        if len(values) != self.node_count or xyz_sha256(values) != self.xyz_sha256:
            raise ValueError(f"{self.cad_id}: response/query node count or normalized xyz order differs")

    def tensor(self, device="cpu"):
        import torch
        return torch.as_tensor(self.basis_scaled, dtype=torch.float32, device=device)

    def record(self):
        return {"cad_id": self.cad_id, "cache": str(self.path), **{k: v for k, v in file_identity(self.path).items() if k != "path"},
                "node_count": self.node_count, "xyz_sha256": self.xyz_sha256,
                "teacher_fingerprint": self.teacher_fingerprint, "gate_sha256": self.gate_sha256,
                "scale_mm": self.scale_mm, "center_mm": 0.0, "basis_order": list(BASIS_ORDER),
                "component_order": list(COMPONENT_ORDER), "feature_names": list(FEATURE_NAMES),
                "feature_dim": FEATURE_DIM, "schema_version": SCHEMA_VERSION,
                "source": "frozen trained u predictor; no GT u/stress cache inputs",
                "extra_u_supervision": True, "all_loads_share_cache": True, "normalization": self.normalization}


def load_response_cache(directory, cad_id, *, xyz=None, expected_teacher=None, manifest=None) -> FrozenResponseCache:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", cad_id) or cad_id in (".", ".."):
        raise ValueError("Invalid response CAD key")
    directory = host_path(directory)
    manifest = manifest or load_response_manifest(directory, expected_teacher=expected_teacher)
    if expected_teacher is not None and expected_teacher != manifest["teacher_fingerprint"]:
        raise ValueError("Response cache teacher differs from the fixed training contract")
    if cad_id not in manifest["CADs"]:
        raise ValueError("CAD is outside the frozen train/validation response allowlist")
    record = manifest["CADs"][cad_id]
    path = directory / f"{cad_id}.npz"
    if record["file"] != path.name:
        raise ValueError("Response cache path escaped its CAD identity")
    verify_file_identity(path, record)
    required = {"basis_scaled", "cad_id", "node_count", "xyz_sha256", "basis_order", "component_order",
                "scale_mm", "center_mm", "schema_version", "teacher_fingerprint", "gate_sha256"}
    with np.load(path, allow_pickle=False) as stored:
        if required - set(stored.files) or set(stored.files) & {"stress", "displacement", "u_true", "u", "stress_true"}:
            raise ValueError("Response cache has missing metadata or forbidden ground-truth fields")
        scalar = lambda k: np.asarray(stored[k]).item()
        basis = stored["basis_scaled"]
        count, sha, scale = int(scalar("node_count")), str(scalar("xyz_sha256")), float(scalar("scale_mm"))
        normalization = json.loads(str(scalar("normalization_json"))) if "normalization_json" in stored.files else {"status": "unknown"}
        if "response_fingerprint" in record and ("response_fingerprint" not in stored.files or str(scalar("response_fingerprint")) != record["response_fingerprint"]):
            raise ValueError("Response cache geometry/source fingerprint differs")
        if (str(scalar("cad_id")) != cad_id or count != record["node_count"] or sha != record["xyz_sha256"]
                or int(scalar("schema_version")) != SCHEMA_VERSION or float(scalar("center_mm")) != 0
                or scale != manifest["scale_mm"] or str(scalar("teacher_fingerprint")) != manifest["teacher_fingerprint"]
                or str(scalar("gate_sha256")) != manifest["gate"]["sha256"]
                or normalization != record.get("normalization", {"status": "unknown"})
                or stored["basis_order"].tolist() != list(BASIS_ORDER) or stored["component_order"].tolist() != list(COMPONENT_ORDER)):
            raise ValueError("Response cache metadata differs from its completed manifest")
    if count < 1 or not re.fullmatch(r"[0-9a-f]{64}", sha) or basis.dtype != np.dtype("float32") or basis.shape != (count, 3, 3) or not np.isfinite(basis).all():
        raise ValueError("Response basis must be finite float32[N,3loads,3components]")
    result = FrozenResponseCache(np.ascontiguousarray(basis), scale, cad_id, count, sha, path,
                                 manifest["teacher_fingerprint"], manifest["gate"]["sha256"], normalization)
    if xyz is not None:
        result.verify_xyz(xyz)
    return result


def displacement_condition(basis_scaled, node_indices, alpha, *, mode="actual"):
    """Gather cached bases then compose three scaled u channels, NumPy or torch.

    alpha is [3] or [queryN,3] in ver/hor/tor order. No physical-unit centering,
    scale refitting, derivatives, target labels, or stress combination is used.
    """
    if mode not in ("zero", "actual") or basis_scaled.ndim != 3 or basis_scaled.shape[1:] != (3, 3):
        raise ValueError("Expected basis[N,3,3] and mode zero/actual")
    if hasattr(basis_scaled, "device") and hasattr(basis_scaled, "detach"):
        import torch
        basis = basis_scaled if node_indices is None else basis_scaled.index_select(0, torch.as_tensor(node_indices, dtype=torch.long, device=basis_scaled.device))
        coefficients = torch.as_tensor(alpha, dtype=basis.dtype, device=basis.device)
        if coefficients.shape == (3,):
            coefficients = coefficients.expand(len(basis), -1)
        if coefficients.shape != (len(basis), 3):
            raise ValueError("alpha must be [3] or [queryN,3]")
        return basis.new_zeros((len(basis), FEATURE_DIM)) if mode == "zero" else torch.einsum("nl,nlc->nc", coefficients, basis)
    values = np.asarray(basis_scaled, dtype=np.float32)
    if node_indices is not None:
        indices = np.asarray(node_indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= len(values)):
            raise ValueError("Node indices must be valid original-row integers")
        values = values[indices]
    coefficients = np.asarray(alpha, dtype=np.float32)
    if coefficients.shape == (3,):
        coefficients = np.broadcast_to(coefficients, (len(values), 3))
    if coefficients.shape != (len(values), 3) or not np.isfinite(coefficients).all():
        raise ValueError("alpha must be finite [3] or [queryN,3]")
    return np.zeros((len(values), FEATURE_DIM), np.float32) if mode == "zero" else np.einsum("nl,nlc->nc", coefficients, values)


def alpha_from_original_features(features, case_key, guard):
    """Validate original load slots against already frozen target-free metadata."""
    cad, load = case_key.split("/", 1)
    if load not in BASIS_ORDER:
        raise ValueError("This training stage excludes dia and other held-out loads")
    if features.ndim != 2 or features.shape[1] != 25 or not len(features):
        raise ValueError("Alpha validation needs original25 features before appending conditions")
    slots = features[:, 7:13]
    if hasattr(slots, "detach"):
        slots = slots.detach().cpu().numpy()
    slots = np.asarray(slots, np.float64)
    records = {r["cad_id"]: r for r in guard["cases"]}
    if cad not in records or guard.get("compatible") is not True or guard["canonical_basis"]["order"] != list(BASIS_ORDER):
        raise ValueError("CAD/load coefficients are not in the frozen rank3 metadata")
    alpha = np.asarray(records[cad]["alpha_by_load"][load]["alpha"], np.float64)
    mapping = np.asarray(guard["prepared_feature_mapping"]["feature_to_alpha_matrix_3x6"], np.float64)
    if alpha.shape != (3,) or not np.isfinite(alpha).all() or not np.allclose(slots @ mapping.T, alpha[None], rtol=1e-6, atol=2e-7):
        raise ValueError("Original force/moment slots disagree with the frozen physical alpha")
    return alpha.astype(np.float32)


__all__ = ["FrozenResponseCache", "load_response_manifest", "load_response_cache", "displacement_condition",
           "alpha_from_original_features", "FEATURE_NAMES", "FEATURE_DIM", "BASIS_ORDER", "COMPONENT_ORDER"]
