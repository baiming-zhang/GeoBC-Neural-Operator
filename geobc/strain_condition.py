"""Corner-P1 recovered frozen strain conditions, not exact TETRA10 strain.

Solve a constant gradient from four corner displacements, symmetrize, and
volume-average each incident cell value onto all ten original node IDs. The
six channels are tensor strains xx,yy,zz,xy,xz,yz (shear is not doubled).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .displacement_condition import (BASIS_ORDER, file_identity, host_path, load_response_manifest,
                                     verify_file_identity)
from .local_geometry import normalized_xyz_array, xyz_sha256

COMPONENT_ORDER = ("xx", "yy", "zz", "xy", "xz", "yz")
FEATURE_NAMES = tuple(f"predicted_P1_epsilon_{c}_over_train_common_strain_RMS" for c in COMPONENT_ORDER)
FEATURE_DIM = 6
SCHEMA_VERSION = 1
RECOVERY = "four-corner P1 constant gradient; symmetric tensor; absolute corner-tetra volume average to all original10 nodes"


def load_strain_manifest(directory, *, expected_parent=None):
    path = host_path(directory) / "manifest.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "completed" or result.get("complete") is not True or result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Recovered strain cache must be complete with supported schema")
    parent_identity = result["parent_u_manifest"]
    parent_path = host_path(parent_identity["path"])
    verify_file_identity(parent_path, parent_identity, small_sha=True)
    parent = load_response_manifest(parent_path.parent, expected_teacher=expected_parent)
    parent_gate = json.loads(host_path(parent["gate"]["path"]).read_text(encoding="utf-8"))
    split = json.loads(host_path(parent_gate["split"]["path"]).read_text(encoding="utf-8"))
    fingerprint = parent["teacher_fingerprint"]
    if (result["parent_response_fingerprint"] != fingerprint or result["parent_teacher_fingerprint"] != fingerprint
            or set(result["CADs"]) != set(parent["CADs"]) or len(result["CADs"]) != 45
            or result["basis_order"] != list(BASIS_ORDER) or result["component_order"] != list(COMPONENT_ORDER)
            or result["groups"] != split["groups"] or result["basis_units"] != "raw physical dimensionless tensor strain"
            or result["recovery"] != RECOVERY or result.get("engineering_shear") is not False):
        raise ValueError("Recovered strain parent/axes/recovery contract differs")
    scale = float(result["scale"])
    scaler = result["scaler"]
    if (not math.isfinite(scale) or scale <= 0 or scale != scaler["scale"] or scaler["center"] != 0
            or scaler["fit_CADs"] != result["groups"]["train"] or len(scaler["fit_CADs"]) != 40
            or scaler["fit_loads"] != list(BASIS_ORDER) or scaler["channels"] != list(COMPONENT_ORDER)
            or scaler["fit_nodes"] != sum(result["CADs"][c]["node_count"] for c in scaler["fit_CADs"])
            or scaler["denominator"] != scaler["fit_nodes"] * 3 * 6
            or scaler["sum_squared_raw_strain"] != sum(result["CADs"][c]["sum_squared_raw_basis_strain"] for c in scaler["fit_CADs"])
            or not math.isclose(scale, math.sqrt(scaler["sum_squared_raw_strain"] / scaler["denominator"]), rel_tol=1e-14)):
        raise ValueError("Strain scale must be one positive zero-centered train40 predicted-basis RMS")
    for cad, record in result["CADs"].items():
        source = record["raw_geometry_source"]
        if host_path(source["path"]).exists():
            verify_file_identity(host_path(source["path"]), source)
        if record["xyz_sha256"] != parent["CADs"][cad]["xyz_sha256"] or record["node_count"] != parent["CADs"][cad]["node_count"]:
            raise ValueError("Recovered strain and parent u query node identities differ")
    return result


@dataclass(frozen=True)
class StrainResponseCache:
    basis_strain: np.ndarray
    scale: float
    cad_id: str
    node_count: int
    xyz_sha256: str
    path: Path
    parent_response_fingerprint: str
    recovery_fingerprint: str
    metadata: dict

    def verify_xyz(self, xyz):
        values = normalized_xyz_array(xyz)
        if len(values) != self.node_count or xyz_sha256(values) != self.xyz_sha256:
            raise ValueError(f"{self.cad_id}: strain/query normalized xyz order or N differs")

    def tensor(self, device="cpu"):
        import torch
        return torch.as_tensor(self.basis_strain, dtype=torch.float32, device=device)

    def record(self):
        return {"cad_id": self.cad_id, "cache": str(self.path), **{k: v for k, v in file_identity(self.path).items() if k != "path"},
                "node_count": self.node_count, "xyz_sha256": self.xyz_sha256, "scale": self.scale, "center": 0.0,
                "basis_order": list(BASIS_ORDER), "component_order": list(COMPONENT_ORDER),
                "parent_response_fingerprint": self.parent_response_fingerprint,
                "recovery_fingerprint": self.recovery_fingerprint, "basis_units": "raw physical dimensionless tensor strain",
                "recovery": RECOVERY, "engineering_shear": False, "feature_names": list(FEATURE_NAMES),
                "feature_dim": FEATURE_DIM, "schema_version": SCHEMA_VERSION,
                "normalization": self.metadata["normalization"], "raw_geometry_source": self.metadata["raw_geometry_source"],
                "extra_u_supervision": True, "GT_field_inputs": False}


def load_strain_cache(directory, cad_id, *, xyz=None, expected_parent=None, manifest=None):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", cad_id) or cad_id in (".", ".."):
        raise ValueError("Invalid strain CAD key")
    directory = host_path(directory)
    manifest = manifest or load_strain_manifest(directory, expected_parent=expected_parent)
    if expected_parent is not None and expected_parent != manifest["parent_response_fingerprint"]:
        raise ValueError("Recovered strain uses a different frozen parent teacher")
    if cad_id not in manifest["CADs"]:
        raise ValueError("CAD outside the train40+validation5 strain cache allowlist")
    record = manifest["CADs"][cad_id]
    path = directory / f"{cad_id}.npz"
    if record["file"] != path.name:
        raise ValueError("Strain CAD path escaped its identity")
    verify_file_identity(path, record)
    with np.load(path, allow_pickle=False) as stored:
        required = {"basis_strain", "cad_id", "node_count", "xyz_sha256", "basis_order", "component_order", "schema_version",
                    "parent_response_fingerprint", "recovery_fingerprint", "record_json"}
        if required - set(stored.files) or set(stored.files) & {"u", "u_true", "displacement", "stress", "stress_true"}:
            raise ValueError("Strain cache has missing metadata or forbidden GT fields")
        scalar = lambda k: np.asarray(stored[k]).item()
        values = stored["basis_strain"]
        metadata = json.loads(str(scalar("record_json")))
        if (str(scalar("cad_id")) != cad_id or int(scalar("node_count")) != record["node_count"]
                or str(scalar("xyz_sha256")) != record["xyz_sha256"] or int(scalar("schema_version")) != SCHEMA_VERSION
                or str(scalar("parent_response_fingerprint")) != manifest["parent_response_fingerprint"]
                or str(scalar("recovery_fingerprint")) != manifest["recovery_fingerprint"]
                or stored["basis_order"].tolist() != list(BASIS_ORDER) or stored["component_order"].tolist() != list(COMPONENT_ORDER)
                or metadata != {k: v for k, v in record.items() if k not in ("file", "bytes", "mtime_ns", "sha256")}):
            raise ValueError("Strain cache metadata differs from its completed manifest")
    count, sha = record["node_count"], record["xyz_sha256"]
    if count < 1 or not re.fullmatch(r"[0-9a-f]{64}", sha) or values.dtype != np.dtype("float32") or values.shape != (count, 3, 6) or not np.isfinite(values).all():
        raise ValueError("Expected finite float32 raw physical strain[N,3loads,6tensor]")
    result = StrainResponseCache(np.ascontiguousarray(values), manifest["scale"], cad_id, count, sha, path,
                                 manifest["parent_response_fingerprint"], manifest["recovery_fingerprint"], metadata)
    if xyz is not None:
        result.verify_xyz(xyz)
    return result


def strain_condition(basis_strain, node_indices, alpha, *, scale, mode="actual"):
    if not math.isfinite(scale) or scale <= 0 or mode not in ("zero", "actual") or basis_strain.ndim != 3 or basis_strain.shape[1:] != (3, 6):
        raise ValueError("Expected raw strain[N,3,6], positive fixed scale, and zero/actual mode")
    if hasattr(basis_strain, "detach"):
        import torch
        values = basis_strain if node_indices is None else basis_strain.index_select(0, torch.as_tensor(node_indices, dtype=torch.long, device=basis_strain.device))
        coefficients = torch.as_tensor(alpha, dtype=values.dtype, device=values.device)
        if coefficients.shape == (3,):
            coefficients = coefficients.expand(len(values), -1)
        if coefficients.shape != (len(values), 3):
            raise ValueError("alpha needs [3] or [queryN,3]")
        return values.new_zeros((len(values), FEATURE_DIM)) if mode == "zero" else torch.einsum("nl,nlc->nc", coefficients, values) / scale
    values = np.asarray(basis_strain, np.float32)
    if node_indices is not None:
        ids = np.asarray(node_indices)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= len(values)):
            raise ValueError("Invalid original-node strain query index")
        values = values[ids]
    coefficients = np.asarray(alpha, np.float32)
    if coefficients.shape == (3,):
        coefficients = np.broadcast_to(coefficients, (len(values), 3))
    if coefficients.shape != (len(values), 3) or not np.isfinite(coefficients).all():
        raise ValueError("alpha needs finite [3] or [queryN,3]")
    return np.zeros((len(values), FEATURE_DIM), np.float32) if mode == "zero" else np.einsum("nl,nlc->nc", coefficients, values) / np.float32(scale)


def recover_corner_p1_strain(xyz_normalized, cells, basis_scaled, *, u_scale_mm, characteristic_length_mm,
                             batch_size=262144, device="cpu"):
    """FP64 solve/volume scatter; float32 RAW physical strain and diagnostics.

    D[edge,coordinate] * H[coordinate,component] = delta u[edge,component].
    Physical gradient is (u_scale_mm/L_mm)*H.T. All ten node IDs receive each
    cell's constant strain with positive absolute physical corner-tetra volume.
    No cell is skipped, no strain clipping/smoothing parameter is fitted.
    """
    xyz = np.ascontiguousarray(xyz_normalized, np.float64)
    topology = np.ascontiguousarray(cells, np.int64)
    values = np.ascontiguousarray(basis_scaled, np.float64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or topology.ndim != 2 or topology.shape[1] != 10 or not len(topology)
            or values.shape != (len(xyz), 3, 3) or topology.min() < 0 or topology.max() >= len(xyz)
            or not np.isfinite(xyz).all() or not np.isfinite(values).all() or batch_size < 1
            or not math.isfinite(u_scale_mm) or u_scale_mm <= 0
            or not math.isfinite(characteristic_length_mm) or characteristic_length_mm <= 0):
        raise ValueError("Recovery needs finite geometry/u basis, source TETRA10 topology and fixed physical scales")
    factor = u_scale_mm / characteristic_length_mm
    minimum, maximum = math.inf, 0.0
    if str(device) == "cpu":
        accum = np.zeros((len(xyz), 18), np.float64)
        weights = np.zeros(len(xyz), np.float64)
        for start in range(0, len(topology), batch_size):
            ids = topology[start:start + batch_size]
            corner = ids[:, :4]
            d = xyz[corner[:, 1:]] - xyz[corner[:, :1]]
            volume = np.abs(np.linalg.det(d)) * characteristic_length_mm ** 3 / 6
            if not np.isfinite(volume).all() or np.any(volume <= 0):
                raise ValueError("Degenerate/nonfinite corner tetrahedron; P1 recovery cannot silently skip it")
            du = (values[corner[:, 1:]] - values[corner[:, :1]]).reshape(-1, 3, 9)
            h = np.linalg.solve(d, du).reshape(-1, 3, 3, 3).transpose(0, 2, 1, 3)
            strain = np.stack((h[:, :, 0, 0], h[:, :, 1, 1], h[:, :, 2, 2],
                               .5 * (h[:, :, 0, 1] + h[:, :, 1, 0]),
                               .5 * (h[:, :, 0, 2] + h[:, :, 2, 0]),
                               .5 * (h[:, :, 1, 2] + h[:, :, 2, 1])), axis=-1) * factor
            weighted = strain.reshape(-1, 18) * volume[:, None]
            if not np.isfinite(weighted).all():
                raise FloatingPointError("Nonfinite FP64 recovered cell strain")
            for column in range(10):
                np.add.at(accum, ids[:, column], weighted)
                np.add.at(weights, ids[:, column], volume)
            minimum, maximum = min(minimum, float(volume.min())), max(maximum, float(volume.max()))
        uncovered = int(np.count_nonzero(weights <= 0))
        if uncovered:
            raise ValueError(f"{uncovered} original nodes lack incident positive tetra volume")
        output = (accum / weights[:, None]).reshape(-1, 3, 6).astype(np.float32)
    else:
        import torch
        target = torch.device(device)
        tx, tc, tu = (torch.as_tensor(a, device=target) for a in (xyz, topology, values))
        accum = torch.zeros((len(xyz), 18), dtype=torch.float64, device=target)
        weights = torch.zeros(len(xyz), dtype=torch.float64, device=target)
        with torch.no_grad():
            for start in range(0, len(topology), batch_size):
                ids = tc[start:start + batch_size]
                corner = ids[:, :4]
                d = tx[corner[:, 1:]] - tx[corner[:, :1]]
                volume = torch.linalg.det(d).abs() * characteristic_length_mm ** 3 / 6
                if not torch.isfinite(volume).all().item() or (volume <= 0).any().item():
                    raise ValueError("Degenerate/nonfinite source corner tetrahedron")
                du = (tu[corner[:, 1:]] - tu[corner[:, :1]]).reshape(-1, 3, 9)
                solution, info = torch.linalg.solve_ex(d, du)
                if (info != 0).any().item():
                    raise ValueError("Singular FP64 corner geometry solve")
                h = solution.reshape(-1, 3, 3, 3).permute(0, 2, 1, 3)
                strain = torch.stack((h[:, :, 0, 0], h[:, :, 1, 1], h[:, :, 2, 2],
                                      .5 * (h[:, :, 0, 1] + h[:, :, 1, 0]),
                                      .5 * (h[:, :, 0, 2] + h[:, :, 2, 0]),
                                      .5 * (h[:, :, 1, 2] + h[:, :, 2, 1])), dim=-1) * factor
                weighted = strain.reshape(-1, 18) * volume[:, None]
                if not torch.isfinite(weighted).all().item():
                    raise FloatingPointError("Nonfinite FP64 recovered cell strain")
                for column in range(10):
                    accum.index_add_(0, ids[:, column], weighted)
                    weights.index_add_(0, ids[:, column], volume)
                minimum, maximum = min(minimum, volume.min().item()), max(maximum, volume.max().item())
            uncovered = int((weights <= 0).sum().item())
            if uncovered:
                raise ValueError(f"{uncovered} original nodes lack incident positive tetra volume")
            output = (accum / weights[:, None]).reshape(-1, 3, 6).float().cpu().numpy()
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite final float32 physical strain proxy")
    return output, {"nodes": len(xyz), "tetrahedra": len(topology), "uncovered_nodes": 0,
                    "minimum_abs_corner_volume_mm3": minimum, "maximum_abs_corner_volume_mm3": maximum,
                    "solve_precision": "float64", "scatter_precision": "float64", "output_precision": "float32",
                    "physical_gradient_factor": factor, "engineering_shear": False, "recovery": RECOVERY}


__all__ = ["load_strain_manifest", "load_strain_cache", "StrainResponseCache", "strain_condition",
           "recover_corner_p1_strain", "FEATURE_NAMES", "FEATURE_DIM", "COMPONENT_ORDER", "RECOVERY"]
