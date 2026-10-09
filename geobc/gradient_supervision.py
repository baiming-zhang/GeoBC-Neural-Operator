"""Cell P1 supervision derived from signed known-load displacement labels.

This reference is not a solver tensor label. Edges have axis [edge,coordinate]
in physical mm. inv(D) @ delta_u gives [coordinate,component]; the symmetric
tensor is stored xx,yy,zz,xy,xz,yz, with undoubled tensor shear.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re

import numpy as np

from .displacement_condition import (BASIS_ORDER, file_identity, host_path, json_sha256,
                                      verify_file_identity)
from .local_geometry import normalized_xyz_array, xyz_sha256

COMPONENT_ORDER = ("xx", "yy", "zz", "xy", "xz", "yz")
FROBENIUS_WEIGHTS = (1., 1., 1., 2., 2., 2.)
SCHEMA_VERSION = 1
REFERENCE = "reference corner-P1 tensor strain from signed known-load displacement"
ORIENTATION = "D[edge,coordinate]=x_corner1..3_mm-x_corner0_mm; H[coordinate,component]=inverse(D)@delta_u_mm; epsilon=sym(H.T)"
NORMALIZATION_FLOOR = 1e-24


def cell_tensor_strain(predicted_u_mm, inverse_physical_edge_mm):
    """[T,4,3] physical u -> [T,6] FP64 tensor, retaining parameter gradients.

    Coordinates and inverse edges are fixed data. No coordinate derivative,
    autograd Jacobian, or second derivative is used. Casting u to double does
    not recover precision already lost in a native FP16 network output.
    """
    if predicted_u_mm.ndim != 3 or predicted_u_mm.shape[1:] != (4, 3) or inverse_physical_edge_mm.shape != (len(predicted_u_mm), 3, 3):
        raise ValueError("Expected u_mm[T,4,3] and inverse physical edges[T,3,3]")
    if hasattr(predicted_u_mm, "detach"):
        import torch
        values = predicted_u_mm.to(dtype=torch.float64)
        inverse = torch.as_tensor(inverse_physical_edge_mm, dtype=torch.float64, device=values.device)
        h = inverse @ (values[:, 1:] - values[:, :1])
        return torch.stack((h[:, 0, 0], h[:, 1, 1], h[:, 2, 2],
                            .5 * (h[:, 0, 1] + h[:, 1, 0]), .5 * (h[:, 0, 2] + h[:, 2, 0]),
                            .5 * (h[:, 1, 2] + h[:, 2, 1])), dim=-1)
    values = np.asarray(predicted_u_mm, np.float64)
    h = np.asarray(inverse_physical_edge_mm, np.float64) @ (values[:, 1:] - values[:, :1])
    return np.stack((h[:, 0, 0], h[:, 1, 1], h[:, 2, 2], .5*(h[:, 0, 1]+h[:, 1, 0]),
                     .5*(h[:, 0, 2]+h[:, 2, 0]), .5*(h[:, 1, 2]+h[:, 2, 1])), axis=-1)


def relative_frobenius_loss(predicted, reference, volumes, mean_volume_mm3, reference_squared_rms, floor=NORMALIZATION_FLOOR):
    """Uniform-cell importance estimator of full-volume relative tensor MSE.

    Scalar mean volume and reference RMS squared must be fixed full TRAIN-case
    cache values. Neither batch-volume normalization nor a mean over six
    components is applied. Caller enforces the train-only normalization guard.
    """
    if predicted.ndim != 2 or predicted.shape[1] != 6 or reference.shape != predicted.shape or volumes.shape != (len(predicted),) or not len(predicted):
        raise ValueError("Expected matching nonempty tensor[T,6] and volume[T]")
    mean_volume_mm3, reference_squared_rms, floor = map(float, (mean_volume_mm3, reference_squared_rms, floor))
    if not math.isfinite(mean_volume_mm3) or mean_volume_mm3 <= 0 or not math.isfinite(reference_squared_rms) or reference_squared_rms < 0 or floor != NORMALIZATION_FLOOR:
        raise ValueError("Invalid fixed case volume/RMS or changed declared floor")
    if hasattr(predicted, "detach"):
        import torch
        reference = torch.as_tensor(reference, dtype=torch.float64, device=predicted.device)
        v = torch.as_tensor(volumes, dtype=torch.float64, device=predicted.device)
        w = predicted.new_tensor(FROBENIUS_WEIGHTS, dtype=torch.float64)
        return (((predicted.double()-reference).square()*w).sum(-1) * (v/mean_volume_mm3)).mean() / max(reference_squared_rms, floor)
    error = np.asarray(predicted, np.float64)-np.asarray(reference, np.float64)
    return float(np.mean(np.sum(error*error*np.asarray(FROBENIUS_WEIGHTS), axis=-1)*np.asarray(volumes)/mean_volume_mm3) / max(reference_squared_rms, floor))


def recover_node_tensor_strain(cell_strain, volumes, cells, node_count, *, batch_size=262144):
    """Incident |V| average of constant P1 cell tensors onto all10 node IDs."""
    if cell_strain.ndim != 2 or cell_strain.shape[1] != 6 or cells.shape != (len(cell_strain), 10) or volumes.shape != (len(cell_strain),) or node_count < 1 or batch_size < 1:
        raise ValueError("Expected cell tensor[T,6], original cells[T,10], volumes[T]")
    if hasattr(cell_strain, "detach"):
        import torch
        values = cell_strain.double()
        v = torch.as_tensor(volumes, dtype=torch.float64, device=values.device)
        ids = torch.as_tensor(cells, dtype=torch.long, device=values.device)
        summed = values.new_zeros((node_count, 6)); incident = values.new_zeros(node_count)
        for start in range(0, len(values), batch_size):
            end = min(start+batch_size, len(values))
            weighted = values[start:end]*v[start:end, None]
            for slot in range(10):
                summed.index_add_(0, ids[start:end, slot], weighted)
                incident.index_add_(0, ids[start:end, slot], v[start:end])
        if not bool((incident > 0).all()):
            raise ValueError("Original node is not incident to a positive-volume cell")
        return summed/incident[:, None]
    values, v, ids = np.asarray(cell_strain, np.float64), np.asarray(volumes, np.float64), np.asarray(cells, np.int64)
    summed = np.zeros((node_count, 6), np.float64); incident = np.zeros(node_count, np.float64)
    for start in range(0, len(values), batch_size):
        end = min(start+batch_size, len(values)); weighted = values[start:end]*v[start:end, None]
        for slot in range(10):
            np.add.at(summed, ids[start:end, slot], weighted); np.add.at(incident, ids[start:end, slot], v[start:end])
    if not np.all(incident > 0):
        raise ValueError("Original node is not incident to a positive-volume cell")
    return summed/incident[:, None]


def node_tensor_metrics(reference, prediction):
    """All-node unweighted symmetric Frobenius diagnostics, raw dimensionless."""
    if hasattr(reference, "detach"):
        reference = reference.detach().cpu().numpy()
    if hasattr(prediction, "detach"):
        prediction = prediction.detach().cpu().numpy()
    truth, pred = np.asarray(reference, np.float64), np.asarray(prediction, np.float64)
    if truth.shape != pred.shape or truth.ndim != 2 or truth.shape[1] != 6 or not len(truth) or not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("Finite matching node tensor[N,6] required")
    w = np.asarray(FROBENIUS_WEIGHTS); error = pred-truth
    true_sq, pred_sq, error_sq = ((a*a*w).sum(-1) for a in (truth, pred, error))
    relative = float(np.sqrt(error_sq.sum()/true_sq.sum())) if true_sq.sum() else None
    return {"nodes": len(truth), "symmetric_Frobenius_relative_l2": relative, "relative_l2": relative,
            "truth_energy": float(true_sq.sum()), "error_energy": float(error_sq.sum()),
            "plain6_relative_l2": float(np.linalg.norm(error)/np.linalg.norm(truth)) if np.linalg.norm(truth) else None,
            "reference_tensor_Frobenius_RMS": float(np.sqrt(true_sq.mean())), "prediction_tensor_Frobenius_RMS": float(np.sqrt(pred_sq.mean())),
            "error_tensor_Frobenius_RMS": float(np.sqrt(error_sq.mean())), "max_error_tensor_Frobenius_norm": float(np.sqrt(error_sq.max())),
            "reference_max_abs_component": float(np.abs(truth).max()), "prediction_max_abs_component": float(np.abs(pred).max())}


def load_gradient_manifest(directory):
    """JSON/small source SHA/stat only; never opens fields, NPZ, weights, CUDA."""
    directory = host_path(directory)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("status") != "completed" or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Gradient-supervision cache must be completed")
    verify_file_identity(directory/"config.json", manifest["config_source"], small_sha=True)
    config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
    fingerprint_config = {k:v for k,v in config.items() if k!="supervision_fingerprint"}
    if json_sha256(fingerprint_config)!=config["supervision_fingerprint"] or any(manifest.get(k)!=v for k,v in config.items()):
        raise ValueError("Completed gradient manifest differs from its immutable preparation recipe")
    verify_file_identity(host_path(manifest["design_source"]["path"]), manifest["design_source"], small_sha=True)
    split_source = manifest["split_source"]
    verify_file_identity(host_path(split_source["path"]), split_source, small_sha=True)
    split = json.loads(host_path(split_source["path"]).read_text(encoding="utf-8"))
    groups = manifest["groups"]
    allowed = groups["train"]+groups["validation"]
    if (groups != split["groups"] or len(groups["train"]) != 40 or len(groups["validation"]) != 5 or len(allowed) != len(set(allowed))
            or set(allowed) & set(groups["test"]) or set(manifest["CADs"]) != set(allowed)
            or manifest["basis_order"] != list(BASIS_ORDER) or manifest["component_order"] != list(COMPONENT_ORDER)
            or manifest["reference"] != REFERENCE or manifest["orientation"] != ORIENTATION
            or manifest["frobenius_weights"] != list(FROBENIUS_WEIGHTS) or manifest["normalization_floor"] != NORMALIZATION_FLOOR
            or manifest["engineering_shear"] is not False or manifest["strain_units"] != "dimensionless"):
        raise ValueError("Gradient supervision scope, axes, or physical objective differs")
    for key in ("parent_u_manifest", "load_basis_source"):
        verify_file_identity(host_path(manifest[key]["path"]), manifest[key], small_sha=True)
    for cad, record in manifest["CADs"].items():
        if cad not in allowed or record["cad_id"] != cad or record["file"] != f"{cad}.npz":
            raise ValueError("Gradient CAD allowlist/path differs")
        verify_file_identity(directory/record["file"], record)
        for identity in [record["raw_geometry_source"], *record["displacement_sources"]]:
            verify_file_identity(host_path(identity["path"]), identity)
        if record["mean_volume_mm3"] <= 0 or not math.isfinite(record["mean_volume_mm3"]):
            raise ValueError("Nonpositive mean cell volume")
        norms = record["reference_squared_rms_by_load"]
        if set(norms) != set(BASIS_ORDER) or any(not math.isfinite(x) or x < 0 for x in norms.values()):
            raise ValueError("Nonfinite reference tensor norm")
        expected = {load: max(norms[load], NORMALIZATION_FLOOR) for load in BASIS_ORDER} if cad in groups["train"] else None
        active = {load: norms[load] < NORMALIZATION_FLOOR for load in BASIS_ORDER} if cad in groups["train"] else None
        if record["loss_normalization_by_load"] != expected or record["floor_active_by_load"] != active:
            raise ValueError("Only training cases may supply fixed gradient loss weights")
    return manifest


@dataclass(frozen=True)
class GradientCache:
    corners: np.ndarray
    cells: np.ndarray
    inverse_physical_edge_mm: np.ndarray
    volumes: np.ndarray
    reference_cell_tensor_strain: np.ndarray
    metadata: dict
    path: object

    @property
    def node_count(self): return self.metadata["node_count"]
    @property
    def mean_volume_mm3(self): return self.metadata["mean_volume_mm3"]
    @property
    def reference_squared_rms(self): return self.metadata["reference_squared_rms_by_load"]
    @property
    def loss_normalization_by_load(self): return self.metadata["loss_normalization_by_load"]
    @property
    def cad_id(self): return self.metadata["cad_id"]
    def verify_xyz(self, xyz):
        values = normalized_xyz_array(xyz)
        if len(values) != self.node_count or xyz_sha256(values) != self.metadata["xyz_sha256"]:
            raise ValueError("Gradient cache normalized XYZ/node order differs")
    def tensor(self, device="cpu"):
        import torch
        return {"corners": torch.as_tensor(self.corners, dtype=torch.long, device=device),
                "cells": torch.as_tensor(self.cells, dtype=torch.long, device=device),
                "inverse_physical_edge_mm": torch.as_tensor(self.inverse_physical_edge_mm, dtype=torch.float64, device=device),
                "volumes": torch.as_tensor(self.volumes, dtype=torch.float64, device=device),
                "reference_cell_tensor_strain": torch.as_tensor(self.reference_cell_tensor_strain, dtype=torch.float32, device=device)}
    def record(self): return {"cache": str(self.path), **self.metadata}


def load_gradient_cache(directory, cad_id, *, xyz=None, manifest=None):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", cad_id) or cad_id in (".", ".."):
        raise ValueError("Invalid CAD key")
    directory = host_path(directory); manifest = manifest or load_gradient_manifest(directory)
    if cad_id not in manifest["CADs"]:
        raise ValueError("CAD outside train40+val5 allowlist")
    record = manifest["CADs"][cad_id]; path = directory/record["file"]
    verify_file_identity(path, record)
    metadata = {k:v for k,v in record.items() if k not in ("file", "bytes", "mtime_ns", "sha256")}
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) & {"displacement", "u", "u_true", "stress", "stress_true"}:
            raise ValueError("Unexpected raw field in derivative cache")
        if json.loads(str(stored["record_json"].item())) != metadata or str(stored["supervision_fingerprint"].item()) != manifest["supervision_fingerprint"]:
            raise ValueError("Gradient cache/manifest metadata differs")
        corners, cells, inverse, volumes, reference = (stored[k] for k in ("corner_ids", "cells10", "inverse_physical_edge_mm", "volumes_mm3", "reference_cell_tensor_strain"))
    count = record["cell_count"]
    if (corners.dtype != np.dtype("int64") or corners.shape != (count,4) or cells.dtype != np.dtype("int64") or cells.shape != (count,10)
            or not np.array_equal(corners, cells[:,:4]) or cells.min() < 0 or cells.max() >= record["node_count"]
            or inverse.dtype != np.dtype("float64") or inverse.shape != (count,3,3) or not np.isfinite(inverse).all()
            or volumes.dtype != np.dtype("float64") or volumes.shape != (count,) or not np.isfinite(volumes).all() or np.any(volumes<=0)
            or reference.dtype != np.dtype("float32") or reference.shape != (count,3,6) or not np.isfinite(reference).all()
            or not math.isclose(float(volumes.mean()), record["mean_volume_mm3"], rel_tol=1e-14)):
        raise ValueError("Invalid original connectivity, FP64 physical geometry, or reference P1 tensor")
    result = GradientCache(corners,cells,inverse,volumes,reference,metadata,path)
    if xyz is not None: result.verify_xyz(xyz)
    return result
