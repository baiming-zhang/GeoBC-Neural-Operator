"""Geometry-conditioned continuous hash field: a GeoBC candidate baseline.

This composes a small shared PointNet with the existing official TCNN field.
It is a fitting/generalization baseline to evaluate, not a claim of a new
algorithm. Geometry clouds contain normalized coordinates only. They must be
sampled independently of targets and reused for every load of the same CAD.

Training recipe: select 1--4 unique CADs, call encode_geometry once for those
CADs, and pass the returned tensor plus local cad_index for all queried nodes.
Do not detach or cache encoded contexts across optimizer steps. Cache the
fixed point clouds instead. Evaluation may cache contexts under no_grad().
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from .tcnn_fields import TCNNField, TCNNFieldConfig


@dataclass(frozen=True)
class ConditionedFieldConfig:
    geometry_mode: Literal["shape", "none", "id"] = "shape"
    descriptor_dim: int = 32
    condition_dim: int = 25
    cloud_points: int = 1024
    point_width: int = 64
    out_channels: int = 1
    seed: int = 2026
    n_levels: int = 12
    n_features_per_level: int = 2
    log2_hashmap_size: int = 19
    base_resolution: int = 8
    per_level_scale: float = 1.6
    hidden_dim: int = 64
    hidden_layers: int = 2

    def __post_init__(self) -> None:
        if self.geometry_mode not in ("shape", "none", "id"):
            raise ValueError("geometry_mode must be shape, none, or diagnostic id")
        if self.descriptor_dim not in (32, 64):
            raise ValueError("Use a 32- or 64-dimensional global descriptor")
        if self.condition_dim < 1 or self.cloud_points < 1 or self.point_width < 1:
            raise ValueError("Condition, cloud, and point widths must be positive")
        self.decoder_config()  # Validate the unchanged TCNN decoder settings.

    def decoder_config(self) -> TCNNFieldConfig:
        return TCNNFieldConfig(
            out_channels=self.out_channels, input_mode="both",
            condition_dim=self.condition_dim + self.descriptor_dim,
            n_levels=self.n_levels, n_features_per_level=self.n_features_per_level,
            log2_hashmap_size=self.log2_hashmap_size,
            base_resolution=self.base_resolution, per_level_scale=self.per_level_scale,
            hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers, seed=self.seed)

    def record(self) -> dict:
        return {
            **asdict(self), "method": "PointNet-conditioned continuous hash field",
            "claim": "GeoBC candidate baseline; no new-algorithm or generalization claim",
            "decoder": self.decoder_config().record(),
            "encoder": {
                "point_mlp": [3, 32, self.point_width, self.point_width],
                "pooling": "concatenated masked max and masked mean",
                "head": [2 * self.point_width, self.point_width, self.descriptor_dim],
                "activation": "ReLU", "output": "non-affine LayerNorm then tanh, bounded [-1,1]",
                "batch_norm": False, "dropout": False,
            },
            "conditioning": "existing local BC/material features followed by global geometry descriptor",
            "geometry_source": "fixed per-CAD normalized xyz only; prefer unique original boundary FEM nodes",
            "geometry_sampling": "CAD-key/seed-stable sampling without replacement; masked padding for small clouds",
            "geometry_target_independent": True,
            "geometry_load_independent": True,
            "context_gradient": "index_select/concatenate preserve gradients through native TCNN inputs",
            "context_training_cache": "clouds only; re-encode once per unique CAD each optimizer step",
            "control": "none/id retain the same decoder condition dimension and width as shape",
            "id_policy": "fixed case identity codes for a diagnostic only; cannot encode unseen geometry",
        }


class PointNetGeometryEncoder(nn.Module):
    """Permutation-invariant coordinates -> bounded global context, [B,D]."""

    def __init__(self, descriptor_dim: int = 32, point_width: int = 64) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, point_width), nn.ReLU(),
            nn.Linear(point_width, point_width), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(2 * point_width, point_width), nn.ReLU(),
            nn.Linear(point_width, descriptor_dim),
            nn.LayerNorm(descriptor_dim, elementwise_affine=False), nn.Tanh())

    def forward(self, cloud: Tensor, mask: Tensor | None = None) -> Tensor:
        if cloud.ndim != 3 or cloud.shape[-1] != 3 or min(cloud.shape[:2]) < 1:
            raise ValueError("geometry_cloud must have shape [B,P,3] with nonempty B/P")
        if mask is None:
            mask = torch.ones(cloud.shape[:2], dtype=torch.bool, device=cloud.device)
        if mask.dtype != torch.bool or mask.shape != cloud.shape[:2] or mask.device != cloud.device:
            raise ValueError("geometry_mask must be bool [B,P] on the cloud device")
        counts = mask.sum(dim=1, keepdim=True)
        if torch.any(counts == 0):
            raise ValueError("Every CAD cloud needs at least one valid geometry point")
        values = self.point_mlp(cloud.float())
        expanded = mask.unsqueeze(-1)
        maximum = values.masked_fill(~expanded, -torch.inf).amax(dim=1)
        average = (values * expanded).sum(dim=1) / counts.to(values.dtype)
        return self.head(torch.cat((maximum, average), dim=-1))


class ConditionedField(nn.Module):
    """Nodewise field with an explicitly shared, differentiable CAD context.

    forward(xyz[N,3], features[N,25], geometry_cloud[B,P,3], cad_index[N])
    or encode_geometry(...) followed by forward(..., context=context).

    cad_index indexes the B unique CADs in *this batch*, not dataset-global IDs.
    The same context must be supplied for all loads of one CAD. A missing
    cad_index is allowed only for a single context. An injected decoder is
    intended for CPU interface checks; it does not verify native CUDA gradients.
    """

    def __init__(self, config: ConditionedFieldConfig | None = None,
                 device: torch.device | str = "cuda", decoder: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config or ConditionedFieldConfig()
        self.encoder = (PointNetGeometryEncoder(self.config.descriptor_dim, self.config.point_width)
                        if self.config.geometry_mode == "shape" else None)
        expected = self.config.decoder_config()
        if decoder is not None and hasattr(decoder, "config") and decoder.config != expected:
            raise ValueError("Injected decoder config does not match the conditioned decoder")
        self.decoder = decoder if decoder is not None else TCNNField(config=expected, device=device)
        self.backend = getattr(self.decoder, "backend", "injected_cpu_interface_decoder")
        self.to(device)

    @property
    def source_config(self) -> dict:
        result = self.config.record()
        result.update(
            actual_decoder_backend=self.backend,
            decoder_parameters=sum(p.numel() for p in self.decoder.parameters()),
            geometry_encoder_parameters=sum(p.numel() for p in self.encoder.parameters()) if self.encoder is not None else 0)
        return result

    def encode_geometry(self, geometry_cloud: Tensor, geometry_mask: Tensor | None = None) -> Tensor:
        if geometry_cloud.ndim == 2:
            geometry_cloud = geometry_cloud.unsqueeze(0)
            if geometry_mask is not None and geometry_mask.ndim == 1:
                geometry_mask = geometry_mask.unsqueeze(0)
        if self.config.geometry_mode == "id":
            raise ValueError("Diagnostic id needs stable per-CAD codes passed as context, not geometry")
        if self.encoder is None:
            if geometry_cloud.ndim != 3 or geometry_cloud.shape[-1] != 3:
                raise ValueError("geometry_cloud must have shape [B,P,3]")
            return geometry_cloud.new_zeros((len(geometry_cloud), self.config.descriptor_dim), dtype=torch.float32)
        return self.encoder(geometry_cloud, geometry_mask)

    def forward(self, xyz: Tensor, features: Tensor,
                geometry_cloud: Tensor | None = None, cad_index: Tensor | None = None,
                *, geometry_mask: Tensor | None = None, context: Tensor | None = None) -> Tensor:
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            raise ValueError("xyz must have shape [N,3]")
        if features.shape != (len(xyz), self.config.condition_dim) or features.device != xyz.device:
            raise ValueError("features must have shape [N,condition_dim] on the xyz device")
        if cad_index is not None and (cad_index.shape != (len(xyz),) or cad_index.dtype != torch.long
                                      or cad_index.device != xyz.device):
            raise ValueError("cad_index must be long [N] on the xyz device")
        if context is not None and geometry_cloud is not None:
            raise ValueError("Provide pre-encoded context or geometry_cloud, not both")
        if self.config.geometry_mode == "none":
            # Keep exactly the shape-model decoder layout, including zero slots.
            node_context = features.new_zeros((len(xyz), self.config.descriptor_dim), dtype=torch.float32)
        else:
            if context is None:
                if geometry_cloud is None:
                    raise ValueError("Shape mode requires a cloud/context; diagnostic id requires context")
                context = self.encode_geometry(geometry_cloud, geometry_mask)
            if context.ndim != 2 or context.shape[1] != self.config.descriptor_dim or len(context) < 1:
                raise ValueError("context must have shape [unique_CADs,descriptor_dim]")
            if context.device != xyz.device:
                raise ValueError("Context, coordinates, and features must share a device")
            if self.training and torch.is_grad_enabled() and self.config.geometry_mode == "shape" and not context.requires_grad:
                raise ValueError("Training context was detached: call encode_geometry without no_grad/detach")
            if cad_index is None:
                if len(context) != 1:
                    raise ValueError("Multiple CAD contexts require per-node cad_index")
                node_context = context.expand(len(xyz), -1)
            else:
                node_context = context.index_select(0, cad_index)
        conditioned = torch.cat((features.float(), node_context.float()), dim=-1)
        return self.decoder(xyz, conditioned)


@dataclass(frozen=True)
class GeometryCloud:
    points: np.ndarray
    mask: np.ndarray
    node_indices: np.ndarray
    cad_key: str
    source: str
    pool: str
    candidate_nodes: int
    seed: int

    def record(self) -> dict:
        return {"cad_key": self.cad_key, "source": self.source, "pool": self.pool,
                "candidate_nodes": self.candidate_nodes, "selected_nodes": int(self.mask.sum()),
                "cloud_points": len(self.points), "seed": self.seed,
                "coordinate_system": "prepared normalized xyz; isotropic and aspect-ratio preserving",
                "target_independent": True, "all_loads_share_one_cad_cloud": True,
                "sampling": "uniform unique FEM nodes, not uniform surface area"}


def _cad_seed(cad_key: str, seed: int) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}:{cad_key}".encode("utf-8"), digest_size=8).digest(), "little")


def load_geometry_cloud(path: str | Path, *, cloud_points: int = 1024,
                        seed: int = 2026, cad_key: str | None = None) -> GeometryCloud:
    """Read xyz/faces/key only; never access stress, u, features, or split targets.

    Choose one canonical prepared file per CAD, cache this cloud, and reuse it
    for every load. No target-based sampling, saliency, or centroid is used.
    """
    if cloud_points < 1:
        raise ValueError("cloud_points must be positive")
    path = Path(path)
    with np.load(path, allow_pickle=False) as stored:
        xyz = np.asarray(stored["xyz"], dtype=np.float32)
        key = str(stored["sample_key"].item())
        faces = np.asarray(stored["faces"]) if "faces" in stored.files else None
    cad = cad_key or key.rsplit("/", 1)[0]
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 1 or not np.isfinite(xyz).all():
        raise ValueError("Prepared xyz must be finite nonempty [N,3]")
    if np.min(xyz) < -1e-5 or np.max(xyz) > 1 + 1e-5:
        raise ValueError("Use prepared normalized geometry coordinates in [0,1]")
    if faces is not None and faces.size:
        if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
            raise ValueError("Prepared boundary faces must be integer [F,3]")
        candidates = np.unique(faces)
        if candidates[0] < 0 or candidates[-1] >= len(xyz):
            raise ValueError("Boundary node IDs do not index prepared xyz")
        pool = "unique original quadratic-boundary node IDs"
    else:
        candidates = np.arange(len(xyz), dtype=np.int64)
        pool = "all original FEM nodes; boundary unavailable"
    count = min(cloud_points, len(candidates))
    chosen = np.random.default_rng(_cad_seed(cad, seed)).choice(candidates, count, replace=False)
    points = np.zeros((cloud_points, 3), dtype=np.float32)
    mask = np.zeros(cloud_points, dtype=np.bool_)
    indices = np.full(cloud_points, -1, dtype=np.int64)
    points[:count], mask[:count], indices[:count] = xyz[chosen], True, chosen
    return GeometryCloud(points, mask, indices, cad, str(path.resolve()), pool, len(candidates), seed)


def diagnostic_case_codes(cad_keys: list[str], *, descriptor_dim: int = 32,
                          seed: int = 2026, device: torch.device | str = "cpu") -> Tensor:
    """Stable fixed identity codes, not learned geometry and not a formal method."""
    if descriptor_dim < 1 or len(set(cad_keys)) != len(cad_keys) or not cad_keys:
        raise ValueError("Use positive descriptor width and unique nonempty CAD keys")
    values = np.stack([np.random.default_rng(_cad_seed(str(key), seed)).uniform(-1, 1, descriptor_dim)
                       for key in cad_keys]).astype(np.float32)
    return torch.from_numpy(values).to(device)


__all__ = ["ConditionedFieldConfig", "PointNetGeometryEncoder", "ConditionedField",
           "GeometryCloud", "load_geometry_cloud", "diagnostic_case_codes"]
