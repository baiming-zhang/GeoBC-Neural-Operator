"""Six GeoBC solver families with one shared geometry/BC input contract.

This is a new solver-selection configuration, not exact slide reproduction.
All families own the same PointNet xyz-cloud encoder. The trainer owns target
scalers/losses, case identity, point sampling and FNO grid/query preparation.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn

from .conditioned_field import PointNetGeometryEncoder

METHODS = ("mlp", "hashgrid", "transolver", "transolver_pp", "ts3", "fno")
ALIASES = {
    "mlp": "mlp", "hash": "hashgrid", "hashgrid": "hashgrid", "tcnn": "hashgrid",
    "transolver": "transolver", "transolver_original": "transolver",
    "transolverpp": "transolver_pp", "transolver++": "transolver_pp",
    "transolver_pp": "transolver_pp", "transolver_plus": "transolver_pp",
    "ts3": "ts3", "transolver3": "ts3", "transolver-3": "ts3",
    "transolver_3": "ts3", "fno": "fno", "fno3d": "fno",
}
OUTPUT_CHANNELS = ("ux_mm", "uy_mm", "uz_mm", "signed_stress_MPa")
CLOUD_POINTS = 1024
CONTEXT_DIM = 32
FEATURE_DIM = 25
POINT_INPUT_DIM = 60
GRID_INPUT_DIM = 29
FNO_INPUT_DIM = 61
FNO_RESOLUTION = 64


def _canonical(method: str) -> str:
    try:
        return ALIASES[method.strip().lower()]
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"Unknown solver family; choose one of {METHODS}") from exc


def _source_sha(name: str) -> str:
    return hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()


class SelectionModel(nn.Module):
    """encode_geometry(cloud,mask), then point or FNO-grid forward.

Pointwise MLP/Hash may be called in chunks with the same live context graph.
Transformer forward_points receives the entire case; TS3 chunks internally
with upstream global states at every layer. FNO requires forward_grid and
the trainer's geometry-only node-query cache, not independent point calls.
"""

    def __init__(self, method: str, device: torch.device | str, seed: int) -> None:
        super().__init__()
        self.method = _canonical(method)
        self.seed = int(seed)
        device = torch.device(device)
        # Same creation order/seed gives the same initial PointNet in every family.
        torch.manual_seed(self.seed)
        self.encoder = PointNetGeometryEncoder(descriptor_dim=CONTEXT_DIM, point_width=64)
        if self.method == "mlp":
            layers = []
            incoming = POINT_INPUT_DIM
            for _ in range(4):
                layers.extend((nn.Linear(incoming, 256), nn.ReLU()))
                incoming = 256
            layers.append(nn.Linear(256, 4))
            self.solver = nn.Sequential(*layers)
            backbone = {"family": "plain_coordinate_BC_global_MLP", "hidden_width": 256,
                        "hidden_layers": 4, "activation": "ReLU", "input_dim": 60,
                        "output_dim": 4, "hash_encoding": False, "backend": "torch"}
        elif self.method == "hashgrid":
            from .tcnn_fields import TCNNField, TCNNFieldConfig
            config = TCNNFieldConfig(out_channels=4, input_mode="both", condition_dim=57,
                                     n_levels=12, n_features_per_level=2, log2_hashmap_size=19,
                                     base_resolution=8, per_level_scale=1.6,
                                     hidden_dim=64, hidden_layers=2, seed=self.seed)
            self.solver = TCNNField(config=config, device=device)
            backbone = config.record()
            backbone["backend"] = self.solver.backend
        elif self.method in ("transolver", "transolver_pp", "ts3"):
            from .transformers import build_transformer
            self.solver = build_transformer(self.method, input_dim=POINT_INPUT_DIM,
                                            output_dim=4, device=device,
                                            checkpoint_blocks=True, chunk_size=32768)
            backbone = copy.deepcopy(self.solver.source_config)
            # Upstream wrapper has historical28-channel descriptive text; actual
            # constructor dimensions are unchanged here except fun_dim57/out4.
            backbone["input"] = "canonical xyz3 + original BC/material25 + shared PointNet global32"
            backbone["output"] = "Four trainer-encoded signed physical field channels"
        else:
            from .fno import FNO3d
            self.solver = FNO3d(in_channels=FNO_INPUT_DIM, out_channels=4, width=16,
                                modes=12, layers=4, padding=0, head_width=128,
                                checkpoint_blocks=True)
            backbone = self.solver.config.record()
            backbone["xyz_channels"] = True
            backbone["input_schema"] = "occupancy1 + canonical xyz3 + BC/material25 + shared global32"
            backbone["resolution"] = FNO_RESOLUTION
        self.to(device)
        self.parameter_counts = {
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "solver": sum(p.numel() for p in self.solver.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
        self.capabilities = {
            "forward_points": self.method != "fno",
            "forward_grid": self.method == "fno",
            "pointwise_external_chunks": self.method in ("mlp", "hashgrid"),
            "whole_case_point_context": self.method in ("transolver", "transolver_pp", "ts3"),
            "native_global_chunks": self.method == "ts3",
            "eval_stochastic": self.method == "transolver_pp",
            "native_continuous_query_implemented": False,
        }
        self.source_config = {
            "method": self.method, "purpose": "New six-family geometry/BC solver selection",
            "exact_slides_reproduction": False, "seed": self.seed,
            "adapter_source_sha256": _source_sha("solver_selection_models.py"),
            "geometry_encoder": {
                "class": "PointNetGeometryEncoder", "source": "geobc/conditioned_field.py",
                "source_sha256": _source_sha("conditioned_field.py"),
                "cloud_shape": ["B", CLOUD_POINTS, 3], "mask_shape": ["B", CLOUD_POINTS],
                "mask_dtype": "bool", "point_width": 64, "descriptor_dim": 32,
                "coordinates": "Prepared canonical xyz only; no role channels, targets or case ID",
                "shared_architecture_and_seed_across_all_families": True,
            },
            "input_schema": {
                "xyz": ["N", 3], "BC_material": ["N", 25], "context": [1, 32],
                "points": "xyz3 + original25 + global32; total60",
                "hash": "xyz3 encoded by native Hash12; original25+global32 identity condition_dim57",
                "FNO_grid_supplied": [1, 29, 64, 64, 64],
                "FNO_grid_channels": "occupancy0; xyz1:4; original25 BC/material4:29",
                "FNO_internal": [1, 61, 64, 64, 64],
                "FNO_global_broadcast": "Append32 to every cube voxel, including occupancy0 voxels",
            },
            "output_schema": {
                "points": ["N", 4], "grid": [1, 4, 64, 64, 64],
                "channel_order": list(OUTPUT_CHANNELS),
                "scaling": "Encoded signed fields; trainer owns common train-only scaler/decode",
                "activation": "Unrestricted output; no abs, clipping or stress replacement",
            },
            "backbone": backbone, "parameter_counts": self.parameter_counts,
            "capabilities": self.capabilities,
            "precision": "PointNet/torch solvers float32 inputs; Hash native FP16 compute/FP32 master/float32 output; trainer owns autocast/TF32 policy",
            "context_lifetime": "Recompute encoder each optimizer update; reuse same live context across point chunks and accumulate its gradients",
            "distributed_cleanup": "Trainer finally calls close_selection_process_group; closes only adapter-owned ++ group",
            "no_target_scaler_or_case_identity_in_model": True,
        }

    def encode_geometry(self, cloud: Tensor, mask: Tensor) -> Tensor:
        if cloud.ndim != 3 or cloud.shape[1:] != (CLOUD_POINTS, 3) or not cloud.is_floating_point():
            raise ValueError("cloud must be floating [B,1024,3] canonical xyz")
        if mask.shape != cloud.shape[:2] or mask.dtype != torch.bool or mask.device != cloud.device:
            raise ValueError("mask must be bool [B,1024] on the cloud device")
        if cloud.device != next(self.encoder.parameters()).device:
            raise ValueError("Cloud and geometry encoder must share a device")
        return self.encoder(cloud, mask)

    def _context(self, context: Tensor, device: torch.device) -> Tensor:
        if context.shape != (1, CONTEXT_DIM) or not context.is_floating_point() or context.device != device:
            raise ValueError("Per-case context must be floating [1,32] on the input device")
        return context.float()

    def forward_points(self, xyz: Tensor, features: Tensor, context: Tensor) -> Tensor:
        if self.method == "fno":
            raise ValueError("FNO needs forward_grid plus the trainer's geometry-only node-query cache")
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not xyz.is_floating_point() or len(xyz) == 0:
            raise ValueError("xyz must be nonempty floating [N,3]")
        if features.shape != (len(xyz), FEATURE_DIM) or not features.is_floating_point() or features.device != xyz.device:
            raise ValueError("features must be floating [N,25] on the xyz device")
        if xyz.device != next(self.encoder.parameters()).device:
            raise ValueError("Point inputs and model must share a device")
        global_features = self._context(context, xyz.device).expand(len(xyz), -1)
        conditions = torch.cat((features.float(), global_features), dim=-1)
        if self.method == "hashgrid":
            result = self.solver(xyz, conditions)
        else:
            inputs = torch.cat((xyz.float(), conditions), dim=-1)
            result = self.solver(inputs) if self.method == "mlp" else self.solver(inputs.unsqueeze(0)).squeeze(0)
        if result.shape != (len(xyz), 4):
            raise RuntimeError("Solver must return [N,4] in the declared channel order")
        return result

    def forward_grid(self, grid: Tensor, context: Tensor) -> Tensor:
        if self.method != "fno":
            raise ValueError("forward_grid is only available for the FNO family")
        expected = (1, GRID_INPUT_DIM, FNO_RESOLUTION, FNO_RESOLUTION, FNO_RESOLUTION)
        if grid.shape != expected or not grid.is_floating_point():
            raise ValueError("FNO frontend must supply floating grid[1,29,64,64,64]")
        if grid.device != next(self.encoder.parameters()).device:
            raise ValueError("Grid and model must share a device")
        global_grid = self._context(context, grid.device)[:, :, None, None, None].expand(
            1, CONTEXT_DIM, FNO_RESOLUTION, FNO_RESOLUTION, FNO_RESOLUTION)
        result = self.solver(torch.cat((grid.float(), global_grid), dim=1))
        if result.shape != (1, 4, 64, 64, 64):
            raise RuntimeError("FNO must return grid[1,4,64,64,64]")
        return result

    def forward(self, xyz: Tensor, features: Tensor, context: Tensor) -> Tensor:
        return self.forward_points(xyz, features, context)


def build_selection_model(method: str, device: torch.device | str = "cuda", seed: int = 2026) -> SelectionModel:
    """Build the fixed family preset; no silent backend substitution or probes."""
    return SelectionModel(method, device, seed)


def close_selection_process_group() -> None:
    """Trainer finally cleanup; never destroy a process group owned elsewhere."""
    from .transformers import close_owned_process_group
    close_owned_process_group()


__all__ = ["METHODS", "SelectionModel", "build_selection_model", "close_selection_process_group",
           "OUTPUT_CHANNELS", "CLOUD_POINTS", "CONTEXT_DIM", "FEATURE_DIM", "FNO_RESOLUTION"]
