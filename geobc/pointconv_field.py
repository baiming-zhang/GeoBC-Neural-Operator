"""Full-original-node PointConv U-Net: a new GeoBC solver screen.

Seven distinct density-scaled coordinate convolution kernels implement a
three-resolution point encoder/decoder. The supplied hierarchy is geometry
only. This adapts PointConv factorization/deconvolution, not an exact author
segmentation replay, mesh message-passing model or off-node field contract.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .conditioned_field import PointNetGeometryEncoder

CLOUD_POINTS = 1024
CONTEXT_DIM = 32
FEATURE_DIM = 25
NEIGHBORS = 32
KERNEL_DIM = 16
QUERY_CHUNK = 2048
ANCHOR_COUNTS = (2048, 128)
PC_CHANNELS = {
    "fine": (32, 32), "down01": (32, 64), "mid": (64, 64),
    "down12": (64, 128), "bottom": (128, 128),
    "up_mid": (128, 64), "up_fine": (64, 32),
}
NEIGHBOR_LEVELS = {"neighbors00": (0, 0), "neighbors01": (0, 1),
                   "neighbors11": (1, 1), "neighbors12": (1, 2),
                   "neighbors22": (2, 2)}
INTERPOLATION_LEVELS = {"interp21": (2, 1), "interp10": (1, 0)}
OUTPUT_CHANNELS = ("ux_mm", "uy_mm", "uz_mm", "source_signed_stress_MPa")
POINTCONV_PAPER = "https://arxiv.org/abs/1811.07246"
TF_COMMIT = "f39dc3e101af2f52544181ee20c14f73279b48ae"
PYTORCH_COMMIT = "4a01ee9e1742d4b837ab0ef44896be4e31c63378"


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _scaler_metadata(scaler: dict) -> dict:
    """Validate supplied shared TRAIN-only JSON; do not read or fit targets."""
    if (not isinstance(scaler, dict) or scaler.get("channels") != [0, 1, 2, 3]
            or len(scaler.get("mean", [])) != 4 or len(scaler.get("std", [])) != 4):
        raise ValueError("Expected the existing four-field TRAIN-only scaler")
    mean, std = [float(v) for v in scaler["mean"]], [float(v) for v in scaler["std"]]
    stress_scale = float(scaler.get("stress_scale", 0.))
    if (not all(math.isfinite(v) for v in mean+std+[stress_scale])
            or min(std) <= 0 or stress_scale <= 0):
        raise ValueError("Invalid shared signed-field scaler metadata")
    return {"mean": mean, "std": std, "stress_scale": stress_scale,
            "scaler_sha256": _digest(scaler), "scaler_fits": 0}


class _PointConv(nn.Module):
    """Factorized convolution: (density-scaled features)^T @ spatial basis."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_net = nn.Sequential(
            nn.Linear(3, 8), nn.LayerNorm(8, elementwise_affine=False), nn.ReLU(),
            nn.Linear(8, 8), nn.LayerNorm(8, elementwise_affine=False), nn.ReLU(),
            nn.Linear(8, KERNEL_DIM), nn.ReLU())
        # No normalization anywhere in this scalar-input density branch.
        # In particular, LN of its one-channel output would erase its signal.
        self.density_net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(),
                                         nn.Linear(16, 1), nn.Sigmoid())
        self.mix = nn.Linear(in_channels*KERNEL_DIM, out_channels)
        self.output_norm = nn.LayerNorm(out_channels, elementwise_affine=False)
        self.output_activation = nn.ReLU()

    def _query_chunk(self, source_features: Tensor, source_xyz: Tensor,
                     query_xyz: Tensor, source_density: Tensor,
                     neighbor_indices: Tensor) -> Tensor:
        queries = len(query_xyz)
        flat_indices = neighbor_indices.reshape(-1)
        features = source_features.index_select(0, flat_indices).reshape(
            queries, NEIGHBORS, self.in_channels)
        delta = source_xyz.index_select(0, flat_indices).reshape(
            queries, NEIGHBORS, 3)-query_xyz.unsqueeze(1)
        inverse = source_density.index_select(0, flat_indices).reciprocal().reshape(
            queries, NEIGHBORS, 1)
        ratio = inverse/inverse.amax(dim=1, keepdim=True)
        scale = self.density_net(ratio)
        spatial_basis = self.weight_net(delta)
        # Feature-channel-major flattening: [query,Cin,16] -> [query,Cin*16].
        # Never materialize [query,K,Cin,Cout] dynamic filters.
        moment = torch.bmm((features*scale).transpose(1, 2), spatial_basis)
        values = self.mix(moment.reshape(queries, self.in_channels*KERNEL_DIM))
        return self.output_activation(self.output_norm(values))

    def forward(self, source_features: Tensor, source_xyz: Tensor,
                query_xyz: Tensor, source_density: Tensor,
                neighbor_indices: Tensor) -> Tensor:
        outputs = []
        for start in range(0, len(query_xyz), QUERY_CHUNK):
            end = min(start+QUERY_CHUNK, len(query_xyz))
            args = (source_features, source_xyz, query_xyz[start:end],
                    source_density, neighbor_indices[start:end])
            # All feature gathers, relative positions, density/kernel MLPs,
            # bmm, channel mixing and normalization are inside checkpoint.
            if self.training and torch.is_grad_enabled():
                values = checkpoint(self._query_chunk, *args, use_reentrant=False,
                                    preserve_rng_state=False)
            else:
                values = self._query_chunk(*args)
            outputs.append(values)
        # One final output per resolution; no full-N accumulator per chunk.
        return torch.cat(outputs, dim=0)


def _interpolation_chunk(source_features: Tensor, indices: Tensor,
                         weights: Tensor) -> Tensor:
    neighbors = source_features.index_select(0, indices.reshape(-1)).reshape(
        len(indices), 3, source_features.shape[1])
    return (neighbors*weights.unsqueeze(-1)).sum(dim=1)


def _interpolate(source_features: Tensor, indices: Tensor, weights: Tensor,
                 *, checkpoint_enabled: bool) -> Tensor:
    outputs = []
    for start in range(0, len(indices), QUERY_CHUNK):
        end = min(start+QUERY_CHUNK, len(indices))
        args = (source_features, indices[start:end], weights[start:end])
        if checkpoint_enabled and torch.is_grad_enabled():
            values = checkpoint(_interpolation_chunk, *args, use_reentrant=False,
                                preserve_rng_state=False)
        else:
            values = _interpolation_chunk(*args)
        outputs.append(values)
    return torch.cat(outputs, dim=0)


class _PointConvSolver(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(60, 64), nn.ReLU(), nn.Linear(64, 32))
        self.convolutions = nn.ModuleDict(
            {name: _PointConv(*channels) for name, channels in PC_CHANNELS.items()})
        self.fuse_mid = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.fuse_fine = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.readout = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 4))

    def encode(self, point_inputs: Tensor, hierarchy: Mapping) -> Tensor:
        h = hierarchy
        c = self.convolutions
        xyz0, xyz1, xyz2 = h["xyz0"], h["xyz1"], h["xyz2"]
        p0, p1, p2 = h["p0"], h["p1"], h["p2"]
        stem = self.stem(point_inputs)
        # An exact original-row identity path survives the first fine operator.
        fine_skip = stem+c["fine"](stem, xyz0, xyz0, p0, h["neighbors00"])
        mid = c["down01"](fine_skip, xyz0, xyz1, p0, h["neighbors01"])
        mid_skip = c["mid"](mid, xyz1, xyz1, p1, h["neighbors11"])
        bottom = c["down12"](mid_skip, xyz1, xyz2, p1, h["neighbors12"])
        bottom = c["bottom"](bottom, xyz2, xyz2, p2, h["neighbors22"])
        # Interpolate to the finer point set first, then convolve on that set.
        # Consequently up_mid uses p1/NN11, and up_fine uses p0/NN00.
        mid_interpolated = _interpolate(bottom, h["interp21_indices"], h["interp21_weights"],
                                       checkpoint_enabled=self.training)
        mid_up = c["up_mid"](mid_interpolated, xyz1, xyz1, p1, h["neighbors11"])
        mid_decoded = self.fuse_mid(torch.cat((mid_up, mid_skip), dim=-1))
        fine_interpolated = _interpolate(mid_decoded, h["interp10_indices"], h["interp10_weights"],
                                        checkpoint_enabled=self.training)
        fine_up = c["up_fine"](fine_interpolated, xyz0, xyz0, p0, h["neighbors00"])
        return self.fuse_fine(torch.cat((fine_up, fine_skip), dim=-1))


class PointConvField(nn.Module):
    """Full-case encode then chunkable signed encoded readout.

    H0 is the only live input needed by readout. A trainer may accumulate all
    original-node head losses through an H0 leaf, then backpropagate its whole
    gradient once through the original H0. No feature/state cache survives a
    training step in this module. Arbitrary off-node queries are not exposed.
    """

    def __init__(self, scaler: dict, seed: int = 2026,
                 device: str | torch.device = "cuda") -> None:
        super().__init__()
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        self.seed = seed
        self.method = "pointconv_unet"
        self.backend = "torch_factorized_full_node_pointconv"
        self._scaler = _scaler_metadata(scaler)
        torch.manual_seed(seed)
        # Same old xyz-only encoder constructor/order as other solver screens;
        # there is no full-model/capacity matching claim across families.
        self.encoder = PointNetGeometryEncoder(descriptor_dim=32, point_width=64)
        self.solver = _PointConvSolver()
        self.to(device)

    def parameter_counts(self) -> dict:
        encoder = sum(p.numel() for p in self.encoder.parameters())
        stem = sum(p.numel() for p in self.solver.stem.parameters())
        blocks = {}
        for name, block in self.solver.convolutions.items():
            blocks[name] = {"weight_net": sum(p.numel() for p in block.weight_net.parameters()),
                            "density_net": sum(p.numel() for p in block.density_net.parameters()),
                            "mix": sum(p.numel() for p in block.mix.parameters()),
                            "nonaffine_LayerNorm": 0,
                            "total": sum(p.numel() for p in block.parameters())}
        fusion = sum(p.numel() for p in self.solver.fuse_mid.parameters())+sum(
            p.numel() for p in self.solver.fuse_fine.parameters())
        head = sum(p.numel() for p in self.solver.readout.parameters())
        return {"PointNet": encoder, "stem": stem, "PointConv_blocks": blocks,
                "PointConv_total": sum(row["total"] for row in blocks.values()),
                "fusion": fusion, "readout": head,
                "solver": sum(p.numel() for p in self.solver.parameters()),
                "total": sum(p.numel() for p in self.parameters()),
                "all_parameters_real": True, "complex_parameters": 0,
                "count_policy": "actual public Parameter.numel; not capacity matched"}

    def gradient_components(self) -> dict[str, nn.Module]:
        # Temporary ModuleLists expose .parameters() without registering aliases
        # into this model's named_parameters, state_dict, or optimizer groups.
        c = self.solver.convolutions
        return {"encoder": self.encoder, "stem": self.solver.stem, "fine": c["fine"],
                "down": nn.ModuleList([c["down01"], c["down12"]]),
                "middle": c["mid"], "bottom": c["bottom"],
                "decode": nn.ModuleList([c["up_mid"], c["up_fine"]]),
                "fusion": nn.ModuleList([self.solver.fuse_mid, self.solver.fuse_fine]),
                "readout": self.solver.readout}

    @property
    def source_config(self) -> dict:
        own_path = Path(__file__).resolve()
        encoder_path = own_path.with_name("conditioned_field.py")
        return {
            "method": self.method, "backend": self.backend, "seed": self.seed,
            "source": "geobc/pointconv_field.py",
            "source_sha256": hashlib.sha256(own_path.read_bytes()).hexdigest(),
            "encoder_source": "geobc/conditioned_field.py:PointNetGeometryEncoder",
            "encoder_source_sha256": hashlib.sha256(encoder_path.read_bytes()).hexdigest(),
            "parameter_counts": self.parameter_counts(), "scaler": copy.deepcopy(self._scaler),
            "input": {"node": "canonical normalized xyz3 + all original25 + xyz-cloud global32",
                      "stem_dimensions": [60, 64, 32], "force_slots": list(range(7, 13)),
                      "force_slots_retained": True, "cloud_shape": [1, 1024, 3],
                      "cloud_target_independent": True, "cloud_load_independent": True,
                      "node_order": "all original aligned FEM rows; repeated xyz rows kept distinct"},
            "encoder": {"point_mlp": [3, 32, 64, 64], "head": [128, 64, 32],
                        "activation": "ReLU", "pooling": "masked max + masked mean",
                        "output": "non-affine LN32 then tanh"},
            "point_hierarchy": {"anchor_counts": list(ANCHOR_COUNTS),
                                "anchors": "two pure-geometry FPS original-row subsets, not connected mesh aggregates",
                                "FPS": "O(N) running minimum; original ID ties; no N2 or per-iteration item synchronization",
                                "neighbors": 32, "neighbors_layout": dict(NEIGHBOR_LEVELS),
                                "self_or_anchor_row_first": True, "query_order_preserved": True,
                                "density_cache": "p0,p1,p2 are geometry-only source-density values",
                                "source_bandwidth": "sigma_l=median source-layer d32; positive or preparation fails",
                                "KDE": "mean of exp(-d2/(2 sigma_l2)) over self+31 source neighbors",
                                "KDE_adaptation": "finite KNN KDE; not original all-point Gaussian density",
                                "distance_precision": "stored canonical float32 coordinates promoted to float64 in input preparation",
                                "model_relative_xyz": "source_xyz[neighbor]-query_xyz in canonical /L frame; no second /L",
                                "interpolation": "cached 3NN; exact-zero neighbors only and equally weighted, otherwise normalized 1/d",
                                "coarse_coverage": "stride KNN does not guarantee every source reaches coarse layers",
                                "fine_coverage": "every original node has fine convolution, exact row identity path, skip and output"},
            "architecture": {"seven_independent_PC_channels": dict(PC_CHANNELS),
                             "fine_identity": "fine_skip=stem+PC32to32(stem)",
                             "encoder": ["fine32", "FPS2048 PC32to64", "midPC64to64",
                                         "FPS128 PC64to128", "bottomPC128to128"],
                             "decoder": ["interp2to1 then PC128to64 on level1 then concat mid64 skip/Linear128to64/ReLU",
                                         "interp1to0 then PC64to32 on level0 then concat fine32 skip/Linear64to32/ReLU"],
                             "PointDeconv_order": "interpolation -> fine-coordinate PointConv -> concat skip -> pointwise fusion",
                             "V_cycles": False, "u9_response_basis": False,
                             "PointConv_feature_preMLP": False},
            "convolution": {"factorization": "Z=(s(r)*neighbor_features)^T@M(delta); flatten Cin-major x16; Linear(Cin*16,Cout)",
                            "density_ratio": "r=inverse_p_neighbors/max inverse_p_neighbors per query",
                            "density_dimensions": [1, 16, 1], "density_hidden": "ReLU",
                            "density_output": "sigmoid", "density_LN_or_BN": False,
                            "density_floor_or_target_fit": False,
                            "weight_dimensions": [3, 8, 8, 16],
                            "weight_hidden": "non-affine LN8 then ReLU at each of two hidden layers",
                            "weight_output": "ReLU, without normalization",
                            "mixed_output": "non-affine LN(Cout) then ReLU",
                            "extra_receiver_feature_message": False, "E_Cin_Cout_dynamic_filters": False},
            "output": {"head_dimensions": [32, 128, 4], "hidden_activation": "ReLU", "final_activation": "linear",
                       "channels": list(OUTPUT_CHANNELS),
                       "values": "shared TRAIN-scaler encoded signed4; trainer decodes physical mm/MPa",
                       "stress_linear_or_constitutive_formula": False},
            "memory": {"query_chunk": QUERY_CHUNK,
                       "PointConv_checkpoint": "non-reentrant around gather/delta/density/kernel/bmm/mix/norm",
                       "interpolation_checkpoint": "non-reentrant gather/weighted sum per query chunk",
                       "preserve_rng_state": False, "stochastic_layers": False,
                       "states": "one complete latent per used resolution; chunk outputs concatenate, with transient result allocation",
                       "per_chunk_full_N_buffer": False, "dense_N2": False, "E_Cin_Cout": False,
                       "neighbors_dropped": False, "input_points_dropped": False,
                       "H0_leaf_chainrule": "only final H0 needed; complete chunked readout grads then one original H0 backward",
                       "actual_GPU_memory_and_checkpoint_backward": "deferred to first formal complete-case step"},
            "precision": "float32 model/input, caller common TF32/high/no-autocast policy; no AMP in this module",
            "references": {"paper": POINTCONV_PAPER,
                           "author_TF": f"https://github.com/DylanWusee/pointconv/blob/{TF_COMMIT}/PointConv.py",
                           "author_PyTorch": f"https://github.com/DylanWusee/pointconv_pytorch/blob/{PYTORCH_COMMIT}/utils/pointconv_util.py",
                           "adaptations": ["finite local KDE and geometry-dependent fixed bandwidth rule",
                                           "channelwise non-affine LN replaces cross-point BN in weight/mixed feature paths",
                                           "TF sigmoid scalar density path without BN/LN",
                                           "factorized direct input features without author SA extra feature preMLP",
                                           "fullN checkpointed queries and interpolation", "original-node stem identity skip"],
                           "author_PyTorch_scope": "classification encoder only; not an official complete segmentation decoder"},
            "native_offnode": False,
            "limitations": ["new PointConv adaptation, not exact official replay or novel PointConv mechanism",
                            "no strict capacity or full-head initialization matching with prior solver families",
                            "Euclidean neighbors may cross thin walls/holes; coarse anchors can omit small role patches",
                            "role membership is not node Dirichlet; density/kernel is not FEM stiffness or exact quadrature",
                            "no arbitrary off-node original25 semantics, continuous derivative, PDE or engineering guarantee",
                            "source readiness does not demonstrate training resources, accuracy or gradient quality"],
        }

    def _validate_hierarchy(self, xyz: Tensor, hierarchy: Mapping) -> None:
        if not isinstance(hierarchy, Mapping):
            raise ValueError("Expected target-independent point hierarchy")
        device = next(self.solver.parameters()).device
        sizes = []
        for level in range(3):
            coordinates = hierarchy.get(f"xyz{level}")
            if (not isinstance(coordinates, Tensor) or coordinates.ndim != 2 or coordinates.shape[1] != 3
                    or len(coordinates) < NEIGHBORS or coordinates.dtype != torch.float32
                    or coordinates.device != device):
                raise ValueError(f"xyz{level} must be float32 [Ni,3] with at least32 points")
            sizes.append(len(coordinates))
            density = hierarchy.get(f"p{level}")
            if (not isinstance(density, Tensor) or density.shape != (len(coordinates),)
                    or density.dtype != torch.float32 or density.device != device):
                raise ValueError(f"p{level} must be float32 [Ni] source geometry density")
        if sizes != [len(xyz), *ANCHOR_COUNTS]:
            raise ValueError("Expected all original N rows followed by actual2048 and128 anchors")
        for name, (_, query_level) in NEIGHBOR_LEVELS.items():
            indices = hierarchy.get(name)
            if (not isinstance(indices, Tensor) or indices.shape != (sizes[query_level], NEIGHBORS)
                    or indices.dtype != torch.int64 or indices.device != device):
                raise ValueError(f"{name} must be int64 [Nquery,32] on model device")
        for name, (_, query_level) in INTERPOLATION_LEVELS.items():
            indices, weights = hierarchy.get(name+"_indices"), hierarchy.get(name+"_weights")
            shape = (sizes[query_level], 3)
            if (not isinstance(indices, Tensor) or indices.shape != shape or indices.dtype != torch.int64
                    or indices.device != device or not isinstance(weights, Tensor) or weights.shape != shape
                    or weights.dtype != torch.float32 or weights.device != device):
                raise ValueError(f"{name} requires int64 indices and float32 weights [Nquery,3]")
        # Bounds, row uniqueness, positive density, complete declared32 support,
        # interpolation row sums and source/xyz identities belong to immutable
        # producer/loader guards. Do not repeat whole input scans each forward.

    def encode_case(self, xyz: Tensor, features25: Tensor, oldcloud: Tensor,
                    mask: Tensor, hierarchy: Mapping) -> Tensor:
        device = next(self.solver.parameters()).device
        if (not isinstance(xyz, Tensor) or xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < ANCHOR_COUNTS[0]
                or not xyz.is_floating_point() or xyz.device != device
                or not isinstance(features25, Tensor) or features25.shape != (len(xyz), 25)
                or not features25.is_floating_point() or features25.device != device
                or not isinstance(oldcloud, Tensor) or oldcloud.shape != (1, CLOUD_POINTS, 3)
                or not oldcloud.is_floating_point() or oldcloud.device != device
                or not isinstance(mask, Tensor) or mask.shape != (1, CLOUD_POINTS)
                or mask.dtype != torch.bool or mask.device != device):
            raise ValueError("Expected original xyz[N,3], all features25[N,25], cloud[1,1024,3], bool mask")
        self._validate_hierarchy(xyz, hierarchy)
        context = self.encoder(oldcloud.float(), mask)
        inputs = torch.cat((xyz.float(), features25.float(), context.expand(len(xyz), -1)), dim=-1)
        states = self.solver.encode(inputs, hierarchy)
        if states.shape != (len(xyz), 32):
            raise ValueError("Final full-node H0 shape changed")
        return states

    def readout(self, H0: Tensor) -> Tensor:
        if (not isinstance(H0, Tensor) or H0.ndim != 2 or H0.shape[1] != 32 or len(H0) < 1
                or not H0.is_floating_point() or H0.device != next(self.solver.parameters()).device):
            raise ValueError("readout requires final [N,32] states on model device")
        return self.solver.readout(H0.float())

    def forward_case(self, xyz: Tensor, features25: Tensor, oldcloud: Tensor,
                     mask: Tensor, hierarchy: Mapping) -> Tensor:
        return self.readout(self.encode_case(xyz, features25, oldcloud, mask, hierarchy))

    def forward(self, xyz: Tensor, features25: Tensor, oldcloud: Tensor,
                mask: Tensor, hierarchy: Mapping) -> Tensor:
        return self.forward_case(xyz, features25, oldcloud, mask, hierarchy)

