"""Geometry/BC DeepONet and a concat-MLP with common response semantics.

The trainer supplies a target-independent fixed1024 geometry/BC cloud, the
existing TRAIN-only scaler and rank3 load metadata. This module reads no data
or fields. Both modes share only the encoder construction/initialization;
their heads, counts and full-model initialization are different by design.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

MODES = ("deeponet", "concat_mlp")
NONFORCE_FEATURE_INDICES = tuple(range(7)) + tuple(range(13, 25))
LOAD_SCALES = [42300., 42300., 42300., 565000., 565000., 565000.]
RANK = 64
CLOUD_POINTS = 1024
CONTEXT_DIM = 32


def _digest(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _matrix(value, rows: int, columns: int, name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows or any(not isinstance(row, list) or len(row) != columns for row in value):
        raise ValueError(f"{name} must be {rows}x{columns}")
    result = [[float(entry) for entry in row] for row in value]
    if not all(math.isfinite(entry) for row in result for entry in row):
        raise ValueError(f"Nonfinite {name}")
    return result


def _metadata(scaler: dict, basis: dict) -> dict:
    """Validate supplied JSON metadata; never fit scales or inspect targets."""
    if scaler.get("channels") != [0, 1, 2, 3] or len(scaler.get("mean", [])) != 4 or len(scaler.get("std", [])) != 4:
        raise ValueError("Expected the existing four-field TRAIN-only scaler")
    mean, std = [float(x) for x in scaler["mean"]], [float(x) for x in scaler["std"]]
    if not all(math.isfinite(x) for x in mean+std) or min(std) <= 0:
        raise ValueError("Invalid shared scaler mean/std")
    stress_scale = scaler.get("stress_scale")
    if stress_scale is None or not math.isfinite(float(stress_scale)) or float(stress_scale) <= 0:
        raise ValueError("The shared signed-stress asinh scaler must be retained")
    if (basis.get("status") != "compatible" or basis.get("compatible") is not True
            or basis.get("CADs_checked") != 45 or basis.get("compatible_CADs") != 45 or basis.get("badcases") != []):
        raise ValueError("Expected the compatible45 TRAIN/DEV load-basis guard")
    canonical, mapping = basis["canonical_basis"], basis["prepared_feature_mapping"]
    if (canonical.get("order") != ["ver", "hor", "tor"] or canonical.get("exact_rank") != 3
            or canonical.get("physical_vector_order") != ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]):
        raise ValueError("Canonical rank3 physical load basis/order changed")
    if (mapping.get("slots") != list(range(7, 13)) or mapping.get("physical_scale") != LOAD_SCALES
            or mapping.get("alpha_order") != ["ver", "hor", "tor"]):
        raise ValueError("Original25 normalized F/M schema changed")
    alpha_matrix = _matrix(mapping["feature_to_alpha_matrix_3x6"], 3, 6, "feature-to-alpha")
    normalized_basis = _matrix(mapping["normalized_basis_6x3"], 6, 3, "normalized basis")
    physical_basis = _matrix(canonical["physical_matrix_6x3"], 6, 3, "physical basis")
    for i in range(3):
        for j in range(3):
            if abs(sum(alpha_matrix[i][k]*normalized_basis[k][j] for k in range(6))-float(i == j)) > 1e-12:
                raise ValueError("Alpha map is not a left inverse of the load basis")
    for i in range(6):
        for j in range(3):
            if abs(normalized_basis[i][j]-physical_basis[i][j]/LOAD_SCALES[i]) > 1e-12:
                raise ValueError("Physical and normalized load bases differ")
    tolerance = float(mapping["tolerance_normalized"])
    if tolerance != 1e-7:
        raise ValueError("Frozen normalized load-span tolerance changed")
    records = basis.get("cases", [])
    flags = ("compatible", "basis_matches_shared_reference", "constraints_identical",
             "zero_fixed_constraints", "material_identical_within_CAD", "load_interface_identical",
             "STATICS_without_nonlinear_entries", "prepared_slot_mapping_compatible")
    if (len(records) != 45 or len({row.get("cad_id") for row in records}) != 45
            or sum(row.get("group") == "train" for row in records) != 40
            or sum(row.get("group") == "validation" for row in records) != 5
            or any(row.get("basis_rank") != 3 or any(row.get(key) is not True for key in flags) for row in records)):
        raise ValueError("Metadata does not support the common static operator per TRAIN/DEV CAD")
    return {"mean": mean, "std": std, "stress_scale": float(stress_scale),
            "alpha_matrix": alpha_matrix, "normalized_basis": normalized_basis,
            "physical_basis": physical_basis, "load_span_tolerance": tolerance,
            "scaler_sha256": _digest(scaler), "basis_metadata_sha256": _digest(basis)}


@dataclass(frozen=True)
class DeepONetFieldConfig:
    mode: Literal["deeponet", "concat_mlp"] = "deeponet"
    seed: int = 2026

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Mode must be one of {MODES}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")

    def record(self) -> dict:
        return {**asdict(self), "rank": RANK, "cloud_points": CLOUD_POINTS,
                "cloud_channels": 22, "query_channels": 22, "global_context": CONTEXT_DIM,
                "encoder_point_mlp": [22, 32, 64, 64], "encoder_head": [128, 64, 32],
                "encoder_activation": "ReLU; non-affine LN32 and tanh at output",
                "head_activation": "GELU", "hash_grid": False, "voxel_grid": False,
                "deeponet_query_trunk": [22, 128, 128, 64],
                "deeponet_u_coefficients": [32, 128, 9*64],
                "deeponet_stress_coefficients": [38, 128, 64],
                "deeponet_dot_divisor": math.sqrt(RANK), "deeponet_stress_scalar_bias": True,
                "concat_query_trunk": [54, 128, 128, 128], "concat_u_head": [128, 9],
                "concat_stress_head": [134, 128, 1],
                "active_architecture": self.mode,
                "capacity_policy": "Different heads/parameter counts; no width/rank sweep or strict capacity matching"}


class GeometryBCPointNet(nn.Module):
    """Same set encoder in both modes; no current force/moment channels."""

    def __init__(self):
        super().__init__()
        self.point_mlp = nn.Sequential(nn.Linear(22, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32),
            nn.LayerNorm(32, elementwise_affine=False), nn.Tanh())

    def forward(self, cloud: Tensor, mask: Tensor) -> Tensor:
        values = self.point_mlp(cloud.float())
        expanded = mask.unsqueeze(-1)
        count = mask.sum(dim=1, keepdim=True)
        if bool(torch.any(count == 0)):
            raise ValueError("Cloud must have at least one valid point")
        maximum = values.masked_fill(~expanded, -torch.inf).amax(dim=1)
        average = (values*expanded).sum(dim=1)/count.to(values.dtype)
        return self.head(torch.cat((maximum, average), dim=-1))


class _DeepONetStressHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficients = nn.Sequential(nn.Linear(38, 128), nn.GELU(), nn.Linear(128, RANK))
        self.scalar_bias = nn.Parameter(torch.zeros(1))

    def forward(self, trunk: Tensor, context: Tensor, load6: Tensor) -> Tensor:
        coefficients = self.coefficients(torch.cat((context, load6), dim=-1))
        return trunk @ coefficients.transpose(0, 1)/math.sqrt(RANK) + self.scalar_bias


class _DeepONetSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_trunk = nn.Sequential(nn.Linear(22, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(), nn.Linear(128, RANK))
        self.u_basis_head = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 9*RANK))
        self.stress_head = _DeepONetStressHead()

    def forward(self, query22: Tensor, context: Tensor, load6: Tensor) -> tuple[Tensor, Tensor]:
        trunk = self.query_trunk(query22)
        coefficients = self.u_basis_head(context).reshape(9, RANK)
        basis = trunk @ coefficients.transpose(0, 1)/math.sqrt(RANK)
        return basis, self.stress_head(trunk, context, load6)


class _ConcatSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_trunk = nn.Sequential(nn.Linear(54, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.u_basis_head = nn.Linear(128, 9)
        self.stress_head = nn.Sequential(nn.Linear(134, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, query22: Tensor, context: Tensor, load6: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.query_trunk(torch.cat((query22, context.expand(len(query22), -1)), dim=-1))
        basis = self.u_basis_head(latent)
        stress = self.stress_head(torch.cat((latent, load6.expand(len(query22), -1)), dim=-1))
        return basis, stress


class DeepONetField(nn.Module):
    """Encode one CAD cloud, then query all nodes in arbitrary trainer chunks.

`load6` is the global normalized original25 features7:13, not raw mixed-unit
F/M. All differentiable encoder dependence is through the explicit context;
there is no hidden case-state/latent cache. A trainer may detach the context
to a leaf, accumulate full-node loss gradients and backpropagate it once.
"""

    def __init__(self, mode: str, scaler: dict, basis_metadata: dict, *,
                 seed: int = 2026, device: str | torch.device = "cuda"):
        super().__init__()
        self.config = DeepONetFieldConfig(mode=mode, seed=seed)
        self.mode, self.seed, self.method = mode, seed, "deeponet_bc_response"
        metadata = _metadata(scaler, basis_metadata)
        torch.manual_seed(seed)
        # Encoder is created first with identical shape/RNG; subsequent heads differ.
        self.encoder = GeometryBCPointNet()
        self.solver = _DeepONetSolver() if mode == "deeponet" else _ConcatSolver()
        self.register_buffer("u_mean", torch.tensor(metadata["mean"][:3], dtype=torch.float64))
        self.register_buffer("u_std", torch.tensor(metadata["std"][:3], dtype=torch.float64))
        self.register_buffer("u_encoded_zero", (-self.u_mean/self.u_std).to(torch.float32))
        self.register_buffer("feature_to_alpha", torch.tensor(metadata["alpha_matrix"], dtype=torch.float64))
        self.register_buffer("normalized_load_basis", torch.tensor(metadata["normalized_basis"], dtype=torch.float64))
        self.load_span_tolerance = metadata["load_span_tolerance"]
        self.metadata_record = metadata
        self.capabilities = {"forward_points": True, "forward_grid": False,
            "pointwise_external_chunks": True, "all_encoder_dependencies_explicit_context": True,
            "native_continuous_query_implemented": False,
            "zero_load_u": "Algebraic physical zero up to ordinary scaler/FP32 rounding, not boundary enforcement"}
        self.to(device)

    @property
    def parameter_counts(self) -> dict:
        encoder = sum(parameter.numel() for parameter in self.encoder.parameters())
        solver = sum(parameter.numel() for parameter in self.solver.parameters())
        return {"encoder": encoder, "solver": solver, "total": encoder+solver}

    def gradient_components(self) -> dict[str, nn.Module]:
        """Public modules for first-real-step checks, no guessed parameter offsets."""
        return {"encoder": self.encoder, "query_trunk": self.solver.query_trunk,
                "u_basis_head": self.solver.u_basis_head, "stress_head": self.solver.stress_head}

    @property
    def source_config(self) -> dict:
        return {"method": self.method, "mode": self.mode, "seed": self.seed,
            "configuration": self.config.record(), "backend": "PyTorch real FP32 trainables; no native/hash/FNO dependency",
            "parameter_counts": self.parameter_counts,
            "parameter_group_counts": {name: sum(p.numel() for p in module.parameters())
                for name, module in self.gradient_components().items()},
            "parameter_count_policy": "Actual public real trainable numel; buffers excluded; architectures intentionally differ",
            "geometry_encoder": {"class": "GeometryBCPointNet", "cloud_shape": [1, 1024, 22],
                "mask_shape": [1, 1024], "mask_dtype": "bool", "point_width": 64, "descriptor_dim": 32,
                "pooling": "Concatenated masked max and mean", "input": "xyz3 then original nonforce19 in canonical selected-node order",
                "index_identity": "New target-free data helper must verify true original cloud node indices/xyz identity and record BC-role coverage; no nearest-node copy or resampling"},
            "query_schema": {"xyz": ["N", 3], "features": ["N", 25], "query22": "xyz3 followed by features0:7 and features13:25",
                "excluded_feature_slots": list(range(7, 13)), "nonforce_feature_indices": list(NONFORCE_FEATURE_INDICES),
                "context": [1, 32], "load6": "Separate normalized global features7:13 = F/42300N and M/565000Nmm",
                "no_case_id_or_target_input": True},
            "output_schema": {"points": ["N", 4], "components": {"basis_nodes": ["N", 9], "stress_nodes": ["N", 1], "alpha": [1, 3]},
                "u_basis_order": ["ver_x", "ver_y", "ver_z", "hor_x", "hor_y", "hor_z", "tor_x", "tor_y", "tor_z"],
                "basis_units": "V[k,j]=physical basis displacement B[k,j] / frozen TRAIN std_j; zero-centered basis, not individually centered encoded fields",
                "u_composition": "encoded_u_j=sum_k alpha_k V_kj - mean_j/std_j once; physical_u_j=std_j sum_k alpha_k V_kj",
                "stress": "Independent nonlinear real-load-conditioned head outputs shared signed-scalar asinh encoding; never scalar linear superposition"},
            "load_metadata": {"order": ["ver", "hor", "tor"], "rank": 3,
                "physical_basis_6x3": self.metadata_record["physical_basis"], "feature_to_alpha_3x6": self.metadata_record["alpha_matrix"],
                "prepared_feature_scale": LOAD_SCALES, "load_span_tolerance_normalized": self.load_span_tolerance,
                "basis_metadata_sha256": self.metadata_record["basis_metadata_sha256"], "scope": "Existing compatible45 TRAIN/DEV JSON only"},
            "scaler": {"mean": self.metadata_record["mean"], "std": self.metadata_record["std"],
                "stress_scale": self.metadata_record["stress_scale"], "channels": [0, 1, 2, 3],
                "supplied_record_sha256": self.metadata_record["scaler_sha256"], "fit_in_model": False},
            "initialization_policy": "Same seed and encoder created first gives common encoder initialization; heads, full-model initialization/counts differ, so no complete parameter equality or strict capacity causal claim",
            "context_gradient": "All query encoder dependence is the explicit ctx argument; no detached hidden state or train-time descriptor cache",
            "gradient_policy": "Finite positive aggregate encoder/trunk/u-head/stress-head gradients at first actual step; one-hot inactive basis output rows may be zero",
            "sources": {"deeponet_field.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "primary_DeepONet": "https://doi.org/10.1038/s42256-021-00302-5", "author_code": "https://github.com/lululxvi/deeponet",
                "adaptation": "New PyTorch geometry/BC set-branch + nine-response basis implementation; not a copied official preset or fixed-sensor DeepONet reproduction"},
            "limitations": ["Original nonforce19 off-node extension is undefined; no native continuous/PDE/BC acceptance",
                "Discrete roles/current1024 sampling may omit interface points; no covert coverage repair",
                "Low rank is a capacity constraint, not a demonstrated error cause", "Only verified rank3 load span, not arbitrary six-load response",
                "Signed scalar stress has unknown constitutive recovery semantics", "No GT-linearity or engineering guarantee",
                "No old-final or sealed new-test input/target read in model"]}

    def encode_geometry(self, cloud22: Tensor, mask: Tensor) -> Tensor:
        if (not isinstance(cloud22, Tensor) or cloud22.shape != (1, CLOUD_POINTS, 22) or not cloud22.is_floating_point()
                or not isinstance(mask, Tensor) or mask.shape != (1, CLOUD_POINTS) or mask.dtype != torch.bool
                or cloud22.device != mask.device or cloud22.device != next(self.encoder.parameters()).device):
            raise ValueError("Expected shared cloud22[1,1024,22] and bool mask[1,1024] on model device")
        return self.encoder(cloud22, mask)

    def load_coefficients(self, load6: Tensor) -> Tensor:
        if (not isinstance(load6, Tensor) or load6.shape not in ((6,), (1, 6)) or not load6.is_floating_point()
                or load6.device != self.feature_to_alpha.device):
            raise ValueError("load6 must be normalized features7:13 floating[6] or[1,6] on model device")
        value = load6.reshape(1, 6).to(torch.float64)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite normalized load vector")
        alpha = value @ self.feature_to_alpha.transpose(0, 1)
        residual = value-alpha @ self.normalized_load_basis.transpose(0, 1)
        if float(residual.abs().max()) > self.load_span_tolerance:
            raise ValueError("Load is outside the verified rank3 basis")
        return alpha.to(torch.float32)

    def forward_components(self, xyz: Tensor, features: Tensor, context: Tensor, load6: Tensor) -> dict[str, Tensor]:
        if (not isinstance(xyz, Tensor) or xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0
                or not xyz.is_floating_point() or not isinstance(features, Tensor) or features.shape != (len(xyz), 25)
                or not features.is_floating_point() or features.device != xyz.device
                or not isinstance(context, Tensor) or context.shape != (1, 32) or not context.is_floating_point()
                or context.device != xyz.device or xyz.device != next(self.solver.parameters()).device):
            raise ValueError("Expected xyz[N,3], original features[N,25] and explicit context[1,32] on model device")
        alpha = self.load_coefficients(load6)
        query22 = torch.cat((xyz.float(), features[:, :7].float(), features[:, 13:25].float()), dim=-1)
        basis, stress = self.solver(query22, context.float(), load6.reshape(1, 6).float())
        if basis.shape != (len(xyz), 9) or stress.shape != (len(xyz), 1):
            raise ValueError("Response component output shapes changed")
        return {"basis_nodes": basis, "stress_nodes": stress, "alpha": alpha}

    def compose_encoded_nodes(self, components: dict[str, Tensor]) -> Tensor:
        basis, stress, alpha = (components[key] for key in ("basis_nodes", "stress_nodes", "alpha"))
        if (basis.ndim != 2 or basis.shape[1] != 9 or len(basis) == 0 or stress.shape != (len(basis), 1)
                or alpha.shape != (1, 3) or basis.device != self.u_encoded_zero.device
                or stress.device != basis.device or alpha.device != basis.device):
            raise ValueError("Expected aligned basis_nodes[N,9], stress_nodes[N,1], alpha[1,3]")
        u = (basis.float().reshape(-1, 3, 3)*alpha.float().reshape(1, 3, 1)).sum(dim=1)
        encoded_u = u+self.u_encoded_zero.reshape(1, 3)
        return torch.cat((encoded_u, stress.float()), dim=1)

    def forward_points(self, xyz: Tensor, features: Tensor, context: Tensor, load6: Tensor) -> Tensor:
        return self.compose_encoded_nodes(self.forward_components(xyz, features, context, load6))

    def forward(self, xyz: Tensor, features: Tensor, context: Tensor, load6: Tensor) -> Tensor:
        return self.forward_points(xyz, features, context, load6)


def build_deeponet_model(mode: str, scaler: dict, basis_metadata: dict, *,
                         seed: int = 2026, device: str | torch.device = "cuda") -> DeepONetField:
    return DeepONetField(mode, scaler, basis_metadata, seed=seed, device=device)
