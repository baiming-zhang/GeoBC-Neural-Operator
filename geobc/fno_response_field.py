"""Matched FNO displacement-response basis with an independent stress head.

This isolated candidate reuses the frozen four-quadrant spectral layer. Both
arms have identical stored trainable tensors and creation order; only the six
current-load channels presented to the shared FNO trunk differ. Scalers and
load metadata are supplied by the trainer; this module reads no dataset.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .conditioned_field import PointNetGeometryEncoder
from .fno import FNO_SOURCES, SpectralConv3d

MODES = ("linear_basis", "load_dependent_basis")
GRID_LOAD_SLICE = slice(11, 17)  # occupancy0 + xyz1:4 + original25[7:13]
NORMALIZED_LOAD_SCALES = [42300., 42300., 42300., 565000., 565000., 565000.]


def _sha(filename: str) -> str:
    return hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()


def _finite_matrix(value, rows: int, columns: int, name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows or any(not isinstance(row, list) or len(row) != columns for row in value):
        raise ValueError(f"{name} must be {rows}x{columns}")
    matrix = [[float(entry) for entry in row] for row in value]
    if not all(math.isfinite(entry) for row in matrix for entry in row):
        raise ValueError(f"Nonfinite {name}")
    return matrix


def _metadata(scaler: dict, basis: dict) -> dict:
    """Validate supplied small metadata, not fields or a newly fitted scaler."""
    if scaler.get("channels") != [0, 1, 2, 3] or len(scaler.get("mean", [])) != 4 or len(scaler.get("std", [])) != 4:
        raise ValueError("Expected the shared four-field TRAIN-only scaler")
    mean, std = [float(x) for x in scaler["mean"]], [float(x) for x in scaler["std"]]
    if not all(math.isfinite(x) for x in mean+std) or min(std) <= 0:
        raise ValueError("Shared scaler mean/std are invalid")
    stress_scale = scaler.get("stress_scale")
    if stress_scale is None or not math.isfinite(float(stress_scale)) or float(stress_scale) <= 0:
        raise ValueError("This response screen retains the shared asinh-stress scaler")
    if basis.get("compatible") is not True or basis.get("CADs_checked") != 45 or basis.get("compatible_CADs") != 45 or basis.get("badcases") != []:
        raise ValueError("Expected the existing compatible45 TRAIN/validation metadata guard")
    canonical, mapping = basis["canonical_basis"], basis["prepared_feature_mapping"]
    if canonical.get("order") != ["ver", "hor", "tor"] or canonical.get("exact_rank") != 3:
        raise ValueError("Expected verified ver/hor/tor rank3")
    if canonical.get("physical_vector_order") != ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]:
        raise ValueError("Physical load-vector order changed")
    if mapping.get("slots") != list(range(7, 13)) or mapping.get("physical_scale") != NORMALIZED_LOAD_SCALES or mapping.get("alpha_order") != ["ver", "hor", "tor"]:
        raise ValueError("Prepared25 load feature schema changed")
    matrix = _finite_matrix(mapping["feature_to_alpha_matrix_3x6"], 3, 6, "feature-to-alpha")
    normalized_basis = _finite_matrix(mapping["normalized_basis_6x3"], 6, 3, "normalized basis")
    physical_basis = _finite_matrix(canonical["physical_matrix_6x3"], 6, 3, "physical basis")
    for i in range(3):
        for j in range(3):
            result = sum(matrix[i][k]*normalized_basis[k][j] for k in range(6))
            if abs(result-float(i == j)) > 1e-12:
                raise ValueError("Load coefficient map is not a left inverse of the verified basis")
    for i in range(6):
        for j in range(3):
            if abs(normalized_basis[i][j]-physical_basis[i][j]/NORMALIZED_LOAD_SCALES[i]) > 1e-12:
                raise ValueError("Physical/normalized load basis differs")
    tolerance = float(mapping["tolerance_normalized"])
    if tolerance != 1e-7:
        raise ValueError("Frozen normalized load-span tolerance changed")
    records = basis.get("cases", [])
    flags = ("compatible", "basis_matches_shared_reference", "constraints_identical",
             "zero_fixed_constraints", "material_identical_within_CAD",
             "load_interface_identical", "STATICS_without_nonlinear_entries")
    if len(records) != 45 or any(record.get("basis_rank") != 3 or any(record.get(key) is not True for key in flags) for record in records):
        raise ValueError("Metadata does not support one unchanged homogeneous linear-static operator per CAD")
    return {"mean": mean, "std": std, "stress_scale": float(stress_scale),
        "alpha_matrix": matrix, "normalized_basis": normalized_basis,
        "physical_basis": physical_basis, "load_span_tolerance": tolerance}


@dataclass(frozen=True)
class FNOResponseFieldConfig:
    mode: Literal["linear_basis", "load_dependent_basis"] = "linear_basis"
    seed: int = 2026

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Response mode must be one of {MODES}")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")

    def record(self) -> dict:
        return {**asdict(self), "family": "FNO response-basis candidate",
            "grid_input_channels": 29, "global_context": 32, "FNO_input_channels": 61,
            "resolution": 64, "width": 16, "modes": 12, "layers": 4, "padding": 0,
            "fft_norm": "backward", "frequency_semantics": "Frozen geobc.fno.SpectralConv3d four-quadrant m12; real/imag stored separately, not NeuralOperator total-modes12",
            "checkpoint_blocks": True, "u_basis_head": [16,128,9],
            "stress_head": [22,128,1], "head_activation": "GELU",
            "u_basis_order": ["ver_x","ver_y","ver_z","hor_x","hor_y","hor_z","tor_x","tor_y","tor_z"],
            "load_policy": "zero grid channels11:17 before FNO lifting" if self.mode == "linear_basis" else "retain grid channels11:17 diagnostic control",
            "stress_load_policy": "Append the same live normalized physical F/M6 to latent16 in both arms",
            "effective_capacity": "Stored parameter layout matches; clearing load columns reduces active conditioning, so equal effective capacity is not claimed",
            "sources": FNO_SOURCES}


class _ResponseSolver(nn.Module):
    """Old FNO latent body plus two explicitly separate pointwise heads."""

    def __init__(self):
        super().__init__()
        self.lift = nn.Conv1d(61, 16, 1)
        self.spectral = nn.ModuleList([SpectralConv3d(16, 16, 12) for _ in range(4)])
        self.pointwise = nn.ModuleList([nn.Conv1d(16, 16, 1) for _ in range(4)])
        self.u_basis_head = nn.Sequential(nn.Conv1d(16,128,1), nn.GELU(), nn.Conv1d(128,9,1))
        self.stress_head = nn.Sequential(nn.Conv1d(22,128,1), nn.GELU(), nn.Conv1d(128,1,1))

    def _block(self, x: Tensor, index: int) -> Tensor:
        local = self.pointwise[index](x.flatten(2)).reshape_as(x)
        result = self.spectral[index](x) + local
        return F.gelu(result) if index < 3 else result

    def forward(self, inputs: Tensor, load6: Tensor) -> tuple[Tensor,Tensor]:
        x = self.lift(inputs.flatten(2)).reshape(1,16,64,64,64)
        for index in range(4):
            if self.training and torch.is_grad_enabled():
                x = checkpoint(lambda value, i=index: self._block(value,i), x, use_reentrant=False)
            else:
                x = self._block(x,index)
        flat = x.flatten(2)
        basis = self.u_basis_head(flat).reshape(1,9,64,64,64)
        force_features = load6[:,:,None].expand(1,6,flat.shape[-1])
        stress = self.stress_head(torch.cat((flat,force_features),dim=1)).reshape(1,1,64,64,64)
        return basis, stress


class FNOResponseField(nn.Module):
    """encode_geometry -> forward_grid -> frozen legacy trilinear node query.

`load6` is the global vector in original features7:13, not physical mixed-unit
F/M and not case IDs. `basis_metadata` is the existing45-CAD JSON. Outputs are
the shared four-field encoding; the trainer alone accesses targets/decodes.
"""

    def __init__(self, mode: str, scaler: dict, basis_metadata: dict, *,
                 seed: int = 2026, device: str | torch.device = "cuda"):
        super().__init__()
        self.config = FNOResponseFieldConfig(mode=mode, seed=seed)
        self.mode, self.seed, self.method = mode, seed, "fno"
        metadata = _metadata(scaler, basis_metadata)
        # Creation order and RNG are identical in both arms, including heads.
        torch.manual_seed(seed)
        self.encoder = PointNetGeometryEncoder(descriptor_dim=32, point_width=64)
        self.solver = _ResponseSolver()
        self.register_buffer("u_mean",torch.tensor(metadata["mean"][:3],dtype=torch.float64))
        self.register_buffer("u_std",torch.tensor(metadata["std"][:3],dtype=torch.float64))
        self.register_buffer("u_encoded_zero",(-self.u_mean/self.u_std).to(torch.float32))
        self.register_buffer("feature_to_alpha",torch.tensor(metadata["alpha_matrix"],dtype=torch.float64))
        self.register_buffer("normalized_load_basis",torch.tensor(metadata["normalized_basis"],dtype=torch.float64))
        self.load_span_tolerance = metadata["load_span_tolerance"]
        self.metadata_record = metadata
        self.capabilities = {"forward_grid":True, "forward_points":False,
            "native_continuous_query_implemented":False, "zero_load_u": "Algebraic physical zero up to ordinary floating point/scaler/query rounding, not strict boundary enforcement"}
        self.to(device)

    @property
    def parameter_counts(self) -> dict:
        encoder = sum(parameter.numel() for parameter in self.encoder.parameters())
        solver = sum(parameter.numel() for parameter in self.solver.parameters())
        return {"encoder":encoder,"solver":solver,"total":encoder+solver}

    @property
    def source_config(self) -> dict:
        counts = self.parameter_counts
        components = {name:sum(p.numel() for p in getattr(self.solver,name).parameters())
            for name in ("lift","spectral","pointwise","u_basis_head","stress_head")}
        return {"method":"fno_response_basis","response_mode":self.mode,"seed":self.seed,
            "configuration":self.config.record(), "geometry_encoder":{"class":"PointNetGeometryEncoder",
                "cloud_shape":[1,1024,3],"mask_shape":[1,1024],"width":64,"descriptor_dim":32,
                "coordinates":"Same fixed xyz-only cloud; independent of current load/targets"},
            "input_schema":"grid29=occupancy1+xyz3+original25; append live global32; load6 separately uses original25 slots7:13",
            "output_schema":{"grid":[1,4,64,64,64],"channels":["encoded_ux","encoded_uy","encoded_uz","encoded_source_signed_stress"],
                "u_basis":"V[k,j]=physical u for canonical basis[k] divided by existing TRAIN std[j], zero-centered; nine values are not individually mean-centered field encodings",
                "u_composition":"encoded_u[j]=sum_k alpha[k]*V[k,j]-mean[j]/std[j]; physical_u[j]=std[j]*sum_k alpha[k]*V[k,j]",
                "stress":"Independent nonlinear force-conditioned scalar head in existing asinh/center/std encoding; no scalar superposition or constitutive GT claim"},
            "load_metadata":{"order":["ver","hor","tor"],"rank":3,"physical_basis_6x3":self.metadata_record["physical_basis"],
                "feature_to_alpha_3x6":self.metadata_record["alpha_matrix"],"prepared_feature_scale":NORMALIZED_LOAD_SCALES,
                "load_span_tolerance_normalized":self.load_span_tolerance,"support":"Existing45 TRAIN/validation guard; no new CAD or field-relation verification"},
            "query_equivalence":{"equation":"T(sum(alpha*V)-mu/std)=sum(alpha*T(V))-mu/std when each static trilinear row sums to1",
                "scope":"alpha constant across case, T linear; means applied once. Actual FP32 row-weight/accumulation/scaler rounding can differ, so no bitwise equivalence or exact-zero numerical claim",
                "runtime_verification":"Deferred to first actual case; author AST only, no model/toy/probe"},
            "parameter_counts":counts,"solver_component_counts":components,
            "parameter_count_policy":"sum(numel) of actual public trainable real tensors; spectral complex values use a final real/imag axis2 and count both components; buffers excluded",
            "stored_initialization_policy":"Both arms same constructor, seed, tensor layout and order. Natural predictions differ from changed inputs; old direct4 FNO is not a matched control",
            "gradient_policy":"Current known one-hot alpha may give zero gradients in inactive basis output rows; require finite gradients and positive aggregate active-head/trunk/encoder gradient, not all9 rows positive",
            "source_sha256":{"fno_response_field.py":_sha("fno_response_field.py"),"fno.py":_sha("fno.py"),"conditioned_field.py":_sha("conditioned_field.py")},
            "no_dataset_or_scaler_fit_in_model":True,
            "limitations":["Only verified load span, not arbitrary six-component loading","Geometry/BC accuracy and engineering small-deformation validity unproven","No predicted-strain to source scalar stress identity","No native off-node/continuous/PDE validation","No old-final or sealed-test input/target access"]}

    def encode_geometry(self, cloud: Tensor, mask: Tensor) -> Tensor:
        if cloud.shape != (1,1024,3) or not cloud.is_floating_point() or mask.shape != (1,1024) or mask.dtype != torch.bool or cloud.device != mask.device:
            raise ValueError("Expected shared per-case cloud[1,1024,3] and bool mask[1,1024]")
        if cloud.device != next(self.encoder.parameters()).device:
            raise ValueError("Cloud and geometry encoder devices differ")
        return self.encoder(cloud,mask)

    def load_coefficients(self, load6: Tensor) -> Tensor:
        if not isinstance(load6,Tensor) or load6.shape not in ((6,),(1,6)) or not load6.is_floating_point() or load6.device != self.feature_to_alpha.device:
            raise ValueError("load6 must be floating[6] or[1,6] normalized features7:13 on the model device")
        value = load6.reshape(1,6).to(torch.float64)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite physical-load feature vector")
        alpha = value @ self.feature_to_alpha.T
        residual = value-alpha @ self.normalized_load_basis.T
        if bool((residual.abs()>self.load_span_tolerance).any()):
            raise ValueError("Load outside the verified ver/hor/tor span; unsupported Fy/Mx/My are not mapped to case IDs")
        return alpha.to(torch.float32)

    def forward_components(self, grid: Tensor, context: Tensor, load6: Tensor) -> dict[str,Tensor]:
        """One latent pass; expose components for actual-case algebra witnesses."""
        if grid.shape != (1,29,64,64,64) or not grid.is_floating_point():
            raise ValueError("Expected the unchanged selection frontend grid[1,29,64,64,64]")
        if context.shape != (1,32) or not context.is_floating_point() or context.device != grid.device or grid.device != next(self.solver.parameters()).device:
            raise ValueError("Live per-case context[1,32], grid and response solver must share a device")
        alpha = self.load_coefficients(load6)
        template = grid.float()
        if self.mode == "linear_basis":
            template = template.clone()
            template[:,GRID_LOAD_SLICE,:,:,:] = 0
        global_grid = context.float()[:,:,None,None,None].expand(1,32,64,64,64)
        basis, stress = self.solver(torch.cat((template,global_grid),dim=1),load6.reshape(1,6).float())
        return {"basis_grid":basis,"stress_grid":stress,"alpha":alpha}

    def compose_encoded_grid(self, components: dict[str,Tensor]) -> Tensor:
        basis, stress, alpha = components["basis_grid"],components["stress_grid"],components["alpha"]
        if basis.shape != (1,9,64,64,64) or stress.shape != (1,1,64,64,64) or alpha.shape != (1,3):
            raise ValueError("Expected basis9, encoded stress1 and ver/hor/tor alpha3")
        displacement = torch.einsum("bk,bkjxyz->bjxyz",alpha.to(basis.dtype),basis.reshape(1,3,3,64,64,64))
        displacement = displacement+self.u_encoded_zero.to(basis.dtype)[None,:,None,None,None]
        return torch.cat((displacement,stress),dim=1)

    def forward_grid(self, grid: Tensor, context: Tensor, load6: Tensor) -> Tensor:
        """Encoded4 grid; legacy query_grid consumes every original node."""
        return self.compose_encoded_grid(self.forward_components(grid,context,load6))

    def compose_encoded_nodes(self, basis_nodes: Tensor, stress_nodes: Tensor, load6: Tensor) -> Tensor:
        """Basis-first node query algebra; no node copying/new-query guarantee."""
        if basis_nodes.ndim != 2 or basis_nodes.shape[1] != 9 or stress_nodes.shape != (len(basis_nodes),1) or basis_nodes.device != stress_nodes.device:
            raise ValueError("Expected queried basis[N,9] and encoded stress[N,1]")
        alpha = self.load_coefficients(load6)[0].to(basis_nodes.dtype)
        displacement = torch.einsum("k,nkj->nj",alpha,basis_nodes.reshape(-1,3,3))
        displacement = displacement+self.u_encoded_zero.to(basis_nodes.dtype)[None,:]
        return torch.cat((displacement,stress_nodes),dim=1)

    def forward(self, grid: Tensor, context: Tensor, load6: Tensor) -> Tensor:
        return self.forward_grid(grid,context,load6)


def build_fno_response_model(mode: str, scaler: dict, basis_metadata: dict, *,
                             seed: int = 2026, device: str | torch.device = "cuda") -> FNOResponseField:
    return FNOResponseField(mode,scaler,basis_metadata,seed=seed,device=device)
