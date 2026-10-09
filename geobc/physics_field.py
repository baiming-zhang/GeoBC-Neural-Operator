"""A load-linear displacement-response basis candidate for GeoBC.

The shared geometry-conditioned field predicts nine scaled displacement
components, arranged as [ver/hor/tor, ux/uy/uz]. Reference-load coefficients
are supplied by the trainer. No coefficient files, displacement scales or
stress labels are read here. This linear ansatz is a baseline to test, not a
claim that a new physics method or generalization result has been established.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .conditioned_field import ConditionedField, ConditionedFieldConfig

REFERENCE_LOAD_ORDER = ("ver", "hor", "tor")
COMPONENT_ORDER = ("ux", "uy", "uz")
LOAD_CHANNELS = slice(7, 13)


@dataclass(frozen=True)
class PhysicsFieldConfig:
    mode: Literal["linear_basis", "load_dependent_basis"] = "linear_basis"
    geometry_mode: Literal["shape", "none"] = "shape"
    condition_dim: int = 25
    descriptor_dim: int = 32
    hidden_dim: int = 128
    hidden_layers: int = 2
    cloud_points: int = 1024
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.mode not in ("linear_basis", "load_dependent_basis"):
            raise ValueError("mode must be linear_basis or load_dependent_basis")
        if self.geometry_mode not in ("shape", "none"):
            raise ValueError("Physics basis supports shape/none geometry, without case identity")
        if self.condition_dim not in (25, 38):
            raise ValueError("Use the original25 BC features, optionally followed by13 local geometry features")
        self.conditioned_config()  # Validate the same shared hash/MLP settings.

    def conditioned_config(self) -> ConditionedFieldConfig:
        return ConditionedFieldConfig(
            geometry_mode=self.geometry_mode, descriptor_dim=self.descriptor_dim,
            condition_dim=self.condition_dim, out_channels=9,
            cloud_points=self.cloud_points, hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers, seed=self.seed)

    def record(self) -> dict:
        return {
            **asdict(self), "method": "GeoBC load-linear displacement response-basis candidate",
            "claim": "linear ansatz/control experiment; efficacy and generalization require measured results",
            "conditioned_field": self.conditioned_config().record(),
            "output_channels": 9, "basis_shape": "[N,3 reference loads,3 displacement components]",
            "reference_load_order": list(REFERENCE_LOAD_ORDER), "component_order": list(COMPONENT_ORDER),
            "flat_output_order": [f"{load}_{component}" for load in REFERENCE_LOAD_ORDER for component in COMPONENT_ORDER],
            "combination": "scaled_u[n,c] = sum_l alpha[n,l] * basis[n,l,c]",
            "coefficient_source": "external target-independent reference-load metadata; no files parsed here",
            "load_feature_policy": "zero original25 columns7:13" if self.mode == "linear_basis" else "retain original25 columns7:13 diagnostic control",
            "input_layout": "original25 BC/material + optional13 local geometry + shared global geometry descriptor",
            "scale_policy": "trainer fits one train-only RMS(u_mm) scalar; center0; same scale for all basis components and loads",
            "physical_decode": "u_mm = common_u_scale_mm * scaled_u; no per-load bias or learned scale",
            "stress_policy": "No stress output, training, scale fitting or linear stress combination in this module",
            "linearity_scope": "linear_basis only, with identical geometry and nonload BC/material/local features; load_dependent_basis need not be load-linear",
        }


def combine_basis(basis: Tensor, alpha: Tensor) -> Tensor:
    """Compose three displacement responses with no intercept, [N,3].

    Physical coefficients may be negative. The same zero-centered scalar must
    decode every basis response. Shape/device validation is kept here; the
    trainer validates metadata finiteness once before GPU sampling/evaluation.
    """
    if basis.ndim != 3 or basis.shape[1:] != (3, 3):
        raise ValueError("basis must have shape [N,3 reference loads,3 components]")
    if not isinstance(alpha, Tensor) or alpha.shape != (len(basis), 3):
        raise ValueError("alpha must be a tensor shaped [N,3] in ver/hor/tor order")
    if alpha.device != basis.device or not alpha.is_floating_point():
        raise ValueError("alpha must be floating point on the basis device")
    return torch.einsum("nl,nlc->nc", alpha.to(dtype=basis.dtype), basis)


class PhysicsField(nn.Module):
    """A nine-output ConditionedField with explicit load masking/composition.

    encode_geometry(...) -> context[B,D], recomputed per training optimizer step.
    response_basis(xyz,features,context=...,cad_index=...) -> basis[N,3,3].
    forward(xyz,features,alpha,context=...,cad_index=...) -> scaled_u[N,3].

    Global/local geometry and BC constraints stay fixed while alpha changes.
    linear_basis clears only the original force/moment columns7:13 on a clone;
    the caller's features and the layout are preserved. The load-dependent
    control changes this mask alone, with the identical seed/parameter layout.
    """

    def __init__(self, config: PhysicsFieldConfig | None = None,
                 device: torch.device | str = "cuda", decoder: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config or PhysicsFieldConfig()
        self.field = ConditionedField(self.config.conditioned_config(), device=device, decoder=decoder)
        self.backend = self.field.backend

    @property
    def encoder(self):
        return self.field.encoder

    @property
    def decoder(self):
        return self.field.decoder

    @property
    def source_config(self) -> dict:
        result = self.config.record()
        result.update(
            conditioned_field=self.field.source_config,
            actual_decoder_backend=self.backend,
            encoder_parameters=sum(p.numel() for p in self.encoder.parameters()) if self.encoder is not None else 0,
            decoder_parameters=sum(p.numel() for p in self.decoder.parameters()))
        return result

    def encode_geometry(self, geometry_cloud: Tensor, geometry_mask: Tensor | None = None) -> Tensor:
        return self.field.encode_geometry(geometry_cloud, geometry_mask)

    def condition_features(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.config.condition_dim:
            raise ValueError("features must contain the original25 channels plus optional13 local channels")
        if self.config.mode == "load_dependent_basis":
            return features
        masked = features.clone()
        masked[:, LOAD_CHANNELS] = 0
        return masked

    def response_basis(self, xyz: Tensor, features: Tensor,
                       geometry_cloud: Tensor | None = None, cad_index: Tensor | None = None,
                       *, geometry_mask: Tensor | None = None, context: Tensor | None = None) -> Tensor:
        flat = self.field(xyz, self.condition_features(features), geometry_cloud, cad_index,
                          geometry_mask=geometry_mask, context=context)
        if flat.shape != (len(xyz), 9):
            raise ValueError("Conditioned decoder must emit exactly nine displacement basis channels")
        return flat.reshape(len(xyz), 3, 3)

    def forward(self, xyz: Tensor, features: Tensor, alpha: Tensor,
                geometry_cloud: Tensor | None = None, cad_index: Tensor | None = None,
                *, geometry_mask: Tensor | None = None, context: Tensor | None = None) -> Tensor:
        return combine_basis(self.response_basis(xyz, features, geometry_cloud, cad_index,
                                                 geometry_mask=geometry_mask, context=context), alpha)


__all__ = ["PhysicsFieldConfig", "PhysicsField", "combine_basis",
           "REFERENCE_LOAD_ORDER", "COMPONENT_ORDER"]
