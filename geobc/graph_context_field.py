"""Isolated43-input extension of the frozen fused query displacement basis.

The caller appends either Euclidean-interface5 or mesh-edge-path5; both are
active and use exactly the same model. Geometry/BC clouds and Gaussian query
context remain unchanged. No graph, labels, scales or files are read here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .query_context_field import QueryContextFieldConfig, QueryContextPhysicsField

ORIGINAL_LOCAL_DIM = 38
DISTANCE_DIM = 5
CONDITION_DIM = 43
NATIVE_CONDITION_DIM = 107
NOMINAL_ENCODED_DIM = 131
PADDED_ENCODED_DIM = 144
RAW_NATIVE_INPUT_DIM = 110
EXPECTED_NATIVE_PARAMETERS = 7_770_496
EXPECTED_FULL_PARAMETERS = 7_789_376


@dataclass(frozen=True)
class GraphContextFieldConfig(QueryContextFieldConfig):
    condition_dim: int = CONDITION_DIM
    query_mode: Literal["actual"] = "actual"
    distance_mode: Literal["euclidean", "graph"] = "euclidean"

    def __post_init__(self):
        # The frozen PhysicsFieldConfig accepts25/38 and its query subclass
        # accepts38. Validate this isolated extension directly instead of
        # mutating or calling those intentionally narrower guards.
        if (self.mode != "linear_basis" or self.geometry_mode != "shape"
                or self.condition_dim != CONDITION_DIM or self.descriptor_dim != 32
                or self.hidden_dim != 128 or self.hidden_layers != 2 or self.cloud_points != 1024
                or self.bc_role_mode != "actual" or self.query_mode != "actual"
                or self.query_dim != 32 or self.sigma != .15 or self.query_chunk != 4096
                or self.distance_mode not in ("euclidean", "graph") or self.seed < 0):
            raise ValueError("Graph-distance pair requires43inputs, active BC/query, linear9basis,128x2,cloud1024,sigma.15/chunk4096")
        self.conditioned_config()  # Inherited dynamic43+query32 ->75; appends global32 ->107.

    def record(self):
        result = super().record()
        result.update(method="GeoBC matched Euclidean-interface versus mesh-edge-distance context candidate",
            claim="distance-representation utility ablation; no new algorithm/stiffness/generalization claim",
            input_layout="masked original25 +local13 +active interface5 +actual query32 +global32 =107 Identity conditions",
            role_control="retain the same actual canonical support_union/load_member cloud channels on BOTH distance modes",
            scale_policy="trainer directly reuses existing TRAIN120 common u RMS with center0; model reads/fits no scale",
            interface_distance={"mode": self.distance_mode, "dimension": DISTANCE_DIM,
                "outer_slice": [ORIGINAL_LOCAL_DIM, CONDITION_DIM], "both_modes_active": True,
                "source_order": "four support member groups sorted by master GRID ID, then actual loaded member group",
                "normalization": "same physical endpoint distance/L; no fitted transform or clipping",
                "source": "external target-independent original-node cache, verified by trainer; no data/graph construction here",
                "zero_branch": False,
                "scope_limit": "edge-graph paths are not stiffness or exact continuous/remeshing-invariant geodesics"})
        result["query_context"]["zero_control"] = "None; query32 remains actual on BOTH distance modes"
        result["capacity_cost"] = {
            "distance_inputs_active_each_model": 5,
            "existing_query_projection_parameters": 2080,
            "new_learned_head_parameters": 0,
            "old_query_native_Identity_conditions": 102, "new_native_Identity_conditions": NATIVE_CONDITION_DIM,
            "old_query_nominal_encoded_channels": 126, "nominal_encoded_channels": NOMINAL_ENCODED_DIM,
            "source_derived_padded_encoded_channels": PADDED_ENCODED_DIM,
            "source_derived_additional_MLP_parameters_vs_old_query": 2048,
            "native_padding_policy": "Composite pads its final Identity107 to120 with13ones, total24+120=144; actual public combined native.numel must be checked",
            "historical_control_limit": "Both fresh distance models have enlarged equal architecture; old smaller query/native initialization is not a matched control",
            "distance_limit": "Matched source groups and active feature count; different distance distributions/conditioning still prevent a pure topology-causality claim"}
        return result


class GraphContextPhysicsField(QueryContextPhysicsField):
    """Same physics/query API; only fixed conditioning dimension is extended.

distance_mode is truthful metadata for the caller's cache choice. The field
does not change inputs, initialize differently, or disable any of the five
distance channels in either mode. PhysicsField still masks original7:13.
"""
    def __init__(self, config: GraphContextFieldConfig | None = None, device="cuda", decoder=None):
        config = config or GraphContextFieldConfig()
        if not isinstance(config, GraphContextFieldConfig):
            raise TypeError("Expected GraphContextFieldConfig")
        super().__init__(config, device=device, decoder=decoder)
        if self.backend == "tcnn_fully_fused":
            self.validate_native_layout()

    def validate_native_layout(self):
        """Measure the public combined buffer/API, without guessed flat slices."""
        if self.backend != "tcnn_fully_fused" or not hasattr(self.decoder, "native"):
            raise ValueError("CPU-injected interfaces cannot certify official native layout")
        native = self.decoder.native
        cpp = native.native_tcnn_module
        if (native.n_input_dims != RAW_NATIVE_INPUT_DIM or native.n_output_dims != 9
                or cpp.n_input_dims() != RAW_NATIVE_INPUT_DIM or cpp.n_output_dims() != 16
                or native.params.numel() != EXPECTED_NATIVE_PARAMETERS
                or native.params.dtype != torch.float32 or native.dtype != torch.float16
                or sum(p.numel() for p in self.decoder.parameters()) != EXPECTED_NATIVE_PARAMETERS
                or sum(p.numel() for p in self.parameters()) != EXPECTED_FULL_PARAMETERS):
            raise ValueError("Measured native API/combined count/precision differs from fixed fused43-context contract")
        return {"raw_public_input_dims": native.n_input_dims, "raw_native_input_dims": cpp.n_input_dims(),
            "public_output_dims": native.n_output_dims, "native_padded_output_dims": cpp.n_output_dims(),
            "combined_public_parameter_count": native.params.numel(), "combined_public_parameter": "field.decoder.native.params",
            "master_dtype": str(native.params.dtype), "compute_dtype": str(native.dtype),
            "Identity_conditions": NATIVE_CONDITION_DIM, "nominal_encoded_channels": NOMINAL_ENCODED_DIM,
            "source_derived_padded_encoded_channels": PADDED_ENCODED_DIM,
            "Identity_source_derived_padding_count": 13, "Identity_source_derived_padding_value": 1,
            "encoded_width_proof": "official Composite/Encoding alignment source plus measured total7770496 consistent with36864 MLP weights; internal encoded width not independently exposed/measured by this public API",
            "flat_buffer_offsets_inferred": False}

    def condition_features(self, features):
        if features.ndim != 2 or features.shape[1] != CONDITION_DIM:
            raise ValueError("Expected43 query inputs: original25 +local13 +active interface_distance5")
        return super().condition_features(features)

    @property
    def source_config(self):
        result = super().source_config
        # The frozen query source hardcodes102 in its record; override all
        # actual dimensions and avoid carrying its old38 capacity narrative.
        result.update(native_identity_condition_dim=NATIVE_CONDITION_DIM,
            native_condition_dim=NATIVE_CONDITION_DIM, native_raw_input_dim=RAW_NATIVE_INPUT_DIM,
            native_nominal_encoded_channels=NOMINAL_ENCODED_DIM,
            native_source_derived_padded_encoded_channels=PADDED_ENCODED_DIM,
            native_parameter_buffers=["field.decoder.native.params"] if self.backend == "tcnn_fully_fused" else [],
            full_model_parameters=sum(p.numel() for p in self.parameters()),
            distance_mode=self.config.distance_mode,
            native_layout=self.validate_native_layout() if self.backend == "tcnn_fully_fused" else None,
            CPU_injection_limit="Injected Torch components check interface only, not native CUDA layout/counts" if self.backend != "tcnn_fully_fused" else None,
            new_learned_head_parameters=0, hash_modulation_enabled=False,
            initialization_policy="fresh equal complete parameters and constructor RNG in euclidean/graph; naturally different distance inputs need not have equal initial predictions")
        result["conditioned_field"]["conditioning"] = "original38 then active interface5 then actual query32 then global32"
        result["conditioned_field"]["geometry_source"] = "unchanged saved canonical actual-BC-role5 cloud; distance5 supplied only as external query inputs"
        return result


__all__ = ["GraphContextFieldConfig", "GraphContextPhysicsField"]
