"""Fresh matched fixed masks on public 12-level TCNN hash activations.

Both variants store the same separately initialized Encoding and Network.
Only a nonlearned output mask differs. This changes effective spatial capacity,
not stored capacity, and does not guarantee a band-limited displacement field.
No old fused buffer offsets, learned gate, LoD control, targets or files are used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .bc_role_field import _encoder_record
from .graph_context_field import GraphContextFieldConfig, GraphContextPhysicsField
from .physics_field import PhysicsFieldConfig
from .query_context_field import QueryContextFieldConfig
from .tcnn_fields import TCNNFieldConfig, TCNN_COMMIT, TCNN_SOURCE

HASH_DIM = 24
CONDITION_DIM = 43
NATIVE_CONDITION_DIM = 107
NOMINAL_INPUT_DIM = 131
NETWORK_INPUT_DIM = 144
EXPLICIT_PADDING = 13
PADDED_OUTPUT_DIM = 16
EXPECTED_ENCODING_PARAMETERS = 7_733_632
EXPECTED_NETWORK_PARAMETERS = 36_864
EXPECTED_NATIVE_PARAMETERS = 7_770_496
EXPECTED_FULL_PARAMETERS = 7_789_376


@dataclass(frozen=True)
class HashBandwidthFieldConfig(GraphContextFieldConfig):
    distance_mode: Literal["euclidean"] = "euclidean"
    hash_bandwidth_mode: Literal["full12", "coarse8"] = "full12"

    def __post_init__(self):
        super().__post_init__()
        if self.distance_mode != "euclidean" or self.hash_bandwidth_mode not in ("full12", "coarse8"):
            raise ValueError("Fixed bandwidth pair uses Euclidean43 and only full12/coarse8 masks")

    def record(self):
        # Start from the basis record, replacing historical query/fused layout
        # narratives rather than carrying their narrower dimensions forward.
        result = PhysicsFieldConfig.record(self)
        result["conditioned_field"]["encoder"] = _encoder_record(self.descriptor_dim)
        result["conditioned_field"]["geometry_source"] = "unchanged canonical actual BC-role5 cloud"
        result["conditioned_field"]["decoder"] = {
            "implementation": "separate official Encoding and Network; model record supplies actual native counts",
            "outer_interface_config": asdict(self.conditioned_config().decoder_config()),
            "outer_interface_only": True, "native_Encoding_input_output": [3, HASH_DIM],
            "native_Network_input_output": [NETWORK_INPUT_DIM, 9], "explicit_one_padding_channels": EXPLICIT_PADDING}
        result["query_context"] = QueryContextFieldConfig.record(self)["query_context"]
        result["query_context"]["zero_control"] = "None; query32 remains actual in both bandwidth modes"
        result.update(method="GeoBC fixed hash bandwidth/capacity utility ablation",
            claim="one matched full12/coarse8 utility comparison; no frequency-causality or generalization claim",
            input_layout="masked original25 +local13 +active Euclidean5 +actual query32 +actual global32 =107 conditions",
            native_condition_dim=NATIVE_CONDITION_DIM, role_order=["support_union_original_DOF0_to5", "load_member_original_feature6"],
            global_cloud_input_dimension=5, geometry_target_independent=True, geometry_load_independent=True,
            scale_policy="trainer directly reuses fixed TRAIN120 common u RMS, center0; no model scale fitting",
            hash_bandwidth={"mode": self.hash_bandwidth_mode, "allocated_levels": 12, "features_per_level": 2,
                "active_levels": 12 if self.hash_bandwidth_mode == "full12" else 8,
                "encoded_width_both": HASH_DIM, "nonlearned_mask": "all24 ones" if self.hash_bandwidth_mode == "full12" else "first16 ones, last8 zeros",
                "parameter_equality_scope": "complete named trainable parameters; mode-specific fixed mask buffer explicitly differs",
                "natural_initial_prediction_equality_required": False,
                "capacity_limit": "equal stored parameters, different effective usable spatial capacity and gradient paths",
                "smoothness_limit": "continuous piecewise-linear grid interpolation; query/local/ReLU variation remains",
                "performance_limit": "all12 encoding levels computed in both modes; no lookup speedup claimed",
                "fresh_control": "both new split Encoding/Network constructions; no old fused checkpoint/control"},
            capacity_cost={"PointNet_parameters": 16800, "query_projection_parameters": 2080,
                "encoding_parameters_source_derived": EXPECTED_ENCODING_PARAMETERS,
                "network_parameters_source_derived": EXPECTED_NETWORK_PARAMETERS,
                "full_parameters_source_derived": EXPECTED_FULL_PARAMETERS, "new_learned_head_parameters": 0,
                "hash_channels": HASH_DIM, "Identity_condition_channels": NATIVE_CONDITION_DIM,
                "nominal_encoded_channels": NOMINAL_INPUT_DIM, "explicit_constant_one_channels": EXPLICIT_PADDING,
                "public_Network_input_channels": NETWORK_INPUT_DIM,
                "performance_limit": "public hash24 materialization and separate native call/autograd boundary"})
        return result


class SplitHashBandwidthDecoder(nn.Module):
    """Public Encoding -> fixed mask -> Identity107/ones13 -> Network144.

Activation capture is opt-in for the initial gradient acceptance only. The
retained encoded_activation is BEFORE the mask: its inactive-channel gradient
must vanish. The post-mask activation gradient need not vanish.
"""
    def __init__(self, config: TCNNFieldConfig, mode: str, device="cuda", *, encoding=None, network=None):
        super().__init__()
        expected = TCNNFieldConfig(out_channels=9, input_mode="both", condition_dim=NATIVE_CONDITION_DIM,
                                  hidden_dim=128, hidden_layers=2, seed=config.seed)
        if config != expected or mode not in ("full12", "coarse8"):
            raise ValueError("Split bandwidth decoder requires fixed12x2/conditions107/128x2/9basis")
        if (encoding is None) != (network is None):
            raise ValueError("Inject both public-interface components or neither")
        self.config, self.mode = config, mode
        self._injected = encoding is not None
        device = torch.device(device)
        if not self._injected:
            if device.type != "cuda":
                raise ValueError("Official split TCNN construction requires CUDA")
            try:
                import tinycudann as tcnn
            except (ImportError, OSError) as error:
                raise RuntimeError("Official tiny-cuda-nn CUDA extension unavailable") from error
            grid = dict(config.encoding_config()["nested"][0]); grid.pop("n_dims_to_encode")
            with torch.cuda.device(device):
                encoding = tcnn.Encoding(3, grid, seed=config.seed, dtype=torch.float16)
                network = tcnn.Network(NETWORK_INPUT_DIM, 9, config.network_config(), seed=config.seed)
                encoding.jit_fusion = False
                network.jit_fusion = False
        if (encoding.n_input_dims != 3 or encoding.n_output_dims != HASH_DIM
                or network.n_input_dims != NETWORK_INPUT_DIM or network.n_output_dims != 9):
            raise ValueError("Public Encoding/Network dimensions differ from24/144/9")
        self.encoding, self.network = encoding, network
        active = HASH_DIM if mode == "full12" else 16
        self.register_buffer("feature_mask", (torch.arange(HASH_DIM) < active).float())
        self.capture_activation_gradients = False
        self.encoded_activation = None
        self.masked_activation = None
        self.backend = "injected_cpu_split_bandwidth_interface" if self._injected else "tcnn_split_encoding_network_hash_bandwidth"
        self.to(device)
        if not self._injected:
            self.validate_native_layout()

    def native_parameter_groups(self):
        return {"encoding": list(self.encoding.parameters()), "network": list(self.network.parameters())}

    def clear_activation_capture(self):
        self.capture_activation_gradients = False
        self.encoded_activation = None
        self.masked_activation = None

    def validate_native_layout(self):
        if self._injected:
            raise ValueError("CPU injected components do not certify native layout/counts")
        enc, net = self.encoding.native_tcnn_module, self.network.native_tcnn_module
        groups = self.native_parameter_groups()
        if (enc.n_input_dims() != 3 or enc.n_output_dims() != HASH_DIM
                or net.n_input_dims() != NETWORK_INPUT_DIM or net.n_output_dims() != PADDED_OUTPUT_DIM
                or len(groups["encoding"]) != 1 or len(groups["network"]) != 1
                or self.encoding.params.numel() != EXPECTED_ENCODING_PARAMETERS
                or self.network.params.numel() != EXPECTED_NETWORK_PARAMETERS
                or self.encoding.params.dtype != torch.float32 or self.network.params.dtype != torch.float32
                or self.encoding.dtype != torch.float16 or self.network.dtype != torch.float16
                or sum(p.numel() for p in self.parameters()) != EXPECTED_NATIVE_PARAMETERS):
            raise ValueError("Native public dimensions/FP32 masters/FP16 compute/counts differ")
        return {"encoding_public_input_dims": 3, "encoding_public_output_dims": self.encoding.n_output_dims,
            "encoding_native_output_dims": enc.n_output_dims(), "network_public_input_dims": self.network.n_input_dims,
            "network_native_input_dims": net.n_input_dims(), "network_public_output_dims": self.network.n_output_dims,
            "network_native_padded_output_dims": net.n_output_dims(), "Identity_conditions": NATIVE_CONDITION_DIM,
            "nominal_encoded_channels": NOMINAL_INPUT_DIM, "explicit_one_padding_channels": EXPLICIT_PADDING,
            "encoding_parameters": self.encoding.params.numel(), "network_parameters": self.network.params.numel(),
            "native_total_parameters": sum(p.numel() for p in self.parameters()),
            "public_parameter_buffers": ["encoding.params", "network.params"],
            "master_dtype": str(self.encoding.params.dtype), "compute_dtype": str(self.encoding.dtype),
            "flat_buffer_offsets_inferred": False, "new_learned_head_parameters": 0}

    def forward(self, xyz: Tensor, features: Tensor):
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or features.shape != (len(xyz), NATIVE_CONDITION_DIM)
                or xyz.device != self.feature_mask.device or features.device != xyz.device):
            raise ValueError("Split decoder needs same-device xyz[N,3]/conditions[N,107]")
        encoded = self.encoding(xyz.detach().float().contiguous()).float()
        masked = encoded * self.feature_mask
        if self.capture_activation_gradients:
            self.encoded_activation, self.masked_activation = encoded, masked
            if encoded.requires_grad:
                encoded.retain_grad()
        else:
            self.encoded_activation = self.masked_activation = None
        values = torch.cat((masked, features.float(), features.new_ones((len(features), EXPLICIT_PADDING), dtype=torch.float32)), -1)
        return self.network(values.contiguous()).float()

    @property
    def source_config(self):
        grid = dict(self.config.encoding_config()["nested"][0]); grid.pop("n_dims_to_encode")
        return {"implementation": "separate official Encoding/Network and nonlearned post-Encoding level mask",
            "backend": self.backend, "source": TCNN_SOURCE, "source_commit": TCNN_COMMIT,
            "outer_interface_config": asdict(self.config), "outer_interface_only": True,
            "encoding": {"class": "tinycudann.Encoding", "n_input_dims": 3, "n_output_dims": HASH_DIM,
                "encoding_config": grid, "seed": self.config.seed, "parameters": sum(p.numel() for p in self.encoding.parameters()),
                "public_parameter": "encoding.params", "compute_dtype": str(getattr(self.encoding, "dtype", torch.float32))},
            "network": {"class": "tinycudann.Network", "n_input_dims": NETWORK_INPUT_DIM,
                "n_output_dims": 9, "native_padded_output_dims_source_derived": PADDED_OUTPUT_DIM,
                "network_config": self.config.network_config(), "seed": self.config.seed,
                "parameters": sum(p.numel() for p in self.network.parameters()), "public_parameter": "network.params",
                "compute_dtype": str(getattr(self.network, "dtype", torch.float32))},
            "layout": "hash24(level-major) *mask24, Identity107 unchanged, explicit13 ones -> public Network144",
            "mask": {"mode": self.mode, "buffer": "feature_mask", "learned": False,
                "values": self.feature_mask.detach().cpu().tolist(), "active_levels": 12 if self.mode == "full12" else 8},
            "master_dtype": "torch.float32", "native_buffers": ["encoding.params", "network.params"],
            "jit_fusion": False, "learned_gate": False, "LoD_used": False, "new_learned_parameters": 0,
            "raw_xyz_after_encoding": False, "coordinate_derivatives": False, "flat_buffer_offsets_inferred": False,
            "native_layout_verified": not self._injected,
            "initialization_limit": "same fresh split pair seeds/order; no equality to historical fused initialization assumed",
            "performance_limit": "all12 hash levels computed; materialized24 activations and separate native calls",
            "CPU_injection_limit": "Torch interface checks do not certify native dimensions/counts/precision" if self._injected else None}


class HashBandwidthPhysicsField(GraphContextPhysicsField):
    def __init__(self, config: HashBandwidthFieldConfig | None = None, device="cuda", decoder=None):
        config = config or HashBandwidthFieldConfig()
        if not isinstance(config, HashBandwidthFieldConfig):
            raise TypeError("Expected HashBandwidthFieldConfig")
        expected = config.conditioned_config().decoder_config()
        decoder = decoder or SplitHashBandwidthDecoder(expected, config.hash_bandwidth_mode, device)
        if not isinstance(decoder, SplitHashBandwidthDecoder) or decoder.mode != config.hash_bandwidth_mode:
            raise ValueError("Injected decoder must match the fixed bandwidth mode")
        super().__init__(config, device=device, decoder=decoder)

    def native_parameter_groups(self):
        return self.decoder.native_parameter_groups()

    def validate_native_layout(self):
        result = self.decoder.validate_native_layout()
        if sum(p.numel() for p in self.parameters()) != EXPECTED_FULL_PARAMETERS:
            raise ValueError("Native full model parameter count differs")
        return {**result, "full_model_parameters": sum(p.numel() for p in self.parameters())}

    @property
    def source_config(self):
        result = super().source_config
        result["conditioned_field"]["decoder"] = self.decoder.source_config
        result["conditioned_field"]["conditioning"] = "masked original25/local13/Euclidean5, actual query32, actual global32"
        result.update(decoder_implementation=self.decoder.source_config, actual_decoder_backend=self.backend,
            native_parameter_buffers=["field.decoder.encoding.params", "field.decoder.network.params"],
            native_encoding_parameters=sum(p.numel() for p in self.decoder.encoding.parameters()),
            native_network_parameters=sum(p.numel() for p in self.decoder.network.parameters()),
            full_model_parameters=sum(p.numel() for p in self.parameters()),
            native_condition_dim=NATIVE_CONDITION_DIM, native_identity_condition_dim=NATIVE_CONDITION_DIM,
            native_nominal_encoded_channels=NOMINAL_INPUT_DIM, native_public_network_input_dim=NETWORK_INPUT_DIM,
            native_explicit_one_padding_channels=EXPLICIT_PADDING, native_public_encoding_input_dim=3,
            hash_bandwidth_mode=self.config.hash_bandwidth_mode, hash_modulation_enabled=False,
            native_layout=self.validate_native_layout() if not self.decoder._injected else None,
            CPU_injection_limit="Torch interface checks do not certify native CUDA" if self.decoder._injected else None,
            initialization_policy="complete named trainable parameters equal; only nonlearned feature_mask differs; natural initial outputs need not equal")
        result.pop("native_source_derived_padded_encoded_channels", None)
        result.pop("native_raw_input_dim", None)
        return result


__all__ = ["HashBandwidthFieldConfig", "HashBandwidthPhysicsField", "SplitHashBandwidthDecoder"]
