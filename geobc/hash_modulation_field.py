"""Matched zero/actual geometry-context modulation of exposed hash features.

This standard multiplicative gate is a GeoBC candidate, not a new algorithm.
Both branches keep actual query/global conditioning and use the same separately
initialized official Encoding and Network modules. Old fused checkpoints are
not an initialization/control for this new matched architecture.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .query_context_field import QueryContextFieldConfig, QueryContextPhysicsField
from .tcnn_fields import TCNNFieldConfig, TCNN_COMMIT, TCNN_SOURCE

HASH_DIM = 24
CONDITION_DIM = 102
NETWORK_INPUT_DIM = 126
PADDED_INPUT_DIM = 128
PADDED_OUTPUT_DIM = 16
EXPECTED_ENCODING_PARAMETERS = 7_733_632
EXPECTED_NETWORK_PARAMETERS = 34_816


@dataclass(frozen=True)
class HashModulationFieldConfig(QueryContextFieldConfig):
    query_mode: Literal["actual"] = "actual"
    hash_modulation_mode: Literal["zero", "actual"] = "zero"

    def __post_init__(self):
        super().__post_init__()
        if (self.hash_modulation_mode not in ("zero", "actual") or self.query_mode != "actual"
                or self.mode != "linear_basis" or self.geometry_mode != "shape"
                or self.hidden_dim != 128 or self.hidden_layers != 2 or self.cloud_points != 1024):
            raise ValueError("Fixed hash-gate pair requires linear9basis, actual query/global cloud, width128x2/cloud1024")

    def record(self):
        result = super().record()
        result.update(method="GeoBC geometry-context multiplicative hash-feature gate candidate",
            claim="standard feature-modulation ablation; no new-method or generalization claim",
            hash_modulation={"mode": self.hash_modulation_mode, "hash_features": HASH_DIM,
                "context": "query32 then global32, actual on BOTH branches",
                "head": [64, 24], "head_parameters": 1560,
                "formula": "hash24 * (1 + tanh(Linear64to24(query32,global32)))",
                "initialization": "gate weight and bias exactly0; multiplier initially1",
                "scale_open_bounds": [0, 2], "extra_strength_coefficient": None,
                "zero_control": "compute same tanh gate then multiply its delta by0; gate gradients present and exactly0",
                "load_dependency": "geometry/BC contexts only, no alpha or original force/moment7:13",
                "input_order": "hash24, masked original38, actual query32, actual global32",
                "fresh_pair": "same split Encoding/Network seeds, same gate/encoder/projection construction; old fused model is not the matched control"})
        result["capacity_cost"]["native_padding_policy"] = "split Network uses Identity126->128 with two constant1 padding channels; actual public module parameter counts recorded"
        result["capacity_cost"]["new_hash_gate_parameters"] = 1560
        result["input_layout"] = "masked original38 + actual query32 + actual global32 are102 conditions; prepend exposed gated hash24, giving Network Identity126->128"
        return result


class SplitHashModulationDecoder(nn.Module):
    """Frozen field's decoder interface, implemented with two public TCNN modules.

    CPU component injection checks the Torch interface only. Official native
    construction and parameter/gradient checks require scheduled CUDA.
    """
    def __init__(self, config: TCNNFieldConfig, mode: str, device="cuda", *, encoding=None, network=None):
        super().__init__()
        expected = TCNNFieldConfig(out_channels=9, input_mode="both", condition_dim=102,
                                  hidden_dim=128, hidden_layers=2, seed=config.seed)
        if config != expected or mode not in ("zero", "actual"):
            raise ValueError("Split decoder requires the fixed12x2 hash/condition102/MLP128x2/9basis layout")
        if (encoding is None) != (network is None):
            raise ValueError("Inject both native-interface components or neither")
        self.config = config  # Exact outer interface expected by the frozen injection contract.
        self.mode = mode
        self._injected = encoding is not None
        device = torch.device(device)
        if not self._injected:
            if device.type != "cuda":
                raise ValueError("Official split TCNN modules require CUDA")
            try:
                import tinycudann as tcnn
            except (ImportError, OSError) as error:
                raise RuntimeError("Official tiny-cuda-nn CUDA extension unavailable") from error
            grid_config = dict(config.encoding_config()["nested"][0])
            grid_config.pop("n_dims_to_encode")
            with torch.cuda.device(device):
                encoding = tcnn.Encoding(n_input_dims=3, encoding_config=grid_config,
                                         seed=config.seed, dtype=torch.float16)
                network = tcnn.Network(n_input_dims=NETWORK_INPUT_DIM, n_output_dims=9,
                                       network_config=config.network_config(), seed=config.seed)
                encoding.jit_fusion = False
                network.jit_fusion = False
        if (encoding.n_input_dims != 3 or encoding.n_output_dims != HASH_DIM
                or network.n_input_dims != NETWORK_INPUT_DIM or network.n_output_dims != 9):
            raise ValueError("Public standalone Encoding/Network dimensions differ")
        self.encoding = encoding
        self.network = network
        # Both branches perform the same Linear construction/RNG calls, then
        # explicitly clear its parameters. Initial state equality is tested.
        self.gate = nn.Linear(64, HASH_DIM, dtype=torch.float32)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.backend = "injected_cpu_split_interface" if self._injected else "tcnn_split_encoding_network_hash_gate"
        self.to(device)
        if not self._injected:
            self.validate_native_layout()

    def native_parameter_groups(self):
        """Public modules only: no guessed offsets or fabricated flat buffer."""
        return {"encoding": list(self.encoding.parameters()), "network": list(self.network.parameters())}

    def validate_native_layout(self):
        if self._injected:
            raise ValueError("Injected CPU components cannot certify native CUDA layout")
        enc = self.encoding.native_tcnn_module
        net = self.network.native_tcnn_module
        if (self.encoding.params.numel() != EXPECTED_ENCODING_PARAMETERS
                or self.network.params.numel() != EXPECTED_NETWORK_PARAMETERS
                or enc.n_output_dims() != HASH_DIM or net.n_input_dims() != NETWORK_INPUT_DIM
                or net.n_output_dims() != PADDED_OUTPUT_DIM
                or self.encoding.params.dtype != torch.float32 or self.network.params.dtype != torch.float32):
            raise ValueError("Official native dimensions, FP32 masters or parameter counts differ from source-derived contract")
        return {"encoding_public_output": self.encoding.n_output_dims,
                "encoding_native_output": enc.n_output_dims(),
                "network_public_input": self.network.n_input_dims,
                "network_native_input": net.n_input_dims(),
                "network_internal_MLP_input": PADDED_INPUT_DIM,
                "network_internal_MLP_input_proof": "official Identity alignment16 and measured34816 MLP parameters",
                "network_public_output": self.network.n_output_dims,
                "network_native_padded_output": net.n_output_dims(),
                "encoding_parameters": self.encoding.params.numel(),
                "network_parameters": self.network.params.numel(),
                "gate_parameters": sum(p.numel() for p in self.gate.parameters())}

    def gate_delta(self, features: Tensor):
        if features.ndim != 2 or features.shape[1] != CONDITION_DIM:
            raise ValueError("Decoder conditioning must be original38/query32/global32")
        value = torch.tanh(self.gate(features[:, 38:102].float()))
        return value * 0 if self.mode == "zero" else value

    def forward(self, xyz: Tensor, features: Tensor):
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or features.shape != (len(xyz), CONDITION_DIM)
                or xyz.device != self.gate.weight.device or features.device != xyz.device):
            raise ValueError("Split decoder requires same-device xyz[N,3]/conditions[N,102]")
        # Native Encoding parameters remain differentiable with detached xyz.
        hashed = self.encoding(xyz.detach().float().contiguous()).float()
        modulated = hashed * (1 + self.gate_delta(features))
        inputs = torch.cat((modulated, features.float()), dim=-1).contiguous()
        return self.network(inputs).float()

    @property
    def source_config(self):
        grid = dict(self.config.encoding_config()["nested"][0])
        grid.pop("n_dims_to_encode")
        return {"implementation": "separate official Encoding and Network with Torch multiplicative geometry-context gate",
            "backend": self.backend, "source": TCNN_SOURCE, "source_commit": TCNN_COMMIT,
            "outer_interface_config": asdict(self.config),
            "encoding": {"class": "tinycudann.Encoding", "n_input_dims": 3,
                "n_output_dims": self.encoding.n_output_dims, "encoding_config": grid,
                "seed": self.config.seed, "compute_dtype": "torch.float16", "master_dtype": "torch.float32",
                "parameters": sum(p.numel() for p in self.encoding.parameters()), "public_parameter": "encoding.params"},
            "network": {"class": "tinycudann.Network", "n_input_dims": NETWORK_INPUT_DIM,
                "network_config": self.config.network_config(), "seed": self.config.seed,
                "internal_Identity_padded_input": PADDED_INPUT_DIM, "input_padding_count": 2,
                "input_padding_value": 1, "padding_proof": "official Identity encoding pads with1, FullyFusedMLP alignment16",
                "requested_output_dims": 9, "native_padded_output_dims": PADDED_OUTPUT_DIM,
                "master_dtype": "torch.float32", "compute_dtype": "torch.float16",
                "parameters": sum(p.numel() for p in self.network.parameters()), "public_parameter": "network.params"},
            "gate": {"head": [64, 24], "parameters": sum(p.numel() for p in self.gate.parameters()),
                "mode": self.mode, "context_slice": [38, 102], "context_order": ["query32", "global32"],
                "formula": "hash * (1 + tanh(gate(context)))", "zero_control": "tanh delta multiplied by0",
                "weight_bias_initialization": "exact0", "scale_open_bounds": [0, 2]},
            "layout": "hash24 then original masked38/query32/global32; nominal126 -> nativeMLP128",
            "native_buffers": ["encoding.params", "network.params"], "fake_flat_buffer": False,
            "jit_fusion": False, "raw_xyz_after_encoding": False, "coordinate_derivatives": False,
            "hash_compute_roundtrip": "Encoding FP16 output -> Torch FP32 modulation -> Network internal FP16 Identity",
            "initialization_limit": "split modules use independent supplied seeds; no exact match to old fused grid initialization claimed",
            "cost": "materialized hash24 plus extra native call/autograd boundary; scheduled smoke records cost",
            "native_layout_verified": not self._injected,
            "CPU_injection_limit": "injected components check interface only, their dimensions/counts are not native CUDA evidence" if self._injected else None}


class HashModulationPhysicsField(QueryContextPhysicsField):
    """Original query bundle/basis API with a truthful split-decoder record."""
    def __init__(self, config: HashModulationFieldConfig | None = None, device="cuda", decoder=None):
        config = config or HashModulationFieldConfig()
        if not isinstance(config, HashModulationFieldConfig):
            raise TypeError("Expected HashModulationFieldConfig")
        expected = config.conditioned_config().decoder_config()
        if decoder is None:
            decoder = SplitHashModulationDecoder(expected, config.hash_modulation_mode, device)
        if not isinstance(decoder, SplitHashModulationDecoder) or decoder.mode != config.hash_modulation_mode:
            raise ValueError("Injected split decoder must have the same gate mode")
        super().__init__(config, device=device, decoder=decoder)

    @property
    def gate_head(self):
        return self.decoder.gate

    @property
    def source_config(self):
        result = super().source_config
        # Frozen base records the outer TCNN interface as Composite; replace
        # that decoder record with the actual split implementation, explicitly.
        result["conditioned_field"]["decoder"] = self.decoder.source_config
        result.update(decoder_implementation=self.decoder.source_config,
            actual_decoder_backend=self.decoder.backend,
            native_encoding_parameters=sum(p.numel() for p in self.decoder.encoding.parameters()),
            native_network_parameters=sum(p.numel() for p in self.decoder.network.parameters()),
            hash_gate_parameters=sum(p.numel() for p in self.gate_head.parameters()),
            native_parameter_buffers=["field.decoder.encoding.params", "field.decoder.network.params"],
            full_model_parameters=sum(p.numel() for p in self.parameters()),
            native_condition_dim=102, native_identity_condition_dim=126,
            full_native_network_input_dim=126)
        return result


__all__ = ["HashModulationFieldConfig", "HashModulationPhysicsField", "SplitHashModulationDecoder"]
