"""Official tiny-cuda-nn hash-grid neural fields with explicit native precision.

This is a separate implementation from the earlier torch-head baselines.
The default architecture matches the slide's stated NGP dimensions. TCNN's
network has no bias parameters. The historical default uses FP16 FullyFusedMLP
with FP32 master parameters; native FP32 uses a separately compiled CutlassMLP.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn


TCNN_COMMIT = "3daa6e5b8b8e0fa17971a1a20acfbb3754494b38"
TCNN_SOURCE = f"https://github.com/NVlabs/tiny-cuda-nn/tree/{TCNN_COMMIT}"


@dataclass(frozen=True)
class TCNNFieldConfig:
    out_channels: int = 1
    input_mode: Literal["xyz", "both"] = "xyz"
    condition_dim: int = 25
    n_levels: int = 12
    n_features_per_level: int = 2
    log2_hashmap_size: int = 19
    base_resolution: int = 8
    per_level_scale: float = 1.6
    hidden_dim: int = 64
    hidden_layers: int = 2
    seed: int = 1337
    native_precision: Literal["fp16", "fp32"] = "fp16"

    def __post_init__(self) -> None:
        if self.native_precision not in ("fp16", "fp32"):
            raise ValueError("Native precision must be fp16 or fp32.")
        if self.input_mode not in ("xyz", "both"):
            raise ValueError("TCNN fields accept input_mode='xyz' or 'both'.")
        if self.out_channels < 1 or self.condition_dim < 0:
            raise ValueError("Invalid output/condition dimensions.")
        if self.input_mode == "both" and self.condition_dim < 1:
            raise ValueError("The both mode needs condition channels.")
        if self.hidden_dim not in (16, 32, 64, 128) or self.hidden_layers < 1:
            raise ValueError("FullyFusedMLP needs width 16/32/64/128 and positive depth.")
        if self.n_levels < 1 or self.n_features_per_level < 1:
            raise ValueError("Hash-grid levels/features must be positive.")
        if not 1 <= self.log2_hashmap_size <= 30 or self.base_resolution < 2 or self.per_level_scale < 1:
            raise ValueError("Invalid hash-grid size/resolution/growth.")

    @property
    def input_dim(self) -> int:
        return 3 + (self.condition_dim if self.input_mode == "both" else 0)

    def encoding_config(self) -> dict[str, Any]:
        grid = {"otype": "HashGrid", "n_levels": self.n_levels,
                "n_features_per_level": self.n_features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": self.base_resolution,
                "per_level_scale": self.per_level_scale,
                "interpolation": "Linear", "hash": "CoherentPrime"}
        if self.input_mode == "xyz":
            return grid
        return {"otype": "Composite", "nested": [
            {**grid, "n_dims_to_encode": 3},
            {"otype": "Identity", "n_dims_to_encode": self.condition_dim,
             "scale": 1.0, "offset": 0.0},
        ]}

    def network_config(self) -> dict[str, Any]:
        return {"otype": "CutlassMLP" if self.native_precision == "fp32" else "FullyFusedMLP", "activation": "ReLU",
                "output_activation": "None", "n_neurons": self.hidden_dim,
                "n_hidden_layers": self.hidden_layers}

    def record(self) -> dict[str, Any]:
        return {**asdict(self), "source": TCNN_SOURCE, "source_commit": TCNN_COMMIT,
                "encoding": self.encoding_config(), "network": self.network_config(),
                "sample_id": False, "raw_xyz_after_encoding": False,
                "bias_parameters": False, "jit_fusion": False,
                "precision": f"native TCNN {self.native_precision.upper()} computation, FP32 master parameters, float32 returned output",
                "alignment": "native TCNN channel/batch padding retained",
                "layout": "xyz[N,3], features[N,condition_dim] -> prediction[N,out_channels]"}


class TCNNField(nn.Module):
    """An official fully fused field; prediction is always shaped [N,C].

    Normalize xyz to the chosen training domain before calling this module.
    Feature scaling and label transforms belong to the dataset/trainer.
    No lookup cache is used, because the native fused path handles encoding.
    """

    def __init__(self, out_channels: int = 1,
                 input_mode: Literal["xyz", "both"] = "xyz",
                 condition_dim: int = 25, device: torch.device | str = "cuda",
                 config: TCNNFieldConfig | None = None, seed: int = 1337) -> None:
        super().__init__()
        self.config = config or TCNNFieldConfig(out_channels, input_mode, condition_dim, seed=seed)
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("Official tiny-cuda-nn fields require a CUDA device.")
        try:
            import tinycudann as tcnn
        except (ImportError, OSError) as error:
            raise RuntimeError("The official tiny-cuda-nn CUDA extension is unavailable.") from error
        with torch.cuda.device(device):
            self.native = tcnn.NetworkWithInputEncoding(
                n_input_dims=self.config.input_dim,
                n_output_dims=self.config.out_channels,
                encoding_config=self.config.encoding_config(),
                network_config=self.config.network_config(), seed=self.config.seed)
            # Use the declared native backend without introducing JIT fusion.
            self.native.jit_fusion = False
        expected = torch.float32 if self.config.native_precision == "fp32" else torch.float16
        # Master parameters alone are insufficient: the extension can cast them
        # to half internally. Check the native module's actual compute dtype.
        if self.native.dtype != expected:
            raise RuntimeError(f"TCNN native dtype {self.native.dtype} does not match requested {expected}; use a matching compiled extension.")
        self.backend = "tcnn_cutlass_fp32" if self.config.native_precision == "fp32" else "tcnn_fully_fused"
        self.source_config = {**self.config.record(), "actual_native_dtype": str(self.native.dtype),
                              "actual_native_hyperparams": self.native.native_tcnn_module.hyperparams()}
        self.to(device)

    def forward(self, xyz: Tensor, features: Tensor | None = None) -> Tensor:
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape [N,3].")
        if xyz.device != self.native.params.device:
            raise ValueError("Coordinates and native parameters must share a device.")
        values = xyz.float()
        if self.config.input_mode == "both":
            if features is None or features.shape != (len(xyz), self.config.condition_dim):
                raise ValueError("The both mode needs features[N,condition_dim].")
            if features.device != xyz.device:
                raise ValueError("Coordinates and features must share a device.")
            values = torch.cat((values, features.float()), dim=-1)
        return self.native(values.contiguous()).float()

    def prepare_lookup(self, xyz: Tensor, chunk_size: int = 65536) -> None:
        return None


__all__ = ["TCNN_COMMIT", "TCNN_SOURCE", "TCNNFieldConfig", "TCNNField"]
