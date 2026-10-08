"""Shared coordinate neural fields used by the DeepJEB overfitting experiments.

The hash-grid defaults reproduce slides 15/16: 12 levels, 2 features per
level, log2 table cap 19, base resolution 8 and growth 1.6, followed by a
64-wide MLP with two hidden layers and an unrestricted scalar output.
There is deliberately no sample ID or per-case embedding.

``input_mode='xyz'`` uses only encoded coordinates. ``'both'`` concatenates
the encoded coordinates with the 25 BC/material channels; raw xyz is not
duplicated after hash encoding. ``'features'`` is a plain-MLP ablation.
Coordinates must already have been normalized to [0, 1] by the dataset.
The implementation does not clamp them, because clamping masks data errors.

The torch fallback follows tiny-cuda-nn's grid convention (half-cell
stagger, dense coarse levels, CoherentPrime hashed fine levels):
https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/common_device.h
https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/encodings/grid.h
Backend checkpoints are separate; identical initialization across kernels
and bitwise parity are not assumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HashGridConfig:
    n_levels: int = 12
    n_features_per_level: int = 2
    log2_hashmap_size: int = 19
    base_resolution: int = 8
    per_level_scale: float = 1.6

    def __post_init__(self) -> None:
        if self.n_levels < 1 or self.n_features_per_level < 1:
            raise ValueError("Hash-grid levels and features must be positive.")
        if not 1 <= self.log2_hashmap_size <= 30:
            raise ValueError("log2_hashmap_size must be in [1, 30].")
        if self.base_resolution < 2 or self.per_level_scale < 1:
            raise ValueError("base_resolution must be >= 2 and scale must be >= 1.")

    @property
    def output_dim(self) -> int:
        return self.n_levels * self.n_features_per_level

    def tcnn_config(self) -> dict:
        return {"otype": "HashGrid", **asdict(self), "interpolation": "Linear", "hash": "CoherentPrime"}


@dataclass(frozen=True)
class ModelConfig:
    encoder: Literal["hashgrid", "mlp"] = "hashgrid"
    input_mode: Literal["xyz", "features", "both"] = "both"
    condition_dim: int = 25
    hidden_dim: int = 64
    hidden_layers: int = 2
    hash_grid: HashGridConfig = HashGridConfig()

    def __post_init__(self) -> None:
        if self.encoder not in ("hashgrid", "mlp"):
            raise ValueError("encoder must be 'hashgrid' or 'mlp'.")
        if self.input_mode not in ("xyz", "features", "both"):
            raise ValueError("input_mode must be 'xyz', 'features' or 'both'.")
        if self.encoder == "hashgrid" and self.input_mode == "features":
            raise ValueError("A hash grid needs xyz; use encoder='mlp' for a features-only ablation.")
        if self.condition_dim < 0 or self.hidden_dim < 1 or self.hidden_layers < 1:
            raise ValueError("Invalid feature width or MLP size.")
        if self.input_mode == "features" and self.condition_dim == 0:
            raise ValueError("A features-only MLP requires at least one feature.")


@dataclass(frozen=True)
class HashGridLookup:
    """Fixed-coordinate interpolation cache; no learned values are cached.

    Each row defines its own coordinate through [levels, 8] indices and
    interpolation weights. Use ``select`` with the same row indices used
    for features/targets. Cached evaluation intentionally takes no xyz
    argument, so it cannot silently pair a cache with different positions.
    Coordinates cannot require gradients when creating this cache.
    Float32 weights plus int32 indices cost 768 bytes/point at 12 levels.
    """

    indices: Tensor
    weights: Tensor
    signature: tuple

    def __len__(self) -> int:
        return self.indices.shape[0]

    @property
    def device(self) -> torch.device:
        return self.indices.device

    def select(self, rows: Tensor | slice) -> "HashGridLookup":
        return HashGridLookup(self.indices[rows], self.weights[rows], self.signature)

    def to(self, device: torch.device | str) -> "HashGridLookup":
        return HashGridLookup(self.indices.to(device), self.weights.to(device), self.signature)


class TorchHashGrid(nn.Module):
    """Vectorized 3D trilinear hash grid, runnable on torch CPU or CUDA."""

    def __init__(self, config: HashGridConfig = HashGridConfig()) -> None:
        super().__init__()
        self.config = config
        self.output_dim = config.output_dim
        scales = [config.base_resolution * config.per_level_scale**level - 1.0
                  for level in range(config.n_levels)]
        resolutions = [math.ceil(scale) + 1 for scale in scales]
        # Tiny-cuda-nn rounds dense level allocations to multiples of 8.
        sizes = [min(((resolution**3 + 7) // 8) * 8, 1 << config.log2_hashmap_size)
                 for resolution in resolutions]
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        if offsets[-1] > torch.iinfo(torch.int32).max:
            raise ValueError("This grid exceeds the int32 cached-index capacity.")
        self.signature = (config.n_levels, config.n_features_per_level,
                          config.log2_hashmap_size, config.base_resolution,
                          config.per_level_scale)
        self.register_buffer("_scales", torch.tensor(scales, dtype=torch.float32), persistent=False)
        self.register_buffer("_resolutions", torch.tensor(resolutions, dtype=torch.int64), persistent=False)
        self.register_buffer("_sizes", torch.tensor(sizes, dtype=torch.int64), persistent=False)
        self.register_buffer("_offsets", torch.tensor(offsets[:-1], dtype=torch.int64), persistent=False)
        self.register_buffer("_dense", torch.tensor([s >= r**3 for s, r in zip(sizes, resolutions)]), persistent=False)
        self.register_buffer("_corners", torch.tensor([[i & 1, (i >> 1) & 1, (i >> 2) & 1]
                                                      for i in range(8)], dtype=torch.int64), persistent=False)
        self.table = nn.Parameter(torch.empty(offsets[-1], config.n_features_per_level))
        nn.init.uniform_(self.table, -1.0e-4, 1.0e-4)

    def _lookup(self, xyz: Tensor) -> HashGridLookup:
        _check_xyz(xyz)
        if xyz.device != self.table.device:
            raise ValueError("Coordinates and the model must be on the same device.")
        positions = xyz.float()[:, None, :] * self._scales[None, :, None] + 0.5
        lower = positions.floor()
        fractional = positions - lower
        vertices = lower.to(torch.int64)[:, :, None, :] + self._corners[None, None, :, :]
        weights = torch.where(self._corners[None, None, :, :].bool(),
                              fractional[:, :, None, :],
                              1.0 - fractional[:, :, None, :]).prod(dim=-1)
        resolution = self._resolutions[None, :, None]
        dense_index = vertices[..., 0] + vertices[..., 1] * resolution + vertices[..., 2] * resolution.square()
        # uint32 overflow is explicit; x uses factor 1 for CoherentPrime.
        hashed_index = (vertices[..., 0] ^ (vertices[..., 1] * 2654435761)
                        ^ (vertices[..., 2] * 805459861)) & 0xFFFFFFFF
        index = torch.where(self._dense[None, :, None], dense_index, hashed_index)
        index = index.remainder(self._sizes[None, :, None]) + self._offsets[None, :, None]
        return HashGridLookup(index.to(torch.int32), weights, self.signature)

    def forward(self, xyz: Tensor) -> Tensor:
        return self.forward_cached(self._lookup(xyz))

    @torch.no_grad()
    def prepare_lookup(self, xyz: Tensor, chunk_size: int = 65536) -> HashGridLookup:
        """Precompute indices/weights once for fixed training coordinates.

        Chunking limits temporary [N, levels, corners, xyz] tensors. The
        caller may keep a full cache on CUDA, or cache/select on CPU and
        transfer the selected minibatch. Only the learned table changes
        during training. This path must not be used for PDE input gradients.
        """
        if xyz.requires_grad:
            raise ValueError("Fixed-coordinate caching discards xyz gradients; use forward(xyz).")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        if len(xyz) == 0:
            return self._lookup(xyz)
        chunks = [self._lookup(part) for part in xyz.split(chunk_size)]
        return HashGridLookup(torch.cat([part.indices for part in chunks]),
                              torch.cat([part.weights for part in chunks]), self.signature)

    def forward_cached(self, lookup: HashGridLookup) -> Tensor:
        if lookup.signature != self.signature:
            raise ValueError("The lookup belongs to a different hash-grid configuration.")
        if lookup.device != self.table.device or lookup.weights.device != self.table.device:
            raise ValueError("The lookup and model must be on the same device.")
        if lookup.indices.shape != lookup.weights.shape or lookup.indices.shape[1:] != (self.config.n_levels, 8):
            raise ValueError("Malformed hash-grid lookup.")
        corner_features = F.embedding(lookup.indices, self.table)
        encoded = (corner_features * lookup.weights[..., None]).sum(dim=2)
        return encoded.flatten(start_dim=1)


def _check_xyz(xyz: Tensor) -> None:
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not xyz.is_floating_point():
        raise ValueError("xyz must be a floating tensor of shape [N, 3], normalized to [0, 1].")


class SharedStressField(nn.Module):
    """One coordinate encoder and MLP shared across all DeepJEB cases."""

    def __init__(self, config: ModelConfig, backend: str, encoding: nn.Module | None) -> None:
        super().__init__()
        self.config = config
        self.backend = backend
        self.encoding = encoding
        coordinate_dim = 0 if config.input_mode == "features" else (
            config.hash_grid.output_dim if config.encoder == "hashgrid" else 3)
        condition_dim = config.condition_dim if config.input_mode in ("features", "both") else 0
        self.input_dim = coordinate_dim + condition_dim
        layers: list[nn.Module] = []
        width = self.input_dim
        for _ in range(config.hidden_layers):
            layers.extend([nn.Linear(width, config.hidden_dim), nn.ReLU()])
            width = config.hidden_dim
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def _predict(self, encoded_xyz: Tensor, features: Tensor | None) -> Tensor:
        count = encoded_xyz.shape[0]
        if self.config.input_mode in ("features", "both") and self.config.condition_dim:
            if features is None or not features.is_floating_point():
                raise ValueError("This model requires floating BC/material features.")
            if features.ndim == 1:
                features = features[None, :]
            if features.ndim != 2 or features.shape[1] != self.config.condition_dim or features.shape[0] not in (1, count):
                raise ValueError(f"features must be [N, {self.config.condition_dim}] or one broadcast row.")
            if features.device != encoded_xyz.device:
                raise ValueError("Coordinates and features must be on the same device.")
            inputs = torch.cat((encoded_xyz, features.expand(count, -1)), dim=-1)
        else:
            inputs = encoded_xyz
        # Encoding output can be fp16 in TCNN; the shared torch MLP owns
        # its dtype. CUDA autocast can still accelerate its matrix multiplies.
        inputs = inputs.to(dtype=next(self.mlp.parameters()).dtype)
        return self.mlp(inputs).squeeze(-1)

    def forward(self, xyz: Tensor, features: Tensor | None = None) -> Tensor:
        _check_xyz(xyz)
        if self.config.input_mode == "features":
            encoded = xyz.new_empty((xyz.shape[0], 0))
        elif self.encoding is None:
            encoded = xyz
        else:
            encoded = self.encoding(xyz.float().contiguous())
        return self._predict(encoded, features)

    def prepare_lookup(self, xyz: Tensor, chunk_size: int = 65536) -> HashGridLookup | None:
        if isinstance(self.encoding, TorchHashGrid):
            return self.encoding.prepare_lookup(xyz, chunk_size)
        return None

    def forward_cached(self, lookup: HashGridLookup, features: Tensor | None = None) -> Tensor:
        if not isinstance(self.encoding, TorchHashGrid):
            raise ValueError("Lookup caching is available only for the torch hash-grid backend.")
        return self._predict(self.encoding.forward_cached(lookup), features)


def build_model(backend: Literal["auto", "torch", "tcnn", "triton"] = "auto",
                config: ModelConfig | None = None,
                device: torch.device | str | None = None) -> SharedStressField:
    """Construct the requested model on ``device``; auto prefers TCNN on CUDA.

    TCNN needs its separately installed CUDA extension (typically in WSL).
    An explicit ``tcnn`` request raises on an unavailable extension rather
    than silently changing the experiment. Both hash backends use the same
    torch MLP to preserve the architecture and an output that can be negative.
    Explicit ``triton`` fuses interpolation and table-gradient scatter for
    fixed coordinates. Plain MLP ablations use torch and reject explicit
    ``tcnn``/``triton`` requests because there is no hash grid to accelerate.
    """
    config = config or ModelConfig()
    if backend not in ("auto", "torch", "tcnn", "triton"):
        raise ValueError("backend must be 'auto', 'torch', 'tcnn' or 'triton'.")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoding: nn.Module | None = None
    actual_backend = "torch"
    if config.encoder == "hashgrid":
        if backend == "triton":
            if device.type != "cuda":
                raise ValueError("The Triton backend requires a CUDA device.")
            try:
                from .triton_hashgrid import TritonHashGrid
            except (ImportError, OSError) as error:
                raise RuntimeError("Triton is unavailable in this Python environment.") from error
            encoding = TritonHashGrid(config.hash_grid)
            actual_backend = "triton"
        if backend == "tcnn" and device.type != "cuda":
            raise ValueError("The tiny-cuda-nn backend requires a CUDA device.")
        if backend in ("auto", "tcnn") and device.type == "cuda":
            try:
                import tinycudann as tcnn
                with torch.cuda.device(device):
                    encoding = tcnn.Encoding(n_input_dims=3,
                                             encoding_config=config.hash_grid.tcnn_config())
                actual_backend = "tcnn"
            except (ImportError, OSError, RuntimeError) as error:
                if backend == "tcnn":
                    raise RuntimeError("tiny-cuda-nn is unavailable or incompatible with this CUDA environment.") from error
        if encoding is None:
            encoding = TorchHashGrid(config.hash_grid)
    elif backend in ("tcnn", "triton"):
        raise ValueError("Use backend='torch' for a plain MLP ablation.")
    return SharedStressField(config, actual_backend, encoding).to(device)


__all__ = ["HashGridConfig", "ModelConfig", "HashGridLookup", "TorchHashGrid",
           "SharedStressField", "build_model"]
