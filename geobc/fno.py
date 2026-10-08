"""Four-block, coordinate-free 3D FNO matching the slide 10--13 dimensions.

The spectral transform follows the author's four-quadrant implementation:
https://github.com/neuraloperator/physics_informed/blob/3b6bc307c63c64057d0496753bf1af5f44ac5108/models/basics.py
Paper: https://arxiv.org/abs/2010.08895v3

Pointwise layers are Conv1d(kernel_size=1), equivalent to a linear map at
each voxel. No xyz channels, sample IDs, learned coordinate encodings, or
mask-dependent output interpolation are added to the architecture.
The source experiment directory and its 18-channel schema are unavailable;
the grid preparation module explicitly records its replacement schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


FNO_SOURCES = {
    "paper": "https://arxiv.org/abs/2010.08895v3",
    "code_commit": "3b6bc307c63c64057d0496753bf1af5f44ac5108",
    "spectral_code": "https://github.com/neuraloperator/physics_informed/blob/3b6bc307c63c64057d0496753bf1af5f44ac5108/models/basics.py",
    "pointwise_code": "https://github.com/neuraloperator/physics_informed/blob/3b6bc307c63c64057d0496753bf1af5f44ac5108/models/fourier3d.py",
}


@dataclass(frozen=True)
class FNOConfig:
    in_channels: int = 18
    out_channels: int = 4
    width: int = 16
    modes: int = 12
    layers: int = 4
    head_width: int = 128
    padding: int = 0
    checkpoint_blocks: bool = False

    def record(self) -> dict[str, Any]:
        return {**asdict(self), "layout": "B,C,X,Y,Z", "xyz_channels": False,
                "fft_norm": "backward", "sources": FNO_SOURCES,
                "unknown_reference_details": ["18 input channel definitions", "padding", "projection width", "optimizer/scaling schedule"]}


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels, self.out_channels, self.modes = in_channels, out_channels, modes
        scale = 1.0 / (in_channels * out_channels)
        # Real storage preserves the usual independent real/imaginary Adam
        # updates while allowing fused Adam, which rejects complex Parameters.
        self.weights = nn.ParameterList([
            nn.Parameter(scale * torch.rand(in_channels, out_channels, modes, modes, modes, 2))
            for _ in range(4)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nx, ny, nz = x.shape[-3:]
        m = self.modes
        if 2 * m > nx or 2 * m > ny or m > nz // 2 + 1:
            raise ValueError(f"Modes {m} overlap or exceed the FFT grid {nx, ny, nz}")
        # Non-power-of-two 192^3 FFTs must remain float32 under CUDA autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            coefficients = torch.fft.rfftn(x.float(), dim=(-3, -2, -1), norm="backward")
            # High z modes are zero. irfftn(s=...) pads the omitted modes;
            # this avoids allocating a mostly-zero full 192x192x97 spectrum.
            result = torch.zeros(x.shape[0], self.out_channels, nx, ny, m,
                                 dtype=torch.complex64, device=x.device)
            slices = [(slice(0, m), slice(0, m)), (slice(-m, None), slice(0, m)),
                      (slice(0, m), slice(-m, None)), (slice(-m, None), slice(-m, None))]
            for weight, (sx, sy) in zip(self.weights, slices):
                result[:, :, sx, sy, :] = torch.einsum(
                    "bixyz,ioxyz->boxyz", coefficients[:, :, sx, sy, :m],
                    torch.view_as_complex(weight.contiguous()))
            return torch.fft.irfftn(result, s=(nx, ny, nz), dim=(-3, -2, -1), norm="backward")


class FNO3d(nn.Module):
    """Forward: float [B,in_channels,Rx,Ry,Rz] -> [B,out_channels,Rx,Ry,Rz]."""

    def __init__(self, in_channels: int = 18, out_channels: int = 4, width: int = 16,
                 modes: int = 12, layers: int = 4, padding: int = 0,
                 head_width: int = 128, checkpoint_blocks: bool = False) -> None:
        super().__init__()
        if min(in_channels, out_channels, width, modes, layers, head_width) < 1 or padding < 0:
            raise ValueError("FNO dimensions must be positive, padding nonnegative")
        self.config = FNOConfig(in_channels, out_channels, width, modes, layers, head_width,
                                padding, checkpoint_blocks)
        self.lift = nn.Conv1d(in_channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv3d(width, width, modes) for _ in range(layers)])
        self.pointwise = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.project1 = nn.Conv1d(width, head_width, 1)
        self.project2 = nn.Conv1d(head_width, out_channels, 1)

    def _block(self, x: torch.Tensor, index: int) -> torch.Tensor:
        local = self.pointwise[index](x.flatten(2)).reshape_as(x)
        result = self.spectral[index](x) + local
        return F.gelu(result) if index < self.config.layers - 1 else result

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != self.config.in_channels:
            raise ValueError(f"Expected [B,{self.config.in_channels},X,Y,Z]")
        spatial = inputs.shape[-3:]
        x = self.lift(inputs.flatten(2)).reshape(inputs.shape[0], self.config.width, *spatial)
        padding = self.config.padding
        if padding:
            x = F.pad(x, (0, padding, 0, padding, 0, padding))
        for index in range(self.config.layers):
            if self.config.checkpoint_blocks and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(lambda value, i=index: self._block(value, i), x, use_reentrant=False)
            else:
                x = self._block(x, index)
        if padding:
            x = x[..., :spatial[0], :spatial[1], :spatial[2]]
        x = F.gelu(self.project1(x.flatten(2)))
        return self.project2(x).reshape(inputs.shape[0], self.config.out_channels, *spatial)


def query_grid(grid: torch.Tensor, query_indices: torch.Tensor, query_weights: torch.Tensor,
               chunk_size: int = 65536) -> torch.Tensor:
    """Masked trilinear query cache -> physical or encoded nodal values.

    grid [C,X,Y,Z] returns [N,C]; [B,C,X,Y,Z] returns [B,N,C].
    Static query_weights already include occupancy masking and renormalization.
    Nodes without an occupied corner have all-zero weights and return zero.
    Gradients flow to grid values. No nearest-node fallback or query truncation.
    """
    unbatched = grid.ndim == 4
    if unbatched:
        grid = grid.unsqueeze(0)
    if grid.ndim != 5 or query_indices.ndim != 2 or query_indices.shape[1] != 8:
        raise ValueError("Expected grid [B,C,X,Y,Z], query indices [N,8]")
    if query_weights.shape != query_indices.shape or chunk_size < 1:
        raise ValueError("Invalid query weights or chunk size")
    flat = grid.flatten(2)
    chunks = []
    for start in range(0, len(query_indices), chunk_size):
        indices = query_indices[start:start + chunk_size].to(device=grid.device, dtype=torch.long)
        weights = query_weights[start:start + chunk_size].to(device=grid.device, dtype=grid.dtype)
        values = flat[:, :, indices]
        chunks.append((values * weights[None, None]).sum(-1).transpose(1, 2))
    result = torch.cat(chunks, dim=1)
    return result[0] if unbatched else result


query_nodes = query_grid

__all__ = ["FNO3d", "FNOConfig", "SpectralConv3d", "query_grid", "query_nodes", "FNO_SOURCES"]
