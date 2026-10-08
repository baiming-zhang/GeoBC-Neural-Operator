"""Optional fused CUDA evaluation of the torch grid's fixed-coordinate cache.

This backend keeps exactly the torch table layout and model architecture.
One Triton kernel gathers/interpolates eight corners without materializing
the [points, levels, 8, features] tensor. The backward kernel scatters the
table gradient with atomic adds. Coordinates requiring differentiation use
the regular torch path; the fused operator differentiates only the table.
Atomic reductions are nondeterministic, as with CUDA embedding backward.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from .models import HashGridConfig, HashGridLookup, TorchHashGrid


@triton.jit
def _interpolate(TABLE, INDICES, WEIGHTS, OUTPUT, ROWS,
                 FEATURES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    corner = tl.arange(0, 8)
    mask = row[:, None] < ROWS
    index = tl.load(INDICES + row[:, None] * 8 + corner[None, :], mask=mask, other=0)
    weight = tl.load(WEIGHTS + row[:, None] * 8 + corner[None, :], mask=mask, other=0).to(tl.float32)
    for feature in tl.static_range(FEATURES):
        values = tl.load(TABLE + index * FEATURES + feature, mask=mask, other=0).to(tl.float32)
        result = tl.sum(values * weight, axis=1)
        tl.store(OUTPUT + row * FEATURES + feature, result, mask=row < ROWS)


@triton.jit
def _scatter_gradient(INDICES, WEIGHTS, DOUTPUT, DTABLE, ROWS,
                      FEATURES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    corner = tl.arange(0, 8)
    mask = row[:, None] < ROWS
    index = tl.load(INDICES + row[:, None] * 8 + corner[None, :], mask=mask, other=0)
    weight = tl.load(WEIGHTS + row[:, None] * 8 + corner[None, :], mask=mask, other=0).to(tl.float32)
    for feature in tl.static_range(FEATURES):
        gradient = tl.load(DOUTPUT + row * FEATURES + feature, mask=row < ROWS, other=0).to(tl.float32)
        tl.atomic_add(DTABLE + index * FEATURES + feature,
                      gradient[:, None] * weight, mask=mask, sem="relaxed")


class _CachedInterpolation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, table: Tensor, indices: Tensor, weights: Tensor) -> Tensor:
        indices = indices.contiguous()
        weights = weights.contiguous()
        count, levels, _ = indices.shape
        features = table.shape[1]
        output = torch.empty((count, levels * features), dtype=torch.float32, device=table.device)
        rows = count * levels
        if rows:
            _interpolate[(triton.cdiv(rows, 128),)](
                table, indices, weights, output, rows, FEATURES=features, BLOCK=128, num_warps=4)
        ctx.save_for_backward(indices, weights)
        ctx.table_shape = table.shape
        ctx.table_dtype = table.dtype
        ctx.rows = rows
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient: Tensor):
        indices, weights = ctx.saved_tensors
        # Accumulate in float32 even if table parameters are explicitly half.
        # The default optimizer/table remain float32 under normal CUDA AMP.
        table_gradient = torch.zeros(ctx.table_shape, dtype=torch.float32, device=gradient.device)
        if ctx.rows:
            _scatter_gradient[(triton.cdiv(ctx.rows, 128),)](
                indices, weights, gradient.contiguous(), table_gradient, ctx.rows,
                FEATURES=ctx.table_shape[1], BLOCK=128, num_warps=4)
        return table_gradient.to(ctx.table_dtype), None, None


class TritonHashGrid(TorchHashGrid):
    """Torch grid with a fused cache path and identical state_dict keys."""

    def __init__(self, config: HashGridConfig = HashGridConfig()) -> None:
        super().__init__(config)

    def forward_cached(self, lookup: HashGridLookup) -> Tensor:
        if lookup.weights.requires_grad:
            return super().forward_cached(lookup)
        if lookup.signature != self.signature:
            raise ValueError("The lookup belongs to a different hash-grid configuration.")
        if not self.table.is_cuda or lookup.device != self.table.device or lookup.weights.device != self.table.device:
            raise ValueError("The Triton lookup and model must be on the same CUDA device.")
        if lookup.indices.shape != lookup.weights.shape or lookup.indices.shape[1:] != (self.config.n_levels, 8):
            raise ValueError("Malformed hash-grid lookup.")
        if not self.table.is_contiguous():
            raise ValueError("The Triton hash table must be contiguous.")
        return _CachedInterpolation.apply(self.table, lookup.indices, lookup.weights)


__all__ = ["TritonHashGrid"]
