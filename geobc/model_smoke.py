"""Lightweight mathematical checks; no dataset loading or optimization.

Run ``python -m geobc.model_smoke --device cpu`` (or ``--device cuda``).
"""

from __future__ import annotations

import argparse
import math

import torch

from .models import HashGridConfig, ModelConfig, TorchHashGrid, build_model


def run(device: str = "cpu", check_triton: bool = False) -> dict:
    torch.manual_seed(7)
    target_device = torch.device(device)
    # Keep this test small; include both dense and hashed levels.
    config = HashGridConfig(n_levels=3, n_features_per_level=2,
                            log2_hashmap_size=5, base_resolution=2, per_level_scale=2)
    encoding = TorchHashGrid(config).to(target_device)
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
                        [0.2, 0.35, 0.7], [0.75, 0.125, 0.6]], device=target_device)
    cache = encoding.prepare_lookup(xyz, chunk_size=2)
    torch.testing.assert_close(cache.weights.sum(dim=-1), torch.ones((4, 3), device=target_device))
    torch.testing.assert_close(encoding(xyz), encoding.forward_cached(cache), rtol=0, atol=0)
    rows = torch.tensor([3, 1], device=target_device)
    torch.testing.assert_close(encoding(xyz[rows]), encoding.forward_cached(cache.select(rows)), rtol=0, atol=0)

    # Compare vectorized encoding with an independent scalar eight-corner
    # calculation, including uint32 hashing and dense boundary wraparound.
    with torch.no_grad():
        encoding.table.copy_(torch.arange(encoding.table.numel(), device=target_device).reshape_as(encoding.table) / 100)
    result = encoding(xyz).detach().cpu()
    table = encoding.table.detach().cpu()
    for row, point in enumerate(xyz.cpu().tolist()):
        offset = 0
        for level in range(config.n_levels):
            scale = config.base_resolution * config.per_level_scale**level - 1
            resolution = math.ceil(scale) + 1
            size = min(((resolution**3 + 7) // 8) * 8, 1 << config.log2_hashmap_size)
            position = [v * scale + 0.5 for v in point]
            lower = [math.floor(v) for v in position]
            fraction = [v - a for v, a in zip(position, lower)]
            expected = torch.zeros(2)
            for corner in range(8):
                vertex = [lower[axis] + ((corner >> axis) & 1) for axis in range(3)]
                weight = math.prod(fraction[axis] if (corner >> axis) & 1 else 1 - fraction[axis]
                                   for axis in range(3))
                if size >= resolution**3:
                    index = vertex[0] + resolution * vertex[1] + resolution**2 * vertex[2]
                else:
                    index = (vertex[0] ^ vertex[1] * 2654435761 ^ vertex[2] * 805459861) & 0xFFFFFFFF
                expected += table[offset + index % size] * weight
            torch.testing.assert_close(result[row, level * 2:(level + 1) * 2], expected)
            offset += size

    # Input and learned table both retain gradients in the uncached path.
    differentiable_xyz = xyz.clone().requires_grad_(True)
    encoding(differentiable_xyz).square().sum().backward()
    assert differentiable_xyz.grad is not None and torch.isfinite(differentiable_xyz.grad).all()
    assert encoding.table.grad is not None and encoding.table.grad.abs().sum() > 0

    if check_triton:
        from .triton_hashgrid import TritonHashGrid
        fused = TritonHashGrid(config).to(target_device)
        fused.load_state_dict(encoding.state_dict())
        encoding.table.grad = None
        reference = encoding.forward_cached(cache)
        reference.square().sum().backward()
        candidate = fused.forward_cached(cache)
        candidate.square().sum().backward()
        torch.testing.assert_close(candidate, reference)
        torch.testing.assert_close(fused.table.grad, encoding.table.grad, rtol=2e-5, atol=2e-5)

    for encoder, mode in [("hashgrid", "xyz"), ("hashgrid", "both"),
                          ("mlp", "xyz"), ("mlp", "features"), ("mlp", "both")]:
        model = build_model("torch", ModelConfig(encoder=encoder, input_mode=mode, hash_grid=config), target_device)
        features = torch.randn(4, 25, device=target_device)
        prediction = model(xyz, features)
        assert prediction.shape == (4,) and torch.isfinite(prediction).all()
        lookup = model.prepare_lookup(xyz)
        if lookup is not None:
            torch.testing.assert_close(prediction, model.forward_cached(lookup, features), rtol=0, atol=0)
        # Force a negative scalar: the output layer must be unrestricted.
        with torch.no_grad():
            model.mlp[-1].weight.zero_()
            model.mlp[-1].bias.fill_(-2.0)
        torch.testing.assert_close(model(xyz, features), xyz.new_full((4,), -2))
    return {"device": str(target_device), "checks": "dense/hash interpolation, cache, gradients, modes, signed output",
            "triton_parity": check_triton, "passed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--check-triton", action="store_true")
    args = parser.parse_args()
    print(run(args.device, args.check_triton))
