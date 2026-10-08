"""Small correctness checks and optional one-step synthetic FNO CUDA profiles.

This is not an overfitting experiment. --benchmark 64:4 64:1 192:1 profiles
joint/stress heads with one synthetic Adam update each and reports actual
wall time and allocator peak. Source-node/grid errors are in prepared NPZ
metadata, not inferred from this synthetic profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from geobc.fno import FNO3d, SpectralConv3d, FNO_SOURCES, query_grid
from tools.prepare_grids import (TETRA10_EDGES, build_geometry_grid, interpolate_nodal,
                                masked_trilinear_cache, quadratic_shape)


def correctness(device: torch.device) -> dict:
    torch.manual_seed(2026)
    corners = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    midsides = 0.5 * (corners[TETRA10_EDGES[:, 0]] + corners[TETRA10_EDGES[:, 1]])
    midsides[0, 2] += 0.05
    xyz = np.concatenate((corners, midsides))
    reference_nodes = np.concatenate((np.eye(4), 0.5 * (np.eye(4)[TETRA10_EDGES[:, 0]] + np.eye(4)[TETRA10_EDGES[:, 1]])))
    basis = quadratic_shape(torch.as_tensor(reference_nodes, device=device))
    kronecker_error = float((basis - torch.eye(10, device=device)).abs().max())
    assert kronecker_error < 1e-12
    cache, geometry = build_geometry_grid(xyz, np.arange(10)[None], 16, device, pair_batch_size=2048)
    bary = cache["barycentric"].astype(np.float64)
    nodal_polynomial = (reference_nodes[:, 0] ** 2 + 2 * reference_nodes[:, 1] * reference_nodes[:, 2] - reference_nodes[:, 3])[:, None]
    interpolated = interpolate_nodal(nodal_polynomial, np.arange(10)[None], cache, device)[:, 0]
    expected = bary[:, 0] ** 2 + 2 * bary[:, 1] * bary[:, 2] - bary[:, 3]
    quadratic_error = float(np.max(np.abs(interpolated - expected)))
    assert quadratic_error < 2e-6
    assert geometry["maximum_reconstructed_coordinate_error_normalized"] < 2e-6
    assert np.max(np.abs(cache["shape_weights"].sum(1) - 1)) < 1e-6
    coordinate = np.linspace(0, 1, 4, dtype=np.float32)
    gx, gy, gz = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    linear = (2 * gx + 3 * gy - 4 * gz)[None]
    queries = np.asarray([[0, 0, 0], [1, 1, 1], [0.13, 0.27, 0.81], [1, 0.2, 0]], np.float32)
    indices, weights, coverage = masked_trilinear_cache(queries, np.ones((4, 4, 4), bool))
    grid = torch.tensor(linear, device=device, requires_grad=True)
    queried = query_grid(grid, torch.as_tensor(indices, device=device), torch.as_tensor(weights, device=device), chunk_size=3)
    query_truth = 2 * queries[:, 0] + 3 * queries[:, 1] - 4 * queries[:, 2]
    query_error = float(np.max(np.abs(queried.detach().cpu().numpy()[:, 0] - query_truth)))
    assert coverage.all() and query_error < 1e-6 and len(queried) == len(queries)
    queried.sum().backward()
    assert abs(float(grid.grad.sum()) - len(queries)) < 1e-5
    empty_indices, empty_weights, empty_coverage = masked_trilinear_cache(queries, np.zeros((4, 4, 4), bool))
    empty_result = query_grid(grid.detach(), torch.as_tensor(empty_indices, device=device), torch.as_tensor(empty_weights, device=device))
    assert not empty_coverage.any() and torch.count_nonzero(empty_result) == 0
    convolution = SpectralConv3d(2, 2, 2).to(device)
    x = torch.randn(2, 2, 8, 8, 8, device=device)
    actual = convolution(x)
    coefficients = torch.fft.rfftn(x, dim=(-3, -2, -1))
    full = torch.zeros_like(coefficients)
    slices = [(slice(0, 2), slice(0, 2)), (slice(-2, None), slice(0, 2)),
              (slice(0, 2), slice(-2, None)), (slice(-2, None), slice(-2, None))]
    for weight, (sx, sy) in zip(convolution.weights, slices):
        full[:, :, sx, sy, :2] = torch.einsum("bixyz,ioxyz->boxyz", coefficients[:, :, sx, sy, :2], torch.view_as_complex(weight))
    full_result = torch.fft.irfftn(full, s=(8, 8, 8), dim=(-3, -2, -1))
    spectrum_error = float((actual - full_result).detach().abs().max())
    assert spectrum_error < 1e-6
    model = FNO3d(18, 4, width=4, modes=3, layers=4, head_width=8).to(device)
    output = model(torch.randn(1, 18, 16, 16, 16, device=device))
    assert output.shape == (1, 4, 16, 16, 16)
    output.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    return {"status": "passed", "device": str(device), "kronecker_shape_error": kronecker_error,
            "curved_isoparametric_geometry": geometry, "quadratic_reference_field_max_error": quadratic_error,
            "masked_trilinear_linear_field_max_error": query_error, "query_tail_and_zero_support": "passed",
            "truncated_vs_full_spectrum_max_error": spectrum_error, "four_block_forward_backward": "passed"}


def profile(resolution: int, out_channels: int, device: torch.device, checkpoint_blocks: bool) -> dict:
    if device.type != "cuda":
        raise ValueError("Full-grid profile requires CUDA; CPU checks remain available without --benchmark")
    modes = 12 if resolution == 64 else 16
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(2026)
    model = FNO3d(18, out_channels, width=16, modes=modes, layers=4,
                  checkpoint_blocks=checkpoint_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, fused=True)
    inputs = torch.randn(1, 18, resolution, resolution, resolution, device=device)
    torch.cuda.synchronize(device)
    began = time.perf_counter()
    prediction = model(inputs)
    loss = prediction.square().mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - began
    result = {"resolution": resolution, "out_channels": out_channels, "modes": modes,
              "configuration": model.config.record(), "real_parameter_count": sum(p.numel() for p in model.parameters()),
              "synthetic_optimizer_steps": 1, "forward_backward_fused_Adam_seconds": elapsed,
              "peak_allocated_GiB": torch.cuda.max_memory_allocated(device) / (1 << 30),
              "peak_reserved_GiB": torch.cuda.max_memory_reserved(device) / (1 << 30),
              "input_and_model_dtype": "float32", "includes_optimizer_state": True,
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "timing_note": "One cold synthetic step, including cuFFT/pointwise initial plans; not steady-state training throughput or an overfit result."}
    del optimizer, model, inputs, prediction, loss
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--benchmark", nargs="*", default=[], help="Optional resolution:out_channels, e.g. 64:4 64:1 192:1")
    parser.add_argument("--checkpoint-blocks", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "notes/fno_validation.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    report = {"torch": torch.__version__, "numpy": np.__version__, "sources": FNO_SOURCES,
              "correctness": correctness(device), "profiles": []}
    if device.type == "cuda":
        report["GPU"] = torch.cuda.get_device_name(device)
        report["GPU_total_GiB"] = torch.cuda.get_device_properties(device).total_memory / (1 << 30)
    for configuration in args.benchmark:
        resolution, channels = map(int, configuration.split(":"))
        result = profile(resolution, channels, device, args.checkpoint_blocks)
        report["profiles"].append(result)
        print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
