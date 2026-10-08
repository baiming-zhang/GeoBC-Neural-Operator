#!/usr/bin/env python3
"""Sequential synthetic one-update profiles of unchanged official backbones.

No DeepJEB labels, trained checkpoints or substantive fitting are used.
Example: python tools/transformer_smoke.py --device cuda --nodes 158221
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from geobc.transformers import build_transformer, close_owned_process_group


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile(method: str, nodes: int, device: torch.device, chunk_size: int, checkpoint_blocks: bool) -> dict:
    torch.manual_seed(2026)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2026)
        torch.cuda.empty_cache()
    model = build_transformer(method, input_dim=28, output_dim=4, device=device,
                              checkpoint_blocks=checkpoint_blocks, chunk_size=chunk_size)
    x = torch.randn((1, nodes, 28), device=device) * 0.1
    x[..., :3].uniform_(0, 1)
    target = torch.randn((1, nodes, 4), device=device) * 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    output = model(x)
    assert output.shape == (1, nodes, 4) and torch.isfinite(output).all()
    loss = (output - target).square().mean()
    synchronize(device)
    forward_seconds = time.monotonic() - started
    backward_started = time.monotonic()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    gradient_norm = float(torch.stack([gradient.square().sum() for gradient in gradients]).sum().sqrt())
    assert gradient_norm > 0
    optimizer.step()
    synchronize(device)
    backward_seconds = time.monotonic() - backward_started
    record = {"method": model.method, "synthetic_nodes": nodes, "input_shape": list(x.shape),
              "profiling_matmul_precision": torch.get_float32_matmul_precision(),
              "output_shape": list(output.shape), "forward_seconds": forward_seconds,
              "backward_and_adam_seconds": backward_seconds, "one_update_seconds": time.monotonic() - started,
              "gradient_norm": gradient_norm, "synthetic_loss": float(loss.detach()), "source_config": model.source_config,
              "meaning": "one synthetic forward/backward/Adam update; not a trained result or steady-state timing"}
    if device.type == "cuda":
        record["peak_allocated_GiB"] = torch.cuda.max_memory_allocated(device) / 2**30
        record["peak_reserved_GiB"] = torch.cuda.max_memory_reserved(device) / 2**30
    # Tiny synthetic probes verify the context adapter and upstream eval behavior,
    # without repeatedly running a full mesh or claiming fitting accuracy.
    model.eval()
    probe = x[:, :min(nodes, 512)]
    with torch.no_grad():
        first = model(probe)
        second = model(probe)
        record["eval_probe_nodes"] = probe.shape[1]
        record["eval_repeat_max_abs_difference"] = float((first - second).abs().max())
        record["eval_repeat_exactly_equal"] = bool(torch.equal(first, second))
        if model.method == "ts3":
            # Matrix associativity is sensitive to TF32 rounding. Check the
            # upstream global-chunk math in strict FP32, independent of the
            # explicitly recorded profiling precision setting.
            previous_precision = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("highest")
            previous_chunk_size = model.chunk_size
            model.chunk_size = max(1, probe.shape[1] // 4)
            global_chunked = model(probe)
            unchunked = model.backbone(probe, input_list=False, use_checkpoint=False)
            torch.testing.assert_close(global_chunked, unchunked, rtol=2e-4, atol=2e-5)
            record["global_chunk_vs_full_probe_max_abs_difference"] = float((global_chunked - unchunked).abs().max())
            record["context_consistency_probe_matmul_precision"] = "highest"
            model.chunk_size = previous_chunk_size
            torch.set_float32_matmul_precision(previous_precision)
    close_owned_process_group()
    del output, loss, gradients, optimizer, model, x, target, first, second, probe
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["transolver", "transolver_pp", "ts3"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--no-checkpoint-blocks", action="store_true")
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    parser.add_argument("--output", type=Path, default=ROOT / "references" / "transformer_smoke.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision(args.matmul_precision)
    device = torch.device(args.device)
    if args.nodes < 1:
        parser.error("--nodes must be positive")
    metadata = {"torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "device": str(device), "dtype": "float32", "matmul_precision": args.matmul_precision,
                "checkpoint_blocks": not args.no_checkpoint_blocks,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "profiles": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        try:
            record = profile(method, args.nodes, device, args.chunk_size, not args.no_checkpoint_blocks)
            record["passed"] = True
        except torch.cuda.OutOfMemoryError as error:
            close_owned_process_group()
            gc.collect()
            torch.cuda.empty_cache()
            record = {"method": method, "synthetic_nodes": args.nodes, "passed": False, "error": str(error), "error_type": "CUDA out of memory"}
        metadata["profiles"].append(record)
        args.output.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({key: record[key] for key in ("method", "passed", "one_update_seconds", "peak_allocated_GiB") if key in record}), flush=True)
    print(f"Profile records: {args.output}", flush=True)


if __name__ == "__main__":
    main()
