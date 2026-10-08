"""Thin adapters over pinned official Transolver-family source files.

Input is [batch, mesh_nodes, 28]: normalized xyz followed by 25 BC/material
features. Output is [batch, mesh_nodes, 4]: uXYZ and source signed stress.
The adapter performs no target scaling and adds no sample ID or new layers.

Transolver uses official constructor defaults. Transolver++ uses the slide's
explicit 16 layers / width 256, with other official constructor defaults.
TS3 uses the official volume preset (16 / 256 / 8 heads / 64 slices / ratio 2).
The unavailable original server leaves some reference hyperparameters unknown;
``source_config`` records which settings are defaults, not verified slide facts.

TS3 chunks share global physical states at EVERY layer through the official
two-pass ``forward_chunks``. Evaluating independent spatial chunks for the
other models changes the model and is deliberately not offered here.
"""

from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from typing import Any

import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "third_party" / "transformer_sources.json"
ALIASES = {
    "transolver": "transolver", "transolver_original": "transolver", "original_transolver": "transolver",
    "transolver++": "transolver_pp", "transolverpp": "transolver_pp", "transolver_pp": "transolver_pp",
    "transolver_plus": "transolver_pp", "transolver-3": "ts3", "transolver3": "ts3", "transolver_3": "ts3", "ts3": "ts3",
}
DEFAULTS = {
    "transolver": dict(n_layers=5, n_hidden=256, n_head=8, slice_num=32, mlp_ratio=1, dropout=0.0),
    "transolver_pp": dict(n_layers=16, n_hidden=256, n_head=8, slice_num=32, mlp_ratio=1, dropout=0.0),
    "ts3": dict(n_layers=16, n_hidden=256, n_head=8, slice_num=64, mlp_ratio=2, dropout=0.0),
}
_LOADED: dict[str, ModuleType] = {}
_CREATED_PROCESS_GROUP = False
_PROCESS_GROUP_STORE: tempfile.TemporaryDirectory | None = None


def _canonical(method: str) -> str:
    try:
        return ALIASES[method.lower().strip()]
    except KeyError as error:
        raise ValueError("Unknown transformer; use transolver, transolver_pp/Transolver++, or ts3/Transolver-3.") from error


def _sources() -> dict:
    return json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))


def _load_source(method: str) -> ModuleType:
    """Execute unchanged upstream files with isolated import bindings.

    The source imports timm only for trunc_normal_. Its standard PyTorch
    equivalent is bound here, avoiding a torchvision dependency. Architecture,
    forward math, Gumbel randomness and learned parameter layouts are retained.
    """
    if method in _LOADED:
        return _LOADED[method]
    record = _sources()[method]
    root = ROOT / record["directory"]
    modules: dict[str, ModuleType] = {}

    def load_file(relative: str) -> ModuleType:
        if relative in modules:
            return modules[relative]
        path = root / relative
        source = path.read_bytes()
        expected = record["file_sha256"].get(relative)
        if expected is None or hashlib.sha256(source).hexdigest() != expected:
            raise RuntimeError(f"Official source differs from the pinned manifest: {path}")
        name = f"_geobc_official_{method}_{path.stem}"
        module = ModuleType(name)
        module.__file__ = str(path)
        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": vendor_import}
        modules[relative] = module
        sys.modules[name] = module
        exec(compile(source, str(path), "exec"), module.__dict__)
        return module

    def vendor_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ("timm.models.layers", "timm.layers") and "trunc_normal_" in fromlist:
            module = ModuleType(name)
            module.trunc_normal_ = nn.init.trunc_normal_
            return module
        if method == "transolver" and name in ("model.Embedding", "model.Physics_Attention"):
            return load_file("PDE-Solving-StandardBenchmark/" + name.replace(".", "/") + ".py")
        return builtins.__import__(name, globals, locals, fromlist, level)

    try:
        module = load_file(record["model_file"])
    except ModuleNotFoundError as error:
        if error.name == "einops":
            raise RuntimeError("The pinned official models require einops; install it in this Python environment.") from error
        raise
    _LOADED[method] = module
    return module


def _ensure_single_rank_group(device: torch.device) -> None:
    global _CREATED_PROCESS_GROUP, _PROCESS_GROUP_STORE
    if dist.is_initialized():
        if dist.get_world_size() != 1:
            raise RuntimeError("This DeepJEB adapter expects one full geometry per single-rank process group.")
        return
    _PROCESS_GROUP_STORE = tempfile.TemporaryDirectory(prefix="geobc_transolverpp_")
    store = Path(_PROCESS_GROUP_STORE.name) / "rendezvous"
    dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo",
                            init_method=store.as_uri(), rank=0, world_size=1)
    _CREATED_PROCESS_GROUP = True


def close_owned_process_group() -> None:
    """Release only a process group this adapter created; useful for smoke runs."""
    global _CREATED_PROCESS_GROUP, _PROCESS_GROUP_STORE
    if _CREATED_PROCESS_GROUP and dist.is_initialized():
        dist.destroy_process_group()
    _CREATED_PROCESS_GROUP = False
    if _PROCESS_GROUP_STORE is not None:
        _PROCESS_GROUP_STORE.cleanup()
        _PROCESS_GROUP_STORE = None


class OfficialTransformer(nn.Module):
    def __init__(self, method: str, backbone: nn.Module, input_dim: int, output_dim: int,
                 checkpoint_blocks: bool, chunk_size: int, source_config: dict):
        super().__init__()
        self.method = method
        self.backbone = backbone
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.checkpoint_blocks = checkpoint_blocks
        self.chunk_size = chunk_size
        self.source_config = source_config

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim or not x.is_floating_point():
            raise ValueError(f"Expected floating input [B,N,{self.input_dim}].")
        if not x.shape[1]:
            raise ValueError("The mesh must contain at least one node.")
        if self.method == "transolver_pp":
            _ensure_single_rank_group(x.device)
        if self.method == "ts3":
            # One global physical state per layer across ALL original nodes.
            chunks = list(x.split(self.chunk_size, dim=1))
            outputs = self.backbone(chunks, input_list=True,
                                    use_checkpoint=self.checkpoint_blocks and torch.is_grad_enabled())
            return torch.cat(outputs, dim=1)
        if self.training and self.checkpoint_blocks and torch.is_grad_enabled():
            # Identical upstream preprocessing and blocks; only activation
            # storage/recomputation changes. RNG state preserves ++ Gumbel.
            fx = self.backbone.preprocess(x)
            fx = fx + self.backbone.placeholder[None, None, :]
            for block in self.backbone.blocks:
                fx = checkpoint(block, fx, use_reentrant=False, preserve_rng_state=True)
            return fx
        if self.method == "transolver":
            return self.backbone(x[..., :3], x[..., 3:])
        # All BC/material inputs already reside in x; no separate condition
        # embedding is added beyond the requested 28-channel reference input.
        return self.backbone((x, x[..., :3], None))


def build_transformer(method: str, input_dim: int = 28, output_dim: int = 4,
                      device: str | torch.device = "cpu", checkpoint_blocks: bool = True,
                      chunk_size: int = 32768, model_overrides: dict[str, Any] | None = None) -> OfficialTransformer:
    """Build a pinned official backbone; forward([B,N,input_dim])->[B,N,out].

    The TS3 chunk size only controls memory, retaining whole-mesh context.
    Original/++ require the whole geometry in one forward. Their checkpointed
    training path keeps the official network and stores fewer activations.
    ++ already checkpoints attention/MLP internally and remains stochastic
    in eval mode because upstream Gumbel sampling has no eval-mode guard.
    """
    method = _canonical(method)
    if input_dim < 3 or output_dim < 1 or chunk_size < 1:
        raise ValueError("Invalid input/output dimensions or TS3 chunk size.")
    module = _load_source(method)
    kwargs = {**DEFAULTS[method], "space_dim": 3, "fun_dim": input_dim - 3,
              "out_dim": output_dim, "act": "gelu", "unified_pos": False, "ref": 8}
    overrides = model_overrides or {}
    allowed = {"n_layers", "n_hidden", "n_head", "slice_num", "mlp_ratio", "dropout", "act"}
    if set(overrides).difference(allowed):
        raise ValueError(f"Unsupported architecture override: {sorted(set(overrides).difference(allowed))}")
    kwargs.update(overrides)
    if kwargs["n_hidden"] % kwargs["n_head"]:
        raise ValueError("Hidden width must be divisible by attention heads.")
    record = _sources()[method]
    source_config = {
        "method": method, "repository": record["repository"], "commit": record["commit"],
        "model_file": record["model_file"], "file_sha256": record["file_sha256"],
        "license": record["license"], "constructor_kwargs": kwargs,
        "preset_source": record["preset_source"], "slide_explicit": record["slide_explicit"],
        "reference_gaps": record["reference_gaps"], "architecture_overrides": overrides,
        "input": "xyz[3] + BC/material[25]; no sample ID", "output": "uXYZ + source signed stress",
        "timm_initializer_binding": "torch.nn.init.trunc_normal_ (same standard truncated-normal initializer)",
        "checkpoint_blocks": checkpoint_blocks,
        "checkpoint_strategy": "official global chunk stats and chunk deslice/MLP" if method == "ts3" else "whole upstream blocks, RNG preserved",
        "context": "every supplied mesh node contributes at every layer; no independent chunk evaluation",
        "chunk_size": chunk_size if method == "ts3" else None,
        "eval_stochastic": method == "transolver_pp",
        "gumbel_note": "Upstream ++ draws Gumbel noise during both train and eval; no guard or replacement added." if method == "transolver_pp" else None,
        "distributed": "world_size=1 process group for unchanged official all_reduce calls" if method == "transolver_pp" else None,
    }
    model = OfficialTransformer(method, module.Model(**kwargs), input_dim, output_dim,
                                checkpoint_blocks, chunk_size, source_config).to(device)
    model.source_config["parameter_count"] = sum(p.numel() for p in model.parameters())
    return model


__all__ = ["build_transformer", "OfficialTransformer", "close_owned_process_group"]
