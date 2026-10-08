"""Overfit one or several prepared DeepJEB cases with a shared stress model.

Each input NPZ contains xyz (N, 3) in [0, 1], features (N, 25), stress
(N,) in physical units, and a scalar sample_key. Every evaluation visits every
node. Run ``python -m geobc.train --help`` for the command line interface.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from .models import ModelConfig, build_model


SCALER_DEFINITION = {
    "fit_data": "all labels of the selected training cases; no validation/test labels",
    "scale": "90th percentile of abs(stress), or RMS if that percentile is zero; floor 1e-12",
    "center": "pooled mean of asinh(stress / scale)",
    "spread": "pooled population standard deviation of asinh(stress / scale); floor 1e-8",
    "encode": "(asinh(stress / scale) - center) / spread",
    "decode": "scale * sinh(center + spread * prediction)",
}


@dataclass(frozen=True)
class StressScaler:
    scale: float
    center: float
    spread: float

    @classmethod
    def fit(cls, labels: list[np.ndarray]) -> "StressScaler":
        values = np.concatenate([np.asarray(y, dtype=np.float64) for y in labels])
        scale = float(np.quantile(np.abs(values), 0.90))
        if scale <= 1e-12:
            scale = float(np.sqrt(np.mean(np.square(values))))
        scale = max(scale, 1e-12)
        transformed = np.arcsinh(values / scale)
        return cls(scale, float(transformed.mean()), max(float(transformed.std()), 1e-8))

    def encode(self, values: np.ndarray) -> np.ndarray:
        return ((np.arcsinh(np.asarray(values, dtype=np.float64) / self.scale) - self.center)
                / self.spread).astype(np.float32)

    def decode(self, values: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            result = self.scale * np.sinh(self.center + self.spread * np.asarray(values, dtype=np.float64))
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite decoded stress: reduce learning rate or inspect the model output.")
        return result

    def record(self) -> dict[str, Any]:
        return {**asdict(self), "definition": SCALER_DEFINITION}


@dataclass
class TrainingCase:
    key: str
    source: Path
    xyz: torch.Tensor
    features: torch.Tensor
    target: torch.Tensor
    physical_stress: np.ndarray
    lookup: Any = None
    hard_indices: torch.Tensor | None = None

    @property
    def count(self) -> int:
        return len(self.physical_stress)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _key_from_archive(archive: Any, source: Path) -> str:
    if "sample_key" not in archive:
        raise ValueError(f"{source}: missing sample_key")
    value = np.asarray(archive["sample_key"])
    if value.size != 1:
        raise ValueError(f"{source}: sample_key must be scalar")
    key = value.reshape(-1)[0].item()
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    if not str(key):
        raise ValueError(f"{source}: empty sample_key")
    return str(key)


def find_case_files(data: list[str], case_keys: list[str] | None, num_cases: int) -> list[Path]:
    files: list[Path] = []
    for entry in data:
        path = Path(entry).expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(path.rglob("*.npz")))
        elif path.is_file() and path.suffix.lower() == ".npz":
            files.append(path)
        else:
            raise FileNotFoundError(f"Expected a prepared NPZ or directory: {path}")
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No prepared NPZ cases found")
    if case_keys:
        by_key: dict[str, Path] = {}
        requested = set(case_keys)
        if len(requested) != len(case_keys):
            raise ValueError("--case-keys contains duplicates")
        for path in files:
            with np.load(path, allow_pickle=False) as archive:
                key = _key_from_archive(archive, path)
            if key in requested:
                if key in by_key:
                    raise ValueError(f"Duplicate sample_key {key!r} in the input files")
                by_key[key] = path
        missing = requested.difference(by_key)
        if missing:
            raise ValueError(f"Requested sample keys not found: {sorted(missing)}")
        return [by_key[key] for key in case_keys]
    if len(files) < num_cases:
        raise ValueError(f"Requested {num_cases} cases, but found only {len(files)}")
    return files[:num_cases]


def load_cases(files: list[Path], device: torch.device) -> tuple[list[TrainingCase], StressScaler]:
    loaded: list[tuple[str, Path, np.ndarray, np.ndarray, np.ndarray]] = []
    keys: set[str] = set()
    for source in files:
        with np.load(source, allow_pickle=False) as archive:
            if "label_alignment" not in archive.files or str(archive["label_alignment"].item()) != "official CSV GRID coordinates -> H5 vertices":
                raise ValueError(f"{source}: verified CSV-to-geometry label alignment required; old H5-only data are invalid")
            required = {"xyz", "features", "stress", "sample_key"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{source}: missing {sorted(missing)}")
            key = _key_from_archive(archive, source)
            xyz = np.asarray(archive["xyz"], dtype=np.float32)
            features = np.asarray(archive["features"], dtype=np.float32)
            stress = np.asarray(archive["stress"], dtype=np.float64)
        if key in keys:
            raise ValueError(f"Duplicate sample_key: {key}")
        keys.add(key)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
            raise ValueError(f"{source}: xyz must have nonempty shape [N, 3]")
        if features.shape != (len(xyz), 25) or stress.shape != (len(xyz),):
            raise ValueError(f"{source}: expected features [N, 25] and stress [N]")
        if not all(np.isfinite(array).all() for array in (xyz, features, stress)):
            raise ValueError(f"{source}: nonfinite inputs or stress labels")
        if xyz.min() < -1e-6 or xyz.max() > 1 + 1e-6:
            raise ValueError(f"{source}: xyz is outside the required normalized [0, 1] range")
        loaded.append((key, source, np.clip(xyz, 0, 1), features, stress))
    scaler = StressScaler.fit([entry[4] for entry in loaded])
    cases = [TrainingCase(key, source, torch.as_tensor(xyz, device=device),
                          torch.as_tensor(features, device=device),
                          torch.as_tensor(scaler.encode(stress), device=device), stress)
             for key, source, xyz, features, stress in loaded]
    return cases, scaler


def stress_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    """Physical-unit metrics; hotspot means exactly ceil(1% N) largest |truth| nodes."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = prediction - truth
    hotspot_count = max(1, math.ceil(len(truth) * 0.01))
    hotspots = np.argpartition(np.abs(truth), len(truth) - hotspot_count)[-hotspot_count:]
    denominator = max(float(np.linalg.norm(truth)), 1e-12)
    hotspot_denominator = max(float(np.linalg.norm(truth[hotspots])), 1e-12)
    return {
        "nodes": len(truth),
        "relative_l2": float(np.linalg.norm(residual) / denominator),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs_error": float(np.max(np.abs(residual))),
        "hotspot_nodes": hotspot_count,
        "hotspot_relative_l2": float(np.linalg.norm(residual[hotspots]) / hotspot_denominator),
        "hotspot_mae": float(np.mean(np.abs(residual[hotspots]))),
        "true_peak": float(np.max(truth)),
        "predicted_peak": float(np.max(prediction)),
        "peak_abs_error": float(abs(np.max(prediction) - np.max(truth))),
    }


def _forward(model: torch.nn.Module, case: TrainingCase, indices: Any) -> torch.Tensor:
    if case.lookup is not None:
        output = model.forward_cached(case.lookup.select(indices), case.features[indices])
    else:
        output = model(case.xyz[indices], case.features[indices])
    return output.reshape(-1)


def _refresh_hard_indices(cases: list[TrainingCase], predictions: dict[str, np.ndarray],
                          hard_node_count: int) -> None:
    """Refresh fixed-size GPU sets from an already completed all-node physical evaluation."""
    for case in cases:
        residual = np.abs(predictions[case.key] - case.physical_stress)
        count = min(hard_node_count, case.count)
        selected = np.argpartition(residual, case.count - count)[-count:]
        selected = selected[np.argsort(residual[selected])[::-1]].copy()
        case.hard_indices = torch.as_tensor(selected, dtype=torch.long, device=case.xyz.device)


def _sample_case_indices(case: TrainingCase, count: int, hard_node_fraction: float = 0.0) -> torch.Tensor:
    # Preserve the original uniform RNG path exactly when refinement is disabled.
    if hard_node_fraction == 0.0:
        return torch.randint(case.count, (count,), device=case.xyz.device)
    hard_count = int(count * hard_node_fraction)
    if hard_count == 0:
        return torch.randint(case.count, (count,), device=case.xyz.device)
    if case.hard_indices is None or not len(case.hard_indices):
        raise RuntimeError("Hard-node sampling requires a preceding complete physical evaluation")
    uniform = torch.randint(case.count, (count - hard_count,), device=case.xyz.device)
    hard_rows = torch.randint(len(case.hard_indices), (hard_count,), device=case.xyz.device)
    return torch.cat((uniform, case.hard_indices[hard_rows]))


@torch.no_grad()
def evaluate(model: torch.nn.Module, cases: list[TrainingCase], scaler: StressScaler,
             batch_size: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    predictions: dict[str, np.ndarray] = {}
    per_case: dict[str, dict[str, float | int]] = {}
    for case in cases:
        encoded = np.empty(case.count, dtype=np.float32)
        for start in range(0, case.count, batch_size):
            stop = min(start + batch_size, case.count)
            encoded[start:stop] = _forward(model, case, slice(start, stop)).float().cpu().numpy()
        prediction = scaler.decode(encoded)
        predictions[case.key] = prediction
        per_case[case.key] = stress_metrics(case.physical_stress, prediction)
    pooled = stress_metrics(np.concatenate([case.physical_stress for case in cases]),
                            np.concatenate([predictions[case.key] for case in cases]))
    mean_metrics = {name: float(np.mean([entry[name] for entry in per_case.values()]))
                    for name in pooled if name not in {"nodes", "hotspot_nodes"}}
    worst_key = max(per_case, key=lambda key: float(per_case[key]["relative_l2"]))
    return {"pooled": pooled, "case_mean": mean_metrics,
            "worst_case_key": worst_key, "worst_case": per_case[worst_key],
            "per_case": per_case}, predictions


def save_predictions(folder: Path, cases: list[TrainingCase], predictions: dict[str, np.ndarray],
                     step: int, scaler: StressScaler) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {}
    # Numerical filenames avoid path injection and name collisions in external case keys.
    for number, case in enumerate(cases):
        filename = f"case_{number:03d}.npz"
        temporary = folder / (filename + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, sample_key=np.asarray(case.key),
                                xyz=case.xyz.detach().cpu().numpy(),
                                stress_true=case.physical_stress,
                                stress_pred=predictions[case.key], step=np.asarray(step),
                                scaler_scale=np.asarray(scaler.scale),
                                scaler_center=np.asarray(scaler.center),
                                scaler_spread=np.asarray(scaler.spread))
        temporary.replace(folder / filename)
        index[case.key] = {"file": filename, "source": str(case.source), "nodes": case.count}
    write_json(folder / "index.json", {"step": step, "cases": index})


def _checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                scaler: StressScaler, config: dict[str, Any], step: int,
                best_error: float | None, best_step: int | None, metrics: dict[str, Any],
                gradient_scaler: Any = None) -> None:
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scaler": asdict(scaler), "scaler_definition": SCALER_DEFINITION,
               "config": config, "step": step, "best_error": best_error,
               "best_step": best_step, "metrics": metrics,
               "amp_scaler": gradient_scaler.state_dict() if gradient_scaler is not None else None,
               "random_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
               "torch_rng_state": torch.get_rng_state(),
               "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True, help="Prepared case NPZ files or directories")
    parser.add_argument("--num-cases", type=int, default=1, help="First N cases in sorted path order; use 1 then 5")
    parser.add_argument("--case-keys", nargs="+", help="Explicit case keys; overrides --num-cases selection")
    parser.add_argument("--output", type=Path, help="Run directory; defaults to runs/<timestamp>")
    parser.add_argument("--encoder", choices=("hashgrid", "mlp"), default="hashgrid")
    parser.add_argument("--backend", choices=("auto", "torch", "tcnn", "triton"), default="auto")
    parser.add_argument("--input-mode", choices=("auto", "xyz", "features", "both"), default="auto",
                        help="auto uses xyz for one case, both for several cases")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--steps", type=int, help="Balanced random updates; default 1000")
    duration.add_argument("--epochs", type=int, help="True full epochs: each case node exactly once per epoch")
    parser.add_argument("--nodes-per-case", type=int, default=16384)
    parser.add_argument("--hard-node-fraction", type=float, default=0.0,
                        help="Additional biased tail refinement: fraction from worst physical-residual nodes (0..0.5)")
    parser.add_argument("--hard-node-count", type=int, default=256,
                        help="Per-case top-K absolute decoded physical residuals, refreshed after each full evaluation")
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--target-error", type=float, default=0.01,
                        help="Stop once worst case decoded relative L2 reaches this value; 0 disables")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--cache-lookups", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--cache-budget-gb", type=float, default=0.0,
                        help="Auto-cache budget; 0 uses 35%% of free CUDA memory or 1 GiB on CPU")
    parser.add_argument("--amp", action="store_true", help="CUDA float16 autocast; disabled by default")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path, help="Resume a checkpoint produced by this trainer")
    parser.add_argument("--override-resume-lr", action="store_true",
                        help="When resuming, explicitly replace optimizer learning rates with --learning-rate")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("num_cases", "nodes_per_case", "eval_batch_size", "eval_every", "hidden_dim"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.hidden_layers < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Hidden layers and learning rate must be positive; weight decay must be nonnegative")
    if (args.steps is not None and args.steps < 1) or (args.epochs is not None and args.epochs < 1):
        raise ValueError("--steps and --epochs must be positive")
    if args.target_error < 0:
        raise ValueError("--target-error must be nonnegative")
    if args.cache_budget_gb < 0:
        raise ValueError("--cache-budget-gb must be nonnegative")
    if not 0.0 <= args.hard_node_fraction <= 0.5:
        raise ValueError("--hard-node-fraction must be between 0 and 0.5")
    if args.hard_node_count < 1:
        raise ValueError("--hard-node-count must be positive")
    if args.epochs and args.hard_node_fraction > 0:
        raise ValueError("Hard-node refinement supports random updates only, not --epochs")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    started_monotonic = time.monotonic()
    started_at = utc_now()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this Python environment")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires a CUDA device")
    output = (args.output or (args.resume.parent if args.resume else
                             Path("runs") / ("deepjeb_" + datetime.now().strftime("%Y%m%d_%H%M%S")))).resolve()
    if output.exists() and any(output.iterdir()) and args.resume is None:
        raise FileExistsError(f"Run directory is nonempty; choose a new --output or use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {"started_at_utc": started_at, "output": str(output), "device": str(device),
                              "phase": "loading", "step": 0, "total_steps": None}

    def update_status(phase: str, **extra: Any) -> None:
        elapsed = time.monotonic() - started_monotonic
        status.update(extra)
        status.update(phase=phase, updated_at_utc=utc_now(), elapsed_seconds=elapsed)
        completed = max(0, int(status.get("step", 0)) - int(status.get("resume_step", 0)))
        total = status.get("total_steps")
        remaining = max(0, total - int(status.get("step", 0))) if total is not None else None
        eta = elapsed / completed * remaining if completed > 0 and remaining is not None else None
        if phase in {"completed", "failed"}:
            eta = 0.0 if phase == "completed" else None
        status["eta_seconds"] = eta
        status["estimated_completion_utc"] = (datetime.now(timezone.utc) + timedelta(seconds=eta)).isoformat() if eta is not None else None
        write_json(output / "status.json", status)

    update_status("loading")
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        files = find_case_files(args.data, args.case_keys, args.num_cases)
        cases, scaler = load_cases(files, device)
        input_mode = args.input_mode if args.input_mode != "auto" else ("xyz" if len(cases) == 1 else "both")
        model_config = ModelConfig(encoder=args.encoder, input_mode=input_mode,
                                   hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers,
                                   condition_dim=25)
        model = build_model(backend=args.backend, config=model_config, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                                     fused=(device.type == "cuda"))
        batches_per_epoch = max(math.ceil(case.count / args.nodes_per_case) for case in cases)
        total_steps = args.epochs * batches_per_epoch if args.epochs else (args.steps or 1000)
        config: dict[str, Any] = {
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "model": asdict(model_config), "dataset": "DeepJEB", "device": str(device),
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "optimizer_fused": device.type == "cuda",
            "scaler": scaler.record(), "total_steps": total_steps,
            "sampling": ("each node once per full epoch" if args.epochs else
                         ("case-balanced biased tail refinement: uniform plus refreshed top-K physical-residual sampling with replacement"
                          if args.hard_node_fraction > 0 else
                          "equal nodes per case per update, sampled with replacement")),
            "loss": "mean across participating cases of MSE in encoded stress",
            "checkpoint_selection": "minimum worst-case decoded physical relative L2",
            "relative_l2_denominator_floor": 1e-12,
            "hotspot_definition": "exactly ceil(0.01*N) nodes with highest absolute true physical stress; pooled uses all nodes",
            "evaluation": "all nodes of all selected training cases; this is an overfit experiment, not held-out generalization",
            "cases": [{"sample_key": case.key, "source": str(case.source), "nodes": case.count} for case in cases],
            "started_at_utc": started_at,
        }
        if args.hard_node_fraction > 0:
            config["hard_node_sampling"] = {
                "scope": "Additional post-reproduction improvement, outside the original reference training budget",
                "fraction": args.hard_node_fraction,
                "top_k_per_case": args.hard_node_count,
                "hard_nodes_per_case_update": int(args.nodes_per_case * args.hard_node_fraction),
                "selection": "top min(K,N) absolute decoded physical residuals per case; both draws use replacement",
                "refresh": "after each complete all-node physical evaluation, before subsequent random updates",
                "refresh_steps": [],
                "evaluation_and_best_criterion": "unchanged unweighted all-node physical metrics",
            }
        start_step = 0
        best_error: float | None = None
        best_step: int | None = None
        checkpoint: dict[str, Any] | None = None
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            saved_keys = [entry["sample_key"] for entry in checkpoint["config"]["cases"]]
            if saved_keys != [case.key for case in cases] or checkpoint["config"]["model"] != asdict(model_config):
                raise ValueError("Resume checkpoint cases or model configuration do not match")
            if checkpoint["scaler"] != asdict(scaler):
                raise ValueError("Resume labels/scaler do not match the current input cases")
            model.load_state_dict(checkpoint["model"])
            for group in checkpoint["optimizer"]["param_groups"]:
                group["fused"] = device.type == "cuda"
            optimizer.load_state_dict(checkpoint["optimizer"])
            if args.override_resume_lr:
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate
            start_step = int(checkpoint["step"])
            best_error, best_step = checkpoint.get("best_error"), checkpoint.get("best_step")
            config["resumed_from"] = str(args.resume.resolve())
            config["resume_previous_best_error"] = best_error
            if bool(checkpoint["config"]["arguments"].get("amp", False)) != args.amp:
                raise ValueError("Resume --amp setting must match the checkpoint")
            if not (output / "best.pt").exists():
                # A new output directory starts a new best-of-run history, while retaining provenance.
                best_error, best_step = None, None
            random.setstate(checkpoint["random_state"])
            np.random.set_state(checkpoint["numpy_rng_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
            if args.epochs and start_step % batches_per_epoch:
                raise ValueError("Full-epoch resume requires a checkpoint at an epoch boundary; resume with --steps otherwise")
        if start_step > total_steps:
            raise ValueError("Requested total training duration is shorter than the resume checkpoint")
        config["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        config["optimizer_learning_rates"] = [group["lr"] for group in optimizer.param_groups]
        config["actual_backend"] = str(getattr(model, "backend", getattr(model, "runtime_backend", type(model).__name__)))
        write_json(output / "config.json", config)
        update_status("preparing", step=start_step, resume_step=start_step, total_steps=total_steps,
                      cases=config["cases"], encoder=args.encoder, input_mode=input_mode,
                      backend=config["actual_backend"], target_error=args.target_error,
                      hard_node_fraction=args.hard_node_fraction)
        cache_bytes = sum(case.count for case in cases) * model_config.hash_grid.n_levels * 8 * 8
        if args.cache_budget_gb:
            cache_budget = int(args.cache_budget_gb * (1 << 30))
        elif device.type == "cuda":
            cache_budget = int(torch.cuda.mem_get_info(device)[0] * 0.35)
        else:
            cache_budget = 1 << 30
        config["estimated_lookup_bytes"] = cache_bytes
        config["auto_lookup_budget_bytes"] = cache_budget
        should_cache = args.cache_lookups == "on" or (args.cache_lookups == "auto" and cache_bytes <= cache_budget)
        if should_cache:
            prepare_lookup = getattr(model, "prepare_lookup", None)
            for case in cases:
                case.lookup = prepare_lookup(case.xyz) if prepare_lookup is not None else None
            if args.cache_lookups == "on" and any(case.lookup is None for case in cases):
                raise ValueError("The selected model/backend does not support cached lookup tables")
        config["cached_lookups"] = [case.key for case in cases if case.lookup is not None]
        write_json(output / "config.json", config)
        gradient_scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
        if checkpoint is not None and checkpoint.get("amp_scaler"):
            gradient_scaler.load_state_dict(checkpoint["amp_scaler"])
        metrics_path = output / "metrics.csv"
        csv_fields = ["step", "epoch", "elapsed_seconds", "training_loss", "pooled_relative_l2",
                      "case_mean_relative_l2", "worst_case_relative_l2", "worst_case_key",
                      "pooled_hotspot_relative_l2", "case_mean_hotspot_relative_l2",
                      "worst_hotspot_relative_l2", "pooled_rmse", "pooled_mae",
                      "max_abs_error", "is_best", "eval_seconds"]
        last_metrics: dict[str, Any] = {}
        last_predictions: dict[str, np.ndarray] = {}
        last_loss: float | None = None
        step = start_step
        stop_reason = "duration_complete"

        def evaluate_and_record(current_step: int, loss: float | None) -> bool:
            nonlocal best_error, best_step, last_metrics, last_predictions
            update_status("evaluating", step=current_step)
            eval_started = time.monotonic()
            last_metrics, last_predictions = evaluate(model, cases, scaler, args.eval_batch_size)
            if args.hard_node_fraction > 0:
                _refresh_hard_indices(cases, last_predictions, args.hard_node_count)
                config["hard_node_sampling"]["refresh_steps"].append(current_step)
                write_json(output / "config.json", config)
            score = float(last_metrics["worst_case"]["relative_l2"])
            improved = best_error is None or score < best_error
            if improved:
                best_error, best_step = score, current_step
                _checkpoint(output / "best.pt", model, optimizer, scaler, config,
                            current_step, best_error, best_step, last_metrics, gradient_scaler)
                save_predictions(output / "predictions_best", cases, last_predictions, current_step, scaler)
                write_json(output / "metrics_best.json", {"step": current_step, **last_metrics})
            row = {
                "step": current_step, "epoch": current_step / batches_per_epoch if args.epochs else "",
                "elapsed_seconds": time.monotonic() - started_monotonic, "training_loss": loss,
                "pooled_relative_l2": last_metrics["pooled"]["relative_l2"],
                "case_mean_relative_l2": last_metrics["case_mean"]["relative_l2"],
                "worst_case_relative_l2": score, "worst_case_key": last_metrics["worst_case_key"],
                "pooled_hotspot_relative_l2": last_metrics["pooled"]["hotspot_relative_l2"],
                "case_mean_hotspot_relative_l2": last_metrics["case_mean"]["hotspot_relative_l2"],
                "worst_hotspot_relative_l2": max(entry["hotspot_relative_l2"] for entry in last_metrics["per_case"].values()),
                "pooled_rmse": last_metrics["pooled"]["rmse"], "pooled_mae": last_metrics["pooled"]["mae"],
                "max_abs_error": last_metrics["pooled"]["max_abs_error"],
                "is_best": int(improved), "eval_seconds": time.monotonic() - eval_started,
            }
            needs_header = not metrics_path.exists() or metrics_path.stat().st_size == 0
            with metrics_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=csv_fields)
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)
            write_json(output / "metrics_latest.json", {"step": current_step, **last_metrics})
            _checkpoint(output / "last.pt", model, optimizer, scaler, config,
                        current_step, best_error, best_step, last_metrics, gradient_scaler)
            update_status("training", step=current_step, latest_metrics=last_metrics,
                          best_worst_case_relative_l2=best_error, best_step=best_step)
            print(f"step {current_step}/{total_steps} | pooled relL2={last_metrics['pooled']['relative_l2']:.6g} "
                  f"| worst relL2={score:.6g} | hotspot relL2={last_metrics['pooled']['hotspot_relative_l2']:.6g}", flush=True)
            return args.target_error > 0 and score <= args.target_error

        target_reached = evaluate_and_record(step, last_loss)
        permutations: dict[str, torch.Tensor] = {}
        last_status_time = time.monotonic()
        while step < total_steps and not target_reached:
            epoch_batch = step % batches_per_epoch if args.epochs else None
            if args.epochs and epoch_batch == 0:
                permutations = {case.key: torch.randperm(case.count, device=device) for case in cases}
            model.train()
            optimizer.zero_grad(set_to_none=True)
            selections: list[tuple[TrainingCase, torch.Tensor]] = []
            for case in cases:
                if args.epochs:
                    start = int(epoch_batch) * args.nodes_per_case
                    indices = permutations[case.key][start:start + args.nodes_per_case]
                    if not len(indices):
                        continue
                else:
                    indices = _sample_case_indices(case, args.nodes_per_case, args.hard_node_fraction)
                selections.append((case, indices))
            detached_losses: list[torch.Tensor] = []
            for case, indices in selections:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                    prediction = _forward(model, case, indices)
                    loss = torch.mean((prediction.float() - case.target[indices]).square()) / len(selections)
                gradient_scaler.scale(loss).backward()
                detached_losses.append(loss.detach())
            # A single scalar transfer avoids synchronizing CUDA separately for every case.
            loss_total = float(torch.stack(detached_losses).sum())
            if not math.isfinite(loss_total):
                raise FloatingPointError(f"Nonfinite training loss at step {step + 1}")
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            step += 1
            last_loss = loss_total
            if time.monotonic() - last_status_time >= 1.0 or step == total_steps:
                update_status("training", step=step, training_loss=last_loss,
                              epoch=step / batches_per_epoch if args.epochs else None)
                last_status_time = time.monotonic()
            if step % args.eval_every == 0 or step == total_steps:
                target_reached = evaluate_and_record(step, last_loss)
        if target_reached:
            stop_reason = "target_error_reached"
        # Last evaluation is always at the stopping step, including an untrained baseline.
        save_predictions(output / "predictions_final", cases, last_predictions, step, scaler)
        summary = {"status": "completed", "stop_reason": stop_reason, "step": step,
                   "total_steps": total_steps, "best_step": best_step,
                   "best_worst_case_relative_l2": best_error,
                   "target_error": args.target_error,
                   "target_met": bool(args.target_error > 0 and best_error is not None
                                      and best_error <= args.target_error),
                   "elapsed_seconds": time.monotonic() - started_monotonic,
                   "started_at_utc": started_at, "completed_at_utc": utc_now(),
                   "final_metrics": last_metrics, "scaler": scaler.record(),
                   "best_metrics": json.loads((output / "metrics_best.json").read_text(encoding="utf-8"))
                                   if (output / "metrics_best.json").exists() else None,
                   "artifacts": {"config": "config.json", "metrics": "metrics.csv",
                                 "best_metrics": "metrics_best.json",
                                 "best_checkpoint": "best.pt", "last_checkpoint": "last.pt",
                                 "best_predictions": "predictions_best/index.json",
                                 "final_predictions": "predictions_final/index.json"}}
        write_json(output / "summary.json", summary)
        update_status("completed", step=step, stop_reason=stop_reason, summary="summary.json")
        print(f"Completed: {output}", flush=True)
        return summary
    except BaseException as exc:
        update_status("failed", error_type=type(exc).__name__, error=str(exc))
        raise


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
