"""Frozen-Hash50 transfer pilots and independent single-case fit diagnostics.

Evaluation and diagnostic fits use the parent checkpoint's asinh scaler without
fitting any new labels. Fits start from scratch with the parent's architecture.
Neither mode is a pristine final blind-test protocol.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import fields
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from .train import (StressScaler, TrainingCase, find_case_files, load_cases,
                    save_predictions, stress_metrics, utc_now, write_json)
from .tcnn_fields import TCNNField, TCNNFieldConfig


@contextmanager
def exclusive_lock(path: Path):
    """Process-lifetime lock; stale PID text alone never blocks recovery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(" ")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(f"Another process holds {path}") from error
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Another process holds {path}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at_utc": utc_now()}))
        handle.flush()
        yield
    finally:
        handle.close()


def checkpoint_scaler(checkpoint: dict[str, Any]) -> StressScaler:
    """Require the saved scaler, with no data-dependent fallback."""
    record = checkpoint["scaler"]
    names = ("scale", "center", "spread")
    scaler = StressScaler(**{name: float(record[name]) for name in names})
    if (not all(math.isfinite(getattr(scaler, name)) for name in names)
            or scaler.scale <= 0 or scaler.spread <= 0):
        raise ValueError("Invalid checkpoint scaler")
    saved_config = checkpoint["config"].get("scaler", {})
    if any(float(saved_config.get(name, math.nan)) != getattr(scaler, name) for name in names):
        raise ValueError("Checkpoint scaler and checkpoint config disagree")
    return scaler


def case_exposure(checkpoint: dict[str, Any], keys: list[str], mode: str) -> dict[str, Any]:
    trained = [entry["sample_key"] for entry in checkpoint["config"]["cases"]]
    by_cad: dict[str, list[str]] = {}
    for key in trained:
        cad, load = key.split("/", 1)
        by_cad.setdefault(cad, []).append(load)
    per_case = {}
    for key in keys:
        cad, load = key.split("/", 1)
        per_case[key] = {"cad": cad, "load": load, "seen_cad": cad in by_cad,
                         "seen_case": key in trained,
                         "training_loads_for_cad": by_cad.get(cad, [])}
    if mode == "evaluate" and any(item["seen_case"] for item in per_case.values()):
        raise ValueError("Transfer evaluation cannot include a parent training CAD/load case")
    if mode == "fit" and any(not item["seen_case"] for item in per_case.values()):
        raise ValueError("Independent diagnostics require a case represented in parent metrics")
    unseen_cad = all(not item["seen_cad"] for item in per_case.values())
    scope = ("independent single-case fit diagnostic" if mode == "fit" else
             "unseen-CAD migration pilot" if unseen_cad else
             "unseen-load-combination migration pilot")
    if mode == "evaluate" and unseen_cad:
        assert set(key.split("/", 1)[0] for key in keys).isdisjoint(by_cad)
    return {"scope": scope, "trained_case_keys": trained, "trained_cad_keys": sorted(by_cad),
            "evaluated_case_keys": keys, "all_evaluated_CADs_unseen": unseen_cad,
            "parent_training_case_overlap": [], "per_case": per_case,
            "claim_limit": "diagnosis/pilot; historical experiment use excludes a pristine final blind-test claim"}


def architecture(checkpoint: dict[str, Any], seed: int | None = None) -> TCNNFieldConfig:
    config = checkpoint["config"]
    if config.get("actual_backend") != "tcnn_fully_fused":
        raise ValueError("This phase requires the official FullyFusedMLP Hash50 checkpoint")
    record = config["implementation_details"]
    arguments = {entry.name: record[entry.name] for entry in fields(TCNNFieldConfig)}
    if seed is not None:
        arguments["seed"] = seed
    result = TCNNFieldConfig(**arguments)
    if result.input_mode != "both" or result.condition_dim != 25 or result.out_channels != 1:
        raise ValueError("Expected parent xyz + 25 conditions -> signed stress architecture")
    current = result.record()
    for name in ("source_commit", "encoding", "network"):
        if current[name] != record[name]:
            raise ValueError(f"Parent TCNN {name} differs from the installed implementation")
    return result


def request_identity(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve()
    stat = checkpoint.stat()
    identity = {"mode": args.mode, "checkpoint": str(checkpoint),
                "checkpoint_bytes": stat.st_size, "checkpoint_mtime_ns": stat.st_mtime_ns,
                "data": [str(Path(item).resolve()) for item in args.data],
                "case_keys": args.case_keys}
    if args.mode == "fit":
        identity.update(steps=args.steps, epochs=args.epochs, batch_size=args.batch_size,
                        learning_rate=args.learning_rate, seed=args.seed,
                        eval_every=args.eval_every, eval_every_epochs=args.eval_every_epochs)
    return identity


def identity_digest(identity: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def aggregate_metrics(per_case: dict[str, Any], cases: list[TrainingCase],
                      predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    pooled = stress_metrics(np.concatenate([case.physical_stress for case in cases]),
                            np.concatenate([predictions[case.key] for case in cases]))
    means = {name: float(np.mean([item[name] for item in per_case.values()]))
             for name in pooled if name not in {"nodes", "hotspot_nodes"}}
    worst_key = max(per_case, key=lambda key: per_case[key]["relative_l2"])
    return {"per_case": per_case, "pooled": pooled, "case_mean": means,
            "worst_case_key": worst_key, "worst_case": per_case[worst_key],
            "case_mean_relative_l2": means["relative_l2"],
            "case_mean_hotspot_relative_l2": means["hotspot_relative_l2"],
            "worst_case_relative_l2": per_case[worst_key]["relative_l2"],
            "evaluation": "all original nodes, physical signed stress, no weights or truncation"}


@torch.no_grad()
def evaluate_all(model: torch.nn.Module, cases: list[TrainingCase], scaler: StressScaler,
                 batch_size: int, update) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    total = sum(case.count for case in cases)
    done = 0
    predictions, per_case = {}, {}
    for case in cases:
        encoded = np.empty(case.count, dtype=np.float32)
        for start in range(0, case.count, batch_size):
            stop = min(start + batch_size, case.count)
            encoded[start:stop] = model(case.xyz[start:stop], case.features[start:stop]).reshape(-1).float().cpu().numpy()
            done += stop - start
            update("evaluating", evaluated_nodes=done, evaluation_total_nodes=total,
                   current_case=case.key)
        prediction = scaler.decode(encoded)
        predictions[case.key] = prediction
        per_case[case.key] = stress_metrics(case.physical_stress, prediction)
    return aggregate_metrics(per_case, cases, predictions), predictions


def atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def comparison(parent: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    result = {}
    parent_per = parent["metrics"]["per_case"]
    for key, current in metrics["per_case"].items():
        if key not in parent_per:
            continue
        baseline = parent_per[key]
        result[key] = {"parent_checkpoint_step": parent["step"], "parent_shared50": baseline,
                       "independent_fit": current,
                       "relative_l2_ratio_independent_to_parent": current["relative_l2"] / max(baseline["relative_l2"], 1e-12),
                       "claim": "per-case capacity/optimization diagnostic, not transfer/generalization"}
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    identity = request_identity(args)
    digest = identity_digest(identity)
    output = args.output.resolve()
    if output == args.checkpoint.resolve().parent:
        raise ValueError("An exploration output cannot overwrite the parent run")
    if args.mode == "fit" and len(args.case_keys) != 1:
        raise ValueError("Each independent-fit job must contain exactly one case")
    parent = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    scaler = checkpoint_scaler(parent)
    exposure = case_exposure(parent, args.case_keys, args.mode)
    model_config = architecture(parent, args.seed if args.mode == "fit" else None)
    files = find_case_files(args.data, args.case_keys, len(args.case_keys))
    if args.dry_run:
        cases, supplied = load_cases(files, torch.device("cpu"), scaler=scaler)
        record = {"dry_run": True, "output": str(output), "request_identity": identity,
                  "scaler": supplied.record(), "scaler_source": "parent checkpoint; never refitted",
                  "exposure": exposure, "architecture": model_config.record(),
                  "cases": [{"sample_key": case.key, "nodes": case.count, "source": str(case.source)} for case in cases]}
        print(json.dumps(record, indent=2), flush=True)
        return record
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(output / ".run.lock"):
        summary_path = output / "summary.json"
        if summary_path.exists():
            existing = json.loads(summary_path.read_text())
            if existing.get("status") == "completed":
                if existing["config"].get("request_digest") != digest:
                    raise ValueError("Completed output belongs to a different request; choose a new directory")
                print(f"Reusing completed {output}", flush=True)
                return existing
        config_path = output / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()).get("request_digest") != digest:
            raise ValueError("Incomplete output belongs to a different request; choose a new directory")
        if args.mode == "fit" and config_path.exists() and args.resume is None:
            raise ValueError("Incomplete fitting requires --resume from its checkpoint")
        elapsed_previous = 0.0
        status = {"phase": "loading", "mode": args.mode, "scope": exposure["scope"],
                  "pid": os.getpid(), "started_at_utc": utc_now(), "step": 0}
        phase_start = time.monotonic()
        last_phase = "loading"
        def update(phase: str, **extra):
            nonlocal phase_start, last_phase
            if phase != last_phase:
                phase_start, last_phase = time.monotonic(), phase
            status.update(extra, phase=phase, elapsed_seconds=elapsed_previous + time.monotonic() - started,
                          active_session_elapsed_seconds=time.monotonic() - started,
                          updated_at_utc=utc_now())
            eta = None
            if args.mode == "evaluate" and phase == "evaluating" and status.get("evaluated_nodes", 0):
                eta = (time.monotonic() - phase_start) / status["evaluated_nodes"] * (status["evaluation_total_nodes"] - status["evaluated_nodes"])
            elif args.mode == "fit" and status.get("step", 0) > status.get("resumed_step", 0):
                progress = status["step"] - status.get("resumed_step", 0)
                eta = (time.monotonic() - started) / progress * max(0, status.get("total_steps", 0) - status["step"])
            if phase == "completed":
                eta = 0.0
            status["eta_seconds"] = eta
            status["estimated_completion_utc"] = ((datetime.now(timezone.utc) + timedelta(seconds=eta)).isoformat() if eta is not None else None)
            write_json(output / "status.json", status)
        update("loading")
        try:
            device = torch.device(args.device)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            cases, fixed_scaler = load_cases(files, device, scaler=scaler)
            assert fixed_scaler == scaler
            model = TCNNField(config=model_config, device=device)
            if args.mode == "evaluate":
                model.load_state_dict(parent["model"])
            count = sum(case.count for case in cases)
            batches = math.ceil(count / args.batch_size)
            total_steps = args.epochs * batches if args.epochs else args.steps
            config = {"arguments": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
                      "request_identity": identity, "request_digest": digest,
                      "method": exposure["scope"], "dataset": "original DeepJEB v1.0",
                      "exposure": exposure, "parent_checkpoint": str(args.checkpoint.resolve()),
                      "parent_checkpoint_step": parent["step"],
                      "parent_reference_per_case": parent["metrics"]["per_case"],
                      "parent_training_cases": parent["config"]["cases"],
                      "scaler": scaler.record(), "scaler_source": "saved parent checkpoint scaler; no evaluation/diagnostic label fitting",
                      "cases": [{"sample_key": case.key, "source": str(case.source), "nodes": case.count} for case in cases],
                      "total_nodes": count, "implementation_details": model.config.record(),
                      "actual_backend": model.backend, "parameter_count": sum(p.numel() for p in model.parameters()),
                      "initialization": "frozen parent weights" if args.mode == "evaluate" else "scratch random weights; parent provides architecture and scaler only",
                      "input_schema": {"xyz": "float32[N,3], per-CAD normalized [0,1]", "features": "float32[N,25], existing geometry/BC/material schema",
                                       "stress": "physical signed MPa [N]", "alignment": "official CSV GRID coordinates -> H5 vertices"},
                      "precision": {"native": model.config.record()["precision"], "matmul": "high", "allow_tf32": True, "cudnn_allow_tf32": True},
                      "protocol": "frozen all-node evaluation" if args.mode == "evaluate" else "each node once per shuffled full epoch" if args.epochs else "uniform random nodes with replacement",
                      "batch_size": args.batch_size, "batches_per_epoch": batches,
                      "total_steps": total_steps if args.mode == "fit" else 0,
                      "checkpoint_selection": "minimum all-original-node decoded physical stress relative L2 on the fitted case" if args.mode == "fit" else "parent best checkpoint, no evaluation-driven selection"}
            write_json(config_path, config)
            if args.mode == "evaluate":
                update("evaluating", evaluated_nodes=0, evaluation_total_nodes=count)
                metrics, predictions = evaluate_all(model, cases, scaler, args.eval_batch_size, update)
                metrics.update(step=parent["step"], elapsed_seconds=time.monotonic() - started)
                save_predictions(output / "predictions", cases, predictions, parent["step"], scaler)
                write_json(output / "metrics.json", metrics)
                summary = {"status": "completed", "mode": args.mode, "scope": exposure["scope"],
                           "transfer_classification": exposure["per_case"],
                           "metrics": metrics, "best_metrics": metrics, "final_metrics": metrics,
                           "prediction_index": str(output / "predictions" / "index.json"),
                           "elapsed_seconds": time.monotonic() - started, "config": config, "completed_at_utc": utc_now()}
                write_json(summary_path, summary)
                update("completed")
                print(f"pilot nodes={count} mean={metrics['case_mean_relative_l2']:.6g} worst={metrics['worst_case_relative_l2']:.6g}", flush=True)
                return summary

            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, fused=device.type == "cuda")
            step, best_step, best_score = 0, None, float("inf")
            permutation = None
            last_metrics = None
            if args.resume:
                if args.resume.resolve().parent != output:
                    raise ValueError("Resume must be an exploration checkpoint inside the same output directory")
                resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
                if resumed["config"]["request_digest"] != digest or checkpoint_scaler(resumed) != scaler:
                    raise ValueError("Resume request/scaler differ")
                model.load_state_dict(resumed["model"])
                optimizer.load_state_dict(resumed["optimizer"])
                step, best_step, best_score = resumed["step"], resumed["best_step"], resumed["best_score"]
                permutation = resumed.get("epoch_permutation")
                if permutation is not None:
                    permutation = permutation.to(device)
                torch.set_rng_state(resumed["torch_rng_state"].cpu())
                if device.type == "cuda":
                    torch.cuda.set_rng_state_all([state.cpu() for state in resumed["cuda_rng_state"]])
                np.random.set_state(resumed["numpy_rng_state"])
                random.setstate(resumed["python_rng_state"])
                elapsed_previous = resumed.get("elapsed_seconds", 0.0)
                last_metrics = resumed["metrics"]
                config["resumed_from"] = str(args.resume.resolve())
            config["optimizer"] = {"name": "Adam", "fused": device.type == "cuda", "learning_rate": optimizer.param_groups[0]["lr"], "betas": list(optimizer.param_groups[0]["betas"])}
            write_json(config_path, config)
            update("training", step=step, resumed_step=step, total_steps=total_steps)
            def checkpoint(path: Path, metrics: dict[str, Any]):
                atomic_checkpoint(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": config, "step": step, "scaler": scaler.record(), "metrics": metrics,
                    "best_step": best_step, "best_score": best_score,
                    "epoch_permutation": permutation.cpu() if permutation is not None else None,
                    "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                    "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
                    "elapsed_seconds": elapsed_previous + time.monotonic() - started})
            def evaluate_fit(loss: float | None):
                nonlocal best_step, best_score
                update("evaluating", evaluated_nodes=0, evaluation_total_nodes=count, step=step)
                metrics, predictions = evaluate_all(model, cases, scaler, args.eval_batch_size, update)
                metrics.update(step=step, epoch=step / batches if args.epochs else None,
                               training_loss=loss, elapsed_seconds=elapsed_previous + time.monotonic() - started)
                if metrics["worst_case_relative_l2"] < best_score:
                    best_score, best_step = metrics["worst_case_relative_l2"], step
                    checkpoint(output / "best.pt", metrics)
                    write_json(output / "metrics_best.json", metrics)
                    save_predictions(output / "predictions_best", cases, predictions, step, scaler)
                checkpoint(output / "last.pt", metrics)
                write_json(output / "metrics_latest.json", metrics)
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, allow_nan=False) + "\n")
                update("training", step=step, best_step=best_step, best_score=best_score, latest_metrics=metrics)
                print(f"independent {cases[0].key} {step}/{total_steps} stress={metrics['worst_case_relative_l2']:.6g}", flush=True)
                return metrics
            if last_metrics is None:
                last_metrics = evaluate_fit(None)
            case = cases[0]
            loss_value = None
            while step < total_steps:
                if args.epochs:
                    batch = step % batches
                    if batch == 0:
                        permutation = torch.randperm(case.count, device=device)
                    if permutation is None:
                        raise ValueError("Full-epoch resume is missing the saved permutation")
                    rows = permutation[batch * args.batch_size:(batch + 1) * args.batch_size]
                else:
                    rows = torch.randint(case.count, (args.batch_size,), device=device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = (model(case.xyz[rows], case.features[rows]).reshape(-1) - case.target[rows]).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at {step + 1}")
                loss.backward()
                optimizer.step()
                step += 1
                should_eval = (step == total_steps or
                               (not args.epochs and step % args.eval_every == 0) or
                               (args.epochs and step % batches == 0 and (step // batches) % args.eval_every_epochs == 0))
                if should_eval or step % 100 == 0:
                    loss_value = float(loss.detach())
                    update("training", step=step, training_loss=loss_value, epoch=step / batches if args.epochs else None)
                if should_eval:
                    last_metrics = evaluate_fit(loss_value)
            best_metrics = json.loads((output / "metrics_best.json").read_text())
            summary = {"status": "completed", "mode": args.mode, "scope": exposure["scope"],
                       "step": step, "best_step": best_step, "best_metrics": best_metrics,
                       "final_metrics": last_metrics, "parent_comparison": comparison(parent, best_metrics),
                       "prediction_index": str(output / "predictions_best" / "index.json"),
                       "elapsed_seconds": elapsed_previous + time.monotonic() - started,
                       "config": config, "completed_at_utc": utc_now()}
            write_json(output / "parent_comparison.json", summary["parent_comparison"])
            write_json(summary_path, summary)
            update("completed", step=step)
            return summary
        except BaseException as error:
            update("failed", error_type=type(error).__name__, error=str(error))
            raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("evaluate", "fit"), required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--data", nargs="+", required=True)
    result.add_argument("--case-keys", nargs="+", required=True)
    duration = result.add_mutually_exclusive_group()
    duration.add_argument("--steps", type=int)
    duration.add_argument("--epochs", type=int)
    result.add_argument("--batch-size", type=int, default=51200)
    result.add_argument("--learning-rate", type=float, default=.003)
    result.add_argument("--eval-every", type=int, default=200)
    result.add_argument("--eval-every-epochs", type=int, default=10)
    result.add_argument("--eval-batch-size", type=int, default=65536)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--device", default="cuda")
    result.add_argument("--resume", type=Path)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--dry-run", action="store_true", help="CPU-only checkpoint, scaler, exposure and input checks")
    return result


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    if args.steps is None and args.epochs is None:
        args.steps = 2000
    if min(args.steps if args.steps is not None else 1,
           args.epochs if args.epochs is not None else 1, args.batch_size, args.eval_every,
           args.eval_every_epochs, args.eval_batch_size) < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        cli.error("Durations, batches, evaluation intervals and learning rate must be positive")
    if args.mode == "evaluate" and args.resume:
        cli.error("Frozen evaluation uses --checkpoint; --resume is for fitting only")
    run(args)


if __name__ == "__main__":
    main()
