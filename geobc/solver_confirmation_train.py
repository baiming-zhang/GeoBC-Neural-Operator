"""Budgeted solver confirmation using the frozen solver-screen mathematics.

Candidate methods and the actual budget belong to a separate frozen plan.
Source checking is standard-library-only; this module never prepares data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

from . import solver_selection_train as screen

PASS_CASES = 120
SEEDS = (2026, 2027)
BUDGET_MODE = "total_training_seconds"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _recipe(plan, method=None, seed=None, require_frozen=False):
    recipe = dict(plan["recipe"])
    methods = [screen.canonical_method(item) for item in recipe["methods"]]
    if len(set(methods)) != len(methods) or (not methods and (method is not None or require_frozen)):
        raise ValueError("Confirmation methods must be a nonempty unique subset of the screen families")
    seeds = recipe.get("seeds", [recipe.get("seed", 2026)])
    if not seeds or any(item not in SEEDS for item in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Confirmation seeds must be a subset of 2026/2027")
    seed = recipe.get("seed", seeds[0]) if seed is None else seed
    if seed not in seeds or (method is not None and method not in methods):
        raise ValueError("Requested method/seed is absent from the confirmation recipe")
    expected = {"train_loads": ["ver", "hor", "tor"], "cloud_seed": 2026, "device": "cuda",
                "learning_rate": .001, "weight_decay": 0., "optimizer": "Adam",
                "cloud_points": 1024, "geometry_dim": 32, "grid_resolution": 64,
                "microbatch_nodes": screen.CHUNK, "eval_batch_size": screen.CHUNK,
                "stress_transform": "asinh", "channels": [0, 1, 2, 3],
                "training_readout_keys": ["101_479/ver", "107_320/hor", "10_58/tor"]}
    if any(recipe.get(key, value) != value for key, value in expected.items()):
        raise ValueError("Confirmation cannot change the shared screen data/model/loss recipe")
    for key, value in expected.items():
        recipe.setdefault(key, value)
    recipe.update(methods=methods, seeds=list(seeds), eval_every=recipe.get("eval_every", 120),
                  train_budget_seconds=recipe.get("train_budget_seconds", 600.),
                  max_steps=recipe.get("max_steps", 12000), budget_mode=BUDGET_MODE)
    if plan["recipe"].get("budget_mode", BUDGET_MODE) != BUDGET_MODE:
        raise ValueError("Only cumulative total training seconds are a confirmation budget")
    if recipe["eval_every"] != PASS_CASES:
        raise ValueError("Confirmation evaluates at every complete 120-case TRAIN pass")
    budget, maximum = recipe["train_budget_seconds"], recipe["max_steps"]
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("train_budget_seconds must be finite and positive")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < PASS_CASES:
        raise ValueError("max_steps must be an integer at least 120")
    if require_frozen and plan.get("candidates_frozen") is not True:
        raise ValueError("Actual confirmation requires plan.candidates_frozen=true")
    return recipe, seed


def case_schedule(keys, seed, max_steps):
    """Independent Python RNG; never advances model or training RNG streams."""
    sampler = random.Random(seed)
    order = []
    while len(order) < max_steps:
        permutation = list(keys); sampler.shuffle(permutation); order.extend(permutation)
    return order[:max_steps]


def _schedule_rng(keys, seed, step):
    sampler = random.Random(seed)
    for _ in range(math.ceil(step / PASS_CASES)):
        permutation = list(keys); sampler.shuffle(permutation)
    return sampler.getstate()


def _source_identities():
    return {**screen._source_identities(),
            "geobc/solver_confirmation_train.py": screen.source_sha("geobc/solver_confirmation_train.py")}


def request_contract(plan_path, method, seed=None, warmstart=None):
    """Read existing cache metadata only; no arrays, weights or model creation."""
    from .solver_selection_data import read_cache_metadata
    path = screen.host_path(plan_path); plan = screen.read_json(path)
    method = screen.canonical_method(method)
    recipe, seed = _recipe(plan, method, seed)
    data_plan = {**plan, "recipe": {**recipe, "seed": 2026}}
    manifest = read_cache_metadata(data_plan); screen._check_manifest(manifest)
    order = case_schedule(manifest["train_keys"], seed, recipe["max_steps"])
    if seed == 2026 and order[:min(480, len(order))] != manifest["case_order"][:min(480, len(order))]:
        raise ValueError("Seed2026 must preserve the exact first480 screen case order")
    initialization = {"mode": "fresh"}
    if warmstart is not None:
        if seed != 2026 or recipe["max_steps"] < 480:
            raise ValueError("Screen warmstart requires seed2026 and max_steps>=480")
        checkpoint = screen.host_path(warmstart); stat = checkpoint.stat()
        initialization = {"mode": "warmstart", "checkpoint": str(checkpoint),
                          "checkpoint_stat": {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                          "parent_config": screen.read_json(checkpoint.parent / "config.json")}
    identity = {"recipe": recipe, **{key: plan[key] for key in ("existing_split", "prepared_data", "data_cache")}}
    contract = {"plan": str(path), "recipe": recipe, "recipe_sha256": _digest(identity),
                "method": method, "seed": seed, "initialization": initialization,
                "cache_contract": manifest["contract"], "scaler": manifest["scaler"],
                "train_keys": manifest["train_keys"], "validation_keys": manifest["validation_keys"],
                "training_readout_keys": manifest["training_readout_keys"],
                "case_order_sha256": _digest(order), "source_sha256": _source_identities()}
    return contract, manifest


def source_check(plan_path):
    """One future plan/source check; no scientific dependencies are imported."""
    import ast
    plan = screen.read_json(plan_path); recipe, seed = _recipe(plan)
    path = screen.host_path("geobc/solver_confirmation_train.py")
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {"status": "source_ready", "recipe": recipe, "default_seed": seed,
            "candidates_frozen": plan.get("candidates_frozen") is True,
            "source_sha256": _source_identities(), "plan_sha256": screen.source_sha(plan_path),
            "scientific_arrays_read": 0, "models_constructed": 0, "CUDA_initialized": False}


def _validate_warmstart(saved, config, manifest):
    previous = saved["config"]
    if saved["step"] != 480 or previous["protocol"] != screen.PROTOCOL:
        raise ValueError("Warmstart must be the seed2026 committed screen last480 checkpoint")
    if previous != config["contract"]["initialization"]["parent_config"]:
        raise ValueError("Screen checkpoint/config metadata differ")
    for key in ("method", "scaler", "model", "parameter_counts", "cache_contract",
                "train_keys", "validation_keys", "training_readout_keys"):
        if previous[key] != config[key]:
            raise ValueError(f"Warmstart changed shared {key}")
    if previous["case_order"] != manifest["case_order"] or saved["next_case_position"] != 480:
        raise ValueError("Warmstart changed the completed screen case schedule")
    expected = {**previous["contract"], "source_sha256": screen._source_identities()}
    if not screen.contracts_compatible(previous["contract"], expected):
        raise ValueError("Warmstart screen math/model/data sources differ")
    if saved["case_shuffle_rng_state"] != _schedule_rng(manifest["train_keys"], 2026, 480):
        raise ValueError("Warmstart independent shuffle RNG differs")


def _same_config(previous, current):
    return {k: v for k, v in previous.items() if k != "report_phase"} == {
        k: v for k, v in current.items() if k != "report_phase"}


def run(args):
    if args.source_check:
        result = source_check(args.plan); print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    if args.output is None:
        raise ValueError("--output is required for actual confirmation")
    method = screen.canonical_method(args.method)
    plan = screen.read_json(args.plan); recipe, seed = _recipe(plan, method, args.seed, require_frozen=True)
    output = screen.host_path(args.output)
    warmstart = args.warmstart
    if args.resume is not None and warmstart is None and (output / "config.json").is_file():
        warmstart = screen.read_json(output / "config.json")["contract"]["initialization"].get("checkpoint")
    contract, manifest = request_contract(args.plan, method, seed, warmstart)
    if (output / "summary.json").is_file():
        stored = screen.read_json(output / "summary.json")
        if stored.get("status") == "completed":
            if stored["config"]["contract"] != contract or not _same_config(screen.read_json(output / "config.json"), stored["config"]):
                raise ValueError("Completed confirmation contract differs")
            print(f"Reusing completed confirmation {method}/seed{seed}: {output}", flush=True)
            return stored
    resume = None if args.resume is None else (output / "last.pt" if args.resume == "last" else screen.host_path(args.resume))
    if resume is not None and not resume.is_file():
        raise FileNotFoundError(resume)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError("Incomplete nonempty confirmation output requires --resume")
    output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic(); prior_elapsed = 0.; parent_elapsed = 0.; parent_train_seconds = 0.
    state = {"method": method, "seed": seed, "started_at_utc": screen.utc_now(), "step": 0,
             "max_steps": recipe["max_steps"], "budget_mode": BUDGET_MODE,
             "train_budget_seconds": recipe["train_budget_seconds"], "evaluated_steps": []}
    def elapsed():
        return prior_elapsed + time.monotonic() - began
    def status(phase, **details):
        state.update(details); state.update(phase=phase, updated_at_utc=screen.utc_now(), elapsed_seconds=elapsed())
        costs = state.get("costs", {}); count = costs.get("train_case_exposures", 0)
        eta = None
        if count > 0:
            average = costs["train_seconds"] / count
            needed = math.ceil(max(0., recipe["train_budget_seconds"] - costs["train_seconds"]) / average) if average > 0 else 0
            endpoint = min(recipe["max_steps"], PASS_CASES * math.ceil((state["step"] + needed) / PASS_CASES))
            remaining = max(0, endpoint - state["step"])
            evaluations = len(state["evaluated_steps"])
            eta = average * remaining
            if evaluations:
                eta += costs["evaluation_seconds"] / evaluations * math.ceil(remaining / PASS_CASES)
        if phase == "completed": eta = 0.
        if phase == "failed": eta = None
        state["eta_seconds"] = eta
        state["estimated_completion_utc"] = (datetime.now(timezone.utc) + timedelta(seconds=eta)).isoformat() if eta is not None else None
        screen.write_json(output / "status.json", state)
    status("loading_shared_cache_metadata")
    cleanup = None
    try:
        import numpy as np
        import torch
        from .benchmark import FieldScaler
        from .solver_selection_data import load_case, query_grid
        from .solver_selection_models import build_selection_model, close_selection_process_group
        cleanup = close_selection_process_group
        device = torch.device("cuda")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        model = build_selection_model(method, device=device, seed=seed)
        scaler = FieldScaler(**manifest["scaler"])
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0, fused=True)
        order = case_schedule(manifest["train_keys"], seed, recipe["max_steps"])
        protocol = {**screen.PROTOCOL, "seed": seed, "updates": recipe["max_steps"], "passes": None,
                    "eval_steps": "initial + every120 + safety-cap endpoint", "eval_every": PASS_CASES,
                    "train_budget_seconds": recipe["train_budget_seconds"], "budget_mode": BUDGET_MODE}
        config = {"contract": contract, "method": method, "seed": seed, "protocol": protocol,
                  "report_phase": plan.get("report_phase", "solver_confirmation"),
                  "scaler": manifest["scaler"], "model": model.source_config,
                  "parameter_counts": model.parameter_counts, "case_order": order,
                  "case_order_sha256": contract["case_order_sha256"], "cache_contract": manifest["contract"],
                  "train_keys": manifest["train_keys"], "validation_keys": manifest["validation_keys"],
                  "training_readout_keys": manifest["training_readout_keys"],
                  "geometry_context": "unchanged cloud_seed2026 and PointNet32; differentiable encode each complete case",
                  "loss": screen.PROTOCOL["loss"], "checkpoint_selection": screen.PROTOCOL["selection_score"],
                  "precision": "screen FP32/TF32; native Hash FP16 compute/FP32 master; no autocast",
                  "evaluation": "VAL15 and TRAIN3 full nodes; ++ fixed case seed fork_rng restores training RNG",
                  "torch_version": torch.__version__}
        start_step, phase_start_step, best_step, best_score = 0, 0, None, math.inf
        best_metrics, gradient_check, phase_initial = None, None, None
        history = []; exposures = {key: 0 for key in manifest["train_keys"]}
        costs = {"train_seconds": 0., "inference_seconds": 0., "evaluation_seconds": 0.,
                 "train_case_exposures": 0, "train_node_exposures": 0,
                 "inference_case_exposures": 0, "inference_node_exposures": 0}
        prior_peak = {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
        budget_crossing_step = None; best_checkpoint_origin = None
        saved = None
        if resume is not None or warmstart is not None:
            checkpoint_path = resume if resume is not None else screen.host_path(warmstart)
            saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if resume is not None:
                if not _same_config(saved["config"], config):
                    raise ValueError("Resume confirmation science/geometry/optimizer config differs")
                parent_train_seconds = saved["parent_train_seconds"]
                parent_elapsed = saved["parent_elapsed_seconds"]
                phase_start_step = saved["phase_start_step"]
                phase_initial = saved["phase_initial_metrics"]
                budget_crossing_step = saved["budget_crossing_step"]
                best_checkpoint_origin = saved["best_checkpoint_origin"]
            else:
                _validate_warmstart(saved, config, manifest)
                parent_train_seconds = saved["costs"]["train_seconds"]
                parent_elapsed = saved["elapsed_seconds"]
                phase_start_step = saved["step"]; phase_initial = saved["metrics"]
                best_checkpoint_origin = {"phase": "screen", "checkpoint": str(checkpoint_path.parent / "best.pt"), "step": saved["best_step"]}
            start_step = saved["step"]
            if start_step % PASS_CASES and start_step != recipe["max_steps"]:
                raise ValueError("Resume requires a committed pass endpoint or max_steps endpoint")
            if saved["next_case_position"] != start_step or start_step > recipe["max_steps"]:
                raise ValueError("Saved case position exceeds the confirmation schedule")
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
            screen._restore_rng(torch, np, saved["rng"])
            if saved["case_shuffle_rng_state"] != _schedule_rng(manifest["train_keys"], seed, start_step):
                raise ValueError("Saved independent sampler RNG differs")
            best_step, best_score, best_metrics = saved["best_step"], saved["best_score"], saved["best_metrics"]
            gradient_check, history = saved["first_update_gradient_check"], saved["history"]
            costs, exposures = saved["costs"], saved["case_exposure_counts"]
            prior_elapsed = saved["elapsed_seconds"]; prior_peak = saved["memory"]
            prefix_counts = Counter(order[:start_step])
            if exposures != {key: prefix_counts[key] for key in manifest["train_keys"]} or costs["train_case_exposures"] != start_step:
                raise ValueError("Saved exposures differ from the exact scheduled prefix")
            if best_metrics is not None:
                screen.write_json(output / "metrics_best.json", best_metrics)
            if resume is None:
                inherited_best = checkpoint_path.parent / "best.pt"
                if not inherited_best.is_file():
                    raise FileNotFoundError(inherited_best)
                shutil.copyfile(inherited_best, output / "best.pt")
        screen.write_json(output / "config.json", config)
        if saved is not None:
            (output / "metrics.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in history), encoding="utf-8")
        torch.cuda.reset_peak_memory_stats(device)
        def memory():
            actual = screen._memory(torch, device)
            return {key: max(actual[key], prior_peak[key]) for key in actual}
        def budget_record(step):
            actual = costs["train_seconds"]
            return {"mode": BUDGET_MODE, "requested_seconds": recipe["train_budget_seconds"],
                    "parent_train_seconds": parent_train_seconds,
                    "incremental_train_seconds": actual - parent_train_seconds,
                    "total_train_seconds": actual, "actual_seconds": actual,
                    "delta_seconds": actual - recipe["train_budget_seconds"],
                    "overshoot_seconds": max(0., actual - recipe["train_budget_seconds"]),
                    "crossing_step": budget_crossing_step,
                    "overshoot_updates": max(0, step - budget_crossing_step) if budget_crossing_step is not None else 0,
                    "complete_pass": step % PASS_CASES == 0, "max_steps": recipe["max_steps"]}
        def payload(step, metrics):
            return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                    "step": step, "metrics": metrics, "best_step": best_step, "best_score": best_score,
                    "best_metrics": best_metrics, "first_update_gradient_check": gradient_check,
                    "history": history, "costs": dict(costs), "case_exposure_counts": dict(exposures),
                    "elapsed_seconds": elapsed(), "memory": memory(), "rng": screen._rng_state(torch, np),
                    "case_shuffle_rng_state": _schedule_rng(manifest["train_keys"], seed, step),
                    "next_case_position": step, "case_order_sha256": config["case_order_sha256"],
                    "phase_start_step": phase_start_step, "phase_initial_metrics": phase_initial,
                    "parent_train_seconds": parent_train_seconds, "parent_elapsed_seconds": parent_elapsed,
                    "budget_crossing_step": budget_crossing_step, "budget": budget_record(step),
                    "best_checkpoint_origin": best_checkpoint_origin}
        def evaluate(step, training_loss):
            nonlocal best_step, best_score, best_metrics, phase_initial, best_checkpoint_origin
            evaluation_began = time.monotonic(); model.eval(); status("evaluating", step=step)
            groups = {}
            with torch.no_grad():
                all_keys = manifest["training_readout_keys"] + manifest["validation_keys"]
                for group, keys in (("train_readout", manifest["training_readout_keys"]), ("validation", manifest["validation_keys"])):
                    per_case = {}
                    for key in keys:
                        case = load_case(manifest["records"][key], device, for_fno=method == "fno")
                        screen._sync(torch, device); inference_began = time.monotonic()
                        devices = [device.index if device.index is not None else torch.cuda.current_device()]
                        with torch.random.fork_rng(devices=devices):
                            if method == "transolver_pp":
                                torch.manual_seed(seed + 100000 + all_keys.index(key))
                            encoded = screen._predict_case(torch, model, method, case, query_grid)
                        screen._sync(torch, device)
                        costs["inference_seconds"] += time.monotonic() - inference_began
                        costs["inference_case_exposures"] += 1; costs["inference_node_exposures"] += len(case["xyz"])
                        prediction = scaler.decode(encoded)
                        per_case[key] = screen._physical_metrics(np, case["physical"], prediction)
                        del case, encoded, prediction
                    groups[group] = screen._aggregate(np, per_case)
            costs["evaluation_seconds"] += time.monotonic() - evaluation_began
            score = groups["validation"]["selection_score"]
            metrics = {"step": step, "elapsed_seconds": elapsed(), "phase_elapsed_seconds": elapsed() - parent_elapsed,
                       "training_loss": training_loss, **groups, "selection_score": score,
                       "costs": dict(costs), "budget": budget_record(step), "case_exposure_counts": dict(exposures),
                       "memory": memory(), "first_update_gradient_check": gradient_check}
            if phase_initial is None:
                phase_initial = metrics
            improved = score < best_score
            if improved:
                best_step, best_score, best_metrics = step, score, metrics
                best_checkpoint_origin = {"phase": "confirmation", "checkpoint": str(output / "best.pt"), "step": step}
            history.append(metrics)
            saved_payload = payload(step, metrics)
            screen._checkpoint(torch, output / "last.pt", saved_payload)
            if improved:
                screen._checkpoint(torch, output / "best.pt", saved_payload)
                screen.write_json(output / "metrics_best.json", metrics)
            screen.write_json(output / "metrics_latest.json", metrics)
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False) + "\n")
            status("training", step=step, costs=dict(costs), budget=budget_record(step), memory=memory(),
                   best_step=best_step, selection_score=score, evaluated_steps=[row["step"] for row in history])
            print(f"confirmation {method}/seed{seed} step={step} train_s={costs['train_seconds']:.3f} U={groups['validation']['case_mean_u_relative_l2']:.6g} stress={groups['validation']['case_mean_stress_relative_l2']:.6g} score={score:.6g}", flush=True)
            return metrics
        last = history[-1] if saved is not None else evaluate(0, None)
        if saved is not None:
            screen._checkpoint(torch, output / "last.pt", payload(start_step, last))
            screen.write_json(output / "metrics_latest.json", last)
        state["evaluated_steps"] = [row["step"] for row in history]
        stop_reason = None; step = start_step
        if start_step >= recipe["max_steps"]:
            stop_reason = "max_steps"
        elif costs["train_seconds"] >= recipe["train_budget_seconds"]:
            budget_crossing_step = start_step if budget_crossing_step is None else budget_crossing_step
            stop_reason = "train_budget_seconds"
        while stop_reason is None:
            step += 1; key = order[step - 1]; training_began = time.monotonic()
            case = load_case(manifest["records"][key], device, for_fno=method == "fno")
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss = screen._train_case(torch, model, method, case, query_grid)
            if gradient_check is None:
                gradient_check = {"step": step, "case": key, **screen._gradient_check(torch, model)}
                screen.write_json(output / "first_update_gradient_check.json", gradient_check)
            optimizer.step(); screen._sync(torch, device)
            costs["train_seconds"] += time.monotonic() - training_began
            costs["train_case_exposures"] += 1; costs["train_node_exposures"] += len(case["xyz"])
            exposures[key] += 1; del case
            if budget_crossing_step is None and costs["train_seconds"] >= recipe["train_budget_seconds"]:
                budget_crossing_step = step
            if step % 10 == 0:
                status("training", step=step, case=key, training_loss=loss, costs=dict(costs), budget=budget_record(step), memory=memory())
            if step % PASS_CASES == 0 or step == recipe["max_steps"]:
                last = evaluate(step, loss)
                if step == recipe["max_steps"]:
                    stop_reason = "max_steps"
                elif budget_crossing_step is not None:
                    stop_reason = "train_budget_seconds"
        prefix_counts = Counter(order[:step])
        if exposures != {key: prefix_counts[key] for key in manifest["train_keys"]} or costs["train_case_exposures"] != step:
            raise AssertionError("Confirmation case exposure differs from the exact scheduled prefix")
        summary = {"status": "completed", "method": method, "seed": seed, "endpoint_step": step,
                   "stop_reason": stop_reason, "elapsed_seconds": elapsed(),
                   "phase_elapsed_seconds": elapsed() - parent_elapsed, "parent_elapsed_seconds": parent_elapsed,
                   "config": config, "initial_metrics": history[0], "phase_initial_metrics": phase_initial,
                   "final_metrics": last, "endpoint_metrics": history, "best_step": best_step,
                   "best_metrics": best_metrics, "selection_score": last["selection_score"],
                   "best_checkpoint_origin": best_checkpoint_origin, "budget": budget_record(step),
                   "costs": dict(costs), "memory": memory(), "parameter_counts": model.parameter_counts,
                   "case_exposure_counts": exposures, "first_update_gradient_check": gradient_check,
                   "case_order_sha256": config["case_order_sha256"], "completed_at_utc": screen.utc_now(),
                   "scope": "same TRAIN40/VAL5 x known3; shared cloud_seed2026/scaler; all-node signed4 loss; cumulative train-time confirmation"}
        screen.write_json(output / "summary.json", summary)
        status("completed", step=step, stop_reason=stop_reason, costs=dict(costs), budget=budget_record(step))
        return summary
    except Exception as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if cleanup is not None:
            cleanup()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, required=True)
    result.add_argument("--method", required=True)
    result.add_argument("--seed", type=int, choices=SEEDS)
    result.add_argument("--output", type=Path)
    result.add_argument("--warmstart", type=Path)
    result.add_argument("--resume", nargs="?", const="last")
    result.add_argument("--source-check", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
