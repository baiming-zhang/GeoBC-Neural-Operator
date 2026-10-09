"""Shared full-case GeoBC training for the six solver-selection families.

The source check reads JSON and source identities only. Training never opens
test/dia cases, and every family is supervised at all original FEM nodes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("mlp", "hashgrid", "transolver", "transolver_pp", "ts3", "fno")
ALIASES = {"hash": "hashgrid", "transolver++": "transolver_pp", "transolverpp": "transolver_pp",
           "transolver-3": "ts3", "transolver3": "ts3"}
ENDPOINTS = (0, 120, 240, 480)
CHUNK = 32768
PROTOCOL = {"seed": 2026, "updates": 480, "passes": 4, "train_cases": 120, "validation_cases": 15,
            "optimizer": "Adam", "learning_rate": .001, "weight_decay": 0., "batch_cases": 1,
            "point_chunk": CHUNK, "eval_steps": list(ENDPOINTS),
            "selection_score": "0.5*(validation case-mean displacement vector relative L2 + signed-stress relative L2)",
            "outputs": ["ux_mm", "uy_mm", "uz_mm", "signed_stress_MPa"],
            "loss": "mean encoded squared error across all original case nodes and four channels",
            "training_best_selection": False}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def host_path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", value):
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def read_json(path):
    return json.loads(host_path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_sha(path):
    return hashlib.sha256(host_path(path).read_bytes()).hexdigest()


def canonical_method(value):
    value = ALIASES.get(value.lower(), value.lower())
    if value not in METHODS:
        raise ValueError(f"Unknown solver family {value!r}")
    return value


def case_order(keys, seed):
    """Four balanced passes, independent of model construction/evaluation RNG."""
    if len(keys) != 120 or len(set(keys)) != 120:
        raise ValueError("Expected 120 distinct TRAIN cases")
    sampler = random.Random(seed)
    order = []
    for _ in range(4):
        permutation = list(range(len(keys))); sampler.shuffle(permutation)
        order.extend(permutation)
    return order, sampler.getstate()


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory(torch, device):
    if device.type != "cuda":
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}


def _checkpoint(torch, path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def _rng_state(torch, np):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def _restore_rng(torch, np, state):
    random.setstate(state["python"]); torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _gradient_check(torch, model):
    """The first real optimizer update is the only gradient acceptance step."""
    rows, missing, allowed_missing = [], [], []
    inactive = {"solver.backbone.embedding.weight", "solver.backbone.embedding.bias"} if model.method == "transolver_pp" else set()
    named = list(model.named_parameters())
    for name, parameter in named:
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            (allowed_missing if name in inactive else missing).append(name); continue
        grad = parameter.grad.detach()
        rows.append((name, torch.isfinite(grad).all(), grad.float().square().sum()))
    if not rows:
        raise FloatingPointError("First actual update has no gradients")
    finite = bool(torch.stack([row[1] for row in rows]).all().item())
    total = torch.stack([row[2] for row in rows]).sum()
    encoder = [row[2] for row in rows if row[0].startswith("encoder.")]
    solver = [row[2] for row in rows if row[0].startswith("solver.")]
    norms = {"total": float(total.sqrt().item()),
             "encoder": float(torch.stack(encoder).sum().sqrt().item()) if encoder else 0.,
             "solver": float(torch.stack(solver).sum().sqrt().item()) if solver else 0.}
    passed = finite and not missing and all(math.isfinite(v) and v > 0 for v in norms.values())
    record = {"passed": passed, "all_present_gradients_finite": finite,
              "trainable_tensors": sum(p.requires_grad for _, p in named),
              "gradient_tensors": len(rows), "missing_gradient_names": missing,
              "allowed_missing": allowed_missing,
              "allowed_missing_reason": "Transolver++ condition=None; all 60 conditioning channels already enter the main input, so the separate conditional embedding is inactive" if allowed_missing else None,
              "L2_gradient_norms": norms, "source": "first actual full-case loss backward; no probe or warmup"}
    if not passed:
        raise FloatingPointError("First actual full-case gradient check failed: " + json.dumps(record))
    return record


def _physical_metrics(np, truth, prediction):
    truth = np.asarray(truth, dtype=np.float64); prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 4 or not len(truth):
        raise ValueError("Physical fields must be matching nonempty [N,4]")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite physical output")
    error = prediction - truth
    u_energy = float(np.square(truth[:, :3]).sum()); s_energy = float(np.square(truth[:, 3]).sum())
    if u_energy <= 0 or s_energy <= 0:
        raise ValueError("A selection case has a zero physical reference norm")
    hotspot_count = max(1, math.ceil(.01 * len(truth)))
    hotspot = np.argpartition(np.abs(truth[:, 3]), len(truth) - hotspot_count)[-hotspot_count:]
    hotspot_energy = float(np.square(truth[hotspot, 3]).sum())
    return {"nodes": len(truth),
            "u_relative_l2": math.sqrt(float(np.square(error[:, :3]).sum()) / u_energy),
            "u_vector_error_RMS_mm": math.sqrt(float(np.square(error[:, :3]).sum()) / len(truth)),
            "u_max_vector_error_mm": float(np.linalg.norm(error[:, :3], axis=1).max()),
            "stress_relative_l2": math.sqrt(float(np.square(error[:, 3]).sum()) / s_energy),
            "stress_RMSE_MPa": math.sqrt(float(np.square(error[:, 3]).sum()) / len(truth)),
            "stress_MAE_MPa": float(np.abs(error[:, 3]).mean()),
            "stress_hotspot_relative_l2": math.sqrt(float(np.square(error[hotspot, 3]).sum()) / hotspot_energy),
            "stress_hotspot_nodes": hotspot_count}


def _aggregate(np, per_case):
    u = [item["u_relative_l2"] for item in per_case.values()]
    stress = [item["stress_relative_l2"] for item in per_case.values()]
    return {"cases": len(per_case), "nodes": sum(item["nodes"] for item in per_case.values()),
            "case_mean_u_relative_l2": float(np.mean(u)), "worst_case_u_relative_l2": max(u),
            "case_mean_stress_relative_l2": float(np.mean(stress)), "worst_case_stress_relative_l2": max(stress),
            "case_mean_stress_hotspot_relative_l2": float(np.mean([item["stress_hotspot_relative_l2"] for item in per_case.values()])),
            "selection_score": .5 * (float(np.mean(u)) + float(np.mean(stress))), "per_case": per_case}


def _train_case(torch, model, method, case, query_grid):
    context = model.encode_geometry(case["cloud"], case["mask"])
    target = case["target"]
    count = len(case["xyz"])
    if target.shape != (count, 4):
        raise ValueError("Encoded target shape must be [N,4]")
    if method in ("mlp", "hashgrid"):
        losses = []
        for start in range(0, count, CHUNK):
            stop = min(start + CHUNK, count)
            prediction = model.forward_points(case["xyz"][start:stop], case["features"][start:stop], context)
            loss = (prediction.float() - target[start:stop]).square().mean() * ((stop - start) / count)
            loss.backward(retain_graph=stop < count)
            losses.append(loss.detach())
        total_loss = float(torch.stack(losses).sum().item())
        if not math.isfinite(total_loss):
            raise FloatingPointError("Nonfinite full-case chunk loss")
        return total_loss
    if method == "fno":
        grid = model.forward_grid(case["grid"], context)
        prediction = query_grid(grid, case)
    else:
        prediction = model.forward_points(case["xyz"], case["features"], context)
    if prediction.shape != target.shape:
        raise ValueError("Full-case model output does not match [N,4] target")
    loss = (prediction.float() - target).square().mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("Nonfinite full-case loss")
    loss.backward()
    return float(loss.detach().item())


def _predict_case(torch, model, method, case, query_grid):
    context = model.encode_geometry(case["cloud"], case["mask"])
    if method == "fno":
        grid = model.forward_grid(case["grid"], context)
        return query_grid(grid, case).float().cpu().numpy()
    if method in ("mlp", "hashgrid"):
        return torch.cat([model.forward_points(case["xyz"][start:start + CHUNK], case["features"][start:start + CHUNK], context).float()
                          for start in range(0, len(case["xyz"]), CHUNK)]).cpu().numpy()
    return model.forward_points(case["xyz"], case["features"], context).float().cpu().numpy()


def _source_identities():
    paths = ("geobc/solver_selection_train.py", "geobc/solver_selection_models.py",
             "geobc/solver_selection_data.py", "geobc/benchmark.py", "geobc/conditioned_field.py",
             "geobc/tcnn_fields.py", "geobc/transformers.py", "geobc/fno.py",
             "third_party/transformer_sources.json")
    return {path: source_sha(path) for path in paths}


def request_contract(plan_path, method):
    """Completed reuse checks existing cache JSON/stat; never prepares arrays."""
    from .solver_selection_data import read_cache_metadata
    path = host_path(plan_path); plan = read_json(path)
    _check_recipe(plan)
    manifest = read_cache_metadata(plan)
    method = canonical_method(method)
    _check_manifest(manifest)
    recipe_identity = {key: plan[key] for key in ("recipe", "existing_split", "prepared_data", "data_cache")}
    contract = {"plan": str(path), "recipe": plan["recipe"],
                "recipe_sha256": hashlib.sha256(json.dumps(recipe_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "method": method,
                "protocol": PROTOCOL, "cache_contract": manifest["contract"],
                "case_order_sha256": manifest["case_order_sha256"], "scaler": manifest["scaler"],
                "train_keys": manifest["train_keys"], "validation_keys": manifest["validation_keys"],
                "training_readout_keys": manifest["training_readout_keys"], "source_sha256": _source_identities()}
    return contract, manifest


def contracts_compatible(previous, current):
    """Only a receipt-authorized trainer SHA substitution can bridge contracts."""
    if previous == current:
        return True
    try:
        fix = read_json("notes/solver_selection_status_fix.json")
        key = "geobc/solver_selection_train.py"
        old_sha, new_sha = previous["source_sha256"][key], current["source_sha256"][key]
        if old_sha not in fix["compatible_trainer_sha256"] or new_sha != fix["new_trainer_sha256"]:
            return False
        substituted = {**previous, "source_sha256": {**previous["source_sha256"], key: new_sha}}
        return substituted == current
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _check_recipe(plan):
    expected = {"methods": list(METHODS), "train_loads": ["ver", "hor", "tor"],
                "seed": 2026, "cloud_seed": 2026, "device": "cuda", "steps": 480, "passes": 4,
                "eval_steps": list(ENDPOINTS), "learning_rate": .001, "weight_decay": 0.,
                "optimizer": "Adam", "cloud_points": 1024, "geometry_dim": 32,
                "grid_resolution": 64, "microbatch_nodes": CHUNK, "eval_batch_size": CHUNK,
                "stress_transform": "asinh", "channels": [0, 1, 2, 3],
                "training_readout_keys": ["101_479/ver", "107_320/hor", "10_58/tor"]}
    if any(plan["recipe"].get(key) != value for key, value in expected.items()):
        raise ValueError("Plan differs from the fixed shared six-family screening recipe")


def _check_manifest(manifest):
    if manifest["status"] != "completed":
        raise ValueError("Shared selection cache is not complete; trainer cannot prepare it")
    train = manifest["train_keys"]; validation = manifest["validation_keys"]
    if len(train) != 120 or len(set(train)) != 120 or len(validation) != 15 or len(set(validation)) != 15:
        raise ValueError("Expected TRAIN120 and validation15")
    train_cads = {key.split("/")[0] for key in train}; val_cads = {key.split("/")[0] for key in validation}
    if len(train_cads) != 40 or len(val_cads) != 5 or train_cads & val_cads:
        raise ValueError("Expected distinct whole-CAD 40/5 split")
    for keys, cads in ((train, train_cads), (validation, val_cads)):
        if set(keys) != {f"{cad}/{load}" for cad in cads for load in ("ver", "hor", "tor")}:
            raise ValueError("Only known-load ver/hor/tor TRAIN/validation cases are permitted")
    schedule, _ = case_order(train, PROTOCOL["seed"])
    if manifest["case_order"] != [train[index] for index in schedule]:
        raise ValueError("Shared four-pass schedule differs from Python Random(seed2026)")
    readout = manifest["training_readout_keys"]
    if readout != ["101_479/ver", "107_320/hor", "10_58/tor"] or not set(readout).issubset(train):
        raise ValueError("Training readout must be the three fixed cases")
    if manifest["scaler"]["channels"] != [0, 1, 2, 3]:
        raise ValueError("All four direct output channels require the shared train-only scaler")
    if set(manifest["records"]) != set(train + validation):
        raise ValueError("Selection cache records must contain only TRAIN/validation known loads")


def source_check(plan_path):
    """One standard-library schema/source check; no cache preparation or CUDA."""
    import ast
    plan = read_json(plan_path)
    _check_recipe(plan)
    required = {"geobc/solver_selection_models.py": {"build_selection_model", "close_selection_process_group"},
                "geobc/solver_selection_data.py": {"read_cache_metadata", "load_case", "query_grid", "cache_contract"}}
    for path, functions in required.items():
        tree = ast.parse(host_path(path).read_text(encoding="utf-8"), filename=path)
        declared = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if not functions.issubset(declared):
            raise ValueError(f"Missing adapter APIs in {path}: {sorted(functions - declared)}")
    compile(host_path("geobc/solver_selection_train.py").read_text(encoding="utf-8"), "solver_selection_train.py", "exec")
    return {"status": "source_ready", "plan_sha256": source_sha(plan_path), "plan_keys": list(plan),
            "protocol": PROTOCOL, "source_sha256": _source_identities(),
            "scientific_arrays_read": 0, "models_constructed": 0, "CUDA_initialized": False,
            "actual_gradient_check": "pending first real update", "cache_preparation": "never performed by trainer"}


def run(args):
    method = canonical_method(args.method)
    if args.source_check:
        result = source_check(args.plan); print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    if args.output is None:
        raise ValueError("--output is required for an actual run")
    began = time.monotonic(); output = host_path(args.output)
    prior_elapsed = 0.
    contract, manifest = request_contract(args.plan, method)
    if (output / "summary.json").is_file():
        stored = read_json(output / "summary.json")
        if stored.get("status") == "completed":
            if not contracts_compatible(stored["config"]["contract"], contract) or read_json(output / "config.json") != stored["config"]:
                raise ValueError("Completed solver selection contract differs")
            print(f"Reusing completed {method}: {output}", flush=True)
            return stored
    resume = None
    if args.resume is not None:
        resume = output / "last.pt" if args.resume == "last" else host_path(args.resume)
        if not resume.is_file():
            raise FileNotFoundError(resume)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError("Incomplete nonempty solver output requires explicit --resume")
    output.mkdir(parents=True, exist_ok=True)
    state = {"method": method, "started_at_utc": utc_now(), "step": 0, "total_steps": 480,
             "evaluated_steps": []}
    def status(phase, **details):
        state.update(phase=phase, updated_at_utc=utc_now(), elapsed_seconds=prior_elapsed + time.monotonic() - began, **details)
        actual_costs = state.get("costs", {})
        completed_cases = actual_costs.get("train_case_exposures", 0)
        evaluated = len(state["evaluated_steps"])
        eta = None
        if completed_cases > 0:
            eta = actual_costs["train_seconds"] / completed_cases * max(0, 480 - completed_cases)
            if evaluated:
                eta += actual_costs["evaluation_seconds"] / evaluated * (len(ENDPOINTS) - evaluated)
        if phase == "completed": eta = 0.
        if phase == "failed": eta = None
        state["eta_seconds"] = eta
        state["eta_definition"] = "observed train seconds/case times remaining cases + observed mean full evaluation seconds times pending endpoints"
        state["estimated_completion_utc"] = (datetime.now(timezone.utc) + timedelta(seconds=eta)).isoformat() if eta is not None else None
        write_json(output / "status.json", state)
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
        seed = PROTOCOL["seed"]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        model = build_selection_model(method, device=device, seed=seed)
        scaler = FieldScaler(**manifest["scaler"])
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0, fused=True)
        _, shuffle_rng_state = case_order(manifest["train_keys"], seed)
        config = {"contract": contract, "method": method, "protocol": PROTOCOL,
                  "scaler": manifest["scaler"], "model": model.source_config,
                  "parameter_counts": model.parameter_counts, "case_order": manifest["case_order"],
                  "case_order_sha256": manifest["case_order_sha256"], "cache_contract": manifest["contract"],
                  "train_keys": manifest["train_keys"], "validation_keys": manifest["validation_keys"],
                  "training_readout_keys": manifest["training_readout_keys"],
                  "precision": "float32 Torch and PointNet; TF32 enabled; native Hash FP16 compute/FP32 master; no autocast",
                  "evaluation": "all original nodes for fixed train3 and validation15; ++ case-seeded fork_rng restores training RNG",
                  "geometry_context": "common1024 xyz cloud -> PointNet32; one differentiable encode per case/update",
                  "point_chunks": "MLP/Hash loss weighted by chunk node count and shared-context backward retained until final chunk",
                  "FNO_query": "standard trilinear encoded grid -> every original node; no grid-label loss or coverage exclusions",
                  "checkpoint_selection": PROTOCOL["selection_score"], "torch_version": torch.__version__}
        start_step, best_step, best_score = 0, None, math.inf
        gradient_check = None; best_metrics = None; history = []
        costs = {"train_seconds": 0., "inference_seconds": 0., "evaluation_seconds": 0.,
                 "train_case_exposures": 0, "train_node_exposures": 0,
                 "inference_case_exposures": 0, "inference_node_exposures": 0}
        exposures = {key: 0 for key in manifest["train_keys"]}
        prior_peak = {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
        if resume is not None:
            saved = torch.load(resume, map_location=device, weights_only=False)
            if (not contracts_compatible(saved["config"]["contract"], config["contract"])
                    or {k: v for k, v in saved["config"].items() if k != "contract"}
                    != {k: v for k, v in config.items() if k != "contract"}):
                raise ValueError("Resume data/model/protocol/scaler contract differs")
            start_step = saved["step"]
            if start_step not in ENDPOINTS:
                raise ValueError("Resume requires a committed evaluation endpoint")
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
            _restore_rng(torch, np, saved["rng"])
            shuffle_rng_state = saved["case_shuffle_rng_state"]
            best_step, best_score, best_metrics = saved["best_step"], saved["best_score"], saved["best_metrics"]
            gradient_check, history = saved["first_update_gradient_check"], saved["history"]
            costs, exposures = saved["costs"], saved["case_exposure_counts"]
            prior_elapsed = saved["elapsed_seconds"]
            prior_peak = saved["memory"]
            if saved["next_case_position"] != start_step or saved["case_order_sha256"] != manifest["case_order_sha256"]:
                raise ValueError("Resume case schedule position differs")
            if best_step == start_step and not (output / "best.pt").is_file():
                _checkpoint(torch, output / "best.pt", saved)
            if best_metrics is not None:
                write_json(output / "metrics_best.json", best_metrics)
            state["evaluated_steps"] = [row["step"] for row in history]
        write_json(output / "config.json", config)
        if resume is not None:
            # Checkpoint history is authoritative after a partial interrupted append.
            (output / "metrics.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in history), encoding="utf-8")
        torch.cuda.reset_peak_memory_stats(device)
        def elapsed():
            return prior_elapsed + time.monotonic() - began
        def memory():
            actual = _memory(torch, device)
            return {key: max(actual[key], prior_peak[key]) for key in actual}
        def payload(step, metrics):
            return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                    "step": step, "metrics": metrics, "best_step": best_step, "best_score": best_score,
                    "best_metrics": best_metrics, "first_update_gradient_check": gradient_check,
                    "history": history, "costs": dict(costs), "case_exposure_counts": dict(exposures),
                    "elapsed_seconds": elapsed(), "memory": memory(), "rng": _rng_state(torch, np),
                    "case_shuffle_rng_state": shuffle_rng_state,
                    "next_case_position": step, "case_order_sha256": manifest["case_order_sha256"]}
        def evaluate(step, training_loss):
            nonlocal best_step, best_score, best_metrics
            evaluation_began = time.monotonic(); model.eval(); status("evaluating", step=step)
            groups = {}
            with torch.no_grad():
                for group, keys in (("train_readout", manifest["training_readout_keys"]), ("validation", manifest["validation_keys"])):
                    per_case = {}
                    for key in keys:
                        case = load_case(manifest["records"][key], device, for_fno=method == "fno")
                        _sync(torch, device); inference_began = time.monotonic()
                        devices = [device.index if device.index is not None else torch.cuda.current_device()]
                        with torch.random.fork_rng(devices=devices):
                            if method == "transolver_pp":
                                eval_seed = seed + 100000 + (manifest["training_readout_keys"] + manifest["validation_keys"]).index(key)
                                torch.manual_seed(eval_seed)
                            encoded = _predict_case(torch, model, method, case, query_grid)
                        _sync(torch, device)
                        costs["inference_seconds"] += time.monotonic() - inference_began
                        costs["inference_case_exposures"] += 1; costs["inference_node_exposures"] += len(case["xyz"])
                        prediction = scaler.decode(encoded)
                        per_case[key] = _physical_metrics(np, case["physical"], prediction)
                        del case, encoded, prediction
                    groups[group] = _aggregate(np, per_case)
            costs["evaluation_seconds"] += time.monotonic() - evaluation_began
            score = groups["validation"]["selection_score"]
            metrics = {"step": step, "elapsed_seconds": elapsed(), "training_loss": training_loss,
                       **groups, "selection_score": score, "costs": dict(costs),
                       "case_exposure_counts": dict(exposures), "memory": memory(),
                       "first_update_gradient_check": gradient_check}
            improved = score < best_score
            if improved:
                best_step, best_score, best_metrics = step, score, metrics
            history.append(metrics)
            saved_payload = payload(step, metrics)
            _checkpoint(torch, output / "last.pt", saved_payload)
            _checkpoint(torch, output / f"step{step}.pt", saved_payload)
            if improved:
                _checkpoint(torch, output / "best.pt", saved_payload); write_json(output / "metrics_best.json", metrics)
            write_json(output / "metrics_latest.json", metrics)
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False) + "\n")
            status("training", step=step, best_step=best_step, selection_score=score,
                   costs=dict(costs), memory=memory(), evaluated_steps=[row["step"] for row in history])
            print(f"{method} {step}/480 validation U={groups['validation']['case_mean_u_relative_l2']:.6g} stress={groups['validation']['case_mean_stress_relative_l2']:.6g} score={score:.6g}", flush=True)
            return metrics
        last = history[-1] if resume is not None else evaluate(0, None)
        for step in range(start_step + 1, 481):
            key = manifest["case_order"][step - 1]
            training_began = time.monotonic()
            case = load_case(manifest["records"][key], device, for_fno=method == "fno")
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss = _train_case(torch, model, method, case, query_grid)
            if gradient_check is None:
                gradient_check = {"step": step, "case": key, **_gradient_check(torch, model)}
                write_json(output / "first_update_gradient_check.json", gradient_check)
            optimizer.step(); _sync(torch, device)
            costs["train_seconds"] += time.monotonic() - training_began
            costs["train_case_exposures"] += 1; costs["train_node_exposures"] += len(case["xyz"])
            exposures[key] += 1
            del case
            if step % 10 == 0:
                status("training", step=step, case=key, training_loss=loss, costs=dict(costs), memory=memory())
            if step in ENDPOINTS:
                last = evaluate(step, loss)
        if any(count != 4 for count in exposures.values()) or costs["train_case_exposures"] != 480:
            raise AssertionError("Full four-pass case exposure contract was not completed")
        summary = {"status": "completed", "method": method, "endpoint_step": 480,
                   "elapsed_seconds": elapsed(), "config": config, "initial_metrics": history[0],
                   "final_metrics": last, "endpoint_metrics": history, "best_step": best_step,
                   "best_metrics": best_metrics, "selection_score": last["selection_score"],
                   "costs": dict(costs), "memory": memory(), "parameter_counts": model.parameter_counts,
                   "case_exposure_counts": exposures, "first_update_gradient_check": gradient_check,
                   "case_order_sha256": manifest["case_order_sha256"], "completed_at_utc": utc_now(),
                   "scope": "TRAIN40/validation5 x known3; direct signed4; no test/dia, no pretrained teacher or residual field"}
        write_json(output / "summary.json", summary); status("completed", step=480)
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
    result.add_argument("--output", type=Path)
    result.add_argument("--resume", nargs="?", const="last")
    result.add_argument("--source-check", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
