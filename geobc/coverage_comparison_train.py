"""Frozen 40/200-CAD, five-family exposure comparison.

Import and request_contract are standard-library/metadata only.  Actual stages
require the parent's release.  New DEV labels are unavailable until every
predeclared endpoint has an immutable prediction or terminal-failure record.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

ROOT = Path(__file__).resolve().parents[1]
OWN = "geobc/coverage_comparison_train.py"
KIND = "geobc_solver_coverage40vs200_transolver_all_versions"
METHODS = ("transolver", "transolver_pp", "ts3", "fno", "mlp")
STEPS = (0, 1200, 2400, 4800)
CHUNK = 32768
DECISION = "release_frozen_coverage_comparison"
SOURCES = (OWN, "geobc/coverage_comparison_data.py", "geobc/solver_selection_models.py",
           "geobc/solver_selection_train.py", "geobc/solver_selection_data.py",
           "geobc/conditioned_field.py", "geobc/transformers.py", "geobc/fno.py",
           "geobc/benchmark.py", "third_party/transformer_sources.json")


def utc():
    return datetime.now(timezone.utc).isoformat()


def path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", value):
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    p = Path(value).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def read(filename):
    return json.loads(path(filename).read_text(encoding="utf-8-sig"))


def write(filename, value):
    filename = path(filename); filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(filename.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, filename)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(filename):
    h = hashlib.sha256()
    with path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arms(plan):
    rows = plan["scientific"]["arm_order"]
    require(len(rows) == 10 and len({r["id"] for r in rows}) == 10, "Expected ten distinct frozen arms")
    require({(r["method"], r["train_CADs"]) for r in rows} == {(m, n) for m in METHODS for n in (40, 200)}, "Five-family/two-scale inventory differs")
    return rows


def arm_record(plan, method, train_cads):
    rows = [r for r in arms(plan) if r["method"] == method and r["train_CADs"] == train_cads]
    require(len(rows) == 1, "Unknown frozen arm")
    return rows[0]


def arm_output(plan, arm):
    return path(plan["training_output"]) / arm["id"]


def case_schedule(keys):
    require(len(keys) in (120, 600) and len(set(keys)) == len(keys), "TRAIN inventory differs")
    rng = random.Random(2026); order = []
    for _ in range(4800 // len(keys)):
        rows = list(keys); rng.shuffle(rows); order.extend(rows)
    require(len(order) == 4800, "Full-pass schedule differs")
    return order


def request_contract(plan_path, method=None, train_cads=None, stage="train"):
    """Only source, JSON and stat; no array/checkpoint/model imports or reads."""
    plan = read(plan_path); science = plan["scientific"]; recipe = science["recipe"]
    require(plan["kind"] == KIND and digest(science) == plan["scientific_digest"], "Frozen science identity differs")
    expected = {"methods": list(METHODS), "train_loads": ["ver", "hor", "tor"], "seed": 2026,
                "fixed_updates": 4800, "evaluation_updates": list(STEPS), "learning_rate": .001,
                "weight_decay": 0., "optimizer": "Adam", "cloud_points": 1024,
                "geometry_dim": 32, "grid_resolution": 64, "microbatch_nodes": CHUNK,
                "eval_batch_size": CHUNK, "train_budget_seconds_each_arm": 7200,
                "fresh_initialization": True, "warmstart_previous_models": False,
                "channels": [0, 1, 2, 3], "stress_transform": "asinh"}
    require(all(recipe.get(k) == v for k, v in expected.items()), "Fixed exposure/configuration differs")
    require(stage in ("train", "predict", "evaluate"), "Stage differs")
    arm = None if stage == "evaluate" else arm_record(plan, method, train_cads)
    from . import coverage_comparison_data as data
    bootstrap = stage == "train" and train_cads == 40
    if bootstrap:
        # This identity remains independent of acquisition/full-cache status
        # for the entire stage, including later checkpoint validation.
        input_contract, _ = data.request_bootstrap_contract(plan_path)
        manifest = data.read_bootstrap_TRAIN40(plan_path)
    else:
        input_contract, _ = data.request_contract(plan_path)
        manifest = data.read_cache_metadata(plan_path, contract=input_contract)
    require(manifest["status"] == "completed" and manifest["contract"] == input_contract,
            "Input preparation incomplete or incompatible")
    require(manifest.get("feasibility_pass") is True, "Negative input readiness is terminal before models/CUDA")
    require(manifest["scaler"] == science["scaler"]["values"], "Frozen original TRAIN40 scaler differs")
    require(len(manifest["validation_keys"]) == 60 and len(set(manifest["validation_keys"])) == 60, "Fixed DEV60 differs")
    if bootstrap:
        require(set(manifest["train_keys"]) == {"40"} and len(manifest["train_keys"]["40"]) == 120,
                "Bootstrap must contain only the frozen original TRAIN120")
    else:
        require([len(manifest["train_keys"][str(n)]) for n in (40, 200)] == [120, 600], "TRAIN scales differ")
    pinned = plan.get("source_sha256", {})
    sources = {name: sha(name) for name in sorted(set(SOURCES) | set(pinned))}
    require(all(pinned.get(name, actual) == actual for name, actual in sources.items()), "Frozen source changed")
    order = None if arm is None else case_schedule(manifest["train_keys"][str(arm["train_CADs"])])
    contract = {"schema": "geobc_coverage_comparison_v1", "kind": KIND, "stage": stage,
                "scientific_digest": plan["scientific_digest"], "plan_science": science,
                "arm": arm, "input_contract": input_contract, "input_contract_sha256": digest(input_contract),
                "source_sha256": sources, "scaler": manifest["scaler"],
                "input_phase": "train40_bootstrap" if bootstrap else "complete_shared660",
                "validation_keys": manifest["validation_keys"],
                "case_order_sha256": None if order is None else digest(order)}
    return contract, plan


def release(contract, plan):
    review = plan["root_review"]
    marker_path = review.get("path", "notes/geobc_coverage_root_review.json") if isinstance(review, dict) else review
    marker = read(marker_path)
    require(marker.get("passed") is True and marker.get("preparation_and_training_released") is True
            and marker.get("scientific_digest", marker.get("science_digest")) == contract["scientific_digest"]
            and marker.get("source_pins") == contract["source_sha256"], "Authentic root release missing or source identity differs")


@contextmanager
def lock(filename):
    filename = path(filename); filename.parent.mkdir(parents=True, exist_ok=True)
    handle = filename.open("a+")
    try:
        if os.name != "nt":
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(" "); handle.flush()
            handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        handle.seek(0); handle.truncate(); handle.write(json.dumps({"pid": os.getpid(), "UTC": utc()})); handle.flush()
        yield
    finally:
        handle.close()


class TrainClock:
    """Cumulative complete-update clock; evaluation/checkpoint I/O are separate.

An interrupted open update is charged conservatively, never granted a new
7200s allowance.  Model/cursor recovery comes exclusively from last.pt.
"""
    def __init__(self, output):
        self.file = output / "clock.json"
        self.data = read(self.file) if self.file.exists() else {"train_seconds": 0., "unknown_interruption_charge_seconds": 0.,
            "attempts": 0, "active_update": False, "last_epoch": None}
        if self.data["active_update"]:
            gap = max(0., time.time() - self.data["last_epoch"])
            self.data["train_seconds"] += gap; self.data["unknown_interruption_charge_seconds"] += gap
        self.data["attempts"] += 1; self.data["active_update"] = False; self.started = None
        write(self.file, self.data)

    def begin(self):
        self.data.update(active_update=True, last_epoch=time.time())
        write(self.file, self.data); self.started = time.monotonic()

    def finish(self):
        if self.started is not None:
            seconds = time.monotonic() - self.started
            self.data["train_seconds"] += seconds; self.started = None
        else:
            seconds = 0.
        self.data.update(active_update=False, last_epoch=time.time()); write(self.file, self.data)
        return seconds


def save_checkpoint(torch, filename, payload):
    temporary = filename.with_name(filename.name + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, filename)


def initial_identity(model):
    h = hashlib.sha256(); rows = {}
    for name, parameter in model.named_parameters():
        raw = parameter.detach().cpu().contiguous().numpy().tobytes()
        record = {"shape": list(parameter.shape), "dtype": str(parameter.dtype), "numel": parameter.numel(), "sha256": hashlib.sha256(raw).hexdigest()}
        rows[name] = record; h.update(name.encode()); h.update(digest(record).encode())
    return {"all_named_trainable_sha256": h.hexdigest(), "parameters": rows}


def model_environment(manifest, method):
    import numpy as np
    import torch
    from .solver_selection_models import build_selection_model, close_selection_process_group
    from .solver_selection_train import _rng_state, _restore_rng, _gradient_check, _predict_case
    from .solver_selection_data import query_grid
    from .benchmark import FieldScaler
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
    torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True; torch.cuda.init(); torch.cuda.reset_peak_memory_stats()
    model = build_selection_model(method, "cuda", 2026)
    return np, torch, model, FieldScaler(**manifest["scaler"]), close_selection_process_group, _rng_state, _restore_rng, _gradient_check, _predict_case, query_grid


def train_case(torch, model, case, query_grid):
    count = len(case["xyz"]); target = case["target"]
    require(target.shape == (count, 4), "TRAIN encoded target shape differs")
    context = model.encode_geometry(case["cloud"], case["mask"])
    if model.method == "mlp":
        # Exact chain rule: chunk heads accumulate on a context leaf, followed
        # by ONE original encoder backward, with no retained per-chunk graph.
        leaf = context.detach().requires_grad_(True); loss_total = 0.
        for start in range(0, count, CHUNK):
            stop = min(count, start + CHUNK)
            prediction = model.forward_points(case["xyz"][start:stop], case["features"][start:stop], leaf)
            loss = (prediction.float() - target[start:stop]).square().mean() * ((stop - start) / count)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Nonfinite complete-case chunk loss")
            loss.backward(); loss_total += float(loss.detach())
        require(leaf.grad is not None, "Context leaf gradient missing")
        if not bool(torch.isfinite(leaf.grad).all()):
            raise FloatingPointError("Nonfinite context leaf gradient")
        context.backward(leaf.grad); return loss_total
    prediction = query_grid(model.forward_grid(case["grid"], context), case) if model.method == "fno" else model.forward_points(case["xyz"], case["features"], context)
    require(prediction.shape == target.shape, "Full original-node output differs")
    loss = (prediction.float() - target).square().mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("Nonfinite complete-case loss")
    loss.backward(); return float(loss.detach())


def run_train(plan_path, method, train_cads, output=None):
    contract, plan = request_contract(plan_path, method, train_cads, "train"); release(contract, plan)
    from . import coverage_comparison_data as data
    manifest = data.read_bootstrap_TRAIN40(plan_path) if train_cads == 40 else data.read_cache_metadata(plan_path, contract=contract["input_contract"])
    base = path(plan["training_output"])
    require(not (base / "prediction_barrier.json").exists() and not (base / "evaluation" / "summary.json").exists()
            and not (path(plan["data_cache"]) / "DEV_targets" / "manifest.json").exists(),
            "Immutable endpoint readout has closed this stage to all further training")
    arm = contract["arm"]; output = path(output) if output else arm_output(plan, arm); output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists() and read(summary_path).get("status") == "completed":
        saved = read(summary_path); require(saved["config"]["contract"] == contract, "Completed arm contract differs"); return saved
    began = time.monotonic(); clock = TrainClock(output); cleanup = None
    train_keys = manifest["train_keys"][str(train_cads)]; order = case_schedule(train_keys)
    state = {"status": "running", "kind": KIND, "arm_id": arm["id"], "method": method,
             "train_CADs": train_cads, "step": 0, "total_steps": 4800, "phase": "initialization", "updated_at_utc": utc()}
    write(output / "status.json", state)
    costs = {"train_seconds": clock.data["train_seconds"], "initialization_seconds": 0., "checkpoint_seconds": 0.,
             "TRAIN_readout_seconds": 0., "train_case_exposures_committed": 0, "train_node_exposures_committed": 0}
    exposures = {key: 0 for key in train_keys}; step = 0; first_grad = None; history = []; last_loss = None
    reason = None; config = None; peak_prior = {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    try:
        np, torch, model, scaler, cleanup, rng_state, restore_rng, grad_check, predict, query_grid = model_environment(manifest, method)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0., fused=True)
        config = {"contract": contract, "method": method, "train_CADs": train_cads, "arm_id": arm["id"],
                  "model": model.source_config, "parameter_counts": model.parameter_counts,
                  "scaler": manifest["scaler"], "case_order_sha256": digest(order),
                  "initial_parameters": initial_identity(model), "precision": "FP32 Torch/high matmul/TF32 enabled/no autocast",
                  "training": "complete-case encoded4 mean MSE; one Adam update; no DEV model selection",
                  "initialization": "fresh seed2026; identical constructor and initialization across scales within each family; capacities differ across families"}
        if (output / "config.json").exists():
            require(digest(read(output / "config.json")) == digest(config), "Resume initial/config identity differs")
        write(output / "config.json", config)
        costs["initialization_seconds"] = time.monotonic() - began
        if (output / "last.pt").exists():
            saved = torch.load(output / "last.pt", map_location="cuda", weights_only=False)
            require(digest(saved["config"]) == digest(config), "Resume model/scaler/source/input/initial parameters differs")
            step = saved["step"]; require(0 <= step <= 4800 and saved["next_case_position"] == step, "Resume cursor differs")
            model.load_state_dict(saved["model"], strict=True); optimizer.load_state_dict(saved["optimizer"])
            restore_rng(torch, np, saved["rng"])
            exposures, first_grad, history = saved["case_exposure_counts"], saved["first_update_gradient_check"], saved["history"]
            fresh_initialization = time.monotonic() - began
            costs = saved["costs"]
            measured = read(output / "costs.json") if (output / "costs.json").exists() else {}
            for name in ("initialization_seconds", "checkpoint_seconds", "TRAIN_readout_seconds"):
                costs[name] = max(float(costs[name]), float(measured.get(name, 0.)))
            costs["initialization_seconds"] += fresh_initialization
            costs["train_seconds"] = max(float(costs["train_seconds"]), clock.data["train_seconds"])
            clock.data["train_seconds"] = costs["train_seconds"]
            peak_prior = saved["memory"]; last_loss = saved["last_training_loss"]
            require(sum(exposures.values()) == step and costs["train_case_exposures_committed"] == step, "Committed exposure/cursor differs")
        def memory():
            return {"peak_allocated_bytes": max(peak_prior["peak_allocated_bytes"], int(torch.cuda.max_memory_allocated())),
                    "peak_reserved_bytes": max(peak_prior["peak_reserved_bytes"], int(torch.cuda.max_memory_reserved()))}
        def payload():
            return {"config": config, "step": step, "next_case_position": step, "case_order_sha256": digest(order),
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state(torch, np),
                    "case_exposure_counts": dict(exposures), "costs": dict(costs), "memory": memory(),
                    "first_update_gradient_check": first_grad, "history": history, "last_training_loss": last_loss}
        def commit(endpoint=False):
            t = time.monotonic(); saved = payload(); save_checkpoint(torch, output / "last.pt", saved)
            if endpoint:
                save_checkpoint(torch, output / f"step{step}.pt", saved)
                write(output / f"step{step}.json", {"status": "completed", "config": config, "step": step,
                      "checkpoint": str(output / f"step{step}.pt"), "case_exposure_counts": dict(exposures), "costs": dict(costs)})
            costs["checkpoint_seconds"] += time.monotonic() - t
            write(output / "costs.json", costs)
        def readout_train():
            t = time.monotonic(); per = {}; original_rng = rng_state(torch, np); model.eval()
            try:
                with torch.no_grad():
                    for key in plan["scientific"]["recipe"]["training_readout_keys"]:
                        case = data.load_case(manifest, key, with_targets=True, device="cuda", for_fno=method == "fno")
                        torch.manual_seed(2026 + int(hashlib.sha256(f"TRAIN:{step}:{key}".encode()).hexdigest()[:8], 16))
                        physical = scaler.decode(predict(torch, model, method, case, query_grid))
                        per[key] = physical_metrics(np, case["physical"], physical)
                        del case, physical
                torch.cuda.synchronize(); costs["TRAIN_readout_seconds"] += time.monotonic() - t
            finally:
                restore_rng(torch, np, original_rng)
            row = {"step": step, "training_loss_last_case": last_loss, "TRAIN_readout": aggregate(per, 3), "costs": dict(costs)}
            history.append(row); write(output / "train_readout.json", history)
            write(output / "costs.json", costs)
        if step == 0 and not (output / "step0.pt").exists():
            readout_train(); commit(True)
        while step < 4800:
            if clock.data["train_seconds"] >= 7200:
                reason = "incomplete_cost_limit"; break
            key = order[step]; state.update(phase="training", step=step, case_key=key, train_seconds=clock.data["train_seconds"], updated_at_utc=utc())
            if step % 10 == 0:
                write(output / "status.json", state)
            clock.begin()
            try:
                case = data.load_case(manifest, key, with_targets=True, device="cuda", for_fno=method == "fno")
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss = train_case(torch, model, case, query_grid)
                if first_grad is None:
                    first_grad = {"step": step + 1, "case": key, **grad_check(torch, model)}
                    write(output / "first_update_gradient_check.json", first_grad)
                optimizer.step(); torch.cuda.synchronize()
                node_count = len(case["xyz"])
            finally:
                clock.finish(); costs["train_seconds"] = clock.data["train_seconds"]
            step += 1; last_loss = loss; exposures[key] += 1
            costs["train_case_exposures_committed"] += 1; costs["train_node_exposures_committed"] += node_count
            with (output / "loss.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step, "case": key, "nodes": node_count, "encoded4_case_mean_loss": loss,
                                         "train_seconds": costs["train_seconds"]}, allow_nan=False) + "\n")
            del case
            if step in STEPS:
                readout_train(); commit(True)
            elif step % 25 == 0:
                commit()
        commit()
        if step == 4800:
            require(all(v == 4800 // len(train_keys) for v in exposures.values()), "Full balanced-pass exposure differs")
        completed = step == 4800
        summary = {"status": "completed", "kind": KIND, "stage": "train", "config": config,
                   "arm_id": arm["id"], "method": method, "train_CADs": train_cads, "completed4800": completed,
                   "endpoint_step": step, "stop_reason": "fixed_updates_completed" if completed else reason,
                   "available_checkpoints": [s for s in STEPS if (output / f"step{s}.pt").exists()],
                   "first_update_gradient_check": first_grad, "case_exposure_counts": exposures,
                   "costs": costs, "memory": memory(), "TRAIN_endpoint_readouts": history,
                   "train_clock": clock.data, "elapsed_this_attempt_seconds": time.monotonic() - began,
                   "budget_overshoot_seconds": max(0., costs["train_seconds"] - 7200),
                   "DEV_target_reads": 0, "scaler_fits": 0, "checkpoint_selection": "fixed predeclared steps only; no best selection"}
    except Exception as error:
        # OOM/nonfinite/backend is an observed recipe incompatibility; all
        # absent fixed endpoints later receive explicit terminal entries.
        text = str(error).lower()
        negative = isinstance(error, FloatingPointError) or "out of memory" in text or "cuda" in text
        summary = {"status": "completed" if negative else "failed", "kind": KIND, "stage": "train", "config": config or {"contract": contract},
                   "arm_id": arm["id"], "method": method, "train_CADs": train_cads, "completed4800": False,
                   "endpoint_step": step, "stop_reason": "terminal_backend_or_nonfinite" if negative else "technical_failure",
                   "error": type(error).__name__ + ": " + str(error),
                   "available_checkpoints": [s for s in STEPS if (output / f"step{s}.pt").exists()],
                   "first_update_gradient_check": first_grad, "case_exposure_counts": exposures, "costs": costs,
                   "DEV_target_reads": 0, "scaler_fits": 0, "elapsed_this_attempt_seconds": time.monotonic() - began}
    finally:
        clock.finish()
        if cleanup is not None:
            cleanup()
    write(output / "costs.json", costs)
    write(summary_path, summary); write(output / "status.json", {"status": summary["status"], "phase": "terminal", "step": summary["endpoint_step"], "stop_reason": summary["stop_reason"]})
    return summary


def array_save(np, filename, array):
    filename = path(filename); filename.parent.mkdir(parents=True, exist_ok=True)
    value = np.ascontiguousarray(array); temporary = filename.with_name(filename.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, filename); stat = filename.stat()
    return {"path": str(filename), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "shape": list(value.shape), "dtype": str(value.dtype), "values_sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def prediction_folder(output, step, key):
    return output / "predictions" / f"step{step}" / key.replace("/", "_")


def run_predict(plan_path, method, train_cads, output=None):
    contract, plan = request_contract(plan_path, method, train_cads, "predict"); release(contract, plan)
    from . import coverage_comparison_data as data
    manifest = data.read_cache_metadata(plan_path, contract=contract["input_contract"])
    arm = contract["arm"]; output = path(output) if output else arm_output(plan, arm)
    train = read(output / "summary.json")
    require(train["status"] == "completed", "Arm training is not terminal; no prediction inventory may be committed")
    expected_train, _ = request_contract(plan_path, method, train_cads, "train")
    require(train["config"]["contract"] == expected_train, "Terminal arm identity differs")
    if (output / "prediction_summary.json").exists():
        saved = read(output / "prediction_summary.json")
        if saved["status"] == "completed":
            require(saved["config"]["contract"] == contract, "Completed inference identity differs"); return saved
    if not any((output / f"step{s}.pt").exists() for s in STEPS):
        # No real endpoint survived the terminal training incompatibility.
        # Commit failures without instantiating the same unsuccessful backend.
        records = {}
        for step in STEPS:
            for key in manifest["validation_keys"]:
                filename = prediction_folder(output, step, key) / "manifest.json"
                item = {"status": "completed", "terminal": True, "contract": contract, "arm": arm["id"],
                        "key": key, "step": step, "checkpoint_id": step, "input_contract_sha256": contract["input_contract_sha256"],
                        "prediction_complete": False, "finite_allN": False,
                        "reason": "No committed fixed endpoint: " + train["stop_reason"]}
                if filename.exists():
                    require(read(filename) == item, "Committed terminal failure differs")
                else:
                    write(filename, item)
                records[f"{arm['id']}:{step}:{key}"] = {"path": str(filename), "sha256": sha(filename)}
        result = {"status": "completed", "kind": KIND, "stage": "predict", "config": {"contract": contract},
                  "arm_id": arm["id"], "required_entries": 240, "entries": records,
                  "costs": {"inference_seconds": None, "initialization_seconds": None, "reason": "No inference attempted"},
                  "memory": None, "DEV_target_reads": 0, "scaler_fits": 0}
        write(output / "prediction_summary.json", result); return result
    began = time.monotonic(); cleanup = None; records = {}
    prior = read(output / "prediction_costs.json") if (output / "prediction_costs.json").exists() else {}
    inference_seconds = float(prior.get("inference_seconds", 0.)); init_seconds = float(prior.get("initialization_seconds", 0.))
    checkpoint_seconds = float(prior.get("checkpoint_load_and_identity_seconds", 0.)); input_seconds = float(prior.get("input_read_seconds", 0.))
    def persist_costs():
        write(output / "prediction_costs.json", {"inference_seconds": inference_seconds, "initialization_seconds": init_seconds,
              "checkpoint_load_and_identity_seconds": checkpoint_seconds, "input_read_seconds": input_seconds})
    try:
        np, torch, model, scaler, cleanup, rng_state, restore_rng, _, predict, query_grid = model_environment(manifest, method)
        init_seconds += time.monotonic() - began; persist_costs()
        for step in STEPS:
            checkpoint = output / f"step{step}.pt"; checkpoint_info = None; endpoint_error = None
            if checkpoint.exists():
                t = time.monotonic(); saved = torch.load(checkpoint, map_location="cuda", weights_only=False)
                require(saved["step"] == step and digest(saved["config"]) == digest(train["config"]), "Fixed endpoint model/config differs")
                model.load_state_dict(saved["model"], strict=True); model.eval()
                checkpoint_info = {"path": str(checkpoint), "step": step, "bytes": checkpoint.stat().st_size,
                                   "mtime_ns": checkpoint.stat().st_mtime_ns, "sha256": sha(checkpoint)}
                checkpoint_seconds += time.monotonic() - t
                persist_costs()
            else:
                endpoint_error = "Fixed checkpoint unavailable: " + train["stop_reason"]
            for rank, key in enumerate(manifest["validation_keys"]):
                folder = prediction_folder(output, step, key); filename = folder / "manifest.json"
                token = f"{arm['id']}:{step}:{key}"
                if filename.exists():
                    record = read(filename); require(record["contract"] == contract and record["key"] == key and record["checkpoint_id"] == step, "Committed prediction identity differs")
                    records[token] = {"path": str(filename), "sha256": sha(filename)}; continue
                record = {"status": "completed", "terminal": True, "contract": contract, "arm": arm["id"],
                          "key": key, "step": step, "checkpoint_id": step, "input_contract_sha256": contract["input_contract_sha256"],
                          "prediction_complete": False, "finite_allN": False, "reason": endpoint_error}
                if endpoint_error is None:
                    case = None
                    try:
                        t = time.monotonic(); case = data.load_case(manifest, key, with_targets=False, device="cuda", for_fno=method == "fno")
                        input_seconds += time.monotonic() - t
                        require("target" not in case and "physical" not in case, "DEV inference loader leaked target")
                        torch.cuda.synchronize(); t = time.monotonic()
                        devices = [torch.cuda.current_device()]
                        with torch.random.fork_rng(devices=devices), torch.no_grad():
                            # The same endpoint/case seed across both CAD scales;
                            # no dependency on arm order, training RNG or labels.
                            eval_seed = 2026 + int(hashlib.sha256(f"DEV:{step}:{key}".encode()).hexdigest()[:8], 16)
                            torch.manual_seed(eval_seed)
                            encoded = predict(torch, model, method, case, query_grid)
                        torch.cuda.synchronize(); seconds = time.monotonic() - t; inference_seconds += seconds
                        require(encoded.shape == (len(case["xyz"]), 4) and np.isfinite(encoded).all(), "Incomplete/nonfinite fullN encoded prediction")
                        physical = scaler.decode(encoded)
                        require(np.isfinite(physical).all(), "Nonfinite physical prediction")
                        record.update(prediction_complete=True, reason=None, all_original_nodes=True, node_count=len(case["xyz"]),
                                      encoded=array_save(np, folder / "encoded4.npy", encoded.astype(np.float32)),
                                      physical=array_save(np, folder / "physical4.npy", physical), checkpoint=checkpoint_info,
                                      inference_seconds=seconds, eval_rng_seed=eval_seed,
                                      units=["mm", "mm", "mm", "MPa"], signed_stress_unchanged=True)
                        record.update(finite_allN=True, encoded_identity=record["encoded"], physical_identity=record["physical"],
                                      encoded_shape=list(encoded.shape), physical_shape=list(physical.shape))
                        del encoded, physical
                    except Exception as error:
                        text = str(error).lower()
                        if not (isinstance(error, FloatingPointError) or isinstance(error, ValueError) or "cuda" in text or "out of memory" in text):
                            raise
                        record.update(reason=type(error).__name__ + ": " + str(error), prediction_complete=False, finite_allN=False)
                    finally:
                        if case is not None:
                            del case
                write(filename, record); records[token] = {"path": str(filename), "sha256": sha(filename)}
                persist_costs()
                write(output / "prediction_status.json", {"status": "running", "step": step, "case": key, "entries_committed": len(records), "required_entries": 240})
        memory = {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
    finally:
        if cleanup is not None:
            cleanup()
    result = {"status": "completed", "kind": KIND, "stage": "predict", "config": {"contract": contract}, "arm_id": arm["id"],
              "required_entries": 240, "entries": records, "costs": {"inference_seconds": inference_seconds,
              "initialization_seconds": init_seconds, "checkpoint_load_and_identity_seconds": checkpoint_seconds,
              "input_read_seconds": input_seconds, "elapsed_seconds": time.monotonic() - began}, "memory": memory,
              "DEV_target_reads": 0, "scaler_fits": 0}
    write(output / "prediction_summary.json", result); write(output / "prediction_status.json", {"status": "completed", "entries_committed": len(records)})
    return result


def commit_barrier(plan_path):
    contract, plan = request_contract(plan_path, stage="evaluate"); release(contract, plan)
    destination = path(plan["training_output"]) / "prediction_barrier.json"
    required = []; records = {}; finite = 0; failures = 0
    for arm in arms(plan):
        output = arm_output(plan, arm); terminal = read(output / "summary.json")
        require(terminal["status"] == "completed", "Arm still live/failed; stage cannot be closed for DEV readout")
        expected, _ = request_contract(plan_path, arm["method"], arm["train_CADs"], "predict")
        prediction = read(output / "prediction_summary.json")
        require(prediction["status"] == "completed" and prediction["config"]["contract"] == expected, "Arm prediction inventory not terminal")
        for step in STEPS:
            for key in contract["validation_keys"]:
                token = f"{arm['id']}:{step}:{key}"; filename = prediction_folder(output, step, key) / "manifest.json"
                item = read(filename)
                require(item["status"] == "completed" and item["terminal"] is True and item["contract"] == expected
                        and item["arm"] == arm["id"] and item["checkpoint_id"] == step and item["key"] == key,
                        "Missing/nonterminal prediction entry")
                if item["prediction_complete"]:
                    require(item["finite_allN"] is True and item["all_original_nodes"] is True and item["node_count"] > 0
                            and item["physical"]["shape"] == item["encoded"]["shape"] == [item["node_count"], 4], "Incomplete fullN field commit")
                    for name in ("physical", "encoded"):
                        info = path(item[name]["path"]).stat()
                        require(info.st_size == item[name]["bytes"] and info.st_mtime_ns == item[name]["mtime_ns"], "Prediction changed after commit")
                    finite += 1
                else:
                    require(isinstance(item["reason"], str) and bool(item["reason"]), "Failure entry lacks terminal reason")
                    failures += 1
                required.append({"arm": arm["id"], "step": step, "key": key})
                records[token] = {"path": str(filename), "sha256": sha(filename)}
    proof = {"status": "completed", "passed": True, "input_contract_sha256": contract["input_contract_sha256"],
             "validation_keys": contract["validation_keys"], "required_predictions": required,
             "prediction_manifests": records, "finite_fullN_entries": finite, "terminal_failure_entries": failures,
             "required_entries": 2400, "no_further_training_or_endpoint_changes": True, "DEV_target_reads_before_barrier": 0}
    proof["training_permanently_closed"] = True
    require(len(required) == len(records) == finite + failures == 2400, "Fixed barrier inventory differs")
    if destination.exists():
        require(read(destination) == proof, "Immutable closed-stage barrier changed")
    else:
        write(destination, proof)
    return proof


def physical_metrics(np, truth, prediction):
    truth = np.asarray(truth, dtype=np.float64); prediction = np.asarray(prediction, dtype=np.float64)
    require(truth.shape == prediction.shape and truth.ndim == 2 and truth.shape[1] == 4 and len(truth) > 0, "Physical4 schema differs")
    if not (np.isfinite(truth).all() and np.isfinite(prediction).all()):
        raise FloatingPointError("Nonfinite physical metrics")
    error = prediction - truth; u2 = float(np.square(truth[:, :3]).sum()); s2 = float(np.square(truth[:, 3]).sum())
    hotspot_n = max(1, math.ceil(.01 * len(truth))); hot = np.argpartition(np.abs(truth[:, 3]), len(truth) - hotspot_n)[-hotspot_n:]
    hot2 = float(np.square(truth[hot, 3]).sum())
    return {"nodes": len(truth), "u_relative_l2": math.sqrt(float(np.square(error[:, :3]).sum()) / u2) if u2 > 0 else None,
            "stress_relative_l2": math.sqrt(float(np.square(error[:, 3]).sum()) / s2) if s2 > 0 else None,
            "stress_hotspot_relative_l2": math.sqrt(float(np.square(error[hot, 3]).sum()) / hot2) if hot2 > 0 else None,
            "u_vector_error_RMS_mm": math.sqrt(float(np.square(error[:, :3]).sum()) / len(truth)),
            "u_max_vector_error_mm": float(np.linalg.norm(error[:, :3], axis=1).max()),
            "stress_RMSE_MPa": math.sqrt(float(np.square(error[:, 3]).sum()) / len(truth)),
            "stress_MAE_MPa": float(np.abs(error[:, 3]).mean()), "stress_max_abs_error_MPa": float(np.abs(error[:, 3]).max()),
            "stress_hotspot_nodes": hotspot_n, "u_zero_reference": u2 <= 0, "stress_zero_reference": s2 <= 0, "hotspot_zero_reference": hot2 <= 0}


def aggregate(per_case, denominator=60):
    result = {"fixed_denominator": denominator, "metrics_available_cases": sum(v is not None for v in per_case.values()), "per_case": per_case}
    for name, prefix in (("u_relative_l2", "u"), ("stress_relative_l2", "stress"), ("stress_hotspot_relative_l2", "stress_hotspot")):
        values = [item.get(name) if item is not None else None for item in per_case.values()]
        complete = len(values) == denominator and all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values)
        result["case_mean_" + prefix + "_relative_l2"] = sum(values) / denominator if complete else None
        result["worst_case_" + prefix + "_relative_l2"] = max(values) if complete else None
    u, s = result["case_mean_u_relative_l2"], result["case_mean_stress_relative_l2"]
    result["selection_score"] = .5 * (u + s) if u is not None and s is not None else None
    result["all_cases_complete"] = result["metrics_available_cases"] == denominator and result["selection_score"] is not None
    return result


def run_evaluate(plan_path, output=None):
    contract, plan = request_contract(plan_path, stage="evaluate"); release(contract, plan)
    output = path(output) if output else path(plan["training_output"]) / "evaluation"
    if (output / "summary.json").exists():
        previous = read(output / "summary.json")
        if previous["status"] == "completed":
            require(previous["config"]["contract"] == contract, "Completed readout identity differs"); return previous
    proof = commit_barrier(plan_path); output.mkdir(parents=True, exist_ok=True)
    from . import coverage_comparison_data as data
    labels = data.read_dev_targets_after_barrier(plan_path, plan["data_cache"], proof)
    require(labels["status"] == "completed" and set(labels["records"]) == set(contract["validation_keys"]), "Shared DEV label readout incomplete")
    import numpy as np
    began = time.monotonic(); results = {}; target_cache_reads = 0; prediction_reads = 0
    for arm in arms(plan):
        endpoints = {}; parent = read(arm_output(plan, arm) / "summary.json")
        for step in STEPS:
            per = {}
            for key in contract["validation_keys"]:
                item = read(prediction_folder(arm_output(plan, arm), step, key) / "manifest.json")
                if not item["prediction_complete"]:
                    per[key] = None; continue
                record = labels["records"][key]; truth = np.load(path(record["physical_path"]), mmap_mode="r", allow_pickle=False)
                predicted = np.load(path(item["physical"]["path"]), mmap_mode="r", allow_pickle=False)
                target_cache_reads += 1; prediction_reads += 1
                require(hashlib.sha256(np.ascontiguousarray(predicted).tobytes()).hexdigest() == item["physical"]["values_sha256"], "Persisted prediction bytes changed")
                per[key] = physical_metrics(np, truth, predicted)
                del truth, predicted
            endpoints[str(step)] = {"step": step, "validation": aggregate(per), "fixed_checkpoint": True}
        results[arm["id"]] = {"method": arm["method"], "train_CADs": arm["train_CADs"], "completed4800": parent["completed4800"],
                              "training": parent, "endpoints": endpoints, "primary": endpoints["4800"], "best_selection_performed": False}
        write(arm_output(plan, arm) / "metrics.json", endpoints)
    comparisons = {}
    limits = plan["scientific"]["coverage_advancement_gate"]
    for method in METHODS:
        control = next(results[a["id"]]["primary"]["validation"] for a in arms(plan) if a["method"] == method and a["train_CADs"] == 40)
        larger = next(results[a["id"]]["primary"]["validation"] for a in arms(plan) if a["method"] == method and a["train_CADs"] == 200)
        def gain(name):
            x, y = control.get(name), larger.get(name)
            return (x - y) / x if isinstance(x, (int, float)) and x > 0 and y is not None else None
        def ratio(name):
            x, y = control.get(name), larger.get(name)
            return y / x if isinstance(x, (int, float)) and x > 0 and y is not None else None
        s_gain = gain("case_mean_stress_relative_l2"); hot_gain = gain("case_mean_stress_hotspot_relative_l2")
        s_worst = ratio("worst_case_stress_relative_l2"); u_worst = ratio("worst_case_u_relative_l2")
        tests = {"all60_finite_complete": control["all_cases_complete"] and larger["all_cases_complete"],
                 "S_mean_gain": s_gain is not None and s_gain >= limits["relative_S_mean_gain_min"],
                 "hotspot_mean_gain": hot_gain is not None and hot_gain >= limits["relative_hotspot_mean_gain_min"],
                 "S_worst_ratio": s_worst is not None and s_worst <= limits["S_worst_ratio_max"],
                 "U_worst_ratio": u_worst is not None and u_worst <= limits["U_worst_ratio_max"]}
        comparisons[method] = {"passed": all(tests.values()), "criteria": tests, "relative_S_mean_gain": s_gain,
                               "relative_hotspot_mean_gain": hot_gain, "S_worst_ratio": s_worst, "U_worst_ratio": u_worst,
                               "engineering_qualification_claimed": False}
    summary = {"status": "completed", "kind": KIND, "stage": "evaluate", "config": {"contract": contract},
               "arms": results, "coverage_comparisons": comparisons, "validation_keys": contract["validation_keys"],
               "prediction_barrier": proof, "label_read_receipt": labels.get("read_receipt"),
               "read_receipt": {"owned_shared_DEV_physical_array_reads": target_cache_reads,
                                "own_prediction_array_reads": prediction_reads,
                                "new_DEV_source_reads": labels.get("read_receipt"),
                                "scaler_fits": 0, "training_updates": 0},
               "fixed_DEV_denominator": 60, "scaler_fits": 0, "training_updates": 0,
               "costs": {"readout_metric_seconds": time.monotonic() - began},
               "interpretation": "Fixed endpoint single-seed coverage/exposure comparison; original-node signed4 only; capacity differs across families; no final solver or continuous-gradient qualification"}
    write(output / "metrics.json", {"arms": results, "coverage_comparisons": comparisons}); write(output / "summary.json", summary)
    write(output / "status.json", {"status": "completed", "stage": "evaluate", "fixed_DEV_denominator": 60})
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", default="notes/geobc_coverage_comparison_plan.json")
    p.add_argument("--stage", choices=("train", "predict", "evaluate"), required=True)
    p.add_argument("--method", choices=METHODS); p.add_argument("--train-cads", type=int, choices=(40, 200)); p.add_argument("--output")
    args = p.parse_args()
    if args.stage != "evaluate" and (args.method is None or args.train_cads is None):
        p.error("train/predict requires --method and --train-cads")
    function = {"train": run_train, "predict": run_predict, "evaluate": run_evaluate}[args.stage]
    result = function(args.plan, output=args.output) if args.stage == "evaluate" else function(args.plan, args.method, args.train_cads, args.output)
    print(json.dumps({"status": result["status"], "stage": result["stage"], "kind": result["kind"]}, ensure_ascii=False), flush=True)
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
