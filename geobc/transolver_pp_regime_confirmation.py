"""One frozen TRAIN-only ++ author-regime confirmation, not a family ranking.

Import/default CLI are standard-library/source-only. Scientific content is
opened only after a genuine release, parent CPU/terminal proof and GPU lock.
"""
from __future__ import annotations

import argparse
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
OWN = "geobc/transolver_pp_regime_confirmation.py"
PLAN = "notes/geobc_transolver_pp_regime_plan.json"
KIND = "transolver_pp_TRAIN120_author_regime_confirmation"
DECISION = "proceed_frozen_PP_TRAIN120_author_regime"
STEPS = (4800, 24000)
ZERO = ("DEV_target_reads", "held5_target_reads", "raw_source_reads", "old_final_reads",
        "sealed_test_reads", "DEV_dia_reads", "scaler_fits", "pretrained_weight_reads",
        "model_selection_updates", "new_data_preparations")
ACTUAL = ("model_initializations", "TRAIN_input_load_attempts", "TRAIN_input_load_completed",
          "input_array_reads", "cached_TRAIN_target_array_reads", "backward_calls",
          "optimizer_steps", "scheduler_steps", "gradient_clip_calls", "own_checkpoint_read_attempts",
          "own_checkpoint_reads", "own_checkpoint_writes", "own_optimizer_restores",
          "own_scheduler_restores", "TRAIN_readout_case_attempts", "TRAIN_fullN_predictions",
          "TRAIN_metric_cases")


class NotReady(RuntimeError):
    pass


class Budget(RuntimeError):
    pass


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
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def sha(filename):
    h = hashlib.sha256()
    with path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, message, error=ValueError):
    if not ok:
        raise error(message)


def schedule(keys):
    rng = random.Random(2026); rows = []
    for _ in range(200):
        current = list(keys); rng.shuffle(current); rows.extend(current)
    return rows


def request_contract(plan_path=PLAN):
    """Only pinned source and small JSON; no Torch, scientific arrays or PT."""
    plan = read(plan_path); science = plan["scientific"]
    require(plan["kind"] == KIND and plan.get("scientific_rules_frozen") is True
            and digest(science) == plan["scientific_digest"], "PP scientific recipe is not frozen")
    require(science["fixed_updates"] == 24000 and science["complete_passes"] == 200
            and science["TRAIN_readout_updates"] == list(STEPS) and science["seed"] == 2026
            and science["budget"] == {"training_seconds": 28800, "nontraining_seconds": 1200,
                                      "whole_wall_seconds": 30000, "peak_GPU_GiB": 24}, "Fixed exposure/budget differs")
    require(science["optimizer"] == {"name": "Adam", "max_lr": .001, "weight_decay": 0.,
            "fused": True, "gradient_clip_norm": 1.0}, "Frozen Adam/clip bundle differs")
    expected_cycle = {"max_lr": .001, "total_steps": 24000, "pct_start": .3,
        "anneal_strategy": "cos", "cycle_momentum": True, "base_momentum": .85,
        "max_momentum": .95, "div_factor": 25., "final_div_factor": 1000.,
        "three_phase": False, "last_epoch": -1}
    require(science["OneCycleLR"] == expected_cycle, "OneCycle horizon/defaults differ")
    parent = read(plan["parent_plan"])
    require(parent["scientific_digest"] == science["parent_scientific_digest"] == digest(parent["scientific"])
            and parent["source_sha256"] == plan["parent_source_sha256"], "Frozen parent protocol changed")
    pins = {name: sha(name) for name in plan["parent_source_sha256"]}
    require(pins == plan["parent_source_sha256"], "Frozen twenty-source map changed")
    for name, expected in plan["additional_source_sha256"].items():
        pins[name] = sha(name); require(pins[name] == expected, "Additional accepted source changed")
    pins[OWN] = sha(OWN)
    require(sha(science["Torch_scheduler_source"]["path"]) == science["Torch_scheduler_source"]["sha256"],
            "Actual Torch scheduler source differs")
    from . import coverage_comparison_data as data
    from . import coverage_comparison_train as train
    manifest = data.read_bootstrap_TRAIN40(plan["parent_plan"])
    keys = science["TRAIN_keys"]
    require(len(keys) == 120 and len(set(keys)) == 120 and manifest["train_keys"] == {"40": keys}
            and set(manifest["records"]) == set(keys) and manifest["scaler"] == science["scaler"]
            and manifest["complete_shared660"] is False, "Only immutable original TRAIN120 is permitted")
    original_contract, _ = train.request_contract(plan["parent_plan"], "transolver_pp", 40, "train")
    reference = read(plan["reference_training_summary"])
    require(sha(plan["reference_training_summary"]) == plan["reference_training_summary_sha256"]
            and reference["status"] == "completed" and reference["completed4800"] is True
            and reference["config"]["contract"] == original_contract, "Immutable constant4800 reference differs")
    require(sha(plan["reference_TRAIN_fit_summary"]) == plan["reference_TRAIN_fit_summary_sha256"],
            "Immutable fullTRAIN120 reference metrics changed")
    order = schedule(keys)
    require(order[:4800] == train.case_schedule(keys) and len(order) == 24000, "Case-order prefix changed")
    return {"schema": 1, "kind": KIND, "scientific_digest": plan["scientific_digest"],
        "parent_scientific_digest": science["parent_scientific_digest"], "source_sha256": pins,
        "Torch_scheduler_source": science["Torch_scheduler_source"], "TRAIN_keys": keys,
        "input_contract_sha256": digest(manifest["contract"]), "scaler": manifest["scaler"],
        "case_order_sha256": digest(order), "fixed_updates": 24000, "readout_updates": list(STEPS),
        "reference_config_sha256": digest(reference["config"]),
        "reference_training_summary_sha256": plan["reference_training_summary_sha256"],
        "reference_TRAIN_fit_summary_sha256": plan["reference_TRAIN_fit_summary_sha256"],
        "read_scope": "Original cached TRAIN120 only; no held5/DEV/source labels"}, plan


def release(contract, plan):
    marker = read(plan["root_review"])
    require(marker.get("passed") is True and marker.get("regime_confirmation_released") is True
            and marker.get("actual_execution_released") is True
            and marker.get("decision") == DECISION and marker.get("scientific_digest") == contract["scientific_digest"]
            and marker.get("source_sha256") == contract["source_sha256"][OWN]
            and marker.get("source_pins") == contract["source_sha256"], "Separate authentic root release required")


def same_contract(saved, current, plan):
    """Only a specifically reviewed driver-only technical repair may migrate."""
    if saved == current:
        return True
    old = json.loads(json.dumps(saved)); new = json.loads(json.dumps(current))
    prior = old.get("source_sha256", {}).get(OWN)
    accepted = read(plan["root_review"]).get("accepted_prior_driver_sha256", [])
    if prior not in accepted:
        return False
    old["source_sha256"][OWN] = new["source_sha256"][OWN]
    return old == new


def same_config(saved, current, plan):
    old = dict(saved); new = dict(current)
    if not same_contract(old.pop("contract"), new.pop("contract"), plan):
        return False
    return digest(old) == digest(new)


def parent_gate(plan):
    """Prove a real CPU waiter, or the original runner's input-negative exit."""
    from .coverage_train_fit_diagnostic import waiting_coordinator, live_process
    require(os.name != "nt" and Path("/proc").is_dir(), "Use the common WSL process namespace", NotReady)
    record = read(plan["coordinator_status"])
    if record.get("phase") == "waiting_for_acquisition":
        return {"mode": "live_CPU_wait", **waiting_coordinator(plan)}
    require(record.get("phase") == "failed" or record.get("phase") == "completed",
            "Parent is neither CPU-waiting nor input-terminal", NotReady)
    require(live_process(record.get("pid")) is None and live_process(record.get("child_pid")) is None,
            "Parent/child process is still active", NotReady)
    folder = path(plan["parent_output"])
    acquired_file = folder / "acquisition_terminal.json"
    input_file = folder / "coordinator_summary.json"
    if acquired_file.exists():
        terminal = read(acquired_file)
        raw = read(terminal["acquisition_summary_path"])
        require(terminal.get("status") == "completed" and terminal.get("phase") == "acquisition_terminal_negative"
                and terminal.get("no_case_replacement") is True
                and raw.get("status") in ("failed", "completed_negative", "budget_exhausted")
                and path(terminal["acquisition_summary_path"]) == path(plan["acquisition_summary"]),
                "Acquisition negative proof differs", NotReady)
        proof_path = acquired_file; reason = terminal.get("terminal_reason")
    else:
        terminal = read(input_file)
        require(terminal.get("status") == "completed" and terminal.get("phase") == "input_scientific_negative"
                and terminal.get("no_case_replacement") is True, "No authentic input-negative proof", NotReady)
        proof_path = input_file; reason = terminal.get("reason")
    return {"mode": "parent_input_terminal_negative", "parent_PID": record.get("pid"),
            "parent_phase": record["phase"], "parent_and_child_exited": True,
            "proof": {"path": str(proof_path), "sha256": sha(proof_path)}, "reason": reason,
            "observed_at_utc": utc(), "TRAIN120_remains_valid": True}


def parent_identity(proof):
    if proof["mode"] == "live_CPU_wait":
        return {k: proof[k] for k in ("mode", "pid", "start_ticks", "argv", "cwd")}
    return {k: proof[k] for k in ("mode", "parent_PID", "parent_phase", "proof")}


class Clock:
    """Compact cumulative ledger; no per-update growing-prefix rewrites.

An interrupted open interval is conservatively charged through next resume,
unless a root-observed exit epoch for that exact attempt is provided.
"""
    def __init__(self, output, entry_epoch):
        self.file = output / "clock.json"; self.output = output
        self.data = read(self.file) if self.file.exists() else {
            "total_seconds": 0., "training_seconds": 0., "nontraining_seconds": 0.,
            "unknown_interruption_charge_seconds": 0., "attempts": 0, "active": False}
        if self.data["active"]:
            exit_file = output / "observed_attempt_exit.json"
            observed = read(exit_file) if exit_file.exists() else {}
            valid = observed.get("attempt_id") == self.data["attempt_id"] and isinstance(observed.get("exit_epoch"), (int, float))
            end = min(time.time(), max(self.data["last_epoch"], observed["exit_epoch"])) if valid else time.time()
            gap = max(0., end - self.data["last_epoch"])
            self.data["total_seconds"] += gap
            self.data["training_seconds" if self.data["phase"] == "training" else "nontraining_seconds"] += gap
            self.data["unknown_interruption_charge_seconds"] += gap
        self.data["attempts"] += 1
        self.data.update(active=True, closed=False, phase="initialization", attempt_id=str(time.time_ns()),
                         pid=os.getpid(), started_epoch=entry_epoch, last_epoch=time.time())
        initial = max(0., time.time() - entry_epoch)
        self.data["total_seconds"] += initial; self.data["nontraining_seconds"] += initial
        self.last = time.monotonic(); self.last_persist = 0.; self.persist()

    def update(self):
        now = time.monotonic(); seconds = now - self.last; self.last = now
        self.data["total_seconds"] += seconds
        self.data["training_seconds" if self.data["phase"] == "training" else "nontraining_seconds"] += seconds
        self.data["last_epoch"] = time.time()

    def persist(self):
        self.update(); write(self.file, self.data); self.last_persist = time.monotonic()

    def phase(self, name):
        self.update()
        if self.data["phase"] != name:
            self.data["phase"] = name; self.persist()

    def guard(self, torch=None):
        self.update()
        if self.data["total_seconds"] >= 30000 or self.data["training_seconds"] >= 28800 or self.data["nontraining_seconds"] >= 1200:
            raise Budget("Cumulative frozen training/nontraining/whole-wall limit reached")
        if torch is not None and max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()) > 24 * 1024**3:
            raise Budget("Actual allocator peak exceeds24GiB")

    def close(self):
        self.update(); self.data.update(active=False, closed=True, phase="terminal", closed_at_utc=utc()); self.persist()


def brief(summary, filename):
    lines = ["++固定作者制度确认；仅原TRAIN120，不含DEV/held5，也不是最终家族排名。",
             f"实际更新 {summary['endpoint_step']}/24000；状态 {summary['stop_reason']}。",
             "| 固定端点 | TRAIN U / S均值 | 热点均值 / 最坏S |", "|---|---:|---:|"]
    def percent(value): return "缺失" if value is None else f"{value*100:.2f}%"
    for step in STEPS:
        a = summary["endpoints"].get(str(step), {}).get("aggregate", {})
        lines.append(f"| {step} | {percent(a.get('case_mean_u_relative_l2'))} / {percent(a.get('case_mean_stress_relative_l2'))} | "
                     f"{percent(a.get('case_mean_stress_hotspot_relative_l2'))} / {percent(a.get('worst_case_stress_relative_l2'))} |")
    filename = path(filename); temporary = filename.with_name(filename.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8"); os.replace(temporary, filename)


def run(plan_path=PLAN, output=None, start=False):
    entry_epoch = time.time(); plan = read(plan_path)
    if not start:
        return {"status": "source_only_ready", "kind": KIND, "writes": False, "actual_execution": False}
    contract, plan = request_contract(plan_path); release(contract, plan)
    output = path(output or plan["output"])
    require(output == path(plan["output"]), "The frozen output path cannot change")
    previous = None
    if (output / "summary.json").exists():
        previous = read(output / "summary.json")
        require(same_contract(previous["config"]["contract"], contract, plan), "Existing campaign identity differs")
        if previous["status"] == "completed":
            return previous
    before = parent_gate(plan)
    from . import coverage_comparison_train as train
    held = train.lock(plan["shared_GPU_lock"])
    try:
        held.__enter__()
    except (OSError, RuntimeError) as error:
        raise NotReady("Shared GPU lock busy; no scientific content opened") from error
    output.mkdir(parents=True, exist_ok=True)
    clock = None; torch = None; cleanup = None; summary = None
    run_process = None
    receipt = {name: 0 for name in ZERO + ACTUAL}; receipt["input_read_receipt_exact"] = True
    costs = {name: 0. for name in ("initialization_seconds", "checkpoint_seconds", "TRAIN_readout_seconds")}
    step = 0; exposures = {key: 0 for key in contract["TRAIN_keys"]}; first_grad = None; last_loss = None
    endpoints = {}; peak_prior = {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    if previous is not None and previous.get("memory_measurement_available") is True:
        peak_prior = previous["memory"]
    try:
        clock = Clock(output, entry_epoch)
        from .coverage_train_fit_diagnostic import live_process
        run_process = live_process(os.getpid()); require(run_process is not None, "Cannot prove own Linux process", NotReady)
        write(output / "execution.json", {"run_process": run_process, "attempt_id": clock.data["attempt_id"], "UTC": utc()})
        after = parent_gate(plan)
        require(parent_identity(before) == parent_identity(after), "Parent state changed while acquiring GPU", NotReady)
        refreshed, _ = request_contract(plan_path); require(refreshed == contract, "Source/input metadata changed before science")
        if (output / "read_receipt.json").exists():
            receipt = read(output / "read_receipt.json")
            require(all(type(receipt[k]) is int and receipt[k] == 0 for k in ZERO), "Forbidden read recorded")
        if (output / "costs.json").exists(): costs = read(output / "costs.json")
        write(output / "parent_wait_or_terminal_proof.json", {"before_lock": before, "after_lock": after,
              "shared_GPU_lock": str(path(plan["shared_GPU_lock"])), "lock_acquired": True})
        from . import coverage_comparison_data as data
        manifest = data.read_bootstrap_TRAIN40(plan["parent_plan"]); keys = contract["TRAIN_keys"]; order = schedule(keys)
        reference = read(plan["reference_training_summary"]); began = time.monotonic()
        receipt["model_initializations"] += 1
        np, torch, model, scaler, cleanup, rng_state, restore_rng, grad_check, predict, query_grid = train.model_environment(manifest, "transolver_pp")
        torch.set_grad_enabled(True)
        require(sha(torch.optim.lr_scheduler.__file__) == contract["Torch_scheduler_source"]["sha256"], "Imported Torch scheduler source differs")
        require(digest(model.source_config) == digest(reference["config"]["model"])
                and train.initial_identity(model) == reference["config"]["initial_parameters"], "Fresh model/init differs from original recipe")
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0., fused=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **plan["scientific"]["OneCycleLR"])
        config = {"contract": contract, "model": model.source_config, "parameter_counts": model.parameter_counts,
                  "scaler": manifest["scaler"], "initial_parameters": reference["config"]["initial_parameters"],
                  "precision": reference["config"]["precision"], "TRAIN_only": True,
                  "optimizer": plan["scientific"]["optimizer"], "OneCycleLR": plan["scientific"]["OneCycleLR"]}
        if (output / "config.json").exists(): require(same_config(read(output / "config.json"), config, plan), "Resume initial/source/config differs")
        write(output / "config.json", config)
        if (output / "last.pt").exists():
            receipt["own_checkpoint_read_attempts"] += 1
            saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False); receipt["own_checkpoint_reads"] += 1
            require(same_config(saved["config"], config, plan) and 0 <= saved["step"] <= 24000
                    and saved["next_case_position"] == saved["step"] and saved["case_order_sha256"] == contract["case_order_sha256"], "Resume cursor/config differs")
            model.load_state_dict(saved["model"], strict=True)
            # Construct both first, then restore both; never step to chase a cursor.
            optimizer.load_state_dict(saved["optimizer"]); receipt["own_optimizer_restores"] += 1
            scheduler.load_state_dict(saved["scheduler"]); receipt["own_scheduler_restores"] += 1
            step = saved["step"]; exposures = saved["case_exposure_counts"]
            require(sum(exposures.values()) == step and scheduler.last_epoch == step, "Optimizer/scheduler exposure cursor differs")
            for group, old in zip(optimizer.param_groups, saved["optimizer"]["param_groups"]):
                require(group["lr"] == old["lr"] and group["betas"] == old["betas"], "Resume LR/beta1 differs")
            restore_rng(torch, np, saved["rng"]); first_grad = saved["first_update_gradient_check"]
            for name in ACTUAL: receipt[name] = max(receipt[name], saved["read_receipt"][name])
            last_loss = saved["last_training_loss"]
            peak_prior = {k: max(peak_prior[k], saved["memory"][k]) for k in peak_prior}; del saved
        require(set(exposures) == set(keys), "Resume TRAIN whitelist differs")
        for endpoint in STEPS:
            filename = output / "TRAIN_readout" / f"step{endpoint}.json"
            if filename.exists():
                value = read(filename); require(same_contract(value["config_contract"], contract, plan), "Saved TRAIN endpoint identity differs")
                endpoints[str(endpoint)] = value
        costs["initialization_seconds"] += time.monotonic() - began
        checkpoint_at = time.monotonic(); heartbeat_at = time.monotonic()
        def memory():
            return {"peak_allocated_bytes": max(peak_prior["peak_allocated_bytes"], int(torch.cuda.max_memory_allocated())),
                    "peak_reserved_bytes": max(peak_prior["peak_reserved_bytes"], int(torch.cuda.max_memory_reserved()))}
        def persist_state():
            clock.update()
            costs.update(training_seconds=clock.data["training_seconds"],nontraining_seconds=clock.data["nontraining_seconds"],
                         total_seconds=clock.data["total_seconds"])
            write(output / "read_receipt.json", receipt); write(output / "costs.json", costs); clock.persist()
        def checkpoint(endpoint=False):
            nonlocal checkpoint_at
            clock.phase("checkpoint"); clock.guard(torch); t = time.monotonic()
            payload = {"config": config, "step": step, "next_case_position": step,
                "case_order_sha256": contract["case_order_sha256"], "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state(torch, np),
                "case_exposure_counts": dict(exposures), "first_update_gradient_check": first_grad,
                "last_training_loss": last_loss, "memory": memory(), "read_receipt": dict(receipt)}
            train.save_checkpoint(torch, output / "last.pt", payload); receipt["own_checkpoint_writes"] += 1
            if endpoint:
                filename = output / f"step{step}.pt"
                if not filename.exists():
                    train.save_checkpoint(torch, filename, payload); receipt["own_checkpoint_writes"] += 1
                info = filename.stat()
                write(output / f"step{step}.json", {"status": "completed", "step": step,
                    "config_contract_sha256": digest(contract), "checkpoint": {"path": str(filename),
                    "bytes": info.st_size, "mtime_ns": info.st_mtime_ns, "sha256": sha(filename)},
                    "case_exposure_counts": dict(exposures), "scheduler_last_epoch": scheduler.last_epoch})
            costs["checkpoint_seconds"] += time.monotonic() - t; persist_state(); checkpoint_at = time.monotonic()
        def load_case(key):
            require(key in keys and key in manifest["records"], "TRAIN whitelist violated")
            receipt["TRAIN_input_load_attempts"] += 1
            try: case = data.load_case(manifest, key, with_targets=True, device="cuda", for_fno=False)
            except Exception:
                receipt["input_read_receipt_exact"] = False; raise
            receipt["TRAIN_input_load_completed"] += 1; native = case["_actual_read_receipt"]
            require(native["DEV_target_reads"] == native["source_archive_reads"] == native["scaler_fits"] == 0, "Unexpected target/source scope")
            receipt["input_array_reads"] += native["input_array_reads"]
            receipt["cached_TRAIN_target_array_reads"] += native["cached_TRAIN_target_array_reads"]
            return case
        def readout():
            clock.phase("TRAIN_readout"); clock.guard(torch); t = time.monotonic()
            filename = output / "TRAIN_readout" / f"step{step}.json"
            if filename.exists():
                value = read(filename); require(same_contract(value["config_contract"], contract, plan), "Immutable readout contract differs")
                if value["all120_metrics_complete"]:
                    endpoints[str(step)] = value; return
                rows = value["per_case"]; failures = value["case_failures"]
                require(set(rows) == set(failures) == set(keys), "Partial readout fixed denominator differs")
            else:
                rows = {key: None for key in keys}; failures = {key: "not_attempted" for key in keys}
            original_rng = rng_state(torch, np); model.eval()
            try:
                with torch.no_grad():
                    for key in keys:
                        if rows[key] is not None: continue
                        clock.guard(torch); receipt["TRAIN_readout_case_attempts"] += 1; case = load_case(key)
                        torch.manual_seed(2026 + int(hashlib.sha256(f"TRAIN:{step}:{key}".encode()).hexdigest()[:8],16))
                        encoded = predict(torch, model, "transolver_pp", case, query_grid)
                        torch.cuda.synchronize(); physical = scaler.decode(encoded)
                        require(physical.shape == (len(case["xyz"]),4) and bool(np.isfinite(physical).all()), "Nonfinite fullN TRAIN prediction", FloatingPointError)
                        receipt["TRAIN_fullN_predictions"] += 1
                        rows[key] = train.physical_metrics(np, case["physical"], physical); failures[key] = None
                        receipt["TRAIN_metric_cases"] += 1; del case, encoded, physical
                        if time.monotonic()-clock.last_persist >= 20: persist_state()
            finally:
                restore_rng(torch, np, original_rng); costs["TRAIN_readout_seconds"] += time.monotonic()-t
                aggregate = train.aggregate(rows,120); aggregate.pop("selection_score",None)
                value = {"step":step,"TRAIN_only":True,"config_contract_sha256":digest(contract),"config_contract":contract,
                    "fixed_denominator":120,"per_case":rows,"case_failures":failures,"aggregate":aggregate,
                    "per_case_seed":"2026+SHA256(TRAIN:step:key)[:8]","all120_metrics_complete":aggregate["all_cases_complete"]}
                # Manual technical resume may complete missing TRAIN cases;
                # already committed metrics are never repeated or discarded.
                write(filename,value); endpoints[str(step)]=value; persist_state()
            clock.guard(torch)
            require(value["all120_metrics_complete"], "A fixed TRAIN120 readout is incomplete", FloatingPointError)
        if not (output / "last.pt").exists(): checkpoint(True)
        if step in STEPS and not endpoints.get(str(step),{}).get("all120_metrics_complete"): checkpoint(True); readout()
        reason = "fixed24000_completed"
        while step < 24000:
            clock.phase("training"); clock.guard(torch); key = order[step]; case = load_case(key)
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss = train.train_case(torch,model,case,query_grid); receipt["backward_calls"] += 1
            if first_grad is None:
                first_grad={"step":step+1,"case":key,**grad_check(torch,model)}
                write(output / "first_update_gradient_check.json",first_grad)
            clip = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            receipt["gradient_clip_calls"] += 1
            lr_before = optimizer.param_groups[0]["lr"]; beta1_before=optimizer.param_groups[0]["betas"][0]
            optimizer.step(); receipt["optimizer_steps"] += 1
            scheduler.step(); receipt["scheduler_steps"] += 1; torch.cuda.synchronize()
            nodes=len(case["xyz"]); del case
            step += 1; exposures[key] += 1; last_loss=loss
            require(scheduler.last_epoch == step, "Scheduler did not advance once with Adam")
            with (output / "loss.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps({"step":step,"key":key,"nodes":nodes,"encoded4_case_mean_loss":loss,
                    "lr_before_Adam":lr_before,"beta1_before_Adam":beta1_before,
                    "preclip_gradient_norm":float(clip.item()),"scheduler_last_epoch":scheduler.last_epoch},allow_nan=False)+"\n")
            clock.guard(torch)
            if step in STEPS: checkpoint(True); readout()
            elif time.monotonic()-checkpoint_at >= 600: checkpoint()
            if time.monotonic()-heartbeat_at >= 20:
                write(output / "status.json",{"status":"running","kind":KIND,"phase":"training",
                    "step":step,"complete_passes":step//120,"TRAIN_only":True,
                    "run_process":run_process,
                    "training_seconds":clock.data["training_seconds"],"total_seconds":clock.data["total_seconds"],"updated_at_utc":utc()})
                persist_state(); heartbeat_at=time.monotonic()
        require(exposures=={key:200 for key in keys},"Final full balanced exposure differs")
        checkpoint(); clock.guard(torch)
        completed=True
    except Exception as error:
        negative = isinstance(error,(Budget,FloatingPointError)) or (isinstance(error,RuntimeError) and
            ("out of memory" in str(error).lower() or "cuda" in str(error).lower() or "non-finite" in str(error).lower()))
        completed=False; reason="terminal_budget_or_backend_or_nonfinite" if negative else "technical_failure"
        summary={"status":"completed" if negative else "failed","error_type":type(error).__name__,
                 "error_message":str(error)[:500]}
    finally:
        try:
            if clock is not None:
                clock.phase("terminal_persistence")
                if cleanup is not None:
                    try: cleanup()
                    except Exception as error:
                        completed=False; reason="technical_cleanup_failure"; summary={"status":"failed",
                            "error_type":type(error).__name__,"error_message":str(error)[:500]}
                config=locals().get("config",{"contract":contract})
                summary=summary or {"status":"completed"}
                summary.update(kind=KIND,stage="TRAIN_only_regime_confirmation",terminal=summary["status"]=="completed",
                    config=config,endpoint_step=step,completed24000=completed,stop_reason=reason,endpoints=endpoints,
                    run_process=run_process,
                    case_keys=contract["TRAIN_keys"],case_exposure_counts=exposures,read_receipt=receipt,
                    first_update_gradient_check=first_grad,costs=costs,clock=dict(clock.data),
                    reference_constant4800={"summary_path":plan["reference_TRAIN_fit_summary"],
                        "sha256":plan["reference_TRAIN_fit_summary_sha256"],"no_reference_weights_read":True},
                    completion_gate={"reached24000":completed,"both_fixed_TRAIN120_readouts_complete":all(
                        endpoints.get(str(s),{}).get("all120_metrics_complete") is True for s in STEPS),
                        "TRAIN_only":True,"DEV_selection_performed":False,"final_solver_selection":False},
                    memory=locals().get("memory",lambda:peak_prior)(),memory_measurement_available="memory" in locals())
                clock.update()
                within=clock.data["total_seconds"]<30000 and clock.data["training_seconds"]<28800 and clock.data["nontraining_seconds"]<1200
                summary["completion_gate"]["within_frozen_budget_precommit"]=within
                summary["completion_gate"]["passed"]=completed and summary["completion_gate"]["both_fixed_TRAIN120_readouts_complete"] and within
                costs.update(training_seconds=clock.data["training_seconds"],nontraining_seconds=clock.data["nontraining_seconds"],
                    total_seconds_before_terminal_persist=clock.data["total_seconds"],
                    unknown_interruption_charge_seconds=clock.data["unknown_interruption_charge_seconds"],
                    closed_clock_is_authoritative=True,external30000_guard_is_authoritative=True)
                baseline=read(plan["reference_TRAIN_fit_summary"])["arms"]["transolver_pp_cad40_fresh4800"]["aggregate"]
                summary["reference_constant4800"]["aggregate"]={k:v for k,v in baseline.items() if k!="per_case"}
                summary["interpretation"]="TRAIN4800 same-exposure bundle comparison; TRAIN24000 bundles regime and exposure. No DEV, family optimum or engineering qualification."
                write(output / "costs.json",costs); write(output / "read_receipt.json",receipt)
                brief(summary,output / "BRIEF.md"); write(output / "summary.json",summary)
                write(output / "status.json",{"status":summary["status"],"kind":KIND,"phase":"terminal",
                    "step":step,"stop_reason":reason,"TRAIN_only":True,"run_process":run_process,"updated_at_utc":utc()})
                clock.close()
        finally:
            held.__exit__(None,None,None)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",default=PLAN); parser.add_argument("--output"); parser.add_argument("--start",action="store_true")
    args=parser.parse_args(); result=run(args.plan,args.output,args.start)
    print({"status":result["status"],"kind":KIND,"endpoint_step":result.get("endpoint_step")},flush=True)


if __name__=="__main__":
    main()
