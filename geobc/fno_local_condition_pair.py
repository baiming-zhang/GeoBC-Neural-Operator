"""TRAIN-only FNO local25 representation pair; import/default CLI are stdlib.

The only arm difference is grid-roundtrip versus original-node local25.
Scientific reads require a genuine release, predecessor proof and GPU lock.
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
OWN = "geobc/fno_local_condition_pair.py"
PLAN = "notes/geobc_fno_local_condition_pair_plan.json"
KIND = "FNO_TRAIN120_local25_representation_pair"
ARMS = ("FNO_latent_query_grid25", "FNO_latent_query_native25")
DECISION = "proceed_frozen_FNO_local25_pair_TRAIN120"
CHUNK = 32768
ZERO = ("DEV_target_reads", "held5_target_reads", "raw_source_reads", "old_final_reads",
        "sealed_test_reads", "DEV_dia_reads", "scaler_fits", "pretrained_weight_reads",
        "new_data_preparations", "model_selection_updates", "physical_U_or_tensor_input_reads")
ACTUAL = ("model_initializations", "TRAIN_input_load_attempts", "TRAIN_input_load_completed",
          "input_array_reads", "cached_TRAIN_target_array_reads", "unreturned_load_receipts",
          "geometry_encoder_calls", "FNO_fullcase_calls", "query_chunk_calls", "query_chunk_backward_calls",
          "original_encode_backward_calls", "optimizer_steps", "own_checkpoint_read_attempts",
          "own_checkpoint_reads", "own_checkpoint_writes", "own_optimizer_restores",
          "TRAIN_readout_case_attempts", "TRAIN_fullN_predictions", "TRAIN_metric_cases",
          "initial_parameter_hash_reads")


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
    tmp = filename.with_name(filename.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, filename)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def sha(filename):
    h = hashlib.sha256()
    with path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, message, error=ValueError):
    if not ok:
        raise error(message)


def request_contract(plan_path=PLAN):
    """Source/small JSON/stat only; never model, target array or checkpoint body."""
    plan = read(plan_path); science = plan["scientific"]
    require(plan.get("kind") == KIND and plan.get("scientific_rules_frozen") is True
            and digest(science) == plan["scientific_digest"], "FNO pair science is not frozen")
    require(science["arms"] == list(ARMS) and science["seed"] == 2026
            and science["fixed_updates"] == 4800 and science["complete_passes"] == 40
            and science["TRAIN_readout_updates"] == [0, 4800]
            and science["chunk_nodes"] == CHUNK, "Frozen exposure/readout differs")
    require(science["optimizer"] == {"name": "Adam", "lr": .001, "weight_decay": 0.,
            "fused": True, "scheduler": "constant", "gradient_clip_norm": None}, "Frozen optimizer differs")
    require(science["budget"] == {"training_seconds_each_arm": 3600, "whole_wall_seconds": 9000,
            "peak_GPU_GiB": 24}, "Frozen budget differs")
    require(science["progress_gate"] == {"S_mean_ratio_max": .8, "U_mean_ratio_max": 1.05,
            "S_worst_ratio_max": 1., "hotspot_worst_ratio_max": 1., "positive_denominator_required": True},
            "Predeclared TRAIN progress gate differs")
    require(science["model"] == {"cloud_points": 1024, "PointNet_dim": 32, "PointNet_width": 64,
            "grid_resolution": 64, "grid_in_channels": 61, "FNO_width": 16, "modes": 12,
            "blocks": 4, "padding": 0, "projection_width": 128, "latent_out_channels": 16,
            "checkpoint_blocks": True, "query_layers": [76, 128, 128, 4], "query_activation": "GELU"},
            "Common architecture differs")
    parent = read(plan["parent_plan"])
    require(digest(parent["scientific"]) == science["parent_scientific_digest"] == parent["scientific_digest"]
            and parent["source_sha256"] == plan["parent_source_sha256"], "Original parent protocol changed")
    pins = {name: sha(name) for name in plan["parent_source_sha256"]}
    require(pins == plan["parent_source_sha256"], "Original twenty-source map changed")
    for name, expected in plan["additional_source_sha256"].items():
        pins[name] = sha(name); require(pins[name] == expected, "Additional source changed: " + name)
    pins[OWN] = sha(OWN)
    if plan.get("own_source_sha256"):
        require(pins[OWN] == plan["own_source_sha256"], "FNO pair driver source changed")
    require(plan.get("source_sha256", pins) == pins, "Combined plan source map differs")
    from . import coverage_comparison_data as data
    from . import coverage_comparison_train as train
    manifest = data.read_bootstrap_TRAIN40(plan["parent_plan"]); keys = science["TRAIN_keys"]
    require(manifest.get("status") == "completed" and manifest.get("feasibility_pass") is True
            and manifest["train_keys"] == {"40": keys} and len(keys) == len(set(keys)) == 120
            and set(manifest["records"]) == set(keys) and manifest["scaler"] == science["scaler"]
            and manifest.get("complete_shared660") is False, "Only frozen TRAIN120 bootstrap is allowed")
    order = train.case_schedule(keys)
    require(len(order) == 4800 and all(order.count(key) == 40 for key in keys), "Balanced case-order differs")
    return {"schema": 1, "kind": KIND, "scientific_digest": plan["scientific_digest"],
        "design_scientific_digest": science["design_scientific_digest"], "plan_science": science,
        "source_sha256": pins, "TRAIN_keys": keys, "arms": list(ARMS), "fixed_updates": 4800,
        "input_contract_sha256": digest(manifest["contract"]), "scaler": manifest["scaler"],
        "case_order_sha256": digest(order), "readout_updates": [0, 4800],
        "read_scope": "Original cached TRAIN120 only; no DEV/held5/raw/physical-prediction inputs"}, plan


def release(contract, plan):
    marker = read(plan["root_review"])
    require(marker.get("passed") is True and marker.get("fno_pair_released") is True
            and marker.get("actual_execution_released") is True
            and marker.get("decision") == DECISION and marker.get("scientific_digest") == contract["scientific_digest"]
            and marker.get("source_sha256") == contract["source_sha256"][OWN]
            and marker.get("source_pins") == contract["source_sha256"], "Separate authentic FNO pair release required", NotReady)


def predecessor_gate(plan):
    """Actual WSL process proof; phase strings alone do not establish idleness."""
    from .coverage_train_fit_diagnostic import live_process
    require(os.name != "nt" and Path("/proc").is_dir(), "Use the shared WSL process namespace", NotReady)
    parent_status = read(plan["parent_coordinator_status"])
    require(parent_status.get("phase") in ("completed", "failed")
            and live_process(parent_status.get("pid")) is None
            and live_process(parent_status.get("child_pid")) is None, "Original parent has not exited", NotReady)
    terminal_path = path(plan["parent_output"]) / "acquisition_terminal.json"
    if terminal_path.exists():
        terminal = read(terminal_path); raw = read(terminal["acquisition_summary_path"])
        require(terminal.get("status") == "completed" and terminal.get("phase") == "acquisition_terminal_negative"
                and terminal.get("no_case_replacement") is True
                and raw.get("status") in ("failed", "completed_negative", "budget_exhausted")
                and path(terminal["acquisition_summary_path"]) == path(plan["acquisition_summary"]),
                "Original acquisition-negative proof differs", NotReady)
    else:
        terminal_path = path(plan["parent_output"]) / "coordinator_summary.json"
        terminal = read(terminal_path)
        require(terminal.get("status") == "completed" and terminal.get("phase") == "input_scientific_negative"
                and terminal.get("no_case_replacement") is True, "Original input-negative proof is missing", NotReady)
    pp = read(plan["PP_plan"]); pp_marker_path = path(pp["root_review"])
    require(pp.get("scientific_rules_frozen") is True and digest(pp["scientific"]) == pp["scientific_digest"]
            == plan["PP_scientific_digest"], "PP predecessor recipe differs", NotReady)
    marker = read(pp_marker_path) if pp_marker_path.exists() else {}
    folder = path(pp["output"])
    status = read(folder / "status.json") if (folder / "status.json").exists() else {}
    execution = read(folder / "execution.json") if (folder / "execution.json").exists() else {}
    process_record = status.get("run_process", execution.get("run_process", execution))
    pid = process_record.get("pid", status.get("pid"))
    process = live_process(pid)
    exited = process is None or (process_record.get("start_ticks") is not None
              and process.get("start_ticks") != process_record["start_ticks"])
    require(exited and live_process(status.get("child_pid")) is None, "PP process is still active", NotReady)
    if not (marker.get("regime_confirmation_released") is True and marker.get("actual_execution_released") is True):
        pp_proof = {"mode": "PP_not_released", "PP_plan": str(path(plan["PP_plan"])),
                    "PP_marker": str(pp_marker_path), "regime_confirmation_released": False}
    else:
        require(marker.get("passed") is True and marker.get("scientific_digest") == pp["scientific_digest"],
                "PP marker identity differs", NotReady)
        summary = read(folder / "summary.json"); pp_clock = read(folder / "clock.json")
        require(summary.get("status") == "completed" and summary.get("kind") == pp["kind"]
                and summary.get("config", {}).get("contract", {}).get("scientific_digest") == pp["scientific_digest"]
                and summary["config"]["contract"]["source_sha256"]["geobc/transolver_pp_regime_confirmation.py"] == marker.get("source_sha256")
                and summary["config"]["contract"]["source_sha256"] == marker.get("source_pins")
                and pp_clock.get("closed") is True and pp_clock.get("active") is False,
                "Released PP is not terminal with all processes exited", NotReady)
        pp_proof = {"mode": "PP_terminal_and_exited", "summary": str(folder / "summary.json"),
            "summary_sha256": sha(folder / "summary.json"), "endpoint_step": summary.get("endpoint_step"),
            "stop_reason": summary.get("stop_reason"), "pid": status.get("pid"), "process_exited": True}
    return {"parent_input_negative": {"path": str(terminal_path), "sha256": sha(terminal_path),
            "phase": terminal["phase"], "parent_pid": parent_status.get("pid"), "process_exited": True},
            "PP": pp_proof, "observed_at_utc": utc()}


class Ledger:
    def __init__(self, output, contract):
        self.file = output / "read_receipt.json"
        self.value = read(self.file) if self.file.exists() else {"contract_sha256": digest(contract),
            **{name: 0 for name in ZERO + ACTUAL}, "attempted_reads": {}, "completed_reads": {}}
        require(self.value["contract_sha256"] == digest(contract), "Receipt contract differs")

    def bump(self, name, amount=1):
        self.value[name] += int(amount)
        # Read attempts/completions and commit boundaries are durable. Pure
        # chunk-work counters are flushed at the complete-case boundary.
        if name in ("TRAIN_input_load_attempts", "unreturned_load_receipts", "optimizer_steps",
                    "own_checkpoint_read_attempts", "own_checkpoint_reads", "own_checkpoint_writes",
                    "own_optimizer_restores", "TRAIN_metric_cases"):
            write(self.file, self.value)

    def load(self, data, manifest, key):
        self.bump("TRAIN_input_load_attempts"); self.value["attempted_reads"][key] = self.value["attempted_reads"].get(key, 0) + 1
        write(self.file, self.value)
        try:
            case = data.load_case(manifest, key, with_targets=True, device="cuda", for_fno=True)
        except Exception:
            self.bump("unreturned_load_receipts"); raise
        receipt = case["_actual_read_receipt"]
        require(receipt["DEV_target_reads"] == receipt["source_archive_reads"] == receipt["scaler_fits"] == 0,
                "Unexpected input-reader scope")
        self.value["TRAIN_input_load_completed"] += 1
        for name in ("input_array_reads", "cached_TRAIN_target_array_reads"):
            self.value[name] += receipt[name]
        self.value["completed_reads"][key] = self.value["completed_reads"].get(key, 0) + 1
        write(self.file, self.value); return case


class Clock:
    """Own cumulative wall and whole-case training clocks, including failures.

An unclosed attempt is charged until observed exit when supplied, otherwise
until restart; the conservative unknown interval is separately disclosed.
"""
    def __init__(self, output, entry_epoch, entry_monotonic):
        self.file = output / "clock.json"; now = time.time()
        self.value = read(self.file) if self.file.exists() else {"cumulative_wall_seconds": 0.,
            "training_seconds": {arm: 0. for arm in ARMS}, "phase_seconds": {}, "attempts": 0,
            "arm_wall_seconds": {arm: 0. for arm in ARMS}, "arm_phase_seconds": {arm: {} for arm in ARMS},
            "unknown_interruption_charge_seconds": 0., "active": False, "phase": "metadata", "arm": None}
        if self.value["active"]:
            observed = output / "observed_attempt_exit.json"
            end = now
            if observed.exists():
                record = read(observed)
                if record.get("pid") == self.value.get("pid") and isinstance(record.get("exit_epoch"), (float, int)):
                    end = min(now, max(self.value["last_epoch"], float(record["exit_epoch"])))
            gap = max(0., end - self.value["last_epoch"]); self.charge(gap)
            self.value["unknown_interruption_charge_seconds"] += gap
        self.last = time.monotonic(); self.value["phase"] = "metadata"; self.value["arm"] = None
        self.charge(max(0., self.last - entry_monotonic))
        self.value.update(active=True, closed=False, attempts=self.value["attempts"] + 1, pid=os.getpid(),
                          attempt_started_epoch=entry_epoch, last_epoch=now); self.persist()

    def charge(self, seconds):
        self.value["cumulative_wall_seconds"] += seconds
        name = self.value["phase"]; self.value["phase_seconds"][name] = self.value["phase_seconds"].get(name, 0.) + seconds
        if self.value.get("arm") in ARMS:
            arm = self.value["arm"]; self.value["arm_wall_seconds"][arm] += seconds
            phases = self.value["arm_phase_seconds"][arm]; phases[name] = phases.get(name, 0.) + seconds
        if name == "training" and self.value.get("arm") in ARMS:
            self.value["training_seconds"][self.value["arm"]] += seconds

    def tick(self):
        now = time.monotonic(); self.charge(max(0., now - self.last)); self.last = now; self.value["last_epoch"] = time.time()

    def persist(self):
        write(self.file, self.value)

    def phase(self, name, arm=None):
        self.tick(); self.value.update(phase=name, arm=arm); self.persist()

    def guard(self):
        self.tick()
        if self.value["cumulative_wall_seconds"] >= 9000:
            raise Budget("whole_campaign_wall_budget_exhausted")

    def close(self):
        self.tick(); self.value.update(active=False, closed=True, phase="terminal", arm=None, closed_at_utc=utc()); self.persist()


def seed_all(np, torch):
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)


def rng_state(np, torch):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(np, torch, state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu()); torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def model_class(torch):
    from .conditioned_field import PointNetGeometryEncoder
    from .fno import FNO3d, query_grid
    nn = torch.nn
    class FNOQueryField(nn.Module):
        def __init__(self, arm):
            super().__init__(); self.arm = arm
            self.encoder = PointNetGeometryEncoder(32, 64)
            self.fno = FNO3d(in_channels=61, out_channels=16, width=16, modes=12, layers=4,
                           padding=0, head_width=128, checkpoint_blocks=True)
            self.head = nn.Sequential(nn.Linear(76, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 4))

        def encode_case(self, case):
            context = self.encoder(case["cloud"], case["mask"])
            grid = torch.cat((case["grid"], context[:, :, None, None, None].expand(1, 32, 64, 64, 64)), dim=1)
            return context, self.fno(grid)

        def query(self, case, start, stop, context, latent_grid):
            indices = case["query_indices"][start:stop]; weights = case["query_weights"][start:stop]
            latent = query_grid(latent_grid[0], indices, weights, CHUNK)
            local = query_grid(case["grid"][0, 4:29], indices, weights, CHUNK) if self.arm == ARMS[0] else case["features"][start:stop]
            return self.head(torch.cat((case["xyz"][start:stop], local, context.expand(stop-start, -1), latent), dim=1))

        def counts(self):
            groups = {"PointNet": self.encoder, "FNO_latent": self.fno, "query_head": self.head}
            counts = {name: sum(p.numel() for p in part.parameters()) for name, part in groups.items()}
            return {**counts, "total": sum(counts.values()), "all_parameters_real": True}
    return FNOQueryField


def gradient_check(torch, model, context_leaf, latent_leaf):
    missing = []; norms = {}; finite = True
    for name, part in (("PointNet", model.encoder), ("FNO_latent", model.fno), ("query_head", model.head)):
        energy = 0.
        for pname, p in part.named_parameters():
            if p.grad is None:
                missing.append(name + "." + pname); continue
            finite = finite and bool(torch.isfinite(p.grad).all())
            energy += float(p.grad.detach().double().square().sum())
        norms[name] = math.sqrt(energy)
    leaf_norms = {}
    for name, leaf in (("context", context_leaf), ("latent_grid", latent_leaf)):
        require(leaf.grad is not None, "Missing " + name + " leaf gradient", FloatingPointError)
        finite = finite and bool(torch.isfinite(leaf.grad).all())
        leaf_norms[name] = float(leaf.grad.detach().double().norm())
    passed = finite and not missing and all(v > 0 for v in list(norms.values()) + list(leaf_norms.values()))
    result = {"passed": passed, "all_present_gradients_finite": finite, "missing_gradient_names": missing,
        "group_L2_norms": norms, "leaf_L2_norms": leaf_norms,
        "trainable_tensors": sum(1 for _ in model.parameters()), "PointNet_encodes_per_case": 1,
        "FNO_encodes_per_case": 1, "original_joint_encode_backward_calls_per_case": 1,
        "source": "first actual complete original-node TRAIN update before Adam; no toy or probe"}
    require(passed, "First actual complete-case gradient acceptance failed", FloatingPointError); return result


def train_case(torch, model, case, ledger, clock, first=False):
    count = len(case["xyz"])
    require(case["xyz"].shape == (count, 3) and case["features"].shape == (count, 25)
            and case["target"].shape == (count, 4) and count > 0, "Original-node TRAIN schema differs")
    context, latent = model.encode_case(case); ledger.bump("geometry_encoder_calls"); ledger.bump("FNO_fullcase_calls")
    require(context.shape == (1, 32) and latent.shape == (1, 16, 64, 64, 64), "Context/latent schema differs")
    context_leaf = context.detach().requires_grad_(True); latent_leaf = latent.detach().requires_grad_(True)
    total = 0.
    for start in range(0, count, CHUNK):
        clock.guard(); stop = min(count, start + CHUNK)
        prediction = model.query(case, start, stop, context_leaf, latent_leaf); ledger.bump("query_chunk_calls")
        require(prediction.shape == (stop-start, 4), "All-node query output schema differs")
        loss = (prediction.float() - case["target"][start:stop]).square().mean() * ((stop-start) / count)
        require(bool(torch.isfinite(loss)), "Nonfinite complete-case loss", FloatingPointError)
        loss.backward(); ledger.bump("query_chunk_backward_calls"); total += float(loss.detach())
    require(context_leaf.grad is not None and latent_leaf.grad is not None
            and bool(torch.isfinite(context_leaf.grad).all()) and bool(torch.isfinite(latent_leaf.grad).all()),
            "Original encode chain-rule gradient missing/nonfinite", FloatingPointError)
    # The FNO's graph also depends on original context. One joint backward
    # accumulates its contribution and the direct query-context contribution.
    torch.autograd.backward((context, latent), (context_leaf.grad, latent_leaf.grad))
    ledger.bump("original_encode_backward_calls")
    check = gradient_check(torch, model, context_leaf, latent_leaf) if first else None
    return total, check


def full_readout(np, torch, model, manifest, data, scaler, ledger, clock, folder, step):
    from .coverage_comparison_train import physical_metrics, aggregate
    filename = folder / f"TRAIN120_step{step}.json"
    if filename.exists():
        previous = read(filename)
        if previous.get("complete120") is True:
            return previous
    previous_rng = rng_state(np, torch); model.eval(); per = {key: None for key in manifest["train_keys"]["40"]}
    completed = 0; reason = None; began = time.monotonic(); clock.phase("TRAIN_full120_readout", model.arm)
    try:
        with torch.no_grad():
            for key in per:
                clock.guard(); ledger.bump("TRAIN_readout_case_attempts"); case = ledger.load(data, manifest, key)
                context, latent = model.encode_case(case); ledger.bump("geometry_encoder_calls"); ledger.bump("FNO_fullcase_calls")
                prediction = np.empty((len(case["xyz"]), 4), dtype=np.float32)
                for start in range(0, len(prediction), CHUNK):
                    clock.guard(); stop = min(len(prediction), start + CHUNK)
                    part = model.query(case, start, stop, context, latent); ledger.bump("query_chunk_calls")
                    require(bool(torch.isfinite(part).all()), "Nonfinite fullN readout", FloatingPointError)
                    prediction[start:stop] = part.cpu().numpy()
                require(int(torch.cuda.max_memory_allocated()) <= 24 * 1024**3,
                        "Observed readout GPU memory cap exceeded", Budget)
                ledger.bump("TRAIN_fullN_predictions")
                physical = scaler.decode(prediction); per[key] = physical_metrics(np, case["physical"], physical)
                completed += 1; ledger.bump("TRAIN_metric_cases")
                del case, context, latent, prediction, physical
    except Budget as error:
        reason = str(error)
    finally:
        torch.cuda.synchronize(); restore_rng(np, torch, previous_rng); clock.phase("metadata", model.arm)
    result = {"status": "completed", "step": step, "arm": model.arm, "complete120": completed == 120,
        "completed_cases": completed, "fixed_denominator": 120, "aggregate": aggregate(per, 120),
        "readout_seconds": time.monotonic() - began, "terminal_negative_reason": reason,
        "prediction_scope": "All original nodes; no subsampling or DEV; U mm and signed S MPa"}
    write(filename, result); return result


def paired_gate(arms, same_initial):
    complete = same_initial and all(arms.get(name, {}).get("completed4800") is True
            and (arms[name].get("final_TRAIN120") or {}).get("complete120") is True
            and (arms[name].get("first_actual_gradient_check") or {}).get("passed") is True
            and len(arms[name].get("case_exposure_counts") or {}) == 120
            and all(value == 40 for value in arms[name]["case_exposure_counts"].values()) for name in ARMS)
    ratios = {}; passed = complete
    for name, maximum in (("case_mean_stress_relative_l2", .8), ("case_mean_u_relative_l2", 1.05),
                          ("worst_case_stress_relative_l2", 1.), ("worst_case_stress_hotspot_relative_l2", 1.)):
        a = (arms.get(ARMS[0], {}).get("final_TRAIN120") or {}).get("aggregate", {}).get(name)
        b = (arms.get(ARMS[1], {}).get("final_TRAIN120") or {}).get("aggregate", {}).get(name)
        valid = isinstance(a, (int, float)) and math.isfinite(a) and a > 0 and isinstance(b, (int, float)) and math.isfinite(b) and b >= 0
        value = b / a if valid else None; ratios[name] = {"candidate_to_control": value, "maximum": maximum}
        passed = passed and value is not None and value <= maximum
    return {"passed": bool(passed), "both_arms_completed4800_and_full120": bool(complete),
        "same_initial_state_proven": same_initial, "ratios": ratios, "DEV_qualified": False,
        "final_solver_selection": False, "scope": "TRAIN representation advance gate only; no generalization or causal family claim"}


def run(plan_path=PLAN, output=None, start=False):
    if not start:
        return {"status": "source_only_ready", "kind": KIND, "writes": False, "actual_execution": False,
                "API": "request_contract(plan)->(contract,plan); run(...,start=True) requires authentic release"}
    entry_epoch, entry_monotonic = time.time(), time.monotonic()
    contract, plan = request_contract(plan_path); release(contract, plan); proof = predecessor_gate(plan)
    output = path(output or plan["output"])
    output.relative_to(ROOT.resolve())
    if (output / "summary.json").exists():
        saved = read(output / "summary.json")
        if saved.get("status") == "completed":
            require(saved["config"]["contract"] == contract, "Completed pair contract differs"); return saved
    from .coverage_comparison_train import lock
    with lock(plan["shared_GPU_lock"]):
        proof = predecessor_gate(plan); output.mkdir(parents=True, exist_ok=True)
        clock = Clock(output, entry_epoch, entry_monotonic); ledger = Ledger(output, contract)
        config = {"contract": contract, "TRAIN_keys": contract["TRAIN_keys"], "validation_keys": [],
                  "arms": list(ARMS), "scaler": contract["scaler"], "case_order_sha256": contract["case_order_sha256"]}
        if (output / "config.json").exists():
            require(read(output / "config.json") == config, "Resume science/input/source/config differs")
        write(output / "config.json", config); write(output / "predecessor_proof.json", proof)
        state = {"kind": KIND, "pid": os.getpid(), "status": "running", "phase": "initialization", "updated_at_utc": utc()}
        write(output / "status.json", state); arms = {}; common = None; error_record = None; torch = None
        try:
            clock.guard(); clock.phase("imports_CUDA_initialization")
            import numpy as np
            import torch
            from . import coverage_comparison_data as data
            from .coverage_comparison_train import case_schedule, initial_identity, save_checkpoint
            from .benchmark import FieldScaler
            manifest = data.read_bootstrap_TRAIN40(plan["parent_plan"]); order = case_schedule(contract["TRAIN_keys"])
            torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True; torch.cuda.init()
            scaler = FieldScaler(**manifest["scaler"]); cls = model_class(torch)
            for arm in ARMS:
                folder = output / arm; folder.mkdir(parents=True, exist_ok=True)
                if (folder / "summary.json").exists() and read(folder / "summary.json").get("status") == "completed":
                    previous = read(folder / "summary.json"); require(previous["config"]["contract"] == contract, "Completed arm contract differs")
                    arms[arm] = previous; common = previous["initial_parameters"] if common is None else common
                    continue
                clock.guard(); clock.phase("model_initialization", arm); seed_all(np, torch)
                torch.cuda.reset_peak_memory_stats(); model = cls(arm).cuda(); ledger.bump("model_initializations")
                optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0., fused=True)
                identity = initial_identity(model); ledger.bump("initial_parameter_hash_reads")
                if common is None:
                    common = read(output / "initial_common.json") if (output / "initial_common.json").exists() else identity
                    require(common == identity, "First arm common initialization changed"); write(output / "initial_common.json", common)
                require(identity == common, "Pair named initial tensors are not identical")
                counts = model.counts(); arm_config = {**config, "arm": arm, "parameter_counts": counts,
                    "initial_parameters": identity, "model": plan["scientific"]["model"],
                    "local25_source": "input_grid_channels4:29_query_roundtrip" if arm == ARMS[0] else "original_node_features25",
                    "precision": "FP32/TF32 high; no autocast", "loss": "allN encoded4 MSE with exact node-weighted query chunks"}
                if (folder / "config.json").exists():
                    require(read(folder / "config.json") == arm_config, "Arm resume configuration/initial state differs")
                write(folder / "config.json", arm_config)
                step = 0; exposures = {key: 0 for key in contract["TRAIN_keys"]}; node_exposures = 0
                first = None; last_loss = None; endpoints = {}; reason = None; prior_peak = 0; prior_reserved = 0
                checkpoint_seconds = 0.; restore_seconds = 0.
                if (folder / "last.pt").exists():
                    clock.phase("checkpoint_load_restore", arm); t = time.monotonic(); ledger.bump("own_checkpoint_read_attempts")
                    saved = torch.load(folder / "last.pt", map_location="cuda", weights_only=False); ledger.bump("own_checkpoint_reads")
                    require(saved["config"] == arm_config and saved["case_order_sha256"] == digest(order)
                            and 0 <= saved["step"] <= 4800 and saved["next_case_position"] == saved["step"], "Checkpoint identity/cursor differs")
                    model.load_state_dict(saved["model"], strict=True); optimizer.load_state_dict(saved["optimizer"])
                    ledger.bump("own_optimizer_restores"); restore_rng(np, torch, saved["rng"])
                    step, exposures, node_exposures = saved["step"], saved["case_exposure_counts"], saved["node_exposures"]
                    first, last_loss, endpoints = saved["first_actual_gradient_check"], saved["last_loss"], saved["endpoints"]
                    checkpoint_seconds = saved["checkpoint_seconds"]; restore_seconds = saved["restore_seconds"]
                    prior_peak, prior_reserved = saved["memory"]["peak_allocated_bytes"], saved["memory"]["peak_reserved_bytes"]
                    require(sum(exposures.values()) == step, "Checkpoint exposure sum differs")
                    metadata = read(folder / "checkpoint.json")
                    checkpoint_seconds = max(checkpoint_seconds, metadata.get("checkpoint_seconds", 0.))
                    restore_seconds = max(restore_seconds, metadata.get("restore_seconds", 0.)) + time.monotonic() - t
                    del saved
                def memory():
                    return {"peak_allocated_bytes": max(prior_peak, int(torch.cuda.max_memory_allocated())),
                            "peak_reserved_bytes": max(prior_reserved, int(torch.cuda.max_memory_reserved()))}
                def commit(endpoint=False):
                    nonlocal checkpoint_seconds
                    clock.phase("checkpoint_persistence", arm); t = time.monotonic()
                    payload = {"config": arm_config, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "rng": rng_state(np, torch), "step": step, "next_case_position": step,
                        "pass_index": step // 120, "pass_offset": step % 120, "case_order_sha256": digest(order),
                        "case_exposure_counts": exposures, "node_exposures": node_exposures,
                        "first_actual_gradient_check": first, "last_loss": last_loss, "endpoints": endpoints,
                        "checkpoint_seconds": checkpoint_seconds, "restore_seconds": restore_seconds, "memory": memory()}
                    save_checkpoint(torch, folder / "last.pt", payload); ledger.bump("own_checkpoint_writes")
                    if endpoint:
                        save_checkpoint(torch, folder / f"step{step}.pt", payload); ledger.bump("own_checkpoint_writes")
                    checkpoint_seconds += time.monotonic() - t; clock.tick(); clock.persist()
                    write(folder / "checkpoint.json", {"status": "completed", "step": step, "pass_index": step // 120,
                        "pass_offset": step % 120, "config_sha256": digest(arm_config), "case_order_sha256": digest(order),
                        "checkpoint_seconds": checkpoint_seconds, "restore_seconds": restore_seconds,
                        "case_exposure_counts": dict(exposures), "node_exposures": node_exposures,
                        "memory": memory(), "path": str(folder / "last.pt"), "bytes": (folder / "last.pt").stat().st_size})
                if step == 0:
                    endpoints["0"] = full_readout(np, torch, model, manifest, data, scaler, ledger, clock, folder, 0); commit(True)
                in_update = False; optimizer_started = False
                try:
                    while step < 4800:
                        clock.guard()
                        if clock.value["training_seconds"][arm] >= 3600:
                            reason = "incomplete_training_cost_limit"; break
                        key = order[step]
                        state.update(phase="training", arm=arm, step=step, pass_index=step // 120, pass_offset=step % 120,
                                     case_key=key, updated_at_utc=utc()); write(output / "status.json", state)
                        clock.phase("training", arm); in_update = True; optimizer_started = False
                        model.train(); optimizer.zero_grad(set_to_none=True)
                        case = ledger.load(data, manifest, key); loss, check = train_case(torch, model, case, ledger, clock, first is None)
                        if check is not None:
                            first = {"step": step + 1, "case": key, **check}; write(folder / "first_actual_gradient_check.json", first)
                        optimizer_started = True; optimizer.step(); torch.cuda.synchronize(); ledger.bump("optimizer_steps")
                        step += 1; exposures[key] += 1; node_exposures += len(case["xyz"]); last_loss = loss
                        in_update = False; clock.phase("metadata", arm)
                        require(memory()["peak_allocated_bytes"] <= 24 * 1024**3, "Observed GPU memory cap exceeded", Budget)
                        with (folder / "loss.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps({"step": step, "case": key, "nodes": len(case["xyz"]),
                                "encoded4_case_mean_loss": loss, "training_seconds": clock.value["training_seconds"][arm],
                                "lr": optimizer.param_groups[0]["lr"], "attempt": clock.value["attempts"]}, allow_nan=False) + "\n")
                        del case
                        if step % 25 == 0:
                            commit()
                except Budget as error:
                    reason = str(error)
                if in_update and optimizer_started:
                    raise RuntimeError("Uncommitted optimizer state; retain prior checkpoint")
                if step % 25 != 0 or in_update:
                    commit()
                clock.guard()
                endpoints[str(step)] = full_readout(np, torch, model, manifest, data, scaler, ledger, clock, folder, step)
                commit(step == 4800)
                if step == 4800:
                    require(exposures == {key: 40 for key in exposures}, "Full balanced exposure differs")
                final = endpoints[str(step)]
                record = {"status": "completed", "kind": KIND, "config": arm_config, "arm": arm,
                    "endpoint_step": step, "completed4800": step == 4800, "stop_reason": reason or "fixed_updates_completed",
                    "initial_parameters": identity, "parameter_counts": counts, "first_actual_gradient_check": first,
                    "case_exposure_counts": exposures, "case_order_sha256": digest(order), "node_exposures": node_exposures,
                    "endpoints": endpoints, "final_TRAIN120": final, "memory": memory(),
                    "costs": {"training_seconds": clock.value["training_seconds"][arm], "checkpoint_seconds": checkpoint_seconds,
                              "restore_seconds": restore_seconds, "arm_wall_seconds": clock.value["arm_wall_seconds"][arm],
                              "phase_seconds": dict(clock.value["arm_phase_seconds"][arm]),
                              "TRAIN_readout_seconds": clock.value["arm_phase_seconds"][arm].get("TRAIN_full120_readout", 0.),
                              "training_budget_overshoot_seconds": max(0., clock.value["training_seconds"][arm] - 3600)},
                    "read_receipt": dict(ledger.value), "no_DEV_model_selection": True}
                write(folder / "summary.json", record); arms[arm] = record
                write(folder / "status.json", {"status": "completed", "phase": "terminal", "pid": os.getpid(), "step": step})
                del model, optimizer; torch.cuda.empty_cache(); clock.phase("metadata")
        except Exception as error:
            error_record = {"type": type(error).__name__, "reason": str(error)}
            bounded = isinstance(error, (Budget, FloatingPointError)) or "out of memory" in str(error).lower()
            state["status"] = "completed" if bounded else "failed"
        else:
            state["status"] = "completed"
        finally:
            if torch is not None:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            clock.close()
            write(ledger.file, ledger.value)
        for arm in ARMS:
            if arm not in arms:
                folder = output / arm
                checkpoint = read(folder / "checkpoint.json") if (folder / "checkpoint.json").exists() else None
                arm_configuration = read(folder / "config.json") if (folder / "config.json").exists() else None
                committed_step = checkpoint["step"] if checkpoint else None
                final_readout_path = folder / f"TRAIN120_step{committed_step}.json"
                final_readout = read(final_readout_path) if final_readout_path.exists() else None
                arms[arm] = {"status": "completed" if state["status"] == "completed" else "failed", "arm": arm,
                    "config": arm_configuration or {"contract": contract}, "completed4800": committed_step == 4800,
                    "endpoint_step": committed_step, "final_TRAIN120": final_readout,
                    "initial_parameters": arm_configuration.get("initial_parameters") if arm_configuration else None,
                    "parameter_counts": arm_configuration.get("parameter_counts") if arm_configuration else None,
                    "first_actual_gradient_check": read(folder / "first_actual_gradient_check.json")
                        if (folder / "first_actual_gradient_check.json").exists() else None,
                    "case_exposure_counts": checkpoint.get("case_exposure_counts") if checkpoint else None,
                    "node_exposures": checkpoint.get("node_exposures") if checkpoint else None,
                    "last_checkpoint_metadata": checkpoint, "memory": checkpoint.get("memory") if checkpoint else None,
                    "costs": {"training_seconds": clock.value["training_seconds"][arm],
                              "arm_wall_seconds": clock.value["arm_wall_seconds"][arm],
                              "phase_seconds": dict(clock.value["arm_phase_seconds"][arm])},
                    "stop_reason": "terminal_incomplete_no_rescue", "error": error_record}
        same_initial = common is not None and all(arms[a].get("initial_parameters") == common for a in ARMS)
        gate = paired_gate(arms, same_initial)
        gate["within_frozen_training_and_wall_budget"] = clock.value["cumulative_wall_seconds"] < 9000 and all(
            clock.value["training_seconds"][arm] <= 3600 for arm in ARMS)
        gate["passed"] = gate["passed"] and state["status"] == "completed" and gate["within_frozen_training_and_wall_budget"]
        summary = {"status": state["status"], "kind": KIND, "config": config, "TRAIN_keys": contract["TRAIN_keys"],
            "validation_keys": [], "arms": arms, "fixed_arm_denominator": 2, "fixed_TRAIN_denominator": 120,
            "same_initial_state_proven": same_initial, "completion_gate": gate,
            "predecessor_proof": proof, "read_receipt": ledger.value, "costs": clock.value,
            "elapsed_seconds": clock.value["cumulative_wall_seconds"], "error": error_record,
            "budget_overshoot_seconds": max(0., clock.value["cumulative_wall_seconds"] - 9000),
            "scope": "TRAIN-only representation comparison; no DEV/held5 reads; not solver selection or continuous-gradient qualification"}
        write(output / "summary.json", summary); state.update(phase="terminal", updated_at_utc=utc()); write(output / "status.json", state)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN); parser.add_argument("--output"); parser.add_argument("--start", action="store_true")
    args = parser.parse_args(); result = run(args.plan, args.output, start=args.start)
    print(json.dumps({"status": result["status"], "kind": result["kind"], "actual_execution": bool(args.start)}, ensure_ascii=False))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
