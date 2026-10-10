"""Frozen full-FE physical/local versus global-latent signed-S component pair.

Only metadata is touched by request_contract. Scientific packages, cached arrays,
labels and owned checkpoints are accessible only after the genuine root release.
No physical solve/recovery, displacement target or external learned state is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "geobc/elastic_spatial_recovery.py"
PLAN = "notes/geobc_spatial_physical_recovery_plan.json"
DESIGN = "notes/geobc_spatial_physical_recovery_design.json"
KIND = "spatial_physical_signed_recovery_pair"
TRAIN = ("101_428/ver", "101_56/ver", "101_78/ver")
DEV = ("102_249/ver", "102_38/ver")
CASES = TRAIN + DEV
ARMS = ("local_physical_zero_latent", "global_FNO_physical_latent")
ZERO_KEYS = ("displacement_target_reads", "TRAIN_displacement_target_reads",
    "DEV_displacement_target_reads", "source_csv_or_archive_reads", "canonical_target_member_reads",
    "old_final_reads", "sealed_new_test_reads", "DEV_dia_reads", "new_physical_solves",
    "new_tensor_recoveries", "K_guards_repeated", "external_model_or_weight_reads", "DEV_scaler_fits")
ACTUAL_KEYS = ("GT_reads", "stress_reads", "TRAIN_stress_reads", "DEV_stress_reads",
    "physical_U_input_reads", "cached_input_array_reads", "scaler_fits", "optimizer_steps",
    "own_checkpoint_reads", "own_checkpoint_writes", "own_evaluation_cache_reads",
    "own_prediction_array_reads", "physical_U_copy_reads", "cached_stress_array_reads")
SCIENCE_KEYS = ("train_keys", "validation_keys", "arms", "input_interface", "recipe", "gate", "budget", "read_policy")


def path(value):
    value = str(value)
    if os.name == "nt" and value.startswith("/mnt/"):
        value = value[5].upper() + ":/" + value[7:]
    item = Path(value)
    return item if item.is_absolute() else ROOT / item


def read(value):
    return json.loads(path(value).read_text(encoding="utf-8"))


def write(value, data):
    destination = path(value); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def sha(value):
    h = hashlib.sha256()
    with path(value).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stat_match(record):
    info = path(record["path"]).stat()
    return info.st_size == record["bytes"] and info.st_mtime_ns == record["mtime_ns"]


def request_contract(plan_path=PLAN):
    """Source/small-JSON/stat only: does not import scientific modules or read arrays."""
    plan = read(plan_path); design = read(DESIGN)
    require(plan["kind"] == KIND and tuple(plan["train_keys"]) == TRAIN
        and tuple(plan["validation_keys"]) == DEV and tuple(plan["arms"]) == ARMS,
        "Frozen structural pair scope changed")
    science = {name: design[name] for name in SCIENCE_KEYS}
    require(plan["scientific_digest"] == digest(science) and plan["science_recipe"] == science,
        "Frozen structural recipe changed")
    require(plan["actual_execution_released"] is False and plan["case_replacement"] is False,
        "Release belongs to real root marker; no case replacement")
    sources = {name: sha(name) for name in plan["source_sha256"]}
    require(sources == plan["source_sha256"] and SOURCE in sources, "Frozen source identity differs")
    for name in ("parent_readout_summary", "parent_magnitude_plan"):
        require(sha(design[name]["path"]) == design[name]["sha256"], "Parent metadata identity differs")
    parent = read(design["parent_readout_summary"]["path"])
    require(parent["status"] == "completed" and parent["scalers"] == plan["parent_scalers"],
        "Accepted TRAIN-only scaler metadata differs")
    scalers = parent["scalers"]
    require(tuple(scalers["tensor6"]["fit_keys"]) == TRAIN
        and tuple(scalers["signed_stress"]["fit_keys"]) == TRAIN, "TRAIN-only parent scaler scope differs")
    inherited = 0.0; records = design["records"]
    require(records == plan["records"], "Accepted immutable input identities differ")
    for key in CASES:
        item = records[key]
        for record in list(item["input_arrays"].values()) + [item["original_zero_physical_U"], item["cached_original_signed_stress"]]:
            require(stat_match(record), "Accepted cache stat differs: " + key)
        require(sha(item["original_fine_summary"]["path"]) == item["original_fine_summary"]["sha256"],
            "Original numerical endpoint identity differs")
        original = read(item["original_fine_summary"]["path"])
        require(original["status"] == "completed" and original["numerically_converged"] is True
            and original["true_relative_residual"] <= 1e-8, "Original zero field was not converged")
        for manifest in ("operator_summary", "input_manifest"):
            record = item["physical_feature_provenance"][manifest]
            require(sha(record["path"]) == record["sha256"], "Accepted physical metadata differs")
        inherited += item["accepted_physical_GT_free_cold_seconds"] + item["accepted_feature_preparation_costs"]["elapsed_seconds"]
    require(abs(inherited - 457.480465417) < 1e-8 and abs(plan["inherited_seconds"] - inherited) < 1e-8,
        "Inherited original physics/feature accounting differs")
    return {"schema": "geobc_spatial_physical_pair_v1", "kind": KIND,
        "train_keys": list(TRAIN), "validation_keys": list(DEV), "case_keys": list(CASES), "arms": list(ARMS),
        "records": records, "parent_scalers": scalers, "source_sha256": sources,
        "scientific_digest": digest(science), "science_recipe_sha256": digest(science),
        "science_recipe": science, "inherited_seconds": inherited,
        "array_or_label_reads_in_request_contract": 0, "output": str(path(plan["output"]))}, plan


class ScientificNegative(RuntimeError):
    pass


class Clock:
    """Persistent stage ledger, inherited cost once, conservative unobserved exit gap."""
    def __init__(self, output, inherited, invocation_start):
        self.output = output; self.file = output / "clock.json"; self.inherited = inherited
        previous = read(self.file) if self.file.exists() else {}
        self.prior = float(previous.get("new_actual_wall_seconds", 0))
        self.gap = float(previous.get("unobserved_exit_gap_seconds", 0))
        if previous.get("active"):
            extra = max(0.0, time.time() - previous["observed_unix_seconds"])
            self.prior += extra; self.gap += extra
        self.start = invocation_start; self.torch = None
        self.attempts = int(previous.get("attempts", 0)) + 1
        self.peak_allocated = int(previous.get("peak_GPU_allocated_bytes", 0))
        self.peak_reserved = int(previous.get("peak_GPU_reserved_bytes", 0))
        self.persist(True)

    def sync(self):
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def new_seconds(self):
        self.sync(); return self.prior + time.perf_counter() - self.start

    def total(self):
        return self.inherited + self.new_seconds()

    def persist(self, active=True):
        if self.torch is not None and self.torch.cuda.is_available():
            self.peak_allocated = max(self.peak_allocated, self.torch.cuda.max_memory_allocated())
            self.peak_reserved = max(self.peak_reserved, self.torch.cuda.max_memory_reserved())
        write(self.file, {"inherited_original_physics_feature_seconds": self.inherited,
            "new_actual_wall_seconds": self.new_seconds(), "campaign_accounted_wall_seconds": self.total(),
            "unobserved_exit_gap_seconds": self.gap, "attempts": self.attempts,
            "peak_GPU_allocated_bytes": self.peak_allocated, "peak_GPU_reserved_bytes": self.peak_reserved,
            "active": active, "observed_unix_seconds": time.time()})

    def guard(self):
        self.persist()
        if self.total() >= 1800:
            raise ScientificNegative("Fixed cumulative campaign wall budget exhausted")
        if max(self.peak_allocated, self.peak_reserved) > 24 * 1024 ** 3:
            raise ScientificNegative("Fixed 24GiB GPU peak budget exceeded")


class Ledger:
    def __init__(self, output):
        self.output = output; self.file = output / "read_receipt.json"
        self.data = read(self.file) if self.file.exists() else {name: 0 for name in ZERO_KEYS + ACTUAL_KEYS}
        for key in ZERO_KEYS:
            require(type(self.data.get(key)) is int and self.data[key] == 0, "Forbidden read receipt differs")
        self.data.setdefault("attempted_reads", {}); self.data.setdefault("completed_reads", {})
        self.persist()

    def persist(self):
        write(self.file, self.data)

    def bump(self, key, token=None):
        self.data[key] += 1
        if token is not None:
            self.data["attempted_reads"][token] = self.data["attempted_reads"].get(token, 0) + 1
        self.persist()

    def complete(self, token):
        self.data["completed_reads"][token] = self.data["completed_reads"].get(token, 0) + 1
        self.persist()


def array_load(np, record, ledger, key, token, u_file_sha=None):
    ledger.bump(key, token); require(stat_match(record), "Accepted input cache changed")
    if u_file_sha is not None:
        require(sha(record["path"]) == u_file_sha, "Immutable original U bytes changed")
    result = np.load(path(record["path"]), allow_pickle=False)
    dtype = "float64" if record["dtype"] == "torch.float64" else record["dtype"]
    require(list(result.shape) == record["shape"] and str(result.dtype) == dtype and np.isfinite(result).all(),
        "Accepted input array schema/nonfinite mismatch")
    if "values_sha256" in record:
        require(hashlib.sha256(np.ascontiguousarray(result).tobytes()).hexdigest() == record["values_sha256"],
            "Accepted cached values changed")
    ledger.complete(token); return result


def own_checkpoint(torch, destination, payload, ledger):
    ledger.bump("own_checkpoint_writes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, destination)


def owned_load(torch, destination, ledger):
    ledger.bump("own_checkpoint_reads", "own:" + str(destination))
    result = torch.load(destination, map_location="cpu", weights_only=False)
    ledger.complete("own:" + str(destination)); return result


def model_class(torch):
    from .conditioned_field import PointNetGeometryEncoder
    from .fno import FNO3d
    nn = torch.nn
    class PairDecoder(nn.Module):
        def __init__(self, global_arm=False):
            super().__init__(); self.encoder = PointNetGeometryEncoder(32, 64)
            layers = []; width = 86
            for _ in range(4):
                layers.extend((nn.Linear(width, 256), nn.ReLU())); width = 256
            layers.append(nn.Linear(256, 1)); self.head = nn.Sequential(*layers)
            self.global_fno = None
            self.is_global = global_arm

        def add_global(self):
            self.global_fno = FNO3d(in_channels=71, out_channels=16, width=16, modes=12, layers=4,
                padding=0, head_width=128, checkpoint_blocks=True)

        def points(self, item, start, end, context, latent):
            return self.head(torch.cat((item["xyz"][start:end], item["features"][start:end],
                item["u_normalized"][start:end], item["sigma_normalized"][start:end],
                item["valid"][start:end, None], context.expand(end-start, -1), latent[start:end]), dim=1)).squeeze(-1)

        def fields(self, item):
            from .fno import query_grid
            context = self.encoder(item["cloud"][None], item["mask"][None])
            if self.global_fno is None:
                latent = item["xyz"].new_zeros((len(item["xyz"]), 16))
            else:
                grid = torch.cat((item["grid"], context[0, :, None, None, None].expand(32, 64, 64, 64)), dim=0)[None]
                latent = query_grid(self.global_fno(grid)[0], item["query_indices"], item["query_weights"])
            return context, latent
    return PairDecoder


def common_identity(torch, state):
    values = {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, value in state.items()}
    return {"named_sha256": values, "all_sha256": digest(values)}


def create_model(np, torch, cls, arm, common):
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
    model = cls(arm == ARMS[1]); model.load_state_dict(common, strict=True)
    actual_common = common_identity(torch, model.state_dict())
    if arm == ARMS[1]:
        # Common state is already cloned; additional independent global creation follows it.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(2026); torch.cuda.manual_seed_all(2026); model.add_global()
    return model.cuda(), actual_common


def prepare_inputs(np, contract, ledger, output, clock):
    started = time.perf_counter(); items = {}
    for key in CASES:
        clock.guard(); record = contract["records"][key]; t0 = time.perf_counter()
        item = {name: array_load(np, value, ledger, "cached_input_array_reads", key + ":" + name)
            for name, value in record["input_arrays"].items()}
        item["physical_u_mm"] = array_load(np, record["original_zero_physical_U"], ledger,
            "physical_U_input_reads", key + ":original_U", record["physical_U_file_sha256"])
        item["cached_input_wall_seconds"] = time.perf_counter() - t0; items[key] = item
    scaler_file = output / "scalers.json"
    if scaler_file.exists():
        scalers = read(scaler_file)
        require(scalers["parent_scalers"] == contract["parent_scalers"] and scalers["fit_keys"] == list(TRAIN),
            "Owned shared TRAIN scaler differs")
    else:
        clock.guard(); ledger.bump("scaler_fits"); fit_start = time.perf_counter()
        count = sum(len(items[key]["physical_u_mm"]) for key in TRAIN)
        sums = sum((np.square(items[key]["physical_u_mm"]).sum(axis=0) for key in TRAIN), np.zeros(3, dtype=np.float64))
        rms = np.maximum(np.sqrt(sums / count), 1e-12)
        scalers = {"parent_scalers": contract["parent_scalers"], "fit_keys": list(TRAIN),
            "physical_U_zero_center_component_RMS_mm": rms.tolist(), "physical_U_floor_mm": 1e-12,
            "TRAIN_nodes": count, "new_scaler_fits": 1, "DEV_scaler_fits": 0,
            "new_scaler_fit_wall_seconds": time.perf_counter() - fit_start}
        write(scaler_file, scalers)
    sigma = scalers["parent_scalers"]["tensor6"]
    for key, item in items.items():
        t0 = time.perf_counter()
        item["u_normalized"] = (item["physical_u_mm"] / np.array(scalers["physical_U_zero_center_component_RMS_mm"])).astype(np.float32)
        normalized = ((item["sigma_MPa"] - np.array(sigma["mean"])) / np.array(sigma["std"])).astype(np.float32)
        normalized[~item["valid"]] = 0; item["sigma_normalized"] = normalized
        item["query_normalization_wall_seconds"] = time.perf_counter() - t0
        item["grid_cpu"] = None
    return items, scalers, time.perf_counter() - started


def gpu_item(np, torch, item, global_arm, clock):
    from .solver_selection_data import _stencil, _splat
    t0 = time.perf_counter()
    result = {name: torch.as_tensor(item[name], device="cuda") for name in
        ("xyz", "features", "cloud", "mask", "u_normalized", "sigma_normalized")}
    result["valid"] = torch.as_tensor(item["valid"], device="cuda", dtype=torch.float32)
    if global_arm:
        # Static target-free node raster and query stencil, no target/GT grid ever exists.
        if item["grid_cpu"] is None:
            build_t0 = time.perf_counter()
            xyz = result["xyz"]
            indices, weights = _stencil(xyz, 64)
            values = torch.cat((result["features"], result["u_normalized"],
                result["sigma_normalized"], result["valid"][:, None]), dim=1)
            raster, active = _splat(values, indices, weights, 64)
            axis = torch.linspace(0, 1, 64, device="cuda")
            cube = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"))
            raster = raster.T.reshape(35, 64, 64, 64)
            item["grid_cpu"] = torch.cat((active.float().reshape(1, 64, 64, 64), cube, raster), dim=0).cpu()
            item["query_indices_cpu"], item["query_weights_cpu"] = indices.cpu(), weights.cpu()
            clock.sync(); item["first_actual_targetfree_grid_build_seconds"] = time.perf_counter() - build_t0
        result["grid"] = item["grid_cpu"].cuda()
        result["query_indices"] = item["query_indices_cpu"].cuda()
        result["query_weights"] = item["query_weights_cpu"].cuda()
    clock.sync(); return result, time.perf_counter() - t0


def load_stress(np, contract, key, ledger, output, dev=False):
    # Source cached DEV labels are once-shared, only after both fixed predictions persist.
    own = output / "evaluation_labels" / (key.replace("/", "_") + ".json")
    if dev and own.exists():
        record = read(own)
        return array_load(np, record, ledger, "own_evaluation_cache_reads", "own_DEV:" + key)
    token = ("DEV:" if dev else "TRAIN:") + key
    if dev:
        require(ledger.data["attempted_reads"].get(token, 0) == 0,
            "Uncommitted prior DEV source read cannot be silently repeated")
    ledger.bump("GT_reads"); ledger.bump("stress_reads"); ledger.bump("DEV_stress_reads" if dev else "TRAIN_stress_reads")
    raw = array_load(np, contract["records"][key]["cached_original_signed_stress"], ledger,
        "cached_stress_array_reads", token)
    if dev:
        from .elastic_stress_readout import save_array
        own.parent.mkdir(parents=True, exist_ok=True)
        record = save_array(np, own.with_suffix(".npy"), raw); write(own, record)
    return raw


def check_gradients(torch, model, context_leaf, latent_leaf):
    groups = {"PointNet": model.encoder, "head": model.head}
    if model.global_fno is not None:
        groups["FNO_global"] = model.global_fno
    norms = {}
    for name, module in groups.items():
        values = list(module.parameters())
        require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in values),
            "Actual active gradient missing/nonfinite: " + name)
        norm = sum(float(p.grad.abs().double().square().sum().item()) for p in values)
        require(math.isfinite(norm) and norm > 0, "Actual active gradient zero: " + name)
        norms[name] = math.sqrt(norm)
    require(context_leaf.grad is not None and torch.isfinite(context_leaf.grad).all().item()
        and context_leaf.grad.abs().sum().item() > 0, "Actual geometry leaf gradient missing")
    if model.global_fno is not None:
        require(latent_leaf.grad is not None and torch.isfinite(latent_leaf.grad).all().item()
            and latent_leaf.grad.abs().sum().item() > 0, "Actual global latent gradient missing")
    return {"passed": True, "first_REAL_fullCAD_backward": True, "standalone_preflight": False, "group_gradient_norms": norms}


def schedule_pass(index):
    rng = random.Random(2026); order = list(TRAIN)
    for _ in range(index + 1):
        order = list(TRAIN); rng.shuffle(order)
    return order


def train_arm(np, torch, cls, arm, common, items, targets, contract, output, ledger, clock):
    from .elastic_stress_readout import rng_state, restore_rng
    folder = output / arm; folder.mkdir(parents=True, exist_ok=True)
    summary_file = folder / "training_summary.json"
    if summary_file.exists():
        summary = read(summary_file)
        require(summary["config"]["contract"] == contract and summary["complete"] is True,
            "Completed arm identity differs")
        require(summary["offset"] == 0 and summary["steps"] == 3 * summary["pass_index"]
            and all(value == summary["pass_index"] for value in summary["case_updates"].values())
            and summary["training_seconds"] >= 600 and summary["first_gradient"]["passed"] is True,
            "Completed arm is not a fixed balanced actual-training endpoint")
        return summary
    setup_t0 = time.perf_counter(); model, identity = create_model(np, torch, cls, arm, common)
    require(identity == read(output / "initial_common.json")["identity"], "Common initialization differs across arms")
    write(folder / "initial_common.json", {"identity": identity, "global_initialized_after_common_clone": True})
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0)
    config = {"contract": contract, "arm": arm}; write(folder / "config.json", config)
    counts = {"PointNet": sum(p.numel() for p in model.encoder.parameters()), "head": sum(p.numel() for p in model.head.parameters()),
        "FNO_global": 0 if model.global_fno is None else sum(p.numel() for p in model.global_fno.parameters())}
    counts["total"] = sum(counts.values())
    state = {"pass_index": 0, "offset": 0, "steps": 0, "training_seconds": 0.0,
        "case_updates": {key: 0 for key in TRAIN}, "first_gradient": None, "case_order_prefix": [],
        "checkpoint_training_seconds": 0.0}
    journal_path = folder / "status.json"; checkpoint = folder / "last.pt"
    journal = read(journal_path) if journal_path.exists() else {}
    if checkpoint.exists():
        saved = owned_load(torch, checkpoint, ledger)
        require(saved["config"] == config, "Owned arm checkpoint identity differs")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]; restore_rng(np, torch, saved["rng"])
        lost = max(0, int(journal.get("steps", 0)) - state["steps"])
        state["uncommitted_updates_replayed_after_restore"] = state.get("uncommitted_updates_replayed_after_restore", 0) + lost
        # Costs of updates after last committed model are still charged; no reset.
        state["training_seconds"] = max(state["training_seconds"], float(journal.get("training_seconds", 0)))
        if journal.get("active"):
            state["training_seconds"] += max(0, time.time() - journal["observed_unix_seconds"])
    clock.sync(); setup_seconds = time.perf_counter() - setup_t0
    last_checkpoint_seconds = state["training_seconds"]

    def persist_checkpoint():
        t0 = time.perf_counter()
        state["checkpoint_training_seconds"] = state["training_seconds"]
        own_checkpoint(torch, checkpoint, {"config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "state": dict(state), "rng": rng_state(np, torch)}, ledger)
        clock.sync(); state["training_seconds"] += time.perf_counter() - t0
        write(journal_path, {"status": "running", **state, "active": False, "observed_unix_seconds": time.time()})
        clock.persist()

    while not (state["offset"] == 0 and state["pass_index"] > 0 and state["training_seconds"] >= 600):
        clock.guard(); order = schedule_pass(state["pass_index"]); key = order[state["offset"]]
        write(journal_path, {"status": "running", **state, "active": True, "active_case": key, "observed_unix_seconds": time.time()})
        t0 = time.perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True)
        gpu, _ = gpu_item(np, torch, items[key], arm == ARMS[1], clock)
        target = torch.as_tensor(targets[key], device="cuda", dtype=torch.float32)
        context, latent = model.fields(gpu)
        context_leaf = context.detach().requires_grad_(True)
        latent_leaf = latent.detach().requires_grad_(arm == ARMS[1])
        loss_sum = 0.0; count = len(target)
        for start in range(0, count, 32768):
            end = min(start + 32768, count)
            prediction = model.points(gpu, start, end, context_leaf, latent_leaf)
            loss = (prediction - target[start:end]).square().sum() / count
            require(torch.isfinite(loss).item(), "Actual full-node encoded loss nonfinite")
            loss.backward(); loss_sum += float(loss.detach().item())
        # Joint upstream backward sums direct-local and FNO-context contributions.
        if arm == ARMS[1]:
            torch.autograd.backward((context, latent), (context_leaf.grad, latent_leaf.grad))
        else:
            context.backward(context_leaf.grad)
        if state["first_gradient"] is None:
            state["first_gradient"] = check_gradients(torch, model, context_leaf, latent_leaf)
            write(folder / "first_actual_gradient_check.json", state["first_gradient"])
        else:
            require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
                "Actual trained parameter gradient missing/nonfinite")
        optimizer.step(); ledger.bump("optimizer_steps"); clock.sync()
        state["training_seconds"] += time.perf_counter() - t0
        state["steps"] += 1; state["case_updates"][key] += 1; state["case_order_prefix"].append(key)
        state["offset"] += 1
        if state["offset"] == 3:
            state["offset"] = 0; state["pass_index"] += 1
        write(journal_path, {"status": "running", **state, "last_case": key, "last_case_mean_encoded_MSE": loss_sum,
            "active": False, "observed_unix_seconds": time.time()})
        del gpu, target, context, latent, context_leaf, latent_leaf, prediction, loss
        if state["training_seconds"] - last_checkpoint_seconds >= 60:
            persist_checkpoint(); last_checkpoint_seconds = state["training_seconds"]
        clock.guard()
    persist_checkpoint()
    summary = {"status": "completed", "config": config, "arm": arm, "complete": True,
        "endpoint": "first completed balanced TRAIN3 pass at cumulative actual600s", **state,
        "training_target_seconds": 600, "training_overshoot_seconds": state["training_seconds"] - 600,
        "common_initialization": identity, "parameter_counts": counts, "capacity_matched": False,
        "model_setup_or_owned_restore_wall_seconds": setup_seconds,
        "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)}, "read_receipt": dict(ledger.data)}
    write(summary_file, summary); write(folder / "read_receipt.json", ledger.data)
    write(journal_path, {"status": "completed", **state, "active": False, "observed_unix_seconds": time.time()})
    del model, optimizer; torch.cuda.empty_cache(); clock.persist()
    return summary


def prediction_metadata(output, arm, key):
    folder = output / arm / "predictions" / key.replace("/", "_")
    return folder, folder / "prediction_metadata.json"


def validate_prediction_meta(meta, contract, arm, key):
    require(meta["config"] == {"contract": contract, "arm": arm, "case_key": key}
        and meta["status"] == "completed" and meta["full_original_node_count"] == contract["records"][key]["node_count"],
        "Fixed allN prediction identity differs")
    require(stat_match(meta["signedS"]) and stat_match(meta["physical_U"])
        and meta["physical_U"]["file_sha256"] == contract["records"][key]["physical_U_file_sha256"],
        "Persisted allN prediction stat/original U identity differs")


def predict_arm(np, torch, cls, arm, common, items, contract, output, ledger, clock, training):
    from .elastic_stress_readout import save_array
    all_meta = {}; missing = []
    for key in CASES:
        _, file = prediction_metadata(output, arm, key)
        if file.exists():
            meta = read(file); validate_prediction_meta(meta, contract, arm, key); all_meta[key] = meta
        else:
            missing.append(key)
    if not missing:
        return all_meta
    clock.guard(); setup_t0 = time.perf_counter()
    model, _ = create_model(np, torch, cls, arm, common)
    saved = owned_load(torch, output / arm / "last.pt", ledger)
    require(saved["config"] == {"contract": contract, "arm": arm} and saved["state"]["steps"] == training["steps"],
        "Fixed trained endpoint checkpoint differs")
    model.load_state_dict(saved["model"]); model.eval(); clock.sync()
    model_load_seconds = time.perf_counter() - setup_t0
    sscale = contract["parent_scalers"]["signed_stress"]
    for key in missing:
        clock.guard(); folder, file = prediction_metadata(output, arm, key); folder.mkdir(parents=True, exist_ok=True)
        grid_preexisting = items[key]["grid_cpu"] is not None
        t0 = time.perf_counter(); gpu, input_seconds = gpu_item(np, torch, items[key], arm == ARMS[1], clock)
        forward_t0 = time.perf_counter(); values = np.empty(len(items[key]["xyz"]), dtype=np.float64)
        with torch.no_grad():
            context, latent = model.fields(gpu)
            for start in range(0, len(values), 32768):
                end = min(start + 32768, len(values))
                encoded = model.points(gpu, start, end, context, latent).double().cpu().numpy()
                values[start:end] = np.sinh(encoded * sscale["std"] + sscale["mean"]) * sscale["scale_MPa"]
        require(np.isfinite(values).all(), "Fixed direct unbounded signedS endpoint nonfinite")
        clock.sync(); forward_seconds = time.perf_counter() - forward_t0; persist_t0 = time.perf_counter()
        stress = save_array(np, folder / "signedS.npy", values)
        record = contract["records"][key]; original = path(record["original_zero_physical_U"]["path"])
        ledger.bump("physical_U_copy_reads", key + ":U_copy:" + arm)
        temporary = folder / "physical_u_mm.npy.tmp"; shutil.copyfile(original, temporary)
        require(sha(temporary) == record["physical_U_file_sha256"], "Original U exact byte copy differs")
        os.replace(temporary, folder / "physical_u_mm.npy"); ledger.complete(key + ":U_copy:" + arm)
        stat = (folder / "physical_u_mm.npy").stat()
        u = {"path": str(folder / "physical_u_mm.npy"), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "shape": [record["node_count"], 3], "dtype": "float64", "file_sha256": record["physical_U_file_sha256"],
            "preserved_original_byte_identity": True, "new_U_GT_metric": None,
            "original_U_accuracy_provenance_only": record["original_fine_summary"]}
        persist_seconds = time.perf_counter() - persist_t0; actual_request = time.perf_counter() - t0
        grid_build = items[key].get("first_actual_targetfree_grid_build_seconds", 0.0) if arm == ARMS[1] else 0.0
        # If DEV's first raster build was inside this request, actual_request already includes it.
        grid_already_in_request = arm == ARMS[1] and not grid_preexisting
        attributed_grid = 0.0 if grid_already_in_request else grid_build
        neural_cost = items[key]["cached_input_wall_seconds"] + items[key]["query_normalization_wall_seconds"] + model_load_seconds + actual_request + attributed_grid
        meta = {"status": "completed", "config": {"contract": contract, "arm": arm, "case_key": key},
            "selected_step": training["steps"], "full_original_node_count": len(values), "signedS": stress, "physical_U": u,
            "prediction_persisted_unix_seconds": time.time(),
            "costs": {"cached_input_read_and_verify_seconds": items[key]["cached_input_wall_seconds"],
                "physical_query_normalization_seconds": items[key]["query_normalization_wall_seconds"],
                "targetfree_grid_build_attributable_seconds": grid_build,
                "targetfree_grid_build_already_in_actual_request": grid_already_in_request,
                "shared_arm_model_create_load_seconds_charged_each_request": model_load_seconds,
                "request_input_H2D_and_targetfree_grid_seconds": input_seconds,
                "context_global_query_fullN_decode_seconds": forward_seconds, "prediction_and_exactU_persist_seconds": persist_seconds,
                "actual_cached_prediction_request_wall_seconds": actual_request,
                "composed_cached_neural_increment_seconds": neural_cost,
                "original_zero_GT_free_physical_seconds": record["accepted_physical_GT_free_cold_seconds"],
                "original_accepted_recovery_and_input_seconds": record["accepted_feature_preparation_costs"]["elapsed_seconds"],
                "composed_accounted_fullcold_seconds": neural_cost + record["accepted_physical_GT_free_cold_seconds"] + record["accepted_feature_preparation_costs"]["elapsed_seconds"],
                "fresh_combined_cold_measurement": False, "startpoint": "accepted input/physics cache with inherited original cost",
                "training_and_new_scaler_fit_excluded_from_deployment": True}, "read_receipt": dict(ledger.data)}
        write(file, meta); write(folder / "read_receipt.json", ledger.data)
        all_meta[key] = meta
        del gpu, context, latent, values, encoded; clock.persist(); clock.guard()
    del model, saved; torch.cuda.empty_cache(); return all_meta


def evaluate(np, predictions, contract, output, ledger, clock):
    from .elastic_stress_readout import signed_metrics, aggregate
    for arm in ARMS:
        for key in DEV:
            validate_prediction_meta(predictions[arm][key], contract, arm, key)
    barrier = {"both_fixed_arms_allDEV_predictions_persisted_before_labels": True,
        "arms": list(ARMS), "validation_keys": list(DEV), "verified_unix_seconds": time.time(),
        "prediction_manifest_sha256": {arm: {key: sha(prediction_metadata(output, arm, key)[1]) for key in DEV} for arm in ARMS}}
    barrier_file = output / "prediction_barrier.json"
    if barrier_file.exists():
        require(read(barrier_file)["prediction_manifest_sha256"] == barrier["prediction_manifest_sha256"], "DEV prediction barrier differs")
    else:
        require(ledger.data["DEV_stress_reads"] == 0, "DEV label read before committed prediction barrier")
        write(barrier_file, barrier)
    truth = {}
    for key in DEV:
        clock.guard(); truth[key] = load_stress(np, contract, key, ledger, output, dev=True)
    result = {}
    for arm in ARMS:
        per_case = {}
        for key in DEV:
            folder, _ = prediction_metadata(output, arm, key); summary_file = folder / "summary.json"
            if summary_file.exists():
                saved = read(summary_file)
                require(saved["config"] == {"contract": contract, "arm": arm, "case_key": key}, "Owned metric identity differs")
                per_case[key] = saved["metrics"]; continue
            clock.guard(); t0 = time.perf_counter()
            predicted = array_load(np, predictions[arm][key]["signedS"], ledger, "own_prediction_array_reads", arm + ":" + key + ":S_pred")
            metrics = signed_metrics(np, truth[key], predicted); per_case[key] = metrics
            write(summary_file, {"status": "completed", "config": {"contract": contract, "arm": arm, "case_key": key},
                "metrics": metrics, "original_U_accuracy_provenance_only": contract["records"][key]["original_fine_summary"],
                "new_U_GT_metrics": None, "costs": predictions[arm][key]["costs"],
                "shared_DEV_metric_seconds": time.perf_counter() - t0, "read_receipt": dict(ledger.data)})
        result[arm] = aggregate(per_case, DEV)
    write(output / "evaluation_receipt.json", {"barrier": read(barrier_file), "DEV_source_label_reads_once_per_case": 2,
        "read_receipt": ledger.data}); return result


def gates(arms, metrics, negative=None):
    passed = {}
    for arm in ARMS:
        aggregate = metrics.get(arm, {}); percase = aggregate.get("per_case", {})
        passed[arm] = negative is None and arms.get(arm, {}).get("complete") is True and all(
            finite((percase.get(key) or {}).get("stress_relative_l2")) and percase[key]["stress_relative_l2"] < .25
            and finite(percase[key].get("hotspot_relative_l2")) and percase[key]["hotspot_relative_l2"] < .25 for key in DEV)
    local = metrics.get(ARMS[0], {}).get("stress_relative_l2_mean")
    global_value = metrics.get(ARMS[1], {}).get("stress_relative_l2_mean")
    gain = (local - global_value) / local if finite(local) and local > 0 and finite(global_value) else None
    justified = passed[ARMS[1]] and finite(gain) and gain >= .20
    choice = ARMS[1] if justified else ARMS[0] if passed[ARMS[0]] else None
    return {"passed": choice is not None, "arm_complete_service_gate": passed,
        "global_component_justified": justified, "global_DEV_meanS_relative_improvement": gain,
        "within_fullFE_pair_choice": choice, "not_final_solver_selection": True,
        "capacity_matched": False, "terminal_negative_reason": negative,
        "bothDEVpredictions_persisted_before_labels": bool(metrics)}


def brief(output, summary):
    lines = ["# Physical geometry / global latent pair", "", "Original U is copied byte-identically; its old accuracy is provenance only.",
        "Full-FE inputs differ from the point-cloud benchmark. Global extra parameters are explicit.", "",
        "| Arm | DEV mean S L2 | Worst hot L2 | Composed cold range (s) |", "|---|---:|---:|---:|"]
    for arm in ARMS:
        metrics = summary.get("metrics", {}).get(arm, {}); predictions = summary.get("predictions", {}).get(arm, {})
        cold = [predictions[key]["costs"]["composed_accounted_fullcold_seconds"] for key in DEV if key in predictions]
        value = lambda v: "missing" if not finite(v) else f"{100*v:.2f}%"
        cold_text = "missing" if len(cold) != 2 else f"{min(cold):.2f}–{max(cold):.2f}"
        lines.append(f"| {arm} | {value(metrics.get('stress_relative_l2_mean'))} | {value(metrics.get('hotspot_relative_l2_worst'))} | {cold_text} |")
    gate = summary["completion_gate"]
    lines += ["", f"Within-pair candidate: {gate['within_fullFE_pair_choice']}; global justified: {gate['global_component_justified']}.",
        f"Accounted campaign: {summary['costs']['campaign_accounted_wall_seconds']:.3f}s; inherited original physics/features once: 457.480s.",
        "Cold values compose accepted original physics/recovery and measured neural costs; they are not fresh combined requests.",
        "Training/scaler fitting is separate. No physical/recovery replay or new U GT metric."]
    if gate["terminal_negative_reason"]:
        lines += ["Fixed candidate stopped: " + gate["terminal_negative_reason"]]
    require(len(lines) <= 25, "Brief line limit exceeded")
    (output / "BRIEF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "metrics.md").write_text("\n".join(lines[5:9]) + "\n", encoding="utf-8")


def run(plan_path=PLAN, output=None):
    invocation_start = time.perf_counter(); contract, plan = request_contract(plan_path)
    output = path(output or plan["output"]); require(output == path(plan["output"]), "Frozen output differs")
    marker = read(plan["root_review_marker"])
    require(marker.get("passed") is True and marker.get("source_sha256") == contract["source_sha256"][SOURCE]
        and marker.get("scientific_digest") == contract["scientific_digest"]
        and marker.get("decision") == "proceed_frozen_geometry_BC_physical_spatial_pair", "Actual root release missing")
    output.mkdir(parents=True, exist_ok=True); config = {"contract": contract}
    if (output / "config.json").exists():
        require(read(output / "config.json") == config, "Persistent pair config differs")
    write(output / "config.json", config)
    if (output / "summary.json").exists():
        old = read(output / "summary.json")
        if old.get("status") == "completed":
            require(old["config"] == config, "Completed pair source identity differs"); return old
    clock = Clock(output, contract["inherited_seconds"], invocation_start); ledger = Ledger(output)
    arms = {}; predictions = {}; metrics = {}; negative = None; technical = None
    try:
        clock.guard()
        import numpy as np
        import torch
        require(torch.cuda.is_available(), "Frozen GPU backend unavailable")
        clock.torch = torch; torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        torch.set_num_threads(8)
        cls = model_class(torch)
        initial_file = output / "initial_common.pt"
        if initial_file.exists():
            saved = owned_load(torch, initial_file, ledger); common = saved["common"]
            require(saved["config"] == config and common_identity(torch, common) == read(output / "initial_common.json")["identity"],
                "Owned common initialization differs")
        else:
            random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
            fresh = cls(False); common = {name: value.detach().clone().cpu() for name, value in fresh.state_dict().items()}
            identity = common_identity(torch, common)
            own_checkpoint(torch, initial_file, {"config": config, "common": common}, ledger)
            write(output / "initial_common.json", {"identity": identity, "seed": 2026, "before_any_FNO_initialization": True})
            del fresh
        items, scalers, prepare_seconds = prepare_inputs(np, contract, ledger, output, clock)
        targets = {}; s = contract["parent_scalers"]["signed_stress"]
        for key in TRAIN:
            clock.guard(); raw = load_stress(np, contract, key, ledger, output)
            targets[key] = ((np.arcsinh(raw / s["scale_MPa"]) - s["mean"]) / s["std"]).astype(np.float32)
        for arm in ARMS:
            arms[arm] = train_arm(np, torch, cls, arm, common, items, targets, contract, output, ledger, clock)
        common_prefix = min(arms[arm]["steps"] for arm in ARMS)
        require(arms[ARMS[0]]["case_order_prefix"][:common_prefix] == arms[ARMS[1]]["case_order_prefix"][:common_prefix],
            "Paired balanced shuffled case-order prefix differs")
        for arm in ARMS:
            predictions[arm] = predict_arm(np, torch, cls, arm, common, items, contract, output, ledger, clock, arms[arm])
        metrics = evaluate(np, predictions, contract, output, ledger, clock); clock.guard()
    except ScientificNegative as exc:
        negative = str(exc)
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            negative = "Frozen GPU backend memory negative: " + str(exc)
        else:
            technical = type(exc).__name__ + ": " + str(exc)
    finally:
        # Every exit keeps actual wall/read ledgers and committed checkpoints.
        for arm in ARMS:
            journal_file = output / arm / "status.json"
            if journal_file.exists():
                journal = read(journal_file)
                if journal.get("active"):
                    # A caught failure has an observed exit: exclude subsequent idle time.
                    elapsed = max(0.0, time.time() - journal["observed_unix_seconds"])
                    journal["training_seconds"] += elapsed
                    journal["failed_actual_training_seconds"] = journal.get("failed_actual_training_seconds", 0) + elapsed
                    journal["active"] = False; journal["observed_unix_seconds"] = time.time()
                    write(journal_file, journal)
        clock.persist(False); ledger.persist()
    costs = read(clock.file)
    summary = {"status": "failed" if technical else "completed", "kind": KIND, "config": config,
        "train_keys": list(TRAIN), "validation_keys": list(DEV), "case_keys": list(CASES),
        "cases": {key: {"node_count": contract["records"][key]["node_count"],
            "original_U_file_sha256": contract["records"][key]["physical_U_file_sha256"],
            "original_U_accuracy_provenance_only": contract["records"][key]["original_fine_summary"],
            "new_U_GT_metrics": None} for key in CASES},
        "arms": {arm: arms.get(arm, {"complete": False}) for arm in ARMS}, "predictions": predictions,
        "metrics": metrics, "completion_gate": gates(arms, metrics, negative or technical),
        "costs": costs, "read_receipt": ledger.data, "technical_error": technical,
        "full_FE_vs_pointcloud_information_difference": True, "new_GT_U_accuracy_claim": False}
    write(output / "summary.json", summary); write(output / "status.json", {"status": summary["status"],
        "completion_gate": summary["completion_gate"], "costs": costs, "technical_error": technical})
    brief(output, summary)
    # Include the first full report/receipt persistence in the final stage ledger.
    clock.persist(False); summary["costs"] = {**costs, **read(clock.file)}
    if summary["costs"]["campaign_accounted_wall_seconds"] >= 1800 and technical is None:
        summary["completion_gate"] = gates(arms, metrics, "Fixed cumulative campaign wall budget exhausted including report persistence")
    write(output / "summary.json", summary)
    write(output / "status.json", {"status": summary["status"], "completion_gate": summary["completion_gate"],
        "costs": summary["costs"], "technical_error": technical})
    if technical:
        raise RuntimeError(technical)
    return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--plan", default=PLAN); parser.add_argument("--output")
    args = parser.parse_args(); result = run(args.plan, args.output)
    print(json.dumps({"status": result["status"], "completion_gate": result["completion_gate"], "costs": result["costs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
