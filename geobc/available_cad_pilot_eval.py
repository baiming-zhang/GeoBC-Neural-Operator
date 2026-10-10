"""Independent available-CAD20 readout; default is immediate, zero-I/O source mode.

Actual execution requires the separately frozen root release, authentic closed
predecessors, completed target-free inputs, and the one-shot external wall guard.
No training wrapper or original sealed-DEV reader is called by this adapter.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
import time

PLAN = "notes/geobc_available_cad_pilot_plan.json"
OWN = "geobc/available_cad_pilot_eval.py"
SCI = "c7c994a44d462b94426ecfa034f2d535bbf7751331c1e16ab8925485e5a67770"
KIND = "GeoBC_available_CAD20_candidate9_independent_readout"
CHUNK = 32768
LOADS = ("ver", "hor", "tor")
ENDPOINTS = (
    ("transolver_cad40_fresh4800", "transolver", 4800),
    ("transolver_pp_cad40_fresh4800", "transolver_pp", 4800),
    ("ts3_cad40_fresh4800", "ts3", 4800),
    ("fno_cad40_fresh4800", "fno", 4800),
    ("mlp_cad40_fresh4800", "mlp", 4800),
    ("PP_regime_step4800", "transolver_pp", 4800),
    ("PP_regime_step24000", "transolver_pp", 24000),
    ("FNO_latent_query_grid25_step4800", "FNO_latent_query_grid25", 4800),
    ("FNO_latent_query_native25_step4800", "FNO_latent_query_native25", 4800),
)


class Negative(RuntimeError):
    """A real terminal budget, OOM, nonfinite or interrupted-read outcome."""


def utc():
    return datetime.now(timezone.utc).isoformat()


def case_seed(key):
    return int.from_bytes(hashlib.blake2b(
        f"2026:GeoBC_AVAILABLE20:{key}".encode(), digest_size=8).digest(), "little") % 2**31


def _science(d, contract, plan):
    d.require(plan.get("scientific_rules_frozen") is True
              and d.digest(plan["scientific"]) == plan["scientific_digest"] == SCI
              and contract["scientific_digest"] == SCI, "Independent frozen science differs")
    science = plan["scientific"]; selection = science["CAD_selection"]
    cads = selection["selected_CADs"]; keys = selection["case_keys"]
    d.require(len(cads) == len(set(cads)) == 20 and cads == sorted(selection["eligible_CADs"])[:20]
              and keys == [cad + "/" + load for cad in cads for load in LOADS]
              and len(keys) == 60 and not (set(cads) & set(selection["excluded_old_TRAIN40"]))
              and not (set(cads) & set(selection["excluded_original_DEV20"])), "Fixed CAD20/load3 scope differs")
    candidates = science["candidate_inventory"]["candidates"]
    d.require([(c["candidate_id"], c["method"], c["step"]) for c in candidates] == list(ENDPOINTS)
              and all(c["TRAIN_CADs"] == 40 and c["optimizer_restore"] is False for c in candidates),
              "Fixed nine endpoints or old40 qualification differs")
    d.require(contract["case_keys"] == keys and contract["selected_CADs"] == cads
              and contract["scaler"] == science["input_policy"]["scaler"]
              and contract["groups"]["train_new160"] == [], "Input identity/scaler/target-free branch differs")
    d.require(OWN in contract["source_sha256"] and d.sha(OWN) == contract["source_sha256"][OWN],
              "Evaluator source identity differs")
    return candidates, cads, keys


class Clock:
    """One attempt only; outer guard remains authoritative after forced exit."""
    def __init__(self, d, output, guard, run_process):
        self.d = d; self.file = output / "clock.json"; self.guard = guard
        self.last = float(guard["entry_monotonic"])
        self.value = {"active": True, "closed": False, "attempts": 1, "run_process": run_process,
            "entry_epoch": guard["entry_epoch"], "entry_monotonic": self.last,
            "cumulative_wall_seconds": 0., "phase": "metadata_and_release", "phase_seconds": {},
            "peak_allocated_bytes": 0, "peak_reserved_bytes": 0,
            "external_guard": guard, "automatic_replay": False, "outer_guard_is_authoritative": True}
        self.tick()

    def tick(self):
        now = time.monotonic(); elapsed = max(0., now - self.last); self.last = now
        self.value["cumulative_wall_seconds"] = max(0., now - float(self.guard["entry_monotonic"]))
        name = self.value["phase"]; phases = self.value["phase_seconds"]
        phases[name] = phases.get(name, 0.) + elapsed
        self.value["last_epoch"] = time.time(); self.d.write(self.file, self.value)

    def phase(self, name):
        self.tick(); self.value["phase"] = name

    def check(self, torch=None):
        if (time.time() >= min(self.guard["deadline_epoch"], self.guard["campaign_deadline_epoch"])
                or time.monotonic() >= min(self.guard["deadline_monotonic"], self.guard["campaign_deadline_monotonic"])):
            raise Negative("Actual all-inclusive stage/campaign wall deadline reached; no rescue or replay")
        if torch is not None and torch.cuda.is_initialized():
            allocated = int(torch.cuda.max_memory_allocated()); reserved = int(torch.cuda.max_memory_reserved())
            self.value["peak_allocated_bytes"] = max(self.value["peak_allocated_bytes"], allocated)
            self.value["peak_reserved_bytes"] = max(self.value["peak_reserved_bytes"], reserved)
            if max(allocated, reserved) > 24 * 1024**3:
                raise Negative("Actual allocated/reserved CUDA peak exceeds 24 GiB")

    def close(self, reason):
        self.tick(); self.value.update(active=False, closed=True, phase="terminal",
            closed_at_utc=utc(), terminal_reason=reason); self.d.write(self.file, self.value)


class Ledger:
    def __init__(self, d, output):
        self.d = d; self.file = output / "read_receipt.json"
        self.value = {name: 0 for name in (
            "input_array_read_attempts", "input_array_reads", "checkpoint_read_attempts", "checkpoint_reads",
            "checkpoint_byte_hash_reads", "model_initializations", "prediction_attempts", "finite_allN_predictions",
            "prediction_cache_writes", "prediction_validation_array_reads", "prediction_metric_array_reads",
            "permutation_array_reads", "source_GT_hash_byte_reads", "source_GT_parse_attempts", "source_GT_parses",
            "U_target_case_reads", "S_target_case_reads", "shared_GT_array_writes", "shared_GT_array_reads",
            "optimizer_restores", "training_updates", "scaler_fits", "old_DEV_reads", "old_final_reads",
            "original_new_DEV20_reads", "dia_reads", "strain_reads", "H5_nodal_reads")}
        self.value.update(source_GT_parses_before_barrier=0, GT_source_attempts={},
                          all_inputs_target_free=True, actual_read_receipt_exact=True)
        self.persist()

    def persist(self):
        self.d.write(self.file, self.value)

    def bump(self, name, count=1):
        self.value[name] += count; self.persist()


def _identity(d, filename):
    filename = d.path(filename); stat = filename.stat()
    return {"path": str(filename), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": d.sha(filename)}


def _check_file(d, identity, owned=None, hashed=True):
    filename = d.path(identity["path"])
    if owned is not None:
        filename.relative_to(d.path(owned))
    stat = filename.stat()
    d.require(stat.st_size == identity["bytes"] and stat.st_mtime_ns == identity["mtime_ns"],
              "Bound file stat changed: " + str(filename))
    if hashed:
        d.require(isinstance(identity.get("sha256"), str) and d.sha(filename) == identity["sha256"],
                  "Bound file byte identity changed: " + str(filename))
    return filename


def _save_array(d, np, filename, value):
    filename = d.path(filename); filename.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive final creation makes both predictions and GT owned immutable outputs.
    with filename.open("xb") as handle:
        np.save(handle, value, allow_pickle=False); handle.flush(); os.fsync(handle.fileno())
    return {**_identity(d, filename), "shape": list(value.shape), "dtype": str(value.dtype), "exclusive_write": True}


def _input_member(d, np, manifest, record, name, ledger, clock):
    clock.check(); ledger.bump("input_array_read_attempts")
    item = record["file_identities"][name + "_path"]
    filename = _check_file(d, item, manifest["output"], hashed=True)
    value = np.load(filename, mmap_mode="r", allow_pickle=False); ledger.bump("input_array_reads")
    d.require(list(value.shape) == item["shape"] and str(value.dtype) == item["dtype"], "Input array schema differs")
    d.require(np.isfinite(value).all(), "Nonfinite target-free input")
    return value


def _load_input(d, np, torch, manifest, key, for_fno, ledger, clock):
    record = manifest["records"][key]; n = record["num_nodes"]
    d.require(record["feasibility_pass"] is True and record["status"] == "completed"
              and not ({"target_path", "physical_path"} & set(record)), "Input incomplete or labels present")
    def tensor(name):
        return torch.as_tensor(np.array(_input_member(d, np, manifest, record, name, ledger, clock), copy=True), device="cuda")
    case = {"key": key, "xyz": tensor("xyz"), "features": tensor("features"),
            "cloud": tensor("cloud").reshape(1, 1024, 3), "mask": tensor("mask").reshape(1, 1024).bool()}
    d.require(tuple(case["xyz"].shape) == (n, 3) and tuple(case["features"].shape) == (n, 25), "AllN input shapes differ")
    if for_fno:
        case.update(grid=tensor("grid").reshape(1, 29, 64, 64, 64),
                    query_indices=tensor("query_indices"), query_weights=tensor("query_weights"))
        d.require(tuple(case["query_indices"].shape) == tuple(case["query_weights"].shape) == (n, 8), "Geometry query schema differs")
    clock.check(torch); return case


def _training_metadata(d, candidate, plan):
    """Called only after the authentic PP/FNO terminal/exact-process gate."""
    folder = d.path(candidate["training_output"]); summary = d.read(folder / "summary.json")
    config = d.read(folder / "config.json")
    d.require(summary.get("status") in ("completed", "failed") and summary.get("config") == config,
              "Training output has no genuine terminal configuration")
    contract = config["contract"]
    d.require(contract["scientific_digest"] == candidate["training_scientific_digest"], "Training science differs")
    d.require(config["scaler"] == plan["scientific"]["input_policy"]["scaler"]
              and contract["scaler"] == config["scaler"], "Frozen old40 scaler differs")
    old_keys = [cad + "/" + load for cad in plan["scientific"]["CAD_selection"]["excluded_old_TRAIN40"] for load in LOADS]
    actual_keys = contract.get("TRAIN_keys", contract.get("input_contract", {}).get("train_keys", {}).get("40"))
    d.require(actual_keys is not None and len(actual_keys) == len(set(actual_keys)) == 120
              and set(actual_keys) == set(old_keys), "Endpoint was not trained exclusively on frozen old40")
    if candidate["candidate_id"] in [row[0] for row in ENDPOINTS[:5]]:
        d.require(contract["arm"]["id"] == candidate["candidate_id"]
                  and contract["arm"]["method"] == candidate["method"] and contract["arm"]["train_CADs"] == 40,
                  "Old endpoint family/scale differs")
    elif candidate["method"].startswith("FNO_latent_query_"):
        d.require(config["arm"] == candidate["method"], "FNO pair arm identity differs")
    training_plan = d.read(candidate["training_plan"])
    d.require(d.digest(training_plan["scientific"]) == training_plan["scientific_digest"]
              == candidate["training_scientific_digest"], "Inherited frozen training plan differs")
    pins = contract["source_sha256"]
    # PP recovery accepts explicitly reviewed prior driver versions; never guess one.
    accepted = {}
    if candidate["candidate_id"].startswith("PP_regime_"):
        marker = d.read(training_plan["root_review"])
        accepted["geobc/transolver_pp_regime_confirmation.py"] = marker.get("accepted_prior_driver_sha256", [])
    for name, expected in pins.items():
        d.require(d.sha(name) == expected or expected in accepted.get(name, []), "Inherited training source identity differs: " + name)
    return {"config": config, "config_sha256": d.digest(config), "training_summary": _identity(d, folder / "summary.json"),
        "inherited_costs": summary.get("costs"), "inherited_training_seconds":
            summary.get("costs", {}).get("training_seconds", summary.get("costs", {}).get("train_seconds")),
        "parameter_counts": config.get("parameter_counts", summary.get("parameter_counts")),
        "training_qualification": summary.get("completion_gate", {"completed4800": summary.get("completed4800"),
            "completed24000": summary.get("completed24000"), "stop_reason": summary.get("stop_reason")}),
        "final_training_exposures": summary.get("case_exposure_counts"), "fixed_training_keys": actual_keys}


def _bind_endpoint(d, np, torch, model, candidate, metadata, ledger, clock, binding):
    checkpoint = d.path(candidate["checkpoint_path"]); clock.check(torch)
    if not checkpoint.is_file():
        raise FileNotFoundError("Genuinely absent fixed endpoint after upstream exit: " + str(checkpoint))
    ledger.bump("checkpoint_byte_hash_reads"); identity = _identity(d, checkpoint)
    binding["checkpoint"] = identity
    endpoint_json = d.read(candidate["endpoint_metadata_path"])
    binding["endpoint_metadata"] = _identity(d, candidate["endpoint_metadata_path"])
    binding["producer_metadata_step"] = endpoint_json.get("step")
    if candidate["candidate_id"].startswith("PP_regime_"):
        d.require(endpoint_json.get("step") == candidate["step"]
                  and isinstance(endpoint_json.get("checkpoint"), dict), "PP fixed endpoint metadata differs")
        stored = endpoint_json["checkpoint"]
        d.require(d.path(stored["path"]) == checkpoint and all(stored.get(k) == identity[k]
            for k in ("bytes", "mtime_ns", "sha256")), "PP producer checkpoint byte binding differs")
        d.require(endpoint_json["config_contract_sha256"] == d.digest(metadata["config"]["contract"]),
                  "PP metadata contract binding differs")
    elif candidate["method"].startswith("FNO_latent_query_"):
        # This producer JSON describes last.pt, not the immutable step4800.pt.
        d.require(d.path(endpoint_json["path"]) == d.path(candidate["training_output"]) / "last.pt"
                  and endpoint_json["config_sha256"] == metadata["config_sha256"], "FNO producer metadata config differs")
        binding["producer_metadata_describes_last_pt"] = True
    else:
        d.require(endpoint_json.get("step") == candidate["step"]
                  and d.path(endpoint_json["checkpoint"]) == checkpoint
                  and d.digest(endpoint_json["config"]) == metadata["config_sha256"], "Old fixed endpoint metadata differs")
    ledger.bump("checkpoint_read_attempts")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False); ledger.bump("checkpoint_reads")
    binding.update(payload_step=saved.get("step"), payload_config_sha256=d.digest(saved.get("config")),
        training_source_sha256=saved.get("config", {}).get("contract", {}).get("source_sha256"))
    d.require(saved["step"] == candidate["step"] and d.digest(saved["config"]) == metadata["config_sha256"],
              "Actual PT fixed step/config identity differs")
    exposure = saved["case_exposure_counts"]; passes = candidate["complete_TRAIN120_passes"]
    d.require(exposure == {key: passes for key in metadata["fixed_training_keys"]}, "Fixed endpoint full old40 exposure differs")
    model.load_state_dict(saved["model"], strict=True); model.eval()
    _check_file(d, identity, hashed=False)
    binding.update({"status": "valid", "candidate": candidate, "checkpoint": identity,
        "payload_step": saved["step"], "payload_config_sha256": d.digest(saved["config"]),
        "training_source_sha256": saved["config"]["contract"]["source_sha256"],
        "case_exposure_counts": exposure, "optimizer_restored": False, "model_eval": True,
        "metadata": metadata})
    del saved; clock.check(torch); return binding


@contextmanager
def _rng(np, torch, key):
    py_state = random.getstate(); np_state = np.random.get_state(); seed = case_seed(key)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            yield seed
    finally:
        random.setstate(py_state); np.random.set_state(np_state)


def _predict(np, torch, model, candidate, case, predict, query_grid, clock):
    with _rng(np, torch, case["key"]), torch.no_grad():
        if candidate["method"].startswith("FNO_latent_query_"):
            context, latent = model.encode_case(case)
            encoded = np.empty((len(case["xyz"]), 4), dtype=np.float32)
            for start in range(0, len(encoded), CHUNK):
                clock.check(torch); stop = min(len(encoded), start + CHUNK)
                part = model.query(case, start, stop, context, latent)
                if not bool(torch.isfinite(part).all()):
                    raise FloatingPointError("Nonfinite FNO fullN query")
                encoded[start:stop] = part.float().cpu().numpy()
        else:
            encoded = predict(torch, model, candidate["method"], case, query_grid)
    torch.cuda.synchronize(); clock.check(torch); return encoded


def _record_path(output, candidate_id, key):
    cad, load = key.split("/")
    return output / "predictions" / candidate_id / cad / load / "manifest.json"


def _terminal_record(d, output, candidate, key, contract, input_identity, endpoint, reason=None, failure_scope=None):
    return {"kind": KIND, "status": "completed", "terminal": True, "candidate_id": candidate["candidate_id"],
        "key": key, "fixed_step": candidate["step"], "contract_sha256": d.digest(contract),
        "input_identity": input_identity, "endpoint_identity_sha256": d.digest(endpoint),
        "prediction_complete": False, "finite_allN": False, "failure":
            {"scope": failure_scope, "reason": reason, "real_terminal_outcome": True} if reason else None,
        "eval_rng_seed": case_seed(key), "units": ["mm", "mm", "mm", "MPa"], "signed_stress_unchanged": True}


def _commit_record(d, output, record, inventory):
    filename = _record_path(output, record["candidate_id"], record["key"])
    d.require(not filename.exists(), "Terminal entry already exists; automatic replay is forbidden")
    d.write(filename, record); token = record["candidate_id"] + ":" + record["key"]
    inventory[token] = _identity(d, filename)


def commit_barrier(d, np, output, contract, plan, manifest, endpoints, inventory, ledger, clock, campaign_negative_reason=None):
    """Revalidate exact540 manifests and real owned arrays; no passed-marker shortcut."""
    candidates, cads, keys = _science(d, contract, plan)
    expected = {c["candidate_id"] + ":" + key for c in candidates for key in keys}
    d.require(set(inventory) == expected and len(inventory) == 540 and set(endpoints) == {c["candidate_id"] for c in candidates},
              "Fixed nine-by-sixty inventory differs")
    d.require(ledger.value["source_GT_parse_attempts"] == ledger.value["source_GT_parses"] == 0,
              "Numeric source GT was attempted before barrier")
    finite = 0; failures = 0
    for candidate in candidates:
        endpoint = endpoints[candidate["candidate_id"]]
        d.require(endpoint["candidate"] == candidate, "Fixed candidate endpoint binding differs")
        if endpoint["status"] == "valid":
            d.require(d.path(endpoint["checkpoint"]["path"]) == d.path(candidate["checkpoint_path"])
                and endpoint["payload_step"] == candidate["step"] and endpoint["optimizer_restored"] is False
                and endpoint["model_eval"] is True
                and endpoint["payload_config_sha256"] == endpoint["metadata"]["config_sha256"]
                == d.digest(endpoint["metadata"]["config"])
                and endpoint["training_source_sha256"] == endpoint["metadata"]["config"]["contract"]["source_sha256"]
                and endpoint["case_exposure_counts"] == {key: candidate["complete_TRAIN120_passes"]
                    for key in endpoint["metadata"]["fixed_training_keys"]}, "Actual endpoint identity is incomplete")
            _check_file(d, endpoint["checkpoint"], hashed=False)
            _check_file(d, endpoint["metadata"]["training_summary"])
        else:
            d.require(endpoint["status"] in ("invalid", "not_loaded") and bool(endpoint["real_terminal_reason"]),
                      "Endpoint is neither a real valid binding nor a true terminal failure")
        for key in keys:
            clock.check(); identity = inventory[candidate["candidate_id"] + ":" + key]
            expected_path = _record_path(output, candidate["candidate_id"], key)
            d.require(d.path(identity["path"]) == expected_path, "Inventory ownership/path differs")
            record = d.read(_check_file(d, identity, output))
            d.require(record["terminal"] is True and record["status"] == "completed"
                and record["candidate_id"] == candidate["candidate_id"] and record["key"] == key
                and record["fixed_step"] == candidate["step"] and record["contract_sha256"] == d.digest(contract)
                and record["endpoint_identity_sha256"] == d.digest(endpoint)
                and record["input_identity"] == manifest["records"].get(key)
                and record["eval_rng_seed"] == case_seed(key), "Terminal prediction identities differ")
            if record["prediction_complete"]:
                n = manifest["records"][key]["num_nodes"]
                d.require(endpoint["status"] == "valid" and record["failure"] is None
                    and record["finite_allN"] is True and record["all_original_nodes"] is True
                    and record["node_count"] == n, "Successful prediction endpoint/shape flags differ")
                for name in ("encoded", "physical"):
                    item = record[name]; ledger.bump("prediction_validation_array_reads")
                    d.require(d.path(item["path"]) == expected_path.parent / (name + "4.npy"), "Owned prediction array path differs")
                    array = np.load(_check_file(d, item, expected_path.parent), mmap_mode="r", allow_pickle=False)
                    d.require(item["exclusive_write"] is True and item["shape"] == [n, 4]
                        and list(array.shape) == [n, 4] and str(array.dtype) == item["dtype"]
                        and np.isfinite(array).all(), "Owned immutable finite allN array differs")
                    del array
                finite += 1
            else:
                failure = record["failure"]
                d.require(record["finite_allN"] is False and failure["real_terminal_outcome"] is True
                    and failure["scope"] in ("endpoint", "input", "campaign") and bool(failure["reason"])
                    and "encoded" not in record and "physical" not in record, "True failure inventory differs")
                failures += 1
    complete_inputs = manifest.get("feasibility_pass") is True and set(manifest["records"]) == set(keys) and all(
        manifest["records"][key].get("feasibility_pass") is True for key in keys)
    proof = {"kind": KIND, "scientific_digest": SCI, "source_sha256": contract["source_sha256"],
        "contract_sha256": d.digest(contract), "endpoints_sha256": d.digest(endpoints), "entries": inventory,
        "required_entries": 540, "terminal_entries": finite + failures, "finite_predictions": finite,
        "real_failures": failures, "exact_cartesian_product_verified": True, "owned_allN_arrays_verified": True,
        "all_fixed20_inputs_complete": complete_inputs, "source_GT_parse_attempts_before_barrier": 0,
        "GT_eligible": complete_inputs and finite > 0 and campaign_negative_reason is None,
        "campaign_negative_reason": campaign_negative_reason, "no_further_prediction_or_endpoint_changes": True}
    d.write(output / "prediction_barrier.json", proof); return proof


def _shared_gt(d, np, output, contract, manifest, proof, ledger, clock):
    d.require(proof["GT_eligible"] is True and proof["required_entries"] == proof["terminal_entries"] == 540,
              "Shared numeric GT is not eligible")
    from .coverage_comparison_data import _targets
    folder = output / "shared_GT"; folder.mkdir(parents=True, exist_ok=False)
    cache = {"kind": KIND, "status": "running", "contract_sha256": d.digest(contract),
             "barrier_sha256": d.digest(proof), "records": {}, "source_reads": {}}
    d.write(folder / "manifest.json", cache)
    for cad in contract["selected_CADs"]:
        clock.check(); clock.phase("shared_GT_source_identity_and_permutation")
        records = [manifest["records"][cad + "/" + load] for load in LOADS]; first = records[0]
        source = contract["raw_sources"][cad]["Field"]
        d.require(all(r["deferred_source"] == source and r["permutation"] == first["permutation"]
                      and r["num_nodes"] == first["num_nodes"] for r in records), "Shared perCAD label alignment differs")
        permutation_identity = first["permutation"]
        permutation_path = _check_file(d, permutation_identity, manifest["output"])
        permutation = np.load(permutation_path, allow_pickle=False); ledger.bump("permutation_array_reads")
        n = first["num_nodes"]
        d.require(permutation.shape == (n,) and permutation.dtype.kind in "iu"
                  and np.array_equal(np.sort(permutation), np.arange(n)), "H5-to-CSV permutation is not a bijection")
        columns = manifest["GT_header_metadata"][cad]["columns"]
        names = [load + suffix for load in LOADS for suffix in ("_x_disp(mm)", "_y_disp(mm)", "_z_disp(mm)", "_stress(MPa)")]
        d.require(all(columns.count(name) == 1 for name in names), "Exactly12 original UXYZ/S column names required")
        filename = _check_file(d, source, hashed=False)
        d.require(cad not in ledger.value["GT_source_attempts"], "Uncommitted source read cannot be reparsed")
        ledger.value["GT_source_attempts"][cad] = {"status": "attempted", "source": source,
            "started_at_utc": utc(), "permutation": permutation_identity}
        ledger.bump("source_GT_parse_attempts")
        def loaded():
            ledger.bump("source_GT_parses"); ledger.bump("U_target_case_reads", 3); ledger.bump("S_target_case_reads", 3)
        try:
            ledger.bump("source_GT_hash_byte_reads"); d.require(d.sha(filename) == source["sha256"], "Original GT source byte identity differs")
            clock.check(); clock.phase("shared_GT_12_column_numeric_parse")
            values = _targets(np, filename, columns, permutation, n, on_loaded=loaded)
            _check_file(d, source, hashed=False); clock.check()
            for index, load in enumerate(LOADS):
                key = cad + "/" + load; physical = np.ascontiguousarray(values[:, index*4:index*4+4])
                item = _save_array(d, np, folder / cad / (load + "_physical4.npy"), physical)
                ledger.bump("shared_GT_array_writes")
                cache["records"][key] = {"physical": item, "source": source, "permutation": permutation_identity,
                    "units": ["mm", "mm", "mm", "MPa"], "all_original_nodes": True, "signed_stress_unchanged": True}
            cache["source_reads"][cad] = {"parse_once": True, "num_nodes": n, "source": source,
                "columns": names, "permutation": permutation_identity}
            ledger.value["GT_source_attempts"][cad]["status"] = "committed_owned_cache"
            ledger.persist(); d.write(folder / "manifest.json", cache); del values, permutation
        except BaseException as error:
            ledger.value["GT_source_attempts"][cad].update(status="terminal_uncommitted_read_negative", reason=str(error))
            ledger.persist(); cache.update(status="completed_negative", terminal=True,
                reason="GT interruption/invalid read; no silent reparse: " + str(error)); d.write(folder / "manifest.json", cache)
            raise Negative(cache["reason"]) from error
    d.require(set(cache["records"]) == set(contract["case_keys"]) and ledger.value["source_GT_parses"] == 20
              and ledger.value["U_target_case_reads"] == ledger.value["S_target_case_reads"] == 60,
              "Shared once-perCAD GT receipt differs")
    cache.update(status="completed", terminal=True, source_GT_parses=20, U_case_reads=60, S_case_reads=60)
    d.write(folder / "manifest.json", cache); return cache


def _read_metrics(d, np, output, contract, candidates, inventory, cache, ledger, clock):
    from .coverage_comparison_train import physical_metrics, aggregate
    results = {}
    for candidate in candidates:
        per = {key: None for key in contract["case_keys"]}; failures = {}
        for key in per:
            clock.check(); token = candidate["candidate_id"] + ":" + key
            record = d.read(_check_file(d, inventory[token], output))
            if not record["prediction_complete"]:
                failures[key] = record["failure"]; continue
            ledger.bump("shared_GT_array_reads")
            truth = np.load(_check_file(d, cache["records"][key]["physical"], output), mmap_mode="r", allow_pickle=False)
            ledger.bump("prediction_metric_array_reads")
            prediction = np.load(_check_file(d, record["physical"], output), mmap_mode="r", allow_pickle=False)
            per[key] = physical_metrics(np, truth, prediction); del truth, prediction
        overall = aggregate(per, 60)
        per_load = {load: aggregate({cad + "/" + load: per[cad + "/" + load] for cad in contract["selected_CADs"]}, 20) for load in LOADS}
        overall["ranking_eligible"] = overall["all_cases_complete"] and all(
            overall["case_mean_" + field + "_relative_l2"] is not None for field in ("u", "stress", "stress_hotspot"))
        overall["primary_score"] = overall["selection_score"] if overall["ranking_eligible"] else None
        for item in per_load.values():
            item["ranking_eligible"] = item["all_cases_complete"] and all(
                item["case_mean_" + field + "_relative_l2"] is not None for field in ("u", "stress", "stress_hotspot"))
            item["primary_score"] = item["selection_score"] if item["ranking_eligible"] else None
        results[candidate["candidate_id"]] = {"full60": overall, "per_load20": per_load, "true_failures": failures}
    return results


def _paired(d, results, contract):
    """Every pair retains20 CADs and all three loads; unavailable differences stay null."""
    paired = {}; ids = list(results)
    for index, left in enumerate(ids):
        for right in ids[index+1:]:
            rows = {}
            for cad in contract["selected_CADs"]:
                row = {}
                for metric in ("u_relative_l2", "stress_relative_l2", "stress_hotspot_relative_l2"):
                    deltas = []
                    for load in LOADS:
                        key = cad + "/" + load
                        a = results[left]["full60"]["per_case"][key]; b = results[right]["full60"]["per_case"][key]
                        av = a.get(metric) if a else None; bv = b.get(metric) if b else None
                        deltas.append(bv - av if isinstance(av, (float, int)) and isinstance(bv, (float, int)) else None)
                    row[metric + "_right_minus_left_mean3"] = sum(deltas) / 3 if all(v is not None for v in deltas) else None
                rows[cad] = row
            names = list(next(iter(rows.values())))
            paired[left + "__vs__" + right] = {"left": left, "right": right, "fixed_CAD_denominator": 20,
                "per_CAD": rows, "mean20": {name: sum(r[name] for r in rows.values()) / 20
                    if all(r[name] is not None and math.isfinite(r[name]) for r in rows.values()) else None for name in names}}
    return paired


def _null_results(candidates, keys, cads, reason):
    from .coverage_comparison_train import aggregate
    result = {}
    for candidate in candidates:
        per = {key: None for key in keys}
        full = aggregate(per, 60); full["ranking_eligible"] = False
        full["primary_score"] = None
        result[candidate["candidate_id"]] = {"full60": full,
            "per_load20": {load: aggregate({cad + "/" + load: None for cad in cads}, 20) for load in LOADS},
            "metrics_unavailable_reason": reason, "true_failures": {}}
        for item in result[candidate["candidate_id"]]["per_load20"].values():
            item.update(ranking_eligible=False, primary_score=None)
    return result


def run(plan_path=PLAN, start=False):
    if not start:
        return {"status": "source_only_ready", "kind": KIND, "actual_execution": False,
                "reads": False, "writes": False, "source_phase_numeric_reads": 0, "actual_release": False}
    # All imports and metadata work below are charged from the outer stage entry.
    from . import available_cad_pilot_data as d
    contract, plan = d.request_contract(plan_path); candidates, cads, keys = _science(d, contract, plan)
    d.release(contract, plan, purpose="evaluation"); guard = d.validate_external_guard(contract, "evaluation")
    predecessor = d.predecessor_gate(plan)
    manifest, input_contract, current = d.read_completed_inputs(plan_path)
    d.require(input_contract == contract and current == plan, "Input science/source identity changed")
    manifest = {**manifest, "output": str(d.path(plan["scientific"]["outputs"]["input_cache"]))}
    output = d.path(plan["scientific"]["outputs"]["readout"])
    from .coverage_comparison_train import lock
    held = lock(d.shared_gpu_lock(plan))
    try:
        held.__enter__()
    except (OSError, RuntimeError) as error:
        raise d.NotReady("Public GPU nonblocking lock is busy; no model/array/GT opened") from error
    clock = None; ledger = None; torch = None; cleanup = None; endpoints = {}; inventory = {}; proof = None
    negative = None; results = {}; phase = "setup"
    try:
        after_lock = d.predecessor_gate(plan)
        d.require(after_lock["PP_and_FNO"] == predecessor["PP_and_FNO"], "Predecessor changed while acquiring public GPU lock")
        output.mkdir(parents=True, exist_ok=True)
        d.require(not any((output / name).exists() for name in (
            "execution.json", "clock.json", "read_receipt.json", "endpoint_identities.json", "predictions", "shared_GT")),
                  "This immutable one-shot readout already attempted; no restart or cost reset")
        run_process = d.process_snapshot(os.getpid())
        d.require(run_process and run_process.get("start_ticks") is not None, "Actual evaluator PID/startticks missing")
        execution = {"kind": KIND, "scientific_digest": SCI, "contract_sha256": d.digest(contract),
            "run_process": run_process, "guard": guard, "predecessor_proof": predecessor,
            "source_sha256": contract["source_sha256"], "automatic_replay": False, "started_at_utc": utc()}
        # Exclusive attempt identity survives forced exit and forbids another attempt.
        with (output / "execution.json").open("x", encoding="utf-8") as handle:
            json.dump(execution, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.flush(); os.fsync(handle.fileno())
        clock = Clock(d, output, guard, run_process); ledger = Ledger(d, output)
        d.write(output / "status.json", {"status": "running", "phase": "predictions", "run_process": run_process})
        clock.check(); clock.phase("numeric_imports")
        import numpy as np
        complete_inputs = manifest.get("feasibility_pass") is True and set(manifest.get("records", {})) == set(keys) and all(
            r.get("feasibility_pass") is True and r.get("status") == "completed" for r in manifest.get("records", {}).values())
        if not complete_inputs:
            negative = "Fixed20 input-negative: " + manifest.get("reason", "incomplete/unsupported target-free inputs")
        else:
            import torch
            from .benchmark import FieldScaler
            scaler = FieldScaler(**contract["scaler"])
        for candidate in candidates:
            phase = "endpoint"; cid = candidate["candidate_id"]; model = None; endpoint_error = negative
            predict = query_grid = None; metadata = None
            endpoint = {"status": "binding", "candidate": candidate}
            if endpoint_error is None:
                try:
                    clock.phase("training_metadata_identity"); clock.check(torch)
                    metadata = _training_metadata(d, candidate, plan)
                    endpoint["metadata"] = metadata
                    clock.phase("model_initialization"); ledger.bump("model_initializations")
                    if candidate["method"].startswith("FNO_latent_query_"):
                        from .fno_local_condition_pair import model_class
                        torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                        model = model_class(torch)(candidate["method"]).to("cuda")
                        d.require(model.counts() == metadata["parameter_counts"], "FNO parameter identity differs")
                    else:
                        from .coverage_comparison_train import model_environment
                        _, _, model, original_scaler, cleanup, _, _, _, predict, query_grid = model_environment(manifest, candidate["method"])
                        d.require(d.digest(model.source_config) == d.digest(metadata["config"]["model"])
                                  and model.parameter_counts == metadata["parameter_counts"], "Original model constructor differs")
                    clock.check(torch); clock.phase("checkpoint_hash_load_bind")
                    endpoint = _bind_endpoint(d, np, torch, model, candidate, metadata, ledger, clock, endpoint)
                except Exception as error:
                    endpoint_error = type(error).__name__ + ": " + str(error)
                    if isinstance(error, (Negative, FloatingPointError)) or "out of memory" in str(error).lower():
                        negative = endpoint_error
                    endpoint.update(status="invalid", real_terminal_reason=endpoint_error)
            else:
                endpoint = {"status": "not_loaded", "candidate": candidate, "real_terminal_reason": endpoint_error}
            endpoints[cid] = endpoint; d.write(output / "endpoint_identities.json", endpoints)
            for key in keys:
                phase = "prediction"; reason = negative or endpoint_error
                scope = "input" if not complete_inputs else ("campaign" if negative else "endpoint")
                record = _terminal_record(d, output, candidate, key, contract, manifest["records"].get(key), endpoint, reason, scope)
                if reason is None:
                    case = None; began = None
                    try:
                        clock.phase("input_hash_read_transfer"); clock.check(torch)
                        case = _load_input(d, np, torch, manifest, key,
                            candidate["method"] == "fno" or candidate["method"].startswith("FNO_latent_query_"), ledger, clock)
                        clock.phase("allN_inference"); ledger.bump("prediction_attempts")
                        torch.cuda.synchronize(); began = time.monotonic()
                        encoded = _predict(np, torch, model, candidate, case, predict, query_grid, clock)
                        seconds = time.monotonic() - began; n = manifest["records"][key]["num_nodes"]
                        if encoded.shape != (n, 4) or not np.isfinite(encoded).all():
                            raise FloatingPointError("Incomplete/nonfinite allN encoded4 prediction")
                        physical = scaler.decode(encoded)
                        if physical.shape != (n, 4) or not np.isfinite(physical).all():
                            raise FloatingPointError("Nonfinite old-scaler physical4 prediction")
                        clock.phase("prediction_exclusive_persistence"); folder = _record_path(output, cid, key).parent
                        encoded = np.ascontiguousarray(encoded, dtype=np.float32)
                        if not np.isfinite(encoded).all():
                            raise FloatingPointError("Encoded4 FP32 cache representation overflows")
                        encoded_item = _save_array(d, np, folder / "encoded4.npy", encoded)
                        physical_item = _save_array(d, np, folder / "physical4.npy", physical)
                        ledger.bump("prediction_cache_writes", 2); ledger.bump("finite_allN_predictions")
                        record.update(prediction_complete=True, finite_allN=True, failure=None, all_original_nodes=True,
                            node_count=n, encoded=encoded_item, physical=physical_item, inference_seconds=seconds)
                        del encoded, physical
                    except Exception as error:
                        text = type(error).__name__ + ": " + str(error)
                        negative = text
                        record.update(prediction_complete=False, finite_allN=False,
                            failure={"scope": "campaign", "reason": text, "real_terminal_outcome": True})
                        if began is not None:
                            record["failed_inference_and_persistence_seconds"] = time.monotonic() - began
                        # Partial owned arrays are recorded as abandoned, never accepted as successes.
                        record.pop("encoded", None); record.pop("physical", None)
                    finally:
                        if case is not None:
                            del case
                _commit_record(d, output, record, inventory)
            if torch is not None:
                if model is not None:
                    del model
                if cleanup is not None:
                    cleanup(); cleanup = None
                clock.check(torch) if negative is None else None
                torch.cuda.empty_cache()
        phase = "barrier"; clock.phase("barrier_full540_owned_identity_validation")
        # A genuine budget stop never gets extra numeric work to unlock GT.
        if negative is not None and complete_inputs:
            clock.check()
        proof = commit_barrier(d, np, output, contract, plan, manifest, endpoints, inventory, ledger, clock, negative)
        if proof["GT_eligible"] and negative is None:
            phase = "shared_GT"; cache = _shared_gt(d, np, output, contract, manifest, proof, ledger, clock)
            phase = "metrics"; clock.phase("full60_and_perload20_physical_metrics")
            results = _read_metrics(d, np, output, contract, candidates, inventory, cache, ledger, clock)
        else:
            negative = negative or "No finite valid allN predictions; numeric GT unnecessary and remains zero"
            results = _null_results(candidates, keys, cads, negative)
            for candidate in candidates:
                results[candidate["candidate_id"]]["true_failures"] = {key:
                    d.read(inventory[candidate["candidate_id"] + ":" + key]["path"])["failure"] for key in keys}
    except BaseException as error:
        if clock is None:
            raise
        negative = type(error).__name__ + ": " + str(error)
    finally:
        if clock is not None:
            if torch is not None:
                try:
                    clock.check(torch); torch.cuda.synchronize()
                except Exception as error:
                    negative = negative or type(error).__name__ + ": " + str(error)
                if cleanup is not None:
                    try:
                        cleanup()
                    except Exception as error:
                        negative = negative or "Actual model cleanup failure: " + str(error)
            clock.phase("terminal_metadata_persistence")
            ledger.persist()
            if not results:
                results = _null_results(candidates, keys, cads, negative or "Readout did not complete")
            for cid, item in results.items():
                item["endpoint"] = endpoints.get(cid)
                item["actual_inference_seconds"] = 0.
                item["failed_inference_and_persistence_seconds"] = 0.
                for key in keys:
                    token = cid + ":" + key
                    if token in inventory:
                        record = d.read(inventory[token]["path"])
                        item["actual_inference_seconds"] += record.get("inference_seconds", 0.)
                        item["failed_inference_and_persistence_seconds"] += record.get("failed_inference_and_persistence_seconds", 0.)
                        if record.get("failure"):
                            item["true_failures"][key] = record["failure"]
            summary = {"kind": KIND, "status": "completed_negative" if negative else "completed", "terminal": True,
                "scientific_digest": SCI, "source_sha256": contract["source_sha256"], "contract_sha256": d.digest(contract),
                "candidate_denominator": 9, "case_denominator": 60, "CAD_denominator": 20,
                "run_process": execution["run_process"], "predecessor_proof": predecessor,
                "endpoints": endpoints, "prediction_inventory": inventory, "prediction_barrier": proof,
                "results": results, "paired_CAD20_differences": _paired(d, results, contract) if results else {},
                "read_receipt": ledger.value, "costs": clock.value, "terminal_negative_reason": negative,
                "terminal_phase": phase, "automatic_replay": False, "optimizer_restores": 0, "training_updates": 0,
                "scaler_fits": 0, "primary_score_definition": "0.5*(full60 mean U relativeL2+full60 mean signedS relativeL2)",
                "interpretation": "Availability-conditioned first20 official-TRAIN CADs; all candidates trained only old40. "
                    "Independent candidate pilot, single seed; original DEV20 and final test remain sealed. "
                    "Capacity, recipe, exposure and inherited train cost differ; failures do not establish architecture inferiority. "
                    "No family-ceiling, author-exact-replay or permanent/unbiased final-test claim. "
                    "These20 cease to be held if included in future200TRAIN.",
                "external_guard_terminal_receipt_required": True}
            gt_manifest = output / "shared_GT" / "manifest.json"
            summary["shared_GT_manifest"] = _identity(d, gt_manifest) if gt_manifest.is_file() else None
            summary["preparation_costs"] = {"internal_clock": manifest.get("preparation_clock", manifest.get("prepare_clock")),
                "authoritative_wall_seconds": manifest.get("authoritative_preparation_wall_seconds"),
                "external_terminal": manifest.get("preparation_external_terminal")}
            summary["eligible_score_order"] = sorted([
                {"candidate_id": cid, "primary_score": item["full60"]["primary_score"]}
                for cid, item in results.items() if item["full60"].get("ranking_eligible") is True],
                key=lambda row: row["primary_score"])
            summary["equal_scores_retained"] = True
            d.write(output / "summary.json", summary)
            d.write(output / "status.json", {"status": summary["status"], "phase": "terminal",
                "run_process": execution["run_process"], "reason": negative, "updated_at_utc": utc()})
            clock.close(negative or "fixed540_sharedGT20_readout_completed")
        held.__exit__(None, None, None)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN); parser.add_argument("--start", action="store_true")
    args = parser.parse_args(); result = run(args.plan, start=args.start)
    print(json.dumps({"status": result["status"], "kind": KIND}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
