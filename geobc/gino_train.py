"""Full-original-node GINO screen on the existing TRAIN40/VAL5 known-load cache.

Source checks use only the standard library and small JSON/file metadata.
Actual execution requires the separately released full-input resource result
and a frozen training recipe. No test/dia preparation or scaler fitting occurs.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

from . import solver_selection_train as shared

ROOT = shared.ROOT
ENDPOINTS = (0, 120, 240, 360, 480)
QUERY_CHUNK = 2048
ARRAY_NAMES = tuple(f"{orientation}_{member}" for orientation in ("grid_queries", "point_queries")
                    for member in ("row_ptr", "neighbors_index", "squared_distances")) + ("grid_axis", "grid_xyz")
SOURCE_PATHS = ("geobc/gino_train.py", "geobc/gino_solver.py", "geobc/regular_grid_radius.py",
                "geobc/solver_selection_data.py", "geobc/solver_selection_train.py",
                "geobc/benchmark.py", "geobc/conditioned_field.py", "third_party/neuraloperator_source.json")
ROLE_NAMES = [f"support_reference_constrained_dof_{i}" for i in range(1, 7)] + ["loaded_interface_membership"]


def stat_record(path):
    item = shared.host_path(path).stat()
    return {"bytes": item.st_size, "mtime_ns": item.st_mtime_ns}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _check_recipe(plan):
    fixed = {"method": "gino", "seed": 2026, "cloud_seed": 2026, "train_loads": ["ver", "hor", "tor"],
             "steps": 480, "passes": 4, "eval_steps": list(ENDPOINTS), "optimizer": "Adam",
             "learning_rate": .001, "weight_decay": 0., "cloud_points": 1024, "geometry_dim": 32,
             "grid_resolution": 64, "in_radius": .033, "out_radius": .033,
             "fno_n_modes": [24, 24, 22], "fno_hidden_channels": 16, "fno_n_layers": 4,
             "input_query_chunk": 2048, "output_query_chunk": QUERY_CHUNK, "radius_chunk_points": 8192,
             "stress_transform": "asinh", "channels": [0, 1, 2, 3], "checkpoint_every_updates": 30,
             "training_readout_keys": ["101_479/ver", "107_320/hor", "10_58/tor"]}
    if plan.get("kind") != "gino_training_screen" or any(plan["recipe"].get(k) != v for k, v in fixed.items()):
        raise ValueError("GINO plan differs from the fixed full-node screening recipe")
    if plan["access"]["test_or_dia_arrays"] is not False or plan["access"]["sealed_new_test_values"] is not False:
        raise ValueError("Final/dia/sealed next-cycle fields are outside this training scope")


def request_contract(plan_path):
    """Return (contract, shared manifest) without scientific imports or reads."""
    path = shared.host_path(plan_path); plan = shared.read_json(path); _check_recipe(plan)
    manifest_path = shared.host_path(plan["data_cache"]) / "manifest.json"
    manifest = shared.read_json(manifest_path); shared._check_manifest(manifest)
    split_path = shared.host_path(plan["existing_split"]); split = shared.read_json(split_path)
    expected = {group: [f"{cad}/{load}" for cad in split["groups"][group] for load in ("ver", "hor", "tor")]
                for group in ("train", "validation")}
    cache_contract = manifest["contract"]
    if (manifest["train_keys"] != expected["train"] or manifest["validation_keys"] != expected["validation"]
            or cache_contract["split_sha256"] != shared.source_sha(split_path)
            or cache_contract["cloud_seed"] != 2026 or cache_contract["grid_resolution"] != 64
            or cache_contract["cloud_points"] != 1024 or cache_contract["stress_transform"] != "asinh"):
        raise ValueError("The existing TRAIN-only scaler/split/cloud cache does not match")
    # File stat witnesses preserve the existing producer contract, without NPZ/NPY reads.
    for item in cache_contract["sources"]:
        if stat_record(item["path"]) != {k: item[k] for k in ("bytes", "mtime_ns")}:
            raise ValueError("An existing prepared TRAIN/VAL source stat changed")
    geometries = {}
    for key in manifest["train_keys"] + manifest["validation_keys"]:
        record = manifest["records"][key]; cad = key.split("/")[0]
        if record["group"] != ("train" if key in manifest["train_keys"] else "validation") or record["cad_id"] != cad:
            raise ValueError("Shared case group/identity differs")
        row = {"cad_id": cad, "num_nodes": record["num_nodes"], "xyz_path": record["xyz_path"],
               "xyz_stat": stat_record(record["xyz_path"])}
        if cad in geometries and geometries[cad] != row:
            raise ValueError("Known loads do not share the same canonical CAD node order")
        geometries[cad] = row
    if len(geometries) != 45:
        raise ValueError("Exactly forty TRAIN and five validation CAD geometries are required")
    resource_path = shared.host_path(plan["prerequisites"]["resource_summary"])
    resource = shared.read_json(resource_path) if resource_path.is_file() else None
    resource_identity = {"path": str(resource_path), "status": resource.get("status") if resource else "pending",
                         "sha256": shared.source_sha(resource_path) if resource else None}
    science = {k: plan[k] for k in ("recipe", "existing_split", "prepared_data", "data_cache", "radius_cache",
                                   "radius_pilot_summary", "prerequisites", "access")}
    contract = {"schema": "geobc_gino_training_screen_v1", "plan": str(path), "recipe": plan["recipe"],
                "recipe_sha256": digest(science), "method": "gino", "cache_contract": cache_contract,
                "cache_manifest_sha256": shared.source_sha(manifest_path), "scaler": manifest["scaler"],
                "case_order_sha256": manifest["case_order_sha256"], "train_keys": manifest["train_keys"],
                "validation_keys": manifest["validation_keys"], "training_readout_keys": manifest["training_readout_keys"],
                "geometries": geometries, "resource": resource_identity,
                "source_sha256": {name: shared.source_sha(name) for name in SOURCE_PATHS}}
    return contract, manifest


def source_check(plan_path):
    contract, _ = request_contract(plan_path)
    for name in ("geobc/gino_train.py", "geobc/gino_solver.py", "geobc/regular_grid_radius.py"):
        ast.parse(shared.host_path(name).read_text(encoding="utf-8"), filename=name)
    return {"status": "source_ready", "contract_sha256": digest(contract),
            "recipe_sha256": contract["recipe_sha256"], "source_sha256": contract["source_sha256"],
            "cache_manifest_sha256": contract["cache_manifest_sha256"], "scaler_sha256": digest(contract["scaler"]),
            "case_order_sha256": contract["case_order_sha256"], "TRAIN_cases": 120, "VAL_cases": 15,
            "CADs": len(contract["geometries"]), "resource": contract["resource"],
            "training_rules_frozen": shared.read_json(plan_path).get("training_rules_frozen", False),
            "resource_prerequisite": "Required at actual release; a pending resource is permitted for this source check",
            "scientific_arrays_read": 0, "models_constructed": 0, "CUDA_initialized": False,
            "training_updates": 0, "cache_builds": 0}


def _resource_gate(plan):
    """Source/metadata gate only. Exact success schema is declared in the proposal."""
    if plan.get("training_rules_frozen") is not True:
        raise ValueError("GINO training recipe remains a proposal; root must freeze it before execution")
    result = shared.read_json(plan["prerequisites"]["resource_summary"])
    success_key = plan["prerequisites"]["resource_success_key"]
    value = result
    for key in success_key.split("."):
        value = value[key]
    if result.get("status") != "completed" or value is not True:
        raise ValueError("The separately released full-input/512-query GINO resource prerequisite did not pass")
    identity = result["config"]["contract"]
    if (identity["schema"] != "geobc_gino_native_resource_v1" or not all(result["checks"].values())
            or identity["recipe"]["query_count"] != 512 or identity["recipe"]["input_query_chunk"] != 2048
            or identity["radius_summary"]["sha256"] != shared.source_sha(plan["radius_pilot_summary"])
            or identity["selection_manifest"]["sha256"] != shared.source_sha(shared.host_path(plan["data_cache"])/"manifest.json")
            or any(identity["source_sha256"][name] != shared.source_sha(name)
                   for name in ("geobc/gino_solver.py", "geobc/solver_selection_data.py", "geobc/conditioned_field.py"))):
        raise ValueError("Resource acceptance belongs to a different input cache or GINO implementation")


class RadiusCache:
    """Persist full exact CSR once per CAD; keep only the active CAD on device."""
    def __init__(self, plan, contract, torch, np, device, costs, progress):
        self.plan, self.contract, self.torch, self.np = plan, contract, torch, np
        self.device, self.costs, self.progress = device, costs, progress
        self.root = shared.host_path(plan["radius_cache"])
        self.records = {}; self.active_cad = None; self.active = None

    def _inventory(self, cad, xyz):
        expected = self.contract["geometries"][cad]
        directory = self.root / cad; receipt = directory / "manifest.json"
        if receipt.is_file():
            saved = shared.read_json(receipt)
            if saved["status"] != "completed" or saved["geometry"] != expected or saved["radius"] != .033 or saved["resolution"] != 64:
                raise ValueError("Persistent CAD radius-cache identity differs")
            return saved, False
        if cad == "101_479":
            pilot = shared.read_json(self.plan["radius_pilot_summary"])
            source = pilot["config"]["contract"]["source"]
            if (pilot.get("status") != "completed" or pilot.get("feasible") is not True
                    or source["stat"] != expected["xyz_stat"]
                    or pilot["stats"]["construction"]["R"] != 64 or pilot["stats"]["construction"]["radius"] != .033
                    or shared.host_path(source["xyz_path"]) != shared.host_path(expected["xyz_path"])):
                raise ValueError("The existing 101_479 input-only radius pilot cannot be reused")
            saved = {"status": "completed", "geometry": expected, "radius": .033, "resolution": 64,
                     "inventory": pilot["inventory"], "checks": pilot["checks"],
                     "construction": pilot["stats"]["construction"], "origin": self.plan["radius_pilot_summary"],
                     "cache_bytes": pilot["cache_bytes"], "costs": {"build_seconds": 0., "save_seconds": 0.}}
            return saved, False
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Incomplete CAD CSR cache is preserved; no silent rebuild or neighbor change")
        from .regular_grid_radius import build_regular_grid_radius
        self.progress("building_radius_cache", cad=cad, nodes=len(xyz))
        shared._sync(self.torch, self.device); began = time.monotonic()
        built = build_regular_grid_radius(xyz, resolution=64, radius=.033, chunk_points=8192,
                                         progress=lambda row: self.progress("building_radius_cache", cad=cad, radius_progress=row))
        shared._sync(self.torch, self.device); build_seconds = time.monotonic()-began
        if not all(built["checks"].values()) or bool((built["point_queries"]["row_ptr"][1:] == built["point_queries"]["row_ptr"][:-1]).any()):
            raise ValueError("Full exact radius CSR invariants/point coverage failed; no neighborhood repair")
        self.costs["radius_build_seconds"] += build_seconds
        self.costs["radius_build_count"] += 1
        directory.mkdir(parents=True, exist_ok=True); began = time.monotonic(); inventory = {}
        for name in ARRAY_NAMES:
            if name in ("grid_axis", "grid_xyz"):
                tensor = built[name]
            else:
                orientation, member = next((prefix, name[len(prefix)+1:]) for prefix in ("grid_queries", "point_queries") if name.startswith(prefix + "_"))
                tensor = built[orientation][member]
            value = tensor.detach().cpu().numpy(); path = directory / (name + ".npy")
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                self.np.save(handle, value, allow_pickle=False)
            temporary.replace(path)
            inventory[name] = {"path": str(path), "shape": list(value.shape), "dtype": str(value.dtype),
                               "array_payload_bytes": value.nbytes, **stat_record(path)}
        save_seconds = time.monotonic()-began; self.costs["radius_save_seconds"] += save_seconds
        saved = {"status": "completed", "geometry": expected, "radius": .033, "resolution": 64,
                 "inventory": inventory, "checks": built["checks"], "construction": built["construction"],
                 "origin": "on-demand exact regular_grid_radius; no target/feature input",
                 "cache_bytes": sum(item["bytes"] for item in inventory.values()),
                 "costs": {"build_seconds": build_seconds, "save_seconds": save_seconds}}
        shared.write_json(receipt, saved)
        del built
        return saved, True

    def attach(self, case):
        cad = case["key"].split("/")[0]
        if cad != self.active_cad:
            self.active = None; self.active_cad = None
            saved, newly_built = self._inventory(cad, case["xyz"])
            if set(saved["inventory"]) != set(ARRAY_NAMES):
                raise ValueError("Exact CAD CSR cache must retain all eight original arrays")
            shared._sync(self.torch, self.device); began = time.monotonic(); arrays = {}
            for name, item in saved["inventory"].items():
                path = shared.host_path(item["path"])
                if stat_record(path) != {k: item[k] for k in ("bytes", "mtime_ns")}:
                    raise ValueError("CAD CSR cache file stat differs; no silent repair")
                value = self.np.load(path, mmap_mode="r", allow_pickle=False)
                if list(value.shape) != item["shape"] or str(value.dtype) != item["dtype"]:
                    raise ValueError("CAD CSR cached shape/dtype differs")
                arrays[name] = self.torch.as_tensor(self.np.array(value, copy=True), device=self.device)
            shared._sync(self.torch, self.device); self.costs["radius_read_seconds"] += time.monotonic()-began
            self.costs["radius_read_count"] += 1
            if not newly_built:
                self.costs["radius_cache_reuse_count"] += 1
            self.active = {"grid_xyz": arrays["grid_xyz"],
                           "grid_csr": {member: arrays["grid_queries_"+member] for member in ("row_ptr", "neighbors_index", "squared_distances")},
                           "point_csr": {member: arrays["point_queries_"+member] for member in ("row_ptr", "neighbors_index", "squared_distances")}}
            self.active_cad = cad; self.records[cad] = saved
        case.update(self.active)
        if len(case["point_csr"]["row_ptr"]) != len(case["xyz"])+1:
            raise ValueError("CSR canonical node order/count differs from this complete case")
        return case

    def receipt(self):
        return {"CADs_accessed": sorted(self.records), "unique_CADs": len(self.records),
                "total_unique_cache_bytes": sum(row["cache_bytes"] for row in self.records.values()),
                "per_CAD": self.records, "geometry_build_not_in_train_seconds": True,
                "full_neighbor_policy": "All original nodes and exact radius pairs retained; no cap/drop/sample/cdist"}


def _train_case(torch, model, case, device):
    """Exact chain rule: one input graph, accumulated query-leaf gradient."""
    shared._sync(torch, device); began = time.monotonic()
    original = model.encode_case(case)
    if not original.latent.requires_grad:
        raise RuntimeError("The complete input encode graph must remain differentiable")
    leaf = original.latent.detach().requires_grad_(True)
    query_state = replace(original, latent=leaf)
    count = len(case["xyz"]); target = case["target"]
    if target.shape != (count, 4):
        raise ValueError("Complete encoded signed4 target must be [N,4]")
    shared._sync(torch, device); encode_seconds = time.monotonic()-began
    began = time.monotonic(); total = torch.zeros((), device=device)
    for begin in range(0, count, QUERY_CHUNK):
        end = min(begin+QUERY_CHUNK, count)
        rows = torch.arange(begin, end, dtype=torch.int64, device=device)
        prediction = model.query(query_state, rows)
        if prediction.shape != (end-begin, 4):
            raise ValueError("GINO original-node query must return encoded [Q,4]")
        loss = (prediction.float()-target[begin:end]).square().mean() * ((end-begin)/count)
        loss.backward(); total += loss.detach()
        del prediction, loss, rows
    shared._sync(torch, device); query_backward_seconds = time.monotonic()-began
    if leaf.grad is None or not bool(torch.isfinite(leaf.grad).all()) or not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite/missing full-node loss or accumulated latent gradient")
    began = time.monotonic(); original.latent.backward(leaf.grad)
    shared._sync(torch, device); encode_backward_seconds = time.monotonic()-began
    return float(total), {"encode_seconds": encode_seconds, "query_forward_backward_seconds": query_backward_seconds,
                          "encode_backward_seconds": encode_backward_seconds, "output_chunks": math.ceil(count/QUERY_CHUNK),
                          "original_nodes": count, "input_encode_calls": 1, "input_graph_backward_calls": 1}


def _build_model(torch, device):
    """Construct the one frequency-matched candidate without a smaller model."""
    from .gino_solver import GINOSolver, load_official, DisabledNeighborSearch, COMMIT, PREFIX
    from .conditioned_field import PointNetGeometryEncoder

    class MatchedFrequencyGINO(GINOSolver):
        def __init__(self):
            torch.nn.Module.__init__(self)
            torch.manual_seed(2026)
            # Preserve the base adapter's initialization order and all other
            # constructor defaults. Only the requested spectral modes differ.
            self.encoder = PointNetGeometryEncoder(descriptor_dim=32, point_width=64)
            OfficialGINO, self.integral_module = load_official()
            self.solver = OfficialGINO(in_channels=25, out_channels=4, fno_in_channels=25,
                latent_feature_channels=32, fno_n_modes=(24, 24, 22), fno_hidden_channels=16,
                fno_n_layers=4, in_gno_radius=.033, out_gno_radius=.033,
                gno_weighting_function="half_cos", gno_use_open3d=False, gno_use_torch_scatter=False)
            self.solver.gno_in.neighbor_search = DisabledNeighborSearch()
            self.solver.gno_out.neighbor_search = DisabledNeighborSearch()
            self.query_chunk = 2048
            self.source_config = {
                "official_commit": COMMIT, "version": "2.0.0", "namespace": PREFIX,
                "official_class": "GINO", "adapter_base": "GINOSolver encode_case/query inherited unchanged",
                "input_channels": 25, "output_channels": 4, "FNO_input_before_latent": 25,
                "latent_feature_channels": 32, "FNO_input_after_latent": 57,
                "FNO_hidden": 16, "FNO_modes": [24, 24, 22], "FNO_layers": 4,
                "resolved_frequency_match": "Old FNO m12: x/y24 total and real-FFT z12; official modes24/24/22: x/y24 and real-FFT z12",
                "seed": 2026, "grid_resolution": 64, "input_query_chunk": 2048, "output_query_chunk": QUERY_CHUNK,
                "default_official_parameters": "All base-resource constructor settings/defaults except fno_n_modes; original kernels/lifting/FNO/projection intact",
                "case_encoder": "Shared PointNet xyz-only1024/global32, live broadcast to every latent voxel",
                "input_reduction": "Unweighted mean; empty rows zero",
                "output_reduction": "Original half_cos of cached float64 squared distances followed by sum",
                "runtime_adapter": "Pinned private IntegralTransform GPU segment_reduce; cached complete CSR replaces search",
                "input_checkpoint": "Inherited use_reentrant=False query chunks2048; no edge truncation",
                "kernel_source_unchanged": True, "neighbor_search_executed": False,
                "smaller_candidate_constructed": False,
                "resource_scope": "Prior acceptance covers low-frequency modes12 full input/512 outputs; matched modes24/24/22 and full-output2048 training first accepted only in the actual complete case",
                "capacity_limit": "Retained frequency bands/hidden16/four layers matched; lifting/spectral/skip implementations differ, actual parameter counts disclosed, no exact capacity or GNO-only causal claim"
            }

    return MatchedFrequencyGINO().to(device)


def _gradient_check(torch, model):
    groups = {name: [] for name in ("total", "encoder", "input_gno", "lifting", "fno", "output_gno", "projection")}
    missing = []; finite = True
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name); continue
        grad = parameter.grad.detach(); finite &= bool(torch.isfinite(grad).all())
        norm = float(grad.abs().square().sum()); groups["total"].append(norm)
        for group, prefix in (("encoder", "encoder."), ("input_gno", "solver.gno_in."),
                              ("lifting", "solver.lifting."), ("fno", "solver.fno_blocks."), ("output_gno", "solver.gno_out."),
                              ("projection", "solver.projection.")):
            if name.startswith(prefix):
                groups[group].append(norm)
    norms = {group: math.sqrt(sum(values)) for group, values in groups.items()}
    passed = finite and all(math.isfinite(norm) and norm > 0 for norm in norms.values())
    record = {"passed": passed, "all_present_gradients_finite": finite, "L2_gradient_norms": norms,
              "missing_gradient_names": missing,
              "missing_policy": "Report upstream inactive parameters; require finite active gradients and nonzero gradients across the complete encoder/input/lifting/FNO/output/projection graph groups; complex gradients use magnitude squared",
              "source": "First real all-original-node case loss; no toy/probe/native warmup"}
    if not passed:
        raise FloatingPointError("First complete-case gradient check failed: " + json.dumps(record))
    return record


def _predict_case(torch, np, model, case, device):
    state = model.encode_case(case)
    result = np.empty((len(case["xyz"]), 4), dtype=np.float32)
    for begin in range(0, len(result), QUERY_CHUNK):
        end = min(begin+QUERY_CHUNK, len(result))
        rows = torch.arange(begin, end, device=device, dtype=torch.int64)
        result[begin:end] = model.query(state, rows).float().cpu().numpy()
    return result


def _bc_metrics(np, features, truth, prediction):
    truth = np.asarray(truth, dtype=np.float64); prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction-truth; result = {}
    for index, name in enumerate(ROLE_NAMES):
        mask = features[:, index] == 1; count = int(mask.sum())
        if not count:
            result[name] = {"nodes": 0, "metrics": None, "reason": "No positive source-role nodes"}; continue
        u_ref = float(np.square(truth[mask, :3]).sum()); s_ref = float(np.square(truth[mask, 3]).sum())
        u_error = float(np.square(error[mask, :3]).sum()); s_error = float(np.square(error[mask, 3]).sum())
        result[name] = {"nodes": count, "u_vector_error_RMS_mm": math.sqrt(u_error/count),
                        "stress_RMSE_MPa": math.sqrt(s_error/count), "stress_MAE_MPa": float(np.abs(error[mask, 3]).mean()),
                        "u_relative_l2": math.sqrt(u_error/u_ref) if u_ref > 0 else None,
                        "stress_relative_l2": math.sqrt(s_error/s_ref) if s_ref > 0 else None,
                        "zero_reference_reason": "Relative error undefined for a zero physical regional norm" if u_ref == 0 or s_ref == 0 else None}
    return {"positive_definition": "Exact original feature[:,0:7] == 1; regions may overlap",
            "limit": "Support-reference DOF roles and loaded membership are descriptive source regions, not prescribed node Dirichlet equations",
            "regions": result}


def run(args):
    if args.source_check:
        result = source_check(args.plan); print(json.dumps(result, indent=2), flush=True); return result
    if args.output is None:
        raise ValueError("--output is required for actual training")
    plan = shared.read_json(args.plan); _resource_gate(plan)
    contract, manifest = request_contract(args.plan); output = shared.host_path(args.output)
    if (output / "summary.json").is_file():
        prior = shared.read_json(output / "summary.json")
        if prior.get("status") == "completed":
            if prior["config"]["contract"] != contract or shared.read_json(output / "config.json") != prior["config"]:
                raise ValueError("Completed GINO screen identity differs")
            return prior
    resume = (output / "last.pt" if args.resume == "last" else shared.host_path(args.resume)) if args.resume else None
    if resume and not resume.is_file():
        raise FileNotFoundError(resume)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError("Incomplete output is preserved; use explicit --resume")
    output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic(); prior_elapsed = 0.; completed_step = 0
    state = {"method": "gino", "step": 0, "total_steps": 480, "started_at_utc": shared.utc_now(), "evaluated_steps": []}
    costs = {name: 0. for name in ("train_seconds", "gpu_step_seconds", "base_case_load_seconds", "inference_seconds", "evaluation_seconds",
                                  "radius_build_seconds", "radius_read_seconds", "radius_save_seconds", "checkpoint_seconds")}
    costs.update({name: 0 for name in ("train_case_exposures", "train_node_exposures", "inference_case_exposures",
                                      "inference_node_exposures", "radius_build_count", "radius_read_count", "radius_cache_reuse_count")})
    costs["train_seconds_definition"] = "Like the existing screen, begin before load_case: case/target read + H2D + cached CSR read/H2D + complete full-node update/Adam/sync; subtract only first-time exact CSR construction and save, reported separately"
    costs["gpu_step_seconds_definition"] = "Synchronized model encode/query-backward/input-backward/Adam spans; excludes data/cache IO and first-update diagnostic bookkeeping"
    cache = None; config = {"contract": contract}; prior_peak = {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    def status(phase, **details):
        state.update(details, phase=phase, updated_at_utc=shared.utc_now(), elapsed_seconds=prior_elapsed+time.monotonic()-began,
                     costs=dict(costs))
        trained = costs["train_case_exposures"]; eta = None
        if trained:
            eta = costs["train_seconds"]/trained*(480-completed_step)
            if state["evaluated_steps"]:
                eta += costs["evaluation_seconds"]/len(state["evaluated_steps"])*(len(ENDPOINTS)-len(state["evaluated_steps"]))
        state["eta_seconds"] = 0. if phase == "completed" else None if phase == "failed" else eta
        state["eta_limit"] = "Observed case/evaluation rates; not a prediction of remaining unseen-CAD CSR build costs"
        state["estimated_completion_utc"] = (datetime.now(timezone.utc)+timedelta(seconds=state["eta_seconds"])).isoformat() if state["eta_seconds"] is not None else None
        shared.write_json(output / "status.json", state)
    status("loading_model_after_resource_release")
    try:
        import numpy as np
        import torch
        from .benchmark import FieldScaler
        from .solver_selection_data import load_case, read_cache_metadata
        if read_cache_metadata(plan) != manifest:
            raise ValueError("Shared data metadata changed between contract and actual loading")
        device = torch.device("cuda")
        random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        model = _build_model(torch, device)
        parameter_counts = {"total": sum(p.numel() for p in model.parameters()),
                            "encoder": sum(p.numel() for p in model.encoder.parameters()),
                            "solver": sum(p.numel() for p in model.solver.parameters())}
        # The original spectral tensors are complex; foreach Adam retains
        # PyTorch's complex view-as-real updates without a fused dtype assumption.
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0., fused=False, foreach=True)
        scaler = FieldScaler(**manifest["scaler"])
        config = {"contract": contract, "method": "gino", "recipe": plan["recipe"], "model": model.source_config,
                  "parameter_counts": parameter_counts, "case_order": manifest["case_order"],
                  "scaler": manifest["scaler"], "precision": "float32 Torch/PointNet; TF32 enabled; no autocast; radius squared distances float64",
                  "gradient_protocol": "Full input encode once; detached latent leaf accumulates all node-weighted query losses, then original latent.backward(leaf.grad) once before Adam",
                  "resource_limit": "Prerequisite is lower-frequency modes12 full input +512 output; matched-frequency modes24/24/22 and full2048 all-node training first accepted by actual execution, never by this source check"}
        history = []; best_score = math.inf; best_step = None; best_metrics = None; gradient_check = None
        exposures = {key: 0 for key in manifest["train_keys"]}; passes = []
        _, shuffle_rng = shared.case_order(manifest["train_keys"], 2026)
        if resume:
            saved = torch.load(resume, map_location=device, weights_only=False)
            if saved["config"] != config or saved["case_order_sha256"] != manifest["case_order_sha256"]:
                raise ValueError("Resume geometry/data/model/scaler/schedule identity differs")
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
            shared._restore_rng(torch, np, saved["rng"])
            completed_step = saved["step"]
            if saved["next_case_position"] != completed_step:
                raise ValueError("Resume case-order position differs")
            costs = saved["costs"]; exposures = saved["case_exposure_counts"]; history = saved["history"]
            passes = saved["pass_costs"]; best_score = saved["best_score"]; best_step = saved["best_step"]
            best_metrics = saved["best_metrics"]; gradient_check = saved["first_update_gradient_check"]
            prior_elapsed = saved["elapsed_seconds"]; prior_peak = saved["memory"]; shuffle_rng = saved["case_shuffle_rng_state"]
            state["evaluated_steps"] = [row["step"] for row in history]
            (output / "metrics.jsonl").write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in history), encoding="utf-8")
            # last.pt is the authoritative transaction after an interrupted
            # best/last write; restore its selected best endpoint artifact.
            if best_step is not None:
                selected = resume if best_step == completed_step else resume.parent / f"step{best_step}.pt"
                if not selected.is_file():
                    raise FileNotFoundError("Resume needs the saved validation-best endpoint: " + str(selected))
                if selected.resolve() != (output / "best.pt").resolve():
                    shutil.copy2(selected, output / "best.pt")
                shared.write_json(output / "metrics_best.json", best_metrics)
        shared.write_json(output / "config.json", config)
        torch.cuda.reset_peak_memory_stats(device)
        cache = RadiusCache(plan, contract, torch, np, device, costs, status)
        if resume:
            cache.records = saved["radius_cache"]["per_CAD"]
        def memory():
            actual = shared._memory(torch, device)
            return {name: max(value, prior_peak[name]) for name, value in actual.items()}
        def elapsed():
            return prior_elapsed+time.monotonic()-began
        def payload():
            return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                    "step": completed_step, "next_case_position": completed_step, "case_order": manifest["case_order"],
                    "case_order_sha256": manifest["case_order_sha256"], "case_shuffle_rng_state": shuffle_rng,
                    "rng": shared._rng_state(torch, np), "history": history, "best_score": best_score,
                    "best_step": best_step, "best_metrics": best_metrics, "costs": dict(costs),
                    "case_exposure_counts": dict(exposures), "pass_costs": passes,
                    "first_update_gradient_check": gradient_check, "radius_cache": cache.receipt(),
                    "elapsed_seconds": elapsed(), "memory": memory()}
        def save(path):
            stage = time.monotonic(); shared._checkpoint(torch, path, payload())
            costs["checkpoint_seconds"] += time.monotonic()-stage
        def case_for(key):
            stage = time.monotonic(); case = load_case(manifest["records"][key], device)
            shared._sync(torch, device); costs["base_case_load_seconds"] += time.monotonic()-stage
            return cache.attach(case)
        def evaluate(step, loss):
            nonlocal best_score, best_step, best_metrics
            model.eval(); evaluation_began = time.monotonic(); groups = {}; status("evaluating", step=step)
            with torch.no_grad():
                for group, keys in (("train_readout", manifest["training_readout_keys"]), ("validation", manifest["validation_keys"])):
                    per_case = {}
                    for key in keys:
                        case = case_for(key)
                        status("evaluating", step=step, case=key, active_case_nodes=len(case["xyz"]))
                        shared._sync(torch, device); stage = time.monotonic()
                        encoded = _predict_case(torch, np, model, case, device); shared._sync(torch, device)
                        inference_seconds = time.monotonic()-stage
                        costs["inference_seconds"] += inference_seconds; costs["inference_case_exposures"] += 1
                        costs["inference_node_exposures"] += len(case["xyz"])
                        prediction = scaler.decode(encoded)
                        physical = case["physical"]
                        per_case[key] = {**shared._physical_metrics(np, physical, prediction),
                                         "inference_seconds": inference_seconds,
                                         "BC_regions": _bc_metrics(np, case["features"][:, :7].cpu().numpy(), physical, prediction)}
                        del case, encoded, prediction
                    groups[group] = shared._aggregate(np, per_case)
            costs["evaluation_seconds"] += time.monotonic()-evaluation_began
            if step:
                passes[-1]["physical_readout"] = {
                    group: {name: value for name, value in row.items() if name.startswith("case_mean_") or name.startswith("worst_case_")}
                    for group, row in groups.items()}
            metrics = {"step": step, "elapsed_seconds": elapsed(), "training_loss": loss, **groups,
                       "selection_score": groups["validation"]["selection_score"], "costs": dict(costs),
                       "pass_costs": list(passes), "memory": memory(), "first_update_gradient_check": gradient_check}
            improved = metrics["selection_score"] < best_score
            if improved:
                best_score, best_step, best_metrics = metrics["selection_score"], step, metrics
            history.append(metrics); state["evaluated_steps"] = [row["step"] for row in history]
            if improved:
                save(output / "best.pt"); shared.write_json(output / "metrics_best.json", metrics)
            save(output / "last.pt"); save(output / f"step{step}.pt")
            shared.write_json(output / "metrics_latest.json", metrics)
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, allow_nan=False)+"\n")
            status("training", step=step, best_step=best_step, selection_score=metrics["selection_score"], memory=memory())
            print(f"gino {step}/480 VAL U={groups['validation']['case_mean_u_relative_l2']:.6g} stress={groups['validation']['case_mean_stress_relative_l2']:.6g}", flush=True)
            return metrics
        last = history[-1] if history else evaluate(0, None)
        for step in range(completed_step+1, 481):
            key = manifest["case_order"][step-1]; state["case"] = key
            training_began = time.monotonic()
            frontend_before = costs["radius_build_seconds"]+costs["radius_save_seconds"]
            case = case_for(key); model.train(); optimizer.zero_grad(set_to_none=True)
            status("training", step=completed_step, active_case_nodes=len(case["xyz"]))
            shared._sync(torch, device)
            loss, execution = _train_case(torch, model, case, device)
            if gradient_check is None:
                gradient_check = {"step": step, "case": key, "execution": execution, **_gradient_check(torch, model)}
                shared.write_json(output / "first_update_gradient_check.json", gradient_check)
            optimizer_began = time.monotonic()
            optimizer.step(); shared._sync(torch, device)
            execution["optimizer_seconds"] = time.monotonic()-optimizer_began
            costs["gpu_step_seconds"] += sum(execution[name] for name in ("encode_seconds", "query_forward_backward_seconds", "encode_backward_seconds", "optimizer_seconds"))
            costs["train_seconds"] += time.monotonic()-training_began-(costs["radius_build_seconds"]+costs["radius_save_seconds"]-frontend_before)
            costs["train_case_exposures"] += 1; costs["train_node_exposures"] += len(case["xyz"])
            exposures[key] += 1; completed_step = step; del case
            if step % 120 == 0:
                previous_train = passes[-1]["cumulative_train_seconds"] if passes else 0.
                passes.append({"pass": step//120, "step": step, "case_updates": 120,
                               "incremental_train_seconds": costs["train_seconds"]-previous_train,
                               "cumulative_train_seconds": costs["train_seconds"],
                               "mean_train_seconds_per_case": (costs["train_seconds"]-previous_train)/120,
                               "cumulative_radius_build_seconds": costs["radius_build_seconds"],
                               "cumulative_radius_read_seconds": costs["radius_read_seconds"],
                               "cumulative_radius_save_seconds": costs["radius_save_seconds"]})
                last = evaluate(step, loss)
            elif step % plan["recipe"]["checkpoint_every_updates"] == 0 or step == 1:
                save(output / "last.pt")
            status("training", step=step, training_loss=loss, last_case_execution=execution, memory=memory())
        if costs["train_case_exposures"] != 480 or any(value != 4 for value in exposures.values()):
            raise AssertionError("The common complete four-pass exposure contract was not completed")
        summary = {"status": "completed", "method": "gino", "endpoint_step": 480, "config": config,
                   "elapsed_seconds": elapsed(), "initial_metrics": history[0], "final_metrics": last,
                   "endpoint_metrics": history, "best_step": best_step, "best_metrics": best_metrics,
                   "costs": dict(costs), "pass_costs": passes, "parameter_counts": parameter_counts,
                   "memory": memory(), "case_exposure_counts": exposures, "radius_cache": cache.receipt(),
                   "first_update_gradient_check": gradient_check, "case_order_sha256": manifest["case_order_sha256"],
                   "existing_screen_comparisons": plan["existing_screen_comparisons"],
                   "scope": "TRAIN40/VAL5 x ver/hor/tor; full original nodes; no final/dia/new sealed test; no scaler fit",
                   "completed_at_utc": shared.utc_now()}
        shared.write_json(output / "summary.json", summary); status("completed", step=480)
        return summary
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        status("failed", step=completed_step, error=message)
        summary = {"status": "failed", "method": "gino", "config": config, "error": message,
                   "completed_step": completed_step, "elapsed_seconds": prior_elapsed+time.monotonic()-began,
                   "failed_case": state.get("case"), "full_node_count": state.get("active_case_nodes"),
                   "costs": dict(costs), "radius_cache": cache.receipt() if cache else None,
                   "last_checkpoint": str(output / "last.pt") if (output / "last.pt").is_file() else None,
                   "failure_policy": "Original node count, radii, grid and query_chunk2048 unchanged; no OOM retry with altered science",
                   "new_test_or_dia_read": False}
        shared.write_json(output / "summary.json", summary)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, required=True)
    result.add_argument("--output", type=Path)
    result.add_argument("--resume", nargs="?", const="last")
    result.add_argument("--source-check", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
