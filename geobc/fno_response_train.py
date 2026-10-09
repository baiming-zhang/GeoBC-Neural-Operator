"""Two fresh, matched FNO response-basis arms on TRAIN40/VAL5 known3.

Source checks are standard-library JSON/stat/AST only. Actual execution is
released by the root queue after freezing the recipe. No scaler fitting,
warm start, final-test/dia, or sealed next-cycle input/field access occurs.
"""
from __future__ import annotations

import argparse
import ast
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
MODES = ("linear_basis", "load_dependent_basis")
ENDPOINTS = (0, 120, 240, 480)
SOURCES = ("geobc/fno_response_train.py", "geobc/fno_response_field.py", "geobc/fno.py",
           "geobc/conditioned_field.py", "geobc/solver_selection_data.py",
           "geobc/solver_selection_train.py", "geobc/benchmark.py")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stat_record(path):
    item = shared.host_path(path).stat()
    return {"bytes": item.st_size, "mtime_ns": item.st_mtime_ns}


def _check_recipe(plan):
    fixed = {"modes": list(MODES), "seed": 2026, "cloud_seed": 2026, "train_loads": ["ver", "hor", "tor"],
             "steps": 480, "passes": 4, "eval_steps": list(ENDPOINTS), "optimizer": "Adam",
             "learning_rate": .001, "weight_decay": 0., "fused": True, "cloud_points": 1024,
             "geometry_dim": 32, "grid_resolution": 64, "FNO_width": 16, "FNO_modes": 12,
             "FNO_layers": 4, "u_basis_head": [16,128,9], "stress_head": [22,128,1],
             "stress_transform": "asinh", "channels": [0,1,2,3],
             "training_readout_keys": ["101_479/ver", "107_320/hor", "10_58/tor"]}
    if plan.get("kind") != "fno_response_screen" or any(plan["recipe"].get(k) != v for k,v in fixed.items()):
        raise ValueError("Response plan differs from the matched two-arm screening recipe")
    expected = {"source_xyz_atol": 0., "source_static_features_atol": 0., "grid_static_atol": 2e-6,
                "grid_static_rtol": 1e-6, "grid_BC_role_atol": 2e-6, "alpha_atol": 2e-7,
                "alpha_rtol": 1e-6, "query_affine_atol": 3e-6, "query_affine_rtol": 2e-6}
    if plan["input_guard_tolerances"] != expected:
        raise ValueError("Predeclared source/grid/alpha tolerances changed")
    if any(plan["access"].get(k) is not False for k in ("test_arrays", "dia_arrays", "sealed_new_test_values", "warm_start", "new_scaler_fit")):
        raise ValueError("This fresh training screen excludes final/dia/sealed fields and warm starts")


def _basis_metadata(path, split):
    guard = shared.read_json(path)
    if (guard.get("compatible") is not True or guard.get("status") != "compatible"
            or guard.get("CADs_checked") != 45 or guard.get("compatible_CADs") != 45 or guard.get("badcases")):
        raise ValueError("Existing45 TRAIN/VAL load-basis metadata is not compatible")
    canonical = guard["canonical_basis"]; mapping = guard["prepared_feature_mapping"]
    if canonical["order"] != ["ver","hor","tor"] or canonical["exact_rank"] != 3:
        raise ValueError("Expected the verified shared rank3 ver/hor/tor basis")
    if mapping["slots"] != list(range(7,13)) or mapping["physical_scale"] != [42300]*3+[565000]*3:
        raise ValueError("Original normalized force/moment input units changed")
    matrix = mapping["feature_to_alpha_matrix_3x6"]
    if len(matrix) != 3 or any(len(row) != 6 or not all(math.isfinite(x) for x in row) for row in matrix):
        raise ValueError("Invalid normalized feature-to-alpha matrix")
    records = {row["cad_id"]: row for row in guard["cases"]}
    if set(records) != set(split["groups"]["train"]+split["groups"]["validation"]):
        raise ValueError("Basis guard contains a different CAD allowlist")
    for group in ("train", "validation"):
        for cad in split["groups"][group]:
            row = records[cad]
            if row["group"] != group or not row["compatible"] or row["basis_rank"] != 3 or not row["basis_matches_shared_reference"] or not row["prepared_slot_mapping_compatible"]:
                raise ValueError("Incompatible CAD load basis: " + cad)
            for load in ("ver","hor","tor"):
                item = row["alpha_by_load"][load]
                if len(item["alpha"]) != 3 or not all(math.isfinite(x) for x in item["alpha"]) or item["exact_residual_zero"] is not True:
                    raise ValueError("Invalid verified known-load coefficient")
    return guard


def _metadata(plan_path, mode):
    if mode not in MODES:
        raise ValueError("Unknown response mode")
    plan = shared.read_json(plan_path); _check_recipe(plan)
    manifest_path = shared.host_path(plan["data_cache"])/"manifest.json"
    manifest = shared.read_json(manifest_path); shared._check_manifest(manifest)
    split = shared.read_json(plan["existing_split"])
    for group, keys in (("train", manifest["train_keys"]), ("validation", manifest["validation_keys"])):
        expected = [f"{cad}/{load}" for cad in split["groups"][group] for load in ("ver","hor","tor")]
        if keys != expected:
            raise ValueError("Selection case order/split changed")
    data_contract = manifest["contract"]
    if data_contract["split_sha256"] != shared.source_sha(plan["existing_split"]) or data_contract["cloud_seed"] != 2026 or data_contract["grid_resolution"] != 64 or data_contract["cloud_points"] != 1024:
        raise ValueError("Existing shared cloud/grid/scaler metadata changed")
    for row in data_contract["sources"]:
        if stat_record(row["path"]) != {k: row[k] for k in ("bytes","mtime_ns")}:
            raise ValueError("An existing prepared TRAIN/VAL source stat changed")
    guard = _basis_metadata(plan["load_basis"], split)
    science = {k: plan[k] for k in ("recipe","existing_split","prepared_data","data_cache","load_basis",
                                   "input_guard_cache","input_guard_tolerances","access")}
    contract = {"schema": "geobc_fno_response_screen_v1", "plan": str(shared.host_path(plan_path)),
                "recipe_sha256": digest(science), "recipe": plan["recipe"], "response_mode": mode,
                "cache_manifest_sha256": shared.source_sha(manifest_path), "cache_contract": data_contract,
                "scaler": manifest["scaler"], "case_order_sha256": manifest["case_order_sha256"],
                "train_keys": manifest["train_keys"], "validation_keys": manifest["validation_keys"],
                "training_readout_keys": manifest["training_readout_keys"],
                "load_basis_sha256": shared.source_sha(plan["load_basis"]),
                "source_sha256": {name: shared.source_sha(name) for name in SOURCES}}
    return contract, plan, manifest, guard


def request_contract(plan_path, mode):
    """(contract,plan), standard-library metadata only, including reuse checks."""
    contract, plan, _, _ = _metadata(plan_path, mode)
    return contract, plan


def source_check(plan_path):
    identities = {}
    for mode in MODES:
        contract, _ = request_contract(plan_path, mode); identities[mode] = digest(contract)
    for name in ("geobc/fno_response_train.py", "geobc/fno_response_field.py"):
        ast.parse(shared.host_path(name).read_text(encoding="utf-8"), filename=name)
    return {"status": "source_ready", "contract_sha256_by_mode": identities,
            "recipe_sha256": contract["recipe_sha256"], "case_order_sha256": contract["case_order_sha256"],
            "source_sha256": contract["source_sha256"], "cache_manifest_sha256": contract["cache_manifest_sha256"],
            "load_basis_sha256": contract["load_basis_sha256"], "scaler_sha256": digest(contract["scaler"]),
            "TRAIN_cases": 120, "VAL_cases": 15, "CADs": 45,
            "training_rules_frozen": shared.read_json(plan_path).get("training_rules_frozen",False),
            "scientific_arrays_read": 0, "models_constructed": 0, "CUDA_initialized": False, "training_updates": 0}


class LazyInputGuard:
    """One target-free known3 check per CAD, shared between serial fresh arms."""
    def __init__(self, plan, contract, manifest, guard, np, costs):
        self.plan, self.contract, self.manifest, self.np, self.costs = plan, contract, manifest, np, costs
        self.records = {row["cad_id"]: row for row in guard["cases"]}; self.guard = guard
        self.results = {}; self.root = shared.host_path(plan["input_guard_cache"])

    def ensure(self, cad):
        if cad in self.results:
            return
        stage = time.monotonic(); np = self.np
        inputs = {}
        for load in ("ver","hor","tor"):
            record = self.manifest["records"][cad+"/"+load]
            inputs[load] = {name: {"path": record[name+"_path"], **stat_record(record[name+"_path"])}
                            for name in ("xyz","features","grid","cloud","mask","query_indices","query_weights")}
        identity = {"cad_id": cad, "cache_manifest_sha256": self.contract["cache_manifest_sha256"],
                    "load_basis_sha256": self.contract["load_basis_sha256"], "inputs": inputs,
                    "tolerances": self.plan["input_guard_tolerances"]}
        path = self.root/(cad+".json")
        if path.is_file():
            saved = shared.read_json(path)
            if saved.get("status") != "completed" or saved["identity"] != identity or saved["passed"] is not True:
                raise ValueError("Existing lazy input-guard identity/check differs; no silent rescanning")
            self.results[cad] = saved
            self.costs["input_guard_read_seconds"] += time.monotonic()-stage
            self.costs["input_guard_reuse_count"] += 1
            return
        reference = None; comparisons = {}; tol = identity["tolerances"]
        nonload = [i for i in range(25) if i not in range(7,13)]
        static_grid = [i for i in range(29) if i not in range(11,17)]
        matrix = np.asarray(self.guard["prepared_feature_mapping"]["feature_to_alpha_matrix_3x6"],np.float64)
        normalized_basis = np.asarray(self.guard["prepared_feature_mapping"]["normalized_basis_6x3"],np.float64)
        for load in ("ver","hor","tor"):
            row = inputs[load]
            arrays = {name: np.load(shared.host_path(row[name]["path"]),mmap_mode="r",allow_pickle=False)
                      for name in ("xyz","features","grid")}
            n = self.manifest["records"][cad+"/"+load]["num_nodes"]
            if arrays["xyz"].shape != (n,3) or arrays["features"].shape != (n,25) or arrays["grid"].shape != (29,64,64,64):
                raise ValueError("Lazy known3 input shape differs")
            if any(value.dtype != np.float32 for value in arrays.values()):
                raise ValueError("Lazy known3 input dtype differs")
            expected = np.asarray(self.records[cad]["alpha_by_load"][load]["alpha"],np.float64)
            alpha_error = 0.; span_error = 0.; load_uniform_error = 0.
            first_load = arrays["features"][0,7:13].astype(np.float64)
            for begin in range(0,n,32768):
                feature = arrays["features"][begin:begin+32768]
                current_load = feature[:,7:13].astype(np.float64)
                inferred = current_load@matrix.T
                alpha_error = max(alpha_error,float(np.abs(inferred-expected).max()))
                span_error = max(span_error,float(np.abs(current_load-inferred@normalized_basis.T).max()))
                load_uniform_error = max(load_uniform_error,float(np.abs(current_load-first_load).max()))
                if not np.isfinite(feature).all() or not np.allclose(inferred,expected[None,:],atol=tol["alpha_atol"],rtol=tol["alpha_rtol"]):
                    raise ValueError("Actual normalized load slots disagree with metadata alpha")
            if span_error > 1e-7 or load_uniform_error != 0.:
                raise ValueError("All-node six-slot load must be constant in the verified normalized span")
            metrics = {"alpha_max_abs_error":alpha_error,"source_xyz_max_abs_difference":0.,
                       "normalized_load_span_max_residual":span_error,"source_load6_uniform_max_abs_difference":load_uniform_error,
                       "source_static_features_max_abs_difference":0.,"grid_static_max_abs_difference":0.,
                       "grid_BC_roles_max_abs_difference":0.}
            if reference is not None:
                if arrays["xyz"].shape != reference["xyz"].shape:
                    raise ValueError("Known3 canonical node counts differ")
                for begin in range(0,n,32768):
                    for name, columns, label in (("xyz",slice(None),"source_xyz"),("features",nonload,"source_static_features")):
                        left=arrays[name][begin:begin+32768,columns]; right=reference[name][begin:begin+32768,columns]
                        metrics[label+"_max_abs_difference"]=max(metrics[label+"_max_abs_difference"],float(np.abs(left.astype(np.float64)-right).max()))
                        if not np.isfinite(left).all() or not np.array_equal(left,right):
                            raise ValueError("Source known3 xyz/nonload features differ")
                for channel in static_grid:
                    left=arrays["grid"][channel]; right=reference["grid"][channel]
                    delta=float(np.abs(left.astype(np.float64)-right).max())
                    metrics["grid_static_max_abs_difference"]=max(metrics["grid_static_max_abs_difference"],delta)
                    if not np.isfinite(left).all() or not np.allclose(left,right,atol=tol["grid_static_atol"],rtol=tol["grid_static_rtol"]):
                        raise ValueError("Nonload grid differences exceed declared FP32 splat tolerance")
                    if 4 <= channel < 11:
                        metrics["grid_BC_roles_max_abs_difference"]=max(metrics["grid_BC_roles_max_abs_difference"],delta)
                        if delta > tol["grid_BC_role_atol"]:
                            raise ValueError("BC-role grid differs beyond absolute source-role tolerance")
                for name in ("cloud","mask","query_indices","query_weights"):
                    if row[name] != inputs["ver"][name]:
                        raise ValueError("Known3 geometry cloud/query cache identity differs")
            else:
                reference=arrays
                if not np.isfinite(arrays["xyz"]).all() or not np.isfinite(arrays["grid"]).all():
                    raise ValueError("Nonfinite canonical input/grid")
            comparisons[load]=metrics
        seconds=time.monotonic()-stage
        saved={"status":"completed","identity":identity,"passed":True,"comparisons":comparisons,
               "input_only":True,"labels_read":0,"scientific_members":["xyz","features","grid"],
               "construction_seconds":seconds,"source_vs_grid":"Source xyz/nonload features exact; cached GPU splat grids allow only recorded FP32 tolerance"}
        shared.write_json(path,saved); self.results[cad]=saved
        self.costs["input_guard_build_seconds"] += time.monotonic()-stage
        self.costs["input_guard_build_count"] += 1

    def receipt(self):
        return {"CADs_checked_or_reused":sorted(self.results),"unique_CADs":len(self.results),
                "per_CAD":self.results,"shared_between_arms":True,"normal_case_load_not_replaced":True}


def _query_components(grid, case, count=8):
    flat=grid[0].flatten(1).T
    indices=case["query_indices"][:count]; weights=case["query_weights"][:count]
    return (flat[indices]*weights[...,None]).sum(1)


def _unit_check(torch,np,model,case,components,prediction,manifest,guard,tol):
    cad,load=case["key"].split("/"); count=min(8,len(prediction))
    with torch.no_grad():
        basis=_query_components(components["basis_grid"].detach(),case,count)
        stress=_query_components(components["stress_grid"].detach(),case,count)
        after=model.compose_encoded_nodes(basis,stress,case["features"][0,7:13])
        if not torch.allclose(after,prediction[:count],atol=tol["query_affine_atol"],rtol=tol["query_affine_rtol"]):
            raise ValueError("Actual-case basis-first and affine-grid node queries disagree")
        weights_error=float((case["query_weights"].sum(-1)-1).abs().max())
        if weights_error > 3e-6:
            raise ValueError("All-node trilinear rows fail partition of unity for the affine offset")
        alpha=components["alpha"].detach().cpu().numpy().astype(np.float64)[0]
        record=next(row for row in guard["cases"] if row["cad_id"]==cad)
        expected=np.asarray(record["alpha_by_load"][load]["alpha"],np.float64)
        if not np.allclose(alpha,expected,atol=tol["alpha_atol"],rtol=tol["alpha_rtol"]):
            raise ValueError("First actual alpha differs from physical metadata")
        load6=case["features"][0,7:13].detach().cpu().numpy().astype(np.float64)
        physical=load6*np.asarray(guard["prepared_feature_mapping"]["physical_scale"],np.float64)
        reconstructed=np.asarray(guard["canonical_basis"]["physical_matrix_6x3"],np.float64)@alpha
        force_residual=float(np.abs(physical[:3]-reconstructed[:3]).max())
        moment_residual=float(np.abs(physical[3:]-reconstructed[3:]).max())
        # Tolerances are relative to separate force/moment units, not a mixed norm.
        if force_residual > .01+1e-6*float(np.abs(physical[:3]).max()) or moment_residual > .1+1e-6*float(np.abs(physical[3:]).max()):
            raise ValueError("Force or moment reconstruction fails its distinct physical units")
        basis_np=basis.cpu().numpy().astype(np.float64).reshape(count,3,3)
        zero_centered=np.einsum("k,nkj->nj",alpha,basis_np)
        mean=np.asarray(manifest["scaler"]["mean"][:3],np.float64); std=np.asarray(manifest["scaler"]["std"][:3],np.float64)
        node_encoded=after.cpu().numpy().astype(np.float64)[:,:3]
        expected_encoded=zero_centered-mean/std
        if not np.allclose(node_encoded,expected_encoded,atol=tol["query_affine_atol"],rtol=tol["query_affine_rtol"]):
            raise ValueError("Displacement mean/std must be applied once after basis combination")
        decoded=mean+std*node_encoded; expected_mm=std*zero_centered
        return {"passed":True,"case":case["key"],"full_nodes":len(prediction),"witness_original_node_ids":list(range(count)),
                "basis_grid_shape":list(components["basis_grid"].shape),"stress_grid_shape":list(components["stress_grid"].shape),
                "encoded_node_shape":list(prediction.shape),"alpha":alpha.tolist(),"expected_alpha":expected.tolist(),
                "force_max_reconstruction_residual_N":force_residual,"moment_max_reconstruction_residual_Nmm":moment_residual,
                "force_reconstruction_tolerance_N":.01+1e-6*float(np.abs(physical[:3]).max()),
                "moment_reconstruction_tolerance_Nmm":.1+1e-6*float(np.abs(physical[3:]).max()),
                "query_then_compose_max_abs_error":float((after-prediction[:count]).abs().max()),
                "all_node_partition_of_unity_max_abs_error":weights_error,
                "physical_affine_max_error_mm":float(np.abs(decoded-expected_mm).max()),
                "mean_policy":"B_kj=std_j*V_kj; z=sum(alpha*V)-mean/std once; physical_u=std*sum(alpha*V)",
                "algebraic_zero_load_physical_mm":(mean+std*(-mean/std)).tolist(),
                "extra_body_forwards":0,"scope":"Witnesses reuse components from the first actual full-node training loss; not a toy or superposition GT check"}


def _gradient_check(torch,model,alpha):
    record=shared._gradient_check(torch,model)
    norms={name:math.sqrt(sum(float(p.grad.detach().square().sum()) for p in getattr(model.solver,name).parameters() if p.grad is not None))
           for name in ("u_basis_head","stress_head")}
    if any(not math.isfinite(value) or value <= 0 for value in norms.values()):
        raise FloatingPointError("Active displacement basis/stress head gradient is zero or nonfinite")
    rows=model.solver.u_basis_head[2].weight.grad.detach().reshape(9,-1).norm(dim=1).tolist()
    return {**record,"head_L2_gradient_norms":norms,"u_basis_final_weight_row_norms":rows,
            "alpha":alpha.detach().cpu().tolist(),"zero_row_policy":"One-hot inactive basis output rows may have zero gradients; no requirement that all nine rows be positive"}


def run(args):
    if args.source_check:
        result=source_check(args.plan); print(json.dumps(result,indent=2),flush=True); return result
    if args.mode not in MODES or args.output is None:
        raise ValueError("Actual run requires --mode and --output")
    contract,plan,manifest,guard=_metadata(args.plan,args.mode)
    if plan.get("training_rules_frozen") is not True:
        raise ValueError("Response training recipe remains unfrozen; only root may release actual execution")
    output=shared.host_path(args.output)
    if (output/"summary.json").is_file():
        previous=shared.read_json(output/"summary.json")
        if previous.get("status")=="completed":
            if previous["config"]["contract"]!=contract or shared.read_json(output/"config.json")!=previous["config"]:
                raise ValueError("Completed response run identity differs")
            return previous
    resume=(output/"last.pt" if args.resume=="last" else shared.host_path(args.resume)) if args.resume else None
    if resume and not resume.is_file(): raise FileNotFoundError(resume)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError("Incomplete output requires explicit --resume; no warm start from old solvers")
    output.mkdir(parents=True,exist_ok=True); began=time.monotonic(); prior_elapsed=0.; completed_step=0
    costs={name:0. for name in ("train_seconds","inference_seconds","evaluation_seconds","input_guard_build_seconds","input_guard_read_seconds")}
    costs.update({name:0 for name in ("train_case_exposures","train_node_exposures","inference_case_exposures","inference_node_exposures","input_guard_build_count","input_guard_reuse_count")})
    costs["train_seconds_definition"]="Begin before normal load_case; current case/target/grid/query-cache read and H2D + complete FNO/all-node loss/backward/Adam/sync; shared lazy input guard separately timed"
    state={"kind":"fno_response_screen","response_mode":args.mode,"step":0,"total_steps":480,"evaluated_steps":[],"started_at_utc":shared.utc_now()}
    config={"contract":contract}; input_guard=None
    def status(phase,**details):
        state.update(details,phase=phase,updated_at_utc=shared.utc_now(),elapsed_seconds=prior_elapsed+time.monotonic()-began,costs=dict(costs))
        eta=None
        if costs["train_case_exposures"]:
            eta=costs["train_seconds"]/costs["train_case_exposures"]*(480-completed_step)
            if state["evaluated_steps"]:
                eta+=costs["evaluation_seconds"]/len(state["evaluated_steps"])*(len(ENDPOINTS)-len(state["evaluated_steps"]))
        state["eta_seconds"]=0. if phase=="completed" else None if phase=="failed" else eta
        state["estimated_completion_utc"]=(datetime.now(timezone.utc)+timedelta(seconds=state["eta_seconds"])).isoformat() if state["eta_seconds"] is not None else None
        state["eta_limit"]="Observed train/evaluation rates; unseen-CAD lazy guard construction not extrapolated"
        shared.write_json(output/"status.json",state)
    status("loading_fresh_response_model")
    try:
        import numpy as np
        import torch
        from .benchmark import FieldScaler
        from .solver_selection_data import load_case,query_grid,read_cache_metadata
        from .fno_response_field import FNOResponseField
        if read_cache_metadata(plan)!=manifest: raise ValueError("Shared cache metadata changed")
        device=torch.device("cuda"); random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
        torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
        model=FNOResponseField(args.mode,manifest["scaler"],guard,seed=2026,device=device)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.,fused=True)
        scaler=FieldScaler(**manifest["scaler"])
        config={"contract":contract,"response_mode":args.mode,"method":"fno_response_basis","scaler":manifest["scaler"],
                "model":model.source_config,"parameter_counts":model.parameter_counts,"case_order":manifest["case_order"],
                "case_order_sha256":manifest["case_order_sha256"],"train_keys":manifest["train_keys"],"validation_keys":manifest["validation_keys"],
                "training_readout_keys":manifest["training_readout_keys"],"precision":"float32/TF32, no autocast; old FNO real-stored spectral weights/fused Adam unchanged",
                "fresh_initialization":"Same constructor/seed/order/parameter layout for both arms; no warm start; natural predictions may differ"}
        history=[]; best_score=math.inf; best_step=None; best_metrics=None; first_check=None
        exposures={key:0 for key in manifest["train_keys"]}; passes=[]
        _,shuffle_rng=shared.case_order(manifest["train_keys"],2026)
        prior_peak={"peak_allocated_bytes":0,"peak_reserved_bytes":0}
        if resume:
            saved=torch.load(resume,map_location=device,weights_only=False)
            if saved["config"]!=config or saved["case_order_sha256"]!=manifest["case_order_sha256"]:
                raise ValueError("Resume mode/data/model/scaler/order differs; no cross-arm or old-solver warm start")
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); shared._restore_rng(torch,np,saved["rng"])
            completed_step=saved["step"]
            if saved["next_case_position"]!=completed_step: raise ValueError("Resume schedule position differs")
            costs=saved["costs"]; history=saved["history"]; exposures=saved["case_exposure_counts"]; passes=saved["pass_costs"]
            best_score=saved["best_score"]; best_step=saved["best_step"]; best_metrics=saved["best_metrics"]
            first_check=saved["first_update_gradient_check"]; prior_elapsed=saved["elapsed_seconds"]; prior_peak=saved["memory"]; shuffle_rng=saved["case_shuffle_rng_state"]
            state["evaluated_steps"]=[row["step"] for row in history]
            (output/"metrics.jsonl").write_text("".join(json.dumps(row,allow_nan=False)+"\n" for row in history),encoding="utf-8")
            if best_step is not None:
                selected=resume if best_step==completed_step else resume.parent/f"step{best_step}.pt"
                if not selected.is_file(): raise FileNotFoundError(selected)
                if selected.resolve()!=(output/"best.pt").resolve(): shutil.copy2(selected,output/"best.pt")
                shared.write_json(output/"metrics_best.json",best_metrics)
        shared.write_json(output/"config.json",config); torch.cuda.reset_peak_memory_stats(device)
        input_guard=LazyInputGuard(plan,contract,manifest,guard,np,costs)
        if resume: input_guard.results=saved["input_consistency_guard"]["per_CAD"]
        def memory():
            current=shared._memory(torch,device)
            return {k:max(v,prior_peak[k]) for k,v in current.items()}
        def elapsed(): return prior_elapsed+time.monotonic()-began
        def payload():
            return {"model":model.state_dict(),"optimizer":optimizer.state_dict(),"config":config,"step":completed_step,
                    "next_case_position":completed_step,"case_order":manifest["case_order"],"case_order_sha256":manifest["case_order_sha256"],
                    "case_shuffle_rng_state":shuffle_rng,"rng":shared._rng_state(torch,np),"history":history,"costs":dict(costs),
                    "case_exposure_counts":dict(exposures),"pass_costs":passes,"best_score":best_score,"best_step":best_step,
                    "best_metrics":best_metrics,"first_update_gradient_check":first_check,"elapsed_seconds":elapsed(),
                    "memory":memory(),"input_consistency_guard":input_guard.receipt()}
        def evaluate(step,loss):
            nonlocal best_score,best_step,best_metrics
            model.eval(); stage=time.monotonic(); groups={}; status("evaluating",step=step)
            with torch.no_grad():
                for group,keys in (("train_readout",manifest["training_readout_keys"]),("validation",manifest["validation_keys"])):
                    per_case={}
                    for key in keys:
                        input_guard.ensure(key.split("/")[0])
                        case=load_case(manifest["records"][key],device,for_fno=True)
                        shared._sync(torch,device); infer_began=time.monotonic()
                        context=model.encode_geometry(case["cloud"],case["mask"])
                        encoded=query_grid(model.forward_grid(case["grid"],context,case["features"][0,7:13]),case).float().cpu().numpy()
                        shared._sync(torch,device); infer_seconds=time.monotonic()-infer_began
                        costs["inference_seconds"]+=infer_seconds; costs["inference_case_exposures"]+=1; costs["inference_node_exposures"]+=len(case["xyz"])
                        prediction=scaler.decode(encoded)
                        per_case[key]={**shared._physical_metrics(np,case["physical"],prediction),"inference_seconds":infer_seconds}
                        del case,encoded,prediction,context
                    groups[group]=shared._aggregate(np,per_case)
            costs["evaluation_seconds"]+=time.monotonic()-stage
            metrics={"step":step,"elapsed_seconds":elapsed(),"training_loss":loss,**groups,"selection_score":groups["validation"]["selection_score"],
                     "costs":dict(costs),"memory":memory(),"case_exposure_counts":dict(exposures),"first_actual_check":first_check}
            improved=metrics["selection_score"]<best_score
            if improved: best_score,best_step,best_metrics=metrics["selection_score"],step,metrics
            history.append(metrics); state["evaluated_steps"]=[row["step"] for row in history]
            if improved:
                shared._checkpoint(torch,output/"best.pt",payload()); shared.write_json(output/"metrics_best.json",metrics)
            shared._checkpoint(torch,output/"last.pt",payload()); shared._checkpoint(torch,output/f"step{step}.pt",payload())
            shared.write_json(output/"metrics_latest.json",metrics)
            with (output/"metrics.jsonl").open("a",encoding="utf-8") as handle: handle.write(json.dumps(metrics,allow_nan=False)+"\n")
            status("training",step=step,best_step=best_step,selection_score=metrics["selection_score"],memory=memory())
            print(f"{args.mode} {step}/480 VAL U={groups['validation']['case_mean_u_relative_l2']:.6g} stress={groups['validation']['case_mean_stress_relative_l2']:.6g}",flush=True)
            return metrics
        last=history[-1] if history else evaluate(0,None)
        for step in range(completed_step+1,481):
            key=manifest["case_order"][step-1]; input_guard.ensure(key.split("/")[0])
            training_began=time.monotonic(); case=load_case(manifest["records"][key],device,for_fno=True)
            model.train(); optimizer.zero_grad(set_to_none=True)
            context=model.encode_geometry(case["cloud"],case["mask"])
            components=model.forward_components(case["grid"],context,case["features"][0,7:13])
            prediction=query_grid(model.compose_encoded_grid(components),case)
            if prediction.shape!=case["target"].shape or prediction.shape!=(len(case["xyz"]),4): raise ValueError("Complete signed4 node/target shapes differ")
            unit=None
            if first_check is None: unit=_unit_check(torch,np,model,case,components,prediction,manifest,guard,plan["input_guard_tolerances"])
            loss=(prediction.float()-case["target"]).square().mean()
            if not bool(torch.isfinite(loss)): raise FloatingPointError("Nonfinite complete-case four-channel loss")
            loss.backward()
            if first_check is None:
                first_check={"passed":True,"step":step,"case":key,"gradient":_gradient_check(torch,model,components["alpha"]),"units_alpha_query":unit}
                shared.write_json(output/"first_actual_check.json",first_check)
            optimizer.step(); shared._sync(torch,device); costs["train_seconds"]+=time.monotonic()-training_began
            costs["train_case_exposures"]+=1; costs["train_node_exposures"]+=len(case["xyz"]); exposures[key]+=1; completed_step=step
            loss_value=float(loss.detach()); del case,context,components,prediction,loss
            if step%120==0:
                previous=passes[-1]["cumulative_train_seconds"] if passes else 0.
                passes.append({"pass":step//120,"step":step,"cumulative_train_seconds":costs["train_seconds"],
                               "incremental_train_seconds":costs["train_seconds"]-previous,"mean_train_seconds_per_case":(costs["train_seconds"]-previous)/120})
            if step in ENDPOINTS: last=evaluate(step,loss_value)
            elif step%30==0 or step==1: shared._checkpoint(torch,output/"last.pt",payload())
            if step%10==0: status("training",step=step,case=key,training_loss=loss_value,memory=memory())
        if costs["train_case_exposures"]!=480 or any(value!=4 for value in exposures.values()): raise AssertionError("Complete four-pass exposure contract failed")
        summary={"status":"completed","kind":"fno_response_screen","method":"fno_response_basis","response_mode":args.mode,
                 "endpoint_step":480,"config":config,"elapsed_seconds":elapsed(),"initial_metrics":history[0],"final_metrics":last,
                 "endpoint_metrics":history,"best_step":best_step,"best_metrics":best_metrics,"selection_score":last["selection_score"],
                 "costs":dict(costs),"pass_costs":passes,"parameter_counts":model.parameter_counts,"memory":memory(),
                 "case_exposure_counts":exposures,"case_order_sha256":manifest["case_order_sha256"],"first_actual_check":first_check,
                 "first_update_gradient_check":first_check,"input_consistency_guard":input_guard.receipt(),
                 "scope":"Fresh TRAIN40/VAL5 known3 only; same stored nine-basis/stress-head architecture, load-mask treatment only; no final/dia/sealed field reads",
                 "completed_at_utc":shared.utc_now()}
        shared.write_json(output/"summary.json",summary); status("completed",step=480); return summary
    except Exception as error:
        message=f"{type(error).__name__}: {error}"; status("failed",step=completed_step,error=message)
        shared.write_json(output/"summary.json",{"status":"failed","kind":"fno_response_screen","response_mode":args.mode,"config":config,
                          "completed_step":completed_step,"error":message,"elapsed_seconds":prior_elapsed+time.monotonic()-began,"costs":dict(costs),
                          "input_consistency_guard":input_guard.receipt() if input_guard else None,
                          "last_checkpoint":str(output/"last.pt") if (output/"last.pt").is_file() else None,
                          "no_fallback":"No warm start, loss reweighting, node/grid/radius/chunk reduction or new scope after failure"})
        raise


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan",type=Path,required=True)
    result.add_argument("--mode",choices=MODES)
    result.add_argument("--output",type=Path)
    result.add_argument("--resume",nargs="?",const="last")
    result.add_argument("--source-check",action="store_true")
    return result


if __name__=="__main__":
    run(parser().parse_args())
