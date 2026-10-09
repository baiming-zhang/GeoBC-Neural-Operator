"""Frozen FNO initial guess plus unchanged, target-free physical PCG.

This isolated five-CAD macro solver comparison rebuilds the neural frontend
from permitted inputs. It never calls a target-loading loader and never reads
GT. The completed zero-start displacement is a numerical reference only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_elastic_hybrid_plan.json"
SOURCE = "geobc/elastic_fno_warmstart.py"
KIND = "elastic_frozen_fno_warmstart_screen"
CASES = ("101_428/ver","101_56/ver","101_78/ver","102_249/ver","102_38/ver")
SOURCES = (SOURCE,"geobc/elastic_crosscad.py","geobc/elastic_pilot.py","geobc/elastic_pilot_data.py",
    "geobc/continuous_query_geometry.py","geobc/solver_selection_models.py","geobc/solver_selection_data.py",
    "geobc/conditioned_field.py","geobc/fno.py","geobc/benchmark.py")
ALLOWED = ("xyz","features","grid_node_ids","sample_key","label_alignment","faces")


def helpers():
    from . import elastic_crosscad as cross
    from . import elastic_pilot as base
    return cross,base


def host_path(path):
    return helpers()[0].host_path(path)


def read_json(path):
    return helpers()[0].read_json(path)


def sha(path):
    return helpers()[0].sha(path)


def require(condition,message):
    return helpers()[0].require(condition,message)


def identity(path):
    return helpers()[0].identity(path)


def finite_cost(value):
    return isinstance(value,(int,float)) and math.isfinite(value) and value>=0


def request_contract(plan_path=PLAN):
    """Metadata-only, including a pending base gate; no weights/array imports."""
    cross,base = helpers(); plan = read_json(plan_path)
    require(plan.get("kind") == KIND and tuple(plan.get("case_keys",())) == CASES,"Fixed warmstart five-CAD scope differs")
    neural,physics = plan["neural"],plan["physics"]
    require(neural["method"] == "fno" and neural["seed"] == 2026 and neural["checkpoint_expected_step"] == 2880
        and neural["resolution"] == 64 and neural["cloud_points"] == 1024 and neural["cloud_seed"] == 2026
        and tuple(neural["allowed_canonical_members"]) == ALLOWED,"Frozen FNO/frontend input scope differs")
    require(physics["maximum_attempted_iterations_per_case"] == 10000 and physics["maximum_cumulative_solve_wall_seconds_per_case"] == 300
        and physics["true_relative_residual_tolerance"] == 1e-8 and physics["true_residual_interval"] == 25
        and physics["checkpoint_interval"] == 25 and physics["resume_budget_reset"] is False
        and physics["no_operator_rebuild_or_guard_repeat"] is True and plan["whole_stage_budget"]["maximum_cumulative_wall_seconds"] == 900,
        "Frozen warm PCG/budget policy differs")
    expected_gate = {"minimum_CADs_with_at_least20percent_total_cold_cost_reduction":4,
        "maximum_worst_case_warm_to_zero_cost_ratio":1.10,"both_methods_true_converged_required":5,
        "allN_same_solution_gate_required":5,"inventory_denominator":5}
    require(all(plan["selection_gate"].get(key) == value for key,value in expected_gate.items())
        and plan["comparison"]["fullN_vector_difference_relative_l2_maximum_for_same_solution_gate"] == 1e-5,
        "Frozen macro selection gate differs")
    physical_contract,physical_plan = cross.request_contract(plan["physical_plan"])
    require(host_path(plan["physical_output"]) == host_path(physical_plan["output"]),"Fixed base output differs")
    model_run = host_path(neural["checkpoint"]).parent
    model_config = read_json(model_run/"config.json")
    require(model_config["method"] == "fno" and model_config["seed"] == 2026,"Frozen model run identity differs")
    cache = read_json(neural["data_cache_manifest"])
    require(cache.get("status") == "completed" and cache["scaler"] == model_config["scaler"]
        and cache["contract"] == model_config["cache_contract"],"Strict frozen TRAIN scaler/cache identity differs")
    checkpoint = identity(neural["checkpoint"])
    require(checkpoint["bytes"] == neural["checkpoint_bytes"] and host_path(neural["checkpoint"]).name == "last.pt",
        "Fixed endpoint checkpoint stat/role differs")
    records = {}; base_summary_path = host_path(plan["physical_output"])/"summary.json"
    summary = read_json(base_summary_path) if base_summary_path.exists() else {}
    if summary.get("status") == "completed":
        require(summary.get("kind") == "elastic_crosscad_screen" and summary.get("config",{}).get("contract") == physical_contract,
            "Completed zero screen source contract differs")
    base_records = {}
    for key in CASES:
        record = physical_contract["records"][key]
        case_summary_path = host_path(record["solve_output"])/"summary.json"
        old = read_json(case_summary_path) if case_summary_path.exists() else {}
        valid_identity = old.get("config") == {"contract":record,"stage":"elastic_crosscad_pcg"}
        costs = old.get("costs",{})
        zero = costs.get("cold_case_accounted_wall_seconds"); gt = costs.get("GT_read_and_metrics_seconds")
        input_cost,operator_cost = costs.get("input_preparation_seconds"),costs.get("operator_wall_seconds")
        cold_complete = all(finite_cost(value) for value in (zero,gt,input_cost,operator_cost)) and zero>gt
        proof = {"status":old.get("status"),"identity_valid":valid_identity,"numerical_state":old.get("numerical_state"),
            "numerically_converged":old.get("numerically_converged") is True,"true_relative_residual":old.get("true_relative_residual"),
            "input_compatible":old.get("input_compatible") is True,"operator_compatible":old.get("operator_compatible") is True,
            "original_metrics_available":old.get("metrics") is not None,
            "summary_sha256":sha(case_summary_path) if case_summary_path.exists() else None,
            "physical_u_record":old.get("prediction"),"costs":{"zero_cold_cost_seconds":zero-gt if cold_complete else None,
                "inherited_physical_input_operator_seconds":input_cost+operator_cost if cold_complete else None},
            "cold_components_complete":cold_complete}
        if proof["physical_u_record"] is not None:
            value = proof["physical_u_record"]
            require({part:value[part] for part in ("bytes","mtime_ns")} == {part:identity(value["path"])[part] for part in ("bytes","mtime_ns")},
                "Numerical reference prediction stat changed")
        base_records[key] = proof
        records[key] = {"physical_record":record,"source":record["sources"]["canonical_npz"],"base":proof}
        cache_record = cache["records"][key]
        require(cache_record["num_nodes"] == record["expected_geometry"]["nodes"],"FNO/physical original node count differs")
        source = next(item for item in cache["contract"]["sources"] if item["key"] == key)
        require(host_path(source["path"]) == host_path(record["sources"]["canonical_npz"]["path"])
            and all(source[part] == record["sources"]["canonical_npz"][part] for part in ("bytes","mtime_ns")),"FNO/physical canonical source identity differs")
    completed = summary.get("status") == "completed" and all(item["status"] == "completed" and item["identity_valid"] for item in base_records.values())
    converged = all(item["numerically_converged"] and item["numerical_state"] == "converged"
        and finite_cost(item["true_relative_residual"]) and item["true_relative_residual"]<=1e-8 for item in base_records.values())
    available = all(item["original_metrics_available"] for item in base_records.values())
    eligible = completed and converged and available and all(item["input_compatible"] and item["operator_compatible"] and item["cold_components_complete"] for item in base_records.values())
    source_pins = {name:sha(name) for name in SOURCES}
    for name,value in plan.get("source_sha256",{}).items():
        require(name in source_pins and source_pins[name] == value,"Pinned source differs: "+name)
    return {"schema":"geobc_elastic_frozen_fno_warmstart_v1","kind":KIND,"case_keys":list(CASES),
        "plan_science":{key:plan[key] for key in ("neural","physics","whole_stage_budget","comparison","cost","selection_gate")},
        "physical_contract":physical_contract,"physical_plan":str(host_path(plan["physical_plan"])),
        "base_gate":{"all_five_completed":completed,"all_five_true_converged":converged,
            "all_five_original_metrics_available":available,"eligible":eligible,"records":base_records},"records":records,
        "base_summary_sha256":sha(base_summary_path) if base_summary_path.exists() else None,
        "checkpoint":checkpoint,"checkpoint_config_sha256":cross.digest(model_config),"expected_model_source_config":model_config["model"],
        "expected_parameter_counts":model_config["parameter_counts"],"scaler":cache["scaler"],
        "model_config_file_sha256":sha(model_run/"config.json"),"selection_cache_metadata_sha256":sha(neural["data_cache_manifest"]),
        "source_sha256":source_pins,"output":str(host_path(plan["output"])),"array_or_weight_reads_in_request_contract":0},plan


def zero_receipt():
    return {"GT_reads":0,"displacement_reads":0,"stress_reads":0,"old_final_reads":0,"DEV_dia_reads":0,
        "sealed_new_test_reads":0,"optimizer_steps":0,"scaler_fits":0,"K_guards_repeated":0,"setup_repeated":0}


def load_model(contract,plan,output):
    """Only actual release permits the one fixed checkpoint/scaler/model read."""
    cross,base = helpers(); began = time.monotonic()
    import torch
    from .solver_selection_models import build_selection_model
    device = torch.device("cuda:0"); torch.cuda.init(); torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    model = build_selection_model("fno",device=device,seed=2026)
    require(model.source_config == contract["expected_model_source_config"] and model.parameter_counts == contract["expected_parameter_counts"],
        "Actual frozen FNO source_config/parameter counts differ")
    cross.match(plan["neural"]["checkpoint"],contract["checkpoint"])
    actual_sha = sha(plan["neural"]["checkpoint"])
    attempts_path = output/"neural_weight_read_attempts.json"
    attempts = read_json(attempts_path) if attempts_path.exists() else {}
    if attempts:
        require(attempts["checkpoint_stat"] == contract["checkpoint"],"Weight-read attempt identity differs")
    base.write_json(attempts_path,{"checkpoint_stat":contract["checkpoint"],"checkpoint_sha256":actual_sha,
        "attempts":attempts.get("attempts",0)+1,"only_fixed_FNO_checkpoint":True})
    saved = torch.load(host_path(plan["neural"]["checkpoint"]),map_location="cpu",weights_only=False)
    require(saved["step"] == 2880 and cross.digest(saved["config"]) == contract["checkpoint_config_sha256"]
        and saved["config"]["scaler"] == contract["scaler"] and saved["config"]["model"] == model.source_config,
        "Frozen checkpoint step/config/scaler/source identity differs")
    model.load_state_dict(saved["model"],strict=True); del saved
    model.eval(); base._sync(torch); seconds = time.monotonic()-began
    path = output/"neural_load_receipt.json"; previous = read_json(path) if path.exists() else {}
    if previous:
        require(previous["checkpoint_sha256"] == actual_sha,"Checkpoint SHA changed after first actual load")
    receipt = {"checkpoint_sha256":actual_sha,"checkpoint_stat":contract["checkpoint"],
        "first_actual_load_seconds":previous.get("first_actual_load_seconds",seconds),"this_call_load_seconds":seconds,
        "cumulative_actual_load_seconds":previous.get("cumulative_actual_load_seconds",0.)+seconds,
        "actual_checkpoint_loads":previous.get("actual_checkpoint_loads",0)+1,"parameter_counts":model.parameter_counts,
        "source_config":model.source_config,"GT_reads":0,"optimizer_steps":0}
    base.write_json(path,receipt)
    return model,receipt


def cold_prediction(key,record,model,contract,output):
    cross,base = helpers(); path = output/"neural_prediction.json"
    if path.exists():
        saved = read_json(path); require(saved["input_record"] == record and saved["checkpoint"] == contract["checkpoint"],"Saved neural initial guess identity differs")
        require(base.stat_record(saved["raw_prediction"]["path"]) == {part:saved["raw_prediction"][part] for part in ("bytes","mtime_ns")},"Saved raw prediction stat changed")
        return saved
    began = time.monotonic()
    import numpy as np
    import torch
    from .conditioned_field import load_geometry_cloud
    from .solver_selection_data import _stencil,_splat,query_grid,ALIGNMENT
    source = record["source"]; cross.match(source["path"],source)
    with np.load(host_path(source["path"]),allow_pickle=False) as archive:
        require(str(archive["sample_key"].item()) == key and str(archive["label_alignment"].item()) == ALIGNMENT,"Canonical input identity/alignment differs")
        xyz = np.array(archive["xyz"],dtype=np.float32); features = np.array(archive["features"],dtype=np.float32)
        ids = np.array(archive["grid_node_ids"],dtype=np.int64)
    physical = record["physical_record"]; expected = physical["expected_geometry"]
    from .elastic_pilot_data import _array_sha
    require(xyz.shape == (expected["nodes"],3) and features.shape == (len(xyz),25) and ids.shape == (len(xyz),)
        and np.isfinite(xyz).all() and np.isfinite(features).all()
        and _array_sha(xyz,"<f4") == expected["canonical_xyz_float32_sha256"]
        and _array_sha(ids,"<i8") == expected["grid_node_ids_int64_sha256"],"Cold frontend coordinate/GRID input schema differs")
    cloud = load_geometry_cloud(host_path(source["path"]),cloud_points=1024,seed=2026,cad_key=physical["cad"])
    device = torch.device("cuda:0"); x = torch.as_tensor(xyz,device=device); f = torch.as_tensor(features,device=device)
    cloud_tensor = torch.as_tensor(cloud.points[None],device=device); mask = torch.as_tensor(cloud.mask[None],device=device)
    indices,weights = _stencil(x,64); raster,active = _splat(f,indices,weights,64)
    axis = torch.linspace(0,1,64,device=device); coordinates = torch.stack(torch.meshgrid(axis,axis,axis,indexing="ij"))
    grid = torch.cat((active.float().reshape(1,64,64,64),coordinates,raster.T.reshape(25,64,64,64)),dim=0)[None]
    base._sync(torch); frontend_seconds = time.monotonic()-began; tick = time.monotonic()
    with torch.no_grad():
        context = model.encode_geometry(cloud_tensor,mask); encoded = query_grid(model.forward_grid(grid,context),{"query_indices":indices,"query_weights":weights})
        require(encoded.shape == (len(xyz),4) and torch.isfinite(encoded).all(),"Nonfinite/incorrect FNO prediction")
        mean = torch.tensor(contract["scaler"]["mean"][:3],device=device,dtype=torch.float64)
        std = torch.tensor(contract["scaler"]["std"][:3],device=device,dtype=torch.float64)
        raw_u = encoded[:,:3].double()*std+mean
    base._sync(torch); inference_seconds = time.monotonic()-tick; tick = time.monotonic()
    raw_record = base._save_array(np,output/"raw_FNO_u_mm.npy",raw_u)
    persistence_seconds = time.monotonic()-tick
    saved = {"input_record":record,"checkpoint":contract["checkpoint"],"raw_prediction":raw_record,
        "frontend_seconds":frontend_seconds,"inference_query_decode_seconds":inference_seconds,"prediction_persistence_seconds":persistence_seconds,
        "cloud":cloud.record(),"node_count":len(xyz),"cold_frontend_rebuilt":True,"cached_grid_reads":0,
        "canonical_members_accessed":["xyz","features","grid_node_ids","sample_key","label_alignment","faces (optional by original cloud loader)"],
        "coordinate_GRID_identity_checked":True,"read_receipt":zero_receipt()}
    base.write_json(output/"neural_input_manifest.json",saved)
    base.write_json(path,saved)
    saved["prediction_persistence_seconds"] = time.monotonic()-tick
    base.write_json(output/"neural_input_manifest.json",saved)
    base.write_json(path,saved)
    return saved


def numerical_difference(np,prediction,reference):
    require(prediction.shape == reference.shape and np.isfinite(prediction).all() and np.isfinite(reference).all(),"Nonfinite or non-allN numerical comparison")
    difference = prediction-reference; reference_energy = float(np.square(reference).sum()); error = float(np.square(difference).sum())
    rms = math.sqrt(error/len(reference)); relative = None if reference_energy == 0 else math.sqrt(error/reference_energy)
    return {"relative_l2":relative,"vector_RMS_mm":rms,"max_nodal_vector_difference_mm":float(np.linalg.norm(difference,axis=1).max()),
        "zero_reference":reference_energy == 0,"passed":rms<=1e-12 if reference_energy == 0 else relative<=1e-5,
        "reference_kind":"converged approximate zero-start numerical displacement, not GT"}


def warm_solve(key,record,neural,load_receipt,metadata_seconds,output,clock,contract):
    cross,base = helpers()
    case_contract = {"stage_contract_sha256":cross.digest(contract),"case_key":key,"physical_record":record["physical_record"],
        "checkpoint":contract["checkpoint"],"source_sha256":contract["source_sha256"][SOURCE]}
    config = {"contract":case_contract,"stage":"elastic_fno_warm_pcg"}
    if (output/"config.json").exists():
        require(read_json(output/"config.json") == config,"Warm PCG config differs")
    if (output/"summary.json").exists():
        saved = read_json(output/"summary.json"); require(saved.get("config") == config,"Warm result config differs")
        if saved.get("status") == "completed":
            return saved
    if (output/"endpoint_no_GT.json").exists():
        saved = read_json(output/"endpoint_no_GT.json"); require(saved["config"] == config,"Warm endpoint identity differs")
        base.write_json(output/"summary.json",saved)
        return saved
    base.write_json(output/"config.json",config)
    begun = time.monotonic(); state = None; solve_began = None; base_seconds = None
    case_base = {"key":key,"group":record["physical_record"]["group"],"node_count":record["physical_record"]["expected_geometry"]["nodes"],
        "attempted":True,"config":config,"status":"completed","kind":"elastic_fno_warmstart_case","read_receipt":zero_receipt(),
        "metrics":None,"numerically_converged":False,"numerical_same_solution":None,
        "costs":{"components_complete":False,"zero_cold_cost_seconds":record["base"]["costs"]["zero_cold_cost_seconds"],
            "warm_cold_cost_seconds":None,"warm_to_zero_cost_ratio":None,"cold_cost_reduction_fraction":None}}
    try:
        import numpy as np
        import torch
        from .elastic_pilot_data import load_input
        physical_record = record["physical_record"]
        manifest_path = host_path(physical_record["data_output"])/"manifest.json"
        manifest = read_json(manifest_path); operator_record = read_json(host_path(physical_record["operator_output"])/"summary.json")
        require(manifest.get("status") == "completed" and manifest.get("feasibility_pass") is True
            and manifest.get("contract") == physical_record,"Warm start requires accepted strict input cache")
        require(operator_record.get("status") == "completed" and operator_record.get("feasibility_pass") is True
            and operator_record.get("config") == {"contract":physical_record,"stage":"elastic_crosscad_operator"}
            and operator_record["algebra_guards"]["K_applications"] == 8
            and operator_record["input_manifest_sha256"] == sha(manifest_path),"Warm start requires accepted strict eight-guard operator cache")
        device = torch.device("cuda:0"); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        inputs = load_input(manifest,device); lam,mu = base._validate_inputs(torch,inputs,physical_record)
        arrays = {name:base._load_array(np,torch,value,device) for name,value in operator_record["arrays"].items()}
        require(arrays["gradients"].shape == (len(inputs["cells10"]),4,10,3)
            and arrays["weights"].shape == (len(inputs["cells10"]),4)
            and arrays["free_cholesky"].shape == (len(inputs["free_node_ids"]),3,3)
            and all(value.dtype == torch.float64 for value in arrays.values()),"Committed physical operator dimensions/dtype differ")
        operator = base.MatrixFreeElastic(torch,inputs["cells10"],arrays["gradients"],arrays["weights"],len(inputs["vertices_mm"]),lam,mu)
        free,chol = inputs["free_node_ids"],arrays["free_cholesky"]
        b = inputs["force"][free]; norm_b = float(torch.linalg.vector_norm(b))
        raw_u = base._load_array(np,torch,neural["raw_prediction"],device)
        require(raw_u.shape == inputs["vertices_mm"].shape and raw_u.dtype == torch.float64 and torch.isfinite(raw_u).all(),"Raw FNO original-node displacement differs")
        z = raw_u[free].clone(); r = torch.zeros_like(z); p = torch.zeros_like(z)
        state = {"config":config,"iteration":0,"iterations_executed_total":0,"cumulative_solve_wall_seconds":0.,
            "cumulative_setup_read_seconds":0.,"cumulative_reference_IO_and_comparison_seconds":0.,
            "cumulative_endpoint_verification_seconds":0.,"numerical_reference_reads":0,"K_applications":0,"trace":[],"rz":None}
        if (output/"checkpoint.json").exists():
            state = read_json(output/"checkpoint.json")
            require(state["config"] == config and base.stat_record(state["vectors_path"]) == state["vectors_stat"],"Warm PCG checkpoint identity differs")
            with np.load(host_path(state["vectors_path"]),allow_pickle=False) as archive:
                require(set(archive.files) == {"z","r","p"},"Unexpected warm PCG vector member")
                z,r,p = (torch.as_tensor(np.array(archive[name]),device=device) for name in ("z","r","p"))
            require(all(value.shape == b.shape and value.dtype == torch.float64 and torch.isfinite(value).all() for value in (z,r,p)),"Malformed warm PCG state")
        if (output/"status.json").exists():
            journal = read_json(output/"status.json"); require(journal.get("config") == config,"Warm accounting journal differs")
            for name in ("cumulative_solve_wall_seconds","iterations_executed_total","K_applications","cumulative_setup_read_seconds",
                    "cumulative_reference_IO_and_comparison_seconds","cumulative_endpoint_verification_seconds","numerical_reference_reads"):
                state[name] = max(state.get(name,0),journal.get(name,0))
        base._sync(torch); state["cumulative_setup_read_seconds"] += time.monotonic()-begun
        reference_began = time.monotonic()
        value = record["base"]["physical_u_record"]
        require(base.stat_record(value["path"]) == {part:value[part] for part in ("bytes","mtime_ns")},"Frozen numerical reference stat changed")
        reference = np.load(host_path(value["path"]),allow_pickle=False)
        state["numerical_reference_reads"] = state.get("numerical_reference_reads",0)+1
        require(reference.shape == tuple(inputs["vertices_mm"].shape) and reference.dtype == np.float64 and np.isfinite(reference).all(),"Numerical reference full-node schema differs")
        initial_path = output/"initial_numerical_comparison.json"
        if initial_path.exists():
            initial = read_json(initial_path); require(initial["config"] == config,"Initial comparison identity differs")
        else:
            feasible = base._restore_full(torch,raw_u[free],inputs)
            initial = {"config":config,"raw_FNO_to_physical_reference":numerical_difference(np,raw_u.cpu().numpy(),reference),
                "feasible_initial_to_physical_reference":numerical_difference(np,feasible.cpu().numpy(),reference),
                "raw_fixed_translation_max_abs_mm":float(raw_u[inputs["fixed_node_ids"]].abs().max()),
                "raw_to_feasible_max_nodal_difference_mm":float(torch.linalg.vector_norm(raw_u-feasible,dim=1).max()),
                "feasible_initial_prediction":base._save_array(np,output/"feasible_initial_u_mm.npy",feasible),
                "same_original_node_identity":True,"GT_reads":0}
            base.write_json(initial_path,initial)
        base._sync(torch)
        state["cumulative_reference_IO_and_comparison_seconds"] = state.get("cumulative_reference_IO_and_comparison_seconds",0.)+time.monotonic()-reference_began
        base_seconds,base_K = state["cumulative_solve_wall_seconds"],state["K_applications"]
        solve_began = time.monotonic(); preconditioner_seconds = 0.; checkpoint_seconds = 0.; frozen_solve_seconds = None

        def elapsed():
            base._sync(torch)
            return frozen_solve_seconds if frozen_solve_seconds is not None else base_seconds+time.monotonic()-solve_began

        def reduced_apply(value):
            physical = torch.zeros_like(inputs["vertices_mm"]); physical[free] = value
            return operator.apply(physical)[free]

        def precondition(value):
            nonlocal preconditioner_seconds
            base._sync(torch); tick = time.monotonic()
            result = torch.cholesky_solve(value[...,None],chol).squeeze(-1)
            base._sync(torch); preconditioner_seconds += time.monotonic()-tick
            return result

        def account(phase):
            state.update(cumulative_solve_wall_seconds=elapsed(),K_applications=base_K+operator.calls)
            base.write_json(output/"status.json",{**state,"phase":phase,"trace":None})
            clock.account(phase,key)

        def save():
            nonlocal state,checkpoint_seconds
            tick = time.monotonic(); account("solving")
            state = base._save_pcg(np,output,state,z,r,p)
            checkpoint_seconds += time.monotonic()-tick; account("solving")

        true_r = b-reduced_apply(z); true_relative = base._relative(torch,true_r,norm_b)
        if not torch.isfinite(true_r).all() or true_relative is None:
            raise base.ScientificNegative("Nonfinite/undefined initial/resumed warm residual")
        state["trace"].append({"iteration":state["iteration"],"true_relative_residual":true_relative,
            "cumulative_solve_wall_seconds":elapsed(),"reason":"initial_or_resume_true_verification"})
        if "initial_true_relative_residual" not in state:
            state["initial_true_relative_residual"] = true_relative
        if state["iteration"] == 0:
            r = true_r.clone(); preconditioned = precondition(r); rz = float((r*preconditioned).sum()); p = preconditioned.clone()
        else:
            rz = state["rz"]
        state["rz"] = rz
        numerical_state = "converged" if true_relative<=1e-8 else "solving"
        if numerical_state == "solving" and (not math.isfinite(rz) or rz<=0):
            raise base.ScientificNegative("Warm rTMinvr nonfinite/nonpositive")
        save()
        while numerical_state == "solving":
            if state["iterations_executed_total"]>=10000 or elapsed()>=300 or clock.remaining()<=0:
                numerical_state = "iteration_budget_limit" if state["iterations_executed_total"]>=10000 else (
                    "wall_budget_limit" if elapsed()>=300 else "stage_wall_budget_limit")
                break
            state["iterations_executed_total"] += 1; account("solving")
            Ap = reduced_apply(p); pAp = float((p*Ap).sum())
            if not math.isfinite(pAp) or pAp<=0 or not torch.isfinite(Ap).all():
                raise base.ScientificNegative("Warm PCG pTAp nonfinite/nonpositive; no solver/shift change")
            alpha = rz/pAp; z = z+alpha*p; r = r-alpha*Ap; state["iteration"] += 1
            if not torch.isfinite(z).all() or not torch.isfinite(r).all():
                raise base.ScientificNegative("Nonfinite warm PCG solution/residual")
            recursive = base._relative(torch,r,norm_b)
            if state["iteration"]%25 == 0 or (recursive is not None and recursive<=1e-8):
                true_r = b-reduced_apply(z); true_relative = base._relative(torch,true_r,norm_b)
                if not torch.isfinite(true_r).all() or true_relative is None:
                    raise base.ScientificNegative("Nonfinite/undefined warm true residual")
                state["trace"].append({"iteration":state["iteration"],"true_relative_residual":true_relative,
                    "recursive_relative_residual":recursive,"cumulative_solve_wall_seconds":elapsed()})
                if true_relative<=1e-8:
                    numerical_state = "converged"
            if numerical_state == "solving":
                preconditioned = precondition(r); next_rz = float((r*preconditioned).sum())
                if not math.isfinite(next_rz) or next_rz<=0:
                    raise base.ScientificNegative("Warm PCG rTMinvr nonfinite/nonpositive")
                p = preconditioned+(next_rz/rz)*p; rz = next_rz; state["rz"] = rz
            account("solving")
            if state["iteration"]%25 == 0:
                save()
        true_r = b-reduced_apply(z); true_relative = base._relative(torch,true_r,norm_b)
        if not torch.isfinite(true_r).all() or true_relative is None:
            raise base.ScientificNegative("Nonfinite/undefined warm endpoint true residual")
        if true_relative<=1e-8:
            numerical_state = "converged"
        state["trace"].append({"iteration":state["iteration"],"true_relative_residual":true_relative,
            "cumulative_solve_wall_seconds":elapsed(),"reason":"endpoint_true_verification"})
        save(); solve_seconds = elapsed(); frozen_solve_seconds = solve_seconds
        solve_K_calls = operator.calls; solve_K_seconds = operator.apply_seconds
        verify_began = time.monotonic(); physical = base._restore_full(torch,z,inputs)
        residual = operator.apply(physical)-inputs["force"]
        reaction = torch.zeros_like(residual); reaction[inputs["fixed_node_ids"]] = residual[inputs["fixed_node_ids"]]
        external = inputs["force"]+reaction; reference_point = inputs["reference_xyz_mm"]
        force = external.sum(0); moment = torch.cross(inputs["vertices_mm"]-reference_point,external,dim=-1).sum(0)
        if any(not torch.isfinite(item).all() for item in (physical,residual,force,moment)):
            raise base.ScientificNegative("Nonfinite warm physical endpoint/equilibrium")
        prediction = base._save_array(np,output/"physical_u_mm.npy",physical)
        reaction_record = base._save_array(np,output/"constraint_reaction_N.npy",reaction)
        grid_record = base._save_array(np,output/"source_grid_node_ids.npy",inputs["grid_node_ids"])
        equilibrium = {"true_free_residual_relative":base._relative(torch,residual[free],norm_b),
            "true_free_residual_norm_N":float(torch.linalg.vector_norm(residual[free])),
            "constraint_reaction_sum_N":reaction.sum(0).tolist(),"net_force_with_reaction_N":force.tolist(),
            "net_moment_with_reaction_Nmm":moment.tolist(),"fixed_translation_max_abs_mm":float(physical[inputs["fixed_node_ids"]].abs().max()),
            "full_K_verification_calls":1}
        base._sync(torch)
        state["cumulative_endpoint_verification_seconds"] = state.get("cumulative_endpoint_verification_seconds",0.)+time.monotonic()-verify_began
        endpoint_seconds = state["cumulative_endpoint_verification_seconds"]; tick = time.monotonic()
        same_solution = numerical_difference(np,physical.cpu().numpy(),reference)
        state["cumulative_reference_IO_and_comparison_seconds"] += time.monotonic()-tick
        comparison_seconds = state["cumulative_reference_IO_and_comparison_seconds"]
        inherited = record["base"]["costs"]["inherited_physical_input_operator_seconds"]
        model_charge = load_receipt["first_actual_load_seconds"]+metadata_seconds
        components = [model_charge,neural["frontend_seconds"],neural["inference_query_decode_seconds"],neural["prediction_persistence_seconds"],
            inherited,state["cumulative_setup_read_seconds"],solve_seconds,endpoint_seconds,comparison_seconds]
        complete = all(finite_cost(item) for item in components) and record["base"]["cold_components_complete"]
        warm_cold = sum(components) if complete else None; zero_cold = record["base"]["costs"]["zero_cold_cost_seconds"]
        ratio = warm_cold/zero_cold if complete and zero_cold>0 else None
        summary = {**case_base,"numerical_state":numerical_state,"numerically_converged":numerical_state == "converged",
            "iteration":state["iteration"],"iterations_executed_total":state["iterations_executed_total"],"true_relative_residual":true_relative,
            "initial_true_relative_residual":state["initial_true_relative_residual"],"trace":state["trace"],
            "numerical_same_solution":same_solution,"raw_FNO_to_physical_reference":initial["raw_FNO_to_physical_reference"],
            "feasible_initial_to_physical_reference":initial["feasible_initial_to_physical_reference"],"initial_numerical_comparison":initial,
            "raw_FNO_prediction":neural["raw_prediction"],"prediction":prediction,"physical_u_path":prediction["path"],
            "constraint_reaction":reaction_record,"source_grid_node_ids":grid_record,"equilibrium":equilibrium,
            "costs":{"components_complete":complete,"zero_cold_cost_seconds":zero_cold,"warm_cold_cost_seconds":warm_cold,
                "warm_to_zero_cost_ratio":ratio,"cold_cost_reduction_fraction":None if ratio is None else 1-ratio,
                "neural_model_load_full_charge_seconds":model_charge,"neural_frontend_seconds":neural["frontend_seconds"],
                "neural_inference_query_decode_seconds":neural["inference_query_decode_seconds"],
                "neural_prediction_persistence_seconds":neural["prediction_persistence_seconds"],
                "inherited_physical_input_operator_seconds":inherited,"solve_setup_read_seconds":state["cumulative_setup_read_seconds"],
                "cumulative_solve_wall_seconds":solve_seconds,"this_call_solve_wall_seconds":solve_seconds-base_seconds,
                "endpoint_verification_seconds":endpoint_seconds,"physical_reference_IO_and_comparison_seconds":comparison_seconds,
                "K_apply_seconds_this_call":solve_K_seconds,"preconditioner_seconds_this_call":preconditioner_seconds,
                "checkpoint_seconds_this_call":checkpoint_seconds,"solve_K_count_cumulative":base_K+solve_K_calls,
                "solve_K_count_this_call":solve_K_calls,"verification_K_count":1,
                "cached_incremental_wall_seconds":time.monotonic()-begun,
                "cached_incremental_definition":"Actual solve call after persisted raw neural prediction and accepted FEM caches; not a new cold frontend",
                "batch_amortized_cold_cost_seconds":None if warm_cold is None else warm_cold-model_charge*4/5,
                "solve_budget_overshoot_seconds":max(0.,solve_seconds-300)},
            "memory":base._memory(torch),"read_receipt":{**zero_receipt(),"numerical_reference_reads":state["numerical_reference_reads"],
                "physical_endpoint_and_equilibrium_persisted":True,"raw_and_feasible_initial_guesses_separate":True},
            "neural_frontend":neural,"inherited_operator_guard_K_count":8,"scope":"Numerical reference agreement only; no new GT accuracy"}
        tick = time.monotonic(); base.write_json(output/"endpoint_no_GT.json",summary)
        persistence_seconds = time.monotonic()-tick
        summary["costs"]["endpoint_metadata_persistence_seconds"] = persistence_seconds
        if complete:
            warm_cold += persistence_seconds; ratio = warm_cold/zero_cold
            summary["costs"].update(warm_cold_cost_seconds=warm_cold,warm_to_zero_cost_ratio=ratio,
                cold_cost_reduction_fraction=1-ratio,batch_amortized_cold_cost_seconds=warm_cold-model_charge*4/5)
        base.write_json(output/"endpoint_no_GT.json",summary); base.write_json(output/"summary.json",summary)
        state["numerical_state"] = numerical_state; account("completed")
        return summary
    except base.ScientificNegative as error:
        if state is not None and solve_began is not None:
            state.update(cumulative_solve_wall_seconds=elapsed(),K_applications=base_K+operator.calls)
            base.write_json(output/"status.json",{**state,"trace":None,"phase":"completed","numerical_state":"breakdown"})
        summary = {**case_base,"numerical_state":"breakdown","reason":str(error),"elapsed_seconds":time.monotonic()-begun,
            "read_receipt":{**zero_receipt(),"numerical_reference_reads":0 if state is None else state.get("numerical_reference_reads",0)}}
        base.write_json(output/"summary.json",summary); clock.account("case_negative",key)
        return summary
    except Exception:
        if state is not None and solve_began is not None:
            state.update(cumulative_solve_wall_seconds=elapsed(),K_applications=base_K+operator.calls)
            base.write_json(output/"status.json",{**state,"trace":None,"phase":"failed"})
        clock.account("case_failed",key)
        raise


def selection_gate(cases):
    """Fixed five-case numerical/cold-cost criteria, independent of GT."""
    converged = same = gains = 0; ratios = []
    for key in CASES:
        item = cases[key]; value = item.get("true_relative_residual")
        converged += int(item.get("numerically_converged") is True and item.get("numerical_state") == "converged"
            and finite_cost(value) and value<=1e-8)
        difference = item.get("numerical_same_solution") or {}; relative = difference.get("relative_l2")
        match = (difference.get("zero_reference") is False and finite_cost(relative) and relative<=1e-5) or (
            difference.get("zero_reference") is True and relative is None
            and finite_cost(difference.get("vector_RMS_mm")) and difference["vector_RMS_mm"]<=1e-12)
        same += int(match)
        costs = item.get("costs",{}); ratio = costs.get("warm_to_zero_cost_ratio")
        if costs.get("components_complete") is True and finite_cost(ratio):
            ratios.append(ratio); gains += int(ratio<=.8)
    worst = max(ratios) if len(ratios) == 5 else None
    return {"passed":converged == 5 and same == 5 and gains>=4 and worst is not None and worst<=1.10,
        "numerically_converged_cases":converged,"same_solution_passed_cases":same,
        "at_least20percent_reduction_cases":gains,"worst_warm_to_zero_cost_ratio":worst,"required_denominator":5}


def unattempted_case(key,record,config):
    return {"key":key,"group":record["physical_record"]["group"],"node_count":record["physical_record"]["expected_geometry"]["nodes"],
        "status":"completed","kind":"elastic_fno_warmstart_case","config":config,"attempted":False,
        "numerical_state":"stage_budget_exhausted_negative","numerically_converged":False,"true_relative_residual":None,
        "numerical_same_solution":None,"metrics":None,"read_receipt":zero_receipt(),
        "costs":{"components_complete":False,"zero_cold_cost_seconds":record["base"]["costs"]["zero_cold_cost_seconds"],
            "warm_cold_cost_seconds":None,"warm_to_zero_cost_ratio":None,"cold_cost_reduction_fraction":None},
        "reason":"No new case after cumulative900s; fixed inventory retained with null cold cost, no GT"}


def run(plan_path=PLAN,output_path=None):
    began = time.monotonic(); contract,plan = request_contract(plan_path); cross,base = helpers()
    require(plan.get("scientific_rules_frozen") is True and plan.get("actual_execution_released") is True,
        "Root must release the frozen stage before scientific imports/weight or array reads")
    require(set(plan.get("source_sha256",{})) == set(SOURCES),"Actual stage needs exactly ten frozen source pins")
    require(contract["base_gate"]["eligible"] is True,"All five accepted zero-start endpoints must be completed and true-converged")
    review = read_json(plan["root_review"])
    require(review.get("passed") is True and review.get("source_sha256") == contract["source_sha256"][SOURCE]
        and review.get("base_evidence_decision") == "proceed_frozen_FNO_warmstart",
        "Precise root source review and baseline evidence decision required before science")
    output = host_path(output_path or plan["output"])
    require(output == host_path(plan["output"]),"Output differs from frozen stage destination")
    config = {"contract":contract,"stage":KIND}
    if (output/"config.json").exists():
        require(read_json(output/"config.json") == config,"Existing warm screen identity differs")
    if (output/"summary.json").exists():
        previous = read_json(output/"summary.json"); require(previous.get("config") == config,"Existing warm result identity differs")
        if previous.get("status") == "completed":
            return previous
    output.mkdir(parents=True,exist_ok=True); base.write_json(output/"config.json",config)
    metadata_path = output/"stage_metadata_receipt.json"
    previous_metadata = read_json(metadata_path) if metadata_path.exists() else {}
    if previous_metadata:
        require(previous_metadata["config_sha256"] == cross.digest(config),"Metadata cost receipt identity differs")
    metadata_seconds = previous_metadata.get("first_metadata_seconds",time.monotonic()-began)
    base.write_json(metadata_path,{"config_sha256":cross.digest(config),"first_metadata_seconds":metadata_seconds,
        "cumulative_actual_metadata_seconds":previous_metadata.get("cumulative_actual_metadata_seconds",0.)+time.monotonic()-began})
    clock = cross.StageClock(output,config,began); clock.account("running")
    load_path = output/"neural_load_receipt.json"; load_receipt = read_json(load_path) if load_path.exists() else None
    cases = {}; model = None
    try:
        for key in CASES:
            record = contract["records"][key]; case_output = output/key.replace("/","_")
            case_contract = {"stage_contract_sha256":cross.digest(contract),"case_key":key,"physical_record":record["physical_record"],
                "checkpoint":contract["checkpoint"],"source_sha256":contract["source_sha256"][SOURCE]}
            case_config = {"contract":case_contract,"stage":"elastic_fno_warm_pcg"}
            summary_path = case_output/"summary.json"
            if summary_path.exists():
                item = read_json(summary_path); require(item.get("config") == case_config,"Saved warm case identity differs")
                if item.get("status") == "completed":
                    cases[key] = item; clock.account("case_reused",key); continue
            case_output.mkdir(parents=True,exist_ok=True)
            prediction_path = case_output/"neural_prediction.json"
            if clock.remaining()<=0 and not prediction_path.exists() and not (case_output/"checkpoint.json").exists():
                item = unattempted_case(key,record,case_config)
                base.write_json(case_output/"config.json",case_config); base.write_json(summary_path,item)
                cases[key] = item; clock.account("case_not_attempted",key); continue
            if not prediction_path.exists() and model is None:
                clock.account("fixed_model_load",key); model,load_receipt = load_model(contract,plan,output)
                clock.account("fixed_model_loaded",key)
                if clock.remaining()<=0:
                    item = unattempted_case(key,record,case_config)
                    base.write_json(case_output/"config.json",case_config); base.write_json(summary_path,item)
                    cases[key] = item; clock.account("case_not_attempted",key); continue
            require(load_receipt is not None and load_receipt["checkpoint_stat"] == contract["checkpoint"],
                "Saved initial guess requires its actual model-load cost/identity receipt")
            clock.account("cold_neural_frontend",key)
            neural = cold_prediction(key,record,model,contract,case_output)
            clock.account("warm_physical_solve",key)
            item = warm_solve(key,record,neural,load_receipt,metadata_seconds,case_output,clock,contract)
            cases[key] = item; clock.account("case_completed",key)
        gate = selection_gate(cases); elapsed = clock.elapsed()
        weight_attempts = output/"neural_weight_read_attempts.json"
        weight_reads = read_json(weight_attempts)["attempts"] if weight_attempts.exists() else 0
        receipt = {**zero_receipt(),"model_or_weight_reads":weight_reads,
            "numerical_reference_reads":sum(item.get("read_receipt",{}).get("numerical_reference_reads",0) for item in cases.values()),
            "target_free_cold_frontend":True,"cached_neural_grid_reads":0,"inherited_operator_guard_K_count":40}
        summary = {"status":"completed","kind":KIND,"config":config,"case_keys":list(CASES),"cases":cases,
            "selection_gate":gate,"read_receipt":receipt,"elapsed_seconds":elapsed,
            "costs":{"cumulative_stage_wall_seconds":elapsed,"maximum_stage_wall_seconds":900,
                "stage_budget_overshoot_seconds":max(0.,elapsed-900),"this_call_stage_wall_seconds":time.monotonic()-began,
                "neural_model_load_seconds":None if load_receipt is None else load_receipt["first_actual_load_seconds"],
                "actual_model_load_seconds_including_resumes":0. if load_receipt is None else load_receipt["cumulative_actual_load_seconds"],
                "metadata_seconds":metadata_seconds,"checkpoint_actual_sha256":None if load_receipt is None else load_receipt["checkpoint_sha256"]},
            "training_cost_context":{"seconds":plan["neural"]["training_seconds_context"],"included_in_query_cost":False,
                "checkpoint_step":2880,"role":"Previously fixed learned prior, no new training"},
            "counts":{"fixed_cases":5,"attempted":sum(item["attempted"] for item in cases.values()),
                "numerically_converged":gate["numerically_converged_cases"],"same_solution":gate["same_solution_passed_cases"]},
            "scientific_outcome":"frozen_macro_speed_gate_passed" if gate["passed"] else "frozen_macro_speed_gate_failed",
            "scope":"Five fixed existing TRAIN/DEV ver CADs; approximate physical numerical reference, no GT accuracy or stress comparison",
            "budget_finalization":"No new case or PCG iteration after expiry; finite endpoint/verification persists with explicit overshoot",
            "primary_cost":"accounted single-query cold; inherited FEM input/operator plus full fixed model load each query",
            "secondary_costs":"actual cached incremental and five-query model-load amortization are separate",
            "new_training_updates":0,"old_GT_accuracy_assigned_to_new_endpoint":False}
        base.write_json(output/"summary.json",summary); clock.account("completed")
        return summary
    except Exception as error:
        clock.account("failed")
        weight_attempts = output/"neural_weight_read_attempts.json"
        base.write_json(output/"summary.json",{"status":"failed","kind":KIND,"config":config,"case_keys":list(CASES),"cases":cases,
            "reason":f"{type(error).__name__}: {error}","read_receipt":{**zero_receipt(),
                "model_or_weight_reads":read_json(weight_attempts)["attempts"] if weight_attempts.exists() else 0},
            "costs":{"cumulative_stage_wall_seconds":clock.elapsed()}})
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan",type=Path,default=Path(PLAN))
    result.add_argument("--output",type=Path)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    run(args.plan,args.output)
