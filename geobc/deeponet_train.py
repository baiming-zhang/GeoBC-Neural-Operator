"""Geometry/BC DeepONet and concat-MLP screens, plus fixed DEV5/dia readout.

Source checks use only AST, small JSON, source hashes and file stats. The root
queue freezes this proposal and prepares BC clouds before any actual execution.
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
from .fno_response_train import _basis_metadata

ROOT = shared.ROOT
MODES = ("deeponet", "concat_mlp")
ENDPOINTS = (0, 120, 240, 480)
NONFORCE = list(range(7)) + list(range(13,25))
SOURCES = ("geobc/deeponet_train.py", "geobc/deeponet_field.py", "geobc/deeponet_data.py",
           "geobc/solver_selection_train.py", "geobc/solver_selection_data.py",
           "geobc/fno_response_train.py", "geobc/conditioned_field.py", "geobc/benchmark.py")


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def stat_record(path):
    info=shared.host_path(path).stat()
    return {"bytes":info.st_size,"mtime_ns":info.st_mtime_ns}


def _rules(plan):
    fixed={"modes":list(MODES),"seed":2026,"cloud_seed":2026,"cloud_points":1024,
           "cloud_channels":22,"cloud_nonforce_slots":NONFORCE,"nonforce_slots":NONFORCE,"geometry_dim":32,"encoder_width":64,
           "train_loads":["ver","hor","tor"],"steps":480,"passes":4,"eval_steps":list(ENDPOINTS),
           "microbatch_nodes":32768,"eval_batch_size":32768,"optimizer":"Adam","learning_rate":.001,
           "weight_decay":0.,"fused":True,"stress_transform":"asinh","channels":[0,1,2,3],
           "grid_resolution":64,"training_readout_keys":["101_479/ver","107_320/hor","10_58/tor"]}
    if plan.get("kind")!="deeponet_screen" or any(plan["recipe"].get(k)!=v for k,v in fixed.items()):
        raise ValueError("DeepONet screen differs from the common fixed480 input/loss/optimizer protocol")
    if any(plan["access"].get(key) is not False for key in ("old_final","sealed_new_test","warm_start","scaler_fit")):
        raise ValueError("Old final/sealed inputs, scaler fitting and warm starts are outside this screen")
    if any(plan["development_dia"].get(key)!=value for key,value in {"load":"dia","checkpoint_step":480,"modes":list(MODES),
                                                                    "seed":2026,"all_nodes":True,"training_updates":0,"scaler_fits":0}.items()):
        raise ValueError("Only both fixed480 endpoints on the existing DEV5/dia are declared")
    if plan["numerical_tolerances"]!={"alpha_atol":2e-7,"alpha_rtol":1e-6,"normalized_span_atol":1e-7,
                                    "affine_atol":3e-6,"affine_rtol":2e-6,"force_atol_N":.01,
                                    "moment_atol_Nmm":.1,"physical_load_rtol":1e-6}:
        raise ValueError("Predeclared unit/alpha tolerances changed")


def _metadata(plan_path,mode):
    if mode not in MODES: raise ValueError("Unknown DeepONet screen mode")
    plan=shared.read_json(plan_path); _rules(plan)
    manifest_path=shared.host_path(plan["data_cache"])/"manifest.json"
    manifest=shared.read_json(manifest_path); shared._check_manifest(manifest)
    split=shared.read_json(plan["existing_split"])
    for group,keys in (("train",manifest["train_keys"]),("validation",manifest["validation_keys"])):
        expected=[f"{cad}/{load}" for cad in split["groups"][group] for load in ("ver","hor","tor")]
        if keys!=expected: raise ValueError("Existing known3 CAD split/order differs")
    if plan["development_dia"]["CADs"]!=split["groups"]["validation"]:
        raise ValueError("The dia scope must be exactly the already explored DEV5")
    if manifest["contract"]["split_sha256"]!=shared.source_sha(plan["existing_split"]):
        raise ValueError("Existing selection split identity changed")
    for row in manifest["contract"]["sources"]:
        if stat_record(row["path"])!={k:row[k] for k in ("bytes","mtime_ns")}:
            raise ValueError("Existing known3 prepared source stat changed")
    basis=_basis_metadata(plan["load_basis"],split)
    from .deeponet_data import request_contract as cloud_contract
    bc_contract,_=cloud_contract(plan_path)
    sources={name:shared.source_sha(name) for name in SOURCES}
    if any(plan.get("source_sha256",{}).get(name,value)!=value for name,value in sources.items()):
        raise ValueError("A pinned paired-training source changed")
    science={k:plan[k] for k in ("recipe","existing_split","prepared_data","data_cache","bc_cloud_cache",
                               "load_basis","numerical_tolerances","access","development_dia")}
    contract={"schema":"geobc_deeponet_screen_v1","plan":str(shared.host_path(plan_path)),"mode":mode,
              "recipe_sha256":digest(science),"recipe":plan["recipe"],"bc_cloud_contract":bc_contract,
              "selection_manifest_sha256":shared.source_sha(manifest_path),"selection_contract":manifest["contract"],
              "scaler":manifest["scaler"],"case_order_sha256":manifest["case_order_sha256"],
              "train_keys":manifest["train_keys"],"validation_keys":manifest["validation_keys"],
              "training_readout_keys":manifest["training_readout_keys"],"basis_sha256":shared.source_sha(plan["load_basis"]),
              "source_sha256":sources}
    return contract,plan,manifest,basis


def request_contract(plan_path,mode):
    """Pure metadata API; the pending input cache is not prepared here."""
    contract,plan,_,_=_metadata(plan_path,mode)
    return contract,plan


def request_dia_contract(plan_path):
    contracts={}; runs={}; ready=True
    for mode in MODES:
        contract,plan,manifest,_=_metadata(plan_path,mode); contracts[mode]=contract
        run=shared.host_path(plan["runs"][mode]); item={"run":str(run),"ready":False}
        if (run/"summary.json").is_file() and (run/"config.json").is_file():
            summary=shared.read_json(run/"summary.json"); config=shared.read_json(run/"config.json")
            valid=(summary.get("status")=="completed" and summary.get("endpoint_step")==480
                   and summary.get("config")==config and config.get("contract")==contract
                   and summary.get("case_order_sha256")==manifest["case_order_sha256"]
                   and summary.get("costs",{}).get("train_case_exposures")==480
                   and summary.get("case_exposure_counts")=={key:4 for key in manifest["train_keys"]})
            checkpoint=run/"last.pt"
            item.update(ready=bool(valid and checkpoint.is_file()),summary_sha256=shared.source_sha(run/"summary.json"),
                        config_sha256=shared.source_sha(run/"config.json"),checkpoint=str(checkpoint),
                        checkpoint_stat=stat_record(checkpoint) if checkpoint.is_file() else None)
        runs[mode]=item; ready=ready and item["ready"]
    split=shared.read_json(plan["existing_split"])
    sources={cad+"/dia":{"path":str(shared.host_path(split["case_paths"][cad+"/dia"])),
                          **stat_record(split["case_paths"][cad+"/dia"])} for cad in split["groups"]["validation"]}
    return {"schema":"geobc_deeponet_development_dia_v1","training_contracts":contracts,"routes":runs,
            "all_two_endpoints_completed":ready,"protocol":plan["development_dia"],"sources":sources,
            "training_updates":0,"scaler_fits":0,"old_final_reads":0,"sealed_new_test_reads":0},plan


def source_check(plan_path):
    contracts={mode:request_contract(plan_path,mode)[0] for mode in MODES}
    dia,plan=request_dia_contract(plan_path)
    for name in ("geobc/deeponet_train.py","geobc/deeponet_field.py","geobc/deeponet_data.py"):
        ast.parse(shared.host_path(name).read_text(encoding="utf-8"),filename=name)
    first=contracts[MODES[0]]
    return {"status":"source_ready","source_sha256":first["source_sha256"],"recipe_sha256":first["recipe_sha256"],
            "contract_sha256_by_mode":{mode:digest(value) for mode,value in contracts.items()},
            "case_order_sha256":first["case_order_sha256"],"bc_cloud_contract_sha256":digest(first["bc_cloud_contract"]),
            "training_rules_frozen":plan.get("training_rules_frozen",False),"dia_rules_frozen":plan.get("dia_rules_frozen",False),
            "dia_endpoints_ready":dia["all_two_endpoints_completed"],"TRAIN_cases":120,"VAL_cases":15,"CADs":45,
            "scientific_arrays_read":0,"models_constructed":0,"CUDA_initialized":False,"training_updates":0}


def _unit_check(torch,np,model,case,components,prediction,manifest,basis,plan):
    tol=plan["numerical_tolerances"]; count=min(8,len(prediction)); cad,load=case["key"].split("/")
    alpha=components["alpha"].detach().cpu().numpy().astype(np.float64)[0]
    expected=np.asarray(next(row for row in basis["cases"] if row["cad_id"]==cad)["alpha_by_load"][load]["alpha"],np.float64)
    if not np.allclose(alpha,expected,atol=tol["alpha_atol"],rtol=tol["alpha_rtol"]): raise ValueError("Actual alpha differs from verified load basis")
    normalized=case["features"][0,7:13].detach().cpu().numpy().astype(np.float64)
    physical=normalized*np.asarray(basis["prepared_feature_mapping"]["physical_scale"],np.float64)
    reconstructed=np.asarray(basis["canonical_basis"]["physical_matrix_6x3"],np.float64)@alpha
    residual=[float(np.abs(physical[:3]-reconstructed[:3]).max()),float(np.abs(physical[3:]-reconstructed[3:]).max())]
    limits=[tol["force_atol_N"]+tol["physical_load_rtol"]*float(np.abs(physical[:3]).max()),
            tol["moment_atol_Nmm"]+tol["physical_load_rtol"]*float(np.abs(physical[3:]).max())]
    if any(value>limit for value,limit in zip(residual,limits)): raise ValueError("Force/moment units fail separate reconstruction bounds")
    vectors=components["basis_nodes"][:count].detach().cpu().numpy().astype(np.float64).reshape(count,3,3)
    zero_centered=np.einsum("k,nkj->nj",alpha,vectors)
    mean=np.asarray(manifest["scaler"]["mean"][:3],np.float64); std=np.asarray(manifest["scaler"]["std"][:3],np.float64)
    encoded=prediction[:count,:3].detach().cpu().numpy().astype(np.float64)
    if not np.allclose(encoded,zero_centered-mean/std,atol=tol["affine_atol"],rtol=tol["affine_rtol"]):
        raise ValueError("Displacement mean must be subtracted once after zero-centered basis combination")
    return {"passed":True,"case":case["key"],"witness_original_node_ids":list(range(count)),"alpha":alpha.tolist(),
            "expected_alpha":expected.tolist(),"basis_node_shape":list(components["basis_nodes"].shape),
            "force_reconstruction_residual_N":residual[0],"moment_reconstruction_residual_Nmm":residual[1],
            "force_tolerance_N":limits[0],"moment_tolerance_Nmm":limits[1],
            "affine_encoded_max_abs_error":float(np.abs(encoded-(zero_centered-mean/std)).max()),
            "physical_affine_max_error_mm":float(np.abs(mean+std*encoded-std*zero_centered).max()),
            "algebraic_zero_load_physical_mm":(mean+std*(-mean/std)).tolist(),"extra_forward_calls":0,
            "scope":"First actual complete-case update, reused first chunk components; no synthetic or GT superposition test"}


def _gradient_check(torch,model,leaf):
    record=shared._gradient_check(torch,model)
    groups={}
    for name,module in model.gradient_components().items():
        groups[name]=math.sqrt(sum(float(parameter.grad.detach().float().square().sum()) for parameter in module.parameters() if parameter.grad is not None))
    if any(not math.isfinite(value) or value<=0 for value in groups.values()):
        raise FloatingPointError("A required active encoder/trunk/basis/stress group has zero or nonfinite gradient")
    if leaf.grad is None or not bool(torch.isfinite(leaf.grad).all()) or float(leaf.grad.norm())<=0:
        raise FloatingPointError("Full-case context leaf gradient is zero, missing or nonfinite")
    return {**record,"component_L2_gradient_norms":groups,"context_leaf_L2_gradient_norm":float(leaf.grad.norm()),
            "context_policy":"Encode BC cloud once; node-weighted chunk gradients accumulate on leaf; original context.backward once",
            "zero_row_policy":"Inactive one-hot displacement basis rows may have zero gradient"}


def _predict(torch,np,model,case,chunk=32768):
    context=model.encode_geometry(case["cloud"],case["mask"]); outputs=[]
    for begin in range(0,len(case["xyz"]),chunk):
        end=begin+chunk
        outputs.append(model.forward_points(case["xyz"][begin:end],case["features"][begin:end],context,
                                            case["features"][0,7:13]).float().cpu().numpy())
    return np.concatenate(outputs)


def run(args):
    if args.source_check:
        result=source_check(args.plan); print(json.dumps(result,indent=2),flush=True); return result
    if args.evaluate_dia: return evaluate_dia(args.plan,args.output)
    if args.mode not in MODES or args.output is None: raise ValueError("Training requires --mode and --output")
    contract,plan,manifest,basis=_metadata(args.plan,args.mode)
    if plan.get("training_rules_frozen") is not True: raise ValueError("Root must freeze this training recipe before actual access")
    output=shared.host_path(args.output)
    if (output/"summary.json").is_file():
        previous=shared.read_json(output/"summary.json")
        if previous.get("status")=="completed":
            if previous["config"]["contract"]!=contract or shared.read_json(output/"config.json")!=previous["config"]: raise ValueError("Completed identity differs")
            return previous
    resume=(output/"last.pt" if args.resume=="last" else shared.host_path(args.resume)) if args.resume else None
    if resume and not resume.is_file(): raise FileNotFoundError(resume)
    if output.exists() and any(output.iterdir()) and resume is None: raise FileExistsError("Incomplete run requires explicit same-arm --resume")
    output.mkdir(parents=True,exist_ok=True); began=time.monotonic(); prior_elapsed=0.; completed_step=0
    costs={"train_seconds":0.,"inference_seconds":0.,"evaluation_seconds":0.,"bc_cloud_read_H2D_seconds":0.,
           "train_case_exposures":0,"train_node_exposures":0,"inference_case_exposures":0,"inference_node_exposures":0}
    costs["train_seconds_definition"]="Begin before normal selection load_case + BC22 cloud read/H2D; complete all-node chunk loss/backward, one BC encoder backward, Adam/sync"
    state={"kind":"deeponet_screen","mode":args.mode,"step":0,"total_steps":480,"evaluated_steps":[],"started_at_utc":shared.utc_now()}
    config={"contract":contract}
    def status(phase,**details):
        state.update(details,phase=phase,updated_at_utc=shared.utc_now(),elapsed_seconds=prior_elapsed+time.monotonic()-began,costs=dict(costs))
        eta=None
        if costs["train_case_exposures"]:
            eta=costs["train_seconds"]/costs["train_case_exposures"]*(480-completed_step)
            if state["evaluated_steps"]: eta+=costs["evaluation_seconds"]/len(state["evaluated_steps"])*(len(ENDPOINTS)-len(state["evaluated_steps"]))
        state["eta_seconds"]=0. if phase=="completed" else None if phase=="failed" else eta
        state["estimated_completion_utc"]=(datetime.now(timezone.utc)+timedelta(seconds=state["eta_seconds"])).isoformat() if state["eta_seconds"] is not None else None
        shared.write_json(output/"status.json",state)
    status("loading_fresh_model")
    try:
        import numpy as np
        import torch
        from .benchmark import FieldScaler
        from .solver_selection_data import load_case,read_cache_metadata as selection_metadata
        from .deeponet_data import read_cache_metadata,load_bc_cloud
        from .deeponet_field import DeepONetField
        bc=read_cache_metadata(args.plan,contract=contract["bc_cloud_contract"])
        if bc["contract"]!=contract["bc_cloud_contract"] or selection_metadata(plan)!=manifest: raise ValueError("Input metadata changed")
        costs["shared_bc_cloud_preparation_seconds"]=bc["preparation_seconds"]
        costs["shared_bc_cloud_preparation_policy"]="One common45CAD preparation job; not added to each arm train_seconds or double-counted across arms"
        device=torch.device("cuda"); random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
        torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
        model=DeepONetField(args.mode,manifest["scaler"],basis,seed=2026,device=device)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.,fused=True); scaler=FieldScaler(**manifest["scaler"])
        config={"contract":contract,"mode":args.mode,"method":"geometry_bc_response","scaler":manifest["scaler"],
                "model":model.source_config,"parameter_counts":model.parameter_counts,"case_order":manifest["case_order"],
                "case_order_sha256":manifest["case_order_sha256"],"train_keys":manifest["train_keys"],"validation_keys":manifest["validation_keys"],
                "training_readout_keys":manifest["training_readout_keys"],"precision":"float32/TF32/high; no autocast",
                "bc_cloud_preparation":{"preparation_seconds":bc.get("preparation_seconds"),"read_receipt":bc.get("read_receipt"),
                                        "role_coverage_by_CAD":{cad:row["role_coverage"] for cad,row in bc["records"].items()}},
                "initialization":"Fresh both arms: same common BC encoder constructor/seed; distinct solver heads/layout/counts, no whole-model initialization or capacity equality claim"}
        history=[]; best_score=math.inf; best_step=None; best_metrics=None; first_check=None
        exposures={key:0 for key in manifest["train_keys"]}; passes=[]; _,shuffle_rng=shared.case_order(manifest["train_keys"],2026)
        prior_peak={"peak_allocated_bytes":0,"peak_reserved_bytes":0}
        if resume:
            saved=torch.load(resume,map_location=device,weights_only=False)
            if saved["config"]!=config or saved["case_order_sha256"]!=manifest["case_order_sha256"]: raise ValueError("Resume science/mode/config/order differs")
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); shared._restore_rng(torch,np,saved["rng"])
            completed_step=saved["step"]
            if saved["next_case_position"]!=completed_step: raise ValueError("Resume schedule position differs")
            costs=saved["costs"]; history=saved["history"]; exposures=saved["case_exposure_counts"]; passes=saved["pass_costs"]
            best_score=saved["best_score"]; best_step=saved["best_step"]; best_metrics=saved["best_metrics"]
            first_check=saved["first_actual_check"]; prior_elapsed=saved["elapsed_seconds"]; prior_peak=saved["memory"]; shuffle_rng=saved["case_shuffle_rng_state"]
            state["evaluated_steps"]=[row["step"] for row in history]
            (output/"metrics.jsonl").write_text("".join(json.dumps(row,allow_nan=False)+"\n" for row in history),encoding="utf-8")
            if best_step is not None:
                selected=resume if best_step==completed_step else resume.parent/f"step{best_step}.pt"
                if not selected.is_file(): raise FileNotFoundError(selected)
                if selected.resolve()!=(output/"best.pt").resolve(): shutil.copy2(selected,output/"best.pt")
                shared.write_json(output/"metrics_best.json",best_metrics)
        shared.write_json(output/"config.json",config); torch.cuda.reset_peak_memory_stats(device)
        def memory(): return {key:max(value,prior_peak[key]) for key,value in shared._memory(torch,device).items()}
        def elapsed(): return prior_elapsed+time.monotonic()-began
        def payload():
            return {"model":model.state_dict(),"optimizer":optimizer.state_dict(),"config":config,"step":completed_step,
                    "next_case_position":completed_step,"case_order":manifest["case_order"],"case_order_sha256":manifest["case_order_sha256"],
                    "case_shuffle_rng_state":shuffle_rng,"rng":shared._rng_state(torch,np),"history":history,"costs":dict(costs),
                    "case_exposure_counts":dict(exposures),"pass_costs":passes,"best_score":best_score,"best_step":best_step,
                    "best_metrics":best_metrics,"first_actual_check":first_check,"elapsed_seconds":elapsed(),"memory":memory()}
        def load(key):
            case=load_case(manifest["records"][key],device,for_fno=False); stage=time.monotonic()
            case.update(load_bc_cloud(bc["records"][key.split("/")[0]],device)); shared._sync(torch,device)
            costs["bc_cloud_read_H2D_seconds"]+=time.monotonic()-stage
            if case["cloud"].shape!=(1,1024,22) or case["mask"].shape!=(1,1024): raise ValueError("Canonical BC22 cloud shapes differ")
            return case
        def evaluate(step,loss):
            nonlocal best_score,best_step,best_metrics
            model.eval(); stage=time.monotonic(); groups={}; status("evaluating",step=step)
            with torch.no_grad():
                for group,keys in (("train_readout",manifest["training_readout_keys"]),("validation",manifest["validation_keys"])):
                    per_case={}
                    for key in keys:
                        case=load(key); shared._sync(torch,device); start=time.monotonic()
                        encoded=_predict(torch,np,model,case,plan["recipe"]["eval_batch_size"]); shared._sync(torch,device); seconds=time.monotonic()-start
                        costs["inference_seconds"]+=seconds; costs["inference_case_exposures"]+=1; costs["inference_node_exposures"]+=len(case["xyz"])
                        prediction=scaler.decode(encoded); per_case[key]={**shared._physical_metrics(np,case["physical"],prediction),"inference_seconds":seconds}
                        del case,encoded,prediction
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
            status("training",step=step,best_step=best_step,memory=memory())
            print(f"{args.mode} {step}/480 VAL U={groups['validation']['case_mean_u_relative_l2']:.6g} stress={groups['validation']['case_mean_stress_relative_l2']:.6g}",flush=True)
            return metrics
        last=history[-1] if history else evaluate(0,None)
        for step in range(completed_step+1,481):
            key=manifest["case_order"][step-1]; start=time.monotonic(); case=load(key)
            model.train(); optimizer.zero_grad(set_to_none=True)
            n=len(case["xyz"]); load6=case["features"][0,7:13]
            if n<1 or case["target"].shape!=(n,4) or case["features"].shape!=(n,25): raise ValueError("Complete node/feature/target shapes differ")
            if not bool(torch.isfinite(case["features"]).all()) or not bool((case["features"][:,7:13]==load6).all()): raise ValueError("Original global force/moment6 must be finite and constant across nodes")
            context=model.encode_geometry(case["cloud"],case["mask"]); leaf=context.detach().requires_grad_(True)
            unit=None; loss_value=0.
            for begin in range(0,n,plan["recipe"]["microbatch_nodes"]):
                end=min(n,begin+plan["recipe"]["microbatch_nodes"])
                components=model.forward_components(case["xyz"][begin:end],case["features"][begin:end],leaf,load6)
                prediction=model.compose_encoded_nodes(components)
                if prediction.shape!=(end-begin,4): raise ValueError("Signed4 point output shape differs")
                if first_check is None and begin==0: unit=_unit_check(torch,np,model,case,components,prediction,manifest,basis,plan)
                loss=(prediction.float()-case["target"][begin:end]).square().mean()*((end-begin)/n)
                if not bool(torch.isfinite(loss)): raise FloatingPointError("Nonfinite complete-case chunk loss")
                loss.backward(); loss_value+=float(loss.detach()); del components,prediction,loss
            if leaf.grad is None or not bool(torch.isfinite(leaf.grad).all()): raise FloatingPointError("Complete-case context leaf has missing/nonfinite gradient")
            context.backward(leaf.grad)
            if first_check is None:
                first_check={"passed":True,"step":step,"case":key,"complete_nodes":n,"chunk_nodes":32768,
                             "geometry_encodes":1,"original_context_backwards":1,"gradient":_gradient_check(torch,model,leaf),"units_alpha_query":unit}
                shared.write_json(output/"first_actual_check.json",first_check)
            optimizer.step(); shared._sync(torch,device); costs["train_seconds"]+=time.monotonic()-start
            costs["train_case_exposures"]+=1; costs["train_node_exposures"]+=n; exposures[key]+=1; completed_step=step
            del case,context,leaf
            if step%120==0:
                previous=passes[-1]["cumulative_train_seconds"] if passes else 0.
                passes.append({"pass":step//120,"step":step,"cumulative_train_seconds":costs["train_seconds"],
                               "incremental_train_seconds":costs["train_seconds"]-previous,"mean_train_seconds_per_case":(costs["train_seconds"]-previous)/120})
            if step in ENDPOINTS: last=evaluate(step,loss_value)
            elif step%30==0 or step==1: shared._checkpoint(torch,output/"last.pt",payload())
            if step%10==0: status("training",step=step,case=key,training_loss=loss_value,memory=memory())
        if costs["train_case_exposures"]!=480 or any(value!=4 for value in exposures.values()): raise AssertionError("Four-pass full-case exposures differ")
        summary={"status":"completed","kind":"deeponet_screen","mode":args.mode,"endpoint_step":480,"config":config,
                 "elapsed_seconds":elapsed(),"initial_metrics":history[0],"final_metrics":last,"endpoint_metrics":history,
                 "best_step":best_step,"best_metrics":best_metrics,"costs":dict(costs),"pass_costs":passes,
                 "parameter_counts":model.parameter_counts,"memory":memory(),"case_exposure_counts":exposures,
                 "case_order_sha256":manifest["case_order_sha256"],"first_actual_check":first_check,"first_update_gradient_check":first_check,
                 "scope":"Fresh TRAIN40/DEV5 known3; common BC22 cloud32 encoder and u-factorization, distinct solver-head architectures/counts; no dia in training",
                 "completed_at_utc":shared.utc_now()}
        shared.write_json(output/"summary.json",summary); status("completed",step=480); return summary
    except Exception as error:
        message=f"{type(error).__name__}: {error}"; status("failed",step=completed_step,error=message)
        shared.write_json(output/"summary.json",{"status":"failed","kind":"deeponet_screen","mode":args.mode,"config":config,
                          "completed_step":completed_step,"error":message,"costs":dict(costs),"elapsed_seconds":prior_elapsed+time.monotonic()-began,
                          "last_checkpoint":str(output/"last.pt") if (output/"last.pt").is_file() else None})
        raise


def evaluate_dia(plan_path,output_path):
    if output_path is None: raise ValueError("DEV-dia readout requires --output")
    contract,plan=request_dia_contract(plan_path)
    if plan.get("dia_rules_frozen") is not True or plan.get("training_rules_frozen") is not True or not contract["all_two_endpoints_completed"]:
        raise ValueError("Both fixed480 endpoints and root-frozen DEV-dia rules are prerequisites before input/GT/weights")
    output=shared.host_path(output_path); config={"contract":contract,"kind":"deeponet_development_dia_eval"}
    if (output/"summary.json").is_file():
        previous=shared.read_json(output/"summary.json")
        if previous.get("status")=="completed":
            if previous["config"]!=config: raise ValueError("Completed DEV-dia contract differs")
            return previous
    output.mkdir(parents=True,exist_ok=True); shared.write_json(output/"config.json",config); began=time.monotonic()
    status={"phase":"loading_endpoints","kind":"deeponet_development_dia_eval","cases_completed":0}
    shared.write_json(output/"status.json",status)
    try:
        import numpy as np
        import torch
        from .benchmark import FieldScaler
        from .deeponet_field import DeepONetField
        from .deeponet_data import read_cache_metadata,load_bc_cloud
        bc=read_cache_metadata(plan_path,contract=contract["training_contracts"][MODES[0]]["bc_cloud_contract"])
        _,_,manifest,basis=_metadata(plan_path,MODES[0]); scaler=FieldScaler(**manifest["scaler"])
        device=torch.device("cuda"); torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
        models={}; origins={}
        for mode in MODES:
            route=contract["routes"][mode]; run=shared.host_path(route["run"]); runconfig=shared.read_json(run/"config.json"); summary=shared.read_json(run/"summary.json")
            checkpoint=shared.host_path(route["checkpoint"])
            if stat_record(checkpoint)!=route["checkpoint_stat"]: raise ValueError("Frozen endpoint stat changed")
            saved=torch.load(checkpoint,map_location=device,weights_only=False)
            if saved["step"]!=480 or saved["config"]!=runconfig: raise ValueError("Endpoint480 checkpoint config/step differs")
            model=DeepONetField(mode,manifest["scaler"],basis,seed=2026,device=device)
            if model.source_config!=runconfig["model"] or model.parameter_counts!=runconfig["parameter_counts"]: raise ValueError("Endpoint model source/counts changed")
            model.load_state_dict(saved["model"],strict=True); model.eval(); models[mode]=model
            origins[mode]={"checkpoint":str(checkpoint),"selected_step":480,"parameter_counts":model.parameter_counts,
                           "known3":summary["final_metrics"]["validation"],"training_costs":summary["costs"]}
            del saved
        torch.cuda.reset_peak_memory_stats(device); per_case={mode:{} for mode in MODES}; input_checks={}; inference={mode:0. for mode in MODES}
        for cad in plan["development_dia"]["CADs"]:
            key=cad+"/dia"; receipt=output/"cases"/(cad+".json")
            if receipt.is_file():
                previous=shared.read_json(receipt)
                if previous.get("contract_sha256")!=digest(contract) or previous.get("status")!="completed": raise ValueError("Committed DEV-dia case identity differs")
                for mode in MODES:
                    per_case[mode][key]=previous["metrics"][mode]; inference[mode]+=previous["inference_seconds"][mode]
                input_checks[key]=previous["input_check"]; continue
            source=shared.host_path(contract["sources"][key]["path"])
            with np.load(source,allow_pickle=False) as archive:
                xyz=np.array(archive["xyz"],dtype=np.float32); features=np.array(archive["features"],dtype=np.float32)
            record=manifest["records"][cad+"/ver"]
            canonical=np.load(shared.host_path(record["xyz_path"]),mmap_mode="r",allow_pickle=False)
            ref=np.load(shared.host_path(record["features_path"]),mmap_mode="r",allow_pickle=False)
            if xyz.shape!=(record["num_nodes"],3) or features.shape!=(len(xyz),25) or not np.array_equal(xyz,canonical) or not np.array_equal(features[:,NONFORCE],ref[:,NONFORCE]):
                raise ValueError("DEV-dia canonical original-node order/xyz/nonforce19 input differs")
            if not np.isfinite(xyz).all() or not np.isfinite(features).all() or not np.array_equal(features[:,7:13],np.broadcast_to(features[0,7:13],(len(xyz),6))):
                raise ValueError("DEV-dia input or global normalized force/moment6 is invalid")
            case={"key":key,"xyz":torch.as_tensor(xyz,device=device),"features":torch.as_tensor(features,device=device),**load_bc_cloud(bc["records"][cad],device)}
            predictions={}; times={}
            with torch.no_grad():
                for mode in MODES:
                    shared._sync(torch,device); start=time.monotonic(); encoded=_predict(torch,np,models[mode],case,plan["recipe"]["eval_batch_size"])
                    shared._sync(torch,device); times[mode]=time.monotonic()-start
                    if encoded.shape!=(len(xyz),4) or not np.isfinite(encoded).all(): raise FloatingPointError("DEV-dia complete prediction invalid")
                    predictions[mode]=scaler.decode(encoded)
                    if not np.isfinite(predictions[mode]).all(): raise FloatingPointError("DEV-dia physical decoded prediction invalid before GT")
            # Both complete predictions and physical decoding precede this first label access.
            with np.load(source,allow_pickle=False) as archive:
                truth=np.column_stack((archive["displacement"],archive["stress"])).astype(np.float64)
            if truth.shape!=(len(xyz),4) or not np.isfinite(truth).all(): raise ValueError("DEV-dia physical labels invalid after prediction")
            metrics={mode:{**shared._physical_metrics(np,truth,predictions[mode]),"inference_seconds":times[mode]} for mode in MODES}
            check={"xyz_and_original_node_order_exact":True,"nonforce19_exact":True,"same_BC22_cloud_record_for_both_routes":True,
                   "all_nodes":len(xyz),"both_predictions_before_GT":True,"normalized_load6":features[0,7:13].tolist(),
                   "label_members":["displacement_mm","signed_stress_MPa"],"source_npz":str(source)}
            shared.write_json(receipt,{"status":"completed","contract_sha256":digest(contract),"key":key,"metrics":metrics,
                                       "inference_seconds":times,"input_check":check})
            for mode in MODES: per_case[mode][key]=metrics[mode]; inference[mode]+=times[mode]
            input_checks[key]=check; status.update(phase="evaluating",cases_completed=len(input_checks),elapsed_seconds=time.monotonic()-began)
            shared.write_json(output/"status.json",status); del case,predictions,truth,encoded,xyz,features
        routes={mode:{**origins[mode],"development_dia":shared._aggregate(np,per_case[mode]),"inference_seconds":inference[mode]} for mode in MODES}
        summary={"status":"completed","kind":"deeponet_development_dia_eval","config":config,"routes":routes,
                 "input_checks":input_checks,"elapsed_seconds":time.monotonic()-began,"memory":shared._memory(torch,device),
                 "case_count":5,"route_case_count":10,"training_updates":0,"scaler_fits":0,
                 "scope":"Previously explored DEV5/dia fixed480 diagnostic; no final-test claim or checkpoint/seed selection"}
        shared.write_json(output/"summary.json",summary); status.update(phase="completed",cases_completed=5,elapsed_seconds=summary["elapsed_seconds"])
        shared.write_json(output/"status.json",status); return summary
    except Exception as error:
        message=f"{type(error).__name__}: {error}"; status.update(phase="failed",error=message,elapsed_seconds=time.monotonic()-began)
        shared.write_json(output/"status.json",status); shared.write_json(output/"summary.json",{"status":"failed","kind":"deeponet_development_dia_eval","config":config,"error":message,"elapsed_seconds":time.monotonic()-began})
        raise


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan",type=Path,required=True); result.add_argument("--mode",choices=MODES)
    result.add_argument("--output",type=Path); result.add_argument("--resume",nargs="?",const="last")
    result.add_argument("--source-check",action="store_true"); result.add_argument("--evaluate-dia",action="store_true")
    return result


if __name__=="__main__": run(parser().parse_args())
