"""One direct4 all-node multiscale mesh screen and fixed DEV5/dia readout.

Authors execute only AST and small metadata checks. Actual cache preparation,
model construction, labels, checkpoints and CUDA require root-frozen rules.
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

ROOT=shared.ROOT
ENDPOINTS=(0,120,240,480)
NONFORCE=list(range(7))+list(range(13,25))
SOURCES=("geobc/multiscale_mesh_train.py","geobc/multiscale_mesh_field.py","geobc/multiscale_mesh_data.py",
         "geobc/solver_selection_train.py","geobc/solver_selection_data.py","geobc/conditioned_field.py",
         "geobc/benchmark.py","tools/prepare_topology_context.py","tools/prepare_topology_context_balanced.py")


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def stat_record(path):
    item=shared.host_path(path).stat()
    return {"bytes":item.st_size,"mtime_ns":item.st_mtime_ns}


def _parameter_counts(model):
    value=model.parameter_counts
    return value() if callable(value) else value


def _rules(plan):
    fixed={"seed":2026,"cloud_seed":2026,"cloud_points":1024,"geometry_dim":32,"width":32,
           "cycles":4,"soft_nodes":[2048,128],"hard_maximum_policy":"current_level_node_count",
           "train_loads":["ver","hor","tor"],"steps":480,"passes":4,"eval_steps":list(ENDPOINTS),
           "readout_chunk_nodes":32768,"edge_chunk_size":65536,"optimizer":"Adam","learning_rate":.001,
           "weight_decay":0.,"fused":True,"stress_transform":"asinh","channels":[0,1,2,3],"grid_resolution":64,
           "node_lift":[60,64,32],"message_mlp":[68,64,32],"readout":[32,128,4],
           "initial_coarse_states":"h1=R01@h0; h2=R12@h1",
           "block_update":"dst=affine_LayerNorm32(dst+complete_weighted_aggregate)",
           "edge_direction":"dst_xyz-src_xyz; length in canonical normalized xyz; no second /L",
           "message_activation":"GELU","cycle_weights":"Nine distinct blocks; same block shared across four cycles",
           "training_readout_keys":["101_479/ver","107_320/hor","10_58/tor"]}
    if plan.get("kind")!="multiscale_mesh_screen" or any(plan["recipe"].get(key)!=value for key,value in fixed.items()):
        raise ValueError("Mesh plan differs from the frozen single-candidate direct4 protocol")
    if any(plan["access"].get(key) is not False for key in ("old_final","sealed_new_test","warm_start","scaler_fit","strain_reference","P2_supervision")):
        raise ValueError("Final/sealed fields, warmstart, strain targets and scaler fitting are outside this screen")
    dia=plan["development_dia"]
    if any(dia.get(key)!=value for key,value in {"load":"dia","checkpoint_step":480,"seed":2026,"all_nodes":True,"training_updates":0,"scaler_fits":0}.items()):
        raise ValueError("Only the completed fixed480 endpoint on existing DEV5/dia is declared")


def _metadata(plan_path):
    plan=shared.read_json(plan_path); _rules(plan)
    manifest_path=shared.host_path(plan["data_cache"])/"manifest.json"
    manifest=shared.read_json(manifest_path); shared._check_manifest(manifest)
    split=shared.read_json(plan["existing_split"])
    for group,keys in (("train",manifest["train_keys"]),("validation",manifest["validation_keys"])):
        if keys!=[f"{cad}/{load}" for cad in split["groups"][group] for load in ("ver","hor","tor")]:
            raise ValueError("Existing whole-CAD split or known3 order changed")
    if plan["development_dia"]["CADs"]!=split["groups"]["validation"]: raise ValueError("Dia CAD scope differs from the already explored DEV5")
    if manifest["contract"]["split_sha256"]!=shared.source_sha(plan["existing_split"]): raise ValueError("Original split identity changed")
    for row in manifest["contract"]["sources"]:
        if stat_record(row["path"])!={key:row[key] for key in ("bytes","mtime_ns")}: raise ValueError("Prepared known3 source stat changed")
    from .multiscale_mesh_data import request_contract as hierarchy_contract
    input_contract,_=hierarchy_contract(plan_path)
    sources={name:shared.source_sha(name) for name in SOURCES}
    if any(plan.get("source_sha256",{}).get(name,value)!=value for name,value in sources.items()): raise ValueError("Pinned mesh/source helper changed")
    science={key:plan[key] for key in ("recipe","existing_split","prepared_data","data_cache","gradient_manifest",
                                     "graph_context_manifest","mesh_input_cache","access","development_dia")}
    contract={"schema":"geobc_multiscale_mesh_screen_v1","plan":str(shared.host_path(plan_path)),
              "recipe_sha256":digest(science),"recipe":plan["recipe"],"mesh_input_contract":input_contract,
              "selection_manifest_sha256":shared.source_sha(manifest_path),"selection_contract":manifest["contract"],
              "scaler":manifest["scaler"],"case_order_sha256":manifest["case_order_sha256"],"train_keys":manifest["train_keys"],
              "validation_keys":manifest["validation_keys"],"training_readout_keys":manifest["training_readout_keys"],"source_sha256":sources}
    return contract,plan,manifest


def request_contract(plan_path):
    """(contract,plan), metadata only; never prepare the hierarchy implicitly."""
    contract,plan,_=_metadata(plan_path)
    return contract,plan


def _dia_metadata(contract,plan,manifest):
    run=shared.host_path(plan["training_run"]); route={"run":str(run),"ready":False}
    if (run/"summary.json").is_file() and (run/"config.json").is_file():
        summary=shared.read_json(run/"summary.json"); config=shared.read_json(run/"config.json"); checkpoint=run/"last.pt"
        valid=(summary.get("status")=="completed" and summary.get("endpoint_step")==480 and summary.get("config")==config
               and config.get("contract")==contract and summary.get("case_order_sha256")==manifest["case_order_sha256"]
               and summary.get("costs",{}).get("train_case_exposures")==480
               and summary.get("case_exposure_counts")=={key:4 for key in manifest["train_keys"]})
        route.update(ready=bool(valid and checkpoint.is_file()),checkpoint=str(checkpoint),checkpoint_stat=stat_record(checkpoint) if checkpoint.is_file() else None,
                     summary_sha256=shared.source_sha(run/"summary.json"),config_sha256=shared.source_sha(run/"config.json"))
    split=shared.read_json(plan["existing_split"])
    sources={cad+"/dia":{"path":str(shared.host_path(split["case_paths"][cad+"/dia"])),**stat_record(split["case_paths"][cad+"/dia"])} for cad in split["groups"]["validation"]}
    return {"schema":"geobc_multiscale_mesh_development_dia_v1","training_contract":contract,"route":route,
            "endpoint_completed":route["ready"],"protocol":plan["development_dia"],"sources":sources,
            "training_updates":0,"scaler_fits":0,"old_final_reads":0,"sealed_new_test_reads":0}


def request_dia_contract(plan_path):
    contract,plan,manifest=_metadata(plan_path)
    return _dia_metadata(contract,plan,manifest),plan


def source_check(plan_path):
    contract,plan,manifest=_metadata(plan_path); dia=_dia_metadata(contract,plan,manifest)
    for name in ("geobc/multiscale_mesh_train.py","geobc/multiscale_mesh_field.py","geobc/multiscale_mesh_data.py"):
        ast.parse(shared.host_path(name).read_text(encoding="utf-8"),filename=name)
    return {"status":"source_ready","source_sha256":contract["source_sha256"],"recipe_sha256":contract["recipe_sha256"],
            "contract_sha256":digest(contract),"case_order_sha256":contract["case_order_sha256"],
            "mesh_input_contract_sha256":digest(contract["mesh_input_contract"]),"training_rules_frozen":plan.get("training_rules_frozen",False),
            "dia_rules_frozen":plan.get("dia_rules_frozen",False),"dia_endpoint_ready":dia["endpoint_completed"],
            "TRAIN_cases":120,"VAL_cases":15,"CADs":45,"scientific_arrays_read":0,"models_constructed":0,"CUDA_initialized":False,"training_updates":0}


def _gradient_check(torch,model,leaf):
    named=[(name,parameter) for name,parameter in model.named_parameters() if parameter.requires_grad]
    missing=[name for name,parameter in named if parameter.grad is None]
    finite=all(bool(torch.isfinite(parameter.grad).all()) for _,parameter in named if parameter.grad is not None)
    total=math.sqrt(sum(float(parameter.grad.detach().float().square().sum()) for _,parameter in named if parameter.grad is not None))
    if missing or not finite or not math.isfinite(total) or total<=0: raise FloatingPointError("First actual fullmesh backward has missing/nonfinite/zero aggregate parameter gradients")
    groups={}
    for name,module in model.gradient_components().items():
        groups[name]=math.sqrt(sum(float(parameter.grad.detach().float().square().sum()) for parameter in module.parameters() if parameter.grad is not None))
    if any(not math.isfinite(value) or value<=0 for value in groups.values()): raise FloatingPointError("An active mesh encoder/lift/message/readout gradient group is zero or nonfinite")
    if leaf.grad is None or not bool(torch.isfinite(leaf.grad).all()) or float(leaf.grad.norm())<=0: raise FloatingPointError("Complete H0 leaf gradient is missing, nonfinite or zero")
    return {"passed":True,"all_present_gradients_finite":finite,"trainable_tensors":len(named),"gradient_tensors":len(named)-len(missing),
            "missing_gradient_names":missing,"total_L2_gradient_norm":total,"component_L2_gradient_norms":groups,"H0_leaf_L2_gradient_norm":float(leaf.grad.norm()),
            "chain_rule":"Mesh solve once; all-node weighted readout chunk gradients accumulate on leaf; original H0.backward once"}


def _encode(model,case,hierarchy):
    if case["xyz"].shape!=hierarchy["xyz0"].shape or not bool((case["xyz"]==hierarchy["xyz0"]).all()):
        raise ValueError("Canonical full-node xyz/order differs from cached mesh source")
    return model.encode_case(case["xyz"],case["features"],case["cloud"],case["mask"],hierarchy)


def _predict(torch,np,model,case,hierarchy,chunk):
    H0=_encode(model,case,hierarchy)
    if H0.shape!=(len(case["xyz"]),32): raise ValueError("Complete solved fine state shape differs")
    pieces=[model.readout(H0[begin:begin+chunk]).float().cpu().numpy() for begin in range(0,len(H0),chunk)]
    return np.concatenate(pieces)


def run(args):
    if args.source_check:
        result=source_check(args.plan); print(json.dumps(result,indent=2),flush=True); return result
    if args.evaluate_dia: return evaluate_dia(args.plan,args.output)
    if args.output is None: raise ValueError("Actual training requires --output")
    contract,plan,manifest=_metadata(args.plan)
    if plan.get("training_rules_frozen") is not True: raise ValueError("Root must freeze the recipe before arrays/models/weights/CUDA")
    output=shared.host_path(args.output)
    if (output/"summary.json").is_file():
        previous=shared.read_json(output/"summary.json")
        if previous.get("status")=="completed":
            if previous["config"]["contract"]!=contract or shared.read_json(output/"config.json")!=previous["config"]: raise ValueError("Completed mesh identity differs")
            return previous
    resume=(output/"last.pt" if args.resume=="last" else shared.host_path(args.resume)) if args.resume else None
    if resume and not resume.is_file(): raise FileNotFoundError(resume)
    if output.exists() and any(output.iterdir()) and resume is None: raise FileExistsError("Incomplete mesh run requires explicit same-science --resume")
    output.mkdir(parents=True,exist_ok=True); began=time.monotonic(); prior_elapsed=0.; completed_step=0
    costs={"train_seconds":0.,"inference_seconds":0.,"evaluation_seconds":0.,"hierarchy_read_H2D_seconds":0.,
           "train_case_exposures":0,"train_node_exposures":0,"inference_case_exposures":0,"inference_node_exposures":0}
    costs["train_seconds_definition"]="Normal case/target/cloud plus hierarchy cache read/H2D, full-mesh solve, all-node readout/loss/backward, Adam/sync; static preparation separately reported"
    state={"kind":"multiscale_mesh_screen","step":0,"total_steps":480,"evaluated_steps":[],"started_at_utc":shared.utc_now()}; config={"contract":contract}
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
        from .multiscale_mesh_data import read_cache_metadata,load_hierarchy
        from .multiscale_mesh_field import MultiscaleMeshField
        inputs=read_cache_metadata(args.plan,contract=contract["mesh_input_contract"])
        if inputs["contract"]!=contract["mesh_input_contract"] or selection_metadata(plan)!=manifest: raise ValueError("Static mesh or original cache contract changed")
        costs["shared_mesh_preparation_seconds"]=inputs["preparation_seconds"]
        costs["shared_mesh_preparation_policy"]="One target-free45CAD job; separate from comparable model train_seconds, not counted twice"
        device=torch.device("cuda"); random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
        torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
        model=MultiscaleMeshField(manifest["scaler"],seed=2026,device=device)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.,fused=True); scaler=FieldScaler(**manifest["scaler"])
        config={"contract":contract,"method":"multiscale_mesh","scaler":manifest["scaler"],"model":model.source_config,
                "parameter_counts":_parameter_counts(model),"case_order":manifest["case_order"],"case_order_sha256":manifest["case_order_sha256"],
                "train_keys":manifest["train_keys"],"validation_keys":manifest["validation_keys"],"training_readout_keys":manifest["training_readout_keys"],
                "precision":"float32/TF32/high; no autocast; float64 physical decode/metrics",
                "mesh_preparation":{"preparation_seconds":inputs["preparation_seconds"],"costs":inputs["costs"],
                                    "total_cache_bytes":inputs["total_cache_bytes"],"reused_CADs":inputs["reused_CADs"],"receipt":inputs["read_receipt"],
                                    "per_CAD":{cad:{key:row[key] for key in ("facts","file_identity","legacy_reused","accessed_members","labels_read")} for cad,row in inputs["records"].items()}},
                "initialization":"One fresh seed2026 candidate; no old checkpoint or capacity equality claim"}
        history=[]; best_score=math.inf; best_step=None; best_metrics=None; first_check=None; exposures={key:0 for key in manifest["train_keys"]}; passes=[]
        _,shuffle_rng=shared.case_order(manifest["train_keys"],2026); prior_peak={"peak_allocated_bytes":0,"peak_reserved_bytes":0}
        if resume:
            saved=torch.load(resume,map_location=device,weights_only=False)
            if saved["config"]!=config or saved["case_order_sha256"]!=manifest["case_order_sha256"]: raise ValueError("Resume model/input/scaler/order differs")
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); shared._restore_rng(torch,np,saved["rng"])
            completed_step=saved["step"]
            if saved["next_case_position"]!=completed_step: raise ValueError("Resume schedule position differs")
            costs=saved["costs"]; history=saved["history"]; exposures=saved["case_exposure_counts"]; passes=saved["pass_costs"]
            best_score=saved["best_score"]; best_step=saved["best_step"]; best_metrics=saved["best_metrics"]; first_check=saved["first_actual_check"]
            prior_elapsed=saved["elapsed_seconds"]; prior_peak=saved["memory"]; shuffle_rng=saved["case_shuffle_rng_state"]
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
                    "case_exposure_counts":dict(exposures),"pass_costs":passes,"best_score":best_score,"best_step":best_step,"best_metrics":best_metrics,
                    "first_actual_check":first_check,"elapsed_seconds":elapsed(),"memory":memory()}
        def load(key):
            case=load_case(manifest["records"][key],device,for_fno=False); stage=time.monotonic()
            hierarchy=load_hierarchy(inputs["records"][key.split("/")[0]],device); shared._sync(torch,device)
            costs["hierarchy_read_H2D_seconds"]+=time.monotonic()-stage
            return case,hierarchy
        def evaluate(step,loss):
            nonlocal best_score,best_step,best_metrics
            model.eval(); stage=time.monotonic(); groups={}; status("evaluating",step=step)
            with torch.no_grad():
                for group,keys in (("train_readout",manifest["training_readout_keys"]),("validation",manifest["validation_keys"])):
                    per_case={}
                    for key in keys:
                        case,hierarchy=load(key); shared._sync(torch,device); start=time.monotonic()
                        encoded=_predict(torch,np,model,case,hierarchy,plan["recipe"]["readout_chunk_nodes"]); shared._sync(torch,device); seconds=time.monotonic()-start
                        costs["inference_seconds"]+=seconds; costs["inference_case_exposures"]+=1; costs["inference_node_exposures"]+=len(case["xyz"])
                        prediction=scaler.decode(encoded); per_case[key]={**shared._physical_metrics(np,case["physical"],prediction),"inference_seconds":seconds}
                        del case,hierarchy,encoded,prediction
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
            print(f"multiscale_mesh {step}/480 VAL U={groups['validation']['case_mean_u_relative_l2']:.6g} stress={groups['validation']['case_mean_stress_relative_l2']:.6g}",flush=True)
            return metrics
        last=history[-1] if history else evaluate(0,None)
        for step in range(completed_step+1,481):
            key=manifest["case_order"][step-1]; start=time.monotonic(); case,hierarchy=load(key)
            model.train(); optimizer.zero_grad(set_to_none=True); n=len(case["xyz"])
            if n<1 or case["target"].shape!=(n,4) or case["features"].shape!=(n,25): raise ValueError("Whole-case original node/target shape differs")
            H0=_encode(model,case,hierarchy)
            if H0.shape!=(n,32) or not bool(torch.isfinite(H0).all()): raise FloatingPointError("Full fine state shape/nonfinite failure")
            leaf=H0.detach().requires_grad_(True); loss_value=0.
            for begin in range(0,n,plan["recipe"]["readout_chunk_nodes"]):
                end=min(n,begin+plan["recipe"]["readout_chunk_nodes"]); prediction=model.readout(leaf[begin:end])
                if prediction.shape!=(end-begin,4): raise ValueError("direct4 readout shape differs")
                loss=(prediction.float()-case["target"][begin:end]).square().mean()*((end-begin)/n)
                if not bool(torch.isfinite(loss)): raise FloatingPointError("Nonfinite complete-case direct4 loss")
                loss.backward(); loss_value+=float(loss.detach()); del prediction,loss
            if leaf.grad is None or not bool(torch.isfinite(leaf.grad).all()): raise FloatingPointError("All-node H0 leaf gradient missing/nonfinite")
            H0.backward(leaf.grad)
            if first_check is None:
                first_check={"passed":True,"step":step,"case":key,"complete_nodes":n,"H0_shape":list(H0.shape),"encoded_output_channels":4,
                             "readout_chunk_nodes":32768,"mesh_solves":1,"original_H0_backwards":1,
                             "optimizer_updates_before_check":0,"optimizer_steps_per_complete_case":1,
                             "gradient":_gradient_check(torch,model,leaf),"parameter_counts":_parameter_counts(model),
                             "hierarchy_node_counts":[len(hierarchy["xyz0"]),len(hierarchy["xyz1"]),len(hierarchy["xyz2"])],
                             "hierarchy_metadata":config["mesh_preparation"]["per_CAD"][key.split("/")[0]],"memory":memory(),
                             "scope":"First true complete-case direct4 update; no extra model/geometry probe or strain/alpha supervision"}
                shared.write_json(output/"first_actual_check.json",first_check)
            optimizer.step(); shared._sync(torch,device); costs["train_seconds"]+=time.monotonic()-start
            costs["train_case_exposures"]+=1; costs["train_node_exposures"]+=n; exposures[key]+=1; completed_step=step
            del case,hierarchy,H0,leaf
            if step%120==0:
                previous=passes[-1]["cumulative_train_seconds"] if passes else 0.
                passes.append({"pass":step//120,"step":step,"cumulative_train_seconds":costs["train_seconds"],
                               "incremental_train_seconds":costs["train_seconds"]-previous,"mean_train_seconds_per_case":(costs["train_seconds"]-previous)/120})
            if step in ENDPOINTS: last=evaluate(step,loss_value)
            elif step%30==0 or step==1: shared._checkpoint(torch,output/"last.pt",payload())
            if step%10==0: status("training",step=step,case=key,training_loss=loss_value,memory=memory())
        if costs["train_case_exposures"]!=480 or any(value!=4 for value in exposures.values()): raise AssertionError("Four-pass complete-case exposure contract failed")
        summary={"status":"completed","kind":"multiscale_mesh_screen","endpoint_step":480,"config":config,"elapsed_seconds":elapsed(),
                 "initial_metrics":history[0],"final_metrics":last,"endpoint_metrics":history,"best_step":best_step,"best_metrics":best_metrics,
                 "costs":dict(costs),"pass_costs":passes,"parameter_counts":_parameter_counts(model),"memory":memory(),
                 "case_exposure_counts":exposures,"case_order_sha256":manifest["case_order_sha256"],"first_actual_check":first_check,
                 "first_update_gradient_check":first_check,"scope":"Single fresh all-node direct4 mesh candidate on TRAIN40/DEV5 known3; no final/sealed/strain target/P2loss",
                 "completed_at_utc":shared.utc_now()}
        shared.write_json(output/"summary.json",summary); status("completed",step=480); return summary
    except Exception as error:
        message=f"{type(error).__name__}: {error}"; status("failed",step=completed_step,error=message)
        shared.write_json(output/"summary.json",{"status":"failed","kind":"multiscale_mesh_screen","config":config,"completed_step":completed_step,
                          "error":message,"costs":dict(costs),"elapsed_seconds":prior_elapsed+time.monotonic()-began,
                          "last_checkpoint":str(output/"last.pt") if (output/"last.pt").is_file() else None,
                          "no_fallback":"No node/edge/role truncation, hidden width/chunk/radius change, budget expansion or alternate solver"})
        raise


def evaluate_dia(plan_path,output_path):
    if output_path is None: raise ValueError("DEV-dia requires --output")
    contract,plan=request_dia_contract(plan_path)
    if plan.get("training_rules_frozen") is not True or plan.get("dia_rules_frozen") is not True or not contract["endpoint_completed"]:
        raise ValueError("Root-frozen DEV-dia and completed endpoint480 are prerequisites before arrays/weights/GT")
    output=shared.host_path(output_path); config={"contract":contract,"kind":"multiscale_mesh_development_dia_eval"}
    if (output/"summary.json").is_file():
        previous=shared.read_json(output/"summary.json")
        if previous.get("status")=="completed":
            if previous["config"]!=config: raise ValueError("Completed DEV-dia identity differs")
            return previous
    output.mkdir(parents=True,exist_ok=True); shared.write_json(output/"config.json",config); began=time.monotonic()
    status={"phase":"loading_endpoint","kind":"multiscale_mesh_development_dia_eval","cases_completed":0}; shared.write_json(output/"status.json",status)
    try:
        import numpy as np
        import torch
        from .benchmark import FieldScaler
        from .multiscale_mesh_field import MultiscaleMeshField
        from .multiscale_mesh_data import read_cache_metadata,load_hierarchy
        inputs=read_cache_metadata(plan_path,contract=contract["training_contract"]["mesh_input_contract"])
        _,_,manifest=_metadata(plan_path); scaler=FieldScaler(**manifest["scaler"])
        device=torch.device("cuda"); torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
        route=contract["route"]; run=shared.host_path(route["run"]); runconfig=shared.read_json(run/"config.json"); parent=shared.read_json(run/"summary.json")
        checkpoint=shared.host_path(route["checkpoint"])
        if stat_record(checkpoint)!=route["checkpoint_stat"]: raise ValueError("Completed endpoint checkpoint stat changed")
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        if saved["step"]!=480 or saved["config"]!=runconfig: raise ValueError("Endpoint480 checkpoint config/step differs")
        model=MultiscaleMeshField(manifest["scaler"],seed=2026,device=device)
        if model.source_config!=runconfig["model"] or _parameter_counts(model)!=runconfig["parameter_counts"]: raise ValueError("Endpoint model source/counts differ")
        model.load_state_dict(saved["model"],strict=True); model.eval(); del saved
        torch.cuda.reset_peak_memory_stats(device); per_case={}; input_checks={}; inference_seconds=0.
        for cad in plan["development_dia"]["CADs"]:
            key=cad+"/dia"; receipt=output/"cases"/(cad+".json")
            if receipt.is_file():
                previous=shared.read_json(receipt)
                if previous.get("status")!="completed" or previous.get("contract_sha256")!=digest(contract): raise ValueError("Committed DEV-dia case identity differs")
                per_case[key]=previous["metrics"]; input_checks[key]=previous["input_check"]; inference_seconds+=previous["inference_seconds"]; continue
            source=shared.host_path(contract["sources"][key]["path"])
            with np.load(source,allow_pickle=False) as archive:
                xyz=np.array(archive["xyz"],dtype=np.float32); features=np.array(archive["features"],dtype=np.float32)
            record=manifest["records"][cad+"/ver"]; canonical=np.load(shared.host_path(record["xyz_path"]),mmap_mode="r",allow_pickle=False)
            reference=np.load(shared.host_path(record["features_path"]),mmap_mode="r",allow_pickle=False)
            if xyz.shape!=(record["num_nodes"],3) or features.shape!=(len(xyz),25) or not np.array_equal(xyz,canonical) or not np.array_equal(features[:,NONFORCE],reference[:,NONFORCE]):
                raise ValueError("DEV-dia original node order/xyz/nonforce19 differs from canonical mesh")
            if not np.isfinite(xyz).all() or not np.isfinite(features).all(): raise ValueError("DEV-dia nonfinite input")
            case={"key":key,"xyz":torch.as_tensor(xyz,device=device),"features":torch.as_tensor(features,device=device),
                  "cloud":torch.as_tensor(np.array(np.load(shared.host_path(record["cloud_path"]),allow_pickle=False)),device=device)[None],
                  "mask":torch.as_tensor(np.array(np.load(shared.host_path(record["mask_path"]),allow_pickle=False)),device=device)[None]}
            hierarchy=load_hierarchy(inputs["records"][cad],device); shared._sync(torch,device); start=time.monotonic()
            with torch.no_grad(): encoded=_predict(torch,np,model,case,hierarchy,plan["recipe"]["readout_chunk_nodes"])
            shared._sync(torch,device); seconds=time.monotonic()-start
            if encoded.shape!=(len(xyz),4) or not np.isfinite(encoded).all(): raise FloatingPointError("Complete DEV-dia encoded prediction invalid")
            prediction=scaler.decode(encoded)
            if not np.isfinite(prediction).all(): raise FloatingPointError("DEV-dia physical prediction invalid before GT")
            # First label access is after the sole route's entire prediction and physical decode.
            with np.load(source,allow_pickle=False) as archive: truth=np.column_stack((archive["displacement"],archive["stress"])).astype(np.float64)
            metrics={**shared._physical_metrics(np,truth,prediction),"inference_seconds":seconds}
            check={"xyz_and_original_node_order_exact":True,"nonforce19_exact":True,"canonical_hierarchy_and_cloud_reused":True,
                   "all_nodes":len(xyz),"prediction_and_decode_before_GT":True,"source_npz":str(source)}
            shared.write_json(receipt,{"status":"completed","contract_sha256":digest(contract),"key":key,"metrics":metrics,"input_check":check,"inference_seconds":seconds})
            per_case[key]=metrics; input_checks[key]=check; inference_seconds+=seconds
            status.update(phase="evaluating",cases_completed=len(per_case),elapsed_seconds=time.monotonic()-began); shared.write_json(output/"status.json",status)
            del case,hierarchy,xyz,features,encoded,prediction,truth
        route_metrics={"selected_step":480,"checkpoint":str(checkpoint),"parameter_counts":_parameter_counts(model),
                       "known3_reused":parent["final_metrics"]["validation"],"development_dia":shared._aggregate(np,per_case),
                       "training_costs":parent["costs"],"inference_seconds":inference_seconds}
        summary={"status":"completed","kind":"multiscale_mesh_development_dia_eval","config":config,"routes":{"mesh":route_metrics},
                 "selected_step":480,"parameter_counts":_parameter_counts(model),"inference_seconds":inference_seconds,
                 "input_checks":input_checks,"elapsed_seconds":time.monotonic()-began,"memory":shared._memory(torch,device),"case_count":5,
                 "training_updates":0,"scaler_fits":0,"scope":"Previously explored DEV5/dia fixed480 diagnostic; no final claim or checkpoint/seed/budget selection"}
        shared.write_json(output/"summary.json",summary); status.update(phase="completed",cases_completed=5,elapsed_seconds=summary["elapsed_seconds"]); shared.write_json(output/"status.json",status)
        return summary
    except Exception as error:
        message=f"{type(error).__name__}: {error}"; status.update(phase="failed",error=message,elapsed_seconds=time.monotonic()-began)
        shared.write_json(output/"status.json",status); shared.write_json(output/"summary.json",{"status":"failed","kind":"multiscale_mesh_development_dia_eval","config":config,"error":message,"elapsed_seconds":time.monotonic()-began})
        raise


def parser():
    result=argparse.ArgumentParser(description=__doc__); result.add_argument("--plan",type=Path,required=True); result.add_argument("--output",type=Path)
    result.add_argument("--resume",nargs="?",const="last"); result.add_argument("--source-check",action="store_true"); result.add_argument("--evaluate-dia",action="store_true")
    return result


if __name__=="__main__": run(parser().parse_args())
