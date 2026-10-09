"""Target-free continuation of the committed TRAIN101_479/ver PCG endpoint.

The original 2000-iteration result remains intact.  Only the true residual
decides convergence under a new, cumulative 10000-iteration/300-second limit.
The accepted input, curved-P2 operator, constraints and block preconditioner
are reused without rebuilding or repeating the eight operator guards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_elastic_convergence_plan.json"
SOURCE = "geobc/elastic_pilot_convergence.py"
KIND = "elastic_residual_continuation"


def host_path(value):
    text = str(value)
    if os.name == "nt" and text.startswith("/mnt/") and len(text)>7 and text[6] == "/":
        text = text[5].upper()+":"+text[6:]
    elif os.name != "nt" and len(text)>2 and text[1] == ":":
        text = "/mnt/"+text[0].lower()+"/"+text[2:].lstrip("/\\").replace("\\","/")
    path = Path(text)
    return path if path.is_absolute() else ROOT/path


def read_json(path):
    return json.loads(host_path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(host_path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def stat_record(path):
    info = host_path(path).stat()
    return {"bytes":info.st_size,"mtime_ns":info.st_mtime_ns}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def request_contract(plan_path=PLAN):
    """Small JSON/source/stat only; no vector, target, geometry or CUDA reads."""
    from . import elastic_pilot as base
    plan = read_json(plan_path)
    require(plan.get("kind") == KIND and plan.get("case") == "101_479/ver", "Fixed residual continuation scope differs")
    expected = {"maximum_iterations":10000,"maximum_cumulative_solve_wall_seconds":300,
        "true_relative_residual_tolerance":1e-8,"true_residual_interval":25,
        "checkpoint_interval":25,"resume_budget_reset":False}
    require(all(plan.get("recipe",{}).get(key) == value for key,value in expected.items()), "Continuation numerical policy differs")
    parent_contract,parent_plan = base.request_contract(plan["parent_plan"])
    operator = base._operator_metadata(parent_contract,parent_plan)
    parent_output = host_path(plan["parent_solve_output"])
    require(parent_output.resolve() == host_path(parent_plan["solve_output"]).resolve(), "Continuation parent output differs")
    parent_config = read_json(parent_output/"config.json")
    parent_summary = read_json(parent_output/"summary.json")
    endpoint = read_json(parent_output/"endpoint_before_GT.json")
    checkpoint = read_json(parent_output/"checkpoint.json")
    require(parent_config == {"contract":parent_contract,"stage":"elastic_pcg_solve",
        "operator_summary_sha256":sha(host_path(parent_plan["operator_output"])/"summary.json")}, "Parent config/source identity differs")
    require(parent_summary.get("status") == "completed" and parent_summary.get("kind") == "elastic_physical_solve"
        and parent_summary.get("config") == parent_config and endpoint.get("config") == parent_config
        and checkpoint.get("config") == parent_config, "Require one committed parent solve endpoint")
    require(parent_summary.get("numerical_state") == "iteration_budget_limit" and parent_summary.get("iteration") == 2000
        and checkpoint.get("iteration") == 2000 and checkpoint.get("iterations_executed_total") == 2000
        and endpoint.get("iteration") == 2000, "Only the original nonconverged 2000 endpoint is supported")
    require(stat_record(checkpoint["vectors_path"]) == checkpoint["vectors_stat"], "Parent PCG vector stat changed")
    journal = read_json(parent_output/"status.json")
    require(journal.get("config") == parent_config and journal.get("phase") == "completed", "Parent accounting journal differs")
    seconds = max(float(checkpoint["cumulative_solve_wall_seconds"]),float(journal.get("cumulative_solve_wall_seconds",0)),
        float(parent_summary["costs"]["cumulative_solve_wall_seconds"]))
    attempts = max(checkpoint["iterations_executed_total"],journal.get("iterations_executed_total",0),parent_summary["iterations_executed_total"])
    require(math.isfinite(seconds) and 0 <= seconds < 300 and attempts == 2000, "Parent accounting cannot be reset")
    sources = {name:sha(name) for name in (SOURCE,"geobc/elastic_pilot.py","geobc/elastic_pilot_data.py",
        "geobc/continuous_query_geometry.py")}
    for name,value in plan.get("source_sha256",{}).items():
        require(name in sources and sources[name] == value, "Continuation source pin differs: "+name)
    parent = {"output":str(parent_output),"config_sha256":sha(parent_output/"config.json"),
        "summary_sha256":sha(parent_output/"summary.json"),"endpoint_sha256":sha(parent_output/"endpoint_before_GT.json"),
        "checkpoint_sha256":sha(parent_output/"checkpoint.json"),"vectors_path":checkpoint["vectors_path"],
        "vectors_stat":checkpoint["vectors_stat"],"iteration":2000,"iterations_executed_total":attempts,
        "cumulative_solve_wall_seconds":seconds,"K_applications":max(checkpoint["K_applications"],journal.get("K_applications",0),
            parent_summary["costs"]["solve_K_count_cumulative"]),
        "true_relative_residual":parent_summary["true_relative_residual"],
        "rz_storage":"absent in parent; reconstruct rTMinvr once using unchanged block Cholesky; preserve z,r,p"}
    science = {"kind":KIND,"case":"101_479/ver","parent_plan":str(host_path(plan["parent_plan"])),
        "parent_contract":parent_contract,"parent":parent,"recipe":expected,"output":str(host_path(plan["output"])),
        "operator_summary_sha256":sha(host_path(parent_plan["operator_output"])/"summary.json"),
        "input_manifest_sha256":operator["input_manifest_sha256"],"source_sha256":sources,
        "label_access":False,"operator_rebuild":False,"repeated_operator_guards":0}
    return {"schema":"geobc_elastic_residual_continuation_v1",**science,"science_sha256":digest(science),
        "array_access_in_request_contract":0,"training_updates":0,"scaler_fits":0},plan


def solve(plan_path=PLAN, output_path=None):
    contract,plan = request_contract(plan_path)
    require(plan.get("scientific_rules_frozen") is True and plan.get("actual_execution_released") is True,
        "Root must freeze and release the residual-only stage before array imports")
    require(set(plan.get("source_sha256",{})) == set(contract["source_sha256"]), "Frozen continuation requires all four source pins")
    output = host_path(output_path or plan["output"])
    require(output.resolve() == host_path(plan["output"]).resolve() and output.resolve() != host_path(plan["parent_solve_output"]).resolve(),
        "Continuation output must be the frozen independent destination")
    config = {"contract":contract,"stage":KIND}
    if (output/"config.json").is_file():
        require(read_json(output/"config.json") == config,"Existing continuation config differs")
    if (output/"summary.json").is_file():
        previous = read_json(output/"summary.json")
        if previous.get("status") == "completed":
            require(previous.get("config") == config,"Completed continuation identity differs")
            return previous
    from . import elastic_pilot as base
    output.mkdir(parents=True,exist_ok=True); base.write_json(output/"config.json",config)
    began = time.monotonic()
    receipt = {"label_reads":0,"labels_read":0,"displacement_reads":0,"signed_displacement_member_reads":0,
        "stress_reads":0,"H5_reads":0,"raw_deck_reads":0,"model_or_weight_reads":0,
        "DEV_dia_reads":0,"old_final_reads":0,"sealed_new_test_reads":0,"optimizer_steps":0,"scaler_fits":0,
        "operator_rebuilds":0,"repeated_operator_guard_K_applications":0,
        "K_guards_repeated":0,"setup_repeated":0,"parent_state_reused":True,
        "input_cache_members_only":True,"stop_uses_true_residual_only":True}
    state = None; solve_began = None; base_seconds = None
    try:
        import numpy as np
        import torch
        from .elastic_pilot_data import read_cache_metadata,load_input
        parent_contract = contract["parent_contract"]; parent_plan = read_json(plan["parent_plan"])
        operator_record = base._operator_metadata(parent_contract,parent_plan)
        manifest = read_cache_metadata(plan["parent_plan"],contract=parent_contract["input_contract"])
        device = torch.device(parent_plan["recipe"]["device"])
        torch.cuda.init(); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        inputs = load_input(manifest,device); lam,mu = base._validate_inputs(torch,inputs,parent_plan)
        arrays = {name:base._load_array(np,torch,record,device) for name,record in operator_record["arrays"].items()}
        require(arrays["gradients"].shape == (len(inputs["cells10"]),4,10,3)
            and arrays["weights"].shape == (len(inputs["cells10"]),4)
            and arrays["free_cholesky"].shape == (len(inputs["free_node_ids"]),3,3)
            and all(value.dtype == torch.float64 for value in arrays.values()), "Committed operator dimensions/dtype differ")
        operator = base.MatrixFreeElastic(torch,inputs["cells10"],arrays["gradients"],arrays["weights"],len(inputs["vertices_mm"]),lam,mu)
        free,chol = inputs["free_node_ids"],arrays["free_cholesky"]
        b = inputs["force"][free]; norm_b = float(torch.linalg.vector_norm(b))
        if not (output/"checkpoint.json").exists():
            state = read_json(host_path(plan["parent_solve_output"])/"checkpoint.json")
            require(sha(host_path(plan["parent_solve_output"])/"checkpoint.json") == contract["parent"]["checkpoint_sha256"], "Parent checkpoint changed")
            require(stat_record(state["vectors_path"]) == state["vectors_stat"],"Parent vector stat changed before copy")
            target = output/"parent_pcg_state_002000.npz"; temporary = target.with_name(target.name+".tmp")
            shutil.copyfile(host_path(state["vectors_path"]),temporary); os.replace(temporary,target)
            require(target.stat().st_size == state["vectors_stat"]["bytes"],"Parent vector copy incomplete")
            state.update(config=config,vectors_path=str(target),vectors_stat=stat_record(target),
                cumulative_solve_wall_seconds=contract["parent"]["cumulative_solve_wall_seconds"],
                iterations_executed_total=contract["parent"]["iterations_executed_total"],K_applications=contract["parent"]["K_applications"],
                rz=None,parent_iteration=2000)
            base.write_json(output/"checkpoint.json",state)
        state = read_json(output/"checkpoint.json")
        require(state["config"] == config and stat_record(state["vectors_path"]) == state["vectors_stat"],"Continuation checkpoint identity differs")
        with np.load(host_path(state["vectors_path"]),allow_pickle=False) as archive:
            require(set(archive.files) == {"z","r","p"},"Unexpected PCG checkpoint member")
            z,r,p = (torch.as_tensor(np.array(archive[name]),device=device) for name in ("z","r","p"))
        require(all(value.shape == b.shape and value.dtype == torch.float64 and torch.isfinite(value).all() for value in (z,r,p)),"Malformed PCG state vectors")
        if (output/"status.json").exists():
            journal = read_json(output/"status.json"); require(journal.get("config") == config,"Continuation journal identity differs")
            for key in ("cumulative_solve_wall_seconds","iterations_executed_total","K_applications"):
                state[key] = max(state[key],journal.get(key,0))
        base._sync(torch); setup_seconds = time.monotonic()-began
        base_seconds,base_K = state["cumulative_solve_wall_seconds"],state["K_applications"]
        solve_began = time.monotonic(); preconditioner_seconds = 0.; checkpoint_seconds = 0.

        def elapsed():
            base._sync(torch)
            return base_seconds+time.monotonic()-solve_began

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

        def save():
            nonlocal state,checkpoint_seconds
            tick = time.monotonic(); account("solving")
            state = base._save_pcg(np,output,state,z,r,p)
            checkpoint_seconds += time.monotonic()-tick; account("solving")

        true_r = b-reduced_apply(z); true_relative = base._relative(torch,true_r,norm_b)
        if not torch.isfinite(true_r).all() or true_relative is None:
            raise base.ScientificNegative("Nonfinite/undefined resumed true residual")
        state["trace"].append({"iteration":state["iteration"],"true_relative_residual":true_relative,
            "cumulative_solve_wall_seconds":elapsed(),"reason":"continuation_or_resume_true_verification"})
        if state.get("rz") is None:
            state["rz"] = float((r*precondition(r)).sum())
        rz = state["rz"]
        numerical_state = "converged" if true_relative <= 1e-8 else "solving"
        if numerical_state == "solving" and (not math.isfinite(rz) or rz <= 0):
            raise base.ScientificNegative("Resumed rTMinvr is nonfinite/nonpositive")
        while numerical_state == "solving":
            if state["iterations_executed_total"] >= 10000 or elapsed() >= 300:
                numerical_state = "iteration_budget_limit" if state["iterations_executed_total"] >= 10000 else "wall_budget_limit"
                break
            state["iterations_executed_total"] += 1; account("solving")
            Ap = reduced_apply(p); pAp = float((p*Ap).sum())
            if not math.isfinite(pAp) or pAp <= 0 or not torch.isfinite(Ap).all():
                raise base.ScientificNegative("PCG pTAp is nonfinite/nonpositive; no shift or alternate solver")
            alpha = rz/pAp; z = z+alpha*p; r = r-alpha*Ap; state["iteration"] += 1
            if not torch.isfinite(z).all() or not torch.isfinite(r).all():
                raise base.ScientificNegative("Nonfinite PCG solution/recursive residual")
            recursive_relative = base._relative(torch,r,norm_b)
            if state["iteration"] % 25 == 0 or (recursive_relative is not None and recursive_relative <= 1e-8):
                true_r = b-reduced_apply(z); true_relative = base._relative(torch,true_r,norm_b)
                if not torch.isfinite(true_r).all() or true_relative is None:
                    raise base.ScientificNegative("Nonfinite/undefined true residual")
                state["trace"].append({"iteration":state["iteration"],"true_relative_residual":true_relative,
                    "recursive_relative_residual":recursive_relative,"cumulative_solve_wall_seconds":elapsed()})
                if true_relative <= 1e-8:
                    numerical_state = "converged"
            if numerical_state == "solving":
                preconditioned = precondition(r); next_rz = float((r*preconditioned).sum())
                if not math.isfinite(next_rz) or next_rz <= 0:
                    raise base.ScientificNegative("PCG rTMinvr is nonfinite/nonpositive")
                p = preconditioned+(next_rz/rz)*p; rz = next_rz; state["rz"] = rz
            account("solving")
            if state["iteration"] % 25 == 0:
                save()
        true_r = b-reduced_apply(z); true_relative = base._relative(torch,true_r,norm_b)
        if not torch.isfinite(true_r).all() or true_relative is None:
            raise base.ScientificNegative("Nonfinite/undefined endpoint true residual")
        if true_relative <= 1e-8:
            numerical_state = "converged"
        state["trace"].append({"iteration":state["iteration"],"true_relative_residual":true_relative,
            "cumulative_solve_wall_seconds":elapsed(),"reason":"endpoint_true_verification"})
        save(); cumulative_seconds = elapsed(); solve_K_count = operator.calls; solve_K_seconds = operator.apply_seconds
        verify_began = time.monotonic(); physical = base._restore_full(torch,z,inputs)
        Ku = operator.apply(physical); residual = Ku-inputs["force"]
        reaction = torch.zeros_like(residual); reaction[inputs["fixed_node_ids"]] = residual[inputs["fixed_node_ids"]]
        external = inputs["force"]+reaction; reference = inputs["reference_xyz_mm"]
        net_force = external.sum(dim=0); net_moment = torch.cross(inputs["vertices_mm"]-reference,external,dim=-1).sum(dim=0)
        if any(not torch.isfinite(value).all() for value in (physical,residual,net_force,net_moment)):
            raise base.ScientificNegative("Nonfinite residual-only physical endpoint/equilibrium")
        prediction_record = base._save_array(np,output/"physical_u_mm.npy",physical)
        reaction_record = base._save_array(np,output/"constraint_reaction_N.npy",reaction)
        grid_record = base._save_array(np,output/"source_grid_node_ids.npy",inputs["grid_node_ids"])
        equilibrium = {"true_free_residual_relative":base._relative(torch,residual[free],norm_b),
            "true_free_residual_norm_N":float(torch.linalg.vector_norm(residual[free])),
            "constraint_reaction_sum_N":reaction.sum(dim=0).tolist(),"net_force_with_reaction_N":net_force.tolist(),
            "net_moment_with_reaction_Nmm":net_moment.tolist(),
            "fixed_translation_max_abs_mm":float(physical[inputs["fixed_node_ids"]].abs().max()),
            "reference_xyz_mm":reference.tolist(),"full_K_verification_calls":1}
        base._sync(torch); verification_seconds = time.monotonic()-verify_began
        endpoint = {"config":config,"node_count":len(physical),"numerical_state":numerical_state,"iteration":state["iteration"],
            "true_relative_residual":true_relative,"prediction":prediction_record,"constraint_reaction":reaction_record,
            "physical_u_path":prediction_record["path"],"source_grid_node_ids":grid_record,"equilibrium":equilibrium,"read_receipt":receipt}
        base.write_json(output/"endpoint_no_GT.json",endpoint)
        summary = {"status":"completed","kind":KIND,**endpoint,"numerically_converged":numerical_state == "converged",
            "iterations_executed_total":state["iterations_executed_total"],"trace":state["trace"],
            "costs":{"parent_cumulative_solve_wall_seconds":contract["parent"]["cumulative_solve_wall_seconds"],
                "cumulative_solve_wall_seconds":cumulative_seconds,
                "continuation_cumulative_solve_wall_seconds":cumulative_seconds-contract["parent"]["cumulative_solve_wall_seconds"],
                "this_call_solve_wall_seconds":cumulative_seconds-base_seconds,"solve_setup_read_seconds":setup_seconds,
                "this_call_K_apply_seconds":solve_K_seconds,"preconditioner_seconds":preconditioner_seconds,
                "checkpoint_seconds":checkpoint_seconds,"endpoint_verification_seconds":verification_seconds,
                "solve_K_count_this_call":solve_K_count,"solve_K_count_cumulative":base_K+solve_K_count,
                "verification_K_count":1,"total_this_call_wall_seconds":time.monotonic()-began,
                "budget_overshoot_seconds":max(0.,cumulative_seconds-300)},
            "memory":base._memory(torch),"scope":"Residual-only continuation of one fixed TRAIN physical model; no accuracy/engineering/generalization claim"}
        base.write_json(output/"summary.json",summary)
        state["numerical_state"] = numerical_state
        account("completed")
        return summary
    except base.ScientificNegative as error:
        if state is not None and solve_began is not None:
            state.update(cumulative_solve_wall_seconds=elapsed(),K_applications=base_K+operator.calls)
            base.write_json(output/"status.json",{**state,"trace":None,"phase":"completed","numerical_state":"breakdown"})
        summary = {"status":"completed","kind":KIND,"config":config,"numerical_state":"breakdown",
            "node_count":179316,"numerically_converged":False,"reason":str(error),"elapsed_seconds":time.monotonic()-began,
            "read_receipt":receipt,"iteration":None if state is None else state["iteration"],
            "costs":{"cumulative_solve_wall_seconds":None if state is None else state["cumulative_solve_wall_seconds"]}}
        base.write_json(output/"summary.json",summary)
        return summary
    except Exception as error:
        if state is not None and solve_began is not None:
            state.update(cumulative_solve_wall_seconds=elapsed(),K_applications=base_K+operator.calls)
            base.write_json(output/"status.json",{**state,"trace":None,"phase":"failed"})
        base.write_json(output/"summary.json",{"status":"failed","kind":KIND,"node_count":179316,"config":config,
            "error":f"{type(error).__name__}: {error}","elapsed_seconds":time.monotonic()-began,"read_receipt":receipt})
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan",type=Path,default=Path(PLAN))
    result.add_argument("--output",type=Path)
    return result


def run(args):
    return solve(args.plan,args.output)


if __name__ == "__main__":
    run(parser().parse_args())
