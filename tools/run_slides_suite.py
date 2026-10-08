"""Serial GPU coordinator for the original-DeepJEB slide reproduction.

CPU data fetching/gridding can run in parallel. A single advisory lock prevents
duplicate coordinators. Existing completed experiments are reused, not replaced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from geobc.train import write_json, utc_now

OUT=Path("runs/slides_reproduction")
EXACT=Path("data/prepared_exact_multifield")
FIVE=Path("data/prepared_benchmark5")
KEYS=["506_533/ver","506_25/tor"]
FIVE_KEYS=KEYS+["10_30/ver","10_421/hor","10_460/dia"]
GRID64=Path("data/grids/506_533_ver_r64_joint.npz")
GRID192=Path("data/grids/506_533_ver_r192_joint.npz")


def plan():
    jobs=[]
    def add(name,module,arguments,requires,slides,assumptions=()):
        command=[sys.executable,"-m",module,*map(str,arguments),"--output",str(OUT/name)]
        if assumptions:command.extend(["--assumptions",*assumptions])
        jobs.append({"name":name,"command":command,"requires":[str(x) for x in requires],"slides":slides,"assumptions":list(assumptions),"output":str(OUT/name)})
    def field(name,method,keys,outputs,duration,extra=(),slides=(),assumptions=()):
        data=FIVE if len(keys)==5 else EXACT
        add(name,"geobc.benchmark",["--method",method,"--data",data,"--case-keys",*keys,"--outputs",outputs,
            *duration,"--eval-every","100",*extra],
            [data/(k.replace("/","_")+".npz") for k in keys],slides,assumptions)
    field("mlp_single_exact","mlp",KEYS[:1],"both",["--epochs","500"],["--input-mode","xyz"],[8],
          ["MLP_learning_rate_absent_using_0.003"])
    for key in KEYS:
        cad,load=key.split("/")
        for outputs in ("stress","u"):
            field(f"ngp4_{cad}_{load}_{outputs}","ngp4",[key],outputs,["--steps","1000"],
                  ["--input-mode","xyz","--protocol","sampled"],[6,9],
                  ["4_levels_known_but_hash_cap_base_growth_MLP_width_unrecoverable_using_logged_explicit_assumptions",
                   "sampling_count_and_learning_rate_not_given_using_16384_and_0.003"])
    for name,keys in (("hash_single_exact",KEYS[:1]),("hash_two_exact",KEYS),("hash_five_aligned",FIVE_KEYS)):
        field(name,"hashgrid",keys,"stress",["--steps","1000"],
              ["--input-mode","xyz" if len(keys)==1 else "both","--protocol","sampled","--stress-transform","asinh","--selection","worst-stress"],[15,16],
              ["asinh_constants_and_LR_absent_using_P90_scaled_centered_asinh_and_0.003",
               "torch_MLP_instead_of_tiny_cuda_nn_fused_MLP"])
    for name,keys in (("tcnn_hash_single_exact",KEYS[:1]),("tcnn_hash_two_exact",KEYS),("tcnn_hash_five_aligned",FIVE_KEYS)):
        field(name,"hashgrid",keys,"stress",["--steps","1000"],
              ["--backend","tcnn-fused","--input-mode","xyz" if len(keys)==1 else "both","--protocol","sampled","--stress-transform","asinh","--selection","worst-stress"],[15,16],
              ["official_tiny_cuda_nn_FullyFusedMLP_reimplementation_of_stated_architecture",
               "historical_asinh_constants_LR_and_initialization_seed_unavailable_using_explicit_recorded_values"])
    field("transolver_single","transolver",KEYS[:1],"both",["--epochs","500"],
          ["--checkpoint-blocks"],[],["additional_user_requested_original_Transolver_using_official_constructor_defaults"])
    field("transolverpp_single_asinh","transolverpp",KEYS[:1],"both",["--epochs","500"],
          ["--stress-transform","asinh","--checkpoint-blocks"],[5],
          ["single_case_extension_of_two_case_reference","asinh_constants_not_given_using_P90"])
    field("ts3_single","ts3",KEYS[:1],"both",["--epochs","500"],
          ["--checkpoint-blocks"],[4], ["TS3_backbone_dimensions_not_given_using_official_volume_preset"])
    for method in ("transolver","transolverpp","ts3"):
        extra=["--checkpoint-blocks"]
        if method=="transolverpp":extra.extend(["--stress-transform","asinh"])
        field(f"{method}_five",method,FIVE_KEYS,"both",["--epochs","500"],extra,[4,5],
              ["five_case_extension_per_case_500_complete_passes","unpublished_reference_details_use_official_preset_and_recorded_defaults"])
    field("ts3_two_short","ts3",KEYS,"both",["--steps","800"],["--checkpoint-blocks"],[4],
          ["TS3_architecture_not_in_slide_using_official_volume_preset"])
    field("transolver_two","transolver",KEYS,"both",["--epochs","500"],["--checkpoint-blocks"],[],
          ["additional_user_requested_original_Transolver_using_official_constructor_defaults"])
    field("transolverpp_two_linear","transolverpp",KEYS,"both",["--epochs","500"],["--checkpoint-blocks"],[5],
          ["25_feature_schema_unavailable_using_explicit_deck_derived_schema"])
    field("transolverpp_two_asinh","transolverpp",KEYS,"both",["--epochs","500"],
          ["--stress-transform","asinh","--checkpoint-blocks"],[5],
          ["asinh_constants_unavailable_using_pooled_P90_centered_standardized_asinh"])
    reference=[str(OUT/f"ngp4_{key.replace('/','_')}_{target}") for key in KEYS for target in ("u","stress")]
    for method in ("ts3","transolverpp"):
        field(f"{method}_residual_two",method,KEYS,"both",["--epochs","500"],
              ["--checkpoint-blocks","--reference-runs",*reference],[6],
              ["frozen_NGP_added_only_at_output_no_coarse_field_in_input","NGP_small_grid_details_unavailable"])
        jobs[-1]["requires"].extend([str(Path(r)/"summary.json") for r in reference])
    for name,outputs in (("fno64_joint","both"),("fno64_stress","stress")):
        add(name,"geobc.grid_train",["--grid",GRID64,"--outputs",outputs,"--modes","12","--steps","500"],[GRID64],[10,11,13],
            ["18_grid_feature_definitions_unavailable_using_occupancy_16_basic_channels_length",
             "projection_padding_LR_not_given_using_official_FNO_projection128_padding0_LR0.003"])
    add("fno192_stress","geobc.grid_train",["--grid",GRID192,"--outputs","stress","--modes","16","--steps","9905","--eval-every","100"],[GRID192],[12,13],
        ["reference_user_stopped_at_9905_reproducing_same_duration_without_assuming_convergence",
         "projection_padding_LR_not_given_using_recorded_defaults"])
    add("fno192_hash_distill","geobc.distill",["--teacher",OUT/"fno192_stress/teacher_grid_best.npz",
        "--nodes",EXACT/"506_533_ver.npz","--steps","1000","--backend","tcnn-fused"],[OUT/"fno192_stress/summary.json",EXACT/"506_533_ver.npz"],[14],
        ["teacher_best_epoch_will_follow_this_run_not_force_original_epoch8200","distillation_LR_sample_count_asinh_constants_unavailable_record_defaults"])
    field("ts3_two_long","ts3",KEYS,"both",["--epochs","5000"],["--checkpoint-blocks"],[4],
          ["reference_exact_TS3_backbone_and_optimization_schedule_unavailable_using_official_volume_preset_and_LR0.003"])
    add("hash50_sampled","geobc.global_field",["--data","data/prepared50_multifield","--steps","10000","--batch-size","51200","--backend","tcnn-fused"],
        ["data/prepared50_multifield/manifest.json"],[17],
        ["original270_train_pool_missing_using_official_first50_CADs_13ver13hor12dia12tor"])
    add("hash50_full1000ep","geobc.global_field",["--data","data/prepared50_multifield","--epochs","1000","--batch-size","51200", 
        "--initialize-weights",OUT/"hash50_sampled/best.pt","--fresh-adam","--backend","tcnn-fused"],
        [OUT/"hash50_sampled/summary.json"],[17,18,19,20],
        ["original270_train_pool_missing_using_explicit_replacement50","steps_per_epoch_recomputed_from_actual_selected_node_count"])
    return jobs


def run(args):
    os.chdir(ROOT)
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=plan()
    selected=[j for j in jobs if not args.only or j["name"] in args.only]
    if args.only and set(args.only)-{j["name"] for j in jobs}:raise ValueError("Unknown job name")
    plan_payload={"dataset":"original DeepJEB v1.0","main_cases":KEYS,"five_cases":FIVE_KEYS,
                  "historical_source":"references/slides_text.json","original_experiment_directory":"unavailable",
                  "data_alignment":"CSV coordinates -> H5 vertices; H5-only old results invalidated",
                  "jobs":jobs,"created_at_utc":utc_now()}
    write_json(ROOT/"notes/all_methods_plan.json",plan_payload)
    if args.dry_run:
        print(json.dumps(plan_payload,indent=2,ensure_ascii=False))
        return
    import fcntl
    lock=(OUT/"coordinator.lock").open("a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic()
    state={"pid":os.getpid(),"phase":"running","started_at_utc":utc_now(),"completed":[],"current":None,
           "total_jobs":len(selected),"failures":[]}
    def update(**extra):
        state.update(extra,updated_at_utc=utc_now(),elapsed_seconds=time.monotonic()-started)
        write_json(OUT/"coordinator_status.json",state)
    update()
    for job in selected:
        folder=Path(job["output"])
        summary=folder/"summary.json"
        if summary.exists() and json.loads(summary.read_text()).get("status")=="completed":
            state["completed"].append({"name":job["name"],"reused":True,"elapsed_seconds":json.loads(summary.read_text()).get("elapsed_seconds")})
            update()
            continue
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Incomplete nonempty stage needs explicit repair/resume: {folder}")
        wait_started=time.monotonic()
        while any(not Path(p).exists() for p in job["requires"]):
            missing=[p for p in job["requires"] if not Path(p).exists()]
            update(phase="waiting_for_inputs",current=job["name"],missing_inputs=missing)
            if time.monotonic()-wait_started>args.input_timeout:raise TimeoutError(f"Missing inputs for {job['name']}: {missing}")
            time.sleep(5)
        update(phase="training",current=job["name"],missing_inputs=[],current_started_at_utc=utc_now())
        job_started=time.monotonic()
        log=(OUT/(job["name"]+".log")).open("a",buffering=1)
        child=subprocess.Popen(job["command"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        update(child_pid=child.pid)
        while child.poll() is None:
            time.sleep(5)
            update(current_elapsed_seconds=time.monotonic()-job_started)
        log.close()
        if child.returncode:
            state["failures"].append({"name":job["name"],"exit_code":child.returncode})
            update(phase="failed",child_pid=None)
            raise RuntimeError(f"Stage failed: {job['name']}; see its log")
        state["completed"].append({"name":job["name"],"elapsed_seconds":time.monotonic()-job_started})
        update(child_pid=None)
    update(phase="completed",current=None,child_pid=None,completed_at_utc=utc_now())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--only",nargs="+")
    p.add_argument("--input-timeout",type=int,default=3600)
    run(p.parse_args())
