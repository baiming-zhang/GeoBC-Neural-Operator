"""One isolated target-free full service request per frozen model/DEV2 case.

The declared start supplies canonical normalized nodes/conditions and a trained
model. Raw CAD/deck/condition generation is upstream and is not measured here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

MODULE_ENTERED = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
SOURCE = "geobc/solver_service_benchmark.py"
PLAN = "notes/geobc_solver_service_benchmark_plan.json"
KIND = "solver_service_cost_screen"
KEYS = ("102_249/ver", "102_38/ver")
REQUESTS = tuple(method+":"+key for method in ("fno", "mlp") for key in KEYS)
MEMBERS = ("xyz", "features", "sample_key", "label_alignment", "faces")
SOURCES = (SOURCE, "geobc/solver_selection_models.py", "geobc/conditioned_field.py", "geobc/solver_selection_data.py", "geobc/benchmark.py", "geobc/fno.py")
ZERO = ("zero_GT_reads", "displacement_target_reads", "stress_target_reads", "training_updates", "scaler_fits", "old_final_reads", "sealed_new_test_reads", "DEV_dia_reads")


def host_path(value):
    text = str(value).replace("\\", "/")
    if os.name == "nt" and text.startswith("/mnt/") and len(text)>7 and text[5].isalpha() and text[6] == "/":
        text = text[5].upper()+":"+text[6:]
    elif os.name != "nt" and len(text)>2 and text[1:3] == ":/":
        text = "/mnt/"+text[0].lower()+text[2:]
    path = Path(text)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def read(path):
    return json.loads(host_path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = host_path(path); temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, path)


def sha(path):
    return hashlib.sha256(host_path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stat_ok(record):
    value = host_path(record["path"]).stat()
    return value.st_size == record["bytes"] and value.st_mtime_ns == record["mtime_ns"]


def pinned_json(record):
    require(host_path(record["path"]).stat().st_size == record["bytes"] and sha(record["path"]) == record["sha256"], "Pinned small JSON identity differs")
    return read(record["path"])


def folder(key):
    return key.replace(":", "_").replace("/", "_")


def request_contract(plan_path=PLAN):
    """Standard-library JSON/source hashes/stat only, never arrays/weights/GPU."""
    plan = read(plan_path); recipe = plan["recipe"]
    require(plan["kind"] == KIND and tuple(plan["request_keys"]) == REQUESTS and tuple(plan["case_keys"]) == KEYS
        and set(plan["models"]) == {"fno", "mlp"} and set(plan["records"]) == set(KEYS), "Fixed four-request scope differs")
    fixed = {"seed":2026,"device":"cuda:0","cloud_seed":2026,"cloud_points":1024,"grid_resolution":64,"point_chunk":32768,
        "maximum_cumulative_stage_wall_seconds":900,"maximum_request_wall_seconds":180,"maximum_peak_memory_bytes":24*1024**3,
        "partial_cold_relative_to_declared_xyz25_start":False,"raw_CAD_end_to_end":False}
    require(all(recipe[name] == value for name,value in fixed.items()) and tuple(recipe["allowed_prepared_members"]) == MEMBERS
        and all(plan["read_policy"][name] == 0 for name in ZERO), "Frozen input/service/budget policy differs")
    split = pinned_json(plan["existing_split"]); manifest = pinned_json(plan["selection_manifest"])
    require(manifest["status"] == "completed" and manifest["scaler"] == plan["scaler"] and plan["scaler"]["channels"] == [0,1,2,3], "Inherited TRAIN-only scaler differs")
    for key in KEYS:
        record = plan["records"][key]; archive = record["canonical_archive"]
        require(record["group"] == "DEV" and key.split("/")[0] in split["groups"]["validation"] and key in manifest["validation_keys"]
            and record["node_count"] == manifest["records"][key]["num_nodes"] and stat_ok(archive)
            and host_path(archive["path"]) == host_path(split["case_paths"][key]), "DEV canonical input stat/scope differs")
        source = next(item for item in manifest["contract"]["sources"] if item["key"] == key)
        require(host_path(source["path"]) == host_path(archive["path"]) and source["bytes"] == archive["bytes"]
            and source["mtime_ns"] == archive["mtime_ns"], "Original input archive identity differs")
    models = {}
    for method,step in (("fno",2880),("mlp",7080)):
        item = plan["models"][method]; parent = pinned_json(item["summary"]); config = pinned_json(item["config"])
        seed27 = pinned_json(item["seed2027_summary_reference_only"])
        require(item["method"] == method and item["seed"] == 2026 and item["selected_step"] == step and stat_ok(item["checkpoint"])
            and host_path(item["checkpoint"]["path"]).name == "last.pt" and parent["status"] == "completed"
            and parent["method"] == method and parent["seed"] == 2026 and parent["endpoint_step"] == parent["final_metrics"]["step"] == step
            and parent["config"] == config and config["scaler"] == plan["scaler"], "Fixed completed600s last endpoint identity differs")
        require(seed27["status"] == "completed" and seed27["method"] == method and seed27["seed"] == 2027
            and seed27["endpoint_step"] == seed27["final_metrics"]["step"], "Historical2027 summary differs")
        models[method] = {**item,"expected_config_sha256":digest(config),"source_config":config["model"],"parameter_counts":config["parameter_counts"],
            "budget":parent["budget"],"existing_accuracy_seed2026":{key:parent["final_metrics"]["validation"]["per_case"][key] for key in KEYS},
            "existing_accuracy_seed2027_reference_only":{key:seed27["final_metrics"]["validation"]["per_case"][key] for key in KEYS}}
    physics = {}
    for key in KEYS:
        refs = plan["historical_physical_references"][key]; zero = pinned_json(refs["old_zero_accuracy_reference"]); coarse = pinned_json(refs["new_coarse_numerical_reference"])
        require(zero["status"] == coarse["status"] == "completed" and zero["numerically_converged"] is True and coarse["numerically_converged"] is True
            and coarse["same_solution_passed"] is True and coarse["read_receipt"]["GT_reads"] == 0 and coarse["metrics"] is None, "Accepted separate physical evidence differs")
        physics[key] = {"sources":refs,"old_zero_GT_accuracy_only":zero["metrics"],"old_zero_true_residual":zero["true_relative_residual"],
            "new_coarse_true_residual":coarse["true_relative_residual"],"new_coarse_numerical_comparison":coarse["numerical_comparison"],
            "new_coarse_charged_costs":coarse["costs"],"new_coarse_GT_reads":0,"fresh_physical_cold_measured_here":False,"old_GT_error_not_assigned_to_coarse":True}
    sources = {name:sha(name) for name in SOURCES}
    require(all(name in sources and sources[name] == value for name,value in plan.get("source_sha256",{}).items()), "Pinned frozen source differs")
    science = {name:plan[name] for name in ("recipe","read_policy","models","records","scaler","selection_manifest","existing_split","historical_physical_references")}
    return {"schema":"geobc_solver_service_benchmark_v1","kind":KIND,"request_keys":list(REQUESTS),"models":models,"records":plan["records"],
        "scaler":plan["scaler"],"plan_science":science,"science_recipe_sha256":digest(science),"source_sha256":sources,
        "historical_physical_evidence":physics,"output":str(host_path(plan["output"])),"array_or_weight_reads_in_request_contract":0},plan


def released(contract, plan):
    require(plan.get("scientific_rules_frozen") is True and plan.get("actual_execution_released") is True
        and plan.get("source_sha256") == contract["source_sha256"], "Exact root release/source pins required before child scientific execution")
    marker = read(plan["root_review"])
    require(marker.get("passed") is True and marker.get("source_sha256") == contract["source_sha256"][SOURCE], "Exact passed own-source root marker required")


def empty_receipt():
    return {**{name:0 for name in ZERO},"model_checkpoint_reads":0,"model_checkpoint_read_attempts":0,"model_checkpoint_read_completed":0,
        "prepared_archive_reads":0,"prepared_archive_read_attempts":0,"prepared_archive_read_completed":0,"prepared_member_reads":0,
        "prepared_member_read_attempts":{name:0 for name in MEMBERS},"prepared_member_read_completed":{name:0 for name in MEMBERS}}


def combined_receipt(requests):
    result = empty_receipt(); result["per_request"] = {}
    for key,item in requests.items():
        value = item.get("read_receipt",empty_receipt()); result["per_request"][key] = value
        for name in result:
            if name == "per_request":
                continue
            if isinstance(result[name],dict):
                for member in MEMBERS:
                    result[name][member] += value[name][member]
            else:
                result[name] += value[name]
    return result


class BudgetEnd(Exception):
    pass


def serve_child(plan_path, output_path, request_key, launched, allowance):
    metadata_began = time.monotonic(); contract,plan = request_contract(plan_path); released(contract,plan)
    require(request_key in REQUESTS and 0<allowance<=180 and launched<=MODULE_ENTERED, "Invalid one-request child launch")
    output = host_path(output_path); require(output == host_path(plan["output"]), "Frozen child output differs")
    directory = output/folder(request_key); directory.mkdir(exist_ok=True)
    config = {"contract":contract,"request_key":request_key,"stage":KIND}; parent_meta = read(directory/"request_metadata.json")
    require(parent_meta["config_sha256"] == digest({"contract":contract,"stage":KIND}) and parent_meta["launch_monotonic"] == launched, "Parent single-request launch identity differs")
    method,key = request_key.split(":",1); record = contract["records"][key]; endpoint = contract["models"][method]
    receipt = empty_receipt(); phases = {"interpreter_stdlib_module_startup_seconds":MODULE_ENTERED-launched,"metadata_release_seconds":time.monotonic()-metadata_began}
    deadline = launched+allowance; result = {"status":"running","request_key":request_key,"method":method,"case_key":key,"selected_step":endpoint["selected_step"],
        "node_count":record["node_count"],"config":config,"read_receipt":receipt,"costs":{},"prediction":None,"historical_accuracy_only":endpoint["existing_accuracy_seed2026"][key]}
    write(directory/"config.json",config)
    def journal(phase):
        result["costs"].update(child_until_receipt_wall_seconds=time.monotonic()-launched,phase_seconds=phases,
            request_budget_seconds=allowance,partial_cold=False,startpoint=plan["recipe"]["request_start"],missing_upstream=plan["recipe"]["missing_upstream"],raw_CAD_end_to_end=False)
        write(directory/"read_receipt.json",receipt); write(directory/"status.json",{"status":result["status"],"phase":phase,"costs":result["costs"],"read_receipt":receipt})
    def ensure():
        if time.monotonic()>=deadline:
            raise BudgetEnd("One-request deadline reached; no fallback, warm replay or GT evaluation")
    def phase(name, function, gpu=False):
        ensure()
        if gpu:
            torch.cuda.synchronize()
        tick = time.monotonic(); value = function()
        if gpu:
            torch.cuda.synchronize()
        phases[name] = time.monotonic()-tick; journal(name+":completed"); return value
    journal("metadata_complete")
    try:
        ensure(); tick = time.monotonic()
        import numpy as np
        import torch
        from .conditioned_field import _cad_seed
        from .solver_selection_models import build_selection_model
        from .solver_selection_data import _stencil, _splat, query_grid
        from .benchmark import FieldScaler
        torch.cuda.set_device(0); torch.cuda.init(); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high"); device = torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(min(1.,24*1024**3/device.total_memory),0); torch.cuda.reset_peak_memory_stats(0); torch.cuda.synchronize()
        phases["scientific_imports_CUDA_startup_seconds"] = time.monotonic()-tick; journal("CUDA_started")
        def inputs():
            archive = record["canonical_archive"]; require(stat_ok(archive), "Canonical input archive stat differs")
            receipt["prepared_archive_reads"] += 1; receipt["prepared_archive_read_attempts"] += 1; journal("prepared_archive_attempt")
            values = {}
            with np.load(host_path(archive["path"]),allow_pickle=False) as stored:
                def member(name):
                    ensure()
                    require(name in MEMBERS, "Forbidden prepared member access")
                    receipt["prepared_member_reads"] += 1; receipt["prepared_member_read_attempts"][name] += 1; journal("prepared_member_attempt:"+name)
                    value = np.asarray(stored[name]); receipt["prepared_member_read_completed"][name] += 1; journal("prepared_member_completed:"+name); return value
                for name in MEMBERS[:-1]:
                    ensure(); values[name] = member(name)
                values["faces"] = member("faces") if "faces" in stored.files else None
            receipt["prepared_archive_read_completed"] += 1; journal("prepared_archive_completed")
            xyz = np.asarray(values["xyz"],np.float32); features = np.asarray(values["features"],np.float32)
            require(str(values["sample_key"].item()) == key and str(values["label_alignment"].item()) == record["label_alignment"]
                and xyz.shape == (record["node_count"],3) and features.shape == (record["node_count"],25)
                and np.isfinite(xyz).all() and np.isfinite(features).all() and xyz.min()>=-1e-5 and xyz.max()<=1+1e-5, "Canonical input schema/identity/nonfinite differs")
            result["actual_input_schema"] = {"xyz":[len(xyz),3],"features":[len(xyz),25],"coordinate_frame":"original canonical isotropic normalized [0,1]",
                "allowed_members_accessed":{name:receipt["prepared_member_read_completed"][name] for name in MEMBERS},"target_members_accessed":[]}
            return xyz,features,values["faces"]
        xyz,features,faces = phase("prepared_input_IO_identity_seconds",inputs)
        def cloud():
            if faces is not None and faces.size:
                require(faces.ndim == 2 and faces.shape[1] == 3 and np.issubdtype(faces.dtype,np.integer), "Boundary faces schema differs")
                pool = np.unique(faces); require(pool[0]>=0 and pool[-1]<len(xyz), "Boundary IDs outside canonical nodes")
                source = "unique original quadratic-boundary node IDs"
            else:
                pool = np.arange(len(xyz),dtype=np.int64); source = "all original FEM nodes; boundary unavailable"
            count = min(1024,len(pool)); chosen = np.random.default_rng(_cad_seed(key.split('/')[0],2026)).choice(pool,count,replace=False)
            points = np.zeros((1024,3),np.float32); mask = np.zeros(1024,np.bool_); points[:count] = xyz[chosen]; mask[:count] = True
            result["geometry_cloud"] = {"points":1024,"valid_points":count,"pool":source,"pool_nodes":len(pool),"seed":2026,"exact_old_sampling_rule":True}
            return points,mask
        points,mask = phase("PointNet_cloud_preparation_seconds",cloud)
        def transfer():
            return {"xyz":torch.as_tensor(xyz,device="cuda:0"),"features":torch.as_tensor(features,device="cuda:0"),
                "cloud":torch.as_tensor(points,device="cuda:0").unsqueeze(0),"mask":torch.as_tensor(mask,device="cuda:0").unsqueeze(0)}
        case = phase("input_H2D_seconds",transfer,True)
        if method == "fno":
            def frontend():
                indices,weights = _stencil(case["xyz"],64); raster,active = _splat(case["features"],indices,weights,64)
                axis = torch.linspace(0,1,64,device="cuda:0"); coordinates = torch.stack(torch.meshgrid(axis,axis,axis,indexing="ij"))
                grid = torch.cat((active.float().reshape(1,64,64,64),coordinates,raster.T.reshape(25,64,64,64)),dim=0)
                case.update(grid=grid.unsqueeze(0),query_indices=indices,query_weights=weights)
                result["FNO_frontend"] = {"schema":[1,29,64,64,64],"rebuilt_target_free":True,"cached_grid_cloud_stencil_reads":0,"query_coverage":1.0}
            phase("FNO_R64_stencil_splat_grid_seconds",frontend,True)
        def load_model():
            require(stat_ok(endpoint["checkpoint"]), "Fixed checkpoint stat differs")
            receipt["model_checkpoint_reads"] += 1; receipt["model_checkpoint_read_attempts"] += 1; journal("model_checkpoint_attempt")
            payload = torch.load(host_path(endpoint["checkpoint"]["path"]),map_location="cpu",weights_only=False)
            receipt["model_checkpoint_read_completed"] += 1; journal("model_checkpoint_completed")
            expected = pinned_json(endpoint["config"])
            require(payload["step"] == endpoint["selected_step"] and payload["config"] == expected
                and digest(expected) == endpoint["expected_config_sha256"] and expected["scaler"] == contract["scaler"], "Loaded fixed endpoint config/step/scaler differs")
            model = build_selection_model(method,"cuda:0",2026)
            require(model.source_config == endpoint["source_config"] and model.parameter_counts == endpoint["parameter_counts"], "Frozen public model source/config/counts differ")
            model.load_state_dict(payload["model"],strict=True); model.eval(); result["parameter_counts"] = model.parameter_counts
            return model
        model = phase("model_construction_checkpoint_load_H2D_seconds",load_model,True)
        with torch.no_grad():
            context = phase("PointNet_context_encode_seconds",lambda:model.encode_geometry(case["cloud"],case["mask"]),True)
            if method == "fno":
                grid_prediction = phase("FNO_forward_grid_seconds",lambda:model.forward_grid(case["grid"],context),True)
                encoded = phase("allN_trilinear_query_seconds",lambda:query_grid(grid_prediction,case),True)
            else:
                encoded = phase("MLP_allN_forward_seconds",lambda:torch.cat([model.forward_points(case["xyz"][start:start+32768],
                    case["features"][start:start+32768],context) for start in range(0,len(xyz),32768)]),True)
            require(encoded.shape == (len(xyz),4) and torch.isfinite(encoded).all(), "Nonfinite/incomplete encoded response")
            encoded_cpu = phase("allN_output_D2H_seconds",lambda:encoded.float().cpu().numpy(),True)
        physical = phase("frozen_physical_decode_seconds",lambda:FieldScaler(**contract["scaler"]).decode(encoded_cpu))
        require(physical.shape == (len(xyz),4) and np.isfinite(physical).all(), "Incomplete/nonfinite physical response")
        def persist():
            path = directory/"physical_field.npy"; temporary = path.with_name(path.name+".tmp")
            with temporary.open("wb") as stream:
                np.save(stream,physical,allow_pickle=False)
            os.replace(temporary,path); stat = path.stat()
            result["prediction"] = {"path":str(path),"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns,"shape":list(physical.shape),"dtype":str(physical.dtype),
                "values_sha256":hashlib.sha256(physical.tobytes()).hexdigest(),"allN":True,"finite":True,"node_order":"unchanged canonical input row order",
                "channels":["ux_mm","uy_mm","uz_mm","signed_stress_MPa"],"GT_re_evaluated":False}
        phase("allN_physical_persistence_seconds",persist)
        result["memory"] = {"peak_allocated_bytes":torch.cuda.max_memory_allocated(0),"peak_reserved_bytes":torch.cuda.max_memory_reserved(0),
            "ceiling_bytes":24*1024**3,"device_name":device.name,"torch_version":str(torch.__version__)}
        require(result["memory"]["peak_reserved_bytes"]<=24*1024**3, "Measured reserved peak exceeds24GiB ceiling")
        ensure(); result["status"] = "completed"
    except BudgetEnd as error:
        result.update(status="budget_limited",reason=str(error))
    except Exception as error:
        result.update(status="failed",reason=f"{type(error).__name__}: {error}")
    journal("child_finished"); result["costs"]["child_until_receipt_wall_seconds"] = time.monotonic()-launched
    write(directory/"summary.json",result); journal("child_summary_committed")
    return result


class StageClock:
    def __init__(self, output, config, began):
        self.output,self.config,self.began = output,config,began
        self.state = read(output/"state.json") if (output/"state.json").exists() else {"config":config,"requests":{},"costs":{"cumulative_stage_wall_seconds":0.}}
        require(self.state["config"] == config, "Persistent service science identity differs")
        self.prior = self.state["costs"]["cumulative_stage_wall_seconds"]
        require(math.isfinite(self.prior) and self.prior>=0, "Invalid prior actual service clock")
        # An interrupted parent cannot know when its live child stopped. Charge
        # the unobserved interval conservatively; never reset its stage budget.
        if any(item.get("status") == "launched" for item in self.state["requests"].values()):
            saved = self.state.get("last_persisted_epoch_seconds")
            require(saved is not None and math.isfinite(saved), "Interrupted launch has no persistent budget clock")
            gap = max(0.,time.time()-saved); self.prior += gap
            self.state["costs"]["conservatively_charged_resume_gap_seconds"] = self.state["costs"].get("conservatively_charged_resume_gap_seconds",0.)+gap
    def elapsed(self):
        return self.prior+time.monotonic()-self.began
    def remaining(self):
        return max(0.,900-self.elapsed())
    def touch(self, phase):
        self.state["costs"]["cumulative_stage_wall_seconds"] = self.elapsed(); self.state["last_persisted_epoch_seconds"] = time.time()
        write(self.output/"state.json",self.state)
        write(self.output/"status.json",{"status":"running","phase":phase,"costs":self.state["costs"],"read_receipt":combined_receipt(self.state["requests"])})


def report(summary, plan):
    """Small JSON only; historical precision copied, never recomputed."""
    output = host_path(plan["results_output"]); output.mkdir(parents=True,exist_ok=True)
    write(output/"metrics.json",summary)
    lines = ["# 共同 DEV2 完整服务回放", "", "固定2026累计600s last端点：FNO2880、MLP7080；四个独立进程各一次，零训练/新scaler/GT。", "",
        "| 方法 / ver CAD | 实测服务完整进程 wall(s) | 既存 U / S L2% | 本次交付 |", "|---|---:|---:|---|"]
    for key in REQUESTS:
        item = summary["requests"][key]; wall = item.get("costs",{}).get("cold_request_wall_seconds"); old = item["historical_accuracy_only"]
        display = "null" if wall is None else f"{wall:.3f}"
        lines.append(f"| {key} | {display} | {100*old['u_relative_l2']:.2f} / {100*old['stress_relative_l2']:.2f} | {item['status']} |")
    physics = summary["historical_physical_evidence"]; values = [physics[key] for key in KEYS]
    lines += ["", "请求起点是已给 canonical xyz/25条件及可用边界faces、固定模型；完整包含进程启动/import/CUDA、input IO/cloud/grid/H2D/model、全N推理/query、物理decode/persist与退出。", "",
        "原始deck、归一坐标/边界提取与25条件生成在上游：这不是 raw-CAD 端到端。缺失或budget行保留null，partialcold及阶段账本见metrics.json。", "",
        f"旧 zero 端点 GT U：{100*values[0]['old_zero_GT_accuracy_only']['u_vector_relative_l2']:.6f}% / {100*values[1]['old_zero_GT_accuracy_only']['u_vector_relative_l2']:.6f}%；只属于旧端点。",
        f"新coarse仅引用既存同解/true residual证据，charged cold为{values[0]['new_coarse_charged_costs']['coarse_cold_accounted_wall_seconds']:.3f}s / {values[1]['new_coarse_charged_costs']['coarse_cold_accounted_wall_seconds']:.3f}s，physical fresh仍未测，本次没有物理回放。", "",
        "2027只引用既存精度；本次不重评价精度，GPU原子splat重建可能有FP32舍入差。完整FE与点/条件输入接口不同，不能拼成架构因果胜出。", "",
        "U-only 与 U/S 能力分开：物理 signed scalar exporter缺口仍在。final/dia/sealed保持关闭；无warm benchmark或新增资源/数学探测。"]
    require(len(lines)<=25, "Service BRIEF exceeds25 lines")
    (output/"BRIEF.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(plan_path=PLAN, output_path=None):
    began = time.monotonic(); contract,plan = request_contract(plan_path); released(contract,plan)
    output = host_path(output_path or plan["output"]); require(output == host_path(plan["output"]), "Frozen service output differs")
    config = {"contract":contract,"stage":KIND}; output.mkdir(parents=True,exist_ok=True)
    if (output/"config.json").exists():
        require(read(output/"config.json") == config, "Prior service config differs")
    if (output/"summary.json").exists():
        saved = read(output/"summary.json"); require(saved["config"] == config, "Prior service summary differs")
        if saved["status"] == "completed":
            return saved
    write(output/"config.json",config); clock = StageClock(output,config,began); clock.touch("starting")
    requests = clock.state["requests"]; failed = None
    try:
        for key in REQUESTS:
            method,case_key = key.split(":",1); directory = output/folder(key); directory.mkdir(exist_ok=True)
            if key in requests:
                previous = requests[key]
                if previous.get("status") in ("completed","budget_limited","budget_not_attempted"):
                    continue
                if previous.get("status") == "failed":
                    raise RuntimeError("Single replay failed; preserve its observed exit/cost/reads without automatically repeating: "+key)
                if (directory/"summary.json").exists():
                    child = read(directory/"summary.json"); require(child["config"]["contract"] == contract, "Interrupted child identity differs")
                    child["costs"].update(cold_request_wall_seconds=None,parent_launch_to_process_exit_seconds=None,complete_cold_cost_accounted=False,
                        partial_cold=True,missing_process_exit_accounting=True)
                    requests[key] = child; clock.touch("interrupted_child_receipt_adopted")
                    if child["status"] == "failed":
                        raise RuntimeError("Single replay failed and is not automatically repeated: "+key)
                    continue
                raise RuntimeError("Previously launched child has no finite exit receipt; preserve ledger and do not launch another GPU request")
            if clock.remaining()<=0:
                break
            allowance = min(180.,clock.remaining()); launch = time.monotonic()
            metadata = {"config_sha256":digest(config),"request_key":key,"launch_monotonic":launch,"allowed_seconds":allowance,"attempt":1}
            write(directory/"request_metadata.json",metadata)
            requests[key] = {"status":"launched","request_key":key,"method":method,"case_key":case_key,"read_receipt":empty_receipt(),"launch_metadata":metadata}
            clock.touch("launching:"+key)
            command = [sys.executable,"-m","geobc.solver_service_benchmark","--plan",str(host_path(plan_path)),"--output",str(output),
                "--_request",key,"--_launch-monotonic",repr(launch),"--_allowance",repr(allowance)]
            with (directory/"child.log").open("x",encoding="utf-8") as log:
                process = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                requests[key]["child_pid"] = process.pid; clock.touch("child_running:"+key); timeout = False
                while process.poll() is None:
                    if time.monotonic()-launch>=allowance:
                        timeout = True; process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait(timeout=5)
                        break
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        if (directory/"read_receipt.json").exists():
                            requests[key]["read_receipt"] = read(directory/"read_receipt.json")
                        clock.touch("child_running:"+key)
                process_exit = time.monotonic()
            wall = process_exit-launch
            if (directory/"summary.json").exists():
                child = read(directory/"summary.json"); require(child["config"]["contract"] == contract and child["request_key"] == key, "Child committed summary identity differs")
            else:
                child = {**requests[key],"status":"budget_limited" if timeout else "failed","reason":"Child deadline termination" if timeout else "Child exited without committed summary",
                    "costs":{},"prediction":None}
                if (directory/"read_receipt.json").exists():
                    child["read_receipt"] = read(directory/"read_receipt.json")
            if timeout:
                child["status"] = "budget_limited"
            child["costs"].update(cold_request_wall_seconds=wall if child.get("prediction") else None,parent_launch_to_process_exit_seconds=wall,
                parent_shutdown_collection_seconds=max(0.,wall-child["costs"].get("child_until_receipt_wall_seconds",wall)),complete_cold_cost_accounted=child.get("prediction") is not None,
                partial_cold=child.get("prediction") is None,request_budget_seconds=allowance,request_overshoot_seconds=max(0.,wall-allowance),
                startpoint=plan["recipe"]["request_start"],missing_upstream=plan["recipe"]["missing_upstream"],raw_CAD_end_to_end=False)
            requests[key] = child; write(directory/"summary.json",child); clock.touch("request_exit_accounted:"+key)
            if child["status"] == "failed" or (process.returncode != 0 and not timeout):
                raise RuntimeError("Technical single replay failure: "+key+": "+str(child.get("reason")))
    except Exception as error:
        failed = f"{type(error).__name__}: {error}"
    for key in REQUESTS:
        method,case_key = key.split(":",1)
        requests.setdefault(key,{"status":"budget_not_attempted" if failed is None else "technical_not_attempted","request_key":key,"method":method,"case_key":case_key,
            "node_count":contract["records"][case_key]["node_count"],"selected_step":contract["models"][method]["selected_step"],"read_receipt":empty_receipt(),
            "prediction":None,"costs":{"cold_request_wall_seconds":None,"complete_cold_cost_accounted":False,"partial_cold":True,
                "startpoint":plan["recipe"]["request_start"],"missing_upstream":plan["recipe"]["missing_upstream"]}})
        requests[key]["historical_accuracy_only"] = contract["models"][method]["existing_accuracy_seed2026"][case_key]
    receipt = combined_receipt(requests); complete = all(requests[key]["status"] == "completed" and requests[key]["costs"].get("complete_cold_cost_accounted") is True
        and (requests[key].get("prediction") or {}).get("allN") is True and requests[key]["costs"].get("request_overshoot_seconds",0) == 0 for key in REQUESTS)
    passed = complete and all(receipt[name] == 0 for name in ZERO) and clock.elapsed()<=900
    summary = {"status":"failed" if failed else "completed","kind":KIND,"config":config,"request_keys":list(REQUESTS),"requests":requests,"read_receipt":receipt,
        "completion_gate":{"passed":passed,"four_full_finite_deliveries":complete,"zero_GT_training_or_forbidden_reads":all(receipt[name] == 0 for name in ZERO),"fixed_request_denominator":4},
        "scientific_outcome":"complete_declared_interface_service" if passed else "partial_or_budget_negative","failure_reason":failed,
        "historical_accuracy_seed2027_reference_only":{method:contract["models"][method]["existing_accuracy_seed2027_reference_only"] for method in ("fno","mlp")},
        "historical_physical_evidence":contract["historical_physical_evidence"],"elapsed_seconds":clock.elapsed(),
        "costs":{"cumulative_stage_wall_seconds":clock.elapsed(),"maximum_stage_wall_seconds":900,"stage_overshoot_seconds":max(0.,clock.elapsed()-900),
            "startpoint":plan["recipe"]["request_start"],"missing_upstream":plan["recipe"]["missing_upstream"],"partial_cold":not complete,"raw_CAD_end_to_end":False,
            "training_and_GT_accuracy_costs_inherited_only_not_charged_to_service":True,"fresh_physical_cold_measured_here":False},"limits":plan["limitations"]}
    summary["costs"]["conservatively_charged_resume_gap_seconds"] = clock.state["costs"].get("conservatively_charged_resume_gap_seconds",0.)
    clock.touch("finalization"); write(output/"summary.json",summary)
    if failed is None:
        report(summary,plan)
    summary["costs"]["cumulative_stage_wall_seconds"] = clock.elapsed(); summary["elapsed_seconds"] = clock.elapsed()
    summary["costs"]["stage_overshoot_seconds"] = max(0.,clock.elapsed()-900)
    if clock.elapsed()>900:
        summary["completion_gate"]["passed"] = False; summary["scientific_outcome"] = "partial_or_budget_negative"
    write(output/"summary.json",summary); clock.touch("failed" if failed else "completed")
    write(output/"status.json",{"status":summary["status"],"costs":summary["costs"],"read_receipt":receipt})
    if failed:
        raise RuntimeError(failed)
    write(host_path(plan["results_output"])/"metrics.json",summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--plan",type=Path,default=Path(PLAN)); parser.add_argument("--output",type=Path)
    parser.add_argument("--_request",help=argparse.SUPPRESS); parser.add_argument("--_launch-monotonic",type=float,help=argparse.SUPPRESS); parser.add_argument("--_allowance",type=float,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._request:
        served = serve_child(args.plan,args.output,args._request,args._launch_monotonic,args._allowance)
        if served["status"] == "failed":
            raise SystemExit(1)
    else:
        run(args.plan,args.output)
