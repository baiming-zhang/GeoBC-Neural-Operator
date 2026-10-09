"""One frozen physical-tensor versus zero-slot signed-stress rough screen.

Metadata entry points use the standard library. Actual release permits only
the five accepted zero-start physical input caches and canonical input/stress
members. No GT displacement, new solve, operator guard or external model is
used. Both complete DEV predictions precede either shared DEV stress read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_stress_readout_plan.json"
SOURCE = "geobc/elastic_stress_readout.py"
KIND = "elastic_signed_stress_readout_screen"
TRAIN = ("101_428/ver","101_56/ver","101_78/ver")
DEV = ("102_249/ver","102_38/ver")
CASES = TRAIN+DEV
ARMS = ("direct_zero_slots","physical_tensor_slots")
SOURCES = (SOURCE,"geobc/elastic_crosscad.py","geobc/elastic_pilot.py","geobc/elastic_pilot_data.py","geobc/conditioned_field.py")
INPUT_MEMBERS = ("xyz","features","grid_node_ids","sample_key","label_alignment","faces")


def helpers():
    from . import elastic_crosscad as cross
    from . import elastic_pilot as base
    return cross,base


def path(value):
    return helpers()[0].host_path(value)


def read(value):
    return helpers()[0].read_json(value)


def write(value,data):
    helpers()[1].write_json(path(value),data)


def require(condition,message):
    helpers()[0].require(condition,message)


def finite(value):
    return isinstance(value,(int,float)) and math.isfinite(value)


def digest(value):
    return helpers()[0].digest(value)


def request_contract(plan_path=PLAN):
    """Small JSON/stat/source-only identity; never imports scientific packages."""
    cross,base = helpers(); plan = read(plan_path)
    require(plan.get("kind") == KIND and tuple(plan.get("train_keys",())) == TRAIN
        and tuple(plan.get("validation_keys",())) == DEV and plan.get("case_replacement") is False,"Frozen TRAIN3/DEV2 scope differs")
    neural = plan["neural"]
    require(tuple(neural["arms"]) == ARMS and neural["input_dim"] == 67 and neural["seed"] == 2026
        and neural["learning_rate"] == .001 and neural["weight_decay"] == 0 and neural["passes_per_arm"] == 40
        and neural["full_case_updates_per_arm"] == 120 and neural["point_chunk"] == 32768,"Frozen scalar training recipe differs")
    require(tuple(plan["label_access"]["canonical_input_members"]) == INPUT_MEMBERS
        and plan["label_access"]["allowed_label_member"] == "stress only"
        and all(plan["label_access"][key] == 0 for key in ("displacement_reads","old_final_reads","DEV_dia_reads","sealed_new_test_reads")),
        "Closed displacement/final/dia/sealed scope differs")
    require(plan["budgets"]["maximum_cumulative_stage_wall_seconds"] == 900
        and plan["budgets"]["maximum_cumulative_training_wall_seconds_per_arm"] == 300,"Frozen cumulative budgets differ")
    guards = plan["physics_features"]["new_guards"]
    require(guards["partition_relative_tolerance"] == 1e-9 and guards["affine_relative_tolerance"] == 1e-9
        and guards["tensor_symmetry_relative_tolerance"] == 1e-12 and guards["constant_recovery_relative_tolerance"] == 1e-12,
        "Physical-feature tolerance differs")
    physical,physical_plan = cross.request_contract(plan["physical_plan"])
    require(path(plan["physical_output"]) == path(physical_plan["output"]),"Zero-start reference destination differs")
    old = read(path(plan["physical_output"])/"summary.json")
    require(old.get("status") == "completed" and old.get("config",{}).get("contract") == physical,"Completed original physical screen identity differs")
    records = {}
    for key in CASES:
        item = physical["records"][key]; saved = read(path(item["solve_output"])/"summary.json")
        require(saved.get("config") == {"contract":item,"stage":"elastic_crosscad_pcg"}
            and saved.get("status") == "completed" and saved.get("numerical_state") == "converged"
            and saved.get("numerically_converged") is True and finite(saved.get("true_relative_residual"))
            and saved["true_relative_residual"]<=1e-8 and saved.get("input_compatible") is True
            and saved.get("operator_compatible") is True,"Original zero-start physical endpoint not accepted: "+key)
        manifest_path = path(item["data_output"])/"manifest.json"; operator_path = path(item["operator_output"])/"summary.json"
        manifest,operator = read(manifest_path),read(operator_path)
        require(manifest.get("feasibility_pass") is True and manifest.get("contract") == item
            and operator.get("feasibility_pass") is True and operator.get("config") == {"contract":item,"stage":"elastic_crosscad_operator"}
            and operator["algebra_guards"]["K_applications"] == 8 and operator["input_manifest_sha256"] == cross.sha(manifest_path),
            "Accepted input/operator cache identity differs: "+key)
        prediction = saved["prediction"]
        require(base.stat_record(prediction["path"]) == {name:prediction[name] for name in ("bytes","mtime_ns")},"Original physical displacement stat differs")
        for value in operator["arrays"].values():
            require(base.stat_record(value["path"]) == {name:value[name] for name in ("bytes","mtime_ns")},"Committed operator array stat differs")
        costs = saved["costs"]; cold,gt = costs.get("cold_case_accounted_wall_seconds"),costs.get("GT_read_and_metrics_seconds")
        require(finite(cold) and finite(gt) and cold>gt>=0,"Original GT-free physical cold cost missing")
        records[key] = {"physical_record":item,"physical_prediction":prediction,"canonical_source":item["sources"]["canonical_npz"],
            "input_manifest":{"path":str(manifest_path),"sha256":cross.sha(manifest_path)},
            "operator_summary":{"path":str(operator_path),"sha256":cross.sha(operator_path)},
            "original_solve_summary_sha256":cross.sha(path(item["solve_output"])/"summary.json"),
            "physical_GT_free_cold_seconds":cold-gt,"node_count":item["expected_geometry"]["nodes"]}
    schedule = []; rng = random.Random(2026)
    for _ in range(40):
        order = list(TRAIN); rng.shuffle(order); schedule.extend(order)
    sources = {name:cross.sha(name) for name in SOURCES}
    for name,value in plan.get("source_sha256",{}).items():
        require(name in sources and sources[name] == value,"Pinned source differs: "+name)
    science = {key:plan[key] for key in ("physics_features","neural","label_access","budgets","comparison")}
    return {"schema":"geobc_elastic_signed_stress_readout_v1","kind":KIND,"train_keys":list(TRAIN),"validation_keys":list(DEV),
        "plan_science":science,"records":records,"physical_contract":physical,"case_order":schedule,
        "case_order_sha256":digest(schedule),"base_gate":{"all_five_original_zero_true_converged":True,"accepted_operator_guard_count":40},
        "source_sha256":sources,"output":str(path(plan["output"])),"array_or_label_reads_in_request_contract":0},plan


def zero_receipt():
    return {"TRAIN_stress_reads":0,"DEV_stress_reads":0,"stress_reads":0,"GT_reads":0,"displacement_reads":0,
        "old_final_reads":0,"DEV_dia_reads":0,"sealed_new_test_reads":0,"optimizer_steps":0,"scaler_fits":0,
        "external_model_or_weight_reads":0,"new_physical_solves":0,"K_guards_repeated":0}


def save_array(np,destination,value):
    destination = path(destination); temporary = destination.with_name(destination.name+".tmp")
    array = np.ascontiguousarray(value)
    with temporary.open("wb") as stream:
        np.save(stream,array,allow_pickle=False)
    os.replace(temporary,destination); info = destination.stat()
    return {"path":str(destination),"shape":list(array.shape),"dtype":str(array.dtype),"bytes":info.st_size,"mtime_ns":info.st_mtime_ns,
        "values_sha256":hashlib.sha256(array.tobytes()).hexdigest()}


def load_array(np,record,verify_values=False):
    require(helpers()[1].stat_record(record["path"]) == {name:record[name] for name in ("bytes","mtime_ns")},"Own cache file stat changed")
    value = np.load(path(record["path"]),allow_pickle=False)
    require(list(value.shape) == record["shape"] and str(value.dtype) == record["dtype"],"Own cache array schema differs")
    if verify_values:
        require(hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == record["values_sha256"],"Own cache value identity differs")
    return value


def signed_metrics(np,truth,prediction):
    require(truth.shape == prediction.shape and truth.ndim == 1 and len(truth)>0
        and np.isfinite(truth).all() and np.isfinite(prediction).all(),"Invalid allN signed-stress metric arrays")
    delta = prediction-truth; reference = float(np.square(truth).sum()); error = float(np.square(delta).sum())
    count = max(1,math.ceil(.01*len(truth)))
    ids = np.lexsort((np.arange(len(truth)), -np.abs(truth)))[:count]
    hot_reference = float(np.square(truth[ids]).sum()); hot_error = float(np.square(delta[ids]).sum())
    return {"num_nodes":len(truth),"stress_relative_l2":None if reference == 0 else math.sqrt(error/reference),
        "RMSE_MPa":math.sqrt(error/len(truth)),"MAE_MPa":float(np.abs(delta).mean()),"max_abs_error_MPa":float(np.abs(delta).max()),
        "zero_reference":reference == 0,"reference_energy_MPa2":reference,"error_energy_MPa2":error,
        "hotspot_relative_l2":None if hot_reference == 0 else math.sqrt(hot_error/hot_reference),"hotspot_zero_reference":hot_reference == 0,
        "hotspot_nodes":count,"hotspot_ties":"descending absolute GT signed stress, ascending original node row"}


def aggregate(per_case,keys):
    result = {"per_case":{key:per_case.get(key) for key in keys},"denominator":len(keys),
        "metrics_available_cases":sum(per_case.get(key) is not None for key in keys)}
    for name in ("stress_relative_l2","hotspot_relative_l2","RMSE_MPa","MAE_MPa","max_abs_error_MPa"):
        values = [(per_case.get(key) or {}).get(name) for key in keys]
        valid = all(finite(value) for value in values)
        result[name+"_mean"] = sum(values)/len(values) if valid else None
        result[name+"_worst"] = max(values) if valid else None
    return result


def build_model(seed=2026):
    import numpy as np
    import torch
    from torch import nn
    from .conditioned_field import PointNetGeometryEncoder
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    class ScalarReadout(nn.Module):
        def __init__(self):
            super().__init__(); self.encoder = PointNetGeometryEncoder(32,64)
            layers = []; width = 67
            for _ in range(4):
                layers.extend((nn.Linear(width,256),nn.ReLU())); width = 256
            layers.append(nn.Linear(256,1)); self.head = nn.Sequential(*layers)
        def encode_geometry(self,cloud,mask):
            return self.encoder(cloud,mask)
        def forward_points(self,xyz,features,context,tensor6,valid):
            return self.head(torch.cat((xyz,features,context.expand(len(xyz),-1),tensor6,valid[:,None]),dim=1)).squeeze(-1)
    return ScalarReadout().cuda()


def parameter_identity(model):
    values = {name:hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name,value in model.named_parameters() if value.requires_grad}
    return {"all_sha256":digest(values),"named_sha256":values,"parameter_counts":{"encoder":sum(value.numel() for value in model.encoder.parameters()),
        "head":sum(value.numel() for value in model.head.parameters()),"total":sum(value.numel() for value in model.parameters())}}


def rng_state(np,torch):
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}


def restore_rng(np,torch,state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def gradient_check(torch,model,context_leaf):
    missing = []; invalid = []; squared = {"encoder":0.,"head":0.}
    for name,value in model.named_parameters():
        if not value.requires_grad:
            continue
        if value.grad is None:
            missing.append(name)
        elif not torch.isfinite(value.grad).all():
            invalid.append(name)
        else:
            squared["encoder" if name.startswith("encoder.") else "head"] += float(value.grad.double().square().sum())
    context_ok = context_leaf.grad is not None and bool(torch.isfinite(context_leaf.grad).all())
    norm = float(context_leaf.grad.double().norm()) if context_ok else None
    passed = not missing and not invalid and all(value>0 and math.isfinite(value) for value in squared.values()) and context_ok and norm>0
    return {"passed":passed,"all_trainable_present_and_finite":not missing and not invalid,"missing":missing,"nonfinite":invalid,
        "encoder_norm":math.sqrt(squared["encoder"]),"head_norm":math.sqrt(squared["head"]),"context_leaf_norm":norm,
        "geometry_encodes":1,"original_context_backwards":1,"optimizer_steps":0,"zero_input_column_gradients_allowed":True}


def build_features(key,record,output,contract):
    cross,base = helpers(); output.mkdir(parents=True,exist_ok=True); destination = output/"manifest.json"
    config = {"contract_sha256":digest(contract),"key":key,"stage":"physical_signed_stress_inputs"}
    if destination.exists():
        saved = read(destination); require(saved.get("config") == config,"Own feature cache identity differs")
        if saved.get("status") == "completed":
            return saved
    begun = time.monotonic(); result = {"status":"running","config":config,"key":key,"node_count":record["node_count"],
        "feature_valid":False,"read_receipt":zero_receipt(),"costs":{},"arrays":{}}
    try:
        import numpy as np
        import torch
        from .conditioned_field import load_geometry_cloud
        from .elastic_pilot_data import _array_sha
        device = torch.device("cuda:0"); tick = time.monotonic()
        source = record["canonical_source"]; cross.match(source["path"],source)
        with np.load(path(source["path"]),allow_pickle=False) as archive:
            require(str(archive["sample_key"].item()) == key
                and str(archive["label_alignment"].item()) == "official CSV GRID coordinates -> H5 vertices","Canonical row identity differs")
            xyz,features,ids = (np.array(archive[name],dtype=dtype) for name,dtype in
                (("xyz",np.float32),("features",np.float32),("grid_node_ids",np.int64)))
        expected = record["physical_record"]["expected_geometry"]
        require(xyz.shape == (record["node_count"],3) and features.shape == (len(xyz),25) and ids.shape == (len(xyz),)
            and np.isfinite(xyz).all() and np.isfinite(features).all()
            and _array_sha(xyz,"<f4") == expected["canonical_xyz_float32_sha256"]
            and _array_sha(ids,"<i8") == expected["grid_node_ids_int64_sha256"],"Canonical input geometry/GRID identity differs")
        cloud = load_geometry_cloud(path(source["path"]),cloud_points=1024,seed=2026,cad_key=key.split("/")[0])
        arrays = {name:save_array(np,output/(name+".npy"),value) for name,value in
            (("xyz",xyz),("features",features),("cloud",cloud.points),("mask",cloud.mask),("grid_node_ids",ids))}
        neural_seconds = time.monotonic()-tick
        manifest = read(record["input_manifest"]["path"]); operator = read(record["operator_summary"]["path"])
        require(cross.sha(record["input_manifest"]["path"]) == record["input_manifest"]["sha256"]
            and cross.sha(record["operator_summary"]["path"]) == record["operator_summary"]["sha256"],"Accepted physical metadata changed")
        cross.match(manifest["cache_file"],manifest["file_identity"])
        tick = time.monotonic()
        with np.load(path(manifest["cache_file"]),allow_pickle=False) as stored:
            cells_cpu = np.array(stored["cells10"],dtype=np.int64)
        require(cells_cpu.shape == (expected["cells"],10) and cells_cpu.min()>=0 and cells_cpu.max()<len(xyz)
            and _array_sha(cells_cpu,cells_cpu.dtype) == manifest["array_schema"]["cells10"]["sha256"],"Accepted TET10 incidence identity differs")
        cells = torch.as_tensor(cells_cpu,device=device)
        incidence = torch.bincount(cells.flatten(),minlength=len(xyz)); valid = incidence>0
        base._sync(torch); arrays["valid"] = save_array(np,output/"valid.npy",valid.cpu().numpy())
        incidence_seconds = time.monotonic()-tick; tick = time.monotonic()
        with np.load(path(manifest["cache_file"]),allow_pickle=False) as stored:
            vertices_cpu,lame_cpu,grid_cpu = (np.array(stored[name]) for name in ("vertices_mm","lame","grid_node_ids"))
        for name,value in (("vertices_mm",vertices_cpu),("lame",lame_cpu),("grid_node_ids",grid_cpu)):
            require(_array_sha(value,value.dtype) == manifest["array_schema"][name]["sha256"],"Accepted physical input array changed: "+name)
        require(np.array_equal(ids,grid_cpu) and vertices_cpu.shape == xyz.shape and lame_cpu.shape == (2,),"Physical/canonical GRID row mismatch")
        physical = base._load_array(np,torch,record["physical_prediction"],device)
        gradient = base._load_array(np,torch,operator["arrays"]["gradients"],device)
        weights = base._load_array(np,torch,operator["arrays"]["weights"],device)
        vertices = torch.as_tensor(vertices_cpu,device=device,dtype=torch.float64)
        require(physical.shape == vertices.shape and physical.dtype == torch.float64 and gradient.shape == (len(cells),4,10,3)
            and weights.shape == (len(cells),4) and gradient.dtype == weights.dtype == torch.float64,"Physical proxy shapes/dtypes differ")
        if not (torch.isfinite(physical).all() and torch.isfinite(vertices).all() and torch.isfinite(gradient).all()
                and torch.isfinite(weights).all() and (weights>0).all() and np.isfinite(lame_cpu).all()):
            raise base.ScientificNegative("Nonfinite physical proxy or nonpositive accepted quadrature weights")
        lam,mu = map(float,lame_cpu); require(mu>0 and 3*lam+2*mu>0,"Accepted material positivity differs")
        nodal = torch.zeros((len(xyz),6),device=device,dtype=torch.float64)
        mass = torch.zeros(len(xyz),device=device,dtype=torch.float64); constant = torch.zeros_like(mass)
        eye = torch.eye(3,device=device,dtype=torch.float64)
        partition_max = affine_max = symmetry_max = 0.
        for start in range(0,len(cells),8192):
            c = cells[start:start+8192]; g = gradient[start:start+8192]; w = weights[start:start+8192]
            scale = torch.linalg.vector_norm(g.flatten(2),dim=-1)
            part = torch.linalg.vector_norm(g.sum(2),dim=-1)/scale
            affine = torch.einsum("cai,cqaj->cqij",vertices[c],g)
            partition_max = max(partition_max,float(part.max()))
            affine_max = max(affine_max,float(torch.linalg.matrix_norm(affine-eye,ord="fro").max()/math.sqrt(3)))
            derivative = torch.einsum("cai,cqaj->cqij",physical[c],g)
            strain = .5*(derivative+derivative.transpose(-1,-2))
            sigma = 2*mu*strain+lam*strain.diagonal(dim1=-2,dim2=-1).sum(-1)[...,None,None]*eye
            denominator = sigma.abs().amax(dim=(-1,-2)).clamp_min(1e-300)
            asymmetry = (sigma-sigma.transpose(-1,-2)).abs().amax(dim=(-1,-2))/denominator
            symmetry_max = max(symmetry_max,float(asymmetry.max()))
            six = torch.stack((sigma[...,0,0],sigma[...,1,1],sigma[...,2,2],sigma[...,0,1],sigma[...,0,2],sigma[...,1,2]),dim=-1)
            volume = w.sum(1); cell_mean = (six*w[...,None]).sum(1)/volume[:,None]
            incident = (volume/10)[:,None].expand(-1,10)
            mass.index_add_(0,c.flatten(),incident.flatten())
            constant.index_add_(0,c.flatten(),(incident*torch.ones_like(incident)).flatten())
            nodal.index_add_(0,c.flatten(),(cell_mean[:,None,:]*incident[...,None]).reshape(-1,6))
        covered = mass>0; coverage_same = bool(torch.equal(covered,valid))
        nodal[covered] /= mass[covered,None]; nodal[~covered] = 0
        constant_error = float((constant[covered]/mass[covered]-1).abs().max()) if covered.any() else None
        checked = {"all_finite":bool(torch.isfinite(nodal).all()),"positive_physical_weights":True,
            "partition_relative_error":partition_max,"affine_relative_error":affine_max,"tensor_symmetry_relative_error":symmetry_max,
            "constant_recovery_relative_error":constant_error,"exact_GRID_and_row_identity":True,"incidence_matches_positive_volume_coverage":coverage_same}
        passed = checked["all_finite"] and coverage_same and partition_max<=1e-9 and affine_max<=1e-9 and symmetry_max<=1e-12
        passed = passed and constant_error is not None and constant_error<=1e-12
        checked["passed"] = passed
        if not passed:
            raise base.ScientificNegative("Frozen actual physical-feature guards failed: "+json.dumps(checked))
        base._sync(torch); arrays["sigma_MPa"] = save_array(np,output/"sigma_MPa.npy",nodal.cpu().numpy())
        physical_sha = cross.sha(record["physical_prediction"]["path"])
        tensor_seconds = time.monotonic()-tick
        result.update(status="completed",feature_valid=True,arrays=arrays,feature_guards=checked,
            coverage={"all_original_nodes":len(xyz),"valid_nodes":int(valid.sum()),"missing_incident_nodes":int((~valid).sum()),
                "missing_policy":"valid=0, tensor slots zero; not a physical zero-stress claim"},cloud=cloud.record(),
            costs={"neural_input_preparation_seconds":neural_seconds,"incidence_mask_seconds":incidence_seconds,"tensor_recovery_seconds":tensor_seconds,
                "elapsed_seconds":time.monotonic()-begun},
            feature_source={"physical_prediction":record["physical_prediction"],"actual_physical_prediction_file_sha256":physical_sha,
                "operator_summary":record["operator_summary"],"input_manifest":record["input_manifest"],"physical_GT_free_cold_seconds":record["physical_GT_free_cold_seconds"]},
            accessed_members={"canonical_npz":list(INPUT_MEMBERS),"accepted_input_cache":["cells10","vertices_mm","lame","grid_node_ids"],
                "operator_arrays":["gradients","weights"],"numerical_prediction":"original zero-start physical_u; not GT"})
        result["read_receipt"].update(physical_prediction_array_reads=1,operator_grad_weight_array_reads=2)
        write(destination,result); return result
    except base.ScientificNegative as error:
        result.update(status="completed",feature_valid=False,reason=str(error),costs={"elapsed_seconds":time.monotonic()-begun})
        write(destination,result); return result
    except Exception as error:
        result.update(status="failed",reason=f"{type(error).__name__}: {error}",costs={"elapsed_seconds":time.monotonic()-begun})
        write(destination,result); raise


def read_label_once(key,scope,record,output,clock):
    """The transaction counts actual source member access, including interrupted reads."""
    import numpy as np
    directory = output/"label_cache"; directory.mkdir(exist_ok=True)
    stem = key.replace("/","_"); started_path = directory/(stem+"_read_started.json"); committed_path = directory/(stem+".json")
    identity = {"key":key,"scope":scope,"source":record["canonical_source"],"member":"stress"}
    if committed_path.exists():
        committed = read(committed_path); require(committed["identity"] == identity,"Own stress-label cache identity differs")
        return load_array(np,committed["array"])
    require(not started_path.exists(),"Interrupted stress member read has no committed cache; refuse an unrecorded repeated label read")
    if clock.remaining()<=0:
        return None
    helpers()[0].match(identity["source"]["path"],identity["source"])
    write(started_path,{"identity":identity,"member_read_attempted":False,"member_read_completed":False})
    tick = time.monotonic()
    with np.load(path(identity["source"]["path"]),allow_pickle=False) as archive:
        write(started_path,{"identity":identity,"member_read_attempted":True,"member_read_completed":False})
        truth = np.array(archive["stress"],dtype=np.float64)
    write(started_path,{"identity":identity,"member_read_attempted":True,"member_read_completed":True})
    require(truth.shape in ((record["node_count"],),(record["node_count"],1)) and np.isfinite(truth).all(),"Invalid source-signed scalar stress")
    truth = truth.reshape(-1); array = save_array(np,directory/(stem+".npy"),truth)
    write(committed_path,{"identity":identity,"array":array,"elapsed_seconds":time.monotonic()-tick})
    clock.account("shared_"+scope+"_stress_read",key)
    return truth


def label_receipt(output):
    result = zero_receipt()
    result["stress_member_reads_completed"] = 0
    for key in CASES:
        started = output/"label_cache"/(key.replace("/","_")+"_read_started.json")
        if started.exists():
            value = read(started); count = int(value.get("member_read_attempted") is True)
            result["TRAIN_stress_reads" if key in TRAIN else "DEV_stress_reads"] += count
            result["stress_member_reads_completed"] += int(value.get("member_read_completed") is True)
    result["stress_reads"] = result["GT_reads"] = result["TRAIN_stress_reads"]+result["DEV_stress_reads"]
    scaler_path = output/"scaler_fit_receipt.json"
    if scaler_path.exists():
        result["scaler_fits"] = read(scaler_path)["actual_fits"]
    return result


def fit_scalers(features,labels,output,clock,contract):
    import numpy as np
    target = output/"scalers.json"; config_sha = digest(contract)
    if target.exists():
        saved = read(target); require(saved["contract_sha256"] == config_sha,"Shared scaler identity differs")
        return saved
    require(not (output/"scaler_fit_receipt.json").exists(),"Interrupted scaler transaction; refuse silently fitting again")
    if clock.remaining()<=0:
        return None
    tick = time.monotonic(); receipt = {"contract_sha256":config_sha,"actual_fits":1,"fit_scope":list(TRAIN),"DEV_fits":0}
    write(output/"scaler_fit_receipt.json",receipt)
    tensors = [load_array(np,features[key]["arrays"]["sigma_MPa"])[load_array(np,features[key]["arrays"]["valid"])] for key in TRAIN]
    values = np.concatenate(tensors,axis=0); require(len(values)>0,"TRAIN physical proxy has no valid nodes")
    mean,std = values.mean(axis=0,dtype=np.float64),values.std(axis=0,dtype=np.float64)
    tensor = {"mean":mean.tolist(),"std":np.maximum(std,1e-8).tolist(),"raw_std":std.tolist(),"floor_MPa":1e-8,
        "channels":["xx","yy","zz","xy","xz","yz"],"valid_TRAIN_nodes":len(values),"fit_keys":list(TRAIN)}
    receipt["actual_fits"] = 2; write(output/"scaler_fit_receipt.json",receipt)
    truth = np.concatenate([labels[key] for key in TRAIN]); scale = max(float(np.quantile(np.abs(truth),.9)),1e-12)
    encoded = np.arcsinh(truth/scale); mean,std = float(encoded.mean()),float(encoded.std())
    stress = {"scale_MPa":scale,"mean":mean,"std":max(std,1e-8),"raw_std":std,"std_floor":1e-8,"scale_floor_MPa":1e-12,
        "fit_keys":list(TRAIN),"signed_transform":"(asinh(S/scale)-mean)/std","TRAIN_nodes":len(truth)}
    targets = {key:save_array(np,output/"label_cache"/(key.replace("/","_")+"_encoded.npy"),
        ((np.arcsinh(labels[key]/scale)-mean)/stress["std"]).astype(np.float32)) for key in TRAIN}
    saved = {"contract_sha256":config_sha,"tensor6":tensor,"signed_stress":stress,"targets":targets,"fit_seconds":time.monotonic()-tick}
    write(target,saved); clock.account("two_shared_TRAIN_only_scalers"); return saved


def load_case_inputs(np,torch,key,arm,features,scalers,training=False):
    arrays = features[key]["arrays"]; device = torch.device("cuda:0")
    def tensor(name,dtype=torch.float32):
        return torch.as_tensor(np.array(load_array(np,arrays[name]),copy=True),device=device,dtype=dtype)
    result = {"xyz":tensor("xyz"),"features":tensor("features"),"cloud":tensor("cloud")[None],
        "mask":tensor("mask",torch.bool)[None],"valid":tensor("valid")}
    if arm == "physical_tensor_slots":
        sigma = tensor("sigma_MPa",torch.float64)
        mean = torch.tensor(scalers["tensor6"]["mean"],device=device,dtype=torch.float64)
        std = torch.tensor(scalers["tensor6"]["std"],device=device,dtype=torch.float64)
        result["tensor6"] = torch.where(result["valid"][:,None].bool(),(sigma-mean)/std,0.).float()
    else:
        result["tensor6"] = torch.zeros((len(result["xyz"]),6),device=device)
    if training:
        result["target"] = torch.as_tensor(np.array(load_array(np,scalers["targets"][key]),copy=True),device=device)
    require(result["xyz"].shape[1] == 3 and result["features"].shape[1] == 25 and result["tensor6"].shape[1] == 6
        and result["cloud"].shape == (1,1024,3) and result["mask"].shape == (1,1024),"Scalar model input schema differs")
    return result


def atomic_checkpoint(torch,destination,saved):
    temporary = destination.with_name(destination.name+".tmp"); torch.save(saved,temporary); os.replace(temporary,destination)


def train_arm(arm,features,scalers,output,clock,contract,shared_initial=None):
    import numpy as np
    import torch
    cross,base = helpers(); directory = output/arm; directory.mkdir(exist_ok=True)
    config = {"contract_sha256":digest(contract),"arm":arm,"scaler_identity":digest(scalers),"case_order_sha256":contract["case_order_sha256"]}
    summary_path = directory/"training_summary.json"
    if summary_path.exists():
        saved = read(summary_path); require(saved["config"] == config,"Arm completed identity differs")
        if saved.get("status") in ("completed","incomplete"):
            if shared_initial is not None:
                require(saved["initial_parameters_sha256"] == shared_initial,"Two-arm initial states differ")
            return saved
    if (directory/"config.json").exists():
        require(read(directory/"config.json") == config,"Arm resume config differs")
    write(directory/"config.json",config); tick = time.monotonic(); model = build_model()
    initial = parameter_identity(model)
    require(shared_initial is None or initial["all_sha256"] == shared_initial,"Actual shared constructor/seed initial parameters differ")
    optimizer = torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.,fused=True)
    state = {"step":0,"train_seconds":0.,"actual_optimizer_steps":0,"attempted_updates":0,"node_exposures":0,
        "training_checkpoint_reads":0,"case_exposure_counts":{key:0 for key in TRAIN},"history":[],"first_actual_gradient_check":None,"pass_costs":[]}
    last = directory/"last.pt"; checkpoint_reads = 0
    if last.exists():
        saved = torch.load(last,map_location="cpu",weights_only=False); checkpoint_reads += 1
        require(saved["config"] == config and saved["initial_parameters_sha256"] == initial["all_sha256"],"Saved arm optimizer/model identity differs")
        model.load_state_dict(saved["model"],strict=True); optimizer.load_state_dict(saved["optimizer"]); state = saved["state"]
        restore_rng(np,torch,saved["rng"])
    if (directory/"status.json").exists():
        journal = read(directory/"status.json"); require(journal["config"] == config,"Arm cumulative cost journal differs")
        for name in ("train_seconds","actual_optimizer_steps","attempted_updates","training_checkpoint_reads"):
            state[name] = max(state[name],journal.get(name,0))
    state["training_checkpoint_reads"] += checkpoint_reads
    base._sync(torch); model_build_seconds = time.monotonic()-tick
    initial_path = directory/"initial_parameters.json"
    if initial_path.exists():
        previous = read(initial_path); require(previous["all_sha256"] == initial["all_sha256"],"Fresh initialization identity changed")
        model_build_seconds += previous.get("cumulative_model_build_seconds",0.)
    write(initial_path,{**initial,"cumulative_model_build_seconds":model_build_seconds})
    def status(phase):
        write(directory/"status.json",{"config":config,"phase":phase,"step":state["step"],"train_seconds":state["train_seconds"],
            "actual_optimizer_steps":state["actual_optimizer_steps"],"attempted_updates":state["attempted_updates"],
            "training_checkpoint_reads":state["training_checkpoint_reads"]})
        clock.account(phase,arm)
    def save():
        saved = {"config":config,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"state":state,
            "rng":rng_state(np,torch),"initial_parameters_sha256":initial["all_sha256"]}
        atomic_checkpoint(torch,last,saved)
        if state["step"] == 120:
            atomic_checkpoint(torch,directory/"step120.pt",saved)
    save(); reason = None; model.train(); active_began = None
    try:
        while state["step"]<120:
            if state["train_seconds"]>=300 or clock.remaining()<=0:
                reason = "arm_training_budget_limit" if state["train_seconds"]>=300 else "stage_wall_budget_limit"
                break
            key = contract["case_order"][state["step"]]; state["attempted_updates"] += 1; status("training_full_case")
            active_began = time.monotonic(); case = load_case_inputs(np,torch,key,arm,features,scalers,training=True)
            optimizer.zero_grad(set_to_none=True); context = model.encode_geometry(case["cloud"],case["mask"])
            leaf = context.detach().requires_grad_(True); nodes = len(case["xyz"]); total_loss = 0.
            for start in range(0,nodes,32768):
                stop = min(start+32768,nodes)
                prediction = model.forward_points(case["xyz"][start:stop],case["features"][start:stop],leaf,
                    case["tensor6"][start:stop],case["valid"][start:stop])
                loss = (prediction-case["target"][start:stop]).square().sum()/nodes
                if not torch.isfinite(loss):
                    raise base.ScientificNegative("Nonfinite complete-case signed-stress loss")
                total_loss += float(loss.detach()); loss.backward()
            if leaf.grad is None or not torch.isfinite(leaf.grad).all():
                raise base.ScientificNegative("Missing/nonfinite complete-case geometry-context gradient")
            context.backward(leaf.grad)
            if state["first_actual_gradient_check"] is None:
                check = gradient_check(torch,model,leaf); state["first_actual_gradient_check"] = check
                write(directory/"first_actual_gradient_check.json",check)
                if not check["passed"]:
                    raise base.ScientificNegative("Actual first whole-case gradient acceptance failed")
            elif any(value.grad is None or not torch.isfinite(value.grad).all() for value in model.parameters() if value.requires_grad):
                raise base.ScientificNegative("Missing/nonfinite active trainable gradient")
            optimizer.step(); state["actual_optimizer_steps"] += 1; base._sync(torch)
            if state["step"] == 0:
                state["first_actual_gradient_check"]["optimizer_steps"] = 1
                write(directory/"first_actual_gradient_check.json",state["first_actual_gradient_check"])
            duration = time.monotonic()-active_began; active_began = None
            state["train_seconds"] += duration; state["step"] += 1; state["node_exposures"] += nodes; state["case_exposure_counts"][key] += 1
            state["history"].append({"step":state["step"],"key":key,"encoded_scalar_MSE":total_loss,"train_seconds":state["train_seconds"],"case_seconds":duration})
            if state["step"]%3 == 0:
                prior = state["pass_costs"][-1]["cumulative_train_seconds"] if state["pass_costs"] else 0.
                state["pass_costs"].append({"pass":state["step"]//3,"cumulative_train_seconds":state["train_seconds"],"pass_train_seconds":state["train_seconds"]-prior})
            status("update_committed"); save()
        save(); status("completed" if state["step"] == 120 else "incomplete")
    except base.ScientificNegative as error:
        if active_began is not None:
            base._sync(torch); state["train_seconds"] += time.monotonic()-active_began; active_began = None
        reason = str(error); save(); status("scientific_negative")
    except Exception:
        if active_began is not None:
            base._sync(torch); state["train_seconds"] += time.monotonic()-active_began
        status("failed"); raise
    result = {"config":config,"status":"completed" if state["step"] == 120 else "incomplete","completed120":state["step"] == 120,
        "endpoint_step":state["step"],"reason":reason,"initial_parameters_sha256":initial["all_sha256"],"parameter_counts":initial["parameter_counts"],
        "first_actual_gradient_check":state["first_actual_gradient_check"],"case_order_sha256":contract["case_order_sha256"],
        "case_exposure_counts":state["case_exposure_counts"],"node_exposures":state["node_exposures"],"history":state["history"],"pass_costs":state["pass_costs"],
        "costs":{"train_seconds":state["train_seconds"],"initial_model_build_seconds":model_build_seconds,"training_checkpoint_reads_this_call":checkpoint_reads,
            "training_checkpoint_reads_cumulative":state["training_checkpoint_reads"],
            "train_budget_overshoot_seconds":max(0.,state["train_seconds"]-300)},"actual_optimizer_steps":state["actual_optimizer_steps"],
        "last_checkpoint":helpers()[0].identity(last),"memory":base._memory(torch),"validation":aggregate({},DEV),"training_readout":aggregate({},TRAIN)}
    write(summary_path,result); return result


def predict_arm(arm,training,features,scalers,output,clock,contract):
    import numpy as np
    import torch
    cross,base = helpers(); directory = output/arm; predictions_path = directory/"predictions.json"
    if predictions_path.exists():
        saved = read(predictions_path); require(saved["config"] == training["config"],"Endpoint predictions identity differs")
    else:
        saved = {"config":training["config"],"cases":{},"model_load_seconds":None,"endpoint_step":training["endpoint_step"]}
    if not training["completed120"]:
        return saved
    need_keys = [key for key in DEV+TRAIN if key not in saved["cases"]]
    if not need_keys or clock.remaining()<=0:
        return saved
    tick = time.monotonic(); model = build_model(); checkpoint = torch.load(directory/"last.pt",map_location="cpu",weights_only=False)
    require(checkpoint["config"] == training["config"] and checkpoint["state"]["step"] == 120,"Fixed120 endpoint checkpoint differs")
    model.load_state_dict(checkpoint["model"],strict=True); del checkpoint; model.eval(); base._sync(torch)
    duration = time.monotonic()-tick
    saved["model_load_seconds"] = duration if saved["model_load_seconds"] is None else saved["model_load_seconds"]
    saved["actual_cumulative_model_load_seconds"] = saved.get("actual_cumulative_model_load_seconds",0.)+duration
    saved["training_checkpoint_reads"] = saved.get("training_checkpoint_reads",0)+1; write(predictions_path,saved)
    for key in need_keys:
        if clock.remaining()<=0:
            break
        tick = time.monotonic(); case = load_case_inputs(np,torch,key,arm,features,scalers,training=False)
        base._sync(torch); input_seconds = time.monotonic()-tick; tick = time.monotonic()
        with torch.no_grad():
            context = model.encode_geometry(case["cloud"],case["mask"]); chunks = []
            for start in range(0,len(case["xyz"]),32768):
                stop = start+32768; encoded = model.forward_points(case["xyz"][start:stop],case["features"][start:stop],context,
                    case["tensor6"][start:stop],case["valid"][start:stop]).double()
                scalar = scalers["signed_stress"]; physical = torch.sinh(encoded*scalar["std"]+scalar["mean"])*scalar["scale_MPa"]
                if not torch.isfinite(physical).all():
                    raise base.ScientificNegative("Nonfinite fixed120 signed-stress prediction/FP64 decoding")
                chunks.append(physical.cpu().numpy())
        prediction = np.concatenate(chunks); require(prediction.shape == (contract["records"][key]["node_count"],),"Endpoint is not all original nodes")
        record = save_array(np,directory/(key.replace("/","_")+"_signed_stress_MPa.npy"),prediction)
        base._sync(torch); inference_seconds = time.monotonic()-tick
        source = features[key]["costs"]; tensor = arm == "physical_tensor_slots"
        components = {"physical_GT_free_cold_seconds":contract["records"][key]["physical_GT_free_cold_seconds"] if tensor else 0.,
            "tensor_recovery_seconds":source["tensor_recovery_seconds"] if tensor else 0.,"incidence_mask_seconds":source["incidence_mask_seconds"],
            "neural_input_preparation_seconds":source["neural_input_preparation_seconds"]+input_seconds,
            "model_load_seconds":saved["model_load_seconds"],"inference_decode_persistence_seconds":inference_seconds}
        complete = all(finite(value) and value>=0 for value in components.values())
        saved["cases"][key] = {"prediction":record,"num_nodes":len(prediction),"allN_prediction_persisted":True,"GT_read_before_prediction":False,
            "endpoint_step":120,"costs":{**components,"components_complete":complete,"total_seconds":sum(components.values()) if complete else None,
                "cached_input_load_H2D_seconds":input_seconds,"cached_incremental_inference_seconds":input_seconds+inference_seconds,
                "fresh_joint_cold_measurement":False,"physical_and_tensor_work_excluded_from_direct_deployment":not tensor}}
        write(predictions_path,saved); clock.account("fixed120_allN_prediction",arm+":"+key)
    return saved


def empty_arm(reason):
    return {"status":"incomplete","completed120":False,"endpoint_step":0,"reason":reason,"parameter_counts":{},
        "initial_parameters_sha256":None,"first_actual_gradient_check":None,"actual_optimizer_steps":0,
        "costs":{"train_seconds":0.,"inference_seconds":None,"model_load_seconds":None,"cold_deployment_per_case":{key:None for key in DEV}},
        "validation":aggregate({},DEV),"training_readout":aggregate({},TRAIN),"case_exposure_counts":{key:0 for key in TRAIN}}


def accuracy_gate(arms,receipt):
    direct,tensor = (arms[name] for name in ARMS)
    both = all(item.get("completed120") is True and item.get("endpoint_step") == 120
        and (item.get("first_actual_gradient_check") or {}).get("passed") is True for item in arms.values())
    both = both and direct.get("initial_parameters_sha256") is not None and direct.get("initial_parameters_sha256") == tensor.get("initial_parameters_sha256")
    both = both and receipt["optimizer_steps"] == 240 and receipt["scaler_fits"] == 2
    d,t = direct["validation"],tensor["validation"]
    values = [(t["per_case"].get(key) or {}).get("stress_relative_l2") for key in DEV]
    each = all(finite(value) and value<.25 for value in values)
    def improvement(dvalue,tvalue):
        return (dvalue-tvalue)/dvalue if finite(dvalue) and finite(tvalue) and dvalue>0 else None
    stress_gain = improvement(d.get("stress_relative_l2_mean"),t.get("stress_relative_l2_mean"))
    hotspot_gain = improvement(d.get("hotspot_relative_l2_mean"),t.get("hotspot_relative_l2_mean"))
    dworst,tworst = d.get("stress_relative_l2_worst"),t.get("stress_relative_l2_worst")
    worst = tworst/dworst if finite(dworst) and finite(tworst) and dworst>0 else None
    complete = both and d["metrics_available_cases"] == 2 and t["metrics_available_cases"] == 2
    passed = complete and each and finite(stress_gain) and stress_gain>=.2 and finite(worst) and worst<=1.05 and finite(hotspot_gain) and hotspot_gain>=.1
    return {"passed":passed,"required_DEV_denominator":2,"both_arms_complete120":both,"all_DEV_metrics_available":complete,
        "tensor_each_DEV_below25pct":each,"mean_stress_reduction":stress_gain,"worst_tensor_to_direct_ratio":worst,
        "mean_hotspot_reduction":hotspot_gain,"missing_or_zero_reference_relative_metrics_never_pass":True}


def all_dev_predictions(predictions,arms,contract):
    base = helpers()[1]
    for arm in ARMS:
        if not arms[arm]["completed120"]:
            return False
        saved = predictions.get(arm,{})
        for key in DEV:
            case = saved.get("cases",{}).get(key)
            if case is None or case.get("endpoint_step") != 120 or case.get("allN_prediction_persisted") is not True:
                return False
            value = case["prediction"]
            require(value["shape"] == [contract["records"][key]["node_count"]] and value["dtype"] == "float64"
                and base.stat_record(value["path"]) == {name:value[name] for name in ("bytes","mtime_ns")},"Deferred DEV read requires allN prediction stat/schema")
    return True


def actual_optimizer_steps(arms,output):
    total = 0
    for arm in ARMS:
        count = arms[arm].get("actual_optimizer_steps",0); journal = output/arm/"status.json"
        if journal.exists():
            count = max(count,read(journal).get("actual_optimizer_steps",0))
        total += count
    return total


def finalize_arms(np,arms,predictions,features,labels,dev_labels):
    for arm in ARMS:
        saved = predictions.get(arm,{"cases":{}}); per_case = {}; cached = saved["cases"]
        for key in CASES:
            truth = labels.get(key) if key in TRAIN else dev_labels.get(key)
            if key in cached and truth is not None:
                prediction = load_array(np,cached[key]["prediction"])
                per_case[key] = signed_metrics(np,truth,prediction)
        arms[arm]["validation"] = aggregate(per_case,DEV); arms[arm]["training_readout"] = aggregate(per_case,TRAIN)
        arms[arm]["predictions"] = cached
        arms[arm]["costs"].update(inference_seconds=sum(item["costs"]["inference_decode_persistence_seconds"] for item in cached.values()),
            model_load_seconds=saved.get("model_load_seconds"),cold_deployment_per_case={key:cached.get(key,{}).get("costs") for key in DEV},
            actual_model_load_seconds_including_resumes=saved.get("actual_cumulative_model_load_seconds"),
            cached_incremental_inference_seconds=sum(item["costs"]["cached_incremental_inference_seconds"] for item in cached.values()))
    return arms


def run(plan_path=PLAN,output_path=None):
    began = time.monotonic(); contract,plan = request_contract(plan_path); cross,base = helpers()
    require(plan.get("scientific_rules_frozen") is True and plan.get("actual_execution_released") is True,
        "Root must release frozen stress screen before scientific imports/arrays/labels/model/GPU")
    require(set(plan.get("source_sha256",{})) == set(SOURCES),"Actual stage needs all five source pins")
    review = read(plan["root_review"])
    require(review.get("passed") is True and review.get("source_sha256") == contract["source_sha256"][SOURCE]
        and review.get("stage_decision") == "rough_screen_physical_tensor_stress_readout","Precise root source review/stage decision required")
    output = path(output_path or plan["output"]); require(output == path(plan["output"]),"Frozen output destination differs")
    config = {"contract":contract,"stage":KIND}
    if (output/"config.json").exists():
        require(read(output/"config.json") == config,"Existing stress-screen identity differs")
    if (output/"summary.json").exists():
        previous = read(output/"summary.json"); require(previous.get("config") == config,"Existing summary identity differs")
        if previous.get("status") == "completed":
            return previous
    output.mkdir(parents=True,exist_ok=True); write(output/"config.json",config)
    clock = cross.StageClock(output,config,began); clock.account("starting")
    features = {}; labels = {}; dev_labels = {}; scalers = None; predictions = {}; arms = {arm:empty_arm("not_started") for arm in ARMS}
    science_negative = None
    try:
        import numpy as np
        import torch
        torch.cuda.init(); torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.set_float32_matmul_precision("high")
        inputs_directory = output/"input_cache"; inputs_directory.mkdir(exist_ok=True)
        for key in CASES:
            own_manifest = inputs_directory/key.replace("/","_")/"manifest.json"
            reusable = own_manifest.exists() and read(own_manifest).get("status") == "completed"
            if clock.remaining()<=0 and not reusable:
                science_negative = "stage_budget_before_complete_input_features"; break
            clock.account("target_free_actual_feature_build",key)
            feature = build_features(key,contract["records"][key],inputs_directory/key.replace("/","_"),contract)
            features[key] = feature; clock.account("actual_features_committed",key)
            if not feature["feature_valid"]:
                science_negative = "actual_physical_feature_guards_failed"; break
        if len(features) == 5 and all(item["feature_valid"] for item in features.values()):
            for key in TRAIN:
                value = read_label_once(key,"TRAIN",contract["records"][key],output,clock)
                if value is None:
                    science_negative = "stage_budget_before_all_TRAIN_labels"; break
                labels[key] = value
            if len(labels) == 3:
                scalers = fit_scalers(features,labels,output,clock,contract)
                if scalers is None:
                    science_negative = "stage_budget_before_shared_scalers"
                else:
                    initial = None
                    for arm in ARMS:
                        prior_arm = (output/arm/"last.pt").exists() or (output/arm/"training_summary.json").exists()
                        if clock.remaining()<=0 and not prior_arm:
                            arms[arm] = empty_arm("stage_budget_before_arm"); continue
                        clock.account("train_arm",arm)
                        arms[arm] = train_arm(arm,features,scalers,output,clock,contract,initial)
                        if initial is None:
                            initial = arms[arm]["initial_parameters_sha256"]
                        clock.account("fixed120_endpoint_inference",arm)
                        predictions[arm] = predict_arm(arm,arms[arm],features,scalers,output,clock,contract)
                    if all_dev_predictions(predictions,arms,contract):
                        proof = {"config_sha256":digest(config),"both_arms_completed120":True,
                            "all_four_allN_DEV_predictions_persisted":True,"predictions":{arm:{key:predictions[arm]["cases"][key]["prediction"] for key in DEV} for arm in ARMS},
                            "DEV_stress_reads_before_proof":label_receipt(output)["DEV_stress_reads"]}
                        existing_proof = output/"DEV_prediction_before_GT.json"
                        if existing_proof.exists():
                            require(read(existing_proof)["predictions"] == proof["predictions"],"Shared DEV prediction proof differs")
                        else:
                            require(proof["DEV_stress_reads_before_proof"] == 0,"DEV stress exposed before both-arm prediction proof")
                            write(existing_proof,proof)
                        for key in DEV:
                            value = read_label_once(key,"DEV",contract["records"][key],output,clock)
                            if value is not None:
                                dev_labels[key] = value
                    else:
                        science_negative = "incomplete_fixed120_or_missing_allN_DEV_predictions_no_DEV_stress_read"
    except base.ScientificNegative as error:
        science_negative = str(error)
    except Exception as error:
        clock.account("failed"); receipt = label_receipt(output)
        receipt["optimizer_steps"] = actual_optimizer_steps(arms,output)
        write(output/"summary.json",{"status":"failed","kind":KIND,"config":config,"train_keys":list(TRAIN),"validation_keys":list(DEV),
            "arms":arms,"cases":features,"reason":f"{type(error).__name__}: {error}","read_receipt":receipt,
            "costs":{"cumulative_stage_wall_seconds":clock.elapsed()}})
        raise
    # Metrics only use already-read shared labels and already-committed allN predictions.
    metrics_began = time.monotonic()
    try:
        for arm in ARMS:
            prediction_path = output/arm/"predictions.json"
            if arm not in predictions and prediction_path.exists():
                predictions[arm] = read(prediction_path)
        if predictions:
            finalize_arms(np,arms,predictions,features,labels,dev_labels)
    except Exception as error:
        clock.account("failed_metrics"); receipt = label_receipt(output); receipt["optimizer_steps"] = actual_optimizer_steps(arms,output)
        write(output/"summary.json",{"status":"failed","kind":KIND,"config":config,"train_keys":list(TRAIN),"validation_keys":list(DEV),
            "arms":arms,"cases":features,"reason":f"{type(error).__name__}: {error}","read_receipt":receipt,
            "costs":{"cumulative_stage_wall_seconds":clock.elapsed()}})
        raise
    metrics_seconds = time.monotonic()-metrics_began
    receipt = label_receipt(output); receipt["optimizer_steps"] = actual_optimizer_steps(arms,output)
    receipt["all_four_allN_DEV_predictions_before_DEV_stress"] = (output/"DEV_prediction_before_GT.json").exists()
    receipt["training_checkpoint_reads"] = sum(item.get("costs",{}).get("training_checkpoint_reads_cumulative",0)
        +predictions.get(arm,{}).get("training_checkpoint_reads",0) for arm,item in arms.items())
    gate = accuracy_gate(arms,receipt); elapsed = clock.elapsed()
    cases = {key:features.get(key,{"key":key,"status":"completed","feature_valid":False,"attempted":False,"reason":"stage_budget_or_shared_input_negative"}) for key in CASES}
    summary = {"status":"completed","kind":KIND,"config":config,"train_keys":list(TRAIN),"validation_keys":list(DEV),
        "arms":arms,"cases":cases,"scalers":scalers,"read_receipt":receipt,"selection_gate":gate,"elapsed_seconds":elapsed,
        "case_order_sha256":contract["case_order_sha256"],"scientific_outcome":"stress_readout_accuracy_gate_passed" if gate["passed"] else "stress_readout_accuracy_gate_failed",
        "scientific_negative_reason":science_negative,"costs":{"cumulative_stage_wall_seconds":elapsed,"maximum_stage_wall_seconds":900,
            "stage_budget_overshoot_seconds":max(0.,elapsed-900),"this_call_wall_seconds":time.monotonic()-began,
            "shared_feature_preparation_seconds":sum(item.get("costs",{}).get("elapsed_seconds",0.) for item in features.values()),
            "shared_TRAIN_scaler_fit_seconds":None if scalers is None else scalers["fit_seconds"],"post_read_metrics_seconds":metrics_seconds},
        "memory":base._memory(torch),"limits":["TRAIN3/DEV2/ver only; rough information screen, not geometry/load generalization",
            "Original signed scalar stress is learned; constitutive tensor is only an approximate physical proxy",
            "Cold deployment ledger is accounted components, not a fresh joint cold measurement",
            "Direct deployment excludes physical solve and tensor recovery, but includes independently measured topology validity mask",
            "Original physical displacement U accuracy is not recomputed or assigned as scalar-model accuracy",
            "No OptiStruct exactness, engineering qualification, speed superiority or final four-field solver claim"],
        "two_fresh_arms_same_actual_initial_parameters":arms[ARMS[0]].get("initial_parameters_sha256") is not None
            and arms[ARMS[0]].get("initial_parameters_sha256") == arms[ARMS[1]].get("initial_parameters_sha256"),
        "budget_finalization":"Incomplete arms retain actual step/checkpoint; no new case/update after expiry; finite endpoint overshoot explicit",
        "DEV_label_policy":"Two arms times two DEV allN physical predictions persisted before one shared read per DEV stress member"}
    write(output/"summary.json",summary); clock.account("completed"); return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan",type=Path,default=Path(PLAN)); result.add_argument("--output",type=Path)
    return result


if __name__ == "__main__":
    args = parser().parse_args(); run(args.plan,args.output)
