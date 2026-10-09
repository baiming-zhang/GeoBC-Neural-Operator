"""Frozen tensor-only signed factor: S_hat = constitutive VM * tanh(logit).

Metadata uses only the standard library. Released actual execution reads only
the pinned parent sigma/valid/signed-S caches and this stage's own checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "geobc/elastic_factored_stress.py"
PLAN = "notes/geobc_factored_stress_plan.json"
KIND = "elastic_factored_signed_stress_screen"
TRAIN = ("101_428/ver", "101_56/ver", "101_78/ver")
DEV = ("102_249/ver", "102_38/ver")
KEYS = TRAIN + DEV
ARRAYS = ("sigma_MPa", "valid", "signed_stress")
ZERO_READS = ("source_stress_member_reads", "canonical_archive_reads", "displacement_reads", "old_final_reads", "DEV_dia_reads",
    "sealed_new_test_reads", "external_model_or_weight_reads", "new_physical_solves", "K_guards_repeated", "scaler_fits")


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


def write(path, data):
    path = host_path(path); temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(host_path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def stat_ok(record):
    stat = host_path(record["path"]).stat()
    return stat.st_size == record["bytes"] and stat.st_mtime_ns == record["mtime_ns"]


def pinned_json(record):
    require(host_path(record["path"]).stat().st_size == record["bytes"] and sha(record["path"]) == record["sha256"], "Pinned JSON identity differs")
    return read(record["path"])


def request_contract(plan_path=PLAN):
    """Pinned JSON/source hashes and file stats only; no scientific imports."""
    plan = read(plan_path); recipe = plan["recipe"]
    require(plan["kind"] == KIND and tuple(plan["train_keys"]) == TRAIN and tuple(plan["validation_keys"]) == DEV
        and set(plan["records"]) == set(KEYS), "Fixed TRAIN3/DEV2 scope differs")
    fixed = {"input_dim":7, "seed":2026, "classification_weight":.1, "learning_rate":.001, "weight_decay":0., "passes":20,
        "full_case_updates":60, "point_chunk":32768, "maximum_training_wall_seconds":120, "maximum_cumulative_stage_wall_seconds":180,
        "unit":"MPa", "tensor_shear":True, "optimizer":"Adam", "model":"Fresh MLP7->64 ReLU->64 ReLU->1 logit",
        "q_formula":"sqrt(0.5*((xx-yy)^2+(yy-zz)^2+(zz-xx)^2)+3*(xy^2+xz^2+yz^2))"}
    require(all(recipe[key] == value for key, value in fixed.items()), "Frozen architecture/optimization/budget differs")
    require(plan["read_policy"]["allowed_arrays_per_case"] == list(ARRAYS) and all(plan["read_policy"][key] == 0 for key in ZERO_READS), "Cache-only read policy differs")
    gate = plan["selection_gate"]
    require(gate["complete_actual_updates"] == 60 and gate["each_DEV_signed_relativeL2_strict_below"] == .25
        and gate["DEV_mean_hotspot_relativeL2_strict_below"] == .25 and gate["each_DEV_S_squared_weighted_sign_error_at_most"] == .05
        and gate["null_missing_incomplete_never_pass"] is True, "Frozen DEV gate differs")
    parent = pinned_json(plan["parent_summary"]); parent_dir = host_path(plan["parent_summary"]["path"]).parent
    require(parent["status"] == "completed" and parent["kind"] == "elastic_signed_stress_readout_screen"
        and tuple(parent["train_keys"]) == TRAIN and tuple(parent["validation_keys"]) == DEV, "Parent completed scope differs")
    require(parent["scalers"]["tensor6"] == plan["inherited_tensor_scaler"] and tuple(plan["inherited_tensor_scaler"]["fit_keys"]) == TRAIN
        and len(plan["inherited_tensor_scaler"]["mean"]) == len(plan["inherited_tensor_scaler"]["std"]) == 6
        and all(math.isfinite(value) for value in plan["inherited_tensor_scaler"]["mean"])
        and all(math.isfinite(value) and value>0 for value in plan["inherited_tensor_scaler"]["std"]), "Exact inherited TRAIN-only tensor scaler differs")
    magnitude_plan = pinned_json(plan["magnitude_plan"]); magnitude = pinned_json(plan["magnitude_summary"])
    require(magnitude["status"] == "completed" and magnitude["kind"] == "elastic_stress_magnitude_diagnostic"
        and magnitude_plan["records"] == plan["records"] and set(magnitude["cases"]) == set(KEYS), "Accepted fixed-q provenance differs")
    for key in KEYS:
        record = plan["records"][key]; stem = key.replace("/", "_")
        feature = pinned_json(record["feature_manifest"]); committed = pinned_json(record["label_committed"]); started = pinned_json(record["label_read_started"])
        require(record["group"] == ("TRAIN" if key in TRAIN else "DEV") and feature == parent["cases"][key]
            and feature["feature_valid"] is True and feature["feature_guards"]["passed"] is True and feature["node_count"] == record["node_count"], "Feature provenance differs")
        require(committed["identity"] == started["identity"] == record["label_identity"] and started["member_read_attempted"] is True
            and started["member_read_completed"] is True and record["label_identity"]["member"] == "stress"
            and record["label_identity"]["key"] == key and record["label_identity"]["scope"] == record["group"], "Committed signed-S provenance differs")
        require(feature["arrays"]["sigma_MPa"] == record["sigma_MPa"] and feature["arrays"]["valid"] == record["valid"]
            and committed["array"] == record["signed_stress"], "Exact cached array records differ")
        for name in ARRAYS:
            item = record[name]; expected = parent_dir/"label_cache"/(stem+".npy") if name == "signed_stress" else parent_dir/"input_cache"/stem/(name+".npy")
            require(host_path(item["path"]) == expected and stat_ok(item) and len(item["values_sha256"]) == 64
                and item["shape"] == ([record["node_count"],6] if name == "sigma_MPa" else [record["node_count"]])
                and item["dtype"] == ("bool" if name == "valid" else "float64"), "Pinned cached path/stat/schema differs")
        cold = plan["inherited_cold_components"][key]
        require(all(math.isfinite(cold[name]) and cold[name]>=0 for name in ("physical_GT_free_cold_seconds", "sigma_recovery_seconds", "validity_incidence_seconds")), "Inherited cost is invalid")
    science = {name:plan[name] for name in ("recipe", "selection_gate", "read_policy", "inherited_tensor_scaler", "inherited_cold_components")}
    source_sha = sha(SOURCE)
    require(all(name == SOURCE and value == source_sha for name, value in plan.get("source_sha256", {}).items()), "Only exact new source pin allowed")
    return {"schema":"geobc_elastic_factored_stress_v1", "kind":KIND, "train_keys":list(TRAIN), "validation_keys":list(DEV),
        "parent_summary":plan["parent_summary"], "magnitude_plan":plan["magnitude_plan"], "magnitude_summary":plan["magnitude_summary"],
        "records":plan["records"], "plan_science":science, "science_recipe_sha256":digest(science), "source_sha256":{SOURCE:source_sha},
        "output":str(host_path(plan["output"])), "array_or_GT_reads_in_request_contract":0}, plan


class Clock:
    def __init__(self, output, config, began):
        self.output, self.config, self.began = output, config, began
        if (output/"state.json").exists():
            self.state = read(output/"state.json"); require(self.state["config"] == config, "Resume state identity differs")
        else:
            schedule = []; sampler = random.Random(2026)
            for _ in range(20):
                order = list(TRAIN); sampler.shuffle(order); schedule.extend(order)
            self.state = {"config":config, "step":0, "case_order":schedule, "case_order_sha256":digest(schedule), "cases":{},
                "reads":{key:{name:{"attempted":0,"completed":0} for name in ARRAYS} for key in KEYS},
                "own_reads":{"checkpoint_attempted":0,"checkpoint_completed":0,"head_attempted":0,"head_completed":0,"prediction_attempted":0,"prediction_completed":0},
                "optimizer_step_attempts":0,"costs":{"cumulative_stage_wall_seconds":0.,"training_wall_seconds":0.,"checkpoint_seconds":0.,"cached_signed_S_read_seconds":0.}}
        self.prior = self.state["costs"]["cumulative_stage_wall_seconds"]
        require(math.isfinite(self.prior) and self.prior>=0 and len(self.state["case_order"]) == 60
            and self.state["case_order_sha256"] == digest(self.state["case_order"]), "Invalid cumulative clock/schedule")
    def elapsed(self):
        return self.prior+time.monotonic()-self.began
    def remaining(self):
        return max(0.,180-self.elapsed())
    def touch(self, phase):
        self.state["costs"]["cumulative_stage_wall_seconds"] = self.elapsed()
        write(self.output/"state.json", self.state)
        write(self.output/"status.json", {"status":"running", "phase":phase, "step":self.state["step"], "costs":self.state["costs"]})


def read_receipt(state):
    counts = state["reads"]; attempts = {name:sum(counts[key][name]["attempted"] for key in KEYS) for name in ARRAYS}
    return {**{name:0 for name in ZERO_READS}, "GT_reads":attempts["signed_stress"], "stress_reads":attempts["signed_stress"],
        "cached_signed_stress_reads":attempts["signed_stress"], "TRAIN_stress_reads":sum(counts[key]["signed_stress"]["attempted"] for key in TRAIN),
        "DEV_stress_reads":sum(counts[key]["signed_stress"]["attempted"] for key in DEV), "sigma_array_reads":attempts["sigma_MPa"], "valid_array_reads":attempts["valid"],
        "attempted_array_reads":attempts, "completed_array_reads":{name:sum(counts[key][name]["completed"] for key in KEYS) for name in ARRAYS},
        "per_case":counts, "own_stage_reads":state["own_reads"], "optimizer_steps":state["step"], "optimizer_step_attempts":state["optimizer_step_attempts"], "training_updates":state["step"]}


class BudgetEnd(Exception):
    pass


class ScientificNegative(Exception):
    pass


def cached_array(np, clock, key, name, record):
    if clock.remaining()<=0:
        raise BudgetEnd("180s stage budget before cached read")
    if key in DEV and name == "signed_stress":
        require(clock.state["step"] == 60 and all(clock.state["cases"].get(item, {}).get("prediction_committed") is True for item in DEV), "Both full DEV predictions must persist before either DEV cached-S read")
    item = record[name]; require(stat_ok(item), "Cached input stat differs before read")
    counts = clock.state["reads"][key][name]; counts["attempted"] += 1; clock.touch("cached_read_attempt:"+key+":"+name)
    array = np.load(host_path(item["path"]), allow_pickle=False)
    counts["completed"] += 1; clock.touch("cached_read_completed:"+key+":"+name)
    require(list(array.shape) == item["shape"] and str(array.dtype) == item["dtype"]
        and hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() == item["values_sha256"], "Loaded cache value/schema differs")
    return array


def prepare_case(np, torch, clock, key, record, scaler):
    began = time.monotonic()
    sigma = torch.as_tensor(cached_array(np, clock, key, "sigma_MPa", record), device="cuda:0")
    valid = torch.as_tensor(cached_array(np, clock, key, "valid", record), device="cuda:0")
    require(sigma.dtype == torch.float64 and valid.dtype == torch.bool and torch.isfinite(sigma).all(), "Nonfinite sigma or invalid dtype")
    mean = torch.tensor(scaler["mean"], dtype=torch.float64, device=sigma.device); std = torch.tensor(scaler["std"], dtype=torch.float64, device=sigma.device)
    normalized = (sigma-mean)/std; normalized[~valid] = 0
    inputs = torch.cat((normalized.float(), valid.float().unsqueeze(1)), dim=1)
    xx,yy,zz,xy,xz,yz = sigma.unbind(1)
    q = torch.sqrt(.5*((xx-yy).square()+(yy-zz).square()+(zz-xx).square())+3*(xy.square()+xz.square()+yz.square()))
    require(torch.isfinite(inputs).all() and torch.isfinite(q).all() and (q>=0).all(), "Nonfinite standardized input/fixed VM")
    torch.cuda.synchronize(); seconds = time.monotonic()-began
    clock.state["cases"].setdefault(key, {"key":key,"group":record["group"],"num_nodes":record["node_count"],"metrics":None})
    clock.state["cases"][key]["input_preparation_seconds"] = seconds
    clock.state["cases"][key]["valid_nodes"] = int(valid.sum()); clock.touch("input_prepared:"+key)
    return {"inputs":inputs, "q":q, "valid":valid}


def build_head(torch):
    return torch.nn.Sequential(torch.nn.Linear(7,64), torch.nn.ReLU(), torch.nn.Linear(64,64), torch.nn.ReLU(), torch.nn.Linear(64,1)).cuda()


def gradient_check(torch, model):
    rows = {}; energy = 0.
    for name, parameter in model.named_parameters():
        require(parameter.grad is not None and torch.isfinite(parameter.grad).all(), "Missing/nonfinite first actual gradient: "+name)
        norm = float(parameter.grad.double().square().sum()); rows[name] = math.sqrt(norm); energy += norm
    require(math.isfinite(energy) and energy>0, "First actual fullcase gradient must be finite and nonzero")
    return {"passed":True, "all_named_parameters_present_finite":True, "global_gradient_norm":math.sqrt(energy), "named_gradient_norms":rows}


def save_checkpoint(torch, np, clock, model, optimizer):
    began = time.monotonic(); temporary = clock.output/"last.pt.tmp"
    torch.save({"config":clock.config,"step":clock.state["step"],"case_order":clock.state["case_order"],"model":model.state_dict(),
        "optimizer":optimizer.state_dict(), "state":copy.deepcopy(clock.state), "rng":{"python":random.getstate(),"numpy":np.random.get_state(),
        "torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}}, temporary)
    os.replace(temporary, clock.output/"last.pt")
    clock.state["costs"]["checkpoint_seconds"] += time.monotonic()-began; clock.touch("checkpoint_committed")


def own_load(torch, clock, path, kind, weights_only=False):
    if clock.remaining()<=0:
        raise BudgetEnd("Stage budget before own checkpoint/head read")
    counts = clock.state["own_reads"]; counts[kind+"_attempted"] += 1; clock.touch(kind+"_read_attempt")
    payload = torch.load(path, map_location="cuda:0", weights_only=weights_only)
    counts[kind+"_completed"] += 1; clock.touch(kind+"_read_completed")
    return payload


def save_array(np, path, values):
    with path.with_name(path.name+".tmp").open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    os.replace(path.with_name(path.name+".tmp"), path); stat = path.stat()
    return {"path":str(path),"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns,"shape":list(values.shape),"dtype":str(values.dtype),
        "values_sha256":hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()}


def predict(np, torch, clock, model, key, case, record, head_load_seconds, common_metadata_seconds, cold):
    output = clock.output/key.replace("/", "_"); output.mkdir(exist_ok=True); prediction, logits = [], []; began = time.monotonic()
    model.eval()
    with torch.no_grad():
        for start in range(0, record["node_count"], 32768):
            if clock.remaining()<=0:
                raise BudgetEnd("180s stage budget during allN prediction; DEV labels remain closed")
            logit = model(case["inputs"][start:start+32768]).squeeze(1)
            value = case["q"][start:start+32768]*torch.tanh(logit.double())
            require(torch.isfinite(logit).all() and torch.isfinite(value).all(), "Nonfinite factored prediction")
            logits.append(logit); prediction.append(value)
    torch.cuda.synchronize(); inference_seconds = time.monotonic()-began; tick = time.monotonic()
    full_prediction = torch.cat(prediction); full_logit = torch.cat(logits)
    files = {"prediction_MPa":save_array(np, output/"prediction_MPa.npy", full_prediction.cpu().numpy()),
        "logit":save_array(np, output/"logit.npy", full_logit.cpu().numpy())}
    persist_seconds = time.monotonic()-tick
    components = {"physical_GT_free_cold_seconds":cold["physical_GT_free_cold_seconds"],"sigma_recovery_seconds":cold["sigma_recovery_seconds"],
        "validity_incidence_seconds":cold["validity_incidence_seconds"],"input_cache_identity_H2D_normalization_q_seconds":clock.state["cases"][key]["input_preparation_seconds"],
        "head_load_seconds":head_load_seconds,"inference_seconds":inference_seconds,"prediction_persistence_seconds":persist_seconds,"common_metadata_seconds":common_metadata_seconds}
    increment = sum(components[name] for name in ("input_cache_identity_H2D_normalization_q_seconds","head_load_seconds","inference_seconds","prediction_persistence_seconds","common_metadata_seconds"))
    costs = {**components,"accounted_cold_seconds":sum(components.values()),"cached_increment_seconds":increment,"components_complete":True,
        "fresh_combined_cold_remeasured":False,"unneeded_xyz_25_cloud_preparation_excluded_seconds":cold["excluded_unneeded_neural_input_preparation_seconds"],
        "charge_convention":"Unamortized single request: common metadata/runtime startup and head load measured once, charged in full per case; no fresh combined measurement"}
    result = clock.state["cases"][key]; result.update(status="prediction_committed",prediction_committed=True,prediction_step=60,files=files,costs=costs,
        allN_prediction_persisted_before_DEV_cached_S=True)
    write(output/"prediction_committed.json", {"config_sha256":digest(clock.config),"key":key,"step":60,"files":files}); clock.touch("allN_prediction_committed:"+key)
    return full_prediction, full_logit


def reload_prediction(np, torch, clock, key):
    values = []
    for name in ("prediction_MPa", "logit"):
        if clock.remaining()<=0:
            raise BudgetEnd("180s budget before own prediction read")
        item = clock.state["cases"][key]["files"][name]; require(stat_ok(item), "Own persisted prediction stat differs")
        clock.state["own_reads"]["prediction_attempted"] += 1; clock.touch("own_prediction_read_attempt")
        value = np.load(host_path(item["path"]), allow_pickle=False)
        clock.state["own_reads"]["prediction_completed"] += 1; clock.touch("own_prediction_read_completed")
        require(list(value.shape) == item["shape"] and str(value.dtype) == item["dtype"]
            and hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == item["values_sha256"], "Own prediction value identity differs")
        values.append(torch.as_tensor(value, device="cuda:0"))
    return values


def metrics(torch, prediction, logit, q, target):
    require(torch.isfinite(target).all(), "Nonfinite original signed S")
    energy = float(target.square().sum()); delta = prediction-target; error = float(delta.square().sum()); count = len(target)
    hot = torch.argsort(target.abs(), descending=True, stable=True)[:math.ceil(.01*count)]
    hot_energy = float(target[hot].square().sum()); hot_error = float(delta[hot].square().sum())
    sign_wrong = (target!=0)&((q==0)|(logit==0)|(torch.sign(logit)!=torch.sign(target)))
    sign_error = float(target[sign_wrong].square().sum()); floor_energy = float(torch.relu(target.abs()-q).square().sum())
    require(all(math.isfinite(value) for value in (energy,error,hot_energy,hot_error,sign_error,floor_energy)), "Nonfinite metric energy")
    relative = lambda numerator, denominator: None if denominator == 0 else math.sqrt(numerator)/math.sqrt(denominator)
    result = {"signed_relative_l2":relative(error,energy),"RMSE_MPa":math.sqrt(error/count),"MAE_MPa":float(delta.abs().mean()),
        "max_abs_error_MPa":float(delta.abs().max()),"hotspot_relative_l2":relative(hot_error,hot_energy),"hotspot_RMSE_MPa":math.sqrt(hot_error/len(hot)),
        "hotspot_MAE_MPa":float(delta[hot].abs().mean()),"hotspot_max_abs_error_MPa":float(delta[hot].abs().max()),
        "S_squared_weighted_sign_error":None if energy == 0 else sign_error/energy,"range_floor_relative_l2":relative(floor_energy,energy),
        "zero_reference":energy == 0,"hotspot_zero_reference":hot_energy == 0,"num_nodes":count,"hotspot_nodes":len(hot),
        "hotspot_ties":"descending abs original cached S; stable ascending original row", "reference_energy_MPa2":energy,"error_energy_MPa2":error}
    require(all(value is None or not isinstance(value,float) or math.isfinite(value) for value in result.values()), "Nonfinite signed metric")
    return result


def aggregate(cases, keys):
    result = {"fixed_denominator":len(keys),"case_keys":list(keys)}
    for name in ("signed_relative_l2","hotspot_relative_l2","S_squared_weighted_sign_error","range_floor_relative_l2","RMSE_MPa"):
        values = [(cases.get(key, {}).get("metrics") or {}).get(name) for key in keys]
        valid = all(isinstance(value,(int,float)) and math.isfinite(value) for value in values)
        result[name+"_mean"] = sum(values)/len(keys) if valid else None; result[name+"_worst"] = max(values) if valid else None
    return result


def selection_gate(state):
    rows = [state["cases"].get(key, {}).get("metrics") for key in DEV]
    complete = state["step"] == 60 and all(row is not None for row in rows)
    values = [row.get(name) for row in rows if row is not None for name in ("signed_relative_l2","hotspot_relative_l2","S_squared_weighted_sign_error")]
    finite = len(values) == 6 and all(isinstance(value,(int,float)) and math.isfinite(value) for value in values)
    checks = {"actual60":state["step"] == 60,"both_DEV_metrics_present_finite":complete and finite,
        "each_DEV_signed_relativeL2_strict_below_25pct":finite and all(row["signed_relative_l2"]<.25 for row in rows),
        "DEV_mean_hotspot_relativeL2_strict_below_25pct":finite and sum(row["hotspot_relative_l2"] for row in rows)/2<.25,
        "each_DEV_energy_sign_error_at_most_5pct":finite and all(row["S_squared_weighted_sign_error"]<=.05 for row in rows)}
    return {"passed":all(checks.values()),"checks":checks,"fixed_DEV_denominator":2,"no_DEV_best_or_recipe_rescue":True}


def actual_stage(np, torch, clock, plan, metadata_seconds):
    if clock.remaining()<=0:
        raise BudgetEnd("Stage180s budget after scientific-package/runtime startup")
    state = clock.state; random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.set_float32_matmul_precision("high")
    init_began = time.monotonic(); model = build_head(torch); optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    torch.cuda.synchronize(); state["costs"]["fresh_model_initialization_seconds"] = state["costs"].get("fresh_model_initialization_seconds",0.)+time.monotonic()-init_began
    actual_count = sum(parameter.numel() for parameter in model.parameters()); require(actual_count == 4737, "Actual tensor-only head parameter count differs")
    if (clock.output/"last.pt").exists():
        payload = own_load(torch, clock, clock.output/"last.pt", "checkpoint")
        require(payload["config"] == clock.config and payload["case_order"] == state["case_order"] and 0<=payload["step"]<=60, "Own checkpoint science/order differs")
        require(payload["step"]>=state["step"], "Metadata step cannot advance beyond committed checkpoint")
        state["step"] = payload["step"]; state["costs"]["training_wall_seconds"] = max(state["costs"]["training_wall_seconds"], payload["state"]["costs"]["training_wall_seconds"])
        state["optimizer_step_attempts"] = max(state["optimizer_step_attempts"], payload["state"]["optimizer_step_attempts"])
        for name in ("initial_parameter_sha256", "first_actual_gradient_check", "first_actual_shapes"):
            if name in payload["state"]:
                state[name] = payload["state"][name]
        model.load_state_dict(payload["model"], strict=True); optimizer.load_state_dict(payload["optimizer"])
        rng = payload["rng"]; random.setstate(rng["python"]); np.random.set_state(rng["numpy"]); torch.set_rng_state(rng["torch"].cpu()); torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
    else:
        require(state["step"] == 0, "Cannot reset a nonzero step without checkpoint")
        state["initial_parameter_sha256"] = {name:hashlib.sha256(parameter.detach().cpu().numpy().tobytes()).hexdigest() for name,parameter in model.named_parameters()}
        save_checkpoint(torch, np, clock, model, optimizer)
    state["parameter_count"] = actual_count; inputs = {}; targets = {}; predictions = {}; scaler = plan["inherited_tensor_scaler"]
    def get_input(key):
        if key not in inputs:
            inputs[key] = prepare_case(np, torch, clock, key, plan["records"][key], scaler)
        return inputs[key]
    def get_target(key):
        if key not in targets:
            tick = time.monotonic(); targets[key] = torch.as_tensor(cached_array(np, clock, key, "signed_stress", plan["records"][key]), device="cuda:0")
            require(torch.isfinite(targets[key]).all(), "Nonfinite original signed S"); torch.cuda.synchronize()
            state["costs"]["cached_signed_S_read_seconds"] += time.monotonic()-tick; clock.touch("cached_signed_S_loaded:"+key)
        return targets[key]
    while state["step"]<60:
        if clock.remaining()<=0 or state["costs"]["training_wall_seconds"]>=120:
            raise BudgetEnd("Cumulative training120s or stage180s reached; no new fullcase update")
        key = state["case_order"][state["step"]]; case = get_input(key); target = get_target(key); energy = target.square().sum()
        require(torch.isfinite(energy), "Nonfinite fullcase signed energy")
        if float(energy) == 0:
            state["cases"][key].update(status="zero_case_energy_scientific_negative", zero_case_energy=True)
            raise ScientificNegative("Zero fullcase signed energy: no floor or fabricated update")
        if clock.remaining()<=0:
            raise BudgetEnd("Stage budget after normal cache preparation")
        began = time.monotonic(); model.train(); optimizer.zero_grad(set_to_none=True)
        try:
            for start in range(0,len(target),32768):
                truth = target[start:start+32768]; logit = model(case["inputs"][start:start+32768]).squeeze(1).double()
                require(logit.shape == truth.shape, "Actual point head output shape differs")
                if state["step"] == 0 and start == 0:
                    state["first_actual_shapes"] = {"full_case_nodes":len(target),"input_channels":case["inputs"].shape[1],
                        "first_chunk_nodes":len(logit),"model_input_dtype":str(case["inputs"].dtype),"q_and_target_dtype":str(truth.dtype),"actual_parameter_count":actual_count}
                predicted = case["q"][start:start+32768]*torch.tanh(logit)
                field = (predicted-truth).square().sum(); nonzero = truth!=0
                sign_term = (truth[nonzero].square()*torch.nn.functional.softplus(-torch.sign(truth[nonzero])*logit[nonzero])).sum()
                loss = (field+.1*sign_term)/energy; require(torch.isfinite(loss), "Nonfinite fullcase chunk loss"); loss.backward()
            if state["step"] == 0:
                state["first_actual_gradient_check"] = gradient_check(torch, model)
            state["optimizer_step_attempts"] += 1
            optimizer.step(); torch.cuda.synchronize(); state["step"] += 1
        finally:
            state["costs"]["training_wall_seconds"] += time.monotonic()-began
        save_checkpoint(torch, np, clock, model, optimizer)
    if clock.remaining()<=0 or state["costs"]["training_wall_seconds"]>120:
        raise BudgetEnd("Training120s or stage180s budget at fixed60 endpoint")
    head_path = clock.output/"head.pt"; torch.save(model.state_dict(), head_path)
    tick = time.monotonic(); deployment_head = build_head(torch); deployment_head.load_state_dict(own_load(torch, clock, head_path, "head", weights_only=True), strict=True)
    torch.cuda.synchronize(); head_load_seconds = time.monotonic()-tick
    state["costs"]["own_deployment_head_load_seconds"] = state["costs"].get("own_deployment_head_load_seconds",0.)+head_load_seconds
    for key in KEYS:
        if state["cases"].get(key, {}).get("prediction_committed") is True:
            continue
        if clock.remaining()<=0:
            raise BudgetEnd("Stage budget before fullN endpoint prediction")
        predictions[key] = predict(np, torch, clock, deployment_head, key, get_input(key), plan["records"][key], head_load_seconds, metadata_seconds, plan["inherited_cold_components"][key])
    require(all(state["cases"][key]["prediction_committed"] is True for key in DEV), "DEV transaction has not persisted both predictions")
    for key in KEYS:
        if state["cases"][key].get("status") == "completed":
            continue
        if clock.remaining()<=0:
            raise BudgetEnd("Stage budget before cached-S endpoint metrics")
        predicted, logit = predictions[key] if key in predictions else reload_prediction(np, torch, clock, key)
        row = metrics(torch, predicted, logit, get_input(key)["q"], get_target(key)); torch.cuda.synchronize()
        state["cases"][key].update(status="completed", metrics=row, DEV_prediction_transaction_passed=True)
        write(clock.output/key.replace("/", "_")/"summary.json", state["cases"][key]); clock.touch("metrics_committed:"+key)


def run(plan_path=PLAN, output_path=None):
    began = time.monotonic(); contract, plan = request_contract(plan_path)
    require(plan.get("scientific_rules_frozen") is True and plan.get("actual_execution_released") is True, "Root release required before scientific imports/arrays/weights/GPU")
    require(plan.get("source_sha256") == contract["source_sha256"], "Exact new source pin required")
    marker = read(plan["root_review"]); require(marker.get("passed") is True and marker.get("source_sha256") == contract["source_sha256"][SOURCE], "Exact passed root source marker required")
    output = host_path(output_path or plan["output"]); require(output == host_path(plan["output"]), "Frozen output differs")
    config = {"contract":contract,"stage":KIND}
    if (output/"summary.json").exists():
        saved = read(output/"summary.json"); require(saved["config"] == config, "Prior result identity differs")
        if saved["status"] == "completed":
            return saved
    output.mkdir(parents=True, exist_ok=True)
    if (output/"config.json").exists():
        require(read(output/"config.json") == config, "Resume config differs")
    write(output/"config.json", config); clock = Clock(output, config, began); metadata_seconds = time.monotonic()-began
    clock.state["costs"]["common_metadata_wall_seconds"] = clock.state["costs"].get("common_metadata_wall_seconds",0.)+metadata_seconds
    clock.touch("released_metadata_complete")
    outcome, reason = "fixed60_completed", None
    try:
        if clock.remaining()<=0:
            raise BudgetEnd("Cumulative180s stage budget before scientific imports")
        startup_began = time.monotonic()
        import numpy as np
        import torch
        torch.cuda.set_device(0); torch.cuda.init(); torch.cuda.synchronize()
        startup_seconds = time.monotonic()-startup_began
        clock.state["costs"]["imports_CUDA_startup_seconds"] = clock.state["costs"].get("imports_CUDA_startup_seconds",0.)+startup_seconds
        actual_stage(np, torch, clock, plan, metadata_seconds+startup_seconds)
    except (BudgetEnd, ScientificNegative) as error:
        outcome, reason = ("budget_limited" if isinstance(error,BudgetEnd) else "scientific_negative"), str(error)
    except Exception as error:
        clock.touch("failed"); write(output/"summary.json", {"status":"failed","kind":KIND,"config":config,"train_keys":list(TRAIN),"validation_keys":list(DEV),
            "endpoint_step":clock.state["step"],"cases":clock.state["cases"],"reason":f"{type(error).__name__}: {error}","read_receipt":read_receipt(clock.state),"costs":clock.state["costs"]}); raise
    state = clock.state
    for key in KEYS:
        state["cases"].setdefault(key, {"key":key,"group":plan["records"][key]["group"],"num_nodes":plan["records"][key]["node_count"],"metrics":None,"costs":None})
        state["cases"][key].setdefault("status", "not_completed")
        state["cases"][key]["read_receipt"] = {name:dict(state["reads"][key][name]) for name in ARRAYS}
    if clock.elapsed()>180 or state["costs"]["training_wall_seconds"]>120:
        outcome = "budget_limited"; reason = reason or "Explicit finite endpoint finalization overshoot"
    gate = selection_gate(state)
    if outcome != "fixed60_completed":
        gate["passed"] = False; gate["checks"]["stage_completed_within_budget"] = False
    clock.touch("scientific_finalization")
    costs = {**state["costs"],"maximum_training_wall_seconds":120,"maximum_cumulative_stage_wall_seconds":180,
        "stage_budget_overshoot_seconds":max(0.,clock.elapsed()-180),"training_budget_overshoot_seconds":max(0.,state["costs"]["training_wall_seconds"]-120),
        "fresh_combined_cold_remeasured":False,"training_oneoff_separate_from_deployment":True}
    summary = {"status":"completed","kind":KIND,"config":config,"train_keys":list(TRAIN),"validation_keys":list(DEV),"cases":state["cases"],
        "endpoint_step":state["step"],"completed60":state["step"] == 60,"scientific_outcome":outcome,"scientific_negative_reason":reason,
        "groups":{"TRAIN":aggregate(state["cases"],TRAIN),"DEV":aggregate(state["cases"],DEV)},"selection_gate":gate,"read_receipt":read_receipt(state),
        "first_actual_gradient_check":state.get("first_actual_gradient_check"),"first_actual_shapes":state.get("first_actual_shapes"),
        "initial_parameter_sha256":state.get("initial_parameter_sha256"),"parameter_count":state.get("parameter_count"),
        "case_order":state["case_order"],"case_order_sha256":state["case_order_sha256"],"case_exposure_counts":{key:state["case_order"][:state["step"]].count(key) for key in TRAIN},
        "node_exposures":sum(plan["records"][key]["node_count"] for key in state["case_order"][:state["step"]]),"elapsed_seconds":clock.elapsed(),"costs":costs,
        "limitations":plan["limitations"],"original_signed_labels_unchanged":True,"DEV_allN_predictions_persist_before_either_DEV_cached_S":True}
    write(output/"summary.json", summary); clock.touch("completed")
    if clock.elapsed()>180:
        summary["scientific_outcome"] = "budget_limited"; summary["scientific_negative_reason"] = "Finalization crossed cumulative180s; explicit overshoot"
        summary["selection_gate"]["passed"] = False; summary["selection_gate"]["checks"]["stage_completed_within_budget"] = False
    summary["elapsed_seconds"] = clock.elapsed()
    summary["costs"].update(cumulative_stage_wall_seconds=clock.elapsed(),stage_budget_overshoot_seconds=max(0.,clock.elapsed()-180))
    write(output/"summary.json", summary)
    write(output/"status.json", {"status":"completed","endpoint_step":state["step"],"scientific_outcome":summary["scientific_outcome"],"costs":summary["costs"]})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--plan", type=Path, default=Path(PLAN)); parser.add_argument("--output", type=Path)
    args = parser.parse_args(); run(args.plan, args.output)
