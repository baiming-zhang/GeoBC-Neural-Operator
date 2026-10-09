"""Frozen cached constitutive-VM versus absolute signed-S magnitude diagnostic.

Only the three pinned parent arrays per case are read in the released actual
stage. No source scalar sign rule, calibration, training or physical solve is
inferred. Metadata is standard-library only; no parent trainer is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_stress_magnitude_plan.json"
SOURCE = "geobc/elastic_stress_magnitude.py"
KIND = "elastic_stress_magnitude_diagnostic"
TRAIN = ("101_428/ver","101_56/ver","101_78/ver")
DEV = ("102_249/ver","102_38/ver")
CASES = TRAIN+DEV
ARRAYS = ("sigma_MPa","valid","signed_stress")
METRICS = ("relativeL2","RMSE_MPa","MAE_MPa","max_MPa","hotspot_relativeL2","hotspot_MAE_MPa","hotspot_max_MPa")
FORBIDDEN = ("source_stress_member_reads","canonical_archive_reads","displacement_reads","old_final_reads","DEV_dia_reads",
    "sealed_new_test_reads","model_or_weight_reads","training_updates","optimizer_steps","scaler_fits","new_physical_solves","K_guards_repeated")


def host_path(value):
    text = str(value).replace("\\","/")
    if os.name == "nt" and text.startswith("/mnt/") and len(text)>7 and text[5].isalpha() and text[6] == "/":
        text = text[5].upper()+":"+text[6:]
    elif os.name != "nt" and len(text)>2 and text[1:3] == ":/":
        text = "/mnt/"+text[0].lower()+text[2:]
    result = Path(text)
    return result.resolve() if result.is_absolute() else (ROOT/result).resolve()


def read(value):
    return json.loads(host_path(value).read_text(encoding="utf-8-sig"))


def write(value,data):
    destination = host_path(value); temporary = destination.with_name(destination.name+".tmp")
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(temporary,destination)


def sha(value):
    return hashlib.sha256(host_path(value).read_bytes()).hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def require(condition,message):
    if not condition:
        raise ValueError(message)


def stat_matches(record):
    info = host_path(record["path"]).stat()
    return info.st_size == record["bytes"] and info.st_mtime_ns == record["mtime_ns"]


def pinned_json(record):
    require(host_path(record["path"]).stat().st_size == record["bytes"] and sha(record["path"]) == record["sha256"],"Pinned parent metadata identity differs")
    return read(record["path"])


def request_contract(plan_path=PLAN):
    """Only pinned parent JSON, three-array stat/schema and own small source SHA."""
    plan = read(plan_path)
    require(plan.get("kind") == KIND and tuple(plan.get("train_keys",())) == TRAIN and tuple(plan.get("validation_keys",())) == DEV
        and set(plan["records"]) == set(CASES),"Frozen five-case magnitude scope differs")
    science = plan["science"]
    require(science["tensor_slots"] == ["xx","yy","zz","xy","xz","yz"] and science["tensor_shear"] is True and science["unit"] == "MPa"
        and science["maximum_cumulative_stage_wall_seconds"] == 60 and science["scopes"] == ["allN","valid_only"],"Frozen tensor/unit/budget scope differs")
    require(tuple(science["read_policy"]["forbidden_zero"]) == FORBIDDEN
        and science["read_policy"]["allowed_arrays_per_case"] == ["cached_sigma_MPa","cached_valid","cached_original_signed_stress"],"Cached-only read policy differs")
    expected = {"strong_magnitude_support":{"DEV_max_valid_relativeL2_at_most":.1,"DEV_mean_valid_hotspot_relativeL2_at_most":.2,
        "TRAIN_max_valid_relativeL2_at_most":.1,"TRAIN_mean_valid_hotspot_relativeL2_at_most":.2},
        "proxy_magnitude_concern":{"any_DEV_valid_relativeL2_at_least":.25,"DEV_mean_valid_hotspot_relativeL2_at_least":.5,
        "TRAIN_anomaly_any_valid_relativeL2_at_least":.25,"TRAIN_anomaly_mean_valid_hotspot_relativeL2_at_least":.5}}
    require(all(science["interpretation"][key] == value for key,value in expected.items()),"Frozen TRAIN/DEV interpretation thresholds differ")
    parent = pinned_json(plan["parent_summary"]); parent_dir = host_path(plan["parent_summary"]["path"]).parent
    require(parent.get("status") == "completed" and parent.get("kind") == "elastic_signed_stress_readout_screen"
        and tuple(parent["train_keys"]) == TRAIN and tuple(parent["validation_keys"]) == DEV,"Parent completed readout scope differs")
    for key in CASES:
        record = plan["records"][key]; stem = key.replace("/","_")
        require(record["group"] == ("TRAIN" if key in TRAIN else "DEV") and record["node_count"]>0,"Fixed case group/count differs")
        feature = pinned_json(record["feature_manifest"]); committed = pinned_json(record["label_committed"]); started = pinned_json(record["label_read_started"])
        require(feature == parent["cases"][key] and feature.get("feature_valid") is True and feature["feature_guards"]["passed"] is True
            and feature["node_count"] == record["node_count"],"Parent feature provenance differs")
        require(committed["identity"] == started["identity"] == record["label_identity"] and started["member_read_attempted"] is True
            and started["member_read_completed"] is True and record["label_identity"]["member"] == "stress"
            and record["label_identity"]["key"] == key and record["label_identity"]["scope"] == record["group"],"Committed/read-started signed stress provenance differs")
        require(feature["arrays"]["sigma_MPa"] == record["sigma_MPa"] and feature["arrays"]["valid"] == record["valid"]
            and committed["array"] == record["signed_stress"],"Exact parent array records differ")
        for name in ARRAYS:
            value = record[name]
            expected_path = parent_dir/"label_cache"/(stem+".npy") if name == "signed_stress" else parent_dir/"input_cache"/stem/(name+".npy")
            require(host_path(value["path"]) == expected_path and stat_matches(value) and len(value["values_sha256"]) == 64
                and value["shape"] == ([record["node_count"],6] if name == "sigma_MPa" else [record["node_count"]])
                and value["dtype"] == ("bool" if name == "valid" else "float64"),"Pinned cached array path/stat/schema differs")
    source_sha = sha(SOURCE)
    for name,value in plan.get("source_sha256",{}).items():
        require(name == SOURCE and value == source_sha,"Only exact new diagnostic source pin is allowed")
    return {"schema":"geobc_elastic_stress_magnitude_v1","kind":KIND,"train_keys":list(TRAIN),"validation_keys":list(DEV),
        "parent_summary":plan["parent_summary"],"records":plan["records"],"science":science,"source_sha256":{SOURCE:source_sha},
        "output":str(host_path(plan["output"])),"results_output":str(host_path(plan["results_output"])),
        "array_or_GT_reads_in_request_contract":0},plan


class Clock:
    def __init__(self,output,config,began):
        self.output,self.config,self.began,self.prior = output,config,began,0.
        self.identity = digest(config)
        if (output/"status.json").exists():
            saved = read(output/"status.json"); require(saved["config_sha256"] == self.identity,"Cumulative diagnostic status differs")
            self.prior = saved["cumulative_stage_wall_seconds"]
        if (output/"summary.json").exists():
            saved = read(output/"summary.json"); require(saved["config"] == config,"Prior diagnostic summary differs")
            self.prior = max(self.prior,saved["costs"]["cumulative_stage_wall_seconds"])
        require(math.isfinite(self.prior) and self.prior>=0,"Invalid prior diagnostic wall")
    def elapsed(self):
        return self.prior+time.monotonic()-self.began
    def remaining(self):
        return max(0.,60-self.elapsed())
    def account(self,phase,key=None):
        write(self.output/"status.json",{"config_sha256":self.identity,"phase":phase,"case":key,"cumulative_stage_wall_seconds":self.elapsed(),"maximum_stage_wall_seconds":60})


def initial_reads():
    return {key:{name:{"attempted":0,"completed":0} for name in ARRAYS} for key in CASES}


def receipts(counts):
    attempted = {name:sum(counts[key][name]["attempted"] for key in CASES) for name in ARRAYS}
    completed = {name:sum(counts[key][name]["completed"] for key in CASES) for name in ARRAYS}
    return {**{key:0 for key in FORBIDDEN},"GT_reads":attempted["signed_stress"],"stress_reads":attempted["signed_stress"],
        "cached_signed_stress_reads":attempted["signed_stress"],"TRAIN_stress_reads":sum(counts[key]["signed_stress"]["attempted"] for key in TRAIN),
        "DEV_stress_reads":sum(counts[key]["signed_stress"]["attempted"] for key in DEV),"sigma_array_reads":attempted["sigma_MPa"],
        "valid_array_reads":attempted["valid"],"attempted_array_reads":attempted,"completed_array_reads":completed,"per_case":counts}


def scoped_metrics(torch,q,truth,rows):
    count = len(rows)
    if count == 0:
        return {"node_count":0,**{name:None for name in METRICS},"zero_reference":True,"hotspot_zero_reference":True,"hotspot_nodes":0,"empty_scope":True}
    predicted,actual = q[rows],truth[rows]; delta = predicted-actual; ref = float(actual.square().sum()); energy = float(delta.square().sum())
    n_hot = math.ceil(.01*count); order = torch.argsort(actual,descending=True,stable=True)[:n_hot]
    hot_ref = float(actual[order].square().sum()); hot_delta = delta[order]; hot_energy = float(hot_delta.square().sum())
    require(all(math.isfinite(value) for value in (ref,energy,hot_ref,hot_energy)),"Nonfinite magnitude metric energy")
    result = {"node_count":count,"relativeL2":None if ref == 0 else math.sqrt(energy)/math.sqrt(ref),"RMSE_MPa":math.sqrt(energy/count),
        "MAE_MPa":float(delta.abs().mean()),"max_MPa":float(delta.abs().max()),"hotspot_relativeL2":None if hot_ref == 0 else math.sqrt(hot_energy)/math.sqrt(hot_ref),
        "hotspot_MAE_MPa":float(hot_delta.abs().mean()),"hotspot_max_MPa":float(hot_delta.abs().max()),
        "zero_reference":ref == 0,"hotspot_zero_reference":hot_ref == 0,"hotspot_nodes":n_hot,"empty_scope":False,
        "hotspot_ties":"descending abs cached original signed S, stable ascending original row","reference_energy_MPa2":ref,"error_energy_MPa2":energy}
    require(all(result[name] is None or math.isfinite(result[name]) for name in METRICS),"Nonfinite magnitude metric")
    return result


def diagnose_case(key,record,directory,config,clock,counts,persist_reads):
    directory.mkdir(exist_ok=True); target = directory/"summary.json"; case_config = {"contract_sha256":digest(config["contract"]),"key":key}
    if target.exists():
        saved = read(target); require(saved["config"] == case_config,"Committed case result identity differs")
        return saved
    write(directory/"config.json",case_config); begun = time.monotonic()
    case = {"key":key,"group":record["group"],"status":"budget","config":case_config,"node_count":record["node_count"],"scopes":{},"magnitude_only":True}
    try:
        if clock.remaining()<=0:
            case["reason"] = "Cumulative60s budget exhausted before this case; no scientific import or array read"
        else:
            import numpy as np
            import torch
        values = {}; io_began = time.monotonic()
        for name in ARRAYS:
            if clock.remaining()<=0:
                case["reason"] = "cumulative60s budget before cached array read"; break
            value = record[name]; require(stat_matches(value),"Cached array stat changed before actual read")
            counts[key][name]["attempted"] += 1; persist_reads(); clock.account("cached_array_read_attempt",key+":"+name)
            write(directory/"status.json",{"status":"running","phase":"cached_array_read_attempt","array":name,
                "read_counts":counts[key],"cumulative_stage_wall_seconds":clock.elapsed()})
            array = np.load(host_path(value["path"]),allow_pickle=False)
            counts[key][name]["completed"] += 1; persist_reads(); clock.account("cached_array_read_completed",key+":"+name)
            require(list(array.shape) == value["shape"] and str(array.dtype) == value["dtype"]
                and hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() == value["values_sha256"],"Loaded cached value identity/schema differs")
            values[name] = torch.as_tensor(array,device="cuda:0")
        if values:
            torch.cuda.synchronize()
        io_seconds = time.monotonic()-io_began
        if len(values) == 3 and clock.remaining()>0:
            sigma,valid,signed = (values[name] for name in ARRAYS)
            require(sigma.dtype == signed.dtype == torch.float64 and valid.dtype == torch.bool
                and torch.isfinite(sigma).all() and torch.isfinite(signed).all(),"Nonfinite cached sigma/signed S")
            tick = time.monotonic(); xx,yy,zz,xy,xz,yz = sigma.unbind(1)
            q = torch.sqrt(.5*((xx-yy).square()+(yy-zz).square()+(zz-xx).square())+3*(xy.square()+xz.square()+yz.square()))
            require(torch.isfinite(q).all() and (q>=0).all(),"Nonfinite or negative VM magnitude; no clipping")
            truth = signed.abs(); rows = torch.arange(len(signed),device=signed.device)
            case["scopes"] = {"allN":scoped_metrics(torch,q,truth,rows),"valid_only":scoped_metrics(torch,q,truth,rows[valid])}
            torch.cuda.synchronize(); metric_seconds = time.monotonic()-tick
            required = [case["scopes"][scope][name] for scope in ("allN","valid_only") for name in ("relativeL2","hotspot_relativeL2")]
            case.update(status="completed" if all(value is not None and math.isfinite(value) for value in required) else "invalid",
                valid_nodes=int(valid.sum()),missing_incident_nodes=int((~valid).sum()),cached_value_identity_passed=True,all_finite=True,
                costs={"cached_IO_identity_H2D_seconds":io_seconds,"q_and_metrics_seconds":metric_seconds})
            if clock.remaining()<=0:
                case.update(status="budget",reason="Finite current-case metrics completed with explicit60s overshoot")
        elif len(values) == 3:
            case["reason"] = "Cumulative60s budget exhausted after cache identity/H2D; magnitude metrics not started"
    except (ValueError,OSError) as error:
        case.update(status="invalid",reason=f"{type(error).__name__}: {error}")
    case.setdefault("costs",{}).update(case_wall_seconds=time.monotonic()-begun)
    case["read_receipt"] = {name:dict(counts[key][name]) for name in ARRAYS}
    write(target,case); write(directory/"status.json",{"status":case["status"],"case_wall_seconds":case["costs"]["case_wall_seconds"],
        "cumulative_stage_wall_seconds":clock.elapsed(),"read_counts":counts[key]})
    clock.account("case_finished",key); return case


def aggregate(cases,keys):
    output = {"fixed_denominator":len(keys),"cases":list(keys)}
    for scope in ("allN","valid_only"):
        metrics = {}
        for name in METRICS:
            values = [(cases.get(key,{}).get("scopes",{}).get(scope) or {}).get(name) for key in keys]
            valid = all(isinstance(value,(int,float)) and math.isfinite(value) for value in values)
            metrics[name+"_mean"] = sum(values)/len(keys) if valid else None; metrics[name+"_worst"] = max(values) if valid else None
        output[scope] = metrics
    return output


def interpretation(cases,groups,budget_limited):
    complete = all(cases[key]["status"] == "completed" for key in CASES)
    train,dev = groups["TRAIN"]["valid_only"],groups["DEV"]["valid_only"]
    values = [train["relativeL2_worst"],train["hotspot_relativeL2_mean"],dev["relativeL2_worst"],dev["hotspot_relativeL2_mean"]]
    valid = complete and all(value is not None and math.isfinite(value) for value in values)
    if budget_limited or not valid:
        return {"classification":"budget_limited" if budget_limited else "diagnostic_invalid","diagnostic_valid":False,"predicts_source_sign":False}
    train_strong = values[0]<=.1 and values[1]<=.2; dev_strong = values[2]<=.1 and values[3]<=.2
    train_anomaly = values[0]>=.25 or values[1]>=.5; dev_concern = values[2]>=.25 or values[3]>=.5
    return {"classification":"strong_magnitude_support" if train_strong and dev_strong else ("proxy_magnitude_concern" if train_anomaly or dev_concern else "inconclusive"),
        "diagnostic_valid":True,"TRAIN_strong_thresholds_met":train_strong,"DEV_strong_thresholds_met":dev_strong,"TRAIN_anomaly":train_anomaly,"DEV_magnitude_concern":dev_concern,
        "predicts_source_sign":False,"no_unique_causal_attribution":True,"thresholds_frozen_before_actual":True}


def run(plan_path=PLAN,output_path=None):
    began = time.monotonic(); contract,plan = request_contract(plan_path)
    require(plan.get("scientific_rules_frozen") is True and plan.get("actual_execution_released") is True,"Root release required before scientific packages/cache values/GPU")
    require(plan.get("source_sha256") == contract["source_sha256"],"Exact diagnostic source pin required")
    marker = read(plan["root_review"]); require(marker.get("passed") is True and marker.get("source_sha256") == contract["source_sha256"][SOURCE],"Exact passed root source marker required")
    output = host_path(output_path or plan["output"]); require(output == host_path(plan["output"]),"Frozen diagnostic output differs")
    config = {"contract":contract,"stage":KIND}
    if (output/"config.json").exists():
        require(read(output/"config.json") == config,"Prior diagnostic identity differs")
    if (output/"summary.json").exists():
        saved = read(output/"summary.json"); require(saved["config"] == config,"Prior completed result identity differs")
        if saved.get("status") == "completed":
            return saved
    output.mkdir(parents=True,exist_ok=True); write(output/"config.json",config); clock = Clock(output,config,began); clock.account("starting")
    receipt_path = output/"read_receipt.json"; counts = initial_reads()
    if receipt_path.exists():
        saved = read(receipt_path); require(saved["config_sha256"] == digest(config),"Prior cached read ledger identity differs"); counts = saved["counts"]
    def persist_reads():
        write(receipt_path,{"config_sha256":digest(config),"counts":counts,"read_receipt":receipts(counts)})
    persist_reads(); cases = {}
    try:
        for key in CASES:
            directory = output/key.replace("/","_")
            cases[key] = diagnose_case(key,contract["records"][key],directory,config,clock,counts,persist_reads)
        groups = {"TRAIN":aggregate(cases,TRAIN),"DEV":aggregate(cases,DEV)}
        result = interpretation(cases,groups,clock.elapsed()>60 or any(item["status"] == "budget" for item in cases.values()))
        summary = {"status":"completed","kind":KIND,"config":config,"train_keys":list(TRAIN),"validation_keys":list(DEV),"cases":cases,"groups":groups,
            "interpretation":result,"scientific_outcome":result["classification"],"read_receipt":receipts(counts),"magnitude_only":True,"predicts_source_sign":False,
            "costs":{"cumulative_stage_wall_seconds":clock.elapsed(),"maximum_stage_wall_seconds":60,"budget_overshoot_seconds":max(0.,clock.elapsed()-60)},
            "limitations":plan["science"]["interpretation"]["limitations"],"invalid_nodes":"allN keeps missing proxy nodes; valid_only separates coverage, no physical-zero claim"}
        results = host_path(plan["results_output"]); results.mkdir(parents=True,exist_ok=True)
        lines = ["# Cached stress magnitude diagnostic", "", "Classification: **"+result["classification"]+"**.", "",
            "FP64 constitutive VM magnitude versus |cached original signed S|; no source-sign inference, fitting, new solve or recipe rescue.", "",
            "| Group | Nodes | valid relative L2 mean / worst | valid hotspot relative L2 mean |", "|---|---:|---:|---:|"]
        def display(value):
            return "null" if value is None else f"{100*value:.3f}%"
        for group,keys in (("TRAIN",TRAIN),("DEV",DEV)):
            values = groups[group]["valid_only"]; lines.append(f"| {group} ({len(keys)} fixed CADs) | {sum(plan['records'][key]['node_count'] for key in keys)} | {display(values['relativeL2_mean'])} / {display(values['relativeL2_worst'])} | {display(values['hotspot_relativeL2_mean'])} |")
        lines += ["", "All-node and valid-only MPa metrics, actual attempted/completed cached reads and missing values are preserved in metrics.json.", "", plan["science"]["interpretation"]["limitations"]]
        (results/"BRIEF.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        write(results/"metrics.json",summary); clock.account("results_persisted")
        if clock.elapsed()>60:
            summary["interpretation"] = interpretation(cases,groups,True); summary["scientific_outcome"] = "budget_limited"
            lines[2] = "Classification: **budget_limited**."
            (results/"BRIEF.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        summary["costs"].update(cumulative_stage_wall_seconds=clock.elapsed(),budget_overshoot_seconds=max(0.,clock.elapsed()-60))
        write(output/"summary.json",summary); write(results/"metrics.json",summary); clock.account("completed"); return summary
    except Exception as error:
        clock.account("failed"); write(output/"summary.json",{"status":"failed","kind":KIND,"config":config,"train_keys":list(TRAIN),"validation_keys":list(DEV),"cases":cases,
            "reason":f"{type(error).__name__}: {error}","read_receipt":receipts(counts),"costs":{"cumulative_stage_wall_seconds":clock.elapsed()}}); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--plan",type=Path,default=Path(PLAN)); parser.add_argument("--output",type=Path)
    args = parser.parse_args(); run(args.plan,args.output)
