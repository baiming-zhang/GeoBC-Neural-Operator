"""CPU JSON-only report of the fixed480 FNO response pair and DEV5/dia.

Never import a solver/trainer, open arrays/weights, select checkpoints or train.
Matplotlib is imported lazily only when root actually writes the single figure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = "notes/geobc_fno_response_plan.json"
DEFAULT_EVAL_PLAN = "notes/geobc_fno_response_eval_plan.json"
DEFAULT_EVAL_RUN = "runs/geobc_exploration/fno_response_dev5_dia_endpoint480"
MODES = ("load_dependent_basis", "linear_basis")
ROUTES = (*MODES,"historical_direct4")
LABELS = {"load_dependent_basis":"Load-dependent u9", "linear_basis":"Force-free u9", "historical_direct4":"Historical direct4"}
CN_LABELS = {"load_dependent_basis":"保留载荷u9（control）", "linear_basis":"力无关u9（candidate）", "historical_direct4":"历史direct4（非matched）"}
FIELDS = ("case_mean_u_relative_l2", "worst_case_u_relative_l2", "case_mean_stress_relative_l2", "worst_case_stress_relative_l2", "case_mean_stress_hotspot_relative_l2")
MAX_JSON_BYTES = 32*1024*1024


def path(value):
    value = str(value).replace("\\","/")
    if "/GeoBC Neural Operator/" in value:
        value = value.split("/GeoBC Neural Operator/",1)[1]
    result = (ROOT/Path(value)).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError(f"Report path leaves workspace: {result}")
    return result


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def invalid_constant(value):
    raise ValueError(f"Nonfinite JSON: {value}")


def read_json(filename, diagnostics):
    filename=path(filename)
    receipt={"path":str(filename),"present":filename.is_file()}
    if not receipt["present"]:
        return {},receipt
    size=filename.stat().st_size
    if filename.suffix != ".json" or size>MAX_JSON_BYTES:
        raise ValueError("Only bounded small JSON receipts are allowed")
    contents=filename.read_bytes()
    receipt.update(bytes=len(contents),sha256=hashlib.sha256(contents).hexdigest())
    try:
        result=json.loads(contents.decode("utf-8-sig"),parse_constant=invalid_constant)
        if not isinstance(result,dict): raise ValueError("Expected JSON object")
        return result,receipt
    except (ValueError,UnicodeError) as error:
        diagnostics.append(f"{filename.name}: {error}")
        return {},receipt


def metric_view(value, cases, diagnostics, label):
    if not isinstance(value,dict) or value.get("cases") != cases or not isinstance(value.get("nodes"),int) or value["nodes"] <= 0:
        diagnostics.append(f"{label}: missing full-node{cases}-case aggregate")
        return None
    if any(not finite(value.get(field)) or value[field]<0 for field in FIELDS):
        diagnostics.append(f"{label}: incomplete/nonfinite physical metrics")
        return None
    per_case=value.get("per_case",{})
    if not isinstance(per_case,dict) or len(per_case)!=cases:
        diagnostics.append(f"{label}: per-case inventory differs")
        return None
    if any(not isinstance(record,dict) or not isinstance(record.get("nodes"),int) or record["nodes"]<=0
            or any(not finite(record.get(key)) or record[key]<0 for key in ("u_relative_l2","stress_relative_l2","stress_hotspot_relative_l2"))
            for record in per_case.values()) or sum(record["nodes"] for record in per_case.values())!=value["nodes"]:
        diagnostics.append(f"{label}: per-case finite metrics or full-node totals differ")
        return None
    worst_ids={}
    for group,key in (("u","u_relative_l2"),("stress","stress_relative_l2"),("stress_hotspot","stress_hotspot_relative_l2")):
        values={name:record.get(key) for name,record in per_case.items()}
        if all(finite(item) for item in values.values()):
            maximum=max(values.values())
            worst_ids[group]=[name for name,item in values.items() if item==maximum]
    return {**value,"worst_case_ids":worst_ids,
        "physical_units":{"u":"mm signed XYZ vector","source_signed_stress":"MPa"},
        "hotspot_definition":"Per-case top1% largest absolute source stress; same signed target, no displacement hotspot"}


def train_row(mode,directory):
    row={"mode":mode,"label":LABELS[mode],"matched_pair_member":mode in MODES,
        "status":"pending_run_definition","endpoint_step":None,"validation":None,
        "train_readout":None,"diagnostics":[],"sources":{}}
    if directory is None: return row
    directory=path(directory); row["output_directory"]=str(directory)
    summary,row["sources"]["summary"]=read_json(directory/"summary.json",row["diagnostics"])
    status,row["sources"]["status"]=read_json(directory/"status.json",row["diagnostics"])
    if summary.get("status")!="completed":
        row["status"]=status.get("status",status.get("phase","not_completed"))
        row["running_step"]=status.get("step"); row["elapsed_seconds"]=status.get("elapsed_seconds")
        return row
    config=summary.get("config",{})
    endpoint=summary.get("final_metrics",{})
    declared_mode=summary.get("response_mode",summary.get("mode",config.get("model",{}).get("response_mode")))
    validation=metric_view(endpoint.get("validation"),15,row["diagnostics"],mode+" VAL15")
    exposures=summary.get("case_exposure_counts",{})
    costs=summary.get("costs",{})
    coherent=(summary.get("endpoint_step")==480 and endpoint.get("step")==480 and validation is not None
        and len(exposures)==120 and all(count==4 for count in exposures.values())
        and costs.get("train_case_exposures")==480 and isinstance(costs.get("train_node_exposures"),int)
        and costs["train_node_exposures"]>0 and finite(costs.get("train_seconds")) and costs["train_seconds"]>=0)
    if mode in MODES:
        coherent=coherent and declared_mode==mode
    else:
        coherent=coherent and summary.get("method")=="fno"
    if not coherent:
        row["diagnostics"].append("Completed summary is not the fixed480/4pass full-node endpoint; not published as completed")
        row["status"]="invalid_completed_summary"
        return row
    grad=summary.get("first_update_gradient_check")
    actual_checks=summary.get("first_actual_checks",summary.get("first_actual_case_checks",summary.get("first_actual_check")))
    row.update(status="completed",endpoint_step=480,validation=validation,
        train_readout=endpoint.get("train_readout"),elapsed_seconds=summary.get("elapsed_seconds"),
        costs=costs,memory=summary.get("memory"),parameter_counts=summary.get("parameter_counts"),
        case_exposure_counts=exposures,case_order_sha256=summary.get("case_order_sha256",config.get("case_order_sha256")),
        first_update_gradient_check=grad,first_actual_checks=actual_checks,
        input_consistency_guard=summary.get("input_consistency_guard"),
        initialization_declaration=config.get("fresh_initialization"),
        common_inputs={key:config.get("contract",{}).get(key) for key in ("scaler","cache_contract","train_keys","validation_keys")},
        config=config,checkpoint_policy="Fixed endpoint480 only; no best/seed selection by report",
        cost_scope="Actual complete-case training update includes normal load/H2D/body/loss/backward/Adam/sync; independent input-guard build/read excluded and separately timed. Wall includes guards; evaluation/inference timers overlap, do not sum as disjoint costs")
    return row


def common_model_record(record):
    """Only the predeclared load-mask mode/policy may differ in model metadata."""
    result={key:value for key,value in record.items() if key!="response_mode"}
    configuration=result.get("configuration",{})
    result["configuration"]={key:value for key,value in configuration.items() if key not in ("mode","load_policy")}
    return result


def pair_view(rows):
    control,candidate=(rows[key] for key in MODES)
    result={"status":"pending", "gain":None, "comparability":None}
    if any(row["status"]!="completed" for row in (control,candidate)):
        return result
    common=control["common_inputs"]
    shared_inputs_present=all(common.get(key) is not None for key in ("scaler","cache_contract","train_keys","validation_keys"))
    contracts=[{key:value for key,value in row["config"].get("contract",{}).items() if key!="response_mode"} for row in (control,candidate)]
    models=[row["config"].get("model",{}) for row in (control,candidate)]
    first_checks=[row["first_actual_checks"] or {} for row in (control,candidate)]
    declarations=[row["initialization_declaration"] for row in (control,candidate)]
    checks={"both_completed_endpoint480":True,"same_case_order":control["case_order_sha256"] is not None and control["case_order_sha256"]==candidate["case_order_sha256"],
        "same_shared_inputs_scaler":shared_inputs_present and common==candidate["common_inputs"],
        "same_recipe_source_contract_except_mode":bool(contracts[0]) and contracts[0]==contracts[1],
        "same_architecture_metadata_except_load_mask":bool(models[0]) and common_model_record(models[0])==common_model_record(models[1]),
        "same_actual_parameter_counts":control["parameter_counts"] is not None and control["parameter_counts"]==candidate["parameter_counts"],
        "declared_same_fresh_constructor_seed_layout":bool(declarations[0]) and declarations[0]==declarations[1] and all(model.get("seed")==2026 for model in models),
        "first_grad_control_passed":first_checks[0].get("gradient",{}).get("passed") is True and first_checks[0].get("passed") is True,
        "first_grad_candidate_passed":first_checks[1].get("gradient",{}).get("passed") is True and first_checks[1].get("passed") is True,
        "first_units_alpha_query_control_passed":first_checks[0].get("units_alpha_query",{}).get("passed") is True,
        "first_units_alpha_query_candidate_passed":first_checks[1].get("units_alpha_query",{}).get("passed") is True,
        "both_input_only45CAD_guards_complete":all((row["input_consistency_guard"] or {}).get("unique_CADs")==45 for row in (control,candidate))}
    zero_checks={mode:rows[mode]["first_actual_checks"] for mode in MODES}
    gain={"known_u_mean_improvement_percentage_points":100*(control["validation"][FIELDS[0]]-candidate["validation"][FIELDS[0]]),
        "known_u_worst_deterioration_percentage_points":100*(candidate["validation"][FIELDS[1]]-control["validation"][FIELDS[1]]),
        "known_stress_mean_improvement_percentage_points":100*(control["validation"][FIELDS[2]]-candidate["validation"][FIELDS[2]]),
        "known_stress_worst_deterioration_percentage_points":100*(candidate["validation"][FIELDS[3]]-control["validation"][FIELDS[3]]),
        "known_stress_hotspot_improvement_percentage_points":100*(control["validation"][FIELDS[4]]-candidate["validation"][FIELDS[4]])}
    return {"status":"completed", "gain":gain,"comparability":checks,
        "first_actual_mathematical_receipts":zero_checks,
        "initialization_evidence":{"same_constructor_seed_layout_declared":checks["declared_same_fresh_constructor_seed_layout"],
            "complete_initial_parameter_hashes_saved":False,"full_trainable_tensor_equality_verified":None,
            "policy":"No saved full initialization hash or weight reads; constructor/seed/layout declaration is not a tensor-by-tensor hash proof"},
        "metadata_checks_passed":all(value is True for value in checks.values()),
        "readout_rule":"Positive mean/hotspot gain means candidate lower; positive worst deterioration means candidate higher. No undeclared threshold or automatic winner.",
        "limitations":"Different input information/effective capacity despite equal stored tensors; single seed four passes, not engineering/GT-linearity proof"}


def eval_view(directory,rows):
    result={"status":"not_completed","routes":{},"diagnostics":[],"sources":{}}
    summary,result["sources"]["summary"]=read_json(path(directory)/"summary.json",result["diagnostics"])
    status,result["sources"]["status"]=read_json(path(directory)/"status.json",result["diagnostics"])
    if summary.get("status")!="completed":
        result["status"]=status.get("status",status.get("phase","not_completed"))
        return result
    routes=summary.get("routes",{})
    valid=(summary.get("kind")=="fno_response_eval" and summary.get("all_nodes") is True
        and summary.get("training_updates")==summary.get("scaler_fits")==0
        and len(summary.get("dia_case_keys",[]))==5 and set(routes)==set(ROUTES))
    extracted={}
    for mode in ROUTES:
        route=routes.get(mode,{})
        metrics=metric_view(route.get("dia"),5,result["diagnostics"],mode+" DEV5 dia")
        reused=route.get("known3_endpoint_validation_reused")
        valid=valid and metrics is not None and route.get("checkpoint",{}).get("checkpoint_step")==480
        if rows[mode]["status"]=="completed" and reused is not None:
            # Source aggregate is compared, before this report's additive metadata.
            expected={key:value for key,value in rows[mode]["validation"].items()
                if key not in ("worst_case_ids","physical_units","hotspot_definition")}
            valid=valid and expected==reused
        extracted[mode]={**route,"dia":metrics,"historical_direct4_is_matched_control":False if mode=="historical_direct4" else None}
    if not valid:
        result["status"]="invalid_completed_summary"
        result["diagnostics"].append("DEV5 summary differs from fixed480/all-node/three-route scope or reused known3 endpoint")
        return result
    result.update(status="completed",routes=extracted,case_keys=summary["dia_case_keys"],
        checks=summary.get("checks"),input_receipts=summary.get("input_receipts"),
        costs=summary.get("costs"),elapsed_seconds=summary.get("elapsed_seconds"),memory=summary.get("memory"),
        read_receipt=summary.get("read_receipt"),interpretation=summary.get("interpretation"),
        config=summary.get("config"),precision_receipt=summary.get("precision_receipt"),
        checkpoint_selection="All three endpoint480; no dia-driven selection, fitting or rescue")
    return result


def _job_mode(job):
    direct=job.get("mode",job.get("response_mode"))
    if direct in MODES:return direct
    command=job.get("command",[])
    if isinstance(command,list):
        for flag in ("--mode","--response-mode"):
            if flag in command and command.index(flag)+1<len(command):
                candidate=command[command.index(flag)+1]
                if candidate in MODES:return candidate
    return None


def build_report(plan_path=DEFAULT_PLAN,eval_plan_path=DEFAULT_EVAL_PLAN,eval_run=DEFAULT_EVAL_RUN):
    diagnostics=[]
    plan,plan_receipt=read_json(plan_path,diagnostics)
    eval_plan,eval_plan_receipt=read_json(eval_plan_path,diagnostics)
    directories={}
    for job in plan.get("jobs",plan.get("queue_jobs",[])):
        mode=_job_mode(job)
        if mode in MODES and job.get("output"):
            directories[mode]=job["output"]
    for route in eval_plan.get("routes",[]):
        mode=route.get("route_id")
        if mode in ROUTES and route.get("run"):
            if mode in directories and path(directories[mode])!=path(route["run"]):
                raise ValueError("Training/evaluation plans disagree on output origin")
            directories[mode]=route["run"]
    # The historical context is an explicitly fixed existing receipt, not a run inferred from results.
    directories.setdefault("historical_direct4","runs/geobc_exploration/solver_screen_fno_480")
    rows={mode:train_row(mode,directories.get(mode)) for mode in ROUTES}
    pair=pair_view(rows); dia=eval_view(eval_run,rows)
    return {"schema":"geobc_fno_response_report_v1","generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "status":"completed" if pair["status"]==dia["status"]=="completed" else "pending_or_partial",
        "training_plan":plan_receipt,"training_recipe":plan.get("recipe",plan.get("protocol")),
        "eval_plan":eval_plan_receipt,"runs":rows,"pair":pair,"development_dia":dia,
        "report_tool_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "diagnostics":diagnostics,
        "scope":{"primary":"Two fresh seed2026 fixed480 matched-layout response arms on DEV5 xknown3; all original nodes",
            "secondary":"Independent completed DEV5/dia endpoint480 three-route development diagnostic; previously explored CADs, not blind final",
            "historical":"Old direct4 FNO480 context only; different head/init, never counted as u9 matched control",
            "loss":"Complete-case signed4 MSE in frozen TRAIN-only encoding; stress has its own nonlinear force-conditioned head",
            "physical_units":{"displacement":"mm","source_signed_stress":"MPa"},
            "no_checkpoint_selection":"Report endpoint480 only, no best/seed or task-mixing ratios",
            "limits":"Single seed/four passes; no statistical significance, engineering acceptance, exact GT-linearity or native continuous/BC claim",
            "excluded":"Old final/new sealed test, arrays/weights/models/GPU; no loss or budget tuning by this report"}}


def fmt(value,percent=False):
    return f"{100*value if percent else value:.2f}" if finite(value) else "—"


def metrics_text(record):
    if record is None:return "—","—"
    return " / ".join(fmt(record[key],True) for key in FIELDS[:2]), " / ".join(fmt(record[key],True) for key in FIELDS[2:])


def brief(report):
    rows=report["runs"]
    lines=["# FNO 响应基结构诊断","","固定seed2026、480完整案例更新/四passes；两fresh u9 arm为配对，历史direct4仅上下文。U为signed XYZ向量、S为源signed scalar，均在全原节点物理单位上评估。", "",
        "| known3 DEV15 / 固定480 | U mean / worst % | S mean / worst / hotspot % | actual训练秒 |",
        "|---|---:|---:|---:|"]
    for mode in ROUTES:
        row=rows[mode];u,s=metrics_text(row["validation"])
        label=CN_LABELS[mode] if row["status"]=="completed" else CN_LABELS[mode]+f"（{row['status']}）"
        lines.append(f"| {label} | {u} | {s} | {fmt(row.get('costs',{}).get('train_seconds'))} |")
    pair=report["pair"]
    if pair["status"]=="completed":
        gain=pair["gain"]
        lines += ["",f"候选相对fresh control：U均值改善 {gain['known_u_mean_improvement_percentage_points']:+.2f}pp、S均值改善 {gain['known_stress_mean_improvement_percentage_points']:+.2f}pp；最差U/S恶化 {gain['known_u_worst_deterioration_percentage_points']:+.2f}/{gain['known_stress_worst_deterioration_percentage_points']:+.2f}pp（正值为变差）。实值逐项报告，未自动判赢家。"]
        controls="PASS" if pair["metadata_checks_passed"] else "未完整证明，请查JSON"
        lines += [f"共同recipe/输入/scaler、案例顺序、参数布局及首实际梯度/单位见证：{controls}。两臂声明同constructor/seed，未保存完整初始参数哈希；不能把该声明当全tensor比对。zero-load均值偏移/α/query见证原样保存在metrics.json。"]
    else:lines += ["","新配对尚未完整完成，未将中途或best指标填入固定端点。"]
    dia=report["development_dia"]
    if dia["status"]=="completed":
        lines += ["","| 独立DEV5 dia / 同endpoint480 | U mean / worst % | S mean / worst / hotspot % |","|---|---:|---:|"]
        for mode in ROUTES:
            u,s=metrics_text(dia["routes"][mode]["dia"])
            lines.append(f"| {CN_LABELS[mode]} | {u} | {s} |")
    else:lines += ["",f"DEV5 dia：{dia['status']}；仅completed后并列三route，不以中途结果选checkpoint/seed。"]
    lines += ["","训练节点曝光、参数、wall/累计推断秒、独立input-guard成本、每case绝对mm/MPa误差与最差ID见[metrics.json](metrics.json)；计时有重叠，不能相加。stress独立非线性分支，不声称标量应力叠加；位移载荷线性是结构假设，不能当GT线性或工程合格证明。单seed四passes；DEV已被探索，旧final与新sealed test未读取，未据结果扩预算/调loss。"]
    if report.get("figure",{}).get("generated"):
        lines += ["","![固定端点物理误差与实际成本](comparison.png)"]
    return "\n".join(lines)+"\n"


def plot(report,filename):
    rows=report["runs"]
    completed=[mode for mode in ROUTES if rows[mode]["status"]=="completed"]
    if not completed:return {"generated":False,"reason":"No completed endpoint metrics"}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.0),constrained_layout=True)
    def errors(ax,metrics,title):
        names=[mode for mode in ROUTES if mode in metrics]
        if not names:
            ax.text(.5,.5,"Not completed",ha="center",va="center",transform=ax.transAxes)
            ax.set(title=title,xticks=[],yticks=[]);return
        width=.34
        for offset,field,color,label in ((-width/2,FIELDS[0],"#4477aa","U mean"),(width/2,FIELDS[2],"#cc6677","S mean")):
            bars=ax.bar([i+offset for i in range(len(names))],[100*metrics[name][field] for name in names],width,color=color,label=label)
            for bar in bars:ax.annotate(f"{bar.get_height():.1f}",(bar.get_x()+bar.get_width()/2,bar.get_height()),ha="center",va="bottom",fontsize=8,xytext=(0,3),textcoords="offset points")
        ax.set_xticks(list(range(len(names))),[LABELS[name].replace(" ","\n",1) for name in names],fontsize=8)
        ax.set(ylabel="Physical relative L2 (%)",title=title);ax.margins(y=.18);ax.grid(axis="y",alpha=.16);ax.legend(fontsize=8)
    errors(axes[0],{name:rows[name]["validation"] for name in completed},"Known3 DEV15 / endpoint480")
    dia=report["development_dia"]
    errors(axes[1],{name:route["dia"] for name,route in dia["routes"].items()} if dia["status"]=="completed" else {},"DEV5 dia / endpoint480")
    timed=[name for name in completed if finite(rows[name].get("costs",{}).get("train_seconds"))]
    bars=axes[2].bar(list(range(len(timed))),[rows[name]["costs"]["train_seconds"] for name in timed],color=["#888888" if name=="historical_direct4" else "#66aa99" for name in timed])
    for bar in bars:axes[2].annotate(f"{bar.get_height():.1f}",(bar.get_x()+bar.get_width()/2,bar.get_height()),ha="center",va="bottom",fontsize=8,xytext=(0,3),textcoords="offset points")
    axes[2].set_xticks(list(range(len(timed))),[LABELS[name].replace(" ","\n",1) for name in timed],fontsize=8)
    axes[2].set(ylabel="Actual training seconds",title="480 complete-case update cost");axes[2].margins(y=.18);axes[2].grid(axis="y",alpha=.16)
    fig.suptitle("Response-basis diagnostic; historical direct4 is not the matched control",fontsize=11)
    fig.savefig(filename,dpi=170);plt.close(fig)
    return {"generated":True,"file":filename.name,"policy":"Completed actual percent/time values only; bars, no interpolation or checkpoint selection"}


def write(filename,text):
    temporary=filename.with_suffix(filename.suffix+".tmp")
    temporary.write_text(text,encoding="utf-8");temporary.replace(filename)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",default=DEFAULT_PLAN)
    parser.add_argument("--eval-plan",default=DEFAULT_EVAL_PLAN)
    parser.add_argument("--eval-run",default=DEFAULT_EVAL_RUN)
    parser.add_argument("--output",default="results/fno_response")
    parser.add_argument("--no-plot",action="store_true")
    parser.add_argument("--check",action="store_true",help="JSON-only snapshot with no writing or plotting")
    args=parser.parse_args(argv)
    report=build_report(args.plan,args.eval_plan,args.eval_run)
    if args.check:
        print(json.dumps({"status":report["status"],"runs":{name:row["status"] for name,row in report["runs"].items()},"dia_status":report["development_dia"]["status"],"arrays_read":0,"weights_read":0,"GPU_work":False}))
        return report
    output=path(args.output);output.mkdir(parents=True,exist_ok=True)
    report["figure"]={"generated":False,"reason":"--no-plot"}
    if not args.no_plot:
        try:report["figure"]=plot(report,output/"comparison.png")
        except ImportError as error:report["figure"]={"generated":False,"reason":f"CPU Matplotlib unavailable: {error}"}
    write(output/"metrics.json",json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    write(output/"BRIEF.md",brief(report))
    print(json.dumps({"status":report["status"],"output":str(output),"figure":report["figure"]}))
    return report


if __name__=="__main__":
    main()
