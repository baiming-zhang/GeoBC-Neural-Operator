"""Small-JSON CPU report of the fixed480 GeometryBC DeepONet/concat screen.

Read completed summaries and input-only manifest receipts, never arrays,
weights, solvers or trainers. Matplotlib is lazy and only creates one figure
when the parent actually exports this report. Pending values remain absent.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

# Frozen reporting helpers are standard-library only at import; no model/data imports.
if __package__:
    from .summarize_fno_response import finite, fmt, metric_view, path, read_json, write
else:
    from summarize_fno_response import finite, fmt, metric_view, path, read_json, write

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = "notes/geobc_deeponet_plan.json"
DEFAULT_DIA = "runs/geobc_exploration/deeponet_bc_development_dia480"
MODES = ("deeponet", "concat_mlp")
LABELS = {"deeponet": "DeepONet", "concat_mlp": "Concat MLP"}
METRICS = ("case_mean_u_relative_l2", "worst_case_u_relative_l2", "case_mean_stress_relative_l2",
           "worst_case_stress_relative_l2", "case_mean_stress_hotspot_relative_l2")
HISTORY = {
    "fno_force_free_u9": ("FNO force-free u9", "runs/geobc_exploration/fno_response_linear_basis_seed2026_480"),
    "fno_retained_u9": ("FNO retained-load u9", "runs/geobc_exploration/fno_response_load_dependent_basis_seed2026_480"),
    "fno_direct4": ("FNO direct4", "runs/geobc_exploration/solver_screen_fno_480"),
    "mlp_direct4": ("MLP direct4", "runs/geobc_exploration/solver_screen_mlp_480")}


def aggregate_without_report_metadata(record):
    return {key:value for key,value in record.items() if key not in ("worst_case_ids", "physical_units", "hotspot_definition")}


def training_view(mode, directory, *, historical=False, label=None):
    row = {"mode": mode, "label": label or LABELS[mode], "historical": historical,
           "status": "pending_run_definition", "validation": None, "train_readout": None,
           "costs": None, "memory": None, "parameter_counts": None, "sources": {}, "diagnostics": []}
    if not directory:
        return row
    directory = path(directory); row["output"] = str(directory)
    summary,row["sources"]["summary"] = read_json(directory/"summary.json",row["diagnostics"])
    state,row["sources"]["status"] = read_json(directory/"status.json",row["diagnostics"])
    if summary.get("status") != "completed":
        row["status"] = state.get("status",state.get("phase",summary.get("status","not_completed")))
        row["running_step"] = state.get("step",summary.get("completed_step"))
        return row
    final = summary.get("final_metrics",{})
    validation = metric_view(final.get("validation"),15,row["diagnostics"],row["label"]+" known15")
    costs,exposures = summary.get("costs",{}),summary.get("case_exposure_counts",{})
    fixed = (summary.get("endpoint_step") == final.get("step") == 480 and validation is not None
             and len(exposures) == 120 and all(value == 4 for value in exposures.values())
             and costs.get("train_case_exposures") == 480 and isinstance(costs.get("train_node_exposures"),int)
             and costs["train_node_exposures"] > 0 and finite(costs.get("train_seconds")) and costs["train_seconds"] >= 0)
    if not historical:
        fixed = fixed and summary.get("kind") == "deeponet_screen" and summary.get("mode") == mode
    if not fixed:
        row["status"] = "invalid_completed_summary"
        row["diagnostics"].append("Not a complete fixed480/four-pass/all-original-node known15 receipt")
        return row
    config = summary.get("config",{})
    row.update(status="completed",endpoint_step=480,validation=validation,train_readout=final.get("train_readout"),
        costs=costs,pass_costs=summary.get("pass_costs"),elapsed_seconds=summary.get("elapsed_seconds"),
        memory=summary.get("memory"),parameter_counts=summary.get("parameter_counts"),
        case_exposure_counts=exposures,case_order_sha256=summary.get("case_order_sha256"),
        first_actual_check=summary.get("first_actual_check"),first_update_gradient_check=summary.get("first_update_gradient_check"),
        bc_cloud_preparation=config.get("bc_cloud_preparation"),initialization_declaration=config.get("initialization"),
        config=config,checkpoint_policy="Fixed endpoint480 only; best/600-second confirmation/final-test results excluded",
        cost_scope="Actual complete-case training includes normal input/BC cloud H2D, chunk loss/backward, encoder backward, Adam/sync. Inference/evaluation/BC-read timers may overlap; do not sum as disjoint costs")
    return row


def cloud_view(cache):
    result = {"status": "not_completed", "sources": {}, "diagnostics": []}
    manifest,result["sources"]["manifest"] = read_json(path(cache)/"manifest.json",result["diagnostics"])
    state,result["sources"]["status"] = read_json(path(cache)/"status.json",result["diagnostics"])
    if manifest.get("status") != "completed":
        result["status"] = state.get("status",state.get("phase","not_completed"))
        return result
    records,receipt = manifest.get("records",{}),manifest.get("read_receipt",{})
    identities = {cad:record.get("identity",{}) for cad,record in records.items()}
    checks = {
        "actual_BC_cloud_kind": manifest.get("kind") == "deeponet_bc_cloud_input",
        "45_TRAIN_DEV_CADs": len(records) == 45 and sum(row.get("group") == "train" for row in records.values()) == 40
            and sum(row.get("group") == "validation" for row in records.values()) == 5,
        "zero_target_weight_test_dia_access": all(receipt.get(key) == 0 for key in
            ("target_arrays_read","weight_arrays_read","scaler_fits","test_or_dia_archives_read")),
        "zero_nearest_queries_and_sampling_changes": receipt.get("nearest_queries") == receipt.get("sampling_changes") == 0,
        "exact_original_coordinate_identity": len(identities) == 45 and all(row.get("max_abs_coordinate_difference") == row.get("coordinate_tolerance") == 0
            and row.get("ambiguity_allowed") is False and row.get("sampling_changed") is False
            and row.get("source_row_order_preserved") is True for row in identities.values()),
        "all_seven_role_coverage_receipts_present": len(manifest.get("role_coverage",{})) == 7
            and all(len(row.get("role_coverage",{})) == 7 for row in records.values())}
    if not all(checks.values()):
        result.update(status="invalid_completed_manifest",checks=checks)
        return result
    missing = {cad:[name for name,role in row.get("role_coverage",{}).items()
                    if role.get("source_absent") is False and role.get("represented_in_cloud") is False]
               for cad,row in records.items()}
    result.update(status="completed",kind=manifest.get("kind"),contract=manifest.get("contract"),schema=manifest.get("schema"),
        preparation_seconds=manifest.get("preparation_seconds"),read_receipt=receipt,checks=checks,
        role_coverage=manifest.get("role_coverage"),role_coverage_by_CAD={cad:row.get("role_coverage") for cad,row in records.items()},
        missing_existing_roles_by_CAD=missing,CADs_with_missing_existing_roles=[cad for cad,roles in missing.items() if roles],
        original_node_identity_by_CAD=identities,records=records,limitations=manifest.get("limitations"),
        report_policy="JSON-only identities/counts. BC input membership is not pointwise Dirichlet enforcement; missing roles preserved, no repair/arrays inspected")
    return result


def pair_view(rows, cloud):
    result = {"status":"pending", "shared_recipe_checks":None, "gain":None}
    if any(rows[mode]["status"] != "completed" for mode in MODES):
        return result
    a,b = (rows[mode] for mode in MODES)
    contracts = [{key:value for key,value in row["config"].get("contract",{}).items() if key != "mode"} for row in (a,b)]
    checks = {"same_contract_except_architecture_mode": bool(contracts[0]) and contracts[0] == contracts[1],
        "same_case_order": a["case_order_sha256"] is not None and a["case_order_sha256"] == b["case_order_sha256"],
        "same_complete_case_and_node_exposures": a["case_exposure_counts"] == b["case_exposure_counts"]
            and a["costs"]["train_node_exposures"] == b["costs"]["train_node_exposures"],
        "common_encoder_count_only": (a["parameter_counts"] or {}).get("encoder") is not None
            and (a["parameter_counts"] or {}).get("encoder") == (b["parameter_counts"] or {}).get("encoder"),
        "common_encoder_initialization_declared": bool(a["initialization_declaration"])
            and a["initialization_declaration"] == b["initialization_declaration"],
        "BC22_input_cache_complete": cloud["status"] == "completed"}
    checks["same_BC22_cache_contract"] = bool(cloud.get("contract")) and contracts[0].get("bc_cloud_contract") == cloud.get("contract")
    for mode in MODES:
        first = rows[mode]["first_actual_check"] or {}
        gradient,unit = first.get("gradient",{}),first.get("units_alpha_query",{})
        groups = gradient.get("component_L2_gradient_norms",{})
        checks[mode+"_first_real_gradient_and_units"] = (first.get("passed") is True and gradient.get("passed") is True
            and unit.get("passed") is True and first.get("geometry_encodes") == first.get("original_context_backwards") == 1
            and set(groups) == {"encoder","query_trunk","u_basis_head","stress_head"}
            and all(finite(value) and value > 0 for value in groups.values())
            and finite(gradient.get("context_leaf_L2_gradient_norm")) and gradient["context_leaf_L2_gradient_norm"] > 0)
    gain = {"deeponet_vs_concat_known_u_mean_improvement_pp":100*(b["validation"][METRICS[0]]-a["validation"][METRICS[0]]),
        "deeponet_vs_concat_known_u_worst_deterioration_pp":100*(a["validation"][METRICS[1]]-b["validation"][METRICS[1]]),
        "deeponet_vs_concat_known_S_mean_improvement_pp":100*(b["validation"][METRICS[2]]-a["validation"][METRICS[2]]),
        "deeponet_vs_concat_known_S_worst_deterioration_pp":100*(a["validation"][METRICS[3]]-b["validation"][METRICS[3]]),
        "deeponet_vs_concat_known_S_hotspot_improvement_pp":100*(b["validation"][METRICS[4]]-a["validation"][METRICS[4]])}
    return {"status":"completed", "shared_recipe_checks":checks,"shared_recipe_checks_passed":all(checks.values()),
        "gain":gain,"first_actual_receipts":{mode:rows[mode]["first_actual_check"] for mode in MODES},
        "initialization_scope":"Common encoder constructor/seed declaration, not a saved tensor-equality proof; heads/full-model init/counts differ by design",
        "capacity_match_claimed":False,"whole_model_initialization_match_claimed":False,
        "decision":"Read out real signed pp differences only; no undeclared gate/automatic winner/seed or budget selection"}


def dia_view(directory, rows):
    result = {"status":"not_completed","routes":{},"diagnostics":[],"sources":{}}
    summary,result["sources"]["summary"] = read_json(path(directory)/"summary.json",result["diagnostics"])
    state,result["sources"]["status"] = read_json(path(directory)/"status.json",result["diagnostics"])
    if summary.get("status") != "completed":
        result["status"] = state.get("status",state.get("phase",summary.get("status","not_completed")))
        return result
    routes = summary.get("routes",{})
    valid = (summary.get("kind") == "deeponet_development_dia_eval" and set(routes) == set(MODES)
        and summary.get("case_count") == 5 and summary.get("route_case_count") == 10
        and summary.get("training_updates") == summary.get("scaler_fits") == 0)
    extracted = {}
    for mode in MODES:
        route = routes.get(mode,{})
        metrics = metric_view(route.get("development_dia"),5,result["diagnostics"],mode+" DEV5 dia")
        valid = valid and metrics is not None and route.get("selected_step") == 480 and rows[mode]["status"] == "completed"
        if rows[mode]["status"] == "completed":
            valid = valid and route.get("known3") == aggregate_without_report_metadata(rows[mode]["validation"])
        extracted[mode] = {**route,"development_dia":metrics}
    if not valid:
        result.update(status="invalid_completed_summary")
        result["diagnostics"].append("DEV5/dia fixed480 scope or reused known15 endpoint differs")
        return result
    result.update(status="completed",routes=extracted,input_checks=summary.get("input_checks"),
        elapsed_seconds=summary.get("elapsed_seconds"),memory=summary.get("memory"),config=summary.get("config"),
        shared_bc_cloud_preparation_seconds=summary.get("shared_bc_cloud_preparation_seconds"),
        training_updates=0,scaler_fits=0,scope=summary.get("scope"),checkpoint_selection="Both480 endpoints; no dia-driven selection")
    return result


def build_report(plan_path=DEFAULT_PLAN, dia_run=None):
    diagnostics=[]
    plan,receipt=read_json(plan_path,diagnostics)
    directories=plan.get("runs",{})
    rows={mode:training_view(mode,directories.get(mode)) for mode in MODES}
    if dia_run is None:
        dia_run=next((job["output"] for job in plan.get("queue_jobs",[]) if job.get("kind")=="deeponet_development_dia_eval"),DEFAULT_DIA)
    cloud=cloud_view(plan.get("bc_cloud_cache","data/geobc_geometry_bc_clouds"))
    pair=pair_view(rows,cloud); dia=dia_view(dia_run,rows)
    history={key:training_view(key,directory,historical=True,label=label) for key,(label,directory) in HISTORY.items()}
    return {"schema":"geobc_deeponet_screen_report_v1", "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "status":"completed" if pair["status"] == cloud["status"] == dia["status"] == "completed" else "pending_or_partial",
        "plan":receipt,"recipe":plan.get("recipe"),"runs":rows,"pair":pair,"development_dia":dia,
        "BC_cloud_input":cloud,"historical480_context":history,"diagnostics":diagnostics,
        "sources":{"reporter_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "frozen_JSON_reporting_helpers_sha256":hashlib.sha256(Path(__file__).with_name("summarize_fno_response.py").read_bytes()).hexdigest()},
        "scope":{"primary":"Fixed480/4pass/seed2026 complete-case TRAIN120, DEVknown15; full original signed fields",
            "dia":"Previously explored DEV5 combined loads, separate structural diagnostic; no blind/final claim",
            "physical_units":{"u":"mm signed XYZ vector","source_signed_stress":"MPa"},
            "historical":"Only completed same480 oldFNOresponse/MLP/FNO; different inputs/heads/init, not matched or same-cost causal controls",
            "excluded":"Old-final/new sealed test/600-second confirmation, arrays, weights, models and GPU",
            "limitations":"Single-seed early screen, different stored/effective capacity; no statistical significance, solver qualification, scalar stress linearity, native continuous/BC acceptance"}}


def text_metrics(record):
    if record is None:return "—","—"
    return " / ".join(fmt(record[key],True) for key in METRICS[:2])," / ".join(fmt(record[key],True) for key in METRICS[2:])


def brief(report):
    rows=report["runs"]; dia=report["development_dia"]
    lines=["# GeometryBC DeepONet 结构初筛","",
        "两个fresh模式共同BC22→global32、力无关u9与独立非线性signed stress头。固定seed2026/480完整案例更新/四pass；U为全原节点mm向量，S为MPa源signed scalar。", "",
        "| known15 / endpoint480 | U mean / worst % | S mean / worst / hotspot % | actual训练秒 |",
        "|---|---:|---:|---:|"]
    for mode in MODES:
        row=rows[mode];u,s=text_metrics(row["validation"])
        label=LABELS[mode] if row["status"]=="completed" else LABELS[mode]+"（"+row["status"]+"）"
        lines.append(f"| {label} | {u} | {s} | {fmt((row['costs'] or {}).get('train_seconds'))} |")
    if report["pair"]["status"]=="completed":
        gain=report["pair"]["gain"]
        lines += ["",f"DeepONet相对concat：U/S均值改善 {gain['deeponet_vs_concat_known_u_mean_improvement_pp']:+.2f}/{gain['deeponet_vs_concat_known_S_mean_improvement_pp']:+.2f}pp；最差U/S恶化 {gain['deeponet_vs_concat_known_u_worst_deterioration_pp']:+.2f}/{gain['deeponet_vs_concat_known_S_worst_deterioration_pp']:+.2f}pp。正改善为误差降低，正恶化为变差；不预设赢家。"]
        checks="PASS" if report["pair"]["shared_recipe_checks_passed"] else "未完整证明，见JSON"
        lines += [f"共同recipe/scaler/案例与节点曝光、首实际梯度/单位收据：{checks}；不是full-model初始化或容量匹配。"]
    if dia["status"]=="completed":
        lines += ["","| DEV5 dia / 两个480端点 | U mean / worst % | S mean / worst / hotspot % |","|---|---:|---:|"]
        for mode in MODES:
            u,s=text_metrics(dia["routes"][mode]["development_dia"])
            lines.append(f"| {LABELS[mode]} | {u} | {s} |")
    else:
        lines += ["",f"DEV5 dia：{dia['status']}。未完成不填中途或best数值。"]
    history=[f"{row['label']} {fmt(row['validation'][METRICS[0]],True)}/{fmt(row['validation'][METRICS[2]],True)}%" for row in report["historical480_context"].values() if row["status"]=="completed"]
    if history:lines += ["","历史同480 known15 U/S mean（非matched）："+"；".join(history)+"。输入/架构/容量不同，不能作同成本因果结论。"]
    cloud=report["BC_cloud_input"]
    if cloud["status"]=="completed":
        lines += ["",f"45CAD BC cloud保持真实原索引与精确坐标、零target/weights读取；{len(cloud['CADs_with_missing_existing_roles'])}/45 CAD采样漏已有角色，按原样保留。共同encoder初始化仅声明同constructor/seed；full heads/容量不同。"]
    else:lines += ["",f"BC输入缓存：{cloud['status']}；尚无完整角色/身份收据。"]
    lines += ["","全节点曝光、首真正梯度/ctx一次反传、μ一次偏移/α/零载荷见证、参数/显存、wall/推断/共享准备秒与每case最差ID见[metrics.json](metrics.json)。计时可能重叠，不相加；应力不假设标量叠加。单seed早期结构诊断，不是工程资格/GT线性/连续BC证明。旧final与新sealed test未读取，不据dia选seed/best或调参。"]
    if report.get("figure",{}).get("generated"):lines += ["","![实际固定端点误差与成本](comparison.png)"]
    return "\n".join(lines)+"\n"


def plot(report, filename):
    rows=report["runs"]; names=[mode for mode in MODES if rows[mode]["status"]=="completed"]
    if not names:return {"generated":False,"reason":"No completed endpoint metrics"}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(10.7,3.8),constrained_layout=True)
    def errors(ax,records,title):
        active=[mode for mode in MODES if mode in records]
        if not active:
            ax.text(.5,.5,"Not completed",ha="center",va="center",transform=ax.transAxes)
            ax.set(title=title,xticks=[],yticks=[]);return
        for offset,field,color,label in ((-.17,METRICS[0],"#4477aa","U mean"),(.17,METRICS[2],"#cc6677","S mean")):
            bars=ax.bar([i+offset for i in range(len(active))],[100*records[mode][field] for mode in active],.34,color=color,label=label)
            for bar in bars:ax.annotate(f"{bar.get_height():.1f}",(bar.get_x()+bar.get_width()/2,bar.get_height()),ha="center",va="bottom",fontsize=8,xytext=(0,3),textcoords="offset points")
        ax.set_xticks(range(len(active)),[LABELS[mode] for mode in active]);ax.set(ylabel="Physical relative L2 (%)",title=title)
        ax.margins(y=.18);ax.grid(axis="y",alpha=.16)
        ax.legend(fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.16),ncol=2)
    errors(axes[0],{mode:rows[mode]["validation"] for mode in names},"Known15 / endpoint480")
    dia=report["development_dia"]
    errors(axes[1],{mode:row["development_dia"] for mode,row in dia["routes"].items()} if dia["status"]=="completed" else {},"DEV5 dia / endpoint480")
    bars=axes[2].bar(range(len(names)),[rows[mode]["costs"]["train_seconds"] for mode in names],color="#66aa99")
    for bar in bars:axes[2].annotate(f"{bar.get_height():.1f}",(bar.get_x()+bar.get_width()/2,bar.get_height()),ha="center",va="bottom",fontsize=8,xytext=(0,3),textcoords="offset points")
    axes[2].set_xticks(range(len(names)),[LABELS[mode] for mode in names]);axes[2].set(ylabel="Actual training seconds",title="480 complete-case cost")
    axes[2].margins(y=.18);axes[2].grid(axis="y",alpha=.16)
    fig.suptitle("Shared geometry/BC input; different heads/counts; one seed early screen",fontsize=10)
    fig.savefig(filename,dpi=170,bbox_inches="tight");plt.close(fig)
    return {"generated":True,"file":filename.name,"policy":"Completed actual percent/time bars only; no interpolation or best/seed selection"}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",default=DEFAULT_PLAN)
    parser.add_argument("--dia-run")
    parser.add_argument("--output",default="results/deeponet_screen")
    parser.add_argument("--no-plot",action="store_true")
    parser.add_argument("--check",action="store_true",help="Read only small JSON snapshot; no exports or plotting")
    args=parser.parse_args(argv);report=build_report(args.plan,args.dia_run)
    if args.check:
        print(json.dumps({"status":report["status"],"runs":{mode:row["status"] for mode,row in report["runs"].items()},
            "BC_cache":report["BC_cloud_input"]["status"],"dia":report["development_dia"]["status"],"arrays_read":0,"weights_read":0,"GPU_work":False}))
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
