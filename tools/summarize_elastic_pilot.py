"""JSON-only CPU export of the single TRAIN/ver approximate physical baseline.

No trainer, model, array, raw-deck or target loader imports. Matplotlib is lazy,
and plots only actual positive true-residual observations when exporting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = "notes/geobc_elastic_pilot_plan.json"
DEFAULT_OUTPUT = "results/elastic_pilot"
FNO_BACKGROUND = "runs/geobc_exploration/solver_screen_fno_480/summary.json"
CASE = "101_479/ver"
NODES = 179316


def path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/",value):
        value = value[5].upper()+":/"+value[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/",value):
        value = "/mnt/"+value[0].lower()+"/"+value[3:]
    result = Path(value).expanduser()
    return (result if result.is_absolute() else ROOT/result).resolve()


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def fmt(value, *, percent=False):
    if not finite(value):
        return "未记录"
    return f"{100*value:.4f}%" if percent else f"{value:.6g}"


def read_json(filename, diagnostics):
    filename = path(filename)
    if not filename.is_file():
        return {},{"path":str(filename),"present":False}
    try:
        raw = filename.read_bytes()
        if len(raw) > 8*1024*1024:
            raise ValueError("Report input exceeds the declared small-JSON limit")
        value = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(value,dict):
            raise ValueError("Expected a JSON object")
        stat = filename.stat()
        return value,{"path":str(filename),"present":True,"bytes":len(raw),"mtime_ns":stat.st_mtime_ns,
            "sha256":hashlib.sha256(raw).hexdigest()}
    except (ValueError,OSError,UnicodeError) as error:
        diagnostics.append(f"{filename}: {type(error).__name__}: {error}")
        return {},{"path":str(filename),"present":True,"invalid":True}


def write_json(filename, value):
    filename = path(filename); filename.parent.mkdir(parents=True,exist_ok=True)
    temporary = filename.with_suffix(filename.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(filename)


def _background(diagnostics):
    summary,identity = read_json(FNO_BACKGROUND,diagnostics)
    final = summary.get("final_metrics",{})
    row = final.get("train_readout",{}).get("per_case",{}).get(CASE)
    if (summary.get("status") != "completed" or summary.get("endpoint_step") != 480 or final.get("step") != 480
        or not isinstance(row,dict) or row.get("nodes") != NODES):
        return None
    values = {key:row.get(key) for key in ("u_relative_l2","u_vector_error_RMS_mm","u_max_vector_error_mm")}
    if not all(finite(value) for value in values.values()):
        return None
    return {"case":CASE,"nodes":NODES,"selected_step":480,"U_only":values,
        "train_seconds_all120_cases":summary.get("costs",{}).get("train_seconds"),"source":identity,
        "scope":"Historical FNO full480 multi-CAD training readout on this TRAIN case; no new inference and no same-budget/VAL ranking"}


def _trace(solve, diagnostics):
    observations = []; omitted = 0
    for index,point in enumerate(solve.get("trace",[])):
        if not isinstance(point,dict):
            omitted += 1; continue
        seconds = point.get("cumulative_solve_wall_seconds"); residual = point.get("true_relative_residual")
        if not finite(seconds) or seconds < 0 or not finite(residual) or residual < 0:
            omitted += 1; continue
        observations.append({"iteration":point.get("iteration"),"cumulative_solve_wall_seconds":seconds,
            "true_relative_residual":residual,"source_index":index,"reason":point.get("reason")})
    if omitted:
        diagnostics.append(f"{omitted} invalid/unrecorded trace entries excluded; no synthetic replacements")
    return observations


def build_report(plan_path=DEFAULT_PLAN):
    diagnostics = []; plan,plan_identity = read_json(plan_path,diagnostics)
    if not plan:
        return {"status":"incomplete","outcome":"plan_unavailable","diagnostics":diagnostics,"sources":{"plan":plan_identity},"stages":{},"trace":[]}
    stages = {}; identities = {"plan":plan_identity}
    for name,directory,filename in (("input",plan["data_output"],"manifest.json"),
            ("operator",plan["operator_output"],"summary.json"),("solve",plan["solve_output"],"summary.json")):
        stages[name],identities[name] = read_json(path(directory)/filename,diagnostics)
    data,operator,solve = (stages[name] for name in ("input","operator","solve"))
    checks = {"declared_single_TRAIN_ver":plan.get("case") == CASE and plan.get("expected_geometry",{}).get("nodes") == NODES}
    status,outcome = "incomplete","awaiting_input"
    if data.get("status") == "failed":
        status,outcome = "failed","input_source_exception"
    elif data.get("status") == "completed":
        checks["input_kind"] = data.get("kind") == "elastic_pilot_input"
        checks["input_zero_labels_stress_updates"] = all(data.get(key) == 0 for key in ("labels_read","stress_reads","training_updates","scaler_fits"))
        if data.get("feasibility_pass") is not True:
            status,outcome = "completed","input_negative"
        elif operator.get("status") == "failed":
            status,outcome = "failed","operator_exception"
        elif operator.get("status") != "completed":
            outcome = "awaiting_operator"
        else:
            checks["operator_kind"] = operator.get("kind") == "elastic_input_operator"
            checks["same_input_contract"] = operator.get("config",{}).get("contract",{}).get("input_contract") == data.get("contract")
            if operator.get("feasibility_pass") is not True:
                status,outcome = "completed","operator_negative"
            elif solve.get("status") == "failed":
                status,outcome = "failed","solve_exception"
            elif solve.get("status") != "completed":
                outcome = "awaiting_solve"
            else:
                checks["operator_eight_guard_K_calls"] = operator.get("algebra_guards",{}).get("passed") is True and operator["algebra_guards"].get("K_applications") == 8
                receipt = operator.get("read_receipt",{})
                checks["operator_zero_GT_model_updates"] = all(receipt.get(key) == 0 for key in ("displacement_reads","stress_reads","model_or_weight_reads","optimizer_steps","scaler_fits"))
                checks["solve_kind"] = solve.get("kind") == "elastic_physical_solve"
                checks["same_operator_solve_contract"] = solve.get("config",{}).get("contract") == operator.get("config",{}).get("contract")
                numerical = solve.get("numerical_state")
                status = "completed"
                if numerical == "breakdown":
                    outcome = "numerical_breakdown"
                    checks["breakdown_not_marked_converged"] = solve.get("numerically_converged") is False
                    checks["breakdown_no_GT"] = solve.get("signed_displacement_reads") == solve.get("stress_reads") == 0 and not solve.get("metrics")
                elif numerical in {"converged","iteration_budget_limit","wall_budget_limit"}:
                    outcome = "numerically_converged" if numerical == "converged" else "finite_budget_endpoint_unconverged"
                    metrics = solve.get("metrics",{}); receipt = solve.get("read_receipt",{})
                    checks["physical_all_original_nodes"] = metrics.get("nodes") == NODES and all(finite(metrics.get(key)) for key in ("vector_RMS_mm","max_nodal_vector_error_mm"))
                    checks["one_U_read_after_prediction_equilibrium"] = receipt.get("signed_displacement_member_reads") == 1 and receipt.get("prediction_and_equilibrium_persisted_before_GT") is True
                    checks["zero_stress_other_GT_training_scaler"] = all(receipt.get(key) == 0 for key in ("stress_reads","old_final_reads","DEV_dia_reads","sealed_new_test_reads","optimizer_steps","scaler_fits"))
                    residual = solve.get("true_relative_residual")
                    checks["convergence_distinct_from_endpoint"] = (finite(residual) and residual >= 0 and
                        ((numerical == "converged" and residual <= 1e-8 and solve.get("numerically_converged") is True)
                         or (numerical != "converged" and residual > 1e-8 and solve.get("numerically_converged") is False)))
                else:
                    outcome = "unknown_completed_numerical_state"; checks["recognized_numerical_state"] = False
    if not all(checks.values()):
        status = "invalid_receipt"
        diagnostics.append("Declared metadata checks failed; no success claim is produced")
    report = {"status":status,"outcome":outcome,"kind":"elastic_physical_pilot_report","case":CASE,"nodes":NODES,
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"sources":identities,"checks":checks,"diagnostics":diagnostics,
        "stages":stages,"trace":_trace(solve,diagnostics),"numerical_state":solve.get("numerical_state"),
        "true_relative_residual":solve.get("true_relative_residual"),"metrics":solve.get("metrics"),"equilibrium":solve.get("equilibrium"),
        "costs":{"input":data.get("costs"),"input_preparation_seconds":data.get("preparation_seconds"),
            "operator":operator.get("costs"),"solve":solve.get("costs")},"memory":{"operator":operator.get("memory"),"solve":solve.get("memory")},
        "FNO480_same_TRAIN_case_background":_background(diagnostics),"plot":{},
        "limits":["One fixed TRAIN/ver physical baseline, not cross-CAD neural-operator evidence",
            "Four-point curved-P2 quadrature and weighted-LS RBE3 are explicit approximations, not OptiStruct exact",
            "Numerical convergence is distinct from displacement-label accuracy and engineering validation",
            "GT is read after the frozen endpoint; it must not select load, tolerance, budget or preconditioner",
            "Signed scalar stress is not compared or reconstructed; old final/DEV dia/new sealed inputs stay closed",
            "Cost subfields can overlap: checkpoint/K/preconditioner are inside cumulative solve wall; geometry/preconditioner/guards inside operator wall. Do not sum nested timers"]}
    return report


def brief(report):
    lines = ["# 单 TRAIN/ver 近似物理基线", "",f"范围：{CASE}，全原节点179316；报告状态 `{report['status']}`，读出 `{report['outcome']}`。",""]
    stages = report.get("stages",{}); data = stages.get("input",{}); operator = stages.get("operator",{}); solve = stages.get("solve",{})
    if data.get("status") != "completed" or data.get("feasibility_pass") is not True:
        reason = data.get("failure",{}).get("message",data.get("error","输入尚未完成"))
        lines += [f"输入：{reason}。没有可报告的物理解。",""]
    elif operator.get("status") != "completed" or operator.get("feasibility_pass") is not True:
        lines += [f"输入算子：{operator.get('reason',operator.get('error','尚未完成'))}。尚无可报告的求解精度。",""]
    else:
        guards = operator.get("algebra_guards",{})
        lines += [f"输入/算子：全 TET10 关联与真实约束/载荷 receipt 已保存；{guards.get('K_applications','未记录')}次 K 检查，几何/代数可行性 `{operator.get('feasibility_pass')}`。输入与算子共同耗时 {fmt(operator.get('costs',{}).get('combined_data_operator_wall_seconds'))} s。",""]
        if solve.get("status") != "completed":
            lines += ["求解尚未完成；不写精度或收敛结论。",""]
        else:
            costs = solve.get("costs",{}); metrics = solve.get("metrics",{})
            lines += [f"求解：`{solve.get('numerical_state','未记录')}`，迭代 {solve.get('iterations_executed_total','未记录')}，终点真实相对残差 {fmt(solve.get('true_relative_residual'))}（门限1e−8）；累计 solve wall {fmt(costs.get('cumulative_solve_wall_seconds'))} s。预算终点或 breakdown 均不算数值成功。",""]
            if metrics:
                lines += [f"冻结解后仅一次 TRAIN 位移读出：全N U relative-L2 {fmt(metrics.get('u_vector_relative_l2'),percent=True)}，vector RMS {fmt(metrics.get('vector_RMS_mm'))} mm，最大节点 vector error {fmt(metrics.get('max_nodal_vector_error_mm'))} mm；固定 support/load/interior 分项在 metrics.json。",""]
                equilibrium = solve.get("equilibrium",{})
                lines += [f"平衡：自由残差 {fmt(equilibrium.get('true_free_residual_relative'))}；含反力净力 N `{equilibrium.get('net_force_with_reaction_N')}`、净矩 N·mm `{equilibrium.get('net_moment_with_reaction_Nmm')}`。预测与平衡先保存再读GT；stress读取0。",""]
            else:
                lines += [f"没有位移标签读出：{solve.get('reason',solve.get('error','未记录原因'))}。",""]
            memory = solve.get("memory",{})
            allocated,reserved = memory.get("peak_allocated_bytes"),memory.get("peak_reserved_bytes")
            allocated_mb = allocated/1e6 if finite(allocated) else None
            reserved_mb = reserved/1e6 if finite(reserved) else None
            lines += [f"本次调用成本分列：setup/read {fmt(costs.get('solve_setup_read_seconds'))} s，checkpoint {fmt(costs.get('checkpoint_seconds'))} s，终点verification {fmt(costs.get('endpoint_verification_seconds'))} s，GT/metrics {fmt(costs.get('GT_read_and_metrics_seconds'))} s；GPU peak allocated/reserved {fmt(allocated_mb)}/{fmt(reserved_mb)} MB。checkpoint/K/preconditioner属于solve子计时，不能再次相加；完整累计与本次调用字段保留在 JSON。",""]
    history = report.get("FNO480_same_TRAIN_case_background")
    if history:
        lines += [f"历史背景：原 FNO480 对同一个 TRAIN/ver 的 U relative-L2 {fmt(history['U_only']['u_relative_l2'],percent=True)}（480次、多CAD训练总耗时 {fmt(history.get('train_seconds_all120_cases'))} s）。只作这一案例背景，不与600秒物理求解或VAL排行混为同预算对照。",""]
    lines += ["这是一例近似物理基线：4Q曲线P2积分和 weighted-LS RBE3均非 OptiStruct exact；小残差不保证标签一致或工程验证。未比较语义未明的标量应力，未训练/拟合scaler，旧final、DEVdia及新sealed关闭。GT不能用于停机或调参。"]
    if report.get("diagnostics"):
        lines += ["", "元数据诊断："+"；".join(report["diagnostics"])]
    return "\n".join(lines)+"\n"


def plot_trace(report, output):
    points = report.get("trace",[])
    positive = [p for p in points if p["true_relative_residual"] > 0]
    metadata = {"actual_true_residual_observations":len(points),"positive_logscale_observations":len(positive),
        "zero_residuals_not_drawn_on_logscale":len(points)-len(positive),"interpolation":"none","smoothed":False}
    if len(positive) < 2:
        metadata.update(status="not_created",reason="Fewer than two actual positive true-residual observations; no placeholder figure")
        return metadata
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure,axis = plt.subplots(figsize=(7.2,4.3),layout="constrained")
    axis.scatter([p["cumulative_solve_wall_seconds"] for p in positive],[p["true_relative_residual"] for p in positive],s=18,color="#236da2",label="Actual true residual")
    axis.axhline(1e-8,color="#bf6e27",linestyle="--",linewidth=1,label="Frozen tolerance 1e-8")
    axis.set_yscale("log"); axis.set_xlabel("Cumulative solve wall seconds"); axis.set_ylabel("True relative residual")
    axis.set_title("TRAIN101_479/ver: fixed approximate elasticity PCG")
    axis.grid(True,alpha=.2); axis.legend()
    filename = output/"residual_cost.png"; figure.savefig(filename,dpi=180); plt.close(figure)
    metadata.update(status="created",path=str(filename))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",default=DEFAULT_PLAN)
    parser.add_argument("--output",default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.plan); output = path(args.output); output.mkdir(parents=True,exist_ok=True)
    report["plot"] = plot_trace(report,output)
    if report["plot"]["status"] != "created" and (output/"residual_cost.png").is_file():
        (output/"residual_cost.png").unlink()  # Remove an obsolete report image, never data/checkpoints.
    write_json(output/"metrics.json",report)
    (output/"BRIEF.md").write_text(brief(report),encoding="utf-8")
    print(json.dumps({"status":report["status"],"outcome":report["outcome"],"output":str(output),"plot":report["plot"]["status"]}),flush=True)


if __name__ == "__main__":
    main()
