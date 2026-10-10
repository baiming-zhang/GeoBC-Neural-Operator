#!/usr/bin/env python3
"""Export the completed ten-arm GeoBC screen using small JSON metadata only.

No model, scientific data loader, raw/target/prediction array, weight or GPU is
opened. Matplotlib is imported only after the immutable readout gate passes.
This report tool is outside the frozen scientific source map.
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
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_coverage_comparison_plan.json"
OUTPUT = "results/geobc_coverage_comparison"
KIND = "geobc_solver_coverage40vs200_transolver_all_versions"
METHODS = ("transolver", "transolver_pp", "ts3", "fno", "mlp")
STEPS = (0, 1200, 2400, 4800)
LABELS = {"transolver": "Transolver", "transolver_pp": "Transolver++", "ts3": "Transolver-3",
          "fno": "FNO", "mlp": "MLP"}
COLORS = {"transolver": "#0072B2", "transolver_pp": "#D55E00", "ts3": "#009E73",
          "fno": "#CC79A7", "mlp": "#7B6D46"}


def path(value):
    text = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", text):
        text = text[5].upper() + ":/" + text[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", text):
        text = "/mnt/" + text[0].lower() + "/" + text[3:]
    result = Path(text)
    return (result if result.is_absolute() else ROOT / result).resolve()


def relative(value):
    return path(value).relative_to(ROOT.resolve()).as_posix()


def read(value):
    return json.loads(path(value).read_text(encoding="utf-8-sig"))


def sha(value):
    h = hashlib.sha256()
    with path(value).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(value, payload):
    filename = path(value); filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(filename.name + ".tmp")
    if isinstance(payload, str):
        temporary.write_text(payload, encoding="utf-8")
    else:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, filename)


def load_completed(plan_path=PLAN, summary_path=None):
    """Read completed metadata only, with the frozen case/arm/barrier identities."""
    plan = read(plan_path); science = plan["scientific"]
    require(plan["kind"] == KIND and digest(science) == plan["scientific_digest"], "Scientific plan identity differs")
    run = path(plan["training_output"])
    summary_path = path(summary_path) if summary_path else run / "evaluation/summary.json"
    summary = read(summary_path)
    require(summary.get("status") == "completed" and summary.get("stage") == "evaluate"
            and summary.get("kind") == KIND, "Formal completed evaluation is required; no incomplete export")
    contract = summary["config"]["contract"]
    require(contract["scientific_digest"] == plan["scientific_digest"] and contract["plan_science"] == science
            and contract["source_sha256"] == plan["source_sha256"], "Completed readout source/science identity differs")
    expected_arms = science["arm_order"]
    require(len(expected_arms) == 10 and len({a["id"] for a in expected_arms}) == 10
            and {(a["method"], a["train_CADs"]) for a in expected_arms} == {(m, n) for m in METHODS for n in (40, 200)},
            "Fixed five-family/two-scale inventory differs")
    validation = [cad + "/" + load for cad in science["DEV_CADs"] for load in ("ver", "hor", "tor")]
    require(len(validation) == 60 and len(set(validation)) == 60 and summary["validation_keys"] == validation
            and summary["fixed_DEV_denominator"] == 60 and contract["validation_keys"] == validation,
            "Fixed60 DEV inventory differs")
    require(set(summary["arms"]) == {a["id"] for a in expected_arms}, "Completed arm inventory differs")
    for arm in expected_arms:
        actual = summary["arms"][arm["id"]]
        require(actual["method"] == arm["method"] and actual["train_CADs"] == arm["train_CADs"]
                and set(actual["endpoints"]) == {str(s) for s in STEPS}, "Frozen endpoint/family differs")
        for step in STEPS:
            endpoint = actual["endpoints"][str(step)]
            require(endpoint["step"] == step and endpoint["fixed_checkpoint"] is True
                    and endpoint["validation"]["fixed_denominator"] == 60
                    and set(endpoint["validation"]["per_case"]) == set(validation), "Endpoint denominator/per-case inventory differs")
    proof = summary["prediction_barrier"]
    require(proof.get("passed") is True and proof.get("status") == "completed" and proof.get("required_entries") == 2400
            and len(proof["prediction_manifests"]) == len(proof["required_predictions"]) == 2400
            and proof["finite_fullN_entries"] + proof["terminal_failure_entries"] == 2400
            and proof.get("training_permanently_closed") is True and proof["DEV_target_reads_before_barrier"] == 0,
            "Complete closed2400 prediction-or-failure barrier required")
    require(proof["input_contract_sha256"] == contract["input_contract_sha256"], "Barrier input identity differs")
    input_manifest = read(path(plan["data_cache"]) / "manifest.json")
    labels = read(path(plan["data_cache"]) / "DEV_targets/manifest.json")
    require(input_manifest["status"] == "completed" and input_manifest["feasibility_pass"] is True
            and labels["status"] == "completed" and labels["training_permanently_closed"] is True
            and input_manifest["contract"] == labels["contract"] == contract["input_contract"]
            and labels["barrier_sha256"] == digest(proof) and set(labels["records"]) == set(validation),
            "Completed shared input/DEV readout identity differs")
    return {"plan": plan, "science": science, "summary": summary, "summary_path": summary_path,
            "run": run, "contract": contract, "inputs": input_manifest, "labels": labels, "proof": proof}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def percentage(value):
    return f"{100 * value:.2f}%" if finite(value) else "缺失"


def minutes(value):
    return f"{value / 60:.1f}" if finite(value) else "缺失"


def normalized_arms(context):
    summary = context["summary"]; result = {}
    for spec in context["science"]["arm_order"]:
        arm_id = spec["id"]; item = summary["arms"][arm_id]; training = item["training"]
        config = training["config"]; primary = item["primary"]["validation"]
        folder = context["run"] / arm_id
        pred_file = folder / "prediction_summary.json"
        prediction = read(pred_file) if pred_file.exists() else None
        if prediction is not None:
            require(prediction["status"] == "completed" and prediction["arm_id"] == arm_id, "Prediction cost metadata not terminal")
        all60 = primary.get("all_cases_complete") is True
        eligible = item["completed4800"] is True and all60 and finite(primary.get("selection_score"))
        costs = training.get("costs", {})
        recorded = [costs.get(name) for name in ("train_seconds", "initialization_seconds", "checkpoint_seconds", "TRAIN_readout_seconds")]
        result[arm_id] = {"method": spec["method"], "label": LABELS[spec["method"]], "train_CADs": spec["train_CADs"],
            "train_cases": spec["train_cases"], "declared_complete_passes": spec["complete_passes"],
            "completed4800": item["completed4800"], "committed_updates": training["endpoint_step"],
            "stop_reason": training["stop_reason"], "ranking_eligible": eligible, "all60_primary_complete": all60,
            "primary": item["primary"], "endpoints": item["endpoints"], "training_costs": costs,
            "recorded_training_component_sum_seconds": sum(recorded) if all(finite(v) for v in recorded) else None,
            "prediction_costs": prediction.get("costs") if prediction else None,
            "cost_definition": "train_seconds=accumulated complete-case update IO/forward/backward/Adam; initialization/checkpoints/TRAIN readouts separate. Prediction costs are warm model/input/checkpoint stages, not a fresh cold service.",
            "memory": training.get("memory"), "parameter_counts": config.get("parameter_counts"),
            "model": config.get("model"), "precision": config.get("precision"), "initial_parameters": config.get("initial_parameters"),
            "case_order_sha256": config.get("case_order_sha256"), "case_exposure_counts": training.get("case_exposure_counts"),
            "train_input_phase": config.get("contract", {}).get("input_phase"),
            "training_config": config, "best_checkpoint_selection_performed": False}
    return result


def checkpoint_index(context):
    rows = []
    key = context["summary"]["validation_keys"][0]
    for spec in context["science"]["arm_order"]:
        for step in STEPS:
            token = f"{spec['id']}:{step}:{key}"
            reference = context["proof"]["prediction_manifests"][token]
            manifest = read(reference["path"])
            require(manifest["arm"] == spec["id"] and manifest["step"] == step and manifest["key"] == key,
                    "Checkpoint-index prediction identity differs")
            checkpoint = manifest.get("checkpoint")
            filename = context["run"] / spec["id"] / f"step{step}.pt"
            stat = filename.stat() if filename.is_file() else None
            if checkpoint is not None and stat is not None:
                require(stat.st_size == checkpoint["bytes"] and stat.st_mtime_ns == checkpoint["mtime_ns"],
                        "Checkpoint stat changed after frozen inference")
            rows.append({"arm": spec["id"], "step": step, "path": relative(filename), "available_locally": stat is not None,
                "bytes": stat.st_size if stat else None, "mtime_ns": stat.st_mtime_ns if stat else None,
                "sha256": checkpoint.get("sha256") if checkpoint else None,
                "sha256_origin": "Already recorded by the frozen prediction stage; exporter does not read weights",
                "included_in_repository": False, "predeclared_endpoint": True, "prediction_manifest": reference})
    return rows


def brief(context, arms):
    eligible = [(key, arm) for key, arm in arms.items() if arm["ranking_eligible"]]
    eligible.sort(key=lambda pair: pair[1]["primary"]["validation"]["selection_score"])
    incomplete = [key for key, item in arms.items() if not item["ranking_eligible"]]
    if eligible:
        key, winner = eligible[0]; v = winner["primary"]["validation"]
        lead = (f"本固定配方筛选分数最低：**{winner['label']} / TRAIN{winner['train_CADs']}**，"
                f"DEV60节点U均值{percentage(v['case_mean_u_relative_l2'])}、signed S均值{percentage(v['case_mean_stress_relative_l2'])}。")
    else:
        lead = "当前配方没有同时完成4800更新与DEV60全节点指标的候选，不能给出完整候选排名。"
    rows = [lead, "", "| 方法 | U均值：40→200 | signed S均值：40→200 | 更新训练分钟：40→200 |",
            "|---|---:|---:|---:|"]
    by = {(a["method"], a["train_CADs"]): a for a in arms.values()}
    for method in METHODS:
        a, b = by[(method, 40)], by[(method, 200)]
        x, y = a["primary"]["validation"], b["primary"]["validation"]
        rows.append(f"| {LABELS[method]} | {percentage(x.get('case_mean_u_relative_l2'))}→{percentage(y.get('case_mean_u_relative_l2'))} | "
                    f"{percentage(x.get('case_mean_stress_relative_l2'))}→{percentage(y.get('case_mean_stress_relative_l2'))} | "
                    f"{minutes(a['training_costs'].get('train_seconds'))}→{minutes(b['training_costs'].get('train_seconds'))} |")
    reference = by[("fno", 200)]; comparisons = []
    for method in METHODS[:3]:
        candidate = by[(method, 200)]; x = candidate["primary"]["validation"].get("case_mean_stress_relative_l2")
        y = reference["primary"]["validation"].get("case_mean_stress_relative_l2")
        tx, ty = candidate["training_costs"].get("train_seconds"), reference["training_costs"].get("train_seconds")
        if candidate["ranking_eligible"] and reference["ranking_eligible"] and finite(y) and y > 0:
            cost = f"，更新训练耗时{tx / ty:.2f}倍" if finite(tx) and finite(ty) and ty > 0 else ""
            comparisons.append(f"{LABELS[method]}的S误差为FNO的{x / y:.2f}倍{cost}")
        else:
            comparisons.append(f"{LABELS[method]}与FNO的200CAD完整对比缺失")
    rows += ["", "200CAD下，" + "；".join(comparisons) + "。"]
    gains = [LABELS[m] for m, item in context["summary"]["coverage_comparisons"].items() if item["passed"]]
    rows.append("40→200通过预声明覆盖收益门：" + ("、".join(gains) if gains else "暂无") + "。")
    if incomplete:
        rows.append("未获完整排名资格：" + "、".join(incomplete) + "；具体缺失/预算/后端原因见详细指标。")
    else:
        rows.append("10臂均完成固定4800更新和DEV60指标。")
    rows += ["", "这是单seed2026、同原25输入和TRAIN40固定scaler的原节点U/S筛选；家族容量不同。"
             "两规模同为4800全CAD更新，分别40与8遍；更新计时之外的初始化、保存、读出成本另列，推理计时属于暖模型阶段。"
             "本结果未资格认定连续位置/梯度或最终工程求解器。",
             "", "[精度与成本图](accuracy_cost.png) · [固定端点曲线](endpoint_curves.png) · [全部10臂/60例指标](metrics.json) · "
             "[配置和数据索引](manifest.json) · [本地checkpoint索引](checkpoint_index.json)", ""]
    return "\n".join(rows)


def plots(arms, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), constrained_layout=True)
    plotted = 0
    for item in arms.values():
        if not item["ranking_eligible"]:
            continue
        metrics = item["primary"]["validation"]; seconds = item["training_costs"].get("train_seconds")
        if not finite(seconds):
            continue
        plotted += 1
        for ax, name in zip(axes, ("case_mean_u_relative_l2", "case_mean_stress_relative_l2")):
            ax.scatter(seconds / 60, 100 * metrics[name], s=95, color=COLORS[item["method"]],
                       marker="o" if item["train_CADs"] == 40 else "^", edgecolors="white", linewidths=.7)
    for ax, field in zip(axes, ("U-vector", "Signed stress")):
        ax.set_xlabel("Accumulated complete-update wall time [min]"); ax.set_ylabel(field + " case-mean relative L2 [%]")
        ax.grid(alpha=.2); ax.set_title("Fixed4800 / all60 complete candidates")
        if not plotted:
            ax.text(.5, .5, "No complete fixed4800/all60 candidate", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10)
    handles = [Line2D([], [], color=COLORS[m], marker="o", linestyle="None", label=LABELS[m]) for m in METHODS]
    handles += [Line2D([], [], color="#555555", marker=marker, linestyle="None", label=f"TRAIN{n}") for marker, n in (("o", 40), ("^", 200))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.08), ncol=7, frameon=False)
    fig.savefig(output / "accuracy_cost.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.3), constrained_layout=True)
    for item in arms.values():
        column = 0 if item["train_CADs"] == 40 else 1
        for row, name in enumerate(("case_mean_u_relative_l2", "case_mean_stress_relative_l2")):
            values = [item["endpoints"][str(s)]["validation"].get(name) for s in STEPS]
            y = [100 * v if finite(v) else math.nan for v in values]
            axes[row, column].plot(STEPS, y, color=COLORS[item["method"]], marker="o", linewidth=1.5, label=LABELS[item["method"]])
    for column, n in enumerate((40, 200)):
        for row, field in enumerate(("U-vector", "Signed stress")):
            ax = axes[row, column]; ax.set_title(f"TRAIN{n}: fixed declared endpoints")
            ax.set_xlabel("Committed complete-case updates"); ax.set_ylabel(field + " mean relative L2 [%]")
            ax.set_xticks(STEPS); ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.savefig(output / "endpoint_curves.png", dpi=180); plt.close(fig)


def run(plan_path=PLAN, output=OUTPUT):
    began = time.monotonic(); context = load_completed(plan_path); destination = path(output)
    require(destination == (ROOT / OUTPUT).resolve(), "Exporter owns only results/geobc_coverage_comparison")
    destination.mkdir(parents=True, exist_ok=True); metadata = destination / "metadata"; metadata.mkdir(exist_ok=True)
    rows = normalized_arms(context)
    eligible = sorted((key for key, item in rows.items() if item["ranking_eligible"]),
                      key=lambda key: rows[key]["primary"]["validation"]["selection_score"])
    declaration = {"case": context["science"]["DEV_CADs"][0] + "/ver", "step": 4800,
        "methods": ["transolver", "fno"], "train_CADs": 200, "selection": "Predeclared source choices, no GT/score-dependent representative selection",
        "display_max_points": 20000, "metrics_population": "all original nodes; completed evaluator metrics only"}
    payload = {"status": "completed_export", "scientific_digest": context["plan"]["scientific_digest"],
        "fixed_DEV_denominator": 60, "arms": rows, "ranking": {"eligible_fixed4800_arms_by_declared_score": eligible,
            "definition": "0.5*(case_mean_U_relativeL2 + case_mean_signedS_relativeL2); report separate U/S/worst/hotspot, never drop failed cases",
            "final_solver_selected": False, "architecture_eliminated": False},
        "coverage_comparisons": context["summary"]["coverage_comparisons"],
        "readout_costs": context["summary"]["costs"], "readout_read_receipt": context["summary"]["read_receipt"],
        "shared_preparation_costs": {name: context["inputs"].get(name) for name in
                                    ("elapsed_seconds", "this_call_seconds", "prior_recorded_preparation_wall_seconds", "total_owned_cache_bytes")},
        "illustration_declaration": declaration, "limitations": context["summary"]["interpretation"]}
    write(destination / "metrics.json", payload); write(destination / "BRIEF.md", brief(context, rows))
    write(destination / "checkpoint_index.json", {"weights_copied": False, "weights_read_by_exporter": 0,
                                                  "checkpoints": checkpoint_index(context)})
    metadata_records = []
    for target, value, source in (
        ("scientific_plan.json", context["plan"], path(plan_path)),
        ("input_manifest.json", context["inputs"], path(context["plan"]["data_cache"]) / "manifest.json"),
        ("DEV_target_manifest_index.json", context["labels"], path(context["plan"]["data_cache"]) / "DEV_targets/manifest.json"),
        ("prediction_barrier.json", context["proof"], context["run"] / "prediction_barrier.json"),
        ("evaluation_source_identity.json", {"path": relative(context["summary_path"]), "sha256": sha(context["summary_path"]),
             "source_pins": context["contract"]["source_sha256"], "scientific_digest": context["plan"]["scientific_digest"]}, context["summary_path"])
    ):
        write(metadata / target, value)
        metadata_records.append({"path": "metadata/" + target, "source": relative(source), "bundled": "JSON metadata only"})
    plots(rows, destination)
    artifact_paths = [destination / name for name in
                      ("BRIEF.md", "metrics.json", "checkpoint_index.json", "accuracy_cost.png", "endpoint_curves.png")]
    artifact_paths += [metadata / row["path"].split("/")[-1] for row in metadata_records]
    field_index = destination / "fields/manifest.json"
    if field_index.exists():
        fields = read(field_index)
        require(fields["scientific_digest"] == context["plan"]["scientific_digest"]
                and fields["case"] == declaration["case"] and fields["step"] == 4800
                and fields["methods"] == declaration["methods"] and fields["train_CADs"] == 200,
                "Optional fixed illustration inventory differs")
        artifact_paths.append(field_index)
        for illustration in fields["illustrations"]:
            if illustration["status"] == "completed":
                image_path = path(illustration["path"])
                require(image_path.is_relative_to(destination / "fields") and image_path.suffix == ".png",
                        "Illustration path escaped the allowed PNG output")
                artifact_paths.append(image_path)
        with (destination / "BRIEF.md").open("a", encoding="utf-8") as stream:
            stream.write("\n[固定样本云图与抽样说明](fields/manifest.json)\n")
    artifacts = [{"path": relative(filename), "bytes": filename.stat().st_size, "sha256": sha(filename)}
                 for filename in sorted(artifact_paths)]
    manifest = {"status": "completed", "kind": "geobc_coverage_completed_delivery", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_digest": context["plan"]["scientific_digest"], "source_completed_evaluation": relative(context["summary_path"]),
        "source_completed_evaluation_sha256": sha(context["summary_path"]), "metadata_sources": metadata_records,
        "artifacts": artifacts, "exporter_source_sha256": {"tools/export_geobc_coverage_results.py": sha(__file__)},
        "elapsed_seconds": time.monotonic() - began, "read_receipt": {"scientific_array_reads": 0,
            "raw_or_GT_array_reads": 0, "weight_reads": 0, "model_or_scientific_module_imports": 0, "GPU_calls": 0},
        "omitted": ["raw H5/CSV/decks", "GT arrays", "full-node predictions", "checkpoint weights"],
        "optional_field_renderer": "python -m tools.plot_geobc_coverage_fields --plan " + PLAN}
    write(destination / "manifest.json", manifest)
    print(json.dumps({"status": "completed", "output": str(destination), "rankable_arms": len(eligible),
                      "total_arms": 10, "seconds": manifest["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN); parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args(); run(args.plan, args.output)


if __name__ == "__main__":
    main()
