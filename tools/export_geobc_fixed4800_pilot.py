"""Compact, JSON-only export of fixed4800 TRAIN120 and held-five U/S evidence.

Never imports a scientific evaluator, loads a checkpoint/array/raw field, or
recomputes a field metric. Plotting uses saved scalar metrics only. Missing or
nonterminal known-U readout stays explicitly pending, with no invented values.
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
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("transolver", "transolver_pp", "ts3", "fno", "mlp")
CASES = ("489_12/ver", "616_205/ver", "416_282/ver", "247_50/ver", "142_454/ver")
LABELS = {"transolver": "Transolver", "transolver_pp": "Transolver++", "ts3": "Transolver3",
          "fno": "FNO", "mlp": "MLP"}
TRAIN = "runs/geobc_coverage_comparison/TRAIN_FIT_DIAGNOSTIC"
STRESS = "runs/geobc_exploration/coverage_S_field_blind5_fixed4800"
KNOWN_U = "runs/geobc_exploration/coverage_known_U5_fixed4800"
OUTPUT = "results/geobc_fixed4800_pilot"
KINDS = {"TRAIN": "coverage_TRAIN120_fixed4800_fit_diagnostic",
         "S": "coverage_cad40_fixed4800_U_known_S_field_blind5",
         "known_U": "coverage_fixed4800_known_U5_CPU_readout"}
TOP_JSON = ("config.json", "summary.json", "status.json", "clock.json", "read_receipt.json",
            "prediction_barrier.json", "publication_receipt.json", "failure.json",
            "external_launch.json", "once_claim.json")


def path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", value):
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    filename = Path(value).expanduser()
    return (filename if filename.is_absolute() else ROOT / filename).resolve()


def read(filename):
    filename = path(filename)
    if filename.suffix.lower() != ".json":
        raise ValueError("Only small JSON evidence may be read")
    return json.loads(filename.read_text(encoding="utf-8-sig"))


def write(filename, value):
    filename = path(filename); filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(filename.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, filename)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def file_identity(filename):
    filename = path(filename)
    # This function is used exclusively on owned source or small JSON evidence;
    # checkpoint and prediction identities come from saved manifests, not reads.
    if filename.suffix.lower() not in (".json", ".py"):
        raise ValueError("No binary/weight/raw hashing in exporter")
    content = filename.read_bytes(); stat = filename.stat()
    return {"path": str(filename), "bytes": len(content), "mtime_ns": stat.st_mtime_ns,
            "sha256": hashlib.sha256(content).hexdigest()}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def stage(folder, name):
    folder = path(folder); filename = folder / "summary.json"
    if not filename.exists():
        return {"state": "pending", "folder": str(folder), "summary": None, "identity": None}
    summary = read(filename)
    if summary.get("kind") != KINDS[name]:
        raise ValueError("Unexpected " + name + " summary kind")
    terminal = summary.get("status") in ("completed", "failed", "completed_negative", "budget_exhausted")
    if not terminal:
        return {"state": "not_terminal", "folder": str(folder), "summary": None,
                "identity": file_identity(filename), "reported_status": summary.get("status")}
    if name != "TRAIN" and summary.get("terminal") is not True:
        raise ValueError("Independent readout lacks terminal receipt")
    if name != "TRAIN" and (summary.get("methods") != list(METHODS) or summary.get("case_keys") != list(CASES)):
        raise ValueError("Fixed five-method/five-case scope differs")
    if name == "TRAIN" and (summary.get("fixed_case_denominator_each_arm") != 120
                            or summary.get("fixed_prediction_denominator") != 600):
        raise ValueError("TRAIN120 denominator differs")
    return {"state": "terminal", "folder": str(folder), "summary": summary, "identity": file_identity(filename)}


def worst_ids(per_case, field):
    pairs = [(key, row.get(field)) for key, row in per_case.items() if row is not None and finite(row.get(field))]
    if not pairs:
        return []
    worst = max(value for _, value in pairs)
    return [key for key, value in pairs if value == worst]


def collect(train_stage, stress_stage, u_stage):
    train = train_stage["summary"]; stress = stress_stage["summary"]; known = u_stage["summary"]
    if known is not None:
        if known.get("U_is_blind") is not False or known.get("S_labels_read_by_this_stage") is not False:
            raise ValueError("known-U readout identity differs")
        if stress is None:
            raise ValueError("known-U evidence requires its independent S prediction producer")
        contract = known["config"]["contract"]
        if contract["S_summary"]["sha256"] != stress_stage["identity"]["sha256"]:
            raise ValueError("known-U readout refers to another terminal S producer")
        if contract["S_contract_sha256"] != digest(stress["config"]["contract"]):
            raise ValueError("known-U source contract differs")
    rows = {}
    for method in METHODS:
        trained = None if train is None else train["arms"].get(method + "_cad40_fresh4800")
        stressed = None if stress is None else stress["arms"].get(method)
        u = None if known is None else known["results"].get(method)
        if trained is not None and (trained["checkpoint_step"] != 4800 or trained["train_CADs"] != 40):
            raise ValueError("TRAIN fixed endpoint/scale metadata differs")
        if stressed is not None and stressed["checkpoint_step"] != 4800:
            raise ValueError("S fixed endpoint metadata differs")
        ta = {} if trained is None else trained["aggregate"]
        sa = {} if stressed is None else stressed["validation"]
        ua = {} if u is None else u["aggregate"]
        tc = (trained or {}).get("original_training_costs")
        if tc is None:
            tc = (stressed or {}).get("training_cost_context")
        per_train = (trained or {}).get("per_case", {})
        per_S = sa.get("per_case", {key: None for key in CASES})
        per_U = (u or {}).get("per_case", {key: None for key in CASES})
        for name, per in (("S", per_S), ("U", per_U)):
            if set(per) != set(CASES):
                raise ValueError(name + " fixed denominator case keys differ")
        for key in CASES:
            if per_S[key] is not None and per_U[key] is not None and per_S[key]["num_nodes"] != per_U[key]["nodes"]:
                raise ValueError("U/S original-node count differs")
        rows[method] = {
            "method": method, "display_name": LABELS[method], "seed": 2026, "checkpoint_step": 4800,
            "TRAIN120": {"fixed_denominator": 120, "aggregate": {k: v for k, v in ta.items() if k != "per_case"},
                         "per_case": per_train, "CAD_three_load_aggregate": (trained or {}).get("CAD_three_load_aggregate"),
                         "worst_U_case_keys": worst_ids(per_train, "u_relative_l2"),
                         "worst_S_case_keys": worst_ids(per_train, "stress_relative_l2")},
            "held5": {"fixed_denominator": 5,
                      "U_known": {"aggregate": {k: v for k, v in ua.items() if k != "per_case"}, "per_case": per_U,
                                  "worst_case_keys": worst_ids(per_U, "u_relative_l2"), "U_is_blind": False},
                      "S_field_blind": {"aggregate": {k: v for k, v in sa.items() if k != "per_case"}, "per_case": per_S,
                                        "worst_case_keys": worst_ids(per_S, "stress_relative_l2"),
                                        "worst_hotspot_case_keys": worst_ids(per_S, "stress_hotspot_relative_l2")}},
            "training_cost_context": tc,
            "training_minutes": tc["train_seconds"] / 60 if tc and finite(tc.get("train_seconds")) else None,
            "parameter_counts": (stressed or {}).get("parameter_counts"),
            "pilot_prediction_costs": (stressed or {}).get("costs"),
            "TRAIN_diagnostic_costs": (trained or {}).get("costs")}
    return rows


def copy_json(filename, destination, reason, inventory):
    filename = path(filename)
    if not filename.exists():
        return
    read(filename)  # Validate JSON, never infer arrays by following stored paths.
    destination = path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(filename, destination)
    inventory.append({"source": file_identity(filename), "exported": file_identity(destination), "reason": reason})


def export_evidence(output, stages):
    inventory = []; source_index = {}; checkpoint_index = {}; prediction_index = {}
    for name, value in stages.items():
        if value["summary"] is None:
            continue
        folder = path(value["folder"]); summary = value["summary"]
        for filename in TOP_JSON:
            copy_json(folder / filename, output / "evidence" / name / filename, "Authoritative small terminal metadata", inventory)
        contract = summary["config"]["contract"]
        source_index[name] = contract.get("source_sha256")
        if name == "TRAIN":
            for method in METHODS:
                arm = summary["arms"][method + "_cad40_fresh4800"]
                checkpoint_index[method] = {"TRAIN_committed_identity": arm["checkpoint"]}
        elif name == "S":
            barrier = read(folder / "prediction_barrier.json")
            if barrier.get("passed") is True and (barrier.get("required_entries") != 25 or barrier.get("finite_fullN_entries") != 25):
                raise ValueError("Saved prediction barrier inventory differs")
            for method in METHODS:
                for key in CASES:
                    source = folder / "predictions" / method / key.replace("/", "_") / "manifest.json"
                    if not source.exists():
                        continue
                    row = read(source)
                    if row.get("method") != method or row.get("key") != key:
                        raise ValueError("Prediction index metadata scope differs")
                    copy_json(source, output / "evidence" / "S" / "predictions" / method / (key.replace("/", "_") + ".json"),
                              "Metadata and existing array hashes only; no NPY copied/read", inventory)
                    prediction_index[method + ":" + key] = row
                    if row.get("prediction_complete"):
                        entry = checkpoint_index.setdefault(method, {})
                        previous = entry.get("TRAIN_committed_identity")
                        if previous and previous["sha256"] != row["checkpoint"]["sha256"]:
                            raise ValueError("TRAIN/S endpoints are different checkpoints")
                        entry["S_committed_identity"] = row["checkpoint"]
            for key in CASES:
                for filename in ("read_started.json", "manifest.json"):
                    copy_json(folder / "S_labels" / key.replace("/", "_") / filename,
                              output / "evidence" / "S" / "S_labels" / key.replace("/", "_") / filename,
                              "Single-column S claim/alignment/hash provenance; label array excluded", inventory)
    for method in METHODS:
        checkpoint_index.setdefault(method, {})
    indices = {"sources": source_index, "checkpoints": checkpoint_index, "predictions": prediction_index,
               "binary_files_read_or_copied": 0,
               "checkpoint_hash_origin": "Existing scientific-stage checkpoint byte hashes, never recalculated by exporter"}
    write(output / "source_checkpoint_index.json", indices)
    write(output / "export_manifest.json", {"kind": "fixed4800_pilot_compact_metadata_export", "files": inventory,
          "exclusion": ["raw files", "NPY/NPZ/H5/CSV", "PT/checkpoints/optimizer state", "complete prediction or target arrays"],
          "indices": "source_checkpoint_index.json", "binary_files_read_or_copied": 0})
    return indices


def percentage(value):
    return "未完成/缺失" if not finite(value) else f"{100 * value:.2f}%"


def metric(row, section, field):
    if section == "TRAIN":
        return row["TRAIN120"]["aggregate"].get(field)
    return row["held5"][section]["aggregate"].get(field)


def brief(rows, stages):
    fully_complete = all(value["summary"] is not None and value["summary"].get("completion_gate", {}).get("passed") is True
                         for value in stages.values())
    lines = ["# 固定4800端点：TRAIN拟合与五CAD读出", "",
             "三项读出已完整完成。" if fully_complete else "部分读出尚未完整完成；缺失保持空值，不作排名。",
             "固定 `489_12、616_205、416_282、247_50、142_454/ver`，每例全部原节点。U此前已揭晓；S在25份预测完成后才共享读取原signed字段。", "",
             "| 方法 | TRAIN120 U / S | 五CAD U / S / S热点 | 五CAD最坏 U / S | 原训练分钟 |",
             "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        row = rows[method]
        train = " / ".join(percentage(metric(row, "TRAIN", x)) for x in ("case_mean_u_relative_l2", "case_mean_stress_relative_l2"))
        held = " / ".join((percentage(metric(row, "U_known", "case_mean_u_relative_l2")),
                           percentage(metric(row, "S_field_blind", "case_mean_stress_relative_l2")),
                           percentage(metric(row, "S_field_blind", "case_mean_stress_hotspot_relative_l2"))))
        worst = " / ".join((percentage(metric(row, "U_known", "worst_case_u_relative_l2")),
                            percentage(metric(row, "S_field_blind", "worst_case_stress_relative_l2"))))
        minutes = "缺失" if row["training_minutes"] is None else f"{row['training_minutes']:.1f}"
        lines.append(f"| {LABELS[method]} | {train} | {held} | {worst} | {minutes} |")
    if fully_complete:
        minimum_u = min(METHODS, key=lambda m: metric(rows[m], "U_known", "case_mean_u_relative_l2"))
        minimum_s = min(METHODS, key=lambda m: metric(rows[m], "S_field_blind", "case_mean_stress_relative_l2"))
        lines += ["", f"本轮固定配方的五CAD均值U最低为 {LABELS[minimum_u]}，signed-S最低为 {LABELS[minimum_s]}；应力绝对误差、逐例最坏ID和真实成本分项见 metrics.json。"]
    lines += ["", "均值按病例等权，U为物理位移向量L2，S保持原signed MPa标签；热点为原|S|最高1%节点，零参考相对误差为null。",
              "单seed2026、同4800完整案例更新；当前配置容量和训练成本不同，不能据此判定全家族上限或工程资格。两例属官方TRAIN、三例属官方TEST，但全五均未参与本轮TRAIN40训练。",
              "TRAIN120含40CAD×三载荷，五CAD仅ver；图中差距同时包含病例/载荷组成变化，无置信区间或独立60样本假设。此pilot不替代仍冻结的20DEV/十臂覆盖试验，也未验证连续梯度或BC服务性能。",
              "", "完整原始小JSON、读取屏障、source/checkpoint索引保存在 evidence/ 与 source_checkpoint_index.json；不导出raw、权重或预测/标签数组。"]
    return "\n".join(lines) + "\n"


def plot(rows, filename):
    values = [[metric(rows[m], "TRAIN", field) for m in METHODS] for field in
              ("case_mean_u_relative_l2", "case_mean_stress_relative_l2")]
    values += [[metric(rows[m], section, field) for m in METHODS] for section, field in
               (("U_known", "case_mean_u_relative_l2"), ("S_field_blind", "case_mean_stress_relative_l2"))]
    if not all(finite(v) for group in values for v in group):
        return {"created": False, "reason": "All five actual TRAIN/held-five U/S means are required; no placeholder figure"}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), sharey=True)
    positions = list(range(len(METHODS))); width = .36
    for i, ax in enumerate(axes):
        left = [100 * v for v in values[i]]; right = [100 * v for v in values[i + 2]]
        a = ax.bar([x - width / 2 for x in positions], left, width, label="TRAIN40 x 3 loads", color="#406a9c")
        b = ax.bar([x + width / 2 for x in positions], right, width, label="Held-five ver", color="#d48942")
        ax.bar_label(a, fmt="%.1f", padding=3, fontsize=8); ax.bar_label(b, fmt="%.1f", padding=3, fontsize=8)
        ax.set_xticks(positions, ["TS", "TS++", "TS3", "FNO", "MLP"])
        ax.set_title("Vector displacement U" if i == 0 else "Original signed stress S", fontsize=11)
        ax.set_ylim(0, max(v for group in values for v in group) * 100 * 1.18)
        ax.grid(axis="y", alpha=.22); ax.set_axisbelow(True)
    axes[0].set_ylabel("Case-mean physical relative L2 (%)")
    fig.suptitle("Fixed4800 current presets | single seed2026", fontsize=12)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .105), ncol=2, frameon=False)
    fig.text(.5, .035, "U was already known; S field was blind until the 25-prediction barrier.\nDifferent case/load mixtures and capacities; no CI or family-ceiling claim.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=.08, right=.99, top=.85, bottom=.23, wspace=.12)
    filename = path(filename); fig.savefig(filename, dpi=180); plt.close(fig)
    return {"created": True, "path": str(filename), "source": "Saved JSON case-mean scalars only", "confidence_intervals": False}


def run(train_run=TRAIN, stress_run=STRESS, known_u_run=KNOWN_U, output=OUTPUT, make_plot=True):
    began = time.perf_counter()
    stages = {"TRAIN": stage(train_run, "TRAIN"), "S": stage(stress_run, "S"), "known_U": stage(known_u_run, "known_U")}
    rows = collect(stages["TRAIN"], stages["S"], stages["known_U"])
    output = path(output); output.mkdir(parents=True, exist_ok=True)
    indices = export_evidence(output, stages)
    chart = plot(rows, output / "comparison.png") if make_plot else {"created": False, "reason": "Disabled by explicit CLI"}
    result = {"kind": "fixed4800_TRAIN120_U_known_S_field_blind5_report", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "methods": list(METHODS), "case_keys": list(CASES), "fixed_held_denominator": 5, "seed": 2026, "checkpoint_step": 4800,
              "stages": {name: {"state": value["state"], "summary_identity": value["identity"],
                          "completion_gate": value["summary"].get("completion_gate") if value["summary"] else None,
                          "costs": value["summary"].get("costs") if value["summary"] else None,
                          "read_receipt": value["summary"].get("read_receipt") if value["summary"] else None,
                          "failure": value["summary"].get("failure") if value["summary"] else None} for name, value in stages.items()},
              "results": rows, "figure": chart, "source_checkpoint_index": "source_checkpoint_index.json",
              "node_prediction_index_entries": len(indices["predictions"]),
              "reader_receipt": {"raw_field_reads": 0, "scientific_array_reads": 0, "weight_reads": 0,
                                 "metric_recomputations": 0, "model_calls": 0, "GPU_calls": 0},
              "scope": {"U_is_blind": False, "S_was_field_blind": True, "untouched_file_blind": False,
                        "official_split": {"489_12": "TRAIN", "247_50": "TRAIN", "616_205": "TEST", "416_282": "TEST", "142_454": "TEST"},
                        "all_five_unused_in_current_training": True, "replaces_frozen20DEV_or10arms": False,
                        "statistical_significance_or_family_ceiling": False, "final_solver_selection": False}}
    write(output / "metrics.json", result)
    (output / "BRIEF.md").write_text(brief(rows, stages), encoding="utf-8")
    complete = (len(indices["predictions"]) == 25 and all(
        value["summary"] is not None and value["summary"].get("completion_gate", {}).get("passed") is True
        for value in stages.values()))
    identities = {}
    for name in ("metrics.json", "BRIEF.md", "source_checkpoint_index.json", "export_manifest.json", "comparison.png"):
        filename = output / name
        if filename.exists():
            content = filename.read_bytes()
            identities[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    elapsed = time.perf_counter() - began
    write(output / "manifest.json", {"status": "completed" if complete else "incomplete_or_pending",
          "all_three_readout_gates_passed": complete, "prediction_index_entries": len(indices["predictions"]),
          "formal_ten_arm_completed": False, "final_solver_selected": False,
          "report_files": identities, "export_wall_seconds_before_manifest_persist": elapsed,
          "scientific_array_or_weight_reads": 0, "source_sha256": file_identity(__file__)["sha256"]})
    return {"output": str(output), "status": "completed" if complete else "incomplete_or_pending",
            "states": {k: v["state"] for k, v in stages.items()}, "figure": chart,
            "export_wall_seconds_before_manifest_persist": elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run", default=TRAIN); parser.add_argument("--S-run", dest="stress_run", default=STRESS)
    parser.add_argument("--known-U-run", dest="known_u_run", default=KNOWN_U); parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.train_run, args.stress_run, args.known_u_run, args.output, not args.no_plot), ensure_ascii=False))


if __name__ == "__main__":
    main()
