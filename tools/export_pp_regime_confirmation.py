"""Export saved TRAIN-only PP regime evidence, only after the real run exits.

Default invocation is source-only: it reads no training files and writes nothing.
Explicit --export must run in the original WSL /proc namespace. This standalone
exporter never imports project model/data modules, reads PT/arrays, or runs models.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_transolver_pp_regime_plan.json"
THRESHOLD_PLAN = "notes/geobc_after_fixed4800_solver_plan_draft.json"
RUN = "runs/geobc_coverage_comparison/PP_REGIME_CONFIRMATION"
DESTINATION = "results/geobc_pp_regime_confirmation"
KIND = "transolver_pp_TRAIN120_author_regime_confirmation"
DRIVER = "geobc/transolver_pp_regime_confirmation.py"
DRIVER_SHA = "7b35c294495727ef8b117841046c3d31fda980147769a44ec00f06346cb2956e"
SCIENCE_SHA = "dad6be58506fd5079470e8d31d231ff8cf52234c23cece4007740f812164dc89"
GUARD_SHA = "dce73a03ac4ced3d68be7a22ba1e5f1dbd9893d018808631dcad67db87cdfea6"
RECOVERY_GUARD = "tools/run_geobc_pp_recovery_guard.py"
RECOVERY_REVIEW = "notes/pp_checkpoint_resume_root_review.json"
REFERENCE_SHA = "f20f2d68d1a277e05c42b05a9440572a8dd32b051b6445417c04d7d69e75e161"
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_LOSS_BYTES = 16 * 1024 * 1024
THRESHOLDS = {
    "S_case_mean_ratio_candidate_to_control_max": .8,
    "U_case_mean_ratio_candidate_to_control_max": 1.05,
    "S_worst_case_ratio_max": 1.0,
    "hotspot_worst_case_ratio_max": 1.0,
}
ZERO_RECEIPTS = (
    "DEV_target_reads", "held5_target_reads", "raw_source_reads", "old_final_reads",
    "sealed_test_reads", "DEV_dia_reads", "scaler_fits", "pretrained_weight_reads",
    "model_selection_updates", "new_data_preparations",
)
RELATIVE_FIELDS = {
    "u_relative_l2": ("case_mean_u_relative_l2", "worst_case_u_relative_l2"),
    "stress_relative_l2": ("case_mean_stress_relative_l2", "worst_case_stress_relative_l2"),
    "stress_hotspot_relative_l2": ("case_mean_stress_hotspot_relative_l2", "worst_case_stress_hotspot_relative_l2"),
}
LOSS_FIELDS = ("step", "key", "nodes", "encoded4_case_mean_loss", "lr_before_Adam",
               "beta1_before_Adam", "preclip_gradient_norm", "scheduler_last_epoch")


class NotReady(RuntimeError):
    """Closure is a hard export barrier, independent of scientific success."""


def utc():
    return datetime.now(timezone.utc).isoformat()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def mapping(value):
    return value if isinstance(value, dict) else {}


def utc_span(start, end):
    """Keep the recorded UTC axis distinct from monotonic duration."""
    try:
        first, last = (datetime.fromisoformat(value.replace("Z", "+00:00"))
                       for value in (start, end))
        if first.tzinfo is None or last.tzinfo is None:
            return None
        duration = (last - first).total_seconds()
        return duration if finite(duration) and duration >= 0 else None
    except (TypeError, ValueError, AttributeError, OverflowError):
        return None


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def local(relative):
    """Every read/write target is a fixed project-relative path."""
    candidate = (ROOT / relative).resolve()
    if ROOT not in candidate.parents:
        raise ValueError("Path must remain inside this project")
    return candidate


def json_record(relative, index):
    filename = local(relative)
    if filename.suffix != ".json":
        raise ValueError("Only JSON evidence is allowed here")
    if filename.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"JSON exceeds the scalar-evidence limit: {relative}")
    raw = filename.read_bytes()
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Evidence must be a JSON object: {relative}")
    index[relative] = {"path": relative, "bytes": len(raw),
                       "sha256": hashlib.sha256(raw).hexdigest()}
    return value


def optional_json(relative, index, issues):
    try:
        return json_record(relative, index)
    except (OSError, ValueError, UnicodeError) as error:
        issues[relative] = f"{type(error).__name__}: {error}"
        return {}


def exact_process_exited(identity):
    """A reused PID is distinct from the recorded PID + Linux start_ticks."""
    pid, ticks = identity.get("pid"), identity.get("start_ticks")
    if type(pid) is not int or pid <= 0 or type(ticks) is not int or ticks < 0:
        raise NotReady("Missing real Linux PID/start_ticks identity")
    stat_file = Path("/proc") / str(pid) / "stat"
    try:
        raw = stat_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"pid": pid, "start_ticks": ticks, "exact_process_exited": True,
                "observation": "PID absent", "observed_at_utc": utc()}
    except OSError as error:
        raise NotReady(f"Cannot verify /proc identity: {error}") from error
    try:
        fields = raw[raw.rfind(")") + 2:].split()
        actual_ticks, state = int(fields[19]), fields[0]
    except (ValueError, IndexError) as error:
        raise NotReady("Cannot parse Linux process identity") from error
    exited = actual_ticks != ticks or state in ("Z", "X")
    if not exited:
        raise NotReady(f"Recorded PP process is still alive: PID {pid}, start_ticks {ticks}")
    return {"pid": pid, "start_ticks": ticks, "exact_process_exited": True,
            "observation": "PID reused" if actual_ticks != ticks else "exited process state",
            "current_start_ticks": actual_ticks, "current_state": state, "observed_at_utc": utc()}


def closure(index):
    """Read only four compact terminal control records before metric/loss reads."""
    if os.name == "nt" or not Path("/proc").is_dir() or os.environ.get("WSL_DISTRO_NAME") != "cold-surge-ubuntu-24.04":
        raise NotReady("Explicit export requires the original WSL Linux process namespace")
    records = {name: json_record(f"{RUN}/{name}.json", index)
               for name in ("status", "clock", "execution", "external_launch")}
    status, clock, execution, guard = (records[k] for k in ("status", "clock", "execution", "external_launch"))
    recovered = local(f"{RUN}/recovery_external_launch.json").exists()
    guard_identity_valid = guard.get("guard_source_sha256") == GUARD_SHA
    if recovered:
        original = guard
        guard = json_record(f"{RUN}/recovery_external_launch.json", index)
        review = json_record(RECOVERY_REVIEW, index)
        helper_sha = hashlib.sha256(local(RECOVERY_GUARD).read_bytes()).hexdigest()
        guard_identity_valid = (review.get("passed") is True and review.get("recovery_released") is True
            and review.get("scientific_digest") == SCIENCE_SHA
            and review.get("guard_source_sha256") == guard.get("guard_source_sha256") == helper_sha
            and guard.get("kind") == "one_shot_PP_original_checkpoint_recovery_guard"
            and guard.get("scientific_digest") == SCIENCE_SHA
            and mapping(guard.get("root_review_identity")).get("path") == RECOVERY_REVIEW
            and mapping(guard.get("root_review_identity")).get("sha256") == index[RECOVERY_REVIEW]["sha256"]
            and guard.get("original_launch_sha256") == index[f"{RUN}/external_launch.json"]["sha256"]
            and original.get("guard_source_sha256") == GUARD_SHA
            and guard.get("original_guard_source_sha256") == GUARD_SHA
            and guard.get("actual_child_exited") is True
            and guard.get("native_clock_closed_at_guard_exit") is True)
        records["original_external_launch"] = original
        records["external_launch"] = guard
    if not (status.get("phase") == "terminal" and status.get("status") in ("completed", "failed")
            and status.get("kind") == KIND and status.get("TRAIN_only") is True
            and clock.get("closed") is True and clock.get("active") is False
            and clock.get("phase") == "terminal" and clock.get("closed_at_utc")):
        raise NotReady("PP actual terminal status and authoritative closed/inactive clock are required")
    identity = mapping(execution.get("run_process"))
    if not (status.get("run_process") == identity and clock.get("pid") == identity.get("pid")
            and clock.get("attempt_id") == execution.get("attempt_id")
            and execution.get("attempt_id") and guard.get("child_pid") == identity.get("pid")
            and guard.get("child_argv") == identity.get("argv")
            and "geobc.transolver_pp_regime_confirmation" in identity.get("argv", [])):
        raise NotReady("Terminal/execution/clock/external-child identities do not agree")
    if not (guard.get("status") == "terminal" and guard.get("ended_at_utc")
            and type(guard.get("exit_code")) is int and guard.get("external_budget_seconds") == 30000
            and guard_identity_valid and guard.get("automatic_replay") is False):
        raise NotReady("Authentic external guard terminal evidence is required")
    proof = exact_process_exited(identity)
    if recovered:
        proof["recovery_guard_exact_exit"] = exact_process_exited(mapping(guard.get("guard_process")))
    return records, proof


def clean(value, location, issues):
    """Keep incomplete/nonfinite evidence explicit; never filter to a subset."""
    if isinstance(value, dict):
        return {str(k): clean(v, f"{location}.{k}", issues) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v, f"{location}[{i}]", issues) for i, v in enumerate(value)]
    if type(value) is float and not math.isfinite(value):
        issues[location] = "Nonfinite saved scalar; represented as null"
        return None
    return value


def full120(block, keys, endpoint=False):
    """Check every saved case and aggregate; no new field computation."""
    block = mapping(block)
    rows, aggregate = mapping(block.get("per_case")), mapping(block.get("aggregate"))
    failures = mapping(block.get("case_failures"))
    checks = {
        "fixed_keys120": isinstance(rows, dict) and set(rows) == set(keys) and len(rows) == 120,
        "native_complete": block.get("all120_metrics_complete" if endpoint else "all120_metrics_available") is True,
        "fixed_denominator120": aggregate.get("fixed_denominator") == 120
            and aggregate.get("metrics_available_cases") == 120 and aggregate.get("all_cases_complete") is True,
        "no_case_failures": isinstance(failures, dict) and set(failures) == set(keys)
            and all(failures.get(k) is None for k in keys),
    }
    if endpoint:
        checks["native_TRAIN120"] = block.get("TRAIN_only") is True and block.get("fixed_denominator") == 120
    complete = checks["fixed_keys120"]
    scalar_rows = [rows[k] for k in keys] if complete else []
    checks["all_saved_case_scalars_finite"] = complete and all(
        isinstance(row, dict) and type(row.get("nodes")) is int and row["nodes"] > 0
        and all(not (type(v) is float and not math.isfinite(v)) for v in row.values())
        and all(finite(row.get(field)) and row[field] >= 0 for field in RELATIVE_FIELDS)
        for row in scalar_rows)
    checks["saved_aggregate_matches_all120"] = checks["all_saved_case_scalars_finite"]
    if checks["saved_aggregate_matches_all120"]:
        for field, (mean_name, worst_name) in RELATIVE_FIELDS.items():
            values = [row[field] for row in scalar_rows]
            expected = {mean_name: math.fsum(values) / 120, worst_name: max(values)}
            checks["saved_aggregate_matches_all120"] &= all(finite(aggregate.get(name))
                and math.isclose(aggregate[name], value, rel_tol=1e-11, abs_tol=1e-13)
                for name, value in expected.items())
    return {"passed": all(checks.values()), "checks": checks}


def paired(candidate, reference, candidate_gate, reference_gate):
    constraints = (
        ("case_mean_stress_relative_l2", .8), ("case_mean_u_relative_l2", 1.05),
        ("worst_case_stress_relative_l2", 1.0), ("worst_case_stress_hotspot_relative_l2", 1.0),
    )
    comparisons = {}
    for field, limit in constraints:
        new, old = candidate.get(field), reference.get(field)
        valid = candidate_gate["passed"] and reference_gate["passed"]
        valid &= finite(new) and finite(old) and new >= 0 and old > 0
        comparisons[field] = {"candidate": new, "constant4800": old, "maximum_ratio": limit,
            "ratio": new / old if valid else None, "passed": bool(valid and new <= limit * old)}
    return {"passed": candidate_gate["passed"] and reference_gate["passed"]
            and all(v["passed"] for v in comparisons.values()), "comparisons": comparisons}


def saved_loss(index, issues):
    filename, relative = local(f"{RUN}/loss.jsonl"), f"{RUN}/loss.jsonl"
    rows, errors, sequence, hasher = [], [], [], hashlib.sha256()
    try:
        if filename.stat().st_size > MAX_LOSS_BYTES:
            raise ValueError("Loss scalar log exceeds fixed24000 size limit")
        with filename.open("rb") as stream:
            for number, raw in enumerate(stream, 1):
                hasher.update(raw)
                if number > 24000 or len(raw) > 8192:
                    raise ValueError("Loss log exceeds fixed24000 line/record limit")
                try:
                    row = json.loads(raw.decode("utf-8"))
                    if not isinstance(row, dict):
                        raise ValueError("Loss row must be an object")
                    selected = {field: row.get(field) for field in LOSS_FIELDS}
                    rows.append(selected)
                    sequence.append(selected["step"])
                    if not (type(selected["step"]) is int and selected["step"] == number
                            and selected["scheduler_last_epoch"] == selected["step"]
                            and type(selected["nodes"]) is int and selected["nodes"] > 0
                            and all(finite(selected[field]) for field in LOSS_FIELDS[3:-1])
                            and selected["encoded4_case_mean_loss"] >= 0
                            and selected["lr_before_Adam"] > 0
                            and .85 - 1e-12 <= selected["beta1_before_Adam"] <= .95 + 1e-12
                            and selected["preclip_gradient_norm"] >= 0):
                        errors.append(f"line{number}: missing/nonfinite/invalid native scalar or cursor")
                except (ValueError, UnicodeError) as error:
                    errors.append(f"line{number}: {type(error).__name__}: {error}")
        index[relative] = {"path": relative, "bytes": filename.stat().st_size,
                           "sha256": hasher.hexdigest()}
    except (OSError, ValueError) as error:
        issues[relative] = f"{type(error).__name__}: {error}"
        errors.append(issues[relative])
    return rows, {"available_records": len(rows), "expected_final_records": 24000,
        "complete24000": len(rows) == 24000 and sequence == list(range(1, 24001)) and not errors,
        "errors": errors, "scope": "Saved encoded signed4 per-case optimizer loss; not full120 physical accuracy"}


def scalar_plots(rows, issues, destination):
    """Standard CPU/Agg plots of saved scalars; imported only in explicit export."""
    panels = (("encoded4_case_mean_loss", "Encoded signed4 case mean MSE (raw saved scalar)", True),
              ("lr_before_Adam", "Learning rate before Adam", False),
              ("beta1_before_Adam", "Adam beta1 before update", False),
              ("preclip_gradient_norm", "Gradient L2 norm before clip (raw saved scalar)", True))
    states = {field: {"available_points": sum(type(r.get("step")) is int and 1 <= r["step"] <= 24000
        and finite(r.get(field)) for r in rows), "available": False, "files": [], "axis": "linear"}
        for field, _, _ in panels}
    for state in states.values():
        state["missing_points"] = 24000 - state["available_points"]
    figure = None
    previous_config = os.environ.get("MPLCONFIGDIR")
    temporary_config = tempfile.TemporaryDirectory(prefix=".matplotlib_", dir=destination)
    os.environ["MPLCONFIGDIR"] = temporary_config.name
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
        with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
            figure, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
            for axis, (field, title, use_log) in zip(axes, panels):
                by_step = {r["step"]: r.get(field) for r in rows if type(r.get("step")) is int and 1 <= r["step"] <= 24000}
                steps = list(range(1, 24001))
                values = [by_step.get(step) if finite(by_step.get(step)) else float("nan") for step in steps]
                good = [value for value in values if finite(value)]
                axis.set_title(title, loc="left", fontsize=11)
                axis.set_xlim(1, 24000); axis.grid(alpha=.2)
                if good:
                    axis.plot(steps, values, color="#2463a5", linewidth=.55)
                    if use_log and all(value > 0 for value in good):
                        axis.set_yscale("log"); states[field]["axis"] = "log"
                    states[field]["available"] = True
                else:
                    axis.text(.5, .5, "MISSING: no finite saved scalar", transform=axis.transAxes, ha="center")
                axis.text(.995, .94, f"{states[field]['available_points']}/24000 saved points", transform=axis.transAxes,
                          ha="right", va="top", fontsize=9)
            axes[-1].set_xlabel("Committed Adam updates")
            figure.suptitle("PP TRAIN-only optimizer traces (saved scalars; not physical accuracy)", fontsize=12)
            artifacts = {}
            for suffix in ("png", "svg"):
                buffer = io.BytesIO(); figure.savefig(buffer, format=suffix, dpi=160)
                artifacts[f"optimizer_traces.{suffix}"] = buffer.getvalue()
            for state in states.values():
                state["files"] = list(artifacts)
            return artifacts, states
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        issues["optimizer_plots"] = f"{type(error).__name__}: {error}; plots unavailable"
        for state in states.values():
            state["available"] = False; state["files"] = []; state["error"] = issues["optimizer_plots"]
        return {}, states
    finally:
        if figure is not None:
            plt.close(figure)
        if previous_config is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = previous_config
        temporary_config.cleanup()


def percentage(value):
    return f"{value*100:.2f}%" if finite(value) else "缺失"


def brief(report):
    qualified = report["qualification"]["passed"]
    state = "通过完整实验资格" if qualified else "未通过完整实验资格"
    physical_gate = report["paired_comparisons"]["24000"]["passed"]
    lines = [f"++ 作者训练制度确认：{state}。实际更新 {report['native_summary'].get('endpoint_step', '缺失')}/24000；"
             f"原生状态 {report['native_summary'].get('status', '缺失')} / {report['native_summary'].get('stop_reason', '缺失')}。",
             f"完整TRAIN120物理指标改善幅度门：{'通过' if physical_gate else '未通过'}；它与成本/过程完整性资格分别报告。",
             "原DeepJEB，原40CAD×ver/hor/tor=120；signed UXYZ(mm)/S(MPa)，seed2026。下表相对L2均为无量纲，热点使用原top1% |S|。",
             "", "| 固定TRAIN120端点 | U均值 | S均值 | 热点均值 | 最坏S | 最坏热点S | 完整finite |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for label in ("old_constant4800", "new_bundle4800", "new_bundle24000", "FNO_reference4800"):
        block = report["endpoints"][label]
        a = mapping(block.get("aggregate")) if block["validation"]["passed"] else {}
        fields = ("case_mean_u_relative_l2", "case_mean_stress_relative_l2", "case_mean_stress_hotspot_relative_l2",
                  "worst_case_stress_relative_l2", "worst_case_stress_hotspot_relative_l2")
        lines.append(f"| {label} | " + " | ".join(percentage(a.get(f)) for f in fields)
                     + f" | {block['validation']['passed']} |")
    c, outer = report["authoritative_clock"], report["external_guard"]
    axes = report["recorded_time_axes"]
    seconds = lambda value: f"{value:.1f}s" if finite(value) else "缺失"
    lines += ["", f"闭合monotonic clock：训练 {seconds(c.get('training_seconds'))}/28800s，非训练 "
              f"{seconds(c.get('nontraining_seconds'))}/1200s，总计 {seconds(c.get('total_seconds'))}/30000s；"
              f"守卫monotonic launch→exit {seconds(outer.get('elapsed_launch_to_exit_seconds'))}/30000s。",
              f"守卫记录UTC起止跨度 {seconds(axes.get('external_UTC_launch_to_exit_seconds'))}/30000s；"
              f"UTC减守卫monotonic {seconds(axes.get('external_UTC_minus_monotonic_seconds'))}。"
              "分phase的UTC耗时未记录，保留原monotonic值，不按比例补算；两轴差异原因未确定。",
              "检查点恢复后，原未闭合guard记录保持原样；整段守卫monotonic及精确计算次数无法恢复。"
              "回滚更新、已记录额外计算与unknown区间单列；只展示实际完整的固定端点，24000缺失时列为空，严格成本资格不强行通过。"
              if report.get("checkpoint_recovery") else "本次未使用检查点中断恢复。",
              "最终24000门槛：S均值≤0.8×old、U均值≤1.05×old、最坏S及最坏热点S≤old；并须完整120、有限、原生完成gate、预算、24GiB峰值及零禁读receipt全部通过。",
              f"未通过项：{', '.join(k for k,v in report['qualification']['checks'].items() if not v) or '无'}。",
              "新4800对旧constant4800是同曝光制度组合比较；新24000同时改变制度与曝光，不能作单独scheduler因果结论。FNO4800仅作原fullTRAIN参考，容量和时间不同。",
              "本轮没有DEV/held5选优、泛化或最终solver胜者结论。缺失、technicalfailed或科学负结果如实保留为false。",
              f"曲线：保存的loss/LR/beta1/preclip gradient共 {report['loss_log']['available_records']}/24000 行；"
              f"PNG/SVG可用 {any(v['available'] for v in report['plot_status'].values())}；缺点/坏行见metrics与manifest。"]
    return "\n".join(lines) + "\n"


def run(export=False):
    if not export:
        return {"status": "source_only_ready", "actual_export": False,
                "training_files_read": 0, "writes": 0, "requires_explicit_export": True}
    index, issues = {}, {}
    terminal, exit_proof = closure(index)
    closure_hashes = {k: v["sha256"] for k, v in index.items()}
    plan = json_record(PLAN, index)
    science = plan.get("scientific", {})
    if not (plan.get("kind") == KIND and plan.get("scientific_rules_frozen") is True
            and plan.get("scientific_digest") == SCIENCE_SHA and digest(science) == SCIENCE_SHA
            and plan.get("output") == RUN and plan.get("reference_TRAIN_fit_summary_sha256") == REFERENCE_SHA):
        raise NotReady("Frozen PP plan/scientific identity differs")
    keys = science.get("TRAIN_keys", [])
    if len(keys) != 120 or len(set(keys)) != 120:
        raise NotReady("Frozen TRAIN120 key list is invalid")
    threshold_plan = optional_json(THRESHOLD_PLAN, index, issues)
    threshold_native = mapping(mapping(threshold_plan.get("scientific")).get("paired_progress_gate"))
    threshold_match = all(threshold_native.get(k) == v for k, v in THRESHOLDS.items())
    summary = optional_json(f"{RUN}/summary.json", index, issues)
    config = optional_json(f"{RUN}/config.json", index, issues)
    receipt = optional_json(f"{RUN}/read_receipt.json", index, issues)
    costs = optional_json(f"{RUN}/costs.json", index, issues)
    gradient = optional_json(f"{RUN}/first_update_gradient_check.json", index, issues)
    reference = optional_json(plan["reference_TRAIN_fit_summary"], index, issues)
    reference_identity = index.get(plan["reference_TRAIN_fit_summary"], {}).get("sha256") == REFERENCE_SHA
    old_training = {}
    for arm in ("transolver_pp_cad40_fresh4800", "fno_cad40_fresh4800"):
        relative = f"runs/geobc_coverage_comparison/{arm}/summary.json"
        saved = optional_json(relative, index, issues)
        old_training[arm] = {k: saved.get(k) for k in ("status", "endpoint_step", "completed4800",
            "stop_reason", "costs", "memory", "first_update_gradient_check", "DEV_target_reads", "scaler_fits")}
        saved_config = mapping(saved.get("config"))
        old_training[arm]["config_metadata"] = {k: saved_config.get(k) for k in
            ("model", "parameter_counts", "precision", "initial_parameters")}
        old_training[arm]["source_metadata"] = mapping(saved_config.get("contract")).get("source_sha256", {})
        expected_sha = mapping(mapping(mapping(reference.get("arms")).get(arm)).get("original_training_summary")).get("sha256")
        old_training[arm]["saved_summary_SHA_matches_reference"] = bool(expected_sha
            and index.get(relative, {}).get("sha256") == expected_sha)
    endpoints, metadata = {}, {}
    for step in (0, 4800, 24000):
        metadata[str(step)] = optional_json(f"{RUN}/step{step}.json", index, issues)
        if step:
            endpoints[str(step)] = optional_json(f"{RUN}/TRAIN_readout/step{step}.json", index, issues)
    blocks = {
        "old_constant4800": mapping(mapping(reference.get("arms")).get("transolver_pp_cad40_fresh4800")),
        "new_bundle4800": endpoints["4800"], "new_bundle24000": endpoints["24000"],
        "FNO_reference4800": mapping(mapping(reference.get("arms")).get("fno_cad40_fresh4800")),
    }
    packed = {}
    for label, block in blocks.items():
        packed[label] = {k: block.get(k) for k in ("step", "checkpoint_step", "method", "status", "aggregate",
            "per_case", "case_failures", "costs", "per_case_costs", "original_training_costs",
            "original_training_elapsed_this_attempt_seconds", "checkpoint", "original_training_summary") if k in block}
        packed[label]["validation"] = full120(block, keys, endpoint=label.startswith("new_"))
    old_gate = packed["old_constant4800"]["validation"]
    comparisons = {str(step): paired(mapping(endpoints[str(step)].get("aggregate")),
        mapping(blocks["old_constant4800"].get("aggregate")), packed[f"new_bundle{step}"]["validation"], old_gate)
        for step in (4800, 24000)}
    contract = mapping(config.get("contract"))
    try:
        contract_sha = digest(contract)
    except (TypeError, ValueError):
        contract_sha = None
        issues[f"{RUN}/config.json.contract"] = "Missing/nonfinite/invalid native contract digest"
    expected_sources = dict(mapping(plan.get("parent_source_sha256")))
    expected_sources.update(mapping(plan.get("additional_source_sha256"))); expected_sources[DRIVER] = DRIVER_SHA
    contract_identity = contract.get("scientific_digest") == SCIENCE_SHA and contract.get("source_sha256") == expected_sources
    contract_identity &= contract.get("TRAIN_keys") == keys and contract.get("fixed_updates") == 24000
    contract_identity &= contract.get("readout_updates") == [4800, 24000] and contract.get("scaler") == science.get("scaler")
    contract_identity &= config.get("TRAIN_only") is True and config.get("optimizer") == science.get("optimizer")
    contract_identity &= config.get("OneCycleLR") == science.get("OneCycleLR")
    endpoint_identity = all(endpoints[str(s)].get("config_contract") == contract
        and endpoints[str(s)].get("config_contract_sha256") == contract_sha and contract_sha is not None
        and endpoints[str(s)].get("step") == s for s in (4800, 24000))
    endpoint_checkpoints = all(metadata[str(s)].get("status") == "completed"
        and metadata[str(s)].get("step") == s and metadata[str(s)].get("scheduler_last_epoch") == s
        and metadata[str(s)].get("config_contract_sha256") == contract_sha and contract_sha is not None
        and metadata[str(s)].get("case_exposure_counts") == {k: s // 120 for k in keys}
        and isinstance(metadata[str(s)].get("checkpoint"), dict)
        and isinstance(metadata[str(s)]["checkpoint"].get("sha256"), str)
        and len(metadata[str(s)]["checkpoint"]["sha256"]) == 64 for s in (0, 4800, 24000))
    rows, loss_gate = saved_loss(index, issues)
    loss_keys = all(r.get("key") in keys for r in rows)
    clock, guard = terminal["clock"], terminal["external_launch"]
    budget = all(finite(clock.get(k)) and 0 <= clock[k] < limit for k, limit in
        (("training_seconds", 28800), ("nontraining_seconds", 1200), ("total_seconds", 30000)))
    clock_accounting = budget and math.isclose(clock["training_seconds"] + clock["nontraining_seconds"],
                                             clock["total_seconds"], rel_tol=1e-10, abs_tol=1e-6)
    outer_budget = guard.get("within_external_budget") is True and guard.get("external_timeout") is False
    outer_budget &= finite(guard.get("elapsed_launch_to_exit_seconds")) and 0 <= guard.get("elapsed_launch_to_exit_seconds", -1) <= 30000
    outer_utc = utc_span(guard.get("started_at_utc"), guard.get("ended_at_utc"))
    native_utc = clock["last_epoch"] - clock["started_epoch"] if (
        finite(clock.get("last_epoch")) and finite(clock.get("started_epoch"))
        and clock["last_epoch"] >= clock["started_epoch"]) else None
    outer_utc_budget = finite(outer_utc) and 0 <= outer_utc <= 30000
    time_axes = {
        "external_UTC_launch_to_exit_seconds": outer_utc,
        "external_monotonic_launch_to_exit_seconds": guard.get("elapsed_launch_to_exit_seconds"),
        "external_UTC_minus_monotonic_seconds": outer_utc - guard["elapsed_launch_to_exit_seconds"]
            if finite(outer_utc) and finite(guard.get("elapsed_launch_to_exit_seconds")) else None,
        "native_latest_attempt_epoch_start_to_last_update_seconds": native_utc,
        "native_monotonic_total_seconds": clock.get("total_seconds"),
        "native_UTC_minus_monotonic_seconds": native_utc - clock["total_seconds"]
            if finite(native_utc) and finite(clock.get("total_seconds")) and clock.get("attempts") == 1 else None,
        "native_axis_comparability": "latest-attempt epoch cannot be subtracted from multi-attempt cumulative monotonic"
            if clock.get("attempts") != 1 else "single attempt",
        "native_training_UTC_seconds": None, "native_nontraining_UTC_seconds": None,
        "phase_UTC_status": "not_recorded; monotonic phase costs retained without rescaling",
        "difference_cause": "undetermined",
        "live_guard_source_modified": False,
        "whole_budget_requires_both_external_axes": True,
    }
    recovery = None
    if "original_external_launch" in terminal:
        preparation = optional_json("notes/pp_checkpoint_resume_preparation.json", index, issues)
        recovery = {"original_external_launch": terminal["original_external_launch"],
            "actual_recovery_guard": guard, "preparation": preparation,
            "native_cumulative_unknown_interruption_charge_seconds": clock.get("unknown_interruption_charge_seconds"),
            "whole_campaign_monotonic_elapsed_seconds": None,
            "strict_whole_campaign_cost_accounting_exact": False,
            "actual_computation_counts_exact": False,
            "fixed_endpoint_all120_validation_rules_unchanged": True,
            "actual_fixed24000_all120_metrics_complete": packed["new_bundle24000"]["validation"]["passed"]}
    memory = mapping(summary.get("memory"))
    memory_gate = summary.get("memory_measurement_available") is True and all(
        type(memory.get(k)) is int and 0 < memory[k] <= 24*1024**3
        for k in ("peak_allocated_bytes", "peak_reserved_bytes"))
    receipt_gate = receipt.get("input_read_receipt_exact") is True and all(
        type(receipt.get(k)) is int and receipt[k] == 0 for k in ZERO_RECEIPTS)
    receipt_gate &= all(receipt.get(k) == 24000 for k in
        ("backward_calls", "optimizer_steps", "scheduler_steps", "gradient_clip_calls"))
    receipt_gate &= receipt.get("TRAIN_metric_cases") == 240 and receipt.get("TRAIN_fullN_predictions") == 240
    native_gate = mapping(summary.get("completion_gate"))
    gradient_norms = mapping(gradient.get("L2_gradient_norms"))
    checks = {
        "frozen_thresholds": threshold_match, "immutable_reference_JSON_SHA": reference_identity,
        "old_PP_training_summary_SHA": old_training["transolver_pp_cad40_fresh4800"]["saved_summary_SHA_matches_reference"],
        "old_constant_full120_finite": old_gate["passed"], "both_new_full120_finite": all(
            packed[f"new_bundle{s}"]["validation"]["passed"] for s in (4800, 24000)),
        "final24000_thresholds": comparisons["24000"]["passed"],
        "native_success_and_gate": summary.get("status") == "completed" and summary.get("terminal") is True
            and summary.get("completed24000") is True and summary.get("endpoint_step") == 24000
            and summary.get("stop_reason") == "fixed24000_completed" and native_gate.get("passed") is True
            and native_gate.get("reached24000") is True and native_gate.get("both_fixed_TRAIN120_readouts_complete") is True
            and native_gate.get("TRAIN_only") is True and native_gate.get("DEV_selection_performed") is False
            and native_gate.get("final_solver_selection") is False,
        "native_terminal_records_agree": terminal["status"].get("status") == summary.get("status")
            and terminal["status"].get("step") == summary.get("endpoint_step")
            and terminal["status"].get("stop_reason") == summary.get("stop_reason"),
        "actual_closed_clock_budget": budget and clock_accounting,
        "actual_outer_guard_budget_and_success": bool(outer_budget and guard.get("exit_code") == 0
            and guard.get("in_flight_costs_exact") is True and guard.get("summary_present") is True),
        "actual_external_UTC_whole_budget": bool(outer_utc_budget),
        "native_cost_authority_flags": costs.get("closed_clock_is_authoritative") is True
            and costs.get("external30000_guard_is_authoritative") is True,
        "native_memory24GiB": memory_gate, "native_TRAIN_only_read_receipt": bool(receipt_gate),
        "native_receipt_matches_summary": receipt == summary.get("read_receipt"),
        "actual_first_gradient": gradient.get("step") == 1 and gradient.get("case") in keys
            and gradient.get("passed") is True and gradient.get("all_present_gradients_finite") is True
            and gradient.get("missing_gradient_names") == [] and gradient == summary.get("first_update_gradient_check")
            and isinstance(gradient.get("allowed_missing"), list) and all(type(v) is str for v in gradient["allowed_missing"])
            and set(gradient["allowed_missing"]).issubset({
                "solver.backbone.embedding.weight", "solver.backbone.embedding.bias"})
            and all(finite(gradient_norms.get(k)) and gradient_norms[k] > 0
                    for k in ("total", "encoder", "solver")),
        "full200passes120": summary.get("case_keys") == keys
            and summary.get("case_exposure_counts") == {k: 200 for k in keys},
        "frozen_native_source_metadata": contract_identity and summary.get("config") == config
            and summary.get("run_process") == terminal["execution"].get("run_process"),
        "fixed_endpoint_identity": endpoint_identity and endpoint_checkpoints,
        "saved_loss24000_finite_and_TRAIN_only": loss_gate["complete24000"] and loss_keys,
        "summary_endpoint_copies_match_native": summary.get("endpoints") == endpoints,
    }
    report = {"schema": 1, "kind": "PP_regime_confirmation_saved_scalar_export", "exported_at_utc": utc(),
        "dataset": "Original DeepJEB", "fixed_case_denominator": 120, "case_keys": keys,
        "units": {"uXYZ": "mm", "signed_stress": "MPa", "relative_metrics": "dimensionless"},
        "thresholds": THRESHOLDS, "qualification": {"passed": all(checks.values()), "checks": checks,
            "scope": "TRAIN progress only; final24000 regime+exposure bundle, no final solver qualification"},
        "paired_comparisons": comparisons, "endpoints": packed,
        "native_summary": {k: summary.get(k) for k in ("status", "terminal", "kind", "endpoint_step", "completed24000",
            "stop_reason", "error_type", "error_message", "completion_gate", "case_exposure_counts")},
        "authoritative_clock": clock, "external_guard": guard, "recorded_time_axes": time_axes,
        "checkpoint_recovery": recovery,
        "exact_process_exit_proof": exit_proof,
        "native_costs": costs, "native_memory": memory, "native_read_receipt": receipt,
        "first_actual_gradient_check": gradient, "loss_log": loss_gate,
        "native_config": config, "checkpoint_metadata_index": metadata,
        "reference_metadata_index": {arm: mapping(mapping(mapping(reference.get("config")).get("contract")).get("endpoint_inventory")).get(arm, {})
            for arm in ("transolver_pp_cad40_fresh4800", "fno_cad40_fresh4800")},
        "reference_training_scalar_metadata": old_training,
        "reference_native_costs": reference.get("costs", {}), "reference_native_read_receipt": reference.get("read_receipt", {}),
        "limitations": ["4800 compares equal-exposure training bundles; not an isolated scheduler effect",
            "24000 changes both regime and exposure", "FNO4800 differs in capacity/time and is reference only",
            "Native and guard durations are monotonic; external UTC span is separately budget-checked; phase UTC costs were not recorded",
            "Stored training loss is not a full120 physical accuracy curve", "No DEV/held5 or generalization/final-family selection"],
        "evidence_issues": issues}
    report = clean(report, "metrics", issues)
    # Recheck the hard barrier and its original bytes immediately before any write.
    final_index = {}; _, final_exit = closure(final_index)
    if {k: v["sha256"] for k, v in final_index.items()} != closure_hashes:
        raise NotReady("Terminal control evidence changed during scalar export preparation")
    report["exact_process_exit_proof_before_write"] = final_exit
    destination = local(DESTINATION)
    destination.mkdir(parents=True, exist_ok=False)
    plots, plot_status = scalar_plots(rows, issues, destination)
    artifact_complete = set(plots) == {"optimizer_traces.png", "optimizer_traces.svg"}
    report["plot_status"] = plot_status
    report["evidence_issues"] = dict(issues)
    def save(name, content):
        with (destination / name).open("x", encoding="utf-8", newline="") as stream:
            stream.write(content)
    save("metrics.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    save("BRIEF.md", brief(report))
    for name, raw in plots.items():
        with (destination / name).open("xb") as stream:
            stream.write(raw)
    buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=LOSS_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: v if not (type(v) is float and not math.isfinite(v)) else "" for k,v in row.items()})
    save("optimizer_scalars.csv", buffer.getvalue())
    artifacts = {}
    for name in ("metrics.json", "BRIEF.md", "optimizer_scalars.csv", *plots):
        raw = (destination / name).read_bytes()
        artifacts[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    manifest = {"schema": 1, "kind": "PP_regime_confirmation_export_manifest", "created_at_utc": utc(),
        "status": "completed" if artifact_complete else "completed_incomplete",
        "artifact_complete": artifact_complete,
        "source_SHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source": "tools/export_pp_regime_confirmation.py",
        "scientific_digest": SCIENCE_SHA, "read_files": index, "artifacts": artifacts,
        "native_evidence_modified": False, "PT_or_weight_content_reads": 0, "scientific_array_reads": 0,
        "raw_or_GT_reads": 0, "model_or_contract_executions": 0, "inference_reruns": 0, "GPU_operations": 0,
        "checkpoint_SHA_policy": "JSON files are hashed; recorded PT hashes/stat metadata are copied without opening/statting PT",
        "closed_clock_authoritative": True, "summary_preclose_clock_used": False,
        "qualification_passed": report["qualification"]["passed"], "plot_status": plot_status,
        "missing_or_invalid_evidence": issues}
    save("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return {"status": "exported" if artifact_complete else "completed_incomplete",
            "destination": str(destination), "qualification_passed": report["qualification"]["passed"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true", help="Explicitly export only after genuine terminal/clock/process/guard closure")
    arguments = parser.parse_args()
    try:
        result = run(arguments.export)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        if result.get("status") == "completed_incomplete":
            raise SystemExit(1)
    except (NotReady, OSError, ValueError) as error:
        raise SystemExit(f"No export: {type(error).__name__}: {error}") from error


if __name__ == "__main__":
    main()
