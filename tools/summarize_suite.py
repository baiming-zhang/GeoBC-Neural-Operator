"""Read-only progress/results report for the valid original-DeepJEB suite.

Only outputs explicitly listed in all_methods_plan.json are included. This
script imports neither torch nor a model, and works while the GPU coordinator
is running. Missing measurements remain pending rather than becoming zeros.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path, warnings: list[str]):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        warnings.append(f"{path}: {exc}")
        return None


def number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def mean(values):
    values = [number(x) for x in values]
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else None


def resolve_source(path, root: Path) -> Path:
    """Accept paths saved by either Windows or WSL without executing a shell."""
    value = str(path).replace("\\", "/")
    marker = "/mnt/c/"
    if value.startswith(marker) and sys.platform == "win32":
        value = "C:/" + value[len(marker):]
    elif len(value) > 2 and value[1:3] == ":/" and sys.platform != "win32":
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    result = Path(value)
    return result if result.is_absolute() else root / result


def command_option(command, option, default=None, many=False):
    if option not in command:
        return [] if many and default is None else default
    start = command.index(option) + 1
    if not many:
        return command[start] if start < len(command) else default
    values = []
    for value in command[start:]:
        if str(value).startswith("--"):
            break
        values.append(value)
    return values


def normalize_field(value):
    if not isinstance(value, dict):
        return {}
    # Benchmark/FNO store named fields; pooled hash and distillation are stress-only.
    if "stress" in value or "displacement" in value:
        return {name: value[name] for name in ("stress", "displacement")
                if isinstance(value.get(name), dict)}
    if "relative_l2" in value:
        return {"stress": value}
    return {}


def normalize_metrics(metrics, config):
    if not isinstance(metrics, dict):
        return {"per_case": {}, "pooled": {}, "grid": {}, "teacher": {}}
    per_case = {key: normalize_field(value) for key, value in metrics.get("per_case", {}).items()}
    key = metrics.get("sample_key") or config.get("case")
    if "nodes" in metrics and isinstance(metrics["nodes"], dict):
        pooled = normalize_field(metrics["nodes"])
        if key:
            per_case[str(key)] = pooled
    else:
        pooled = normalize_field(metrics.get("pooled"))
    result = {"step": metrics.get("step"), "epoch": metrics.get("epoch"),
              "per_case": per_case, "pooled": pooled,
              "grid": normalize_field(metrics.get("grid")),
              "teacher": normalize_field(metrics.get("teacher")),
              "selection_score": metrics.get("selection_score"),
              "elapsed_seconds": metrics.get("elapsed_seconds")}
    for field in ("stress", "displacement"):
        pairs = [(key, value[field]) for key, value in per_case.items() if field in value]
        pairs = [(key, value) for key, value in pairs if number(value.get("relative_l2")) is not None]
        if pairs:
            worst_key, worst = max(pairs, key=lambda p: p[1]["relative_l2"])
            best_key, best = min(pairs, key=lambda p: p[1]["relative_l2"])
            result[field + "_case_mean_l2"] = mean([value["relative_l2"] for _, value in pairs])
            result[field + "_worst_l2"] = worst["relative_l2"]
            result[field + "_worst_key"] = worst_key
            result[field + "_best_l2"] = best["relative_l2"]
            result[field + "_best_key"] = best_key
            if field == "stress":
                result["stress_cases_below_1pct"] = sum(value["relative_l2"] < .01 for _, value in pairs)
                result["stress_case_count"] = len(pairs)
                result["stress_case_mean_hotspot_l2"] = mean([value.get("hotspot_relative_l2") for _, value in pairs])
                by_load = {}
                for case_key, value in pairs:
                    by_load.setdefault(case_key.rsplit("/", 1)[-1], []).append(value["relative_l2"])
                result["stress_load_groups"] = {
                    load: {"cases": len(values), "case_mean_relative_l2": mean(values)}
                    for load, values in sorted(by_load.items())}
    return result


def read_history(folder: Path, warnings):
    path = folder / "metrics.jsonl"
    if not path.exists():
        return []
    rows = {}
    # A concurrent writer can leave the final line incomplete for this snapshot.
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        warnings.append(f"{path}: {exc}")
        return []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
            if isinstance(row, dict) and number(row.get("step")) is not None:
                rows[int(row["step"])] = row
        except ValueError:
            if index != len(lines) - 1:
                warnings.append(f"{path}: malformed line {index + 1}")
    return [rows[step] for step in sorted(rows)]


def read_run(job, root, coordinator, warnings):
    folder = resolve_source(job["output"], root)
    summary = read_json(folder / "summary.json", warnings) or {}
    config = read_json(folder / "config.json", warnings) or summary.get("config") or {}
    status = read_json(folder / "status.json", warnings) or {}
    best = summary.get("best_metrics") or read_json(folder / "metrics_best.json", warnings)
    latest = summary.get("final_metrics") or read_json(folder / "metrics_latest.json", warnings)
    if summary.get("status") == "completed":
        phase = "completed"
    elif status.get("phase") == "failed":
        phase = "failed"
    elif status:
        # Completion requires the durable run summary, including externally
        # appended jobs that are not part of the coordinator's startup list.
        phase = "running"
    elif coordinator.get("current") == job["name"]:
        phase = "waiting" if coordinator.get("phase") == "waiting_for_inputs" else "running"
    else:
        phase = "pending"
    args = config.get("arguments", {})
    case_keys = [c["sample_key"] for c in config.get("cases", [])]
    if not case_keys and config.get("case"):
        case_keys = [config["case"]]
    if not case_keys:
        case_keys = command_option(job["command"], "--case-keys", many=True)
    if not case_keys and "geobc.grid_train" in job["command"]:
        case_keys = ["506_533/ver"]  # The plan explicitly uses only this FNO cache.
    if not case_keys and "geobc.distill" in job["command"]:
        case_keys = ["506_533/ver"]  # Explicit --nodes argument in the current plan.
    case_count = len(case_keys) or (50 if "geobc.global_field" in job["command"] else None)
    normalized = normalize_metrics(best, config)
    # Actual metrics also identify a case when a partial config lacks it.
    if not case_keys and normalized["per_case"]:
        case_keys = list(normalized["per_case"])
        case_count = len(case_keys)
    total_steps = config.get("total_steps") or status.get("total_steps")
    if total_steps is None:
        raw = command_option(job["command"], "--steps")
        if raw is not None:
            total_steps = int(raw)
        elif "geobc.global_field" not in job["command"]:
            epochs = command_option(job["command"], "--epochs")
            total_steps = int(epochs) * case_count if epochs and case_count else None
    return {"name": job["name"], "output": job["output"], "slides": job.get("slides", []),
            "planned_command": job["command"],
            "phase": phase, "status_detail": status.get("phase"),
            "step": summary.get("step", status.get("step")), "total_steps": total_steps,
            "best_step": summary.get("best_step", normalized.get("step")),
            "best_is_provisional": phase != "completed" and best is not None,
            "elapsed_seconds": summary.get("elapsed_seconds", status.get("elapsed_seconds")),
            "eta_seconds": status.get("eta_seconds") if phase == "running" else None,
            "estimated_completion_utc": status.get("estimated_completion_utc") if phase == "running" else None,
            "case_keys": case_keys, "case_count": case_count,
            "total_nodes": config.get("total_nodes") or sum(c.get("nodes", 0) for c in config.get("cases", [])) or config.get("FEM_nodes"),
            "method": config.get("method") or command_option(job["command"], "--method") or job["name"].split("_")[0],
            "outputs": args.get("outputs") or command_option(job["command"], "--outputs", "stress"),
            "config": config, "assumptions": list(dict.fromkeys(job.get("assumptions", []) + config.get("assumptions", []))),
            "best": normalized, "latest": normalize_metrics(latest, config),
            "truth_roundtrip": read_json(folder / "truth_roundtrip.json", warnings),
            "history": [normalize_metrics(row, config) for row in read_history(folder, warnings)],
            "measurement_source": str(folder / ("summary.json" if summary.get("best_metrics") else "metrics_best.json")) if best else None,
            "error": status.get("error")}


def percent(value):
    value = number(value)
    return f"{100 * value:.5f}" if value is not None else "pending"


def metric_cell(value, applicable=True):
    return percent(value) if applicable else "—"


def field_l2(record, field="stress", domain="pooled"):
    return number(record.get(domain, {}).get(field, {}).get("relative_l2"))


def duration(value):
    value = number(value)
    if value is None:
        return "pending"
    if value >= 3600:
        return f"{value / 3600:.2f} h"
    if value >= 60:
        return f"{value / 60:.2f} min"
    return f"{value:.2f} s"


def escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def host_report_path(value):
    """Format local links for the Windows desktop, preserving JSON runtime paths."""
    path = str(value).replace("\\", "/")
    mounted = re.match(r"^/mnt/([A-Za-z])/(.*)$", path)
    if mounted:
        return mounted[1].upper() + ":/" + mounted[2]
    if re.match(r"^[A-Za-z]:/", path):
        return path[0].upper() + path[1:]
    return path


def configuration_details(run, sources):
    config = run["config"]
    model = config.get("model", {})
    method = run["method"]
    source_key = {"transolverpp": "transolver_pp", "transolver": "transolver", "ts3": "ts3"}.get(method)
    source = sources.get(source_key, {})
    args = config.get("arguments", {})
    backend = args.get("backend") or command_option(run["planned_command"], "--backend")
    details = config.get("implementation_details", {})
    network = details.get("network", details.get("networkConfig", config.get("networkConfig", {})))
    fused = backend == "tcnn-fused" or network.get("otype") == "FullyFusedMLP"
    known = model.get("constructor_kwargs", model.get("preset", model.get("model_kwargs", {})))
    if fused:
        encoding = details.get("encoding", details.get("encodingConfig", {}))
        grid = encoding.get("nested", [encoding])[0]
        levels = grid.get("n_levels", model.get("n_levels", 12))
        features = grid.get("n_features_per_level", model.get("n_features_per_level", 2))
        cap = grid.get("log2_hashmap_size", model.get("log2_hashmap_size", 19))
        base = grid.get("base_resolution", model.get("base_resolution", 8))
        growth = grid.get("per_level_scale", model.get("per_level_scale", 1.6))
        width = network.get("n_neurons", model.get("hidden_dim", 64))
        depth = network.get("n_hidden_layers", model.get("hidden_layers", 2))
        activation = network.get("activation", "ReLU")
        mode = args.get("input_mode") or model.get("input_mode") or command_option(run["planned_command"], "--input-mode", "both" if run["case_count"] != 1 else "xyz")
        actual = f"{levels}×{features} hash, cap log2={cap}, base{base}, growth{growth}; official FullyFusedMLP {depth}×{width} {activation}; input={mode}; native FP16 / FP32 master"
        if not details:
            actual = "计划：" + actual + "（实际 implementation_details 待 run/config.json）"
        reference = "slides 14–17: stated hash dimensions and 2×64 fused MLP; no sample ID"
        source = {"repository": details.get("source", "https://github.com/NVlabs/tiny-cuda-nn").split("/tree/")[0],
                  "commit": details.get("source_commit", "3daa6e5b8b8e0fa17971a1a20acfbb3754494b38"),
                  "reference_gaps": ["Historical learning rate, asinh constants and initialization seed are unavailable."]}
        if mode == "both":
            source["reference_gaps"].append("Slides specify 25 BC/material channels without definitions; this run uses an explicit real-FEM-deck-derived replacement schema.")
    elif source_key:
        settings = model.get("model_settings", model.get("preset_parameters", {})) or known
        actual = source.get("preset_source", "pending")
        if settings:
            actual = json.dumps(settings, ensure_ascii=False, sort_keys=True)
        reference = json.dumps(source.get("slide_explicit", {}), ensure_ascii=False) or "{}"
    elif method == "mlp":
        actual, reference = "xyz→4×256 ReLU→uXYZ/stress", "slide 8: 4×256 ReLU, xyz, joint output, 500 epochs"
    elif method == "ngp4":
        actual, reference = "4 levels; cap log2=14, base=8, growth=2, MLP 2×32 (recorded substitute)", "slide 9: 4 levels, 76928 parameters, 1000 updates"
    elif method == "FNO3d" or run["name"].startswith("fno") and "distill" not in run["name"]:
        size = config.get("grid_size") or (192 if "192" in run["name"] else 64)
        modes = config.get("model", {}).get("modes") or (16 if size == 192 else 12)
        actual, reference = f"{size}³, 4 blocks, width16, modes{modes}, 18 channels; projection128, padding0", "slides 10–13: grid size / width / modes / blocks / 18 inputs"
    else:
        actual, reference = "12×2 hash, cap log2=19, base8, growth1.6, MLP 2×64; torch MLP", "slides 14–17: hash dimensions; original fused MLP"
    selection = config.get("checkpoint_selection") or config.get("selection")
    if not selection:
        selection = "pending (not run yet)"
    return {"run": run["name"], "slide_explicit": reference, "implementation": actual,
            "selection": selection, "source": source.get("repository"), "commit": source.get("commit"),
            "reference_gaps": source.get("reference_gaps", []), "assumptions": run["assumptions"],
            "precision": config.get("precision") or details.get("precision") or "pending (not run yet)",
            "backend": "tcnn-fused" if fused else backend, "implementation_details": details,
            "network_config": network if fused else None}


GAP_TRANSLATIONS = {
    "MLP_learning_rate_absent_using_0.003": "MLP 历史学习率未知；本次 0.003。",
    "asinh_constants_and_LR_absent_using_P90_scaled_centered_asinh_and_0.003": "历史 asinh 常数/学习率未知；本次 pooled |stress| P90、中心化标准化、LR 0.003。",
    "torch_MLP_instead_of_tiny_cuda_nn_fused_MLP": "MLP 使用 torch 实现；历史 fused MLP 后端未复刻。",
    "single_case_extension_of_two_case_reference": "单样本为两样本实验的补充。",
    "five_case_extension_per_case_500_complete_passes": "五样本为补充实验，每个案例完整遍历 500 次。",
    "frozen_NGP_added_only_at_output_no_coarse_field_in_input": "冻结 NGP 只在输出相加，未作为 correction 输入。",
    "NGP_small_grid_details_unavailable": "小 NGP 完整配置缺失，采用已记录的替代值。",
    "reference_user_stopped_at_9905_reproducing_same_duration_without_assuming_convergence": "复刻 9905 次停止预算，不称自动收敛。",
    "teacher_best_epoch_will_follow_this_run_not_force_original_epoch8200": "teacher 选本次最佳网格 checkpoint，不强制历史 epoch 8200。",
    "original270_train_pool_missing_using_official_first50_CADs_13ver13hor12dia12tor": "原 270 训练池缺失，改用官方前 50 CAD；13 ver/13 hor/12 dia/12 tor。",
    "original270_train_pool_missing_using_explicit_replacement50": "50 CAD 为明确替代集合，与历史随机 50 CAD 不同。",
    "steps_per_epoch_recomputed_from_actual_selected_node_count": "真 epoch 更新数按实际节点总数重算。",
    "official_tiny_cuda_nn_FullyFusedMLP_reimplementation_of_stated_architecture": "官方 TCNN FullyFusedMLP 实现原稿明确的 hash/MLP 架构；native FP16 computation / FP32 master。",
    "historical_asinh_constants_LR_and_initialization_seed_unavailable_using_explicit_recorded_values": "历史 LR/asinh 常数/初始化 seed 未给；采用明确保存的本次值。",
    "asinh_constants_and_LR_not_published_using_recorded_P90_and_0.003": "历史 LR/asinh 常数未给；本次采用已保存的 P90/中心化标准化与 LR0.003。",
}


def translate_gap(value):
    if value in GAP_TRANSLATIONS:
        return GAP_TRANSLATIONS[value]
    if "TS3" in value and ("preset" in value or "dimensions" in value):
        return "TS3 历史骨干/优化配置缺失；使用官方 volume preset 16×256、64 slices，已记录替代。"
    if "asinh" in value:
        return "历史 asinh 常数缺失；使用本次记录的 pooled P90/均值/标准差。"
    if "4_levels" in value:
        return "小 NGP 仅 4 levels 可确认；hash cap/base/growth/MLP width 为记录的替代值。"
    if "18_grid" in value:
        return "原 18 网格输入定义缺失；使用 occupancy+16 基本 BC/material+尺度。"
    if "projection" in value:
        return "FNO projection/padding/学习率的历史值缺失；本次 projection128/padding0/LR0.003。"
    if "sampling_count" in value or "distillation_LR" in value:
        return "历史采样量/学习率/缩放细节缺失；使用并保存本次明确设置。"
    if "25_feature" in value:
        return "slides 仅提 25 维，未定义通道；本次从真实 FEM deck 构造明确的替代 BC/material schema。"
    if "additional_user_requested" in value:
        return "原始 Transolver 为新增对照；采用官方 constructor defaults。"
    if "unpublished_reference" in value:
        return "未公布的配置使用官方 preset/default，不能称历史配置完全恢复。"
    return value.replace("_", " ")


def atomic_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def reference_comparisons(root: Path, runs, warnings):
    """Transcribe reference claims, preserving chart rounding and note provenance.

    Never reconstruct the unavailable source summary files. Exact note literals
    can supersede rounded labels, but remain explicitly marked speaker notes.
    Unlabeled curve estimates are not treated as comparison measurements.
    """
    transcribed = read_json(root / "notes/slide_reference_metrics.json", warnings) or {}
    slides = read_json(root / "references/slides_text.json", warnings) or []
    by_slide = {slide["slide"]: slide for slide in slides}
    notes = {n: "\n".join(slide.get("notes", [])) for n, slide in by_slide.items()}
    records = {}

    def put(run, case, field, domain, value_percent, digits, slide, kind, budget,
            literal=None, display=None, note=""):
        value = number(value_percent)
        if value is None:
            return
        previous = records.get((run, case, field, domain), {})
        records[(run, case, field, domain)] = {
            "run": run, "case": case, "field": field, "domain": domain,
            "reference_percent": value, "reference_decimal_places": digits,
            "reference_display": display or f"{value:.{digits}f}",
            "reference_literal": literal, "slide": slide, "source_kind": kind,
            "source": "references/slides_text.json" if kind.startswith("speaker_notes") or kind == "printed_slide_label" else "notes/slide_reference_metrics.json",
            "source_artifact_available": False, "reference_budget": budget, "note": note,
        }
        rounded = previous.get("rounded_chart_label")
        if previous and previous.get("source_kind", "").startswith("printed") and kind.startswith("speaker_notes"):
            rounded = {"display": previous["reference_display"], "source_kind": previous["source_kind"], "source": previous["source"]}
        if rounded:
            records[(run, case, field, domain)]["rounded_chart_label"] = rounded

    mapping = {
        "TS3 short diagnostic": "ts3_two_short", "TS3 long": "ts3_two_long",
        "Transolver++ direct linear": "transolverpp_two_linear",
        "Transolver++ direct asinh": "transolverpp_two_asinh",
        "NGP + TS3 correction": "ts3_residual_two",
        "NGP + Transolver++ correction": "transolverpp_residual_two",
        "Shared large hash grid before continuation": "hash50_sampled",
        "Shared large hash grid after continuation": "hash50_full1000ep",
    }
    for item in transcribed.get("metrics", []):
        method, case = item.get("method"), item.get("case")
        run = mapping.get(method)
        if method == "Small NGP reference":
            # Keep this rounded frozen-reference chart claim separately from slide 9.
            run = "frozen_ngp_reference"
        if not run:
            continue
        field = "stress_case_mean" if run.startswith("hash50") else "stress"
        if run.startswith("hash50"):
            case = "50 training CADs / aggregate"
        put(run, case, field, "pooled", item.get("value_percent"),
            int(item.get("read_precision_percent_decimal_places", 1)), item["slide"],
            item.get("source_kind", "printed_chart_label"), item.get("budget", {}),
            note=item.get("comparison_precision", ""))
    for item in transcribed.get("note_metrics", []):
        run = mapping.get(item.get("method"))
        if item.get("method") == "NGP displacement-only reference":
            run = "ngp4_" + item["case"].replace("/", "_") + "_u"
        if not run:
            continue
        value = item.get("value_percent")
        literal = str(value)
        field = "displacement" if "displacement" in item.get("metric", "") else "stress"
        digits = len(literal.split(".", 1)[1]) if "." in literal else 0
        put(run, item["case"], field, "pooled", value, digits, item["slide"],
            "speaker_notes_claim", item.get("budget", {}), literal=literal)

    # These percent literals occur verbatim in the notes, including trailing zeros.
    match = re.search(r"best9900 previously reported u([0-9.]+)%/([0-9.]+)%,stress([0-9.]+)%/([0-9.]+)%", notes.get(4, ""))
    if match:
        for case, u, stress in (("506_533/ver", match[1], match[3]), ("506_25/tor", match[2], match[4])):
            for field, literal in (("displacement", u), ("stress", stress)):
                put("ts3_two_long", case, field, "pooled", float(literal), len(literal.split(".")[-1]), 4,
                    "speaker_notes_literal_percent", {"epochs": 5000, "updates": 10000, "best_update": 9900}, literal=literal)
    match = re.search(r"u-only first results([0-9.]+)%/([0-9.]+)%", notes.get(6, ""))
    if match:
        for case, literal in (("506_533/ver", match[1]), ("506_25/tor", match[2])):
            put("ngp4_" + case.replace("/", "_") + "_u", case, "displacement", "pooled", float(literal),
                len(literal.split(".")[-1]), 6, "speaker_notes_literal_percent", {"updates": None}, literal=literal)

    # Some notes contain a precise reported L2 fraction in an embedded JSON block.
    # Display four percentage decimals, recording the original literal separately.
    note_runs = {8: "mlp_single_exact", 9: "ngp4_506_533_ver_stress", 10: "fno64_joint",
                 11: "fno64_stress", 12: "fno192_stress", 14: "fno192_hash_distill",
                 15: "hash_single_exact", 16: "hash_two_exact"}
    budgets = {8: {"epochs": 500}, 9: {"updates": 1000}, 10: {"epochs": 500},
               11: {"epochs": 500, "best_epoch": 300}, 12: {"stopped_epoch": 9905, "best_epoch": 8200},
               14: {"updates": 1000}, 15: {"updates": 1000, "best_update": 600},
               16: {"updates": 1000, "nodes_per_case_per_update": 16384}}
    for slide, run in note_runs.items():
        match = re.search(r'"stress_L2"\s*:\s*([0-9.eE+-]+)', notes.get(slide, ""))
        case = re.search(r'"case"\s*:\s*"([^"]+)"', notes.get(slide, ""))
        load = re.search(r'"load"\s*:\s*"([^"]+)"', notes.get(slide, ""))
        if match and case and load:
            literal = match[1]
            put(run, case[1] + "/" + load[1], "stress", "pooled", float(literal) * 100, 4, slide,
                "speaker_notes_literal_relative_l2", budgets[slide], literal=literal,
                note="备注中的 relative L2 fraction 转为百分比后显示四位小数；原 summary.json 未恢复。")
    # Values mentioned in prose rather than the embedded JSON field.
    literals = [
        (9, r"Torsion stress([0-9.]+)%", "ngp4_506_25_tor_stress", "506_25/tor", "stress", "pooled", {"updates": 1000}),
        (10, r" u([0-9.]+)%", "fno64_joint", "506_533/ver", "displacement", "pooled", {"epochs": 500}),
        (12, r"Best grid physicalL2([0-9.]+)%", "fno192_stress", "506_533/ver", "stress", "grid", budgets[12]),
    ]
    for slide, pattern, run, case, field, domain, budget in literals:
        match = re.search(pattern, notes.get(slide, ""))
        if match:
            literal = match[1]
            put(run, case, field, domain, float(literal), len(literal.split(".")[-1]), slide,
                "speaker_notes_literal_percent", budget, literal=literal)
    teacher_text = "\n".join(by_slide.get(14, {}).get("text", []))
    match = re.search(r"([0-9.]+)% error at teacher grid points", teacher_text)
    if match:
        literal = match[1]
        put("fno192_hash_distill", "506_533/ver", "stress", "teacher", float(literal), len(literal.split(".")[-1]), 14,
            "printed_slide_label", budgets[14], literal=literal)
    # Slide 16's vertical-case value is rounded in prose and has no precise note JSON.
    text16 = "\n".join(by_slide.get(16, {}).get("text", []))
    match = re.search(r"([0-9.]+)% / ([0-9.]+)%", text16)
    if match:
        literal = match[1]
        put("hash_two_exact", "506_533/ver", "stress", "pooled", float(literal), len(literal.split(".")[-1]), 16,
            "printed_slide_label", budgets[16], literal=literal)
    # Aggregate metrics for a different historical 50-CAD cohort, never case pairs.
    aggregate_patterns = [
        (r"PooledL2 across allnodes([0-9.]+)%", "stress_pooled"),
        (r"caseaverage([0-9.]+)%", "stress_case_mean"),
        (r"Best/worstcase([0-9.]+)%/([0-9.]+)%", "stress_best_and_worst"),
    ]
    for pattern, field in aggregate_patterns:
        match = re.search(pattern, notes.get(18, ""))
        if match:
            pairs = [(field, match[1])] if field != "stress_best_and_worst" else [("stress_best", match[1]), ("stress_worst", match[2])]
            for metric, literal in pairs:
                put("hash50_full1000ep", "50 training CADs / aggregate", metric, "pooled", float(literal),
                    len(literal.split(".")[-1]), 18, "speaker_notes_literal_percent",
                    {"epochs": 1000, "best_epoch": 990, "best_update": 215820, "updates": 218000}, literal=literal,
                    note="历史与本次 50-CAD 集合不同，数值仅并列记录，不能视为同数据重跑。")

    # Native TCNN alternatives use the same slide reference numbers; the five-
    # case extension compares only the two named cases, with its cohort gap explicit.
    for native, earlier in (("tcnn_hash_single_exact", "hash_single_exact"),
                            ("tcnn_hash_two_exact", "hash_two_exact"),
                            ("tcnn_hash_five_aligned", "hash_two_exact")):
        for record in list(records.values()):
            if record["run"] == earlier:
                copied = {**record, "run": native, "implementation_variant": "official TCNN FullyFusedMLP, native FP16 computation / FP32 master"}
                if native.endswith("five_aligned"):
                    copied["cohort_note"] = "Slide 16 trains two cases; this extension trains five. Only the two identically named cases are shown side by side, not a same-cohort repeat."
                records[(native, record["case"], record["field"], record["domain"])] = copied

    run_map = {run["name"]: run for run in runs}
    result = []
    for record in records.values():
        run = run_map.get(record["run"])
        if record["run"] == "frozen_ngp_reference":
            run = run_map.get("ngp4_" + record["case"].replace("/", "_") + "_stress")
        actual = None
        if run:
            field, domain = record["field"], record["domain"]
            if field == "stress_case_mean":
                actual = number(run["best"].get("stress_case_mean_l2"))
            elif field in ("stress_best", "stress_worst"):
                actual = number(run["best"].get(field + "_l2"))
            elif field == "stress_pooled":
                actual = field_l2(run["best"])
            elif domain == "pooled":
                actual = number(run["best"].get("per_case", {}).get(record["case"], {}).get(field, {}).get("relative_l2"))
            else:
                actual = field_l2(run["best"], field, domain)
        result.append({**record, "current_run": run["name"] if run else None,
                       "current_relative_l2": actual, "current_best_step": run["best_step"] if run else None,
                       "current_phase": run["phase"] if run else "pending"})
    return result


def planned_budget(run):
    args, command = run["config"].get("arguments", {}), run["planned_command"]
    epochs = args.get("epochs") or command_option(command, "--epochs")
    steps = run["total_steps"]
    if epochs:
        return f"{epochs} epochs / {steps if steps is not None else 'pending'} updates"
    return f"{steps if steps is not None else 'pending'} updates"


def format_reference_budget(value):
    labels = {"epochs": "epochs", "true_epochs": "true epochs", "updates": "updates",
              "sampled_updates": "sampled updates", "evaluation_update": "eval update",
              "stopped_update_from_notes": "stop update", "stopped_epoch": "stop epoch",
              "best_epoch": "best epoch", "best_update": "best update",
              "best_epoch_from_notes": "best epoch", "best_update_from_notes": "best update",
              "learning_rate": "LR"}
    result = [f"{labels[key]}={v}" for key, v in value.items() if key in labels and v is not None]
    return ", ".join(result) or "unknown"


def comparison_report(payload):
    main_cases = ["506_533/ver", "506_25/tor"]
    refs = {(r["run"], r["case"], r["field"], r["domain"]): r for r in payload["slide_comparisons"]}
    lines = ["## 与原稿数字逐案例对照", "",
             "原稿列来自实际图中文字或 speaker notes，不是已恢复的原 run/summary.json。图中一位小数保持一位；备注给出更精确数字时明确标为“备注”，fraction 转换的显示精度也已记录在 JSON。原稿短 TS3 是 update 800 的评估，本次列为所运行预算中的最佳 checkpoint，二者选择时点不同。同一 run 的各案例/指标都来自同一个选定 checkpoint，并非逐案例另挑最低值。当前进行中的值仍是暂定最佳。", "",
             "| 实验 / 指标 | 506_533/ver 原稿 % | 本次 best % | 506_25/tor 原稿 % | 本次 best % | 原稿预算 → 本次预算 | 主要差异 / 缺口 |",
             "|---|---:|---:|---:|---:|---|---|"]
    field_names = {"stress": "stress", "displacement": "u vector"}
    for run in payload["runs"]:
        if run["case_count"] == 50:
            continue
        fields = ["stress"] if run["outputs"] == "stress" else ["displacement"] if run["outputs"] == "u" else ["stress", "displacement"]
        domains = ["pooled"]
        if run["name"].startswith("fno"):
            domains += ["teacher"] if "distill" in run["name"] else ["grid"]
        run_references = [r for r in payload["slide_comparisons"] if r["run"] == run["name"]]
        for domain in domains:
            for field in fields:
                cells, budgets = [], []
                has_reference_numbers = False
                for case in main_cases:
                    ref = refs.get((run["name"], case, field, domain))
                    tag = "备注" if ref and ref["source_kind"].startswith("speaker_notes") else "图"
                    original = ref["reference_display"] + f"（{tag}）" if ref else ("未提供" if case in run["case_keys"] and run_references else "—")
                    if ref:
                        has_reference_numbers = True
                        budgets.append(format_reference_budget(ref["reference_budget"]))
                    if domain == "pooled":
                        actual = run["best"].get("per_case", {}).get(case, {}).get(field, {}).get("relative_l2")
                    else:
                        actual = field_l2(run["best"], field, domain) if case in run["case_keys"] else None
                    cells += [original, metric_cell(actual, case in run["case_keys"])]
                if not budgets and run_references:
                    budgets = [format_reference_budget(run_references[0]["reference_budget"])]
                reference_budget = " / ".join(dict.fromkeys(budgets)) or "本 slides 未给对应配置/数值"
                gap = " ".join(dict.fromkeys(translate_gap(x) for x in run["assumptions"]))
                backend = run["config"].get("arguments", {}).get("backend") or command_option(run["planned_command"], "--backend")
                if backend == "tcnn-fused":
                    gap += " 官方 FullyFused native FP16 / FP32 master；与早期 Triton encoding + torch MLP 分列。"
                    input_mode = run["config"].get("arguments", {}).get("input_mode") or command_option(run["planned_command"], "--input-mode")
                    if input_mode == "both":
                        gap += " 25 通道为真实 FEM deck 替代 schema，历史定义缺失。"
                    if run["case_count"] == 5 and has_reference_numbers:
                        gap += " 原稿共享2案例、本次共享5案例；同名两case仅并列，训练集合及每update总采样量不同。"
                if run["name"].startswith("ngp4"):
                    count = run["config"].get("parameter_count", "pending")
                    capacity = "原稿 stress-only 76928 parameters" if field == "stress" else "原稿 u-only 参数总数未知"
                    gap += f" {capacity}；本次 {count} parameters。完整设置不可恢复，不能声称等容量。"
                if not has_reference_numbers and run_references:
                    gap = "原稿未给该指标；未补造数字。 " + gap
                elif not run_references:
                    gap = "新增/补充对照，无同预算原稿数字。 " + gap
                metric = field_names[field] + (" FEM nodes" if domain == "pooled" else " " + domain + " points")
                lines.append("| " + " | ".join(escape(value) for value in [run["name"] + " / " + metric, *cells,
                    reference_budget + " → " + planned_budget(run), gap]) + " |")
    # Show the rounded frozen-reference chart alongside the current independently fit NGPs.
    cells = []
    for case in main_cases:
        ref = refs.get(("frozen_ngp_reference", case, "stress", "pooled"))
        cells += [ref["reference_display"] + "（图）" if ref else "pending", percent(ref.get("current_relative_l2") if ref else None)]
    lines.append("| 冻结 NGP reference（slide 6） / stress | " + " | ".join(cells) + " | 1000 NGP updates → 本次每案例 1000 updates | 原图 8.5/12.9 仅一位小数；不代表与 residual 的底层误差完全相同。NGP 完整配置/容量缺失。 |")
    lines += ["", "本次 25 通道并非演示稿定义：slides 只注明 25 维，本次从真实 FEM deck 构造了明确替代 schema。小 NGP 完整容量未知；大 hash 的历史 LR/asinh 常数及 fused-MLP 后端细节也未知。因此保留差异，不为对齐数字调整配置或指标。", "",
              "| 50 CAD 统计量 | 原稿 % | 本次 best % | 原稿预算 → 本次预算 | 集合差异 |",
              "|---|---:|---:|---|---|"]
    for ref in payload["slide_comparisons"]:
        if ref["run"].startswith("hash50"):
            run = next((r for r in payload["runs"] if r["name"] == ref["run"]), None)
            tag = "备注" if ref["source_kind"].startswith("speaker_notes") else "图"
            lines.append("| " + " | ".join(escape(value) for value in [ref["run"] + " / " + ref["field"],
                ref["reference_display"] + f"（{tag}）", percent(ref["current_relative_l2"]),
                format_reference_budget(ref["reference_budget"]) + " → " + (planned_budget(run) if run else "pending"),
                "原270池随机50 vs 官方前50；case/node数不同，不能视为同集合复现。"]) + " |")
    lines.append("")
    return lines


def plot_results(runs, folder: Path, root: Path):
    # Windows bundled runtime needs its explicitly installed matplotlib dependencies.
    if sys.platform == "win32" and (root / ".local_deps").is_dir():
        sys.path.insert(0, str(root / ".local_deps"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    artifacts = []
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    cohorts = {}
    for run in runs:
        if run["phase"] == "completed" and field_l2(run["best"]) is not None:
            cohorts.setdefault(tuple(sorted(run["case_keys"])), []).append(run)
    for i, (keys, group) in enumerate(cohorts.items()):
        values = np.asarray([field_l2(run["best"]) * 100 for run in group])
        fig, ax = plt.subplots(figsize=(12, max(3.7, .42 * len(group) + 2.2)))
        y = np.arange(len(group))
        bars = ax.barh(y, values, color=["#4477AA" if run["phase"] == "completed" else "#88CCEE" for run in group])
        for bar, run in zip(bars, group):
            if run["phase"] != "completed":
                bar.set_hatch("//")
        ax.set_yticks(y, [run["name"] for run in group])
        ax.invert_yaxis()
        positive = values[values > 0]
        use_log = len(positive) == len(values) and len(positive) and positive.max() / positive.min() > 100
        if use_log:
            ax.set_xscale("log")
            ax.set_xlim(positive.min() / 2, positive.max() * 2.5)
        else:
            ax.set_xlim(0, max(float(values.max()) * 1.4, .001))
        for j, value in enumerate(values):
            ax.annotate(f"{value:.5f}%", (value, j), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9)
        ax.set_xlabel("All original FEM nodes: decoded stress relative L2 (%)" + ("; log axis" if use_log else ""))
        title = ", ".join(keys) if len(keys) <= 5 else f"{len(keys)} selected fitting cases"
        ax.set_title("Observed best checkpoints — " + title, fontsize=11, pad=14)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        fig.text(.02, .025, "Completed fitting runs only. Different budgets, capacities and checkpoint objectives; no generalization claim.", fontsize=8)
        fig.tight_layout(rect=(0, .07, 1, 1))
        path = folder / f"stress_cases_{len(keys):02d}_{i:02d}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        artifacts.append(str(path))

    active = [run for run in runs if run["phase"] == "completed" and run["history"]]
    # Short pages keep both labels and legends readable as all methods finish.
    for page in range(0, len(active), 6):
        group = active[page:page + 6]
        rows_on_page = math.ceil(len(group) / 2)
        fig, axes = plt.subplots(rows_on_page, 2, figsize=(14, 4 * rows_on_page), squeeze=False)
        for ax, run in zip(axes.flat, group):
            rows = run["history"]
            args = run["config"].get("arguments", {})
            true_epoch = bool(args.get("epochs"))
            batches = run["config"].get("batches_per_epoch")
            if true_epoch and all(number(row.get("epoch")) is not None for row in rows):
                x = [row["epoch"] for row in rows]
                xlabel = "Complete fitting epochs"
            elif true_epoch and batches:
                x = [row["step"] / batches for row in rows]
                xlabel = "Complete pooled-node epochs"
            else:
                x, xlabel = [row["step"] for row in rows], "Optimizer updates"
            series = [("FEM stress", "stress", "pooled", "#4477AA"),
                      ("FEM displacement", "displacement", "pooled", "#228833"),
                      ("Grid stress", "stress", "grid", "#CC6677"),
                      ("Teacher points", "stress", "teacher", "#AA3377")]
            if run["case_count"] and run["case_count"] > 1:
                series += [("Worst-case stress", "stress_worst_l2", None, "#EE7733")]
            for label, field, domain, color in series:
                pairs = [(xx, field_l2(row, field, domain) if domain else number(row.get(field))) for xx, row in zip(x, rows)]
                pairs = [(xx, yy * 100) for xx, yy in pairs if yy is not None]
                if pairs:
                    ax.plot(*zip(*pairs), label=label, color=color, linewidth=1.4)
            ax.set_yscale("symlog", linthresh=.01)
            ax.set_ylim(bottom=0)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Physical relative L2 (%)")
            ax.set_title(run["name"] + " — " + run["phase"], fontsize=10)
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=8, loc="best")
        for ax in list(axes.flat)[len(group):]:
            ax.set_visible(False)
        fig.suptitle("Original DeepJEB fitting histories; every shown value is a saved evaluation", fontsize=13)
        fig.text(.02, .015, "Fitting only; no heldout or remeshing generalization. Error metrics are decoded and uncapped. Symlog y axis.", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .97))
        path = folder / f"learning_curves_{page // 6 + 1:02d}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        artifacts.append(str(path))
    return artifacts


def make_report(payload):
    runs = payload["runs"]
    counts = payload["status_counts"]
    completed = counts.get("completed", 0)
    lines = ["# DeepJEB 全方法复现进度与结果", "",
             f"快照时间：{payload['generated_at_utc']}。已完成 {completed}/{len(runs)} 个计划实验。",
             "",
             "本表使用 original DeepJEB v1.0；应力保留源数据符号。只汇总当前计划明确列出的实验，旧 H5 字段与坐标错位的结果已撤回，不纳入本表。所有误差均为解码后的物理相对 L2，表内单位为 %，没有截断。— 表示该实验不输出/不涉及该指标或案例；pending 仅表示尚无保存的测量。原稿未公开的指标写“未提供”。进行中实验的最佳值只是当前已观测最佳。",
             "",
             "主对照是 506_533/ver 与 506_25/tor，共 243564 个原始节点。五样本补充对照再加入 10_30/ver、10_421/hor、10_460/dia。50 CAD 使用官方前 50 个 CAD 的明确替代集合；原 270 训练池和历史随机 50 CAD 无法获取，因此不能把其数值与历史结果视为同一集合的重复实验。",
             "",
             "这些实验检验训练样本拟合；没有证明 heldout、重网格或跨边界条件泛化。不同容量、预算和 checkpoint 目标不能作为等容量模型排名。",
             "",
             "## 复现范围", "",
             "| 原稿中的方法/实验 | 本次状态 |",
             "|---|---|",
             "| 坐标 MLP、小 NGP、12-level neural field、Transolver++、TS3、输出端 residual、64³/192³ FNO、FNO teacher distillation、50 CAD shared field | 有量化实验与可恢复的部分配置，按计划实际运行；已完成/待运行见下表。原始 Transolver 为用户新增对照。 |",
             "| Geometry+BC separate branches、mesh-flow priors、geometry latent TS3、BenchCAD tension/multi-BC、FNO branches | 仅历史概述；本 slides 未提供量化结果/完整配置，当前无法精确复现。未声称这些历史架构已训练完成。 |",
             "",
             "## FEM 节点结果", "",
             "| 实验 | 状态 | 最佳步 / 已跑 / 计划 | 应力 pooled L2 % | 应力 case均值 / 最差 % | 位移 pooled L2 % | 已耗时 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    phase_names = {"pending": "pending", "waiting": "等待输入", "running": "进行中（暂定最佳）", "completed": "已完成", "failed": "失败"}
    for run in runs:
        metric = run["best"]
        has_stress = run["outputs"] in ("stress", "both")
        has_u = run["outputs"] in ("u", "both")
        steps = " / ".join(str(run[name]) if run.get(name) is not None else "pending" for name in ("best_step", "step", "total_steps"))
        avg_worst = percent(metric.get("stress_case_mean_l2")) + " / " + percent(metric.get("stress_worst_l2")) if has_stress else "—"
        lines.append(f"| {escape(run['name'])} | {phase_names[run['phase']]} | {steps} | {metric_cell(field_l2(metric), has_stress)} | {avg_worst} | {metric_cell(field_l2(metric, 'displacement'), has_u)} | {duration(run['elapsed_seconds'])} |")
    lines += ["", "最佳 checkpoint 的选择标准见下表；显示 FEM 误差并不表示使用 FEM 误差选择网格模型或蒸馏模型。", "",
              "## 网格与 teacher 的独立误差", "",
              "| 实验 | occupied-grid stress L2 % | teacher点 stress L2 % | FEM-node stress L2 % | 真场 roundtrip stress L2 % |",
              "|---|---:|---:|---:|---:|"]
    for run in runs:
        if run["name"].startswith("fno"):
            roundtrip = (run.get("truth_roundtrip") or {}).get("metrics", {}).get("stress", {}).get("relative_l2")
            is_distill = "distill" in run["name"]
            lines.append(f"| {run['name']} | {metric_cell(field_l2(run['best'], domain='grid'), not is_distill)} | {metric_cell(field_l2(run['best'], domain='teacher'), is_distill)} | {percent(field_l2(run['best']))} | {metric_cell(roundtrip, not is_distill)} |")
    lines += ["", "真场 roundtrip 是实际 FEM→网格→节点的表示/插值损失，不是模型误差的数学下界。蒸馏仅使用 FNO occupied-grid 预测监督；FEM 标签不参与神经场 scaler、loss 或 checkpoint 选择。", ""]
    lines += comparison_report(payload)
    lines += ["## 配置与缺口", "",
              "原实验服务器目录不可访问；演示稿明确给出的配置予以保留，缺失部分使用明确记录的官方 defaults/preset 或替代值，不能称逐项恢复了历史配置。Transformer 保持整张 FEM 网格上下文；TS3 分块是官方全局统计/回写算法，未把独立分块误当全网格评估。Transolver++ 保留官方 eval 时的随机 Gumbel 行为。", "",
              "| 实验 | 演示稿可确认项 | 本次配置 / 来源 | checkpoint 选择 | 配置缺口与补充 |",
              "|---|---|---|---|---|"]
    for gap in payload["configuration_gaps"]:
        changes = " ".join(dict.fromkeys(translate_gap(x) for x in gap["assumptions"])) or "原目录/源码版本/25 特征精确定义不可获取。"
        if gap.get("backend") == "tcnn-fused" and any("25 BC/material" in g for g in gap["reference_gaps"]):
            changes += " slides 仅提25维；本次从真实 FEM deck 构造明确替代 schema，历史定义缺失。"
        actual = gap["implementation"]
        if gap["source"]:
            actual += f"；[{gap['commit'][:12]}]({gap['source']}/tree/{gap['commit']})"
        lines.append("| " + " | ".join(escape(x) for x in (gap["run"], gap["slide_explicit"], actual, gap["selection"], changes)) + " |")
    lines += ["", "已运行实验的精确参数、scaler 数值、精度设置、数据路径和原始配置缺口保存于 notes/all_methods_results.json，以及各 run/config.json。Torch/Transformer 使用 float32 张量并明确允许 TF32；官方 TCNN FullyFused 使用 native FP16 computation、FP32 master parameters、返回 float32 输出。历史精度模式未知。", "",
              "50 样本随机阶段使用全池均匀、有放回节点采样；后续每个 true epoch 将实际入选节点打乱后逐一遍历，最后不足 batch 的部分也计入。batch 总量为 51200，实际 updates/epoch 由节点数重新计算；continuation 从本次最佳权重开始，重新初始化 Adam。", ""]
    for run in runs:
        if run["case_count"] == 50 and run["best"].get("stress_case_count"):
            m = run["best"]
            lines.append(f"{run['name']} 当前最佳：{m['stress_cases_below_1pct']}/{m['stress_case_count']} 个拟合案例应力 L2 <1%；最优 {m['stress_best_key']}（{percent(m['stress_best_l2'])}%），最差 {m['stress_worst_key']}（{percent(m['stress_worst_l2'])}%），case均值 hotspot L2={percent(m.get('stress_case_mean_hotspot_l2'))}%。")
            groups = ", ".join(f"{load}: {percent(info['case_mean_relative_l2'])}% ({info['cases']} cases)" for load, info in m["stress_load_groups"].items())
            lines += ["", "按 load 的逐案例平均应力 L2：" + groups + "。", ""]
    if payload["plots"]:
        lines += ["## 静态科学图", ""]
        for path in payload["plots"]:
            link = host_report_path(path)
            lines.append(f"- [{link.rsplit('/', 1)[-1]}](<{link}>)")
        lines.append("")
    fields = (payload.get("representative_fields") or {}).get("rendered", [])
    if fields:
        lines += ["## 代表性场图", "", "同一案例的 GT / prediction 使用共同 signed color scale；误差图为 absolute residual / true RMS，颜色与指标均不截断。仅绘制已完成且与正确 prepared 节点顺序、标签吻合的实验。", ""]
        for item in fields:
            link = host_report_path(item["figure"])
            lines.append(f"- [{item['label']} — {item['sample_key']}，stress L2={percent(item['relative_l2'])}%](<{link}>)")
        lines.append("")
    if payload["warnings"]:
        lines += ["## 本次读取说明", ""]
        lines += ["- " + escape(w) for w in payload["warnings"]]
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path, default=Path("notes/all_methods_plan.json"))
    parser.add_argument("--report", type=Path, default=Path("ALL_METHODS_RESULTS.md"))
    parser.add_argument("--json", type=Path, default=Path("notes/all_methods_results.json"))
    parser.add_argument("--plots", type=Path, default=Path("plots/all_methods"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    warnings = []
    plan_path = resolve_source(args.plan, root)
    plan = read_json(plan_path, warnings)
    if not isinstance(plan, dict) or not isinstance(plan.get("jobs"), list):
        raise ValueError(f"Missing or invalid suite plan: {plan_path}")
    coordinator = read_json(root / "runs/slides_reproduction/coordinator_status.json", warnings) or {}
    runs = [read_run(job, root, coordinator, warnings) for job in plan["jobs"]]
    sources = read_json(root / "third_party/transformer_sources.json", warnings) or {}
    payload = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "dataset": plan.get("dataset"), "main_cases": plan.get("main_cases"),
               "five_cases": plan.get("five_cases"), "plan_source": str(plan_path),
               "scope": "training-case fitting only; no generalization claim",
               "qualitative_only_methods": [
                   {"method": name, "slide": 2, "status": "historical overview only; these slides do not provide quantitative results or a complete configuration; exact reproduction currently unavailable"}
                   for name in ("Geometry+BC separate branches", "mesh-flow priors", "geometry latent TS3", "BenchCAD tension/multi-BC", "FNO branches")],
               "exclusion": "withdrawn H5-only coordinate/field-mismatched results and all runs outside the plan",
               "metric_units": "relative L2 fractions in JSON; physical-unit decoding; no clipping",
               "status_counts": dict(Counter(run["phase"] for run in runs)), "coordinator": coordinator,
               "completion_basis": "completed summary.json for each allowlisted plan job; not coordinator startup total_jobs (which may omit appended TCNN jobs)",
               "runs": runs, "configuration_gaps": [configuration_details(run, sources) for run in runs],
               "slide_comparisons": reference_comparisons(root, runs, warnings),
               "representative_fields": read_json(root / "plots/all_methods/fields/field_plot_metrics.json", warnings),
               "plots": [], "warnings": warnings}
    if not args.no_plots:
        try:
            payload["plots"] = plot_results(runs, resolve_source(args.plots, root), root)
        except ImportError as exc:
            warnings.append(f"Plots unavailable: {exc}; measurements and report were still saved.")
    report_path = resolve_source(args.report, root)
    json_path = resolve_source(args.json, root)
    atomic_text(json_path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    atomic_text(report_path, make_report(payload))
    print(json.dumps({"report": str(report_path), "json": str(json_path),
                      "plots": payload["plots"], "status_counts": payload["status_counts"],
                      "warnings": warnings}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
