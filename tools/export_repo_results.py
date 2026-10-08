"""Export the aligned, allowlisted suite results for a future private repository.

This CPU-only exporter does not invoke Git, GitHub, a model, or a renderer.
Default publication gate: all 30 plan jobs have completed summaries and the
local report snapshot has been refreshed. --allow-incomplete makes an explicitly
marked preparation bundle. Large data, checkpoints and full-node predictions
are indexed, not copied or hashed by default.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SMALL_RUN_FILES = ("config.json", "summary.json", "metrics_best.json", "metrics_latest.json",
                   "metrics.jsonl", "metrics.csv", "truth_roundtrip.json")
METADATA_FILES = ("notes/all_methods_plan.json", "notes/all_methods_results.json",
                  "notes/slide_reference_metrics.json", "references/slides_text.json",
                  "third_party/transformer_sources.json")
LINK = re.compile(r"(?P<opening>!?\[[^\]\n]*\]\()(?P<target><[^>\n]*>|[^)\n]*)(?P<closing>\))")


def native_path(value, root=ROOT):
    value = str(value).replace("\\", "/")
    mounted = re.match(r"^/mnt/([A-Za-z])/(.*)$", value)
    if os.name == "nt" and mounted:
        value = mounted[1].upper() + ":/" + mounted[2]
    elif os.name != "nt" and re.match(r"^[A-Za-z]:/", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    result = Path(value).expanduser()
    return (result if result.is_absolute() else root / result).resolve()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_values(command, option):
    if option not in command:
        return []
    result = []
    for value in command[command.index(option) + 1:]:
        if str(value).startswith("--"):
            break
        result.append(value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--hash-unbundled", action="store_true",
                        help="Optionally scan omitted data/checkpoints/predictions; no scanning of mutable incomplete-run checkpoints")
    args = parser.parse_args()
    began = time.monotonic()
    output = native_path(args.output)
    # This script owns results/ only. Never recursively erase the workspace.
    if not output.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("Export destination must stay within this workspace's results/ directory")
    plan = load_json(ROOT / "notes/all_methods_plan.json")
    if not isinstance(plan, dict) or not isinstance(plan.get("jobs"), list) or len(plan["jobs"]) != 30:
        raise ValueError("The current valid reproduction plan must contain exactly 30 jobs")
    jobs = plan["jobs"]
    names = [job["name"] for job in jobs]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", "..") for name in names):
        raise ValueError("Invalid or duplicate plan run name")
    configs, summaries, folders = {}, {}, {}
    for job in jobs:
        folder = native_path(job["output"])
        if not folder.is_relative_to((ROOT / "runs/slides_reproduction").resolve()):
            raise ValueError(f"Run outside the valid aligned suite: {folder}")
        folders[job["name"]] = folder
        configs[job["name"]] = load_json(folder / "config.json") or {}
        summaries[job["name"]] = load_json(folder / "summary.json") or {}
    completed = [name for name in names if summaries[name].get("status") == "completed"]
    pending = [name for name in names if name not in completed]
    snapshot = load_json(ROOT / "notes/all_methods_results.json") or {}
    snapshot_names = {run.get("name") for run in snapshot.get("runs", [])}
    snapshot_complete = snapshot.get("status_counts", {}).get("completed") == 30 and snapshot_names == set(names)
    if not args.allow_incomplete and (pending or not snapshot_complete):
        raise RuntimeError(f"Formal export blocked: completed summaries {len(completed)}/30; refresh tools/summarize_suite.py after all stages complete. Use --allow-incomplete for a preparation bundle.")
    output.mkdir(parents=True, exist_ok=True)
    previous = load_json(output / "manifest.json") or {}
    exported, mapping, warnings, artifacts = [], {}, [], {}

    def save(value: bytes, target: Path, source=None, role="result"):
        destination = output / target
        if not destination.resolve().is_relative_to(output):
            raise ValueError("Export path escaped results/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(value)
        temporary.replace(destination)
        exported.append({"path": target.as_posix(), "bytes": len(value), "sha256": sha_bytes(value),
                         "role": role, "source": str(source) if source else None})
        if source:
            mapping[source.resolve()] = destination
        return destination

    def copy(source, target, role):
        if not source.is_file():
            warnings.append(f"Not available; not copied: {source}")
            return None
        return save(source.read_bytes(), target, source, role)

    def json_save(payload, target, role):
        return save((json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"), target, role=role)

    def index_artifact(path, kind, run=None):
        path = native_path(path)
        key = str(path)
        available = path.is_file()
        if key not in artifacts:
            artifacts[key] = {"path": key, "kind": [], "runs": [], "available": available,
                              "bytes": path.stat().st_size if available else None,
                              "bundled": False, "sha256": None,
                              "sha256_status": "not hashed: omitted artifact; default avoids multi-GB scans" if available else "unavailable"}
        entry = artifacts[key]
        if kind not in entry["kind"]:
            entry["kind"].append(kind)
        if run and run not in entry["runs"]:
            entry["runs"].append(run)

    run_records = []
    manifest_dirs = set()
    for job in jobs:
        name, folder, config = job["name"], folders[job["name"]], configs[job["name"]]
        copied = []
        for filename in SMALL_RUN_FILES:
            source = folder / filename
            if source.is_file():
                copy(source, Path("runs") / name / filename, "allowlisted_run_result")
                copied.append(filename)
        if name in completed and not all(filename in copied for filename in ("config.json", "summary.json", "metrics_best.json", "metrics_latest.json")):
            raise ValueError(f"Completed run lacks its required small result files: {name}")
        for case in config.get("cases", []):
            source = native_path(case["source"])
            index_artifact(source, "prepared_source", name)
            manifest_dirs.add(source.parent)
        for key in ("grid_source", "teacher_source"):
            if config.get(key):
                index_artifact(config[key], key, name)
        for option in ("--data", "--nodes", "--grid", "--teacher"):
            for value in command_values(job["command"], option):
                path = native_path(value)
                if option == "--data":
                    manifest_dirs.add(path if path.is_dir() else path.parent)
                else:
                    index_artifact(path, option[2:] + "_source", name)
                    if option == "--nodes":
                        manifest_dirs.add(path.parent)
        for filename in ("best.pt", "last.pt"):
            index_artifact(folder / filename, "checkpoint", name)
        for selection in ("best", "final"):
            pred_dir = folder / ("predictions_" + selection)
            index = load_json(pred_dir / "index.json")
            if index:
                copy(pred_dir / "index.json", Path("runs") / name / pred_dir.name / "index.json", "prediction_availability_index")
                for item in index.get("cases", index).values():
                    if isinstance(item, dict) and item.get("file"):
                        index_artifact(pred_dir / item["file"], "full_node_prediction", name)
        if (folder / "teacher_grid_best.npz").exists():
            index_artifact(folder / "teacher_grid_best.npz", "occupied_grid_teacher_prediction", name)
        run_records.append({"name": name, "completed": name in completed, "output_source": str(folder),
                            "best_step": summaries[name].get("best_step"), "copied_small_files": copied})

    # Only data directories actually requested by the aligned 30-job plan.
    for directory in sorted(manifest_dirs):
        if not directory.is_relative_to((ROOT / "data").resolve()) or directory.name == "prepared":
            raise ValueError(f"Unexpected or withdrawn prepared directory: {directory}")
        source = directory / "manifest.json"
        if source.is_file():
            manifest = load_json(source)
            if "original" not in str(manifest.get("dataset", "")).lower() or "++" in str(manifest.get("dataset", "")):
                raise ValueError(f"Non-original dataset manifest: {source}")
            copy(source, Path("data_manifests") / directory.name / "manifest.json", "prepared_data_manifest")
            for case in manifest.get("cases", []):
                if isinstance(case, dict) and case.get("npz"):
                    index_artifact(case["npz"], "prepared_source")

    for filename in METADATA_FILES:
        source = ROOT / filename
        copy(source, Path("metadata") / Path(filename).name, "suite_provenance")
    figures = list(snapshot.get("plots", []))
    fields = snapshot.get("representative_fields") or load_json(ROOT / "plots/all_methods/fields/field_plot_metrics.json") or {}
    valid_folders = set(folders.values())
    for record in fields.get("rendered", []):
        if native_path(record["run"]) in valid_folders:
            figures.append(record["figure"])
    figure_root = (ROOT / "plots/all_methods").resolve()
    for value in dict.fromkeys(figures):
        source = native_path(value)
        if not source.is_relative_to(figure_root):
            raise ValueError(f"Figure outside the current scientific suite: {source}")
        copy(source, Path("figures") / source.relative_to(figure_root), "scientific_figure")
    field_metadata = figure_root / "fields/field_plot_metrics.json"
    if field_metadata.exists():
        copy(field_metadata, Path("figures/fields/field_plot_metrics.json"), "representative_field_provenance")

    if args.hash_unbundled:
        def optional_hash(entry):
            if not entry["available"]:
                return
            if any(name not in completed for name in entry["runs"]) and "checkpoint" in entry["kind"]:
                entry["sha256_status"] = "not hashed: incomplete run can still write this checkpoint"
                return
            entry["sha256"] = sha_file(Path(entry["path"]))
            entry["sha256_status"] = "hashed on explicit --hash-unbundled request"
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(optional_hash, artifacts.values()))
    artifact_list = list(artifacts.values())
    json_save({"default_policy": "omitted data/checkpoint/prediction files are availability/size-indexed without a SHA scan",
               "hash_unbundled_requested": args.hash_unbundled, "artifacts": artifact_list}, Path("artifact_availability.json"), "large_artifact_availability")

    report_source = ROOT / "ALL_METHODS_RESULTS.md"
    report = report_source.read_text(encoding="utf-8")
    def portable_link(match):
        raw = match["target"].strip()
        target = raw[1:-1] if raw.startswith("<") and raw.endswith(">") else raw
        if re.match(r"^(https?://|mailto:|#)", target):
            return match[0]
        source = native_path(target, report_source.parent)
        destination = mapping.get(source)
        if destination and destination.is_file():
            relative = os.path.relpath(destination, output).replace("\\", "/")
            return match["opening"] + "<" + relative + ">" + match["closing"]
        # Code/source files supplied separately by the repository publication step.
        if source.is_file() and source.is_relative_to(ROOT) and source.suffix in (".py", ".md", ".txt"):
            relative = "../" + source.relative_to(ROOT).as_posix()
            return match["opening"] + "<" + relative + ">" + match["closing"]
        label = match["opening"].lstrip("!")[1:-2]
        warnings.append(f"Local report asset unavailable or outside selected bundle: {source}")
        return label + "（未打包本地资产）"
    report = LINK.sub(portable_link, report)
    prep = args.allow_incomplete
    status = "INCOMPLETE_PREPARATION_BUNDLE" if prep else "COMPLETE_RESULT_BUNDLE"
    heading = "# 准备包：未完成，不能作为最终复现结果发布" if prep else "# 已完成的复现结果包"
    notice = (f"{heading}\n\n导出时 completed summary 为 {len(completed)}/30；本地报告快照为 "
              f"{snapshot.get('status_counts', {}).get('completed', 'unknown')}/30。"
              "正式导出需全部 30 项完成并刷新报告。\n\n" if prep else
              f"{heading}\n\n30/30 个计划实验均有 completed summary。\n\n")
    notice += ("本包仅包含当前 30 项 aligned original-DeepJEB 计划的小型结果、数据 manifest 和现有科学图。"
               "撤回的旧 H5-only 拟合结果不在包内。原始数据、checkpoint、全节点 prediction 没有复制；"
               "其本地可用性与大小见 artifact_availability.json，默认 SHA 为 null 并注明未扫描。"
               "该导出脚本没有创建 Git 仓库或写入 GitHub。\n\n")
    save((notice + report).encode("utf-8"), Path("REPORT.md"), report_source, "portable_report")

    # Remove only previously indexed exporter-owned files which are no longer selected.
    # No recursive delete, and no removal of unindexed/manual files.
    selected = {item["path"] for item in exported}
    for item in previous.get("files", []):
        if item.get("path") not in selected:
            stale = (output / item["path"]).resolve()
            if not stale.is_relative_to(output):
                raise ValueError("Prior export index contained an escaping path")
            if stale.is_file():
                stale.unlink()
    complete = not args.allow_incomplete and len(completed) == 30 and snapshot_complete
    payload = {"format": "geobc-suite-results-v1", "status": status, "complete": complete,
               "generated_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": plan.get("dataset"),
               "allowlisted_job_count": 30, "completed_job_count": len(completed), "incomplete_jobs": pending,
               "report_snapshot_generated_at_utc": snapshot.get("generated_at_utc"),
               "publication_gate": "all 30 allowlisted completed summaries plus a refreshed complete report; no GitHub operation is performed",
               "excluded": ["withdrawn H5-only results", "all runs outside the plan", "raw datasets", "checkpoint tensors", "full-node prediction arrays"],
               "hash_policy": "SHA256 for every exported file; unbundled large artifacts not scanned by default",
               "bundle_file_count_including_manifest": len(exported) + 1,
               "payload_bytes_excluding_manifest": sum(item["bytes"] for item in exported),
               "scientific_figure_count": sum(item["role"] == "scientific_figure" for item in exported),
               "artifact_availability_count": len(artifact_list), "runs": run_records, "files": exported,
               "warnings": warnings, "elapsed_seconds": time.monotonic() - began}
    manifest_path = output / "manifest.json"
    manifest_temporary = manifest_path.with_suffix(".json.tmp")
    manifest_temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest_temporary.replace(manifest_path)
    legacy_manifest = output / "export_manifest.json"
    if legacy_manifest.is_file():
        legacy_manifest.unlink()
    print(json.dumps({"output": str(output), "status": status, "complete": complete, "completed": f"{len(completed)}/30",
                      "bundle_files": payload["bundle_file_count_including_manifest"],
                      "payload_bytes_excluding_manifest": payload["payload_bytes_excluding_manifest"],
                      "scientific_figures": payload["scientific_figure_count"],
                      "unbundled_artifacts_indexed": len(artifact_list), "warnings": warnings,
                      "elapsed_seconds": payload["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
