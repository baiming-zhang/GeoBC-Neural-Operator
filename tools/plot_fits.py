#!/usr/bin/env python3
"""Render original DeepJEB surfaces and fitting-only metrics as static PNGs.

Examples:
  python tools/plot_fits.py --runs runs/single runs/five --output plots
  python tools/plot_fits.py --reference data/prepared/10_30_ver.npz --output plots

No mesh resampling or face decimation is performed. Original surface
triangles are orthographically projected, depth-sorted and drawn as opaque
2D polygons, avoiding matplotlib's expensive 3D object pipeline. Colors are
triangle means of the nodal fields. Metrics use every original FEM node,
including interior nodes, and are never clipped to the display scale.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
import time

# Match the project-local Windows preprocessing/plotting dependencies.
ROOT = Path(__file__).resolve().parents[1]
if os.name == 'nt' and (ROOT / ".local_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".local_deps"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from PIL import Image

LOAD_NAMES = {"ver": "vertical", "hor": "horizontal", "dia": "diagonal", "tor": "torsion"}


@dataclass
class Fit:
    key: str
    xyz: np.ndarray
    faces: np.ndarray
    truth: np.ndarray
    prediction: np.ndarray
    step: int
    metrics: dict
    source: Path


def _native_path(value: str | Path) -> Path:
    text = str(value)
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", text):
        return Path("/mnt") / text[0].lower() / text[3:].replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-z]/", text):
        return Path(text[5].upper() + ":/" + text[7:])
    return Path(text).expanduser()


def _scalar(archive, key: str):
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar")
    result = value.reshape(-1)[0].item()
    return result.decode("utf-8") if isinstance(result, bytes) else result


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    residual = prediction - truth
    count = max(1, math.ceil(len(truth) * 0.01))
    hotspots = np.argpartition(np.abs(truth), len(truth) - count)[-count:]
    return {
        "nodes": len(truth), "hotspot_nodes": count,
        "relative_l2": float(np.linalg.norm(residual) / max(np.linalg.norm(truth), 1e-12)),
        "hotspot_relative_l2": float(np.linalg.norm(residual[hotspots]) / max(np.linalg.norm(truth[hotspots]), 1e-12)),
        "rmse_MPa": float(np.sqrt(np.mean(residual**2))),
        "true_rms_MPa": float(np.sqrt(np.mean(truth**2))),
        "max_abs_error_MPa": float(np.max(np.abs(residual))),
        "max_error_percent_true_rms": float(np.max(np.abs(residual)) / max(np.sqrt(np.mean(truth**2)), 1e-12) * 100),
    }


def _mesh(path: Path) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"sample_key", "xyz_mm", "faces", "stress"}
        if not required.issubset(archive.files):
            raise ValueError(f"{path}: missing {sorted(required.difference(archive.files))}")
        key = str(_scalar(archive, "sample_key"))
        xyz = np.asarray(archive["xyz_mm"], dtype=np.float64)
        faces = np.asarray(archive["faces"], dtype=np.int32)
        stress = np.asarray(archive["stress"], dtype=np.float64).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or stress.shape != (len(xyz),):
        raise ValueError(f"{path}: expected xyz_mm [N,3], stress [N]")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError(f"{path}: expected nonempty source triangles [T,3]")
    if faces.min() < 0 or faces.max() >= len(xyz):
        raise ValueError(f"{path}: invalid surface vertex indices")
    if not np.isfinite(xyz).all() or not np.isfinite(stress).all():
        raise ValueError(f"{path}: nonfinite coordinates or stress")
    return key, xyz, faces, stress


def _prepared_index(entries: list[Path] | None) -> dict[str, Path]:
    mapping = {}
    for entry in entries or []:
        entry = _native_path(entry).resolve()
        paths = sorted(entry.rglob("*.npz")) if entry.is_dir() else [entry]
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                key = str(_scalar(archive, "sample_key"))
            if key in mapping:
                raise ValueError(f"Duplicate prepared case {key}: {path}")
            mapping[key] = path
    return mapping


def _curve_history(folder: Path, config: dict, visited: set[Path] | None = None) -> tuple[list[dict], list[str]]:
    """Join actual ancestor evaluations only up to each resumed checkpoint."""
    visited = set() if visited is None else visited
    if folder in visited:
        return [], []
    visited.add(folder)
    with (folder / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        current = list(csv.DictReader(handle))
    history, sources = [], []
    resume = config.get("arguments", {}).get("resume")
    if resume and current:
        checkpoint = _native_path(resume)
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        parent = checkpoint.resolve().parent
        parent_config_path = parent / "config.json"
        if parent not in visited and parent_config_path.exists() and (parent / "metrics.csv").exists():
            parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
            keys = lambda record: [case["sample_key"] for case in record.get("cases", [])]
            if keys(parent_config) == keys(config):
                previous, sources = _curve_history(parent, parent_config, visited)
                resume_step = min(float(row["step"]) for row in current)
                history = [row for row in previous if float(row["step"]) <= resume_step]
    # The child evaluation at the resume update is authoritative if duplicated.
    by_step = {float(row["step"]): row for row in history + current}
    return [by_step[step] for step in sorted(by_step)], sources + [str(folder / "metrics.csv")]


def _load_run(folder: Path, prepared: dict[str, Path], selection: str) -> dict:
    folder = _native_path(folder).resolve()
    predictions = folder / f"predictions_{selection}"
    index = json.loads((predictions / "index.json").read_text(encoding="utf-8"))
    config_path = folder / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    fits = []
    for key, record in index["cases"].items():
        source = prepared.get(key, _native_path(record["source"]))
        if not source.exists():
            local = ROOT / "data" / "prepared" / (key.replace("/", "_") + ".npz")
            source = local if local.exists() else folder / source
        mesh_key, xyz, faces, source_stress = _mesh(source)
        with np.load(predictions / record["file"], allow_pickle=False) as archive:
            saved_key = str(_scalar(archive, "sample_key"))
            truth = np.asarray(archive["stress_true"], dtype=np.float64).reshape(-1)
            prediction = np.asarray(archive["stress_pred"], dtype=np.float64).reshape(-1)
            step = int(_scalar(archive, "step"))
        if key != saved_key or key != mesh_key or truth.shape != source_stress.shape or prediction.shape != truth.shape:
            raise ValueError(f"{key}: prepared and prediction case/node ordering differ")
        if not np.allclose(truth, source_stress, rtol=2e-6, atol=1e-8):
            raise ValueError(f"{key}: saved labels differ from original prepared labels")
        if not np.isfinite(prediction).all():
            raise ValueError(f"{key}: nonfinite physical prediction")
        fits.append(Fit(key, xyz, faces, truth, prediction, step, _metrics(truth, prediction), source))
    curve, curve_sources = _curve_history(folder, config)
    return {"folder": folder, "fits": fits, "curve": curve, "config": config,
            "label": f"{len(fits)}-case fit", "selection": selection, "curve_sources": curve_sources}


def _project(xyz: np.ndarray, faces: np.ndarray, elev: float, azim: float) -> tuple[np.ndarray, np.ndarray, tuple]:
    azimuth, elevation = np.deg2rad([azim, elev])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    up = np.array([-np.sin(elevation) * np.cos(azimuth), -np.sin(elevation) * np.sin(azimuth), np.cos(elevation)])
    toward_camera = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    centered = xyz - (xyz.min(axis=0) + xyz.max(axis=0)) * 0.5
    xy = np.column_stack((centered @ right, centered @ up))
    depth = centered @ toward_camera
    order = np.argsort(depth[faces].mean(axis=1), kind="stable")
    vertices = xy[faces[order]]
    minimum, maximum = vertices.reshape(-1, 2).min(axis=0), vertices.reshape(-1, 2).max(axis=0)
    padding = max(float((maximum - minimum).max()), 1e-9) * 0.06
    limits = (minimum[0] - padding, maximum[0] + padding, minimum[1] - padding, maximum[1] + padding)
    return vertices, order, limits


def _surface(ax, vertices: np.ndarray, limits: tuple, values: np.ndarray, norm, cmap: str, title: str) -> None:
    # Opaque faces in far-to-near order; no alpha/RGB blending or lighting.
    colors = matplotlib.colormaps[cmap](norm(values))
    ax.add_collection(PolyCollection(vertices, facecolors=colors, edgecolors="none", antialiaseds=False))
    ax.set_xlim(limits[:2])
    ax.set_ylim(limits[2:])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, pad=9)


def _load_description(key: str) -> str:
    code = key.rsplit("/", 1)[-1]
    return LOAD_NAMES.get(code, code)


def _save(fig, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def _render_fit(fit: Fit, label: str, path: Path, dpi: int, elev: float, azim: float) -> None:
    vertices, order, limits = _project(fit.xyz, fit.faces, elev, azim)
    face_truth = fit.truth[fit.faces].mean(axis=1)[order]
    face_prediction = fit.prediction[fit.faces].mean(axis=1)[order]
    # Mean nodal absolute error per face, not abs(mean signed residual).
    error = np.abs(fit.prediction - fit.truth) / max(fit.metrics["true_rms_MPa"], 1e-12) * 100
    face_error = error[fit.faces].mean(axis=1)[order]
    signed_limit = max(float(np.max(np.abs(fit.truth))), float(np.max(np.abs(fit.prediction))), 1e-12)
    signed_norm = TwoSlopeNorm(vmin=-signed_limit, vcenter=0, vmax=signed_limit)
    error_norm = Normalize(vmin=0, vmax=max(float(error.max()), 1e-12))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.235, top=0.765, wspace=0.035)
    fig.suptitle(f"{label} | {fit.key} ({_load_description(fit.key)}) | FITTING ONLY", fontsize=16, y=0.975)
    fig.text(0.5, 0.905,
             f"Selected update {fit.step:,}  |  Physical relative L2 {fit.metrics['relative_l2'] * 100:.4g}%"
             f"  |  Hotspot relative L2 {fit.metrics['hotspot_relative_l2'] * 100:.4g}%"
             f"  |  True RMS {fit.metrics['true_rms_MPa']:.5g} MPa", ha="center", fontsize=11)
    for ax, values, norm, cmap, title in zip(axes, [face_truth, face_prediction, face_error],
                                           [signed_norm, signed_norm, error_norm],
                                           ["coolwarm", "coolwarm", "magma"],
                                           ["Ground truth", "Prediction", "Absolute error / true RMS"]):
        _surface(ax, vertices, limits, values, norm, cmap, title)
    signed_bar = fig.add_axes([0.105, 0.14, 0.46, 0.035])
    fig.colorbar(ScalarMappable(norm=signed_norm, cmap="coolwarm"), cax=signed_bar, orientation="horizontal")
    signed_bar.set_xlabel("Source signed stress (MPa) — common ground-truth / prediction scale", fontsize=10)
    error_bar = fig.add_axes([0.715, 0.14, 0.23, 0.035])
    fig.colorbar(ScalarMappable(norm=error_norm, cmap="magma"), cax=error_bar, orientation="horizontal")
    error_bar.set_xlabel("Absolute error (% of all-node true RMS)", fontsize=10)
    fig.text(0.5, 0.025,
             f"Original surface: {len(fit.faces):,} triangles; {len(fit.truth):,} FEM nodes. "
             f"Colors: face means. Metrics: all nodes; hotspot = highest-|truth| 1% ({fit.metrics['hotspot_nodes']:,} nodes). "
             "No held-out evaluation.", ha="center", fontsize=9, color="#505966")
    _save(fig, path, dpi)


def _curves(runs: list[dict], path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.4))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.19, top=0.80, wspace=0.20)
    fig.suptitle("DeepJEB fitting curves | original nodes | FITTING ONLY", fontsize=16, y=0.96)
    colors = matplotlib.colormaps["tab10"]
    for number, run in enumerate(runs):
        rows = run["curve"]
        x = np.array([float(row["step"]) for row in rows])
        color = colors(number % 10)
        for ax, key in zip(axes, ("worst_case_relative_l2", "worst_hotspot_relative_l2")):
            y = np.array([float(row[key]) * 100 for row in rows])
            ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=3, label=run["label"])
        best_step = run["fits"][0].step
        axes[0].axvline(best_step, color=color, linewidth=1, linestyle=":", alpha=0.6)
    for ax, title in zip(axes, ("Worst-case physical relative L2", "Worst-case hotspot relative L2")):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Optimizer update")
        ax.set_ylabel("Error (%)")
        # Symlog preserves exact zero rather than imposing a metric floor.
        ax.set_yscale("symlog", linthresh=0.001)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.20, which="both")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.03, "Decoded physical-unit errors; resumed histories joined at checkpoint updates. Dotted lines: selected updates. "
             "Hotspots: highest-|true stress| 1% per case. Selected cases are training cases.",
             ha="center", fontsize=9, color="#505966")
    _save(fig, path, dpi)


def _bars(runs: list[dict], path: Path, dpi: int) -> None:
    selected = [run for run in runs if len(run["fits"]) > 1] or runs
    entries = [(run["label"], fit) for run in selected for fit in run["fits"]]
    x = np.arange(len(entries))
    l2 = np.array([fit.metrics["relative_l2"] * 100 for _, fit in entries])
    hotspot = np.array([fit.metrics["hotspot_relative_l2"] * 100 for _, fit in entries])
    fig, ax = plt.subplots(figsize=(16, 4.4))
    fig.subplots_adjust(left=0.065, right=0.975, bottom=0.23, top=0.81)
    fig.suptitle("Selected checkpoint: per-case physical errors | FITTING ONLY", fontsize=16, y=0.96)
    first = ax.bar(x - 0.19, l2, 0.36, label="All-node relative L2", color="#4867a8")
    second = ax.bar(x + 0.19, hotspot, 0.36, label="Hotspot relative L2", color="#df8950")
    for bars in (first, second):
        ax.bar_label(bars, labels=[f"{bar.get_height():.4g}%" for bar in bars], fontsize=10, padding=3)
    labels = [fit.key + ("\n" + label if len(selected) > 1 else "") for label, fit in entries]
    ax.set_xticks(x, labels)
    ax.set_ylabel("Physical relative L2 (%)")
    ax.set_ylim(0, max(float(max(l2.max(), hotspot.max())) * 1.45, 0.01))
    ax.grid(axis="y", alpha=0.20)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.text(0.5, 0.035, "Unclipped decoded predictions. All-node errors include interior FEM nodes. "
             "Hotspots are selected from true stress; fitting does not establish generalization.",
             ha="center", fontsize=9, color="#505966")
    _save(fig, path, dpi)


def _contact(paths: list[Path], target: Path, width: int) -> None:
    # Compose completed figure rasters, keeping only one mesh figure live.
    sizes = []
    for path in paths:
        with Image.open(path) as source:
            sizes.append((width, round(source.height * width / source.width)))
    gap = 22
    canvas = Image.new("RGB", (width, sum(height for _, height in sizes) + gap * (len(sizes) + 1)), "white")
    offset = gap
    for path, size in zip(paths, sizes):
        with Image.open(path) as source:
            resized = source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
            canvas.paste(resized, (0, offset))
        offset += size[1] + gap
    canvas.save(target)


def _reference(source: Path, target: Path, dpi: int, elev: float, azim: float) -> dict:
    key, xyz, faces, truth = _mesh(source)
    signed_limit = max(float(np.max(np.abs(truth))), 1e-12)
    norm = TwoSlopeNorm(vmin=-signed_limit, vcenter=0, vmax=signed_limit)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.235, top=0.77, wspace=0.035)
    fig.suptitle(f"{key} ({_load_description(key)}) | original DeepJEB ground truth | view reference", fontsize=16, y=0.975)
    fig.text(0.5, 0.905,
             f"{len(xyz):,} original FEM nodes | {len(faces):,} original surface triangles | "
             f"Signed stress range [{truth.min():.5g}, {truth.max():.5g}] MPa", ha="center", fontsize=11)
    for ax, angle in zip(axes, [azim, azim + 90, azim + 180]):
        vertices, order, limits = _project(xyz, faces, elev, angle)
        _surface(ax, vertices, limits, truth[faces].mean(axis=1)[order], norm, "coolwarm",
                 f"Azimuth {angle:g}° / elevation {elev:g}°")
    cax = fig.add_axes([0.26, 0.14, 0.48, 0.035])
    fig.colorbar(ScalarMappable(norm=norm, cmap="coolwarm"), cax=cax, orientation="horizontal")
    cax.set_xlabel("Source signed stress (MPa) — identical scale for all views", fontsize=10)
    fig.text(0.5, 0.025, "Original quadratic FEM boundary; original midside nodes retained. "
             "Orthographic projection with depth sorting; triangle-mean nodal colors.",
             ha="center", fontsize=9, color="#505966")
    _save(fig, target, dpi)
    return {"sample_key": key, "nodes": len(xyz), "surface_triangles": len(faces), "figure": str(target)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--runs", nargs="+", type=Path, help="Trainer run directories")
    mode.add_argument("--reference", nargs="+", type=Path, help="Prepared NPZs for ground-truth view references")
    parser.add_argument("--prepared", nargs="+", type=Path, help="Optional source NPZ files or directories overriding index paths")
    parser.add_argument("--run-labels", nargs="+", help="Optional names matching --runs order")
    parser.add_argument("--selection", choices=("best", "final"), default="best")
    parser.add_argument("--output", type=Path, default=ROOT / "plots")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--contact-width", type=int, default=2400)
    parser.add_argument("--elev", type=float, default=24)
    parser.add_argument("--azim", type=float, default=-65)
    args = parser.parse_args()
    started = time.monotonic()
    output = _native_path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "normal"})
    if args.reference:
        records = []
        for number, source in enumerate(args.reference):
            target = output / ("reference.png" if number == 0 else f"reference_{number:02d}.png")
            records.append(_reference(_native_path(source), target, args.dpi, args.elev, args.azim))
        print(json.dumps({"reference": records, "elapsed_seconds": time.monotonic() - started}, indent=2), flush=True)
        return
    if args.run_labels and len(args.run_labels) != len(args.runs):
        parser.error("--run-labels must have one entry per run")
    prepared = _prepared_index(args.prepared)
    runs = [_load_run(folder, prepared, args.selection) for folder in args.runs]
    for number, run in enumerate(runs):
        if args.run_labels:
            run["label"] = args.run_labels[number]
    individual = output / "individuals"
    individual.mkdir(exist_ok=True)
    curve_path, bar_path = output / "learning_curves.png", output / "case_errors.png"
    _curves(runs, curve_path, args.dpi)
    _bars(runs, bar_path, args.dpi)
    images = [curve_path, bar_path]
    records = []
    for run_number, run in enumerate(runs):
        for fit in run["fits"]:
            safe_key = re.sub(r"[^A-Za-z0-9_.-]", "_", fit.key)
            target = individual / f"run_{run_number:02d}__{safe_key}.png"
            _render_fit(fit, run["label"], target, args.dpi, args.elev, args.azim)
            images.append(target)
            records.append({"run": str(run["folder"]), "label": run["label"], "sample_key": fit.key,
                            "step": fit.step, "prepared_source": str(fit.source), "figure": str(target), **fit.metrics})
            print(f"Rendered {run['label']} {fit.key}: L2 {fit.metrics['relative_l2'] * 100:.5g}%", flush=True)
    _contact(images, output / "contact.png", args.contact_width)
    report = {"dataset": "DeepJEB original", "evaluation": "selected training cases; fitting only",
              "selection": args.selection, "surface_display": "original quadratic FEM boundary triangles; face-mean colors; orthographic depth sorting",
              "metrics": "all original FEM nodes; unclipped physical predictions; hotspot = ceil(0.01*N) highest-|truth| nodes",
              "error_map": "mean nodal absolute error / all-node true RMS * 100 per surface triangle",
              "curve_sources": [{"label": run["label"], "files": run["curve_sources"]} for run in runs],
              "view": {"elevation": args.elev, "azimuth": args.azim}, "cases": records,
              "elapsed_seconds": time.monotonic() - started}
    (output / "plot_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"contact": str(output / "contact.png"), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
