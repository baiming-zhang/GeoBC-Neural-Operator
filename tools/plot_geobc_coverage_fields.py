#!/usr/bin/env python3
"""Two predeclared original-node field illustrations, after closed DEV readout.

Fixed representatives: original Transolver200 and FNO200 at4800, first frozen
new DEV CAD/ver. Display sampling uses row indices only. No model, Torch, raw
dataset, checkpoint or GPU is opened. Metrics remain the evaluator's fullN
metrics, independent of graph sampling. Coordinates are restored from the
input's normalization metadata solely for display.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from tools.export_geobc_coverage_results import (ROOT, PLAN, OUTPUT, LABELS, path, relative,
                                               read, write, sha, require, load_completed)

METHODS = ("transolver", "fno")
STEP = 4800


def load_array(np, identity, receipt, kind):
    filename = path(identity["path"]); stat = filename.stat()
    require(stat.st_size == identity["bytes"] and stat.st_mtime_ns == identity["mtime_ns"], "Frozen field/input stat differs")
    value = np.load(filename, allow_pickle=False)
    receipt[kind] += 1
    if "shape" in identity:
        require(list(value.shape) == identity["shape"] and str(value.dtype) == identity["dtype"], "Field/input array schema differs")
    require(np.isfinite(value).all(), "Nonfinite completed field illustration")
    if "values_sha256" in identity:
        observed = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        require(observed == identity["values_sha256"], "Stored prediction values changed after evaluation")
    return value


def run(plan_path=PLAN, output=None, max_points=20000, dpi=160):
    began = time.monotonic(); context = load_completed(plan_path)
    destination = path(output or (ROOT / OUTPUT / "fields"))
    require(destination == (ROOT / OUTPUT / "fields").resolve(), "Renderer owns only results/geobc_coverage_comparison/fields")
    require(type(max_points) is int and max_points >= 0 and 60 <= dpi <= 300, "Invalid display budget")
    destination.mkdir(parents=True, exist_ok=True)
    # All metadata gates are satisfied before the first scientific/GT array read.
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    key = context["science"]["DEV_CADs"][0] + "/ver"; cad = key.split("/")[0]
    receipt = {"owned_GT_array_reads": 0, "owned_prediction_array_reads": 0, "owned_input_array_reads": 0,
               "raw_source_or_label_parses": 0, "weight_reads": 0, "model_or_Torch_imports": 0, "GPU_calls": 0}
    record = context["inputs"]["records"][key]
    xyz = load_array(np, record["file_identities"]["xyz_path"], receipt, "owned_input_array_reads")
    truth = load_array(np, context["labels"]["records"][key]["file_identities"]["physical_path"], receipt, "owned_GT_array_reads")
    require(xyz.ndim == 2 and xyz.shape[1] == 3 and truth.shape == (len(xyz), 4), "Aligned field/coordinate population differs")
    norm = context["inputs"]["CADs"][cad]["normalization"]
    xyz_mm = xyz.astype(np.float64) * float(norm["characteristic_length_mm"]) + np.asarray(norm["bbox_min_mm"], np.float64)
    n = len(xyz); display_n = n if max_points == 0 else min(n, max_points)
    shown = np.linspace(0, n - 1, display_n, dtype=np.int64)
    xyz_show = xyz_mm[shown]; result = []
    for method in METHODS:
        arm = next(a for a in context["science"]["arm_order"] if a["method"] == method and a["train_CADs"] == 200)
        token = f"{arm['id']}:{STEP}:{key}"; ref = context["proof"]["prediction_manifests"][token]
        require(sha(ref["path"]) == ref["sha256"], "Committed field manifest changed")
        prediction_manifest = read(ref["path"])
        require(prediction_manifest["arm"] == arm["id"] and prediction_manifest["key"] == key
                and prediction_manifest["step"] == STEP and prediction_manifest["terminal"] is True, "Fixed representative identity differs")
        if prediction_manifest["prediction_complete"] is not True:
            result.append({"arm": arm["id"], "status": "unavailable_fixed_endpoint", "reason": prediction_manifest["reason"]})
            continue
        prediction = load_array(np, prediction_manifest["physical"], receipt, "owned_prediction_array_reads")
        require(prediction.shape == truth.shape, "FullN representative shape differs")
        stress_limit = max(float(np.abs(truth[:, 3]).max()), float(np.abs(prediction[:, 3]).max()), 1e-30)
        s_error = prediction[:, 3] - truth[:, 3]; error_limit = max(float(np.abs(s_error).max()), 1e-30)
        u_true, u_pred = np.linalg.norm(truth[:, :3], axis=1), np.linalg.norm(prediction[:, :3], axis=1)
        u_error = np.linalg.norm(prediction[:, :3] - truth[:, :3], axis=1)
        u_limit = max(float(u_true.max()), float(u_pred.max()), 1e-30)
        values = (truth[:, 3], prediction[:, 3], s_error, u_true, u_pred, u_error)
        titles = ("GT signed S [MPa]", "Prediction signed S [MPa]", "Signed S error: pred-GT [MPa]",
                  "GT displacement vector norm [mm]", "Prediction vector norm [mm]", "Displacement vector error norm [mm]")
        normalizers = (TwoSlopeNorm(vmin=-stress_limit, vcenter=0, vmax=stress_limit),) * 2 + (
            TwoSlopeNorm(vmin=-error_limit, vcenter=0, vmax=error_limit),) + (Normalize(0, u_limit),) * 2 + (
            Normalize(0, max(float(u_error.max()), 1e-30)),)
        fig = plt.figure(figsize=(14, 8.8), constrained_layout=True)
        extents = np.ptp(xyz_mm, axis=0); extents = np.maximum(extents, float(extents.max()) * .02)
        for index, (value, title, normalizer) in enumerate(zip(values, titles, normalizers)):
            ax = fig.add_subplot(2, 3, index + 1, projection="3d")
            scatter = ax.scatter(*xyz_show.T, c=value[shown], cmap="coolwarm" if index < 3 else "viridis",
                                 norm=normalizer, s=1.5, linewidths=0, rasterized=True)
            ax.set_title(title, fontsize=10); ax.set_box_aspect(extents); ax.view_init(elev=24, azim=-65)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
            fig.colorbar(scatter, ax=ax, shrink=.65, pad=.04)
        display = "all nodes" if display_n == n else f"deterministic node-index display sample {display_n}/{n}"
        fig.suptitle(f"{LABELS[method]} / TRAIN200 / fixed4800 / {key}\n{display}; reported metrics use all original nodes", fontsize=12)
        target = destination / (method + "_200_step4800_" + key.replace("/", "_") + ".png")
        fig.savefig(target, dpi=dpi); plt.close(fig)
        full_metrics = context["summary"]["arms"][arm["id"]]["primary"]["validation"]["per_case"][key]
        result.append({"arm": arm["id"], "status": "completed", "path": relative(target), "sha256": sha(target),
            "sample_key": key, "step": STEP, "population_nodes": n, "display_nodes": display_n,
            "display_sampling": "np.linspace over original H5 row indices, not field values, errors or metrics",
            "full_node_metrics": full_metrics, "metrics_recomputed_on_display_subset": False,
            "units": {"signed_stress": "MPa", "U_vector_norm": "mm", "U_vector_error_norm": "mm"},
            "source_prediction_manifest": ref, "coordinate_display": "normalized original-row xyz restored by frozen bbox/length metadata"})
        del prediction
    manifest = {"status": "completed_fixed_illustration_inventory", "scientific_digest": context["plan"]["scientific_digest"],
        "case": key, "step": STEP, "methods": list(METHODS), "train_CADs": 200, "illustrations": result,
        "selection": "Fixed before actual readout; unavailable representatives retained, no replacement or best-case selection",
        "read_receipt": receipt, "elapsed_seconds": time.monotonic() - began,
        "source_sha256": {"tools/plot_geobc_coverage_fields.py": sha(__file__), "tools/export_geobc_coverage_results.py": sha(ROOT / "tools/export_geobc_coverage_results.py")}}
    write(destination / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "case": key, "rendered": sum(r["status"] == "completed" for r in result),
                      "output": str(destination)}, ensure_ascii=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN); parser.add_argument("--output")
    parser.add_argument("--max-points", type=int, default=20000, help="Display-only deterministic row sample; 0 displays all original nodes")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(); run(args.plan, args.output, args.max_points, args.dpi)


if __name__ == "__main__":
    main()
