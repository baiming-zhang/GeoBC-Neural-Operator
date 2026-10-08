"""Render at most three completed, aligned suite stress fields on one FEM case.

Default representatives: coordinate MLP, official FullyFused TCNN neural field,
and 192^3 FNO. A pending run is skipped. No model, torch or GPU is loaded.
Original quadratic boundary triangles and scientific drawing helpers are
reused from plot_fits; ground truth and prediction have the same signed scale.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.plot_fits import Fit, _mesh, _metrics, _native_path, _render_fit, _scalar
import numpy as np


DEFAULT_RUNS = ["mlp_single_exact", "tcnn_hash_single_exact", "fno192_stress"]
LABELS = {"mlp_single_exact": "Coordinate MLP (4 x 256 ReLU)",
          "tcnn_hash_single_exact": "Official TCNN FullyFused (12-level hash)",
          "fno192_stress": "FNO 192^3 (stress only)"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value) -> Path:
    native = _native_path(value)
    return (native if native.is_absolute() else ROOT / native).resolve()


def load_fit(folder: Path, source: Path, case: str):
    summary = read_json(folder / "summary.json")
    config = read_json(folder / "config.json")
    best = summary.get("best_metrics") or read_json(folder / "metrics_best.json")
    step = int(summary.get("best_step", best["step"]))
    directory = folder / "predictions_best"
    index = read_json(directory / "index.json")
    index = index.get("cases", index)
    if case not in index:
        raise ValueError(f"{folder.name}: no saved prediction for {case}")
    mesh_key, xyz_mm, faces, source_stress = _mesh(source)
    with np.load(source, allow_pickle=False) as archive:
        normalized_xyz = np.asarray(archive["xyz"], np.float64)
        alignment = str(_scalar(archive, "label_alignment")) if "label_alignment" in archive.files else None
    prediction_file = directory / index[case]["file"]
    with np.load(prediction_file, allow_pickle=False) as archive:
        saved_key = str(_scalar(archive, "sample_key"))
        truth = np.asarray(archive["stress_true"], np.float64).reshape(-1)
        prediction = np.asarray(archive["stress_pred"], np.float64).reshape(-1)
        saved_xyz = np.asarray(archive["xyz"], np.float64)
        if "step" in archive.files and int(_scalar(archive, "step")) != step:
            raise ValueError("Prediction and selected-checkpoint steps differ")
    if mesh_key != case or saved_key != case:
        raise ValueError("Prepared and prediction sample keys differ")
    if truth.shape != source_stress.shape or prediction.shape != truth.shape or saved_xyz.shape != xyz_mm.shape:
        raise ValueError("Prepared/predicted node counts or coordinate shapes differ")
    if not np.isfinite(prediction).all() or not np.isfinite(saved_xyz).all():
        raise ValueError("Nonfinite prediction or coordinates")
    if not np.allclose(truth, source_stress, rtol=2e-6, atol=1e-8):
        raise ValueError("Saved truth differs from aligned original prepared stress")

    # FNO queries use the padded grid cube, rather than the prepared bbox cube.
    # Convert back to physical coordinates before asserting original node order.
    domain = config.get("grid_metadata", {}).get("grid_domain")
    if domain:
        reconstructed = np.asarray(domain["lower_mm"], np.float64) + saved_xyz * float(domain["length_mm"])
        coordinate_error_mm = float(np.max(np.abs(reconstructed - xyz_mm)))
        if coordinate_error_mm > max(float(domain["length_mm"]) * 2e-6, 1e-5):
            raise ValueError("FNO query coordinates do not match original FEM node order")
        coordinate_check = "prediction xyz transformed from saved physical FNO cube and compared to prepared xyz_mm"
    else:
        coordinate_error_mm = None
        if not np.allclose(saved_xyz, normalized_xyz, rtol=0, atol=2e-6):
            raise ValueError("Prediction normalized coordinates do not match prepared node order")
        coordinate_check = "prediction xyz compared directly to prepared normalized xyz"
    metrics = _metrics(truth, prediction)
    expected = best.get("per_case", {}).get(case, {}).get("stress", {}).get("relative_l2")
    if expected is None:
        expected = best.get("nodes", {}).get("stress", {}).get("relative_l2")
    if expected is not None and not np.isclose(metrics["relative_l2"], float(expected), rtol=2e-6, atol=1e-10):
        raise ValueError("Recomputed all-node stress L2 differs from selected checkpoint metrics")
    fit = Fit(case, xyz_mm, faces, truth, prediction, step, metrics, source)
    metadata = {"sample_key": case, "run": str(folder), "prepared_source": str(source),
                "prediction_source": str(prediction_file), "step": step,
                "selection": config.get("checkpoint_selection", config.get("selection")),
                "implementation_details": config.get("implementation_details", {}),
                "precision": config.get("precision"), "label_alignment": alignment,
                "coordinate_order_check": coordinate_check,
                "maximum_coordinate_error_mm": coordinate_error_mm,
                "surface_triangles": len(faces), **metrics}
    return fit, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="Plan run names or their output directories; at most three")
    parser.add_argument("--case", default="506_533/ver")
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/prepared_exact_multifield")
    parser.add_argument("--output", type=Path, default=ROOT / "plots/all_methods/fields")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--elev", type=float, default=24)
    parser.add_argument("--azim", type=float, default=-65)
    args = parser.parse_args()
    if len(args.runs) > 3:
        parser.error("This representative renderer accepts at most three runs")
    began = time.monotonic()
    plan = read_json(ROOT / "notes/all_methods_plan.json")
    jobs = {job["name"]: job for job in plan["jobs"]}
    allowed_paths = {resolve_path(job["output"]): job["name"] for job in jobs.values()}
    prepared = resolve_path(args.prepared)
    source = prepared / (args.case.replace("/", "_") + ".npz") if prepared.is_dir() else prepared
    if not source.exists():
        raise FileNotFoundError(source)
    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rendered, skipped = [], []
    for requested in args.runs:
        if requested in jobs:
            folder = resolve_path(jobs[requested]["output"])
            name = requested
        else:
            folder = resolve_path(requested)
            if folder not in allowed_paths:
                raise ValueError(f"Run outside the current valid suite plan: {requested}")
            name = allowed_paths[folder]
        summary_file = folder / "summary.json"
        if not summary_file.exists() or read_json(summary_file).get("status") != "completed":
            skipped.append({"run": name, "reason": "pending or incomplete; no completed summary.json"})
            continue
        fit, metadata = load_fit(folder, source, args.case)
        safe_key = re.sub(r"[^A-Za-z0-9_.-]", "_", args.case)
        target = output / f"{name}__{safe_key}.png"
        label = LABELS.get(name, name)
        _render_fit(fit, label, target, args.dpi, args.elev, args.azim)
        rendered.append({"label": label, "figure": str(target), **metadata})
        print(f"Rendered {name}: stress L2={100 * fit.metrics['relative_l2']:.6f}%", flush=True)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": "original DeepJEB v1.0",
              "scope": "selected training-case fitting only; no generalization claim", "selection": "best checkpoint from completed run",
              "display": "original quadratic FEM surface; opaque depth-sorted face means; shared GT/pred signed coolwarm scale",
              "error": "mean nodal absolute residual / all-original-node true RMS * 100; full color range; no clipping",
              "metrics": "all original nodes, including interior nodes; hotspot = ceil(1% N) highest absolute true stress",
              "view": {"elevation": args.elev, "azimuth": args.azim}, "rendered": rendered, "skipped": skipped,
              "elapsed_seconds": time.monotonic() - began}
    (output / "field_plot_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"figures": [r["figure"] for r in rendered], "skipped": skipped,
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
