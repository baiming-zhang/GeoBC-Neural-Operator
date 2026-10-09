#!/usr/bin/env python3
"""Plot the first fixed representative of a completed frozen dia load probe.

Uses only exported xyz/sample_key/u arrays. Magnitudes are visualization only;
training and reported all-node metrics retain the signed displacement vector.
The representative is fixed by index order, never selected by error.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if os.name == "nt" and (ROOT / ".local_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".local_deps"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator


def strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


def resolve_index(reference: str, run: Path) -> Path:
    candidate = Path(reference)
    if candidate.is_file():
        return candidate
    # Exported references can be saved on WSL while plotting runs on Windows.
    text = reference.replace("\\", "/")
    marker = text.find("runs/")
    if marker >= 0:
        candidate = ROOT / text[marker:]
        if candidate.is_file():
            return candidate
    candidate = run / "predictions" / Path(text).name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Cannot resolve exported representative index: {reference}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/geobc_exploration/frozen_validation_dia_probe")
    parser.add_argument("--output", type=Path, default=ROOT / "results/geobc_exploration/load_transfer.png")
    parser.add_argument("--max-nodes", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.max_nodes <= 8000:
        parser.error("--max-nodes must be between 1 and 8000")
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or summary.get("final_test_fields_opened") != []:
        raise ValueError("Expected a completed frozen validation-only probe with final test unopened")
    config = summary["config"]
    if "signed uXYZ in mm" not in config.get("metric_units", ""):
        raise ValueError("The artifact does not explicitly establish signed displacement units in mm")
    if not any("per-CAD normalized xyz" in text for text in strings(config)):
        raise ValueError("The artifact does not explicitly establish normalized coordinate geometry")
    index_path = resolve_index(summary["representative_index"], args.run)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not index.get("cases"):
        raise ValueError("No fixed exported representative cases")
    case_key, entry = next(iter(index["cases"].items()))
    if not case_key.endswith("/dia"):
        raise ValueError("Expected the fixed dia representative")
    filename = entry["file"].replace("\\", "/")
    if Path(filename).name != filename:
        raise ValueError("Representative file must be local to the export index directory")
    artifact = index_path.parent / filename
    with np.load(artifact, allow_pickle=False) as stored:
        # No stress, oracle, raw source, or final-test arrays are read.
        xyz = stored["xyz"]
        exported_key = str(stored["sample_key"])
        truth = stored["u_true"]
        linear = stored["u_linear_basis"]
        dependent = stored["u_load_dependent_basis"]
    n = len(xyz)
    if exported_key != case_key or n != entry["nodes"] or xyz.shape != (n, 3):
        raise ValueError("Exported case/node metadata mismatch")
    if any(x.shape != (n, 3) or not np.isfinite(x).all() for x in (truth, linear, dependent, xyz)):
        raise ValueError("Expected finite aligned xyz/uXYZ arrays")
    if xyz.min() < -1e-6 or xyz.max() > 1+1e-6:
        raise ValueError("Exported xyz disagrees with the normalized-coordinate declaration")
    stride = max(1, math.ceil(n / args.max_nodes))
    nodes = np.arange(0, n, stride)
    position = xyz[nodes]
    magnitudes = [np.linalg.norm(x, axis=1) for x in (truth, linear, dependent)]
    vector_error = np.linalg.norm(linear-truth, axis=1)
    shared_max = max(float(x.max()) for x in magnitudes)
    error_max = float(vector_error.max())
    shared_norm = Normalize(vmin=0, vmax=max(shared_max, 1e-12))
    error_norm = Normalize(vmin=0, vmax=max(error_max, 1e-12))
    linear_metric = summary["displacement"]["linear_basis"]["per_case"][case_key]
    dependent_metric = summary["displacement"]["load_dependent_basis"]["per_case"][case_key]
    if linear_metric["nodes"] != n or dependent_metric["nodes"] != n:
        raise ValueError("All-node metrics and representative node count disagree")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    figure = plt.figure(figsize=(14.8, 5.1), facecolor="white")
    grid = figure.add_gridspec(2, 4, height_ratios=(1, .055), left=.025, right=.985,
                              bottom=.19, top=.79, wspace=.04, hspace=.055)
    titles = ("FEM displacement", "Linear basis prediction", "Dependent basis prediction", "Linear basis vector error")
    fields = magnitudes + [vector_error]
    lower, upper = xyz.min(axis=0), xyz.max(axis=0)
    extent = upper-lower
    if np.any(extent <= 0):
        raise ValueError("Invalid three-dimensional coordinate extent")
    scatters = []
    for i, (title, field) in enumerate(zip(titles, fields)):
        axis = figure.add_subplot(grid[0, i], projection="3d")
        axis.set_proj_type("ortho")
        scatter = axis.scatter(position[:, 0], position[:, 1], position[:, 2], c=field[nodes],
                               cmap="viridis" if i < 3 else "magma", norm=shared_norm if i < 3 else error_norm,
                               s=2.3, linewidths=0, alpha=1, depthshade=False, rasterized=True)
        scatters.append(scatter)
        axis.view_init(elev=25, azim=-58)
        axis.set_box_aspect(extent)
        pad = .035*extent
        axis.set_xlim(lower[0]-pad[0], upper[0]+pad[0])
        axis.set_ylim(lower[1]-pad[1], upper[1]+pad[1])
        axis.set_zlim(lower[2]-pad[2], upper[2]+pad[2])
        axis.set_xlabel("x", labelpad=-1)
        axis.set_ylabel("y", labelpad=-1)
        axis.set_zlabel("z", labelpad=-1)
        axis.tick_params(labelsize=7, pad=-1)
        for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
            coordinate_axis.set_major_locator(MaxNLocator(nbins=3))
            coordinate_axis.pane.fill = False
            coordinate_axis.pane.set_edgecolor("#dddddd")
        axis.grid(False)
        axis.set_title(title, fontsize=10, pad=2)
    shared_bar = figure.colorbar(scatters[0], cax=figure.add_subplot(grid[1, :3]), orientation="horizontal")
    shared_bar.set_label(r"Displacement magnitude $|\mathbf{u}|$ [mm]", fontsize=9)
    error_bar = figure.colorbar(scatters[3], cax=figure.add_subplot(grid[1, 3]), orientation="horizontal")
    error_bar.set_label(r"$\|\mathbf{u}_{linear}-\mathbf{u}_{FEM}\|_2$ [mm]", fontsize=9)
    for bar in (shared_bar, error_bar):
        bar.locator = MaxNLocator(nbins=4)
        bar.update_ticks()
        bar.ax.tick_params(labelsize=8)
    figure.suptitle(f"{case_key}  |  Frozen load-transfer probe", fontsize=14, fontweight="bold", y=.967)
    figure.text(.5, .891, f"All-node vector relative L2: linear {100*linear_metric['relative_l2']:.2f}%  |  "
                f"dependent {100*dependent_metric['relative_l2']:.2f}%  |  N = {n:,}", ha="center", fontsize=10)
    caption = (f"Fixed first representative; {len(nodes):,} / {n:,} original nodes shown (stride {stride}). "
               "Same nodes, view, and magnitude color scale in the first three panels.\n"
               "Coordinates: per-CAD normalized xyz (unitless). Magnitudes are visualization only; training/evaluation use signed uXYZ. "
               "This case does not summarize the five-CAD aggregate.")
    figure.text(.025, .045, caption, ha="left", va="bottom", fontsize=8.1, color="#444444", linespacing=1.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=190, facecolor="white", metadata={"Description": caption,
                   "Source": str(args.run / "summary.json"), "Case": case_key})
    plt.close(figure)
    print(json.dumps({"output": str(args.output), "case": case_key, "original_nodes": n,
                      "displayed_nodes": len(nodes), "stride": stride, "coordinate_units": "normalized unitless xyz",
                      "displacement_units": "mm", "linear_all_node_relative_l2": linear_metric["relative_l2"],
                      "dependent_all_node_relative_l2": dependent_metric["relative_l2"],
                      "magnitude_color_max_mm": shared_max, "error_color_max_mm": error_max}))


if __name__ == "__main__":
    main()
