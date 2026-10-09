"""Plot only the completed matched full12/coarse8 analysis JSON."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ("full12", "coarse8")
LABELS = ("Full 12 levels", "First 8 levels")
COLORS = ("#507eaa", "#d78c59")


def main():
    source = ROOT / "notes/geobc_hash_bandwidth_pair_result.json"
    raw = source.read_bytes()
    note = json.loads(raw)
    if not (note.get("status") == "completed" and note.get("matched_controls_passed") is True
            and note.get("pair_controls_passed") is True):
        raise ValueError("Completed verified bandwidth pair required")
    runs = {mode: note["runs"][mode] for mode in MODES}
    if {run["case_counts"]["validation"] for run in runs.values()} != {15}:
        raise ValueError("Both branches must use the same 15 known-load cases")
    nodes = {run["all_node_counts"]["validation"] for run in runs.values()}
    if len(nodes) != 1:
        raise ValueError("Validation node scopes differ")
    fields = ("validation_u_mean_relative_l2", "validation_u_worst_relative_l2", "validation_P1_mean_relative_l2")
    values = {mode: [100 * float(runs[mode][key]) for key in fields] for mode in MODES}
    if not all(math.isfinite(v) and v >= 0 for row in values.values() for v in row):
        raise ValueError("Invalid physical relative error")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"wspace": .33})
    fig.subplots_adjust(left=.09, right=.985, top=.76, bottom=.29)
    fig.suptitle("Fixed Hash feature mask: full 12 versus first 8 levels", y=.96, fontsize=11)
    fig.text(.5, .89, "Same inputs, stored parameters, loss and budget; lower is better", ha="center")
    width = .32
    for index, mode in enumerate(MODES):
        offset = (index - .5) * width
        bars = axes[0].bar([offset, 1 + offset], values[mode][:2], width,
                           color=COLORS[index], label=LABELS[index], zorder=3)
        axes[0].bar_label(bars, labels=[f"{v:.2f}" for v in values[mode][:2]], padding=4)
    axes[0].set_xticks([0, 1], ["Case mean", "Worst case"])
    axes[0].set_ylim(0, max(v for row in values.values() for v in row[:2]) * 1.24)
    axes[0].set_ylabel("Displacement relative L2 (%)")
    axes[0].set_title("Physical displacement vector", pad=10)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    strain = [values[mode][2] for mode in MODES]
    bars = axes[1].bar([0, 1], strain, .55, color=COLORS, zorder=3)
    axes[1].bar_label(bars, labels=[f"{v:.2f}" for v in strain], padding=4)
    axes[1].set_xticks([0, 1], LABELS)
    axes[1].set_ylim(0, max(100, *strain) * 1.25)
    axes[1].set_ylabel("P1 strain relative L2 (%)")
    axes[1].set_title("All-node P1 tensor: case mean", pad=10)
    axes[1].axhline(100, color="#555555", linestyle="--", linewidth=1)
    axes[1].annotate("Zero-strain prediction: 100%", (.98, 100),
                     xycoords=axes[1].get_yaxis_transform(), xytext=(0, 4),
                     textcoords="offset points", ha="right", fontsize=8,
                     bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5e5e5", linewidth=.6)
        ax.tick_params(axis="x", length=0)
    steps = "/".join(str(runs[mode]["best_step"]) for mode in MODES)
    fig.text(.5, .19, f"Selected steps (full12/coarse8): {steps}; P1 at the same u-selected checkpoints.", ha="center", fontsize=8)
    fig.text(.5, .14, f"All {next(iter(nodes)):,} validation nodes; recovered P1 proxy, not solver strain GT.", ha="center", fontsize=8)
    fig.text(.5, .09, "Coarse8 reduces effective spatial capacity; this does not isolate frequency causality.", ha="center", fontsize=8)
    fig.text(.5, .04, "Single paired seed; no error bars; fresh split-native control; DIA/final test unused.", ha="center", fontsize=8, color="#555555")
    target = ROOT / "results/geobc_exploration/hash_bandwidth_comparison.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=300, facecolor="white")
    plt.close(fig)
    metadata = {"status": "rendered",
        "source": {"path": source.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()},
        "figure": {"path": target.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "dpi": 300, "size_inches": [8, 4]},
        "modes": list(MODES), "percentages": values, "metric_order": list(fields),
        "matched_controls_passed": True, "validation_cases": 15, "validation_nodes": next(iter(nodes)),
        "selection": "Case-mean all-node physical u; P1 at same selected checkpoint",
        "best_steps": {mode: runs[mode]["best_step"] for mode in MODES},
        "aggregation": "Unweighted 15-case mean or maximum over the same 15 cases",
        "units": "Percent=100*relativeL2; physical u in mm; P1 Frobenius squared-shear weight2",
        "zero_based_y_axes": True, "zero_strain_reference_percent": 100, "error_bars": False,
        "limits": ["One seed and reused development validation, not final-test performance.",
                   "Effective capacity and optimization paths differ; no isolated frequency cause.",
                   "Both compute all 12 Encoding levels; no lookup speedup claim.",
                   "P1 proxy is not sampled cell squared loss or solver GT strain.",
                   "JSON-only CPU plot; no data arrays, checkpoints or GPU."]}
    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"figure": str(target), "percentages": values}))


if __name__ == "__main__":
    main()
