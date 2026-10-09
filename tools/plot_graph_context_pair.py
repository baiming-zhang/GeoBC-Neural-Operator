"""Render a completed matched boundary-distance pair from small analysis JSON."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ("euclidean", "graph")
LABELS = ("Euclidean", "Mesh-edge path")
COLORS = ("#507eaa", "#d78c59")


def main():
    source = ROOT / "notes/geobc_graph_context_pair_result.json"
    raw = source.read_bytes()
    note = json.loads(raw)
    if not (note.get("status") == "completed" and note.get("matched_controls_passed") is True
            and note.get("pair_controls_passed") is True):
        raise ValueError("A completed verified matched distance pair is required")
    runs = {mode: note["runs"][mode] for mode in MODES}
    if {run["case_counts"]["validation"] for run in runs.values()} != {15}:
        raise ValueError("Both branches must evaluate the same15 known-load validation cases")
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
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), gridspec_kw={"wspace": .33})
    fig.subplots_adjust(left=.09, right=.985, top=.77, bottom=.27)
    fig.suptitle("Same boundary interfaces: Euclidean versus mesh-edge distance", y=.96, fontsize=11)
    fig.text(.5, .89, "Both distance5 inputs active; same fresh model capacity; lower is better", ha="center")
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
    axes[1].set_ylim(0, max(100, *strain) * 1.18)
    axes[1].set_ylabel("P1 strain relative L2 (%)")
    axes[1].set_title("All-node P1 tensor: case mean", pad=10)
    axes[1].axhline(100, color="#555555", linestyle="--", linewidth=1)
    axes[1].text(.98, 118, "Zero-strain prediction: 100%", transform=axes[1].get_yaxis_transform(),
                 ha="right", fontsize=8, color="#444444", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5e5e5", linewidth=.6)
        ax.tick_params(axis="x", length=0)
    steps = "/".join(str(runs[mode]["best_step"]) for mode in MODES)
    fig.text(.5, .16, f"Selected steps (Euclidean/graph): {steps}; P1 uses the same u-selected checkpoints.", ha="center", fontsize=8)
    fig.text(.5, .105, f"All {next(iter(nodes)):,} validation nodes; case means; recovered P1 tensor proxy, not solver strain GT.", ha="center", fontsize=8)
    fig.text(.5, .05, "Single paired seed, no error bars; no historical control; DIA and final test unused.", ha="center", fontsize=8, color="#555555")
    target = ROOT / "results/geobc_exploration/graph_context_comparison.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=300, facecolor="white")
    plt.close(fig)
    metadata = {
        "status": "rendered", "source": {"path": source.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()},
        "figure": {"path": target.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "dpi": 300, "size_inches": [8, 3.8]},
        "modes": list(MODES), "percentages": values, "metric_order": list(fields),
        "matched_controls_passed": True, "both_distance_channels_active": 5,
        "validation_cases": 15, "validation_nodes": next(iter(nodes)),
        "selection": "Case-mean all-node physical u relativeL2; P1 at same selected checkpoint",
        "best_steps": {mode: runs[mode]["best_step"] for mode in MODES},
        "aggregation": "Unweighted15-case mean or maximum among same15 cases",
        "units": "Percent=100*dimensionless relativeL2; physical u inmm, symmetric P1 tensor with doubled shear energy",
        "zero_based_y_axes": True, "zero_strain_reference_percent": 100, "error_bars": False,
        "limits": ["Single seed and reused development validation, not final-test generalization.",
                   "Fixed mesh edge distances are not stiffness, exact continuous geodesics or remeshing invariants.",
                   "Different metric distributions also change conditioning; representation utility is not pure topology causality.",
                   "P1 all-node proxy is not sampled cell squared loss or solver strain GT.",
                   "JSON-only CPU plotting; no dataset, checkpoint or GPU access."]}
    target.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"figure": str(target), "percentages": values}))


if __name__ == "__main__":
    main()
