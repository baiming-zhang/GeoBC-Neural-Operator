"""Plot the completed fresh split-native Hash modulation pair from its analysis note."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"zero": "#507eaa", "actual": "#d78c59"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, default=ROOT / "notes/geobc_hash_modulation_pair_result.json")
    result.add_argument("--output", type=Path, default=ROOT / "results/geobc_exploration/hash_modulation_comparison.png")
    return result


def _path_record(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def main() -> None:
    args = parser().parse_args()
    raw = args.input.read_bytes()
    note = json.loads(raw)
    if not (note.get("status") == "completed" and note.get("matched_controls_passed") is True
            and note.get("pair_controls_passed") is True and note.get("fresh_split_pair") is True
            and note.get("historical_control_reused") is False):
        raise ValueError("The figure requires a completed, verified fresh split-native matched pair.")
    runs = {mode: note["runs"][mode] for mode in ("zero", "actual")}
    case_counts = {mode: runs[mode]["case_counts"]["validation"] for mode in runs}
    node_counts = {mode: runs[mode]["all_node_counts"]["validation"] for mode in runs}
    if set(case_counts.values()) != {15} or len(set(node_counts.values())) != 1:
        raise ValueError("Both branches must use the same 15-case known-load validation scope.")
    fields = ("validation_u_mean_relative_l2", "validation_u_worst_relative_l2", "validation_P1_mean_relative_l2")
    percentages = {}
    for mode, run in runs.items():
        values = {field: 100 * float(run[field]) for field in fields}
        if not all(math.isfinite(value) and value >= 0 for value in values.values()):
            raise ValueError("Cannot plot nonfinite or negative relative errors.")
        percentages[mode] = values

    # No Torch, model, checkpoint, or raw-field loading is needed to render this figure.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 10, "axes.labelsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), gridspec_kw={"wspace": 0.32})
    fig.subplots_adjust(left=0.085, right=0.985, top=0.77, bottom=0.27)
    fig.suptitle("Hash feature modulation: fresh matched split pair", y=0.96, fontsize=11)
    fig.text(0.5, 0.89, "Known-load validation: 15 cases; lower is better", ha="center", fontsize=9)

    ax = axes[0]
    width = 0.32
    for mode, offset, label in (("zero", -width / 2, "Inactive (zero)"), ("actual", width / 2, "Active (actual)")):
        values = [percentages[mode][field] for field in fields[:2]]
        bars = ax.bar([offset, 1 + offset], values, width, color=COLORS[mode], label=label, zorder=3)
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4, fontsize=9)
    ax.set_xticks([0, 1], ["Case mean", "Worst case"])
    ax.set_ylim(0, max(percentages[mode][fields[1]] for mode in runs) * 1.24)
    ax.set_ylabel("Displacement relative L2 (%)")
    ax.set_title("Physical displacement vector", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    values = [percentages[mode][fields[2]] for mode in runs]
    bars = ax.bar([0, 1], values, 0.55, color=[COLORS[mode] for mode in runs], zorder=3)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4, fontsize=9)
    ax.set_xticks([0, 1], ["Inactive (zero)", "Active (actual)"])
    ax.set_ylim(0, max(100, max(values)) * 1.18)
    ax.set_ylabel("P1 strain relative L2 (%)")
    ax.set_title("All-node P1 tensor: case mean", pad=10)
    ax.axhline(100, color="#555555", linestyle="--", linewidth=1, zorder=4)
    ax.text(0.98, 118, "Zero-strain prediction: 100%", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="#444444",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5}, zorder=5)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        ax.tick_params(axis="x", length=0)

    steps = "/".join(str(runs[mode]["best_step"]) for mode in runs)
    node_count = next(iter(node_counts.values()))
    fig.text(0.5, 0.16, f"Selected steps (zero/actual): {steps}; same u-selected checkpoints for P1.",
             ha="center", fontsize=8)
    fig.text(0.5, 0.105, f"All {node_count:,} validation nodes; unweighted case means. P1 is a recovered symmetric-tensor proxy.",
             ha="center", fontsize=8)
    fig.text(0.5, 0.05, "Single paired seed; no error bars. Historical models excluded; DIA and final test unused.",
             ha="center", fontsize=8, color="#555555")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, facecolor="white")
    plt.close(fig)

    metadata = {
        "status": "rendered",
        "metric_source": {"path": _path_record(args.input), "sha256": hashlib.sha256(raw).hexdigest()},
        "figure": {"path": _path_record(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                   "size_inches": [8.0, 3.8], "dpi": 300, "zero_based_y_axes": True, "error_bars": False},
        "source_stage": note["stage"], "matched_controls_passed": True,
        "selection": {"criterion": "known-load validation case-mean physical displacement vector relative L2",
                      "best_step": {mode: runs[mode]["best_step"] for mode in runs},
                      "P1_checkpoint": "same checkpoint selected by displacement; not selected by strain"},
        "validation_scope": {"case_count": 15, "all_node_count": node_count,
                             "DIA_used": False, "final_test_used": False},
        "units": {"display": "percent: 100 times dimensionless relative L2",
                  "u": "physical mm vector; ||u_pred-u_true||_2 / ||u_true||_2",
                  "P1": "all-node recovered symmetric tensor Frobenius relative L2, tensor shear weighting 2"},
        "aggregation": {"mean": "unweighted arithmetic mean of the 15 case relative L2 errors",
                        "worst": "maximum among the same 15 case errors"},
        "percentages": percentages, "zero_strain_reference_percent": 100,
        "limits": [
            "Only the new fresh matched split-native zero/actual pair is compared; no historical control reused.",
            "P1 is a corner finite-element, volume-averaged all-node proxy; it is not the sampled cell squared loss.",
            "Single paired seed and repeatedly used development validation; no confidence interval or final-test claim.",
            note["effective_capacity_limit"],
            "Synthetic native acceptance has no physical-performance metric and is not plotted.",
            "Rendering reads only the completed analysis JSON, without raw fields, model weights, Torch, or GPU execution.",
        ],
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(args.output), "metadata": str(metadata_path), "percentages": percentages}))


if __name__ == "__main__":
    main()
