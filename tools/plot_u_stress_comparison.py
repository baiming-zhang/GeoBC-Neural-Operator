#!/usr/bin/env python3
"""Plot the three completed u3 stress-validation metrics from one JSON receipt."""
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = (
    ("validation_mean_relative_l2", "Case mean"),
    ("validation_worst_relative_l2", "Worst case"),
    ("validation_hotspot_mean_relative_l2", "Hotspot mean"),
)


def plot(source: Path, output: Path) -> dict:
    receipt = json.loads(source.read_text(encoding="utf-8"))
    if receipt.get("status") != "completed" or receipt.get("pair_control_checks_passed") is not True:
        raise ValueError("Require a completed pair with matched control checks passed")
    if receipt.get("final_test_and_dia_opened") is not False:
        raise ValueError("This plot is restricted to the completed known-load validation pair")
    runs = receipt["runs"]
    values = {
        variant: np.asarray([100.0 * runs[variant][key] for key, _ in METRICS], dtype=float)
        for variant in ("zero", "actual")
    }
    if any(not np.isfinite(v).all() or (v < 0).any() for v in values.values()):
        raise ValueError("Require finite nonnegative physical L2 metrics")
    steps = {variant: int(runs[variant]["best_step"]) for variant in values}
    x = np.arange(len(METRICS))
    width = 0.31
    colors = {"zero": "#637b8b", "actual": "#008f96"}
    labels = {"zero": "Zero u3 slots", "actual": "Frozen predicted u3"}

    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.axisbelow": True,
    }):
        fig, ax = plt.subplots(figsize=(8.6, 5.1), dpi=180)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.79, bottom=0.26)
        for variant, offset in (("zero", -width / 2), ("actual", width / 2)):
            bars = ax.bar(x + offset, values[variant], width=width,
                          color=colors[variant], label=labels[variant], zorder=3)
            ax.bar_label(bars, labels=[f"{value:.2f}" for value in values[variant]],
                         padding=4, fontsize=11)
        maximum = max(float(v.max()) for v in values.values())
        upper = max(100.0, 10 * math.ceil((maximum + 12) / 10))
        ax.set_ylim(0, upper)
        ax.set_ylabel("Physical signed-stress relative L2 [%]")
        ax.set_xticks(x, [label for _, label in METRICS])
        ax.set_yticks(np.arange(0, upper + 1, 20))
        ax.grid(axis="y", color="#dfe4e7", linewidth=0.7)
        ax.spines["left"].set_color("#83919b")
        ax.spines["bottom"].set_color("#83919b")
        ax.tick_params(axis="x", length=0, pad=9)
        fig.suptitle("Frozen displacement conditioning: stress validation",
                     x=0.10, y=0.97, ha="left", fontsize=15, fontweight="bold")
        fig.text(0.10, 0.91, "Lower is better", fontsize=11, color="#4d5a63")
        ax.legend(loc="lower left", bbox_to_anchor=(-0.006, 1.005),
                  ncol=2, frameon=False, handlelength=1.3, columnspacing=2.0)
        caption = (
            "Known-load development validation: 15 cases, all original FEM nodes.\n"
            "Each checkpoint selected by minimum case-mean L2; "
            f"zero best step {steps['zero']:,}, actual best step {steps['actual']:,}.\n"
            "Descriptive paired results; no statistical significance inference."
        )
        fig.text(0.10, 0.14, caption, va="top", ha="left", fontsize=9.4,
                 color="#4d5a63", linespacing=1.45)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, facecolor="white", metadata={
            "Description": caption,
            "Source": str(source),
        })
        plt.close(fig)
    return {
        "source": str(source), "output": str(output),
        "metrics_percent": {variant: v.tolist() for variant, v in values.items()},
        "metric_order": [key for key, _ in METRICS], "best_steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "notes/geobc_u_stress_pair_result.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/geobc_exploration/u_stress_comparison.png")
    args = parser.parse_args()
    print(json.dumps(plot(args.source.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
