"""Plot completed TRAIN120 endpoint scalar metrics; never load fields or weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("transolver", "transolver_pp", "ts3", "fno", "mlp")
LABELS = {"transolver": "Transolver", "transolver_pp": "Transolver++",
          "ts3": "Transolver-3", "fno": "FNO", "mlp": "MLP"}


def project_path(value):
    filename = Path(value)
    return filename if filename.is_absolute() else ROOT / filename


def run(summary="results/geobc_train_fit/summary.json",
        output="results/geobc_train_fit/comparison.png", save_pdf=True):
    began = time.perf_counter()
    filename = project_path(summary)
    raw = filename.read_bytes()
    record = json.loads(raw.decode("utf-8-sig"))
    if not (record.get("status") == "completed"
            and record.get("kind") == "coverage_TRAIN120_fixed4800_fit_diagnostic"
            and record.get("completed_predictions") == 600
            and record.get("fixed_case_denominator_each_arm") == 120):
        raise ValueError("This plot requires the completed five-arm TRAIN120 diagnostic")
    arms = {item["method"]: item for item in record["arms"].values()}
    if set(arms) != set(METHODS):
        raise ValueError("The five fixed methods are required")
    values = []
    for method in METHODS:
        arm = arms[method]; metrics = arm["aggregate"]
        if not (arm["checkpoint_step"] == 4800 and metrics["fixed_denominator"] == 120
                and metrics["metrics_available_cases"] == 120 and metrics["all_cases_complete"] is True):
            raise ValueError("No partial endpoint may be presented as a full TRAIN120 mean")
        u = metrics["case_mean_u_relative_l2"] * 100
        stress = metrics["case_mean_stress_relative_l2"] * 100
        minutes = arm["original_training_costs"]["train_seconds"] / 60
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in (u, stress, minutes)):
            raise ValueError("Plot scalars must be finite nonnegative values")
        values.append({"method": method, "U_mean_percent": u,
                       "signed_S_mean_percent": stress, "original_training_minutes": minutes})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12, "axes.labelsize": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, (error_axis, time_axis) = plt.subplots(
        1, 2, figsize=(12.6, 6.0), gridspec_kw={"width_ratios": [3.2, 1.05], "wspace": .14}, sharey=True)
    y = list(range(5)); height = .29; offset = .17
    u_values = [item["U_mean_percent"] for item in values]
    s_values = [item["signed_S_mean_percent"] for item in values]
    times = [item["original_training_minutes"] for item in values]
    error_axis.barh([i - offset for i in y], u_values, height=height,
                    color="#2C78AB", label="U vector [mm]")
    error_axis.barh([i + offset for i in y], s_values, height=height,
                    color="#DE7A35", label="Signed S [MPa]")
    error_axis.set_yticks(y, [LABELS[method] for method in METHODS])
    error_axis.set_ylim(-.65, 4.65); error_axis.invert_yaxis()
    error_axis.set_xlim(0, max(100., max(u_values + s_values) * 1.12))
    error_axis.xaxis.set_major_locator(MultipleLocator(20))
    error_axis.set_xlabel("Mean per-case relative L2 error [%]")
    error_axis.set_title("Accuracy at the fixed 4800-update endpoint", loc="left", pad=13)
    handles, legend_labels = error_axis.get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper left", frameon=False, ncols=2,
                  bbox_to_anchor=(.108, .89), fontsize=10)
    for position, u, stress in zip(y, u_values, s_values):
        error_axis.text(u + 1.1, position - offset, f"{u:.2f}%", va="center", fontsize=10, color="#225D85")
        error_axis.text(stress + 1.1, position + offset, f"{stress:.2f}%", va="center", fontsize=10, color="#9C481E")

    time_axis.barh(y, times, height=.54, color="#6B7887")
    time_axis.set_xlim(0, max(times) * 1.31)
    time_axis.set_xlabel("Original training time [min]")
    time_axis.set_title("Recorded update clock", loc="left", pad=13)
    time_axis.tick_params(axis="y", left=False, labelleft=False)
    for position, minutes in zip(y, times):
        time_axis.text(minutes + max(times) * .035, position, f"{minutes:.2f}", va="center", fontsize=10, color="#344454")
    for axis in (error_axis, time_axis):
        axis.set_axisbelow(True); axis.grid(axis="x", color="#DFE4EA", linewidth=.8)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color("#BAC4CF")
        axis.tick_params(axis="y", length=0)

    figure.suptitle("TRAIN-only fit diagnostic | original DeepJEB", x=.115, y=.977,
                   ha="left", fontsize=17, weight="semibold", color="#223244")
    figure.text(.115, .921, "40 training CADs × 3 loads · all 120 cases · full original nodes · fixed seed 2026",
                ha="left", fontsize=11, color="#586878")
    figure.text(.115, .058, "U and S are separate physical outputs; relative errors are dimensionless. No DEV results or confidence intervals.",
                ha="left", fontsize=9.4, color="#586878")
    figure.text(.115, .027, "Time is the original training-update clock; initialization, checkpoints and diagnostic readout are excluded.",
                ha="left", fontsize=9.4, color="#586878")
    figure.subplots_adjust(left=.115, right=.985, top=.79, bottom=.15)
    destination = project_path(output); destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, facecolor="white")
    outputs = [str(destination)]
    if save_pdf:
        pdf = destination.with_suffix(".pdf")
        figure.savefig(pdf, facecolor="white"); outputs.append(str(pdf))
    plt.close(figure)
    elapsed = time.perf_counter() - began
    metadata = {"kind": "TRAIN120_fixed4800_scalar_comparison_plot", "summary": str(filename),
                "summary_sha256": hashlib.sha256(raw).hexdigest(), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scalar_values": values, "outputs": outputs, "plot_generation_seconds": elapsed,
                "scope": "TRAIN-only fixed4800, five methods x120 cases; no formal DEV ranking",
                "source_fields": ["arms.*.aggregate.case_mean_{u,stress}_relative_l2", "arms.*.original_training_costs.train_seconds"],
                "raw_arrays_GT_weights_network_GPU_reads": 0, "confidence_intervals": False,
                "architecture_causal_claim": False}
    destination.with_name("comparison_plot_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/geobc_train_fit/summary.json")
    parser.add_argument("--output", default="results/geobc_train_fit/comparison.png")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args(); value = run(args.summary, args.output, not args.no_pdf)
    print(json.dumps({"outputs": value["outputs"], "generation_seconds": value["plot_generation_seconds"],
                      "source_sha256": value["source_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
