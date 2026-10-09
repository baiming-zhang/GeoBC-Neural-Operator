"""Plot one fixed training-CAD value/derivative diagnostic; JSON only."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    source = ROOT / "notes/geobc_fixed_train_strain_quality.json"
    record = json.loads(source.read_text(encoding="utf8"))
    if record["status"] != "completed" or record["CAD"] != "101_479":
        raise ValueError("Only the completed predeclared training-CAD diagnosis is plotted")
    loads = record["loads"]
    u = [100 * record["metrics"][load]["u"]["vector_relative_l2"] for load in loads]
    strain = [100 * record["metrics"][load]["strain_reference_P1_from_training_u"]
              ["symmetric_Frobenius_relative_l2"] for load in loads]
    x = np.arange(len(loads))
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    ax.bar(x - .19, u, .38, label="Displacement", color="#477EA8")
    ax.bar(x + .19, strain, .38, label="Recovered P1 strain", color="#CE7954")
    for delta, values in ((-.19, u), (.19, strain)):
        for index, value in enumerate(values):
            ax.text(index + delta, value * 1.14, f"{value:.1f}%", ha="center", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(5, 2300)
    ax.set_yticks([10, 100, 1000], ["10", "100", "1000"])
    ax.set_xticks(x, loads)
    ax.set_ylabel("Relative L2 / Frobenius error (%) — log scale")
    ax.set_title("Fixed training CAD 101_479: field values versus derivatives")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2, frameon=False)
    ax.grid(axis="y", which="major", alpha=.18)
    ax.set_axisbelow(True)
    fig.text(.1, .035, "Same 179,316 original nodes. Reference strain is derived P1 from training u;\n"
             "not a solver tensor label. One CAD diagnostic; no validation or test fields.", fontsize=8.5)
    fig.subplots_adjust(left=.12, right=.98, top=.89, bottom=.28)
    output = ROOT / "results/geobc_exploration/strain_quality_diagnosis.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(json.dumps({"plot": str(output), "source": str(source)}))


if __name__ == "__main__":
    main()
