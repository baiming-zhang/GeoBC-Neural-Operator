"""CPU visualization of already committed CAD20 fields; no model or raw CSV read."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def plot():
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    from geobc.available_cad_pilot_data import path, exact_process_exited
    def read(name):
        return json.loads(path(name).read_text(encoding="utf-8-sig"))
    report = root / "results/geobc_available_cad_pilot"
    assert read(report / "manifest.json")["scientific_qualification"] is True
    run = root / "runs/geobc_exploration/available_CAD20_candidate9"
    campaign = read(run / "campaign_external_terminal.json")
    assert campaign["closed"] and not campaign["active"]
    assert exact_process_exited(campaign["guard_process"])
    summary = read(run / "summary.json")
    assert summary["status"] == "completed" and summary["terminal"]
    assert summary["prediction_barrier"]["GT_eligible"]
    inputs = read(root / "data/geobc_available_cad_pilot/manifest.json")
    gt = read(run / "shared_GT/manifest.json")
    assert gt["status"] == "completed" and gt["source_GT_parses"] == 20
    # Metadata-only selection, independent of field values and error ranking.
    key = inputs["case_keys"][0]
    receipt = {"kind": "owned_CAD20_original_node_field_visualization", "case": key,
               "selection": "first fixed metadata case", "raw_CSV_reads": 0,
               "H5_reads": 0, "checkpoint_reads": 0, "model_calls": 0,
               "CUDA_calls": 0, "owned_array_reads": [], "outputs": []}
    import numpy as np
    def array(identity):
        filename = path(identity["path"])
        filename.relative_to(root)
        assert hashlib.sha256(filename.read_bytes()).hexdigest() == identity["sha256"]
        values = np.load(filename, allow_pickle=False)
        assert list(values.shape) == identity["shape"] and np.isfinite(values).all()
        receipt["owned_array_reads"].append(identity)
        return values
    record = inputs["records"][key]
    xyz = array(record["file_identities"]["xyz_path"])
    fields = [("Original FEM CSV", array(gt["records"][key]["physical"]))]
    for candidate, label in [("FNO_latent_query_grid25_step4800", "FNO grid25"),
                             ("transolver_pp_cad40_fresh4800", "Transolver++")]:
        entry = summary["prediction_inventory"][candidate + ":" + key]
        assert hashlib.sha256(path(entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
        prediction = read(entry["path"])
        assert prediction["finite_allN"] and prediction["all_original_nodes"]
        assert prediction["units"] == ["mm", "mm", "mm", "MPa"]
        fields.append((label, array(prediction["physical"])))
    assert all(v.shape == (len(xyz), 4) for _, v in fields)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    display = np.linspace(0, len(xyz)-1, min(30000, len(xyz)), dtype=np.int64)
    receipt.update(original_nodes=len(xyz), displayed_nodes=len(display),
                   display_sampling="deterministic uniform index subset; all metrics remain full-node",
                   coordinate_units="prepared normalized xyz, original-node order",
                   U_display="vector magnitude in mm, visualization only; vector labels unchanged",
                   S_display="unaltered signed original scalar in MPa")
    coords = xyz[display]
    extent = np.ptp(coords, axis=0)
    for field in ("signed_stress", "displacement_magnitude"):
        values = [v[:, 3] if field == "signed_stress" else np.linalg.norm(v[:, :3], axis=1)
                  for _, v in fields]
        if field == "signed_stress":
            vmax = float(np.percentile(np.abs(values[0]), 99))
            vmin = -vmax; cmap = "coolwarm"; unit = "Signed stress (MPa)"
            receipt["stress_color_range"] = {"absolute_limit_MPa": vmax,
                "rule": "shared symmetric GT 99th-percentile absolute S; color saturation only, values unchanged",
                "saturated_nodes": {label: int(np.count_nonzero(np.abs(value) > vmax))
                                    for (label, _), value in zip(fields, values)}}
        else:
            vmin = 0.; vmax = max(float(np.max(v)) for v in values)
            cmap = "viridis"; unit = "Displacement magnitude (mm)"
        fig = plt.figure(figsize=(13.2, 4.6), layout="constrained")
        axes = []
        for i, ((label, _), value) in enumerate(zip(fields, values)):
            ax = fig.add_subplot(1, 3, i+1, projection="3d"); axes.append(ax)
            artist = ax.scatter(*coords.T, c=value[display], cmap=cmap, vmin=vmin,
                                vmax=vmax, s=.7, rasterized=True, depthshade=False)
            ax.set_title(label); ax.set_box_aspect(np.maximum(extent, 1e-6))
            ax.view_init(elev=22, azim=-55)
            ax.set_xlabel("x", labelpad=0); ax.set_ylabel("y", labelpad=0); ax.set_zlabel("z", labelpad=-1)
            ax.tick_params(labelsize=7, pad=0)
        fig.colorbar(artist, ax=axes, shrink=.75, label=unit)
        subtitle = "shared GT 99% stress color range" if field == "signed_stress" else "U magnitude (visualization only)"
        fig.suptitle(key + " | normalized xyz; 30k original nodes\n" + subtitle, fontsize=12)
        filename = report / ("field_" + field + ".png")
        assert not filename.exists(), "Existing plot is preserved; no automatic overwrite"
        fig.savefig(filename, dpi=160); plt.close(fig)
        receipt["outputs"].append(filename.relative_to(root).as_posix())
    receipt["elapsed_CPU_seconds"] = time.perf_counter() - start
    receipt["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (report / "field_visualization_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"status": "completed", "case": key, "outputs": receipt["outputs"],
                      "elapsed_CPU_seconds": receipt["elapsed_CPU_seconds"], "raw_CSV_reads": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.plot:
        plot()
    else:
        print('{"status":"source_only","reads":0,"writes":0}')
