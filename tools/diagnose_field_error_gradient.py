"""One fixed exported TRAIN field decomposition; no models or optimization.

The contract and dry run import only the standard library. Runtime reads two
existing prediction NPZ files and one existing gradient cache. The affine fit
uses GT and is a descriptive oracle, not an improved deployable prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KEY, CAD, LOAD, STEP, NODES, CELLS = "10_58/tor", "10_58", "tor", 10000, 254645, 161200
MODES = ("euclidean", "graph")
TOLERANCE = {"relative": 2e-5, "absolute": 1e-8}
SOURCES = ("tools/diagnose_field_error_gradient.py", "geobc/gradient_supervision.py",
           "geobc/displacement_condition.py", "geobc/local_geometry.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(value):
    """Resolve this workspace's recorded Windows/WSL paths without a shell."""
    value = str(value).replace("\\", "/")
    for prefix in ("/mnt/c/Users/ASUS/Desktop/GeoBC Neural Operator/",
                   "C:/Users/ASUS/Desktop/GeoBC Neural Operator/"):
        if value.startswith(prefix):
            return ROOT / value[len(prefix):]
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def portable(value):
    return path(value).resolve().relative_to(ROOT.resolve()).as_posix()


def read_json(value):
    return json.loads(path(value).read_text(encoding="utf-8"))


def identity(value, *, small=False):
    p = path(value)
    s = p.stat()
    record = {"path": portable(p), "bytes": s.st_size, "mtime_ns": s.st_mtime_ns}
    if small:
        record["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return record


def atomic_json(value, data):
    p = path(value)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, p)


def request_contract(plan_path):
    """JSON/source SHA/stat only: no arrays, weights, NumPy, Torch, or CUDA."""
    p = path(plan_path)
    plan = read_json(p)
    require(plan.get("status") == "predeclared" and plan.get("stage") == "fixed_field_error_gradient_decomposition",
            "The root's fixed field-error protocol must be predeclared")
    selected, scope = plan["selection"], plan["scope"]
    require((selected["case_key"], selected["group"], selected["nodes"], selected["cells"], selected["best_step_both"])
            == (KEY, "train", NODES, CELLS, STEP) and selected["historical_smaller_or_gate_models_used"] is False,
            "Only the common current TRAIN10_58/tor field at selected10000 is allowed")
    require(scope["existing_prediction_files"] == 2 and scope["active_load"] == LOAD
            and all(scope[name] == 0 for name in ("raw_prepared_GT_files_opened", "raw_mesh_or_deck_files_opened",
                "validation_files_opened", "dia_or_final_test_files_opened", "weights_or_models_loaded",
                "training_or_inference_runs", "parameter_updates")), "Diagnostic read scope changed")
    require(plan["compute"]["inputs_read_once"] and plan["compute"]["geometry_computed_once_and_shared"]
            and plan["predeclared_readout"]["no_automatic_training"], "One fixed diagnostic is required")
    pair_path = path(selected["pair_analysis"])
    pair = read_json(pair_path)
    require(pair.get("status") == "completed" and pair["matched_controls_passed"] is True
            and pair["pair_controls_passed"] is True and pair["fresh_fused_pair"] is True
            and pair["both_distance_modes_active"] is True and pair["historical_control_reused"] is False,
            "The two completed fresh matched distance fields are required")
    manifest_path = path(selected["gradient_manifest"])
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "completed" and manifest["complete"] is True
            and CAD in manifest["groups"]["train"] and CAD not in manifest["groups"]["validation"] + manifest["groups"]["test"]
            and manifest["basis_order"] == ["ver", "hor", "tor"]
            and manifest["component_order"] == ["xx", "yy", "zz", "xy", "xz", "yz"]
            and manifest["frobenius_weights"] == [1., 1., 1., 2., 2., 2.]
            and manifest["engineering_shear"] is False, "Fixed TRAIN cache scope/tensor convention differs")
    cache_record = manifest["CADs"][CAD]
    cache_path = path(selected["gradient_cache"])
    require(cache_path == manifest_path.parent / cache_record["file"]
            and cache_record["node_count"] == NODES and cache_record["cell_count"] == CELLS
            and cache_record["normalization"]["status"] == "known", "Selected cache/path/physical domain differs")
    cache_identity = identity(cache_path)
    require(all(cache_identity[k] == cache_record[k] for k in ("bytes", "mtime_ns")), "Fixed gradient cache stat differs")
    branches, index_keys = {}, {}
    helper_sha = identity(ROOT / "geobc/gradient_supervision.py", small=True)["sha256"]
    for mode in MODES:
        prediction = path(selected[mode])
        run = prediction.parent.parent
        require(run == ROOT / f"runs/geobc_exploration/displacement_interface_{mode}_10k"
                and prediction.name == "case_001.npz", "Only the two current exported fields are allowed")
        index_path = prediction.parent / "index.json"
        index, config, summary, metrics = (read_json(q) for q in (index_path, run / "config.json",
                                                              run / "summary.json", run / "metrics_best.json"))
        require(summary.get("status") == "completed" and summary["config"] == config
                and summary["best_step"] == metrics["step"] == index["step"] == STEP,
                f"{mode}: completed u-selected10000 field is required")
        entry = index["cases"][KEY]
        require(entry["group"] == "train" and entry["nodes"] == NODES
                and entry["file"] == prediction.name and config["arguments"]["distance_mode"] == mode
                and config["gradient_supervision"]["manifest_sha256"] == identity(manifest_path, small=True)["sha256"]
                and config["contract"]["code_sha256"]["gradient_supervision.py"] == helper_sha,
                f"{mode}: export/case/immutable P1 source differs")
        require(metrics["train"]["per_case"][KEY]["nodes"] == NODES
                and metrics["train"]["strain"]["per_case"][KEY]["nodes"] == NODES,
                f"{mode}: stored all-node metrics differ")
        branches[mode] = {"prediction": identity(prediction), "index": identity(index_path, small=True),
                          "config": identity(run / "config.json", small=True),
                          "summary": identity(run / "summary.json", small=True),
                          "metrics": identity(run / "metrics_best.json", small=True),
                          "reference_metrics": {"u": metrics["train"]["per_case"][KEY],
                                                "strain": metrics["train"]["strain"]["per_case"][KEY]},
                          "selected_step": STEP, "u_scale_mm": config["scaler"]["scale_mm"]}
        index_keys[mode] = set(index["cases"])
    require(sorted(index_keys["euclidean"] & index_keys["graph"]) == [KEY], "Exported representative intersection changed")
    return {"version": 1, "stage": plan["stage"], "plan": identity(p, small=True), "protocol": plan,
            "case_key": KEY, "selected_step": STEP, "nodes": NODES, "cells": CELLS,
            "branches": branches, "pair_analysis": identity(pair_path, small=True),
            "gradient_manifest": identity(manifest_path, small=True), "gradient_cache": cache_identity,
            "cache_metadata": cache_record,
            "source_sha256": {name: identity(ROOT / name, small=True)["sha256"] for name in SOURCES},
            "metric_matching_tolerance": TOLERANCE,
            "geometry_bins": {"measures": ["minimum_corner_edge_mm", "volume_length_mm"],
                "volume_length": "(6*absV_mm3)^(1/3)", "quantiles": [0., .25, .5, .75, 1.],
                "ties": "right=True: equal internal cuts enter the higher bin; retain all four bins, including empty tied bins",
                "energy_domain": "volume-weighted cell tensors only; no all-node energy bin additivity claim"},
            "arrays_opened_in_contract": 0, "weights_loaded": 0, "native_models_loaded": 0,
            "forward_calls": 0, "backwards": 0, "optimizer_updates": 0}


def scalar(tensor):
    value = float(tensor)
    require(math.isfinite(value), "Nonfinite physical reduction")
    return value


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def affine_decomposition(xyz, error):
    """FP64 equal-node affine LS; full-rank four-column normal system."""
    import torch
    centered = xyz.double() - xyz.double().mean(0)
    design = torch.cat((torch.ones_like(centered[:, :1]), centered), dim=1)
    gram = design.T @ design
    eigenvalues = torch.linalg.eigvalsh(gram)
    require(scalar(eigenvalues[0]) > scalar(eigenvalues[-1]) * 1e-12, "Affine design is rank deficient")
    coefficients = torch.linalg.solve(gram, design.T @ error.double())
    affine = design @ coefficients
    residual = error.double() - affine
    return affine, residual, {"coefficient_rows": ["translation_mm", "centered_x_normalized", "centered_y_normalized", "centered_z_normalized"],
        "coefficients_mm": coefficients.cpu().tolist(), "xyz_center_normalized": xyz.double().mean(0).cpu().tolist(),
        "normal_system_condition_number": scalar(eigenvalues[-1] / eigenvalues[0]),
        "normal_equation_residual_max_abs": scalar((design.T @ residual).abs().max()),
        "GT_dependent_oracle": True}


def partition_energies(raw, affine, residual, reference, weights, measure=None):
    """Preserve measured signed cross terms and the original truth denominator."""
    import torch
    w = torch.as_tensor(weights, dtype=torch.float64, device=raw.device)
    m = torch.ones(len(raw), dtype=torch.float64, device=raw.device) if measure is None else measure.double()
    def dot(a, b):
        return scalar(((a.double() * b.double() * w).sum(-1) * m).sum())
    truth, eraw, eaff, eres, cross = dot(reference, reference), dot(raw, raw), dot(affine, affine), dot(residual, residual), 2*dot(affine, residual)
    total = scalar(m.sum())
    closure = eraw-eaff-eres-cross
    def relative(energy):
        value = ratio(energy, truth)
        return math.sqrt(value) if value is not None else None
    return {"reference_energy": truth, "raw_error_energy": eraw, "affine_error_energy": eaff,
            "residual_error_energy": eres, "signed_two_affine_residual_cross": cross,
            "energy_identity_closure_absolute": closure,
            "energy_identity_closure_relative": ratio(abs(closure), max(eraw, eaff+eres+abs(cross))),
            "field_identity_max_abs": scalar((raw-affine-residual).abs().max()),
            "reference_RMS": math.sqrt(truth/total), "raw_error_RMS": math.sqrt(eraw/total),
            "affine_error_RMS": math.sqrt(eaff/total), "residual_error_RMS": math.sqrt(eres/total),
            "raw_relative_l2": relative(eraw), "affine_relative_l2": relative(eaff), "residual_relative_l2": relative(eres),
            "residual_over_raw_error_energy": ratio(eres, eraw), "measure_total": total}


def geometry_bins(inverse, volumes, length_mm):
    """Both input-only physical measures and four fixed quartile bins once."""
    import torch
    edges = torch.linalg.inv(inverse)
    six = torch.stack((edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 1]-edges[:, 0],
                       edges[:, 2]-edges[:, 0], edges[:, 2]-edges[:, 1]), dim=1)
    measures = {"minimum_corner_edge_mm": torch.linalg.vector_norm(six, dim=2).amin(1),
                "volume_length_mm": (6*volumes.abs()).pow(1./3.)}
    result = {}
    for name, values in measures.items():
        require(bool(torch.isfinite(values).all()) and bool((values > 0).all()), "Invalid physical cell length")
        quantiles = torch.quantile(values, values.new_tensor([0., .25, .5, .75, 1.]))
        labels = torch.bucketize(values, quantiles[1:4], right=True)
        result[name] = {"labels": labels, "quantiles_mm": quantiles.cpu().tolist(),
                        "quantiles_over_characteristic_length": (quantiles/length_mm).cpu().tolist(),
                        "ties_at_internal_boundaries": bool((quantiles[1:] == quantiles[:-1]).any())}
    return result


def bin_cell_energies(raw, affine, residual, reference, volumes, bins):
    """Only disjoint CELL energies are summed; node recovery is not binned."""
    import torch
    w = raw.new_tensor([1., 1., 1., 2., 2., 2.])
    energies = {"raw_error_energy": (raw.square()*w).sum(1)*volumes,
                "affine_error_energy": (affine.square()*w).sum(1)*volumes,
                "residual_error_energy": (residual.square()*w).sum(1)*volumes,
                "reference_energy": (reference.square()*w).sum(1)*volumes,
                "signed_two_affine_residual_cross": 2*(affine*residual*w).sum(1)*volumes}
    totals = {name: scalar(value.sum()) for name, value in energies.items()}
    volume_total = scalar(volumes.sum())
    out = {}
    for name, record in bins.items():
        rows = []
        for b in range(4):
            mask = record["labels"] == b
            count, volume = int(mask.sum()), scalar(volumes[mask].sum())
            sums = {label: scalar(value[mask].sum()) for label, value in energies.items()}
            rows.append({"bin": b, "lower_mm": record["quantiles_mm"][b], "upper_mm": record["quantiles_mm"][b+1],
                "cell_count": count, "cell_count_share": count/len(volumes), "volume_mm3": volume,
                "physical_volume_share": volume/volume_total, **sums,
                "raw_error_share": ratio(sums["raw_error_energy"], totals["raw_error_energy"]),
                "residual_error_share": ratio(sums["residual_error_energy"], totals["residual_error_energy"]),
                "reference_energy_share": ratio(sums["reference_energy"], totals["reference_energy"]),
                "residual_error_squared_relative_to_full_case_truth": ratio(sums["residual_error_energy"], totals["reference_energy"]),
                "raw_error_Frobenius_RMS": math.sqrt(sums["raw_error_energy"]/volume) if volume else None,
                "residual_error_Frobenius_RMS": math.sqrt(sums["residual_error_energy"]/volume) if volume else None,
                "reference_Frobenius_RMS": math.sqrt(sums["reference_energy"]/volume) if volume else None,
                "energy_identity_closure_absolute": sums["raw_error_energy"]-sums["affine_error_energy"]-sums["residual_error_energy"]-sums["signed_two_affine_residual_cross"]})
        out[name] = {"quantiles_mm": record["quantiles_mm"],
            "quantiles_over_characteristic_length": record["quantiles_over_characteristic_length"],
            "ties_at_internal_boundaries": record["ties_at_internal_boundaries"], "bins": rows,
            "bin_energy_sum_closure": {label: sum(row[label] for row in rows)-total for label, total in totals.items()},
            "metric": "physical-volume-weighted CELL Frobenius energies; not recovered-node relativeL2"}
    return out


def matching(expected, actual, name):
    require(math.isclose(expected, actual, rel_tol=TOLERANCE["relative"], abs_tol=TOLERANCE["absolute"]),
            f"Stored {name} mismatch: computed={actual}, expected={expected}")
    return {"expected": expected, "computed": actual, "absolute_difference": abs(expected-actual),
            "relative_tolerance": TOLERANCE["relative"], "absolute_tolerance": TOLERANCE["absolute"], "passed": True}


def memory_record(device):
    import torch
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        cpu = peak/(1024**2) if sys.platform == "darwin" else peak/1024
    except ImportError:
        cpu = None
    return {"process_peak_RSS_MiB": cpu,
            "CUDA_peak_allocated_MiB": torch.cuda.max_memory_allocated(device)/1024**2 if device.type == "cuda" else None,
            "CUDA_peak_reserved_MiB": torch.cuda.max_memory_reserved(device)/1024**2 if device.type == "cuda" else None}


def run(args):
    contract = request_contract(args.plan)
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "output": portable(args.output), "contract": contract}, indent=2))
        return
    output = path(args.output)
    if (output/"summary.json").exists():
        saved = read_json(output/"summary.json")
        require(saved.get("status") == "completed" and saved["config"]["contract"] == contract,
                "Existing result is incomplete or has a different fixed-field contract")
        result_path = ROOT/"notes/geobc_error_gradient_result.json"
        if result_path.exists():
            require(read_json(result_path)["config"]["contract"] == contract,
                    "Existing formal readout belongs to a different fixed-field contract")
        else:
            atomic_json(result_path, saved)
        print(json.dumps({"status": "completed_reused", "output": portable(output)}))
        return
    if output.exists() and any(output.iterdir()):
        require((output/"config.json").exists() and read_json(output/"config.json")["contract"] == contract
                and (output/"status.json").exists() and read_json(output/"status.json").get("status") == "failed",
                "Nonempty output is not a failed identical diagnostic; do not overwrite")
    output.mkdir(parents=True, exist_ok=True)
    lock = output/"running.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    start = time.perf_counter()
    config = {"arguments": vars(args), "contract": contract}
    def update(status, **extra):
        atomic_json(output/"status.json", {"status": status, "pid": os.getpid(), "elapsed_seconds": time.perf_counter()-start,
            "case_key": KEY, "parameter_updates": 0, "backwards": 0, "model_forward_calls": 0, **extra})
    atomic_json(output/"config.json", config)
    try:
        import numpy as np
        import torch
        from geobc.gradient_supervision import cell_tensor_strain, recover_node_tensor_strain, load_gradient_cache
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        update("loading_fixed_exports", device=str(device), CPU_fallback=device.type == "cpu")
        fields = {}
        allowed = {"xyz", "u_true_mm", "u_pred_mm", "alpha", "sample_key", "step", "channels", "u_scale_mm"}
        for mode in MODES:
            with np.load(path(contract["branches"][mode]["prediction"]["path"]), allow_pickle=False) as stored:
                require(set(stored.files) == allowed, "Unexpected field in fixed prediction export")
                fields[mode] = {key: stored[key] for key in allowed}
            f = fields[mode]
            require(f["xyz"].shape == f["u_true_mm"].shape == f["u_pred_mm"].shape == (NODES,3)
                    and f["xyz"].dtype == np.dtype("float32") and f["alpha"].shape == (3,)
                    and str(f["sample_key"].item()) == KEY and int(f["step"].item()) == STEP
                    and f["channels"].tolist() == ["ux_mm", "uy_mm", "uz_mm"]
                    and np.array_equal(f["alpha"], np.asarray([0.,0.,1.]))
                    and all(np.isfinite(f[k]).all() for k in ("xyz", "u_true_mm", "u_pred_mm", "alpha", "u_scale_mm")),
                    f"{mode}: invalid all-node signed displacement export")
            require(math.isclose(float(f["u_scale_mm"]), contract["branches"][mode]["u_scale_mm"], rel_tol=1e-12),
                    f"{mode}: exported fixed scale differs")
        require(np.array_equal(fields["euclidean"]["xyz"], fields["graph"]["xyz"])
                and np.array_equal(fields["euclidean"]["u_true_mm"], fields["graph"]["u_true_mm"]),
                "The two selected fields do not share exactly the same XYZ and signed GT")
        manifest = read_json(contract["gradient_manifest"]["path"])
        cache = load_gradient_cache(path(contract["gradient_cache"]["path"]).parent, CAD,
                                    xyz=fields["euclidean"]["xyz"], manifest=manifest)
        length = float(cache.metadata["normalization"]["characteristic_length_mm"])
        corners = torch.as_tensor(cache.corners, dtype=torch.long, device=device)
        cells = torch.as_tensor(cache.cells, dtype=torch.long, device=device)
        inverse = torch.as_tensor(cache.inverse_physical_edge_mm, dtype=torch.float64, device=device)
        volumes = torch.as_tensor(cache.volumes, dtype=torch.float64, device=device)
        reference_cell = torch.as_tensor(cache.reference_cell_tensor_strain[:,2].copy(), dtype=torch.float64, device=device)
        xyz = torch.as_tensor(fields["euclidean"]["xyz"], dtype=torch.float64, device=device)
        true = torch.as_tensor(fields["euclidean"]["u_true_mm"], dtype=torch.float64, device=device)
        with torch.no_grad():
            update("geometry_and_reference", device=str(device))
            bins = geometry_bins(inverse, volumes, length)
            reference_node = recover_node_tensor_strain(reference_cell, volumes, cells, NODES)
            true_cell = cell_tensor_strain(true[corners], inverse)
            rounding_cell = true_cell-reference_cell
            rounding_node = recover_node_tensor_strain(rounding_cell, volumes, cells, NODES)
            zero_cell, zero_node = torch.zeros_like(reference_cell), torch.zeros_like(reference_node)
            models = {}
            for position,mode in enumerate(MODES):
                update("decomposing_fixed_field", mode=mode, completed_models=position, device=str(device))
                pred = torch.as_tensor(fields[mode]["u_pred_mm"], dtype=torch.float64, device=device)
                error = pred-true
                affine, residual, fit = affine_decomposition(xyz, error)
                fit["nominal_physical_gradient_coordinate_by_component"] = (torch.tensor(fit["coefficients_mm"][1:], dtype=torch.float64)/length).tolist()
                fit["nominal_gradient_limit"] = "LS coefficient/L for rounded input xyz; descriptive approximation only. Actual affine P1 uses nodal affine values and unchanged cached physical inverse edges."
                raw_cell = cell_tensor_strain(error[corners], inverse)
                affine_cell = cell_tensor_strain(affine[corners], inverse)
                residual_cell = cell_tensor_strain(residual[corners], inverse)
                raw_node = recover_node_tensor_strain(raw_cell, volumes, cells, NODES)
                affine_node = recover_node_tensor_strain(affine_cell, volumes, cells, NODES)
                residual_node = recover_node_tensor_strain(residual_cell, volumes, cells, NODES)
                u = partition_energies(error, affine, residual, true, [1.,1.,1.])
                u.update({"units": "mm; vector RMS uses N, component RMSE uses3N", "raw_component_RMSE_mm": u["raw_error_RMS"]/math.sqrt(3.)})
                cell = partition_energies(raw_cell, affine_cell, residual_cell, reference_cell, [1.,1.,1.,2.,2.,2.], volumes)
                node = partition_energies(raw_node, affine_node, residual_node, reference_node, [1.,1.,1.,2.,2.,2.])
                cell["units"] = "dimensionless tensor RMS; cell energy integrates tensor squared over physical mm3"
                node["units"] = "dimensionless tensor RMS; node energy is unweighted original-node tensor squared sum"
                # Match the original evaluator's pred-minus-cached-reference convention.
                predicted_cell = cell_tensor_strain(pred[corners], inverse)
                predicted_node = recover_node_tensor_strain(predicted_cell, volumes, cells, NODES)
                stored_error = predicted_node-reference_node
                stored_partition = partition_energies(stored_error, zero_node, stored_error, reference_node, [1.,1.,1.,2.,2.,2.])
                expected = contract["branches"][mode]["reference_metrics"]
                checks = {"u_relative_l2": matching(expected["u"]["relative_l2"], u["raw_relative_l2"], "u relativeL2"),
                          "u_component_RMSE_mm": matching(expected["u"]["rmse_mm"], u["raw_component_RMSE_mm"], "u RMSE"),
                          "stored_reference_P1_relative_l2": matching(expected["strain"]["relative_l2"], stored_partition["raw_relative_l2"], "stored-reference P1 relativeL2"),
                          "P1_reference_energy": matching(expected["strain"]["truth_energy"], stored_partition["reference_energy"], "P1 reference energy")}
                models[mode] = {"selected_step": STEP, "affine_oracle_fit": fit,
                    "u": u, "cell_P1": cell, "all_node_P1": node,
                    "operator_P1_over_u_relative_error_ratio": node["raw_relative_l2"]/u["raw_relative_l2"]
                        if node["raw_relative_l2"] is not None and u["raw_relative_l2"] is not None and u["raw_relative_l2"] > 0 else None,
                    "relative_error_ratio_limit": "A(pred-true) recovered-node P1 relativeL2 divided by u relativeL2, using original case denominators. Descriptive only, not a frequency or causal proof. Zero u error or undefined truth denominator gives null.",
                    "stored_reference_metrics": {"P1_relative_l2": stored_partition["raw_relative_l2"],
                        "P1_error_energy": stored_partition["raw_error_energy"], "P1_error_RMS": stored_partition["raw_error_RMS"],
                        "operator_error_P1_relative_l2": node["raw_relative_l2"],
                        "stored_minus_operator_error_max_abs": scalar((stored_error-raw_node).abs().max())},
                    "stored_metric_checks": checks, "geometry_cell_bins": bin_cell_energies(raw_cell, affine_cell, residual_cell, reference_cell, volumes, bins)}
            consistency = {"cell": partition_energies(rounding_cell, zero_cell, rounding_cell, reference_cell, [1.,1.,1.,2.,2.,2.], volumes),
                           "all_node": partition_energies(rounding_node, zero_node, rounding_node, reference_node, [1.,1.,1.,2.,2.,2.]),
                           "definition": "A(u_true from export)-original cached FP32 reference; not a new truth or a correction"}
            non_affine = all(models[m]["all_node_P1"]["residual_over_raw_error_energy"] >= .8 for m in MODES)
            concentration = {}
            for mode in MODES:
                first = models[mode]["geometry_cell_bins"]["volume_length_mm"]["bins"][0]
                share,volume_share = first["residual_error_share"],first["physical_volume_share"]
                concentration[mode] = {"smallest_h_quartile_residual_error_share": share,
                    "smallest_h_quartile_physical_volume_share": volume_share,
                    "smallest_h_quartile_GT_reference_energy_share": first["reference_energy_share"],
                    "concentration_over_volume_share": ratio(share,volume_share) if share is not None else None,
                    "flag": share is not None and share >= .5 and share >= 2*volume_share}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        # Check that no source changed while this bounded, read-only diagnostic ran.
        require(request_contract(args.plan) == contract, "Fixed source/field contract changed during diagnostic")
        elapsed = time.perf_counter()-start
        summary = {"status": "completed", "stage": contract["stage"], "case_key": KEY, "group": "train",
            "selected_step_both": STEP, "nodes": NODES, "cells": CELLS, "elapsed_seconds": elapsed, "config": config,
            "device": str(device), "CPU_fallback": device.type == "cpu", "physical_arithmetic_dtype": "float64",
            "models": models, "cached_reference_consistency": consistency,
            "readout": {"non_affine_dominance_flag": non_affine, "non_affine_threshold": .8,
                "small_cell_concentration_by_model": concentration,
                "small_cell_concentration_flag": all(concentration[m]["flag"] for m in MODES),
                "flag_measure": "input h=(6V)^(1/3), smallest quartile, volume-weighted CELL residual energy",
                "no_automatic_training": True},
            "cost": {"elapsed_seconds": elapsed, "memory": memory_record(device), "prediction_NPZ_files_read": 2,
                "gradient_NPZ_files_read": 1, "active_reference_load": "tor", "model_forward_calls": 0,
                "backwards": 0, "optimizer_updates": 0, "raw_prepared_GT_or_mesh_files_opened": 0},
            "limits": contract["protocol"]["limits"] + ["Operator-error decomposition uses A(pred-true); original evaluator uses A(pred)-cachedRef. Their recorded FP32-reference consistency residual is retained.",
                "Input bins partition cell energies only. Recovered-node fields are not treated as disjoint additive bin energies."]}
        atomic_json(output/"summary.json", summary)
        atomic_json(ROOT/"notes/geobc_error_gradient_result.json", summary)
        update("completed", device=str(device), elapsed_seconds=elapsed, ETA_seconds=0., memory=summary["cost"]["memory"])
        print(json.dumps({"status": "completed", "output": portable(output), "elapsed_seconds": elapsed, "readout": summary["readout"]}))
    except Exception as exc:
        update("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        lock.unlink(missing_ok=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", default="notes/geobc_error_gradient_plan.json")
    p.add_argument("--output", default="runs/geobc_exploration/fixed_field_error_gradient_decomposition")
    p.add_argument("--dry-run", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
