"""Prediction-only zero-support translation pilot on three saved representatives.

No inference, weights, stress, dia, test fields or new cases are accessed. The
BC correction uses prediction means at known homogeneous-zero constrained
components. GT is used only for metrics and a separately named oracle offset.
Subtracting a constant preserves every displacement difference and P1 strain;
it zeros support component means, not every constrained node displacement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if os.name=="nt" and (ROOT/".local_deps").is_dir():
    sys.path.insert(0,str(ROOT/".local_deps"))
import numpy as np

from geobc.displacement_condition import BASIS_ORDER,file_identity,host_path,verify_file_identity
from geobc.local_geometry import xyz_sha256

KEYS = ("10_492/tor","107_91/hor","102_249/tor")
RUNS = {"gradient_zero":"runs/geobc_exploration/displacement_gradient_zero_10k",
        "gradient_actual":"runs/geobc_exploration/displacement_gradient_actual_10k"}
AXES = ("x","y","z")


def _json(path): return json.loads(path.read_text(encoding="utf-8"))


def vector_metrics(truth,prediction):
    """Original signed physical vector metrics; truth never enters BC offset."""
    error = prediction-truth
    truth_energy = float(np.einsum("nc,nc->",truth,truth))
    error_energy = float(np.einsum("nc,nc->",error,error))
    centroid_error = error.mean(0)
    translation_energy = len(error)*float(centroid_error@centroid_error)
    return {"vector_relative_l2":float(np.sqrt(error_energy/truth_energy)) if truth_energy else None,
            "component_RMSE_mm":float(np.sqrt(error_energy/error.size)),
            "max_error_vector_norm_mm":float(np.linalg.norm(error,axis=1).max()),
            "centroid_error_mm":centroid_error.tolist(),"centroid_error_norm_mm":float(np.linalg.norm(centroid_error)),
            "truth_energy_mm2":truth_energy,"error_energy_mm2":error_energy,
            "mean_error_translation_energy_mm2":translation_energy,
            "fraction_error_energy_from_mean_translation":translation_energy/error_energy if error_energy else 0.}


def support_stats(prediction,flags):
    per_axis = {}; sum_squared = 0.; total = 0; max_abs = 0.
    for j,axis in enumerate(AXES):
        values = prediction[flags[:,j],j]
        if not len(values): raise ValueError(f"No zero-BC support nodes for {axis}; offset is undefined")
        squared = float(values@values)
        per_axis[axis] = {"constrained_components":len(values),"prediction_mean_mm":float(values.mean()),
                          "prediction_RMS_mm":float(np.sqrt(squared/len(values))),"prediction_max_abs_mm":float(np.abs(values).max())}
        sum_squared += squared; total += len(values); max_abs = max(max_abs,float(np.abs(values).max()))
    return {"per_axis":per_axis,"constrained_component_count":total,
            "constrained_component_RMS_mm":float(np.sqrt(sum_squared/total)),"constrained_component_max_abs_mm":max_abs}


def support_bias(prediction,flags):
    """Uses only predictions + known zero DOF masks, intentionally no truth arg."""
    if prediction.shape!=(len(flags),3) or flags.shape!=(len(prediction),3) or flags.dtype!=np.dtype(bool):
        raise ValueError("Expected original-node predictions[N3] and bool DOF masks[N3]")
    if not np.isfinite(prediction).all() or not np.all(flags.sum(0)>0):
        raise ValueError("Finite predictions and at least one support for every component required")
    return np.asarray([prediction[flags[:,j],j].mean() for j in range(3)],np.float64)


def analyze_case(truth,prediction,flags):
    bias = support_bias(prediction,flags)
    corrected = prediction-bias
    raw = vector_metrics(truth,prediction); bc = vector_metrics(truth,corrected)
    # This distinct oracle is descriptive only: it requires GT and is not the
    # proposed boundary-based correction or an allowed deployed input.
    oracle_bias = (prediction-truth).mean(0)
    oracle = vector_metrics(truth,prediction-oracle_bias)
    difference_error = float(np.max(np.abs(np.diff(corrected,axis=0)-np.diff(prediction,axis=0))))
    numeric_tolerance = 16*np.finfo(np.float64).eps*max(1.,float(np.abs(prediction).max()))
    if difference_error>numeric_tolerance:
        raise FloatingPointError("Constant subtraction unexpectedly changed pairwise displacement differences")
    return {"bias_mm_prediction_only_support_mean":bias.tolist(),"bias_vector_norm_mm":float(np.linalg.norm(bias)),
            "raw":raw,"corrected_using_zero_BC":bc,"support_before":support_stats(prediction,flags),
            "support_after":support_stats(corrected,flags),
            "oracle_GT_mean_error_translation_removal":{"oracle_bias_mm":oracle_bias.tolist(),"metrics":oracle,
                "usage":"GT-dependent descriptive oracle only; never computes or replaces BC correction"},
            "relative_L2_change_percentage_points":100*(bc["vector_relative_l2"]-raw["vector_relative_l2"]),
            "fraction_error_energy_removed_by_BC_offset":(raw["error_energy_mm2"]-bc["error_energy_mm2"])/raw["error_energy_mm2"] if raw["error_energy_mm2"] else 0.,
            "strain_invariance":{"all_adjacent_original_node_pair_differences_max_abs_change_mm":difference_error,
                "numeric_tolerance_mm":numeric_tolerance,"passed":True,
                "analytical":"(u_a-b)-(u_c-b)=u_a-u_c for all nodes, so every fixed-geometry P1 strain is unchanged"}}


def diagnose(args):
    begun = time.monotonic()
    split_path = host_path(args.split); split = _json(split_path)
    guard_path = host_path(args.load_basis); guard = _json(guard_path)
    guards = {record["cad_id"]:record for record in guard["cases"]}
    if guard.get("compatible") is not True or guard["groups"]!={k:split["groups"][k] for k in ("train","validation")}:
        raise ValueError("Frozen zero-support/load-basis scope differs from the fixed split")
    groups = {key:("train" if key.split('/')[0] in split["groups"]["train"] else "validation") for key in KEYS}
    if groups!={"10_492/tor":"train","107_91/hor":"validation","102_249/tor":"validation"}:
        raise ValueError("Predeclared three representative scopes changed")
    methods = {}; indexes = {}; configs = {}; summaries = {}; metadata = {}
    for method,run_name in RUNS.items():
        run = host_path(run_name); summary = _json(run/"summary.json"); config = _json(run/"config.json")
        index = _json(run/"predictions_best/index.json")
        weight = 0 if method=="gradient_zero" else 1
        if summary.get("status")!="completed" or summary.get("gradient_weight")!=weight or index["step"]!=summary["best_step"]:
            raise ValueError("Representative predictions must belong to completed matched pair best steps")
        if any(key not in index["cases"] for key in KEYS):
            raise ValueError("A predeclared shared prediction is absent; no substitute cases are selected")
        indexes[method] = index; summaries[method] = summary; configs[method] = config
        metadata[method] = {"run":str(run),"best_step":summary["best_step"],"gradient_weight":weight,
            "summary_source":file_identity(run/"summary.json",small_sha=True),"config_source":file_identity(run/"config.json",small_sha=True),
            "index_source":file_identity(run/"predictions_best/index.json",small_sha=True)}
        methods[method] = {}
    actual_intersection = set(indexes["gradient_zero"]["cases"]) & set(indexes["gradient_actual"]["cases"])
    if actual_intersection!=set(KEYS): raise ValueError("Current shared representative intersection differs from the declared three cases")
    prepared_sources = []
    for key in KEYS:
        cad,load = key.split("/"); group = groups[key]
        if load not in BASIS_ORDER or cad in split["groups"]["test"] or cad not in split["groups"][group] or guards[cad].get("zero_fixed_constraints") is not True:
            raise ValueError("Only existing known-load train/validation homogeneous-zero supports are allowed")
        source_path = host_path(split["case_paths"][key])
        if not source_path.is_relative_to(host_path(args.data)):
            raise ValueError("Prepared masks escaped the declared data directory")
        for method,config in configs.items():
            source_records = {r["key"]:r for r in config["contract"]["selected_sources"]}
            expected = source_records[key]
            if host_path(expected["path"])!=source_path: raise ValueError("Saved model prepared-source identity differs")
            verify_file_identity(source_path,expected)
        with np.load(source_path,allow_pickle=False) as stored:
            features = np.asarray(stored["features"],np.float32)
            xyz = np.asarray(stored["xyz"],np.float32)
            if str(stored["sample_key"].item())!=key or str(stored["u_units"].item())!="mm" or str(stored["label_alignment"].item())!="official CSV GRID coordinates -> H5 vertices":
                raise ValueError("Prepared sample/unit/alignment guard differs")
        if features.shape!=(len(xyz),25) or xyz.shape!=(len(features),3) or not np.isfinite(xyz).all() or not np.isfinite(features[:,:3]).all() or not np.isin(features[:,:3],[0,1]).all():
            raise ValueError("Exact original25 first-three zero-support masks are required")
        flags = features[:,:3].astype(bool)
        truths = []
        for method,run_name in RUNS.items():
            record = indexes[method]["cases"][key]
            if record["group"]!=group or record["nodes"]!=len(xyz) or host_path(record["source"])!=source_path or Path(record["file"]).name!=record["file"]:
                raise ValueError("Representative node/source/group/path identity differs")
            path = host_path(run_name)/"predictions_best"/record["file"]
            with np.load(path,allow_pickle=False) as stored:
                pred_xyz = np.asarray(stored["xyz"],np.float32)
                truth = np.asarray(stored["u_true_mm"],np.float64); pred = np.asarray(stored["u_pred_mm"],np.float64)
                alpha = np.asarray(stored["alpha"],np.float64)
                if (str(stored["sample_key"].item())!=key or int(stored["step"].item())!=summaries[method]["best_step"]
                        or stored["channels"].tolist()!=["ux_mm","uy_mm","uz_mm"] or not np.isfinite(float(stored["u_scale_mm"].item())) or float(stored["u_scale_mm"].item())<=0):
                    raise ValueError("Prediction sample/best-step/physical-channel units differ")
            if not np.array_equal(pred_xyz,xyz) or truth.shape!=xyz.shape or pred.shape!=xyz.shape or not np.isfinite(truth).all() or not np.isfinite(pred).all():
                raise ValueError("Prediction exact XYZ/node order or signed physical u differs")
            if not np.array_equal(alpha,np.asarray(guards[cad]["alpha_by_load"][load]["alpha"],np.float64)):
                raise ValueError("Saved representative load basis differs")
            analysis = analyze_case(truth,pred,flags)
            saved_l2 = summaries[method]["best_metrics"][group]["per_case"][key]["relative_l2"]
            if not np.isclose(analysis["raw"]["vector_relative_l2"],saved_l2,rtol=1e-5,atol=1e-8):
                raise ValueError("Saved prediction metric does not reproduce the completed best metric")
            methods[method][key] = {"group":group,"nodes":len(xyz),"xyz_sha256":xyz_sha256(xyz),
                "support_component_counts":flags.sum(0).tolist(),"prediction_source":file_identity(path),**analysis}
            truths.append(truth)
        if not np.array_equal(truths[0],truths[1]): raise ValueError("Pair stored different reference displacement fields")
        prepared_sources.append({"sample_key":key,**file_identity(source_path),
            "members_read":["features (only columns0:3 used)","xyz","sample_key","u_units","label_alignment"],
            "raw_displacement_or_stress_labels_read":False})
    aggregates = {}
    for method,cases in methods.items():
        aggregates[method] = {"representative_count":3,"raw_mean_vector_relative_l2":float(np.mean([r["raw"]["vector_relative_l2"] for r in cases.values()])),
            "corrected_mean_vector_relative_l2":float(np.mean([r["corrected_using_zero_BC"]["vector_relative_l2"] for r in cases.values()])),
            "raw_mean_translation_energy_fraction":float(np.mean([r["raw"]["fraction_error_energy_from_mean_translation"] for r in cases.values()])),
            "mean_relative_L2_change_percentage_points":float(np.mean([r["relative_L2_change_percentage_points"] for r in cases.values()])),
            "improved_cases":sum(r["relative_L2_change_percentage_points"]<0 for r in cases.values()),
            "max_pair_difference_change_mm":max(r["strain_invariance"]["all_adjacent_original_node_pair_differences_max_abs_change_mm"] for r in cases.values())}
    result = {"status":"completed","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"Selected representative pilot only: oneTRAIN and twovalidation known-load cases; same saved intersection for both methods",
        "case_keys":list(KEYS),"basis_order":list(BASIS_ORDER),"units":"mm",
        "correction":"b_j=mean(predicted_u_j[original25 support DOF_j==1]); corrected_u=predicted_u-b; known prescribed support displacement0; GT never enters b",
        "GT_usage":"Stored u_true_mm is used only for physical metrics and explicitly separate oracle mean-error translation removal",
        "strict_boundary_enforcement":False,"P1_strain_unchanged":True,"inference_or_training_updates":0,"GPU_used":False,
        "fields_not_read":["stress","dia","final5 test","new representative cases","weights","original prepared displacement labels","large geometry caches"],
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"split_source":file_identity(split_path,small_sha=True),
        "load_basis_source":file_identity(guard_path,small_sha=True),"run_sources":metadata,"prepared_mask_sources":prepared_sources,
        "per_method":methods,"selected_representative_aggregates":aggregates,"elapsed_seconds":time.monotonic()-begun,
        "limits":"Cases were selected by existing representative-export policy, not a new random/all-validation cohort. These numbers cannot estimate all5-validation performance. Correction removes only a constant translation and support means; residual pernode violations remain, strain does not improve, and GT-oracle correction is not deployable."}
    output = host_path(args.output); temp = output.with_suffix(output.suffix+".tmp")
    temp.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8");temp.replace(output)
    print(json.dumps({"status":"completed","elapsed_seconds":result["elapsed_seconds"],"aggregates":aggregates,"output":str(output)}),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",default="notes/geobc_split.json")
    parser.add_argument("--load-basis",default="notes/geobc_load_basis_45.json")
    parser.add_argument("--data",default="data/prepared_geobc_multiload")
    parser.add_argument("--output",default="notes/geobc_boundary_rigid_offset.json")
    diagnose(parser.parse_args())


if __name__=="__main__":main()
