"""Predeclared TRAIN101_479 u/strain-quality diagnostic, CPU and read-only.

Only this already trained CAD's ver/hor/tor signed displacement labels may be
read. The reference strain is P1 recovered from training u, not a solver GT
tensor or a statement about the exported scalar stress. No student inputs,
scalers, models, fits or cached predictions are modified.
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
sys.path.insert(0, str(ROOT))
if os.name == "nt" and (ROOT / ".local_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".local_deps"))
import numpy as np

from geobc.displacement_condition import (BASIS_ORDER, alpha_from_original_features, file_identity,
    host_path, load_response_cache, load_response_manifest, verify_file_identity)
from geobc.local_geometry import xyz_sha256
from geobc.strain_condition import COMPONENT_ORDER, RECOVERY, load_strain_cache, load_strain_manifest, recover_corner_p1_strain
from tools.prepare_frozen_strain_response import read_geometry

CAD = "101_479"
SYMMETRIC_WEIGHTS = np.asarray([1, 1, 1, 2, 2, 2], np.float64)
AUTHORIZED_MEMBERS = ("sample_key", "label_alignment", "u_units", "xyz", "grid_node_ids", "features", "displacement")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector_metrics(reference, prediction):
    reference, prediction = np.asarray(reference, np.float64), np.asarray(prediction, np.float64)
    if reference.shape != prediction.shape or reference.ndim != 2 or reference.shape[1] != 3 or not len(reference):
        raise ValueError("Signed physical u metrics require matching original [N,3] fields")
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite reference/prediction u")
    error = prediction - reference
    denominator = float(np.linalg.norm(reference))
    return {"nodes": len(reference), "vector_relative_l2": float(np.linalg.norm(error) / denominator) if denominator else None,
            "reference_component_RMS_mm": float(np.sqrt(np.mean(reference ** 2))),
            "prediction_component_RMS_mm": float(np.sqrt(np.mean(prediction ** 2))),
            "error_component_RMSE_mm": float(np.sqrt(np.mean(error ** 2))),
            "reference_max_vector_norm_mm": float(np.linalg.norm(reference, axis=1).max()),
            "prediction_max_vector_norm_mm": float(np.linalg.norm(prediction, axis=1).max()),
            "max_error_vector_norm_mm": float(np.linalg.norm(error, axis=1).max())}


def strain_metrics(reference, prediction):
    reference, prediction = np.asarray(reference, np.float64), np.asarray(prediction, np.float64)
    if reference.shape != prediction.shape or reference.ndim != 2 or reference.shape[1] != 6 or not len(reference):
        raise ValueError("Tensor strain metrics require matching [N,xx yy zz xy xz yz]")
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite P1 reference/predicted strain")
    error = prediction - reference
    denominator_plain = float(np.linalg.norm(reference))
    reference_norm_sq = np.sum(reference ** 2 * SYMMETRIC_WEIGHTS, axis=1)
    prediction_norm_sq = np.sum(prediction ** 2 * SYMMETRIC_WEIGHTS, axis=1)
    error_norm_sq = np.sum(error ** 2 * SYMMETRIC_WEIGHTS, axis=1)
    denominator_sym = float(np.sqrt(reference_norm_sq.sum()))
    return {"nodes": len(reference),
            "plain6_relative_l2": float(np.linalg.norm(error) / denominator_plain) if denominator_plain else None,
            "symmetric_Frobenius_relative_l2": float(np.sqrt(error_norm_sq.sum()) / denominator_sym) if denominator_sym else None,
            "reference_plain6_component_RMS": float(np.sqrt(np.mean(reference ** 2))),
            "prediction_plain6_component_RMS": float(np.sqrt(np.mean(prediction ** 2))),
            "error_plain6_component_RMSE": float(np.sqrt(np.mean(error ** 2))),
            "reference_tensor_Frobenius_RMS": float(np.sqrt(reference_norm_sq.mean())),
            "prediction_tensor_Frobenius_RMS": float(np.sqrt(prediction_norm_sq.mean())),
            "error_tensor_Frobenius_RMS": float(np.sqrt(error_norm_sq.mean())),
            "reference_max_abs_component": float(np.abs(reference).max()),
            "prediction_max_abs_component": float(np.abs(prediction).max()),
            "error_max_abs_component": float(np.abs(error).max()),
            "max_error_tensor_Frobenius_norm": float(np.sqrt(error_norm_sq.max()))}


def diagnose(args):
    began = time.monotonic()
    parent_directory = host_path(args.u_response_cache)
    strain_directory = host_path(args.strain_response_cache)
    parent = load_response_manifest(parent_directory)
    manifest = load_strain_manifest(strain_directory, expected_parent=parent["teacher_fingerprint"])
    parent_gate = _json(host_path(parent["gate"]["path"]))
    split = _json(host_path(parent_gate["split"]["path"]))
    if CAD not in split["groups"]["train"] or CAD in split["groups"]["validation"] + split["groups"]["test"]:
        raise ValueError("The predeclared diagnostic is restricted to fixed TRAIN101_479")
    guard = _json(host_path(parent_gate["load_basis"]["path"]))
    expected_parent = file_identity(parent_directory / "manifest.json", small_sha=True)
    if manifest["parent_u_manifest"] != expected_parent:
        # Paths may be serialized on the other host; equality is checked after
        # resolving paths, while content SHA and stat must remain exact.
        saved = manifest["parent_u_manifest"]
        if host_path(saved["path"]) != parent_directory / "manifest.json" or any(saved[k] != expected_parent[k] for k in ("bytes", "mtime_ns", "sha256")):
            raise ValueError("Diagnostic u/strain parent manifest identities differ")
    data = host_path(args.data)
    # The teacher gate locks the completed config; use only source identity JSON.
    teacher_config = _json(host_path(parent["teacher"]["config"]["path"]))
    source_records = {r["key"]: r for r in teacher_config["contract"]["selected_sources"]}
    labels = []
    source_index = []
    original_xyz = grid_ids = static_features = None
    for row, load in enumerate(BASIS_ORDER):
        key = f"{CAD}/{load}"
        path = host_path(split["case_paths"][key])
        if not path.is_relative_to(data) or host_path(source_records[key]["path"]) != path:
            raise ValueError("Fixed train source escaped the predeclared aligned prepared directory")
        verify_file_identity(path, source_records[key])
        with np.load(path, allow_pickle=False) as stored:
            if str(stored["sample_key"].item()) != key or str(stored["label_alignment"].item()) != "official CSV GRID coordinates -> H5 vertices":
                raise ValueError("Original CSV GRID-aligned signed training displacement is required")
            if str(stored["u_units"].item()) != "mm":
                raise ValueError("Reference training u must retain original physical mm units")
            xyz = np.asarray(stored["xyz"], np.float32)
            ids = np.asarray(stored["grid_node_ids"])
            features = np.asarray(stored["features"], np.float32)
            displacement = np.asarray(stored["displacement"], np.float64)
        if (displacement.shape != (len(xyz), 3) or not np.isfinite(displacement).all()
                or ids.shape != (len(xyz),) or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids)
                or xyz_sha256(xyz) != parent["CADs"][CAD]["xyz_sha256"]):
            raise ValueError("Training reference N/GRID identity/xyz order/physical u shape differs")
        alpha = alpha_from_original_features(features, key, guard)
        if not np.array_equal(alpha, np.eye(3, dtype=np.float32)[row]):
            raise ValueError("Fixed ver/hor/tor source does not match canonical response-basis coefficients")
        static = features.copy(); static[:, 7:13] = 0
        if row == 0:
            original_xyz, grid_ids, static_features = xyz, ids, static
        elif not np.array_equal(ids, grid_ids) or not np.array_equal(xyz, original_xyz) or not np.array_equal(static, static_features):
            raise ValueError("GRID/node order or nonload geometry/material/constraints vary across fixed loads")
        labels.append(displacement)
        source_index.append({"sample_key": key, **file_identity(path), "members_read": list(AUTHORIZED_MEMBERS),
                             "alpha": alpha.tolist(), "label_alignment": "official CSV GRID coordinates -> H5 vertices"})
    u_cache = load_response_cache(parent_directory, CAD, xyz=original_xyz, manifest=parent)
    strain_cache = load_strain_cache(strain_directory, CAD, xyz=original_xyz, expected_parent=parent["teacher_fingerprint"], manifest=manifest)
    raw_path = host_path(args.raw_meshes) / f"{CAD}.h5"
    source = manifest["CADs"][CAD]["raw_geometry_source"]
    if host_path(source["path"]) != raw_path:
        raise ValueError("Reference P1 must use the same original raw geometry as frozen predicted strain")
    verify_file_identity(raw_path, source)
    coordinates, cells = read_geometry(raw_path, u_cache.normalization, u_cache.xyz_sha256, u_cache.node_count)
    reference_u = np.stack(labels, axis=1)  # [originalN,ver/hor/tor,ux/uy/uz]
    predicted_u = u_cache.basis_scaled.astype(np.float64) * u_cache.scale_mm
    recovery_began = time.monotonic()
    reference_strain, diagnostics = recover_corner_p1_strain(coordinates, cells, reference_u / u_cache.scale_mm,
        u_scale_mm=u_cache.scale_mm, characteristic_length_mm=u_cache.normalization["characteristic_length_mm"],
        batch_size=args.batch_size, device="cpu")
    recovery_seconds = time.monotonic() - recovery_began
    if diagnostics["recovery"] != manifest["recovery"] or RECOVERY != manifest["recovery"]:
        raise ValueError("Reference recovery differs from the frozen predicted-strain estimator")
    per_load = {}
    for row, load in enumerate(BASIS_ORDER):
        u = vector_metrics(reference_u[:, row], predicted_u[:, row])
        epsilon = strain_metrics(reference_strain[:, row], strain_cache.basis_strain[:, row])
        per_load[load] = {"u": u, "strain_reference_P1_from_training_u": epsilon,
                          "strain_to_u_relative_error_ratio": epsilon["symmetric_Frobenius_relative_l2"] / u["vector_relative_l2"] if u["vector_relative_l2"] else None}
    aggregate = {"u_mean_vector_relative_l2": float(np.mean([v["u"]["vector_relative_l2"] for v in per_load.values()])),
                 "strain_mean_plain6_relative_l2": float(np.mean([v["strain_reference_P1_from_training_u"]["plain6_relative_l2"] for v in per_load.values()])),
                 "strain_mean_symmetric_Frobenius_relative_l2": float(np.mean([v["strain_reference_P1_from_training_u"]["symmetric_Frobenius_relative_l2"] for v in per_load.values()]))}
    aggregate["mean_error_ratio_strain_to_u"] = aggregate["strain_mean_symmetric_Frobenius_relative_l2"] / aggregate["u_mean_vector_relative_l2"]
    result = {"status": "completed", "scope": "predeclared fixed TRAIN101_479 ver/hor/tor diagnostic only", "CAD": CAD,
              "loads": list(BASIS_ORDER), "nodes": len(original_xyz), "tetrahedra": len(cells),
              "metrics": per_load, "aggregate": aggregate, "symmetric_tensor_weights": SYMMETRIC_WEIGHTS.tolist(),
              "units": {"u": "original signed components mm", "strain": "dimensionless raw physical tensor strain", "relative_l2": "fraction"},
              "reference_name": "reference P1 from training u", "recovery": RECOVERY,
              "reference_policy": "Original signed u converted using frozen common RMS and recovered with the exact same P1 geometry/unit rule; no source solver strain/stress tensor is read or inferred.",
              "source_index": {"training_labels": source_index, "raw_geometry": source, "raw_H5_members_read": ["vertices", "cells"],
                  "u_cache": u_cache.record(), "strain_cache": strain_cache.record(),
                  "parent_u_manifest": expected_parent, "strain_manifest": file_identity(strain_directory / "manifest.json", small_sha=True)},
              "alignment_checks": {"sample_and_label_alignment": True, "original_N_xyz_sha": u_cache.xyz_sha256,
                  "GRID_ids_unique_and_same_order_across_loads": True, "same_nonload_features": True, "canonical_unit_alphas": True,
                  "raw_H5_normalized_xyz_matches": True, "no_node_permutation": True},
              "reference_recovery_diagnostics": diagnostics, "cost": {"CPU_recovery_seconds": recovery_seconds,
                  "elapsed_seconds": time.monotonic() - began, "GPU_used": False, "retraining_updates": 0},
              "source_sha256": {Path(__file__).name: _sha(Path(__file__)), "strain_condition.py": _sha(ROOT / "geobc/strain_condition.py")},
              "stress_fields_read": 0, "validation_field_sources_opened": [], "dia_field_sources_opened": [], "final_test_field_sources_opened": [],
              "student_inputs_or_cache_modified": False, "new_scalers_or_corrections_fit": False,
              "limitations": "One fixed training CAD only; this compares a proxy to reference P1 from training u, not exact quadratic strain or solver GT tensors. Extra derivative error does not identify signed-stress exporter semantics or establish any validation/generalization claim.",
              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    output = host_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": "completed", "CAD": CAD, "aggregate": aggregate,
                      "per_load_error_ratio": {k: v["strain_to_u_relative_error_ratio"] for k, v in per_load.items()},
                      "CPU_seconds": result["cost"]["elapsed_seconds"], "output": str(output)}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u-response-cache", type=Path, default=Path("data/geobc_frozen_u_response"))
    parser.add_argument("--strain-response-cache", type=Path, default=Path("data/geobc_frozen_strain_response"))
    parser.add_argument("--data", type=Path, default=Path("data/prepared_geobc_multiload"))
    parser.add_argument("--raw-meshes", type=Path, default=Path("data/raw/FieldMesh"))
    parser.add_argument("--batch-size", type=int, default=262144)
    parser.add_argument("--output", type=Path, default=Path("notes/geobc_fixed_train_strain_quality.json"))
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    diagnose(args)


if __name__ == "__main__":
    main()
