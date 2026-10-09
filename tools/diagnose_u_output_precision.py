#!/usr/bin/env python3
"""Fixed TRAIN101_479 oracle output-quantization sensitivity, CPU only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Limit this tiny diagnostic to one CPU BLAS thread before importing NumPy.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.name == "nt" and (ROOT / ".local_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".local_deps"))
import numpy as np

from geobc.displacement_condition import file_identity, host_path, verify_file_identity
from geobc.local_geometry import xyz_sha256
from geobc.strain_condition import RECOVERY, recover_corner_p1_strain
from tools.diagnose_train_strain_quality import strain_metrics, vector_metrics
from tools.prepare_frozen_strain_response import read_geometry

CAD = "101_479"
LOADS = ("ver", "hor", "tor")
ALIGNMENT = "official CSV GRID coordinates -> H5 vertices"
MEMBERS = ("sample_key", "label_alignment", "u_units", "xyz", "grid_node_ids", "displacement")


def small_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(output: Path, include_fp32: bool = True) -> dict:
    began = time.monotonic()
    quality_path = ROOT / "notes/geobc_fixed_train_strain_quality.json"
    quality = small_json(quality_path)
    if quality["status"] != "completed" or quality["CAD"] != CAD or quality["loads"] != list(LOADS):
        raise ValueError("Require the completed predeclared fixed-training-CAD diagnostic")
    split = small_json(ROOT / "notes/geobc_split.json")
    if CAD not in split["groups"]["train"] or CAD in split["groups"]["validation"] + split["groups"]["test"]:
        raise ValueError("Oracle sensitivity is restricted to TRAIN101_479")
    parent = quality["source_index"]["u_cache"]  # JSON metadata, not predicted cache arrays.
    scale = float(parent["scale_mm"])
    normalization = parent["normalization"]
    length = float(normalization["characteristic_length_mm"])
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(length) or length <= 0:
        raise ValueError("Require the existing positive fixed common physical scales")
    source_records = {r["sample_key"]: r for r in quality["source_index"]["training_labels"]}
    labels, source_index = [], []
    canonical_xyz = canonical_ids = None
    for load in LOADS:
        key = f"{CAD}/{load}"
        path = host_path(split["case_paths"][key])
        expected = source_records[key]
        if path != host_path(expected["path"]) or path.parent != ROOT / "data/prepared_geobc_multiload":
            raise ValueError("Original fixed training source differs from the prior diagnostic")
        verify_file_identity(path, expected)
        with np.load(path, allow_pickle=False) as stored:
            if str(stored["sample_key"].item()) != key or str(stored["label_alignment"].item()) != ALIGNMENT:
                raise ValueError("Require original GRID-aligned training displacement")
            if str(stored["u_units"].item()) != "mm":
                raise ValueError("Require physical signed displacement components in mm")
            xyz = np.asarray(stored["xyz"], np.float32)
            ids = np.asarray(stored["grid_node_ids"])
            u = np.asarray(stored["displacement"], np.float64)
        if (u.shape != (parent["node_count"], 3) or not np.isfinite(u).all()
                or ids.shape != (len(u),) or xyz_sha256(xyz) != parent["xyz_sha256"]):
            raise ValueError("Fixed source N/physical u/xyz identity differs")
        if canonical_xyz is None:
            canonical_xyz, canonical_ids = xyz, ids
        elif not np.array_equal(xyz, canonical_xyz) or not np.array_equal(ids, canonical_ids):
            raise ValueError("Fixed training-load node order differs")
        labels.append(u)
        source_index.append({"sample_key": key, **file_identity(path), "members_read": list(MEMBERS)})
    true_u = np.stack(labels, axis=1)  # [N,ver/hor/tor,ux/uy/uz]
    raw_source = quality["source_index"]["raw_geometry"]
    raw_path = ROOT / "data/raw/FieldMesh" / f"{CAD}.h5"
    if host_path(raw_source["path"]) != raw_path:
        raise ValueError("Raw canonical geometry differs")
    verify_file_identity(raw_path, raw_source)
    coordinates, cells = read_geometry(raw_path, normalization, parent["xyz_sha256"], parent["node_count"])
    recovery_args = dict(u_scale_mm=scale, characteristic_length_mm=length,
                         batch_size=262144, device="cpu")
    reference_strain, diagnostics = recover_corner_p1_strain(coordinates, cells, true_u / scale, **recovery_args)
    if diagnostics["recovery"] != quality["recovery"] or diagnostics["recovery"] != RECOVERY:
        raise ValueError("Reference P1 recovery differs from the fixed prior diagnostic")
    methods = [("fp16_oracle_roundtrip", np.float16)]
    if include_fp32:
        methods.append(("fp32_oracle_roundtrip", np.float32))
    results = {}
    for name, dtype in methods:
        # Exact requested oracle sequence. FP64 source division, output cast,
        # promote native half to float32, decode using the unchanged common scale.
        roundtrip = (true_u / scale).astype(dtype).astype(np.float32) * scale
        if not np.isfinite(roundtrip).all():
            raise FloatingPointError("Nonfinite oracle roundtrip; no clipping applied")
        recovered, method_diagnostics = recover_corner_p1_strain(
            coordinates, cells, roundtrip.astype(np.float64) / scale, **recovery_args)
        per_load = {
            load: {
                "u": vector_metrics(true_u[:, row], roundtrip[:, row]),
                "strain_reference_P1_from_training_u": strain_metrics(reference_strain[:, row], recovered[:, row]),
            }
            for row, load in enumerate(LOADS)
        }
        results[name] = {
            "formula": f"(true_u_mm/common_scale_mm).astype({np.dtype(dtype).name}).astype(float32)*common_scale_mm",
            "roundtrip_physical_dtype": str(roundtrip.dtype),
            "per_load": per_load,
            "u_mean_vector_relative_l2": float(np.mean([r["u"]["vector_relative_l2"] for r in per_load.values()])),
            "strain_mean_symmetric_Frobenius_relative_l2": float(np.mean([
                r["strain_reference_P1_from_training_u"]["symmetric_Frobenius_relative_l2"] for r in per_load.values()])),
            "oracle_roundtrip_u_sha256": hashlib.sha256(np.ascontiguousarray(roundtrip, dtype="<f4").tobytes()).hexdigest(),
            "recovery_diagnostics": method_diagnostics,
        }
    previous = quality["aggregate"]
    q = results["fp16_oracle_roundtrip"]
    oracle_fraction = q["strain_mean_symmetric_Frobenius_relative_l2"] / previous["strain_mean_symmetric_Frobenius_relative_l2"]
    build = small_json(ROOT / "notes/tcnn_build.json")
    if oracle_fraction < 0.1:
        conclusion = (
            "For this fixed training CAD and its three loads, rounding perfect source u to the native output type produces "
            "much smaller P1 strain error than the previously observed learned-field error. Output rounding alone does not "
            "account for the observed magnitude in this oracle comparison. This supports investigating learned-field variation "
            "and gradient optimization, without proving a unique cause or excluding native arithmetic/input-precision effects."
        )
    else:
        conclusion = (
            "Oracle output rounding is not negligible relative to the previously observed P1 error on this fixed CAD. "
            "This sensitivity comparison cannot assign a causal share or explain native training behavior by itself."
        )
    result = {
        "status": "completed", "CAD": CAD, "loads": list(LOADS),
        "scope": "Predeclared fixed TRAIN101_479 source-u oracle quantization sensitivity only; CPU, no model prediction.",
        "nodes": len(true_u), "tetrahedra": len(cells), "common_u_scale_mm": scale,
        "normalization": normalization, "physical_gradient_factor": scale / length,
        "xyz_sha256": parent["xyz_sha256"], "tensor_weights": [1, 1, 1, 2, 2, 2],
        "recovery": RECOVERY, "reference_recovery_diagnostics": diagnostics,
        "methods": results,
        "prior_learned_field_diagnostic_JSON_only": {
            "source": quality_path.relative_to(ROOT).as_posix(), "sha256": sha(quality_path),
            "u_mean_vector_relative_l2": previous["u_mean_vector_relative_l2"],
            "strain_mean_symmetric_Frobenius_relative_l2": previous["strain_mean_symmetric_Frobenius_relative_l2"],
            "fp16_oracle_to_prior_strain_error_ratio": oracle_fraction,
            "not_a_causal_error_decomposition": True,
        },
        "native_output_source_evidence": {
            "check": "Source/build-metadata only; no native module import or GPU execution.",
            "commit": build["commit"], "build_half_precision": build["half_precision"],
            "official_source_root": build["source_checkout"],
            "source_locations_read": {
                "include/tiny-cuda-nn/common.h:122-125": "TCNN_HALF_PRECISION selects network_precision_t=__half rather than float.",
                "src/cpp_api.cu:157": "NetworkWithInputEncoding is instantiated with network_precision_t.",
                "bindings/torch/tinycudann/bindings.cpp:100": "Forward output tensor uses c10_output_precision().",
                "bindings/torch/tinycudann/modules.py:107-111": "Fp16 maps to torch.half; Fp32 maps to torch.float.",
                "geobc/tcnn_fields.py:129": "The wrapper promotes the native output using .float(), which cannot restore rounded bits.",
            },
            "runtime_output_dtype_not_rechecked": True,
        },
        "conclusion": conclusion,
        "limitations": [
            "Perfect-source-u rounding is an oracle sensitivity test, not a rigorous attainable-error lower bound, model prediction, or native mixed-precision simulation.",
            "FP16 internal activations/encoding/input arithmetic, learned high-frequency errors, geometry discretization and optimization can add other effects. This test isolates only the requested final-output roundtrip.",
            "The P1 reference is derived from training u with the existing operator, not an exact TETRA10 strain or solver GT stress tensor. No signed scalar-stress formula is tested.",
            "One fixed training CAD only; no generalization or new architecture/loss conclusion follows.",
        ],
        "source_index": {"training_u": source_index, "raw_geometry": raw_source,
                         "raw_H5_members_read": ["vertices", "cells"],
                         "teacher_scale_source": "prior completed diagnostic u_cache JSON record, no cache array read"},
        "source_sha256": {
            Path(__file__).name: sha(Path(__file__)),
            "strain_condition.py": sha(ROOT / "geobc/strain_condition.py"),
            "diagnose_train_strain_quality.py": sha(ROOT / "tools/diagnose_train_strain_quality.py"),
            "prepare_frozen_strain_response.py": sha(ROOT / "tools/prepare_frozen_strain_response.py"),
        },
        "elapsed_seconds": time.monotonic() - began,
        "GPU_used": False, "model_or_teacher_prediction_arrays_read": [], "weights_read": [],
        "stress_arrays_read": [], "validation_or_dia_or_final_test_arrays_read": [],
        "scalers_fit": False, "existing_helpers_or_cache_or_training_modified": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "notes/geobc_u_output_precision.json")
    parser.add_argument("--skip-fp32", action="store_true")
    args = parser.parse_args()
    result = diagnose(host_path(args.output), not args.skip_fp32)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
        "methods": {name: {k: r[k] for k in ("u_mean_vector_relative_l2", "strain_mean_symmetric_Frobenius_relative_l2")}
                    for name, r in result["methods"].items()},
        "oracle_to_prior_strain_error_ratio": result["prior_learned_field_diagnostic_JSON_only"]["fp16_oracle_to_prior_strain_error_ratio"],
        "conclusion": result["conclusion"], "output": str(host_path(args.output))}, indent=2))


if __name__ == "__main__":
    main()
