"""Generate one frozen predicted-u response basis cache per train/validation CAD.

Only canonical ver xyz/features/key/alignment members are read. Displacement
and stress labels, dia archives, and final-test archives are never opened.
The completed manifest is written after releasing teacher CUDA tensors.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.name == "nt" and (ROOT / ".local_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".local_deps"))
import numpy as np

from geobc.displacement_condition import (BASIS_ORDER, COMPONENT_ORDER, FEATURE_NAMES, SCHEMA_VERSION,
    alpha_from_original_features, file_identity, host_path, json_sha256, load_response_manifest,
    teacher_fingerprint, verify_file_identity)
from geobc.local_geometry import load_local_geometry, normalized_xyz_array, xyz_sha256

PARENT_SOURCE_COMMIT = "ac2247683896eeeade7b7c310ba1ff003417ae8e"
INFERENCE_MODULES = ("physics_field.py", "conditioned_field.py", "tcnn_fields.py", "local_geometry.py")
SOURCE_MEMBERS = ("xyz", "features", "sample_key", "label_alignment")
GEOMETRY_METADATA_MEMBERS = ("characteristic_length_mm", "bbox_min_mm")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _preflight(args):
    """JSON/stat only: no torch import, checkpoint deserialization, or NPZ read."""
    gate_path = host_path(args.gate)
    gate = _json(gate_path)
    if gate.get("status") != "frozen" or gate.get("selection_uses_dia_or_stress_metrics") is not False:
        raise ValueError("A teacher selected only by known-load u validation must be frozen first")
    if (gate.get("cache_load") != "ver" or gate.get("student_u_features") != 3
            or gate.get("basis_order") != list(BASIS_ORDER) or gate.get("component_order") != list(COMPONENT_ORDER)
            or gate.get("final_test_excluded") is not True or gate.get("dia_excluded_from_this_stage") is not True
            or set(gate.get("cache_generation_label_members_forbidden", [])) != {"stress", "displacement"}):
        raise ValueError("Unexpected frozen cache axes/source/label-access contract")
    output = host_path(args.output)
    if output != host_path(gate["cache_directory"]):
        raise ValueError("Response cache output differs from its frozen gate")
    for key in ("split", "load_basis"):
        verify_file_identity(host_path(gate[key]["path"]), gate[key], small_sha=True)
    split = _json(host_path(gate["split"]["path"]))
    guard = _json(host_path(gate["load_basis"]["path"]))
    cads = split["groups"]["train"] + split["groups"]["validation"]
    if (len(split["groups"]["train"]) != 40 or len(split["groups"]["validation"]) != 5
            or len(set(cads)) != 45 or gate["cache_CADs"] != cads or set(cads) & set(split["groups"]["test"])
            or split["train_loads"] != list(BASIS_ORDER) or split["heldout_load"] != "dia"):
        raise ValueError("Cache scope must be exactly strict train40 + validation5, canonical ver")
    if (guard.get("compatible") is not True or guard.get("CADs_checked") != 45
            or set(r["cad_id"] for r in guard["cases"]) != set(cads)
            or guard["canonical_basis"]["order"] != list(BASIS_ORDER)):
        raise ValueError("The source rank3 metadata allowlist differs")
    teacher = gate["teacher"]
    for name in ("config", "summary", "metrics_best", "checkpoint", "geometry_clouds"):
        record = teacher[name]
        verify_file_identity(host_path(record["path"]), record, small_sha=name in ("config", "summary", "metrics_best"))
    run = host_path(teacher["run"])
    if any(host_path(teacher[k]["path"]) != run / filename for k, filename in
           (("config", "config.json"), ("summary", "summary.json"), ("metrics_best", "metrics_best.json"),
            ("checkpoint", "best.pt"), ("geometry_clouds", "geometry_clouds.npz"))):
        raise ValueError("Frozen teacher files differ from the selected run")
    config, summary, best = (_json(host_path(teacher[k]["path"])) for k in ("config", "summary", "metrics_best"))
    if (summary.get("status") != "completed" or summary["step"] != 10000 or summary["config"] != config
            or summary["best_metrics"] != best or teacher["best_step"] != summary["best_step"]):
        raise ValueError("Teacher completed config/summary/best metadata differs")
    if best["step"] != teacher["best_step"] or best["selection_score"] != teacher["known_validation_u_mean_relative_l2"] or best["selection_score"] != best["validation"]["case_mean_relative_l2"]:
        raise ValueError("Teacher is not the frozen known-load validation-selected checkpoint")
    if datetime.fromisoformat(summary["completed_at_utc"]) > datetime.fromisoformat(gate["created_at_utc"]):
        raise ValueError("Teacher completed after it was frozen")
    if "validation" not in config["checkpoint_selection"] or config["groups"] != split["groups"] or config["train_loads"] != list(BASIS_ORDER) or config["heldout_load"] != "dia":
        raise ValueError("Teacher trained/selected on an unexpected split or load")
    expected_keys = {g: [f"{c}/{l}" for c in split["groups"][g] for l in BASIS_ORDER] for g in ("train", "validation")}
    for group, keys in expected_keys.items():
        if [c["sample_key"] for c in config["cases"][group]] != keys:
            raise ValueError("Teacher train/validation archive allowlist differs")
    scale = teacher["scaler"]
    if (scale != config["scaler"] or scale["center_mm"] != 0 or scale["transform"] != "linear_zero_center_common_scalar"
            or scale["fit_case_keys"] != expected_keys["train"] or scale["fit_nodes"] != sum(c["nodes"] for c in config["cases"]["train"])
            or not math.isfinite(scale["scale_mm"]) or scale["scale_mm"] <= 0):
        raise ValueError("Teacher common scale must be fixed zero-centered train120-only")
    if (teacher["model"] != config["model"] or config["physics_mode"] != "linear_basis"
            or teacher["model"]["mode"] != "linear_basis" or teacher["model"]["geometry_mode"] != "shape"
            or teacher["model"]["hidden_dim"] != 128 or teacher["model"]["condition_dim"] != 38
            or teacher["geometry"] != config["geometry"] or teacher["local_geometry"] != config["local_geometry"]
            or config["local_geometry"]["mode"] != "actual" or teacher["u_loss"] != config["u_loss"]["name"]):
        raise ValueError("Frozen teacher must be the shared shape/width128/local13 linear-basis model")
    if (config["contract"]["split_sha256"] != gate["split"]["sha256"]
            or config["contract"]["load_basis"]["sha256"] != gate["load_basis"]["sha256"]
            or teacher["source_code_sha256"] != config["contract"]["code_sha256"]):
        raise ValueError("Frozen teacher parent split/basis/code provenance differs")
    current_source_sha = {name: _sha(ROOT / "geobc" / name) for name in INFERENCE_MODULES}
    if any(current_source_sha[n] != teacher["source_code_sha256"][n] for n in INFERENCE_MODULES):
        raise ValueError("A frozen inference module changed; historical conditional_train SHA is provenance only")
    candidates = gate["candidates"]
    for candidate in candidates:
        record = _json(host_path(candidate["run"]) / "metrics_best.json")
        if record["step"] != candidate["best_step"] or record["validation"]["case_mean_relative_l2"] != candidate["known_validation_u_mean_relative_l2"]:
            raise ValueError("A predeclared teacher candidate's known-load selection metric changed")
    selected = min(range(len(candidates)), key=lambda i: (candidates[i]["known_validation_u_mean_relative_l2"], i))
    if host_path(candidates[selected]["run"]) != run:
        raise ValueError("Frozen teacher differs from the predeclared known-load selection rule")
    data = host_path(config["arguments"]["data"])
    local_directory = host_path(config["arguments"]["local_geometry"])
    source_records = {r["key"]: r for r in config["contract"]["selected_sources"]}
    if set(source_records) != set(expected_keys["train"] + expected_keys["validation"]):
        raise ValueError("Teacher source identities include unapproved cases")
    canonical_sources = {}
    for cad in cads:
        key = f"{cad}/ver"
        path = host_path(split["case_paths"][key])
        source = source_records[key]
        if host_path(source["path"]) != path or not path.is_relative_to(data):
            raise ValueError("Canonical geometry escaped the approved teacher source directory")
        verify_file_identity(path, source)
        canonical_sources[cad] = source
        local = teacher["local_geometry"]["CADs"][cad]
        if host_path(local["cache"]) != local_directory / f"{cad}.npz" or local["target_independent"] is not True:
            raise ValueError("Canonical target-independent local cache differs")
        verify_file_identity(host_path(local["cache"]), local)
    return {"gate": gate, "gate_path": str(gate_path), "gate_sha256": _sha(gate_path), "config": config,
            "summary": summary, "best": best, "split": split, "guard": guard, "CADs": cads,
            "teacher_fingerprint": teacher_fingerprint(gate), "output": str(output),
            "canonical_sources": canonical_sources, "local_directory": str(local_directory),
            "inference_source_sha256": current_source_sha}


def _read_inputs(path, key):
    # Do not materialize other members of this prepared archive.
    with np.load(path, allow_pickle=False) as stored:
        if str(stored["sample_key"].item()) != key or str(stored["label_alignment"].item()) != "official CSV GRID coordinates -> H5 vertices":
            raise ValueError("Canonical source key or CSV GRID alignment differs")
        xyz = normalized_xyz_array(stored["xyz"])
        features = np.asarray(stored["features"], np.float32)
        normalization = {"status": "unknown", "characteristic_length_mm": None, "bbox_min_mm": None,
                         "source": "only existing canonical prepared geometry metadata; no H5/raw geometry scan"}
        accessed = list(SOURCE_MEMBERS)
        if "characteristic_length_mm" in stored.files:
            length = float(stored["characteristic_length_mm"].item())
            if not math.isfinite(length) or length <= 0:
                raise ValueError("Invalid existing isotropic characteristic_length_mm")
            normalization.update(status="length_known", characteristic_length_mm=length)
            accessed.append("characteristic_length_mm")
        if "bbox_min_mm" in stored.files:
            origin = np.asarray(stored["bbox_min_mm"], np.float64)
            if origin.shape != (3,) or not np.isfinite(origin).all():
                raise ValueError("Invalid existing bbox_min_mm")
            normalization["bbox_min_mm"] = origin.tolist()
            accessed.append("bbox_min_mm")
        if normalization["characteristic_length_mm"] is not None and normalization["bbox_min_mm"] is not None:
            normalization.update(status="known", definition="existing isotropic xyz=(xyz_mm-bbox_min_mm)/characteristic_length_mm")
    if features.shape != (len(xyz), 25) or not np.isfinite(features).all():
        raise ValueError("Canonical response inputs must be finite original25 features")
    return xyz, features, normalization, accessed


def generate(args, metadata):
    import torch
    from geobc.physics_field import PhysicsField, PhysicsFieldConfig
    output = host_path(args.output)
    if (output / "manifest.json").is_file():
        existing = load_response_manifest(output, expected_teacher=metadata["teacher_fingerprint"])
        for cad, record in existing["CADs"].items():
            verify_file_identity(output / f"{cad}.npz", record)
        print(json.dumps({"status": "completed", "reused": True, "output": str(output)}), flush=True)
        return existing
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Nonempty incomplete response output is not silently overwritten")
    output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    status = {"phase": "loading_frozen_teacher", "completed_CADs": 0, "total_CADs": 45}
    def update(phase, **extra):
        status.update(extra, phase=phase, elapsed_seconds=time.monotonic() - began,
                      updated_at_utc=datetime.now(timezone.utc).isoformat())
        done = status["completed_CADs"]
        status["eta_seconds"] = status["elapsed_seconds"] / done * (45 - done) if done else None
        _write(output / "status.json", status)
    update("loading_frozen_teacher")
    try:
        device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("Native frozen teacher generation requires CUDA; --dry-run uses metadata only")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        teacher, cfg = metadata["gate"]["teacher"], metadata["config"]
        model_config = PhysicsFieldConfig(**{f.name: cfg["model"][f.name] for f in fields(PhysicsFieldConfig)})
        model = PhysicsField(model_config, device=device)
        if model.source_config != cfg["model"]:
            raise ValueError("Current teacher factory differs from its frozen architecture")
        saved = torch.load(host_path(teacher["checkpoint"]["path"]), map_location="cpu", weights_only=False)
        verify_file_identity(host_path(teacher["checkpoint"]["path"]), teacher["checkpoint"])
        if saved["config"] != cfg or saved["step"] != teacher["best_step"] or saved["metrics"] != metadata["best"] or saved["scaler"] != teacher["scaler"]:
            raise ValueError("Teacher weights are not the declared completed known-load best checkpoint")
        model.load_state_dict(saved["model"])
        del saved
        model.eval().requires_grad_(False)
        with np.load(host_path(teacher["geometry_clouds"]["path"]), allow_pickle=False) as stored:
            cloud, mask, cloud_keys = stored["points"], stored["mask"], stored["cad_keys"].tolist()
        if (len(cloud_keys) != 45 or len(set(cloud_keys)) != 45 or set(cloud_keys) != set(metadata["CADs"])
                or cloud.shape != (45, model_config.cloud_points, 3) or mask.shape != cloud.shape[:2]
                or _sha(host_path(metadata["gate_path"])) != metadata["gate_sha256"]
                or hashlib.sha256(cloud.tobytes()).hexdigest() != cfg["geometry"]["coordinates_sha256"]
                or hashlib.sha256(mask.tobytes()).hexdigest() != cfg["geometry"]["mask_sha256"]):
            raise ValueError("Saved frozen geometry clouds/scope changed")
        verify_file_identity(host_path(teacher["geometry_clouds"]["path"]), teacher["geometry_clouds"])
        with torch.no_grad():
            contexts = model.encode_geometry(torch.as_tensor(cloud, device=device), torch.as_tensor(mask, device=device))
        row_by_cad = {cad: row for row, cad in enumerate(cloud_keys)}
        config = {"gate": {"path": metadata["gate_path"], "sha256": metadata["gate_sha256"]},
                  "teacher_fingerprint": metadata["teacher_fingerprint"], "teacher": teacher,
                  "parent_source_commit": PARENT_SOURCE_COMMIT, "inference_source_sha256": metadata["inference_source_sha256"],
                  "generator_source_sha256": {Path(__file__).name: _sha(Path(__file__)),
                                               "displacement_condition.py": _sha(ROOT / "geobc/displacement_condition.py")},
                  "source_members_read": list(SOURCE_MEMBERS), "optional_geometry_metadata_members": list(GEOMETRY_METADATA_MEMBERS),
                  "label_members_read": [], "optimizer_updates": 0,
                  "precision": {"matmul": "high", "cuda_tf32": True, "cudnn_tf32": True,
                                "decoder": "frozen native TCNN FP16 compute/FP32 master parameters/float32 output"},
                  "batch_size": args.batch_size, "basis_order": list(BASIS_ORDER), "component_order": list(COMPONENT_ORDER),
                  "cost_policy": metadata["gate"]["cost_policy"], "limitations": metadata["gate"]["limitations"]}
        _write(output / "config.json", config)
        records = {}
        def one_cad(cad):
            source = metadata["canonical_sources"][cad]
            path = host_path(source["path"])
            verify_file_identity(path, source)
            xyz, features, normalization, accessed = _read_inputs(path, f"{cad}/ver")
            alpha_from_original_features(features, f"{cad}/ver", metadata["guard"])
            local_record = teacher["local_geometry"]["CADs"][cad]
            local = load_local_geometry(metadata["local_directory"], cad, xyz=xyz)
            actual = local.record()
            verify_file_identity(local.path, local_record)
            for key in ("node_count", "xyz_sha256", "schema_sha256", "feature_names", "target_independent"):
                if actual[key] != local_record[key]:
                    raise ValueError("Frozen local geometry metadata/input changed")
            basis = np.empty((len(xyz), 3, 3), np.float32)
            context = contexts[row_by_cad[cad]:row_by_cad[cad] + 1]
            with torch.no_grad():
                for start in range(0, len(xyz), args.batch_size):
                    stop = min(start + args.batch_size, len(xyz))
                    tx = torch.as_tensor(xyz[start:stop], device=device)
                    condition = torch.as_tensor(np.concatenate((features[start:stop], local.features[start:stop]), axis=1), device=device)
                    predicted = model.response_basis(tx, condition, context=context)
                    basis[start:stop] = predicted.float().cpu().numpy()
            if not np.isfinite(basis).all():
                raise FloatingPointError(f"{cad}: nonfinite frozen predicted response basis")
            sha = xyz_sha256(xyz)
            fingerprint = json_sha256({"teacher": metadata["teacher_fingerprint"], "xyz_sha256": sha,
                                      "canonical_source": {k: source[k] for k in ("bytes", "mtime_ns")},
                                      "local_cache": {k: actual[k] for k in ("bytes", "mtime_ns", "schema_sha256")},
                                      "normalization": normalization})
            target = output / f"{cad}.npz"
            temporary = target.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez(handle, basis_scaled=basis, cad_id=np.asarray(cad), node_count=np.asarray(len(xyz)), xyz_sha256=np.asarray(sha),
                         basis_order=np.asarray(BASIS_ORDER), component_order=np.asarray(COMPONENT_ORDER),
                         scale_mm=np.asarray(teacher["scaler"]["scale_mm"]), center_mm=np.asarray(0.),
                         schema_version=np.asarray(SCHEMA_VERSION), teacher_fingerprint=np.asarray(metadata["teacher_fingerprint"]),
                         gate_sha256=np.asarray(metadata["gate_sha256"]), response_fingerprint=np.asarray(fingerprint),
                         normalization_json=np.asarray(json.dumps(normalization, sort_keys=True)))
            temporary.replace(target)
            return {"file": target.name, **{k: v for k, v in file_identity(target).items() if k != "path"},
                    "node_count": len(xyz), "xyz_sha256": sha, "response_fingerprint": fingerprint,
                    "canonical_source": {**source, "members_read": accessed}, "local_cache": actual, "normalization": normalization,
                    "basis_shape": list(basis.shape), "dtype": "float32", "finite": True}
        for cad in metadata["CADs"]:
            update("generating_frozen_basis", current_CAD=cad)
            records[cad] = one_cad(cad)
            status["completed_CADs"] += 1
            update("generating_frozen_basis")
        # Release all model/context CUDA references before unblocking students.
        del one_cad, model, contexts
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        if _sha(host_path(metadata["gate_path"])) != metadata["gate_sha256"]:
            raise ValueError("Teacher gate changed during generation")
        manifest = {"status": "completed", "complete": True, "schema_version": SCHEMA_VERSION,
                    "gate": config["gate"], "teacher_fingerprint": metadata["teacher_fingerprint"], "teacher": teacher,
                    "parent_source_commit": PARENT_SOURCE_COMMIT, "source_config": config,
                    "split_sha256": metadata["gate"]["split"]["sha256"], "load_basis_sha256": metadata["gate"]["load_basis"]["sha256"],
                    "CADs": records, "CAD_count": len(records), "node_count": sum(r["node_count"] for r in records.values()),
                    "total_bytes": sum(r["bytes"] for r in records.values()), "scale_mm": teacher["scaler"]["scale_mm"], "center_mm": 0.,
                    "basis_order": list(BASIS_ORDER), "component_order": list(COMPONENT_ORDER), "student_feature_names": list(FEATURE_NAMES),
                    "label_members_read": [], "dia_sources_opened": [], "final_test_sources_opened": [], "optimizer_updates": 0,
                    "teacher_GPU_released_before_manifest": True, "elapsed_seconds": time.monotonic() - began,
                    "cost": {"cache_generation_elapsed_seconds": time.monotonic() - began,
                             "teacher_training_elapsed_seconds": teacher["training_elapsed_seconds"],
                             "candidate_training_costs": [{"run": c["run"], "elapsed_seconds": c["training_elapsed_seconds"],
                                                           "u_loss": c["u_loss"]} for c in metadata["gate"]["candidates"]]},
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "supervision_note": "Cache reads no labels, but the frozen teacher learned from train u supervision; this is extra supervised information and pretraining cost."}
        _write(output / "manifest.json", manifest)
        update("completed", teacher_GPU_released=True)
        print(json.dumps({"status": "completed", "output": str(output), "CADs": 45,
                          "nodes": manifest["node_count"], "cache_bytes": manifest["total_bytes"],
                          "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
        return manifest
    except Exception as exc:
        update("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=65536)
    p.add_argument("--dry-run", action="store_true", help="JSON/stat only; no arrays, checkpoints, CUDA, or labels")
    args = p.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    metadata = _preflight(args)
    if args.dry_run:
        print(json.dumps({"status": "frozen_metadata_ready", "CADs": len(metadata["CADs"]),
                          "teacher_fingerprint": metadata["teacher_fingerprint"], "best_step": metadata["gate"]["teacher"]["best_step"],
                          "checkpoint_opened": False, "NPZ_arrays_opened": 0, "GPU_used": False}, indent=2))
        return
    generate(args, metadata)


if __name__ == "__main__":
    main()
