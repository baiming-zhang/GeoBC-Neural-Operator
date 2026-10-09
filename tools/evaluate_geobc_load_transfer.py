"""One predeclared frozen load-span probe on five validation CADs.

planned gates cannot open checkpoints or dia fields. A frozen gate identifies
completed ver/hor/tor-selected physics models and an independent signed-stress
model before any dia labels are accessed. No fitting/updates/model selection
occur here, and the five final-test CADs are never opened.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
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

MODEL_NAMES = ("linear_basis", "load_dependent_basis", "stress")
BASIS_ORDER = ("ver", "hor", "tor")


def _path(value):
    # Same host/WSL path convention as the frozen conditional trainer, without
    # importing torch/models during a metadata-only planned dry run.
    value = str(value).replace("\\", "/")
    if os.name == "nt" and value.startswith("/mnt/") and len(value) > 7:
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and len(value) > 2 and value[1:3] == ":/":
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    p = Path(value)
    return (p if p.is_absolute() else ROOT / p).resolve()


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path):
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": None, "sha256_status": "not hashed: checkpoint/geometry arrays are not scanned for unrelated provenance"}


def _check_identity(path, expected):
    """Cheap identity check; no multi-GB checkpoint/field hashing."""
    actual = _identity(path)
    if actual["bytes"] != expected["bytes"]:
        raise ValueError(f"{path}: frozen file size changed")
    if "mtime_ns" in expected and actual["mtime_ns"] != expected["mtime_ns"]:
        raise ValueError(f"{path}: frozen file modification time changed")
    if "last_write_time_utc" in expected:
        declared = datetime.fromisoformat(expected["last_write_time_utc"]).timestamp()
        # .NET records seven decimal places; datetime has microsecond precision.
        if abs(path.stat().st_mtime - declared) > 1e-6:
            raise ValueError(f"{path}: frozen UTC modification time changed")
    return actual


def _write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def displacement_metrics(truth, prediction):
    truth, prediction = np.asarray(truth, np.float64), np.asarray(prediction, np.float64)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 3 or not len(truth):
        raise ValueError("Signed displacement metrics require matching nonempty [N,3] fields")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite physical displacement")
    error = prediction - truth
    norm = np.linalg.norm(error, axis=1)
    return {"nodes": len(truth), "relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(truth), 1e-12)),
            "rmse_mm": float(np.sqrt(np.mean(error ** 2))),
            "vector_rmse_mm": float(np.sqrt(np.mean(norm ** 2))),
            "max_vector_error_mm": float(norm.max())}


def summarize_vectors(truths, predictions, cad_keys):
    if not truths or len(truths) != len(predictions) or len(truths) != len(cad_keys) or len(set(cad_keys)) != len(cad_keys):
        raise ValueError("Case metrics require one matched prediction per unique CAD")
    per = {f"{cad}/dia": displacement_metrics(t, p) for cad, t, p in zip(cad_keys, truths, predictions)}
    worst = max(per, key=lambda k: per[k]["relative_l2"])
    return {"per_case": per, "pooled": displacement_metrics(np.concatenate(truths), np.concatenate(predictions)),
            "case_mean_relative_l2": float(np.mean([v["relative_l2"] for v in per.values()])),
            "worst_case_relative_l2": per[worst]["relative_l2"], "worst_case_key": worst}


def oracle_superposition(reference_u_mm, alpha):
    """Source signed-mm fields only, never source scalar stress."""
    values, alpha = np.asarray(reference_u_mm, np.float64), np.asarray(alpha, np.float64)
    if values.ndim != 3 or values.shape[0] != 3 or values.shape[2] != 3 or alpha.shape != (3,):
        raise ValueError("Oracle needs [ver/hor/tor,N,xyz] mm fields and alpha[3]")
    if not np.isfinite(values).all() or not np.isfinite(alpha).all():
        raise ValueError("Nonfinite source oracle fields/coefficients")
    return np.einsum("l,lnc->nc", alpha, values)


def _metadata(args):
    gate_path = _path(args.gate)
    gate = _json(gate_path)
    split_path = _path(gate["split"])
    split = _json(split_path)
    cads = gate["validation_cads"]
    if cads != split["groups"]["validation"] or len(cads) != 5 or len(set(cads)) != 5:
        raise ValueError("Gate must list exactly the five validation CADs in the locked split order")
    if set(cads) & set(split["groups"]["test"]) or gate.get("heldout_load") != "dia":
        raise ValueError("Final-test fields are excluded; only validation dia is allowed")
    if split["train_loads"] != list(BASIS_ORDER) or split.get("heldout_load") != "dia":
        raise ValueError("Unexpected source train/heldout loads")
    output = _path(args.output)
    if output != _path(gate["output"]):
        raise ValueError("Output differs from the predeclared gate")
    if set(gate["frozen_models"]) != set(MODEL_NAMES):
        raise ValueError("Gate must name the constrained/control physics pair and one frozen stress model")
    result = {"gate": gate, "gate_path": str(gate_path), "gate_sha256": _sha(gate_path),
              "split": split, "split_path": str(split_path), "split_sha256": _sha(split_path),
              "validation_cads": cads, "output": str(output), "models": {}, "ready": False}
    if gate.get("status") != "frozen":
        if gate.get("status") != "planned":
            raise ValueError("Gate status must be planned or frozen")
        return result  # No checkpoint, geometry array, or dia field is opened.
    frozen_at = gate.get("frozen_at_utc")
    if not frozen_at:
        raise ValueError("A frozen gate must record frozen_at_utc")
    freeze_time = datetime.fromisoformat(frozen_at)
    if freeze_time.tzinfo is None:
        raise ValueError("Frozen gate timestamp needs an explicit timezone")
    if gate.get("holdout_opened_at_freeze") is not False:
        raise ValueError("Frozen gate must declare held-out targets unopened at freezing")
    expected_keys = {group: [f"{cad}/{load}" for cad in split["groups"][group] for load in BASIS_ORDER]
                     for group in ("train", "validation")}
    lock = gate["locked_config"]
    for name in MODEL_NAMES:
        entry = gate["frozen_models"][name]
        run = _path(entry["run"])
        config_path, summary_path = run / "config.json", run / "summary.json"
        if any(_path(entry[k]) != expected for k, expected in
               (("config", config_path), ("summary", summary_path), ("checkpoint", run / "best.pt"))):
            raise ValueError(f"{name}: explicit frozen source path differs from its run")
        config, summary, best = _json(config_path), _json(summary_path), _json(run / "metrics_best.json")
        if entry.get("config_sha256") != _sha(config_path) or entry.get("summary_sha256") != _sha(summary_path):
            raise ValueError(f"{name}: configuration/summary changed after freezing")
        if summary.get("status") != "completed" or summary.get("step") != lock["steps"]:
            raise ValueError(f"{name}: the predeclared training budget is not completed")
        if datetime.fromisoformat(summary["completed_at_utc"]) > freeze_time:
            raise ValueError(f"{name}: training completed after the declared freeze")
        if config != summary["config"] or best != summary["best_metrics"]:
            raise ValueError(f"{name}: durable config/summary/best-metrics mismatch")
        if best["step"] != entry.get("best_step") or best["step"] != summary["best_step"]:
            raise ValueError(f"{name}: selected best step differs from the frozen gate")
        if best["selection_score"] != best["validation"]["case_mean_relative_l2"] or "validation" not in config["checkpoint_selection"]:
            raise ValueError(f"{name}: checkpoint was not selected by ver/hor/tor validation case mean")
        if entry["selection"] != config["checkpoint_selection"]:
            raise ValueError(f"{name}: frozen checkpoint-selection rule differs")
        if config["groups"] != split["groups"] or config["train_loads"] != list(BASIS_ORDER) or config["heldout_load"] != "dia":
            raise ValueError(f"{name}: split or fitted loads differ")
        for group in expected_keys:
            if [c["sample_key"] for c in config["cases"][group]] != expected_keys[group]:
                raise ValueError(f"{name}: train/validation archive allowlist differs")
        for key in ("seed", "steps", "variant", "hidden_dim", "local_mode", "batch_size", "cases_per_batch", "learning_rate"):
            if config["arguments"].get(key) != lock[key]:
                raise ValueError(f"{name}: predeclared {key} changed")
        if _path(config["arguments"]["data"]) != _path(gate["data"]) or _path(config["arguments"]["local_geometry"]) != _path(gate["local_geometry"]):
            raise ValueError(f"{name}: frozen prepared data/local geometry directory differs")
        if config["model"]["condition_dim"] != 38 or config["model"]["geometry_mode"] != "shape":
            raise ValueError("This gate requires the frozen shape+actual-local13 input layout")
        if name != "stress" and config.get("physics_mode") != name:
            raise ValueError("Physics pair modes were swapped")
        checkpoint = run / "best.pt"
        clouds = run / "geometry_clouds.npz"
        if not checkpoint.is_file() or not clouds.is_file():
            raise FileNotFoundError(f"{name}: missing frozen checkpoint/geometry cloud")
        checkpoint_identity = _check_identity(checkpoint, entry["checkpoint_identity"])
        scale = config["scaler"]
        if name == "stress":
            if not all(math.isfinite(scale[k]) for k in ("scale", "center", "spread")) or min(scale["scale"], scale["spread"]) <= 0:
                raise ValueError("Invalid serialized frozen stress scaler")
        elif (scale["fit_case_keys"] != expected_keys["train"] or scale["fit_nodes"] != sum(c["nodes"] for c in config["cases"]["train"])
              or scale["center_mm"] != 0 or scale["transform"] != "linear_zero_center_common_scalar"
              or not math.isfinite(scale["scale_mm"]) or scale["scale_mm"] <= 0):
            raise ValueError("Displacement scaler must be the serialized zero-centered train120-only common scale")
        # Only stat the five actual evaluation caches; no array/target access.
        for cad in cads:
            record = config["local_geometry"]["CADs"][cad]
            cache = _path(gate["local_geometry"]) / f"{cad}.npz"
            if _path(record["cache"]) != cache or record["target_independent"] is not True:
                raise ValueError("Frozen target-independent geometry cache differs")
            _check_identity(cache, record)
        result["models"][name] = {"run": str(run), "config": config, "summary": summary, "best": best,
                                  "source": {"checkpoint": checkpoint_identity, "config_sha256": _sha(config_path),
                                             "summary_sha256": _sha(summary_path), "best_metrics_sha256": _sha(run / "metrics_best.json"),
                                             "best_step": best["step"], "checkpoint_selection": config["checkpoint_selection"],
                                             "geometry_clouds": _identity(clouds)}}
    pair = [result["models"][n]["config"] for n in MODEL_NAMES[:2]]
    for key in ("scaler", "geometry", "local_geometry", "initial_decoder_parameter_sha256"):
        if pair[0][key] != pair[1][key]:
            raise ValueError(f"Physics pair is not matched in {key}")
    for key in ("condition_dim", "descriptor_dim", "hidden_dim", "hidden_layers", "cloud_points", "seed"):
        if pair[0]["model"][key] != pair[1]["model"][key]:
            raise ValueError(f"Physics pair architecture differs in {key}")
    stress = result["models"]["stress"]
    for cfg in pair:
        reference = cfg["stress_reference"]
        if (_path(reference["checkpoint"]["path"]) != _path(stress["source"]["checkpoint"]["path"])
                or reference["config_sha256"] != stress["source"]["config_sha256"]
                or reference["best_metrics_sha256"] != stress["source"]["best_metrics_sha256"]
                or reference["best_step"] != stress["best"]["step"] or reference["scaler"] != stress["config"]["scaler"]):
            raise ValueError("Physics runs must reference the same declared independent frozen stress model")
    basis_path = _path(gate["load_basis"])
    basis = _json(basis_path)
    if basis.get("compatible") is not True or basis.get("CADs_checked") != 45 or basis["canonical_basis"]["order"] != list(BASIS_ORDER):
        raise ValueError("The load-span metadata was not verified for the 45 allowed CADs")
    records = {r["cad_id"]: r for r in basis["cases"]}
    if set(records) != set(split["groups"]["train"] + cads):
        raise ValueError("Coefficient metadata includes missing/unapproved CADs")
    for cad in cads:
        alpha = records[cad]["alpha_by_load"]["dia"]
        coefficients = np.asarray(alpha["alpha"], np.float64)
        if alpha.get("exact_residual_zero") is not True or coefficients.shape != (3,) or not np.isfinite(coefficients).all():
            raise ValueError("Dia must be the declared exact in-span combination, without coefficient fitting")
    for cfg in pair:
        if cfg["contract"]["load_basis"]["sha256"] != _sha(basis_path):
            raise ValueError("Coefficient metadata changed since physics training")
    result.update(ready=True, basis=basis, basis_sha256=_sha(basis_path))
    return result


def _read_case(path, key, *, fields_to_read):
    with np.load(path, allow_pickle=False) as stored:
        if str(stored["sample_key"].item()) != key or str(stored["label_alignment"].item()) != "official CSV GRID coordinates -> H5 vertices":
            raise ValueError(f"{key}: source key or original CSV node alignment differs")
        result = {name: np.asarray(stored[name]) for name in fields_to_read}
    xyz = result["xyz"]
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError(f"{key}: invalid/nonfinite original fields")
    return result


def evaluate(args, metadata):
    # All imports/model loading below are unreachable under a planned gate.
    import torch
    from geobc.conditioned_field import ConditionedField, ConditionedFieldConfig
    from geobc.physics_field import PhysicsField, PhysicsFieldConfig
    from geobc.local_geometry import load_local_geometry, xyz_sha256
    from geobc.train import StressScaler, stress_metrics
    from geobc.physics_train import DisplacementScaler
    began = time.monotonic()
    output = _path(args.output)
    sources = {name: entry["source"] for name, entry in metadata["models"].items()}
    if (output / "summary.json").is_file():
        summary = _json(output / "summary.json")
        if summary.get("status") == "completed" and summary["config"]["gate_sha256"] == metadata["gate_sha256"] and summary["config"]["frozen_sources"] == sources:
            print(json.dumps({"status": "completed", "reused": True, "output": str(output)}), flush=True)
            return summary
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Nonempty probe output is not silently overwritten")
    output.mkdir(parents=True, exist_ok=True)
    status = {"phase": "loading_frozen_models", "completed_cases": 0, "total_cases": 5}
    def update(phase, **extra):
        status.update(extra, phase=phase, elapsed_seconds=time.monotonic() - began,
                      updated_at_utc=datetime.now(timezone.utc).isoformat())
        done = status["completed_cases"]
        status["eta_seconds"] = status["elapsed_seconds"] / done * (5 - done) if done else None
        _write(output / "status.json", status)
    update("loading_frozen_models")
    try:
        if args.device != "cuda" and not args.device.startswith("cuda:"):
            raise ValueError("Frozen official TCNN evaluation requires CUDA; use --dry-run for CPU metadata checks")
        device = torch.device(args.device)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        models, contexts, cad_rows, scales = {}, {}, {}, {}
        allowed_geometry = set(metadata["split"]["groups"]["train"] + metadata["validation_cads"])
        for name in MODEL_NAMES:
            entry = metadata["models"][name]
            cfg = entry["config"]
            saved = torch.load(_path(entry["source"]["checkpoint"]["path"]), map_location="cpu", weights_only=False)
            _check_identity(_path(entry["source"]["checkpoint"]["path"]), entry["source"]["checkpoint"])
            if saved["config"] != cfg or saved["step"] != entry["best"]["step"] or saved["metrics"] != entry["best"]:
                raise ValueError(f"{name}: checkpoint tensors/metadata are not the declared best")
            if name == "stress":
                cls = ConditionedFieldConfig
                model_cfg = cls(**{f.name: cfg["model"][f.name] for f in fields(cls)})
                model = ConditionedField(model_cfg, device)
                scale_record = cfg["scaler"]
                scales[name] = StressScaler(**{k: float(scale_record[k]) for k in ("scale", "center", "spread")})
            else:
                cls = PhysicsFieldConfig
                model_cfg = cls(**{f.name: cfg["model"][f.name] for f in fields(cls)})
                model = PhysicsField(model_cfg, device)
                if saved["scaler"] != cfg["scaler"]:
                    raise ValueError("Serialized displacement scales disagree")
                scales[name] = DisplacementScaler.from_record(saved["scaler"])
            if model.source_config != cfg["model"]:
                raise ValueError(f"{name}: active model implementation differs from the frozen architecture")
            model.load_state_dict(saved["model"])
            del saved
            model.eval().requires_grad_(False)
            with np.load(_path(entry["run"]) / "geometry_clouds.npz", allow_pickle=False) as archive:
                points, mask = archive["points"], archive["mask"]
                cad_keys = archive["cad_keys"].tolist()
            if len(cad_keys) != 45 or set(cad_keys) != allowed_geometry or points.shape != (45, model_cfg.cloud_points, 3) or mask.shape != points.shape[:2]:
                raise ValueError("Saved geometry clouds include missing/final-test CADs or mismatched point layout")
            expected_point_hash = cfg["geometry"].get("coordinates_sha256", cfg["geometry"].get("coordinate_sha256"))
            if hashlib.sha256(points.tobytes()).hexdigest() != expected_point_hash or hashlib.sha256(mask.tobytes()).hexdigest() != cfg["geometry"]["mask_sha256"]:
                raise ValueError("Saved geometry cloud changed since model training")
            with torch.no_grad():
                contexts[name] = model.encode_geometry(torch.as_tensor(points, device=device), torch.as_tensor(mask, device=device))
            models[name], cad_rows[name] = model, {cad: i for i, cad in enumerate(cad_keys)}
        config = {"gate": metadata["gate"], "gate_sha256": metadata["gate_sha256"], "split_sha256": metadata["split_sha256"],
                  "load_basis_sha256": metadata["basis_sha256"], "frozen_sources": sources,
                  "frozen_architectures": {n: e["config"]["model"] for n, e in metadata["models"].items()},
                  "source_sha256": _sha(Path(__file__)), "case_allowlist": [f"{c}/dia" for c in metadata["validation_cads"]],
                  "scalers": {n: s.record() for n, s in scales.items()}, "no_refitting": True, "optimizer_updates": 0,
                  "evaluation": {"device": str(device), "batch_size": args.batch_size, "matmul_precision": "high",
                                 "cuda_allow_tf32": True, "cudnn_allow_tf32": True,
                                 "decoder_precision": "frozen native TCNN FP16 computation/FP32 master parameters"},
                  "metric_units": "relative L2 fractions; signed uXYZ in mm; signed source scalar stress in MPa",
                  "rmse_mm_definition": "sqrt(mean(component squared error)); vector_rmse_mm separately reports RMS node norm",
                  "interpretation": "validation-CAD held-out combination within the existing rank3 load span, not a new independent BC or final-test result"}
        # Commit frozen provenance before the first dia archive/label access.
        _write(output / "config.json", config)
        _write(output / "frozen_sources.json", sources)
        truths, predictions = [], {n: [] for n in ("linear_basis", "load_dependent_basis", "oracle")}
        stress_truths, stress_predictions, per_stress = [], [], {}
        records = {r["cad_id"]: r for r in metadata["basis"]["cases"]}
        mapping = np.asarray(metadata["basis"]["prepared_feature_mapping"]["feature_to_alpha_matrix_3x6"], np.float64)
        index = {}
        folder = output / "predictions"
        folder.mkdir()
        data = _path(metadata["gate"]["data"])
        read_sources = []
        for cad in metadata["validation_cads"]:
            key = f"{cad}/dia"
            path = _path(metadata["split"]["case_paths"][key])
            if not path.is_relative_to(data):
                raise ValueError("A dia case escaped the declared prepared data directory")
            update("evaluating_validation_dia", current_case=key)
            case = _read_case(path, key, fields_to_read=("xyz", "features", "displacement", "stress"))
            xyz = np.asarray(case["xyz"], np.float32)
            features = np.asarray(case["features"], np.float32)
            u_true, stress_true = np.asarray(case["displacement"], np.float64), np.asarray(case["stress"], np.float64)
            if features.shape != (len(xyz), 25) or u_true.shape != (len(xyz), 3) or stress_true.shape != (len(xyz),):
                raise ValueError("Held-out source fields do not retain original node/component layout")
            alpha = np.asarray(records[cad]["alpha_by_load"]["dia"]["alpha"], np.float64)
            if not np.allclose(features[:, 7:13].astype(np.float64) @ mapping.T, alpha[None], rtol=1e-6, atol=2e-7):
                raise ValueError("Dia prepared load slots disagree with predeclared physical coefficients")
            references = []
            static = features.copy(); static[:, 7:13] = 0
            for load in BASIS_ORDER:
                refkey = f"{cad}/{load}"
                refpath = _path(metadata["split"]["case_paths"][refkey])
                if not refpath.is_relative_to(data):
                    raise ValueError("Reference field escaped the declared data directory")
                ref = _read_case(refpath, refkey, fields_to_read=("xyz", "features", "displacement"))
                ref_static = np.asarray(ref["features"], np.float32).copy(); ref_static[:, 7:13] = 0
                if xyz_sha256(ref["xyz"]) != xyz_sha256(xyz) or not np.array_equal(ref_static, static):
                    raise ValueError("Geometry/nonload BC changed: source superposition is not a matched load-span probe")
                references.append(np.asarray(ref["displacement"], np.float64))
                read_sources.append({"sample_key": refkey, "path": str(refpath), "members": ["xyz", "features", "displacement"]})
            oracle = oracle_superposition(np.stack(references), alpha)
            cache = load_local_geometry(_path(metadata["gate"]["local_geometry"]), cad, xyz=xyz)
            for entry in metadata["models"].values():
                _check_identity(_path(entry["config"]["local_geometry"]["CADs"][cad]["cache"]), entry["config"]["local_geometry"]["CADs"][cad])
            tx, tf = torch.as_tensor(xyz, device=device), torch.as_tensor(features, device=device)
            local = cache.tensor(mode="actual", device=device)
            conditioned = torch.cat((tf, local), dim=-1)
            outputs = {}
            with torch.no_grad():
                for name in MODEL_NAMES:
                    row = cad_rows[name][cad]
                    context = contexts[name][row:row + 1]
                    encoded = np.empty((len(xyz), 3) if name != "stress" else len(xyz), np.float32)
                    for start in range(0, len(xyz), args.batch_size):
                        stop = min(start + args.batch_size, len(xyz))
                        if name == "stress":
                            pred = models[name](tx[start:stop], conditioned[start:stop], context=context).reshape(-1)
                        else:
                            coefficients = torch.as_tensor(alpha, dtype=torch.float32, device=device).expand(stop - start, -1)
                            pred = models[name](tx[start:stop], conditioned[start:stop], coefficients, context=context)
                        encoded[start:stop] = pred.float().cpu().numpy()
                    outputs[name] = scales[name].decode(encoded)
            truths.append(u_true)
            for name in MODEL_NAMES[:2]:
                predictions[name].append(outputs[name])
            predictions["oracle"].append(oracle)
            stress_truths.append(stress_true); stress_predictions.append(outputs["stress"])
            per_stress[key] = stress_metrics(stress_true, outputs["stress"])
            read_sources.append({"sample_key": key, "path": str(path), "members": ["xyz", "features", "displacement", "stress"]})
            # At most two representatives, fixed by predeclared validation order.
            if cad in metadata["validation_cads"][:2]:
                filename = f"{cad}_dia.npz"
                np.savez(folder / filename, xyz=xyz, sample_key=np.asarray(key), alpha=alpha,
                         u_true=u_true, u_linear_basis=outputs["linear_basis"],
                         u_load_dependent_basis=outputs["load_dependent_basis"], u_oracle=oracle,
                         stress_true=stress_true, stress_pred=outputs["stress"])
                index[key] = {"file": filename, "source": str(path), "nodes": len(xyz), "cad_id": cad}
            status["completed_cases"] += 1
            update("evaluating_validation_dia")
        u_metrics = {name: summarize_vectors(truths, preds, metadata["validation_cads"]) for name, preds in predictions.items()}
        worst = max(per_stress, key=lambda k: per_stress[k]["relative_l2"])
        stress_metrics_group = {"per_case": per_stress, "pooled": stress_metrics(np.concatenate(stress_truths), np.concatenate(stress_predictions)),
                                "case_mean_relative_l2": float(np.mean([m["relative_l2"] for m in per_stress.values()])),
                                "case_mean_hotspot_relative_l2": float(np.mean([m["hotspot_relative_l2"] for m in per_stress.values()])),
                                "worst_case_relative_l2": per_stress[worst]["relative_l2"], "worst_case_key": worst}
        _write(folder / "index.json", {"cases": index, "selection": "first two fixed validation CADs, not selected by error"})
        _write(output / "source_index.json", {"field_sources": read_sources, "test_fields_opened": [], "model_sources": sources})
        summary = {"status": "completed", "config": config, "displacement": u_metrics, "stress": stress_metrics_group,
                   "paired_case_delta_control_minus_linear": {key: u_metrics["load_dependent_basis"]["per_case"][key]["relative_l2"] - u_metrics["linear_basis"]["per_case"][key]["relative_l2"] for key in u_metrics["linear_basis"]["per_case"]},
                   "elapsed_seconds": time.monotonic() - began, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                   "oracle_definition": "residual=u_dia-alphaV*u_ver-alphaH*u_hor-alphaT*u_tor, physical original signed mm",
                   "no_training_or_selection_on_dia": True, "final_test_fields_opened": [], "representative_index": str(folder / "index.json")}
        _write(output / "metrics.json", {"displacement": u_metrics, "stress": stress_metrics_group})
        _write(output / "summary.json", summary)
        update("completed")
        print(json.dumps({"status": "completed", "output": str(output), "displacement_case_mean_l2": {n: m["case_mean_relative_l2"] for n, m in u_metrics.items()},
                          "stress_case_mean_l2": stress_metrics_group["case_mean_relative_l2"]}, indent=2), flush=True)
        return summary
    except Exception as exc:
        update("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true", help="Gate/small metadata only; never read checkpoints or dia fields")
    p.add_argument("--batch-size", type=int, default=65536)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    metadata = _metadata(args)
    if args.dry_run:
        print(json.dumps({"status": metadata["gate"]["status"], "ready_for_frozen_evaluation": metadata["ready"],
                          "validation_cads": metadata["validation_cads"], "checkpoints_opened": 0,
                          "dia_fields_opened": 0, "final_test_fields_opened": 0}, indent=2))
        return
    if not metadata["ready"]:
        raise RuntimeError("Gate is planned; root must freeze completed model sources before any checkpoint/dia access")
    evaluate(args, metadata)


if __name__ == "__main__":
    main()
