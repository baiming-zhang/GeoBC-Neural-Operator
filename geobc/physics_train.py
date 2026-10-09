"""Matched displacement-basis fields on the locked 40/5/5 CAD protocol.

Only train/validation ver,hor,tor fields are opened. Nine signed displacement
responses use one train-only, zero-center linear scale. A frozen strict scalar
stress checkpoint is referenced separately, never linearly combined or loaded
into this displacement optimization.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from .conditional_train import (_scope as conditional_scope, _path,
                                _encoder_gradient, _restore_history_row, _save_checkpoint)
from .conditioned_field import load_geometry_cloud
from .local_geometry import FEATURE_DIM as LOCAL_DIM, FEATURE_NAMES as LOCAL_NAMES, load_local_geometry, xyz_sha256
from .train import utc_now, write_json

ROOT = Path(__file__).resolve().parents[1]
BASIS_ORDER = ("ver", "hor", "tor")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _load_basis(path: Path, split: dict[str, Any]) -> dict[str, Any]:
    """Read verified load metadata only, not any displacement or stress field."""
    guard = _json(path)
    if (guard.get("status") != "compatible" or guard.get("compatible") is not True
            or guard.get("CADs_checked") != 45 or guard.get("badcases")):
        raise ValueError("All45 train/validation CAD load-basis metadata must be compatible")
    if guard["canonical_basis"]["order"] != list(BASIS_ORDER) or guard["canonical_basis"]["exact_rank"] != 3:
        raise ValueError("Expected the verified rank3 ver/hor/tor load basis")
    mapping = guard["prepared_feature_mapping"]
    if mapping["slots"] != list(range(7, 13)) or mapping["physical_scale"] != [42300] * 3 + [565000] * 3:
        raise ValueError("Prepared force/moment feature slots or physical scales changed")
    matrix = np.asarray(mapping["feature_to_alpha_matrix_3x6"], dtype=np.float64)
    if matrix.shape != (3, 6) or not np.isfinite(matrix).all():
        raise ValueError("Invalid prepared-feature to alpha transformation")
    records = {entry["cad_id"]: entry for entry in guard["cases"]}
    allowed = split["groups"]["train"] + split["groups"]["validation"]
    if len(records) != 45 or set(records) != set(allowed):
        raise ValueError("Load-basis CAD allowlist differs from the strict split")
    for group in ("train", "validation"):
        for cad in split["groups"][group]:
            record = records[cad]
            if (record.get("group") != group or record.get("compatible") is not True
                    or record.get("basis_rank") != 3
                    or record.get("basis_matches_shared_reference") is not True
                    or record.get("prepared_slot_mapping_compatible") is not True):
                raise ValueError(f"Incompatible physical load basis: {cad}")
            for load in BASIS_ORDER:
                item = record["alpha_by_load"][load]
                alpha = np.asarray(item["alpha"], np.float64)
                if alpha.shape != (3,) or not np.isfinite(alpha).all() or item.get("exact_residual_zero") is not True:
                    raise ValueError(f"Invalid verified alpha for {cad}/{load}")
    return guard


def _stress_reference(checkpoint: Path, split: dict[str, Any], keys: dict[str, list[str]]) -> dict[str, Any]:
    """Small frozen-reference metadata checks; never load the stress weights."""
    if checkpoint.name != "best.pt" or not checkpoint.is_file():
        raise ValueError("--stress-checkpoint must be the best.pt of a completed strict stress run")
    folder = checkpoint.parent
    config = _json(folder / "config.json")
    summary = _json(folder / "summary.json")
    metrics = _json(folder / "metrics_best.json")
    if summary.get("status") != "completed" or config.get("groups") != split["groups"]:
        raise ValueError("Frozen stress branch must be completed and use this strict CAD split")
    if config.get("train_loads") != split["train_loads"] or config.get("heldout_load") != "dia":
        raise ValueError("Frozen stress branch must exclude dia from training/selection")
    for group in ("train", "validation"):
        if [case["sample_key"] for case in config["cases"][group]] != keys[group]:
            raise ValueError(f"Frozen stress {group} case allowlist differs")
    if (metrics["step"] != summary["best_step"]
            or metrics["selection_score"] != metrics["validation"]["case_mean_relative_l2"]):
        raise ValueError("Frozen stress reference is not the declared validation-selected best")
    for name in ("scale", "center", "spread"):
        if not math.isfinite(float(config["scaler"][name])):
            raise ValueError("Invalid frozen stress scaler")
    if config["scaler"]["scale"] <= 0 or config["scaler"]["spread"] <= 0:
        raise ValueError("Invalid frozen stress scale/spread")
    return {"checkpoint": _identity(checkpoint), "config_sha256": _sha((folder / "config.json").read_bytes()),
            "best_metrics_sha256": _sha((folder / "metrics_best.json").read_bytes()),
            "best_step": metrics["step"], "scaler": config["scaler"],
            "train_metrics": metrics["train"], "validation_metrics": metrics["validation"],
            "policy": "independently supervised signed scalar stress, frozen reference; no stress optimization or linear composition"}


def _scope(args):
    """Return the standard five-tuple using metadata/stat/code checks only."""
    split, split_path, keys, files, contract = conditional_scope(args)
    basis_path = _path(args.load_basis)
    guard = _load_basis(basis_path, split)
    stress = _stress_reference(_path(args.stress_checkpoint), split, keys)
    contract["version"] = 2
    contract["requested"]["physics_mode"] = args.physics_mode
    contract["load_basis"] = {**_identity(basis_path), "sha256": _sha(basis_path.read_bytes()),
                              "basis_order": guard["canonical_basis"]["order"]}
    contract["stress_reference"] = stress
    for filename in ("physics_train.py", "physics_field.py", "train.py"):
        contract["code_sha256"][filename] = _sha((ROOT / "geobc" / filename).read_bytes())
    return split, split_path, keys, files, contract


@dataclass(frozen=True)
class DisplacementScaler:
    scale_mm: float
    fit_nodes: int
    fit_case_keys: tuple[str, ...]

    @classmethod
    def fit(cls, cases) -> "DisplacementScaler":
        sum_squared = 0.0
        count = 0
        for case in cases:
            values = np.asarray(case.physical_u, dtype=np.float64)
            sum_squared += float(np.einsum("nc,nc->", values, values))
            count += len(values)
        if count == 0 or not math.isfinite(sum_squared):
            raise ValueError("Cannot fit displacement scale from empty/nonfinite training labels")
        return cls(max(math.sqrt(sum_squared / (3 * count)), 1e-12), count, tuple(case.key for case in cases))

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "DisplacementScaler":
        if record.get("center_mm") != 0.0 or record.get("transform") != "linear_zero_center_common_scalar":
            raise ValueError("Displacement basis requires a zero-center linear scaler")
        result = cls(float(record["scale_mm"]), int(record["fit_nodes"]), tuple(record["fit_case_keys"]))
        if not math.isfinite(result.scale_mm) or result.scale_mm <= 0 or result.fit_nodes <= 0:
            raise ValueError("Invalid saved displacement scaler")
        return result

    def encode(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, np.float64) / self.scale_mm).astype(np.float32)

    def decode(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, np.float64) * self.scale_mm
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite decoded signed displacement")
        return result

    def record(self) -> dict[str, Any]:
        return {"scale_mm": self.scale_mm, "center_mm": 0.0,
                "transform": "linear_zero_center_common_scalar", "fit_nodes": self.fit_nodes,
                "fit_case_keys": list(self.fit_case_keys), "channels": ["ux_mm", "uy_mm", "uz_mm"],
                "definition": "max(sqrt(sum_train_u_squared/(3*N_train_case_nodes)),1e-12 mm)",
                "fit_scope": "strict40 training CADs x ver/hor/tor; validation/dia/finaltest excluded"}


@dataclass
class DisplacementCase:
    key: str
    source: Path
    xyz: torch.Tensor
    features: torch.Tensor
    physical_u: np.ndarray
    alpha: torch.Tensor
    target: torch.Tensor | None = None

    @property
    def count(self) -> int:
        return len(self.physical_u)


def load_displacement_cases(keys: list[str], files: dict[str, Path], guard: dict[str, Any], device) -> list[DisplacementCase]:
    records = {entry["cad_id"]: entry for entry in guard["cases"]}
    mapping = np.asarray(guard["prepared_feature_mapping"]["feature_to_alpha_matrix_3x6"], np.float64)
    canonical = {}
    result = []
    for key in keys:
        cad, load = key.split("/", 1)
        if load not in BASIS_ORDER:
            raise ValueError("A held-out load cannot enter the displacement loader")
        path = files[key]
        with np.load(path, allow_pickle=False) as stored:
            if str(stored["label_alignment"].item()) != "official CSV GRID coordinates -> H5 vertices":
                raise ValueError(f"Unverified source field alignment: {key}")
            if str(stored["sample_key"].item()) != key:
                raise ValueError(f"Prepared case key differs from strict allowlist: {key}")
            xyz = np.asarray(stored["xyz"], np.float32)
            features = np.asarray(stored["features"], np.float32)
            # No stress, resultant, dia or test member is read by this loader.
            u = np.asarray(stored["displacement"], np.float64)
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz)
                or features.shape != (len(xyz), 25) or u.shape != (len(xyz), 3)
                or not all(np.isfinite(value).all() for value in (xyz, features, u))):
            raise ValueError(f"Invalid signed displacement/input shape or values: {key}")
        if xyz.min() < -1e-6 or xyz.max() > 1 + 1e-6:
            raise ValueError(f"Normalized xyz outside the source domain: {key}")
        alpha = np.asarray(records[cad]["alpha_by_load"][load]["alpha"], np.float64)
        inferred = features[:, 7:13].astype(np.float64) @ mapping.T
        if not np.allclose(inferred, alpha[None, :], rtol=1e-6, atol=2e-7):
            raise ValueError(f"Prepared force/moment slots disagree with the verified alpha: {key}")
        static = features.copy()
        static[:, 7:13] = 0.0
        signature = {"xyz": xyz_sha256(xyz), "static_features": _sha(static.tobytes(order="C")), "nodes": len(xyz)}
        if cad in canonical and signature != canonical[cad]:
            raise ValueError(f"Geometry or load-independent BC/material inputs changed between basis loads: {cad}")
        canonical[cad] = signature
        result.append(DisplacementCase(key, path, torch.as_tensor(xyz, device=device),
                                      torch.as_tensor(features, device=device), u,
                                      torch.as_tensor(alpha, dtype=torch.float32, device=device)))
    return result


def displacement_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth, prediction = np.asarray(truth, np.float64), np.asarray(prediction, np.float64)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 3 or not len(truth):
        raise ValueError("Displacement metrics require matching nonempty signed[N,3] fields")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite physical displacement metrics")
    error = prediction - truth
    return {"nodes": len(truth), "relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(truth), 1e-12)),
            "rmse_mm": float(np.sqrt(np.mean(error ** 2))),
            "mae_component_mm": float(np.mean(np.abs(error))),
            "max_vector_error_mm": float(np.linalg.norm(error, axis=1).max())}


@torch.no_grad()
def _predict(model, case, context, scaler, batch_size):
    encoded = np.empty((case.count, 3), np.float32)
    for start in range(0, case.count, batch_size):
        stop = min(start + batch_size, case.count)
        alpha = case.alpha.expand(stop - start, -1)
        encoded[start:stop] = model(case.xyz[start:stop], case.features[start:stop], alpha,
                                    context=context).float().cpu().numpy()
    return scaler.decode(encoded)


@torch.no_grad()
def _evaluate_group(model, cases, contexts, cad_to_row, scaler, batch_size):
    per, truth, predictions = {}, [], []
    for case in cases:
        row = cad_to_row[case.key.split("/", 1)[0]]
        pred = _predict(model, case, contexts[row:row + 1], scaler, batch_size)
        per[case.key] = displacement_metrics(case.physical_u, pred)
        truth.append(case.physical_u)
        predictions.append(pred)
    worst = max(per, key=lambda key: per[key]["relative_l2"])
    return {"per_case": per, "pooled": displacement_metrics(np.concatenate(truth), np.concatenate(predictions)),
            "case_mean_relative_l2": float(np.mean([item["relative_l2"] for item in per.values()])),
            "worst_case_key": worst, "worst_case_relative_l2": per[worst]["relative_l2"]}


def run(args):
    began = time.monotonic()
    if (min(args.steps, args.batch_size, args.cases_per_batch, args.eval_every, args.eval_batch_size, args.cloud_points) < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not 1 <= args.cases_per_batch <= 4 or args.batch_size % args.cases_per_batch):
        raise ValueError("Positive sizes/LR and1--4 evenly balanced cases per batch are required")
    split, split_path, keys, files, contract = _scope(args)
    output = _path(args.output)
    resume = (output / "last.pt" if str(args.resume) == "last" else _path(args.resume)) if args.resume else None
    if resume is not None and resume != (output / "last.pt").resolve():
        raise ValueError("Resume must use this output's last.pt to preserve committed historical best")
    if (output / "summary.json").is_file():
        summary = _json(output / "summary.json")
        if summary.get("status") == "completed":
            if summary["config"].get("contract") != contract:
                raise ValueError("Completed physics run has a different strict contract")
            for name in ("best.pt", "last.pt", "config.json", "metrics_best.json", "metrics_latest.json"):
                if not (output / name).is_file():
                    raise ValueError(f"Completed run lacks {name}")
            print(f"Reusing completed physics run {output}", flush=True)
            return summary
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError("Incomplete nonempty output requires explicit --resume")
    stored_config = _json(output / "config.json") if resume is not None else None
    if resume is not None and (not resume.is_file() or stored_config.get("contract") != contract):
        raise ValueError("Resume checkpoint is missing or its strict contract changed")
    output.mkdir(parents=True, exist_ok=True)
    status = {"phase": "loading", "physics_mode": args.physics_mode, "step": 0,
              "total_steps": args.steps, "started_at_utc": utc_now()}
    start_step, prior_elapsed = 0, 0.0
    def elapsed():
        return prior_elapsed + time.monotonic() - began
    def update(phase, **extra):
        status.update(extra, phase=phase, elapsed_seconds=elapsed(), updated_at_utc=utc_now())
        progress = status["step"] - start_step
        eta = (time.monotonic() - began) / progress * max(0, args.steps - status["step"]) if progress > 0 else None
        if phase == "completed":
            eta = 0.0
        status["eta_seconds"] = eta
        status["estimated_completion_utc"] = (datetime.now(timezone.utc) + timedelta(seconds=eta)).isoformat() if eta is not None else None
        write_json(output / "status.json", status)
    update("loading")
    try:
        from .physics_field import PhysicsField, PhysicsFieldConfig
        device = torch.device(args.device)
        if device.type != "cuda":
            raise ValueError("TCNN training requires CUDA; CPU checks use injected model helpers only")
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        guard = _load_basis(_path(args.load_basis), split)
        train_cases = load_displacement_cases(keys["train"], files, guard, device)
        val_cases = load_displacement_cases(keys["validation"], files, guard, device)
        scaler = DisplacementScaler.from_record(stored_config["scaler"]) if stored_config else DisplacementScaler.fit(train_cases)
        if scaler.fit_case_keys != tuple(keys["train"]) or scaler.fit_nodes != sum(case.count for case in train_cases):
            raise ValueError("Saved u scaler scope/counts differ from the explicit training cases")
        all_cases = train_cases + val_cases
        for case in all_cases:
            case.target = torch.as_tensor(scaler.encode(case.physical_u), device=device)
        cad_keys = split["groups"]["train"] + split["groups"]["validation"]
        cad_to_row = {cad: number for number, cad in enumerate(cad_keys)}
        local_records = {}
        if args.local_mode != "none":
            by_cad = {cad: [] for cad in cad_keys}
            for case in all_cases:
                by_cad[case.key.split("/", 1)[0]].append(case)
            for cad in cad_keys:
                cache = load_local_geometry(_path(args.local_geometry), cad)
                local_tensor = cache.tensor(mode=args.local_mode, device=device)
                for case in by_cad[cad]:
                    cache.verify_xyz(case.xyz)
                    case.features = torch.cat((case.features, local_tensor), dim=-1)
                local_records[cad] = cache.record()
        clouds = [load_geometry_cloud(files[f"{cad}/ver"], cloud_points=args.cloud_points, seed=args.seed, cad_key=cad) for cad in cad_keys]
        cloud_array, mask_array = np.stack([item.points for item in clouds]), np.stack([item.mask for item in clouds])
        geometry_clouds, geometry_masks = torch.as_tensor(cloud_array, device=device), torch.as_tensor(mask_array, device=device)
        model_config = PhysicsFieldConfig(mode=args.physics_mode,
            geometry_mode="shape" if args.variant == "geometry_pointnet" else "none",
            condition_dim=25 + (LOCAL_DIM if args.local_mode != "none" else 0), descriptor_dim=args.geometry_dim,
            hidden_dim=args.hidden_dim, cloud_points=args.cloud_points, seed=args.seed)
        model = PhysicsField(model_config, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, fused=True)
        case_rng = np.random.default_rng(args.seed + 23)
        node_rng = torch.Generator(device=device).manual_seed(args.seed + 17)
        config = {"contract": contract, "method": "GeoBC matched displacement-basis diagnostic",
            "physics_mode": args.physics_mode, "variant": args.variant,
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "split_source": str(split_path), "scope": "40train/5validation CADs x ver/hor/tor; dia and finaltest fields locked",
            "groups": split["groups"], "train_loads": split["train_loads"], "heldout_load": "dia",
            "cases": {group: [{"sample_key": case.key, "source": str(case.source), "nodes": case.count,
                               "alpha": guard_case_alpha(guard, case.key)} for case in cases]
                      for group, cases in (("train", train_cases), ("validation", val_cases))},
            "scaler": scaler.record(), "stress_reference": contract["stress_reference"],
            "optimizer": {"name": "Adam", "fused": True, "learning_rate": args.learning_rate,
                          "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0.0},
            "basis_order": list(BASIS_ORDER), "basis_layout": "[N,basis_ver_hor_tor,component_xyz]",
            "force_moment_slots": list(range(7, 13)), "model": model.source_config,
            "initial_decoder_parameter_sha256": _sha(model.decoder.native.params.detach().cpu().numpy().tobytes()),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "decoder_parameter_count": sum(parameter.numel() for parameter in model.decoder.parameters()),
            "geometry": {"coordinates_sha256": _sha(cloud_array.tobytes()), "mask_sha256": _sha(mask_array.tobytes()),
                         "cloud_points": args.cloud_points, "source": "canonical ver geometry only; same context for all loads"},
            "local_geometry": {"mode": args.local_mode, "feature_names": list(LOCAL_NAMES) if args.local_mode != "none" else [], "CADs": local_records},
            "sampling": "four distinct permitted cases, equal nodes per case with replacement",
            "sampling_rng": {"case": "NumPy default_rng(seed+23)", "node": "torch.Generator(device),seed+17"},
            "loss": "case-balanced MSE of3 signed u components with train-only common RMS and exactly zero center",
            "checkpoint_selection": "minimum validation15-case mean all-original-node physical displacement vector relativeL2",
            "stress_policy": "freeze independently supervised strict stress branch; do not combine stress or train it with u loss",
            "precision": "PointNet float32/high TF32; native TCNN FP16 compute/FP32 master; physical metrics float64",
            "total_steps": args.steps, "torch_version": torch.__version__, "started_at_utc": utc_now()}
        best_score, best_step, best_metrics, gradient_check = float("inf"), None, None, None
        if resume is not None:
            for name in ("contract", "scaler", "model", "geometry", "local_geometry", "cases", "precision"):
                if config[name] != stored_config[name]:
                    raise ValueError(f"Resume {name} differs")
            saved = torch.load(resume, map_location=device, weights_only=False)
            if saved["config"] != stored_config:
                raise ValueError("last.pt and config.json differ")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            start_step = int(saved["step"])
            if not 0 <= start_step <= args.steps:
                raise ValueError("Resume step outside requested budget")
            case_rng.bit_generator.state = saved["case_rng_state"]
            node_rng.set_state(saved["node_rng_state"].cpu())
            torch.set_rng_state(saved["torch_rng_state"].cpu())
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng_state"]])
            np.random.set_state(saved["numpy_rng_state"])
            random.setstate(saved["python_rng_state"])
            best_step, best_score, best_metrics = saved["best_step"], saved["best_score"], saved["best_metrics"]
            if best_metrics["step"] != best_step or best_metrics["selection_score"] != best_score:
                raise ValueError("Committed historical best is inconsistent")
            if best_step == start_step:
                _save_checkpoint(saved, output / "best.pt")
            elif not (output / "best.pt").is_file() or _json(output / "metrics_best.json") != best_metrics:
                raise ValueError("Earlier historical best checkpoint is missing/inconsistent")
            write_json(output / "metrics_best.json", best_metrics)
            write_json(output / "metrics_latest.json", saved["metrics"])
            _restore_history_row(output, saved["metrics"])
            gradient_check, prior_elapsed = saved.get("encoder_gradient_check"), float(saved["elapsed_seconds"])
            config = stored_config
        else:
            write_json(output / "config.json", config)
            write_json(output / "cloud_metadata.json", {"CADs": {cad: cloud.record() for cad, cloud in zip(cad_keys, clouds)}, "target_independent": True})
            np.savez(output / "geometry_clouds.npz", points=cloud_array, mask=mask_array, cad_keys=np.asarray(cad_keys), node_indices=np.stack([cloud.node_indices for cloud in clouds]))

        def checkpoint_payload(step, metrics):
            return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                    "step": step, "scaler": scaler.record(), "metrics": metrics, "best_step": best_step,
                    "best_score": best_score, "best_metrics": best_metrics, "elapsed_seconds": elapsed(),
                    "encoder_gradient_check": gradient_check, "case_rng_state": case_rng.bit_generator.state,
                    "node_rng_state": node_rng.get_state(), "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all(), "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate()}
        @torch.no_grad()
        def evaluate(step, loss):
            nonlocal best_score, best_step, best_metrics
            eval_began = time.monotonic()
            model.eval()
            update("evaluating", step=step)
            contexts = model.encode_geometry(geometry_clouds, geometry_masks)
            train = _evaluate_group(model, train_cases, contexts, cad_to_row, scaler, args.eval_batch_size)
            validation = _evaluate_group(model, val_cases, contexts, cad_to_row, scaler, args.eval_batch_size)
            score = validation["case_mean_relative_l2"]
            metrics = {"step": step, "selection_score": score, "training_loss": loss,
                       "train": train, "validation": validation, "elapsed_seconds": elapsed(),
                       "eval_seconds": time.monotonic() - eval_began, "encoder_gradient_check": gradient_check}
            improved = score < best_score
            if improved:
                best_score, best_step, best_metrics = score, step, metrics
            payload = checkpoint_payload(step, metrics)
            # Commit the complete recoverable state before its derived best/JSON artifacts.
            _save_checkpoint(payload, output / "last.pt")
            if improved:
                _save_checkpoint(payload, output / "best.pt")
                write_json(output / "metrics_best.json", metrics)
            write_json(output / "metrics_latest.json", metrics)
            _restore_history_row(output, metrics)
            update("training", step=step, best_step=best_step, best_score=best_score,
                   latest_train_case_mean_l2=train["case_mean_relative_l2"], latest_validation_case_mean_l2=score)
            print(f"{args.physics_mode} {step}/{args.steps} u_train={100*train['case_mean_relative_l2']:.4f}% u_val={100*score:.4f}% best={best_step}", flush=True)
            return metrics
        update("training", step=start_step, resume_step=start_step)
        last = saved["metrics"] if resume is not None else evaluate(0, None)
        nodes_per_case = args.batch_size // args.cases_per_batch
        for step in range(start_step + 1, args.steps + 1):
            model.train()
            selected = case_rng.choice(len(train_cases), args.cases_per_batch, replace=False)
            batch_cases = [train_cases[int(index)] for index in selected]
            cads = list(dict.fromkeys(case.key.split("/", 1)[0] for case in batch_cases))
            rows = torch.tensor([cad_to_row[cad] for cad in cads], device=device)
            context = model.encode_geometry(geometry_clouds[rows], geometry_masks[rows])
            check_gradient = model.encoder is not None and gradient_check is None
            if check_gradient:
                context.retain_grad()
            local_rows = {cad: number for number, cad in enumerate(cads)}
            xyz, features, alpha, target, indices, witness_nodes = [], [], [], [], [], []
            for case in batch_cases:
                nodes = torch.randint(case.count, (nodes_per_case,), generator=node_rng, device=device)
                xyz.append(case.xyz[nodes]); features.append(case.features[nodes]); target.append(case.target[nodes])
                alpha.append(case.alpha.expand(nodes_per_case, -1))
                indices.append(torch.full((nodes_per_case,), local_rows[case.key.split("/", 1)[0]], dtype=torch.long, device=device))
                if step == start_step + 1:
                    witness_nodes.append(nodes[:32].cpu().numpy().tobytes())
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.cat(xyz), torch.cat(features), torch.cat(alpha), cad_index=torch.cat(indices), context=context)
            loss = (prediction - torch.cat(target)).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite u loss at step{step}")
            loss.backward()
            if check_gradient:
                gradient_check = _encoder_gradient(model, context)
                gradient_check["step"] = step
                write_json(output / "encoder_gradient_check.json", gradient_check)
            optimizer.step()
            if step == start_step + 1:
                witness = selected.astype(np.int64).tobytes() + b"".join(witness_nodes)
                write_json(output / ("sampling_first_batch.json" if start_step == 0 else "sampling_resume_batch.json"),
                           {"step": step, "case_keys": [case.key for case in batch_cases], "nodes_per_case": nodes_per_case,
                            "witness_sha256": _sha(witness), "sampler_streams_independent_of_model": True})
            if step % 100 == 0:
                update("training", step=step, training_loss=float(loss.detach()), best_step=best_step, best_score=best_score)
            if step % args.eval_every == 0 or step == args.steps:
                last = evaluate(step, float(loss.detach()))
        update("exporting_representatives", step=args.steps, best_step=best_step)
        chosen = _json(output / "metrics_best.json")
        saved_best = torch.load(output / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved_best["model"])
        model.eval()
        by_key = {case.key: case for case in all_cases}
        reps = []
        for group in ("train", "validation"):
            ordered = sorted(chosen[group]["per_case"], key=lambda key: chosen[group]["per_case"][key]["relative_l2"])
            reps.extend((group, key) for key in dict.fromkeys((ordered[0], ordered[-1])))
        folder = output / "predictions_best"
        folder.mkdir(exist_ok=True)
        index = {}
        with torch.no_grad():
            contexts = model.encode_geometry(geometry_clouds, geometry_masks)
            for number, (group, key) in enumerate(reps[:4]):
                case = by_key[key]
                row = cad_to_row[key.split("/", 1)[0]]
                pred = _predict(model, case, contexts[row:row + 1], scaler, args.eval_batch_size)
                if not np.isclose(displacement_metrics(case.physical_u, pred)["relative_l2"], chosen[group]["per_case"][key]["relative_l2"], rtol=1e-5, atol=1e-8):
                    raise ValueError("Representative u prediction differs from saved best")
                filename = f"case_{number:03d}.npz"
                np.savez(folder / filename, xyz=case.xyz.cpu().numpy(), u_true_mm=case.physical_u, u_pred_mm=pred,
                         alpha=case.alpha.cpu().numpy(), sample_key=np.asarray(key), step=np.asarray(best_step),
                         channels=np.asarray(["ux_mm", "uy_mm", "uz_mm"]), u_scale_mm=np.asarray(scaler.scale_mm))
                index[key] = {"file": filename, "source": str(case.source), "group": group, "nodes": case.count}
        write_json(folder / "index.json", {"step": best_step, "cases": index, "policy": "at most4 train/validation representatives; no dia/test"})
        summary = {"status": "completed", "step": args.steps, "best_step": best_step, "best_score": best_score,
                   "elapsed_seconds": elapsed(), "best_metrics": chosen, "final_metrics": last, "config": config,
                   "stress_reference": config["stress_reference"], "encoder_gradient_check": gradient_check,
                   "representative_predictions": str(folder / "index.json"), "completed_at_utc": utc_now()}
        write_json(output / "summary.json", summary)
        update("completed", step=args.steps, best_step=best_step, best_score=best_score)
        return summary
    except Exception as error:
        update("failed", error=f"{type(error).__name__}: {error}")
        raise


def guard_case_alpha(guard: dict[str, Any], key: str) -> list[float]:
    cad, load = key.split("/", 1)
    return next(record for record in guard["cases"] if record["cad_id"] == cad)["alpha_by_load"][load]["alpha"]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--physics-mode", choices=("linear_basis", "load_dependent_basis"), required=True)
    result.add_argument("--split", type=Path, default=Path("notes/geobc_split.json"))
    result.add_argument("--data", type=Path, default=Path("data/prepared_geobc_multiload"))
    result.add_argument("--load-basis", type=Path, default=Path("notes/geobc_load_basis_45.json"))
    result.add_argument("--stress-checkpoint", type=Path, required=True)
    result.add_argument("--variant", choices=("geometry_none", "geometry_pointnet"), default="geometry_pointnet")
    result.add_argument("--steps", type=int, default=10000)
    result.add_argument("--batch-size", type=int, default=51200)
    result.add_argument("--cases-per-batch", type=int, default=4)
    result.add_argument("--eval-every", type=int, default=1000)
    result.add_argument("--eval-batch-size", type=int, default=65536)
    result.add_argument("--learning-rate", type=float, default=.003)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--cloud-points", type=int, default=1024)
    result.add_argument("--geometry-dim", type=int, choices=(32, 64), default=32)
    result.add_argument("--hidden-dim", type=int, choices=(64, 128), default=128)
    result.add_argument("--local-mode", choices=("none", "zero", "actual"), default="none")
    result.add_argument("--local-geometry", type=Path)
    result.add_argument("--device", default="cuda")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--resume", nargs="?", const="last", default=None)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
