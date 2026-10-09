"""Fixed TRAIN101_479 three-load specialist, capped diagnostic endpoints only.

This is one dedicated per-CAD fitting/optimization diagnostic. Both250 and10000
are fixed readouts, with no validation, best-by-u/strain or adaptive early stop.
The full10k exposure is40x greater per load than the shared120-case budget.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import copy
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .gradient_balance_train import (
    _scope as shared_scope, parser as shared_parser, _sha, _json,
    _memory_record, mixed_query_indices, gradient_batch_loss, _evaluate_group,
    BASIS_ORDER, STRAIN_FLOOR, UNIFORM_NODES, CELLS_PER_CASE, NODES_PER_CASE)
from .physics_train import (DisplacementScaler, load_displacement_cases,
    relative_case_mse, displacement_metrics, guard_case_alpha,
    _load_basis, _predict, RELATIVE_CASE_RMS_FLOOR_MM)
from .conditional_train import (_path, _encoder_gradient, _restore_history_row,
    _save_checkpoint, _initial_parameters)
from .conditioned_field import load_geometry_cloud
from .local_geometry import FEATURE_DIM as LOCAL_DIM, FEATURE_NAMES as LOCAL_NAMES, load_local_geometry
from .train import utc_now, write_json

ROOT = Path(__file__).resolve().parents[1]
SPECIALIST_CAD = "101_479"
SPECIALIST_KEYS = tuple(f"{SPECIALIST_CAD}/{load}" for load in BASIS_ORDER)
ENDPOINTS = (250, 10000)


def _identity(path):
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _fixed_loss_record(reference, scaler):
    """Reuse three original train case weights without any target/stat fit."""
    source = reference["u_loss"]
    if (source["name"] != "relative_case_mse" or source["common_scale_mm"] != scaler.scale_mm
            or source["decode_center_mm"] != 0.0 or tuple(source["fit_case_keys"]) != scaler.fit_case_keys
            or source["RMS_floor_mm"] != RELATIVE_CASE_RMS_FLOOR_MM
            or source["weight_cap"] is not None or source["extra_weight_normalization"] is not None):
        raise ValueError("Frozen shared train-only u normalization differs")
    records = {}
    for key in SPECIALIST_KEYS:
        record = source["case_records"][key]
        rms, weight = float(record["truth_RMS_mm"]), float(record["weight"])
        expected = scaler.scale_mm ** 2 / max(rms ** 2, RELATIVE_CASE_RMS_FLOOR_MM ** 2)
        if (key not in scaler.fit_case_keys or record["nodes"] != 179316 or not math.isfinite(rms)
                or rms <= 0 or not math.isfinite(weight) or weight != expected or record["floor_active"]):
            raise ValueError(f"Invalid frozen specialist weight: {key}")
        records[key] = record
    return {**copy.deepcopy(source), "case_records": records, "active_case_keys": list(SPECIALIST_KEYS),
            "normalization_refit": False, "original_scaler_and_weight_fit_scope": source["fit_scope"],
            "diagnostic_scope": "Three original TRAIN101_479 loads; original120-case scaler/weights retained"}


def _scope(args):
    """Strict parent metadata qualification, filtered to three fixed TRAIN labels."""
    if (args.steps != 10000 or args.eval_every != 1000 or args.learning_rate != .003 or args.seed != 2026
            or args.gradient_weight != 0.0004106650618638651 or args.cloud_points != 1024):
        raise ValueError("Specialist budget/seed/LR/calibration/endpoints are fixed")
    split, split_path, parent_keys, parent_files, contract = shared_scope(args)
    if SPECIALIST_CAD not in split["groups"]["train"] or not set(SPECIALIST_KEYS).issubset(parent_keys["train"]):
        raise ValueError("Specialist CAD must remain in the original TRAIN scope")
    reference_directory = _path(args.reference_run)
    if reference_directory != (ROOT / "runs/geobc_exploration/displacement_gradient_calibrated_10k").resolve():
        raise ValueError("Only the completed original shared calibrated run is a reference")
    config_path, summary_path = reference_directory / "config.json", reference_directory / "summary.json"
    reference, summary = _json(config_path), _json(summary_path)
    if (summary.get("status") != "completed" or summary["step"] != 10000
            or summary["final_metrics"]["step"] != 10000 or reference["contract"] != contract
            or summary["config"] != reference):
        raise ValueError("Shared reference must be complete and match the unchanged parent recipe/source contract")
    scaler = DisplacementScaler.from_record(reference["scaler"])
    weights = _fixed_loss_record(reference, scaler)
    final_train = summary["final_metrics"]["train"]
    per = {key: final_train["per_case"][key] for key in SPECIALIST_KEYS}
    strain_per = {key: final_train["strain"]["per_case"][key] for key in SPECIALIST_KEYS}
    worst = max(per, key=lambda key: per[key]["relative_l2"])
    strain_worst = max(strain_per, key=lambda key: strain_per[key]["relative_l2"])
    reference_final = {"step": 10000, "source": "completed summary.final_metrics.train; selected best8000 is NOT used",
        "per_case": per, "case_mean_relative_l2": float(np.mean([item["relative_l2"] for item in per.values()])),
        "worst_case_key": worst, "worst_case_relative_l2": per[worst]["relative_l2"],
        "strain": {"per_case": strain_per, "case_mean_relative_l2": float(np.mean([item["relative_l2"] for item in strain_per.values()])),
                   "worst_case_key": strain_worst, "worst_case_relative_l2": strain_per[strain_worst]["relative_l2"]}}
    cloud_path = reference_directory / "geometry_clouds.npz"
    contract["version"] = 7
    contract["specialist"] = {"CAD": SPECIALIST_CAD, "case_keys": list(SPECIALIST_KEYS), "validation_cases": [],
        "fixed_endpoints": list(ENDPOINTS), "checkpoint_selection": "fixed10000_endpoint_no_selection",
        "case_slot_sampling": "Four uniform case slots WITH replacement from three loads; all mixed node/tet queries retained",
        "context": "Exactly one original xyz3 PointNet CAD context; no role5 model",
        "reference_config": {**_identity(config_path), "sha256": _sha(config_path.read_bytes())},
        "reference_summary": {**_identity(summary_path), "sha256": _sha(summary_path.read_bytes())},
        "reference_cloud": _identity(cloud_path), "reference_final_10000": reference_final,
        "fixed_scaler": reference["scaler"], "frozen_active_u_case_weights": weights["case_records"],
        "reference_initial_parameters": reference["initial_parameters"], "refit_label_scaler_or_weights": False,
        "parent_scope_metadata_retained": "Full strict40/5 qualification is metadata only; only three filtered NPZ targets are opened by this diagnostic",
        "expected_per_load_exposure": {"specialist250": 250 * 4 / 3, "shared10000": 10000 * 4 / 120, "specialist10000": 10000 * 4 / 3},
        "interpretation": "Dedicated per-CAD parameters/optimization and40x case exposure confound10k improvement; not a method or transfer result"}
    contract["code_sha256"]["specialist_train.py"] = _sha((ROOT / "geobc/specialist_train.py").read_bytes())
    return split, split_path, {"train": list(SPECIALIST_KEYS), "validation": []}, {key: parent_files[key] for key in SPECIALIST_KEYS}, contract


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
        raise ValueError("Resume must use this output's last.pt to preserve committed endpoint state")
    if (output / "summary.json").is_file():
        summary = _json(output / "summary.json")
        if summary.get("status") == "completed":
            if (summary["config"].get("contract") != contract
                    or summary.get("checkpoint_selection") != "fixed10000_endpoint_no_selection"
                    or summary.get("best_step") != ENDPOINTS[-1]
                    or not summary.get("best_fields_are_fixed_endpoint_aliases")
                    or set(summary.get("endpoint_metrics", {})) != {str(step) for step in ENDPOINTS}):
                raise ValueError("Completed specialist has a different strict endpoint contract")
            for name in ("last.pt", "endpoint_250.pt", "endpoint_10000.pt", "config.json", "metrics_best.json", "metrics_latest.json", "metrics_endpoint_250.json", "metrics_endpoint_10000.json", "sampling_first_batch.json"):
                if not (output / name).is_file():
                    raise ValueError(f"Completed specialist lacks {name}")
            if (_json(output / "config.json") != summary["config"]
                    or _json(output / "metrics_endpoint_10000.json") != summary["best_metrics"]
                    or _json(output / "metrics_endpoint_250.json") != summary["endpoint_metrics"]["250"]
                    or _json(output / "metrics_best.json") != summary["best_metrics"]
                    or _json(output / "metrics_latest.json") != summary["final_metrics"]):
                raise ValueError("Completed specialist endpoint/config artifacts disagree")
            return summary
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError("Incomplete nonempty output requires explicit --resume")
    stored_config = _json(output / "config.json") if resume is not None else None
    if resume is not None and (not resume.is_file() or stored_config.get("contract") != contract):
        raise ValueError("Resume checkpoint is missing or its strict contract changed")
    output.mkdir(parents=True, exist_ok=True)
    status = {"phase": "loading", "physics_mode": args.physics_mode, "u_loss": args.u_loss, "step": 0,
              "gradient_weight": args.gradient_weight, "diagnostic": "fixed_train_single_CAD", "CAD": SPECIALIST_CAD, "total_steps": args.steps, "started_at_utc": utc_now()}
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
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        guard = _load_basis(_path(args.load_basis), split)
        reference_config = _json(_path(contract["specialist"]["reference_config"]["path"]))
        train_cases = load_displacement_cases(keys["train"], files, guard, device)
        val_cases = []
        scaler = DisplacementScaler.from_record(reference_config["scaler"])
        loss_record = _fixed_loss_record(reference_config, scaler)
        if any(case.count != loss_record["case_records"][case.key]["nodes"] for case in train_cases):
            raise ValueError("Fixed specialist label node counts differ from reference normalizer metadata")
        loss_weights = torch.as_tensor([loss_record["case_records"][case.key]["weight"] for case in train_cases], dtype=torch.float32, device=device)
        if not torch.isfinite(loss_weights).all():
            raise ValueError("Frozen case weights are not finite in training precision")
        all_cases = train_cases + val_cases
        for case in all_cases:
            case.target = torch.as_tensor(scaler.encode(case.physical_u), device=device)
        cad_keys = [SPECIALIST_CAD]
        cad_to_row = {cad: number for number, cad in enumerate(cad_keys)}
        from .gradient_supervision import load_gradient_manifest, load_gradient_cache
        gradient_manifest = load_gradient_manifest(_path(args.gradient_cache))
        gradient_data, gradient_records = {}, {}
        by_cad = {cad: [] for cad in cad_keys}
        for case in all_cases:
            by_cad[case.key.split("/", 1)[0]].append(case)
        for cad in cad_keys:
            cache = load_gradient_cache(_path(args.gradient_cache), cad,
                xyz=by_cad[cad][0].xyz, manifest=gradient_manifest)
            for case in by_cad[cad]:
                cache.verify_xyz(case.xyz)
            gradient_data[cad] = {"cache": cache, "tensors": cache.tensor(device=device)}
            gradient_records[cad] = cache.record()
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
        with np.load(_path(contract["specialist"]["reference_cloud"]["path"]), allow_pickle=False) as reference_cloud:
            row = list(reference_cloud["cad_keys"]).index(SPECIALIST_CAD)
            if (not np.array_equal(reference_cloud["points"][row], cloud_array[0])
                    or not np.array_equal(reference_cloud["mask"][row], mask_array[0])
                    or not np.array_equal(reference_cloud["node_indices"][row], clouds[0].node_indices)):
                raise ValueError("Specialist must reuse the exact original canonical xyz3 cloud")
        geometry_clouds, geometry_masks = torch.as_tensor(cloud_array, device=device), torch.as_tensor(mask_array, device=device)
        model_config = PhysicsFieldConfig(mode=args.physics_mode,
            geometry_mode="shape" if args.variant == "geometry_pointnet" else "none",
            condition_dim=25 + (LOCAL_DIM if args.local_mode != "none" else 0), descriptor_dim=args.geometry_dim,
            hidden_dim=args.hidden_dim, cloud_points=args.cloud_points, seed=args.seed)
        model = PhysicsField(model_config, device)
        initial_parameters = _initial_parameters(model)
        if (initial_parameters != reference_config["initial_parameters"]
                or _sha(model.decoder.native.params.detach().cpu().numpy().tobytes()) != reference_config["initial_decoder_parameter_sha256"]
                or model.source_config != reference_config["model"]):
            raise ValueError("Fresh specialist architecture/initial parameters differ from the frozen shared xyz3 reference")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, fused=True)
        case_rng = np.random.default_rng(args.seed + 23)
        node_rng = torch.Generator(device=device).manual_seed(args.seed + 17)
        tet_rng = torch.Generator(device=device).manual_seed(args.seed + 41)
        config = {"contract": contract, "method": "Fixed TRAIN101_479 specialist representability/optimization diagnostic",
            "physics_mode": args.physics_mode, "variant": args.variant,
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "split_source": str(split_path), "scope": "Only fixed TRAIN101_479 x ver/hor/tor; no validation/dia/finaltest fields",
            "groups": {"train": [SPECIALIST_CAD], "validation": []}, "strict_parent_groups": split["groups"], "train_loads": list(BASIS_ORDER), "heldout_load": "dia",
            "cases": {group: [{"sample_key": case.key, "source": str(case.source), "nodes": case.count,
                               "alpha": guard_case_alpha(guard, case.key)} for case in cases]
                      for group, cases in (("train", train_cases), ("validation", val_cases))},
            "scaler": scaler.record(), "stress_reference": contract["stress_reference"],
            "optimizer": {"name": "Adam", "fused": True, "learning_rate": args.learning_rate,
                          "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0.0},
            "basis_order": list(BASIS_ORDER), "basis_layout": "[N,basis_ver_hor_tor,component_xyz]",
            "force_moment_slots": list(range(7, 13)), "model": model.source_config,
            "initial_decoder_parameter_sha256": _sha(model.decoder.native.params.detach().cpu().numpy().tobytes()),
            "initial_parameters": initial_parameters,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "decoder_parameter_count": sum(parameter.numel() for parameter in model.decoder.parameters()),
            "geometry": {"coordinates_sha256": _sha(cloud_array.tobytes()), "mask_sha256": _sha(mask_array.tobytes()),
                         "cloud_points": args.cloud_points, "source": "canonical ver geometry only; same context for all loads"},
            "local_geometry": {"mode": args.local_mode, "feature_names": list(LOCAL_NAMES) if args.local_mode != "none" else [], "CADs": local_records},
            "sampling": "Four uniformly sampled case SLOTS WITH replacement from three known loads; each6400 uniform nodes+1600 uniform tetra sets x4; duplicate slots/queries retained",
            "sampling_rng": {"case": "NumPy default_rng(seed+23)", "node": "torch.Generator(device),seed+17", "tet": "torch.Generator(device),seed+41"},
            "change_from_historical": "Specialist has one CAD and four with-replacement slots instead of four distinct cases from120; same architecture/loss/node-tet sampler, dedicated optimization and40x expected per-load exposure at10k are diagnostics, not a matched generalization comparison",
            "gradient_supervision": {"gradient_weight": args.gradient_weight, "cache": str(_path(args.gradient_cache)),
                "manifest_sha256": contract["gradient_supervision"]["manifest_sha256"], "CADs": gradient_records,
                "objective": contract["gradient_supervision"]["formula"], "tensor_Frobenius_weights": [1,1,1,2,2,2],
                "floor": STRAIN_FLOOR, "floor_required_inactive": True, "FP64_physical_corner_arithmetic": True,
                "prediction": "parameter-first-order derivatives through four model u values; no coordinate or second derivatives",
                "lambda0": "same strain forward arithmetic, detached from backward; u loss on same12800 mixed queries",
                "selection": "Fixed endpoints250/10000 only; neither u nor strain selects a step",
                "evaluation_forward": "native9 basis once perCAD, exact unit known-load selection; same predictions reused forstrain",
                "GT_input_features": False, "GT_stress_in_optimizer": False,
                "cost": gradient_manifest.get("cost", gradient_manifest.get("elapsed_seconds"))},
            "u_loss": loss_record,
            "loss": "case-balanced relative-case MSE of3 signed u components; fixed train-only RMS weights, unchanged common zero-center decode"
                    if args.u_loss == "relative_case_mse" else "case-balanced MSE of3 signed u components with train-only common RMS and exactly zero center",
            "checkpoint_selection": "fixed10000_endpoint_no_selection", "fixed_endpoints": list(ENDPOINTS),
            "reference_final_10000": contract["specialist"]["reference_final_10000"],
            "specialist": contract["specialist"],
            "stress_policy": "freeze independently supervised strict stress branch; do not combine stress or train it with u loss",
            "precision": "PointNet float32/high TF32; native TCNN FP16 compute/FP32 master; physical u/strain geometry arithmetic and metrics float64",
            "total_steps": args.steps, "torch_version": torch.__version__, "started_at_utc": utc_now()}
        gradient_check = None
        endpoint_metrics = {}
        latest_loss_components = None
        if resume is not None:
            for name in ("contract", "scaler", "u_loss", "model", "geometry", "local_geometry", "cases", "precision", "gradient_supervision", "initial_parameters", "specialist"):
                if config[name] != stored_config[name]:
                    raise ValueError(f"Resume {name} differs")
            saved = torch.load(resume, map_location=device, weights_only=False)
            if saved["config"] != stored_config:
                raise ValueError("last.pt and config.json differ")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            start_step = int(saved["step"])
            if not 0 <= start_step <= args.steps:
                raise ValueError("Resume step outside fixed budget")
            case_rng.bit_generator.state = saved["case_rng_state"]
            node_rng.set_state(saved["node_rng_state"].cpu())
            tet_rng.set_state(saved["tet_rng_state"].cpu())
            torch.set_rng_state(saved["torch_rng_state"].cpu())
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng_state"]])
            np.random.set_state(saved["numpy_rng_state"])
            random.setstate(saved["python_rng_state"])
            gradient_check, prior_elapsed = saved.get("encoder_gradient_check"), float(saved["elapsed_seconds"])
            endpoint_metrics = saved["endpoint_metrics"]
            if any(int(key) not in ENDPOINTS or int(key) > start_step or record["step"] != int(key) for key, record in endpoint_metrics.items()):
                raise ValueError("Saved fixed endpoints are inconsistent")
            for key, record in endpoint_metrics.items():
                path = output / f"endpoint_{key}.pt"
                if not path.is_file():
                    if int(key) != start_step:
                        raise ValueError("Earlier fixed endpoint checkpoint missing; do not replace with later weights")
                    _save_checkpoint(saved, path)
                write_json(output / f"metrics_endpoint_{key}.json", record)
            last = saved["metrics"]
            if last is not None:
                write_json(output / "metrics_latest.json", last)
                _restore_history_row(output, last)
                latest_loss_components = last.get("training_loss_components")
            config = stored_config
        else:
            write_json(output / "config.json", config)
            write_json(output / "cloud_metadata.json", {"CADs": {cad: cloud.record() for cad, cloud in zip(cad_keys, clouds)}, "target_independent": True, "reference_exact_xyz_mask_indices": True})
            np.savez(output / "geometry_clouds.npz", points=cloud_array, mask=mask_array, cad_keys=np.asarray(cad_keys), node_indices=np.stack([cloud.node_indices for cloud in clouds]))
            last = None

        def checkpoint_payload(step, metrics):
            final = endpoint_metrics.get(str(ENDPOINTS[-1]))
            return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                    "step": step, "scaler": scaler.record(), "metrics": metrics,
                    "best_step": ENDPOINTS[-1] if final is not None else None, "best_score": None, "best_metrics": final,
                    "endpoint_metrics": endpoint_metrics, "checkpoint_selection": "fixed10000_endpoint_no_selection", "elapsed_seconds": elapsed(),
                    "encoder_gradient_check": gradient_check, "case_rng_state": case_rng.bit_generator.state,
                    "node_rng_state": node_rng.get_state(), "torch_rng_state": torch.get_rng_state(), "tet_rng_state": tet_rng.get_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all(), "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate()}

        @torch.no_grad()
        def evaluate(step, loss):
            eval_began = time.monotonic()
            model.eval()
            update("evaluating", step=step)
            contexts = model.encode_geometry(geometry_clouds, geometry_masks)
            train = _evaluate_group(model, train_cases, contexts, cad_to_row, scaler, args.eval_batch_size, gradient_data)
            metrics = {"step": step, "selection_score": None, "checkpoint_selection": "fixed10000_endpoint_no_selection",
                       "training_loss": loss, "train": train, "elapsed_seconds": elapsed(),
                       "eval_seconds": time.monotonic() - eval_began, "encoder_gradient_check": gradient_check,
                       "training_loss_components": latest_loss_components, "memory": _memory_record(device)}
            if step in ENDPOINTS:
                endpoint_metrics[str(step)] = metrics
            payload = checkpoint_payload(step, metrics)
            _save_checkpoint(payload, output / "last.pt")
            if step in ENDPOINTS:
                _save_checkpoint(payload, output / f"endpoint_{step}.pt")
                write_json(output / f"metrics_endpoint_{step}.json", metrics)
            write_json(output / "metrics_latest.json", metrics)
            _restore_history_row(output, metrics)
            update("training", step=step, latest_train_case_mean_l2=train["case_mean_relative_l2"],
                   latest_train_strain_case_mean_l2=train["strain"]["case_mean_relative_l2"])
            print(f"specialist {step}/{args.steps} u={100*train['case_mean_relative_l2']:.4f}% P1={100*train['strain']['case_mean_relative_l2']:.4f}% fixed_endpoint={step in ENDPOINTS}", flush=True)
            return metrics
        update("training", step=start_step, resume_step=start_step)
        if resume is None:
            _save_checkpoint(checkpoint_payload(0, None), output / "last.pt")
        nodes_per_case = args.batch_size // args.cases_per_batch
        for step in range(start_step + 1, args.steps + 1):
            model.train()
            selected = case_rng.choice(len(train_cases), args.cases_per_batch, replace=True)
            batch_cases = [train_cases[int(index)] for index in selected]
            cads = list(dict.fromkeys(case.key.split("/", 1)[0] for case in batch_cases))
            rows = torch.tensor([cad_to_row[cad] for cad in cads], device=device)
            context = model.encode_geometry(geometry_clouds[rows], geometry_masks[rows])
            check_gradient = model.encoder is not None and gradient_check is None
            if check_gradient:
                context.retain_grad()
            local_rows = {cad: number for number, cad in enumerate(cads)}
            xyz, features, alpha, target, indices, witness_nodes, sampled_cells = [], [], [], [], [], [], []
            for case in batch_cases:
                tensors = gradient_data[case.key.split("/", 1)[0]]["tensors"]
                nodes, uniform_nodes, cells, corners = mixed_query_indices(case, tensors, node_rng, tet_rng)
                sampled_cells.append(cells)
                xyz.append(case.xyz[nodes]); features.append(case.features[nodes]); target.append(case.target[nodes])
                alpha.append(case.alpha.expand(nodes_per_case, -1))
                indices.append(torch.full((nodes_per_case,), local_rows[case.key.split("/", 1)[0]], dtype=torch.long, device=device))
                if step == start_step + 1:
                    witness_nodes.extend(value.cpu().numpy().tobytes() for value in (uniform_nodes, cells, corners))
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.cat(xyz), torch.cat(features), torch.cat(alpha), cad_index=torch.cat(indices), context=context)
            selected_rows = torch.as_tensor(selected, dtype=torch.long, device=device)
            u_loss_value = relative_case_mse(prediction, torch.cat(target), loss_weights[selected_rows])
            gradient_loss_value = gradient_batch_loss(prediction, batch_cases, sampled_cells,
                gradient_data, scaler.scale_mm, args.gradient_weight)
            loss = u_loss_value + args.gradient_weight * gradient_loss_value
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite u loss at step{step}")
            loss.backward()
            if check_gradient:
                gradient_check = _encoder_gradient(model, context)
                decoder_parameters = list(model.decoder.parameters())
                if not decoder_parameters or any(parameter.grad is None for parameter in decoder_parameters):
                    raise RuntimeError("First update has missing decoder parameter gradients")
                if any(not bool(torch.isfinite(parameter.grad).all()) for parameter in decoder_parameters):
                    raise FloatingPointError("First update has nonfinite decoder parameter gradients")
                decoder_gradient_l1 = sum(float(parameter.grad.detach().double().abs().sum())
                                          for parameter in decoder_parameters)
                if not math.isfinite(decoder_gradient_l1) or decoder_gradient_l1 <= 0:
                    raise FloatingPointError("First update decoder gradient L1 must be finite and positive")
                gradient_check["decoder_gradient_l1"] = decoder_gradient_l1
                gradient_check["decoder_gradients_present_and_finite"] = True
                gradient_check["decoder_parameter_tensors"] = len(decoder_parameters)
                gradient_check["step"] = step
                write_json(output / "encoder_gradient_check.json", gradient_check)
            optimizer.step()
            latest_loss_components = {"relative_case_u_loss": float(u_loss_value.detach()),
                                     "unweighted_relative_cell_strain_loss": float(gradient_loss_value.detach()),
                                     "gradient_weight": args.gradient_weight, "total": float(loss.detach())}
            if step == start_step + 1:
                witness = selected.astype(np.int64).tobytes() + b"".join(witness_nodes)
                write_json(output / ("sampling_first_batch.json" if start_step == 0 else "sampling_resume_batch.json"),
                           {"step": step, "case_keys": [case.key for case in batch_cases], "nodes_per_case": nodes_per_case,
                            "uniform_nodes_per_case": UNIFORM_NODES, "uniform_cells_per_case": CELLS_PER_CASE,
                            "corner_queries_per_case": 4 * CELLS_PER_CASE,
                            "witness_includes": "ordered caseindices + ALL uniformnodeindices/tetindices/fourcornerindices",
                            "witness_sha256": _sha(witness), "sampler_streams_independent_of_model": True, "case_slots_with_replacement": True, "single_CAD_context": True})
            if step % 100 == 0:
                update("training", step=step, training_loss=float(loss.detach()), loss_components=latest_loss_components,
                       memory=_memory_record(device))
            if step in ENDPOINTS or step % args.eval_every == 0:
                last = evaluate(step, float(loss.detach()))
        if set(endpoint_metrics) != {str(step) for step in ENDPOINTS} or last is None or last["step"] != ENDPOINTS[-1]:
            raise ValueError("Both preregistered fixed endpoint evaluations must be committed")
        chosen = endpoint_metrics[str(ENDPOINTS[-1])]
        write_json(output / "metrics_best.json", chosen)
        update("exporting_fixed_endpoint", step=args.steps)
        folder = output / "predictions_endpoint_10000"
        folder.mkdir(exist_ok=True)
        index = {}
        model.eval()
        with torch.no_grad():
            contexts = model.encode_geometry(geometry_clouds, geometry_masks)
            for number, case in enumerate(train_cases):
                pred = _predict(model, case, contexts, scaler, args.eval_batch_size)
                if not np.isclose(displacement_metrics(case.physical_u, pred)["relative_l2"], chosen["train"]["per_case"][case.key]["relative_l2"], rtol=1e-5, atol=1e-8):
                    raise ValueError("Fixed endpoint prediction differs from committed metric")
                filename = f"case_{number:03d}.npz"
                np.savez(folder / filename, xyz=case.xyz.cpu().numpy(), u_true_mm=case.physical_u, u_pred_mm=pred,
                         alpha=case.alpha.cpu().numpy(), sample_key=np.asarray(case.key), step=np.asarray(ENDPOINTS[-1]),
                         channels=np.asarray(["ux_mm", "uy_mm", "uz_mm"]), u_scale_mm=np.asarray(scaler.scale_mm))
                index[case.key] = {"file": filename, "source": str(case.source), "group": "train", "nodes": case.count}
        write_json(folder / "index.json", {"step": ENDPOINTS[-1], "cases": index, "policy": "All three fixed TRAIN loads; fixed endpoint, no best selection"})
        summary = {"status": "completed", "mode": "fixed_train_single_CAD_specialist", "step": args.steps,
                   "best_step": ENDPOINTS[-1], "best_score": None, "best_metrics": chosen, "final_metrics": last,
                   "checkpoint_selection": "fixed10000_endpoint_no_selection", "best_fields_are_fixed_endpoint_aliases": True,
                   "endpoint_metrics": endpoint_metrics, "fixed_endpoints": list(ENDPOINTS), "config": config,
                   "reference_final_10000": contract["specialist"]["reference_final_10000"],
                   "elapsed_seconds": elapsed(), "memory": _memory_record(device), "gradient_weight": args.gradient_weight,
                   "stress_reference": config["stress_reference"], "encoder_gradient_check": gradient_check,
                   "representative_predictions": str(folder / "index.json"), "completed_at_utc": utc_now()}
        write_json(output / "summary.json", summary)
        update("completed", step=args.steps)
        return summary
    except Exception as error:
        update("failed", error=f"{type(error).__name__}: {error}")
        raise


def parser():
    result = shared_parser()
    result.description = __doc__
    result.add_argument("--reference-run", type=Path, default=Path("runs/geobc_exploration/displacement_gradient_calibrated_10k"))
    return result


if __name__ == "__main__":
    run(parser().parse_args())
