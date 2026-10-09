"""Matched five-channel BC-role displacement teachers with fixed calibrated strain loss.

Both zero/actual role branches use the same five-input PointNet, calibrated
coefficient and mixed uniform-node/tetra-corner sampler. The old three-channel
encoder is context only, not the matched control. P1 targets derive from train
displacement labels, never input features; best selection stays u-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .physics_train import (DisplacementScaler, load_displacement_cases, training_loss_record,
                            relative_case_mse, displacement_metrics, guard_case_alpha,
                            _scope as physics_scope, _load_basis, _predict)
from .conditional_train import (_path, _encoder_gradient, _restore_history_row,
                                _save_checkpoint, _initial_parameters)
from .bc_role_condition import load_bc_role_cloud
from .local_geometry import FEATURE_DIM as LOCAL_DIM, FEATURE_NAMES as LOCAL_NAMES, load_local_geometry
from .train import utc_now, write_json

ROOT = Path(__file__).resolve().parents[1]
BASIS_ORDER = ("ver", "hor", "tor")
STRAIN_FLOOR = 1e-24
UNIFORM_NODES = 6400
CELLS_PER_CASE = 1600
NODES_PER_CASE = 12800


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _scope(args):
    """Five-tuple strict metadata-only contract, including train-only gradient norms."""
    if args.bc_role_mode not in ("zero", "actual") or args.gradient_weight != 0.0004106650618638651:
        raise ValueError("BC-role pair requires zero|actual roles with the fixed calibrated gradient coefficient")
    reference_balance_hash = "b45b5734c91e5112b83da511d90a2c11b065b5ec2099d24b8a3b432efa25265a"
    if _sha((ROOT / "geobc/gradient_balance_train.py").read_bytes()) != reference_balance_hash:
        raise ValueError("Frozen calibrated trainer source changed")
    calibration_path = _path(args.gradient_calibration)
    calibration_bytes = calibration_path.read_bytes()
    calibration = json.loads(calibration_bytes)
    locked_weight = 0.0004106650618638651
    original_hash = "a27c3845507b60b86b7792a5f1cdcec184f969b26edc83d454665ce5e54eb393"
    if (calibration.get("status") != "predeclared"
            or calibration.get("gradient_weight") != locked_weight
            or calibration.get("source_module_sha256") != original_hash
            or calibration.get("source_run") != "runs/geobc_exploration/displacement_gradient_zero_10k"
            or calibration.get("steps") != [6000, 7000, 8000, 9000, 10000]
            or _sha(_path(calibration["history_source"]).read_bytes()) != calibration["history_sha256"]
            or _sha((ROOT / "geobc/gradient_train.py").read_bytes()) != original_hash):
        raise ValueError("Calibration must be the predeclared fixed train-history coefficient from the frozen lambda0 run")
    if (args.physics_mode != "linear_basis" or args.u_loss != "relative_case_mse"
            or args.local_mode != "actual" or args.variant != "geometry_pointnet"
            or args.hidden_dim != 128 or args.geometry_dim != 32
            or args.batch_size != 51200 or args.cases_per_batch != 4
            or args.gradient_weight not in (0.0, locked_weight)):
        raise ValueError("Matched gradient pair requires linear/relative-case-u, PointNet128/global32/localactual,51200/4cases,lambda0|fixed-calibration")
    split, split_path, keys, files, contract = physics_scope(args)
    from .gradient_supervision import load_gradient_manifest
    directory = _path(args.gradient_cache)
    manifest_path = directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = load_gradient_manifest(directory)
    allowed = split["groups"]["train"] + split["groups"]["validation"]
    if manifest["groups"] != split["groups"] or set(manifest["CADs"]) != set(allowed):
        raise ValueError("Gradient cache differs from strict40/5 CAD split")
    identities = []
    for cad in allowed:
        record = manifest["CADs"][cad]
        path = directory / f"{cad}.npz"
        stat = path.stat()
        if record["file"] != path.name or record["bytes"] != stat.st_size or record["mtime_ns"] != stat.st_mtime_ns:
            raise ValueError(f"{cad}: gradient cache identity changed")
        if len(record["displacement_sources"]) != 3:
            raise ValueError("Gradient references must use exactly knownver/hor/tor sources")
        for load, source in zip(BASIS_ORDER, record["displacement_sources"]):
            key = f"{cad}/{load}"
            source_stat = files[key].stat()
            if (source["key"] != key or _path(source["path"]) != files[key]
                    or source["bytes"] != source_stat.st_size or source["mtime_ns"] != source_stat.st_mtime_ns):
                raise ValueError("Gradient reference and requested signed-u sources differ")
        if cad in split["groups"]["train"]:
            for load in BASIS_ORDER:
                norm = float(record["loss_normalization_by_load"][load])
                if not math.isfinite(norm) or norm <= STRAIN_FLOOR or record["floor_active_by_load"][load]:
                    raise ValueError(f"{cad}/{load}: inactive1e-24 strain floor required")
        elif record["loss_normalization_by_load"] is not None:
            raise ValueError("Validation gradient references cannot fit training loss norms")
        identities.append({"cad_id": cad, "path": str(path), "bytes": stat.st_size,
                           "mtime_ns": stat.st_mtime_ns, "metadata": record})
    contract["version"] = 4
    contract["requested"]["gradient_cache"] = str(directory)
    contract["requested"]["gradient_weight"] = args.gradient_weight
    contract["gradient_supervision"] = {"manifest": str(manifest_path), "manifest_sha256": _sha(manifest_bytes),
        "metadata": {k: v for k, v in manifest.items() if k != "CADs"}, "selected_sources": identities,
        "gradient_weight": args.gradient_weight, "floor": STRAIN_FLOOR,
        "formula": "mean_cases mean_uniform_cells[(V/mean_fullcase_V)*Frob_tensor_error_squared/R_case_squared]",
        "Frobenius_components": [1, 1, 1, 2, 2, 2], "batch_volume_renormalization": False,
        "loss_reference_scope": "train120 u-derived corner-P1 strain only; validation strain diagnostic only",
        "selection": "validation mean allnode physical u only; strain never selects",
        "sampler": {"uniform_original_nodes": UNIFORM_NODES, "uniform_cells": CELLS_PER_CASE,
                    "corners_per_cell": 4, "queries_per_case": NODES_PER_CASE,
                    "change_from_historical": "half corner-incidence replaces half pure-uniform original-node queries in BOTH branches"}}
    for name in ("gradient_train.py", "gradient_supervision.py"):
        contract["code_sha256"][name] = _sha((ROOT / "geobc" / name).read_bytes())
    calibration_stat = calibration_path.stat()
    contract["version"] = 5
    contract["requested"]["gradient_calibration"] = str(calibration_path)
    contract["gradient_calibration"] = {"path": str(calibration_path),
        "bytes": calibration_stat.st_size, "mtime_ns": calibration_stat.st_mtime_ns,
        "sha256": _sha(calibration_bytes), "metadata": calibration,
        "reference_module_sha256": original_hash}
    contract["code_sha256"]["gradient_balance_train.py"] = _sha((ROOT / "geobc/gradient_balance_train.py").read_bytes())
    contract["version"] = 6
    contract["requested"]["bc_role_mode"] = args.bc_role_mode
    cloud_sources = []
    for cad in allowed:
        path = files[f"{cad}/ver"]
        stat = path.stat()
        cloud_sources.append({"cad_id": cad, "canonical_case": f"{cad}/ver",
            "path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    reference_directory = ROOT / "runs/geobc_exploration/displacement_gradient_calibrated_10k"
    reference_config_path = reference_directory / "config.json"
    reference_summary_path = reference_directory / "summary.json"
    reference_cloud_path = reference_directory / "geometry_clouds.npz"
    reference_config = _json(reference_config_path)
    reference_summary = _json(reference_summary_path)
    if (reference_summary.get("status") != "completed" or reference_config["groups"] != split["groups"]
            or reference_config["geometry"]["cloud_points"] != args.cloud_points
            or reference_config["contract"]["code_sha256"]["gradient_balance_train.py"] != reference_balance_hash):
        raise ValueError("Cloud reference must be the completed frozen calibrated known-load geometry")
    reference_cloud_stat = reference_cloud_path.stat()
    contract["bc_roles"] = {"bc_role_mode": args.bc_role_mode,
        "cloud_channels": ["x", "y", "z", "support_union", "load_membership"],
        "role_definition": "support_union=any(original features[:,0:6] nonzero); load_membership=original features[:,6] nonzero",
        "role_source": "Canonical ver prepared input features0:7 only; no force/moment magnitude or target label",
        "cloud_points": args.cloud_points, "cloud_seed": reference_config["arguments"]["seed"],
        "training_seed": args.seed,
        "cloud_sampling": "Frozen load_geometry_cloud coordinates/mask/node_indices unchanged; roles appended at exactly those indices",
        "matched_encoder": "Five input channels for BOTH zero/actual; roles cleared only at zero-mode encoder entry",
        "scope": "One shared canonical cloud per allowed CAD; train40 and validation5 only",
        "target_independent": True, "canonical_sources": cloud_sources,
        "reference_cloud": {"path": str(reference_cloud_path), "bytes": reference_cloud_stat.st_size,
            "mtime_ns": reference_cloud_stat.st_mtime_ns,
            "config_path": str(reference_config_path), "config_sha256": _sha(reference_config_path.read_bytes()),
            "summary_path": str(reference_summary_path), "summary_sha256": _sha(reference_summary_path.read_bytes()),
            "seed": reference_config["arguments"]["seed"],
            "coordinates_sha256": reference_config["geometry"]["coordinates_sha256"],
            "mask_sha256": reference_config["geometry"]["mask_sha256"],
            "acceptance": "Reconstruction exact equality of coordinates, mask, CAD order and original node indices before training"},
        "reference_calibrated_module_sha256": reference_balance_hash,
        "old_three_channel_baseline_reused_as_control": False}
    for name in ("bc_role_train.py", "bc_role_field.py", "bc_role_condition.py"):
        contract["code_sha256"][name] = _sha((ROOT / "geobc" / name).read_bytes())
    return split, split_path, keys, files, contract


def _memory_record(device):
    try:
        import resource
        cpu_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    except ImportError:
        cpu_peak = None
    return {"cpu_peak_rss_bytes": cpu_peak,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None}


def mixed_query_indices(case, cache_tensors, node_rng, tet_rng):
    """Independent streams; all four tetra corners are queried in original node order."""
    uniform = torch.randint(case.count, (UNIFORM_NODES,), generator=node_rng, device=case.xyz.device)
    cells = torch.randint(len(cache_tensors["corners"]), (CELLS_PER_CASE,), generator=tet_rng, device=case.xyz.device)
    corners = cache_tensors["corners"][cells]
    return torch.cat((uniform, corners.reshape(-1))), uniform, cells, corners


def gradient_batch_loss(prediction, batch_cases, sampled_cells, gradient_data, scale_mm, weight):
    """No coordinate gradients or second derivatives. Same FP64 arithmetic both branches."""
    from .gradient_supervision import cell_tensor_strain, relative_frobenius_loss
    blocks = prediction.reshape(len(batch_cases), NODES_PER_CASE, 3)
    result = []
    for row, (case, cells) in enumerate(zip(batch_cases, sampled_cells)):
        item = gradient_data[case.key.split("/", 1)[0]]
        tensors, cache = item["tensors"], item["cache"]
        load = case.key.split("/", 1)[1]
        load_index = BASIS_ORDER.index(load)
        corners = blocks[row, UNIFORM_NODES:].reshape(CELLS_PER_CASE, 4, 3)
        # lambda0 computes identical values while deliberately excluding the
        # derivative target from its backward graph.
        physical = (corners if weight else corners.detach()).double() * scale_mm
        strain = cell_tensor_strain(physical, tensors["inverse_physical_edge_mm"][cells])
        target = tensors["reference_cell_tensor_strain"][cells, load_index].double()
        result.append(relative_frobenius_loss(strain, target, tensors["volumes"][cells],
            cache.mean_volume_mm3, cache.loss_normalization_by_load[load], STRAIN_FLOOR))
    return torch.stack(result).mean()


def _strain_metrics(truth, prediction):
    truth, prediction = np.asarray(truth, np.float64), np.asarray(prediction, np.float64)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 6 or not np.isfinite(prediction).all():
        raise ValueError("Expected finite matching allnode tensor6 strain")
    weights = np.asarray([1., 1., 1., 2., 2., 2.])
    numerator = float(np.sum((prediction - truth) ** 2 * weights))
    denominator = float(np.sum(truth ** 2 * weights))
    relative = math.sqrt(numerator / max(denominator, STRAIN_FLOOR))
    return {"nodes": len(truth), "relative_l2": relative, "symmetric_Frobenius_relative_l2": relative,
            "error_energy": numerator, "truth_energy": denominator,
            "definition": "allnode unweighted FrobeniusL2 of incident-volume-averaged cornerP1 tensor; shear squared weight2"}


@torch.no_grad()
def _evaluate_group(model, cases, contexts, cad_to_row, scaler, batch_size, gradient_data):
    from .gradient_supervision import cell_tensor_strain, recover_node_tensor_strain
    per, strain_per, truths, predictions = {}, {}, [], []
    grouped = {}
    for case in cases:
        grouped.setdefault(case.key.split("/", 1)[0], []).append(case)
    for cad, cad_cases in grouped.items():
        first = cad_cases[0]
        row = cad_to_row[cad]
        # Linear mode masks force slots; loader verifies identical geometry and
        # remaining static features across known loads. Native nine outputs are
        # evaluated once per CAD, then unit basis coefficients select each load.
        basis = np.empty((first.count, 3, 3), np.float32)
        for start in range(0, first.count, batch_size):
            stop = min(start + batch_size, first.count)
            basis[start:stop] = model.response_basis(first.xyz[start:stop], first.features[start:stop],
                context=contexts[row:row + 1]).float().cpu().numpy()
        for case in cad_cases:
            load = case.key.split("/", 1)[1]
            coefficients = case.alpha.cpu().numpy()
            if not np.array_equal(coefficients, np.eye(3, dtype=np.float32)[BASIS_ORDER.index(load)]):
                raise ValueError("Known-load fast evaluation requires verified canonical unit coefficients")
            pred = scaler.decode(np.einsum("l,nlc->nc", coefficients, basis))
            per[case.key] = displacement_metrics(case.physical_u, pred)
            truths.append(case.physical_u); predictions.append(pred)
            item = gradient_data[cad]; tensors = item["tensors"]
            physical = torch.as_tensor(pred, dtype=torch.float64, device=case.xyz.device)
            count = len(tensors["corners"])
            cell_prediction = torch.empty((count, 6), dtype=torch.float64, device=case.xyz.device)
            for start in range(0, count, batch_size):
                stop = min(start + batch_size, count)
                cell_prediction[start:stop] = cell_tensor_strain(physical[tensors["corners"][start:stop]],
                    tensors["inverse_physical_edge_mm"][start:stop])
            recovered = recover_node_tensor_strain(cell_prediction, tensors["volumes"], tensors["cells"], case.count, batch_size=batch_size)
            reference = recover_node_tensor_strain(tensors["reference_cell_tensor_strain"][:, BASIS_ORDER.index(load)].double(),
                tensors["volumes"], tensors["cells"], case.count, batch_size=batch_size)
            strain_per[case.key] = _strain_metrics(reference.cpu().numpy(), recovered.cpu().numpy())
            del physical, cell_prediction, recovered, reference
        del basis
    worst = max(per, key=lambda key: per[key]["relative_l2"])
    strain_worst = max(strain_per, key=lambda key: strain_per[key]["relative_l2"])
    numerator = sum(m["error_energy"] for m in strain_per.values())
    denominator = sum(m["truth_energy"] for m in strain_per.values())
    return {"per_case": per, "pooled": displacement_metrics(np.concatenate(truths), np.concatenate(predictions)),
            "case_mean_relative_l2": float(np.mean([m["relative_l2"] for m in per.values()])),
            "worst_case_key": worst, "worst_case_relative_l2": per[worst]["relative_l2"],
            "strain": {"per_case": strain_per, "case_mean_relative_l2": float(np.mean([m["relative_l2"] for m in strain_per.values()])),
                       "worst_case_key": strain_worst, "worst_case_relative_l2": strain_per[strain_worst]["relative_l2"],
                       "pooled": {"relative_l2": math.sqrt(numerator / max(denominator, STRAIN_FLOOR)), "error_energy": numerator, "truth_energy": denominator},
                       "selection_used": False, "scope": "all original nodes; P1 proxy diagnostic only"}}


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
            if (_json(output / "config.json") != summary["config"]
                    or _json(output / "metrics_best.json") != summary["best_metrics"]
                    or _json(output / "metrics_latest.json") != summary["final_metrics"]
                    or not (output / "sampling_first_batch.json").is_file()):
                raise ValueError("Completed mixed-gradient run configuration/metrics/witness differs")
            print(f"Reusing completed physics run {output}", flush=True)
            return summary
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError("Incomplete nonempty output requires explicit --resume")
    stored_config = _json(output / "config.json") if resume is not None else None
    if resume is not None and (not resume.is_file() or stored_config.get("contract") != contract):
        raise ValueError("Resume checkpoint is missing or its strict contract changed")
    output.mkdir(parents=True, exist_ok=True)
    status = {"phase": "loading", "physics_mode": args.physics_mode, "u_loss": args.u_loss, "step": 0,
              "gradient_weight": args.gradient_weight, "total_steps": args.steps, "started_at_utc": utc_now()}
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
        from .bc_role_field import PhysicsRoleField, PhysFieldRoleConfig
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
        train_cases = load_displacement_cases(keys["train"], files, guard, device)
        val_cases = load_displacement_cases(keys["validation"], files, guard, device)
        scaler = DisplacementScaler.from_record(stored_config["scaler"]) if stored_config else DisplacementScaler.fit(train_cases)
        if scaler.fit_case_keys != tuple(keys["train"]) or scaler.fit_nodes != sum(case.count for case in train_cases):
            raise ValueError("Saved u scaler scope/counts differ from the explicit training cases")
        loss_record = training_loss_record(train_cases, scaler, args.u_loss,
                                          stored_config["u_loss"] if stored_config else None)
        loss_weights = (torch.as_tensor([loss_record["case_records"][case.key]["weight"] for case in train_cases],
                                       dtype=torch.float32, device=device)
                        if args.u_loss == "relative_case_mse" else None)
        if loss_weights is not None and not torch.isfinite(loss_weights).all():
            raise ValueError("Relative-case weights are not finite in the training precision")
        all_cases = train_cases + val_cases
        for case in all_cases:
            case.target = torch.as_tensor(scaler.encode(case.physical_u), device=device)
        cad_keys = split["groups"]["train"] + split["groups"]["validation"]
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
        clouds = [load_bc_role_cloud(files[f"{cad}/ver"], cloud_points=args.cloud_points,
            seed=contract["bc_roles"]["reference_cloud"]["seed"], cad_key=cad) for cad in cad_keys]
        cloud_array, mask_array = np.stack([item.points for item in clouds]), np.stack([item.mask for item in clouds])
        if cloud_array.shape != (len(cad_keys), args.cloud_points, 5):
            raise ValueError("BC-role PointNet requires the same five-channel cloud in BOTH branches")
        if not np.isfinite(cloud_array).all() or not np.isin(cloud_array[..., 3:], (0, 1)).all():
            raise ValueError("Cloud role channels must be finite binary input membership")
        node_index_array = np.stack([cloud.node_indices for cloud in clouds])
        role_array = np.ascontiguousarray(cloud_array[..., 3:])
        if np.any(role_array[~mask_array] != 0):
            raise ValueError("Padded role channels must remain exactly zero")
        with np.load(_path(contract["bc_roles"]["reference_cloud"]["path"]), allow_pickle=False) as reference_cloud:
            if (list(reference_cloud["cad_keys"]) != cad_keys
                    or not np.array_equal(reference_cloud["points"], cloud_array[..., :3])
                    or not np.array_equal(reference_cloud["mask"], mask_array)
                    or not np.array_equal(reference_cloud["node_indices"], node_index_array)):
                raise ValueError("BC-role cloud must exactly reuse frozen calibrated xyz/mask/node indices")
        encoder_role_array = np.zeros_like(role_array) if args.bc_role_mode == "zero" else role_array
        geometry_clouds, geometry_masks = torch.as_tensor(cloud_array, device=device), torch.as_tensor(mask_array, device=device)
        model_config = PhysFieldRoleConfig(mode=args.physics_mode, bc_role_mode=args.bc_role_mode,
            geometry_mode="shape" if args.variant == "geometry_pointnet" else "none",
            condition_dim=25 + (LOCAL_DIM if args.local_mode != "none" else 0), descriptor_dim=args.geometry_dim,
            hidden_dim=args.hidden_dim, cloud_points=args.cloud_points, seed=args.seed)
        model = PhysicsRoleField(model_config, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, fused=True)
        case_rng = np.random.default_rng(args.seed + 23)
        node_rng = torch.Generator(device=device).manual_seed(args.seed + 17)
        tet_rng = torch.Generator(device=device).manual_seed(args.seed + 41)
        config = {"contract": contract, "method": "GeoBC matched five-channel BC-role PointNet displacement/gradient diagnostic",
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
            "initial_parameters": _initial_parameters(model),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "decoder_parameter_count": sum(parameter.numel() for parameter in model.decoder.parameters()),
            "geometry": {"coordinates_sha256": _sha(np.ascontiguousarray(cloud_array[..., :3]).tobytes()),
                         "cloud_features5_sha256": _sha(cloud_array.tobytes()),
                         "raw_roles_sha256": _sha(role_array.tobytes()),
                         "encoder_roles_sha256": _sha(encoder_role_array.tobytes()),
                         "node_indices_sha256": _sha(node_index_array.tobytes()),
                         "mask_sha256": _sha(mask_array.tobytes()), "bc_role_mode": args.bc_role_mode,
                         "cloud_points": args.cloud_points, "cloud_channels": 5,
                         "source": "canonical ver xyz plus input support-union/load-membership; same raw cloud and context geometry for all loads",
                         "reference_cloud": contract["bc_roles"]["reference_cloud"],
                         "reference_xyz_mask_node_indices_exact_equal": True,
                         "known_load_static_BC_agreement": "Full original features with only7:13 cleared hash-identical across all known loads, checked by frozen displacement loader",
                         "CADs": {cad: cloud.record() for cad, cloud in zip(cad_keys, clouds)}},
            "local_geometry": {"mode": args.local_mode, "feature_names": list(LOCAL_NAMES) if args.local_mode != "none" else [], "CADs": local_records},
            "sampling": "four distinct cases; each6400 uniform original nodes +1600 uniform tetra corner sets x4; duplicates retained in BOTH zero/actual BC-role branches",
            "sampling_rng": {"case": "NumPy default_rng(seed+23)", "node": "torch.Generator(device),seed+17", "tet": "torch.Generator(device),seed+41"},
            "change_from_historical": "Both new five-channel zero/actual role models retain the same calibrated12800 half-uniform/half-corner objective; the old three-channel encoder is contextual evidence and cannot be reused as this matched control",
            "gradient_supervision": {"gradient_weight": args.gradient_weight, "cache": str(_path(args.gradient_cache)),
                "manifest_sha256": contract["gradient_supervision"]["manifest_sha256"], "CADs": gradient_records,
                "objective": contract["gradient_supervision"]["formula"], "tensor_Frobenius_weights": [1,1,1,2,2,2],
                "floor": STRAIN_FLOOR, "floor_required_inactive": True, "FP64_physical_corner_arithmetic": True,
                "prediction": "parameter-first-order derivatives through four model u values; no coordinate or second derivatives",
                "lambda0": "same strain forward arithmetic, detached from backward; u loss on same12800 mixed queries",
                "selection": "u validation mean only; allnode recoveredP1 strain diagnostic at same selected ucheckpoint",
                "evaluation_forward": "native9 basis once perCAD, exact unit known-load selection; same predictions reused forstrain",
                "GT_input_features": False, "GT_stress_in_optimizer": False,
                "cost": gradient_manifest.get("cost", gradient_manifest.get("elapsed_seconds"))},
            "u_loss": loss_record,
            "loss": "case-balanced relative-case MSE of3 signed u components; fixed train-only RMS weights, unchanged common zero-center decode"
                    if args.u_loss == "relative_case_mse" else "case-balanced MSE of3 signed u components with train-only common RMS and exactly zero center",
            "checkpoint_selection": "minimum validation15-case mean all-original-node physical displacement vector relativeL2",
            "stress_policy": "freeze independently supervised strict stress branch; do not combine stress or train it with u loss",
            "precision": "PointNet float32/high TF32; native TCNN FP16 compute/FP32 master; physical u/strain geometry arithmetic and metrics float64",
            "total_steps": args.steps, "torch_version": torch.__version__, "started_at_utc": utc_now()}
        best_score, best_step, best_metrics, gradient_check = float("inf"), None, None, None
        if resume is not None:
            for name in ("contract", "scaler", "u_loss", "model", "geometry", "local_geometry", "cases", "precision", "gradient_supervision", "initial_parameters"):
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
            tet_rng.set_state(saved["tet_rng_state"].cpu())
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
                    "tet_rng_state": tet_rng.get_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all(), "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate()}
        @torch.no_grad()
        def evaluate(step, loss):
            nonlocal best_score, best_step, best_metrics
            eval_began = time.monotonic()
            model.eval()
            update("evaluating", step=step)
            contexts = model.encode_geometry(geometry_clouds, geometry_masks)
            train = _evaluate_group(model, train_cases, contexts, cad_to_row, scaler, args.eval_batch_size, gradient_data)
            validation = _evaluate_group(model, val_cases, contexts, cad_to_row, scaler, args.eval_batch_size, gradient_data)
            score = validation["case_mean_relative_l2"]
            metrics = {"step": step, "selection_score": score, "training_loss": loss,
                       "train": train, "validation": validation, "elapsed_seconds": elapsed(),
                       "eval_seconds": time.monotonic() - eval_began, "encoder_gradient_check": gradient_check,
                       "training_loss_components": latest_loss_components, "memory": _memory_record(device)}
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
        latest_loss_components = saved["metrics"].get("training_loss_components") if resume is not None else None
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
                            "witness_sha256": _sha(witness), "sampler_streams_independent_of_model": True})
            if step % 100 == 0:
                update("training", step=step, training_loss=float(loss.detach()), loss_components=latest_loss_components,
                       memory=_memory_record(device), best_step=best_step, best_score=best_score)
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
                   "memory": _memory_record(device), "gradient_weight": args.gradient_weight,
                   "stress_reference": config["stress_reference"], "encoder_gradient_check": gradient_check,
                   "representative_predictions": str(folder / "index.json"), "completed_at_utc": utc_now()}
        write_json(output / "summary.json", summary)
        update("completed", step=args.steps, best_step=best_step, best_score=best_score)
        return summary
    except Exception as error:
        update("failed", error=f"{type(error).__name__}: {error}")
        raise



def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--physics-mode", choices=("linear_basis",), default="linear_basis")
    result.add_argument("--u-loss", choices=("relative_case_mse",), default="relative_case_mse")
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
    result.add_argument("--local-mode", choices=("actual",), default="actual")
    result.add_argument("--local-geometry", type=Path, default=Path("data/geobc_local_geometry"))
    result.add_argument("--device", default="cuda")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--resume", nargs="?", const="last", default=None)
    result.add_argument("--gradient-cache", type=Path, default=Path("data/geobc_gradient_supervision"))
    result.add_argument("--gradient-weight", type=float, choices=(0.0004106650618638651,), default=0.0004106650618638651)
    result.add_argument("--bc-role-mode", choices=("zero", "actual"), required=True)
    result.add_argument("--gradient-calibration", type=Path, default=Path("notes/geobc_gradient_calibration.json"))
    return result


if __name__ == "__main__":
    run(parser().parse_args())
