"""Fresh matched fixed Hash feature bandwidth: full12 versus coarse8.

Both split-native models retain actual Euclidean5, BC-role5 and query/global
contexts. Only the nonlearned last8 encoded feature mask differs; all named
trainable parameter layouts, physical objectives and sampler remain matched.
The old fused distance pair is contextual and cannot substitute as a control.
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
                                _save_checkpoint)
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


def _query_scope(args):
    """Five-tuple strict metadata-only contract, including train-only gradient norms."""
    if args.bc_role_mode != "actual" or args.query_context_mode not in ("zero", "actual"):
        raise ValueError("Query context pair requires actual global BC roles and zero|actual query mode")
    reference_bc_hash = "9f7ca1b12c2358be6a936ff1dc0c99eb51a82f550d2d584dd0cbb39b4f2b3186"
    if _sha((ROOT / "geobc/bc_role_train.py").read_bytes()) != reference_bc_hash:
        raise ValueError("Frozen BC-role trainer source changed")
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
    role_directory = ROOT / "runs/geobc_exploration/displacement_bc_role_actual_10k"
    role_config_path, role_summary_path = role_directory / "config.json", role_directory / "summary.json"
    role_cloud_path = role_directory / "geometry_clouds.npz"
    role_config, role_summary = _json(role_config_path), _json(role_summary_path)
    if (role_summary.get("status") != "completed" or role_config["groups"] != split["groups"]
            or role_config["arguments"]["bc_role_mode"] != "actual"
            or role_config["geometry"]["cloud_channels"] != 5
            or role_config["contract"]["code_sha256"]["bc_role_train.py"] != reference_bc_hash):
        raise ValueError("Query input cloud must reference the completed frozen actual-BC-role5 inputs")
    role_cloud_stat = role_cloud_path.stat()
    contract["version"] = 8
    contract["requested"]["query_context_mode"] = args.query_context_mode
    contract["bc_roles"]["matched_encoder"] = "Both query branches retain actual canonical BC-role5 input; only query32 response is zeroed in the query control"
    contract["query_context"] = {"query_context_mode": args.query_context_mode,
        "sigma": 0.15, "dimension": 32, "query_chunk": 4096,
        "projection": "Shared learned linear point64->32,2080parameters present in BOTH branches",
        "weighting": "masked_softmax[-squared_normalized_coordinate_distance/(2*sigma^2)]; FP32 direct coordinate subtraction, no TF32 dot distance",
        "pool_output": "Gaussian weighted projected point32 -> non-affine LayerNorm32 -> tanh; bounded query32",
        "native_input_layout": "query_original25+local13 (38),query_context32,global_descriptor32; native Identity102 plus hash24",
        "zero_control": "Compute the same position-dependent projected context then multiply by0; projection/token gradients must be exactly0",
        "actual_control": "Retain query32 context; first projection/token gradients must be finite and nonzero",
        "reference_bc_module_sha256": reference_bc_hash,
        "reference_role_cloud": {"path": str(role_cloud_path), "bytes": role_cloud_stat.st_size,
            "mtime_ns": role_cloud_stat.st_mtime_ns,
            "config_path": str(role_config_path), "config_sha256": _sha(role_config_path.read_bytes()),
            "summary_path": str(role_summary_path), "summary_sha256": _sha(role_summary_path.read_bytes()),
            "raw_features5_sha256": role_config["geometry"]["cloud_features5_sha256"],
            "raw_roles_sha256": role_config["geometry"]["raw_roles_sha256"],
            "node_indices_sha256": role_config["geometry"]["node_indices_sha256"],
            "acceptance": "Exact45CAD raw5 cloud/mask/nodeindices/CAD order equality before training; no trained weights are loaded"},
        "old_global_only_control_reused": False, "GT_inputs": False,
        "selection": "Known-load all-node u validation mean only; P1 diagnostic never selects"}
    for name in ("query_context_train.py", "query_context_field.py"):
        contract["code_sha256"][name] = _sha((ROOT / "geobc" / name).read_bytes())
    return split, split_path, keys, files, contract


def _graph_scientific_record(record):
    """Exclude write-time cost/stat additions from the embedded scientific identity."""
    ignored = {"file", "bytes", "mtime_ns", "sha256", "timings", "elapsed_seconds", "cache_write_seconds"}
    return {name: value for name, value in record.items() if name not in ignored}


def _scope(args):
    """Strict five-tuple contract: JSON/source SHA/stat only, no array or model loading."""
    if args.distance_mode != "euclidean" or args.bandwidth_mode not in ("full12", "coarse8") or args.query_context_mode != "actual" or args.bc_role_mode != "actual":
        raise ValueError("Both bandwidth modes require the same actual Euclidean5/query32/BC-role5")
    reference_hash = "aa13d487202d9744ae6f60fdad1e9a761963de23dd48832f0ae9d0de50a1b3b5"
    frozen_ready = _json(ROOT / "notes/query_context_train_ready.json")
    for name, expected_hash in frozen_ready["source_sha256"].items():
        if _sha(_path(name).read_bytes()) != expected_hash:
            raise ValueError(f"Frozen query dependency source changed: {name}")
    split, split_path, keys, files, contract = _query_scope(args)
    plan_path = _path(args.bandwidth_plan)
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if _sha(plan_bytes) != "8fc4e3355c8892f143428383049e39fa6f0fe380968cbefd59bb3b3058e42f12" or plan.get("status") != "predeclared":
        raise ValueError("Hash bandwidth plan differs from its frozen predeclaration")
    architecture, recipe, scope = plan["architecture"], plan["recipe"], plan["scope"]
    if (plan.get("stage") != "matched_fixed_hash_bandwidth"
            or architecture["condition_dim"] != 43 or architecture["native_condition_dim"] != 107
            or architecture["distance_mode"] != "euclidean" or architecture["query_context_mode"] != "actual"
            or architecture["bc_role_mode"] != "actual" or architecture["new_learned_head_parameters"] != 0
            or plan["pair"]["modes"] != ["full12", "coarse8"] or plan["pair"]["historical_control_reused"]
            or plan["pair"]["warmstart"] or plan["selection"]["cutoff_sweep"]
            or plan["selection"]["chosen_active_levels"] != 8
            or scope["known_loads"] != list(BASIS_ORDER) or scope["train_CADs"] != 40 or scope["validation_CADs"] != 5
            or any(scope[name] for name in ("dia_enabled", "final_test_enabled", "stress_targets_enabled"))):
        raise ValueError("Fresh full12/coarse8 pair architecture or strict known-load scope changed")
    frozen_graph_hash = "52c7f6cc53a7f46fcb250d2815b092d472f94fe411cf68f66ea04560db26e35f"
    if _sha((ROOT / "geobc/graph_context_train.py").read_bytes()) != frozen_graph_hash:
        raise ValueError("Frozen parent distance trainer source changed")
    expected = {"seed": recipe["seed"], "steps": recipe["steps"], "batch_size": recipe["batch_size"],
        "cases_per_batch": recipe["case_slots"], "eval_every": recipe["eval_every"],
        "learning_rate": recipe["optimizer"]["learning_rate"], "gradient_weight": recipe["gradient_weight"],
        "cloud_points": 1024, "hidden_dim": 128, "geometry_dim": 32}
    if any(getattr(args, name) != value for name, value in expected.items()):
        raise ValueError("Requested recipe differs from the predeclared fixed-bandwidth pair")
    field_path = ROOT / "geobc/hash_bandwidth_field.py"
    field_ready_path = ROOT / "notes/hash_bandwidth_field_ready.json"
    field_ready = _json(field_ready_path)
    field_sha = field_ready["source_sha256"]
    if isinstance(field_sha, dict):field_sha = field_sha["geobc/hash_bandwidth_field.py"]
    if (field_ready.get("status") != "ready" or field_sha != _sha(field_path.read_bytes())
            or field_ready["plan_sha256"] != _sha(plan_bytes)):
        raise ValueError("Field must match its frozen CPU-ready source and candidate plan")

    normalization_directory = ROOT / "runs/geobc_exploration/displacement_query_context_actual_10k"
    normal_config_path, normal_summary_path = normalization_directory / "config.json", normalization_directory / "summary.json"
    normal_config, normal_summary = _json(normal_config_path), _json(normal_summary_path)
    scaler_record, loss_record = normal_config["scaler"], normal_config["u_loss"]
    if (normal_summary.get("status") != "completed" or normal_summary["config"] != normal_config
            or normal_config["groups"] != split["groups"] or normal_config["train_loads"] != list(BASIS_ORDER)
            or normal_config["contract"]["code_sha256"]["query_context_train.py"] != reference_hash
            or scaler_record["fit_case_keys"] != keys["train"] or scaler_record["scale_mm"] != recipe["u_scale_mm"]
            or scaler_record["center_mm"] != 0 or recipe["u_center_mm"] != 0
            or loss_record["name"] != "relative_case_mse" or loss_record["fit_case_keys"] != keys["train"]
            or set(loss_record["case_records"]) != set(keys["train"])
            or normal_config["gradient_supervision"]["manifest_sha256"] != contract["gradient_supervision"]["manifest_sha256"]):
        raise ValueError("Normalizer records must be the completed query-actual train120 records; no refit")

    directory = _path(args.graph_context)
    manifest_path = directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    ready_path = ROOT / "notes/graph_context_data_ready.json"
    data_ready = _json(ready_path)
    if (data_ready.get("status") != "ready" or not data_ready.get("complete")
            or data_ready["manifest_sha256"] != _sha(manifest_bytes)
            or data_ready["source_sha256"] != _sha((ROOT / "tools/prepare_graph_context.py").read_bytes())
            or _path(data_ready["manifest"]) != manifest_path
            or manifest.get("status") != "completed" or manifest.get("complete") is not True
            or manifest["groups"] != split["groups"] or manifest["split_sha256"] != _sha(split_path.read_bytes())
            or manifest["array_keys"] != ["euclidean", "graph"] or manifest["channels_per_array"] != 5
            or manifest["canonical_load"] != "ver" or manifest["labels_read"] != 0 or manifest["input_labels_read"] != 0
            or manifest["target_scalers_fit"] or manifest["model_or_weights_loaded"] or manifest["failures"]):
        raise ValueError("Target-independent distance cache readiness, source, arrays or split differs")
    if (_sha(manifest_bytes) != plan["data"]["manifest_sha256"]
            or _path(plan["data"]["graph_context"]) != directory or plan["data"]["array_key"] != "euclidean"
            or not plan["data"]["both_modes_same_distance_values"] or not plan["data"]["no_refit"]):
        raise ValueError("Both bandwidth modes must use the identical immutable Euclidean5 cache")
    data_contract = manifest["contract"]
    fingerprint = _sha(json.dumps(data_contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    if fingerprint != manifest["contract_sha256"] or fingerprint != data_ready["contract_sha256"]:
        raise ValueError("Graph-cache source contract fingerprint differs")
    for name, expected_sha in data_contract["source_sha256"].items():
        if _sha(_path(name).read_bytes()) != expected_sha:
            raise ValueError(f"Graph-cache preparation source changed: {name}")
    for identity in (data_contract["plan"], data_contract["split"]):
        path = _path(identity["path"])
        stat = path.stat()
        if stat.st_size != identity["bytes"] or stat.st_mtime_ns != identity["mtime_ns"] or _sha(path.read_bytes()) != identity["sha256"]:
            raise ValueError("Graph-cache metadata plan/split identity changed")
    allowed = split["groups"]["train"] + split["groups"]["validation"]
    if set(manifest["CADs"]) != set(allowed) or manifest["CAD_order"] != allowed:
        raise ValueError("Exactly the same45 train/validation CAD caches are required")
    identities = []
    gradient_records = {item["cad_id"]: item["metadata"] for item in contract["gradient_supervision"]["selected_sources"]}
    for cad in allowed:
        record = manifest["CADs"][cad]
        path = directory / f"{cad}.npz"
        stat = path.stat()
        groups = record["source_groups"]
        length = float(record["normalization"]["characteristic_length_mm"])
        if (record["file"] != path.name or record["bytes"] != stat.st_size or record["mtime_ns"] != stat.st_mtime_ns
                or record["status"] != "completed" or record["cad_id"] != cad or record["canonical_case"] != f"{cad}/ver"
                or record["contract_sha256"] != fingerprint or record["array_keys"] != ["euclidean", "graph"]
                or record["array_dtype"] != "float32" or record["finite"] is not True or record["target_independent"] is not True
                or record["labels_read"] != 0 or record["input_labels_read"] != 0 or record["targets_fitted"]
                or record["feature_names"] != manifest["feature_names"] or record["node_count"] < 1
                or record["xyz_sha256"] != gradient_records[cad]["xyz_sha256"]
                or not math.isfinite(length) or length <= 0 or len(record["normalization"]["bbox_min_mm"]) != 3
                or len(groups) != 5 or [group["channel"] for group in groups] != list(range(5))
                or [group["role"] for group in groups] != ["support"] * 4 + ["load"]
                or [group["master_node_id"] for group in groups[:4]] != sorted(group["master_node_id"] for group in groups[:4])
                or any(group["unreachable_nodes"] != 0 or group["finite_nodes"] != record["node_count"] or group["member_count"] < 1 for group in groups)):
            raise ValueError(f"{cad}: graph-cache node/order/source/normalization/finite identity changed")
        for identity in record["input_sources"].values():
            source = _path(identity["path"])
            source_stat = source.stat()
            if source_stat.st_size != identity["bytes"] or source_stat.st_mtime_ns != identity["mtime_ns"]:
                raise ValueError(f"{cad}: graph-cache source stat changed")
        if _path(record["input_sources"]["canonical_prepared"]["path"]) != files[f"{cad}/ver"]:
            raise ValueError("Distance cache uses different canonical input geometry")
        identities.append({"cad_id": cad, "path": str(path), "bytes": stat.st_size,
                           "mtime_ns": stat.st_mtime_ns, "metadata": record})
    contract["version"] = 11
    contract["requested"].update(distance_mode=args.distance_mode, graph_context=str(directory), bandwidth_mode=args.bandwidth_mode, bandwidth_plan=str(plan_path))
    contract["fixed_normalizers"] = {"config_path": str(normal_config_path), "config_sha256": _sha(normal_config_path.read_bytes()),
        "summary_path": str(normal_summary_path), "summary_sha256": _sha(normal_summary_path.read_bytes()),
        "scaler": scaler_record, "u_loss": loss_record,
        "policy": "Direct reuse of completed query-actual TRAIN120 records; no refit or recalibration"}
    plan_stat = plan_path.stat()
    contract["graph_context"] = {"distance_mode": args.distance_mode, "directory": str(directory),
        "manifest": str(manifest_path), "manifest_sha256": _sha(manifest_bytes),
        "metadata": {name: value for name, value in manifest.items() if name != "CADs"}, "selected_sources": identities,
        "plan": {"path": str(plan_path), "sha256": _sha(plan_bytes), "bytes": plan_stat.st_size,
                 "mtime_ns": plan_stat.st_mtime_ns, "metadata": plan},
        "field_ready": {"path": str(field_ready_path), "sha256": _sha(field_ready_path.read_bytes()), "source_sha256": field_sha},
        "data_ready": {"path": str(ready_path), "sha256": _sha(ready_path.read_bytes()), "source_sha256": data_ready["source_sha256"]},
        "outer_slice": [38, 43], "both_bandwidth_modes_same_active_Euclidean5": True, "target_independent": True,
        "normalization": "Cache already contains physical distance/L; no second scale, clipping or fitted transform",
        "native_backend": "Official split Encoding24 + fixed mask + Identity107 +13 explicit ones -> Network144/FullyFusedMLP128x2",
        "native_parameter_scope": "Two public decoder.encoding.params/network.params buffers; no flat offsets",
        "fresh_pair": True, "historical_control_reused": False,
        "capacity_limit": plan["pair"]["capacity_limit"], "reference_query_trainer_sha256": reference_hash}
    contract["query_context"]["zero_control"] = "None; both bandwidth representations retain the same active query32"
    contract["query_context"]["actual_control"] = "Both modes require finite positive projection/token gradients"
    contract["query_context"]["native_input_layout"] = "masked original38+same Euclidean5+query32+global32=107 Identity conditions; masked hash24 +13 explicit ones -> public Network144"
    contract["bc_roles"]["matched_encoder"] = "Both bandwidth modes retain the same actualBC-role5 and query/global layout"
    for name in ("graph_context_train.py", "graph_context_field.py", "hash_bandwidth_train.py", "hash_bandwidth_field.py"):
        contract["code_sha256"][name] = _sha((ROOT / "geobc" / name).read_bytes())
    contract["hash_bandwidth"] = {"bandwidth_mode": args.bandwidth_mode,
        "plan": {"path": str(plan_path), "sha256": _sha(plan_bytes), "bytes": plan_stat.st_size,
                 "mtime_ns": plan_stat.st_mtime_ns, "metadata": plan},
        "field_ready": {"path": str(field_ready_path), "sha256": _sha(field_ready_path.read_bytes()), "source_sha256": field_sha},
        "public_native_groups": ["encoding", "network"], "full_encoding_levels_both_modes": 12,
        "feature_channels_both_modes": 24, "first16_active_both": True,
        "last8_policy": "full12 active, coarse8 exact zero after fixed nonlearned mask",
        "named_trainable_parameters_match": True, "nonlearned_mask_buffer_equal_claim": False,
        "fresh_pair": True, "historical_control_reused": False,
        "reference_parent_module_sha256": frozen_graph_hash,
        "objective_sampler_evaluation_selection": "Frozen graph trainer helper math and independent RNG streams unchanged"}
    contract["code_sha256"]["tools/prepare_graph_context.py"] = data_ready["source_sha256"]
    return split, split_path, keys, files, contract


def _load_graph_arrays(directory, cad, xyz, record):
    """Read this CAD cache once, verify both active arrays, and retain original node order."""
    path = directory / record["file"]
    stat = path.stat()
    if stat.st_size != record["bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
        raise ValueError("Distance cache identity changed after metadata scope")
    query = xyz.detach().cpu().numpy() if isinstance(xyz, torch.Tensor) else np.asarray(xyz)
    if query.shape != (record["node_count"], 3) or _sha(np.ascontiguousarray(query, dtype="<f4").tobytes()) != record["xyz_sha256"]:
        raise ValueError(f"{cad}: distance cache query order/xyz differs")
    with np.load(path, allow_pickle=False) as stored:
        embedded = json.loads(str(stored["record_json"].item()))
        if (_graph_scientific_record(embedded) != _graph_scientific_record(record)
                or str(stored["cad_id"].item()) != cad or int(stored["node_count"].item()) != len(query)
                or str(stored["xyz_sha256"].item()) != record["xyz_sha256"]
                or str(stored["contract_sha256"].item()) != record["contract_sha256"]
                or not bool(stored["target_independent"].item())
                or float(stored["characteristic_length_mm"].item()) != record["normalization"]["characteristic_length_mm"]
                or list(stored["feature_names"]) != record["feature_names"]):
            raise ValueError(f"{cad}: embedded distance scientific metadata differs")
        arrays = {mode: stored[mode] for mode in ("euclidean", "graph")}
    for mode, array in arrays.items():
        if (array.dtype != np.float32 or array.shape != (len(query), 5) or not np.isfinite(array).all()
                or np.any(array < 0) or _sha(np.ascontiguousarray(array, dtype="<f4").tobytes()) != record[f"{mode}_features_sha256"]):
            raise ValueError(f"{cad}: {mode} must be the finite nonnegative original-node float32 distance5")
    return arrays


def _query_gradient(model, context, mode):
    """First update only; validate the intended query-branch gradient policy."""
    parameters = list(model.query_projection.parameters())
    if not parameters or any(parameter.grad is None for parameter in parameters):
        raise RuntimeError("Query projection parameter gradients must be present in both branches")
    if any(not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
        raise FloatingPointError("Query projection has nonfinite parameter gradients")
    projected_gradient = context.projected_tokens.grad
    if projected_gradient is None or not bool(torch.isfinite(projected_gradient).all()):
        raise FloatingPointError("Retained projected tokens require finite gradients in both branches")
    projection_l1 = sum(float(parameter.grad.detach().double().abs().sum()) for parameter in parameters)
    tokens_l1 = float(projected_gradient.detach().double().abs().sum())
    if not math.isfinite(projection_l1) or not math.isfinite(tokens_l1):
        raise FloatingPointError("Query gradient L1 reductions must be finite")
    if mode == "actual":
        if projection_l1 <= 0 or tokens_l1 <= 0:
            raise FloatingPointError("Actual query projection/tokens must receive nonzero first-update gradients")
    elif mode == "zero":
        if projection_l1 != 0 or tokens_l1 != 0:
            raise FloatingPointError("Zero query control must have exactly zero projection/token gradients")
    else:
        raise ValueError("Unknown query context mode")
    return {"query_context_mode": mode, "query_projection_gradient_l1": projection_l1,
            "projected_tokens_gradient_l1": tokens_l1,
            "query_projection_parameter_tensors": len(parameters),
            "query_projection_parameter_count": sum(parameter.numel() for parameter in parameters),
            "projected_tokens_shape": list(context.projected_tokens.shape),
            "projection_and_tokens_gradients_present_and_finite": True,
            "expected_gradient": "nonzero" if mode == "actual" else "exact_zero",
            "checked_after_backward_before_optimizer_step": True}


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


def _native_groups(model):
    """Two real public native buffers, disjoint and covering the decoder."""
    groups = model.decoder.native_parameter_groups()
    if set(groups) != {"encoding", "network"} or any(len(values) != 1 for values in groups.values()):
        raise ValueError("Bandwidth decoder requires one public Encoding and one public Network buffer")
    parameters = [groups[name][0] for name in ("encoding", "network")]
    if (len({id(value) for value in parameters}) != 2
            or {id(value) for value in parameters} != {id(value) for value in model.decoder.parameters()}
            or groups["encoding"][0] is not model.decoder.encoding.params
            or groups["network"][0] is not model.decoder.network.params
            or any(not value.requires_grad or value.dtype != torch.float32 for value in parameters)):
        raise ValueError("Public native buffers must cover all FP32 decoder trainables without offsets or extra heads")
    return groups


def _decoder_initial_record(model, mode):
    """Trainable parameters and mode-dependent nonlearned mask recorded separately."""
    groups = _native_groups(model)
    native = {name: [{"shape": list(p.shape), "dtype": str(p.dtype), "count": p.numel(),
                      "sha256": _sha(p.detach().cpu().contiguous().numpy().tobytes())} for p in groups[name]]
              for name in ("encoding", "network")}
    digest = hashlib.sha256()
    for name, parameter in model.decoder.named_parameters():
        value = parameter.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.tobytes())
    if (model.decoder.feature_mask.requires_grad
            or id(model.decoder.feature_mask) not in {id(value) for value in model.decoder.buffers()}):
        raise ValueError("Bandwidth mask must be a registered nonlearned buffer")
    mask = model.decoder.feature_mask.detach().cpu().contiguous()
    expected = torch.ones(24, dtype=mask.dtype)
    if mode == "coarse8":
        expected[16:] = 0
    if mask.shape != (24,) or mask.requires_grad or not torch.equal(mask, expected):
        raise ValueError("Fixed bandwidth mask must be24 channels with only the predeclared last8 policy")
    return {"native_groups": native, "all_decoder_parameter_sha256": digest.hexdigest(),
        "fixed_feature_mask": {"mode": mode, "buffer": "feature_mask", "learned": False,
            "values": mask.tolist(), "active_levels": 8 if mode == "coarse8" else 12,
            "shape": [24], "dtype": str(mask.dtype),
            "sha256": _sha(mask.numpy().tobytes()), "trainable": False,
            "excluded_from_trainable_parameter_equality": True,
            "full_state_equality_claimed": False}, "flat_buffer_offsets_inferred": False}


def _initial_parameters(model, decoder_record):
    """Hash all named trainables without consuming RNG or including mask buffers."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.tobytes())
    return {"all_parameter_sha256": digest.hexdigest(),
        "decoder_parameter_sha256": decoder_record["all_decoder_parameter_sha256"],
        "native_public_parameter_sha256": {name: values[0]["sha256"] for name, values in decoder_record["native_groups"].items()},
        "recorded_before_optimizer_or_resume": True, "paired_modes": ["full12", "coarse8"],
        "paired_architecture_and_seed_required": True, "hash_scope": "Named trainable parameters only",
        "nonlearned_mask_buffer_excluded": True, "full_state_equality_claimed": False}


def _bandwidth_gradient(model, mode):
    """First backward only: public buffer gradients and pre-mask feature gradient."""
    groups = _native_groups(model)
    native = {}
    for name, parameters in groups.items():
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise FloatingPointError(f"Missing/nonfinite public {name} gradients")
        l1 = sum(float(p.grad.detach().double().abs().sum()) for p in parameters)
        if not math.isfinite(l1) or l1 <= 0:
            raise FloatingPointError(f"Public {name} gradient must be finite and positive in both modes")
        native[name] = {"gradient_l1": l1, "parameters": sum(p.numel() for p in parameters),
                        "parameter_tensors": len(parameters), "gradients_present_and_finite": True}
    encoded, masked = model.decoder.encoded_activation, model.decoder.masked_activation
    if (encoded is None or masked is None or encoded.ndim != 2 or encoded.shape[1] != 24
            or masked.shape != encoded.shape or encoded.grad is None
            or not bool(torch.isfinite(encoded).all()) or not bool(torch.isfinite(masked).all())
            or not bool(torch.isfinite(encoded.grad).all())):
        raise FloatingPointError("Captured pre-mask Encoding24 and its gradients must be finite/present")
    first = float(encoded.grad[:, :16].double().abs().sum())
    fine = float(encoded.grad[:, 16:].double().abs().sum())
    masked_fine = float(masked[:, 16:].detach().double().abs().max())
    if first <= 0 or (mode == "coarse8" and (fine != 0 or masked_fine != 0)) or (mode == "full12" and fine <= 0):
        raise FloatingPointError("Bandwidth feature values/gradients do not follow the locked full12/coarse8 policy")
    return {"bandwidth_mode": mode, "native_parameter_groups": native,
        "native_gradient_scope": "Two entire public Encoding/Network buffers; no flat-offset partitions",
        "bandwidth_activation_gradient": {"pre_mask_first16_gradient_l1": first,
            "pre_mask_last8_gradient_l1": fine, "post_mask_last8_activation_max_abs": masked_fine,
            "pre_mask_last8_activation_max_abs": float(encoded[:, 16:].detach().double().abs().max()),
            "expected_last8_gradient": "exact_zero" if mode == "coarse8" else "positive",
            "pre_mask_gradient_not_post_mask_gradient": True,
            "checked_after_backward_before_optimizer_step": True}}


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
              "gradient_weight": args.gradient_weight, "distance_mode": args.distance_mode, "bandwidth_mode": args.bandwidth_mode,
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
        from .hash_bandwidth_field import HashBandwidthPhysicsField, HashBandwidthFieldConfig
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
        scaler = DisplacementScaler.from_record(contract["fixed_normalizers"]["scaler"])
        if scaler.fit_case_keys != tuple(keys["train"]) or scaler.fit_nodes != sum(case.count for case in train_cases):
            raise ValueError("Saved u scaler scope/counts differ from the explicit training cases")
        loss_record = training_loss_record(train_cases, scaler, args.u_loss,
                                          contract["fixed_normalizers"]["u_loss"])
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
        distance_records = {item["cad_id"]: item["metadata"] for item in contract["graph_context"]["selected_sources"]}
        for cad in cad_keys:
            record = distance_records[cad]
            arrays = _load_graph_arrays(_path(args.graph_context), cad, by_cad[cad][0].xyz, record)
            distance_tensor = torch.as_tensor(arrays[args.distance_mode], device=device)
            for case in by_cad[cad]:
                if case.features.shape != (record["node_count"], 38):
                    raise ValueError("Distance5 must follow unchanged original25+actual local13")
                if not math.isclose(float(case.features[0, 24]), record["normalization"]["characteristic_length_mm"] / 200,
                                    rel_tol=1e-6, abs_tol=1e-7):
                    raise ValueError("Distance normalization L differs from original characteristic-length input")
                case.features = torch.cat((case.features, distance_tensor), dim=-1)
            del arrays, distance_tensor
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
        with np.load(_path(contract["query_context"]["reference_role_cloud"]["path"]), allow_pickle=False) as role_reference:
            if (list(role_reference["cad_keys"]) != cad_keys
                    or not np.array_equal(role_reference["points"], cloud_array)
                    or not np.array_equal(role_reference["mask"], mask_array)
                    or not np.array_equal(role_reference["node_indices"], node_index_array)):
                raise ValueError("Query branches must exactly reuse the frozen actual-BC-role5 cloud")
        geometry_clouds, geometry_masks = torch.as_tensor(cloud_array, device=device), torch.as_tensor(mask_array, device=device)
        model_config = HashBandwidthFieldConfig(mode=args.physics_mode, bc_role_mode=args.bc_role_mode,
            distance_mode=args.distance_mode, hash_bandwidth_mode=args.bandwidth_mode,
            query_mode=args.query_context_mode, sigma=0.15, query_dim=32, query_chunk=4096,
            geometry_mode="shape" if args.variant == "geometry_pointnet" else "none",
            condition_dim=43, descriptor_dim=args.geometry_dim,
            hidden_dim=args.hidden_dim, cloud_points=args.cloud_points, seed=args.seed)
        model = HashBandwidthPhysicsField(model_config, device)
        native_layout = model.validate_native_layout()
        decoder_initial_record = _decoder_initial_record(model, args.bandwidth_mode)
        groups = _native_groups(model)
        actual_counts = {"encoding": sum(p.numel() for p in groups["encoding"]),
            "network": sum(p.numel() for p in groups["network"]),
            "native_total": sum(p.numel() for values in groups.values() for p in values),
            "PointNet": sum(parameter.numel() for parameter in model.encoder.parameters()),
            "query_projection": sum(parameter.numel() for parameter in model.query_projection.parameters()),
            "total": sum(parameter.numel() for parameter in model.parameters())}
        expected_counts = contract["hash_bandwidth"]["plan"]["metadata"]["architecture"]["source_derived_counts"]
        if (any(actual_counts[name] != expected_counts[name] for name in ("encoding", "network", "PointNet", "query_projection"))
                or actual_counts["native_total"] != expected_counts["total_native"]
                or actual_counts["total"] != expected_counts["total_model"]):
            raise ValueError("Actual two public native/encoder/projection counts differ from the fresh pair")
        initial_record = _initial_parameters(model, decoder_initial_record)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, fused=True)
        case_rng = np.random.default_rng(args.seed + 23)
        node_rng = torch.Generator(device=device).manual_seed(args.seed + 17)
        tet_rng = torch.Generator(device=device).manual_seed(args.seed + 41)
        config = {"contract": contract, "method": "GeoBC matched fixed full12 versus coarse8 Hash-feature bandwidth displacement/P1 candidate",
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
            "initial_decoder_parameter_sha256": decoder_initial_record["all_decoder_parameter_sha256"],
            "initial_decoder_parameters": decoder_initial_record,
            "initial_parameters": initial_record, "actual_parameter_counts": actual_counts, "native_layout": native_layout,
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
            "sampling": "four distinct cases; each6400 uniform original nodes +1600 uniform tetra corner sets x4; duplicates retained in BOTH fixed-bandwidth modes with the same active Euclidean/query/global BC inputs",
            "sampling_rng": {"case": "NumPy default_rng(seed+23)", "node": "torch.Generator(device),seed+17", "tet": "torch.Generator(device),seed+41"},
            "change_from_historical": "Both fresh split-bandwidth models retain the same actual Euclidean5/query/global context; same stored full12 Encoding and public Network144, only last8 nonlearned feature mask differs. Old fused distance or learned-gate models are contextual, not matched controls",
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
            "query_context": contract["query_context"],
            "hash_bandwidth": contract["hash_bandwidth"],
            "bandwidth_mask": decoder_initial_record["fixed_feature_mask"],
            "graph_context": {**contract["graph_context"], "CADs": distance_records,
                              "input_dtype": "float32", "fit_or_extra_transform": False},
            "u_loss": loss_record,
            "loss": "case-balanced relative-case MSE of3 signed u components; fixed train-only RMS weights, unchanged common zero-center decode"
                    if args.u_loss == "relative_case_mse" else "case-balanced MSE of3 signed u components with train-only common RMS and exactly zero center",
            "checkpoint_selection": "minimum validation15-case mean all-original-node physical displacement vector relativeL2",
            "stress_policy": "freeze independently supervised strict stress branch; do not combine stress or train it with u loss",
            "precision": "PointNet/query float32/high TF32; official split TCNN FP16 compute/FP32 master; physical u/strain geometry arithmetic and metrics float64",
            "total_steps": args.steps, "torch_version": torch.__version__, "started_at_utc": utc_now()}
        best_score, best_step, best_metrics, gradient_check = float("inf"), None, None, None
        if resume is not None:
            for name in ("contract", "scaler", "u_loss", "model", "geometry", "local_geometry", "cases", "precision", "gradient_supervision", "initial_parameters", "query_context", "graph_context", "native_layout", "actual_parameter_counts", "hash_bandwidth", "bandwidth_mask", "initial_decoder_parameters"):
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
            model.decoder.capture_activation_gradients = check_gradient
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
                gradient_check.update(_bandwidth_gradient(model, args.bandwidth_mode))
                model.decoder.capture_activation_gradients = False
                gradient_check.update(_query_gradient(model, context, args.query_context_mode))
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
                   "distance_mode": args.distance_mode, "bandwidth_mode": args.bandwidth_mode,
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
    result.add_argument("--bc-role-mode", choices=("actual",), default="actual")
    result.add_argument("--query-context-mode", choices=("actual",), default="actual")
    result.add_argument("--distance-mode", choices=("euclidean",), default="euclidean")
    result.add_argument("--graph-context", type=Path, default=Path("data/geobc_graph_context"))
    result.add_argument("--bandwidth-mode", choices=("full12", "coarse8"), required=True)
    result.add_argument("--bandwidth-plan", type=Path, default=Path("notes/geobc_hash_bandwidth_plan.json"))
    result.add_argument("--gradient-calibration", type=Path, default=Path("notes/geobc_gradient_calibration.json"))
    return result


if __name__ == "__main__":
    run(parser().parse_args())
