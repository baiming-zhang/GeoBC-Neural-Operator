"""One frozen TRAIN query-use/gradient-alignment diagnostic; no optimization.

Metadata preflight never opens arrays or weights. Runtime reads only the fixed
four training CADs' three known signed-u loads, using both last-10000 states.
P1 references are recovered from training u, not solver strain tensor labels.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import itertools
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

CADS = ("101_479", "101_428", "101_56", "101_78")
LOADS = ("ver", "hor", "tor")
KEYS = tuple(f"{cad}/{load}" for cad in CADS for load in LOADS)
SEED = 2026
STEP = 10000
LAMBDA = 0.0004106650618638651
UNIFORM = 6400
TETS = 1600
ROWS = 12800
BLOCKS = ("native", "encoder", "projection")
SOURCE_NAMES = ("geobc/query_context_train.py", "geobc/query_context_field.py",
    "geobc/gradient_train.py", "geobc/gradient_supervision.py", "geobc/physics_train.py",
    "geobc/physics_field.py", "geobc/conditioned_field.py", "geobc/tcnn_fields.py",
    "geobc/bc_role_field.py", "geobc/bc_role_condition.py", "geobc/local_geometry.py",
    "geobc/conditional_train.py", "geobc/displacement_condition.py")


def _path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and value.startswith("/mnt/") and len(value) > 7:
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and len(value) > 2 and value[1:3] == ":/":
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    result = Path(value)
    return (result if result.is_absolute() else ROOT / result).resolve()


def _portable(path):
    path = _path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _json(path):
    return json.loads(_path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(_path(path).read_bytes()).hexdigest()


def _identity(path, *, sha=False):
    path = _path(path)
    stat = path.stat()
    return {"path": _portable(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha(path) if sha else None}


def _match_stat(path, record):
    actual = _identity(path)
    if any(actual[name] != record[name] for name in ("bytes", "mtime_ns")):
        raise ValueError(f"Frozen size/mtime differs: {path}")
    if record.get("path") is not None and _path(path) != _path(record["path"]):
        raise ValueError(f"Frozen path differs: {path}")
    return actual


def _write(path, value):
    path = _path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _now():
    return datetime.now(timezone.utc).isoformat()


def request_contract(plan_path):
    """Pure JSON/small-source SHA/stat preflight, suitable for completed gating.

    No numpy/torch import, NPZ member, checkpoint content, validation field or
    stress target is accessed here. Big checkpoint/data identities use stat.
    """
    plan_path = _path(plan_path)
    plan = _json(plan_path)
    data = plan["data_contract"]
    if (plan["status"] != "selected_after_failed_query_gate"
            or tuple(data["CAD_order"]) != CADS or tuple(data["known_load_order"]) != LOADS
            or tuple(data["cases"]) != KEYS or data["gradient_weight"] != LAMBDA
            or plan["predeclared_stop"]["max_case_backwards"] != 24
            or plan["predeclared_stop"]["optimizer_updates"] != 0
            or not plan["sampling_predeclaration"]["shared_across_three_loads_and_both_models_and_ablation"]
            or plan["sampling_predeclaration"]["uniform_node_rng_entropy"] != [SEED, 1701, "CAD_rank"]
            or plan["sampling_predeclaration"]["uniform_tet_rng_entropy"] != [SEED, 1702, "CAD_rank"]):
        raise ValueError("Plan differs from the single fixed TRAIN diagnostic")
    sources = {name: _sha(ROOT / name) for name in SOURCE_NAMES}
    sources["tools/diagnose_query_learning.py"] = _sha(__file__)
    configs, branches = {}, {}
    for mode in ("zero", "actual"):
        frozen = plan["frozen_checkpoints"][mode]
        run = _path(frozen["run"])
        config_path, summary_path = run / "config.json", run / "summary.json"
        checkpoint = _path(frozen["checkpoint"])
        if (checkpoint != run / "last.pt" or frozen["required_checkpoint_step"] != STEP
                or _sha(config_path) != frozen["config_sha256"]
                or _sha(summary_path) != frozen["summary_sha256"]):
            raise ValueError(f"{mode}: predeclared last10k config/summary identity differs")
        config, summary = _json(config_path), _json(summary_path)
        if summary["status"] != "completed" or summary["step"] != STEP or summary["config"] != config:
            raise ValueError(f"{mode}: completed final10000 state is required")
        requested = config["contract"]["requested"]
        expected = {"steps": STEP, "seed": SEED, "physics_mode": "linear_basis",
                    "u_loss": "relative_case_mse", "local_mode": "actual", "hidden_dim": 128,
                    "geometry_dim": 32, "bc_role_mode": "actual", "query_context_mode": mode,
                    "gradient_weight": LAMBDA, "batch_size": 51200, "cases_per_batch": 4}
        if any(requested.get(name) != value for name, value in expected.items()):
            raise ValueError(f"{mode}: frozen training architecture/objective differs")
        for name, expected_sha in config["contract"]["code_sha256"].items():
            full = f"geobc/{name}"
            if full in sources and sources[full] != expected_sha:
                raise ValueError(f"Frozen inference/loss helper changed: {full}")
        branches[mode] = {"checkpoint": _identity(checkpoint), "required_step": STEP,
            "config": _identity(config_path, sha=True), "summary": _identity(summary_path, sha=True),
            "last_metrics": _identity(run / "metrics_latest.json", sha=True),
            "geometry_cloud_artifact": _identity(run / "geometry_clouds.npz"),
            "selection": "fixed last.pt at10000, not validation-selected best"}
        if _json(run / "metrics_latest.json")["step"] != STEP:
            raise ValueError("Final metrics must correspond to the fixed final step")
        configs[mode] = config
    config = configs["actual"]
    for name in ("scaler", "u_loss", "groups", "train_loads", "local_geometry", "geometry", "gradient_supervision"):
        if configs["zero"][name] != config[name]:
            raise ValueError(f"Matched frozen branches differ in {name}")
    split_path = _path(config["split_source"])
    split = _json(split_path)
    if (_sha(split_path) != config["contract"]["split_sha256"]
            or split["groups"] != config["groups"] or tuple(split["train_loads"]) != LOADS
            or len(split["groups"]["train"]) != 40 or len(split["groups"]["validation"]) != 5
            or not set(CADS) <= set(split["groups"]["train"])
            or set(CADS) & set(split["groups"]["validation"] + split["groups"]["test"])
            or tuple(["101_479"] + sorted(cad for cad in split["groups"]["train"] if cad != "101_479")[:3]) != CADS):
        raise ValueError("Fixed CADs must be the predeclared train40 subset")
    fit_keys = [f"{cad}/{load}" for cad in split["groups"]["train"] for load in LOADS]
    if (config["scaler"]["fit_case_keys"] != fit_keys or config["scaler"]["center_mm"] != 0
            or config["scaler"]["scale_mm"] != .05511113670847156
            or set(config["u_loss"]["case_records"]) != set(fit_keys)):
        raise ValueError("Saved train-only scaler/relative-case normalization differs")
    selected_records = {row["key"]: row for row in config["contract"]["selected_sources"]}
    selected_sources = {}
    for key in KEYS:
        path = _path(split["case_paths"][key])
        selected_sources[key] = _match_stat(path, selected_records[key])
        norm = config["u_loss"]["case_records"][key]
        if not math.isfinite(float(norm["weight"])) or norm["weight"] <= 0 or norm["floor_active"]:
            raise ValueError("Existing relative-u floor must be inactive on this probe")
    gradient_path = _path(config["arguments"]["gradient_cache"])
    manifest_path = gradient_path / "manifest.json"
    manifest = _json(manifest_path)
    if (manifest["status"] != "completed" or manifest["complete"] is not True
            or manifest["groups"] != split["groups"]
            or _sha(manifest_path) != config["gradient_supervision"]["manifest_sha256"]):
        raise ValueError("Existing completed gradient cache contract differs")
    geometry = {}
    for cad in CADS:
        gradient = manifest["CADs"][cad]
        cached = _match_stat(gradient_path / gradient["file"], gradient)
        for load in LOADS:
            if gradient["floor_active_by_load"][load] or gradient["loss_normalization_by_load"][load] <= 1e-24:
                raise ValueError("Existing train P1 loss denominator/floor differs")
        local = config["local_geometry"]["CADs"][cad]
        local_stat = _match_stat(local["cache"], local)
        geometry[cad] = {"gradient": cached, "gradient_metadata": gradient,
                         "local": local_stat, "local_metadata": local,
                         "cloud": config["geometry"]["CADs"][cad]}
    guard_path = _path(config["arguments"]["load_basis"])
    frozen_guard = config["contract"]["load_basis"]
    if configs["zero"]["contract"]["load_basis"] != frozen_guard:
        raise ValueError("Matched branches must share the frozen alpha/BC guard")
    _match_stat(guard_path, frozen_guard)
    if _sha(guard_path) != frozen_guard["sha256"]:
        raise ValueError("Frozen alpha/BC guard SHA differs from training contract")
    return {"version": 1, "plan": _identity(plan_path, sha=True), "sources_sha256": sources,
        "frozen_checkpoints": branches, "split": _identity(split_path, sha=True),
        "load_basis": _identity(guard_path, sha=True), "CADs": list(CADS), "cases": list(KEYS),
        "known_loads": list(LOADS), "selected_sources": selected_sources,
        "gradient_manifest": _identity(manifest_path, sha=True), "geometry": geometry,
        "scaler": config["scaler"], "u_loss": {"name": config["u_loss"]["name"],
            "case_records": {key: config["u_loss"]["case_records"][key] for key in KEYS}},
        "sampling": {"seed": SEED, "node_entropy": [SEED, 1701, "CAD_rank"],
            "tet_entropy": [SEED, 1702, "CAD_rank"], "uniform_nodes": UNIFORM,
            "uniform_tets": TETS, "corners_per_tet": 4, "rows": ROWS,
            "with_replacement": True, "shared_per_CAD_across_loads_models_intervention": True},
        "gradient_weight": LAMBDA, "checkpoint_step": STEP, "case_backwards": 24,
        "optimizer_updates": 0, "normalizers_refitted": False,
        "forbidden_fields": ["stress", "validation", "dia", "finaltest"]}


def sample_witness(node_count, corners, cad_rank):
    import numpy as np
    corners = np.asarray(corners, dtype=np.int64)
    node_entropy, tet_entropy = [SEED, 1701, cad_rank], [SEED, 1702, cad_rank]
    uniform = np.random.default_rng(np.random.SeedSequence(node_entropy)).integers(node_count, size=UNIFORM, dtype=np.int64)
    cells = np.random.default_rng(np.random.SeedSequence(tet_entropy)).integers(len(corners), size=TETS, dtype=np.int64)
    chosen_corners = corners[cells]
    rows = np.concatenate((uniform, chosen_corners.reshape(-1)))
    arrays = {"uniform_nodes": uniform, "sampled_cells": cells, "corner_nodes": chosen_corners, "query_nodes": rows}
    hashes = {name: hashlib.sha256(np.ascontiguousarray(value, dtype="<i8").tobytes()).hexdigest() for name, value in arrays.items()}
    return arrays, {"node_entropy": node_entropy, "tet_entropy": tet_entropy, "node_count": node_count,
                    "cell_count": len(corners), "rows": len(rows), "indices_sha256": hashes,
                    "hash_encoding": "C-contiguous little-endian int64 bytes, no npy header"}


def forward_with_query(model, xyz, features, alpha, context, query):
    """Use the unchanged decoder/global path, injecting explicit final query32.

    The actual intervention replaces only final bounded query slots by zero;
    it does not mutate model.config or any learned parameter/state.
    """
    import torch
    from geobc.conditioned_field import ConditionedField
    from geobc.physics_field import combine_basis
    conditioned = torch.cat((model.condition_features(features).float(), query.float()), dim=-1)
    flat = ConditionedField.forward(model.field, xyz.detach(), conditioned, context=context.global_tensor)
    return combine_basis(flat.reshape(len(xyz), 3, 3), alpha)


def parameter_blocks(model):
    named = list(model.named_parameters())
    owned = {"native": list(model.decoder.parameters()), "encoder": list(model.encoder.parameters()),
             "projection": list(model.query_projection.parameters())}
    ids = [id(parameter) for values in owned.values() for parameter in values]
    if len(ids) != len(set(ids)) or set(ids) != {id(parameter) for _, parameter in named}:
        raise ValueError("Parameter blocks must be disjoint and cover the full model")
    result = {}
    for block in BLOCKS:
        subset = {id(parameter) for parameter in owned[block]}
        result[block] = [(name, parameter) for name, parameter in named if id(parameter) in subset]
        if not result[block] or any(not parameter.requires_grad for _, parameter in result[block]):
            raise ValueError("Each prescribed block must contain differentiable parameters")
    return result


def collect_gradients(blocks, *, zero_projection=False):
    import torch
    result = {}
    for block, parameters in blocks.items():
        if any(parameter.grad is None for _, parameter in parameters):
            raise RuntimeError(f"Missing {block} parameter gradient")
        if any(not bool(torch.isfinite(parameter.grad).all()) for _, parameter in parameters):
            raise FloatingPointError(f"Nonfinite {block} parameter gradient")
        result[block] = torch.cat([parameter.grad.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
                                   for _, parameter in parameters])
    if zero_projection and bool(result["projection"].ne(0).any()):
        raise RuntimeError("Trained query-zero projection must have exactly zero gradient")
    return result


def gradient_alignment(cad_gradients):
    """Dot products in float64; full-model energies sum the disjoint blocks."""
    import torch
    cads = list(cad_gradients)
    if any(not bool(torch.isfinite(value).all()) for parts in cad_gradients.values() for value in parts.values()):
        raise FloatingPointError("Nonfinite averaged CAD gradient")
    mean = {block: torch.stack([cad_gradients[cad][block] for cad in cads]).mean(0) for block in BLOCKS}
    def dot(left, right, block):
        value = float(torch.dot(left[block], right[block])) if block != "full_model" else sum(float(torch.dot(left[b], right[b])) for b in BLOCKS)
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite gradient dot product")
        return value
    result = {}
    for block in ("full_model",) + BLOCKS:
        norms = {cad: math.sqrt(max(dot(cad_gradients[cad], cad_gradients[cad], block), 0)) for cad in cads}
        mean_norm = math.sqrt(max(dot(mean, mean, block), 0))
        pairs = []
        for a, b in itertools.combinations(cads, 2):
            value = dot(cad_gradients[a], cad_gradients[b], block)
            zero = norms[a] == 0 or norms[b] == 0
            pairs.append({"CADs": [a, b], "dot": value, "cosine": None if zero else max(-1., min(1., value / (norms[a] * norms[b]))), "zero_norm": zero})
        directions = {}
        for cad in cads:
            value = dot(cad_gradients[cad], mean, block)
            others = (len(cads) * value - norms[cad] ** 2) / (len(cads) - 1)
            directions[cad] = {"gradient_norm": norms[cad], "dot_with_mean4": value,
                "dot_with_mean_other3": others,
                "cosine_with_mean4": None if norms[cad] == 0 or mean_norm == 0 else max(-1., min(1., value / (norms[cad] * mean_norm))),
                "plain_gradient_mean4_predicts_local_loss_increase": value < 0}
        result[block] = {"CAD_gradient_norms": norms, "mean4_norm": mean_norm, "pairs": pairs, "joint_direction": directions}
    return result


def query_statistics(query):
    import torch
    value = query.detach().double()
    if value.ndim != 2 or value.shape[1] != 32 or not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Expected finite sampled query32")
    mean = value.mean(0)
    centered = value - mean
    cast = query.detach().half().float().double()
    norms = torch.linalg.vector_norm(value, dim=1)
    rms = float(value.square().mean().sqrt())
    centered_rms = float(centered.square().mean().sqrt())
    return {"queries": len(value), "mean_vector": mean.cpu().tolist(), "total_RMS": rms,
        "query_centered_RMS": centered_rms, "channel_population_std": centered.square().mean(0).sqrt().cpu().tolist(),
        "norm_min": float(norms.min()), "norm_median": float(norms.median()), "norm_max": float(norms.max()),
        "tanh_abs_ge_095_fraction": float((value.abs() >= .95).double().mean()),
        "tanh_abs_ge_099_fraction": float((value.abs() >= .99).double().mean()),
        "FP16_cast_total_RMS": float(cast.square().mean().sqrt()),
        "FP16_cast_query_centered_RMS": float((cast - cast.mean(0)).square().mean().sqrt()),
        "FP16_cast_error_RMS": float((cast - value).square().mean().sqrt()),
        "FP16_cast_centered_variation_ratio": float((cast - cast.mean(0)).square().mean().sqrt()) / centered_rms if centered_rms else None,
        "structural_exact_zero": bool(value.eq(0).all()),
        "FP16_limit": "cast-only query activation diagnostic; not a native-kernel precision audit"}


def _context_distances(contexts):
    import numpy as np
    result = []
    for a, b in itertools.combinations(CADS, 2):
        result.append({"CADs": [a, b], "query_mean_vector_RMS_distance": float(np.sqrt(np.mean(
            (np.asarray(contexts[a]["query"]["mean_vector"]) - np.asarray(contexts[b]["query"]["mean_vector"])) ** 2))),
            "global32_RMS_distance": float(np.sqrt(np.mean((np.asarray(contexts[a]["global32"]) - np.asarray(contexts[b]["global32"])) ** 2)))})
    return result


def brief_readout(results):
    """Compact aggregation of existing readouts; no additional model calls."""
    result = {"scope": "fixed query-sampled TRAIN4x3 pilot; P1 loss is squared normalized cell-error, not all-node tensor L2"}
    for mode in ("zero", "actual"):
        full = results[mode]["CAD_joint_gradients"]["full_model"]
        contexts = results[mode]["context_statistics"]
        result[mode] = {"full_model_negative_CAD_pairs": sum(pair["cosine"] is not None and pair["cosine"] < 0 for pair in full["pairs"]),
            "full_model_CAD_pairs_total": len(full["pairs"]),
            "CADs_with_negative_dot_gradient_mean4": [cad for cad, value in full["joint_direction"].items() if value["dot_with_mean4"] < 0],
            "query_centered_RMS_by_CAD": {cad: contexts[cad]["query"]["query_centered_RMS"] for cad in CADS},
            "query_centered_RMS_mean4": sum(contexts[cad]["query"]["query_centered_RMS"] for cad in CADS) / len(CADS)}
    interventions = list(results["actual"]["actual_final_query_zero_intervention"].values())
    keys = ("relative_Lu", "normalized_volume_P1_Frobenius_loss", "joint")
    result["actual_same_weights_final_query_zero"] = {
        "sampled_cases": len(interventions),
        "mean_delta_u_over_true_u_norm": sum(row["delta_u_over_true_u_norm"] for row in interventions) / len(interventions),
        "mean_intact_losses": {key: sum(row["intact"][key] for row in interventions) / len(interventions) for key in keys},
        "mean_masked_losses": {key: sum(row["final_query32_masked"][key] for row in interventions) / len(interventions) for key in keys},
        "mean_delta_masked_minus_intact": {key: sum(row["delta_masked_minus_intact"][key] for row in interventions) / len(interventions) for key in keys},
        "cases_masking_increases_loss": {key: sum(row["delta_masked_minus_intact"][key] > 0 for row in interventions) for key in keys}}
    return result


def _loss_values(prediction, case, target, cells, gradient_data, scaler, fixed_weight):
    import torch
    from geobc.query_context_train import relative_case_mse, gradient_batch_loss
    u = relative_case_mse(prediction, target, torch.tensor([fixed_weight], device=prediction.device, dtype=torch.float32))
    p1 = gradient_batch_loss(prediction, [case], [cells], gradient_data, scaler.scale_mm, LAMBDA)
    return u, p1, u + LAMBDA * p1


def diagnose(args):
    contract = request_contract(args.plan)
    output = _path(args.output)
    if args.dry_run:
        return {"status": "metadata_preflight_passed", "contract": contract, "arrays_or_weights_opened": False, "GPU_used": False}
    if output.exists() and any(output.iterdir()):
        summary_path = output / "summary.json"
        if summary_path.exists():
            summary = _json(summary_path)
            if summary.get("status") == "completed" and summary["config"]["contract"] == contract:
                return {"status": "completed_reused", "summary": _portable(summary_path)}
        raise ValueError("Nonempty diagnostic output requires an exact completed contract; incomplete output is not silently overwritten")
    import numpy as np
    import torch
    from geobc.physics_train import DisplacementScaler, _load_basis, load_displacement_cases
    from geobc.gradient_supervision import load_gradient_manifest, load_gradient_cache
    from geobc.local_geometry import load_local_geometry
    from geobc.bc_role_condition import load_bc_role_cloud
    from geobc.query_context_field import QueryContextPhysicsField, QueryContextFieldConfig
    if not torch.cuda.is_available():
        raise RuntimeError("Native frozen-checkpoint diagnostic requires scheduled CUDA; use --dry-run for CPU metadata preflight")
    device = torch.device("cuda")
    began = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    config_record = {"contract": contract, "started_at_utc": _now(), "selection": "fixed last10000 for both models; no best/reselection",
        "precision": "frozen high TF32 FP32 encoder/pooling; TCNN native FP16 with FP32 master; physical P1 FP64; detached gradient dot products CPU FP64",
        "scope": "one selected TRAIN4-CAD x3knownload witness pilot; no validation/dia/finaltest/stress arrays",
        "reference": "reference corner P1 from training u, not solver GT strain tensor",
        "limits": ["Plain parameter-gradient cosine/dot is not the Adam update direction or a causal diagnosis.",
                   "Actual final-query32 zero intervention can be off-distribution; sensitivity alone does not show usefulness.",
                   "One sampled training witness with duplicate rows cannot establish all-node performance or generalization.",
                   "Differences between the two trained checkpoint histories remain observational."]}
    _write(output / "config.json", config_record)
    def status(phase, **kwargs):
        _write(output / "status.json", {"status": phase, "updated_at_utc": _now(), "elapsed_seconds": time.monotonic() - began, **kwargs})
    model = None
    try:
        status("loading_fixed_train_subset")
        torch.manual_seed(SEED)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
        configs = {mode: _json(contract["frozen_checkpoints"][mode]["config"]["path"]) for mode in ("zero", "actual")}
        reference = configs["actual"]
        split = _json(contract["split"]["path"])
        files = {key: _path(contract["selected_sources"][key]["path"]) for key in KEYS}
        guard = _load_basis(_path(contract["load_basis"]["path"]), split)
        # Explicit subset is the only keys argument. This frozen loader never accesses stress.
        cases = load_displacement_cases(list(KEYS), files, guard, "cpu")
        by_key = {case.key: case for case in cases}
        scaler = DisplacementScaler.from_record(reference["scaler"])
        gradient_directory = _path(reference["arguments"]["gradient_cache"])
        manifest = load_gradient_manifest(gradient_directory)
        data, clouds, witnesses = {}, {}, {}
        for rank, cad in enumerate(CADS):
            group = [by_key[f"{cad}/{load}"] for load in LOADS]
            grids = None
            for case in group:
                with np.load(case.source, allow_pickle=False) as stored:
                    if str(stored["u_units"].item()) != "mm":
                        raise ValueError("Signed original displacement units must be mm")
                    current = np.asarray(stored["grid_node_ids"], dtype=np.int64)
                if len(current) != case.count or len(np.unique(current)) != case.count:
                    raise ValueError("Original GRID order must have one unique ID per node")
                if grids is not None and not np.array_equal(grids, current):
                    raise ValueError("Known-load original GRID order differs")
                grids = current
            cache = load_gradient_cache(gradient_directory, cad, xyz=group[0].xyz, manifest=manifest)
            local = load_local_geometry(_path(reference["arguments"]["local_geometry"]), cad)
            local_tensor = local.tensor(mode="actual", device="cpu")
            for case in group:
                cache.verify_xyz(case.xyz)
                local.verify_xyz(case.xyz)
                case.features = torch.cat((case.features, local_tensor), dim=-1)
            cloud = load_bc_role_cloud(files[f"{cad}/ver"], cad_key=cad,
                seed=reference["geometry"]["CADs"][cad]["seed"], cloud_points=1024)
            expected_cloud = reference["geometry"]["CADs"][cad]
            actual_cloud = cloud.record()
            for name in ("points5_sha256", "coordinates_sha256", "roles_sha256", "mask_sha256", "node_indices_sha256", "source_xyz_sha256"):
                if actual_cloud[name] != expected_cloud[name]:
                    raise ValueError(f"{cad}: fixed saved cloud {name} differs")
            clouds[cad] = (torch.as_tensor(cloud.points[None], device=device), torch.as_tensor(cloud.mask[None], device=device))
            arrays, witness = sample_witness(group[0].count, cache.corners, rank)
            witness.update(file=f"witness_{cad}.npz", normalized_xyz_sha256=cache.metadata["xyz_sha256"],
                           original_GRID_ids_sha256=hashlib.sha256(np.ascontiguousarray(grids, dtype="<i8").tobytes()).hexdigest(),
                           cloud_hash_verified_against_both_configs=True)
            np.savez(output / witness["file"], **arrays)
            witnesses[cad] = {"arrays": arrays, "record": witness}
            # Geometry/reference arrays for only the four allowed CADs move to
            # CUDA; original-node u and local features remain on CPU.
            cells = arrays["sampled_cells"]
            data[cad] = {"cache": cache, "tensors": {name: torch.as_tensor(value, device=device) for name, value in
                (("inverse_physical_edge_mm", cache.inverse_physical_edge_mm), ("volumes", cache.volumes),
                 ("reference_cell_tensor_strain", cache.reference_cell_tensor_strain))}}
        _write(output / "witnesses.json", {cad: witnesses[cad]["record"] for cad in CADS})
        results, backwards = {}, 0
        for mode in ("zero", "actual"):
            status("loading_checkpoint", model=mode, case_backwards=backwards)
            cfg = configs[mode]
            checkpoint_path = _path(contract["frozen_checkpoints"][mode]["checkpoint"]["path"])
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            latest = _json(contract["frozen_checkpoints"][mode]["last_metrics"]["path"])
            if (saved["step"] != STEP or saved["config"] != cfg or saved["scaler"] != cfg["scaler"]
                    or saved["metrics"] != latest or saved["metrics"] != _json(contract["frozen_checkpoints"][mode]["summary"]["path"])["final_metrics"]):
                raise ValueError("Loaded last.pt content differs from frozen final10k metadata")
            model_cfg = QueryContextFieldConfig(**{item.name: cfg["model"][item.name] for item in fields(QueryContextFieldConfig) if item.name in cfg["model"]})
            model = QueryContextPhysicsField(model_cfg, device=device)
            if model.source_config != cfg["model"]:
                raise ValueError("Reconstructed frozen model architecture/source configuration differs")
            model.load_state_dict(saved["model"], strict=True)
            del saved  # optimizer history is never applied, restored or used
            model.eval()
            blocks = parameter_blocks(model)
            block_record = {block: [{"name": name, "shape": list(parameter.shape), "count": parameter.numel()} for name, parameter in values] for block, values in blocks.items()}
            gradients, losses, contexts, interventions = {}, {}, {}, {}
            for cad in CADS:
                nodes = witnesses[cad]["arrays"]["query_nodes"]
                cell_ids = torch.as_tensor(witnesses[cad]["arrays"]["sampled_cells"], device=device)
                gradients[cad] = None
                losses[cad] = {}
                for load in LOADS:
                    case = by_key[f"{cad}/{load}"]
                    xyz = case.xyz[nodes].to(device)
                    features = case.features[nodes].to(device)
                    target = torch.as_tensor(scaler.encode(case.physical_u[nodes]), device=device)
                    alpha = case.alpha.to(device).expand(ROWS, -1)
                    context = model.encode_geometry(*clouds[cad])
                    query = model.field.query_context(xyz, context)
                    if load == LOADS[0]:
                        contexts[cad] = {"query": query_statistics(query), "global32": context.global_tensor.detach().double().cpu()[0].tolist()}
                    model.zero_grad(set_to_none=True)
                    prediction = forward_with_query(model, xyz, features, alpha, context, query)
                    fixed_weight = cfg["u_loss"]["case_records"][case.key]["weight"]
                    u, p1, joint = _loss_values(prediction, case, target, cell_ids, data, scaler, fixed_weight)
                    if not all(bool(torch.isfinite(value)) for value in (u, p1, joint)):
                        raise FloatingPointError("Nonfinite joint diagnostic loss")
                    losses[cad][load] = {"relative_Lu": float(u.detach()), "normalized_volume_P1_Frobenius_loss": float(p1.detach()), "joint": float(joint.detach())}
                    if mode == "actual":
                        with torch.no_grad():
                            masked = forward_with_query(model, xyz, features, alpha, context, query * 0)
                            mu, mp1, mjoint = _loss_values(masked, case, target, cell_ids, data, scaler, fixed_weight)
                            if not all(bool(torch.isfinite(value).all()) for value in (masked, mu, mp1, mjoint)):
                                raise FloatingPointError("Nonfinite actual-query mask intervention")
                            true = case.physical_u[nodes]
                            delta = (masked.double() - prediction.detach().double()).cpu().numpy() * scaler.scale_mm
                            denom = float(np.linalg.norm(true))
                            if not math.isfinite(denom) or denom <= 0:
                                raise ValueError("Sampled signed-u intervention denominator must be positive")
                            interventions[case.key] = {"query_rows": ROWS, "true_component_RMS_mm": float(np.sqrt(np.mean(true ** 2))),
                                "delta_u_component_RMS_mm": float(np.sqrt(np.mean(delta ** 2))), "delta_u_over_true_u_norm": float(np.linalg.norm(delta) / denom),
                                "intact": losses[cad][load], "final_query32_masked": {"relative_Lu": float(mu), "normalized_volume_P1_Frobenius_loss": float(mp1), "joint": float(mjoint)},
                                "delta_masked_minus_intact": {"relative_Lu": float(mu) - float(u.detach()), "normalized_volume_P1_Frobenius_loss": float(mp1) - float(p1.detach()), "joint": float(mjoint) - float(joint.detach())}}
                    joint.backward()
                    backwards += 1
                    current = collect_gradients(blocks, zero_projection=mode == "zero")
                    if gradients[cad] is None:
                        gradients[cad] = {block: value / len(LOADS) for block, value in current.items()}
                    else:
                        for block in BLOCKS:
                            gradients[cad][block].add_(current[block], alpha=1 / len(LOADS))
                    del current, context, query, prediction, u, p1, joint
                    status("case_backward_completed", model=mode, case=case.key, case_backwards=backwards, total_case_backwards=24)
            results[mode] = {"fixed_step": STEP, "parameter_blocks": block_record,
                "parameter_gradient_finite_and_present": True, "case_losses": losses,
                "CAD_mean_losses": {cad: {name: float(np.mean([losses[cad][load][name] for load in LOADS])) for name in losses[cad][LOADS[0]]} for cad in CADS},
                "CAD_joint_gradients": gradient_alignment(gradients), "context_statistics": contexts,
                "context_CAD_pair_distances": _context_distances(contexts), "actual_final_query_zero_intervention": interventions,
                "projection_zero_gradient_required": mode == "zero"}
            del gradients, blocks
            model.zero_grad(set_to_none=True)
            del model
            model = None
            torch.cuda.empty_cache()
        if backwards != 24 or request_contract(args.plan) != contract:
            raise RuntimeError("Diagnostic cap or frozen input identity changed during execution")
        memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        # Release all CUDA tensors before publishing completed state to the queue.
        del data, clouds, xyz, features, target, alpha, cell_ids
        if "masked" in locals():
            del masked, mu, mp1, mjoint
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        summary = {"status": "completed", "config": config_record, "fixed_step": STEP,
            "case_backwards": backwards, "optimizer_updates": 0, "witnesses": {cad: witnesses[cad]["record"] for cad in CADS},
            "models": results, "readout": brief_readout(results), "elapsed_seconds": time.monotonic() - began, "memory": memory,
            "completed_at_utc": _now(), "GPU_tensors_released_before_completion": True}
        _write(output / "summary.json", summary)
        status("completed", case_backwards=backwards, summary="summary.json")
        return {"status": "completed", "summary": _portable(output / "summary.json"), "elapsed_seconds": summary["elapsed_seconds"]}
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if model is not None:
            model.zero_grad(set_to_none=True)
            del model
        torch.cuda.empty_cache()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", default="notes/geobc_query_learning_plan.json")
    result.add_argument("--output", default="runs/geobc_exploration/query_learning_frozen_10k")
    result.add_argument("--dry-run", action="store_true", help="JSON/SHA/stat only; no arrays, weights, torch import or CUDA")
    return result


if __name__ == "__main__":
    result = diagnose(parser().parse_args())
    if result.get("status") == "metadata_preflight_passed":
        result = {"status": result["status"], "plan": result["contract"]["plan"],
            "cases": result["contract"]["cases"], "case_backwards": 24, "optimizer_updates": 0,
            "arrays_or_weights_opened": False, "GPU_used": False}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
