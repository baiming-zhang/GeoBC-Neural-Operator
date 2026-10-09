"""Frozen TRAIN-only hash-gate use/saturation probe, with no optimization.

request_contract and --dry-run use JSON/source SHA/stat only. Runtime is
conditional on root selecting the failed-pair diagnostic and uses 16 basis
forwards, no backwards, and no updates. Only four TRAIN CADs' ver/hor/tor u
references are loaded. Reported losses are sampled squared objectives.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.diagnose_query_learning import (
    CADS, LOADS, KEYS, SEED, STEP, LAMBDA, UNIFORM, TETS, ROWS, SOURCE_NAMES,
    _path, _portable, _json, _sha, _identity, _match_stat, _write, _now,
    sample_witness, _loss_values,
)

SELECTED_STATUS = "selected_after_failed_hash_modulation_gate"
PAIR_RESULT = ROOT / "notes/geobc_hash_modulation_pair_result.json"
HASH_PLAN = ROOT / "notes/geobc_hash_modulation_plan.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def request_contract(plan_path):
    """Reject conditional/uncompleted inputs before any NumPy/Torch import.

Only small JSON, small source files, and stat of named checkpoint/cache/data
files are read here. Completed-pair analyzer hashes lock the actual run JSON.
"""
    plan_path = _path(plan_path)
    plan = _json(plan_path)
    require(plan.get("status") == SELECTED_STATUS,
            f"Hash-gate use probe is not selected: expected {SELECTED_STATUS}; no arrays/weights opened")
    require(tuple(plan["scope"]["CAD_order"]) == CADS and tuple(plan["scope"]["known_load_order"]) == LOADS
            and plan["scope"]["training_labels_only"] is True
            and not any(plan["scope"][k] for k in ("validation_arrays", "dia_arrays", "final_test_arrays", "stress_arrays"))
            and plan["execution"]["optimizer_updates"] == 0 and plan["execution"]["backwards"] == 0
            and plan["execution"]["max_basis_forward_calls"] == 16,
            "Selected design must remain one TRAIN4 x known3 gate-use probe with 16 forwards/0 backwards/updates")
    sampling = plan["sampling"]
    require(sampling["seed"] == SEED and sampling["rows_per_CAD"] == ROWS
            and sampling["uniform_original_nodes"] == UNIFORM and sampling["uniform_tetra_corner_sets"] == TETS
            and sampling["redraw_for_counterfactual"] is False
            and plan["checkpoints"]["modes"] == ["zero", "actual"], "Frozen sampler/checkpoint modes changed")
    sampler_plan_path = _path(sampling["same_as"])
    sampler_plan = _json(sampler_plan_path)
    sp = sampler_plan["sampling_predeclaration"]
    require(sp["uniform_node_rng_entropy"] == [SEED, 1701, "CAD_rank"]
            and sp["uniform_tet_rng_entropy"] == [SEED, 1702, "CAD_rank"]
            and sp["shared_across_three_loads_and_both_models_and_ablation"], "Prior frozen witness scheme changed")
    pair = _json(PAIR_RESULT)
    selection = plan["parent_selection"]
    require(_path(selection["pair_result"]) == PAIR_RESULT and selection["pair_result_sha256"] == _sha(PAIR_RESULT)
            and selection["matched_controls_passed"] is True and selection["clear_gain_gate_passed"] is False,
            "Root's frozen pair selection identity differs")
    require(pair.get("status") == "completed" and pair["matched_controls_passed"] is True
            and pair["clear_gain_gate_passed"] is False and pair["fresh_split_pair"] is True
            and pair["historical_control_reused"] is False and pair["plan"]["sha256"] == _sha(HASH_PLAN),
            "Requires a completed matched fresh pair that failed its frozen gate; no probe is scheduled here")
    source_names = tuple(SOURCE_NAMES) + ("geobc/hash_modulation_field.py", "geobc/hash_modulation_train.py",
        "tools/diagnose_query_learning.py", "tools/analyze_hash_modulation_pair.py")
    sources = {name: _sha(ROOT / name) for name in source_names}
    sources["tools/diagnose_hash_modulation_use.py"] = _sha(__file__)
    configs, branches = {}, {}
    for mode in ("zero", "actual"):
        run = _path(plan["checkpoints"]["directories"][mode])
        require(run == ROOT / f"runs/geobc_exploration/displacement_hash_modulation_{mode}_10k",
                "Only the current fresh split-native pair is allowed")
        for record in pair["source_artifacts"][mode].values():
            require(_sha(record["path"]) == record["sha256"], f"{mode}: analyzed small JSON artifact changed")
        config, summary, latest = (_json(run / name) for name in ("config.json", "summary.json", "metrics_latest.json"))
        require(summary.get("status") == "completed" and summary["step"] == STEP
                and summary["config"] == config and latest["step"] == STEP and summary["final_metrics"] == latest,
                f"{mode}: completed final10000 state is required")
        expected = {"steps": STEP, "seed": SEED, "physics_mode": "linear_basis", "u_loss": "relative_case_mse",
            "local_mode": "actual", "hidden_dim": 128, "geometry_dim": 32, "bc_role_mode": "actual",
            "query_context_mode": "actual", "hash_modulation_mode": mode, "gradient_weight": LAMBDA,
            "batch_size": 51200, "cases_per_batch": 4}
        require(all(config["arguments"].get(k) == v for k, v in expected.items()), f"{mode}: frozen recipe differs")
        require(config["model"]["actual_decoder_backend"] == "tcnn_split_encoding_network_hash_gate",
                "An old Composite checkpoint cannot replace the new split decoder")
        for name, digest in config["contract"]["code_sha256"].items():
            full = f"geobc/{name}"
            if full in sources:
                require(sources[full] == digest, f"Frozen inference/loss source changed: {full}")
        branches[mode] = {"checkpoint": _identity(run / "last.pt"), "required_step": STEP,
            "config": _identity(run / "config.json", sha=True), "summary": _identity(run / "summary.json", sha=True),
            "last_metrics": _identity(run / "metrics_latest.json", sha=True),
            "selection": "fixed last.pt at10000, not validation-selected best"}
        configs[mode] = config
    config = configs["actual"]
    for name in ("scaler", "u_loss", "groups", "train_loads", "local_geometry", "geometry", "gradient_supervision"):
        require(configs["zero"][name] == config[name], f"Matched frozen branches differ in {name}")
    split_path = _path(config["split_source"])
    split = _json(split_path)
    require(_sha(split_path) == config["contract"]["split_sha256"] and split["groups"] == config["groups"]
            and tuple(split["train_loads"]) == LOADS and len(split["groups"]["train"]) == 40
            and len(split["groups"]["validation"]) == 5 and set(CADS) <= set(split["groups"]["train"])
            and not (set(CADS) & set(split["groups"]["validation"] + split["groups"]["test"]))
            and tuple(["101_479"] + sorted(c for c in split["groups"]["train"] if c != "101_479")[:3]) == CADS,
            "Fixed predeclared TRAIN CAD subset/split changed")
    fit_keys = [f"{cad}/{load}" for cad in split["groups"]["train"] for load in LOADS]
    require(config["scaler"]["fit_case_keys"] == fit_keys and config["scaler"]["center_mm"] == 0
            and config["scaler"]["scale_mm"] == .05511113670847156
            and set(config["u_loss"]["case_records"]) == set(fit_keys), "Existing TRAIN-only normalizers changed")
    selected_records = {r["key"]: r for r in config["contract"]["selected_sources"]}
    selected_sources = {key: _match_stat(split["case_paths"][key], selected_records[key]) for key in KEYS}
    gradient_path = _path(config["arguments"]["gradient_cache"])
    manifest_path = gradient_path / "manifest.json"
    manifest = _json(manifest_path)
    require(manifest["status"] == "completed" and manifest["complete"] is True
            and manifest["groups"] == split["groups"]
            and _sha(manifest_path) == config["gradient_supervision"]["manifest_sha256"], "Frozen P1 cache differs")
    geometry = {}
    for cad in CADS:
        record = manifest["CADs"][cad]
        for load in LOADS:
            norm = config["u_loss"]["case_records"][f"{cad}/{load}"]
            require(not norm["floor_active"] and math.isfinite(norm["weight"]) and norm["weight"] > 0
                    and not record["floor_active_by_load"][load] and record["loss_normalization_by_load"][load] > 1e-24,
                    "Existing u/P1 floor must be inactive on this TRAIN probe")
        local = config["local_geometry"]["CADs"][cad]
        geometry[cad] = {"gradient": _match_stat(gradient_path / record["file"], record),
            "gradient_metadata": record, "local": _match_stat(local["cache"], local),
            "local_metadata": local, "cloud": config["geometry"]["CADs"][cad]}
    guard_path = _path(config["arguments"]["load_basis"])
    guard = config["contract"]["load_basis"]
    require(configs["zero"]["contract"]["load_basis"] == guard and _sha(guard_path) == guard["sha256"],
            "Frozen load-basis guard differs")
    _match_stat(guard_path, guard)
    return {"version": 1, "plan": _identity(plan_path, sha=True), "sources_sha256": sources,
        "matched_pair_result": _identity(PAIR_RESULT, sha=True), "hash_plan": _identity(HASH_PLAN, sha=True),
        "prior_sampling_plan": _identity(sampler_plan_path, sha=True), "frozen_checkpoints": branches,
        "split": _identity(split_path, sha=True), "load_basis": _identity(guard_path, sha=True),
        "CADs": list(CADS), "cases": list(KEYS), "known_loads": list(LOADS), "selected_sources": selected_sources,
        "gradient_manifest": _identity(manifest_path, sha=True), "geometry": geometry,
        "scaler": config["scaler"], "u_loss": {"name": config["u_loss"]["name"],
            "case_records": {key: config["u_loss"]["case_records"][key] for key in KEYS}},
        "sampling": {"seed": SEED, "node_entropy": [SEED, 1701, "CAD_rank"], "tet_entropy": [SEED, 1702, "CAD_rank"],
            "uniform_nodes": UNIFORM, "uniform_tets": TETS, "corners_per_tet": 4, "rows": ROWS,
            "with_replacement": True, "shared_per_CAD_across_loads_models_intervention": True},
        "gradient_weight": LAMBDA, "checkpoint_step": STEP, "max_basis_forward_calls": 16,
        "backwards": 0, "optimizer_updates": 0, "normalizers_refitted": False,
        "forbidden_fields": ["stress", "validation", "dia", "finaltest"]}


def gamma_statistics(gamma):
    import torch
    value = gamma.detach().double()
    require(value.ndim == 2 and value.shape[1] == 24 and len(value) == ROWS, "Expected sampled gamma[12800,24]")
    finite = torch.isfinite(value)
    require(bool(finite.all()), "Nonfinite gate multiplier")
    delta = value - 1
    mean = delta.mean(0)
    centered = delta - mean
    return {"queries": len(value), "channels": 24, "finite_fraction": float(finite.double().mean()),
        "delta_mean": float(delta.mean()), "delta_RMS": float(delta.square().mean().sqrt()),
        "spatial_centered_RMS": float(centered.square().mean().sqrt()),
        "gamma_channel_min": value.amin(0).cpu().tolist(), "gamma_channel_max": value.amax(0).cpu().tolist(),
        "gamma_channel_mean": value.mean(0).cpu().tolist(),
        "delta_channel_population_std": centered.square().mean(0).sqrt().cpu().tolist(),
        "gamma_min": float(value.min()), "gamma_max": float(value.max()),
        "near_lower_endpoint_fraction": float((value <= .05).double().mean()),
        "near_upper_endpoint_fraction": float((value >= 1.95).double().mean()),
        "near_either_endpoint_fraction": float(((value <= .05) | (value >= 1.95)).double().mean()),
        "exact_lower_endpoint_fraction": float(value.eq(0).double().mean()),
        "exact_upper_endpoint_fraction": float(value.eq(2).double().mean()),
        "exact_either_endpoint_fraction": float((value.eq(0) | value.eq(2)).double().mean()),
        "scope": "same mixed TRAIN query rows; duplicate queries retained, not volume/area-weighted or whole-node distribution"}


def forward_basis(model, xyz, features, context, query):
    """One shared 9-response forward, with only decoder.mode mutable outside."""
    import torch
    from geobc.conditioned_field import ConditionedField
    conditioned = torch.cat((model.condition_features(features).float(), query.float()), dim=-1)
    flat = ConditionedField.forward(model.field, xyz.detach(), conditioned, context=context.global_tensor)
    require(flat.shape == (ROWS, 9), "Expected one shared nine-basis output")
    return flat.reshape(ROWS, 3, 3)


def loss_record(prediction, case, target, cells, data, scaler, weight):
    import torch
    u, p1, joint = _loss_values(prediction, case, target, cells, data, scaler, weight)
    require(all(bool(torch.isfinite(x)) for x in (u, p1, joint)), "Nonfinite sampled squared loss")
    return {"relative_Lu": float(u), "normalized_volume_P1_Frobenius_loss": float(p1), "joint": float(joint)}


def brief_readout(results):
    names = ("relative_Lu", "normalized_volume_P1_Frobenius_loss", "joint")
    result = {"scope": "fixed TRAIN4x3, sampled squared losses, not all-node physical U/P1 relativeL2 percentages"}
    for mode in ("zero", "actual"):
        cases = list(results[mode]["cases"].values())
        gamma = list(results[mode]["gamma_by_CAD"].values())
        result[mode] = {"sampled_cases": len(cases),
            "mean_natural_losses": {k: sum(r["natural"][k] for r in cases) / len(cases) for k in names},
            "mean_masked_losses": {k: sum(r["gamma_masked"][k] for r in cases) / len(cases) for k in names},
            "mean_delta_masked_minus_natural": {k: sum(r["delta_masked_minus_natural"][k] for r in cases) / len(cases) for k in names},
            "cases_masking_increases_loss": {k: sum(r["delta_masked_minus_natural"][k] > 0 for r in cases) for k in names},
            "cases_masking_decreases_loss": {k: sum(r["delta_masked_minus_natural"][k] < 0 for r in cases) for k in names},
            "mean_delta_u_over_true_u_norm": sum(r["delta_u_over_true_u_norm"] for r in cases) / len(cases),
            "gamma_delta_RMS_mean4": sum(r["delta_RMS"] for r in gamma) / len(gamma),
            "gamma_spatial_centered_RMS_mean4": sum(r["spatial_centered_RMS"] for r in gamma) / len(gamma),
            "near_either_endpoint_fraction_mean4": sum(r["near_either_endpoint_fraction"] for r in gamma) / len(gamma),
            "exact_either_endpoint_fraction_mean4": sum(r["exact_either_endpoint_fraction"] for r in gamma) / len(gamma)}
    return result


def diagnose(args):
    contract = request_contract(args.plan)
    output = _path(args.output)
    if args.dry_run:
        return {"status": "metadata_preflight_passed", "contract": contract, "arrays_or_weights_opened": False, "GPU_used": False}
    if output.exists() and any(output.iterdir()):
        if (output / "summary.json").is_file():
            summary = _json(output / "summary.json")
            if summary.get("status") == "completed" and summary["config"]["contract"] == contract:
                return {"status": "completed_reused", "summary": _portable(output / "summary.json")}
        raise ValueError("Nonempty output is not overwritten without an exact completed contract")
    import numpy as np
    import torch
    from geobc.physics_train import DisplacementScaler, _load_basis, load_displacement_cases
    from geobc.physics_field import combine_basis
    from geobc.gradient_supervision import load_gradient_manifest, load_gradient_cache
    from geobc.local_geometry import load_local_geometry
    from geobc.bc_role_condition import load_bc_role_cloud
    from geobc.hash_modulation_field import HashModulationPhysicsField, HashModulationFieldConfig
    require(torch.cuda.is_available(), "Frozen native diagnostic requires root's scheduled CUDA; metadata --dry-run is CPU-only")
    began = time.monotonic()
    device = torch.device("cuda")
    output.mkdir(parents=True, exist_ok=True)
    config_record = {"contract": contract, "started_at_utc": _now(),
        "selection": "both fixed last10000, no best/reselection", "precision": "frozen high/TF32 FP32 context/gate, native FP16/masterFP32, physical P1 FP64",
        "scope": "TRAIN4 x ver/hor/tor only", "loss_units": "sampled normalized squared objectives, not recovered all-node tensor L2 percentages",
        "counterfactual": "only decoder.mode temporarily set to zero; same actual query/global context and trained parameters",
        "zero_prediction_check": "exact equality (rtol0/atol0) on the same mixed rows",
        "limits": ["Inference masking can be out of distribution and does not predict retrained performance.",
            "Saturation/use is descriptive TRAIN evidence, not the cause of validation failure.",
            "Equal stored parameters does not mean equal active function capacity; historical Composite is not a matched control."]}
    _write(output / "config.json", config_record)
    def status(phase, **extra):
        _write(output / "status.json", {"status": phase, "updated_at_utc": _now(), "elapsed_seconds": time.monotonic() - began, **extra})
    model = None
    forward_calls = 0
    try:
        torch.manual_seed(SEED)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
        configs = {m: _json(contract["frozen_checkpoints"][m]["config"]["path"]) for m in ("zero", "actual")}
        reference = configs["actual"]
        split = _json(contract["split"]["path"])
        files = {key: _path(contract["selected_sources"][key]["path"]) for key in KEYS}
        guard = _load_basis(_path(contract["load_basis"]["path"]), split)
        # Frozen loader only opens the explicitly passed TRAIN known-load keys and never stress.
        cases = load_displacement_cases(list(KEYS), files, guard, "cpu")
        by_key = {case.key: case for case in cases}
        scaler = DisplacementScaler.from_record(reference["scaler"])
        gradient_directory = _path(reference["arguments"]["gradient_cache"])
        manifest = load_gradient_manifest(gradient_directory)
        data, clouds, witnesses = {}, {}, {}
        status("loading_fixed_train_subset")
        for rank, cad in enumerate(CADS):
            group = [by_key[f"{cad}/{load}"] for load in LOADS]
            cache = load_gradient_cache(gradient_directory, cad, xyz=group[0].xyz, manifest=manifest)
            local = load_local_geometry(_path(reference["arguments"]["local_geometry"]), cad)
            local_tensor = local.tensor(mode="actual", device="cpu")
            for case in group:
                cache.verify_xyz(case.xyz)
                local.verify_xyz(case.xyz)
                case.features = torch.cat((case.features, local_tensor), dim=-1)
            cloud = load_bc_role_cloud(files[f"{cad}/ver"], cad_key=cad,
                seed=reference["geometry"]["CADs"][cad]["seed"], cloud_points=1024)
            actual_cloud, expected_cloud = cloud.record(), reference["geometry"]["CADs"][cad]
            for name in ("points5_sha256", "coordinates_sha256", "roles_sha256", "mask_sha256", "node_indices_sha256", "source_xyz_sha256"):
                require(actual_cloud[name] == expected_cloud[name], f"{cad}: fixed saved cloud {name} differs")
            clouds[cad] = (torch.as_tensor(cloud.points[None], device=device), torch.as_tensor(cloud.mask[None], device=device))
            arrays, witness = sample_witness(group[0].count, cache.corners, rank)
            witness.update(file=f"witness_{cad}.npz", normalized_xyz_sha256=cache.metadata["xyz_sha256"],
                           cloud_hash_verified_against_both_configs=True)
            np.savez(output / witness["file"], **arrays)
            witnesses[cad] = {"arrays": arrays, "record": witness}
            data[cad] = {"cache": cache, "tensors": {name: torch.as_tensor(value, device=device) for name, value in
                (("inverse_physical_edge_mm", cache.inverse_physical_edge_mm), ("volumes", cache.volumes),
                 ("reference_cell_tensor_strain", cache.reference_cell_tensor_strain))}}
        _write(output / "witnesses.json", {cad: witnesses[cad]["record"] for cad in CADS})
        results = {}
        # Entire model forward/context/loss section is inference only.
        with torch.no_grad():
            for mode in ("zero", "actual"):
                status("loading_checkpoint", model=mode, basis_forward_calls=forward_calls)
                cfg = configs[mode]
                checkpoint_path = _path(contract["frozen_checkpoints"][mode]["checkpoint"]["path"])
                saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                latest = _json(contract["frozen_checkpoints"][mode]["last_metrics"]["path"])
                require(saved["step"] == STEP and saved["config"] == cfg and saved["scaler"] == cfg["scaler"]
                        and saved["metrics"] == latest, "Loaded last.pt differs from frozen final10k metadata")
                model_cfg = HashModulationFieldConfig(**{f.name: cfg["model"][f.name] for f in fields(HashModulationFieldConfig) if f.name in cfg["model"]})
                model = HashModulationPhysicsField(model_cfg, device=device)
                require(model.source_config == cfg["model"], "Reconstructed actual split source_config differs")
                model.load_state_dict(saved["model"], strict=True)
                del saved  # optimizer state is never applied or used
                model.eval()
                versions = {name: parameter._version for name, parameter in model.named_parameters()}
                head = {"weight_norm": float(model.gate_head.weight.double().norm()), "bias_norm": float(model.gate_head.bias.double().norm()),
                    "weight_abs_max": float(model.gate_head.weight.abs().max()), "bias_abs_max": float(model.gate_head.bias.abs().max()),
                    "weight_and_bias_exact_zero": bool(model.gate_head.weight.eq(0).all() and model.gate_head.bias.eq(0).all()),
                    "parameters": sum(p.numel() for p in model.gate_head.parameters())}
                if mode == "zero":
                    require(head["weight_and_bias_exact_zero"], "Zero trained gate W/b must remain exact0")
                rows, gamma_by_cad, zero_checks = {}, {}, {}
                for cad in CADS:
                    nodes = witnesses[cad]["arrays"]["query_nodes"]
                    cell_ids = torch.as_tensor(witnesses[cad]["arrays"]["sampled_cells"], device=device)
                    canonical = by_key[f"{cad}/ver"]
                    xyz, features = canonical.xyz[nodes].to(device), canonical.features[nodes].to(device)
                    context = model.encode_geometry(*clouds[cad])
                    query = model.field.query_context(xyz, context)
                    captured = []
                    def capture_gate(module, inputs, result):
                        captured.append(1 + torch.tanh(result.detach().float()))
                    handle = model.gate_head.register_forward_hook(capture_gate)
                    try:
                        require(model.decoder.mode == mode, "Natural decoder gate mode differs from loaded model")
                        natural = forward_basis(model, xyz, features, context, query)
                        forward_calls += 1
                    finally:
                        handle.remove()
                    require(len(captured) == 1, "One transparent gate hook per natural basis forward required")
                    gamma_by_cad[cad] = gamma_statistics(captured[0])
                    previous_mode = model.decoder.mode
                    try:
                        model.decoder.mode = "zero"
                        masked = forward_basis(model, xyz, features, context, query)
                        forward_calls += 1
                    finally:
                        model.decoder.mode = previous_mode
                    require(bool(torch.isfinite(natural).all() and torch.isfinite(masked).all()), "Nonfinite sampled basis")
                    if mode == "zero":
                        require(torch.equal(natural, masked), "Zero natural/masked basis must be exactly identical")
                        zero_checks[cad] = {"natural_masked_basis_exact_equal": True, "max_abs_scaled_basis_delta": 0.0}
                    for load in LOADS:
                        case = by_key[f"{cad}/{load}"]
                        alpha = case.alpha.to(device).expand(ROWS, -1)
                        prediction, counterfactual = combine_basis(natural, alpha), combine_basis(masked, alpha)
                        target = torch.as_tensor(scaler.encode(case.physical_u[nodes]), device=device)
                        weight = cfg["u_loss"]["case_records"][case.key]["weight"]
                        intact = loss_record(prediction, case, target, cell_ids, data, scaler, weight)
                        intervened = loss_record(counterfactual, case, target, cell_ids, data, scaler, weight)
                        true = case.physical_u[nodes]
                        denom = float(np.linalg.norm(true))
                        require(math.isfinite(denom) and denom > 0, "Sampled signed-u norm must be positive")
                        delta = (counterfactual.double() - prediction.double()).cpu().numpy() * scaler.scale_mm
                        rows[case.key] = {"query_rows": ROWS, "natural": intact, "gamma_masked": intervened,
                            "delta_masked_minus_natural": {k: intervened[k] - v for k, v in intact.items()},
                            "masked_to_natural_loss_ratio": {k: intervened[k] / v if v > 0 else None for k, v in intact.items()},
                            "natural_loss_zero": {k: v == 0 for k, v in intact.items()},
                            "delta_u_component_RMS_mm": float(np.sqrt(np.mean(delta ** 2))),
                            "delta_u_over_true_u_norm": float(np.linalg.norm(delta) / denom),
                            "scope": "signed-u/P1 TRAIN target on one mixed node/corner witness; not all-node metric"}
                        if mode == "zero":
                            require(intact == intervened, "Zero natural/masked sampled losses differ")
                    del natural, masked, captured, context, query
                    status("CAD_probe_completed", model=mode, CAD=cad, basis_forward_calls=forward_calls)
                require(versions == {name: p._version for name, p in model.named_parameters()}
                        and all(p.grad is None for p in model.parameters()) and model.decoder.mode == mode,
                        "Frozen model parameters/gradients/mode changed during inference")
                results[mode] = {"fixed_step": STEP, "trained_gate_head": head, "gamma_by_CAD": gamma_by_cad,
                    "cases": rows, "zero_control_checks": zero_checks, "parameters_unchanged": True,
                    "gradients_absent": True, "gate_mode_restored": True}
                del model
                model = None
                torch.cuda.empty_cache()
        require(forward_calls == 16 and request_contract(args.plan) == contract, "Forward cap or frozen input identity changed")
        memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        del data, clouds, xyz, features, target, alpha, cell_ids, prediction, counterfactual
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        summary = {"status": "completed", "config": config_record, "fixed_step": STEP,
            "basis_forward_calls": forward_calls, "backwards": 0, "optimizer_updates": 0,
            "witnesses": {cad: witnesses[cad]["record"] for cad in CADS}, "models": results,
            "readout": brief_readout(results), "elapsed_seconds": time.monotonic() - began,
            "memory": memory, "completed_at_utc": _now(), "GPU_tensors_released_before_completion": True}
        _write(output / "summary.json", summary)
        status("completed", basis_forward_calls=forward_calls, summary="summary.json")
        return {"status": "completed", "summary": _portable(output / "summary.json"), "elapsed_seconds": summary["elapsed_seconds"]}
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", default="notes/geobc_hash_modulation_use_plan.json")
    result.add_argument("--output", default="runs/geobc_exploration/hash_modulation_use_frozen_10k")
    result.add_argument("--dry-run", action="store_true", help="JSON/SHA/stat only; no arrays, weights, NumPy/Torch import or CUDA")
    return result


if __name__ == "__main__":
    value = diagnose(parser().parse_args())
    if value.get("status") == "metadata_preflight_passed":
        value = {"status": value["status"], "plan": value["contract"]["plan"], "cases": value["contract"]["cases"],
            "max_basis_forward_calls": 16, "backwards": 0, "optimizer_updates": 0,
            "arrays_or_weights_opened": False, "GPU_used": False}
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
