"""One root-scheduled synthetic CUDA check of fused43 distance-context fields.

CPU request_contract/--dry-run read only the plan and small source identities.
No dataset/checkpoint/target or optimizer is used. Native-ready is written
only after actual public native counts, equal initialization/common-input
forward, force masking and two first-order backwards pass on CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.validate_hash_modulation_native import _path, _portable, _json, _sha, _now, _write, parameter_gradient_record

SEED = 2026
MODES = ("euclidean", "graph")
COUNTS = {"hash": 7733632, "network": 36864, "native_total": 7770496,
          "PointNet": 16800, "query_projection": 2080, "total_expected_each_model": 7789376}
SOURCE_NAMES = ("geobc/graph_context_field.py", "tools/validate_graph_context_native.py",
    "geobc/query_context_field.py", "geobc/conditioned_field.py", "geobc/physics_field.py",
    "geobc/bc_role_field.py", "geobc/tcnn_fields.py", "geobc/query_context_train.py",
    "geobc/conditional_train.py", "tools/validate_hash_modulation_native.py")
FROZEN_QUERY_FIELD_SHA = "47526d5642c1c3b08d18ca9c32806f910c635cf1c4857194a16dbb2d5ee3d39c"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def request_contract(plan_path):
    """JSON/source SHA only; no numpy/torch/tinycudann imports or data stats."""
    plan_path = _path(plan_path)
    plan = _json(plan_path)
    arch, pair, derived, recipe = (plan[k] for k in ("architecture", "pair", "native_source_derivation", "recipe"))
    require(plan["status"] == "predeclared" and pair["distance_modes"] == list(MODES)
            and pair["fresh_complete_initialization"] is True and pair["same_constructor_order"] is True
            and pair["third_zero_group_enabled"] is False and pair["historical_control_reused"] is False
            and pair["warmstart"] is False and pair["seed"] == SEED,
            "Requires the one predeclared fresh Euclidean-interface/mesh-edge-path pair")
    require(arch["condition_dim"] == 43 and arch["native_condition_dim"] == 107
            and arch["raw_Composite_input_dim"] == 110 and arch["interface_distance_slice"] == [38, 43]
            and arch["query_context_mode"] == arch["bc_role_mode"] == "actual"
            and arch["hash_modulation_enabled"] is False
            and arch["native_Identity_input_order"] == ["masked_original38", "active_interface_distance5", "actual_query32", "actual_global32"]
            and arch["network"] == {"otype": "FullyFusedMLP", "n_neurons": 128, "n_hidden_layers": 2, "activation": "ReLU", "output_activation": "None"}
            and arch["hash"] == {"n_levels": 12, "n_features_per_level": 2, "log2_hashmap_size": 19,
                "base_resolution": 8, "per_level_scale": 1.6, "interpolation": "Linear", "hash": "CoherentPrime"},
            "Fixed43/107/hash/fused128 architecture changed")
    require(derived["official_commit"] == "3daa6e5b8b8e0fa17971a1a20acfbb3754494b38"
            and derived["source_derived_counts"] == COUNTS and derived["nominal_encoded_channels"] == 131
            and derived["derived_padded_encoded_channels"] == 144 and derived["Identity_padding_count"] == 13
            and derived["Identity_padding_value"] == 1 and derived["no_flat_offset_guess"] is True,
            "Fixed official padding/count source derivation changed")
    require(recipe["steps"] == 10000 and recipe["batch_size"] == 51200 and recipe["case_slots"] == 4
            and recipe["gradient_weight"] == .0004106650618638651
            and recipe["u_scale_mm"] == .05511113670847156 and recipe["u_center_mm"] == 0
            and plan["scope"]["train_CADs"] == 40 and plan["scope"]["validation_CADs"] == 5
            and not any(plan["scope"][k] for k in ("dia_enabled", "final_test_enabled", "stress_targets_enabled")),
            "Frozen recipe/closed scope changed")
    sources = {name: _sha(ROOT / name) for name in SOURCE_NAMES}
    require(sources["geobc/query_context_field.py"] == FROZEN_QUERY_FIELD_SHA, "Frozen query helper changed")
    return {"version": 1, "plan": {"path": _portable(plan_path), "sha256": _sha(plan_path)},
        "sources_sha256": sources, "distance_modes": list(MODES), "seed": SEED,
        "synthetic_rows": 512, "synthetic_CADs": 2, "cloud_points": 1024,
        "model_config": {"condition_dim": 43, "query_mode": "actual", "bc_role_mode": "actual",
            "mode": "linear_basis", "geometry_mode": "shape", "descriptor_dim": 32,
            "query_dim": 32, "sigma": .15, "query_chunk": 4096,
            "hidden_dim": 128, "hidden_layers": 2, "seed": SEED, "cloud_points": 1024},
        "native_counts_expected": COUNTS, "native_combined_public_parameter": "field.decoder.native.params",
        "encoded_layout_source_derived": {"conditions": 107, "nominal": 131, "padded": 144, "Identity_padding": 13, "padding_value": 1},
        "natural_step0_prediction_equality_required": False,
        "common_identical_distance_fixture_prediction_equality_required": True,
        "basis_forward_calls": 6, "backwards": 2, "optimizer_updates": 0,
        "no_GT": True, "dataset_files_opened": 0, "checkpoint_files_opened": 0,
        "scope": "synthetic native API/layout/gradient check, not a scientific data experiment"}


def parameter_hashes(model):
    """Hash complete named parameters and the real public combined buffer."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.tobytes())
    native = model.decoder.native.params.detach().cpu().contiguous().numpy()
    return {"all_named_parameter_sha256": digest.hexdigest(),
            "combined_native_buffer_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
            "native_buffer_count": int(native.size), "native_buffer_public_name": "field.decoder.native.params",
            "flat_subbuffer_offsets_inferred": False}


def smoke(args):
    contract = request_contract(args.plan)
    if args.dry_run:
        return {"status": "metadata_preflight_passed", "contract": contract, "GPU_used": False, "arrays_or_weights_opened": False}
    field_ready = _json(ROOT / "notes/graph_context_field_ready.json")
    require(field_ready["status"] == "ready" and field_ready["plan_sha256"] == contract["plan"]["sha256"]
            and field_ready["source_sha256"] == contract["sources_sha256"]["geobc/graph_context_field.py"]
            and field_ready["smoke_sha256"] == contract["sources_sha256"]["tools/validate_graph_context_native.py"],
            "Scheduled native smoke must match final CPU field/smoke ready artifacts")
    output = _path(args.output)
    native_ready_path = ROOT / "notes/graph_context_native_ready.json"
    if output.exists() and any(output.iterdir()):
        summary_path = output / "summary.json"
        if summary_path.exists() and native_ready_path.exists():
            summary = _json(summary_path)
            if (summary.get("status") == "completed" and summary["config"]["contract"] == contract
                    and _json(native_ready_path)["summary_sha256"] == _sha(summary_path)):
                return {"status": "completed_reused", "summary": _portable(summary_path)}
        raise ValueError("Nonempty incomplete/changed smoke output must be retained or explicitly archived")
    import torch
    from geobc.graph_context_field import GraphContextFieldConfig, GraphContextPhysicsField
    from geobc.physics_field import combine_basis
    from geobc.query_context_train import _query_gradient
    from geobc.conditional_train import _encoder_gradient
    require(torch.cuda.is_available(), "Actual CUDA required; --dry-run is metadata only")
    device = torch.device("cuda")
    began = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    config = {"contract": contract, "started_at_utc": _now(), "no_GT": True,
              "optimizer": None, "precision": "frozen high/TF32 FP32 PointNet/query; native FP16 compute/FP32 masters"}
    _write(output / "config.json", config)
    _write(output / "status.json", {"status": "running", "started_at_utc": _now()})
    models = {}
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
        generator = torch.Generator(device="cpu").manual_seed(SEED + 1803)
        cloud_cpu = torch.rand((2, 1024, 5), generator=generator)
        cloud_cpu[..., 3:] = (cloud_cpu[..., 3:] > .9).float()
        mask_cpu = torch.ones((2, 1024), dtype=torch.bool)
        mask_cpu[:, -8:] = False
        cloud_cpu[~mask_cpu] = 0
        xyz_cpu = torch.rand((512, 3), generator=generator)
        features_cpu = torch.rand((512, 43), generator=generator)
        features_cpu[:, :7] = (features_cpu[:, :7] > .8).float()
        graph_features_cpu = features_cpu.clone()
        graph_features_cpu[:, 38:43] *= 1 + .2 * torch.rand((512, 5), generator=generator)
        alpha_cpu = torch.eye(3)[torch.arange(512) % 3]
        cad_cpu = torch.arange(512) % 2
        witness = {"seed": SEED + 1803, "generator": "independent CPU generator, no source fields/targets",
            **{name + "_sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in
               {"cloud": cloud_cpu, "mask": mask_cpu, "xyz": xyz_cpu, "common_euclidean_fixture": features_cpu,
                "natural_graph_fixture": graph_features_cpu}.items()}, "no_target_tensor": True}
        cloud, mask, xyz, common, graph_features, alpha, cad_index = [x.to(device) for x in
            (cloud_cpu, mask_cpu, xyz_cpu, features_cpu, graph_features_cpu, alpha_cpu, cad_cpu)]
        layouts, initial = {}, {}
        for mode in MODES:
            torch.manual_seed(SEED)
            models[mode] = GraphContextPhysicsField(GraphContextFieldConfig(distance_mode=mode, **contract["model_config"]), device=device)
            layouts[mode] = models[mode].validate_native_layout()
            initial[mode] = parameter_hashes(models[mode])
        require(initial["euclidean"] == initial["graph"], "Whole/native initial parameter hashes differ")
        states = [models[m].state_dict() for m in MODES]
        require(list(states[0]) == list(states[1]) and all(torch.equal(states[0][k], states[1][k]) for k in states[0]),
                "Complete initial states differ")
        state_names = list(states[0])
        del states
        common_predictions, natural_predictions, records = {}, {}, {}
        forward_calls = 0
        for mode in MODES:
            model = models[mode]
            context = model.encode_geometry(cloud, mask)
            context.retain_grad()
            basis = model.response_basis(xyz, common, context=context, cad_index=cad_index)
            forward_calls += 1
            prediction = combine_basis(basis, alpha)
            require(bool(torch.isfinite(prediction).all()), "Nonfinite common synthetic prediction")
            common_predictions[mode] = prediction.detach().cpu()
            with torch.no_grad():
                perturbed = common.clone()
                perturbed[:, 7:13] += 123
                force_basis = model.response_basis(xyz, perturbed, context=context, cad_index=cad_index)
                forward_calls += 1
                require(torch.equal(basis.detach(), force_basis), "Force7:13 perturbation changed common linear basis")
                natural_features = common if mode == "euclidean" else graph_features
                natural_basis = model.response_basis(xyz, natural_features, context=context, cad_index=cad_index)
                forward_calls += 1
                natural_predictions[mode] = combine_basis(natural_basis, alpha).cpu()
                require(bool(torch.isfinite(natural_predictions[mode]).all()), "Nonfinite natural-distance prediction")
            # No target or optimizer: one scalar backward tests first-order routing.
            objective = prediction.square().mean() + prediction.mean()
            objective.backward()
            records[mode] = {"native_layout": layouts[mode], "source_config": model.source_config,
                "actual_counts": {"native_total": model.decoder.native.params.numel(),
                    "PointNet": sum(p.numel() for p in model.encoder.parameters()),
                    "query_projection": sum(p.numel() for p in model.query_projection.parameters()),
                    "total_expected_each_model": sum(p.numel() for p in model.parameters())},
                "native_gradient": parameter_gradient_record([model.decoder.native.params]),
                "encoder_gradient": _encoder_gradient(model, context),
                "query_projection_gradient": _query_gradient(model, context, "actual"),
                "force_mask_exact_basis_invariant": True, "objective": float(objective.detach()),
                "natural_distance_inputs_active": True, "no_optimizer_or_parameter_update": True}
            require(parameter_hashes(model) == initial[mode], "Synthetic forwards/backward changed parameters")
            model.zero_grad(set_to_none=True)
            del context, basis, prediction, objective, force_basis, natural_basis, perturbed, natural_features
        require(torch.equal(common_predictions["euclidean"], common_predictions["graph"]), "Identical distance fixture must predict exactly equally")
        require(forward_calls == 6 and request_contract(args.plan) == contract, "Forward cap or frozen source/plan changed")
        natural_delta = float((natural_predictions["graph"] - natural_predictions["euclidean"]).double().abs().max())
        memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        del model, models, cloud, mask, xyz, common, graph_features, alpha, cad_index
        models = {}
        torch.cuda.synchronize(device)
        import tinycudann as tcnn
        tcnn.free_temporary_memory()
        torch.cuda.empty_cache()
        summary = {"status": "completed", "config": config, "paired_initial_state_exact_equal": True,
            "initial_parameters": initial, "paired_state_names": state_names,
            "common_identical_distance_fixture_prediction_exact_equal": True,
            "natural_step0_prediction_equality_required": False, "natural_distance_fixture_max_prediction_delta": natural_delta,
            "shared_synthetic_inputs": witness, "native_checks": records, "basis_forward_calls": forward_calls,
            "backwards": 2, "optimizer_updates": 0, "no_GT": True, "dataset_or_checkpoint_files_opened": 0,
            "elapsed_seconds": time.monotonic() - began, "memory": memory, "GPU_models_released_before_completion": True,
            "completed_at_utc": _now(), "limit": "Synthetic API/count/gradient availability only; not geometry/PDE/data performance"}
        _write(output / "summary.json", summary)
        _write(native_ready_path, {"status": "ready", "source_sha256": contract["sources_sha256"],
            "plan_sha256": contract["plan"]["sha256"], "summary": _portable(output / "summary.json"),
            "summary_sha256": _sha(output / "summary.json"), "native_layout": layouts,
            "paired_complete_parameters_exact_equal": True, "common_fixture_forward_exact_equal": True,
            "force_mask_and_native_encoder_query_gradients_passed": True, "natural_forward_equality_not_required": True,
            "actual_CUDA_passed": True, "no_GT": True, "optimizer_updates": 0, "completed_at_utc": _now()})
        _write(output / "status.json", {"status": "completed", "elapsed_seconds": summary["elapsed_seconds"]})
        return {"status": "completed", "summary": _portable(output / "summary.json"), "elapsed_seconds": summary["elapsed_seconds"]}
    except Exception as error:
        _write(output / "status.json", {"status": "failed", "error": f"{type(error).__name__}: {error}", "elapsed_seconds": time.monotonic() - began})
        raise
    finally:
        if models:
            del models
        torch.cuda.empty_cache()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", default="notes/geobc_graph_context_candidate_plan.json")
    result.add_argument("--output", default="runs/geobc_exploration/graph_context_native_smoke")
    result.add_argument("--dry-run", action="store_true")
    return result


if __name__ == "__main__":
    value = smoke(parser().parse_args())
    if value.get("status") == "metadata_preflight_passed":
        value = {"status": value["status"], "plan": value["contract"]["plan"], "GPU_used": False, "arrays_or_weights_opened": False}
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
