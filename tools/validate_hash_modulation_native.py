"""One scheduled synthetic CUDA check of the matched split-native hash gate.

No dataset, checkpoint, target field, optimizer or update is used. CPU --dry-run
only verifies small JSON/source identities and the selected architecture plan.
Native-ready is written only after both native modules and gate backprop pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED = 2026
SOURCE_NAMES = ("geobc/hash_modulation_field.py", "tools/validate_hash_modulation_native.py",
    "geobc/query_context_field.py", "geobc/query_context_train.py", "geobc/conditioned_field.py",
    "geobc/physics_field.py", "geobc/bc_role_field.py", "geobc/tcnn_fields.py", "geobc/conditional_train.py")
FROZEN_QUERY_FIELD_SHA = "47526d5642c1c3b08d18ca9c32806f910c635cf1c4857194a16dbb2d5ee3d39c"


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


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    path = _path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def request_contract(plan_path):
    """Metadata only; does not import torch, numpy or tinycudann."""
    plan_path = _path(plan_path)
    plan = _json(plan_path)
    common, gate, native, pair = (plan[name] for name in ("common_architecture", "gate_modulation", "native_backend", "pair"))
    expected_common = {"physics_mode": "linear_basis", "BC_role_mode": "actual", "query_context_mode": "actual",
        "global_dim": 32, "query_dim": 32, "condition_dim": 38, "hidden_width": 128,
        "hidden_layers": 2, "output_channels": 9, "sigma": .15, "query_chunk": 4096}
    if (plan["status"] != "predeclared" or any(common.get(k) != v for k, v in expected_common.items())
            or gate["unique_formula"] != "gamma = 1 + tanh(Linear64to24(context64))"
            or gate["context_order"] != ["query32", "global32"]
            or gate["head_parameters"] != 1560 or gate["extra_strength_parameter"] is not None
            or gate["additive_shift"] or gate["alternate_gate_forms"]
            or pair["hash_modulation_modes"] != ["zero", "actual"]
            or not pair["both_new_split_native_models"] or pair["reuse_old_Composite_control"]
            or native["official_commit"] != "3daa6e5b8b8e0fa17971a1a20acfbb3754494b38"
            or native["public_parameter_buffers"] != ["encoding.params", "network.params"]):
        raise ValueError("Smoke requires the single predeclared matched split-native hash gate")
    enc, net = native["encoding"], native["network"]
    if (any(enc.get(k) != v for k, v in {"n_input_dims": 3, "n_levels": 12, "n_features_per_level": 2,
            "log2_hashmap_size": 19, "base_resolution": 8, "per_level_scale": 1.6, "expected_output_dims": 24, "seed": SEED}.items())
            or any(net.get(k) != v for k, v in {"n_input_dims": 126, "n_output_dims": 9,
                "n_neurons": 128, "n_hidden_layers": 2, "seed": SEED}.items())
            or any(pair[name] != SEED for name in ("torch_seed", "encoding_seed", "network_seed"))):
        raise ValueError("Standalone native dimensions/hash/initialization seeds differ")
    hashes = {name: _sha(ROOT / name) for name in SOURCE_NAMES}
    if hashes["geobc/query_context_field.py"] != FROZEN_QUERY_FIELD_SHA:
        raise ValueError("Frozen query context field changed")
    return {"version": 1, "plan": {"path": _portable(plan_path), "sha256": _sha(plan_path)},
        "sources_sha256": hashes, "formula": gate["unique_formula"], "gate_context_order": gate["context_order"],
        "modes": ["zero", "actual"], "seed": SEED, "rows": 512, "synthetic_CADs": 2,
        "cloud_points": 1024, "query_chunk": 4096, "native_counts_expected": native["source_derived_counts"],
        "no_GT": True, "dataset_files_opened": 0, "checkpoint_files_opened": 0,
        "optimizer_updates": 0, "backwards": 2, "scope": "synthetic native interface validation, not a data experiment"}


def parameter_gradient_record(parameters, *, zero=False):
    import torch
    values = list(parameters)
    if not values or any(p.grad is None for p in values):
        raise RuntimeError("Required native/gate gradients must be present")
    if any(not bool(torch.isfinite(p.grad).all()) for p in values):
        raise FloatingPointError("Nonfinite native/gate gradients")
    l1 = sum(float(p.grad.detach().double().abs().sum()) for p in values)
    if zero and l1 != 0 or not zero and l1 <= 0:
        raise RuntimeError("Native/gate gradient violates exact-zero/nonzero policy")
    return {"parameter_tensors": len(values), "parameters": sum(p.numel() for p in values),
            "gradient_L1": l1, "present_and_finite": True, "expected": "exact_zero" if zero else "nonzero"}


def smoke(args):
    contract = request_contract(args.plan)
    if args.dry_run:
        return {"status": "metadata_preflight_passed", "contract": contract, "GPU_used": False}
    output = _path(args.output)
    if output.exists() and any(output.iterdir()):
        if (output / "summary.json").exists():
            summary = _json(output / "summary.json")
            ready_path = ROOT / "notes/hash_modulation_native_ready.json"
            if (summary.get("status") == "completed" and summary["config"]["contract"] == contract
                    and ready_path.exists() and _json(ready_path)["summary_sha256"] == _sha(output / "summary.json")):
                return {"status": "completed_reused", "summary": _portable(output / "summary.json")}
        raise ValueError("Nonempty incomplete or changed-contract smoke output must be retained/archived explicitly")
    import torch
    from geobc.hash_modulation_field import HashModulationFieldConfig, HashModulationPhysicsField
    from geobc.query_context_train import _query_gradient
    from geobc.conditional_train import _encoder_gradient
    if not torch.cuda.is_available():
        raise RuntimeError("Scheduled native smoke requires CUDA; --dry-run is metadata-only")
    device = torch.device("cuda")
    began = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    config = {"contract": contract, "started_at_utc": _now(), "no_GT": True, "optimizer": None,
              "precision": "FP32/highTF32 PointNet/query/gate, nativeFP16/FP32master, first-order backward only"}
    _write(output / "config.json", config)
    _write(output / "status.json", {"status": "running", "started_at_utc": _now()})
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
        generator = torch.Generator(device="cpu").manual_seed(SEED + 1703)
        cloud_cpu = torch.rand((2, 1024, 5), generator=generator)
        cloud_cpu[..., 3:] = (cloud_cpu[..., 3:] > .9).float()
        mask_cpu = torch.ones((2, 1024), dtype=torch.bool)
        mask_cpu[:, -8:] = False
        cloud_cpu[~mask_cpu] = 0
        xyz_cpu = torch.rand((512, 3), generator=generator)
        features_cpu = torch.rand((512, 38), generator=generator)
        features_cpu[:, :7] = (features_cpu[:, :7] > .8).float()
        alpha_cpu = torch.eye(3)[torch.arange(512) % 3]
        cad_index_cpu = torch.arange(512) % 2
        witness = {"seed": SEED + 1703, "generator": "independent CPU torch.Generator; one shared synthetic input for both branches",
            "cloud_sha256": hashlib.sha256(cloud_cpu.numpy().tobytes()).hexdigest(),
            "mask_sha256": hashlib.sha256(mask_cpu.numpy().tobytes()).hexdigest(),
            "xyz_sha256": hashlib.sha256(xyz_cpu.numpy().tobytes()).hexdigest(),
            "features_sha256": hashlib.sha256(features_cpu.numpy().tobytes()).hexdigest(),
            "no_target_tensor": True}
        cloud, mask, xyz, features, alpha, cad_index = [value.to(device) for value in
            (cloud_cpu, mask_cpu, xyz_cpu, features_cpu, alpha_cpu, cad_index_cpu)]
        models, layouts = {}, {}
        for mode in ("zero", "actual"):
            torch.manual_seed(SEED)
            models[mode] = HashModulationPhysicsField(HashModulationFieldConfig(hash_modulation_mode=mode), device=device)
            layouts[mode] = models[mode].decoder.validate_native_layout()
            if (sum(p.numel() for p in models[mode].parameters()) != 7_788_888
                    or models[mode].decoder.encoding.dtype != torch.float16
                    or models[mode].decoder.network.dtype != torch.float16
                    or bool(models[mode].gate_head.weight.ne(0).any()) or bool(models[mode].gate_head.bias.ne(0).any())):
                raise ValueError("Native precision/counts or exact-zero gate initialization differs")
        zero_state, actual_state = models["zero"].state_dict(), models["actual"].state_dict()
        if list(zero_state) != list(actual_state) or not all(torch.equal(zero_state[name], actual_state[name]) for name in zero_state):
            raise RuntimeError("Paired complete initial parameter/buffer states differ")
        state_names = list(zero_state)
        del zero_state, actual_state
        predictions, records = {}, {}
        for mode in ("zero", "actual"):
            model = models[mode]
            context = model.encode_geometry(cloud, mask)
            context.retain_grad()
            prediction = model(xyz, features, alpha, context=context, cad_index=cad_index)
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError("Nonfinite synthetic native prediction")
            predictions[mode] = prediction.detach().cpu()
            delta = model.decoder.gate_delta(torch.cat((features, features.new_zeros((512, 64))), dim=1))
            if bool(delta.ne(0).any()):
                raise RuntimeError("Both initial hash multipliers must be exactly1")
            # A synthetic scalar objective with no labels/target and no update.
            objective = prediction.square().mean() + prediction.mean()
            objective.backward()
            native_groups = model.decoder.native_parameter_groups()
            records[mode] = {"native_layout": layouts[mode], "source_config": model.source_config,
                "full_parameters": sum(p.numel() for p in model.parameters()), "step0_multiplier_exact1": True,
                "native_encoding_gradient": parameter_gradient_record(native_groups["encoding"]),
                "native_network_gradient": parameter_gradient_record(native_groups["network"]),
                "gate_head_gradient": parameter_gradient_record(model.gate_head.parameters(), zero=mode == "zero"),
                "query_projection_gradient": _query_gradient(model, context, "actual"),
                "encoder_gradient": _encoder_gradient(model, context), "objective": float(objective.detach()),
                "parameter_names": [name for name, _ in model.named_parameters()], "no_optimizer_or_parameter_update": True}
            model.zero_grad(set_to_none=True)
            del context, prediction, objective, delta, native_groups
        if not torch.equal(predictions["zero"], predictions["actual"]):
            raise RuntimeError("Same initialized split-native branches must give exactly equal synthetic predictions")
        if request_contract(args.plan) != contract:
            raise ValueError("Plan/source identities changed during the native smoke")
        memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        del model, models, cloud, mask, xyz, features, alpha, cad_index
        torch.cuda.synchronize(device)
        import tinycudann as tcnn
        tcnn.free_temporary_memory()
        torch.cuda.empty_cache()
        summary = {"status": "completed", "config": config, "source_sha256": contract["sources_sha256"],
            "paired_initial_state_exact_equal": True, "paired_state_names": state_names,
            "paired_step0_prediction_exact_equal": True, "paired_shared_synthetic_inputs": witness,
            "native_and_gate_checks": records, "elapsed_seconds": time.monotonic() - began,
            "backwards": 2, "optimizer_updates": 0, "no_GT": True, "dataset_or_checkpoint_files_opened": 0,
            "memory": memory, "GPU_models_released_before_completion": True, "completed_at_utc": _now(),
            "limit": "native synthetic API/gradient availability validation only; no empirical performance or useful-learning claim"}
        _write(output / "summary.json", summary)
        _write(ROOT / "notes/hash_modulation_native_ready.json", {"status": "ready",
            "source_sha256": contract["sources_sha256"], "plan_sha256": contract["plan"]["sha256"],
            "summary": _portable(output / "summary.json"), "summary_sha256": _sha(output / "summary.json"),
            "native_layout": layouts, "paired_init_and_forward_exact": True,
            "all_native_gate_encoder_projection_gradient_policies_passed": True,
            "no_GT": True, "optimizer_updates": 0, "completed_at_utc": _now()})
        _write(output / "status.json", {"status": "completed", "elapsed_seconds": summary["elapsed_seconds"]})
        return {"status": "completed", "summary": _portable(output / "summary.json"), "elapsed_seconds": summary["elapsed_seconds"]}
    except Exception as error:
        _write(output / "status.json", {"status": "failed", "error": f"{type(error).__name__}: {error}", "elapsed_seconds": time.monotonic() - began})
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", default="notes/geobc_hash_modulation_plan.json")
    result.add_argument("--output", default="runs/geobc_exploration/hash_modulation_native_smoke")
    result.add_argument("--dry-run", action="store_true")
    return result


if __name__ == "__main__":
    result = smoke(parser().parse_args())
    if result.get("status") == "metadata_preflight_passed":
        result = {"status": result["status"], "plan": result["contract"]["plan"], "GPU_used": False, "arrays_or_weights_opened": False}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
