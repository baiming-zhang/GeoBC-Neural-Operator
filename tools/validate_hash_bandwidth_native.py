"""One root-scheduled synthetic native check of the fixed full12/coarse8 pair.

Metadata --dry-run is standard-library only. --cpu-check injects small Torch
components for interfaces, not native evidence. Normal execution needs CUDA
and writes native-ready only after both public buffers and masked gradients
pass. No dataset, target, checkpoint, optimizer or parameter update is used.
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
PLAN_SHA = "8fc4e3355c8892f143428383049e39fa6f0fe380968cbefd59bb3b3058e42f12"
SOURCES = ("geobc/hash_bandwidth_field.py", "tools/validate_hash_bandwidth_native.py",
    "geobc/graph_context_field.py", "geobc/query_context_field.py", "geobc/physics_field.py",
    "geobc/conditioned_field.py", "geobc/bc_role_field.py", "geobc/tcnn_fields.py",
    "geobc/query_context_train.py", "geobc/conditional_train.py")


def path(value):
    text = str(value).replace("\\", "/")
    if os.name == "nt" and text.startswith("/mnt/") and len(text) > 7:
        text = text[5].upper() + ":/" + text[7:]
    elif os.name != "nt" and len(text) > 2 and text[1:3] == ":/":
        text = "/mnt/" + text[0].lower() + "/" + text[3:]
    result = Path(text)
    return (result if result.is_absolute() else ROOT/result).resolve()


def portable(value):
    return path(value).relative_to(ROOT).as_posix()


def sha(value):
    return hashlib.sha256(path(value).read_bytes()).hexdigest()


def read_json(value):
    return json.loads(path(value).read_text(encoding="utf-8"))


def write_json(value, record):
    p = path(value); p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix+".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(p)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def request_contract(plan_path):
    """Immutable JSON/source identities only; no scientific runtime imports."""
    plan_path = path(plan_path); plan = read_json(plan_path)
    require(sha(plan_path) == PLAN_SHA and plan["status"] == "predeclared"
            and plan["stage"] == "matched_fixed_hash_bandwidth", "Canonical fixed bandwidth plan changed")
    arch = plan["architecture"]
    require(arch["condition_dim"] == 43 and arch["native_condition_dim"] == 107
            and arch["distance_mode"] == "euclidean" and arch["query_context_mode"] == "actual"
            and arch["bc_role_mode"] == "actual" and arch["new_learned_head_parameters"] == 0
            and plan["selection"]["unique_pair"] == ["full12", "coarse8"]
            and plan["pair"]["modes"] == ["full12", "coarse8"]
            and plan["pair"]["seed"] == SEED and not plan["pair"]["warmstart"]
            and not plan["pair"]["historical_control_reused"], "Only the fresh Euclidean43 mask pair is permitted")
    counts = arch["source_derived_counts"]
    require(counts == {"encoding": 7733632, "network": 36864, "PointNet": 16800,
            "query_projection": 2080, "total_native": 7770496, "total_model": 7789376}, "Fixed source counts differ")
    return {"version": 1, "plan": {"path": portable(plan_path), "sha256": sha(plan_path)},
        "sources_sha256": {name: sha(ROOT/name) for name in SOURCES}, "seed": SEED,
        "modes": ["full12", "coarse8"], "distance_mode": "euclidean", "native_counts_expected": counts,
        "rows": 512, "synthetic_CADs": 2, "cloud_points": 1024, "query_chunk": 4096,
        "native_encoding_input_output": [3, 24], "native_network_input_output": [144, 9],
        "network_native_output_padding_source_derived": 16, "explicit_one_padding_channels": 13,
        "parameter_equality_scope": "all named trainable parameters; fixed feature_mask buffer differs",
        "dataset_files_opened": 0, "checkpoint_files_opened": 0, "GT_fields_read": 0,
        "backwards": 2, "optimizer_updates": 0, "scope": "synthetic API/gradient acceptance only"}


def named_parameter_record(model):
    """Complete public named trainable identity; no private/native flat slices."""
    digest = hashlib.sha256(); records = {}
    for name, p in model.named_parameters():
        data = p.detach().cpu().contiguous().numpy().tobytes()
        item = {"shape": list(p.shape), "dtype": str(p.dtype), "numel": p.numel(), "sha256": hashlib.sha256(data).hexdigest()}
        records[name] = item
        digest.update(name.encode()); digest.update(str(p.dtype).encode()); digest.update(str(tuple(p.shape)).encode()); digest.update(data)
    return {"complete_named_trainable_sha256": digest.hexdigest(), "parameters": records,
            "total_parameters": sum(p.numel() for p in model.parameters())}


def native_gradient_record(parameters):
    import torch
    parameters = list(parameters)
    require(len(parameters) == 1 and parameters[0].grad is not None, "Each public native group needs one present gradient buffer")
    value = parameters[0].grad
    require(bool(torch.isfinite(value).all()), "Native parameter gradient must be finite")
    l1 = float(value.detach().double().abs().sum())
    require(l1 > 0, "Public native group gradient must be positive")
    return {"parameter_tensors": 1, "parameters": parameters[0].numel(), "present_finite": True, "gradient_L1": l1}


def activation_gradient_record(decoder, mode):
    import torch
    encoded, masked = decoder.encoded_activation, decoder.masked_activation
    require(encoded is not None and masked is not None and encoded.shape[1] == masked.shape[1] == 24
            and encoded.grad is not None and bool(torch.isfinite(encoded).all()) and bool(torch.isfinite(masked).all())
            and bool(torch.isfinite(encoded.grad).all()), "Captured pre/post-mask values and pre-mask gradients must be finite")
    active_l1 = float(encoded.grad[:, :16].double().abs().sum())
    fine_l1 = float(encoded.grad[:, 16:].double().abs().sum())
    raw_fine_l1 = float(encoded[:, 16:].detach().double().abs().sum())
    post_fine_l1 = float(masked[:, 16:].detach().double().abs().sum())
    require(active_l1 > 0 and raw_fine_l1 > 0, "Active gradients and natural fine-level values must exist")
    if mode == "coarse8":
        require(fine_l1 == 0 and post_fine_l1 == 0 and bool(masked[:, 16:].eq(0).all()),
                "Coarse8 requires exact-zero last8 pre-mask gradient and post-mask value")
    else:
        require(torch.equal(encoded, masked) and fine_l1 > 0, "Full12 must retain all24 activations and fine gradients")
    return {"capture_point": "FP32 Encoding output BEFORE mask", "shape": list(encoded.shape),
        "active_first16_gradient_L1": active_l1, "fine_last8_gradient_L1": fine_l1,
        "pre_mask_fine_last8_value_L1": raw_fine_l1, "post_mask_fine_last8_value_L1": post_fine_l1,
        "coarse_fine_gradient_exact_zero": fine_l1 == 0 if mode == "coarse8" else None,
        "coarse_post_mask_value_exact_zero": post_fine_l1 == 0 if mode == "coarse8" else None,
        "all_present_and_finite": True, "no_per_level_parameter_offsets": True}


def synthetic_inputs(rows=512, points=1024):
    import torch
    rng = torch.Generator(device="cpu").manual_seed(SEED+1703)
    cloud = torch.rand((2, points, 5), generator=rng)
    cloud[..., 3:] = (cloud[..., 3:] > .9).float()
    mask = torch.ones((2, points), dtype=torch.bool); mask[:, -8:] = False; cloud[~mask] = 0
    xyz = torch.rand((rows, 3), generator=rng)
    features = torch.rand((rows, 43), generator=rng); features[:, :7] = (features[:, :7] > .8).float()
    alpha = torch.eye(3)[torch.arange(rows) % 3]
    cad = torch.arange(rows) % 2
    return cloud, mask, xyz, features, alpha, cad


def cpu_check():
    """One small injected interface fixture; no tinycudann or real arrays."""
    import torch
    from torch import nn
    from geobc.hash_bandwidth_field import HashBandwidthFieldConfig, HashBandwidthPhysicsField, SplitHashBandwidthDecoder
    from geobc.query_context_train import _query_gradient
    from geobc.conditional_train import _encoder_gradient
    class Encoding(nn.Module):
        n_input_dims, n_output_dims, dtype = 3, 24, torch.float32
        def __init__(self):
            super().__init__(); self.params = nn.Parameter(torch.rand(24)*.01+.01)
        def forward(self, xyz):
            return (xyz.repeat(1, 8)+.2)*self.params
    class Network(nn.Module):
        n_input_dims, n_output_dims, dtype = 144, 9, torch.float32
        def __init__(self):
            super().__init__(); self.params = nn.Parameter(torch.rand(9*144)*.01)
        def forward(self, values):
            return values@self.params.reshape(9,144).T
    cloud, mask, xyz, features, alpha, cad = synthetic_inputs(32, 32)
    models = {}
    rng_end = {}
    for mode in ("full12", "coarse8"):
        torch.manual_seed(SEED)
        cfg = HashBandwidthFieldConfig(hash_bandwidth_mode=mode)
        decoder = SplitHashBandwidthDecoder(cfg.conditioned_config().decoder_config(), mode, "cpu", encoding=Encoding(), network=Network())
        models[mode] = HashBandwidthPhysicsField(cfg, "cpu", decoder)
        rng_end[mode] = torch.get_rng_state()
    records = {mode: named_parameter_record(model) for mode, model in models.items()}
    require(records["full12"] == records["coarse8"] and torch.equal(rng_end["full12"], rng_end["coarse8"]), "Paired parameter/RNG identity failed")
    require(not torch.equal(models["full12"].decoder.feature_mask, models["coarse8"].decoder.feature_mask), "Fixed masks must honestly differ")
    gradients = {}
    for mode, model in models.items():
        context = model.encode_geometry(cloud, mask); context.retain_grad()
        model.decoder.capture_activation_gradients = True
        pred = model(xyz, features, alpha, context=context, cad_index=cad)
        (pred.square().mean()+pred.mean()).backward()
        gradients[mode] = {"activation": activation_gradient_record(model.decoder, mode),
            "native_interface_encoding": native_gradient_record(model.native_parameter_groups()["encoding"]),
            "native_interface_network": native_gradient_record(model.native_parameter_groups()["network"]),
            "query": _query_gradient(model, context, "actual"), "encoder": _encoder_gradient(model, context)}
        model.decoder.clear_activation_capture()
        with torch.no_grad():
            altered = features.clone(); altered[:, 7:13] += 123
            a = model.response_basis(xyz, features, context=context, cad_index=cad)
            b = model.response_basis(xyz, altered, context=context, cad_index=cad)
            require(a.shape == (32,3,3) and torch.equal(a,b), "Inherited force mask/basis layout failed")
            require(model.decoder.encoded_activation is None, "Capture disabled must release graph references")
        source = model.source_config
        require(source["native_condition_dim"] == 107 and source["native_public_network_input_dim"] == 144
                and source["conditioned_field"]["decoder"]["network"]["n_input_dims"] == 144,
                "Split107/144 metadata must be truthful")
    # Common mask fixture, distinct from natural predictions: same parameters.
    with torch.no_grad():
        saved = models["coarse8"].decoder.feature_mask.clone()
        models["coarse8"].decoder.feature_mask.copy_(models["full12"].decoder.feature_mask)
        a = models["full12"](xyz, features, alpha, cloud, cad, geometry_mask=mask)
        b = models["coarse8"](xyz, features, alpha, cloud, cad, geometry_mask=mask)
        require(torch.equal(a,b), "Identical input/mask fixture must match")
        models["coarse8"].decoder.feature_mask.copy_(saved)
    try:
        HashBandwidthFieldConfig(distance_mode="graph")
        raise AssertionError("Invalid distance mode accepted")
    except ValueError:
        pass
    return {"status": "CPU_interface_checks_passed", "GPU_used": False, "real_arrays_or_weights_read": False,
        "checks": ["complete named trainable/RNG paired identity", "nonlearned mask difference declared", "common mask fixture parity",
            "force7:13 invariant 9basis", "pre-mask inactive gradients zero/postmask values zero", "both public-interface parameter groups/query/encoder finite positive",
            "capture releases references", "truthful107/144 split metadata", "invalid graph mode rejected"],
        "gradients": gradients, "native_CUDA_certified": False}


def smoke(args):
    contract = request_contract(args.plan)
    if args.dry_run:
        return {"status": "metadata_preflight_passed", "contract": contract, "GPU_used": False}
    if args.cpu_check:
        return cpu_check()
    output = path(args.output)
    if output.exists() and any(output.iterdir()):
        if (output/"summary.json").exists():
            summary = read_json(output/"summary.json"); ready = ROOT/"notes/hash_bandwidth_native_ready.json"
            if summary.get("status") == "completed" and summary["config"]["contract"] == contract and ready.exists() and read_json(ready)["summary_sha256"] == sha(output/"summary.json"):
                return {"status": "completed_reused", "summary": portable(output/"summary.json")}
        raise ValueError("Nonempty incomplete/changed smoke output requires explicit root recovery")
    import torch
    from geobc.hash_bandwidth_field import HashBandwidthFieldConfig, HashBandwidthPhysicsField
    from geobc.query_context_train import _query_gradient
    from geobc.conditional_train import _encoder_gradient
    require(torch.cuda.is_available(), "Scheduled native smoke requires CUDA; --dry-run is metadata only")
    device = torch.device("cuda"); began = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    config = {"contract": contract, "optimizer": None, "precision": "FP32/highTF32 PointNet/query; nativeFP16/FP32master; first-order parameter backward"}
    write_json(output/"config.json", config); write_json(output/"status.json", {"status": "running"})
    try:
        torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
        inputs_cpu = synthetic_inputs()
        witness = {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in zip(("cloud", "mask", "xyz", "features", "alpha", "cad_index"), inputs_cpu)}
        cloud, mask, xyz, features, alpha, cad = [v.to(device) for v in inputs_cpu]
        models, initial, layouts, rng_end = {}, {}, {}, {}
        for mode in ("full12", "coarse8"):
            torch.manual_seed(SEED)
            models[mode] = HashBandwidthPhysicsField(HashBandwidthFieldConfig(hash_bandwidth_mode=mode), device)
            layouts[mode] = models[mode].validate_native_layout()
            initial[mode] = named_parameter_record(models[mode])
            rng_end[mode] = torch.get_rng_state()
        require(initial["full12"] == initial["coarse8"] and torch.equal(rng_end["full12"], rng_end["coarse8"]), "Complete paired trainable initialization/RNG differ")
        masks = {m: models[m].decoder.feature_mask.detach().cpu().tolist() for m in models}
        require(masks["full12"] == [1.]*24 and masks["coarse8"] == [1.]*16+[0.]*8, "Mode-specific fixed mask mismatch")
        predictions, records = {}, {}
        for mode, model in models.items():
            context = model.encode_geometry(cloud, mask); context.retain_grad()
            model.decoder.capture_activation_gradients = True
            pred = model(xyz, features, alpha, context=context, cad_index=cad)
            require(bool(torch.isfinite(pred).all()), "Synthetic native prediction nonfinite")
            predictions[mode] = pred.detach().cpu()
            objective = pred.square().mean()+pred.mean(); objective.backward()
            groups = model.native_parameter_groups()
            records[mode] = {"layout": layouts[mode], "source_config": model.source_config,
                "native_encoding_gradient": native_gradient_record(groups["encoding"]),
                "native_network_gradient": native_gradient_record(groups["network"]),
                "activation_gradient": activation_gradient_record(model.decoder, mode),
                "query_gradient": _query_gradient(model, context, "actual"), "encoder_gradient": _encoder_gradient(model, context),
                "objective": float(objective.detach())}
            model.decoder.clear_activation_capture(); model.zero_grad(set_to_none=True)
            with torch.no_grad():
                changed = features.clone(); changed[:,7:13] += 123
                check = model(xyz, changed, alpha, context=context, cad_index=cad)
                require(torch.equal(check.detach().cpu(), predictions[mode]), "Force7:13 perturbation changed load-linear basis output")
            records[mode]["force_slot_invariant"] = True
            require(named_parameter_record(model) == initial[mode], "Parameters changed without an optimizer")
            del context, pred, objective, groups, changed, check
        require(request_contract(args.plan) == contract, "Frozen plan/source identity changed")
        memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        del model, models, cloud, mask, xyz, features, alpha, cad
        torch.cuda.synchronize(device)
        import tinycudann as tcnn
        tcnn.free_temporary_memory(); torch.cuda.empty_cache()
        summary = {"status": "completed", "config": config, "source_sha256": contract["sources_sha256"],
            "paired_named_trainable_initial_exact_equal": True, "initial_parameters": initial,
            "nonlearned_mask_buffers": masks, "nonlearned_mask_buffers_equal": False,
            "natural_step0_prediction_exact_equal_observed": torch.equal(predictions["full12"], predictions["coarse8"]),
            "natural_step0_prediction_equality_required": False, "shared_synthetic_input_sha256": witness,
            "native_checks": records, "memory": memory, "elapsed_seconds": time.monotonic()-began,
            "backwards": 2, "optimizer_updates": 0, "GT_fields_read": 0, "dataset_or_checkpoint_files_opened": 0,
            "GPU_models_released_before_completion": True,
            "limit": "synthetic API/gradient acceptance; equal stored capacity does not imply equal effective capacity or useful learning"}
        write_json(output/"summary.json", summary)
        write_json(ROOT/"notes/hash_bandwidth_native_ready.json", {"status": "ready", "plan_sha256": PLAN_SHA,
            "source_sha256": contract["sources_sha256"], "summary": portable(output/"summary.json"), "summary_sha256": sha(output/"summary.json"),
            "native_layout": layouts, "paired_trainable_init_exact": True, "mask_buffer_difference_declared": True,
            "public_native_encoder_query_and_inactive_activation_gradients_passed": True, "GT_fields_read": 0, "optimizer_updates": 0})
        write_json(output/"status.json", {"status": "completed", "elapsed_seconds": summary["elapsed_seconds"]})
        return {"status": "completed", "summary": portable(output/"summary.json"), "elapsed_seconds": summary["elapsed_seconds"]}
    except Exception as error:
        write_json(output/"status.json", {"status": "failed", "error": f"{type(error).__name__}: {error}", "elapsed_seconds": time.monotonic()-began})
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", default="notes/geobc_hash_bandwidth_plan.json")
    p.add_argument("--output", default="runs/geobc_exploration/hash_bandwidth_native_smoke")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cpu-check", action="store_true")
    return p


if __name__ == "__main__":
    print(json.dumps(smoke(parser().parse_args()), allow_nan=False))
