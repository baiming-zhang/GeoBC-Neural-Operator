"""Frozen TRAIN4 topology-message pilot: self_only versus neighbors.

The sole model difference is A1/A2. Only the twelve declared original-node
TRAIN cases are loaded. Evaluation uses fixed endpoints 0/1000, never a best
checkpoint. --preflight uses only the standard library, JSON, source and stat.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA256 = "05fd5a5afba10871c58d18fde2e31997b85b1c1aa695b828e4ee416c43e43e68"
CAD_ORDER = ("101_428", "101_479", "101_56", "101_78")
BASIS_ORDER = ("ver", "hor", "tor")
CASE_ORDER = tuple(f"{cad}/{load}" for cad in CAD_ORDER for load in BASIS_ORDER)
STEPS, SEED, BATCH_SIZE, CASE_SLOTS = 1000, 2026, 51200, 4
UNIFORM_NODES, CELLS_PER_CASE, NODES_PER_CASE = 6400, 1600, 12800
U_SCALE_MM = 0.05511113670847156
GRADIENT_WEIGHT = 0.0004106650618638651
STRAIN_FLOOR = 1e-24
FIXED_METADATA_SHA256 = {
    "notes/geobc_split.json": "1dd3d53bfdfebc790c5f4e8c457ca6af88c95be900f664eccabeb6697b128859",
    "notes/geobc_load_basis_45.json": "d2560829a07a20f30071f49f4bb694127b0e8d3ddb4724e3a4b0b795795c4991",
    "runs/geobc_exploration/displacement_query_context_actual_10k/config.json": "451d419512a6256f0c7dd12d8668c277319e87e8b72f9242b1c40870856c96a9",
    "runs/geobc_exploration/displacement_query_context_actual_10k/summary.json": "8603ff844c633db7d938232301a2aefb0a4d331cb4391e7a6764b43448cbc7a3",
    "data/geobc_local_geometry/manifest.json": "8784b8ba3272b82c9d0894f1fa7521c2595d3d4cee81d22295458490db22622a",
    "data/geobc_gradient_supervision/manifest.json": "052df4057186e37b129f8a44bb2c3e7279b24024a72a8b947d76239a984ac018",
    "data/geobc_graph_context/manifest.json": "91f8630b38ccb31a5165b92b31f8aaf61beb94fe420f52aa8b526d70f5c6d041",
}


def _path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", value):
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError("Topology pilot paths must stay inside this project")
    return path


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _identity(path, *, sha=False):
    path = _path(path)
    stat = path.stat()
    result = {"path": path.relative_to(ROOT).as_posix(), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if sha:
        result["sha256"] = _sha(path.read_bytes())
    return result


def _verify_stat(identity):
    path = _path(identity["path"])
    stat = path.stat()
    if stat.st_size != identity["bytes"] or stat.st_mtime_ns != identity["mtime_ns"]:
        raise ValueError(f"Frozen file stat changed: {identity['path']}")
    return path


def _scope(args):
    """Strict metadata-only contract. No NumPy/Torch/array/model imports."""
    if args.message_mode not in ("self_only", "neighbors") or args.eval_batch_size < 1 or args.device != "cuda":
        raise ValueError("Use the declared message mode, positive eval chunk and CUDA")
    plan_path = _path(args.plan)
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if _sha(plan_bytes) != PLAN_SHA256 or plan.get("status") != "predeclared":
        raise ValueError("Topology-message plan differs from its frozen predeclaration")
    scope, recipe = plan["scope"], plan["recipe"]
    if (tuple(scope["CAD_order"]) != CAD_ORDER or tuple(scope["case_order"]) != CASE_ORDER
            or scope["known_loads"] != list(BASIS_ORDER) or scope["train_cases"] != 12
            or any(scope[name] for name in ("validation_enabled", "dia_enabled", "final_test_enabled", "stress_targets_enabled"))
            or not scope["no_new_normalizer_fit"] or recipe["steps"] != STEPS or recipe["seed"] != SEED
            or recipe["batch_size"] != BATCH_SIZE or recipe["gradient_weight"] != GRADIENT_WEIGHT):
        raise ValueError("Frozen TRAIN4 twelve-case recipe changed")
    metadata = {}
    for name, expected in FIXED_METADATA_SHA256.items():
        raw = _path(name).read_bytes()
        if _sha(raw) != expected:
            raise ValueError(f"Frozen metadata changed: {name}")
        metadata[name] = json.loads(raw)
    reference = metadata["runs/geobc_exploration/displacement_query_context_actual_10k/config.json"]
    reference_summary = metadata["runs/geobc_exploration/displacement_query_context_actual_10k/summary.json"]
    split = metadata["notes/geobc_split.json"]
    if reference_summary.get("status") != "completed" or reference["groups"] != split["groups"]:
        raise ValueError("Normalizer reference must be the completed frozen TRAIN120 run")
    scaler = reference["scaler"]
    if (scaler["scale_mm"] != U_SCALE_MM or scaler["center_mm"] != 0.0
            or scaler["transform"] != "linear_zero_center_common_scalar" or len(scaler["fit_case_keys"]) != 120
            or not set(CASE_ORDER).issubset(scaler["fit_case_keys"])):
        raise ValueError("Reuse the original TRAIN120 scaler without changing its fit provenance")
    u_loss = reference["u_loss"]
    if (u_loss["name"] != "relative_case_mse" or u_loss["common_scale_mm"] != U_SCALE_MM
            or u_loss["decode_center_mm"] != 0.0 or u_loss["fit_case_keys"] != scaler["fit_case_keys"]
            or u_loss["weight_cap"] is not None or u_loss["extra_weight_normalization"] is not None):
        raise ValueError("Frozen relative-case loss identity changed")
    keys = {"train": list(CASE_ORDER)}
    source_records = {item["key"]: item for item in reference["contract"]["selected_sources"]}
    files, selected_sources, weights = {}, [], {}
    for key in CASE_ORDER:
        item = source_records[key]
        if item["group"] != "train" or key.split("/", 1)[0] not in split["groups"]["train"]:
            raise ValueError("Pilot case is outside frozen TRAIN membership")
        files[key] = _verify_stat(item)
        selected_sources.append({**_identity(files[key]), "key": key, "group": "train"})
        weight = u_loss["case_records"][key]
        rms = float(weight["truth_RMS_mm"])
        if (not math.isfinite(rms) or rms <= 1e-12 or weight["floor_active"]
                or weight["weight"] != U_SCALE_MM ** 2 / rms ** 2):
            raise ValueError(f"Invalid frozen relative-case weight: {key}")
        weights[key] = weight
    topology_dir = ROOT / "data/geobc_topology_balanced_train4"
    evidence = plan["input_evidence"]
    manifest_path = _path(evidence["cache_manifest"])
    manifest_bytes = manifest_path.read_bytes()
    if _sha(manifest_bytes) != evidence["cache_manifest_sha256"]:
        raise ValueError("Frozen balanced topology manifest changed")
    topology_manifest = json.loads(manifest_bytes)
    for identity in (topology_manifest["contract"]["plan"], topology_manifest["contract"]["split"]):
        path = _verify_stat(identity)
        if _sha(path.read_bytes()) != identity["sha256"]:
            raise ValueError("Balanced topology metadata plan/split SHA changed")
    for name, expected in topology_manifest["contract"]["source_sha256"].items():
        if _sha(_path(name).read_bytes()) != expected:
            raise ValueError(f"Balanced topology preparation source changed: {name}")
    if (_sha(_path(evidence["result"]).read_bytes()) != evidence["result_sha256"]
            or _sha(_path(evidence["preparation_source"]).read_bytes()) != evidence["preparation_source_sha256"]):
        raise ValueError("Balanced topology frozen input evidence changed")
    if (topology_manifest["status"] != "completed" or not topology_manifest["feasibility_pass"]
            or tuple(topology_manifest["contract"]["CAD_order"]) != CAD_ORDER
            or set(topology_manifest["CADs"]) != set(CAD_ORDER)
            or topology_manifest["cache_bytes"] != evidence["cache_bytes"]
            or topology_manifest["contract"]["plan"]["sha256"] != plan["input_plan_sha256"]):
        raise ValueError("Balanced input hierarchy readiness/order differs")
    gradients = metadata["data/geobc_gradient_supervision/manifest.json"]
    graph = metadata["data/geobc_graph_context/manifest.json"]
    local_records = {item["cad_id"]: item for item in reference["contract"]["selected_local_sources"]}
    selected_caches = {}
    for cad in CAD_ORDER:
        topo, grad, distance = topology_manifest["CADs"][cad], gradients["CADs"][cad], graph["CADs"][cad]
        topo_identity = evidence["CADs"][cad]["file_identity"]
        if topo["file_identity"] != topo_identity:
            raise ValueError(f"{cad}: topology evidence identity differs")
        _verify_stat(topo_identity)
        _verify_stat({"path": f"data/geobc_gradient_supervision/{grad['file']}", **{k: grad[k] for k in ("bytes", "mtime_ns")}})
        _verify_stat({"path": f"data/geobc_graph_context/{distance['file']}", **{k: distance[k] for k in ("bytes", "mtime_ns")}})
        local_path = _verify_stat(local_records[cad])
        if (topo["xyz_sha256"] != grad["xyz_sha256"] or topo["xyz_sha256"] != distance["xyz_sha256"]
                or grad["node_count"] != topo["facts"]["N0"] or distance["node_count"] != grad["node_count"]
                or evidence["CADs"][cad]["N0"] != grad["node_count"]
                or topo["labels_read"] != 0 or not topo["target_independent"] or topo["CUDA_initialized"]
                or topo["facts"]["N1"] != 2048 or topo["facts"]["N2"] != 128
                or distance["array_keys"] != ["euclidean", "graph"] or not distance["target_independent"]):
            raise ValueError(f"{cad}: input cache node order/provenance differs")
        for load in BASIS_ORDER:
            norm = float(grad["loss_normalization_by_load"][load])
            if not math.isfinite(norm) or norm <= STRAIN_FLOOR or grad["floor_active_by_load"][load]:
                raise ValueError(f"{cad}/{load}: frozen P1 denominator must have inactive floor")
            if weights[f"{cad}/{load}"]["nodes"] != grad["node_count"]:
                raise ValueError("Frozen selected displacement weight node count differs")
        for item in topo["source"].values():
            if isinstance(item, dict) and {"path", "bytes", "mtime_ns"}.issubset(item):
                _verify_stat(item)
        if len(grad["displacement_sources"]) != 3:
            raise ValueError("P1 references require exactly the three declared known loads")
        for load, item in zip(BASIS_ORDER, grad["displacement_sources"]):
            if item["key"] != f"{cad}/{load}" or _verify_stat(item) != files[item["key"]]:
                raise ValueError("P1 references must share the exact selected displacement sources")
        selected_caches[cad] = {"topology": topo, "gradient": grad, "distance": distance,
                                "local_identity": _identity(local_path),
                                "cloud_reference": reference["geometry"]["CADs"][cad]}
    source_sha = {}
    for item in plan["source_evidence"]:
        if _sha(_path(item["path"]).read_bytes()) != item["sha256"]:
            raise ValueError(f"Frozen plan source changed: {item['path']}")
        source_sha[item["path"]] = item["sha256"]
    for name, expected in reference["contract"]["code_sha256"].items():
        path = "geobc/" + name
        if _sha(_path(path).read_bytes()) != expected:
            raise ValueError(f"Frozen helper source changed: {path}")
        source_sha[path] = expected
    for path in ("geobc/topology_message_train.py", "geobc/topology_message_field.py", "geobc/hash_bandwidth_field.py"):
        source_sha[path] = _sha(_path(path).read_bytes())
    field_ready_path = ROOT / "notes/topology_message_field_ready.json"
    field_ready = _json(field_ready_path)
    ready_sha = field_ready["source_sha256"]
    if isinstance(ready_sha, dict):
        ready_sha = ready_sha["geobc/topology_message_field.py"]
    if field_ready.get("status") != "ready" or ready_sha != source_sha["geobc/topology_message_field.py"] or field_ready["plan_sha256"] != PLAN_SHA256:
        raise ValueError("Topology field must match its source-only ready record and frozen plan")
    contract = {"schema_version": 1, "stage": plan["stage"], "plan_sha256": PLAN_SHA256, "plan": _identity(plan_path, sha=True),
        "message_mode": args.message_mode, "CAD_order": list(CAD_ORDER), "case_order": list(CASE_ORDER),
        "steps": STEPS, "seed": SEED, "batch_size": BATCH_SIZE, "case_slots": CASE_SLOTS,
        "eval_endpoints": [0, STEPS], "eval_batch_size": args.eval_batch_size, "device": args.device,
        "selection": "fixed_endpoint1000; no best selection", "gradient_weight": GRADIENT_WEIGHT,
        "optimizer": {"name": "Adam", "fused": True, "lr": .003, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0},
        "source_sha256": source_sha, "metadata_sha256": FIXED_METADATA_SHA256,
        "topology_manifest": _identity(manifest_path, sha=True), "input_cache_bytes": evidence["cache_bytes"],
        "field_ready": _identity(field_ready_path, sha=True), "selected_sources": selected_sources,
        "selected_caches": selected_caches, "fixed_normalizers": {"scaler": scaler, "selected_case_records": weights,
        "source_config": "runs/geobc_exploration/displacement_query_context_actual_10k/config.json", "new_fit": False},
        "sampler": {"case_rng": "NumPy default_rng(seed+23)", "node_rng": "Torch generator(seed+17)",
        "tet_rng": "Torch generator(seed+41)", "uniform_nodes": UNIFORM_NODES, "uniform_cells": CELLS_PER_CASE,
        "corner_queries": 6400, "duplicates_retained": True, "ordered_witness_every_step": True},
        "context": {"per_step_CAD_dedup": True, "maximum_CADs": 4, "differentiable_recompute_each_step": True,
        "full_node_Q_once_per_active_CAD": True, "explicit_node_ids": True, "cross_step_reuse": False},
        "physical_P1_RMS": "sqrt(sum_node symmetric Frobenius error squared / Nnodes); shear weight2; no /6",
        "clear_gain_gate": plan["clear_gain_gate"], "validation_DIA_final_stress_enabled": False}
    return plan, keys, files, contract, metadata["notes/geobc_load_basis_45.json"], topology_manifest


def preflight(args):
    began = time.monotonic()
    _, keys, _, contract, _, _ = _scope(args)
    return {"status": "ready", "message_mode": args.message_mode, "plan_sha256": PLAN_SHA256,
            "source_sha256": contract["source_sha256"], "contract_sha256": _sha(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()),
            "train_cases": len(keys["train"]), "CAD_order": list(CAD_ORDER), "steps": STEPS,
            "array_files_opened": 0, "labels_read": 0, "models_constructed": 0, "updates": 0,
            "NumPy_imported": False, "Torch_imported": False, "CUDA_initialized": False,
            "elapsed_seconds": time.monotonic() - began}


def _initial_parameters(model):
    """Every named initial trainable tensor, plus disjoint public native groups."""
    from .hash_bandwidth_train import _decoder_initial_record, _native_groups
    import numpy as np
    decoder = _decoder_initial_record(model, "coarse8")
    digest, named = hashlib.sha256(), {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.detach().cpu().contiguous().numpy()
        raw = value.tobytes()
        named[name] = {"shape": list(value.shape), "dtype": str(value.dtype), "count": parameter.numel(), "sha256": _sha(raw)}
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(raw)
    groups = _native_groups(model)
    counts = {"encoding": groups["encoding"][0].numel(), "network": groups["network"][0].numel(),
              "native_total": sum(p.numel() for values in groups.values() for p in values),
              "PointNet": sum(p.numel() for p in model.encoder.parameters()),
              "topology": sum(p.numel() for p in model.topology_encoder.parameters()),
              "total": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if counts != {"encoding": 7733632, "network": 36864, "native_total": 7770496, "PointNet": 16800, "topology": 9856, "total": 7797152}:
        raise ValueError(f"Actual public model counts differ from the declared single architecture: {counts}")
    if hasattr(model, "query_projection"):
        raise ValueError("Topology field must not retain an old Gaussian query projection")
    return {"all_parameter_sha256": digest.hexdigest(), "named_trainable_tensors": named,
            "named_parameter_sha256": {name: item["sha256"] for name, item in named.items()},
            "native_public_parameter_sha256": {k: v[0]["sha256"] for k, v in decoder["native_groups"].items()},
            "decoder": decoder, "counts": counts, "paired_modes": ["self_only", "neighbors"],
            "recorded_before_optimizer_or_resume": True}


def _first_backward_check(model, context):
    import torch
    from .hash_bandwidth_train import _bandwidth_gradient
    result = {"passed": True, "checked_after_backward_before_optimizer_step": True, "groups": {}}
    groups = {"PointNet": model.encoder, **{name: getattr(model.topology_encoder, name)
              for name in ("embed", "coarse0", "coarse1", "fine0", "fine1")}}
    for name, module in groups.items():
        parameters = list(module.parameters())
        if not parameters or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise FloatingPointError(f"Missing/nonfinite first-backward gradients: {name}")
        l1 = sum(float(p.grad.detach().double().abs().sum()) for p in parameters)
        if not math.isfinite(l1) or l1 <= 0:
            raise FloatingPointError(f"First-backward {name} gradient must be finite and nonzero")
        result["groups"][name] = {"gradient_l1": l1, "parameter_count": sum(p.numel() for p in parameters),
                                  "gradients_present_and_finite": True}
    for name, tensors in (("global_context", (context.global_tensor,)), ("fine_tokens", context.fine_tokens),
                          ("full_node_Q", context.node_contexts)):
        values = []
        for tensor in tensors:
            if tensor.grad is None or not bool(torch.isfinite(tensor.grad).all()):
                raise FloatingPointError(f"Missing/nonfinite retained context gradients: {name}")
            values.append(float(tensor.grad.detach().double().abs().sum()))
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise FloatingPointError(f"Every active CAD must have nonzero {name} gradients")
        result[name] = {"gradient_l1_by_tensor": values, "present_finite_nonzero": True}
    result.update(_bandwidth_gradient(model, "coarse8"))
    return result


def _evaluate_train(model, cases, context, scaler, batch_size, gradient_data, weights):
    """One native basis9 pass per CAD; same signed u for allnode P1 recovery."""
    import numpy as np
    import torch
    from .physics_train import displacement_metrics
    from .gradient_supervision import cell_tensor_strain, recover_node_tensor_strain
    per, strain_per, u_losses, strain_losses = {}, {}, [], []
    grouped = {cad: [case for case in cases if case.key.split("/", 1)[0] == cad] for cad in CAD_ORDER}
    with torch.no_grad():
        for row, (cad, cad_cases) in enumerate(grouped.items()):
            first = cad_cases[0]
            basis = np.empty((first.count, 3, 3), np.float32)
            for start in range(0, first.count, batch_size):
                stop = min(start + batch_size, first.count)
                ids = torch.arange(start, stop, dtype=torch.long, device=first.xyz.device)
                cad_ids = torch.full_like(ids, row)
                basis[start:stop] = model.response_basis(first.xyz[start:stop], first.features[start:stop],
                    node_indices=ids, cad_index=cad_ids, context=context).float().cpu().numpy()
            item = gradient_data[cad]; tensors, cache = item["tensors"], item["cache"]
            for case in cad_cases:
                load = case.key.split("/", 1)[1]; load_index = BASIS_ORDER.index(load)
                alpha = case.alpha.cpu().numpy()
                if not np.array_equal(alpha, np.eye(3, dtype=np.float32)[load_index]):
                    raise ValueError("Known-load evaluation requires the frozen unit coefficients")
                prediction = scaler.decode(np.einsum("l,nlc->nc", alpha, basis))
                per[case.key] = displacement_metrics(case.physical_u, prediction)
                error = prediction - case.physical_u
                u_loss = float(np.mean((error / scaler.scale_mm) ** 2) * weights[case.key]["weight"])
                physical = torch.as_tensor(prediction, dtype=torch.float64, device=first.xyz.device)
                count = len(tensors["corners"])
                cell_prediction = torch.empty((count, 6), dtype=torch.float64, device=first.xyz.device)
                for start in range(0, count, batch_size):
                    stop = min(start + batch_size, count)
                    cell_prediction[start:stop] = cell_tensor_strain(physical[tensors["corners"][start:stop]],
                        tensors["inverse_physical_edge_mm"][start:stop])
                target = tensors["reference_cell_tensor_strain"][:, load_index].double()
                frob_weights = torch.as_tensor([1, 1, 1, 2, 2, 2], dtype=torch.float64, device=physical.device)
                raw_strain_loss = float(((cell_prediction - target).square() * frob_weights).sum(-1).mul(tensors["volumes"]).sum()
                                        / tensors["volumes"].sum() / cache.loss_normalization_by_load[load])
                recovered = recover_node_tensor_strain(cell_prediction, tensors["volumes"], tensors["cells"], case.count, batch_size=batch_size)
                reference = recover_node_tensor_strain(target, tensors["volumes"], tensors["cells"], case.count, batch_size=batch_size)
                energy = float(((recovered - reference).square() * frob_weights).sum())
                truth_energy = float((reference.square() * frob_weights).sum())
                strain_per[case.key] = {"nodes": case.count, "relative_l2": math.sqrt(energy / max(truth_energy, STRAIN_FLOOR)),
                    "symmetric_Frobenius_relative_l2": math.sqrt(energy / max(truth_energy, STRAIN_FLOOR)),
                    "error_energy": energy, "truth_energy": truth_energy, "physical_tensor_error_RMS": math.sqrt(energy / case.count),
                    "physical_RMS_definition": "sqrt(Frobenius error energy / Nnodes); shear weight2; no /6",
                    "raw_relative_cell_strain_loss": raw_strain_loss}
                per[case.key]["relative_case_u_loss"] = u_loss
                u_losses.append(u_loss); strain_losses.append(raw_strain_loss)
                del physical, cell_prediction, recovered, reference, target
            del basis
    u_worst = max(per, key=lambda k: per[k]["relative_l2"])
    s_worst = max(strain_per, key=lambda k: strain_per[k]["relative_l2"])
    energy = sum(v["error_energy"] for v in strain_per.values())
    truth_energy = sum(v["truth_energy"] for v in strain_per.values())
    node_count = sum(v["nodes"] for v in strain_per.values())
    train = {"per_case": per, "case_mean_relative_l2": float(np.mean([v["relative_l2"] for v in per.values()])),
        "worst_case_key": u_worst, "worst_case_relative_l2": per[u_worst]["relative_l2"],
        "case_mean_physical_u_RMS_mm": float(np.mean([v["rmse_mm"] for v in per.values()])),
        "strain": {"per_case": strain_per, "case_mean_relative_l2": float(np.mean([v["relative_l2"] for v in strain_per.values()])),
        "worst_case_key": s_worst, "worst_case_relative_l2": strain_per[s_worst]["relative_l2"],
        "mean_case_physical_tensor_error_RMS": float(np.mean([v["physical_tensor_error_RMS"] for v in strain_per.values()])),
        "pooled": {"relative_l2": math.sqrt(energy / max(truth_energy, STRAIN_FLOOR)), "error_energy": energy,
        "truth_energy": truth_energy, "physical_tensor_error_RMS": math.sqrt(energy / node_count)},
        "selection_used": False, "scope": "all original nodes; incident-volume-recovered corner-P1 tensor proxy"}}
    lu, ls = float(np.mean(u_losses)), float(np.mean(strain_losses))
    return train, {"relative_case_u_loss": lu, "unweighted_relative_cell_strain_loss": ls,
                   "gradient_weight": GRADIENT_WEIGHT, "weighted_strain_loss": GRADIENT_WEIGHT * ls,
                   "total": lu + GRADIENT_WEIGHT * ls, "scope": "exact full-node u and full-volume cell P1 case means"}


def run(args):
    if args.preflight:
        record = preflight(args)
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
        return record
    began = time.monotonic()
    plan, keys, files, contract, guard, topology_manifest = _scope(args)
    output = _path(args.output)
    resume = output / "last.pt" if args.resume == "last" else _path(args.resume) if args.resume else None
    if resume is not None and resume != (output / "last.pt").resolve():
        raise ValueError("Resume must use this output's last.pt")
    if (output / "summary.json").is_file():
        summary = _json(output / "summary.json")
        if summary.get("status") == "completed":
            if summary["config"]["contract"] != contract or summary["step"] != STEPS:
                raise ValueError("Completed topology pilot differs from the strict contract")
            for name in ("config.json", "last.pt", "endpoint_0000.pt", "endpoint_1000.pt", "metrics_step0000.json",
                         "metrics_step1000.json", "sampling_first_batch.json", "sampler_witnesses.jsonl", "first_backward_check.json"):
                if not (output / name).is_file():
                    raise ValueError(f"Completed pilot lacks {name}")
            if (_json(output / "config.json") != summary["config"] or _json(output / "metrics_step1000.json") != summary["final_metrics"]
                    or _json(output / "metrics_step0000.json") != summary["initial_metrics"]):
                raise ValueError("Completed pilot configuration/endpoint metrics differs")
            print(f"Reusing completed fixed-endpoint pilot {output}", flush=True)
            return summary
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError("Incomplete nonempty output requires --resume")
    stored_config = _json(output / "config.json") if resume is not None else None
    if resume is not None and (not resume.is_file() or stored_config["contract"] != contract):
        raise ValueError("Resume checkpoint missing or strict source/cache/scaler/recipe contract changed")
    if resume is not None and any(not (output / name).is_file() for name in ("endpoint_0000.pt", "metrics_step0000.json")):
        raise ValueError("Resume must retain the committed fixed initial endpoint")
    output.mkdir(parents=True, exist_ok=True)
    prior_elapsed, start_step = 0.0, 0
    status = {"phase": "loading", "message_mode": args.message_mode, "step": 0, "total_steps": STEPS,
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    def elapsed():
        return prior_elapsed + time.monotonic() - began
    def update(phase, **extra):
        status.update(extra, phase=phase, elapsed_seconds=elapsed(), updated_at_utc=datetime.now(timezone.utc).isoformat())
        progress = status["step"] - start_step
        eta = (time.monotonic() - began) / progress * (STEPS - status["step"]) if progress > 0 else None
        if phase == "completed": eta = 0.0
        status["eta_seconds"] = eta
        status["estimated_completion_utc"] = (datetime.now(timezone.utc) + timedelta(seconds=eta)).isoformat() if eta is not None else None
        _write_json(output / "status.json", status)
    update("loading")
    try:
        import numpy as np
        import torch
        from .physics_train import DisplacementScaler, load_displacement_cases, relative_case_mse
        from .bc_role_condition import load_bc_role_cloud
        from .local_geometry import load_local_geometry
        from .gradient_supervision import load_gradient_manifest, load_gradient_cache
        from .hash_bandwidth_train import mixed_query_indices, gradient_batch_loss, _memory_record
        from .conditional_train import _save_checkpoint
        from .topology_message_field import TopologyMessageField, TopologyMessageFieldConfig, load_topology_input
        device = torch.device(args.device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Full reset is repeated immediately before model construction, so input
        # loading or future metadata helpers cannot alter matched initialization.
        def reset_seed():
            random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        reset_seed()
        cases = load_displacement_cases(keys["train"], files, guard, device)
        scaler = DisplacementScaler.from_record(contract["fixed_normalizers"]["scaler"])
        weight_records = contract["fixed_normalizers"]["selected_case_records"]
        loss_weights = torch.as_tensor([weight_records[case.key]["weight"] for case in cases], dtype=torch.float32, device=device)
        if not bool(torch.isfinite(loss_weights).all()):
            raise ValueError("Frozen weights are not finite in training precision")
        by_cad = {cad: [case for case in cases if case.key.split("/", 1)[0] == cad] for cad in CAD_ORDER}
        gradient_manifest = load_gradient_manifest(ROOT / "data/geobc_gradient_supervision")
        gradient_data, topology_inputs, clouds, input_records = {}, [], [], {}
        for cad in CAD_ORDER:
            cad_cases = by_cad[cad]; first = cad_cases[0]
            selected = contract["selected_caches"][cad]
            local = load_local_geometry(ROOT / "data/geobc_local_geometry", cad, xyz=first.xyz)
            local_tensor = local.tensor(mode="actual", device=device)
            distance_record = selected["distance"]
            # Read only the active Euclidean input member for this declared CAD.
            with np.load(ROOT / "data/geobc_graph_context" / distance_record["file"], allow_pickle=False) as saved:
                distance = saved["euclidean"]
            if (distance.dtype != np.float32 or distance.shape != (first.count, 5) or not np.isfinite(distance).all()
                    or np.any(distance < 0) or _sha(distance.tobytes()) != distance_record["euclidean_features_sha256"]):
                raise ValueError(f"{cad}: active Euclidean5 cache array differs")
            distance_tensor = torch.as_tensor(distance, device=device)
            for case in cad_cases:
                local.verify_xyz(case.xyz)
                case.features = torch.cat((case.features, local_tensor, distance_tensor), dim=-1)
                # Loads are represented solely by alpha after the frozen guard
                # checked the original force/moment slots against that alpha.
                case.features = case.features.clone(); case.features[:, 7:13] = 0
                case.target = torch.as_tensor(scaler.encode(case.physical_u), device=device)
                if not torch.equal(case.features, first.features):
                    raise ValueError("Known-load masked F43 features must agree exactly")
            topo = load_topology_input(ROOT / "data/geobc_topology_balanced_train4", cad,
                xyz=first.xyz, features43=first.features, manifest=topology_manifest, device=device)
            topology_inputs.append(topo)
            cache = load_gradient_cache(ROOT / "data/geobc_gradient_supervision", cad, xyz=first.xyz, manifest=gradient_manifest)
            if cache.loss_normalization_by_load != selected["gradient"]["loss_normalization_by_load"]:
                raise ValueError("Runtime P1 denominators differ from frozen metadata")
            gradient_data[cad] = {"cache": cache, "tensors": cache.tensor(device=device)}
            cloud = load_bc_role_cloud(files[f"{cad}/ver"], cloud_points=1024, seed=SEED, cad_key=cad)
            cloud_record = cloud.record(); ref_cloud = selected["cloud_reference"]
            for name in ("points5_sha256", "coordinates_sha256", "roles_sha256", "mask_sha256", "node_indices_sha256"):
                if cloud_record[name] != ref_cloud[name]:
                    raise ValueError(f"{cad}: frozen global-cloud identity differs: {name}")
            clouds.append(cloud)
            input_records[cad] = {"local": local.record(), "cloud": cloud_record, "topology": topo.metadata,
                                  "masked_features43_sha256": _sha(first.features.detach().cpu().contiguous().numpy().tobytes())}
        cloud_array = np.stack([c.points for c in clouds]); mask_array = np.stack([c.mask for c in clouds])
        geometry_clouds, geometry_masks = torch.as_tensor(cloud_array, device=device), torch.as_tensor(mask_array, device=device)
        reset_seed()
        model = TopologyMessageField(TopologyMessageFieldConfig(message_mode=args.message_mode, seed=SEED), device)
        initial = _initial_parameters(model)
        native_layout = model.validate_native_layout()
        optimizer = torch.optim.Adam(model.parameters(), lr=.003, betas=(.9, .999), eps=1e-8, weight_decay=0, fused=True)
        case_rng = np.random.default_rng(SEED + 23)
        node_rng = torch.Generator(device=device).manual_seed(SEED + 17)
        tet_rng = torch.Generator(device=device).manual_seed(SEED + 41)
        config = {"contract": contract, "method": "GeoBC fixed TRAIN4 topology-message pilot", "message_mode": args.message_mode,
            "scope": "4TRAIN CAD x ver/hor/tor original nodes only; fitting diagnostics", "scaler": scaler.record(),
            "u_loss": {"name": "relative_case_mse", "common_scale_mm": U_SCALE_MM, "case_records": weight_records,
                       "refit": False, "original_fit_case_keys": list(scaler.fit_case_keys)},
            "model": model.source_config, "initial_parameters": initial, "actual_parameter_counts": initial["counts"],
            "native_layout": native_layout, "input_records": input_records,
            "cases": {"train": [{"sample_key": c.key, "source": _identity(c.source), "nodes": c.count,
                                 "alpha": c.alpha.cpu().tolist()} for c in cases]},
            "optimizer": contract["optimizer"], "total_steps": STEPS, "endpoint_selection": "fixed1000",
            "precision": "FP32 high/TF32 context; unchanged split-native arithmetic; FP64 physical u/P1 metrics",
            "torch_version": torch.__version__, "started_at_utc": datetime.now(timezone.utc).isoformat()}
        gradient_check, latest_loss, last_metrics, witnesses, step_timings = None, None, None, [], []
        if resume is not None:
            for name in ("contract", "scaler", "u_loss", "model", "initial_parameters", "actual_parameter_counts", "native_layout", "input_records", "cases", "precision", "optimizer", "torch_version"):
                if config[name] != stored_config[name]:
                    raise ValueError(f"Resume {name} changed")
            saved = torch.load(resume, map_location=device, weights_only=False)
            if saved["config"] != stored_config: raise ValueError("last.pt/config.json differs")
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
            start_step = int(saved["step"])
            if not 0 <= start_step <= STEPS: raise ValueError("Resume step outside fixed budget")
            case_rng.bit_generator.state = saved["case_rng_state"]
            node_rng.set_state(saved["node_rng_state"].cpu()); tet_rng.set_state(saved["tet_rng_state"].cpu())
            torch.set_rng_state(saved["torch_rng_state"].cpu())
            torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng_state"]])
            np.random.set_state(saved["numpy_rng_state"]); random.setstate(saved["python_rng_state"])
            prior_elapsed = float(saved["elapsed_seconds"])
            gradient_check, latest_loss, last_metrics = saved["first_backward_check"], saved["training_loss_components"], saved["metrics"]
            witnesses, step_timings = saved["sampler_witnesses"], saved["step_timings"]
            if start_step > 0 and (gradient_check is None or not gradient_check.get("passed")):
                raise ValueError("Resume after an update requires the committed passed first-backward check")
            if len(witnesses) != start_step or [w["step"] for w in witnesses] != list(range(1, start_step + 1)):
                raise ValueError("Committed ordered sampler witnesses differ from resume step")
            config = stored_config
            (output / "sampler_witnesses.jsonl").write_text("".join(json.dumps(w, ensure_ascii=False) + "\n" for w in witnesses), encoding="utf-8")
        else:
            _write_json(output / "config.json", config)
        def checkpoint_payload(step, metrics):
            return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config, "step": step,
                "metrics": metrics, "elapsed_seconds": elapsed(), "first_backward_check": gradient_check,
                "training_loss_components": latest_loss, "sampler_witnesses": witnesses, "step_timings": step_timings,
                "case_rng_state": case_rng.bit_generator.state, "node_rng_state": node_rng.get_state(), "tet_rng_state": tet_rng.get_state(),
                "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate()}
        def evaluate(step):
            eval_began = time.monotonic(); model.eval(); update("evaluating", step=step)
            with torch.no_grad():
                torch.cuda.synchronize(device); context_began = time.monotonic()
                context = model.encode_context(geometry_clouds, geometry_masks, topology_inputs, record_timing=True)
                torch.cuda.synchronize(device); context_seconds = time.monotonic() - context_began
                train, objective = _evaluate_train(model, cases, context, scaler, args.eval_batch_size, gradient_data, weight_records)
            metrics = {"step": step, "message_mode": args.message_mode, "train": train, "selection_used": False,
                "training_loss_components": latest_loss, "evaluation_loss_components": objective,
                "elapsed_seconds": elapsed(), "eval_seconds": time.monotonic() - eval_began, "context_seconds": context_seconds,
                "context_phase_timings": context.timings,
                "first_backward_check": gradient_check, "memory": _memory_record(device)}
            payload = checkpoint_payload(step, metrics)
            _save_checkpoint(payload, output / "last.pt")
            _save_checkpoint(payload, output / f"endpoint_{step:04d}.pt")
            _write_json(output / f"metrics_step{step:04d}.json", metrics)
            _write_json(output / "metrics_latest.json", metrics)
            update("training", step=step, latest_train_case_mean_l2=train["case_mean_relative_l2"])
            print(f"{args.message_mode} {step}/{STEPS} u_train={100*train['case_mean_relative_l2']:.4f}% P1_train={100*train['strain']['case_mean_relative_l2']:.4f}%", flush=True)
            return metrics
        update("training", step=start_step, resume_step=start_step)
        if resume is None:
            last_metrics = evaluate(0)
        cad_to_row = {cad: i for i, cad in enumerate(CAD_ORDER)}
        for step in range(start_step + 1, STEPS + 1):
            step_began = time.monotonic()
            phase_events = {name: torch.cuda.Event(enable_timing=True) for name in
                            ("context_start", "context_end", "query_start", "query_end", "backward_start", "backward_end", "optimizer_start", "optimizer_end")}
            model.train(); optimizer.zero_grad(set_to_none=True)
            selected = case_rng.choice(len(cases), CASE_SLOTS, replace=False)
            batch_cases = [cases[int(i)] for i in selected]
            active_cads = list(dict.fromkeys(c.key.split("/", 1)[0] for c in batch_cases))
            rows = torch.as_tensor([cad_to_row[cad] for cad in active_cads], dtype=torch.long, device=device)
            phase_events["context_start"].record()
            context = model.encode_context(geometry_clouds[rows], geometry_masks[rows], [topology_inputs[cad_to_row[cad]] for cad in active_cads], record_timing=(step == 1))
            phase_events["context_end"].record()
            check_gradient = gradient_check is None
            if check_gradient: context.retain_grad()
            local_rows = {cad: i for i, cad in enumerate(active_cads)}
            xyz, features, alpha, targets, node_ids, cad_ids, sampled_cells, witness_parts = [], [], [], [], [], [], [], []
            witness_parts.append(selected.astype("<i8").tobytes())
            witness_slots = []
            for case in batch_cases:
                cad = case.key.split("/", 1)[0]
                nodes, uniform, cells, corners = mixed_query_indices(case, gradient_data[cad]["tensors"], node_rng, tet_rng)
                sampled_cells.append(cells)
                xyz.append(case.xyz[nodes]); features.append(case.features[nodes]); targets.append(case.target[nodes])
                alpha.append(case.alpha.expand(NODES_PER_CASE, -1)); node_ids.append(nodes)
                cad_ids.append(torch.full((NODES_PER_CASE,), local_rows[cad], dtype=torch.long, device=device))
                slot = {"case_key": case.key}
                for name, value in (("uniform_node_ids", uniform), ("original_cell_ids", cells), ("ordered_corner_ids", corners)):
                    raw = value.detach().cpu().numpy().astype("<i8", copy=False).tobytes()
                    witness_parts.append(raw); slot[name + "_sha256"] = _sha(raw)
                witness_slots.append(slot)
            phase_events["query_start"].record(); model.decoder.capture_activation_gradients = check_gradient
            prediction = model(torch.cat(xyz), torch.cat(features), torch.cat(alpha), node_indices=torch.cat(node_ids), cad_index=torch.cat(cad_ids), context=context)
            phase_events["query_end"].record()
            lu = relative_case_mse(prediction, torch.cat(targets), loss_weights[torch.as_tensor(selected, dtype=torch.long, device=device)])
            ls = gradient_batch_loss(prediction, batch_cases, sampled_cells, gradient_data, U_SCALE_MM, GRADIENT_WEIGHT)
            loss = lu + GRADIENT_WEIGHT * ls
            if not bool(torch.isfinite(loss)): raise FloatingPointError(f"Nonfinite loss at step {step}")
            phase_events["backward_start"].record(); loss.backward(); phase_events["backward_end"].record()
            if check_gradient:
                gradient_check = {"step": step, "message_mode": args.message_mode, **_first_backward_check(model, context)}
                _write_json(output / "first_backward_check.json", gradient_check)
                model.decoder.capture_activation_gradients = False
            phase_events["optimizer_start"].record(); optimizer.step(); phase_events["optimizer_end"].record()
            torch.cuda.synchronize(device)
            context_seconds = phase_events["context_start"].elapsed_time(phase_events["context_end"]) / 1000
            query_decoder_seconds = phase_events["query_start"].elapsed_time(phase_events["query_end"]) / 1000
            backward_seconds = phase_events["backward_start"].elapsed_time(phase_events["backward_end"]) / 1000
            optimizer_seconds = phase_events["optimizer_start"].elapsed_time(phase_events["optimizer_end"]) / 1000
            latest_loss = {"relative_case_u_loss": float(lu.detach()), "unweighted_relative_cell_strain_loss": float(ls.detach()),
                           "gradient_weight": GRADIENT_WEIGHT, "weighted_strain_loss": float(ls.detach()) * GRADIENT_WEIGHT,
                           "total": float(loss.detach())}
            witness = {"step": step, "case_indices": selected.tolist(), "case_keys": [c.key for c in batch_cases],
                "ordered_slots": witness_slots, "witness_sha256": _sha(b"".join(witness_parts)),
                "active_CAD_order": active_cads, "uniform_nodes_per_case": UNIFORM_NODES, "uniform_cells_per_case": CELLS_PER_CASE,
                "corner_queries_per_case": 6400, "queries_per_case": NODES_PER_CASE, "duplicates_retained": True,
                "witness_includes": "ordered case IDs plus all uniform node IDs/cell IDs/ordered corner IDs"}
            witnesses.append(witness)
            with (output / "sampler_witnesses.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(witness, ensure_ascii=False) + "\n")
            if step == 1: _write_json(output / "sampling_first_batch.json", witness)
            step_timings.append({"step": step, "context_seconds": context_seconds, "query_and_native_decoder_seconds": query_decoder_seconds,
                "context_phase_timings": context.timings,
                "backward_seconds": backward_seconds, "optimizer_seconds": optimizer_seconds,
                "full_step_seconds": time.monotonic() - step_began, "active_CADs": len(active_cads)})
            # No trainable context is detached/cached or reused across updates.
            del context, prediction, loss, lu, ls
            if step % 100 == 0:
                update("training", step=step, training_loss_components=latest_loss, memory=_memory_record(device), last_step_timing=step_timings[-1])
                _save_checkpoint(checkpoint_payload(step, last_metrics), output / "last.pt")
            if step == STEPS:
                last_metrics = evaluate(STEPS)
        if start_step == STEPS and (output / "metrics_step1000.json").is_file():
            last_metrics = _json(output / "metrics_step1000.json")
        elif start_step == STEPS:
            last_metrics = evaluate(STEPS)
        _write_json(output / "step_timings.json", {"steps": step_timings, "timing_method": "CUDA events for phase times; one end-of-step synchronize; full step wall clock; query includes native decoder"})
        summary = {"status": "completed", "step": STEPS, "message_mode": args.message_mode, "fixed_endpoint": STEPS, "endpoint_step": STEPS,
            "selection_used": False, "elapsed_seconds": elapsed(), "final_metrics": last_metrics,
            "initial_metrics": _json(output / "metrics_step0000.json"), "config": config, "memory": _memory_record(device),
            "first_backward_check": gradient_check, "sampler_witness_steps": len(witnesses),
            "sampling_witnesses": witnesses,
            "sampler_witness_sha256": _sha(json.dumps(witnesses, sort_keys=True, separators=(",", ":")).encode()),
            "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        _write_json(output / "summary.json", summary); update("completed", step=STEPS)
        return summary
    except Exception as error:
        update("failed", error=f"{type(error).__name__}: {error}")
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, default=Path("notes/geobc_topology_message_plan.json"))
    result.add_argument("--message-mode", required=True, choices=("self_only", "neighbors"))
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", choices=("cuda",), default="cuda")
    result.add_argument("--eval-batch-size", type=int, default=65536)
    result.add_argument("--resume", nargs="?", const="last", default=None)
    result.add_argument("--preflight", action="store_true", help="JSON/source/stat checks only; no arrays or GPU")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
