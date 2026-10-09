"""Frozen-route final all-node evaluation; labels are opened after prediction.

This entry point performs no training, scaler fitting, cache preparation,
continuous-query diagnostic, or final-data model/checkpoint selection.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

from . import solver_selection_train as screen

ROLES = ("endpoint_primary", "validation_best_secondary")
SEEDS = (2026, 2027)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_identities():
    paths = ("geobc/solver_final_eval.py", "geobc/solver_final_eval_data.py",
             "geobc/solver_confirmation_train.py")
    return {**screen._source_identities(), **{path: screen.source_sha(path) for path in paths}}


def _rules(plan, require_frozen=False):
    if plan.get("known_loads") != ["ver", "hor", "tor"] or plan.get("combination_loads") != ["dia"]:
        raise ValueError("Final evaluation is final5 known3 and separately final5/dia")
    routes = plan.get("routes", [])
    if require_frozen and (plan.get("final_rules_frozen") is not True or not routes):
        raise ValueError("Actual final evaluation requires nonempty frozen final routes")
    identities = []
    for route in routes:
        method = screen.canonical_method(route["method"])
        if route["seed"] not in SEEDS or route["checkpoint_role"] not in ROLES:
            raise ValueError("Final routes require seed2026/2027 and a declared primary/secondary role")
        identities.append((method, route["seed"], route["checkpoint_role"]))
    methods = {row[0] for row in identities}
    if len(methods) > 2 or len(routes) > 8 or len(set(identities)) != len(identities):
        raise ValueError("Final routes are bounded by two methods x two seeds x two checkpoint roles")
    if require_frozen and set(identities) != {(method, seed, role) for method in methods for seed in SEEDS for role in ROLES}:
        raise ValueError("Every frozen method must retain both seeds and both declared checkpoint roles")
    return routes


def _origin(value):
    value = dict(value)
    if "checkpoint" in value:
        value["checkpoint"] = str(screen.host_path(value["checkpoint"]))
    return value


def _compatible_config(previous, expected):
    # Historical screen bests retain their original config and acceptance SHA.
    previous_other = {k: v for k, v in previous.items() if k not in ("contract", "report_phase")}
    expected_other = {k: v for k, v in expected.items() if k not in ("contract", "report_phase")}
    return previous_other == expected_other and screen.contracts_compatible(previous["contract"], expected["contract"])


def _route_metadata(route, require_frozen=False):
    method = screen.canonical_method(route["method"]); seed = route["seed"]
    role = route["checkpoint_role"]; run = screen.host_path(route["confirmation_run"])
    summary = screen.read_json(run / "summary.json"); config = screen.read_json(run / "config.json")
    if summary.get("status") != "completed" or summary["config"] != config:
        raise ValueError("Selected confirmation must have matching completed summary/config")
    if summary["method"] != method or summary["seed"] != seed:
        raise ValueError("Selected confirmation method/seed differs from the frozen route")
    primary = role == "endpoint_primary"
    selected = summary["final_metrics"] if primary else summary["best_metrics"]
    step = summary["endpoint_step"] if primary else summary["best_step"]
    checkpoint = screen.host_path(route["checkpoint"])
    if checkpoint != (run / ("last.pt" if primary else "best.pt")).resolve() or selected["step"] != step:
        raise ValueError("Frozen checkpoint role/path/selected endpoint disagree")
    origin = {"phase": "confirmation", "checkpoint": str(checkpoint), "step": step} if primary else summary["best_checkpoint_origin"]
    expected_config = config
    if not primary and origin["phase"] == "screen":
        initialization = config["contract"]["initialization"]
        if initialization["mode"] != "warmstart" or seed != 2026:
            raise ValueError("A historical screen best requires the exact seed2026 warmstart lineage")
        expected_config = initialization["parent_config"]
        if screen.host_path(origin["checkpoint"]) != screen.host_path(initialization["checkpoint"]).parent / "best.pt":
            raise ValueError("Historical best origin differs from the exact screen parent")
    elif origin["phase"] != "confirmation":
        raise ValueError("Unknown checkpoint origin")
    stat = checkpoint.stat(); checkpoint_stat = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if require_frozen and any(key not in route for key in ("checkpoint_stat", "selected_step", "budget_origin")):
        raise ValueError("Frozen route must declare checkpoint stat, selected step and budget origin")
    if route.get("checkpoint_stat", checkpoint_stat) != checkpoint_stat or route.get("selected_step", step) != step:
        raise ValueError("Frozen checkpoint stat/selected step changed")
    if _origin(route.get("budget_origin", origin)) != _origin(origin):
        raise ValueError("Frozen checkpoint budget origin differs from the selected confirmation record")
    if "budget" in route and route["budget"] != summary["budget"]:
        raise ValueError("Frozen cumulative confirmation selection budget changed")
    for key, actual in (("expected_model_source_config", config["model"]), ("parameter_counts", config["parameter_counts"])):
        if key in route and route[key] != actual:
            raise ValueError(f"Frozen route {key} changed")
    metadata = {"route_id": f"{method}_seed{seed}_{role}", "method": method, "seed": seed,
                "checkpoint_role": role, "confirmation_run": str(run), "checkpoint": str(checkpoint),
                "checkpoint_stat": checkpoint_stat, "selected_step": step, "checkpoint_trained": step > 0,
                "budget_origin": _origin(origin), "confirmation_budget": summary["budget"],
                "selected_checkpoint_costs": selected["costs"], "selected_validation_score": selected["selection_score"],
                "expected_model_source_config": config["model"], "parameter_counts": config["parameter_counts"],
                "confirmation_config_sha256": _digest(config), "expected_checkpoint_config_sha256": _digest(expected_config),
                "evaluation_id": _digest([str(run), method, seed, step])[:20]}
    return metadata, expected_config


def request_contract(plan_path):
    """Pure metadata identity; preparation and prediction are separate releases."""
    from .solver_final_eval_data import request_contract as input_contract
    data_contract, plan = input_contract(plan_path)
    routes = _rules(plan)
    resolved = [_route_metadata(route, require_frozen=plan.get("final_rules_frozen") is True)[0] for route in routes]
    return {"schema": "geobc_solver_final_eval_v1", "plan": str(screen.host_path(plan_path)),
            "input_contract": data_contract, "routes": resolved, "source_sha256": _source_identities(),
            "rules": {"known_loads": plan["known_loads"], "combination_loads": plan["combination_loads"],
                      "max_methods": 2, "seeds": list(SEEDS), "checkpoint_roles": list(ROLES),
                      "max_case_predictions": 160, "label_access": "after full encoded prediction and physical decode",
                      "final_selection_performed": False}}, plan


def source_check(plan_path):
    """Pending plans are allowed; no scientific dependency, array or weight load."""
    import ast
    plan = screen.read_json(plan_path); routes = _rules(plan)
    for path in ("geobc/solver_final_eval.py", "geobc/solver_final_eval_data.py"):
        ast.parse(screen.host_path(path).read_text(encoding="utf-8"), filename=path)
    return {"status": "source_ready", "final_rules_frozen": plan.get("final_rules_frozen") is True,
            "routes": len(routes), "source_sha256": _source_identities(),
            "scientific_arrays_read": 0, "checkpoint_weights_read": 0,
            "models_constructed": 0, "CUDA_initialized": False}


def _group(np, per_case):
    result = screen._aggregate(np, per_case)
    result.update(worst_case_u_key=max(per_case, key=lambda key: per_case[key]["u_relative_l2"]),
                  worst_case_stress_key=max(per_case, key=lambda key: per_case[key]["stress_relative_l2"]),
                  worst_case_stress_hotspot_relative_l2=max(item["stress_hotspot_relative_l2"] for item in per_case.values()),
                  worst_case_stress_hotspot_key=max(per_case, key=lambda key: per_case[key]["stress_hotspot_relative_l2"]))
    for name in ("u_vector_error_RMS_mm", "u_max_vector_error_mm", "stress_RMSE_MPa", "stress_MAE_MPa"):
        result["case_mean_" + name] = float(np.mean([item[name] for item in per_case.values()]))
    return result


def _labels_after_prediction(np, record, prediction):
    if prediction.shape != (record["num_nodes"], 4) or not np.isfinite(prediction).all():
        raise FloatingPointError("Complete finite physical prediction is required before opening labels")
    source_path = screen.host_path(record["source_npz_path"]); stat = source_path.stat()
    if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != record["source_stat"]:
        raise ValueError("Original aligned label source stat changed after input preparation")
    with np.load(source_path, allow_pickle=False) as source:
        displacement = np.asarray(source["displacement"], dtype=np.float32)
        stress = np.asarray(source["stress"], dtype=np.float32)
    if displacement.shape != (len(prediction), 3) or stress.shape not in ((len(prediction),), (len(prediction), 1)):
        raise ValueError("Final signed displacement/stress label shapes differ from predicted node order")
    return np.column_stack((displacement, stress)).astype(np.float32)


def _seed_summary(routes):
    output = []
    for method, role in sorted({(row["method"], row["checkpoint_role"]) for row in routes}):
        members = [row for row in routes if row["method"] == method and row["checkpoint_role"] == role]
        groups = {}
        for group in ("known", "combination"):
            values = {}
            for field in ("case_mean_u_relative_l2", "case_mean_stress_relative_l2", "worst_case_u_relative_l2",
                          "worst_case_stress_relative_l2", "case_mean_stress_hotspot_relative_l2"):
                samples = [row[group][field] for row in members]
                values[field] = {"mean": sum(samples) / len(samples), "min": min(samples), "max": max(samples)}
            groups[group] = values
        output.append({"method": method, "checkpoint_role": role, "seeds": [row["seed"] for row in members],
                       "groups": groups, "interpretation": "two predetermined seeds; descriptive mean/range, no final-data selection"})
    return output


def run(args):
    if args.source_check:
        result = source_check(args.plan); print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    if args.output is None:
        raise ValueError("--output is required for actual final evaluation")
    contract, plan = request_contract(args.plan)
    _rules(plan, require_frozen=True)
    gate = contract["input_contract"]["confirmation_gate"]
    if gate["all_six_completed"] is not True or gate["invalid_runs"]:
        raise ValueError("All six identity-valid confirmation runs must complete before final evaluation")
    output = screen.host_path(args.output)
    if (output / "summary.json").is_file():
        stored = screen.read_json(output / "summary.json")
        if stored.get("status") == "completed":
            if stored["config"]["contract"] != contract or screen.read_json(output / "config.json") != stored["config"]:
                raise ValueError("Completed final evaluation identity differs")
            print(f"Reusing completed final evaluation: {output}", flush=True)
            return stored
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Incomplete nonempty final output requires explicit --resume")
    config = {"contract": contract, "label_policy": "input-only case -> complete prediction -> physical decode -> signed source labels",
              "groups": {"known": "final5 x ver/hor/tor", "combination": "final5 x dia, separate transfer readout"},
              "selection": "frozen routes/seeds/roles; no final-data reranking or checkpoint selection",
              "training_updates": 0, "scaler_fits": 0, "cache_preparation": "external input-only helper only"}
    if (output / "config.json").is_file() and screen.read_json(output / "config.json") != config:
        raise ValueError("Resume final evaluation identity differs")
    output.mkdir(parents=True, exist_ok=True); screen.write_json(output / "config.json", config)
    began = time.monotonic()
    previous_status = screen.read_json(output / "status.json") if (output / "status.json").is_file() else {}
    prior_elapsed = previous_status.get("elapsed_seconds", 0.)
    state = {"started_at_utc": previous_status.get("started_at_utc", screen.utc_now()),
             "route_count": len(contract["routes"]), "training_updates": 0, "scaler_fits": 0}
    def elapsed():
        return prior_elapsed + time.monotonic() - began
    def status(phase, **details):
        state.update(details); state.update(phase=phase, updated_at_utc=screen.utc_now(), elapsed_seconds=elapsed())
        screen.write_json(output / "status.json", state)
    status("loading_input_only_cache_metadata")
    cleanup = None
    try:
        import numpy as np
        import torch
        from .benchmark import FieldScaler
        from .solver_selection_data import query_grid
        from .solver_selection_models import build_selection_model, close_selection_process_group
        from .solver_final_eval_data import read_cache_metadata, load_evaluation_case
        cleanup = close_selection_process_group
        manifest = read_cache_metadata(args.plan, contract=contract["input_contract"])
        if manifest["contract"] != contract["input_contract"] or manifest["scaler"] != contract["input_contract"]["scaler"]:
            raise ValueError("Input cache/scaler identity differs from the frozen final request")
        known, combination = manifest["known_keys"], manifest["combination_keys"]
        keys = known + combination
        if len(known) != 15 or len(combination) != 5 or len(set(keys)) != 20 or set(manifest["records"]) != set(keys):
            raise ValueError("Exactly the frozen final15 known and final5 combination cases are required")
        scaler = FieldScaler(**manifest["scaler"]); device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        unique = {}
        for route in contract["routes"]:
            unique.setdefault(route["evaluation_id"], route)
        evaluations = {}
        contract_digest = _digest(contract)
        for evaluation_id, route in unique.items():
            directory = output / "evaluations" / evaluation_id; directory.mkdir(parents=True, exist_ok=True)
            result_path = directory / "result.json"
            if result_path.is_file():
                result = screen.read_json(result_path)
                if result.get("status") != "completed" or result["contract_sha256"] != contract_digest:
                    raise ValueError("Committed route receipt identity differs")
                evaluations[evaluation_id] = result
                status("reusing_committed_route", evaluation_id=evaluation_id, committed_routes=len(evaluations))
                continue
            case_results = {}
            for index, key in enumerate(keys):
                case_path = directory / "cases" / f"{index:02d}.json"
                if case_path.is_file():
                    item = screen.read_json(case_path)
                    if item.get("status") != "completed" or item["contract_sha256"] != contract_digest or item["evaluation_id"] != evaluation_id or item["key"] != key:
                        raise ValueError("Committed case receipt identity differs")
                    case_results[key] = item
            pending = [(index, key) for index, key in enumerate(keys) if key not in case_results]
            startup_path = directory / "startup.json"
            if case_results and not startup_path.is_file():
                raise ValueError("Committed final cases require their original checkpoint identity receipt")
            startup = screen.read_json(startup_path) if startup_path.is_file() else {
                "checkpoint_deserialization_seconds": 0., "model_construction_and_load_seconds": 0., "checkpoint_loads": 0,
                "memory": {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}}
            model = None
            if pending:
                status("loading_frozen_checkpoint", evaluation_id=evaluation_id, method=route["method"], seed=route["seed"])
                matching = next(item for item in plan["routes"] if f"{screen.canonical_method(item['method'])}_seed{item['seed']}_{item['checkpoint_role']}" == route["route_id"])
                current_route, expected_config = _route_metadata(matching, require_frozen=True)
                if current_route != route:
                    raise ValueError("Route metadata changed during final evaluation")
                checkpoint_began = time.monotonic()
                saved = torch.load(screen.host_path(route["checkpoint"]), map_location="cpu", weights_only=False)
                startup["checkpoint_deserialization_seconds"] += time.monotonic() - checkpoint_began
                if saved["step"] != route["selected_step"] or not _compatible_config(saved["config"], expected_config):
                    raise ValueError("Actual checkpoint step/config differs from its exact frozen origin")
                startup["checkpoint_identity"] = {"checkpoint": route["checkpoint"], "stat": route["checkpoint_stat"],
                                                  "selected_step": saved["step"], "config_sha256": _digest(saved["config"]),
                                                  "training_source_sha256": saved["config"]["contract"]["source_sha256"]}
                model_began = time.monotonic()
                torch.cuda.init(); torch.cuda.reset_peak_memory_stats(device)
                model = build_selection_model(route["method"], device=device, seed=route["seed"])
                if model.source_config != route["expected_model_source_config"] or model.parameter_counts != route["parameter_counts"]:
                    raise ValueError("Actual solver source configuration/parameter counts changed")
                model.load_state_dict(saved["model"]); del saved
                model.eval(); screen._sync(torch, device)
                startup["model_construction_and_load_seconds"] += time.monotonic() - model_began
                startup["checkpoint_loads"] += 1
                startup["memory"] = {key: max(value, startup["memory"][key]) for key, value in screen._memory(torch, device).items()}
                screen.write_json(startup_path, startup)
            for index, key in pending:
                record = manifest["records"][key]
                status("predicting_input_only_case", evaluation_id=evaluation_id, key=key, completed_cases=len(case_results))
                case_began = time.monotonic(); input_began = time.monotonic()
                case = load_evaluation_case(record, device, for_fno=route["method"] == "fno")
                if "target" in case or "physical" in case or record.get("input_only") is not True:
                    raise ValueError("Final model inputs must contain no target/physical fields")
                screen._sync(torch, device); input_seconds = time.monotonic() - input_began
                inference_began = time.monotonic()
                with torch.no_grad():
                    devices = [device.index if device.index is not None else torch.cuda.current_device()]
                    with torch.random.fork_rng(devices=devices):
                        if route["method"] == "transolver_pp":
                            torch.manual_seed(route["seed"] + 100000 + index)
                        encoded = screen._predict_case(torch, model, route["method"], case, query_grid)
                screen._sync(torch, device); inference_seconds = time.monotonic() - inference_began
                decode_began = time.monotonic(); prediction = scaler.decode(encoded)
                decode_seconds = time.monotonic() - decode_began
                del case, encoded
                status("reading_labels_after_prediction", evaluation_id=evaluation_id, key=key)
                labels_began = time.monotonic(); truth = _labels_after_prediction(np, record, prediction)
                labels_seconds = time.monotonic() - labels_began
                metrics_began = time.monotonic(); metrics = screen._physical_metrics(np, truth, prediction)
                metrics_seconds = time.monotonic() - metrics_began
                item = {"status": "completed", "contract_sha256": contract_digest, "evaluation_id": evaluation_id,
                        "key": key, "scope": record["scope"], "metrics": metrics,
                        "costs": {"input_preparation_seconds": input_seconds, "inference_seconds": inference_seconds,
                                  "decode_seconds": decode_seconds, "label_read_seconds": labels_seconds,
                                  "physical_metrics_seconds": metrics_seconds, "elapsed_seconds": time.monotonic() - case_began},
                        "memory": screen._memory(torch, device), "label_read_after_prediction": True,
                        "accessed_label_members": ["displacement", "stress"], "source_npz_path": record["source_npz_path"],
                        "label_alignment": record["label_alignment"]}
                screen.write_json(directory / "cases" / f"{index:02d}.json", item)
                case_results[key] = item; del truth, prediction
                status("case_committed", evaluation_id=evaluation_id, key=key, completed_cases=len(case_results))
            if model is not None:
                del model; cleanup(); gc.collect(); torch.cuda.empty_cache()
            groups = {"known": _group(np, {key: case_results[key]["metrics"] for key in known}),
                      "combination": _group(np, {key: case_results[key]["metrics"] for key in combination})}
            for group, group_keys in (("known", known), ("combination", combination)):
                groups[group]["costs"] = {name: sum(case_results[key]["costs"][name] for key in group_keys)
                                           for name in next(iter(case_results.values()))["costs"]}
                groups[group]["costs"].update(inference_case_exposures=len(group_keys),
                                             inference_node_exposures=sum(case_results[key]["metrics"]["nodes"] for key in group_keys))
            costs = {name: sum(item["costs"][name] for item in case_results.values()) for name in next(iter(case_results.values()))["costs"]}
            costs.update(checkpoint_deserialization_seconds=startup["checkpoint_deserialization_seconds"],
                         model_construction_and_load_seconds=startup["model_construction_and_load_seconds"],
                         checkpoint_loads=startup["checkpoint_loads"], inference_case_exposures=len(keys),
                         inference_node_exposures=sum(item["metrics"]["nodes"] for item in case_results.values()))
            memory = {name: max([startup["memory"][name]] + [item["memory"][name] for item in case_results.values()]) for name in startup["memory"]}
            result = {"status": "completed", "contract_sha256": contract_digest, "evaluation_id": evaluation_id,
                      "route": route, **groups, "costs": costs, "memory": memory,
                      "checkpoint_identity": startup.get("checkpoint_identity"),
                      "parameter_counts": route["parameter_counts"], "cases": case_results,
                      "training_updates": 0, "scaler_fits": 0, "final_selection_performed": False}
            screen.write_json(result_path, result); evaluations[evaluation_id] = result
            status("route_committed", evaluation_id=evaluation_id, committed_routes=len(evaluations))
        route_results = [{**route, "result": str(output / "evaluations" / route["evaluation_id"] / "result.json"),
                          **{key: evaluations[route["evaluation_id"]][key] for key in ("known", "combination", "costs", "memory", "parameter_counts", "checkpoint_identity")},
                          "deduplicated": route["route_id"] != unique[route["evaluation_id"]]["route_id"]} for route in contract["routes"]]
        total_costs = {name: sum(result["costs"][name] for result in evaluations.values())
                       for name in next(iter(evaluations.values()))["costs"]}
        peak_memory = {name: max(result["memory"][name] for result in evaluations.values())
                       for name in next(iter(evaluations.values()))["memory"]}
        summary = {"status": "completed", "config": config, "elapsed_seconds": elapsed(),
                   "routes": route_results, "unique_evaluations": len(evaluations), "requested_routes": len(route_results),
                   "costs": total_costs, "memory": peak_memory,
                   "unique_case_predictions": len(evaluations) * len(keys),
                   "deduplicated_case_predictions": (len(route_results) - len(evaluations)) * len(keys),
                   "seed_summary": _seed_summary(route_results), "training_updates": 0, "scaler_fits": 0,
                   "label_read_after_prediction": True, "final_selection_performed": False,
                   "known_keys": known, "combination_keys": combination, "completed_at_utc": screen.utc_now(),
                   "scope": "frozen <=2 routes x2 seeds x primary/VAL-best; final5 known3 separate from final5 dia; original signed physical all-node fields"}
        screen.write_json(output / "summary.json", summary)
        status("completed", committed_routes=len(evaluations), unique_case_predictions=summary["unique_case_predictions"])
        return summary
    except Exception as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if cleanup is not None:
            cleanup()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, required=True)
    result.add_argument("--output", type=Path)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--source-check", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
