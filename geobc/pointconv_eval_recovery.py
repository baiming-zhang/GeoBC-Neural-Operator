"""Fixed480 PointConv DEV-dia evaluator with JSON-equivalent metadata checks.

The frozen trainer stores tuple-valued source metadata in its checkpoint,
whereas JSON files necessarily store lists. This separate adapter changes
only two comparisons to canonical JSON equivalence. It does not repair or
rewrite the trained state, frozen sources, plan, targets or evaluation math.
Authors perform source/JSON checks only; root releases actual evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from . import pointconv_train as original

shared = original.shared
DIAGNOSIS_PATH = "notes/pointconv_checkpoint_metadata_diagnosis.json"
ADAPTER_PATH = "geobc/pointconv_eval_recovery.py"


def canonical_metadata_equal(left, right) -> bool:
    """JSON preserves metadata values while treating tuple/list alike.

    Do not coerce tensors, numbers, paths or other values; invalid/nonfinite
    JSON metadata raises. Sorting affects dictionary order only.
    """
    options = {"sort_keys": True, "separators": (",", ":"), "allow_nan": False}
    return json.dumps(left, **options) == json.dumps(right, **options)


def _diagnosis() -> dict:
    diagnosis = shared.read_json(DIAGNOSIS_PATH)
    expected_paths = {
        "config.model.point_hierarchy.neighbors_layout."+key
        for key in ("neighbors00", "neighbors01", "neighbors11", "neighbors12", "neighbors22")
    } | {
        "config.model.architecture.seven_independent_PC_channels."+key
        for key in ("fine", "down01", "mid", "down12", "bottom", "up_mid", "up_fine")
    }
    differences = diagnosis.get("differences", [])
    if (diagnosis.get("checkpoint_step") != 480 or diagnosis.get("canonical_JSON_equal") is not True
            or len(differences) != 12 or {row.get("path") for row in differences} != expected_paths
            or any(row.get("reason") != "sequence type" or row.get("checkpoint_type") != "tuple"
                   or row.get("JSON_type") != "list" for row in differences)):
        raise ValueError("Expected the fixed480 diagnosis: twelve tuple/list-only metadata differences")
    return diagnosis


def request_dia_contract(plan_path):
    """Original fixed480 metadata contract plus this explicit recovery identity.

    Reads only the original metadata/summary/stat inputs and recorded diagnosis;
    no checkpoint bytes, arrays, models or CUDA. The adapter hashes its source
    file without embedding that hash in its source, so there is no hash cycle.
    """
    contract, plan = original.request_dia_contract(plan_path)
    diagnosis = _diagnosis()
    contract = {**contract, "recovery_adapter": {
        "source": ADAPTER_PATH, "source_sha256": shared.source_sha(ADAPTER_PATH),
        "diagnosis": DIAGNOSIS_PATH, "diagnosis_sha256": shared.source_sha(DIAGNOSIS_PATH),
        "diagnosed_checkpoint_step": diagnosis["checkpoint_step"],
        "diagnosed_tuple_list_differences": len(diagnosis["differences"]),
        "reason": "Frozen checkpoint config/source metadata uses tuples; equivalent JSON uses lists",
        "comparison": "canonical JSON, sort_keys=True, compact separators, allow_nan=False",
        "changed_comparisons": ["checkpoint.config versus run config JSON",
                                "model.source_config versus run config.model JSON"],
        "training_or_scientific_protocol_changes": False,
    }}
    return contract, plan


def request_contract(plan_path):
    """Metadata-only alias for coordinators using the ordinary entry name."""
    return request_dia_contract(plan_path)


def evaluate_dia(plan_path, output_path):
    if output_path is None:
        raise ValueError("DEV-dia requires --output")
    contract, plan = request_dia_contract(plan_path)
    if (plan.get("training_rules_frozen") is not True or plan.get("dia_rules_frozen") is not True
            or not contract["endpoint_completed"]):
        raise ValueError("Root-frozen DEV-dia and completed endpoint480 are prerequisites before arrays/weights/GT")
    output = shared.host_path(output_path)
    config = {"contract": contract, "kind": "pointconv_development_dia_eval"}
    if (output/"summary.json").is_file():
        previous = shared.read_json(output/"summary.json")
        if previous.get("status") == "completed":
            if previous["config"] != config:
                raise ValueError("Completed DEV-dia recovery identity differs")
            return previous
    output.mkdir(parents=True, exist_ok=True)
    shared.write_json(output/"config.json", config)
    began = time.monotonic()
    status = {"phase": "loading_endpoint", "kind": "pointconv_development_dia_eval", "cases_completed": 0}
    shared.write_json(output/"status.json", status)
    try:
        import numpy as np
        import torch
        from .benchmark import FieldScaler
        from .pointconv_field import PointConvField
        from .pointconv_data import read_cache_metadata, load_hierarchy
        inputs = read_cache_metadata(plan_path, contract=contract["training_contract"]["pointconv_input_contract"])
        _, _, manifest = original._metadata(plan_path)
        scaler = FieldScaler(**manifest["scaler"])
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        route = contract["route"]
        run = shared.host_path(route["run"])
        runconfig = shared.read_json(run/"config.json")
        parent = shared.read_json(run/"summary.json")
        checkpoint = shared.host_path(route["checkpoint"])
        if original.stat_record(checkpoint) != route["checkpoint_stat"]:
            raise ValueError("Completed endpoint checkpoint stat changed")
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        # Only comparison 1 changes: tuple/list representation, not values.
        if saved["step"] != 480 or not canonical_metadata_equal(saved["config"], runconfig):
            raise ValueError("Endpoint480 checkpoint canonical config/step differs")
        model = PointConvField(manifest["scaler"], seed=2026, device=device)
        # Only comparison 2 changes; public parameter counts remain strict.
        if (not canonical_metadata_equal(model.source_config, runconfig["model"])
                or original._parameter_counts(model) != runconfig["parameter_counts"]):
            raise ValueError("Endpoint canonical model source/public counts differ")
        model.load_state_dict(saved["model"], strict=True)
        model.eval()
        del saved
        torch.cuda.reset_peak_memory_stats(device)
        per_case, input_checks, inference_seconds = {}, {}, 0.
        for cad in plan["development_dia"]["CADs"]:
            key = cad+"/dia"
            receipt = output/"cases"/(cad+".json")
            if receipt.is_file():
                previous = shared.read_json(receipt)
                if (previous.get("status") != "completed"
                        or previous.get("contract_sha256") != original.digest(contract)):
                    raise ValueError("Committed DEV-dia recovery case identity differs")
                per_case[key] = previous["metrics"]
                input_checks[key] = previous["input_check"]
                inference_seconds += previous["inference_seconds"]
                continue
            source = shared.host_path(contract["sources"][key]["path"])
            with np.load(source, allow_pickle=False) as archive:
                xyz = np.array(archive["xyz"], dtype=np.float32)
                features = np.array(archive["features"], dtype=np.float32)
            record = manifest["records"][cad+"/ver"]
            canonical = np.load(shared.host_path(record["xyz_path"]), mmap_mode="r", allow_pickle=False)
            reference = np.load(shared.host_path(record["features_path"]), mmap_mode="r", allow_pickle=False)
            if (xyz.shape != (record["num_nodes"], 3) or features.shape != (len(xyz), 25)
                    or not np.array_equal(xyz, canonical)
                    or not np.array_equal(features[:, original.NONFORCE], reference[:, original.NONFORCE])):
                raise ValueError("DEV-dia original node order/xyz/nonforce19 differs from canonical point hierarchy")
            if not np.isfinite(xyz).all() or not np.isfinite(features).all():
                raise ValueError("DEV-dia nonfinite input")
            case = {"key": key, "xyz": torch.as_tensor(xyz, device=device),
                    "features": torch.as_tensor(features, device=device),
                    "cloud": torch.as_tensor(np.array(np.load(shared.host_path(record["cloud_path"]),
                                                               allow_pickle=False)), device=device)[None],
                    "mask": torch.as_tensor(np.array(np.load(shared.host_path(record["mask_path"]),
                                                              allow_pickle=False)), device=device)[None]}
            hierarchy = load_hierarchy(inputs["records"][cad], device)
            shared._sync(torch, device)
            start = time.monotonic()
            with torch.no_grad():
                encoded = original._predict(torch, np, model, case, hierarchy,
                                            plan["recipe"]["readout_chunk_nodes"])
            shared._sync(torch, device)
            seconds = time.monotonic()-start
            if encoded.shape != (len(xyz), 4) or not np.isfinite(encoded).all():
                raise FloatingPointError("Complete DEV-dia encoded prediction invalid")
            prediction = scaler.decode(encoded)
            if not np.isfinite(prediction).all():
                raise FloatingPointError("DEV-dia physical prediction invalid before GT")
            # Same sole route, full original nodes and physical decode before GT.
            with np.load(source, allow_pickle=False) as archive:
                truth = np.column_stack((archive["displacement"], archive["stress"])).astype(np.float64)
            metrics = {**shared._physical_metrics(np, truth, prediction), "inference_seconds": seconds}
            check = {"xyz_and_original_node_order_exact": True, "nonforce19_exact": True,
                     "canonical_hierarchy_and_cloud_reused": True, "all_nodes": len(xyz),
                     "prediction_and_decode_before_GT": True, "source_npz": str(source)}
            shared.write_json(receipt, {"status": "completed", "contract_sha256": original.digest(contract),
                                       "key": key, "metrics": metrics, "input_check": check,
                                       "inference_seconds": seconds})
            per_case[key] = metrics
            input_checks[key] = check
            inference_seconds += seconds
            status.update(phase="evaluating", cases_completed=len(per_case), elapsed_seconds=time.monotonic()-began)
            shared.write_json(output/"status.json", status)
            del case, hierarchy, xyz, features, encoded, prediction, truth
        route_metrics = {"selected_step": 480, "checkpoint": str(checkpoint),
                         "parameter_counts": original._parameter_counts(model),
                         "known3_reused": parent["final_metrics"]["validation"],
                         "development_dia": shared._aggregate(np, per_case),
                         "training_costs": parent["costs"], "inference_seconds": inference_seconds}
        summary = {"status": "completed", "kind": "pointconv_development_dia_eval", "config": config,
                   "routes": {"pointconv": route_metrics}, "selected_step": 480,
                   "parameter_counts": original._parameter_counts(model), "inference_seconds": inference_seconds,
                   "input_checks": input_checks, "elapsed_seconds": time.monotonic()-began,
                   "memory": shared._memory(torch, device), "case_count": 5,
                   "training_updates": 0, "scaler_fits": 0,
                   "scope": "Previously explored DEV5/dia fixed480 diagnostic; tuple/list metadata recovery only, no final claim or checkpoint/seed/budget selection"}
        shared.write_json(output/"summary.json", summary)
        status.update(phase="completed", cases_completed=5, elapsed_seconds=summary["elapsed_seconds"])
        shared.write_json(output/"status.json", status)
        return summary
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        status.update(phase="failed", error=message, elapsed_seconds=time.monotonic()-began)
        shared.write_json(output/"status.json", status)
        shared.write_json(output/"summary.json", {"status": "failed", "kind": "pointconv_development_dia_eval",
                                                "config": config, "error": message,
                                                "elapsed_seconds": time.monotonic()-began})
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--evaluate-dia", action="store_true", help="Compatibility flag; this entry always evaluates DEV-dia")
    return result


def run(args):
    return evaluate_dia(args.plan, args.output)


if __name__ == "__main__":
    run(parser().parse_args())

