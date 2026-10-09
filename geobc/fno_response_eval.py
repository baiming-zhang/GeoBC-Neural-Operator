"""Frozen DEV5/dia response diagnostic; source check is stdlib metadata only.

This module never trains or fits a scaler. Actual execution requires root's
frozen protocol and both response arms' completed 480-update endpoints.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
CADS = ("102_249", "102_38", "107_91", "114_202", "116_440")
ROUTES = ("linear_basis", "load_dependent_basis", "historical_direct4")
ALIGNMENT = "official CSV GRID coordinates -> H5 vertices"
DEFAULT_PLAN = "notes/geobc_fno_response_eval_plan.json"
PROTOCOL = {"CADs": list(CADS), "load": "dia", "seed": 2026,
    "checkpoint_step": 480, "train_cases": 120, "passes": 4,
    "channels": ["ux_mm", "uy_mm", "uz_mm", "signed_stress_MPa"],
    "grid_resolution": 64, "load_feature_slots": [7, 13],
    "normalized_span_tolerance": 1e-7, "all_nodes": True,
    "precision": {"model_inputs_compute":"float32 with original training TF32 policy",
        "float32_matmul_precision":"high", "matmul_allow_tf32":True,
        "cudnn_allow_tf32":True, "physical_decode_and_metrics":"float64",
        "same_for_all_three_routes":True},
    "training_updates": 0, "scaler_fits": 0,
    "known3": "reuse completed endpoint summaries only",
    "checkpoint_selection": "endpoint480 only; no dia selection of best, seed or budget",
    "scope": "Previously explored development-validation CAD5; not unseen final generalization"}
SOURCES = ("geobc/fno_response_eval.py", "geobc/fno_response_field.py",
    "geobc/solver_selection_models.py", "geobc/solver_selection_data.py",
    "geobc/solver_selection_train.py", "geobc/benchmark.py",
    "geobc/conditioned_field.py", "geobc/fno.py")


def host_path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", value):
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def read_json(path):
    return json.loads(host_path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with host_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stat_identity(path):
    path = host_path(path); stat = path.stat()
    # Portable pin: the author uses Windows; root executes the same files in WSL.
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _rules(plan, actual=False):
    if plan.get("protocol") != PROTOCOL or tuple(route["route_id"] for route in plan["routes"]) != ROUTES:
        raise ValueError("Evaluation scope/protocol or three fixed routes changed")
    if actual and (plan.get("status") != "frozen" or plan.get("protocol_frozen") is not True):
        raise ValueError("Root must freeze this DEV dia scope/protocol before actual input or field access")
    if host_path(plan["selection_manifest"]) != host_path("data/geobc_solver_selection/manifest.json") or host_path(plan["prepared_manifest"]) != host_path("data/prepared_geobc_multiload/manifest.json") or host_path(plan["basis_metadata"]) != host_path("notes/geobc_load_basis_45.json"):
        raise ValueError("Only the declared existing selection/aligned-input/rank3 metadata sources are allowed")
    for route in plan["routes"]:
        run = host_path(route["run"])
        expected = "step480.pt" if route["route_id"] == "historical_direct4" else "last.pt"
        if route.get("step") != 480 or route.get("seed") != 2026 or host_path(route["checkpoint"]) != run / expected:
            raise ValueError("Only exact seed2026 endpoint480 checkpoint origins are allowed")
        if route["route_id"] == "historical_direct4" and run != host_path("runs/geobc_exploration/solver_screen_fno_480"):
            raise ValueError("Historical direct4 origin must be the existing six-family FNO screen")


def _metadata_inputs(plan):
    manifest = read_json(plan["selection_manifest"])
    prepared = read_json(plan["prepared_manifest"])
    basis = read_json(plan["basis_metadata"])
    if manifest.get("status") != "completed" or len(manifest["train_keys"]) != 120 or len(set(manifest["train_keys"])) != 120:
        raise ValueError("Expected the existing completed TRAIN-only selection cache")
    expected_validation = {f"{cad}/{load}" for cad in CADS for load in ("ver", "hor", "tor")}
    if set(manifest["validation_keys"]) != expected_validation or any(key.rsplit("/",1)[-1] not in ("ver","hor","tor") for key in manifest["train_keys"]):
        raise ValueError("Existing cache must be TRAIN known3 / exactly this DEV5 known3")
    scaler = manifest["scaler"]
    if scaler["channels"] != [0,1,2,3] or len(scaler["mean"]) != 4 or len(scaler["std"]) != 4 or not all(math.isfinite(float(v)) for v in scaler["mean"] + scaler["std"]) or min(scaler["std"]) <= 0 or not math.isfinite(scaler["stress_scale"]) or scaler["stress_scale"] <= 0:
        raise ValueError("Shared four-channel TRAIN-only asinh scaler is invalid")
    flags = ("compatible", "basis_matches_shared_reference", "constraints_identical", "zero_fixed_constraints", "material_identical_within_CAD", "load_interface_identical", "STATICS_without_nonlinear_entries")
    if basis.get("compatible") is not True or basis.get("CADs_checked") != 45 or basis.get("compatible_CADs") != 45 or basis.get("badcases") != [] or len(basis.get("cases",[])) != 45 or any(row.get("basis_rank") != 3 or any(row.get(flag) is not True for flag in flags) for row in basis["cases"]):
        raise ValueError("Existing45 homogeneous rank3/material/SPC/interface metadata guard failed")
    if basis["canonical_basis"]["order"] != ["ver","hor","tor"] or basis["canonical_basis"]["exact_rank"] != 3 or basis["prepared_feature_mapping"]["tolerance_normalized"] != 1e-7:
        raise ValueError("Only the already verified rank3 span is supported")
    prepared_rows = {row["sample_key"]: row for row in prepared["cases"]}
    records, identities = [], {}
    for cad in CADS:
        key = f"{cad}/dia"; row = prepared_rows[key]; old = manifest["records"][f"{cad}/ver"]
        npz = host_path(f"data/prepared_geobc_multiload/{cad}_dia.npz")
        if host_path(row["npz"]) != npz or row["label_alignment"] != ALIGNMENT or row["nodes"] != old["num_nodes"] or old["group"] != "validation" or any(row.get(flag) is not True for flag in ("all_four_loads_have_identical_constraints","all_four_loads_use_same_loaded_interface")):
            raise ValueError("DEV dia source/alignment/node-count or homogeneous interface metadata differs")
        record = {"key":key,"cad_id":cad,"nodes":row["nodes"],"npz":str(npz),
                  "physical_load6":row["force_N"] + row["moment_Nmm"]}
        for name in ("xyz","cloud","mask","query_indices","query_weights"):
            path = host_path(old[f"{name}_path"])
            expected = host_path(f"data/geobc_solver_selection/geometry/{cad}/{name}.npy")
            if path != expected:
                raise ValueError("DEV input geometry/stencil path is outside the fixed old frontend")
            record[f"{name}_path"] = str(path)
        record["reference_features_path"] = str(host_path(f"data/geobc_solver_selection/cases/{cad}_ver/features.npy"))
        if host_path(old["features_path"]) != host_path(record["reference_features_path"]):
            raise ValueError("Unexpected target-free reference feature cache")
        record["known3_reference_features"] = {}
        for load in ("ver","hor","tor"):
            known = manifest["records"][f"{cad}/{load}"]
            if known["group"] != "validation" or known["num_nodes"] != row["nodes"] or any(host_path(known[f"{name}_path"]) != host_path(record[f"{name}_path"]) for name in ("xyz","cloud","mask","query_indices","query_weights")):
                raise ValueError("Known3 canonical geometry/cloud/stencil cache is not shared within this DEV CAD")
            feature_path = host_path(f"data/geobc_solver_selection/cases/{cad}_{load}/features.npy")
            if host_path(known["features_path"]) != feature_path:
                raise ValueError("Known3 non-force reference path differs")
            record["known3_reference_features"][load] = str(feature_path)
            identities[f"{key}:known3_{load}_features"] = stat_identity(feature_path)
        for name in ("npz","xyz_path","cloud_path","mask_path","query_indices_path","query_weights_path","reference_features_path"):
            identities[f"{key}:{name}"] = stat_identity(record[name])
        records.append(record)
    return manifest, basis, records, identities


def _route_metadata(route, manifest, actual=False):
    run = host_path(route["run"]); checkpoint = host_path(route["checkpoint"])
    result = {**route, "run":str(run), "checkpoint":str(checkpoint), "ready":False}
    if not (run / "summary.json").exists() or not (run / "config.json").exists():
        if actual:
            raise ValueError("Both new response arms must complete endpoint480 before any dia access")
        return result
    summary = read_json(run / "summary.json"); config = read_json(run / "config.json")
    complete = summary.get("status") == "completed" and summary.get("endpoint_step") == 480
    if not complete:
        if actual:
            raise ValueError("A required route has not completed its fixed endpoint480")
        return result
    if summary.get("config") != config or config.get("scaler") != manifest["scaler"] or config["model"].get("seed") != 2026:
        raise ValueError("Completed route config/scaler/seed lineage differs")
    exposures = summary["case_exposure_counts"]
    if set(exposures) != set(manifest["train_keys"]) or any(value != 4 for value in exposures.values()) or summary["costs"]["train_case_exposures"] != 480 or summary["case_order_sha256"] != manifest["case_order_sha256"] or summary["final_metrics"]["step"] != 480:
        raise ValueError("Completed endpoint must be four identical known3 passes / 480 exposures")
    if config.get("train_keys") != manifest["train_keys"] or config.get("validation_keys") != manifest["validation_keys"]:
        raise ValueError("Training/evaluation lineage is not the fixed TRAIN40/DEV5 known3 cache")
    if route["route_id"] == "historical_direct4":
        if summary.get("method") != "fno" or config["model"].get("method") != "fno":
            raise ValueError("Historical baseline must be old direct4 FNO, never a response u9 head")
    elif config["model"].get("response_mode") != route["route_id"]:
        raise ValueError("Response mode/checkpoint origin differs")
    result.update(ready=True,checkpoint_stat=stat_identity(checkpoint),
        config_sha256=sha(run / "config.json"),summary_sha256=sha(run / "summary.json"),
        known3_endpoint_validation=summary["final_metrics"]["validation"],
        known3_endpoint_train_readout=summary["final_metrics"].get("train_readout"),
        training_costs=summary["costs"],parameter_counts=summary["parameter_counts"])
    if route.get("checkpoint_stat") is not None and route["checkpoint_stat"] != result["checkpoint_stat"]:
        raise ValueError("Frozen checkpoint stat changed")
    return result


def request_contract(plan_path=DEFAULT_PLAN):
    """Pure stdlib: JSON/source SHA/input stat, no scientific imports/weights."""
    path = host_path(plan_path); plan = read_json(path); _rules(plan)
    manifest, basis, records, identities = _metadata_inputs(plan)
    sources = {name:sha(name) for name in SOURCES}
    for name, expected in plan.get("source_sha256",{}).items():
        if sha(name) != expected:
            raise ValueError(f"Frozen source identity changed: {name}")
    metadata = {name:sha(plan[name]) for name in ("selection_manifest","prepared_manifest","basis_metadata")}
    if plan.get("metadata_sha256",metadata) != metadata:
        raise ValueError("Frozen input metadata changed")
    if plan.get("input_files",identities) != identities:
        raise ValueError("Frozen dia / target-free geometry input stat changed")
    contract = {"plan":str(path),"plan_sha256":sha(path),"protocol":PROTOCOL,
        "source_sha256":sources,"metadata_sha256":metadata,"input_files":identities,
        "records":records,"scaler":manifest["scaler"],
        "case_order_sha256":manifest["case_order_sha256"],
        "routes":[_route_metadata(route,manifest) for route in plan["routes"]]}
    return contract, plan


def _prepare_case(np, torch, frontend, record, basis, device):
    started = time.perf_counter()
    with np.load(record["npz"],allow_pickle=False) as archive:
        # Explicit target-free allowlist; no displacement/stress access here.
        key = str(archive["sample_key"].item()); alignment = str(archive["label_alignment"].item())
        xyz = np.asarray(archive["xyz"],np.float32); features = np.asarray(archive["features"],np.float32)
    cached = {name:np.load(record[f"{name}_path"],allow_pickle=False) for name in ("xyz","cloud","mask","query_indices","query_weights")}
    n = record["nodes"]
    if key != record["key"] or alignment != ALIGNMENT or xyz.shape != (n,3) or features.shape != (n,25) or not np.array_equal(xyz,cached["xyz"]):
        raise ValueError("Loaded dia geometry/node ordering/nonload features differ from the old target-free cache")
    nonforce = list(range(7))+list(range(13,25)); nonforce_deltas = {}
    for load, path in record["known3_reference_features"].items():
        reference = np.load(path,allow_pickle=False)
        if reference.shape != features.shape or not np.isfinite(reference).all():
            raise ValueError(f"Known3 {load} input reference schema/nonfinite failure")
        nonforce_deltas[load] = float(np.abs(features[:,nonforce].astype(np.float64)-reference[:,nonforce].astype(np.float64)).max())
        if not np.array_equal(features[:,nonforce],reference[:,nonforce]):
            raise ValueError(f"Dia non-force19 differs from canonical known3 {load}: max_FP32_difference={nonforce_deltas[load]}")
        del reference
    if cached["cloud"].shape != (1024,3) or cached["mask"].shape != (1024,) or cached["mask"].dtype != np.bool_ or cached["query_indices"].shape != (n,8) or cached["query_weights"].shape != (n,8):
        raise ValueError("Cached geometry cloud/static stencil schema differs")
    if not all(np.isfinite(value).all() for value in (xyz,features,cached["cloud"],cached["query_weights"])):
        raise FloatingPointError("Nonfinite target-free dia input")
    if not np.issubdtype(cached["query_indices"].dtype,np.integer) or cached["query_indices"].min() < 0 or cached["query_indices"].max() >= 64**3 or not np.allclose(cached["query_weights"].sum(1),1,rtol=0,atol=2e-6) or cached["query_weights"].min() < -2e-6 or cached["query_weights"].max() > 1+2e-6:
        raise ValueError("Cached original-node stencil has invalid indices/weights")
    if not np.array_equal(features[:,7:13],np.broadcast_to(features[0,7:13],(n,6))):
        raise ValueError("Prepared normalized six-slot load is not case-global")
    load6 = features[0,7:13].astype(np.float64)
    mapping = basis["prepared_feature_mapping"]
    if not np.allclose(load6,np.asarray(record["physical_load6"])/np.asarray(mapping["physical_scale"]),rtol=0,atol=1e-7):
        raise ValueError("Loaded six-slot load differs from declared DEV dia source metadata")
    alpha = np.asarray(mapping["feature_to_alpha_matrix_3x6"],np.float64) @ load6
    residual = load6 - np.asarray(mapping["normalized_basis_6x3"],np.float64) @ alpha
    if np.max(np.abs(residual)) > 1e-7:
        raise ValueError("Loaded dia outside verified rank3 span; not an arbitrary6D API")
    torch.cuda.synchronize(device); input_seconds = time.perf_counter()-started
    started = time.perf_counter()
    indices = torch.as_tensor(cached["query_indices"].astype(np.int64),device=device)
    weights = torch.as_tensor(cached["query_weights"],device=device,dtype=torch.float32)
    raster, active = frontend._splat(torch.as_tensor(features,device=device),indices,weights,64)
    axis = torch.linspace(0,1,64,device=device)
    coords = torch.stack(torch.meshgrid(axis,axis,axis,indexing="ij"))
    grid = torch.cat((active.float().reshape(1,64,64,64),coords,raster.T.reshape(25,64,64,64)))[None]
    case = {"grid":grid,"query_indices":indices,"query_weights":weights,
        "cloud":torch.as_tensor(cached["cloud"],device=device)[None],
        "mask":torch.as_tensor(cached["mask"],device=device)[None],
        "load6":torch.as_tensor(load6,device=device,dtype=torch.float32)}
    torch.cuda.synchronize(device)
    receipt = {"nodes":n,"alpha_ver_hor_tor":alpha.tolist(),"normalized_load6":load6.tolist(),
        "max_normalized_span_residual":float(np.abs(residual).max()),
        "input_read_seconds":input_seconds,"grid_build_seconds":time.perf_counter()-started,
        "grid_builds":1,"shared_routes":3,"geometry_and_nonload_features_exact_cache_replay":True,
        "xyz_max_FP32_difference":0.,"nonforce19_max_FP32_difference_by_known3":nonforce_deltas,
        "shared_cloud_and_stencil_same_known3_file_paths":True,
        "input_archive_members_read":["sample_key","label_alignment","xyz","features"]}
    return case, receipt


def _load_models(torch, contract, basis, device):
    from .fno_response_field import build_fno_response_model
    from .solver_selection_models import build_selection_model
    models, receipts = {}, {}
    for route in contract["routes"]:
        started = time.perf_counter(); name = route["route_id"]
        config = read_json(host_path(route["run"]) / "config.json")
        before = stat_identity(route["checkpoint"])
        checkpoint_sha = sha(route["checkpoint"])
        if route.get("checkpoint_sha256") is not None and checkpoint_sha != route["checkpoint_sha256"]:
            raise ValueError("Frozen endpoint checkpoint SHA changed")
        saved = torch.load(host_path(route["checkpoint"]),map_location="cpu",weights_only=False)
        if before != stat_identity(route["checkpoint"]) or saved["step"] != 480 or saved["config"] != config:
            raise ValueError("Checkpoint identity/endpoint/config changed during load")
        model = build_selection_model("fno",device="cpu",seed=2026) if name == "historical_direct4" else build_fno_response_model(name,contract["scaler"],basis,seed=2026,device="cpu")
        if model.source_config != config["model"] or model.parameter_counts != config["parameter_counts"]:
            raise ValueError("Actual architecture/source/scaler differs from endpoint config")
        model.load_state_dict(saved["model"],strict=True); model.to(device); model.eval()
        models[name] = model
        receipts[name] = {"checkpoint":route["checkpoint"],"checkpoint_step":480,
            "checkpoint_sha256":checkpoint_sha,"checkpoint_stat":before,
            "checkpoint_config_sha256":route["config_sha256"],
            "source_config":config["model"],"parameter_counts":config["parameter_counts"],
            "load_seconds":time.perf_counter()-started}
        del saved; gc.collect()
    return models, receipts


def run(plan_path, output):
    started = time.perf_counter(); contract, plan = request_contract(plan_path); _rules(plan,actual=True)
    if not plan.get("source_sha256") or any(name not in plan["source_sha256"] for name in SOURCES):
        raise ValueError("Actual execution requires all final evaluation dependencies frozen by root")
    manifest, basis, _, _ = _metadata_inputs(plan)
    for route in plan["routes"]:
        _route_metadata(route,manifest,actual=True)
    output = host_path(output); output.mkdir(parents=True,exist_ok=True)
    if (output / "summary.json").exists():
        previous = read_json(output / "summary.json")
        if previous.get("status") == "completed" and previous.get("config",{}).get("contract") == contract:
            print(json.dumps({"status":"completed_reused","output":str(output)})); return previous
        raise ValueError("Existing result is not a strict completed contract match; do not overwrite")
    config = {"contract":contract,"training_updates":0,"scaler_fits":0,"all_nodes":True}
    if (output / "config.json").exists() and read_json(output / "config.json") != config:
        raise ValueError("Existing output config differs from frozen contract")
    write_json(output / "config.json",config)
    write_json(output / "status.json",{"status":"running","kind":"fno_response_eval","training_updates":0,"scaler_fits":0})
    try:
        # Scientific imports begin only after frozen scope + both-arm endpoint gates.
        import numpy as np
        import torch
        from . import solver_selection_data as frontend
        from . import solver_selection_train as screen
        from .benchmark import FieldScaler
        device = torch.device("cuda:0")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        scaler = FieldScaler(**contract["scaler"])
        models, checkpoints = _load_models(torch,contract,basis,device)
        torch.cuda.reset_peak_memory_stats(device)
        per_route = {name:{} for name in ROUTES}; input_receipts = {}; costs = {"input_read_seconds":0.,"grid_build_seconds":0.,"truth_read_seconds":0.,"metrics_seconds":0.,"inference_seconds":{name:0. for name in ROUTES}}
        for record in contract["records"]:
            case, receipt = _prepare_case(np,torch,frontend,record,basis,device)
            input_receipts[record["key"]] = receipt
            costs["input_read_seconds"] += receipt["input_read_seconds"]; costs["grid_build_seconds"] += receipt["grid_build_seconds"]
            predictions = {}
            for name in ROUTES:
                torch.cuda.synchronize(device); tick = time.perf_counter()
                with torch.no_grad():
                    model = models[name]; context = model.encode_geometry(case["cloud"],case["mask"])
                    grid = model.forward_grid(case["grid"],context) if name == "historical_direct4" else model.forward_grid(case["grid"],context,case["load6"])
                    if grid.shape != (1,4,64,64,64):
                        raise ValueError("Each route must return composed encoded4; no direct u9 comparison")
                    encoded = frontend.query_grid(grid,case).float().cpu().numpy()
                torch.cuda.synchronize(device); costs["inference_seconds"][name] += time.perf_counter()-tick
                predictions[name] = scaler.decode(encoded)
                if predictions[name].shape != (record["nodes"],4):
                    raise ValueError("All original-node physical output shape differs")
                del grid, encoded, context
            # Exactly this one DEV dia GT is opened after every route predicts.
            tick = time.perf_counter()
            with np.load(record["npz"],allow_pickle=False) as archive:
                displacement = np.asarray(archive["displacement"],np.float64)
                stress = np.asarray(archive["stress"],np.float64)
            if displacement.shape != (record["nodes"],3) or stress.shape != (record["nodes"],):
                raise ValueError("Aligned signed4 target shape differs")
            truth = np.column_stack((displacement,stress)); costs["truth_read_seconds"] += time.perf_counter()-tick
            tick = time.perf_counter()
            for name in ROUTES:
                per_route[name][record["key"]] = screen._physical_metrics(np,truth,predictions[name])
            costs["metrics_seconds"] += time.perf_counter()-tick
            # Small partial metrics are a failure receipt, not a selection or resume policy.
            write_json(output / "partial_metrics.json",{"completed_case_keys":list(input_receipts),"per_route":per_route,"input_receipts":input_receipts})
            del case, predictions, truth, displacement, stress; gc.collect()
        routes = {name:{"dia":screen._aggregate(np,per_route[name]),
            "known3_endpoint_validation_reused":next(route["known3_endpoint_validation"] for route in contract["routes"] if route["route_id"] == name),
            "known3_endpoint_train_readout_reused":next(route["known3_endpoint_train_readout"] for route in contract["routes"] if route["route_id"] == name),
            "checkpoint":checkpoints[name],"inference_seconds":costs["inference_seconds"][name]} for name in ROUTES}
        for name in ROUTES:
            routes[name]["dia"]["worst_case_stress_hotspot_relative_l2"] = max(row["stress_hotspot_relative_l2"] for row in per_route[name].values())
        torch.cuda.synchronize(device)
        summary = {"status":"completed","kind":"fno_response_eval","config":config,
            "training_updates":0,"scaler_fits":0,"all_nodes":True,"dia_case_keys":[f"{cad}/dia" for cad in CADS],
            "routes":routes,"input_receipts":input_receipts,"costs":costs,
            "elapsed_seconds":time.perf_counter()-started,
            "memory":{"peak_allocated_bytes":torch.cuda.max_memory_allocated(device),"peak_reserved_bytes":torch.cuda.max_memory_reserved(device)},
            "checks":{"both_response_arms_completed_endpoint480":True,"same_known3_case_order_and_four_passes":True,"shared_TRAIN_only_scaler":True,"metadata45_compatible_rank3":True,"loaded_sixslot_span_all5":True,"exact_cached_geometry_and_nonload_features_all5":True,"all_original_nodes_three_routes":True,"all_three_predictions_before_each_GT_read":True},
            "precision_receipt":{"policy":PROTOCOL["precision"],
                "actual_float32_matmul_precision":torch.get_float32_matmul_precision(),
                "actual_matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,
                "actual_cudnn_allow_tf32":torch.backends.cudnn.allow_tf32},
            "read_receipt":{"dia_target_cases":5,"dia_target_members":["displacement","stress"],"known3_target_cases":0,"old_final_cases":0,"sealed_test_cases":0,"training_or_optimizer_steps":0,"scaler_fits":0,"input_grid_builds":5,"array_files_written":0},
            "interpretation":{"load_span_rank":3,"development_group_has_historical_Hash_exposure":True,"blind_or_final_generalization":False,"historical_direct4_is_matched_control":False,"no_dia_selection_or_retuning":True,"stress_superposition_assumed":False,"zero_field_relative_L2_reference":1.0},
            "completed_at_utc":datetime.now(timezone.utc).isoformat()}
        write_json(output / "summary.json",summary)
        write_json(output / "status.json",{"status":"completed","kind":"fno_response_eval","elapsed_seconds":summary["elapsed_seconds"]})
        print(json.dumps({"status":"completed","seconds":summary["elapsed_seconds"],"dia_cases":5,"nodes_per_route":sum(row["nodes"] for row in contract["records"]),"dia":{name:{key:routes[name]["dia"][key] for key in ("case_mean_u_relative_l2","worst_case_u_relative_l2","case_mean_stress_relative_l2","worst_case_stress_relative_l2")} for name in ROUTES}}))
        return summary
    except Exception as error:
        write_json(output / "status.json",{"status":"failed","kind":"fno_response_eval","elapsed_seconds":time.perf_counter()-started,"error_type":type(error).__name__,"error":str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",default=DEFAULT_PLAN); parser.add_argument("--output",required=True)
    parser.add_argument("--source-check",action="store_true")
    args = parser.parse_args()
    if args.source_check:
        ast.parse(Path(__file__).read_text(encoding="utf-8"))
        contract, plan = request_contract(args.plan)
        print(json.dumps({"status":"source_check_passed","protocol_frozen":plan.get("protocol_frozen",False),"ready_routes":[route["route_id"] for route in contract["routes"] if route["ready"]],"scientific_imports":False,"arrays_read":0,"GPU_runs":0}))
    else:
        run(args.plan,args.output)


if __name__ == "__main__":
    main()
