"""One root-released, target-independent pinned-GINO CUDA resource acceptance.

The author/source-check route uses only the standard library and file metadata.
Scientific imports and the twelve allowlisted input NPY reads occur in run().
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "b307030d9b7ee530823d6a05bddb8238ba22a743"
SOURCE_FILES = ["models/gino.py", "models/base_model.py", "layers/channel_mlp.py",
    "layers/embeddings.py", "layers/fno_block.py", "layers/spectral_convolution.py",
    "layers/gno_block.py", "layers/gno_weighting_functions.py", "layers/complex.py",
    "layers/normalization_layers.py", "layers/skip_connections.py", "layers/einsum_utils.py",
    "layers/base_spectral_conv.py", "layers/resample.py", "layers/integral_transform.py",
    "layers/neighbor_search.py", "layers/segment_csr.py", "utils.py"]
INPUT_MEMBERS = {"xyz": ("xyz_path", [179316,3], "float32"),
    "features": ("features_path", [179316,25], "float32"),
    "cloud": ("cloud_path", [1024,3], "float32"), "mask": ("mask_path", [1024], "bool")}
CACHE_SCHEMA = {
    "grid_queries_row_ptr": ([262145], "int64"),
    "grid_queries_neighbors_index": ([6547680], "int64"),
    "grid_queries_squared_distances": ([6547680], "float64"),
    "point_queries_row_ptr": ([179317], "int64"),
    "point_queries_neighbors_index": ([6547680], "int64"),
    "point_queries_squared_distances": ([6547680], "float64"),
    "grid_axis": ([64], "float32"), "grid_xyz": ([262144,3], "float32")}


def resolve(value):
    """Map saved WSL/Windows workspace paths to the current identical checkout."""
    value = str(value).replace("\\", "/")
    for marker in ("/GeoBC Neural Operator/",):
        if marker in value:
            return ROOT / value.split(marker, 1)[1]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path):
    return resolve(path).resolve().relative_to(ROOT.resolve()).as_posix()


def read_json(path):
    return json.loads(resolve(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(resolve(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def stat_record(path):
    path = resolve(path)
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"Expected a file: {path}")
    return {"path": relative(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def identity(path):
    result = stat_record(path)
    result["sha256"] = sha(path)  # only small source/JSON identities call this
    return result


def require(value, message):
    if not value:
        raise ValueError(message)


def request_contract(plan_path):
    """Pure stdlib: locked source/JSON and allowlisted NPY stat, never NPY data."""
    plan = read_json(plan_path)
    require(plan.get("kind") == "gino_native_resource_acceptance" and plan.get("status") == "predeclared", "Expected frozen predeclared resource plan")
    recipe = plan["recipe"]
    fixed = {"case_key":"101_479/ver", "seed":2026, "device":"cuda", "input_points":179316,
        "resolution":64, "radius":.033, "expected_edges_per_orientation":6547680,
        "query_count":512, "input_query_chunk":2048, "in_channels":25, "out_channels":4,
        "fno_in_channels":25, "latent_feature_channels":32, "fno_hidden_channels":16,
        "fno_n_modes":[12,12,12], "fno_n_layers":4, "cloud_points":1024,
        "global_dim":32, "point_width":64, "output_weighting":"half_cos", "main_backwards":1,
        "optimizer_steps":0}
    require(all(recipe.get(k) == v for k,v in fixed.items()), "Resource recipe changed")
    parity = recipe["backend_parity"]
    require(parity["max_rows_per_orientation"] == 8 and parity["atol"] == 1e-6 and parity["rtol"] == 1e-5 and parity["autograd_grad_calls"] == 4, "Parity contract changed")
    access = plan["access"]
    require(access["allowed_case_keys"] == ["101_479/ver"] and access["labels_read"] == access["weights_read"] == access["scaler_fits"] == 0, "Input-only access changed")
    frozen = plan["frozen_identity"]
    manifest_id = identity(plan["selection_manifest"])
    split_id = identity(plan["existing_split"])
    radius_id = identity(plan["radius_summary"])
    require(manifest_id["sha256"] == frozen["selection_manifest_sha256"], "Selection manifest identity changed")
    require(split_id["sha256"] == frozen["split_sha256"], "Split identity changed")
    require(radius_id["sha256"] == frozen["radius_summary_sha256"], "Completed radius evidence changed")
    manifest, split, radius = read_json(plan["selection_manifest"]), read_json(plan["existing_split"]), read_json(plan["radius_summary"])
    record = manifest["records"]["101_479/ver"]
    require(record["group"] == "train" and record["cad_id"] == "101_479" and record["num_nodes"] == 179316, "Only fixed TRAIN CAD allowed")
    require("101_479" in split["splits"]["train"]["cad_ids"], "CAD absent from locked TRAIN split")
    require(record["label_alignment"] == "official CSV GRID coordinates -> H5 vertices", "Source alignment changed")
    require(digest(manifest["contract"]) == frozen["selection_contract_sha256"], "Selection producer contract changed")
    require(radius["status"] == "completed" and radius["feasible"] is True and all(v is True for v in radius["checks"].values()), "Radius pilot not completed/accepted")
    old_contract = radius["config"]["contract"]
    require(old_contract["selection_contract_sha256"] == frozen["selection_contract_sha256"] and old_contract["recipe"]["case_key"] == "101_479/ver", "Radius source contract mismatch")
    require(radius["stats"]["point_queries"]["query_coverage"] == 1., "Fixed point CSR lacks full input coverage")
    inputs = {}
    for name, (field, shape, dtype) in INPUT_MEMBERS.items():
        path = resolve(record[field]); info = stat_record(path)
        require(path.resolve().is_relative_to((ROOT / "data/geobc_solver_selection").resolve()), "Case input escaped selection cache")
        inputs[name] = {**info, "shape":shape, "dtype":dtype}
    require(inputs["xyz"]["path"] == relative(old_contract["source"]["xyz_path"]), "Radius/cache coordinates differ")
    require({k:inputs["xyz"][k] for k in ("bytes","mtime_ns")} == old_contract["source"]["stat"], "Canonical xyz stat changed")
    inventory = radius["inventory"]
    require(set(inventory) == set(CACHE_SCHEMA), "Expected exactly eight cached radius members")
    cache = {}
    cache_root = resolve(plan["radius_summary"]).parent.resolve()
    for name, (shape, dtype) in CACHE_SCHEMA.items():
        item = inventory[name]; current = stat_record(item["path"])
        require(resolve(item["path"]).parent.resolve() == cache_root, "Radius member escaped fixed cache")
        require(item["shape"] == shape and item["dtype"] == dtype, f"Radius schema changed: {name}")
        require(all(current[k] == item[k] for k in ("bytes","mtime_ns")), f"Radius member stat changed: {name}")
        cache[name] = {**current, "shape":shape, "dtype":dtype,
            "array_values_sha256_from_completed_pilot":item["array_values_sha256"]}
    official = read_json(plan["official_source"])
    require(official["commit"] == frozen["official_commit"] == COMMIT and official["version"] == frozen["official_version"] == "2.0.0", "Official source pin changed")
    upstream = {}
    for item in SOURCE_FILES:
        path = "third_party/neuraloperator/neuralop/" + item
        upstream[path] = sha(path)
    for path, expected in official["source_files"].items():
        require(sha("third_party/neuraloperator/"+path) == expected, f"Pinned upstream source changed: {path}")
    sources = {path:sha(path) for path in ("geobc/gino_solver.py", "tools/validate_gino_native_resource.py", "geobc/conditioned_field.py", "geobc/tcnn_fields.py", "geobc/solver_selection_data.py")}
    require(sources["geobc/solver_selection_data.py"] == frozen["producer_sha256"], "Frozen cache producer changed")
    contract = {"schema":"geobc_gino_native_resource_v1", "plan":identity(plan_path),
        "recipe":recipe, "selection_manifest":manifest_id, "selection_contract_sha256":frozen["selection_contract_sha256"],
        "split":split_id, "radius_summary":radius_id, "official_source":identity(plan["official_source"]),
        "official_commit":COMMIT, "upstream_closure_sha256":upstream, "source_sha256":sources,
        "source_case":{"case_key":"101_479/ver", "group":"train", "cad_id":"101_479", "label_alignment":record["label_alignment"]},
        "case_inputs":inputs, "radius_inputs":cache, "access":access,
        "upstream_package_initializers_executed":False, "no_runtime_install_clone":True,
        "limitations":plan["limitations"]}
    contract["request_digest"] = digest(contract)
    return contract


def source_check(plan_path):
    began = time.perf_counter()
    for path in ("geobc/gino_solver.py", "tools/validate_gino_native_resource.py"):
        ast.parse(resolve(path).read_text(encoding="utf-8"), filename=path)
    contract = request_contract(plan_path)
    forbidden = [k for k in sys.modules if k.split(".")[0] in ("numpy","torch","h5py","tinycudann","tensorly","tltorch") or k.startswith("_geobc_neuralop_")]
    require(not forbidden, "Source-check unexpectedly imported a scientific module")
    return {"status":"source_check_passed", "contract":contract,
        "elapsed_seconds":time.perf_counter()-began,
        "receipt":{"scientific_files_opened":[], "scientific_arrays_read":0, "labels_read":0,
            "models_constructed":0, "GPU_initialized":False, "forbidden_imports":forbidden,
            "toy_runs":0, "training_updates":0}}


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def run(plan_path, output):
    began = time.perf_counter(); output = resolve(output)
    contract = request_contract(plan_path)
    summary_path = output / "summary.json"
    if summary_path.exists():
        previous = read_json(summary_path)
        require(previous.get("status") == "completed" and previous["config"]["contract"] == contract, "Existing resource output differs/incomplete; no silent rerun")
        return {"status":"completed", "reused":True, "summary":relative(summary_path), "scientific_arrays_read":0, "GPU_initialized":False}
    require(not output.exists() or not any(output.iterdir()), "Partial output immutable; use root-controlled explicit fresh path")
    output.mkdir(parents=True, exist_ok=True)
    config = {"contract":contract, "purpose":"Full input/512-query resource and backend acceptance, target-independent", "training_updates":0}
    write_json(output / "config.json", config)
    write_json(output / "status.json", {"status":"running", "phase":"loading_allowlisted_inputs", "config":config})
    receipt = {"scientific_files_opened":[], "labels_read":0, "weights_read":0,
        "scaler_fits":0, "optimizer_steps":0, "training_updates":0, "main_backwards":0,
        "backend_autograd_grad_calls":0, "models_constructed":0, "GPU_initialized":False,
        "source_model_files_edited":0, "neighbor_search_calls":0, "dense_cdist_calls":0,
        "allowed_case_keys":["101_479/ver"]}
    try:
        import numpy as np
        import torch
        sys.path.insert(0,str(ROOT))
        from geobc.gino_solver import GINOSolver, PREFIX
        require(torch.cuda.is_available(), "CUDA required; no CPU/runtime fallback")
        torch.cuda.init(); torch.cuda.synchronize(); receipt["GPU_initialized"] = True
        torch.cuda.reset_peak_memory_stats()
        costs = {}; phase = time.perf_counter(); arrays = {}
        for name, item in {**contract["case_inputs"], **contract["radius_inputs"]}.items():
            path = resolve(item["path"])
            require(stat_record(path) == {k:item[k] for k in ("path","bytes","mtime_ns")}, f"Input stat changed before read: {name}")
            array = np.load(path, allow_pickle=False)
            receipt["scientific_files_opened"].append(relative(path))
            require(list(array.shape) == item["shape"] and str(array.dtype) == item["dtype"], f"Input dtype/shape mismatch: {name}")
            require(bool(np.isfinite(array).all()), f"Nonfinite input: {name}")
            arrays[name] = array
        require(bool(arrays["mask"].any()), "Geometry cloud has no valid point")
        costs["input_cache_read_validation_seconds"] = time.perf_counter()-phase
        phase = time.perf_counter(); tensors = {name:torch.from_numpy(array).to("cuda") for name,array in arrays.items()}
        for orientation, rows, bound in (("grid_queries",262144,179316),("point_queries",179316,262144)):
            ptr = tensors[orientation+"_row_ptr"]; ids = tensors[orientation+"_neighbors_index"]; distance=tensors[orientation+"_squared_distances"]
            require(int(ptr[0]) == 0 and int(ptr[-1]) == 6547680 and bool((ptr[1:]>=ptr[:-1]).all()), f"Bad cached {orientation} CSR")
            require(bool(((ids>=0)&(ids<bound)).all()) and bool(((distance>=0)&(distance<=.033**2)).all()), f"Bad {orientation} bounds/distance")
        torch.cuda.synchronize(); costs["H2D_runtime_CSR_guard_seconds"] = time.perf_counter()-phase
        case = {name:tensors[name] for name in ("xyz","features","grid_xyz")}
        case.update(cloud=tensors["cloud"].unsqueeze(0),mask=tensors["mask"].unsqueeze(0))
        for name,orientation in (("grid_csr","grid_queries"),("point_csr","point_queries")):
            case[name] = {key:tensors[orientation+"_"+key] for key in ("row_ptr","neighbors_index","squared_distances")}
        phase=time.perf_counter(); model=GINOSolver(seed=2026,query_chunk=2048).to("cuda"); receipt["models_constructed"]=1
        torch.cuda.synchronize(); costs["model_initialization_seconds"]=time.perf_counter()-phase
        parameter_counts = {"PointNet":sum(p.numel() for p in model.encoder.parameters()),
            "official_GINO":sum(p.numel() for p in model.solver.parameters()),
            "total":sum(p.numel() for p in model.parameters())}
        phase=time.perf_counter(); state=model.encode_case(case); torch.cuda.synchronize(); costs["full_input_encode_seconds"]=time.perf_counter()-phase
        phase=time.perf_counter(); predictions=model.query(state,torch.arange(512,device="cuda",dtype=torch.int64)); torch.cuda.synchronize(); costs["query512_seconds"]=time.perf_counter()-phase
        require(predictions.shape == (512,4) and bool(torch.isfinite(predictions).all()), "Nonfinite or wrong-shaped query output")
        objective=predictions.square().mean()
        phase=time.perf_counter(); objective.backward(); receipt["main_backwards"]=1; torch.cuda.synchronize(); costs["backward_seconds"]=time.perf_counter()-phase
        gradient_groups={}; missing=[]; nonfinite=[]
        for name,parameter in model.named_parameters():
            group=("PointNet" if name.startswith("encoder.") else "input_GNO" if name.startswith("solver.gno_in.") else "output_GNO" if name.startswith("solver.gno_out.") else "projection" if name.startswith("solver.projection.") else "FNO_lifting")
            current=gradient_groups.setdefault(group,{"parameters":0,"gradient_L1":0.,"finite":True})
            current["parameters"]+=parameter.numel()
            if parameter.grad is None:
                missing.append(name); continue
            finite=bool(torch.isfinite(parameter.grad).all()); current["finite"] &= finite
            if not finite: nonfinite.append(name)
            current["gradient_L1"]+=float(parameter.grad.abs().sum())
        main_gradient_passed=not missing and not nonfinite and all(row["finite"] and row["gradient_L1"]>0 for row in gradient_groups.values())
        torch.cuda.synchronize(); main_peak={"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()}
        phase=time.perf_counter(); parity=model.reducer_parity(case,state,atol=1e-6,rtol=1e-5); receipt["backend_autograd_grad_calls"]=parity["autograd_grad_calls"]
        torch.cuda.synchronize(); costs["small_real_CSR_backend_parity_seconds"]=time.perf_counter()-phase
        checks={"full_input_count":state.input_points==179316,"latent_shape":list(state.latent.shape)==[1,262144,16],
            "query_shape":list(predictions.shape)==[512,4],"output_finite":bool(torch.isfinite(predictions).all()),
            "objective_finite":bool(torch.isfinite(objective)),"main_gradients_finite_positive":main_gradient_passed,
            "official_reducer_value_gradient_parity":parity["passed"],"no_optimizer_or_GT":True}
        costs["elapsed_seconds"]=time.perf_counter()-began
        loaded={name:relative(module.__file__) for name,module in sys.modules.items() if name.startswith(PREFIX) and getattr(module,"__file__",None)}
        summary={"status":"completed","feasible":all(checks.values()),"config":config,
            "checks":checks,"api":{"encode_case":"full canonical TRAIN input -> reusable latent case state",
                "query":"first512 original node IDs -> [512,4]", "input_points":state.input_points,
                "latent_shape":list(state.latent.shape),"query_shape":list(predictions.shape),"arbitrary_query_csr_required":True},
            "model_source_config":model.source_config,"parameter_counts":parameter_counts,
            "gradients":{"groups":gradient_groups,"missing_parameter_names":missing,"nonfinite_parameter_names":nonfinite},
            "backend_parity":parity,"costs":costs,"memory":{"main_encode_query_backward":main_peak,
                "including_backend_parity":{"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()},
                "scope":"Actual Torch allocator for this full-input/512-output task; not fullN-output training or total driver memory"},
            "target_independent_objective":{"definition":"output.square().mean()","value":float(objective.detach()),"is_accuracy_metric":False},
            "receipt":receipt,"loaded_official_modules":loaded,
            "environment":{"torch_version":torch.__version__,"CUDA_runtime":torch.version.cuda,"GPU_name":torch.cuda.get_device_name()},
            "limitations":contract["limitations"],"completed_at_utc":datetime.now(timezone.utc).isoformat()}
        write_json(summary_path,summary); write_json(output / "status.json",{"status":"completed","feasible":summary["feasible"],"summary":"summary.json"})
        return {"status":"completed","feasible":summary["feasible"],"summary":relative(summary_path),"elapsed_seconds":costs["elapsed_seconds"],"main_backwards":1,"optimizer_steps":0}
    except Exception as error:
        failure={"status":"failed","config":config,"error":type(error).__name__+": "+str(error),
            "traceback":traceback.format_exc(),"receipt":receipt,"elapsed_seconds":time.perf_counter()-began,
            "no_fallback_or_retry":True}
        write_json(summary_path,failure); write_json(output / "status.json",failure)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",default="notes/geobc_gino_native_resource_plan.json")
    parser.add_argument("--output",default="runs/geobc_exploration/gino_native_resource_101_479")
    parser.add_argument("--source-check",action="store_true")
    args=parser.parse_args()
    result=source_check(args.plan) if args.source_check else run(args.plan,args.output)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
