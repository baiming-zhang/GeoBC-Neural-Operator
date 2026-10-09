"""One fixed five-CAD zero-start approximate elastic screen.

Metadata is standard-library only. Actual input reads are limited to vertices,
cells10, canonical xyz/GRID IDs and the real deck. Accepted curved-P2/4Q,
SPC/RBE2 and weighted-LS RBE3 mathematics are reused without global monkey
patches. A finite physical endpoint/equilibrium precedes one displacement read;
stress, dia, final/sealed tests, learned weights and optimization stay closed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_elastic_crosscad_plan.json"
SOURCE = "geobc/elastic_crosscad.py"
KIND = "elastic_crosscad_screen"
CASES = ("101_428/ver","101_56/ver","101_78/ver","102_249/ver","102_38/ver")
MEMBERS = {"geometry_h5":["vertices"],"cells_npz":["cells10"],"canonical_npz":["xyz","grid_node_ids"]}
SOLVE = {"maximum_iterations":10000,"maximum_cumulative_solve_wall_seconds":300,
    "true_relative_residual_tolerance":1e-8,"true_residual_interval":25,
    "checkpoint_interval":25,"resume_budget_reset":False}


def host_path(value):
    from .elastic_pilot_data import _path
    return _path(value)


def read_json(path):
    return json.loads(host_path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    result = hashlib.sha256()
    with host_path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def require(condition,message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = host_path(path); info = path.stat()
    return {"path":path.relative_to(ROOT).as_posix(),"bytes":info.st_size,"mtime_ns":info.st_mtime_ns}


def match(path,record):
    require(identity(path) == {key:record[key] for key in ("path","bytes","mtime_ns")},"Input/cache source stat changed: "+str(path))


def request_contract(plan_path=PLAN):
    """Explicit five-case metadata contract, no old single-CAD entrypoint."""
    plan = read_json(plan_path)
    require(plan.get("kind") == KIND and tuple(plan.get("case_keys",())) == CASES,"Fixed five-CAD order differs")
    recipe = plan["recipe"]
    require(recipe.get("maximum_stage_wall_seconds") == 900 and recipe.get("initial_guess") == "zero"
        and all(recipe.get("solve",{}).get(key) == value for key,value in SOLVE.items()),"Frozen zero-start/total budget policy differs")
    parent = read_json(plan["parent_plan"])
    physics = copy.deepcopy(parent["recipe"])
    require(parent.get("kind") == "elastic_physical_pilot" and physics.get("device") == "cuda:0"
        and physics.get("dtype") == "float64" and physics.get("cell_chunk") == 8192
        and physics.get("quadrature_points_per_cell") == 4 and physics["input_operator"]["maximum_guard_K_applications"] == 8
        and physics["input_operator"]["relative_algebra_guard_tolerance"] == 1e-9
        and physics["geometry_guard"]["geometry_repair"] is False,"Accepted physical recipe differs")
    for key in ("device","dtype","seed","units","continuum","node_dofs","cell_chunk","quadrature","quadrature_points_per_cell",
            "geometry_guard","reader_scope","load_transfer","input_operator","preconditioner"):
        if key in recipe:
            require(recipe[key] == physics[key],"Accepted physical recipe changed: "+key)
    physics["solve"].update(SOLVE)
    physics["post_solve"]["label_scope"] = "Only current fixed CAD/ver displacement after persisted finite endpoint/equilibrium and positive remaining stage budget"
    for field in ("parent_plan","existing_split","gradient_manifest","design"):
        if field+"_sha256" in plan:
            require(sha(plan[field]) == plan[field+"_sha256"],"Pinned metadata differs: "+field)
    split = read_json(plan["existing_split"]); gradient = read_json(plan["gradient_manifest"])
    require(gradient.get("status") == "completed" and gradient.get("complete") is True,"45-CAD geometry metadata incomplete")
    require(tuple(sorted(cad for cad in split["groups"]["train"] if cad != "101_479")[:3]) == tuple(key.split("/")[0] for key in CASES[:3])
        and tuple(sorted(split["groups"]["validation"])[:2]) == tuple(key.split("/")[0] for key in CASES[3:]),"Fixed existing split selection differs")
    records = {}
    for index,key in enumerate(CASES):
        cad = key.split("/")[0]; item = gradient["CADs"][cad]; normal = item["normalization"]
        require(normal.get("status") == "known" and normal["characteristic_length_mm"]>0 and len(normal["bbox_min_mm"]) == 3,"Missing physical geometry normalization")
        paths = {"deck":f"data/raw/SolverDeck/{cad}.fem","geometry_h5":f"data/raw/FieldMesh/{cad}.h5",
            "cells_npz":str(host_path(plan["gradient_manifest"]).parent/item["file"]),"canonical_npz":split["case_paths"][key]}
        require(host_path(paths["canonical_npz"]) == host_path(f"data/prepared_geobc_multiload/{cad}_ver.npz")
            and host_path(paths["geometry_h5"]) == host_path(item["raw_geometry_source"]["path"]),"Fixed source path mismatch")
        sources = {name:identity(path) for name,path in paths.items()}
        frozen_sources = plan.get("cases",{}).get(key,{}).get("source_stat_at_draft",{})
        for name,frozen in frozen_sources.items():
            require(name in sources and all(sources[name][part] == frozen[part] for part in ("path","bytes","mtime_ns")),
                "Frozen case source stat differs: "+key+"/"+name)
        require(all(sources["cells_npz"][part] == item[part] for part in ("bytes","mtime_ns"))
            and all(sources["geometry_h5"][part] == item["raw_geometry_source"][part] for part in ("bytes","mtime_ns")),"Inherited geometry-container stat differs")
        sources["cells_npz"]["sha256"] = item["sha256"]
        expected = {"nodes":item["node_count"],"cells":item["cell_count"],
            "canonical_xyz_float32_sha256":item["xyz_sha256"],"grid_node_ids_int64_sha256":item["grid_node_ids_sha256"],
            "characteristic_length_mm":normal["characteristic_length_mm"],"bbox_min_mm":normal["bbox_min_mm"]}
        records[key] = {"key":key,"cad":cad,"group":"train" if index<3 else "validation", "sources":sources,
            "expected_geometry":expected,"input_members":MEMBERS,"recipe":physics,
            "array_identity_policy":"vertices/cells10 first formal input read hashes recorded; no invented inherited hashes",
            "deferred_displacement_source":sources["canonical_npz"],
            "data_output":plan.get("cases",{}).get(key,{}).get("data_output",f"data/geobc_elastic_crosscad5/{cad}/input"),
            "operator_output":plan.get("cases",{}).get(key,{}).get("operator_output",f"data/geobc_elastic_crosscad5/{cad}/operator"),
            "solve_output":plan.get("cases",{}).get(key,{}).get("solve_output",str(host_path(plan["output"])/(cad+"_ver")))}
        for field in ("data_output","operator_output"):
            require(host_path(records[key][field]) == host_path(f"data/geobc_elastic_crosscad5/{cad}/"+("input" if field == "data_output" else "operator")),"New physical cache path differs")
        require(host_path(records[key]["solve_output"]) == host_path(plan["output"])/(cad+"_ver"),"New case solve output differs")
    source_pins = {name:sha(name) for name in (SOURCE,"geobc/elastic_pilot.py","geobc/elastic_pilot_data.py","geobc/continuous_query_geometry.py")}
    for name,value in plan.get("source_sha256",{}).items():
        require(name in source_pins and source_pins[name] == value,"Pinned source differs: "+name)
    science = {"kind":KIND,"case_keys":list(CASES),"recipe":recipe,"physical_recipe":physics,"records":records,
        "existing_split_sha256":sha(plan["existing_split"]),"gradient_manifest_sha256":sha(plan["gradient_manifest"]),
        "parent_plan_sha256":sha(plan["parent_plan"]),"output":str(host_path(plan["output"])),"source_sha256":source_pins,
        "post_endpoint_label_members":["displacement"],"stress_reads":0,"replacement_CADs":False}
    if "design" in plan:
        science["design_sha256"] = sha(plan["design"])
    return {"schema":"geobc_elastic_crosscad_v1",**science,"science_sha256":digest(science),
        "array_access_in_request_contract":0,"raw_deck_read_in_request_contract":False},plan


class StageClock:
    def __init__(self,output,config,began):
        from .elastic_pilot import write_json
        self.output,self.config,self.began,self.write = output,config,began,write_json
        self.config_sha256 = digest(config)
        self.prior = 0.
        if (output/"status.json").exists():
            old = read_json(output/"status.json"); require(old.get("config_sha256") == self.config_sha256,"Stage accounting config differs")
            self.prior = float(old.get("cumulative_stage_wall_seconds",0))
        if (output/"summary.json").exists():
            old = read_json(output/"summary.json"); require(old.get("config") == config,"Stage summary config differs")
            self.prior = max(self.prior,float(old.get("costs",{}).get("cumulative_stage_wall_seconds",0)))
        require(math.isfinite(self.prior) and self.prior >= 0,"Invalid recorded stage cost")

    def elapsed(self):
        return self.prior+time.monotonic()-self.began

    def remaining(self):
        return max(0.,900-self.elapsed())

    def account(self,phase,key=None):
        # The immutable full config is written once, rather than repeated for every PCG step.
        self.write(self.output/"status.json",{"config_sha256":self.config_sha256,"phase":phase,"case":key,
            "cumulative_stage_wall_seconds":self.elapsed(),"maximum_stage_wall_seconds":900})

    def check(self):
        from .elastic_pilot import ScientificNegative
        if self.remaining() <= 0:
            raise ScientificNegative("stage_wall_budget_exhausted")


def prepare_input(record,output,clock):
    from . import elastic_pilot_data as data
    from .elastic_pilot import ScientificNegative
    began = time.monotonic(); budget = min(300.,clock.remaining())
    config = {"contract":record,"stage":"elastic_crosscad_input"}
    if (output/"manifest.json").exists():
        manifest = read_json(output/"manifest.json")
        require(manifest.get("config") == config,"Input cache belongs to another contract")
        if manifest.get("status") == "completed":
            if manifest.get("feasibility_pass"):
                match(manifest["cache_file"],manifest["file_identity"])
            return manifest
    output.mkdir(parents=True,exist_ok=True); data._write(output/"config.json",config)
    manifest = {"kind":"elastic_crosscad_input","status":"running","config":config,"contract":record,
        "feasibility_pass":False,"node_count":record["expected_geometry"]["nodes"],"cell_count":record["expected_geometry"]["cells"],
        "labels_read":0,"stress_reads":0,"read_receipt":{"accessed_members":{},"labels_read":0,"stress_reads":0,"raw_deck_open_attempted":False}}
    try:
        clock.check(); clock.account("input",record["key"])
        import numpy as np
        import h5py
        sources = record["sources"]; expected = record["expected_geometry"]; n,t = expected["nodes"],expected["cells"]
        tick = time.monotonic()
        for source in sources.values():
            match(source["path"],source)
        actual_cells_archive_sha = sha(sources["cells_npz"]["path"])
        inherited_cells_archive_sha = sources["cells_npz"].get("sha256")
        require(inherited_cells_archive_sha is None or actual_cells_archive_sha == inherited_cells_archive_sha,
            "Inherited cells archive byte identity differs")
        manifest["cells_archive_identity"] = {"actual_sha256":actual_cells_archive_sha,
            "inherited_sha256":inherited_cells_archive_sha,
            "verification_status":"first_actual_unpinned" if inherited_cells_archive_sha is None else "inherited_sha256_verified"}
        with h5py.File(host_path(sources["geometry_h5"]["path"]),"r") as source:
            dataset = source["vertices"]; require(dataset.shape == (n,3) and dataset.dtype == np.dtype("float32"),"Vertices schema differs")
            vertices = dataset[...]
        manifest["read_receipt"]["accessed_members"][sources["geometry_h5"]["path"]] = ["vertices"]
        with np.load(host_path(sources["cells_npz"]["path"]),allow_pickle=False) as source:
            cells = source["cells10"]
        manifest["read_receipt"]["accessed_members"][sources["cells_npz"]["path"]] = ["cells10"]
        with np.load(host_path(sources["canonical_npz"]["path"]),allow_pickle=False) as source:
            xyz,ids = source["xyz"],source["grid_node_ids"]
        manifest["read_receipt"]["accessed_members"][sources["canonical_npz"]["path"]] = ["xyz","grid_node_ids"]
        require(cells.shape == (t,10) and cells.dtype == np.dtype("int64") and xyz.shape == (n,3) and xyz.dtype == np.dtype("float32")
            and ids.shape == (n,) and ids.dtype == np.dtype("int64"),"Canonical input schema differs")
        require(data._array_sha(xyz,"<f4") == expected["canonical_xyz_float32_sha256"]
            and data._array_sha(ids,"<i8") == expected["grid_node_ids_int64_sha256"],"Inherited canonical coordinate/order identity differs")
        reproduced = ((vertices-np.asarray(expected["bbox_min_mm"],np.float32))/expected["characteristic_length_mm"]).astype(np.float32)
        require(np.array_equal(reproduced,xyz) and np.isfinite(vertices).all() and (cells>=0).all() and (cells<n).all()
            and (np.diff(np.sort(cells,axis=1),axis=1)>0).all() and (ids>0).all() and len(np.unique(ids)) == n,"Corrupt geometry/index/GRID mapping")
        array_seconds = time.monotonic()-tick; data._deadline(began,budget)
        manifest["read_receipt"].update(raw_deck_open_attempted=True,raw_deck=sources["deck"]["path"])
        tick = time.monotonic(); deck = data._parse_deck(sources["deck"]["path"],record["recipe"],began,budget)
        parse_seconds = time.monotonic()-tick; data._deadline(began,budget)
        tick = time.monotonic(); arrays,facts = data._build(deck,vertices.astype(np.float64),cells,ids,record,began,budget)
        algebra_seconds = time.monotonic()-tick; data._deadline(began,budget)
        target = output/"input.npz"; temp = target.with_name(target.name+".tmp"); tick = time.monotonic()
        with temp.open("wb") as stream:
            np.savez(stream,**arrays)
        os.replace(temp,target)
        cache_identity = {**identity(target),"sha256":sha(target)}
        manifest.update(status="completed",feasibility_pass=True,cache_file=identity(target)["path"],file_identity=cache_identity,
            facts=facts,array_schema={name:{"shape":list(value.shape),"dtype":str(value.dtype),"sha256":data._array_sha(value,value.dtype)} for name,value in arrays.items()},
            region_array_keys=[name for name in arrays if name.startswith("region_")],
            actual_input_array_sha256={"vertices_float32":data._array_sha(vertices,"<f4"),"cells10_int64":data._array_sha(cells,"<i8")},
            costs={"source_array_seconds":array_seconds,"deck_parse_seconds":parse_seconds,"input_algebra_seconds":algebra_seconds,"save_seconds":time.monotonic()-tick},
            total_cache_bytes=cache_identity["bytes"])
        data._deadline(began,budget)
    except (data.UnsupportedInput,ScientificNegative) as error:
        manifest.update(status="completed",feasibility_pass=False,scientific_outcome="unsupported_input_or_budget",reason=str(error))
    except Exception as error:
        manifest.update(status="failed",feasibility_pass=False,reason=f"{type(error).__name__}: {error}",preparation_seconds=time.monotonic()-began)
        data._write(output/"manifest.json",manifest); raise
    manifest["preparation_seconds"] = manifest["elapsed_seconds"] = time.monotonic()-began
    data._write(output/"manifest.json",manifest); data._write(output/"summary.json",manifest)
    clock.account("input_completed",record["key"])
    return manifest


def prepare_operator(record,manifest,output,clock):
    from . import elastic_pilot as base
    from .elastic_pilot_data import load_input
    config = {"contract":record,"stage":"elastic_crosscad_operator"}; path = output/"summary.json"
    if path.exists():
        previous = read_json(path); require(previous.get("config") == config,"Operator cache config differs")
        if previous.get("status") == "completed":
            if previous.get("feasibility_pass"):
                require(previous["algebra_guards"]["K_applications"] == 8 and sha(output.parent/"input"/"manifest.json") == previous["input_manifest_sha256"],"Committed operator/input guard identity differs")
                for value in previous["arrays"].values():
                    require({key:value[key] for key in ("bytes","mtime_ns")} == {key:identity(value["path"])[key] for key in ("bytes","mtime_ns")},"Operator cache stat differs")
            return previous
    output.mkdir(parents=True,exist_ok=True); base.write_json(output/"config.json",config)
    began = time.monotonic(); require(manifest.get("feasibility_pass") is True,"Input negative, no CUDA")
    try:
        clock.check(); clock.account("operator",record["key"])
        import numpy as np
        import torch
        device = torch.device(record["recipe"]["device"]); torch.cuda.init(); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        inputs = load_input(manifest,device)
        remaining = min(300-float(manifest["preparation_seconds"])-(time.monotonic()-began),clock.remaining())
        if remaining <= 0:
            raise base.ScientificNegative("Input/operator or whole-stage wall budget exhausted")
        # The unchanged helper's 300s deadline now represents the smaller remaining scope.
        helper_began = time.monotonic(); prior = 300-remaining
        operator,chol,diagonal,geometry = base._build_geometry(torch,inputs,record,helper_began,prior)
        guards = base._algebra_guards(torch,operator,diagonal,inputs,record,helper_began,prior)
        saved = {"gradients":base._save_array(np,output/"physical_gradients.npy",operator.gradients),
            "weights":base._save_array(np,output/"quadrature_weights.npy",operator.weights),
            "free_cholesky":base._save_array(np,output/"free_cholesky.npy",chol)}
        base._deadline(torch,helper_began,prior); clock.check()
        summary = {"status":"completed","kind":"elastic_crosscad_operator","config":config,"feasibility_pass":True,
            "geometry":geometry,"algebra_guards":guards,"arrays":saved,"input_manifest_sha256":sha(output.parent/"input"/"manifest.json"),
            "node_count":len(inputs["vertices_mm"]),"cell_count":len(inputs["cells10"]),"costs":{"operator_wall_seconds":time.monotonic()-began,
                "data_preparation_seconds":manifest["preparation_seconds"],"geometry_seconds":geometry["geometry_seconds"],
                "preconditioner_seconds":geometry["preconditioner_seconds"],"guard_K_seconds":operator.apply_seconds,"guard_K_count":8},
            "memory":base._memory(torch),"displacement_reads":0,"stress_reads":0}
    except base.ScientificNegative as error:
        summary = {"status":"completed","kind":"elastic_crosscad_operator","config":config,"feasibility_pass":False,
            "reason":str(error),"costs":{"operator_wall_seconds":time.monotonic()-began},"displacement_reads":0,"stress_reads":0}
    except Exception as error:
        base.write_json(path,{"status":"failed","kind":"elastic_crosscad_operator","config":config,"reason":f"{type(error).__name__}: {error}"}); raise
    base.write_json(path,summary); clock.account("operator_completed",record["key"])
    return summary


def zero_receipt():
    return {"displacement_reads":0,"stress_reads":0,"old_final_reads":0,"DEV_dia_reads":0,
        "sealed_new_test_reads":0,"model_or_weight_reads":0,"optimizer_steps":0,"scaler_fits":0}


def case_base(record):
    return {"key":record["key"],"cad":record["cad"],"group":record["group"],
        "node_count":record["expected_geometry"]["nodes"],"cell_count":record["expected_geometry"]["cells"],
        "attempted":False,"input_compatible":False,"operator_compatible":False,
        "numerically_converged":False,"metrics":None,"read_receipt":zero_receipt()}


def finish_endpoint(record,output,clock):
    """Resume a committed finite endpoint without recomputing it or rereading GT."""
    from . import elastic_pilot as base
    endpoint_path = output/"endpoint_no_GT.json"; endpoint = read_json(endpoint_path)
    summary = endpoint["summary_before_labels"]
    require(summary["config"] == {"contract":record,"stage":"elastic_crosscad_pcg"},"Persisted endpoint config differs")
    for name in ("prediction","constraint_reaction","source_grid_node_ids"):
        value = summary[name]
        require({key:value[key] for key in ("bytes","mtime_ns")} == {key:identity(value["path"])[key] for key in ("bytes","mtime_ns")},"Persisted endpoint array changed")
    label_result = output/"label_metrics.json"; started = output/"label_read_started.json"
    if label_result.exists():
        cached = read_json(label_result)
        require(cached["config"] == summary["config"] and cached["endpoint_sha256"] == sha(endpoint_path),"Committed target result belongs to another endpoint")
        summary["metrics"] = cached["metrics"]; summary["read_receipt"]["displacement_reads"] = 1
        summary["costs"]["GT_read_and_metrics_seconds"] = cached["elapsed_seconds"]
    elif started.exists():
        raise RuntimeError("Previous label read was started but not committed; never repeat this CAD's GT read")
    elif clock.remaining()>0:
        clock.account("post_endpoint_displacement",record["key"])
        source = record["deferred_displacement_source"]; match(source["path"],source)
        claim = {"config":summary["config"],"endpoint_sha256":sha(endpoint_path),"member":"displacement","maximum_reads":1}
        base.write_json(started,claim)
        tick = time.monotonic()
        import numpy as np
        with np.load(host_path(source["path"]),allow_pickle=False) as archive:
            truth = np.array(archive["displacement"],dtype=np.float64)
        summary["read_receipt"]["displacement_reads"] = 1
        base.write_json(started,{**claim,"member_read_completed":True,"displacement_reads":1})
        prediction = np.load(host_path(summary["prediction"]["path"]),allow_pickle=False)
        require(truth.shape == prediction.shape and truth.shape == (record["expected_geometry"]["nodes"],3)
            and np.isfinite(truth).all(),"Post-endpoint displacement shape/finite differs")
        # Only fixed input-derived regions are read from the new target-free cache.
        manifest = read_json(host_path(record["data_output"])/"manifest.json")
        with np.load(host_path(manifest["cache_file"]),allow_pickle=False) as cache:
            regions = {name.removeprefix("region_"):np.array(cache[name]) for name in manifest["region_array_keys"]}
        summary["metrics"] = base._metrics(np,truth,prediction,regions)
        for metric in (summary["metrics"],*summary["metrics"]["regions"].values()):
            if metric["zero_reference"]:
                metric["u_vector_relative_l2"] = None
        duration = time.monotonic()-tick
        base.write_json(label_result,{**claim,"metrics":summary["metrics"],"elapsed_seconds":duration})
        summary["costs"]["GT_read_and_metrics_seconds"] = duration
        clock.account("post_endpoint_displacement_completed",record["key"])
    else:
        summary["GT_skipped_reason"] = "Whole-stage900s budget exhausted before target read"
    summary["costs"]["cold_case_accounted_wall_seconds"] = sum(summary["costs"].get(name,0.) for name in
        ("input_preparation_seconds","operator_wall_seconds","solve_setup_read_seconds","cumulative_solve_wall_seconds",
         "endpoint_verification_seconds","GT_read_and_metrics_seconds"))
    summary["costs"]["cold_case_accounting_definition"] = "Recorded phase walls; shared stage metadata/recovery overhead is separately in total stage wall"
    base.write_json(output/"summary.json",summary)
    base.write_json(output/"status.json",{"config":summary["config"],"phase":"completed","numerical_state":summary["numerical_state"],
        "cumulative_solve_wall_seconds":summary["costs"]["cumulative_solve_wall_seconds"],"iterations_executed_total":summary["iterations_executed_total"]})
    return summary


def solve_case(record,manifest,operator_record,output,clock):
    from . import elastic_pilot as base
    from .elastic_pilot_data import load_input
    config = {"contract":record,"stage":"elastic_crosscad_pcg"}
    if (output/"config.json").exists():
        require(read_json(output/"config.json") == config,"Existing PCG config differs")
    if (output/"summary.json").exists():
        old = read_json(output/"summary.json"); require(old.get("config") == config,"Existing case result config differs")
        if old.get("status") == "completed":
            return old
    if (output/"endpoint_no_GT.json").exists():
        return finish_endpoint(record,output,clock)
    output.mkdir(parents=True,exist_ok=True); base.write_json(output/"config.json",config)
    began = time.monotonic(); clock.account("solve_setup",record["key"])
    state = None; solve_began = None; base_seconds = None
    try:
        clock.check()
        import numpy as np
        import torch
        device = torch.device(record["recipe"]["device"]); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        inputs = load_input(manifest,device); lam,mu = base._validate_inputs(torch,inputs,record)
        arrays = {name:base._load_array(np,torch,value,device) for name,value in operator_record["arrays"].items()}
        require(arrays["gradients"].shape == (len(inputs["cells10"]),4,10,3)
            and arrays["weights"].shape == (len(inputs["cells10"]),4)
            and arrays["free_cholesky"].shape == (len(inputs["free_node_ids"]),3,3)
            and all(value.dtype == torch.float64 for value in arrays.values()),"Accepted operator cache dimensions/dtype differ")
        operator = base.MatrixFreeElastic(torch,inputs["cells10"],arrays["gradients"],arrays["weights"],len(inputs["vertices_mm"]),lam,mu)
        free,chol = inputs["free_node_ids"],arrays["free_cholesky"]
        b = inputs["force"][free]; norm_b = float(torch.linalg.vector_norm(b))
        state = {"config":config,"iteration":0,"iterations_executed_total":0,"cumulative_solve_wall_seconds":0.,
            "cumulative_setup_read_seconds":0.,"K_applications":0,"trace":[],"rz":None}
        z = torch.zeros_like(b); r = b.clone(); p = torch.zeros_like(b)
        if (output/"checkpoint.json").exists():
            state = read_json(output/"checkpoint.json")
            require(state["config"] == config and base.stat_record(state["vectors_path"]) == state["vectors_stat"],"PCG checkpoint identity differs")
            with np.load(host_path(state["vectors_path"]),allow_pickle=False) as archive:
                require(set(archive.files) == {"z","r","p"},"Unexpected PCG member")
                z,r,p = (torch.as_tensor(np.array(archive[name]),device=device) for name in ("z","r","p"))
            require(all(value.shape == b.shape and value.dtype == torch.float64 and torch.isfinite(value).all() for value in (z,r,p)),"Invalid PCG state")
        if (output/"status.json").exists():
            journal = read_json(output/"status.json"); require(journal.get("config") == config,"PCG journal identity differs")
            for name in ("cumulative_solve_wall_seconds","iterations_executed_total","K_applications","cumulative_setup_read_seconds"):
                state[name] = max(state.get(name,0),journal.get(name,0))
        base._sync(torch); state["cumulative_setup_read_seconds"] += time.monotonic()-began
        base_seconds,base_K = state["cumulative_solve_wall_seconds"],state["K_applications"]
        solve_began = time.monotonic(); preconditioner_seconds = 0.; checkpoint_seconds = 0.

        def elapsed():
            base._sync(torch)
            return base_seconds+time.monotonic()-solve_began

        def reduced_apply(value):
            physical = torch.zeros_like(inputs["vertices_mm"]); physical[free] = value
            return operator.apply(physical)[free]

        def precondition(value):
            nonlocal preconditioner_seconds
            base._sync(torch); tick = time.monotonic()
            result = torch.cholesky_solve(value[...,None],chol).squeeze(-1)
            base._sync(torch); preconditioner_seconds += time.monotonic()-tick
            return result

        def account(phase):
            state.update(cumulative_solve_wall_seconds=elapsed(),K_applications=base_K+operator.calls)
            base.write_json(output/"status.json",{**state,"phase":phase,"trace":None})
            clock.account(phase,record["key"])

        def save():
            nonlocal state,checkpoint_seconds
            tick = time.monotonic(); account("solving")
            state = base._save_pcg(np,output,state,z,r,p)
            checkpoint_seconds += time.monotonic()-tick; account("solving")

        true_r = b-reduced_apply(z); true_relative = base._relative(torch,true_r,norm_b)
        if not torch.isfinite(true_r).all() or true_relative is None:
            raise base.ScientificNegative("Nonfinite/undefined initial or resumed true residual")
        state["trace"].append({"iteration":state["iteration"],"true_relative_residual":true_relative,
            "cumulative_solve_wall_seconds":elapsed(),"reason":"initial_or_resume_true_verification"})
        if state["iteration"] == 0:
            r = true_r.clone(); preconditioned = precondition(r); rz = float((r*preconditioned).sum()); p = preconditioned.clone()
        else:
            rz = state.get("rz")
            if rz is None:
                rz = float((r*precondition(r)).sum())
        state["rz"] = rz
        numerical_state = "converged" if true_relative <= 1e-8 else "solving"
        if numerical_state == "solving" and (not math.isfinite(rz) or rz <= 0):
            raise base.ScientificNegative("Initial/resumed rTMinvr is nonfinite/nonpositive")
        save()  # Commit zero/start or resumed recurrence before an attempted K step.
        while numerical_state == "solving":
            if state["iterations_executed_total"] >= 10000 or elapsed() >= 300 or clock.remaining() <= 0:
                numerical_state = "iteration_budget_limit" if state["iterations_executed_total"] >= 10000 else (
                    "wall_budget_limit" if elapsed() >= 300 else "stage_wall_budget_limit")
                break
            state["iterations_executed_total"] += 1; account("solving")
            Ap = reduced_apply(p); pAp = float((p*Ap).sum())
            if not math.isfinite(pAp) or pAp <= 0 or not torch.isfinite(Ap).all():
                raise base.ScientificNegative("PCG pTAp nonfinite/nonpositive; no alternate solver or shift")
            alpha = rz/pAp; z = z+alpha*p; r = r-alpha*Ap; state["iteration"] += 1
            if not torch.isfinite(z).all() or not torch.isfinite(r).all():
                raise base.ScientificNegative("Nonfinite PCG solution/recursive residual")
            recursive = base._relative(torch,r,norm_b)
            if state["iteration"] % 25 == 0 or (recursive is not None and recursive <= 1e-8):
                true_r = b-reduced_apply(z); true_relative = base._relative(torch,true_r,norm_b)
                if not torch.isfinite(true_r).all() or true_relative is None:
                    raise base.ScientificNegative("Nonfinite/undefined true residual")
                state["trace"].append({"iteration":state["iteration"],"true_relative_residual":true_relative,
                    "recursive_relative_residual":recursive,"cumulative_solve_wall_seconds":elapsed()})
                if true_relative <= 1e-8:
                    numerical_state = "converged"
            if numerical_state == "solving":
                preconditioned = precondition(r); next_rz = float((r*preconditioned).sum())
                if not math.isfinite(next_rz) or next_rz <= 0:
                    raise base.ScientificNegative("PCG rTMinvr nonfinite/nonpositive")
                p = preconditioned+(next_rz/rz)*p; rz = next_rz; state["rz"] = rz
            account("solving")
            if state["iteration"] % 25 == 0:
                save()
        true_r = b-reduced_apply(z); true_relative = base._relative(torch,true_r,norm_b)
        if not torch.isfinite(true_r).all() or true_relative is None:
            raise base.ScientificNegative("Nonfinite/undefined endpoint residual")
        if true_relative <= 1e-8:
            numerical_state = "converged"
        state["trace"].append({"iteration":state["iteration"],"true_relative_residual":true_relative,
            "cumulative_solve_wall_seconds":elapsed(),"reason":"endpoint_true_verification"})
        save(); solve_seconds = elapsed(); solve_K_count = operator.calls; solve_K_seconds = operator.apply_seconds
        verify_began = time.monotonic(); physical = base._restore_full(torch,z,inputs)
        residual = operator.apply(physical)-inputs["force"]
        reaction = torch.zeros_like(residual); reaction[inputs["fixed_node_ids"]] = residual[inputs["fixed_node_ids"]]
        external = inputs["force"]+reaction; reference = inputs["reference_xyz_mm"]
        net_force = external.sum(dim=0); net_moment = torch.cross(inputs["vertices_mm"]-reference,external,dim=-1).sum(dim=0)
        if any(not torch.isfinite(value).all() for value in (physical,residual,net_force,net_moment)):
            raise base.ScientificNegative("Nonfinite physical endpoint/equilibrium")
        prediction = base._save_array(np,output/"physical_u_mm.npy",physical)
        reaction_record = base._save_array(np,output/"constraint_reaction_N.npy",reaction)
        grid_record = base._save_array(np,output/"source_grid_node_ids.npy",inputs["grid_node_ids"])
        equilibrium = {"true_free_residual_relative":base._relative(torch,residual[free],norm_b),
            "true_free_residual_norm_N":float(torch.linalg.vector_norm(residual[free])),
            "constraint_reaction_sum_N":reaction.sum(dim=0).tolist(),"net_force_with_reaction_N":net_force.tolist(),
            "net_moment_with_reaction_Nmm":net_moment.tolist(),"fixed_translation_max_abs_mm":float(physical[inputs["fixed_node_ids"]].abs().max()),
            "reference_xyz_mm":reference.tolist(),"full_K_verification_calls":1}
        base._sync(torch); verification_seconds = time.monotonic()-verify_began
        summary = {**case_base(record),"status":"completed","kind":"elastic_crosscad_case","config":config,
            "attempted":True,"input_compatible":True,"operator_compatible":True,"numerical_state":numerical_state,
            "numerically_converged":numerical_state == "converged","iteration":state["iteration"],
            "iterations_executed_total":state["iterations_executed_total"],"true_relative_residual":true_relative,"trace":state["trace"],
            "prediction":prediction,"physical_u_path":prediction["path"],"constraint_reaction":reaction_record,
            "source_grid_node_ids":grid_record,"equilibrium":equilibrium,
            "costs":{"input_preparation_seconds":manifest["preparation_seconds"],"operator_wall_seconds":operator_record["costs"]["operator_wall_seconds"],
                "solve_setup_read_seconds":state["cumulative_setup_read_seconds"],"cumulative_solve_wall_seconds":solve_seconds,
                "this_call_solve_wall_seconds":solve_seconds-base_seconds,"this_call_K_apply_seconds":solve_K_seconds,
                "preconditioner_seconds":preconditioner_seconds,"checkpoint_seconds":checkpoint_seconds,
                "endpoint_verification_seconds":verification_seconds,"GT_read_and_metrics_seconds":0.,
                "solve_K_count_this_call":solve_K_count,"solve_K_count_cumulative":base_K+solve_K_count,
                "verification_K_count":1,"budget_overshoot_seconds":max(0.,solve_seconds-300)},
            "memory":{name:max(base._memory(torch)[name],operator_record.get("memory",{}).get(name,0)) for name in ("peak_allocated_bytes","peak_reserved_bytes")},
            "input_manifest":str(host_path(record["data_output"])/"manifest.json"),
            "operator_summary":str(host_path(record["operator_output"])/"summary.json"),
            "read_receipt":{**zero_receipt(),"finite_allN_prediction_and_equilibrium_persisted_before_GT":True,
                "operator_K_guards_this_CAD":8,"operator_rebuilds_in_solve":0}}
        base.write_json(output/"endpoint_no_GT.json",{"summary_before_labels":summary,"read_receipt":zero_receipt()})
        clock.account("finite_endpoint_persisted",record["key"])
        return finish_endpoint(record,output,clock)
    except base.ScientificNegative as error:
        if state is not None and solve_began is not None:
            state.update(cumulative_solve_wall_seconds=elapsed(),K_applications=base_K+operator.calls)
            base.write_json(output/"status.json",{**state,"trace":None,"phase":"completed","numerical_state":"breakdown"})
        summary = {**case_base(record),"status":"completed","kind":"elastic_crosscad_case","config":config,
            "attempted":True,"input_compatible":True,"operator_compatible":True,
            "numerical_state":"stage_wall_budget_limit" if str(error) == "stage_wall_budget_exhausted" else "breakdown","reason":str(error),
            "costs":{"input_preparation_seconds":manifest["preparation_seconds"],"operator_wall_seconds":operator_record["costs"]["operator_wall_seconds"],
                "cumulative_solve_wall_seconds":0. if state is None else state["cumulative_solve_wall_seconds"],
                "cold_case_accounted_wall_seconds":manifest["preparation_seconds"]+operator_record["costs"]["operator_wall_seconds"]
                    +(time.monotonic()-began)}}
        base.write_json(output/"summary.json",summary); clock.account("case_negative",record["key"])
        return summary
    except Exception:
        if state is not None and solve_began is not None:
            state.update(cumulative_solve_wall_seconds=elapsed(),K_applications=base_K+operator.calls)
            base.write_json(output/"status.json",{**state,"trace":None,"phase":"failed"})
        clock.account("case_failed",record["key"])
        raise


def run(plan_path=PLAN,output_path=None):
    began = time.monotonic(); contract,plan = request_contract(plan_path)
    require(plan.get("scientific_rules_frozen") is True and plan.get("actual_execution_released") is True,
        "Root must freeze and release the fixed five-CAD stage before scientific imports")
    require(set(plan.get("source_sha256",{})) == set(contract["source_sha256"]),"Actual stage needs all four frozen source pins")
    output = host_path(output_path or plan["output"])
    require(output == host_path(plan["output"]),"Stage output differs from frozen destination")
    from . import elastic_pilot as base
    config = {"contract":contract,"stage":KIND}
    if (output/"config.json").exists():
        require(read_json(output/"config.json") == config,"Existing screen contract differs")
    if (output/"summary.json").exists():
        previous = read_json(output/"summary.json")
        if previous.get("status") == "completed":
            require(previous.get("config") == config,"Completed screen identity differs")
            return previous
    output.mkdir(parents=True,exist_ok=True); base.write_json(output/"config.json",config)
    clock = StageClock(output,config,began); clock.account("running")
    cases = {}
    try:
        for key in CASES:
            record = contract["records"][key]; case_output = host_path(record["solve_output"])
            cached = case_output/"summary.json"
            if cached.exists():
                result = read_json(cached)
                require(result.get("config") == {"contract":record,"stage":"elastic_crosscad_pcg"},"Committed case identity differs")
                if result.get("status") == "completed":
                    cases[key] = result; clock.account("case_reused",key); continue
            case_output.mkdir(parents=True,exist_ok=True)
            if clock.remaining() <= 0:
                result = {**case_base(record),"status":"completed","kind":"elastic_crosscad_case",
                    "config":{"contract":record,"stage":"elastic_crosscad_pcg"},"numerical_state":"stage_budget_exhausted_negative",
                    "reason":"No new CAD attempt after cumulative900s wall budget", "costs":{"cold_case_accounted_wall_seconds":0.}}
                base.write_json(cached,result); cases[key] = result; clock.account("case_not_attempted",key); continue
            manifest = prepare_input(record,host_path(record["data_output"]),clock)
            if not manifest.get("feasibility_pass"):
                result = {**case_base(record),"status":"completed","kind":"elastic_crosscad_case",
                    "config":{"contract":record,"stage":"elastic_crosscad_pcg"},"attempted":True,
                    "numerical_state":"unsupported_input_negative","reason":manifest.get("reason","Input model unsupported"),
                    "input_manifest":str(host_path(record["data_output"])/"manifest.json"),
                    "costs":{"input_preparation_seconds":manifest["preparation_seconds"],"cold_case_accounted_wall_seconds":manifest["preparation_seconds"]}}
                base.write_json(cached,result)
            else:
                operator_record = prepare_operator(record,manifest,host_path(record["operator_output"]),clock)
                if not operator_record.get("feasibility_pass"):
                    spent = manifest["preparation_seconds"]+operator_record["costs"]["operator_wall_seconds"]
                    result = {**case_base(record),"status":"completed","kind":"elastic_crosscad_case",
                        "config":{"contract":record,"stage":"elastic_crosscad_pcg"},"attempted":True,"input_compatible":True,
                        "numerical_state":"operator_infeasible_negative","reason":operator_record.get("reason","Operator guards failed"),
                        "input_manifest":str(host_path(record["data_output"])/"manifest.json"),
                        "operator_summary":str(host_path(record["operator_output"])/"summary.json"),
                        "costs":{"input_preparation_seconds":manifest["preparation_seconds"],"operator_wall_seconds":operator_record["costs"]["operator_wall_seconds"],
                            "cold_case_accounted_wall_seconds":spent}}
                    base.write_json(cached,result)
                else:
                    result = solve_case(record,manifest,operator_record,case_output,clock)
            cases[key] = result; clock.account("case_completed",key)
        receipt = {**zero_receipt(),"displacement_reads":sum(item["read_receipt"]["displacement_reads"] for item in cases.values())}
        counts = {"fixed_cases":5,"attempted":sum(item["attempted"] for item in cases.values()),
            "input_compatible":sum(item["input_compatible"] for item in cases.values()),
            "operator_compatible":sum(item["operator_compatible"] for item in cases.values()),
            "numerically_converged":sum(item["numerically_converged"] for item in cases.values()),
            "metrics_available":sum(item.get("metrics") is not None for item in cases.values()),
            "displacement_reads":receipt["displacement_reads"]}
        elapsed = clock.elapsed()
        summary = {"status":"completed","kind":KIND,"config":config,"case_keys":list(CASES),"cases":cases,
            "counts":counts,"read_receipt":receipt,"elapsed_seconds":elapsed,
            "costs":{"cumulative_stage_wall_seconds":elapsed,"maximum_stage_wall_seconds":900,
                "budget_overshoot_seconds":max(0.,elapsed-900),"this_call_stage_wall_seconds":time.monotonic()-began},
            "scientific_outcome":"fixed_bounded_screen_completed; per-case compatibility/convergence/metric availability remain separate",
            "scope":"Five fixed existing TRAIN3/DEV2 ver cases, zero start approximate physical model; not OptiStruct exact, engineering validation or learned cross-CAD generalization",
            "budget_finalization":"After expiry preserve finite endpoint/equilibrium with explicit overshoot, but no new GT read or later CAD attempt",
            "original_101479_reopened":False,"training_updates":0,"scaler_fits":0}
        base.write_json(output/"summary.json",summary); clock.account("completed")
        return summary
    except Exception as error:
        clock.account("failed")
        committed_reads = 0; uncommitted_attempts = 0
        for key in CASES:
            label_result = host_path(contract["records"][key]["solve_output"])/"label_metrics.json"
            started = host_path(contract["records"][key]["solve_output"])/"label_read_started.json"
            if label_result.exists():
                committed_reads += 1
            elif started.exists():
                claim = read_json(started)
                committed_reads += int(claim.get("member_read_completed") is True)
                uncommitted_attempts += int(claim.get("member_read_completed") is not True)
        base.write_json(output/"summary.json",{"status":"failed","kind":KIND,"config":config,"case_keys":list(CASES),"cases":cases,
            "reason":f"{type(error).__name__}: {error}","costs":{"cumulative_stage_wall_seconds":clock.elapsed()},
            "read_receipt":{**zero_receipt(),"displacement_reads":committed_reads,"interrupted_label_read_attempts":uncommitted_attempts}})
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan",type=Path,default=Path(PLAN))
    result.add_argument("--output",type=Path)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    run(args.plan,args.output)
