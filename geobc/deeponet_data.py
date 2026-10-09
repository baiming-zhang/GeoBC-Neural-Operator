"""Target-free BC22 clouds at the existing 1024 sampled FEM-node identities.

Only root's frozen preparation may read arrays. This module does not alter the
old selection cache, resample clouds, fit scalers, or open targets/test/dia.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = "notes/geobc_deeponet_plan.json"
NONFORCE_SLOTS = list(range(7)) + list(range(13,25))
ROLE_NAMES = [f"support_reference_dof_{i}" for i in range(1,7)] + ["load_interface_membership"]
SCHEMA = {"name":"geobc_geometry_bc_cloud22_v1", "cloud_shape":[1024,22],
    "cloud_dtype":"float32", "index_shape":[1024], "index_dtype":"int64",
    "index_meaning":"Zero-based row in canonical prepared xyz/features; original FEM-node ordering, never deck nodeID minus1",
    "channels":"Existing sampled canonical xyz3 then original25 slots0:7 and13:25 (non-force19)",
    "mask":"Existing unchanged bool1024; masked index=-1 and nonforce padding=0",
    "cloud_points":1024, "cloud_seed":2026, "nonforce_slots":NONFORCE_SLOTS,
    "scope":"Existing TRAIN40 + development-validation5 canonical ver inputs only",
    "role_semantics":"Six support-reference DOF membership roles plus load-interface membership; not node Dirichlet conditions or field GT",
    "identity":"Exact FP32 coordinate identity with coincident-row ambiguity rejection; original sampler replay allowed only for missing exact matches and only with exact canonical xyz/cloud/mask verification",
    "no_resampling_or_nearest_copy":True, "no_scaler_fit":True, "target_independent":True}
SOURCE_FILES = ("geobc/deeponet_data.py", "geobc/conditioned_field.py", "geobc/solver_selection_data.py")


def host_path(value):
    value = str(value).replace("\\","/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/",value):
        value = value[5].upper()+":/"+value[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/",value):
        value = "/mnt/"+value[0].lower()+"/"+value[3:]
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


def read_json(path):
    return json.loads(host_path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(host_path(path).read_bytes()).hexdigest()


def stat_identity(path):
    path = host_path(path); stat = path.stat()
    return {"path":path.relative_to(ROOT).as_posix(), "bytes":stat.st_size, "mtime_ns":stat.st_mtime_ns}


def _write(path,value):
    path = host_path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def _save(np,path,value):
    path = host_path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as handle:
        np.save(handle,value,allow_pickle=False)
    temporary.replace(path)


def _rules(plan,actual=False):
    for key,value in (("existing_split","notes/geobc_split.json"),
        ("prepared_data","data/prepared_geobc_multiload"),
        ("data_cache","data/geobc_solver_selection"),
        ("bc_cloud_cache","data/geobc_geometry_bc_clouds")):
        if host_path(plan[key]) != host_path(value):
            raise ValueError(f"DeepONet BC cloud source/destination changed: {key}")
    recipe = plan["recipe"]
    if recipe.get("cloud_points") != 1024 or recipe.get("cloud_seed") != 2026 or recipe.get("cloud_nonforce_slots") != NONFORCE_SLOTS or recipe.get("train_loads") != ["ver","hor","tor"]:
        raise ValueError("Use exactly existing seed2026/cloud1024/nonforce19 known3 input protocol")
    if actual and plan.get("training_rules_frozen") is not True:
        raise ValueError("Root must freeze the paired solver/input protocol before BC cloud preparation")


def request_contract(plan_path=DEFAULT_PLAN):
    """Pure stdlib JSON/stat/source SHA; no array or scientific library import."""
    plan = read_json(plan_path); _rules(plan)
    split = read_json(plan["existing_split"]); groups = split["groups"]
    if [len(groups[name]) for name in ("train","validation","test")] != [40,5,5] or any(set(groups[a])&set(groups[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))):
        raise ValueError("Expected the fixed disjoint whole-CAD40/5/5 split")
    old_manifest_path = host_path(plan["data_cache"])/"manifest.json"
    old = read_json(old_manifest_path)
    train_keys = [f"{cad}/{load}" for cad in groups["train"] for load in ("ver","hor","tor")]
    validation_keys = [f"{cad}/{load}" for cad in groups["validation"] for load in ("ver","hor","tor")]
    if old.get("status") != "completed" or old["train_keys"] != train_keys or old["validation_keys"] != validation_keys or old["contract"]["cloud_seed"] != 2026 or old["contract"]["cloud_points"] != 1024 or old["contract"]["split_sha256"] != sha(plan["existing_split"]):
        raise ValueError("Original completed selection cloud cache does not match the fixed known3 CADs/sampling")
    old_sources = {row["key"]:row for row in old["contract"]["sources"]}
    records = {}
    for group in ("train","validation"):
        for cad in groups[group]:
            key = f"{cad}/ver"; old_record = old["records"][key]
            source = host_path(plan["prepared_data"])/f"{cad}_ver.npz"
            if old_record["group"] != group or old_record["cad_id"] != cad or old_record.get("label_alignment") != "official CSV GRID coordinates -> H5 vertices" or host_path(split["case_paths"][key]) != source:
                raise ValueError("Canonical source is outside the selected TRAIN/DEV ver inputs")
            source_stat = stat_identity(source); original = old_sources[key]
            if host_path(original["path"]) != source or any(source_stat[name] != original[name] for name in ("bytes","mtime_ns")):
                raise ValueError("Canonical prepared input changed since original selection cloud construction")
            record = {"cad_id":cad,"group":group,"canonical_key":key,
                "num_nodes":old_record["num_nodes"], "source_npz_path":str(source)}
            files = {}
            for name in ("xyz","cloud","mask"):
                expected = host_path(plan["data_cache"])/"geometry"/cad/f"{name}.npy"
                if host_path(old_record[f"{name}_path"]) != expected:
                    raise ValueError("Unexpected original geometry cache path")
                record[f"{name}_path"] = str(expected); files[name] = stat_identity(expected)
            features = host_path(plan["data_cache"])/"cases"/f"{cad}_ver"/"features.npy"
            if host_path(old_record["features_path"]) != features:
                raise ValueError("Unexpected canonical input feature cache path")
            record["features_path"] = str(features); files["features"] = stat_identity(features)
            cloud_meta_path = host_path(plan["data_cache"])/"geometry"/cad/"cloud_metadata.json"
            meta = read_json(cloud_meta_path)
            if meta["cad_key"] != cad or host_path(meta["source"]) != source or meta["seed"] != 2026 or meta["cloud_points"] != 1024 or meta.get("target_independent") is not True or meta.get("all_loads_share_one_cad_cloud") is not True or not (1 <= meta["selected_nodes"] <= 1024):
                raise ValueError("Original cloud metadata does not support unchanged canonical sampling")
            record["sampling_metadata"] = meta
            record["files"] = {**files,"source_npz":source_stat,"cloud_metadata":{**stat_identity(cloud_meta_path),"sha256":sha(cloud_meta_path)}}
            records[cad] = record
    sources = {name:sha(name) for name in SOURCE_FILES}
    for name in SOURCE_FILES:
        expected = plan.get("source_sha256",{}).get(name)
        if expected is not None and expected != sources[name]:
            raise ValueError(f"Frozen BC cloud source changed: {name}")
    contract = {"schema":SCHEMA,"source_sha256":sources,
        "split_sha256":sha(plan["existing_split"]),"selection_manifest_sha256":sha(old_manifest_path),
        "cache":str(host_path(plan["bc_cloud_cache"])),
        "train_CADs":groups["train"],"validation_CADs":groups["validation"],
        "records":records,"input_read_policy":{"arrays":"Canonical ver cached xyz/features/cloud/mask only; fallback source NPZ xyz/faces/sample_key only","target_members":[],"test_or_dia_archives":0,"nearest_or_resampling":False},
        "identity_algorithm":"exact sorted FP32 rows; reject any multiple row match; missing only may replay pinned original sampler, never tolerance search"}
    return contract, plan


def read_cache_metadata(plan_path=DEFAULT_PLAN,*,contract=None):
    """Strict completed reuse via small JSON/stat; no cached-array reads."""
    if contract is None:
        contract,plan = request_contract(plan_path)
    else:
        plan = read_json(plan_path); _rules(plan)
    manifest = read_json(host_path(plan["bc_cloud_cache"])/"manifest.json")
    if manifest.get("status") != "completed" or manifest.get("contract") != contract or set(manifest["records"]) != set(contract["records"]):
        raise ValueError("BC22 input cache is incomplete or differs from current source/input contract")
    for cad,record in manifest["records"].items():
        for name in ("indices","cloud_bc22"):
            expected = host_path(plan["bc_cloud_cache"])/cad/f"{name}.npy"
            if host_path(record[f"{name}_path"]) != expected or stat_identity(expected) != record["output_files"][name]:
                raise ValueError("Prepared BC22 indices/cloud file identity changed")
        if host_path(record["mask_path"]) != host_path(contract["records"][cad]["mask_path"]):
            raise ValueError("BC22 cloud must reuse the original mask path")
    return manifest


def _source_replay(np,record,xyz,cloud,mask):
    """Exact copy of conditioned_field.load_geometry_cloud's pinned sampler.

    It is only a missing-match fallback, never a coincident-node tie breaker.
    No alternate normalization, nearest node, tolerance, new seed or sample.
    """
    meta = record["sampling_metadata"]
    with np.load(record["source_npz_path"],allow_pickle=False) as archive:
        source_xyz = np.asarray(archive["xyz"],np.float32)
        key = str(archive["sample_key"].item())
        faces = np.asarray(archive["faces"]) if "faces" in archive.files else None
    if key != record["canonical_key"] or not np.array_equal(source_xyz,xyz):
        raise ValueError("Original sampler source xyz is not the unchanged canonical cache; rounding cannot be guessed")
    if faces is not None and faces.size:
        if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype,np.integer):
            raise ValueError("Original boundary replay needs actual integer faces[F,3]")
        candidates = np.unique(faces)
        if candidates[0] < 0 or candidates[-1] >= len(xyz):
            raise ValueError("Boundary indices do not address original prepared rows")
        pool = "unique original quadratic-boundary node IDs"
    else:
        candidates = np.arange(len(xyz),dtype=np.int64)
        pool = "all original FEM nodes; boundary unavailable"
    count = min(1024,len(candidates))
    seed = int.from_bytes(hashlib.blake2b(f"2026:{record['cad_id']}".encode("utf-8"),digest_size=8).digest(),"little")
    chosen = np.random.default_rng(seed).choice(candidates,count,replace=False)
    points = np.zeros((1024,3),np.float32); replay_mask = np.zeros(1024,np.bool_)
    indices = np.full(1024,-1,np.int64)
    points[:count], replay_mask[:count], indices[:count] = xyz[chosen],True,chosen
    if meta["candidate_nodes"] != len(candidates) or meta["pool"] != pool or meta["selected_nodes"] != count or not np.array_equal(mask,replay_mask) or not np.array_equal(cloud,points):
        raise ValueError("Exact original chosen-index replay does not reproduce the cached cloud/mask")
    return indices


def _recover_indices(np,record,xyz,cloud,mask):
    unique, first, counts = np.unique(xyz,axis=0,return_index=True,return_counts=True)
    row_dtype = np.dtype([(name,"<f4") for name in ("x","y","z")])
    ordered = np.ascontiguousarray(unique,dtype="<f4").view(row_dtype).reshape(-1)
    query = np.ascontiguousarray(cloud[mask],dtype="<f4").view(row_dtype).reshape(-1)
    positions = np.searchsorted(ordered,query)
    within = positions < len(unique); found = np.zeros(len(query),np.bool_)
    found[within] = np.all(unique[positions[within]] == cloud[mask][within],axis=1)
    if np.any(counts[positions[found]] > 1):
        ambiguous = np.flatnonzero(found & np.asarray([counts[p] > 1 if p < len(counts) else False for p in positions]))
        raise ValueError(f"Ambiguous exact sampled coordinate maps to multiple original FEM rows; first_cloud_slots={np.flatnonzero(mask)[ambiguous[:8]].tolist()}")
    if found.all():
        indices = np.full(1024,-1,np.int64); indices[mask] = first[positions]
        method = "exact_FP32_coordinate_identity"
        fallback_members = []
    else:
        indices = _source_replay(np,record,xyz,cloud,mask)
        method = "exact_original_boundary_sampler_replay"
        fallback_members = ["xyz","sample_key","faces_if_present"]
    if len(np.unique(indices[mask])) != int(mask.sum()) or not np.array_equal(xyz[indices[mask]],cloud[mask]):
        raise ValueError("Recovered original row identities do not reproduce every sampled point exactly")
    return indices, {"method":method,"exact_coordinate_matches":int(found.sum()),
        "selected_points":int(mask.sum()),"fallback_input_members":fallback_members,
        "max_abs_coordinate_difference":0.,"coordinate_tolerance":0.,"ambiguity_allowed":False,
        "sampling_changed":False,"source_row_order_preserved":True}


def _prepare_one(np,record,directory):
    began = time.perf_counter()
    arrays = {name:np.load(record[f"{name}_path"],allow_pickle=False) for name in ("xyz","features","cloud","mask")}
    xyz,features,cloud,mask = (arrays[name] for name in ("xyz","features","cloud","mask"))
    if xyz.shape != (record["num_nodes"],3) or features.shape != (record["num_nodes"],25) or cloud.shape != (1024,3) or mask.shape != (1024,) or xyz.dtype != np.float32 or features.dtype != np.float32 or cloud.dtype != np.float32 or mask.dtype != np.bool_:
        raise ValueError("Canonical BC22 input schema/dtype differs from original FP32 cache")
    if not all(np.isfinite(value).all() for value in (xyz,features,cloud)):
        raise ValueError("Nonfinite canonical input; no sanitization")
    count = record["sampling_metadata"]["selected_nodes"]
    expected_mask = np.arange(1024) < count
    if not np.array_equal(mask,expected_mask) or not np.all(cloud[~mask] == 0):
        raise ValueError("Original selected-prefix mask/padded point identity changed")
    indices, identity = _recover_indices(np,record,xyz,cloud,mask)
    cloud22 = np.zeros((1024,22),np.float32); cloud22[:,:3] = cloud
    cloud22[mask,3:] = features[indices[mask]][:,NONFORCE_SLOTS]
    if not np.array_equal(cloud22[:,:3],cloud) or not np.array_equal(cloud22[mask,3:],features[indices[mask]][:,NONFORCE_SLOTS]):
        raise ValueError("BC22 gather did not preserve original point coordinates/features")
    roles = features[:,:7]
    if not np.all((roles == 0) | (roles == 1)):
        raise ValueError("Canonical role membership channels are not source binary flags; no clamping")
    role_coverage = {}
    for i,name in enumerate(ROLE_NAMES):
        source_count = int((roles[:,i] == 1).sum())
        selected_count = int((roles[indices[mask],i] == 1).sum())
        role_coverage[name] = {"source_positive_nodes":source_count,"sampled_positive_nodes":selected_count,
            "sampled_active_nodes":int(mask.sum()),"sampled_positive_fraction":selected_count/int(mask.sum()),
            "fraction_of_source_positive_nodes":selected_count/source_count if source_count else None,
            "represented_in_cloud":bool(selected_count),"source_absent":source_count == 0}
    indices_path = directory/"indices.npy"; cloud_path = directory/"cloud_bc22.npy"
    _save(np,indices_path,indices); _save(np,cloud_path,cloud22)
    return {"cad_id":record["cad_id"],"group":record["group"],"canonical_key":record["canonical_key"],
        "num_nodes":len(xyz),"indices_path":str(indices_path),"cloud_bc22_path":str(cloud_path),
        "mask_path":record["mask_path"],"schema":SCHEMA,"identity":identity,"role_coverage":role_coverage,
        "coordinate_range":{"min":float(cloud[mask].min()),"max":float(cloud[mask].max()),"clamped":False},
        "array_identity_sha256":{"canonical_xyz_FP32":hashlib.sha256(xyz.tobytes(order="C")).hexdigest(),
            "original_cloud_FP32":hashlib.sha256(cloud.tobytes(order="C")).hexdigest(),
            "original_mask_bool":hashlib.sha256(mask.tobytes(order="C")).hexdigest(),
            "recovered_indices_int64":hashlib.sha256(indices.tobytes(order="C")).hexdigest(),
            "gathered_cloud_bc22_FP32":hashlib.sha256(cloud22.tobytes(order="C")).hexdigest()},
        "elapsed_seconds":time.perf_counter()-began,
        "source_files":record["files"],"output_files":{"indices":stat_identity(indices_path),"cloud_bc22":stat_identity(cloud_path)}}


def prepare_data(plan_path=DEFAULT_PLAN):
    started = time.perf_counter(); contract,plan = request_contract(plan_path); _rules(plan,actual=True)
    cache = host_path(plan["bc_cloud_cache"])
    if (cache/"manifest.json").exists():
        return read_cache_metadata(plan_path,contract=contract)
    if (cache/"config.json").exists() and read_json(cache/"config.json") != {"contract":contract}:
        raise ValueError("Existing BC cloud preparation config has a different contract")
    for name in SOURCE_FILES:
        if plan.get("source_sha256",{}).get(name) != contract["source_sha256"][name]:
            raise ValueError("Root must pin the final BC cloud source and original sampling/data helpers")
    _write(cache/"config.json",{"contract":contract})
    _write(cache/"status.json",{"status":"running","kind":"deeponet_bc_cloud_input","completed_CADs":0,"target_arrays_read":0})
    try:
        import numpy as np
        records = {}
        for cad,record in contract["records"].items():
            records[cad] = _prepare_one(np,record,cache/cad)
            _write(cache/"status.json",{"status":"running","kind":"deeponet_bc_cloud_input","completed_CADs":len(records),"current_CAD":cad,"elapsed_seconds":time.perf_counter()-started,"target_arrays_read":0})
        current_contract,_ = request_contract(plan_path)
        if current_contract != contract:
            raise ValueError("Canonical input/source identity changed during BC cloud preparation")
        coverage = {name:{"CADs_with_source_role":sum(not row["role_coverage"][name]["source_absent"] for row in records.values()),
            "CADs_with_sampled_role":sum(row["role_coverage"][name]["represented_in_cloud"] for row in records.values()),
            "source_positive_nodes":sum(row["role_coverage"][name]["source_positive_nodes"] for row in records.values()),
            "sampled_positive_nodes":sum(row["role_coverage"][name]["sampled_positive_nodes"] for row in records.values())} for name in ROLE_NAMES}
        manifest = {"status":"completed","kind":"deeponet_bc_cloud_input","contract":contract,"schema":SCHEMA,
            "records":records,"role_coverage":coverage,"preparation_seconds":time.perf_counter()-started,
            "read_receipt":{"canonical_CADs":45,"TRAIN_CADs":40,"DEV_CADs":5,"cached_input_members":["xyz","features","cloud","mask"],"fallback_CADs":[cad for cad,row in records.items() if row["identity"]["fallback_input_members"]],"target_arrays_read":0,"weight_arrays_read":0,"scaler_fits":0,"test_or_dia_archives_read":0,"GPU_runs":0,"old_cache_files_written":0,"new_array_files":90,"nearest_queries":0,"sampling_changes":0},
            "limitations":["Unchanged uniform original-node sample can omit rare BC roles; coverage is descriptive, never fixed by resampling","Role flags describe source reference/interface membership, not point Dirichlet conditions","No learned-field accuracy, geometry regularity, continuum BC enforcement or blind-final evidence from this cache"],
            "completed_at_utc":datetime.now(timezone.utc).isoformat()}
        _write(cache/"manifest.json",manifest)
        _write(cache/"status.json",{"status":"completed","kind":"deeponet_bc_cloud_input","completed_CADs":45,"elapsed_seconds":manifest["preparation_seconds"],"target_arrays_read":0})
        print(json.dumps({"status":"completed","CADs":45,"seconds":manifest["preparation_seconds"],"fallback_CADs":len(manifest["read_receipt"]["fallback_CADs"]),"role_coverage":coverage}))
        return manifest
    except Exception as error:
        _write(cache/"status.json",{"status":"failed","kind":"deeponet_bc_cloud_input","elapsed_seconds":time.perf_counter()-started,"error_type":type(error).__name__,"error":str(error),"target_arrays_read":0})
        raise


def load_bc_cloud(record,device):
    """Trainer adds this cloud/mask to its unchanged original-node case loader."""
    import numpy as np
    import torch
    cloud = np.load(host_path(record["cloud_bc22_path"]),allow_pickle=False)
    mask = np.load(host_path(record["mask_path"]),allow_pickle=False)
    if cloud.shape != (1024,22) or cloud.dtype != np.float32 or mask.shape != (1024,) or mask.dtype != np.bool_ or not mask.any() or not np.isfinite(cloud).all():
        raise ValueError("BC22 runtime cloud/mask schema/nonfinite failure")
    return {"cloud":torch.as_tensor(cloud,device=device)[None],"mask":torch.as_tensor(mask,device=device)[None]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",default=DEFAULT_PLAN); parser.add_argument("--source-check",action="store_true")
    args = parser.parse_args()
    if args.source_check:
        ast.parse(Path(__file__).read_text(encoding="utf-8"))
        contract,plan = request_contract(args.plan)
        print(json.dumps({"status":"source_check_passed","CADs":len(contract["records"]),"cache_exists":(host_path(plan["bc_cloud_cache"])/"manifest.json").exists(),"arrays_read":0,"scientific_imports":False,"GPU_runs":0}))
    else:
        prepare_data(args.plan)


if __name__ == "__main__":
    main()
