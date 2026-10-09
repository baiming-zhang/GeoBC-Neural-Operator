"""Fixed target-free PointConv geometry at the original 45 CAD node rows.

Only the root-frozen preparation runs geometry algorithms. Module import and
request_contract use stdlib metadata; no scientific library is imported there.
This cache changes neither the existing case inputs nor the original cloud.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = "notes/geobc_pointconv_plan.json"
DESIGN = "notes/geobc_pointconv_candidate_design.md"
DESIGN_SHA = "0b01738c44cc3ca3472fc9f36a77300adac0144438ff5509f626cd0be45519e2"
SOURCE = "geobc/pointconv_data.py"
SCHEMA = {
    "name":"geobc_pointconv_geometry_v1", "kind":"pointconv_input",
    "scope":"TRAIN40/development-validation5 canonical ver; original full node rows",
    "fps_nodes":[2048,128], "neighbors32":32, "query_block":2048,
    "FPS":"pure Torch CUDA FP64 running minimum squared distances; smallest original row first; max-distance ties smallest original row; no loop host scalar synchronization",
    "FPS_failure":"zero maximum remaining distance before quota: completed scientific negative; no duplicate anchor or alternative quota",
    "neighbors":"self row then nearest31 different source rows; exact FP64 distance/original-row ordering; exact cutoff ties completed by radius search",
    "coordinates":"existing canonical FP32 xyz promoted FP64 for searches, FPS and KDE; save FP32; no new normalization",
    "density":"source-level sigma=median d32; p=mean32 exp(-d2/(2sigma2)); self included; sigma<=0 negative, no floor",
    "interpolation":"exact nearest3 by distance/original row; zero-distance neighbors alone equal weighted; otherwise normalized1/d without floor",
    "delta_direction":"source_xyz-query_xyz, existing normalized coordinates, no second /L",
    "stride":"neighbors01=neighbors00[anchors01]; neighbors12=neighbors11[anchors12]",
    "runtime":"xyz0/1/2,p0/1/2 float32; neighbors00/01/11/12/22 and interp21/10_indices int64; interp21/10_weights float32",
    "anchors":"zero-based original row indices, never GRID_ID minus1; coarse rows retain FPS order",
    "cache_indices":"int32 neighbor/interpolation tables; int64 anchors and original coarse row IDs",
    "input_array_allowlist":["canonical xyz.npy"],
    "identity_only":["canonical features.npy","original cloud.npy","original mask.npy","canonical prepared archive stat"],
    "coverage":"geometry row/source participation only; BC role coverage unmeasured, no preservation claim",
    "FP32_interpolation_partition_tolerance":2e-7,
    "cost_semantics":"build_seconds includes FPS/knn/KDE/interpolation; elapsed_seconds is measured CAD wall, not sum of overlapping cost fields",
    "target_independent":True, "labels_read":0, "scaler_fits":0,
    "all_point_KDE":False, "topology_or_Dirichlet_claim":False,
    "latent_activations_cached":False,
}


def _path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", value):
        value = value[5].upper()+":/"+value[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", value):
        value = "/mnt/"+value[0].lower()+"/"+value[3:]
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


def _json(path):
    return json.loads(_path(path).read_text(encoding="utf-8-sig"))


def _sha(path):
    hasher = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _identity(path, *, sha=False):
    path = _path(path); stat = path.stat()
    result = {"path":path.relative_to(ROOT).as_posix(),"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns}
    if sha:
        result["sha256"] = _sha(path)
    return result


def _match(path, identity):
    actual = _identity(path)
    if actual["bytes"] != identity["bytes"] or actual["mtime_ns"] != identity["mtime_ns"]:
        raise ValueError(f"Input/cache file stat changed: {actual['path']}")
    if "path" in identity and _path(identity["path"]) != _path(path):
        raise ValueError("Input/cache path differs from its pinned identity")


def _write(path, value):
    path = _path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def _require(predicate, message):
    if not predicate:
        raise ValueError(message)


def _rules(plan, actual=False):
    expected = {"existing_split":"notes/geobc_split.json", "prepared_data":"data/prepared_geobc_multiload",
        "data_cache":"data/geobc_solver_selection", "pointconv_input_cache":"data/geobc_pointconv_inputs45",
        "design":DESIGN}
    for key,value in expected.items():
        _require(_path(plan[key]) == _path(value),f"PointConv source/destination changed: {key}")
    _require(plan.get("design_sha256") == DESIGN_SHA and _sha(DESIGN) == DESIGN_SHA,"PointConv proposal identity differs")
    recipe = plan["recipe"]
    _require(recipe.get("fps_nodes") == [2048,128] and recipe.get("neighbors32") == 32
        and recipe.get("query_block") == 2048 and recipe.get("cloud_points") == 1024
        and recipe.get("cloud_seed") == 2026 and recipe.get("train_loads") == ["ver","hor","tor"],
        "Fixed PointConv geometry/cloud/known3 recipe changed")
    if actual:
        _require(plan.get("training_rules_frozen") is True,"Root must freeze the protocol before actual geometry preparation")


def request_contract(plan_path=DEFAULT_PLAN):
    """Pure stdlib JSON/source SHA/file stat, including pending-cache preparation."""
    plan = _json(plan_path); _rules(plan)
    split = _json(plan["existing_split"]); groups = split["groups"]
    _require([len(groups[name]) for name in ("train","validation","test")] == [40,5,5]
        and not any(set(groups[a])&set(groups[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))),
        "Expected frozen disjoint whole-CAD40/5/5 split")
    manifest_path = _path(plan["data_cache"])/"manifest.json"
    old = _json(manifest_path)
    keys = {group:[f"{cad}/{load}" for cad in groups[group] for load in ("ver","hor","tor")]
        for group in ("train","validation")}
    _require(old.get("status") == "completed" and old["train_keys"] == keys["train"]
        and old["validation_keys"] == keys["validation"]
        and old["contract"]["cloud_seed"] == 2026 and old["contract"]["cloud_points"] == 1024
        and old["contract"]["split_sha256"] == _sha(plan["existing_split"]),
        "Original completed45 selection cache identity differs")
    old_sources = {row["key"]:row for row in old["contract"]["sources"]}
    inputs = {}
    for group in ("train","validation"):
        for cad in groups[group]:
            key = f"{cad}/ver"; row = old["records"][key]
            prepared = _path(plan["prepared_data"])/f"{cad}_ver.npz"
            _require(row["key"] == key and row["cad_id"] == cad and row["group"] == group
                and row.get("label_alignment") == "official CSV GRID coordinates -> H5 vertices"
                and _path(split["case_paths"][key]) == prepared,"Canonical original-node source/scope differs")
            _match(prepared,old_sources[key])
            canonical = {}
            for name in ("xyz","features","cloud","mask"):
                source = _path(row[name+"_path"])
                base = _path(plan["data_cache"])/("cases" if name == "features" else "geometry")
                expected = base/(f"{cad}_ver" if name == "features" else cad)/(name+".npy")
                _require(source == expected,"Only the existing canonical xyz/features/cloud/mask cache is allowed")
                canonical[name] = _identity(source)
            _require(isinstance(row["num_nodes"],int) and 0 < row["num_nodes"] < 2**31,"Unsupported source row count")
            inputs[cad] = {"cad_id":cad,"group":group,"key":key,"num_nodes":row["num_nodes"],
                "label_alignment":row["label_alignment"],"prepared_identity_only":_identity(prepared),
                "canonical":canonical}
    source_sha = _sha(SOURCE)
    _require(plan.get("source_sha256",{}).get(SOURCE,source_sha) == source_sha,"Pinned PointConv data source changed")
    contract = {"kind":"pointconv_input","schema":SCHEMA,"source_sha256":{SOURCE:source_sha},
        "design_sha256":DESIGN_SHA,"split_sha256":_sha(plan["existing_split"]),
        "selection_manifest_sha256":_sha(manifest_path),"CAD_order":groups["train"]+groups["validation"],
        "train_CADs":groups["train"],"validation_CADs":groups["validation"],"inputs":inputs,
        "cache":_path(plan["pointconv_input_cache"]).relative_to(ROOT).as_posix(),
        "labels_read":0,"scaler_fits":0,"old_final_or_sealed_or_dia_inputs":0,"raw_sources_opened":False}
    return contract,plan


def _manifest(plan, contract, *, require_feasible=True):
    cache = _path(plan["pointconv_input_cache"])
    manifest = _json(cache/"manifest.json")
    _require(manifest.get("status") == "completed" and manifest.get("contract") == contract,
        "PointConv cache incomplete or contract changed")
    if require_feasible:
        _require(manifest.get("feasibility_pass") is True and set(manifest["records"]) == set(contract["CAD_order"]),
            "PointConv geometry scientific negative: do not initialize CUDA/model")
    for cad,row in manifest["records"].items():
        _require(_path(row["path"]) == cache/f"{cad}.npz","Hierarchy cache path changed")
        _match(row["path"],row["file_identity"])
    return manifest


def read_cache_metadata(plan_path=DEFAULT_PLAN, *, contract=None):
    """Strict completed/feasible metadata gate before model or CUDA initialization."""
    if contract is None:
        contract,plan = request_contract(plan_path)
    else:
        plan = _json(plan_path); _rules(plan)
    return _manifest(plan,contract)


class ScientificBlock(RuntimeError):
    def __init__(self, code, evidence):
        super().__init__(code)
        self.code,self.evidence = code,evidence


def _fps(xyz64, original_ids, quota, device):
    """All per-step selection and indexing stay on CUDA; one result receipt."""
    import numpy as np
    import torch
    n = len(xyz64)
    if n < quota:
        raise ScientificBlock("insufficient_source_rows",{"source_rows":n,"requested_anchors":quota})
    torch.cuda.synchronize(device); started = time.perf_counter()
    xyz = torch.as_tensor(xyz64,dtype=torch.float64,device=device)
    ids = torch.as_tensor(original_ids,dtype=torch.int64,device=device)
    dmin = torch.full((n,),float("inf"),dtype=torch.float64,device=device)
    anchors = torch.full((quota,),-1,dtype=torch.int64,device=device)
    selected_dist = torch.empty((quota,),dtype=torch.float64,device=device)
    tie_counts = torch.empty((quota,),dtype=torch.int64,device=device)
    valid = torch.ones((),dtype=torch.bool,device=device)
    sentinel = torch.full_like(ids,torch.iinfo(torch.int64).max)
    zero = torch.zeros((),dtype=torch.int64,device=device)
    for step in range(quota):
        maximum = dmin.max()
        tied = dmin == maximum
        choice = torch.argmin(torch.where(tied,ids,sentinel))
        if step:
            valid = valid & (maximum > 0)
        # A failed slot contains -1, never a duplicate or replacement anchor.
        emitted = torch.where(valid,choice,torch.full_like(choice,-1))
        anchors[step:step+1].copy_(emitted.reshape(1))
        selected_dist[step:step+1].copy_(maximum.reshape(1))
        tie_counts[step:step+1].copy_(tied.sum().reshape(1))
        safe = torch.where(valid,choice,zero)
        center = torch.index_select(xyz,0,safe.reshape(1))
        squared = ((xyz-center)**2).sum(dim=1)
        dmin = torch.minimum(dmin,squared)
    # One end-of-stage synchronization/copy, none inside the FPS loop.
    packed = torch.cat((torch.stack((anchors.to(torch.float64),selected_dist,tie_counts.to(torch.float64)),dim=1).reshape(-1),
        dmin.max().reshape(1))).cpu().numpy()
    receipt,remaining = packed[:-1].reshape(quota,3),float(packed[-1])
    torch.cuda.synchronize(device); seconds = time.perf_counter()-started
    chosen = receipt[:,0].astype(np.int64)
    invalid = np.flatnonzero(chosen < 0)
    if len(invalid):
        step = int(invalid[0])
        raise ScientificBlock("zero_remaining_FPS_distance",{"source_rows":n,"requested_anchors":quota,
            "first_invalid_anchor_slot":step,"valid_anchors_before_failure":step,
            "maximum_squared_distance_at_failure":float(receipt[step,1]),"fps_seconds":seconds,
            "duplicate_anchors_saved":0})
    _require(len(np.unique(chosen)) == quota,"FPS emitted duplicate row IDs")
    return chosen,{"source_rows":n,"anchor_count":quota,"first_original_row":int(original_ids[chosen[0]]),
        "max_distance_tie_steps_after_first":int((receipt[1:,2]>1).sum()),
        "maximum_tie_multiplicity_after_first":int(receipt[1:,2].max()) if quota>1 else 0,
        "minimum_selected_squared_distance_after_first":float(receipt[1:,1].min()) if quota>1 else None,
        "final_max_nearest_squared_distance":remaining,"fps_seconds":seconds,
        "loop_host_scalar_syncs":0,"unique_anchors":quota,"tie_rule":"minimum original row ID"}


def _exact_neighbors(tree, source, original_ids, queries, count, *, self_rows=None):
    """Exact KDTree candidates plus exact cutoff-radius tie completion.

    nextafter enlarges only the tree candidate radius by one representable ULP;
    exact recomputed squared-distance <= cutoff filters the resulting candidates.
    It never changes the chosen distance or admits a farther neighbor.
    """
    import numpy as np
    n = len(source); q = len(queries); take = count-(1 if self_rows is not None else 0)
    _require(n >= count,"Too few rows for fixed exact neighborhood")
    width = min(n,take+1+(1 if self_rows is not None else 0))
    indices = np.empty((q,count),dtype=np.int32)
    distances = np.empty((q,count),dtype=np.float64)
    ties = {"cutoff_tie_rows":0,"radius_extra_candidates":0,"maximum_radius_candidates":0}
    for start in range(0,q,2048):
        stop = min(q,start+2048); query = queries[start:stop]
        _,candidate = tree.query(query,k=width,eps=0,p=2,workers=-1)
        candidate = np.asarray(candidate,dtype=np.int64).reshape(len(query),width)
        d2 = ((source[candidate]-query[:,None,:])**2).sum(axis=2)
        local_self = None if self_rows is None else self_rows[start:stop]
        if local_self is not None:
            d2[candidate == local_self[:,None]] = np.inf
        order = np.lexsort((original_ids[candidate],d2),axis=1)
        ordered_ids = np.take_along_axis(candidate,order,axis=1)
        ordered_d2 = np.take_along_axis(d2,order,axis=1)
        chosen = ordered_ids[:,:take].copy(); chosen_d2 = ordered_d2[:,:take].copy()
        boundary = np.flatnonzero(ordered_d2[:,take] == chosen_d2[:,-1]) if width>take else np.empty(0,np.int64)
        for row in boundary:
            cutoff = chosen_d2[row,-1]
            radius = np.nextafter(np.sqrt(cutoff),np.inf)
            expanded = np.asarray(tree.query_ball_point(query[row],radius,p=2,eps=0,workers=1),dtype=np.int64)
            if local_self is not None:
                expanded = expanded[expanded != local_self[row]]
            expanded_d2 = ((source[expanded]-query[row])**2).sum(axis=1)
            exact = expanded_d2 <= cutoff
            expanded,expanded_d2 = expanded[exact],expanded_d2[exact]
            # Include the original exact candidates in case tree-radius rounding
            # rejected a boundary point; no guessed or farther point is added.
            expanded = np.unique(np.concatenate((expanded,chosen[row])))
            expanded_d2 = ((source[expanded]-query[row])**2).sum(axis=1)
            exact = expanded_d2 <= cutoff
            expanded,expanded_d2 = expanded[exact],expanded_d2[exact]
            _require(len(expanded)>=take,"Exact cutoff completion lost required candidates")
            ordered = np.lexsort((original_ids[expanded],expanded_d2))[:take]
            chosen[row],chosen_d2[row] = expanded[ordered],expanded_d2[ordered]
            ties["cutoff_tie_rows"] += 1
            ties["radius_extra_candidates"] += max(0,len(expanded)-take)
            ties["maximum_radius_candidates"] = max(ties["maximum_radius_candidates"],len(expanded))
        if local_self is None:
            indices[start:stop],distances[start:stop] = chosen,np.sqrt(chosen_d2)
        else:
            indices[start:stop,0] = local_self; distances[start:stop,0] = 0
            indices[start:stop,1:],distances[start:stop,1:] = chosen,np.sqrt(chosen_d2)
        _require(np.isfinite(distances[start:stop]).all(),"Nonfinite exact neighbor distance")
        _require((np.diff(np.sort(indices[start:stop],axis=1),axis=1)>0).all(),"Duplicate source row in neighborhood")
    return indices,distances,ties


def _knn_and_density(xyz64, original_ids, device):
    import numpy as np
    import torch
    from scipy.spatial import cKDTree
    started = time.perf_counter(); tree = cKDTree(xyz64)
    neighbors,distances,ties = _exact_neighbors(tree,xyz64,original_ids,xyz64,32,
        self_rows=np.arange(len(xyz64),dtype=np.int64))
    knn_seconds = time.perf_counter()-started
    sigma = float(np.median(distances[:,-1]))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ScientificBlock("nonpositive_density_bandwidth",{"source_rows":len(xyz64),"sigma":sigma if np.isfinite(sigma) else None,
            "zero_d32_rows":int((distances[:,-1]==0).sum()),"knn_seconds":knn_seconds,"ties":ties})
    torch.cuda.synchronize(device); tick = time.perf_counter()
    density = np.empty(len(xyz64),np.float32)
    for start in range(0,len(xyz64),32768):
        stop = min(len(xyz64),start+32768)
        d = torch.as_tensor(distances[start:stop],dtype=torch.float64,device=device)
        density[start:stop] = torch.exp(-d.square()/(2*sigma*sigma)).mean(dim=1).cpu().numpy().astype(np.float32)
    torch.cuda.synchronize(device); kde_seconds = time.perf_counter()-tick
    _require(np.isfinite(density).all() and (density>=np.float32(1/32)).all() and (density<=1).all(),
        "KDE finite/self lower-bound invariant failed")
    return tree,neighbors,density,{"source_rows":len(xyz64),"neighbor_count":32,
        "self_rows":len(xyz64),"distinct_source_rows_per_query":32,"ties":ties,
        "d32_min":float(distances[:,-1].min()),"d32_median":sigma,"d32_max":float(distances[:,-1].max()),
        "sigma":sigma,"density_min":float(density.min()),"density_max":float(density.max()),
        "self_lower_bound":1/32,"knn_seconds":knn_seconds,"kde_seconds":kde_seconds}


def _interpolation(tree, source, original_ids, queries):
    import numpy as np
    started = time.perf_counter()
    indices,distances,ties = _exact_neighbors(tree,source,original_ids,queries,3)
    zero = distances == 0; has_zero = zero.any(axis=1)
    weights = np.empty_like(distances)
    weights[has_zero] = zero[has_zero]/zero[has_zero].sum(axis=1,keepdims=True)
    positive = ~has_zero
    inverse = 1/distances[positive]
    weights[positive] = inverse/inverse.sum(axis=1,keepdims=True)
    stored = weights.astype(np.float32)
    _require(np.isfinite(stored).all() and (stored>=0).all(),"Invalid3NN weights")
    residual = float(np.abs(stored.astype(np.float64).sum(1)-1).max())
    _require(residual <= 2e-7,"FP32 3NN weights lost partition of unity")
    return indices,stored,{"query_rows":len(queries),"source_rows":len(source),"supports":3,
        "zero_distance_query_rows":int(has_zero.sum()),"all_zero_selected_rows":int(zero.all(1).sum()),
        "weight_sum_max_abs_error_fp32":residual,"unique_source_rows_used":int(len(np.unique(indices))),
        "ties":ties,"interpolation_seconds":time.perf_counter()-started,"distance_floor":None}


def _build(xyz, device):
    import numpy as np
    original0 = np.arange(len(xyz),dtype=np.int64)
    coords0 = xyz.astype(np.float64)
    anchors01,fps1 = _fps(coords0,original0,2048,device)
    coords1,original1 = coords0[anchors01],original0[anchors01]
    anchors12,fps2 = _fps(coords1,original1,128,device)
    coords2,original2 = coords1[anchors12],original1[anchors12]
    arrays = {"xyz0":xyz,"xyz1":coords1.astype(np.float32),"xyz2":coords2.astype(np.float32),
        "anchors01":anchors01,"anchors12":anchors12,"original_ids1":original1,"original_ids2":original2}
    trees,knn = [],[]
    for level,(coords,original) in enumerate(((coords0,original0),(coords1,original1),(coords2,original2))):
        tree,neighbors,p,facts = _knn_and_density(coords,original,device)
        arrays[f"neighbors{level}{level}"],arrays[f"p{level}"] = neighbors,p
        trees.append(tree); knn.append(facts)
    interp21,weight21,facts21 = _interpolation(trees[2],coords2,original2,coords1)
    interp10,weight10,facts10 = _interpolation(trees[1],coords1,original1,coords0)
    arrays.update(interp21_indices=interp21,interp21_weights=weight21,interp10_indices=interp10,interp10_weights=weight10)
    facts = {"fps":[fps1,fps2],"knn":knn,"kde":[{key:item[key] for key in ("sigma","density_min","density_max","self_lower_bound","kde_seconds")} for item in knn],
        "interpolation":{"21":facts21,"10":facts10},
        "coverage":{"fine_query_rows":len(xyz),"fine_same_layer_source_rows":len(xyz),
            "stride01_unique_source_rows":int(len(np.unique(arrays["neighbors00"][anchors01]))),
            "stride12_unique_source_rows":int(len(np.unique(arrays["neighbors11"][anchors12]))),
            "BC_role_coverage":"unmeasured; FPS does not guarantee role preservation",
            "all_sources_enter_coarsest":False},
        "coordinate_identity":{"fine_original_row_order":True,"fine_dtype":"float32",
            "xyz0_sha256":hashlib.sha256(xyz.tobytes(order="C")).hexdigest(),
            "xyz1_exact_original_subset":bool(np.array_equal(arrays["xyz1"],xyz[anchors01])),
            "xyz2_exact_original_subset":bool(np.array_equal(arrays["xyz2"],xyz[original2])),
            "original_ID_meaning":"zero-based prepared original row; no reindexing or resampling"},
        "labels_read":0,"scaler_fits":0}
    costs = {"fps_seconds":fps1["fps_seconds"]+fps2["fps_seconds"],
        "knn_seconds":sum(item["knn_seconds"] for item in knn),
        "kde_seconds":sum(item["kde_seconds"] for item in knn),
        "interpolation_seconds":facts21["interpolation_seconds"]+facts10["interpolation_seconds"]}
    return arrays,facts,costs


def _read_receipt():
    return {"input_only":True,"array_members_read":["canonical xyz.npy"],
        "identity_stat_only":["features.npy","cloud.npy","mask.npy","canonical prepared archive"],
        "labels_read":0,"weights_read":0,"scaler_fits":0,"cells_or_FieldCSV_read":0,
        "old_final_or_sealed_or_dia_inputs":0,"latent_activations_written":0}


def prepare_data(plan_path=DEFAULT_PLAN):
    """Root-only45 geometry preparation, resumable at completed-CAD boundaries."""
    started = time.perf_counter(); contract,plan = request_contract(plan_path); _rules(plan,actual=True)
    _require(plan.get("source_sha256",{}).get(SOURCE) == contract["source_sha256"][SOURCE],
        "Root must pin the final PointConv data source")
    cache = _path(plan["pointconv_input_cache"])
    if (cache/"manifest.json").is_file() and _json(cache/"manifest.json").get("status") == "completed":
        return _manifest(plan,contract,require_feasible=False)
    cache.mkdir(parents=True,exist_ok=True); config = {"contract":contract}
    if (cache/"config.json").is_file():
        _require(_json(cache/"config.json") == config,"Existing PointConv preparation config differs")
    _write(cache/"config.json",config)
    records = {}; reused = []; current = None; device = None; resume_verify_seconds = 0.
    def status(state, **extra):
        _write(cache/"status.json",{"status":state,"kind":"pointconv_input","completed_CADs":len(records),
            "elapsed_seconds":time.perf_counter()-started,**extra})
    def finish(feasible, **extra):
        keys = ("input_seconds","build_seconds","fps_seconds","knn_seconds","kde_seconds","interpolation_seconds","write_seconds","identity_hash_seconds")
        costs = {key:sum(row["costs"].get(key,0.) for row in records.values()) for key in keys}
        costs["resume_verify_seconds"] = resume_verify_seconds
        memory = {}
        if device is not None:
            import torch
            memory = {"peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes":torch.cuda.max_memory_reserved(device),"scope":"this preparation process; no model/backward"}
        manifest = {"kind":"pointconv_input","status":"completed","feasibility_pass":feasible,"contract":contract,
            "records":records,"preparation_seconds":time.perf_counter()-started,"costs":costs,
            "total_cache_bytes":sum(row["file_identity"]["bytes"] for row in records.values()),
            "resumed_completed_CADs":reused,"read_receipt":_read_receipt(),"memory":memory,
            "completed_at":datetime.now(timezone.utc).isoformat(),**extra}
        _write(cache/"manifest.json",manifest); status("completed",feasibility_pass=feasible)
        return manifest
    status("running")
    try:
        for cad in contract["CAD_order"]:
            current = cad; source = contract["inputs"][cad]; receipt_path = cache/f"{cad}.json"; target = cache/f"{cad}.npz"
            if receipt_path.is_file():
                tick = time.perf_counter()
                previous = _json(receipt_path)
                _require(previous.get("status") == "completed" and previous.get("contract_sha256") == _digest(contract)
                    and previous["record"]["input_source"] == source,"Completed-CAD preparation receipt differs")
                row = previous["record"]; _match(target,row["file_identity"])
                _require(_sha(target) == row["file_identity"]["sha256"],"Completed-CAD cache SHA changed")
                resume_verify_seconds += time.perf_counter()-tick
                row["reused_completed_CAD"] = True; records[cad] = row; reused.append(cad); status("running",current_CAD=cad)
                continue
            import numpy as np
            import torch
            cad_started = tick = time.perf_counter()
            for name,identity in source["canonical"].items():
                _match(identity["path"],identity)
            # Only this original xyz member is opened. Features/cloud/mask stay
            # the existing trainer's inputs and are not array-read here.
            xyz = np.array(np.load(_path(source["canonical"]["xyz"]["path"]),allow_pickle=False),copy=True)
            _require(xyz.dtype == np.float32 and xyz.shape == (source["num_nodes"],3)
                and np.isfinite(xyz).all(),"Corrupt canonical xyz shape/dtype/nonfinite values")
            input_seconds = time.perf_counter()-tick
            if device is None:
                device = torch.device("cuda:0")
                torch.cuda.set_device(device)
                # Initialize the context before peak reset; resident geometry is
                # subsequently included, with no model/optimizer constructed.
                torch.empty(0,device=device)
                torch.cuda.reset_peak_memory_stats(device)
            tick = time.perf_counter()
            arrays,facts,costs = _build(xyz,device)
            costs["input_seconds"],costs["build_seconds"] = input_seconds,time.perf_counter()-tick
            tick = time.perf_counter(); temporary = target.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez(handle,**arrays)
            temporary.replace(target); costs["write_seconds"] = time.perf_counter()-tick
            tick = time.perf_counter(); file_identity = _identity(target,sha=True)
            costs["identity_hash_seconds"] = time.perf_counter()-tick
            row = {"cad_id":cad,"path":target.relative_to(ROOT).as_posix(),"file_identity":file_identity,
                "num_nodes":len(xyz),"node_counts":[len(xyz),2048,128],"facts":facts,"costs":costs,
                "input_source":source,"reused_completed_CAD":False,"elapsed_seconds":time.perf_counter()-cad_started,
                "array_bytes":sum(value.nbytes for value in arrays.values()),"read_receipt":_read_receipt()}
            _write(receipt_path,{"status":"completed","kind":"pointconv_input","contract_sha256":_digest(contract),"record":row})
            records[cad] = row; status("running",current_CAD=cad)
        return finish(True)
    except ScientificBlock as exc:
        return finish(False,failure={"CAD":current,"code":exc.code,"evidence":exc.evidence,
            "no_partial_hierarchy_saved_for_failed_CAD":True,"training_allowed":False})
    except Exception as exc:
        status("failed",current_CAD=current,error=f"{type(exc).__name__}: {exc}")
        _write(cache/"manifest.json",{"kind":"pointconv_input","status":"failed","feasibility_pass":False,
            "contract":contract,"records":records,"preparation_seconds":time.perf_counter()-started,
            "error":f"{type(exc).__name__}: {exc}","read_receipt":_read_receipt()})
        raise


def load_hierarchy(record, device):
    """Load only this input cache; coarse stride tables reuse exact anchor rows."""
    import numpy as np
    import torch
    _match(record["path"],record["file_identity"])
    expected = {"xyz0","xyz1","xyz2","anchors01","anchors12","original_ids1","original_ids2",
        "neighbors00","neighbors11","neighbors22","p0","p1","p2",
        "interp21_indices","interp21_weights","interp10_indices","interp10_weights"}
    with np.load(_path(record["path"]),allow_pickle=False) as archive:
        _require(set(archive.files) == expected,"PointConv input member schema differs")
        arrays = {name:np.array(archive[name],copy=True) for name in expected}
    counts = record["node_counts"]
    for level,n in enumerate(counts):
        _require(arrays[f"xyz{level}"].shape == (n,3) and arrays[f"xyz{level}"].dtype == np.float32
            and arrays[f"p{level}"].shape == (n,) and arrays[f"p{level}"].dtype == np.float32
            and arrays[f"neighbors{level}{level}"].shape == (n,32) and arrays[f"neighbors{level}{level}"].dtype == np.int32,
            "Cached level shape/dtype differs")
    for name,nsource,nquery in (("21",counts[2],counts[1]),("10",counts[1],counts[0])):
        index,weight = arrays[f"interp{name}_indices"],arrays[f"interp{name}_weights"]
        _require(index.shape == weight.shape == (nquery,3) and index.dtype == np.int32 and weight.dtype == np.float32,
            "Cached interpolation shape/dtype differs")
    for name,nsource,nquery in (("01",counts[0],counts[1]),("12",counts[1],counts[2])):
        index = arrays[f"anchors{name}"]
        _require(index.dtype == np.int64 and index.shape == (nquery,) and (index>=0).all() and (index<nsource).all(),
            "Cached anchor shape/indices differ")
    arrays["neighbors01"] = arrays["neighbors00"][arrays["anchors01"]]
    arrays["neighbors12"] = arrays["neighbors11"][arrays["anchors12"]]
    result = {}
    for name,value in arrays.items():
        if name.startswith("anchors") or name.startswith("original_ids"):
            continue
        dtype = torch.int64 if name.startswith("neighbors") or name.endswith("_indices") else torch.float32
        result[name] = torch.as_tensor(value,dtype=dtype,device=device)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",default=DEFAULT_PLAN)
    args = parser.parse_args()
    manifest = prepare_data(args.plan)
    print(json.dumps({"status":manifest["status"],"feasibility_pass":manifest["feasibility_pass"],
        "completed_CADs":len(manifest["records"]),"preparation_seconds":manifest["preparation_seconds"],
        "total_cache_bytes":manifest["total_cache_bytes"]},allow_nan=False))


if __name__ == "__main__":
    main()
