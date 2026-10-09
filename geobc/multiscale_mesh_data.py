"""Target-free45-CAD connected mesh hierarchy for the new full-node solver.

Reuse numerical helpers, never their frozen four-CAD entry points or globals.
Author source checks are metadata only; root alone prepares geometry arrays.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

from tools import prepare_topology_context as base

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = "notes/geobc_multiscale_mesh_plan.json"
SOURCES = ("geobc/multiscale_mesh_data.py", "tools/prepare_topology_context.py", "tools/prepare_topology_context_balanced.py")
MAPS = {"W0":(0,0),"W1":(1,1),"W2":(2,2),"R01":(0,1),"R12":(1,2),"P12":(2,1),"P01":(1,0)}
SCHEMA = {"kind":"multiscale_mesh_input","levels":3,"soft_nodes":[2048,128],
    "hard_maximum_policy":"current_level_node_count",
    "ownership":"balanced_owners; connected exact seven-flag signature; minimum original descendant ordering",
    "W":"binary original corner-mid subedges; symmetric quotient cut multiplicities; no RBE shortcuts",
    "R":"hard aggregate mass restriction; original volume/10 proxy; row coarse dst, column fine src",
    "P":"complete two Jacobi passes J^2G; omega2/3; row fine dst, column coarse src",
    "P_resource_guard":"Original helper rejects >64 average nnz at either pass; no truncation, topk or support pruning",
    "CSR":"row=receiver dst, column=sender src; complete support; indices int64",
    "W_runtime_weight":"cut_W/weighted_degree(dst), float32 after FP64 normalization",
    "R_P_runtime_weight":"saved original float32 map weights; no extra normalization",
    "edge_delta":"receiver_xyz[dst]-sender_xyz[src] in existing normalized coordinates",
    "edge_length":"norm(delta_xyz), same original characteristic_length_mm for all levels",
    "runtime_keys":["src","dst","weight","delta_xyz","length","indptr","shape","csr"],
    "runtime_xyz":"xyz0/xyz1/xyz2 float32; coarse mass-weighted centers in same canonical frame",
    "role_semantics":"support-reference DOF/load-interface membership, not nodal Dirichlet; P may mix signatures as original helper records",
    "full_support":True,"latent_activations_cached":False,"labels_read":0}


def _rules(plan,actual=False):
    expected = {"existing_split":"notes/geobc_split.json","prepared_data":"data/prepared_geobc_multiload",
        "data_cache":"data/geobc_solver_selection","gradient_manifest":"data/geobc_gradient_supervision/manifest.json",
        "graph_context_manifest":"data/geobc_graph_context/manifest.json","mesh_input_cache":"data/geobc_multiscale_mesh_inputs45"}
    for key,value in expected.items():
        base.require(base.path(plan[key]) == base.path(value),f"Mesh input source/destination changed: {key}")
    recipe = plan["recipe"]
    base.require(recipe.get("soft_nodes") == [2048,128] and recipe.get("hard_maximum_policy") == "current_level_node_count"
        and recipe.get("cloud_points") == 1024 and recipe.get("cloud_seed") == 2026
        and recipe.get("train_loads") == ["ver","hor","tor"],"Fixed hierarchy/cloud/known3 recipe changed")
    if actual:
        base.require(plan.get("training_rules_frozen") is True,"Root must freeze the new mesh protocol before preparation")


def request_contract(plan_path=DEFAULT_PLAN):
    """Stdlib JSON/source SHA/stat only; no numerical arrays or scientific imports."""
    plan = base.read_json(plan_path); _rules(plan)
    split = base.read_json(plan["existing_split"]); groups = split["groups"]
    base.require([len(groups[name]) for name in ("train","validation","test")] == [40,5,5]
        and not any(set(groups[a])&set(groups[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))),"Expected frozen disjoint whole-CAD40/5/5")
    old = base.read_json(base.path(plan["data_cache"])/"manifest.json")
    grad = base.read_json(plan["gradient_manifest"]); graph = base.read_json(plan["graph_context_manifest"])
    base.require(old.get("status") == grad.get("status") == graph.get("status") == "completed"
        and grad.get("complete") is True and graph.get("complete") is True,"Existing45 source manifests must be completed")
    train_keys = [f"{cad}/{load}" for cad in groups["train"] for load in ("ver","hor","tor")]
    val_keys = [f"{cad}/{load}" for cad in groups["validation"] for load in ("ver","hor","tor")]
    base.require(old["train_keys"] == train_keys and old["validation_keys"] == val_keys
        and old["contract"]["split_sha256"] == base.digest_file(plan["existing_split"]),"Original selection identity/order differs")
    old_sources = {row["key"]:row for row in old["contract"]["sources"]}
    inputs = {}; cache_inputs = {}
    for group in ("train","validation"):
        for cad in groups[group]:
            g,e = grad["CADs"][cad],graph["CADs"][cad]
            normalization = {key:g["normalization"][key] for key in ("bbox_min_mm","characteristic_length_mm")}
            base.require(g["node_count"] == e["node_count"] and g["xyz_sha256"] == e["xyz_sha256"]
                and normalization == e["normalization"],"Existing geometry/node-order/normalization metadata differs")
            named = e["source_groups"]
            base.require([row["channel"] for row in named] == list(range(5))
                and [row["role"] for row in named] == ["support"]*4+["load"]
                and [row["master_node_id"] for row in named[:4]] == sorted(row["master_node_id"] for row in named[:4]),"Five-interface source order differs")
            prepared = base.path(plan["prepared_data"])/f"{cad}_ver.npz"
            gradient = base.path(plan["gradient_manifest"]).parent/g["file"]
            interface = base.path(plan["graph_context_manifest"]).parent/e["file"]
            base.require(base.path(split["case_paths"][f"{cad}/ver"]) == prepared,"Only canonical selected ver input is allowed")
            base.match_stat(prepared,old_sources[f"{cad}/ver"])
            base.match_stat(gradient,g); base.match_stat(interface,e)
            inputs[cad] = {"prepared":base.identity(prepared),"gradient_geometry":base.identity(gradient),
                "interface_inputs":base.identity(interface),"node_count":g["node_count"],"cell_count":g["cell_count"],
                "xyz_sha256":g["xyz_sha256"],"normalization":normalization,"sum_volume_mm3":g["sum_volume_mm3"],
                "cells_sha256":e["cells_sha256"],"edges_sha256":e["graph_topology"]["edges_sha256"],
                "source_groups":[{key:row[key] for key in ("channel","role","master_node_id","card_type","member_count","member_GRID_IDs_sha256","member_node_indices_sha256")} for row in named]}
            old_record = old["records"][f"{cad}/ver"]
            base.require(old_record["num_nodes"] == g["node_count"] and old_record["group"] == group,"Original selection full-node metadata differs")
            cache_inputs[cad] = {name:base.identity(old_record[name+"_path"]) for name in ("xyz","features","cloud","mask")}
    sources = {name:base.digest_file(name) for name in SOURCES}
    for name,value in sources.items():
        base.require(plan.get("source_sha256",{}).get(name,value) == value,f"Pinned mesh input source changed: {name}")
    reuse = {}
    legacy_path = base.path("data/geobc_topology_balanced_train4/manifest.json")
    if legacy_path.is_file():
        legacy = base.read_json(legacy_path)
        if legacy.get("status") == "completed" and all(legacy["contract"]["source_sha256"].get(name) == sources[name] for name in SOURCES[1:]):
            for cad,row in legacy.get("CADs",{}).items():
                if cad in inputs and row.get("status") == "completed" and row.get("source") == inputs[cad] and [v["soft_target"] for v in row["facts"]["ownership"]] == [2048,128]:
                    path = legacy_path.parent/row["file"]
                    base.match_stat(path,row["file_identity"])
                    reuse[cad] = {"file":base.identity(path),"sha256":row["file_identity"]["sha256"],
                        "facts":row["facts"],"legacy_manifest_sha256":base.digest_file(legacy_path)}
    contract = {"kind":"multiscale_mesh_input","schema":SCHEMA,"source_sha256":sources,
        "split_sha256":base.digest_file(plan["existing_split"]),
        "manifests":{key:base.identity(plan[key],sha=True) for key in ("gradient_manifest","graph_context_manifest")},
        "selection_manifest_sha256":base.digest_file(base.path(plan["data_cache"])/"manifest.json"),
        "train_CADs":groups["train"],"validation_CADs":groups["validation"],"CAD_order":groups["train"]+groups["validation"],
        "inputs":inputs,"canonical_cache_inputs":cache_inputs,"legacy_reuse":reuse,
        "cache":base.portable(plan["mesh_input_cache"]),
        "members":{"prepared":["xyz","features","sample_key","label_alignment","bbox_min_mm","characteristic_length_mm"],
            "gradient":["cells10","volumes_mm3"],"graph_context":["euclidean","cad_id","xyz_sha256"]},
        "labels_read":0,"reference_strain_read":0,"raw_sources_opened":False,"old_final_or_sealed_or_dia_inputs":0}
    return contract,plan


def read_cache_metadata(plan_path=DEFAULT_PLAN,*,contract=None):
    if contract is None:
        contract,plan = request_contract(plan_path)
    else:
        plan = base.read_json(plan_path); _rules(plan)
    manifest = base.read_json(base.path(plan["mesh_input_cache"])/"manifest.json")
    base.require(manifest.get("status") == "completed" and manifest.get("feasibility_pass") is True
        and manifest.get("contract") == contract and set(manifest["records"]) == set(contract["CAD_order"]),"Mesh input cache incomplete/negative or contract differs")
    for cad,row in manifest["records"].items():
        expected = base.path(plan["mesh_input_cache"])/f"{cad}.npz"
        base.require(base.path(row["path"]) == expected,"Prepared hierarchy path differs")
        base.match_stat(expected,row["file_identity"])
    return manifest


def _build(xyz,cells,volumes,roles,named):
    import numpy as np
    from tools import prepare_topology_context_balanced as balanced
    started = time.perf_counter()
    W0,edges,vectors,lengths = base.graph_from_cells(xyz,cells)
    mass0 = np.zeros(len(xyz),np.float64)
    np.add.at(mass0,cells.ravel(),np.repeat(volumes/10,10))
    base.require(np.isfinite(mass0).all() and (mass0 > 0).all()
        and math.isclose(float(mass0.sum()),float(volumes.sum()),rel_tol=1e-12,abs_tol=1e-12),"Invalid full-node mass proxy")
    raw = ((roles[:,None] >> np.arange(7)) & 1).astype(np.int64)
    arrays = {"xyz0_normalized":xyz.astype(np.float32),"role_bits0":roles,"mass0_mm3":mass0,
        "named5_counts0":named,"role_member_counts0":raw,"edge_endpoints":edges.astype(np.int64),
        "edge_vectors_normalized":vectors,"edge_lengths_normalized":lengths}
    state = {"W":W0,"mass":mass0,"xyz":xyz,"roles":roles,"named":named,"raw":raw,"descendant":np.arange(len(xyz),dtype=np.int64)}
    graphs,ownership,maps = [],[],{}
    for level,soft in enumerate((2048,128)):
        facts,degree = base.graph_facts(state["W"]); graphs.append(facts)
        arrays[f"degree{level}"] = degree; base.pack_csr(arrays,f"W{level}",state["W"],np.float64)
        owner,parent,admission,coarse_roles,descendant,owner_facts = balanced.balanced_owners(
            state["W"],state["roles"],state["descendant"],soft,state["W"].shape[0],level)
        ownership.append(owner_facts)
        arrays[f"aggregate{level}{level+1}"],arrays[f"bfs_parent{level}"] = owner,parent
        arrays[f"bfs_admission_rank{level}"],arrays[f"descendant_min{level}"] = admission,state["descendant"]
        W,G,R,mass,centers,named_counts,raw_counts = base.aggregate(state["W"],owner,state["mass"],state["xyz"],state["named"],state["raw"])
        base.require(np.array_equal(named_counts.sum(0),named.sum(0)) and np.array_equal(raw_counts.sum(0),raw.sum(0))
            and math.isclose(float(mass.sum()),float(mass0.sum()),rel_tol=1e-12,abs_tol=1e-12),"Mass/source counts not conserved")
        saved_R,r_facts = base.mapping_facts(R)
        P,p_facts = base.prolongation(state["W"],G,state["roles"],coarse_roles,level)
        for name,matrix in ((f"R{level}{level+1}",saved_R),(f"P{level}{level+1}",P)):
            base.pack_csr(arrays,name,matrix,np.float32)
        maps[f"R{level}{level+1}"],maps[f"P{level}{level+1}"] = r_facts,p_facts
        arrays[f"xyz{level+1}_normalized"],arrays[f"mass{level+1}_mm3"] = centers.astype(np.float32),mass
        arrays[f"role_bits{level+1}"],arrays[f"named5_counts{level+1}"],arrays[f"role_member_counts{level+1}"] = coarse_roles,named_counts,raw_counts
        state = {"W":W,"mass":mass,"xyz":centers,"roles":coarse_roles,"named":named_counts,"raw":raw_counts,"descendant":descendant}
    facts,degree = base.graph_facts(state["W"]); graphs.append(facts)
    base.pack_csr(arrays,"W2",state["W"],np.float64)
    arrays["degree2"],arrays["descendant_min2"] = degree,state["descendant"]
    return arrays,{"N0":len(xyz),"N1":len(arrays["mass1_mm3"]),"N2":len(arrays["mass2_mm3"]),
        "graphs":graphs,"ownership":ownership,"maps":maps,"build_seconds":time.perf_counter()-started,
        "input_volume_proxy_sum_mm3":float(mass0.sum()),"mass_and_source_count_conservation":True,"labels_read":0}


def prepare_data(plan_path=DEFAULT_PLAN):
    started = time.perf_counter(); contract,plan = request_contract(plan_path); _rules(plan,actual=True)
    cache = base.path(plan["mesh_input_cache"])
    if (cache/"manifest.json").is_file():
        return read_cache_metadata(plan_path,contract=contract)
    for name,value in contract["source_sha256"].items():
        base.require(plan.get("source_sha256",{}).get(name) == value,"Root must pin all final hierarchy helper/source identities")
    cache.mkdir(parents=True,exist_ok=True)
    config = {"contract":contract}
    if (cache/"config.json").exists():
        base.require(base.read_json(cache/"config.json") == config,"Existing input preparation config differs")
    base.write_json(cache/"config.json",config)
    records = {}; reused = []; costs = {"input_seconds":0.,"build_seconds":0.,"write_seconds":0.,"reuse_seconds":0.,"identity_hash_seconds":0.}
    def status(state,**extra):
        base.write_json(cache/"status.json",{"status":state,"kind":"multiscale_mesh_input","completed_CADs":len(records),"elapsed_seconds":time.perf_counter()-started,**extra})
    status("running")
    try:
        import numpy as np
        for cad in contract["CAD_order"]:
            source = contract["inputs"][cad]; target = cache/f"{cad}.npz"; tick = time.perf_counter()
            if cad in contract["legacy_reuse"]:
                old = contract["legacy_reuse"][cad]; original = base.path(old["file"]["path"])
                base.require(base.digest_file(original) == old["sha256"],"Legacy input-only cache SHA changed")
                temporary = target.with_suffix(".npz.tmp"); shutil.copyfile(original,temporary); temporary.replace(target)
                facts = copy.deepcopy(old["facts"])
                facts["ownership_bound_reuse"] = "Original caps did not alter selection; relaxed natural-N bounds select identical owners; old proof/count facts retained"
                costs["reuse_seconds"] += time.perf_counter()-tick; reused.append(cad)
                accessed = {"legacy_input_cache_bytes":True,"prepared":[],"gradient":[],"graph_context":[]}
            else:
                xyz,cells,volumes,roles,named = base.read_inputs(cad,source)
                cached_xyz = np.load(base.path(contract["canonical_cache_inputs"][cad]["xyz"]["path"]),allow_pickle=False)
                base.require(np.array_equal(xyz,cached_xyz),"Canonical prepared geometry does not replay old selection original-node ordering")
                costs["input_seconds"] += time.perf_counter()-tick
                arrays,facts = _build(xyz,cells,volumes,roles,named); costs["build_seconds"] += facts["build_seconds"]
                tick = time.perf_counter(); temporary = target.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle: np.savez(handle,**arrays)
                temporary.replace(target); costs["write_seconds"] += time.perf_counter()-tick
                del arrays,xyz,cells,volumes,roles,named,cached_xyz
                accessed = contract["members"]
            tick = time.perf_counter(); output_identity = base.identity(target,sha=True)
            costs["identity_hash_seconds"] += time.perf_counter()-tick
            records[cad] = {"cad_id":cad,"path":base.portable(target),"group":"train" if cad in contract["train_CADs"] else "validation",
                "file_identity":output_identity,"source":source,"facts":facts,"schema":SCHEMA,
                "xyz_sha256":source["xyz_sha256"],"normalization":source["normalization"],
                "accessed_members":accessed,"legacy_reused":cad in reused,"labels_read":0,"latent_activations_cached":False}
            status("running",current_CAD=cad)
        current,_ = request_contract(plan_path); base.require(current == contract,"Input/source identity changed during preparation")
        manifest = {"status":"completed","kind":"multiscale_mesh_input","feasibility_pass":True,"contract":contract,
            "records":records,"schema":SCHEMA,"preparation_seconds":time.perf_counter()-started,"costs":costs,"peak_RSS":base.peak_rss(),
            "total_cache_bytes":sum(row["file_identity"]["bytes"] for row in records.values()),"reused_CADs":reused,
            "read_receipt":{"CADs":45,"TRAIN_CADs":40,"DEV_CADs":5,"labels_read":0,"reference_strain_read":0,
                "weights_or_models_loaded":0,"scaler_fits":0,"raw_sources_opened":False,"old_final_or_sealed_or_dia_inputs":0,
                "GPU_runs":0,"old_cache_files_written":0,"latent_activations_written":0,"support_truncated":False},
            "completed_at_utc":datetime.now(timezone.utc).isoformat()}
        base.write_json(cache/"manifest.json",manifest); status("completed",feasibility_pass=True)
        print(json.dumps({"status":"completed","kind":"multiscale_mesh_input","CADs":45,"seconds":manifest["preparation_seconds"],"cache_bytes":manifest["total_cache_bytes"],"reused_CADs":reused}))
        return manifest
    except base.ScientificBlock as error:
        manifest = {"status":"completed","kind":"multiscale_mesh_input","feasibility_pass":False,"contract":contract,
            "records":records,"preparation_seconds":time.perf_counter()-started,"reason":str(error),"diagnostics":error.evidence,"labels_read":0}
        base.write_json(cache/"manifest.json",manifest); status("completed",feasibility_pass=False,reason=str(error)); return manifest
    except Exception as error:
        status("failed",error_type=type(error).__name__,error=str(error)); raise


def load_hierarchy(record,device):
    """Load one CAD's immutable complete sparse supports, never latent features."""
    import numpy as np
    import torch
    base.match_stat(record["path"],record["file_identity"])
    hierarchy = {}
    with np.load(base.path(record["path"]),allow_pickle=False) as stored:
        for level in range(3):
            hierarchy[f"xyz{level}"] = torch.as_tensor(stored[f"xyz{level}_normalized"],dtype=torch.float32,device=device)
        for name,(sender,receiver) in MAPS.items():
            shape = tuple(int(v) for v in stored[name+"_shape"])
            indptr_np,src_np = stored[name+"_indptr"],stored[name+"_indices"]
            weight_np = stored[name+"_data"]
            dst_np = np.repeat(np.arange(shape[0],dtype=np.int64),np.diff(indptr_np))
            if name.startswith("W"):
                weight_np = (weight_np/stored[f"degree{receiver}"][dst_np]).astype(np.float32)
            src = torch.as_tensor(src_np,dtype=torch.int64,device=device)
            dst = torch.as_tensor(dst_np,dtype=torch.int64,device=device)
            weight = torch.as_tensor(weight_np,dtype=torch.float32,device=device)
            indptr = torch.as_tensor(indptr_np,dtype=torch.int64,device=device)
            delta = hierarchy[f"xyz{receiver}"][dst]-hierarchy[f"xyz{sender}"][src]
            hierarchy[name] = {"src":src,"dst":dst,"weight":weight,"delta_xyz":delta,
                "length":torch.linalg.vector_norm(delta,dim=1,keepdim=True),"indptr":indptr,"shape":shape,
                "csr":torch.sparse_csr_tensor(indptr,src,weight,size=shape,device=device)}
    return hierarchy


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--plan",default=DEFAULT_PLAN)
    parser.add_argument("--source-check",action="store_true"); args = parser.parse_args()
    if args.source_check:
        ast.parse(Path(__file__).read_text(encoding="utf-8"))
        contract,plan = request_contract(args.plan)
        print(json.dumps({"status":"source_check_passed","CADs":len(contract["CAD_order"]),"legacy_reuse_CADs":list(contract["legacy_reuse"]),"arrays_read":0,"models_or_GPU":0}))
    else:
        prepare_data(args.plan)


if __name__ == "__main__": main()
