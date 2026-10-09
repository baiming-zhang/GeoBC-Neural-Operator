"""Frozen TRAIN-only masked-Hash versus ALL107-condition path bookkeeping.

The four corner-anchored recipes are off-manifold counterfactuals, not causal
error contributions. Contract/dry run are standard-library JSON/source/stat.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.diagnose_field_error_gradient import path, read_json, identity, atomic_json, require, memory_record

CAD = "101_479"
MODES = ("full12", "coarse8")
LOADS = ("ver", "hor", "tor")
SCALE = .05511113670847156
PLAN_SHA = "ce67d4352ae1f3a7aec328fea52b0a549ed0ef689065854a12daa10b03d43d97"
RTOL, ATOL = 1e-10, 1e-14


def request_contract(plan_path):
    """Only small JSON/source SHA and named binary-file stats; no imports of numerical libraries."""
    plan = read_json(plan_path)
    require(identity(plan_path, small=True)["sha256"] == PLAN_SHA and plan["status"] == "predeclared"
            and plan["stage"] == "frozen_hash_condition_path_decomposition", "Root's fixed path protocol changed")
    require(plan["selection"]["CAD"] == CAD and plan["selection"]["models"] == list(MODES)
            and plan["sampling"]["maximum_cells"] == 1024 and plan["sampling"]["BC_change_quota"] == 512
            and plan["data"]["validation_dia_final_arrays"] is False, "Fixed TRAIN scope/sampling changed")
    pair = read_json(plan["selection"]["parent_result"])
    require(pair["status"] == "completed" and pair["matched_controls_passed"] is True
            and pair["clear_gain_gate_passed"] is False
            and identity(plan["selection"]["parent_result"], small=True)["sha256"] == plan["selection"]["parent_result_sha256"],
            "Requires the exact completed matched parent pair")
    gradient_manifest = path("data/geobc_gradient_supervision/manifest.json")
    gm = read_json(gradient_manifest)
    require(gm["complete"] is True and gm["status"] == "completed" and CAD in gm["groups"]["train"]
            and CAD not in gm["groups"]["validation"] + gm["groups"]["test"]
            and gm["basis_order"] == list(LOADS) and gm["frobenius_weights"] == [1.,1.,1.,2.,2.,2.]
            and gm["engineering_shear"] is False, "TRAIN P1 cache convention differs")
    record = gm["CADs"][CAD]
    inputs = {"prepared": identity(f"data/prepared_geobc_multiload/{CAD}_ver.npz"),
              "local": identity(f"data/geobc_local_geometry/{CAD}.npz"),
              "distance": identity(f"data/geobc_graph_context/{CAD}.npz"),
              "gradient": identity(gradient_manifest.parent / record["file"])}
    require(all(inputs["gradient"][k] == record[k] for k in ("bytes", "mtime_ns")), "Fixed gradient geometry stat changed")
    branches = {}
    for mode in MODES:
        run = ROOT/f"runs/geobc_exploration/displacement_hash_{mode}_10k"
        config, summary, best = (read_json(run/name) for name in ("config.json", "summary.json", "metrics_best.json"))
        require(summary["status"] == "completed" and summary["config"] == config
                and summary["best_metrics"] == best and best["step"] == summary["best_step"] == 9000
                and config["arguments"]["bandwidth_mode"] == mode and config["groups"] == gm["groups"]
                and config["train_loads"] == list(LOADS) and config["scaler"]["center_mm"] == 0
                and config["scaler"]["scale_mm"] == SCALE
                and config["gradient_supervision"]["manifest_sha256"] == identity(gradient_manifest, small=True)["sha256"],
                f"{mode}: frozen best9000, scope or common scale differs")
        for name in ("hash_bandwidth_field.py", "hash_bandwidth_train.py", "gradient_supervision.py", "query_context_field.py",
                     "graph_context_field.py", "bc_role_field.py", "physics_field.py", "conditioned_field.py", "tcnn_fields.py"):
            require(identity(ROOT/"geobc"/name, small=True)["sha256"] == config["contract"]["code_sha256"][name],
                    f"Frozen source changed: {name}")
        canonical = config["graph_context"]["CADs"][CAD]
        local = config["local_geometry"]["CADs"][CAD]
        require(all(inputs["distance"][k] == canonical[k] for k in ("bytes", "mtime_ns"))
                and all(inputs["local"][k] == local[k] for k in ("bytes", "mtime_ns"))
                and canonical["xyz_sha256"] == local["xyz_sha256"] == record["xyz_sha256"], "Input cache identity differs")
        source = record["displacement_sources"][0]
        require(path(source["path"]) == path(inputs["prepared"]["path"])
                and all(inputs["prepared"][k] == source[k] for k in ("bytes", "mtime_ns")), "Canonical input source differs")
        branches[mode] = {"run": str(run.relative_to(ROOT)), "checkpoint": identity(run/"best.pt"),
            "cloud": identity(run/"geometry_clouds.npz"), "config": identity(run/"config.json", small=True),
            "summary": identity(run/"summary.json", small=True), "best_metrics": identity(run/"metrics_best.json", small=True),
            "model_config": config["model"], "geometry": config["geometry"], "local_metadata": local,
            "distance_metadata": canonical, "CAD_order": gm["groups"]["train"]+gm["groups"]["validation"],
            "frozen_runtime_code_sha256": {name:config["contract"]["code_sha256"][name] for name in
                ("hash_bandwidth_field.py", "query_context_field.py", "graph_context_field.py", "bc_role_field.py",
                 "physics_field.py", "conditioned_field.py", "tcnn_fields.py")}}
    require(branches["full12"]["geometry"] == branches["coarse8"]["geometry"], "Frozen context geometry differs")
    sources = ("tools/diagnose_hash_condition_paths.py", "tools/diagnose_field_error_gradient.py",
               "geobc/hash_bandwidth_field.py", "geobc/gradient_supervision.py", "geobc/query_context_field.py")
    return {"version": 1, "plan": identity(plan_path, small=True), "protocol": plan, "CAD": CAD,
        "parent_result": identity(plan["selection"]["parent_result"], small=True), "branches": branches,
        "inputs": inputs, "gradient_manifest": identity(gradient_manifest, small=True), "gradient_metadata": record,
        "source_sha256": {p: identity(ROOT/p, small=True)["sha256"] for p in sources},
        "common_u_scale_mm": SCALE, "known_loads": list(LOADS), "selected_checkpoint_steps": [9000,9000],
        "sampling": {"maximum_cells":1024,"change_quota":512,"seed":2026+97},
        "network_forwards_max":10,"model_backwards":0,"optimizer_updates":0,
        "no_GT_prepared_members":True,"no_validation_dia_final_arrays":True,"numeric_imports_in_contract":0,
        "reference_container_read":"After witness commit, NumPy decompresses the complete same-TRAIN reference_cell_tensor_strain member; only selected cells enter diagnostics. This does not claim1024-row ZIP byte access.",
        "cloud_container_read":"Only selected TRAIN row is decoded; ZIP seeks can decompress skipped bytes. Other CAD rows are not materialized or used."}


def sample_cells(corners, bits):
    import numpy as np
    labels = np.any(bits[corners] != bits[corners[:, :1]], axis=(1,2))
    pools = {"BC_change":np.flatnonzero(labels), "no_BC_change":np.flatnonzero(~labels)}
    rng = np.random.default_rng(2026+97)
    change = rng.choice(pools["BC_change"], min(512,len(pools["BC_change"])), replace=False)
    unchanged = rng.choice(pools["no_BC_change"], min(1024-len(change),len(pools["no_BC_change"])), replace=False)
    selected = np.concatenate((change,unchanged)).astype(np.int64)
    require(len(selected)>0 and len(selected)==len(np.unique(selected)), "No eligible unique TRAIN cells")
    witness = {"indices":selected.tolist(),"indices_sha256":hashlib.sha256(selected.tobytes()).hexdigest(),
        "corners_sha256":hashlib.sha256(np.ascontiguousarray(corners[selected]).tobytes()).hexdigest(),
        "seed":2026+97,"population_cells":len(corners),"sample_cells":len(selected),
        "strata":{name:{"population":len(pool),"sample":len(change if name=="BC_change" else unchanged),
            "inclusion_fraction":len(change if name=="BC_change" else unchanged)/len(pool) if len(pool) else None}
            for name,pool in pools.items()},"selection_uses_predictions_or_reference":False}
    return selected, labels[selected], witness


def npz_row(archive, member, row):
    """Decode only one C-order NPY row; other cloud rows are skipped as bytes."""
    import numpy as np
    with archive.open(member+".npy") as stream:
        version = np.lib.format.read_magic(stream)
        reader = np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream)
        require(not fortran and not dtype.hasobject and 0<=row<shape[0], "Cloud member must be nonobject C-order rows")
        count = math.prod(shape[1:]); size = count*dtype.itemsize
        stream.seek(row*size, 1)
        data = stream.read(size)
        require(len(data)==size, "Truncated cloud row")
        return np.frombuffer(data,dtype=dtype).copy().reshape(shape[1:])


def path_differences(f00, f10, f01, f11, scale=SCALE):
    """Promote native9 outputs before subtraction and apply physical scale once."""
    a,b,c,d = (v.double() for v in (f00,f10,f01,f11))
    h = .5*((b-a)+(d-c))*scale
    condition = .5*((c-a)+(d-b))*scale
    raw = (d-a)*scale
    interaction = (d-b-c+a)*scale
    return h,condition,raw,interaction,(b-a)*scale,(d-c)*scale,(c-a)*scale,(d-b)*scale


def physical_statistics(h, condition, predicted, reference, interaction, volumes):
    import torch
    w = h.new_tensor([1.,1.,1.,2.,2.,2.]); volume = float(volumes.sum())
    require(volume>0, "Positive sampled physical volume required")
    def dot(a,b): return float(((a*b*w).sum(-1)*volumes).sum())
    eh,ec,ep,er,ei = (dot(v,v) for v in (h,condition,predicted,reference,interaction))
    hc,hr,cr = 2*dot(h,condition),2*dot(h,reference),2*dot(condition,reference)
    ee = dot(predicted-reference,predicted-reference)
    require(bool(torch.allclose(predicted,h+condition,rtol=RTOL,atol=ATOL)), "Physical P1 path identity failed")
    require(math.isclose(ep,eh+ec+hc,rel_tol=RTOL,abs_tol=ATOL)
            and math.isclose(ee,eh+ec+er+hc-hr-cr,rel_tol=RTOL,abs_tol=ATOL), "Signed energy closure failed")
    return {"cells":len(volumes),"sample_volume_mm3":volume,"H_energy":eh,"C_energy":ec,
        "prediction_energy":ep,"reference_energy":er,"error_energy":ee,"interaction_energy":ei,
        "signed_two_H_C_cross":hc,"signed_two_H_reference_cross":hr,"signed_two_C_reference_cross":cr,
        "prediction_energy_closure":ep-eh-ec-hc,"error_energy_closure":ee-eh-ec-er-hc+hr+cr,
        "P1_field_closure_max_abs":float((predicted-h-condition).abs().max()),
        "H_RMS":math.sqrt(eh/volume),"C_RMS":math.sqrt(ec/volume),"prediction_RMS":math.sqrt(ep/volume),
        "reference_RMS":math.sqrt(er/volume),"error_RMS":math.sqrt(ee/volume),"interaction_RMS":math.sqrt(ei/volume),
        "sampled_cell_relative_l2":math.sqrt(ee/er) if er>1e-24 else None,
        "units":"dimensionless Frobenius RMS; energies tensor-squared * physical mm3; shear squared weight2",
        "scope":"Conditional volume-weighted diagnostic cell sample, not whole-CAD/all-node accuracy"}


def run(args):
    contract = request_contract(args.plan)
    if args.dry_run:
        print(json.dumps({"status":"dry_run","contract":contract},indent=2));return
    output = path(args.output)
    formal = ROOT/"notes/geobc_hash_condition_path_result.json"
    if (output/"summary.json").exists():
        saved=read_json(output/"summary.json")
        require(saved["status"]=="completed" and saved["config"]["contract"]==contract,"Completed output contract differs")
        if not formal.exists(): atomic_json(formal,saved)
        else: require(read_json(formal)["config"]["contract"]==contract,"Formal result belongs to another contract")
        print("completed_reused");return
    require(not output.exists() or not any(output.iterdir()),"Nonempty incomplete output requires root archival before retry")
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();config={"arguments":vars(args),"contract":contract}
    atomic_json(output/"config.json",config)
    def update(status,**extra):
        atomic_json(output/"status.json",{"status":status,"elapsed_seconds":time.perf_counter()-started,
            "CAD":CAD,"backwards":0,"optimizer_updates":0,**extra})
    update("loading_input_only")
    try:
        import numpy as np
        import torch
        from geobc.gradient_supervision import cell_tensor_strain
        from geobc.hash_bandwidth_field import HashBandwidthFieldConfig,HashBandwidthPhysicsField
        require(torch.cuda.is_available(),"Frozen official native models require root's CUDA runtime")
        device=torch.device("cuda");torch.cuda.reset_peak_memory_stats(device)
        torch.set_float32_matmul_precision("high");torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        with np.load(path(contract["inputs"]["prepared"]["path"]),allow_pickle=False) as stored:
            xyz,original=stored["xyz"],stored["features"]
            require(str(stored["sample_key"].item())==CAD+"/ver" and 'official CSV GRID coordinates' in str(stored["label_alignment"].item()),"Canonical aligned input key differs")
        n=contract["gradient_metadata"]["node_count"]
        require(xyz.shape==(n,3) and original.shape==(n,25) and xyz.dtype==original.dtype==np.dtype('float32')
                and np.isfinite(xyz).all() and np.isfinite(original).all() and np.isin(original[:,:7],(0,1)).all(),"Invalid canonical input features")
        xyz_hash=hashlib.sha256(np.ascontiguousarray(xyz,dtype='<f4').tobytes()).hexdigest()
        require(xyz_hash==contract["gradient_metadata"]["xyz_sha256"],"Canonical XYZ order differs")
        # Geometry members first; do not touch the reference member until the witness is committed.
        with np.load(path(contract["inputs"]["gradient"]["path"]),allow_pickle=False) as stored:
            corners,inverse,volumes=stored["corner_ids"],stored["inverse_physical_edge_mm"],stored["volumes_mm3"]
            require(corners.shape==(contract["gradient_metadata"]["cell_count"],4) and corners.dtype==np.dtype('int64')
                    and inverse.shape==(len(corners),3,3) and volumes.shape==(len(corners),) and np.all(volumes>0),"Invalid physical cell geometry")
            chosen,changed,witness=sample_cells(corners,original[:,:7])
            atomic_json(output/"sampling_witness.json",witness)
            update("input_selection_frozen",sampling_indices_sha256=witness["indices_sha256"],cells=len(chosen))
            reference=stored["reference_cell_tensor_strain"][chosen].copy()
        with np.load(path(contract["inputs"]["local"]["path"]),allow_pickle=False) as stored:
            local=stored["features"]
        with np.load(path(contract["inputs"]["distance"]["path"]),allow_pickle=False) as stored:
            distance=stored["euclidean"]
        require(local.shape==(n,13) and distance.shape==(n,5) and local.dtype==distance.dtype==np.dtype('float32')
                and np.isfinite(local).all() and np.isfinite(distance).all(),"Invalid target-free local/distance inputs")
        ids=corners[chosen];rows=ids.reshape(-1);sample_inverse=torch.as_tensor(inverse[chosen],dtype=torch.float64,device=device)
        sample_volumes=torch.as_tensor(volumes[chosen],dtype=torch.float64,device=device)
        ref=torch.as_tensor(reference,dtype=torch.float64,device=device)
        x=torch.as_tensor(xyz[rows],device=device)
        feature=torch.as_tensor(np.concatenate((original[rows],local[rows],distance[rows]),axis=1),device=device)
        results={};network_calls=0
        with torch.no_grad():
            for mode in MODES:
                update("frozen_model",mode=mode,network_calls=network_calls)
                branch=contract["branches"][mode];cfg=read_json(branch["config"]["path"])
                model_cfg=HashBandwidthFieldConfig(**{f.name:cfg["model"][f.name] for f in fields(HashBandwidthFieldConfig)})
                model=HashBandwidthPhysicsField(model_cfg,device)
                saved=torch.load(path(branch["checkpoint"]["path"]),map_location='cpu',weights_only=False)
                require(saved["step"]==9000 and saved["config"]==cfg,"Frozen best checkpoint/config differs")
                model.load_state_dict(saved["model"]);del saved;model.eval()
                row=branch["CAD_order"].index(CAD)
                with zipfile.ZipFile(path(branch["cloud"]["path"])) as archive:
                    points=npz_row(archive,'points',row);mask=npz_row(archive,'mask',row);node_indices=npz_row(archive,'node_indices',row)
                    with archive.open('cad_keys.npy') as stream:
                        keys=np.lib.format.read_array(stream,allow_pickle=False)
                require(str(keys[row])==CAD and points.shape==(1024,5) and mask.shape==node_indices.shape==(1024,),"Selected TRAIN cloud row differs")
                require(np.array_equal(points[mask,:3],xyz[node_indices[mask]])
                        and np.array_equal(points[mask,3],np.any(original[node_indices[mask],:6],axis=1))
                        and np.array_equal(points[mask,4],original[node_indices[mask],6]),"Own frozen cloud differs from canonical inputs")
                cloud_record=branch['geometry']['CADs'][CAD]
                for member,value in (('points5',points),('coordinates',points[:,:3]),('roles',points[:,3:]),
                                     ('mask',mask),('node_indices',node_indices)):
                    require(hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()==cloud_record[member+'_sha256'],
                            f"Own TRAIN cloud {member} hash differs")
                context=model.encode_geometry(torch.as_tensor(points[None],device=device),torch.as_tensor(mask[None],device=device))
                captures=[]
                handle=model.decoder.network.register_forward_pre_hook(lambda module,inputs:captures.append(inputs[0].detach().clone()))
                try: natural=model.response_basis(x,feature,context=context)
                finally: handle.remove()
                require(len(captures)==1,"Natural preparation exceeded one Network call")
                assembled=captures[0];network_calls+=1
                require(assembled.shape==(len(rows),144) and bool(torch.isfinite(assembled).all())
                        and torch.equal(assembled[:,131:],torch.ones_like(assembled[:,131:])),"Captured public144 layout/13ones differs")
                require(bool((assembled[:,31:37]==0).all()),"Load-force slots must remain cleared in the linear basis")
                if mode=='coarse8':require(bool((assembled[:,16:24]==0).all()),"Coarse last8 masked Hash channels differ")
                replay=model.decoder.network(assembled).float();network_calls+=1
                require(torch.equal(replay.contiguous().view(torch.uint8),natural.reshape(-1,9).float().contiguous().view(torch.uint8)),"F11 replay is not bitwise equal")
                q=assembled.reshape(len(chosen),4,144);anchor=q[:,:1].expand_as(q)
                f11=replay.reshape(len(chosen),4,3,3)
                recipes={'F00':anchor.clone(),'F10':q.clone(),'F01':q.clone()}
                recipes['F10'][:,:,24:131]=anchor[:,:,24:131]
                recipes['F01'][:,:,:24]=anchor[:,:,:24]
                outputs={name:model.decoder.network(value.reshape(-1,144)).reshape(len(chosen),4,3,3) for name,value in recipes.items()}
                network_calls+=3
                require(torch.equal(outputs['F00'],outputs['F00'][:,:1].expand_as(outputs['F00'])),"Actual corner0 baseline must be constant inside each tetra")
                h,c,raw,interaction,h0,h1,c0,c1=path_differences(outputs['F00'],outputs['F10'],outputs['F01'],f11)
                require(bool(torch.allclose(raw,h+c,rtol=RTOL,atol=ATOL)),"Symmetric output path identity failed")
                per_load={}
                for load_index,load in enumerate(LOADS):
                    hs,cs,pred,inter=(cell_tensor_strain(v[:,:,load_index],sample_inverse) for v in (h,c,f11.double()*SCALE,interaction))
                    rows_out={}
                    for group in ('combined','BC_change','no_BC_change'):
                        select=torch.ones(len(chosen),dtype=torch.bool,device=device) if group=='combined' else torch.as_tensor(changed if group=='BC_change' else ~changed,device=device)
                        if not bool(select.any()):rows_out[group]=None;continue
                        rows_out[group]=physical_statistics(hs[select],cs[select],pred[select],ref[select,load_index],inter[select],sample_volumes[select])
                    per_load[load]=rows_out
                condition=q[:,:,24:131];delta=condition-condition[:,:1]
                witness_blocks={name:{"corner_difference_RMS":float(delta[:,:,lo:hi].double().square().mean().sqrt()),
                    "max_abs_corner_difference":float(delta[:,:,lo:hi].double().abs().max())}
                    for name,lo,hi in [('BC_bits',0,7),('smooth_original',13,25),('local13',25,38),('Euclidean5',38,43),('query32',43,75),('global32',75,107)]}
                directional={name:[float((v[:,:,i].double().square().sum(-1).mean(1)*sample_volumes).sum().div(sample_volumes.sum()).sqrt()) for i in range(3)]
                    for name,v in [('H_at_C0',h0),('H_at_Ck',h1),('C_at_H0',c0),('C_at_Hk',c1),('interaction',interaction)]}
                results[mode]={"checkpoint_step":9000,"F11_bitwise_replay":True,"Network_calls":5,
                    "per_load":per_load,"input_condition_witness":witness_blocks,"directed_corner_vector_RMS_mm":directional,
                    "directed_corner_RMS_weighting":"physical-volume weighted per-cell mean over4corners; three known-load axes",
                    "interaction_signed_component_mean_mm":interaction.mean((0,1)).cpu().tolist(),
                    "interaction_signed_component_min_mm":interaction.amin((0,1)).cpu().tolist(),
                    "interaction_signed_component_max_mm":interaction.amax((0,1)).cpu().tolist()}
                np.savez(output/f'{mode}_interaction.npz',interaction_mm=interaction.cpu().numpy(),selected_cells=chosen)
                del model,context,captures,assembled,q,outputs,recipes
        require(network_calls==10 and request_contract(args.plan)==contract,"Network budget or source identity changed")
        torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-started
        summary={"status":"completed","stage":contract['protocol']['stage'],"CAD":CAD,"group":"train",
            "checkpoint_steps":{"full12":9000,"coarse8":9000},"known_loads":list(LOADS),"config":config,
            "sampling":witness,"models":results,"elapsed_seconds":elapsed,"device":str(device),
            "cost":{"Network_calls":10,"natural9basis_preparations":2,"backwards":0,"optimizer_updates":0,
                    "elapsed_seconds":elapsed,"memory":memory_record(device)},
            "interpretation":contract['protocol']['interpretation'],"sample_only_no_allnode_claim":True}
        atomic_json(output/'summary.json',summary);atomic_json(formal,summary);update('completed',network_calls=10,ETA_seconds=0.)
        print(json.dumps({'status':'completed','elapsed_seconds':elapsed,'Network_calls':10}))
    except Exception as exc:
        update('failed',error=f'{type(exc).__name__}: {exc}');raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',default='notes/geobc_hash_condition_path_plan.json')
    p.add_argument('--output',default='runs/geobc_exploration/hash_condition_path_101_479')
    p.add_argument('--dry-run',action='store_true')
    return p


if __name__=='__main__':run(parser().parse_args())
