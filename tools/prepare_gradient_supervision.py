"""Prepare fixed known-load corner-P1 references, never solver tensor labels.

Only existing raw vertices/cells and original GRID-aligned ver/hor/tor
displacement arrays for train40+validation5 are read. Validation references
are diagnostic-only. No stress, dia, test field, model or download is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if os.name == "nt" and (ROOT/".local_deps").is_dir():
    sys.path.insert(0,str(ROOT/".local_deps"))
import numpy as np

from geobc.displacement_condition import (BASIS_ORDER, alpha_from_original_features, file_identity,
    host_path, json_sha256, load_response_manifest, verify_file_identity)
from geobc.gradient_supervision import (COMPONENT_ORDER,FROBENIUS_WEIGHTS,NORMALIZATION_FLOOR,
    ORIENTATION,REFERENCE,SCHEMA_VERSION,cell_tensor_strain,load_gradient_manifest)
from geobc.local_geometry import xyz_sha256
from tools.prepare_frozen_strain_response import read_geometry

AUTHORIZED_MEMBERS = ("sample_key","label_alignment","u_units","xyz","grid_node_ids","features","displacement")


def _json(path): return json.loads(path.read_text(encoding="utf-8"))
def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def _write(path,value):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)


def preflight(args):
    """Pure JSON/stat/small SHA. No archive, field, Torch, CUDA or H5 read."""
    parent_dir = host_path(args.u_response_cache)
    parent = load_response_manifest(parent_dir)
    parent_gate = _json(host_path(parent["gate"]["path"]))
    split_path = host_path(args.split)
    verify_file_identity(split_path,parent_gate["split"],small_sha=True)
    split = _json(split_path); groups = split["groups"]
    cads = groups["train"]+groups["validation"]
    if (len(groups["train"])!=40 or len(groups["validation"])!=5 or len(cads)!=len(set(cads))
            or set(cads)&set(groups["test"]) or set(parent["CADs"])!=set(cads)
            or split["train_loads"]!=list(BASIS_ORDER)):
        raise ValueError("Scope must be exact locked train40+validation5, ver/hor/tor only")
    guard_path = host_path(parent_gate["load_basis"]["path"])
    guard = _json(guard_path)
    guards = {r["cad_id"]:r for r in guard["cases"]}
    teacher_config = _json(host_path(parent["teacher"]["config"]["path"]))
    source_records = {r["key"]:r for r in teacher_config["contract"]["selected_sources"]}
    raw = host_path(args.raw_meshes); data = host_path(args.data)
    sources = {}; raw_sources = {}
    for cad in cads:
        p = parent["CADs"][cad]
        normalization = p["normalization"]
        if normalization.get("status")!="known" or normalization["characteristic_length_mm"]<=0 or len(normalization["bbox_min_mm"])!=3:
            raise ValueError("Exact existing isotropic physical mm normalization is required")
        raw_sources[cad] = file_identity(raw/f"{cad}.h5")
        for row,load in enumerate(BASIS_ORDER):
            key = f"{cad}/{load}"; path = host_path(split["case_paths"][key])
            if not path.is_relative_to(data) or host_path(source_records[key]["path"])!=path:
                raise ValueError("Aligned u source escaped the locked prepared directory")
            verify_file_identity(path,source_records[key])
            alpha = np.asarray(guards[cad]["alpha_by_load"][load]["alpha"],np.float64)
            if not np.array_equal(alpha,np.eye(3)[row]):
                raise ValueError("Reference axes require exact unit ver/hor/tor coefficients")
            sources[key] = {"key":key,**file_identity(path)}
    design_path = ROOT/"notes/geobc_gradient_teacher_plan.json"
    design = _json(design_path)
    config = {"schema_version":SCHEMA_VERSION,"source_parent_commit":design["source_parent_commit"],
        "design_source":file_identity(design_path,small_sha=True),"split_source":file_identity(split_path,small_sha=True),
        "parent_u_manifest":file_identity(parent_dir/"manifest.json",small_sha=True),
        "parent_response_fingerprint":parent["teacher_fingerprint"],"load_basis_source":file_identity(guard_path,small_sha=True),
        "groups":groups,"CAD_order":cads,"case_keys":[f"{cad}/{load}" for cad in cads for load in BASIS_ORDER],
        "geometry_query_identities":{cad:{k:parent["CADs"][cad][k] for k in ("node_count","xyz_sha256","normalization")} for cad in cads},
        "raw_geometry_sources":raw_sources,"displacement_sources":sources,
        "basis_order":list(BASIS_ORDER),"component_order":list(COMPONENT_ORDER),"frobenius_weights":list(FROBENIUS_WEIGHTS),
        "reference":REFERENCE,"orientation":ORIENTATION,"engineering_shear":False,"strain_units":"dimensionless",
        "displacement_units":"mm","inverse_edge_units":"1/mm","volume_units":"mm^3","normalization_floor":NORMALIZATION_FLOOR,
        "loss_formula":"mean_uniform_tets[(V/mean_full_case_V)*sum(error_tensor6^2*[1,1,1,2,2,2])]/max(train_case_full_volume_weighted_reference_squared_RMS,1e-24)",
        "reference_squared_RMS_formula":"sum_full_cells[V*sum(reference_tensor6^2*[1,1,1,2,2,2])]/sum_full_cells[V]",
        "loss_normalization_scope":"train40 x ver/hor/tor only; validation tensors and norms diagnostic-only, validation loss weights null",
        "source_members_read":{"raw_H5":["vertices","cells"],"aligned_prepared":list(AUTHORIZED_MEMBERS)},
        "source_members_excluded":["stress","dia archives","final5 test archives","H5 nodal_variables","weights"],
        "source_sha256":{"geobc/gradient_supervision.py":_sha(ROOT/"geobc/gradient_supervision.py"),
                         "tools/prepare_gradient_supervision.py":_sha(Path(__file__)),
                         "tools/prepare_frozen_strain_response.py":_sha(ROOT/"tools/prepare_frozen_strain_response.py")},
        "requested":{"split":str(split_path),"u_response_cache":str(parent_dir),"data":str(data),"raw_meshes":str(raw),
                     "output":str(host_path(args.output)),"batch_size":args.batch_size,"device":args.device},
        "limitations":"Four-corner P1 proxy uses physical straight tetra edges, ignores midside u and curved TETRA10 geometry. It is extra displacement-derived derivative supervision, not a recovered solver tensor or stress constitutive relation. FP64 gradients cannot restore native network FP16 output precision."}
    config["supervision_fingerprint"] = json_sha256(config)
    return {"config":config,"parent":parent,"guard":guard,"output":host_path(args.output)}


def _load_reference(cad,metadata):
    cfg = metadata["config"]; p = cfg["geometry_query_identities"][cad]
    xyz0 = ids0 = static0 = None; values = []; grid_sha = None
    for row,load in enumerate(BASIS_ORDER):
        key = f"{cad}/{load}"; source = cfg["displacement_sources"][key]; path = host_path(source["path"])
        verify_file_identity(path,source)
        with np.load(path,allow_pickle=False) as stored:
            if str(stored["sample_key"].item())!=key or str(stored["label_alignment"].item())!="official CSV GRID coordinates -> H5 vertices" or str(stored["u_units"].item())!="mm":
                raise ValueError("Reference must retain original CSV GRID-aligned signed physical u")
            xyz = np.asarray(stored["xyz"],np.float32); ids = np.asarray(stored["grid_node_ids"])
            features = np.asarray(stored["features"],np.float32); u = np.asarray(stored["displacement"],np.float64)
        if (xyz.shape!=(p["node_count"],3) or xyz_sha256(xyz)!=p["xyz_sha256"] or u.shape!=xyz.shape or not np.isfinite(u).all()
                or ids.shape!=(len(xyz),) or ids.dtype.kind not in "iu" or len(np.unique(ids))!=len(ids)
                or features.shape!=(len(xyz),25) or not np.isfinite(features).all()):
            raise ValueError("Reference node count, XYZ/GRID layout or u arrays differ")
        alpha = alpha_from_original_features(features,key,metadata["guard"])
        if not np.array_equal(alpha,np.eye(3,dtype=np.float32)[row]):
            raise ValueError("Reference force/moment slots do not match unit response basis")
        features[:,7:13] = 0
        if row==0:
            xyz0,ids0,static0 = xyz,ids,features
            grid_sha = hashlib.sha256(np.ascontiguousarray(ids,dtype="<i8").tobytes()).hexdigest()
        elif not np.array_equal(xyz,xyz0) or not np.array_equal(ids,ids0) or not np.array_equal(features,static0):
            raise ValueError("Known loads changed normalized geometry, GRID order or nonload features")
        values.append(u); verify_file_identity(path,source)
    return np.stack(values,axis=1),grid_sha


def _prepare_arrays(coordinates,cells,u_mm,length,batch_size,device):
    import torch
    corners = np.ascontiguousarray(cells[:,:4],np.int64)
    count = len(cells)
    inverse = np.empty((count,3,3),np.float64); volumes = np.empty(count,np.float64)
    reference = np.empty((count,3,6),np.float32)
    with torch.no_grad():
        for start in range(0,count,batch_size):
            end = min(count,start+batch_size); ids = corners[start:end]
            # Original raw-mm vertices were promoted before normalization by read_geometry.
            edge = (coordinates[ids[:,1:]]-coordinates[ids[:,:1]])*length
            d = torch.as_tensor(edge,dtype=torch.float64,device=device)
            det = torch.linalg.det(d); inv = torch.linalg.inv(d); vol = det.abs()/6
            if not bool(torch.isfinite(inv).all()) or not bool(torch.isfinite(vol).all()) or not bool((vol>0).all()):
                raise ValueError("Nonfinite/degenerate physical corner tetrahedron; none skipped")
            inverse[start:end] = inv.cpu().numpy(); volumes[start:end] = vol.cpu().numpy()
            for load in range(3):
                u = torch.as_tensor(u_mm[ids,load,:],dtype=torch.float64,device=device)
                reference[start:end,load,:] = cell_tensor_strain(u,inv).float().cpu().numpy()
            del d,det,inv,vol,u
    if not np.isfinite(reference).all(): raise FloatingPointError("Nonfinite reference P1 tensor")
    weights = np.asarray(FROBENIUS_WEIGHTS,np.float64)
    reference64 = reference.astype(np.float64)
    squared = np.sum(volumes[:,None]*np.sum(reference64**2*weights,axis=-1),axis=0)/volumes.sum()
    if not np.isfinite(squared).all(): raise FloatingPointError("Nonfinite full-volume reference RMS")
    return corners,inverse,volumes,reference,{load:float(squared[row]) for row,load in enumerate(BASIS_ORDER)}


def _partial_record(path,config,cad):
    with np.load(path,allow_pickle=False) as stored:
        record = json.loads(str(stored["record_json"].item()))
        if str(stored["supervision_fingerprint"].item())!=config["supervision_fingerprint"] or record["cad_id"]!=cad or not record["finite"]:
            raise ValueError("Partial cache differs from identical resume contract")
    return {"file":path.name,**{k:v for k,v in file_identity(path).items() if k!="path"},**record}


def generate(args,metadata):
    import torch
    cfg = metadata["config"]; output = metadata["output"]
    if (output/"manifest.json").is_file():
        existing = load_gradient_manifest(output)
        if existing["supervision_fingerprint"]!=cfg["supervision_fingerprint"]:
            raise ValueError("Completed cache has different geometry, labels, source or recipe")
        print(json.dumps({"status":"completed","reused":True,"output":str(output)}),flush=True)
        return existing
    if output.exists() and any(output.iterdir()):
        if not args.resume or not (output/"config.json").is_file() or _json(output/"config.json")!=cfg:
            raise FileExistsError("Nonempty incomplete directory needs --resume and identical full contract")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Formal preparation uses root-authorized CUDA queue; --dry-run is metadata-only")
    output.mkdir(parents=True,exist_ok=True)
    if not (output/"config.json").exists(): _write(output/"config.json",cfg)
    previous = _json(output/"status.json").get("elapsed_seconds",0) if args.resume and (output/"status.json").exists() else 0
    begun = time.monotonic(); records = {}; status = {"completed_CADs":0,"total_CADs":45}
    def update(phase,**extra):
        elapsed = previous+time.monotonic()-begun; done = status["completed_CADs"]
        status.update(extra,phase=phase,elapsed_seconds=elapsed,eta_seconds=elapsed/done*(45-done) if done else None,
                      updated_at_utc=datetime.now(timezone.utc).isoformat())
        _write(output/"status.json",status)
    update("preparing_gradient_references")
    try:
        for cad in cfg["CAD_order"]:
            update("preparing_gradient_references",current_CAD=cad)
            path = output/f"{cad}.npz"; p = cfg["geometry_query_identities"][cad]
            if path.exists():
                if not args.resume: raise FileExistsError("Existing CAD cache requires explicit resume")
                records[cad] = _partial_record(path,cfg,cad)
            else:
                raw_source = cfg["raw_geometry_sources"][cad]
                verify_file_identity(host_path(raw_source["path"]),raw_source)
                coords,cells = read_geometry(host_path(raw_source["path"]),p["normalization"],p["xyz_sha256"],p["node_count"])
                if cells.min()<0 or cells.max()>=p["node_count"] or len(np.unique(cells))!=p["node_count"]:
                    raise ValueError("Original TETRA10 connectivity misses/out-of-range original nodes")
                u,grid_sha = _load_reference(cad,metadata)
                corners,inverse,volumes,ref,norms = _prepare_arrays(coords,cells,u,p["normalization"]["characteristic_length_mm"],args.batch_size,args.device)
                verify_file_identity(host_path(raw_source["path"]),raw_source)
                train = cad in cfg["groups"]["train"]
                record = {"cad_id":cad,**p,"cell_count":len(cells),"grid_node_ids_sha256":grid_sha,
                    "raw_geometry_source":raw_source,"displacement_sources":[cfg["displacement_sources"][f"{cad}/{load}"] for load in BASIS_ORDER],
                    "mean_volume_mm3":float(volumes.mean()),"sum_volume_mm3":float(volumes.sum()),"min_volume_mm3":float(volumes.min()),"max_volume_mm3":float(volumes.max()),
                    "reference_squared_rms_by_load":norms,"loss_normalization_by_load":{k:max(v,NORMALIZATION_FLOOR) for k,v in norms.items()} if train else None,
                    "floor_active_by_load":{k:v<NORMALIZATION_FLOOR for k,v in norms.items()} if train else None,
                    "role":"train_loss_and_diagnostic" if train else "validation_diagnostic_only","finite":True,
                    "geometry_dtype":"float64","reference_dtype":"float32","reference":REFERENCE,"strain_units":"dimensionless"}
                temp = path.with_suffix(".npz.tmp")
                with temp.open("wb") as handle:
                    np.savez(handle,corner_ids=corners,cells10=cells,inverse_physical_edge_mm=inverse,volumes_mm3=volumes,
                             reference_cell_tensor_strain=ref,supervision_fingerprint=np.asarray(cfg["supervision_fingerprint"]),
                             record_json=np.asarray(json.dumps(record,sort_keys=True,allow_nan=False)))
                temp.replace(path)
                records[cad] = {"file":path.name,**{k:v for k,v in file_identity(path).items() if k!="path"},**record}
                del coords,cells,u,corners,inverse,volumes,ref
            status["completed_CADs"] += 1
            _write(output/"progress.json",{"supervision_fingerprint":cfg["supervision_fingerprint"],"CADs":records})
            update("preparing_gradient_references")
        # Release CUDA tensors/context allocations before the completion prerequisite.
        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        elapsed = previous+time.monotonic()-begun
        manifest = {**cfg,"CADs":records,"status":"completed","complete":True,
            "config_source":file_identity(output/"config.json",small_sha=True),
            "completed_at_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":elapsed,
            "cache_bytes":sum(r["bytes"] for r in records.values()),"total_cells":sum(r["cell_count"] for r in records.values()),
            "total_nodes":sum(r["node_count"] for r in records.values()),"train_loss_cases":120,"validation_loss_cases":0,
            "train_floor_active_cases":[f"{cad}/{load}" for cad in cfg["groups"]["train"] for load in BASIS_ORDER if records[cad]["floor_active_by_load"][load]],
            "CUDA_released_before_manifest":True,"cost":{"preparation_elapsed_seconds":elapsed,"training_updates":0}}
        _write(output/"manifest.json",manifest)
        update("completed",current_CAD=None)
        print(json.dumps({"status":"completed","CADs":45,"cache_bytes":manifest["cache_bytes"],"elapsed_seconds":elapsed,
                          "floor_active_cases":manifest["train_floor_active_cases"],"output":str(output)}),flush=True)
        return manifest
    except Exception as error:
        update("failed",error=f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",default="notes/geobc_split.json")
    parser.add_argument("--u-response-cache",default="data/geobc_frozen_u_response")
    parser.add_argument("--data",default="data/prepared_geobc_multiload")
    parser.add_argument("--raw-meshes",default="data/raw/FieldMesh")
    parser.add_argument("--output",default="data/geobc_gradient_supervision")
    parser.add_argument("--batch-size",type=int,default=262144)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    args = parser.parse_args()
    if args.batch_size<1: parser.error("--batch-size must be positive")
    metadata = preflight(args)
    if args.dry_run:
        print(json.dumps({"status":"metadata_dry_run","CADs":45,"cases":135,"train_loss_cases":120,
                          "validation_loss_cases":0,"arrays_opened":False,"CUDA_used":False,
                          "supervision_fingerprint":metadata["config"]["supervision_fingerprint"]}),flush=True)
    else: generate(args,metadata)


if __name__=="__main__": main()
