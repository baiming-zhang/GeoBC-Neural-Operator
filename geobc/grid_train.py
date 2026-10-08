"""Full-grid FNO fitting and original-node evaluation; grid and node errors differ."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from .fno import FNO3d
from .train import stress_metrics, write_json, utc_now
from .benchmark import field_metrics


def run(args):
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Nonempty output: {output}")
    output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    status={"started_at_utc":utc_now(),"phase":"loading","step":0,"total_steps":args.steps}
    def update(phase,**extra):
        status.update(extra,phase=phase,elapsed_seconds=time.monotonic()-started,updated_at_utc=utc_now())
        progress=status["step"]-status.get("resume_step",0)
        eta=status["elapsed_seconds"]/progress*max(0,args.steps-status["step"]) if progress>0 else None
        status["eta_seconds"]=eta
        status["estimated_completion_utc"]=(datetime.now(timezone.utc)+timedelta(seconds=eta)).isoformat() if eta else None
        write_json(output/"status.json",status)
    update("loading")
    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        device=torch.device(args.device)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        with np.load(args.grid,allow_pickle=False) as f:
            inputs=np.asarray(f["inputs"],np.float32)
            target=np.asarray(f["targets"],np.float32)
            mask=np.asarray(f["mask"],bool)
            occupied=np.asarray(f["occupied_flat_index"],np.int64)
            occupied_xyz=np.asarray(f["occupied_xyz"],np.float32)
            node_xyz=np.asarray(f["node_xyz"],np.float32)
            node_target=np.asarray(f["node_targets"],np.float32)
            query_indices=np.asarray(f["query_indices"],np.int64)
            query_weights=np.asarray(f["query_weights"],np.float32)
            coverage=np.asarray(f["query_coverage"],bool)
            domain_lower=np.asarray(f["domain_lower_mm"] if "domain_lower_mm" in f.files else f["bbox_min_mm"],np.float64)
            domain_length=float(f["domain_length_mm"] if "domain_length_mm" in f.files else f["characteristic_length_mm"])
            key=str(f["sample_key"].item()) if "sample_key" in f.files else args.grid.stem
            meta=json.loads(str(f["metadata_json"].item())) if "metadata_json" in f.files else {}
        channels=[0,1,2,3] if target.shape[0]==4 else [3]
        if args.outputs=="stress" and len(channels)==4:
            target=target[3:4]
            node_target=node_target[:,3:4]
            channels=[3]
        if args.outputs=="both" and len(channels)!=4:
            raise ValueError("Joint uXYZ/stress run requires a four-channel grid cache")
        rows=target.reshape(len(channels),-1)[:,occupied].T.astype(np.float64)
        mean=rows.mean(0)
        std=np.maximum(rows.std(0),1e-8)
        x=torch.as_tensor(inputs[None],device=device)
        flat_indices=torch.as_tensor(occupied,device=device)
        truth_grid=torch.as_tensor(rows,dtype=torch.float32,device=device)
        mean_t=torch.as_tensor(mean,dtype=torch.float32,device=device)
        std_t=torch.as_tensor(std,dtype=torch.float32,device=device)
        y=(truth_grid-mean_t)/std_t
        model=FNO3d(in_channels=inputs.shape[0],out_channels=len(channels),width=16,modes=args.modes,layers=4).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay,fused=device.type=="cuda")
        config={"arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                "method":"FNO3d","grid_size":inputs.shape[-1],"channels":channels,
                "model":{"width":16,"modes":args.modes,"layers":4,"in_channels":inputs.shape[0],"out_channels":len(channels)},
                "model_details":model.config.record(),
                "parameter_count":sum(p.numel() for p in model.parameters()),"grid_source":str(args.grid.resolve()),
                "grid_metadata":meta,"scaler":{"mean":mean.tolist(),"std":std.tolist(),"fit":"occupied grid labels, linear per-channel"},
                "loss":"mean encoded squared error at actual occupied grid points only",
                "selection":"physical normalized MSE on occupied grid, never original nodes",
                "node_query":"mask-normalized trilinear interpolation, uncovered nodes return zero and remain in metrics",
                "uncovered_nodes":int((~coverage).sum()),"case":key,"torch_version":torch.__version__,
                "assumptions":args.assumptions,"total_steps":args.steps,"precision":"float32 tensors/FFT; matmul precision high, TF32 allowed for CUDA matmul/cuDNN"}
        start_step=0
        if args.resume:
            ckpt=torch.load(args.resume,map_location=device,weights_only=False)
            if ckpt["config"]["model"]!=config["model"] or ckpt["config"]["scaler"]!=config["scaler"]:
                raise ValueError("Resume model/scaler mismatch")
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step=ckpt["step"]
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
            if device.type=="cuda":torch.cuda.set_rng_state_all([z.cpu() for z in ckpt["cuda_rng_state"]])
            config["resumed_from"]=str(args.resume.resolve())
        write_json(output/"config.json",config)
        query_index_t=torch.as_tensor(query_indices,device=device)
        query_weight_t=torch.as_tensor(query_weights,device=device)
        def query(grid):
            flat=grid.reshape(len(channels),-1).T
            # Query data are grouped in the original H5 node order.
            return (flat[query_index_t]*query_weight_t[...,None]).sum(1)
        truth=torch.as_tensor(target,device=device)
        truth_roundtrip=query(truth).cpu().numpy()
        write_json(output/"truth_roundtrip.json",{"metrics":field_metrics(node_target,truth_roundtrip,channels),
                   "query_coverage":float(coverage.mean()),"uncovered_nodes":int((~coverage).sum()),
                   "definition":"observed representation loss of original quadratic FEM -> occupied Cartesian grid -> FEM nodes; not a mathematical lower bound"})
        best_score=float("inf")
        best_step=None
        if args.resume and (output/"metrics_best.json").is_file():
            historical=json.loads((output/"metrics_best.json").read_text())
            best_score=historical["selection_score"]
            best_step=historical["step"]
        def checkpoint(path,step,metrics):
            torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,"config":config,"metrics":metrics,
                        "torch_rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state_all() if device.type=="cuda" else None},path)
        @torch.no_grad()
        def evaluate(step,loss):
            nonlocal best_score,best_step
            model.eval()
            update("evaluating",step=step)
            encoded=model(x)[0]
            physical=encoded*std_t[:,None,None,None]+mean_t[:,None,None,None]
            physical=physical*torch.as_tensor(mask,device=device)[None]
            grid_pred=physical.reshape(len(channels),-1)[:,flat_indices].T.cpu().numpy()
            prediction=query(physical).cpu().numpy()
            score=float(np.mean(((grid_pred-rows)/std)**2))
            metrics={"step":step,"elapsed_seconds":time.monotonic()-started,"training_loss":loss,
                     "selection_score":score,"grid":field_metrics(rows,grid_pred,channels),
                     "nodes":field_metrics(node_target,prediction,channels),"sample_key":key}
            if score<best_score:
                best_score,best_step=score,step
                checkpoint(output/"best.pt",step,metrics)
                write_json(output/"metrics_best.json",metrics)
                np.savez(output/"teacher_grid_best.npz",xyz=occupied_xyz,teacher_prediction=grid_pred,channels=np.asarray(channels),
                         sample_key=np.asarray(key),teacher_step=np.asarray(step),grid_source=np.asarray(str(args.grid.resolve())),
                         domain_lower_mm=domain_lower,domain_length_mm=np.asarray(domain_length))
                pred_dir=output/"predictions_best"
                pred_dir.mkdir(exist_ok=True)
                payload={"xyz":node_xyz,"truth":node_target,"prediction":prediction,"channels":np.asarray(channels),"sample_key":np.asarray(key)}
                if 3 in channels:
                    j=channels.index(3)
                    payload.update(stress_true=node_target[:,j],stress_pred=prediction[:,j])
                np.savez(pred_dir/"case_000.npz",**payload)
                write_json(pred_dir/"index.json",{key:{"file":"case_000.npz","source":str(args.grid.resolve())}})
            checkpoint(output/"last.pt",step,metrics)
            write_json(output/"metrics_latest.json",metrics)
            with (output/"metrics.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps(metrics,allow_nan=False)+"\n")
            update("training",step=step,best_step=best_step,best_score=best_score,latest_metrics=metrics,
                   peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device.type=="cuda" else 0)
            metric="stress" if 3 in channels else "displacement"
            print(f"FNO step {step}/{args.steps} grid {metric}={metrics['grid'][metric]['relative_l2']:.6g} node={metrics['nodes'][metric]['relative_l2']:.6g}",flush=True)
            return metrics
        update("training",step=start_step,resume_step=start_step)
        last=evaluate(start_step,None)
        for step in range(start_step+1,args.steps+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            encoded=model(x)[0].reshape(len(channels),-1)[:,flat_indices].T
            loss=(encoded-y).square().mean()
            loss.backward()
            optimizer.step()
            if not torch.isfinite(loss):raise FloatingPointError(f"Nonfinite loss at {step}")
            if step%args.status_every==0:update("training",step=step,training_loss=float(loss.detach()))
            if step%args.eval_every==0 or step==args.steps:last=evaluate(step,float(loss.detach()))
        summary={"status":"completed","step":args.steps,"best_step":best_step,"elapsed_seconds":time.monotonic()-started,
                 "best_metrics":json.loads((output/"metrics_best.json").read_text()),"final_metrics":last,"config":config,"completed_at_utc":utc_now()}
        write_json(output/"summary.json",summary)
        update("completed",step=args.steps)
        return summary
    except BaseException as exc:
        update("failed",error=str(exc),error_type=type(exc).__name__)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--outputs",choices=["both","stress"],default="stress")
    p.add_argument("--modes",type=int,default=12)
    p.add_argument("--steps",type=int,default=500)
    p.add_argument("--learning-rate",type=float,default=.003)
    p.add_argument("--weight-decay",type=float,default=0)
    p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--status-every",type=int,default=10)
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--device",default="cuda")
    p.add_argument("--resume",type=Path)
    p.add_argument("--assumptions",nargs="*",default=[])
    return p


if __name__=="__main__":run(parser().parse_args())
