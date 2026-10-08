"""Fit only the occupied-grid FNO teacher; FEM ground truth is evaluation-only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .models import build_model, ModelConfig
from .train import StressScaler, stress_metrics, write_json, utc_now


def run(args):
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    output.mkdir(parents=True)
    started=time.monotonic()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    device=torch.device(args.device)
    try:
        with np.load(args.teacher,allow_pickle=False) as f:
            xyz=np.asarray(f["xyz"],np.float32)
            source_channels=f["channels"].tolist()
            teacher=np.asarray(f["teacher_prediction"],np.float64)[:,source_channels.index(3)]
            key=str(f["sample_key"].item())
            teacher_step=int(f["teacher_step"])
            domain_lower=np.asarray(f["domain_lower_mm"],np.float64)
            domain_length=float(f["domain_length_mm"])
        # This scaler is fit ONLY on the teacher predictions, not original FEM.
        scaler=StressScaler.fit([teacher])
        xyz_t=torch.as_tensor(xyz,device=device)
        y=torch.as_tensor(scaler.encode(teacher),device=device)
        if args.backend=="tcnn-fused":
            from .tcnn_fields import TCNNField
            model=TCNNField(input_mode="xyz",device=device,seed=args.seed)
        else:
            model=build_model(args.backend,ModelConfig(input_mode="xyz"),device)
        teacher_lookup=model.prepare_lookup(xyz_t)
        with np.load(args.nodes,allow_pickle=False) as f:
            node_xyz=((np.asarray(f["xyz_mm"],np.float64)-domain_lower)/domain_length).astype(np.float32)
            node_truth=np.asarray(f["stress"],np.float64)
            if str(f["sample_key"].item())!=key:raise ValueError("Teacher and FEM case differ")
        node_xyz_t=torch.as_tensor(node_xyz,device=device)
        node_lookup=model.prepare_lookup(node_xyz_t)
        optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate,fused=device.type=="cuda")
        config={"arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                "method":"hashgrid FNO teacher distillation","teacher_source":str(args.teacher.resolve()),"teacher_step":teacher_step,
                "supervision":"only FNO physical predictions at occupied Cartesian grid points",
                "FEM_ground_truth_usage":"all-node evaluation only; excluded from scaler, training loss, teacher selection and checkpoint selection",
                "scaler":scaler.record(),"case":key,"teacher_points":len(xyz),"FEM_nodes":len(node_xyz),
                "model":{"levels":12,"features_per_level":2,"log2_cap":19,"base":8,"growth":1.6,"MLP_width":64,"MLP_hidden_layers":2},
                "selection":"minimum decoded teacher-point relative L2","parameter_count":sum(p.numel() for p in model.parameters()),
                "coordinate_domain":{"lower_mm":domain_lower.tolist(),"length_mm":domain_length,"definition":"teacher and FEM queries use the same FNO physical cube"},
                "sampling":"random teacher points with replacement","total_steps":args.steps,"assumptions":args.assumptions}
        config["actual_backend"]=model.backend
        config["implementation_details"]=model.config.record() if hasattr(model.config,"record") else {}
        write_json(output/"config.json",config)
        best=float("inf")
        best_step=None
        def checkpoint(path,step,metrics):
            torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,"metrics":metrics,"config":config,
                        "scaler":scaler.record(),"torch_rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state_all()},path)
        def predict(coords,lookup):
            values=[]
            for i in range(0,len(coords),args.eval_batch_size):
                sl=slice(i,i+args.eval_batch_size)
                z=model.forward_cached(lookup.select(sl)) if lookup else model(coords[sl])
                values.append(z.reshape(-1).cpu().numpy())
            return scaler.decode(np.concatenate(values))
        @torch.no_grad()
        def evaluate(step,loss):
            nonlocal best,best_step
            model.eval()
            teacher_pred=predict(xyz_t,teacher_lookup)
            node_pred=predict(node_xyz_t,node_lookup)
            metrics={"step":step,"training_loss":loss,"elapsed_seconds":time.monotonic()-started,
                     "teacher":stress_metrics(teacher,teacher_pred),"nodes":stress_metrics(node_truth,node_pred),"sample_key":key}
            score=metrics["teacher"]["relative_l2"]
            if score<best:
                best,best_step=score,step
                checkpoint(output/"best.pt",step,metrics)
                write_json(output/"metrics_best.json",metrics)
                pred_dir=output/"predictions_best"
                pred_dir.mkdir(exist_ok=True)
                np.savez(pred_dir/"case_000.npz",xyz=node_xyz,stress_true=node_truth,stress_pred=node_pred,
                         prediction=node_pred[:,None],truth=node_truth[:,None],channels=np.asarray([3]),sample_key=np.asarray(key))
                write_json(pred_dir/"index.json",{key:{"file":"case_000.npz","source":str(args.nodes.resolve())}})
            checkpoint(output/"last.pt",step,metrics)
            write_json(output/"metrics_latest.json",metrics)
            with (output/"metrics.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps(metrics)+"\n")
            write_json(output/"status.json",{"phase":"training","step":step,"total_steps":args.steps,"best_step":best_step,"elapsed_seconds":time.monotonic()-started,"updated_at_utc":utc_now()})
            print(f"distill {step}/{args.steps} teacher={score:.6g} FEM={metrics['nodes']['relative_l2']:.6g}",flush=True)
            return metrics
        last=evaluate(0,None)
        for step in range(1,args.steps+1):
            model.train()
            idx=torch.randint(len(xyz_t),(args.nodes_per_update,),device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction=model.forward_cached(teacher_lookup.select(idx)) if teacher_lookup else model(xyz_t[idx])
            loss=(prediction.reshape(-1)-y[idx]).square().mean()
            loss.backward()
            optimizer.step()
            if step%args.eval_every==0 or step==args.steps:last=evaluate(step,float(loss.detach()))
        summary={"status":"completed","step":args.steps,"best_step":best_step,"elapsed_seconds":time.monotonic()-started,
                 "best_metrics":json.loads((output/"metrics_best.json").read_text()),"final_metrics":last,"config":config,"completed_at_utc":utc_now()}
        write_json(output/"summary.json",summary)
        write_json(output/"status.json",{"phase":"completed","step":args.steps,"total_steps":args.steps,"elapsed_seconds":summary["elapsed_seconds"],"updated_at_utc":utc_now()})
        return summary
    except BaseException as exc:
        write_json(output/"status.json",{"phase":"failed","error":str(exc),"error_type":type(exc).__name__,"elapsed_seconds":time.monotonic()-started,"updated_at_utc":utc_now()})
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--nodes",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--backend",choices=["torch","triton","tcnn","tcnn-fused"],default="tcnn-fused")
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--nodes-per-update",type=int,default=16384)
    p.add_argument("--eval-batch-size",type=int,default=65536)
    p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--learning-rate",type=float,default=.003)
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--device",default="cuda")
    p.add_argument("--assumptions",nargs="*",default=[])
    return p


if __name__=="__main__":run(parser().parse_args())
