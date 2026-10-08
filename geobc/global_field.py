"""Pooled 50-CAD hash fitting with sampled updates or genuine full-node epochs."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .models import ModelConfig, build_model
from .train import load_cases, find_case_files, stress_metrics, write_json, utc_now


def run(args):
    started=time.monotonic()
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    device=torch.device(args.device)
    status={"phase":"loading","step":0,"started_at_utc":utc_now()}
    def update(phase,**extra):
        status.update(extra,phase=phase,elapsed_seconds=time.monotonic()-started,updated_at_utc=utc_now())
        step=status.get("step",0)
        eta=status["elapsed_seconds"]/step*max(0,status.get("total_steps",0)-step) if step else None
        status["eta_seconds"]=eta
        status["estimated_completion_utc"]=(datetime.now(timezone.utc)+timedelta(seconds=eta)).isoformat() if eta else None
        write_json(output/"status.json",status)
    update("loading")
    try:
        files=find_case_files(args.data,None,args.num_cases)
        cases,scaler=load_cases(files,device)
        counts=[c.count for c in cases]
        offsets=np.r_[0,np.cumsum(counts)]
        xyz=torch.cat([c.xyz for c in cases])
        features=torch.cat([c.features for c in cases])
        target=torch.cat([c.target for c in cases])
        total_nodes=len(xyz)
        batches=math.ceil(total_nodes/args.batch_size)
        total_steps=args.epochs*batches if args.epochs else args.steps
        if args.backend=="tcnn-fused":
            from .tcnn_fields import TCNNField
            model=TCNNField(input_mode="both",device=device,seed=args.seed)
        else:
            model=build_model(args.backend,ModelConfig(input_mode="both"),device)
        optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate,fused=device.type=="cuda")
        config={"arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                "method":"shared50 hashgrid","dataset":"original DeepJEB v1.0","total_nodes":total_nodes,
                "cases":[{"sample_key":c.key,"source":str(c.source),"nodes":c.count} for c in cases],
                "scaler":scaler.record(),"model":{"levels":12,"features_per_level":2,"log2_cap":19,"base":8,"growth":1.6,"MLP_width":64,"hidden_layers":2,"conditions":25,"sample_ID":False},
                "protocol":"each pooled node exactly once per complete epoch, shuffled" if args.epochs else "uniform pooled random nodes with replacement",
                "batch_size_total":args.batch_size,"batches_per_epoch":batches,"total_steps":total_steps,
                "checkpoint_selection":"minimum worst-case all-original-node decoded stress relative L2",
                "assumptions":args.assumptions,"precision":"float32 tensors; matmul precision high, TF32 allowed", "parameter_count":sum(p.numel() for p in model.parameters())}
        config["actual_backend"]=model.backend
        config["implementation_details"]=model.config.record() if hasattr(model.config,"record") else {}
        if args.backend=="tcnn-fused":config["precision"]=model.config.record()["precision"]
        if args.initialize_weights:
            if not args.fresh_adam:raise ValueError("Reference continuation requires explicitly fresh Adam")
            parent=torch.load(args.initialize_weights,map_location=device,weights_only=False)
            if parent["config"]["cases"]!=config["cases"] or parent["scaler"]!=scaler.record():raise ValueError("Initialization data/scaler differ")
            model.load_state_dict(parent["model"])
            config["initialized_from_best_weights"]=str(args.initialize_weights.resolve())
            config["parent_checkpoint_step"]=parent["step"]
            config["optimizer_continuation"]="fresh Adam; parent optimizer is deliberately not restored, matching slides"
        update("preparing_lookup",total_steps=total_steps)
        lookup=model.prepare_lookup(xyz)
        write_json(output/"config.json",config)
        best=float("inf")
        best_step=None
        def checkpoint(path,step,metrics):
            torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"config":config,"step":step,
                        "metrics":metrics,"scaler":scaler.record(),"torch_rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state_all()},path)
        @torch.no_grad()
        def evaluate(step,loss):
            nonlocal best,best_step
            model.eval()
            update("evaluating",step=step)
            encoded=np.empty(total_nodes,np.float32)
            for i in range(0,total_nodes,args.eval_batch_size):
                sl=slice(i,i+args.eval_batch_size)
                pred=model.forward_cached(lookup.select(sl),features[sl]) if lookup else model(xyz[sl],features[sl])
                encoded[sl]=pred.reshape(-1).cpu().numpy()
            prediction=scaler.decode(encoded)
            truth=np.concatenate([c.physical_stress for c in cases])
            per={c.key:stress_metrics(c.physical_stress,prediction[offsets[i]:offsets[i+1]]) for i,c in enumerate(cases)}
            worst=max(per,key=lambda k:per[k]["relative_l2"])
            score=per[worst]["relative_l2"]
            metrics={"step":step,"epoch":step/batches if args.epochs else None,"elapsed_seconds":time.monotonic()-started,
                     "training_loss":loss,"per_case":per,"pooled":stress_metrics(truth,prediction),"worst_case_key":worst,
                     "worst_case":per[worst],"case_mean_relative_l2":float(np.mean([p["relative_l2"] for p in per.values()])),
                     "worst_case_relative_l2":score,"case_mean_hotspot_relative_l2":float(np.mean([p["hotspot_relative_l2"] for p in per.values()]))}
            if score<best:
                best,best_step=score,step
                checkpoint(output/"best.pt",step,metrics)
                write_json(output/"metrics_best.json",metrics)
                folder=output/"predictions_best"
                folder.mkdir(exist_ok=True)
                index={}
                for i,c in enumerate(cases):
                    pred=prediction[offsets[i]:offsets[i+1]]
                    filename=f"case_{i:03d}.npz"
                    np.savez(folder/filename,xyz=c.xyz.cpu().numpy(),stress_pred=pred,stress_true=c.physical_stress,
                             prediction=pred[:,None],truth=c.physical_stress[:,None],channels=np.asarray([3]),sample_key=np.asarray(c.key))
                    index[c.key]={"file":filename,"source":str(c.source)}
                write_json(folder/"index.json",index)
            checkpoint(output/"last.pt",step,metrics)
            write_json(output/"metrics_latest.json",metrics)
            with (output/"metrics.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps(metrics)+"\n")
            update("training",step=step,best_step=best_step,best_score=best,latest_metrics=metrics)
            print(f"hash50 {step}/{total_steps} mean={metrics['case_mean_relative_l2']:.6g} worst={score:.6g}",flush=True)
            return metrics
        last=evaluate(0,None)
        step=0
        epochs=args.epochs or 1
        for epoch in range(epochs):
            permutation=torch.randperm(total_nodes,device=device) if args.epochs else None
            count=batches if args.epochs else args.steps
            for batch in range(count):
                idx=permutation[batch*args.batch_size:(batch+1)*args.batch_size] if permutation is not None else torch.randint(total_nodes,(args.batch_size,),device=device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                pred=model.forward_cached(lookup.select(idx),features[idx]) if lookup else model(xyz[idx],features[idx])
                loss=(pred.reshape(-1)-target[idx]).square().mean()
                loss.backward()
                optimizer.step()
                step+=1
                if not torch.isfinite(loss):raise FloatingPointError(f"Nonfinite loss at {step}")
                if step%100==0:update("training",step=step,epoch=step/batches if args.epochs else None,training_loss=float(loss.detach()))
                if not args.epochs and (step%args.eval_every==0 or step==total_steps):last=evaluate(step,float(loss.detach()))
            if args.epochs and ((epoch+1)%args.eval_every_epochs==0 or epoch+1==epochs):last=evaluate(step,float(loss.detach()))
        summary={"status":"completed","step":step,"epochs":args.epochs,"best_step":best_step,"elapsed_seconds":time.monotonic()-started,
                 "best_metrics":json.loads((output/"metrics_best.json").read_text()),"final_metrics":last,"config":config,"completed_at_utc":utc_now()}
        write_json(output/"summary.json",summary)
        update("completed",step=step)
        return summary
    except BaseException as exc:
        update("failed",error=str(exc),error_type=type(exc).__name__)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",nargs="+",required=True)
    p.add_argument("--num-cases",type=int,default=50)
    duration=p.add_mutually_exclusive_group(required=True)
    duration.add_argument("--steps",type=int)
    duration.add_argument("--epochs",type=int)
    p.add_argument("--batch-size",type=int,default=51200)
    p.add_argument("--learning-rate",type=float,default=.003)
    p.add_argument("--eval-every",type=int,default=1000)
    p.add_argument("--eval-every-epochs",type=int,default=10)
    p.add_argument("--eval-batch-size",type=int,default=65536)
    p.add_argument("--backend",choices=["torch","triton","tcnn","tcnn-fused"],default="tcnn-fused")
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--device",default="cuda")
    p.add_argument("--initialize-weights",type=Path)
    p.add_argument("--fresh-adam",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--assumptions",nargs="*",default=[])
    return p


if __name__=="__main__":run(parser().parse_args())
