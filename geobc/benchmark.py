"""Full-node, multifield reproduction runner for the methods in the slides.

Transformers always see an entire FEM case. Pointwise fields may be evaluated
in chunks. Full-case updates and sampled hash updates are separate protocols.
Unspecified historical options are recorded as assumptions, not recovered facts.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .models import ModelConfig, HashGridConfig, build_model
from .train import find_case_files, stress_metrics, write_json, utc_now


@dataclass
class FieldScaler:
    mean: list[float]
    std: list[float]
    stress_scale: float | None
    channels: list[int]

    @classmethod
    def fit(cls, targets, channels, stress_transform="linear"):
        physical = np.concatenate([x[:, channels].astype(np.float64) for x in targets])
        scale = None
        if stress_transform == "asinh" and 3 in channels:
            j = channels.index(3)
            scale = max(float(np.quantile(np.abs(physical[:, j]), .9)), 1e-12)
            physical[:, j] = np.arcsinh(physical[:, j] / scale)
        return cls(physical.mean(0).tolist(), np.maximum(physical.std(0), 1e-8).tolist(), scale, channels)

    def encode(self, target):
        result = np.asarray(target, dtype=np.float64)[:, self.channels].copy()
        if self.stress_scale is not None:
            j = self.channels.index(3)
            result[:, j] = np.arcsinh(result[:, j] / self.stress_scale)
        return ((result - np.asarray(self.mean)) / np.asarray(self.std)).astype(np.float32)

    def decode(self, prediction):
        result = np.asarray(prediction, dtype=np.float64) * np.asarray(self.std) + np.asarray(self.mean)
        if self.stress_scale is not None:
            j = self.channels.index(3)
            result[:, j] = self.stress_scale * np.sinh(result[:, j])
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite physical field after decoding")
        return result


@dataclass
class Case:
    key: str
    path: Path
    xyz: torch.Tensor
    features: torch.Tensor
    physical: np.ndarray
    target: torch.Tensor | None = None
    lookup: object = None
    reference: np.ndarray | None = None


def load_fields(paths, device):
    cases = []
    for path in paths:
        with np.load(path, allow_pickle=False) as f:
            if "label_alignment" not in f.files or str(f["label_alignment"].item()) != "official CSV GRID coordinates -> H5 vertices":
                raise ValueError(f"Unverified nodal-field alignment: {path}")
            xyz = np.asarray(f["xyz"], np.float32)
            features = np.asarray(f["features"], np.float32)
            u = np.asarray(f["displacement"], np.float32)
            stress = np.asarray(f["stress"], np.float32)
            key = str(f["sample_key"].item())
        if u.shape != (len(xyz), 3) or features.shape != (len(xyz), 25):
            raise ValueError(f"Incorrect multifield shape: {path}")
        physical = np.column_stack((u, stress))
        if not all(np.isfinite(x).all() for x in (xyz, features, physical)):
            raise ValueError(f"Nonfinite fields in {path}")
        cases.append(Case(key, path, torch.as_tensor(xyz, device=device),
                          torch.as_tensor(features, device=device), physical))
    return cases


def field_metrics(truth, prediction, channels):
    result = {}
    if 3 in channels:
        j = channels.index(3)
        result["stress"] = stress_metrics(truth[:, j], prediction[:, j])
    u_indices = [channels.index(i) for i in (0, 1, 2) if i in channels]
    if u_indices:
        u, pred_u = truth[:, u_indices], prediction[:, u_indices]
        result["displacement"] = {
            "relative_l2": float(np.linalg.norm(u - pred_u) / max(np.linalg.norm(u), 1e-12)),
            "rmse_mm": float(np.sqrt(np.mean((u - pred_u) ** 2))),
            "max_vector_error_mm": float(np.linalg.norm(u - pred_u, axis=1).max()),
        }
    return result


def model_factory(args, device):
    output_dim = len(args.channels)
    if args.backend == "tcnn-fused":
        if args.method != "hashgrid":
            raise ValueError("The fused reference field backend supports the 12-level hashgrid method")
        from .tcnn_fields import TCNNField
        return TCNNField(output_dim,args.input_mode,device=device,seed=args.seed)
    if args.method in ("transolver", "transolverpp", "ts3"):
        from .transformers import build_transformer
        return build_transformer(args.method, input_dim=28, output_dim=output_dim, device=device,
                                 checkpoint_blocks=args.checkpoint_blocks)
    if args.method == "mlp":
        config = ModelConfig(encoder="mlp", input_mode=args.input_mode,
                             hidden_dim=256, hidden_layers=4)
        backend = "torch"
    else:
        small = args.method == "ngp4"
        grid = HashGridConfig(n_levels=4, log2_hashmap_size=args.small_hash_log2,
                              base_resolution=args.small_base, per_level_scale=args.small_growth) if small else HashGridConfig()
        config = ModelConfig(encoder="hashgrid", input_mode=args.input_mode,
                             hidden_dim=args.small_width if small else 64,
                             hidden_layers=2, hash_grid=grid)
        backend = args.backend
    model = build_model(config=config, backend=backend, device=device)
    model.mlp[-1] = nn.Linear(model.mlp[-1].in_features, output_dim).to(device)
    return model


def load_references(cases, folders, channels):
    covered = {case.key: {} for case in cases}
    for folder in folders:
        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text())
        source_channels = config["channels"]
        index = json.loads((folder / "predictions_best" / "index.json").read_text())
        for case in cases:
            if case.key not in index:
                continue
            entry = index[case.key]
            path = folder / "predictions_best" / entry["file"]
            with np.load(path, allow_pickle=False) as f:
                prediction = f["prediction"]
                if prediction.shape != (len(case.xyz), len(source_channels)):
                    raise ValueError("Reference point count/channel mismatch")
                if not np.array_equal(f["xyz"], case.xyz.cpu().numpy()):
                    raise ValueError("Frozen reference coordinate/order mismatch")
                for j, c in enumerate(source_channels):
                    if c in covered[case.key]:
                        raise ValueError("Duplicate frozen-reference output channel")
                    covered[case.key][c] = prediction[:, j]
    for case in cases:
        if any(c not in covered[case.key] for c in channels):
            raise ValueError(f"Missing frozen reference for {case.key}, requested channels {channels}")
        case.reference = np.column_stack([covered[case.key][c] for c in channels])


def run(args):
    started = time.monotonic()
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.channels = {"both": [0, 1, 2, 3], "stress": [3], "u": [0, 1, 2]}[args.outputs]
    is_context = args.method in ("transolver", "transolverpp", "ts3")
    if is_context and args.protocol != "full-case":
        raise ValueError("Contextual models require a full FEM case; point subsampling changes the method")
    if args.reference_runs and args.stress_transform != "linear":
        raise ValueError("Residual references use linear physical output scaling")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    status = {"phase": "loading", "started_at_utc": utc_now(), "step": 0}
    def update(phase, **extra):
        status.update(extra, phase=phase, updated_at_utc=utc_now(), elapsed_seconds=time.monotonic()-started)
        progress = status.get("step", 0) - status.get("resume_step", 0)
        remaining = max(0, status.get("total_steps", 0)-status.get("step", 0))
        eta = status["elapsed_seconds"] / progress * remaining if progress > 0 else None
        status["eta_seconds"] = eta
        status["estimated_completion_utc"] = (datetime.now(timezone.utc)+timedelta(seconds=eta)).isoformat() if eta else None
        write_json(output / "status.json", status)
    update("loading")
    try:
        paths = find_case_files(args.data, args.case_keys, args.num_cases)
        cases = load_fields(paths, device)
        scaler = FieldScaler.fit([case.physical for case in cases], args.channels, args.stress_transform)
        if args.reference_runs:
            load_references(cases, args.reference_runs, args.channels)
        for case in cases:
            target = scaler.encode(case.physical)
            if case.reference is not None:
                target = ((case.physical[:, args.channels]-case.reference) / np.asarray(scaler.std)).astype(np.float32)
            case.target = torch.as_tensor(target, device=device)
        model = model_factory(args, device)
        # Cache only fixed-coordinate interpolation, never predictions.
        if not is_context and getattr(model, "encoding", None) is not None:
            for case in cases:
                case.lookup = model.prepare_lookup(case.xyz)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay, fused=device.type == "cuda")
        total_steps = args.steps if args.steps is not None else args.epochs * len(cases)
        if args.protocol == "sampled" and args.epochs:
            raise ValueError("Sampled updates are not full-data epochs")
        config = {"arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  "method": args.method, "channels": args.channels, "scaler": asdict(scaler),
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "model": asdict(model.config) if hasattr(model, "config") and hasattr(model.config, "__dataclass_fields__") else getattr(model, "source_config", {}),
                  "implementation_details": model.config.record() if hasattr(model,"config") and hasattr(model.config,"record") else getattr(model,"source_config",{}),
                  "dataset": "original DeepJEB v1.0", "cases": [{"sample_key": c.key, "source": str(c.path), "nodes":len(c.xyz)} for c in cases],
                  "training_protocol": "one complete FEM case per update, cycling cases" if args.protocol == "full-case" else "one case-balanced random batch per update",
                  "evaluation": "every original node; contextual methods always use complete case context",
                  "checkpoint_selection": args.selection,
                  "physical_mse_definition": "case-balanced mean of per-case MSE after dividing each physical output channel by its pooled population std; historical case aggregation unavailable",
                  "reference_usage": "frozen source field is added only at output; NOT a model input" if args.reference_runs else None,
                  "assumptions": args.assumptions, "total_steps": total_steps,
                  "torch_version": torch.__version__, "precision": "float32 tensors; matmul precision high, TF32 allowed for CUDA matmul/cuDNN",
                  "started_at_utc": utc_now()}
        if args.backend=="tcnn-fused":
            config["precision"]=model.config.record()["precision"]
        start_step = 0
        best_score = math.inf
        best_step = None
        if args.resume and (output/"metrics_best.json").is_file():
            historical=json.loads((output/"metrics_best.json").read_text())
            best_score=historical["selection_score"]
            best_step=historical["step"]
        if args.resume:
            saved = torch.load(args.resume, map_location=device, weights_only=False)
            if saved["config"]["cases"] != config["cases"] or saved["scaler"] != asdict(scaler):
                raise ValueError("Resume case/scaler mismatch")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            if args.override_resume_lr:
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate
            start_step = saved["step"]
            torch.set_rng_state(saved["torch_rng_state"].cpu())
            if device.type == "cuda": torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng_state"]])
            np.random.set_state(saved["numpy_rng_state"])
            random.setstate(saved["random_state"])
            config["resumed_from"] = str(args.resume.resolve())
        write_json(output/"config.json", config)
        update("training", step=start_step, resume_step=start_step, total_steps=total_steps, method=args.method)
        physical_std = np.maximum(np.concatenate([c.physical[:,args.channels] for c in cases]).astype(np.float64).std(0),1e-8)

        def forward(case, indices):
            if is_context:
                x = torch.cat((case.xyz, case.features), -1).unsqueeze(0)
                return model(x).reshape(len(case.xyz), len(args.channels))
            if case.lookup is not None:
                result = model.forward_cached(case.lookup.select(indices), case.features[indices])
            else:
                result = model(case.xyz[indices], case.features[indices])
            return result.reshape(-1, len(args.channels))

        def checkpoint(path, step, metrics):
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step":step,
                        "scaler":asdict(scaler), "config":config, "metrics":metrics,
                        "torch_rng_state":torch.get_rng_state(), "cuda_rng_state":torch.cuda.get_rng_state_all() if device.type=="cuda" else None,
                        "numpy_rng_state":np.random.get_state(), "random_state":random.getstate()}, path)

        @torch.no_grad()
        def evaluate(step, loss):
            nonlocal best_score, best_step
            model.eval()
            update("evaluating",step=step)
            per_case, predictions = {}, {}
            mse = []
            for case in cases:
                if is_context:
                    encoded = forward(case, slice(None)).float().cpu().numpy()
                else:
                    encoded = np.concatenate([forward(case,slice(i,i+args.eval_batch_size)).float().cpu().numpy() for i in range(0,len(case.xyz),args.eval_batch_size)])
                prediction = scaler.decode(encoded) if case.reference is None else case.reference+encoded*np.asarray(scaler.std)
                truth = case.physical[:,args.channels]
                predictions[case.key] = prediction
                per_case[case.key] = field_metrics(truth,prediction,args.channels)
                mse.append(float(np.mean(((prediction-truth)/physical_std)**2)))
            metric_field = "stress" if 3 in args.channels else "displacement"
            worst_key = max(per_case,key=lambda key:per_case[key][metric_field]["relative_l2"])
            score = max(p[metric_field]["relative_l2"] for p in per_case.values()) if args.selection == "worst-stress" else float(np.mean(mse))
            metrics = {"step":step,"epoch":step/len(cases) if args.protocol=="full-case" else None,
                       "elapsed_seconds":time.monotonic()-started,"training_loss":loss,"selection_score":score,
                       "physical_normalized_mse":float(np.mean(mse)), "per_case":per_case,"worst_case_key":worst_key,
                       "pooled":field_metrics(np.concatenate([c.physical[:,args.channels] for c in cases]),np.concatenate([predictions[c.key] for c in cases]),args.channels),
                       "case_mean_relative_l2":float(np.mean([p[metric_field]["relative_l2"] for p in per_case.values()])),
                       "worst_case_relative_l2":per_case[worst_key][metric_field]["relative_l2"]}
            if score < best_score:
                best_score, best_step = score, step
                checkpoint(output/"best.pt",step,metrics)
                pred_dir = output/"predictions_best"
                pred_dir.mkdir(exist_ok=True)
                index={}
                for i,case in enumerate(cases):
                    filename=f"case_{i:03d}.npz"
                    # Both generic matrices and stress-specific fields for plotting.
                    payload={"xyz":case.xyz.cpu().numpy(), "prediction":predictions[case.key], "truth":case.physical[:,args.channels], "channels":np.asarray(args.channels),"sample_key":np.asarray(case.key)}
                    if 3 in args.channels:
                        j=args.channels.index(3)
                        payload.update(stress_pred=predictions[case.key][:,j], stress_true=case.physical[:,3])
                    np.savez(pred_dir/filename,**payload)
                    index[case.key]={"file":filename,"source":str(case.path)}
                write_json(pred_dir/"index.json",index)
                write_json(output/"metrics_best.json",metrics)
            checkpoint(output/"last.pt",step,metrics)
            write_json(output/"metrics_latest.json",metrics)
            with (output/"metrics.jsonl").open("a",encoding="utf8") as f:
                f.write(json.dumps(metrics,allow_nan=False)+"\n")
            update("training",step=step,best_step=best_step,best_score=best_score,latest_metrics=metrics,
                   peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device.type=="cuda" else 0)
            print(f"{args.method} step {step}/{total_steps} {metric_field} mean={metrics['case_mean_relative_l2']:.6g} worst={metrics['worst_case_relative_l2']:.6g}",flush=True)
            return metrics

        last=evaluate(start_step,None)
        for step in range(start_step+1,total_steps+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if args.protocol == "full-case":
                chosen=[(cases[(step-1)%len(cases)],slice(None))]
            else:
                chosen=[(c,torch.randint(len(c.xyz),(args.nodes_per_case,),device=device)) for c in cases]
            losses=[]
            for case,indices in chosen:
                prediction=forward(case,indices)
                loss=(prediction-case.target[indices]).square().mean()/len(chosen)
                loss.backward()
                losses.append(loss.detach())
            optimizer.step()
            loss_value=float(torch.stack(losses).sum())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"Nonfinite loss at {step}")
            if step%args.status_every==0:
                update("training",step=step,training_loss=loss_value)
            if step%args.eval_every==0 or step==total_steps:
                last=evaluate(step,loss_value)
        summary={"status":"completed","step":total_steps,"best_step":best_step,"best_score":best_score,
                 "elapsed_seconds":time.monotonic()-started,"best_metrics":json.loads((output/"metrics_best.json").read_text()),
                 "final_metrics":last,"config":config,"completed_at_utc":utc_now()}
        write_json(output/"summary.json",summary)
        update("completed",step=total_steps)
        return summary
    except BaseException as exc:
        update("failed",error_type=type(exc).__name__,error=str(exc))
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method",choices=["mlp","ngp4","hashgrid","transolver","transolverpp","ts3"],required=True)
    p.add_argument("--data",nargs="+",default=["data/prepared_multifield"])
    p.add_argument("--case-keys",nargs="+")
    p.add_argument("--num-cases",type=int,default=1)
    p.add_argument("--outputs",choices=["both","stress","u"],default="both")
    p.add_argument("--input-mode",choices=["xyz","both"],default="both")
    p.add_argument("--stress-transform",choices=["linear","asinh"],default="linear")
    p.add_argument("--protocol",choices=["full-case","sampled"],default="full-case")
    duration=p.add_mutually_exclusive_group(required=True)
    duration.add_argument("--steps",type=int)
    duration.add_argument("--epochs",type=int)
    p.add_argument("--nodes-per-case",type=int,default=16384)
    p.add_argument("--learning-rate",type=float,default=.003)
    p.add_argument("--weight-decay",type=float,default=0)
    p.add_argument("--selection",choices=["physical-mse","worst-stress"],default="physical-mse")
    p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--status-every",type=int,default=10)
    p.add_argument("--eval-batch-size",type=int,default=65536)
    p.add_argument("--backend",default="triton",choices=["torch","triton","tcnn","tcnn-fused"])
    p.add_argument("--device",default="cuda")
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--checkpoint-blocks",action="store_true")
    p.add_argument("--small-hash-log2",type=int,default=14)
    p.add_argument("--small-base",type=int,default=8)
    p.add_argument("--small-growth",type=float,default=2)
    p.add_argument("--small-width",type=int,default=32)
    p.add_argument("--reference-runs",nargs="+")
    p.add_argument("--assumptions",nargs="*",default=[])
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--resume",type=Path)
    p.add_argument("--override-resume-lr",action="store_true")
    return p


if __name__=="__main__":
    run(parser().parse_args())
