"""Train the geometry/control pair on a fixed CAD split, stress only.

Only train/validation CADs and train_loads are opened. Test CADs and the
held-out load are neither loaded nor evaluated. Best selection uses validation
case-mean physical relative L2. Sampling RNGs are independent of model RNGs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch

from .conditioned_field import ConditionedField, ConditionedFieldConfig, load_geometry_cloud
from .train import load_cases, stress_metrics, utc_now, write_json

ROOT = Path(__file__).resolve().parents[1]


def _path(value, base=ROOT):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", value):
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save_checkpoint(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restore_history_row(output, metrics):
    """Repair only a trailing partial append and a missing committed eval row."""
    path = output / "metrics.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    rows, valid_lines, changed = [], [], False
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError:
            if i != len(lines) - 1:
                raise ValueError("Evaluation history contains a malformed nonfinal row")
            changed = True
            break
        rows.append(row)
        valid_lines.append(line)
    if metrics["step"] not in {row["step"] for row in rows}:
        valid_lines.append(json.dumps(metrics, allow_nan=False))
        changed = True
    if changed or not path.is_file():
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("\n".join(valid_lines) + "\n", encoding="utf-8")
        temporary.replace(path)


def _scope(args):
    """Resolve an explicit allowlist; never discover/read held-out archives."""
    split_path = _path(args.split)
    split_bytes = split_path.read_bytes()
    split = json.loads(split_bytes)
    groups = split["groups"]
    expected_counts = {"train": 40, "validation": 5, "test": 5}
    for name, count in expected_counts.items():
        if len(groups[name]) != count or len(set(groups[name])) != count:
            raise ValueError(f"Expected {count} distinct whole CADs in {name}")
    if any(set(groups[a]) & set(groups[b]) for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("CAD groups overlap")
    loads = split["train_loads"]
    if len(loads) != 3 or set(loads) != {"ver", "hor", "tor"} or split.get("heldout_load") != "dia":
        raise ValueError("This protocol fits ver/hor/tor and keeps dia locked")
    if set(split.get("checkpoint_validation_loads", loads)) != set(loads):
        raise ValueError("Validation loads must equal the fitted loads")
    data = _path(args.data)
    keys = {group: [f"{cad}/{load}" for cad in groups[group] for load in loads]
            for group in ("train", "validation")}
    files = {}
    identities = []
    for group in ("train", "validation"):
        for key in keys[group]:
            path = _path(split["case_paths"][key])
            if not path.is_relative_to(data) or path.suffix != ".npz" or not path.is_file():
                raise ValueError(f"Selected case is missing or outside --data: {key}: {path}")
            stat = path.stat()
            files[key] = path
            identities.append({"key": key, "group": group, "path": str(path),
                               "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    requested = {name: getattr(args, name) for name in
                 ("variant", "steps", "batch_size", "cases_per_batch", "eval_every",
                  "eval_batch_size", "learning_rate", "seed", "cloud_points", "geometry_dim", "device")}
    contract = {"version": 1, "requested": requested, "split_sha256": _sha(split_bytes),
                "selected_sources": identities,
                "source_identity_policy": "path/size/mtime; no multi-GB label/checkpoint SHA scans",
                "code_sha256": {name: _sha((ROOT / "geobc" / name).read_bytes()) for name in
                                ("conditional_train.py", "conditioned_field.py", "tcnn_fields.py")}}
    return split, split_path, keys, files, contract


@torch.no_grad()
def _predict(model, case, context, scaler, batch_size):
    encoded = np.empty(case.count, dtype=np.float32)
    for start in range(0, case.count, batch_size):
        stop = min(start + batch_size, case.count)
        encoded[start:stop] = model(case.xyz[start:stop], case.features[start:stop],
                                    context=context).reshape(-1).cpu().numpy()
    return scaler.decode(encoded)


@torch.no_grad()
def _evaluate_group(model, cases, contexts, cad_to_row, scaler, batch_size):
    per_case, truths, predictions = {}, [], []
    for case in cases:
        row = cad_to_row[case.key.split("/", 1)[0]]
        prediction = _predict(model, case, contexts[row:row + 1], scaler, batch_size)
        per_case[case.key] = stress_metrics(case.physical_stress, prediction)
        truths.append(case.physical_stress)
        predictions.append(prediction)
    worst = max(per_case, key=lambda key: per_case[key]["relative_l2"])
    return {"per_case": per_case, "pooled": stress_metrics(np.concatenate(truths), np.concatenate(predictions)),
            "case_mean_relative_l2": float(np.mean([m["relative_l2"] for m in per_case.values()])),
            "worst_case_key": worst, "worst_case_relative_l2": per_case[worst]["relative_l2"],
            "case_mean_hotspot_relative_l2": float(np.mean([m["hotspot_relative_l2"] for m in per_case.values()]))}


def _encoder_gradient(model, context):
    if model.encoder is None:
        return {"applicable": False}
    parameters = list(model.encoder.parameters())
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError("PointNet encoder has missing/nonfinite gradients through the TCNN decoder")
    norm = sum(float(p.grad.detach().abs().sum()) for p in parameters)
    if not math.isfinite(norm) or norm <= 0 or context.grad is None or not torch.isfinite(context.grad).all():
        raise FloatingPointError("The global geometry context did not receive finite, nonzero gradients")
    return {"applicable": True, "encoder_gradient_l1": norm,
            "context_gradient_l1_per_cad": context.grad.detach().abs().sum(dim=-1).cpu().tolist(),
            "checked_after_backward_before_optimizer_step": True}


def run(args):
    began = time.monotonic()
    if min(args.steps, args.batch_size, args.cases_per_batch, args.eval_every,
           args.eval_batch_size, args.cloud_points) < 1 or args.learning_rate <= 0:
        raise ValueError("Steps, sizes, intervals and learning rate must be positive")
    if args.batch_size % args.cases_per_batch:
        raise ValueError("batch-size must be divisible by cases-per-batch for equal per-case sampling")
    if not 1 <= args.cases_per_batch <= 4:
        raise ValueError("Use 1--4 cases per batch to limit repeated geometry encoding")
    split, split_path, keys, files, contract = _scope(args)
    output = _path(args.output)
    resume = (output / "last.pt" if str(args.resume) == "last" else _path(args.resume)) if args.resume else None
    if resume is not None and resume != (output / "last.pt").resolve():
        raise ValueError("Resume must use this output directory's last.pt, preserving historical best selection")
    # Completed reuse needs only small metadata/file identity checks, no model or labels.
    if (output / "summary.json").is_file():
        summary = _json(output / "summary.json")
        if summary.get("status") == "completed":
            if summary.get("config", {}).get("contract") != contract:
                raise ValueError("Completed run differs from the requested strict configuration/data contract")
            for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
                if not (output / filename).is_file():
                    raise ValueError(f"Completed run lacks {filename}")
            if summary["best_metrics"] != _json(output / "metrics_best.json") or summary["final_metrics"] != _json(output / "metrics_latest.json"):
                raise ValueError("Completed summary and saved metrics disagree")
            print(json.dumps({"status": "completed", "reused": True, "output": str(output),
                              "best_step": summary["best_step"]}), flush=True)
            return summary
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError("Nonempty output requires explicit --resume; nothing will be silently overwritten")
    if resume is not None and not resume.is_file():
        raise FileNotFoundError(resume)
    output.mkdir(parents=True, exist_ok=True)
    if resume is not None and _json(output / "config.json").get("contract") != contract:
        raise ValueError("Resume contract differs; data, model, budget, sampling and LR must be unchanged")
    status = {"phase": "loading", "step": 0, "total_steps": args.steps,
              "started_at_utc": utc_now(), "variant": args.variant}
    prior_elapsed, start_step = 0.0, 0

    def elapsed():
        return prior_elapsed + time.monotonic() - began

    def update(phase, **extra):
        status.update(extra, phase=phase, elapsed_seconds=elapsed(), updated_at_utc=utc_now())
        progress = status["step"] - start_step
        eta = ((time.monotonic() - began) / progress * (args.steps - status["step"])) if progress > 0 else None
        status["eta_seconds"] = eta
        status["estimated_completion_utc"] = (datetime.now(timezone.utc) + timedelta(seconds=eta)).isoformat() if eta else None
        write_json(output / "status.json", status)

    update("loading")
    try:
        device = torch.device(args.device)
        if device.type != "cuda":
            raise ValueError("The official TCNN decoder requires CUDA; CPU is for syntax/interface checks only")
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Only these explicit train/validation keys are opened. No glob or test/dia archive access.
        train_cases, scaler = load_cases([files[k] for k in keys["train"]], device)
        validation_cases, validation_scaler = load_cases([files[k] for k in keys["validation"]], device, scaler=scaler)
        if validation_scaler != scaler:
            raise AssertionError("Validation unexpectedly changed the training scaler")
        if [c.key for c in train_cases] != keys["train"] or [c.key for c in validation_cases] != keys["validation"]:
            raise ValueError("Prepared archive keys disagree with the split allowlist")
        all_cases = train_cases + validation_cases
        cad_keys = split["groups"]["train"] + split["groups"]["validation"]
        cad_to_row = {cad: i for i, cad in enumerate(cad_keys)}
        clouds = [load_geometry_cloud(files[f"{cad}/ver"], cloud_points=args.cloud_points,
                                      seed=args.seed, cad_key=cad) for cad in cad_keys]
        cloud_array = np.stack([c.points for c in clouds])
        mask_array = np.stack([c.mask for c in clouds])
        geometry_clouds = torch.as_tensor(cloud_array, device=device)
        geometry_masks = torch.as_tensor(mask_array, device=device)
        geometry_records = {cad: {**cloud.record(), "coordinates_sha256": _sha(cloud.points.tobytes()),
                                 "mask_sha256": _sha(cloud.mask.tobytes()),
                                 "node_indices": cloud.node_indices.tolist()}
                            for cad, cloud in zip(cad_keys, clouds)}
        model_config = ConditionedFieldConfig(
            geometry_mode="shape" if args.variant == "geometry_pointnet" else "none",
            descriptor_dim=args.geometry_dim, cloud_points=args.cloud_points,
            seed=args.seed, out_channels=1)
        model = ConditionedField(model_config, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, fused=True)
        case_rng = np.random.default_rng(args.seed + 23)
        node_rng = torch.Generator(device=device).manual_seed(args.seed + 17)
        config = {
            "contract": contract, "method": "GeoBC geometry-conditioned field candidate baseline",
            "variant": args.variant, "split_source": str(split_path), "dataset": split.get("dataset"),
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "scope": "40 training CADs / 5 validation CADs; ver,hor,tor only; test CADs and dia locked",
            "groups": split["groups"], "train_loads": split["train_loads"], "heldout_load": "dia",
            "cases": {group: [{"sample_key": c.key, "source": str(c.source), "nodes": c.count}
                              for c in cases] for group, cases in (("train", train_cases), ("validation", validation_cases))},
            "scaler": scaler.record(), "scaler_fit_scope": "train 40 CAD x ver/hor/tor, all original labels; validation uses fixed scaler",
            "model": model.source_config, "total_steps": args.steps,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "decoder_parameter_count": sum(p.numel() for p in model.decoder.parameters()),
            "encoder_parameter_count": sum(p.numel() for p in model.encoder.parameters()) if model.encoder is not None else 0,
            "sampling": "uniform distinct cases per batch; equal nodes per selected case, with replacement",
            "sampling_rng": {"case": "independent NumPy default_rng(seed+23)", "node": "independent torch.Generator(device), seed+17"},
            "batch_size": args.batch_size, "cases_per_batch": args.cases_per_batch,
            "geometry": {"cloud_file": "geometry_clouds.npz", "metadata_file": "cloud_metadata.json",
                         "canonical_source": "one ver prepared geometry per train/validation CAD, reused for all loads",
                         "coordinate_sha256": _sha(cloud_array.tobytes()), "mask_sha256": _sha(mask_array.tobytes()),
                         "target_independent": True, "cad_key_is_model_input": False},
            "training_loss": "case-balanced sampled mean squared error in fixed train-scaled centered asinh stress",
            "evaluation": "all original nodes; physical signed stress, no clipping; cloud context encoded once per CAD per evaluation",
            "checkpoint_selection": "minimum validation case_mean_relative_l2; train/test/dia do not select",
            "precision": "PointNet float32/high TF32; native TCNN FP16 compute, FP32 master, float32 returned output",
            "torch_version": torch.__version__, "started_at_utc": utc_now(),
            "claim": "candidate baseline, not a new algorithm or an already established generalization result"}
        best_score, best_step, best_metrics, gradient_check = float("inf"), None, None, None
        if resume is not None:
            stored_config = _json(output / "config.json")
            if any(stored_config[k] != config[k] for k in ("contract", "scaler", "model", "geometry", "cases", "precision")):
                raise ValueError("Resume scaler/model/geometry/case identities differ")
            saved = torch.load(resume, map_location=device, weights_only=False)
            if saved["config"] != stored_config:
                raise ValueError("last.pt and config.json disagree")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            start_step = int(saved["step"])
            if not 0 <= start_step <= args.steps:
                raise ValueError("Resume step is outside the requested budget")
            case_rng.bit_generator.state = saved["case_rng_state"]
            node_rng.set_state(saved["node_rng_state"].cpu())
            torch.set_rng_state(saved["torch_rng_state"].cpu())
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng_state"]])
            random.setstate(saved["python_rng_state"])
            np.random.set_state(saved["numpy_rng_state"])
            # Atomic last.pt is authoritative; JSON files can lag a completed
            # checkpoint write if the process stopped between file replacements.
            best_metrics = saved["best_metrics"]
            best_step, best_score = best_metrics["step"], best_metrics["selection_score"]
            if saved["best_step"] != best_step or saved["best_score"] != best_score:
                raise ValueError("Historical best selection inside last.pt is inconsistent")
            json_best = _json(output / "metrics_best.json") if (output / "metrics_best.json").is_file() else None
            if best_step == start_step:
                if json_best != best_metrics or not (output / "best.pt").is_file():
                    _save_checkpoint(saved, output / "best.pt")
            elif json_best != best_metrics or not (output / "best.pt").is_file():
                raise ValueError("The earlier historical best checkpoint/metrics are missing or inconsistent")
            write_json(output / "metrics_best.json", best_metrics)
            write_json(output / "metrics_latest.json", saved["metrics"])
            _restore_history_row(output, saved["metrics"])
            gradient_check = saved.get("encoder_gradient_check")
            prior_elapsed = float(saved["elapsed_seconds"])
            config = stored_config
        else:
            write_json(output / "config.json", config)
            write_json(output / "cloud_metadata.json", {"CADs": geometry_records, "label_members_read": [],
                                                      "policy": "geometry helper reads xyz/faces/key only"})
            np.savez(output / "geometry_clouds.npz", points=cloud_array, mask=mask_array,
                     node_indices=np.stack([c.node_indices for c in clouds]), cad_keys=np.asarray(cad_keys))

        def checkpoint(path, step, metrics):
            _save_checkpoint({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                        "step": step, "metrics": metrics, "best_step": best_step, "best_score": best_score,
                        "best_metrics": best_metrics,
                        "elapsed_seconds": elapsed(), "encoder_gradient_check": gradient_check,
                        "case_rng_state": case_rng.bit_generator.state, "node_rng_state": node_rng.get_state(),
                        "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all(),
                        "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state()}, path)

        @torch.no_grad()
        def evaluate(step, loss):
            nonlocal best_score, best_step, best_metrics
            eval_began = time.monotonic()
            model.eval()
            update("evaluating", step=step)
            contexts = model.encode_geometry(geometry_clouds, geometry_masks)
            train = _evaluate_group(model, train_cases, contexts, cad_to_row, scaler, args.eval_batch_size)
            validation = _evaluate_group(model, validation_cases, contexts, cad_to_row, scaler, args.eval_batch_size)
            score = validation["case_mean_relative_l2"]
            metrics = {"step": step, "elapsed_seconds": elapsed(), "training_loss": loss,
                       "selection_score": score, "train": train, "validation": validation,
                       "encoder_gradient_check": gradient_check, "eval_seconds": time.monotonic() - eval_began}
            improved = score < best_score
            if improved:
                best_score, best_step = score, step
                best_metrics = metrics
            # Commit a self-contained recovery point before derived files.
            checkpoint(output / "last.pt", step, metrics)
            if improved:
                checkpoint(output / "best.pt", step, metrics)
                write_json(output / "metrics_best.json", metrics)
            write_json(output / "metrics_latest.json", metrics)
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            update("training", step=step, best_step=best_step, best_score=best_score,
                   latest_train_case_mean_l2=train["case_mean_relative_l2"], latest_validation_case_mean_l2=score)
            print(f"{args.variant} {step}/{args.steps} train_mean={100*train['case_mean_relative_l2']:.4f}% val_mean={100*score:.4f}% best={best_step}", flush=True)
            return metrics

        update("training", step=start_step, resume_step=start_step)
        last = saved["metrics"] if resume is not None else evaluate(0, None)
        sampling_signature = hashlib.sha256()
        nodes_per_case = args.batch_size // args.cases_per_batch
        for step in range(start_step + 1, args.steps + 1):
            model.train()
            selected = case_rng.choice(len(train_cases), args.cases_per_batch, replace=False)
            batch_cases = [train_cases[int(i)] for i in selected]
            unique_cads = list(dict.fromkeys(c.key.split("/", 1)[0] for c in batch_cases))
            global_rows = torch.tensor([cad_to_row[cad] for cad in unique_cads], device=device)
            context = model.encode_geometry(geometry_clouds[global_rows], geometry_masks[global_rows])
            check_gradient = model.encoder is not None and gradient_check is None
            if check_gradient:
                context.retain_grad()
            local_rows = {cad: i for i, cad in enumerate(unique_cads)}
            xyz, features, target, indices = [], [], [], []
            sampling_signature.update(selected.astype(np.int64).tobytes())
            for case in batch_cases:
                node_ids = torch.randint(case.count, (nodes_per_case,), generator=node_rng, device=device)
                xyz.append(case.xyz[node_ids]); features.append(case.features[node_ids]); target.append(case.target[node_ids])
                indices.append(torch.full((nodes_per_case,), local_rows[case.key.split("/", 1)[0]], dtype=torch.long, device=device))
                # Small first-batch witness, not an expensive full sampling audit.
                if step == start_step + 1:
                    sampling_signature.update(node_ids[:32].cpu().numpy().tobytes())
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.cat(xyz), torch.cat(features), cad_index=torch.cat(indices), context=context).reshape(-1)
            loss = (prediction - torch.cat(target)).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite encoded training loss at step {step}")
            loss.backward()
            if check_gradient:
                gradient_check = _encoder_gradient(model, context)
                gradient_check["step"] = step
                write_json(output / "encoder_gradient_check.json", gradient_check)
            optimizer.step()
            if step == start_step + 1:
                write_json(output / ("sampling_first_batch.json" if start_step == 0 else "sampling_resume_batch.json"),
                           {"step": step, "case_keys": [c.key for c in batch_cases],
                            "nodes_per_case": nodes_per_case, "witness_sha256": sampling_signature.hexdigest(),
                            "same_seed_variants_use_identical_case_and_node_generators": True})
            if step % 100 == 0:
                update("training", step=step, training_loss=float(loss.detach()), best_step=best_step, best_score=best_score)
            if step % args.eval_every == 0 or step == args.steps:
                last = evaluate(step, float(loss.detach()))
        # Export at most four train/validation cases, once, from the chosen best.
        update("exporting_representatives", step=args.steps, best_step=best_step)
        chosen = _json(output / "metrics_best.json")
        saved_best = torch.load(output / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved_best["model"])
        model.eval()
        by_key = {c.key: c for c in all_cases}
        representatives = []
        for group in ("train", "validation"):
            ordered = sorted(chosen[group]["per_case"], key=lambda key: chosen[group]["per_case"][key]["relative_l2"])
            representatives.extend((group, key) for key in dict.fromkeys((ordered[0], ordered[-1])))
        folder = output / "predictions_best"
        folder.mkdir(exist_ok=True)
        index = {}
        with torch.no_grad():
            contexts = model.encode_geometry(geometry_clouds, geometry_masks)
            for i, (group, key) in enumerate(representatives[:4]):
                case = by_key[key]
                row = cad_to_row[key.split("/", 1)[0]]
                prediction = _predict(model, case, contexts[row:row + 1], scaler, args.eval_batch_size)
                metrics = stress_metrics(case.physical_stress, prediction)
                if not np.isclose(metrics["relative_l2"], chosen[group]["per_case"][key]["relative_l2"], rtol=1e-5, atol=1e-8):
                    raise ValueError("Representative prediction differs from saved best metrics")
                filename = f"case_{i:03d}.npz"
                np.savez(folder / filename, xyz=case.xyz.cpu().numpy(), stress_true=case.physical_stress,
                         stress_pred=prediction, sample_key=np.asarray(key), step=np.asarray(best_step))
                index[key] = {"file": filename, "source": str(case.source), "group": group, "nodes": case.count}
        write_json(folder / "index.json", {"step": best_step, "cases": index,
                                           "policy": "at most four train/validation cases, no test or dia"})
        summary = {"status": "completed", "step": args.steps, "best_step": best_step, "best_score": best_score,
                   "elapsed_seconds": elapsed(), "best_metrics": chosen, "final_metrics": last,
                   "config": config, "encoder_gradient_check": gradient_check,
                   "representative_predictions": str(folder / "index.json"), "completed_at_utc": utc_now()}
        write_json(output / "summary.json", summary)
        update("completed", step=args.steps, best_step=best_step, best_score=best_score)
        return summary
    except Exception as exc:
        update("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, default=Path("notes/geobc_split.json"))
    p.add_argument("--data", type=Path, default=Path("data/prepared_geobc_multiload"))
    p.add_argument("--variant", choices=("geometry_none", "geometry_pointnet"), required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=51200)
    p.add_argument("--cases-per-batch", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-batch-size", type=int, default=65536)
    p.add_argument("--learning-rate", type=float, default=.003)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--cloud-points", type=int, default=1024)
    p.add_argument("--geometry-dim", type=int, choices=(32, 64), default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", nargs="?", const="last", default=None,
                   help="Resume this output's last.pt; provide no value or its explicit path")
    return p


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
