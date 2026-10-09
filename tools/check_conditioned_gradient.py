"""One real-data native-CUDA input-gradient check; not a training experiment."""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from geobc.conditioned_field import ConditionedField, ConditionedFieldConfig, load_geometry_cloud
from geobc.train import StressScaler, write_json, utc_now

def main():
    started = time.monotonic()
    torch.manual_seed(2026)
    device = torch.device('cuda')
    parent = json.loads((ROOT/'runs/slides_reproduction/hash50_full1000ep/config.json').read_text())
    scaler = StressScaler(**{k:parent['scaler'][k] for k in ('scale','center','spread')})
    clouds, masks, coordinates, features, targets = [], [], [], [], []
    keys = ['101_479/tor', '107_320/tor']
    for key in keys:
        path = ROOT/'data/prepared50_multifield'/(key.replace('/','_')+'.npz')
        cloud = load_geometry_cloud(path, cloud_points=1024, seed=2026)
        clouds.append(torch.as_tensor(cloud.points,device=device))
        masks.append(torch.as_tensor(cloud.mask,device=device))
        with np.load(path,allow_pickle=False) as z:
            coordinates.append(torch.as_tensor(z['xyz'][:512].copy(),device=device))
            features.append(torch.as_tensor(z['features'][:512].copy(),device=device))
            targets.append(torch.as_tensor(scaler.encode(z['stress'][:512]),device=device))
    model = ConditionedField(ConditionedFieldConfig(geometry_mode='shape',seed=2026),device)
    model.train()
    context = model.encode_geometry(torch.stack(clouds),torch.stack(masks))
    context.retain_grad()
    indices = torch.arange(2,device=device).repeat_interleave(512)
    prediction = model(torch.cat(coordinates),torch.cat(features),cad_index=indices,context=context).reshape(-1)
    loss = (prediction-torch.cat(targets)).square().mean()
    loss.backward()
    context_l1 = context.grad.abs().sum(dim=1)
    gradients = {name:p.grad for name,p in model.encoder.named_parameters()}
    finite = all(g is not None and torch.isfinite(g).all().item() for g in gradients.values())
    nonzero = all(g is not None and g.abs().sum().item()>0 for g in gradients.values())
    passed = finite and nonzero and torch.isfinite(context.grad).all().item() and (context_l1>0).all().item()
    record = {'status':'passed' if passed else 'failed','scope':'native TCNN -> PointNet CUDA backward interface check only',
              'real_training_cad_keys':keys,'final_test_labels_used':False,'training_updates':0,
              'context_gradient_l1_per_cad':context_l1.tolist(),'encoder_gradient_finite':finite,
              'all_encoder_parameter_tensors_have_nonzero_gradient':nonzero,
              'model':model.source_config,'elapsed_seconds':time.monotonic()-started,'completed_at_utc':utc_now()}
    write_json(ROOT/'notes/geometry_candidate_validation.json',record)
    print(json.dumps({k:record[k] for k in ('status','context_gradient_l1_per_cad','encoder_gradient_finite','elapsed_seconds')}),flush=True)
    if not passed: raise RuntimeError('Native TCNN geometry-conditioning gradient check failed')

if __name__=='__main__': main()
