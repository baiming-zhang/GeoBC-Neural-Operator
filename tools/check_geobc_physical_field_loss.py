"""One CPU synthetic objective check against the existing physical benchmark.

This does not qualify actual wholeN training, resources or accuracy improvement.
"""
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone


def main():
    began = time.perf_counter()
    import numpy as np
    import torch
    from geobc.benchmark import FieldScaler, field_metrics
    from geobc.physical_field_loss import (PhysicalScale, norms_from_complete_physical_target,
                                         physical_relative_squared_loss, decode_physical)
    root = Path(__file__).resolve().parents[1]
    filename = root/'notes/geobc_physical_loss_synthetic_check.json'
    assert not filename.exists(), 'Do not overwrite a completed synthetic receipt'
    assert not torch.cuda.is_initialized()
    metadata = json.loads((root/'notes/geobc_common_grid_train_plan.json').read_text(encoding='utf-8-sig'))
    mapping = metadata['scientific']['scaler']
    scale = PhysicalScale.from_mapping(mapping)
    scaler = FieldScaler(**mapping)
    # Explicit synthetic mm / signed MPa fields, not DeepJEB numerical truth.
    truth = np.array([[.02,-.01,.04,-160.],[-.03,.008,.1,15.], [.01,.003,-.02,230.],
                      [.08,-.02,.06,-30.],[.004,.01,.003,2.],[-.02,-.005,.09,-81.],
                      [.03,.002,-.01,44.]], dtype=np.float64)
    physical_target = torch.tensor(truth, dtype=torch.float64)
    prediction = scaler.encode(truth).astype(np.float64)
    prediction += np.array([[.13,-.07,.03,.11],[-.01,.02,-.09,-.06],[.04,.02,.03,.12],
                            [-.03,.04,.06,-.08],[.07,-.01,.03,.16],[.03,.05,-.02,-.13],
                            [-.09,.03,.02,.04]])
    encoded = torch.tensor(prediction, dtype=torch.float64, requires_grad=True)
    norms = norms_from_complete_physical_target(physical_target)
    loss = physical_relative_squared_loss(encoded, physical_target, scale, norms)
    physical = scaler.decode(prediction)
    metrics = field_metrics(truth, physical, [0,1,2,3])
    expected = .5*(metrics['displacement']['relative_l2']**2 + metrics['stress']['relative_l2']**2)
    metric_error = abs(float(loss.detach())-expected)
    decode_error = float(np.max(np.abs(decode_physical(encoded, scale).detach().numpy()-physical)))
    assert metric_error < 1e-12 and decode_error < 1e-10
    direct_gradient = torch.autograd.grad(loss, encoded)[0]
    chunked = sum(physical_relative_squared_loss(encoded[a:b], physical_target[a:b], scale, norms)
                  for a,b in [(0,3),(3,6),(6,7)])
    chunk_gradient = torch.autograd.grad(chunked, encoded)[0]
    chunk_loss_error = abs(float(chunked.detach())-expected)
    chunk_gradient_error = float((chunk_gradient-direct_gradient).abs().max())
    assert chunk_loss_error < 1e-12 and chunk_gradient_error < 1e-12
    checks = []
    for row,column in [(0,0),(1,1),(2,2),(0,3),(2,3),(5,3)]:
        epsilon = 1e-6
        plus,minus = encoded.detach().clone(),encoded.detach().clone()
        plus[row,column] += epsilon; minus[row,column] -= epsilon
        fd = (physical_relative_squared_loss(plus,physical_target,scale,norms)-
              physical_relative_squared_loss(minus,physical_target,scale,norms))/(2*epsilon)
        actual = direct_gradient[row,column]
        error = abs(float(fd-actual)) / max(abs(float(actual)), 1e-12)
        assert error < 1e-7
        checks.append({'row':row,'encoded_channel':column,'AD':float(actual),'FD':float(fd),
                       'relative_difference':error})
    zero_norm_rejections = 0
    for columns in (slice(0,3),slice(3,4)):
        zero = physical_target.clone(); zero[:,columns] = 0
        try: norms_from_complete_physical_target(zero)
        except ValueError: zero_norm_rejections += 1
    assert zero_norm_rejections == 2
    assert not torch.cuda.is_initialized()
    sources = ['geobc/physical_field_loss.py','tools/check_geobc_physical_field_loss.py',
               'geobc/benchmark.py','geobc/common_grid_solver.py']
    result = {'status':'completed_synthetic_only','observed_UTC':datetime.now(timezone.utc).isoformat(),
              'actual_seconds':time.perf_counter()-began,'source_sha256':{
                  name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
              'frozen_scaler':mapping,'synthetic_case_nodes':7,
              'objective':'.5*(physical displacement relative L2 squared + original signed stress relative L2 squared)',
              'metric_reference':'benchmark.FieldScaler.decode and field_metrics, unmodified',
              'reporting_primary_unchanged':'.5*(unsquared mean relative_U_L2 + unsquared mean relative_S_L2)',
              'metric_loss_absolute_error':metric_error,'decode_max_absolute_error':decode_error,
              'unequal_chunk_loss_absolute_error':chunk_loss_error,
              'unequal_chunk_gradient_max_absolute_error':chunk_gradient_error,
              'central_finite_difference_checks':checks,'zero_norm_cases_rejected':zero_norm_rejections,
              'physical_truth_source':'explicit synthetic constants only, no DeepJEB labels',
              'actual_model_fullN_gradient_gate_passed':False,'actual_GPU_resource_gate_passed':False,
              'actual_training_activation_released':False,'actual_DeepJEB_array_GT_PT_reads':0,
              'actual_model_construction_or_training':0,'CUDA_initialized':torch.cuda.is_initialized(),
              'limitations':['Synthetic checks do not measure solver accuracy or explain existing error',
                             'No existing model, loss, scaler, labels, scientific identity or dataset split changed',
                             'Future activation requires evidence from the fixed current pair and single-case diagnostic',
                             'Whole-case joint model gradients, FP64 objective cost and GPU memory still require real qualification']}
    filename.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({name:result[name] for name in ('status','actual_seconds','metric_loss_absolute_error',
          'unequal_chunk_gradient_max_absolute_error','zero_norm_cases_rejected','CUDA_initialized')}))


if __name__ == '__main__':
    main()
