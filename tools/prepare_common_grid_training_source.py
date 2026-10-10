"""Freeze the matched macro pair after genuine resource/gradient measurement."""
import ast
import json
from geobc import available_cad_pilot_data as d


def run():
    probe=d.read('notes/geobc_common_grid_probe_plan.json')
    accepted=d.read('notes/geobc_common_probe_interpretation_root_review.json')
    assert accepted['passed'] and accepted['actual_resource_and_active_gradient_qualified']
    parent=d.read('notes/geobc_fno_local_condition_pair_plan.json')
    import random
    order=[]; rng=random.Random(2026)
    for _ in range(40):
        keys=list(parent['scientific']['TRAIN_keys']);rng.shuffle(keys);order.extend(keys)
    science={**{k:v for k,v in probe['scientific'].items() if k not in ('role','TRAIN_case','optimizer_updates','budget','followup','predecessors')},
        'role':'Fixed4800 macro pair; same information, query head, exposure and nominal/active capacity within 0.11%',
        'TRAIN_keys':parent['scientific']['TRAIN_keys'],'case_order_sha256':d.digest(order),
        'fixed_updates_each':4800,'passes_each':40,
        'optimizer':{'name':'Adam','lr':.001,'weight_decay':0,'fused':True,'scheduler':'constant','clip':None},
        'checkpoint_updates':list(range(120,4801,120)),
        'endpoints':[4800],'early_stopping':False,
        'readout':'Full120 known TRAIN U(mm)/signed S(MPa), all original nodes, fixed endpoint only',
        'readout_seed':'2026 + original TRAIN manifest index, same formula for both arms',
        'budget':{'training_seconds_each':18000,'TRAIN_readout_seconds_each':1200,'whole_seconds':42000,'GPU_GiB':24},
        'probe_acceptance':'notes/geobc_common_probe_interpretation_root_review.json',
        'probe_plan':'notes/geobc_common_grid_probe_plan.json',
        'measurement':{'FNO_forward_backward_seconds':1.9774560819999998,'PP_forward_backward_seconds':1.775090809,
                       'FNO_reserved_bytes':864026624,'PP_reserved_bytes':10003415040},
        'expected_inactive_PP_parameters':['operator.backbone.embedding.weight','operator.backbone.embedding.bias'],
        'expected_inactive_PP_count':1024,
        'nominal_parameters':{'FNO9':4526858,'PP256':4523268},
        'active_parameters':{'FNO9':4526858,'PP256':4522244},
        'interpretation':'TRAIN gates do not determine final solver; separately evaluate both fixed endpoints on all60 already-known development cases, no new sealed DEV',
        'followup':'No new hyperparameter selection using CAD20; compare macro architectures and cost. No automatic replay or budget reset.'}
    pins=dict(probe['source_pins'])
    for name in ['tools/train_geobc_common_grid.py','tools/run_geobc_common_training_guard.py']:
        ast.parse(d.path(name).read_text(encoding='utf-8-sig'));pins[name]=d.sha(name)
    plan={'scientific':science,'scientific_digest':d.digest(science),'source_pins':pins,
          'output':'runs/geobc_exploration/common_grid_matched4800_v1',
          'root_review':'notes/geobc_common_grid_train_root_review.json'}
    assert not d.path('notes/geobc_common_grid_train_plan.json').exists()
    d.write('notes/geobc_common_grid_train_plan.json',plan)
    d.write(plan['root_review'],{'passed':True,'actual_training_released':True,'reviewer':'root',
        'scientific_digest':plan['scientific_digest'],'source_pins':pins,
        'source_review':['Both complete model information and query chain already actually exercised in resource probe',
        'Optional official embedding unused with condition=None; inactive gradient names/count explicitly recorded, no attention math changed',
        'Fresh same seed/shared bytes; full 4800 balanced schedule; highest forward/backward; all finite optimized gradients',
        'Nonblocking common GPU lock; exact child/guard exits and immutable probe evidence before CUDA',
        'Dual epoch/monotonic whole42000/per-arm18000/readout1200 deadlines; all costs preserved, no auto-replay',
        '120-step model/Adam/RNG/exposure checkpoint; all original-node known TRAIN physical readout; no DEV/raw GT/scaler fit'],
        'probe_acceptance_identity':d.identity(science['probe_acceptance'],hashed=True),'actual_updates_at_release':0})
    print(json.dumps({'status':'source_frozen_root_released','science':plan['scientific_digest']}))


if __name__=='__main__':run()
