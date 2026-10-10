"""Source/metadata only; genuine future training terminal required for release."""
import ast
import json
from geobc import available_cad_pilot_data as d


def run():
    parent=d.read('notes/geobc_common_grid_train_plan.json')
    original=d.read('notes/geobc_available_cad_pilot_plan.json')
    assert original['scientific_digest']==d.digest(original['scientific'])
    manifest=d.read('data/geobc_available_cad_pilot/manifest.json')
    keys=manifest['case_keys']
    assert len(keys)==len(set(keys))==60 and manifest['status']=='completed' and manifest['feasibility_pass']
    assert len({k.split('/')[0] for k in keys})==20
    assert manifest['scaler']==parent['scientific']['scaler']
    science={'question':'Does the FNO versus ++ cross-CAD gap survive common R64 information/native25 head/capacity/exposure?',
        'role':'Known development followup; no blind/unbiased/final-test or offnode FEM accuracy claim',
        'parent_plan':'notes/geobc_common_grid_train_plan.json','parent_scientific_digest':parent['scientific_digest'],
        'original_pilot_scientific_digest':original['scientific_digest'],
        'arms':['FNO9','PP256'],'step':4800,'case_keys':keys,'case_denominator':60,'prediction_denominator':120,
        'GPU_lock':'runs/geobc_exploration/.coordinator.lock','budget':{'whole_seconds':1800,'GPU_GiB':24},
        'input_manifest_identity':d.identity('data/geobc_available_cad_pilot/manifest.json',hashed=True),
        'GT_manifest_identity':d.identity('runs/geobc_exploration/available_CAD20_candidate9/shared_GT/manifest.json',hashed=True),
        'read_order':'All120 real finite arrays/Cartesian commit barrier before numerical label-cache reads; any runtime/budget/nonfinite interruption whole-negative and no extra GT readout/replay',
        'GT_cache_reads':60,'raw_CSV_reparses':0,'GT':{'units':['mm','mm','mm','MPa'],'signed_stress_unchanged':True,'all_original_nodes':True},
        'inference_seed':'2026 + fixed case_keys index, both arms same, official ++ Gumbel unchanged',
        'precision':'highest whole float32, TF32 disabled, same as matched training',
        'endpoint_binding':'Exact actually completed step4800 model source/science/exposure/byteSHA, no optimizer restoration',
        'primary':'0.5*(full60 mean U relativeL2+full60 mean signed S relativeL2); any missing/zero norm null/no subset ranking',
        'secondary':['per-load20','pairedCAD20 three-load mean U/S/hot','worst U/S/hot','physical-unit U/S full nodes','nominal/active capacity','TRAIN/readout wall/model infer cost/memory'],
        'old_baselines':'Original pilot full60 oldFNO and old++ scalar rows; common pair changes representation/precision and is not original slides replay',
        'training_updates_newGT_scales_sealedDEV':0}
    pins=dict(parent['source_pins'])
    for name in ['tools/evaluate_geobc_common_grid.py','tools/run_geobc_common_readout_guard.py']:
        ast.parse(d.path(name).read_text(encoding='utf-8-sig'));pins[name]=d.sha(name)
    plan={'scientific':science,'scientific_digest':d.digest(science),'source_pins':pins,
          'root_review':'notes/geobc_common_grid_readout_root_review.json',
          'output':'runs/geobc_exploration/common_grid_development60_v1'}
    assert not d.path('notes/geobc_common_grid_readout_plan.json').exists()
    d.write('notes/geobc_common_grid_readout_plan.json',plan)
    review={'passed':True,'reviewer':'root','SOURCE_ready':True,'actual_readout_released':False,
        'actual_GPU_executions':0,'actual_array_reads':0,'actual_GT_reads':0,
        'scientific_digest':plan['scientific_digest'],'source_pins':pins,
        'future_terminal_bindings':None,
        'review_basis':['Complete evaluator/guard source inspected: all common64 voxels/full original nodes/strict own4800/source/science/exposure binding',
          'Native and external genuinely terminal/inactive, exact live process identity absence, per-arm and whole budget qualification before lock/CUDA/model reads',
          'All120 immutable finite encoded+physical arrays re-opened/hash+shape before60 existing GT cache reads, no CSV reparse or frozenDEV',
          'Same fixed percase seeds/highest precision; nonfinite/OOM/budget/errors preserve denominator and cancel leader; no auto replay',
          'Whole1800s dual epoch/monotonic and24GiB common lock, no source-ready completion claim'],
        'release_condition':'Root binds actual final training summary/external identities and exact child/guard exits after both4800 budgetqualified; then mark actual_readout_released true without changing frozen science/source'}
    d.write(plan['root_review'],review)
    print(json.dumps({'status':'source_ready_waiting_genuine_training_terminal','science':plan['scientific_digest'],'actual_readout':0}))


if __name__=='__main__':run()
