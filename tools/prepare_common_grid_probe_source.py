"""Root source preparation only: immutable new plan, no arrays/models/CUDA."""
import ast
import json
from geobc import available_cad_pilot_data as d


def run():
    base = d.read('notes/geobc_fp32_query_ablation_plan.json')
    sources = ['geobc/common_grid_solver.py','tools/probe_geobc_common_grid.py','tools/run_geobc_common_probe_guard.py']
    pins = dict(base['source_pins'])
    for name in sources:
        ast.parse(d.path(name).read_text(encoding='utf-8-sig')); pins[name]=d.sha(name)
    parent = d.read('notes/geobc_fno_local_condition_pair_plan.json')
    science = {'question':'Does FNO still lead ++ with common grid information, query head and matched capacity?',
        'role':'One genuine full TRAIN-case gradient/resource feasibility probe; not solver accuracy or training',
        'arms':['FNO9','PP256'],'seed':2026,'parent_plan':parent['parent_plan'],
        'TRAIN_case':parent['scientific']['TRAIN_keys'][0],
        'TRAIN_keys_sha256':d.digest(parent['scientific']['TRAIN_keys']),
        'scaler':parent['scientific']['scaler'],'grid_resolution':64,'grid_tokens':262144,
        'input':'same occupancy+xyz3+BC/material25+PointNet32; PP reorder puts xyz first, no dropped voxels',
        'common_query':'native25+xyz3+PointNet32+latent16 -> 128 GELU -> 128 GELU -> 4',
        'FNO':{'width':9,'modes':12,'layers':4,'latent':16},
        'PP':{'layers':16,'width':256,'heads':8,'slices':32,'ratio':1,'latent':16,'unchanged_official_Gumbel':True},
        'precision':'highest float32 for complete forward AND checkpoint backward; TF32 disabled',
        'optimizer_updates':0,'cached_TRAIN_encoded_targets_only':True,
        'budget':{'whole_seconds':900,'GPU_GiB':24},
        'GPU_lock':'runs/geobc_exploration/.coordinator.lock',
        'predecessors':['runs/geobc_exploration/continuous_query_qualification_v1',
                        'runs/geobc_exploration/FP32_query_ablation_v1'],
        'followup':'Only resource PASS allows a new separately designed fixed4800 training protocol based on actual cost',
        'DEV_raw_GT_downloads_scalerfit':0}
    plan = {'scientific':science,'scientific_digest':d.digest(science),'source_pins':pins,
            'root_review':'notes/geobc_common_grid_probe_root_review.json',
            'output':'runs/geobc_exploration/common_grid_fullcase_probe_v1'}
    filename=d.path('notes/geobc_common_grid_probe_plan.json')
    assert not filename.exists(), 'New plan already frozen'
    d.write(filename,plan)
    review = {'passed':True,'actual_probe_released':True,'actual_training_released':False,
        'scientific_digest':plan['scientific_digest'],'source_pins':pins,
        'reviewer':'root','review_basis':['Complete new model/probe/guard source inspected by root',
         'All grid voxels and full original nodes, common init/head/information, explicit channel reorder',
         'Whole highest-FP32 graph/recompute; unchanged official attention/Gumbel',
         'Same existing GPU lock and exact terminal predecessor gates, dual 900s budget, one attempt',
         'Cached original TRAIN encoded target only, zero optimizer steps/new GT/DEV/scaler fits'],
        'AST_passed':sources,'source_preparation_only':True,'actual_probe_at_review':0}
    d.write(plan['root_review'],review)
    print(json.dumps({'status':'source_frozen_root_released','science':plan['scientific_digest']}))


if __name__=='__main__':run()
