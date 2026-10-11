"""Matched-exposure sharing of three fixed TRAIN fields by decoder package."""
import random
from . import available_cad_pilot_data as d
from .single_decoder_pair_contract_v60 import terminal,complete_physical,check_identity,MODEL_SOURCES

SOURCES=('geobc/shared_decoder_pair_contract_v61.py','tools/train_geobc_shared_decoder_v61.py',
    'tools/run_geobc_shared_decoder_guard_v61.py','tools/prepare_geobc_shared_decoder_v61.py')


def order(keys):
    rng=random.Random(2026);rows=[]
    for _ in range(1000):
        part=list(keys);rng.shuffle(part);rows.extend(part)
    return rows


def inputs():
    p,a,e,b=terminal('notes/geobc_single_decoder_v60_plan.json');keys=p['scientific']['selected_TRAIN_keys']
    report=d.read('results/geobc_single_decoder_v60/manifest.json')
    assert report['status']=='completed' and report['SCI']==p['scientific_digest'] and report['terminal_binding']==b
    assert a['training_complete'] and a['optimizer_step_attempts']==a['optimizer_step_completions']==6000
    for endpoints in a['aggregate_endpoints'].values():
        for metric in endpoints.values():complete_physical(metric,keys)
    return dict(plan=p,summary=a,external=e,binding=b,keys=keys)


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);up=inputs();ps=up['plan']['scientific']
    assert p['kind']=='GeoBC_fresh_shared_three_decoder_pair_v61'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest']==r['scientific_digest']==d.digest(s) and p['source_pins']==r['source_pins']
    assert set(SOURCES+MODEL_SOURCES)<=set(p['source_pins']) and all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert all(p['source_pins'].get(k)==h for k,h in up['plan']['source_pins'].items())
    assert r['upstream_terminal_binding']==up['binding']
    for k in ('seed','train_CAD_count','dataset_identity','scaler','optimizer','loss','GPU_lock','memory_limit_bytes',
        'decoder_chunk_size','parent_TRAIN_keys','selected_TRAIN_keys','families','parameter_counts','runtime_factory',
        'whole_budget_seconds','training_budget_seconds','cache_setup_budget_seconds','representative_field','visualization_sample_nodes_max'):
        assert s[k]==ps[k]
    assert s['fresh_models_per_family']==1 and s['updates_per_field']==1000 and s['total_updates_per_family']==3000 and s['total_updates']==6000
    assert s['case_pass_endpoints']==[0,100,1000] and s['fresh_initialization'] and s['parent_checkpoint_reads_allowed']==0
    assert s['full_order_digest']==d.digest(order(up['keys']))
    assert s['expected_actual_initialization']==up['summary']['model_initialization']
    assert s['independent_endpoint_reference']==up['summary']['aggregate_endpoints']
    assert s['DEV_FINAL_reads_allowed']==0 and not s['automatic_continuation'] and not s['FINAL_reuse_for_selection']
    assert p['output']=='runs/geobc_exploration/FNO_globalHash_shared_three_v61'
    m=d.read(check_identity(s['dataset_identity']))
    assert all(m['case_roles'][k]=='TRAIN' and k in m['train_keys']['100']for k in up['keys'])
    return p,r,m,up
