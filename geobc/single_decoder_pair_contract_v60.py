"""Fresh paired local decoder-package diagnostic on three fixed TRAIN fields."""
from . import available_cad_pilot_data as d
from .single_field_learning_contract_v59 import terminal,check_identity,complete_physical

SOURCES=('geobc/single_decoder_pair_contract_v60.py','tools/train_geobc_single_decoder_v60.py',
    'tools/run_geobc_single_decoder_guard_v60.py','tools/prepare_geobc_single_decoder_v60.py')
MODEL_SOURCES=('geobc/operator_hash_metadata_v6.py','geobc/operator_hash_training.py',
    'geobc/operator_conditioned_hash.py','geobc/tcnn_fields.py','geobc/common_grid_solver.py',
    'geobc/fno.py','geobc/conditioned_field.py','geobc/physical_field_loss.py')


def inputs():
    p,a,e,b=terminal('notes/geobc_single_field_v59_plan.json');keys=p['scientific']['selected_TRAIN_keys']
    assert a['training_complete'] and a['optimizer_step_attempts']==a['optimizer_step_completions']==2700
    report=d.read('results/geobc_single_field_v59/manifest.json')
    assert report['status']=='completed' and report['SCI']==p['scientific_digest'] and report['terminal_binding']==b
    for metric in a['aggregate_endpoints'].values():complete_physical(metric,keys)
    return dict(plan=p,summary=a,external=e,binding=b,keys=keys)


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);up=inputs();ps=up['plan']['scientific']
    assert p['kind']=='GeoBC_fresh_FNO9_globalHash_three_single_field_decoder_pair_v60'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest']==r['scientific_digest']==d.digest(s)
    assert p['source_pins']==r['source_pins'] and set(SOURCES+MODEL_SOURCES)<=set(p['source_pins'])
    assert all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert all(p['source_pins'].get(k)==h for k,h in up['plan']['source_pins'].items())
    assert r['upstream_terminal_binding']==up['binding']
    for k in ('seed','train_CAD_count','dataset_identity','scaler','optimizer','loss','GPU_lock','memory_limit_bytes','decoder_chunk_size','parent_TRAIN_keys'):
        assert s[k]==ps[k]
    assert s['selected_TRAIN_keys']==up['keys'] and len(up['keys'])==3
    assert s['families']==['FNO9','FNO_GLOBAL_HASH'] and s['fresh_models_per_family']==3
    assert s['updates_per_field']==1000 and s['total_updates_per_family']==3000 and s['total_updates']==6000
    assert s['endpoints']==[0,100,1000] and s['fresh_initialization'] and s['parent_checkpoint_reads_allowed']==0
    assert s['parameter_counts']==dict(FNO9=4526858,FNO_GLOBAL_HASH=12245894)
    assert s['runtime_factory']=='geobc.operator_hash_metadata_v6.build_solver'
    assert s['whole_budget_seconds']==1200 and s['training_budget_seconds']==900 and s['cache_setup_budget_seconds']==180
    assert s['DEV_FINAL_reads_allowed']==0 and not s['automatic_continuation'] and not s['FINAL_reuse_for_selection']
    assert s['representative_field']=='229_200/ver' and s['visualization_sample_nodes_max']==20000
    assert p['output']=='runs/geobc_exploration/FNO_globalHash_fresh_three_single_v60'
    m=d.read(check_identity(s['dataset_identity']))
    assert all(m['case_roles'][k]=='TRAIN' and k in m['train_keys']['100']for k in up['keys'])
    return p,r,m,up
