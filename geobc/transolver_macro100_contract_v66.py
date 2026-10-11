"""Same originalTS/continuous decoder, fresh fixed100CAD short capability test."""
from . import available_cad_pilot_data as d
from .transolver_bf16_contract_v65 import inputs as precision_inputs,terminal,complete_physical,check_identity
from .macro_training import balanced_schedule

TS='notes/geobc_transolver_bf16_v65_plan.json'
HASH='notes/geobc_global_hash_physical_recovery_v18_plan.json'
FNO='notes/geobc_macro100_FNO9_physical5_v12_plan.json'
SOURCES=('geobc/transolver_macro100_contract_v66.py','tools/train_geobc_transolver_macro100_v66.py',
    'tools/run_geobc_transolver_macro100_guard_v66.py','tools/prepare_geobc_transolver_macro100_v66.py')


def inputs():
    up=precision_inputs();p,a,e,b=terminal(TS)
    report=d.read('results/geobc_transolver_bf16_v65/manifest.json')
    assert report['status']=='completed' and report['SCI']==p['scientific_digest'] and report['terminal_binding']==b
    assert a['training_complete'] and a['optimizer_step_completions']==3000 and not a['possible_partial_optimizer_update']
    family='TS_ORIGINAL_BF16_GLOBAL_HASH';row=a['models'][family]
    assert row['case_exposure_counts']=={k:1000 for k in p['scientific']['selected_TRAIN_keys']}
    metric=row['endpoints']['1000'];complete_physical(metric,p['scientific']['selected_TRAIN_keys'])
    # Ability/cost screen only; beating tiny Hash does not establish newCAD performance.
    assert metric['case_mean_u_relative_l2']<=.15 and metric['case_mean_stress_relative_l2']<=.20 and metric['case_mean_stress_hotspot_relative_l2']<=.30
    hp,ha,he,hb=terminal(HASH);fp,fa,fe,fb=terminal(FNO)
    keys=hp['scientific']['TRAIN_keys'];assert len(keys)==300 and keys==p['scientific']['parent_TRAIN_keys']
    for name in ('dataset_identity','scaler','optimizer','loss','TRAIN_keys'):
        assert hp['scientific'][name]==fp['scientific'][name]
    for name in ('dataset_identity','scaler','optimizer','loss'):
        assert hp['scientific'][name]==p['scientific'][name]
    assert ha['training_complete'] and fa['training_complete'] and ha['step']==fa['step']==1500
    for aa in (ha,fa):
        assert aa['case_exposure_counts']=={k:5 for k in keys}
        complete_physical(aa['endpoints']['1500']['full_TRAIN'],keys)
    up.update(TS=p,TS_summary=a,TS_external=e,TS_binding=b,Hash=hp,Hash_summary=ha,Hash_binding=hb,
        FNO=fp,FNO_summary=fa,FNO_binding=fb,keys=keys)
    return up


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);up=inputs();old=up['TS']['scientific']
    assert p['kind']=='GeoBC_originalTS_BF16_continuous_decoder_TRAIN100_physical5_v66'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest']==r['scientific_digest']==d.digest(s) and p['source_pins']==r['source_pins']
    assert set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==h for k,h in p['source_pins'].items())
    for name in ('TS','Hash','FNO'):
        assert r['upstream_terminal_bindings'][name]==up[name+'_binding']
        assert all(p['source_pins'].get(k)==h for k,h in up[name]['source_pins'].items())
    for name in ('families','seed','train_CAD_count','dataset_identity','scaler','optimizer','loss','GPU_lock','memory_limit_bytes','decoder_chunk_size',
        'parent_TRAIN_keys','representative_field','visualization_sample_nodes_max','expected_reference_initialization','expected_FP32_TS_initialization','official_TS_kwargs','precision'):
        assert s[name]==old[name]
    assert s['TRAIN_keys']==up['keys'] and s['selected_TRAIN_keys']==up['keys'] and s['fresh_models_per_family']==1
    assert s['fresh_initialization'] and s['parent_checkpoint_reads_allowed']==0
    assert s['updates_per_field']==5 and s['total_updates']==s['total_updates_per_family']==1500 and s['case_pass_endpoints']==[0,1,5]
    assert s['full_order_digest']==d.digest(balanced_schedule(up['keys'],5,2026))
    assert s['memory_limit_bytes']==30500000000 and s['cache_setup_budget_seconds']==600 and s['training_budget_seconds']==900 and s['whole_budget_seconds']==1800
    assert not s['automatic_continuation'] and s['DEV_FINAL_reads_allowed']==0 and not s['FINAL_reuse_for_selection']
    assert s['reference_TRAIN5']=={name:up[name+'_summary']['endpoints']['1500']['full_TRAIN']for name in ('Hash','FNO')}
    m=d.read(check_identity(s['dataset_identity']));assert m['train_keys']['100']==up['keys']
    witnesses=[max([k for k in up['keys']if k.endswith('/'+load)],key=lambda k:m['records'][k]['num_nodes'])for load in ('ver','hor','tor')]
    assert s['qualification_TRAIN_keys']==witnesses and all(m['case_roles'][k]=='TRAIN'for k in up['keys'])
    assert s['tiny_TRAIN_eligibility']==dict(U_at_most=.15,S_at_most=.20,hot_at_most=.30,scope='fitting ability and completed cost only, not newCAD selection')
    assert p['output']=='runs/geobc_exploration/Transolver_BF16_continuous_decoder_TRAIN100_physical5_v66'
    return p,r,m,up
