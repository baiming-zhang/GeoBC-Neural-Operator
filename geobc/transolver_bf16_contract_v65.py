"""Independent precision branch after real FP32 cost-budget failure."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,complete_physical,check_identity,order

PARENT='notes/geobc_transolver_decoder_v64_plan.json'
SOURCES=('geobc/transolver_bf16_hash_v65.py','geobc/transolver_bf16_contract_v65.py',
    'tools/train_geobc_transolver_bf16_v65.py','tools/run_geobc_transolver_bf16_guard_v65.py',
    'tools/prepare_geobc_transolver_bf16_v65.py')


def fp32_budget_binding():
    p=d.read(PARENT);r=d.read(p['root_review']);o=d.path(p['output'])
    assert r['passed'] and r['actual_execution_released'] and p['scientific_digest']==r['scientific_digest']==d.digest(p['scientific'])
    assert p['source_pins']==r['source_pins'] and all(d.sha(k)==h for k,h in p['source_pins'].items())
    a=d.read(o/'summary.json');e=d.read(o/'external_terminal.json')
    assert a==d.read(o/'status.json') and a['status']=='failed' and a['closed'] and not a['active']
    assert a['error_type']=='TimeoutError' and a['error']=='Frozen earlier epoch/monotonic deadline'
    assert e['closed'] and not e['active'] and e['exit_code']!=0 and not e['within_external_budget']
    assert not e['external_timeout'] and e['guard_error_type'] is None
    assert e['actual_child_exited']
    assert a['execution']['scientific_digest']==e['scientific_digest']==p['scientific_digest'] and e['source_pins']==p['source_pins']
    assert a['execution']['run_process']==e['child_process'] and a['execution']['guard']['guard_process']==e['guard_process']
    assert all(d.exact_process_exited(e[k])for k in ('child_process','guard_process'))
    assert not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
    assert a['optimizer_step_attempts']==a['optimizer_step_completions']<3000
    assert max(a['elapsed_epoch_seconds']-a['cache_setup_epoch_seconds'],
        a['elapsed_monotonic_seconds']-a['cache_setup_monotonic_seconds'])>=900
    family='TS_ORIGINAL_GLOBAL_HASH';row=a['models'][family];keys=p['scientific']['selected_TRAIN_keys']
    assert sum(row['case_exposure_counts'].values())==row['optimizer_step']==a['optimizer_step_completions']
    assert set(row['gradient_checks'])==set(keys) and all(q['passed']for q in row['gradient_checks'].values())
    assert '1000'not in row['endpoints'] and set(row['endpoints'])=={'0','100'}
    for metric in row['endpoints'].values():complete_physical(metric,keys)
    assert a['physical_TRAIN_reads']==3 and a['resident_physical_target_reuses']==6
    assert all(a[k]==0 for k in ('encoded_TRAIN_reads','raw_GT_reads','DEV_FINAL_reads','scaler_fits','parent_checkpoint_loads'))
    d.stat_matches(row['last_checkpoint_stat'])
    binding={k:d.identity(o/f,hashed=True)for k,f in [('summary','summary.json'),('status','status.json'),('external','external_terminal.json')]}
    return p,a,e,binding


def inputs():
    p,a,e,b=fp32_budget_binding();sp,sa,se,sb=terminal('notes/geobc_shared_decoder_v61_plan.json')
    assert sa['training_complete'] and sp['scientific']['selected_TRAIN_keys']==p['scientific']['selected_TRAIN_keys']
    report=d.read('results/geobc_shared_decoder_v61/manifest.json')
    assert report['status']=='completed' and report['terminal_binding']==sb and report['SCI']==sp['scientific_digest']
    return dict(shared=sp,shared_summary=sa,shared_external=se,shared_binding=sb,fp32=p,fp32_summary=a,fp32_external=e,fp32_binding=b,keys=p['scientific']['selected_TRAIN_keys'])


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);up=inputs();old=up['fp32']['scientific']
    assert p['kind']=='GeoBC_originalTS_BF16_same_decoder_shared3_v65'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest']==r['scientific_digest']==d.digest(s) and p['source_pins']==r['source_pins']
    assert set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==h for k,h in p['source_pins'].items())
    for k in ('shared','fp32'):
        assert r['upstream_terminal_bindings'][k]==up[k+'_binding']
        assert all(p['source_pins'].get(a)==h for a,h in up[k]['source_pins'].items())
    for k in ('seed','train_CAD_count','dataset_identity','scaler','optimizer','loss','GPU_lock','memory_limit_bytes','decoder_chunk_size',
        'parent_TRAIN_keys','selected_TRAIN_keys','whole_budget_seconds','training_budget_seconds','cache_setup_budget_seconds','representative_field',
        'visualization_sample_nodes_max','fresh_models_per_family','updates_per_field','total_updates_per_family','total_updates','case_pass_endpoints',
        'fresh_initialization','parent_checkpoint_reads_allowed','expected_reference_initialization','FNO_Hash_shared_reference','full_order_digest',
        'DEV_FINAL_reads_allowed','automatic_continuation','FINAL_reuse_for_selection','official_TS_kwargs'):
        assert s[k]==old[k]
    assert s['families']==['TS_ORIGINAL_BF16_GLOBAL_HASH']
    assert s['expected_FP32_TS_initialization']==up['fp32_summary']['model_initialization']['TS_ORIGINAL_GLOBAL_HASH']
    assert s['FP32_TS_partial_endpoints']==up['fp32_summary']['models']['TS_ORIGINAL_GLOBAL_HASH']['endpoints']
    assert s['memory_limit_bytes']==30500000000 and s['whole_budget_seconds']==1200 and s['training_budget_seconds']==900
    assert not s['old_budget_reset'] and not s['old_training_replayed'] and s['new_precision_branch_fresh']
    assert s['full_order_digest']==d.digest(order(up['keys']))
    assert p['output']=='runs/geobc_exploration/Transolver_BF16_continuous_decoder_shared3_v65'
    m=d.read(check_identity(s['dataset_identity']))
    assert all(m['case_roles'][k]=='TRAIN' and k in m['train_keys']['100']for k in up['keys'])
    return p,r,m,up
