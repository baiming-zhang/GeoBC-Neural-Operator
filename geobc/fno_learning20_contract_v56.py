"""A separately released fixed endpoint, justified by real 5-to10 gains."""
from . import available_cad_pilot_data as d
from .fno_stress_contract_v54 import metadata_inputs as prior_inputs,terminal
from .global_hash_physical_readout_contract_v18 import check_identity

SOURCES=('geobc/fno_learning20_contract_v56.py','tools/train_geobc_fno_learning20_v56.py',
    'tools/run_geobc_fno_learning20_guard_v56.py','tools/prepare_geobc_fno_learning20_v56.py')


def metadata_inputs():
    old=prior_inputs();pp=old['control'];pa=old['control_summary']
    assert pa['TRAIN_gate']['passed'] and old['DEV_summary']['DEV_gate']['passed']
    assert pa['step']==3000 and pa['case_exposure_counts']=={k:10 for k in pp['scientific']['TRAIN_keys']}
    assert pa['original_parent_fresh_initial_model_sha256']
    d.stat_matches(pa['endpoints']['3000']['checkpoint_identity'])
    lp,la,le,lb=terminal('notes/geobc_fno_case_batch_v55_plan.json')
    assert la['training_complete'] and not la['TRAIN_gate']['passed']
    assert la['step']==1875 and la['cursor']==3000
    assert la['additional_optimizer_step_completions']==375 and la['DEV_FINAL_reads']==0
    return dict(parent=pp,parent_summary=pa,parent_external=old['control_external'],parent_binding=old['control_binding'],
        original5_binding=old['parent_binding'],DEV_plan=old['DEV_plan'],DEV_summary=old['DEV_summary'],DEV_binding=old['DEV_binding'],
        last_recipe_plan=lp,last_recipe_summary=la,last_recipe_binding=lb,dataset=old['dataset'])


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);up=metadata_inputs()
    assert p['kind']=='GeoBC_FNO9_evidence_gated_physical10_to20_v56'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest'] and p['source_pins']==r['source_pins']
    assert set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert all(p['source_pins'].get(k)==h for k,h in up['last_recipe_plan']['source_pins'].items())
    assert r['upstream_terminal_bindings']=={k:up[k+'_binding']for k in ('parent','original5','DEV','last_recipe')}
    assert s['family']=='FNO9' and s['seed']==2026 and s['train_CAD_count']==100
    assert s['memory_limit_bytes']==30500000000 and s['GPU_lock']=='runs/geobc_exploration/.coordinator.lock'
    assert s['cache_setup_budget_seconds']==s['training_budget_seconds']==600
    assert s['TRAIN_readout_budget_seconds']==300 and s['whole_budget_seconds']==1800
    assert s['parent_step']==3000 and s['additional_updates']==3000 and s['fixed_updates']==6000
    assert s['endpoint_passes']==[20] and not s['fresh_initialization']
    assert s['decoder_chunk_size']==32768 and s['loss']==up['parent']['scientific']['loss']
    assert s['precision']=='highest_FP32_TF32_disabled' and not s['automatic_continuation']
    assert s['DEV_GT_reads_allowed']==0 and not s['FINAL_reuse_for_selection']
    ps=up['parent']['scientific']
    for k in ('TRAIN_keys','dataset_identity','scaler','optimizer'):assert s[k]==ps[k]
    assert s['parent_training_SCI']==up['parent']['scientific_digest']
    assert s['parent_checkpoint']==up['parent_summary']['endpoints']['3000']['checkpoint_identity']
    assert s['parent_case_exposure_counts']==up['parent_summary']['case_exposure_counts']
    assert s['parent_norms_identity']==ps['parent_norms_identity'];check_identity(s['parent_norms_identity'])
    assert s['parent_TRAIN300']==up['parent_summary']['endpoints']['3000']['full_TRAIN']
    assert s['TRAIN_gate']==dict(primary_absolute_improvement_at_least=.03,
        maximum_either_field_absolute_deterioration=.02,hotspot_deterioration_max=.02,fixed_denominator=300)
    assert s['previous_5to10_TRAIN_gate']==up['parent_summary']['TRAIN_gate']
    assert s['previous_5to10_DEV_gate']==up['DEV_summary']['DEV_gate']
    assert s['release_scope']=='One selected fixed20 trend check; no automatic40 or24000 or other solver extension'
    from .macro_training import balanced_schedule
    order=balanced_schedule(s['TRAIN_keys'],20,2026)
    assert d.digest(order)==s['full_order_digest']
    assert d.digest(order[:3000])==s['parent_full_order_digest']==ps['full_order_digest']
    assert p['output']=='runs/geobc_exploration/FNO100_physical20_v56'
    return p,r,up['dataset'],order,up
