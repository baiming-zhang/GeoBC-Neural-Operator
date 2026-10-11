"""Metadata-only release of one exposure-matched four-case Adam recipe."""
from . import available_cad_pilot_data as d
from .fno_stress_contract_v54 import terminal,metadata_inputs as parent_inputs
from .global_hash_physical_readout_contract_v18 import check_identity

SOURCES=('geobc/fno_case_batch_v55.py','geobc/fno_case_batch_contract_v55.py',
    'tools/train_geobc_fno_case_batch_v55.py','tools/run_geobc_fno_case_batch_guard_v55.py',
    'tools/prepare_geobc_fno_case_batch_v55.py')


def metadata_inputs():
    up=parent_inputs()
    p,a,e,b=terminal('notes/geobc_fno_stress_v54_plan.json')
    assert a['training_complete'] and not a['TRAIN_gate']['passed']
    assert a['additional_optimizer_step_completions']==1500 and a['DEV_FINAL_reads']==0
    up.update(previous_plan=p,previous_summary=a,previous_external=e,previous_binding=b)
    return up


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);up=metadata_inputs()
    assert p['kind']=='GeoBC_FNO9_four_case_mean_gradient_v55'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest'] and p['source_pins']==r['source_pins']
    assert set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert all(p['source_pins'].get(k)==h for k,h in up['previous_plan']['source_pins'].items())
    assert r['upstream_terminal_bindings']=={k:up[k+'_binding'] for k in ('parent','control','DEV','previous')}
    assert s['family']=='FNO9' and s['seed']==2026 and s['train_CAD_count']==100
    assert s['memory_limit_bytes']==30500000000 and s['GPU_lock']=='runs/geobc_exploration/.coordinator.lock'
    assert s['cache_setup_budget_seconds']==s['training_budget_seconds']==600
    assert s['TRAIN_readout_budget_seconds']==300 and s['whole_budget_seconds']==1800
    assert s['parent_step']==s['parent_exposure_cursor']==1500
    assert s['additional_optimizer_updates']==375 and s['fixed_optimizer_steps']==1875
    assert s['additional_case_exposures']==1500 and s['fixed_exposure_cursor']==3000
    assert s['cases_per_optimizer_update']==4 and s['case_gradient_weights']==[.25]*4
    assert s['endpoint_passes']==[10] and not s['fresh_initialization']
    assert s['decoder_chunk_size']==32768 and s['loss']=='mean4_full_original_node_half_U_squared_plus_half_S_squared'
    assert s['precision']=='highest_FP32_TF32_disabled' and not s['automatic_continuation']
    assert s['DEV_GT_reads_allowed']==0 and not s['FINAL_reuse_for_selection']
    ps=up['parent']['scientific'];cs=up['control']['scientific']
    for k in ('TRAIN_keys','dataset_identity','scaler','optimizer'):assert s[k]==ps[k]==cs[k]
    assert s['parent_training_SCI']==up['parent']['scientific_digest']
    assert s['parent_checkpoint']==cs['parent_checkpoint'] and s['parent_case_exposure_counts']==up['parent_summary']['case_exposure_counts']
    assert s['expected_loaded_parent_model_sha256']==up['control_summary']['loaded_parent_model_sha256']
    assert s['parent_norms_identity']==cs['parent_norms_identity'];check_identity(s['parent_norms_identity'])
    assert s['control_TRAIN300']==up['control_summary']['endpoints']['3000']['full_TRAIN']
    assert s['TRAIN_gate']==dict(primary_absolute_improvement_at_least=.03,
        maximum_either_field_absolute_deterioration=.02,hotspot_deterioration_max=.02,fixed_denominator=300)
    assert s['cursor_semantics']=='step=actual Adam count; cursor=completed ordered case exposures'
    from .macro_training import balanced_schedule
    order=balanced_schedule(s['TRAIN_keys'],10,2026)
    assert d.digest(order)==s['full_order_digest']==cs['full_order_digest']
    assert d.digest(order[:1500])==s['parent_full_order_digest']==ps['full_order_digest']
    assert all(len(set(order[i:i+4]))==4 for i in range(1500,3000,4))
    assert p['output']=='runs/geobc_exploration/FNO100_case_batch4_v55'
    return p,r,up['dataset'],order,up
