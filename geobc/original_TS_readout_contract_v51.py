"""Own original TS physical10, full knownDEV60 even scientific negative."""
from . import available_cad_pilot_data as d
from .global_bc_training_contract_v48 import terminal
from .global_hash_physical_readout_contract_v18 import check_identity,complete_physical
from .original_TS_continuation_contract_v50 import METHOD

TRAIN='notes/geobc_original_TS_v50_plan.json'


def training_terminal():
    b,p,a,e=terminal(TRAIN);s=p['scientific'];m=d.read(check_identity(s['dataset_identity']))
    assert all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert a['training_complete'] and a['family']=='R16_transolver'
    assert a['step']==a['optimizer_step_attempts']==a['optimizer_step_completions']==3000
    assert a['additional_optimizer_step_attempts']==a['additional_optimizer_step_completions']==1500
    assert a['parent_optimizer_steps']==1500 and not a['fresh_initialization']
    assert a['case_exposure_counts']=={k:10 for k in s['TRAIN_keys']} and a['saved_Adam_steps_equal_cursor']
    assert a['parent_saved_Adam_state_count']==a['saved_Adam_state_count']
    assert not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
    assert a['restored_model_Adam_fourRNG_cursor_exposure'] and a['actual_qualification_model_Adam_unchanged']
    q=a['actual_largest_fullN_gradient_check'];assert q['passed'] and q['allowed_unused']==[]
    assert q['nodes']==max(m['records'][k]['num_nodes'] for k in s['TRAIN_keys'])
    assert all(v>0 for v in q['groups_L2'].values()) and q['leaf_gradients']['geometry_context_L2']>0 and q['leaf_gradients']['operator_latent_L2']>0
    assert a['parent_checkpoint_loads']==a['parent_checkpoint_hash_reads']==1
    assert a['cache_load_complete'] and a['cache_receipt']['physical_TRAIN_loaded_cases']==a['cache_receipt']['physical_TRAIN_target_read_attempts']==a['cache_receipt']['physical_TRAIN_target_reads']==300
    assert a['physical_TRAIN_total_array_reads']==a['cached_TRAIN_physical_readout_reuses']==300
    assert a['cached_TRAIN_physical_read_attempts']==a['cached_TRAIN_physical_reads']==a['raw_GT_reads']==a['DEV_FINAL_reads']==a['scaler_fits']==0
    assert max(a['cache_resume_epoch_seconds'],a['cache_resume_monotonic_seconds'])<=600
    assert max(a['qualification_epoch_seconds'],a['qualification_monotonic_seconds'])<=120
    assert max(a['training_epoch_seconds'],a['training_monotonic_seconds'])<=900
    assert max(a['elapsed_epoch_seconds'],a['elapsed_monotonic_seconds'],e['elapsed_epoch_seconds'],e['elapsed_monotonic_seconds'])<=2100
    assert max(a[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))<=30500000000
    point=a['endpoints']['3000'];complete_physical(point['full_TRAIN'],s['TRAIN_keys'])
    assert point['case_exposure_count']==10 and max(point['epoch_seconds'],point['monotonic_seconds'])<=300
    assert a['trajectory_training_epoch_seconds']==a['parent_cost']['training_epoch_seconds']+a['training_epoch_seconds']
    assert a['trajectory_training_monotonic_seconds']==a['parent_cost']['training_monotonic_seconds']+a['training_monotonic_seconds']
    assert d.relative(point['checkpoint_identity']['path'])==p['output']+'/step3000.pt';d.stat_matches(point['checkpoint_identity'])
    return b,p,a,e,m


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review'])
    assert p['kind']=='GeoBC_original_R16_TS_physical10_knownDEV60_v51'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest'] and p['source_pins']==r['source_pins']
    assert all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert s['memory_limit_bytes']==30500000000 and s['whole_budget_seconds']==1800
    assert s['GPU_lock']=='runs/geobc_exploration/.coordinator.lock' and s['seed']==2026 and s['decoder_chunk_size']==32768
    assert s['GT_policy']=='All requested full-node predictions committed and revalidated before shared owned-knownDEV60 once'
    assert s['pool']=='known_development60_availability_conditional_not_FINAL' and not s['FINAL_reuse_for_selection']
    assert s['fixed_prediction_denominator']==60 and not s['automatic_continuation'] and s['score_scientific_negative_endpoints']
    assert p['output']=='runs/geobc_exploration/original_R16_TS100_physical10_knownDEV_v51'
    b,t,a,e,m=training_terminal()
    assert r['training_binding']==b and s['training_plan_identity']==d.identity(TRAIN,hashed=True) and s['training_SCI']==t['scientific_digest']
    assert all(p['source_pins'].get(k)==h for k,h in t['source_pins'].items())
    assert s['dataset_identity']==t['scientific']['dataset_identity'] and s['scaler']==t['scientific']['scaler']==m['scaler']
    keys=s['case_keys'];assert keys==m['development_keys'] and len(keys)==len(set(keys))==60
    assert all(m['case_roles'][k]=='DEV' for k in keys) and len({k.split('/')[0] for k in keys})==20
    known=d.read('notes/geobc_exposure_readout_plan.json')['scientific'];assert s['GT_manifest_identity']==known['GT_manifest_identity'] and keys==known['case_keys']
    truths=d.read(check_identity(s['GT_manifest_identity']));assert truths['status']=='completed' and truths['source_GT_parses']==20 and set(truths['records'])==set(keys)
    for k in keys:
        v=truths['records'][k];assert v['all_original_nodes'] and v['signed_stress_unchanged'] and v['physical']['shape']==[m['records'][k]['num_nodes'],4]
    assert s['candidates']==[dict(name=METHOD,method='transolver',family='R16_transolver',step=3000)]
    qualified={METHOD:dict(plan=t,summary=a,external=e,checkpoint=a['endpoints']['3000']['checkpoint_identity'])}
    for ref in s['scalar_references']:
        for item in ref['binding'].values():check_identity(item)
        old=d.read(ref['binding']['summary']['path']);oe=d.read(ref['binding']['external']['path'])
        assert old==d.read(ref['binding']['status']['path']) and old['status']=='completed' and old['protocol_qualified']
        assert old['closed'] and not old['active'] and oe['closed'] and not oe['active'] and oe['exit_code']==0 and oe['within_external_budget']
        assert old['execution']['scientific_digest']==oe['scientific_digest']==ref['SCI']
        complete_physical(old['candidates'][ref['name']]['full60'],keys)
    assert [x['name'] for x in s['scalar_references']]==['FNO100_PHYSICAL_1500','R16_TRANSOLVER_1500','FNO100_PHYSICAL_3000']
    assert s['matching_baselines']=={METHOD:'R16_TRANSOLVER_1500'}
    assert s['DEV_gate']==dict(primary_absolute_improvement_at_least=.03,stress_absolute_improvement_at_least=.03,maximum_either_field_absolute_deterioration=.02,fixed_denominator=60)
    return p,r,m,truths,qualified
