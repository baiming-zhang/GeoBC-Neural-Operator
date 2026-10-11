"""Actual two fresh global-BC endpoints, knownDEV120-before-GT60 barrier."""
from . import available_cad_pilot_data as d
from .global_hash_physical_readout_contract_v18 import check_identity,complete_physical
from .global_bc_training_contract_v48 import METHODS,INITIAL,terminal

TRAIN='notes/geobc_global_BC_v48_plan.json'


def training_terminal():
    b,p,a,e=terminal(TRAIN);s=p['scientific'];m=d.read(check_identity(s['dataset_identity']));f=d.path(p['output'])
    assert all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert a['training_complete'] and max(a['elapsed_epoch_seconds'],a['elapsed_monotonic_seconds'],e['elapsed_epoch_seconds'],e['elapsed_monotonic_seconds'])<=3600
    assert max(a[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))<=30500000000
    assert a['raw_GT_reads']==a['DEV_FINAL_reads']==a['scaler_fits']==a['trained_PT_loads']==0
    assert a['cache_load_complete'] and a['shared_cache_load_attempts']==1
    assert a['BC_descriptor_attempts']==a['BC_descriptor_completions']==300
    assert a['cache_receipt']['physical_TRAIN_target_read_attempts']==a['cache_receipt']['physical_TRAIN_target_reads']==a['cache_receipt']['physical_TRAIN_loaded_cases']==300
    assert a['cached_physical_TRAIN_readout_reuses']==600 and set(a['methods'])==set(METHODS)
    assert max(a['cache_setup_epoch_seconds'],a['cache_setup_monotonic_seconds'])<=600
    for n in METHODS:
        arm=a['methods'][n]
        assert arm==d.read(f/n/'summary.json')==d.read(f/n/'status.json')
        assert arm['status']=='completed' and arm['training_complete'] and arm['closed'] and not arm['active']
        assert arm['step']==arm['optimizer_step_attempts']==arm['optimizer_step_completions']==1500
        assert not arm['optimizer_step_in_progress'] and not arm['possible_partial_optimizer_update']
        assert arm['case_exposure_counts']=={k:5 for k in s['TRAIN_keys']} and arm['saved_Adam_steps_equal_cursor']
        assert arm['actual_old_backbone_initial_SHA256']==INITIAL[n] and arm['actual_initial_wholeN_prediction_exact_old']
        assert arm['actual_frozen_BC_reference_unchanged_after_training']
        q=arm['actual_largest_fullN_gradient_check']
        assert q['passed'] and q['initial_prediction_exact_old'] and q['checked_with_training_autograd_enabled'] and q['frozen_reference_gradients_none']
        assert q['full300_physical_and_input_descriptor_cache_qualified'] and q['nodes']==max(m['records'][k]['num_nodes'] for k in s['TRAIN_keys'])
        assert set(q['groups_L2'])=={'geometry_encoder','global_BC_branch','head','operator'} and all(v>0 for v in q['groups_L2'].values())
        assert max(arm['setup_epoch_seconds'],arm['setup_monotonic_seconds'])<=120
        assert max(arm['training_epoch_seconds'],arm['training_monotonic_seconds'])<=900
        assert max(arm['elapsed_epoch_seconds'],arm['elapsed_monotonic_seconds'])<=1500
        point=arm['endpoints']['1500'];complete_physical(point['full_TRAIN'],s['TRAIN_keys'])
        assert point['case_exposure_count']==5 and max(point['epoch_seconds'],point['monotonic_seconds'])<=300
        assert d.relative(point['checkpoint_identity']['path'])==d.relative(f/n/'step1500.pt');d.stat_matches(point['checkpoint_identity'])
    return b,p,a,e,m


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review'])
    assert p['kind']=='GeoBC_two_globalBC_operators_knownDEV60_v49'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest'] and p['source_pins']==r['source_pins']
    assert all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert s['memory_limit_bytes']==30500000000 and s['whole_budget_seconds']==1800
    assert s['GPU_lock']=='runs/geobc_exploration/.coordinator.lock' and s['seed']==2026 and s['decoder_chunk_size']==32768
    assert s['GT_policy']=='All requested full-node predictions committed and revalidated before shared owned-knownDEV60 once'
    assert s['pool']=='known_development60_availability_conditional_not_FINAL' and not s['FINAL_reuse_for_selection']
    assert s['fixed_prediction_denominator']==120 and not s['automatic_continuation']
    assert p['output']=='runs/geobc_exploration/global_BC_two100_knownDEV_v49'
    b,t,a,e,m=training_terminal()
    assert r['training_binding']==b and s['training_plan_identity']==d.identity(TRAIN,hashed=True) and s['training_SCI']==t['scientific_digest']
    assert all(p['source_pins'].get(k)==h for k,h in t['source_pins'].items())
    assert s['dataset_identity']==t['scientific']['dataset_identity'] and s['scaler']==t['scientific']['scaler']==m['scaler']
    keys=s['case_keys'];assert keys==m['development_keys'] and len(keys)==len(set(keys))==60
    assert all(m['case_roles'][k]=='DEV' for k in keys) and len({k.split('/')[0] for k in keys})==20
    known=d.read('notes/geobc_exposure_readout_plan.json')['scientific']
    assert s['GT_manifest_identity']==known['GT_manifest_identity'] and keys==known['case_keys']
    truths=d.read(check_identity(s['GT_manifest_identity']))
    assert truths['status']=='completed' and truths['source_GT_parses']==20 and set(truths['records'])==set(keys)
    for k in keys:
        truth=truths['records'][k]
        assert truth['all_original_nodes'] and truth['signed_stress_unchanged'] and truth['physical']['shape']==[m['records'][k]['num_nodes'],4]
    assert s['candidates']==[dict(name=n,method=n,family=n,step=1500) for n in METHODS]
    qualified={n:dict(plan=t,summary=a,arm=a['methods'][n],external=e,checkpoint=a['methods'][n]['endpoints']['1500']['checkpoint_identity']) for n in METHODS}
    for ref in s['scalar_references']:
        for item in ref['binding'].values():check_identity(item)
        old=d.read(ref['binding']['summary']['path']);oe=d.read(ref['binding']['external']['path'])
        assert old==d.read(ref['binding']['status']['path']) and old['status']=='completed' and old['protocol_qualified']
        assert old['closed'] and not old['active'] and oe['closed'] and not oe['active'] and oe['exit_code']==0 and oe['within_external_budget']
        assert old['execution']['scientific_digest']==oe['scientific_digest']==ref['SCI']
        complete_physical(old['candidates'][ref['name']]['full60'],keys)
    assert [x['name'] for x in s['scalar_references']]==['FNO100_PHYSICAL_1500','R16_TRANSOLVER_1500','FNO100_PHYSICAL_3000']
    assert s['matching_baselines']==dict(FNO9_GLOBAL_BC='FNO100_PHYSICAL_1500',R16_TRANSOLVER_GLOBAL_BC='R16_TRANSOLVER_1500')
    assert s['DEV_gate']==dict(primary_absolute_improvement_at_least=.03,stress_absolute_improvement_at_least=.03,maximum_either_field_absolute_deterioration=.02,fixed_denominator=60)
    return p,r,m,truths,qualified
