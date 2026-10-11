"""One fixed fresh structured3D endpoint on the existing knownDEV60.

Only direct, current-stage metadata checks; no recursive history replay.
"""
from . import available_cad_pilot_data as d
from .global_hash_physical_readout_contract_v18 import check_identity, complete_physical

TRAIN = 'notes/geobc_structured_TS_v39_plan.json'
METHOD = 'transolver_structured3d'
NAME = 'R16_STRUCTURED_TRANSOLVER_1500'
REFERENCES = ['R16_TRANSOLVER_1500', 'R16_FNO16_CONTROL_1500', 'FNO100_PHYSICAL_1500', 'R64_FNO_W16_1500']


def training_terminal():
    p=d.read(TRAIN);r=d.read(p['root_review']);s=p['scientific'];f=d.path(p['output'])
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['kind']=='GeoBC_official_structured3D_sameR16_freshphysical100_short5_v39'
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest'] and p['source_pins']==r['source_pins']
    assert all(d.sha(k)==h for k,h in p['source_pins'].items())
    m=d.read(check_identity(s['dataset_identity']))
    assert m['status']=='completed' and m['feasibility_pass'] and m['scaler']==s['scaler']
    assert m['train_keys']['100']==s['TRAIN_keys'] and len(s['TRAIN_keys'])==300
    assert s['methods']==[METHOD] and s['additional_passes']==5 and s['additional_updates_per_arm']==1500
    assert s['teacher_prediction_distillation'] is False and s['inherited_teacher_passes']==0
    a=d.read(f/'summary.json');e=d.read(f/'external_terminal.json')
    assert a==d.read(f/'status.json') and a['status']=='completed' and a['training_complete']
    assert a['closed'] and not a['active'] and e['closed'] and not e['active']
    assert e['exit_code']==0 and e['within_external_budget'] and not e['external_timeout']
    assert e['source_pins']==p['source_pins'] and all(d.exact_process_exited(e[k]) for k in ('child_process','guard_process'))
    assert a['execution']['run_process']==e['child_process'] and a['execution']['guard']['guard_process']==e['guard_process']
    assert a['execution']['boot_id']==e['boot_id'] and a['execution']['scientific_digest']==e['scientific_digest']==p['scientific_digest']
    assert max(a['elapsed_epoch_seconds'],a['elapsed_monotonic_seconds'],e['elapsed_epoch_seconds'],e['elapsed_monotonic_seconds'])<=2100
    assert max(a[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))<=30500000000
    assert a['raw_GT_reads']==a['DEV_FINAL_reads']==a['scaler_fits']==a['PT_checkpoint_loads']==a['inherited_teacher_passes']==0
    assert a['cache_load_complete'] and a['shared_cache_load_attempts']==1 and a['cached_physical_TRAIN_readout_reuses']==300
    assert a['cache_receipt']['physical_TRAIN_loaded_cases']==a['cache_receipt']['physical_TRAIN_target_reads']==300
    assert max(a['cache_setup_epoch_seconds'],a['cache_setup_monotonic_seconds'])<=600
    assert set(a['methods'])=={METHOD};arm=a['methods'][METHOD]
    assert arm==d.read(f/METHOD/'summary.json')==d.read(f/METHOD/'status.json')
    assert arm['status']=='completed' and arm['closed'] and not arm['active'] and arm['training_complete']
    assert arm['step']==arm['optimizer_step_attempts']==arm['optimizer_step_completions']==arm['additional_optimizer_step_completions']==1500
    assert arm['parent_optimizer_steps']==0 and not arm['optimizer_step_in_progress'] and not arm['possible_partial_optimizer_update']
    assert arm['case_exposure_counts']=={k:5 for k in s['TRAIN_keys']}
    assert arm['actual_fresh_initial_state_matches_frozen_candidate'] and arm['actual_fresh_initial_model_sha256']==s['fresh_initial_SHA256'][METHOD]
    assert arm['actual_reset_Adam_empty'] and arm['saved_Adam_steps_equal_cursor']
    q=arm['actual_largest_fullN_gradient_check'];assert q['passed'] and q['full300_physical_cache_qualified']
    assert q['nodes']==max(m['records'][k]['num_nodes'] for k in s['TRAIN_keys'])
    assert q['allowed_unused']==s['allowed_unused_parameters']==['operator.backbone.placeholder']
    assert all(v>0 for v in q['groups_L2'].values()) and q['leaf_gradients']['geometry_context_L2']>0 and q['leaf_gradients']['operator_latent_L2']>0
    assert max(arm['setup_epoch_seconds'],arm['setup_monotonic_seconds'])<=120
    assert max(arm['training_epoch_seconds'],arm['training_monotonic_seconds'])<=900
    assert max(arm['elapsed_epoch_seconds'],arm['elapsed_monotonic_seconds'])<=1500
    point=arm['endpoints']['1500'];complete_physical(point['full_TRAIN'],s['TRAIN_keys'])
    assert point['case_exposure_count']==5 and point['additional_passes']==5 and point['teacher_derived_initialization_passes']==0
    assert max(point['epoch_seconds'],point['monotonic_seconds'])<=300
    assert d.relative(point['checkpoint_identity']['path'])==d.relative(f/METHOD/'step1500.pt')
    d.stat_matches(point['checkpoint_identity'])
    binding={k:d.identity(f/file,hashed=True) for k,file in [('summary','summary.json'),('status','status.json'),('external','external_terminal.json')]}
    return binding,p,a,e,m


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review'])
    assert p['kind']=='GeoBC_structured3D_TS100_freshphysical5_knownDEV60_v40'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest'] and p['source_pins']==r['source_pins']
    assert all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert s['memory_limit_bytes']==30500000000 and s['whole_budget_seconds']==1800
    assert s['GPU_lock']=='runs/geobc_exploration/.coordinator.lock' and s['seed']==2026 and s['decoder_chunk_size']==32768
    assert s['GT_policy']=='All requested full-node predictions committed and revalidated before shared owned-knownDEV60 once'
    assert s['pool']=='known_development60_availability_conditional_not_FINAL'
    assert s['fixed_prediction_denominator']==60 and not s['FINAL_reuse_for_selection']
    assert p['output']=='runs/geobc_exploration/structured_TS100_knownDEV_freshphysical5_v40'
    binding,t,a,e,m=training_terminal()
    assert r['training_binding']==binding and s['training_plan_identity']==d.identity(TRAIN,hashed=True) and s['training_SCI']==t['scientific_digest']
    assert all(p['source_pins'].get(k)==h for k,h in t['source_pins'].items())
    assert s['dataset_identity']==t['scientific']['dataset_identity'] and s['scaler']==t['scientific']['scaler']==m['scaler']
    keys=s['case_keys'];assert keys==m['development_keys'] and len(keys)==len(set(keys))==60
    assert len({k.split('/')[0] for k in keys})==20 and all(m['case_roles'][k]=='DEV' for k in keys)
    known=d.read('notes/geobc_exposure_readout_plan.json')['scientific']
    assert s['GT_manifest_identity']==known['GT_manifest_identity'] and keys==known['case_keys']
    truths=d.read(check_identity(s['GT_manifest_identity']))
    assert truths['status']=='completed' and truths['source_GT_parses']==20 and set(truths['records'])==set(keys)
    for key in keys:
        truth=truths['records'][key]
        assert truth['all_original_nodes'] and truth['signed_stress_unchanged'] and truth['physical']['shape']==[m['records'][key]['num_nodes'],4]
    assert s['candidates']==[dict(name=NAME,method=METHOD,family='R16_'+METHOD,step=1500)]
    qualified={NAME:dict(plan=t,summary=a,arm=a['methods'][METHOD],external=e,checkpoint=a['methods'][METHOD]['endpoints']['1500']['checkpoint_identity'])}
    prior=s['prior_development_binding']
    for item in prior.values():check_identity(item)
    old=d.read(prior['summary']['path']);oe=d.read(prior['external']['path'])
    assert old==d.read(prior['status']['path']) and old['status']=='completed' and old['protocol_qualified']
    assert old['closed'] and not old['active'] and oe['closed'] and not oe['active'] and oe['exit_code']==0 and oe['within_external_budget']
    assert all(d.exact_process_exited(oe[k]) for k in ('child_process','guard_process'))
    assert old['execution']['run_process']==oe['child_process'] and old['execution']['guard']['guard_process']==oe['guard_process']
    assert old['execution']['scientific_digest']==oe['scientific_digest']==s['prior_development_SCI']
    assert old['committed_finite_predictions']==180 and old['owned_GT_array_reads']==old['owned_GT_read_attempts']==60
    assert old['raw_GT_reads']==old['FINAL_numeric_reads']==old['training_updates']==old['scaler_fits']==0
    assert s['reference_names']==REFERENCES
    for name in REFERENCES:complete_physical(old['candidates'][name]['full60'],keys)
    assert s['DEV_gates']==dict(TS_own_primary_improvement_at_least=.05,FNO_priority_primary_improvement_at_least=.03,maximum_either_field_absolute_deterioration=.02,fixed_denominator=60)
    assert not s['automatic_continuation'] and not s['pure_architecture_causality_claimed']
    return p,r,m,truths,qualified
