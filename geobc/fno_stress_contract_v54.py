"""One current100 matched-parent stress-only branch; metadata-only gates."""
from . import available_cad_pilot_data as d
from .global_hash_physical_readout_contract_v18 import check_identity,complete_physical

CONTROL='notes/geobc_macro100_FNO9_physical10_v15_plan.json'
DEV='notes/geobc_physical10_readout_v16_plan.json'
SOURCES=('geobc/fno_stress_specialist_v54.py','geobc/fno_stress_contract_v54.py',
    'tools/train_geobc_fno_stress_v54.py','tools/run_geobc_fno_stress_guard_v54.py',
    'tools/prepare_geobc_fno_stress_v54.py')


def terminal(plan_path):
    p=d.read(plan_path);f=d.path(p['output']);r=d.read(p['root_review'])
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest']==r['scientific_digest']==d.digest(p['scientific'])
    assert p['source_pins']==r['source_pins'] and all(d.sha(k)==h for k,h in p['source_pins'].items())
    a=d.read(f/'summary.json');e=d.read(f/'external_terminal.json')
    assert a==d.read(f/'status.json') and a['status']=='completed' and a['closed'] and not a['active']
    assert e['exit_code']==0 and e['within_external_budget'] and e['closed'] and not e['active'] and not e['external_timeout']
    assert a['execution']['scientific_digest']==e['scientific_digest']==p['scientific_digest']
    assert e['source_pins']==p['source_pins'] and a['execution']['run_process']==e['child_process']
    assert a['execution']['guard']['guard_process']==e['guard_process']
    assert all(d.exact_process_exited(e[k]) for k in ('child_process','guard_process'))
    return p,a,e,{k:d.identity(f/name,hashed=True) for k,name in
        [('summary','summary.json'),('status','status.json'),('external','external_terminal.json')]}


def metadata_inputs():
    cp,ca,ce,cb=terminal(CONTROL);cs=cp['scientific']
    pp,pa,pe,pb=terminal(cs['parent_training_plan_identity']['path']);ps=pp['scientific']
    vp,va,ve,vb=terminal(DEV)
    assert cs['family']==ps['family']=='FNO9' and ca['training_complete'] and pa['training_complete']
    keys=ps['TRAIN_keys'];assert len(keys)==300 and ps['train_CAD_count']==100
    for a,count,step in [(pa,5,1500),(ca,10,3000)]:
        assert a['step']==a['optimizer_step_attempts']==a['optimizer_step_completions']==step
        assert not a['possible_partial_optimizer_update'] and not a['optimizer_step_in_progress']
        assert a['case_exposure_counts']=={k:count for k in keys}
        assert a['actual_largest_fullN_gradient_check']['passed']
        complete_physical(a['endpoints'][str(step)]['full_TRAIN'],keys)
        assert a['raw_GT_reads']==a['DEV_FINAL_reads']==a['scaler_fits']==0
    assert ca['loaded_parent_model_sha256'] and ca['restored_model_Adam_fourRNG_cursor_exposure']
    assert cs['parent_checkpoint']==pa['endpoints']['1500']['checkpoint_identity']
    d.stat_matches(cs['parent_checkpoint'])
    assert cs['dataset_identity']==ps['dataset_identity'] and cs['scaler']==ps['scaler']
    assert cs['optimizer']==ps['optimizer'] and cs['parent_case_exposure_counts']==pa['case_exposure_counts']
    m=d.read(check_identity(cs['dataset_identity']));assert m['scaler']==cs['scaler'] and m['train_keys']['100']==keys
    assert va['protocol_qualified'] and va['committed_finite_predictions']==va['owned_GT_array_reads']==60
    ref=va['candidates']['FNO100_PHYSICAL_3000'];complete_physical(ref['full60'],m['development_keys'])
    assert ref['checkpoint']==ca['endpoints']['3000']['checkpoint_identity']
    assert cs['parent_training_SCI']==pp['scientific_digest']
    return dict(parent=pp,parent_summary=pa,parent_external=pe,parent_binding=pb,
        control=cp,control_summary=ca,control_external=ce,control_binding=cb,
        DEV_plan=vp,DEV_summary=va,DEV_external=ve,DEV_binding=vb,dataset=m)


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);up=metadata_inputs()
    assert p['kind']=='GeoBC_FNO9_matched_stress_specialization_v54'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest'] and p['source_pins']==r['source_pins']
    assert set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert all(p['source_pins'].get(k)==h for k,h in up['control']['source_pins'].items())
    assert r['upstream_terminal_bindings']=={k:up[k+'_binding'] for k in ('parent','control','DEV')}
    assert s['family']=='FNO9' and s['seed']==2026 and s['train_CAD_count']==100
    assert s['memory_limit_bytes']==30500000000 and s['GPU_lock']=='runs/geobc_exploration/.coordinator.lock'
    assert s['cache_setup_budget_seconds']==600 and s['training_budget_seconds']==600
    assert s['TRAIN_readout_budget_seconds']==300 and s['whole_budget_seconds']==1800
    assert s['parent_step']==1500 and s['additional_updates']==1500 and s['fixed_updates']==3000
    assert s['endpoint_passes']==[10] and not s['fresh_initialization']
    assert s['decoder_chunk_size']==32768 and s['stress_loss_coefficient']==.5
    assert s['loss']=='full_original_node_half_physical_S_squared_no_U_loss'
    assert s['precision']=='highest_FP32_TF32_disabled' and not s['automatic_continuation']
    assert s['DEV_GT_reads_allowed']==0 and not s['FINAL_reuse_for_selection']
    ps=up['parent']['scientific'];cs=up['control']['scientific']
    for key in ('TRAIN_keys','dataset_identity','scaler','optimizer'):assert s[key]==ps[key]==cs[key]
    assert s['parent_training_SCI']==up['parent']['scientific_digest']
    assert s['parent_checkpoint']==cs['parent_checkpoint'] and s['parent_case_exposure_counts']==up['parent_summary']['case_exposure_counts']
    assert s['expected_loaded_parent_model_sha256']==up['control_summary']['loaded_parent_model_sha256']
    assert s['parent_norms_identity']==cs['parent_norms_identity']
    check_identity(s['parent_norms_identity'])
    assert s['control_TRAIN300']==up['control_summary']['endpoints']['3000']['full_TRAIN']
    assert s['TRAIN_gate']==dict(absolute_S_improvement_min=.03,hotspot_deterioration_max=.02,fixed_denominator=300)
    from .macro_training import balanced_schedule
    order=balanced_schedule(s['TRAIN_keys'],10,2026)
    assert d.digest(order)==s['full_order_digest']==cs['full_order_digest']
    assert d.digest(order[:1500])==s['parent_full_order_digest']==ps['full_order_digest']
    assert p['output']=='runs/geobc_exploration/FNO100_stress_specialist_v54'
    return p,r,up['dataset'],order,up
