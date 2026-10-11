"""Two fresh short screens: input-only global BC routing, no inherited models."""
from . import available_cad_pilot_data as d
from .global_hash_physical_readout_contract_v18 import check_identity,complete_physical
from .resolution_training_contract_v42 import baseline_binding

METHODS = ['FNO9_GLOBAL_BC','R16_TRANSOLVER_GLOBAL_BC']
INITIAL = dict(FNO9_GLOBAL_BC='e3c06ebfb126495173ae11b94fb3ef0cf87392976d56ba36238358844dd516da',
    R16_TRANSOLVER_GLOBAL_BC='13d31df15c487a1240b8f1f81f0a7ccf7a15a73f32cbbb29fb3dff34632e5abb')


def terminal(plan_name):
    p=d.read(plan_name);r=d.read(p['root_review']);f=d.path(p['output'])
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest'] and p['source_pins']==r['source_pins']
    a=d.read(f/'summary.json');e=d.read(f/'external_terminal.json')
    assert a==d.read(f/'status.json') and a['status']=='completed' and a['closed'] and not a['active']
    assert e['closed'] and not e['active'] and e['exit_code']==0 and e['within_external_budget'] and not e['external_timeout']
    assert a['execution']['run_process']==e['child_process'] and a['execution']['guard']['guard_process']==e['guard_process']
    assert a['execution']['scientific_digest']==e['scientific_digest']==p['scientific_digest'] and e['source_pins']==p['source_pins']
    assert all(d.exact_process_exited(e[k]) for k in ('child_process','guard_process'))
    return dict(plan=d.identity(plan_name,hashed=True),terminal={k:d.identity(f/file,hashed=True)
        for k,file in [('summary','summary.json'),('status','status.json'),('external','external_terminal.json')]}),p,a,e


def predecessor_binding():
    b,p,a,e=terminal('notes/geobc_operator_residual_readout_v46_plan.json')
    assert a['protocol_qualified'] and a['committed_finite_predictions']==120
    assert a['owned_GT_read_attempts']==a['owned_GT_array_reads']==60
    assert a['raw_GT_reads']==a['FINAL_numeric_reads']==a['training_updates']==a['scaler_fits']==0
    for row in a['candidates'].values(): complete_physical(row['full60'],p['scientific']['case_keys'])
    return b,p


def original_TS_binding():
    b,p,a,e=terminal('notes/geobc_coarse_three100_physical5_v20_plan.json')
    arm=a['methods']['transolver']; assert a['training_complete'] and arm['training_complete']
    assert arm['step']==arm['optimizer_step_completions']==1500 and arm['case_exposure_counts']=={k:5 for k in p['scientific']['TRAIN_keys']}
    assert arm['actual_fresh_initial_model_sha256']==INITIAL[METHODS[1]] and arm['saved_Adam_steps_equal_cursor']
    complete_physical(arm['endpoints']['1500']['full_TRAIN'],p['scientific']['TRAIN_keys'])
    return b,p


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review'])
    assert p['kind']=='GeoBC_two_fresh_globalBC_operators_physical100_short5_v48'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest'] and p['source_pins']==r['source_pins']
    assert all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert s['methods']==METHODS and s['seed']==2026 and s['train_CAD_count']==100
    assert s['fixed_updates_per_arm']==1500 and s['passes']==5 and s['initial_steps']=={k:0 for k in METHODS}
    assert s['old_backbone_initial_SHA256']==INITIAL
    assert s['expected_parameters']==dict(FNO9_GLOBAL_BC=4539850,R16_TRANSOLVER_GLOBAL_BC=1895164)
    assert s['expected_trainable_parameters']==dict(FNO9_GLOBAL_BC=4533354,R16_TRANSOLVER_GLOBAL_BC=1888668)
    assert s['BC_descriptor_dim']==68 and s['added_mutable_parameters']==s['added_frozen_parameters']==6496
    assert s['memory_limit_bytes']==30500000000 and s['GPU_lock']=='runs/geobc_exploration/.coordinator.lock'
    assert [s[k] for k in ('cache_setup_budget_seconds','arm_setup_budget_seconds','training_budget_seconds_per_arm','TRAIN_readout_budget_seconds_per_arm','arm_budget_seconds','whole_budget_seconds')]==[600,120,900,300,1500,3600]
    assert s['decoder_chunk_size']==32768 and s['loss']=='full_original_node_physical_relative_U_S_squared_equal_case'
    assert s['optimizer']==dict(name='Adam',learning_rate=.001,weight_decay=0.,betas=[.9,.999],eps=1e-8,fused=True)
    assert s['precision']=='highest FP32; TF32 disabled' and s['DEV_FINAL_reads_allowed']==s['scaler_fits_allowed']==0
    assert s['shared_cache_loads']==1 and s['BC_descriptors_once_per_TRAIN_case']==300
    assert s['qualification_RNG_restored'] and s['TRAIN_readout_reuses_physical_cache'] and s['original_R64_grids_retained']
    assert s['inherited_teacher_passes']==s['trained_checkpoint_loads_allowed']==0
    assert not s['capacity_matched_to_old_baseline'] and not s['automatic40'] and not s['automatic24000']
    prior,pp=predecessor_binding(); assert s['predecessor_terminal']==prior
    base,bp=baseline_binding(); assert s['old_FNO5_terminal']==base
    ts,tp=original_TS_binding(); assert s['old_TS5_terminal']==ts
    assert s['dataset_identity']==pp['scientific']['dataset_identity']==bp['scientific']['dataset_identity']==tp['scientific']['dataset_identity']
    assert s['scaler']==pp['scientific']['scaler']==bp['scientific']['scaler']==tp['scientific']['scaler']
    m=d.read(check_identity(s['dataset_identity']));keys=m['train_keys']['100']
    assert m['status']=='completed' and m['feasibility_pass'] and keys==s['TRAIN_keys'] and len(set(keys))==len(keys)==300
    assert all(m['case_roles'][k]=='TRAIN' for k in keys) and m['scaler']==s['scaler']
    from .macro_training import balanced_schedule
    order=balanced_schedule(keys,5,2026)
    assert d.digest(order)==s['full_order_digest']==bp['scientific']['full_order_digest']==tp['scientific']['full_order_digest']
    proof=d.read(check_identity(s['CPU_fixture']))
    assert proof['status']=='actual_synthetic_CPU_two_globalBC_operators_zero_output_joint_physical_gradient_passed_not_nativeGPU'
    assert proof['GPU_calls']==proof['dataset_GT_PT_numeric_reads']==proof['optimizer_updates']==0
    assert proof['source']==d.identity('geobc/global_bc_solver_v48.py',hashed=True) and proof['checker']==d.identity('tools/check_global_bc_solver_v48_CPU.py',hashed=True)
    for method in METHODS:
        q=proof['methods'][method]
        assert q['initial_context_latent_prediction_exact_old'] and q['old_backbone_initial_exact'] and q['frozen_reference_gradients_none']
        assert all(v>0 for v in q['all_four_trainable_groups_L2'].values()) and q['current_case_descriptor_required']
    assert p['output']=='runs/geobc_exploration/global_BC_two100_physical5_v48'
    return p,r,m,order
