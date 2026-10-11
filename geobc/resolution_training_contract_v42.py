"""Fixed R128 FNO5 with direct current terminal and baseline metadata."""
from . import available_cad_pilot_data as d
from .global_hash_physical_readout_contract_v18 import check_identity,complete_physical

METHOD='FNO9_R128'
INITIAL='e3c06ebfb126495173ae11b94fb3ef0cf87392976d56ba36238358844dd516da'


def predecessor_binding():
    p=d.read('notes/geobc_structured_TS_readout_v40_plan.json');r=d.read(p['root_review']);f=d.path(p['output'])
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest'] and p['source_pins']==r['source_pins']
    a=d.read(f/'summary.json');e=d.read(f/'external_terminal.json')
    assert a==d.read(f/'status.json') and a['status']=='completed' and a['protocol_qualified']
    assert a['closed'] and not a['active'] and e['closed'] and not e['active'] and e['exit_code']==0 and e['within_external_budget']
    assert all(d.exact_process_exited(e[k]) for k in ('child_process','guard_process'))
    assert a['execution']['run_process']==e['child_process'] and a['execution']['guard']['guard_process']==e['guard_process']
    assert a['execution']['scientific_digest']==e['scientific_digest']==p['scientific_digest'] and e['source_pins']==p['source_pins']
    assert a['committed_finite_predictions']==a['owned_GT_array_reads']==a['owned_GT_read_attempts']==60
    assert a['raw_GT_reads']==a['FINAL_numeric_reads']==a['training_updates']==a['scaler_fits']==0
    for row in a['candidates'].values():complete_physical(row['full60'],p['scientific']['case_keys'])
    return dict(plan=d.identity('notes/geobc_structured_TS_readout_v40_plan.json',hashed=True),terminal={k:d.identity(f/file,hashed=True) for k,file in [('summary','summary.json'),('status','status.json'),('external','external_terminal.json')]}),p


def baseline_binding():
    filename='notes/geobc_macro100_FNO9_physical5_v12_plan.json';p=d.read(filename);r=d.read(p['root_review']);f=d.path(p['output'])
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest'] and p['source_pins']==r['source_pins']
    a=d.read(f/'summary.json');e=d.read(f/'external_terminal.json')
    assert a==d.read(f/'status.json') and a['status']=='completed' and a['training_complete']
    assert a['closed'] and not a['active'] and e['closed'] and not e['active'] and e['exit_code']==0 and e['within_external_budget']
    assert all(d.exact_process_exited(e[k]) for k in ('child_process','guard_process'))
    assert a['execution']['run_process']==e['child_process'] and a['execution']['guard']['guard_process']==e['guard_process']
    assert a['execution']['scientific_digest']==e['scientific_digest']==p['scientific_digest']
    assert a['step']==a['optimizer_step_completions']==1500 and a['case_exposure_counts']=={k:5 for k in p['scientific']['TRAIN_keys']}
    assert a['actual_fresh_initial_model_sha256']==INITIAL and a['saved_Adam_steps_equal_cursor']
    complete_physical(a['endpoints']['1500']['full_TRAIN'],p['scientific']['TRAIN_keys'])
    return dict(plan=d.identity(filename,hashed=True),terminal={k:d.identity(f/file,hashed=True) for k,file in [('summary','summary.json'),('status','status.json'),('external','external_terminal.json')]}),p


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review'])
    assert p['kind']=='GeoBC_same_FNO9_parameters_R128_freshphysical100_short5_v42'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest'] and p['source_pins']==r['source_pins']
    assert all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert s['methods']==[METHOD] and s['seed']==2026 and s['train_CAD_count']==100
    assert s['grid_resolution']==128 and s['native_only_cache'] and s['R64_grid_reference_cases']==1
    assert s['additional_updates_per_arm']==1500 and s['additional_passes']==5
    assert s['initial_steps']=={METHOD:0} and s['endpoint_steps']=={METHOD:1500}
    assert s['expected_parameter_counts']=={METHOD:4526858} and s['fresh_initial_SHA256']=={METHOD:INITIAL}
    assert s['allowed_unused_parameters']==[] and s['memory_limit_bytes']==30500000000 and s['GPU_lock']=='runs/geobc_exploration/.coordinator.lock'
    assert [s[k] for k in ('cache_setup_budget_seconds','arm_setup_budget_seconds','training_budget_seconds_per_arm','TRAIN_readout_budget_seconds_per_arm','arm_budget_seconds','whole_budget_seconds')]==[600,120,1500,300,2100,2700]
    assert s['loss']=='full_original_node_physical_relative_squared_equal_U_S_case'
    assert s['optimizer']==dict(name='Adam',learning_rate=.001,weight_decay=0.,betas=[.9,.999],eps=1e-8,fused=True)
    assert s['precision']=='highest FP32; TF32 disabled' and s['DEV_FINAL_reads_allowed']==s['scaler_fits_allowed']==0
    assert s['teacher_prediction_distillation'] is False and s['inherited_teacher_passes']==0 and s['fresh5_physical_exposure_matched']
    prior,pp=predecessor_binding();assert s['predecessor_terminal']==prior
    base,bp=baseline_binding();assert s['baseline_R64_binding']==base
    assert all(p['source_pins'].get(k)==h for k,h in pp['source_pins'].items())
    assert s['dataset_identity']==pp['scientific']['dataset_identity']==bp['scientific']['dataset_identity']
    assert s['scaler']==pp['scientific']['scaler']==bp['scientific']['scaler']
    m=d.read(check_identity(s['dataset_identity']));keys=m['train_keys']['100']
    assert m['status']=='completed' and m['feasibility_pass'] and keys==s['TRAIN_keys'] and len(set(keys))==len(keys)==300
    assert all(m['case_roles'][k]=='TRAIN' for k in keys) and m['scaler']==s['scaler']
    from .macro_training import balanced_schedule
    order=balanced_schedule(keys,5,2026)
    assert d.digest(order)==s['full_order_digest']==s['arm_full_order_digests'][METHOD]==bp['scientific']['full_order_digest']
    proof=d.read(check_identity(s['CPU_fixture']))
    assert proof['status']=='actual_synthetic_CPU_raster_and_initialization_passed_not_FNO_forward_GPU_or_physical_training'
    assert proof['source']==d.identity('geobc/resolution_fno_v41.py',hashed=True)
    assert proof['full_fresh_initial_SHA256']==INITIAL and proof['full_initial_state_matches_R64_FNO9'] and proof['parameter_count']==4526858
    assert proof['operator_forward_calls']==proof['optimizer_updates']==proof['GT_PT_dataset_reads']==proof['GPU_calls']==0
    assert p['output']=='runs/geobc_exploration/resolution_FNO100_R128_freshphysical5_v42'
    return p,r,m,order
