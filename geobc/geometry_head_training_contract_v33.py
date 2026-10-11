"""Three initialization/continuation controls; flat current-terminal metadata."""
from . import available_cad_pilot_data as d
from .global_hash_physical_readout_contract_v18 import complete_physical

METHODS = ['transolver_headwarm', 'FNO16_headwarm_reset', 'FNO16_continue']


def terminal_binding(plan_path, development):
    plan=d.read(plan_path);review=d.read(plan['root_review']);folder=d.path(plan['output'])
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_execution_released']
    assert d.digest(plan['scientific'])==plan['scientific_digest']==review['scientific_digest']
    assert plan['source_pins']==review['source_pins']
    summary=d.read(folder/'summary.json');external=d.read(folder/'external_terminal.json')
    assert summary==d.read(folder/'status.json') and summary['status']=='completed'
    assert summary['closed'] and not summary['active'] and external['closed'] and not external['active']
    assert external['exit_code']==0 and external['within_external_budget'] and not external['external_timeout']
    assert all(d.exact_process_exited(external[k]) for k in ('child_process','guard_process'))
    assert summary['execution']['run_process']==external['child_process']
    assert summary['execution']['guard']['guard_process']==external['guard_process']
    assert summary['execution']['scientific_digest']==external['scientific_digest']==plan['scientific_digest']
    assert external['source_pins']==plan['source_pins']
    if development:
        assert summary['protocol_qualified'] and summary['committed_finite_predictions']==120
        assert summary['owned_GT_array_reads']==summary['owned_GT_read_attempts']==60
        assert summary['raw_GT_reads']==summary['FINAL_numeric_reads']==summary['training_updates']==summary['scaler_fits']==0
        for row in summary['candidates'].values():complete_physical(row['full60'],plan['scientific']['case_keys'])
    else:
        assert summary['training_complete'] and summary['cache_load_complete']
        assert summary['raw_GT_reads']==summary['DEV_FINAL_reads']==summary['scaler_fits']==0
        assert summary['cache_receipt']['physical_TRAIN_loaded_cases']==summary['cache_receipt']['physical_TRAIN_target_reads']==300
        arm=summary['methods']['FNO16_control']
        assert arm==d.read(folder/'FNO16_control'/'summary.json')==d.read(folder/'FNO16_control'/'status.json')
        assert arm['status']=='completed' and arm['step']==arm['optimizer_step_attempts']==arm['optimizer_step_completions']==1500
        assert arm['case_exposure_counts']=={k:5 for k in plan['scientific']['TRAIN_keys']}
        assert arm['actual_largest_fullN_gradient_check']['passed'] and arm['saved_Adam_steps_equal_cursor']
        complete_physical(arm['endpoints']['1500']['full_TRAIN'],plan['scientific']['TRAIN_keys'])
        d.stat_matches(arm['endpoints']['1500']['checkpoint_identity'])
    assert max(summary['elapsed_epoch_seconds'],summary['elapsed_monotonic_seconds'],external['elapsed_epoch_seconds'],external['elapsed_monotonic_seconds'])<=plan['scientific']['whole_budget_seconds']
    assert max(summary[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))<=30500000000
    binding=dict(plan=d.identity(plan_path,hashed=True),terminal={k:d.identity(folder/file,hashed=True) for k,file in [('summary','summary.json'),('status','status.json'),('external','external_terminal.json')]})
    return binding,plan,summary,external


def approved_contract(plan_path):
    plan=d.read(plan_path);s=plan['scientific'];r=d.read(plan['root_review'])
    assert plan['kind']=='GeoBC_R16_three_geometry_head_transfer_same100_additional5_v33'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s)==plan['scientific_digest']==r['scientific_digest']
    assert plan['source_pins']==r['source_pins'] and all(d.sha(p)==h for p,h in plan['source_pins'].items())
    assert s['methods']==METHODS and s['additional_updates_per_arm']==1500 and s['additional_passes']==5
    assert s['train_CAD_count']==100 and s['seed']==2026 and s['decoder_chunk_size']==32768
    assert s['memory_limit_bytes']==30500000000 and s['GPU_lock']=='runs/geobc_exploration/.coordinator.lock'
    assert [s[k] for k in ('cache_setup_budget_seconds','arm_setup_budget_seconds','training_budget_seconds_per_arm','TRAIN_readout_budget_seconds_per_arm','arm_budget_seconds','whole_budget_seconds')]==[600,120,900,300,1500,3600]
    assert s['loss']=='full_original_node_physical_relative_squared_equal_U_S_case'
    assert s['optimizer']==dict(name='Adam',learning_rate=.001,weight_decay=0.,betas=[.9,.999],eps=1e-8,fused=True)
    assert s['precision']=='highest FP32; TF32 disabled' and s['DEV_FINAL_reads_allowed']==s['scaler_fits_allowed']==0
    assert s['teacher_prediction_distillation'] is False and s['teacher_inherited_passes']==5
    prior,_,_,_=terminal_binding('notes/geobc_response_gain_v31_plan.json',True)
    teacher,tp,ts,te=terminal_binding('notes/geobc_coarse_FNO100_physical5_v22_plan.json',False)
    assert s['predecessor_terminal']==prior and s['teacher_binding']==teacher
    assert all(plan['source_pins'].get(k)==h for k,h in d.read(prior['plan']['path'])['source_pins'].items())
    assert all(plan['source_pins'].get(k)==h for k,h in tp['source_pins'].items())
    arm=ts['methods']['FNO16_control']
    assert s['teacher_checkpoint']==arm['endpoints']['1500']['checkpoint_identity']
    assert s['teacher_SCI']==tp['scientific_digest'] and s['teacher_prior_full_order_digest']==tp['scientific']['full_order_digest']
    assert s['dataset_identity']==tp['scientific']['dataset_identity'] and s['scaler']==tp['scientific']['scaler']
    assert d.identity(s['dataset_identity']['path'],hashed=True)==s['dataset_identity']
    manifest=d.read(s['dataset_identity']['path']);keys=manifest['train_keys']['100']
    assert manifest['status']=='completed' and manifest['feasibility_pass'] and keys==s['TRAIN_keys'] and len(set(keys))==len(keys)==300
    assert all(manifest['case_roles'][k]=='TRAIN' for k in keys) and manifest['scaler']==s['scaler']
    from .macro_training import balanced_schedule
    full=balanced_schedule(keys,10,2026);order=full[1500:]
    assert d.digest(full[:1500])==s['teacher_prior_full_order_digest']
    assert d.digest(order)==s['marginal_order_digest'] and s['same_marginal_order_all_arms']
    assert s['arm_full_order_digests']=={m:d.digest(full if m=='FNO16_continue' else order) for m in METHODS}
    assert s['initial_steps']==dict(transolver_headwarm=0,FNO16_headwarm_reset=0,FNO16_continue=1500)
    assert s['endpoint_steps']==dict(transolver_headwarm=1500,FNO16_headwarm_reset=1500,FNO16_continue=3000)
    assert s['expected_parameter_counts']==dict(transolver_headwarm=1882172,FNO16_headwarm_reset=4244244,FNO16_continue=4244244)
    assert not s['fresh5_fair_comparison_claimed'] and not s['pure_architecture_causality_claimed']
    item=s['CPU_fixture'];assert d.identity(item['path'],hashed=True)==item;proof=d.read(item['path'])
    assert proof['status']=='actual_synthetic_geometry_head_transfer_CPU_passed_not_GPU' and proof['synthetic_teacher_not_trained']
    assert proof['real_numeric_GT_reads']==proof['PT_reads']==proof['GPU_calls']==proof['optimizer_updates']==0
    assert proof['synthetic_nodes']==33337 and proof['chunks']==[32768,569]
    assert proof['source']==d.identity('geobc/geometry_head_transfer_v32.py',hashed=True)
    assert all(v['all_named_gradients_close'] and v['transfer']['operator_unchanged'] and v['transfer']['encoder_head_equal_teacher'] for v in proof['results'].values())
    assert plan['output']=='runs/geobc_exploration/geometry_head_transfer100_additional5_v33'
    return plan,r,manifest,order
