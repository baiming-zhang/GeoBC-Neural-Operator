"""Fixed teacher and two node-latent students; all180 before knownGT60."""
from . import available_cad_pilot_data as d
from .global_hash_physical_readout_contract_v18 import check_identity, complete_physical

TRAIN = 'notes/geobc_node_latent_teacher_v36_plan.json'


def training_terminal():
    from .node_latent_training_contract_v36 import approved_contract
    plan, _, manifest, order = approved_contract(TRAIN); science = plan['scientific']; folder = d.path(plan['output'])
    summary = d.read(folder/'summary.json'); status = d.read(folder/'status.json'); external = d.read(folder/'external_terminal.json')
    assert summary == status and summary['status'] == 'completed' and summary['training_complete']
    assert summary['closed'] and not summary['active'] and external['closed'] and not external['active']
    assert external['exit_code'] == 0 and external['within_external_budget'] and not external['external_timeout']
    assert all(d.exact_process_exited(external[k]) for k in ('child_process', 'guard_process'))
    assert summary['execution']['run_process'] == external['child_process']
    assert summary['execution']['guard']['guard_process'] == external['guard_process']
    assert summary['execution']['boot_id'] == external['boot_id']
    assert summary['execution']['scientific_digest'] == external['scientific_digest'] == plan['scientific_digest']
    assert summary['raw_GT_reads'] == summary['DEV_FINAL_reads'] == summary['scaler_fits'] == 0
    assert summary['cache_load_complete'] and summary['shared_cache_load_attempts'] == 1
    assert summary['cache_receipt']['physical_TRAIN_loaded_cases'] == summary['cache_receipt']['physical_TRAIN_target_reads'] == 300
    assert summary['cached_physical_TRAIN_readout_reuses'] == 600
    assert max(summary['cache_setup_epoch_seconds'], summary['cache_setup_monotonic_seconds']) <= 600
    assert max(summary['elapsed_epoch_seconds'], summary['elapsed_monotonic_seconds'], external['elapsed_epoch_seconds'], external['elapsed_monotonic_seconds']) <= 3600
    assert max(summary[k] for k in ('peak_allocated_bytes', 'peak_reserved_bytes', 'peak_total_device_used_bytes_sampled')) <= 30500000000
    assert set(summary['methods']) == set(science['methods'])
    for method in science['methods']:
        arm = summary['methods'][method]
        assert arm == d.read(folder/method/'summary.json') == d.read(folder/method/'status.json')
        assert arm['status'] == 'completed' and arm['closed'] and not arm['active'] and arm['training_complete']
        assert arm['step'] == arm['optimizer_step_completions'] == arm['optimizer_step_attempts'] == science['endpoint_steps'][method]
        assert arm['additional_optimizer_step_attempts'] == arm['additional_optimizer_step_completions'] == len(order) == 1500
        assert arm['parent_optimizer_steps'] == science['initial_steps'][method]
        assert not arm['optimizer_step_in_progress'] and not arm['possible_partial_optimizer_update']
        assert arm['case_exposure_counts'] == {k: science['endpoint_steps'][method]//300 for k in science['TRAIN_keys']}
        assert arm['actual_largest_fullN_gradient_check']['passed'] and arm['actual_largest_fullN_gradient_check']['full300_physical_cache_qualified']
        assert arm['actual_fresh_initial_state_matches_original_control']
        assert arm['actual_transfer']['operator_unchanged'] and arm['actual_transfer']['encoder_head_equal_teacher'] and arm['actual_reset_Adam_empty']
        assert arm['actual_transfer']['teacher_state_digest'] == summary['teacher_loaded_model_sha256']
        assert arm['trainable_parameter_groups'] == ['operator'] and arm['frozen_parameter_groups'] == ['encoder','head']
        assert arm['initial_fullTRAIN_teacher_node_latent']['all_cases_complete'] and arm['initial_fullTRAIN_teacher_node_latent']['fixed_denominator'] == 300
        assert arm['inherited_teacher_case_exposure_counts'] == {k: 5 for k in science['TRAIN_keys']}
        assert arm['saved_Adam_steps_equal_cursor'] and arm['cached_physical_TRAIN_readout_reuses'] == 300
        assert max(arm['setup_epoch_seconds'], arm['setup_monotonic_seconds']) <= 120
        assert max(arm['training_epoch_seconds'], arm['training_monotonic_seconds']) <= 900
        assert max(arm['elapsed_epoch_seconds'], arm['elapsed_monotonic_seconds']) <= 1500
        endpoint_step = science['endpoint_steps'][method]; point = arm['endpoints'][str(endpoint_step)]; complete_physical(point['full_TRAIN'], science['TRAIN_keys'])
        assert point['full_TRAIN_teacher_node_latent']['all_cases_complete'] and point['full_TRAIN_teacher_node_latent']['fixed_denominator'] == 300
        assert point['additional_pass_kind'] == 'teacher_latent_target_not_FEM'
        assert point['case_exposure_count'] == endpoint_step//300 and point['additional_passes'] == point['teacher_derived_initialization_passes'] == 5 and max(point['epoch_seconds'], point['monotonic_seconds']) <= 300
        assert d.relative(point['checkpoint_identity']['path']) == d.relative(folder/method/('step'+str(endpoint_step)+'.pt'))
        d.stat_matches(point['checkpoint_identity'])  # No PT body at metadata stage.
    assert summary['teacher_latent_cache_complete'] and summary['teacher_forward_attempts'] == 300
    assert summary['teacher_physical_forward_attempts'] == summary['teacher_physical_forward_cases'] == 300
    assert summary['FEM_used_for_optimizer_updates'] is False
    complete_physical(summary['full_TRAIN_teacher_to_original_FEM'], science['TRAIN_keys'])
    assert summary['teacher_checkpoint_load_attempts'] == summary['teacher_checkpoint_loads'] == summary['teacher_checkpoint_hash_attempts'] == 1
    assert summary['inherited_teacher_cost']['prior_passes'] == 5
    binding = {kind: d.identity(folder/file, hashed=True) for kind, file in [('summary', 'summary.json'), ('status', 'status.json'), ('external', 'external_terminal.json')]}
    return binding, plan, summary, external, manifest



def approved_contract(plan_path):
    plan=d.read(plan_path);s=plan['scientific'];r=d.read(plan['root_review'])
    assert plan['kind']=='GeoBC_R16_node_latent_teacher_knownDEV60_v37'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s)==plan['scientific_digest']==r['scientific_digest']
    assert plan['source_pins']==r['source_pins'] and all(d.sha(k)==h for k,h in plan['source_pins'].items())
    assert s['memory_limit_bytes']==30500000000 and s['whole_budget_seconds']==1800
    assert s['GPU_lock']=='runs/geobc_exploration/.coordinator.lock' and s['seed']==2026 and s['decoder_chunk_size']==32768
    assert s['GT_policy']=='All requested full-node predictions committed and revalidated before shared owned-knownDEV60 once'
    assert s['pool']=='known_development60_availability_conditional_not_FINAL'
    assert s['fixed_prediction_denominator']==180 and not s['FINAL_reuse_for_selection']
    assert plan['output']=='runs/geobc_exploration/node_latent_teacher100_knownDEV_short5_v37'
    binding,train,summary,external,manifest=training_terminal()
    assert r['training_binding']==binding and s['training_plan_identity']==d.identity(TRAIN,hashed=True)
    assert s['training_SCI']==train['scientific_digest'] and s['dataset_identity']==train['scientific']['dataset_identity']
    assert s['scaler']==train['scientific']['scaler']==manifest['scaler']
    assert all(plan['source_pins'].get(k)==h for k,h in train['source_pins'].items())
    keys=s['case_keys'];assert keys==manifest['development_keys'] and len(set(keys))==len(keys)==60
    assert all(manifest['case_roles'][k]=='DEV' for k in keys) and len({k.split('/')[0] for k in keys})==20
    known=d.read('notes/geobc_exposure_readout_plan.json')['scientific']
    assert s['GT_manifest_identity']==known['GT_manifest_identity'] and keys==known['case_keys']
    truths=d.read(check_identity(s['GT_manifest_identity']))
    assert truths['status']=='completed' and truths['source_GT_parses']==20 and set(truths['records'])==set(keys)
    for key in keys:
        truth=truths['records'][key];assert truth['all_original_nodes'] and truth['signed_stress_unchanged']
        assert truth['physical']['shape']==[manifest['records'][key]['num_nodes'],4]
    expected=[('TRAIN_TEACHER_R16_FNO16_1500','FNO16_control','R16_FNO16_control',True),('R16_TRANSOLVER_LATENT_1500','transolver_latent_teacher','R16_transolver_latent_teacher',False),('R16_FNO16_LATENT_1500','FNO16_latent_teacher','R16_FNO16_latent_teacher',False)]
    assert [(c['name'],c['method'],c['family'],c['is_teacher']) for c in s['candidates']]==expected
    assert all(c['step']==1500 for c in s['candidates'])
    assert s['teacher_checkpoint']==train['scientific']['teacher_checkpoint'] and s['teacher_SCI']==train['scientific']['teacher_SCI']
    assert s['teacher_target_policy']=='Complete detached R16 teacher latent16 interpolated at all original nodes; teacher basis is not physical GT'
    assert s['learning_diagnostic']==dict(TRAIN_mean_teacher_latent_relative_L2_ratio_at_most=.5,TRAIN_mean_teacher_latent_relative_L2_at_most=.25,automatic_continuation=False)
    qualified={}
    for c in s['candidates']:
        if c['is_teacher']:
            tb=train['scientific']['teacher_binding'];tp=d.read(tb['plan']['path']);ts=d.read(tb['terminal']['summary']['path']);te=d.read(tb['terminal']['external']['path'])
            qualified[c['name']]=dict(plan=tp,summary=ts,arm=ts['methods']['FNO16_control'],external=te,checkpoint=s['teacher_checkpoint'])
        else:
            arm=summary['methods'][c['method']];qualified[c['name']]=dict(plan=train,summary=summary,arm=arm,external=external,checkpoint=arm['endpoints']['1500']['checkpoint_identity'])
    assert s['prior_development_binding']==train['scientific']['predecessor_terminal']
    prior=d.read(check_identity(s['prior_development_binding']['terminal']['summary']))
    for row in prior['candidates'].values():complete_physical(row['full60'],keys)
    assert not s['fresh5_fair_comparison_claimed'] and s['teacher_inherited_passes']==5
    return plan,r,manifest,truths,qualified
