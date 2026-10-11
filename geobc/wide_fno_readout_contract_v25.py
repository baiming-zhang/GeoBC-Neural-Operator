"""One fixed wider R64 FNO control, whole60 prediction barrier before knownGT60."""
from . import available_cad_pilot_data as d
from .global_hash_physical_readout_contract_v18 import check_identity, complete_physical

TRAIN = 'notes/geobc_wide_FNO64_100_physical5_v24_plan.json'


def training_terminal():
    from .wide_fno_training_contract_v24 import approved_contract
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
    assert summary['cached_physical_TRAIN_readout_reuses'] == 300
    assert max(summary['cache_setup_epoch_seconds'], summary['cache_setup_monotonic_seconds']) <= 600
    assert max(summary['elapsed_epoch_seconds'], summary['elapsed_monotonic_seconds'], external['elapsed_epoch_seconds'], external['elapsed_monotonic_seconds']) <= 1800
    assert max(summary[k] for k in ('peak_allocated_bytes', 'peak_reserved_bytes', 'peak_total_device_used_bytes_sampled')) <= 30500000000
    assert set(summary['methods']) == set(science['methods'])
    for method in science['methods']:
        arm = summary['methods'][method]
        assert arm == d.read(folder/method/'summary.json') == d.read(folder/method/'status.json')
        assert arm['status'] == 'completed' and arm['closed'] and not arm['active'] and arm['training_complete']
        assert arm['step'] == arm['optimizer_step_completions'] == arm['optimizer_step_attempts'] == len(order) == 1500
        assert not arm['optimizer_step_in_progress'] and not arm['possible_partial_optimizer_update']
        assert arm['case_exposure_counts'] == {k: 5 for k in science['TRAIN_keys']}
        assert arm['actual_largest_fullN_gradient_check']['passed'] and arm['actual_largest_fullN_gradient_check']['full300_physical_cache_qualified']
        assert arm['actual_shared_initial_encoder_head_equal_seeded_FNO']
        assert arm['saved_Adam_steps_equal_cursor'] and arm['cached_physical_TRAIN_readout_reuses'] == 300
        assert max(arm['setup_epoch_seconds'], arm['setup_monotonic_seconds']) <= 120
        assert max(arm['training_epoch_seconds'], arm['training_monotonic_seconds']) <= 900
        assert max(arm['elapsed_epoch_seconds'], arm['elapsed_monotonic_seconds']) <= 1500
        point = arm['endpoints']['1500']; complete_physical(point['full_TRAIN'], science['TRAIN_keys'])
        assert point['case_exposure_count'] == 5 and max(point['epoch_seconds'], point['monotonic_seconds']) <= 300
        assert d.relative(point['checkpoint_identity']['path']) == d.relative(folder/method/'step1500.pt')
        d.stat_matches(point['checkpoint_identity'])  # No PT body at metadata stage.
    binding = {kind: d.identity(folder/file, hashed=True) for kind, file in [('summary', 'summary.json'), ('status', 'status.json'), ('external', 'external_terminal.json')]}
    return binding, plan, summary, external, manifest


def approved_contract(plan_path):
    plan = d.read(plan_path); science = plan['scientific']; review = d.read(plan['root_review'])
    assert plan['kind'] == 'GeoBC_R64_W16FNO100_physical5_knownDEV60_v25'
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_execution_released']
    assert d.digest(science) == plan['scientific_digest'] == review['scientific_digest']
    assert plan['source_pins'] == review['source_pins'] and all(d.sha(p) == pin for p, pin in plan['source_pins'].items())
    assert science['memory_limit_bytes'] == 30500000000 and science['whole_budget_seconds'] == 1800
    assert science['GPU_lock'] == 'runs/geobc_exploration/.coordinator.lock'
    assert science['seed'] == 2026 and science['decoder_chunk_size'] == 32768
    assert science['GT_policy'] == 'All requested full-node predictions committed and revalidated before shared owned-knownDEV60 once'
    assert science['pool'] == 'known_development60_availability_conditional_not_FINAL'
    assert science['fixed_prediction_denominator'] == 60 and not science['FINAL_reuse_for_selection']
    assert plan['output'] == 'runs/geobc_exploration/wide_FNO64_100_knownDEV_physical5_v25'
    binding, train, summary, external, manifest = training_terminal()
    assert review['training_binding'] == binding and science['training_plan_identity'] == d.identity(TRAIN, hashed=True)
    assert science['training_SCI'] == train['scientific_digest']
    assert train['scientific']['dataset_identity'] == science['dataset_identity'] and train['scientific']['scaler'] == science['scaler'] == manifest['scaler']
    assert all(plan['source_pins'].get(p) == h for p, h in train['source_pins'].items())
    keys = science['case_keys']; assert keys == manifest['development_keys'] and len(keys) == len(set(keys)) == 60
    assert all(manifest['case_roles'][k] == 'DEV' for k in keys)
    assert len({k.split('/')[0] for k in keys}) == 20
    known = d.read('notes/geobc_exposure_readout_plan.json')['scientific']
    assert science['GT_manifest_identity'] == known['GT_manifest_identity'] and keys == known['case_keys']
    truths = d.read(check_identity(science['GT_manifest_identity']))
    assert truths['status'] == 'completed' and truths['source_GT_parses'] == 20 and set(truths['records']) == set(keys)
    for key in keys:
        truth = truths['records'][key]
        assert truth['all_original_nodes'] and truth['signed_stress_unchanged']
        assert truth['physical']['shape'] == [manifest['records'][key]['num_nodes'], 4]
    names = ['R64_FNO_W16_1500']
    candidates = science['candidates']
    assert [(c['name'], c['method'], c['family'], c['step']) for c in candidates] == [(n, m, 'FNO_R64_W16', 1500) for n, m in zip(names, train['scientific']['methods'])]
    qualified = {}
    for candidate in candidates:
        method = candidate['method']; arm = summary['methods'][method]
        qualified[candidate['name']] = dict(plan=train, summary=summary, arm=arm, external=external,
                                            checkpoint=arm['endpoints']['1500']['checkpoint_identity'])
    baseline_binding = science['baseline_development_binding']
    for item in baseline_binding.values(): check_identity(item)
    baseline = d.read(baseline_binding['summary']['path']); be = d.read(baseline_binding['external']['path'])
    assert baseline == d.read(baseline_binding['status']['path'])
    assert baseline['status'] == 'completed' and baseline['closed'] and not baseline['active'] and baseline['protocol_qualified']
    assert be['exit_code'] == 0 and be['within_external_budget'] and be['closed'] and not be['active']
    assert all(d.exact_process_exited(be[k]) for k in ('child_process', 'guard_process'))
    assert baseline['execution']['run_process'] == be['child_process'] and baseline['execution']['guard']['guard_process'] == be['guard_process']
    assert baseline['execution']['scientific_digest'] == be['scientific_digest'] == science['baseline_development_SCI']
    assert baseline['committed_finite_predictions'] == baseline['owned_GT_array_reads'] == 60
    assert baseline['raw_GT_reads'] == baseline['FINAL_numeric_reads'] == baseline['training_updates'] == baseline['scaler_fits'] == 0
    complete_physical(baseline['candidates']['FNO100_PHYSICAL_1500']['full60'], keys)
    assert science['DEV_gate'] == dict(primary_absolute_improvement_at_least=.03, maximum_either_field_absolute_deterioration=.02, fixed_denominator=60)
    from .wide_fno_training_contract_v24 import predecessor_terminal
    assert science['compact_development_binding'] == predecessor_terminal()
    return plan, review, manifest, truths, qualified
