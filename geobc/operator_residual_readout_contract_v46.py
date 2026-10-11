"""Actual two-residual terminal handoff and knownDEV120 prediction barrier."""
from . import available_cad_pilot_data as d
from .global_hash_physical_readout_contract_v18 import check_identity, complete_physical

TRAIN = 'notes/geobc_operator_residual_v45_plan.json'
METHODS = ['FNO5_MLP_RESIDUAL','FNO5_TCNN_RESIDUAL']


def training_terminal():
    p = d.read(TRAIN); r = d.read(p['root_review']); s = p['scientific']; f = d.path(p['output'])
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['kind'] == 'GeoBC_frozen_FNO5_two_zero_residual_decoders_physical5_v45'
    assert d.digest(s) == p['scientific_digest'] == r['scientific_digest'] and p['source_pins'] == r['source_pins']
    assert all(d.sha(k) == h for k,h in p['source_pins'].items())
    m = d.read(check_identity(s['dataset_identity'])); a = d.read(f/'summary.json'); e = d.read(f/'external_terminal.json')
    assert a == d.read(f/'status.json') and a['status'] == 'completed' and a['training_complete']
    assert a['closed'] and not a['active'] and e['closed'] and not e['active']
    assert e['exit_code'] == 0 and e['within_external_budget'] and not e['external_timeout']
    assert e['source_pins'] == p['source_pins'] and all(d.exact_process_exited(e[k]) for k in ('child_process','guard_process'))
    assert a['execution']['run_process'] == e['child_process'] and a['execution']['guard']['guard_process'] == e['guard_process']
    assert a['execution']['boot_id'] == e['boot_id'] and a['execution']['scientific_digest'] == e['scientific_digest'] == p['scientific_digest']
    assert max(a['elapsed_epoch_seconds'],a['elapsed_monotonic_seconds'],e['elapsed_epoch_seconds'],e['elapsed_monotonic_seconds']) <= 3600
    assert max(a[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled')) <= 30500000000
    assert a['raw_GT_reads'] == a['DEV_FINAL_reads'] == a['scaler_fits'] == 0
    assert a['teacher_PT_numeric_loads'] == a['teacher_PT_hash_reads'] == a['shared_cache_load_attempts'] == 1
    assert a['actual_teacher_checkpoint_qualification']['passed'] and a['cache_load_complete']
    assert a['teacher_feature_forward_attempts'] == a['teacher_feature_forward_completions'] == a['released_original_grid_cases'] == 300
    assert a['cache_receipt']['physical_TRAIN_target_read_attempts'] == a['cache_receipt']['physical_TRAIN_target_reads'] == a['cache_receipt']['physical_TRAIN_loaded_cases'] == 300
    assert a['cached_physical_TRAIN_readout_reuses'] == 600 and a['released_original_grid_bytes'] > 0
    assert max(a['cache_teacher_setup_epoch_seconds'],a['cache_teacher_setup_monotonic_seconds']) <= 600
    assert set(a['methods']) == set(METHODS) and s['methods'] == METHODS
    for method in METHODS:
        arm = a['methods'][method]
        assert arm == d.read(f/method/'summary.json') == d.read(f/method/'status.json')
        assert arm['status'] == 'completed' and arm['closed'] and not arm['active'] and arm['training_complete']
        assert arm['step'] == arm['optimizer_step_attempts'] == arm['optimizer_step_completions'] == 1500
        assert not arm['optimizer_step_in_progress'] and not arm['possible_partial_optimizer_update']
        assert arm['case_exposure_counts'] == {k:5 for k in s['TRAIN_keys']}
        assert arm['inherited_teacher_passes'] == 5 and arm['frozen_reference_passes'] == 0
        assert arm['saved_Adam_steps_equal_cursor'] and arm['actual_frozen_states_unchanged_after_training']
        assert arm['frozen_teacher_digest'] == a['actual_teacher_trained_state_digest']
        assert arm['frozen_reference_digest'] == arm['decoder_initial_state_digest']
        q = arm['actual_largest_fullN_gradient_check']
        assert q['passed'] and q['initial_prediction_exact_teacher'] and q['checked_with_training_autograd_enabled']
        assert q['frozen_gradients_none'] and q['frozen_states_unchanged'] and q['full300_physical_and_frozen_feature_cache_qualified']
        assert q['nodes'] == max(m['records'][k]['num_nodes'] for k in s['TRAIN_keys']) and q['mutable_decoder_gradient_L2'] > 0
        assert arm['model_profile']['teacher_weights_frozen'] and arm['model_profile']['reference_weights_frozen']
        assert arm['model_profile']['backend'] == ('torch_FP32_MLP' if method == METHODS[0] else 'tcnn_fully_fused')
        assert max(arm['setup_epoch_seconds'],arm['setup_monotonic_seconds']) <= 120
        assert max(arm['training_epoch_seconds'],arm['training_monotonic_seconds']) <= 900
        assert max(arm['elapsed_epoch_seconds'],arm['elapsed_monotonic_seconds']) <= 1500
        point = arm['endpoints']['1500']; complete_physical(point['full_TRAIN'],s['TRAIN_keys'])
        assert point['decoder_passes'] == point['teacher_passes'] == 5 and point['reference_passes'] == 0
        assert max(point['epoch_seconds'],point['monotonic_seconds']) <= 300
        assert d.relative(point['checkpoint_identity']['path']) == d.relative(f/method/'step1500.pt')
        d.stat_matches(point['checkpoint_identity'])
    binding = {k:d.identity(f/file,hashed=True) for k,file in [('summary','summary.json'),('status','status.json'),('external','external_terminal.json')]}
    return binding,p,a,e,m


def approved_contract(plan_path):
    p = d.read(plan_path); s = p['scientific']; r = d.read(p['root_review'])
    assert p['kind'] == 'GeoBC_frozen_FNO5_two_residuals_knownDEV60_v46'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s) == p['scientific_digest'] == r['scientific_digest'] and p['source_pins'] == r['source_pins']
    assert all(d.sha(k) == h for k,h in p['source_pins'].items())
    assert s['memory_limit_bytes'] == 30500000000 and s['whole_budget_seconds'] == 1800
    assert s['GPU_lock'] == 'runs/geobc_exploration/.coordinator.lock' and s['seed'] == 2026 and s['decoder_chunk_size'] == 32768
    assert s['GT_policy'] == 'All requested full-node predictions committed and revalidated before shared owned-knownDEV60 once'
    assert s['pool'] == 'known_development60_availability_conditional_not_FINAL' and not s['FINAL_reuse_for_selection']
    assert s['fixed_prediction_denominator'] == 120 and not s['automatic_continuation']
    assert p['output'] == 'runs/geobc_exploration/frozen_FNO5_two_residual_knownDEV_v46'
    binding,t,a,e,m = training_terminal()
    assert r['training_binding'] == binding and s['training_plan_identity'] == d.identity(TRAIN,hashed=True) and s['training_SCI'] == t['scientific_digest']
    assert all(p['source_pins'].get(k) == h for k,h in t['source_pins'].items())
    assert s['dataset_identity'] == t['scientific']['dataset_identity'] and s['scaler'] == t['scientific']['scaler'] == m['scaler']
    keys = s['case_keys']; assert keys == m['development_keys'] and len(keys) == len(set(keys)) == 60
    assert all(m['case_roles'][k] == 'DEV' for k in keys) and len({k.split('/')[0] for k in keys}) == 20
    known = d.read('notes/geobc_exposure_readout_plan.json')['scientific']
    assert s['GT_manifest_identity'] == known['GT_manifest_identity'] and keys == known['case_keys']
    truths = d.read(check_identity(s['GT_manifest_identity']))
    assert truths['status'] == 'completed' and truths['source_GT_parses'] == 20 and set(truths['records']) == set(keys)
    for key in keys:
        truth = truths['records'][key]
        assert truth['all_original_nodes'] and truth['signed_stress_unchanged'] and truth['physical']['shape'] == [m['records'][key]['num_nodes'],4]
    assert s['candidates'] == [dict(name=n,method=n,family=n,step=1500) for n in METHODS]
    qualified = {n:dict(plan=t,summary=a,arm=a['methods'][n],external=e,checkpoint=a['methods'][n]['endpoints']['1500']['checkpoint_identity']) for n in METHODS}
    for reference in s['scalar_references']:
        for item in reference['binding'].values(): check_identity(item)
        old = d.read(reference['binding']['summary']['path']); oe = d.read(reference['binding']['external']['path'])
        assert old == d.read(reference['binding']['status']['path']) and old['status'] == 'completed' and old['protocol_qualified']
        assert old['closed'] and not old['active'] and oe['closed'] and not oe['active'] and oe['exit_code'] == 0 and oe['within_external_budget']
        assert old['execution']['scientific_digest'] == oe['scientific_digest'] == reference['SCI']
        complete_physical(old['candidates'][reference['name']]['full60'],keys)
    assert [x['name'] for x in s['scalar_references']] == ['FNO100_PHYSICAL_1500','FNO100_PHYSICAL_3000']
    assert s['DEV_gate'] == dict(primary_absolute_improvement_at_least=.03,stress_absolute_improvement_at_least=.03,maximum_either_field_absolute_deterioration=.02,fixed_denominator=60)
    return p,r,m,truths,qualified
