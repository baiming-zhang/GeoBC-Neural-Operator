"""Metadata-only contract for future selected macro known-DEV readouts.

No training plan or numerical evaluation is released by this module. The fixed
known development pool remains distinct from the reserved FINAL pool.
"""
import math
import re
from . import available_cad_pilot_data as d

KIND = 'GeoBC_macro_short_knownDEV60_v2'
REQUIRED_SOURCES = (
    'geobc/macro_screen_readout_contract_v2.py', 'geobc/macro_readout_protocol.py',
    'tools/evaluate_geobc_macro_screen_v2.py', 'tools/run_geobc_macro_screen_readout_guard_v2.py',
    'geobc/prospective_dataset_v2.py', 'geobc/prospective_dataset.py',
    'geobc/macro_training.py', 'geobc/macro_screen_contract_v2.py',
    'geobc/available_cad_pilot_eval.py', 'geobc/available_cad_pilot_data.py',
    'geobc/coverage_comparison_train.py', 'geobc/benchmark.py')


def check_identity(item):
    actual = d.identity(item['path'], hashed=True)
    # Historical owned-GT receipts use absolute WSL paths; new receipts use
    # project-relative paths. Compare canonical location without changing them.
    assert d.relative(actual['path']) == d.relative(item['path'])
    assert all(actual[k] == item[k] for k in ('bytes', 'mtime_ns', 'sha256'))
    return d.path(item['path'])


def complete_physical(metric, keys):
    assert metric['all_cases_complete'] and metric['fixed_denominator'] == len(keys)
    assert set(metric['per_case']) == set(keys)
    fields = ('u_relative_l2', 'stress_relative_l2', 'stress_hotspot_relative_l2')
    assert all(row is not None and all(type(row.get(k)) in (int, float)
        and math.isfinite(row[k]) and row[k] >= 0 for k in fields)
        for row in metric['per_case'].values()), 'Complete finite U/S/hot denominators required'


def approved_contract(plan_path):
    plan = d.read(plan_path); science = plan['scientific']; review = d.read(plan['root_review'])
    assert plan['kind'] == KIND
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_execution_released']
    assert review['scientific_digest'] == plan['scientific_digest'] == d.digest(science)
    assert review['source_pins'] == plan['source_pins']
    assert set(REQUIRED_SOURCES) <= set(plan['source_pins'])
    assert all(d.sha(p) == h for p, h in plan['source_pins'].items())
    assert science['GPU_lock'] == 'runs/geobc_exploration/.coordinator.lock'
    assert science['memory_limit_bytes'] == 30500000000
    assert type(science['whole_budget_seconds']) is int and science['whole_budget_seconds'] > 0
    assert science['seed'] == 2026 and science['decoder_chunk_size'] == 32768
    assert science['GT_policy'] == 'All requested full-node predictions committed and revalidated before shared owned-knownDEV60 once'
    assert science['pool'] == 'known_development60_availability_conditional_not_FINAL'
    assert d.relative(plan['output']).startswith('runs/geobc_exploration/macro_DEV_')
    assert d.relative(science['dataset_identity']['path']) == 'data/geobc_prospective_dataset_v1/manifest.json'
    manifest = d.read(check_identity(science['dataset_identity']))
    assert manifest['status'] == 'completed' and manifest['feasibility_pass']
    keys = science['case_keys']
    assert keys == manifest['development_keys'] and len(keys) == len(set(keys)) == 60
    cads = {k.split('/')[0] for k in keys}
    assert len(cads) == 20 and set(keys) == {c+'/'+load for c in cads for load in ('ver', 'hor', 'tor')}
    assert all(manifest['case_roles'][k] == 'DEV' for k in keys)
    assert science['scaler'] == manifest['scaler']
    known = d.read('notes/geobc_exposure_readout_plan.json')['scientific']
    assert keys == known['case_keys'] and science['GT_manifest_identity'] == known['GT_manifest_identity']
    truths = d.read(check_identity(science['GT_manifest_identity']))  # JSON only; no numerical arrays.
    assert truths['status'] == 'completed' and truths['source_GT_parses'] == 20 and set(truths['records']) == set(keys)
    for key in keys:
        truth = truths['records'][key]
        assert truth['all_original_nodes'] and truth['signed_stress_unchanged']
        assert truth['physical']['shape'] == [manifest['records'][key]['num_nodes'], 4]
    candidates = science['candidates']; names = [c['name'] for c in candidates]
    assert names and len(names) == len(set(names))
    assert all(re.fullmatch(r'[A-Za-z0-9_-]+', n) for n in names)
    assert set(review['training_bindings']) == set(names)
    assert science['fixed_prediction_denominator'] == len(names)*60
    from .macro_screen_contract_v2 import approved_contract as training_contract
    qualified = {}
    for candidate in candidates:
        name = candidate['name']; train_path = check_identity(candidate['training_plan_identity'])
        train, _, dataset, order = training_contract(d.relative(train_path))
        ts = train['scientific']; binding = review['training_bindings'][name]; folder = d.path(train['output'])
        for kind, file in (('summary', 'summary.json'), ('external', 'external_terminal.json'), ('status', 'status.json')):
            assert d.relative(binding[kind]['path']) == d.relative(folder/file)
            check_identity(binding[kind])
        summary = d.read(binding['summary']['path']); external = d.read(binding['external']['path'])
        status = d.read(binding['status']['path'])
        assert summary == status and summary['status'] == 'completed' and summary['training_complete']
        assert summary['closed'] and not summary['active'] and external['closed'] and not external['active']
        assert external['within_external_budget'] and external['exit_code'] == 0 and not external['external_timeout']
        assert all(d.exact_process_exited(external[k]) for k in ('guard_process', 'child_process'))
        assert summary['execution']['run_process'] == external['child_process']
        assert summary['execution']['guard']['guard_process'] == external['guard_process']
        assert summary['execution']['scientific_digest'] == external['scientific_digest'] == train['scientific_digest']
        assert summary['execution']['boot_id'] == external['boot_id']
        assert summary['raw_GT_reads'] == summary['DEV_FINAL_reads'] == summary['scaler_fits'] == 0
        assert summary['cache_load_complete'] and summary['actual_largest_fullN_gradient_check']['passed']
        assert not summary['possible_partial_optimizer_update'] and not summary['optimizer_step_in_progress']
        assert summary['step'] == summary['optimizer_step_completions'] == summary['optimizer_step_attempts'] == len(order)
        assert summary['case_exposure_counts'] == {k: max(ts['endpoint_passes']) for k in ts['TRAIN_keys']}
        assert max(summary['training_epoch_seconds'], summary['training_monotonic_seconds']) <= ts['training_budget_seconds']
        assert max(summary['cache_setup_epoch_seconds'], summary['cache_setup_monotonic_seconds']) <= ts['cache_setup_budget_seconds']
        assert max(summary['elapsed_epoch_seconds'], summary['elapsed_monotonic_seconds'], external['elapsed_epoch_seconds'], external['elapsed_monotonic_seconds']) <= ts['whole_budget_seconds']
        assert max(summary[k] for k in ('peak_allocated_bytes', 'peak_reserved_bytes', 'peak_total_device_used_bytes_sampled')) <= ts['memory_limit_bytes']
        assert ts['dataset_identity'] == science['dataset_identity'] and dataset == manifest and ts['scaler'] == science['scaler']
        assert all(plan['source_pins'].get(p) == h for p, h in train['source_pins'].items())
        step = candidate['step']; assert type(step) is int and step in [len(ts['TRAIN_keys'])*p for p in ts['endpoint_passes']]
        point = summary['endpoints'][str(step)]
        complete_physical(point['full_TRAIN'], ts['TRAIN_keys'])
        assert point['case_exposure_count'] == step//len(ts['TRAIN_keys'])
        assert max(point['epoch_seconds'], point['monotonic_seconds']) <= ts['TRAIN_readout_budget_seconds']
        checkpoint = point['checkpoint_identity']
        assert d.relative(checkpoint['path']) == d.relative(folder/('step'+str(step)+'.pt'))
        d.stat_matches(checkpoint)  # Numerical checkpoint hashing/loading is guarded actual work.
        qualified[name] = dict(plan=train, summary=summary, external=external, checkpoint=checkpoint)
    return plan, review, manifest, truths, qualified
