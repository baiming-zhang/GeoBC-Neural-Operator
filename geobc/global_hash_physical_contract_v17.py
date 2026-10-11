"""Fixed short decoder comparison; only metadata is read before release."""
from . import available_cad_pilot_data as d

REQUIRED = ('geobc/global_hash_physical_contract_v17.py', 'tools/train_geobc_global_hash_physical_v17.py',
    'tools/run_geobc_global_hash_physical_guard_v17.py', 'geobc/operator_hash_training.py',
    'geobc/operator_conditioned_hash.py', 'geobc/tcnn_fields.py', 'geobc/train_physical_cache_v12.py',
    'geobc/physical_field_loss.py', 'geobc/macro_physical_contract_v12.py')


def approved_contract(plan_path):
    plan = d.read(plan_path); science = plan['scientific']; review = d.read(plan['root_review'])
    assert plan['kind'] == 'GeoBC_globalHash100_shared_physical_short5_v17'
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_execution_released']
    assert plan['scientific_digest'] == review['scientific_digest'] == d.digest(science)
    assert plan['source_pins'] == review['source_pins'] and set(REQUIRED) <= set(plan['source_pins'])
    assert all(d.sha(p) == h for p, h in plan['source_pins'].items())
    assert science['family'] == 'FNO_GLOBAL_HASH' and science['train_CAD_count'] == 100 and science['seed'] == 2026
    assert science['memory_limit_bytes'] == 30500000000 and science['GPU_lock'] == 'runs/geobc_exploration/.coordinator.lock'
    assert science['loss'] == 'full_original_node_physical_relative_U_S_squared_equal_case'
    assert science['optimizer'] == dict(name='Adam', learning_rate=.001, weight_decay=0., betas=[.9,.999], eps=1e-8, fused=True)
    assert science['fresh_initialization'] and science['decoder_chunk_size'] == 32768
    assert science['endpoint_passes'] == [5] and science['fixed_updates'] == 1500
    assert science['parameter_count'] == 12245894 and not science['capacity_precision_matched']
    assert science['cache_setup_budget_seconds'] == 600 and science['training_budget_seconds'] == 900
    assert science['TRAIN_readout_budget_seconds'] == 300 and science['whole_budget_seconds'] == 1800
    assert not science['FINAL_reuse_for_selection'] and science['DEV_GT_reads_allowed'] == 0
    assert science['TRAIN_gate'] == 'complete finite300 and genuine largestN joint gradients; no accuracy filtering'
    for field in ('design_reference', 'dataset_identity', 'baseline_plan_identity'):
        item = science[field]; assert d.identity(item['path'], hashed=True) == item
    manifest = d.read(science['dataset_identity']['path']); keys = manifest['train_keys']['100']
    assert manifest['status'] == 'completed' and science['TRAIN_keys'] == keys and len(set(keys)) == len(keys) == 300
    assert science['scaler'] == manifest['scaler'] and all(manifest['case_roles'][k] == 'TRAIN' for k in keys)
    from .macro_physical_contract_v12 import approved_contract as baseline_contract
    bp, _, bm, bo = baseline_contract('notes/geobc_macro100_FNO9_physical5_v12_plan.json')
    assert bm == manifest and bp['scientific_digest'] == science['baseline_scientific_digest']
    assert d.identity('notes/geobc_macro100_FNO9_physical5_v12_plan.json', hashed=True) == science['baseline_plan_identity']
    for field in ('seed', 'train_CAD_count', 'TRAIN_keys', 'dataset_identity', 'scaler', 'optimizer',
                  'fresh_initialization', 'decoder_chunk_size', 'endpoint_passes', 'fixed_updates', 'full_order_digest', 'loss'):
        assert science[field] == bp['scientific'][field]
    binding = science['baseline_terminal']
    for k, name in (('summary', 'summary.json'), ('status', 'status.json'), ('external', 'external_terminal.json')):
        item = binding[k]
        assert d.relative(item['path']) == d.relative(d.path(bp['output'])/name)
        assert d.identity(item['path'], hashed=True) == item
    baseline = d.read(binding['summary']['path']); external = d.read(binding['external']['path'])
    assert baseline == d.read(binding['status']['path']) and baseline['status'] == 'completed' and baseline['training_complete']
    assert baseline['closed'] and not baseline['active'] and external['closed'] and not external['active']
    assert external['exit_code'] == 0 and external['within_external_budget'] and not external['external_timeout']
    assert all(d.exact_process_exited(external[k]) for k in ('child_process', 'guard_process'))
    assert baseline['execution']['run_process'] == external['child_process']
    assert baseline['execution']['guard']['guard_process'] == external['guard_process']
    assert baseline['execution']['scientific_digest'] == external['scientific_digest'] == bp['scientific_digest']
    assert baseline['DEV_FINAL_reads'] == baseline['raw_GT_reads'] == baseline['scaler_fits'] == 0
    assert baseline['actual_largest_fullN_gradient_check']['passed'] and baseline['step'] == 1500
    assert baseline['optimizer_step_attempts'] == baseline['optimizer_step_completions'] == 1500
    assert not baseline['possible_partial_optimizer_update'] and baseline['case_exposure_counts'] == {k: 5 for k in keys}
    from .physical_readout_contract_v12 import complete_physical
    metric = baseline['endpoints']['1500']['full_TRAIN']; complete_physical(metric, keys)
    assert metric == science['baseline_TRAIN300']
    assert plan['output'] == 'runs/geobc_exploration/macro100_FNO_GLOBAL_HASH_physical5_v17'
    return plan, review, manifest, bo
