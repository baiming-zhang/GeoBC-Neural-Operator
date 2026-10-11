"""Metadata-only release contract for future selected macro training jobs."""
from . import available_cad_pilot_data as d

KIND = 'GeoBC_macro_full_case_training_v1'
PARENT = 'notes/geobc_exposure_readout_plan.json'
GPU_LOCK = 'runs/geobc_exploration/.coordinator.lock'
REQUIRED_SOURCES = (
    'geobc/macro_campaign_contract.py', 'geobc/macro_training.py',
    'geobc/train_resident_cache.py', 'geobc/prospective_dataset.py',
    'geobc/native_context_solver.py', 'geobc/common_grid_solver.py',
    'geobc/conditioned_field.py', 'geobc/tcnn_fields.py', 'geobc/fno.py',
    'geobc/fno_arbitrary_query.py', 'geobc/models.py', 'geobc/train.py',
    'geobc/transformers.py', 'geobc/benchmark.py',
    'geobc/coverage_comparison_train.py', 'geobc/available_cad_pilot_data.py',
    'third_party/transformer_sources.json',
    'tools/train_geobc_macro_campaign.py', 'tools/run_geobc_macro_campaign_guard.py',
    'tools/export_geobc_exposure_comparison.py',
    'tools/evaluate_geobc_exposure_endpoints.py',
    'tools/train_geobc_single_case_diagnostic.py')


def approved_contract(plan_path):
    """Check actual parent exit, identities, science and scope before libraries.

    No prepare/release function invents an endpoint or sets approval here.
    The future selected stage's root review must bind genuine final evidence.
    """
    plan = d.read(plan_path)
    science = plan['scientific']
    review = d.read(plan['root_review'])
    assert plan['kind'] == KIND
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_execution_released'], 'Macro source/science/actual release is required before any libraries or data arrays'
    assert review['scientific_digest'] == plan['scientific_digest'] == d.digest(science)
    assert review['source_pins'] == plan['source_pins']
    assert set(REQUIRED_SOURCES) <= set(plan['source_pins'])
    assert all(d.sha(p) == sha for p, sha in plan['source_pins'].items())
    assert science['parent_plan'] == PARENT and science['GPU_lock'] == GPU_LOCK
    assert science['family'] in ('FNO9', 'PP256_NATIVE', 'HASH_PN')
    assert science['train_CAD_count'] in (40, 100) and science['seed'] == 2026
    assert science['memory_limit_bytes'] == 30500000000
    assert science['loss'] == 'full_original_node_encoded4_case_mean_MSE'
    assert science['optimizer'] == dict(name='Adam', learning_rate=.001, weight_decay=0.,
                                       betas=[.9, .999], eps=1e-8, fused=True)
    assert science['fresh_initialization'] and science['decoder_chunk_size'] == 32768
    passes = science['endpoint_passes']
    assert passes and passes == sorted(set(passes)) and all(type(x) is int and x > 0 for x in passes)
    assert all(type(science[k]) is int and science[k] > 0 for k in
               ('cache_setup_budget_seconds', 'training_budget_seconds', 'TRAIN_readout_budget_seconds', 'whole_budget_seconds'))
    assert science['parent_scientific_digest'] == d.read(PARENT)['scientific_digest']
    parent = d.read(PARENT)
    for name, filename in [('summary', 'summary.json'), ('external', 'external_terminal.json')]:
        identity = review['parent_binding'][name]
        assert d.relative(identity['path']) == d.relative(d.path(parent['output'])/filename)
        assert d.identity(identity['path'], hashed=True) == identity
    from tools.export_geobc_exposure_comparison import terminal
    from tools.evaluate_geobc_exposure_endpoints import parent_qualified
    summary, external = terminal(parent)
    assert summary['status'] == 'completed' and summary['protocol_qualified']
    assert external['within_external_budget'] and external['exit_code'] == 0 and not external['external_timeout']
    assert summary['committed_finite_predictions'] == 240 and summary['owned_GT_array_reads'] == 60
    assert summary['raw_GT_reads'] == summary['sealed_DEV_FINAL_reads'] == summary['training_updates'] == summary['scaler_fits'] == 0
    parent_qualified(parent, d.read(parent['root_review']))
    manifest_identity = science['dataset_identity']
    assert d.relative(manifest_identity['path']) == 'data/geobc_prospective_dataset_v1/manifest.json'
    assert d.identity(manifest_identity['path'], hashed=True) == manifest_identity
    manifest = d.read(manifest_identity['path'])
    assert manifest['status'] == 'completed' and manifest['feasibility_pass']
    assert manifest['scaler'] == science['scaler']
    keys = manifest['train_keys'][str(science['train_CAD_count'])]
    assert keys == science['TRAIN_keys'] and len(keys) == 3*science['train_CAD_count']
    assert all(manifest['case_roles'][k] == 'TRAIN' for k in keys)
    from .macro_training import balanced_schedule
    order = balanced_schedule(keys, max(passes), science['seed'])
    assert d.digest(order) == science['full_order_digest']
    assert science['fixed_updates'] == len(order)
    output = d.path(plan['output'])
    assert d.relative(output).startswith('runs/geobc_exploration/macro_')
    return plan, review, manifest, order


__all__ = ['KIND', 'PARENT', 'GPU_LOCK', 'REQUIRED_SOURCES', 'approved_contract']
