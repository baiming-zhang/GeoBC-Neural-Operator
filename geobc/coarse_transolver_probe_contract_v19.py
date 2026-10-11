"""Metadata release for six real TRAIN gradient witnesses; not training."""
from . import available_cad_pilot_data as d


def approved_contract(plan_path):
    plan = d.read(plan_path); science = plan['scientific']; review = d.read(plan['root_review'])
    assert plan['kind'] == 'GeoBC_coarseTransolver_R16_resourcequalification_v19'
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_execution_released']
    assert plan['scientific_digest'] == review['scientific_digest'] == d.digest(science)
    assert plan['source_pins'] == review['source_pins'] and all(d.sha(p) == h for p, h in plan['source_pins'].items())
    required = ('geobc/coarse_transolver_v19.py', 'geobc/coarse_transolver_probe_contract_v19.py',
                'tools/probe_geobc_coarse_transolver_v19.py', 'tools/run_geobc_coarse_transolver_probe_guard_v19.py',
                'geobc/physical_field_loss.py', 'geobc/transformers.py')
    assert set(required) <= set(plan['source_pins'])
    assert science['methods'] == ['transolver', 'transolver_pp', 'ts3'] and science['seed'] == 2026
    assert science['memory_limit_bytes'] == 30500000000 and science['whole_budget_seconds'] == 300
    assert science['setup_budget_seconds'] == 120 and science['GPU_lock'] == 'runs/geobc_exploration/.coordinator.lock'
    assert science['gradient_attempts'] == 6 and science['optimizer_updates'] == 0
    assert science['precision'] == 'highest FP32; TF32 disabled' and science['global_tokens'] == 4096
    assert science['loss'] == 'full_original_node_physical_relative_U_S_squared_equal_case'
    assert not science['DEV_FINAL_numeric_reads_allowed'] and not science['scaler_fits_allowed']
    item = science['dataset_identity']; assert d.identity(item['path'], hashed=True) == item
    manifest = d.read(item['path']); assert manifest['status'] == 'completed' and manifest['feasibility_pass']
    assert science['scaler'] == manifest['scaler']
    cad_keys = sorted((k for k in manifest['train_keys']['100'] if k.endswith('/ver')),
                      key=lambda k: (manifest['records'][k]['num_nodes'], k))
    assert len(cad_keys) == 100
    keys = [cad_keys[len(cad_keys)//2], cad_keys[-1]]
    assert science['case_keys'] == keys and len(set(keys)) == 2
    assert all(manifest['case_roles'][k] == 'TRAIN' for k in keys)
    fixture = science['CPU_fixture']; assert d.identity(fixture['path'], hashed=True) == fixture
    proof = d.read(fixture['path'])
    assert proof['status'] == 'actual_CPU_math_fixture_passed' and proof['official_models_instantiated'] == proof['GPU_calls'] == 0
    assert proof['source'] == d.identity('geobc/coarse_transolver_v19.py', hashed=True)
    assert plan['output'] == 'runs/geobc_exploration/coarse_transolver_resource_v19'
    return plan, review, manifest, keys
