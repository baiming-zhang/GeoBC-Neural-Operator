"""Fixed three-arm short screen; metadata-only release and terminal binding."""
from . import available_cad_pilot_data as d


def probe_terminal():
    receipt = d.read('notes/geobc_coarse_transolver_probe_v19_terminal_receipt.json')
    assert receipt['status'] == 'actual_completed_real6_wholeN_gradients'
    for item in receipt['terminal'].values():
        assert d.identity(item['path'], hashed=True) == item
    summary = d.read(receipt['terminal']['summary']['path'])
    external = d.read(receipt['terminal']['external']['path'])
    assert summary == d.read(receipt['terminal']['status']['path'])
    assert summary['status'] == 'completed' and summary['closed'] and not summary['active']
    assert summary['resource_qualified'] and summary['gradient_completions'] == 6
    assert external['closed'] and not external['active'] and external['exit_code'] == 0
    assert external['within_external_budget'] and external['actual_child_exited']
    assert all(d.exact_process_exited(external[k]) for k in ('child_process', 'guard_process'))
    assert summary['execution']['run_process'] == external['child_process']
    assert summary['execution']['guard']['guard_process'] == external['guard_process']
    assert summary['execution']['scientific_digest'] == external['scientific_digest'] == receipt['SCI']
    return dict(receipt=d.identity('notes/geobc_coarse_transolver_probe_v19_terminal_receipt.json', hashed=True),
                terminal=receipt['terminal'])


def approved_contract(plan_path):
    plan = d.read(plan_path); science = plan['scientific']; review = d.read(plan['root_review'])
    assert plan['kind'] == 'GeoBC_R16_threeTransolvers_shared100_physical5_v20'
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_execution_released']
    assert d.digest(science) == plan['scientific_digest'] == review['scientific_digest']
    assert plan['source_pins'] == review['source_pins']
    assert all(d.sha(name) == pin for name, pin in plan['source_pins'].items())
    assert science['methods'] == ['transolver', 'transolver_pp', 'ts3']
    assert science['seed'] == 2026 and science['train_CAD_count'] == 100
    assert science['fixed_updates_per_arm'] == 1500 and science['passes'] == 5
    assert science['memory_limit_bytes'] == 30500000000
    assert science['GPU_lock'] == 'runs/geobc_exploration/.coordinator.lock'
    assert science['cache_setup_budget_seconds'] == 600 and science['arm_setup_budget_seconds'] == 120
    assert science['training_budget_seconds_per_arm'] == 900
    assert science['TRAIN_readout_budget_seconds_per_arm'] == 300
    assert science['arm_budget_seconds'] == 1500 and science['whole_budget_seconds'] == 5400
    assert science['global_tokens'] == 4096 and science['decoder_chunk_size'] == 32768
    assert science['loss'] == 'full_original_node_physical_relative_U_S_squared_equal_case'
    assert science['precision'] == 'highest FP32; TF32 disabled'
    assert science['optimizer'] == dict(name='Adam', learning_rate=.001, weight_decay=0., betas=[.9, .999], eps=1e-8, fused=True)
    assert science['DEV_FINAL_reads_allowed'] == 0 and science['scaler_fits_allowed'] == 0
    assert science['independent_initializations'] and science['shared_cache_loads'] == 1
    assert science['qualification_RNG_restored'] and science['TRAIN_readout_reuses_physical_cache']
    assert science['probe_terminal'] == probe_terminal()
    item = science['dataset_identity']; assert d.identity(item['path'], hashed=True) == item
    manifest = d.read(item['path']); keys = manifest['train_keys']['100']
    assert manifest['status'] == 'completed' and manifest['feasibility_pass']
    assert science['TRAIN_keys'] == keys and len(keys) == len(set(keys)) == 300
    assert all(manifest['case_roles'][key] == 'TRAIN' for key in keys)
    assert science['scaler'] == manifest['scaler']
    from .macro_training import balanced_schedule
    order = balanced_schedule(keys, 5, 2026)
    assert d.digest(order) == science['full_order_digest']
    assert plan['output'] == 'runs/geobc_exploration/coarse_three100_physical5_v20'
    return plan, review, manifest, order
