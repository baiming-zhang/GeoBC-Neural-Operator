"""Fixed paired encoded4 objective control after actual occupiedTS readout; metadata only."""
from . import available_cad_pilot_data as d


def predecessor_terminal():
    from .occupied_ts_readout_contract_v27 import approved_contract, complete_physical
    path = 'notes/geobc_occupied_ts_readout_v27_plan.json'
    plan, _, _, _, _ = approved_contract(path); folder = d.path(plan['output'])
    summary = d.read(folder/'summary.json'); external = d.read(folder/'external_terminal.json')
    assert summary == d.read(folder/'status.json')
    assert summary['status'] == 'completed' and summary['protocol_qualified'] and summary['closed'] and not summary['active']
    assert external['closed'] and not external['active'] and external['exit_code'] == 0 and external['within_external_budget']
    assert all(d.exact_process_exited(external[k]) for k in ('child_process', 'guard_process'))
    assert summary['execution']['run_process'] == external['child_process']
    assert summary['execution']['guard']['guard_process'] == external['guard_process']
    assert summary['execution']['scientific_digest'] == external['scientific_digest'] == plan['scientific_digest']
    assert summary['committed_finite_predictions'] == summary['owned_GT_array_reads'] == summary['owned_GT_read_attempts'] == 60
    assert summary['eligible_best'] == ['R64_FNO_W16_1500']
    assert not summary['DEV_gates']['R16_OCCUPIED_TRANSOLVER_1500']['passed']
    for row in summary['candidates'].values(): complete_physical(row['full60'], plan['scientific']['case_keys'])
    return dict(plan=d.identity(path, hashed=True), terminal={k: d.identity(folder/f, hashed=True) for k, f in
        [('summary', 'summary.json'), ('status', 'status.json'), ('external', 'external_terminal.json')]})


def approved_contract(plan_path):
    plan = d.read(plan_path); science = plan['scientific']; review = d.read(plan['root_review'])
    assert plan['kind'] == 'GeoBC_R16_TS_FNO_encoded4_objective_same100_short5_v28'
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_execution_released']
    assert d.digest(science) == plan['scientific_digest'] == review['scientific_digest']
    assert plan['source_pins'] == review['source_pins'] and all(d.sha(p) == h for p, h in plan['source_pins'].items())
    assert science['methods'] == ['transolver_encoded', 'FNO16_encoded'] and science['seed'] == 2026 and science['train_CAD_count'] == 100
    assert science['fixed_updates_per_arm'] == 1500 and science['passes'] == 5
    assert science['memory_limit_bytes'] == 30500000000 and science['GPU_lock'] == 'runs/geobc_exploration/.coordinator.lock'
    assert science['cache_setup_budget_seconds'] == 600 and science['arm_setup_budget_seconds'] == 120
    assert science['training_budget_seconds_per_arm'] == 900 and science['TRAIN_readout_budget_seconds_per_arm'] == 300
    assert science['arm_budget_seconds'] == 1500 and science['whole_budget_seconds'] == 3600
    assert science['global_cells'] == 4096 and science['decoder_chunk_size'] == 32768
    assert science['loss'] == 'full_original_node_standardized_encoded4_MSE_equal_encoded_channel_and_case'
    assert science['precision'] == 'highest FP32; TF32 disabled'
    assert science['optimizer'] == dict(name='Adam', learning_rate=.001, weight_decay=0., betas=[.9, .999], eps=1e-8, fused=True)
    assert science['DEV_FINAL_reads_allowed'] == science['scaler_fits_allowed'] == 0
    assert science['fresh_initialization'] and science['shared_cache_loads'] == 1 and science['qualification_RNG_restored']
    assert science['TRAIN_readout_reuses_physical_cache'] and science['predecessor_terminal'] == predecessor_terminal()
    item = science['dataset_identity']; assert d.identity(item['path'], hashed=True) == item
    manifest = d.read(item['path']); assert manifest['status'] == 'completed' and manifest['feasibility_pass']
    keys = manifest['train_keys']['100']; assert science['TRAIN_keys'] == keys and len(keys) == len(set(keys)) == 300
    assert all(manifest['case_roles'][key] == 'TRAIN' for key in keys) and science['scaler'] == manifest['scaler']
    from .macro_training import balanced_schedule
    order = balanced_schedule(keys, 5, 2026); assert d.digest(order) == science['full_order_digest']
    assert science['operator_policy'] == 'dense originalTS and FNO16 control unchanged; only training loss changes; sameR16/head/query'
    assert science['expected_parameter_counts'] == dict(transolver_encoded=1882172, FNO16_encoded=4244244)
    assert not science['exact_capacity_matched'] and science['exact_capacity_and_initial_state_matched_to_own_physical_controls']
    assert set(science['initial_SHA256']) == set(science['methods'])
    for method, item in science['physical_control_TRAIN_bindings'].items():
        assert d.identity(item['path'], hashed=True) == item
        original = d.read(item['path'])['methods'][{'transolver_encoded': 'transolver', 'FNO16_encoded': 'FNO16_control'}[method]]
        assert original['status'] == 'completed' and original['step'] == 1500
        assert original['model_profile']['parameter_count'] == science['expected_parameter_counts'][method]
        assert science['initial_SHA256'][method] == original['actual_fresh_initial_model_sha256']
    item = science['CPU_fixture']; assert d.identity(item['path'], hashed=True) == item
    proof = d.read(item['path']); assert proof['status'] == 'actual_synthetic_encoded4_two_model_chain_CPU_passed_not_GPU'
    assert proof['real_numeric_GT_reads'] == proof['GPU_calls'] == proof['optimizer_updates'] == 0
    assert proof['synthetic_nodes'] == 33337 and proof['chunks'] == [32768, 569]
    for method in science['methods']:
        assert proof['results'][method]['all_named_gradients_close'] and proof['results'][method]['initial_states_equal']
    assert proof['source'] == d.identity('geobc/encoded_objective_v28.py', hashed=True)
    assert plan['output'] == 'runs/geobc_exploration/encoded4_TS_FNO100_short5_v28'
    return plan, review, manifest, order
