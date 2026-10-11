"""Fixed paired dimensional response prior after actual objective-control readout; metadata only."""
from . import available_cad_pilot_data as d


def predecessor_terminal():
    from .global_hash_physical_readout_contract_v18 import complete_physical
    path = 'notes/geobc_encoded_pair_readout_v29_plan.json'
    plan = d.read(path); parent_review = d.read(plan['root_review'])
    assert parent_review['passed'] and parent_review['whole_source_reviewed'] and parent_review['actual_execution_released']
    assert d.digest(plan['scientific']) == plan['scientific_digest'] == parent_review['scientific_digest']
    assert plan['source_pins'] == parent_review['source_pins']
    folder = d.path(plan['output'])
    summary = d.read(folder/'summary.json'); external = d.read(folder/'external_terminal.json')
    assert summary == d.read(folder/'status.json')
    assert summary['status'] == 'completed' and summary['protocol_qualified'] and summary['closed'] and not summary['active']
    assert external['closed'] and not external['active'] and external['exit_code'] == 0 and external['within_external_budget']
    assert all(d.exact_process_exited(external[k]) for k in ('child_process', 'guard_process'))
    assert summary['execution']['run_process'] == external['child_process']
    assert summary['execution']['guard']['guard_process'] == external['guard_process']
    assert summary['execution']['scientific_digest'] == external['scientific_digest'] == plan['scientific_digest']
    assert external['source_pins'] == plan['source_pins']
    assert summary['raw_GT_reads'] == summary['FINAL_numeric_reads'] == summary['training_updates'] == summary['scaler_fits'] == 0
    assert max(summary['elapsed_epoch_seconds'], summary['elapsed_monotonic_seconds'], external['elapsed_epoch_seconds'], external['elapsed_monotonic_seconds']) <= 1800
    assert max(summary[k] for k in ('peak_allocated_bytes', 'peak_reserved_bytes', 'peak_total_device_used_bytes_sampled')) <= 30500000000
    assert summary['committed_finite_predictions'] == 120 and summary['owned_GT_array_reads'] == summary['owned_GT_read_attempts'] == 60
    assert summary['eligible_best'] == ['R64_FNO_W16_1500']
    assert all(not gate['passed'] for gate in summary['DEV_gates'].values())
    for row in summary['candidates'].values(): complete_physical(row['full60'], plan['scientific']['case_keys'])
    return dict(plan=d.identity(path, hashed=True), terminal={k: d.identity(folder/f, hashed=True) for k, f in
        [('summary', 'summary.json'), ('status', 'status.json'), ('external', 'external_terminal.json')]})


def approved_contract(plan_path):
    plan = d.read(plan_path); science = plan['scientific']; review = d.read(plan['root_review'])
    assert plan['kind'] == 'GeoBC_R16_TS_FNO_input_response_gain_same100_short5_v30'
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_execution_released']
    assert d.digest(science) == plan['scientific_digest'] == review['scientific_digest']
    assert plan['source_pins'] == review['source_pins'] and all(d.sha(p) == h for p, h in plan['source_pins'].items())
    assert science['methods'] == ['transolver_gain', 'FNO16_gain'] and science['seed'] == 2026 and science['train_CAD_count'] == 100
    assert science['fixed_updates_per_arm'] == 1500 and science['passes'] == 5
    assert science['memory_limit_bytes'] == 30500000000 and science['GPU_lock'] == 'runs/geobc_exploration/.coordinator.lock'
    assert science['cache_setup_budget_seconds'] == 600 and science['arm_setup_budget_seconds'] == 120
    assert science['training_budget_seconds_per_arm'] == 900 and science['TRAIN_readout_budget_seconds_per_arm'] == 300
    assert science['arm_budget_seconds'] == 1500 and science['whole_budget_seconds'] == 3600
    assert science['global_cells'] == 4096 and science['decoder_chunk_size'] == 32768
    assert science['loss'] == 'full_original_node_physical_relative_squared_equal_U_S_case_with_input_response_gain'
    assert science['precision'] == 'highest FP32; TF32 disabled'
    assert science['optimizer'] == dict(name='Adam', learning_rate=.001, weight_decay=0., betas=[.9, .999], eps=1e-8, fused=True)
    assert science['DEV_FINAL_reads_allowed'] == science['scaler_fits_allowed'] == 0
    assert science['fresh_initialization'] and science['shared_cache_loads'] == 1 and science['qualification_RNG_restored']
    parent_binding = predecessor_terminal()
    assert science['TRAIN_readout_reuses_physical_cache'] and science['predecessor_terminal'] == parent_binding
    assert all(plan['source_pins'].get(k) == h for k, h in d.read(parent_binding['plan']['path'])['source_pins'].items())
    item = science['dataset_identity']; assert d.identity(item['path'], hashed=True) == item
    manifest = d.read(item['path']); assert manifest['status'] == 'completed' and manifest['feasibility_pass']
    keys = manifest['train_keys']['100']; assert science['TRAIN_keys'] == keys and len(keys) == len(set(keys)) == 300
    assert all(manifest['case_roles'][key] == 'TRAIN' for key in keys) and science['scaler'] == manifest['scaler']
    from .macro_training import balanced_schedule
    order = balanced_schedule(keys, 5, 2026); assert d.digest(order) == science['full_order_digest']
    assert science['operator_policy'] == 'dense originalTS and FNO16 control weights/native25/R16/query unchanged; fixed positive input-derived dimensional U/S gain only; original physical loss'
    assert science['expected_parameter_counts'] == dict(transolver_gain=1882172, FNO16_gain=4244244)
    assert not science['exact_capacity_matched'] and science['exact_capacity_and_initial_state_matched_to_own_physical_controls']
    assert set(science['initial_SHA256']) == set(science['methods'])
    for method, item in science['physical_control_TRAIN_bindings'].items():
        assert d.identity(item['path'], hashed=True) == item
        original = d.read(item['path'])['methods'][{'transolver_gain': 'transolver', 'FNO16_gain': 'FNO16_control'}[method]]
        assert original['status'] == 'completed' and original['step'] == 1500
        assert original['model_profile']['parameter_count'] == science['expected_parameter_counts'][method]
        assert science['initial_SHA256'][method] == original['actual_fresh_initial_model_sha256']
    item = science['CPU_fixture']; assert d.identity(item['path'], hashed=True) == item
    proof = d.read(item['path']); assert proof['status'] == 'actual_synthetic_response_gain_two_model_chain_CPU_passed_not_GPU'
    assert proof['real_numeric_GT_reads'] == proof['GPU_calls'] == proof['optimizer_updates'] == 0
    assert proof['synthetic_nodes'] == 33337 and proof['chunks'] == [32768, 569]
    for method in science['methods']:
        assert proof['results'][method]['all_named_gradients_close'] and proof['results'][method]['initial_weights_equal_own_physical_control']
    assert proof['source'] == d.identity('geobc/response_gain_v30.py', hashed=True)
    from .response_gain_v30 import FORMULA
    assert science['response_gain_formula'] == FORMULA
    assert science['input_features_unchanged'] and science['labels_unchanged'] and not science['exact_FEM_similarity_claimed']
    assert not science['initial_functions_equal'] and science['fixed_reference_constants_not_fitted']
    assert proof['input_formula_algebra_passed'] and proof['signed_decode_roundtrip_passed'] and proof['invalid_inputs_rejected']
    assert plan['output'] == 'runs/geobc_exploration/response_gain_TS_FNO100_physical5_v30'
    return plan, review, manifest, order
