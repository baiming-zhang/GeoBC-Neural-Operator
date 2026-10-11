"""Fixed occupied-token originalTS control after actual wider FNO readout; metadata only."""
from . import available_cad_pilot_data as d


def predecessor_terminal():
    from .wide_fno_readout_contract_v25 import approved_contract, complete_physical
    path = 'notes/geobc_wide_fno_readout_v25_plan.json'
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
    assert not summary['DEV_gates']['R64_FNO_W16_1500']['passed']
    for row in summary['candidates'].values(): complete_physical(row['full60'], plan['scientific']['case_keys'])
    return dict(plan=d.identity(path, hashed=True), terminal={k: d.identity(folder/f, hashed=True) for k, f in
        [('summary', 'summary.json'), ('status', 'status.json'), ('external', 'external_terminal.json')]})


def approved_contract(plan_path):
    plan = d.read(plan_path); science = plan['scientific']; review = d.read(plan['root_review'])
    assert plan['kind'] == 'GeoBC_occupied_originalTS_samephysical100_short5_v26'
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_execution_released']
    assert d.digest(science) == plan['scientific_digest'] == review['scientific_digest']
    assert plan['source_pins'] == review['source_pins'] and all(d.sha(p) == h for p, h in plan['source_pins'].items())
    assert science['methods'] == ['occupied_transolver'] and science['seed'] == 2026 and science['train_CAD_count'] == 100
    assert science['fixed_updates_per_arm'] == 1500 and science['passes'] == 5
    assert science['memory_limit_bytes'] == 30500000000 and science['GPU_lock'] == 'runs/geobc_exploration/.coordinator.lock'
    assert science['cache_setup_budget_seconds'] == 600 and science['arm_setup_budget_seconds'] == 120
    assert science['training_budget_seconds_per_arm'] == 900 and science['TRAIN_readout_budget_seconds_per_arm'] == 300
    assert science['arm_budget_seconds'] == 1500 and science['whole_budget_seconds'] == 1800
    assert science['global_cells'] == 4096 and science['decoder_chunk_size'] == 32768
    assert science['loss'] == 'full_original_node_physical_relative_U_S_squared_equal_case'
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
    assert science['operator_policy'] == 'originalTS unchanged architecture and initialstate; input-only occupiedR16 tokens; zero emptylatent; same oldquery/head'
    assert science['expected_parameter_count'] == 1882172 and not science['exact_capacity_matched']
    dense = science['dense_originalTS_TRAIN_summary_identity']; assert d.identity(dense['path'], hashed=True) == dense
    original = d.read(dense['path'])['methods']['transolver']
    assert original['status'] == 'completed' and original['model_profile']['parameter_count'] == 1882172
    assert science['dense_originalTS_initial_SHA256'] == original['actual_fresh_initial_model_sha256'] and science['exact_capacity_matched_to_dense_originalTS']
    item = science['CPU_fixture']; assert d.identity(item['path'], hashed=True) == item
    proof = d.read(item['path']); assert proof['status'] == 'actual_synthetic_occupied_TS_CPU_passed_not_GPU'
    assert proof['real_numeric_GT_reads'] == proof['GPU_calls'] == proof['optimizer_updates'] == 0
    assert proof['profile']['parameter_count'] == 1882172 and proof['all_initial_state_equal_dense_originalTS'] and proof['exact_empty_latent_zero']
    assert proof['source'] == d.identity('geobc/occupied_transolver_v26.py', hashed=True)
    assert plan['output'] == 'runs/geobc_exploration/occupied_TS100_physical5_v26'
    return plan, review, manifest, order
