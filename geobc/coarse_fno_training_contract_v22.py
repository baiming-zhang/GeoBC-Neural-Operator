"""SameR16 FNO control after actual threeTS negative knownDEV; metadata only."""
from . import available_cad_pilot_data as d


def predecessor_terminal():
    from .coarse_transolver_readout_contract_v21 import approved_contract, complete_physical
    path = 'notes/geobc_coarse_three_readout_v21_plan.json'
    plan, _, _, _, _ = approved_contract(path); folder = d.path(plan['output'])
    summary = d.read(folder/'summary.json'); external = d.read(folder/'external_terminal.json')
    assert summary == d.read(folder/'status.json')
    assert summary['status'] == 'completed' and summary['protocol_qualified'] and summary['closed'] and not summary['active']
    assert external['closed'] and not external['active'] and external['exit_code'] == 0 and external['within_external_budget']
    assert all(d.exact_process_exited(external[k]) for k in ('child_process', 'guard_process'))
    assert summary['execution']['run_process'] == external['child_process']
    assert summary['execution']['guard']['guard_process'] == external['guard_process']
    assert summary['execution']['scientific_digest'] == external['scientific_digest'] == plan['scientific_digest']
    assert summary['committed_finite_predictions'] == 180 and summary['owned_GT_array_reads'] == summary['owned_GT_read_attempts'] == 60
    assert all(not gate['passed'] for gate in summary['DEV_gates'].values())
    assert summary['eligible_best'] == ['FNO100_PHYSICAL_1500']
    for row in summary['candidates'].values(): complete_physical(row['full60'], plan['scientific']['case_keys'])
    return dict(plan=d.identity(path, hashed=True), terminal={k: d.identity(folder/f, hashed=True) for k, f in
        [('summary', 'summary.json'), ('status', 'status.json'), ('external', 'external_terminal.json')]})


def approved_contract(plan_path):
    plan = d.read(plan_path); science = plan['scientific']; review = d.read(plan['root_review'])
    assert plan['kind'] == 'GeoBC_R16_FNO_samephysical100_short5_control_v22'
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_execution_released']
    assert d.digest(science) == plan['scientific_digest'] == review['scientific_digest']
    assert plan['source_pins'] == review['source_pins'] and all(d.sha(p) == h for p, h in plan['source_pins'].items())
    assert science['methods'] == ['FNO16_control'] and science['seed'] == 2026 and science['train_CAD_count'] == 100
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
    assert science['FNO_config'] == dict(width=16, modes=8, layers=4, head_width=128, checkpoint_blocks=True)
    assert science['expected_parameter_count'] == 4244244 and not science['exact_capacity_matched']
    item = science['CPU_fixture']; assert d.identity(item['path'], hashed=True) == item
    proof = d.read(item['path']); assert proof['status'] == 'actual_synthetic_CPU_interface_passed_not_GPU'
    assert proof['real_numeric_GT_reads'] == proof['GPU_calls'] == proof['optimizer_updates'] == 0
    assert proof['profile']['parameter_count'] == 4244244 and proof['actual_shared_encoder_head_equal_seeded_FNO']
    assert proof['source'] == d.identity('geobc/coarse_fno_control_v22.py', hashed=True)
    assert plan['output'] == 'runs/geobc_exploration/coarse_FNO100_physical5_control_v22'
    return plan, review, manifest, order
