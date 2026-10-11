"""One-shot two-decoder physical screen, with direct current terminal bindings."""
from . import available_cad_pilot_data as d
from .global_hash_physical_readout_contract_v18 import check_identity, complete_physical
from .resolution_training_contract_v42 import baseline_binding

METHODS = ['FNO5_MLP_RESIDUAL', 'FNO5_TCNN_RESIDUAL']


def predecessor_binding():
    name = 'notes/geobc_resolution_FNO_readout_v43_plan.json'
    p = d.read(name); folder = d.path(p['output'])
    r = d.read(p['root_review']); a = d.read(folder/'summary.json'); e = d.read(folder/'external_terminal.json')
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(p['scientific']) == p['scientific_digest'] == r['scientific_digest']
    assert p['source_pins'] == r['source_pins']
    assert a == d.read(folder/'status.json') and a['status'] == 'completed' and a['protocol_qualified']
    assert a['closed'] and not a['active'] and e['closed'] and not e['active']
    assert e['exit_code'] == 0 and e['within_external_budget']
    assert all(d.exact_process_exited(e[k]) for k in ('child_process', 'guard_process'))
    assert a['execution']['run_process'] == e['child_process']
    assert a['execution']['guard']['guard_process'] == e['guard_process']
    assert a['execution']['scientific_digest'] == e['scientific_digest'] == p['scientific_digest']
    assert a['committed_finite_predictions'] == a['owned_GT_array_reads'] == a['owned_GT_read_attempts'] == 60
    assert a['raw_GT_reads'] == a['FINAL_numeric_reads'] == a['training_updates'] == a['scaler_fits'] == 0
    for row in a['candidates'].values(): complete_physical(row['full60'], p['scientific']['case_keys'])
    return dict(plan=d.identity(name, hashed=True), terminal={k:d.identity(folder/f, hashed=True) for k,f in
        [('summary','summary.json'), ('status','status.json'), ('external','external_terminal.json')]}), p


def approved_contract(plan_path):
    p = d.read(plan_path); s = p['scientific']; r = d.read(p['root_review'])
    assert p['kind'] == 'GeoBC_frozen_FNO5_two_zero_residual_decoders_physical5_v45'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s) == p['scientific_digest'] == r['scientific_digest']
    assert p['source_pins'] == r['source_pins'] and all(d.sha(k) == h for k,h in p['source_pins'].items())
    assert s['methods'] == METHODS and s['seed'] == 2026 and s['train_CAD_count'] == 100
    assert s['memory_limit_bytes'] == 30500000000 and s['GPU_lock'] == 'runs/geobc_exploration/.coordinator.lock'
    assert s['additional_updates_per_arm'] == 1500 and s['additional_passes'] == 5
    assert s['teacher_passes'] == 5 and s['teacher_step'] == 1500 and s['reference_passes'] == 0
    assert s['optimizer'] == dict(name='Adam', learning_rate=.001, weight_decay=0., betas=[.9,.999], eps=1e-8, fused=True)
    assert s['loss'] == 'full_original_node_physical_relative_squared_equal_U_S_case'
    assert s['decoder_modes'] == dict(FNO5_MLP_RESIDUAL='mlp', FNO5_TCNN_RESIDUAL='hash')
    assert s['residual_formula'] == 'teacher_encoded4 + f_theta - f_theta0'
    assert s['original_R64_grid_and_stencils'] and s['release_consumed_input_grids']
    assert s['teacher_feature_cache_once'] and not s['teacher_prediction_distillation']
    assert s['DEV_FINAL_reads_allowed'] == s['scaler_fits_allowed'] == 0
    assert [s[k] for k in ('cache_teacher_budget_seconds','arm_setup_budget_seconds','training_budget_seconds_per_arm','TRAIN_readout_budget_seconds_per_arm','arm_budget_seconds','whole_budget_seconds')] == [600,120,900,300,1500,3600]
    predecessor, pp = predecessor_binding(); assert s['predecessor_terminal'] == predecessor
    binding, bp = baseline_binding(); assert s['teacher_terminal'] == binding
    assert s['dataset_identity'] == pp['scientific']['dataset_identity'] == bp['scientific']['dataset_identity']
    assert s['scaler'] == pp['scientific']['scaler'] == bp['scientific']['scaler']
    parent = d.read(binding['terminal']['summary']['path'])
    assert s['teacher_checkpoint'] == parent['endpoints']['1500']['checkpoint_identity']
    d.stat_matches(s['teacher_checkpoint'])
    assert s['teacher_model_state_digest'] == 'validate_actual_trained_state_at_load_not_initial'
    assert s['teacher_old_cost'] == dict(training_epoch_seconds=parent['training_epoch_seconds'],training_monotonic_seconds=parent['training_monotonic_seconds'],whole_epoch_seconds=parent['elapsed_epoch_seconds'],whole_monotonic_seconds=parent['elapsed_monotonic_seconds'])
    m = d.read(check_identity(s['dataset_identity'])); keys = m['train_keys']['100']
    assert m['status'] == 'completed' and m['feasibility_pass'] and keys == s['TRAIN_keys'] and len(set(keys)) == len(keys) == 300
    assert all(m['case_roles'][k] == 'TRAIN' for k in keys) and m['scaler'] == s['scaler']
    from .macro_training import balanced_schedule
    order = balanced_schedule(keys, 5, 2026)
    assert d.digest(order) == s['full_order_digest'] == bp['scientific']['full_order_digest']
    from .tcnn_fields import TCNNFieldConfig
    assert s['TCNN_config'] == TCNNFieldConfig(out_channels=4,input_mode='both',condition_dim=73,seed=2026).record()
    proof = d.read(check_identity(s['CPU_fixture']))
    assert proof['status'] == 'actual_synthetic_CPU_zero_residual_and_first_gradient_passed_not_native_GPU'
    assert proof['GPU_calls'] == proof['trained_PT_reads'] == proof['dataset_GT_reads'] == proof['optimizer_updates'] == 0
    assert proof['source'] == d.identity('geobc/operator_residual_v44.py', hashed=True)
    physical_proof = d.read(check_identity(s['physical_CPU_fixture']))
    assert physical_proof['status'] == 'actual_synthetic_CPU_physical_chunk_loss_gradient_passed_not_native_GPU'
    assert physical_proof['GPU_calls'] == physical_proof['trained_PT_reads'] == physical_proof['dataset_GT_reads'] == physical_proof['optimizer_updates'] == 0
    assert physical_proof['source'] == d.identity('geobc/residual_physical_training_v45.py', hashed=True)
    assert physical_proof['checker'] == d.identity('tools/check_operator_residual_physical_v45_CPU.py', hashed=True)
    assert p['output'] == 'runs/geobc_exploration/frozen_FNO5_two_residual_physical5_v45'
    return p,r,m,order
