"""Fixed TRAIN teacher-node operator diagnostic; direct terminal metadata only."""
from . import available_cad_pilot_data as d
from .geometry_head_training_contract_v33 import terminal_binding
from .global_hash_physical_readout_contract_v18 import complete_physical

METHODS = ['transolver_latent_teacher', 'FNO16_latent_teacher']


def predecessor_binding():
    p = d.read('notes/geobc_geometry_head_v34_plan.json')
    r = d.read(p['root_review']); folder = d.path(p['output'])
    assert r['passed'] and r['actual_execution_released'] and r['whole_source_reviewed']
    assert d.digest(p['scientific']) == p['scientific_digest'] == r['scientific_digest']
    assert p['source_pins'] == r['source_pins']
    summary = d.read(folder/'summary.json'); external = d.read(folder/'external_terminal.json')
    assert summary == d.read(folder/'status.json') and summary['status'] == 'completed'
    assert summary['closed'] and not summary['active'] and external['closed'] and not external['active']
    assert external['exit_code'] == 0 and external['within_external_budget'] and not external['external_timeout']
    assert all(d.exact_process_exited(external[k]) for k in ('child_process', 'guard_process'))
    assert summary['execution']['run_process'] == external['child_process']
    assert summary['execution']['guard']['guard_process'] == external['guard_process']
    assert summary['execution']['scientific_digest'] == external['scientific_digest'] == p['scientific_digest']
    assert external['source_pins'] == p['source_pins']
    assert summary['protocol_qualified'] and summary['committed_finite_predictions'] == 180
    assert summary['owned_GT_array_reads'] == summary['owned_GT_read_attempts'] == 60
    assert summary['raw_GT_reads'] == summary['FINAL_numeric_reads'] == summary['training_updates'] == summary['scaler_fits'] == 0
    for row in summary['candidates'].values(): complete_physical(row['full60'], p['scientific']['case_keys'])
    assert max(summary['elapsed_epoch_seconds'], summary['elapsed_monotonic_seconds'], external['elapsed_epoch_seconds'], external['elapsed_monotonic_seconds']) <= p['scientific']['whole_budget_seconds']
    assert max(summary[k] for k in ('peak_allocated_bytes', 'peak_reserved_bytes', 'peak_total_device_used_bytes_sampled')) <= 30500000000
    binding = dict(plan=d.identity('notes/geobc_geometry_head_v34_plan.json', hashed=True), terminal={k:d.identity(folder/file, hashed=True) for k,file in [('summary','summary.json'),('status','status.json'),('external','external_terminal.json')]})
    return binding, p


def approved_contract(plan_path):
    p = d.read(plan_path); s = p['scientific']; r = d.read(p['root_review'])
    assert p['kind'] == 'GeoBC_R16_frozen_teacher_node_latent_same100_short5_v36'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s) == p['scientific_digest'] == r['scientific_digest']
    assert p['source_pins'] == r['source_pins'] and all(d.sha(k) == h for k,h in p['source_pins'].items())
    assert s['methods'] == METHODS and s['additional_updates_per_arm'] == 1500 and s['additional_passes'] == 5
    assert s['train_CAD_count'] == 100 and s['seed'] == 2026 and s['decoder_chunk_size'] == 32768
    assert s['memory_limit_bytes'] == 30500000000 and s['GPU_lock'] == 'runs/geobc_exploration/.coordinator.lock'
    assert [s[k] for k in ('cache_setup_budget_seconds','arm_setup_budget_seconds','training_budget_seconds_per_arm','TRAIN_readout_budget_seconds_per_arm','arm_budget_seconds','whole_budget_seconds')] == [600,120,900,300,1500,3600]
    assert s['loss'] == 'full_original_node_teacher_latent_relative_squared_equal_case_not_FEM'
    assert s['frozen_groups'] == ['encoder','head'] and s['trainable_groups'] == ['operator']
    assert s['initial_fullTRAIN_latent_readout'] and s['no_epsilon_rescue']
    assert s['optimizer'] == dict(name='Adam',learning_rate=.001,weight_decay=0.,betas=[.9,.999],eps=1e-8,fused=True)
    assert s['precision'] == 'highest FP32; TF32 disabled' and s['DEV_FINAL_reads_allowed'] == s['scaler_fits_allowed'] == 0
    assert s['teacher_prediction_distillation'] and s['teacher_inherited_passes'] == 5
    prior, pp = predecessor_binding()
    teacher, tp, ts, te = terminal_binding('notes/geobc_coarse_FNO100_physical5_v22_plan.json', False)
    assert s['predecessor_terminal'] == prior and s['teacher_binding'] == teacher
    assert all(p['source_pins'].get(k) == h for k,h in pp['source_pins'].items())
    assert all(p['source_pins'].get(k) == h for k,h in tp['source_pins'].items())
    assert s['teacher_checkpoint'] == ts['methods']['FNO16_control']['endpoints']['1500']['checkpoint_identity']
    assert s['teacher_SCI'] == tp['scientific_digest'] and s['teacher_prior_full_order_digest'] == tp['scientific']['full_order_digest']
    assert s['dataset_identity'] == tp['scientific']['dataset_identity'] and s['scaler'] == tp['scientific']['scaler']
    assert d.identity(s['dataset_identity']['path'], hashed=True) == s['dataset_identity']
    manifest = d.read(s['dataset_identity']['path']); keys = manifest['train_keys']['100']
    assert manifest['status'] == 'completed' and manifest['feasibility_pass'] and keys == s['TRAIN_keys'] and len(set(keys)) == len(keys) == 300
    assert all(manifest['case_roles'][k] == 'TRAIN' for k in keys) and manifest['scaler'] == s['scaler']
    from .macro_training import balanced_schedule
    order = balanced_schedule(keys, 5, 2026)
    assert d.digest(order) == s['full_order_digest'] == s['teacher_prior_full_order_digest']
    assert s['initial_steps'] == {m:0 for m in METHODS} and s['endpoint_steps'] == {m:1500 for m in METHODS}
    assert s['fresh_initial_SHA256'] == dict(transolver_latent_teacher='13d31df15c487a1240b8f1f81f0a7ccf7a15a73f32cbbb29fb3dff34632e5abb',FNO16_latent_teacher='0e91784589a573b0d048a738aa17db0b855f88cb5faca72a96533e06f19d6a97')
    assert s['expected_parameter_counts'] == dict(transolver_latent_teacher=1882172,FNO16_latent_teacher=4244244)
    assert not s['fresh5_fair_comparison_claimed'] and not s['pure_architecture_causality_claimed']
    item = s['CPU_fixture']; assert d.identity(item['path'], hashed=True) == item
    proof = d.read(item['path']); assert proof['status'] == 'actual_synthetic_node_latent_CPU_passed_not_GPU' and proof['synthetic_teacher_not_trained']
    assert proof['real_numeric_GT_reads'] == proof['PT_reads'] == proof['GPU_calls'] == proof['optimizer_updates'] == 0
    assert proof['synthetic_nodes'] == 33337 and proof['chunks'] == [32768,569] and proof['zero_teacher_norm_rejected']
    assert proof['source'] == d.identity('geobc/node_latent_teacher_v35.py', hashed=True)
    assert all(v['all_named_operator_gradients_close'] and v['frozen_groups_gradients_absent'] and v['frozen_groups_unchanged'] for v in proof['results'].values())
    assert p['output'] == 'runs/geobc_exploration/node_latent_teacher100_short5_v36'
    return p,r,manifest,order
