"""One fresh official structured-3D TS physical5 candidate, direct metadata."""
from . import available_cad_pilot_data as d
from .global_hash_physical_readout_contract_v18 import complete_physical


def predecessor_binding():
    p=d.read('notes/geobc_node_latent_teacher_v37_plan.json');r=d.read(p['root_review']);folder=d.path(p['output'])
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest'] and p['source_pins']==r['source_pins']
    s=d.read(folder/'summary.json');e=d.read(folder/'external_terminal.json')
    assert s==d.read(folder/'status.json') and s['status']=='completed' and s['protocol_qualified']
    assert s['closed'] and not s['active'] and e['closed'] and not e['active']
    assert e['exit_code']==0 and e['within_external_budget'] and not e['external_timeout']
    assert all(d.exact_process_exited(e[k]) for k in ('child_process','guard_process'))
    assert s['execution']['run_process']==e['child_process'] and s['execution']['guard']['guard_process']==e['guard_process']
    assert s['execution']['scientific_digest']==e['scientific_digest']==p['scientific_digest'] and e['source_pins']==p['source_pins']
    assert s['committed_finite_predictions']==180 and s['owned_GT_array_reads']==s['owned_GT_read_attempts']==60
    assert s['raw_GT_reads']==s['FINAL_numeric_reads']==s['training_updates']==s['scaler_fits']==0
    assert s['prediction_barrier_complete'] and s['teacher_comparisons_complete']
    assert max(s['elapsed_epoch_seconds'],s['elapsed_monotonic_seconds'],e['elapsed_epoch_seconds'],e['elapsed_monotonic_seconds'])<=1800
    assert max(s[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))<=30500000000
    for row in s['candidates'].values():complete_physical(row['full60'],p['scientific']['case_keys'])
    return dict(plan=d.identity('notes/geobc_node_latent_teacher_v37_plan.json',hashed=True),terminal={k:d.identity(folder/file,hashed=True) for k,file in [('summary','summary.json'),('status','status.json'),('external','external_terminal.json')]}),p


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review'])
    assert p['kind']=='GeoBC_official_structured3D_sameR16_freshphysical100_short5_v39'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest']
    assert p['source_pins']==r['source_pins'] and all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert s['methods']==['transolver_structured3d'] and s['seed']==2026 and s['train_CAD_count']==100
    assert s['additional_updates_per_arm']==1500 and s['additional_passes']==5
    assert s['initial_steps']==dict(transolver_structured3d=0) and s['endpoint_steps']==dict(transolver_structured3d=1500)
    assert s['expected_parameter_counts']==dict(transolver_structured3d=4776636)
    assert s['fresh_initial_SHA256']==dict(transolver_structured3d='9634f587089e40d86b1f447c455f1dd3b7d494a59110a842d315e7b626cf990c')
    assert s['allowed_unused_parameters']==['operator.backbone.placeholder']
    assert s['memory_limit_bytes']==30500000000 and s['GPU_lock']=='runs/geobc_exploration/.coordinator.lock'
    assert [s[k] for k in ('cache_setup_budget_seconds','arm_setup_budget_seconds','training_budget_seconds_per_arm','TRAIN_readout_budget_seconds_per_arm','arm_budget_seconds','whole_budget_seconds')]==[600,120,900,300,1500,2100]
    assert s['loss']=='full_original_node_physical_relative_squared_equal_U_S_case'
    assert s['optimizer']==dict(name='Adam',learning_rate=.001,weight_decay=0.,betas=[.9,.999],eps=1e-8,fused=True)
    assert s['precision']=='highest FP32; TF32 disabled' and s['DEV_FINAL_reads_allowed']==s['scaler_fits_allowed']==0
    assert s['teacher_prediction_distillation'] is False and s['inherited_teacher_passes']==0
    assert s['fresh5_physical_exposure_matched'] and not s['pure_architecture_causality_claimed']
    prior,pp=predecessor_binding();assert s['predecessor_terminal']==prior
    assert all(p['source_pins'].get(k)==h for k,h in pp['source_pins'].items())
    assert s['dataset_identity']==pp['scientific']['dataset_identity'] and s['scaler']==pp['scientific']['scaler']
    assert d.identity(s['dataset_identity']['path'],hashed=True)==s['dataset_identity']
    manifest=d.read(s['dataset_identity']['path']);keys=manifest['train_keys']['100']
    assert manifest['status']=='completed' and manifest['feasibility_pass'] and keys==s['TRAIN_keys'] and len(set(keys))==len(keys)==300
    assert all(manifest['case_roles'][k]=='TRAIN' for k in keys) and manifest['scaler']==s['scaler']
    from .macro_training import balanced_schedule
    order=balanced_schedule(keys,5,2026);assert d.digest(order)==s['full_order_digest']==s['arm_full_order_digests']['transolver_structured3d']
    item=s['CPU_fixture'];assert d.identity(item['path'],hashed=True)==item;proof=d.read(item['path'])
    assert proof['status']=='actual_synthetic_structured3D_CPU_equivalence_passed_not_GPU'
    assert proof['actual_GT_reads']==proof['PT_reads']==proof['GPU_calls']==proof['optimizer_updates']==0
    assert proof['source']==d.identity('geobc/structured_transolver_v38.py',hashed=True)
    assert proof['upstream_source']==s['structured_source_identity'] and proof['profile']['parameter_count']==4776636
    assert proof['full_named_operator_gradients_close'] and proof['input_gradients_close'] and proof['encoder_head_initial_SHA_matches_original']
    assert proof['allowed_unused_parameters']==['backbone.placeholder'] and proof['placeholder_correctly_not_added_for_features']
    assert p['output']=='runs/geobc_exploration/structured_TS100_freshphysical5_v39'
    return p,r,manifest,order
