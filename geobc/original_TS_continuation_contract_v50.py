"""One bounded original TS5-to10 continuation, direct real terminal bindings."""
from . import available_cad_pilot_data as d
from .global_bc_training_contract_v48 import terminal,original_TS_binding
from .global_hash_physical_readout_contract_v18 import check_identity,complete_physical

METHOD='R16_TRANSOLVER_PHYSICAL_3000'


def predecessor_binding():
    b,p,a,e=terminal('notes/geobc_global_BC_v49_plan.json')
    assert a['protocol_qualified'] and a['committed_finite_predictions']==120 and a['owned_GT_array_reads']==a['owned_GT_read_attempts']==60
    assert a['raw_GT_reads']==a['FINAL_numeric_reads']==a['training_updates']==a['scaler_fits']==0
    for row in a['candidates'].values():complete_physical(row['full60'],p['scientific']['case_keys'])
    return b,p


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review'])
    assert p['kind']=='GeoBC_original_R16_TS_ownphysical5to10_v50'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest'] and p['source_pins']==r['source_pins']
    assert all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert s['family']=='R16_transolver' and s['method']=='transolver' and s['candidate_name']==METHOD
    assert s['seed']==2026 and s['train_CAD_count']==100 and not s['fresh_initialization']
    assert s['parent_step']==1500 and s['additional_updates']==1500 and s['fixed_updates']==3000 and s['endpoint_passes']==[10]
    assert s['memory_limit_bytes']==30500000000 and s['GPU_lock']=='runs/geobc_exploration/.coordinator.lock'
    assert [s[k] for k in ('cache_setup_budget_seconds','qualification_budget_seconds','training_budget_seconds','TRAIN_readout_budget_seconds','whole_budget_seconds')]==[600,120,900,300,2100]
    assert s['precision']=='highest_FP32_TF32_disabled' and s['decoder_chunk_size']==32768
    assert s['loss']=='full_original_node_physical_relative_U_S_squared_equal_case'
    assert s['DEV_GT_reads_allowed']==s['scaler_fits_allowed']==0 and not s['FINAL_reuse_for_selection']
    assert not s['automatic40'] and not s['automatic24000'] and not s['GBC_branch']
    assert s['expected_parameter_count']==1882172 and s['allowed_unused_parameters']==[]
    assert s['TRAIN_gate']==dict(primary_absolute_improvement_at_least=.03,stress_absolute_improvement_at_least=.03,maximum_either_field_absolute_deterioration=.02,fixed_denominator=300)
    b,pp=predecessor_binding();assert s['predecessor_terminal']==b
    parent,tp=original_TS_binding();assert s['parent_terminal']==parent
    a=d.read(parent['terminal']['summary']['path']);arm=a['methods']['transolver'];f=d.path(tp['output'])
    assert s['parent_training_plan_identity']==parent['plan'] and s['parent_training_SCI']==tp['scientific_digest']
    assert s['parent_summary_identity']==d.identity(f/'transolver'/'summary.json',hashed=True)
    assert s['parent_external_identity']==parent['terminal']['external']
    assert d.read(s['parent_summary_identity']['path'])==arm
    assert s['parent_checkpoint']==arm['endpoints']['1500']['checkpoint_identity'];d.stat_matches(s['parent_checkpoint'])
    assert s['parent_norms_identity']==d.identity(f/'physical_case_norms.json',hashed=True)
    assert s['parent_case_exposure_counts']==arm['case_exposure_counts']=={k:5 for k in tp['scientific']['TRAIN_keys']}
    assert s['parent_TRAIN300']==arm['endpoints']['1500']['full_TRAIN']
    assert s['old_shared_cache_epoch_seconds']==a['cache_setup_epoch_seconds'] and s['old_shared_cache_monotonic_seconds']==a['cache_setup_monotonic_seconds']
    for k in ('seed','train_CAD_count','TRAIN_keys','dataset_identity','scaler','optimizer','decoder_chunk_size','loss'):
        assert s[k]==tp['scientific'][k]
    assert s['dataset_identity']==pp['scientific']['dataset_identity'] and s['scaler']==pp['scientific']['scaler']
    m=d.read(check_identity(s['dataset_identity']));keys=m['train_keys']['100']
    assert m['status']=='completed' and m['feasibility_pass'] and keys==s['TRAIN_keys'] and len(keys)==len(set(keys))==300
    assert all(m['case_roles'][k]=='TRAIN' for k in keys) and m['scaler']==s['scaler']
    from .macro_training import balanced_schedule
    order=balanced_schedule(keys,10,2026)
    assert len(order)==3000 and d.digest(order[:1500])==s['parent_full_order_digest']==tp['scientific']['full_order_digest']
    assert d.digest(order)==s['full_order_digest']
    proof=d.read(check_identity(s['scalar_diagnostic']))
    assert proof['status']=='actual_CPU_scalar_loss_only_diagnostic' and proof['new_GT_PT_array_GPU_reads']==0
    assert proof['curves']['original_TS5'][-1]<proof['curves']['original_TS5'][-2]
    assert p['output']=='runs/geobc_exploration/original_R16_TS100_physical10_v50'
    return p,r,m,order
