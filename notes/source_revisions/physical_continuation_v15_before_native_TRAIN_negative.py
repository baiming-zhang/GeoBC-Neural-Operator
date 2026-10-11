"""Same-model physical5 to physical10; release waits genuine solver comparison."""
from . import available_cad_pilot_data as d


def approved_contract(plan_path):
    plan=d.read(plan_path);s=plan['scientific'];r=d.read(plan['root_review'])
    assert plan['kind']=='GeoBC_samephysical100_checkpoint5to10_v15'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert plan['scientific_digest']==r['scientific_digest']==d.digest(s)
    assert plan['source_pins']==r['source_pins'] and all(d.sha(p)==h for p,h in plan['source_pins'].items())
    assert s['family'] in ('FNO9','PP256_NATIVE') and r['selected_family']==s['family']
    assert not s['fresh_initialization'] and s['parent_step']==1500 and s['fixed_updates']==3000
    assert s['additional_updates']==1500 and s['endpoint_passes']==[10]
    assert s['memory_limit_bytes']==30500000000 and s['GPU_lock']=='runs/geobc_exploration/.coordinator.lock'
    assert s['cache_setup_budget_seconds']==600 and s['TRAIN_readout_budget_seconds']==300
    assert s['training_budget_seconds']==(900 if s['family']=='FNO9' else 3600)
    assert s['whole_budget_seconds']==(1800 if s['family']=='FNO9' else 4500)
    assert s['precision']=='highest_FP32_TF32_disabled' and s['seed']==2026
    assert s['loss']=='full_original_node_physical_relative_U_S_squared_equal_case'
    assert s['decoder_chunk_size']==32768 and not s['FINAL_reuse_for_selection'] and s['DEV_GT_reads_allowed']==0
    assert s['TRAIN_gate']==dict(primary_absolute_improvement_at_least=.03,maximum_either_field_absolute_deterioration=.02,fixed_denominator=300)
    from .native_physical_readout_contract_v13 import approved_contract as native_readout
    prev,_,manifest,_,native=native_readout('notes/geobc_native_physical100_readout_v13_plan.json')
    assert prev['scientific_digest']==s['native_DEV_SCI']
    for kind,name in [('summary','summary.json'),('status','status.json'),('external','external_terminal.json')]:
        item=s['native_DEV_terminal'][kind]
        assert d.relative(item['path'])==prev['output']+'/'+name and d.identity(item['path'],hashed=True)==item
    dev=d.read(s['native_DEV_terminal']['summary']['path']);ext=d.read(s['native_DEV_terminal']['external']['path'])
    assert dev==d.read(s['native_DEV_terminal']['status']['path']) and dev['status']=='completed' and dev['protocol_qualified']
    assert dev['closed'] and not dev['active'] and ext['closed'] and not ext['active']
    assert ext['exit_code']==0 and ext['within_external_budget'] and all(d.exact_process_exited(ext[k]) for k in ('child_process','guard_process'))
    assert dev['execution']['scientific_digest']==ext['scientific_digest']==s['native_DEV_SCI']
    assert dev['execution']['run_process']==ext['child_process'] and dev['execution']['guard']['guard_process']==ext['guard_process']
    assert dev['committed_finite_predictions']==dev['owned_GT_array_reads']==60
    assert d.identity(s['decision_identity']['path'],hashed=True)==s['decision_identity']
    decision=d.read(s['decision_identity']['path']);assert s['family'] in decision['continuation_families_released']
    assert decision['native_DEV_SCI']==s['native_DEV_SCI']
    if s['family']=='PP256_NATIVE':upstream=native['PP_NATIVE100_PHYSICAL_1500']
    else:
        from .physical_readout_contract_v12 import approved_contract as fno_readout
        _,_,fm,_,fq=fno_readout('notes/geobc_physical100_readout_v12_plan.json')
        assert fm==manifest;upstream=fq['FNO100_PHYSICAL_1500']
    parent=upstream['plan'];ps=parent['scientific']
    assert d.identity(s['parent_training_plan_identity']['path'],hashed=True)==s['parent_training_plan_identity']
    assert d.read(s['parent_training_plan_identity']['path'])==parent and s['parent_training_SCI']==parent['scientific_digest']
    assert s['parent_checkpoint']==upstream['checkpoint'] and s['parent_TRAIN300']==upstream['summary']['endpoints']['1500']['full_TRAIN']
    for field,file,proof in [('parent_summary_identity','summary.json',upstream['summary']),('parent_external_identity','external_terminal.json',upstream['external'])]:
        item=s[field]
        assert d.relative(item['path'])==parent['output']+'/'+file and d.identity(item['path'],hashed=True)==item
        assert d.read(item['path'])==proof
    assert s['parent_case_exposure_counts']==upstream['summary']['case_exposure_counts']=={k:5 for k in ps['TRAIN_keys']}

    assert d.identity(s['parent_norms_identity']['path'],hashed=True)==s['parent_norms_identity']
    assert d.relative(s['parent_norms_identity']['path'])==parent['output']+'/physical_case_norms.json'
    for k in ('family','seed','train_CAD_count','TRAIN_keys','dataset_identity','scaler','optimizer','decoder_chunk_size','loss'):
        assert s[k]==ps[k]
    assert s['train_CAD_count']==100 and len(s['TRAIN_keys'])==300
    assert all(plan['source_pins'].get(p)==h for p,h in prev['source_pins'].items())
    from .macro_training import balanced_schedule
    order=balanced_schedule(s['TRAIN_keys'],10,2026)
    assert d.digest(order[:1500])==ps['full_order_digest']==s['parent_full_order_digest']
    assert d.digest(order)==s['full_order_digest'] and len(order)==3000
    assert plan['output']=='runs/geobc_exploration/macro100_'+s['family']+'_physical10_v15'
    return plan,r,manifest,order
