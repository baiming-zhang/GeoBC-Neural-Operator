"""New evidence-gated modest continuation; original five-pass gates preserved."""
import random
from . import available_cad_pilot_data as d
from .shared_decoder_pair_contract_v61 import terminal,complete_physical,check_identity,MODEL_SOURCES

PARENT='notes/geobc_global_hash_physical_recovery_v18_plan.json'
FNO10='notes/geobc_macro100_FNO9_physical10_v15_plan.json'
SOURCES=('geobc/physical_target_only_cache_v62.py','geobc/global_hash_learning10_contract_v62.py',
    'tools/train_geobc_global_hash_learning10_v62.py','tools/run_geobc_global_hash_learning10_guard_v62.py',
    'tools/prepare_geobc_global_hash_learning10_v62.py')


def order(keys):
    rng=random.Random(2026);result=[]
    for _ in range(10):
        part=list(keys);rng.shuffle(part);result.extend(part)
    return result


def inputs():
    pp,pa,pe,pb=terminal(PARENT);sp,sa,se,sb=terminal('notes/geobc_shared_decoder_v61_plan.json')
    fp,fa,fe,fb=terminal(FNO10);keys=pp['scientific']['TRAIN_keys'];ps=pp['scientific']
    assert sa['training_complete'] and sa['optimizer_step_completions']==6000
    report=d.read('results/geobc_shared_decoder_v61/manifest.json')
    assert report['status']=='completed' and report['SCI']==sp['scientific_digest'] and report['terminal_binding']==sb
    assert sa['aggregate_endpoints']['FNO_GLOBAL_HASH']['1000']['case_mean_stress_relative_l2']<.01
    assert ps['family']=='FNO_GLOBAL_HASH' and pa['training_complete'] and pa['step']==1500
    assert pa['optimizer_step_attempts']==pa['optimizer_step_completions']==1500 and pa['case_exposure_counts']=={key:5 for key in keys}
    assert pa['saved_Adam_steps_equal_cursor'] and not pa['possible_partial_optimizer_update'] and not pa['optimizer_step_in_progress']
    assert fa['training_complete'] and fa['step']==3000 and fa['case_exposure_counts']=={key:10 for key in keys}
    for name in ('TRAIN_keys','dataset_identity','scaler','optimizer','loss'):assert fp['scientific'][name]==ps[name]
    assert keys==sp['scientific']['parent_TRAIN_keys'] and len(keys)==300
    complete_physical(pa['endpoints']['1500']['full_TRAIN'],keys);complete_physical(fa['endpoints']['3000']['full_TRAIN'],keys)
    stat=pa['last_checkpoint']['stat_identity'];d.stat_matches(stat)
    norms=d.identity(d.path(pp['output'])/'physical_case_norms.json',hashed=True)
    return dict(parent=pp,parent_summary=pa,parent_external=pe,parent_binding=pb,shared=sp,shared_summary=sa,shared_binding=sb,
        FNO=fp,FNO_summary=fa,FNO_binding=fb,checkpoint_stat=stat,norms_identity=norms,keys=keys)


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);up=inputs();ps=up['parent']['scientific']
    assert p['kind']=='GeoBC_globalHash_evidence_gated_physical5_to10_v62'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest']==r['scientific_digest']==d.digest(s) and p['source_pins']==r['source_pins']
    assert set(SOURCES+MODEL_SOURCES)<=set(p['source_pins']) and all(d.sha(k)==h for k,h in p['source_pins'].items())
    for name in ('parent','shared','FNO'):
        assert r['upstream_terminal_bindings'][name]==up[name+'_binding']
        assert all(p['source_pins'].get(k)==h for k,h in up[name]['source_pins'].items())
    for name in ('family','seed','train_CAD_count','TRAIN_keys','dataset_identity','scaler','memory_limit_bytes','GPU_lock','loss','optimizer','decoder_chunk_size'):
        assert s[name]==ps[name]
    assert s['parameter_count']==12245894 and s['parent_step']==1500 and s['additional_updates']==1500 and s['fixed_updates']==3000
    assert s['endpoint_passes']==[10] and not s['fresh_initialization'] and not s['automatic_continuation']
    assert s['cache_setup_budget_seconds']==s['training_budget_seconds']==600 and s['TRAIN_readout_budget_seconds']==300 and s['whole_budget_seconds']==1800
    assert s['parent_checkpoint_stat']==up['checkpoint_stat'] and s['parent_training_SCI']==up['parent']['scientific_digest']
    assert s['parent_norms_identity']==up['norms_identity'] and s['parent_case_exposure_counts']==up['parent_summary']['case_exposure_counts']
    assert s['parent_TRAIN300']==up['parent_summary']['endpoints']['1500']['full_TRAIN']
    assert s['same10_FNO_TRAIN300']==up['FNO_summary']['endpoints']['3000']['full_TRAIN']
    assert s['TRAIN_gate']==dict(own5_to10_S_absolute_improvement_min=.03,U_deterioration_max=.02,fixed_denominator=300)
    assert s['DEV_GT_reads_allowed']==0 and not s['FINAL_reuse_for_selection'] and s['Hash_architecture_hyperparameter_changes']==0
    rows=order(up['keys']);assert d.digest(rows)==s['full_order_digest'] and d.digest(rows[:1500])==s['parent_full_order_digest']==ps['full_order_digest']
    assert p['output']=='runs/geobc_exploration/GlobalHash100_physical10_v62'
    m=d.read(check_identity(s['dataset_identity']));assert m['train_keys']['100']==up['keys']
    assert s['parent_replay_fixed_key']==max(up['keys'],key=lambda k:m['records'][k]['num_nodes'])
    return p,r,m,rows,up
