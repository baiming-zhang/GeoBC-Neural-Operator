"""One known-development readout after actual Hash100 TRAIN improvement."""
from . import available_cad_pilot_data as d
from .global_hash_learning10_contract_v62 import terminal,approved_contract as training_contract,check_identity,complete_physical

TRAIN='notes/geobc_global_hash_learning10_v62_plan.json'
METHOD='GLOBALHASH100_PHYSICAL_3000'
SOURCES=('geobc/global_hash10_readout_contract_v63.py','tools/evaluate_geobc_global_hash10_v63.py',
    'tools/run_geobc_global_hash10_readout_guard_v63.py','tools/prepare_geobc_global_hash10_readout_v63.py')


def training_binding():
    p,_,_,_,parents=training_contract(TRAIN);_,a,e,b=terminal(TRAIN)
    assert a['training_complete'] and a['TRAIN_gate']['passed'] and a['step']==3000
    assert a['additional_optimizer_step_attempts']==a['additional_optimizer_step_completions']==1500
    assert a['case_exposure_counts']=={k:10 for k in p['scientific']['TRAIN_keys']}
    assert not a['possible_partial_optimizer_update'] and a['typed_model_Adam_restore_verified']
    assert a['parent_preupdate_fullN_replay']['passed'] and a['actual_largest_fullN_gradient_check']['passed']
    complete_physical(a['endpoints']['3000']['full_TRAIN'],p['scientific']['TRAIN_keys'])
    ck=a['endpoints']['3000']['checkpoint_identity'];d.stat_matches(ck)
    report=d.read('results/geobc_global_hash_learning10_v62/manifest.json')
    assert report['status']=='completed' and report['SCI']==p['scientific_digest'] and report['terminal_binding']==b
    hp,ha,he,hb=terminal('notes/geobc_global_hash_physical_readout_v18_plan.json')
    fp,fa,fe,fb=terminal('notes/geobc_fno20_readout_v57_plan.json')
    old=hp['scientific'];known=fp['scientific']
    for name in ('case_keys','dataset_identity','scaler','GT_manifest_identity','GT_policy','pool'):
        assert old[name]==known[name]
    assert old['dataset_identity']==p['scientific']['dataset_identity'] and old['scaler']==p['scientific']['scaler']
    hash_names=[n for n,r in ha['candidates'].items()if r['family']=='FNO_GLOBAL_HASH']
    assert len(hash_names)==1
    refs={hash_names[0]:ha['candidates'][hash_names[0]],**fa['candidates']}
    assert len(refs)==3
    for row in refs.values():complete_physical(row['full60'],old['case_keys'])
    costs=dict(training_epoch=a['trajectory_training_epoch_seconds'],training_monotonic=a['trajectory_training_monotonic_seconds'],
        stage_epoch=parents['parent_external']['elapsed_epoch_seconds']+e['elapsed_epoch_seconds'],
        stage_monotonic=parents['parent_external']['elapsed_monotonic_seconds']+e['elapsed_monotonic_seconds'],
        old_technical_failure_and_waits=parents['parent_summary']['technical_recovery_cost'])
    return dict(plan=p,summary=a,external=e,binding=b,checkpoint=ck,known=old,references=refs,
        Hash5_name=hash_names[0],reference_bindings={'Hash5':hb,'FNO':fb},reference_plans=[hp,fp],complete0to10_cost=costs)


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);up=training_binding()
    assert p['kind']=='GeoBC_globalHash_physical10_knownDEV60_v63'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest']==r['scientific_digest']==d.digest(s) and p['source_pins']==r['source_pins']
    assert set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==h for k,h in p['source_pins'].items())
    for upstream in [up['plan']]+up['reference_plans']:
        assert all(p['source_pins'].get(k)==h for k,h in upstream['source_pins'].items())
    assert r['training_bindings']=={METHOD:up['binding']} and r['reference_bindings']==up['reference_bindings']
    for name in ('GPU_lock','memory_limit_bytes','seed','decoder_chunk_size','GT_policy','pool','dataset_identity','case_keys','scaler','GT_manifest_identity'):
        assert s[name]==up['known'][name]
    assert s['memory_limit_bytes']==30500000000 and s['whole_budget_seconds']==600
    assert s['training_SCI']==up['plan']['scientific_digest'] and s['fixed_prediction_denominator']==60
    assert not s['FINAL_reuse_for_selection'] and not s['automatic_continuation']
    assert s['candidates']==[dict(name=METHOD,family='FNO_GLOBAL_HASH',step=3000,training_plan_identity=d.identity(TRAIN,hashed=True))]
    assert s['reference_bindings']==up['reference_bindings'] and s['Hash5_name']==up['Hash5_name']
    assert s['DEV_gate']==dict(primary_absolute_improvement_at_least=.03,maximum_either_field_absolute_deterioration=.02,hotspot_deterioration_max=.02,fixed_denominator=60)
    m=d.read(check_identity(s['dataset_identity']));keys=s['case_keys'];assert keys==m['development_keys'] and len(set(keys))==60
    cads={k.split('/')[0]for k in keys};assert len(cads)==20 and set(keys)=={c+'/'+l for c in cads for l in ('ver','hor','tor')}
    assert all(m['case_roles'][k]=='DEV'for k in keys)
    truths=d.read(check_identity(s['GT_manifest_identity']))
    assert truths['status']=='completed' and truths['source_GT_parses']==20 and set(truths['records'])==set(keys)
    for k in keys:
        t=truths['records'][k];assert t['all_original_nodes'] and t['signed_stress_unchanged']
        assert t['physical']['shape']==[m['records'][k]['num_nodes'],4]
        assert not {'target_path','physical_path'}&set(m['records'][k]['file_identities'])
    assert p['output']=='runs/geobc_exploration/GlobalHash100_knownDEV60_physical10_v63'
    return p,r,m,truths,{METHOD:up}
