"""All60 knownDEV predictions before GT, even when completed TRAIN accuracy is poor."""
from . import available_cad_pilot_data as d
from .transolver_macro100_contract_v66 import terminal,approved_contract as train_contract,complete_physical,check_identity

TRAIN='notes/geobc_transolver_macro100_v66_plan.json'
METHOD='TS_BF16_CONTINUOUS100_PHYSICAL_1500'
REFERENCE_PLANS={'five':'notes/geobc_global_hash_physical_readout_v18_plan.json',
    'FNO20':'notes/geobc_fno20_readout_v57_plan.json','Hash10':'notes/geobc_global_hash10_readout_v63_plan.json',
    'TSdirect':'notes/geobc_original_TS_v51_plan.json'}
SOURCES=('geobc/transolver_macro100_readout_contract_v67.py','tools/evaluate_geobc_transolver_macro100_v67.py',
    'tools/run_geobc_transolver_macro100_readout_guard_v67.py','tools/prepare_geobc_transolver_macro100_readout_v67.py')


def training_binding():
    p,_,_,_=train_contract(TRAIN);_,a,e,b=terminal(TRAIN);s=p['scientific'];family=s['families'][0];row=a['models'][family]
    assert a['training_complete'] and a['optimizer_step_attempts']==a['optimizer_step_completions']==1500
    assert not a['possible_partial_optimizer_update'] and not a['optimizer_step_in_progress']
    assert row['case_exposure_counts']=={k:5 for k in s['TRAIN_keys']} and row['optimizer_step']==1500
    complete_physical(row['endpoints']['5'],s['TRAIN_keys']);ck=row['checkpoint_identity'];d.stat_matches(ck)
    report=d.read('results/geobc_transolver_macro100_v66/manifest.json')
    assert report['status']=='completed' and report['SCI']==p['scientific_digest'] and report['terminal_binding']==b
    refs={};bindings={};plans=[];known=None
    wanted={'five':{'GLOBALHASH100_PHYSICAL_1500','FNO100_PHYSICAL_1500'},
        'FNO20':{'FNO100_PHYSICAL_3000','FNO100_PHYSICAL_6000'},'Hash10':{'GLOBALHASH100_PHYSICAL_3000'},
        'TSdirect':{'R16_TRANSOLVER_1500','R16_TRANSOLVER_PHYSICAL_3000'}}
    for label,path in REFERENCE_PLANS.items():
        rp,ra,re,rb=terminal(path);rs=rp['scientific'];plans.append(rp);bindings[label]=rb
        if known is None:known=rs
        for k in ('case_keys','dataset_identity','scaler','GT_manifest_identity','GT_policy','pool'):assert rs[k]==known[k]
        assert rs['dataset_identity']==s['dataset_identity'] and rs['scaler']==s['scaler']
        assert wanted[label]<=set(ra['candidates'])
        for name in sorted(wanted[label]):
            metric=ra['candidates'][name];complete_physical(metric['full60'],rs['case_keys'])
            assert name not in refs;refs[name]=metric
    assert len(refs)==7
    return dict(plan=p,summary=a,external=e,binding=b,checkpoint=ck,known=known,references=refs,
        reference_bindings=bindings,reference_plans=plans,family=family,
        same5_reference_name='GLOBALHASH100_PHYSICAL_1500')


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);up=training_binding()
    assert p['kind']=='GeoBC_originalTS_BF16_continuous100_physical5_knownDEV60_v67'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest']==r['scientific_digest']==d.digest(s) and p['source_pins']==r['source_pins']
    assert set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert r['training_bindings']=={METHOD:up['binding']} and r['reference_bindings']==up['reference_bindings']
    for parent in [up['plan']]+up['reference_plans']:
        assert all(p['source_pins'].get(k)==h for k,h in parent['source_pins'].items())
    for name in ('GPU_lock','memory_limit_bytes','seed','decoder_chunk_size','GT_policy','pool','dataset_identity','case_keys','scaler','GT_manifest_identity'):
        assert s[name]==up['known'][name]
    assert s['memory_limit_bytes']==30500000000 and s['whole_budget_seconds']==600
    assert s['training_SCI']==up['plan']['scientific_digest'] and s['fixed_prediction_denominator']==60
    assert s['candidates']==[dict(name=METHOD,family=up['family'],step=1500,training_plan_identity=d.identity(TRAIN,hashed=True))]
    assert s['reference_bindings']==up['reference_bindings'] and s['same5_reference_name']==up['same5_reference_name']
    assert s['DEV_gate']==dict(primary_absolute_improvement_at_least=.03,maximum_either_field_absolute_deterioration=.02,hotspot_deterioration_max=.02,fixed_denominator=60)
    assert not s['FINAL_reuse_for_selection'] and not s['automatic_continuation']
    m=d.read(check_identity(s['dataset_identity']));keys=s['case_keys'];assert keys==m['development_keys'] and len(set(keys))==60
    cads={k.split('/')[0]for k in keys};assert len(cads)==20 and set(keys)=={c+'/'+l for c in cads for l in ('ver','hor','tor')}
    assert all(m['case_roles'][k]=='DEV'for k in keys)
    truths=d.read(check_identity(s['GT_manifest_identity']))
    assert truths['status']=='completed' and truths['source_GT_parses']==20 and set(truths['records'])==set(keys)
    for k in keys:
        t=truths['records'][k];assert t['all_original_nodes'] and t['signed_stress_unchanged'] and t['physical']['shape']==[m['records'][k]['num_nodes'],4]
        assert not {'target_path','physical_path'}&set(m['records'][k]['file_identities'])
    assert p['output']=='runs/geobc_exploration/Transolver_BF16_continuous100_knownDEV60_physical5_v67'
    return p,r,m,truths,{METHOD:up}
