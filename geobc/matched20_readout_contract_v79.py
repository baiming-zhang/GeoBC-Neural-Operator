"""Two fixed20 endpoints, genuine training terminal first; metadata only."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,complete_physical,check_identity

TRAIN='notes/geobc_matched20_solver_v78_plan.json'
REFERENCE='notes/geobc_fno40_pair_readout_v73_plan.json'
PLAN='notes/geobc_matched20_readout_v79_plan.json'
REVIEW='notes/geobc_matched20_readout_v79_root_review.json'
SOURCE_REVIEW='notes/geobc_matched20_readout_v79_source_review.json'
SOURCES=('geobc/matched20_readout_contract_v79.py','tools/evaluate_geobc_matched20_v79.py',
    'tools/run_geobc_matched20_readout_guard_v79.py','tools/prepare_geobc_matched20_readout_v79.py')


def inputs():
    p,a,e,b=terminal(TRAIN);rp,ra,re,rb=terminal(REFERENCE)
    s,known=p['scientific'],rp['scientific']
    assert a['training_complete'] and a['protocol_qualified'] and not a['possible_partial_optimizer_update'] and not a['optimizer_step_in_progress']
    assert a['additional_optimizer_step_attempts']==a['additional_optimizer_step_completions']==7500
    assert a['cache_receipt']['physical_TRAIN_target_reads']==a['cache_receipt']['physical_TRAIN_target_read_attempts']==300
    assert a['resident_physical_target_reuses']==902
    assert a['raw_GT_reads']==a['DEV_FINAL_reads']==a['scaler_fits']==a['cache_receipt']['encoded_TRAIN_target_reads']==0
    assert s['dataset_identity']==known['dataset_identity'] and s['scaler']==known['scaler']
    assert ra['protocol_qualified'] and ra['prediction_barrier_complete']
    for z in ra['candidates'].values():complete_physical(z['full60'],known['case_keys'])
    qualified={}
    for arm in s['arms']:
        name=arm['name'];state=a['arms'][name]
        assert state['status']=='completed' and state['step']==6000 and state['case_exposure_counts']=={k:20 for k in s['TRAIN_keys']}
        assert state['restored_model_Adam_fourRNG_cursor_exposure'] and state['qualification']['passed']
        complete_physical(state['endpoints']['6000']['full_TRAIN'],s['TRAIN_keys'])
        ck=state['endpoints']['6000']['checkpoint_identity'];d.stat_matches(ck)
        assert d.relative(ck['path'])==p['output']+'/'+name+'/step6000.pt'
        qualified[name]=dict(plan=p,summary=a,external=e,binding=b,checkpoint=ck,arm=arm,state=state)
    return dict(plan=p,summary=a,external=e,binding=b,reference_plan=rp,reference_summary=ra,reference_binding=rb,known=known,qualified=qualified)


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);up=inputs()
    assert p['kind']=='GeoBC_matched20_Hash_TS_knownDEV60_v79'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest'] and p['source_pins']==r['source_pins']
    assert set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==h for k,h in p['source_pins'].items())
    for k in ('GPU_lock','memory_limit_bytes','seed','decoder_chunk_size','GT_policy','pool','dataset_identity','case_keys','scaler','GT_manifest_identity'):
        assert s[k]==up['known'][k]
    assert s['whole_budget_seconds']==900 and s['fixed_prediction_denominator']==120 and s['memory_limit_bytes']==30500000000
    assert s['training_SCI']==up['plan']['scientific_digest'] and s['reference_SCI']==up['reference_plan']['scientific_digest']
    assert s['reference_binding']==r['reference_binding']==up['reference_binding']
    assert s['candidates']==[dict(name=x['name'],family=x['family'],step=6000,training_plan_identity=d.identity(TRAIN,hashed=True)) for x in up['plan']['scientific']['arms']]
    assert r['training_bindings']=={name:up['binding']for name in up['qualified']}
    assert not s['FINAL_reuse_for_selection'] and not s['automatic_continuation'] and not s['TRAIN_gate_filters_candidates']
    assert s['DEV_gate']==up['plan']['scientific']['comparative_gate']
    m=d.read(check_identity(s['dataset_identity']));keys=s['case_keys']
    assert keys==m['development_keys'] and len(keys)==len(set(keys))==60
    assert all(m['case_roles'][k]=='DEV'for k in keys)
    assert not {k.split('/')[0]for k in keys}&{k.split('/')[0]for k in up['plan']['scientific']['TRAIN_keys']}
    truth=d.read(check_identity(s['GT_manifest_identity']))
    assert truth['status']=='completed' and truth['source_GT_parses']==20 and set(truth['records'])==set(keys)
    for k in keys:
        t=truth['records'][k]
        assert t['all_original_nodes'] and t['signed_stress_unchanged'] and t['physical']['shape']==[m['records'][k]['num_nodes'],4]
        assert not {'target_path','physical_path'}&set(m['records'][k]['file_identities'])
    assert p['output']=='runs/geobc_exploration/Hash_TS_matched20_knownDEV60_v79'
    return p,r,m,truth,up['qualified']
