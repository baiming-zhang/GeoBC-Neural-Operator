"""Fixed oracle error decomposition of two completed development endpoints."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity

PLAN='notes/geobc_field_amplitude_shape_v75_plan.json'
REVIEW='notes/geobc_field_amplitude_shape_v75_root_review.json'
PARENT='notes/geobc_fno40_pair_readout_v73_plan.json'
SOURCES=('geobc/field_amplitude_shape_contract_v75.py','tools/measure_geobc_field_amplitude_shape_v75.py',
         'tools/run_geobc_field_amplitude_shape_guard_v75.py')


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review'])
    parent,a,e,b=terminal(PARENT)
    assert p['kind']=='GeoBC_existing_predictions_oracle_amplitude_shape_v75'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest']==r['scientific_digest']==d.digest(s)
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins'])
    assert all(d.sha(k)==v for k,v in p['source_pins'].items())
    assert all(p['source_pins'].get(k)==v for k,v in parent['source_pins'].items())
    assert r['parent_terminal_binding']==b and p['parent_terminal_binding']==b
    assert a['protocol_qualified'] and a['prediction_barrier_complete'] and a['committed_finite_predictions']==120
    known=parent['scientific']
    for k in ('pool','case_keys','dataset_identity','scaler','GT_manifest_identity','GPU_lock','memory_limit_bytes'):
        assert s[k]==known[k]
    assert s['candidate_names']==[c['name']for c in known['candidates']]
    assert s['whole_budget_seconds']==300 and s['memory_limit_bytes']==30500000000
    assert s['fixed_candidate_case_denominator']==s['prediction_physical4_reads_planned']==120
    assert s['shared_cached_GT_case_reads_planned']==60
    assert s['original_source_GT_reads_allowed']==s['PT_reads_allowed']==s['new_training_updates_allowed']==s['FINAL_reads_allowed']==0
    assert not s['automatic_replay']
    original=d.read('notes/geobc_field_amplitude_shape_v75_design.json')
    assert original['scientific']==s and original['scientific_digest']==p['scientific_digest']
    barrier=d.read(check_identity(s['prediction_barrier_identity']))
    assert barrier['status']=='completed' and barrier['finite_predictions']==120
    assert barrier['inventory']==a['prediction_inventory']
    expected={n+':'+k for n in s['candidate_names']for k in s['case_keys']}
    assert set(barrier['inventory'])==expected
    manifest=d.read(check_identity(s['dataset_identity']))
    truth=d.read(check_identity(s['GT_manifest_identity']))
    assert s['case_keys']==manifest['development_keys'] and len(set(s['case_keys']))==60
    assert set(truth['records'])==set(s['case_keys']) and truth['status']=='completed'
    assert p['output']=='runs/geobc_exploration/field_amplitude_shape_v75'
    return p,r,manifest,truth,dict(parent_summary=a,inventory=barrier['inventory'])
