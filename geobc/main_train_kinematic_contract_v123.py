"""Actual complete source100 predecessor binds one main TRAIN kinematic production attempt."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_main_train_kinematic_inputs_v123_design.json'
PLAN='notes/geobc_main_train_kinematic_inputs_v123_plan.json'
REVIEW='notes/geobc_main_train_kinematic_inputs_v123_root_review.json'
SOURCE_REVIEW='notes/geobc_main_train_kinematic_inputs_v123_source_review.json'
OUTPUT='runs/geobc_exploration/main_TRAIN100_physical_U_grad_v123'
SOURCES=('geobc/main_train_physics_lanes_v123.py','geobc/main_train_kinematic_io_v123.py','geobc/main_train_kinematic_contract_v123.py','tools/run_geobc_main_train_kinematic_v123.py','tools/prepare_geobc_main_train_kinematic_v123.py','tools/guard_geobc_main_train_kinematic_v123.py')
def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    p,a,e,b=terminal('notes/geobc_main_train_source_qualification_v122_plan.json');assert a['source_qualification_passed'] and a['source_barrier_complete'] and a['committed_CADs']==100 and b==s['prior_terminal']
    assert d.read(check_identity(s['prior_report_identity']))['PNG_visually_accepted']
    assert list(a['case_records'])==s['selection']['CADs'] and len(s['legacy'])==4 and len(s['selection']['CADs'])==100
    for cad,row in a['case_records'].items():check_identity(row['manifest_identity']);d.stat_matches(row['input_identity'])
    for cad,row in s['legacy'].items():
        q=d.read(check_identity(row['VER_summary_identity']));assert q['numerical_state']=='converged' and q['true_relative_residual']<=1e-8 and q['prediction']==row['VER_prediction'];d.stat_matches(row['VER_prediction'])
        op=d.read(check_identity(row['operator_manifest_identity']));assert op['feasibility_pass'] and op['algebra_guards']['passed'] and op['arrays']==row['operator_arrays']
        for item in row['operator_arrays'].values():d.stat_matches(item)
        if row.get('coarse_manifest_identity'):
            cm=d.read(check_identity(row['coarse_manifest_identity']));assert cm['status']=='completed' and cm['algebra_guards']['passed']
            for key,item in cm['arrays'].items():
                for v in ([item[x] for x in ('rowptr','columns','values')] if 'rowptr' in item else [item]):d.stat_matches(v)
    return design,a,b
def approved_contract(path):
    p=d.read(path);r=d.read(p['root_review']);design,a,b=inputs()
    assert path==PLAN and p['output']==OUTPUT and p['scientific']==design['scientific'] and d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest']
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released'] and r['predecessor_binding']==b
    assert r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True) and p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins'])
    assert all(d.sha(k)==v for k,v in p['source_pins'].items())
    return p,r,a
