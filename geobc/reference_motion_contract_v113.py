"""Metadata-only release for fixed reference motion and work-conjugate energy."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal, check_identity
DESIGN='notes/geobc_reference_motion_response_v113_design.json'
PLAN='notes/geobc_reference_motion_response_v113_plan.json'
REVIEW='notes/geobc_reference_motion_response_v113_root_review.json'
SOURCE_REVIEW='notes/geobc_reference_motion_response_v113_source_review.json'
OUTPUT='runs/geobc_exploration/twoCAD_reference_motion_response_v113'
SOURCES=('geobc/reference_motion_response_v113.py','geobc/reference_motion_contract_v113.py',
 'tools/evaluate_geobc_reference_motion_v113.py','tools/guard_geobc_reference_motion_v113.py','tools/prepare_geobc_reference_motion_v113.py')

def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    _,a,_,b=terminal('notes/geobc_physical_U_reserve_accuracy_v112_plan.json')
    assert b==s['prior_terminal'] and a['original_U_accuracy_gate_passed']
    assert d.read(check_identity(s['prior_report_identity']))['PNG_visually_accepted']
    _,physics,_,pb=terminal('notes/geobc_physical_U_new_geometry_deployment_v111_plan.json')
    assert pb==s['physics_terminal'] and physics['deployment_gate_passed'] and physics['GT_reads']==0
    check_identity(s['prototype_identity'])
    assert s['selection']['CADs']==['50_380','624_483'] and s['budgets']['whole_seconds']==900
    for cad,row in s['cases'].items():
        d.stat_matches(row['input_cache_identity'])
        assert row['source_transfer_facts']['REFC']=='123456' and row['source_transfer_facts']['normal_rank']==6
        for name,record in row['basis_records'].items():
            item=record['manifest_identity'];rec=d.read(check_identity(item))
            group='canonical_basis' if name in ('ver','hor','tor') else 'added_basis'
            assert item==physics['prediction_inventory'][cad+'/'+group+'/'+name]
            assert rec['key']==name and rec['physical_U']==record['physical_U']
            assert rec['initial_guess']=='independent_zero' and rec['numerical_state']=='converged' and rec['true_relative_residual']<=1e-8
            d.stat_matches(rec['physical_U'])
        for name,item in row['independent_mixed_manifest_identities'].items():
            assert item==physics['prediction_inventory'][cad+'/independent_mixed/'+name]
            rec=d.read(check_identity(item));assert rec['key']==name and rec['initial_guess']=='independent_zero'
            assert rec['numerical_state']=='converged' and rec['true_relative_residual']<=1e-8
            d.stat_matches(rec['physical_U'])
    return design,b

def approved_contract(path):
    p=d.read(path);r=d.read(p['root_review']);design,b=inputs()
    assert p['kind']=='GeoBC_reference_motion_response_v113' and p['output']==OUTPUT
    assert all(p['scientific'][k]==v for k,v in design['scientific'].items())
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest'] and r['predecessor_binding']==b
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True)
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==v for k,v in p['source_pins'].items())
    return p,r
