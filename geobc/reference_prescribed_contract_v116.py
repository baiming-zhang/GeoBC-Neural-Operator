"""Metadata release for fixed reference spring boundary adaptation."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_reference_prescribed_motion_v116_design.json'
PLAN='notes/geobc_reference_prescribed_motion_v116_plan.json'
REVIEW='notes/geobc_reference_prescribed_motion_v116_root_review.json'
SOURCE_REVIEW='notes/geobc_reference_prescribed_motion_v116_source_review.json'
OUTPUT='runs/geobc_exploration/twoCAD_reference_prescribed_motion_v116'
SOURCES=('geobc/reference_prescribed_motion_v116.py','geobc/reference_constraint_operator_v116.py','geobc/reference_prescribed_contract_v116.py',
 'tools/evaluate_geobc_reference_prescribed_v116.py','tools/guard_geobc_reference_prescribed_v116.py','tools/prepare_geobc_reference_prescribed_v116.py')

def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    _,a,_,b=terminal('notes/geobc_reference_spring_BC_v115_plan.json')
    assert b==s['prior_terminal'] and a['prediction_barrier_complete']and a['spring_BC_gate_passed']and a['committed_finite_endpoints']==6
    assert all(v['rawC_fineK_basis_bytes_unchanged']and v['passed']for v in a['case_records'].values())
    assert d.read(check_identity(s['prior_report_identity']))['PNG_visually_accepted'];check_identity(s['prototype_identity'])
    assert s['selection']['CADs']==['50_380','624_483']and s['budgets']['whole_seconds']==1800
    for cad,row in s['cases'].items():
        d.stat_matches(row['input_cache_identity']);assert row['source_transfer_facts']['REFC']=='123456'
        assert row['source_transfer_facts']['normal_rank']==6 and row['absent_H5_reference_output_row']
        for name,item in row['basis_records'].items():
            rec=d.read(check_identity(item['manifest_identity']));assert rec['physical_U']==item['physical_U']and rec['key']==name
            assert rec['initial_guess']=='independent_zero'and rec['numerical_state']=='converged'and rec['true_relative_residual']<=1e-8
            d.stat_matches(rec['physical_U'])
        for item in row['operator_cache'].values():d.stat_matches(item)
        coarse=d.read(check_identity(row['coarse_cache_identity']));assert coarse['status']=='completed'and coarse['algebra']['passed']and coarse['parallel_M_qualification']['passed']
        for key in ('P','PT','Ac'):
            for item in (coarse['arrays'][key][v]for v in ('rowptr','columns','values')):d.stat_matches(item)
        d.stat_matches(coarse['arrays']['diagonal'])
    return design,b

def approved_contract(path):
    p=d.read(path);r=d.read(p['root_review']);design,b=inputs()
    assert p['kind']=='GeoBC_reference_prescribed_motion_v116'and p['output']==OUTPUT
    assert all(p['scientific'][k]==v for k,v in design['scientific'].items())
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest']and r['predecessor_binding']==b
    assert r['passed']and r['whole_source_reviewed']and r['actual_execution_released']and r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True)
    assert p['source_pins']==r['source_pins']and set(SOURCES)<=set(p['source_pins'])and all(d.sha(k)==v for k,v in p['source_pins'].items())
    return p,r
