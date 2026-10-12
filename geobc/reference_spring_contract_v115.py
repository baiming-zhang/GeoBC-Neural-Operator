"""Metadata release for fixed reference spring boundary adaptation."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_reference_spring_BC_v115_design.json'
PLAN='notes/geobc_reference_spring_BC_v115_plan.json'
REVIEW='notes/geobc_reference_spring_BC_v115_root_review.json'
SOURCE_REVIEW='notes/geobc_reference_spring_BC_v115_source_review.json'
OUTPUT='runs/geobc_exploration/twoCAD_reference_spring_BC_v115'
SOURCES=('geobc/reference_spring_response_v115.py','geobc/reference_spring_operator_v115.py','geobc/reference_spring_contract_v115.py',
 'tools/evaluate_geobc_reference_spring_v115.py','tools/guard_geobc_reference_spring_v115.py','tools/prepare_geobc_reference_spring_v115.py')

def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    _,a,_,b=terminal('notes/geobc_reference_motion_recovery_v114_plan.json')
    assert b==s['prior_terminal'] and a['prediction_barrier_complete']and a['independent_mixed_array_reads']==6
    assert all(v['rawC_input_basis_bytes_unchanged']and all(x['passed']for x in v['gradient_checks'])for v in a['case_records'].values())
    assert not a['reference_motion_gate_passed']and all(not v['canonical_derived_translation_witness']['available']for v in a['case_records'].values())
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
    assert p['kind']=='GeoBC_reference_spring_BC_v115'and p['output']==OUTPUT
    assert all(p['scientific'][k]==v for k,v in design['scientific'].items())
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest']and r['predecessor_binding']==b
    assert r['passed']and r['whole_source_reviewed']and r['actual_execution_released']and r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True)
    assert p['source_pins']==r['source_pins']and set(SOURCES)<=set(p['source_pins'])and all(d.sha(k)==v for k,v in p['source_pins'].items())
    return p,r
