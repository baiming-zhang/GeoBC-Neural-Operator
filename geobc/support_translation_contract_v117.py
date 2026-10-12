"""Metadata release for fixed reference spring boundary adaptation."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_support_translation_port_v117_design.json'
PLAN='notes/geobc_support_translation_port_v117_plan.json'
REVIEW='notes/geobc_support_translation_port_v117_root_review.json'
SOURCE_REVIEW='notes/geobc_support_translation_port_v117_source_review.json'
OUTPUT='runs/geobc_exploration/twoCAD_support_translation_port_v117'
SOURCES=('geobc/support_translation_response_v117.py','geobc/support_translation_pcg_v117.py','geobc/support_translation_contract_v117.py',
 'tools/evaluate_geobc_support_translation_v117.py','tools/guard_geobc_support_translation_v117.py','tools/prepare_geobc_support_translation_v117.py')

def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    _,a,_,b=terminal('notes/geobc_reference_prescribed_motion_v116_plan.json')
    assert b==s['prior_terminal'] and a['prediction_barrier_complete']and a['prescribed_motion_gate_passed']and a['committed_finite_endpoints']==6
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
    for row in s['cases'].values():
        facts=row['source_support_constraints'];refs=facts['support_reference_GRID_IDs']
        assert refs==sorted(refs)and len(set(refs))==4 and facts['actual_card_CM']==['123456']*4
        assert row['selected_support_reference_GRID_ID']==refs[0]and row['actual_source_region_key']=='region_support_0'
    return design,b

def approved_contract(path):
    p=d.read(path);r=d.read(p['root_review']);design,b=inputs()
    assert p['kind']=='GeoBC_support_translation_port_v117'and p['output']==OUTPUT
    assert all(p['scientific'][k]==v for k,v in design['scientific'].items())
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest']and r['predecessor_binding']==b
    assert r['passed']and r['whole_source_reviewed']and r['actual_execution_released']and r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True)
    assert p['source_pins']==r['source_pins']and set(SOURCES)<=set(p['source_pins'])and all(d.sha(k)==v for k,v in p['source_pins'].items())
    return p,r
