"""Metadata-only release for an immutable physical U cache package."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_physical_U_response_package_v110_design.json'
PLAN='notes/geobc_physical_U_response_package_v110_plan.json'
REVIEW='notes/geobc_physical_U_response_package_v110_root_review.json'
SOURCE_REVIEW='notes/geobc_physical_U_response_package_v110_source_review.json'
DESCRIPTORS='notes/geobc_physical_U_response_package_v110_descriptors.json'
OUTPUT='runs/geobc_exploration/threeCAD_physical_U_response_package_v110'
SOURCES=('geobc/physical_U_response_package_v110.py','geobc/physical_U_response_package_contract_v110.py',
 'tools/evaluate_geobc_physical_U_package_v110.py','tools/guard_geobc_physical_U_package_v110.py','tools/prepare_geobc_physical_U_package_v110.py')

def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    _,a,_,b=terminal('notes/geobc_located_force6_U_response_v109_plan.json')
    assert b==s['prior_terminal'] and a['coupled_API_gate_passed'] and a['prediction_barrier_complete']
    assert a['actual_coupled_gradient_points']==6 and a['GT_reads']==0
    assert d.read(check_identity(s['prior_report_identity']))['PNG_visually_accepted']
    check_identity(s['prototype_identity'])
    assert s['selection']['CADs']==['120_153','131_345','131_351'] and s['budgets']['whole_seconds']==900
    assert s['evaluation']['complete_U_queries']==36 and s['evaluation']['outside_count']==3
    for row in s['cases'].values():
        d.stat_matches(row['input_cache_identity'])
        assert row['source_transfer_facts']['REFC']=='123456' and row['source_transfer_facts']['normal_rank']==6
        for name,item in row['basis_manifest_identities'].items():
            rec=d.read(check_identity(item));assert rec['key']==name and rec['initial_guess']=='independent_zero'
            assert rec['numerical_state']=='converged' and rec['true_relative_residual']<=1e-8
            d.stat_matches(rec['physical_U'])
    return design,b

def descriptors(plan):
    obj=d.read(check_identity(plan['scientific']['implementation']['descriptor_identity']))
    assert set(obj['packages'])==set(plan['scientific']['selection']['CADs'])
    expected=plan['scientific']['implementation']['reviewed_payload_digests']
    for cad,package in obj['packages'].items():
        assert package['payload']['CAD']==cad and package['payload_digest']==expected[cad]==d.digest(package['payload'])
    return obj

def approved_contract(path):
    p=d.read(path);r=d.read(p['root_review']);design,b=inputs()
    assert p['kind']=='GeoBC_physical_U_response_package_v110' and p['output']==OUTPUT
    assert all(p['scientific'][k]==v for k,v in design['scientific'].items())
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest'] and r['predecessor_binding']==b
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True)
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins'])
    assert all(d.sha(k)==v for k,v in p['source_pins'].items())
    descriptors(p)
    return p,r
