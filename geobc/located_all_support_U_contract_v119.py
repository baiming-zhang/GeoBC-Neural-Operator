"""Metadata-only release for actual unlocated six-force response coupling."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_located_all_support_U_response_v119_design.json'
PLAN='notes/geobc_located_all_support_U_response_v119_plan.json'
REVIEW='notes/geobc_located_all_support_U_response_v119_root_review.json'
SOURCE_REVIEW='notes/geobc_located_all_support_U_response_v119_source_review.json'
OUTPUT='runs/geobc_exploration/twoCAD_located_all_support_U_v119'
SOURCES=('geobc/located_all_support_U_response_v119.py','geobc/located_all_support_U_contract_v119.py',
         'tools/evaluate_geobc_located_all_support_U_v119.py','tools/guard_geobc_located_all_support_U_v119.py','tools/prepare_geobc_located_all_support_U_v119.py')

def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    pp,a,e,b=terminal('notes/geobc_all_support_translation_v118_plan.json')
    assert b==s['prior_terminal']and a['support_translation_gate_passed']and a['committed_finite_endpoints']==24 and a['GT_reads']==0
    assert d.read(check_identity(s['prior_report_identity']))['PNG_visually_accepted'];check_identity(s['prototype_identity'])
    assert s['selection']['CADs']==['50_380','624_483']and s['budgets']['whole_seconds']==900
    assert s['selection']['actual_U_queries']==48 and s['selection']['actual_owners']==16 and s['selection']['outsidebox_decisions']==2
    for row in s['cases'].values():
        d.stat_matches(row['input_cache_identity']);assert row['source_transfer_facts']['REFC']=='123456'and row['source_transfer_facts']['normal_rank']==6
        for name,item in row['force_basis_records'].items():
            rec=d.read(check_identity(item['manifest_identity']));assert rec['key']==name and rec['numerical_state']=='converged'and rec['true_relative_residual']<=1e-8
            assert rec['physical_U']==item['physical_U'];d.stat_matches(rec['physical_U'])
        assert len(row['support12_basis_manifest_identities'])==12
        for item in row['support12_basis_manifest_identities']:
            rec=d.read(check_identity(item));assert rec['numerical_state']=='converged'and rec['true_free_relative_residual']<=1e-8
            assert rec['geometry_only_fixed_lift']and rec['initial_free_guess']=='independent_zero';d.stat_matches(rec['physical_U'])
        assert row['actual_support_ports']['source_fixed_union_exact']and row['source_support_constraints']['actual_card_CM']==['123456']*4
    return design,b

def approved_contract(path):
    p=d.read(path);r=d.read(p['root_review']);design,b=inputs()
    assert p['kind']=='GeoBC_located_all_support_U_response_v119' and p['output']==OUTPUT
    assert all(p['scientific'][k]==v for k,v in design['scientific'].items())
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest'] and r['predecessor_binding']==b
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released'] and r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True)
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==v for k,v in p['source_pins'].items())
    return p,r
