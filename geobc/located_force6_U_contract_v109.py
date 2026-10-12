"""Metadata-only release for actual unlocated six-force response coupling."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_located_force6_U_response_v109_design.json'
PLAN='notes/geobc_located_force6_U_response_v109_plan.json'
REVIEW='notes/geobc_located_force6_U_response_v109_root_review.json'
SOURCE_REVIEW='notes/geobc_located_force6_U_response_v109_source_review.json'
OUTPUT='runs/geobc_exploration/threeCAD_located_force6_U_v109'
SOURCES=('geobc/located_force6_U_response_v109.py','geobc/located_force6_U_contract_v109.py',
         'tools/evaluate_geobc_located_force6_U_v109.py','tools/guard_geobc_located_force6_U_v109.py','tools/prepare_geobc_located_force6_U_v109.py')

def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    pp,a,e,b=terminal('notes/geobc_physics_U_fullrank6_v108_plan.json')
    assert b==s['prior_terminal'] and a['fullrank6_gate_passed'] and a['committed_finite_endpoints']==18 and a['GT_reads']==0
    assert d.read(check_identity(s['prior_report_identity']))['PNG_visually_accepted']
    check_identity(s['prototype_identity'])
    assert s['selection']['CADs']==['120_153','131_345','131_351'] and s['budgets']['whole_seconds']==900
    assert s['evaluation']['complete_U_query_count']==72 and s['evaluation']['outside_count']==48
    for row in s['cases'].values():
        d.stat_matches(row['input_cache_identity']);assert row['source_transfer_facts']['REFC']=='123456' and row['source_transfer_facts']['normal_rank']==6
        assert set(row['basis_manifest_identities'])=={'ver','hor','tor','Fy','Mx','My'}
        for name,item in row['basis_manifest_identities'].items():
            rec=d.read(check_identity(item));assert rec['key']==name and rec['numerical_state']=='converged' and rec['true_relative_residual']<=1e-8
            d.stat_matches(rec['physical_U'])
    return design,b

def approved_contract(path):
    p=d.read(path);r=d.read(p['root_review']);design,b=inputs()
    assert p['kind']=='GeoBC_located_force6_U_response_v109' and p['output']==OUTPUT
    assert all(p['scientific'][k]==v for k,v in design['scientific'].items())
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest'] and r['predecessor_binding']==b
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released'] and r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True)
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==v for k,v in p['source_pins'].items())
    return p,r
