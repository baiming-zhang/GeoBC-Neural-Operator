"""Metadata-only freeze and genuine predecessor binding for full-CAD location."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal, check_identity
DESIGN='notes/geobc_physical_U_global_locator_v106_design.json'
PLAN='notes/geobc_physical_U_global_locator_v106_plan.json'
REVIEW='notes/geobc_physical_U_global_locator_v106_root_review.json'
SOURCE_REVIEW='notes/geobc_physical_U_global_locator_v106_source_review.json'
OUTPUT='runs/geobc_exploration/threeCAD_physical_U_global_locator_v106'
SOURCES=('geobc/physical_U_global_locator_v106.py','geobc/physical_U_global_locator_contract_v106.py',
         'tools/evaluate_geobc_physical_U_global_locator_v106.py','tools/guard_geobc_physical_U_global_locator_v106.py',
         'tools/prepare_geobc_physical_U_global_locator_v106.py')

def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    pp,a,e,b=terminal('notes/geobc_physical_U_query_readout_v105_plan.json')
    assert b==s['prior_terminal'] and a['API_gate_passed'] and a['committed_interior_queries']==2304
    assert d.read(check_identity(s['prior_report_identity']))['PNG_visually_accepted']
    check_identity(s['prior_API_source_identity'])
    assert s['selection']['CADs']==['120_153','131_345','131_351']
    assert s['budgets']['whole_seconds']==900 and s['budgets']['memory_bytes']==30500000000
    for row in s['cases'].values():
        d.stat_matches(row['input_cache_identity'])
        for item in row['canonical_basis_manifest_identities'].values():
            rec=d.read(check_identity(item));assert rec['numerical_state']=='converged' and rec['true_relative_residual']<=1e-8
            d.stat_matches(rec['physical_U'])
        for key in ('query_input_xyz_identity','held_numeric_reference_U_identity','held_numeric_reference_cell_rows_identity'):
            d.stat_matches(row[key])
    return design,b

def approved_contract(path):
    p=d.read(path);r=d.read(p['root_review']);design,b=inputs()
    assert p['kind']=='GeoBC_fullCAD_physical_U_locator_v106' and p['output']==OUTPUT
    assert all(p['scientific'][k]==v for k,v in design['scientific'].items())
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest']
    assert r['predecessor_binding']==b and r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True)
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins'])
    assert all(d.sha(k)==v for k,v in p['source_pins'].items())
    return p,r
