"""Metadata-only release for true cached physical-field coordinate/load API."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_physical_U_continuous_API_v104_design.json'
PLAN='notes/geobc_physical_U_continuous_API_v104_plan.json'
REVIEW='notes/geobc_physical_U_continuous_API_v104_root_review.json'
SOURCE_REVIEW='notes/geobc_physical_U_continuous_API_v104_source_review.json'
OUTPUT='runs/geobc_exploration/threeCAD_physical_U_query_v104'
SOURCES=('geobc/physical_U_query_v104.py','geobc/physical_U_query_contract_v104.py','tools/evaluate_geobc_physical_U_query_v104.py','tools/guard_geobc_physical_U_query_v104.py','tools/prepare_geobc_physical_U_query_v104.py')

def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    pp,a,e,b=terminal('notes/geobc_physics_crossCAD_paired_v103_plan.json')
    assert b==s['prior_terminal'] and a['physics_reference_compatible'] and a['cost_gate_passed'] and a['committed_finite_endpoints']==18 and a['GT_reads']==6
    assert d.read(check_identity(s['prior_report_identity']))['PNG_visually_accepted']
    assert s['selection']['CADs']==['120_153','131_345','131_351'] and s['budgets']['whole_seconds']==900 and s['budgets']['memory_bytes']==30500000000
    assert s['actual_tests']['query_count_perCAD']==128 and s['actual_tests']['finite_difference_h_fraction_of_element_scale']==1e-5
    for row in s['cases'].values():
        d.stat_matches(row['input_cache_identity'])
        for name,item in row['canonical_basis_manifest_identities'].items():
            rec=d.read(check_identity(item));assert rec['key']==name and rec['numerical_state']=='converged' and rec['true_relative_residual']<=1e-8 and rec['initial_guess']=='independent_zero';d.stat_matches(rec['physical_U'])
    return design,b

def approved_contract(path):
    p=d.read(path);r=d.read(p['root_review']);design,b=inputs()
    assert p['kind']=='GeoBC_true_physical_U_query_API_v104' and p['output']==OUTPUT
    for k,v in design['scientific'].items():assert p['scientific'][k]==v
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest'] and r['predecessor_binding']==b
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released'] and r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True)
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==v for k,v in p['source_pins'].items())
    assert p['scientific']['prototype_identity']==d.identity('geobc/physical_U_query_v104.py',hashed=True)
    return p,r
