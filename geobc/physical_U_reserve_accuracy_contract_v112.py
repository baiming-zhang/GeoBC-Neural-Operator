"""Independent metadata release for original-U accuracy closure, CPU only."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_physical_U_reserve_accuracy_v112_design.json'
PLAN='notes/geobc_physical_U_reserve_accuracy_v112_plan.json'
REVIEW='notes/geobc_physical_U_reserve_accuracy_v112_root_review.json'
SOURCE_REVIEW='notes/geobc_physical_U_reserve_accuracy_v112_source_review.json'
OUTPUT='runs/geobc_exploration/two_reserve_original_U_accuracy_v112'
SOURCES=('geobc/physical_U_reserve_accuracy_contract_v112.py','tools/readout_geobc_physical_U_reserve_v112.py',
 'tools/guard_geobc_physical_U_reserve_v112.py','tools/prepare_geobc_physical_U_reserve_v112.py')

def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    _,a,_,b=terminal('notes/geobc_physical_U_new_geometry_deployment_v111_plan.json')
    assert b==s['prior_terminal'] and a['deployment_gate_passed'] and a['committed_finite_endpoints']==18 and a['GT_reads']==0
    assert d.read(check_identity(s['prior_report_identity']))['PNG_visually_accepted']
    assert s['selection']['CADs']==['50_380','624_483'] and s['budgets']['whole_seconds']==300
    for cad,row in s['cases'].items():
        d.stat_matches(row['original_CSV_identity']);d.stat_matches(row['geometry_npz_identity'])
        d.stat_matches(row['GRID_identity']);d.stat_matches(row['H5_row_to_CSV_identity'])
        for load,item in row['physical_U_manifests'].items():
            assert item==a['prediction_inventory'][cad+'/canonical_basis/'+load]
            rec=d.read(check_identity(item));assert rec['key']==load and rec['initial_guess']=='independent_zero'
            assert rec['numerical_state']=='converged' and rec['true_relative_residual']<=1e-8;d.stat_matches(rec['physical_U'])
    return design,b

def approved_contract(path):
    p=d.read(path);r=d.read(p['root_review']);design,b=inputs()
    assert p['kind']=='GeoBC_physical_U_reserve_original_accuracy_v112' and p['output']==OUTPUT
    assert all(p['scientific'][k]==v for k,v in design['scientific'].items())
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest'] and r['predecessor_binding']==b
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released'] and r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True)
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==v for k,v in p['source_pins'].items())
    return p,r
