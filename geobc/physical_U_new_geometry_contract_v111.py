"""Target-free source-to-response deployment release and metadata-only gates."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_physical_U_new_geometry_deployment_v111_design.json'
PLAN='notes/geobc_physical_U_new_geometry_deployment_v111_plan.json'
REVIEW='notes/geobc_physical_U_new_geometry_deployment_v111_root_review.json'
SOURCE_REVIEW='notes/geobc_physical_U_new_geometry_deployment_v111_source_review.json'
OUTPUT='runs/geobc_exploration/two_new_geometry_physical_U_deployment_v111'
SOURCES=('geobc/physical_U_new_geometry_rhs_v111.py','geobc/physical_U_new_geometry_contract_v111.py',
 'tools/run_geobc_physical_U_new_geometry_v111.py','tools/guard_geobc_physical_U_new_geometry_v111.py',
 'tools/prepare_geobc_physical_U_new_geometry_v111.py','tools/review_geobc_new_geometry_packages_v111.py')

def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    _,a,_,b=terminal('notes/geobc_physical_U_response_package_v110_plan.json')
    assert b==s['prior_terminal'] and a['package_gate_passed'] and a['prediction_barrier_complete']
    assert a['loader_calls']==3 and a['GT_reads']==0
    assert d.read(check_identity(s['prior_report_identity']))['PNG_visually_accepted']
    check_identity(s['prototype_identity'])
    assert s['selection']['CADs']==['50_380','624_483'] and s['solve']['coarse_steps']==32
    assert s['budgets']['whole_seconds']==1800 and s['budgets']['memory_bytes']==30500000000
    for cad,row in s['cases'].items():
        for item in row['source_input_identities'].values():d.stat_matches(item)
        d.stat_matches(row['GRID_map_identity'])
        geo=d.read(check_identity(row['metadata_selection_identity']))
        assert geo['cad_id']==cad and geo['num_nodes']==row['expected_geometry']['nodes']
        assert geo['cell_count']==row['expected_geometry']['cells']
    return design,b

def approved_contract(path):
    p=d.read(path);r=d.read(p['root_review']);design,b=inputs()
    assert p['kind']=='GeoBC_physical_U_new_geometry_deployment_v111' and p['output']==OUTPUT
    assert all(p['scientific'][k]==v for k,v in design['scientific'].items())
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest'] and r['predecessor_binding']==b
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True)
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins'])
    assert all(d.sha(k)==v for k,v in p['source_pins'].items())
    return p,r
