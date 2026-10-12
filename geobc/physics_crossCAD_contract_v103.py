"""Metadata-only fixed twoCAD paired physics release, no scientific bodies."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_physics_crossCAD_paired_v103_design.json'
PLAN='notes/geobc_physics_crossCAD_paired_v103_plan.json'
REVIEW='notes/geobc_physics_crossCAD_paired_v103_root_review.json'
SOURCE_REVIEW='notes/geobc_physics_crossCAD_paired_v103_source_review.json'
OUTPUT='runs/geobc_exploration/twoCAD_paired_physics_v103'
SOURCES=('geobc/physics_crossCAD_contract_v103.py','tools/run_geobc_physics_crossCAD_v103.py','tools/guard_geobc_physics_crossCAD_v103.py','tools/prepare_geobc_physics_crossCAD_v103.py')

def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    pp,a,e,b=terminal('notes/geobc_sixload_coarse_preconditioner_v102_plan.json')
    assert b==s['prior_terminal'] and a['precision_gate_passed'] and max(e['elapsed_epoch_seconds'],e['elapsed_monotonic_seconds'])<=pp['scientific']['gate']['required_whole_new_seconds_le']
    assert d.read(check_identity(s['prior_report_identity']))['PNG_visually_accepted']
    assert s['selection']['CADs']==['131_345','131_351'] and s['coarse_recipe']['coarse_inverse']['steps']==32
    assert s['budgets']['whole_seconds']==1800 and s['budgets']['memory_bytes']==30500000000
    for row in s['cases'].values():
        for item in row['source_input_identities'].values():d.stat_matches(item)
        d.stat_matches(row['GRID_map_identity'])
    return design,b

def approved_contract(path):
    p=d.read(path);r=d.read(p['root_review']);design,b=inputs()
    assert p['kind']=='GeoBC_twoCAD_paired_physics_v103' and p['output']==OUTPUT
    for k,v in design['scientific'].items():assert p['scientific'][k]==v
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest'] and r['predecessor_binding']==b
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released'] and r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True)
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==v for k,v in p['source_pins'].items())
    gt=d.read(check_identity(p['scientific']['owned_GT_manifest_identity']))
    for cad in design['scientific']['selection']['CADs']:
        for load in ('ver','hor','tor'):d.stat_matches(gt['records'][cad+'/'+load]['physical'])
    return p,r,gt
