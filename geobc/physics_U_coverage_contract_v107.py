"""Metadata-only complete knownDEV20 physical U coverage release."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_physics_U_knownDEV_coverage_v107_design.json'
PLAN='notes/geobc_physics_U_knownDEV_coverage_v107_plan.json'
REVIEW='notes/geobc_physics_U_knownDEV_coverage_v107_root_review.json'
SOURCE_REVIEW='notes/geobc_physics_U_knownDEV_coverage_v107_source_review.json'
OUTPUT='runs/geobc_exploration/knownDEV20_physics_U_coverage_v107'
SOURCES=('geobc/physics_U_coverage_contract_v107.py','tools/run_geobc_physics_U_coverage_v107.py',
         'tools/guard_geobc_physics_U_coverage_v107.py','tools/prepare_geobc_physics_U_coverage_v107.py')

def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    pp,a,e,b=terminal('notes/geobc_physical_U_global_locator_v106_plan.json')
    assert b==s['prior_terminal'] and a['locator_gate_passed'] and a['prediction_barrier_complete']
    assert d.read(check_identity(s['prior_report_identity']))['PNG_visually_accepted']
    ds=d.read(check_identity(s['dataset_identity']))
    assert sorted({key.split('/')[0]for key in ds['development_keys']})==s['selection']['CADs']
    assert len(s['selection']['new_solve_CADs'])==17 and s['selection']['reused_CADs']==['120_153','131_345','131_351']
    assert s['coarse_recipe']['coarse_inverse']['steps']==32 and s['budgets']['whole_seconds']==2700
    for row in s['cases'].values():
        for item in row['source_input_identities'].values():d.stat_matches(item)
        d.stat_matches(row['GRID_map_identity'])
        for item in row.get('reuse_converged_canonical_manifest_identities',{}).values():
            rec=d.read(check_identity(item));assert rec['numerical_state']=='converged' and rec['true_relative_residual']<=1e-8;d.stat_matches(rec['physical_U'])
    return design,b

def approved_contract(path):
    p=d.read(path);r=d.read(p['root_review']);design,b=inputs()
    assert p['kind']=='GeoBC_knownDEV20_physics_U_coverage_v107' and p['output']==OUTPUT
    assert all(p['scientific'][k]==v for k,v in design['scientific'].items())
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest'] and r['predecessor_binding']==b
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True)
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==v for k,v in p['source_pins'].items())
    gt=d.read(check_identity(p['scientific']['readout']['GT_manifest_identity']))
    assert {cad+'/'+load for cad in p['scientific']['selection']['CADs']for load in ('ver','hor','tor')}==set(gt['records'])
    for rec in gt['records'].values():d.stat_matches(rec['physical'])
    return p,r,gt
