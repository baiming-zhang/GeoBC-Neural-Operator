"""Metadata-only binding for oneCAD/six independent physical RHS."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_load_linear_independent_physics_v100_design.json'
PLAN='notes/geobc_load_linear_independent_physics_v100_plan.json'
REVIEW='notes/geobc_load_linear_independent_physics_v100_root_review.json'
SOURCE_REVIEW='notes/geobc_load_linear_independent_physics_v100_source_review.json'
OUTPUT='runs/geobc_exploration/fixedCAD_sixload_independent_P2_v100'
SOURCES=('geobc/load_linear_physics_inputs_v100.py','geobc/load_linear_physics_pcg_v100.py','geobc/load_linear_physics_contract_v100.py','tools/run_geobc_load_linear_physics_v100.py','tools/guard_geobc_load_linear_physics_v100.py','tools/prepare_geobc_load_linear_physics_v100.py')


def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    p,a,e,b=terminal('notes/geobc_load_linear_identity_readout_v99_plan.json')
    assert b==s['prior_readout_terminal']and a['GT_reads']==60 and a['committed_finite_predictions']==540 and a['protocol_qualified']
    assert d.read(check_identity(s['prior_readout_report_identity']))['PNG_visually_accepted']
    assert s['selection']['CAD']=='120_153'and s['solve']['RHS_count']==6 and s['solve']['initial_guess'].startswith('Independent zero')
    for item in s['source_input_identities'].values():d.stat_matches(item)
    old=d.read(check_identity(s['old_accepted_physics_plan_identity']));assert old['recipe']==s['old_accepted_physics_recipe']
    return design,p,b


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);design,pp,pb=inputs()
    assert p['kind']=='GeoBC_independent_P2_sixload_v100'and p['output']==OUTPUT
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest']and r['passed']and r['whole_source_reviewed']and r['actual_execution_released']
    assert p['source_pins']==r['source_pins']and set(SOURCES)<=set(p['source_pins'])and all(d.sha(k)==v for k,v in p['source_pins'].items())
    for k,v in design['scientific'].items():assert s[k]==v
    assert s['design_identity']==d.identity(DESIGN,hashed=True)and r['predecessor_binding']==pb
    assert s['budget']['whole_seconds']==1800 and s['budget']['memory_bytes']==30500000000 and s['parallel_RHS_groups']==[['ver','hor','tor'],['mix0','mix1','mix2']]
    d.stat_matches(s['GRID_map_identity']);g=d.read(check_identity(s['owned_original_GT_manifest_identity']))
    assert all(k in g['records']for k in s['canonical_keys'])
    return p,r,g
