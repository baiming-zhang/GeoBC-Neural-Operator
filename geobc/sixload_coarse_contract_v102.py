"""Metadata-only new sixload coarse release; prior real terminals are bound."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_sixload_coarse_preconditioner_v102_design.json'
PLAN='notes/geobc_sixload_coarse_preconditioner_v102_plan.json'
REVIEW='notes/geobc_sixload_coarse_preconditioner_v102_root_review.json'
SOURCE_REVIEW='notes/geobc_sixload_coarse_preconditioner_v102_source_review.json'
OUTPUT='runs/geobc_exploration/fixedCAD_sixload_coarse_v102'
SOURCES=('geobc/sixload_coarse_v102.py','geobc/sixload_coarse_contract_v102.py','tools/run_geobc_sixload_coarse_v102.py','tools/guard_geobc_sixload_coarse_v102.py','tools/prepare_geobc_sixload_coarse_v102.py')

def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    p,a,e,b=terminal('notes/geobc_load_linear_physics_readout_v101_plan.json')
    assert a['physics_reference_compatible'] and a['committed_finite_endpoints']==6 and a['GT_reads']==3
    assert d.read(check_identity(s['prior_report_identity']))['PNG_visually_accepted']
    for name in ('summary','status','external'):check_identity(s['prior_terminal'][name])
    old=d.read(check_identity(s['original_fine_summary_identity']))
    assert old['parallel_K_qualification']['passed'] and old['algebra_guards']['K_applications']==8
    assert len(old['prediction_inventory'])==6 and old['combined_solve_seconds']==s['baseline']['exclusive_group_wall_sum_seconds']
    check_identity(s['original_fine_plan_identity']);d.stat_matches(s['input_cache_identity'])
    for item in s['operator_cache'].values():d.stat_matches(item)
    for item in s['numeric_reference_inventory'].values():check_identity(item)
    check_identity(s['accepted_coarse_plan_identity']);check_identity(s['accepted_coarse_positive_BRIEF_identity'])
    assert s['accepted_coarse_recipe']['coarse_inverse']['steps']==32 and s['budgets']['whole_seconds']==900
    assert s['budgets']['memory_bytes']==30500000000 and s['gate']['required_whole_new_seconds_le']==115.29428690671921
    return design,b

def approved_contract(plan_path):
    p=d.read(plan_path);r=d.read(p['root_review']);design,b=inputs()
    assert p['kind']=='GeoBC_sixload_fixed32_coarse_v102' and p['output']==OUTPUT
    assert p['scientific']==design['scientific'] and p['scientific_digest']==design['scientific_digest']==d.digest(p['scientific'])==r['scientific_digest']
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released'] and r['predecessor_binding']==b
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==v for k,v in p['source_pins'].items())
    assert r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True)
    return p,r
