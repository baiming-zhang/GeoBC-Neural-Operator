"""Metadata-only fullrank6 reference-force response release."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_physics_U_fullrank6_v108_design.json'
PLAN='notes/geobc_physics_U_fullrank6_v108_plan.json'
REVIEW='notes/geobc_physics_U_fullrank6_v108_root_review.json'
SOURCE_REVIEW='notes/geobc_physics_U_fullrank6_v108_source_review.json'
OUTPUT='runs/geobc_exploration/threeCAD_physics_U_fullrank6_v108'
SOURCES=('geobc/physical_U_fullrank6_v108.py','geobc/physics_U_fullrank6_contract_v108.py',
         'tools/run_geobc_physics_U_fullrank6_v108.py','tools/guard_geobc_physics_U_fullrank6_v108.py','tools/prepare_geobc_physics_U_fullrank6_v108.py')

def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    pp,a,e,b=terminal('notes/geobc_physics_U_knownDEV_coverage_v107_plan.json')
    assert b==s['prior_terminal'] and a['coverage_gate_passed'] and a['GT_reads']==60 and a['prediction_barrier_complete']
    assert d.read(check_identity(s['prior_report_identity']))['PNG_visually_accepted']
    assert s['selection']['CADs']==['120_153','131_345','131_351'] and s['budgets']['whole_seconds']==1200
    for row in s['cases'].values():
        assert row['source_load_transfer_facts']['REFC']=='123456' and row['source_load_transfer_facts']['normal_rank']==6 and row['source_load_transfer_facts']['passed']
        d.stat_matches(row['input_cache_identity']);check_identity(row['coarse_manifest_identity'])
        for item in row['operator_cache'].values():d.stat_matches(item)
        for item in row['coarse_arrays'].values():
            if 'rowptr'in item:
                for part in ('rowptr','columns','values'):d.stat_matches(item[part])
            else:d.stat_matches(item)
        for item in row['old_canonical_manifest_identities'].values():
            rec=d.read(check_identity(item));assert rec['numerical_state']=='converged' and rec['true_relative_residual']<=1e-8;d.stat_matches(rec['physical_U'])
    return design,b

def approved_contract(path):
    p=d.read(path);r=d.read(p['root_review']);design,b=inputs()
    assert p['kind']=='GeoBC_physical_U_fullrank6_v108' and p['output']==OUTPUT
    assert p['scientific']==design['scientific'] and d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest']
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released'] and r['predecessor_binding']==b
    assert r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True)
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==v for k,v in p['source_pins'].items())
    return p,r
