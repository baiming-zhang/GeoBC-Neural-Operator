"""Metadata gates for a single fixed main TRAIN100 target-free source attempt."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_main_train_source_qualification_v122_design.json'
PLAN='notes/geobc_main_train_source_qualification_v122_plan.json'
REVIEW='notes/geobc_main_train_source_qualification_v122_root_review.json'
SOURCE_REVIEW='notes/geobc_main_train_source_qualification_v122_source_review.json'
OUTPUT='runs/geobc_exploration/main_TRAIN100_targetfree_sources_v122'
SOURCES=('geobc/main_train_physics_sources_v122.py','geobc/main_train_source_contract_v122.py','tools/prepare_geobc_main_train_sources_v122.py','tools/run_geobc_main_train_sources_v122.py','tools/guard_geobc_main_train_sources_v122.py')

def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    p,a,e,b=terminal('notes/geobc_all_support_rigid_motion_v120_plan.json');assert b==s['prior_terminal'] and a['support_rigid_motion_gate_passed']
    assert d.read(check_identity(s['prior_report_identity']))['PNG_visually_accepted']
    inventory=d.read(check_identity(s['inventory_identity']));assert len(inventory['rows'])==100 and list(inventory['rows'])==s['selection']['CADs'] and not inventory['actual_physics_input_qualified']
    for row in inventory['rows'].values():
        assert row['scientific_group']=='TRAIN' and row['source_asset_metadata_present']
        for item in row['raw_source_stat_identities'].values():d.stat_matches(item)
    for cad,row in s['legacy_inputs'].items():
        m=d.read(check_identity(row['manifest_identity']));assert m['feasibility_pass'] and m['file_identity']=={k:v for k,v in row['cache'].items() if k!='array_schema'}
        d.stat_matches(row['cache']);q=d.read(check_identity(row['solve_summary_identity']));assert q['status']=='completed' and q['numerical_state']=='converged' and q['true_relative_residual']<=1e-8
        for key in ('deck','geometry_h5'):
            original=m['contract']['sources'][key];current=inventory['rows'][cad]['raw_source_stat_identities']['SolverDeck' if key=='deck' else 'FieldMesh']
            assert d.relative(d.path(original['path']))==current['path'] and all(original[k]==current[k] for k in ('bytes','mtime_ns'))
    return design,inventory,b

def approved_contract(path):
    p=d.read(path);r=d.read(p['root_review']);design,inventory,b=inputs()
    assert path==PLAN and p['output']==OUTPUT and p['scientific']==design['scientific'] and d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest']
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released'] and r['predecessor_binding']==b
    assert r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True) and p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins'])
    assert all(d.sha(k)==v for k,v in p['source_pins'].items())
    return p,r,inventory
