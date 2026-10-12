"""Metadata release for authentic four-support rotation boundary responses."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_all_support_rigid_motion_v120_design.json'
PLAN='notes/geobc_all_support_rigid_motion_v120_plan.json'
REVIEW='notes/geobc_all_support_rigid_motion_v120_root_review.json'
SOURCE_REVIEW='notes/geobc_all_support_rigid_motion_v120_source_review.json'
OUTPUT='runs/geobc_exploration/twoCAD_all_support_rigid_motion_v120'
SOURCES=('geobc/support_rigid_motion_response_v120.py','geobc/support_rigid_motion_inputs_v120.py','geobc/all_support_rigid_motion_pcg_v120.py','geobc/all_support_rigid_motion_contract_v120.py',
 'tools/evaluate_geobc_all_support_rigid_motion_v120.py','tools/guard_geobc_all_support_rigid_motion_v120.py','tools/prepare_geobc_all_support_rigid_motion_v120.py')

def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    _,a,_,b=terminal('notes/geobc_located_all_support_U_response_v119_plan.json')
    assert b==s['prior_terminal']and a['coupled_API_gate_passed']and a['committed_U_queries']==48 and a['actual_coupled_gradient_points']==4
    assert all(v['input_basis_bytes_unchanged']and v['passed']for v in a['case_records'].values())
    assert d.read(check_identity(s['prior_report_identity']))['PNG_visually_accepted'];check_identity(s['prototype_identity'])
    assert s['selection']['CADs']==['50_380','624_483']and s['budgets']['whole_seconds']==1800 and s['selection']['new_independent_fields']==30
    for row in s['cases'].values():
        d.stat_matches(row['input_cache_identity']);d.stat_matches(row['raw_deck_coordinate_identity'])
        assert row['source_transfer_facts']['REFC']=='123456'and row['source_transfer_facts']['normal_rank']==6
        for name,item in row['force_basis_records'].items():
            rec=d.read(check_identity(item['manifest_identity']));assert rec['key']==name and rec['numerical_state']=='converged'and rec['true_relative_residual']<=1e-8
            assert rec['physical_U']==item['physical_U'];d.stat_matches(rec['physical_U'])
        assert len(row['support12_basis_manifest_identities'])==12
        for item in row['support12_basis_manifest_identities']:
            rec=d.read(check_identity(item));assert rec['initial_free_guess']=='independent_zero'and rec['numerical_state']=='converged'and rec['true_free_relative_residual']<=1e-8
            assert rec['geometry_only_fixed_lift']and not rec['physicsbasis_warmstart_used'];d.stat_matches(rec['physical_U'])
        for item in row['operator_cache'].values():d.stat_matches(item)
        coarse=d.read(check_identity(row['coarse_cache_identity']));assert coarse['status']=='completed'and coarse['algebra']['passed']and coarse['parallel_M_qualification']['passed']
        for key in ('P','PT','Ac'):
            for field in ('rowptr','columns','values'):d.stat_matches(coarse['arrays'][key][field])
        d.stat_matches(coarse['arrays']['diagonal'])
        assert row['source_support_constraints']['actual_card_CM']==['123456']*4 and row['source_support_constraints']['direct_fixed_physical_nodes']==0
    return design,b

def approved_contract(path):
    p=d.read(path);r=d.read(p['root_review']);design,b=inputs()
    assert p['kind']=='GeoBC_all_support_rigid_motion_v120'and p['output']==OUTPUT
    assert all(p['scientific'][k]==v for k,v in design['scientific'].items())
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest']and r['predecessor_binding']==b
    check_identity(p['scientific']['implementation']['accepted_reader_recipe_metadata_identity'])
    assert r['passed']and r['whole_source_reviewed']and r['actual_execution_released']and r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True)
    assert p['source_pins']==r['source_pins']and set(SOURCES)<=set(p['source_pins'])and all(d.sha(k)==v for k,v in p['source_pins'].items())
    return p,r
