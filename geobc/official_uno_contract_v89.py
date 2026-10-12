"""True terminal and original baseline metadata gate for the official UNO screen."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal, check_identity

DESIGN = 'notes/geobc_official_UNO_v89_design.json'
PLAN = 'notes/geobc_official_UNO_v89_plan.json'
REVIEW = 'notes/geobc_official_UNO_v89_root_review.json'
SOURCE_REVIEW = 'notes/geobc_official_UNO_v89_source_review.json'
OUTPUT = 'runs/geobc_exploration/Official_UNO_TRAIN9_v89'
SOURCES = ('geobc/official_uno_solver_v89.py', 'geobc/official_uno_contract_v89.py',
    'tools/train_geobc_official_UNO_v89.py', 'tools/run_geobc_official_UNO_guard_v89.py', 'tools/prepare_geobc_official_UNO_v89.py')


def inputs():
    design = d.read(DESIGN); s = design['scientific']; assert d.digest(s) == design['scientific_digest']
    _, va, _, vb = terminal('notes/geobc_local_spatial_operator_v88_plan.json')
    bp, ba, _, bb = terminal('notes/geobc_decoder_packages_v77_plan.json')
    assert design['prior_v88_terminal_binding'] == vb and design['baseline_v77_terminal_binding'] == bb
    decision = d.read(check_identity(design['prior_v88_decision_identity']))
    assert decision['PNG_visually_accepted'] and not decision['automatic_expand']
    assert va['protocol_qualified'] and not va['TRAIN_gate']['passed']
    old = ba['arms']['FNO9_plain_query_MLP']
    assert s['baseline_scalar_endpoints'] == old['endpoints']
    assert s['expected_original_model_digest'] == old['initialization']['model_digest']
    assert s['expected_original_shared_body_digests'] == old['initialization']['shared_body_digests']
    assert s['selected_TRAIN_keys'] == bp['scientific']['selected_TRAIN_keys']
    return design, vb, bb


def approved_contract(plan_path):
    p = d.read(plan_path); r = d.read(p['root_review']); design, vb, bb = inputs(); s = p['scientific']
    assert p['kind'] == 'GeoBC_official_UNO_TRAIN9_v89' and p['output'] == OUTPUT
    assert s == design['scientific'] and d.digest(s) == p['scientific_digest'] == r['scientific_digest']
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert r['negative_binding'] == vb and r['baseline_binding'] == bb
    assert p['source_pins'] == r['source_pins'] and set(SOURCES) <= set(p['source_pins'])
    assert all(d.sha(k) == h for k, h in p['source_pins'].items())
    assert s['fixed_updates'] == 900 and s['case_exposures'] == 100 and s['case_pass_endpoints'] == [0, 10, 100]
    assert s['official_commit'] == 'b307030d9b7ee530823d6a05bddb8238ba22a743' and s['real_parameter_DOF_limit'] == 30000000
    check_identity(s['official_source_identity']); check_identity(s['official_license_identity'])
    assert s['UNO_kwargs']['positional_embedding'] is None and s['UNO_kwargs']['n_layers'] == 5
    assert s['optimizer']['fused'] is False and s['optimizer']['foreach'] is False
    assert s['memory_limit_bytes'] == 30500000000
    m = d.read(check_identity(s['dataset_identity'])); keys = s['selected_TRAIN_keys']
    assert len(keys) == len(set(keys)) == 9 and all(m['case_roles'][k] == 'TRAIN' and k in m['train_keys']['100'] for k in keys)
    assert m['scaler'] == s['scaler']
    from .macro_training import balanced_schedule
    order = balanced_schedule(keys, 100, s['seed'])
    assert d.digest(order) == s['full_order_digest'] == p['full_order_digest']
    return p, r, m, order, vb
