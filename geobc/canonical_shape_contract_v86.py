"""Canonical geometry cloud comparison with true predecessor/scalar baseline."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal, check_identity

DESIGN = 'notes/geobc_canonical_shape_encoding_v86_design.json'
PLAN = 'notes/geobc_canonical_shape_encoding_v86_plan.json'
REVIEW = 'notes/geobc_canonical_shape_encoding_v86_root_review.json'
SOURCE_REVIEW = 'notes/geobc_canonical_shape_encoding_v86_source_review.json'
OUTPUT = 'runs/geobc_exploration/FNO_canonical_shape_TRAIN9_v86'
SOURCES = ('geobc/canonical_shape_cloud_v86.py', 'geobc/canonical_shape_contract_v86.py',
    'tools/train_geobc_canonical_shape_v86.py', 'tools/run_geobc_canonical_shape_guard_v86.py', 'tools/prepare_geobc_canonical_shape_v86.py')


def inputs():
    design = d.read(DESIGN); s = design['scientific']; assert d.digest(s) == design['scientific_digest']
    vp, va, ve, vb = terminal('notes/geobc_signed_stress_factorization_v85_plan.json')
    bp, ba, be, bb = terminal('notes/geobc_decoder_packages_v77_plan.json')
    assert design['prior_negative_v85_binding'] == vb and design['baseline_terminal_binding'] == bb
    assert not va['TRAIN_gate']['passed'] and va['protocol_qualified']
    decision = d.read(check_identity(design['prior_negative_decision_identity'])); assert decision['PNG_visually_accepted'] and not decision['automatic_expand']
    old = ba['arms']['FNO9_plain_query_MLP']
    assert s['baseline_scalar_endpoints'] == old['endpoints'] and s['expected_model_initial_digest'] == old['initialization']['model_digest']
    assert s['expected_shared_body_digests'] == old['initialization']['shared_body_digests']
    assert s['selected_TRAIN_keys'] == bp['scientific']['selected_TRAIN_keys']
    return design, vb, bb


def approved_contract(plan_path):
    p = d.read(plan_path); r = d.read(p['root_review']); design, vb, bb = inputs(); s = p['scientific']
    assert p['kind'] == 'GeoBC_canonical_shape_TRAIN9_v86' and p['output'] == OUTPUT
    assert s == design['scientific'] and d.digest(s) == p['scientific_digest'] == r['scientific_digest']
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['source_pins'] == r['source_pins'] and set(SOURCES) <= set(p['source_pins'])
    assert all(d.sha(k) == h for k, h in p['source_pins'].items())
    assert r['negative_binding'] == vb and r['baseline_binding'] == bb
    assert s['fixed_updates'] == 900 and s['case_exposures'] == 100 and s['parameter_count_expected'] == 4526858
    assert s['memory_limit_bytes'] == 30500000000
    m = d.read(check_identity(s['dataset_identity'])); keys = s['selected_TRAIN_keys']
    assert len(keys) == len(set(keys)) == 9 and all(m['case_roles'][k] == 'TRAIN' and k in m['train_keys']['100'] for k in keys)
    assert m['scaler'] == s['scaler']
    from .macro_training import balanced_schedule
    order = balanced_schedule(keys, 100, s['seed']); assert d.digest(order) == s['full_order_digest'] == p['full_order_digest']
    return p, r, m, order, vb
