"""Fixed TRAIN9 new representation, old scalar baseline only; metadata gate."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal, check_identity, complete_physical

DESIGN = 'notes/geobc_signed_stress_factorization_v85_design.json'
PLAN = 'notes/geobc_signed_stress_factorization_v85_plan.json'
REVIEW = 'notes/geobc_signed_stress_factorization_v85_root_review.json'
SOURCE_REVIEW = 'notes/geobc_signed_stress_factorization_v85_source_review.json'
OUTPUT = 'runs/geobc_exploration/FNO_signed_stress_factorization_TRAIN9_v85'
SOURCES = ('geobc/signed_stress_factorization_v85.py', 'geobc/signed_stress_factorization_contract_v85.py',
           'tools/train_geobc_signed_stress_v85.py', 'tools/run_geobc_signed_stress_guard_v85.py',
           'tools/prepare_geobc_signed_stress_v85.py')


def inputs():
    design = d.read(DESIGN); s = design['scientific']
    assert d.digest(s) == design['scientific_digest']
    dp, da, de, db = terminal('notes/geobc_signed_stress_bottleneck_v84_plan.json')
    bp, ba, be, bb = terminal('notes/geobc_decoder_packages_v77_plan.json')
    assert design['upstream_diagnosis_binding'] == db and design['baseline_v77_terminal_binding'] == bb
    assert da['exact_SSE_identity_all240'] and da['cached_prediction_reference_matched'] and da['protocol_qualified']
    assert d.read(check_identity(design['baseline_report_identity']))['status'] == 'completed'
    decision = d.read(check_identity(design['diagnosis_actual_decision_identity']))
    assert decision['terminal_binding'] == db and decision['final_solver_selected'] is False
    old = ba['arms']['FNO9_plain_query_MLP']
    assert s['baseline_scalar_endpoints'] == old['endpoints'] and s['expected_canonical_initial_model_digest'] == old['initialization']['model_digest']
    assert s['expected_shared_body_digests'] == old['initialization']['shared_body_digests']
    assert s['selected_TRAIN_keys'] == bp['scientific']['selected_TRAIN_keys']
    return design, db, bb


def approved_contract(plan_path):
    p = d.read(plan_path); r = d.read(p['root_review']); design, db, bb = inputs(); s = p['scientific']
    assert p['kind'] == 'GeoBC_signed_stress_factorization_TRAIN9_v85' and p['output'] == OUTPUT
    assert s == design['scientific'] and d.digest(s) == p['scientific_digest'] == r['scientific_digest']
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['source_pins'] == r['source_pins'] and set(SOURCES) <= set(p['source_pins'])
    assert all(d.sha(k) == h for k, h in p['source_pins'].items())
    assert r['diagnosis_binding'] == db and r['baseline_binding'] == bb
    assert s['fixed_updates'] == 900 and s['case_exposures'] == 100 and s['auxiliary_weight'] == .1
    assert s['parameter_count_expected'] == 4526987 and s['memory_limit_bytes'] == 30500000000
    m = d.read(check_identity(s['dataset_identity'])); keys = s['selected_TRAIN_keys']
    assert len(keys) == len(set(keys)) == 9 and all(m['case_roles'][k] == 'TRAIN' and k in m['train_keys']['100'] for k in keys)
    assert m['scaler'] == s['scaler']
    from .macro_training import balanced_schedule
    order = balanced_schedule(keys, 100, s['seed']); assert d.digest(order) == s['full_order_digest']
    assert p['full_order_digest'] == s['full_order_digest']
    return p, r, m, order, db
