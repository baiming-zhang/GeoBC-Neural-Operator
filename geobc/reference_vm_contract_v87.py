"""TRAIN-only reference-displacement magnitude closure; metadata gate."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import check_identity, terminal

PLAN = 'notes/geobc_reference_vm_closure_v87_plan.json'
REVIEW = 'notes/geobc_reference_vm_closure_v87_root_review.json'
OUTPUT = 'runs/geobc_exploration/reference_displacement_VM_TRAIN3_v87'
CASES = ('101_428/ver', '101_56/ver', '101_78/ver')
SOURCES = ('geobc/reference_vm_contract_v87.py', 'tools/diagnose_geobc_reference_vm_v87.py',
           'tools/run_geobc_reference_vm_guard_v87.py')


def approved(plan_path):
    p = d.read(plan_path); s = p['scientific']; r = d.read(p['root_review'])
    assert p['output'] == OUTPUT and s['case_keys'] == list(CASES)
    assert d.digest(s) == p['scientific_digest'] == r['scientific_digest']
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['source_pins'] == r['source_pins'] and set(SOURCES) <= set(p['source_pins'])
    assert all(d.sha(k) == h for k, h in p['source_pins'].items())
    _, a, _, binding = terminal('notes/geobc_canonical_shape_encoding_v86_plan.json')
    assert binding == p['upstream_terminal_binding'] and not a['TRAIN_gate']['passed']
    decision = d.read(check_identity(p['upstream_decision_identity']))
    assert decision['PNG_visually_accepted'] and not decision['automatic_expand']
    assert s['whole_campaign_budget_seconds'] == 180 and s['memory_limit_bytes'] == 30500000000
    assert s['training_updates'] == s['DEV_FINAL_reads'] == s['new_physical_solves'] == 0
    assert s['sign_rule_search'] is False and s['fit_or_calibration'] is False
    for identity in p['parent_metadata_identities']:
        check_identity(identity)
    for key in CASES:
        q = s['records'][key]
        for name in ('input_cache', 'GT_U_archive', 'signed_stress'): d.stat_matches(q[name])
        for value in q['operator_arrays'].values(): d.stat_matches(value)
        assert q['GT_U_archive']['path'] == q['original_S_source_identity']['source']['path']
        assert q['original_S_source_identity']['key'] == key and q['original_S_source_identity']['scope'] == 'TRAIN'
        assert q['input_schema']['cells10']['shape'] == [q['cells'], 10]
        assert q['input_schema']['vertices_mm']['shape'] == [q['nodes'], 3]
    return p
