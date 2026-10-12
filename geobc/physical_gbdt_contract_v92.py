"""Metadata-only release of the legacy3TRAIN/2knownDEV GPU tree screen."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal, check_identity

DESIGN = 'notes/geobc_physical_tensor_GBDT_v92_design.json'
PLAN = 'notes/geobc_physical_tensor_GBDT_v92_plan.json'
REVIEW = 'notes/geobc_physical_tensor_GBDT_v92_root_review.json'
SOURCE_REVIEW = 'notes/geobc_physical_tensor_GBDT_v92_source_review.json'
OUTPUT = 'runs/geobc_exploration/physical_tensor_GPU_GBDT_legacy5_v92'
TRAIN = ('101_428/ver', '101_56/ver', '101_78/ver')
DEV = ('102_249/ver', '102_38/ver')
SOURCES = ('geobc/physical_gbdt_contract_v92.py', 'tools/train_geobc_physical_GBDT_v92.py',
           'tools/run_geobc_physical_GBDT_guard_v92.py', 'tools/prepare_geobc_physical_GBDT_v92.py')


def inputs():
    design = d.read(DESIGN); s = design['scientific']
    assert d.digest(s) == design['scientific_digest'] and tuple(s['train_keys']) == TRAIN and tuple(s['knownDEV_keys']) == DEV
    _, a, _, b = terminal('notes/geobc_official_UNO_valid_skip_recovery_v91_plan.json')
    assert b == design['prior_UNO91_terminal_binding'] and not a['TRAIN_gate']['passed']
    decision = d.read(check_identity(design['prior_UNO91_decision_identity']))
    assert decision['PNG_visually_accepted'] and not decision['automatic_expand']
    parent = d.read(check_identity(design['parent_readout_identity']))
    mag = d.read(check_identity(design['parent_magnitude_plan_identity']))
    assert parent['status'] == 'completed' and parent['train_keys'] == list(TRAIN) and parent['validation_keys'] == list(DEV)
    assert s['tensor6_scaler'] == parent['scalers']['tensor6'] and s['tensor6_scaler']['fit_keys'] == list(TRAIN)
    assert s['baseline_scalar_reuse']['physical_tensor_MLP'] == parent['arms']['physical_tensor_slots']
    assert s['baseline_scalar_reuse']['direct_MLP'] == parent['arms']['direct_zero_slots']
    for key in TRAIN+DEV:
        q = s['records'][key]; original = mag['records'][key]
        assert q == {name: original[name] for name in q}
        f = q['feature_manifest']; assert d.sha(f['path']) == f['sha256']
        features = d.read(f['path'])
        assert features['status'] == 'completed' and features['feature_valid'] and features['feature_guards']['passed']
        assert features['feature_guards']['exact_GRID_and_row_identity']
        assert features['accessed_members']['numerical_prediction'] == 'original zero-start physical_u; not GT'
        assert features['arrays']['sigma_MPa'] == q['sigma_MPa'] and features['arrays']['valid'] == q['valid']
        assert q['group'] == ('TRAIN' if key in TRAIN else 'DEV') and q['label_identity']['scope'] == q['group']
        assert q['label_identity']['key'] == key and q['label_identity']['member'] == 'stress'
        for name in ('sigma_MPa', 'valid', 'signed_stress'): d.stat_matches(q[name])
    dep = d.read(check_identity(s['dependency_receipt_identity']))
    assert dep['status'] == 'completed' and dep['installed_version'] == '3.0.5' and dep['compiled_GPU_support']
    assert dep['installed_library_sha256'] == s['library_so_sha256']
    return design, b


def approved(plan_path):
    p = d.read(plan_path); r = d.read(p['root_review']); design, b = inputs(); s = p['scientific']
    assert p['output'] == OUTPUT and s == design['scientific']
    assert d.digest(s) == p['scientific_digest'] == r['scientific_digest']
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['source_pins'] == r['source_pins'] and set(SOURCES) <= set(p['source_pins'])
    assert all(d.sha(k) == h for k, h in p['source_pins'].items()) and r['upstream_terminal_binding'] == b
    assert s['num_boost_round'] == 400 and s['parameters']['device'] == 'cuda:0' and s['parameters']['tree_method'] == 'hist'
    assert s['whole_campaign_budget_seconds'] == 900 and s['memory_limit_bytes'] == 30500000000
    return p
