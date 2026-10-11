"""Finite matched5/10 original-node TS experiment; metadata-only release."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal, complete_physical, check_identity
from .macro_training import balanced_schedule

DESIGN = 'notes/geobc_nativeTS100_matched10_v80_design.json'
PLAN = 'notes/geobc_nativeTS100_matched10_v80_plan.json'
REVIEW = 'notes/geobc_nativeTS100_matched10_v80_root_review.json'
SOURCE_REVIEW = 'notes/geobc_nativeTS100_matched10_v80_source_review.json'
SOURCES = ('geobc/nativeTS100_contract_v80.py', 'geobc/nativeTS100_cache_v80.py',
    'tools/train_geobc_nativeTS100_v80.py', 'tools/run_geobc_nativeTS100_guard_v80.py',
    'tools/prepare_geobc_nativeTS100_v80.py')


def inputs():
    design = d.read(DESIGN)
    assert design['scientific_digest'] == d.digest(design['scientific'])
    tp, ta, te, tb = terminal('notes/geobc_matched20_solver_v78_plan.json')
    ep, ea, ee, eb = terminal('notes/geobc_matched20_readout_v79_plan.json')
    np, na, ne, nb = terminal('notes/geobc_transolver_native_recovery_v70_plan.json')
    dp, da, de, db = terminal('notes/geobc_transolver_macro100_v66_plan.json')
    bindings = dict(matched20_training=tb, matched20_knownDEV=eb, native3_recovery=nb)
    assert bindings == design['upstream_terminal_bindings']
    assert d.identity('geobc/transolver_native_hash_v68.py', hashed=True) == design['source_model_identity']
    assert ta['training_complete'] and ta['protocol_qualified'] and ea['protocol_qualified']
    assert na['training_complete'] and da['training_complete']
    s = design['scientific']; keys = s['TRAIN_keys']
    for name in ('dataset_identity', 'scaler', 'TRAIN_keys'):
        assert s[name] == tp['scientific'][name] == dp['scientific'][name]
    complete_physical(da['models']['TS_ORIGINAL_BF16_GLOBAL_HASH']['endpoints']['5'], keys)
    dense10 = ta['arms']['TS_CONTINUOUS20']['endpoints']['3000']['full_TRAIN']
    complete_physical(dense10, keys)
    m = d.read(check_identity(s['dataset_identity']))
    assert m['train_keys']['100'] == keys and len(keys) == 300
    witness = max(keys, key=lambda k: m['records'][k]['num_nodes'])
    assert m['records'][witness]['num_nodes'] == 325387
    for name in ('geobc_matched20_solver_v78', 'geobc_matched20_readout_v79'):
        assert d.read('results/'+name+'/manifest.json')['status'] == 'completed'
    return dict(design=design, bindings=bindings, dense_plan=dp, dense_summary=da,
        dense_binding=db, dense10=dense10, manifest=m, witness=witness,
        upstream_plans=[tp, ep, np, dp])


def approved_contract(plan_path):
    p = d.read(plan_path); s = p['scientific']; r = d.read(p['root_review']); up = inputs()
    assert p['kind'] == 'GeoBC_native_originalTS100_matched5_10_v80'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s) == p['scientific_digest'] == r['scientific_digest']
    assert p['source_pins'] == r['source_pins'] and set(SOURCES) <= set(p['source_pins'])
    assert all(d.sha(k) == h for k, h in p['source_pins'].items())
    assert p['design_identity'] == d.identity(DESIGN, hashed=True)
    assert r['upstream_terminal_bindings'] == up['bindings'] and r['dense5_binding'] == up['dense_binding']
    for name, value in up['design']['scientific'].items():
        assert s[name] == value
    for name in ('optimizer', 'loss', 'expected_FP32_TS_initialization', 'official_TS_kwargs'):
        assert s[name] == up['dense_plan']['scientific'][name]
    assert s['full_order_digest'] == d.digest(balanced_schedule(s['TRAIN_keys'], 10, 2026))
    assert s['qualification_TRAIN_key'] == up['witness']
    assert s['reference_TRAIN5'] == up['dense_summary']['models']['TS_ORIGINAL_BF16_GLOBAL_HASH']['endpoints']['5']
    assert s['reference_TRAIN10'] == up['dense10']
    assert s['gate'] == dict(primary_improvement_min=.03, stress_improvement_min=.03,
        field_deterioration_max=.02, hotspot_deterioration_max=.02)
    assert s['fixed_steps'] == [1500, 3000] and s['maximum_optimizer_updates'] == 3000
    assert s['memory_limit_bytes'] == 30500000000 and s['whole_campaign_budget_seconds'] == 2400
    assert s['training_budget_seconds'] == 1200 and s['cache_budget_seconds'] == 300
    assert s['each_TRAIN_readout_budget_seconds'] == 300
    assert all(up['manifest']['case_roles'][k] == 'TRAIN' for k in s['TRAIN_keys'])
    assert p['output'] == 'runs/geobc_exploration/Native_originalTS100_matched10_v80'
    return p, r, up['manifest'], up
