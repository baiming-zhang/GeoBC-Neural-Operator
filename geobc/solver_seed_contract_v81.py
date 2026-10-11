"""Metadata release for paired independent seeds, fixed original300 TRAIN."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal, complete_physical, check_identity
from .macro_training import balanced_schedule

DESIGN = 'notes/geobc_solver_seed_confirmation_v81_design.json'
PLAN = 'notes/geobc_solver_seed_confirmation_v81_plan.json'
REVIEW = 'notes/geobc_solver_seed_confirmation_v81_root_review.json'
SOURCE_REVIEW = 'notes/geobc_solver_seed_confirmation_v81_source_review.json'
SOURCES = ('geobc/solver_seed_contract_v81.py', 'tools/train_geobc_solver_seeds_v81.py',
    'tools/run_geobc_solver_seeds_guard_v81.py', 'tools/prepare_geobc_solver_seeds_v81.py')


def inputs():
    z = d.read(DESIGN); s = z['scientific']
    assert d.digest(s) == z['scientific_digest']
    pp, pa, pe, pb = terminal('notes/geobc_nativeTS100_matched10_v80_plan.json')
    tp, ta, te, tb = terminal('notes/geobc_matched20_solver_v78_plan.json')
    ep, ea, ee, eb = terminal('notes/geobc_matched20_readout_v79_plan.json')
    bindings = dict(nativeTS100_negative=pb, matched20_training=tb, matched20_knownDEV=eb)
    assert bindings == z['upstream_terminal_bindings']
    assert pa['training_complete'] and not pa['TRAIN_gate']['passed'] and ea['protocol_qualified']
    assert d.identity('geobc/decoder_packages_v77.py', hashed=True) == z['model_source_identity']
    assert z['existing_seed2026_summary_identity'] == eb['summary']
    assert d.read('results/geobc_nativeTS100_matched10_v80/manifest.json')['status'] == 'completed'
    for name in ('TRAIN_keys', 'dataset_identity', 'scaler', 'optimizer', 'loss'):
        assert s[name] == pp['scientific'][name]
    for name in ('FNO100_PHYSICAL_6000', 'FNO_GLOBAL_HASH20'):
        complete_physical(ea['candidates'][name]['full60'], s['later_knownDEV']['case_keys'])
        assert ea['candidates'][name]['case_exposure_count'] == 20
    m = d.read(check_identity(s['dataset_identity']))
    assert m['train_keys']['100'] == s['TRAIN_keys'] and len(s['TRAIN_keys']) == 300
    witness = max(s['TRAIN_keys'], key=lambda k: m['records'][k]['num_nodes'])
    assert m['records'][witness]['num_nodes'] == 325387
    return dict(design=z, bindings=bindings, manifest=m, witness=witness, upstream_plans=[pp, tp, ep])


def approved_contract(plan_path):
    p = d.read(plan_path); s = p['scientific']; r = d.read(p['root_review']); up = inputs()
    assert p['kind'] == 'GeoBC_FNO_decoder_packages_paired_seeds_TRAIN100_v81'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s) == p['scientific_digest'] == r['scientific_digest']
    assert p['source_pins'] == r['source_pins'] and set(SOURCES) <= set(p['source_pins'])
    assert all(d.sha(k) == h for k, h in p['source_pins'].items())
    assert r['upstream_terminal_bindings'] == up['bindings']
    assert p['design_identity'] == d.identity(DESIGN, hashed=True)
    for name, value in up['design']['scientific'].items():
        assert s[name] == value
    assert s['fresh_seeds'] == [2027, 2028] and s['fixed_updates_per_arm'] == 6000
    assert s['fixed_passes'] == 20 and s['total_new_optimizer_updates'] == 24000
    assert s['physical_TRAIN_arrays_read_once'] == 300 and s['resident_TRAIN_readout_reuses'] == 1200
    assert s['training_budget_seconds_per_arm'] == 900 and s['cache_budget_seconds'] == 300
    assert s['TRAIN_readout_budget_seconds_per_arm'] == 300 and s['whole_training_campaign_budget_seconds'] == 3000
    assert s['memory_limit_bytes'] == 30500000000
    assert s['qualification_TRAIN_key'] == up['witness']
    orders = {str(seed): balanced_schedule(s['TRAIN_keys'], 20, seed) for seed in s['fresh_seeds']}
    assert {seed: d.digest(order) for seed, order in orders.items()} == p['full_order_digests']
    assert len(s['arms']) == 4 and len({a['name'] for a in s['arms']}) == 4
    assert all(up['manifest']['case_roles'][k] == 'TRAIN' for k in s['TRAIN_keys'])
    assert p['output'] == 'runs/geobc_exploration/FNO_Hash_paired_seeds_TRAIN100_v81'
    return p, r, up['manifest'], orders, up
