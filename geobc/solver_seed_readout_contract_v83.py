"""Four terminal new-seed packages, fixed knownDEV60; metadata only."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal, complete_physical, check_identity

TRAIN = 'notes/geobc_solver_seed_metadata_recovery_v82_plan.json'
REFERENCE = 'notes/geobc_matched20_readout_v79_plan.json'
PLAN = 'notes/geobc_solver_seed_readout_v83_plan.json'
REVIEW = 'notes/geobc_solver_seed_readout_v83_root_review.json'
SOURCE_REVIEW = 'notes/geobc_solver_seed_readout_v83_source_review.json'
OUTPUT = 'runs/geobc_exploration/FNO_Hash_paired_seeds_knownDEV60_v83'
SOURCES = ('geobc/solver_seed_readout_contract_v83.py', 'tools/evaluate_geobc_solver_seeds_v83.py',
           'tools/run_geobc_solver_seed_readout_guard_v83.py', 'tools/prepare_geobc_solver_seed_readout_v83.py')


def inputs():
    p, a, e, b = terminal(TRAIN); rp, ra, re, rb = terminal(REFERENCE)
    s, known = p['scientific'], rp['scientific']
    assert a['training_complete'] and a['protocol_qualified'] and a['matched_body_initial_bytes_verified']
    assert not a['possible_partial_optimizer_update'] and not a['optimizer_step_in_progress']
    assert a['optimizer_step_attempts'] == a['optimizer_step_completions'] == 24000
    assert a['cache_receipt']['physical_TRAIN_target_read_attempts'] == a['cache_receipt']['physical_TRAIN_target_reads'] == 300
    assert a['total_across_attempts_physical_TRAIN_reads'] == 600 and a['resident_physical_target_reuses'] == 1200
    assert a['raw_GT_reads'] == a['DEV_FINAL_reads'] == a['scaler_fits'] == a['parent_PT_reads'] == a['cache_receipt']['encoded_TRAIN_target_reads'] == 0
    assert s['dataset_identity'] == known['dataset_identity'] and s['scaler'] == known['scaler']
    assert s['later_knownDEV']['case_keys'] == known['case_keys'] and len(known['case_keys']) == 60
    assert ra['protocol_qualified'] and ra['prediction_barrier_complete'] and ra['owned_GT_array_reads'] == 60
    qualified = {}
    for spec in s['arms']:
        name = spec['name']; state = a['arms'][name]
        assert state['status'] == 'completed' and state['step'] == 6000 and state['case_exposure_counts'] == {k: 20 for k in s['TRAIN_keys']}
        assert state['package'] == spec['package'] and state['seed'] == spec['seed']
        assert state['qualification']['passed'] and state['qualification']['original_nodes'] == 325387
        assert state['qualification']['model_unchanged'] and state['qualification']['Adam_unchanged'] and state['qualification']['fourRNG_restored']
        complete_physical(state['endpoints']['20']['full_TRAIN'], s['TRAIN_keys'])
        ck = state['endpoints']['20']['checkpoint_identity']; d.stat_matches(ck)
        assert d.relative(ck['path']) == p['output']+'/'+name+'/step6000.pt'
        qualified[name] = dict(plan=p, summary=a, external=e, binding=b, checkpoint=ck, arm=spec, state=state)
    assert set(a['arms']) == set(qualified) and len(qualified) == 4
    for seed in s['fresh_seeds']:
        pair = [x['state'] for x in qualified.values() if x['arm']['seed'] == seed]
        assert len(pair) == 2 and pair[0]['initialization']['shared_body_digests'] == pair[1]['initialization']['shared_body_digests']
    for name in ('FNO100_PHYSICAL_6000', 'FNO_GLOBAL_HASH20', 'TS_CONTINUOUS20'):
        complete_physical(ra['candidates'][name]['full60'], known['case_keys'])
    return dict(plan=p, summary=a, external=e, binding=b, known=known, qualified=qualified,
                reference_plan=rp, reference_summary=ra, reference_binding=rb)


def science_for(up):
    known, ts = up['known'], up['plan']['scientific']
    s = {k: known[k] for k in ('GPU_lock', 'memory_limit_bytes', 'seed', 'decoder_chunk_size', 'GT_policy',
         'pool', 'dataset_identity', 'case_keys', 'scaler', 'GT_manifest_identity')}
    s.update(whole_budget_seconds=900, fixed_prediction_denominator=240,
        training_SCI=up['plan']['scientific_digest'], reference_SCI=up['reference_plan']['scientific_digest'],
        reference_binding=up['reference_binding'],
        candidates=[dict(**x, family=up['qualified'][x['name']]['state']['model_family'], step=6000,
                         training_plan_identity=d.identity(TRAIN, hashed=True)) for x in ts['arms']],
        reference_names=['FNO100_PHYSICAL_6000', 'FNO_GLOBAL_HASH20', 'TS_CONTINUOUS20'],
        gate=ts['gate'], TRAIN_gate_filters_candidates=False, automatic_continuation=False, FINAL_reuse_for_selection=False,
        limits=ts['limits'], decision_policy=ts['decision_policy'])
    return s


def approved_contract(plan_path):
    p = d.read(plan_path); s = p['scientific']; r = d.read(p['root_review']); up = inputs()
    assert p['kind'] == 'GeoBC_paired_seeds_knownDEV60_v83'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s) == p['scientific_digest'] == r['scientific_digest'] and s == science_for(up)
    assert p['source_pins'] == r['source_pins'] and set(SOURCES) <= set(p['source_pins'])
    assert all(d.sha(k) == h for k, h in p['source_pins'].items())
    assert r['training_binding'] == up['binding'] and r['reference_binding'] == up['reference_binding']
    assert s['whole_budget_seconds'] == 900 and s['fixed_prediction_denominator'] == 240 and s['memory_limit_bytes'] == 30500000000
    m = d.read(check_identity(s['dataset_identity'])); keys = s['case_keys']
    assert keys == m['development_keys'] and len(keys) == len(set(keys)) == 60
    assert all(m['case_roles'][k] == 'DEV' for k in keys)
    assert not {k.split('/')[0] for k in keys} & {k.split('/')[0] for k in up['plan']['scientific']['TRAIN_keys']}
    truth = d.read(check_identity(s['GT_manifest_identity']))
    assert truth['status'] == 'completed' and truth['source_GT_parses'] == 20 and set(truth['records']) == set(keys)
    for k in keys:
        t = truth['records'][k]
        assert t['all_original_nodes'] and t['signed_stress_unchanged'] and t['physical']['shape'] == [m['records'][k]['num_nodes'], 4]
        assert not {'target_path', 'physical_path'} & set(m['records'][k]['file_identities'])
    assert p['output'] == OUTPUT
    return p, r, m, truth, up['qualified']
