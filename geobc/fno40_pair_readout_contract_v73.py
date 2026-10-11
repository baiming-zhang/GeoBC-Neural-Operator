"""Both fixed FNO30/40 endpoints on the same knownDEV60; metadata only."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal, complete_physical, check_identity

TRAIN = 'notes/geobc_fno_learning40_v72_plan.json'
REFERENCE = 'notes/geobc_transolver_macro100_readout_v67_plan.json'
PLAN = 'notes/geobc_fno40_pair_readout_v73_plan.json'
REVIEW = 'notes/geobc_fno40_pair_readout_v73_root_review.json'
SOURCES = ('geobc/fno40_pair_readout_contract_v73.py',
           'tools/evaluate_geobc_fno40_pair_v73.py',
           'tools/run_geobc_fno40_pair_readout_guard_v73.py',
           'tools/prepare_geobc_fno40_pair_readout_v73.py')
METHODS = {9000:'FNO100_PHYSICAL_9000',12000:'FNO100_PHYSICAL_12000'}


def inputs():
    p, a, e, b = terminal(TRAIN)
    rp, ra, re, rb = terminal(REFERENCE)
    s, known = p['scientific'], rp['scientific']
    assert a['training_complete'] and a['step'] == a['optimizer_step_completions'] == 12000
    assert a['additional_optimizer_step_completions'] == 6000
    assert not a['possible_partial_optimizer_update'] and not a['optimizer_step_in_progress']
    assert a['case_exposure_counts'] == {k:40 for k in s['TRAIN_keys']}
    assert a['strict_loaded_checkpoint_tensor_state_verified'] and a['actual_largest_fullN_gradient_check']['passed']
    assert a['raw_GT_reads'] == a['DEV_FINAL_reads'] == a['scaler_fits'] == 0
    assert s['endpoint_passes'] == [30,40] and s['family'] == 'FNO9'
    assert s['dataset_identity'] == known['dataset_identity'] and s['scaler'] == known['scaler']
    assert ra['protocol_qualified'] and ra['prediction_barrier_complete']
    assert ra['raw_GT_reads'] == ra['FINAL_numeric_reads'] == ra['training_updates'] == ra['scaler_fits'] == 0
    for z in ra['candidates'].values():
        complete_physical(z['full60'], known['case_keys'])
    checkpoints = {}
    for step in METHODS:
        point = a['endpoints'][str(step)]
        complete_physical(point['full_TRAIN'], s['TRAIN_keys'])
        assert point['case_exposure_count'] == step//300
        ck = point['checkpoint_identity']
        assert d.relative(ck['path']) == p['output']+'/step'+str(step)+'.pt'
        d.stat_matches(ck)  # PT hashing/loading belongs to the guarded actual stage.
        checkpoints[step] = ck
    report = d.read('results/geobc_fno_learning40_v72/manifest.json')
    assert report['status'] == 'completed' and report['SCI'] == p['scientific_digest'] and report['terminal_binding'] == b
    qa = d.read('notes/geobc_fno_learning40_v72_actual_decision.json')
    assert qa['terminal_binding'] == b and qa['PNG_visually_verified']
    assert qa['both_endpoints_readout_required_regardless_TRAIN_gate']
    assert qa['comparison_identity'] == d.identity('results/geobc_fno_learning40_v72/comparison.png',hashed=True)
    costs = d.read('results/geobc_fno_learning40_v72/metrics.json')['complete0to40_cost']
    return dict(plan=p,summary=a,external=e,binding=b,reference_plan=rp,
        reference_summary=ra,reference_binding=rb,known=known,checkpoints=checkpoints,complete0to40_cost=costs)


def approved_contract(plan_path):
    p = d.read(plan_path)
    s, r, up = p['scientific'], d.read(p['root_review']), inputs()
    assert p['kind'] == 'GeoBC_FNO30_and40_knownDEV60_paired_v73'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest'] == r['scientific_digest'] == d.digest(s)
    assert p['source_pins'] == r['source_pins'] and set(SOURCES) <= set(p['source_pins'])
    assert all(d.sha(k) == h for k,h in p['source_pins'].items())
    for previous in (up['plan'],up['reference_plan']):
        assert all(p['source_pins'].get(k) == h for k,h in previous['source_pins'].items())
    for k in ('GPU_lock','memory_limit_bytes','seed','decoder_chunk_size','GT_policy','pool',
              'dataset_identity','case_keys','scaler','GT_manifest_identity'):
        assert s[k] == up['known'][k]
    assert s['memory_limit_bytes'] == 30500000000 and s['whole_budget_seconds'] == 600
    assert s['training_SCI'] == up['plan']['scientific_digest']
    assert s['fixed_prediction_denominator'] == 120 and not s['FINAL_reuse_for_selection']
    assert not s['automatic_continuation'] and not s['TRAIN_gate_filters_candidates']
    candidates = [dict(name=name,family='FNO9',step=step,training_plan_identity=d.identity(TRAIN,hashed=True))
                  for step,name in METHODS.items()]
    assert s['candidates'] == candidates
    assert r['training_bindings'] == {name:up['binding'] for name in METHODS.values()}
    assert s['reference_binding'] == r['reference_binding'] == up['reference_binding']
    assert s['reference_SCI'] == up['reference_plan']['scientific_digest']
    assert s['training_report_identity'] == d.identity('results/geobc_fno_learning40_v72/metrics.json',hashed=True)
    assert s['DEV_gate'] == dict(primary_absolute_improvement_at_least=.03,
        maximum_either_field_absolute_deterioration=.02,hotspot_deterioration_max=.02,fixed_denominator=60)
    m = d.read(check_identity(s['dataset_identity']))
    keys = s['case_keys']
    assert keys == m['development_keys'] and len(keys) == len(set(keys)) == 60
    cads = {k.split('/')[0] for k in keys}
    assert len(cads) == 20 and set(keys) == {c+'/'+l for c in cads for l in ('ver','hor','tor')}
    assert all(m['case_roles'][k] == 'DEV' for k in keys)
    assert not cads & {k.split('/')[0] for k in up['plan']['scientific']['TRAIN_keys']}
    truths = d.read(check_identity(s['GT_manifest_identity']))
    assert truths['status'] == 'completed' and truths['source_GT_parses'] == 20 and set(truths['records']) == set(keys)
    for k in keys:
        t = truths['records'][k]
        assert t['all_original_nodes'] and t['signed_stress_unchanged']
        assert t['physical']['shape'] == [m['records'][k]['num_nodes'],4]
        assert not {'target_path','physical_path'} & set(m['records'][k]['file_identities'])
    assert p['output'] == 'runs/geobc_exploration/FNO100_knownDEV60_physical30_40_v73'
    return p,r,m,truths,{name:dict(up,checkpoint=up['checkpoints'][step]) for step,name in METHODS.items()}
