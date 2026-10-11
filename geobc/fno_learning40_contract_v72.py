"""Fixed exposure trend after actual FNO20 gains and native-TS no-gain result."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal, complete_physical, check_identity

PARENT = 'notes/geobc_fno_learning20_v56_plan.json'
DEV = 'notes/geobc_fno20_readout_v57_plan.json'
NATIVE = 'notes/geobc_transolver_native_recovery_v70_plan.json'
SOURCES = ('geobc/fno_learning40_contract_v72.py',
           'tools/train_geobc_fno_learning40_v72.py',
           'tools/run_geobc_fno_learning40_guard_v72.py')


def metadata_inputs():
    p, a, e, b = terminal(PARENT)
    vp, va, ve, vb = terminal(DEV)
    np, na, ne, nb = terminal(NATIVE)
    assert a['training_complete'] and a['TRAIN_gate']['passed'] and va['DEV_gate']['passed']
    assert a['step'] == a['optimizer_step_completions'] == 6000
    assert a['case_exposure_counts'] == {k:20 for k in p['scientific']['TRAIN_keys']}
    complete_physical(a['endpoints']['6000']['full_TRAIN'], p['scientific']['TRAIN_keys'])
    assert va['protocol_qualified'] and va['owned_GT_array_reads'] == va['committed_finite_predictions'] == 60
    assert na['training_complete'] and na['optimizer_step_completions'] == 3000
    dec = d.read('notes/geobc_transolver_native_v70_actual_decision.json')
    assert dec['terminal_binding'] == nb and dec['PNG_visually_verified']
    assert dec['status'] == 'actual_shared3_native_representation_no_substantive_gain'
    for path, bp, bb in [('results/geobc_fno_learning20_v56/manifest.json',p,b),
                         ('results/geobc_fno20_readout_v57/manifest.json',vp,vb),
                         ('results/geobc_transolver_native_shared3_v70/manifest.json',np,nb)]:
        report = d.read(path)
        assert report['status'] == 'completed' and report['SCI'] == bp['scientific_digest'] and report['terminal_binding'] == bb
    ck = a['endpoints']['6000']['checkpoint_identity']
    d.stat_matches(ck)
    costs = d.read('results/geobc_fno_learning20_v56/metrics.json')['complete0to20_cost']
    return dict(parent=p, summary=a, external=e, binding=b, DEV=vp,
        DEV_summary=va, DEV_binding=vb, native_binding=nb, checkpoint=ck,
        complete0to20_cost=costs, dataset=d.read(check_identity(p['scientific']['dataset_identity'])))


def approved_contract(plan_path):
    p = d.read(plan_path)
    s, r, up = p['scientific'], d.read(p['root_review']), metadata_inputs()
    assert p['kind'] == 'GeoBC_FNO9_evidence_gated_physical20_to40_v72'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest'] == r['scientific_digest'] == d.digest(s)
    assert p['source_pins'] == r['source_pins'] and set(SOURCES) <= set(p['source_pins'])
    assert all(d.sha(k) == h for k,h in p['source_pins'].items())
    assert r['upstream_terminal_bindings'] == {'parent':up['binding'],'DEV20':up['DEV_binding'],'native_TS':up['native_binding']}
    old = up['parent']['scientific']
    for k in ('family','seed','train_CAD_count','TRAIN_keys','dataset_identity','scaler','memory_limit_bytes',
              'GPU_lock','optimizer','precision','decoder_chunk_size','loss','parent_norms_identity','parameter_count'):
        assert s[k] == old[k]
    assert s['family'] == 'FNO9' and s['memory_limit_bytes'] == 30500000000
    assert s['parent_step'] == 6000 and s['fixed_updates'] == 12000 and s['additional_updates'] == 6000
    assert s['endpoint_passes'] == [30,40] and s['cache_setup_budget_seconds'] == s['training_budget_seconds'] == 600
    assert s['TRAIN_readout_budget_seconds'] == 300 and s['whole_budget_seconds'] == 1500
    assert not s['fresh_initialization'] and not s['automatic_continuation'] and not s['FINAL_reuse_for_selection']
    assert s['DEV_GT_reads_allowed'] == 0
    assert s['parent_training_SCI'] == up['parent']['scientific_digest'] and s['parent_checkpoint'] == up['checkpoint']
    assert s['parent_TRAIN300'] == up['summary']['endpoints']['6000']['full_TRAIN']
    assert s['parent_case_exposure_counts'] == up['summary']['case_exposure_counts']
    assert s['complete0to20_cost'] == up['complete0to20_cost']
    assert s['TRAIN_gate'] == dict(primary_absolute_improvement_at_least=.03,
        stress_absolute_improvement_at_least=.05,maximum_either_field_absolute_deterioration=.02,
        hotspot_deterioration_max=.02,fixed_denominator=300)
    from .macro_training import balanced_schedule
    order = balanced_schedule(s['TRAIN_keys'],40,2026)
    assert d.digest(order) == s['full_order_digest']
    assert d.digest(order[:6000]) == s['parent_full_order_digest'] == old['full_order_digest']
    assert p['output'] == 'runs/geobc_exploration/FNO100_physical40_v72'
    return p, r, up['dataset'], order, up
