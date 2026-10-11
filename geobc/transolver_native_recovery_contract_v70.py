"""Explicit zero-update metadata-failure recovery, preserving the old deadline."""
from . import available_cad_pilot_data as d
from .transolver_native_fit_contract_v69 import approved_contract as original_contract

ORIGINAL = 'notes/geobc_transolver_native_fit_v69_plan.json'
SOURCES = ('geobc/transolver_native_recovery_contract_v70.py',
           'tools/train_geobc_transolver_native_recovery_v70.py',
           'tools/run_geobc_transolver_native_recovery_guard_v70.py')


def original_failed_binding():
    bp, br, m, up = original_contract(ORIGINAL)
    o = d.path(bp['output'])
    a, e = d.read(o/'summary.json'), d.read(o/'external_terminal.json')
    assert a == d.read(o/'status.json') and a['status'] == 'failed' and a['closed'] and not a['active']
    assert a['error_type'] == 'KeyError' and a['error'] == "'shared_external'"
    assert a['optimizer_step_attempts'] == a['optimizer_step_completions'] == 0 and a['models'] == {}
    assert not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
    assert a['physical_TRAIN_reads'] == 3 and a['encoded_TRAIN_reads'] == 0 and a['input_array_reads'] == 12
    assert e['exit_code'] == 1 and e['closed'] and not e['active'] and not e['external_timeout']
    assert e['guard_error_type'] is None and e['actual_child_exited']
    assert all(d.exact_process_exited(e[k]) for k in ('child_process','guard_process'))
    assert not (o/'checkpoints').exists()
    binding = {k:d.identity(o/f, hashed=True) for k,f in
               [('summary','summary.json'),('status','status.json'),('external','external_terminal.json')]}
    return bp, a, e, binding, m, up


def approved_contract(plan_path):
    bp, a, e, binding, m, up = original_failed_binding()
    p = d.read(plan_path)
    s, r = p['scientific'], d.read(p['root_review'])
    assert p['kind'] == 'GeoBC_nativeTS_zero_update_metadata_recovery_v70'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest'] == r['scientific_digest'] == d.digest(s)
    assert r['failed_attempt_binding'] == binding == s['recovery_failed_attempt_binding']
    assert p['source_pins'] == r['source_pins'] and set(SOURCES) <= set(p['source_pins'])
    assert all(d.sha(k) == h for k,h in p['source_pins'].items())
    assert all(p['source_pins'][k] == h for k,h in bp['source_pins'].items())
    for k, value in bp['scientific'].items():
        assert s[k] == value
    assert s['recovery_original_guard_clock'] == {k:e[k] for k in
        ('entry_epoch','entry_monotonic','deadline_epoch','deadline_monotonic','boot_id')}
    assert s['prior_failed_physical_TRAIN_reads'] == 3 and not s['budget_reset']
    assert p['output'] == 'runs/geobc_exploration/Transolver_native_continuous_shared3_v70_recovery'
    up['recovery_failure_cost'] = dict(terminal_binding=binding,
        elapsed_epoch_seconds=e['elapsed_epoch_seconds'], physical_TRAIN_reads=3,
        input_array_reads=12, optimizer_updates=0, lost_ephemeral_cache=True)
    return p, r, m, up
