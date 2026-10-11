"""Zero-update metadata repair, preserving science, original deadline and cost."""
from . import available_cad_pilot_data as d
from .solver_seed_contract_v81 import approved_contract as original_contract
from .transolver_decoder_contract_v64 import check_identity

PLAN = 'notes/geobc_solver_seed_metadata_recovery_v82_plan.json'
REVIEW = 'notes/geobc_solver_seed_metadata_recovery_v82_root_review.json'
FAILURE = 'notes/geobc_solver_seed_confirmation_v81_technical_failure_receipt.json'
SOURCES = ('geobc/solver_seed_recovery_contract_v82.py',
    'tools/train_geobc_solver_seed_recovery_v82.py', 'tools/run_geobc_solver_seed_recovery_guard_v82.py')


def inputs():
    failure = d.read(FAILURE)
    original_path = check_identity(failure['original_plan_identity'])
    p, r, m, orders, up = original_contract(original_path)
    a = d.read(check_identity(failure['summary_identity']))
    assert a == d.read(check_identity(failure['status_identity']))
    e = d.read(check_identity(failure['external_identity']))
    x = d.read(check_identity(failure['execution_identity']))
    assert a['status'] == 'failed' and a['closed'] and not a['active']
    assert a['optimizer_step_attempts'] == a['optimizer_step_completions'] == 0
    assert not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update'] and not a['arms']
    assert a['error_type'] == 'AttributeError' and a['error'] == "'CommonGridSolver' object has no attribute 'macro_family'"
    assert a['cache_receipt']['physical_TRAIN_target_reads'] == 300 and a['parent_PT_reads'] == 0
    assert e['exit_code'] == 1 and e['closed'] and not e['active']
    assert d.exact_process_exited(x['run_process']) and d.exact_process_exited(x['guard']['guard_process'])
    original = x['guard']
    assert failure['original_guard'] == original
    debit = max(a['elapsed_epoch_seconds'], a['elapsed_monotonic_seconds'])
    assert 0 < debit < p['scientific']['training_budget_seconds_per_arm']
    return dict(original_plan=p, original_guard=original, native=a, external=e,
        failure=failure, manifest=m, orders=orders, upstream=up, first_arm_debit_seconds=debit)


def approved_contract(plan_path):
    p = d.read(plan_path); r = d.read(p['root_review']); up = inputs()
    assert p['kind'] == 'GeoBC_solver_seed_zero_update_metadata_recovery_v82'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific'] == up['original_plan']['scientific']
    assert d.digest(p['scientific']) == p['scientific_digest'] == r['scientific_digest']
    assert p['full_order_digests'] == up['original_plan']['full_order_digests']
    assert p['source_pins'] == r['source_pins'] and set(SOURCES) <= set(p['source_pins'])
    assert all(d.sha(k) == h for k, h in p['source_pins'].items())
    recovery = p['recovery']
    assert recovery['failure_identity'] == d.identity(FAILURE, hashed=True)
    assert recovery['original_guard'] == up['original_guard']
    assert recovery['first_arm_debit_seconds'] == up['first_arm_debit_seconds']
    assert recovery['first_arm_budget_remaining_seconds'] == p['scientific']['training_budget_seconds_per_arm']-up['first_arm_debit_seconds']
    assert r['recovery'] == recovery
    assert p['output'] == 'runs/geobc_exploration/FNO_Hash_paired_seeds_v82_metadata_recovery'
    return p, r, up['manifest'], up['orders'], up
