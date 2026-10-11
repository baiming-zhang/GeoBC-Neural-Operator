"""Single metadata repair within original clocks; no budget reset or replay."""
from . import available_cad_pilot_data as d

KIND = 'GeoBC_wholeBC_hash_metadata_recovery_v6'


def approved_contract(plan_path):
    plan=d.read(plan_path);science=plan['scientific'];review=d.read(plan['root_review'])
    assert plan['kind']==KIND and review['passed'] and review['whole_source_reviewed'] and review['actual_execution_released']
    assert review['scientific_digest']==plan['scientific_digest']==d.digest(science)
    assert review['source_pins']==plan['source_pins']
    assert all(d.sha(p)==sha for p,sha in plan['source_pins'].items())
    identity=science['original_plan_identity']
    assert d.identity(identity['path'],hashed=True)==identity
    from .operator_hash_campaign_contract_v5 import approved_contract as original_contract
    original,_,manifest,order=original_contract(identity['path'])
    assert original['scientific']['family']=='FNO_GLOBAL_HASH'
    for field,value in original['scientific'].items():
        if field!='cache_setup_budget_seconds':
            assert science[field]==value, 'No original scientific or budget change: '+field
    folder=d.path(original['output'])
    for name,filename in [('summary','summary.json'),('external','external_terminal.json')]:
        item=review['original_failure_binding'][name]
        assert d.relative(item['path'])==d.relative(folder/filename)
        assert d.identity(item['path'],hashed=True)==item
    summary=d.read(folder/'summary.json');outer=d.read(folder/'external_terminal.json')
    assert summary['status']=='failed' and summary['error_type']=='AttributeError'
    assert summary['closed'] and not summary['active'] and outer['closed'] and not outer['active']
    assert outer['exit_code']==1 and all(d.exact_process_exited(outer[k]) for k in ('guard_process','child_process'))
    assert summary['step']==summary['optimizer_step_attempts']==summary['optimizer_step_completions']==0
    assert summary['last_checkpoint'] is None and not summary['possible_partial_optimizer_update']
    assert summary['raw_GT_reads']==summary['DEV_FINAL_reads']==summary['scaler_fits']==0
    assert science['original_boot_id']==outer['boot_id']
    from pathlib import Path
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==outer['boot_id']
    assert science['original_deadline_epoch']==outer['deadline_epoch']
    assert science['original_deadline_monotonic']==outer['deadline_monotonic']
    consumed=max(summary['elapsed_epoch_seconds'],summary['elapsed_monotonic_seconds'],
                 outer['elapsed_epoch_seconds'],outer['elapsed_monotonic_seconds'])
    assert science['prior_active_setup_cost_seconds']==consumed
    assert science['cache_setup_budget_seconds']==original['scientific']['cache_setup_budget_seconds']-consumed>0
    assert science['repair_policy']=='Only metadata property fix; zero original updates/checkpoints, same fresh seeded model; original clocks and prior costs retained'
    required=('geobc/operator_hash_metadata_v6.py','geobc/operator_hash_recovery_contract_v6.py',
              'tools/train_geobc_operator_hash_recovery_v6.py','tools/run_geobc_operator_hash_recovery_guard_v6.py')
    assert set(required)<=set(plan['source_pins'])
    assert d.relative(plan['output']).startswith('runs/geobc_exploration/macro_')
    return plan,review,manifest,order
