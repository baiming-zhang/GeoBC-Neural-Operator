"""One typed-device interface repair, no original deadline/cost reset."""
from . import available_cad_pilot_data as d


def approved_contract(plan_path,require_release=True):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review'])
    assert p['kind']=='GeoBC_GINO_physical5_typed_device_recovery_v52'
    assert r['passed'] and r['whole_source_reviewed']
    if require_release:assert r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest']
    assert p['source_pins']==r['source_pins'] and all(d.sha(k)==h for k,h in p['source_pins'].items())
    parent=s['parent_recovery_plan_identity'];assert d.identity(parent['path'],hashed=True)==parent
    from .gino_physical_recovery_contract_v52 import approved_contract as previous
    pp,_,m,order,inventory=previous(parent['path'])
    for name,value in pp['scientific'].items():
        if name not in ('cache_setup_budget_seconds','prior_failure_cost','repair_policy','prior_active_setup_cost_seconds'):
            assert s[name]==value,name
    f=d.path(pp['output'])
    for name,file in [('summary','summary.json'),('status','status.json'),('external','external_terminal.json')]:
        assert s['parent_failure_binding'][name]==d.identity(f/file,hashed=True)
    a=d.read(f/'summary.json');e=d.read(f/'external_terminal.json')
    assert a==d.read(f/'status.json') and a['status']=='failed' and a['closed'] and not a['active']
    assert a['error_type']=='AttributeError' and a['error']=="'str' object has no attribute 'type'"
    assert e['closed'] and not e['active'] and e['exit_code']==1 and not e['external_timeout']
    assert all(d.exact_process_exited(e[k]) for k in ('child_process','guard_process'))
    assert a['step']==a['optimizer_step_attempts']==a['optimizer_step_completions']==0
    assert a['last_checkpoint'] is None and not a['possible_partial_optimizer_update']
    assert a['cache_receipt']['physical_TRAIN_target_reads']==300
    assert s['expected_fresh_initial_model_sha256']==a['actual_fresh_initial_model_sha256']
    cost=max(a['elapsed_epoch_seconds'],a['elapsed_monotonic_seconds'],e['elapsed_epoch_seconds'],e['elapsed_monotonic_seconds'])
    assert s['prior_active_setup_cost_seconds']==pp['scientific']['prior_active_setup_cost_seconds']+cost
    assert s['cache_setup_budget_seconds']==pp['scientific']['cache_setup_budget_seconds']-cost>0
    prior=pp['scientific']['prior_failure_cost'];total=s['prior_failure_cost']
    for name in ('external_epoch_seconds','external_monotonic_seconds','native_epoch_seconds','native_monotonic_seconds'):
        member=name.removeprefix('external_').removeprefix('native_')
        assert total[name]==prior[name]+(e if name.startswith('external_') else a)['elapsed_'+member]
    assert total['physical_TRAIN_target_reads']==600 and total['input_array_reads']==1200 and total['updates']==0
    assert s['repair_policy']=='Only typed torch.device for proven old CSR backend; zero updates; same real captured fresh model SHA; all prior costs/original absolute deadlines retained'
    assert p['output']=='runs/geobc_exploration/GINO100_physical5_v52_device_recovery'
    return p,r,m,order,inventory
