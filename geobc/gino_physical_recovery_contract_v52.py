"""One root-reviewed zero-update metadata recovery inside original clocks."""
from pathlib import Path
from . import available_cad_pilot_data as d


def approved_contract(plan_path,require_release=True):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review'])
    assert p['kind']=='GeoBC_GINO_physical5_metadata_recovery_v52'
    assert r['passed'] and r['whole_source_reviewed']
    if require_release:assert r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest']
    assert p['source_pins']==r['source_pins'] and all(d.sha(k)==v for k,v in p['source_pins'].items())
    item=s['original_plan_identity'];assert d.identity(item['path'],hashed=True)==item
    from .gino_physical_contract_v52 import approved_contract as original
    op,_,m,order,inventory=original(item['path'])
    for name,value in op['scientific'].items():
        if name!='cache_setup_budget_seconds':assert s[name]==value,name
    f=d.path(op['output'])
    for name,filename in (('summary','summary.json'),('status','status.json'),('external','external_terminal.json')):
        assert s['failure_binding'][name]==d.identity(f/filename,hashed=True)
    a=d.read(f/'summary.json');e=d.read(f/'external_terminal.json')
    assert a==d.read(f/'status.json') and a['status']=='failed' and a['closed'] and not a['active']
    assert a['error_type']=='AttributeError' and a['error']=="'dict' object has no attribute 'detach'"
    assert e['closed'] and not e['active'] and e['exit_code']==1 and not e['external_timeout']
    assert all(d.exact_process_exited(e[k]) for k in ('guard_process','child_process'))
    assert a['step']==a['optimizer_step_attempts']==a['optimizer_step_completions']==0
    assert a['last_checkpoint'] is None and not a['possible_partial_optimizer_update']
    assert a['cache_receipt']['physical_TRAIN_target_reads']==300 and a['cache_load_complete']
    assert s['original_boot_id']==e['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert s['original_deadline_epoch']==e['deadline_epoch'] and s['original_deadline_monotonic']==e['deadline_monotonic']
    cost=max(a['elapsed_epoch_seconds'],a['elapsed_monotonic_seconds'],e['elapsed_epoch_seconds'],e['elapsed_monotonic_seconds'])
    assert s['prior_active_setup_cost_seconds']==cost
    assert s['cache_setup_budget_seconds']==op['scientific']['cache_setup_budget_seconds']-cost>0
    assert s['prior_failure_cost']['external_epoch_seconds']==e['elapsed_epoch_seconds']
    assert s['prior_failure_cost']['physical_TRAIN_target_reads']==300
    assert s['repair_policy']=='Metadata-aware hashing only; zero original updates/no checkpoint; same seeded fresh reconstruction, original clocks and failed cost retained'
    assert p['output']=='runs/geobc_exploration/GINO100_physical5_v52_metadata_recovery'
    return p,r,m,order,inventory
