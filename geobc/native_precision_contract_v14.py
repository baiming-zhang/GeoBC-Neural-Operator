"""Future fixed-weight precision probe; metadata only before root release."""
from . import available_cad_pilot_data as d


def approved_contract(plan_path):
    plan=d.read(plan_path); s=plan['scientific']; review=d.read(plan['root_review'])
    assert plan['kind']=='GeoBC_nativePP_TF32_fixedweight_feasibility_v14'
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_execution_released']
    assert plan['scientific_digest']==review['scientific_digest']==d.digest(s)
    assert plan['source_pins']==review['source_pins']
    assert all(d.sha(p)==h for p,h in plan['source_pins'].items())
    assert s['memory_limit_bytes']==30500000000 and s['whole_budget_seconds']==600
    assert s['GPU_lock']=='runs/geobc_exploration/.coordinator.lock'
    assert s['weight_updates']==s['DEV_FINAL_GT_reads']==s['scaler_fits']==0
    assert s['measured_backward_repeats']==3 and s['warmup_backward_per_mode_case']==1
    assert d.identity(s['design_identity']['path'],hashed=True)==s['design_identity']
    design=d.read(s['design_identity']['path'])
    assert s['case_keys']==design['case_keys'] and s['acceptance']==design['acceptance']
    assert s['precision_modes']==design['precision_modes']
    from .native_physical_readout_contract_v13 import approved_contract as previous_contract
    prev,_,manifest,_,qualified=previous_contract('notes/geobc_native_physical100_readout_v13_plan.json')
    assert prev['scientific_digest']==s['native_DEV_SCI']
    for kind,name in [('summary','summary.json'),('status','status.json'),('external','external_terminal.json')]:
        item=s['native_DEV_terminal'][kind]
        assert d.relative(item['path'])==prev['output']+'/'+name
        assert d.identity(item['path'],hashed=True)==item
    summary=d.read(s['native_DEV_terminal']['summary']['path']); ext=d.read(s['native_DEV_terminal']['external']['path'])
    assert summary==d.read(s['native_DEV_terminal']['status']['path'])
    assert summary['status']=='completed' and summary['protocol_qualified'] and summary['closed'] and not summary['active']
    assert ext['closed'] and not ext['active'] and ext['exit_code']==0 and ext['within_external_budget']
    assert all(d.exact_process_exited(ext[k]) for k in ('guard_process','child_process'))
    assert summary['execution']['run_process']==ext['child_process']
    assert summary['execution']['guard']['guard_process']==ext['guard_process']
    assert summary['execution']['scientific_digest']==ext['scientific_digest']==s['native_DEV_SCI']
    assert summary['committed_finite_predictions']==summary['owned_GT_array_reads']==60
    upstream=qualified['PP_NATIVE100_PHYSICAL_1500']
    assert upstream['checkpoint']==s['checkpoint']
    assert upstream['plan']['scientific_digest']==s['native_TRAIN_SCI']
    assert len(s['case_keys'])==len(set(s['case_keys']))==6
    assert all(k in manifest['train_keys']['100'] and manifest['case_roles'][k]=='TRAIN' for k in s['case_keys'])
    assert all(plan['source_pins'].get(p)==h for p,h in prev['source_pins'].items())
    assert plan['output']=='runs/geobc_exploration/native_TF32_fixedweight_v14'
    return plan,review,manifest,upstream
