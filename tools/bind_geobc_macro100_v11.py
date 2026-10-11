"""Metadata-only terminal handoffs; neither launches work nor reads arrays."""
import argparse

OWN='tools/bind_geobc_macro100_v11.py'
REVIEW='notes/geobc_macro100_v11_handoff_source_review.json'
PP='notes/geobc_macro100_PP256_NATIVE_5pass_v11_plan.json'
FNO='notes/geobc_macro100_FNO9_5pass_v11_plan.json'
DEV='notes/geobc_macro100_readout_v11_plan.json'


def terminal(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.macro_screen_contract_v11 import approved_contract
    from geobc.macro100_readout_contract_v11 import complete_physical
    plan,_,_,order=approved_contract(plan_path);science=plan['scientific'];folder=d.path(plan['output'])
    summary=d.read(folder/'summary.json');external=d.read(folder/'external_terminal.json');status=d.read(folder/'status.json')
    assert summary==status and summary['status']=='completed' and summary['training_complete']
    assert summary['closed'] and not summary['active'] and external['closed'] and not external['active']
    assert external['exit_code']==0 and external['within_external_budget'] and not external['external_timeout']
    assert all(d.exact_process_exited(external[k]) for k in ('child_process','guard_process'))
    assert summary['execution']['run_process']==external['child_process']
    assert summary['execution']['guard']['guard_process']==external['guard_process']
    assert summary['execution']['scientific_digest']==external['scientific_digest']==plan['scientific_digest']
    assert summary['execution']['boot_id']==external['boot_id']
    assert summary['cache_load_complete'] and summary['actual_largest_fullN_gradient_check']['passed']
    assert summary['step']==summary['optimizer_step_attempts']==summary['optimizer_step_completions']==len(order)==1500
    assert summary['case_exposure_counts']=={key:5 for key in science['TRAIN_keys']}
    assert not summary['possible_partial_optimizer_update'] and not summary['optimizer_step_in_progress']
    assert summary['raw_GT_reads']==summary['DEV_FINAL_reads']==summary['scaler_fits']==0
    assert max(summary['training_epoch_seconds'],summary['training_monotonic_seconds'])<=science['training_budget_seconds']
    assert max(summary['cache_setup_epoch_seconds'],summary['cache_setup_monotonic_seconds'])<=science['cache_setup_budget_seconds']
    assert max(summary['elapsed_epoch_seconds'],summary['elapsed_monotonic_seconds'],external['elapsed_epoch_seconds'],external['elapsed_monotonic_seconds'])<=science['whole_budget_seconds']
    assert max(summary[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))<=science['memory_limit_bytes']
    endpoint=summary['endpoints']['1500'];complete_physical(endpoint['full_TRAIN'],science['TRAIN_keys'])
    assert endpoint['case_exposure_count']==5
    assert max(endpoint['epoch_seconds'],endpoint['monotonic_seconds'])<=science['TRAIN_readout_budget_seconds']
    d.stat_matches(endpoint['checkpoint_identity'])  # No PT byte/body read.
    return {kind:d.identity(folder/file,hashed=True) for kind,file in [('summary','summary.json'),('external','external_terminal.json'),('status','status.json')]}


def release(mode):
    from geobc import available_cad_pilot_data as d
    marker=d.read(REVIEW)
    assert marker['passed'] and marker['whole_source_reviewed'] and marker['conditional_metadata_release_allowed']
    assert marker['source_sha256']==d.sha(OWN)
    if mode=='fno':
        bindings=terminal(PP);plan=d.read(FNO);review=d.read(plan['root_review'])
        assert not review['actual_execution_released'] and review['PP100_short5_terminal_binding'] is None
        key='PP100_short5_terminal_binding';value={k:bindings[k] for k in ('summary','external')}
    else:
        value={'FNO100_1500':terminal(FNO),'PP_NATIVE100_1500':terminal(PP)}
        plan=d.read(DEV);review=d.read(plan['root_review'])
        assert not review['actual_execution_released'] and review['training_bindings'] is None
        key='training_bindings'
    assert review['passed'] and review['whole_source_reviewed']
    assert review['source_pins']==plan['source_pins'] and all(d.sha(name)==pin for name,pin in plan['source_pins'].items())
    assert review['scientific_digest']==plan['scientific_digest']==d.digest(plan['scientific'])
    assert marker['fixed_plan_digests'][d.relative(FNO if mode=='fno' else DEV)]==plan['scientific_digest']
    journal='notes/geobc_macro100_v11_'+mode+'_actual_metadata_release.json'
    assert not d.path(journal).exists(),'No automatic metadata handoff replay'
    original=d.identity(plan['root_review'],hashed=True)
    d.write(journal,dict(status='verified_terminal_bindings',operation=mode,source_sha256=d.sha(OWN),scientific_digest=plan['scientific_digest'],
        original_root_review=original,bindings=value,numeric_arrays_read=0,PT_body_reads=0,GPU_calls=0,work_launched=False))
    review[key]=value;review['actual_execution_released']=True
    d.write(plan['root_review'],review)
    print(dict(status='actual_metadata_released',operation=mode,plan_scientific_digest=plan['scientific_digest'],numeric_arrays_read=0,PT_body_reads=0,work_launched=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--release',choices=['fno','dev'])
    args=parser.parse_args()
    if args.release:release(args.release)
