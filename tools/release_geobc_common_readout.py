"""Bind real completed training receipts and release the frozen readout."""
import argparse
import os
from geobc import available_cad_pilot_data as d


def run():
    assert os.name=='posix'
    plan=d.read('notes/geobc_common_grid_readout_plan.json');review=d.read(plan['root_review'])
    science=plan['scientific'];parent=d.read(science['parent_plan']);folder=d.path(parent['output'])
    final=d.read(folder/'summary.json');external=d.read(folder/'external_terminal.json')
    assert review['passed'] and not review['actual_readout_released']
    assert review['scientific_digest']==plan['scientific_digest']==d.digest(science)
    assert review['source_pins']==plan['source_pins']
    assert all(d.sha(name)==sha for name,sha in plan['source_pins'].items())
    assert all(d.sha(name)==sha for name,sha in parent['source_pins'].items())
    assert final['execution']['scientific_digest']==science['parent_scientific_digest']==parent['scientific_digest']
    assert final['status']=='completed' and final['both4800'] and final['same_shared_init']
    assert final['closed'] and not final['active'] and external['closed'] and not external['active']
    assert external['exit_code']==0 and external['within_external_budget']
    assert all(d.exact_process_exited(external[key]) for key in ('child_process','guard_process'))
    for arm in science['arms']:
        row=final['arms'][arm]
        assert row['status']=='completed' and row['step']==row['optimizer_steps']==4800
        assert row['TRAIN120']['all_cases_complete'] and row['first_actual_fullN_gradient_check']['passed']
        assert row['case_exposure_counts']=={key:40 for key in parent['scientific']['TRAIN_keys']}
        assert max(row['training_elapsed_epoch_seconds'],row['training_elapsed_monotonic_seconds'])<=18000
        assert max(row['readout_elapsed_epoch_seconds'],row['readout_elapsed_monotonic_seconds'])<=1200
    assert not (d.path(plan['output'])/'external_launch.json').exists()
    review.update(actual_readout_released=True,future_terminal_bindings={
        'summary':d.identity(folder/'summary.json',True),'external':d.identity(folder/'external_terminal.json',True),
        'run_process':external['child_process'],'related_guard_process':external['guard_process'],
        'exact_processes_exited':True,'release_raw_array_GT_PT_GPU_reads':0})
    d.write(plan['root_review'],review)
    print({'status':'actual_common_readout_released','TRAIN':{arm:{
        'U':final['arms'][arm]['TRAIN120']['case_mean_u_relative_l2'],
        'S':final['arms'][arm]['TRAIN120']['case_mean_stress_relative_l2'],
        'training_seconds':final['arms'][arm]['training_elapsed_epoch_seconds']} for arm in science['arms']}})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--release',action='store_true')
    if parser.parse_args().release:run()
