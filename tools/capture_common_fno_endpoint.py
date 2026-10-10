"""One scalar-only immutable interim receipt; no inference or label reads."""
import hashlib
import json
from datetime import datetime,timezone
from geobc import available_cad_pilot_data as d


def run():
    filename=d.path('runs/geobc_exploration/common_grid_matched4800_v1/status.json')
    raw=filename.read_bytes();status=json.loads(raw.decode('utf-8-sig'))
    plan=d.read('notes/geobc_common_grid_train_plan.json')
    assert status['execution']['scientific_digest']==plan['scientific_digest']
    assert status['status']=='running' and status['current_arm']=='PP256'
    fno=status['arms']['FNO9'];assert fno['status']=='completed' and fno['step']==4800
    assert fno['optimizer_steps']==4800 and set(fno['case_exposure_counts'].values())=={40}
    assert fno['TRAIN120']['all_cases_complete'] and fno['TRAIN120']['metrics_available_cases']==120
    assert fno['first_actual_fullN_gradient_check']['passed']
    execution=status['execution'];guard=execution['guard']['guard_process'];child=execution['run_process']
    live={key:d.process_snapshot(record['pid']) for key,record in [('guard',guard),('child',child)]}
    assert all(live[key] and all(live[key][k]==record[k] for k in ('pid','start_ticks','argv')) for key,record in [('guard',guard),('child',child)])
    scalars={key:value for key,value in fno['TRAIN120'].items() if key!='per_case'}
    result={'status':'completed_interim_FNO_only','pair_complete':False,'readout_DEV60_executed':False,
            'observed_UTC':datetime.now(timezone.utc).isoformat(),'scientific_digest':plan['scientific_digest'],
            'FNO_nominal_parameters':fno['counts']['total'],'shared_init_sha256':fno['shared_init_sha256'],
            'FNO_training_seconds_epoch':fno['training_elapsed_epoch_seconds'],
            'FNO_training_seconds_monotonic':fno['training_elapsed_monotonic_seconds'],
            'FNO_TRAIN120_readout_seconds_epoch':fno['readout_elapsed_epoch_seconds'],
            'FNO_TRAIN120_readout_seconds_monotonic':fno['readout_elapsed_monotonic_seconds'],
            'FNO_TRAIN120':scalars,'checkpoint_identity':fno['checkpoint_identity'],
            'mutable_source_capture_SHA256':hashlib.sha256(raw).hexdigest(),
            'actual_live_processes':live,'actual_current_arm':'PP256',
            'interpretation':'Training fit still inaccurate; no cross-CAD gain or matched pair winner claimed. Entire fixed protocol unchanged.',
            'new_GT_PT_arrays_GPU_reads':0,'final_solver_selected':False}
    out=d.path('notes/geobc_common_FNO4800_interim.json')
    assert not out.exists();d.write(out,result)
    print({'status':result['status'],'TRAIN_U':scalars['case_mean_u_relative_l2'],
           'TRAIN_S':scalars['case_mean_stress_relative_l2'],'TRAIN_hot':scalars['case_mean_stress_hotspot_relative_l2'],
           'training_seconds':result['FNO_training_seconds_epoch'],'PP_actual_live':True})


if __name__=='__main__':run()
