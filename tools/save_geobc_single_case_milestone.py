"""Light scalar/process snapshot; no dataset/model/GT reads."""
from datetime import datetime, timezone, timedelta
import time
from geobc import available_cad_pilot_data as d


def main():
    folder='runs/geobc_exploration/matched_solver_single_case_fit_v1'
    external=d.read(folder+'/external_launch.json');status=d.read(folder+'/status.json')
    live={key:d.process_snapshot(external[key]['pid']) for key in ('guard_process','child_process')}
    assert all(live[key] and all(live[key][k]==external[key][k] for k in ('pid','start_ticks','argv')) for key in live)
    assert status['first_actual_fullN_gradient_check']['passed']
    step=status['step'];elapsed=status['training_epoch_seconds']
    remaining=elapsed*(4800-step)/step if step>0 else None
    now=datetime.now(timezone.utc)
    result={'observed_UTC':now.isoformat(),'actual_child_and_guard_live':True,'processes':live,
        'selected':status['selected'],'step':step,'training_epoch_seconds':elapsed,
        'remaining_training_seconds_linear_estimate':remaining,
        'estimated_BJT_endpoint':(now+timedelta(seconds=remaining)).astimezone(timezone(timedelta(hours=8))).isoformat() if remaining is not None else None,
        'endpoint_keys_present':list(status['endpoints']),'first_fullN_gradients_passed':True,
        'cached_encoded_reads':status['cached_TRAIN_encoded_reads'],'cached_physical_reads':status['cached_TRAIN_physical_reads'],
        'new_GPU_GT_PT_dataset_reads':0}
    d.write('notes/geobc_single_case_live_milestone.json',result)
    print({k:result[k] for k in ('observed_UTC','step','training_epoch_seconds','remaining_training_seconds_linear_estimate','estimated_BJT_endpoint')})


if __name__=='__main__':main()
