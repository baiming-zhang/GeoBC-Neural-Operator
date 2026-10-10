"""One lightweight scalar/process receipt; no CUDA imports or model/GT reads."""
from datetime import datetime, timezone, timedelta
import json
import time
from geobc import available_cad_pilot_data as d


def run():
    folder='runs/geobc_exploration/common_grid_matched4800_v1'
    external=d.read(folder+'/external_launch.json');status=d.read(folder+'/status.json')
    processes={key:d.process_snapshot(external[key]['pid']) for key in ('guard_process','child_process')}
    assert all(processes[key] and all(processes[key][k]==external[key][k] for k in ('pid','start_ticks','argv')) for key in processes)
    arm=status['current_arm']; row=status['arms'][arm]
    assert row['first_actual_fullN_gradient_check']['passed']
    loss=d.path(folder+'/'+arm+'/loss.jsonl').read_text(encoding='utf-8').splitlines()
    latest=json.loads(loss[-1]);step=latest['step']; elapsed=latest['training_epoch_seconds']
    remaining=(4800-step)*elapsed/step if step>=30 else None
    now=datetime.now(timezone.utc)
    value={'observed_UTC':now.isoformat(),'status':status['status'],'current_arm':arm,
           'actual_child_and_guard_live':True,'actual_processes':processes,
           'latest_logged_step':step,'latest_update_training_seconds':elapsed,
           'current_arm_remaining_seconds_linear_estimate':remaining,
           'current_arm_finish_BJT_linear_estimate':(now+timedelta(seconds=remaining)).astimezone(timezone(timedelta(hours=8))).isoformat() if remaining else None,
           'estimate_limit':'Early per-case timing; full pair ETA requires actual PP training progress, not first probe speed',
           'no_GPU_queries':True,'GT_reads':0}
    d.write('notes/geobc_common_grid_live_milestone.json',value)
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':run()
