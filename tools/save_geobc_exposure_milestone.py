"""One CPU status/process receipt and provisional active-arm ETA; no GPU/model/GT."""
import argparse
import datetime
import time
from geobc import available_cad_pilot_data as d


def run():
    plan=d.read('notes/geobc_exposure_extension_plan.json'); folder=d.path(plan['output'])
    launch=d.read(folder/'external_launch.json'); status=d.read(folder/'status.json')
    boot=d.path('/proc/sys/kernel/random/boot_id').read_text().strip()
    processes={k:d.process_snapshot(launch[k]['pid']) for k in ('guard_process','child_process')}
    live=boot==launch['boot_id'] and all(processes[k]==launch[k] for k in processes)
    arm=status.get('current_arm'); row=status.get('arms',{}).get(arm,{})
    step=row.get('step'); elapsed=row.get('training_epoch_seconds'); remaining=None; finish=None
    if live and isinstance(step,int) and step>4800 and isinstance(elapsed,(int,float)):
        remaining=(24000-step)*elapsed/(step-4800)
        finish=datetime.datetime.fromtimestamp(time.time()+remaining,datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    receipt=dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),scientific_digest=plan['scientific_digest'],
        actual_live=live,boot_id=boot,processes=processes,status=status['status'],current_arm=arm,step=step,
        new_training_epoch_seconds=elapsed,remaining_estimate_seconds=remaining,provisional_active_arm_finish_BJT=finish,
        estimate_scope='Current arm only; no remaining queued-arm estimate without its actual progress',GPU_cache=status.get('GPU_cache'),
        first_actual_cached_fullN_gradient_check=row.get('first_actual_cached_fullN_gradient_check'),
        status_identity=d.identity(folder/'status.json',hashed=True),CPU_GT_PT_model_GPU_reads=0)
    d.write('notes/geobc_exposure_live_milestone.json',receipt)
    print({k:receipt[k] for k in ['actual_live','current_arm','step','new_training_epoch_seconds','remaining_estimate_seconds','provisional_active_arm_finish_BJT']})


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--save',action='store_true')
    if parser.parse_args().save:run()
