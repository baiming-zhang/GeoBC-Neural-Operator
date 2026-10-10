"""CPU-only preparation wall guard; does not acquire or use the GPU."""
import argparse
import os
import signal
import subprocess
import sys
import time
from geobc import available_cad_pilot_data as d
from tools.prepare_geobc_prospective_train_targets import PLAN, approved


def run():
    started_epoch,started_mono = time.time(),time.monotonic()
    plan,_=approved()
    output=d.path(plan['output'])
    assert not output.exists(), 'Prior attempt exists; no automatic replay or budget reset'
    output.mkdir(parents=True)
    own=d.process_snapshot(os.getpid());assert own
    argv=[sys.executable,'-u','-m','tools.prepare_geobc_prospective_train_targets','--start']
    environment=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):environment[key]='1'
    child=None;record=None;timeout=False;error=None
    def interrupted(signum,frame):raise KeyboardInterrupt('CPU guard interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        with (output/'native_stdout.log').open('x') as stdout,(output/'native_stderr.log').open('x') as stderr:
            child=subprocess.Popen(argv,cwd=d.ROOT,stdout=stdout,stderr=stderr,env=environment,start_new_session=True)
            record=d.process_snapshot(child.pid);assert record and record['argv']==argv
            parent=d.read('runs/geobc_exploration/common_grid_matched4800_v1/external_launch.json')['child_process']
            peer=d.process_snapshot(parent['pid'])
            if not peer or not all(peer[k]==parent[k] for k in ('pid','start_ticks','argv')):peer=None
            d.write(output/'external_launch.json',{'scientific_digest':plan['scientific_digest'],
                'guard_process':own,'child_process':record,'started_epoch':started_epoch,
                'started_monotonic':started_mono,'whole_budget_seconds':1800,
                'parallel_process_observed':peer,'CUDA_work':0,
                'cost_scope':'CPU IO may overlap the GPU job; machine timing is not an isolated runtime comparison'})
            while child.poll() is None:
                if min(started_epoch+1800-time.time(),started_mono+1800-time.monotonic()) <= 0:
                    timeout=True
                    raise TimeoutError('Original whole CPU preparation budget exhausted')
                time.sleep(.1)
    except BaseException as failure:
        error={'type':type(failure).__name__,'message':str(failure)}
    finally:
        if child is not None and child.poll() is None:
            # The still-unreaped Popen child belongs to this guard. Do not target
            # a stale/reused PID from a state file or another task's processes.
            if record:
                snapshot=d.process_snapshot(child.pid)
                assert snapshot and all(snapshot[k]==record[k] for k in ('pid','start_ticks','argv'))
            child.terminate()
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:child.kill();child.wait()
    exact_exit = record is not None and d.exact_process_exited(record)
    summary=d.read(output/'summary.json') if (output/'summary.json').exists() else None
    success=child is not None and child.returncode==0 and exact_exit and not timeout and error is None and summary is not None and summary['status']=='completed'
    d.write(output/'external_terminal.json',{'status':'completed' if success else 'failed',
        'scientific_digest':plan['scientific_digest'],'active':False,'closed':True,
        'child_process':record,'guard_process':own,'exact_child_exited':exact_exit,'exit_code':child.returncode if child else None,
        'timeout':timeout,'elapsed_epoch_seconds':time.time()-started_epoch,
        'elapsed_monotonic_seconds':time.monotonic()-started_mono,
        'summary_identity':d.identity(output/'summary.json',True) if summary else None,
        'CUDA_work':summary['GPU_work'] if summary else None,'error':error,'no_automatic_replay':True})
    print('CPU TRAIN target preparation '+('completed' if success else 'failed'))
    if not success:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true')
    if parser.parse_args().start:run()
