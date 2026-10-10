"""Single 42000-second, dual-clock guard for the fixed-checkpoint field diagnostic."""
import time
ENTRY_EPOCH = time.time()
ENTRY_MONOTONIC = time.monotonic()
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def run():
    from geobc import available_cad_pilot_data as d
    plan = d.read('notes/geobc_common_grid_train_plan.json')
    review = d.read(plan['root_review'])
    assert review['passed'] and review['actual_training_released']
    assert review['scientific_digest'] == plan['scientific_digest'] == d.digest(plan['scientific'])
    assert review['source_pins'] == plan['source_pins']
    for name, sha in plan['source_pins'].items():
        assert d.sha(name) == sha
    output = d.path(plan['output']); output.mkdir(parents=True,exist_ok=True)
    launch = output/'external_launch.json'
    assert not launch.exists() and not (output/'execution.json').exists(), 'Prior actual attempt exists'
    guard = {'guard_process':d.process_snapshot(os.getpid()),'scientific_digest':plan['scientific_digest'],
             'entry_epoch':ENTRY_EPOCH,'entry_monotonic':ENTRY_MONOTONIC,
             'deadline_epoch':ENTRY_EPOCH+42000,'deadline_monotonic':ENTRY_MONOTONIC+42000}
    receipt = {**guard,'status':'starting','active':True,'closed':False,'automatic_replay':False,
               'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
               'source_pins':plan['source_pins'],'attempts':1}
    with launch.open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    env = os.environ.copy();env['GEOBC_COMMON_TRAIN_GUARD']=json.dumps(guard)
    child=None;code=None;timeout=False
    try:
        with (output/'process.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-u','-m','tools.train_geobc_common_grid','--start'],
                                   cwd=d.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            for _ in range(50):
                identity=d.process_snapshot(child.pid)
                if identity:break
                if child.poll() is not None:break
                time.sleep(.01)
            assert identity and identity['ppid']==guard['guard_process']['pid']
            receipt.update(status='running',child_process=identity);d.write(launch,receipt)
            while child.poll() is None:
                remaining=min(guard['deadline_epoch']-time.time(),guard['deadline_monotonic']-time.monotonic())
                if remaining <= 0:
                    timeout=True;child.terminate()
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
                    break
                try:child.wait(timeout=min(1.,remaining))
                except subprocess.TimeoutExpired:pass
            code=child.returncode
    finally:
        if child is not None and child.poll() is None:
            child.kill();child.wait(timeout=5)
        receipt.update(status='terminal',active=False,closed=True,exit_code=code,external_timeout=timeout,
            elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC,
            actual_child_exited=('child_process' in receipt and d.exact_process_exited(receipt['child_process'])))
        receipt['within_external_budget']=(not timeout and code==0 and receipt['actual_child_exited']
            and receipt['elapsed_epoch_seconds']<=42000 and receipt['elapsed_monotonic_seconds']<=42000)
        d.write(output/'external_terminal.json',receipt);d.write(launch,receipt)
    print(json.dumps({'status':'terminal','exit_code':code,'within_external_budget':receipt['within_external_budget']}))
    return 0 if receipt['within_external_budget'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');args=parser.parse_args()
    if args.start:raise SystemExit(run())
    print('{"status":"source_only","reads":0,"writes":0}')
