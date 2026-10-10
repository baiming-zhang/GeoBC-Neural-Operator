"""Explicit technical continuation with original absolute wall origins."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from geobc import available_cad_pilot_data as d


def run():
    receipt=d.read('notes/geobc_common_readout_recovery.json')
    assert receipt['root_recovery_released'] and receipt['prior_predictions']==receipt['prior_GT_reads']==0
    assert not d.path(receipt['output']+'/external_launch.json').exists()
    original=d.read(receipt['archived_external_launch']['path'])
    assert min(original['deadline_epoch']-time.time(),original['deadline_monotonic']-time.monotonic())>0
    for item in receipt['archived_identities'].values():
        assert d.identity(item['path'],True)==item
    plan=d.read('notes/geobc_common_grid_readout_plan.json');review=d.read(plan['root_review'])
    assert review['passed'] and review['actual_readout_released']
    assert review['source_pins']==plan['source_pins']
    assert all(d.sha(name)==sha for name,sha in plan['source_pins'].items())
    prior=d.read(receipt['archived_identities']['external_terminal.json']['path'])
    epoch,mono=time.time(),time.monotonic()
    remaining=original['deadline_epoch']-epoch
    assert 0<remaining<=1800
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    # Across WSL boots a monotonic origin cannot be transplanted. Use only
    # the remaining original UTC budget as a new local monotonic watchdog;
    # retain both real observed segments and the UTC gap separately.
    deadline_mono=(original['deadline_monotonic'] if boot==original['boot_id']
                   else mono+remaining)
    guard={'guard_process':d.process_snapshot(os.getpid()),'scientific_digest':plan['scientific_digest'],
           'entry_epoch':epoch,'entry_monotonic':mono,'deadline_epoch':original['deadline_epoch'],
           'deadline_monotonic':deadline_mono}
    output=d.path(receipt['output'])
    external={**guard,'status':'starting','active':True,'closed':False,'boot_id':boot,
              'source_pins':plan['source_pins'],'original_entry_epoch':original['entry_epoch'],
              'original_entry_monotonic':original['entry_monotonic'],'original_boot_id':original['boot_id'],
              'attempts':2,'automatic_replay':False,'original1800_budget_not_reset':True}
    d.write(output/'external_launch.json',external)
    environment=os.environ.copy();environment['GEOBC_COMMON_READOUT_GUARD']=json.dumps(guard)
    child=None;code=None;timeout=False
    try:
        with (output/'process.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-u','-m','tools.evaluate_geobc_common_grid','--start'],
                cwd=d.ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
            identity=d.process_snapshot(child.pid)
            assert identity and identity['ppid']==guard['guard_process']['pid']
            external.update(status='running',child_process=identity);d.write(output/'external_launch.json',external)
            while child.poll() is None:
                available=min(guard['deadline_epoch']-time.time(),deadline_mono-time.monotonic())
                if available<=0:
                    timeout=True;child.terminate()
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
                    break
                try:child.wait(timeout=min(1.,available))
                except subprocess.TimeoutExpired:pass
            code=child.returncode
    finally:
        if child is not None and child.poll() is None:child.kill();child.wait(timeout=5)
        active_mono=prior['elapsed_monotonic_seconds']+time.monotonic()-mono
        external.update(status='terminal',active=False,closed=True,exit_code=code,external_timeout=timeout,
            elapsed_epoch_seconds=time.time()-original['entry_epoch'],elapsed_monotonic_seconds=active_mono,
            continuation_monotonic_seconds=time.monotonic()-mono,
            prior_attempt_monotonic_seconds=prior['elapsed_monotonic_seconds'],
            intervening_UTC_wall_seconds=epoch-(original['entry_epoch']+prior['elapsed_epoch_seconds']),
            monotonic_cost_scope='Sum of actual observed attempt segments; UTC entire campaign also includes gap',
            actual_child_exited=('child_process' in external and d.exact_process_exited(external['child_process'])))
        external['within_external_budget']=(not timeout and code==0 and external['actual_child_exited']
            and 0<=external['elapsed_epoch_seconds']<=1800 and 0<=active_mono<=1800)
        d.write(output/'external_launch.json',external);d.write(output/'external_terminal.json',external)
    print(json.dumps({'status':'terminal','exit_code':code,'within_external_budget':external['within_external_budget']}))
    if code!=0 or not external['within_external_budget']:raise SystemExit(code or 1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--continue-reviewed',action='store_true')
    if parser.parse_args().continue_reviewed:run()
