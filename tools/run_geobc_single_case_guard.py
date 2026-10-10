"""Single attempt, original20000s dual deadline and exact process receipts."""
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
    plan = d.read('notes/geobc_single_case_fit_plan.json'); review = d.read(plan['root_review'])
    assert review['passed'] and review['actual_execution_released']
    assert review['scientific_digest'] == plan['scientific_digest'] == d.digest(plan['scientific'])
    assert review['source_pins'] == plan['source_pins']
    assert all(d.sha(p) == h for p, h in plan['source_pins'].items())
    # No CUDA imports; unresolved upstream stage cannot start this guard.
    from tools.train_geobc_single_case_diagnostic import terminal_binding, select_case
    parent = terminal_binding(review['upstream_bindings']['training'])
    development = terminal_binding(review['upstream_bindings']['development'])
    assert select_case(parent, development) == review['selected']
    output = d.path(plan['output']); output.mkdir(parents=True, exist_ok=True)
    launch = output/'external_launch.json'
    assert not launch.exists() and not (output/'execution.json').exists(), 'Existing attempt preserved'
    guard = {'guard_process': d.process_snapshot(os.getpid()), 'scientific_digest': plan['scientific_digest'],
             'entry_epoch': ENTRY_EPOCH, 'entry_monotonic': ENTRY_MONOTONIC,
             'deadline_epoch': ENTRY_EPOCH+20000, 'deadline_monotonic': ENTRY_MONOTONIC+20000}
    receipt = {**guard, 'status': 'starting', 'active': True, 'closed': False, 'attempts': 1,
        'automatic_replay': False, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'source_pins': plan['source_pins']}
    with launch.open('x', encoding='utf-8') as file: json.dump(receipt, file, indent=2)
    env = os.environ.copy(); env['GEOBC_SINGLE_CASE_GUARD'] = json.dumps(guard)
    child = None; code = None; timeout = False
    try:
        with (output/'process.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-u', '-m', 'tools.train_geobc_single_case_diagnostic', '--start'],
                cwd=d.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            identity = None
            for _ in range(50):
                identity = d.process_snapshot(child.pid)
                if identity or child.poll() is not None: break
                time.sleep(.01)
            assert identity and identity['ppid'] == guard['guard_process']['pid']
            receipt.update(status='running', child_process=identity); d.write(launch, receipt)
            while child.poll() is None:
                left = min(guard['deadline_epoch']-time.time(), guard['deadline_monotonic']-time.monotonic())
                if left <= 0:
                    timeout = True; child.terminate()
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)
                    break
                try: child.wait(timeout=min(1., left))
                except subprocess.TimeoutExpired: pass
            code = child.returncode
    finally:
        if child is not None and child.poll() is None:
            child.kill(); child.wait(timeout=5)
        receipt.update(status='terminal', active=False, closed=True, exit_code=code, external_timeout=timeout,
            elapsed_epoch_seconds=time.time()-ENTRY_EPOCH, elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC,
            actual_child_exited=('child_process' in receipt and d.exact_process_exited(receipt['child_process'])))
        receipt['within_external_budget'] = (not timeout and code == 0 and receipt['actual_child_exited']
            and receipt['elapsed_epoch_seconds'] <= 20000 and receipt['elapsed_monotonic_seconds'] <= 20000)
        d.write(output/'external_terminal.json', receipt); d.write(launch, receipt)
    print(json.dumps({'exit_code': code, 'within_external_budget': receipt['within_external_budget']}))
    return 0 if receipt['within_external_budget'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--start', action='store_true')
    if parser.parse_args().start: raise SystemExit(run())
