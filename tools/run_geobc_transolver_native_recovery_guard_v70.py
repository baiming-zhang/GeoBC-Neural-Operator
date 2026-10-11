"""One-shot dual-clock guard for native originalTS shared3 fitting diagnostic."""
import time
ENTRY_EPOCH, ENTRY_MONOTONIC = time.time(), time.monotonic()
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.transolver_native_recovery_contract_v70 import approved_contract
    plan, review, manifest, upstream = approved_contract(plan_path)
    budget = plan['scientific']['whole_budget_seconds']
    origin = plan['scientific']['recovery_original_guard_clock']
    assert origin['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert min(origin['deadline_epoch']-time.time(), origin['deadline_monotonic']-time.monotonic()) > 0
    output = d.path(plan['output'])
    output.mkdir(parents=True, exist_ok=True)
    launch = output/'external_launch.json'
    assert not launch.exists() and not (output/'execution.json').exists(), 'One attempt; independent technical recovery required'
    guard = dict(guard_process=d.process_snapshot(os.getpid()), scientific_digest=plan['scientific_digest'],
        entry_epoch=origin['entry_epoch'], entry_monotonic=origin['entry_monotonic'],
        deadline_epoch=origin['deadline_epoch'], deadline_monotonic=origin['deadline_monotonic'],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    receipt = dict(**guard, status='starting', active=True, closed=False, attempts=1,
        automatic_replay=False, source_pins=plan['source_pins'], plan_path=plan_path,
        actual_child_exited=False, guard_budget_seconds=budget, guard_error_type=None)
    with launch.open('x', encoding='utf-8') as file:
        json.dump(receipt, file, indent=2)
    env = os.environ.copy()
    env['GEOBC_TRANSOLVER_NATIVE_RECOVERY_V70_GUARD'] = json.dumps(guard)
    child = None
    code = None
    timeout = False

    def terminate_child():
        if child is None or child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)

    try:
        with (output/'process.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-u', '-m', 'tools.train_geobc_transolver_native_recovery_v70',
                                      '--start', '--plan', plan_path], cwd=d.ROOT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            identity = None
            for _ in range(50):
                identity = d.process_snapshot(child.pid)
                if identity or child.poll() is not None:
                    break
                time.sleep(.01)
            assert identity and identity['ppid'] == guard['guard_process']['pid']
            receipt.update(status='running', child_process=identity)
            d.write(launch, receipt)
            while child.poll() is None:
                remaining = min(guard['deadline_epoch']-time.time(), guard['deadline_monotonic']-time.monotonic())
                if remaining <= 0:
                    timeout = True
                    terminate_child()
                    break
                try:
                    child.wait(timeout=min(1., remaining))
                except subprocess.TimeoutExpired:
                    pass
            code = child.returncode
    except BaseException as error:
        receipt['guard_error_type'] = type(error).__name__
        raise
    finally:
        terminate_child()
        if child is not None and child.poll() is not None:
            code = child.returncode
        receipt.update(status='terminal', active=False, closed=True, exit_code=code,
            external_timeout=timeout, elapsed_epoch_seconds=time.time()-origin['entry_epoch'],
            elapsed_monotonic_seconds=time.monotonic()-origin['entry_monotonic'],
            actual_child_exited=('child_process' in receipt and d.exact_process_exited(receipt['child_process'])))
        receipt['within_external_budget'] = (receipt['guard_error_type'] is None and not timeout and code == 0 and receipt['actual_child_exited']
            and max(receipt['elapsed_epoch_seconds'], receipt['elapsed_monotonic_seconds']) <= budget)
        d.write(output/'external_terminal.json', receipt)
        d.write(launch, receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'exit_code', 'within_external_budget')}))
    return 0 if receipt['within_external_budget'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--plan')
    args = parser.parse_args()
    if args.start:
        if not args.plan:
            parser.error('--start requires a frozen separately released --plan')
        raise SystemExit(run(args.plan))
    print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
