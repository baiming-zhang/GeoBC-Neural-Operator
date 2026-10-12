"""Run one numerical experiment under a whole-stage dual-clock deadline."""
import time
ENTRY_EPOCH, ENTRY_MONOTONIC = time.time(), time.monotonic()
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def process_identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')')+2:].split()
        return {'pid': pid, 'ppid': int(fields[1]), 'start_ticks': int(fields[19]),
                'argv': Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0')}
    except FileNotFoundError:
        return None


def run(args):
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    seconds = config['whole_seconds']
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    launch = {'entry_epoch': ENTRY_EPOCH, 'entry_monotonic': ENTRY_MONOTONIC,
              'deadline_epoch': ENTRY_EPOCH+seconds, 'deadline_monotonic': ENTRY_MONOTONIC+seconds,
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'guard_process': process_identity(os.getpid())}
    if args.resume_from:
        old = json.loads((Path(args.resume_from)/'launch.json').read_text())
        terminal = json.loads((Path(args.resume_from)/'terminal.json').read_text())
        if terminal['exit_code'] == 0 or terminal['active'] or not terminal['closed']:
            raise ValueError('Recovery needs a closed failed attempt')
        launch['original_entry_epoch'] = old['entry_epoch']
        launch['original_entry_monotonic'] = old['entry_monotonic']
        launch['deadline_epoch'] = old['deadline_epoch']
        launch['prior_attempt_seconds'] = terminal['elapsed_seconds']
        if launch['boot_id'] == old['boot_id']:
            launch['deadline_monotonic'] = old['deadline_monotonic']
        else:
            gap = max(0., ENTRY_EPOCH-old['entry_epoch']-terminal['elapsed_seconds']['UTC'])
            carry = seconds-terminal['elapsed_seconds']['monotonic']-gap
            launch['deadline_monotonic'] = ENTRY_MONOTONIC+min(old['deadline_epoch']-ENTRY_EPOCH, carry)
        if min(launch['deadline_epoch']-time.time(), launch['deadline_monotonic']-time.monotonic()) <= 0:
            raise TimeoutError('Original recovery budget exhausted')
    env = os.environ.copy()
    env['GEOBC_EXPERIMENT_CLOCK'] = json.dumps(launch)
    command = [sys.executable, '-B', '-u', '-m', args.module, '--config', args.config,
               '--source', args.source, '--output', str(output), '--lock', args.lock]
    if args.resume_from:
        command += ['--resume-from', args.resume_from]
    child = None
    timeout = False
    try:
        with (output/'process.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(50):
                identity = process_identity(child.pid)
                if identity is not None or child.poll() is not None:
                    break
                time.sleep(.01)
            launch['child_process'] = identity
            (output/'launch.json').write_text(json.dumps(launch, indent=2)+'\n', encoding='utf-8')
            while child.poll() is None:
                remaining = min(launch['deadline_epoch']-time.time(), launch['deadline_monotonic']-time.monotonic())
                if remaining <= 0:
                    timeout = True
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    break
                try:
                    child.wait(timeout=min(1., remaining))
                except subprocess.TimeoutExpired:
                    pass
            child.wait()
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=5)
        elapsed = {'UTC': time.time()-ENTRY_EPOCH, 'monotonic': time.monotonic()-ENTRY_MONOTONIC}
        terminal = {'exit_code': child.returncode if child is not None else None, 'timeout': timeout,
                    'elapsed_seconds': elapsed, 'active': False, 'closed': True,
                    'within_budget': not timeout and max(elapsed.values()) <= seconds and child is not None and child.returncode == 0}
        terminal['campaign_UTC_seconds'] = time.time()-launch.get('original_entry_epoch', ENTRY_EPOCH)
        terminal['original_deadline_respected'] = time.time() <= launch['deadline_epoch'] and time.monotonic() <= launch['deadline_monotonic']
        terminal['within_budget'] &= terminal['original_deadline_respected']
        (output/'terminal.json').write_text(json.dumps(terminal, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(terminal))
    return 0 if terminal['within_budget'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('module')
    parser.add_argument('--config', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lock', required=True)
    parser.add_argument('--resume-from')
    raise SystemExit(run(parser.parse_args()))
