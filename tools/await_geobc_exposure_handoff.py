"""One-shot pidfd handoff of already approved exposure/readout/export stages.

No new science, training restart, raw labels, checkpoints or GPU observation
are introduced. Default is inert. A live controller supersedes manual launch
by the thread watchdog; downstream guards still enforce their own one-shot
claims, parent evidence, source pins, public lock and original budgets.
"""
import time
ENTRY_EPOCH, ENTRY_MONOTONIC = time.time(), time.monotonic()
import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys

OWN = 'tools/await_geobc_exposure_handoff.py'
REVIEW = 'notes/geobc_exposure_handoff_root_review.json'
OUTPUT = 'runs/geobc_exploration/matched_exposure_handoff_v1'
PARENT = 'notes/geobc_exposure_extension_plan.json'
READOUT = 'notes/geobc_exposure_readout_plan.json'


def run():
    from geobc import available_cad_pilot_data as d
    review = d.read(REVIEW)
    parent, readout = d.read(PARENT), d.read(READOUT)
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_controller_released']
    assert review['controller_source_sha256'] == d.sha(OWN)
    assert review['parent_scientific_digest'] == parent['scientific_digest']
    assert review['readout_scientific_digest'] == readout['scientific_digest']
    assert review['parent_source_pins'] == parent['source_pins']
    assert review['readout_source_pins'] == readout['source_pins']
    assert all(d.sha(p) == h for pins in (parent['source_pins'], readout['source_pins'], review['extra_source_pins']) for p, h in pins.items())
    root = d.read(readout['root_review'])
    assert root['passed'] and root['whole_SOURCE_reviewed']
    exporter_review = d.read('notes/geobc_exposure_exporter_root_review.json')
    assert exporter_review['passed'] and exporter_review['whole_SOURCE_reviewed'] and exporter_review['actual_export_released']
    assert exporter_review['exporter_source_sha256'] == d.sha('tools/export_geobc_exposure_comparison.py')
    upstream = d.read(d.path(parent['output'])/'external_launch.json')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert upstream['boot_id'] == boot and upstream['scientific_digest'] == parent['scientific_digest']
    assert d.process_snapshot(upstream['guard_process']['pid']) == upstream['guard_process']
    assert d.process_snapshot(upstream['child_process']['pid']) == upstream['child_process']
    assert hasattr(os, 'pidfd_open'), 'Exact kernel process-exit wait required'
    output = d.path(OUTPUT)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output/'status.json'
    assert not state_path.exists(), 'One controller attempt; no automatic restart'
    state = dict(status='running', active=True, closed=False, phase='wait_exact_parent_guard_exit',
        controller_process=d.process_snapshot(os.getpid()), boot_id=boot,
        parent_scientific_digest=parent['scientific_digest'], readout_scientific_digest=readout['scientific_digest'],
        parent_guard_process=upstream['guard_process'], parent_child_process=upstream['child_process'],
        entry_epoch=ENTRY_EPOCH, entry_monotonic=ENTRY_MONOTONIC,
        deadline_epoch=upstream['deadline_epoch']+3900, deadline_monotonic=upstream['deadline_monotonic']+3900,
        GPU_observations=0, raw_array_GT_PT_reads=0, new_training_launches=0,
        readout_launches=0, CPU_exports=0, child_invocations=[], automatic_replay=False,
        waiting_wall_overlaps_parent_training_not_extra_training_cost=True, error=None)
    with state_path.open('x', encoding='utf-8') as file:
        json.dump(state, file, indent=2)
    fd = None
    owned = None

    def persist():
        state.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,
                     elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(state_path, state)
    def remaining():
        return min(state['deadline_epoch']-time.time(), state['deadline_monotonic']-time.monotonic())
    def stop_child():
        if owned is None or owned.poll() is not None:
            return
        try:
            os.killpg(owned.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            owned.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(owned.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            owned.wait(timeout=5)
    def invoke(phase, module, arguments, budget):
        nonlocal owned
        assert remaining() > 0
        started = (time.time(), time.monotonic())
        state['phase'] = phase
        persist()
        row = dict(phase=phase, command=[sys.executable, '-u', '-m', module]+arguments,
                   entry_epoch=started[0], entry_monotonic=started[1], budget_seconds=budget,
                   exact_process=None, exit_code=None, closed=False)
        state['child_invocations'].append(row)
        with (output/(phase+'.log')).open('x', encoding='utf-8') as log:
            owned = subprocess.Popen(row['command'], cwd=d.ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            identity = None
            for _ in range(50):
                identity = d.process_snapshot(owned.pid)
                if identity or owned.poll() is not None:
                    break
                time.sleep(.01)
            assert identity and identity['ppid'] == state['controller_process']['pid']
            row['exact_process'] = identity
            persist()
            while owned.poll() is None:
                left = min(remaining(), started[0]+budget-time.time(), started[1]+budget-time.monotonic())
                if left <= 0:
                    stop_child()
                    raise TimeoutError('One-shot handoff child/whole deadline')
                try:
                    owned.wait(timeout=min(1., left))
                except subprocess.TimeoutExpired:
                    pass
            row.update(exit_code=owned.returncode, closed=True,
                       elapsed_epoch_seconds=time.time()-started[0], elapsed_monotonic_seconds=time.monotonic()-started[1],
                       actual_exact_process_exited=d.exact_process_exited(identity))
            persist()
            assert row['exit_code'] == 0 and row['actual_exact_process_exited']
        owned = None

    signal.signal(signal.SIGTERM, lambda *args: (_ for _ in ()).throw(InterruptedError('Controller terminated; no replay')))
    try:
        fd = os.pidfd_open(upstream['guard_process']['pid'])
        assert d.process_snapshot(upstream['guard_process']['pid']) == upstream['guard_process']
        events = select.poll()
        events.register(fd, select.POLLIN)
        while True:
            left = min(remaining(), upstream['deadline_epoch']-time.time(), upstream['deadline_monotonic']-time.monotonic())
            if left <= 0:
                raise TimeoutError('Original upstream guard deadline; do not restart')
            if events.poll(max(1, int(min(60., left)*1000))):
                break
        os.close(fd); fd = None
        state.update(waiting_epoch_seconds=time.time()-ENTRY_EPOCH,
                     waiting_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC,
                     phase='verify_actual_parent_terminal')
        persist()
        assert d.exact_process_exited(upstream['guard_process']) and d.exact_process_exited(upstream['child_process'])
        from tools.prepare_geobc_exposure_readout import release
        from tools.evaluate_geobc_exposure_endpoints import parent_qualified
        from tools.export_geobc_exposure_comparison import terminal
        parent_summary, external = terminal(parent)
        assert parent_summary['status'] == 'completed' and parent_summary['both24000']
        assert external['within_external_budget'] and external['exit_code'] == 0
        state['actual_parent_terminal_binding'] = {
            k: d.identity(d.path(parent['output'])/f, hashed=True)
            for k, f in [('summary', 'summary.json'), ('external', 'external_terminal.json')]}
        marker = d.read(readout['root_review'])
        if not marker['actual_readout_released']:
            invoke('release_existing_readout', 'tools.prepare_geobc_exposure_readout', ['--release'], 60)
        parent_qualified(readout, d.read(readout['root_review']))
        stage = d.path(readout['output'])
        if (stage/'external_launch.json').exists():
            # Another authorized coordinator won the existing atomic launch.
            # Adopt its recorded identity and leave observation/export to root.
            state.update(status='completed_existing_handoff_adopted', phase='existing_readout',
                         existing_readout=d.read(stage/'external_launch.json'))
        else:
            state['readout_launches'] = 1
            invoke('run_existing_readout', 'tools.run_geobc_exposure_readout_guard', ['--start'], 3690)
            development, outer = terminal(readout)
            assert development['status'] == 'completed' and development['protocol_qualified']
            assert outer['within_external_budget'] and outer['exit_code'] == 0
            state['actual_readout_terminal_binding'] = {
                k: d.identity(stage/f, hashed=True) for k, f in [('summary', 'summary.json'), ('external', 'external_terminal.json')]}
            state['CPU_exports'] = 1
            invoke('CPU_export_existing_six_endpoints', 'tools.export_geobc_exposure_comparison', ['--export'], 180)
            report = d.read('results/geobc_exposure_comparison/manifest.json')
            assert report['status'] == 'completed'
            state.update(status='completed', phase='await_root_PNG_analysis_and_macro_decision',
                         report='results/geobc_exposure_comparison/BRIEF.md')
    except Exception as error:
        state.update(status='failed', error=type(error).__name__+': '+str(error))
        raise
    finally:
        stop_child()
        if fd is not None:
            os.close(fd)
        state.update(active=False, closed=True)
        persist()
        d.write(output/'summary.json', state)
    print(json.dumps({k: state[k] for k in ('status', 'phase', 'readout_launches', 'CPU_exports')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', action='store_true')
    if parser.parse_args().start:
        run()
    else:
        print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
