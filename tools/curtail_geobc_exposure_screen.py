"""Intentionally end the long screen; preserve original science and checkpoints."""
import argparse
import os
import signal
import time
from pathlib import Path
from geobc import available_cad_pilot_data as d


def run():
    destination = d.path('notes/geobc_exposure_short_screen_curtailment.json')
    assert not destination.exists(), 'One decision/attempt; no replay'
    plan = d.read('notes/geobc_exposure_extension_plan.json')
    folder = d.path(plan['output'])
    launch = d.read(folder/'external_launch.json')
    controller = d.read('notes/geobc_exposure_handoff_actual_launch.json')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert boot == launch['boot_id'] == controller['boot_id']
    identities = dict(controller=controller['controller_process'], child=launch['child_process'], guard=launch['guard_process'])
    assert all(d.process_snapshot(p['pid']) == p for p in identities.values())
    state = d.read(folder/'status.json')
    assert state['active'] and not state['closed'] and state['current_arm'] == 'PP256'
    assert all(state['arms'][a]['endpoints']['12000']['TRAIN120']['all_cases_complete'] for a in ('FNO9','PP256'))
    receipt = dict(status='intentional_curtailment_requested', reason='Initial solver screening is too long; use fixed same12000 endpoints, then broad representation screen',
        original_scientific_digest=plan['scientific_digest'], original_24000_campaign_completed=False,
        started_UTC=d.utc(), boot_id=boot, exact_processes=identities,
        immutable_endpoint_identities={a:state['arms'][a]['endpoints']['12000']['checkpoint_identity'] for a in ('FNO9','PP256')},
        pre_stop_status=state, original_plan_sources_checkpoints_logs_preserved=True,
        new_training_updates=0, GT_array_reads=0, signals=[], replay=False)
    d.write(destination, receipt)
    def stop(name, sig, seconds):
        p = identities[name]
        assert d.process_snapshot(p['pid']) == p
        fd = os.pidfd_open(p['pid'])
        try:
            assert d.process_snapshot(p['pid']) == p
            signal.pidfd_send_signal(fd, sig)
            receipt['signals'].append(dict(process=name, signal=int(sig), UTC=d.utc()))
            d.write(destination, receipt)
            deadline = time.monotonic()+seconds
            while not d.exact_process_exited(p) and time.monotonic()<deadline:
                time.sleep(.2)
            assert d.exact_process_exited(p), 'No automatic escalation/replay; inspect process'
        finally:
            os.close(fd)
    # Controller cannot launch the old four-endpoint readout after cancellation.
    stop('controller', signal.SIGTERM, 10)
    # KeyboardInterrupt runs native finally; the guard records the real nonzero exit.
    stop('child', signal.SIGINT, 40)
    deadline = time.monotonic()+10
    while not d.exact_process_exited(identities['guard']) and time.monotonic()<deadline:
        time.sleep(.2)
    assert d.exact_process_exited(identities['guard'])
    terminal = d.read(folder/'external_terminal.json')
    final = d.read(folder/'summary.json')
    assert terminal['closed'] and not terminal['active'] and terminal['actual_child_exited']
    assert terminal['exit_code'] != 0 and final['closed'] and not final['active']
    assert not final.get('both24000', False)
    checkpoint = folder/'PP256'/'last.pt'
    with (folder/'PP256'/'loss.jsonl').open(encoding='utf-8') as f:
        last = None
        for line in f:
            if line.strip(): last = __import__('json').loads(line)
    receipt.update(status='intentionally_curtailed', closed=True, active=False, ended_UTC=d.utc(),
        final_native_summary_identity=d.identity(folder/'summary.json', hashed=True),
        actual_external_terminal_identity=d.identity(folder/'external_terminal.json', hashed=True),
        actual_external_exit_code=terminal['exit_code'], final_native_status=final['status'],
        actual_terminal_summary=final, last_complete_loss_row=last,
        last_full_pass_checkpoint_stat=d.identity(checkpoint, hashed=False),
        partial_uncheckpointed_work_preserved_in_log=True,
        partial_update_cost_unknown=True, unknown_cost_must_not_be_zero_filled=True,
        exact_processes_exited={n:d.exact_process_exited(p) for n,p in identities.items()},
        original_four_endpoint_readout_launches=0)
    d.write(destination, receipt)
    print({k:receipt[k] for k in ('status','actual_external_exit_code','exact_processes_exited')})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stop', action='store_true')
    if parser.parse_args().stop: run()
