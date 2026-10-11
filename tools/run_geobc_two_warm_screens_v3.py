"""One bounded sequential GPU campaign, ten passes per independent candidate."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from geobc import available_cad_pilot_data as d


def run():
    output=d.path('runs/geobc_exploration/two_warm_screens_v3');output.mkdir(parents=True,exist_ok=True)
    assert not (output/'status.json').exists(), 'One attempt; no automatic recovery'
    state=dict(status='running',active=True,closed=False,controller_process=d.process_snapshot(os.getpid()),
        boot_id=d.path('/proc/sys/kernel/random/boot_id').read_text().strip(),entry_epoch=time.time(),
        entry_monotonic=time.monotonic(),arms={},fixed_families=['FNO9','HASH_PN'],
        planned_total_updates_per_candidate=4800,actual_additional_updates_target=3600,training_cap_per_candidate_seconds=600,GT_array_reads=0)
    child=None
    def persist():
        state.update(elapsed_epoch_seconds=time.time()-state['entry_epoch'],elapsed_monotonic_seconds=time.monotonic()-state['entry_monotonic'])
        d.write(output/'status.json',state)
    persist()
    try:
        for family in state['fixed_families']:
            plan_path='notes/geobc_macro_warm_'+family+'_40pass_v3_plan.json'
            from geobc.macro_warm_screen_contract_v3 import approved_contract
            plan,_,_,_=approved_contract(plan_path)
            folder=d.path(plan['output']);assert not (folder/'external_launch.json').exists()
            row=dict(status='starting',plan=plan_path,scientific_digest=plan['scientific_digest'],entry_epoch=time.time())
            state['arms'][family]=row;state['current_family']=family;persist()
            with (output/(family+'.log')).open('x',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,'-u','-m','tools.run_geobc_macro_warm_screen_guard_v3','--start','--plan',plan_path],cwd=d.ROOT,stdout=log,stderr=subprocess.STDOUT)
                command=[sys.executable,'-u','-m','tools.run_geobc_macro_warm_screen_guard_v3','--start','--plan',plan_path]
                identity=None
                for _ in range(100):
                    identity=d.process_snapshot(child.pid)
                    if identity and identity['argv']==command and identity['ppid']==state['controller_process']['pid']:break
                    if child.poll() is not None:break
                    time.sleep(.01)
                assert identity and identity['argv']==command and identity['ppid']==state['controller_process']['pid']
                row['guard_process']=identity
                persist()
                # The child guard enforces its frozen dual 2400s deadline; no polling of CUDA.
                code=child.wait(timeout=1230)
            assert d.exact_process_exited(row['guard_process'])
            external=d.read(folder/'external_terminal.json');summary=d.read(folder/'summary.json')
            assert external['closed'] and not external['active'] and summary['closed'] and not summary['active']
            assert external['actual_child_exited'] and d.exact_process_exited(external['child_process'])
            row.update(status=summary['status'],exit_code=code,child_process=external['child_process'],
                summary_identity=d.identity(folder/'summary.json',hashed=True),external_identity=d.identity(folder/'external_terminal.json',hashed=True))
            persist();child=None
            if code:
                # Closed resource/numerical negatives are preserved, not retried or ranked by partial data.
                assert summary.get('error_type') in ('MemoryError','OutOfMemoryError','TimeoutError','FloatingPointError'), 'Technical failure needs root cause review; no automatic next launch'
        state.update(status='training_stages_terminal',current_family=None,
            needs_root_analysis_and_independent_knownDEV_barrier=True,final_solver_selected=False)
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error),automatic_replay=False)
        raise
    finally:
        if child is not None and child.poll() is None:
            signal.pidfd_send_signal(os.pidfd_open(child.pid),signal.SIGINT)
            child.wait(timeout=15)
        state.update(active=False,closed=True);persist();d.write(output/'summary.json',state)
    print(json.dumps({'status':state['status'],'elapsed_epoch_seconds':state['elapsed_epoch_seconds']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true')
    if parser.parse_args().start:run()
