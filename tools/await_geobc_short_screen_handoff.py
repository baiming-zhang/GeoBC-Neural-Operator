"""Wait on the actual screen controller, then one DEV barrier and CPU export."""
import time
ENTRY_EPOCH, ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
import os
import select
import signal
import subprocess
import sys
from geobc import available_cad_pilot_data as d

OWN='tools/await_geobc_short_screen_handoff.py'
REVIEW='notes/geobc_short_screen_handoff_root_review.json'
PLAN='notes/geobc_macro_screen_readout_v2_plan.json'


def run():
    review=d.read(REVIEW);plan=d.read(PLAN)
    assert review['passed'] and review['conditional_actual_handoff_released'] and review['source_sha256']==d.sha(OWN)
    assert review['scientific_digest']==plan['scientific_digest']
    assert review['readout_source_pins']==plan['source_pins']
    assert all(d.sha(p)==h for p,h in plan['source_pins'].items())
    upstream=review['upstream_controller'];boot=d.path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert boot==review['boot_id'] and d.process_snapshot(upstream['pid'])==upstream
    output=d.path('runs/geobc_exploration/short_screen_handoff_v2');output.mkdir(parents=True,exist_ok=True)
    assert not (output/'status.json').exists(),'One handoff attempt'
    epoch_deadline=review['upstream_origin_epoch']+3*2430+2100
    mono_deadline=review['upstream_origin_monotonic']+3*2430+2100
    state=dict(status='running',active=True,closed=False,phase='wait_exact_controller_exit',
        controller_process=d.process_snapshot(os.getpid()),boot_id=boot,upstream_controller=upstream,
        scientific_digest=plan['scientific_digest'],child_invocations=[],readout_launches=0,CPU_exports=0,
        deadline_epoch=epoch_deadline,deadline_monotonic=mono_deadline)
    child=None;fd=None
    def remaining():return min(epoch_deadline-time.time(),mono_deadline-time.monotonic())
    def persist():
        state.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(output/'status.json',state)
    def interrupt_owned_child():
        identity=d.process_snapshot(child.pid)
        assert identity and identity['ppid']==state['controller_process']['pid']
        owned_fd=os.pidfd_open(child.pid)
        try:
            assert d.process_snapshot(child.pid)==identity
            signal.pidfd_send_signal(owned_fd,signal.SIGINT)
        finally:os.close(owned_fd)
    def invoke(phase,module,args,budget):
        nonlocal child
        state['phase']=phase;row=dict(phase=phase,module=module,entry_epoch=time.time(),entry_monotonic=time.monotonic(),closed=False)
        state['child_invocations'].append(row);persist()
        with (output/(phase+'.log')).open('x',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-u','-m',module,*args],cwd=d.ROOT,stdout=log,stderr=subprocess.STDOUT)
            identity=d.process_snapshot(child.pid);assert identity and identity['ppid']==state['controller_process']['pid']
            row['exact_process']=identity;persist()
            while child.poll() is None:
                left=min(remaining(),row['entry_epoch']+budget-time.time(),row['entry_monotonic']+budget-time.monotonic())
                if left<=0:
                    interrupt_owned_child()
                    child.wait(timeout=15);raise TimeoutError('Original bounded handoff deadline')
                try:child.wait(timeout=min(1.,left))
                except subprocess.TimeoutExpired:pass
            row.update(exit_code=child.returncode,closed=True,actual_exact_process_exited=d.exact_process_exited(identity),
                elapsed_epoch_seconds=time.time()-row['entry_epoch'],elapsed_monotonic_seconds=time.monotonic()-row['entry_monotonic'])
            persist();assert row['exit_code']==0 and row['actual_exact_process_exited']
        child=None
    persist()
    signal.signal(signal.SIGTERM,lambda *args: (_ for _ in ()).throw(InterruptedError('Handoff terminated; no replay')))
    try:
        fd=os.pidfd_open(upstream['pid']);assert d.process_snapshot(upstream['pid'])==upstream
        events=select.poll();events.register(fd,select.POLLIN)
        while True:
            left=remaining();assert left>0,'Upstream/whole deadline expired'
            if events.poll(max(1,int(min(60.,left)*1000))):break
        os.close(fd);fd=None
        assert d.exact_process_exited(upstream)
        campaign=d.read('runs/geobc_exploration/three_short_screens_v2/summary.json')
        assert campaign['controller_process']==upstream and campaign['boot_id']==boot
        assert campaign['closed'] and not campaign['active'] and campaign['status']=='training_stages_terminal'
        assert len(campaign['arms'])==3 and all(row['status']=='completed' and row['exit_code']==0 for row in campaign['arms'].values()),'Incomplete three-candidate screen: root review, no subset GT/ranking'
        bindings={}
        for candidate in plan['scientific']['candidates']:
            train=d.read(candidate['training_plan_identity']['path']);folder=d.path(train['output'])
            native=d.read(folder/'summary.json');outer=d.read(folder/'external_terminal.json')
            assert native['status']=='completed' and native['closed'] and not native['active'] and native['training_complete']
            assert outer['within_external_budget'] and outer['exit_code']==0 and outer['closed'] and not outer['active']
            assert all(d.exact_process_exited(outer[k]) for k in ('guard_process','child_process'))
            assert native['execution']['scientific_digest']==outer['scientific_digest']==train['scientific_digest']
            assert native['step']==native['optimizer_step_completions']==native['optimizer_step_attempts']==1200
            assert native['actual_largest_fullN_gradient_check']['passed']
            assert native['endpoints']['1200']['full_TRAIN']['all_cases_complete']
            assert all(d.sha(p)==h for p,h in train['source_pins'].items())
            bindings[candidate['name']]={kind:d.identity(folder/file,hashed=True) for kind,file in (
                ('summary','summary.json'),('external','external_terminal.json'),('status','status.json'))}
        root=d.read(plan['root_review']);assert root['passed'] and root['whole_source_reviewed']
        assert not root['actual_execution_released'] and root['training_bindings']=={}
        root.update(training_bindings=bindings,actual_execution_released=True,actual_terminal_release_UTC=d.utc(),
            actual_upstream_controller_exited=True,conditional_release_source_sha256=d.sha(OWN))
        d.write(plan['root_review'],root)
        from geobc.macro_screen_readout_contract_v2 import approved_contract
        approved_contract(PLAN)  # Full original resource/cost/RNG/endpoint/scope qualification.
        stage=d.path(plan['output']);assert not (stage/'external_launch.json').exists()
        state['readout_launches']=1;persist()
        invoke('knownDEV180_barrier','tools.run_geobc_macro_screen_readout_guard_v2',['--start','--plan',PLAN],1845)
        summary=d.read(stage/'summary.json');external=d.read(stage/'external_terminal.json')
        assert summary['status']=='completed' and summary['protocol_qualified'] and external['within_external_budget'] and external['exit_code']==0
        assert all(d.exact_process_exited(external[k]) for k in ('guard_process','child_process'))
        state['CPU_exports']=1;persist()
        invoke('CPU_one_page','tools.export_geobc_three_short_screens',['--export'],180)
        assert d.read('results/geobc_three_short_screens/manifest.json')['status']=='completed'
        state.update(status='completed',phase='await_root_actual_PNG_analysis_and_next_macro_decision',final_solver_selected=False)
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error),automatic_replay=False)
        raise
    finally:
        if child is not None and child.poll() is None:
            interrupt_owned_child();child.wait(timeout=15)
        if fd is not None:os.close(fd)
        state.update(active=False,closed=True);persist();d.write(output/'summary.json',state)
    print({'status':state['status'],'readout_launches':state['readout_launches'],'CPU_exports':state['CPU_exports']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true')
    if parser.parse_args().start:run()
