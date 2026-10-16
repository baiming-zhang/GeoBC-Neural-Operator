"""Wait for the original GPU stage, then recover only its failed ++ arm.

The inherited whole-stage and conservative original-arm deadlines never move.
Waiting overlaps the original stage and is not billed twice as GPU-stage cost.
"""
import argparse,json,os,select,signal,subprocess,sys,time
from pathlib import Path
from tools.run_experiment import process_identity


def recovery_clock(original,summary,now_epoch,now_monotonic,boot):
    if original['boot_id']!=boot:raise ValueError('Same boot required for conservative dual-clock recovery')
    before=[]
    for method in ('mlp','ngp4','hashgrid','tcnn_hash','fno','transolver'):
        a=summary['arms']['fifty_'+method]
        if a['updates']!=5000 or a['status']!='completed':raise ValueError('Preceding full5000 arms required')
        before.append(a['elapsed_seconds'])
    # Excluding loading and gaps places the start BEFORE its actual start.
    # This lower bound can only shorten, never extend, its original allowance.
    starts={k:original['entry_epoch' if k=='UTC' else 'entry_monotonic']+sum(a[k] for a in before) for k in ('UTC','monotonic')}
    deadlines={k:min(original['deadline_epoch' if k=='UTC' else 'deadline_monotonic'],starts[k]+10800) for k in starts}
    if min(deadlines['UTC']-now_epoch,deadlines['monotonic']-now_monotonic)<=60:raise TimeoutError('Inherited recovery allowance exhausted')
    return dict(entry_epoch=now_epoch,entry_monotonic=now_monotonic,deadline_epoch=deadlines['UTC'],deadline_monotonic=deadlines['monotonic'],boot_id=boot,
        original_entry_epoch=original['entry_epoch'],original_entry_monotonic=original['entry_monotonic'],original_deadline_epoch=original['deadline_epoch'],original_deadline_monotonic=original['deadline_monotonic'],
        conservative_original_arm_start=starts,arm_allowance_seconds=10800,deadline_policy='minimum of unchanged whole-stage deadline and conservative original ++ arm-start lower bound +10800; includes intervening wait')


def run(args):
    original=json.loads((args.original/'launch.json').read_text());expected=original['guard_process']
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();queue_start=time.time()
    def persist(phase,**kwargs):
        args.queue_state.write_text(json.dumps(dict(phase=phase,queue_process=process_identity(os.getpid()),original_guard=expected,boot_id=boot,waiting_since_epoch=queue_start,**kwargs),indent=2)+'\n')
    if boot!=original['boot_id']:raise ValueError('Original boot changed')
    actual=process_identity(expected['pid'])
    if actual is not None:
        if actual!=expected:raise ValueError('Original guard identity changed')
        fd=os.pidfd_open(expected['pid']);persist('waiting_for_original_guard_exit')
        try:
            poll=select.poll();poll.register(fd,select.POLLIN)
            if not poll.poll(max(0,int((original['deadline_epoch']-time.time())*1000))):raise TimeoutError('Original whole-stage deadline reached')
        finally:os.close(fd)
    summary=json.loads((args.original/'summary.json').read_text());terminal=json.loads((args.original/'terminal.json').read_text())
    if terminal['exit_code']!=0 or terminal['active'] or not terminal['closed'] or not terminal['within_budget'] or summary['active'] or not summary['closed']:raise ValueError('Original stage must close successfully before isolated arm recovery')
    for name,a in summary['arms'].items():
        if name=='fifty_transolverpp':
            if a['status']!='failed' or a['updates']!=21 or a.get('error')!='Declared GPU memory cap' or a.get('partial_backward') or a.get('optimizer_step_in_progress'):raise ValueError('Only complete-step21 memory-cap failure may be recovered')
        elif a['status']!='completed' or a['updates']!=5000 or a.get('endpoint',{}).get('step')!=5000:raise ValueError('Retain all other7 complete endpoints')
    if len(summary['arms'])!=8 or any(process_identity(original[k]['pid'])==original[k] for k in ('guard_process','child_process')):raise ValueError('Original exact processes must have exited')
    launch=recovery_clock(original,summary,time.time(),time.monotonic(),boot)
    launch['guard_process']=process_identity(os.getpid());launch['queue_wait_seconds']=time.time()-queue_start
    args.output.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env['GEOBC_EXPERIMENT_CLOCK']=json.dumps(launch)
    command=[sys.executable,'-B','-u','-m','tools.recover_fifty_u_overfitting','--original',str(args.original),'--config',str(args.config),'--source',str(args.source),'--output',str(args.output),'--lock',str(args.lock)]
    child=None;timeout=False
    persist('recovering_saved_step21',output=str(args.output),inherited_deadline_epoch=launch['deadline_epoch'])
    try:
        with (args.output/'process.log').open('x') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            launch['child_process']=process_identity(child.pid)
            (args.output/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
            while child.poll() is None:
                remaining=min(launch['deadline_epoch']-time.time(),launch['deadline_monotonic']-time.monotonic())
                if remaining<=0:
                    timeout=True;os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                    break
                try:child.wait(timeout=min(1,remaining))
                except subprocess.TimeoutExpired:pass
            child.wait()
    finally:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=5)
        elapsed=dict(UTC=time.time()-launch['entry_epoch'],monotonic=time.monotonic()-launch['entry_monotonic'])
        respected=time.time()<=launch['deadline_epoch'] and time.monotonic()<=launch['deadline_monotonic']
        t=dict(exit_code=child.returncode if child else None,active=False,closed=True,timeout=timeout,elapsed_seconds=elapsed,within_budget=child is not None and child.returncode==0 and respected and not timeout,original_deadline_respected=respected,inherited_clock=launch)
        (args.output/'terminal.json').write_text(json.dumps(t,indent=2)+'\n');persist('terminal',terminal=t)
    print(json.dumps(t));return 0 if t['within_budget'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,default=Path('runs/fifty_u_overfitting'));p.add_argument('--output',type=Path,default=Path('runs/fifty_u_recovery'));p.add_argument('--config',type=Path,default=Path('configs/fifty_u_overfitting.json'));p.add_argument('--source',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);p.add_argument('--queue-state',type=Path,required=True);a=p.parse_args();raise SystemExit(run(a))
