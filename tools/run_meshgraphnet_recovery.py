"""Run only the failed five-case arm under its unchanged original deadlines."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from tools.run_experiment import process_identity


def recovery_clock(original,summary,now_epoch,now_mono,boot):
    preceding=summary['arms']['single_meshgraphnet']['elapsed_seconds']
    # Omitting data loading and gaps puts this bound before the actual arm start.
    start=original['entry_epoch']+preceding['UTC']
    epoch=min(original['deadline_epoch'],start+21600)
    if boot==original['boot_id']:
        mono=min(original['deadline_monotonic'],original['entry_monotonic']+preceding['monotonic']+21600)
    else:
        # After reboot, conservatively charge all wall time since the lower bound
        # to the new monotonic budget as well. No downtime is credited back.
        mono=now_mono+max(0,epoch-now_epoch)
    if min(epoch-now_epoch,mono-now_mono)<=60:raise TimeoutError('Original arm deadline exhausted')
    return dict(entry_epoch=now_epoch,entry_monotonic=now_mono,deadline_epoch=epoch,deadline_monotonic=mono,boot_id=boot,
        original_boot_id=original['boot_id'],original_entry_epoch=original['entry_epoch'],
        conservative_original_arm_start_epoch=start,arm_seconds=21600,
        deadline_policy='unchanged whole/arm deadlines; loading omitted from start lower bound; downtime charged, never reset')


def validate_original(s,t,log):
    if s['active'] or not s['closed'] or t['active'] or not t['closed'] or t['timeout'] or not t['original_deadline_respected']:raise ValueError('Closed original attempt required')
    if set(s['arms'])!={'single_meshgraphnet','five_meshgraphnet'}:raise ValueError('No fifty model may initialize')
    a=s['arms']['single_meshgraphnet'];b=s['arms']['five_meshgraphnet']
    if a['status']!='completed' or a['updates']!=5000 or a['endpoint']['step']!=5000:raise ValueError('Retain complete single endpoint')
    if b['status']!='failed' or b['updates']!=4 or b.get('optimizer_step_in_progress') or b['error_type']!='MemoryError' or b['error']!='Declared GPU memory cap':raise ValueError('Only known completed-step4 failure allowed')
    # Exact trace confirms the failing guard ran after forward and before loss/backward.
    if "guard();pred=model" not in log or "MemoryError: Declared GPU memory cap" not in log:raise ValueError('Pre-backward failure evidence required')


def run(a):
    old=json.loads((a.original/'launch.json').read_text());s=json.loads((a.original/'summary.json').read_text());t=json.loads((a.original/'terminal.json').read_text())
    validate_original(s,t,(a.original/'process.log').read_text())
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if boot==old['boot_id'] and any(process_identity(old[k]['pid'])==old[k] for k in ('guard_process','child_process')):raise ValueError('Original exact process still live')
    launch=recovery_clock(old,s,time.time(),time.monotonic(),boot);launch['guard_process']=process_identity(os.getpid())
    a.output.mkdir(parents=True,exist_ok=False);env=os.environ.copy();env['GEOBC_EXPERIMENT_CLOCK']=json.dumps(launch)
    command=[sys.executable,'-B','-u','-m','tools.recover_meshgraphnet_u_overfitting','--original',str(a.original),'--config',str(a.config),'--source',str(a.source),'--output',str(a.output),'--lock',str(a.lock)]
    child=None;timeout=False
    try:
        with (a.output/'process.log').open('x') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            launch['child_process']=process_identity(child.pid);(a.output/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
            while child.poll() is None:
                left=min(launch['deadline_epoch']-time.time(),launch['deadline_monotonic']-time.monotonic())
                if left<=0:
                    timeout=True;os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                    break
                try:child.wait(timeout=min(1,left))
                except subprocess.TimeoutExpired:pass
            child.wait()
    finally:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=5)
        elapsed=dict(UTC=time.time()-launch['entry_epoch'],monotonic=time.monotonic()-launch['entry_monotonic']);respected=time.time()<=launch['deadline_epoch'] and time.monotonic()<=launch['deadline_monotonic']
        terminal=dict(exit_code=child.returncode if child else None,timeout=timeout,active=False,closed=True,elapsed_seconds=elapsed,original_deadline_respected=respected,within_budget=child is not None and child.returncode==0 and respected and not timeout)
        (a.output/'terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
    print(json.dumps(terminal));return 0 if terminal['within_budget'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,default=Path('runs/meshgraphnet_u_overfitting'));p.add_argument('--output',type=Path,default=Path('runs/meshgraphnet_u_recovery'));p.add_argument('--config',type=Path,default=Path('configs/meshgraphnet_u_overfitting.json'));p.add_argument('--source',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);raise SystemExit(run(p.parse_args()))
