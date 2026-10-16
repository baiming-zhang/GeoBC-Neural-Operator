"""Wait for the prior exact guard to exit, then rerun every early-stopped arm."""
import time,json,os,select,subprocess,sys
from pathlib import Path


def main():
    previous=Path('runs/uniform_onecycle_u_overfitting');target=Path('runs/fullcycle_u_overfitting')
    config_path=Path('configs/fullcycle_u_overfitting.json')
    workspace=Path.cwd().parents[1];record_path=workspace/'notes/.tmp_fullcycle_queue.json'
    launch=json.loads((previous/'launch.json').read_text());entry=time.time()
    def record(phase,**extra):
        data=dict(phase=phase,queue_entry_epoch=entry,waiting_seconds=time.time()-entry,previous_run=str(previous),target_run=str(target),**extra)
        temp=record_path.with_suffix('.json.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(record_path)
        print(json.dumps(data),flush=True)
    if target.exists():raise FileExistsError('Rerun already exists; never start twice')
    if not (previous/'terminal.json').exists():
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=launch['boot_id']:raise RuntimeError('Prior boot changed')
        identity=launch['guard_process'];pid=identity['pid']
        try:fd=os.pidfd_open(pid)
        except ProcessLookupError:
            if not (previous/'terminal.json').exists():raise RuntimeError('Prior guard exited without terminal')
        else:
            try:
                raw=Path(f'/proc/{pid}/stat').read_text();fields=raw[raw.rfind(')')+2:].split()
                argv=Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0')
                if int(fields[19])!=identity['start_ticks'] or int(fields[1])!=identity['ppid'] or argv!=identity['argv']:raise RuntimeError('Prior exact guard mismatch')
                record('waiting_for_prior_guard',guard_process=identity)
                poller=select.poll();poller.register(fd,select.POLLIN)
                timeout=max(0,int((launch['deadline_epoch']-time.time()+30)*1000))
                if not poller.poll(timeout):raise TimeoutError('Prior absolute deadline passed')
            finally:os.close(fd)
    terminal=json.loads((previous/'terminal.json').read_text());summary=json.loads((previous/'summary.json').read_text())
    if terminal['exit_code']!=0 or terminal['active'] or not terminal['closed'] or not terminal['within_budget'] or summary['active'] or not summary['closed'] or summary['status']!='completed' or len(summary['arms'])!=16:raise RuntimeError('Prior stage did not close successfully; no replay launched')
    selected=[]
    for name,arm in summary['arms'].items():
        if arm['status']!='completed':raise RuntimeError('Prior arm failed; not an early-stop replay')
        if arm['updates']<5000:
            if arm['stop_reason']!='early_stopping_plateau':raise RuntimeError('Incomplete arm was not early stopped')
            selected.append(name)
        elif arm['updates']!=5000 or arm.get('endpoint',{}).get('step')!=5000:raise RuntimeError('Prior endpoint invalid')
    config=json.loads(config_path.read_text());config['selected_arms']=selected
    if not selected:raise RuntimeError('No early-stopped arms need rerun')
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    record('launching_fullcycle_reruns',selected_arms=selected,queue_wait_excluded_from_additive_training_cost=True)
    command=[sys.executable,'-B','-u','-m','tools.run_experiment','tools.train_sample_overfitting','--config',str(config_path),'--source',str(workspace/'notes/overfit_sample_sources.json'),'--output',str(target),'--lock',str(workspace/'runs/geobc_exploration/.coordinator.lock')]
    result=subprocess.run(command)
    record('reruns_closed',exit_code=result.returncode,selected_arms=selected)
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
