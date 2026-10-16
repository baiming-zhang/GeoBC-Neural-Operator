"""Serial ablation queue: start grid TS only after exact point-stage closure."""
import json,os,select,sys,time
from pathlib import Path
from tools.run_experiment import process_identity

def main():
 root=Path.cwd().parents[1];prior=Path('runs/transolver_hybrid_u_overfitting');target=Path('runs/transolver_grid_hybrid_u_overfitting');record=root/'notes/.tmp_ts_hybrid_queue.json'
 launch=json.loads((prior/'launch.json').read_text());entry=time.time()
 def save(phase,**kw):
  value=dict(phase=phase,entry_epoch=entry,waiting_seconds=time.time()-entry,queue_process=process_identity(os.getpid()),boot_id=launch['boot_id'],**kw)
  record.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True)
 if target.exists():raise FileExistsError('Grid run already exists; never launch twice')
 if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=launch['boot_id']:raise RuntimeError('Prior boot changed')
 if not (prior/'terminal.json').exists():
  expected=launch['guard_process'];fd=os.pidfd_open(expected['pid'])
  try:
   if process_identity(expected['pid'])!=expected:raise RuntimeError('Prior exact identity mismatch')
   save('waiting_for_point_stage',prior_guard=expected)
   poller=select.poll();poller.register(fd,select.POLLIN)
   if not poller.poll(max(0,int((launch['deadline_epoch']-time.time()+30)*1000))):raise TimeoutError('Prior deadline passed')
  finally:os.close(fd)
 terminal=json.loads((prior/'terminal.json').read_text());s=json.loads((prior/'summary.json').read_text())
 if terminal['exit_code']!=0 or not terminal['within_budget'] or terminal['active'] or not terminal['closed'] or s['active'] or not s['closed'] or len(s['arms'])!=3:raise RuntimeError('Point stage failed; preserve state, do not launch/replay')
 for identity in (launch['guard_process'],launch['child_process']):
  if process_identity(identity['pid'])==identity:raise RuntimeError('Prior exact process still live')
 for a in s['arms'].values():
  if a['status']!='completed' or a['updates']!=5000 or a.get('endpoint',{}).get('step')!=5000:raise RuntimeError('Point stage incomplete')
 save('launching_grid_stage',queue_wait_excluded_from_additive_cost=True)
 command=[sys.executable,'-B','-u','-m','tools.run_experiment','tools.train_transolver_grid_hybrid','--config','configs/transolver_grid_hybrid_u_overfitting.json','--source',str(root/'notes/fifty_u_sources.json'),'--output',str(target),'--lock',str(root/'runs/geobc_exploration/.coordinator.lock')]
 os.execv(sys.executable,command)

if __name__=='__main__':main()
