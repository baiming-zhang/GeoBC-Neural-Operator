"""GPU FP64 oracle amplitude/pattern diagnostic; no model or prediction rewrite."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
import json
import os
from pathlib import Path
import signal


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.field_amplitude_shape_contract_v75 import approved_contract,check_identity
    p,r,manifest,truth,up=approved_contract(plan_path)
    s=p['scientific'];outer=json.loads(os.environ['GEOBC_AMPLITUDE_SHAPE_V75_GUARD'])
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert outer['scientific_digest']==p['scientific_digest'] and outer['boot_id']==boot
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    out=d.path(p['output']);assert out.exists() and not (out/'execution.json').exists()
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,boot_id=boot,scientific_digest=p['scientific_digest'])
    d.write(out/'execution.json',execution)
    z=dict(status='running',active=True,closed=False,execution=execution,phase='prediction_barrier',
        fixed_case_denominator=60,fixed_candidate_case_denominator=120,prediction_read_attempts=0,prediction_reads=0,
        prediction_body_hash_checks=0,cached_GT_read_attempts=0,cached_GT_reads=0,cached_GT_body_hash_checks=0,
        raw_GT_reads=0,PT_body_reads=0,FINAL_reads=0,training_updates=0,model_forward_calls=0,scaler_fits=0,
        peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0,
        backend_external_workspace_exact_peak_known=False,cases={},aggregate={},automatic_replay=False,
        deployable_correction=False,label_relabeling=False,final_solver_selected=False)
    torch=None;lock=None;predictions={}

    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:
            raise TimeoutError('Frozen oracle-diagnostic deadline')
        if torch is not None and torch.cuda.is_initialized():
            z['peak_allocated_bytes']=max(z['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
            z['peak_reserved_bytes']=max(z['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
            free,total=torch.cuda.mem_get_info()
            z['peak_total_device_used_bytes_sampled']=max(z['peak_total_device_used_bytes_sampled'],total-free)
            if max(z[k]for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))>s['memory_limit_bytes']:
                raise MemoryError('30.5GB allocated/reserved/sampled cap')

    def persist():
        z.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(out/'status.json',z)

    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('Outer deadline')))
    try:
        guard()
        import fcntl
        lock=d.path(s['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch as torch_module
        torch=torch_module;torch.set_num_threads(4)
        device=torch.cuda.current_device();total=torch.cuda.get_device_properties(device).total_memory
        assert s['memory_limit_bytes']<total
        torch.cuda.set_per_process_memory_fraction(s['memory_limit_bytes']/total,device);torch.cuda.reset_peak_memory_stats()
        # Reopen and validate every immutable physical prediction before this
        # diagnostic reads any GT. Retain read-only mmap objects for later reuse.
        for n in s['candidate_names']:
            z['cases'][n]={}
            for k in s['case_keys']:
                guard();record=d.read(check_identity(up['inventory'][n+':'+k]))
                assert record['candidate']==n and record['key']==k and record['status']=='completed'
                assert record['signed_stress_unchanged'] and record['all_original_nodes']==manifest['records'][k]['num_nodes']
                z['prediction_read_attempts']+=1
                file=check_identity(record['physical']);z['prediction_body_hash_checks']+=1
                value=np.load(file,mmap_mode='r',allow_pickle=False);z['prediction_reads']+=1
                assert value.shape==(record['all_original_nodes'],4) and value.dtype==np.float32 and np.isfinite(value).all()
                predictions[n+':'+k]=value;persist()
        assert z['prediction_reads']==120 and z['cached_GT_reads']==0
        d.write(out/'prediction_barrier.json',dict(completed=True,full_physical_predictions=120,GT_reads_before_barrier=0,
            immutable_parent_barrier=s['prediction_barrier_identity']))
        z['phase']='shared_GT_once_and_FP64_reductions';persist()

        def decomposition(pred,target):
            pp=float(torch.sum(pred*pred));yy=float(torch.sum(target*target));py=float(torch.sum(pred*target))
            error=float(torch.sum((pred-target).square()))
            if yy==0:
                return dict(zero_GT=True,zero_prediction=pp==0,alpha=None,amplitude_ratio=None,raw_relative_L2=None,
                    oracle_shape_relative_L2=None,cosine=None,raw_squared_error=error)
            alpha=None if pp==0 else py/pp
            residual=yy if alpha is None else float(torch.sum((alpha*pred-target).square()))
            assert residual<=error+1e-10*max(yy,error)
            return dict(zero_GT=False,zero_prediction=pp==0,alpha=alpha,amplitude_ratio=(pp/yy)**.5,
                raw_relative_L2=(error/yy)**.5,oracle_shape_relative_L2=(residual/yy)**.5,
                cosine=None if pp==0 else py/(pp*yy)**.5,raw_squared_relative_error=error/yy,
                oracle_squared_relative_error=residual/yy,oracle_fraction_squared_error_removed=None if error==0 else 1-residual/error)

        for k in s['case_keys']:
            guard();z['cached_GT_read_attempts']+=1
            file=check_identity(truth['records'][k]['physical']);z['cached_GT_body_hash_checks']+=1
            target=np.load(file,mmap_mode='r',allow_pickle=False);z['cached_GT_reads']+=1
            assert target.shape==(manifest['records'][k]['num_nodes'],4) and target.dtype==np.float32 and np.isfinite(target).all()
            t=torch.tensor(np.asarray(target),dtype=torch.float64,device='cuda')
            for n in s['candidate_names']:
                pred=torch.tensor(np.asarray(predictions[n+':'+k]),dtype=torch.float64,device='cuda')
                values={field:decomposition(pred[:,:3]if field=='U'else pred[:,3],t[:,:3]if field=='U'else t[:,3])for field in ('U','S')}
                old=up['parent_summary']['candidates'][n]['full60']['per_case'][k]
                for field,metric in [('U','u_relative_l2'),('S','stress_relative_l2')]:
                    assert abs(values[field]['raw_relative_L2']-old[metric])<1e-8
                z['cases'][n][k]=values;del pred
            del t,target;guard();persist()
        assert z['cached_GT_reads']==z['cached_GT_read_attempts']==60
        for n in s['candidate_names']:
            z['aggregate'][n]={}
            for label,keys in [('full60',s['case_keys'])]+[(load+'20',[k for k in s['case_keys']if k.endswith('/'+load)])for load in ('ver','hor','tor')]:
                point={}
                for field in ('U','S'):
                    vals=[z['cases'][n][k][field]for k in keys]
                    assert len(vals)==len(keys)
                    complete=all(v['raw_relative_L2']is not None and v['oracle_shape_relative_L2']is not None for v in vals)
                    point[field]=dict(fixed_denominator=len(keys),valid_relative_cases=sum(v['raw_relative_L2']is not None for v in vals),
                        all_cases_complete=complete,mean_raw_relative_L2=None if not complete else sum(v['raw_relative_L2']for v in vals)/len(keys),
                        mean_oracle_shape_relative_L2=None if not complete else sum(v['oracle_shape_relative_L2']for v in vals)/len(keys),
                        fraction_mean_squared_error_removed=None if not complete or sum(v['raw_squared_relative_error']for v in vals)==0 else
                            1-sum(v['oracle_squared_relative_error']for v in vals)/sum(v['raw_squared_relative_error']for v in vals),
                        negative_alpha_cases=sum(v['alpha']is not None and v['alpha']<0 for v in vals))
                z['aggregate'][n][label]=point
        guard();z.update(status='completed',phase='terminal',protocol_qualified=True)
    except Exception as err:
        z.update(status='failed',phase='terminal_failure',protocol_qualified=False,error_type=type(err).__name__,error=str(err))
        raise
    finally:
        predictions.clear()
        z.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(out/'summary.json',z);d.write(out/'status.json',z)
        if lock is not None:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');parser.add_argument('--plan');args=parser.parse_args()
    if args.start:
        if not args.plan:parser.error('--start requires separately frozen --plan')
        run(args.plan)
    else:print('{"status":"source_only","scientific_IO":0,"GPU":0}')
