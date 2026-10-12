"""One fixed sixload coarse-PCG attempt; sameK/numeric references, GT free."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,json,os,signal
from pathlib import Path

def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.sixload_coarse_contract_v102 import approved_contract
    p,r=approved_contract(plan_path);s=p['scientific'];o=d.path(p['output']);outer=json.loads(os.environ['GEOBC_SIXLOAD_COARSE_V102_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest'] and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=p['scientific_digest'],boot_id=outer['boot_id']);d.write(o/'execution.json',execution)
    a=dict(status='running',active=True,closed=False,phase='cache',execution=execution,input_cache_reads=0,operator_cache_reads=0,numerical_reference_reads=0,owned_array_reads=0,GT_reads=0,neural_PT_reads=0,neural_inference=0,training_updates=0,scaler_fits=0,FINAL_reads=0,raw_H5_deck_GRID_reads=0,geometry_operator_rebuilds=0,inherited8fineK_guards_repeated=0,prediction_inventory={},fixed_endpoint_denominator=6,committed_finite_endpoints=0,completed_groups={},combined_solve_seconds=0.,peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0,protocol_qualified=False,cost_gate_passed=False,final_solver_selected=False)
    torch=None;lock=None;ledger=None
    def elapsed(tick):return max(time.time()-tick[0],time.monotonic()-tick[1])
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Whole900 earlier dualclock deadline')
        if torch is not None and torch.cuda.is_initialized():
            a['peak_allocated_bytes']=max(a['peak_allocated_bytes'],torch.cuda.max_memory_allocated());a['peak_reserved_bytes']=max(a['peak_reserved_bytes'],torch.cuda.max_memory_reserved());free,total=torch.cuda.mem_get_info();a['peak_total_device_used_bytes_sampled']=max(a['peak_total_device_used_bytes_sampled'],total-free)
            if max(a[k]for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))>30500000000:raise MemoryError('30.5GB cap')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        if ledger is not None:a['coarse_ledger']=dict(ledger.data)
        d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External coarse guard')))
    try:
        import fcntl
        lock=d.path(s['budgets']['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch as tm
        torch=tm
        from geobc import elastic_pilot as old,elastic_coarse as coarse
        from geobc.load_linear_physics_pcg_v100 import ParallelElastic
        from geobc.sixload_coarse_v102 import Ledger,ThreeLaneCoarse,qualify,solve_group
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        total=torch.cuda.get_device_properties(0).total_memory;assert 30500000000<total;torch.cuda.set_per_process_memory_fraction(30500000000/total,0);torch.cuda.reset_peak_memory_stats()
        setup=(time.time(),time.monotonic());persist()
        def read_array(item):
            guard();expected={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')};expected['path']=d.relative(d.path(item['path']));assert d.identity(item['path'],hashed=True)==expected
            value=np.load(d.path(item['path']),allow_pickle=False);a['owned_array_reads']+=1
            dtype={'torch.float64':'float64','torch.float32':'float32','torch.int64':'int64'}.get(item['dtype'],item['dtype'])
            assert list(value.shape)==item['shape'] and str(value.dtype)==dtype and np.isfinite(value).all();return value
        assert d.identity(s['input_cache_identity']['path'],hashed=True)==s['input_cache_identity']
        with np.load(d.path(s['input_cache_identity']['path']),allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files}
        a['input_cache_reads']+=1;assert all(np.isfinite(v).all()for v in arrays.values())
        forces=torch.as_tensor(arrays.pop('all_six_forces'),device='cuda');inputs={k:torch.as_tensor(v,device='cuda')for k,v in arrays.items()};del arrays
        assert forces.shape==(6,188196,3) and forces.dtype==torch.float64
        operator_arrays={k:torch.as_tensor(read_array(v),device='cuda')for k,v in s['operator_cache'].items()};a['operator_cache_reads']=3
        assert inputs['vertices_mm'].shape==(188196,3) and inputs['cells10'].shape==(117554,10) and inputs['cells10'].dtype==torch.int64
        assert operator_arrays['gradients'].shape==(117554,4,10,3) and operator_arrays['weights'].shape==(117554,4) and operator_arrays['free_cholesky'].shape==(len(inputs['free_node_ids']),3,3)
        lam,mu=map(float,inputs['lame'].tolist());op=old.MatrixFreeElastic(torch,inputs['cells10'],operator_arrays['gradients'],operator_arrays['weights'],188196,lam,mu)
        chol=operator_arrays['free_cholesky'];batched=ParallelElastic(torch,op,guard)
        a['phase']='coarse_setup';persist();ledger=Ledger(torch,guard,persist,o,setup)
        P,PT,corners,coarse_map,free_map,mapping=coarse.make_mapping(torch,inputs,ledger)
        Ac,diagonal,gamma,assembly=coarse.assemble_coarse(torch,inputs,operator_arrays,coarse_map,free_map,ledger)
        scalar=coarse.AdditiveCoarse(torch,P,PT,Ac,diagonal,gamma,chol,ledger)
        algebra=coarse.algebra_guards(torch,scalar,op,inputs,ledger)
        parallel=ThreeLaneCoarse(torch,scalar,guard);qualification=qualify(torch,parallel,scalar,len(inputs['free_node_ids']))
        assert algebra['passed'] and qualification['passed'];a.update(coarse_mapping=mapping,coarse_assembly=assembly,coarse_algebra_guards=algebra,parallel_M_qualification=qualification)
        cache=o/'coarse_cache';cache.mkdir(exist_ok=False)
        # Save constructed sparse/state cache once; technical recovery can retain it.
        saved={k:coarse.save_sparse(np,old,cache,k,m)for k,m in [('P',P),('PT',PT),('Ac',Ac)]}
        saved.update(diagonal=old._save_array(np,cache/'diagonal.npy',diagonal),corners=old._save_array(np,cache/'corners.npy',corners))
        for item in saved.values():
            if 'shape' in item and 'rowptr' not in item:item.update(d.identity(item['path'],hashed=True))
            elif 'rowptr'in item:
                for part in ('rowptr','columns','values'):item[part].update(d.identity(item[part]['path'],hashed=True))
        d.write(cache/'manifest.json',dict(status='completed',scientific_digest=p['scientific_digest'],mapping=mapping,assembly=assembly,algebra=algebra,parallel_qualification=qualification,arrays=saved))
        ledger.check();a['cache_and_coarse_setup_seconds']=elapsed(setup);assert a['cache_and_coarse_setup_seconds']<=120;ledger.setup_active=False;persist()
        for i,names in enumerate(s['selection']['groups']):
            guard();a['phase']='PCG_group'+str(i);persist();tick=(time.time(),time.monotonic())
            def remaining_solve():return 600-a['combined_solve_seconds']-elapsed(tick)
            assert remaining_solve()>0
            inv,cost=solve_group(torch,np,batched,chol,inputs,forces[i*3:i*3+3],names,o/('group'+str(i)),a,guard,persist,remaining_solve,parallel.apply)
            wall=elapsed(tick);assert wall<=300;a['combined_solve_seconds']+=wall;a['completed_groups'][str(i)]=dict(names=names,wall_seconds=wall,solver_reported_seconds=cost,initial_guess='independent_zero_all3')
            a['prediction_inventory'].update(inv);a['committed_finite_endpoints']+=3;persist()
        assert a['combined_solve_seconds']<=600 and a['committed_finite_endpoints']==6
        a['phase']='all6_finite_barrier';persist();tick=(time.time(),time.monotonic());fields={};records={}
        names=['ver','hor','tor','mix0','mix1','mix2'];assert set(a['prediction_inventory'])==set(names)
        for name in names:
            item=a['prediction_inventory'][name];assert d.identity(item['path'],hashed=True)==item;rec=d.read(item['path']);assert rec['status']=='completed' and rec['initial_guess']=='independent_zero' and rec['all_original_nodes']==188196 and rec['key']==name
            value=read_array(rec['physical_U']);assert value.shape==(188196,3) and value.dtype==np.float64;fields[name]=value;records[name]=rec
        d.write(o/'numeric_prediction_barrier.json',dict(status='completed',all6_original_nodes=True,reference_reads_before_barrier=0,GT_reads=0,inventory=a['prediction_inventory']));a['prediction_barrier_complete']=True;persist()
        metrics={}
        for name in names:
            item=s['numeric_reference_inventory'][name];assert d.identity(item['path'],hashed=True)==item;refrec=d.read(item['path']);assert refrec['numerical_state']=='converged' and refrec['true_relative_residual']<=1e-8 and refrec['initial_guess']=='independent_zero'
            ref=read_array(refrec['physical_U']);a['numerical_reference_reads']+=1;assert ref.shape==(188196,3) and ref.dtype==np.float64
            norm=float(np.square(ref).sum());sse=float(np.square(fields[name]-ref).sum());rms=(sse/len(ref))**.5;rel=(sse/norm)**.5 if norm>0 else None
            metrics[name]=dict(relative_U=rel,RMS_mm=rms,zero_reference=norm==0,same_solution_pass=(rel<=1e-5 if rel is not None else rms<=1e-12),true_relative_residual=records[name]['true_relative_residual'],converged=records[name]['numerical_state']=='converged',iterations=records[name]['iterations'])
            del ref;guard()
        a['endpoint_records']=records;a['same_solution_metrics']=metrics;a['readout_seconds']=elapsed(tick);assert a['readout_seconds']<=120 and a['numerical_reference_reads']==6
        precision=all(v['same_solution_pass'] and v['converged'] and v['true_relative_residual']<=1e-8 for v in metrics.values())
        a.update(protocol_qualified=True,precision_gate_passed=precision,status='completed',phase='terminal',scientific_outcome='complete_sixload_precision_evidence_cost_gate_requires_external_terminal',operator_calls=dict(scalar=op.calls,batch=batched.calls,coarse_parallel=parallel.calls),actual_GT_new_physical_accuracy=False)
    except Exception as error:
        from geobc.elastic_pilot import ScientificNegative
        if isinstance(error,ScientificNegative):a.update(status='completed',phase='terminal_scientific_negative',reason=str(error),scientific_outcome='coarse_or_PCG_scientific_negative',protocol_qualified=False)
        else:a.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error));raise
    finally:
        a.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        if ledger is not None:a['coarse_ledger']=dict(ledger.data)
        d.write(o/'summary.json',a);d.write(o/'status.json',a)
        if lock is not None:lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');p.add_argument('--plan');args=p.parse_args()
    if args.start:
        if not args.plan:p.error('--start requires frozen --plan')
        run(args.plan)
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
