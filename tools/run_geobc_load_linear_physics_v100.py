"""One released raw-input CAD, fixedK and six independently solved GPU RHS."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,json,os,signal
from pathlib import Path


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.load_linear_physics_contract_v100 import approved_contract
    p,r,truths=approved_contract(plan_path);s=p['scientific'];o=d.path(p['output']);outer=json.loads(os.environ['GEOBC_LOAD_LINEAR_PHYSICS_V100_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest']and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()and d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=p['scientific_digest'],boot_id=outer['boot_id']);d.write(o/'execution.json',execution)
    a=dict(status='running',active=True,closed=False,phase='targetfree_input',execution=execution,source_body_hash_reads=0,H5_input_attempts=0,H5_input_reads=0,H5_members_read=[],GRID_map_reads=0,deck_parses=0,deck_header_reads=0,owned_array_reads=0,GT_attempts=0,GT_reads=0,raw_CSV_H5_nodal_target_reads=0,training_updates=0,scaler_fits=0,neural_PT_reads=0,neural_inference=0,FINAL_reads=0,prediction_inventory={},fixed_endpoint_denominator=6,committed_finite_endpoints=0,completed_groups={},combined_solve_seconds=0.,peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0,protocol_qualified=False,physics_reference_compatible=False,final_solver_selected=False)
    torch=None;lock=None
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Whole1800 earlier dualclock deadline')
        if torch is not None and torch.cuda.is_initialized():
            a['peak_allocated_bytes']=max(a['peak_allocated_bytes'],torch.cuda.max_memory_allocated());a['peak_reserved_bytes']=max(a['peak_reserved_bytes'],torch.cuda.max_memory_reserved());free,total=torch.cuda.mem_get_info();a['peak_total_device_used_bytes_sampled']=max(a['peak_total_device_used_bytes_sampled'],total-free)
            if max(a[k]for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))>s['budget']['memory_bytes']:raise MemoryError('30.5GB cap')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External physical guard')))
    try:
        import fcntl
        lock=d.path(s['budget']['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch as tm
        torch=tm
        from geobc import elastic_pilot as old,elastic_pilot_data as olddata
        from geobc.load_linear_physics_inputs_v100 import prepare_inputs
        from geobc.load_linear_physics_pcg_v100 import ParallelElastic,solve_group
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        total=torch.cuda.get_device_properties(0).total_memory;assert s['budget']['memory_bytes']<total;torch.cuda.set_per_process_memory_fraction(s['budget']['memory_bytes']/total,0);torch.cuda.reset_peak_memory_stats()
        setup=(time.time(),time.monotonic());persist();arrays,force_np,record=prepare_inputs(np,s,a,guard,setup[1]);guard()
        cache=o/'geometry_and_rhs_input.npz'
        with cache.open('xb')as f:np.savez(f,**arrays,all_six_forces=force_np)
        a['input_cache_identity']=d.identity(cache,hashed=True)
        inputs={k:torch.as_tensor(v,device='cuda')for k,v in arrays.items()};forces=torch.as_tensor(force_np,device='cuda');del arrays,force_np
        a['phase']='geometry_operator8guards';persist();remain=min(300-max(time.time()-setup[0],time.monotonic()-setup[1]),outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic());assert remain>0
        began=time.monotonic();prior=300-remain;op,chol,diagonal,geometry=old._build_geometry(torch,inputs,record,began,prior);guard()
        algebra=old._algebra_guards(torch,op,diagonal,inputs,record,began,prior);assert algebra['passed']and algebra['K_applications']==8;guard()
        a['geometry']=geometry;a['algebra_guards']=algebra;a['phase']='parallel_operator_actual_fullN_qualification';persist()
        batched=ParallelElastic(torch,op,guard);generator=torch.Generator(device='cuda').manual_seed(2026);w=torch.zeros_like(forces[:3]);w[:,inputs['free_node_ids']]=torch.randn((3,len(inputs['free_node_ids']),3),dtype=torch.float64,device='cuda',generator=generator)
        actual=batched.apply(w);ref=torch.stack([op.apply(w[i])for i in range(3)]);relative=torch.linalg.vector_norm(actual-ref,dim=(1,2))/torch.linalg.vector_norm(ref,dim=(1,2));assert torch.isfinite(actual).all()and torch.isfinite(relative).all()and bool((relative<=1e-9).all())
        a['parallel_K_qualification']=dict(passed=True,actual_all_original_nodes=len(inputs['vertices_mm']),actual_RHS_lanes=3,relative_K_parity=relative.cpu().tolist(),old_extra_qualification_K_calls=3,new_batched_qualification_K_calls=1,original8guards_repeated=False);del w,actual,ref,relative;guard()
        for name,value in [('gradients',op.gradients),('weights',op.weights),('free_cholesky',chol)]:
            item=old._save_array(np,o/'operator_cache'/ (name+'.npy'),value);item.update(d.identity(item['path'],hashed=True));a.setdefault('operator_cache',{})[name]=item
        a['input_operator_seconds']=max(time.time()-setup[0],time.monotonic()-setup[1]);assert a['input_operator_seconds']<=300;persist();del diagonal
        for i,names in enumerate(s['parallel_RHS_groups']):
            guard();a['phase']='PCG_group'+str(i);persist();tick=(time.time(),time.monotonic())
            def remaining_solve():return 900-a['combined_solve_seconds']-max(time.time()-tick[0],time.monotonic()-tick[1])
            assert remaining_solve()>0
            inv,elapsed=solve_group(torch,np,batched,chol,inputs,forces[i*3:i*3+3],names,o/('group'+str(i)),a,guard,persist,remaining_solve)
            cost=max(time.time()-tick[0],time.monotonic()-tick[1]);assert cost<=300
            a['combined_solve_seconds']+=cost;a['completed_groups'][str(i)]=dict(names=names,wall_seconds=cost,solver_reported_seconds=elapsed,initial_guess='independent_zero_all3',original_geometry_operator_rebuilt=False);a['prediction_inventory'].update(inv);a['committed_finite_endpoints']+=3;persist()
        assert a['combined_solve_seconds']<=900 and a['committed_finite_endpoints']==6
        a['phase']='all6_fields_equilibrium_barrier_GTclosed';persist();readout=(time.time(),time.monotonic())
        def read_array(item):
            guard();expected={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')};expected['path']=d.relative(d.path(item['path']));assert d.identity(item['path'],hashed=True)==expected
            value=np.load(d.path(item['path']),mmap_mode='r',allow_pickle=False);a['owned_array_reads']+=1;assert list(value.shape)==item['shape']and str(value.dtype)==item['dtype']and np.isfinite(value).all();return value
        n=s['expected_geometry']['nodes'];fields={};endpoint_records={}
        for name in ['ver','hor','tor','mix0','mix1','mix2']:
            item=a['prediction_inventory'][name];assert d.identity(item['path'],hashed=True)==item;rec=d.read(item['path']);assert rec['all_original_nodes']==n and rec['initial_guess']=='independent_zero'and rec['key']==name and rec['status']=='completed'
            value=read_array(rec['physical_U']);assert value.shape==(n,3)and value.dtype==np.float64;fields[name]=value;endpoint_records[name]=rec
        d.write(o/'physics_prediction_barrier.json',dict(status='completed',all6_original_nodes=True,GT_reads_before_barrier=0,inventory=a['prediction_inventory'],endpoint_numerical_states={k:v['numerical_state']for k,v in endpoint_records.items()}));a['prediction_barrier_complete']=True;persist()
        def metrics(pred,ref):
            norm=float(np.square(ref).sum());sse=float(np.square(pred-ref).sum());return dict(U_relative_L2=(sse/norm)**.5 if norm>0 else None,U_RMS_mm=(sse/len(ref))**.5,U_max_mm=float(np.linalg.norm(pred-ref,axis=1).max()),fixed_allN=len(ref),zero_reference=norm<=0)
        canonical={};a['phase']='shared_original_GT3_once';persist();cad=s['selection']['CAD']
        for load in ('ver','hor','tor'):
            key=cad+'/'+load;item=truths['records'][key]['physical'];guard();a['GT_attempts']+=1;gt=read_array(item);a['GT_reads']+=1;assert gt.shape==(n,4)and gt.dtype==np.float32
            canonical[key]=metrics(fields[load],gt[:,:3].astype(np.float64));del gt;persist()
        newload={}
        for candidate in ('FNO20','FNO_HASH20','TS20'):
            newload[candidate]={}
            for kind in ('raw','adapter'):
                rows={}
                for i in range(3):
                    key=cad+'/mix'+str(i)+'@'+kind;item=s['neural_prediction_inventory'][candidate+':'+key];assert d.identity(item['path'],hashed=True)==item;rec=d.read(item['path']);pred=read_array(rec['physical']);assert pred.shape==(n,4);rows[key]=metrics(pred[:,:3].astype(np.float64),fields['mix'+str(i)]);del pred
                vals=[v['U_relative_L2']for v in rows.values()];complete=len(vals)==3 and all(v is not None and np.isfinite(v)for v in vals);newload[candidate][kind]=dict(fixed_denominator=3,per_case=rows,all3_complete=complete,mean_U=sum(vals)/3 if complete else None,worst_U=max(vals)if complete else None,reference='independently_solved_approximate_curvedP2_fourQ_not_originalOptiStruct_newloadGT')
        a['endpoint_records']=endpoint_records;a['canonical_original_GT_U']=canonical;a['mixed_neural_U_against_independent_approx_P2']=newload
        allconv=all(v['numerical_state']=='converged'and v['true_relative_residual']<=1e-8 for v in endpoint_records.values());canpass=len(canonical)==3 and all(v['U_relative_L2']is not None and v['U_relative_L2']<=.01 for v in canonical.values())
        a['physics_reference_compatible']=allconv and canpass;a['reference_compatibility_gate']=dict(passed=a['physics_reference_compatible'],all6_true_residual_pass=allconv,canonical3_le1percent=canpass,does_not_claim_OptiStruct_equivalence=True,does_not_select_final_solver=True)
        a['final_readout_seconds']=max(time.time()-readout[0],time.monotonic()-readout[1]);assert a['final_readout_seconds']<=300 and a['GT_attempts']==a['GT_reads']==3
        a['old_and_batch_operator_calls']=dict(old_total=op.calls,batched_total=batched.calls,old_guard_calls=8,old_qualification_calls=3,new_batch_qualification_calls=1,original_endpoint_equilibrium_K_calls=6);guard();a.update(status='completed',phase='terminal',protocol_qualified=True,scientific_outcome='compatible_approximate_physics_reference'if a['physics_reference_compatible']else'finite_completed_reference_compatibility_negative')
    except Exception as error:
        from geobc import elastic_pilot_data as olddata,elastic_pilot as old
        if isinstance(error,(olddata.UnsupportedInput,old.ScientificNegative)):
            a.update(status='completed',phase='terminal_scientific_negative',scientific_outcome='unsupported_input_operator_or_numerical_breakdown',reason=str(error),physics_reference_compatible=False,protocol_qualified=False)
        else:
            a.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error));raise
    finally:
        a.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'summary.json',a);d.write(o/'status.json',a)
        if lock is not None:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');parser.add_argument('--plan');args=parser.parse_args()
    if args.start:
        if not args.plan:parser.error('--start requires fixed --plan')
        run(args.plan)
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
