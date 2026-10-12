"""Complete20CAD canonicalU coverage:17new fixedcoarse groups and3reused immutable bases."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,json,os,signal,gc
from pathlib import Path

def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.physics_U_coverage_contract_v107 import approved_contract
    p,r,truths=approved_contract(plan_path);s=p['scientific'];o=d.path(p['output']);outer=json.loads(os.environ['GEOBC_PHYSICS_U_COVERAGE_V107_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest'] and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=p['scientific_digest'],boot_id=outer['boot_id']);d.write(o/'execution.json',execution)
    a=dict(status='running',active=True,closed=False,phase='source',execution=execution,source_body_hash_reads=0,H5_input_attempts=0,H5_input_reads=0,H5_members_read=[],GRID_map_reads=0,deck_parses=0,deck_header_reads=0,owned_array_reads=0,GT_attempts=0,GT_reads=0,raw_CSV_H5_nodal_target_reads=0,neural_PT_reads=0,neural_inference=0,training_updates=0,scaler_fits=0,FINAL_reads=0,prediction_inventory={},fixed_endpoint_denominator=60,committed_finite_endpoints=0,combined_solve_seconds=0.,case_records={},completed_groups={},peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0,protocol_qualified=False,physics_reference_compatible=False,coverage_gate_passed=False,final_solver_selected=False)
    torch=None;lock=None;ledger=None
    def elapsed(t):return max(time.time()-t[0],time.monotonic()-t[1])
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Whole2700 earlier absolute deadline')
        if torch is not None and torch.cuda.is_initialized():
            a['peak_allocated_bytes']=max(a['peak_allocated_bytes'],torch.cuda.max_memory_allocated());a['peak_reserved_bytes']=max(a['peak_reserved_bytes'],torch.cuda.max_memory_reserved());free,total=torch.cuda.mem_get_info();a['peak_total_device_used_bytes_sampled']=max(a['peak_total_device_used_bytes_sampled'],total-free)
            if max(a[k]for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))>30500000000:raise MemoryError('30.5GB cap')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        if ledger is not None:a['running_coarse_ledger']=dict(ledger.data)
        d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External physics guard')))
    try:
        import fcntl
        lock=d.path(s['budgets']['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch as tm
        torch=tm
        from geobc import elastic_pilot as old,elastic_pilot_data as olddata,elastic_coarse as coarse
        from geobc.load_linear_physics_inputs_v100 import prepare_inputs
        from geobc.load_linear_physics_pcg_v100 import ParallelElastic
        from geobc.sixload_coarse_v102 import Ledger,ThreeLaneCoarse,qualify,solve_group as solve_coarse
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        total=torch.cuda.get_device_properties(0).total_memory;assert 30500000000<total;torch.cuda.set_per_process_memory_fraction(30500000000/total,0);torch.cuda.reset_peak_memory_stats()
        for cad in s['selection']['reused_CADs']:
            row=s['cases'][cad]
            for load,item in row['reuse_converged_canonical_manifest_identities'].items():
                assert d.identity(item['path'],hashed=True)==item
                rec=d.read(item['path']);assert rec['numerical_state']=='converged' and rec['true_relative_residual']<=1e-8
                a['prediction_inventory'][cad+'/'+load]=item
                a['committed_finite_endpoints']+=1
            a['case_records'][cad]=dict(reused_existing_endpoints=True,new_K_PCG_input_reads=0,inherited_actual_basis_cost=row['inherited_actual_basis_cost'])
        for cad in s['selection']['new_solve_CADs']:
            guard();folder=o/cad;folder.mkdir(exist_ok=False);ledger=None;a['current_CAD']=cad;a['phase']='targetfree_source';persist();tick=(time.time(),time.monotonic())
            cs=dict(s['cases'][cad],selection=dict(CAD=cad,mixed_coefficients=s['selection']['mixed_coefficients']),old_accepted_physics_recipe=s['physics_recipe'])
            arrays,forces_np,record=prepare_inputs(np,cs,a,guard,tick[1]);guard()
            cache=folder/'geometry_and_rhs_input.npz'
            with cache.open('xb')as f:np.savez(f,**arrays,all_six_forces=forces_np)
            input_identity=d.identity(cache,hashed=True);inputs={k:torch.as_tensor(v,device='cuda')for k,v in arrays.items()};forces=torch.as_tensor(forces_np,device='cuda');del arrays,forces_np
            prior=elapsed(tick);assert prior<300;a['phase']='new_geometry_operator8guards';persist();start=time.monotonic()
            op,chol,diagonal,geometry=old._build_geometry(torch,inputs,record,start,prior)
            algebra=old._algebra_guards(torch,op,diagonal,inputs,record,start,prior);assert algebra['passed'] and algebra['K_applications']==8
            batch=ParallelElastic(torch,op,guard);gen=torch.Generator(device='cuda').manual_seed(2026);w=torch.zeros_like(forces[:3]);w[:,inputs['free_node_ids']]=torch.randn((3,len(inputs['free_node_ids']),3),dtype=torch.float64,device='cuda',generator=gen)
            actual=batch.apply(w);ref=torch.stack([op.apply(w[i])for i in range(3)]);errors=torch.linalg.vector_norm(actual-ref,dim=(1,2))/torch.linalg.vector_norm(ref,dim=(1,2));assert torch.isfinite(actual).all() and bool((errors<=1e-9).all())
            kqual=dict(passed=True,actual_allN=len(inputs['vertices_mm']),actual_lanes=3,relative_parity=errors.cpu().tolist(),old_extra_calls=3,batch_calls=1);del w,actual,ref,errors,diagonal
            saved={}
            for name,value in [('gradients',op.gradients),('weights',op.weights),('free_cholesky',chol)]:
                item=old._save_array(np,folder/'operator_cache'/(name+'.npy'),value);item.update(d.identity(item['path'],hashed=True));saved[name]=item
            source_wall=elapsed(tick);assert source_wall<=300
            a['case_records'][cad]=dict(expected_geometry=cs['expected_geometry'],input_identity=input_identity,source_operator_wall_seconds=source_wall,source_facts=a['physics_input_facts'],geometry=geometry,algebra_guards=algebra,parallel_K_qualification=kqual,operator_cache=saved,groups={});persist()
            def group(names,role,fn,precondition=None):
                guard();a['phase']=cad+'/'+role;persist();began=(time.time(),time.monotonic())
                def remaining():return 1800-a['combined_solve_seconds']-elapsed(began)
                force=forces[3:]if role=='coarse_mixed'else forces[:3]
                if precondition is None:inv,reported=fn(torch,np,batch,chol,inputs,force,names,folder/role,a,guard,persist,remaining)
                else:inv,reported=fn(torch,np,batch,chol,inputs,force,names,folder/role,a,guard,persist,remaining,precondition)
                wall=elapsed(began);assert wall<=300;a['combined_solve_seconds']+=wall
                data=dict(wall_seconds=wall,solver_reported_seconds=reported,initial_guess='independent_zero_all3');a['case_records'][cad]['groups'][role]=data;a['completed_groups'][cad+'/'+role]=data
                a['prediction_inventory'].update({cad+'/'+name:item for name,item in inv.items()});a['committed_finite_endpoints']+=3;persist()
            guard();a['phase']='coarse_construction';persist();ct=(time.time(),time.monotonic());ledger=Ledger(torch,guard,persist,folder,ct)
            oa=dict(gradients=op.gradients,weights=op.weights,free_cholesky=chol)
            P,PT,corners,cmap,fmap,mapping=coarse.make_mapping(torch,inputs,ledger);Ac,cd,gamma,assembly=coarse.assemble_coarse(torch,inputs,oa,cmap,fmap,ledger)
            scalar=coarse.AdditiveCoarse(torch,P,PT,Ac,cd,gamma,chol,ledger);ca=coarse.algebra_guards(torch,scalar,op,inputs,ledger)
            parallel=ThreeLaneCoarse(torch,scalar,guard);mq=qualify(torch,parallel,scalar,len(inputs['free_node_ids']));assert ca['passed'] and mq['passed']
            cc=folder/'coarse_cache';cc.mkdir(exist_ok=False);saved_coarse={name:coarse.save_sparse(np,old,cc,name,value)for name,value in [('P',P),('PT',PT),('Ac',Ac)]}
            saved_coarse.update(diagonal=old._save_array(np,cc/'diagonal.npy',cd),corners=old._save_array(np,cc/'corners.npy',corners))
            for item in saved_coarse.values():
                if 'rowptr'in item:
                    for part in ('rowptr','columns','values'):item[part].update(d.identity(item[part]['path'],hashed=True))
                else:item.update(d.identity(item['path'],hashed=True))
            d.write(cc/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],mapping=mapping,assembly=assembly,algebra=ca,parallel_M_qualification=mq,arrays=saved_coarse))
            ledger.check();coarse_wall=elapsed(ct);assert coarse_wall<=120;ledger.setup_active=False;a['case_records'][cad].update(coarse_setup_seconds=coarse_wall,coarse_mapping=mapping,coarse_assembly=assembly,coarse_algebra_guards=ca,parallel_M_qualification=mq);persist()
            group(['ver','hor','tor'],'coarse_canonical',solve_coarse,parallel.apply)
            case=a['case_records'][cad]
            case.update(new_source_coarse_canonical_seconds=source_wall+coarse_wall+case['groups']['coarse_canonical']['wall_seconds'],operator_calls=dict(scalar=op.calls,batch=batch.calls,coarse_parallel=parallel.calls),coarse_ledger=dict(ledger.data),reused_existing_endpoints=False)
            d.write(folder/'case_numeric_terminal.json',case);persist()
            del inputs,forces,op,chol,batch,ledger,oa,P,PT,Ac,cd,corners,cmap,fmap,scalar,parallel;ledger=None;gc.collect();torch.cuda.empty_cache()
        assert a['combined_solve_seconds']<=1800 and a['committed_finite_endpoints']==60
        assert a['H5_input_reads']==a['GRID_map_reads']==a['deck_parses']==a['deck_header_reads']==17
        a['phase']='complete60_prediction_barrier_GTclosed';persist();rt=(time.time(),time.monotonic());fields={};records={}
        def read_array(item,owned=True):
            guard();expected={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')};expected['path']=d.relative(d.path(item['path']));assert d.identity(item['path'],hashed=True)==expected
            value=np.load(d.path(item['path']),allow_pickle=False)
            if owned:a['owned_array_reads']+=1
            dtype={'torch.float64':'float64','torch.float32':'float32','torch.int64':'int64'}.get(item['dtype'],item['dtype'])
            assert list(value.shape)==item['shape'] and str(value.dtype)==dtype and np.isfinite(value).all();return value
        expected={cad+'/'+name for cad in s['selection']['CADs']for name in ('ver','hor','tor')};assert set(a['prediction_inventory'])==expected
        for key,item in a['prediction_inventory'].items():
            assert d.identity(item['path'],hashed=True)==item;rec=d.read(item['path']);cad,load=key.split('/')
            assert rec['status']=='completed' and rec['initial_guess']=='independent_zero' and rec['key']==load and rec['all_original_nodes']==s['cases'][cad]['expected_geometry']['nodes']
            value=read_array(rec['physical_U']);assert value.shape==(rec['all_original_nodes'],3) and value.dtype==np.float64
            fields[key]=value;records[key]=rec
        d.write(o/'physics_prediction_barrier.json',dict(status='completed',all60_finite_immutable_original_nodes=True,GT_reads_before_barrier=0,inventory=a['prediction_inventory'],new_zero_solved=51,reused_immutable=9));a['prediction_barrier_complete']=True;persist()
        metrics={}
        for key,value in fields.items():
            guard();a['GT_attempts']+=1;gt=read_array(truths['records'][key]['physical'],owned=False);a['GT_reads']+=1
            assert gt.shape==(len(value),4) and gt.dtype==np.float32
            ref=gt[:,:3].astype(np.float64);delta=value-ref;norm=float(np.linalg.norm(ref));error=float(np.linalg.norm(delta));absolute=float(np.linalg.norm(delta,axis=1).max());relative=error/norm if norm>0 else None
            metrics[key]=dict(U_relative_L2=relative,U_RMS_mm=float(np.square(delta).sum()/len(value))**.5,U_max_mm=absolute,original_nodes=len(value),zero_reference=norm==0,true_relative_residual=records[key]['true_relative_residual'],converged=records[key]['numerical_state']=='converged',U_gate_passed=relative<=.01 if relative is not None else absolute<=1e-12)
            del gt,ref,delta;persist()
        allconv=all(v['numerical_state']=='converged' and v['true_relative_residual']<=1e-8 for v in records.values());passU=all(v['U_gate_passed']for v in metrics.values())
        assert a['GT_attempts']==a['GT_reads']==60 and a['owned_array_reads']==60
        a.update(endpoint_records=records,canonical_original_GT_U=metrics,precision_gate_passed=allconv and passU,coverage_gate_passed=allconv and passU,physics_reference_compatible=allconv and passU,protocol_qualified=True,readout_seconds=elapsed(rt));assert a['readout_seconds']<=180
        a.update(status='completed',phase='terminal',scientific_outcome='complete20CAD_U_coverage_PASS'if a['coverage_gate_passed']else'complete20CAD_fixed_U_gate_negative')
    except Exception as error:
        from geobc.elastic_pilot import ScientificNegative
        from geobc.elastic_pilot_data import UnsupportedInput
        if isinstance(error,(ScientificNegative,UnsupportedInput)):a.update(status='completed',phase='terminal_scientific_negative',reason=str(error),scientific_outcome='unsupported_input_or_operator_or_solver',protocol_qualified=False)
        else:a.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error));raise
    finally:
        a.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        if ledger is not None:a['running_coarse_ledger']=dict(ledger.data)
        d.write(o/'summary.json',a);d.write(o/'status.json',a)
        if lock is not None:lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');p.add_argument('--plan');args=p.parse_args()
    if args.start:
        if not args.plan:p.error('--start requires fixed --plan')
        run(args.plan)
    else:print('{"status":"source_only","GT_arrays_GPU":0}')
