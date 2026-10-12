"""Two new CADs, exact sameK paired block/coarse with global prediction barrier."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,json,os,signal,gc
from pathlib import Path

def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.physics_crossCAD_contract_v103 import approved_contract
    p,r,truths=approved_contract(plan_path);s=p['scientific'];o=d.path(p['output']);outer=json.loads(os.environ['GEOBC_PHYSICS_CROSSCAD_V103_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest'] and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=p['scientific_digest'],boot_id=outer['boot_id']);d.write(o/'execution.json',execution)
    a=dict(status='running',active=True,closed=False,phase='source',execution=execution,source_body_hash_reads=0,H5_input_attempts=0,H5_input_reads=0,H5_members_read=[],GRID_map_reads=0,deck_parses=0,deck_header_reads=0,owned_array_reads=0,GT_attempts=0,GT_reads=0,raw_CSV_H5_nodal_target_reads=0,neural_PT_reads=0,neural_inference=0,training_updates=0,scaler_fits=0,FINAL_reads=0,prediction_inventory={},fixed_endpoint_denominator=18,committed_finite_endpoints=0,combined_solve_seconds=0.,case_records={},completed_groups={},peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0,protocol_qualified=False,physics_reference_compatible=False,cost_gate_passed=False,final_solver_selected=False)
    torch=None;lock=None;ledger=None
    def elapsed(t):return max(time.time()-t[0],time.monotonic()-t[1])
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Whole1800 earlier absolute deadline')
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
        from geobc.load_linear_physics_pcg_v100 import ParallelElastic,solve_group as solve_block
        from geobc.sixload_coarse_v102 import Ledger,ThreeLaneCoarse,qualify,solve_group as solve_coarse
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        total=torch.cuda.get_device_properties(0).total_memory;assert 30500000000<total;torch.cuda.set_per_process_memory_fraction(30500000000/total,0);torch.cuda.reset_peak_memory_stats()
        for cad in s['selection']['CADs']:
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
                def remaining():return 1200-a['combined_solve_seconds']-elapsed(began)
                force=forces[3:]if role=='coarse_mixed'else forces[:3]
                if precondition is None:inv,reported=fn(torch,np,batch,chol,inputs,force,names,folder/role,a,guard,persist,remaining)
                else:inv,reported=fn(torch,np,batch,chol,inputs,force,names,folder/role,a,guard,persist,remaining,precondition)
                wall=elapsed(began);assert wall<=300;a['combined_solve_seconds']+=wall
                data=dict(wall_seconds=wall,solver_reported_seconds=reported,initial_guess='independent_zero_all3');a['case_records'][cad]['groups'][role]=data;a['completed_groups'][cad+'/'+role]=data
                a['prediction_inventory'].update({cad+'/'+role+'/'+name:item for name,item in inv.items()});a['committed_finite_endpoints']+=3;persist()
            group(['ver','hor','tor'],'block_canonical',solve_block)
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
            group(['mix0','mix1','mix2'],'coarse_mixed',solve_coarse,parallel.apply)
            case=a['case_records'][cad];baseline=case['groups']['block_canonical']['wall_seconds'];newcost=coarse_wall+case['groups']['coarse_canonical']['wall_seconds']
            case.update(paired_primary_baseline_seconds=baseline,paired_primary_coarse_seconds=newcost,paired_reduction_fraction=1-newcost/baseline,paired_cost_gate_passed=newcost<=.75*baseline,operator_calls=dict(scalar=op.calls,batch=batch.calls,coarse_parallel=parallel.calls),coarse_ledger=dict(ledger.data))
            d.write(folder/'case_numeric_terminal.json',case);persist()
            del inputs,forces,op,chol,batch,ledger,oa,P,PT,Ac,cd,corners,cmap,fmap,scalar,parallel;ledger=None;gc.collect();torch.cuda.empty_cache()
        assert a['combined_solve_seconds']<=1200 and a['committed_finite_endpoints']==18
        a['phase']='all18_prediction_barrier_GTclosed';persist();rt=(time.time(),time.monotonic());fields={};records={}
        def read_array(item):
            guard();expected={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')};expected['path']=d.relative(d.path(item['path']));assert d.identity(item['path'],hashed=True)==expected
            value=np.load(d.path(item['path']),allow_pickle=False);a['owned_array_reads']+=1;dtype={'torch.float64':'float64','torch.float32':'float32','torch.int64':'int64'}.get(item['dtype'],item['dtype']);assert list(value.shape)==item['shape'] and str(value.dtype)==dtype and np.isfinite(value).all();return value
        expected={cad+'/'+role+'/'+name for cad in s['selection']['CADs']for role,names in [('block_canonical',['ver','hor','tor']),('coarse_canonical',['ver','hor','tor']),('coarse_mixed',['mix0','mix1','mix2'])]for name in names};assert set(a['prediction_inventory'])==expected
        for key,item in a['prediction_inventory'].items():
            assert d.identity(item['path'],hashed=True)==item;rec=d.read(item['path']);cad=key.split('/')[0];n=s['cases'][cad]['expected_geometry']['nodes'];assert rec['status']=='completed' and rec['initial_guess']=='independent_zero' and rec['key']==key.split('/')[-1] and rec['all_original_nodes']==n
            value=read_array(rec['physical_U']);assert value.shape==(n,3) and value.dtype==np.float64;fields[key]=value;records[key]=rec
        d.write(o/'physics_prediction_barrier.json',dict(status='completed',all18_finite_immutable_original_nodes=True,GT_reads_before_barrier=0,inventory=a['prediction_inventory']));a['prediction_barrier_complete']=True;persist()
        def metric(pred,ref):
            norm=float(np.square(ref).sum());sse=float(np.square(pred-ref).sum());return dict(U_relative_L2=(sse/norm)**.5 if norm>0 else None,U_RMS_mm=(sse/len(ref))**.5,U_max_mm=float(np.linalg.norm(pred-ref,axis=1).max()),allN=len(ref),zero_reference=norm==0)
        numerical={};canonical={}
        for cad in s['selection']['CADs']:
            basis=[fields[cad+'/block_canonical/'+load]for load in ('ver','hor','tor')]
            for i,load in enumerate(('ver','hor','tor')):numerical[cad+'/canonical/'+load]=metric(fields[cad+'/coarse_canonical/'+load],basis[i])
            for i,c in enumerate(s['selection']['mixed_coefficients']):numerical[cad+'/mixed/mix'+str(i)]=metric(fields[cad+'/coarse_mixed/mix'+str(i)],sum(c[j]*basis[j]for j in range(3)))
            for load in ('ver','hor','tor'):
                key=cad+'/'+load;guard();a['GT_attempts']+=1;gt=read_array(truths['records'][key]['physical']);a['GT_reads']+=1;assert gt.shape==(len(basis[0]),4) and gt.dtype==np.float32
                canonical[key]={role:metric(fields[cad+'/'+role+'/'+load],gt[:,:3].astype(np.float64))for role in ('block_canonical','coarse_canonical')};del gt;persist()
        allconv=all(v['numerical_state']=='converged' and v['true_relative_residual']<=1e-8 for v in records.values())
        same=all(v['U_relative_L2']<=1e-5 if v['U_relative_L2']is not None else v['U_RMS_mm']<=1e-12 for v in numerical.values())
        gtpass=all(v['coarse_canonical']['U_relative_L2']is not None and v['coarse_canonical']['U_relative_L2']<=.01 for v in canonical.values())
        a.update(endpoint_records=records,same_solution_metrics=numerical,canonical_original_GT_U=canonical,physics_reference_compatible=allconv and same and gtpass,precision_gate_passed=allconv and same and gtpass,cost_gate_passed=all(a['case_records'][cad]['paired_cost_gate_passed']for cad in s['selection']['CADs']),protocol_qualified=True,readout_seconds=elapsed(rt));assert a['GT_attempts']==a['GT_reads']==6 and a['readout_seconds']<=120
        a.update(status='completed',phase='terminal',scientific_outcome='complete_twoCAD_precision_and_cost'if a['physics_reference_compatible'] and a['cost_gate_passed']else'complete_twoCAD_fixed_gate_negative')
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
