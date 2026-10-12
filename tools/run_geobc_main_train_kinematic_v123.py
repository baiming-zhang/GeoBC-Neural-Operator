"""Main TRAIN100 physical U/fullgrad predictors:296 new zero solves,4 accepted VER reuse."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,json,os,signal,gc,hashlib
from pathlib import Path

def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.main_train_kinematic_contract_v123 import approved_contract
    p,r,catalog=approved_contract(plan_path);s=p['scientific'];o=d.path(p['output']);outer=json.loads(os.environ['GEOBC_MAIN_TRAIN_KINEMATIC_V123_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest'] and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=p['scientific_digest'],boot_id=outer['boot_id']);d.write(o/'execution.json',execution)
    a=dict(status='running',active=True,closed=False,phase='cache',execution=execution,source_cache_reads=0,owned_array_reads=0,raw_H5_deck_GRID_CSV_reads=0,GT_reads=0,DEV_FINAL_reads=0,neural_PT_reads=0,training_updates=0,scaler_fits=0,fixed_CAD_denominator=100,fixed_load_denominator=300,new_zero_endpoint_count=0,reused_VER_endpoint_count=0,committed_feature_CADs=0,prediction_inventory={},feature_inventory={},case_records={},completed_groups={},combined_solve_seconds=0.,feature_seconds=0.,peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0,complete_input_gate_passed=False,neural_training_released=False)
    torch=None;lock=None;ledger=None
    def elapsed(tick):return max(time.time()-tick[0],time.monotonic()-tick[1])
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Whole7200 earlier absolute deadline')
        if torch is not None and torch.cuda.is_initialized():
            a['peak_allocated_bytes']=max(a['peak_allocated_bytes'],torch.cuda.max_memory_allocated());a['peak_reserved_bytes']=max(a['peak_reserved_bytes'],torch.cuda.max_memory_reserved());free,total=torch.cuda.mem_get_info();a['peak_total_device_used_bytes_sampled']=max(a['peak_total_device_used_bytes_sampled'],total-free)
            if max(a[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))>30500000000:raise MemoryError('30.5GB cap')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        if ledger is not None:a['running_coarse_ledger']=dict(ledger.data)
        d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External mainTRAIN guard')))
    try:
        import fcntl,numpy as np,torch as tm
        torch=tm;lock=d.path(s['budgets']['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        from geobc import elastic_pilot as old,elastic_coarse as coarse
        from geobc.main_train_physics_lanes_v123 import CanonicalLanesElastic,CanonicalLanesCoarse,solve_group
        from geobc.main_train_kinematic_io_v123 import load_source,load_array,load_sparse,qualify_derivative
        from geobc.kinematic_query_inputs_v121 import recover_kinematic_inputs
        from geobc.sixload_coarse_v102 import Ledger
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        total=torch.cuda.get_device_properties(0).total_memory;assert total>30500000000;torch.cuda.set_per_process_memory_fraction(30500000000/total,0);torch.cuda.reset_peak_memory_stats()
        for cad in s['selection']['CADs']:
            guard();a['phase']=cad+'/input_and_operator';persist();folder=o/cad;folder.mkdir(exist_ok=False);tick=(time.time(),time.monotonic());ledger=None
            case=catalog['case_records'][cad];inputs,forces=load_source(np,torch,case,a,guard);record=dict(recipe=s['physics_recipe'],expected_geometry=case['expected_geometry']);legacy=s['legacy'].get(cad)
            if legacy:
                arrays={k:load_array(np,torch,v,a,guard) for k,v in legacy['operator_arrays'].items()}
                lam,mu=old._validate_inputs(torch,inputs,record);op=old.MatrixFreeElastic(torch,inputs['cells10'],arrays['gradients'],arrays['weights'],len(inputs['vertices_mm']),lam,mu);chol=arrays['free_cholesky']
                geometry=legacy['geometry'];algebra=legacy['algebra_guards'];names=['hor','tor'];selected=forces[1:]
                inherited=d.read(legacy['VER_summary_identity']['path']);assert inherited['numerical_state']=='converged' and inherited['true_relative_residual']<=1e-8
                a['prediction_inventory'][cad+'/ver']=dict(kind='legacy_VER',summary_identity=legacy['VER_summary_identity'],physical_U=legacy['VER_prediction'],numerical_history=legacy['numerical_history']);a['reused_VER_endpoint_count']+=1
            else:
                prior=elapsed(tick);start=time.monotonic();op,chol,diagonal,geometry=old._build_geometry(torch,inputs,record,start,prior);algebra=old._algebra_guards(torch,op,diagonal,inputs,record,start,prior);assert algebra['passed'] and algebra['K_applications']==8
                arrays=dict(gradients=op.gradients,weights=op.weights,free_cholesky=chol);names=['ver','hor','tor'];selected=forces
                saved={}
                for k,v in arrays.items():item=old._save_array(np,folder/'operator_cache'/(k+'.npy'),v);item.update(d.identity(item['path'],hashed=True));saved[k]=item
                d.write(folder/'operator_manifest.json',dict(status='completed',arrays=saved,geometry=geometry,algebra_guards=algebra,SCI=p['scientific_digest']));del diagonal
            parallel=CanonicalLanesElastic(torch,op,guard);gen=torch.Generator(device='cuda').manual_seed(2026);w=torch.zeros_like(selected);w[:,inputs['free_node_ids']]=torch.randn((len(names),len(inputs['free_node_ids']),3),dtype=torch.float64,device='cuda',generator=gen)
            actual=parallel.apply(w);ref=torch.stack([op.apply(v) for v in w]);errors=torch.linalg.vector_norm(actual-ref,dim=(1,2))/torch.linalg.vector_norm(ref,dim=(1,2));assert bool((errors<=1e-9).all() and torch.isfinite(actual).all())
            kq=dict(passed=True,actual_allN=len(inputs['vertices_mm']),actual_lanes=len(names),relative=errors.cpu().tolist());del actual,ref,errors
            source_wall=elapsed(tick);assert source_wall<=300;ct=(time.time(),time.monotonic());ledger=Ledger(torch,guard,persist,folder,ct)
            if legacy and legacy.get('coarse_manifest_identity'):
                cm=d.read(legacy['coarse_manifest_identity']['path']);c=cm['arrays'];P=load_sparse(np,torch,c['P'],a,guard);PT=load_sparse(np,torch,c['PT'],a,guard);Ac=load_sparse(np,torch,c['Ac'],a,guard);cd=load_array(np,torch,c['diagonal'],a,guard);gamma=cm['assembly']['symmetric_scaled_Gershgorin_Gamma']
                assert P.shape[0]==len(inputs['free_node_ids']) and PT.shape==(P.shape[1],P.shape[0]) and Ac.shape==(3*P.shape[1],3*P.shape[1]) and cd.shape==(Ac.shape[0],) and bool((cd>0).all())
                ca=cm['algebra_guards'];mapping=cm['mapping'];assembly=cm['assembly'];coarse_reused=True
                scalar=coarse.AdditiveCoarse(torch,P,PT,Ac,cd,gamma,chol,ledger)
            else:
                P,PT,corners,cmap,fmap,mapping=coarse.make_mapping(torch,inputs,ledger);Ac,cd,gamma,assembly=coarse.assemble_coarse(torch,inputs,arrays,cmap,fmap,ledger)
                scalar=coarse.AdditiveCoarse(torch,P,PT,Ac,cd,gamma,chol,ledger);ca=coarse.algebra_guards(torch,scalar,op,inputs,ledger);assert ca['passed'];coarse_reused=False
                cc=folder/'coarse_cache';cc.mkdir();saved={k:coarse.save_sparse(np,old,cc,k,v) for k,v in [('P',P),('PT',PT),('Ac',Ac)]};saved['diagonal']=old._save_array(np,cc/'diagonal.npy',cd)
                for v in saved.values():
                    for item in ([v[x] for x in ('rowptr','columns','values')] if 'rowptr' in v else [v]):item.update(d.identity(item['path'],hashed=True))
                d.write(cc/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],mapping=mapping,assembly=assembly,algebra_guards=ca,arrays=saved));del corners,cmap,fmap
            precondition=CanonicalLanesCoarse(torch,scalar,guard);free_w=w[:,inputs['free_node_ids']];ma=precondition.apply(free_w);mr=torch.stack([scalar.apply(v) for v in free_w]);me=torch.linalg.vector_norm(ma-mr,dim=(1,2))/torch.linalg.vector_norm(mr,dim=(1,2));assert bool((me<=1e-9).all() and torch.isfinite(ma).all())
            mq=dict(passed=True,actual_freeN=len(inputs['free_node_ids']),actual_lanes=len(names),relative=me.cpu().tolist(),fixed_inner_steps=32);del w,free_w,ma,mr,me
            ledger.check();cw=elapsed(ct);assert cw<=120;ledger.setup_active=False
            a['case_records'][cad]=dict(source_cache_identity=case['input_identity'],actual_allN=len(inputs['vertices_mm']),actual_cells=len(inputs['cells10']),geometry=geometry,algebra_guards=algebra,parallel_K_qualification=kq,parallel_M_qualification=mq,coarse_algebra=ca,coarse_reused=coarse_reused,cache_operator_seconds=source_wall,coarse_seconds=cw,legacy_VER_reused=bool(legacy));persist()
            a['phase']=cad+'/canonical_zero_PCG';persist();gtick=(time.time(),time.monotonic())
            def remaining():return 5000-a['combined_solve_seconds']-elapsed(gtick)
            inv,reported=solve_group(torch,np,parallel,chol,inputs,selected,names,folder/'canonical',a,guard,persist,remaining,precondition.apply);wall=elapsed(gtick);assert wall<=300;a['combined_solve_seconds']+=wall
            converged=[]
            for load,item in inv.items():
                rec=d.read(item['path']);converged.append(rec['numerical_state']=='converged' and rec['true_relative_residual']<=1e-8)
                a['prediction_inventory'][cad+'/'+load]=dict(kind='new_zero',manifest_identity=item,physical_U=rec['physical_U']);a['new_zero_endpoint_count']+=1
            a['completed_groups'][cad]=dict(wall_seconds=wall,reported_seconds=reported,loads=names,initial_guess='independent_zero');persist()
            if not all(converged):raise old.ScientificNegative('Canonical finite budget endpoint(s) not converged; no replay')
            a['phase']=cad+'/physical_U_gradient_inputs';persist();ft=(time.time(),time.monotonic())
            U=torch.stack([load_array(np,torch,a['prediction_inventory'][cad+'/'+load]['physical_U'],a,guard) for load in ('ver','hor','tor')])
            dq=qualify_derivative(np,torch,inputs,U,guard);before=hashlib.sha256(U.cpu().numpy().tobytes()).hexdigest()
            features,fq=recover_kinematic_inputs(torch,inputs['vertices_mm'],inputs['cells10'],U,op.weights.sum(1),guard)
            assert hashlib.sha256(U.cpu().numpy().tobytes()).hexdigest()==before and torch.equal(features[...,:3],U)
            item=old._save_array(np,folder/'physical_U_grad12.npy',features);item.update(d.identity(item['path'],hashed=True));fw=elapsed(ft);a['feature_seconds']+=fw;assert a['feature_seconds']<=1200
            a['feature_inventory'][cad]=dict(array=item,case_keys=[cad+'/'+load for load in ('ver','hor','tor')],component_order=['U_x_mm','U_y_mm','U_z_mm']+['1000*dU'+u+'/d'+x for u in 'xyz' for x in 'xyz'],qualification=fq,independent_derivative_qualification=dq,actual_feature_seconds=fw)
            a['case_records'][cad].update(actual_solve_seconds=wall,feature_seconds=fw,scalar_K_calls=op.calls,parallel_K_calls=parallel.calls,coarse_parallel_calls=precondition.calls);a['committed_feature_CADs']+=1
            d.write(folder/'case_numeric_terminal.json',a['case_records'][cad]);persist()
            del inputs,forces,selected,U,features,arrays,op,chol,parallel,P,PT,Ac,cd,scalar,precondition;ledger=None;gc.collect();torch.cuda.empty_cache()
        assert a['new_zero_endpoint_count']==296 and a['reused_VER_endpoint_count']==4 and a['committed_feature_CADs']==100 and a['source_cache_reads']==100
        assert set(a['prediction_inventory'])=={cad+'/'+load for cad in s['selection']['CADs'] for load in ('ver','hor','tor')} and set(a['feature_inventory'])==set(s['selection']['CADs'])
        for item in a['feature_inventory'].values():d.stat_matches(item['array'])
        d.write(o/'complete_input_barrier.json',dict(status='completed',all300_physics_and100_fullN_gradient_inputs=True,prediction_inventory=a['prediction_inventory'],feature_inventory=a['feature_inventory'],GT_reads=0,scaler_fits=0,neural_training_released=False))
        a.update(status='completed',phase='terminal',complete_input_gate_passed=True,main300_kinematic_inputs_ready=True)
    except Exception as error:
        from geobc.elastic_pilot import ScientificNegative
        if isinstance(error,ScientificNegative) or isinstance(error,ValueError) and str(error).startswith(('Original curved cells','Original nodes missing','Actual P2 physical derivative')):
            a.update(status='completed',phase='terminal_scientific_input_negative',complete_input_gate_passed=False,reason=str(error),main300_kinematic_inputs_ready=False)
        else:a.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error));raise
    finally:
        a.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        if ledger is not None:a['running_coarse_ledger']=dict(ledger.data)
        d.write(o/'summary.json',a);d.write(o/'status.json',a)
        if lock is not None:lock.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');p.add_argument('--plan');args=p.parse_args()
    if args.start:
        if not args.plan:p.error('--start requires frozen --plan')
        run(args.plan)
    else:print('{"status":"SOURCE_only","GPU_arrays_GT":0}')
