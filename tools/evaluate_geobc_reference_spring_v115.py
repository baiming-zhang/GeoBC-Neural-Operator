"""Independent zero PCG for a declared physical reference spring, GT-free."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,json,os,signal,hashlib,gc
from pathlib import Path

def run(path):
    from geobc import available_cad_pilot_data as d
    from geobc.reference_spring_contract_v115 import approved_contract
    p,r=approved_contract(path);s=p['scientific'];o=d.path(p['output']);outer=json.loads(os.environ['GEOBC_REFERENCE_SPRING_V115_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest']and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=p['scientific_digest'],boot_id=outer['boot_id']);d.write(o/'execution.json',execution)
    a=dict(status='running',phase='cache',active=True,closed=False,execution=execution,input_cache_reads=0,basis_array_reads=0,operator_array_reads=0,coarse_array_reads=0,
        owned_array_reads=0,GT_reads=0,raw_H5_deck_GRID_reads=0,geometry_K_coarse_rebuilds=0,inherited_eight_fine_K_guards_repeated=0,
        neural_PT_reads=0,training_updates=0,scaler_fits=0,FINAL_reads=0,case_records={},prediction_inventory={},completed_groups={},
        committed_finite_endpoints=0,fixed_endpoint_denominator=6,combined_solve_seconds=0.,prediction_barrier_complete=False,
        spring_BC_gate_passed=False,new_original_support_GT_accuracy=False,signed_S_solved=False,final_solver_selected=False,
        peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0)
    torch=None;lock=None
    def elapsed(t):return max(time.time()-t[0],time.monotonic()-t[1])
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Whole1800 earlier absolute deadline')
        if torch is not None and torch.cuda.is_initialized():
            a['peak_allocated_bytes']=max(a['peak_allocated_bytes'],torch.cuda.max_memory_allocated());a['peak_reserved_bytes']=max(a['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
            free,total=torch.cuda.mem_get_info();a['peak_total_device_used_bytes_sampled']=max(a['peak_total_device_used_bytes_sampled'],total-free)
            if max(a[k]for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))>30500000000:raise MemoryError('30.5GB cap')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External springBC guard')))
    try:
        import fcntl
        lock=d.path(s['budgets']['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch as tm
        torch=tm
        from geobc import elastic_pilot as old,elastic_coarse as coarse
        from geobc.load_linear_physics_pcg_v100 import ParallelElastic
        from geobc.sixload_coarse_v102 import Ledger,ThreeLaneCoarse,solve_group
        from geobc.reference_motion_response_v113 import ReferenceMotionResponse
        from geobc.reference_spring_response_v115 import ReferenceSpringResponse
        from geobc.reference_spring_operator_v115 import ScalarReferenceSpring,ThreeLaneReferenceSpring,qualify
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        total=torch.cuda.get_device_properties(0).total_memory;assert total>30500000000
        torch.cuda.set_per_process_memory_fraction(30500000000/total,0);torch.cuda.reset_peak_memory_stats()
        def read_array(item,count):
            guard();expected={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')};expected['path']=d.relative(d.path(item['path']))
            assert d.identity(item['path'],hashed=True)==expected
            value=np.load(d.path(item['path']),allow_pickle=False);a[count]+=1
            dtype={'torch.float64':'float64','torch.float32':'float32','torch.int64':'int64'}.get(item['dtype'],item['dtype'])
            assert list(value.shape)==item['shape']and str(value.dtype)==dtype and np.isfinite(value).all();return value
        def tensor(v):return torch.as_tensor(v,device='cuda')
        def checksum(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        def metric(v,ref,tol):
            norm=float(torch.linalg.vector_norm(ref));error=float(torch.linalg.vector_norm(v-ref));absolute=float((v-ref).abs().max());relative=error/norm if norm>0 else None
            return dict(relative=relative,maximum_absolute=absolute,passed=relative<=tol if relative is not None else absolute<=1e-12)
        def load_sparse(item):
            rr,cc,vv=[tensor(read_array(item[k],'coarse_array_reads'))for k in ('rowptr','columns','values')];nr,nc=item['shape']
            assert rr.dtype==cc.dtype==torch.int64 and vv.dtype==torch.float64 and rr.shape==(nr+1,)and cc.shape==vv.shape
            assert int(rr[0])==0 and int(rr[-1])==len(vv)and bool((rr[1:]>=rr[:-1]).all())and bool(((cc>=0)&(cc<nc)).all())
            return torch.sparse_csr_tensor(rr,cc,vv,size=(nr,nc),device='cuda')
        coeff=torch.tensor(s['selection']['mixed_coefficients'],dtype=torch.float64,device='cuda');resident={}
        for cad in s['selection']['CADs']:
            ct=(time.time(),time.monotonic());row=s['cases'][cad];folder=o/cad;folder.mkdir(exist_ok=False);a.update(current_CAD=cad,phase='strict_cache_and_spring');persist()
            identity=row['input_cache_identity'];assert d.identity(identity['path'],hashed=True)==identity
            with np.load(d.path(identity['path']),allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files}
            a['input_cache_reads']+=1;assert all(np.isfinite(v).all()for v in arrays.values());arrays.pop('all_six_forces')
            n,nc=row['expected_geometry']['nodes'],row['expected_geometry']['cells'];assert arrays['vertices_mm'].shape==(n,3)and arrays['cells10'].shape==(nc,10)
            assert arrays['vertices_mm'].dtype==np.float64 and arrays['cells10'].dtype==np.int64
            inputs={k:tensor(v)for k,v in arrays.items()};del arrays
            H,ids,L=inputs['rbe3_Hbar'],inputs['rbe3_member_node_ids'],float(inputs['rbe3_length_mm']);free=inputs['free_node_ids']
            assert inputs['derived_node_ids'].numel()==0 and bool(torch.isin(ids,free).all())and L==row['source_transfer_facts']['length_mm']
            B=torch.tensor(row['reference_load6_columns'],dtype=torch.float64,device='cuda').T.contiguous()
            basis=torch.stack([tensor(read_array(row['basis_records'][name]['physical_U'],'basis_array_reads'))for name in s['selection']['basis_order']])
            assert basis.shape==(6,n,3)and basis.dtype==torch.float64
            reference=ReferenceMotionResponse(torch,H,ids,L,basis,B,guard);spring=ReferenceSpringResponse(torch,reference,guard)
            op_arrays={k:tensor(read_array(item,'operator_array_reads'))for k,item in row['operator_cache'].items()}
            assert op_arrays['gradients'].shape==(nc,4,10,3)and op_arrays['weights'].shape==(nc,4)and op_arrays['free_cholesky'].shape==(len(free),3,3)
            assert all(v.dtype==torch.float64 for v in op_arrays.values())
            lam,mu=map(float,inputs['lame'].tolist());fine=old.MatrixFreeElastic(torch,inputs['cells10'],op_arrays['gradients'],op_arrays['weights'],n,lam,mu)
            basebatch=ParallelElastic(torch,fine,guard);scalar=ScalarReferenceSpring(torch,fine,H,ids,spring.spring_diagonal,guard)
            batch=ThreeLaneReferenceSpring(torch,basebatch,scalar,guard)
            before=dict(basis=checksum(basis),rawC=checksum(reference.C),**{k:checksum(v)for k,v in op_arrays.items()})
            coarse_meta=d.read(row['coarse_cache_identity']['path']);cm=coarse_meta['arrays'];P,PT,Ac=[load_sparse(cm[name])for name in ('P','PT','Ac')]
            diagonal=tensor(read_array(cm['diagonal'],'coarse_array_reads'));chol=op_arrays['free_cholesky']
            assert diagonal.dtype==torch.float64 and bool((diagonal>0).all())and P.shape[0]==PT.shape[1]==len(free)
            ledger=Ledger(torch,guard,persist,folder,ct);ledger.setup_active=False
            scalarM=coarse.AdditiveCoarse(torch,P,PT,Ac,diagonal,coarse_meta['assembly']['symmetric_scaled_Gershgorin_Gamma'],chol,ledger)
            parallelM=ThreeLaneCoarse(torch,scalarM,guard)
            qualification=qualify(torch,scalar,batch,free)
            forces6=coeff@B.T;forces=torch.zeros((3,n,3),dtype=torch.float64,device='cuda')
            member_forces=((forces6/reference.scales)@H).reshape(3,-1,3);forces.index_add_(1,ids,member_forces)
            torch.cuda.synchronize();cache_seconds=elapsed(ct);assert cache_seconds<=120
            rec=dict(actual_allN=n,actual_cells=nc,source_REF_GRID=row['source_transfer_facts']['reference_GRID_ID'],fit_length_mm=L,
                source_REFc='123456',reference_metadata=reference.metadata,spring_metadata=spring.metadata,
                normalized_spring_diagonal_N_per_mm=spring.spring_diagonal.cpu().tolist(),modified_K_qualification=qualification,
                cache_constructor_qualification_seconds=cache_seconds,inherited_source_to_six_basis_seconds=row['inherited_source_to_six_basis_seconds'],
                fresh_engineering_fullcold_seconds=None)
            a['case_records'][cad]=rec;a['phase']=cad+'/independent_modified_zero_PCG';persist();st=(time.time(),time.monotonic())
            def remaining():return 600-a['combined_solve_seconds']-elapsed(st)
            inv,reported=solve_group(torch,np,batch,chol,inputs,forces,['spring_mix0','spring_mix1','spring_mix2'],folder/'modified_PCG',a,guard,persist,remaining,parallelM.apply)
            wall=elapsed(st);assert wall<=300;a['combined_solve_seconds']+=wall;rec['independent_group_seconds']=wall;rec['reported_group_seconds']=reported
            a['completed_groups'][cad]=dict(complete=True,wall_seconds=wall,all_initial_guesses='independent_zero',immutable_endpoint_count=3)
            a['prediction_inventory'].update({cad+'/'+name:item for name,item in inv.items()});a['committed_finite_endpoints']+=3
            resident[cad]=dict(inputs=inputs,H=H,ids=ids,B=B,basis=basis,reference=reference,spring=spring,fine=fine,scalar=scalar,op_arrays=op_arrays,
                before=before,forces=forces,forces6=forces6)
            persist()
        assert a['committed_finite_endpoints']==6 and a['input_cache_reads']==2 and a['basis_array_reads']==12 and a['operator_array_reads']==6 and a['coarse_array_reads']==20
        a['phase']='all6_independent_equilibrium_field_barrier';persist();rt=(time.time(),time.monotonic());fields={};records={}
        expected={cad+'/spring_mix'+str(j)for cad in s['selection']['CADs']for j in range(3)};assert set(a['prediction_inventory'])==expected
        for key,item in a['prediction_inventory'].items():
            assert d.identity(item['path'],hashed=True)==item;record=d.read(item['path']);cad,name=key.split('/')
            assert record['initial_guess']=='independent_zero'and record['key']==name and record['all_original_nodes']==s['cases'][cad]['expected_geometry']['nodes']
            v=tensor(read_array(record['physical_U'],'owned_array_reads'));assert v.dtype==torch.float64 and v.shape==(record['all_original_nodes'],3)
            fields[key]=v;records[key]=record
        d.write(o/'independent6_prediction_barrier.json',dict(status='completed',all6_finite_hash_body_Cartesian=True,GT_reads=0,
            compact_fullfield_comparisons_before_barrier=0,endpoint_equation='modified_scalar_Kspring',inventory=a['prediction_inventory']))
        a['prediction_barrier_complete']=True;persist()
        for cad,c in resident.items():
            a.update(current_CAD=cad,phase='postbarrier_compact_BC_comparison');persist();rec=a['case_records'][cad];spring=c['spring'];reference=c['reference'];inputs=c['inputs']
            qt=(time.time(),time.monotonic());out=spring.query_force6(c['forces6']);cs=torch.linalg.solve(c['B'],out['effective_original_force6'].T).T
            compact=torch.einsum('jk,kna->jna',cs,c['basis']);torch.cuda.synchronize();rec['compact_update_and_fullfield_seconds']=elapsed(qt)
            metrics={};equilibrium={}
            for j in range(3):
                key=cad+'/spring_mix'+str(j);v=fields[key];record=records[key];scaled=c['H']@v[c['ids']].reshape(-1)
                motion=scaled/reference.scales;fineKv=c['fine'].apply(v);springKv=c['scalar'].spring_action(v)
                spring_force=-springKv
                netforce=torch.tensor(record['net_force_N'],dtype=torch.float64,device='cuda')+spring_force.sum(0)
                netmoment=torch.tensor(record['net_moment_Nmm'],dtype=torch.float64,device='cuda')+torch.cross(inputs['vertices_mm']-inputs['reference_xyz_mm'],spring_force,dim=-1).sum(0)
                external_work=(c['forces'][j]*v).sum();body_work=(fineKv*v).sum();spring_work=(scaled.square()*spring.spring_diagonal).sum()
                eq_energy=metric((body_work+spring_work).reshape(1),external_work.reshape(1),1e-5)
                compact_energy=metric(out['total_strain_energy_Nmm'][j:j+1],(.5*external_work).reshape(1),1e-5)
                metrics[str(j)]=dict(fullN_U=metric(compact[j],v,1e-5),scaled_reference_pose=metric(out['reference_motion6'][j]*reference.scales,scaled,1e-5),
                    translation_mm=metric(out['reference_motion6'][j,:3],motion[:3],1e-5),rotation_rad=metric(out['reference_motion6'][j,3:],motion[3:],1e-5),
                    body_plus_spring_energy_vs_external_work=eq_energy,compact_total_energy_vs_independent=compact_energy)
                equilibrium[str(j)]=dict(net_force_including_grounded_spring_N=netforce.cpu().tolist(),net_moment_including_grounded_spring_Nmm=netmoment.cpu().tolist(),
                    original_endpoint_unbalanced_without_spring_force_N=record['net_force_N'],original_endpoint_unbalanced_without_spring_moment_Nmm=record['net_moment_Nmm'],
                    spring_reaction_force_N=spring_force.sum(0).cpu().tolist(),spring_strain_energy_Nmm=float(.5*spring_work),body_strain_energy_Nmm=float(.5*body_work),
                    external_work_Nmm=float(external_work),true_residual_equation_includes_spring=True)
            after=dict(basis=checksum(c['basis']),rawC=checksum(reference.C),**{k:checksum(v)for k,v in c['op_arrays'].items()})
            assert after==c['before']
            rec.update(compact_vs_independent_modified_equation=metrics,equilibrium_including_grounded_spring=equilibrium,
                rawC_fineK_basis_bytes_unchanged=True,modified_K_scalar_calls=c['scalar'].calls,unmodified_fine_K_calls_including_new_actions=c['fine'].calls,
                passed=all(v[k]['passed']for v in metrics.values()for k in ('fullN_U','scaled_reference_pose','body_plus_spring_energy_vs_external_work','compact_total_energy_vs_independent')))
            persist()
        a['endpoint_records']=records;a['readout_seconds']=elapsed(rt);assert a['readout_seconds']<=180
        conv=all(v['numerical_state']=='converged'and v['true_relative_residual']<=1e-8 for v in records.values())
        passed=conv and all(v['passed']for v in a['case_records'].values())
        a.update(status='completed',phase='terminal',spring_BC_gate_passed=passed,scientific_outcome='fixed_reference_spring_BC_PASS'if passed else'fixed_reference_spring_BC_negative')
    except Exception as error:
        from geobc.elastic_pilot import ScientificNegative
        if isinstance(error,ScientificNegative):a.update(status='completed',phase='terminal_scientific_negative',reason=str(error),scientific_outcome='fixed_reference_spring_BC_negative')
        else:a.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error));raise
    finally:
        a.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(o/'summary.json',a);d.write(o/'status.json',a)
        if lock is not None:lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');p.add_argument('--plan');args=p.parse_args()
    if args.start:
        if not args.plan:p.error('--start requires fixed --plan')
        run(args.plan)
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
