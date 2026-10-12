"""Missing force directions and independent mixed validation, GT-free."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,json,os,signal,gc,hashlib
from pathlib import Path

def run(path):
    from geobc import available_cad_pilot_data as d
    from geobc.physics_U_fullrank6_contract_v108 import approved_contract
    p,r=approved_contract(path);s=p['scientific'];o=d.path(p['output'])
    outer=json.loads(os.environ['GEOBC_PHYSICS_U_FULLRANK6_V108_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest'] and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=p['scientific_digest'],boot_id=outer['boot_id']);d.write(o/'execution.json',execution)
    a=dict(status='running',phase='cache',active=True,closed=False,execution=execution,input_cache_reads=0,operator_array_reads=0,coarse_array_reads=0,
        old_canonical_reference_reads=0,owned_array_reads=0,GT_reads=0,raw_H5_deck_GRID_reads=0,geometry_K_coarse_rebuilds=0,inherited8fineguards_repeated=0,
        neural_PT_reads=0,neural_inference=0,training_updates=0,scaler_fits=0,FINAL_reads=0,prediction_inventory={},case_records={},
        fixed_new_endpoint_denominator=18,committed_finite_endpoints=0,combined_solve_seconds=0.,completed_groups={},API_query_inventory={},
        actual_mixed_query_count=0,actual_force6_gradient_points=0,peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0,
        prediction_barrier_complete=False,fullrank6_gate_passed=False,new_OptiStruct_load_or_offnode_GT_accuracy=False,signed_S_solved=False,final_solver_selected=False)
    torch=None;lock=None;ledger=None
    def elapsed(t):return max(time.time()-t[0],time.monotonic()-t[1])
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Whole1200 earlier absolute deadline')
        if torch is not None and torch.cuda.is_initialized():
            a['peak_allocated_bytes']=max(a['peak_allocated_bytes'],torch.cuda.max_memory_allocated());a['peak_reserved_bytes']=max(a['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
            free,total=torch.cuda.mem_get_info();a['peak_total_device_used_bytes_sampled']=max(a['peak_total_device_used_bytes_sampled'],total-free)
            if max(a[k]for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))>30500000000:raise MemoryError('30.5GB cap')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External fullrank6 guard')))
    try:
        import fcntl
        lock=d.path(s['budgets']['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch as tm
        torch=tm
        from geobc import elastic_pilot as old,elastic_coarse as coarse
        from geobc.load_linear_physics_pcg_v100 import ParallelElastic
        from geobc.sixload_coarse_v102 import Ledger,ThreeLaneCoarse,solve_group
        from geobc.physical_U_fullrank6_v108 import FullRank6PhysicalUQuery
        from geobc.physical_U_query_v104 import shape
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        total=torch.cuda.get_device_properties(0).total_memory;assert 30500000000<total;torch.cuda.set_per_process_memory_fraction(30500000000/total,0);torch.cuda.reset_peak_memory_stats()
        def read_array(item):
            guard();expected={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')};expected['path']=d.relative(d.path(item['path']));assert d.identity(item['path'],hashed=True)==expected
            v=np.load(d.path(item['path']),allow_pickle=False);a['owned_array_reads']+=1
            dtype={'torch.float64':'float64','torch.float32':'float32','torch.int64':'int64'}.get(item['dtype'],item['dtype']);assert list(v.shape)==item['shape'] and str(v.dtype)==dtype and np.isfinite(v).all();return v
        def load_sparse(item):
            rr,cc,vv=[torch.as_tensor(read_array(item[k]),device='cuda')for k in ('rowptr','columns','values')];a['coarse_array_reads']+=3
            nr,nc=item['shape'];assert rr.dtype==cc.dtype==torch.int64 and vv.dtype==torch.float64 and rr.shape==(nr+1,) and cc.shape==vv.shape
            assert int(rr[0])==0 and int(rr[-1])==len(vv) and bool((rr[1:]>=rr[:-1]).all()) and bool(((cc>=0)&(cc<nc)).all())
            return torch.sparse_csr_tensor(rr,cc,vv,size=(nr,nc),device='cuda')
        def save_array(file,value):
            guard();item=old._save_array(np,file,value);item.update(d.identity(item['path'],hashed=True));return item
        def metric(pred,ref,limit):
            norm=float(torch.linalg.vector_norm(ref));error=float(torch.linalg.vector_norm(pred-ref));absolute=float((pred-ref).abs().max());relative=error/norm if norm>0 else None
            return dict(relative=relative,max_absolute_mm=absolute,passed=(relative<=limit if relative is not None else absolute<=1e-12))
        # Hold existing canonical field bodies closed until every new solve is saved.
        retained={};coeff=torch.tensor(s['selection']['mixed_coefficients'],dtype=torch.float64,device='cuda')
        for cad in s['selection']['CADs']:
            ct=(time.time(),time.monotonic());row=s['cases'][cad];folder=o/cad;folder.mkdir(exist_ok=False);a.update(current_CAD=cad,phase='strict_cached_operator');persist()
            cache=row['input_cache_identity'];assert d.identity(cache['path'],hashed=True)==cache
            with np.load(d.path(cache['path']),allow_pickle=False)as z:arrays={k:z[k]for k in z.files}
            a['input_cache_reads']+=1;assert all(np.isfinite(v).all()for v in arrays.values());old_rhs=arrays.pop('all_six_forces')
            n=row['expected_geometry']['nodes'];nc=row['expected_geometry']['cells'];assert arrays['vertices_mm'].shape==(n,3) and arrays['vertices_mm'].dtype==np.float64 and arrays['cells10'].shape==(nc,10) and arrays['cells10'].dtype==np.int64
            inputs={k:torch.as_tensor(v,device='cuda')for k,v in arrays.items()};del arrays
            B=torch.tensor(row['reference_load6_columns'],dtype=torch.float64,device='cuda').T.contiguous();assert int(torch.linalg.matrix_rank(B))==6 and float(torch.linalg.cond(B))<=1e12
            H=inputs['rbe3_Hbar'];length=float(inputs['rbe3_length_mm']);members=inputs['rbe3_member_node_ids'];offset=inputs['vertices_mm'][members]-inputs['reference_xyz_mm']
            canonical_rhs=[];transfer=[]
            for j in range(6):
                f6=B[:,j];scaled=torch.cat((f6[:3],f6[3:]/length));member_force=(H.T@scaled).reshape(-1,3);force=torch.zeros((n,3),dtype=torch.float64,device='cuda');force.index_add_(0,members,member_force)
                ferr=float((member_force.sum(0)-f6[:3]).abs().max())/max(1.,float(scaled.norm()))
                merr=float((torch.cross(offset,member_force,dim=-1).sum(0)-f6[3:]).abs().max())/max(1.,length*float(scaled.norm()))
                witness=torch.arange(member_force.numel(),dtype=torch.float64,device='cuda').sin();work=float((witness@(H.T@scaled)-(H@witness)@scaled).abs())/max(1.,float(witness.norm())*float((H.T@scaled).norm()))
                if max(ferr,merr,work)>s['physics_recipe']['load_transfer']['relative_guard_tolerance']:raise old.ScientificNegative('Full6RBE force/moment/work transfer failed')
                transfer.append(dict(force_relative=ferr,moment_relative=merr,virtual_work_relative=work,passed=True));canonical_rhs.append(force)
            forces=torch.stack(canonical_rhs);old_t=torch.as_tensor(old_rhs[:3],device='cuda')
            assert torch.allclose(forces[:3],old_t,rtol=1e-12,atol=1e-12);del old_rhs,old_t,canonical_rhs
            mixed=torch.einsum('mk,kna->mna',coeff,forces);assert bool(torch.isfinite(mixed).all())
            op_arrays={k:torch.as_tensor(read_array(item),device='cuda')for k,item in row['operator_cache'].items()};a['operator_array_reads']+=3
            assert op_arrays['gradients'].shape==(nc,4,10,3) and op_arrays['weights'].shape==(nc,4)
            assert op_arrays['free_cholesky'].shape==(len(inputs['free_node_ids']),3,3)
            lam,mu=map(float,inputs['lame'].tolist());op=old.MatrixFreeElastic(torch,inputs['cells10'],op_arrays['gradients'],op_arrays['weights'],n,lam,mu);chol=op_arrays['free_cholesky'];batch=ParallelElastic(torch,op,guard)
            cm=row['coarse_arrays'];P,PT,Ac=[load_sparse(cm[name])for name in ('P','PT','Ac')]
            diagonal=torch.as_tensor(read_array(cm['diagonal']),device='cuda');a['coarse_array_reads']+=1
            assert diagonal.dtype==torch.float64 and bool((diagonal>0).all())
            assert P.shape[0]==PT.shape[1]==len(inputs['free_node_ids']) and P.shape[1]==PT.shape[0]
            assert 3*P.shape[1]==Ac.shape[0]==Ac.shape[1]==len(diagonal)
            ledger=Ledger(torch,guard,persist,folder,ct);ledger.setup_active=False
            scalar=coarse.AdditiveCoarse(torch,P,PT,Ac,diagonal,row['coarse_assembly']['symmetric_scaled_Gershgorin_Gamma'],chol,ledger);parallel=ThreeLaneCoarse(torch,scalar,guard)
            rec=dict(actual_allN=n,actual_cells=nc,physical_load_rank=6,physical_load_condition=float(torch.linalg.cond(B)),source_REFc=row['source_load_transfer_facts']['REFC'],six_transfer_qualification=transfer,cache_seconds=elapsed(ct),inherited_actual_basis_cost=row['inherited_actual_basis_cost'],groups={})
            assert rec['cache_seconds']<=120;a['case_records'][cad]=rec
            np.savez(folder/'six_RBE_reference_rhs.npz',basis_force6=B.cpu().numpy(),basis_nodal_rhs=forces.cpu().numpy(),mixed_nodal_rhs=mixed.cpu().numpy())
            rec['actual_rhs_cache_identity']=d.identity(folder/'six_RBE_reference_rhs.npz',hashed=True);persist()
            for role,names,rhs in [('missing_basis',s['selection']['new_basis_names'],forces[3:]),('independent_mixed6',s['selection']['validation_mix_names'],mixed)]:
                a['phase']=cad+'/'+role;persist();began=(time.time(),time.monotonic())
                def remaining():return 900-a['combined_solve_seconds']-elapsed(began)
                inv,reported=solve_group(torch,np,batch,chol,inputs,rhs,names,folder/role,a,guard,persist,remaining,parallel.apply)
                wall=elapsed(began);assert wall<=300;a['combined_solve_seconds']+=wall;rec['groups'][role]=dict(wall_seconds=wall,reported_seconds=reported,initial_guess='independent_zero_all3')
                a['prediction_inventory'].update({cad+'/'+name:item for name,item in inv.items()});a['committed_finite_endpoints']+=3;persist()
            rec.update(whole_numeric_case_seconds=elapsed(ct),operator_calls=dict(scalar=op.calls,batch=batch.calls,coarse_parallel=parallel.calls));d.write(folder/'case_numeric_terminal.json',rec)
            # Retain only physical geometry on host for the post-barrier API; no reread.
            retained[cad]=(inputs['vertices_mm'].cpu().numpy(),inputs['cells10'].cpu().numpy(),B.cpu().numpy())
            del inputs,forces,mixed,rhs,op_arrays,op,chol,batch,P,PT,Ac,diagonal,scalar,parallel,H,members,offset,f6,force,member_force,ledger;ledger=None;gc.collect();torch.cuda.empty_cache()
        assert a['committed_finite_endpoints']==18 and a['input_cache_reads']==3 and a['operator_array_reads']==9 and a['coarse_array_reads']==30 and a['old_canonical_reference_reads']==0
        a['phase']='all18_new_independent_field_barrier';persist();readout=(time.time(),time.monotonic());fields={};records={}
        expected={cad+'/'+name for cad in s['selection']['CADs']for name in s['selection']['new_basis_names']+s['selection']['validation_mix_names']};assert set(a['prediction_inventory'])==expected
        for key,item in a['prediction_inventory'].items():
            assert d.identity(item['path'],hashed=True)==item;rec=d.read(item['path']);cad,name=key.split('/');assert rec['initial_guess']=='independent_zero' and rec['key']==name and rec['all_original_nodes']==s['cases'][cad]['expected_geometry']['nodes']
            v=read_array(rec['physical_U']);assert v.shape==(rec['all_original_nodes'],3) and v.dtype==np.float64;fields[key]=v;records[key]=rec
        d.write(o/'new18_prediction_barrier.json',dict(status='completed',all18_finite_hash_body_Cartesian=True,old_canonical_reads_before_barrier=0,GT_reads=0,inventory=a['prediction_inventory']));a['prediction_barrier_complete']=True;persist()
        for cad in s['selection']['CADs']:
            a.update(current_CAD=cad,phase='fullrank6_API_and_force_gradient');persist();row=s['cases'][cad];basis_np=[]
            for load in ('ver','hor','tor'):
                inv=row['old_canonical_manifest_identities'][load];assert d.identity(inv['path'],hashed=True)==inv
                v=read_array(d.read(inv['path'])['physical_U']);a['old_canonical_reference_reads']+=1;basis_np.append(v)
            basis_np.extend(fields[cad+'/'+name]for name in s['selection']['new_basis_names']);basis_np=np.stack(basis_np)
            xn,cn,bn=retained.pop(cad);x=torch.as_tensor(xn,device='cuda');cells=torch.as_tensor(cn,device='cuda');B=torch.as_tensor(bn,device='cuda');basis=torch.as_tensor(basis_np,device='cuda')
            api=FullRank6PhysicalUQuery(torch,x,cells,basis,B,guard);before=hashlib.sha256(basis_np.tobytes()).hexdigest();rec=a['case_records'][cad];same={}
            for i,c in enumerate(coeff):
                actual=torch.as_tensor(fields[cad+'/mix6_'+str(i)],device='cuda');derived=torch.einsum('k,kna->na',c,basis);same['mix6_'+str(i)]=metric(derived,actual,1e-5)
            selected=torch.linspace(0,len(cells)-1,32,device='cuda').to(torch.int64);assert selected.unique().numel()==32
            ids=selected.repeat_interleave(4);bary=torch.tensor([[.25,.25,.25],[.4,.2,.2],[.2,.4,.2],[.2,.2,.4]],dtype=torch.float64,device='cuda').repeat(32,1)
            N,_,_=shape(torch,bary);xyz=torch.einsum('qi,qia->qa',N,x[cells[ids]]);outputs=[];qm={}
            for i,c in enumerate(coeff):
                actual=api.query_coefficients(xyz,ids,c.expand(128,-1));independent=torch.as_tensor(fields[cad+'/mix6_'+str(i)],device='cuda');ref=torch.einsum('qi,qia->qa',N,independent[cells[ids]])
                qm['mix6_'+str(i)]=metric(actual,ref,1e-5);outputs.append(actual)
            out=torch.stack(outputs);chunk=torch.cat([api.query_coefficients(xyz[j:j+32],ids[j:j+32],coeff[0].expand(32,-1))for j in range(0,128,32)]);parity=metric(chunk,out[0],1e-10)
            f=(coeff[0]@B.T).expand(64,-1).clone().requires_grad_(True);u=api.query(xyz[:64],ids[:64],f);grad=[]
            for j in range(3):grad.append(torch.autograd.grad(u[:,j].sum(),f,retain_graph=j<2)[0])
            grad=torch.stack(grad,1);analytic=api.basis_at(xyz[:64],ids[:64]).transpose(1,2)@torch.linalg.inv(B)
            gm=metric(grad,analytic,1e-10);assert bool(torch.isfinite(grad).all()) and bool((grad.square().sum((0,1))>0).all())
            assert hashlib.sha256(basis.detach().cpu().numpy().tobytes()).hexdigest()==before
            item=save_array(o/cad/'full6_mixed_queries_U_mm.npy',out);a['API_query_inventory'][cad]=item;a['actual_mixed_query_count']+=384;a['actual_force6_gradient_points']+=64
            rec.update(fullrank6_API=api.metadata,numeric_same_equation=same,query_independent_P2=qm,chunk_parity=parity,physical_force6_autograd=gm,basis_immutable=True,passed=all(v['passed']for v in [*same.values(),*qm.values(),parity,gm]))
            persist();del api,x,cells,B,basis,xn,cn,bn,basis_np,v,actual,derived,independent,ref,out,outputs,grad,analytic,u,f,chunk;gc.collect();torch.cuda.empty_cache()
        assert a['old_canonical_reference_reads']==9 and a['actual_mixed_query_count']==1152 and a['actual_force6_gradient_points']==192
        for cad,item in a['API_query_inventory'].items():v=read_array(item);assert v.shape==(3,128,3) and v.dtype==np.float64
        conv=all(v['numerical_state']=='converged' and v['true_relative_residual']<=1e-8 for v in records.values());a['endpoint_records']=records
        a.update(status='completed',phase='terminal',fullrank6_gate_passed=conv and all(v['passed']for v in a['case_records'].values()),readout_and_API_seconds=elapsed(readout),API_prediction_barrier_complete=True)
        assert a['readout_and_API_seconds']<=120
        a['scientific_outcome']='fullrank6_reference_response_PASS'if a['fullrank6_gate_passed']else'fixed_fullrank6_gate_negative'
    except Exception as error:
        from geobc.elastic_pilot import ScientificNegative
        if isinstance(error,ScientificNegative):a.update(status='completed',phase='terminal_scientific_negative',reason=str(error),scientific_outcome='fixed_fullrank6_physics_negative')
        else:a.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error));raise
    finally:
        a.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'summary.json',a);d.write(o/'status.json',a)
        if lock is not None:lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');p.add_argument('--plan');args=p.parse_args()
    if args.start:
        if not args.plan:p.error('--start requires fixed --plan')
        run(args.plan)
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
