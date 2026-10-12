"""Three real physical bases: continuous explicit-cell API actual qualification."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,os,json,signal,hashlib,gc
from pathlib import Path

def run(path):
    from geobc import available_cad_pilot_data as d
    from geobc.physical_U_query_contract_v104 import approved_contract
    p,r=approved_contract(path);s=p['scientific'];o=d.path(p['output']);outer=json.loads(os.environ['GEOBC_PHYSICAL_U_QUERY_V104_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest'] and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=p['scientific_digest'],boot_id=outer['boot_id']);d.write(o/'execution.json',execution)
    a=dict(status='running',phase='cache',active=True,closed=False,execution=execution,input_cache_reads=0,canonical_basis_array_reads=0,owned_array_reads=0,GT_reads=0,raw_H5_deck_GRID_reads=0,K_PCG_replays=0,neural_PT_reads=0,neural_inference=0,training_updates=0,scaler_fits=0,FINAL_reads=0,case_records={},prediction_inventory={},committed_interior_queries=0,fixed_interior_query_denominator=2304,nodal_witness_positions=0,nodal_canonical_vectors=0,gradient_query_points=0,peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0,protocol_qualified=False,API_gate_passed=False,new_offnode_GT_accuracy=False,final_solver_selected=False)
    torch=None;lock=None
    def elapsed(t):return max(time.time()-t[0],time.monotonic()-t[1])
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Whole900 earlier absolute deadline')
        if torch is not None and torch.cuda.is_initialized():
            a['peak_allocated_bytes']=max(a['peak_allocated_bytes'],torch.cuda.max_memory_allocated());a['peak_reserved_bytes']=max(a['peak_reserved_bytes'],torch.cuda.max_memory_reserved());free,total=torch.cuda.mem_get_info();a['peak_total_device_used_bytes_sampled']=max(a['peak_total_device_used_bytes_sampled'],total-free)
            if max(a[k]for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))>30500000000:raise MemoryError('30.5GB cap')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External API guard')))
    try:
        import fcntl
        lock=d.path(s['budgets']['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch as tm
        torch=tm
        from geobc.physical_U_query_v104 import PhysicalUQuery,shape
        from geobc.continuous_query_geometry import quadratic_shape,shape_gradient,REFERENCE_BARYCENTRIC
        from geobc import elastic_pilot as old
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        total=torch.cuda.get_device_properties(0).total_memory;assert 30500000000<total;torch.cuda.set_per_process_memory_fraction(30500000000/total,0);torch.cuda.reset_peak_memory_stats()
        thresholds=s['actual_tests'];bary_np=np.array(thresholds['interior_barycentric_points'],dtype=np.float64)
        coeffs=torch.tensor(thresholds['query_load_coefficients'],dtype=torch.float64,device='cuda')
        def checksum(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        def metric(value,reference,limit):
            guard();delta=value-reference;norm=float(torch.linalg.vector_norm(reference));error=float(torch.linalg.vector_norm(delta));rms=float(torch.square(delta).mean().sqrt());rel=error/norm if norm>0 else None
            return dict(relative=rel,RMS_absolute=rms,reference_norm=norm,zero_reference=norm==0,passed=(rel<=limit if rel is not None else rms<=1e-12),fixed_scalar_components=reference.numel())
        def require(row,name):
            if not row['passed']:raise ValueError('Actual API qualification failed: '+name)
        def save_array(file,value):
            item=old._save_array(np,file,value);item.update(d.identity(item['path'],hashed=True));return item
        def read_field(item):
            expected={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')};expected['path']=d.relative(d.path(item['path']));assert d.identity(item['path'],hashed=True)==expected
            v=np.load(d.path(item['path']),allow_pickle=False);a['owned_array_reads']+=1;dtype={'torch.float64':'float64','torch.float32':'float32'}.get(item['dtype'],item['dtype']);assert list(v.shape)==item['shape'] and str(v.dtype)==dtype and np.isfinite(v).all();return v
        for cad in s['selection']['CADs']:
            a['current_CAD']=cad;a['phase']='input_and_basis_cache';persist();ct=(time.time(),time.monotonic());row=s['cases'][cad];folder=o/cad;folder.mkdir(exist_ok=False)
            cache=row['input_cache_identity'];assert d.identity(cache['path'],hashed=True)==cache
            with np.load(d.path(cache['path']),allow_pickle=False)as z:
                x_np=z['vertices_mm'];cells_np=z['cells10']
            a['input_cache_reads']+=1;n=row['expected_geometry']['nodes'];nc=row['expected_geometry']['cells'];assert x_np.shape==(n,3) and cells_np.shape==(nc,10) and x_np.dtype==np.float64 and cells_np.dtype==np.int64 and np.isfinite(x_np).all()
            basis_np=[]
            for load in ('ver','hor','tor'):
                inv=row['canonical_basis_manifest_identities'][load];assert d.identity(inv['path'],hashed=True)==inv;rec=d.read(inv['path']);v=read_field(rec['physical_U']);assert v.shape==(n,3) and v.dtype==np.float64;basis_np.append(v);a['canonical_basis_array_reads']+=1
            basis_np=np.stack(basis_np);x=torch.as_tensor(x_np,device='cuda');cells=torch.as_tensor(cells_np,device='cuda');basis=torch.as_tensor(basis_np,device='cuda');B=torch.tensor([row['reference_load6'][load]for load in ('ver','hor','tor')],dtype=torch.float64,device='cuda').T.contiguous()
            before={key:hashlib.sha256(value.tobytes()).hexdigest()for key,value in [('vertices',x_np),('cells',cells_np),('basis',basis_np)]};del x_np,cells_np,basis_np,v
            api=PhysicalUQuery(torch,x,cells,basis,B,guard);record=dict(API_constructor_actual_fullN_qualification=api.metadata,cache_init_seconds=elapsed(ct),qualification={},inherited_actual_cost=s['inherited_actual_basis_costs'][cad]);a['case_records'][cad]=record;persist()
            selected=torch.tensor(np.linspace(0,nc-1,32,dtype=np.int64),device='cuda');assert selected.unique().numel()==32
            element_rows=selected.repeat_interleave(4);b_np=np.tile(bary_np,(32,1));N=torch.tensor(quadratic_shape(b_np),device='cuda');Dold=torch.tensor(shape_gradient(b_np),device='cuda');nodes=x[cells[element_rows]]
            xyz=torch.einsum('qi,qia->qa',N,nodes);expected_basis=torch.einsum('qi,kqia->qka',N,basis[:,cells[element_rows]])
            Nt,Dt,bt=shape(torch,torch.tensor(b_np[:,:3],device='cuda'));q={}
            q['reference_shape']=metric(Nt,N,1e-10);require(q['reference_shape'],'shape weights')
            Nactual,Dactual,bactual,_=api.invert(xyz,element_rows);mapped=torch.einsum('qi,qia->qa',Nactual,nodes);scale=torch.linalg.vector_norm(nodes.amax(1)-nodes.amin(1),dim=-1)
            q['coordinate_roundtrip']=dict(maximum_relative=float((torch.linalg.vector_norm(mapped-xyz,dim=-1)/scale).max()),passed=bool((torch.linalg.vector_norm(mapped-xyz,dim=-1)/scale<=1e-9).all()));require(q['coordinate_roundtrip'],'physical roundtrip')
            q['basis_evaluation']=metric(api.basis_at(xyz,element_rows),expected_basis,1e-10);require(q['basis_evaluation'],'real physical basis')
            outputs=[];qt=(time.time(),time.monotonic())
            for c in coeffs:
                cb=c.expand(128,-1);actual=api.query_coefficients(xyz,element_rows,cb);reference=torch.einsum('qk,qka->qa',cb,expected_basis);qm=metric(actual,reference,1e-10);require(qm,'load-specific continuous field');outputs.append(actual)
            torch.cuda.synchronize();record['six_load_query_seconds']=elapsed(qt);out=torch.stack(outputs)
            chunks=torch.cat([api.query_coefficients(xyz[i:i+32],element_rows[i:i+32],coeffs[3].expand(min(32,128-i),-1))for i in range(0,128,32)])
            q['chunk_parity']=metric(chunks,out[3],1e-10);require(q['chunk_parity'],'chunk parity')
            amp=api.query_coefficients(xyz,element_rows,1.5*coeffs[3].expand(128,-1));q['homogeneity']=metric(amp,1.5*out[3],1e-10);require(q['homogeneity'],'load homogeneity')
            q['superposition']=metric(out[3],out[0]+.5*out[1],1e-10);require(q['superposition'],'load superposition')
            # Nodal witnesses use real boundary queries; no coordinate-gradient claim there.
            nodal_rows=selected.repeat_interleave(10);nodal_xyz=x[cells[selected]].reshape(-1,3);nodal=api.basis_at(nodal_xyz,nodal_rows);nodal_ref=basis[:,cells[selected]].permute(1,2,0,3).reshape(320,3,3)
            q['nodal_basis']=metric(nodal,nodal_ref,1e-10);require(q['nodal_basis'],'all real ten-node values');a['nodal_witness_positions']+=320;a['nodal_canonical_vectors']+=960
            a['phase']='actual_coordinate_and_rank3_gradient';persist();gx=xyz[:64].clone().requires_grad_(True);gc3=coeffs[3].expand(64,-1).clone().requires_grad_(True);val=api.query_coefficients(gx,element_rows[:64],gc3)
            gradx=[];gradc=[]
            for component in range(3):
                xx,cc=torch.autograd.grad(val[:,component].sum(),(gx,gc3),retain_graph=component<2);gradx.append(xx);gradc.append(cc)
            gradx=torch.stack(gradx,1);gradc=torch.stack(gradc,1);assert torch.isfinite(gradx).all() and torch.isfinite(gradc).all() and float(gradx.norm())>0 and float(gradc.norm())>0
            Jold=torch.einsum('qia,qir->qar',nodes[:64],Dold[:64]);effective=torch.einsum('qk,kqia->qia',gc3.detach(),basis[:,cells[element_rows[:64]]]);param=torch.einsum('qia,qir->qar',effective,Dold[:64]);analytic=param@torch.linalg.inv(Jold)
            q['coordinate_autograd_analytic']=metric(gradx,analytic,1e-8);require(q['coordinate_autograd_analytic'],'coordinate gradient frame')
            q['load_coefficient_gradient']=metric(gradc,expected_basis[:64].transpose(1,2),1e-10);require(q['load_coefficient_gradient'],'actual rank3 coefficient gradient')
            h=1e-5*scale[:64];fd=[]
            for direction in range(3):
                offset=torch.zeros_like(gx);offset[:,direction]=h
                plus=api.query_coefficients(gx.detach()+offset,element_rows[:64],gc3.detach());minus=api.query_coefficients(gx.detach()-offset,element_rows[:64],gc3.detach());fd.append((plus-minus)/(2*h[:,None]))
            fd=torch.stack(fd,-1);q['centered_finite_difference']=metric(fd,analytic,1e-5);require(q['centered_finite_difference'],'fixedh coordinate FD');a['gradient_query_points']+=64
            # Real B outside-span rejection, no synthetic field or learned sixth DOF.
            rejected=False
            try:api.coefficients(torch.tensor([[0.,1.,0.,0.,0.,0.]],dtype=torch.float64,device='cuda'))
            except ValueError:rejected=True
            assert rejected;q['outside_load_span_rejected']=dict(passed=True,actual_physical_load6=[0,1,0,0,0,0])
            after=dict(vertices=checksum(x),cells=checksum(cells),basis=checksum(basis));assert after==before;q['owned_arrays_immutable']=dict(passed=True,initial_digests=before,final_digests=after)
            inv=dict(CAD=cad,physical_U=save_array(folder/'queries_U_mm.npy',out),query_xyz=save_array(folder/'query_xyz_mm.npy',xyz),query_cell_rows=save_array(folder/'query_cell_rows.npy',element_rows),fixed_query_count=768,selected_original_cell_rows=selected.cpu().tolist(),load_coefficients=coeffs.cpu().tolist(),GT_reads=0,qualification=q)
            manifest=folder/'manifest.json';d.write(manifest,inv);a['prediction_inventory'][cad]=d.identity(manifest,hashed=True);a['committed_interior_queries']+=768;record.update(qualification=q,actual_nodal_positions=320,actual_nodal_vectors=960,actual_gradient_points=64,whole_case_seconds=elapsed(ct),passed=True);persist()
            del api,x,cells,basis,B,nodes,N,Dold,expected_basis,Nt,Dt,bt,Nactual,Dactual,bactual,mapped,scale,out,outputs,nodal,nodal_ref,gx,gc3,val,gradx,gradc,Jold,effective,param,analytic,h,fd,chunks,amp,plus,minus;gc.collect();torch.cuda.empty_cache()
        assert a['input_cache_reads']==3 and a['canonical_basis_array_reads']==9 and a['committed_interior_queries']==2304 and a['nodal_witness_positions']==960 and a['nodal_canonical_vectors']==2880 and a['gradient_query_points']==192
        a['phase']='all2304_query_barrier';persist();assert set(a['prediction_inventory'])==set(s['selection']['CADs'])
        for cad,item in a['prediction_inventory'].items():
            assert d.identity(item['path'],hashed=True)==item;inv=d.read(item['path']);assert inv['CAD']==cad and inv['fixed_query_count']==768 and inv['GT_reads']==0
            for name in ('physical_U','query_xyz','query_cell_rows'):read_field(inv[name])
        d.write(o/'query_prediction_barrier.json',dict(status='completed',complete_query_denominator=2304,allfinite_hash_checked=True,GT_reads=0,inventory=a['prediction_inventory']));a.update(status='completed',phase='terminal',prediction_barrier_complete=True,protocol_qualified=True,API_gate_passed=all(v['passed']for v in a['case_records'].values()),scientific_outcome='true_physical_U_query_and_coordinate_load_gradients_PASS')
    except Exception as error:
        if isinstance(error,ValueError):a.update(status='completed',phase='terminal_scientific_negative',scientific_outcome='fixed_API_qualification_negative',reason=str(error),protocol_qualified=False)
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
