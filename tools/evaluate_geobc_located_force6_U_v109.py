"""Actual reusable coordinate-only six-force response and batch rejection."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,json,os,signal,hashlib,gc
from pathlib import Path

def run(path):
    from geobc import available_cad_pilot_data as d
    from geobc.located_force6_U_contract_v109 import approved_contract
    from geobc.physical_U_global_locator_v106 import LocatorNegative
    p,r=approved_contract(path);s=p['scientific'];o=d.path(p['output']);outer=json.loads(os.environ['GEOBC_LOCATED_FORCE6_U_V109_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest'] and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=p['scientific_digest'],boot_id=outer['boot_id']);d.write(o/'execution.json',execution)
    a=dict(status='running',phase='cache',active=True,closed=False,execution=execution,input_cache_reads=0,basis_array_reads=0,owned_array_reads=0,
        direct_reference_API_calls=0,direct_reference_calls_before_barrier=0,GT_reads=0,K_PCG_replays=0,raw_H5_deck_GRID_reads=0,
        neural_PT_reads=0,neural_inference=0,training_updates=0,scaler_fits=0,FINAL_reads=0,case_records={},prediction_inventory={},
        committed_U_queries=0,committed_positive_owner_points=0,committed_outside_points=0,mixed_batch_rejection_count=0,extra_mixed_batch_point_status_count=0,
        actual_coupled_gradient_points=0,peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0,
        prediction_barrier_complete=False,coupled_API_gate_passed=False,new_offnode_GT_accuracy=False,signed_S_solved=False,final_solver_selected=False)
    torch=None;lock=None
    def elapsed(t):return max(time.time()-t[0],time.monotonic()-t[1])
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Whole900 earlier absolute deadline')
        if torch is not None and torch.cuda.is_initialized():
            a['peak_allocated_bytes']=max(a['peak_allocated_bytes'],torch.cuda.max_memory_allocated());a['peak_reserved_bytes']=max(a['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
            free,total=torch.cuda.mem_get_info();a['peak_total_device_used_bytes_sampled']=max(a['peak_total_device_used_bytes_sampled'],total-free)
            if max(a[k]for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))>30500000000:raise MemoryError('30.5GB cap')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External coupledAPI guard')))
    try:
        import fcntl
        lock=d.path(s['budgets']['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch as tm
        torch=tm
        from geobc import elastic_pilot as old
        from geobc.physical_U_fullrank6_v108 import FullRank6PhysicalUQuery
        from geobc.physical_U_global_locator_v106 import PhysicalULocator
        from geobc.located_force6_U_response_v109 import LocatedForce6UResponse
        from geobc.physical_U_query_v104 import shape
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        total=torch.cuda.get_device_properties(0).total_memory;assert 30500000000<total;torch.cuda.set_per_process_memory_fraction(30500000000/total,0);torch.cuda.reset_peak_memory_stats()
        coeff=torch.tensor(s['evaluation']['mixed_coefficients'],dtype=torch.float64,device='cuda');resident={}
        def read_array(item):
            guard();expected={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')};expected['path']=d.relative(d.path(item['path']));assert d.identity(item['path'],hashed=True)==expected
            v=np.load(d.path(item['path']),allow_pickle=False);a['owned_array_reads']+=1
            dtype={'torch.float64':'float64','torch.float32':'float32','torch.int64':'int64'}.get(item['dtype'],item['dtype']);assert list(v.shape)==item['shape'] and str(v.dtype)==dtype and np.isfinite(v).all();return v
        def save_array(file,value):
            guard();item=old._save_array(np,file,value);item.update(d.identity(item['path'],hashed=True));return item
        def checksum(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        def metric(value,ref):
            error=float(torch.linalg.vector_norm(value-ref));norm=float(torch.linalg.vector_norm(ref));absolute=float((value-ref).abs().max());rel=error/norm if norm>0 else None
            return dict(relative=rel,maximum_absolute=absolute,passed=rel<=1e-10 if rel is not None else absolute<=1e-12)
        for cad in s['selection']['CADs']:
            a.update(current_CAD=cad,phase='strict_cache');persist();ct=(time.time(),time.monotonic());row=s['cases'][cad];folder=o/cad;folder.mkdir(exist_ok=False)
            item=row['input_cache_identity'];assert d.identity(item['path'],hashed=True)==item
            with np.load(d.path(item['path']),allow_pickle=False)as z:xn,cn=z['vertices_mm'],z['cells10']
            a['input_cache_reads']+=1;n=row['expected_geometry']['nodes'];nc=row['expected_geometry']['cells'];assert xn.shape==(n,3) and xn.dtype==np.float64 and cn.shape==(nc,10) and cn.dtype==np.int64 and np.isfinite(xn).all()
            values=[]
            for name in ('ver','hor','tor','Fy','Mx','My'):
                inv=row['basis_manifest_identities'][name];assert d.identity(inv['path'],hashed=True)==inv
                v=read_array(d.read(inv['path'])['physical_U']);a['basis_array_reads']+=1;assert v.shape==(n,3) and v.dtype==np.float64;values.append(v)
            bn=np.stack(values);before={k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in [('vertices',xn),('cells',cn),('basis',bn)]}
            x=torch.as_tensor(xn,device='cuda');cells=torch.as_tensor(cn,device='cuda');basis=torch.as_tensor(bn,device='cuda');B=torch.tensor(row['reference_load6_columns'],dtype=torch.float64,device='cuda').T.contiguous()
            api=FullRank6PhysicalUQuery(torch,x,cells,basis,B,guard);it=(time.time(),time.monotonic());locator=PhysicalULocator(torch,x,cells,guard);torch.cuda.synchronize()
            source=dict(REFC=row['source_transfer_facts']['REFC'],reference_GRID_ID=row['source_transfer_facts']['reference_GRID_ID'],material_constraints_fixed=True,physical_U_units='mm',force_units='N',moment_units='Nmm')
            facade=LocatedForce6UResponse(api,locator,source)
            rec=dict(actual_fullN=api.metadata,index_metadata=locator.metadata,index_build_seconds=elapsed(it),interface_metadata=facade.metadata,cache_and_constructor_seconds=elapsed(ct),inherited_actual_basis_cost=row['inherited_actual_basis_cost'],missing_basis_cost_seconds=row['missing_basis_cost_seconds'],independent_validation_cost_seconds=row['independent_validation_cost_seconds']);a['case_records'][cad]=rec
            # Fixture row IDs generate predictor-input coordinates only. No IDs enter facade.
            fixture=torch.linspace(0,nc-1,8,device='cuda').to(torch.int64);assert fixture.unique().numel()==8
            bary=torch.tensor(s['implementation']['geometry_fixture_bary'][:3],dtype=torch.float64,device='cuda').expand(8,-1);N,_,_=shape(torch,bary);xyz=torch.einsum('qi,qia->qa',N,x[cells[fixture]])
            center=(locator.domain_low+locator.domain_high)/2;extent=locator.extent
            directions=torch.cat((torch.eye(3,device='cuda'),-torch.eye(3,device='cuda'))).to(torch.float64)
            corners=torch.cartesian_prod(*(torch.tensor([-1.,1.],dtype=torch.float64,device='cuda')for _ in range(3)))
            outside=torch.cat((center+2*extent*directions,center+2*extent*corners,center[None]+3*extent*directions[:2]));assert outside.shape==(16,3)
            a['phase']='coordinate_force6_prediction_no_IDs';persist();qt=(time.time(),time.monotonic());outputs=[];owner=None;location_records=None
            for c in coeff:
                f=(c@B.T).expand(8,-1);result=facade.query(xyz,f)
                if result['status']!='completed' or not all(v['status']=='strict_interior' for v in result['location_records']):raise LocatorNegative('New fixedcoupled positive point not strictowner')
                if owner is not None:assert torch.equal(owner,result['located_cell_rows'])
                owner=result['located_cell_rows'];location_records=result['location_records'];outputs.append(result['physical_U'])
            out=torch.stack(outputs);negative=facade.query(outside,(coeff[0]@B.T).expand(16,-1))
            if negative['status']!='rejected' or negative['physical_U']is not None or not bool((negative['located_cell_rows']==-1).all())or not all(v['status']=='outside_global_box'for v in negative['location_records']):raise LocatorNegative('Outsidebatch rejection violated')
            # A mixed valid/invalid request returns no partial field.
            mixed_xyz=torch.cat((xyz,outside[:1]));mixed=facade.query(mixed_xyz,(coeff[0]@B.T).expand(9,-1))
            if mixed['status']!='rejected' or mixed['physical_U']is not None or mixed['subset_prediction_allowed']:raise LocatorNegative('Mixedbatch returned subset')
            assert bool((mixed['located_cell_rows'][:8]>=0).all()) and int(mixed['located_cell_rows'][-1])==-1
            a['mixed_batch_rejection_count']+=1;a['extra_mixed_batch_point_status_count']+=9
            chunk=torch.cat([facade.query(xyz[j:j+4],(coeff[0]@B.T).expand(4,-1))['physical_U']for j in (0,4)])
            parity=metric(chunk,out[0]);assert parity['passed'];torch.cuda.synchronize();rec['actual_query_reject_chunk_seconds']=elapsed(qt)
            d.write(folder/'location_decisions.json',dict(positive=location_records,outside=negative['location_records'],extra_mixed_rejection=mixed['location_records'],wholebatch_rejection=True))
            inv=dict(CAD=cad,U=save_array(folder/'U_mm.npy',out),located_IDs=save_array(folder/'located_cell_rows.npy',owner),outside_IDs=save_array(folder/'outside_cell_rows.npy',negative['located_cell_rows']),decisions_identity=d.identity(folder/'location_decisions.json',hashed=True),query_count=24,positive_points=8,outside_points=16,direct_reference_calls_before_barrier=0,caller_element_IDs_supplied=False,chunk_parity=parity)
            d.write(folder/'manifest.json',inv);a['prediction_inventory'][cad]=d.identity(folder/'manifest.json',hashed=True);a['committed_U_queries']+=24;a['committed_positive_owner_points']+=8;a['committed_outside_points']+=16
            rec.update(whole_prediction_case_seconds=elapsed(ct),chunk_parity=parity)
            # Resident samefield data stays in memory; no repeated body loads for references.
            resident[cad]=dict(api=api,locator=locator,facade=facade,x=x,cells=cells,basis=basis,B=B,xyz=xyz,fixture=fixture,out=out,owner=owner,before=before)
            persist();del xn,cn,bn,values,v,outputs,chunk
        assert a['input_cache_reads']==3 and a['basis_array_reads']==18 and a['direct_reference_API_calls']==0 and a['committed_U_queries']==72 and a['committed_positive_owner_points']==24 and a['committed_outside_points']==48
        assert a['mixed_batch_rejection_count']==3 and a['extra_mixed_batch_point_status_count']==27
        a['phase']='complete72_24_48_hash_body_barrier';persist();assert set(a['prediction_inventory'])==set(s['selection']['CADs'])
        for cad,item in a['prediction_inventory'].items():
            assert d.identity(item['path'],hashed=True)==item;inv=d.read(item['path']);assert inv['CAD']==cad and inv['query_count']==24 and not inv['caller_element_IDs_supplied'] and inv['direct_reference_calls_before_barrier']==0
            uv=read_array(inv['U']);iv=read_array(inv['located_IDs']);ov=read_array(inv['outside_IDs']);assert uv.shape==(3,8,3) and iv.shape==(8,) and ov.shape==(16,) and (ov==-1).all()
            assert d.identity(inv['decisions_identity']['path'],hashed=True)==inv['decisions_identity']
        d.write(o/'prediction_barrier.json',dict(status='completed',complete_U_queries=72,positive_owner_points=24,outside_points=48,extra_mixed_batch_rejections=3,extra_mixed_point_statuses=27,all_finite_hash_body_Cartesian=True,direct_reference_calls_before_barrier=0,GT_reads=0,inventory=a['prediction_inventory']));a['prediction_barrier_complete']=True;persist()
        for cad,cache in resident.items():
            a.update(current_CAD=cad,phase='postbarrier_samefield_coupled_gradient');persist();api,facade=cache['api'],cache['facade'];xyz,ids,B=cache['xyz'],cache['fixture'],cache['B'];out=cache['out'];rec=a['case_records'][cad]
            qm={}
            for j,c in enumerate(coeff):
                direct=api.query(xyz,ids,(c@B.T).expand(8,-1));a['direct_reference_API_calls']+=1;qm[str(j)]=metric(out[j],direct)
            owner_match=bool(torch.equal(cache['owner'],ids))
            xx=xyz[:2].clone().requires_grad_(True);ff=(coeff[0]@B.T).expand(2,-1).clone().requires_grad_(True)
            val=facade.query(xx,ff);assert val['status']=='completed';gx=[];gf=[]
            for component in range(3):
                dx,df=torch.autograd.grad(val['physical_U'][:,component].sum(),(xx,ff),retain_graph=component<2);gx.append(dx);gf.append(df)
            gx,gf=torch.stack(gx,1),torch.stack(gf,1);assert bool(torch.isfinite(gx).all() and torch.isfinite(gf).all()) and float(gx.norm())>0 and bool((gf.square().sum((0,1))>0).all())
            rx=xyz[:2].clone().requires_grad_(True);rf=(coeff[0]@B.T).expand(2,-1).clone().requires_grad_(True);rv=api.query(rx,ids[:2],rf);a['direct_reference_API_calls']+=1;rgx=[];rgf=[]
            for component in range(3):
                dx,df=torch.autograd.grad(rv[:,component].sum(),(rx,rf),retain_graph=component<2);rgx.append(dx);rgf.append(df)
            mx=metric(gx,torch.stack(rgx,1));mf=metric(gf,torch.stack(rgf,1));a['actual_coupled_gradient_points']+=2
            after=dict(vertices=checksum(cache['x']),cells=checksum(cache['cells']),basis=checksum(cache['basis']));assert after==cache['before']
            rec.update(samefield_metrics=qm,exact_fixture_owner_match=owner_match,coupled_XYZ_gradient=mx,coupled_force6_gradient=mf,input_basis_bytes_unchanged=True,passed=owner_match and all(v['passed']for v in qm.values())and mx['passed']and mf['passed']);persist()
        assert a['actual_coupled_gradient_points']==6 and a['direct_reference_API_calls']==12
        a.update(status='completed',phase='terminal',coupled_API_gate_passed=all(v['passed']for v in a['case_records'].values()),scientific_outcome='located_force6_U_coupling_PASS'if all(v['passed']for v in a['case_records'].values())else'fixed_coupling_negative')
    except LocatorNegative as error:
        a.update(status='completed',phase='terminal_scientific_negative',reason=str(error),scientific_outcome='fixed_locator_coupling_negative')
    except Exception as error:
        a.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error));raise
    finally:
        a.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'summary.json',a);d.write(o/'status.json',a)
        if lock is not None:lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');p.add_argument('--plan');args=p.parse_args()
    if args.start:
        if not args.plan:p.error('--start requires fixed --plan')
        run(args.plan)
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
