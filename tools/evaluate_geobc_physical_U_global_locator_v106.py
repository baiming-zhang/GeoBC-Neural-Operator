"""Predict full-CAD owners/continuous U before opening held numerical references."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,os,json,signal,hashlib,gc
from pathlib import Path

def run(path):
    from geobc import available_cad_pilot_data as d
    from geobc.physical_U_global_locator_contract_v106 import approved_contract
    from geobc.physical_U_global_locator_v106 import LocatorNegative
    p,r=approved_contract(path);s=p['scientific'];o=d.path(p['output'])
    outer=json.loads(os.environ['GEOBC_PHYSICAL_U_LOCATOR_V106_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest']
    assert outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=p['scientific_digest'],boot_id=outer['boot_id'])
    d.write(o/'execution.json',execution)
    a=dict(status='running',phase='cache',active=True,closed=False,execution=execution,input_cache_reads=0,
        canonical_basis_array_reads=0,query_input_array_reads=0,held_numeric_reference_array_reads=0,owned_array_reads=0,
        GT_reads=0,raw_H5_deck_GRID_reads=0,K_PCG_replays=0,neural_PT_reads=0,neural_inference=0,
        training_updates=0,scaler_fits=0,FINAL_reads=0,case_records={},prediction_inventory={},
        committed_field_queries=0,committed_located_IDs=0,committed_outside_decisions=0,
        fixed_denominators=dict(field_queries=2304,located_IDs=384,outside_decisions=48,CADs=3),
        peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0,
        prediction_barrier_complete=False,locator_gate_passed=False,new_offnode_GT_accuracy=False,
        signed_S_solved=False,final_solver_selected=False)
    torch=None;lock=None
    def elapsed(t):return max(time.time()-t[0],time.monotonic()-t[1])
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Whole900 earlier deadline')
        if torch is not None and torch.cuda.is_initialized():
            a['peak_allocated_bytes']=max(a['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
            a['peak_reserved_bytes']=max(a['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
            free,total=torch.cuda.mem_get_info();a['peak_total_device_used_bytes_sampled']=max(a['peak_total_device_used_bytes_sampled'],total-free)
            if max(a[k]for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))>30500000000:raise MemoryError('30.5GB cap')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External locator guard')))
    try:
        import fcntl
        lock=d.path(s['budgets']['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch as tm
        torch=tm
        from geobc.physical_U_query_v104 import PhysicalUQuery
        from geobc.physical_U_global_locator_v106 import PhysicalULocator,LocatorNegative
        from geobc import elastic_pilot as old
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        total=torch.cuda.get_device_properties(0).total_memory;assert 30500000000<total
        torch.cuda.set_per_process_memory_fraction(30500000000/total,0);torch.cuda.reset_peak_memory_stats()
        coeffs=torch.tensor(s['evaluation']['coefficient_order'],dtype=torch.float64,device='cuda')
        def checksum(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        def read_field(item):
            guard();expected={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')}
            expected['path']=d.relative(d.path(item['path']));assert d.identity(item['path'],hashed=True)==expected
            v=np.load(d.path(item['path']),allow_pickle=False);a['owned_array_reads']+=1
            dtype={'torch.float64':'float64','torch.float32':'float32','torch.int64':'int64'}.get(item['dtype'],item['dtype'])
            assert list(v.shape)==item['shape'] and str(v.dtype)==dtype and np.isfinite(v).all();return v
        def save_array(file,value):
            guard();item=old._save_array(np,file,value);item.update(d.identity(item['path'],hashed=True));return item
        def metric(value,reference):
            delta=value-reference;norm=float(np.linalg.norm(reference));err=float(np.linalg.norm(delta))
            absolute=float(np.max(np.abs(delta)));rel=err/norm if norm>0 else None
            return dict(relative=rel,maximum_absolute_mm=absolute,reference_norm=norm,
                        passed=(rel<=1e-10 if rel is not None else absolute<=1e-12))
        for cad in s['selection']['CADs']:
            a.update(current_CAD=cad,phase='input_basis_cache');persist();ct=(time.time(),time.monotonic())
            row=s['cases'][cad];folder=o/cad;folder.mkdir(exist_ok=False);cache=row['input_cache_identity']
            assert d.identity(cache['path'],hashed=True)==cache
            with np.load(d.path(cache['path']),allow_pickle=False)as z:x_np,cells_np=z['vertices_mm'],z['cells10']
            a['input_cache_reads']+=1;n=row['expected_geometry']['nodes'];nc=row['expected_geometry']['cells']
            assert x_np.shape==(n,3) and cells_np.shape==(nc,10) and x_np.dtype==np.float64 and cells_np.dtype==np.int64 and np.isfinite(x_np).all()
            basis_np=[]
            for load in ('ver','hor','tor'):
                item=row['canonical_basis_manifest_identities'][load];assert d.identity(item['path'],hashed=True)==item
                v=read_field(d.read(item['path'])['physical_U']);assert v.shape==(n,3) and v.dtype==np.float64
                basis_np.append(v);a['canonical_basis_array_reads']+=1
            basis_np=np.stack(basis_np);before={k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in [('vertices',x_np),('cells',cells_np),('basis',basis_np)]}
            x=torch.as_tensor(x_np,device='cuda');cells=torch.as_tensor(cells_np,device='cuda');basis=torch.as_tensor(basis_np,device='cuda')
            B=torch.tensor([row['reference_load6'][load]for load in ('ver','hor','tor')],dtype=torch.float64,device='cuda').T.contiguous()
            xyz=torch.as_tensor(read_field(row['query_input_xyz_identity']),device='cuda');a['query_input_array_reads']+=1;assert xyz.shape==(128,3)
            api=PhysicalUQuery(torch,x,cells,basis,B,guard)
            rec=dict(API_fullN_constructor=api.metadata,inherited_actual_basis_cost=row['inherited_actual_basis_cost'],
                     prior_explicitcell_query_and_check_seconds=row['prior_explicitcell_768query_and_check_seconds'],cache_seconds=elapsed(ct))
            a['case_records'][cad]=rec;a['phase']='fullCAD_Bezier_index';persist();it=(time.time(),time.monotonic())
            locator=PhysicalULocator(torch,x,cells,guard);torch.cuda.synchronize();rec.update(index=locator.metadata,index_build_seconds=elapsed(it))
            st=(time.time(),time.monotonic())
            np.savez(folder/'locator_index_cache.npz',owners=locator.owners.cpu().numpy(),ptr=locator.ptr.cpu().numpy(),
                     low=locator.low.cpu().numpy(),high=locator.high.cpu().numpy(),
                     domain_low=locator.domain_low.cpu().numpy(),domain_high=locator.domain_high.cpu().numpy())
            rec.update(index_cache_identity=d.identity(folder/'locator_index_cache.npz',hashed=True),index_persistence_seconds=elapsed(st))
            persist()
            # Six axial and eight corner points, plus two farther axial points.
            center=(locator.domain_low+locator.domain_high)/2;extent=locator.extent
            directions=torch.cat((torch.eye(3,device='cuda'),-torch.eye(3,device='cuda'))).to(torch.float64)
            corners=torch.cartesian_prod(*(torch.tensor([-1.,1.],device='cuda',dtype=torch.float64)for _ in range(3)))
            outside=torch.cat((center+2*extent*directions,center+2*extent*corners,
                               center[None]+3*extent*directions[:2]))
            assert outside.shape==(16,3) and bool((((outside<locator.domain_low)|(outside>locator.domain_high)).any(1)).all())
            a['phase']='actual_coordinate_only_location';persist();qt=(time.time(),time.monotonic())
            ids,decisions=locator.locate(xyz);outside_ids,outside_decisions=locator.locate(outside)
            torch.cuda.synchronize();rec['location_seconds']=elapsed(qt)
            d.write(folder/'location_decisions.json',dict(interior=decisions,outside=outside_decisions))
            if not all(v['status']=='strict_interior' for v in decisions):raise LocatorNegative('fixed positive owner missing/ambiguous/noninterior')
            if not bool((outside_ids==-1).all()) or not all(v['status']=='outside_global_box' for v in outside_decisions):raise LocatorNegative('outside-box rejection failed')
            qt=(time.time(),time.monotonic());outputs=[]
            for c in coeffs:outputs.append(api.query_coefficients(xyz,ids,c.expand(128,-1)))
            out=torch.stack(outputs);assert bool(torch.isfinite(out).all())
            chunk=torch.cat([api.query_coefficients(xyz[i:i+32],ids[i:i+32],coeffs[3].expand(32,-1))for i in range(0,128,32)])
            parity=metric(chunk.cpu().numpy(),out[3].cpu().numpy());assert parity['passed']
            torch.cuda.synchronize();rec['six_load_query_and_chunk_seconds']=elapsed(qt)
            after=dict(vertices=checksum(x),cells=checksum(cells),basis=checksum(basis));assert after==before
            inv=dict(CAD=cad,physical_U=save_array(folder/'queries_U_mm.npy',out),located_cell_rows=save_array(folder/'located_cell_rows.npy',ids),
                     outside_decisions=save_array(folder/'outside_cell_rows.npy',outside_ids),
                     decisions_identity=d.identity(folder/'location_decisions.json',hashed=True),coefficient_order=coeffs.tolist(),
                     complete_field_queries=768,positive_points=128,outside_points=16,GT_reads=0,
                     held_reference_bodies_read_before_submission=0,chunk_parity=parity,owned_arrays_unchanged=before==after)
            d.write(folder/'manifest.json',inv);a['prediction_inventory'][cad]=d.identity(folder/'manifest.json',hashed=True)
            a['committed_field_queries']+=768;a['committed_located_IDs']+=128;a['committed_outside_decisions']+=16
            rec.update(whole_case_seconds=elapsed(ct),maximum_bucket_candidates=max(v['bucket_candidates']for v in decisions),
                       maximum_inverse_candidates=max(v['inverse_candidates']for v in decisions),chunk_parity=parity,owned_arrays_unchanged=True)
            persist();del api,locator,x,cells,basis,B,xyz,ids,out,outside,outside_ids,outputs,chunk,x_np,cells_np,basis_np,v
            gc.collect();torch.cuda.empty_cache()
        assert a['committed_field_queries']==2304 and a['committed_located_IDs']==384 and a['committed_outside_decisions']==48
        assert a['held_numeric_reference_array_reads']==0 and a['input_cache_reads']==3 and a['canonical_basis_array_reads']==9
        a['phase']='complete_prediction_hash_body_barrier';persist()
        assert set(a['prediction_inventory'])==set(s['selection']['CADs'])
        predicted={}
        for cad,item in a['prediction_inventory'].items():
            assert d.identity(item['path'],hashed=True)==item;inv=d.read(item['path'])
            assert inv['CAD']==cad and inv['complete_field_queries']==768 and inv['held_reference_bodies_read_before_submission']==0
            assert d.identity(inv['decisions_identity']['path'],hashed=True)==inv['decisions_identity']
            u=read_field(inv['physical_U']);ids=read_field(inv['located_cell_rows']);outside=read_field(inv['outside_decisions'])
            assert u.shape==(6,128,3) and ids.shape==(128,) and outside.shape==(16,) and (outside==-1).all()
            predicted[cad]=(u,ids)
        d.write(o/'prediction_barrier.json',dict(status='completed',all_finite_hash_body_Cartesian=True,
                denominator=a['fixed_denominators'],held_numeric_reference_reads_at_barrier=0,inventory=a['prediction_inventory']))
        a['prediction_barrier_complete']=True;a['phase']='held_numeric_samefield_reference';persist()
        for cad,(u,ids)in predicted.items():
            row=s['cases'][cad]
            ref=read_field(row['held_numeric_reference_U_identity']);a['held_numeric_reference_array_reads']+=1
            refids=read_field(row['held_numeric_reference_cell_rows_identity']);a['held_numeric_reference_array_reads']+=1
            m=metric(u,ref);match=bool((ids==refids).all())
            a['case_records'][cad].update(parent_field_parity=m,exact_original_owner_match=match,passed=m['passed'] and match)
        assert a['held_numeric_reference_array_reads']==6
        a.update(status='completed',phase='terminal',locator_gate_passed=all(v['passed']for v in a['case_records'].values()),
                 scientific_outcome='fullCAD_coordinate_locator_PASS' if all(v['passed']for v in a['case_records'].values())else 'fixed_locator_numeric_negative')
    except LocatorNegative as error:
        a.update(status='completed',phase='terminal_scientific_negative',scientific_outcome='fixed_locator_negative',reason=str(error))
    except Exception as error:
        a.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error));raise
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
