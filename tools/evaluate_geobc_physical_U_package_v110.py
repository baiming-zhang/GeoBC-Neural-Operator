"""Actual reviewed package single-load and complete coordinate/force6 queries."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,json,os,signal,hashlib,gc
from pathlib import Path

def run(path):
    from geobc import available_cad_pilot_data as d
    from geobc.physical_U_response_package_contract_v110 import approved_contract,descriptors
    from geobc.physical_U_global_locator_v106 import LocatorNegative
    p,r=approved_contract(path);s=p['scientific'];o=d.path(p['output']);outer=json.loads(os.environ['GEOBC_PHYSICAL_U_PACKAGE_V110_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest'] and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=p['scientific_digest'],boot_id=outer['boot_id']);d.write(o/'execution.json',execution)
    a=dict(status='running',phase='cache',active=True,closed=False,execution=execution,input_cache_reads=0,basis_array_reads=0,owned_array_reads=0,
        direct_reference_API_calls=0,direct_reference_calls_before_barrier=0,GT_reads=0,K_PCG_replays=0,raw_H5_deck_GRID_reads=0,
        neural_PT_reads=0,neural_inference=0,training_updates=0,scaler_fits=0,FINAL_reads=0,case_records={},prediction_inventory={},
        committed_U_queries=0,committed_positive_owner_points=0,committed_outside_points=0,metadata_tamper_rejections=0,metadata_tamper_fixtures=[],
        actual_coupled_gradient_points=0,loader_calls=0,peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0,
        prediction_barrier_complete=False,package_gate_passed=False,new_offnode_GT_accuracy=False,signed_S_solved=False,final_solver_selected=False)
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
        # All tampered payloads reject on the reviewed digest before imports,
        # body reads or scientific constructors; their self-digests are updated
        # deliberately so rejecting only a stale self-digest cannot pass.
        import copy
        from geobc.physical_U_response_package_v110 import validate_metadata,load_response
        packages=descriptors(p)['packages'];expected=s['implementation']['reviewed_payload_digests']
        for cad,package in packages.items():
            for kind in s['implementation']['tamper_fixtures']:
                changed=copy.deepcopy(package)
                if kind=='physical_U_unit':changed['payload']['scope']['physical_U_units']='m'
                elif kind=='field_identity':changed['payload']['basis_records']['ver']['physical_U']['sha256']='0'*64
                elif kind=='basis_order':changed['payload']['basis_order'][0:2]=['hor','ver']
                else:raise AssertionError('Unknown fixed tamper fixture')
                changed['payload_digest']=d.digest(changed['payload'])
                try:validate_metadata(d,changed,expected[cad])
                except ValueError as error:
                    assert str(error)=='Unreviewed or changed response package metadata'
                    a['metadata_tamper_rejections']+=1
                    a['metadata_tamper_fixtures'].append(dict(CAD=cad,fixture=kind,rejected=True,body_reads=0,constructor_calls=0))
                else:raise AssertionError('Tampered metadata admitted')
        assert a['metadata_tamper_rejections']==9 and a['loader_calls']==0
        persist()
        import numpy as np
        import torch as tm
        torch=tm
        from geobc import elastic_pilot as old
        from geobc.physical_U_fullrank6_v108 import FullRank6PhysicalUQuery
        from geobc.physical_U_global_locator_v106 import PhysicalULocator
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
            # A scoped profiler observes unchanged accepted constructors. It
            # does not replace modules, change their math, or manually assemble
            # response objects; the reviewed factory remains the only loader.
            import sys
            timings={};starts={}
            targets={FullRank6PhysicalUQuery.__init__.__code__:'API_constructor_seconds',PhysicalULocator.__init__.__code__:'index_constructor_seconds'}
            def profile(frame,event,arg):
                name=targets.get(frame.f_code)
                if name and event=='call':
                    torch.cuda.synchronize();starts[name]=(time.time(),time.monotonic())
                elif name and event=='return':
                    torch.cuda.synchronize();timings[name]=elapsed(starts.pop(name))
            assert sys.getprofile() is None
            lt=(time.time(),time.monotonic());sys.setprofile(profile)
            try:facade=load_response(d,np,torch,packages[cad],expected[cad],guard)
            finally:sys.setprofile(None)
            torch.cuda.synchronize();loader_seconds=elapsed(lt);a['loader_calls']+=1
            counts=facade.package_metadata['body_read_counts']
            assert counts['geometry_body_reads']==1 and counts['basis_body_reads']==6
            assert all(counts[k]==0 for k in ('GT_reads','raw_H5_deck_GRID_reads','K_PCG_replays','neural_PT_reads'))
            a['input_cache_reads']+=counts['geometry_body_reads'];a['basis_array_reads']+=counts['basis_body_reads']
            api,locator=facade.api,facade.locator
            x,cells,basis,B=api.vertices,api.cells,api.basis,api.B
            n,nc=row['expected_geometry']['nodes'],row['expected_geometry']['cells']
            assert api.metadata['actual_allN']==n and api.metadata['actual_cells']==nc and api.metadata['physical_load_rank']==6
            before=dict(vertices=checksum(x),cells=checksum(cells),basis=checksum(basis))
            rec=dict(actual_fullN=api.metadata,index_metadata=locator.metadata,interface_metadata=facade.metadata,
                package_metadata=facade.package_metadata,loader_total_seconds=loader_seconds,**timings,
                metadata_hash_IO_staging_and_other_loader_seconds=max(0,loader_seconds-sum(timings.values())),
                inherited_actual_basis_cost=row['inherited_actual_basis_cost'],missing_basis_cost_seconds=row['missing_basis_cost_seconds'],
                independent_validation_cost_seconds=row['independent_validation_cost_seconds'])
            a['case_records'][cad]=rec
            # Fixture row IDs generate predictor-input coordinates only. No IDs enter facade.
            fixture=torch.linspace(0,nc-1,4,device='cuda').to(torch.int64);assert fixture.unique().numel()==4
            bary=torch.tensor(s['evaluation']['new_geometry_query_bary'][:3],dtype=torch.float64,device='cuda').expand(4,-1);N,_,_=shape(torch,bary);xyz=torch.einsum('qi,qia->qa',N,x[cells[fixture]])
            center=(locator.domain_low+locator.domain_high)/2;extent=locator.extent
            outside=center[None]+2*extent*torch.tensor([[1.,0.,0.]],dtype=torch.float64,device='cuda');assert outside.shape==(1,3)
            a['phase']='coordinate_force6_prediction_no_IDs';persist();qt=(time.time(),time.monotonic());outputs=[];owner=None;location_records=None
            for c in coeff:
                f=(c@B.T).expand(4,-1);result=facade.query(xyz,f)
                if result['status']!='completed' or not all(v['status']=='strict_interior' for v in result['location_records']):raise LocatorNegative('New fixedcoupled positive point not strictowner')
                if owner is not None:assert torch.equal(owner,result['located_cell_rows'])
                owner=result['located_cell_rows'];location_records=result['location_records'];outputs.append(result['physical_U'])
            out=torch.stack(outputs);negative=facade.query(outside,(coeff[0]@B.T).expand(1,-1))
            if negative['status']!='rejected' or negative['physical_U']is not None or not bool((negative['located_cell_rows']==-1).all())or not all(v['status']=='outside_global_box'for v in negative['location_records']):raise LocatorNegative('Outsidebatch rejection violated')
            chunk=torch.cat([facade.query(xyz[j:j+2],(coeff[0]@B.T).expand(2,-1))['physical_U']for j in (0,2)])
            parity=metric(chunk,out[0]);assert parity['passed'];torch.cuda.synchronize();rec['actual_query_reject_chunk_seconds']=elapsed(qt)
            d.write(folder/'location_decisions.json',dict(positive=location_records,outside=negative['location_records'],wholebatch_rejection=True))
            inv=dict(CAD=cad,U=save_array(folder/'U_mm.npy',out),located_IDs=save_array(folder/'located_cell_rows.npy',owner),outside_IDs=save_array(folder/'outside_cell_rows.npy',negative['located_cell_rows']),decisions_identity=d.identity(folder/'location_decisions.json',hashed=True),query_count=12,positive_points=4,outside_points=1,direct_reference_calls_before_barrier=0,caller_element_IDs_supplied=False,chunk_parity=parity)
            d.write(folder/'manifest.json',inv);a['prediction_inventory'][cad]=d.identity(folder/'manifest.json',hashed=True);a['committed_U_queries']+=12;a['committed_positive_owner_points']+=4;a['committed_outside_points']+=1
            rec.update(whole_prediction_case_seconds=elapsed(ct),chunk_parity=parity)
            # Resident samefield data stays in memory; no repeated body loads for references.
            resident[cad]=dict(api=api,locator=locator,facade=facade,x=x,cells=cells,basis=basis,B=B,xyz=xyz,fixture=fixture,out=out,owner=owner,before=before)
            persist();del outputs,chunk
        assert a['input_cache_reads']==3 and a['basis_array_reads']==18 and a['direct_reference_API_calls']==0 and a['committed_U_queries']==36 and a['committed_positive_owner_points']==12 and a['committed_outside_points']==3
        assert a['loader_calls']==3 and a['metadata_tamper_rejections']==9
        a['phase']='complete36_12_3_hash_body_barrier';persist();assert set(a['prediction_inventory'])==set(s['selection']['CADs'])
        for cad,item in a['prediction_inventory'].items():
            assert d.identity(item['path'],hashed=True)==item;inv=d.read(item['path']);assert inv['CAD']==cad and inv['query_count']==12 and not inv['caller_element_IDs_supplied'] and inv['direct_reference_calls_before_barrier']==0
            uv=read_array(inv['U']);iv=read_array(inv['located_IDs']);ov=read_array(inv['outside_IDs']);assert uv.shape==(3,4,3) and iv.shape==(4,) and ov.shape==(1,) and (ov==-1).all()
            assert d.identity(inv['decisions_identity']['path'],hashed=True)==inv['decisions_identity']
        d.write(o/'prediction_barrier.json',dict(status='completed',complete_U_queries=36,positive_owner_points=12,outside_points=3,metadata_tamper_rejections=9,all_finite_hash_body_Cartesian=True,direct_reference_calls_before_barrier=0,GT_reads=0,inventory=a['prediction_inventory']));a['prediction_barrier_complete']=True;persist()
        for cad,cache in resident.items():
            a.update(current_CAD=cad,phase='postbarrier_package_samefield_no_new_gradtests');persist();api,facade=cache['api'],cache['facade'];xyz,ids,B=cache['xyz'],cache['fixture'],cache['B'];out=cache['out'];rec=a['case_records'][cad]
            qm={}
            for j,c in enumerate(coeff):
                direct=api.query(xyz,ids,(c@B.T).expand(4,-1));a['direct_reference_API_calls']+=1;qm[str(j)]=metric(out[j],direct)
            owner_match=bool(torch.equal(cache['owner'],ids))
            after=dict(vertices=checksum(cache['x']),cells=checksum(cache['cells']),basis=checksum(cache['basis']));assert after==cache['before']
            rec.update(samefield_metrics=qm,exact_fixture_owner_match=owner_match,inherited_coupled_gradient_evidence=s['prior_terminal'],input_basis_bytes_unchanged=True,passed=owner_match and all(v['passed']for v in qm.values()));persist()
        assert a['actual_coupled_gradient_points']==0 and a['direct_reference_API_calls']==9
        a.update(status='completed',phase='terminal',package_gate_passed=all(v['passed']for v in a['case_records'].values()),scientific_outcome='strict_response_package_load_PASS'if all(v['passed']for v in a['case_records'].values())else'fixed_package_negative')
    except LocatorNegative as error:
        a.update(status='completed',phase='terminal_scientific_negative',reason=str(error),scientific_outcome='fixed_package_locator_negative')
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
