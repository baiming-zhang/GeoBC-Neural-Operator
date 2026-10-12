"""Two cache-miss source geometries to six independent bases and loaded response."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,json,os,signal,gc
from pathlib import Path

def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.physical_U_new_geometry_contract_v111 import approved_contract
    p,r=approved_contract(plan_path);s=p['scientific'];o=d.path(p['output']);outer=json.loads(os.environ['GEOBC_PHYSICAL_U_NEW_GEOMETRY_V111_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest'] and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=p['scientific_digest'],boot_id=outer['boot_id']);d.write(o/'execution.json',execution)
    a=dict(status='running',active=True,closed=False,phase='source',execution=execution,source_body_hash_reads=0,H5_input_attempts=0,H5_input_reads=0,H5_members_read=[],GRID_map_reads=0,deck_parses=0,deck_header_reads=0,owned_array_reads=0,GT_attempts=0,GT_reads=0,raw_CSV_H5_nodal_target_reads=0,neural_PT_reads=0,neural_inference=0,training_updates=0,scaler_fits=0,FINAL_reads=0,prediction_inventory={},fixed_endpoint_denominator=18,committed_finite_endpoints=0,combined_solve_seconds=0.,case_records={},completed_groups={},peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0,protocol_qualified=False,field_barrier_complete=False,numeric_gate_passed=False,API_barrier_complete=False,
        deployment_gate_passed=False,loader_calls=0,loader_geometry_body_reads=0,loader_basis_body_reads=0,
        committed_U_queries=0,committed_owner_points=0,committed_outside_points=0,API_inventory={},
        direct_reference_calls_before_API_barrier=0,direct_reference_API_calls=0,final_solver_selected=False,signed_S_solved=False,
        new_OptiStruct_GT_accuracy=False,original_reserve_canonical_GT_accuracy_measured=False)
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
        from geobc.physical_U_new_geometry_rhs_v111 import build_reference_rhs
        from geobc.physical_U_response_package_v110 import load_response
        from geobc.physical_U_query_v104 import shape
        from geobc.physical_U_global_locator_v106 import LocatorNegative
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        total=torch.cuda.get_device_properties(0).total_memory;assert 30500000000<total;torch.cuda.set_per_process_memory_fraction(30500000000/total,0);torch.cuda.reset_peak_memory_stats()
        for cad in s['selection']['CADs']:
            guard();folder=o/cad;folder.mkdir(exist_ok=False);ledger=None;a['current_CAD']=cad;a['phase']='targetfree_source';persist();tick=(time.time(),time.monotonic())
            cs=dict(s['cases'][cad],selection=dict(CAD=cad,mixed_coefficients=s['implementation']['canonical_prepare_inputs_mix3']),old_accepted_physics_recipe=s['physics_recipe'])
            arrays,forces_np,record=prepare_inputs(np,cs,a,guard,tick[1]);guard()
            cache=folder/'geometry_and_rhs_input.npz'
            with cache.open('xb')as f:np.savez(f,**arrays,all_six_forces=forces_np)
            input_identity=d.identity(cache,hashed=True);inputs={k:torch.as_tensor(v,device='cuda')for k,v in arrays.items()};forces=torch.as_tensor(forces_np,device='cuda');del arrays,forces_np
            sourcefacts=a['physics_input_facts'];loadguard=sourcefacts['original']['load_guard']
            if loadguard['REFC']!='123456' or loadguard['normal_rank']!=6:raise olddata.UnsupportedInput('Fresh source does not support six reference DOFs')
            columns=[sourcefacts['all_three_source_LOAD_expansion'][name]['reference_load6']for name in ('ver','hor','tor')]
            B,basis_rhs,sixqual=build_reference_rhs(torch,inputs,forces[:3],columns,s['selection']['added_Fy_N'],s['selection']['added_Mx_My_Nmm'],guard,s['physics_recipe']['load_transfer']['relative_guard_tolerance'])
            coefficients=torch.tensor(s['selection']['mixed_coefficients'],dtype=torch.float64,device='cuda')
            mixed_rhs=torch.einsum('mk,kna->mna',coefficients,basis_rhs)
            with(folder/'reference6_rhs.npz').open('xb')as f:np.savez(f,force6_columns=B.cpu().numpy(),basis_nodal_rhs=basis_rhs.cpu().numpy(),mixed_nodal_rhs=mixed_rhs.cpu().numpy())
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
            a['case_records'][cad]=dict(expected_geometry=cs['expected_geometry'],input_identity=input_identity,source_operator_wall_seconds=source_wall,source_facts=a['physics_input_facts'],geometry=geometry,algebra_guards=algebra,parallel_K_qualification=kqual,operator_cache=saved,six_transfer_qualification=sixqual,reference_load6_columns=B.T.cpu().tolist(),rhs_cache_identity=d.identity(folder/'reference6_rhs.npz',hashed=True),groups={});persist()
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
            for names,role,rhs in [(['ver','hor','tor'],'canonical_basis',basis_rhs[:3]),(['Fy','Mx','My'],'added_basis',basis_rhs[3:]),(['mix6_0','mix6_1','mix6_2'],'independent_mixed',mixed_rhs)]:
                guard();a['phase']=cad+'/'+role;persist();began=(time.time(),time.monotonic())
                def remaining():return 1200-a['combined_solve_seconds']-elapsed(began)
                inv,reported=solve_coarse(torch,np,batch,chol,inputs,rhs,names,folder/role,a,guard,persist,remaining,parallel.apply)
                wall=elapsed(began);assert wall<=300;a['combined_solve_seconds']+=wall
                data=dict(wall_seconds=wall,solver_reported_seconds=reported,initial_guess='independent_zero_all3')
                a['case_records'][cad]['groups'][role]=data;a['completed_groups'][cad+'/'+role]=data
                a['prediction_inventory'].update({cad+'/'+role+'/'+name:item for name,item in inv.items()});a['committed_finite_endpoints']+=3
                if role=='added_basis':a['case_records'][cad]['source_to_six_basis_ready_seconds']=elapsed(tick)
                persist()
            case=a['case_records'][cad]
            case.update(whole_source_numeric_case_seconds=elapsed(tick),operator_calls=dict(scalar=op.calls,batch=batch.calls,coarse_parallel=parallel.calls),coarse_ledger=dict(ledger.data))
            d.write(folder/'case_numeric_terminal.json',case);persist()
            del inputs,forces,basis_rhs,mixed_rhs,rhs,B,coefficients,op,chol,batch,ledger,oa,P,PT,Ac,cd,corners,cmap,fmap,scalar,parallel;ledger=None;gc.collect();torch.cuda.empty_cache()
        assert a['combined_solve_seconds']<=1200 and a['committed_finite_endpoints']==18
        a['phase']='all18_fresh_field_barrier_GT_closed';persist();rt=(time.time(),time.monotonic());fields={};records={}
        def read_array(item):
            guard();expected={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')};expected['path']=d.relative(d.path(item['path']));assert d.identity(item['path'],hashed=True)==expected
            value=np.load(d.path(item['path']),allow_pickle=False);a['owned_array_reads']+=1
            dtype={'torch.float64':'float64','torch.float32':'float32','torch.int64':'int64'}.get(item['dtype'],item['dtype'])
            assert list(value.shape)==item['shape'] and str(value.dtype)==dtype and np.isfinite(value).all();return value
        def metric(pred,ref,limit):
            norm=float(np.linalg.norm(ref));error=float(np.linalg.norm(pred-ref));absolute=float(np.abs(pred-ref).max());relative=error/norm if norm>0 else None
            return dict(relative=relative,maximum_absolute_mm=absolute,passed=relative<=limit if relative is not None else absolute<=1e-12)
        expected={cad+'/'+role+'/'+name for cad in s['selection']['CADs']for role,names in [('canonical_basis',['ver','hor','tor']),('added_basis',['Fy','Mx','My']),('independent_mixed',['mix6_0','mix6_1','mix6_2'])]for name in names}
        assert set(a['prediction_inventory'])==expected
        for key,item in a['prediction_inventory'].items():
            assert d.identity(item['path'],hashed=True)==item;rec=d.read(item['path']);cad=key.split('/')[0];n=s['cases'][cad]['expected_geometry']['nodes']
            assert rec['status']=='completed' and rec['initial_guess']=='independent_zero' and rec['key']==key.split('/')[-1] and rec['all_original_nodes']==n
            value=read_array(rec['physical_U']);assert value.shape==(n,3) and value.dtype==np.float64;fields[key]=value;records[key]=rec
        d.write(o/'physics_prediction_barrier.json',dict(status='completed',all18_finite_immutable_original_nodes=True,GT_reads=0,inventory=a['prediction_inventory']))
        a['field_barrier_complete']=True;a['endpoint_records']=records
        numerical={};packages={}
        for cad in s['selection']['CADs']:
            case=a['case_records'][cad];basis=[];basis_records={}
            for name in s['selection']['basis_names']:
                role='canonical_basis'if name in ('ver','hor','tor')else'added_basis';key=cad+'/'+role+'/'+name
                basis.append(fields[key]);basis_records[name]=dict(manifest_identity=a['prediction_inventory'][key],physical_U=records[key]['physical_U'])
            for i,c in enumerate(s['selection']['mixed_coefficients']):
                numerical[cad+'/mix6_'+str(i)]=metric(sum(c[j]*basis[j]for j in range(6)),fields[cad+'/independent_mixed/mix6_'+str(i)],1e-5)
            load=case['source_facts']['original']['load_guard']
            payload=dict(kind='independent_approximate_P2_physical_U_force6',CAD=cad,basis_order=s['selection']['basis_names'],
                geometry_identity=case['input_identity'],geometry_schema=s['cases'][cad]['expected_geometry'],basis_records=basis_records,
                reference_load6_columns=case['reference_load6_columns'],scope=dict(REFC=load['REFC'],reference_GRID_ID=load['reference_GRID_ID'],material_constraints_fixed=True,physical_U_units='mm',force_units='N',moment_units='Nmm'),
                source_provenance=dict(SCI=p['scientific_digest'],execution=execution,source_input_identities=s['cases'][cad]['source_input_identities'],GRID_map_identity=s['cases'][cad]['GRID_map_identity'],actual_source_facts=case['source_facts'],recipe='curvedP2_fourQ_weightedLS_RBE3;originalDEBUG_PSOLID_unknown'),
                inherited_cost=dict(source_operator_seconds=case['source_operator_wall_seconds'],coarse_setup_seconds=case['coarse_setup_seconds'],source_to_six_basis_ready_seconds=case['source_to_six_basis_ready_seconds'],groups=case['groups'],fresh_engineering_fullcold_seconds=None),
                new_OptiStruct_GT_accuracy=False,signedS_provided=False)
            packages[cad]=dict(schema='GeoBC_physical_U_response_package_v1',payload=payload,payload_digest=d.digest(payload))
        allconv=all(v['numerical_state']=='converged' and v['true_relative_residual']<=1e-8 for v in records.values())
        a['numeric_same_equation_metrics']=numerical;a['numeric_gate_passed']=allconv and all(v['passed']for v in numerical.values())
        a['field_barrier_and_numeric_readout_seconds']=elapsed(rt);assert a['field_barrier_and_numeric_readout_seconds']<=180
        # Actual root review is independent of the descriptor producer. Runtime
        # waits with the original clocks still running, never approving itself.
        if not a['numeric_gate_passed']:
            a.update(status='completed',phase='terminal_scientific_negative',scientific_outcome='fixed_new_geometry_numeric_negative');persist();return
        proposal=o/'package_proposal.json'
        d.write(proposal,dict(SCI=p['scientific_digest'],source_pins=p['source_pins'],packages=packages,field_barrier_identity=d.identity(o/'physics_prediction_barrier.json',hashed=True)))
        a.update(phase='awaiting_independent_package_root_release',package_proposal_identity=d.identity(proposal,hashed=True));persist()
        wait=(time.time(),time.monotonic());release_path=o/'package_release.json'
        while not release_path.exists():guard();time.sleep(.5)
        guard();release=d.read(release_path)
        assert release['passed'] and release['reviewer']=='root' and release['SCI']==p['scientific_digest'] and release['source_pins']==p['source_pins']
        assert release['proposal_identity']==d.identity(proposal,hashed=True) and release['field_barrier_identity']==d.identity(o/'physics_prediction_barrier.json',hashed=True)
        a['independent_root_wait_seconds']=elapsed(wait);a['actual_package_root_release_identity']=d.identity(release_path,hashed=True)
        a['phase']='actual_clean_load_new_own_packages';persist();api_start=(time.time(),time.monotonic());resident={}
        coeff=torch.tensor(s['selection']['mixed_coefficients'],dtype=torch.float64,device='cuda')
        def save_array(file,value):
            guard();item=old._save_array(np,file,value);item.update(d.identity(item['path'],hashed=True));return item
        def checksum(value):
            import hashlib
            return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for cad in s['selection']['CADs']:
            a['current_CAD']=cad;persist();lt=(time.time(),time.monotonic())
            response=load_response(d,np,torch,packages[cad],release['reviewed_payload_digests'][cad],guard);torch.cuda.synchronize();a['loader_calls']+=1
            counts=response.package_metadata['body_read_counts'];assert counts['geometry_body_reads']==1 and counts['basis_body_reads']==6
            assert all(counts[k]==0 for k in ('GT_reads','raw_H5_deck_GRID_reads','K_PCG_replays','neural_PT_reads'))
            a['loader_geometry_body_reads']+=1;a['loader_basis_body_reads']+=6
            api,locator=response.api,response.locator;x,cells,B=api.vertices,api.cells,api.B
            n,nc=s['cases'][cad]['expected_geometry']['nodes'],s['cases'][cad]['expected_geometry']['cells']
            assert api.metadata['actual_allN']==n and api.metadata['actual_cells']==nc and api.metadata['physical_load_rank']==6
            rec=a['case_records'][cad];rec.update(loader_seconds=elapsed(lt),package_metadata=response.package_metadata,actual_API_metadata=api.metadata,actual_fullCAD_locator=locator.metadata)
            before=dict(vertices=checksum(x),cells=checksum(cells),basis=checksum(api.basis))
            fixture=torch.linspace(0,nc-1,4,device='cuda').to(torch.int64);assert fixture.unique().numel()==4
            bary=torch.tensor([.20,.30,.25],dtype=torch.float64,device='cuda').expand(4,-1);N,_,_=shape(torch,bary);xyz=torch.einsum('qi,qia->qa',N,x[cells[fixture]])
            outside=(locator.domain_low+locator.domain_high)[None]/2+2*locator.extent*torch.tensor([[1.,0.,0.]],dtype=torch.float64,device='cuda')
            qt=(time.time(),time.monotonic());outputs=[];owners=None
            for c in coeff:
                value=response.query(xyz,(c@B.T).expand(4,-1))
                if value['status']!='completed' or not all(v['status']=='strict_interior'for v in value['location_records']):raise LocatorNegative('Fresh fixed query has no unambiguous interior owner')
                if owners is not None:assert torch.equal(owners,value['located_cell_rows'])
                owners=value['located_cell_rows'];outputs.append(value['physical_U'])
            out=torch.stack(outputs);negative=response.query(outside,(coeff[0]@B.T).expand(1,-1))
            if negative['status']!='rejected' or negative['physical_U']is not None or not all(v['status']=='outside_global_box'for v in negative['location_records']):raise LocatorNegative('Fresh outside wholebatch not rejected')
            assert int(negative['located_cell_rows'][0])==-1
            chunk=torch.cat([response.query(xyz[j:j+2],(coeff[0]@B.T).expand(2,-1))['physical_U']for j in (0,2)])
            parity=metric(chunk.detach().cpu().numpy(),out[0].detach().cpu().numpy(),1e-10)
            torch.cuda.synchronize();rec['query_reject_chunk_seconds']=elapsed(qt);rec['chunk_parity']=parity
            folder=o/cad/'response_queries';folder.mkdir(exist_ok=False)
            d.write(folder/'decisions.json',dict(positive=value['location_records'],outside=negative['location_records'],outside_wholebatch_rejected=True,caller_element_IDs_supplied=False))
            inv=dict(CAD=cad,U=save_array(folder/'U_mm.npy',out),owners=save_array(folder/'located_IDs.npy',owners),outside=save_array(folder/'outside_IDs.npy',negative['located_cell_rows']),decisions_identity=d.identity(folder/'decisions.json',hashed=True),U_queries=12,owner_points=4,outside_points=1,direct_reference_calls_before_barrier=0)
            d.write(folder/'manifest.json',inv);a['API_inventory'][cad]=d.identity(folder/'manifest.json',hashed=True)
            a['committed_U_queries']+=12;a['committed_owner_points']+=4;a['committed_outside_points']+=1
            resident[cad]=dict(response=response,xyz=xyz,fixture=fixture,owners=owners,out=out,before=before);persist()
        assert a['loader_calls']==2 and a['loader_geometry_body_reads']==2 and a['loader_basis_body_reads']==12
        assert a['committed_U_queries']==24 and a['committed_owner_points']==8 and a['committed_outside_points']==2 and a['direct_reference_API_calls']==0
        assert set(a['API_inventory'])==set(s['selection']['CADs'])
        for cad,item in a['API_inventory'].items():
            assert d.identity(item['path'],hashed=True)==item;inv=d.read(item['path']);assert inv['CAD']==cad and inv['U_queries']==12 and inv['owner_points']==4 and inv['outside_points']==1
            uv=read_array(inv['U']);iv=read_array(inv['owners']);ov=read_array(inv['outside'])
            assert uv.shape==(3,4,3) and uv.dtype==np.float64 and iv.shape==(4,) and iv.dtype==np.int64 and ov.shape==(1,) and ov.dtype==np.int64 and (ov==-1).all()
            assert d.identity(inv['decisions_identity']['path'],hashed=True)==inv['decisions_identity']
        d.write(o/'API_prediction_barrier.json',dict(status='completed',all24_U_8owners_2outside_finite_hash_body_Cartesian=True,inventory=a['API_inventory'],GT_reads=0,direct_reference_calls_before_barrier=0));a['API_barrier_complete']=True;persist()
        for cad,cache in resident.items():
            api=cache['response'].api;qm={}
            for j,c in enumerate(coeff):
                ref=api.query(cache['xyz'],cache['fixture'],(c@api.B.T).expand(4,-1));a['direct_reference_API_calls']+=1
                qm[str(j)]=metric(cache['out'][j].detach().cpu().numpy(),ref.detach().cpu().numpy(),1e-10)
            after=dict(vertices=checksum(api.vertices),cells=checksum(api.cells),basis=checksum(api.basis));assert after==cache['before']
            row=a['case_records'][cad];owner_match=bool(torch.equal(cache['owners'],cache['fixture']))
            row.update(samefield_metrics=qm,exact_fixture_owner_match=owner_match,source_basis_bytes_unchanged=True,API_passed=owner_match and row['chunk_parity']['passed'] and all(v['passed']for v in qm.values()))
            persist()
        a['readout_package_API_seconds']=a['field_barrier_and_numeric_readout_seconds']+elapsed(api_start);assert a['readout_package_API_seconds']<=180
        assert a['direct_reference_API_calls']==6 and a['H5_input_reads']==a['GRID_map_reads']==a['deck_parses']==a['deck_header_reads']==2 and a['source_body_hash_reads']==4
        a.update(status='completed',phase='terminal',protocol_qualified=True,deployment_gate_passed=a['numeric_gate_passed']and all(v['API_passed']for v in a['case_records'].values()),
            scientific_outcome='new_geometry_source_to_response_PASS'if a['numeric_gate_passed']and all(v['API_passed']for v in a['case_records'].values())else'fixed_new_geometry_deployment_negative')
    except Exception as error:
        from geobc.elastic_pilot import ScientificNegative
        from geobc.elastic_pilot_data import UnsupportedInput
        from geobc.physical_U_global_locator_v106 import LocatorNegative
        if isinstance(error,(ScientificNegative,UnsupportedInput,LocatorNegative)):a.update(status='completed',phase='terminal_scientific_negative',reason=str(error),scientific_outcome='unsupported_input_or_operator_or_solver',protocol_qualified=False)
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
