"""Authentic support rotations: independent original-freeK PCG then response."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,json,os,signal,hashlib
from pathlib import Path

def run(path):
    from geobc import available_cad_pilot_data as d
    from geobc.all_support_rigid_motion_contract_v120 import approved_contract
    p,r=approved_contract(path);s=p['scientific'];o=d.path(p['output']);outer=json.loads(os.environ['GEOBC_ALL_SUPPORT_RIGID_MOTION_V120_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest']and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=p['scientific_digest'],boot_id=outer['boot_id']);d.write(o/'execution.json',execution)
    a=dict(status='running',phase='cache',active=True,closed=False,execution=execution,input_cache_reads=0,basis_array_reads=0,operator_array_reads=0,coarse_array_reads=0,
        owned_array_reads=0,deck_coordinate_hash_reads=0,deck_coordinate_passes=0,GT_reads=0,raw_H5_GRID_reads=0,geometry_K_coarse_rebuilds=0,inherited_eight_fine_K_guards_repeated=0,
        neural_PT_reads=0,training_updates=0,scaler_fits=0,FINAL_reads=0,case_records={},prediction_inventory={},completed_groups={},
        committed_finite_endpoints=0,fixed_endpoint_denominator=30,combined_solve_seconds=0.,prediction_barrier_complete=False,
        support_rigid_motion_gate_passed=False,new_original_support_GT_accuracy=False,signed_S_solved=False,final_solver_selected=False,
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
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External support-port guard')))
    try:
        import fcntl
        lock=d.path(s['budgets']['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch as tm
        torch=tm
        from geobc import elastic_pilot as old,elastic_coarse as coarse
        from geobc.load_linear_physics_pcg_v100 import ParallelElastic
        from geobc.sixload_coarse_v102 import Ledger,ThreeLaneCoarse
        from geobc.all_support_rigid_motion_pcg_v120 import solve_group
        from geobc.support_rigid_motion_inputs_v120 import source_reference_coordinates,validate_source_ports,make_lift
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
        basis_motions=torch.tensor(s['selection']['rotation_basis_eachport_rad'],dtype=torch.float64,device='cuda')
        mixed_motions=torch.tensor(s['selection']['independent_mixed_support_motion6'],dtype=torch.float64,device='cuda')
        coeff=torch.tensor(s['selection']['applied_force_coefficients'],dtype=torch.float64,device='cuda');resident={}
        newnames=s['implementation']['new_rotation_basis_names'];mixednames=s['implementation']['independent_mixed_names']
        for cad in s['selection']['CADs']:
            ct=(time.time(),time.monotonic());row=s['cases'][cad];folder=o/cad;folder.mkdir(exist_ok=False);a.update(current_CAD=cad,phase='strict_source_port_cache');persist()
            identity=row['input_cache_identity'];assert d.identity(identity['path'],hashed=True)==identity
            with np.load(d.path(identity['path']),allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files if k not in ('all_six_forces','force')}
            a['input_cache_reads']+=1;assert all(np.isfinite(v).all()for v in arrays.values())
            n,nc=row['expected_geometry']['nodes'],row['expected_geometry']['cells'];assert arrays['vertices_mm'].shape==(n,3)and arrays['cells10'].shape==(nc,10)
            assert arrays['vertices_mm'].dtype==np.float64 and arrays['cells10'].dtype==np.int64
            inputs={k:tensor(v)for k,v in arrays.items()};del arrays
            H,ids,L=inputs['rbe3_Hbar'],inputs['rbe3_member_node_ids'],float(inputs['rbe3_length_mm']);free=inputs['free_node_ids']
            assert inputs['derived_node_ids'].numel()==0 and L==row['source_transfer_facts']['length_mm']
            D=torch.tensor([1.,1.,1.,L,L,L],dtype=torch.float64,device='cuda');B=torch.tensor(row['reference_load6_columns'],dtype=torch.float64,device='cuda').T.contiguous()
            reference_np,source_facts=source_reference_coordinates(np,row,s['implementation']['source_reader_scope'],a,guard)
            reference_xyz=tensor(reference_np);ports,others,port_facts=validate_source_ports(torch,inputs,row,source_facts)
            d.write(folder/'authentic_source_support_coordinates.json',source_facts)
            source_public_facts=dict(source_facts,actual_source_RBE2_records=[dict(
                reference_GRID_ID=v['reference_GRID_ID'],CM=v['CM'],source_line=v['source_line'],
                actual_member_count=len(v['member_GRID_IDs']),actual_member_GRID_ID_digest=d.digest(v['member_GRID_IDs']))
                for v in source_facts['actual_source_RBE2_records']],
                full_source_facts_local_identity=d.identity(folder/'authentic_source_support_coordinates.json',hashed=True))
            qualification_lift=make_lift(torch,inputs,ports,others,reference_xyz,mixed_motions)
            assert bool(torch.isfinite(qualification_lift).all());del qualification_lift
            op_arrays={k:tensor(read_array(item,'operator_array_reads'))for k,item in row['operator_cache'].items()}
            assert op_arrays['gradients'].shape==(nc,4,10,3)and op_arrays['weights'].shape==(nc,4)and op_arrays['free_cholesky'].shape==(len(free),3,3)
            assert all(v.dtype==torch.float64 for v in op_arrays.values());before={k:checksum(v)for k,v in op_arrays.items()}
            lam,mu=map(float,inputs['lame'].tolist());fine=old.MatrixFreeElastic(torch,inputs['cells10'],op_arrays['gradients'],op_arrays['weights'],n,lam,mu)
            batch=ParallelElastic(torch,fine,guard)
            coarse_meta=d.read(row['coarse_cache_identity']['path']);cm=coarse_meta['arrays'];Pc,PT,Ac=[load_sparse(cm[name])for name in ('P','PT','Ac')]
            diagonal=tensor(read_array(cm['diagonal'],'coarse_array_reads'));chol=op_arrays['free_cholesky']
            assert diagonal.dtype==torch.float64 and bool((diagonal>0).all())and Pc.shape[0]==PT.shape[1]==len(free)
            ledger=Ledger(torch,guard,persist,folder,ct);ledger.setup_active=False
            scalarM=coarse.AdditiveCoarse(torch,Pc,PT,Ac,diagonal,coarse_meta['assembly']['symmetric_scaled_Gershgorin_Gamma'],chol,ledger)
            M=ThreeLaneCoarse(torch,scalarM,guard)
            forces6=coeff@B.T;forces=torch.zeros((3,n,3),dtype=torch.float64,device='cuda')
            forces.index_add_(1,ids,((forces6/D)@H).reshape(3,-1,3))
            torch.cuda.synchronize();cache_seconds=elapsed(ct);assert cache_seconds<=120
            rec=dict(actual_allN=n,actual_cells=nc,source_RBE3_REF_GRID=row['source_transfer_facts']['reference_GRID_ID'],
                source_support_ports=port_facts,cache_constructor_port_qualification_seconds=cache_seconds,
                inherited_source_to_six_force_basis_seconds=row['inherited_force6_source_seconds'],inherited_port0_basis_seconds=row['inherited_port0_basis_seconds'],inherited_remaining9_translation_basis_seconds=row['actual_remaining9_basis_seconds'],authentic_source_support_coordinates=source_public_facts,
                physics_basis_reads_before_independent_solve=0,fresh_engineering_fullcold_seconds=None,groups={})
            a['case_records'][cad]=rec
            groups=[]
            for j in range(4):
                motions=torch.zeros((3,4,6),dtype=torch.float64,device='cuda');motions[:,j,3:]=basis_motions
                groups.append(('rotation_port'+str(j),motions,torch.zeros_like(forces),['rotation_p'+str(j)+'_'+axis for axis in ('rx','ry','rz')]))
            groups.append(('independent_mixed',mixed_motions,forces,mixednames))
            for role,motions,rhs,names in groups:
                a['phase']=cad+'/'+role;persist();st=(time.time(),time.monotonic())
                def remaining():return 1200-a['combined_solve_seconds']-elapsed(st)
                inv,reported=solve_group(torch,np,batch,M,inputs,ports,others,reference_xyz,motions,rhs,names,folder/role,a,guard,persist,remaining)
                wall=elapsed(st);assert wall<=300;a['combined_solve_seconds']+=wall
                rec['groups'][role]=dict(wall_seconds=wall,reported_seconds=reported,all_initial_free_guesses='independent_zero',geometry_only_fixed_lift=True)
                a['completed_groups'][cad+'/'+role]=dict(complete=True,wall_seconds=wall,immutable_endpoint_count=3)
                a['prediction_inventory'].update({cad+'/'+name:item for name,item in inv.items()});a['committed_finite_endpoints']+=3;persist()
            resident[cad]=dict(inputs=inputs,H=H,ids=ids,B=B,D=D,ports=ports,others=others,fine=fine,op_arrays=op_arrays,before=before,forces6=forces6,reference_xyz=reference_xyz)
        assert a['committed_finite_endpoints']==30 and a['input_cache_reads']==2 and a['basis_array_reads']==0 and a['operator_array_reads']==6 and a['coarse_array_reads']==20
        a['phase']='all30_independent_support_field_equilibrium_barrier';persist();rt=(time.time(),time.monotonic());fields={};records={}
        names=newnames+mixednames
        expected={cad+'/'+name for cad in s['selection']['CADs']for name in names};assert set(a['prediction_inventory'])==expected
        for key,item in a['prediction_inventory'].items():
            assert d.identity(item['path'],hashed=True)==item;record=d.read(item['path']);cad,name=key.split('/')
            assert record['initial_free_guess']=='independent_zero'and record['geometry_only_fixed_lift']and not record['physicsbasis_warmstart_used']
            assert record['key']==name and record['all_original_nodes']==s['cases'][cad]['expected_geometry']['nodes']
            v=tensor(read_array(record['physical_U'],'owned_array_reads'));assert v.dtype==torch.float64 and v.shape==(record['all_original_nodes'],3)
            assert np.asarray(record['all_support_reaction_forces_N']).shape==(4,3)and np.isfinite(record['all_support_reaction_forces_N']).all()
            assert np.asarray(record['all_support_local_moments_Nmm']).shape==(4,3)and np.isfinite(record['all_support_local_moments_Nmm']).all()
            fields[key]=v;records[key]=record
        d.write(o/'independent30_prediction_barrier.json',dict(status='completed',all30_finite_hash_body_Cartesian=True,GT_reads=0,
            old_force_basis_reads_before_barrier=0,new_support_basis_used_for_independent_mixed_warmstart=False,
            compact_comparisons_before_barrier=0,endpoint_equation='Original freeK,actual four sourceRBE2 nonzero fixedlifts',inventory=a['prediction_inventory']))
        a['prediction_barrier_complete']=True;persist()
        from geobc.support_rigid_motion_response_v120 import SupportRigidMotionResponse
        for cad,c in resident.items():
            a.update(current_CAD=cad,phase='postbarrier_force6_support24_response');persist();row=s['cases'][cad];rec=a['case_records'][cad];bt=(time.time(),time.monotonic())
            force_basis=torch.stack([tensor(read_array(row['force_basis_records'][name]['physical_U'],'basis_array_reads'))for name in s['selection']['force_basis_order']])
            translation_basis=torch.stack([tensor(read_array(d.read(item['path'])['physical_U'],'basis_array_reads'))for item in row['support12_basis_manifest_identities']]).reshape(4,3,row['expected_geometry']['nodes'],3)
            rotation_basis=torch.stack([fields[cad+'/'+name]for name in newnames]).reshape(4,3,row['expected_geometry']['nodes'],3)
            assert force_basis.shape==(6,row['expected_geometry']['nodes'],3)and rotation_basis.shape==translation_basis.shape
            before_force=checksum(force_basis);before_translation=checksum(translation_basis);before_rotation=checksum(rotation_basis)
            api=SupportRigidMotionResponse(torch,c['B'],force_basis,translation_basis,rotation_basis,row['source_support_constraints']['support_reference_GRID_IDs'],s['selection']['translation_basis_unit_mm'],s['selection']['rotation_basis_unit_rad'],guard)
            torch.cuda.synchronize();rec['postbarrier_force_basis_load_API_constructor_seconds']=elapsed(bt)
            qt=(time.time(),time.monotonic());out=api.query(c['forces6'],mixed_motions)['physical_U_mm'];torch.cuda.synchronize()
            rec['compact_force6_support24_update_fullfield_seconds']=elapsed(qt);metrics={}
            for j,name in enumerate(mixednames):
                v=fields[cad+'/'+name];metrics[name]=metric(out[j],v,1e-5)
            assert checksum(force_basis)==before_force and checksum(translation_basis)==before_translation and checksum(rotation_basis)==before_rotation and {k:checksum(v)for k,v in c['op_arrays'].items()}==c['before']
            theta=torch.tensor(s['selection']['rigid_rotation_rad'],dtype=torch.float64,device='cuda');origin=c['inputs']['reference_xyz_mm']
            target=torch.zeros((1,4,6),dtype=torch.float64,device='cuda');target[0,:,:3]=torch.cross(theta.expand(4,3),c['reference_xyz']-origin,dim=-1);target[0,:,3:]=theta
            rigidout=api.query(torch.zeros((1,6),dtype=torch.float64,device='cuda'),target)['physical_U_mm'][0]
            expected_rotation=torch.cross(theta.expand(row['expected_geometry']['nodes'],3),c['inputs']['vertices_mm']-origin,dim=-1)
            rigidmetric=metric(rigidout,expected_rotation,1e-5)
            assert checksum(force_basis)==before_force and checksum(translation_basis)==before_translation and checksum(rotation_basis)==before_rotation
            rec.update(rigid_rotation_numeric_compatibility=rigidmetric,compact_vs_independent_mixed=metrics,original_K_oldnew_basis_bytes_unchanged=True,
                actual_thirty_direction_API_metadata=api.metadata,passed=rigidmetric['passed']and all(v['passed']for v in metrics.values()))
            persist()
        assert a['basis_array_reads']==36 and a['owned_array_reads']==30 and a['deck_coordinate_hash_reads']==a['deck_coordinate_passes']==2
        a['endpoint_records']=records;a['readout_seconds']=elapsed(rt);assert a['readout_seconds']<=180
        conv=all(v['numerical_state']=='converged'and v['true_free_relative_residual']<=1e-8 and v['all_fixed_boundary_passed']and v['body_energy_virtualwork_passed']for v in records.values())
        passed=conv and all(v['passed']for v in a['case_records'].values())
        a.update(status='completed',phase='terminal',support_rigid_motion_gate_passed=passed,scientific_outcome='fixed_all_support_rigid_motion_PASS'if passed else'fixed_all_support_rigid_motion_negative')
    except Exception as error:
        from geobc.elastic_pilot import ScientificNegative
        from geobc.elastic_pilot_data import UnsupportedInput
        if isinstance(error,(ScientificNegative,UnsupportedInput)):a.update(status='completed',phase='terminal_scientific_negative',reason=str(error),scientific_outcome='fixed_all_support_rigid_motion_negative')
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
