"""Actual fixed six-force reference API; predictions precede independent fields."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,json,os,signal,hashlib
from pathlib import Path

def run(path):
    from geobc import available_cad_pilot_data as d
    from geobc.reference_motion_contract_v113 import approved_contract
    p,r=approved_contract(path);s=p['scientific'];o=d.path(p['output']);outer=json.loads(os.environ['GEOBC_REFERENCE_MOTION_V113_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest'] and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=p['scientific_digest'],boot_id=outer['boot_id']);d.write(o/'execution.json',execution)
    a=dict(status='running',phase='cache',active=True,closed=False,execution=execution,input_cache_reads=0,basis_array_reads=0,
        independent_mixed_array_reads=0,owned_prediction_array_reads=0,GT_reads=0,K_PCG_replays=0,raw_H5_deck_GRID_reads=0,
        neural_PT_reads=0,training_updates=0,scaler_fits=0,FINAL_reads=0,case_records={},prediction_inventory={},
        committed_pose_requests=0,committed_pose_components=0,committed_energy_values=0,actual_force6_gradient_points=0,
        prediction_barrier_complete=False,reference_motion_gate_passed=False,new_original_rotation_GT_accuracy=False,
        signed_S_solved=False,final_solver_selected=False,peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0)
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
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External reference-motion guard')))
    try:
        import fcntl
        lock=d.path(s['budgets']['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch as tm
        torch=tm
        from geobc.reference_motion_response_v113 import ReferenceMotionResponse
        from geobc import elastic_pilot as old
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        total=torch.cuda.get_device_properties(0).total_memory;assert total>30500000000
        torch.cuda.set_per_process_memory_fraction(30500000000/total,0);torch.cuda.reset_peak_memory_stats()
        def read_array(item,count):
            guard();expected={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')};expected['path']=d.relative(d.path(item['path']))
            assert d.identity(item['path'],hashed=True)==expected
            v=np.load(d.path(item['path']),allow_pickle=False);a[count]+=1
            dtype={'torch.float64':'float64','torch.float32':'float32','torch.int64':'int64'}.get(item['dtype'],item['dtype'])
            assert list(v.shape)==item['shape'] and str(v.dtype)==dtype and np.isfinite(v).all();return v
        def tensor(v,dtype):return torch.as_tensor(v,dtype=dtype,device='cuda')
        def checksum(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        def metric(v,ref,tol):
            error=float(torch.linalg.vector_norm(v-ref));norm=float(torch.linalg.vector_norm(ref));absolute=float((v-ref).abs().max())
            relative=error/norm if norm>0 else None
            return dict(relative=relative,maximum_absolute=absolute,passed=relative<=tol if relative is not None else absolute<=1e-12)
        def save(file,v):
            guard();item=old._save_array(np,file,v);item.update(d.identity(item['path'],hashed=True));return item
        coeff=tensor(s['selection']['mixed_coefficients'],torch.float64);resident={}
        for cad in s['selection']['CADs']:
            a.update(current_CAD=cad,phase='strict_predictor_cache');persist();row=s['cases'][cad];folder=o/cad;folder.mkdir(exist_ok=False);ct=(time.time(),time.monotonic())
            assert d.identity(row['input_cache_identity']['path'],hashed=True)==row['input_cache_identity']
            with np.load(d.path(row['input_cache_identity']['path']),allow_pickle=False)as z:
                values={k:z[k].copy()for k in s['predictor_inputs']['members']}
            a['input_cache_reads']+=1
            H,ids,derived=values['rbe3_Hbar'],values['rbe3_member_node_ids'],values['derived_node_ids'];L=float(values['rbe3_length_mm'])
            assert H.dtype==np.float64 and ids.dtype==derived.dtype==np.int64 and derived.shape==(1,)
            assert L==row['source_transfer_facts']['length_mm'] and len(ids)==row['source_transfer_facts']['member_count']
            H,ids,derived=tensor(H,torch.float64),tensor(ids,torch.int64),tensor(derived,torch.int64)
            basis=torch.stack([tensor(read_array(row['basis_records'][name]['physical_U'],'basis_array_reads'),torch.float64)for name in s['selection']['basis_order']])
            assert basis.shape==(6,row['expected_geometry']['nodes'],3) and bool(((derived>=0)&(derived<basis.shape[1])).all())
            B=tensor(row['reference_load6_columns'],torch.float64).T
            before=dict(Hbar=checksum(H),members=checksum(ids),derivedIDs=checksum(derived),basis=checksum(basis),B6=checksum(B))
            torch.cuda.synchronize();load_seconds=elapsed(ct);it=(time.time(),time.monotonic());api=ReferenceMotionResponse(torch,H,ids,L,basis,B,guard);torch.cuda.synchronize()
            rec=dict(actual_fullN=basis.shape[1],constructor_metadata=api.metadata,predictor_cache_seconds=load_seconds,
                API_constructor_seconds=elapsed(it),fit_length_mm=L,inherited_source_to_six_basis_seconds=row['inherited_source_to_six_basis_seconds'],
                fresh_engineering_fullcold_seconds=None,source_REF_GRID=row['source_transfer_facts']['reference_GRID_ID'])
            a['case_records'][cad]=rec;rawC=checksum(api.C)
            forces=torch.cat((B[:,:3].T,coeff@B.T));qt=(time.time(),time.monotonic());result=api.query_force6(forces);pose=result['reference_motion6'];energy=result['strain_energy_Nmm']
            chunk=torch.cat([api.query_force6(forces[j:j+2])['reference_motion6']for j in (0,2,4)])
            hom=api.query_force6(.5*forces)['reference_motion6'];sup=api.query_force6(forces[:3]+forces[3:])['reference_motion6']
            rec['chunk']=metric(chunk*api.scales,pose*api.scales,1e-10)
            rec['homogeneity']=metric(hom*api.scales,.5*pose*api.scales,1e-10)
            rec['superposition']=metric(sup*api.scales,(pose[:3]+pose[3:])*api.scales,1e-10)
            torch.cuda.synchronize();rec['query_chunk_properties_seconds']=elapsed(qt)
            gt=(time.time(),time.monotonic());checks=[]
            for f in forces[3:5]:
                g=f.detach().clone().requires_grad_(True)
                jac=torch.autograd.functional.jacobian(lambda v:api.query_force6(v[None])['reference_motion6'][0],g)
                out=api.query_force6(g[None]);eg=torch.autograd.grad(out['strain_energy_Nmm'].sum(),g)[0]
                nonzero=bool(torch.isfinite(jac).all()and torch.isfinite(eg).all()and (torch.linalg.vector_norm(jac,dim=0)>0).all())
                jm=metric(api.scales[:,None]*jac*api.scales[None,:],api.scales[:,None]*api.C*api.scales[None,:],1e-10)
                em=metric(eg*api.scales,out['reference_motion6'][0]*api.scales,1e-5)
                checks.append(dict(all6_Jacobian_columns_nonzero_finite=nonzero,Jacobian_vs_rawC=jm,energy_gradient_vs_motion=em,passed=nonzero and jm['passed']and em['passed']))
                a['actual_force6_gradient_points']+=1
            torch.cuda.synchronize();rec.update(gradient_checks=checks,actual_gradient_seconds=elapsed(gt))
            inv=dict(CAD=cad,request_order=['ver','hor','tor','mix6_0','mix6_1','mix6_2'],force6=save(folder/'force6.npy',forces),
                reference_motion6=save(folder/'reference_motion6.npy',pose),energy_Nmm=save(folder/'energy_Nmm.npy',energy),
                requests=6,components=36,energy_values=6,independent_references_before_barrier=0)
            d.write(folder/'manifest.json',inv);a['prediction_inventory'][cad]=d.identity(folder/'manifest.json',hashed=True)
            a['committed_pose_requests']+=6;a['committed_pose_components']+=36;a['committed_energy_values']+=6
            resident[cad]=dict(api=api,H=H,ids=ids,derived=derived,basis=basis,B=B,before=before,rawC=rawC,pose=pose)
            persist()
        assert a['committed_pose_requests']==12 and a['committed_pose_components']==72 and a['committed_energy_values']==12
        assert a['input_cache_reads']==2 and a['basis_array_reads']==12 and a['independent_mixed_array_reads']==0
        assert set(a['prediction_inventory'])==set(s['selection']['CADs'])
        for cad,item in a['prediction_inventory'].items():
            assert d.identity(item['path'],hashed=True)==item;inv=d.read(item['path']);assert inv['CAD']==cad and inv['requests']==6 and inv['independent_references_before_barrier']==0
            for key,shape in (('force6',(6,6)),('reference_motion6',(6,6)),('energy_Nmm',(6,))):
                v=read_array(inv[key],'owned_prediction_array_reads');assert v.shape==shape
        d.write(o/'prediction_barrier.json',dict(status='completed',complete_requests=12,components=72,energy_values=12,all_finite_hash_body_Cartesian=True,
            independent_reference_reads_before_barrier=0,GT_reads=0,inventory=a['prediction_inventory']));a['prediction_barrier_complete']=True;persist()
        for cad,c in resident.items():
            a.update(current_CAD=cad,phase='postbarrier_independent_mixed_motion');persist();rt=(time.time(),time.monotonic());row=s['cases'][cad];rec=a['case_records'][cad];api=c['api']
            references={}
            for j in range(3):
                item=row['independent_mixed_manifest_identities']['mix6_'+str(j)];assert d.identity(item['path'],hashed=True)==item;manifest=d.read(item['path'])
                v=tensor(read_array(manifest['physical_U'],'independent_mixed_array_reads'),torch.float64)
                scaled=c['H']@v[c['ids']].reshape(-1);physical=scaled/api.scales
                references[str(j)]=dict(scaled_pose_mm=metric(c['pose'][j+3]*api.scales,scaled,1e-5),
                    translation_mm=metric(c['pose'][j+3,:3],physical[:3],1e-5),rotation_rad=metric(c['pose'][j+3,3:],physical[3:],1e-5),
                    independently_zero_solved_same_equation=True,original_OptiStruct_GT=False)
            witness=metric(api.motion_columns[:3,:3].T,c['basis'][:3,c['derived'][0]],1e-10)
            after=dict(Hbar=checksum(c['H']),members=checksum(c['ids']),derivedIDs=checksum(c['derived']),basis=checksum(c['basis']),B6=checksum(c['B']))
            assert after==c['before'] and checksum(api.C)==c['rawC']
            m=api.metadata;energy_pass=m['energy_positive']and m['energy_normalized_condition']<=1e12
            rec.update(independent_mixed_metrics=references,canonical_derived_translation_witness=witness,rawC_input_basis_bytes_unchanged=True,
                independent_reference_seconds=elapsed(rt),passed=m['reciprocity_relative']<=1e-5 and energy_pass and witness['passed']
                and all(v['scaled_pose_mm']['passed']for v in references.values())and all(v['passed']for v in rec['gradient_checks'])
                and all(rec[k]['passed']for k in ('chunk','homogeneity','superposition')))
            persist()
        assert a['independent_mixed_array_reads']==6 and a['actual_force6_gradient_points']==4
        passed=all(v['passed']for v in a['case_records'].values())
        a.update(status='completed',phase='terminal',reference_motion_gate_passed=passed,scientific_outcome='reference_motion_energy_PASS'if passed else'fixed_reference_motion_negative')
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
