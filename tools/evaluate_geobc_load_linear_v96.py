"""Released reusable API qualification and108 full-node predictions, GT0."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
import json
import os
from pathlib import Path
import signal


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.load_linear_contract_v96 import approved_contract
    p,review,m,prior=approved_contract(plan_path);s=p['scientific'];o=d.path(p['output'])
    outer=json.loads(os.environ['GEOBC_LOAD_LINEAR_V96_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest']and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,boot_id=outer['boot_id'],scientific_digest=p['scientific_digest'])
    assert not(o/'execution.json').exists();d.write(o/'execution.json',execution)
    a=dict(status='running',active=True,closed=False,phase='target_free_cache',execution=execution,
        input_array_attempts=0,input_array_reads=0,checkpoint_hash_reads=0,checkpoint_loads=0,own_array_hash_attempts=0,own_array_reads=0,
        GT_reads=0,CSV_H5_deck_reads=0,training_updates=0,scaler_fits=0,candidates={},prediction_inventory={},
        committed_finite_predictions=0,fixed_prediction_denominator=108,property_rows=0,fixed_property_denominator=81,
        peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0,protocol_qualified=False,final_solver_selected=False)
    torch=None;lock=None
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Frozen900 whole dual-clock deadline')
        if torch is not None and torch.cuda.is_initialized():
            a['peak_allocated_bytes']=max(a['peak_allocated_bytes'],torch.cuda.max_memory_allocated());a['peak_reserved_bytes']=max(a['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
            free,total=torch.cuda.mem_get_info();a['peak_total_device_used_bytes_sampled']=max(a['peak_total_device_used_bytes_sampled'],total-free)
            if max(a[k]for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))>s['budget']['memory_bytes']:raise MemoryError('30.5GB cap')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External guard terminated child')))
    try:
        import fcntl
        lock=d.path(s['budget']['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch as tm
        torch=tm
        from geobc.load_linear_io_v96 import load_inputs,load_model,read_owned_array
        from geobc.load_linear_adapter_v96 import LoadLinearDisplacementAdapter
        from geobc.macro_training import encode_case
        from geobc.fno_stress_specialist_v54 import tree_digest
        from geobc.physical_field_loss import PhysicalScale
        from geobc.solver_selection_data import _stencil
        from geobc.coverage_comparison_data import _save
        from geobc.macro_readout_protocol import verify_barrier
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        total=torch.cuda.get_device_properties(0).total_memory;assert s['budget']['memory_bytes']<total
        torch.cuda.set_per_process_memory_fraction(s['budget']['memory_bytes']/total,0);torch.cuda.reset_peak_memory_stats()
        cases,arrays=load_inputs(np,torch,m,s['selected_TRAIN_input_keys'],a,guard)
        original_digests={k:tree_digest(v)for k,v in arrays.items()};scale=PhysicalScale.from_mapping(s['scaler'])
        cads=sorted({k.split('/')[0]for k in cases});keys=[];counts={};inventory={};specs=[]
        for cad in cads:
            for load in ('ver','hor','tor'):
                for alpha in s['amplitude_values']:keys.append(cad+'/'+load+'@a'+str(alpha))
            for i in range(3):keys.append(cad+'/mix'+str(i))
        for key in keys:counts[key]=len(cases[key.split('/')[0]+'/ver']['xyz'])
        for c in s['frozen_candidates']:
            guard();name=c['name'];model=load_model(torch,c,s,a,guard);md=tree_digest(model.state_dict())
            row=dict(strict_checkpoint_verified=True,predictions=0,geometry_profiles={},qualification=None,
                inference_epoch_seconds=0.,inference_monotonic_seconds=0.,API_prepare_epoch_seconds=0.,API_prepare_monotonic_seconds=0.)
            a['candidates'][name]=row;spec=dict(name=name,family=c['family'],step=6000,checkpoint=c['checkpoint']);specs.append(spec)
            witness=max(cads,key=lambda cad:len(cases[cad+'/ver']['xyz']))
            for cad in cads:
                guard();bases=[cases[cad+'/'+load]for load in ('ver','hor','tor')];began=(time.time(),time.monotonic())
                with torch.no_grad():adapter=LoadLinearDisplacementAdapter(model,bases,scale,guard).prepare()
                row['API_prepare_epoch_seconds']+=time.time()-began[0];row['API_prepare_monotonic_seconds']+=time.monotonic()-began[1]
                max_nonload=max(float((x['grid'][:,:11]-bases[0]['grid'][:,:11]).abs().max())for x in bases)
                max_nonload=max(max_nonload,max(float((x['grid'][:,17:]-bases[0]['grid'][:,17:]).abs().max())for x in bases))
                row['geometry_profiles'][cad]=dict(real_rank3=True,normalized_force_basis=adapter.basis.detach().cpu().tolist(),condition_number=adapter.condition_number,canonical_nonload_grid_maxabsdiff=max_nonload,same_zero_constraints_verified=prior['physics_gate'][cad]['same_constraints_across3'])
                assert prior['physics_gate'][cad]['same_constraints_across3']and max_nonload<=1e-4
                def mixed_case(coeff):
                    coeff=torch.as_tensor(coeff,dtype=torch.float64,device='cuda');force=adapter.basis@coeff
                    base=bases[0];case=dict(base);case['features']=base['features'].clone();case['features'][:,7:13]=force.to(torch.float32)[None]
                    case['grid']=base['grid'].clone();case['grid'][:,11:17]=sum(coeff[i]*bases[i]['grid'][:,11:17].to(torch.float64)for i in range(3)).to(torch.float32)
                    return case,force
                if cad==witness:
                    # Real64 spatial queries and load-space coefficient gradients,
                    # using actual largest geometry/model/canonical latent fields.
                    qc=s['qualification_in_span_coefficients'];whole,force=mixed_case(qc)
                    with torch.no_grad():ctx,latent=encode_case(model,whole)
                    selected=torch.linspace(0,len(whole['xyz'])-1,64,device='cuda').long()
                    xyz=whole['xyz'][selected].detach().clone().requires_grad_(True)
                    coeff=torch.tensor(qc,device='cuda',dtype=torch.float64,requires_grad=True);f6=adapter.basis@coeff
                    features=whole['features'][selected].detach().clone();features[:,7:13]=f6.to(torch.float32)[None]
                    ind,weights=_stencil(xyz,64);query=dict(xyz=xyz,features=features,query_indices=ind,query_weights=weights)
                    physical,encoded=adapter.query(query,0,64,f6,ctx,latent)
                    gx,gc=torch.autograd.grad(physical[:,:3].square().sum(),(xyz,coeff))
                    assert torch.isfinite(gx).all()and torch.isfinite(gc).all()and gx.norm()>0 and gc.norm()>0
                    with torch.no_grad():
                        parts=[adapter.query(query,start,min(start+17,64),f6,ctx,latent)[1]for start in range(0,64,17)]
                        error=float((torch.cat(parts)-encoded).abs().max())
                    tol=s['query_chunk_encoded_tolerance_FP32']if c['family']=='FNO9'else s['query_chunk_encoded_tolerance_TCNN_BF16']
                    assert error<=tol and tree_digest(model.state_dict())==md
                    row['qualification']=dict(passed=True,real_largest_CAD=cad,original_nodes=len(whole['xyz']),actual_queries=64,coordinate_gradient_L2=float(gx.norm()),in_span_three_coefficient_gradient_L2=float(gc.norm()),query_chunk_encoded_maxabs=error,tolerance=tol,model_bytes_unchanged=True,independent_offnode_FEM_accuracy_proven=False)
                    del whole,force,ctx,latent,query,physical,encoded,parts,gx,gc;guard();persist()
                variants=[]
                for load_index,load in enumerate(('ver','hor','tor')):
                    for alpha in s['amplitude_values']:variants.append((cad+'/'+load+'@a'+str(alpha),'amplitude',[alpha if i==load_index else 0. for i in range(3)],load_index,alpha))
                for i,coeff in enumerate(s['mixed_coefficients']):variants.append((cad+'/mix'+str(i),'mixture',coeff,None,None))
                with torch.no_grad():
                    for virtual,kind,coeff,index,alpha in variants:
                        guard()
                        if kind=='amplitude':
                            base=bases[index];case=dict(base);case['features']=base['features'].clone();case['features'][:,7:13]*=alpha;case['grid']=base['grid'].clone();case['grid'][:,11:17]*=alpha
                            force=case['features'][0,7:13].to(torch.float64)
                        else:case,force=mixed_case(coeff)
                        began=(time.time(),time.monotonic());ctx,latent=encode_case(model,case);n=len(case['xyz']);ep=np.empty((n,4),np.float32);pp=np.empty((n,4),np.float32)
                        for start in range(0,n,32768):
                            guard();stop=min(start+32768,n);physical,encoded=adapter.query(case,start,stop,force,ctx,latent)
                            ep[start:stop]=encoded.cpu().numpy();pp[start:stop]=physical.cpu().numpy()
                        torch.cuda.synchronize();row['inference_epoch_seconds']+=time.time()-began[0];row['inference_monotonic_seconds']+=time.monotonic()-began[1]
                        assert np.isfinite(ep).all()and np.isfinite(pp).all();folder=o/'predictions'/name/virtual
                        record=dict(candidate=name,key=virtual,family=c['family'],step=6000,checkpoint=c['checkpoint'],all_original_nodes=n,status='completed',signed_stress_unchanged=True,kind=kind,coefficients=coeff,
                            encoded=_save(np,folder/'encoded4.npy',ep),physical=_save(np,folder/'physical4.npy',pp))
                        file=folder/'manifest.json';d.write(file,record);inventory[name+':'+virtual]=d.identity(file,hashed=True);row['predictions']+=1;a['committed_finite_predictions']+=1;persist()
                        del case,force,ctx,latent,ep,pp,physical,encoded;guard()
                del adapter,bases;torch.cuda.empty_cache();guard()
            assert row['qualification']['passed']and row['predictions']==36 and tree_digest(model.state_dict())==md
            row['model_bytes_unchanged']=True;d.stat_matches(c['checkpoint']);del model;torch.cuda.empty_cache();guard()
        def record_reader(item):
            guard();assert d.identity(item['path'],hashed=True)==item;return d.read(item['path'])
        def array_reader(item):return read_owned_array(np,item,a,guard)
        barrier=verify_barrier(np,specs,keys,counts,inventory,record_reader,array_reader,guard);assert barrier['finite_predictions']==108
        d.write(o/'prediction_barrier.json',barrier);a.update(prediction_barrier_complete=True,prediction_inventory=inventory,phase='properties_and_preservation');persist()
        def udefect(value,ref):
            norm=float(np.square(ref).sum());sse=float(np.square(value-ref).sum());return (sse/norm)**.5 if norm>0 else None
        for c in s['frozen_candidates']:
            name=c['name'];new={};old_u={};amp_rows={};mix_rows={};canonical={};stress={}
            for key in keys:
                rec=record_reader(inventory[name+':'+key]);new[key]=array_reader(rec['physical'])[:,:3].astype(np.float64)
                if rec['kind']=='amplitude':
                    oldrec=record_reader(s['owned_v95_inventory'][name+':'+key]);encoded_old=array_reader(oldrec['encoded']);encoded_new=array_reader(rec['encoded'])
                    diff=float(np.abs(encoded_old[:,3]-encoded_new[:,3]).max());stress[key]=diff
                    if key.endswith('@a1.0'):old_u[key]=array_reader(oldrec['physical'])[:,:3].astype(np.float64);canonical[key]=udefect(new[key],old_u[key])
            for cad in cads:
                for load in ('ver','hor','tor'):
                    base=new[cad+'/'+load+'@a1.0']
                    for alpha in [.5,1.5]:
                        key=cad+'/'+load+'@a'+str(alpha);amp_rows[key]=udefect(new[key],alpha*base);a['property_rows']+=1
                for i,coeff in enumerate(s['mixed_coefficients']):
                    key=cad+'/mix'+str(i);ref=sum(coeff[j]*new[cad+'/'+load+'@a1.0']for j,load in enumerate(('ver','hor','tor')))
                    mix_rows[key]=udefect(new[key],ref);a['property_rows']+=1
            complete=all(v is not None and np.isfinite(v)for rows in (amp_rows,mix_rows,canonical)for v in rows.values())
            inv=s['invariants'];stol=inv['original_encodedS_maxabs_difference_FP32']if c['family']=='FNO9'else inv['original_encodedS_maxabs_difference_TCNN_BF16']
            row=a['candidates'][name];row['properties']=dict(amplitude_fixed_denominator=18,mixture_fixed_denominator=9,canonical_fixed_denominator=9,stress_fixed_denominator=27,all_cases_complete=complete,
                amplitude_per_case=amp_rows,mixture_per_case=mix_rows,canonical_per_case=canonical,raw_encodedS_per_case=stress,
                mean_amplitude_defect=sum(amp_rows.values())/18 if complete else None,worst_amplitude_defect=max(amp_rows.values())if complete else None,
                mean_mixture_defect=sum(mix_rows.values())/9 if complete else None,worst_mixture_defect=max(mix_rows.values())if complete else None,
                worst_canonical_U_difference=max(canonical.values())if complete else None,worst_raw_encodedS_difference=max(stress.values()),
                invariants_passed=complete and max(amp_rows.values())<=inv['homogeneity_relative_defect_max']and max(mix_rows.values())<=inv['internal_superposition_relative_defect_max']and max(canonical.values())<=inv['canonical_U_relative_difference_max']and max(stress.values())<=stol)
            persist()
        assert a['committed_finite_predictions']==108 and a['property_rows']==81 and a['checkpoint_loads']==a['checkpoint_hash_reads']==3
        assert original_digests=={k:tree_digest(v)for k,v in arrays.items()}
        a.update(status='completed',phase='terminal',protocol_qualified=True,input_tensor_bytes_unchanged=True,all_model_invariants_passed=all(x['properties']['invariants_passed']for x in a['candidates'].values()),GT_accuracy_measured=False);guard()
    except BaseException as error:
        a.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error));raise
    finally:
        a.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'summary.json',a);d.write(o/'status.json',a)
        if lock is not None:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');parser.add_argument('--plan');args=parser.parse_args()
    if args.start:
        if not args.plan:parser.error('--start requires frozen --plan')
        run(args.plan)
    else:print('{"status":"source_only","GT_PT_arrays_GPU":0}')
