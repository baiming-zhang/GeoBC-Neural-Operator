"""Released540 fullN prediction-first/canonicalGT and conditionalmixedU."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
import json
import os
from pathlib import Path
import signal


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.load_linear_knownDEV_contract_v97 import approved_contract
    p,review,m,truths=approved_contract(plan_path);s=p['scientific'];o=d.path(p['output'])
    outer=json.loads(os.environ['GEOBC_LOAD_LINEAR_KNOWNDEV_V97_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest']and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,boot_id=outer['boot_id'],scientific_digest=p['scientific_digest'])
    assert not(o/'execution.json').exists();d.write(o/'execution.json',execution)
    a=dict(status='running',active=True,closed=False,phase='target_free_cache',execution=execution,
        input_array_attempts=0,input_array_reads=0,checkpoint_hash_reads=0,checkpoint_loads=0,own_array_hash_attempts=0,own_array_reads=0,
        GT_reads=0,CSV_H5_reads=0,training_updates=0,scaler_fits=0,candidates={},prediction_inventory={},
        committed_finite_predictions=0,fixed_prediction_denominator=540,source_deck_hash_reads=0,source_deck_parse_attempts=0,source_deck_parses=0,source_deck_header_reads=0,physics_gate={},owned_GT_attempts=0,
        peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0,protocol_qualified=False,final_solver_selected=False)
    torch=None;lock=None
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Frozen1800 whole dual-clock deadline')
        if torch is not None and torch.cuda.is_initialized():
            a['peak_allocated_bytes']=max(a['peak_allocated_bytes'],torch.cuda.max_memory_allocated());a['peak_reserved_bytes']=max(a['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
            free,total=torch.cuda.mem_get_info();a['peak_total_device_used_bytes_sampled']=max(a['peak_total_device_used_bytes_sampled'],total-free)
            if max(a[k]for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))>s['memory_limit_bytes']:raise MemoryError('30.5GB cap')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External guard terminated child')))
    try:
        import fcntl
        lock=d.path(s['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch as tm
        torch=tm
        from geobc.load_linear_io_v96 import load_model,read_owned_array
        from geobc.load_linear_knownDEV_io_v97 import load_inputs,source_gate
        from geobc.physical_field_loss import decode_physical
        from geobc.coverage_comparison_train import physical_metrics,aggregate
        from geobc.solver_selection_data import _splat
        from geobc import elastic_pilot_data as parser
        from geobc.macro_training import query_case
        from geobc.load_linear_adapter_v96 import LoadLinearDisplacementAdapter
        from geobc.macro_training import encode_case
        from geobc.fno_stress_specialist_v54 import tree_digest
        from geobc.physical_field_loss import PhysicalScale
        from geobc.solver_selection_data import _stencil
        from geobc.coverage_comparison_data import _save
        from geobc.macro_readout_protocol import verify_barrier
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        total=torch.cuda.get_device_properties(0).total_memory;assert s['memory_limit_bytes']<total
        torch.cuda.set_per_process_memory_fraction(s['memory_limit_bytes']/total,0);torch.cuda.reset_peak_memory_stats()
        a['phase']='source_gate20';persist();forces,materials=source_gate(np,s,a,guard);persist()
        cases,arrays=load_inputs(np,torch,m,s['case_keys'],a,guard)
        assert all(case['scientific_group']=='DEV'for case in cases.values())
        for key,case in cases.items():
            guard();fv=torch.as_tensor(forces[key],device='cuda',dtype=torch.float32);mat=torch.as_tensor(materials[key.split('/')[0]],device='cuda',dtype=torch.float32)
            if not torch.allclose(case['features'][:,7:13],fv[None].expand(len(case['xyz']),-1),atol=s['force_absolute_tolerance'],rtol=0)or not torch.allclose(case['features'][:,13:16],mat[None].expand(len(case['xyz']),-1),atol=s['force_absolute_tolerance'],rtol=0):raise parser.UnsupportedInput('Actual source force/material and features differ')
            raster,active=_splat(case['features'],case['query_indices'],case['query_weights'],64)
            error=float((raster.T.reshape(25,64,64,64)-case['grid'][0,4:]).abs().max())
            if error>s['raster_absolute_tolerance']or not torch.equal(active.reshape(64,64,64).float(),case['grid'][0,0]):raise parser.UnsupportedInput('Original loadgrid raster mismatch')
            a['physics_gate'][key.split('/')[0]].setdefault('input_witness',{})[key]=dict(raster_all25_maxabs=error,raw_force_material_matches=True)
            del raster,active;guard()
        a['phase']='commit540_predictions';persist()
        original_digests={k:tree_digest(v)for k,v in arrays.items()};scale=PhysicalScale.from_mapping(s['scaler'])
        cads=sorted({k.split('/')[0]for k in cases});keys=[];counts={};inventory={};specs=[]
        for cad in cads:
            keys.extend(cad+'/'+load for load in ('ver','hor','tor'))
            keys.extend(cad+'/mix'+str(i)+'@'+kind for i in range(3)for kind in ('raw','adapter'))
        for key in keys:counts[key]=len(cases[key.split('/')[0]+'/ver']['xyz'])
        for c in s['frozen_candidates']:
            guard();name=c['name'];model=load_model(torch,c,s,a,guard);md=tree_digest(model.state_dict())
            row=dict(strict_checkpoint_verified=True,predictions=0,geometry_profiles={},qualification=None,
                inference_epoch_seconds=0.,inference_monotonic_seconds=0.,API_prepare_epoch_seconds=0.,API_prepare_monotonic_seconds=0.)
            a['candidates'][name]=row;spec=dict(name=name,family=c['family'],step=6000,checkpoint=c['checkpoint']);specs.append(spec)
            witness=max(cads,key=lambda cad:len(cases[cad+'/ver']['xyz']))
            for cad in [witness]+[x for x in cads if x!=witness]:
                guard();bases=[cases[cad+'/'+load]for load in ('ver','hor','tor')];began=(time.time(),time.monotonic())
                with torch.no_grad():adapter=LoadLinearDisplacementAdapter(model,bases,scale,guard).prepare()
                row['API_prepare_epoch_seconds']+=time.time()-began[0];row['API_prepare_monotonic_seconds']+=time.monotonic()-began[1]
                max_nonload=max(float((x['grid'][:,:11]-bases[0]['grid'][:,:11]).abs().max())for x in bases)
                max_nonload=max(max_nonload,max(float((x['grid'][:,17:]-bases[0]['grid'][:,17:]).abs().max())for x in bases))
                row['geometry_profiles'][cad]=dict(real_rank3=True,normalized_force_basis=adapter.basis.detach().cpu().tolist(),condition_number=adapter.condition_number,canonical_nonload_grid_maxabsdiff=max_nonload,same_zero_constraints_verified=a['physics_gate'][cad]['same_constraints_across3'])
                assert a['physics_gate'][cad]['same_constraints_across3']and max_nonload<=1e-4
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
                variants=[(cad+'/'+load,'canonical',None,index)for index,load in enumerate(('ver','hor','tor'))]
                variants.extend((cad+'/mix'+str(i)+'@'+kind,kind,coeff,None)for i,coeff in enumerate(s['coefficients'])for kind in ('raw','adapter'))
                with torch.no_grad():
                    for virtual,kind,coeff,index in variants:
                        guard()
                        if kind=='canonical':case=bases[index];force=case['features'][0,7:13].to(torch.float64)
                        else:case,force=mixed_case(coeff)
                        began=(time.time(),time.monotonic());ctx,latent=encode_case(model,case);n=len(case['xyz']);ep=np.empty((n,4),np.float32);pp=np.empty((n,4),np.float32)
                        for start in range(0,n,32768):
                            guard();stop=min(start+32768,n)
                            if kind=='raw':
                                encoded=query_case(model,case,start,stop,ctx,latent).to(torch.float32);physical=decode_physical(encoded,scale)
                            else:physical,encoded=adapter.query(case,start,stop,force,ctx,latent)
                            ep[start:stop]=encoded.cpu().numpy();pp[start:stop]=physical.cpu().numpy()
                        torch.cuda.synchronize();row['inference_epoch_seconds']+=time.time()-began[0];row['inference_monotonic_seconds']+=time.monotonic()-began[1]
                        assert np.isfinite(ep).all()and np.isfinite(pp).all();folder=o/'predictions'/name/virtual
                        record=dict(candidate=name,key=virtual,family=c['family'],step=6000,checkpoint=c['checkpoint'],all_original_nodes=n,status='completed',signed_stress_unchanged=True,kind=kind,coefficients=coeff,conditional_mixed_reference_only=kind!='canonical',
                            encoded=_save(np,folder/'encoded4.npy',ep),physical=_save(np,folder/'physical4.npy',pp))
                        file=folder/'manifest.json';d.write(file,record);inventory[name+':'+virtual]=d.identity(file,hashed=True);row['predictions']+=1;a['committed_finite_predictions']+=1;persist()
                        del case,force,ctx,latent,ep,pp,physical,encoded;guard()
                del adapter,bases;torch.cuda.empty_cache();guard()
            assert row['qualification']['passed']and row['predictions']==180 and tree_digest(model.state_dict())==md
            row['model_bytes_unchanged']=True;d.stat_matches(c['checkpoint']);del model;torch.cuda.empty_cache();guard()
        def record_reader(item):
            guard();assert d.identity(item['path'],hashed=True)==item;return d.read(item['path'])
        def array_reader(item):return read_owned_array(np,item,a,guard)
        barrier=verify_barrier(np,specs,keys,counts,inventory,record_reader,array_reader,guard);assert barrier['finite_predictions']==540
        d.write(o/'prediction_barrier.json',barrier);a.update(prediction_barrier_complete=True,prediction_inventory=inventory,phase='canonical_preservation_GT_still_closed');persist()
        def udefect(value,ref):
            norm=float(np.square(ref).sum());sse=float(np.square(value-ref).sum());return (sse/norm)**.5 if norm>0 else None
        # Preservation and same requested-load rawS checked before any GT read.
        for c in s['frozen_candidates']:
            name=c['name'];row=a['candidates'][name];pres={};stress={}
            old_inventory=s['reference_inventory'][name]
            for key in s['case_keys']:
                rec=record_reader(inventory[name+':'+key]);old_item=next(v for k,v in old_inventory.items()if k.split(':',1)[1]==key);oldrec=record_reader(old_item)
                pred=array_reader(rec['physical']);ref=array_reader(oldrec['physical']);pres[key]=udefect(pred[:,:3].astype(np.float64),ref[:,:3].astype(np.float64));del pred,ref
                newencoded=array_reader(rec['encoded']);oldencoded=array_reader(oldrec['encoded']);stress[key]=float(np.abs(newencoded[:,3]-oldencoded[:,3]).max());del newencoded,oldencoded
            for cad in cads:
                for i in range(3):
                    raw=record_reader(inventory[name+':'+cad+'/mix'+str(i)+'@raw']);new=record_reader(inventory[name+':'+cad+'/mix'+str(i)+'@adapter']);re=array_reader(raw['encoded']);ne=array_reader(new['encoded']);diff=float(np.abs(re[:,3]-ne[:,3]).max());stress[cad+'/mix'+str(i)]=diff;del re,ne
            complete=len(pres)==60 and all(x is not None and np.isfinite(x)for x in pres.values())
            stol=s['canonical_encodedS_FP32_tolerance']if c['family']=='FNO9'else s['canonical_encodedS_BF16_tolerance']
            row['canonical_preservation']=dict(per_case_U_difference=pres,per_case_encodedS_maxabs=stress,all60_complete=complete,worst_U_difference=max(pres.values())if complete else None,worst_encodedS_difference=max(stress.values()),passed=complete and max(pres.values())<=s['canonical_U_relative_tolerance']and max(stress.values())<=stol)
            persist()
        assert all(row['canonical_preservation']['passed']for row in a['candidates'].values()),'Canonical preservation fails; GT stays closed'
        a['phase']='GT60_once_conditional_mixedU';persist()
        canonical={c['name']:{}for c in s['frozen_candidates']};mixed={c['name']:{'raw':{},'adapter':{}}for c in s['frozen_candidates']}
        # Each CAD's3 actual GT arrays read once, retained only for its3 mixtures.
        for cad in cads:
            actual=[]
            for load in ('ver','hor','tor'):
                key=cad+'/'+load;item=truths['records'][key]['physical'];guard();a['owned_GT_attempts']+=1
                assert d.identity(item['path'],hashed=True)=={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')}
                gt=np.load(d.path(item['path']),allow_pickle=False);a['GT_reads']+=1
                assert gt.shape==(counts[key],4)and gt.dtype==np.float32 and np.isfinite(gt).all();actual.append(gt)
                for c in s['frozen_candidates']:
                    name=c['name'];rec=record_reader(inventory[name+':'+key]);pred=array_reader(rec['physical']);canonical[name][key]=physical_metrics(np,gt,pred);del pred
                persist()
            for i,coeff in enumerate(s['coefficients']):
                conditional_u=sum(coeff[j]*actual[j][:,:3].astype(np.float64)for j in range(3))
                for c in s['frozen_candidates']:
                    name=c['name']
                    for kind in ('raw','adapter'):
                        key=cad+'/mix'+str(i)+'@'+kind;rec=record_reader(inventory[name+':'+key]);pred=array_reader(rec['physical'])[:,:3].astype(np.float64);res=pred-conditional_u
                        mixed[name][kind][cad+'/mix'+str(i)]=dict(u_relative_l2=udefect(pred,conditional_u),u_RMS_mm=float((np.square(res).sum()/len(res))**.5),u_max_mm=float(np.linalg.norm(res,axis=1).max()),reference='conditional_linear_static_derived_U_not_independent_new_FEM')
                        del pred,res
            del actual,conditional_u;guard();persist()
        assert a['GT_reads']==a['owned_GT_attempts']==60
        for c in s['frozen_candidates']:
            name=c['name'];row=a['candidates'][name];row['canonical_full60']=aggregate(canonical[name],60)
            row['canonical_per_load20']={load:aggregate({k:v for k,v in canonical[name].items()if k.endswith('/'+load)},20)for load in ('ver','hor','tor')}
            row['conditional_mixed_U']={}
            for kind,per in mixed[name].items():
                vals=[x['u_relative_l2']for x in per.values()];complete=len(vals)==60 and all(v is not None and np.isfinite(v)for v in vals)
                row['conditional_mixed_U'][kind]=dict(fixed_denominator=60,per_case=per,all60_complete=complete,mean_U=sum(vals)/60 if complete else None,worst_U=max(vals)if complete else None,per_mix20={str(i):sum(x['u_relative_l2']for k,x in per.items()if k.endswith('/mix'+str(i)))/20 if complete else None for i in range(3)})
            raw=row['conditional_mixed_U']['raw'];adapt=row['conditional_mixed_U']['adapter'];complete=raw['all60_complete']and adapt['all60_complete']and row['canonical_full60']['all_cases_complete'];gain=raw['mean_U']-adapt['mean_U']if complete else None
            row['conditional_macro_gate']=dict(passed=complete and gain>=.03 and row['canonical_preservation']['passed'],U_improvement=gain,threshold=.03,not_independent_new_FEM=True,does_not_select_final_solver=True)
            row['paired_CAD20']={cad:sum(mixed[name]['adapter'][cad+'/mix'+str(i)]['u_relative_l2']-mixed[name]['raw'][cad+'/mix'+str(i)]['u_relative_l2']for i in range(3))/3 if complete else None for cad in cads}
        assert a['committed_finite_predictions']==540 and a['checkpoint_hash_reads']==a['checkpoint_loads']==3
        assert original_digests=={k:tree_digest(v)for k,v in arrays.items()}
        a.update(status='completed',phase='terminal',protocol_qualified=True,input_tensor_bytes_unchanged=True,original_canonical_GT_accuracy_measured=True,independent_newload_FEM_GT_accuracy_measured=False,conditional_derived_mixed_U_reference_used=True);guard()
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
