"""Matched ZERO/ACTUAL physical-information experiment on original signed U/S."""
import time
ENTRY_EPOCH, ENTRY_MONOTONIC = time.time(), time.monotonic()
import argparse
import json
import os
from pathlib import Path
import signal


def run(args):
    config=json.loads(Path(args.config).read_text()); catalog=json.loads(Path(args.source).read_text())
    clock=json.loads(os.environ['GEOBC_EXPERIMENT_CLOCK']); output=Path(args.output)
    state=dict(status='running',active=True,closed=False,phase='imports',arms={},predictions={},
               committed_predictions=0,development_GT_reads=0,training_target_reads=0,
               optimizer_step_in_progress=False,possible_partial_optimizer_update=False,
               scaler_fits=0,FINAL_reads=0,peak_allocated_bytes=0,peak_reserved_bytes=0)
    torch=None; cache=None; model=None; optimizer=None; arm=None; cursor=0; order=None
    def persist():
        state['native_seconds']=dict(UTC=time.time()-ENTRY_EPOCH,monotonic=time.monotonic()-ENTRY_MONOTONIC)
        temporary=output/'status.json.tmp'; temporary.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        os.replace(temporary,output/'status.json')
    def guard():
        if min(clock['deadline_epoch']-time.time(),clock['deadline_monotonic']-time.monotonic())<=0:
            raise TimeoutError('Information trial deadline')
        if torch is not None and torch.cuda.is_initialized():
            state['peak_allocated_bytes']=max(state['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
            state['peak_reserved_bytes']=max(state['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
            free,total=torch.cuda.mem_get_info()
            if max(total-free,state['peak_allocated_bytes'],state['peak_reserved_bytes'])>config['memory_bytes']:
                raise MemoryError('Information trial memory budget')
    def rng():
        return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
    def restore_rng(saved):
        random.setstate(saved['python']); np.random.set_state(saved['numpy'])
        torch.set_rng_state(saved['torch']); torch.cuda.set_rng_state_all(saved['cuda'])
    def checkpoint():
        if model is None or optimizer is None or arm is None:
            return
        guard_path=output/arm/'checkpoint.pt'; guard_path.parent.mkdir(exist_ok=True)
        temporary=guard_path.with_suffix('.pt.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),cursor=cursor,
                        order=order,rng=rng(),config=config,cost=state['arms'][arm],arm=arm),temporary)
        os.replace(temporary,guard_path)
    def save_prediction(path,value):
        path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix('.npy.tmp')
        with temporary.open('wb') as file: np.save(file,value,allow_pickle=False)
        os.replace(temporary,path)
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(TimeoutError('External experiment deadline')))
    try:
        import gc
        import random
        import numpy as np
        import torch as tm
        torch=tm
        from geobc.runtime import GPUQueue
        from geobc.learning_data import LearningCache
        from geobc.physics_inputs import local_path
        from geobc.kinematic_neural_field import KinematicCovariateFNO
        from geobc.physical_loss import backward_case,decode
        from geobc.query_stencil import stencil
        scale=json.loads(Path('configs/output_scaler.json').read_text())
        split=json.loads(Path('configs/split.json').read_text())
        train_keys=[cad+'/'+load for cad in split['training'] for load in config['loads']]
        dev_keys=[cad+'/'+load for cad in split['development'] for load in config['loads']]
        if set(catalog['TRAIN'])!=set(train_keys) or set(catalog['DEV'])!=set(dev_keys) or catalog['scaler']!=scale:
            raise ValueError('Matched dataset/scaler differs')
        torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        with GPUQueue(args.lock):
            torch.cuda.set_per_process_memory_fraction(config['memory_bytes']/torch.cuda.get_device_properties(0).total_memory)
            torch.cuda.reset_peak_memory_stats()
            state['phase']='cache'; persist(); tick=(time.time(),time.monotonic())
            cache=LearningCache(catalog,guard)
            cache.load_predictors(catalog['TRAIN'],catalog['training_features'],'TRAIN')
            cache.load_predictors(catalog['DEV'],catalog['development_features'],'DEV')
            cache.load_training_targets()
            state.update(training_target_reads=cache.training_target_reads,cache_seconds=dict(UTC=time.time()-tick[0],monotonic=time.monotonic()-tick[1]),
                         input_reads=cache.input_reads,physical_covariate_reads=cache.feature_reads)
            if max(state['cache_seconds'].values())>config['cache_seconds']:
                raise TimeoutError('Cache setup deadline')
            persist()
            witness=cache.cases[config['qualification_case']]
            if len(witness['xyz'])!=325387:
                raise ValueError('Actual maximum training geometry differs')
            for seed,modes in zip(config['seeds'],config['arm_order']):
                reference={k:v.detach().clone() for k,v in KinematicCovariateFNO('ZERO',seed).state_dict().items()}
                generator=random.Random(seed); order=[]
                for _ in range(config['passes']):
                    exposure=list(train_keys); generator.shuffle(exposure); order.extend(exposure)
                if len(order)!=config['updates_per_arm']:
                    raise ValueError('Full balanced exposure schedule differs')
                for mode in modes:
                    guard(); arm=mode.lower()+'_seed'+str(seed); cursor=0
                    model=KinematicCovariateFNO(mode,seed).cuda()
                    if not all(torch.equal(v.detach().cpu(),reference[k]) for k,v in model.state_dict().items()):
                        raise ValueError('Paired initial weights differ')
                    count=sum(v.numel() for v in model.parameters())
                    if count!=4528394:
                        raise ValueError('Model capacity differs')
                    opt=config['optimizer']
                    optimizer=torch.optim.Adam(model.parameters(),lr=opt['lr'],betas=tuple(opt['betas']),eps=opt['eps'],weight_decay=opt['weight_decay'],fused=opt['fused'])
                    state['arms'][arm]=dict(seed=seed,mode=mode,parameters=count,updates=0,status='qualification')
                    state['phase']=arm+'/qualification'; persist()
                    saved_rng=rng(); qt=(time.time(),time.monotonic())
                    loss,leaf=backward_case(model,witness,cache.targets[witness['key']],scale,guard)
                    gradients={}
                    for name,module in [('encoder',model.encoder),('operator',model.operator),('head',model.head)]:
                        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in module.parameters()):
                            raise ArithmeticError('Incomplete full-model gradients')
                        norm=sum(float(p.grad.double().square().sum()) for p in module.parameters())**.5
                        if norm<=0: raise ArithmeticError('Zero model gradient')
                        gradients[name]=norm
                    extra=float(model.head[0].weight.grad[:,76:].norm())
                    if mode=='ZERO' and extra!=0 or mode=='ACTUAL' and extra<=0:
                        raise ArithmeticError('Covariate control gradient differs')
                    optimizer.step()
                    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                        raise FloatingPointError('Nonfinite Adam qualification step')
                    model.load_state_dict(reference)
                    optimizer.state.clear(); optimizer.zero_grad(set_to_none=True); restore_rng(saved_rng)
                    if not all(torch.equal(v.detach().cpu(),reference[k]) for k,v in model.state_dict().items()):
                        raise ArithmeticError('Qualification changed initial state')
                    # Real model queries keep the coordinate and ACTUAL covariate paths.
                    with torch.no_grad(): context,latent=model.encode_case(witness)
                    selected=torch.linspace(0,len(witness['xyz'])-1,64,device='cuda').long()
                    xyz=witness['xyz'][selected].detach().clone().requires_grad_(True)
                    cov=witness['kinematic_inputs'][selected].detach().clone().requires_grad_(True)
                    ids,weights=stencil(xyz)
                    query=dict(xyz=xyz,kinematic_inputs=cov,features=witness['features'][selected],query_indices=ids,query_weights=weights)
                    prediction=model.query(query,0,64,context,latent)
                    xyzgrad,covgrad=torch.autograd.grad(prediction.square().sum(),(xyz,cov),allow_unused=True)
                    if xyzgrad is None or not bool(torch.isfinite(xyzgrad).all()) or float(xyzgrad.norm())<=0:
                        raise ArithmeticError('Actual coordinate gradient missing')
                    if mode=='ACTUAL' and (covgrad is None or not bool(torch.isfinite(covgrad).all()) or float(covgrad.norm())<=0):
                        raise ArithmeticError('Actual physical-input gradient missing')
                    with torch.no_grad():
                        alternate=torch.cat([model.query(query,start,min(start+17,64),context,latent) for start in range(0,64,17)])
                        norm=float(prediction.norm()); discrepancy=float((alternate-prediction).norm())
                        matched=discrepancy<=config['query_chunk_relative_tolerance']*norm if norm>0 else float((alternate-prediction).abs().max())<=config['query_chunk_zero_absolute_tolerance']
                        if not matched:
                            raise ArithmeticError('Actual query chunk mismatch')
                        largest=max((cache.cases[key] for key in dev_keys),key=lambda case:len(case['xyz']))
                        dc,dl=model.encode_case(largest)
                        for start in range(0,len(largest['xyz']),32768):
                            guard(); value=model.query(largest,start,min(start+32768,len(largest['xyz'])),dc,dl)
                            if not bool(torch.isfinite(value).all()): raise FloatingPointError('Maximum development geometry query failed')
                    model.zero_grad(set_to_none=True); restore_rng(saved_rng)
                    state['arms'][arm]['qualification']=dict(original_nodes=len(witness['xyz']),loss=loss,leaf_gradients=leaf,
                                                            parameter_gradients=gradients,extra_column_gradient=extra,
                                                            maximum_development_nodes=len(largest['xyz']),coordinate_gradient=float(xyzgrad.norm()),
                                                            physical_input_gradient=float(covgrad.norm()) if covgrad is not None else 0.,
                                                            query_chunk_difference=discrepancy,
                                                            Adam_step_restored=True,seconds=max(time.time()-qt[0],time.monotonic()-qt[1]))
                    del context,latent,xyz,cov,ids,weights,query,prediction,alternate,xyzgrad,covgrad,dc,dl,value
                    state['phase']=arm+'/training'; tick=(time.time(),time.monotonic()); persist()
                    for key in order:
                        guard()
                        if max(time.time()-tick[0],time.monotonic()-tick[1])>=config['arm_seconds']-10:
                            raise TimeoutError('Fixed arm training budget')
                        optimizer.zero_grad(set_to_none=True)
                        loss,_=backward_case(model,cache.cases[key],cache.targets[key],scale,guard)
                        state['optimizer_step_in_progress']=True
                        optimizer.step(); cursor+=1
                        state['optimizer_step_in_progress']=False
                        row=state['arms'][arm]
                        row.update(updates=cursor,last_key=key,last_loss=loss,
                                   training_seconds=dict(UTC=time.time()-tick[0],monotonic=time.monotonic()-tick[1]))
                        if cursor%config['checkpoint_updates']==0:
                            checkpoint(); persist()
                    state['arms'][arm]['status']='trained'; checkpoint(); persist()
                    state['phase']=arm+'/development_predictions'; model.eval(); predict_tick=(time.time(),time.monotonic())
                    state['predictions'][arm]={}
                    with torch.no_grad():
                        for key in dev_keys:
                            guard(); case=cache.cases[key]; context,latent=model.encode_case(case); chunks=[]
                            for start in range(0,len(case['xyz']),32768):
                                guard(); stop=min(start+32768,len(case['xyz']))
                                chunks.append(decode(model.query(case,start,stop,context,latent),scale).cpu().numpy())
                            value=np.concatenate(chunks)
                            if value.shape!=(len(case['xyz']),4) or not np.isfinite(value).all():
                                raise ValueError('Development prediction incomplete')
                            path=output/arm/'predictions'/(key.replace('/','_')+'.npy'); save_prediction(path,value)
                            state['predictions'][arm][key]=str(path.resolve()); state['committed_predictions']+=1
                        state['arms'][arm]['prediction_seconds']=dict(UTC=time.time()-predict_tick[0],monotonic=time.monotonic()-predict_tick[1])
                    persist(); model=None; optimizer=None; gc.collect(); torch.cuda.empty_cache()
                del reference
            if state['committed_predictions']!=config['prediction_denominator'] or len(state['predictions'])!=6:
                raise ValueError('All-six prediction barrier incomplete')
            state['phase']='shared_development_truth'; persist()
            state['metrics']={name:[] for name in state['predictions']}
            for key in dev_keys:
                guard(); item=catalog['development_truth'][key]
                truth=np.load(local_path(item['path']),allow_pickle=False); state['development_GT_reads']+=1
                if truth.dtype!=np.float32 or list(truth.shape)!=item['shape'] or not np.isfinite(truth).all():
                    raise ValueError('Original signed development fields differ')
                unorm=np.linalg.norm(truth[:,:3]); snorm=np.linalg.norm(truth[:,3])
                if unorm==0 or snorm==0: raise ValueError('Undefined GT norm')
                for name,paths in state['predictions'].items():
                    prediction=np.load(paths[key],allow_pickle=False)
                    if prediction.shape!=truth.shape or not np.isfinite(prediction).all():
                        raise ValueError('Saved full-node prediction differs')
                    U=float(np.linalg.norm(prediction[:,:3]-truth[:,:3])/unorm)*100
                    S=float(np.linalg.norm(prediction[:,3]-truth[:,3])/snorm)*100
                    state['metrics'][name].append(dict(key=key,U=U,S=S,primary=.5*(U+S)))
            state['aggregate']={name:{field:sum(row[field] for row in rows)/60 for field in ['U','S','primary']} for name,rows in state['metrics'].items()}
            differences=[{field:state['aggregate']['zero_seed'+str(seed)][field]-state['aggregate']['actual_seed'+str(seed)][field] for field in ['U','S','primary']} for seed in config['seeds']]
            means={field:sum(row[field] for row in differences)/3 for field in ['U','S','primary']}
            gate=config['practical_benefit']
            state.update(paired_improvement_pp=means,paired_primary_seed_wins=sum(row['primary']>0 for row in differences),
                         practical_benefit_passed=means['primary']>=gate['mean_primary_reduction_pp'] and means['S']>=gate['mean_stress_reduction_pp'] and sum(row['primary']>0 for row in differences)>=gate['primary_seed_wins'],
                         status='completed',phase='terminal')
    except Exception as error:
        state['possible_partial_optimizer_update']=state['optimizer_step_in_progress']
        state.update(status='failed',phase='failure',error=type(error).__name__+': '+str(error))
        raise
    finally:
        if model is not None and optimizer is not None:
            checkpoint()
        state.update(active=False,closed=True); persist()
        (output/'summary.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['config','source','output','lock']: parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
