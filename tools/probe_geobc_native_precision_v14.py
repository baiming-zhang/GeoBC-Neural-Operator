"""Six actual TRAIN cases; same weights/RNG, no optimizer or development GT."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
import json
import os
from pathlib import Path
import signal


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.native_precision_contract_v14 import approved_contract
    plan,review,manifest,upstream=approved_contract(plan_path); s=plan['scientific']
    outer=json.loads(os.environ['GEOBC_NATIVE_PRECISION_V14_GUARD'])
    assert outer['scientific_digest']==plan['scientific_digest']
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==outer['boot_id']
    out=d.path(plan['output']); assert out.exists() and not (out/'execution.json').exists()
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=plan['scientific_digest'])
    d.write(out/'execution.json',execution)
    result=dict(status='running',active=True,closed=False,execution=execution,phase='loading',
        checkpoint_reads=0,input_array_reads=0,TRAIN_physical_array_reads=0,weight_updates=0,
        DEV_FINAL_GT_reads=0,scaler_fits=0,cases={},full_backward_attempts=0,full_backward_completions=0,
        prediction_attempts=0,completed_prediction_modes=0,peak_allocated_bytes=0,peak_reserved_bytes=0,
        peak_total_device_used_bytes_sampled=0,automatic_replay=False,qualified_TF32=False)
    lock=None; torch=None

    def guard():
        assert min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())>0,'Fixed dual deadline'
        if torch is not None and torch.cuda.is_initialized():
            for key,value in [('peak_allocated_bytes',torch.cuda.max_memory_allocated()),('peak_reserved_bytes',torch.cuda.max_memory_reserved())]:
                result[key]=max(result[key],value)
            free,total=torch.cuda.mem_get_info();result['peak_total_device_used_bytes_sampled']=max(result['peak_total_device_used_bytes_sampled'],total-free)
            assert max(result[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))<=s['memory_limit_bytes']

    def persist():
        result.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(out/'status.json',result)

    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External guard termination')))
    try:
        import fcntl
        lock=d.path(s['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        guard()
        import numpy as np
        import torch as torch_module
        torch=torch_module
        import hashlib
        import random
        import math
        from types import SimpleNamespace
        from geobc.macro_training import build_solver,encode_case,query_case
        from geobc.prospective_dataset_v2 import load_case
        from geobc.physical_field_loss import PhysicalScale,norms_from_complete_physical_target,decode_physical
        from geobc.train_native_physical_cache_v13 import full_physical_backward
        from geobc.transformers import close_owned_process_group
        torch.set_num_threads(4)
        result.update(torch_version=torch.__version__,CUDA_runtime_version=torch.version.cuda,device_name=torch.cuda.get_device_name(0),kernel_dtype_profiled=False,
            precision_scope='TF32 permission policy only, not profiler evidence of actual kernel datatype or trained-model accuracy')
        total=torch.cuda.get_device_properties(0).total_memory
        assert s['memory_limit_bytes']<total
        torch.cuda.set_per_process_memory_fraction(s['memory_limit_bytes']/total);torch.cuda.reset_peak_memory_stats()
        endpoint=s['checkpoint'];assert d.identity(endpoint['path'],hashed=True)==endpoint
        saved=torch.load(d.path(endpoint['path']),map_location='cpu',weights_only=False);result['checkpoint_reads']+=1
        assert saved['step']==saved['cursor']==1500 and saved['scientific_digest']==s['native_TRAIN_SCI']
        assert saved['family']=='PP256_NATIVE' and saved['scaler']==manifest['scaler']
        assert all(int(v['step'])==1500 for v in saved['optimizer']['state'].values())
        owned_rng=saved['rng'];model=build_solver('PP256_NATIVE',seed=2026);model.load_state_dict(saved['model'],strict=True);del saved
        scale=PhysicalScale.from_mapping(manifest['scaler']); cache=SimpleNamespace(completed=False,physical_targets={},case_norms={})
        cases={}
        def input_event(stage,key,member,identity):
            assert member not in ('target','physical');guard()
            if stage=='opened':result['input_array_reads']+=1
        for key in s['case_keys']:
            guard();case=load_case(manifest,key,with_targets=False,train_CAD_count=100,device='cuda',for_fno=False,on_read=input_event)
            assert case['scientific_group']=='TRAIN' and not {'target','physical'}&set(case)
            item=manifest['records'][key]['file_identities']['physical_path'];assert d.identity(item['path'],hashed=True)==item
            value=np.load(d.path(item['path']),mmap_mode='r',allow_pickle=False);result['TRAIN_physical_array_reads']+=1
            assert value.shape==(len(case['xyz']),4) and str(value.dtype)=='float32' and np.isfinite(value).all()
            target=torch.as_tensor(np.array(value,copy=True),device='cuda');del value
            cache.physical_targets[key]=target;cache.case_norms[key]=norms_from_complete_physical_target(target);cases[key]=case;persist()
        assert result['TRAIN_physical_array_reads']==6;cache.completed=True
        def state_hash():
            digest=hashlib.sha256()
            for name,value in sorted(model.state_dict().items()):
                digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()
        before=state_hash();result['state_before_sha256']=before
        def restore_rng():
            random.setstate(owned_rng['python']);np.random.set_state(owned_rng['numpy']);torch.set_rng_state(owned_rng['torch']);torch.cuda.set_rng_state_all(owned_rng['cuda'])
        def precision(name):
            mode=s['precision_modes'][name];torch.set_float32_matmul_precision(mode['float32_matmul_precision'])
            torch.backends.cuda.matmul.allow_tf32=mode['matmul_allow_tf32'];torch.backends.cudnn.allow_tf32=mode['cudnn_allow_tf32']
        def gradients():
            groups={'all':[],'encoder':[],'operator':[],'head':[]};missing=[]
            for name,p in model.named_parameters():
                if p.grad is None:missing.append(name);continue
                assert torch.isfinite(p.grad).all()
                value=p.grad.detach().reshape(-1).clone();groups['all'].append(value)
                group='encoder' if name.startswith('encoder.') else 'operator' if name.startswith('operator.') else 'head';groups[group].append(value)
            assert sorted(missing)==['operator.backbone.embedding.bias','operator.backbone.embedding.weight']
            assert all(groups.values())
            return {g:torch.cat(values) for g,values in groups.items()}
        def predictions(case):
            model.eval()
            with torch.no_grad():
                context,latent=encode_case(model,case);pieces=[]
                for start in range(0,len(case['xyz']),32768):
                    guard();stop=min(start+32768,len(case['xyz']));pieces.append(decode_physical(query_case(model,case,start,stop,context,latent),scale))
                value=torch.cat(pieces);assert torch.isfinite(value).all()
            return value
        result['phase']='paired_fullN_probe';persist()
        for index,key in enumerate(s['case_keys']):
            case=cases[key];outputs={};order=['IEEE_FP32','TF32'] if index%2==0 else ['TF32','IEEE_FP32']
            for mode in order:
                guard();precision(mode);model.train();restore_rng();model.zero_grad(set_to_none=True)
                result['full_backward_attempts']+=1;persist()
                full_physical_backward(model,case,cache,scale,s['case_keys'],guard);result['full_backward_completions']+=1;torch.cuda.synchronize()
                timings=[];gradient=None;loss=None
                for repeat in range(3):
                    guard();restore_rng();model.zero_grad(set_to_none=True);model.train();torch.cuda.synchronize()
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    result['full_backward_attempts']+=1;persist()
                    epoch,mono=time.time(),time.monotonic();begin.record()
                    loss,_=full_physical_backward(model,case,cache,scale,s['case_keys'],guard)
                    result['full_backward_completions']+=1;end.record();torch.cuda.synchronize()
                    timings.append(dict(GPU_seconds=begin.elapsed_time(end)/1000,epoch_seconds=time.time()-epoch,monotonic_seconds=time.monotonic()-mono))
                    g=gradients()
                    if gradient is None:gradient=g
                    else:del g
                restore_rng();result['prediction_attempts']+=1;persist();prediction=predictions(case);result['completed_prediction_modes']+=1
                outputs[mode]=dict(timings=timings,gradient=gradient,prediction=prediction,objective=loss)
                assert state_hash()==before,'Parameters or buffers mutated in fixedweight probe'
            a=outputs['IEEE_FP32'];b=outputs['TF32'];drift={};gstats={}
            for field,part in [('U',slice(0,3)),('S',slice(3,4))]:
                den=a['prediction'][:,part].square().sum();assert float(den)>0
                drift[field]=float(((b['prediction'][:,part]-a['prediction'][:,part]).square().sum()/den).sqrt())
            for group in a['gradient']:
                av=a['gradient'][group].double();bv=b['gradient'][group].double();an=av.norm();bn=bv.norm();assert float(an)>0 and float(bn)>0
                gstats[group]=dict(cosine=float(torch.dot(av,bv)/(an*bn)),relative_L2_drift=float((bv-av).norm()/an))
            assert a['objective']>0 and math.isfinite(a['objective']) and math.isfinite(b['objective'])
            objective_drift=abs(b['objective']-a['objective'])/a['objective']
            medians={m:sorted(t['GPU_seconds'] for t in outputs[m]['timings'])[1] for m in outputs};assert all(v>0 for v in medians.values())
            row=dict(nodes=len(case['xyz']),precision_order=order,physical_prediction_drift=drift,gradient_groups=gstats,
                objective_relative_drift=objective_drift,GPU_training_speedup=medians['IEEE_FP32']/medians['TF32'],
                modes={m:dict(timings=v['timings'],objective=v['objective']) for m,v in outputs.items()},state_hash_unchanged=True)
            result['cases'][key]=row;persist();del outputs,a,b,av,bv,gradient,prediction,gstats
        assert set(result['cases'])==set(s['case_keys'])
        assert result['full_backward_attempts']==result['full_backward_completions']==48
        assert result['prediction_attempts']==result['completed_prediction_modes']==12
        assert result['checkpoint_reads']==1 and result['TRAIN_physical_array_reads']==6
        guard()
        threshold=s['acceptance'];rows=list(result['cases'].values())
        speed=math.exp(sum(math.log(r['GPU_training_speedup']) for r in rows)/6)
        numerical=all(r['physical_prediction_drift']['U']<=threshold['maximum_each_physical_U_prediction_relative_drift'] and r['physical_prediction_drift']['S']<=threshold['maximum_each_physical_S_prediction_relative_drift'] and r['objective_relative_drift']<=threshold['maximum_each_relative_objective_drift'] and all(g['cosine']>=threshold['minimum_each_gradient_group_cosine'] and g['relative_L2_drift']<=threshold['maximum_each_gradient_group_relative_L2_drift'] for g in r['gradient_groups'].values()) for r in rows)
        result.update(status='completed',phase='terminal',geometric_mean_GPU_training_speedup=speed,numerical_qualification=numerical,
            qualified_TF32=numerical and speed>=threshold['minimum_geometric_mean_GPU_training_speedup'],state_after_sha256=state_hash())
        assert result['state_after_sha256']==before
        close_owned_process_group()
    except Exception as error:
        result.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error),qualified_TF32=False)
        raise
    finally:
        result.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(out/'summary.json',result);d.write(out/'status.json',result)
        if lock is not None:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');parser.add_argument('--plan');args=parser.parse_args()
    if args.start:
        if not args.plan:parser.error('A separately released immutable --plan is required')
        run(args.plan)
    else:print('{"status":"source_only","execution":0}')
