"""One maximum-N real native++ witness with resident TRAIN100; no scoring."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
import json
import os
from pathlib import Path
import signal


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.pp100_feasibility_contract_v10 import approved_contract
    plan,_,manifest,keys=approved_contract(plan_path);science=plan['scientific']
    outer=json.loads(os.environ['GEOBC_PP100_FEASIBILITY_GUARD'])
    assert outer['scientific_digest']==plan['scientific_digest']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==outer['boot_id']
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    output=d.path(plan['output']);assert output.exists() and not (output/'execution.json').exists()
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=plan['scientific_digest'])
    d.write(output/'execution.json',execution)
    result=dict(status='running',active=True,closed=False,phase='cache_setup',execution=execution,
        update_attempts=0,completed_updates=0,update_in_progress=False,possible_partial_update=False,
        observations=[],DEV_FINAL_GT_reads=0,raw_CSV_H5_reads=0,scaler_fits=0,
        model_retained_for_accuracy=False,peak_allocated_bytes=0,peak_reserved_bytes=0,
        peak_total_device_used_bytes_sampled=0,automatic_replay=False,physical_accuracy_claim=False)
    lock=None;torch=None;cache=None;cap=science['memory_limit_bytes']
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:
            raise TimeoutError('Original dual-clock feasibility deadline')
        if torch is not None and torch.cuda.is_initialized():
            result['peak_allocated_bytes']=max(result['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
            result['peak_reserved_bytes']=max(result['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
            free,total=torch.cuda.mem_get_info()
            result['peak_total_device_used_bytes_sampled']=max(result['peak_total_device_used_bytes_sampled'],total-free)
            if max(result[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))>cap:
                raise MemoryError('Actual usage exceeds30.5GB')
    def persist():
        result.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(output/'status.json',result)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External feasibility timeout')))
    try:
        guard()
        import fcntl
        lock=d.path(science['GPU_lock']).open('a+')
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch as torch_module
        torch=torch_module
        from geobc.train_resident_cache import TRAINResidentCache
        from geobc.macro_training import build_solver,full_case_backward,profile
        import random
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        total=torch.cuda.get_device_properties(0).total_memory;assert cap<total
        torch.cuda.set_per_process_memory_fraction(cap/total);torch.cuda.reset_peak_memory_stats()
        random.seed(science['seed']);np.random.seed(science['seed'])
        torch.manual_seed(science['seed']);torch.cuda.manual_seed_all(science['seed'])
        cache=TRAINResidentCache(manifest,100,'PP256_NATIVE',guard=guard,memory_limit_bytes=cap)
        result['cache_metadata']=cache.metadata_profile();persist()
        cache.load();result['cache_receipt']=dict(cache.receipt)
        assert cache.receipt['loaded_cases']==300
        model=build_solver('PP256_NATIVE',seed=2026).train();result['model_profile']=profile(model)
        recipe=science['optimizer']
        optimizer=torch.optim.Adam(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'],betas=tuple(recipe['betas']),eps=recipe['eps'],fused=recipe['fused'])
        result['cache_setup_epoch_seconds']=time.time()-ENTRY_EPOCH
        result['phase']='maximum_N_gradient_timing';persist()
        case=cache.case(science['witness_key'])
        for index in range(3):
            guard();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
            began_epoch,began_monotonic=time.time(),time.monotonic()
            loss,leaves=full_case_backward(model,case,keys,guard)
            missing=[];groups=dict(encoder=0.,operator=0.,head=0.)
            for name,param in model.named_parameters():
                if param.grad is None:missing.append(name);continue
                assert torch.isfinite(param.grad).all(),'Nonfinite actual parameter gradient'
                group='encoder' if name.startswith('encoder.') else 'operator' if name.startswith('operator.') else 'head'
                groups[group]+=float(param.grad.detach().double().square().sum())
            assert sorted(missing)==['operator.backbone.embedding.bias','operator.backbone.embedding.weight']
            assert all(value>0 for value in groups.values())
            assert leaves['original_nodes_used']==325387 and not leaves['node_sampling']
            assert leaves['geometry_context_L2']>0 and leaves['operator_latent_L2']>0
            guard();result['update_attempts']+=1;result['update_in_progress']=True;persist()
            optimizer.step();torch.cuda.synchronize()
            assert all(torch.isfinite(param).all() for param in model.parameters())
            result['completed_updates']+=1;result['update_in_progress']=False
            result['observations'].append(dict(index=index+1,original_nodes=325387,encoded_TRAIN_MSE=loss,
                epoch_seconds=time.time()-began_epoch,monotonic_seconds=time.monotonic()-began_monotonic,
                group_gradient_L2={k:v**.5 for k,v in groups.items()},leaf_gradients=leaves,allowed_unused=missing))
            guard();persist()
        result['actual_largest_fullN_gradient_passed']=True
        result['status']='completed';result['feasibility_qualified']=True
    except BaseException as error:
        result['status']='failed';result['error_type']=type(error).__name__;result['error_message']=str(error)
        result['possible_partial_update']=result['update_in_progress'];result['feasibility_qualified']=False
        raise
    finally:
        if cache is not None:result['cache_receipt']=dict(cache.receipt)
        result.update(active=False,closed=True)
        persist();d.write(output/'summary.json',result)
        if lock is not None:lock.close()
    print(dict(status=result['status'],elapsed_seconds=result['elapsed_epoch_seconds'],completed_updates=result['completed_updates'],peak_allocated_bytes=result['peak_allocated_bytes'],peak_reserved_bytes=result['peak_reserved_bytes']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');parser.add_argument('--plan')
    args=parser.parse_args()
    if args.start:
        if not args.plan:parser.error('--start requires a released plan')
        run(args.plan)
    else:print('{"status":"source_only","data_reads":0,"GPU_execution":0}')
