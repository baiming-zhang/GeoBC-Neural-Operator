"""Explicit one-shot two-solver parent continuation to matched20 exposures."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
import json
import os
import signal
from pathlib import Path


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.matched20_solver_contract_v78 import approved_contract
    p,r,m,order,parents=approved_contract(plan_path);s=p['scientific'];keys=s['TRAIN_keys']
    outer=json.loads(os.environ['GEOBC_MATCHED20_V78_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest'] and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    o=d.path(p['output']);assert o.exists() and not (o/'execution.json').exists()
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,boot_id=outer['boot_id'],scientific_digest=p['scientific_digest'])
    d.write(o/'execution.json',execution)
    a=dict(status='running',active=True,closed=False,phase='cache_setup',execution=execution,arms={},
        additional_optimizer_step_attempts=0,additional_optimizer_step_completions=0,optimizer_step_in_progress=False,
        possible_partial_optimizer_update=False,raw_GT_reads=0,DEV_FINAL_reads=0,scaler_fits=0,
        encoded_target_reads=0,resident_physical_target_reuses=0,parent_checkpoint_hash_attempts=0,parent_checkpoint_hash_reads=0,
        parent_checkpoint_load_attempts=0,parent_checkpoint_loads=0,cache_receipt=None,
        peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0,
        backend_external_workspace_exact_peak_known=False,automatic_replay=False,automatic40=False)
    torch=None;cache=None;lock=None
    cache_deadline=(ENTRY_EPOCH+s['cache_setup_budget_seconds'],ENTRY_MONOTONIC+s['cache_setup_budget_seconds'])
    training_deadline=None;readout_deadline=None
    def guard():
        limits=[(outer['deadline_epoch'],outer['deadline_monotonic'])]
        limits += [z for z in (cache_deadline,training_deadline,readout_deadline) if z is not None]
        if min(min(x-time.time(),y-time.monotonic()) for x,y in limits)<=0:
            raise TimeoutError('Frozen earlier epoch/monotonic deadline')
        if torch is not None and torch.cuda.is_initialized():
            a['peak_allocated_bytes']=max(a['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
            a['peak_reserved_bytes']=max(a['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
            free,total=torch.cuda.mem_get_info();a['peak_total_device_used_bytes_sampled']=max(a['peak_total_device_used_bytes_sampled'],total-free)
            if max(a[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))>s['memory_limit_bytes']:
                raise MemoryError('30.5GB allocated/reserved or sampled total-device cap exceeded')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        if cache is not None:a['cache_receipt']=dict(cache.receipt)
        d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External guard terminated child')))
    try:
        guard()
        import fcntl
        lock=d.path(s['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import random
        import torch as torch_module
        torch=torch_module
        from geobc.physical_target_only_cache_v62 import PhysicalTargetOnlyCache
        from geobc.physical_field_loss import PhysicalScale,full_common_case_backward
        from geobc.fno_stress_specialist_v54 import tree_digest
        from geobc.common_grid_solver import parameter_gradients
        from geobc.coverage_comparison_train import physical_metrics,aggregate
        from geobc.benchmark import FieldScaler
        from geobc.transolver_decoder_contract_v64 import complete_physical
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device=torch.cuda.current_device();total=torch.cuda.get_device_properties(device).total_memory
        assert s['memory_limit_bytes']<total
        torch.cuda.set_per_process_memory_fraction(s['memory_limit_bytes']/total,device);torch.cuda.reset_peak_memory_stats()
        def event(stage,name,identity):
            guard()
            if stage=='completed' and (cache.receipt['input_array_reads']+cache.receipt['physical_TRAIN_target_reads'])%50==0:persist()
        cache=PhysicalTargetOnlyCache(m,100,'FNO_GLOBAL_HASH',memory_limit_bytes=s['memory_limit_bytes'],guard=guard,on_read=event)
        persist();cache.load();persist()
        assert cache.receipt['physical_TRAIN_target_reads']==300 and cache.receipt['encoded_TRAIN_target_reads']==0
        a.update(cache_setup_epoch_seconds=time.time()-ENTRY_EPOCH,cache_setup_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(o/'cache_index.json',dict(inputs=cache.specs,physical=cache.physical_specs))
        cache_deadline=None
        scale=PhysicalScale.from_mapping(s['scaler']);scaler=FieldScaler(**s['scaler'])
        witness=max(keys,key=lambda k:m['records'][k]['num_nodes'])
        def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
        def restore(state):
            random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
        with (o/'loss.jsonl').open('x',encoding='utf-8') as log:
            for arm in s['arms']:
                name=arm['name'];family=arm['family'];parent=parents[name];arm_epoch,arm_mono=time.time(),time.monotonic()
                a.update(phase='parent_restore',current_arm=name);persist();guard()
                item=arm['parent_checkpoint_identity'];d.stat_matches(item)
                a['parent_checkpoint_hash_attempts']+=1;persist()
                assert d.identity(item['path'],hashed=True)==item
                a['parent_checkpoint_hash_reads']+=1;a['parent_checkpoint_load_attempts']+=1;persist()
                saved=torch.load(d.path(item['path']),map_location='cpu',weights_only=False)
                a['parent_checkpoint_loads']+=1;d.stat_matches(item);guard()
                assert saved['scientific_digest']==arm['parent_SCI'] and saved['family']==family
                cursor=saved['step'] if 'step' in saved else saved['optimizer_step']
                assert cursor==arm['parent_step'] and saved.get('cursor',cursor)==cursor
                assert saved['case_exposure_counts']==arm['parent_case_exposure_counts'] and saved['scaler']==s['scaler'] and saved['dataset_identity']==s['dataset_identity']
                parent_order=saved['full_order_digest'] if 'full_order_digest' in saved else saved['order_digest']
                assert parent_order==d.digest(order[:cursor]) and set(saved['rng'])=={'python','numpy','torch','cuda'}
                if family=='FNO_GLOBAL_HASH':
                    from geobc.operator_hash_metadata_v6 import build_solver
                    model=build_solver(family,seed=s['seed']).train()
                    baseline=parent['summary']['endpoints']['3000']['full_TRAIN']
                else:
                    from geobc.transolver_bf16_hash_v65 import build_solver
                    model=build_solver(family,seed=s['seed']).train()
                    baseline=parent['summary']['aggregate_endpoints'][family]['5']
                    assert saved['diagnostic_selected_keys']==keys and saved['shared_fullTRAIN_model']
                assert model.backend=='tcnn_fully_fused'
                model.load_state_dict(saved['model'],strict=True)
                assert tree_digest(model.state_dict())==tree_digest(saved['model'])
                recipe=arm['parent_optimizer'];lr=recipe['learning_rate']
                optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=tuple(recipe['betas']),eps=recipe['eps'],weight_decay=recipe['weight_decay'],fused=recipe['fused'])
                optimizer.load_state_dict(saved['optimizer'])
                assert optimizer.state and all(int(v['step'])==cursor for v in optimizer.state.values())
                assert all(g['lr']==lr and g['weight_decay']==recipe['weight_decay'] and tuple(g['betas'])==tuple(recipe['betas']) and g['eps']==recipe['eps'] and g['fused']==recipe['fused'] for g in optimizer.param_groups)
                assert tree_digest(optimizer.state_dict())==tree_digest(saved['optimizer'])
                restore(saved['rng']);assert tree_digest(rng())==tree_digest(saved['rng'])
                state=dict(status='running',step=cursor,case_exposure_counts=dict(saved['case_exposure_counts']),endpoints={},
                    restored_model_Adam_fourRNG_cursor_exposure=True,loaded_model_digest=tree_digest(model.state_dict()),loaded_Adam_digest=tree_digest(optimizer.state_dict()),loaded_RNG_digest=tree_digest(rng()),
                    parameter_count=sum(v.numel() for v in model.parameters()),precision=arm['precision'],parent_checkpoint_identity=item,
                    parent_terminal_binding=parent['binding'],parent_native_cost_identity=d.identity(d.path(parent['plan']['output'])/'summary.json',hashed=True),
                    parent_external_cost=dict(epoch_seconds=parent['external']['elapsed_epoch_seconds'],monotonic_seconds=parent['external']['elapsed_monotonic_seconds']),
                    earlier_failures_or_diagnostics_not_replayed=True,interstage_wait_not_zeroed_or_charged_as_training=True,
                    additional_updates_completed=0,optimizer_step_in_progress=False,arm_entry_epoch=arm_epoch,arm_entry_monotonic=arm_mono)
                a['arms'][name]=state;del saved;persist()
                md,ad,rd=state['loaded_model_digest'],state['loaded_Adam_digest'],state['loaded_RNG_digest'];saved_rng=rng()
                began=(time.time(),time.monotonic());optimizer.zero_grad(set_to_none=True)
                case=cache.case(witness)
                loss,leaves=full_common_case_backward(model,case,cache.physical_targets[witness],scale,cache.case_norms[witness],guard)
                groups=parameter_gradients(model)
                assert all(z['finite'] and not z['missing'] and z['L2']>0 for z in groups.values()) and all(v>0 for v in leaves.values())
                optimizer.zero_grad(set_to_none=True)
                model.eval()
                with torch.no_grad():
                    context,latent=model.encode_case(case);pieces=[]
                    for start in range(0,len(case['xyz']),32768):
                        guard();pieces.append(model.query(case,start,min(start+32768,len(case['xyz'])),context,latent))
                    encoded=torch.cat(pieces).cpu().numpy()
                    physical=scaler.decode(encoded)
                    measured=physical_metrics(np,cache.physical_targets[witness].detach().cpu().numpy(),physical)
                    a['resident_physical_target_reuses']+=1
                    reference=baseline['per_case'][witness]
                    for metric in ('u_relative_l2','stress_relative_l2','stress_hotspot_relative_l2'):
                        assert abs(measured[metric]-reference[metric])<=.003,'Restored witness physical metric differs'
                    del context,latent,pieces,encoded,physical
                model.train();restore(saved_rng)
                assert tree_digest(model.state_dict())==md and tree_digest(optimizer.state_dict())==ad and tree_digest(rng())==rd
                state['qualification']=dict(passed=True,key=witness,nodes=len(case['xyz']),groups=groups,leaf_gradients=leaves,model_unchanged=True,Adam_unchanged=True,fourRNG_restored=True,
                    restored_witness_metric=measured,parent_witness_metric=reference,declared_metric_absolute_tolerance=.003,
                    epoch_seconds=time.time()-began[0],monotonic_seconds=time.monotonic()-began[1])
                folder=o/name;folder.mkdir(exist_ok=False)
                def save(step,filename):
                    guard();assert all(int(v['step'])==step for v in optimizer.state.values())
                    tmp=folder/(filename+'.tmp')
                    torch.save(dict(step=step,cursor=step,family=family,scientific_digest=p['scientific_digest'],model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng(),
                        full_order_digest=p['full_order_digest'],case_exposure_counts=dict(state['case_exposure_counts']),dataset_identity=s['dataset_identity'],scaler=s['scaler'],
                        execution=execution,native_origin_clocks=dict(epoch=ENTRY_EPOCH,monotonic=ENTRY_MONOTONIC),cost_status=dict(a)),tmp)
                    os.replace(tmp,folder/filename);state['last_checkpoint']=dict(step=step,identity=d.identity(folder/filename));guard()
                def readout(step):
                    nonlocal readout_deadline
                    read_epoch,read_mono=time.time(),time.monotonic()
                    readout_deadline=(read_epoch+s['TRAIN_readout_budget_seconds_each'],read_mono+s['TRAIN_readout_budget_seconds_each'])
                    saved_state=rng();model.eval();a['phase']='TRAIN_readout';persist()
                    try:
                        rows={k:None for k in keys}
                        with torch.no_grad():
                            for key in keys:
                                guard();one=cache.case(key);context,latent=model.encode_case(one);pieces=[]
                                for start in range(0,len(one['xyz']),32768):
                                    guard();pieces.append(model.query(one,start,min(start+32768,len(one['xyz'])),context,latent))
                                physical=scaler.decode(torch.cat(pieces).cpu().numpy())
                                rows[key]=physical_metrics(np,cache.physical_targets[key].detach().cpu().numpy(),physical)
                                a['resident_physical_target_reuses']+=1
                                del context,latent,pieces,physical
                        metric=aggregate(rows,300);complete_physical(metric,keys)
                        state['endpoints'][str(step)]=dict(full_TRAIN=metric,fixed_passes=step//300,step=step,
                            readout_epoch_seconds=time.time()-read_epoch,readout_monotonic_seconds=time.monotonic()-read_mono)
                    finally:restore(saved_state);model.train();readout_deadline=None
                    a['phase']='training';guard();persist()
                state.update(training_entry_epoch=time.time(),training_entry_monotonic=time.monotonic())
                training_deadline=(state['training_entry_epoch']+arm['training_budget_seconds'],state['training_entry_monotonic']+arm['training_budget_seconds'])
                a['phase']='training';persist()
                endpoints={300*passes for passes in arm['TRAIN_endpoint_passes']}
                for step,key in enumerate(order[cursor:],cursor+1):
                    guard();optimizer.zero_grad(set_to_none=True)
                    loss,_=full_common_case_backward(model,cache.case(key),cache.physical_targets[key],scale,cache.case_norms[key],guard)
                    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
                    state['optimizer_step_in_progress']=True;a['optimizer_step_in_progress']=True;a['additional_optimizer_step_attempts']+=1
                    optimizer.step();torch.cuda.synchronize()
                    state.update(step=step,optimizer_step_in_progress=False,additional_updates_completed=state['additional_updates_completed']+1)
                    a.update(additional_optimizer_step_completions=a['additional_optimizer_step_completions']+1,optimizer_step_in_progress=False)
                    state['case_exposure_counts'][key]+=1
                    log.write(json.dumps(dict(arm=name,step=step,key=key,physical_relative_squared=loss,
                        training_epoch_seconds=time.time()-state['training_entry_epoch'],training_monotonic_seconds=time.monotonic()-state['training_entry_monotonic']))+'\n');log.flush()
                    guard()
                    if step%300==0:save(step,'last.pt')
                    if step in endpoints:
                        assert state['case_exposure_counts']=={k:step//300 for k in keys}
                        save(step,'step'+str(step)+'.pt')
                        if step==6000:
                            state.update(training_epoch_seconds=time.time()-state['training_entry_epoch'],training_monotonic_seconds=time.monotonic()-state['training_entry_monotonic']);training_deadline=None
                        readout(step);state['endpoints'][str(step)]['checkpoint_identity']=d.identity(folder/('step'+str(step)+'.pt'),hashed=True)
                    if step%30==0:
                        if step!=6000:state.update(training_epoch_seconds=time.time()-state['training_entry_epoch'],training_monotonic_seconds=time.monotonic()-state['training_entry_monotonic'])
                        persist()
                assert state['step']==6000 and state['case_exposure_counts']=={k:20 for k in keys}
                state.update(status='completed',arm_epoch_seconds=time.time()-arm_epoch,arm_monotonic_seconds=time.monotonic()-arm_mono,
                    training_interval_includes_intermediate_readout_and_checkpoints=True)
                d.stat_matches(item);del model,optimizer;torch.cuda.empty_cache();guard();persist()
        assert a['additional_optimizer_step_attempts']==a['additional_optimizer_step_completions']==7500
        assert a['parent_checkpoint_hash_attempts']==a['parent_checkpoint_hash_reads']==a['parent_checkpoint_load_attempts']==a['parent_checkpoint_loads']==2
        assert a['resident_physical_target_reuses']==902
        a.update(status='completed',phase='terminal',training_complete=True,protocol_qualified=True,
            separate_DEV_readout_required=True,TRAINgate_does_not_filter_endpoint=True,final_solver_selected=False)
    except BaseException as error:
        a.update(status='failed',phase='terminal',error_type=type(error).__name__,error=str(error),possible_partial_optimizer_update=a['optimizer_step_in_progress'])
        raise
    finally:
        if cache is not None:a['cache_receipt']=dict(cache.receipt)
        a.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(o/'summary.json',a);d.write(o/'status.json',a)
        if lock is not None:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');parser.add_argument('--plan')
    args=parser.parse_args()
    if args.start:
        if not args.plan:parser.error('--start requires frozen released --plan')
        run(args.plan)
    else:print('{"status":"source_only","scientific_IO":0,"GPU":0}')
