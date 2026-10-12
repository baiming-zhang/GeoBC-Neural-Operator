"""Explicitly released, one-shot official UNO TRAIN9 screen."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
import json
import os
import signal
from pathlib import Path


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.official_uno_contract_v89 import approved_contract
    p,r,m,order,binding=approved_contract(plan_path);s=p['scientific'];keys=s['selected_TRAIN_keys']
    outer=json.loads(os.environ['GEOBC_OFFICIAL_UNO_V89_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest'] and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    o=d.path(p['output']);assert o.exists() and not (o/'execution.json').exists()
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,boot_id=outer['boot_id'],scientific_digest=p['scientific_digest'])
    d.write(o/'execution.json',execution)
    a=dict(status='running',active=True,closed=False,phase='cache_setup',execution=execution,arms={},
        optimizer_step_attempts=0,optimizer_step_completions=0,optimizer_step_in_progress=False,
        possible_partial_optimizer_update=False,raw_GT_reads=0,DEV_FINAL_reads=0,scaler_fits=0,
        encoded_target_reads=0,parent_PT_reads=0,resident_physical_target_reuses=0,cache_receipt=None,
        peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0,
        backend_external_workspace_exact_peak_known=False,automatic_replay=False,automatic100=False)
    torch=None;cache=None;lock=None
    phase_deadline=(ENTRY_EPOCH+s['cache_setup_budget_seconds'],ENTRY_MONOTONIC+s['cache_setup_budget_seconds'])
    def guard():
        limits=[(outer['deadline_epoch'],outer['deadline_monotonic'])]
        if phase_deadline is not None:limits.append(phase_deadline)
        if min(min(x-time.time(),y-time.monotonic()) for x,y in limits)<=0:
            raise TimeoutError('Frozen earlier epoch/monotonic deadline')
        if torch is not None and torch.cuda.is_initialized():
            a['peak_allocated_bytes']=max(a['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
            a['peak_reserved_bytes']=max(a['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
            free,total=torch.cuda.mem_get_info()
            a['peak_total_device_used_bytes_sampled']=max(a['peak_total_device_used_bytes_sampled'],total-free)
            if max(a[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))>s['memory_limit_bytes']:
                raise MemoryError('30.5GB budget exceeded')
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
        from geobc.decoder_packages_v77 import full_backward,predict,query_qualification
        from geobc.official_uno_solver_v89 import build_solver,parameter_gradients
        from geobc.decoder_packages_cache_v77 import NineCaseCache
        from geobc.physical_field_loss import PhysicalScale
        from geobc.fno_stress_specialist_v54 import tree_digest
        from geobc.coverage_comparison_train import physical_metrics,aggregate
        from geobc.benchmark import FieldScaler
        from geobc.transolver_decoder_contract_v64 import complete_physical
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device=torch.cuda.current_device();total=torch.cuda.get_device_properties(device).total_memory
        assert s['memory_limit_bytes']<total
        torch.cuda.set_per_process_memory_fraction(s['memory_limit_bytes']/total,device);torch.cuda.reset_peak_memory_stats()
        a['phase']='model_setup_before_GT';persist();guard()
        early_model=build_solver(s['UNO_kwargs'],s['expected_original_model_digest'],s['expected_original_shared_body_digests'],s['seed']).train()
        assert early_model.actual_parameter_profile['total_real_degrees_of_freedom']<=s['real_parameter_DOF_limit']
        a['pre_GT_model_construction_passed']=True
        a['actual_UNO_config']=early_model.actual_UNO_config
        a['actual_parameter_profile']=early_model.actual_parameter_profile
        guard();persist()
        def event(stage,name,identity):
            if stage=='completed':persist()
        cache=NineCaseCache(m,keys,memory_limit_bytes=s['memory_limit_bytes'],guard=guard,on_read=event)
        persist();cache.load();persist()
        assert cache.receipt['loaded_cases']==9 and cache.receipt['physical_TRAIN_target_reads']==9
        assert cache.receipt['encoded_TRAIN_target_reads']==0 and cache.receipt['DEV_FINAL_target_reads']==0
        a.update(cache_setup_epoch_seconds=time.time()-ENTRY_EPOCH,cache_setup_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(o/'cache_index.json',dict(inputs=cache.specs,physical=cache.physical_specs))
        scale=PhysicalScale.from_mapping(s['scaler']);scaler=FieldScaler(**s['scaler'])
        witness=max(keys,key=lambda k:m['records'][k]['num_nodes'])
        def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
        def restore(state):
            random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
        initial_reference=None
        with (o/'loss.jsonl').open('x',encoding='utf-8') as log:
            for arm in [s['candidate']]:
                arm_epoch,arm_mono=time.time(),time.monotonic()
                phase_deadline=(arm_epoch+s['arm_budget_seconds'],arm_mono+s['arm_budget_seconds'])
                a.update(phase='arm_setup',current_arm=arm);persist();guard()
                random.seed(s['seed']);np.random.seed(s['seed']);torch.manual_seed(s['seed']);torch.cuda.manual_seed_all(s['seed'])
                model=early_model;early_model=None
                md=tree_digest(model.state_dict());initial_rng=rng();rd=tree_digest(initial_rng)
                initial=dict(model_digest=md,RNG_digest=rd,parameter_count=sum(v.numel() for v in model.parameters()),counts=model.counts())
                initial['original_initialization']=model.original_initialization
                initial['actual_UNO_config']=model.actual_UNO_config
                initial['parameter_profile']=model.actual_parameter_profile
                initial['shared_encoder_head_digests']={name:tree_digest(getattr(model,name).state_dict()) for name in ('encoder','head')}
                assert initial['shared_encoder_head_digests']==model.original_initialization['kept_encoder_head_digests']
                assert model.original_initialization['original_FNO_model_digest']==s['expected_original_model_digest']
                assert model.original_initialization['encoder_head_bytes_kept']
                recipe=s['optimizer']
                optimizer=torch.optim.Adam(model.parameters(),lr=recipe['lr'],betas=tuple(recipe['betas']),eps=recipe['eps'],weight_decay=recipe['weight_decay'],fused=False,foreach=False)
                ad=tree_digest(optimizer.state_dict())
                state=dict(status='running',step=0,case_exposure_counts={k:0 for k in keys},initialization=initial,
                    optimizer_implementation='torch.optim.Adam fusedFalse/foreachFalse; complex-safe upstreamUNO parameters, hyperparameters equal baseline',endpoints={},
                    arm_entry_epoch=arm_epoch,arm_entry_monotonic=arm_mono,optimizer_step_in_progress=False)
                a['arms'][arm]=state;persist()
                qualification_start=(time.time(),time.monotonic())
                optimizer.zero_grad(set_to_none=True)
                loss,leaves=full_backward(model,cache.case(witness),cache.physical_targets[witness],scale,cache.case_norms[witness],guard)
                gradients=parameter_gradients(model)
                assert all(v['finite'] and not v['missing'] and v['L2']>0 for v in gradients.values())
                assert all(v>0 for v in leaves.values())
                optimizer.zero_grad(set_to_none=True)
                query_check=query_qualification(model,cache.case(witness),guard)
                restore(initial_rng)
                assert tree_digest(model.state_dict())==md and tree_digest(optimizer.state_dict())==ad and tree_digest(rng())==rd
                state['qualification']=dict(passed=True,largest_case=witness,original_nodes=len(cache.case(witness)['xyz']),
                    physical_loss=loss,parameter_gradients=gradients,shared_leaf_gradients=leaves,
                    query_check=query_check,complex_safe_parameter_gradients=True,model_unchanged=True,Adam_unchanged=True,fourRNG_restored=True,
                    epoch_seconds=time.time()-qualification_start[0],monotonic_seconds=time.monotonic()-qualification_start[1])
                folder=o/arm;folder.mkdir(exist_ok=False)
                def save(step,name):
                    guard()
                    if step>0:assert optimizer.state and all(int(v['step'])==step for v in optimizer.state.values())
                    tmp=folder/(name+'.tmp')
                    torch.save(dict(step=step,cursor=step,arm=arm,package=arm,family='OFFICIAL_UNO',seed=s['seed'],scientific_digest=p['scientific_digest'],
                        model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng(),
                        full_order_digest=p['full_order_digest'],case_exposure_counts=dict(state['case_exposure_counts']),
                        dataset_identity=s['dataset_identity'],scaler=s['scaler'],execution=execution,
                        native_origin_clocks=dict(epoch=ENTRY_EPOCH,monotonic=ENTRY_MONOTONIC),official_UNO_config=model.actual_UNO_config,parameter_profile=model.actual_parameter_profile,cost_status=dict(a)),tmp)
                    os.replace(tmp,folder/name)
                    state['last_checkpoint']=dict(step=step,identity=d.identity(folder/name))
                    guard()
                def readout(passes):
                    nonlocal phase_deadline
                    prior=phase_deadline;began_epoch=time.time();began=time.monotonic()
                    limits=(began_epoch+s['readout_budget_seconds_per_endpoint'],began+s['readout_budget_seconds_per_endpoint'])
                    phase_deadline=tuple(min(x,y)for x,y in zip(prior,limits)) if prior is not None else limits
                    saved=rng();model.eval();a['phase']='TRAIN_readout';persist()
                    try:
                        rows={k:None for k in keys}
                        with torch.no_grad():
                            for key in keys:
                                guard();case=cache.case(key)
                                encoded=predict(model,case,guard).cpu().numpy()
                                physical=scaler.decode(encoded)
                                rows[key]=physical_metrics(np,cache.physical_targets[key].detach().cpu().numpy(),physical)
                                a['resident_physical_target_reuses']+=1
                                del encoded,physical
                        metric=aggregate(rows,9);complete_physical(metric,keys)
                        state['endpoints'][str(passes)]=dict(full_TRAIN9=metric,readout_monotonic_seconds=time.monotonic()-began,readout_epoch_seconds=time.time()-began_epoch,
                            fixed_passes=passes,step=passes*9)
                    finally:restore(saved);model.train();phase_deadline=prior;a['phase']='training' if passes<100 else 'final_readout_complete'
                    guard();persist()
                save(0,'step0.pt');readout(0)
                state['endpoints']['0']['checkpoint_identity']=d.identity(folder/'step0.pt',hashed=True)
                state.update(training_entry_epoch=time.time(),training_entry_monotonic=time.monotonic())
                a['phase']='training';persist()
                for step,key in enumerate(order,1):
                    guard();optimizer.zero_grad(set_to_none=True)
                    loss,_=full_backward(model,cache.case(key),cache.physical_targets[key],scale,cache.case_norms[key],guard)
                    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
                    state['optimizer_step_in_progress']=True;a['optimizer_step_in_progress']=True
                    a['optimizer_step_attempts']+=1
                    optimizer.step();torch.cuda.synchronize()
                    state.update(step=step,optimizer_step_in_progress=False)
                    a.update(optimizer_step_completions=a['optimizer_step_completions']+1,optimizer_step_in_progress=False)
                    state['case_exposure_counts'][key]+=1
                    log.write(json.dumps(dict(arm=arm,step=step,key=key,physical_relative_squared=loss,
                        training_epoch_seconds=time.time()-state['training_entry_epoch'],training_monotonic_seconds=time.monotonic()-state['training_entry_monotonic']))+'\n');log.flush()
                    guard()
                    if step%90==0:save(step,'last.pt')
                    if step in (90,900):
                        assert set(state['case_exposure_counts'].values())=={step//9}
                        save(step,'step'+str(step)+'.pt')
                        if step==900:
                            state.update(training_epoch_seconds=time.time()-state['training_entry_epoch'],training_monotonic_seconds=time.monotonic()-state['training_entry_monotonic'])
                            phase_deadline=None
                        readout(step//9)
                        state['endpoints'][str(step//9)]['checkpoint_identity']=d.identity(folder/('step'+str(step)+'.pt'),hashed=True)
                    if step%9==0 and step<900:
                        state.update(training_epoch_seconds=time.time()-state['training_entry_epoch'],training_monotonic_seconds=time.monotonic()-state['training_entry_monotonic'])
                        persist()
                state.update(status='completed',arm_epoch_seconds=time.time()-arm_epoch,arm_monotonic_seconds=time.monotonic()-arm_mono,
                    training_interval_includes_intermediate_readout_and_checkpoint=True,training_interval_excludes_final_readout=True)
                del model,optimizer
                torch.cuda.empty_cache();guard();persist()
        phase_deadline=None;guard()
        assert a['optimizer_step_attempts']==a['optimizer_step_completions']==900
        assert a['resident_physical_target_reuses']==27
        assert set(a['arms'][s['candidate']]['case_exposure_counts'].values())=={100}
        control=s['baseline_scalar_endpoints']['100']['full_TRAIN9']
        candidate=a['arms'][s['candidate']]['endpoints']['100']['full_TRAIN9']
        complete_physical(control,keys);complete_physical(candidate,keys)
        du=candidate['case_mean_u_relative_l2']-control['case_mean_u_relative_l2']
        ds=candidate['case_mean_stress_relative_l2']-control['case_mean_stress_relative_l2']
        dh=candidate['case_mean_stress_hotspot_relative_l2']-control['case_mean_stress_hotspot_relative_l2']
        dp=.5*(du+ds);threshold=s['TRAIN_gate']
        a['TRAIN_gate']=dict(passed=ds<=-threshold['stress_absolute_improvement_min'] and dp<=-threshold['primary_absolute_improvement_min']
            and max(du,ds)<=threshold['field_deterioration_max'] and dh<=threshold['hotspot_deterioration_max'],
            U_difference=du,S_difference=ds,primary_difference=dp,hotspot_difference=dh,thresholds=threshold,fixed_denominator=9,
            direction='official two-scale UNO package minus preserved plain77 scalar; architecture/capacity/complexAdam implementation differs',automatic_larger_experiment_released=False)
        a['baseline_reused_scalar_only']=True
        assert a['arms'][s['candidate']]['qualification']['complex_safe_parameter_gradients']
        a.update(status='completed',phase='terminal',training_complete=True,shared_encoder_head_initial_bytes_verified=True,
                 protocol_qualified=True,final_solver_selected=False,independent_offnode_FEM_accuracy_proven=False)
    except BaseException as error:
        a.update(status='failed',phase='terminal',error_type=type(error).__name__,error=str(error),
                 possible_partial_optimizer_update=a['optimizer_step_in_progress'])
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
        if not args.plan:parser.error('--start requires a frozen released --plan')
        run(args.plan)
    else:print('{"status":"source_only","scientific_IO":0,"GPU":0}')
