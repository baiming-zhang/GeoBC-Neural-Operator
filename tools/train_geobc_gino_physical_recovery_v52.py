"""Single bounded current physical5 GINO run; no automatic replay/recovery."""
import time
ENTRY_EPOCH, ENTRY_MONOTONIC = time.time(), time.monotonic()
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import signal


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.gino_physical_recovery_contract_v52 import approved_contract
    plan, review, manifest, order, inventory = approved_contract(plan_path)
    s = plan['scientific']
    outer = json.loads(os.environ['GEOBC_GINO_RECOVERY_V52_GUARD'])
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert outer['scientific_digest'] == plan['scientific_digest'] and outer['boot_id'] == boot
    assert d.process_snapshot(outer['guard_process']['pid']) == outer['guard_process']
    output = d.path(plan['output'])
    assert output.exists() and not (output/'execution.json').exists()
    execution = dict(run_process=d.process_snapshot(os.getpid()), guard=outer,
        boot_id=boot, scientific_digest=plan['scientific_digest'])
    d.write(output/'execution.json', execution)
    keys = s['TRAIN_keys']
    result = dict(status='running', active=True, closed=False, execution=execution,
        family=s['family'], phase='cache_setup', step=0, optimizer_step_attempts=0,
        optimizer_step_completions=0, optimizer_step_in_progress=False,
        possible_partial_optimizer_update=False, case_exposure_counts={k:0 for k in keys},
        last_checkpoint=None, endpoints={}, cache_load_complete=False,
        raw_GT_reads=0, DEV_FINAL_reads=0, scaler_fits=0, automatic_replay=False,
        cached_TRAIN_physical_readout_reuses=0, peak_allocated_bytes=0,
        peak_reserved_bytes=0, peak_total_device_used_bytes_sampled=0,
        backend_external_workspace_exact_peak_known=False)
    result['prior_failure_cost'] = s['prior_failure_cost']
    result['original_whole_deadline_preserved'] = True
    result['fresh_reconstruction_reason'] = 'Zero original updates and no saved checkpoint; fixed source/init/seed/order unchanged'
    cache = radius = torch = lock = None
    phase_deadline = (ENTRY_EPOCH+s['cache_setup_budget_seconds'],
                      ENTRY_MONOTONIC+s['cache_setup_budget_seconds'])
    train_deadline = None

    def guard():
        clocks = [(outer['deadline_epoch'], outer['deadline_monotonic'])]
        clocks += [v for v in (phase_deadline, train_deadline) if v is not None]
        if min(min(a-time.time(), b-time.monotonic()) for a,b in clocks) <= 0:
            raise TimeoutError('Fixed dual-clock phase/whole budget exhausted')
        if torch is not None and torch.cuda.is_initialized():
            free,total = torch.cuda.mem_get_info()
            vals = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                peak_total_device_used_bytes_sampled=total-free)
            for name,value in vals.items():
                result[name] = max(result[name], value)
                if value > s['memory_limit_bytes']:
                    raise MemoryError('Measured component exceeds30.5GB: '+name)

    def persist():
        result.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,
                      elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        if cache is not None:
            result['cache_receipt'] = dict(cache.receipt)
        if radius is not None:
            result['radius_costs'] = dict(costs)
        if train_deadline is not None:
            result.update(training_epoch_seconds=time.time()-result['training_entry_epoch'],
                training_monotonic_seconds=time.monotonic()-result['training_entry_monotonic'])
        d.write(output/'status.json', result)

    signal.signal(signal.SIGTERM, lambda *args: (_ for _ in ()).throw(TimeoutError('External guard termination')))
    try:
        guard()
        import fcntl
        lock = d.path(s['GPU_lock']).open('a+')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        import numpy as np
        import torch as tm
        torch = tm
        from geobc.gino_physical_cache_v52 import NativeGINOPhysicalCache
        from geobc.gino_physical_training_v52 import build_candidate, full_physical_backward
        from geobc.gino_radius_cache_v52 import CurrentTRAINRadiusCache
        from geobc.gino_train import _gradient_check
        from geobc.gino_state_v52 import state_digest
        from geobc.physical_field_loss import PhysicalScale
        from geobc.benchmark import FieldScaler
        from geobc.coverage_comparison_train import physical_metrics, aggregate
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        assert s['memory_limit_bytes'] < total
        torch.cuda.set_per_process_memory_fraction(s['memory_limit_bytes']/total, device)
        torch.cuda.reset_peak_memory_stats()
        random.seed(s['seed']); np.random.seed(s['seed'])
        torch.manual_seed(s['seed']); torch.cuda.manual_seed_all(s['seed'])

        def cache_event(stage,name,item):
            if stage == 'completed' and (cache.receipt['input_array_reads']+cache.receipt['physical_TRAIN_target_reads'])%30 == 0:
                persist()
        cache = NativeGINOPhysicalCache(manifest, memory_limit_bytes=s['memory_limit_bytes'],
            guard=guard, on_read=cache_event)
        result['cache_disk_bytes_upper'] = cache.disk_bytes_upper
        persist();cache.load()
        result['cache_load_complete'] = True
        d.write(output/'cache_index.json', cache.specs)
        d.write(output/'physical_case_norms.json', {k:asdict(v) for k,v in cache.case_norms.items()})
        model = build_candidate('cuda').train()
        result['parameter_count'] = sum(p.numel() for p in model.parameters())
        result['real_parameter_DOF'] = sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
        assert result['parameter_count'] == s['expected_parameter_count']
        assert result['real_parameter_DOF'] == s['expected_real_parameter_DOF']
        result['model_profile'] = dict(model.source_config)
        result['actual_fresh_initial_model_sha256'] = state_digest(model.state_dict())
        recipe = s['optimizer']
        optimizer = torch.optim.Adam(model.parameters(), lr=recipe['learning_rate'],
            weight_decay=recipe['weight_decay'], betas=tuple(recipe['betas']), eps=recipe['eps'],
            foreach=recipe['foreach'], fused=recipe['fused'])
        scale = PhysicalScale.from_mapping(s['scaler'])
        scaler = FieldScaler(**s['scaler'])
        costs = dict(radius_build_seconds=0.,radius_build_count=0,radius_save_seconds=0.,
            radius_read_seconds=0.,radius_read_count=0,radius_cache_reuse_count=0)
        def progress(phase, **details):
            guard();result['last_radius_progress'] = dict(phase=phase,**details);persist()
        radius = CurrentTRAINRadiusCache(s['radius_plan'], {'geometries':s['geometries']},
            torch,np,'cuda',costs,progress,inventory)
        def rng():
            return dict(python=random.getstate(),numpy=np.random.get_state(),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
        def restore(v):
            random.setstate(v['python']);np.random.set_state(v['numpy'])
            torch.set_rng_state(v['torch']);torch.cuda.set_rng_state_all(v['cuda'])
        def rng_digest(v):
            import hashlib
            h=hashlib.sha256(repr(v['python']).encode())
            state=v['numpy']
            h.update(repr((state[0],*state[2:])).encode());h.update(state[1].tobytes())
            h.update(v['torch'].numpy().tobytes())
            for item in v['cuda']:h.update(item.numpy().tobytes())
            return h.hexdigest()
        guard()
        result.update(cache_setup_epoch_seconds=time.time()-ENTRY_EPOCH,
            cache_setup_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        qstart=(time.time(),time.monotonic())
        phase_deadline=(qstart[0]+s['qualification_budget_seconds'],qstart[1]+s['qualification_budget_seconds'])
        result['phase']='fullN_qualification';persist()
        witness=max(keys,key=lambda k:manifest['records'][k]['num_nodes'])
        saved_rng=rng();before_rng_sha=rng_digest(saved_rng);optimizer.zero_grad(set_to_none=True)
        case=radius.attach(cache.case(witness))
        loss,leaves=full_physical_backward(model,case,cache.physical_targets[witness],
            scale,cache.case_norms[witness],keys,guard)
        gradient=_gradient_check(torch,model)
        assert gradient['missing_gradient_names']==[] and leaves['operator_latent_L2']>0
        guard();optimizer.zero_grad(set_to_none=True);restore(saved_rng)
        assert not optimizer.state and state_digest(model.state_dict())==result['actual_fresh_initial_model_sha256']
        assert rng_digest(rng())==before_rng_sha
        result['actual_largest_fullN_gradient_check']=dict(passed=True,key=witness,
            nodes=len(case['xyz']),physical_loss=loss,leaf_gradients=leaves,groups=gradient,
            actual_optimizer_updates=0,model_and_Adam_unchanged=True,fourRNG_restored=True,
            unchanged_fourRNG_sha256=before_rng_sha)
        del case
        result.update(qualification_epoch_seconds=time.time()-qstart[0],qualification_monotonic_seconds=time.monotonic()-qstart[1])
        guard();phase_deadline=None

        def save(step,name):
            guard()
            if step:
                assert len(optimizer.state)==sum(1 for _ in model.parameters())
                assert all(int(v['step'])==step for v in optimizer.state.values())
            else:
                assert not optimizer.state
            tmp=output/(name+'.tmp')
            torch.save(dict(step=step,cursor=step,scientific_digest=plan['scientific_digest'],
                family=s['family'],model_profile=result['model_profile'],model=model.state_dict(),
                optimizer=optimizer.state_dict(),rng=rng(),case_exposure_counts=dict(result['case_exposure_counts']),
                full_order_digest=s['full_order_digest'],dataset_identity=s['dataset_identity'],scaler=s['scaler'],
                native_origin_clocks=dict(epoch=ENTRY_EPOCH,monotonic=ENTRY_MONOTONIC,boot_id=boot),
                cost_status=dict(result)),tmp)
            os.replace(tmp,output/name)
            result['last_checkpoint']=dict(path=d.relative(output/name),step=step,stat_identity=d.identity(output/name))
            guard()
        result.update(training_entry_epoch=time.time(),training_entry_monotonic=time.monotonic(),phase='training')
        train_deadline=(result['training_entry_epoch']+s['training_budget_seconds'],result['training_entry_monotonic']+s['training_budget_seconds'])
        save(0,'last.pt');persist()
        with (output/'loss.jsonl').open('x',encoding='utf-8') as log:
            for step,key in enumerate(order,1):
                guard();optimizer.zero_grad(set_to_none=True)
                case=radius.attach(cache.case(key))
                loss,_=full_physical_backward(model,case,cache.physical_targets[key],scale,cache.case_norms[key],keys,guard)
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                result['optimizer_step_attempts']+=1;result['optimizer_step_in_progress']=True
                optimizer.step();torch.cuda.synchronize()
                result.update(step=step,optimizer_step_completions=result['optimizer_step_completions']+1,optimizer_step_in_progress=False)
                result['case_exposure_counts'][key]+=1
                log.write(json.dumps(dict(step=step,key=key,physical_relative_squared=loss,
                    training_epoch_seconds=time.time()-result['training_entry_epoch']))+'\n');log.flush()
                del case
                guard()
                if step==1 or step%30==0:
                    save(step,'last.pt');persist()
                if step%300==0:
                    assert set(result['case_exposure_counts'].values())=={step//300}
            save(1500,'step1500.pt')
        result.update(training_epoch_seconds=time.time()-result['training_entry_epoch'],
            training_monotonic_seconds=time.monotonic()-result['training_entry_monotonic'])
        guard();train_deadline=None
        rstart=(time.time(),time.monotonic());phase_deadline=(rstart[0]+s['TRAIN_readout_budget_seconds'],rstart[1]+s['TRAIN_readout_budget_seconds'])
        result['phase']='TRAIN_endpoint_readout';persist();saved_rng=rng();model.eval()
        rows={key:None for key in keys}
        with torch.no_grad():
            for key in keys:
                guard();case=radius.attach(cache.case(key));state=model.encode_case(case)
                prediction=np.empty((len(case['xyz']),4),dtype=np.float32)
                for begin in range(0,len(prediction),model.query_chunk):
                    guard();end=min(begin+model.query_chunk,len(prediction))
                    ids=torch.arange(begin,end,dtype=torch.int64,device='cuda')
                    prediction[begin:end]=model.query(state,ids).cpu().numpy()
                physical=scaler.decode(prediction)
                assert np.isfinite(physical).all()
                truth=cache.physical_targets[key].detach().cpu().numpy()
                rows[key]=physical_metrics(np,truth,physical)
                result['cached_TRAIN_physical_readout_reuses']+=1
                del case,state,prediction,physical,truth,ids
                if result['cached_TRAIN_physical_readout_reuses']%30==0:persist()
        metric=aggregate(rows,300)
        assert metric['all_cases_complete'] and metric['fixed_denominator']==300
        restore(saved_rng);model.train();guard()
        checkpoint=d.identity(output/'step1500.pt',hashed=True)
        result['endpoints']['1500']=dict(full_TRAIN=metric,checkpoint_identity=checkpoint,
            epoch_seconds=time.time()-rstart[0],monotonic_seconds=time.monotonic()-rstart[1])
        guard();phase_deadline=None
        assert result['optimizer_step_attempts']==result['optimizer_step_completions']==result['step']==1500
        assert set(result['case_exposure_counts'].values())=={5}
        assert cache.receipt['physical_TRAIN_target_reads']==300
        result['radius_receipt']=radius.receipt()
        result.update(status='completed',phase='terminal',training_complete=True,
            requires_separate_DEV_prediction_barrier_readout=True,DEV_evaluation_actual_released=False,
            physical_TRAIN_total_array_reads=300+s['prior_failure_cost']['physical_TRAIN_target_reads'],automatic40=False,automatic24000=False)
        guard()
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error),
            possible_partial_optimizer_update=result['optimizer_step_in_progress'])
        raise
    finally:
        if train_deadline is not None:
            result.update(training_epoch_seconds=time.time()-result['training_entry_epoch'],
                training_monotonic_seconds=time.monotonic()-result['training_entry_monotonic'])
        if cache is not None:result['cache_receipt']=dict(cache.receipt)
        if radius is not None:
            result['radius_costs']=dict(costs);result['radius_receipt']=radius.receipt()
        result.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,
            elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC,
            original_whole_epoch_span_seconds=time.time()-s['original_entry_epoch'],
            original_whole_monotonic_span_seconds=time.monotonic()-s['original_entry_monotonic'])
        d.write(output/'summary.json',result);d.write(output/'status.json',result)
        if lock is not None:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--start',action='store_true');parser.add_argument('--plan')
    args=parser.parse_args()
    if args.start:
        if not args.plan:parser.error('--start requires frozen --plan')
        run(args.plan)
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
