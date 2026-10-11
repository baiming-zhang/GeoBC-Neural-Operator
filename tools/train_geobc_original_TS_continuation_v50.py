"""One original R16 Transolver own physical5-to10 continuation.

The scientific plan is intentionally not created here. Actual prior paired
training/development endpoints must exist before separate root release.
"""
import time
ENTRY_EPOCH, ENTRY_MONOTONIC = time.time(), time.monotonic()
import argparse
import json
import os
from pathlib import Path
import random
import signal


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.original_TS_continuation_contract_v50 import approved_contract
    plan, review, manifest, order = approved_contract(plan_path)
    science = plan['scientific']
    outer = json.loads(os.environ['GEOBC_ORIGINAL_TS_V50_GUARD'])
    assert outer['scientific_digest'] == plan['scientific_digest']
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert boot == outer['boot_id']
    assert d.process_snapshot(outer['guard_process']['pid']) == outer['guard_process']
    output = d.path(plan['output'])
    assert output.exists() and not (output/'execution.json').exists()
    execution = dict(run_process=d.process_snapshot(os.getpid()), guard=outer,
                     boot_id=boot, scientific_digest=plan['scientific_digest'])
    d.write(output/'execution.json', execution)
    keys = science['TRAIN_keys']
    checkpoints = {len(keys)*passes: passes for passes in science['endpoint_passes']}
    result = dict(status='running', active=True, closed=False, execution=execution,
        family=science['family'], phase='cache_setup', step=1500, case_exposure_counts=dict(science['parent_case_exposure_counts']),
        parent_optimizer_steps=1500, additional_optimizer_step_attempts=0, additional_optimizer_step_completions=0,
        optimizer_counter_scope='Inherited full trajectory; separate additional counters for this segment', fresh_initialization=False,
        endpoints={},cached_TRAIN_physical_readout_reuses=0,raw_GT_reads=0, DEV_FINAL_reads=0, scaler_fits=0,
        cached_TRAIN_physical_read_attempts=0, cached_TRAIN_physical_reads=0,
        optimizer_step_attempts=1500, optimizer_step_completions=1500,
        optimizer_step_in_progress=False, possible_partial_optimizer_update=False,
        last_checkpoint=None, automatic_replay=False, peak_allocated_bytes=0, peak_reserved_bytes=0,
        peak_total_device_used_bytes_sampled=0, backend_external_workspace_exact_peak_known=False,
        cache_setup_entry_epoch=ENTRY_EPOCH, cache_setup_entry_monotonic=ENTRY_MONOTONIC)
    lock = None
    cache = None
    torch = None
    cleanup = None
    qualification_deadline = None
    cache_deadline = (ENTRY_EPOCH+science['cache_setup_budget_seconds'],
                      ENTRY_MONOTONIC+science['cache_setup_budget_seconds'])
    train_deadline = None
    readout_deadline = None
    cap = science['memory_limit_bytes']

    def guard():
        deadlines = [(outer['deadline_epoch'], outer['deadline_monotonic'])]
        deadlines += [v for v in (cache_deadline, qualification_deadline, train_deadline, readout_deadline) if v is not None]
        if min(min(a-time.time(), b-time.monotonic()) for a, b in deadlines) <= 0:
            raise TimeoutError('Frozen macro dual-clock deadline')
        if torch is not None and torch.cuda.is_initialized():
            result['peak_allocated_bytes'] = max(result['peak_allocated_bytes'], torch.cuda.max_memory_allocated())
            result['peak_reserved_bytes'] = max(result['peak_reserved_bytes'], torch.cuda.max_memory_reserved())
            free, total = torch.cuda.mem_get_info()
            result['peak_total_device_used_bytes_sampled'] = max(result['peak_total_device_used_bytes_sampled'], total-free)
            if max(result[k] for k in ('peak_allocated_bytes', 'peak_reserved_bytes', 'peak_total_device_used_bytes_sampled')) > cap:
                raise MemoryError('Allocated/reserved or sampled total-device usage exceeds30.5GB')

    def persist():
        if train_deadline is not None:
            result.update(training_epoch_seconds=time.time()-result['training_entry_epoch'],
                          training_monotonic_seconds=time.monotonic()-result['training_entry_monotonic'])
        result.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,
                      elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(output/'status.json', result)

    signal.signal(signal.SIGTERM, lambda *args: (_ for _ in ()).throw(TimeoutError('External guard terminated child')))
    try:
        guard()
        import fcntl
        lock = d.path(science['GPU_lock']).open('a+')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        import numpy as np
        import torch as torch_module
        torch = torch_module
        from geobc.train_physical_cache_v12 import PhysicalTRAINCache as CacheFactory
        from geobc.coarse_transolver_v19 import CoarseTransolver,full_physical_backward
        from geobc.geometry_head_transfer_v32 import state_digest
        from geobc.physical_field_loss import PhysicalScale
        from dataclasses import asdict
        import hashlib
        from geobc.benchmark import FieldScaler
        from geobc.coverage_comparison_train import physical_metrics, aggregate
        from geobc.transformers import close_owned_process_group
        cleanup=close_owned_process_group
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        assert cap < total
        torch.cuda.set_per_process_memory_fraction(cap/total, device)
        torch.cuda.reset_peak_memory_stats()
        random.seed(science['seed']); np.random.seed(science['seed'])
        torch.manual_seed(science['seed']); torch.cuda.manual_seed_all(science['seed'])

        def cache_event(stage, name, identity):
            if stage == 'completed':
                result['cache_receipt'] = dict(cache.receipt)
                count = cache.receipt['input_array_reads']+cache.receipt['encoded_TRAIN_target_reads']+cache.receipt['physical_TRAIN_target_reads']
                if count % 20 == 0:
                    persist()
        cache = CacheFactory(manifest,science['train_CAD_count'],'FNO9',
                                   memory_limit_bytes=cap, guard=guard, on_read=cache_event)
        result['cache_metadata_profile'] = cache.metadata_profile()
        persist()
        cache.load()
        result['cache_receipt'] = dict(cache.receipt)
        result['cache_load_complete'] = True
        d.write(output/'cache_index.json', cache.specs)
        d.write(output/'physical_cache_index.json', cache.physical_specs)
        d.write(output/'physical_case_norms.json', {key: asdict(value) for key,value in cache.case_norms.items()})
        assert {key: asdict(value) for key,value in cache.case_norms.items()}==d.read(science['parent_norms_identity']['path'])
        endpoint=science['parent_checkpoint'];guard()
        assert d.identity(endpoint['path'],hashed=True)==endpoint
        result['parent_checkpoint_hash_reads']=1
        saved=torch.load(d.path(endpoint['path']),map_location='cpu',weights_only=False)
        result['parent_checkpoint_loads']=1
        assert saved['step']==saved['cursor']==1500 and saved['family']==science['family'] and saved['method']=='transolver'
        assert saved['scientific_digest']==science['parent_training_SCI']
        assert saved['native_origin_clocks']['boot_id']==d.read(science['parent_terminal']['terminal']['summary']['path'])['execution']['boot_id']
        assert saved['case_exposure_counts']==science['parent_case_exposure_counts']
        assert saved['full_order_digest']==science['parent_full_order_digest'] and saved['scaler']==science['scaler']
        assert saved['dataset_identity']==science['dataset_identity'] and set(saved['rng'])=={'python','numpy','torch','cuda'}
        model=CoarseTransolver('transolver',science['seed'],'cuda').train()
        model.load_state_dict(saved['model'],strict=True)
        parent_summary=d.read(science['parent_summary_identity']['path'])
        parent_external=d.read(science['parent_external_identity']['path'])
        result['parent_cost']=dict(training_epoch_seconds=parent_summary['training_epoch_seconds'],training_monotonic_seconds=parent_summary['training_monotonic_seconds'],
            stage_epoch_seconds=parent_external['elapsed_epoch_seconds'],stage_monotonic_seconds=parent_external['elapsed_monotonic_seconds'],
            parent_summary_identity=science['parent_summary_identity'],parent_external_identity=science['parent_external_identity'],
            parent_arm_elapsed_epoch_seconds=parent_summary['elapsed_epoch_seconds'],parent_arm_elapsed_monotonic_seconds=parent_summary['elapsed_monotonic_seconds'],
            old_shared_cache_epoch_seconds=science['old_shared_cache_epoch_seconds'],old_shared_cache_monotonic_seconds=science['old_shared_cache_monotonic_seconds'],
            stage_cost_scope='Old complete threeTS campaign shared; parent TS update/arm/cache costs separate, do not attribute whole old campaign only to TS',
            inter_stage_analysis_wait_included=False,previous_training_not_replayed=True)
        result['model_profile']=model.profile()
        assert result['model_profile']['parameter_count']==science['expected_parameter_count']
        digest = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        result['loaded_parent_model_sha256'] = digest.hexdigest()
        result['original_parent_fresh_initial_model_sha256'] = parent_summary['actual_fresh_initial_model_sha256']
        recipe = science['optimizer']
        optimizer = torch.optim.Adam(model.parameters(), lr=recipe['learning_rate'],
            weight_decay=recipe['weight_decay'], betas=tuple(recipe['betas']), eps=recipe['eps'], fused=recipe['fused'])
        assert state_digest(saved['model'])==result['loaded_parent_model_sha256']
        assert saved['cost_status']['actual_fresh_initial_model_sha256']==parent_summary['actual_fresh_initial_model_sha256']
        optimizer.load_state_dict(saved['optimizer'])
        assert optimizer.state and len(optimizer.state)==sum(1 for _ in model.parameters()) and all(int(v['step'])==1500 for v in optimizer.state.values())
        assert all(torch.isfinite(v).all() for p in model.parameters() for v in optimizer.state[p].values() if isinstance(v,torch.Tensor))
        result['parent_saved_Adam_state_count']=len(optimizer.state)
        assert all(g['lr']==recipe['learning_rate'] and g['weight_decay']==recipe['weight_decay'] and tuple(g['betas'])==tuple(recipe['betas']) and g['eps']==recipe['eps'] and g['fused'] for g in optimizer.param_groups)
        parent_rng=saved['rng'];del saved
        scaler = FieldScaler(**science['scaler'])
        physical_scale = PhysicalScale.from_mapping(science['scaler'])
        result['objective'] = science['loss']
        result['baseline_TRAIN300'] = science['parent_TRAIN300']

        def full_case_backward(model, case, allowed_keys, guard):
            return full_physical_backward(model,case,cache.physical_targets[case['key']],physical_scale,cache.case_norms[case['key']],allowed_keys,guard)

        def rng():
            return dict(python=random.getstate(), numpy=np.random.get_state(),
                        torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())
        def restore(state):
            random.setstate(state['python']); np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])

        def Adam_digest():
            value=hashlib.sha256()
            value.update(json.dumps(optimizer.param_groups[0],sort_keys=True,default=lambda x:None).encode())
            for name,parameter in sorted(model.named_parameters()):
                value.update(name.encode())
                for key,tensor in sorted(optimizer.state[parameter].items()):
                    value.update(key.encode())
                    if isinstance(tensor,torch.Tensor):
                        value.update(str(tensor.dtype).encode());value.update(str(tuple(tensor.shape)).encode())
                        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                    else:value.update(str(tensor).encode())
            return value.hexdigest()

        restore(parent_rng)
        result['restored_model_Adam_fourRNG_cursor_exposure']=True
        result.update(cache_resume_epoch_seconds=time.time()-ENTRY_EPOCH,cache_resume_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        cache_deadline=None
        qualification_started=(time.time(),time.monotonic())
        qualification_deadline=(qualification_started[0]+science['qualification_budget_seconds'],qualification_started[1]+science['qualification_budget_seconds'])
        # One actual largest fixed TRAIN case, no optimizer mutation. Restore
        # all RNG so qualification cannot change the restored continuation trajectory.
        witness = max(keys, key=lambda k: manifest['records'][k]['num_nodes'])
        saved_rng = rng()
        parent_Adam_digest=Adam_digest()
        optimizer.zero_grad(set_to_none=True)
        _, leaves = full_case_backward(model, cache.case(witness), keys, guard)
        groups = dict(encoder=[], decoder_or_head=[], operator=[])
        missing = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                missing.append(name)
                continue
            assert torch.isfinite(parameter.grad).all(), 'Nonfinite actual parameter gradient'
            group = 'encoder' if name.startswith('encoder.') else 'operator' if name.startswith('operator.') else 'decoder_or_head'
            groups[group].append(float(parameter.grad.detach().abs().double().square().sum()))
        allowed_unused=science['allowed_unused_parameters']
        assert sorted(missing) == sorted(allowed_unused)
        norms = {k: sum(v)**.5 for k, v in groups.items() if v}
        assert all(value > 0 for value in norms.values())
        assert norms.get('encoder', 0) > 0 and norms.get('decoder_or_head', 0) > 0
        if science['family'] != 'HASH_PN':
            assert norms.get('operator', 0) > 0 and leaves['operator_latent_L2'] > 0
        assert leaves['geometry_context_L2'] > 0
        guard()
        result['actual_largest_fullN_gradient_check'] = dict(passed=True, key=witness,
            nodes=manifest['records'][witness]['num_nodes'], groups_L2=norms, allowed_unused=missing,
            leaf_gradients=leaves, official_backend=True, optimizer_updates=0)
        optimizer.zero_grad(set_to_none=True)
        restore(saved_rng)
        assert state_digest(model.state_dict())==result['loaded_parent_model_sha256']
        assert all(int(v['step'])==1500 for v in optimizer.state.values())
        assert Adam_digest()==parent_Adam_digest
        result.update(qualification_epoch_seconds=time.time()-qualification_started[0],qualification_monotonic_seconds=time.monotonic()-qualification_started[1],
            actual_qualification_model_Adam_unchanged=True,loaded_parent_Adam_SHA256=parent_Adam_digest)
        guard();qualification_deadline=None
        result.update(cache_setup_epoch_seconds=time.time()-ENTRY_EPOCH,
                      cache_setup_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC,
                      training_entry_epoch=time.time(), training_entry_monotonic=time.monotonic(), phase='training')
        cache_deadline = None
        train_deadline = (result['training_entry_epoch']+science['training_budget_seconds'],
                          result['training_entry_monotonic']+science['training_budget_seconds'])
        persist()

        def save(step, name):
            guard()
            assert optimizer.state and all(int(state['step']) == step for state in optimizer.state.values())
            result['saved_Adam_state_count'] = len(optimizer.state)
            result['saved_Adam_steps_equal_cursor'] = True
            temporary = output/(name+'.tmp')
            torch.save(dict(step=step, cursor=step, scientific_digest=plan['scientific_digest'],
                family=science['family'],method='transolver',model_profile=model.profile(),training_objective=science['loss'],
                parent_checkpoint=science['parent_checkpoint'],parent_training_SCI=science['parent_training_SCI'],parent_cost=result['parent_cost'],
                model=model.state_dict(), optimizer=optimizer.state_dict(),
                rng=rng(), case_exposure_counts=dict(result['case_exposure_counts']),
                full_order_digest=science['full_order_digest'], dataset_identity=science['dataset_identity'],
                scaler=science['scaler'], native_origin_clocks=dict(epoch=ENTRY_EPOCH, monotonic=ENTRY_MONOTONIC, boot_id=boot),
                outer_guard_clock=outer, cost_status=dict(result)), temporary)
            os.replace(temporary, output/name)
            result['last_checkpoint'] = dict(path=d.relative(output/name), step=step,
                                            stat_identity=d.identity(output/name))
            guard()

        def readout(step):
            nonlocal readout_deadline
            saved = rng()
            started = (time.time(), time.monotonic())
            readout_deadline = (started[0]+science['TRAIN_readout_budget_seconds'],
                                started[1]+science['TRAIN_readout_budget_seconds'])
            result['phase'] = 'TRAIN_endpoint_readout'
            model.eval()
            rows = {key: None for key in keys}
            try:
                with torch.no_grad():
                    for index, key in enumerate(keys):
                        guard()
                        torch.manual_seed(2026+index); torch.cuda.manual_seed_all(2026+index)
                        case = cache.case(key)
                        context, latent = model.encode_case(case)
                        prediction = np.empty((len(case['xyz']), 4), dtype=np.float32)
                        for start in range(0, len(prediction), 32768):
                            guard(); stop = min(start+32768, len(prediction))
                            prediction[start:stop] = model.query(case,start,stop,context,latent).cpu().numpy()
                        physical = scaler.decode(prediction)
                        if not np.isfinite(physical).all():
                            raise FloatingPointError('Nonfinite full-node physical prediction')
                        truth=cache.physical_targets[key].detach().cpu().numpy()
                        result['cached_TRAIN_physical_readout_reuses']+=1
                        assert truth.shape==prediction.shape and np.isfinite(truth).all()
                        rows[key] = physical_metrics(np, truth, physical)
                        del context, latent, prediction, physical, truth
                metric = aggregate(rows, len(keys))
                result['endpoints'][str(step)] = dict(full_TRAIN=metric, case_exposure_count=step//len(keys),
                    epoch_seconds=time.time()-started[0], monotonic_seconds=time.monotonic()-started[1])
                assert metric['all_cases_complete'] and metric['fixed_denominator'] == len(keys)
                guard()
            finally:
                restore(saved); model.train(); readout_deadline = None
            result['phase'] = 'training'
            persist()

        with (output/'loss.jsonl').open('x', encoding='utf-8') as log:
            for step, key in enumerate(order[1500:], 1501):
                guard()
                optimizer.zero_grad(set_to_none=True)
                loss, _ = full_case_backward(model, cache.case(key), keys, guard)
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                result['optimizer_step_attempts'] += 1
                result['additional_optimizer_step_attempts'] += 1
                result['optimizer_step_in_progress'] = True
                optimizer.step(); torch.cuda.synchronize()
                result.update(step=step, optimizer_step_completions=result['optimizer_step_completions']+1,
                              optimizer_step_in_progress=False)
                result['additional_optimizer_step_completions'] += 1
                result['case_exposure_counts'][key] += 1
                log.write(json.dumps(dict(step=step, key=key, physical_relative_squared=loss,
                    training_epoch_seconds=time.time()-result['training_entry_epoch']))+'\n')
                log.flush()
                guard()
                if step % len(keys) == 0:
                    save(step, 'last.pt')
                if step % 30 == 0:
                    persist()
                if step in checkpoints:
                    assert set(result['case_exposure_counts'].values()) == {checkpoints[step]}
                    save(step, 'step'+str(step)+'.pt')
                    if step == len(order):
                        result.update(training_epoch_seconds=time.time()-result['training_entry_epoch'],
                                      training_monotonic_seconds=time.monotonic()-result['training_entry_monotonic'])
                        train_deadline = None
                    readout(step)
                    result['endpoints'][str(step)]['checkpoint_identity'] = d.identity(output/('step'+str(step)+'.pt'), hashed=True)
        guard()
        result.update(status='completed', phase='terminal', training_complete=True,
                      requires_separate_DEV_prediction_barrier_readout=True,
                      training_wall_includes_intermediate_readout_and_checkpoints=True,automatic40=False,automatic24000=False)
        assert result['additional_optimizer_step_attempts']==result['additional_optimizer_step_completions']==1500
        assert result['optimizer_step_attempts']==result['optimizer_step_completions']==result['step']==3000
        result['trajectory_training_epoch_seconds']=result['parent_cost']['training_epoch_seconds']+result['training_epoch_seconds']
        result['trajectory_training_monotonic_seconds']=result['parent_cost']['training_monotonic_seconds']+result['training_monotonic_seconds']
        metric = result['endpoints']['3000']['full_TRAIN']
        baseline = science['parent_TRAIN300']; thresholds = science['TRAIN_gate']
        delta_u = metric['case_mean_u_relative_l2']-baseline['case_mean_u_relative_l2']
        delta_s = metric['case_mean_stress_relative_l2']-baseline['case_mean_stress_relative_l2']
        delta_primary=.5*(delta_u+delta_s)
        result['TRAIN_gate'] = dict(passed=(delta_primary <= -thresholds['primary_absolute_improvement_at_least']
            and delta_s<=-thresholds['stress_absolute_improvement_at_least'] and max(delta_u,delta_s) <= thresholds['maximum_either_field_absolute_deterioration']),
            U_difference=delta_u, S_difference=delta_s, primary_difference=delta_primary,
            direction='physical10 minus ownphysical5; exposure difference only, not new loss',
            thresholds=thresholds, fixed_denominator=300, releases_DEV=False)
        result['physical_TRAIN_total_array_reads'] = cache.receipt['physical_TRAIN_target_reads']+result['cached_TRAIN_physical_reads']
        result['DEV_evaluation_actual_released'] = False
    except Exception as error:
        if cache is not None:
            result['cache_receipt'] = dict(cache.receipt)
        result.update(status='failed', error_type=type(error).__name__, error=str(error),
                      possible_partial_optimizer_update=result['optimizer_step_in_progress'])
        if train_deadline is not None:
            result.update(training_epoch_seconds=time.time()-result['training_entry_epoch'],
                          training_monotonic_seconds=time.monotonic()-result['training_entry_monotonic'])
        raise
    finally:
        try:
            if cleanup is not None:cleanup()
        finally:
            result.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
            d.write(output/'summary.json',result);d.write(output/'status.json',result)
            if lock is not None:lock.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--plan')
    args = parser.parse_args()
    if args.start:
        if not args.plan:
            parser.error('--start requires a frozen separately released --plan')
        run(args.plan)
    else:
        print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
