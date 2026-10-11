"""Two fixed denseR16 architectures, input-derived response-gain conditioning control."""
import time
ENTRY_EPOCH, ENTRY_MONOTONIC = time.time(), time.monotonic()
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.response_gain_training_contract_v30 import approved_contract
    plan, _, manifest, order = approved_contract(plan_path); science = plan['scientific']
    outer = json.loads(os.environ['GEOBC_RESPONSE_GAIN_V30_GUARD'])
    assert outer['scientific_digest'] == plan['scientific_digest']
    assert d.process_snapshot(outer['guard_process']['pid']) == outer['guard_process']
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip(); assert boot == outer['boot_id']
    output = d.path(plan['output']); assert not (output/'execution.json').exists()
    execution = dict(run_process=d.process_snapshot(os.getpid()), guard=outer, boot_id=boot,
                     scientific_digest=plan['scientific_digest'])
    d.write(output/'execution.json', execution)
    keys = science['TRAIN_keys']; cap = science['memory_limit_bytes']
    result = dict(status='running', active=True, closed=False, execution=execution, phase='shared_cache_setup',
        methods={}, cache_load_complete=False, shared_cache_load_attempts=0,
        raw_GT_reads=0, DEV_FINAL_reads=0, scaler_fits=0, automatic_replay=False,
        peak_allocated_bytes=0, peak_reserved_bytes=0, peak_total_device_used_bytes_sampled=0,
        backend_external_workspace_exact_peak_known=False, cached_physical_TRAIN_readout_reuses=0, training_objective=science['loss'])
    torch = None; cache = None; lock = None; cleanup = None; arm = None
    deadlines = {'cache': (ENTRY_EPOCH+600, ENTRY_MONOTONIC+600)}
    signal.signal(signal.SIGTERM, lambda *args: (_ for _ in ()).throw(TimeoutError('External guard terminated campaign')))

    def guard():
        limits = [(outer['deadline_epoch'], outer['deadline_monotonic'])] + list(deadlines.values())
        if min(min(a-time.time(), b-time.monotonic()) for a, b in limits) <= 0:
            raise TimeoutError('Frozen campaign/phase dual-clock deadline')
        if torch is not None and torch.cuda.is_initialized():
            for name, value in [('peak_allocated_bytes', torch.cuda.max_memory_allocated()),
                                ('peak_reserved_bytes', torch.cuda.max_memory_reserved())]:
                result[name] = max(result[name], value)
            free, total = torch.cuda.mem_get_info()
            result['peak_total_device_used_bytes_sampled'] = max(result['peak_total_device_used_bytes_sampled'], total-free)
            if max(result[k] for k in ('peak_allocated_bytes', 'peak_reserved_bytes', 'peak_total_device_used_bytes_sampled')) > cap:
                raise MemoryError('Campaign exceeds30.5GB SI allocated/reserved/sampled total device')

    def persist():
        result.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH, elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        if arm is not None and 'training' in deadlines:
            arm.update(training_epoch_seconds=time.time()-arm['training_entry_epoch'],
                       training_monotonic_seconds=time.monotonic()-arm['training_entry_monotonic'])
        d.write(output/'status.json', result)

    try:
        import fcntl
        lock = d.path(science['GPU_lock']).open('a+'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        import numpy as np
        import torch as torch_module
        torch = torch_module
        from dataclasses import asdict
        from geobc.train_physical_cache_v12 import PhysicalTRAINCache
        from geobc.coarse_transolver_v19 import full_physical_backward
        from geobc.response_gain_v30 import ResponseGainTransolver, ResponseGainFNO, prepare_response_case
        from geobc.coarse_transolver_v19 import CoarseTransolver
        from geobc.coarse_fno_control_v22 import CoarseFNOControl
        from geobc.common_grid_solver import CommonGridSolver
        from geobc.physical_field_loss import PhysicalScale
        from geobc.benchmark import FieldScaler
        from geobc.coverage_comparison_train import physical_metrics, aggregate
        from geobc.transformers import close_owned_process_group
        cleanup = close_owned_process_group
        torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory; assert cap < total
        torch.cuda.set_per_process_memory_fraction(cap/total); torch.cuda.reset_peak_memory_stats()

        def cache_event(stage, name, identity):
            if stage == 'completed':
                result['cache_receipt'] = dict(cache.receipt)
                count = sum(cache.receipt[k] for k in ('input_array_reads', 'encoded_TRAIN_target_reads', 'physical_TRAIN_target_reads'))
                if count % 20 == 0: persist()
        cache = PhysicalTRAINCache(manifest, 100, 'FNO9', memory_limit_bytes=cap, guard=guard, on_read=cache_event)
        result['shared_cache_load_attempts'] += 1; persist(); cache.load()
        assert cache.completed and cache.receipt['physical_TRAIN_loaded_cases'] == 300
        result.update(cache_load_complete=True, cache_receipt=dict(cache.receipt), cache_metadata_profile=cache.metadata_profile(),
                      cache_setup_epoch_seconds=time.time()-ENTRY_EPOCH, cache_setup_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(output/'cache_index.json', cache.specs); d.write(output/'physical_cache_index.json', cache.physical_specs)
        d.write(output/'physical_case_norms.json', {k: asdict(v) for k, v in cache.case_norms.items()})
        result['input_response_gains'] = {}
        for key in keys:
            guard(); gains = prepare_response_case(cache.case(key))
            result['input_response_gains'][key] = gains.detach().cpu().tolist()
        assert len(result['input_response_gains']) == 300
        result['cache_setup_epoch_seconds'] = time.time()-ENTRY_EPOCH
        result['cache_setup_monotonic_seconds'] = time.monotonic()-ENTRY_MONOTONIC
        guard(); del deadlines['cache']
        scaler = FieldScaler(**science['scaler']); scale = PhysicalScale.from_mapping(science['scaler'])

        def rng():
            return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())

        def restore(value):
            random.setstate(value['python']); np.random.set_state(value['numpy'])
            torch.set_rng_state(value['torch']); torch.cuda.set_rng_state_all(value['cuda'])

        def backward(model, key):
            case = cache.case(key)
            return full_physical_backward(model, case, cache.physical_targets[key], scale, cache.case_norms[key], keys, guard)

        for method in science['methods']:
            began = (time.time(), time.monotonic()); result['phase'] = 'arm_setup'; result['current_method'] = method
            deadlines['arm'] = (began[0]+1500, began[1]+1500); deadlines['arm_setup'] = (began[0]+120, began[1]+120)
            arm_folder = output/method; arm_folder.mkdir()
            arm = dict(status='running', active=True, closed=False, method=method, step=0, endpoints={},
                entry_epoch=began[0], entry_monotonic=began[1], case_exposure_counts={key: 0 for key in keys},
                optimizer_step_attempts=0, optimizer_step_completions=0, optimizer_step_in_progress=False,
                possible_partial_optimizer_update=False, last_checkpoint=None, cached_physical_TRAIN_readout_reuses=0)
            result['methods'][method] = arm; persist(); guard()
            random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
            model = (ResponseGainTransolver(scale, 2026) if method == 'transolver_gain' else ResponseGainFNO(scale, 2026)).train()
            model.macro_family = 'R16_'+method
            assert model.profile()['parameter_count'] == science['expected_parameter_counts'][method]
            arm['model_profile'] = dict(model.profile(), native_model_method=model.profile()['method'], method=method, training_objective=science['loss'])
            arm['actual_parameter_dtypes'] = sorted({str(p.dtype) for p in model.parameters()})
            arm['kernel_precision_profiled'] = False
            saved = rng(); reference = CommonGridSolver('FNO9', 2026)
            for group in ('encoder', 'head'):
                actual = getattr(model, group).state_dict(); expected = getattr(reference, group).state_dict()
                assert set(actual) == set(expected) and all(torch.equal(v.detach().cpu(), expected[k]) for k, v in actual.items())
            arm['actual_shared_initial_encoder_head_equal_seeded_FNO'] = True
            del reference; restore(saved)
            digest = hashlib.sha256()
            for name, value in sorted(model.state_dict().items()):
                digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            arm['actual_fresh_initial_model_sha256'] = digest.hexdigest()
            assert digest.hexdigest() == science['initial_SHA256'][method], 'Exact initial state vs own physical control required'
            arm['actual_initial_state_equal_own_physical_control'] = True
            recipe = science['optimizer']
            optimizer = torch.optim.Adam(model.parameters(), lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'],
                betas=tuple(recipe['betas']), eps=recipe['eps'], fused=recipe['fused'])
            witness = max(keys, key=lambda key: manifest['records'][key]['num_nodes'])
            saved = rng(); optimizer.zero_grad(set_to_none=True)
            _, leaves = backward(model, witness); torch.cuda.synchronize()
            groups = dict(encoder=0., head=0., operator=0.); missing = []
            for name, parameter in model.named_parameters():
                if parameter.grad is None: missing.append(name); continue
                assert torch.isfinite(parameter.grad).all(), 'Nonfinite qualification gradient'
                groups[name.split('.')[0]] += float(parameter.grad.double().square().sum())
            allowed = []
            assert sorted(missing) == sorted(allowed) and all(v > 0 for v in groups.values())
            assert leaves['geometry_context_L2'] > 0 and leaves['operator_latent_L2'] > 0
            guard()
            arm['actual_largest_fullN_gradient_check'] = dict(passed=True, key=witness, nodes=manifest['records'][witness]['num_nodes'],
                groups_L2={k: v**.5 for k, v in groups.items()}, allowed_unused=missing, leaf_gradients=leaves,
                full300_physical_cache_qualified=True, optimizer_updates=0)
            optimizer.zero_grad(set_to_none=True); restore(saved)
            arm.update(setup_epoch_seconds=time.time()-began[0], setup_monotonic_seconds=time.monotonic()-began[1],
                       training_entry_epoch=time.time(), training_entry_monotonic=time.monotonic())
            del deadlines['arm_setup']; deadlines['training'] = (arm['training_entry_epoch']+900, arm['training_entry_monotonic']+900)
            result['phase'] = 'training'; persist()

            def save(step, name):
                guard()
                assert optimizer.state and all(int(state['step']) == step for state in optimizer.state.values())
                arm['saved_Adam_state_count'] = len(optimizer.state); arm['saved_Adam_steps_equal_cursor'] = True
                temporary = arm_folder/(name+'.tmp')
                torch.save(dict(step=step, cursor=step, method=method, family=model.macro_family,
                    scientific_digest=plan['scientific_digest'], model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng(),
                    case_exposure_counts=dict(arm['case_exposure_counts']), full_order_digest=science['full_order_digest'],
                    dataset_identity=science['dataset_identity'], scaler=science['scaler'],
                    native_origin_clocks=dict(epoch=ENTRY_EPOCH, monotonic=ENTRY_MONOTONIC, boot_id=boot), outer_guard_clock=outer,
                    cost_status=dict(arm), shared_cache_cost=dict(epoch=result['cache_setup_epoch_seconds'], monotonic=result['cache_setup_monotonic_seconds'])), temporary)
                os.replace(temporary, arm_folder/name)
                arm['last_checkpoint'] = dict(path=d.relative(arm_folder/name), step=step, stat_identity=d.identity(arm_folder/name)); guard()

            with (arm_folder/'loss.jsonl').open('x', encoding='utf-8') as log:
                for step, key in enumerate(order, 1):
                    guard(); optimizer.zero_grad(set_to_none=True); loss, _ = backward(model, key)
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                    arm['optimizer_step_attempts'] += 1; arm['optimizer_step_in_progress'] = True
                    optimizer.step(); torch.cuda.synchronize()
                    arm.update(step=step, optimizer_step_completions=arm['optimizer_step_completions']+1, optimizer_step_in_progress=False)
                    arm['case_exposure_counts'][key] += 1
                    log.write(json.dumps(dict(step=step, key=key, physical_relative_squared_loss=loss, training_epoch_seconds=time.time()-arm['training_entry_epoch']))+'\n'); log.flush()
                    guard()
                    if step % 300 == 0: save(step, 'last.pt')
                    if step % 30 == 0: persist()
                assert set(arm['case_exposure_counts'].values()) == {5}; save(1500, 'step1500.pt')
            arm.update(training_epoch_seconds=time.time()-arm['training_entry_epoch'], training_monotonic_seconds=time.monotonic()-arm['training_entry_monotonic'])
            guard(); del deadlines['training']; result['phase'] = 'TRAIN_endpoint_readout'
            started = (time.time(), time.monotonic()); deadlines['readout'] = (started[0]+300, started[1]+300)
            saved = rng(); model.eval(); rows = {key: None for key in keys}
            with torch.no_grad():
                for index, key in enumerate(keys):
                    guard(); torch.manual_seed(2026+index); torch.cuda.manual_seed_all(2026+index)
                    case = cache.case(key); context, latent = model.encode_case(case)
                    prediction = np.empty((len(case['xyz']), 4), dtype=np.float32)
                    for start in range(0, len(prediction), 32768):
                        guard(); stop = min(start+32768, len(prediction)); prediction[start:stop] = model.query(case, start, stop, context, latent).cpu().numpy()
                    physical = scaler.decode(prediction); assert np.isfinite(physical).all()
                    truth = cache.physical_targets[key].detach().cpu().numpy()
                    assert truth.shape == prediction.shape and np.isfinite(truth).all()
                    rows[key] = physical_metrics(np, truth, physical)
                    arm['cached_physical_TRAIN_readout_reuses'] += 1; result['cached_physical_TRAIN_readout_reuses'] += 1
                    del context, latent, prediction, physical, truth
            metric = aggregate(rows, 300); assert metric['all_cases_complete'] and metric['fixed_denominator'] == 300
            guard(); arm['endpoints']['1500'] = dict(full_TRAIN=metric, case_exposure_count=5,
                epoch_seconds=time.time()-started[0], monotonic_seconds=time.monotonic()-started[1],
                checkpoint_identity=d.identity(arm_folder/'step1500.pt', hashed=True))
            restore(saved); del deadlines['readout']
            optimizer.zero_grad(set_to_none=True); del optimizer, model; cleanup(); torch.cuda.empty_cache(); guard()
            arm.update(status='completed', active=False, closed=True, training_complete=True,
                       elapsed_epoch_seconds=time.time()-began[0], elapsed_monotonic_seconds=time.monotonic()-began[1])
            d.write(arm_folder/'summary.json', arm); d.write(arm_folder/'status.json', arm)
            del deadlines['arm']; result['phase'] = 'between_arms'; persist(); arm = None
        assert all(v['status'] == 'completed' for v in result['methods'].values())
        result.update(status='completed', phase='terminal', training_complete=True,
                      requires_separate_DEV_prediction_barrier_readout=True, capacity_matched=False)
    except Exception as error:
        if cache is not None: result['cache_receipt'] = dict(cache.receipt)
        if arm is not None:
            arm.update(status='failed', active=False, closed=True, error_type=type(error).__name__, error=str(error),
                       possible_partial_optimizer_update=arm['optimizer_step_in_progress'],
                       elapsed_epoch_seconds=time.time()-arm['entry_epoch'], elapsed_monotonic_seconds=time.monotonic()-arm['entry_monotonic'])
            if 'training' in deadlines:
                arm.update(training_epoch_seconds=time.time()-arm['training_entry_epoch'], training_monotonic_seconds=time.monotonic()-arm['training_entry_monotonic'])
            d.write(output/arm['method']/'summary.json', arm); d.write(output/arm['method']/'status.json', arm)
        result.update(status='failed', phase='terminal', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            if cleanup is not None: cleanup()
        finally:
            result.update(active=False, closed=True, elapsed_epoch_seconds=time.time()-ENTRY_EPOCH, elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
            d.write(output/'summary.json', result); d.write(output/'status.json', result)
            if lock is not None: lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--start', action='store_true'); parser.add_argument('--plan')
    args = parser.parse_args()
    if args.start:
        if not args.plan: parser.error('Frozen released --plan required')
        run(args.plan)
