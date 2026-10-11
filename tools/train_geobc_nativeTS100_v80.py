"""One fresh native originalTS model, fixed300 TRAIN fields, endpoints5/10."""
import time
ENTRY_EPOCH, ENTRY_MONOTONIC = time.time(), time.monotonic()
import argparse
import json
import os
import signal
from pathlib import Path


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.nativeTS100_contract_v80 import approved_contract
    from geobc.transolver_decoder_contract_v64 import complete_physical
    p, r, m, up = approved_contract(plan_path); s = p['scientific']; keys = s['TRAIN_keys']
    outer = json.loads(os.environ['GEOBC_NATIVE_TS100_V80_GUARD'])
    assert outer['scientific_digest'] == p['scientific_digest']
    assert outer['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid']) == outer['guard_process']
    output = d.path(p['output']); assert output.exists() and not (output/'execution.json').exists()
    execution = dict(run_process=d.process_snapshot(os.getpid()), guard=outer,
        boot_id=outer['boot_id'], scientific_digest=p['scientific_digest'])
    d.write(output/'execution.json', execution)
    a = dict(status='running', active=True, closed=False, phase='cache_setup', execution=execution,
        step=0, cursor=0, optimizer_step_attempts=0, optimizer_step_completions=0,
        optimizer_step_in_progress=False, possible_partial_optimizer_update=False,
        case_exposure_counts={k: 0 for k in keys}, endpoints={}, physical_TRAIN_reads=0,
        physical_TRAIN_attempts=0, parent_checkpoint_hash_reads=0, parent_checkpoint_loads=0,
        encoded_TRAIN_reads=0, raw_GT_reads=0, DEV_FINAL_reads=0, scaler_fits=0,
        resident_physical_target_readout_reuses=0, peak_allocated_bytes=0,
        peak_reserved_bytes=0, peak_total_device_used_bytes_sampled=0,
        backend_external_workspace_exact_peak_known=False, automatic_replay=False,
        automatic_continuation=False, final_solver_selected=False)
    torch = None; cache = None; lock = None
    phase_deadline = (ENTRY_EPOCH+s['cache_budget_seconds'], ENTRY_MONOTONIC+s['cache_budget_seconds'])

    def guard():
        limits = [(outer['deadline_epoch'], outer['deadline_monotonic'])]
        if phase_deadline is not None:
            limits.append(phase_deadline)
        if min(min(x-time.time(), y-time.monotonic()) for x, y in limits) <= 0:
            raise TimeoutError('Earlier frozen epoch/monotonic deadline reached')
        if torch is not None and torch.cuda.is_initialized():
            a['peak_allocated_bytes'] = max(a['peak_allocated_bytes'], torch.cuda.max_memory_allocated())
            a['peak_reserved_bytes'] = max(a['peak_reserved_bytes'], torch.cuda.max_memory_reserved())
            free, total = torch.cuda.mem_get_info()
            a['peak_total_device_used_bytes_sampled'] = max(a['peak_total_device_used_bytes_sampled'], total-free)
            if max(a[k] for k in ('peak_allocated_bytes', 'peak_reserved_bytes', 'peak_total_device_used_bytes_sampled')) > s['memory_limit_bytes']:
                raise MemoryError('30.5GB SI budget exceeded')

    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,
                 elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        if cache is not None:
            a['cache_receipt'] = dict(cache.receipt)
        d.write(output/'status.json', a)

    signal.signal(signal.SIGTERM, lambda *args: (_ for _ in ()).throw(TimeoutError('External guard terminated child')))
    try:
        guard()
        import fcntl
        lock = d.path(s['GPU_lock']).open('a+'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        import random
        import numpy as np
        import torch as torch_module
        torch = torch_module
        from geobc.nativeTS100_cache_v80 import NativePhysicalCache
        from geobc.transolver_native_fit_v69 import build_solver, profile
        from geobc.transolver_native_hash_v68 import full_native_physical_backward
        from geobc.physical_field_loss import PhysicalScale
        from geobc.fno_stress_specialist_v54 import tree_digest
        from geobc.coverage_comparison_train import physical_metrics, aggregate
        from geobc.benchmark import FieldScaler
        from geobc.macro_training import balanced_schedule
        torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        assert s['memory_limit_bytes'] < total
        torch.cuda.set_per_process_memory_fraction(s['memory_limit_bytes']/total)
        torch.cuda.reset_peak_memory_stats()

        def cache_event(stage, name, identity):
            guard()
            if cache is not None and stage == 'completed':
                count = cache.receipt['input_array_reads']+cache.receipt['physical_TRAIN_target_reads']
                if count % 50 == 0:
                    persist()

        cache = NativePhysicalCache(m, memory_limit_bytes=s['memory_limit_bytes'], guard=guard, on_read=cache_event)
        cache.load()
        assert cache.completed and cache.keys == keys
        a['physical_TRAIN_attempts'] = cache.receipt['physical_TRAIN_target_read_attempts']
        a['physical_TRAIN_reads'] = cache.receipt['physical_TRAIN_target_reads']
        assert a['physical_TRAIN_attempts'] == a['physical_TRAIN_reads'] == 300
        assert cache.receipt['encoded_TRAIN_target_reads'] == 0
        d.write(output/'input_cache_index.json', cache.specs)
        d.write(output/'physical_cache_index.json', cache.physical_specs)
        a.update(cache_setup_epoch_seconds=time.time()-ENTRY_EPOCH,
            cache_setup_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC,
            training_entry_epoch=time.time(), training_entry_monotonic=time.monotonic(), phase='initialization_and_qualification')
        phase_deadline = (a['training_entry_epoch']+s['training_budget_seconds'],
                          a['training_entry_monotonic']+s['training_budget_seconds'])
        random.seed(s['seed']); np.random.seed(s['seed']); torch.manual_seed(s['seed']); torch.cuda.manual_seed_all(s['seed'])
        model = build_solver(s['family'], seed=s['seed']).train()
        a['profile'] = profile(model)
        assert model.backend == 'tcnn_fully_fused' and a['profile']['parameter_count'] == 9601208
        assert not model.operator.source_config['architecture_overrides']
        assert all(model.operator.source_config['constructor_kwargs'][k] == v for k, v in s['official_TS_kwargs'].items())
        assert model.operator.input_dim == 61 and model.operator.output_dim == 16

        def rng():
            return dict(python=random.getstate(), numpy=np.random.get_state(),
                        torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())

        def restore(state):
            random.setstate(state['python']); np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])

        initial_rng = rng(); md = tree_digest(model.state_dict()); rd = tree_digest(initial_rng)
        expected = s['expected_FP32_TS_initialization']
        assert md == expected['model_digest'] and rd == expected['RNG_digest']
        recipe = s['optimizer']
        optimizer = torch.optim.Adam(model.parameters(), lr=recipe['learning_rate'],
            weight_decay=recipe['weight_decay'], betas=tuple(recipe['betas']), eps=recipe['eps'], fused=recipe['fused'])
        assert not optimizer.state
        ad = tree_digest(optimizer.state_dict())
        key = s['qualification_TRAIN_key']; case = cache.case(key)
        assert len(case['xyz']) == 325387 and not {'target', 'grid'} & set(case)
        scale = PhysicalScale.from_mapping(s['scaler']); scaler = FieldScaler(**s['scaler'])
        _, leaves = full_native_physical_backward(model, case, cache.physical_targets[key], scale, cache.case_norms[key], guard)
        groups = dict(encoder=0., operator=0., decoder=0.)
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            group = 'encoder' if name.startswith('encoder.') else 'operator' if name.startswith('operator.') else 'decoder'
            groups[group] += float(parameter.grad.detach().double().square().sum())
        assert all(v > 0 for v in groups.values()) and leaves['context'] > 0 and leaves['latent'] > 0
        optimizer.zero_grad(set_to_none=True); restore(initial_rng)
        assert tree_digest(model.state_dict()) == md and tree_digest(optimizer.state_dict()) == ad and tree_digest(rng()) == rd
        a['qualification'] = dict(passed=True, key=key, nodes=len(case['xyz']),
            parameter_groups_squared_L2=groups, leaf_gradients=leaves,
            initial_dense_TS_model_digest=md, initial_dense_TS_RNG_digest=rd,
            exact_dense_initial_bytes=True, model_unchanged=True, Adam_unchanged=True,
            fourRNG_restored=True, qualification_optimizer_updates=0)
        del case, initial_rng
        schedule = balanced_schedule(keys, 10, 2026)
        assert d.digest(schedule) == s['full_order_digest']
        a['phase'] = 'training'; persist()

        def save(step, filename):
            assert optimizer.state and all(int(v['step']) == step for v in optimizer.state.values())
            file = output/'checkpoints'/filename; file.parent.mkdir(parents=True, exist_ok=True)
            temporary = file.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng(),
                family=s['family'], step=step, cursor=step, case_exposure_counts=dict(a['case_exposure_counts']),
                scientific_digest=p['scientific_digest'], dataset_identity=s['dataset_identity'], scaler=s['scaler'],
                full_order_digest=s['full_order_digest'], execution=execution,
                native_origin=dict(entry_epoch=outer['entry_epoch'], entry_monotonic=outer['entry_monotonic'],
                    deadline_epoch=outer['deadline_epoch'], deadline_monotonic=outer['deadline_monotonic']),
                costs_at_checkpoint=dict(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,
                    elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)), temporary)
            os.replace(temporary, file)
            a['last_checkpoint'] = dict(step=step, identity=d.identity(file)); guard()

        def readout(step):
            nonlocal phase_deadline
            previous_deadline = phase_deadline
            start_epoch, start_mono = time.time(), time.monotonic()
            read_deadline = (start_epoch+s['each_TRAIN_readout_budget_seconds'], start_mono+s['each_TRAIN_readout_budget_seconds'])
            phase_deadline = read_deadline if previous_deadline is None else (
                min(previous_deadline[0], read_deadline[0]), min(previous_deadline[1], read_deadline[1]))
            state = rng(); model.eval(); a['phase'] = 'TRAIN_readout'; persist()
            try:
                rows = {}
                with torch.no_grad():
                    for key in keys:
                        guard(); index = keys.index(key)
                        torch.manual_seed(2026+index); torch.cuda.manual_seed_all(2026+index)
                        one = cache.case(key); context, latent = model.encode_case(one); n = len(one['xyz'])
                        prediction = np.empty((n, 4), dtype=np.float32)
                        for start in range(0, n, 32768):
                            guard(); stop = min(start+32768, n)
                            prediction[start:stop] = model.query(one, start, stop, context, latent).cpu().numpy()
                        physical = scaler.decode(prediction); assert np.isfinite(physical).all()
                        rows[key] = physical_metrics(np, cache.physical_targets[key].detach().cpu().numpy(), physical)
                        a['resident_physical_target_readout_reuses'] += 1
                        del context, latent, prediction, physical
                metric = aggregate(rows, 300); complete_physical(metric, keys)
                a['endpoints'][str(step)] = dict(full_TRAIN=metric, fixed_passes=step//300,
                    readout_epoch_seconds=time.time()-start_epoch, readout_monotonic_seconds=time.monotonic()-start_mono,
                    checkpoint_identity=d.identity(output/'checkpoints'/('step'+str(step)+'.pt'), hashed=True))
            finally:
                restore(state); model.train(); phase_deadline = previous_deadline
            a['phase'] = 'training'; guard(); persist()

        with (output/'loss.jsonl').open('x', encoding='utf-8') as log:
            for step, key in enumerate(schedule, 1):
                guard(); optimizer.zero_grad(set_to_none=True)
                loss, _ = full_native_physical_backward(model, cache.case(key), cache.physical_targets[key], scale, cache.case_norms[key], guard)
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
                a['optimizer_step_attempts'] += 1; a['optimizer_step_in_progress'] = True
                optimizer.step(); torch.cuda.synchronize()
                a['optimizer_step_completions'] += 1; a['optimizer_step_in_progress'] = False
                a.update(step=step, cursor=step); a['case_exposure_counts'][key] += 1
                log.write(json.dumps(dict(step=step, key=key, physical_relative_squared=loss,
                    training_epoch_seconds=time.time()-a['training_entry_epoch'],
                    training_monotonic_seconds=time.monotonic()-a['training_entry_monotonic']))+'\n'); log.flush()
                guard()
                if step % 300 == 0:
                    save(step, 'last.pt')
                if step in s['fixed_steps']:
                    assert a['case_exposure_counts'] == {k: step//300 for k in keys}
                    save(step, 'step'+str(step)+'.pt')
                    if step == 3000:
                        a.update(training_epoch_seconds=time.time()-a['training_entry_epoch'],
                            training_monotonic_seconds=time.monotonic()-a['training_entry_monotonic'])
                        phase_deadline = None
                    readout(step)
                if step % 30 == 0:
                    if step != 3000:
                        a.update(training_epoch_seconds=time.time()-a['training_entry_epoch'],
                            training_monotonic_seconds=time.monotonic()-a['training_entry_monotonic'])
                    persist()
        assert a['optimizer_step_attempts'] == a['optimizer_step_completions'] == a['step'] == 3000
        assert a['case_exposure_counts'] == {k: 10 for k in keys}
        assert a['resident_physical_target_readout_reuses'] == 600
        v = a['endpoints']['3000']['full_TRAIN']; b = s['reference_TRAIN10']; th = s['gate']
        du = v['case_mean_u_relative_l2']-b['case_mean_u_relative_l2']
        ds = v['case_mean_stress_relative_l2']-b['case_mean_stress_relative_l2']
        dh = v['case_mean_stress_hotspot_relative_l2']-b['case_mean_stress_hotspot_relative_l2']
        dp = .5*(du+ds)
        a['TRAIN_gate'] = dict(passed=dp <= -th['primary_improvement_min'] and ds <= -th['stress_improvement_min']
            and max(du, ds) <= th['field_deterioration_max'] and dh <= th['hotspot_deterioration_max'],
            U_difference=du, S_difference=ds, primary_difference=dp, hotspot_difference=dh,
            thresholds=th, scope='TRAIN300 only; independent knownDEV comparison release requires root',
            releases_automatic_continuation=False)
        guard(); a.update(status='completed', phase='terminal', training_complete=True,
            protocol_qualified=True, fitting_only=True, final_solver_selected=False)
    except BaseException as error:
        a.update(status='failed', phase='terminal', error_type=type(error).__name__, error=str(error),
            possible_partial_optimizer_update=a['optimizer_step_in_progress'])
        raise
    finally:
        if cache is not None:
            a['cache_receipt'] = dict(cache.receipt)
        a.update(active=False, closed=True, elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,
                 elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(output/'summary.json', a); d.write(output/'status.json', a)
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--start', action='store_true'); parser.add_argument('--plan')
    args = parser.parse_args()
    if args.start:
        if not args.plan:
            parser.error('--start requires a separately released frozen --plan')
        run(args.plan)
    else:
        print('{"status":"source_only","scientific_bodies":0,"GPU":0}')
