"""Two fixed TRAIN-teacher node-latent operator diagnostics, not FEM updates."""
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
    from geobc.node_latent_training_contract_v36 import approved_contract
    plan, _, manifest, order = approved_contract(plan_path); science = plan['scientific']
    outer = json.loads(os.environ['GEOBC_NODE_LATENT_V36_GUARD'])
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
        backend_external_workspace_exact_peak_known=False, cached_physical_TRAIN_readout_reuses=0, training_objective=science['loss'], teacher_checkpoint_hash_attempts=0, teacher_checkpoint_load_attempts=0, teacher_checkpoint_loads=0)
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
        from geobc.node_latent_teacher_v35 import (freeze_teacher_geometry_head, complete_teacher_node_norm, full_teacher_node_backward, original_node_latent, frozen_group_digests)
        from geobc.geometry_head_transfer_v32 import state_digest
        from geobc.coarse_transolver_v19 import CoarseTransolver
        from geobc.coarse_fno_control_v22 import CoarseFNOControl
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
        endpoint = science['teacher_checkpoint']; guard()
        result['teacher_checkpoint_hash_attempts'] += 1; persist()
        assert d.identity(endpoint['path'], hashed=True) == endpoint
        result['teacher_checkpoint_load_attempts'] += 1; persist()
        teacher = torch.load(d.path(endpoint['path']), map_location='cpu', weights_only=False)
        result['teacher_checkpoint_loads'] += 1
        assert teacher['step'] == teacher['cursor'] == 1500
        assert teacher['method'] == 'FNO16_control' and teacher['family'] == 'R16_FNO16_control'
        assert teacher['scientific_digest'] == science['teacher_SCI']
        assert teacher['full_order_digest'] == science['teacher_prior_full_order_digest']
        assert teacher['case_exposure_counts'] == {key: 5 for key in keys}
        assert teacher['dataset_identity'] == science['dataset_identity'] and teacher['scaler'] == science['scaler']
        assert set(teacher['rng']) == {'python', 'numpy', 'torch', 'cuda'}
        assert teacher['optimizer']['state'] and all(int(v['step']) == 1500 for v in teacher['optimizer']['state'].values())
        assert all(torch.isfinite(v).all() for v in teacher['model'].values())
        teacher_summary = d.read(science['teacher_binding']['terminal']['summary']['path'])
        teacher_external = d.read(science['teacher_binding']['terminal']['external']['path'])
        assert teacher['native_origin_clocks']['boot_id'] == teacher_summary['execution']['boot_id']
        result['teacher_loaded_model_sha256'] = state_digest(teacher['model'])
        result['inherited_teacher_cost'] = dict(training_epoch_seconds=teacher_summary['methods']['FNO16_control']['training_epoch_seconds'],
            training_monotonic_seconds=teacher_summary['methods']['FNO16_control']['training_monotonic_seconds'],
            stage_epoch_seconds=teacher_external['elapsed_epoch_seconds'], stage_monotonic_seconds=teacher_external['elapsed_monotonic_seconds'],
            teacher_binding=science['teacher_binding'], teacher_checkpoint=endpoint, prior_passes=5,
            teacher_stage_shared_once_not_two_independent_replays=True, inter_stage_analysis_wait_included=False)
        result.update(cache_setup_epoch_seconds=time.time()-ENTRY_EPOCH, cache_setup_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        teacher_model = CoarseFNOControl(2026).eval().requires_grad_(False)
        teacher_model.load_state_dict(teacher['model'], strict=True)
        teacher_latents, teacher_contexts, teacher_norms = {}, {}, {}
        result['phase'] = 'shared_teacher_latent_cache'; result['teacher_forward_attempts'] = 0
        with torch.no_grad():
            for index, key in enumerate(keys):
                guard(); torch.manual_seed(2026+index); torch.cuda.manual_seed_all(2026+index)
                result['teacher_forward_attempts'] += 1
                case = cache.case(key); context, latent = teacher_model.encode_case(case)
                assert torch.isfinite(context).all() and torch.isfinite(latent).all()
                teacher_contexts[key] = context.detach(); teacher_latents[key] = latent.detach()
                teacher_norms[key] = complete_teacher_node_norm(teacher_latents[key], case['xyz'], guard)
                if index % 30 == 0: persist()
        assert len(teacher_latents) == len(teacher_norms) == 300
        result['teacher_latent_cache_bytes'] = sum(v.numel()*v.element_size() for v in teacher_latents.values())
        result['teacher_context_cache_bytes'] = sum(v.numel()*v.element_size() for v in teacher_contexts.values())
        result['teacher_latent_cache_complete'] = True
        d.write(output/'teacher_node_norms.json', {k:asdict(v) for k,v in teacher_norms.items()})
        result['teacher_node_norms_identity'] = d.identity(output/'teacher_node_norms.json', hashed=True)
        del teacher_model; torch.cuda.empty_cache()
        result.update(cache_setup_epoch_seconds=time.time()-ENTRY_EPOCH, cache_setup_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        guard(); del deadlines['cache']
        scaler = FieldScaler(**science['scaler'])

        def rng():
            return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())

        def restore(value):
            random.setstate(value['python']); np.random.set_state(value['numpy'])
            torch.set_rng_state(value['torch']); torch.cuda.set_rng_state_all(value['cuda'])

        def backward(model, key):
            case = cache.case(key)
            return full_teacher_node_backward(model, case, teacher_latents[key], teacher_norms[key], keys, guard)

        teacher_physical_cache = {}; teacher_FEM_rows = {}
        result['teacher_physical_forward_attempts'] = 0

        def latent_case_loss(latent, key):
            total = 0.; case = cache.case(key); count = len(case['xyz'])
            for start in range(0, count, 32768):
                guard(); xyz = case['xyz'][start:min(start+32768, count)]
                values = original_node_latent(latent, xyz)
                target = original_node_latent(teacher_latents[key], xyz)
                total += float((values.double()-target.double()).square().sum())/teacher_norms[key].squared_norm
            assert np.isfinite(total) and total >= 0
            return total

        def latent_aggregate(rows):
            assert set(rows) == set(keys) and all(np.isfinite(v) and v >= 0 for v in rows.values())
            return dict(fixed_denominator=300, all_cases_complete=True, per_case_relative_squared=rows,
                case_mean_relative_squared=sum(rows.values())/300,
                case_mean_relative_L2=sum(v**.5 for v in rows.values())/300,
                worst_case_relative_L2=max(rows.values())**.5, target_is_teacher_not_FEM=True)

        for method in science['methods']:
            began = (time.time(), time.monotonic()); result['phase'] = 'arm_setup'; result['current_method'] = method
            deadlines['arm'] = (began[0]+1500, began[1]+1500); deadlines['arm_setup'] = (began[0]+120, began[1]+120)
            initial_step = science['initial_steps'][method]; endpoint_step = science['endpoint_steps'][method]
            arm_folder = output/method; arm_folder.mkdir()
            arm = dict(status='running', active=True, closed=False, method=method, step=initial_step, endpoints={},
                entry_epoch=began[0], entry_monotonic=began[1], case_exposure_counts={key: initial_step//300 for key in keys}, inherited_teacher_case_exposure_counts={key: 5 for key in keys}, additional_optimizer_step_attempts=0, additional_optimizer_step_completions=0, parent_optimizer_steps=initial_step,
                optimizer_step_attempts=initial_step, optimizer_step_completions=initial_step, optimizer_step_in_progress=False,
                possible_partial_optimizer_update=False, last_checkpoint=None, cached_physical_TRAIN_readout_reuses=0)
            result['methods'][method] = arm; persist(); guard()
            random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
            model = (CoarseTransolver('transolver', 2026) if method == 'transolver_latent_teacher' else CoarseFNOControl(2026)).train()
            model.macro_family = 'R16_'+method
            assert model.profile()['parameter_count'] == science['expected_parameter_counts'][method]
            arm['model_profile'] = dict(model.profile(), native_model_method=model.profile()['method'], method=method, training_objective=science['loss'])
            arm['actual_parameter_dtypes'] = sorted({str(p.dtype) for p in model.parameters()})
            arm['kernel_precision_profiled'] = False
            arm['actual_fresh_initial_model_sha256'] = state_digest(model.state_dict())
            assert arm['actual_fresh_initial_model_sha256'] == science['fresh_initial_SHA256'][method]
            arm['actual_fresh_initial_state_matches_original_control'] = True
            arm['actual_transfer'] = freeze_teacher_geometry_head(model, teacher['model'])
            assert arm['actual_transfer']['fresh_model_before'] == arm['actual_fresh_initial_model_sha256']
            assert arm['actual_transfer']['teacher_state_digest'] == result['teacher_loaded_model_sha256']
            arm['frozen_group_initial_digests'] = frozen_group_digests(model)
            model.encoder.eval(); model.head.eval()
            arm['actual_training_initial_model_sha256'] = state_digest(model.state_dict())
            recipe = science['optimizer']
            optimizer = torch.optim.Adam(model.operator.parameters(), lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'],
                betas=tuple(recipe['betas']), eps=recipe['eps'], fused=recipe['fused'])
            assert not optimizer.state
            arm['actual_reset_Adam_empty'] = True
            arm['trainable_parameter_groups'] = ['operator']; arm['frozen_parameter_groups'] = ['encoder','head']
            witness = max(keys, key=lambda key: manifest['records'][key]['num_nodes'])
            saved = rng(); optimizer.zero_grad(set_to_none=True)
            _, leaves = backward(model, witness); torch.cuda.synchronize()
            groups = dict(encoder=0., head=0., operator=0.); missing = []
            for name, parameter in model.named_parameters():
                if parameter.grad is None: missing.append(name); continue
                assert torch.isfinite(parameter.grad).all(), 'Nonfinite qualification gradient'
                groups[name.split('.')[0]] += float(parameter.grad.double().square().sum())
            allowed = [name for name,p in model.named_parameters() if not p.requires_grad]
            assert sorted(missing) == sorted(allowed)
            assert groups['operator'] > 0 and groups['encoder'] == groups['head'] == 0
            assert leaves['operator_latent_gradient_L2'] > 0
            assert frozen_group_digests(model) == arm['frozen_group_initial_digests']
            guard()
            arm['actual_largest_fullN_gradient_check'] = dict(passed=True, key=witness, nodes=manifest['records'][witness]['num_nodes'],
                groups_L2={k: v**.5 for k, v in groups.items()}, allowed_unused=missing, leaf_gradients=leaves,
                full300_physical_cache_qualified=True, optimizer_updates=0)
            optimizer.zero_grad(set_to_none=True)
            model.eval(); initial_rows = {}
            with torch.no_grad():
                for index, key in enumerate(keys):
                    guard(); torch.manual_seed(2026+index); torch.cuda.manual_seed_all(2026+index)
                    context, latent = model.encode_case(cache.case(key))
                    assert torch.equal(context, teacher_contexts[key])
                    initial_rows[key] = latent_case_loss(latent, key)
            arm['initial_fullTRAIN_teacher_node_latent'] = latent_aggregate(initial_rows)
            model.train(); model.encoder.eval(); model.head.eval(); restore(saved)
            arm.update(setup_epoch_seconds=time.time()-began[0], setup_monotonic_seconds=time.monotonic()-began[1],
                       training_entry_epoch=time.time(), training_entry_monotonic=time.monotonic())
            del deadlines['arm_setup']; deadlines['training'] = (arm['training_entry_epoch']+900, arm['training_entry_monotonic']+900)
            result['phase'] = 'training'; persist()

            def save(step, name):
                guard()
                assert frozen_group_digests(model) == arm['frozen_group_initial_digests']
                assert optimizer.state and all(int(state['step']) == step for state in optimizer.state.values())
                arm['saved_Adam_state_count'] = len(optimizer.state); arm['saved_Adam_steps_equal_cursor'] = True
                temporary = arm_folder/(name+'.tmp')
                torch.save(dict(step=step, cursor=step, method=method, family=model.macro_family,
                    scientific_digest=plan['scientific_digest'], model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng(),
                    case_exposure_counts=dict(arm['case_exposure_counts']), full_order_digest=science['arm_full_order_digests'][method], marginal_order_digest=science['marginal_order_digest'], initialization_teacher_checkpoint=science['teacher_checkpoint'], initialization_teacher_SCI=science['teacher_SCI'], inherited_teacher_case_exposure_counts={key: 5 for key in keys}, inherited_teacher_cost=result['inherited_teacher_cost'],
                    dataset_identity=science['dataset_identity'], scaler=science['scaler'], teacher_node_norms_identity=result['teacher_node_norms_identity'], frozen_group_digests=arm['frozen_group_initial_digests'], frozen_groups=['encoder','head'], trainable_groups=['operator'], training_objective=science['loss'], FEM_used_for_updates=False,
                    native_origin_clocks=dict(epoch=ENTRY_EPOCH, monotonic=ENTRY_MONOTONIC, boot_id=boot), outer_guard_clock=outer,
                    cost_status=dict(arm), shared_cache_cost=dict(epoch=result['cache_setup_epoch_seconds'], monotonic=result['cache_setup_monotonic_seconds'])), temporary)
                os.replace(temporary, arm_folder/name)
                arm['last_checkpoint'] = dict(path=d.relative(arm_folder/name), step=step, stat_identity=d.identity(arm_folder/name)); guard()

            with (arm_folder/'loss.jsonl').open('x', encoding='utf-8') as log:
                for additional_step, key in enumerate(order, 1):
                    step = initial_step+additional_step
                    guard(); optimizer.zero_grad(set_to_none=True); loss, _ = backward(model, key)
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.operator.parameters())
                    assert all(p.grad is None for group in ('encoder','head') for p in getattr(model,group).parameters())
                    arm['optimizer_step_attempts'] += 1; arm['additional_optimizer_step_attempts'] += 1; arm['optimizer_step_in_progress'] = True
                    optimizer.step(); torch.cuda.synchronize()
                    arm.update(step=step, optimizer_step_completions=arm['optimizer_step_completions']+1, additional_optimizer_step_completions=arm['additional_optimizer_step_completions']+1, optimizer_step_in_progress=False)
                    arm['case_exposure_counts'][key] += 1
                    log.write(json.dumps(dict(step=step, key=key, teacher_node_latent_relative_squared_loss=loss, additional_step=additional_step, training_epoch_seconds=time.time()-arm['training_entry_epoch']))+'\n'); log.flush()
                    guard()
                    if step % 300 == 0: save(step, 'last.pt')
                    if step % 30 == 0: persist()
                assert set(arm['case_exposure_counts'].values()) == {endpoint_step//300}; save(endpoint_step, 'step'+str(endpoint_step)+'.pt')
            arm.update(training_epoch_seconds=time.time()-arm['training_entry_epoch'], training_monotonic_seconds=time.monotonic()-arm['training_entry_monotonic'])
            arm['teacher_plus_marginal_training_epoch_seconds'] = result['inherited_teacher_cost']['training_epoch_seconds']+arm['training_epoch_seconds']
            arm['teacher_plus_marginal_training_monotonic_seconds'] = result['inherited_teacher_cost']['training_monotonic_seconds']+arm['training_monotonic_seconds']
            guard(); del deadlines['training']; result['phase'] = 'TRAIN_endpoint_readout'
            started = (time.time(), time.monotonic()); deadlines['readout'] = (started[0]+300, started[1]+300)
            saved = rng(); model.eval(); rows = {key: None for key in keys}; teacher_rows = {key: None for key in keys}; latent_rows = {}
            with torch.no_grad():
                for index, key in enumerate(keys):
                    guard(); torch.manual_seed(2026+index); torch.cuda.manual_seed_all(2026+index)
                    case = cache.case(key); context, latent = model.encode_case(case)
                    assert torch.equal(context, teacher_contexts[key])
                    latent_rows[key] = latent_case_loss(latent, key)
                    prediction = np.empty((len(case['xyz']), 4), dtype=np.float32)
                    for start in range(0, len(prediction), 32768):
                        guard(); stop = min(start+32768, len(prediction)); prediction[start:stop] = model.query(case, start, stop, context, latent).cpu().numpy()
                    physical = scaler.decode(prediction); assert np.isfinite(physical).all()
                    truth = cache.physical_targets[key].detach().cpu().numpy()
                    assert truth.shape == prediction.shape and np.isfinite(truth).all()
                    rows[key] = physical_metrics(np, truth, physical)
                    if key not in teacher_physical_cache:
                        result['teacher_physical_forward_attempts'] += 1
                        reference = np.empty_like(prediction)
                        for start in range(0, len(reference), 32768):
                            guard(); stop = min(start+32768, len(reference))
                            reference[start:stop] = model.query(case, start, stop, teacher_contexts[key], teacher_latents[key]).cpu().numpy()
                        teacher_physical_cache[key] = scaler.decode(reference)
                        assert np.isfinite(teacher_physical_cache[key]).all()
                        teacher_FEM_rows[key] = physical_metrics(np, truth, teacher_physical_cache[key])
                        result['teacher_physical_forward_cases'] = len(teacher_physical_cache)
                    teacher_rows[key] = physical_metrics(np, teacher_physical_cache[key], physical)
                    arm['cached_physical_TRAIN_readout_reuses'] += 1; result['cached_physical_TRAIN_readout_reuses'] += 1
                    del context, latent, prediction, physical, truth
            metric = aggregate(rows, 300); assert metric['all_cases_complete'] and metric['fixed_denominator'] == 300
            assert frozen_group_digests(model) == arm['frozen_group_initial_digests']
            guard(); arm['endpoints'][str(endpoint_step)] = dict(full_TRAIN=metric, full_TRAIN_teacher_physical=aggregate(teacher_rows,300), full_TRAIN_teacher_node_latent=latent_aggregate(latent_rows), case_exposure_count=endpoint_step//300, additional_passes=5, additional_pass_kind='teacher_latent_target_not_FEM', teacher_derived_initialization_passes=5,
                epoch_seconds=time.time()-started[0], monotonic_seconds=time.monotonic()-started[1],
                checkpoint_identity=d.identity(arm_folder/('step'+str(endpoint_step)+'.pt'), hashed=True))
            restore(saved); del deadlines['readout']
            optimizer.zero_grad(set_to_none=True); del optimizer, model; cleanup(); torch.cuda.empty_cache(); guard()
            arm.update(status='completed', active=False, closed=True, training_complete=True,
                       elapsed_epoch_seconds=time.time()-began[0], elapsed_monotonic_seconds=time.monotonic()-began[1])
            d.write(arm_folder/'summary.json', arm); d.write(arm_folder/'status.json', arm)
            del deadlines['arm']; result['phase'] = 'between_arms'; persist(); arm = None
        assert all(v['status'] == 'completed' for v in result['methods'].values())
        assert len(teacher_physical_cache) == 300
        result['full_TRAIN_teacher_to_original_FEM'] = aggregate(teacher_FEM_rows,300)
        result['teacher_physical_cache_host_bytes'] = sum(v.nbytes for v in teacher_physical_cache.values())
        result['FEM_used_for_optimizer_updates'] = False
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
