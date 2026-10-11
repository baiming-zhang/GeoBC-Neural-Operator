"""Two physical residual decoders on one actual frozen TRAIN-only FNO5."""
import time
ENTRY_EPOCH, ENTRY_MONOTONIC = time.time(), time.monotonic()
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import signal


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.operator_residual_training_contract_v45 import approved_contract
    plan, _, manifest, order = approved_contract(plan_path); science = plan['scientific']
    outer = json.loads(os.environ['GEOBC_OPERATOR_RESIDUAL_V45_GUARD'])
    assert outer['scientific_digest'] == plan['scientific_digest']
    assert d.process_snapshot(outer['guard_process']['pid']) == outer['guard_process']
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip(); assert boot == outer['boot_id']
    output = d.path(plan['output']); assert not (output/'execution.json').exists()
    execution = dict(run_process=d.process_snapshot(os.getpid()), guard=outer, boot_id=boot,
                     scientific_digest=plan['scientific_digest'])
    d.write(output/'execution.json', execution)
    keys = science['TRAIN_keys']; cap = science['memory_limit_bytes']
    result = dict(status='running', active=True, closed=False, execution=execution, phase='shared_cache_teacher_setup',
        methods={}, cache_load_complete=False, shared_cache_load_attempts=0, raw_GT_reads=0,
        DEV_FINAL_reads=0, scaler_fits=0, automatic_replay=False, peak_allocated_bytes=0,
        peak_reserved_bytes=0, peak_total_device_used_bytes_sampled=0, backend_external_workspace_exact_peak_known=False,
        teacher_PT_hash_reads=0, teacher_PT_numeric_loads=0, teacher_feature_forward_attempts=0,
        teacher_feature_forward_completions=0, teacher_latent_tensor_bytes=0, released_original_grid_bytes=0,
        released_original_grid_cases=0, cached_physical_TRAIN_readout_reuses=0, teacher_old_cost=science['teacher_old_cost'],
        exposure_scope='Shared frozen teacher previously5; each mutable decoder additional5; reference0. Not fresh5/allparameters10.')
    torch = None; cache = None; lock = None; cleanup = None; arm = None
    deadlines = {'cache_teacher': (ENTRY_EPOCH+600, ENTRY_MONOTONIC+600)}
    signal.signal(signal.SIGTERM, lambda *args: (_ for _ in ()).throw(TimeoutError('External guard terminated campaign')))

    def guard():
        limits = [(outer['deadline_epoch'], outer['deadline_monotonic'])] + list(deadlines.values())
        if min(min(a-time.time(), b-time.monotonic()) for a,b in limits) <= 0:
            raise TimeoutError('Frozen campaign/phase dual-clock deadline')
        if torch is not None and torch.cuda.is_initialized():
            for name,value in [('peak_allocated_bytes',torch.cuda.max_memory_allocated()),('peak_reserved_bytes',torch.cuda.max_memory_reserved())]:
                result[name] = max(result[name],value)
            free,total = torch.cuda.mem_get_info()
            result['peak_total_device_used_bytes_sampled'] = max(result['peak_total_device_used_bytes_sampled'],total-free)
            if max(result[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled')) > cap:
                raise MemoryError('Campaign exceeds30.5GB SI allocated/reserved/sampled total device')

    def persist():
        result.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        if arm is not None and 'training' in deadlines:
            arm.update(training_epoch_seconds=time.time()-arm['training_entry_epoch'],training_monotonic_seconds=time.monotonic()-arm['training_entry_monotonic'])
        d.write(output/'status.json',result)

    try:
        import fcntl
        lock = d.path(science['GPU_lock']).open('a+'); fcntl.flock(lock.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        import numpy as np
        import torch as torch_module
        torch = torch_module
        from dataclasses import asdict
        from geobc.train_physical_cache_v12 import PhysicalTRAINCache
        from geobc.common_grid_solver import CommonGridSolver
        from geobc.operator_residual_v44 import FrozenOperatorResidual
        from geobc.residual_physical_training_v45 import full_residual_case_backward
        from geobc.geometry_head_transfer_v32 import state_digest
        from geobc.physical_field_loss import PhysicalScale
        from geobc.benchmark import FieldScaler
        from geobc.coverage_comparison_train import physical_metrics,aggregate
        from geobc.transformers import close_owned_process_group
        cleanup = close_owned_process_group
        torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory; assert cap < total
        torch.cuda.set_per_process_memory_fraction(cap/total); torch.cuda.reset_peak_memory_stats()
        guard(); persist()

        identity = science['teacher_checkpoint']; d.stat_matches(identity)
        result['teacher_PT_hash_reads'] += 1; assert d.sha(identity['path']) == identity['sha256']; guard()
        result['teacher_PT_numeric_loads'] += 1
        parent = torch.load(d.path(identity['path']),map_location='cpu',weights_only=False)
        bp = d.read(science['teacher_terminal']['plan']['path']); bs = bp['scientific']
        assert parent['step'] == parent['cursor'] == 1500
        assert parent['scientific_digest'] == bp['scientific_digest']
        assert parent['dataset_identity'] == science['dataset_identity'] and parent['scaler'] == science['scaler']
        assert parent['case_exposure_counts'] == {k:5 for k in keys} and parent['full_order_digest'] == science['full_order_digest']
        assert set(parent['rng']) == {'python','numpy','torch','cuda'}
        assert parent['optimizer']['state'] and all(int(x['step']) == 1500 for x in parent['optimizer']['state'].values())
        assert all(torch.isfinite(value).all() for value in parent['model'].values())
        teacher = CommonGridSolver('FNO9',2026).eval().requires_grad_(False)
        teacher.load_state_dict(parent['model'],strict=True); teacher.to('cuda')
        teacher_hash = state_digest(teacher.state_dict()); assert teacher_hash == state_digest(parent['model'])
        result['actual_teacher_trained_state_digest'] = teacher_hash
        result['actual_teacher_checkpoint_qualification'] = dict(passed=True,identity=identity,step=1500,
            Adam_steps_validated=True,four_RNG_classes_present=True,all300_exposure5=True,old_Adam_inherited=False,
            old_RNG_restored=False,teacher_state_matches_checkpoint=True,teacher_parameters_frozen=True)
        del parent; guard()

        def cache_event(stage,name,identity):
            if stage == 'completed':
                result['cache_receipt'] = dict(cache.receipt)
                count = sum(cache.receipt[k] for k in ('input_array_reads','encoded_TRAIN_target_reads','physical_TRAIN_target_reads'))
                if count % 20 == 0: persist()
        cache = PhysicalTRAINCache(manifest,100,'FNO9',memory_limit_bytes=cap,guard=guard,on_read=cache_event)
        result['shared_cache_load_attempts'] += 1; persist(); cache.load()
        assert cache.completed and cache.receipt['physical_TRAIN_loaded_cases'] == 300
        result.update(cache_load_complete=True,cache_receipt=dict(cache.receipt),cache_metadata_profile=cache.metadata_profile())
        d.write(output/'cache_index.json',cache.specs); d.write(output/'physical_cache_index.json',cache.physical_specs)
        d.write(output/'physical_case_norms.json',{k:asdict(v) for k,v in cache.case_norms.items()})
        features = {}; references = Counter(cache.case_specs[k]['grid'] for k in keys)
        # Consume each ORIGINAL cached input grid once, then release its last
        # tensor reference. Preserve native xyz/features/original query stencils.
        with torch.no_grad():
            for index,key in enumerate(keys):
                guard(); case = cache.case(key); result['teacher_feature_forward_attempts'] += 1
                context,latent = teacher.encode_case(case)
                assert context.shape == (1,32) and latent.shape == (1,16,64,64,64)
                assert torch.isfinite(context).all() and torch.isfinite(latent).all()
                assert not context.requires_grad and not latent.requires_grad
                features[key] = (context.detach(),latent.detach())
                result['teacher_latent_tensor_bytes'] += sum(x.numel()*x.element_size() for x in features[key])
                result['teacher_feature_forward_completions'] += 1
                path = cache.case_specs[key]['grid']; del case['grid']; references[path] -= 1
                if references[path] == 0:
                    tensor = cache.arrays.pop(path)
                    result['released_original_grid_bytes'] += tensor.numel()*tensor.element_size(); del tensor
                result['released_original_grid_cases'] += 1
                del context,latent
                if index % 20 == 0: persist()
                guard()
        assert len(features) == result['teacher_feature_forward_completions'] == 300 and not any(references.values())
        assert state_digest(teacher.state_dict()) == teacher_hash
        result.update(cache_teacher_setup_epoch_seconds=time.time()-ENTRY_EPOCH,
            cache_teacher_setup_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC,
            resident_cache_tensor_bytes_after_grid_release=cache.receipt['actual_tensor_bytes']-result['released_original_grid_bytes']+result['teacher_latent_tensor_bytes'])
        torch.cuda.empty_cache(); guard(); del deadlines['cache_teacher']
        scaler = FieldScaler(**science['scaler']); scale = PhysicalScale.from_mapping(science['scaler'])

        def rng():
            return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
        def restore(value):
            random.setstate(value['python']); np.random.set_state(value['numpy'])
            torch.set_rng_state(value['torch']); torch.cuda.set_rng_state_all(value['cuda'])
        def backward(model,key):
            return full_residual_case_backward(model,cache.case(key),cache.physical_targets[key],scale,
                cache.case_norms[key],*features[key],keys,guard)

        for method in science['methods']:
            began = (time.time(),time.monotonic()); result.update(phase='arm_setup',current_method=method)
            deadlines['arm'] = (began[0]+1500,began[1]+1500); deadlines['arm_setup'] = (began[0]+120,began[1]+120)
            arm_folder = output/method; arm_folder.mkdir()
            arm = dict(status='running',active=True,closed=False,method=method,step=0,endpoints={},
                entry_epoch=began[0],entry_monotonic=began[1],case_exposure_counts={key:0 for key in keys},
                optimizer_step_attempts=0,optimizer_step_completions=0,optimizer_step_in_progress=False,
                possible_partial_optimizer_update=False,last_checkpoint=None,cached_physical_TRAIN_readout_reuses=0,
                inherited_teacher_passes=5,frozen_reference_passes=0,teacher_old_cost_shared_once=True)
            result['methods'][method] = arm; persist(); guard()
            random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
            model = FrozenOperatorResidual(teacher,science['decoder_modes'][method],2026,'cuda').train()
            arm['model_profile'] = model.profile(); arm['actual_parameter_dtypes'] = sorted({str(p.dtype) for p in model.decoder.parameters()})
            assert model.backend == ('torch_FP32_MLP' if model.mode == 'mlp' else 'tcnn_fully_fused')
            arm['decoder_initial_state_digest'] = state_digest(model.decoder.state_dict())
            reference_hash = state_digest(model.reference.state_dict()); assert reference_hash == arm['decoder_initial_state_digest']
            arm['frozen_teacher_digest'] = teacher_hash; arm['frozen_reference_digest'] = reference_hash
            recipe = science['optimizer']
            optimizer = torch.optim.Adam(model.decoder.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'],
                betas=tuple(recipe['betas']),eps=recipe['eps'],fused=recipe['fused'])
            assert not optimizer.state and all(p.requires_grad for p in model.decoder.parameters())
            witness = max(keys,key=lambda key:manifest['records'][key]['num_nodes']); case = cache.case(witness)
            saved = rng(); context,latent = features[witness]; max_difference = 0.; consumed = 0
            with torch.enable_grad():
                for start in range(0,len(case['xyz']),32768):
                    guard(); stop = min(start+32768,len(case['xyz']))
                    prediction = model.query(case,start,stop,context,latent)
                    baseline = teacher.query(case,start,stop,context,latent)
                    assert prediction.requires_grad and torch.equal(prediction.detach(),baseline) and torch.isfinite(prediction).all()
                    max_difference = max(max_difference,float((prediction-baseline).abs().max())); consumed += stop-start
                    del prediction,baseline
            assert consumed == manifest['records'][witness]['num_nodes']
            optimizer.zero_grad(set_to_none=True); qualification_loss = backward(model,witness); torch.cuda.synchronize()
            gradient_rows = {}; energy = 0.
            for name,parameter in model.decoder.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(),name
                value = float(parameter.grad.detach().double().square().sum()); energy += value
                gradient_rows[name] = dict(present=True,finite=True,L2=value**.5)
            assert energy > 0 and not teacher.training and not model.reference.training
            assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
            assert all(p.grad is None and not p.requires_grad for p in model.reference.parameters())
            assert state_digest(teacher.state_dict()) == teacher_hash and state_digest(model.reference.state_dict()) == reference_hash
            arm['actual_largest_fullN_gradient_check'] = dict(passed=True,key=witness,nodes=consumed,
                initial_prediction_exact_teacher=True,checked_with_training_autograd_enabled=True,initial_encoded_max_abs_difference=max_difference,
                mutable_decoder_gradient_L2=energy**.5,mutable_parameters=gradient_rows,frozen_gradients_none=True,
                frozen_states_unchanged=True,full300_physical_and_frozen_feature_cache_qualified=True,
                physical_loss=qualification_loss,optimizer_updates=0,native_flat_parameter_slicing=False)
            optimizer.zero_grad(set_to_none=True); restore(saved); guard()
            arm.update(setup_epoch_seconds=time.time()-began[0],setup_monotonic_seconds=time.monotonic()-began[1],
                training_entry_epoch=time.time(),training_entry_monotonic=time.monotonic())
            del deadlines['arm_setup']; deadlines['training'] = (arm['training_entry_epoch']+900,arm['training_entry_monotonic']+900)
            result['phase'] = 'training'; persist()

            def save(step,name):
                guard(); assert optimizer.state and all(int(x['step']) == step for x in optimizer.state.values())
                assert state_digest(teacher.state_dict()) == teacher_hash and state_digest(model.reference.state_dict()) == reference_hash
                arm['saved_Adam_state_count'] = len(optimizer.state); arm['saved_Adam_steps_equal_cursor'] = True
                temporary = arm_folder/(name+'.tmp')
                torch.save(dict(step=step,cursor=step,method=method,scientific_digest=plan['scientific_digest'],
                    model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng(),case_exposure_counts=dict(arm['case_exposure_counts']),
                    full_order_digest=science['full_order_digest'],training_objective=science['loss'],teacher_checkpoint=identity,
                    teacher_state_digest=teacher_hash,reference_state_digest=reference_hash,inherited_teacher_passes=5,
                    frozen_reference_passes=0,decoder_initial_state_digest=arm['decoder_initial_state_digest'],
                    dataset_identity=science['dataset_identity'],scaler=science['scaler'],
                    native_origin_clocks=dict(epoch=ENTRY_EPOCH,monotonic=ENTRY_MONOTONIC,boot_id=boot),outer_guard_clock=outer,
                    cost_status=dict(arm),shared_cache_teacher_cost=dict(epoch=result['cache_teacher_setup_epoch_seconds'],monotonic=result['cache_teacher_setup_monotonic_seconds']),teacher_old_cost=science['teacher_old_cost']),temporary)
                os.replace(temporary,arm_folder/name)
                arm['last_checkpoint'] = dict(path=d.relative(arm_folder/name),step=step,stat_identity=d.identity(arm_folder/name)); guard()

            with (arm_folder/'loss.jsonl').open('x',encoding='utf-8') as log:
                for step,key in enumerate(order,1):
                    guard(); optimizer.zero_grad(set_to_none=True); loss = backward(model,key)
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.decoder.parameters())
                    arm['optimizer_step_attempts'] += 1; arm['optimizer_step_in_progress'] = True
                    optimizer.step(); torch.cuda.synchronize()
                    arm.update(step=step,optimizer_step_completions=arm['optimizer_step_completions']+1,optimizer_step_in_progress=False)
                    arm['case_exposure_counts'][key] += 1
                    log.write(json.dumps(dict(step=step,key=key,physical_relative_squared_loss=loss,training_epoch_seconds=time.time()-arm['training_entry_epoch']))+'\n'); log.flush()
                    guard()
                    if step % 300 == 0: save(step,'last.pt')
                    if step % 30 == 0: persist()
                assert set(arm['case_exposure_counts'].values()) == {5}; save(1500,'step1500.pt')
            arm.update(training_epoch_seconds=time.time()-arm['training_entry_epoch'],training_monotonic_seconds=time.monotonic()-arm['training_entry_monotonic'])
            guard(); del deadlines['training']; result['phase'] = 'TRAIN_endpoint_readout'
            started = (time.time(),time.monotonic()); deadlines['readout'] = (started[0]+300,started[1]+300)
            saved = rng(); model.eval(); rows = {key:None for key in keys}
            with torch.no_grad():
                for key in keys:
                    guard(); case = cache.case(key); context,latent = features[key]
                    prediction = np.empty((len(case['xyz']),4),dtype=np.float32)
                    for start in range(0,len(prediction),32768):
                        guard(); stop = min(start+32768,len(prediction))
                        prediction[start:stop] = model.query(case,start,stop,context,latent).cpu().numpy()
                    physical = scaler.decode(prediction); assert np.isfinite(physical).all()
                    truth = cache.physical_targets[key].detach().cpu().numpy()
                    assert truth.shape == prediction.shape and np.isfinite(truth).all()
                    rows[key] = physical_metrics(np,truth,physical)
                    arm['cached_physical_TRAIN_readout_reuses'] += 1; result['cached_physical_TRAIN_readout_reuses'] += 1
                    del prediction,physical,truth
            metric = aggregate(rows,300); assert metric['all_cases_complete'] and metric['fixed_denominator'] == 300
            guard(); arm['endpoints']['1500'] = dict(full_TRAIN=metric,decoder_passes=5,teacher_passes=5,reference_passes=0,
                epoch_seconds=time.time()-started[0],monotonic_seconds=time.monotonic()-started[1],checkpoint_identity=d.identity(arm_folder/'step1500.pt',hashed=True))
            restore(saved); del deadlines['readout']
            assert state_digest(teacher.state_dict()) == teacher_hash and state_digest(model.reference.state_dict()) == reference_hash
            arm['actual_frozen_states_unchanged_after_training'] = True
            optimizer.zero_grad(set_to_none=True); del optimizer,model; cleanup(); torch.cuda.empty_cache(); guard()
            arm.update(status='completed',active=False,closed=True,training_complete=True,
                elapsed_epoch_seconds=time.time()-began[0],elapsed_monotonic_seconds=time.monotonic()-began[1])
            d.write(arm_folder/'summary.json',arm); d.write(arm_folder/'status.json',arm)
            del deadlines['arm']; result['phase'] = 'between_arms'; persist(); arm = None
        assert all(v['status'] == 'completed' for v in result['methods'].values())
        result.update(status='completed',phase='terminal',training_complete=True,requires_separate_DEV_prediction_barrier_readout=True)
    except Exception as error:
        if cache is not None: result['cache_receipt'] = dict(cache.receipt)
        if arm is not None:
            arm.update(status='failed',active=False,closed=True,error_type=type(error).__name__,error=str(error),
                possible_partial_optimizer_update=arm['optimizer_step_in_progress'],elapsed_epoch_seconds=time.time()-arm['entry_epoch'],elapsed_monotonic_seconds=time.monotonic()-arm['entry_monotonic'])
            if 'training' in deadlines:
                arm.update(training_epoch_seconds=time.time()-arm['training_entry_epoch'],training_monotonic_seconds=time.monotonic()-arm['training_entry_monotonic'])
            d.write(output/arm['method']/'summary.json',arm); d.write(output/arm['method']/'status.json',arm)
        result.update(status='failed',phase='terminal',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        try:
            if cleanup is not None: cleanup()
        finally:
            result.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
            d.write(output/'summary.json',result); d.write(output/'status.json',result)
            if lock is not None: lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--start',action='store_true'); parser.add_argument('--plan')
    args = parser.parse_args()
    if args.start:
        if not args.plan: parser.error('Frozen released --plan required')
        run(args.plan)
