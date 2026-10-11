"""One released GPU resource/gradient witness, no optimizer or DEV readout."""
import time
ENTRY_EPOCH, ENTRY_MONOTONIC = time.time(), time.monotonic()
import argparse
import json
import os
from pathlib import Path
import signal


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.coarse_transolver_probe_contract_v19 import approved_contract
    plan, _, manifest, keys = approved_contract(plan_path); science = plan['scientific']
    outer = json.loads(os.environ['GEOBC_COARSE_TRANSOLVER_V19_GUARD'])
    assert outer['scientific_digest'] == plan['scientific_digest']
    assert d.process_snapshot(outer['guard_process']['pid']) == outer['guard_process']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == outer['boot_id']
    output = d.path(plan['output']); assert not (output/'execution.json').exists()
    execution = dict(run_process=d.process_snapshot(os.getpid()), guard=outer, boot_id=outer['boot_id'], scientific_digest=plan['scientific_digest'])
    d.write(output/'execution.json', execution)
    result = dict(status='running', active=True, closed=False, execution=execution, phase='setup',
        input_array_read_attempts=0, input_array_reads=0, physical_TRAIN_read_attempts=0, physical_TRAIN_reads=0,
        gradient_attempts=0, gradient_completions=0, optimizer_updates=0, checkpoint_reads=0,
        DEV_FINAL_numeric_reads=0, raw_CSV_H5_reads=0, scaler_fits=0, methods={},
        peak_allocated_bytes=0, peak_reserved_bytes=0, peak_total_device_used_bytes_sampled=0,
        automatic_replay=False, actual_full100_cache_qualified=False, real_training_released=False)
    lock = None; torch = None; cleanup = None
    signal.signal(signal.SIGTERM, lambda *args: (_ for _ in ()).throw(TimeoutError('External guard terminated probe')))

    def guard():
        remaining = min(outer['deadline_epoch']-time.time(), outer['deadline_monotonic']-time.monotonic())
        if result['phase'] == 'setup':
            remaining = min(remaining, ENTRY_EPOCH+120-time.time(), ENTRY_MONOTONIC+120-time.monotonic())
        if remaining <= 0:
            raise TimeoutError('Fixed probe dual clocks')
        if torch is not None and torch.cuda.is_initialized():
            result['peak_allocated_bytes'] = max(result['peak_allocated_bytes'], torch.cuda.max_memory_allocated())
            result['peak_reserved_bytes'] = max(result['peak_reserved_bytes'], torch.cuda.max_memory_reserved())
            free, total = torch.cuda.mem_get_info()
            result['peak_total_device_used_bytes_sampled'] = max(result['peak_total_device_used_bytes_sampled'], total-free)
            if max(result[k] for k in ('peak_allocated_bytes', 'peak_reserved_bytes', 'peak_total_device_used_bytes_sampled')) > science['memory_limit_bytes']:
                raise MemoryError('Probe exceeds30.5GB')

    def persist():
        result.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH, elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(output/'status.json', result)

    try:
        import fcntl
        lock = d.path(science['GPU_lock']).open('a+'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        import numpy as np
        import torch as torch_module
        torch = torch_module
        from geobc.coarse_transolver_v19 import CoarseTransolver, full_physical_backward
        from geobc.physical_field_loss import PhysicalScale, norms_from_complete_physical_target
        from geobc.prospective_dataset_v2 import load_case
        from geobc.common_grid_solver import CommonGridSolver
        from geobc.transformers import close_owned_process_group
        cleanup = close_owned_process_group
        torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        assert science['memory_limit_bytes'] < total
        torch.cuda.set_per_process_memory_fraction(science['memory_limit_bytes']/total)
        torch.cuda.reset_peak_memory_stats(); guard(); cache = {}

        def input_event(stage, key, name, identity):
            assert name not in ('target', 'physical'); guard()
            if stage == 'attempt': result['input_array_read_attempts'] += 1
            if stage == 'opened': result['input_array_reads'] += 1

        for key in keys:
            case = load_case(manifest, key, with_targets=False, train_CAD_count=100, device='cuda', for_fno=True, on_read=input_event)
            assert case['scientific_group'] == 'TRAIN' and not {'target', 'physical'} & set(case)
            item = manifest['records'][key]['file_identities']['physical_path']; d.stat_matches(item)
            result['physical_TRAIN_read_attempts'] += 1
            value = np.load(d.path(item['path']), allow_pickle=False, mmap_mode='r'); result['physical_TRAIN_reads'] += 1
            assert value.shape == (len(case['xyz']), 4) and str(value.dtype) == 'float32' and np.isfinite(value).all()
            truth = torch.as_tensor(np.array(value, copy=True), device='cuda'); norms = norms_from_complete_physical_target(truth)
            cache[key] = (case, truth, norms); del value; guard(); persist()
        result['phase'] = 'six_wholeN_gradient_witnesses'; scale = PhysicalScale.from_mapping(science['scaler'])
        for method in science['methods']:
            guard(); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
            model = CoarseTransolver(method).train(); row = dict(profile=model.profile(), cases={},
                actual_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                kernel_precision_profiled=False)
            result['methods'][method] = row
            cpu_rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state_all()
            reference = CommonGridSolver('FNO9', 2026)
            torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
            for group in ('encoder', 'head'):
                actual = getattr(model, group).state_dict(); expected = getattr(reference, group).state_dict()
                assert set(actual) == set(expected) and all(torch.equal(v.detach().cpu(), expected[k]) for k, v in actual.items())
            row['actual_shared_initial_encoder_head_equal_seeded_FNO'] = True; del reference
            for key in keys:
                case, truth, norms = cache[key]; model.zero_grad(set_to_none=True); guard()
                result['gradient_attempts'] += 1; persist()
                torch.cuda.synchronize(); began = (time.time(), time.monotonic())
                start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True); start.record()
                loss, leaves = full_physical_backward(model, case, truth, scale, norms, keys, guard)
                end.record(); torch.cuda.synchronize()
                wall = (time.time()-began[0], time.monotonic()-began[1]); groups = dict(encoder=0., head=0., operator=0.); missing = []
                for name, parameter in model.named_parameters():
                    if parameter.grad is None: missing.append(name); continue
                    assert torch.isfinite(parameter.grad).all(), 'Nonfinite real parameter gradient'
                    group = name.split('.')[0]; groups[group] += float(parameter.grad.double().square().sum())
                allowed = ['operator.backbone.embedding.weight', 'operator.backbone.embedding.bias'] if method == 'transolver_pp' else []
                assert sorted(missing) == sorted(allowed) and all(v > 0 for v in groups.values())
                assert leaves['geometry_context_L2'] > 0 and leaves['operator_latent_L2'] > 0
                row['cases'][key] = dict(passed=True, original_nodes=len(case['xyz']), global_tokens=4096,
                    groups_L2={k: v**.5 for k, v in groups.items()}, allowed_unused=missing,
                    leaf_gradients=leaves, qualification_objective=loss, epoch_seconds=wall[0], monotonic_seconds=wall[1],
                    CUDA_event_seconds=start.elapsed_time(end)/1000, timing_scope='One real full forward/backward; no hidden warmup or optimizer')
                result['gradient_completions'] += 1; guard(); persist()
            del model; close_owned_process_group(); torch.cuda.empty_cache(); guard()
        assert result['gradient_attempts'] == result['gradient_completions'] == 6
        result.update(status='completed', phase='terminal', resource_qualified=True,
                      limitation='Two TRAIN witnesses, not full100 cached training or DEV accuracy; mandatory real all300 qualification still required before updates')
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error), resource_qualified=False)
        raise
    finally:
        if cleanup is not None: cleanup()
        result.update(active=False, closed=True, elapsed_epoch_seconds=time.time()-ENTRY_EPOCH, elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(output/'summary.json', result); d.write(output/'status.json', result)
        if lock is not None: lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--start', action='store_true'); parser.add_argument('--plan')
    args = parser.parse_args()
    if args.start:
        if not args.plan: parser.error('Frozen --plan required')
        run(args.plan)
