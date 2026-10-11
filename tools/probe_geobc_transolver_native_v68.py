"""One bounded real fullN qualification, no field truth or optimizer updates."""
import time
ENTRY_EPOCH, ENTRY_MONOTONIC = time.time(), time.monotonic()
import argparse
import json
import os
import signal
from pathlib import Path


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.transolver_native_probe_contract_v68 import approved_contract
    p, r, m, _ = approved_contract(plan_path)
    s = p['scientific']
    outer = json.loads(os.environ['GEOBC_TRANSOLVER_NATIVE_PROBE_V68_GUARD'])
    assert outer['scientific_digest'] == p['scientific_digest']
    assert outer['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid']) == outer['guard_process']
    output = d.path(p['output'])
    execution = dict(run_process=d.process_snapshot(os.getpid()), guard=outer,
                     scientific_digest=p['scientific_digest'])
    assert not (output/'execution.json').exists()
    d.write(output/'execution.json', execution)
    a = dict(status='running', active=True, closed=False, execution=execution,
        cases={}, GT_array_reads=0, PT_body_reads=0, optimizer_updates=0,
        input_array_attempts=0, input_array_reads=0, grid_input_reads=0,
        peak_allocated_bytes=0, peak_reserved_bytes=0,
        peak_total_device_used_bytes_sampled=0, offnode_runtime_qualified=False,
        automatic_training_release=False, physical_prediction_accuracy_measured=False)
    torch, lock = None, None

    def guard():
        if min(outer['deadline_epoch']-time.time(),
               outer['deadline_monotonic']-time.monotonic()) <= 0:
            raise TimeoutError('Frozen earlier UTC/monotonic deadline')
        if torch is not None and torch.cuda.is_initialized():
            a['peak_allocated_bytes'] = max(a['peak_allocated_bytes'], torch.cuda.max_memory_allocated())
            a['peak_reserved_bytes'] = max(a['peak_reserved_bytes'], torch.cuda.max_memory_reserved())
            free, total = torch.cuda.mem_get_info()
            a['peak_total_device_used_bytes_sampled'] = max(a['peak_total_device_used_bytes_sampled'], total-free)
            if max(a[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled')) > s['memory_limit_bytes']:
                raise MemoryError('Frozen 30.5GB SI budget exceeded')

    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,
                 elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(output/'status.json', a)

    signal.signal(signal.SIGTERM, lambda *args: (_ for _ in ()).throw(TimeoutError('External guard terminated child')))
    try:
        import fcntl
        lock = d.path(s['GPU_lock']).open('a+')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard()
        import numpy as np
        import torch as torch_module
        torch = torch_module
        from geobc.prospective_dataset_v2 import load_case
        from geobc.transolver_native_hash_v68 import NativeTransolverHash, full_native_physical_backward
        from geobc.physical_field_loss import PhysicalScale, norms_from_complete_physical_target
        from geobc.fno_stress_specialist_v54 import tree_digest
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        assert total > s['memory_limit_bytes']
        torch.cuda.set_per_process_memory_fraction(s['memory_limit_bytes']/total, device)
        torch.cuda.reset_peak_memory_stats()
        model = NativeTransolverHash(seed=s['seed'], device='cuda')
        initial = tree_digest(model.state_dict())
        assert initial == s['fresh_initial_model_digest']
        a['fresh_initial_model_digest'] = initial
        a['source_config'] = model.source_config
        a['parameter_count'] = sum(x.numel() for x in model.parameters())
        scale = PhysicalScale.from_mapping(s['scaler'])
        persist()

        def event(stage, key, name, identity):
            guard()
            assert key in s['selected_TRAIN_keys'] and name in ('xyz','features','cloud','mask')
            a['input_array_attempts' if stage == 'attempt' else 'input_array_reads'] += 1

        for key in s['selected_TRAIN_keys']:
            began = time.monotonic()
            case = load_case(m, key, with_targets=False, train_CAD_count=100,
                             device='cuda', for_fno=False, on_read=event)
            count = len(case['xyz'])
            assert count == m['records'][key]['num_nodes'] and case['scientific_group'] == 'TRAIN'
            model.eval()
            with torch.no_grad():
                context, latent = model.encode_case(case)
                tokens = model.native_tokens(case, context)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    official = model.operator.backbone(tokens[..., :3], tokens[..., 3:])[0].float()
                delta = float((official-latent).abs().max())
                assert delta <= s['official_forward_atol']
                del official, tokens
                chunks = [model.query(case, start, min(start+32768, count), context, latent)
                          for start in range(0, count, 32768)]
                full = torch.cat(chunks)
                assert full.shape == (count, 4) and torch.isfinite(full).all()
                del chunks
                perm = torch.arange(count-1, -1, -1, device='cuda')
                queries = dict(xyz=case['xyz'][perm], features=case['features'][perm])
                reversed_latent = latent[perm]
                reordered = torch.cat([model.query(queries, start, min(start+16384, count),
                    context, reversed_latent) for start in range(0, count, 16384)])[perm]
                query_delta = float((full-reordered).abs().max())
                assert query_delta <= s['decoder_query_atol']
                del full, reordered, queries, reversed_latent, perm, context, latent
            guard()
            model.train()
            model.zero_grad(set_to_none=True)
            # Deterministic artificial fields exercise the physical loss graph.
            # They are neither DeepJEB labels nor an accuracy experiment.
            xyz = case['xyz']
            artificial = torch.cat((.01*xyz, 100*torch.sin(xyz.sum(-1, keepdim=True))), -1)
            norms = norms_from_complete_physical_target(artificial)
            loss, leaves = full_native_physical_backward(model, case, artificial, scale, norms, guard)
            groups = {}
            for name in ('encoder','operator','head'):
                params = list(getattr(model, name).parameters())
                assert params and all(x.grad is not None and torch.isfinite(x.grad).all() for x in params)
                groups[name] = sum(float(x.grad.double().square().sum()) for x in params)
                assert groups[name] > 0
            assert all(x > 0 for x in leaves.values())
            torch.cuda.synchronize()
            assert tree_digest(model.state_dict()) == initial
            a['cases'][key] = dict(nodes=count, official_fullN_forward_max_abs_delta=delta,
                fullN_query_reverse_and_chunk_max_abs_delta=query_delta,
                artificial_fullN_physical_joint_loss=loss, groups_squared_L2=groups,
                leaves=leaves, unchanged_parameters=True, optimizer_updates=0,
                elapsed_monotonic_seconds=time.monotonic()-began, passed=True)
            model.zero_grad(set_to_none=True)
            del case, xyz, artificial
            torch.cuda.empty_cache()
            guard()
            persist()
        assert len(a['cases']) == 3 and a['input_array_reads'] == a['input_array_attempts'] == 12
        a.update(status='completed', qualification_passed=True,
                 qualification_scope='Real largest fullN per load, official forward/query invariance/full physical-chain artificial-gradient/resource; no truth or offnode experiment')
    except Exception as error:
        a.update(status='failed', error_type=type(error).__name__, error=str(error), qualification_passed=False)
        raise
    finally:
        a.update(active=False, closed=True, elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,
                 elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(output/'summary.json', a)
        d.write(output/'status.json', a)
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--plan')
    args = parser.parse_args()
    if args.start:
        if not args.plan:
            parser.error('--start requires a frozen --plan')
        run(args.plan)
    else:
        print('{"status":"source_only","GPU_calls":0,"GT_reads":0}')
