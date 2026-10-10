"""One fresh single-TRAIN-case fit, following the completed matched pair."""
import argparse
import json
import math
import os
import random
import time
from geobc import available_cad_pilot_data as d

PLAN = 'notes/geobc_single_case_fit_plan.json'


def terminal_binding(binding):
    for field in ('summary', 'external'):
        item = binding[field]
        assert d.identity(item['path'], hashed=True) == item
    summary = d.read(binding['summary']['path'])
    external = d.read(binding['external']['path'])
    assert summary['status'] == 'completed' and summary['closed'] and not summary['active']
    assert external['closed'] and not external['active'] and external['within_external_budget']
    assert external['exit_code'] == 0 and not external['external_timeout']
    assert all(d.exact_process_exited(external[k]) for k in ('guard_process', 'child_process'))
    return summary


def select_case(parent, development):
    """Selection rule is frozen before completing the current pair."""
    assert parent['both4800'] and development['protocol_qualified']
    scores = {arm: development['candidates'][arm]['full60']['selection_score']
              for arm in ('FNO9', 'PP256')}
    assert all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in scores.values())
    tied = sorted(arm for arm, v in scores.items() if v == min(scores.values()))
    arm = tied[0]
    training = parent['arms'][arm]['TRAIN120']
    assert training['all_cases_complete'] and training['fixed_denominator'] == 120
    assert training['case_mean_stress_relative_l2'] > .15, 'Fitting branch not indicated'
    per = training['per_case']
    assert len(per) == 120
    # Highest signed-S relative L2 within existing TRAIN only; key breaks exact ties.
    key = sorted(per, key=lambda k: (-per[k]['stress_relative_l2'], k))[0]
    return {'arm': arm, 'case_key': key, 'development_ties': tied,
            'shared_training_case': per[key], 'development_scores': scores,
            'selection_is_diagnostic_not_final_solver_choice': True}


def run():
    plan = d.read(PLAN); science = plan['scientific']; review = d.read(plan['root_review'])
    assert review['passed'] and review['actual_execution_released']
    assert d.digest(science) == plan['scientific_digest'] == review['scientific_digest']
    assert review['source_pins'] == plan['source_pins']
    assert all(d.sha(p) == h for p, h in plan['source_pins'].items())
    parent = terminal_binding(review['upstream_bindings']['training'])
    development = terminal_binding(review['upstream_bindings']['development'])
    assert parent['execution']['scientific_digest'] == science['parent_training_digest']
    assert development['execution']['scientific_digest'] == science['parent_development_digest']
    selected = select_case(parent, development)
    assert selected == review['selected']
    outer = json.loads(os.environ['GEOBC_SINGLE_CASE_GUARD'])
    assert outer['scientific_digest'] == plan['scientific_digest']
    output = d.path(plan['output']); output.mkdir(parents=True, exist_ok=True)
    assert not (output/'execution.json').exists(), 'One attempt; no automatic recovery'
    execution = {'run_process': d.process_snapshot(os.getpid()), 'guard': outer,
                 'scientific_digest': plan['scientific_digest']}
    d.write(output/'execution.json', execution)
    epoch = time.time(); mono = time.monotonic(); training_start = None; lock = None
    result = {'status': 'running', 'active': True, 'closed': False, 'execution': execution,
              'selected': selected, 'step': 0, 'endpoints': {}, 'raw_GT_reads': 0,
              'DEV_GT_reads': 0, 'scaler_fits': 0, 'pretrained_checkpoint_reads': 0,
              'cached_TRAIN_encoded_reads': 0, 'cached_TRAIN_physical_reads': 0}

    def guard():
        left = min(outer['deadline_epoch']-time.time(), outer['deadline_monotonic']-time.monotonic())
        if training_start:
            left = min(left, training_start[0]+18000-time.time(), training_start[1]+18000-time.monotonic())
        if left <= 0: raise TimeoutError('Original single-case diagnostic deadline')

    def persist():
        result.update(elapsed_epoch_seconds=time.time()-epoch, elapsed_monotonic_seconds=time.monotonic()-mono)
        d.write(output/'status.json', result)

    try:
        import fcntl
        lock = d.path(science['GPU_lock']).open('a+')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        import numpy as np
        import torch
        from geobc import coverage_comparison_data as data
        from geobc.coverage_comparison_train import physical_metrics
        from geobc.common_grid_solver import CommonGridSolver, full_case_backward, parameter_gradients
        from geobc.benchmark import FieldScaler
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        device = torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(24*2**30/torch.cuda.get_device_properties(device).total_memory, device)
        manifest = data.read_bootstrap_TRAIN40(science['original_TRAIN_plan'])
        key = selected['case_key']; arm = selected['arm']
        assert key in science['TRAIN_keys'] and science['TRAIN_keys'] == manifest['train_keys']['40']
        assert science['scaler'] == manifest['scaler']
        scaler = FieldScaler(**science['scaler'])
        case = data.load_case(manifest, key, device='cuda', for_fno=True)
        result['input_query_rows_with_nonzero_weights'] = int((case['query_weights'].abs().sum(dim=1)>0).sum())
        item = manifest['records'][key]['file_identities']['target_path']; data.stat_matches(item)
        case['target'] = torch.as_tensor(np.array(np.load(data.path(item['path']), allow_pickle=False), copy=True), device='cuda')
        result['cached_TRAIN_encoded_reads'] = 1
        item = manifest['records'][key]['file_identities']['physical_path']; data.stat_matches(item)
        truth = np.load(data.path(item['path']), mmap_mode='r', allow_pickle=False)
        result['cached_TRAIN_physical_reads'] = 1
        random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
        model = CommonGridSolver(arm, 2026).to('cuda').train()
        assert model.shared_identity() == parent['arms'][arm]['shared_init_sha256']
        result.update(counts=model.counts(), shared_init_sha256=model.shared_identity(), original_nodes=len(truth))
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0., fused=True)

        def rng():
            return {'python': random.getstate(), 'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}

        def readout(step):
            guard(); saved = rng(); started = time.time(); started_mono = time.monotonic()
            assert result['step'] == step
            train_epoch = started-training_start[0] if training_start else result.get('training_epoch_seconds')
            train_mono = started_mono-training_start[1] if training_start else result.get('training_monotonic_seconds')
            model.eval(); index = science['TRAIN_keys'].index(key)
            torch.manual_seed(2026+index); torch.cuda.manual_seed_all(2026+index)
            with torch.no_grad():
                context, latent = model.encode_case(case)
                prediction = np.empty((len(truth), 4), dtype=np.float32)
                for start in range(0, len(truth), 32768):
                    guard(); stop = min(start+32768, len(truth))
                    prediction[start:stop] = model.query(case, start, stop, context, latent).cpu().numpy()
            result['endpoints'][str(step)] = {'physical': physical_metrics(np, truth, scaler.decode(prediction)),
                'epoch_seconds': time.time()-started, 'monotonic_seconds': time.monotonic()-started_mono,
                'training_epoch_seconds_before_readout': train_epoch,
                'training_monotonic_seconds_before_readout': train_mono,
                'case_exposures': step, 'optimizer_steps': step}
            random.setstate(saved['python']); np.random.set_state(saved['numpy'])
            torch.set_rng_state(saved['torch']); torch.cuda.set_rng_state_all(saved['cuda'])
            model.train(); persist()

        def save(step, name):
            guard(); temporary = output/(name+'.tmp')
            torch.save({'step': step, 'arm': arm, 'case_key': key, 'scientific_digest': plan['scientific_digest'],
                        'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': rng(),
                        'case_exposure_counts': {key: step}, 'cost_status': result}, temporary)
            os.replace(temporary, output/name)

        readout(0); training_start = (time.time(), time.monotonic()); torch.cuda.reset_peak_memory_stats()
        with (output/'loss.jsonl').open('x', encoding='utf-8') as log:
            for step in range(1, 4801):
                guard(); optimizer.zero_grad(set_to_none=True)
                loss, leaves = full_case_backward(model, case, guard)
                if step == 1:
                    gradients = parameter_gradients(model)
                    unused = ['backbone.embedding.weight', 'backbone.embedding.bias'] if arm == 'PP256' else []
                    passed = (gradients['operator']['missing'] == unused and
                              not gradients['encoder']['missing'] and not gradients['head']['missing'] and
                              all(g['finite'] and g['L2'] > 0 for g in gradients.values()) and all(v > 0 for v in leaves.values()))
                    result['first_actual_fullN_gradient_check'] = {'passed': passed, 'gradients': gradients,
                        'leaf_gradients': leaves, 'expected_inactive_parameters': unused}
                    assert passed, 'FullN gradient check failed; exact gradient/leaf/query-coverage diagnostics retained in summary'
                for p in model.parameters():
                    if p.grad is not None and not bool(torch.isfinite(p.grad).all()): raise FloatingPointError('Nonfinite gradient')
                optimizer.step(); torch.cuda.synchronize(); result['step'] = step
                result['training_epoch_seconds'] = time.time()-training_start[0]
                result['training_monotonic_seconds'] = time.monotonic()-training_start[1]
                result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                assert max(result['peak_allocated_bytes'], result['peak_reserved_bytes']) <= 24*2**30
                log.write(json.dumps({'step': step, 'encoded_MSE': loss, 'training_epoch_seconds': result['training_epoch_seconds']})+'\n'); log.flush()
                if step % 120 == 0: save(step, 'last.pt')
                if step in (40, 4800):
                    save(step, 'step'+str(step)+'.pt')
                    if step == 4800:
                        guard()
                        result['training_epoch_seconds'] = time.time()-training_start[0]
                        result['training_monotonic_seconds'] = time.monotonic()-training_start[1]
                        training_start = None
                    readout(step)
                if step % 30 == 0: persist()
        result.update(status='completed', fixed4800=True, comparison_scope='One selected known TRAIN case; no generalization claim',
            exposure_limit='40 same-case exposures vs shared 40;4800 same-total updates vs shared4800. These controls differ in optimizer history and case exposure.',
            training_clock_includes_step40_readout_and_checkpoint_IO=True)
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc), automatic_replay=False)
        raise
    finally:
        result.update(active=False, closed=True, elapsed_epoch_seconds=time.time()-epoch, elapsed_monotonic_seconds=time.monotonic()-mono)
        d.write(output/'summary.json', result); d.write(output/'status.json', result)
        if lock is not None: lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--start', action='store_true')
    if parser.parse_args().start: run()
