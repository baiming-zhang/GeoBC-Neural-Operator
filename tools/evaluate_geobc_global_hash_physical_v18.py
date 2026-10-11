"""Future selected FNO/native++/conditioned-Hash endpoints on known DEV60.

Default is inert. A separately chosen scientific plan must bind genuinely
completed macro training and exact exits before any numerical library or I/O.
"""
import time
ENTRY_EPOCH, ENTRY_MONOTONIC = time.time(), time.monotonic()
import argparse
import json
import os
from pathlib import Path
import signal


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.global_hash_physical_readout_contract_v18 import approved_contract, check_identity, complete_physical
    plan, review, manifest, truths, qualified = approved_contract(plan_path)
    science = plan['scientific']; outer = json.loads(os.environ['GEOBC_GLOBAL_HASH_PHYSICAL_READOUT_V18_GUARD'])
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert outer['scientific_digest'] == plan['scientific_digest'] and outer['boot_id'] == boot
    assert d.process_snapshot(outer['guard_process']['pid']) == outer['guard_process']
    output = d.path(plan['output']); assert output.exists() and not (output/'execution.json').exists()
    execution = dict(run_process=d.process_snapshot(os.getpid()), guard=outer, boot_id=boot,
                     scientific_digest=plan['scientific_digest'])
    d.write(output/'execution.json', execution)
    keys = science['case_keys']; cap = science['memory_limit_bytes']
    result = dict(status='running', active=True, closed=False, execution=execution,
        phase='prediction_commit', candidates={}, prediction_inventory={},
        fixed_prediction_denominator=science['fixed_prediction_denominator'], fixed_case_denominator=60,
        committed_finite_predictions=0, checkpoint_read_attempts=0, checkpoint_reads=0,
        input_array_read_attempts=0, input_array_reads=0, owned_GT_read_attempts=0, owned_GT_array_reads=0,
        raw_GT_reads=0, FINAL_numeric_reads=0, training_updates=0, scaler_fits=0,
        peak_allocated_bytes=0, peak_reserved_bytes=0, peak_total_device_used_bytes_sampled=0,
        backend_external_workspace_exact_peak_known=False, automatic_replay=False,
        eligible_best=None, protocol_qualified=False, final_solver_selected=False,
        independent_offnode_FEM_accuracy_measured=False, pool=science['pool'])
    lock = None; torch = None; model = None

    def guard():
        if min(outer['deadline_epoch']-time.time(), outer['deadline_monotonic']-time.monotonic()) <= 0:
            raise TimeoutError('Frozen macro-readout dual-clock deadline')
        if torch is not None and torch.cuda.is_initialized():
            result['peak_allocated_bytes'] = max(result['peak_allocated_bytes'], torch.cuda.max_memory_allocated())
            result['peak_reserved_bytes'] = max(result['peak_reserved_bytes'], torch.cuda.max_memory_reserved())
            free, total = torch.cuda.mem_get_info()
            result['peak_total_device_used_bytes_sampled'] = max(result['peak_total_device_used_bytes_sampled'], total-free)
            if max(result[k] for k in ('peak_allocated_bytes', 'peak_reserved_bytes', 'peak_total_device_used_bytes_sampled')) > cap:
                raise MemoryError('Allocated/reserved or sampled total device usage exceeds30.5GB')

    def persist():
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
        from geobc.prospective_dataset_v2 import load_case
        from geobc.operator_hash_metadata_v6 import build_solver, encode_case, query_case, profile
        from geobc.macro_readout_protocol import shared_truth_pass
        from geobc.available_cad_pilot_eval import _save_array
        from geobc.coverage_comparison_train import physical_metrics, aggregate
        from geobc.benchmark import FieldScaler
        from geobc.transformers import close_owned_process_group
        torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        device = torch.cuda.current_device(); total = torch.cuda.get_device_properties(device).total_memory
        assert cap < total
        torch.cuda.set_per_process_memory_fraction(cap/total, device); torch.cuda.reset_peak_memory_stats()
        scaler = FieldScaler(**science['scaler']); inventory = {}; candidates = []
        for candidate in science['candidates']:
            guard(); name = candidate['name']; upstream = qualified[name]; ts = upstream['plan']['scientific']
            endpoint = upstream['checkpoint']; step = candidate['step']
            result['checkpoint_read_attempts'] += 1
            saved = torch.load(check_identity(endpoint), map_location='cpu', weights_only=False)
            result['checkpoint_reads'] += 1; guard()
            assert saved['family'] == ts['family'] and saved['step'] == saved['cursor'] == step
            assert saved['scientific_digest'] == upstream['plan']['scientific_digest']
            assert saved['case_exposure_counts'] == {k: step//len(ts['TRAIN_keys']) for k in ts['TRAIN_keys']}
            assert saved['full_order_digest'] == ts['full_order_digest'] and saved['dataset_identity'] == science['dataset_identity']
            assert saved['scaler'] == science['scaler'] and set(saved['rng']) == {'python', 'numpy', 'torch', 'cuda'}
            assert saved['optimizer']['state'] and all(int(s['step']) == step for s in saved['optimizer']['state'].values())
            assert saved['native_origin_clocks']['boot_id'] == upstream['summary']['execution']['boot_id']
            model = build_solver(ts['family'], seed=ts['seed']); model.load_state_dict(saved['model'], strict=True); del saved
            model.eval(); guard()
            spec = dict(name=name, family=ts['family'], step=step, checkpoint=endpoint); candidates.append(spec)
            row = dict(**spec, train_CAD_count=ts['train_CAD_count'], case_exposure_count=step//len(ts['TRAIN_keys']),
                model_profile=profile(model), cases=0, inference_seconds=0., inference_epoch_seconds=0.,
                training_plan_identity=candidate['training_plan_identity'], upstream_terminal_bindings=review['training_bindings'][name],
                terminal_campaign_training_epoch_seconds=upstream['summary']['training_epoch_seconds'],
                terminal_campaign_training_monotonic_seconds=upstream['summary']['training_monotonic_seconds'],
                cost_scope='Same5pass with zero-update metadata failure; failure and repair gap retained',
                technical_recovery_cost=upstream['summary']['technical_recovery_cost'],
                terminal_stage_epoch_seconds=upstream['external']['elapsed_epoch_seconds'],
                terminal_stage_monotonic_seconds=upstream['external']['elapsed_monotonic_seconds'])
            result['candidates'][name] = row

            def input_event(stage, key, member, identity):
                assert member not in ('target', 'physical')
                guard()
                if stage == 'attempt': result['input_array_read_attempts'] += 1
                if stage == 'opened': result['input_array_reads'] += 1

            with torch.no_grad():
                for index, key in enumerate(keys):
                    guard(); case = load_case(manifest, key, with_targets=False, train_CAD_count=ts['train_CAD_count'],
                        device='cuda', for_fno=True, on_read=input_event)
                    assert case['scientific_group'] == 'DEV' and not {'target', 'physical'} & set(case)
                    n = manifest['records'][key]['num_nodes']; assert n == len(case['xyz'])
                    torch.manual_seed(science['seed']+index); torch.cuda.manual_seed_all(science['seed']+index)
                    torch.cuda.synchronize(); started = (time.time(), time.monotonic())
                    context, latent = encode_case(model, case); prediction = np.empty((n, 4), dtype=np.float32)
                    for start in range(0, n, science['decoder_chunk_size']):
                        guard(); stop = min(start+science['decoder_chunk_size'], n)
                        prediction[start:stop] = query_case(model, case, start, stop, context, latent).cpu().numpy()
                    torch.cuda.synchronize(); row['inference_seconds'] += time.monotonic()-started[1]
                    row['inference_epoch_seconds'] += time.time()-started[0]
                    physical = np.asarray(scaler.decode(prediction), dtype=np.float32)
                    assert prediction.shape == physical.shape == (n, 4) and np.isfinite(prediction).all() and np.isfinite(physical).all()
                    directory = output/'predictions'/name/key
                    record = dict(candidate=name, family=ts['family'], step=step, key=key, checkpoint=endpoint,
                        status='completed', all_original_nodes=n, seed=science['seed']+index,
                        encoded=_save_array(d, np, directory/'encoded4.npy', prediction),
                        physical=_save_array(d, np, directory/'physical4.npy', physical), signed_stress_unchanged=True)
                    file = directory/'manifest.json'; d.write(file, record); inventory[name+':'+key] = d.identity(file, hashed=True)
                    result['committed_finite_predictions'] += 1; row['cases'] += 1
                    guard(); persist(); del case, context, latent, prediction, physical
            check_identity(endpoint); del model; model = None
            close_owned_process_group(); torch.cuda.empty_cache(); guard()
        result.update(phase='all_predictions_revalidation', prediction_inventory=inventory); persist()
        counts = {k: manifest['records'][k]['num_nodes'] for k in keys}

        def record_reader(item):
            guard(); return d.read(check_identity(item))
        def array_reader(item):
            guard(); return np.load(check_identity(item), mmap_mode='r', allow_pickle=False)
        def save_barrier(barrier):
            assert result['owned_GT_array_reads'] == result['owned_GT_read_attempts'] == 0
            assert barrier['finite_predictions'] == science['fixed_prediction_denominator']
            d.write(output/'prediction_barrier.json', barrier)
            result.update(phase='shared_knownDEV_GT_once', prediction_barrier_complete=True); persist()
        def truth_reader(key):
            guard(); assert result['prediction_barrier_complete']
            result['owned_GT_read_attempts'] += 1
            return np.load(check_identity(truths['records'][key]['physical']), mmap_mode='r', allow_pickle=False)
        def on_truth(key):
            result['owned_GT_array_reads'] += 1; persist()
        per = shared_truth_pass(np, candidates, keys, counts, inventory, record_reader, array_reader,
                               truth_reader, save_barrier, on_truth, physical_metrics, guard)
        assert result['owned_GT_array_reads'] == result['owned_GT_read_attempts'] == 60
        for spec in candidates:
            name = spec['name']; row = result['candidates'][name]; row['full60'] = aggregate(per[name], 60)
            row['per_load20'] = {load: aggregate({k: per[name][k] for k in keys if k.endswith('/'+load)}, 20)
                                 for load in ('ver', 'hor', 'tor')}
            complete_physical(row['full60'], keys)
        # Reuse previously committed/scored baseline scalar cases. No baseline
        # inference or raw truth replay. Only60 new predictions belong to this stage.
        base = d.read(check_identity(science['baseline_development_binding']['summary']))
        reference = dict(base['candidates']['FNO100_PHYSICAL_1500'])
        reference.update(reused_immutable_scalar_metrics=True, inference_replayed=False,
            current_stage_inference_seconds=0., scalar_metrics_identity=science['baseline_development_binding']['summary'])
        result['candidates']['FNO100_PHYSICAL_1500'] = reference
        vm = result['candidates']['GLOBALHASH100_PHYSICAL_1500']['full60']; bm = reference['full60']
        du = vm['case_mean_u_relative_l2']-bm['case_mean_u_relative_l2']
        ds = vm['case_mean_stress_relative_l2']-bm['case_mean_stress_relative_l2']
        th = science['DEV_gate']
        dp=.5*(du+ds)
        result['DEV_gate'] = dict(passed=(dp <= -th['primary_absolute_improvement_at_least']
            and max(du,ds) <= th['maximum_either_field_absolute_deterioration']), U_difference=du,S_difference=ds,primary_difference=dp,
            direction='globalHashphysical5 minus FNOphysical5; decoder package capacity/precision differ',thresholds=th,fixed_denominator=60,releases_continuation=False)
        scores = {n: row['full60']['selection_score'] for n, row in result['candidates'].items()}
        from tools.export_geobc_exposure_comparison import paired
        names = sorted(scores)
        result['paired_CAD20'] = {a+':'+b: paired(result['candidates'][a]['full60'], result['candidates'][b]['full60'], keys)
            for i, a in enumerate(names) for b in names[i+1:]}
        guard(); result.update(status='completed', phase='terminal', protocol_qualified=True,
                              eligible_best=sorted(n for n, score in scores.items() if score == min(scores.values())))
    except Exception as error:
        result.update(status='failed', phase='terminal_failure', error_type=type(error).__name__, error=str(error),
                      eligible_best=None, protocol_qualified=False)
        raise
    finally:
        result.update(active=False, closed=True, elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,
                      elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(output/'summary.json', result); d.write(output/'status.json', result)
        if lock is not None: lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--start', action='store_true'); parser.add_argument('--plan')
    args = parser.parse_args()
    if args.start:
        if not args.plan: parser.error('--start requires a separately released frozen --plan')
        run(args.plan)
    else:
        print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
