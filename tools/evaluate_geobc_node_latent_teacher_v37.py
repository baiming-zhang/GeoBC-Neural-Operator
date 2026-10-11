"""Fixed TRAIN teacher and two latent-target operators on known DEV60.

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
    from geobc.node_latent_readout_contract_v37 import approved_contract, check_identity, complete_physical
    plan, review, manifest, truths, qualified = approved_contract(plan_path)
    science = plan['scientific']; outer = json.loads(os.environ['GEOBC_NODE_LATENT_READOUT_V37_GUARD'])
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
    lock = None; torch = None; model = None; cleanup = None

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
        from geobc.coarse_transolver_v19 import CoarseTransolver
        from geobc.coarse_fno_control_v22 import CoarseFNOControl
        from geobc.macro_training import encode_case, query_case
        from geobc.macro_readout_protocol import shared_truth_pass
        from geobc.available_cad_pilot_eval import _save_array
        from geobc.coverage_comparison_train import physical_metrics, aggregate
        from geobc.node_latent_teacher_v35 import complete_teacher_node_norm, original_node_latent, frozen_group_digests
        from geobc.geometry_head_transfer_v32 import state_digest
        from dataclasses import asdict
        from geobc.benchmark import FieldScaler
        from geobc.transformers import close_owned_process_group
        cleanup = close_owned_process_group
        torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        device = torch.cuda.current_device(); total = torch.cuda.get_device_properties(device).total_memory
        assert cap < total
        torch.cuda.set_per_process_memory_fraction(cap/total, device); torch.cuda.reset_peak_memory_stats()
        scaler = FieldScaler(**science['scaler']); inventory = {}; candidates = []
        teacher_latents, teacher_contexts, teacher_norms, teacher_latent_hashes = {}, {}, {}, {}
        for candidate in science['candidates']:
            guard(); name = candidate['name']; upstream = qualified[name]; ts = dict(upstream['plan']['scientific']); ts['family'] = candidate['family']
            endpoint = upstream['checkpoint']; step = candidate['step']
            result['checkpoint_read_attempts'] += 1
            saved = torch.load(check_identity(endpoint), map_location='cpu', weights_only=False)
            result['checkpoint_reads'] += 1; guard()
            assert saved['family'] == ts['family'] and saved['method'] == candidate['method'] and saved['step'] == saved['cursor'] == step
            assert saved['scientific_digest'] == upstream['plan']['scientific_digest']
            assert saved['case_exposure_counts'] == {k:5 for k in ts['TRAIN_keys']}
            assert saved['dataset_identity'] == science['dataset_identity'] and saved['scaler'] == science['scaler']
            assert set(saved['rng']) == {'python','numpy','torch','cuda'}
            assert saved['optimizer']['state'] and all(int(v['step']) == 1500 for v in saved['optimizer']['state'].values())
            assert saved['native_origin_clocks']['boot_id'] == upstream['summary']['execution']['boot_id']
            if candidate['is_teacher']:
                assert saved['full_order_digest'] == ts['full_order_digest']
            else:
                assert saved['full_order_digest'] == ts['full_order_digest'] == saved['marginal_order_digest']
                assert saved['initialization_teacher_checkpoint'] == ts['teacher_checkpoint'] and saved['initialization_teacher_SCI'] == ts['teacher_SCI']
                assert saved['inherited_teacher_case_exposure_counts'] == {k:5 for k in ts['TRAIN_keys']}
                assert saved['frozen_groups'] == ['encoder','head'] and saved['trainable_groups'] == ['operator']
                assert saved['training_objective'] == ts['loss'] and not saved['FEM_used_for_updates']
                assert saved['frozen_group_digests'] == upstream['arm']['frozen_group_initial_digests']
                assert saved['teacher_node_norms_identity'] == upstream['summary']['teacher_node_norms_identity']
            model = CoarseTransolver('transolver', seed=2026) if candidate['method']=='transolver_latent_teacher' else CoarseFNOControl(seed=2026)
            model.macro_family = ts['family']; model.load_state_dict(saved['model'],strict=True)
            if not candidate['is_teacher']: assert frozen_group_digests(model)==saved['frozen_group_digests']
            del saved; model.eval().requires_grad_(False); guard()
            spec = dict(name=name, family=ts['family'], step=step, checkpoint=endpoint); candidates.append(spec)
            teacher = candidate['is_teacher']
            row = dict(**spec,method=candidate['method'],is_teacher=teacher,train_CAD_count=100,case_exposure_count=5,
                model_profile=dict(model.profile(),native_model_method=model.profile()['method'],method=candidate['method'],training_objective=ts['loss']),
                cases=0,inference_seconds=0.,inference_epoch_seconds=0.,inference_scope='Full physical prediction plus complete teacher-node latent diagnostic; includes extra diagnostic work, not comparable to historical pure inference timings',
                training_plan_identity=science['training_plan_identity'],upstream_terminal_bindings=review['training_binding'],
                terminal_campaign_training_epoch_seconds=upstream['arm']['training_epoch_seconds'],
                terminal_campaign_training_monotonic_seconds=upstream['arm']['training_monotonic_seconds'],
                teacher_derived_initialization_passes=0 if teacher else 5,
                additional_teacher_target_passes=0 if teacher else 5,
                teacher_plus_marginal_training_epoch_seconds=upstream['arm']['training_epoch_seconds'] if teacher else upstream['arm']['teacher_plus_marginal_training_epoch_seconds'],
                inherited_teacher_cost=None if teacher else upstream['summary']['inherited_teacher_cost'],
                shared_cache_setup_epoch_seconds=upstream['summary']['cache_setup_epoch_seconds'],
                terminal_stage_epoch_seconds=upstream['external']['elapsed_epoch_seconds'],
                terminal_stage_monotonic_seconds=upstream['external']['elapsed_monotonic_seconds'],
                cost_scope='Teacher original physical5; student frozen teacher5 encoder/head plus freshoperator5 latent-target updates, not10 FEM passes or fresh5 fairness')
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
                    if teacher:
                        teacher_latents[key]=latent.detach();teacher_contexts[key]=context.detach()
                        teacher_norms[key]=complete_teacher_node_norm(teacher_latents[key],case['xyz'],guard)
                        teacher_latent_hashes[key]=state_digest({'latent':latent})
                    else:
                        assert key in teacher_latents and torch.equal(context,teacher_contexts[key])
                    residual_squared=0.;consumed=0
                    for start in range(0, n, science['decoder_chunk_size']):
                        guard(); stop = min(start+science['decoder_chunk_size'], n)
                        prediction[start:stop] = query_case(model, case, start, stop, context, latent).cpu().numpy()
                        if not teacher:
                            actual=original_node_latent(latent,case['xyz'][start:stop])
                            target=original_node_latent(teacher_latents[key],case['xyz'][start:stop])
                            residual_squared+=float((actual.double()-target.double()).square().sum())
                        consumed+=stop-start
                    torch.cuda.synchronize(); row['inference_seconds'] += time.monotonic()-started[1]
                    row['inference_epoch_seconds'] += time.time()-started[0]
                    physical = np.asarray(scaler.decode(prediction), dtype=np.float32)
                    assert prediction.shape == physical.shape == (n, 4) and np.isfinite(prediction).all() and np.isfinite(physical).all()
                    assert consumed==teacher_norms[key].node_count==n
                    node_diagnostic=dict(all_original_nodes=n,channels=16,teacher_checkpoint=science['teacher_checkpoint'],
                        teacher_latent_SHA256=teacher_latent_hashes[key],teacher_node_norm=asdict(teacher_norms[key]),
                        squared_residual=residual_squared,relative_squared=residual_squared/teacher_norms[key].squared_norm,
                        teacher_context_equal=True,target_is_teacher_not_FEM=True)
                    assert np.isfinite(node_diagnostic['relative_squared']) and node_diagnostic['relative_squared']>=0
                    directory = output/'predictions'/name/key
                    record = dict(candidate=name, family=ts['family'], step=step, key=key, checkpoint=endpoint,
                        status='completed', all_original_nodes=n, seed=science['seed']+index,
                        encoded=_save_array(d, np, directory/'encoded4.npy', prediction),
                        physical=_save_array(d, np, directory/'physical4.npy', physical), signed_stress_unchanged=True, teacher_node_latent=node_diagnostic)
                    file = directory/'manifest.json'; d.write(file, record); inventory[name+':'+key] = d.identity(file, hashed=True)
                    result['committed_finite_predictions'] += 1; row['cases'] += 1
                    guard(); persist(); del case, context, latent, prediction, physical
            check_identity(endpoint); del model; model = None
            close_owned_process_group(); torch.cuda.empty_cache(); guard()
        result.update(phase='all_predictions_revalidation', prediction_inventory=inventory); persist()
        counts = {k: manifest['records'][k]['num_nodes'] for k in keys}

        def record_reader(item):
            guard();record=d.read(check_identity(item));diag=record['teacher_node_latent']
            assert diag['all_original_nodes']==record['all_original_nodes'] and diag['channels']==16
            assert diag['teacher_checkpoint']==science['teacher_checkpoint'] and diag['teacher_latent_SHA256']==teacher_latent_hashes[record['key']]
            assert diag['teacher_node_norm']==asdict(teacher_norms[record['key']]) and diag['teacher_context_equal'] and diag['target_is_teacher_not_FEM']
            assert np.isfinite(diag['relative_squared']) and diag['relative_squared']>=0 and np.isfinite(diag['squared_residual']) and diag['squared_residual']>=0
            assert diag['relative_squared']==diag['squared_residual']/diag['teacher_node_norm']['squared_norm']
            return record
        def array_reader(item):
            guard(); return np.load(check_identity(item), mmap_mode='r', allow_pickle=False)
        def save_barrier(barrier):
            assert result['owned_GT_array_reads'] == result['owned_GT_read_attempts'] == 0
            assert barrier['finite_predictions'] == science['fixed_prediction_denominator']
            d.write(output/'prediction_barrier.json', barrier)
            result.update(phase='teacher_comparison_after_barrier',prediction_barrier_complete=True);persist()
            teacher_name=science['candidates'][0]['name']
            assert science['candidates'][0]['is_teacher'] and len(teacher_latents)==60
            for spec in candidates:
                if spec['name']==teacher_name:continue
                physical_rows={};latent_rows={}
                for key in keys:
                    guard();teacher_record=record_reader(inventory[teacher_name+':'+key]);student_record=record_reader(inventory[spec['name']+':'+key])
                    physical_rows[key]=physical_metrics(np,array_reader(teacher_record['physical']),array_reader(student_record['physical']))
                    latent_rows[key]=student_record['teacher_node_latent']['relative_squared']
                row=result['candidates'][spec['name']]
                row['full60_to_teacher_physical']=aggregate(physical_rows,60)
                row['full60_to_teacher_node_latent']=dict(fixed_denominator=60,per_case_relative_squared=latent_rows,
                    case_mean_relative_L2=sum(v**.5 for v in latent_rows.values())/60,
                    case_mean_relative_squared=sum(latent_rows.values())/60,
                    worst_case_relative_L2=max(latent_rows.values())**.5,all_cases_complete=True,target_is_teacher_not_FEM=True)
            result.update(phase='shared_knownDEV_GT_once',teacher_comparisons_complete=True);persist()
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
        # Only existing complete physical scalar cases are reused; no old arrays,
        # inference, raw truth or revealed FINAL are opened.
        prior=d.read(check_identity(science['prior_development_binding']['terminal']['summary']))
        for name,old in prior['candidates'].items():
            complete_physical(old['full60'],keys)
            row=dict(old);row.update(reused_immutable_scalar_metrics=True,inference_replayed=False,current_stage_inference_seconds=0.,scalar_metrics_identity=science['prior_development_binding']['terminal']['summary'])
            result['candidates'][name]=row
        from tools.export_geobc_exposure_comparison import paired
        names=sorted(result['candidates'])
        result['paired_CAD20']={a+':'+b:paired(result['candidates'][a]['full60'],result['candidates'][b]['full60'],keys) for i,a in enumerate(names) for b in names[i+1:]}
        result['learning_diagnostics']={}
        for c in science['candidates']:
            if c['is_teacher']:continue
            arm=qualified[c['name']]['arm'];initial=arm['initial_fullTRAIN_teacher_node_latent']['case_mean_relative_L2']
            endpoint=arm['endpoints']['1500']['full_TRAIN_teacher_node_latent']['case_mean_relative_L2']
            th=science['learning_diagnostic'];ratio=endpoint/initial if initial>0 else None
            result['learning_diagnostics'][c['name']]=dict(initial_TRAIN_teacher_latent_relative_L2=initial,endpoint_TRAIN_teacher_latent_relative_L2=endpoint,ratio=ratio,
                passed=ratio is not None and ratio<=th['TRAIN_mean_teacher_latent_relative_L2_ratio_at_most'] and endpoint<=th['TRAIN_mean_teacher_latent_relative_L2_at_most'],
                thresholds=th,automatic_continuation=False,FEM_improvement_proven_by_this_gate=False)
        students=[c['name'] for c in science['candidates'] if not c['is_teacher']]
        scores={n:result['candidates'][n]['full60']['selection_score'] for n in students}
        result['descriptive_best_current_student']=sorted(n for n,v in scores.items() if v==min(scores.values()))
        guard();result.update(status='completed',phase='terminal',protocol_qualified=True,eligible_best=None,
            full_previous_scalar_case_denominator=60,comparison_scope='Teacher and2latentstudents plus14prior physicalscalar rows; excludes olderR64FNO10/40 endpoints; no finalsolver choice or fairfresh5 ranking')
    except Exception as error:
        result.update(status='failed', phase='terminal_failure', error_type=type(error).__name__, error=str(error),
                      eligible_best=None, protocol_qualified=False)
        raise
    finally:
        try:
            if cleanup is not None: cleanup()
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
