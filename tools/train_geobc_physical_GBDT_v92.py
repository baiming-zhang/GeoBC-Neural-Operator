"""One GPU regression screen, with both knownDEV predictions before labels."""
import time
ENTRY_EPOCH, ENTRY_MONOTONIC = time.time(), time.monotonic()
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import signal


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.physical_gbdt_contract_v92 import approved, TRAIN, DEV
    p = approved(plan_path); s = p['scientific']; out = d.path(p['output'])
    outer = json.loads(os.environ['GEOBC_PHYSICAL_GBDT_V92_GUARD'])
    assert outer['scientific_digest'] == p['scientific_digest']
    assert outer['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid']) == outer['guard_process']
    assert out.exists() and not (out/'execution.json').exists()
    execution = dict(run_process=d.process_snapshot(os.getpid()), guard=outer,
                     boot_id=outer['boot_id'], scientific_digest=p['scientific_digest'])
    d.write(out/'execution.json', execution)
    a = dict(status='running', active=True, closed=False, phase='setup', execution=execution,
        rounds_attempted=0, rounds_completed=0, update_in_progress=False, possible_partial_update=False, actual_GPU_training_verified=False,
        reads=[], input_cache={}, TRAIN_target_cache={}, predictions={}, metrics={},
        raw_CSV_H5_deck_archive_reads=0, physical_U_GT_U_PT_reads=0, new_physical_solves=0,
        new20DEV_FINAL_reads=0, scaler_fits=0, baseline_replays=0, explicit_sign_rule_search=False,
        knownDEV_prediction_barrier_passed=False, sampled_peak_device_used_bytes=0,
        external_backend_exact_peak_known=False, automatic_replay=False, own_DEV_prediction_array_reads=0,
        own_DEV_prediction_file_hash_reads=0, own_saved_model_loads=0, per_case_input_costs={})
    torch = None; booster = None; lock = None
    phase_deadline = (ENTRY_EPOCH+s['cache_setup_budget_seconds'], ENTRY_MONOTONIC+s['cache_setup_budget_seconds'])

    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH, elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(out/'status.json', a)
        d.write(out/'clock.json', dict(status=a['status'], active=a['active'], closed=a['closed'],
            entry_epoch=ENTRY_EPOCH, entry_monotonic=ENTRY_MONOTONIC,
            elapsed_epoch_seconds=a['elapsed_epoch_seconds'], elapsed_monotonic_seconds=a['elapsed_monotonic_seconds'], outer=outer))

    def guard():
        limits = [(outer['deadline_epoch'], outer['deadline_monotonic'])]
        if phase_deadline is not None: limits.append(phase_deadline)
        assert min(min(e-time.time(), m-time.monotonic()) for e, m in limits) > 0, 'Earlier absolute deadline exceeded'
        if torch is not None:
            free, total = torch.cuda.mem_get_info(); used = total-free
            a['sampled_peak_device_used_bytes'] = max(a['sampled_peak_device_used_bytes'], used)
            assert used <= s['memory_limit_bytes'], 'Shared device30.5GB memory budget exceeded'

    def values_sha(np, v): return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()

    def read_array(np, key, name):
        guard(); q = s['records'][key][name]; d.stat_matches(q)
        token = dict(key=key, array=name, attempted=True, completed=False, identity=q)
        a['reads'].append(token); persist()
        value = np.load(d.path(q['path']), allow_pickle=False)
        assert list(value.shape) == q['shape'] and str(value.dtype) == q['dtype']
        assert values_sha(np, value) == q['values_sha256'] and np.isfinite(value).all()
        token['completed'] = True; persist(); return value

    def save_array(np, path, value):
        assert not path.exists(); np.save(path, value, allow_pickle=False)
        return dict(**d.identity(path, hashed=True), shape=list(value.shape), dtype=str(value.dtype), values_sha256=values_sha(np, value))

    def metric(np, actual, pred):
        assert actual.shape == pred.shape and np.isfinite(actual).all() and np.isfinite(pred).all()
        energy = float(np.sum(actual**2)); assert energy > 0, 'Zero case norm unsupported, do not floor/drop'
        delta = pred-actual; hot = np.argsort(-np.abs(actual), kind='stable')[:math.ceil(.01*len(actual))]
        hot_energy = float(np.sum(actual[hot]**2)); assert hot_energy > 0
        return dict(nodes=len(actual), stress_relative_l2=float(np.linalg.norm(delta)/math.sqrt(energy)),
            hotspot_relative_l2=float(np.linalg.norm(delta[hot])/math.sqrt(hot_energy)), hotspot_nodes=len(hot),
            RMSE_MPa=float(np.sqrt(np.mean(delta**2))), MAE_MPa=float(np.mean(np.abs(delta))),
            max_abs_error_MPa=float(np.max(np.abs(delta))), reference_energy_MPa2=energy,
            error_energy_MPa2=float(np.sum(delta**2)), S2_weighted_sign_error=float(np.sum(actual[(actual!=0)&(actual*pred<=0)]**2)/energy),
            signed_original_target=True, full_original_node_count=True, all_finite=True)

    signal.signal(signal.SIGTERM, lambda *args: (_ for _ in ()).throw(TimeoutError('External guard stopped child')))
    try:
        persist(); guard()
        import fcntl
        lock = d.path(s['GPU_lock']).open('a+'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        import numpy as np
        import torch as torch_module
        import xgboost as xgb
        torch = torch_module; torch.set_num_threads(4); torch.cuda.set_device(0)
        assert xgb.__version__ == '3.0.5' and xgb.build_info()['USE_CUDA']
        assert d.sha(xgb.core._LIB._name) == s['library_so_sha256']
        a['library_build_info'] = xgb.build_info(); a['CUDA_device'] = torch.cuda.get_device_name()
        free, total = torch.cuda.mem_get_info()
        a['initial_device_used_bytes'] = total-free
        a['GPU_timing_isolation'] = 'Not guaranteed; another project worker observed during root preparation, no other process operated'
        a['library_pre_GT_validation_passed'] = True; guard(); persist()
        a['library_and_metadata_setup_epoch_seconds']=time.time()-ENTRY_EPOCH
        a['library_and_metadata_setup_monotonic_seconds']=time.monotonic()-ENTRY_MONOTONIC
        X = {}; truth = {}; mean = np.asarray(s['tensor6_scaler']['mean']); std = np.asarray(s['tensor6_scaler']['std'])
        assert mean.shape == std.shape == (6,) and np.isfinite(mean).all() and (std>0).all()
        cache = out/'owned_cache'; cache.mkdir(exist_ok=False)
        for key in TRAIN+DEV:
            input_epoch, input_mono=time.time(), time.monotonic()
            a['phase']='input_cache'; a['current_case']=key; persist()
            sigma = read_array(np, key, 'sigma_MPa'); valid = read_array(np, key, 'valid')
            assert sigma.shape == (s['records'][key]['node_count'], 6) and valid.shape == (len(sigma),)
            assert valid.all(), 'Previously accepted allN coverage changed, no node subset'
            normalized = ((sigma-mean)/std).astype(np.float32); normalized[~valid]=0
            X[key] = np.ascontiguousarray(np.column_stack((normalized, valid.astype(np.float32))), dtype=np.float32)
            assert np.isfinite(X[key]).all()
            a['input_cache'][key]=save_array(np, cache/(key.replace('/', '_')+'_X.npy'), X[key])
            a['per_case_input_costs'][key]=dict(read_hash_normalize_ownedcache_epoch_seconds=time.time()-input_epoch,
                read_hash_normalize_ownedcache_monotonic_seconds=time.monotonic()-input_mono)
            persist()
        for key in TRAIN:
            truth[key] = read_array(np, key, 'signed_stress')
            assert truth[key].shape == (s['records'][key]['node_count'],)
            a['TRAIN_target_cache'][key] = save_array(np, cache/(key.replace('/', '_')+'_S.npy'), truth[key]); persist()
        joined = np.ascontiguousarray(np.concatenate([X[k] for k in TRAIN]), dtype=np.float32)
        target = np.ascontiguousarray(np.concatenate([truth[k] for k in TRAIN]), dtype=np.float32)
        energies = {k: float(np.sum(truth[k]**2)) for k in TRAIN}; assert all(v>0 and math.isfinite(v) for v in energies.values())
        weights64 = np.concatenate([np.full(len(truth[k]), 1/energies[k], dtype=np.float64) for k in TRAIN])
        weight_mean = float(weights64.mean()); weights = np.ascontiguousarray(weights64/weight_mean, dtype=np.float32)
        assert (weights>0).all() and np.isfinite(target).all() and np.isfinite(weights).all()
        a['weight_recipe_actual']=dict(TRAIN_energies_MPa2=energies, original_weight_mean=weight_mean,
            normalized_mean=float(weights.mean(dtype=np.float64)), min=float(weights.min()), max=float(weights.max()), TRAIN_nodes=len(target))
        assert len(target) == 683842 and joined.shape == (683842, 7)
        a['phase']='TRAIN_quantile_matrix'; persist(); guard()
        matrix = xgb.QuantileDMatrix(joined, label=target, weight=weights, max_bin=s['parameters']['max_bin'], nthread=4)
        assert matrix.num_row()==683842 and matrix.num_col()==7
        a['TRAIN_quantile_only']=True; guard(); persist()
        phase_deadline = (time.time()+s['training_budget_seconds'], time.monotonic()+s['training_budget_seconds'])
        train_began_epoch, train_began_mono = time.time(), time.monotonic()
        a['phase']='training'; persist()

        def checkpoint(model, rounds):
            guard(); tmp=out/'last_snapshot.pkl.tmp'
            with tmp.open('wb') as f: pickle.dump(dict(booster=model, completed_rounds=rounds,
                xgboost_version=xgb.__version__, scientific_digest=p['scientific_digest'], params=s['parameters'],
                input_cache=a['input_cache'], TRAIN_target_cache=a['TRAIN_target_cache'], weight_recipe=a['weight_recipe_actual'],
                execution=execution, cost=dict(a), complete_internal_RNG_exactness_not_claimed=True), f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, out/'last_snapshot.pkl')
            tmp_model=out/'last_model.tmp.json'; model.save_model(tmp_model); os.replace(tmp_model, out/'last_model.json')
            a['last_checkpoint']=dict(rounds=rounds, snapshot=d.identity(out/'last_snapshot.pkl'), model=d.identity(out/'last_model.json')); guard()

        log=(out/'loss.jsonl').open('x',encoding='utf-8')
        class Callback(xgb.callback.TrainingCallback):
            def before_iteration(self, model, epoch, evals_log):
                guard(); a['update_in_progress']=True; a['rounds_attempted']+=1; persist(); return False

            def after_iteration(self, model, epoch, evals_log):
                a['rounds_completed']=epoch+1; a['update_in_progress']=False
                config=json.loads(model.save_config()); assert config['learner']['generic_param']['device']=='cuda:0'
                assert 'grow_gpu_hist' in json.dumps(config), 'GPU updater not selected; CPU fallback ineligible'
                a['actual_GPU_training_verified']=True
                if epoch==0: a['first_actual_full_TRAIN_GPU_round_config']=config
                assert model.num_boosted_rounds()==epoch+1
                log.write(json.dumps(dict(round=epoch+1, weighted_TRAIN_RMSE_MPa=evals_log['TRAIN']['rmse'][-1],
                    epoch_seconds=time.time()-train_began_epoch, monotonic_seconds=time.monotonic()-train_began_mono))+'\n');log.flush()
                if (epoch+1)%50==0: checkpoint(model, epoch+1)
                guard(); persist(); return False

        try:
            booster = xgb.train(s['parameters'], matrix, num_boost_round=s['num_boost_round'], evals=[(matrix,'TRAIN')],
                callbacks=[Callback()], verbose_eval=False)
        finally: log.close()
        assert booster.num_boosted_rounds()==a['rounds_completed']==400 and a['actual_GPU_training_verified']
        a.update(training_epoch_seconds=time.time()-train_began_epoch, training_monotonic_seconds=time.monotonic()-train_began_mono)
        checkpoint(booster, 400); a['model_config']=json.loads(booster.save_config())
        phase_deadline=(time.time()+s['readout_budget_seconds'], time.monotonic()+s['readout_budget_seconds'])
        load_epoch, load_mono=time.time(), time.monotonic()
        deployed=xgb.Booster(params=s['parameters']); deployed.load_model(out/'last_model.json')
        a['own_saved_model_loads']+=1
        assert deployed.num_boosted_rounds()==400
        assert hashlib.sha256(deployed.save_raw(raw_format='ubj')).hexdigest()==hashlib.sha256(booster.save_raw(raw_format='ubj')).hexdigest()
        deployed.set_param({'device':'cuda:0'})
        assert json.loads(deployed.save_config())['learner']['generic_param']['device']=='cuda:0'
        a['saved_model_reload_verified']=True
        a['saved_model_load_epoch_seconds']=time.time()-load_epoch; a['saved_model_load_monotonic_seconds']=time.monotonic()-load_mono
        a['phase']='allN_prediction'; persist(); resident_predictions={}
        for key in TRAIN+DEV:
            guard(); began_epoch=time.time(); began_mono=time.monotonic()
            evaluation=xgb.QuantileDMatrix(X[key], ref=matrix, max_bin=s['parameters']['max_bin'], nthread=4)
            assert evaluation.num_row()==len(X[key]) and evaluation.num_col()==7
            pred=deployed.predict(evaluation)
            assert pred.shape==(len(X[key]),) and np.isfinite(pred).all()
            value=pred.astype(np.float64); destination=out/(key.replace('/', '_')+'_signedS_MPa.npy')
            a['predictions'][key]=dict(array=save_array(np, destination, value), nodes=len(value),
                inference_matrix_predict_save_epoch_seconds=time.time()-began_epoch,
                inference_matrix_predict_save_monotonic_seconds=time.monotonic()-began_mono,
                groundtruth_deferred=(key in DEV), quantile_ref_TRAIN=True)
            resident_predictions[key]=value; del evaluation; persist()
        assert set(a['predictions'])==set(TRAIN+DEV)
        for key in DEV:
            identity=a['predictions'][key]['array']; d.stat_matches(identity)
            assert d.sha(identity['path'])==identity['sha256']
            a['own_DEV_prediction_file_hash_reads']+=1
            own=np.load(d.path(identity['path']),allow_pickle=False)
            a['own_DEV_prediction_array_reads']+=1
            assert own.shape==(s['records'][key]['node_count'],) and np.isfinite(own).all()
            assert values_sha(np, own)==identity['values_sha256'] and np.array_equal(own,resident_predictions[key])
        assert sum(v['array']=='signed_stress' and v['key'] in DEV for v in a['reads'])==0
        a['knownDEV_prediction_barrier_passed']=True; a['phase']='knownDEV_label_readout'; persist()
        for key in DEV: truth[key]=read_array(np,key,'signed_stress')
        for key in TRAIN+DEV: a['metrics'][key]=metric(np, truth[key], resident_predictions[key])
        for name, keys in [('TRAIN', TRAIN), ('knownDEV', DEV)]:
            a[name+'_aggregate']=dict(fixed_denominator=len(keys), finite_metrics_cases=len(keys),
                mean_signed_stress_relativeL2=sum(a['metrics'][k]['stress_relative_l2'] for k in keys)/len(keys),
                worst_signed_stress_relativeL2=max(a['metrics'][k]['stress_relative_l2'] for k in keys),
                mean_hotspot_relativeL2=sum(a['metrics'][k]['hotspot_relative_l2'] for k in keys)/len(keys),
                worst_hotspot_relativeL2=max(a['metrics'][k]['hotspot_relative_l2'] for k in keys))
        gate=s['gate']; a['selection_gate']=dict(passed=all(a['metrics'][k]['stress_relative_l2']<gate['all2_knownDEV_signed_relativeL2_strict_below'] for k in DEV)
            and a['knownDEV_aggregate']['mean_hotspot_relativeL2']<gate['knownDEV_mean_hotspot_relativeL2_strict_below'],
            thresholds=gate, fixed_denominator=2, final_solver_selected=False, automatic_expansion=False)
        assert len(a['reads'])==15 and all(v['completed'] for v in a['reads'])
        assert sum(v['array']=='signed_stress' and v['key'] in TRAIN for v in a['reads'])==3
        assert sum(v['array']=='signed_stress' and v['key'] in DEV for v in a['reads'])==2
        guard(); torch.cuda.synchronize()
        a.update(status='completed', phase='terminal', protocol_qualified=True, budget_true=True,
                 training_complete=True, differentiable_continuous_field=False, final_solver_selected=False)
    except BaseException as err:
        a.update(status='failed', phase='terminal', error_type=type(err).__name__, error=str(err),
            budget_true=False, possible_partial_update=a['update_in_progress'])
        raise
    finally:
        a.update(active=False, closed=True); persist(); d.write(out/'summary.json', a)
        if lock is not None: lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');parser.add_argument('--plan')
    args=parser.parse_args()
    if args.start:
        if not args.plan:parser.error('--start requires frozen --plan')
        run(args.plan)
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
