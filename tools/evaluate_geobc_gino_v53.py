"""Full knownDEV60 prediction-first GINO readout, inert without root release."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
from collections import Counter
import json
import os
import random
from pathlib import Path
import signal


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.gino_readout_contract_v53 import approved_contract,check_identity,complete_physical
    p,r,m,truths,up=approved_contract(plan_path)
    s=p['scientific'];ts=up['plan']['scientific'];endpoint=up['endpoint']
    outer=json.loads(os.environ['GEOBC_GINO_READOUT_V53_GUARD'])
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert outer['scientific_digest']==p['scientific_digest'] and outer['boot_id']==boot
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    output=d.path(p['output']);assert output.exists() and not (output/'execution.json').exists()
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,boot_id=boot,scientific_digest=p['scientific_digest'])
    d.write(output/'execution.json',execution)
    keys=s['case_keys'];spec=s['candidate'];name=spec['name'];step=spec['step']
    result=dict(status='running',active=True,closed=False,execution=execution,
        phase='checkpoint_validation',candidates={},prediction_inventory={},fixed_prediction_denominator=60,
        fixed_case_denominator=60,committed_finite_predictions=0,input_array_read_attempts=0,
        input_array_reads=0,checkpoint_read_attempts=0,checkpoint_reads=0,checkpoint_body_hash_reads=0,
        owned_GT_read_attempts=0,owned_GT_array_reads=0,raw_GT_reads=0,FINAL_numeric_reads=0,
        training_updates=0,scaler_fits=0,peak_allocated_bytes=0,peak_reserved_bytes=0,
        peak_total_device_used_bytes_sampled=0,backend_external_workspace_exact_peak_known=False,
        automatic_replay=False,eligible_best=None,protocol_qualified=False,
        formal5_eligible=endpoint['formal5_eligible'],endpoint_mode=endpoint['mode'],
        diagnostic_only=not endpoint['formal5_eligible'],final_solver_selected=False,
        independent_offnode_FEM_accuracy_measured=False)
    torch=lock=radius=None

    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:
            raise TimeoutError('Frozen GINO readout dual-clock whole budget')
        if torch is not None and torch.cuda.is_initialized():
            free,total=torch.cuda.mem_get_info()
            values=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_total_device_used_bytes_sampled=total-free)
            for key,value in values.items():
                result[key]=max(result[key],value)
                if value>s['memory_limit_bytes']:raise MemoryError('Readout memory component exceeds30.5GB: '+key)

    def persist():
        result.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        if radius is not None:result['radius_costs']=dict(costs)
        d.write(output/'status.json',result)

    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External guard terminated readout')))
    try:
        guard()
        import fcntl
        lock=d.path(s['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch as tm
        torch=tm
        from geobc.gino_physical_training_v52 import build_candidate
        from geobc.gino_state_v52 import state_digest,load_verified_state
        from geobc.gino_knownDEV_radius_v53 import KnownDEVRadiusCache
        from geobc.prospective_dataset_v2 import load_case
        from geobc.macro_readout_protocol import shared_truth_pass
        from geobc.available_cad_pilot_eval import _save_array
        from geobc.coverage_comparison_train import physical_metrics,aggregate
        from geobc.benchmark import FieldScaler
        from geobc.macro_training import balanced_schedule
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device=torch.cuda.current_device();total=torch.cuda.get_device_properties(device).total_memory
        assert s['memory_limit_bytes']<total
        torch.cuda.set_per_process_memory_fraction(s['memory_limit_bytes']/total,device);torch.cuda.reset_peak_memory_stats()
        random.seed(s['seed']);np.random.seed(s['seed'])
        torch.manual_seed(s['seed']);torch.cuda.manual_seed_all(s['seed'])
        # The private upstream classes must exist before unpickling their
        # constructor metadata; do not torch.load before this pinned build.
        model=build_candidate('cuda')
        assert state_digest(model.state_dict())==ts['expected_fresh_initial_model_sha256']
        ck=endpoint['checkpoint_identity'];guard();d.stat_matches(ck)
        hashed=d.identity(ck['path'],hashed=True);result['checkpoint_body_hash_reads']+=1
        if 'sha256' in ck:assert hashed==ck
        result['checkpoint_read_attempts']+=1
        saved=torch.load(d.path(ck['path']),map_location='cpu',weights_only=False)
        result['checkpoint_reads']+=1;guard();d.stat_matches(ck)
        assert saved['step']==saved['cursor']==step and saved['family']=='GINO_PHYSICAL'
        assert saved['scientific_digest']==up['plan']['scientific_digest']
        assert saved['dataset_identity']==s['dataset_identity'] and saved['scaler']==s['scaler']
        order=balanced_schedule(ts['TRAIN_keys'],5,2026)
        assert saved['full_order_digest']==ts['full_order_digest']==d.digest(order)
        expected={key:0 for key in ts['TRAIN_keys']};expected.update(Counter(order[:step]))
        assert saved['case_exposure_counts']==expected and sum(expected.values())==step
        assert set(saved['rng'])=={'python','numpy','torch','cuda'}
        assert saved['native_origin_clocks']['boot_id']==up['summary']['execution']['boot_id']
        assert saved['optimizer']['state'] and all(int(v['step'])==step for v in saved['optimizer']['state'].values())
        assert saved['model_profile']==up['summary']['model_profile']
        assert saved['cost_status']['actual_largest_fullN_gradient_check']['passed']
        assert saved['cost_status']['actual_fresh_initial_model_sha256']==ts['expected_fresh_initial_model_sha256']
        for group in saved['optimizer']['param_groups']:
            recipe=ts['optimizer']
            assert group['lr']==recipe['learning_rate'] and group['weight_decay']==recipe['weight_decay']
            assert tuple(group['betas'])==tuple(recipe['betas']) and group['eps']==recipe['eps']
            assert group['foreach'] and not group['fused']
        loaded_hash=state_digest(saved['model']);load_verified_state(model,saved['model'])
        assert state_digest(model.state_dict())==loaded_hash
        assert all(torch.isfinite(v).all() for v in model.state_dict().values() if isinstance(v,torch.Tensor))
        result['loaded_endpoint_model_sha256']=loaded_hash
        result['owned_checkpoint_hashed_identity']=hashed
        exposure_counts=dict(saved['case_exposure_counts']);del saved
        model.eval();guard();scaler=FieldScaler(**s['scaler'])
        costs=dict(radius_build_seconds=0.,radius_build_count=0,radius_save_seconds=0.,
            radius_read_seconds=0.,radius_read_count=0,radius_cache_reuse_count=0)
        def progress(phase,**details):
            guard();result['last_radius_progress']=dict(phase=phase,**details);persist()
        radius=KnownDEVRadiusCache(s['radius_plan'],dict(geometries=s['geometries'],case_keys=keys),
            torch,np,torch.device('cuda'),costs,progress)
        row=dict(**spec,formal5_eligible=endpoint['formal5_eligible'],diagnostic_only=not endpoint['formal5_eligible'],
            checkpoint=hashed,checkpoint_identity_scope='Hash computed once inside released readout; training already exited',
            train_CAD_count=100,case_exposure_counts=exposure_counts,
            exposure_min=min(exposure_counts.values()),exposure_max=max(exposure_counts.values()),
            model_profile=up['summary']['model_profile'],parameter_count=up['summary']['parameter_count'],
            real_parameter_DOF=up['summary']['real_parameter_DOF'],cases=0,inference_seconds=0.,inference_epoch_seconds=0.,
            training_plan_identity=s['training_plan_identity'],upstream_terminal_bindings=r['training_binding'],
            terminal_campaign_training_epoch_seconds=up['summary']['training_epoch_seconds'],
            terminal_campaign_training_monotonic_seconds=up['summary']['training_monotonic_seconds'],
            terminal_stage_epoch_seconds=up['external']['elapsed_epoch_seconds'],
            terminal_stage_monotonic_seconds=up['external']['elapsed_monotonic_seconds'],
            original_whole_epoch_span_seconds=up['external']['original_whole_epoch_span_seconds'],
            original_whole_monotonic_span_seconds=up['external']['original_whole_monotonic_span_seconds'],
            prior_failure_cost=ts['prior_failure_cost'],cost_scope='Native setup/qualification/checkpoints/TRAIN readout plus both real zero-update failed setups; original total UTC/mono span includes inter-attempt analysis; no silent zero fill or doubled same-attempt cost',
            endpoint_mode=endpoint['mode'])
        result['candidates'][name]=row;inventory={}
        # Barrier identity uses the actually hashed own checkpoint for both
        # candidate specification and every committed prediction record.
        pred_spec=dict(name=name,family='GINO_PHYSICAL',step=step,checkpoint=hashed)
        def input_event(stage,key,member,item):
            assert member not in ('target','physical');guard()
            if stage=='attempt':result['input_array_read_attempts']+=1
            if stage=='opened':result['input_array_reads']+=1
        result['phase']='prediction_commit';persist()
        with torch.no_grad():
            for index,key in enumerate(keys):
                guard();case=load_case(m,key,with_targets=False,train_CAD_count=100,device='cuda',for_fno=False,on_read=input_event)
                assert case['scientific_group']=='DEV' and not {'target','physical'}&set(case)
                case=radius.attach(case);n=m['records'][key]['num_nodes'];assert len(case['xyz'])==n
                torch.manual_seed(s['seed']+index);torch.cuda.manual_seed_all(s['seed']+index)
                torch.cuda.synchronize();started=(time.time(),time.monotonic());state=model.encode_case(case)
                prediction=np.empty((n,4),dtype=np.float32)
                for begin in range(0,n,2048):
                    guard();end=min(begin+2048,n);ids=torch.arange(begin,end,dtype=torch.int64,device='cuda')
                    prediction[begin:end]=model.query(state,ids).cpu().numpy()
                torch.cuda.synchronize();row['inference_seconds']+=time.monotonic()-started[1]
                row['inference_epoch_seconds']+=time.time()-started[0]
                physical=np.asarray(scaler.decode(prediction),dtype=np.float32)
                assert np.isfinite(prediction).all() and np.isfinite(physical).all()
                directory=output/'predictions'/name/key
                record=dict(candidate=name,family='GINO_PHYSICAL',step=step,key=key,checkpoint=hashed,
                    status='completed',all_original_nodes=n,seed=s['seed']+index,
                    encoded=_save_array(d,np,directory/'encoded4.npy',prediction),
                    physical=_save_array(d,np,directory/'physical4.npy',physical),signed_stress_unchanged=True)
                file=directory/'manifest.json';d.write(file,record);inventory[name+':'+key]=d.identity(file,hashed=True)
                row['cases']+=1;result['committed_finite_predictions']+=1
                guard();persist();del case,state,prediction,physical,ids
        assert row['cases']==60
        d.stat_matches(ck);del model;torch.cuda.empty_cache();guard()
        result.update(phase='all_predictions_revalidation',prediction_inventory=inventory);persist()
        counts={key:m['records'][key]['num_nodes'] for key in keys}
        def record_reader(item):guard();return d.read(check_identity(item))
        def array_reader(item):guard();return np.load(check_identity(item),mmap_mode='r',allow_pickle=False)
        def save_barrier(barrier):
            assert result['owned_GT_array_reads']==result['owned_GT_read_attempts']==0 and barrier['finite_predictions']==60
            d.write(output/'prediction_barrier.json',barrier)
            result.update(phase='shared_knownDEV_GT_once',prediction_barrier_complete=True);persist()
        def truth_reader(key):
            guard();assert result['prediction_barrier_complete'];result['owned_GT_read_attempts']+=1
            return np.load(check_identity(truths['records'][key]['physical']),mmap_mode='r',allow_pickle=False)
        def on_truth(key):result['owned_GT_array_reads']+=1;persist()
        per=shared_truth_pass(np,[pred_spec],keys,counts,inventory,record_reader,array_reader,truth_reader,save_barrier,on_truth,physical_metrics,guard)
        assert result['owned_GT_array_reads']==result['owned_GT_read_attempts']==60
        row['full60']=aggregate(per[name],60);complete_physical(row['full60'],keys)
        row['per_load20']={load:aggregate({k:per[name][k] for k in keys if k.endswith('/'+load)},20) for load in ('ver','hor','tor')}
        for ref in s['scalar_references']:
            old=d.read(check_identity(ref['binding']['summary']));control=dict(old['candidates'][ref['name']])
            complete_physical(control['full60'],keys)
            control.update(reused_immutable_scalar_metrics=True,inference_replayed=False,current_stage_inference_seconds=0.,
                scalar_metrics_identity=ref['binding']['summary'],comparison_scope='Same current100/scaler/full60 physical5 control; different architecture/capacity/backend. Partial-budget GINO is not equal-exposure formal comparison.')
            result['candidates'][ref['name']]=control
        from tools.export_geobc_exposure_comparison import paired
        result['paired_CAD20']={name+':'+ref['name']:paired(row['full60'],result['candidates'][ref['name']]['full60'],keys) for ref in s['scalar_references']}
        scores={n:v['full60']['selection_score'] for n,v in result['candidates'].items()}
        result['comparison_scope']='KnownDEV iterative single-seed. Formal complete5 only eligible; budget-checkpoint full60 diagnosis retains uneven/exact exposure and costs, never substitutes for completed5 or architecture upper bound.'
        result['DEV_gate']=dict(passed=False,releases_continuation=False,scope='No automatic continuation; actual macro decision requires absolute U/S/hot/cost and endpoint eligibility')
        if endpoint['formal5_eligible']:
            result['eligible_best']=sorted(n for n,score in scores.items() if score==min(scores.values()))
        else:
            result['eligible_best']=None;result['formal_selection_score']=None
        guard();result.update(status='completed',phase='terminal',protocol_qualified=True,
            radius_receipt=radius.receipt(),accurate_complete60_diagnostic_available=True)
    except BaseException as error:
        result.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error),
            eligible_best=None,protocol_qualified=False)
        raise
    finally:
        if radius is not None:result['radius_costs']=dict(costs)
        result.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,
            elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(output/'summary.json',result);d.write(output/'status.json',result)
        if lock is not None:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');parser.add_argument('--plan')
    args=parser.parse_args()
    if args.start:
        if not args.plan:parser.error('--start requires separate frozen --plan')
        run(args.plan)
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
