"""One bounded selected-field100-to1000 physical learning continuation."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
import os
import json
import signal
from pathlib import Path


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.single_field_learning_contract_v59 import approved_contract
    p,r,m,up=approved_contract(plan_path);s=p['scientific'];keys=s['selected_TRAIN_keys']
    outer=json.loads(os.environ['GEOBC_SINGLE_FIELD_V59_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest']
    assert outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    o=d.path(p['output']);assert o.exists() and not (o/'execution.json').exists()
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,
        boot_id=outer['boot_id'],scientific_digest=p['scientific_digest']);d.write(o/'execution.json',execution)
    a=dict(status='running',active=True,closed=False,execution=execution,phase='setup',arms={},
        optimizer_step_attempts=0,optimizer_step_completions=0,optimizer_step_in_progress=False,
        possible_partial_optimizer_update=False,parent_checkpoint_hash_reads=0,parent_checkpoint_loads=0,
        input_array_attempts=0,input_array_reads=0,physical_TRAIN_attempts=0,physical_TRAIN_reads=0,
        encoded_TRAIN_reads=0,resident_physical_target_reuses=0,raw_GT_reads=0,DEV_FINAL_reads=0,scaler_fits=0,
        peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0,
        backend_external_workspace_exact_peak_known=False,automatic_replay=False,final_solver_selected=False)
    torch=None;lock=None;phase_deadline=(ENTRY_EPOCH+180,ENTRY_MONOTONIC+180)
    def guard():
        limits=[(outer['deadline_epoch'],outer['deadline_monotonic'])]
        if phase_deadline is not None:limits.append(phase_deadline)
        if min(min(x-time.time(),y-time.monotonic())for x,y in limits)<=0:raise TimeoutError('Frozen earlier epoch/monotonic deadline')
        if torch is not None and torch.cuda.is_initialized():
            a['peak_allocated_bytes']=max(a['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
            a['peak_reserved_bytes']=max(a['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
            free,total=torch.cuda.mem_get_info();a['peak_total_device_used_bytes_sampled']=max(a['peak_total_device_used_bytes_sampled'],total-free)
            if max(a[k]for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))>s['memory_limit_bytes']:raise MemoryError('30.5GB exceeded')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External guard terminated child')))
    try:
        guard()
        import fcntl
        lock=d.path(s['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch as torch_module
        torch=torch_module
        import random,copy
        from types import SimpleNamespace
        from geobc.prospective_dataset_v2 import load_case
        from geobc.prospective_dataset import resolve_case
        from geobc.macro_training import build_solver,encode_case,query_case,balanced_schedule,profile
        from geobc.train_physical_cache_v12 import full_physical_backward
        from geobc.physical_field_loss import PhysicalScale,norms_from_complete_physical_target
        from geobc.fno_stress_specialist_v54 import tree_digest
        from geobc.coverage_comparison_train import physical_metrics,aggregate
        from geobc.benchmark import FieldScaler
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device=torch.cuda.current_device();total=torch.cuda.get_device_properties(device).total_memory
        assert s['memory_limit_bytes']<total
        torch.cuda.set_per_process_memory_fraction(s['memory_limit_bytes']/total,device);torch.cuda.reset_peak_memory_stats()
        random.seed(s['seed']);np.random.seed(s['seed']);torch.manual_seed(s['seed']);torch.cuda.manual_seed_all(s['seed'])
        cases={};cache=SimpleNamespace(completed=False,physical_targets={},case_norms={})
        def event(stage,key,name,identity):
            guard();assert key in keys and name not in ('target','physical')
            if stage=='attempt':a['input_array_attempts']+=1
            if stage=='opened':a['input_array_reads']+=1
        for key in keys:
            guard();cases[key]=load_case(m,key,with_targets=False,train_CAD_count=100,device='cuda',for_fno=True,on_read=event)
            _,ids=resolve_case(m,key,True,100,True);item=ids['physical'];d.stat_matches(item)
            a['physical_TRAIN_attempts']+=1;value=np.load(d.path(item['path']),mmap_mode='r',allow_pickle=False)
            a['physical_TRAIN_reads']+=1
            assert value.shape==(len(cases[key]['xyz']),4) and str(value.dtype)=='float32' and np.isfinite(value).all()
            cache.physical_targets[key]=torch.as_tensor(np.array(value,copy=True),device='cuda')
            cache.case_norms[key]=norms_from_complete_physical_target(cache.physical_targets[key]);del value
            guard();persist()
        assert a['physical_TRAIN_attempts']==a['physical_TRAIN_reads']==3
        cache.completed=True
        scale=PhysicalScale.from_mapping(s['scaler']);scaler=FieldScaler(**s['scaler']);recipe=s['optimizer']
        model=build_solver('FNO9',seed=s['seed']);a['model_profile']=profile(model)
        assert a['model_profile']['parameter_count']==4526858
        a['fields']={};a['previous_shared0to20_cost']=up['summary']['previous_shared0to20_cost']
        a['previous_diagnostic_cost']=dict(stage_epoch_seconds=up['external']['elapsed_epoch_seconds'],stage_monotonic_seconds=up['external']['elapsed_monotonic_seconds'],terminal_binding=up['binding'],selected_prior_independent_segments={k:up['summary']['arms']['independent9'][k]['segment_epoch_seconds']for k in keys})
        def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
        def restore(state):
            random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
        def readout(selected):
            state=rng();model.eval();rows={}
            try:
                with torch.no_grad():
                    for key in selected:
                        guard();index=s['parent_TRAIN_keys'].index(key)
                        torch.manual_seed(2026+index);torch.cuda.manual_seed_all(2026+index)
                        case=cases[key];context,latent=encode_case(model,case);n=len(case['xyz']);pred=np.empty((n,4),dtype=np.float32)
                        for start in range(0,n,32768):
                            guard();stop=min(start+32768,n);pred[start:stop]=query_case(model,case,start,stop,context,latent).cpu().numpy()
                        physical=scaler.decode(pred);assert np.isfinite(physical).all()
                        truth=cache.physical_targets[key].detach().cpu().numpy();a['resident_physical_target_reuses']+=1
                        rows[key]=physical_metrics(np,truth,physical);del context,latent,pred,physical,truth
            finally:restore(state);model.train()
            return aggregate(rows,len(selected))
        a['cache_setup_epoch_seconds']=time.time()-ENTRY_EPOCH;a['cache_setup_monotonic_seconds']=time.monotonic()-ENTRY_MONOTONIC
        a['training_entry_epoch']=time.time();a['training_entry_monotonic']=time.monotonic();a['phase']='single_field_learning'
        phase_deadline=(a['training_entry_epoch']+600,a['training_entry_monotonic']+600);persist()
        with (o/'loss.jsonl').open('x',encoding='utf-8')as log:
            for key in keys:
                guard();began=(time.time(),time.monotonic());ck=s['parent_checkpoints'][key]
                assert d.identity(ck['path'],hashed=True)==ck;a['parent_checkpoint_hash_reads']+=1
                saved=torch.load(d.path(ck['path']),map_location='cpu',weights_only=False);a['parent_checkpoint_loads']+=1
                assert saved['family']=='FNO9' and saved['scientific_digest']==s['parent_training_SCI']
                assert saved['optimizer_step']==6100 and saved['additional_updates']==100 and saved['diagnostic_selected_keys']==[key]
                assert saved['dataset_identity']==s['dataset_identity'] and saved['scaler']==s['scaler'] and not saved['shared_fullTRAIN_model']
                assert saved['case_exposure_counts']=={k:(120 if k==key else 20)for k in s['parent_TRAIN_keys']}
                assert set(saved['rng'])=={'python','numpy','torch','cuda'} and all(int(v['step'])==6100 for v in saved['optimizer']['state'].values())
                model.load_state_dict(saved['model'],strict=True);model.train()
                optimizer=torch.optim.Adam(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'],betas=tuple(recipe['betas']),eps=recipe['eps'],fused=recipe['fused'])
                optimizer.load_state_dict(copy.deepcopy(saved['optimizer']));restore(saved['rng'])
                md=tree_digest(saved['model']);ad=tree_digest(saved['optimizer']);rd=tree_digest(saved['rng'])
                assert tree_digest(model.state_dict())==md and tree_digest(optimizer.state_dict())==ad and tree_digest(rng())==rd
                state=dict(status='running',parent_checkpoint=ck,parent_model_digest=md,parent_Adam_digest=ad,parent_RNG_digest=rd,restored_model_Adam_fourRNG=True,
                    parent_optimizer_step=6100,optimizer_step=6100,additional_updates=0,total_additional_updates_since_shared20=100,
                    case_exposure_counts=dict(saved['case_exposure_counts']),endpoints={})
                a['fields'][key]=state;a['current_field']=key;persist()
                optimizer.zero_grad(set_to_none=True);_,leaves=full_physical_backward(model,cases[key],cache,scale,[key],guard)
                groups=dict(encoder=0.,operator=0.,decoder=0.)
                for name,param in model.named_parameters():
                    assert param.grad is not None and torch.isfinite(param.grad).all()
                    group='encoder' if name.startswith('encoder.') else 'operator' if name.startswith('operator.') else 'decoder'
                    groups[group]+=float(param.grad.detach().double().square().sum())
                assert all(v>0 for v in groups.values()) and leaves['geometry_context_L2']>0 and leaves['operator_latent_L2']>0
                optimizer.zero_grad(set_to_none=True);restore(saved['rng'])
                assert tree_digest(model.state_dict())==md and tree_digest(optimizer.state_dict())==ad and tree_digest(rng())==rd
                state['actual_fullN_gradient_check']=dict(passed=True,key=key,nodes=len(cases[key]['xyz']),groups_squared_L2=groups,leaves=leaves,model_unchanged=True,Adam_unchanged=True,fourRNG_restored=True,optimizer_updates=0)
                baseline=readout([key]);state['endpoints']['100']=baseline
                for field in ('u_relative_l2','stress_relative_l2','stress_hotspot_relative_l2'):
                    assert abs(baseline['per_case'][key][field]-s['parent_scalar_rows'][key][field])<=1e-6,(key,field,'Own100 checkpoint replay differs')
                state['baseline_parent_scalar_match']=True
                def save_endpoint(total_additional):
                    guard();assert all(int(v['step'])==state['optimizer_step']for v in optimizer.state.values())
                    file=o/'checkpoints'/(key.replace('/','_')+'_extra'+str(total_additional)+'.pt');file.parent.mkdir(parents=True,exist_ok=True);temp=file.with_suffix('.tmp')
                    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng(),scientific_digest=p['scientific_digest'],family='FNO9',parent_checkpoint=ck,parent_training_SCI=s['parent_training_SCI'],optimizer_step=state['optimizer_step'],additional_updates_this_stage=state['additional_updates'],total_additional_updates_since_shared20=total_additional,case_exposure_counts=state['case_exposure_counts'],diagnostic_selected_keys=[key],dataset_identity=s['dataset_identity'],scaler=s['scaler'],shared_fullTRAIN_model=False),temp);os.replace(temp,file)
                    state.setdefault('checkpoint_identities',{})[str(total_additional)]=d.identity(file,hashed=True)
                for index in range(1,901):
                    guard();optimizer.zero_grad(set_to_none=True)
                    loss,_=full_physical_backward(model,cases[key],cache,scale,[key],guard)
                    assert all(param.grad is not None and torch.isfinite(param.grad).all()for param in model.parameters())
                    a['optimizer_step_attempts']+=1;a['optimizer_step_in_progress']=True
                    optimizer.step();torch.cuda.synchronize();a['optimizer_step_completions']+=1;a['optimizer_step_in_progress']=False
                    state.update(additional_updates=index,optimizer_step=6100+index,total_additional_updates_since_shared20=100+index)
                    state['case_exposure_counts'][key]=120+index
                    log.write(json.dumps(dict(key=key,additional_update=index,total_additional_since_shared20=100+index,physical_joint_squared=loss,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH))+'\n');log.flush();guard()
                    if index in (200,900):
                        state['endpoints'][str(100+index)]=readout([key]);save_endpoint(100+index)
                    if index%50==0:persist()
                assert state['case_exposure_counts']=={k:(1020 if k==key else 20)for k in s['parent_TRAIN_keys']}
                assert tree_digest(saved['model'])==md and tree_digest(saved['optimizer'])==ad and tree_digest(saved['rng'])==rd
                d.stat_matches(ck);state.update(status='completed',parent_checkpoint_unchanged=True,
                    segment_epoch_seconds=time.time()-began[0],segment_monotonic_seconds=time.monotonic()-began[1]);guard();persist();del saved
        assert a['optimizer_step_attempts']==a['optimizer_step_completions']==2700
        a['aggregate_endpoints']={str(step):aggregate({k:a['fields'][k]['endpoints'][str(step)]['per_case'][k]for k in keys},3)for step in (100,300,1000)}
        assert all(x['all_cases_complete']for x in a['aggregate_endpoints'].values())
        a.update(status='completed',phase='terminal',training_complete=True,fitting_diagnostic_only=True,automatic_continuation=False,
            training_epoch_seconds=time.time()-a['training_entry_epoch'],training_monotonic_seconds=time.monotonic()-a['training_entry_monotonic'])
    except Exception as error:
        a.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error),possible_partial_optimizer_update=a['optimizer_step_in_progress'])
        raise
    finally:
        a.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(o/'summary.json',a);d.write(o/'status.json',a)
        if lock is not None:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');parser.add_argument('--plan');args=parser.parse_args()
    if args.start:
        if not args.plan:parser.error('--start requires frozen released plan')
        run(args.plan)
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
