"""Fresh shared FNO and global Hash on three fixed fields with matched case exposure."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
import os
import json
import signal
from pathlib import Path


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.shared_decoder_pair_contract_v61 import approved_contract
    p,r,m,up=approved_contract(plan_path);s=p['scientific'];keys=s['selected_TRAIN_keys']
    outer=json.loads(os.environ['GEOBC_SHARED_DECODER_V61_GUARD'])
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
        from geobc.operator_hash_metadata_v6 import build_solver,encode_case,query_case,profile
        from geobc.physical_field_loss import full_common_case_backward
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
        a['models']={};a['family_profiles']={};a['model_initialization']={};a['prior_diagnostic_cost_reference']=dict(terminal_binding=up['binding'],stage_epoch_seconds=up['external']['elapsed_epoch_seconds'],not_in_fresh_training_comparison=True)
        maps={};backbone_reference=None
        def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
        def restore(state):
            random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
        def backward(key):
            assert cases[key]['scientific_group']=='TRAIN' and key in keys and model.macro_family in s['families']
            return full_common_case_backward(model,cases[key],cache.physical_targets[key],scale,cache.case_norms[key],guard)
        def readout(key,step,family):
            state=rng();model.eval()
            try:
                with torch.no_grad():
                    guard();index=s['parent_TRAIN_keys'].index(key)
                    torch.manual_seed(2026+index);torch.cuda.manual_seed_all(2026+index)
                    case=cases[key];context,latent=encode_case(model,case);n=len(case['xyz']);pred=np.empty((n,4),dtype=np.float32)
                    for start in range(0,n,32768):
                        guard();stop=min(start+32768,n);pred[start:stop]=query_case(model,case,start,stop,context,latent).cpu().numpy()
                    physical=scaler.decode(pred);assert np.isfinite(physical).all()
                    truth=cache.physical_targets[key].detach().cpu().numpy();a['resident_physical_target_reuses']+=1
                    metric=aggregate({key:physical_metrics(np,truth,physical)},1)
                    if step==1000 and key==s['representative_field']:
                        idx=np.linspace(0,n-1,min(n,s['visualization_sample_nodes_max']),dtype=np.int64)
                        points=case['xyz'].detach().cpu().numpy()[idx].copy()
                        if maps:assert np.array_equal(points,maps['xyz']) and np.array_equal(truth[idx,3],maps['truth'])
                        maps['xyz']=points;maps['truth']=truth[idx,3].copy();maps[family]=physical[idx,3].copy()
                        maps['original_nodes']=n;maps['sample_nodes']=len(idx)
                    del context,latent,pred,physical,truth
                    return metric
            finally:restore(state);model.train()
        guard();a['cache_setup_epoch_seconds']=time.time()-ENTRY_EPOCH;a['cache_setup_monotonic_seconds']=time.monotonic()-ENTRY_MONOTONIC
        a['training_entry_epoch']=time.time();a['training_entry_monotonic']=time.monotonic();a['phase']='shared_decoder_pair'
        phase_deadline=(a['training_entry_epoch']+900,a['training_entry_monotonic']+900);persist()
        from geobc.macro_training import balanced_schedule
        schedule=balanced_schedule(keys,1000,2026);assert d.digest(schedule)==s['full_order_digest']
        with (o/'loss.jsonl').open('x',encoding='utf-8')as log:
            for family in s['families']:
                guard();began=(time.time(),time.monotonic());random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
                model=build_solver(family,seed=2026).train();profile_value=profile(model);a['family_profiles'][family]=profile_value
                assert profile_value['parameter_count']==s['parameter_counts'][family] and not profile_value['actual_CPU_fixture_only']
                if family=='FNO_GLOBAL_HASH':assert model.backend=='tcnn_fully_fused'
                initial={name:value.detach().cpu().clone()for name,value in model.state_dict().items()};initial_rng=rng()
                md=tree_digest(initial);rd=tree_digest(initial_rng)
                backbone={group:tree_digest(getattr(model,group).state_dict())for group in ('encoder','operator')}
                if backbone_reference is None:backbone_reference=backbone
                assert backbone==backbone_reference,'Actual fresh global backbone bytes differ'
                expected=s['expected_actual_initialization'][family]
                assert md==expected['model_digest'] and rd==expected['RNG_digest'] and backbone==expected['shared_backbone_digests']
                a['model_initialization'][family]=dict(model_digest=md,RNG_digest=rd,shared_backbone_digests=backbone,backbone_identity_passed=True,full_model_same_across_families=False,ownv60_exact_initial_match=True)
                optimizer=torch.optim.Adam(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'],betas=tuple(recipe['betas']),eps=recipe['eps'],fused=recipe['fused']);restore(initial_rng)
                assert not optimizer.state and tree_digest(model.state_dict())==md and tree_digest(rng())==rd
                state=dict(status='running',optimizer_step=0,case_exposure_counts={key:0 for key in keys},fresh_model_digest=md,fresh_RNG_digest=rd,fresh_Adam0=True,endpoints={},gradient_checks={})
                a['models'][family]=state;a['current_model']=dict(family=family,shared_keys=keys);persist()
                ad=tree_digest(optimizer.state_dict())
                for key in keys:
                    guard();optimizer.zero_grad(set_to_none=True);_,leaves=backward(key);groups=dict(encoder=0.,operator=0.,decoder=0.)
                    for name,param in model.named_parameters():
                        assert param.grad is not None and torch.isfinite(param.grad).all()
                        group='encoder'if name.startswith('encoder.')else'operator'if name.startswith('operator.')else'decoder'
                        groups[group]+=float(param.grad.detach().double().square().sum())
                    assert all(v>0 for v in groups.values()) and leaves['context']>0 and leaves['latent']>0
                    optimizer.zero_grad(set_to_none=True);restore(initial_rng)
                    assert tree_digest(model.state_dict())==md and tree_digest(optimizer.state_dict())==ad and tree_digest(rng())==rd
                    state['gradient_checks'][key]=dict(passed=True,key=key,nodes=len(cases[key]['xyz']),groups_squared_L2=groups,leaves=leaves,model_unchanged=True,Adam_unchanged=True,fourRNG_restored=True,optimizer_updates=0)
                def endpoint(passes):
                    rows={key:readout(key,passes,family)['per_case'][key]for key in keys}
                    return aggregate(rows,3)
                state['endpoints']['0']=endpoint(0)
                reference=s['independent_endpoint_reference'][family]['0']['per_case']
                assert all(abs(state['endpoints']['0']['per_case'][key][metric]-reference[key][metric])<1e-6 for key in keys for metric in ('u_relative_l2','stress_relative_l2','stress_hotspot_relative_l2'))
                state['initial_full3_scalar_match']=True
                for index,key in enumerate(schedule,1):
                    guard();optimizer.zero_grad(set_to_none=True);loss,_=backward(key)
                    assert all(param.grad is not None and torch.isfinite(param.grad).all()for param in model.parameters())
                    a['optimizer_step_attempts']+=1;a['optimizer_step_in_progress']=True
                    optimizer.step();torch.cuda.synchronize();a['optimizer_step_completions']+=1;a['optimizer_step_in_progress']=False
                    state['optimizer_step']=index;state['case_exposure_counts'][key]+=1
                    log.write(json.dumps(dict(family=family,key=key,step=index,physical_joint_squared=loss,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH))+'\n');log.flush();guard()
                    if index in (300,3000):
                        passes=index//3;assert state['case_exposure_counts']=={k:passes for k in keys}
                        state['endpoints'][str(passes)]=endpoint(passes)
                    if index%50==0:persist()
                assert all(int(v['step'])==3000 for v in optimizer.state.values()) and state['case_exposure_counts']=={key:1000 for key in keys}
                file=o/'checkpoints'/(family+'.pt');file.parent.mkdir(parents=True,exist_ok=True);temporary=file.with_suffix('.tmp')
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng(),scientific_digest=p['scientific_digest'],family=family,optimizer_step=3000,fresh_initial_model_digest=md,case_exposure_counts=state['case_exposure_counts'],diagnostic_selected_keys=keys,dataset_identity=s['dataset_identity'],scaler=s['scaler'],order_digest=s['full_order_digest'],shared_fullTRAIN_model=False),temporary);os.replace(temporary,file)
                state.update(status='completed',segment_epoch_seconds=time.time()-began[0],segment_monotonic_seconds=time.monotonic()-began[1],checkpoint_identity=d.identity(file,hashed=True));guard();persist()
                assert tree_digest(initial)==md and tree_digest(initial_rng)==rd
                del optimizer,model,initial,initial_rng;torch.cuda.empty_cache();guard()
        assert a['optimizer_step_attempts']==a['optimizer_step_completions']==6000
        a['aggregate_endpoints']={family:a['models'][family]['endpoints']for family in s['families']}
        assert all(metric['all_cases_complete']for endpoint in a['aggregate_endpoints'].values()for metric in endpoint.values())
        a['learning_interval_epoch_seconds']=time.time()-a['training_entry_epoch'];a['learning_interval_monotonic_seconds']=time.monotonic()-a['training_entry_monotonic']
        assert all(f in maps for f in s['families']);guard();phase_deadline=None;a['phase']='representative_field_plot';persist();guard()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        vmax=float(np.quantile(np.abs(maps['truth']),.99));assert vmax>0
        fig=plt.figure(figsize=(11.0,4.0));axs=[fig.add_subplot(1,3,i+1,projection='3d')for i in range(3)]
        for ax,label,name in zip(axs,['FEM signed stress','FNO MLP decoder','FNO global Hash decoder'],['truth']+s['families']):
            artist=ax.scatter(*maps['xyz'].T,c=maps[name],s=1.2,cmap='coolwarm',vmin=-vmax,vmax=vmax,rasterized=True)
            ax.view_init(elev=25,azim=40);ax.set_box_aspect(np.maximum(np.ptp(maps['xyz'],axis=0),1e-6));ax.set_axis_off();ax.set_title(label,fontsize=10)
        fig.suptitle(s['representative_field']+' / same native nodes and camera',fontsize=11)
        fig.colorbar(artist,ax=axs,shrink=.65,pad=.02,label='Signed stress (MPa), truth sample P99 range',extend='both');figure=o/'representative_stress.png';fig.savefig(figure,dpi=180,bbox_inches='tight');plt.close(fig);guard()
        a['representative_field_plot']=dict(identity=d.identity(figure,hashed=True),key=s['representative_field'],original_nodes=maps['original_nodes'],display_sample_nodes=maps['sample_nodes'],shared_symmetric_stress_range_MPa=[-vmax,vmax],visual_range_rule='Truth display sample absoluteP99; outside range clipped for display only',display_clipped_fractions={k:float((np.abs(maps[k])>vmax).mean())for k in ['truth']+s['families']},metrics_all_original_nodes=True,original_stress_sign_unchanged=True,no_additional_GT_or_inference=True)
        a.update(status='completed',phase='terminal',training_complete=True,fitting_diagnostic_only=True,automatic_continuation=False)
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
