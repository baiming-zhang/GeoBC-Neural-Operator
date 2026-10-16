"""Resume saved ++ update21 with one active GPU case and unchanged training math."""
import argparse,copy,gc,json,os,random,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from geobc.overfit_models import make_model,backward_case,predict_case
from geobc.overfit_transformers import close_owned_process_group
from geobc.runtime import GPUQueue
from tools.train_sample_overfitting import errors,make_scheduler,read_training_source,write


def restore(model,optimizer,scheduler,payload,config):
    if payload['config']!=config or payload['order']!=config['fifty'] or payload['step']!=21 or payload['cursor']!=21 or payload['arm'].get('partial_backward') or payload['arm'].get('optimizer_step_in_progress'):raise ValueError('Exact saved complete-step21 state required')
    model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
    if scheduler.last_epoch!=21:raise ValueError('Scheduler must continue at21 without restart')
    random.setstate(payload['rng']['python']);np.random.set_state(payload['rng']['numpy']);torch.set_rng_state(payload['rng']['torch'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in payload['rng']['cuda']])


def run(args):
    c=json.loads(args.config.read_text());sources=json.loads(args.source.read_text());clock=json.loads(os.environ['GEOBC_EXPERIMENT_CLOCK'])
    if c.get('early_stopping') or not c['fixed_endpoint'] or c['max_updates_per_arm']!=5000:raise ValueError('Original fixed5000 recipe required')
    os.environ['GEOBC_TRANSFORMER_MANIFEST']=sources['transformer_manifest']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.cuda.set_per_process_memory_fraction(c['memory_bytes']/torch.cuda.get_device_properties(0).total_memory)
    step=21;model=optimizer=scheduler=None;arm={};cache={};last=None
    state=dict(status='running',active=True,closed=False,phase='loading_CPU_cases',arms={},source_reads=0,stress_label_reads=0,heldout_reads=0,training_only=True,checkpoint_reads=0,continued_from='fifty_u_overfitting/fifty_transolverpp/last.pt',starting_step=21,scientific_training_recipe_changed=False,data_residency='CPU shared cache; exactly one full original-node case on GPU; no grid for ++; release allocator cache between cases')
    def update(**kw):state.update(kw);state['elapsed_seconds']=dict(UTC=time.time()-clock['entry_epoch'],monotonic=time.monotonic()-clock['entry_monotonic']);write(args.output/'status.json',state)
    def guard(training=False):
        reserve=60 if training else 0
        if min(clock['deadline_epoch']-time.time(),clock['deadline_monotonic']-time.monotonic())<=reserve:raise TimeoutError('Inherited original arm/whole deadline; no extension')
        free,total=torch.cuda.mem_get_info()
        if total-free>c['memory_bytes']:raise MemoryError('Declared GPU memory cap')
    def release():gc.collect();torch.cuda.empty_cache()
    def case(key):
        a=cache[key];x={k:v.cuda() for k,v in a.items()};x['key']=key
        x['inputs']=torch.cat((x['xyz'],x['features']),1);x['target']=x['physical']/scale
        x['energy']=x['target'].square().sum(1).mean()[None].clamp_min(1e-20);return x
    def checkpoint():
        if model is None or optimizer is None or scheduler is None:return
        arm['case_updates']={key:step//50+int(i<step%50) for i,key in enumerate(c['fifty'])}
        elapsed=dict(UTC=time.time()-clock['entry_epoch'],monotonic=time.monotonic()-clock['entry_monotonic'])
        payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,order=c['fifty'],cursor=step%50,config=c,scale4=scale.cpu(),rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate()),arm=copy.deepcopy(arm),clock=clock,elapsed_seconds=elapsed,prior_failed_arm_seconds=prior_seconds)
        temp=args.output/'last.pt.tmp';torch.save(payload,temp);temp.replace(args.output/'last.pt')
    def evaluate():
        nonlocal last
        model.eval();metrics={}
        for key in c['fifty']:
            guard();item=case(key)
            with torch.no_grad():pred=predict_case('transolverpp',model,item,guard)*scale
            metrics[key]=errors(item['physical'],pred)
            if step==5000:np.savez(args.output/(key.replace('/','_')+'_endpoint.npz'),prediction=pred.cpu().numpy(),sample_key=key,step=step)
            del pred,item;release()
        values=[r['U_percent'] for r in metrics.values()]
        last=dict(step=step,per_case=metrics,mean_U_percent=float(np.mean(values)),worst_field_percent=max(values),learning_rate_last_update=arm['learning_rate_last_update'])
        arm['latest']=last
        if step==5000:arm['endpoint']=last;arm['endpoint_predictions_saved']=50
        with (args.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(last)+'\n')
        update(phase='fifty_transolverpp/training')
    def interrupt(*_):raise KeyboardInterrupt('Guard termination; preserve current actual state')
    signal.signal(signal.SIGTERM,interrupt)
    prior_seconds=json.loads((args.original/'fifty_transolverpp/summary.json').read_text())['elapsed_seconds']
    with GPUQueue(args.lock):
        try:
            update()
            for key in c['fifty']:
                guard();xyz,features,u=read_training_source(sources['cases'][key],key,'u')
                if xyz.shape!=(len(u),3) or features.shape!=(len(u),25) or u.shape!=(len(u),3) or not all(np.isfinite(a).all() for a in (xyz,features,u)):raise ValueError('Malformed aligned TRAIN case')
                cache[key]=dict(xyz=torch.from_numpy(xyz),features=torch.from_numpy(features),physical=torch.from_numpy(u));state['source_reads']+=1
            saved=torch.load(args.original/'fifty_transolverpp/last.pt',map_location='cpu',weights_only=False);state['checkpoint_reads']=1
            scale=saved['scale4'].cuda()
            model=make_model('transolverpp',c['seed'],c['model_settings']['transolverpp'],out_channels=3)
            optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],fused=True);scheduler=make_scheduler(optimizer,c)
            restore(model,optimizer,scheduler,saved,c)
            arm=copy.deepcopy(saved['arm']);arm.update(status='running',recovered_from_actual_step=21,added_updates=0,model_Adam_scheduler_fourRNG_cursor_restored=True,error=None,error_type=None)
            state['arms']['fifty_transolverpp']=arm;state['prior_failed_arm_seconds']=prior_seconds;state['restored_without_readout_or_RNG_draw']=True
            del saved;release();update(phase='fifty_transolverpp/training')
            while step<5000:
                guard(True);model.train();optimizer.zero_grad(set_to_none=True);item=case(c['fifty'][step%50]);arm['partial_backward']=True
                loss=backward_case('transolverpp',model,item,lambda:guard(True))
                if step==21:
                    gradients=[p.grad for p in model.parameters() if p.grad is not None]
                    if not gradients or not all(torch.isfinite(g).all() for g in gradients):raise FloatingPointError('Nonfinite restored full-node backward')
                    arm['resumed_full_node_backward_finite']=True
                    if sum(p.numel() for p in model.parameters() if p.requires_grad and p.grad is not None)!=arm['active_parameters']:raise ValueError('Active architecture changed')
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip_norm'],error_if_nonfinite=True,foreach=True)
                arm['gradient_norm_before_clip']=float(norm);arm['gradient_clipped_updates']=arm.get('gradient_clipped_updates',0)+int(norm>c['gradient_clip_norm'])
                arm['learning_rate_last_update']=optimizer.param_groups[0]['lr'];arm['optimizer_step_in_progress']=True
                optimizer.step();step+=1;scheduler.step();arm.update(updates=step,added_updates=step-21,partial_backward=False,optimizer_step_in_progress=False,training_loss=loss)
                optimizer.zero_grad(set_to_none=True);del item;release()
                if step%10==0:update()
                if step%50==0:evaluate();checkpoint()
            arm.update(status='completed',stop_reason='fixed_update_endpoint',goal_met=last['worst_field_percent']<1)
            state.update(status='completed',active=False,closed=True,phase='terminal')
        except BaseException as exc:
            arm.update(status='failed',error_type=type(exc).__name__,error=str(exc));state.update(status='failed',active=False,closed=True,phase='terminal');traceback.print_exc();raise
        finally:
            checkpoint()
            if arm:
                resumed=dict(UTC=time.time()-clock['entry_epoch'],monotonic=time.monotonic()-clock['entry_monotonic'])
                arm['resumed_elapsed_seconds']=resumed;arm['elapsed_seconds']={k:prior_seconds[k]+resumed[k] for k in resumed};write(args.output/'arm_summary.json',arm)
            update();write(args.output/'summary.json',state);close_owned_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);run(p.parse_args())
