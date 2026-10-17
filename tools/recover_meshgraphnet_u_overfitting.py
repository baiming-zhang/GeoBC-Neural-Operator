"""Resume saved MGN step4; CPU cache, one complete original mesh on GPU."""
import argparse,copy,gc,json,os,random,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from geobc.overfit_meshgraphnet import MeshGraphNet,MeshGraphNetConfig
from geobc.overfit_models import relative_loss
from geobc.runtime import GPUQueue
from tools.train_sample_overfitting import errors,make_scheduler,read_training_source,write


def restore(model,optimizer,scheduler,payload,config,restore_cuda=True):
    if payload['config']!=config or payload['step']!=4 or payload['order']!=config['five'] or payload['cursor']!=4 or payload['arm'].get('optimizer_step_in_progress'):raise ValueError('Exact completed-step4 optimizer boundary required')
    model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
    if scheduler.last_epoch!=4:raise ValueError('OneCycle cursor changed')
    random.setstate(payload['rng']['python']);np.random.set_state(payload['rng']['numpy']);torch.set_rng_state(payload['rng']['torch'].cpu())
    if restore_cuda:torch.cuda.set_rng_state_all([v.cpu() for v in payload['rng']['cuda']])


def run(a):
    c=json.loads(a.config.read_text());sources=json.loads(a.source.read_text());clock=json.loads(os.environ['GEOBC_EXPERIMENT_CLOCK']);keys=c['five']
    prior=json.loads((a.original/'five_meshgraphnet/summary.json').read_text())['elapsed_seconds']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=c['allow_tf32'];torch.backends.cudnn.allow_tf32=c['allow_tf32']
    torch.cuda.set_per_process_memory_fraction(c['memory_bytes']/torch.cuda.get_device_properties(0).total_memory)
    step=4;model=optimizer=scheduler=None;arm={};cache={};scale=None;last=None
    state=dict(status='running',active=True,closed=False,phase='CPU_loading',arms={},source_reads=0,graph_reads=0,stress_label_reads=0,heldout_reads=0,checkpoint_reads=0,starting_step=4,
        data_residency='CPU shared cache, exactly one complete mesh on GPU, allocator released between cases',scientific_training_recipe_changed=False,
        failure_boundary='post-forward guard, before backward/optimizer; incomplete forward discarded, next update5',prior_failed_arm_seconds=prior)
    def update(**kw):
        state.update(kw);state['elapsed_seconds']=dict(UTC=time.time()-clock['entry_epoch'],monotonic=time.monotonic()-clock['entry_monotonic']);write(a.output/'status.json',state)
    def guard(training=False):
        if min(clock['deadline_epoch']-time.time(),clock['deadline_monotonic']-time.monotonic())<=(60 if training else 0):raise TimeoutError('Original arm/whole deadline, no extension')
        free,total=torch.cuda.mem_get_info()
        if total-free>c['memory_bytes']:raise MemoryError('Declared GPU memory cap')
    def release():gc.collect();torch.cuda.empty_cache()
    def case(key):
        item={k:v.cuda() for k,v in cache[key].items()};item['inputs']=torch.cat((item['xyz'],item['features']),1);item['target']=item['physical']/scale
        item['energy']=item['target'].square().sum(1).mean()[None].clamp_min(1e-20)
        relative=item['xyz'][item['senders']]-item['xyz'][item['receivers']];item['edge_features']=torch.cat((relative,torch.linalg.vector_norm(relative,dim=-1,keepdim=True)),1)
        return item
    def checkpoint():
        if model is None or optimizer is None or scheduler is None or scale is None:return
        arm['case_updates']={key:step//5+int(i<step%5) for i,key in enumerate(keys)}
        payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,order=keys,cursor=step%5,config=c,scale4=scale.cpu(),
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate()),arm=copy.deepcopy(arm),clock=clock,prior_failed_arm_seconds=prior)
        tmp=a.output/'last.pt.tmp';torch.save(payload,tmp);tmp.replace(a.output/'last.pt')
    def evaluate():
        nonlocal last
        model.eval();metrics={}
        for key in keys:
            guard();item=case(key)
            with torch.no_grad():pred=model(item['inputs'],item['edge_features'],item['senders'],item['receivers'])*scale
            guard();metrics[key]=errors(item['physical'],pred)
            if step==5000:np.savez(a.output/(key.replace('/','_')+'_endpoint.npz'),prediction=pred.cpu().numpy(),sample_key=key,step=step)
            del pred,item;release()
        values=[v['U_percent'] for v in metrics.values()];last=dict(step=step,per_case=metrics,mean_U_percent=float(np.mean(values)),worst_field_percent=max(values),learning_rate_last_update=arm['learning_rate_last_update']);arm['latest']=last
        if step==5000:arm['endpoint']=last;arm['endpoint_predictions_saved']=5
        with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(last)+'\n')
        update(phase='five_meshgraphnet/training')
    def interrupt(*_):raise KeyboardInterrupt('Preserve actual recovery state')
    signal.signal(signal.SIGTERM,interrupt)
    with GPUQueue(a.lock):
        try:
            update()
            for key in keys:
                guard();xyz,features,u=read_training_source(sources['cases'][key],key,'u');indices=np.load(sources['graphs'][key],allow_pickle=False)
                if xyz.shape!=(len(u),3) or features.shape!=(len(u),25) or u.shape!=(len(u),3) or not all(np.isfinite(v).all() for v in (xyz,features,u)) or indices.ndim!=2 or indices.shape[1]!=2 or indices.min()<0 or indices.max()>=len(u):raise ValueError('Aligned complete TRAIN graph required')
                cache[key]=dict(xyz=torch.from_numpy(xyz),features=torch.from_numpy(features),physical=torch.from_numpy(u),senders=torch.from_numpy(indices[:,0].copy()).long(),receivers=torch.from_numpy(indices[:,1].copy()).long());state['source_reads']+=1;state['graph_reads']+=1
            saved=torch.load(a.original/'five_meshgraphnet/last.pt',map_location='cpu',weights_only=False);state['checkpoint_reads']=1
            model=MeshGraphNet(MeshGraphNetConfig(**c['model_settings']['meshgraphnet'])).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],fused=True);scheduler=make_scheduler(optimizer,c);scale=saved['scale4'].cuda()
            restore(model,optimizer,scheduler,saved,c)
            arm=copy.deepcopy(saved['arm']);arm.update(status='running',partial_backward=False,recovered_from_actual_step=4,added_updates=0,error=None,error_type=None,model_Adam_scheduler_fourRNG_cursor_restored=True,discarded_incomplete_forward=True)
            state['arms']['five_meshgraphnet']=arm;state['restored_without_readout_or_RNG_draw']=True;del saved;release();update(phase='five_meshgraphnet/training')
            while step<5000:
                guard(True);model.train();optimizer.zero_grad(set_to_none=True);item=case(keys[step%5]);arm['partial_backward']=True
                pred=model(item['inputs'],item['edge_features'],item['senders'],item['receivers']);guard(True);loss=relative_loss(pred,item['target'],item['energy'],len(item['xyz']))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-graph loss')
                loss.backward();guard(True)
                if step==4:
                    grads=[p for p in model.parameters() if p.grad is not None]
                    if sum(p.numel() for p in grads)!=4603863 or not all(torch.isfinite(p.grad).all() for p in grads):raise ValueError('Restored full-node active finite gradient required')
                    arm['resumed_full_node_backward_finite']=True
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip_norm'],error_if_nonfinite=True,foreach=True)
                arm['gradient_norm_before_clip']=float(norm);arm['gradient_clipped_updates']=arm.get('gradient_clipped_updates',0)+int(norm>c['gradient_clip_norm']);arm['learning_rate_last_update']=optimizer.param_groups[0]['lr'];arm['optimizer_step_in_progress']=True
                optimizer.step();step+=1;scheduler.step();arm.update(updates=step,added_updates=step-4,partial_backward=False,optimizer_step_in_progress=False,training_loss=float(loss.detach()))
                optimizer.zero_grad(set_to_none=True);del pred,loss,item;release()
                if step%10==0:update()
                if step%50==0:evaluate();checkpoint()
            arm.update(status='completed',stop_reason='fixed_update_endpoint',goal_met=last['worst_field_percent']<1);state.update(status='completed',active=False,closed=True,phase='terminal')
        except BaseException as exc:
            arm.update(status='failed',error_type=type(exc).__name__,error=str(exc));state.update(status='failed',active=False,closed=True,phase='terminal');traceback.print_exc();raise
        finally:
            checkpoint()
            if arm:
                elapsed=dict(UTC=time.time()-clock['entry_epoch'],monotonic=time.monotonic()-clock['entry_monotonic']);arm['resumed_elapsed_seconds']=elapsed;arm['elapsed_seconds']={k:prior[k]+elapsed[k] for k in elapsed};write(a.output/'arm_summary.json',arm)
            update();write(a.output/'summary.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('original','config','source','output','lock'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
