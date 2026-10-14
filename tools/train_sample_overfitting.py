"""One- and five-case U-only or joint U/S original-node fitting."""
import time
import argparse,json,os,random,gc,traceback
from pathlib import Path
import numpy as np
import torch
from geobc.overfit_models import make_model,prepare_grid,backward_case,predict_case
from geobc.overfit_transformers import close_owned_process_group
from geobc.runtime import GPUQueue


def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def errors(truth,pred):
    truth=truth.double();pred=pred.double()
    if truth.shape!=pred.shape or truth.ndim!=2 or truth.shape[1] not in (3,4) or not torch.isfinite(truth).all() or not torch.isfinite(pred).all():
        raise ValueError('Invalid full-node prediction')
    result={'U_percent':float(100*torch.linalg.vector_norm(truth[:,:3]-pred[:,:3])/torch.linalg.vector_norm(truth[:,:3]).clamp_min(1e-20))}
    if truth.shape[1]==4:result['S_percent']=float(100*torch.linalg.vector_norm(truth[:,3]-pred[:,3])/torch.linalg.vector_norm(truth[:,3]).clamp_min(1e-20))
    return result


def read_case_arrays(f,key,output_field):
    """U-only deliberately never requests the stress member of an NPZ."""
    if str(f['label_alignment'].item())!='official CSV GRID coordinates -> H5 vertices' or str(f['sample_key'].item())!=key:
        raise ValueError('Unaligned original training-only fields')
    xyz=np.array(f['xyz'],dtype=np.float32);features=np.array(f['features'],dtype=np.float32)
    y=np.array(f['displacement'],dtype=np.float32)
    if output_field=='us':y=np.column_stack((y,f['stress'])).astype(np.float32)
    elif output_field!='u':raise ValueError('Unknown output field')
    return xyz,features,y


def run(args):
    config=json.loads(Path(args.config).read_text());sources=json.loads(Path(args.source).read_text())
    output_field=config.get('output_field','us')
    if output_field not in ('u','us'):raise ValueError('Unknown output field')
    channels=3 if output_field=='u' else 4
    output=Path(args.output);clock=json.loads(os.environ['GEOBC_EXPERIMENT_CLOCK'])
    os.environ['GEOBC_TRANSFORMER_MANIFEST']=sources['transformer_manifest']
    state={'status':'running','active':True,'closed':False,'phase':'loading','arms':{},'source_reads':0,'training_only':True,'heldout_reads':0,'output_field':output_field,'stress_label_reads':0}
    def update(**kw):
        state.update(kw);state['elapsed_seconds']={'UTC':time.time()-clock['entry_epoch'],'monotonic':time.monotonic()-clock['entry_monotonic']}
        write(output/'status.json',state)
    def guard():
        if time.time()>=clock['deadline_epoch']-30 or time.monotonic()>=clock['deadline_monotonic']-30:raise TimeoutError('Whole-stage deadline')
        free,total=torch.cuda.mem_get_info()
        if total-free>config['memory_bytes']:raise MemoryError('Declared GPU memory cap')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.cuda.set_per_process_memory_fraction(config['memory_bytes']/torch.cuda.get_device_properties(0).total_memory)
    with GPUQueue(args.lock):
        try:
            cache={}
            for key in config['five']:
                guard()
                with np.load(sources['cases'][key],allow_pickle=False) as f:
                    xyz,features,y=read_case_arrays(f,key,output_field)
                    state['stress_label_reads']+=int(output_field=='us')
                if xyz.shape!=(len(y),3) or features.shape!=(len(y),25) or y.shape!=(len(xyz),channels) or not all(np.isfinite(a).all() for a in (xyz,features,y)) or xyz.min()<0 or xyz.max()>1:
                    raise ValueError('Malformed overfitting case')
                case={'key':key,'xyz':torch.from_numpy(xyz).cuda(),'features':torch.from_numpy(features).cuda(),'physical':torch.from_numpy(y).cuda()}
                case['inputs']=torch.cat((case['xyz'],case['features']),1)
                prepare_grid(case);cache[key]=case;state['source_reads']+=1;update()
            groups=config.get('groups',['single','five'])
            for group in groups:
                cases=[cache[k] for k in config[group]]
                all_y=torch.cat([c['physical'].double() for c in cases])
                u_scale=all_y[:,:3].square().sum(1).mean().sqrt()
                scale=torch.stack((u_scale,)) if channels==3 else torch.stack((u_scale,all_y[:,3].square().mean().sqrt()))
                scale=scale.float().clamp_min(1e-12)
                scale4=scale[[0,0,0]] if channels==3 else scale[[0,0,0,1]];del all_y
                for c in cases:
                    c['target']=c['physical']/scale4
                    u_energy=c['target'][:,:3].square().sum(1).mean()
                    c['energy']=(torch.stack((u_energy,)) if channels==3 else torch.stack((u_energy,c['target'][:,3].square().mean()))).clamp_min(1e-20)
                for method in config['methods']:
                    guard();name=group+'_'+method;folder=output/name;folder.mkdir()
                    arm={'status':'running','method':method,'group':group,'output_field':output_field,'output_channels':channels,'updates':0,'best':None,'best_step':None,'goal_met':False,'partial_backward':False,'cases':config[group]}
                    state['arms'][name]=arm;update(phase=name+'/initializing')
                    start=(time.time(),time.monotonic());model=None;optimizer=None;scheduler=None;step=0;last=None
                    def arm_guard():
                        guard()
                        if max(time.time()-start[0],time.monotonic()-start[1])>=config['arm_seconds']-60:raise TimeoutError('Arm training interval closed; retain save/readout reserve')
                    def checkpoint(filename):
                        if model is None:return
                        payload={'model':model.state_dict(),'optimizer':optimizer.state_dict() if optimizer else None,'scheduler':scheduler.state_dict() if scheduler else None,'step':step,'order':config[group],'cursor':step%len(cases),'config':config,'scale4':scale4.cpu(),
                                 'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate()},'arm':arm.copy()}
                        temp=folder/(filename+'.tmp');torch.save(payload,temp);temp.replace(folder/filename)
                    def evaluate():
                        nonlocal last
                        model.eval();predictions={};metrics={}
                        for c in cases:
                            guard();pred=predict_case(method,model,c,guard)*scale4
                            predictions[c['key']]=pred;metrics[c['key']]=errors(c['physical'],pred)
                        worst=max(value for v in metrics.values() for value in v.values())
                        last={'step':step,'per_case':metrics,'worst_field_percent':worst,'mean_U_percent':float(np.mean([m['U_percent'] for m in metrics.values()]))}
                        if channels==4:last['mean_S_percent']=float(np.mean([m['S_percent'] for m in metrics.values()]))
                        if config.get('fixed_endpoint') and step==config['max_updates_per_arm']:
                            for c in cases:np.savez(folder/(c['key'].replace('/','_')+'_endpoint.npz'),prediction=predictions[c['key']].cpu().numpy(),sample_key=c['key'],step=step)
                            arm['endpoint_predictions_saved']=len(cases)
                        arm['latest']=last
                        if arm['best'] is None or worst<arm['best']['worst_field_percent']:
                            arm['best']=last;arm['best_step']=step;checkpoint('best.pt')
                            for c in cases:
                                np.savez(folder/(c['key'].replace('/','_')+'_best.npz'),prediction=predictions[c['key']].cpu().numpy(),sample_key=c['key'])
                        arm['goal_met']=arm['best']['worst_field_percent']<config['target_relative_l2']*100
                        with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(last)+'\n')
                        update(phase=name+'/training');return worst
                    try:
                        model=make_model(method,config['seed'],config.get('model_settings',{}).get(method),out_channels=channels);arm['parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
                        if 'parameter_target' in config and abs(arm['parameters']/config['parameter_target']-1)>config['parameter_tolerance']:
                            raise ValueError('Actual trainable parameter count outside matching tolerance')
                        arm['model_config']=getattr(model,'source_config',{})
                        random.seed(config['seed']);np.random.seed(config['seed'])
                        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],fused=True)
                        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=config['lr_plateau_evaluations'],min_lr=config['minimum_learning_rate'])
                        if config.get('continuation'):
                            saved=torch.load(sources['continuations'][name],map_location='cuda',weights_only=False)
                            if saved['order']!=config[group] or saved['cursor']!=saved['step']%len(cases) or saved['arm'].get('optimizer_step_in_progress'):
                                raise ValueError('Continuation requires complete optimizer cursor and same case order')
                            if not torch.equal(saved['scale4'].cuda(),scale4):raise ValueError('Output scale changed')
                            model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
                            step=saved['step'];arm.update(updates=step,starting_step=step,added_updates=0,checkpoint_reads=1,
                                inherited_seconds=saved['arm'].get('elapsed_seconds'),discarded_incomplete_backward=bool(saved['arm'].get('partial_backward')))
                            reference=saved['arm']['best']
                            random.setstate(saved['rng']['python']);np.random.set_state(saved['rng']['numpy']);torch.set_rng_state(saved['rng']['torch'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in saved['rng']['cuda']])
                            del saved
                            evaluate()
                            if abs(last['worst_field_percent']-reference['worst_field_percent'])>1e-4:raise ValueError('Restored endpoint differs')
                            arm['restored_endpoint_percent']=last['worst_field_percent'];arm['actual_model_Adam_scheduler_RNG_cursor_restore']=True
                        else:
                            evaluate()
                        for attempt in range(1,config['max_updates_per_arm']+1):
                            arm_guard();model.train();optimizer.zero_grad(set_to_none=True);arm['partial_backward']=True
                            loss=backward_case(method,model,cases[step%len(cases)],arm_guard)
                            if step==0:
                                gradients=[p.grad for p in model.parameters() if p.grad is not None]
                                if not gradients or not all(torch.isfinite(g).all() for g in gradients):raise FloatingPointError('Invalid actual full-node gradients')
                                arm['actual_full_node_backward']=True
                            arm['optimizer_step_in_progress']=True
                            optimizer.step();step+=1;arm.update(updates=step,partial_backward=False,optimizer_step_in_progress=False,training_loss=loss)
                            if config.get('continuation'):arm['added_updates']=step-arm['starting_step']
                            if step%10==0:update(phase=name+'/training')
                            if step%config['evaluate_every']==0:
                                metric=evaluate()
                                if not config.get('fixed_learning_rate'):scheduler.step(metric)
                                checkpoint('last.pt')
                                if arm['goal_met'] and not config.get('fixed_endpoint'):break
                        if (last is None or last['step']!=step):evaluate()
                        if config.get('fixed_endpoint'):
                            arm['endpoint']=last;arm['goal_met']=last['worst_field_percent']<config['target_relative_l2']*100
                        arm['stop_reason']='fixed_update_endpoint' if config.get('fixed_endpoint') else ('goal_met' if arm['goal_met'] else 'update_cap')
                        arm['status']='completed'
                    except TimeoutError as exc:
                        if 'Whole-stage' in str(exc):raise
                        arm.update(status='completed',stop_reason='fixed_arm_time_cap',partial_backward=arm['partial_backward'])
                        if model is not None and (last is None or last['step']!=step):evaluate()
                    except Exception as exc:
                        arm.update(status='failed',error_type=type(exc).__name__,error=str(exc));traceback.print_exc()
                    finally:
                        checkpoint('last.pt')
                        arm['elapsed_seconds']={'UTC':time.time()-start[0],'monotonic':time.monotonic()-start[1]}
                        write(folder/'summary.json',arm)
                        close_owned_process_group();del model,optimizer;gc.collect();torch.cuda.empty_cache();update()
            state.update(status='completed',active=False,closed=True);state['all_arms_attempted']=len(state['arms'])==len(groups)*len(config['methods'])
            state['all_goal_met']=all(a['goal_met'] for a in state['arms'].values())
            update(phase='terminal');write(output/'summary.json',state)
        except BaseException as exc:
            update(status='failed',active=False,closed=True,phase='terminal',error_type=type(exc).__name__,error=str(exc));write(output/'summary.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--lock',required=True)
    run(p.parse_args())
