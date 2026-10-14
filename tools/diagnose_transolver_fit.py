"""Bounded U-only correctness and optimization diagnostics, separate from rankings."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,gc,json,os
from pathlib import Path
import numpy as np
import torch
from geobc.overfit_models import make_model,relative_loss
from geobc.overfit_transformers import close_owned_process_group,_ensure_single_rank_group
from geobc.runtime import GPUQueue
from tools.train_sample_overfitting import read_case_arrays,errors


def run(args):
    entry=(ENTRY_EPOCH,ENTRY_MONOTONIC)
    out=Path(args.output);out.mkdir(exist_ok=False,parents=True)
    config=json.loads(Path('configs/matched_capacity_u_overfitting.json').read_text())
    sources=json.loads(Path(args.source).read_text())
    os.environ['GEOBC_TRANSFORMER_MANIFEST']=sources['transformer_manifest']
    report={'status':'running','timing_scope':'before_module_imports','source_reads':0,'stress_label_reads':0,'heldout_reads':0,'parity':{},'saved_endpoints':{},'real_probes':{},'synthetic_probes':{}}
    def save():
        report['elapsed_seconds']={'UTC':time.time()-entry[0],'monotonic':time.monotonic()-entry[1]}
        (out/'diagnostics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def guard():
        if max(time.time()-entry[0],time.monotonic()-entry[1])>600:raise TimeoutError('600-second diagnostic budget')
        free,total=torch.cuda.mem_get_info()
        if total-free>30500000000:raise MemoryError('Declared 30.5GB cap')
    def rng():return (torch.get_rng_state(),torch.cuda.get_rng_state_all())
    def restore(s):torch.set_rng_state(s[0]);torch.cuda.set_rng_state_all(s[1])
    def native(model,x,method):
        if method=='transolver':return model.backbone(x[...,:3],x[...,3:])
        if method=='transolverpp':
            _ensure_single_rank_group(x.device)
            return model.backbone((x,x[...,:3],None))
        return torch.cat(model.backbone(list(x.split(model.chunk_size,dim=1)),input_list=True,use_checkpoint=False),dim=1)
    def fresh(method,checkpoint=True):
        m=make_model(method,2026,config['model_settings'][method],out_channels=3)
        m.checkpoint_blocks=checkpoint
        return m
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(30500000000/torch.cuda.get_device_properties(0).total_memory)
    try:
        with GPUQueue(args.lock):
            key=config['single'][0]
            with np.load(sources['cases'][key],allow_pickle=False) as f:
                xyz,features,u=read_case_arrays(f,key,'u')
            report['source_reads']=1
            xyz=torch.from_numpy(xyz).cuda();features=torch.from_numpy(features).cuda();physical=torch.from_numpy(u).cuda()
            x=torch.cat((xyz,features),1)[None]
            scale=physical.double().square().sum(1).mean().sqrt().float()
            y=physical/scale;energy=y.square().sum(1).mean()[None]
            constant=physical.mean(0,keepdim=True).expand_as(physical)
            report['data']={'sample':key,'nodes':len(xyz),'xyz_min':xyz.amin(0).tolist(),'xyz_max':xyz.amax(0).tolist(),'features_min':features.amin(0).tolist(),'features_max':features.amax(0).tolist(),'features_std':features.std(0).tolist(),'U_RMS_mm':float(scale),'constant_U_mean_error_percent':errors(physical,constant)['U_percent'],'zero_U_error_percent':errors(physical,torch.zeros_like(physical))['U_percent']}
            save();print(json.dumps({'phase':'data','data':report['data']}),flush=True)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            indices=torch.linspace(0,len(xyz)-1,513,device='cuda').long()
            sx=x[:,indices];sy=y[indices];se=sy.square().sum(1).mean()[None]
            for method in ('transolver','transolverpp','ts3'):
                guard();model=fresh(method);model.train();model.chunk_size=257
                state=rng()
                p=model(sx);loss=relative_loss(p[0],sy,se,len(sy));loss.backward()
                grads={n:v.grad.detach().clone() for n,v in model.named_parameters() if v.grad is not None}
                unused=[n for n,v in model.named_parameters() if v.requires_grad and v.grad is None]
                base={n:v.detach().clone() for n,v in model.named_parameters()}
                opt=torch.optim.Adam(model.parameters(),lr=.001,fused=True);opt.step()
                cp_weights={n:v.detach().clone() for n,v in model.named_parameters()}
                with torch.no_grad():
                    for n,v in model.named_parameters():v.copy_(base[n])
                model.zero_grad(set_to_none=True);restore(state)
                q=native(model,sx,method);other=relative_loss(q[0],sy,se,len(sy));other.backward()
                native_grads={n:v.grad.detach().clone() for n,v in model.named_parameters() if v.grad is not None}
                if set(grads)!=set(native_grads):raise ValueError('Native/adapter gradient parameter sets differ')
                diff2=sum((grads[n]-native_grads[n]).double().square().sum() for n in grads)
                norm2=sum(native_grads[n].double().square().sum() for n in grads)
                other_opt=torch.optim.Adam(model.parameters(),lr=.001,fused=True);other_opt.step()
                update_diff=max(float((v.detach()-cp_weights[n]).abs().max()) for n,v in model.named_parameters())
                row={'output_max_abs_difference':float((p.detach()-q.detach()).abs().max()),'gradient_relative_l2_difference':float((diff2/norm2.clamp_min(1e-30)).sqrt()),'one_Adam_update_max_abs_difference':update_diff,'adapter_loss':float(loss.detach()),'native_loss':float(other.detach()),'unused_parameters':unused,'active_parameter_count':sum(v.numel() for n,v in model.named_parameters() if n in grads),'finite_gradients':all(bool(torch.isfinite(g).all()) for g in grads.values())}
                row['passed']=row['output_max_abs_difference']<1e-5 and row['gradient_relative_l2_difference']<1e-4 and update_diff<1e-5 and row['finite_gradients']
                report['parity'][method]=row;save();print(json.dumps({'phase':'native_parity','method':method,**row}),flush=True)
                del model,opt,other_opt,grads,native_grads,base,cp_weights,p,q,loss,other;close_owned_process_group();gc.collect();torch.cuda.empty_cache()
            torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
            for method in ('transolver','transolverpp','ts3'):
                guard();model=fresh(method);model.eval()
                saved=torch.load(Path('runs/matched_capacity_u_overfitting')/('single_'+method)/'last.pt',map_location='cuda',weights_only=False)
                model.load_state_dict(saved['model'],strict=True)
                restore((saved['rng']['torch'].cpu(),[v.cpu() for v in saved['rng']['cuda']]))
                with torch.no_grad():pred=model(x)[0]*saved['scale4'].cuda()
                score=errors(physical,pred)['U_percent']
                variation=float(torch.linalg.vector_norm(pred-pred.mean(0))/torch.linalg.vector_norm(pred).clamp_min(1e-20))
                report['saved_endpoints'][method]={'step':saved['step'],'saved_U_percent':saved['arm']['endpoint']['mean_U_percent'],'recomputed_U_percent':score,'prediction_spatial_variation_relative':variation,'scale_max_abs_difference':float((saved['scale4'].cuda()-scale).abs().max()),'optimizer_steps':sorted({int(v['step']) for v in saved['optimizer']['state'].values()}),'eval_stochastic':method=='transolverpp'}
                save();print(json.dumps({'phase':'saved_endpoint','method':method,**report['saved_endpoints'][method]}),flush=True)
                del model,saved,pred;close_owned_process_group();gc.collect();torch.cuda.empty_cache()
            for label,lr,checkpoint in (('adapter_lr001',.001,True),('native_lr001',.001,False),('adapter_lr0001',.0001,True)):
                guard();model=fresh('transolver',checkpoint);opt=torch.optim.Adam(model.parameters(),lr=lr,fused=True);curve=[]
                for step in range(101):
                    guard();model.train();opt.zero_grad(set_to_none=True);pred=model(x)[0]
                    loss=relative_loss(pred,y,energy,len(y))
                    if step%20==0:curve.append({'step':step,'U_percent':100*float(loss.detach().sqrt())})
                    if step==100:break
                    loss.backward();opt.step()
                report['real_probes'][label]={'updates':100,'learning_rate':lr,'checkpoint_blocks':checkpoint,'curve':curve}
                save();print(json.dumps({'phase':'real_probe','label':label,**report['real_probes'][label]}),flush=True)
                del model,opt,pred,loss;gc.collect();torch.cuda.empty_cache()
            for method in ('transolver','transolverpp','ts3'):
                guard();model=fresh(method);model.chunk_size=257
                coords=sx[0,:,:3];simple=coords-coords.mean(0)
                simple=simple/simple.square().sum(1).mean().sqrt();senergy=simple.square().sum(1).mean()[None]
                opt=torch.optim.Adam(model.parameters(),lr=.0001,fused=True);curve=[]
                for step in range(501):
                    guard();model.train();opt.zero_grad(set_to_none=True);pred=model(sx)[0]
                    loss=relative_loss(pred,simple,senergy,len(simple))
                    if step%100==0:curve.append({'step':step,'synthetic_U_percent':100*float(loss.detach().sqrt())})
                    if step==500:break
                    loss.backward();opt.step()
                report['synthetic_probes'][method]={'updates':500,'learning_rate':.0001,'nodes':len(simple),'target':'centered normalized XYZ, diagnostic only, not physical U accuracy','curve':curve}
                save();print(json.dumps({'phase':'synthetic_probe','method':method,**report['synthetic_probes'][method]}),flush=True)
                del model,opt,pred,loss;close_owned_process_group();gc.collect();torch.cuda.empty_cache()
            report['status']='completed';save()
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc));save();raise
    finally:close_owned_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--lock',required=True);run(p.parse_args())
