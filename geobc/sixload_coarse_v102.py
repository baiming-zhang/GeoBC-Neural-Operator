"""Fixed32 accepted coarse math, three independent simultaneous PCG lanes."""
import time

class Ledger:
    def __init__(self,torch,guard,persist,output,setup_began):
        from .elastic_coarse import COUNTERS,TIMES
        self.torch,self.guard,self.persist,self.output,self.setup_began=torch,guard,persist,output,setup_began
        self.data={**{k:0 for k in COUNTERS},**{k:0. for k in TIMES}}
        self.setup_active=True
    def sync(self):self.torch.cuda.synchronize()
    def check_memory(self):self.guard()
    def check(self):
        self.guard()
        if self.setup_active and max(time.time()-self.setup_began[0],time.monotonic()-self.setup_began[1])>=120:raise TimeoutError('cache+coarse setup120s')
    def account(self):self.check()

class ThreeLaneCoarse:
    def __init__(self,torch,scalar,guard):self.torch,self.scalar,self.guard,self.calls=torch,scalar,guard,0
    def apply(self,value):
        t=self.torch;c=self.scalar;assert value.shape==(3,c.P.shape[0],3)
        self.guard();self.calls+=1
        fine=t.cholesky_solve(value[...,None],c.chol[None]).squeeze(-1)
        # Columns are each lane's contiguous XYZ; CSR operates simultaneously.
        rhs=t.sparse.mm(c.PT,value.permute(1,0,2).reshape(c.P.shape[0],9)).reshape(c.P.shape[1],3,3).permute(0,2,1).reshape(-1,3)
        z=t.zeros_like(rhs)
        for _ in range(32):z=z+c.omega*(rhs-t.sparse.mm(c.Ac,z))/c.diagonal[:,None]
        result=fine+t.sparse.mm(c.P,z.reshape(c.P.shape[1],3,3).permute(0,2,1).reshape(c.P.shape[1],9)).reshape(c.P.shape[0],3,3).permute(1,0,2)
        self.guard();return result

def qualify(torch,parallel,scalar,free_count):
    gen=torch.Generator(device='cuda').manual_seed(2026)
    u=torch.randn((2,free_count,3),dtype=torch.float64,device='cuda',generator=gen)
    # Two fixed nonzero vectors, three real fullN lanes; no new random witness.
    lanes=torch.stack((u[0],u[1],u[0]+u[1]))
    actual=parallel.apply(lanes);reference=torch.stack((scalar.apply(u[0]),scalar.apply(u[1])))
    errors=torch.linalg.vector_norm(actual[:2]-reference,dim=(1,2))/torch.linalg.vector_norm(reference,dim=(1,2))
    linear=torch.linalg.vector_norm(actual[2]-actual[0]-actual[1])/torch.linalg.vector_norm(actual[2])
    assert torch.isfinite(actual).all() and bool((errors<=1e-9).all()) and float(linear)<=1e-9
    return dict(passed=True,actual_free_nodes=free_count,actual_RHS_lanes=3,scalar_parity_relative=errors.cpu().tolist(),linearity_relative=float(linear),fixed_inner_steps=32)

def solve_group(torch,np,op,chol,inputs,forces,names,folder,a,guard,persist,remaining_solve, precondition):
    import time,os
    from . import available_cad_pilot_data as d
    from . import elastic_pilot as old
    began=time.monotonic();began_epoch=time.time();call0=op.calls
    folder.mkdir(parents=True,exist_ok=False);free=inputs['free_node_ids'];b=forces[:,free];norm=torch.linalg.vector_norm(b,dim=(1,2));assert (norm>0).all()and torch.isfinite(norm).all()
    z=torch.zeros_like(b);r=b.clone();p=torch.zeros_like(b);active=torch.ones(3,device=b.device,dtype=torch.bool);iterations=torch.zeros(3,dtype=torch.int64,device=b.device)
    def apply(v):
        physical=torch.zeros_like(forces);physical[:,free]=v;return op.apply(physical)[:,free]
    tr=b-apply(z);relative=torch.linalg.vector_norm(tr,dim=(1,2))/norm;r=tr.clone();q=precondition(r);rz=(r*q).sum((1,2));p=q.clone();assert (rz>0).all()
    trace=[];step=0;active_flags=[True]*3;attempt_updates=[0]*3
    def elapsed():return max(time.monotonic()-began,time.time()-began_epoch)
    def save():
        guard();temp=folder/'last_state.npz.tmp';target=folder/'last_state.npz'
        with temp.open('wb')as f:np.savez(f,z=z.cpu().numpy(),r=r.cpu().numpy(),p=p.cpu().numpy(),rz=rz.cpu().numpy(),active=active.cpu().numpy(),iterations=iterations.cpu().numpy())
        os.replace(temp,target)
        d.write(folder/'last_state.json',dict(names=names,step=step,iterations=iterations.cpu().tolist(),elapsed_seconds=elapsed(),operator_calls=op.calls-call0,original_plan_SCI=a['execution']['scientific_digest'],arrays=d.identity(target,hashed=True),trace=trace))
        a['running_group']=dict(names=names,step=step,iterations=iterations.cpu().tolist(),elapsed_seconds=elapsed(),true_relative_residual=relative.cpu().tolist());persist();guard()
    while bool(active.any()):
        guard()
        if step>=10000 or elapsed()>=290 or remaining_solve()<=10:break
        step+=1;iterations[active]+=1
        attempt_updates=[x+int(flag)for x,flag in zip(attempt_updates,active_flags)]
        a['running_group_attempts']=dict(names=names,step=step,RHS_attempt_updates=attempt_updates,elapsed_seconds=elapsed())
        Ap=apply(p);pAp=(p*Ap).sum((1,2))
        if not torch.isfinite(Ap).all()or not torch.isfinite(pAp[active]).all()or not (pAp[active]>0).all():raise old.ScientificNegative('Batched PCG pTAp breakdown, no shift/fallback')
        alpha=torch.zeros(3,device=b.device,dtype=b.dtype);alpha[active]=rz[active]/pAp[active]
        z+=alpha[:,None,None]*p;r-=alpha[:,None,None]*Ap
        if not torch.isfinite(z).all()or not torch.isfinite(r).all():raise old.ScientificNegative('Nonfinite PCG vectors')
        recursive=torch.linalg.vector_norm(r,dim=(1,2))/norm
        if step%25==0 or bool((recursive[active]<=1e-8).any()):
            tr=b-apply(z);relative=torch.linalg.vector_norm(tr,dim=(1,2))/norm
            if not torch.isfinite(tr).all():raise old.ScientificNegative('Nonfinite true residual')
            trace.append(dict(step=step,true_relative_residual=relative.cpu().tolist(),elapsed_seconds=elapsed()))
            active=active&(relative>1e-8);active_flags=active.cpu().tolist()
        if bool(active.any()):
            q=precondition(r);newrz=(r*q).sum((1,2))
            if not torch.isfinite(newrz[active]).all()or not (newrz[active]>0).all():raise old.ScientificNegative('Batched PCG rTMinvr breakdown')
            beta=torch.zeros(3,device=b.device,dtype=b.dtype);beta[active]=newrz[active]/rz[active]
            p=q+beta[:,None,None]*p;p[~active]=0.;rz=newrz
        else:p.zero_()
        if step%25==0:save()
    tr=b-apply(z);relative=torch.linalg.vector_norm(tr,dim=(1,2))/norm;save()
    outcomes={}
    for i,name in enumerate(names):
        guard();physical=old._restore_full(torch,z[i],inputs);force=forces[i]
        Ku=op.original.apply(physical);residual=Ku-force;reaction=torch.zeros_like(residual);reaction[inputs['fixed_node_ids']]=residual[inputs['fixed_node_ids']]
        external=force+reaction;netforce=external.sum(0);netmoment=torch.cross(inputs['vertices_mm']-inputs['reference_xyz_mm'],external,dim=-1).sum(0)
        assert torch.isfinite(physical).all()and torch.isfinite(residual).all()and torch.isfinite(netforce).all()and torch.isfinite(netmoment).all()
        target=folder/name/'physical_U_mm.npy';item=old._save_array(np,target,physical)
        item.update(d.identity(target,hashed=True))
        old_relative=float(torch.linalg.vector_norm(residual[free])/norm[i])
        numerical='converged'if old_relative<=1e-8 else'finite_budget_endpoint_not_converged'
        record=dict(key=name,all_original_nodes=len(physical),status='completed',numerical_state=numerical,true_relative_residual=old_relative,batched_true_relative_residual=float(relative[i]),iterations=int(iterations[i]),initial_guess='independent_zero',physical_U=item,net_force_N=netforce.cpu().tolist(),net_moment_Nmm=netmoment.cpu().tolist(),constraint_reaction_norm=float(reaction.norm()))
        path=folder/name/'manifest.json';d.write(path,record);outcomes[name]=d.identity(path,hashed=True)
    torch.cuda.synchronize();return outcomes,elapsed()
