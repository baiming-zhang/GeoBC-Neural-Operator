"""Geometry-only reference constraint projection and independent nullspace PCG."""
class ReferenceProjector:
    def __init__(self,t,H,members,free,guard):
        self.t,self.H,self.free,self.guard=t,H,free,guard;guard()
        assert H.dtype==t.float64 and members.dtype==free.dtype==t.int64
        assert bool((free[1:]>free[:-1]).all())
        self.positions=t.searchsorted(free,members);assert bool((self.positions<len(free)).all())and t.equal(free[self.positions],members)
        self.G=H@H.T
        assert int(t.linalg.matrix_rank(self.G))==6 and float(t.linalg.cond(self.G))<=1e12
        self.chol=t.linalg.cholesky(self.G)
        self.metadata=dict(actual_free_nodes=len(free),actual_member_nodes=len(members),actual_rank=6,G_condition=float(t.linalg.cond(self.G)),
            geometry_only=True,physics_basis_used_for_lift=False,pseudoinverse_used=False)
    def response(self,v):
        single=v.ndim==2;v=v[None]if single else v
        assert v.ndim==3 and v.shape[1:]==(len(self.free),3)
        q=v[:,self.positions].reshape(len(v),-1)@self.H.T
        return q[0]if single else q
    def scatter(self,q):
        t=self.t;single=q.ndim==1;q=q[None]if single else q
        out=t.zeros((len(q),len(self.free),3),dtype=q.dtype,device=q.device)
        out[:,self.positions]=(q@self.H).reshape(len(q),-1,3)
        return out[0]if single else out
    def solve_G(self,q):
        single=q.ndim==1;q=q[None]if single else q
        result=self.t.cholesky_solve(q.T,self.chol).T
        return result[0]if single else result
    def project(self,v):
        self.guard();return v-self.scatter(self.solve_G(self.response(v)))
    def lift(self,q):return self.scatter(self.solve_G(q))

class ProjectedThreeK:
    def __init__(self,t,parallel,scalar,projector,inputs,guard):
        self.t,self.parallel,self.scalar,self.projector,self.inputs,self.guard=t,parallel,scalar,projector,inputs,guard;self.calls=0
    def apply(self,v):
        t=self.t;P=self.projector;self.guard();assert v.shape==(3,len(P.free),3)
        physical=t.zeros((3,len(self.inputs['vertices_mm']),3),dtype=v.dtype,device=v.device);physical[:,P.free]=P.project(v)
        self.calls+=1;return P.project(self.parallel.apply(physical)[:,P.free])
    def scalar_apply(self,v):
        t=self.t;P=self.projector;physical=t.zeros_like(self.inputs['vertices_mm']);physical[P.free]=P.project(v)
        return P.project(self.scalar.apply(physical)[P.free])

class ProjectedThreeM:
    def __init__(self,parallel,scalar,projector):self.parallel,self.scalar,self.projector=parallel,scalar,projector
    def apply(self,v):return self.projector.project(self.parallel.apply(self.projector.project(v)))
    def scalar_apply(self,v):return self.projector.project(self.scalar.apply(self.projector.project(v)))

def qualify(t,P,K,M,target):
    index=t.arange(len(P.free)*3,dtype=t.float64,device='cuda').reshape(-1,3)
    u=index.sin();v=(index*.37+.2).cos();pu,pv=P.project(u),P.project(v)
    lanes=t.stack((pu,pv,pu+pv));Kactual=K.apply(lanes);Kref=t.stack([K.scalar_apply(x)for x in lanes])
    Mactual=M.apply(lanes);Mref=t.stack([M.scalar_apply(x)for x in lanes])
    def relative(x,y):return float(t.linalg.vector_norm(x-y)/t.linalg.vector_norm(y))
    kerr=relative(Kactual,Kref);merr=relative(Mactual,Mref);idem=relative(P.project(pu),pu)
    null=float(t.linalg.vector_norm(P.response(pu))/t.linalg.vector_norm(pu))
    symmetry=float(((u*pv).sum()-(v*pu).sum()).abs()/t.sqrt((u*pu).sum()*(v*pv).sum()))
    lift=P.lift(target);constraint=relative(P.response(lift),target)
    energies=[float((lanes[j]*Kactual[j]).sum())for j in (0,1)]
    menergy=[float((lanes[j]*Mactual[j]).sum())for j in (0,1)]
    assert bool(t.isfinite(Kactual).all()and t.isfinite(Mactual).all())and max(kerr,merr)<=1e-9
    assert max(idem,null,symmetry,constraint)<=1e-10 and min(energies)>0 and min(menergy)>0
    return dict(passed=True,actual_free_nodes=len(P.free),actual_RHS_lanes=3,projector_idempotence_relative=idem,
        projector_nullspace_relative=null,projector_symmetry_relative=symmetry,geometry_lift_constraint_relative=constraint,
        projected_K_scalar_parity=kerr,projected_M_scalar_parity=merr,restricted_positive_K_energies=energies,
        restricted_positive_M_energies=menergy,fullspace_projected_K_SPD_claim=False,inherited8fine_coarseguards_repeated=0)

def solve_group(t,np,K,M,P,inputs,target,forces,folder,a,guard,persist,remaining_solve):
    import time,os
    from . import available_cad_pilot_data as d
    from . import elastic_pilot as old
    began=(time.time(),time.monotonic());folder.mkdir(parents=True,exist_ok=False);free=P.free
    def elapsed():return max(time.time()-began[0],time.monotonic()-began[1])
    lift=P.lift(target);full_lift=t.zeros_like(forces);full_lift[:,free]=lift
    b=forces[:,free];projected_b=P.project(b)
    witness=t.linalg.vector_norm(projected_b,dim=(1,2))/t.linalg.vector_norm(b,dim=(1,2))
    assert bool((witness<=1e-10).all())
    rhs=projected_b-P.project(K.parallel.apply(full_lift)[:,free]);norm=t.linalg.vector_norm(rhs,dim=(1,2));assert bool((norm>0).all())
    z=t.zeros_like(rhs);r=rhs.clone();q=M.apply(r);rz=(r*q).sum((1,2));p=q.clone()
    assert bool(t.isfinite(rz).all()and (rz>0).all());active=t.ones(3,dtype=t.bool,device='cuda');iterations=t.zeros(3,dtype=t.int64,device='cuda')
    relative=t.linalg.vector_norm(r,dim=(1,2))/norm;trace=[];step=0;attempts=[0,0,0]
    def save():
        guard();temporary=folder/'last_state.npz.tmp'
        with temporary.open('wb')as file:np.savez(file,z=z.cpu().numpy(),r=r.cpu().numpy(),p=p.cpu().numpy(),rz=rz.cpu().numpy(),active=active.cpu().numpy(),iterations=iterations.cpu().numpy(),lift=lift.cpu().numpy(),rhs=rhs.cpu().numpy())
        os.replace(temporary,folder/'last_state.npz')
        d.write(folder/'last_state.json',dict(step=step,iterations=iterations.cpu().tolist(),elapsed_seconds=elapsed(),original_plan_SCI=a['execution']['scientific_digest'],
            arrays=d.identity(folder/'last_state.npz',hashed=True),trace=trace,initial_nullspace_guess='independent_zero',lift='geometry_only_Rt_solveG_target'))
        a['running_group']=dict(step=step,iterations=iterations.cpu().tolist(),elapsed_seconds=elapsed(),true_projected_relative_residual=relative.cpu().tolist());persist()
    while bool(active.any()):
        guard()
        if step>=10000 or elapsed()>=290 or remaining_solve()<=10:break
        step+=1;iterations[active]+=1;flags=active.cpu().tolist();attempts=[x+int(flag)for x,flag in zip(attempts,flags)]
        a['running_group_attempts']=dict(step=step,RHS_attempt_updates=attempts,elapsed_seconds=elapsed())
        Ap=K.apply(p);pAp=(p*Ap).sum((1,2))
        if not bool(t.isfinite(pAp[active]).all()and (pAp[active]>0).all()):raise old.ScientificNegative('Restricted projected PCG pTAp breakdown;no penalty shift')
        alpha=t.zeros(3,dtype=t.float64,device='cuda');alpha[active]=rz[active]/pAp[active]
        z=P.project(z+alpha[:,None,None]*p);r=P.project(r-alpha[:,None,None]*Ap)
        recursive=t.linalg.vector_norm(r,dim=(1,2))/norm
        if step%25==0 or bool((recursive[active]<=1e-8).any()):
            tr=rhs-K.apply(z);relative=t.linalg.vector_norm(tr,dim=(1,2))/norm
            if not bool(t.isfinite(tr).all()):raise old.ScientificNegative('Nonfinite constrained true residual')
            trace.append(dict(step=step,true_projected_relative_residual=relative.cpu().tolist(),elapsed_seconds=elapsed()))
            active=active&(relative>1e-8)
        if bool(active.any()):
            q=M.apply(r);newrz=(r*q).sum((1,2))
            if not bool(t.isfinite(newrz[active]).all()and (newrz[active]>0).all()):raise old.ScientificNegative('Restricted projected PCG rTMinvr breakdown')
            beta=t.zeros(3,dtype=t.float64,device='cuda');beta[active]=newrz[active]/rz[active]
            p=P.project(q+beta[:,None,None]*p);p[~active]=0;rz=newrz
        else:p.zero_()
        if step%25==0:save()
    tr=rhs-K.apply(z);relative=t.linalg.vector_norm(tr,dim=(1,2))/norm;save();outcomes={}
    for j in range(3):
        guard();u=old._restore_full(t,lift[j]+z[j],inputs);Ku=K.scalar.apply(u);residual=Ku-forces[j]
        lam=P.solve_G(P.response(residual[free]));free_equilibrium=residual[free]-P.scatter(lam)
        true=float(t.linalg.vector_norm(free_equilibrium)/norm[j]);constraint=float(t.linalg.vector_norm(P.response(u[free])-target[j])/t.linalg.vector_norm(target[j]))
        constraint_force=t.zeros_like(u);constraint_force[free]=P.scatter(lam)
        fixed_reaction=t.zeros_like(u);fixed_reaction[inputs['fixed_node_ids']]=residual[inputs['fixed_node_ids']]
        external=forces[j]+constraint_force+fixed_reaction
        netforce=external.sum(0);netmoment=t.cross(inputs['vertices_mm']-inputs['reference_xyz_mm'],external,dim=-1).sum(0)
        physical_reaction=lam*t.tensor([1.,1.,1.,float(inputs['rbe3_length_mm']),float(inputs['rbe3_length_mm']),float(inputs['rbe3_length_mm'])],dtype=t.float64,device='cuda')
        assert bool(t.isfinite(u).all()and t.isfinite(physical_reaction).all()and t.isfinite(free_equilibrium).all())
        name='prescribed'+str(j);target_path=folder/name/'physical_U_mm.npy';item=old._save_array(np,target_path,u);item.update(d.identity(target_path,hashed=True))
        record=dict(key=name,status='completed',all_original_nodes=len(u),numerical_state='converged'if true<=1e-8 and constraint<=1e-10 else'finite_budget_endpoint_not_converged',
            initial_nullspace_guess='independent_zero',geometry_only_lift=True,physicsbasis_warmstart_used=False,true_projected_relative_residual=true,
            recurrence_true_projected_relative_residual=float(relative[j]),projected_RHS_norm=float(norm[j]),reference_constraint_relative=constraint,
            projected_applied_reference_force_relative=float(witness[j]),iterations=int(iterations[j]),physical_U=item,reference_reaction_force6=physical_reaction.cpu().tolist(),
            actual_scaled_reference_pose=P.response(u[free]).cpu().tolist(),body_strain_energy_Nmm=float(.5*(Ku*u).sum()),
            net_force_including_constraint_and_fixed_reaction_N=netforce.cpu().tolist(),net_moment_including_constraint_and_fixed_reaction_Nmm=netmoment.cpu().tolist(),
            normalized_reference_multiplier_N=lam.cpu().tolist(),original_OptiStruct_GT=False)
        manifest=folder/name/'manifest.json';d.write(manifest,record);outcomes[name]=d.identity(manifest,hashed=True)
    t.cuda.synchronize();return outcomes,elapsed()
