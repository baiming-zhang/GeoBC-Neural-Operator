"""All four actual source support ports; geometry-only nonzero fixed lifts."""
def validate_ports(t,inputs,source):
    from .elastic_pilot import ScientificNegative
    facts=source['source_support_constraints'];refs=facts['support_reference_GRID_IDs']
    if refs!=sorted(refs)or len(set(refs))!=4 or facts['actual_card_CM']!=['123456']*4 or facts['direct_fixed_physical_nodes']!=0:
        raise ScientificNegative('Unsupported full source support port scope')
    fixed=inputs['fixed_node_ids'];free=inputs['free_node_ids'];grid=inputs['grid_node_ids'];ports=[];records=[]
    assert fixed.dtype==free.dtype==grid.dtype==t.int64
    for j,reference in enumerate(refs):
        nodes=inputs['region_support_'+str(j)];assert nodes.dtype==t.int64 and nodes.ndim==1
        if nodes.numel()==0 or nodes.unique().numel()!=len(nodes)or not bool(t.isin(nodes,fixed).all()):raise ScientificNegative('Actual source support members incompatible')
        masters=t.where(grid==reference)[0]
        if len(masters)>1 or not bool(t.isin(masters,fixed).all()):raise ScientificNegative('Existing source reference output row incompatible')
        port=t.cat((nodes,masters)).unique(sorted=True)
        if bool(t.isin(port,free).any())or any(bool(t.isin(port,v).any())for v in ports):raise ScientificNegative('Actual source port/free or port/port overlap')
        ports.append(port);records.append(dict(source_reference_GRID_ID=reference,physical_member_rows=len(nodes),
            actual_reference_output_rows=len(masters),actual_port_rows=len(port),source_CM='123456',reference_row_invented=False))
    union=t.cat(ports).unique(sorted=True)
    if not t.equal(union,fixed.sort().values):raise ScientificNegative('Four actual support ports do not cover original fixed partition')
    old=source['existing_port0_actual_facts']
    for key in ('source_reference_GRID_ID','physical_member_rows','actual_reference_output_rows','actual_port_rows'):
        if records[0][key]!=old[key]:raise ScientificNegative('Frozen existing port0 context changed')
    others=fixed[~t.isin(fixed,union)]
    return ports,others,dict(passed=True,ports=records,source_regions_disjoint=True,source_fixed_union_exact=True,
        actual_all_fixed_rows=len(fixed),actual_other_fixed_rows=len(others),rotations_prescribed_zero=True)

def make_lift(t,inputs,ports,others,motions):
    assert motions.shape==(3,4,3)
    lift=t.zeros((3,len(inputs['vertices_mm']),3),dtype=t.float64,device='cuda')
    for j,port in enumerate(ports):
        lift[:,port]=motions[:,j,None,:]
        assert t.equal(lift[:,port],motions[:,j,None,:].expand(-1,len(port),-1))
    assert bool((lift[:,inputs['free_node_ids']]==0).all()and (lift[:,others]==0).all())
    return lift

def solve_group(t,np,op,M,inputs,ports,others,motions,forces,names,folder,a,guard,persist,remaining_solve):
    import time,os
    from . import available_cad_pilot_data as d
    from . import elastic_pilot as old
    began=(time.time(),time.monotonic());folder.mkdir(parents=True,exist_ok=False);free=inputs['free_node_ids']
    def elapsed():return max(time.time()-began[0],time.monotonic()-began[1])
    lift=make_lift(t,inputs,ports,others,motions);rhs=forces[:,free]-op.apply(lift)[:,free]
    norm=t.linalg.vector_norm(rhs,dim=(1,2));assert bool((norm>0).all()and t.isfinite(norm).all())
    def apply(v):
        physical=t.zeros_like(forces);physical[:,free]=v;return op.apply(physical)[:,free]
    z=t.zeros_like(rhs);r=rhs.clone();q=M.apply(r);rz=(r*q).sum((1,2));p=q.clone()
    assert bool(t.isfinite(rz).all()and (rz>0).all());active=t.ones(3,dtype=t.bool,device='cuda');iterations=t.zeros(3,dtype=t.int64,device='cuda')
    relative=t.linalg.vector_norm(r,dim=(1,2))/norm;trace=[];step=0;attempts=[0,0,0]
    def save():
        guard();temporary=folder/'last_state.npz.tmp'
        with temporary.open('wb')as file:np.savez(file,z=z.cpu().numpy(),r=r.cpu().numpy(),p=p.cpu().numpy(),rz=rz.cpu().numpy(),active=active.cpu().numpy(),iterations=iterations.cpu().numpy(),lift=lift.cpu().numpy(),rhs=rhs.cpu().numpy(),motions=motions.cpu().numpy(),all_port_rows=t.cat(ports).cpu().numpy(),port_rowptr=np.cumsum([0]+[len(v)for v in ports],dtype=np.int64))
        os.replace(temporary,folder/'last_state.npz')
        d.write(folder/'last_state.json',dict(names=names,step=step,iterations=iterations.cpu().tolist(),elapsed_seconds=elapsed(),original_plan_SCI=a['execution']['scientific_digest'],
            arrays=d.identity(folder/'last_state.npz',hashed=True),trace=trace,initial_free_guess='independent_zero',lift='geometry_only_nonzero_fixed_port'))
        a['running_group']=dict(names=names,step=step,iterations=iterations.cpu().tolist(),elapsed_seconds=elapsed(),true_free_relative_residual=relative.cpu().tolist());persist()
    while bool(active.any()):
        guard()
        if step>=10000 or elapsed()>=290 or remaining_solve()<=10:break
        step+=1;iterations[active]+=1;flags=active.cpu().tolist();attempts=[x+int(flag)for x,flag in zip(attempts,flags)]
        a['running_group_attempts']=dict(names=names,step=step,RHS_attempt_updates=attempts,elapsed_seconds=elapsed())
        Ap=apply(p);pAp=(p*Ap).sum((1,2))
        if not bool(t.isfinite(pAp[active]).all()and (pAp[active]>0).all()):raise old.ScientificNegative('Support-motion PCG pTAp breakdown;no shift')
        alpha=t.zeros(3,dtype=t.float64,device='cuda');alpha[active]=rz[active]/pAp[active]
        z+=alpha[:,None,None]*p;r-=alpha[:,None,None]*Ap
        recursive=t.linalg.vector_norm(r,dim=(1,2))/norm
        if step%25==0 or bool((recursive[active]<=1e-8).any()):
            tr=rhs-apply(z);relative=t.linalg.vector_norm(tr,dim=(1,2))/norm
            if not bool(t.isfinite(tr).all()):raise old.ScientificNegative('Nonfinite support-motion true residual')
            trace.append(dict(step=step,true_free_relative_residual=relative.cpu().tolist(),elapsed_seconds=elapsed()))
            active=active&(relative>1e-8)
        if bool(active.any()):
            q=M.apply(r);newrz=(r*q).sum((1,2))
            if not bool(t.isfinite(newrz[active]).all()and (newrz[active]>0).all()):raise old.ScientificNegative('Support-motion PCG rTMinvr breakdown')
            beta=t.zeros(3,dtype=t.float64,device='cuda');beta[active]=newrz[active]/rz[active]
            p=q+beta[:,None,None]*p;p[~active]=0;rz=newrz
        else:p.zero_()
        if step%25==0:save()
    tr=rhs-apply(z);relative=t.linalg.vector_norm(tr,dim=(1,2))/norm;save();outcomes={}
    for j,name in enumerate(names):
        guard();u=lift[j].clone();u[free]=z[j]
        # Keep actual nonzero fixed translations; old zero-boundary restore is
        # not used. The accepted loaded reference has no derived H5 output row.
        assert inputs['derived_node_ids'].numel()==0
        Ku=op.original.apply(u);residual=Ku-forces[j];true=float(t.linalg.vector_norm(residual[free])/norm[j])
        wanted=lift[j,inputs['fixed_node_ids']];actual=u[inputs['fixed_node_ids']];targetnorm=float(t.linalg.vector_norm(wanted));error=float(t.linalg.vector_norm(actual-wanted))
        constraint=error/targetnorm if targetnorm>0 else None;absolute=float((actual-wanted).abs().max())
        boundary_pass=constraint<=1e-10 if constraint is not None else absolute<=1e-12
        other_max=float(u[others].abs().max())if len(others)else 0.;boundary_pass=boundary_pass and other_max<=1e-12
        reaction=t.zeros_like(u);reaction[inputs['fixed_node_ids']]=residual[inputs['fixed_node_ids']]
        portforce=t.stack([reaction[v].sum(0)for v in ports]);external=forces[j]+reaction
        netforce=external.sum(0);netmoment=t.cross(inputs['vertices_mm']-inputs['reference_xyz_mm'],external,dim=-1).sum(0)
        external_work=(forces[j]*u).sum();support_work=(motions[j]*portforce).sum();body_work=(u*Ku).sum()
        enorm=float(body_work.abs());eerror=float((body_work-external_work-support_work).abs());energy_relative=eerror/enorm if enorm>0 else None
        energy_pass=energy_relative<=1e-5 if energy_relative is not None else eerror<=1e-12
        assert bool(t.isfinite(u).all()and t.isfinite(reaction).all()and t.isfinite(body_work))
        target_path=folder/name/'physical_U_mm.npy';item=old._save_array(np,target_path,u);item.update(d.identity(target_path,hashed=True))
        record=dict(key=name,status='completed',all_original_nodes=len(u),numerical_state='converged'if true<=1e-8 and boundary_pass else'finite_budget_endpoint_not_converged',
            initial_free_guess='independent_zero',geometry_only_fixed_lift=True,physicsbasis_warmstart_used=False,true_free_relative_residual=true,
            recurrence_true_free_relative_residual=float(relative[j]),actual_nonzero_free_RHS_norm=float(norm[j]),all_fixed_boundary_relative=constraint,
            all_fixed_boundary_maximum_absolute_mm=absolute,other_fixed_maximum_absolute_mm=other_max,all_fixed_boundary_passed=boundary_pass,
            all_support_translations_mm=motions[j].cpu().tolist(),all_support_reaction_forces_N=portforce.cpu().tolist(),all_support_local_moments_Nmm=[None]*4,
            body_strain_energy_Nmm=float(.5*body_work),external_force_work_Nmm=float(external_work),all_support_motion_work_Nmm=float(support_work),
            body_energy_virtualwork_relative=energy_relative,body_energy_virtualwork_absolute_Nmm=eerror,body_energy_virtualwork_passed=energy_pass,
            iterations=int(iterations[j]),physical_U=item,net_force_including_all_fixed_reaction_N=netforce.cpu().tolist(),
            net_moment_about_loaded_reference_including_all_fixed_reaction_Nmm=netmoment.cpu().tolist(),original_OptiStruct_GT=False)
        manifest=folder/name/'manifest.json';d.write(manifest,record);outcomes[name]=d.identity(manifest,hashed=True)
    t.cuda.synchronize();return outcomes,elapsed()
