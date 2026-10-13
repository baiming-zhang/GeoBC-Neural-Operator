"""Produce physics U/gradient inputs for the fixed final CADs without labels."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
import json
import os
from pathlib import Path
import signal


def run(args):
    config=json.loads(Path(args.config).read_text());catalog=json.loads(Path(args.source).read_text())
    clock=json.loads(os.environ['GEOBC_EXPERIMENT_CLOCK']);output=Path(args.output)
    state=dict(status='running',active=True,closed=False,phase='imports',scientific_group='FINAL',CAD_denominator=20,load_denominator=60,
               case_records={},features={},saved_endpoints=0,complete_inputs=False,source_reads=0,
               GT_reads=0,raw_source_reads=0,model_reads=0,combined_solve_seconds=0.,gradient_seconds=0.,
               peak_allocated_bytes=0,peak_reserved_bytes=0)
    torch=None
    def elapsed(tick):return max(time.time()-tick[0],time.monotonic()-tick[1])
    def persist():
        state['native_seconds']=dict(UTC=time.time()-ENTRY_EPOCH,monotonic=time.monotonic()-ENTRY_MONOTONIC)
        temporary=output/'status.json.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,output/'status.json')
    def guard():
        if min(clock['deadline_epoch']-time.time(),clock['deadline_monotonic']-time.monotonic())<=0:
            raise TimeoutError('Final-input stage deadline')
        if torch is not None and torch.cuda.is_initialized():
            state['peak_allocated_bytes']=max(state['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
            state['peak_reserved_bytes']=max(state['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
            free,total=torch.cuda.mem_get_info()
            if max(total-free,state['peak_allocated_bytes'],state['peak_reserved_bytes'])>config['memory_bytes']:
                raise MemoryError('Final-input memory budget')
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
        with temporary.open('wb') as file:np.save(file,value.detach().cpu().numpy(),allow_pickle=False)
        os.replace(temporary,path)
        return dict(path=str(path.resolve()),shape=list(value.shape),dtype=str(value.dtype).removeprefix('torch.'))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('External deadline')))
    try:
        import gc
        import numpy as np
        import torch as tm
        torch=tm
        from geobc.runtime import GPUQueue,CoarseLedger
        from geobc.physics_inputs import load_source,local_path
        from geobc.elasticity import ScientificNegative,_build_geometry,_algebra_guards,_restore_full
        from geobc.coarse import make_mapping,assemble_coarse,AdditiveCoarse,algebra_guards
        from geobc.parallel_elasticity import CanonicalLanesElastic
        from geobc.chebyshev import ChebyshevCoarse
        from geobc.pcg import solve
        from geobc.kinematics import qualify_derivative,recover_kinematic_inputs
        split=json.loads(Path('configs/split.json').read_text());recipe=json.loads(Path('configs/physics.json').read_text())
        if not catalog['complete_sources'] or catalog['scientific_group']!='FINAL' or catalog['CADs']!=split['final'] or len(catalog['sources'])!=20:
            raise ValueError('All fixed final sources must be qualified')
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with GPUQueue(args.lock):
            torch.cuda.set_per_process_memory_fraction(config['memory_bytes']/torch.cuda.get_device_properties(0).total_memory)
            torch.cuda.reset_peak_memory_stats()
            for cad in catalog['CADs']:
                guard();state['phase']=cad+'/source';persist();tick=(time.time(),time.monotonic())
                source=catalog['sources'][cad];folder=output/cad;folder.mkdir()
                inputs,forces=load_source(source['path'],source['expected_geometry'],guard);state['source_reads']+=1
                record=dict(recipe=recipe,expected_geometry=source['expected_geometry'])
                start=time.monotonic();prior=elapsed(tick)
                operator,chol,diagonal,geometry=_build_geometry(torch,inputs,record,start,prior)
                fine=_algebra_guards(torch,operator,diagonal,inputs,record,start,prior)
                arrays=dict(gradients=operator.gradients,weights=operator.weights,free_cholesky=chol)
                for name,value in arrays.items():save(folder/'cache'/(name+'.npy'),value)
                ledger=CoarseLedger(torch,guard);ct=(time.time(),time.monotonic())
                P,PT,corners,cmap,fmap,mapping=make_mapping(torch,inputs,ledger)
                Ac,cd,gamma,assembly=assemble_coarse(torch,inputs,arrays,cmap,fmap,ledger)
                scalar=AdditiveCoarse(torch,P,PT,Ac,cd,gamma,chol,ledger)
                coarse_checks=algebra_guards(torch,scalar,operator,inputs,ledger)
                for name,matrix in [('P',P),('PT',PT),('Ac',Ac)]:
                    for part,value in [('rowptr',matrix.crow_indices()),('columns',matrix.col_indices()),('values',matrix.values())]:save(folder/'cache'/(name+'_'+part+'.npy'),value)
                save(folder/'cache/coarse_diagonal.npy',cd)
                if elapsed(ct)>config['coarse_seconds']:raise ScientificNegative('Fixed coarse setup allowance exhausted')
                free=inputs['free_node_ids'];parallel=CanonicalLanesElastic(torch,operator,guard)
                candidate=ChebyshevCoarse(torch,P,PT,Ac,cd,gamma,chol,ledger)
                gen=torch.Generator(device='cuda').manual_seed(2026)
                w=torch.zeros_like(forces);w[:,free]=torch.randn((3,len(free),3),dtype=torch.float64,device='cuda',generator=gen)
                K=parallel.apply(w);Kref=torch.stack([operator.apply(v) for v in w])
                kp=torch.linalg.vector_norm(K-Kref,dim=(1,2))/torch.linalg.vector_norm(Kref,dim=(1,2))
                M=candidate.apply_lanes(w[:,free]);Mref=torch.stack([candidate.apply(v) for v in w[:,free]])
                mp=torch.linalg.vector_norm(M-Mref,dim=(1,2))/torch.linalg.vector_norm(Mref,dim=(1,2))
                symmetry=abs(float((w[0,free]*M[1]).sum()-(w[1,free]*M[0]).sum()))/float(w[0,free].norm()*M[1].norm()+w[1,free].norm()*M[0].norm())
                if not bool((kp<=1e-9).all() and (mp<=1e-9).all() and ((w[:,free]*M).sum((1,2))>0).all()) or symmetry>1e-9:
                    raise ScientificNegative('New-geometry K/M lane qualification fails')
                setup=elapsed(tick)
                if setup>config['setup_seconds']:raise ScientificNegative('Fixed source/operator allowance exhausted')
                cr=dict(nodes=source['expected_geometry']['nodes'],cells=source['expected_geometry']['cells'],setup_seconds=setup,
                        geometry=geometry,fine=fine,coarse_checks=coarse_checks,mapping=mapping,assembly=assembly,K_relative=kp.cpu().tolist(),M_relative=mp.cpu().tolist(),M_symmetry=symmetry)
                state['case_records'][cad]=cr;state['phase']=cad+'/canonical_zero_PCG';persist()
                del w,K,Kref,M,Mref,diagonal,corners,cmap,fmap
                def apply(value):
                    full=torch.zeros_like(forces);full[:,free]=value
                    return parallel.apply(full)[:,free]
                def checkpoint(values):
                    temporary=folder/'last_state.npz.tmp'
                    with temporary.open('wb') as file:
                        np.savez(file,**{k:values[k].detach().cpu().numpy() for k in ['x','r','p','rz','active','iterations']},rhs=forces[:,free].cpu().numpy())
                    os.replace(temporary,folder/'last_state.npz')
                    state['running_group']=dict(CAD=cad,step=values['step'],seconds=values['seconds'],trace=values['trace']);persist()
                remaining=config['combined_solve_seconds']-state['combined_solve_seconds']
                if remaining<=10:raise ScientificNegative('Combined independent-solve allowance exhausted')
                st=(time.time(),time.monotonic())
                z,result=solve(apply,forces[:,free],candidate.apply_lanes,guard,checkpoint,seconds=min(config['group_seconds'],remaining))
                torch.cuda.synchronize();solve_seconds=elapsed(st);state['combined_solve_seconds']+=solve_seconds
                U=torch.stack([_restore_full(torch,value,inputs) for value in z]);endpoints=[]
                for i,load in enumerate(['ver','hor','tor']):
                    if not bool(torch.isfinite(U[i]).all()):raise ScientificNegative('Nonfinite canonical field')
                    residual=operator.apply(U[i])-forces[i]
                    relative=float(residual[free].norm()/forces[i,free].norm())
                    if not bool(torch.isfinite(residual).all()) or not np.isfinite(relative):
                        raise ScientificNegative('Nonfinite true equilibrium residual')
                    field=save(folder/load/'displacement_mm.npy',U[i]);state['saved_endpoints']+=1
                    endpoints.append(dict(load=load,field=field,true_relative_residual=relative))
                cr.update(solve_seconds=solve_seconds,solver=result,endpoints=endpoints);persist()
                if any(v['true_relative_residual']>1e-8 for v in endpoints):raise ScientificNegative('Fixed canonical group not converged')
                state['phase']=cad+'/full_gradient';persist();ft=(time.time(),time.monotonic());before=U.clone()
                derivative=qualify_derivative(np,torch,inputs,U,guard)
                features,qualification=recover_kinematic_inputs(torch,inputs['vertices_mm'],inputs['cells10'],U,arrays['weights'].sum(1),guard)
                if not torch.equal(before,U) or not torch.equal(features[...,:3],U):raise ArithmeticError('Input recovery changed U')
                item=save(folder/'kinematic_inputs.npy',features);gs=elapsed(ft);state['gradient_seconds']+=gs
                if state['gradient_seconds']>config['gradient_seconds']:raise ScientificNegative('Fixed gradient allowance exhausted')
                cr.update(gradient_seconds=gs,derivative=derivative,qualification=qualification)
                state['features'][cad]=item;state['case_records'][cad]=cr;persist()
                del inputs,forces,operator,arrays,scalar,parallel,candidate,chol,z,U,before,features,P,PT,Ac,cd
                gc.collect();torch.cuda.empty_cache()
            if len(state['features'])!=20 or state['saved_endpoints']!=60:raise ValueError('Fixed final Cartesian input set differs')
            index=dict(CADs=catalog['CADs'],scientific_group='FINAL',canonical_loads=['ver','hor','tor'],features=state['features'],complete_CADs=20,complete_loads=60,GT_reads=0)
            (output/'complete_inputs.json').write_text(json.dumps(index,indent=2)+'\n')
            state.update(status='completed',phase='terminal',complete_inputs=True)
    except Exception as error:
        if type(error).__name__=='ScientificNegative' or isinstance(error,ValueError) and str(error).startswith(('Original curved','Original nodes','Actual P2')):
            state.update(status='completed',phase='scientific_input_negative',reason=str(error),complete_inputs=False)
        else:
            state.update(status='failed',phase='technical_failure',error=type(error).__name__+': '+str(error));raise
    finally:
        state.update(active=False,closed=True);persist()
        (output/'summary.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['config','source','output','lock']:p.add_argument('--'+name,required=True)
    run(p.parse_args())
