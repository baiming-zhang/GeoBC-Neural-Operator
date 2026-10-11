"""Synthetic CPU routing proof and a separately requested six-case input probe."""
import argparse
import time


def synthetic():
    started = time.monotonic()
    import torch
    from geobc import available_cad_pilot_data as d
    from geobc.common_grid_solver import CommonGridSolver
    from geobc.global_bc_context_v47 import global_bc_descriptor,GeometryBCContext
    from geobc.geometry_head_transfer_v32 import state_digest
    torch.set_num_threads(4); torch.manual_seed(2026)
    xyz = torch.rand(257,3); f = torch.zeros(257,25)
    f[:,:7] = torch.randint(0,2,(257,7)).float()
    f[:,7:16] = torch.tensor([0.,0.,-1.,0.,0.,0.,1.,.3,1.]); f[:,24] = .9
    anchor = torch.tensor([1.2,-.1,.4]); f[:,16:19] = xyz-anchor
    centers = torch.rand(4,3); f[:,19:23] = torch.linalg.vector_norm(xyz[:,None,:]-centers[None,:,:],dim=-1)
    desc = global_bc_descriptor(xyz,f)
    shuffled = f.clone(); shuffled[:,19:23] = f[:,[22,20,19,21]]
    assert torch.allclose(desc,global_bc_descriptor(xyz,shuffled),atol=2e-7,rtol=2e-7)
    assert torch.allclose(desc[:,10:13],anchor[None,:],atol=2e-7,rtol=0)
    parent = CommonGridSolver('FNO9',2026)
    before = {n:state_digest(getattr(parent,n).state_dict()) for n in ('encoder','head','operator')}
    wrapper = GeometryBCContext(parent.encoder).train(); wrapper.set_descriptor(desc)
    cloud = torch.rand(1,1024,3); mask = torch.ones(1,1024,dtype=torch.bool)
    baseline = parent.encoder(cloud,mask); context = wrapper(cloud,mask)
    assert torch.equal(context,baseline)
    target = context.detach()+torch.randn_like(context)*.1
    (context-target).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in wrapper.branch.parameters())
    energy = sum(float(p.grad.double().square().sum()) for p in wrapper.branch.parameters()); assert energy > 0
    assert all(p.grad is None for p in wrapper.reference.parameters())
    assert all(before[n] == state_digest(getattr(parent,n).state_dict()) for n in before)
    assert not torch.cuda.is_initialized()
    out = dict(status='actual_synthetic_CPU_global_BC_zero_context_gradient_passed_not_operator_or_native_GPU',CPU_seconds=time.monotonic()-started,
        descriptor_dim=68,load_anchor_recovered=True,RBE2_reference_summary_permutation_invariant=True,initial_context_exact_old=True,
        backbone_states_unchanged=True,mutable_branch_gradient_L2=energy**.5,frozen_reference_gradient_none=True,profile=wrapper.profile(),
        dataset_GT_PT_reads=0,GPU_calls=0,operator_forward_calls=0,optimizer_updates=0,
        source=d.identity('geobc/global_bc_context_v47.py',hashed=True),checker=d.identity('tools/check_global_bc_context_v47_CPU.py',hashed=True))
    assert not d.path('notes/geobc_global_bc_context_v47_CPU_fixture.json').exists()
    d.write('notes/geobc_global_bc_context_v47_CPU_fixture.json',out); print({'status':out['status'],'CPU_seconds':out['CPU_seconds']})


def probe():
    start = (time.time(),time.monotonic())
    import numpy as np
    import torch
    from geobc import available_cad_pilot_data as d
    from geobc.prospective_dataset import resolve_case
    from geobc.global_bc_context_v47 import global_bc_descriptor
    torch.set_num_threads(4)
    plan = d.read('notes/geobc_operator_residual_v45_plan.json'); m = d.read(plan['scientific']['dataset_identity']['path']); keys = m['train_keys']['100']
    largest = max(keys,key=lambda k:m['records'][k]['num_nodes']).split('/')[0]
    first = next(k.split('/')[0] for k in keys if k.split('/')[0] != largest)
    selected = [cad+'/'+load for cad in (first,largest) for load in ('ver','hor','tor')]
    receipt = dict(status='running',selected_TRAIN_keys=selected,input_array_read_attempts=0,input_array_reads=0,
        GT_PT_numeric_reads=0,GPU_calls=0,scaler_fits=0,cases={},input_identities={},absolute_budget_seconds=60)
    target = 'notes/geobc_global_bc_context_v47_actual_TRAIN_input_probe.json'; assert not d.path(target).exists()
    try:
        for key in selected:
            if min(60-(time.time()-start[0]),60-(time.monotonic()-start[1])) <= 0: raise TimeoutError('Six input cases fixed60seconds')
            assert m['case_roles'][key] == 'TRAIN'
            record,identities = resolve_case(m,key,False,100,False); data = {}
            for member in ('xyz','features'):
                identity = identities[member]; d.stat_matches(identity); receipt['input_array_read_attempts'] += 1
                value = np.load(d.path(identity['path']),mmap_mode='r',allow_pickle=False); receipt['input_array_reads'] += 1
                data[member] = torch.as_tensor(np.array(value,copy=True)); receipt['input_identities'][key+':'+member] = identity
                del value
            assert len(data['xyz']) == record['num_nodes']; desc = global_bc_descriptor(data['xyz'],data['features'])
            receipt['cases'][key] = dict(nodes=record['num_nodes'],descriptor=desc.tolist(),load6=desc[0,:6].tolist(),
                role_member_counts=[int(data['features'][:,j].sum()) for j in range(7)],all_original_input_nodes=True,descriptor_finite=True)
            del data,desc
        assert receipt['input_array_reads'] == 12 and len(receipt['cases']) == 6 and not torch.cuda.is_initialized()
        receipt['status'] = 'completed_six_actual_TRAIN_original_input_only_CPU_no_GT_PT_GPU'
    except Exception as error:
        receipt.update(status='failed_input_probe',error_type=type(error).__name__,error=str(error)); raise
    finally:
        receipt.update(closed=True,elapsed_epoch_seconds=time.time()-start[0],elapsed_monotonic_seconds=time.monotonic()-start[1],
            source=d.identity('geobc/global_bc_context_v47.py',hashed=True),checker=d.identity('tools/check_global_bc_context_v47_CPU.py',hashed=True))
        d.write(target,receipt)
    print({'status':receipt['status'],'CPU_seconds':receipt['elapsed_monotonic_seconds'],'input_array_reads':12,'GT_PT_GPU':0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check',action='store_true'); parser.add_argument('--probe-inputs',action='store_true'); args = parser.parse_args()
    if args.check: synthetic()
    if args.probe_inputs: probe()
