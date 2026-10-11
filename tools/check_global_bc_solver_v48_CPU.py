"""Synthetic whole-operator API/chain-gradient fixture; not real/native GPU."""
import argparse
import time


def run():
    started = time.monotonic()
    import torch
    from geobc import available_cad_pilot_data as d
    from geobc.global_bc_solver_v48 import GlobalBCSolver,attach_descriptor,METHODS
    from geobc.common_grid_solver import CommonGridSolver
    from geobc.coarse_transolver_v19 import CoarseTransolver,full_physical_backward
    from geobc.resolution_fno_v41 import original_input_grid
    from geobc.physical_field_loss import PhysicalScale,decode_physical,norms_from_complete_physical_target
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest'); torch.manual_seed(2026)
    n = 257; xyz = torch.rand(n,3); f = torch.zeros(n,25)
    f[:,:7] = torch.randint(0,2,(n,7)).float(); f[:,7:16] = torch.tensor([0.,0.,1.,0.,0.,0.,1.,.3,1.]); f[:,24] = .9
    f[:,16:19] = xyz-torch.tensor([.6,.3,.9]); anchors = torch.rand(4,3)
    f[:,19:23] = torch.linalg.vector_norm(xyz[:,None,:]-anchors[None,:,:],dim=-1)
    case = dict(key='synthetic/ver',scientific_group='TRAIN',xyz=xyz,features=f,cloud=torch.rand(1,1024,3),mask=torch.ones(1,1024,dtype=torch.bool))
    grid,indices,weights = original_input_grid(case,64)
    case.update(grid=grid,query_indices=indices,query_weights=weights); attach_descriptor(case)
    scale = PhysicalScale((0.,0.,0.,0.),(1.,1.,1.,1.),2.)
    expected = {'FNO9_GLOBAL_BC':'e3c06ebfb126495173ae11b94fb3ef0cf87392976d56ba36238358844dd516da',
                'R16_TRANSOLVER_GLOBAL_BC':'13d31df15c487a1240b8f1f81f0a7ccf7a15a73f32cbbb29fb3dff34632e5abb'}
    rows = {}
    for method in METHODS:
        model = GlobalBCSolver(method,2026,'cpu').train()
        old = (CommonGridSolver('FNO9',2026) if method == METHODS[0] else CoarseTransolver('transolver',2026,'cpu')).train()
        assert model.original_backbone_digest() == expected[method]
        reference = model.reference_digest()
        with torch.no_grad():
            oc,ol = old.encode_case(case); nc,nl = model.encode_case(case)
            assert torch.equal(oc,nc) and torch.equal(ol,nl)
            baseline = old.query(case,0,n,oc,ol); prediction = model.query(case,0,n,nc,nl)
            assert torch.equal(baseline,prediction)
            truth = decode_physical(baseline+torch.randn_like(baseline)*.03,scale).float()
        bad = dict(case,global_bc_source_key='other/ver')
        try: model.encode_case(bad)
        except ValueError: pass
        else: raise RuntimeError('Stale per-case descriptor accepted')
        norms = norms_from_complete_physical_target(truth)
        loss,leaf = full_physical_backward(model,case,truth,scale,norms,[case['key']],lambda:None)
        energy = {k:0. for k in ('geometry_encoder','global_BC_branch','head','operator')}
        for name,p in model.named_parameters():
            group = model.gradient_group(name)
            if group == 'frozen_BC_reference':
                assert not p.requires_grad and p.grad is None; continue
            assert p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all(),name
            energy[group] += float(p.grad.detach().abs().double().square().sum())
        assert all(v > 0 for v in energy.values()) and leaf['geometry_context_L2'] > 0 and leaf['operator_latent_L2'] > 0
        assert model.reference_digest() == reference and model.original_backbone_digest() == expected[method]
        rows[method] = dict(profile=model.profile(),old_backbone_initial_exact=True,initial_context_latent_prediction_exact_old=True,
            all_four_trainable_groups_L2={k:v**.5 for k,v in energy.items()},frozen_reference_gradients_none=True,
            current_case_descriptor_required=True,physical_loss=loss,whole_synthetic_nodes=n,leaf_gradients=leaf)
        del model,old,oc,ol,nc,nl,baseline,prediction,truth
    assert not torch.cuda.is_initialized()
    out = dict(status='actual_synthetic_CPU_two_globalBC_operators_zero_output_joint_physical_gradient_passed_not_nativeGPU',
        CPU_seconds=time.monotonic()-started,methods=rows,GPU_calls=0,dataset_GT_PT_numeric_reads=0,optimizer_updates=0,
        source=d.identity('geobc/global_bc_solver_v48.py',hashed=True),checker=d.identity('tools/check_global_bc_solver_v48_CPU.py',hashed=True))
    assert not d.path('notes/geobc_global_bc_solver_v48_CPU_fixture.json').exists()
    d.write('notes/geobc_global_bc_solver_v48_CPU_fixture.json',out); print({'status':out['status'],'CPU_seconds':out['CPU_seconds']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check',action='store_true');args = parser.parse_args()
    if args.check: run()
