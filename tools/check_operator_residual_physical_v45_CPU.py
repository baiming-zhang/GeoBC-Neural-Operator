"""Synthetic CPU proof of whole-node physical chunk gradients; no GT/PT/CUDA."""
import argparse
import time


def run():
    began = time.monotonic()
    import torch
    from torch import nn
    from geobc import available_cad_pilot_data as d
    from geobc.common_grid_solver import CommonGridSolver
    from geobc.operator_residual_v44 import FrozenOperatorResidual
    from geobc.residual_physical_training_v45 import full_residual_case_backward
    from geobc.physical_field_loss import PhysicalScale, decode_physical, norms_from_complete_physical_target, physical_relative_squared_loss
    from geobc.geometry_head_transfer_v32 import state_digest
    from geobc.solver_selection_data import _stencil
    from geobc.tcnn_fields import TCNNFieldConfig
    torch.set_num_threads(4); torch.manual_seed(2026)
    n = 32769; xyz = torch.rand(n,3); native = torch.randn(n,25)*.1
    context = torch.randn(1,32)*.1; latent = torch.randn(1,16,64,64,64)*.1
    indices,weights = _stencil(xyz,64)
    case = dict(key='synthetic/ver',scientific_group='TRAIN',xyz=xyz,features=native,query_indices=indices,query_weights=weights)
    scale = PhysicalScale((0.,0.,0.,0.),(1.,1.,1.,1.),2.)
    class Fixture(nn.Module):
        def __init__(self):
            super().__init__(); self.config = TCNNFieldConfig(out_channels=4,input_mode='both',condition_dim=73,seed=2026)
            self.net = nn.Sequential(nn.Linear(76,16),nn.Tanh(),nn.Linear(16,4))
        def forward(self,x,c): return self.net(torch.cat((x,c),dim=-1))
    rows = {}
    for mode in ('mlp','hash'):
        teacher = CommonGridSolver('FNO9',2026)
        model = FrozenOperatorResidual(teacher,mode,device='cpu',injected_hash_decoder=Fixture() if mode=='hash' else None).train()
        initial = {k:state_digest(getattr(model,k).state_dict()) for k in ('teacher','reference')}
        with torch.no_grad():
            baseline = teacher.query(case,0,n,context,latent)
            target = decode_physical(baseline+torch.randn_like(baseline)*.1,scale).float()
        norms = norms_from_complete_physical_target(target)
        prediction = model.query(case,0,n,context,latent)
        exact_loss = physical_relative_squared_loss(prediction,target,scale,norms)
        exact_loss.backward()
        gradients = {k:p.grad.detach().clone() for k,p in model.decoder.named_parameters()}
        expected_loss = float(exact_loss.detach()); model.zero_grad(set_to_none=True)
        actual = full_residual_case_backward(model,case,target,scale,norms,context,latent,[case['key']])
        assert abs(actual-expected_loss) < 1e-8
        worst = 0.
        for k,p in model.decoder.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert torch.allclose(p.grad,gradients[k],atol=2e-6,rtol=2e-5),k
            worst = max(worst,float((p.grad-gradients[k]).abs().max()))
        assert sum(float(p.grad.double().square().sum()) for p in model.decoder.parameters()) > 0
        for k in ('teacher','reference'):
            assert all(p.grad is None for p in getattr(model,k).parameters()) and initial[k] == state_digest(getattr(model,k).state_dict())
        assert context.grad is None and latent.grad is None
        rows[mode] = dict(nodes=n,chunks=[32768,1],physical_loss_agrees=True,maximum_gradient_difference=worst,
            full_relative_norms_preserved=True,all_mutable_gradients_present_finite=True,frozen_gradients_none=True,frozen_states_unchanged=True,
            native_TCNN_calls=0,hash_scope='Torch interface mock, not TCNN' if mode=='hash' else 'ordinary Torch MLP')
        del model,teacher,prediction,target,gradients,exact_loss,baseline
    assert not torch.cuda.is_initialized()
    out = dict(status='actual_synthetic_CPU_physical_chunk_loss_gradient_passed_not_native_GPU',
        CPU_seconds=time.monotonic()-began,modes=rows,trained_PT_reads=0,dataset_GT_reads=0,GPU_calls=0,optimizer_updates=0,
        source=d.identity('geobc/residual_physical_training_v45.py',hashed=True),checker=d.identity('tools/check_operator_residual_physical_v45_CPU.py',hashed=True))
    assert not d.path('notes/geobc_operator_residual_v45_CPU_fixture.json').exists()
    d.write('notes/geobc_operator_residual_v45_CPU_fixture.json',out)
    print(dict(status=out['status'],CPU_seconds=out['CPU_seconds']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check',action='store_true'); args = parser.parse_args()
    if args.check: run()
