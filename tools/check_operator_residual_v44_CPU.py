"""Synthetic CPU interface: zero residual keeps teacher and still learns.

No trained checkpoint, dataset labels, optimizer, native TCNN or GPU calls.
"""
import argparse
import time


def run():
    began=time.monotonic()
    import torch
    from torch import nn
    from geobc import available_cad_pilot_data as d
    from geobc.common_grid_solver import CommonGridSolver
    from geobc.operator_residual_v44 import FrozenOperatorResidual
    from geobc.tcnn_fields import TCNNFieldConfig
    from geobc.solver_selection_data import _stencil
    from geobc.geometry_head_transfer_v32 import state_digest
    torch.set_num_threads(4);torch.manual_seed(234)
    xyz=torch.rand(257,3)*.5+.1;native=torch.rand(257,25);context=torch.rand(1,32);latent=torch.rand(1,16,64,64,64)
    indices,weights=_stencil(xyz,64);case=dict(xyz=xyz,features=native,query_indices=indices,query_weights=weights)

    class Fixture(nn.Module):
        def __init__(self):
            super().__init__();self.config=TCNNFieldConfig(out_channels=4,input_mode='both',condition_dim=73,seed=2026)
            self.net=nn.Sequential(nn.Linear(76,16),nn.Tanh(),nn.Linear(16,4))
        def forward(self,x,c):return self.net(torch.cat((x,c),dim=-1))

    records={}
    for mode in ('mlp','hash'):
        teacher=CommonGridSolver('FNO9',2026)
        model=FrozenOperatorResidual(teacher,mode,device='cpu',injected_hash_decoder=Fixture() if mode=='hash' else None).train()
        assert not model.teacher.training and not model.reference.training
        before={k:state_digest(getattr(model,k).state_dict()) for k in ('teacher','reference')}
        baseline=torch.cat([teacher.query(case,i,min(i+64,257),context,latent) for i in range(0,257,64)])
        pred=model.query_points(xyz,native,context,latent,64)
        assert torch.equal(pred,baseline)
        cached=torch.cat([model.query(case,i,min(i+64,257),context,latent) for i in range(0,257,64)])
        assert torch.equal(cached,pred)
        del cached
        target=baseline.detach()+torch.randn_like(baseline)*.01
        (pred-target).square().mean().backward()
        energy=0.
        for parameter in model.decoder.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            energy+=float(parameter.grad.double().square().sum())
        assert energy>0
        for k in ('teacher','reference'):
            assert all(parameter.grad is None for parameter in getattr(model,k).parameters())
            assert before[k]==state_digest(getattr(model,k).state_dict())
        records[mode]=dict(initial_teacher_exact=True,trainable_first_gradient_L2=energy**.5,frozen_gradients_none=True,
            frozen_hashes_unchanged=True,cached_nodes_and_explicit_points_equal=True,profile=model.profile(),nodes=257,query_chunk=64,tail=1)
        del model,teacher,pred,baseline,target
    assert not torch.cuda.is_initialized()
    out=dict(status='actual_synthetic_CPU_zero_residual_and_first_gradient_passed_not_native_GPU',CPU_seconds=time.monotonic()-began,
        source=d.identity('geobc/operator_residual_v44.py',hashed=True),checker=d.identity('tools/check_operator_residual_v44_CPU.py',hashed=True),
        modes=records,teacher='Untrained synthetic CPU FNO9, not actual trained5 checkpoint',hash='Injected Torch interface fixture, not nativeTCNN',
        trained_PT_reads=0,dataset_GT_reads=0,GPU_calls=0,optimizer_updates=0)
    d.write('notes/geobc_operator_residual_v44_CPU_fixture.json',out)
    print({'status':out['status'],'CPU_seconds':out['CPU_seconds']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:run()
    else:print('{"status":"source_only","GT_PT_GPU_calls":0}')
