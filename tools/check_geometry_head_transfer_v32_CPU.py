"""Synthetic transfer/whole-gradient proof; no real checkpoint or target I/O."""
import argparse


def run():
    import time
    began=time.monotonic()
    import torch
    from geobc import available_cad_pilot_data as d
    from geobc.geometry_head_transfer_v32 import transfer_geometry_head
    from geobc.coarse_transolver_v19 import CoarseTransolver,full_physical_backward
    from geobc.coarse_fno_control_v22 import CoarseFNOControl
    from geobc.physical_field_loss import PhysicalScale,norms_from_complete_physical_target,physical_relative_squared_loss
    torch.set_num_threads(4);torch.manual_seed(2026)
    teacher=CoarseFNOControl(device='cpu')
    with torch.no_grad():
        for group in ('encoder','head'):
            for parameter in getattr(teacher,group).parameters():parameter.add_(.002)
    state=teacher.state_dict();n=33337
    grid=torch.rand(1,29,64,64,64);grid[:,0]=1.
    case=dict(key='synthetic/CPU',scientific_group='TRAIN',grid=grid,xyz=torch.rand(n,3),features=torch.rand(n,25),
        cloud=torch.rand(1,1024,3),mask=torch.ones(1,1024,dtype=torch.bool))
    target=torch.randn(n,4);target[:,3]*=20.
    scale=PhysicalScale((.2,-.1,.3,.15),(1.1,.8,1.3,.7),20.);norms=norms_from_complete_physical_target(target)
    results={}
    for method,build in [('transolver_headwarm',lambda:CoarseTransolver('transolver',device='cpu')),
                         ('FNO16_headwarm_reset',lambda:CoarseFNOControl(device='cpu'))]:
        chunked,direct=build().train(),build().train()
        receipt=transfer_geometry_head(chunked,state);other=transfer_geometry_head(direct,state)
        assert receipt['transferred_model_after']==other['transferred_model_after']
        assert receipt['fresh_model_before']!=receipt['transferred_model_after']
        loss,leaves=full_physical_backward(chunked,case,target,scale,norms,['synthetic/CPU'],lambda:None)
        context,latent=direct.encode_case(case)
        full=physical_relative_squared_loss(direct.query(case,0,n,context,latent),target,scale,norms);full.backward()
        assert abs(loss-float(full.detach()))<1e-9
        expected=dict(direct.named_parameters());groups=dict(encoder=0.,operator=0.,head=0.);maximum=0.
        for name,parameter in chunked.named_parameters():
            other=expected[name]
            assert parameter.grad is not None and other.grad is not None and torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad,other.grad,atol=2e-6,rtol=3e-4)
            maximum=max(maximum,float((parameter.grad-other.grad).abs().max()))
            groups[name.split('.')[0]]+=float(parameter.grad.double().square().sum())
        assert all(v>0 for v in groups.values()) and leaves['geometry_context_L2']>0 and leaves['operator_latent_L2']>0
        results[method]=dict(transfer=receipt,all_named_gradients_close=True,max_gradient_absolute_difference=maximum,
            group_gradients_L2={k:v**.5 for k,v in groups.items()},leaf_gradients=leaves)
    proof=dict(status='actual_synthetic_geometry_head_transfer_CPU_passed_not_GPU',
        source=d.identity('geobc/geometry_head_transfer_v32.py',hashed=True),synthetic_teacher_not_trained=True,
        synthetic_nodes=n,chunks=[32768,569],results=results,real_numeric_GT_reads=0,PT_reads=0,GPU_calls=0,
        optimizer_updates=0,CPU_seconds=time.monotonic()-began)
    d.write('notes/geobc_geometry_head_transfer_v32_CPU_fixture.json',proof)
    print(dict(status=proof['status'],CPU_seconds=proof['CPU_seconds']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    if parser.parse_args().check:run()
