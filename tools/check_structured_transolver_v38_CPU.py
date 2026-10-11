"""Exact official/checkpointed structured3D operator proof, synthetic CPU only."""
import argparse


def run():
    import time
    began=time.monotonic()
    import torch
    from geobc import available_cad_pilot_data as d
    from geobc.structured_transolver_v38 import CoarseStructuredTransolver,SOURCE,SOURCE_SHA256
    from geobc.coarse_transolver_v19 import CoarseTransolver
    from geobc.geometry_head_transfer_v32 import state_digest
    torch.set_num_threads(4);torch.manual_seed(2026)
    chunked=CoarseStructuredTransolver(device='cpu').train()
    native=CoarseStructuredTransolver(device='cpu').train()
    assert state_digest(chunked.state_dict())==state_digest(native.state_dict())
    original=CoarseTransolver('transolver',device='cpu')
    assert all(state_digest(getattr(chunked,g).state_dict())==state_digest(getattr(original,g).state_dict()) for g in ['encoder','head'])
    native.operator.checkpoint_blocks=False
    centers=(4*torch.arange(16)+1.5)/63
    xyz=torch.stack(torch.meshgrid(centers,centers,centers,indexing='ij'),-1).reshape(1,4096,3)
    value=torch.cat((xyz,torch.rand(1,4096,58)),dim=-1).requires_grad_(True)
    other=value.detach().clone().requires_grad_(True)
    out=chunked.operator(value);expected=native.operator(other)
    torch.testing.assert_close(out,expected,atol=2e-6,rtol=3e-4)
    target=torch.randn_like(out)
    loss=(out-target).double().square().mean();ref=(expected-target).double().square().mean()
    loss.backward();ref.backward()
    torch.testing.assert_close(value.grad,other.grad,atol=2e-6,rtol=3e-4)
    direct=dict(native.operator.named_parameters());unused=[];maximum=0.;energy=0.
    for name,p in chunked.operator.named_parameters():
        q=direct[name]
        if name=='backbone.placeholder':
            assert p.grad is None and q.grad is None;unused.append(name);continue
        assert p.grad is not None and q.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=3e-4)
        maximum=max(maximum,float((p.grad-q.grad).abs().max()))
        energy+=float(p.grad.double().square().sum())
    assert unused==['backbone.placeholder'] and energy>0
    with torch.no_grad():
        chunked.eval();native.eval()
        torch.testing.assert_close(chunked.operator(value),native.operator(other),atol=2e-6,rtol=3e-4)
    proof=dict(status='actual_synthetic_structured3D_CPU_equivalence_passed_not_GPU',source=d.identity('geobc/structured_transolver_v38.py',hashed=True),
        upstream_source=d.identity(SOURCE,hashed=True),upstream_SHA256=SOURCE_SHA256,profile=chunked.profile(),
        canonical_R16_tokens=4096,XYZ_order='C-contiguous X/Y/Z grid; true pooled block centers',
        full_named_operator_gradients_close=True,allowed_unused_parameters=unused,placeholder_correctly_not_added_for_features=True,
        operator_gradient_L2=energy**.5,max_gradient_absolute_difference=maximum,input_gradients_close=True,
        encoder_head_initial_SHA_matches_original=True,full_node_physical_chain_or_GPU_qualified_here=False,
        actual_GT_reads=0,PT_reads=0,GPU_calls=0,optimizer_updates=0,CPU_seconds=time.monotonic()-began)
    d.write('notes/geobc_structured_TS_v38_CPU_fixture.json',proof)
    print(dict(status=proof['status'],CPU_seconds=proof['CPU_seconds'],model_parameters=proof['profile']['parameter_count']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    if parser.parse_args().check:run()
