"""Synthetic CPU chunk-chain vs complete monolithic gradient, two fixed models."""
import argparse


def run():
    import time
    began = time.monotonic()
    import torch
    from geobc import available_cad_pilot_data as d
    from geobc.coarse_transolver_v19 import CoarseTransolver
    from geobc.coarse_fno_control_v22 import CoarseFNOControl
    from geobc.encoded_objective_v28 import full_encoded_backward
    torch.set_num_threads(4); torch.manual_seed(2026)
    n = 33337
    grid = torch.rand(1, 29, 64, 64, 64); grid[:, 0] = 1.
    case = dict(key='synthetic/CPU', scientific_group='TRAIN', grid=grid, xyz=torch.rand(n, 3),
                features=torch.rand(n, 25), cloud=torch.rand(1, 1024, 3), mask=torch.ones(1, 1024, dtype=torch.bool))
    target = torch.randn(n, 4); results = {}
    factories = {'transolver_encoded': lambda: CoarseTransolver('transolver', device='cpu'),
                 'FNO16_encoded': lambda: CoarseFNOControl(device='cpu')}
    for name, build in factories.items():
        chunked = build().train(); direct = build().train()
        assert all(torch.equal(v, direct.state_dict()[k]) for k, v in chunked.state_dict().items())
        loss, leaves = full_encoded_backward(chunked, case, target, ['synthetic/CPU'], lambda: None)
        context, latent = direct.encode_case(case)
        full_loss = (direct.query(case, 0, n, context, latent)-target).square().mean()
        full_loss.backward()
        assert abs(loss-float(full_loss.detach())) < 2e-6
        groups = dict(encoder=0., operator=0., head=0.); max_abs = 0.
        expected = dict(direct.named_parameters())
        for key, p in chunked.named_parameters():
            q = expected[key]
            assert p.grad is not None and q.grad is not None and torch.isfinite(p.grad).all()
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=3e-4)
            max_abs = max(max_abs, float((p.grad-q.grad).abs().max()))
            groups[key.split('.')[0]] += float(p.grad.double().square().sum())
        assert all(v > 0 for v in groups.values()) and leaves['geometry_context_L2'] > 0 and leaves['operator_latent_L2'] > 0
        results[name] = dict(parameter_count=chunked.profile()['parameter_count'], initial_states_equal=True,
                             loss=loss, monolithic_loss=float(full_loss.detach()), max_gradient_absolute_difference=max_abs,
                             all_named_gradients_close=True, groups_L2={k: v**.5 for k, v in groups.items()}, leaf_gradients=leaves)
    proof = dict(status='actual_synthetic_encoded4_two_model_chain_CPU_passed_not_GPU',
        source=d.identity('geobc/encoded_objective_v28.py', hashed=True), synthetic_nodes=n, chunks=[32768, n-32768],
        comparison_tolerance=dict(atol=2e-6, rtol=3e-4), results=results, real_numeric_GT_reads=0,
        GPU_calls=0, optimizer_updates=0, CPU_seconds=time.monotonic()-began)
    d.write('notes/geobc_encoded_objective_v28_CPU_fixture.json', proof)
    print(dict(status=proof['status'], CPU_seconds=proof['CPU_seconds'], max_diffs={k: v['max_gradient_absolute_difference'] for k, v in results.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    if parser.parse_args().check: run()
