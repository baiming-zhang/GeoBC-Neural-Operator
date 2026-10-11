"""Synthetic full-node latent-gradient proof; no real PT, GT, GPU or Adam."""
import argparse


def run():
    import time
    began = time.monotonic()
    import torch
    from geobc import available_cad_pilot_data as d
    from geobc.node_latent_teacher_v35 import (freeze_teacher_geometry_head,
        complete_teacher_node_norm, full_teacher_node_backward, original_node_latent, frozen_group_digests)
    from geobc.coarse_transolver_v19 import CoarseTransolver
    from geobc.coarse_fno_control_v22 import CoarseFNOControl
    torch.set_num_threads(4); torch.manual_seed(2026)
    teacher = CoarseFNOControl(device='cpu').eval().requires_grad_(False)
    # Explicit synthetic teacher perturbation, not a trained checkpoint.
    with torch.no_grad():
        for parameter in teacher.parameters(): parameter.add_(.002)
    n = 33337
    grid = torch.rand(1, 29, 64, 64, 64); grid[:, 0] = 1.
    case = dict(key='synthetic/CPU', scientific_group='TRAIN', grid=grid,
        xyz=torch.rand(n, 3), features=torch.rand(n, 25), cloud=torch.rand(1, 1024, 3),
        mask=torch.ones(1, 1024, dtype=torch.bool))
    with torch.no_grad(): _, target = teacher.encode_case(case)
    norms = complete_teacher_node_norm(target, case['xyz'], lambda: None)
    with torch.no_grad():
        direct_norm = float(original_node_latent(target, case['xyz']).double().square().sum())
    assert abs(norms.squared_norm-direct_norm) <= 1e-12*direct_norm
    results = {}
    for method, build in [('transolver', lambda: CoarseTransolver('transolver', device='cpu')),
                          ('FNO16', lambda: CoarseFNOControl(device='cpu'))]:
        chunked, direct = build().train(), build().train()
        receipt = freeze_teacher_geometry_head(chunked, teacher.state_dict())
        other = freeze_teacher_geometry_head(direct, teacher.state_dict())
        assert receipt['transferred_model_after'] == other['transferred_model_after']
        before = frozen_group_digests(chunked)
        loss, leaves = full_teacher_node_backward(chunked, case, target, norms, ['synthetic/CPU'], lambda: None)
        context, latent = direct.encode_case(case)
        assert not context.requires_grad
        full = (original_node_latent(latent, case['xyz']).double()-
                original_node_latent(target, case['xyz']).double()).square().sum()/norms.squared_norm
        full.backward()
        assert abs(loss-float(full.detach())) < 1e-10
        expected = dict(direct.named_parameters()); total = 0.; maximum = 0.; checked = 0
        for name, parameter in chunked.named_parameters():
            other = expected[name]
            if not name.startswith('operator.'):
                assert not parameter.requires_grad and parameter.grad is None and other.grad is None
                continue
            assert parameter.grad is not None and other.grad is not None and torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-6, rtol=3e-4)
            total += float(parameter.grad.double().square().sum()); checked += 1
            maximum = max(maximum, float((parameter.grad-other.grad).abs().max()))
        assert total > 0 and leaves['operator_latent_gradient_L2'] > 0
        assert frozen_group_digests(chunked) == before
        results[method] = dict(transfer=receipt, frozen_groups_unchanged=True,
            frozen_groups_gradients_absent=True, all_named_operator_gradients_close=True,
            checked_operator_parameters=checked, max_gradient_absolute_difference=maximum,
            operator_gradient_L2=total**.5, normalized_teacher_loss=loss, leaf_receipt=leaves)
    try:
        complete_teacher_node_norm(torch.zeros_like(target), case['xyz'], lambda: None)
    except ValueError: zero_teacher_rejected = True
    else: raise AssertionError('Zero teacher norm must be undefined, not rescued')
    assert all(p.grad is None for p in teacher.parameters())
    proof = dict(status='actual_synthetic_node_latent_CPU_passed_not_GPU',
        source=d.identity('geobc/node_latent_teacher_v35.py', hashed=True),
        synthetic_teacher_not_trained=True, synthetic_nodes=n, chunks=[32768, 569],
        zero_teacher_norm_rejected=zero_teacher_rejected, teacher_gradients_absent=True,
        results=results, real_numeric_GT_reads=0, PT_reads=0, GPU_calls=0,
        optimizer_updates=0, CPU_seconds=time.monotonic()-began)
    d.write('notes/geobc_node_latent_teacher_v35_CPU_fixture.json', proof)
    print(dict(status=proof['status'], CPU_seconds=proof['CPU_seconds']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    if parser.parse_args().check: run()
