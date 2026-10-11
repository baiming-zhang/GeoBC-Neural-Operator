"""Synthetic unit/label/chunk-chain qualification, no real arrays/GT/GPU."""
import argparse


def run():
    import time
    began = time.monotonic()
    import torch
    from geobc import available_cad_pilot_data as d
    from geobc.response_gain_v30 import response_gains, scale_encoded_response, ResponseGainTransolver, ResponseGainFNO
    from geobc.coarse_transolver_v19 import CoarseTransolver, full_physical_backward
    from geobc.coarse_fno_control_v22 import CoarseFNOControl
    from geobc.physical_field_loss import PhysicalScale, decode_physical, norms_from_complete_physical_target, physical_relative_squared_loss
    torch.set_num_threads(4); torch.manual_seed(2026)
    scale = PhysicalScale((.2, -.1, .3, .15), (1.1, .8, 1.3, .7), 20.)
    n = 33337
    features = torch.rand(n, 25)
    features[:, 7:14] = torch.tensor([.4, -.8, .2, .1, .3, -.2, 1.2])
    features[:, 24] = .7
    gains = response_gains(features)
    doubled = features.clone(); doubled[:, 7:13] *= 2
    torch.testing.assert_close(response_gains(doubled), 2*gains)
    # Geometric similarity: F held, M scaled with L; doubling L gives U/2,S/4.
    larger = features.clone(); larger[:, 24] *= 2; larger[:, 10:13] *= 2
    torch.testing.assert_close(response_gains(larger), gains*torch.tensor([.5, .25], dtype=torch.float64))
    stiffer = features.clone(); stiffer[:, 13] *= 2
    torch.testing.assert_close(response_gains(stiffer), gains*torch.tensor([.5, 1.], dtype=torch.float64))
    encoded = torch.randn(79, 4)
    physical = decode_physical(encoded, scale)
    scaled = decode_physical(scale_encoded_response(encoded, gains, scale), scale)
    torch.testing.assert_close(scaled, physical*torch.tensor([gains[0]]*3+[gains[1]]), atol=1e-11, rtol=1e-11)
    assert torch.equal(physical.sign(), scaled.sign())
    invalid = features.clone(); invalid[:, 7:13] = 0.
    try: response_gains(invalid)
    except ValueError: pass
    else: raise AssertionError('Zero loads must be undefined, not epsilon rescued')
    nonconstant = features.clone(); nonconstant[-1, 24] += .01
    try: response_gains(nonconstant)
    except ValueError: pass
    else: raise AssertionError('Nonconstant global inputs must be rejected')
    grid = torch.rand(1, 29, 64, 64, 64); grid[:, 0] = 1.
    case = dict(key='synthetic/CPU', scientific_group='TRAIN', grid=grid, xyz=torch.rand(n, 3),
                features=features, cloud=torch.rand(1, 1024, 3), mask=torch.ones(1, 1024, dtype=torch.bool))
    target = torch.randn(n, 4); target[:, 3] *= 20.; norms = norms_from_complete_physical_target(target)
    results = {}
    factories = {
        'transolver_gain': (lambda: ResponseGainTransolver(scale, device='cpu'), lambda: CoarseTransolver('transolver', device='cpu')),
        'FNO16_gain': (lambda: ResponseGainFNO(scale, device='cpu'), lambda: CoarseFNOControl(device='cpu')),
    }
    for name, (build, control) in factories.items():
        chunked, direct, original = build().train(), build().train(), control().train()
        assert set(chunked.state_dict()) == set(original.state_dict())
        assert all(torch.equal(v, original.state_dict()[k]) for k, v in chunked.state_dict().items())
        loss, leaves = full_physical_backward(chunked, case, target, scale, norms, ['synthetic/CPU'], lambda: None)
        context, latent = direct.encode_case(case)
        full = physical_relative_squared_loss(direct.query(case, 0, n, context, latent), target, scale, norms)
        full.backward(); assert abs(loss-float(full.detach())) < 1e-9
        groups = dict(encoder=0., operator=0., head=0.); maximum = 0.
        expected = dict(direct.named_parameters())
        for key, parameter in chunked.named_parameters():
            other = expected[key]
            assert parameter.grad is not None and other.grad is not None and torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-6, rtol=3e-4)
            maximum = max(maximum, float((parameter.grad-other.grad).abs().max()))
            groups[key.split('.')[0]] += float(parameter.grad.double().square().sum())
        assert all(v > 0 for v in groups.values()) and leaves['geometry_context_L2'] > 0 and leaves['operator_latent_L2'] > 0
        results[name] = dict(initial_weights_equal_own_physical_control=True, all_named_gradients_close=True,
            parameter_count=chunked.profile()['parameter_count'], max_gradient_absolute_difference=maximum,
            groups_L2={k:v**.5 for k,v in groups.items()}, leaves=leaves, chunked_loss=loss, monolithic_loss=float(full.detach()))
    proof = dict(status='actual_synthetic_response_gain_two_model_chain_CPU_passed_not_GPU',
        source=d.identity('geobc/response_gain_v30.py', hashed=True), synthetic_nodes=n, chunks=[32768, 569],
        input_formula_algebra_passed=True, signed_decode_roundtrip_passed=True, invalid_inputs_rejected=True,
        results=results, real_numeric_GT_reads=0, GPU_calls=0, optimizer_updates=0, CPU_seconds=time.monotonic()-began)
    d.write('notes/geobc_response_gain_v30_CPU_fixture.json', proof)
    print(dict(status=proof['status'], CPU_seconds=proof['CPU_seconds'], max_diffs={k:v['max_gradient_absolute_difference'] for k,v in results.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    if parser.parse_args().check: run()
