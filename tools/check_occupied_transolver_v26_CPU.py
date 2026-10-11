"""Synthetic CPU mathematical/interface fixture; no real GT, GPU or optimizer."""
import argparse


def run():
    import time
    began = time.monotonic()
    import torch
    from geobc import available_cad_pilot_data as d
    from geobc.occupied_transolver_v26 import OccupiedTransolver
    from geobc.coarse_transolver_v19 import CoarseTransolver, pool_inputs, full_physical_backward
    from geobc.physical_field_loss import PhysicalScale, norms_from_complete_physical_target
    torch.set_num_threads(4); torch.manual_seed(2026)
    model = OccupiedTransolver(device='cpu').train()
    baseline = CoarseTransolver('transolver', device='cpu').train()
    actual = model.state_dict(); expected = baseline.state_dict()
    assert set(actual) == set(expected) and all(torch.equal(v, expected[k]) for k, v in actual.items())
    grid = torch.rand(1, 29, 64, 64, 64); grid[:, 0] = 0.
    blocks = [(2, 3, 4), (3, 3, 4), (3, 4, 4)]
    for x, y, z in blocks:
        grid[:, 0, x*4:(x+1)*4, y*4:(y+1)*4, z*4:(z+1)*4] = 1.
    centers = (torch.tensor(blocks).float()*4+1.5)/63
    xyz = centers[torch.arange(37) % 3] + (torch.rand(37, 3)-.5)/63
    case = dict(key='synthetic/CPU', scientific_group='TRAIN', grid=grid, xyz=xyz,
                features=torch.rand(37, 25), cloud=torch.rand(1, 1024, 3), mask=torch.ones(1, 1024, dtype=torch.bool))
    occupied = pool_inputs(grid)[0, 0].flatten() > 0
    assert int(occupied.sum()) == 3
    with torch.no_grad():
        context, latent = model.encode_case(case)
        flat = latent.flatten(2).transpose(1, 2)[0]
        assert torch.count_nonzero(flat[~occupied]) == 0 and torch.isfinite(flat).all()
        assert torch.count_nonzero(flat[occupied]) > 0
    truth = torch.randn(37, 4); norms = norms_from_complete_physical_target(truth)
    scale = PhysicalScale.from_mapping(dict(mean=[0.]*4, std=[1.]*4, stress_scale=1., channels=[0, 1, 2, 3]))
    loss, leaves = full_physical_backward(model, case, truth, scale, norms, ['synthetic/CPU'], lambda: None)
    groups = dict(encoder=0., head=0., operator=0.)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        groups[name.split('.')[0]] += float(parameter.grad.double().square().sum())
    assert all(v > 0 for v in groups.values()) and leaves['geometry_context_L2'] > 0 and leaves['operator_latent_L2'] > 0
    proof = dict(status='actual_synthetic_occupied_TS_CPU_passed_not_GPU',
        source=d.identity('geobc/occupied_transolver_v26.py', hashed=True), profile=model.profile(),
        all_initial_state_equal_dense_originalTS=True, synthetic_selected_tokens=3, synthetic_empty_tokens=4093,
        exact_empty_latent_zero=True, groups_L2={k: v**.5 for k, v in groups.items()}, leaf_gradients=leaves,
        synthetic_original_nodes=37, synthetic_objective=loss, real_numeric_GT_reads=0,
        GPU_calls=0, optimizer_updates=0, CPU_seconds=time.monotonic()-began)
    d.write('notes/geobc_occupied_transolver_v26_CPU_fixture.json', proof)
    print(dict(status=proof['status'], parameters=proof['profile']['parameter_count'], CPU_seconds=proof['CPU_seconds']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    if parser.parse_args().check: run()
