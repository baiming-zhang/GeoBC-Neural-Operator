"""Synthetic all-R64 CPU forward/backward interface; never real-data/GPU qualification."""
import argparse


def run():
    import time
    began = time.monotonic()
    import torch
    from geobc import available_cad_pilot_data as d
    from geobc.wide_fno64_v24 import WideFNO64
    from geobc.common_grid_solver import CommonGridSolver
    from geobc.coarse_transolver_v19 import full_physical_backward
    from geobc.physical_field_loss import PhysicalScale, norms_from_complete_physical_target
    torch.set_num_threads(4); torch.manual_seed(2026)
    model = WideFNO64(device='cpu').train(); reference = CommonGridSolver('FNO9', 2026)
    for group in ('encoder', 'head'):
        actual = getattr(model, group).state_dict(); expected = getattr(reference, group).state_dict()
        assert set(actual) == set(expected) and all(torch.equal(value, expected[key]) for key, value in actual.items())
    del reference
    xyz = torch.rand(37, 3); position = xyz*63; low = position.floor().long().clamp(max=62); fraction = position-low
    corners = torch.tensor([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)])
    nodes = low[:, None]+corners
    weights = torch.where(corners.bool(), fraction[:, None], 1-fraction[:, None]).prod(-1)
    indices = nodes[..., 0]*4096+nodes[..., 1]*64+nodes[..., 2]
    assert torch.allclose(weights.sum(-1), torch.ones(37))
    grid = torch.rand(1, 29, 64, 64, 64); grid[:, 0] = 1.
    case = dict(key='synthetic/CPU', scientific_group='TRAIN', grid=grid, xyz=xyz, features=torch.rand(37, 25),
        cloud=torch.rand(1, 1024, 3), mask=torch.ones(1, 1024, dtype=torch.bool), query_indices=indices, query_weights=weights)
    truth = torch.randn(37, 4); norms = norms_from_complete_physical_target(truth)
    scale = PhysicalScale.from_mapping(dict(mean=[0.]*4, std=[1.]*4, stress_scale=1., channels=[0, 1, 2, 3]))
    loss, leaves = full_physical_backward(model, case, truth, scale, norms, ['synthetic/CPU'], lambda: None)
    groups = dict(encoder=0., head=0., operator=0.)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        groups[name.split('.')[0]] += float(parameter.grad.double().square().sum())
    assert all(v > 0 for v in groups.values()) and leaves['geometry_context_L2'] > 0 and leaves['operator_latent_L2'] > 0
    profile = model.profile()
    proof = dict(status='actual_synthetic_R64_CPU_interface_passed_not_GPU', source=d.identity('geobc/wide_fno64_v24.py', hashed=True),
        profile=profile, parameter_ratio_vs_FNO9=profile['parameter_count']/4526858,
        actual_shared_encoder_head_equal_seeded_FNO=True, groups_L2={k: v**.5 for k, v in groups.items()}, leaf_gradients=leaves,
        synthetic_objective=loss, synthetic_original_nodes=37, real_numeric_GT_reads=0, optimizer_updates=0, GPU_calls=0,
        CPU_seconds=time.monotonic()-began, actual_training_execution=0)
    d.write('notes/geobc_wide_fno64_v24_CPU_fixture.json', proof)
    print(dict(status=proof['status'], params=profile['parameter_count'], ratio=proof['parameter_ratio_vs_FNO9'], CPU_seconds=proof['CPU_seconds']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    if parser.parse_args().check: run()
