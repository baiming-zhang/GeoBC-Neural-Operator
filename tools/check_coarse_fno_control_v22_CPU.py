"""Synthetic CPU interface/chain-rule qualification only; no real data or GPU."""
import argparse


def run():
    import time
    began = time.monotonic()
    import torch
    from geobc import available_cad_pilot_data as d
    from geobc.coarse_fno_control_v22 import CoarseFNOControl
    from geobc.coarse_transolver_v19 import full_physical_backward
    from geobc.common_grid_solver import CommonGridSolver
    from geobc.physical_field_loss import PhysicalScale, norms_from_complete_physical_target
    torch.set_num_threads(4); torch.manual_seed(2026)
    model = CoarseFNOControl(device='cpu').train(); reference = CommonGridSolver('FNO9', 2026)
    for group in ('encoder', 'head'):
        actual = getattr(model, group).state_dict(); expected = getattr(reference, group).state_dict()
        assert set(actual) == set(expected) and all(torch.equal(v, expected[k]) for k, v in actual.items())
    del reference
    grid = torch.rand(1, 29, 64, 64, 64); grid[:, 0] = (grid[:, 0] > .5).float()
    case = dict(key='synthetic/CPU', scientific_group='TRAIN', grid=grid, cloud=torch.rand(1, 1024, 3),
                mask=torch.ones(1, 1024, dtype=torch.bool), xyz=torch.rand(37, 3), features=torch.rand(37, 25))
    truth = torch.randn(37, 4); norms = norms_from_complete_physical_target(truth)
    scale = PhysicalScale.from_mapping(dict(mean=[0., 0., 0., 0.], std=[1., 1., 1., 1.], stress_scale=1., channels=[0, 1, 2, 3]))
    loss, leaves = full_physical_backward(model, case, truth, scale, norms, ['synthetic/CPU'], lambda: None)
    groups = dict(encoder=0., head=0., operator=0.)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        groups[name.split('.')[0]] += float(parameter.grad.double().square().sum())
    assert all(v > 0 for v in groups.values()) and leaves['geometry_context_L2'] > 0 and leaves['operator_latent_L2'] > 0
    proof = dict(status='actual_synthetic_CPU_interface_passed_not_GPU', source=d.identity('geobc/coarse_fno_control_v22.py', hashed=True),
        profile=model.profile(), actual_shared_encoder_head_equal_seeded_FNO=True, synthetic_nodes=37,
        groups_L2={k: v**.5 for k, v in groups.items()}, leaf_gradients=leaves, synthetic_objective=loss,
        real_numeric_GT_reads=0, GPU_calls=0, optimizer_updates=0, actual_training_execution=0, CPU_seconds=time.monotonic()-began)
    d.write('notes/geobc_coarse_fno_control_v22_CPU_fixture.json', proof)
    print(dict(status=proof['status'], parameter_count=proof['profile']['parameter_count'], CPU_seconds=proof['CPU_seconds']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    if parser.parse_args().check: run()
