"""Synthetic CPU mathematics only; never qualifies an official GPU model."""
import argparse


def run():
    import copy
    import datetime
    import time
    import torch
    from torch import nn
    from geobc import available_cad_pilot_data as d
    from geobc.coarse_transolver_v19 import pool_inputs, coarse_stencil, CoarseTransolver, full_physical_backward
    from geobc.physical_field_loss import PhysicalScale, norms_from_complete_physical_target, physical_relative_squared_loss
    from geobc.fno import query_grid
    begin = time.monotonic(); torch.set_num_threads(4); torch.manual_seed(2026)
    grid = torch.zeros((1, 29, 64, 64, 64)); a = torch.linspace(0, 1, 64)
    grid[:, 1:4] = torch.stack(torch.meshgrid(a, a, a, indexing='ij')); grid[:, 4:] = 999
    grid[0, 0, 0, 0, 0] = grid[0, 0, 3, 3, 3] = 1
    grid[0, 4:, 0, 0, 0] = 2; grid[0, 4:, 3, 3, 3] = 6
    pooled = pool_inputs(grid)
    assert pooled.shape == (1, 29, 16, 16, 16)
    assert pooled[0, 0, 0, 0, 0] == 2/64 and torch.all(pooled[0, 4:, 0, 0, 0] == 4)
    assert torch.all(pooled[0, 4:, 1, 1, 1] == 0)
    assert torch.allclose(pooled[0, 1:4, 0, 0, 0], torch.full((3,), 1.5/63), atol=1e-7, rtol=0)
    # FP64 affine identity proves the mapping. The first FP32 attempt's ~3.8e-6
    # derivative rounding is preserved separately, not erased or called a pass.
    centers = (torch.arange(16, dtype=torch.float64)*4+1.5)/63
    x, y, z = torch.meshgrid(centers, centers, centers, indexing='ij')
    field = (2*x+3*y-z).unsqueeze(0)
    points = torch.tensor([[.4, .5, .6], [0., .5, .6], [1., .5, .6]], dtype=torch.float64, requires_grad=True)
    indices, weights = coarse_stencil(points); values = query_grid(field, indices, weights)
    expected = 2*points[:, 0].detach().clamp(float(centers[0]), float(centers[-1]))+3*points[:, 1].detach()-points[:, 2].detach()
    assert torch.allclose(values[:, 0], expected, atol=1e-13, rtol=0)
    assert torch.allclose(weights.sum(1), torch.ones(3, dtype=torch.float64), atol=1e-14, rtol=0)
    values.sum().backward()
    assert torch.allclose(points.grad, torch.tensor([[2., 3., -1.], [0., 3., -1.], [0., 3., -1.]], dtype=torch.float64), atol=1e-13, rtol=0)

    class Fixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Parameter(torch.randn(1, 32)*.01)
            self.l = nn.Parameter(torch.randn(1, 16, 16, 16, 16)*.01)
            self.head = nn.Sequential(nn.Linear(76, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 4))

        def encode_case(self, case):
            return self.c, self.l

        query = CoarseTransolver.query

    n = 32775; case = dict(xyz=torch.rand(n, 3), features=torch.randn(n, 25)*.01, scientific_group='TRAIN', key='fixture/train')
    target = torch.randn(n, 4); norms = norms_from_complete_physical_target(target)
    scale = PhysicalScale((0., 0., 0., 0.), (1., 1., 1., 1.), None)
    model = Fixture(); reference = copy.deepcopy(model)
    chunk_loss, _ = full_physical_backward(model, case, target, scale, norms, ['fixture/train'], lambda: None)
    c, l = reference.encode_case(case); prediction = reference.query(case, 0, n, c, l)
    direct = physical_relative_squared_loss(prediction, target, scale, norms); direct.backward()
    assert abs(chunk_loss-float(direct.detach())) < 1e-8
    for (name, p), (other, q) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other and p.grad is not None and torch.allclose(p.grad, q.grad, atol=2e-7, rtol=2e-5), name
    assert not torch.cuda.is_initialized()
    receipt = dict(status='actual_CPU_math_fixture_passed', created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=d.identity('geobc/coarse_transolver_v19.py', hashed=True), checker=d.identity('tools/check_coarse_transolver_v19_CPU.py', hashed=True),
        occupied_pooling=True, empty_cells_zero_without_epsilon=True, actual_block_centers=True,
        FP64_affine_identity_and_boundary_extension=True, FP32_whole_vs32768chunk_chain_rule=True,
        synthetic_nodes=n, CPU_seconds=time.monotonic()-begin, GT_PT_arrays_read=0, GPU_calls=0,
        official_models_instantiated=0, real_largestN_gradient_qualification=False, not_an_actual_training_or_accuracy_result=True,
        first_FP32_derivative_tolerance_failure_preserved='notes/geobc_coarse_transolver_v19_CPU_fixture_first_failure.json')
    assert not d.path('notes/geobc_coarse_transolver_v19_CPU_fixture.json').exists(), 'No completed fixture replay'
    d.write('notes/geobc_coarse_transolver_v19_CPU_fixture.json', receipt)
    print(dict(status=receipt['status'], CPU_seconds=receipt['CPU_seconds'], official_model_or_GPU_runs=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    if parser.parse_args().check:
        run()
