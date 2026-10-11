"""CPU-only interface check; does not qualify actual TCNN training/accuracy."""
import argparse
import time


def run():
    entry = time.monotonic()
    import torch
    from torch import nn
    from geobc import available_cad_pilot_data as d
    from geobc.common_grid_solver import CommonGridSolver
    from geobc.operator_conditioned_hash import OperatorConditionedHashSolver
    from geobc.tcnn_fields import TCNNFieldConfig
    from geobc.solver_selection_data import _stencil
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()

    class FixtureDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = TCNNFieldConfig(out_channels=4, input_mode='both',
                                         condition_dim=73, seed=2026)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(2026)
                self.net = nn.Sequential(nn.Linear(76, 16), nn.Tanh(), nn.Linear(16, 4))

        def forward(self, xyz, features):
            return self.net(torch.cat((xyz, features), dim=-1))

    model = OperatorConditionedHashSolver(device='cpu', injected_decoder=FixtureDecoder())
    baseline = CommonGridSolver('FNO9', seed=2026)
    for group in ('encoder', 'operator'):
        a = getattr(model, group).state_dict()
        b = getattr(baseline, group).state_dict()
        assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    del baseline
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(234)
        xyz = (torch.rand(7, 3) * .5 + .1).requires_grad_(True)
        native25 = torch.rand(7, 25, requires_grad=True)
        context = torch.rand(1, 32, requires_grad=True)
        latent = torch.rand(1, 16, 64, 64, 64, requires_grad=True)
    indices, weights = _stencil(xyz, 64)
    case = dict(xyz=xyz, features=native25, query_indices=indices, query_weights=weights)
    direct = model.query(case, 0, 7, context, latent)
    chunked = model.query_points(xyz, native25, context, latent, chunk_size=3)
    assert torch.allclose(direct, chunked, atol=2e-7, rtol=2e-6)
    perm = torch.tensor([6, 1, 3, 0, 5, 4, 2])
    shuffled = model.query_points(xyz[perm], native25[perm], context, latent, chunk_size=2)
    assert torch.allclose(shuffled, direct[perm], atol=2e-7, rtol=2e-6)
    grads = torch.autograd.grad(direct.sum(), (xyz, native25, context, latent))
    norms = {name: float(g.norm()) for name, g in zip(('xyz', 'native25', 'context', 'latent'), grads)}
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in grads)
    eps = 1e-4
    plus = xyz.detach().clone(); minus = xyz.detach().clone()
    plus[0, 0] += eps; minus[0, 0] -= eps
    with torch.no_grad():
        fd = float((model.query_points(plus, native25, context, latent).sum() -
                    model.query_points(minus, native25, context, latent).sum()) / (2 * eps))
    automatic = float(grads[0][0, 0])
    assert abs(fd - automatic) < .006 * max(1., abs(automatic))
    try:
        model.query_points(xyz, native25[:, :24], context, latent)
    except ValueError:
        rejected = True
    else:
        rejected = False
    assert rejected and not torch.cuda.is_initialized()
    output = d.path('results/geobc_operator_hash_interface')
    assert not (output / 'manifest.json').exists(), 'One interface receipt; no silent overwrite'
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(status='completed_CPU_interface_only', elapsed_seconds=time.monotonic()-entry,
                   FNO_encoder_operator_initial_weights_equal=True,
                   continuous_original_query_chunk_permutation_equal=True,
                   supplied_input_gradients_nonzero=norms, xyz_AD=float(automatic), xyz_FD=fd,
                   invalid_native25_rejected=True, CUDA_initialized=False,
                   dataset_reads=0, numerical_GT_reads=0, checkpoint_reads=0,
                   training_updates=0, actual_native_TCNN_qualified=False,
                   actual_solver_accuracy_measured=False, final_solver_selected=False,
                   source_pins={p: d.sha(p) for p in ('geobc/operator_conditioned_hash.py',
                                'tools/check_operator_conditioned_hash_interface.py')})
    d.write(output / 'manifest.json', receipt)
    (output / 'BRIEF.md').write_text(
        '# 全局算子条件化Hash：接口检查\n\n'
        '候选复用FNO完整R64几何／BC／材料算子，将16维空间latent与32维几何descriptor、25维局部特征一起送入连续Hash解码器。'
        'FNO骨干初始化与原MLP基线完全相同；Hash解码器沿用已有配置，不作小幅救参。\n\n'
        'CPU注入解码器检查通过：原节点／连续点／分块／点重排一致，xyz／局部特征／全局context／latent均有有限非零梯度；坐标AD与FD一致。'
        '未初始化CUDA、未读数据／GT／检查点、未训练。此检查不证明真实TCNN梯度、速度、精度或独立离节点FEM准确性。'
        '实际实验须在当前FNO100覆盖结果之后按证据选定宏观对照、冻结预算并放行。\n', encoding='utf-8')
    print(receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    if parser.parse_args().check:
        run()
