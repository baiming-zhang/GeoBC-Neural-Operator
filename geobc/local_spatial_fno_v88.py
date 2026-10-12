"""Local spatial propagation inside the four original FNO spectral blocks."""
import torch
from torch import nn
from torch.nn import functional as F
from .common_grid_solver import CommonGridSolver
from .fno import FNO3d


class LocalSpatialOperator(FNO3d):
    def __init__(self, base):
        nn.Module.__init__(self)
        self.config = base.config
        for name in ('lift', 'spectral', 'pointwise', 'project1', 'project2'):
            setattr(self, name, getattr(base, name))
        with torch.random.fork_rng(devices=[]):
            self.local_spatial = nn.ModuleList([
                nn.Conv3d(9, 9, 3, padding=1, bias=True) for _ in range(4)])
            for layer in self.local_spatial:
                nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        self.local_enabled = True

    def _block(self, x, index):
        pointwise = self.pointwise[index](x.flatten(2)).reshape_as(x)
        result = self.spectral[index](x) + pointwise
        if self.local_enabled:
            result = result + self.local_spatial[index](x)
        return F.gelu(result) if index < self.config.layers-1 else result


def build_solver(seed=2026):
    model = CommonGridSolver('FNO9', seed)
    model.operator = LocalSpatialOperator(model.operator)
    model.package_arm = 'FNO9_LOCAL3D_BRANCH'
    return model.cuda()


def original_state(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith('operator.local_spatial.')}


def original_body(model):
    return dict(encoder=model.encoder.state_dict(),
                operator={k: v for k, v in model.operator.state_dict().items() if not k.startswith('local_spatial.')})


def local_gradients(model):
    rows = []
    for i, layer in enumerate(model.operator.local_spatial):
        missing = [name for name, p in layer.named_parameters() if p.grad is None]
        finite = all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
        energy = sum(float(p.grad.detach().double().square().sum()) for p in layer.parameters() if p.grad is not None)
        rows.append(dict(block=i, missing=missing, finite=bool(finite), L2=energy**.5))
    return rows


def initial_function_parity(model, case, guard):
    """Real field's64 fixed queries; uses no targets or alternate weights."""
    with torch.no_grad():
        guard(); model.eval()
        context, latent = model.encode_case(case)
        active = model.query(case, 0, 64, context, latent)
        model.operator.local_enabled = False
        try:
            guard(); old_context, old_latent = model.encode_case(case)
            disabled = model.query(case, 0, 64, old_context, old_latent)
        finally:
            model.operator.local_enabled = True; model.train()
        difference = float((active-disabled).abs().max())
        assert torch.equal(context, old_context) and torch.equal(latent, old_latent)
        assert torch.equal(active, disabled), 'Zero extension changes original initial function'
        guard()
    return dict(passed=True, real_case=case['key'], actual_queries=64,
                context_exact=True, whole_R64_latent_exact=True, prediction_exact=True,
                encoded_max_absolute_error=difference, local_parameters_all_zero=True,
                independent_offnode_FEM_accuracy_proven=False)
