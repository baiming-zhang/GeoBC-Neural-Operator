"""Pinned official U-NO with the unchanged GeoBC PointNet/query interface."""
import importlib
import importlib.machinery
from pathlib import Path
import sys
import types
import torch
from torch import nn
from .common_grid_solver import CommonGridSolver
from .fno_stress_specialist_v54 import tree_digest

PREFIX = '_geobc_UNO_b307030_v90'


def official_class():
    root = Path(__file__).resolve().parents[1]/'third_party/neuraloperator/neuralop'
    for suffix, directory in (('', root), ('.models', root/'models'), ('.layers', root/'layers')):
        name = PREFIX+suffix
        if name in sys.modules:
            assert list(sys.modules[name].__path__) == [str(directory)]
            continue
        package = types.ModuleType(name); package.__package__ = name; package.__path__ = [str(directory)]
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = package
    # DomainPadding uses this one absolute upstream utility import.
    # Alias the identical pinned utility module; never run package __init__.
    if 'neuralop' in sys.modules:
        assert sys.modules['neuralop'] is sys.modules[PREFIX], 'Refuse unrelated neuralop namespace'
    else:
        sys.modules['neuralop'] = sys.modules[PREFIX]
    utilities = importlib.import_module(PREFIX+'.utils')
    if 'neuralop.utils' in sys.modules:
        assert sys.modules['neuralop.utils'] is utilities
    else:
        sys.modules['neuralop.utils'] = utilities
    sys.modules['neuralop'].utils = utilities
    return importlib.import_module(PREFIX+'.models.uno').UNO


class WholeCheckpoint(nn.Module):
    def __init__(self, operator):
        super().__init__(); self.official = operator

    def forward(self, inputs):
        if self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self.official, inputs, use_reentrant=False)
        return self.official(inputs)


def counts(model):
    result = {}
    for name in ('encoder', 'head', 'operator'):
        params = list(getattr(model, name).parameters())
        result[name] = dict(storage_numel=sum(p.numel() for p in params),
            real_degrees_of_freedom=sum(p.numel()*(2 if p.is_complex() else 1) for p in params),
            complex_storage_numel=sum(p.numel() for p in params if p.is_complex()),
            dtypes=sorted({str(p.dtype) for p in params}))
    result['total_storage_numel'] = sum(v['storage_numel'] for v in result.values())
    result['total_real_degrees_of_freedom'] = sum(result[k]['real_degrees_of_freedom'] for k in ('encoder', 'head', 'operator'))
    return result


def build_solver(kwargs, expected_original_digest, expected_original_bodies, seed=2026):
    model = CommonGridSolver('FNO9', seed)
    base_digest = tree_digest(model.state_dict())
    base_bodies = {name: tree_digest(getattr(model, name).state_dict()) for name in ('encoder', 'operator')}
    shared = {name: tree_digest(getattr(model, name).state_dict()) for name in ('encoder', 'head')}
    assert base_digest == expected_original_digest and base_bodies == expected_original_bodies
    args = dict(kwargs); args['horizontal_skips_map'] = {int(k): v for k, v in args['horizontal_skips_map'].items()}
    UNO = official_class()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed); model.operator = WholeCheckpoint(UNO(**args))
    assert shared == {name: tree_digest(getattr(model, name).state_dict()) for name in ('encoder', 'head')}
    model.package_arm = 'OFFICIAL_UNO_TRAIN9'
    model.original_initialization = dict(original_FNO_model_digest=base_digest, original_FNO_body_digests=base_bodies,
        kept_encoder_head_digests=shared, encoder_head_bytes_kept=True)
    model.actual_UNO_config = args
    model.actual_parameter_profile = counts(model)
    return model.cuda()


def parameter_gradients(model):
    result = {}
    for name in ('encoder', 'head', 'operator'):
        missing = []; finite = True; energy = 0.
        for key, p in getattr(model, name).named_parameters():
            if p.grad is None: missing.append(key); continue
            finite &= bool(torch.isfinite(p.grad).all())
            energy += float(p.grad.detach().abs().double().square().sum())
        result[name] = dict(missing=missing, finite=finite, L2=energy**.5,
                            complex_gradient_metric='sum(abs(grad)^2), includes real and imaginary')
    return result
