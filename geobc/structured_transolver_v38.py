"""Pinned official structured-3D TS candidate on the existing R16 inputs.

This tests an explicit spatial-neighborhood inductive bias. Hidden128 is one
resource/capacity choice, not a tuning sweep or a pure architecture ablation.
"""
import hashlib
from pathlib import Path
from types import ModuleType
import sys
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .coarse_transolver_v19 import CoarseTransolver


SOURCE = 'third_party/Transolver/PDE-Solving-StandardBenchmark/model/Transolver_Structured_Mesh_3D.py'
SOURCE_SHA256 = 'af84ac8252ad0c065617a3e8e4339ecf982df0feb55a1ec3d947c182be2e1126'
_MODULE = None


def official_module():
    global _MODULE
    from .transformers import _load_source, _sources
    source = Path(__file__).resolve().parents[1]/SOURCE
    body = source.read_bytes()
    if hashlib.sha256(body).hexdigest() != SOURCE_SHA256:
        raise RuntimeError('Pinned official structured-3D source differs')
    if _MODULE is None:
        # Reuse the existing isolated loader's hash-checked Physics_Attention
        # and Embedding imports, and its standard trunc_normal_ binding.
        base = _load_source('transolver')
        _MODULE = ModuleType('_geobc_official_structured_TS_v38')
        _MODULE.__file__ = str(source)
        _MODULE.__dict__['__builtins__'] = base.__dict__['__builtins__']
        sys.modules[_MODULE.__name__] = _MODULE
        exec(compile(body, str(source), 'exec'), _MODULE.__dict__)
    return _MODULE, _sources()['transolver']


class OfficialStructured3D(nn.Module):
    def __init__(self):
        super().__init__()
        module, source = official_module()
        kwargs = dict(space_dim=3,fun_dim=58,out_dim=16,n_layers=5,n_hidden=128,
            n_head=8,slice_num=32,mlp_ratio=1,dropout=0.,act='gelu',
            unified_pos=False,ref=8,H=16,W=16,D=16)
        self.backbone = module.Model(**kwargs)
        self.backbone.use_checkpoint = False
        self.checkpoint_blocks = True
        self.source_config = dict(repository=source['repository'],commit=source['commit'],
            model_file=SOURCE,file_sha256=SOURCE_SHA256,constructor_kwargs=kwargs,
            dependency_hashes=source['file_sha256'],variant='official structured3D, not irregular/++/3',
            macro_difference='Conv3d(kernel3) spatial projections in Physics_Attention',
            capacity_choice='hidden128 once, approximate FNO16-sized candidate; not exact capacity matching',
            upstream_forward_preserved=True,placeholder_unused_when_features_supplied=True,
            allowed_unused_parameters=['backbone.placeholder'])

    def forward(self, tokens):
        if tokens.shape != (1,4096,61) or tokens.dtype != torch.float32:
            raise ValueError('Complete FP32 XYZ-first61 canonical R16 tokens required')
        if self.training and self.checkpoint_blocks and torch.is_grad_enabled():
            # The structured upstream fx-present branch omits placeholder;
            # unlike the irregular variant, adding it would change the model.
            value = self.backbone.preprocess(tokens)
            for block in self.backbone.blocks:
                value = checkpoint(block,value,use_reentrant=False,preserve_rng_state=True)
            return value
        return self.backbone(tokens[..., :3],tokens[..., 3:])


class CoarseStructuredTransolver(nn.Module):
    def __init__(self, seed=2026, device='cuda'):
        super().__init__()
        from .conditioned_field import PointNetGeometryEncoder
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = PointNetGeometryEncoder(32,64)
            self.head = nn.Sequential(nn.Linear(76,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,4))
            self.operator = OfficialStructured3D()
        self.method = 'transolver_structured3d'
        self.macro_family = 'R16_transolver_structured3d'
        self.to(device)

    encode_case = CoarseTransolver.encode_case
    query = CoarseTransolver.query
    profile = CoarseTransolver.profile
