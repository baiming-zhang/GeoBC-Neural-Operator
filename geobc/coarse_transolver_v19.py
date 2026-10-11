"""Candidate compact global tokens; no data, training plan, or launch ownership.

Pools existing target-free R64 inputs over occupied voxels. All original query
nodes remain supervised. Coarse geometry/BC information can be lost; this is
a representation candidate, not a capacity-matched architecture comparison.
"""
import torch
from torch import nn
from torch.nn import functional as F

METHODS = ('transolver', 'transolver_pp', 'ts3')


def pool_inputs(grid):
    if grid.shape != (1, 29, 64, 64, 64) or grid.dtype != torch.float32:
        raise ValueError('Existing complete FP32 R64 input29 required')
    occupancy = grid[:, :1]
    if not torch.isfinite(grid).all() or occupancy.min() < 0 or occupancy.max() > 1:
        raise ValueError('Finite inputs and source occupancy in [0,1] required')
    mass = F.avg_pool3d(occupancy, 4, 4)
    xyz = F.avg_pool3d(grid[:, 1:4], 4, 4)
    numerator = F.avg_pool3d(grid[:, 4:]*occupancy, 4, 4)
    denominator = torch.where(mass > 0, mass, torch.ones_like(mass))
    features = torch.where(mass > 0, numerator/denominator, torch.zeros_like(numerator))
    return torch.cat((mass, xyz, features), dim=1)


def coarse_stencil(xyz):
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not torch.isfinite(xyz).all():
        raise ValueError('Nonempty finite normalized original xyz required')
    if xyz.min() < -2e-6 or xyz.max() > 1+2e-6:
        raise ValueError('Outside normalized reference cube')
    # Token centers are (4*i+1.5)/63, not i/15. Extend the nearest center
    # constantly beyond that interval; original coordinates/labels stay intact.
    position = ((xyz.clamp(0, 1)*63-1.5)/4).clamp(0, 15)
    low = position.floor().long().clamp(max=14)
    fraction = position-low
    corners = torch.tensor([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], device=xyz.device)
    nodes = low[:, None]+corners
    indices = nodes[..., 0]*256+nodes[..., 1]*16+nodes[..., 2]
    weights = torch.where(corners.bool(), fraction[:, None], 1-fraction[:, None]).prod(-1)
    return indices, weights


class CoarseTransolver(nn.Module):
    def __init__(self, method, seed=2026, device='cuda'):
        super().__init__()
        if method not in METHODS:
            raise ValueError('Fixed official Transolver family only')
        from .conditioned_field import PointNetGeometryEncoder
        from .transformers import build_transformer
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = PointNetGeometryEncoder(32, 64)
            self.head = nn.Sequential(nn.Linear(76, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 4))
            self.operator = build_transformer(method, 61, 16, device='cpu', checkpoint_blocks=True)
        self.method = method
        self.macro_family = 'R16_'+method
        self.to(device)

    def encode_case(self, case):
        context = self.encoder(case['cloud'], case['mask'])
        grid = pool_inputs(case['grid'])
        tokens = torch.cat((grid[:, 1:4], grid[:, :1], grid[:, 4:],
                            context[:, :, None, None, None].expand(1, 32, 16, 16, 16)), dim=1)
        tokens = tokens.flatten(2).transpose(1, 2).contiguous()
        if tokens.shape != (1, 4096, 61):
            raise RuntimeError('All 4096 global tokens must enter one operator')
        latent = self.operator(tokens)
        if latent.shape != (1, 4096, 16):
            raise RuntimeError('Original official operator must return token features16')
        return context, latent.transpose(1, 2).reshape(1, 16, 16, 16, 16)

    def query(self, case, start, stop, context, latent):
        from .fno import query_grid
        xyz = case['xyz'][start:stop]; native = case['features'][start:stop]
        if context.shape != (1, 32) or latent.shape != (1, 16, 16, 16, 16) or native.shape != (len(xyz), 25):
            raise ValueError('Original native25 queries and fixed whole-case context required')
        indices, weights = coarse_stencil(xyz)
        values = query_grid(latent[0], indices, weights, 32768)
        return self.head(torch.cat((xyz, native, context.expand(len(xyz), -1), values), dim=-1))

    def profile(self):
        counts = {k: sum(p.numel() for p in getattr(self, k).parameters()) for k in ('encoder', 'head', 'operator')}
        return dict(method=self.method, family=self.macro_family, parameter_count=sum(counts.values()),
                    architecture=self.operator.source_config, counts=counts, global_tokens=4096, grid_resolution=16,
                    precision='highest FP32; caller must disable TF32', all_original_query_nodes=True,
                    input_policy='occupied-voxel means from original R64; no target pooling',
                    query_policy='true block-center trilinear latent; nearest-center boundary extension',
                    capacity_matched=False, independent_offnode_FEM_accuracy_measured=False)


def full_physical_backward(model, case, physical_target, scale, norms, allowed_keys, guard):
    if case.get('scientific_group') != 'TRAIN' or case.get('key') not in allowed_keys:
        raise PermissionError('Fixed TRAIN cases only')
    count = len(case['xyz'])
    if count != norms.node_count or physical_target.shape != (count, 4):
        raise ValueError('Complete original-node targets and full-case norms required')
    from .physical_field_loss import physical_relative_squared_loss
    context, latent = model.encode_case(case)
    c = context.detach().requires_grad_(True); l = latent.detach().requires_grad_(True)
    total = 0.; consumed = 0
    for start in range(0, count, 32768):
        guard(); stop = min(start+32768, count)
        prediction = model.query(case, start, stop, c, l)
        loss = physical_relative_squared_loss(prediction, physical_target[start:stop], scale, norms)
        loss.backward(); total += float(loss.detach()); consumed += stop-start
    if consumed != count or c.grad is None or l.grad is None or not torch.isfinite(c.grad).all() or not torch.isfinite(l.grad).all():
        raise RuntimeError('Whole-node coverage and finite joint leaf gradients required')
    torch.autograd.backward((context, latent), (c.grad, l.grad))
    return total, dict(geometry_context_L2=float(c.grad.norm()), operator_latent_L2=float(l.grad.norm()),
                       original_nodes_used=count, node_sampling=False)
