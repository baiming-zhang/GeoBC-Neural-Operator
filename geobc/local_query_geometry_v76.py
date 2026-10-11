"""Matched FNO query geometry pair; no datasets, launch or optimizer here."""
import torch
from torch import nn
from .common_grid_solver import CommonGridSolver
from .fno import query_grid
from .physical_field_loss import physical_relative_squared_loss

ARMS = ('FNO9_global_token_control', 'FNO9_local_query_token')


def gaussian_weights(xyz, cloud, mask, sigma=.15):
    """All valid cloud points, stable normalized Gaussian; no neighbor cutoff.

    The finite-precision tail can underflow. Coordinates remain differentiable
    when this function is called at new queries. A static nodal basis may be
    cached for training, because xyz/cloud/mask are fixed target-free inputs.
    """
    points, valid = cloud[0], mask[0]
    if cloud.shape != (1, 1024, 3) or mask.shape != (1, 1024) or not valid.any():
        raise ValueError('Original nonempty masked1024 cloud required')
    distance2 = xyz.square().sum(-1, keepdim=True) + points.square().sum(-1)[None] - 2*xyz@points.T
    return (-distance2/(2*sigma*sigma)).masked_fill(~valid[None], -torch.inf).softmax(-1)


class LocalQueryFNO(CommonGridSolver):
    def __init__(self, arm, seed=2026):
        if arm not in ARMS:
            raise ValueError(arm)
        # Preserve the original body initialization, then extend the same head
        # under a private RNG. Both arms have exactly the same state layout.
        super().__init__('FNO9', seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.head = nn.Sequential(nn.Linear(140,128), nn.GELU(), nn.Linear(128,128), nn.GELU(), nn.Linear(128,4))
        self.geometry_arm = arm
        self.macro_family = 'FNO9'

    def encode_case(self, case):
        mask = case['mask'].unsqueeze(-1)
        tokens = self.encoder.point_mlp(case['cloud'].float())
        maximum = tokens.masked_fill(~mask, -torch.inf).amax(1)
        mean = (tokens*mask).sum(1)/case['mask'].sum(1, keepdim=True).to(tokens.dtype)
        context = self.encoder.head(torch.cat((maximum, mean), -1))
        grid = torch.cat((case['grid'], context[:,:,None,None,None].expand(1,32,64,64,64)), 1)
        return context, self.operator(grid), tokens

    def geometry_query(self, case, xyz, tokens, weights=None):
        if self.geometry_arm == ARMS[0]:
            mean = (tokens*case['mask'][...,None]).sum(1)/case['mask'].sum(1,keepdim=True).to(tokens.dtype)
            return mean.expand(len(xyz), -1)
        if weights is None:
            weights = gaussian_weights(xyz, case['cloud'], case['mask'])
        return weights@tokens[0]

    def query_explicit(self, case, xyz, features, indices, weights, context, latent, tokens, geometry_weights=None):
        values = query_grid(latent[0], indices, weights, 32768)
        geometry = self.geometry_query(case, xyz, tokens, geometry_weights)
        return self.head(torch.cat((xyz, features, context.expand(len(xyz),-1), values, geometry), -1))

    def query(self, case, start, stop, context, latent, tokens):
        basis = case.get('geometry_basis') if self.geometry_arm == ARMS[1] else None
        return self.query_explicit(case, case['xyz'][start:stop], case['features'][start:stop],
            case['query_indices'][start:stop], case['query_weights'][start:stop], context, latent, tokens,
            None if basis is None else basis[start:stop])


def full_backward(model, case, physical, scale, norms, guard=lambda:None):
    """Full original-node physical objective; shared encoder graph backward once."""
    if case['scientific_group'] != 'TRAIN' or len(case['xyz']) != norms.node_count:
        raise PermissionError('Complete permitted TRAIN field required')
    context, latent, tokens = model.encode_case(case)
    originals = (context, latent, tokens)
    leaves = tuple(v.detach().requires_grad_(True) for v in originals)
    count = len(case['xyz']); total = 0.; consumed = 0
    for start in range(0,count,32768):
        guard(); stop = min(start+32768,count)
        encoded = model.query(case,start,stop,*leaves)
        loss = physical_relative_squared_loss(encoded,physical[start:stop],scale,norms)
        loss.backward(); total += float(loss.detach()); consumed += stop-start
    if consumed != count or any(v.grad is None or not torch.isfinite(v.grad).all() for v in leaves):
        raise RuntimeError('Missing finite full-case shared gradients')
    torch.autograd.backward(originals,tuple(v.grad for v in leaves))
    return total, dict(zip(('context','latent','tokens'), (float(v.grad.norm()) for v in leaves)))


def predict(model, case, guard=lambda:None):
    context, latent, tokens = model.encode_case(case)
    chunks = []
    for start in range(0,len(case['xyz']),32768):
        guard()
        chunks.append(model.query(case,start,min(start+32768,len(case['xyz'])),context,latent,tokens))
    return torch.cat(chunks)


def query_qualification(model, case, guard=lambda:None):
    """Real-case functional query/chunk/permutation checks, not a FEM oracle."""
    from .solver_selection_data import _stencil
    model.eval(); guard()
    with torch.no_grad():
        context, latent, tokens = model.encode_case(case)
    # Fixed real nodal positions, detached local25. This qualifies the geometry
    # query path and latent stencil, not an analytic extension of nodal BCs.
    idx = torch.linspace(0,len(case['xyz'])-1,64,device=case['xyz'].device).long()
    xyz = case['xyz'][idx].detach().clone().requires_grad_(True)
    features = case['features'][idx]
    indices, weights = _stencil(xyz,64)
    out = model.query_explicit(case,xyz,features,indices,weights,context,latent,tokens)
    gradient, = torch.autograd.grad(out.square().sum(),xyz)
    if not torch.isfinite(out).all() or not torch.isfinite(gradient).all() or gradient.norm() <= 0:
        raise RuntimeError('Actual query path missing finite nonzero coordinate gradient')
    geometry = model.geometry_query(case,xyz,tokens)
    geometry_gradient, = torch.autograd.grad(geometry.square().sum(),xyz,allow_unused=True) if geometry.requires_grad else (None,)
    if model.geometry_arm == ARMS[1] and (geometry_gradient is None or not torch.isfinite(geometry_gradient).all() or geometry_gradient.norm() <= 0):
        raise RuntimeError('Local geometry coordinate path is detached')
    with torch.no_grad():
        parts = [model.query_explicit(case,xyz[a:a+17],features[a:a+17],indices[a:a+17],weights[a:a+17],context,latent,tokens) for a in range(0,64,17)]
        chunk_error = float((torch.cat(parts)-out).abs().max())
        permutation = torch.arange(1023,-1,-1,device=xyz.device)
        permuted = dict(case,cloud=case['cloud'][:,permutation],mask=case['mask'][:,permutation])
        pc, pl, pt = model.encode_case(permuted)
        p = model.query_explicit(permuted,xyz,features,indices,weights,pc,pl,pt)
        permutation_error = float((p-out).abs().max())
        if model.geometry_arm == ARMS[1]:
            fresh = gaussian_weights(case['xyz'][idx],case['cloud'],case['mask'])
            cache_error = float((fresh-case['geometry_basis'][idx]).abs().max())
        else:
            cache_error = None
    if chunk_error > 3e-5 or permutation_error > 3e-5 or (cache_error is not None and cache_error > 3e-5):
        raise RuntimeError('Actual query/chunk/permutation/static-basis equivalence failed')
    model.train(); guard()
    return dict(real_case_key=case['key'],actual_queries=64,chunk_max_absolute_error=chunk_error,
        cloud_permutation_max_absolute_error=permutation_error,static_basis_max_absolute_error=cache_error,
        coordinate_gradient_L2=float(gradient.norm()),local_geometry_coordinate_gradient_L2=None if geometry_gradient is None else float(geometry_gradient.norm()),
        native25_held_constant=True,independent_offnode_FEM_accuracy_proven=False,passed=True)
