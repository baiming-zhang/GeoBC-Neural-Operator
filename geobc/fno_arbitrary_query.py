"""Fixed-context FNO query adapter; no IO, model creation or data loading.

The native25 extension is a declared full-TET10 nodal interpolant. It is
not a new analytic BC feature schema. Physical gradients use the complete
reference-coordinate graph, including local features and latent stencils.
"""
from __future__ import annotations

ARMS = ('FNO_latent_query_grid25', 'FNO_latent_query_native25')
EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


def tet10_queries(physical_element_nodes, normalized_element_nodes,
                  element_features25, reference_r):
    """Caller supplies explicit selected cells [K,10,*] and r=[L1,L2,L3].

    Cell location, connectivity conformity, domain membership, units and
    normalization are caller obligations. This does not locate a whole CAD.
    """
    import torch
    k = len(reference_r)
    if reference_r.shape != (k, 3) or k == 0:
        raise ValueError('Expected nonempty reference coordinates [K,3]')
    for value, channels in ((physical_element_nodes, 3), (normalized_element_nodes, 3),
                            (element_features25, 25)):
        if value.shape != (k, 10, channels) or value.device != reference_r.device:
            raise ValueError('Selected element tensor shape/device differs')
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError('Element inputs must be finite floating tensors')
    if not reference_r.is_floating_point() or not torch.isfinite(reference_r).all():
        raise ValueError('Reference coordinates must be finite floating tensors')
    bary = torch.cat((1-reference_r.sum(-1, keepdim=True), reference_r), dim=-1)
    if bary.min() < -2e-6 or bary.max() > 1+2e-6:
        raise ValueError('Query lies outside the supplied reference element')
    edges = torch.tensor(EDGES, device=reference_r.device)
    weights = torch.cat((bary*(2*bary-1),
                         4*bary[:, edges[:, 0]]*bary[:, edges[:, 1]]), dim=-1)
    dL = reference_r.new_tensor([[-1., -1., -1.], [1., 0., 0.],
                                [0., 1., 0.], [0., 0., 1.]])
    derivatives = torch.cat(((4*bary-1)[..., None]*dL,
        4*(bary[:, edges[:, 1], None]*dL[edges[:, 0]] +
           bary[:, edges[:, 0], None]*dL[edges[:, 1]])), dim=1)
    def interpolate(value):
        return (value*weights.to(value.dtype)[..., None]).sum(1)
    jacobian = torch.einsum('kir,kic->kcr', derivatives.to(physical_element_nodes.dtype),
                            physical_element_nodes)
    return {'physical_points': interpolate(physical_element_nodes),
            'xyz': interpolate(normalized_element_nodes),
            'features': interpolate(element_features25), 'physical_jacobian': jacobian}


def query_fixed_case(model, reference_case, xyz, context, latent_grid,
                     native25=None, chunk_size=32768):
    """Reuse caller's fixed context/grid; recompute stencils at new points.

    Returns encoded signed [K,4]; caller owns decode and physical units.
    No geometry re-encoding, nearest-node fallback or occupancy reweighting.
    """
    import torch
    from .solver_selection_data import _stencil
    if model.arm not in ARMS or chunk_size < 1:
        raise ValueError('Unsupported FNO query arm or chunk size')
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError('Expected nonempty normalized xyz [K,3]')
    if xyz.device != latent_grid.device or xyz.dtype != latent_grid.dtype:
        raise ValueError('Coordinates and latent grid must share dtype/device')
    if model.arm == ARMS[1]:
        if native25 is None or native25.shape != (len(xyz), 25):
            raise ValueError('native25 needs an explicit [K,25] extension')
        if native25.device != xyz.device or native25.dtype != xyz.dtype or not torch.isfinite(native25).all():
            raise ValueError('native25 must share finite floating dtype/device')
    indices, weights = _stencil(xyz, 64)
    query_case = dict(reference_case)
    query_case.update(xyz=xyz, features=native25, query_indices=indices, query_weights=weights)
    return torch.cat([model.query(query_case, start, min(start+chunk_size, len(xyz)),
                                 context, latent_grid)
                      for start in range(0, len(xyz), chunk_size)], dim=0)


def physical_query_gradients(pointwise_fields, reference_r, physical_jacobian):
    """d(encoded fields)/d(physical coordinates), including all r paths.

    Assumes fixed context and independent pointwise decoding. Do not apply
    sum-gradient shortcuts to a Transformer with query-to-query attention.
    Scaler decode must occur before this call for physical U/S gradients.
    Singular Jacobians fail; no pseudoinverse replacement. Mesh regularity
    and condition limits are caller obligations.
    """
    import torch
    if (pointwise_fields.ndim != 2 or len(pointwise_fields) != len(reference_r)
            or physical_jacobian.shape != (len(reference_r), 3, 3)):
        raise ValueError('Field/Jacobian shape differs')
    rows = []
    for channel in range(pointwise_fields.shape[1]):
        grad, = torch.autograd.grad(pointwise_fields[:, channel].sum(), reference_r,
                                    retain_graph=True)
        rows.append(torch.linalg.solve(physical_jacobian.transpose(-1, -2),
                    grad.to(physical_jacobian.dtype)[..., None]).squeeze(-1))
    return torch.stack(rows, dim=1)
