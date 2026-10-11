"""Native-node representation control for the existing common-grid ++ model.

The parameter layout, initialization, geometry encoder and 76-channel query
head are inherited unchanged. All reference FEM nodes contribute to the
official operator in one forward; independent spatial chunk attention is
never used. The fourth input channel is native occupancy=1, preserving the
R64 arm's 61-channel input layout and parameter count.

This is an implementation candidate, not a released training experiment.
Fixed-reference TET10 interpolation defines the off-node latent/feature
extension. It is C0 on conforming faces, not independent FEM truth or C1.
"""
import torch
from .common_grid_solver import CommonGridSolver
from .fno_arbitrary_query import EDGES, tet10_queries


class NativeContextSolver(CommonGridSolver):
    def __init__(self, seed=2026):
        super().__init__('PP256', seed)
        self.arm = 'PP256_NATIVE'

    def encode_case(self, case):
        xyz, features = case['xyz'], case['features']
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
            raise ValueError('Reference xyz must be nonempty [N,3]')
        if features.shape != (len(xyz), 25):
            raise ValueError('Reference features must be [N,25]')
        context = self.encoder(case['cloud'], case['mask'])
        tokens = torch.cat((xyz, xyz.new_ones((len(xyz), 1)), features,
                            context.expand(len(xyz), -1)), dim=-1).unsqueeze(0)
        latent = self.operator(tokens)
        if latent.shape != (1, len(xyz), 16):
            raise ValueError('Operator must return [1,N,16]')
        return context, latent[0]

    def decode_points(self, xyz, features, context, values):
        count = len(xyz)
        if (xyz.shape != (count, 3) or features.shape != (count, 25)
                or context.shape != (1, 32) or values.shape != (count, 16)):
            raise ValueError('Query/context/latent dimensions differ')
        if any(t.device != xyz.device or t.dtype != xyz.dtype
               for t in (features, context, values)):
            raise ValueError('Queries must share dtype and device')
        return self.head(torch.cat((xyz, features, context.expand(count, -1), values), dim=-1))

    def query(self, case, start, stop, context, latent):
        """Original reference nodes: retain their direct operator features."""
        if not 0 <= start < stop <= len(case['xyz']):
            raise ValueError('Invalid original-node query slice')
        if latent.shape != (len(case['xyz']), 16):
            raise ValueError('Latent does not match the fixed reference mesh')
        return self.decode_points(case['xyz'][start:stop], case['features'][start:stop],
                                  context, latent[start:stop])

    def query_elements(self, case, context, latent, element_indices, reference_r,
                       physical_xyz):
        """Explicit valid TET10 cells, without re-encoding the query set.

        Caller supplies original conforming connectivity [K,10], reference
        coordinates [K,3], and physical FEM coordinates [N,3] in mm. Cell
        location, normalization and Jacobian regularity are caller obligations.
        No nearest-node fallback, new field labels or occupancy masking.
        """
        count = len(reference_r)
        if (count == 0 or element_indices.shape != (count, 10)
                or element_indices.dtype != torch.long
                or element_indices.device != latent.device
                or physical_xyz.shape != case['xyz'].shape
                or physical_xyz.device != latent.device
                or latent.shape != (len(case['xyz']), 16)):
            raise ValueError('Explicit reference element dimensions/device differ')
        if element_indices.min() < 0 or element_indices.max() >= len(latent):
            raise ValueError('Element index outside the fixed mesh')
        queries = tet10_queries(physical_xyz[element_indices], case['xyz'][element_indices],
                                case['features'][element_indices], reference_r)
        bary = torch.cat((1-reference_r.sum(-1, keepdim=True), reference_r), dim=-1)
        edges = torch.tensor(EDGES, device=reference_r.device)
        weights = torch.cat((bary*(2*bary-1),
                             4*bary[:, edges[:, 0]]*bary[:, edges[:, 1]]), dim=-1)
        values = (latent[element_indices]*weights.to(latent.dtype)[..., None]).sum(1)
        result = self.decode_points(queries['xyz'], queries['features'], context, values)
        return result, queries


def full_native_case_backward(model, case, guard=lambda: None, chunk_size=32768):
    """Exact full-node encoded4 MSE chain rule; chunks bound head memory only."""
    if chunk_size < 1:
        raise ValueError('Positive decoder chunk size required')
    context, latent = model.encode_case(case)
    count = len(case['xyz'])
    if case['target'].shape != (count, 4):
        raise ValueError('Target must match all original nodes')
    c = context.detach().requires_grad_(True)
    l = latent.detach().requires_grad_(True)
    total = 0.
    for start in range(0, count, chunk_size):
        guard()
        stop = min(start+chunk_size, count)
        prediction = model.query(case, start, stop, c, l)
        loss = (prediction-case['target'][start:stop]).square().mean()*((stop-start)/count)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite full-node objective')
        loss.backward()
        total += float(loss.detach())
    if c.grad is None or l.grad is None:
        raise RuntimeError('Missing decoder-to-context/latent chain rule')
    torch.autograd.backward((context, latent), (c.grad, l.grad))
    return total, {'context': float(c.grad.norm()), 'latent': float(l.grad.norm())}


__all__ = ['NativeContextSolver', 'full_native_case_backward']
