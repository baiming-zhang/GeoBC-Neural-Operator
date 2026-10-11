"""Original TS on all fixed FEM nodes, with the existing continuous decoder.

Only the context representation changes from the v65 R64 composition. The
official original-TS weights/math, PointNet and TCNN parameters are retained.
Queries never become attention/context nodes. Off-node queries require an
explicit conforming TET10 cell and reference coordinate; there is no locator
or independent off-node FEM accuracy claim. The declared extension is C0,
not generally C1, and interpolated native25 flags are not analytic BC fields.
"""
import torch
from .transolver_bf16_hash_v65 import BF16TransolverHash
from .fno_arbitrary_query import EDGES, tet10_queries

FAMILY = 'TS_ORIGINAL_BF16_NATIVE_HASH'


class NativeTransolverHash(BF16TransolverHash):
    def __init__(self, seed=2026, device='cuda'):
        super().__init__(seed=seed, device=device)
        self.arm = self.macro_family = FAMILY

    @staticmethod
    def native_tokens(case, context):
        xyz, features = case['xyz'], case['features']
        count = len(xyz)
        if (xyz.shape != (count, 3) or not count or
                features.shape != (count, 25) or context.shape != (1, 32)):
            raise ValueError('Complete fixed native-node geometry/BC required')
        return torch.cat((xyz, xyz.new_ones((count, 1)), features,
                          context.expand(count, -1)), dim=-1).unsqueeze(0)

    def encode_case(self, case):
        context = self.encoder(case['cloud'], case['mask'])
        tokens = self.native_tokens(case, context)
        if tokens.device.type != 'cuda':
            raise ValueError('This branch requires the real CUDA/TCNN backend')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            latent = self.operator(tokens)
        if latent.shape != (1, len(case['xyz']), 16):
            raise ValueError('Every reference node must retain its latent')
        return context, latent[0].float()

    def query(self, case, start, stop, context, latent):
        count = len(case['xyz'])
        if not 0 <= start < stop <= count or latent.shape != (count, 16):
            raise ValueError('Invalid fixed-reference native query')
        return self.decode_points(case['xyz'][start:stop],
                                  case['features'][start:stop], context,
                                  latent[start:stop])

    def query_elements(self, case, context, latent, element_indices,
                       reference_r, physical_xyz):
        """Explicit valid TET10 interpolation; no re-encoding query points."""
        count = len(reference_r)
        if (not count or element_indices.shape != (count, 10) or
                element_indices.dtype != torch.long or
                element_indices.device != latent.device or
                physical_xyz.shape != case['xyz'].shape or
                physical_xyz.device != latent.device or
                latent.shape != (len(case['xyz']), 16)):
            raise ValueError('Explicit fixed-mesh element dimensions differ')
        if element_indices.min() < 0 or element_indices.max() >= len(latent):
            raise ValueError('Original element node outside fixed mesh')
        queries = tet10_queries(physical_xyz[element_indices],
                                case['xyz'][element_indices],
                                case['features'][element_indices], reference_r)
        bary = torch.cat((1-reference_r.sum(-1, keepdim=True), reference_r), -1)
        edges = torch.tensor(EDGES, device=reference_r.device)
        weights = torch.cat((bary*(2*bary-1),
                             4*bary[:, edges[:, 0]]*bary[:, edges[:, 1]]), -1)
        values = (latent[element_indices]*weights.to(latent.dtype)[..., None]).sum(1)
        return self.decode_points(queries['xyz'], queries['features'],
                                  context, values), queries

    def query_points(self, *args, **kwargs):
        raise ValueError('Native latent requires an explicit valid TET10 cell; R64 stencil is not this model')

    @property
    def source_config(self):
        result = super().source_config
        result.update(method='Fixed whole-native originalTS and same continuous Hash decoder',
            global_input='All original FEM nodes: xyz3 + occupancy1 + native25 + PointNet32',
            continuous_query_policy='Fixed wholemesh context; TET10 latent/native25 extension then existing TCNN decoder',
            context_representation_changed=True, parameters_identical_to_v65=True,
            empty_voxel_causality_established=False,
            arbitrary_xyz_locator_implemented=False,
            independent_offnode_FEM_accuracy_measured=False)
        result['operator']['input'] = result['global_input']
        return result


def full_native_physical_backward(model, case, physical_target, scale, norms,
                                 guard=lambda: None):
    """Exact full-node physical joint objective, chunk only decoder memory."""
    from .physical_field_loss import physical_relative_squared_loss
    count = len(case['xyz'])
    if count != norms.node_count or physical_target.shape != (count, 4):
        raise ValueError('Physical target/norms must cover the complete case')
    guard()
    context, latent = model.encode_case(case)
    c = context.detach().requires_grad_(True)
    l = latent.detach().requires_grad_(True)
    total, consumed = 0., 0
    for start in range(0, count, 32768):
        guard()
        stop = min(start+32768, count)
        prediction = model.query(case, start, stop, c, l)
        loss = physical_relative_squared_loss(prediction,
                    physical_target[start:stop], scale, norms)
        loss.backward()
        total += float(loss.detach())
        consumed += stop-start
    if consumed != count or c.grad is None or l.grad is None:
        raise RuntimeError('Incomplete coverage or missing joint gradients')
    if not torch.isfinite(c.grad).all() or not torch.isfinite(l.grad).all():
        raise FloatingPointError('Nonfinite native latent/context gradient')
    torch.autograd.backward((context, latent), (c.grad, l.grad))
    return total, dict(context=float(c.grad.norm()), latent=float(l.grad.norm()))
