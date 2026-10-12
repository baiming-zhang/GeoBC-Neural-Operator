"""Continuous stress magnitude/sign package; original targets/scaler preserved."""
import torch
from torch import nn
from torch.nn import functional as F
from .common_grid_solver import CommonGridSolver
from .fno import query_grid
from .physical_field_loss import physical_relative_squared_loss
from .fno_stress_specialist_v54 import tree_digest

PACKAGE = 'FNO9_SIGN_MAGNITUDE_AUX'


class SignedStressFNO(CommonGridSolver):
    def __init__(self, seed, scaler):
        super().__init__('FNO9', seed)
        canonical = tree_digest(self.state_dict())
        body = {name: tree_digest(getattr(self, name).state_dict()) for name in ('encoder', 'operator')}
        old = self.head[-1]
        u = tree_digest(dict(weight=old.weight[:3].detach(), bias=old.bias[:3].detach()))
        hidden = tree_digest(self.head[:-1].state_dict())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            new = nn.Linear(128, 5)
        with torch.no_grad():
            new.weight[:4].copy_(old.weight); new.bias[:4].copy_(old.bias)
        self.head[-1] = new
        assert hidden == tree_digest(self.head[:-1].state_dict())
        assert u == tree_digest(dict(weight=new.weight[:3].detach(), bias=new.bias[:3].detach()))
        assert body == {name: tree_digest(getattr(self, name).state_dict()) for name in body}
        assert scaler['stress_scale'] > 0 and scaler['channels'] == [0, 1, 2, 3]
        self.register_buffer('stress_encoded_mean', torch.tensor(scaler['mean'][3], dtype=torch.float32))
        self.register_buffer('stress_encoded_std', torch.tensor(scaler['std'][3], dtype=torch.float32))
        self.package_arm = PACKAGE
        self.initial_copy_proof = dict(canonical_model_digest=canonical, shared_body_digests=body,
            canonical_hidden_and_U_rows_copied=True, added_output_row_parameters=129,
            initial_stress_representation_changed=True)

    def query_parts(self, case, start, stop, context, latent):
        values = query_grid(latent[0], case['query_indices'][start:stop], case['query_weights'][start:stop], 32768)
        raw = self.head(torch.cat((case['xyz'][start:stop], case['features'][start:stop],
            context.expand(stop-start, -1), values), dim=1))
        stress = (torch.asinh(F.softplus(raw[:, 3])*torch.tanh(raw[:, 4]))-self.stress_encoded_mean)/self.stress_encoded_std
        return torch.cat((raw[:, :3], stress[:, None]), dim=1), raw[:, 4]

    def query(self, case, start, stop, context, latent):
        return self.query_parts(case, start, stop, context, latent)[0]


def build_solver(seed, scaler):
    model = SignedStressFNO(seed, scaler).cuda()
    assert sum(p.numel() for p in model.parameters()) == 4526987
    return model


def full_backward(model, case, physical, scale, norms, guard=lambda: None):
    if case['scientific_group'] != 'TRAIN':
        raise PermissionError('Only TRAIN may supervise sign or physical fields')
    count = len(case['xyz'])
    assert physical.shape == (count, 4) and count == norms.node_count
    context, latent = model.encode_case(case)
    assert context.shape == (1, 32) and latent.shape == (1, 16, 64, 64, 64)
    c, l = context.detach().requires_grad_(True), latent.detach().requires_grad_(True)
    total = physical_sum = aux_sum = 0.; consumed = 0
    for start in range(0, count, 32768):
        guard(); stop = min(start+32768, count)
        encoded, logits = model.query_parts(case, start, stop, c, l)
        target = physical[start:stop].detach()
        loss_physical = physical_relative_squared_loss(encoded, target, scale, norms)
        signed_s = target[:, 3].to(torch.float64)
        bce = F.binary_cross_entropy_with_logits(2*logits, (signed_s > 0).to(logits.dtype), reduction='none')
        loss_aux = (bce.to(torch.float64)*signed_s.square()/norms.stress_squared_norm).sum()
        loss = loss_physical + .1*loss_aux
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite joint physical/sign objective')
        loss.backward(); total += float(loss.detach()); physical_sum += float(loss_physical.detach()); aux_sum += float(loss_aux.detach())
        consumed += stop-start
    assert consumed == count and c.grad is not None and l.grad is not None
    torch.autograd.backward((context, latent), (c.grad, l.grad))
    return total, {'context': float(c.grad.norm()), 'latent': float(l.grad.norm())}, dict(
        physical_relative_squared=physical_sum, energy_weighted_sign_BCE=aux_sum, auxiliary_weight=.1, total_objective=total)
