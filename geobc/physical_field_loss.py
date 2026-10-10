"""Optional whole-case physical relative-squared-error objective.

No files, scaler fitting, clipping, field sign changes or model construction.
Current experiments do not import this module. Training activation requires a
separate scientific decision and a genuine full-model gradient/resource check.
"""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class CaseNorms:
    node_count: int
    displacement_squared_norm: float
    stress_squared_norm: float

    def __post_init__(self):
        if type(self.node_count) is not int or self.node_count < 1:
            raise ValueError('A nonempty complete original-node case is required')
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.displacement_squared_norm, self.stress_squared_norm)):
            raise ValueError('Zero/nonfinite field norms are undefined; no epsilon replacement')


@dataclass(frozen=True)
class PhysicalScale:
    mean: tuple
    std: tuple
    stress_scale: float | None

    @classmethod
    def from_mapping(cls, value):
        if value['channels'] != [0, 1, 2, 3]:
            raise ValueError('Expected original signed [Ux,Uy,Uz,S] channel order')
        mean, std = tuple(value['mean']), tuple(value['std'])
        if len(mean) != 4 or len(std) != 4 or not all(math.isfinite(x) for x in mean):
            raise ValueError('Four finite scaler means are required')
        if not all(math.isfinite(x) and x > 0 for x in std):
            raise ValueError('Four finite positive scaler standard deviations are required')
        scale = value['stress_scale']
        if scale is not None and (not math.isfinite(scale) or scale <= 0):
            raise ValueError('Invalid signed-asinh stress scale')
        return cls(mean, std, scale)


def norms_from_complete_physical_target(physical_target):
    """Use the original physical truth, not a roundtrip of encoded targets.

    Caller supplies one complete TRAIN case already loaded under its protocol;
    no dataset or GT file is opened here. These norms remain fixed for chunks.
    """
    if physical_target.ndim != 2 or physical_target.shape[1] != 4:
        raise ValueError('Expected a whole [N,4] physical target')
    if not physical_target.is_floating_point() or not torch.isfinite(physical_target).all():
        raise ValueError('Physical targets must be finite floating values')
    value = physical_target.detach().to(torch.float64)
    return CaseNorms(len(value), float(value[:, :3].square().sum()),
                     float(value[:, 3].square().sum()))


def decode_physical(encoded, scale):
    """Differentiable FP64 decode matching benchmark.FieldScaler.decode.

    FP64 is confined to the pointwise objective, not the model backbone.
    Signed stresses are decoded directly with sinh, without abs or clipping.
    """
    if encoded.ndim != 2 or encoded.shape[1] != 4 or not encoded.is_floating_point():
        raise ValueError('Expected floating encoded [N,4]')
    value = encoded.to(torch.float64)
    mean = value.new_tensor(scale.mean)
    std = value.new_tensor(scale.std)
    transformed = value * std + mean
    stress = transformed[:, 3:4]
    if scale.stress_scale is not None:
        stress = scale.stress_scale * stress.sinh()
    return torch.cat((transformed[:, :3], stress), dim=1)


def physical_relative_squared_loss(encoded, physical_target, scale, norms):
    """Contribution of a node chunk to .5*(relative_U_L2²+relative_S_L2²).

    Sum contributions of all nonoverlapping original-node chunks exactly once.
    Do not average chunk contributions or compute norms per chunk. This is a
    mean of squared relative errors, not the reporting primary score, which
    averages the unsquared relative L2 values.
    """
    if encoded.shape != physical_target.shape or encoded.device != physical_target.device:
        raise ValueError('Predictions and physical targets must share shape/device')
    if not physical_target.is_floating_point() or not 0 < len(encoded) <= norms.node_count:
        raise ValueError('Invalid physical target chunk')
    residual = decode_physical(encoded, scale) - physical_target.detach().to(torch.float64)
    displacement = residual[:, :3].square().sum() / norms.displacement_squared_norm
    stress = residual[:, 3].square().sum() / norms.stress_squared_norm
    loss = .5 * (displacement + stress)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite physical objective; do not clip or silently rescue')
    return loss


def full_common_case_backward(model, case, physical_target, scale, norms, guard=lambda: None):
    """Joint exact whole-case chain rule with the existing decoder chunk size.

    Future common-grid driver only; no optimizer, initialization or RNG edits.
    This function has not yet passed an actual trained-model fullN runtime gate.
    """
    count = len(case['xyz'])
    if count != norms.node_count or physical_target.shape != (count, 4):
        raise ValueError('Norms and targets must cover every original node of this case')
    context, latent = model.encode_case(case)
    if context.shape != (1, 32) or latent.shape != (1, 16, 64, 64, 64):
        raise ValueError('Expected the frozen common-grid representation')
    c = context.detach().requires_grad_(True)
    l = latent.detach().requires_grad_(True)
    total = 0.
    consumed = 0
    for start in range(0, count, 32768):
        guard()
        stop = min(start + 32768, count)
        encoded = model.query(case, start, stop, c, l)
        loss = physical_relative_squared_loss(encoded, physical_target[start:stop], scale, norms)
        loss.backward()
        total += float(loss.detach())
        consumed += stop-start
    if consumed != count or c.grad is None or l.grad is None:
        raise RuntimeError('Incomplete original-node coverage or missing joint leaf gradients')
    torch.autograd.backward((context, latent), (c.grad, l.grad))
    return total, {'context': float(c.grad.norm()), 'latent': float(l.grad.norm())}
