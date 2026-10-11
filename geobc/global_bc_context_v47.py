"""Input-only global BC routing into an existing geometry context.

Existing inputs already contain loads/BCs. This is a representation/routing
experiment, not new information or a physical constraint. RBE2 CM membership
is a coupling role; it is never interpreted as grounded SPC/displacement.
"""
import copy
import torch
from torch import nn

DESCRIPTOR_DIM = 68


def global_bc_descriptor(xyz, features):
    if xyz.ndim != 2 or xyz.shape[1] != 3 or features.shape != (len(xyz),25) or not len(xyz):
        raise ValueError('Original native xyz3/features25 required')
    if xyz.dtype != torch.float32 or features.dtype != xyz.dtype or xyz.device != features.device:
        raise ValueError('Original common FP32 inputs on one device required')
    x,f = xyz.detach(),features.detach()
    if not torch.isfinite(x).all() or not torch.isfinite(f).all():
        raise ValueError('Finite original inputs required')
    if not torch.all((f[:,:7] == 0) | (f[:,:7] == 1)):
        raise ValueError('Original binary CM/member indicators required')
    constants = torch.cat((f[:,7:16],f[:,24:25]),dim=1)
    if not torch.equal(constants,constants[0].expand_as(constants)):
        raise ValueError('Load/material/length must be original per-case constants')
    # xyz=(xyz_mm-bboxmin)/L and native16:19=(xyz_mm-loadref)/L.
    # Their difference gives the actual normalized load-reference position.
    reference = x-f[:,16:19]; anchor = reference.mean(0)
    if not torch.allclose(reference,anchor.expand_as(reference),atol=3e-5,rtol=0):
        raise ValueError('Original normalized load-reference offsets disagree')
    groups = []
    for column in range(7):
        mask = f[:,column] == 1; count = int(mask.sum())
        fraction = f.new_tensor([count/len(x)])
        if count:
            points = x[mask]; center = points.mean(0)
            spread = ((points-center).square().mean(0)).sqrt()
        else:
            # An absent input role is represented explicitly, not an imposed
            # boundary value or a replacement for a zero physical target norm.
            center,spread = x.new_zeros(3),x.new_zeros(3)
        groups.append(torch.cat((fraction,center,spread)))
    distances = f[:,19:23]
    means = distances.mean(0); spreads = distances.std(dim=0,unbiased=False); minima = distances.amin(0)
    # Symmetric across the four original RBE2 reference columns, whose GRID-ID
    # order is not a semantic hole identity across CADs.
    summary = torch.stack((means.mean(),means.std(unbiased=False),spreads.mean(),spreads.amax(),minima.amin(),minima.amax()))
    result = torch.cat((constants[0],anchor,*groups,summary)).reshape(1,DESCRIPTOR_DIM)
    if not torch.isfinite(result).all(): raise ValueError('Finite global descriptor required')
    return result


class GeometryBCContext(nn.Module):
    """Set an explicit case descriptor before calling the supplied base encoder.

    Default descriptor is absent and raises, rather than silently ignoring BCs.
    A future caller must own data, fixed schedule, scaler and actual GPU checks.
    """
    def __init__(self,base_encoder,seed=2026):
        super().__init__(); self.base = base_encoder
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.branch = nn.Sequential(nn.Linear(DESCRIPTOR_DIM,64),nn.GELU(),nn.Linear(64,32))
        self.reference = copy.deepcopy(self.branch).eval().requires_grad_(False)
        self.case_descriptor = None

    def train(self,mode=True):
        super().train(mode); self.reference.eval(); return self

    def set_descriptor(self,value):
        if value.shape != (1,DESCRIPTOR_DIM) or value.requires_grad or not torch.isfinite(value).all():
            raise ValueError('One detached input-only case descriptor required')
        if value.device != next(self.branch.parameters()).device or value.dtype != torch.float32:
            raise ValueError('Descriptor must share module device and FP32')
        self.case_descriptor = value

    def forward(self,cloud,mask):
        if self.case_descriptor is None: raise RuntimeError('Explicit descriptor must be set for each case')
        context = self.base(cloud,mask)
        delta = self.branch(self.case_descriptor)-self.reference(self.case_descriptor)
        return context+delta

    def profile(self):
        return dict(descriptor_dim=DESCRIPTOR_DIM,additional_trainable_parameters=sum(p.numel() for p in self.branch.parameters()),
            additional_frozen_parameters=sum(p.numel() for p in self.reference.parameters()),
            roles='Actual RBE2 CM columns0:6 separately and loaded RBE3 member6; not groundedSPC',
            original_information_changed=False,adds_targets=False,load_material_length_scales='Existing original25 constants, no scaler fit',
            effect='Global BC branch minus fixed initialbranch adds to existing32 geometrycontext; initially exactoldcontext',
            initialized_backbone_changed=False,owns_case_or_launch=False)
