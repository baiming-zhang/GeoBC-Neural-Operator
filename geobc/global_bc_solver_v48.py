"""Existing FNO/Transolver backbones with an explicit per-case global BC path."""
import torch
from torch import nn
from .common_grid_solver import CommonGridSolver
from .coarse_transolver_v19 import CoarseTransolver
from .global_bc_context_v47 import GeometryBCContext,global_bc_descriptor
from .geometry_head_transfer_v32 import state_digest

METHODS = ('FNO9_GLOBAL_BC','R16_TRANSOLVER_GLOBAL_BC')


def attach_descriptor(case):
    if not isinstance(case.get('key'),str) or not case['key']:
        raise ValueError('Explicit source case key required')
    if 'global_bc' in case or 'global_bc_source_key' in case:
        raise ValueError('Input descriptor must be prepared once by its owner')
    case['global_bc'] = global_bc_descriptor(case['xyz'],case['features'])
    case['global_bc_source_key'] = case['key']
    return case


class GlobalBCSolver(nn.Module):
    def __init__(self,method,seed=2026,device='cuda'):
        super().__init__()
        if method not in METHODS: raise ValueError('Two fixed global-BC families only')
        self.base = (CommonGridSolver('FNO9',seed) if method == METHODS[0]
                     else CoarseTransolver('transolver',seed,'cpu'))
        self.old_backbone_initial_digest = state_digest(self.base.state_dict())
        self.old_parameter_count = sum(p.numel() for p in self.base.parameters())
        self.old_profile = (dict(method='FNO9',counts=self.base.counts(),grid_resolution=64)
                            if method == METHODS[0] else self.base.profile())
        self.base.encoder = GeometryBCContext(self.base.encoder,seed)
        self.method = self.macro_family = method
        self.to(device)
        assert self.original_backbone_digest() == self.old_backbone_initial_digest

    def original_backbone_state(self):
        result = {}
        for name,value in self.base.state_dict().items():
            if name.startswith(('encoder.branch.','encoder.reference.')): continue
            if name.startswith('encoder.base.'): name = 'encoder.'+name[len('encoder.base.'):]
            result[name] = value
        return result

    def original_backbone_digest(self):
        return state_digest(self.original_backbone_state())

    def encode_case(self,case):
        if case.get('global_bc_source_key') != case.get('key') or 'global_bc' not in case:
            raise ValueError('Current case descriptor missing or from another case')
        # Always set it here, never rely on the previous case's module state.
        self.base.encoder.set_descriptor(case['global_bc'])
        return self.base.encode_case(case)

    def query(self,case,start,stop,context,latent):
        return self.base.query(case,start,stop,context,latent)

    def gradient_group(self,name):
        if name.startswith('base.encoder.base.'): return 'geometry_encoder'
        if name.startswith('base.encoder.branch.'): return 'global_BC_branch'
        if name.startswith('base.encoder.reference.'): return 'frozen_BC_reference'
        if name.startswith('base.head.'): return 'head'
        if name.startswith('base.operator.'): return 'operator'
        raise ValueError('Unclassified parameter '+name)

    def reference_digest(self):
        return state_digest(self.base.encoder.reference.state_dict())

    def profile(self):
        return dict(method=self.method,grid_resolution=64 if self.method == METHODS[0] else 16,
            parameter_count=sum(p.numel() for p in self.parameters()),
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            old_backbone_parameters=self.old_parameter_count,old_backbone_profile=self.old_profile,
            BC_context=self.base.encoder.profile(),old_backbone_initial_digest=self.old_backbone_initial_digest,
            precision='highest FP32; caller disables TF32',fresh_initialization=True,
            inherited_teacher_passes=0,physical_constraint_enforced=False,
            baseline_information_unchanged=True,capacity_matched_to_old_baseline=False,
            independent_offnode_FEM_accuracy_measured=False)
