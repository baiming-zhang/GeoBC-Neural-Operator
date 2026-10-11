"""TRAIN teacher-operator diagnostic at original nodes, not FEM supervision.

The caller owns the genuine teacher checkpoint, immutable case identities,
optimizer, deadlines and budget. Only a student's operator receives gradients.
"""
from dataclasses import dataclass
import math
import torch
from .coarse_transolver_v19 import coarse_stencil
from .fno import query_grid
from .geometry_head_transfer_v32 import transfer_geometry_head, state_digest


CHUNK = 32768


@dataclass(frozen=True)
class NodeLatentNorm:
    node_count: int
    channels: int
    squared_norm: float


def freeze_teacher_geometry_head(model, teacher_state):
    receipt = transfer_geometry_head(model, teacher_state)
    for group in ('encoder', 'head'):
        getattr(model, group).requires_grad_(False)
    model.operator.requires_grad_(True)
    receipt.update(frozen_groups=['encoder', 'head'], trainable_groups=['operator'],
                   optimizer_parameter_count=sum(p.numel() for p in model.operator.parameters()),
                   objective='TRAIN teacher node-latent approximation, not FEM')
    return receipt


def original_node_latent(latent, xyz):
    if latent.shape != (1, 16, 16, 16, 16):
        raise ValueError('Complete R16 latent16 required')
    indices, weights = coarse_stencil(xyz)
    return query_grid(latent[0], indices, weights, CHUNK)


@torch.no_grad()
def complete_teacher_node_norm(teacher_latent, xyz, guard):
    if teacher_latent.requires_grad or not torch.isfinite(teacher_latent).all():
        raise ValueError('Frozen finite teacher latent required')
    total = 0.; consumed = 0
    for start in range(0, len(xyz), CHUNK):
        guard(); stop = min(start+CHUNK, len(xyz))
        values = original_node_latent(teacher_latent, xyz[start:stop])
        total += float(values.double().square().sum()); consumed += stop-start
    if not consumed or not math.isfinite(total) or total <= 0:
        raise ValueError('Complete nonzero teacher-node norm required; no epsilon rescue')
    return NodeLatentNorm(consumed, 16, total)


def full_teacher_node_backward(model, case, teacher_latent, norms, allowed_keys, guard):
    if case.get('scientific_group') != 'TRAIN' or case.get('key') not in allowed_keys:
        raise PermissionError('Fixed TRAIN cases only')
    if (norms.node_count != len(case['xyz']) or norms.channels != 16
            or not math.isfinite(norms.squared_norm) or norms.squared_norm <= 0):
        raise ValueError('Full original-node teacher normalization required')
    if teacher_latent.requires_grad or not torch.isfinite(teacher_latent).all():
        raise ValueError('Detached finite teacher target required')
    if any(p.requires_grad for group in ('encoder', 'head') for p in getattr(model, group).parameters()):
        raise RuntimeError('Teacher encoder/head must stay frozen')
    if not all(p.requires_grad for p in model.operator.parameters()):
        raise RuntimeError('All student operator parameters must be trainable')
    context, latent = model.encode_case(case)
    if context.requires_grad or not latent.requires_grad:
        raise RuntimeError('Frozen geometry context and trainable operator graph required')
    leaf = latent.detach().requires_grad_(True)
    total = 0.; consumed = 0
    for start in range(0, len(case['xyz']), CHUNK):
        guard(); stop = min(start+CHUNK, len(case['xyz']))
        xyz = case['xyz'][start:stop]
        with torch.no_grad(): target = original_node_latent(teacher_latent, xyz)
        prediction = original_node_latent(leaf, xyz)
        loss = (prediction.double()-target.double()).square().sum()/norms.squared_norm
        loss.backward(); total += float(loss.detach()); consumed += stop-start
    if (consumed != norms.node_count or leaf.grad is None
            or not torch.isfinite(leaf.grad).all() or not math.isfinite(total)):
        raise RuntimeError('Complete node coverage and finite joint latent gradient required')
    torch.autograd.backward(latent, leaf.grad)
    return total, dict(original_nodes_used=consumed, channels=16, node_sampling=False,
                      teacher_node_latent_L2_squared=norms.squared_norm,
                      operator_latent_gradient_L2=float(leaf.grad.norm()),
                      geometry_context_gradient_expected=False, head_gradient_expected=False,
                      FEM_loss_used_for_updates=False)


def frozen_group_digests(model):
    return {group: state_digest(getattr(model, group).state_dict()) for group in ('encoder', 'head')}
