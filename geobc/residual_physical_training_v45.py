"""Whole-node physical residual gradients with genuinely frozen global features.

No dataset/checkpoint bodies, optimizer or runtime ownership here. Unlike the
joint operator helper, this deliberately never differentiates context/latent.
"""
from .physical_field_loss import physical_relative_squared_loss


def full_residual_case_backward(model, case, target, scale, norms, context, latent,
                                allowed_keys, guard=lambda: None):
    if case.get('scientific_group') != 'TRAIN' or case.get('key') not in allowed_keys:
        raise PermissionError('Original fixed TRAIN scope required')
    count = len(case['xyz'])
    if target.shape != (count, 4) or norms.node_count != count:
        raise ValueError('Complete physical norms/target required')
    if context.requires_grad or latent.requires_grad:
        raise ValueError('Frozen cached global features must have no autograd graph')
    if context.shape != (1, 32) or latent.shape != (1, 16, 64, 64, 64):
        raise ValueError('Original FNO9 context/latent required')
    if any(p.requires_grad for p in model.teacher.parameters()) or any(p.requires_grad for p in model.reference.parameters()):
        raise ValueError('Teacher/reference cannot be trainable')
    total = 0.; consumed = 0
    for start in range(0, count, 32768):
        guard(); stop = min(start+32768, count)
        prediction = model.query(case, start, stop, context, latent)
        loss = physical_relative_squared_loss(prediction, target[start:stop], scale, norms)
        loss.backward(); total += float(loss.detach()); consumed += stop-start
    if consumed != count or any(p.grad is not None for p in model.teacher.parameters()) or any(p.grad is not None for p in model.reference.parameters()):
        raise RuntimeError('Incomplete physical coverage or unintended frozen gradient')
    return total
