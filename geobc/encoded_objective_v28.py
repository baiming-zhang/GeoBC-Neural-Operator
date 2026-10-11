"""Whole-original-node standardized four-channel MSE, equal case weighting.

This is a training-objective control, not a physical-error metric. Three U
channels and one signed-asinh S channel receive equal encoded-channel weight.
Caller owns the frozen TRAIN-only scaler and original aligned target cache.
"""
import torch


def full_encoded_backward(model, case, encoded_target, allowed_keys, guard):
    if case.get('scientific_group') != 'TRAIN' or case.get('key') not in allowed_keys:
        raise PermissionError('Fixed TRAIN cases only')
    count = len(case['xyz'])
    if not count or encoded_target.shape != (count, 4) or encoded_target.dtype != torch.float32:
        raise ValueError('Complete original-node standardized FP32 target4 required')
    if encoded_target.device != case['xyz'].device or not torch.isfinite(encoded_target).all():
        raise ValueError('Finite target and input must share device')
    context, latent = model.encode_case(case)
    c = context.detach().requires_grad_(True); l = latent.detach().requires_grad_(True)
    total = 0.; consumed = 0
    for start in range(0, count, 32768):
        guard(); stop = min(start+32768, count)
        prediction = model.query(case, start, stop, c, l)
        loss = (prediction-encoded_target[start:stop].detach()).square().sum()/(4*count)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite encoded MSE; no clipping or rescue')
        loss.backward(); total += float(loss.detach()); consumed += stop-start
    if consumed != count or c.grad is None or l.grad is None or not torch.isfinite(c.grad).all() or not torch.isfinite(l.grad).all():
        raise RuntimeError('Complete original-node coverage and finite joint leaf gradients required')
    torch.autograd.backward((context, latent), (c.grad, l.grad))
    return total, dict(geometry_context_L2=float(c.grad.norm()), operator_latent_L2=float(l.grad.norm()),
                      original_nodes_used=count, node_sampling=False, loss='mean standardized encoded4 squared error')
