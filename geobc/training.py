"""Chunked full-node loss with encoder and operator gradients."""

import torch

def full_case_backward(model, case, guard=lambda: None):
    """Exact full original-node MSE chain rule with bounded decoder memory."""
    context, latent = model.encode_case(case)
    assert context.shape == (1, 32) and latent.shape == (1, 16, 64, 64, 64)
    c = context.detach().requires_grad_(True)
    l = latent.detach().requires_grad_(True)
    count = len(case['xyz'])
    total = 0.0
    assert case['target'].shape == (count, 4)
    for start in range(0, count, 32768):
        guard()
        stop = min(start + 32768, count)
        pred = model.query(case, start, stop, c, l)
        loss = (pred - case['target'][start:stop]).square().mean() * ((stop - start) / count)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        total += float(loss.detach())
    assert c.grad is not None and l.grad is not None
    torch.autograd.backward((context, latent), (c.grad, l.grad))
    return (total, {'context': float(c.grad.norm()), 'latent': float(l.grad.norm())})

def parameter_gradients(model):
    result = {}
    for name, module in [('encoder', model.encoder), ('head', model.head), ('operator', model.operator)]:
        missing = []
        finite = True
        energy = 0.0
        for key, p in module.named_parameters():
            if p.grad is None:
                missing.append(key)
                continue
            finite &= bool(torch.isfinite(p.grad).all())
            energy += float(p.grad.detach().double().square().sum())
        result[name] = {'missing': missing, 'finite': finite, 'L2': energy ** 0.5}
    return result
