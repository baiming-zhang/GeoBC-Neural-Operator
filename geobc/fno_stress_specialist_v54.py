"""Exact original half-weight stress term; shared FNO trajectory, TRAIN only."""
from .physical_field_loss import decode_physical


def tree_digest(value):
    """Exact typed model/Adam/RNG bytes, without mutation or serialization."""
    import hashlib
    import json
    import numpy as np
    import torch
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            h.update(json.dumps(['tensor',str(v.dtype),list(v.shape)]).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,np.ndarray):
            h.update(json.dumps(['array',str(v.dtype),list(v.shape)]).encode())
            h.update(v.tobytes(order='C'))
        elif isinstance(v,dict):
            h.update(b'dict')
            for k in sorted(v,key=lambda k:(type(k).__name__,repr(k))):visit(k);visit(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode())
            for item in v:visit(item)
        else:h.update(json.dumps([type(v).__name__,v],allow_nan=False).encode())
        h.update(b'\0')
    visit(value)
    return h.hexdigest()


def half_stress_squared_loss(encoded, physical_target, scale, norms):
    import torch
    if encoded.shape != physical_target.shape or encoded.device != physical_target.device:
        raise ValueError('Same complete chunk shape/device required')
    if encoded.ndim!=2 or encoded.shape[1]!=4 or not physical_target.is_floating_point() or not 0<len(encoded)<=norms.node_count:
        raise ValueError('Positive original-node chunk and fixed full-case norm required')
    residual=decode_physical(encoded,scale)[:,3]-physical_target[:,3].detach().to(torch.float64)
    loss=.5*residual.square().sum()/norms.stress_squared_norm
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite signed physical stress loss')
    return loss


def full_stress_backward(model,case,cache,scale,allowed_keys,guard=lambda:None):
    import torch
    if case.get('scientific_group')!='TRAIN' or case.get('key')not in allowed_keys:
        raise PermissionError('Only fixed original TRAIN100 cases may update')
    if model.macro_family!='FNO9' or not cache.completed:
        raise ValueError('Same existing FNO9 and completed physical cache required')
    key=case['key'];target=cache.physical_targets[key];norm=cache.case_norms[key];n=len(case['xyz'])
    if n!=norm.node_count or target.shape!=(n,4):raise ValueError('Whole original-node case required')
    context,latent=model.encode_case(case)
    if context.shape!=(1,32) or latent.shape!=(1,16,64,64,64):
        raise ValueError('Parent FNO9 representation cannot change')
    c=context.detach().requires_grad_(True);l=latent.detach().requires_grad_(True)
    total=0.;consumed=0
    for begin in range(0,n,32768):
        guard();end=min(begin+32768,n)
        prediction=model.query(case,begin,end,c,l)
        loss=half_stress_squared_loss(prediction,target[begin:end],scale,norm)
        loss.backward();total+=float(loss.detach());consumed+=end-begin
        del prediction,loss
    if consumed!=n or c.grad is None or l.grad is None:
        raise RuntimeError('Missing exact whole-case chain rule')
    if not torch.isfinite(c.grad).all() or not torch.isfinite(l.grad).all():
        raise FloatingPointError('Nonfinite full stress leaf gradient')
    leaves=dict(geometry_context_L2=float(c.grad.norm()),operator_latent_L2=float(l.grad.norm()),
        original_nodes_used=consumed,node_sampling=False,stress_coefficient=.5,
        U_loss_included=False)
    guard();torch.autograd.backward((context,latent),(c.grad,l.grad))
    return total,leaves
