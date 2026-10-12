"""Whole-original-node relative loss for original physical U and signed S."""
import torch


def decode(encoded, scale):
    value=encoded.double()*encoded.new_tensor(scale['std'],dtype=torch.float64)+encoded.new_tensor(scale['mean'],dtype=torch.float64)
    stress=value[:,3:4]
    if scale['stress_scale'] is not None:
        stress=stress.sinh()*scale['stress_scale']
    return torch.cat((value[:,:3],stress),1)


def backward_case(model, case, physical, scale, guard):
    count=len(case['xyz'])
    norm=physical.double().square().sum(0)
    unorm=norm[:3].sum(); snorm=norm[3]
    if not bool(unorm>0 and snorm>0):
        raise ValueError('Undefined zero target norm')
    context,latent=model.encode_case(case)
    c=context.detach().requires_grad_(True); l=latent.detach().requires_grad_(True)
    loss_sum=0.
    for start in range(0,count,32768):
        guard(); stop=min(start+32768,count)
        pred=model.query(case,start,stop,c,l)
        residual=decode(pred,scale)-physical[start:stop].double()
        loss=.5*(residual[:,:3].square().sum()/unorm+residual[:,3].square().sum()/snorm)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite physical loss')
        loss.backward(); loss_sum+=float(loss.detach())
    torch.autograd.backward((context,latent),(c.grad,l.grad))
    return loss_sum,dict(context=float(c.grad.norm()),latent=float(l.grad.norm()))
