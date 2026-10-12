"""Differentiable trilinear weights on the normalized input grid."""
import torch


def stencil(xyz,resolution=64):
    if not bool(torch.isfinite(xyz).all()) or float(xyz.min()) < -2e-6 or float(xyz.max()) > 1+2e-6:
        raise ValueError('Query outside normalized cube')
    position=xyz.clamp(0,1)*(resolution-1)
    low=position.floor().long().clamp(max=resolution-2); frac=position-low
    corners=torch.tensor([[i&1,(i>>1)&1,(i>>2)&1] for i in range(8)],device=xyz.device)
    nodes=low[:,None,:]+corners
    indices=nodes[...,0]*resolution**2+nodes[...,1]*resolution+nodes[...,2]
    weights=torch.where(corners.bool(),frac[:,None,:],1-frac[:,None,:]).prod(-1)
    return indices,weights
