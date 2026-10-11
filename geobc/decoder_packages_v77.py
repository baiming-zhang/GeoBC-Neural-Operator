"""Three fixed decoder packages on the original common FNO9 body; no IO."""
import math
import torch
from torch import nn
from .common_grid_solver import CommonGridSolver
from .fno import query_grid
from .physical_field_loss import full_common_case_backward

ARMS=('FNO9_plain_query_MLP','FNO9_Fourier_query_MLP','FNO9_TCNN_query')
PARAMETERS={ARMS[0]:4526858,ARMS[1]:4531466,ARMS[2]:12245894}


class FourierQueryFNO(CommonGridSolver):
    def __init__(self,seed=2026):
        super().__init__('FNO9',seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.head=nn.Sequential(nn.Linear(112,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,4))

    def query(self,case,start,stop,context,latent):
        xyz=case['xyz'][start:stop]
        freq=xyz.new_tensor([1,2,4,8,16,32])
        angle=2*math.pi*xyz[:,:,None]*freq
        fourier=torch.cat((angle.sin().flatten(1),angle.cos().flatten(1)),-1)
        values=query_grid(latent[0],case['query_indices'][start:stop],case['query_weights'][start:stop],32768)
        return self.head(torch.cat((xyz,case['features'][start:stop],context.expand(stop-start,-1),values,fourier),-1))


def build_solver(arm,seed=2026):
    if arm==ARMS[0]:model=CommonGridSolver('FNO9',seed).cuda()
    elif arm==ARMS[1]:model=FourierQueryFNO(seed).cuda()
    elif arm==ARMS[2]:
        from .operator_hash_metadata_v6 import build_solver as factory
        model=factory('FNO_GLOBAL_HASH',seed=seed)
        assert model.backend=='tcnn_fully_fused'
    else:raise ValueError(arm)
    model.package_arm=arm
    return model


def full_backward(model,case,physical,scale,norms,guard=lambda:None):
    if case['scientific_group']!='TRAIN':raise PermissionError('TRAIN fields only')
    return full_common_case_backward(model,case,physical,scale,norms,guard)


def predict(model,case,guard=lambda:None):
    context,latent=model.encode_case(case)
    chunks=[]
    for start in range(0,len(case['xyz']),32768):
        guard();chunks.append(model.query(case,start,min(start+32768,len(case['xyz'])),context,latent))
    return torch.cat(chunks)


def query_qualification(model,case,guard=lambda:None):
    from .solver_selection_data import _stencil
    guard();model.eval()
    with torch.no_grad():context,latent=model.encode_case(case)
    selected=torch.linspace(0,len(case['xyz'])-1,64,device=case['xyz'].device).long()
    xyz=case['xyz'][selected].detach().clone().requires_grad_(True)
    indices,weights=_stencil(xyz,64)
    query=dict(xyz=xyz,features=case['features'][selected],query_indices=indices,query_weights=weights)
    prediction=model.query(query,0,64,context,latent)
    gradient,=torch.autograd.grad(prediction.to(torch.float32).square().sum(),xyz)
    if not torch.isfinite(prediction).all() or not torch.isfinite(gradient).all() or gradient.norm()<=0:
        raise RuntimeError('Actual continuous query coordinate chain failed')
    with torch.no_grad():
        chunks=[model.query(query,start,min(start+17,64),context,latent) for start in range(0,64,17)]
        discrepancy=float((torch.cat(chunks)-prediction).abs().max())
    tolerance=.002 if model.package_arm==ARMS[2] else .00003
    if discrepancy>tolerance:raise RuntimeError('Actual decoder chunk discrepancy beyond declared precision tolerance')
    model.train();guard()
    return dict(passed=True,actual_case_key=case['key'],actual_queries=64,coordinate_gradient_L2=float(gradient.norm()),
        chunk_max_absolute_error=discrepancy,declared_absolute_tolerance=tolerance,native25_held_constant=True,
        independent_offnode_FEM_accuracy_proven=False,precision='native TCNN FP16' if model.package_arm==ARMS[2] else 'FP32')
