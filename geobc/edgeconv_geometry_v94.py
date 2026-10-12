"""Custom dynamic EdgeConv geometry descriptor; no data or job I/O.

Inspired by Wang et al., https://arxiv.org/abs/1801.07829. This replaces only
the batch1 global geometry encoder, not an unchanged official classifier.
It keeps the original cloud/global frame and returns the same bounded32
descriptor. Neighbor selection is discrete; edge features retain gradients.
"""
import torch
from torch import nn
from .common_grid_solver import CommonGridSolver


class EdgeGeometryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(a,b),nn.LeakyReLU(.2))
                                  for a,b in ((6,64),(128,64),(128,128))])
        self.merge=nn.Sequential(nn.Linear(256,64),nn.LeakyReLU(.2))
        self.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,32),
                                nn.LayerNorm(32,elementwise_affine=False),nn.Tanh())

    def forward(self,cloud,mask):
        assert cloud.ndim==3 and cloud.shape[-1]==3 and mask.shape==cloud.shape[:2]
        assert mask.dtype==torch.bool and mask.device==cloud.device
        outputs=[];receipts=[]
        for points,valid in zip(cloud,mask):
            x=points[valid].float();assert len(x)>0 and torch.isfinite(x).all()
            k=min(20,len(x));parts=[]
            for block in self.blocks:
                # Selection has no gradient; no padded points can enter the
                # graph because they were removed before every graph layer.
                with torch.no_grad():
                    f=x.detach();norm=f.square().sum(-1)
                    distance=(norm[:,None]+norm[None,:]-2*f@f.T).clamp_min_(0)
                    neighbors=distance.topk(k,dim=-1,largest=False,sorted=False).indices
                centers=x[:,None,:].expand(-1,k,-1)
                edges=torch.cat((centers,x[neighbors]-centers),-1)
                x=block(edges).max(1).values;parts.append(x)
            merged=self.merge(torch.cat(parts,-1))
            pooled=torch.cat((merged.max(0).values,merged.mean(0)),-1)
            outputs.append(self.head(pooled))
            receipts.append(dict(valid_points=len(points[valid]),padded_points=len(points)-len(x),
                                 neighbors_per_layer=[k]*3,padding_excluded=True))
        self.last_graph_receipt=receipts
        result=torch.stack(outputs);assert result.shape==(len(cloud),32) and torch.isfinite(result).all()
        return result


def build_solver(package,seed=2026):
    if package!='FNO9_EDGE_CONV':raise ValueError(package)
    from .fno_stress_specialist_v54 import tree_digest
    model=CommonGridSolver('FNO9',seed)
    baseline_digest=tree_digest(model.state_dict())
    shared={name:tree_digest(getattr(model,name).state_dict())for name in ('head','operator')}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model.encoder=EdgeGeometryEncoder()
    after={name:tree_digest(getattr(model,name).state_dict())for name in ('head','operator')}
    assert shared==after
    model.geometry_initialization_reference=dict(baseline_plain_model_digest=baseline_digest,
        shared_head_operator_initial_digests=shared,shared_initial_bytes_unchanged=True)
    model.package_arm=package
    return model.cuda()
