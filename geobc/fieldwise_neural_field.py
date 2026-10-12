"""One geometry/FNO trunk with displacement- and gradient-specific output heads."""
import copy
import torch
from torch import nn
from .neural_field import FNOField
from .kinematic_neural_field import KinematicCovariateFNO
from .fno import query_grid


class FieldwiseCovariateFNO(FNOField):
    def __init__(self,seed=2026):
        nn.Module.__init__(self)
        initial=KinematicCovariateFNO('ACTUAL',seed)
        self.encoder,self.operator=initial.encoder,initial.operator
        heads={}
        for field,columns,channels in [('U',list(range(3)),slice(0,3)),('S',list(range(3,12)),slice(3,4))]:
            head=copy.deepcopy(initial.head)
            with torch.random.fork_rng(devices=[]):
                first=nn.Linear(76+len(columns),128)
                last=nn.Linear(128,3 if field=='U' else 1)
            with torch.no_grad():
                first.weight.copy_(initial.head[0].weight[:,list(range(76))+[76+i for i in columns]])
                first.bias.copy_(initial.head[0].bias)
                last.weight.copy_(initial.head[-1].weight[channels])
                last.bias.copy_(initial.head[-1].bias[channels])
            head[0],head[-1]=first,last
            heads[field]=head
        self.head=nn.ModuleDict(heads)
        self.input_mode='FIELDWISE'
        self.model_seed=int(seed)

    def query(self,case,start,stop,context,latent):
        cov=case['kinematic_inputs'][start:stop]
        if cov.dtype!=torch.float32 or cov.shape!=(stop-start,12) or not bool(torch.isfinite(cov).all()):
            raise ValueError('Expected finite FP32 physical inputs')
        values=query_grid(latent[0],case['query_indices'][start:stop],case['query_weights'][start:stop],32768)
        native=torch.cat((case['xyz'][start:stop],case['features'][start:stop],context.expand(stop-start,-1),values),1)
        u=self.head['U'](torch.cat((native,cov[:,:3]),1))
        s=self.head['S'](torch.cat((native,cov[:,3:]),1))
        return torch.cat((u,s),1)
