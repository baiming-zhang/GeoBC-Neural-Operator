"""Matched R64 information and native25 query heads for FNO/official ++.

No model owns a scaler, target, identity or off-node feature extrapolation.
The PP arm processes all 262144 voxels, including unoccupied voxels.
"""
import hashlib
import torch
from torch import nn
from .conditioned_field import PointNetGeometryEncoder
from .fno import FNO3d, query_grid
from .transformers import build_transformer


class CommonGridSolver(nn.Module):
    def __init__(self, arm, seed=2026):
        super().__init__()
        if arm not in ('FNO9', 'PP256'): raise ValueError(arm)
        self.arm = arm
        # Create shared modules first; architecture-specific RNG draws cannot
        # change the shared initial weights. Each independent arm has same seed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = PointNetGeometryEncoder(32, 64)
            self.head = nn.Sequential(nn.Linear(76,128), nn.GELU(), nn.Linear(128,128), nn.GELU(), nn.Linear(128,4))
            if arm == 'FNO9':
                self.operator = FNO3d(61,16,width=9,modes=12,layers=4,head_width=128,checkpoint_blocks=True)
            else:
                self.operator = build_transformer('transolver_pp',61,16,device='cpu',checkpoint_blocks=True)

    def encode_case(self, case):
        context = self.encoder(case['cloud'],case['mask'])
        grid = torch.cat((case['grid'],context[:,:,None,None,None].expand(1,32,64,64,64)),dim=1)
        if self.arm == 'FNO9':
            latent = self.operator(grid)
        else:
            # The official adapter interprets the first three channels as xyz.
            # Reorder without discarding occupancy or changing information.
            grid = torch.cat((grid[:,1:4],grid[:,0:1],grid[:,4:]),dim=1)
            tokens = grid.flatten(2).transpose(1,2).contiguous()
            assert tokens.shape == (1,262144,61)
            latent = self.operator(tokens).transpose(1,2).reshape(1,16,64,64,64)
        return context, latent

    def query(self, case, start, stop, context, latent):
        values = query_grid(latent[0],case['query_indices'][start:stop],case['query_weights'][start:stop],32768)
        return self.head(torch.cat((case['xyz'][start:stop],case['features'][start:stop],
                                   context.expand(stop-start,-1),values),dim=1))

    def counts(self):
        groups = {name:sum(p.numel() for p in module.parameters()) for name,module in
                  [('encoder',self.encoder),('head',self.head),('operator',self.operator)]}
        return {**groups,'total':sum(groups.values())}

    def shared_identity(self):
        h = hashlib.sha256()
        for prefix, module in [('encoder',self.encoder),('head',self.head)]:
            for name, value in module.state_dict().items():
                h.update((prefix+'.'+name).encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()


def full_case_backward(model, case, guard=lambda:None):
    """Exact full original-node MSE chain rule with bounded decoder memory."""
    context, latent = model.encode_case(case)
    assert context.shape == (1,32) and latent.shape == (1,16,64,64,64)
    c = context.detach().requires_grad_(True); l = latent.detach().requires_grad_(True)
    count = len(case['xyz']); total = 0.
    assert case['target'].shape == (count,4)
    for start in range(0,count,32768):
        guard(); stop = min(start+32768,count)
        pred = model.query(case,start,stop,c,l)
        loss = (pred-case['target'][start:stop]).square().mean()*((stop-start)/count)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
        loss.backward(); total += float(loss.detach())
    assert c.grad is not None and l.grad is not None
    torch.autograd.backward((context,latent),(c.grad,l.grad))
    return total, {'context':float(c.grad.norm()),'latent':float(l.grad.norm())}


def parameter_gradients(model):
    result = {}
    for name,module in [('encoder',model.encoder),('head',model.head),('operator',model.operator)]:
        missing = []; finite = True; energy = 0.
        for key,p in module.named_parameters():
            if p.grad is None: missing.append(key); continue
            finite &= bool(torch.isfinite(p.grad).all())
            energy += float(p.grad.detach().double().square().sum())
        result[name] = {'missing':missing,'finite':finite,'L2':energy**.5}
    return result
