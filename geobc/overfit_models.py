"""Small-sample U/S models; no labels or sample IDs enter their inputs."""
import torch
from torch import nn
from .fno import FNO3d, query_grid
from .overfit_coordinate_fields import build_model, ModelConfig, HashGridConfig
from .overfit_tcnn import TCNNField
from .overfit_transformers import build_transformer


def prepare_grid(case, resolution=64):
    xyz=case['xyz']; pos=xyz*(resolution-1)
    nearest=pos.round().long()
    flat=(nearest[:,0]*resolution+nearest[:,1])*resolution+nearest[:,2]
    count=torch.zeros(resolution**3,device=xyz.device)
    count.index_add_(0,flat,torch.ones(len(xyz),device=xyz.device))
    grid=torch.zeros(resolution**3,25,device=xyz.device)
    grid.index_add_(0,flat,case['features'])
    grid/=count.clamp_min(1)[:,None]
    case['grid']=torch.cat(((count>0).float()[:,None],grid),1).T.reshape(1,26,resolution,resolution,resolution)
    low=pos.floor().long().clamp(max=resolution-2);fraction=pos-low
    corners=torch.tensor([[i,j,k] for i in (0,1) for j in (0,1) for k in (0,1)],device=xyz.device)
    coords=low[:,None]+corners[None]
    case['query_indices']=(coords[:,:,0]*resolution+coords[:,:,1])*resolution+coords[:,:,2]
    case['query_weights']=torch.where(corners[None].bool(),fraction[:,None],1-fraction[:,None]).prod(-1)


class NativeNodeFNO(nn.Module):
    """R64 FNO latent plus native XYZ/BC query head; train loss is at FEM nodes."""
    def __init__(self):
        super().__init__()
        self.operator=FNO3d(26,16,width=9,modes=12,layers=4,checkpoint_blocks=True)
        self.head=nn.Sequential(nn.Linear(44,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,4))

    def query(self,case,rows,latent):
        local=query_grid(latent[0],case['query_indices'][rows],case['query_weights'][rows])
        return self.head(torch.cat((case['xyz'][rows],case['features'][rows],local),-1))


def make_model(method,seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    if method=='fno':return NativeNodeFNO().cuda()
    if method in ('transolver','transolverpp','ts3'):
        return build_transformer(method,device='cuda',checkpoint_blocks=True)
    if method=='tcnn_hash':return TCNNField(4,'both',seed=seed)
    if method=='mlp':
        config=ModelConfig(encoder='mlp',input_mode='both',hidden_dim=256,hidden_layers=4)
    else:
        grid=HashGridConfig(n_levels=4,log2_hashmap_size=14,per_level_scale=2) if method=='ngp4' else HashGridConfig()
        config=ModelConfig(input_mode='both',hidden_dim=32 if method=='ngp4' else 64,hidden_layers=2,hash_grid=grid)
    model=build_model('torch',config,'cuda')
    model.mlp[-1]=nn.Linear(model.mlp[-1].in_features,4).cuda()
    return model


def relative_loss(pred,target,energy,node_count):
    """Half U-vector relative squared L2 plus half signed-S relative squared L2."""
    diff=(pred-target).square()
    return .5*(diff[:,:3].sum()/energy[0]+diff[:,3].sum()/energy[1])/node_count


def backward_case(method,model,case,guard,chunk_size=32768):
    n=len(case['xyz']);loss_value=0.
    if method in ('transolver','transolverpp','ts3'):
        pred=model(case['inputs'][None])[0]
        loss=relative_loss(pred,case['target'],case['energy'],n)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite contextual loss')
        loss.backward();return float(loss.detach())
    latent=model.operator(case['grid']) if method=='fno' else None
    detached=latent.detach().requires_grad_(True) if latent is not None else None
    for start in range(0,n,chunk_size):
        guard();rows=slice(start,min(start+chunk_size,n))
        pred=model.query(case,rows,detached) if method=='fno' else model(case['xyz'][rows],case['features'][rows])
        loss=relative_loss(pred,case['target'][rows],case['energy'],n)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite node loss')
        loss.backward();loss_value+=float(loss.detach())
    if latent is not None:torch.autograd.backward(latent,detached.grad)
    return loss_value


@torch.no_grad()
def predict_case(method,model,case,guard,chunk_size=32768):
    if method in ('transolver','transolverpp','ts3'):
        return model(case['inputs'][None])[0].float()
    latent=model.operator(case['grid']) if method=='fno' else None
    out=[]
    for start in range(0,len(case['xyz']),chunk_size):
        guard();rows=slice(start,min(start+chunk_size,len(case['xyz'])))
        out.append(model.query(case,rows,latent) if method=='fno' else model(case['xyz'][rows],case['features'][rows]).float())
    return torch.cat(out)
