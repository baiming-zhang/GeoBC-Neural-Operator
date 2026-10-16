"""Grid TS ablation: same R64 occupancy/BC grid and trilinear query as FNO."""
import torch
from torch import nn
from geobc.fno import query_grid
from geobc.overfit_transformers import build_transformer

class GridTSOperator(nn.Module):
 def __init__(self,n_hidden=400):
  super().__init__()
  self.backbone=build_transformer('transolver',input_dim=29,output_dim=16,device='cuda',checkpoint_blocks=True,model_overrides={'n_hidden':n_hidden})
  axis=torch.linspace(0,1,64,device='cuda')
  self.register_buffer('grid_xyz',torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(1,-1,3),persistent=False)
 def forward(self,grid):
  if grid.shape[1:]!=(26,64,64,64):raise ValueError('Same26-channel R64 occupancy/condition grid as FNO required')
  x=torch.cat((self.grid_xyz.expand(len(grid),-1,-1),grid.flatten(2).transpose(1,2)),dim=-1)
  latent=self.backbone(x)
  return latent.transpose(1,2).reshape(len(grid),16,64,64,64)

class GridTransolverHybrid(nn.Module):
 def __init__(self,n_hidden=400,head_width=376,out_channels=3):
  super().__init__();self.operator=GridTSOperator(n_hidden)
  self.head=nn.Sequential(nn.Linear(44,head_width),nn.GELU(),nn.Linear(head_width,head_width),nn.GELU(),nn.Linear(head_width,out_channels))
  self.source_config={'backbone':self.operator.backbone.source_config,'latent_channels':16,'grid_resolution':64,'grid_channels':26,'query_head':[44,head_width,head_width,out_channels],
   'grid_input':'same occupancy+25 cell-mean conditions as FNO, plus deterministic regular-grid XYZ for TS positional input',
   'query':'same trilinear latent16+original xyz+25 conditions as FNO','label_input':False,'all_grid_cells':262144,'independent_context_chunks':False}
 def query(self,case,rows,latent):
  local=query_grid(latent[0],case['query_indices'][rows],case['query_weights'][rows])
  return self.head(torch.cat((case['xyz'][rows],case['features'][rows],local),-1))

def make_grid_model(method,seed,settings=None,out_channels=3):
 if method!='transolver_grid_hybrid':raise ValueError('Only grid TS hybrid belongs to this stage')
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 return GridTransolverHybrid(out_channels=out_channels,**(settings or {})).cuda()
