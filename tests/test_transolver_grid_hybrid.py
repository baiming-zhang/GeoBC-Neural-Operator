"""Scalar-free grid interpolation and full-context gradient checks."""
import copy,unittest
import torch
from torch import nn
from geobc.overfit_models import prepare_grid,backward_case,relative_loss
from geobc.overfit_transolver_grid import GridTransolverHybrid

class ToyGrid(nn.Module):
 def __init__(self):super().__init__();self.conv=nn.Conv3d(26,16,1)
 def forward(self,x):
  y=self.conv(x);return y+y.mean((2,3,4),keepdim=True)

class GridHybridTests(unittest.TestCase):
 def test_no_label_grid_input_and_global_query_gradient(self):
  torch.manual_seed(11);xyz=torch.rand(19,3);features=torch.randn(19,25);target=torch.randn(19,3,dtype=torch.float64)
  case=dict(xyz=xyz,features=features,physical=target,target=target,energy=target.square().sum(1).mean()[None]);prepare_grid(case,resolution=4)
  original_grid=case['grid'].clone();case['physical']=target*100;prepare_grid(case,resolution=4);self.assertTrue(torch.equal(case['grid'],original_grid))
  case['xyz']=xyz.double();case['features']=features.double();case['query_weights']=case['query_weights'].double()
  m=GridTransolverHybrid.__new__(GridTransolverHybrid);nn.Module.__init__(m);m.operator=ToyGrid();m.head=nn.Sequential(nn.Linear(44,12),nn.GELU(),nn.Linear(12,3));m.double()
  reference=copy.deepcopy(m);latent=reference.operator(case['grid'].double());pred=reference.query(case,slice(None),latent);loss=relative_loss(pred,target,case['energy'],19);loss.backward()
  case['grid']=case['grid'].double();got=backward_case('fno',m,case,lambda:None,chunk_size=7);self.assertAlmostEqual(got,float(loss.detach()),places=12)
  for a,b in zip(m.parameters(),reference.parameters()):self.assertTrue(torch.isfinite(a.grad).all());self.assertTrue(torch.allclose(a.grad,b.grad,atol=1e-12,rtol=1e-10))

if __name__=='__main__':unittest.main()
