"""Hybrid query loss must preserve whole-geometry gradients through chunked heads."""
import copy,unittest
import torch
from torch import nn
from geobc.overfit_models import NativeNodeTransolver,backward_case,predict_case,relative_loss

class GlobalContext(nn.Module):
 def __init__(self):
  super().__init__();self.linear=nn.Linear(28,16)
 def forward(self,x):
  z=self.linear(x);return z+z.mean(1,keepdim=True)

class HybridTests(unittest.TestCase):
 def test_head_chunks_keep_global_context_gradients(self):
  torch.manual_seed(7)
  model=NativeNodeTransolver.__new__(NativeNodeTransolver);nn.Module.__init__(model)
  model.operator=GlobalContext();model.head=nn.Sequential(nn.Linear(44,12),nn.GELU(),nn.Linear(12,12),nn.GELU(),nn.Linear(12,3));model.double()
  reference=copy.deepcopy(model)
  xyz=torch.randn(19,3,dtype=torch.float64);features=torch.randn(19,25,dtype=torch.float64);target=torch.randn(19,3,dtype=torch.float64)
  case=dict(xyz=xyz,features=features,inputs=torch.cat((xyz,features),1),target=target,energy=target.square().sum(1).mean()[None])
  latent=reference.operator(case['inputs'][None]);pred=reference.query(case,slice(None),latent)
  loss=relative_loss(pred,target,case['energy'],19);loss.backward()
  got=backward_case('transolver_hybrid',model,case,lambda:None,chunk_size=7)
  self.assertAlmostEqual(got,float(loss.detach()),places=12)
  for a,b in zip(model.parameters(),reference.parameters()):
   self.assertIsNotNone(a.grad);self.assertTrue(torch.isfinite(a.grad).all());self.assertTrue(torch.allclose(a.grad,b.grad,atol=1e-12,rtol=1e-10))
  self.assertTrue(torch.allclose(predict_case('transolver_hybrid',model,case,lambda:None,chunk_size=7),pred.detach(),atol=1e-12,rtol=1e-10))

if __name__=='__main__':unittest.main()
