import unittest
import torch
from geobc.overfit_models import relative_loss
from tools.train_sample_overfitting import errors


class OverfittingTests(unittest.TestCase):
    def test_chunked_full_node_gradient(self):
        torch.manual_seed(1)
        target=torch.randn(19,4,dtype=torch.float64)
        energy=torch.stack((target[:,:3].square().sum(1).mean(),target[:,3].square().mean()))
        p=torch.randn(19,4,dtype=torch.float64,requires_grad=True)
        whole=relative_loss(p,target,energy,len(p));whole.backward();expected=p.grad.clone();p.grad=None
        split=sum(relative_loss(p[a:b],target[a:b],energy,len(p)) for a,b in [(0,7),(7,19)])
        split.backward()
        self.assertTrue(torch.allclose(whole,split,atol=1e-14,rtol=1e-14))
        self.assertTrue(torch.allclose(p.grad,expected,atol=1e-14,rtol=1e-14))

    def test_signed_vector_metric(self):
        truth=torch.tensor([[1.,-2.,3.,-4.],[-2.,4.,-6.,8.]])
        result=errors(truth,truth*.995)
        self.assertAlmostEqual(result['U_percent'],.5,places=4)
        self.assertAlmostEqual(result['S_percent'],.5,places=4)
        flipped=truth.clone();flipped[:,3]*=-1
        self.assertAlmostEqual(errors(truth,flipped)['S_percent'],200)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):errors(torch.ones(3,4),torch.full((3,4),float('nan')))


if __name__=='__main__':unittest.main()
