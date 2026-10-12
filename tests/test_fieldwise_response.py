import unittest
from unittest.mock import Mock
import torch
from geobc.neural_response import LocatedFieldwiseNeuralResponse


class FieldwiseRouting(unittest.TestCase):
    def test_channel_selection_and_coordinate_derivatives(self):
        api=LocatedFieldwiseNeuralResponse.__new__(LocatedFieldwiseNeuralResponse)
        api.scale=dict(mean=[0,0,0,0],std=[1,1,1,1],stress_scale=2.)
        api.context={'ver':(None,None)};api.s_context={'ver':(None,None)}
        api.prepare_query=Mock(side_effect=lambda xyz,rows,load: {'xyz':xyz})
        api.model=Mock();api.s_model=Mock()
        api.model.query.side_effect=lambda query,*args: torch.cat((query['xyz']*3,query['xyz'][:,:1]*99),1)
        api.s_model.query.side_effect=lambda query,*args: torch.cat((query['xyz']*99,query['xyz'][:,:1]*.5),1)
        xyz=torch.tensor([[.2,.3,.4]],requires_grad=True)
        out=api.query_in_cells(xyz,torch.tensor([0]),'ver')
        self.assertTrue(torch.equal(out[:,:3],(xyz*3).double()))
        self.assertTrue(torch.allclose(out[:,3],2*torch.sinh((xyz[:,0]*.5).double())))
        du,=torch.autograd.grad(out[:,:3].sum(),xyz,retain_graph=True)
        ds,=torch.autograd.grad(out[:,3].sum(),xyz)
        self.assertTrue(torch.equal(du,torch.full_like(xyz,3.)))
        self.assertAlmostEqual(float(ds[0,0]),float(torch.cosh(xyz[0,0].detach()*.5)),places=6)
        self.assertTrue(torch.equal(ds[0,1:],torch.zeros(2)))
        api.prepare_query.assert_called_once()


if __name__=='__main__':unittest.main()
