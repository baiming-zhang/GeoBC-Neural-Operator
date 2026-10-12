import unittest
import torch
from geobc.kinematic_neural_field import KinematicCovariateFNO
from geobc.physical_loss import decode


class InformationControl(unittest.TestCase):
    def test_capacity_and_initialization(self):
        zero=KinematicCovariateFNO('ZERO',2026)
        actual=KinematicCovariateFNO('ACTUAL',2026)
        self.assertEqual(sum(p.numel() for p in zero.parameters()),4528394)
        self.assertTrue(all(torch.equal(value,actual.state_dict()[key]) for key,value in zero.state_dict().items()))

    def test_signed_decode_gradient(self):
        encoded=torch.tensor([[.2,.1,-.3,-.5]],requires_grad=True)
        scale=dict(mean=[0,0,0,0],std=[1,1,1,1],stress_scale=90.)
        value=decode(encoded,scale)
        self.assertLess(float(value[0,3].detach()),0)
        value.square().sum().backward()
        self.assertTrue(torch.isfinite(encoded.grad).all())

    def test_component_masks(self):
        case=dict(xyz=torch.ones(2,3),features=torch.ones(2,25),kinematic_inputs=torch.ones(2,12,requires_grad=True),
                  query_indices=torch.zeros(2,8,dtype=torch.int64),query_weights=torch.ones(2,8)/8)
        context=torch.ones(1,32);latent=torch.ones(1,16,2,2,2)
        for mode,inactive in [('U_ONLY',slice(3,12)),('GRAD_ONLY',slice(0,3))]:
            model=KinematicCovariateFNO(mode,2026)
            prediction=model.query(case,0,2,context,latent)
            grad,=torch.autograd.grad(prediction.sum(),case['kinematic_inputs'])
            self.assertTrue((grad[:,inactive]==0).all())
            self.assertGreater(float(grad.norm()),0)


if __name__=='__main__': unittest.main()
