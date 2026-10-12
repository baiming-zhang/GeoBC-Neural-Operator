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


if __name__=='__main__': unittest.main()
