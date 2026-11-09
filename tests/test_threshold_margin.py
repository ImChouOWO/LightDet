import math
import unittest
import torch
from units.model.cards.loss import GroundingLoss,threshold_margin_loss


class ThresholdMarginTests(unittest.TestCase):
    def test_schedule_is_off_then_monotonic(self):
        criterion=GroundingLoss()
        self.assertEqual(criterion.margin_schedule(5),(0.,0.))
        values=[criterion.margin_schedule(e) for e in range(6,16)]
        self.assertTrue(all(a[0]<b[0] for a,b in zip(values,values[1:])))
        self.assertEqual(values[-1],(.5,.1))

    def test_gradient_and_ignored_mask(self):
        logits=torch.tensor([[0.,0.,0.,2.,-2.]],requires_grad=True)
        targets=torch.tensor([[1,0,0,1,0]])
        valid=torch.tensor([[1,1,0,1,1]],dtype=torch.bool)
        threshold_margin_loss(logits,targets,valid,threshold=.5).backward()
        self.assertLess(logits.grad[0,0],0);self.assertGreater(logits.grad[0,1],0)
        self.assertTrue(torch.equal(logits.grad[0,2:],torch.zeros(3)))

    def test_boundary_matches_point_seven(self):
        boundary=math.log(.7/.3)
        logits=torch.tensor([[boundary]],requires_grad=True)
        loss=threshold_margin_loss(logits,torch.ones_like(logits),torch.ones_like(logits,dtype=torch.bool))
        self.assertAlmostEqual(loss.item(),.5,places=5)
        zero=threshold_margin_loss(logits,torch.ones_like(logits),torch.zeros_like(logits,dtype=torch.bool))
        zero.backward();self.assertEqual(zero.item(),0.);self.assertEqual(logits.grad.item(),0.)


if __name__=='__main__':unittest.main()
