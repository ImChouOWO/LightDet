import unittest
import torch
from units.model.cards.gt_roi import gt_roi_loss, gt_roi_scale


class GTRoiTests(unittest.TestCase):
    def test_schedule_holds_then_decays_and_stops(self):
        config = dict(weight=.5, hold_epochs=5, end_epoch=15)
        values = [gt_roi_scale(e, config) for e in range(1, 21)]
        self.assertEqual(values[:5], [.5]*5)
        self.assertEqual(values[9], .25)
        self.assertEqual(values[14:], [0.]*6)
        self.assertTrue(all(a >= b for a,b in zip(values,values[1:])))

    def test_expired_branch_never_accesses_model_or_gt(self):
        box = torch.zeros(1, 2, 4, requires_grad=True)
        loss, metrics = gt_roi_loss(None, {'bbox':box}, None, None, 15,
                                   dict(weight=.5, hold_epochs=5, end_epoch=15))
        loss.backward()
        self.assertEqual(metrics['gt_roi_scale'], 0)
        self.assertEqual(box.grad.abs().sum().item(), 0)

    def test_invalid_schedule_fails(self):
        with self.assertRaises(ValueError):
            gt_roi_scale(1, dict(weight=.5, hold_epochs=5, end_epoch=5))


if __name__ == '__main__':
    unittest.main()
