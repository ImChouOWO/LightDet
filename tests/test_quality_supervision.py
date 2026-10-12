import unittest

import torch

from units.model.cards.loss import AssignmentResult, GroundingLoss, PackedTargets


class QualitySupervisionTests(unittest.TestCase):
    def quality_gradients(self, complete):
        boxes = torch.tensor([[[0., 0., .3, .3], [0., 0., .3, .3], [.6, .6, .9, .9]]])
        targets = [dict(boxes=boxes[0, :1])]
        assignment = AssignmentResult.from_per_batch(
            [(torch.tensor([0]), torch.tensor([0]))], device=torch.device('cpu'), mode='test')
        quality = torch.zeros(1, 3, 1, requires_grad=True)
        loss, _ = GroundingLoss()._branch_loss(
            pred_bbox=boxes, pred_score_logit=quality, assignments=assignment,
            packed_targets=PackedTargets.from_targets(targets, device=torch.device('cpu'), dtype=torch.float32),
            lambda_bbox=0., lambda_giou=0., lambda_score=1., quality_alpha=1.,
            query_loss_weights=None, text_negative_mask=torch.tensor([False]),
            duplicate_alpha=0., hard_negative_alpha=0., negative_iou_ignore_thr=.5,
            enable_extra_negative_losses=False, objects_complete=[complete])
        loss.backward()
        return quality.grad.flatten()

    def test_duplicate_boxes_share_iou_target(self):
        grad = self.quality_gradients(False)
        self.assertLess(grad[0], 0.)
        torch.testing.assert_close(grad[0], grad[1])
        self.assertGreater(grad[2], 0.)

    def test_complete_annotation_supervises_background(self):
        grad = self.quality_gradients(True)
        self.assertGreater(grad[2], 0.)


if __name__ == '__main__':
    unittest.main()
