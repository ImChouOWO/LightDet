import unittest

import torch

from units.model.cards.relation_loss import relation_cost, relation_loss


class RelationSupervisionTests(unittest.TestCase):
    def run_loss(self, complete=False):
        logits = torch.zeros(4, 3, requires_grad=True)
        # Queries 0/1 identify GT A, query 2 identifies B, query 3 is background.
        boxes = torch.tensor([[[0., 0., 1., 1.], [0., 0., 1., 1.],
                               [2., 2., 3., 3.], [4., 4., 5., 5.]]])
        target = dict(boxes=boxes[0, [0, 2]],
                      phrase_relations=torch.tensor([[1, 0], [1, 0], [-1, 1]]),
                      objects_complete=complete)
        loss, rank, metrics = relation_loss([logits], [target],
            [(torch.tensor([0, 2]), torch.tensor([0, 1]))], 2., .15, .25,
            pred_boxes=boxes)
        loss.backward()
        return logits.grad, metrics

    def test_extra_query_keeps_multiple_positives_and_ignore(self):
        grad, metrics = self.run_loss()
        self.assertTrue((grad[1, :2] < 0).all())
        self.assertEqual(grad[1, 2], 0.)
        self.assertTrue((grad[2, :2] > 0).all())
        self.assertTrue((grad[3] > 0).all())
        self.assertEqual(metrics['extra_query_count'], 1.)
        self.assertEqual(metrics['background_query_count'], 1.)

    def test_background_requires_explicit_complete_annotation(self):
        grad, metrics = self.run_loss(True)
        self.assertTrue((grad[3] > 0).all())
        self.assertEqual(metrics['background_query_count'], 1.)

    def test_unknown_overlap_is_not_assigned(self):
        logits = torch.zeros(3, 1, requires_grad=True)
        boxes = torch.tensor([[[0., 0., 1., 1.]] * 3])
        target = dict(boxes=boxes[0, :2], phrase_relations=torch.tensor([[1, 0]]))
        loss, _, metrics = relation_loss([logits], [target],
            [(torch.tensor([0, 1]), torch.tensor([0, 1]))], 2., .15, .25,
            pred_boxes=boxes)
        loss.backward()
        self.assertEqual(logits.grad[2], 0.)
        self.assertEqual(metrics['extra_query_count'], 0.)

    def test_fused_ranking_updates_alignment_only(self):
        logits = torch.tensor([[-2.], [2.]], requires_grad=True)
        quality = torch.zeros(1, 2, 1, requires_grad=True)
        boxes = torch.tensor([[[0., 0., .3, .3], [.6, .6, .9, .9]]])
        target = dict(boxes=boxes[0], phrase_relations=torch.tensor([[1, 0]]))
        _, rank, _ = relation_loss([logits], [target],
            [(torch.tensor([0, 1]), torch.tensor([0, 1]))], 2., .15, .25,
            pred_boxes=boxes, quality_logits=quality)
        rank.backward()
        self.assertLess(logits.grad[0, 0], 0.)
        self.assertGreater(logits.grad[1, 0], 0.)
        self.assertIsNone(quality.grad)

    def test_empty_complete_image_has_background_gradients(self):
        logits = torch.zeros(2, 1, requires_grad=True)
        target = dict(boxes=torch.empty(0, 4), phrase_relations=torch.empty(1, 0), objects_complete=True)
        empty = torch.empty(0, dtype=torch.long)
        loss, _, _ = relation_loss([logits], [target], [(empty, empty)], 2., .15, .25,
            pred_boxes=torch.tensor([[[0., 0., 1., 1.]] * 2]))
        loss.backward()
        self.assertTrue((logits.grad > 0).all())

    def test_matcher_respects_negative_weight(self):
        logits = torch.zeros(1, 2)
        target = dict(phrase_relations=torch.tensor([[1], [0]]))
        self.assertTrue(torch.allclose(relation_cost(logits, target, 0.), torch.tensor([[.69314718]])))
        self.assertTrue((relation_cost(logits, target, 1.) > relation_cost(logits, target, .25)).all())


if __name__ == '__main__':
    unittest.main()
