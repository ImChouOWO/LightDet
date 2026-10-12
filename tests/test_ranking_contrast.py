import unittest

import torch

from units.model.cards.relation_loss import ranking_contrast_loss


class RankingContrastTests(unittest.TestCase):
    def fixture(self):
        visual = torch.tensor([[[1., .1], [.1, 1.]]], requires_grad=True)
        text = torch.tensor([[[1., 0.], [0., 1.], [1., 0.]]], requires_grad=True)
        boxes = torch.tensor([[[0., 0., .4, .4], [.6, .6, 1., 1.]]], requires_grad=True)
        target = dict(boxes=boxes[0].detach(), phrase_char_spans=[[[0, 1]], [[1, 2]], [[2, 3]]],
                      phrase_relations=torch.tensor([[1, 0], [0, 1], [1, -1]]))
        offsets = torch.tensor([[[0, 1], [1, 2], [2, 3]]])
        return visual, text, boxes, target, offsets

    def loss(self, visual, text, boxes, target, offsets):
        return ranking_contrast_loss(visual, text, offsets, torch.ones(1, 3), [target], boxes, .1)

    def test_correct_features_beat_swapped_and_both_branches_receive_gradients(self):
        v, t, b, target, offsets = self.fixture()
        good = self.loss(v, t, b, target, offsets)
        bad = self.loss(v.flip(1), t, b, target, offsets)
        self.assertLess(good.item(), bad.item())
        good.backward()
        for tensor in (v, t):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        self.assertIsNone(b.grad)

    def test_duplicate_queries_do_not_increase_object_weight(self):
        v, t, b, target, offsets = self.fixture()
        before = self.loss(v, t, b, target, offsets)
        after = self.loss(v[:, [0, 0, 1]], t, b[:, [0, 0, 1]], target, offsets)
        torch.testing.assert_close(before, after)

    def test_all_positive_or_unknown_has_no_negative_competition(self):
        for value in (1, -1):
            v, t, b, target, offsets = self.fixture()
            target['phrase_relations'].fill_(value)
            loss = self.loss(v, t, b, target, offsets)
            loss.backward()
            self.assertEqual(loss.item(), 0)
            self.assertEqual(v.grad.abs().sum().item(), 0)

    def test_empty_gt_and_no_matched_roi_are_finite(self):
        v, t, b, target, offsets = self.fixture()
        target['boxes'] = torch.empty(0, 4)
        target['phrase_relations'] = torch.empty(3, 0)
        self.assertEqual(self.loss(v, t, b, target, offsets).item(), 0)
        v, t, b, target, offsets = self.fixture()
        self.assertEqual(self.loss(v, t, b + 10, target, offsets).item(), 0)

    def test_unknown_phrase_is_excluded_from_both_directions(self):
        v, t, b, target, offsets = self.fixture()
        target['phrase_relations'][2].fill_(-1)
        before = self.loss(v, t, b, target, offsets)
        changed = t.detach().clone()
        changed[:, 2] = torch.tensor([-100., 200.])
        torch.testing.assert_close(before, self.loss(v, changed, b, target, offsets))


if __name__ == '__main__':
    unittest.main()
