import unittest

import torch

from units.model.tool.grounding_metrics import GroundingMetrics, matched_count


A = [0., 0., 1., 1.]
B = [2., 2., 3., 3.]
C = [4., 4., 5., 5.]


def update(metric, boxes, gt, scores=None, selected=None, ignored=()):
    boxes = torch.tensor(boxes).reshape(-1, 4)
    scores = torch.tensor(scores if scores is not None else [0.9] * len(boxes))
    metric.update(pred_boxes=boxes, final_scores=scores, quality_scores=scores,
                  alignment_scores=scores, gt_boxes=torch.tensor(gt).reshape(-1, 4),
                  ignored_boxes=torch.tensor(ignored).reshape(-1, 4),
                  selected_boxes=boxes if selected is None else torch.tensor(selected).reshape(-1, 4))


class GroundingMetricTests(unittest.TestCase):
    def test_query_hit_differs_from_object_recall(self):
        metric = GroundingMetrics()
        update(metric, [A, A], [A, B])
        result = metric.compute()
        self.assertEqual(result['hit_at_1'], 1.)
        self.assertEqual(result['object_recall_at_5'], .5)
        self.assertEqual(result['oracle50'], .5)
        self.assertEqual(result['set_success50'], 0.)
        self.assertIsNone(result['no_target_accuracy'])

    def test_duplicates_fail_exact_set(self):
        metric = GroundingMetrics()
        update(metric, [A, A, B], [A, B])
        self.assertEqual(metric.compute()['object_recall_at_5'], 1.)
        self.assertEqual(metric.compute()['set_success50'], 0.)

    def test_maximum_matching_avoids_greedy_failure(self):
        self.assertEqual(matched_count(torch.tensor([[True, True], [True, False]])), 2)

    def test_empty_query_success_and_false_positive(self):
        metric = GroundingMetrics()
        update(metric, [A], [], selected=[])
        update(metric, [A], [])
        result = metric.compute()
        self.assertEqual(result['no_target_accuracy'], .5)
        self.assertEqual(result['set_success50'], .5)
        self.assertIsNone(result['hit_at_1'])
        self.assertEqual(result['no_target_query_count'], 2)

    def test_ignore_is_not_negative_or_certified_empty(self):
        metric = GroundingMetrics()
        update(metric, [B, A], [A], ignored=[B])
        update(metric, [B], [], ignored=[B])
        result = metric.compute()
        self.assertEqual(result['hit_at_1'], 1.)
        self.assertEqual(result['set_success50'], 1.)
        self.assertEqual(result['ignored_query_count'], 1)
        self.assertIsNone(result['no_target_accuracy'])

    def test_iou_boundary_and_no_detections(self):
        metric = GroundingMetrics()
        update(metric, [[0., 0., 2., 1.]], [A])
        self.assertEqual(metric.compute()['hit_at_1'], 1.)
        update(metric, [], [A])
        self.assertEqual(metric.compute()['hit_at_1'], .5)

    def test_ranking_uses_scores(self):
        metric = GroundingMetrics()
        update(metric, [A, C], [A], [.1, .9])
        result = metric.compute()
        self.assertEqual(result['hit_at_1'], 0.)
        self.assertEqual(result['hit_at_5'], 1.)
        self.assertEqual(result['oracle50'], 1.)

    def test_nonfinite_predictions_fail(self):
        with self.assertRaises(ValueError):
            update(GroundingMetrics(), [A], [A], [float('nan')])


if __name__ == '__main__':
    unittest.main()
