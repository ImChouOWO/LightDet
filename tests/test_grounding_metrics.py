import unittest
import torch
from units.model.tool.grounding_metrics import GroundingMetrics
from units.model.tool.runtime import select_predictions

A=[0.,0.,.4,.4];B=[.6,.6,1.,1.];C=[.6,0.,1.,.4]


def update(metric,boxes,gt,scores=None):
    boxes=torch.tensor(boxes).reshape(-1,4)
    metric.update(boxes,torch.tensor(scores if scores is not None else [.9]*len(boxes)),
                  dict(boxes=torch.tensor(gt).reshape(-1,4),valid_size=torch.ones(2)))


class GroundingMetricTests(unittest.TestCase):
    def test_complete_multi_target_set(self):
        metric=GroundingMetrics();update(metric,[A,B],[A,B])
        self.assertEqual(metric.compute()['grec_precision_f1'],1.)
        self.assertIsNone(metric.compute()['rec_acc50'])

    def test_missing_or_extra_prediction_fails(self):
        metric=GroundingMetrics();update(metric,[A],[A,B]);update(metric,[A,B,C],[A,B])
        self.assertEqual(metric.compute()['grec_precision_f1'],0.)

    def test_main_duplicate_fails_without_nms(self):
        metric=GroundingMetrics();update(metric,[A,A,B],[A,B],[.95,.9,.85])
        self.assertEqual(metric.compute()['grec_precision_f1'],0.)

    def test_no_target_and_false_positive(self):
        metric=GroundingMetrics();update(metric,[A],[],[.1]);update(metric,[A],[],[.9])
        self.assertEqual(metric.compute()['n_acc'],.5)
        self.assertEqual(metric.compute()['grec_precision_f1'],.5)

    def test_rec_top1_not_best_of_top10(self):
        metric=GroundingMetrics();update(metric,[B,A],[A],[.9,.8])
        self.assertEqual(metric.compute()['rec_acc50'],0.)
        metric=GroundingMetrics();update(metric,[A],[A],[.1])
        self.assertEqual(metric.compute()['rec_acc50'],1.)
        self.assertEqual(metric.compute()['grec_precision_f1'],0.)

    def test_padding_removed_and_threshold_inclusive(self):
        boxes=torch.tensor([[.8,.8,1.,1.],A]);scores=torch.tensor([.99,.7])
        selected,_=select_predictions(boxes,scores,torch.tensor([.5,.5]))
        self.assertEqual(len(selected),1)

    def test_greedy_matching_is_official_not_maximum_cardinality(self):
        from unittest.mock import patch
        metric=GroundingMetrics()
        with patch('units.model.tool.grounding_metrics.generalized_box_iou',return_value=torch.tensor([[.6,.5],[.5,.1]])):
            update(metric,[A,B],[A,B])
        self.assertEqual(metric.compute()['grec_precision_f1'],0.)


if __name__=='__main__':unittest.main()
