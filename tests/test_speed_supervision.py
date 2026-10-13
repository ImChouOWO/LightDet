import unittest
import torch
from torchvision.ops import generalized_box_iou
from units.model.cards.loss import overlaps,GroundingLoss,assign_dense_targets
from units.tool.card import VisionTextModel
from test_grefcoco import FakeText
from unittest.mock import patch
from collections import OrderedDict


class SpeedSupervisionTests(unittest.TestCase):
    def test_paired_giou_matches_matrix_diagonal_and_gradients(self):
        torch.manual_seed(4)
        start=torch.rand(30,2);size=torch.rand(30,2)*.4+.01
        boxes=torch.cat([start,start+size],-1).requires_grad_()
        target=boxes.detach()+.05
        actual=overlaps(boxes,target,paired=True,giou=True)
        expected=generalized_box_iou(boxes,target).diagonal()
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
        first=torch.autograd.grad(actual.sum(),boxes,retain_graph=True)[0]
        second=torch.autograd.grad(expected.sum(),boxes)[0]
        self.assertTrue(torch.allclose(first,second,atol=1e-5))

    def test_batched_loss_handles_no_target_and_multi_target(self):
        boxes=torch.tensor([[[.1,.1,.4,.4],[.6,.6,.9,.9]],[[.1,.1,.4,.4],[.6,.6,.9,.9]]],requires_grad=True)
        geom=dict(points=torch.tensor([[.2,.2],[.7,.7]]),strides=torch.full((2,2),.1),levels=torch.tensor([0,0]))
        targets=[dict(boxes=boxes[0].detach(),all_boxes=boxes[0].detach(),valid_size=torch.ones(2)),dict(boxes=torch.empty(0,4),all_boxes=boxes[1].detach(),valid_size=torch.ones(2))]
        for target in targets:target['dense_owner'],target['dense_valid']=assign_dense_targets(geom,target)
        outputs=dict(boxes=boxes,quality_logits=torch.zeros(2,2,requires_grad=True),alignment_logits=torch.zeros(2,2,requires_grad=True),selection_logits=torch.zeros(2,2,requires_grad=True),dense=dict(**geom,boxes=boxes,logits=torch.zeros(2,2,requires_grad=True)))
        loss,parts=GroundingLoss()(outputs,targets);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(boxes.grad).all())
        self.assertAlmostEqual(float(parts['bbox'].detach()),0.)

    def test_combined_roi_scoring_preserves_teacher_and_main_outputs(self):
        torch.set_num_threads(2);torch.manual_seed(8)
        with patch('units.tool.card.MultilingualMiniLM',FakeText):
            model=VisionTextModel(hidden_dim=32,backbone_weights=None,dense_growth=8,dense_depth=2,candidate_count=10,roi_size=4,score_heads=4,score_layers=1)
        model.eval();features=OrderedDict(c3=torch.randn(2,32,8,8),c4=torch.randn(2,32,4,4),c5=torch.randn(2,32,2,2))
        text=torch.randn(2,32);main=torch.tensor([[[.1,.1,.4,.4]],[[.2,.2,.6,.6]]]);teacher=torch.tensor([[[.4,.4,.8,.8]],[[.5,.5,.9,.9]]])
        combined=model.score_rois(features,torch.cat([main,teacher],1),text,(64,64))
        independent=model.score_rois(features,main,text,(64,64))
        clean=model.score_rois(features,teacher,text,(64,64))
        for value,reference in zip(combined,independent):self.assertTrue(torch.allclose(value[:,:1],reference,atol=1e-6))
        self.assertTrue(torch.allclose(combined[0][:,1:],clean[0],atol=1e-6))


if __name__=='__main__':unittest.main()
