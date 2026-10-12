import unittest
from unittest.mock import patch
import torch
from torch import nn
from units.tool.card import TextAlignmentTransformer, VisionTextModel


class LocalROIQueryTests(unittest.TestCase):
    def test_initial_query_uses_roi_and_bbox_once(self):
        stage = TextAlignmentTransformer(4, 4, ['c3'], 2, 2, 0, 2., 0., 0.)
        stage.norm = nn.Identity()
        roi = torch.randn(2, 3, 4, 4, requires_grad=True)
        bbox_embedding = torch.randn(2, 3, 4, requires_grad=True)
        tokens = roi + stage.roi_position_embeddings
        with patch.object(stage, '_roi_tokens', return_value=tokens) as extract:
            out, _ = stage._forward_group({}, bbox_embedding,
                torch.zeros(2, 3, 4), (32, 32), 0)
        extract.assert_called_once()
        torch.testing.assert_close(out, roi.mean(2) + bbox_embedding)
        out.sum().backward()
        torch.testing.assert_close(roi.grad, torch.full_like(roi, .25))
        torch.testing.assert_close(bbox_embedding.grad, torch.ones_like(bbox_embedding))

    def test_bbox_conditioning_detaches_coordinates(self):
        model = nn.Module()
        model.score_bbox_conditioning = True
        model.score_bbox_detach = True
        model.bbox_query_encoder = nn.Linear(4, 8)
        boxes = torch.randn(2, 3, 4, requires_grad=True)
        query = VisionTextModel._condition_score_queries(model, boxes)
        query.sum().backward()
        self.assertIsNone(boxes.grad)
        self.assertIsNotNone(model.bbox_query_encoder.weight.grad)

    def test_refinement_is_local_and_has_no_caption_attention(self):
        torch.manual_seed(8)
        stage = TextAlignmentTransformer(4, 4, ['c3'], 2, 2, 2, 2., 0., 0.)
        self.assertFalse(any('text_attention' in key or 'text_norm' in key
                             for key in stage.state_dict()))
        roi = torch.randn(1, 2, 4, 4)
        queries, boxes = torch.randn(1, 2, 4), torch.zeros(1, 2, 4)
        with patch.object(stage, '_roi_tokens', return_value=roi):
            original, _ = stage._forward_group({}, queries, boxes, (32, 32), 0)
        changed = roi.clone()
        changed[:, 1] += torch.tensor([5., -3., 2., 1.])
        with patch.object(stage, '_roi_tokens', return_value=changed):
            updated, _ = stage._forward_group({}, queries, boxes, (32, 32), 0)
        torch.testing.assert_close(original[:, 0], updated[:, 0])
        self.assertFalse(torch.allclose(original[:, 1], updated[:, 1]))
