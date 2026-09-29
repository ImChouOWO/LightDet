import copy
import unittest

import torch

from units.model.pipeline.relations import normalize_record, sample_record
from units.model.cards.relation_loss import relation_loss, phrase_predictions
from units.model.cards.loss import AssignmentResult, HDETRRepeatedHungarianMatcher, GroundingLoss


def record():
    return dict(schema_version="lightdet.relations.v1", filename="test.jpg", width=100, height=100,
        objects=[dict(id="a", bbox=[1, 1, 20, 20]), dict(id="b", bbox=[40, 40, 60, 60])],
        grounding=dict(caption="紅船。含紅。藍船。", phrases=[
            dict(id="p0", phrase="紅船", semantic_key="single_color:紅", tokens_positive=[[0, 2]],
                 positive_object_ids=["a"], negative_object_ids=["b"], ignore_object_ids=[]),
            dict(id="p1", phrase="含紅", semantic_key="contains_color:紅", tokens_positive=[[3, 5]],
                 positive_object_ids=["a"], negative_object_ids=[], ignore_object_ids=["b"]),
            dict(id="p2", phrase="藍船", semantic_key="single_color:藍", tokens_positive=[[6, 8]],
                 positive_object_ids=["b"], negative_object_ids=["a"], ignore_object_ids=[])]),
        metadata=dict(phrase_variants={"紅船": ["紅色船"]}))


class RelationTrainingTests(unittest.TestCase):
    def test_sampling_preserves_identity_and_spans(self):
        source = normalize_record(record())
        views = [sample_record(source, 49, e, 0, True) for e in range(4)]
        self.assertGreater(len({r["caption"] for r in views}), 1)
        for r in views:
            self.assertEqual(len(r["unique_targets"]), 2)
            for p, row in zip(r["regions"], r["phrase_relations"]):
                a, b = p["tokens_positive"][0]
                self.assertEqual(r["caption"][a:b], p["phrase"])
                self.assertEqual(row[0], 0 if p["id"] == "p2" else 1)
        self.assertEqual(source["caption"], record()["grounding"]["caption"])
        self.assertIs(sample_record(source, 49, 99, 0, False), source)

    def test_masked_loss_and_bidirectional_ranking(self):
        target = {"phrase_relations": torch.tensor([[1, 0], [1, -1], [0, 1]])}
        assignment = AssignmentResult.from_per_batch([(torch.tensor([0, 1]), torch.tensor([0, 1]))],
                                                     device=torch.device("cpu"), mode="test")
        logits = torch.tensor([[-3., -3., 3.], [3., 50., -3.]], requires_grad=True)
        loss, rank, _ = relation_loss([logits], [target], assignment, 2., .15, .25)
        self.assertGreater(rank.item(), 0)
        (loss + rank).backward()
        self.assertEqual(logits.grad[1, 1].item(), 0)
        self.assertLess(logits.grad[0, 0].item(), 0)
        self.assertGreater(logits.grad[1, 0].item(), 0)
        correct = torch.tensor([[4., 4., -4.], [-4., 0., 4.]])
        _, good_rank, _ = relation_loss([correct], [target], assignment, 2., .15, .25)
        self.assertEqual(good_rank.item(), 0)

    def test_phrase_pooling_and_truncation(self):
        logits = torch.tensor([[[0., 2.]]], requires_grad=True)
        offsets = torch.tensor([[[0, 1], [1, 2]]])
        target = {"phrase_char_spans": [[[0, 2]]]}
        p = phrase_predictions(logits, offsets, torch.ones(1, 2, dtype=torch.bool), [target])[0]
        torch.testing.assert_close(p.sigmoid(), logits[0].sigmoid().mean(-1, keepdim=True))
        with self.assertRaises(ValueError):
            phrase_predictions(logits, offsets, torch.ones(1, 2, dtype=torch.bool),
                               [{"phrase_char_spans": [[[0, 3]]]}])

    def test_aux_cost_order_and_geometry(self):
        boxes = torch.tensor([[[0.,0.,1.,1.], [.01,0.,1.,1.], [.02,0.,1.,1.], [2.,2.,3.,3.]]])
        target = [{"boxes": boxes[0, :1]}]
        matcher = HDETRRepeatedHungarianMatcher(cost_bbox=1, cost_giou=1, cost_score=0,
                  cost_alignment=10, max_positive_per_gt=2, min_extra_positive_iou=.3)
        result = matcher(boxes, torch.zeros(1, 4, 1), target,
                         phrase_costs=[torch.tensor([[0.],[2.],[.1],[-100.]])])
        # The primary may follow the semantic cost; extras must pass geometry.
        q, _ = next(iter(result))
        self.assertEqual(len(q), 2)
        self.assertEqual(q.unique().numel(), len(q))
        result = matcher(boxes, torch.zeros(1, 4, 1), target,
                         phrase_costs=[torch.tensor([[0.],[2.],[.1],[100.]])])
        q, _ = next(iter(result))
        self.assertEqual(set(q.tolist()), {0, 2})


if __name__ == "__main__":
    unittest.main()
