import math
import unittest

import torch
import torch.nn.functional as F

from units.model.cards.background import BackgroundPolicy
from units.model.cards.relation_loss import ranking_preferences, relation_loss, select_ranking_negatives
from units.model.pipeline.relations import rebuild_targets, sample_record


class LogRankingTests(unittest.TestCase):
    def record(self):
        colors = [['綠色'], ['綠色', '白色'], ['藍色']]
        objects = [dict(id=str(i), bbox=[i, 0, i+1, 1], attributes=dict(colors=c))
                   for i, c in enumerate(colors)]
        phrases = []
        for key, positive in [('single_color:綠色', ['0']), ('contains_color:綠色', ['0', '1']),
                              ('contains_color:藍色', ['2'])]:
            phrases.append(dict(id=key, phrase=key, semantic_key=key, tokens_positive=[[0, 1]],
                positive_object_ids=positive, negative_object_ids=[o['id'] for o in objects if o['id'] not in positive],
                ignore_object_ids=[]))
        return rebuild_targets(dict(objects=objects, regions=phrases, metadata={}))

    def test_color_preferences_do_not_change_relations(self):
        record = self.record()
        self.assertEqual(record['phrase_relations'], [[1, 0, 0], [1, 1, 0], [0, 0, 1]])
        self.assertEqual(record['ranking_object_preferences'][0], [True, True, False])
        labels = torch.tensor(record['phrase_relations']).T
        target = {key: torch.tensor(record[key]) for key in
                  ('ranking_object_preferences', 'ranking_phrase_preferences')}
        phrase, obj = ranking_preferences(target, labels)
        self.assertTrue(phrase[1, 0])
        self.assertFalse(phrase[1, 2])
        self.assertTrue(obj[0, 1])
        self.assertEqual((obj & (labels.T == 0)).sum(), 1)

    def test_caption_shuffle_keeps_preferences_aligned(self):
        sampled = sample_record(self.record(), 49, 1, 3, True)
        for p, region in enumerate(sampled['regions']):
            if region['semantic_key'] == 'single_color:綠色':
                self.assertEqual(sampled['phrase_relations'][p], [1, 0, 0])
                self.assertEqual(sampled['ranking_object_preferences'][p], [True, True, False])
        self.assertEqual(len(sampled['ranking_phrase_preferences']), len(sampled['regions']))

    def test_quota_retains_semantic_negatives_and_fills_remaining_with_high_scores(self):
        scores = torch.tensor([[-1., -2., -3., -4., -5., -6., -7., -8., 9.]])
        negative = torch.tensor([[True]*8+[False]])
        preferred = torch.tensor([[False]*5+[True]*4])
        _, indices = select_ranking_negatives(scores, negative, preferred, 5, 3)
        self.assertEqual(set(indices[0].tolist()), {0, 1, 5, 6, 7})

    def test_missing_preferred_candidates_use_available_negative_budget(self):
        scores = torch.tensor([[-1., -2., -3., -4., -5., -6.]])
        negative = torch.ones_like(scores, dtype=torch.bool)
        preferred = torch.zeros_like(negative)
        preferred[0, 5] = True
        _, indices = select_ranking_negatives(scores, negative, preferred, 5, 3)
        self.assertEqual(set(indices[0].tolist()), {0, 1, 2, 3, 5})
        values, _ = select_ranking_negatives(scores, torch.zeros_like(negative), preferred)
        self.assertFalse(torch.isfinite(values).any())

    def ranking(self, probabilities, quality=None):
        logits = torch.logit(torch.tensor(probabilities)).reshape(-1, 1).requires_grad_()
        boxes = torch.tensor([[[0., 0., .3, .3], [.6, .6, .9, .9]]], requires_grad=True)
        target = dict(boxes=boxes[0].detach(), phrase_relations=torch.tensor([[1, 0]]))
        quality = torch.zeros(1, 2, 1, requires_grad=True) if quality is None else quality
        _, rank, _ = relation_loss([logits], [target],
            [(torch.tensor([0, 1]), torch.tensor([0, 1]))], 2., math.log(2), .25,
            pred_boxes=boxes, quality_logits=quality, background_policy=BackgroundPolicy(enabled=False))
        return rank, logits, quality, boxes

    def test_log_margin_uses_fused_score_ratio_and_detaches_quality_and_boxes(self):
        rank, logits, quality, boxes = self.ranking([.5, .4])
        self.assertAlmostEqual(rank.item(), math.log(2)-.5*math.log(.5/.4), places=6)
        rank.backward()
        self.assertLess(logits.grad[0], 0)
        self.assertGreater(logits.grad[1], 0)
        self.assertIsNone(quality.grad)
        self.assertIsNone(boxes.grad)
        rank, _, _, _ = self.ranking([.8, .2])
        self.assertAlmostEqual(rank.item(), 0, places=6)

    def test_equal_probability_ratios_have_equal_order_loss(self):
        first, _, _, _ = self.ranking([.8, .4])
        second, _, _, _ = self.ranking([.4, .2])
        torch.testing.assert_close(first, second)

    def test_extreme_logits_are_finite_and_log_fusion_preserves_order(self):
        alignment = torch.tensor([[-1000.], [1000.]], requires_grad=True)
        quality = torch.tensor([[1000.], [-1000.]])
        fused = .5 * (F.logsigmoid(alignment) + F.logsigmoid(quality))
        self.assertTrue(torch.isfinite(fused).all())
        fused.sum().backward()
        self.assertTrue(torch.isfinite(alignment.grad).all())
        a, q = torch.tensor([-.2, 1., -3., 2.]), torch.tensor([1., -1., .2, .5])
        raw = (a.sigmoid()*q.sigmoid()).sqrt()
        log = .5*(F.logsigmoid(a)+F.logsigmoid(q))
        self.assertEqual(raw.argsort().tolist(), log.argsort().tolist())

    def test_unknown_is_never_mined_and_multiple_positives_do_not_compete(self):
        scores = torch.tensor([[0., -.1, -.2, -.3]], requires_grad=True)
        labels = torch.tensor([[1, 1, -1, 0]])
        top, _ = select_ranking_negatives(scores, labels == 0, torch.ones_like(labels, dtype=torch.bool))
        top = top[torch.isfinite(top)]
        loss = F.relu(math.log(2)-scores[labels == 1, None]+top[None]).mean()
        loss.backward()
        self.assertEqual(scores.grad[0, 2], 0)
        self.assertTrue((scores.grad[0, :2] < 0).all())
        self.assertGreater(scores.grad[0, 3], 0)

    def test_invalid_budget_is_rejected(self):
        scores = torch.zeros(1, 2)
        mask = torch.ones_like(scores, dtype=torch.bool)
        with self.assertRaises(ValueError):
            select_ranking_negatives(scores, mask, mask, 2, 3)


if __name__ == '__main__':
    unittest.main()
