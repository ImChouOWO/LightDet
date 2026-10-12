import unittest
from dataclasses import replace

import torch

from units.model.cards.background import BackgroundPolicy, background_candidates, mine_background
from units.model.cards.loss import GroundingLoss, AssignmentResult, PackedTargets
from units.model.cards.relation_loss import relation_loss


class BackgroundSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.boxes = torch.tensor([[[0., 0., .2, .2], [.6, .6, .8, .8],
                                    [.3, .6, .5, .8], [.02, .02, .04, .04]]])
        self.target = dict(boxes=self.boxes[0, :1], phrase_relations=torch.tensor([[1]]),
                           objects_complete=True)
        self.assignment = [(torch.tensor([0]), torch.tensor([0]))]
        self.policy = BackgroundPolicy(top_k=1, empty_top_k=1)

    def run_relation(self, complete=True, scale=1., positive=True):
        target = dict(self.target, objects_complete=complete)
        if not positive:
            target['phrase_relations'] = torch.tensor([[0]])
        logits = torch.tensor([[-2.], [3.], [1.], [4.]], requires_grad=True)
        quality = torch.zeros(1, 4, 1, requires_grad=True)
        loss, rank, metrics = relation_loss([logits], [target], self.assignment, 2., .15, .25,
            pred_boxes=self.boxes, quality_logits=quality, background_policy=self.policy,
            background_scale=scale)
        return loss, rank, metrics, logits, quality

    def test_background_ranking_updates_both_candidate_alignment_scores(self):
        _, rank, metrics, logits, quality = self.run_relation()
        rank.backward()
        self.assertLess(logits.grad[0, 0], 0)
        self.assertGreater(logits.grad[1, 0], 0)
        self.assertEqual(logits.grad[2, 0], 0)
        self.assertEqual(logits.grad[3, 0], 0)
        self.assertIsNone(quality.grad)
        self.assertEqual(metrics['background_rank_pair_count'], 1)

    def test_background_runs_without_metadata_flag(self):
        loss, rank, metrics, logits, _ = self.run_relation(complete=False)
        (loss + rank).backward()
        self.assertGreater(metrics['background_candidate_count'], 0)

    def test_warmup_disables_then_ramps_background(self):
        self.assertEqual(self.policy.scale(5), 0)
        self.assertAlmostEqual(self.policy.scale(6), .2)
        self.assertEqual(self.policy.scale(10), 1)
        _, rank, metrics, _, _ = self.run_relation(scale=0)
        self.assertEqual(rank, 0)
        self.assertEqual(metrics['background_query_count'], 0)
        _, partial, _, _, _ = self.run_relation(scale=.2)
        _, full, _, _, _ = self.run_relation(scale=1)
        torch.testing.assert_close(partial, full * .2)

    def test_no_positive_phrase_has_bce_but_no_ranking(self):
        loss, rank, _, logits, _ = self.run_relation(positive=False)
        self.assertEqual(rank, 0)
        loss.backward()
        self.assertGreater(logits.grad[1, 0], 0)

    def test_geometry_protects_fragments_enclosing_and_uncertain_boxes(self):
        gt = torch.tensor([[.4, .4, .6, .6]])
        boxes = torch.tensor([[.45,.45,.46,.46], [0.,0.,1.,1.],
                              [.5,.4,.7,.6], [0.,0.,.1,.1]])
        mask = background_candidates(boxes, gt, True, self.policy)
        self.assertEqual(mask.tolist(), [False,False,False,True])

    def test_hard_mining_respects_per_phrase_budgets(self):
        policy = replace(self.policy, top_k=10, negative_ratio=2., empty_top_k=1)
        scores = torch.stack((torch.arange(12.), torch.arange(12.).flip(0)), 1)
        mask = mine_background(torch.ones(12,dtype=torch.bool), scores, [2,0], policy)
        self.assertEqual(mask.sum(0).tolist(), [4,1])
        self.assertTrue(mask[11,0] and mask[0,1])

    def test_quality_mines_background_without_relabeling_foreground(self):
        criterion = GroundingLoss()
        criterion.background_policy = self.policy
        quality = torch.tensor([[[0.],[3.],[1.],[4.]]], requires_grad=True)
        assignments = AssignmentResult.from_per_batch(self.assignment,device=torch.device('cpu'),mode='test')
        loss, metrics = criterion._branch_loss(
            pred_bbox=self.boxes, pred_score_logit=quality, assignments=assignments,
            packed_targets=PackedTargets.from_targets([self.target],device=torch.device('cpu'),dtype=torch.float32),
            lambda_bbox=0.,lambda_giou=0.,lambda_score=1.,quality_alpha=1.,query_loss_weights=None,
            text_negative_mask=torch.tensor([False]),duplicate_alpha=0.,hard_negative_alpha=0.,
            negative_iou_ignore_thr=.5,enable_extra_negative_losses=False,objects_complete=[True])
        loss.backward()
        self.assertLess(quality.grad[0,0,0],0)
        self.assertGreater(quality.grad[0,1,0],0)
        self.assertEqual(quality.grad[0,2,0],0)
        self.assertEqual(metrics['quality_background_query_count'],1)

    def test_empty_complete_image(self):
        target = dict(boxes=torch.empty(0,4),phrase_relations=torch.empty(1,0),objects_complete=True)
        empty = torch.empty(0,dtype=torch.long)
        logits = torch.zeros(4,1,requires_grad=True)
        loss,rank,metrics = relation_loss([logits],[target],[(empty,empty)],2.,.15,.25,
            pred_boxes=self.boxes,background_policy=self.policy)
        loss.backward()
        self.assertEqual(rank,0)
        self.assertEqual(int((logits.grad > 0).sum()),1)
        self.assertEqual(metrics['background_rank_pair_count'],0)


if __name__ == '__main__':
    unittest.main()
