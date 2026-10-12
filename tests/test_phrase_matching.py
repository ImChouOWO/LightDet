import unittest
import torch
from unittest.mock import patch
from units.tool.card import QueryHead
from units.model.cards.phrase_matching import PhraseMatchingHead, phrase_matrices, matching_for_image
from units.model.cards.relation_loss import phrase_predictions, ranking_contrast_loss
from units.model.tool.runtime import score_queries_for_char_spans


class PhraseMatchingTests(unittest.TestCase):
    def test_head_passes_features_without_query_token_score_matrix(self):
        head = QueryHead(hidden_dim=4)
        queries = torch.randn(2, 3, 4, requires_grad=True)
        text = torch.randn(2, 5, 4, requires_grad=True)
        mask = torch.ones(2, 5, dtype=torch.bool)
        with patch('torch.einsum', side_effect=AssertionError('obsolete token scoring')):
            matching = head.predict_phrase_matching(queries, text, mask)
        self.assertIsInstance(matching, dict)
        self.assertEqual(matching['visual'].shape, (2, 3, 4))
        self.assertEqual(matching['text'].shape, (2, 5, 4))
        offsets = torch.tensor([[[0,1],[1,2],[2,3],[3,4],[4,5]]] * 2)
        logits, _ = phrase_matrices(matching, offsets, mask, [[[[0,2]]], [[[3,5]]]])
        sum(value.square().mean() for value in logits).backward()
        self.assertTrue(torch.isfinite(queries.grad).all())
        self.assertTrue(torch.isfinite(text.grad).all())

    def fixture(self):
        head = PhraseMatchingHead(3)
        visual = torch.tensor([[[1., .1, .2], [.1, 1., .2]]], requires_grad=True)
        text = torch.tensor([[[1.,0.,.2],[.3,.1,.1],[0.,1.,.2]]], requires_grad=True)
        matching = head(visual,text,torch.tensor(2.,requires_grad=True))
        offsets = torch.tensor([[[0,1],[1,2],[3,4]]])
        mask = torch.ones(1,3,dtype=torch.bool)
        target = dict(phrase_char_spans=[[[0,2]],[[3,4]]],
                      phrase_relations=torch.tensor([[1,0],[0,1]]),
                      boxes=torch.tensor([[0.,0.,.4,.4],[.6,.6,1.,1.]]))
        tokens = matching
        return head,visual,text,matching,offsets,mask,target,tokens

    def test_training_and_inference_use_identical_phrase_logits(self):
        _,_,_,matching,offsets,mask,target,tokens = self.fixture()
        predicted = phrase_predictions(tokens,offsets,mask,[target])[0]
        single = matching_for_image(matching,0)
        for p, spans in enumerate(target['phrase_char_spans']):
            result = score_queries_for_char_spans(quality_logit=torch.zeros(2,1),
                phrase_matching=single,token_offsets=offsets[0],char_spans=spans,valid_token_mask=mask[0])
            torch.testing.assert_close(predicted[:,p],result['phrase_alignment_logit'])

    def test_shared_similarity_contrast_and_attention_receive_gradients(self):
        head,visual,text,matching,offsets,mask,target,_ = self.fixture()
        logits,sim = phrase_matrices(matching,offsets,mask,[target['phrase_char_spans']])
        torch.testing.assert_close(logits[0],sim[0]*matching['scale']+matching['bias'])
        contrast = ranking_contrast_loss(visual,text,offsets,mask,[target],target['boxes'][None],.1,matching=matching)
        (contrast + logits[0].square().mean()).backward()
        for tensor in (visual,text,head.token_attention.weight,head.bias,*head.pair_head.parameters()):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(),0)

    def test_phrase_pooling_excludes_outside_tokens_and_rejects_truncation(self):
        _,_,_,matching,offsets,mask,target,_ = self.fixture()
        first = phrase_matrices(matching,offsets,mask,[[target['phrase_char_spans'][0]]])[0][0]
        changed = dict(matching,text=matching['text'].clone())
        changed['text'][:,2] = 100
        second = phrase_matrices(changed,offsets,mask,[[target['phrase_char_spans'][0]]])[0][0]
        torch.testing.assert_close(first,second)
        with self.assertRaises(ValueError):
            phrase_matrices(matching,offsets,mask,[[[[0,8]]]])

    def test_pair_head_changes_the_score_used_by_every_consumer(self):
        head,_,_,matching,offsets,mask,target,_ = self.fixture()
        before = phrase_predictions(matching, offsets, mask, [target])[0].detach()
        with torch.no_grad():
            head.pair_head[-1].bias.add_(1.)
        matching.pop('_matrix_cache')
        logits, affinity = phrase_matrices(matching, offsets, mask, [target['phrase_char_spans']])
        self.assertTrue((logits[0] > before).all())
        torch.testing.assert_close(logits[0], affinity[0]*matching['scale']+matching['bias'])
        for p, spans in enumerate(target['phrase_char_spans']):
            result = score_queries_for_char_spans(quality_logit=torch.zeros(2,1),
                phrase_matching=matching,token_offsets=offsets[0],char_spans=spans,valid_token_mask=mask[0])
            torch.testing.assert_close(logits[0][:,p], result['phrase_alignment_logit'])


if __name__ == '__main__':
    unittest.main()
