import unittest
import torch
from units.model.cards.phrase_matching import PhraseMatchingHead, phrase_matrices, matching_for_image


class BatchedPhraseTests(unittest.TestCase):
    def test_matches_scalar_pooling_reference_in_full_precision(self):
        torch.manual_seed(3)
        visual = torch.randn(2,3,4,requires_grad=True)
        text = torch.randn(2,4,4,requires_grad=True)
        head = PhraseMatchingHead(4)
        matching = head(visual,text,torch.tensor(2.))
        offsets = torch.tensor([[[0,1],[1,2],[3,4],[4,5]]]*2)
        mask = torch.ones(2,4,dtype=torch.bool)
        spans = [[[[0,2]],[[3,5]]],[[[1,2]]]]
        batched = phrase_matrices(matching,offsets,mask,spans)[0]
        reference = []
        for b, groups in enumerate(spans):
            pooled = []
            for group in groups:
                selected = torch.zeros(4,dtype=torch.bool)
                for start,end in group:
                    selected |= (offsets[b,:,1]>start) & (offsets[b,:,0]<end)
                weights = matching['token_weights'][b,selected].softmax(0)
                pooled.append(torch.nn.functional.normalize((text[b,selected]*weights[:,None]).sum(0),dim=-1))
            sim = torch.nn.functional.normalize(visual[b],dim=-1) @ torch.stack(pooled).T
            v = torch.nn.functional.normalize(visual[b], dim=-1)[:, None]
            t = torch.stack(pooled)[None]
            interactions = torch.cat((v*t, (v-t).abs()), dim=-1)
            sim = sim + .25 * head.pair_head(interactions).squeeze(-1).tanh()
            reference.append(sim*matching['scale']+matching['bias'])
        for a,b in zip(batched,reference):
            torch.testing.assert_close(a,b)
        parameters = (visual, text, head.token_attention.weight, *head.pair_head.parameters())
        fast = torch.autograd.grad(sum(v.sum() for v in batched), parameters, retain_graph=True)
        slow = torch.autograd.grad(sum(v.sum() for v in reference), parameters)
        for a,b in zip(fast,slow):
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)

    def test_ragged_batch_equals_individual_results_and_gradients(self):
        torch.manual_seed(7)
        visual = torch.randn(2,3,4,requires_grad=True)
        text = torch.randn(2,4,4,requires_grad=True)
        head = PhraseMatchingHead(4)
        matching = head(visual,text,torch.tensor(2.))
        offsets = torch.tensor([[[0,1],[1,2],[3,4],[4,5]]]*2)
        mask = torch.ones(2,4,dtype=torch.bool)
        spans = [[[[0,2]],[[3,5]]],[[[1,2]]]]
        batched = phrase_matrices(matching,offsets,mask,spans)[0]
        individual = [phrase_matrices(matching_for_image(matching,b),offsets[b:b+1],mask[b:b+1],[spans[b]])[0][0] for b in range(2)]
        for a,b in zip(batched,individual):
            torch.testing.assert_close(a,b)
        gradients = torch.autograd.grad(sum(v.sum() for v in batched),(visual,text),retain_graph=True)
        reference = torch.autograd.grad(sum(v.sum() for v in individual),(visual,text))
        for a,b in zip(gradients,reference):
            torch.testing.assert_close(a,b)

    def test_repeated_consumer_reuses_matrix_and_empty_rows_are_finite(self):
        head = PhraseMatchingHead(4)
        matching = head(torch.randn(2,3,4),torch.randn(2,2,4),torch.tensor(2.))
        offsets = torch.tensor([[[0,1],[1,2]]]*2)
        mask = torch.ones(2,2,dtype=torch.bool)
        spans = [[[[0,2]]],[]]
        first = phrase_matrices(matching,offsets,mask,spans)
        second = phrase_matrices(matching,offsets,mask,spans)
        self.assertIs(first,second)
        self.assertEqual(first[0][1].shape,(3,0))
        self.assertTrue(torch.isfinite(first[0][0]).all())


if __name__ == '__main__':
    unittest.main()
