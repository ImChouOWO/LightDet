import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from units.model.cards.phrase_matching import PhraseMatchingHead
from units.model.ranking_diagnostics import Stats, components, coverage_record, load_checkpoint, top1, transition


class RankingDiagnosticTests(unittest.TestCase):
    def test_ignore_filter_and_transitions(self):
        scores = torch.tensor([9., 2., 1.])
        correct = torch.tensor([False, True, False])
        keep = torch.tensor([False, True, True])
        self.assertEqual(top1(scores,correct,keep),(1,True))
        self.assertEqual(top1(scores,correct,torch.zeros(3,dtype=torch.bool)),(None,False))
        self.assertEqual(transition(False,True),'improved')
        self.assertEqual(transition(True,False),'harmed')
        self.assertIsNone(transition(None,True))

    def test_weighted_statistics_and_missing_denominator(self):
        stats = Stats()
        stats.add('hit',torch.tensor([True,False,True]))
        stats.add('hit',False)
        stats.add('missing',None)
        stats.add('empty',torch.empty(0))
        result = stats.result()
        self.assertEqual(result['hit']['count'],4)
        self.assertEqual(result['hit']['mean'],.5)
        self.assertNotIn('missing',result)
        self.assertIsNone(result['empty']['mean'])
        with self.assertRaises(FloatingPointError):
            stats.add('bad',torch.tensor(float('nan')))

    def test_finite_precision_saturation_can_create_ranking_ties(self):
        logits = torch.tensor([50.,100.])
        correct,keep = torch.tensor([False,True]),torch.ones(2,dtype=torch.bool)
        self.assertEqual(top1(logits,correct,keep),(1,True))
        self.assertEqual(top1(logits.sigmoid(),correct,keep),(0,False))

    def test_components_match_production_and_keep_ablation_local(self):
        torch.manual_seed(1)
        head = PhraseMatchingHead(4)
        visual,text = torch.randn(2,3,4),torch.randn(2,4,4)
        offsets = torch.tensor([[[0,1],[1,2],[3,4],[4,5]]]*2)
        valid = torch.ones(2,4,dtype=torch.bool)
        spans = [[[[0,2]],[[3,5]]],[[[0,2]]]]
        matching = head(visual,text,torch.tensor(2.))
        before = components(matching,offsets,valid,spans)
        changed = text.clone()
        changed[1,2:] = 100
        after = components(head(visual,changed,torch.tensor(2.)),offsets,valid,spans)
        torch.testing.assert_close(before[1]['full'],after[1]['full'])
        for row in before:
            torch.testing.assert_close(row['full']-row['base'],row['delta']*matching['scale'],rtol=1e-4,atol=1e-6)
            self.assertLessEqual(float(row['delta'].detach().abs().max()),.25)
        self.assertEqual(before[1]['full'].shape,(3,1))

    def fixture(self):
        objects = [dict(id='green',bbox=[0,0,10,10],attributes=dict(colors=['綠色'])),
                   dict(id='mixed',bbox=[10,0,20,10],attributes=dict(colors=['綠色','白色']))]
        phrases = []
        for i,(key,pos) in enumerate([('single_color:綠色',['green']),
                                     ('contains_color:綠色',['green','mixed']),
                                     ('single_color:綠色',['green'])]):
            phrases.append(dict(id=str(i),semantic_key=key,phrase='abc'[i],tokens_positive=[[2*i,2*i+1]],
                positive_object_ids=pos,negative_object_ids=[o['id'] for o in objects if o['id'] not in pos],ignore_object_ids=[]))
        return dict(schema_version='lightdet.relations.v1',filename='test.jpg',width=20,height=10,
                    objects=objects,grounding=dict(caption='a。b。c。',phrases=phrases))

    def test_coverage_counts_overlap_as_positive_not_false_negative(self):
        counts,types,examples = coverage_record(self.fixture())
        self.assertEqual(counts['positive_pairs'],4)
        self.assertEqual(counts['negative_pairs'],2)
        self.assertEqual(counts['same_color_object_pairs'],2)
        self.assertEqual(counts['single_vs_contains_object_contrasts'],2)
        self.assertEqual(counts['same_key_relation_conflicts'],0)
        self.assertEqual(types['contains_color'],1)
        self.assertTrue(any(e['kind']=='single_vs_contains' for e in examples))

    def test_coverage_flags_same_semantic_key_with_conflicting_relations(self):
        raw = copy.deepcopy(self.fixture())
        phrase = raw['grounding']['phrases'][2]
        phrase['positive_object_ids']=['mixed']
        phrase['negative_object_ids']=['green']
        self.assertEqual(coverage_record(raw)[0]['same_key_relation_conflicts'],1)

    def test_checkpoint_refuses_silent_random_initialization(self):
        model = nn.Linear(2,1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'weights.pt'
            torch.save(dict(model={'weight':model.weight.detach()}),path)
            with self.assertRaisesRegex(ValueError,'missing'):
                load_checkpoint(path,model,False)
            torch.save(dict(model=model.state_dict(),epoch=7),path)
            info = load_checkpoint(path,model,False)
        self.assertEqual(info['source_epoch'],7)
        self.assertEqual(len(info['sha256']),64)


if __name__ == '__main__':
    unittest.main()
