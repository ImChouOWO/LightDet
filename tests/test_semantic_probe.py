import gzip
import json
from pathlib import Path
import tempfile
import unittest
import torch
from units.model.semantic_probe import color_counterfactual, masked_roi, normalize_map, sample_records


class SemanticProbeTests(unittest.TestCase):
    def test_mask_preserves_position_and_other_candidates(self):
        position = torch.randn(1,1,4,3)
        features = torch.randn(1,2,4,3)
        tokens = features+position
        original=tokens.clone()
        changed=masked_roi(tokens,position,1,torch.tensor([0,2]))
        torch.testing.assert_close(changed[0,1,[0,2]],position[0,0,[0,2]])
        torch.testing.assert_close(changed[0,0],original[0,0])
        torch.testing.assert_close(changed[0,1,[1,3]],original[0,1,[1,3]])
        torch.testing.assert_close(tokens,original)

    def test_color_change_preserves_spans_and_other_phrases(self):
        caption='紅色的船。綠色的船。'
        result=color_counterfactual(caption,[[0,4]])
        self.assertEqual(result[0],'藍色的船。綠色的船。')
        self.assertEqual(len(result[0]),len(caption))
        self.assertIsNone(color_counterfactual('紅色綠色的船',[[0,6]]))
        self.assertIsNone(color_counterfactual('最大的船',[[0,4]]))

    def test_sampling_is_seeded_and_filters_unusable_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'records.gz'
            with gzip.open(path,'wt') as stream:
                for i in range(30):
                    stream.write(json.dumps(dict(filename=str(i),category='wrong_object' if i%2 else 'correct',candidate_available=True))+'\n')
            a,count=sample_records(path,4,49)
            b,_=sample_records(path,4,49)
        self.assertEqual(count,15)
        self.assertEqual(a,b)
        self.assertEqual(len(a),4)
        self.assertTrue(all(int(r['filename'])%2 for r in a))

    def test_empty_attribution_does_not_create_a_false_hotspot(self):
        self.assertEqual(normalize_map(torch.zeros(2,2)).sum(),0)
        with self.assertRaises(FloatingPointError):
            normalize_map(torch.tensor([[float('nan')]]))
