import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import torch
from torch import nn

from units.model.pipeline.data import GRefCOCODataset,GRefBatchSampler,grounding_collate_fn
from units.model.cards.loss import GroundingLoss
from units.tool.card import VisionTextModel


class FakeText(nn.Module):
    def __init__(self,path,hidden_dim,freeze):
        super().__init__();self.embedding=nn.Embedding(20,hidden_dim)
    def forward(self,tokens):
        x=self.embedding(tokens['input_ids']);mask=tokens['attention_mask'].bool()
        return x,mask,x.mean(1)


class GRefDataTests(unittest.TestCase):
    def test_native_ids_multi_empty_and_sentence_cycling(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'labels').mkdir();(root/'images').mkdir()
            Image.new('RGB',(100,50)).save(root/'images/one.jpg')
            instances=dict(images=[dict(id=1,file_name='one.jpg')],annotations=[dict(id=4,image_id=1,bbox=[10,5,20,10])])
            sentences=[dict(sent='red object',sent_id=8),dict(sent='the object',sent_id=9)]
            refs=[dict(ref_id=1,image_id=1,ann_id=[4],no_target=False,split='train',sentences=sentences),dict(ref_id=2,image_id=1,ann_id=[-1],no_target=True,split='train',sentences=sentences)]
            (root/'labels/instances.json').write_text(json.dumps(instances));(root/'labels/grefs(unc).json').write_text(json.dumps(refs))
            dataset=GRefCOCODataset(root,'train',64,True)
            self.assertEqual(dataset[0]['object_ids'],[4]);self.assertEqual(len(dataset[1]['boxes']),0)
            self.assertTrue(torch.allclose(dataset[0]['boxes'][0],torch.tensor([.1,.05,.3,.15])))
            sampler=GRefBatchSampler(dataset,1)
            self.assertEqual(sorted(i for batch in sampler for i in batch),[0,1])
            eval_data=GRefCOCODataset(root,'train',64,False)
            self.assertEqual(len(eval_data),4)
            batch=grounding_collate_fn([dataset[0],dataset[1]])
            self.assertEqual(len(batch['images']),1)
            self.assertEqual(batch['image_indices'].tolist(),[0,0])
            for target in batch['targets']:
                self.assertEqual(target['boxes'].untyped_storage().data_ptr(),batch['packed_targets']['boxes'].untyped_storage().data_ptr())

    def test_checkpoint_contains_complete_resume_state(self):
        from units.model.train import save_checkpoint
        model=nn.Linear(3,1);optimizer=torch.optim.AdamW(model.parameters())
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
        scaler=torch.amp.GradScaler('cpu',enabled=False)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'latest.pt'
            save_checkpoint(path,model,optimizer,scheduler,scaler,3,.4,{}, {})
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            self.assertEqual(checkpoint['schema'],'lightdet.grefcoco.queries.v3')
            self.assertEqual(checkpoint['epoch'],3)
            self.assertTrue({'optimizer','scheduler','scaler','rng_torch','rng_numpy','rng_python','rng_cuda'}<=checkpoint.keys())
            restored=nn.Linear(3,1);restored.load_state_dict(checkpoint['model'],strict=True)
            self.assertTrue(torch.equal(model.weight,restored.weight))

    def test_sampler_visits_unequal_groups_without_dropping(self):
        class Dataset:
            groups=[0]*10+[1]*60+[2]*30
            image_ids=[i//4 for i in range(100)]
            seed=49
            epoch=1
            def __len__(self):return len(self.groups)
        dataset=Dataset();batches=list(GRefBatchSampler(dataset,10))
        flattened=[i for batch in batches for i in batch]
        self.assertEqual(sorted(flattened),list(range(100)))
        self.assertEqual(sum(dataset.groups[i]==0 for i in flattened),10)



if __name__=='__main__':unittest.main()
