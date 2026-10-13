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
from units.tool.card import DenseLocalizationCNN,VisionTextModel


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

    def test_checkpoint_contains_complete_resume_state(self):
        from units.model.train import save_checkpoint
        model=nn.Linear(3,1);optimizer=torch.optim.AdamW(model.parameters())
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
        scaler=torch.amp.GradScaler('cpu',enabled=False)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'latest.pt'
            save_checkpoint(path,model,optimizer,scheduler,scaler,3,.4,{}, {})
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            self.assertEqual(checkpoint['schema'],'lightdet.grefcoco.v1')
            self.assertEqual(checkpoint['epoch'],3)
            self.assertTrue({'optimizer','scheduler','scaler','rng_torch','rng_numpy','rng_python','rng_cuda'}<=checkpoint.keys())
            restored=nn.Linear(3,1);restored.load_state_dict(checkpoint['model'],strict=True)
            self.assertTrue(torch.equal(model.weight,restored.weight))

    def test_tiny_gt_receives_dense_assignment(self):
        dense=dict(points=torch.tensor([[.1,.1],[.3,.3]]),strides=torch.full((2,2),.1),levels=torch.tensor([0,0]))
        target=dict(boxes=torch.tensor([[.14,.14,.15,.15]]),valid_size=torch.ones(2))
        owner,valid=GroundingLoss().dense_targets(dense,target)
        self.assertEqual(int((owner==0).sum()),1)

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

    def test_nms_duplicate_does_not_receive_conflicting_selection_target(self):
        bbox=torch.tensor([[[.2,.2,.6,.6],[.2,.2,.6,.6]]],requires_grad=True)
        alignment=torch.tensor([[2.,2.]],requires_grad=True)
        selection=torch.tensor([[1.,1.]],requires_grad=True)
        out=dict(boxes=bbox,quality_logits=torch.zeros(1,2,requires_grad=True),
                 alignment_logits=alignment,selection_logits=selection,
                 dense=dict(points=torch.tensor([[.3,.3]]),strides=torch.tensor([[.1,.1]]),levels=torch.tensor([0]),
                            boxes=bbox[:,:1],logits=torch.zeros(1,1,requires_grad=True)))
        target=dict(boxes=torch.tensor([[.2,.2,.6,.6]]),all_boxes=torch.tensor([[.2,.2,.6,.6]]),valid_size=torch.ones(2))
        _,parts=GroundingLoss()(out,[target])
        parts['selection'].backward()
        self.assertEqual(int((selection.grad!=0).sum()),1)
        self.assertTrue((selection.grad<=0).all())

    def test_dense_predictions_depend_on_text(self):
        torch.manual_seed(1);module=DenseLocalizationCNN(32,8,2).eval()
        features=dict(c3=torch.randn(1,32,8,8));text=torch.randn(1,4,32);mask=torch.ones(1,4,dtype=torch.bool)
        first=module(features,text,mask,text.mean(1),(64,64))
        second=module(features,-text,mask,-text.mean(1),(64,64))
        self.assertFalse(torch.allclose(first['boxes'],second['boxes']))
        self.assertFalse(any('MultiheadAttention' in type(child).__name__ for child in module.modules()))

    def test_cnn_roi_model_backward_including_empty_target(self):
        torch.set_num_threads(2)
        with patch('units.tool.card.MultilingualMiniLM',FakeText):
            model=VisionTextModel(hidden_dim=32,backbone_weights=None,dense_growth=8,dense_depth=2,candidate_count=10,roi_size=4,score_heads=4,score_layers=1)
        model.train();images=torch.rand(1,3,64,64)
        tokens=dict(input_ids=torch.randint(0,20,(2,4)),attention_mask=torch.ones(2,4,dtype=torch.long))
        targets=[dict(boxes=torch.tensor([[.2,.2,.6,.6]]),all_boxes=torch.tensor([[.2,.2,.6,.6]]),valid_size=torch.ones(2)),dict(boxes=torch.empty(0,4),all_boxes=torch.tensor([[.1,.1,.4,.4]]),valid_size=torch.ones(2))]
        out=model(images,tokens,targets,True,True,image_indices=torch.tensor([0,0]))
        out['selection_logits'].sum().backward(retain_graph=True)
        self.assertIsNone(model.quality_head[-1].weight.grad)
        model.zero_grad(set_to_none=True)
        loss,parts=GroundingLoss()(out,targets);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertEqual(out['boxes'].shape,(2,10,4))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertIsNotNone(model.localization.box.weight.grad)
        self.assertIsNotNone(model.selection_head[-1].weight.grad)


if __name__=='__main__':unittest.main()
