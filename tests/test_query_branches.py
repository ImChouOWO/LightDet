import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torchvision.ops import generalized_box_iou
from units.tool.card import VisionTextModel,ObjectQueryTransformer
from units.model.cards.loss import GroundingLoss,allocate_aux,overlaps
from units.model.tool.runtime import select_predictions,select_branch_predictions
from units.model.tool.grounding_metrics import GroundingMetrics
from units.model.val_official import ExpressionAnnotations,print_results
from units.model.tool.official_grec import RefExpEvaluator
from test_grefcoco import FakeText


class QueryBranchTests(unittest.TestCase):
    def fixture(self,empty=False):
        boxes=torch.tensor([[[.2,.2,.6,.6],[.2,.2,.6,.6]]],requires_grad=True)
        outputs=dict(boxes=boxes,quality_logits=torch.zeros(1,2,requires_grad=True),
            alignment_logits=torch.zeros(1,2,requires_grad=True),selection_logits=torch.zeros(1,2,requires_grad=True))
        target=dict(boxes=torch.empty(0,4) if empty else boxes[0,:1].detach(),all_boxes=boxes[0,:1].detach(),valid_size=torch.ones(2))
        return outputs,[target]

    def test_text_guidance_uses_zero_content_and_three_way_addition(self):
        torch.manual_seed(4)
        module=ObjectQueryTransformer(32,4,1,10,100,[4,2,1]).eval()
        text=torch.randn(2,5,32);mask=torch.tensor([[1,1,1,0,0],[1,1,1,1,1]]).bool()
        pooled=torch.randn(2,32);positions=module.main_queries.weight
        expanded=positions[None].expand(2,-1,-1)
        with patch.object(module.text_attention,'forward',wraps=module.text_attention.forward) as attention:
            guided=module.guide_queries(positions,text,mask,pooled)
        args,kwargs=attention.call_args
        self.assertTrue(torch.equal(args[0],expanded))
        self.assertIs(args[1],text);self.assertIs(args[2],text)
        self.assertTrue(torch.equal(kwargs['key_padding_mask'],~mask))
        context=module.text_attention(expanded,text,text,key_padding_mask=~mask,need_weights=False)[0]
        self.assertTrue(torch.allclose(guided,context+expanded+module.text_condition(pooled)[:,None]))
        changed=text.clone();changed[0,3:]=1000
        self.assertTrue(torch.allclose(guided,module.guide_queries(positions,changed,mask,pooled),atol=1e-5))

    def test_main_ten_aux_hundred_queries_and_text_attention_gradients(self):
        torch.set_num_threads(2)
        module=ObjectQueryTransformer(32,4,1,10,100,[4,2,1]).train()
        features={name:torch.randn(2,32,size,size) for name,size in [('c3',8),('c4',4),('c5',2)]}
        text=torch.randn(2,5,32,requires_grad=True);mask=torch.ones(2,5,dtype=torch.bool)
        main,aux=module(features,text,mask,text.mean(1),torch.ones(2,2),True)
        self.assertEqual(main.shape,(2,10,32));self.assertEqual(aux.shape,(2,100,32))
        (main.square().mean()+aux.square().mean()).backward()
        for parameter in (module.text_attention.in_proj_weight,module.main_queries.weight,
                          module.aux_queries.weight,module.text_condition.weight,text):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(),0)
        module.eval()
        main,aux=module(features,text,mask,text.mean(1),torch.ones(2,2),False)
        self.assertEqual(main.shape,(2,10,32));self.assertIsNone(aux)

    def test_rectangular_matching_keeps_all_gt_when_queries_are_fewer(self):
        truth=torch.tensor([[.02*i,.1,.02*i+.2,.4] for i in range(12)])
        boxes=truth[:10][None].clone().requires_grad_()
        outputs=dict(boxes=boxes,quality_logits=torch.zeros(1,10,requires_grad=True),
            alignment_logits=torch.zeros(1,10,requires_grad=True),selection_logits=torch.zeros(1,10,requires_grad=True))
        targets=[dict(boxes=truth,all_boxes=truth,valid_size=torch.ones(2))]
        loss,parts=GroundingLoss()(outputs,targets,15)
        self.assertTrue(torch.isfinite(loss));parts['main_selection'].backward()
        self.assertTrue((outputs['selection_logits'].grad<0).all())
        self.assertEqual(len(targets[0]['boxes']),12)
        self.assertEqual(truth.shape,(12,4))

    def test_unmatched_duplicate_is_no_object(self):
        outputs,targets=self.fixture();_,parts=GroundingLoss()(outputs,targets)
        parts['main_selection'].backward()
        gradient=outputs['selection_logits'].grad.flatten()
        self.assertEqual(int((gradient<0).sum()),1)
        self.assertEqual(int((gradient>0).sum()),1)

    def test_empty_target_pushes_all_queries_down(self):
        outputs,targets=self.fixture(empty=True);loss,parts=GroundingLoss()(outputs,targets,epoch=20)
        self.assertTrue(torch.isfinite(loss));parts['main_selection'].backward()
        self.assertTrue((outputs['selection_logits'].grad>0).all())

    def test_aux_unique_owner_capacity_and_geometry(self):
        cost=np.array([[.1,.2],[.3,.4],[.5,.6],[.7,.8],[.9,1.]])
        iou=np.array([[.9,.8],[.8,.9],[.7,.6],[.6,.7],[.1,.1]])
        owner=allocate_aux(cost,iou,2,.3)
        self.assertEqual((owner==0).sum(),2);self.assertEqual((owner==1).sum(),2)
        self.assertEqual(owner[4],-1)
        self.assertTrue((allocate_aux(cost,np.zeros_like(iou),3,.3)==-1).all())

    def test_aux_loss_schedule_and_weighted_total(self):
        outputs,targets=self.fixture();outputs['aux']=dict(outputs)
        criterion=GroundingLoss()
        self.assertEqual(criterion.aux_factor(5),0.)
        self.assertAlmostEqual(criterion.aux_factor(15),1.)
        self.assertAlmostEqual(criterion.aux_factor(30),0.)
        main,_=criterion.branch_loss(outputs,targets,15,False)
        aux,_=criterion.branch_loss(outputs['aux'],targets,15,True)
        total,_=criterion(outputs,targets,15)
        self.assertTrue(torch.allclose(total,.8*main+.2*aux))

    def model(self,aux):
        with patch('units.tool.card.MultilingualMiniLM',FakeText):
            return VisionTextModel(hidden_dim=32,backbone_weights=None,num_object_queries=5,num_aux_queries=7,
                query_heads=4,query_layers=1,memory_grids=[4,2,1],roi_size=4,score_heads=4,score_layers=1,aux_enabled=aux)

    def inputs(self):
        images=torch.rand(1,3,64,64)
        tokens=dict(input_ids=torch.randint(0,20,(2,4)),attention_mask=torch.ones(2,4,dtype=torch.long))
        targets=[dict(boxes=torch.tensor([[.2,.2,.6,.6]]),all_boxes=torch.tensor([[.2,.2,.6,.6]]),valid_size=torch.ones(2)),
                 dict(boxes=torch.empty(0,4),all_boxes=torch.tensor([[.1,.1,.4,.4]]),valid_size=torch.ones(2))]
        return images,tokens,targets

    def test_transformer_backward_aux_off_and_on(self):
        torch.set_num_threads(2)
        for enabled in (False,True):
            model=self.model(enabled).train();images,tokens,targets=self.inputs()
            out=model(images,tokens,targets,gt_roi=True,image_indices=torch.tensor([0,0]))
            self.assertEqual('aux' in out,enabled);self.assertEqual(out['boxes'].shape,(2,5,4))
            self.assertTrue((out['boxes'][...,2:]>out['boxes'][...,:2]).all())
            out['selection_logits'].sum().backward(retain_graph=True)
            self.assertIsNone(model.main_head.quality[-1].weight.grad)
            model.zero_grad(set_to_none=True)
            loss,_=GroundingLoss()(out,targets,epoch=15);loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertIsNotNone(model.localization.main_queries.weight.grad)
            self.assertIsNotNone(model.main_head.box[-1].weight.grad)
            if enabled:self.assertIsNotNone(model.aux_head.selection[-1].weight.grad)
            else:self.assertTrue(all(p.grad is None for p in model.aux_head.parameters()))
            model.eval();out=model(images,tokens,targets,image_indices=torch.tensor([0,0]))
            self.assertNotIn('aux',out)

    def test_aux_group_cannot_change_main_forward(self):
        torch.set_num_threads(2);model=self.model(True).train();model.localization.eval()
        images,tokens,targets=self.inputs()
        with torch.no_grad():
            first=model(images,tokens,targets,image_indices=torch.tensor([0,0]))
            model.aux_enabled=False
            second=model(images,tokens,targets,image_indices=torch.tensor([0,0]))
        self.assertTrue(torch.allclose(first['scores'],second['scores'],atol=1e-5))
        self.assertTrue(torch.allclose(first['boxes'],second['boxes'],atol=1e-5))

    def test_main_keeps_duplicates_aux_nms_removes_them(self):
        boxes=torch.tensor([[0.,0.,1.,1.],[0.,0.,1.,1.]])
        scores=torch.tensor([.9,.8]);prediction=dict(boxes=boxes,scores=scores)
        outputs=dict(**prediction,aux=prediction)
        self.assertEqual(len(select_branch_predictions(outputs,torch.ones(2))[0]),2)
        self.assertEqual(len(select_branch_predictions(outputs,torch.ones(2),branch='aux')[0]),1)

    def test_streaming_metrics_match_official_evaluator(self):
        annotations=ExpressionAnnotations();predictions={};metric=GroundingMetrics()
        samples=[([[0,0,1,1]],[.9],[[0,0,1,1]]),([[0,0,1,1],[0,0,1,1]],[.9,.8],[[0,0,1,1]]),([[0,0,1,1]],[.1],[]),([[0,0,1,1]],[.9],[])]
        for i,(boxes,scores,gt) in enumerate(samples):
            box=torch.tensor(boxes).float();score=torch.tensor(scores).float();truth=torch.tensor(gt).reshape(-1,4).float()
            metric.update(box,score,dict(boxes=truth,valid_size=torch.ones(2)))
            annotations.add(i,[dict(bbox=[0,0,1,1],category_id=1)] if gt else [dict(bbox=[0,0,1,1],category_id=-1)])
            predictions[i]=dict(boxes=box,scores=score)
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator=RefExpEvaluator(annotations,('bbox',));evaluator.update(predictions);official=evaluator.summarize()
        actual=metric.compute()
        self.assertEqual(actual['grec_precision_f1'],official['F1_score'])
        self.assertEqual(actual['n_acc'],official['N_acc']);self.assertEqual(actual['t_acc'],official['T_acc'])

    def test_paired_giou_matches_matrix_diagonal(self):
        start=torch.rand(12,2);boxes=torch.cat([start,start+.3],-1).requires_grad_();gt=boxes.detach()+.05
        actual=overlaps(boxes,gt,paired=True,giou=True);expected=generalized_box_iou(boxes,gt).diagonal()
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
        self.assertTrue(torch.allclose(torch.autograd.grad(actual.sum(),boxes,retain_graph=True)[0],torch.autograd.grad(expected.sum(),boxes)[0],atol=1e-5))

    def test_official_print_is_tabular(self):
        result=dict(splits=dict(val=dict(expression_count=100,grec_precision_f1=.6,n_acc=.9,t_acc=.5,positive_precision_f1=.2)))
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):print_results(result)
        self.assertIn('GREC Pr@(F1=1)',stream.getvalue());self.assertIn('60.00%',stream.getvalue())


if __name__=='__main__':unittest.main()
