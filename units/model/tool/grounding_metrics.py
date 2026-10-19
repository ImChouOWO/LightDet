"""GREC reference GIoU greedy matching; Main inference has no NMS."""
import torch
from torchvision.ops import box_iou
from units.model.tool.official_grec import generalized_box_iou
from units.model.tool.runtime import select_predictions


class GroundingMetrics:
    def __init__(self,score_threshold=.7,use_nms=False,nms_iou=.5,max_detections=100):
        self.settings=dict(score_threshold=score_threshold,use_nms=use_nms,nms_iou=nms_iou,max_detections=max_detections)
        self.total=self.correct=self.single=self.single_correct=0
        self.empty=self.empty_correct=self.positive=self.positive_correct=self.target_detected=self.multi=0

    def update(self,boxes,scores,target):
        boxes,scores=boxes.detach().float().cpu(),scores.detach().float().cpu()
        gt=target['boxes'].detach().float().cpu();size=target['valid_size'].detach().cpu()
        if not all(torch.isfinite(t).all() for t in (boxes,scores,gt)):raise ValueError('Non-finite evaluation input')
        raw,_=select_predictions(boxes,scores,size,score_threshold=-float('inf'),use_nms=False,max_detections=1)
        predicted,_=select_predictions(boxes,scores,size,**self.settings)
        matches=0
        if len(predicted) and len(gt):
            matrix=generalized_box_iou(predicted,gt)
            for _ in range(min(len(predicted),len(gt))):
                value,index=matrix.flatten().max(0)
                if value<.5:break
                row=index//len(gt);column=index%len(gt)
                matches+=1;matrix[row,:]=0.;matrix[:,column]=0.
        correct=int(len(predicted)==len(gt)==matches)
        self.total+=1;self.correct+=correct
        if not len(gt):self.empty+=1;self.empty_correct+=int(not len(predicted))
        else:
            self.positive+=1;self.positive_correct+=correct;self.target_detected+=int(len(predicted)>0)
            if len(gt)==1:
                self.single+=1;self.single_correct+=int((box_iou(raw,gt)>.5).any())
            else:self.multi+=1

    def compute(self):
        def ratio(a,b):return a/b if b else None
        return dict(grec_precision_f1=ratio(self.correct,self.total),rec_acc50=ratio(self.single_correct,self.single),
            n_acc=ratio(self.empty_correct,self.empty),t_acc=ratio(self.target_detected,self.positive),
            positive_precision_f1=ratio(self.positive_correct,self.positive),expression_count=self.total,
            single_target_count=self.single,multi_target_count=self.multi,no_target_count=self.empty,
            metric_schema='lightdet.grefcoco.queries.v2',protocol=dict(overlap='GIoU',iou_threshold=.5,
                matching='greedy maximum overlap one-to-one',**self.settings))
