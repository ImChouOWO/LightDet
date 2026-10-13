"""GREC set accuracy and single-target REC accuracy at IoU 0.5."""
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

from units.model.tool.runtime import select_predictions


class GroundingMetrics:
    def __init__(self,score_threshold=.5,nms_iou=.5,max_detections=100):
        self.settings=dict(score_threshold=score_threshold,nms_iou=nms_iou,max_detections=max_detections)
        self.total=self.correct=self.single=self.single_correct=0
        self.empty=self.empty_correct=self.positive=self.target_detected=0
        self.multi=0

    def update(self,boxes,scores,target):
        boxes,scores=boxes.detach().float().cpu(),scores.detach().float().cpu()
        gt=target['boxes'].detach().float().cpu()
        valid_size=target['valid_size'].detach().cpu()
        if not all(torch.isfinite(value).all() for value in (boxes,scores,gt)):
            raise ValueError('Non-finite evaluation input')
        # REC uses the highest-scoring valid prediction, even below the set threshold.
        raw,_=select_predictions(boxes,scores,valid_size,score_threshold=-float('inf'),
                                  nms_iou=self.settings['nms_iou'],max_detections=1)
        selected,_=select_predictions(boxes,scores,valid_size,**self.settings)
        matches=0
        if len(selected) and len(gt):
            edges=box_iou(selected,gt)>=.5
            rows,cols=linear_sum_assignment(-edges.int().numpy())
            matches=int(edges[rows,cols].sum())
        self.total+=1
        self.correct+=int(len(selected)==len(gt)==matches)
        if not len(gt):
            self.empty+=1;self.empty_correct+=int(not len(selected))
        else:
            self.positive+=1;self.target_detected+=int(len(selected)>0)
            if len(gt)==1:
                self.single+=1;self.single_correct+=int((box_iou(raw,gt)>.5).any())
            else:self.multi+=1

    def compute(self):
        def ratio(a,b):return a/b if b else None
        return dict(grec_precision_f1=ratio(self.correct,self.total),rec_acc50=ratio(self.single_correct,self.single),
                    n_acc=ratio(self.empty_correct,self.empty),t_acc=ratio(self.target_detected,self.positive),
                    expression_count=self.total,single_target_count=self.single,multi_target_count=self.multi,
                    no_target_count=self.empty,metric_schema='lightdet.grefcoco.v1',
                    protocol=dict(iou_threshold=.5,matching='maximum one-to-one IoU matching',**self.settings))
