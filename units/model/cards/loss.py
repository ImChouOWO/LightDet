"""Dense localization, ROI quality/semantics, and unique set-selection supervision."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_iou, generalized_box_iou
from scipy.optimize import linear_sum_assignment


def focal_bce(logits,targets):
    probability=logits.sigmoid()
    pt=probability*targets+(1-probability)*(1-targets)
    alpha=.25*targets+.75*(1-targets)
    return F.binary_cross_entropy_with_logits(logits,targets,reduction='none')*alpha*(1-pt).square()


class GroundingLoss(nn.Module):
    def __init__(self,bbox_weight=5.,giou_weight=2.,dense_weight=1.,quality_weight=1.,
                 alignment_weight=1.,selection_weight=1.,ranking_weight=.25):
        super().__init__()
        self.weights=dict(bbox=bbox_weight,giou=giou_weight,dense=dense_weight,
                          quality=quality_weight,alignment=alignment_weight,
                          selection=selection_weight,ranking=ranking_weight)

    def dense_targets(self,dense,target):
        points,strides=dense['points'],dense['strides']
        gt=target['boxes']
        valid=(points[:,0]<target['valid_size'][0])&(points[:,1]<target['valid_size'][1])
        owner=torch.full((len(points),),-1,device=points.device,dtype=torch.long)
        if not len(gt):return owner,valid
        center=(gt[:,:2]+gt[:,2:])/2
        inside=(points[:,None,:]>gt[None,:,:2]).all(-1)&(points[:,None,:]<gt[None,:,2:]).all(-1)
        near=((points[:,None,:]-center[None]).abs()<2.5*strides[:,None,:]).all(-1)
        extent=(gt[:,2:]-gt[:,:2]).max(-1).values
        preferred=torch.where(extent<.15,0,torch.where(extent<.3,1,2))
        eligible=inside&near&(dense['levels'][:,None]==preferred[None])&valid[:,None]
        area=(gt[:,2:]-gt[:,:2]).prod(-1)
        costs=area[None,:].expand(len(points),-1).masked_fill(~eligible,float('inf'))
        best,ids=costs.min(1)
        owner[torch.isfinite(best)]=ids[torch.isfinite(best)]
        # Tiny GTs can fall between grid points; provide one nearest free point per GT.
        for obj in range(len(gt)):
            if not (owner==obj).any():
                distance=((points-center[obj])/strides).square().sum(-1)
                distance=distance.masked_fill(~valid|(owner>=0),float('inf'))
                index=distance.argmin()
                if torch.isfinite(distance[index]):owner[index]=obj
        return owner,valid

    def forward(self,outputs,targets):
        zero=outputs['boxes'].sum()*0
        losses={name:zero for name in self.weights}
        dense=outputs['dense']
        for i,target in enumerate(targets):
            gt=target['boxes'];boxes=outputs['boxes'][i]
            owners,valid=self.dense_targets(dense,target)
            positive=owners>=0
            dense_labels=positive.float()
            losses['dense']=losses['dense']+focal_bce(dense['logits'][i][valid],dense_labels[valid]).sum()/max(int(positive.sum()),1)
            if positive.any():
                pred=dense['boxes'][i][positive];truth=gt[owners[positive]]
                losses['bbox']=losses['bbox']+F.l1_loss(pred,truth)
                losses['giou']=losses['giou']+(1-generalized_box_iou(pred.float(),truth.float()).diagonal()).mean()
            semantic=torch.zeros(len(boxes),device=boxes.device)
            selected=torch.zeros_like(semantic)
            suppressed_duplicate=torch.zeros_like(semantic,dtype=torch.bool)
            overlaps=box_iou(boxes.detach().float(),gt.float())
            if len(gt):
                semantic=(overlaps.max(1).values>=.5).float()
                with torch.no_grad():
                    cost=5*torch.cdist(boxes.float(),gt.float(),p=1)-2*generalized_box_iou(boxes.float(),gt.float())
                    cost=cost-outputs['alignment_logits'][i].sigmoid()[:,None]-outputs['selection_logits'][i].sigmoid()[:,None]
                    if not torch.isfinite(cost).all():raise ValueError('Non-finite matching costs')
                    rows,cols=linear_sum_assignment(cost.cpu().numpy())
                    rows=torch.as_tensor(rows,device=boxes.device);cols=torch.as_tensor(cols,device=boxes.device)
                selected[rows]=1;semantic[rows]=1
                suppressed_duplicate=(box_iou(boxes.detach().float(),boxes[rows].detach().float()).max(1).values>.5)&(selected==0)&(semantic==1)
                losses['bbox']=losses['bbox']+F.l1_loss(boxes[rows],gt[cols])
                losses['giou']=losses['giou']+(1-generalized_box_iou(boxes[rows].float(),gt[cols].float()).diagonal()).mean()
            all_gt=target['all_boxes']
            quality=box_iou(boxes.detach().float(),all_gt.float()).max(1).values if len(all_gt) else torch.zeros_like(semantic)
            losses['quality']=losses['quality']+F.binary_cross_entropy_with_logits(outputs['quality_logits'][i],quality)
            losses['alignment']=losses['alignment']+focal_bce(outputs['alignment_logits'][i],semantic).sum()/max(int(semantic.sum()),1)
            # Ignore same-object duplicates already removed by NMS, avoiding contradictory labels.
            known=~suppressed_duplicate
            losses['selection']=losses['selection']+focal_bce(outputs['selection_logits'][i][known],selected[known]).sum()/max(len(gt),1)
            negatives=(selected==0)&known
            if selected.any() and negatives.any():
                log_scores=F.logsigmoid(outputs['selection_logits'][i])
                negative=log_scores[negatives].topk(min(10,int(negatives.sum()))).values
                positive_scores=log_scores[selected==1]
                losses['ranking']=losses['ranking']+F.relu(math.log(2)-positive_scores[:,None]+negative[None,:]).mean()
        losses={name:value/len(targets) for name,value in losses.items()}
        total=sum(self.weights[name]*value for name,value in losses.items())
        if not torch.isfinite(total):raise ValueError('Non-finite grounding loss')
        return total,losses
