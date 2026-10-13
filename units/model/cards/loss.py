"""Batched grounding supervision with one CPU transfer for Hungarian matching."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from scipy.optimize import linear_sum_assignment


def focal_bce(logits,targets):
    probability=logits.sigmoid()
    pt=probability*targets+(1-probability)*(1-targets)
    alpha=.25*targets+.75*(1-targets)
    return F.binary_cross_entropy_with_logits(logits,targets,reduction='none')*alpha*(1-pt).square()


def overlaps(first,second,paired=False,giou=False):
    first,second=first.float(),second.float()
    if not paired:
        first,second=first.unsqueeze(-2),second.unsqueeze(-3)
    intersection=(torch.minimum(first[...,2:],second[...,2:])-torch.maximum(first[...,:2],second[...,:2])).clamp_min(0).prod(-1)
    area1=(first[...,2:]-first[...,:2]).clamp_min(0).prod(-1)
    area2=(second[...,2:]-second[...,:2]).clamp_min(0).prod(-1)
    union=area1+area2-intersection
    iou=intersection/union.clamp_min(1e-12)
    if not giou:return iou
    enclosure=(torch.maximum(first[...,2:],second[...,2:])-torch.minimum(first[...,:2],second[...,:2])).clamp_min(0).prod(-1)
    return iou-(enclosure-union)/enclosure.clamp_min(1e-12)


def assign_dense_targets(dense,target):
    """Compute fixed geometric labels on CPU in dataloader workers."""
    points,strides,levels=(dense[key].detach().cpu() for key in ('points','strides','levels'))
    gt=target['boxes'].detach().cpu()
    size=target['valid_size'].detach().cpu()
    valid=(points<size).all(-1)
    owner=torch.full((len(points),),-1,dtype=torch.long)
    if not len(gt):return owner,valid
    center=(gt[:,:2]+gt[:,2:])/2
    inside=(points[:,None,:]>gt[None,:,:2]).all(-1)&(points[:,None,:]<gt[None,:,2:]).all(-1)
    near=((points[:,None,:]-center[None]).abs()<2.5*strides[:,None,:]).all(-1)
    extent=(gt[:,2:]-gt[:,:2]).max(-1).values
    preferred=torch.where(extent<.15,0,torch.where(extent<.3,1,2))
    eligible=inside&near&(levels[:,None]==preferred[None])&valid[:,None]
    area=(gt[:,2:]-gt[:,:2]).prod(-1)
    costs=area[None,:].expand(len(points),-1).masked_fill(~eligible,float('inf'))
    best,ids=costs.min(1)
    owner[torch.isfinite(best)]=ids[torch.isfinite(best)]
    for obj in range(len(gt)):
        if not (owner==obj).any():
            distance=((points-center[obj])/strides).square().sum(-1)
            distance=distance.masked_fill(~valid|(owner>=0),float('inf'))
            index=distance.argmin()
            if torch.isfinite(distance[index]):owner[index]=obj
    return owner,valid


class GroundingLoss(nn.Module):
    def __init__(self,bbox_weight=5.,giou_weight=2.,dense_weight=1.,quality_weight=1.,
                 alignment_weight=1.,selection_weight=1.,ranking_weight=.25):
        super().__init__()
        self.weights=dict(bbox=bbox_weight,giou=giou_weight,dense=dense_weight,
                          quality=quality_weight,alignment=alignment_weight,
                          selection=selection_weight,ranking=ranking_weight)

    dense_targets=staticmethod(assign_dense_targets)

    def forward(self,outputs,targets):
        boxes=outputs['boxes'];dense=outputs['dense'];batch,count,_=boxes.shape
        device=boxes.device
        sizes=[len(target['boxes']) for target in targets]
        all_sizes=[len(target['all_boxes']) for target in targets]
        def padded(key):
            values=[target[key] for target in targets]
            result=pad_sequence(values,batch_first=True)
            return result if result.shape[1] else boxes.new_zeros(batch,1,4)
        gt=padded('boxes');all_gt=padded('all_boxes')
        counts=torch.tensor(sizes,device=device)
        gt_valid=torch.arange(gt.shape[1],device=device)[None,:]<counts[:,None]
        all_valid=torch.arange(all_gt.shape[1],device=device)[None,:]<torch.tensor(all_sizes,device=device)[:,None]
        owner=torch.stack([target['dense_owner'] for target in targets])
        valid=torch.stack([target['dense_valid'] for target in targets])
        positive=owner>=0
        normalizer=positive.sum(1).clamp_min(1)
        truth=gt.gather(1,owner.clamp_min(0)[...,None].expand(-1,-1,4))
        dense_l1=(dense['boxes']-truth).abs().mean(-1)
        dense_giou=1-overlaps(dense['boxes'],truth,paired=True,giou=True)
        losses=dict(dense=(focal_bce(dense['logits'],positive.float())*valid).sum(1).div(normalizer).mean(),
                    bbox=(dense_l1*positive).sum(1).div(normalizer).mean(),
                    giou=(dense_giou*positive).sum(1).div(normalizer).mean())
        with torch.no_grad():
            cost=5*torch.cdist(boxes.float(),gt.float(),p=1)-2*overlaps(boxes,gt,giou=True)
            cost=cost-outputs['alignment_logits'].sigmoid()[:,:,None]-outputs['selection_logits'].sigmoid()[:,:,None]
            cost_cpu=cost.float().cpu().numpy()
            if not np.isfinite(cost_cpu).all():raise ValueError('Non-finite matching costs')
            chosen=np.zeros((batch,count),dtype=np.bool_)
            objects=np.zeros((batch,count),dtype=np.int64)
            for i,size in enumerate(sizes):
                if size:
                    rows,cols=linear_sum_assignment(cost_cpu[i,:,:size])
                    chosen[i,rows]=True;objects[i,rows]=cols
            selected=torch.from_numpy(chosen).to(device)
            object_index=torch.from_numpy(objects).to(device)
            matched_gt=gt.gather(1,object_index[...,None].expand(-1,-1,4))
            query_iou=overlaps(boxes.detach(),gt).masked_fill(~gt_valid[:,None,:],0)
            semantic=(query_iou.amax(-1)>=.5)|selected
            duplicate_iou=overlaps(boxes.detach(),boxes.detach()).masked_fill(~selected[:,None,:],0)
            suppressed=(duplicate_iou.amax(-1)>.5)&~selected&semantic
            quality=overlaps(boxes.detach(),all_gt).masked_fill(~all_valid[:,None,:],0).amax(-1)
        match_normalizer=selected.sum(1).clamp_min(1)
        losses['bbox']=losses['bbox']+((boxes-matched_gt).abs().mean(-1)*selected).sum(1).div(match_normalizer).mean()
        losses['giou']=losses['giou']+((1-overlaps(boxes,matched_gt,paired=True,giou=True))*selected).sum(1).div(match_normalizer).mean()
        losses['quality']=F.binary_cross_entropy_with_logits(outputs['quality_logits'],quality)
        losses['alignment']=(focal_bce(outputs['alignment_logits'],semantic.float()).sum(1)/semantic.sum(1).clamp_min(1)).mean()
        known=~suppressed
        losses['selection']=(focal_bce(outputs['selection_logits'],selected.float())*known).sum(1).div(counts.clamp_min(1)).mean()
        log_scores=F.logsigmoid(outputs['selection_logits'])
        negatives=~selected&known
        hardest=log_scores.masked_fill(~negatives,-float('inf')).topk(min(10,count),dim=1).values
        active=torch.isfinite(hardest)
        margin=F.relu(math.log(2)-log_scores[:,:,None]+hardest[:,None,:].clamp_min(-1e4))
        pairs=selected[:,:,None]&active[:,None,:]
        losses['ranking']=(margin*pairs).sum((1,2)).div(match_normalizer*active.sum(1).clamp_min(1)).mean()
        total=sum(self.weights[name]*value for name,value in losses.items())
        if not torch.isfinite(total):raise ValueError('Non-finite grounding loss')
        return total,losses
