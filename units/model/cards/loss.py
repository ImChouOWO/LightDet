"""Main one-to-one/no-object supervision and optional geometric one-to-many Aux loss."""
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


def threshold_margin_loss(logits,targets,valid,threshold=.7,margin=.5):
    if not 0<threshold<1 or margin<0:raise ValueError('Invalid threshold or margin')
    boundary=math.log(threshold/(1-threshold));logits=logits.float()
    positive=valid.bool()&(targets==1);negative=valid.bool()&(targets==0)
    first=F.relu(boundary+margin-logits).masked_fill(~positive,0)
    second=F.relu(logits-boundary+margin).masked_fill(~negative,0)
    return first.sum()/positive.sum().clamp_min(1)+second.sum()/negative.sum().clamp_min(1)


def overlaps(first,second,paired=False,giou=False):
    first,second=first.float(),second.float()
    if not paired:first,second=first.unsqueeze(-2),second.unsqueeze(-3)
    intersection=(torch.minimum(first[...,2:],second[...,2:])-torch.maximum(first[...,:2],second[...,:2])).clamp_min(0).prod(-1)
    area1=(first[...,2:]-first[...,:2]).clamp_min(0).prod(-1)
    area2=(second[...,2:]-second[...,:2]).clamp_min(0).prod(-1)
    union=area1+area2-intersection;iou=intersection/union.clamp_min(1e-12)
    if not giou:return iou
    enclosure=(torch.maximum(first[...,2:],second[...,2:])-torch.minimum(first[...,:2],second[...,:2])).clamp_min(0).prod(-1)
    return iou-(enclosure-union)/enclosure.clamp_min(1e-12)


def allocate_aux(cost,iou,k,min_iou):
    """Reserve qualified Hungarian matches, then fill lowest-cost edges with unique ownership."""
    count,objects=cost.shape
    owner=np.full(count,-1,dtype=np.int64);used=np.zeros(objects,dtype=np.int64)
    if not objects:return owner
    rows,cols=linear_sum_assignment(cost)
    for query,obj in zip(rows,cols):
        if iou[query,obj]>=min_iou:owner[query]=obj;used[obj]+=1
    eligible=iou>=min_iou
    edges=np.argwhere(eligible)
    if len(edges):
        order=np.argsort(cost[edges[:,0],edges[:,1]],kind='stable')
        for query,obj in edges[order]:
            if owner[query]<0 and used[obj]<k:owner[query]=obj;used[obj]+=1
    return owner


class GroundingLoss(nn.Module):
    def __init__(self,bbox_weight=5.,giou_weight=2.,quality_weight=1.,alignment_weight=1.,
                 selection_weight=1.,ranking_weight=.25,no_object_weight=.1,
                 main_weight=.8,aux_weight=.2,aux_top_k=3,aux_min_iou=.3,
                 aux_start_epoch=6,aux_warmup_end_epoch=15,aux_decay_start_epoch=23,aux_end_epoch=30,
                 selection_threshold=.7,threshold_margin_weight=.1,threshold_margin_target=.5,
                 threshold_margin_start_epoch=6,threshold_margin_end_epoch=15):
        super().__init__()
        self.weights=dict(bbox=bbox_weight,giou=giou_weight,quality=quality_weight,
                          alignment=alignment_weight,selection=selection_weight,ranking=ranking_weight)
        if not 0<selection_threshold<1 or threshold_margin_weight<0 or threshold_margin_target<0:
            raise ValueError('Invalid threshold margin settings')
        if not 1<=threshold_margin_start_epoch<=threshold_margin_end_epoch:raise ValueError('Invalid margin schedule')
        if not 1<=aux_start_epoch<=aux_warmup_end_epoch<=aux_decay_start_epoch<aux_end_epoch:raise ValueError('Invalid Aux schedule')
        if aux_top_k<1 or not 0<=aux_min_iou<=1 or no_object_weight<=0 or min(main_weight,aux_weight)<0:
            raise ValueError('Invalid branch supervision settings')
        self.main_weight,self.aux_weight=main_weight,aux_weight
        self.aux_top_k,self.aux_min_iou=aux_top_k,aux_min_iou
        self.aux_epochs=(aux_start_epoch,aux_warmup_end_epoch,aux_decay_start_epoch,aux_end_epoch)
        self.no_object_weight=no_object_weight
        self.selection_threshold=selection_threshold
        self.threshold_margin_weight=threshold_margin_weight
        self.threshold_margin_target=threshold_margin_target
        self.threshold_margin_start_epoch=threshold_margin_start_epoch
        self.threshold_margin_end_epoch=threshold_margin_end_epoch

    def margin_schedule(self,epoch):
        start,end=self.threshold_margin_start_epoch,self.threshold_margin_end_epoch
        progress=min(1.,max(0.,(epoch-start+1)/(end-start+1)))
        factor=.5*(1-math.cos(math.pi*progress))
        return self.threshold_margin_target*factor,self.threshold_margin_weight*factor

    def aux_factor(self,epoch):
        start,warm,decay,end=self.aux_epochs
        if epoch<start:return 0.
        if epoch<=warm:return .5*(1-math.cos(math.pi*(epoch-start+1)/(warm-start+1)))
        if epoch<=decay:return 1.
        return .5*(1+math.cos(math.pi*min(1.,(epoch-decay)/(end-decay))))

    def branch_loss(self,outputs,targets,epoch,one_to_many):
        boxes=outputs['boxes'];batch,count,_=boxes.shape;device=boxes.device
        sizes=[len(t['boxes']) for t in targets]
        def padded(key):
            value=pad_sequence([t[key] for t in targets],batch_first=True)
            return value if value.shape[1] else boxes.new_zeros(batch,1,4)
        gt=padded('boxes');all_gt=padded('all_boxes')
        counts=torch.tensor(sizes,device=device)
        gt_valid=torch.arange(gt.shape[1],device=device)[None]<counts[:,None]
        all_valid=torch.arange(all_gt.shape[1],device=device)[None]<torch.tensor([len(t['all_boxes']) for t in targets],device=device)[:,None]
        with torch.no_grad():
            query_iou=overlaps(boxes.detach(),gt).masked_fill(~gt_valid[:,None],0)
            cost=5*torch.cdist(boxes.float(),gt.float(),p=1)-2*overlaps(boxes,gt,giou=True)
            cost=cost-outputs['selection_logits'].float().sigmoid()[:,:,None]
            geometry=torch.stack([cost,query_iou],0).float().cpu().numpy()
            if not np.isfinite(geometry).all():raise ValueError('Non-finite matching geometry')
            owners=np.full((batch,count),-1,dtype=np.int64)
            for i,size in enumerate(sizes):
                if size:
                    if one_to_many:owners[i]=allocate_aux(geometry[0,i,:,:size],geometry[1,i,:,:size],self.aux_top_k,self.aux_min_iou)
                    else:
                        rows,cols=linear_sum_assignment(geometry[0,i,:,:size]);owners[i,rows]=cols
            owner=torch.from_numpy(owners).to(device)
            selected=owner>=0
            truth=gt.gather(1,owner.clamp_min(0)[...,None].expand(-1,-1,4))
            multiplicity=boxes.new_zeros(batch,gt.shape[1]).scatter_add_(1,owner.clamp_min(0),selected.float())
            positive_weight=selected.float()/multiplicity.gather(1,owner.clamp_min(0)).clamp_min(1)
            normalizer=positive_weight.sum(1).clamp_min(1)
            quality=overlaps(boxes.detach(),all_gt).masked_fill(~all_valid[:,None],0).amax(-1)
            semantic=(query_iou.amax(-1)>=.5)|selected
        losses=dict(bbox=((boxes-truth).abs().mean(-1)*positive_weight).sum(1).div(normalizer).mean(),
                    giou=((1-overlaps(boxes,truth,paired=True,giou=True))*positive_weight).sum(1).div(normalizer).mean(),
                    quality=F.binary_cross_entropy_with_logits(outputs['quality_logits'].float(),quality),
                    alignment=(focal_bce(outputs['alignment_logits'].float(),semantic.float()).sum(1)/semantic.sum(1).clamp_min(1)).mean())
        # Every unmatched query is a no-object target, including duplicate boxes.
        weights=torch.where(selected,positive_weight,boxes.new_full((),self.no_object_weight))
        classification=F.binary_cross_entropy_with_logits(outputs['selection_logits'].float(),selected.float(),reduction='none')
        losses['selection']=(classification*weights).sum(1).div(weights.sum(1).clamp_min(1)).mean()
        log_scores=F.logsigmoid(outputs['selection_logits'].float())
        hardest=log_scores.masked_fill(selected,-float('inf')).topk(min(10,count),dim=1).values
        active=torch.isfinite(hardest)
        pairs=selected[:,:,None]&active[:,None]
        ranking=F.relu(math.log(2)-log_scores[:,:,None]+hardest[:,None].clamp_min(-1e4))
        losses['ranking']=(ranking*pairs*positive_weight[:,:,None]).sum((1,2)).div(normalizer*active.sum(1).clamp_min(1)).mean()
        margin,weight=self.margin_schedule(epoch)
        margin_loss=outputs['selection_logits'].float().sum()*0
        if weight>0:
            margin_loss=threshold_margin_loss(outputs['selection_logits'],selected,torch.ones_like(selected),self.selection_threshold,margin)
            losses['selection']=losses['selection']+weight*margin_loss
        total=sum(self.weights[name]*value for name,value in losses.items())
        losses['threshold_margin']=margin_loss
        return total,losses

    def forward(self,outputs,targets,epoch=1):
        main,parts=self.branch_loss(outputs,targets,epoch,False)
        total=self.main_weight*main
        parts={f'main_{name}':value for name,value in parts.items()}
        if 'aux' in outputs:
            aux,aux_parts=self.branch_loss(outputs['aux'],targets,epoch,True)
            total=total+self.aux_weight*self.aux_factor(epoch)*aux
            parts.update({f'aux_{name}':value for name,value in aux_parts.items()})
        if not torch.isfinite(total):raise ValueError('Non-finite grounding loss')
        return total,parts
