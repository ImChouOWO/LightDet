"""Dataset-specific GREC and RefCOCOg metrics; Main inference has no NMS."""
import torch
from torchvision.ops import box_iou
from units.model.tool.official_grec import generalized_box_iou
from units.model.tool.runtime import select_predictions


def build_metrics(config):
    loader = config['data']['dataloader']
    if loader == 'grefcoco_unc':
        return GroundingMetrics(**config['eval'])
    if loader == 'refcocog_umd':
        return RefCOCOgMetrics(**config['eval'])
    raise ValueError(f'Unknown data.dataloader: {loader}')


class RefCOCOgMetrics:
    """Single-GT specialization of the LocateAnything per-expression P/R protocol."""
    def __init__(self, score_threshold=.7, use_nms=False, nms_iou=.5, max_detections=10):
        self.settings = dict(score_threshold=score_threshold, use_nms=use_nms,
                             nms_iou=nms_iou, max_detections=max_detections)
        self.thresholds = torch.tensor([.5, .55, .6, .65, .7, .75, .8, .85, .9, .95], dtype=torch.float64)
        self.precision = torch.zeros(10, dtype=torch.float64)
        self.recall = torch.zeros(10, dtype=torch.float64)
        self.total = self.top1_correct = self.rejected = self.selected = 0

    def update(self, boxes, scores, target):
        boxes, scores = boxes.detach().float().cpu(), scores.detach().float().cpu()
        gt = target['boxes'].detach().double().cpu()
        size = target['valid_size'].detach().cpu()
        if len(gt) != 1:
            raise ValueError('RefCOCOg evaluation requires exactly one GT per expression')
        if not all(torch.isfinite(value).all() for value in (boxes, scores, gt)):
            raise ValueError('Non-finite evaluation input')
        raw, _ = select_predictions(boxes, scores, size, score_threshold=-float('inf'),
                                    use_nms=False, max_detections=1)
        predicted, _ = select_predictions(boxes, scores, size, **self.settings)
        self.total += 1
        self.selected += len(predicted)
        self.rejected += not len(predicted)
        self.top1_correct += int((box_iou(raw.double(), gt) >= .5).any())
        if len(predicted):
            matched = (box_iou(predicted.double(), gt).max() >= self.thresholds).double()
            self.precision += matched / len(predicted)
            self.recall += matched

    def compute(self):
        if not self.total:
            raise ValueError('No RefCOCOg expressions evaluated')
        precision, recall = self.precision / self.total, self.recall / self.total
        # The official protocol averages expression P/R before computing F1.
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        return dict(f1_at_05=float(f1[0]), f1_at_095=float(f1[-1]), f1_mean=float(f1.mean()),
                    rec_acc50=self.top1_correct / self.total, expression_count=self.total,
                    rejection_rate=self.rejected / self.total, avg_selected=self.selected / self.total,
                    by_iou={f'{value:.2f}': dict(precision=float(p), recall=float(r), f1=float(f))
                            for value, p, r, f in zip(self.thresholds, precision, recall, f1)},
                    metric_schema='lightdet.refcocog.umd.v1',
                    protocol=dict(overlap='IoU', iou_thresholds=self.thresholds.tolist(),
                                  reduction='mean expression precision/recall, then F1',
                                  acc50='top-1 without score threshold', **self.settings))


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
