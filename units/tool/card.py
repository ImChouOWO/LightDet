"""Text-conditioned dense CNN localization with local ROI semantic scoring."""
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from torchvision.ops import FeaturePyramidNetwork, roi_align
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[2]


def conv(in_channels, out_channels, kernel=3):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, padding=kernel//2, bias=False),
                         nn.GroupNorm(8, out_channels), nn.SiLU())


class MultilingualMiniLM(nn.Module):
    """Encode Chinese/English tokens and pool a single referring expression."""
    def __init__(self, path, hidden_dim=256, freeze=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(path, local_files_only=True)
        self.freeze = freeze
        self.encoder.requires_grad_(not freeze)
        self.projection = nn.Sequential(nn.Linear(self.encoder.config.hidden_size, hidden_dim),
                                        nn.LayerNorm(hidden_dim), nn.GELU())

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def forward(self, tokens):
        with torch.set_grad_enabled(self.training and not self.freeze):
            raw = self.encoder(**tokens).last_hidden_state
        text = self.projection(raw)
        mask = tokens['attention_mask'].bool()
        pooled = (text * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return text, mask, pooled


class ResNetFPN(nn.Module):
    """Keep stride-8/16/32 maps for dense prediction and ROI sampling."""
    def __init__(self, hidden_dim, weights):
        super().__init__()
        self.backbone = resnet50(weights=None)
        if weights:
            state = torch.load(weights, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=True)
        self.backbone.fc = nn.Identity()
        self.fpn = FeaturePyramidNetwork([512, 1024, 2048], hidden_dim)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None,:,None,None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None,:,None,None])

    def train(self, mode=True):
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, images):
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1((images-self.mean)/self.std))))
        x = b.layer1(x)
        c3 = b.layer2(x); c4 = b.layer3(c3); c5 = b.layer4(c4)
        return self.fpn(OrderedDict(c3=c3, c4=c4, c5=c5))


class DenseLocalizationCNN(nn.Module):
    """Fuse text at each FPN location; densely connected convolutions predict center/size boxes."""
    def __init__(self, hidden_dim, growth=64, depth=3):
        super().__init__()
        self.film = nn.Linear(hidden_dim, hidden_dim*2)
        self.token_gate = nn.Conv2d(hidden_dim*2, hidden_dim, 1)
        self.coordinates = nn.Conv2d(2, hidden_dim, 1)
        self.layers = nn.ModuleList([conv(hidden_dim+i*growth, growth) for i in range(depth)])
        self.merge = conv(hidden_dim+depth*growth, hidden_dim, 1)
        self.box = nn.Conv2d(hidden_dim, 4, 1)
        self.objectness = nn.Conv2d(hidden_dim, 1, 1)
        nn.init.constant_(self.objectness.bias, -4.595)
        self.hidden_dim = hidden_dim

    def forward(self, features, text, text_mask, text_pooled, image_size):
        height, width = image_size
        scale, bias = self.film(text_pooled).chunk(2, -1)
        boxes, logits, points, strides, levels = [], [], [], [], []
        for level, feature in enumerate(features.values()):
            batch, dim, h, w = feature.shape
            gy,gx = torch.meshgrid(torch.linspace(-1,1,h,device=feature.device),
                                   torch.linspace(-1,1,w,device=feature.device),indexing='ij')
            positions = torch.stack([gx,gy])[None].expand(batch,-1,-1,-1).to(feature.dtype)
            feature = feature+self.coordinates(positions)
            pixels = feature.flatten(2).transpose(1,2)
            attention = (pixels @ text.transpose(1,2)) / dim**.5
            attention = attention.masked_fill(~text_mask[:,None,:], float('-inf')).softmax(-1)
            local_text = (attention @ text).transpose(1,2).reshape(batch, dim, h, w)
            x = self.token_gate(torch.cat([feature, local_text], 1))
            x = x * (1+torch.tanh(scale[:,:,None,None])) + bias[:,:,None,None]
            parts = [x]
            for layer in self.layers:
                parts.append(layer(torch.cat(parts, 1)))
            fused = self.merge(torch.cat(parts, 1))
            y, xcoord = torch.meshgrid(torch.arange(h,device=x.device),torch.arange(w,device=x.device),indexing='ij')
            centers = torch.stack([(xcoord+.5)/w,(y+.5)/h],-1).reshape(-1,2)
            stride = torch.tensor([1/w,1/h],device=feature.device,dtype=torch.float32)
            regression = self.box(fused).flatten(2).transpose(1,2)
            predicted_center = centers[None]+regression[:,:,:2].tanh()*stride*2
            size = F.softplus(regression[:,:,2:])*stride*4
            pred = torch.cat([predicted_center-size/2,predicted_center+size/2],-1)
            boxes.append(pred); logits.append(self.objectness(fused).flatten(1))
            points.append(centers); strides.append(stride.expand(h*w,2))
            levels.append(torch.full((h*w,),level,device=x.device,dtype=torch.long))
        return dict(boxes=torch.cat(boxes,1), logits=torch.cat(logits,1),
                    points=torch.cat(points), strides=torch.cat(strides), levels=torch.cat(levels))


class TextAlignmentTransformer(nn.Module):
    """A pooled ROI query attends only to its own ROI tokens; text is scored downstream."""
    def __init__(self, hidden_dim, heads, layers, roi_size):
        super().__init__()
        self.position = nn.Parameter(torch.randn(1,roi_size*roi_size,hidden_dim)*.02)
        self.geometry = nn.Sequential(nn.Linear(4,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU())
        self.layers = nn.ModuleList([nn.ModuleDict(dict(
            attention=nn.MultiheadAttention(hidden_dim,heads,batch_first=True),
            norm1=nn.LayerNorm(hidden_dim), norm2=nn.LayerNorm(hidden_dim),
            ffn=nn.Sequential(nn.Linear(hidden_dim,hidden_dim*2),nn.GELU(),nn.Linear(hidden_dim*2,hidden_dim)))) for _ in range(layers)])

    def forward(self, roi_tokens, boxes):
        memory = roi_tokens+self.position
        query = roi_tokens.mean(1,keepdim=True)+self.geometry(boxes)[:,None,:]
        for layer in self.layers:
            query = layer['norm1'](query+layer['attention'](query,memory,memory,need_weights=False)[0])
            query = layer['norm2'](query+layer['ffn'](query))
        return query[:,0]


class VisionTextModel(nn.Module):
    def __init__(self, hidden_dim=256, encoder_path='units/model/bert', freeze_text=True,
                 backbone_weights='units/model/resnet/checkpoints/resnet50-11ad3fa6.pth',
                 dense_growth=64, dense_depth=3, candidate_count=100, roi_size=8,
                 score_heads=8, score_layers=2):
        super().__init__()
        self.vision = ResNetFPN(hidden_dim, str(ROOT/backbone_weights) if backbone_weights else None)
        self.text = MultilingualMiniLM(str(ROOT/encoder_path),hidden_dim,freeze_text)
        self.localization = DenseLocalizationCNN(hidden_dim,dense_growth,dense_depth)
        self.refinement = TextAlignmentTransformer(hidden_dim,score_heads,score_layers,roi_size)
        self.visual_projection = nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,hidden_dim))
        self.text_projection = nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,hidden_dim))
        self.quality_head = nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,1))
        self.selection_head = nn.Sequential(nn.Linear(hidden_dim*2,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,1))
        self.logit_scale = nn.Parameter(torch.tensor(2.3))
        self.alignment_bias = nn.Parameter(torch.tensor(-2.2))
        nn.init.constant_(self.selection_head[-1].bias,-2.2)
        self.candidate_count, self.roi_size = candidate_count, roi_size

    def pool_rois(self, features, boxes, image_size):
        batch, count, _ = boxes.shape
        height, width = image_size
        pixel_boxes = boxes.detach().clamp(0,1)*boxes.new_tensor([width,height,width,height])
        rois = torch.cat([torch.arange(batch,device=boxes.device)[:,None,None].expand(batch,count,1),pixel_boxes],-1).reshape(-1,5)
        area = ((pixel_boxes[:,:,2:]-pixel_boxes[:,:,:2]).clamp_min(0).prod(-1)).sqrt()
        levels = (torch.floor(4+torch.log2(area.clamp_min(1)/224))-3).clamp(0,2).long().flatten()
        maps = list(features.values())
        pooled = maps[0].new_zeros(batch*count,maps[0].shape[1],self.roi_size,self.roi_size)
        for level, feature in enumerate(maps):
            ids = torch.where(levels==level)[0]
            if len(ids):
                pooled[ids] = roi_align(feature,rois[ids].float(),self.roi_size,
                    spatial_scale=feature.shape[-1]/width,sampling_ratio=2,aligned=True).to(pooled.dtype)
        return pooled.flatten(2).transpose(1,2)

    def score_rois(self, features, boxes, pooled_text, image_size):
        count = boxes.shape[1]
        roi = self.pool_rois(features, boxes, image_size)
        refined = self.refinement(roi, boxes.detach().reshape(-1,4)).reshape(len(boxes),count,-1)
        visual = F.normalize(self.visual_projection(refined),dim=-1)
        phrase = F.normalize(self.text_projection(pooled_text),dim=-1)
        alignment = (visual*phrase[:,None,:]).sum(-1)*self.logit_scale.exp().clamp(max=100)+self.alignment_bias
        quality = self.quality_head(roi.mean(1)).reshape(len(boxes),count)
        selection = self.selection_head(torch.cat([refined,pooled_text[:,None,:].expand_as(refined)],-1)).squeeze(-1)
        return alignment,quality,selection

    def forward(self, images, tokens, targets=None, guide_candidates=False, gt_roi=False, features=None, valid_sizes=None, image_indices=None):
        features = self.vision(images) if features is None else features
        if image_indices is not None:
            features = OrderedDict((name,value.index_select(0,image_indices)) for name,value in features.items())
        text, mask, pooled_text = self.text(tokens)
        dense = self.localization(features,text,mask,pooled_text,images.shape[-2:])
        count = min(self.candidate_count,dense['logits'].shape[1])
        proposal_logits = dense['logits']
        if valid_sizes is None and targets is not None:
            valid_sizes = torch.stack([target['valid_size'] for target in targets])
        if valid_sizes is not None:
            valid = (dense['points'][None,:,:] < valid_sizes[:,None,:]).all(-1)
            proposal_logits = proposal_logits.masked_fill(~valid,float('-inf'))
        ids = proposal_logits.topk(count,1).indices
        if guide_candidates and targets is not None:
            for i,target in enumerate(targets):
                gt = target['boxes']
                if len(gt):
                    centers = (gt[:,:2]+gt[:,2:])/2
                    distance = (dense['points'][None]-centers[:,None]).square().sum(-1)
                    guided = distance.argmin(1).unique()
                    rest = ids[i][~torch.isin(ids[i],guided)]
                    ids[i] = torch.cat([guided,rest])[:count]
        boxes = dense['boxes'].gather(1,ids[...,None].expand(-1,-1,4)).clamp(0,1)
        teacher_count = max((len(target['boxes']) for target in targets),default=0) if gt_roi and targets is not None else 0
        scoring_boxes = boxes
        if teacher_count:
            truth = boxes.new_zeros(len(boxes),teacher_count,4)
            for i,target in enumerate(targets):
                truth[i,:len(target['boxes'])] = target['boxes']
            scoring_boxes = torch.cat([boxes,truth],1)
        all_alignment,all_quality,all_selection = self.score_rois(features,scoring_boxes,pooled_text,images.shape[-2:])
        alignment,quality,selection = all_alignment[:,:count],all_quality[:,:count],all_selection[:,:count]
        base = torch.sqrt(quality.detach().float().sigmoid()*alignment.float().sigmoid()).clamp(1e-6,1-1e-6)
        selection = torch.logit(base)+selection.float()
        scores = selection.sigmoid()
        outputs = dict(boxes=boxes,scores=scores,quality_logits=quality,
                       alignment_logits=alignment,selection_logits=selection,dense=dense)
        if gt_roi and targets is not None:
            outputs['gt_alignment_logits'] = [all_alignment[i,count:count+len(target['boxes'])]
                for i,target in enumerate(targets) if len(target['boxes'])]
        return outputs
