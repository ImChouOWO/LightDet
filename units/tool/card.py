"""Transformer object queries with Main set prediction and optional Aux supervision."""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from torchvision.ops import FeaturePyramidNetwork, roi_align
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[2]


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
    """Keep stride-8/16/32 maps for query attention and ROI sampling."""
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


class ObjectQueryTransformer(nn.Module):
    """Zero-content queries read text, add learned positions and pooled projection, then attend image/text memory."""
    def __init__(self,dim,heads,layers,queries,aux_queries,grids):
        super().__init__()
        self.main_queries=nn.Embedding(queries,dim)
        self.aux_queries=nn.Embedding(aux_queries,dim)
        self.coordinates=nn.Linear(2,dim)
        self.level_embedding=nn.Parameter(torch.randn(3,dim)*.02)
        self.text_type=nn.Parameter(torch.randn(1,1,dim)*.02)
        self.text_condition=nn.Linear(dim,dim)
        self.text_attention=nn.MultiheadAttention(dim,heads,dropout=.1,batch_first=True)
        self.memory_norm=nn.LayerNorm(dim)
        layer=nn.TransformerDecoderLayer(dim,heads,dim*4,dropout=.1,batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(layer,layers,norm=nn.LayerNorm(dim))
        self.grids=grids

    def guide_queries(self,positions,text,text_mask,pooled_text):
        positions=positions[None].expand(len(text),-1,-1)
        content=torch.zeros_like(positions)
        guided=self.text_attention(content+positions,text,text,
            key_padding_mask=~text_mask,need_weights=False)[0]
        return guided+positions+self.text_condition(pooled_text)[:,None]

    def forward(self,features,text,text_mask,pooled_text,valid_sizes,use_aux):
        memories=[];padding=[]
        for level,(feature,grid) in enumerate(zip(features.values(),self.grids)):
            h,w=min(grid,feature.shape[-2]),min(grid,feature.shape[-1])
            feature=F.adaptive_avg_pool2d(feature,(h,w))
            y,x=torch.meshgrid((torch.arange(h,device=feature.device)+.5)/h,
                               (torch.arange(w,device=feature.device)+.5)/w,indexing='ij')
            coordinates=torch.stack([x,y],-1).reshape(-1,2)
            position=self.coordinates(coordinates.to(feature.dtype))+self.level_embedding[level]
            memories.append(feature.flatten(2).transpose(1,2)+position[None])
            padding.append(~(coordinates[None]<valid_sizes[:,None]).all(-1))
        memories.append(text+self.text_type)
        padding.append(~text_mask)
        memory=self.memory_norm(torch.cat(memories,1))
        query=self.main_queries.weight
        attention_mask=None
        if use_aux:
            main_count=len(query)
            query=torch.cat([query,self.aux_queries.weight],0)
            attention_mask=torch.zeros(len(query),len(query),dtype=torch.bool,device=query.device)
            attention_mask[:main_count,main_count:]=True
            attention_mask[main_count:,:main_count]=True
        query=self.guide_queries(query,text,text_mask,pooled_text)
        decoded=self.decoder(query,memory,tgt_mask=attention_mask,memory_key_padding_mask=torch.cat(padding,1))
        main=decoded[:,:self.main_queries.num_embeddings]
        aux=decoded[:,self.main_queries.num_embeddings:] if use_aux else None
        return main,aux


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


class GroundingHead(nn.Module):
    """Query-conditioned box and ROI/text scores; final logits classify target versus no-object."""
    def __init__(self,dim):
        super().__init__()
        self.box=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,4))
        self.visual_projection=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim))
        self.text_projection=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim))
        self.quality=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,1))
        self.query_condition=nn.Sequential(nn.Linear(dim,dim),nn.LayerNorm(dim))
        self.selection=nn.Sequential(nn.Linear(dim*2,dim),nn.GELU(),nn.Linear(dim,1))
        self.logit_scale=nn.Parameter(torch.tensor(2.3))
        self.alignment_bias=nn.Parameter(torch.tensor(-2.2))
        nn.init.constant_(self.selection[-1].bias,-2.2)

    def predict_boxes(self,queries,valid_sizes):
        center,size=self.box(queries).float().sigmoid().chunk(2,-1)
        center=center*valid_sizes[:,None]
        size=size.clamp_min(1e-4)*valid_sizes[:,None]
        low=(center-size/2).clamp_min(0)
        high=torch.minimum(center+size/2,valid_sizes[:,None])
        return torch.cat([low,high],-1)

    def score(self,raw_roi,refined,queries,text):
        visual=F.normalize(self.visual_projection(refined),dim=-1)
        phrase=F.normalize(self.text_projection(text),dim=-1)
        alignment=(visual*phrase[:,None]).sum(-1)*self.logit_scale.exp().clamp(max=100)+self.alignment_bias
        quality=self.quality(raw_roi).squeeze(-1)
        selection_feature=refined+self.query_condition(queries)
        residual=self.selection(torch.cat([selection_feature,text[:,None].expand_as(refined)],-1)).squeeze(-1)
        base=torch.sqrt(quality.detach().float().sigmoid()*alignment.float().sigmoid()).clamp(1e-6,1-1e-6)
        selection=torch.logit(base)+residual.float()
        return dict(scores=selection.sigmoid(),quality_logits=quality,alignment_logits=alignment,
                    selection_logits=selection,no_object_logits=-selection,selection_residual=residual)


class VisionTextModel(nn.Module):
    def __init__(self,hidden_dim=256,encoder_path='units/model/bert',freeze_text=True,
                 backbone_weights='units/model/resnet/checkpoints/resnet50-11ad3fa6.pth',
                 num_object_queries=10,num_aux_queries=100,query_heads=8,query_layers=2,
                 memory_grids=(20,10,5),roi_size=8,score_heads=8,score_layers=2,aux_enabled=False):
        super().__init__()
        if len(memory_grids)!=3 or min(memory_grids)<1 or min(num_object_queries,num_aux_queries)<1:
            raise ValueError('Invalid query or memory dimensions')
        self.vision=ResNetFPN(hidden_dim,str(ROOT/backbone_weights) if backbone_weights else None)
        self.text=MultilingualMiniLM(str(ROOT/encoder_path),hidden_dim,freeze_text)
        self.localization=ObjectQueryTransformer(hidden_dim,query_heads,query_layers,num_object_queries,num_aux_queries,memory_grids)
        self.refinement=TextAlignmentTransformer(hidden_dim,score_heads,score_layers,roi_size)
        self.main_head=GroundingHead(hidden_dim)
        self.aux_head=deepcopy(self.main_head)
        self.roi_size=roi_size
        self.aux_enabled=aux_enabled
        if not aux_enabled:
            self.aux_head.requires_grad_(False)
            self.localization.aux_queries.requires_grad_(False)

    def pool_rois(self,features,boxes,image_size):
        batch,count,_=boxes.shape
        height,width=image_size
        pixel_boxes=boxes.detach()*boxes.new_tensor([width,height,width,height])
        rois=torch.cat([torch.arange(batch,device=boxes.device)[:,None,None].expand(batch,count,1),pixel_boxes],-1).reshape(-1,5)
        area=((pixel_boxes[:,:,2:]-pixel_boxes[:,:,:2]).clamp_min(0).prod(-1)).sqrt()
        levels=(torch.floor(4+torch.log2(area.clamp_min(1)/224))-3).clamp(0,2).long().flatten()
        maps=list(features.values())
        pooled=maps[0].new_zeros(batch*count,maps[0].shape[1],self.roi_size,self.roi_size)
        for level,feature in enumerate(maps):
            ids=torch.where(levels==level)[0]
            if len(ids):
                pooled[ids]=roi_align(feature,rois[ids].float(),self.roi_size,
                    spatial_scale=feature.shape[-1]/width,sampling_ratio=2,aligned=True).to(pooled.dtype)
        return pooled.flatten(2).transpose(1,2)

    def score_branch(self,head,queries,features,text,valid_sizes,image_size):
        boxes=head.predict_boxes(queries,valid_sizes)
        roi=self.pool_rois(features,boxes,image_size)
        refined=self.refinement(roi,boxes.detach().reshape(-1,4)).reshape_as(queries)
        output=head.score(roi.mean(1).reshape_as(queries),refined,queries,text)
        output['boxes']=boxes
        return output

    def forward(self,images,tokens,targets=None,gt_roi=False,features=None,valid_sizes=None,image_indices=None):
        features=self.vision(images) if features is None else features
        if image_indices is not None:
            features=OrderedDict((name,value.index_select(0,image_indices)) for name,value in features.items())
        text,mask,pooled_text=self.text(tokens)
        if valid_sizes is None:
            valid_sizes=torch.stack([target['valid_size'] for target in targets]) if targets is not None else images.new_ones(len(text),2)
        use_aux=self.training and self.aux_enabled
        main,aux=self.localization(features,text,mask,pooled_text,valid_sizes,use_aux)
        output=self.score_branch(self.main_head,main,features,pooled_text,valid_sizes,images.shape[-2:])
        if use_aux:
            output['aux']=self.score_branch(self.aux_head,aux,features,pooled_text,valid_sizes,images.shape[-2:])
        teacher_count=max((len(t['boxes']) for t in targets),default=0) if gt_roi and targets is not None else 0
        if teacher_count:
            truth=main.new_zeros(len(main),teacher_count,4)
            for i,target in enumerate(targets):truth[i,:len(target['boxes'])]=target['boxes']
            roi=self.pool_rois(features,truth,images.shape[-2:])
            refined=self.refinement(roi,truth.reshape(-1,4)).reshape(len(main),teacher_count,-1)
            visual=F.normalize(self.main_head.visual_projection(refined),dim=-1)
            phrase=F.normalize(self.main_head.text_projection(pooled_text),dim=-1)
            logits=(visual*phrase[:,None]).sum(-1)*self.main_head.logit_scale.exp().clamp(max=100)+self.main_head.alignment_bias
            output['gt_alignment_logits']=[logits[i,:len(t['boxes'])] for i,t in enumerate(targets) if len(t['boxes'])]
        return output
