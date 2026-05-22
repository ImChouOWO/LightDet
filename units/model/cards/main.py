import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
print("Project Root :", PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)



import torch
import torch.nn as nn

from LightDet.units.tool.roi import build_cell_roi_tensor
from LightDet.units.tool.card import BackBone, Bert
from LightDet.units.model.pipeline.data import build_dataloaders

def get_config(path=f"{PROJECT_ROOT}/LightDet/units/model/cards/config/model.yaml"):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


config = get_config()
model_cfg = config["model"]

# Global Model Scaling Parameters


NUM_LAYER = model_cfg["num_layer"]
DYHEAD_LAYER = model_cfg["dyhead_layer"]
HIDDEN_DIM = model_cfg["hidden_dim"]
NUM_HEADS = model_cfg["num_heads"]
MLP_RATIO = model_cfg["mlp_ratio"]


# IMAGE_GRID_SIZE = 20 means image cells = 20 x 20 = 400
IMAGE_GRID_SIZE = model_cfg["image_grid_size"]

print("[Info]: Model Config: \n", model_cfg)


# Cross Attention Block


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=0.1
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, \
            f"hidden_dim must be divisible by num_heads, got hidden_dim={hidden_dim}, num_heads={num_heads}"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.img_gate = nn.Parameter(torch.tensor(0.0))
        self.txt_gate = nn.Parameter(torch.tensor(0.0))

        # image <- text
        self.norm_img_1 = nn.LayerNorm(hidden_dim)
        self.norm_txt_1 = nn.LayerNorm(hidden_dim)

        self.img_to_txt_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # text <- image
        self.norm_txt_2 = nn.LayerNorm(hidden_dim)
        self.norm_img_2 = nn.LayerNorm(hidden_dim)

        self.txt_to_img_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        ffn_dim = int(hidden_dim * mlp_ratio)

        self.img_ffn_norm = nn.LayerNorm(hidden_dim)
        self.txt_ffn_norm = nn.LayerNorm(hidden_dim)

        self.img_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        self.txt_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, img_x, txt_x, text_mask=None, img_mask=None):
        """
        img_x:
            [B, NUM_IMAGE_CELLS, HIDDEN_DIM]

        txt_x:
            [B, T*L, HIDDEN_DIM]

        text_mask:
            [B, T*L]
            1 = valid token
            0 = padding token

        img_mask:
            [B, NUM_IMAGE_CELLS]
            1 = valid image cell
            0 = invalid / ignored image cell
        """

        img_alpha = torch.sigmoid(self.img_gate)
        txt_alpha = torch.sigmoid(self.txt_gate)

        # -------------------------------------------------
        # image <- text
        # -------------------------------------------------
        txt_norm_1 = self.norm_txt_1(txt_x)

        img_attn_out, img_attn_weights = self.img_to_txt_attn(
            query=self.norm_img_1(img_x),
            key=txt_norm_1,
            value=txt_norm_1,
            key_padding_mask=(text_mask == 0) if text_mask is not None else None
        )

        img_x = img_x + img_alpha * img_attn_out
        img_x = img_x + self.img_ffn(self.img_ffn_norm(img_x))

        # -------------------------------------------------
        # text <- image
        # -------------------------------------------------
        img_norm_2 = self.norm_img_2(img_x)

        txt_attn_out, txt_attn_weights = self.txt_to_img_attn(
            query=self.norm_txt_2(txt_x),
            key=img_norm_2,
            value=img_norm_2,
            key_padding_mask=(img_mask == 0) if img_mask is not None else None
        )

        txt_x = txt_x + txt_alpha * txt_attn_out
        txt_x = txt_x + self.txt_ffn(self.txt_ffn_norm(txt_x))

        return img_x, txt_x, img_attn_weights, txt_attn_weights



# Cross Attention Module


class CrossAtt(nn.Module):
    def __init__(
        self,
        img_dim=2048,
        txt_dim=768,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYER,
        mlp_ratio=MLP_RATIO,
        dropout=0.1
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, \
            f"hidden_dim must be divisible by num_heads, got hidden_dim={hidden_dim}, num_heads={num_heads}"

        self.img_dim = img_dim
        self.txt_dim = txt_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio

        self.q_proj = nn.Linear(img_dim, hidden_dim)
        self.kv_proj = nn.Linear(txt_dim, hidden_dim)

        self.layers = nn.ModuleList([
            CrossAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_text_norm = nn.LayerNorm(hidden_dim)

    def forward(self, fused_feats, text_tokens, text_mask):
        """
        fused_feats:
            [B, NUM_IMAGE_CELLS, 2048]

        text_tokens:
            [T, L, 768]

        text_mask:
            [T, L]

        return:
            x:
                [B, NUM_IMAGE_CELLS, HIDDEN_DIM]

            attn_weights_all:
                {
                    "img_to_text": List[[B, NUM_IMAGE_CELLS, T*L]],
                    "text_to_img": List[[B, T*L, NUM_IMAGE_CELLS]]
                }
        """

        B = fused_feats.shape[0]
        T, L, _ = text_tokens.shape

        # Image feature:
        # [B, NUM_IMAGE_CELLS, 2048] -> [B, NUM_IMAGE_CELLS, HIDDEN_DIM]
        x = self.q_proj(fused_feats)

        # Text feature:
        # [T, L, 768] -> [T*L, 768] -> [B, T*L, 768] -> [B, T*L, HIDDEN_DIM]
        text_tokens = text_tokens.reshape(T * L, -1)
        text_tokens = text_tokens.unsqueeze(0).expand(B, -1, -1)
        kv = self.kv_proj(text_tokens)

        # Text mask:
        # [T, L] -> [T*L] -> [B, T*L]
        text_mask = text_mask.reshape(T * L)
        text_mask = text_mask.unsqueeze(0).expand(B, -1)

        img_attn_weights_all = []
        txt_attn_weights_all = []

        img_skip_features = []
        txt_skip_features = []

        for i, layer in enumerate(self.layers):
            x, kv, img_attn_weights, txt_attn_weights = layer(
                img_x=x,
                txt_x=kv,
                text_mask=text_mask,
                img_mask=None
            )

            # Cross-layer skip:
            # layer0 -> layer2
            # layer1 -> layer3
            # layer2 -> layer4
            # layer3 -> layer5
            if i >= 2:
                x = x + img_skip_features[i - 2]
                kv = kv + txt_skip_features[i - 2]

            img_skip_features.append(x)
            txt_skip_features.append(kv)

            img_attn_weights_all.append(img_attn_weights)
            txt_attn_weights_all.append(txt_attn_weights)

        x = self.final_norm(x)
        kv = self.final_text_norm(kv)

        attn_weights_all = {
            "img_to_text": img_attn_weights_all,
            "text_to_img": txt_attn_weights_all
        }

        return x, attn_weights_all



# Spatial Attention
# Cell-to-cell attention


class SpatialAttn(nn.Module):
    def __init__(
        self,
        dim=HIDDEN_DIM,
        num_heads=NUM_HEADS
    ):
        super().__init__()

        assert dim % num_heads == 0, \
            f"dim must be divisible by num_heads, got dim={dim}, num_heads={num_heads}"

        self.dim = dim
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x:
            [B, NUM_IMAGE_CELLS, HIDDEN_DIM]
        """

        x_norm = self.norm(x)

        out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm
        )

        return x + out



# Channel Attention
# Channel-to-task weighting


class ChannelAttn(nn.Module):
    def __init__(
        self,
        dim=HIDDEN_DIM,
        reduction=4
    ):
        super().__init__()

        assert dim % reduction == 0, \
            f"dim must be divisible by reduction, got dim={dim}, reduction={reduction}"

        self.dim = dim
        self.reduction = reduction

        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x:
            [B, NUM_IMAGE_CELLS, HIDDEN_DIM]
        """

        # Global pooling over image cells:
        # [B, NUM_IMAGE_CELLS, HIDDEN_DIM] -> [B, HIDDEN_DIM]
        w = x.mean(dim=1)

        # [B, HIDDEN_DIM] -> [B, HIDDEN_DIM]
        w = self.fc(w)

        # [B, NUM_IMAGE_CELLS, HIDDEN_DIM]
        return x * w.unsqueeze(1)



# Dynamic Head Block


class DynamicHeadBlock(nn.Module):
    def __init__(
        self,
        dim=HIDDEN_DIM,
        num_heads=NUM_HEADS
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        self.spatial = SpatialAttn(
            dim=dim,
            num_heads=num_heads
        )

        self.channel = ChannelAttn(
            dim=dim
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x:
            [B, NUM_IMAGE_CELLS, HIDDEN_DIM]
        """

        x = self.spatial(x)
        x = self.channel(x)
        x = self.norm(x)

        return x



# Vision-Text Model
# Backbone + BERT + Cross Attention


class VisionTextModel(nn.Module):
    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYER,
        mlp_ratio=MLP_RATIO,
        image_grid_size=IMAGE_GRID_SIZE
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio

        self.image_grid_size = image_grid_size
        self.num_image_cells = image_grid_size * image_grid_size

        self.backbone = BackBone(
            out_channels=1024,
            target_size=(self.image_grid_size, self.image_grid_size)
        )

        self.text_encoder = Bert(
            local_model_dir=f"{PROJECT_ROOT}/LightDet/units/model/bert"
        )

        self.cAtt = CrossAtt(
            img_dim=2048,
            txt_dim=768,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio
        )

    def forward(self, images, boxes_per_image, texts, image_size=(640, 640)):
        """
        images:
            [B, C, H, W]

        boxes_per_image:
            list of tensors

        texts:
            list[str] or tokenizer-compatible input

        image_size:
            original input image size
        """

        feat_map, fc = self.backbone(images)

        cell_feats, roi_feats, roi_mask, roi_mean, fused_feats = build_cell_roi_tensor(
            feat_map=feat_map,
            boxes_per_image=boxes_per_image,
            image_size=image_size,
            roi_out_size=1,
            max_rois_per_cell=12
        )

        assert fused_feats.dim() == 3, \
            f"Expected fused_feats to be [B, N, C], but got shape={tuple(fused_feats.shape)}"

        assert fused_feats.shape[-1] == 2048, \
            f"Expected fused_feats last dim=2048, but got {fused_feats.shape[-1]}"

        actual_num_cells = fused_feats.shape[1]

        if actual_num_cells != self.num_image_cells:
            print(
                f"[Warning] Config image_grid_size={self.image_grid_size}, "
                f"expected cells={self.num_image_cells}, "
                f"but actual fused_feats cells={actual_num_cells}. "
                f"Using actual fused_feats cells for attention."
            )

        text_outputs = self.text_encoder(texts)

        text_tokens = text_outputs["last_hidden_state"]
        text_global = text_outputs["pooler_output"]
        atten_mask = text_outputs["attention_mask"]

        att_out, att_weights = self.cAtt(
            fused_feats=fused_feats,
            text_tokens=text_tokens,
            text_mask=atten_mask
        )

        return {
            "feat_map": feat_map,
            "fc": fc,
            "cell_feats": cell_feats,
            "roi_feats": roi_feats,
            "roi_mask": roi_mask,
            "roi_mean": roi_mean,
            "fused_feats": fused_feats,
            "text_tokens": text_tokens,
            "text_global": text_global,
            "att_out": att_out,
            "att_weights": att_weights,
            "actual_num_cells": actual_num_cells
        }



# Full Model
# VisionTextModel + DynamicHead + Prediction Heads


class Model(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYER,
        dyhead_layers=DYHEAD_LAYER,
        mlp_ratio=MLP_RATIO,
        image_grid_size=IMAGE_GRID_SIZE
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, \
            f"hidden_dim must be divisible by num_heads, got hidden_dim={hidden_dim}, num_heads={num_heads}"

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dyhead_layers = dyhead_layers
        self.mlp_ratio = mlp_ratio

        self.image_grid_size = image_grid_size
        self.num_image_cells = image_grid_size * image_grid_size

        self.vis_text_model = VisionTextModel(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            image_grid_size=image_grid_size
        )

        self.dyhead = nn.ModuleList([
            DynamicHeadBlock(
                dim=hidden_dim,
                num_heads=num_heads
            )
            for _ in range(dyhead_layers)
        ])

        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

        self.text_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

        self.score_head = nn.Linear(hidden_dim, 1)

        # BERT 768 -> attention hidden dim
        self.text_proj = nn.Linear(768, hidden_dim)

    def forward(self, images, boxes_per_image, texts, image_size=(640, 640)):
        outputs = self.vis_text_model(
            images=images,
            boxes_per_image=boxes_per_image,
            texts=texts,
            image_size=image_size
        )

        # [B, NUM_IMAGE_CELLS, HIDDEN_DIM]
        x = outputs["att_out"]

        for layer in self.dyhead:
            x = layer(x)

        # [B, NUM_IMAGE_CELLS, 4]
        bbox = self.bbox_head(x).sigmoid()

        # [B, NUM_IMAGE_CELLS, num_classes]
        text = self.text_head(x)

        # [B, NUM_IMAGE_CELLS, 1]
        score = self.score_head(x)

        # text_global: [T, 768]
        # text_feat:   [T, HIDDEN_DIM]
        text_feat = self.text_proj(outputs["text_global"])

        outputs.update({
            "bbox": bbox,
            "text_pred": text,
            "score": score,
            "text_feat": text_feat
        })

        return outputs



# Utility: Count Parameters


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable
    }



# Main Test


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    train_image_paths = [
        "/home/soic/Desktop/LightDet/datasets/images/train/2021_10_20_13_28_41_00400.jpg",
        "/home/soic/Desktop/LightDet/datasets/images/train/2021_10_20_13_28_41_00200.jpg",
    ]

    train_anno_paths = [
        "/home/soic/Desktop/LightDet/datasets/labels/train/2021_10_20_13_28_41_00400.json",
        "/home/soic/Desktop/LightDet/datasets/labels/train/2021_10_20_13_28_41_00200.json",
    ]

    val_image_paths = train_image_paths
    val_anno_paths = train_anno_paths

    train_loader, _ = build_dataloaders(
        train_image_paths=train_image_paths,
        train_anno_paths=train_anno_paths,
        val_image_paths=val_image_paths,
        val_anno_paths=val_anno_paths,
        batch_size=2,
        image_size=(640, 640),
        num_workers=0
    )

    model = Model(
        num_classes=1000,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYER,
        dyhead_layers=DYHEAD_LAYER,
        mlp_ratio=MLP_RATIO,
        image_grid_size=IMAGE_GRID_SIZE
    ).to(device)

    model.eval()

    param_info = count_parameters(model)
    print("Total Parameters    :", param_info["total"])
    print("Trainable Parameters:", param_info["trainable"])

    with torch.no_grad():
        for batch in train_loader:
            images = batch["images"].to(device)
            boxes_per_image = [b.to(device) for b in batch["boxes_per_image"]]

            texts = batch["flat_texts"]

            outputs = model(
                images=images,
                boxes_per_image=boxes_per_image,
                texts=texts,
                image_size=(640, 640)
            )

            print("feat_map     :", outputs["feat_map"].shape)
            print("cell_feats   :", outputs["cell_feats"].shape)
            print("roi_feats    :", outputs["roi_feats"].shape)
            print("roi_mask     :", outputs["roi_mask"].shape)
            print("roi_mean     :", outputs["roi_mean"].shape)
            print("fused_feats  :", outputs["fused_feats"].shape)
            print("text_tokens  :", outputs["text_tokens"].shape)
            print("text_global  :", outputs["text_global"].shape)

            print("att_out      :", outputs["att_out"].shape)

            print("bbox         :", outputs["bbox"].shape)
            print("text_pred    :", outputs["text_pred"].shape)
            print("score        :", outputs["score"].shape)
            print("text_feat    :", outputs["text_feat"].shape)

            print("img_to_text layers:", len(outputs["att_weights"]["img_to_text"]))
            for i, w in enumerate(outputs["att_weights"]["img_to_text"]):
                print(f"img_to_text[{i}]:", w.shape)

            print("text_to_img layers:", len(outputs["att_weights"]["text_to_img"]))
            for i, w in enumerate(outputs["att_weights"]["text_to_img"]):
                print(f"text_to_img[{i}]:", w.shape)

            break