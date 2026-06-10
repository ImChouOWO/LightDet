import os
import warnings
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.cuda")
warnings.filterwarnings("ignore", category=UserWarning, module=r"huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module=r"transformers")

from transformers import BertTokenizerFast, BertModel
from transformers.utils import logging as transformers_logging
from huggingface_hub import logging as hub_logging
from huggingface_hub.utils import disable_progress_bars
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import time
transformers_logging.set_verbosity_error()
hub_logging.set_verbosity_error()
disable_progress_bars()

CARD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
print("Card Root :",CARD_ROOT)
class FusionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim=512,
        fusion_token_num=16
    ):
        super().__init__()

        self.fusion_tokens = nn.Parameter(
            torch.randn(
                1,
                fusion_token_num,
                hidden_dim
            )
        )

        self.img_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        img_tokens,
        text_tokens
    ):
        """
        img_tokens:
            [B,400,512]

        text_tokens:
            [B,32,512]
        """
        img_global = img_tokens.mean(dim=1)

        fusion_tokens = self.fusion_tokens.expand(img_tokens.shape[0],-1,-1)
        fusion_tokens = (fusion_tokens+self.img_adapter(img_global).unsqueeze(1))

        x = torch.cat(
            [fusion_tokens,img_tokens,text_tokens],dim=1)

        return x

class TransformerBlock(nn.Module):
    def __init__(
        self,
        num_layer=1,
        in_channel=512,
        out_channel=512
    
    ):
        super().__init__()
        self.fuse = FusionBlock(in_channel,out_channel)
    def forward(self,img_token, text_token):
        x = self.fuse(img_token, text_token)
        return x



class Bert(nn.Module):
    def __init__(
        self,
        local_model_dir=f"{CARD_ROOT}/LightDet/units/model/bert",
        out_dim=512,
        max_length=32,
        max_cache_size=20000,
        freeze_bert=True
    ):
        super().__init__()

        local_model_dir = Path(local_model_dir)

        if not local_model_dir.exists():
            raise FileNotFoundError(
                f"找不到本機 BERT 模型資料夾: {local_model_dir}"
            )

        self.tokenizer = BertTokenizerFast.from_pretrained(
            str(local_model_dir),
            local_files_only=True
        )

        self.model = BertModel.from_pretrained(
            str(local_model_dir),
            local_files_only=True
        )

        self.max_length = max_length
        self.max_cache_size = max_cache_size
        self.cache = {}

        bert_dim = self.model.config.hidden_size

        self.proj = nn.Sequential(
            nn.Linear(bert_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

        if freeze_bert:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        else:
            self.model.train()

    def clear_cache(self):
        self.cache.clear()

    def _encode_texts(self, texts, device):
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        bert_trainable = any(
            p.requires_grad for p in self.model.parameters()
        )

        if bert_trainable:
            outputs = self.model(**inputs)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)

        return {
            "last_hidden_state": outputs.last_hidden_state,
            "attention_mask": inputs["attention_mask"]
        }

    def forward(self, texts):
        """
        texts:
            str 或 List[str]

        return:
            text_tokens: [B, L, out_dim]
            text_mask:   [B, L]
        """

        if isinstance(texts, str):
            texts = [texts]

        texts = [
            str(t).strip()
            for t in texts
        ]

        device = next(self.proj.parameters()).device

        # 如果 BERT 是 freeze，才使用 cache
        bert_trainable = any(
            p.requires_grad for p in self.model.parameters()
        )

        if bert_trainable:
            encoded = self._encode_texts(texts, device)

            text_tokens = self.proj(
                encoded["last_hidden_state"]
            )

            return {
                "text_tokens": text_tokens,
                "text_mask": encoded["attention_mask"]
            }

        missing = []
        for text in texts:
            if text not in self.cache:
                missing.append(text)

        if len(missing) > 0:
            encoded = self._encode_texts(missing, device)

            for i, text in enumerate(missing):
                if len(self.cache) >= self.max_cache_size:
                    self.cache.clear()

                self.cache[text] = {
                    "last_hidden_state": encoded["last_hidden_state"][i].detach().cpu(),
                    "attention_mask": encoded["attention_mask"][i].detach().cpu()
                }

        hidden_states = []
        masks = []

        for text in texts:
            item = self.cache[text]

            hidden_states.append(
                item["last_hidden_state"]
            )

            masks.append(
                item["attention_mask"]
            )

        hidden_states = torch.stack(
            hidden_states,
            dim=0
        ).to(device)

        masks = torch.stack(
            masks,
            dim=0
        ).to(device)

        text_tokens = self.proj(hidden_states)

        return {
            "text_tokens": text_tokens,
            "text_mask": masks
        }


class ResNet50Extractor(nn.Module):
    def __init__(self, weights=ResNet50_Weights.DEFAULT):
        super().__init__()
        torch.hub.set_dir(f"{CARD_ROOT}/LightDet/units/model/resnet")
        model = resnet50(weights=weights)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        return_nodes = {
            "layer2": "feat_layer2",
            "layer4": "feat_layer4",
            "fc": "logits"
        }

        self.feature_extractor = create_feature_extractor(
            model,
            return_nodes=return_nodes
        )

    def forward(self, x):
        outputs = self.feature_extractor(x)
        feat_layer2 = outputs["feat_layer2"]
        feat_layer4 = outputs["feat_layer4"]
        fc = outputs["logits"]
        return feat_layer2, feat_layer4, fc

class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        groups=1,
        act=True
    ):
        super().__init__()

        padding = kernel_size // 2

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        ]

        if act:
            layers.append(nn.SiLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MobileNetV4ConvBlock(nn.Module):
    """
    MobileNetV4-style UIB Block.

    設計邏輯：
    1. optional start depthwise conv
    2. pointwise expand
    3. optional middle depthwise conv
    4. pointwise project
    5. residual connection
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expand_ratio=2.0,
        start_dw=True,
        middle_dw=True,
        kernel_size=3,
        use_residual=True
    ):
        super().__init__()

        hidden_channels = int(in_channels * expand_ratio)

        self.use_residual = (
            use_residual
            and stride == 1
            and in_channels == out_channels
        )

        self.start_dw = (
            ConvBNAct(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=in_channels,
                act=True
            )
            if start_dw
            else nn.Identity()
        )

        self.expand = ConvBNAct(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act=True
        )

        self.middle_dw = (
            ConvBNAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=1 if start_dw else stride,
                groups=hidden_channels,
                act=True
            )
            if middle_dw
            else nn.Identity()
        )

        self.project = ConvBNAct(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act=False
        )

    def forward(self, x):
        identity = x

        x = self.start_dw(x)
        x = self.expand(x)
        x = self.middle_dw(x)
        x = self.project(x)

        if self.use_residual:
            x = x + identity

        return x

class ImgProjector(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        out_channels=512,
        layer_num=3,
        expand_ratio=2.0,
        target_size = (40,40)
    ):
        super().__init__()
        self.target_size  = target_size
        self.larger_view = self._make_layers(
            in_channels,
            out_channels,
            layer_num,
            expand_ratio
        )

        self.middle_view = self._make_layers(
            in_channels,
            out_channels,
            layer_num,
            expand_ratio
        )

        self.smaller_view = self._make_layers(
            in_channels,
            out_channels,
            layer_num,
            expand_ratio
        )
        self.resize_view = nn.Sequential(
            nn.LazyConv2d(
                512,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True)
        )

    def _make_layers(
        self,
        in_channels,
        out_channels,
        layer_num,
        expand_ratio
    ):
        layers = []

        for i in range(layer_num):
            layers.append(
                MobileNetV4ConvBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    stride=1,
                    expand_ratio=expand_ratio,
                    start_dw=True,
                    middle_dw=True,
                    kernel_size=3
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        large_x = x

        middle_x = F.interpolate(
            x,
            scale_factor=0.5,
            mode="bilinear",
            align_corners=False
        )

        small_x = F.interpolate(
            x,
            scale_factor=0.25,
            mode="bilinear",
            align_corners=False
        )

        large_feat = self.larger_view(large_x)
        middle_feat = self.middle_view(middle_x)
        small_feat = self.smaller_view(small_x)

        target_size =self.target_size

        large_feat = F.interpolate(
            large_feat,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )
        middle_feat = F.interpolate(
            large_feat,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        small_feat = F.interpolate(
            small_feat,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        large_feat = self.resize_view(large_feat)
        middle_feat = self.resize_view(middle_feat)
        small_feat = self.resize_view(small_feat)
        x = sum([large_feat, middle_feat, small_feat])
        
        return x

class BackBone(nn.Module):
    def __init__(self, in_channels,out_channels=1024, target_size=(40, 40)):
        super().__init__()
        self.backbone = ImgProjector(
            in_channels=in_channels,
            out_channels=out_channels,
            layer_num=3,
            expand_ratio=2.0,
            target_size = target_size
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(2).transpose(1, 2)
        return x


if __name__ == "__main__":
    img_model = BackBone(
        in_channels=1024,
        out_channels=512,
        target_size=(20, 20)
    )

    text_model = Bert(
        out_dim=512,
        max_length=32,
        freeze_bert=True
    )
    transformer = TransformerBlock()
    img = torch.randn(1, 1024, 40, 40)
    texts = ["ship"]
    img_token = img_model(img)
    text_out = text_model(texts)
    text_token = text_out["text_tokens"]
    text_mask = text_out["text_mask"]
    fuse = transformer(img_token,text_token)
    print(fuse.shape)

    
   