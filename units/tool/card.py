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
class Bert(nn.Module):
    def __init__(
        self,
        local_model_dir=f"{CARD_ROOT}/LightDet/units/model/bert",
        max_cache_size=20000
    ):
        super().__init__()
        local_model_dir = Path(local_model_dir)

        if not local_model_dir.exists():
            raise FileNotFoundError(
                f"找不到本機 BERT 模型資料夾: {local_model_dir}\n"
                f"請先將 tokenizer 與 model 用 save_pretrained() 存到這個路徑。"
            )

        self.tokenizer = BertTokenizerFast.from_pretrained(
            str(local_model_dir),
            local_files_only=True
        )

        self.model = BertModel.from_pretrained(
            str(local_model_dir),
            local_files_only=True
        )

        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.cache = {}
        self.max_cache_size = max_cache_size

    def clear_cache(self):
        self.cache.clear()

    def _encode_uncached(self, texts, device):
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        return {
            "last_hidden_state": outputs.last_hidden_state.detach(),
            "pooler_output": outputs.pooler_output.detach(),
            "attention_mask": inputs["attention_mask"].detach()
        }

    def forward(self, x):
        if isinstance(x, str):
            x = [x]

        x = [str(t).strip() for t in x]

        device = next(self.model.parameters()).device

        missing_texts = []
        missing_indices = []

        for i, text in enumerate(x):
            if text not in self.cache:
                missing_texts.append(text)
                missing_indices.append(i)

        if len(missing_texts) > 0:
            encoded = self._encode_uncached(missing_texts, device)

            for j, text in enumerate(missing_texts):
                if len(self.cache) >= self.max_cache_size:
                    self.cache.clear()

                self.cache[text] = {
                    "last_hidden_state": encoded["last_hidden_state"][j].cpu(),
                    "pooler_output": encoded["pooler_output"][j].cpu(),
                    "attention_mask": encoded["attention_mask"][j].cpu()
                }

        last_hidden_state = []
        pooler_output = []
        attention_mask = []

        for text in x:
            item = self.cache[text]

            last_hidden_state.append(item["last_hidden_state"])
            pooler_output.append(item["pooler_output"])
            attention_mask.append(item["attention_mask"])

        last_hidden_state = torch.stack(last_hidden_state, dim=0).to(device, non_blocking=True)
        pooler_output = torch.stack(pooler_output, dim=0).to(device, non_blocking=True)
        attention_mask = torch.stack(attention_mask, dim=0).to(device, non_blocking=True)

        return {
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooler_output,
            "attention_mask": attention_mask
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


class BackBone(nn.Module):
    def __init__(self, out_channels=1024, target_size=(40, 40)):
        super().__init__()
        self.target_size = target_size
        self.backbone = ResNet50Extractor()

        self.resNet_l2_proj = nn.Sequential(
            nn.Conv2d(512, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.resNet_l4_proj = nn.Sequential(
            nn.Conv2d(2048, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.MBv3 = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )

    def forward(self, x):
        layer_2, layer_4, fc = self.backbone(x)

        layer_2 = self.resNet_l2_proj(layer_2)
        layer_4 = self.resNet_l4_proj(layer_4)

        layer_2 = F.interpolate(layer_2, size=self.target_size, mode="bilinear", align_corners=False)
        layer_4 = F.interpolate(layer_4, size=self.target_size, mode="bilinear", align_corners=False)

        x = torch.cat([layer_4, layer_2], dim=1)
        x = self.MBv3(x)

        return x, fc


if __name__ == "__main__":
    model = Bert(
        # local_model_dir=f"{CARD_ROOT}/LightDet/units/model/bert"
    )
    text = "黃色箱子"

    out = model(text)
    print("output shape:", out.pooler_output) #Bert最終層輸出
   