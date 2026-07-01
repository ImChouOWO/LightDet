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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import BertModel, BertTokenizerFast
from transformers.utils import logging as transformers_logging
from huggingface_hub import logging as hub_logging
from huggingface_hub.utils import disable_progress_bars

transformers_logging.set_verbosity_error()
hub_logging.set_verbosity_error()
disable_progress_bars()

CARD_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../",
    )
)
print("Card Root :", CARD_ROOT)


def make_group_norm(
    channels: int,
    max_groups: int = 32,
) -> nn.GroupNorm:
    num_groups = min(max_groups, channels)

    while channels % num_groups != 0:
        num_groups -= 1

    return nn.GroupNorm(
        num_groups=num_groups,
        num_channels=channels,
    )


class FusionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        fusion_token_num=16,
    ):
        super().__init__()

        self.fusion_tokens = nn.Parameter(
            torch.randn(
                1,
                fusion_token_num,
                hidden_dim,
            )
        )

        self.img_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        img_tokens,
        text_tokens,
        text_mask,
    ):
        """
        img_tokens:
            [B, num_image, hidden_dim]

        text_tokens:
            [B, num_text, hidden_dim]
        """
        img_global = img_tokens.mean(dim=1)

        fusion_tokens = self.fusion_tokens.expand(
            img_tokens.shape[0],
            -1,
            -1,
        )
        fusion_tokens = (
            fusion_tokens
            + self.img_adapter(
                img_global
            ).unsqueeze(1)
        )

        x = torch.cat(
            [
                fusion_tokens,
                img_tokens,
                text_tokens,
            ],
            dim=1,
        )

        if text_mask is not None:
            fusion_mask = torch.ones(
                img_tokens.shape[0],
                fusion_tokens.shape[1],
                device=img_tokens.device,
                dtype=text_mask.dtype,
            )

            img_mask = torch.ones(
                img_tokens.shape[0],
                img_tokens.shape[1],
                device=img_tokens.device,
                dtype=text_mask.dtype,
            )

            attention_mask = torch.cat(
                [
                    fusion_mask,
                    img_mask,
                    text_mask,
                ],
                dim=1,
            )
        else:
            attention_mask = None

        return x, attention_mask


class TransformerBlock(nn.Module):
    """
    Shared Transformer with isolated Main/Aux query groups.

    Isolation is implemented by stacking the two groups on the batch
    dimension before a single TransformerEncoder call:

        [B, T, C] main
        [B, T, C] aux
            -> cat(dim=0)
        [2B, T, C]

    Self-attention never crosses batch elements, so Main and Aux cannot read
    each other's token states. The Transformer weights remain shared and the
    two groups are still processed in one batched GPU call.
    """

    MAIN_GROUP = 0
    AUX_GROUP = 1

    def __init__(
        self,
        hidden_dim=512,
        fusion_token_num=16,
        num_heads=8,
        num_layers=1,
        mlp_ratio=4.0,
        dropout=0.1,
        query_group_init_std=0.02,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)

        self.fuse = FusionBlock(
            hidden_dim=hidden_dim,
            fusion_token_num=fusion_token_num,
        )

        # Distinguishes the duplicated prediction-token groups while keeping
        # all Transformer weights shared. Main starts as the legacy path;
        # Aux receives a small learned offset to break symmetry.
        self.query_group_embeddings = nn.Parameter(
            torch.zeros(
                2,
                1,
                self.hidden_dim,
            )
        )
        if float(query_group_init_std) > 0.0:
            nn.init.normal_(
                self.query_group_embeddings.data[
                    self.AUX_GROUP
                ],
                mean=0.0,
                std=float(query_group_init_std),
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(
                hidden_dim * mlp_ratio
            ),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

    @staticmethod
    def _build_padding_mask(
        mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        return torch.eq(mask, 0)

    def _add_group_embedding(
        self,
        x: torch.Tensor,
        group_index: int,
    ) -> torch.Tensor:
        return (
            x
            + self.query_group_embeddings[
                int(group_index)
            ].to(
                device=x.device,
                dtype=x.dtype,
            )
        )

    def forward(
        self,
        img_token,
        text_token,
        text_mask,
        return_aux=False,
    ):
        """
        Returns:
            main_out: [B, T, C]
            aux_out : [B, T, C] or None
        """
        base_x, mask = self.fuse(
            img_token,
            text_token,
            text_mask,
        )
        padding_mask = self._build_padding_mask(
            mask
        )

        main_x = self._add_group_embedding(
            base_x,
            self.MAIN_GROUP,
        )

        if not bool(return_aux):
            main_out = self.transformer(
                main_x,
                src_key_padding_mask=padding_mask,
            )
            return self.norm(main_out), None

        aux_x = self._add_group_embedding(
            base_x,
            self.AUX_GROUP,
        )

        # Batch-axis grouping gives exact query isolation without creating a
        # 2T x 2T attention matrix. This is more efficient than concatenating
        # both groups along the token dimension with a block-diagonal mask.
        grouped_x = torch.cat(
            [main_x, aux_x],
            dim=0,
        )

        if padding_mask is None:
            grouped_padding_mask = None
        else:
            grouped_padding_mask = torch.cat(
                [padding_mask, padding_mask],
                dim=0,
            )

        grouped_out = self.transformer(
            grouped_x,
            src_key_padding_mask=(
                grouped_padding_mask
            ),
        )
        grouped_out = self.norm(
            grouped_out
        )

        batch_size = int(base_x.shape[0])
        main_out = grouped_out[:batch_size]
        aux_out = grouped_out[batch_size:]

        return main_out, aux_out


class Bert(nn.Module):
    def __init__(
        self,
        local_model_dir=(
            f"{CARD_ROOT}/LightDet/units/model/bert"
        ),
        out_dim=512,
        max_length=32,
        max_cache_size=20000,
        freeze_bert=True,
        precomputed_bert_path=None,
    ):
        super().__init__()

        local_model_dir = Path(
            local_model_dir
        )

        if not local_model_dir.exists():
            raise FileNotFoundError(
                "找不到本機 BERT 模型資料夾: "
                f"{local_model_dir}"
            )

        self.tokenizer = (
            BertTokenizerFast.from_pretrained(
                str(local_model_dir),
                local_files_only=True,
            )
        )

        self.model = BertModel.from_pretrained(
            str(local_model_dir),
            local_files_only=True,
        )

        self.max_length = max_length
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.precomputed_bert_path = (
            precomputed_bert_path
        )
        self.precomputed_cache = None

        bert_dim = (
            self.model.config.hidden_size
        )

        self.proj = nn.Sequential(
            nn.Linear(
                bert_dim,
                out_dim,
            ),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

        if freeze_bert:
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        else:
            self.model.train()

        if (
            precomputed_bert_path is not None
            and os.path.exists(
                precomputed_bert_path
            )
        ):
            try:
                obj = torch.load(
                    precomputed_bert_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                obj = torch.load(
                    precomputed_bert_path,
                    map_location="cpu",
                )

            self.precomputed_cache = obj[
                "cache"
            ]

            print(
                "[BERT] Loaded precomputed "
                "raw cache: "
                f"{len(self.precomputed_cache)} "
                "texts"
            )

    def clear_cache(self):
        self.cache.clear()

    @staticmethod
    def _normalize_texts(texts):
        if isinstance(texts, str):
            texts = [texts]

        return [
            str(text).strip()
            for text in texts
        ]

    def _encode_texts(
        self,
        texts,
        device,
    ):
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        bert_trainable = any(
            parameter.requires_grad
            for parameter
            in self.model.parameters()
        )

        if bert_trainable:
            outputs = self.model(
                **inputs
            )
        else:
            with torch.no_grad():
                outputs = self.model(
                    **inputs
                )

        return {
            "last_hidden_state": (
                outputs.last_hidden_state
            ),
            "attention_mask": (
                inputs["attention_mask"]
            ),
        }

    @torch.no_grad()
    def encode_raw(
        self,
        texts,
        device=None,
    ):
        texts = self._normalize_texts(
            texts
        )

        if device is None:
            device = next(
                self.model.parameters()
            ).device

        encoded = self._encode_texts(
            texts,
            device,
        )

        return {
            "last_hidden_state": (
                encoded[
                    "last_hidden_state"
                ]
            ),
            "attention_mask": (
                encoded[
                    "attention_mask"
                ]
            ),
        }

    def forward(
        self,
        texts,
    ):
        texts = self._normalize_texts(
            texts
        )

        device = next(
            self.proj.parameters()
        ).device

        bert_trainable = any(
            parameter.requires_grad
            for parameter
            in self.model.parameters()
        )

        if bert_trainable:
            encoded = self._encode_texts(
                texts,
                device,
            )

            text_tokens = self.proj(
                encoded[
                    "last_hidden_state"
                ]
            )

            return {
                "text_tokens": text_tokens,
                "text_mask": encoded[
                    "attention_mask"
                ],
            }

        missing = []

        for text in texts:
            in_precomputed = (
                self.precomputed_cache
                is not None
                and text
                in self.precomputed_cache
            )

            in_runtime_cache = (
                text in self.cache
            )

            if (
                not in_precomputed
                and not in_runtime_cache
            ):
                missing.append(text)

        if missing:
            encoded = self._encode_texts(
                missing,
                device,
            )

            for index, text in enumerate(
                missing
            ):
                if (
                    len(self.cache)
                    >= self.max_cache_size
                ):
                    self.cache.clear()

                self.cache[text] = {
                    "last_hidden_state": (
                        encoded[
                            "last_hidden_state"
                        ][index]
                        .detach()
                        .cpu()
                    ),
                    "attention_mask": (
                        encoded[
                            "attention_mask"
                        ][index]
                        .detach()
                        .cpu()
                    ),
                }

        hidden_states = []
        masks = []

        for text in texts:
            if (
                self.precomputed_cache
                is not None
                and text
                in self.precomputed_cache
            ):
                item = (
                    self.precomputed_cache[
                        text
                    ]
                )
            else:
                item = self.cache[text]

            hidden_states.append(
                item[
                    "last_hidden_state"
                ]
            )
            masks.append(
                item["attention_mask"]
            )

        hidden_states = torch.stack(
            hidden_states,
            dim=0,
        ).to(
            device=device,
            dtype=torch.float32,
        )

        masks = torch.stack(
            masks,
            dim=0,
        ).to(device)

        text_tokens = self.proj(
            hidden_states
        )

        return {
            "text_tokens": text_tokens,
            "text_mask": masks,
        }


class ResNet50Extractor(nn.Module):
    def __init__(
        self,
        weights=ResNet50_Weights.DEFAULT,
    ):
        super().__init__()

        torch.hub.set_dir(
            f"{CARD_ROOT}/LightDet/"
            "units/model/resnet"
        )

        model = resnet50(
            weights=weights
        )
        model.eval()

        for parameter in model.parameters():
            parameter.requires_grad_(
                False
            )

        return_nodes = {
            "layer2": "feat_layer2",
            "layer4": "feat_layer4",
            "fc": "logits",
        }

        self.feature_extractor = (
            create_feature_extractor(
                model,
                return_nodes=return_nodes,
            )
        )

    def forward(
        self,
        x,
    ):
        outputs = (
            self.feature_extractor(x)
        )

        return (
            outputs["feat_layer2"],
            outputs["feat_layer4"],
            outputs["logits"],
        )


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        groups=1,
        act=True,
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
                bias=False,
            ),
            make_group_norm(
                out_channels
            ),
        ]

        if act:
            layers.append(
                nn.SiLU(
                    inplace=True
                )
            )

        self.block = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x,
    ):
        return self.block(x)


class MobileNetV4ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expand_ratio=2.0,
        start_dw=True,
        middle_dw=True,
        kernel_size=3,
        use_residual=True,
    ):
        super().__init__()

        hidden_channels = int(
            in_channels * expand_ratio
        )

        self.use_residual = (
            use_residual
            and stride == 1
            and in_channels
            == out_channels
        )

        self.start_dw = (
            ConvGNAct(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=in_channels,
                act=True,
            )
            if start_dw
            else nn.Identity()
        )

        self.expand = ConvGNAct(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act=True,
        )

        self.middle_dw = (
            ConvGNAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=(
                    1
                    if start_dw
                    else stride
                ),
                groups=hidden_channels,
                act=True,
            )
            if middle_dw
            else nn.Identity()
        )

        self.project = ConvGNAct(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act=False,
        )

    def forward(
        self,
        x,
    ):
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
        target_size=(40, 40),
    ):
        super().__init__()

        self.target_size = (
            target_size
        )

        self.larger_view = (
            self._make_layers(
                in_channels,
                out_channels,
                layer_num,
                expand_ratio,
            )
        )

        self.middle_view = (
            self._make_layers(
                in_channels,
                out_channels,
                layer_num,
                expand_ratio,
            )
        )

        self.smaller_view = (
            self._make_layers(
                in_channels,
                out_channels,
                layer_num,
                expand_ratio,
            )
        )

        self.resize_view = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            make_group_norm(
                out_channels
            ),
            nn.SiLU(
                inplace=True
            ),
        )

    @staticmethod
    def _make_layers(
        in_channels,
        out_channels,
        layer_num,
        expand_ratio,
    ):
        layers = []

        for index in range(
            layer_num
        ):
            layers.append(
                MobileNetV4ConvBlock(
                    in_channels=(
                        in_channels
                        if index == 0
                        else out_channels
                    ),
                    out_channels=(
                        out_channels
                    ),
                    stride=1,
                    expand_ratio=(
                        expand_ratio
                    ),
                    start_dw=True,
                    middle_dw=True,
                    kernel_size=3,
                )
            )

        return nn.Sequential(
            *layers
        )

    def forward(
        self,
        x,
    ):
        large_x = x

        middle_x = F.interpolate(
            x,
            scale_factor=0.5,
            mode="bilinear",
            align_corners=False,
        )

        small_x = F.interpolate(
            x,
            scale_factor=0.25,
            mode="bilinear",
            align_corners=False,
        )

        large_feat = self.larger_view(
            large_x
        )
        middle_feat = (
            self.middle_view(
                middle_x
            )
        )
        small_feat = self.smaller_view(
            small_x
        )

        target_size = self.target_size

        large_feat = F.interpolate(
            large_feat,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        middle_feat = F.interpolate(
            middle_feat,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        small_feat = F.interpolate(
            small_feat,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        large_feat = self.resize_view(
            large_feat
        )
        middle_feat = self.resize_view(
            middle_feat
        )
        small_feat = self.resize_view(
            small_feat
        )

        return (
            large_feat
            + middle_feat
            + small_feat
        )


class BackBone(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=1024,
        target_size=(40, 40),
    ):
        super().__init__()

        self.backbone = ImgProjector(
            in_channels=in_channels,
            out_channels=out_channels,
            layer_num=3,
            expand_ratio=2.0,
            target_size=target_size,
        )

    def forward(
        self,
        x,
    ):
        x = self.backbone(x)
        return x.flatten(
            2
        ).transpose(
            1,
            2,
        )


class BottleNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1024,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            ConvGNAct(
                in_channels,
                64,
                kernel_size=3,
                stride=2,
            ),
            ConvGNAct(
                64,
                128,
                kernel_size=3,
                stride=2,
            ),
            ConvGNAct(
                128,
                256,
                kernel_size=3,
                stride=2,
            ),
            ConvGNAct(
                256,
                512,
                kernel_size=3,
                stride=2,
            ),
            ConvGNAct(
                512,
                out_channels,
                kernel_size=1,
                stride=1,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.stem(x)


class DenseHead(nn.Module):
    """
    Single prediction branch.

    The main one-to-one branch and auxiliary one-to-many branch use separate
    DenseHead instances. Their bbox and score parameters are not shared.
    """

    def __init__(
        self,
        hidden_dim=512,
        num_fusion=16,
        num_image=400,
        num_text=32,
        alpha=0.4,
    ):
        super().__init__()

        self.num_fusion = int(
            num_fusion
        )
        self.num_image = int(
            num_image
        )
        self.num_text = int(
            num_text
        )

        if self.num_fusion < 0:
            raise ValueError(
                "num_fusion must be >= 0, "
                f"got {self.num_fusion}"
            )

        if self.num_image <= 0:
            raise ValueError(
                "num_image must be > 0, "
                f"got {self.num_image}"
            )

        if self.num_text < 0:
            raise ValueError(
                "num_text must be >= 0, "
                f"got {self.num_text}"
            )

        self.bbox_head = nn.Sequential(
            nn.LayerNorm(
                hidden_dim
            ),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                4,
            ),
        )

        self.score_head = nn.Sequential(
            nn.LayerNorm(
                hidden_dim
            ),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    @staticmethod
    def _decode_xyxy(
        bbox_raw: torch.Tensor,
    ) -> torch.Tensor:
        bbox_raw = (
            bbox_raw.sigmoid()
        )

        x1 = torch.minimum(
            bbox_raw[..., 0],
            bbox_raw[..., 2],
        )
        y1 = torch.minimum(
            bbox_raw[..., 1],
            bbox_raw[..., 3],
        )
        x2 = torch.maximum(
            bbox_raw[..., 0],
            bbox_raw[..., 2],
        )
        y2 = torch.maximum(
            bbox_raw[..., 1],
            bbox_raw[..., 3],
        )

        return torch.stack(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            dim=-1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        if x.ndim != 3:
            raise ValueError(
                "DenseHead input must have "
                "shape [B, T, C], got "
                f"{tuple(x.shape)}"
            )

        image_start = (
            self.num_fusion
        )
        image_end = (
            image_start
            + self.num_image
        )

        if (
            int(x.shape[1])
            < image_end
        ):
            raise ValueError(
                "DenseHead input token "
                "count is too small: "
                f"tokens={x.shape[1]}, "
                f"required={image_end}"
            )

        img_tokens = x[
            :,
            image_start:image_end,
            :,
        ]

        bbox_raw = self.bbox_head(
            img_tokens
        )
        bbox = self._decode_xyxy(
            bbox_raw
        )

        score_logit = (
            self.score_head(
                img_tokens
            )
        )

        return bbox, score_logit


class VisionTextModel(nn.Module):
    """
    H-DETR-style query-isolated LightDet model.

    Shared:
        BottleNet / image projector / BERT / Transformer weights

    Main query group:
        Isolated Transformer sequence
        self.head
        One-to-one loss
        Used for validation and inference

    Auxiliary query group:
        Isolated Transformer sequence
        self.aux_head
        Repeated-GT one-to-many loss
        Used during training only by default

    The two groups are stacked along the Transformer batch dimension. They
    share weights and source features but cannot exchange self-attention.
    """

    def __init__(
        self,
        img_in_channels=1024,
        hidden_dim=512,
        target_size=(20, 20),
        text_max_length=32,
        fusion_token_num=16,
        num_heads=8,
        num_layers=1,
        mlp_ratio=4.0,
        dropout=0.1,
        freeze_bert=True,
        precomputed_bert_path=None,
        use_auxiliary_head=True,
        auxiliary_in_eval=False,
        initialize_aux_from_main=True,
        query_group_init_std=0.02,
    ):
        super().__init__()

        if len(target_size) != 2:
            raise ValueError(
                "target_size must contain "
                "two dimensions, got "
                f"{target_size}"
            )

        self.hidden_dim = int(
            hidden_dim
        )
        self.target_size = (
            int(target_size[0]),
            int(target_size[1]),
        )
        self.fusion_token_num = int(
            fusion_token_num
        )

        self.num_image = (
            self.target_size[0]
            * self.target_size[1]
        )
        self.num_text = int(
            text_max_length
        )

        self.use_auxiliary_head = bool(
            use_auxiliary_head
        )
        self.auxiliary_in_eval = bool(
            auxiliary_in_eval
        )

        self.bottle_net = BottleNet(
            in_channels=3,
            out_channels=(
                img_in_channels
            ),
        )

        self.img_model = BackBone(
            in_channels=(
                img_in_channels
            ),
            out_channels=hidden_dim,
            target_size=(
                self.target_size
            ),
        )

        self.text_model = Bert(
            out_dim=hidden_dim,
            max_length=(
                text_max_length
            ),
            freeze_bert=(
                freeze_bert
            ),
            precomputed_bert_path=(
                precomputed_bert_path
            ),
        )

        self.transformer = (
            TransformerBlock(
                hidden_dim=hidden_dim,
                fusion_token_num=(
                    fusion_token_num
                ),
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                query_group_init_std=(
                    query_group_init_std
                ),
            )
        )

        # Main one-to-one branch.
        # Keep the original "head.*" namespace for checkpoint compatibility.
        self.head = DenseHead(
            hidden_dim=hidden_dim,
            num_fusion=(
                fusion_token_num
            ),
            num_image=self.num_image,
            num_text=self.num_text,
        )

        # Auxiliary one-to-many branch.
        self.aux_head = (
            DenseHead(
                hidden_dim=hidden_dim,
                num_fusion=(
                    fusion_token_num
                ),
                num_image=(
                    self.num_image
                ),
                num_text=self.num_text,
            )
            if self.use_auxiliary_head
            else None
        )

        if (
            self.aux_head
            is not None
            and bool(
                initialize_aux_from_main
            )
        ):
            self.initialize_auxiliary_head_from_main()

    @property
    def main_head(self):
        return self.head

    @torch.no_grad()
    def initialize_auxiliary_head_from_main(
        self,
    ) -> None:
        if self.aux_head is None:
            return

        self.aux_head.load_state_dict(
            self.head.state_dict(),
            strict=True,
        )

    def set_auxiliary_in_eval(
        self,
        enabled: bool,
    ) -> None:
        self.auxiliary_in_eval = bool(
            enabled
        )

    def _resolve_return_aux(
        self,
        return_aux,
    ) -> bool:
        if (
            not self.use_auxiliary_head
            or self.aux_head is None
        ):
            return False

        if return_aux is not None:
            return bool(
                return_aux
            )

        if self.training:
            return True

        return bool(
            self.auxiliary_in_eval
        )

    def _prepare_legacy_state_dict(
        self,
        state_dict,
    ):
        prepared = state_dict.copy()

        # Old checkpoints do not contain the query-group embedding because
        # Main/Aux previously consumed the same Transformer output.
        group_key = (
            "transformer."
            "query_group_embeddings"
        )
        if group_key not in prepared:
            prepared[group_key] = (
                self.transformer
                .query_group_embeddings
                .detach()
                .clone()
            )

        if self.aux_head is None:
            return {
                key: value
                for key, value
                in prepared.items()
                if not str(key).startswith(
                    "aux_head."
                )
            }

        has_auxiliary_weights = any(
            str(key).startswith(
                "aux_head."
            )
            for key
            in prepared.keys()
        )

        if not has_auxiliary_weights:
            main_prefix = "head."
            aux_prefix = "aux_head."

            for key, value in list(
                prepared.items()
            ):
                key_text = str(key)

                if not key_text.startswith(
                    main_prefix
                ):
                    continue

                suffix = key_text[
                    len(main_prefix):
                ]

                aux_key = (
                    aux_prefix
                    + suffix
                )

                if aux_key not in prepared:
                    prepared[aux_key] = (
                        value
                    )

        return prepared

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        assign=False,
    ):
        prepared = (
            self._prepare_legacy_state_dict(
                state_dict
            )
        )

        try:
            return (
                super().load_state_dict(
                    prepared,
                    strict=strict,
                    assign=assign,
                )
            )
        except TypeError:
            return (
                super().load_state_dict(
                    prepared,
                    strict=strict,
                )
            )

    def forward(
        self,
        img,
        texts,
        image_indices=None,
        return_aux=None,
    ):
        img = self.bottle_net(
            img
        )
        img_token = self.img_model(
            img
        )

        if image_indices is not None:
            if not torch.is_tensor(
                image_indices
            ):
                image_indices = (
                    torch.tensor(
                        image_indices,
                        dtype=torch.long,
                        device=(
                            img_token.device
                        ),
                    )
                )
            else:
                image_indices = (
                    image_indices.to(
                        device=(
                            img_token.device
                        ),
                        dtype=torch.long,
                        non_blocking=True,
                    )
                )

            image_indices = (
                image_indices.reshape(-1)
            )

            if (
                image_indices.numel()
                > 0
            ):
                minimum_index = int(
                    image_indices.min().item()
                )
                maximum_index = int(
                    image_indices.max().item()
                )

                if minimum_index < 0:
                    raise IndexError(
                        "image_indices contains "
                        "a negative index: "
                        f"{minimum_index}"
                    )

                if (
                    maximum_index
                    >= int(
                        img_token.shape[0]
                    )
                ):
                    raise IndexError(
                        "image_indices exceeds "
                        "image batch: "
                        f"max_index="
                        f"{maximum_index}, "
                        f"image_batch="
                        f"{img_token.shape[0]}"
                    )

            img_token = (
                img_token.index_select(
                    0,
                    image_indices,
                )
            )

        text_out = self.text_model(
            texts
        )
        text_token = text_out[
            "text_tokens"
        ]
        text_mask = text_out[
            "text_mask"
        ]

        if (
            int(img_token.shape[0])
            != int(
                text_token.shape[0]
            )
        ):
            raise ValueError(
                "Image-query batch size "
                "mismatch: "
                f"image={img_token.shape[0]}, "
                f"text={text_token.shape[0]}"
            )

        compute_auxiliary = (
            self._resolve_return_aux(
                return_aux
            )
        )

        (
            main_transformer_out,
            aux_transformer_out,
        ) = self.transformer(
            img_token,
            text_token,
            text_mask,
            return_aux=compute_auxiliary,
        )

        (
            main_bbox,
            main_score_logit,
        ) = self.head(
            main_transformer_out
        )

        aux_bbox = None
        aux_score_logit = None

        if compute_auxiliary:
            if aux_transformer_out is None:
                raise RuntimeError(
                    "Auxiliary output was requested "
                    "but Transformer returned None."
                )

            (
                aux_bbox,
                aux_score_logit,
            ) = self.aux_head(
                aux_transformer_out
            )

        return {
            # Backward-compatible main aliases.
            "bbox": main_bbox,
            "score_logit": (
                main_score_logit
            ),

            # Explicit hybrid names.
            "main_bbox": main_bbox,
            "main_score_logit": (
                main_score_logit
            ),
            "aux_bbox": aux_bbox,
            "aux_score_logit": (
                aux_score_logit
            ),
            "aux_computed": bool(
                compute_auxiliary
            ),

            # Backward-compatible diagnostic alias points to Main.
            "transformer_out": (
                main_transformer_out
            ),

            # Explicit isolated-group diagnostics.
            "main_transformer_out": (
                main_transformer_out
            ),
            "aux_transformer_out": (
                aux_transformer_out
            ),
            "query_groups_isolated": True,
            "query_group_batching": (
                "batch_axis"
            ),

            "img_token": img_token,
            "text_token": text_token,
            "text_mask": text_mask,
        }


if __name__ == "__main__":
    model = VisionTextModel(
        img_in_channels=1024,
        hidden_dim=512,
        target_size=(10, 10),
        text_max_length=32,
        fusion_token_num=16,
        num_heads=8,
        num_layers=1,
        use_auxiliary_head=True,
        auxiliary_in_eval=False,
    )

    img = torch.randn(
        1,
        3,
        640,
        640,
    )
    texts = ["ship"]

    model.train()

    with torch.no_grad():
        train_out = model(
            img,
            texts,
        )

    print("[Train mode]")
    print(
        "main bbox       :",
        train_out["bbox"].shape,
    )
    print(
        "main score_logit:",
        train_out[
            "score_logit"
        ].shape,
    )
    print(
        "aux bbox        :",
        train_out[
            "aux_bbox"
        ].shape,
    )
    print(
        "aux score_logit :",
        train_out[
            "aux_score_logit"
        ].shape,
    )

    model.eval()

    with torch.no_grad():
        eval_out = model(
            img,
            texts,
        )

    print("\n[Eval mode]")
    print(
        "main bbox       :",
        eval_out["bbox"].shape,
    )
    print(
        "main score_logit:",
        eval_out[
            "score_logit"
        ].shape,
    )
    print(
        "aux computed    :",
        eval_out[
            "aux_computed"
        ],
    )
    print(
        "aux bbox        :",
        eval_out[
            "aux_bbox"
        ],
    )

    with torch.no_grad():
        debug_out = model(
            img,
            texts,
            return_aux=True,
        )

    print(
        "\n[Eval mode, forced aux]"
    )
    print(
        "aux bbox        :",
        debug_out[
            "aux_bbox"
        ].shape,
    )
    print(
        "aux score_logit :",
        debug_out[
            "aux_score_logit"
        ].shape,
    )

    print(
        "main bbox diff  :",
        (
            eval_out["bbox"]
            - debug_out["bbox"]
        ).abs().max().item(),
    )
    print(
        "main score diff :",
        (
            eval_out["score_logit"]
            - debug_out["score_logit"]
        ).abs().max().item(),
    )
