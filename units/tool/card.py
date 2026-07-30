from __future__ import annotations
import math
from typing import Any, Dict, Optional, Sequence, Mapping
import yaml

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
from torchvision.ops import FeaturePyramidNetwork
from transformers import BertModel, BertTokenizerFast
from transformers.utils import logging as transformers_logging
from huggingface_hub import logging as hub_logging
from huggingface_hub.utils import disable_progress_bars
from collections import OrderedDict
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
    Unified decoder-only Transformer for multimodal object queries.

    Context sequence:
        [Fusion Tokens, Image Tokens, Text Tokens]

    Prediction sequence:
        [Context Tokens, Object Queries]

    Main and auxiliary branches:
      - share Transformer weights;
      - use different learnable object-query embeddings;
      - are stacked along the batch dimension;
      - cannot attend to each other because self-attention never crosses
        different batch elements.

    Prefix-style attention:
      - context tokens may attend only to context tokens;
      - object queries may attend to all context tokens and all object queries.

    This preserves a single self-attention stack while giving object queries
    decoder-like access to image, text, and fusion information.
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

        # The group embedding is applied only to object queries. Context
        # tokens remain identical between Main and Aux branches.
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

        transformer_layer = nn.TransformerEncoderLayer(
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

        # PyTorch calls this TransformerEncoder, but here it is used as one
        # unified decoder-only self-attention stack over context + queries.
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _build_padding_mask(
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """
        Convert valid-token mask (1=valid) to Transformer padding mask
        (True=masked).
        """
        if valid_mask is None:
            return None
        return torch.eq(valid_mask, 0)

    @staticmethod
    def _build_prefix_attention_mask(
        context_length: int,
        query_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Prefix-style attention mask.

        Rows are readers and columns are keys/values.

        Context -> Context: allowed
        Context -> Query  : blocked
        Query   -> Context: allowed
        Query   -> Query  : allowed
        """
        context_length = int(context_length)
        query_length = int(query_length)
        total_length = context_length + query_length

        attention_mask = torch.zeros(
            total_length,
            total_length,
            dtype=torch.bool,
            device=device,
        )

        attention_mask[
            :context_length,
            context_length:,
        ] = True

        return attention_mask

    def _validate_queries(
        self,
        name: str,
        queries: torch.Tensor,
        *,
        batch_size: int,
    ) -> None:
        if queries.ndim != 3:
            raise ValueError(
                f"{name} must have shape [B, Q, C], "
                f"got {tuple(queries.shape)}"
            )

        if int(queries.shape[0]) != int(batch_size):
            raise ValueError(
                f"{name} batch size mismatch: "
                f"{queries.shape[0]} != {batch_size}"
            )

        if int(queries.shape[-1]) != self.hidden_dim:
            raise ValueError(
                f"{name} hidden dimension mismatch: "
                f"{queries.shape[-1]} != {self.hidden_dim}"
            )

        if int(queries.shape[1]) <= 0:
            raise ValueError(
                f"{name} must contain at least one object query"
            )

    def _add_group_embedding(
        self,
        queries: torch.Tensor,
        group_index: int,
    ) -> torch.Tensor:
        return (
            queries
            + self.query_group_embeddings[
                int(group_index)
            ].to(
                device=queries.device,
                dtype=queries.dtype,
            )
        )

    @staticmethod
    def _extend_valid_mask(
        context_valid_mask: torch.Tensor | None,
        *,
        batch_size: int,
        query_length: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if context_valid_mask is None:
            return None

        query_valid_mask = torch.ones(
            int(batch_size),
            int(query_length),
            dtype=context_valid_mask.dtype,
            device=device,
        )

        return torch.cat(
            [
                context_valid_mask,
                query_valid_mask,
            ],
            dim=1,
        )

    def forward(
        self,
        img_token: torch.Tensor,
        text_token: torch.Tensor,
        text_mask: torch.Tensor | None,
        main_queries: torch.Tensor,
        aux_queries: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """
        Args:
            img_token:
                [B, N_image, C]
            text_token:
                [B, N_text, C]
            text_mask:
                [B, N_text], 1 for valid text token.
            main_queries:
                [B, N_query, C]
            aux_queries:
                [B, N_query, C] when return_aux=True.

        Returns:
            main_out:
                [B, N_context + N_query, C]
            aux_out:
                [B, N_context + N_query, C] or None
        """
        context, context_valid_mask = self.fuse(
            img_token,
            text_token,
            text_mask,
        )

        batch_size = int(context.shape[0])
        context_length = int(context.shape[1])

        self._validate_queries(
            "main_queries",
            main_queries,
            batch_size=batch_size,
        )

        query_length = int(main_queries.shape[1])

        main_queries = self._add_group_embedding(
            main_queries,
            self.MAIN_GROUP,
        )
        main_sequence = torch.cat(
            [
                context,
                main_queries,
            ],
            dim=1,
        ) # [Fusion | Image | Text | Main Object Queries]

        complete_valid_mask = self._extend_valid_mask(
            context_valid_mask,
            batch_size=batch_size,
            query_length=query_length,
            device=context.device,
        )
        padding_mask = self._build_padding_mask(
            complete_valid_mask
        )

        attention_mask = self._build_prefix_attention_mask(
            context_length=context_length,
            query_length=query_length,
            device=context.device,
        )

        if not bool(return_aux):
            main_out = self.transformer(
                main_sequence,
                mask=attention_mask,
                src_key_padding_mask=padding_mask,
            )
            return self.norm(main_out), None

        if aux_queries is None:
            raise ValueError(
                "aux_queries is required when return_aux=True"
            )

        self._validate_queries(
            "aux_queries",
            aux_queries,
            batch_size=batch_size,
        )

        if int(aux_queries.shape[1]) != query_length:
            raise ValueError(
                "Main and Aux query counts must match for "
                "batch-axis grouping: "
                f"{query_length} != {aux_queries.shape[1]}"
            )

        aux_queries = self._add_group_embedding(
            aux_queries,
            self.AUX_GROUP,
        )
        aux_sequence = torch.cat(
            [
                context,
                aux_queries,
            ],
            dim=1,
        )

        # Main and Aux sequences are isolated by the batch dimension while
        # still sharing exactly the same Transformer parameters.
        grouped_sequence = torch.cat(
            [
                main_sequence,
                aux_sequence,
            ],
            dim=0,
        )

        if padding_mask is None:
            grouped_padding_mask = None
        else:
            grouped_padding_mask = torch.cat(
                [
                    padding_mask,
                    padding_mask,
                ],
                dim=0,
            )

        grouped_out = self.transformer(
            grouped_sequence,
            mask=attention_mask,
            src_key_padding_mask=grouped_padding_mask,
        )
        grouped_out = self.norm(grouped_out)

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

        self.block = nn.Sequential(*layers)

    def forward(self,x,):
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
    """
    將 FPN 多尺度特徵轉換成 Transformer image tokens。

    預設輸入：
        c3: [B, 256, 128, 128]
        c4: [B, 256,  64,  64]
        c5: [B, 256,  32,  32]

    預設輸出：
        c3 -> 16x16 = 256 tokens
        c4 ->  8x8 =  64 tokens
        c5 ->  4x4 =  16 tokens

        output: [B, 336, 512]
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 512,
        layer_num: int = 2,
        expand_ratio: float = 2.0,
        level_names: Sequence[str] = (
            "c3",
            "c4",
            "c5",
        ),
        token_grids: Sequence[Sequence[int]] = (
            (16, 16),
            (8, 8),
            (4, 4),
        ),
    ):
        super().__init__()

        self._validate_init_args(
            in_channels=in_channels,
            out_channels=out_channels,
            layer_num=layer_num,
            expand_ratio=expand_ratio,
            level_names=level_names,
            token_grids=token_grids,
        )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.layer_num = int(layer_num)
        self.expand_ratio = float(expand_ratio)

        self.level_names = tuple(
            str(name)
            for name in level_names
        )

        self.token_grids = tuple(
            (
                int(grid[0]),
                int(grid[1]),
            )
            for grid in token_grids
        )

        # 每個 FPN Level 使用獨立的投影模組，
        # 避免不同尺度共享完全相同的 projection。
        self.level_projectors = nn.ModuleDict({
            level_name: self._make_level_projector(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                layer_num=self.layer_num,
                expand_ratio=self.expand_ratio,
            )
            for level_name in self.level_names
        })

        # 用於區分 c3、c4、c5 的尺度來源。
        #
        # shape:
        # [num_levels, 1, 1, out_channels]
        self.level_embeddings = nn.Parameter(
            torch.zeros(
                len(self.level_names),
                1,
                1,
                self.out_channels,
            )
        )

        nn.init.normal_(
            self.level_embeddings,
            mean=0.0,
            std=0.02,
        )

        # 各層 token 數量。
        #
        # 預設：
        # 16*16 + 8*8 + 4*4
        # = 256 + 64 + 16
        # = 336
        self.tokens_per_level = {
            level_name: grid_height * grid_width
            for level_name, (
                grid_height,
                grid_width,
            ) in zip(
                self.level_names,
                self.token_grids,
            )
        }

        self.num_tokens = sum(
            self.tokens_per_level.values()
        )

    @staticmethod
    def _validate_init_args(
        in_channels: int,
        out_channels: int,
        layer_num: int,
        expand_ratio: float,
        level_names: Sequence[str],
        token_grids: Sequence[Sequence[int]],
    ) -> None:
        if (
            isinstance(in_channels, bool)
            or not isinstance(in_channels, int)
            or in_channels <= 0
        ):
            raise ValueError(
                "in_channels 必須是大於 0 的整數，"
                f"目前為 {in_channels!r}"
            )

        if (
            isinstance(out_channels, bool)
            or not isinstance(out_channels, int)
            or out_channels <= 0
        ):
            raise ValueError(
                "out_channels 必須是大於 0 的整數，"
                f"目前為 {out_channels!r}"
            )

        if (
            isinstance(layer_num, bool)
            or not isinstance(layer_num, int)
            or layer_num < 0
        ):
            raise ValueError(
                "layer_num 必須是大於等於 0 的整數，"
                f"目前為 {layer_num!r}"
            )

        if expand_ratio <= 0:
            raise ValueError(
                "expand_ratio 必須大於 0，"
                f"目前為 {expand_ratio}"
            )

        if len(level_names) == 0:
            raise ValueError(
                "level_names 不得為空"
            )

        if len(level_names) != len(token_grids):
            raise ValueError(
                "level_names 與 token_grids "
                "必須具有相同長度："
                f"{len(level_names)} != "
                f"{len(token_grids)}"
            )

        if len(set(level_names)) != len(level_names):
            raise ValueError(
                "level_names 不得包含重複名稱"
            )

        for level_name in level_names:
            if (
                not isinstance(level_name, str)
                or not level_name.strip()
            ):
                raise ValueError(
                    "每個 level name 都必須是非空字串，"
                    f"目前為 {level_name!r}"
                )

        for index, grid in enumerate(token_grids):
            if (
                not isinstance(
                    grid,
                    (tuple, list),
                )
                or len(grid) != 2
            ):
                raise ValueError(
                    "每個 token grid 都必須是 "
                    "[height, width]，"
                    f"第 {index} 個為 {grid!r}"
                )

            height = grid[0]
            width = grid[1]

            if (
                isinstance(height, bool)
                or not isinstance(height, int)
                or height <= 0
            ):
                raise ValueError(
                    "token grid height 必須是大於 0 "
                    f"的整數，目前為 {height!r}"
                )

            if (
                isinstance(width, bool)
                or not isinstance(width, int)
                or width <= 0
            ):
                raise ValueError(
                    "token grid width 必須是大於 0 "
                    f"的整數，目前為 {width!r}"
                )

    @staticmethod
    def _make_level_projector(
        in_channels: int,
        out_channels: int,
        layer_num: int,
        expand_ratio: float,
    ) -> nn.Sequential:
        """
        建立單一 FPN Level 的投影模組。
        以迴圈依據輸入的尺度執行投影
        流程：
            1x1 ConvGNAct
            -> MobileNetV4 refinement blocks
        """

        layers = [
            ConvGNAct(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
            )
        ]

        for _ in range(layer_num):
            layers.append(
                MobileNetV4ConvBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    expand_ratio=expand_ratio,
                    start_dw=True,
                    middle_dw=True,
                    kernel_size=3,
                )
            )

        return nn.Sequential(
            *layers
        )

    def _validate_feature(
        self,
        level_name: str,
        feature: torch.Tensor,
    ) -> None:
        if not torch.is_tensor(feature):
            raise TypeError(
                f"{level_name} 必須是 torch.Tensor，"
                f"目前為 {type(feature).__name__}"
            )

        if feature.ndim != 4:
            raise ValueError(
                f"{level_name} 必須是四維 Tensor "
                "[B, C, H, W]，"
                f"目前 shape={tuple(feature.shape)}"
            )

        if int(feature.shape[1]) != self.in_channels:
            raise ValueError(
                f"{level_name} channel 不符合設定："
                f"預期 {self.in_channels}，"
                f"實際為 {feature.shape[1]}"
            )

        if int(feature.shape[2]) <= 0:
            raise ValueError(
                f"{level_name} 的 height 必須大於 0"
            )

        if int(feature.shape[3]) <= 0:
            raise ValueError(
                f"{level_name} 的 width 必須大於 0"
            )

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if not isinstance(
            features,
            (dict, OrderedDict),
        ):
            raise TypeError(
                "ImgProjector 輸入必須是 dict "
                "或 OrderedDict，"
                f"目前為 {type(features).__name__}"
            )

        output_tokens = []
        batch_size = None

        for level_index, (
            level_name,
            target_grid,
        ) in enumerate(zip(
            self.level_names,
            self.token_grids,
        )):
            if level_name not in features:
                raise KeyError(
                    f"缺少 FPN Level：{level_name}。"
                    f"目前 keys={tuple(features.keys())}"
                )

            x = features[level_name]

            self._validate_feature(
                level_name=level_name,
                feature=x,
            )

            if batch_size is None:
                batch_size = int(x.shape[0])
            elif int(x.shape[0]) != batch_size:
                raise ValueError(
                    "所有 FPN Level 的 batch size "
                    "必須一致："
                    f"{level_name}={x.shape[0]}，"
                    f"預期={batch_size}"
                )

            original_size = x.shape[-2:]

            # 從 FPN 全局資訊。
            global_feature = F.adaptive_avg_pool2d(
                x,
                output_size=target_grid,
            )

            #融合前處理
            global_feature = F.interpolate(
                global_feature,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )

            # 根據當前輸入影像動態產生融合權重。
            gate_input = torch.cat([x,global_feature,],dim=1,)

            gate = self.level_gates[
                level_name
            ](gate_input)

            if gate.shape[0] != x.shape[0]:
                raise RuntimeError(
                    f"{level_name} gate batch size 錯誤："
                    f"{gate.shape[0]} != {x.shape[0]}"
                )

            # 原始 FPN 特徵提供細節，
            # Avg Pooling 提供全局資訊。
            x = (gate * x + (1.0 - gate) * global_feature)

            # 每個尺度使用獨立 projector。
            x = self.level_projectors[
                level_name
            ](x)

            # fused tensor size -> target grids
            x = F.adaptive_avg_pool2d(
                x,
                output_size=target_grid,
            )

            # [B, C, H, W]
            # -> [B, C, H*W]
            # -> [B, H*W, C]
            tokens = (
                x
                .flatten(2)
                .transpose(1, 2)
                .contiguous()
            )

            expected_tokens = (
                target_grid[0]
                * target_grid[1]
            )

            if int(tokens.shape[1]) != expected_tokens:
                raise RuntimeError(
                    f"{level_name} token 數量錯誤："
                    f"{tokens.shape[1]} != "
                    f"{expected_tokens}"
                )

            if int(tokens.shape[2]) != self.out_channels:
                raise RuntimeError(
                    f"{level_name} token dimension 錯誤："
                    f"{tokens.shape[2]} != "
                    f"{self.out_channels}"
                )

            level_embedding = (
                self.level_embeddings[
                    level_index
                ]
                .to(
                    device=tokens.device,
                    dtype=tokens.dtype,
                )
            )

            tokens = (
                tokens
                + level_embedding
            )

            output_tokens.append(tokens)

        image_tokens = torch.cat(
            output_tokens,
            dim=1,
        )

        if int(image_tokens.shape[1]) != self.num_tokens:
            raise RuntimeError(
                "ImgProjector 最終 token 數量錯誤："
                f"{image_tokens.shape[1]} != "
                f"{self.num_tokens}"
            )

        if int(image_tokens.shape[2]) != self.out_channels:
            raise RuntimeError(
                "ImgProjector 最終 hidden dimension 錯誤："
                f"{image_tokens.shape[2]} != "
                f"{self.out_channels}"
            )

        return image_tokens


class BackBone(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: Sequence[int] = (
            64,
            128,
            256,
            512,
            1024,
        ),
        base_depths: Sequence[int] = (
            2,
            3,
            2,
        ),
        width_multiple: float = 0.75,
        depth_multiple: float = 0.67,
        max_channels: int = 1024,
        channel_divisor: int = 8,
        fpn_channels: int = 256,
        hidden_dim: int = 512,
        projector_layers: int = 2,
        projector_expand_ratio: float = 2.0,
        level_names: Sequence[str] = (
            "c3",
            "c4",
            "c5",
        ),
        token_grids: Sequence[Sequence[int]] = (
            (16, 16),
            (8, 8),
            (4, 4),
        ),
    ) -> None:
        super().__init__()

        self.bottle_net = BottleNet(
            in_channels=int(in_channels),
            base_channels=tuple(int(value) for value in base_channels),
            base_depths=tuple(int(value) for value in base_depths),
            width_multiple=float(width_multiple),
            depth_multiple=float(depth_multiple),
            max_channels=int(max_channels),
            channel_divisor=int(channel_divisor),
        )

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=list(self.bottle_net.output_channels),
            out_channels=int(fpn_channels),
            norm_layer=None,
        )

        self.img_projector = ImgProjector(
            in_channels=int(fpn_channels),
            out_channels=int(hidden_dim),
            layer_num=int(projector_layers),
            expand_ratio=float(projector_expand_ratio),
            level_names=tuple(str(name) for name in level_names),
            token_grids=tuple(
                (int(grid[0]), int(grid[1]))
                for grid in token_grids
            ),
        )

        self.output_channels = self.bottle_net.output_channels
        self.fpn_channels = int(fpn_channels)
        self.hidden_dim = int(hidden_dim)
        self.tokens_per_level = dict(
            self.img_projector.tokens_per_level
        )
        self.num_tokens = int(
            self.img_projector.num_tokens
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(image):
            raise TypeError(
                "BackBone input must be torch.Tensor, "
                f"got {type(image).__name__}"
            )

        if image.ndim != 4:
            raise ValueError(
                "BackBone input must have shape [B, C, H, W], "
                f"got {tuple(image.shape)}"
            )

        features = self.bottle_net(image)
        fpn_features = self.fpn(features)
        image_tokens = self.img_projector(
            fpn_features
        )

        if image_tokens.ndim != 3:
            raise RuntimeError(
                "BackBone output must have shape [B, N, C], "
                f"got {tuple(image_tokens.shape)}"
            )

        if int(image_tokens.shape[1]) != self.num_tokens:
            raise RuntimeError(
                "BackBone image token count mismatch: "
                f"{image_tokens.shape[1]} != {self.num_tokens}"
            )

        if int(image_tokens.shape[2]) != self.hidden_dim:
            raise RuntimeError(
                "BackBone hidden dimension mismatch: "
                f"{image_tokens.shape[2]} != {self.hidden_dim}"
            )

        return image_tokens


class BottleNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: Sequence[int] = (
            64,
            128,
            256,
            512,
            1024,
        ),
        base_depths: Sequence[int] = (
            2,
            3,
            2,
        ),
        width_multiple: float = 0.75,
        depth_multiple: float = 0.67,
        max_channels: int = 1024,
        channel_divisor: int = 8,
    ):
        super().__init__()

        if len(base_channels) != 5:
            raise ValueError(
                "base_channels 必須包含 5 個數值："
                "Stem1、Stem2、C3、C4、C5"
            )

        if len(base_depths) != 3:
            raise ValueError(
                "base_depths 必須包含 3 個數值："
                "Stage3、Stage4、Stage5"
            )

        if width_multiple <= 0:
            raise ValueError(
                "width_multiple must be > 0"
            )

        if depth_multiple <= 0:
            raise ValueError(
                "depth_multiple must be > 0"
            )

        if max_channels <= 0:
            raise ValueError(
                "max_channels must be > 0"
            )

        if channel_divisor <= 0:
            raise ValueError(
                "channel_divisor must be > 0"
            )

        # stage channel計算
        scaled_channels = [
            self._scale_channels(
                base_channels=channel,
                width_multiple=width_multiple,
                max_channels=max_channels,
                divisor=channel_divisor,
            )
            for channel in base_channels
        ]

        # Stage3、Stage4、Stage5 的實際深度
        scaled_depths = [
            self._scale_depth(
                base_depth=depth,
                depth_multiple=depth_multiple,
            )
            for depth in base_depths
        ]

        stem1_channels = scaled_channels[0]
        stem2_channels = scaled_channels[1]
        c3_channels = scaled_channels[2]
        c4_channels = scaled_channels[3]
        c5_channels = scaled_channels[4]

        stage3_depth = scaled_depths[0]
        stage4_depth = scaled_depths[1]
        stage5_depth = scaled_depths[2]

        self.channels = tuple(scaled_channels)
        self.depths = tuple(scaled_depths)

        # Stem 只負責輸出 stride 4 特徵
        self.stem = nn.Sequential(
            ConvGNAct(
                in_channels,
                stem1_channels,
                kernel_size=3,
                stride=2,
            ),
            ConvGNAct(
                stem1_channels,
                stem2_channels,
                kernel_size=3,
                stride=2,
            ),
        )

        # C3：stride 8
        self.stage3 = self._make_stage(
            in_channels=stem2_channels,
            out_channels=c3_channels,
            depth=stage3_depth,
            stride=2,
        )

        # C4：stride 16
        self.stage4 = self._make_stage(
            in_channels=c3_channels,
            out_channels=c4_channels,
            depth=stage4_depth,
            stride=2,
        )

        # C5：stride 32
        self.stage5 = self._make_stage(
            in_channels=c4_channels,
            out_channels=c5_channels,
            depth=stage5_depth,
            stride=2,
        )

        # FPN必要參數紀錄
        self.output_channels = (
            c3_channels,
            c4_channels,
            c5_channels,
        )

    @staticmethod
    def _scale_channels(
        base_channels: int,
        width_multiple: float,
        max_channels: int,
        divisor: int = 8,
    ) -> int:
        if (
            isinstance(base_channels, bool)
            or not isinstance(base_channels, int)
            or base_channels <= 0
        ):
            raise ValueError(
                "base_channels 必須是大於 0 的整數，"
                f"目前為 {base_channels!r}"
            )

        scaled_channels = min(
            base_channels * width_multiple,
            max_channels,
        )

        #使channel對齊GroupNorm避免scale up 錯誤
        scaled_channels = max(divisor,int( scaled_channels + divisor / 2)// divisor* divisor,)

        return int(scaled_channels)

    @staticmethod
    def _scale_depth(
        base_depth: int,
        depth_multiple: float,
    ) -> int:
        if (
            isinstance(base_depth, bool)
            or not isinstance(base_depth, int)
            or base_depth <= 0
        ):
            raise ValueError(
                "base_depth 必須是大於 0 的整數，"
                f"目前為 {base_depth!r}"
            )

        return max(
            round(base_depth * depth_multiple),
            1,
        )

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int = 2,
    ) -> nn.Sequential:
        if (
            isinstance(in_channels, bool)
            or not isinstance(in_channels, int)
            or in_channels <= 0
        ):
            raise ValueError(
                "in_channels 必須是大於 0 的整數"
            )

        if (
            isinstance(out_channels, bool)
            or not isinstance(out_channels, int)
            or out_channels <= 0
        ):
            raise ValueError(
                "out_channels 必須是大於 0 的整數"
            )

        if (
            isinstance(depth, bool)
            or not isinstance(depth, int)
            or depth < 1
        ):
            raise ValueError(
                "depth 必須是大於等於 1 的整數"
            )

        if stride not in (1, 2):
            raise ValueError(
                "stride must be 1 or 2"
            )

        layers = [
            #執行下採樣
            ConvGNAct(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
            )
        ]

        #後續層不套用下採樣
        for _ in range(depth - 1):
            layers.append(
                ConvGNAct(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                )
            )

        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> OrderedDict[str, torch.Tensor]:
        x = self.stem(x)

        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        return OrderedDict({
            "c3": c3,
            "c4": c4,
            "c5": c5,
        })

class QueryHead(nn.Module):
    """BBox, localization-quality and token-level text-alignment head."""

    SUPPORTED_FUSIONS = {"geometric_mean", "product"}

    def __init__(
        self,
        hidden_dim: int = 512,
        score_fusion: str = "geometric_mean",
        fusion_eps: float = 1e-6,
        alignment_temperature: float = 0.07,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)

        self.bbox_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

        # Scalar localization-quality branch.
        # Keep the historical parameter name for checkpoint compatibility.
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Phrase Grounding alignment projections:
        # object query [B,Q,C] x text token [B,L,C] -> [B,Q,L].
        self.query_alignment_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.token_alignment_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )

        nn.init.eye_(self.query_alignment_proj[-1].weight)
        nn.init.eye_(self.token_alignment_proj[-1].weight)

        temperature = max(float(alignment_temperature), 1e-4)
        self.alignment_logit_scale = nn.Parameter(
            torch.tensor(
                math.log(1.0 / temperature),
                dtype=torch.float32,
            )
        )

        self.score_fusion = self._normalize_fusion(score_fusion)
        self.fusion_eps = max(float(fusion_eps), 1e-8)

    @classmethod
    def _normalize_fusion(cls, value: str) -> str:
        value = str(value).strip().lower()
        aliases = {
            "geometric": "geometric_mean",
            "geometric_mean": "geometric_mean",
            "sqrt_product": "geometric_mean",
            "product": "product",
            "multiply": "product",
        }
        if value not in aliases:
            raise ValueError(
                "score_fusion must be one of "
                f"{sorted(cls.SUPPORTED_FUSIONS)}, got {value!r}"
            )
        return aliases[value]

    def set_score_fusion(self, value: str) -> None:
        self.score_fusion = self._normalize_fusion(value)

    @staticmethod
    def decode_xyxy(bbox_raw: torch.Tensor) -> torch.Tensor:
        bbox_raw = bbox_raw.sigmoid()
        x1 = torch.minimum(bbox_raw[..., 0], bbox_raw[..., 2])
        y1 = torch.minimum(bbox_raw[..., 1], bbox_raw[..., 3])
        x2 = torch.maximum(bbox_raw[..., 0], bbox_raw[..., 2])
        y2 = torch.maximum(bbox_raw[..., 1], bbox_raw[..., 3])
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _validate_query_tokens(
        query_tokens: torch.Tensor,
    ) -> None:
        if query_tokens.ndim != 3:
            raise ValueError(
                "QueryHead input must have shape [B, Q, C], got "
                f"{tuple(query_tokens.shape)}"
            )

    def predict_bbox(
        self,
        query_tokens: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_query_tokens(query_tokens)
        return self.decode_xyxy(self.bbox_head(query_tokens))

    def predict_quality(
        self,
        query_tokens: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_query_tokens(query_tokens)
        return self.score_head(query_tokens)

    def predict_token_alignment(
        self,
        query_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return token-level alignment logits with shape [B, Q, L].

        text_mask must be true only for real caption tokens. Special tokens
        and padding are masked before returning.
        """
        self._validate_query_tokens(query_tokens)

        if text_tokens.ndim != 3:
            raise ValueError(
                "text_tokens must have shape [B, L, C], got "
                f"{tuple(text_tokens.shape)}"
            )
        if text_mask.ndim != 2:
            raise ValueError(
                "text_mask must have shape [B, L], got "
                f"{tuple(text_mask.shape)}"
            )
        if query_tokens.shape[0] != text_tokens.shape[0]:
            raise ValueError(
                "query/text batch mismatch: "
                f"{query_tokens.shape[0]} != {text_tokens.shape[0]}"
            )
        if text_tokens.shape[:2] != text_mask.shape:
            raise ValueError(
                "text token/mask mismatch: "
                f"{tuple(text_tokens.shape[:2])} != "
                f"{tuple(text_mask.shape)}"
            )
        if query_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                "query hidden dimension mismatch: "
                f"{query_tokens.shape[-1]} != {self.hidden_dim}"
            )
        if text_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                "text hidden dimension mismatch: "
                f"{text_tokens.shape[-1]} != {self.hidden_dim}"
            )

        query_features = F.normalize(
            self.query_alignment_proj(query_tokens).float(),
            dim=-1,
        )
        token_features = F.normalize(
            self.token_alignment_proj(text_tokens).float(),
            dim=-1,
        )

        logit_scale = self.alignment_logit_scale.exp().clamp(
            min=1.0,
            max=100.0,
        )

        alignment_logits = torch.einsum(
            "bqc,blc->bql",
            query_features,
            token_features,
        )
        alignment_logits = alignment_logits * logit_scale

        valid_mask = text_mask.to(
            device=alignment_logits.device,
            dtype=torch.bool,
        )
        alignment_logits = alignment_logits.masked_fill(
            ~valid_mask[:, None, :],
            -1.0e4,
        )

        return alignment_logits

    def predict_scores(
        self,
        query_tokens: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        quality_logit = self.predict_quality(query_tokens)

        token_alignment_logits = None
        if text_tokens is not None or text_mask is not None:
            if text_tokens is None or text_mask is None:
                raise ValueError(
                    "text_tokens and text_mask must be provided together"
                )
            token_alignment_logits = self.predict_token_alignment(
                query_tokens,
                text_tokens,
                text_mask,
            )

        # Phrase-specific score fusion requires a selected positive token map.
        # Until matcher/loss performs that reduction, score_logit remains the
        # localization-quality logit.
        final_score_logit = quality_logit
        final_score_logit._quality_logit = quality_logit
        final_score_logit._token_alignment_logits = (
            token_alignment_logits
        )

        return (
            quality_logit,
            token_alignment_logits,
            final_score_logit,
        )

    def fuse_logits(
        self,
        quality_logit: torch.Tensor,
        phrase_alignment_logit: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse quality with an already reduced phrase-alignment scalar.
        """
        quality = quality_logit.float().sigmoid()
        alignment = phrase_alignment_logit.float().sigmoid()

        if self.score_fusion == "product":
            probability = quality * alignment
        else:
            probability = torch.sqrt(
                (quality * alignment).clamp_min(0.0)
            )

        probability = probability.clamp(
            min=self.fusion_eps,
            max=1.0 - self.fusion_eps,
        )
        return torch.logit(probability).to(quality_logit.dtype)

    def forward(
        self,
        query_tokens: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ):
        bbox = self.predict_bbox(query_tokens)
        _, _, final_score_logit = self.predict_scores(
            query_tokens,
            text_tokens=text_tokens,
            text_mask=text_mask,
        )
        return bbox, final_score_logit


class _BaseVisionTextModel(nn.Module):
    """
    Decoder-only, object-query-based LightDet model.

    Shared context:
        BottleNet -> FPN -> image projector -> Image Tokens
        BERT -> Text Tokens
        FusionBlock -> Fusion Tokens

    Unified Transformer input:
        [Fusion, Image, Text, Object Queries]

    Main branch:
        Main object queries
        One-to-one Hungarian matching
        Used for validation and inference

    Auxiliary branch:
        Auxiliary object queries
        Repeated-GT one-to-many matching
        Used during training by default

    Main and Aux share the Transformer weights but remain attention-isolated
    by batch-axis grouping. Their QueryHead parameters are independent.
    """

    def __init__(
        self,
        backbone_config: Mapping[str, Any],
        fpn_config: Mapping[str, Any],
        image_projector_config: Mapping[str, Any],
        hidden_dim: int = 512,
        text_max_length: int = 256,
        fusion_token_num: int = 16,
        num_object_queries: int = 100,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 3.5,
        dropout: float = 0.1,
        freeze_bert: bool = True,
        precomputed_bert_path: str | None = None,
        use_auxiliary_head: bool = True,
        auxiliary_in_eval: bool = False,
        initialize_aux_from_main: bool = True,
        query_init_std: float = 0.02,
        query_group_init_std: float = 0.02,
        freeze_img_projection: bool = False,
    ) -> None:
        super().__init__()

        if not isinstance(backbone_config, Mapping):
            raise TypeError(
                "backbone_config must be a mapping, "
                f"got {type(backbone_config).__name__}"
            )

        if not isinstance(fpn_config, Mapping):
            raise TypeError(
                "fpn_config must be a mapping, "
                f"got {type(fpn_config).__name__}"
            )

        if not isinstance(image_projector_config, Mapping):
            raise TypeError(
                "image_projector_config must be a mapping, "
                f"got {type(image_projector_config).__name__}"
            )

        backbone_config = dict(backbone_config)
        fpn_config = dict(fpn_config)
        image_projector_config = dict(image_projector_config)

        self.hidden_dim = int(hidden_dim)
        self.fusion_token_num = int(fusion_token_num)
        self.num_object_queries = int(num_object_queries)
        self.num_text = int(text_max_length)

        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)

        self.use_auxiliary_head = bool(use_auxiliary_head)
        self.auxiliary_in_eval = bool(auxiliary_in_eval)
        self.freeze_img_projection = bool(freeze_img_projection)

        if self.hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {self.hidden_dim}"
            )

        if self.fusion_token_num <= 0:
            raise ValueError(
                "fusion_token_num must be > 0, "
                f"got {self.fusion_token_num}"
            )

        if self.num_object_queries <= 0:
            raise ValueError(
                "num_object_queries must be > 0, "
                f"got {self.num_object_queries}"
            )

        if self.num_text <= 0:
            raise ValueError(
                f"text_max_length must be > 0, got {self.num_text}"
            )

        if self.num_heads <= 0:
            raise ValueError(
                f"num_heads must be > 0, got {self.num_heads}"
            )

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by "
                f"num_heads={self.num_heads}"
            )

        if self.num_layers <= 0:
            raise ValueError(
                f"num_layers must be > 0, got {self.num_layers}"
            )

        if self.mlp_ratio <= 0:
            raise ValueError(
                f"mlp_ratio must be > 0, got {self.mlp_ratio}"
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {self.dropout}"
            )

        fpn_channels = int(fpn_config["out_channels"])

        projector_in_channels = int(
            image_projector_config.get("in_channels", fpn_channels)
        )

        projector_out_channels = int(
            image_projector_config.get("out_channels", self.hidden_dim)
        )

        if projector_in_channels != fpn_channels:
            raise ValueError(
                "image_projector.in_channels must match "
                "fpn.out_channels: "
                f"{projector_in_channels} != {fpn_channels}"
            )

        if projector_out_channels != self.hidden_dim:
            raise ValueError(
                "image_projector.out_channels must match hidden_dim: "
                f"{projector_out_channels} != {self.hidden_dim}"
            )

        level_names = tuple(
            str(name)
            for name in image_projector_config["level_names"]
        )

        token_grids = tuple(
            (int(grid[0]), int(grid[1]))
            for grid in image_projector_config["token_grids"]
        )

        if len(level_names) == 0:
            raise ValueError(
                "image_projector.level_names must not be empty"
            )

        if len(level_names) != len(token_grids):
            raise ValueError(
                "image_projector.level_names and token_grids "
                "must have the same length"
            )

        self.img_model = BackBone(
            in_channels=int(backbone_config["in_channels"]),
            base_channels=tuple(
                int(value)
                for value in backbone_config["base_channels"]
            ),
            base_depths=tuple(
                int(value)
                for value in backbone_config["base_depths"]
            ),
            width_multiple=float(backbone_config["width_multiple"]),
            depth_multiple=float(backbone_config["depth_multiple"]),
            max_channels=int(backbone_config["max_channels"]),
            channel_divisor=int(
                backbone_config.get("channel_divisor", 8)
            ),
            fpn_channels=fpn_channels,
            hidden_dim=self.hidden_dim,
            projector_layers=int(
                image_projector_config["layer_num"]
            ),
            projector_expand_ratio=float(
                image_projector_config["expand_ratio"]
            ),
            level_names=level_names,
            token_grids=token_grids,
        )

        self.num_image = int(self.img_model.num_tokens)

        if self.num_image <= 0:
            raise RuntimeError(
                f"img_model.num_tokens must be > 0, got {self.num_image}"
            )

        self.image_position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_image, self.hidden_dim)
        )

        nn.init.normal_(
            self.image_position_embeddings,
            mean=0.0,
            std=0.02,
        )

        self.text_model = Bert(
            out_dim=self.hidden_dim,
            max_length=self.num_text,
            freeze_bert=bool(freeze_bert),
            precomputed_bert_path=precomputed_bert_path,
        )

        self.main_object_queries = nn.Embedding(
            self.num_object_queries,
            self.hidden_dim,
        )

        self.aux_object_queries = (
            nn.Embedding(
                self.num_object_queries,
                self.hidden_dim,
            )
            if self.use_auxiliary_head
            else None
        )

        nn.init.normal_(
            self.main_object_queries.weight,
            mean=0.0,
            std=float(query_init_std),
        )

        if self.aux_object_queries is not None:
            nn.init.normal_(
                self.aux_object_queries.weight,
                mean=0.0,
                std=float(query_init_std),
            )

        self.transformer = TransformerBlock(
            hidden_dim=self.hidden_dim,
            fusion_token_num=self.fusion_token_num,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            query_group_init_std=float(query_group_init_std),
        )

        self.head = QueryHead(
            hidden_dim=self.hidden_dim,
        )

        self.aux_head = (
            QueryHead(
                hidden_dim=self.hidden_dim,
            )
            if self.use_auxiliary_head
            else None
        )

        if (
            self.aux_head is not None
            and bool(initialize_aux_from_main)
        ):
            self.initialize_auxiliary_head_from_main()

        for parameter in self.img_model.parameters():
            parameter.requires_grad_(
                not self.freeze_img_projection
            )

    @property
    def main_head(self):
        return self.head

    @torch.no_grad()
    def initialize_auxiliary_head_from_main(
        self,
    ) -> None:
        """
        Initialize the auxiliary prediction branch from Main.

        The object-query table is copied once, but remains an independent
        parameter afterwards. The Aux group embedding still breaks symmetry.
        """
        if self.aux_head is None:
            return

        self.aux_head.load_state_dict(
            self.head.state_dict(),
            strict=True,
        )

        if self.aux_object_queries is not None:
            self.aux_object_queries.weight.copy_(
                self.main_object_queries.weight
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
            or self.aux_object_queries is None
        ):
            return False

        if return_aux is not None:
            return bool(return_aux)

        if self.training:
            return True

        return bool(self.auxiliary_in_eval)

    @staticmethod
    def _expand_object_queries(
        query_embedding: nn.Embedding,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Expand learned query table:
            [N_query, C] -> [B, N_query, C]
        """
        query_weight = query_embedding.weight.to(
            device=device,
            dtype=dtype,
        )

        return (
            query_weight
            .unsqueeze(0)
            .expand(
                int(batch_size),
                -1,
                -1,
            )
        )

    def _prepare_legacy_state_dict(
        self,
        state_dict,
    ):
        """
        Load older dense checkpoints as initialization.

        Existing backbone, Transformer, bbox-head, and score-head weights are
        reused. Newly introduced object queries and image position embeddings
        are initialized from the current model when absent.
        """
        prepared = state_dict.copy()

        default_keys = {
            "transformer.query_group_embeddings": (
                self.transformer
                .query_group_embeddings
                .detach()
                .clone()
            ),
            "image_position_embeddings": (
                self.image_position_embeddings
                .detach()
                .clone()
            ),
            "main_object_queries.weight": (
                self.main_object_queries
                .weight
                .detach()
                .clone()
            ),
        }

        if self.aux_object_queries is not None:
            default_keys[
                "aux_object_queries.weight"
            ] = (
                self.aux_object_queries
                .weight
                .detach()
                .clone()
            )

        for key, value in default_keys.items():
            if key not in prepared:
                prepared[key] = value

        if self.aux_head is None:
            return {
                key: value
                for key, value in prepared.items()
                if (
                    not str(key).startswith("aux_head.")
                    and not str(key).startswith(
                        "aux_object_queries."
                    )
                )
            }

        has_auxiliary_weights = any(
            str(key).startswith("aux_head.")
            for key in prepared.keys()
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
                aux_key = aux_prefix + suffix

                if aux_key not in prepared:
                    prepared[aux_key] = value

        return prepared

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        assign=False,
    ):
        prepared = self._prepare_legacy_state_dict(
            state_dict
        )

        try:
            return super().load_state_dict(
                prepared,
                strict=strict,
                assign=assign,
            )
        except TypeError:
            return super().load_state_dict(
                prepared,
                strict=strict,
            )

    def forward(
        self,
        img,
        texts,
        image_indices=None,
        return_aux=None,
    ):
        img_token = self.img_model(img)

        if int(img_token.shape[1]) != self.num_image:
            raise RuntimeError(
                "Image token count does not match configured multi-scale grids: "
                f"{img_token.shape[1]} != {self.num_image}"
            )

        img_token = (
            img_token
            + self.image_position_embeddings.to(
                device=img_token.device,
                dtype=img_token.dtype,
            )
        )

        if image_indices is not None:
            if not torch.is_tensor(
                image_indices
            ):
                image_indices = torch.tensor(
                    image_indices,
                    dtype=torch.long,
                    device=img_token.device,
                )
            else:
                image_indices = image_indices.to(
                    device=img_token.device,
                    dtype=torch.long,
                    non_blocking=True,
                )

            image_indices = image_indices.reshape(-1)

            if image_indices.numel() > 0:
                minimum_index = int(
                    image_indices.min().item()
                )
                maximum_index = int(
                    image_indices.max().item()
                )

                if minimum_index < 0:
                    raise IndexError(
                        "image_indices contains a negative index: "
                        f"{minimum_index}"
                    )

                if maximum_index >= int(
                    img_token.shape[0]
                ):
                    raise IndexError(
                        "image_indices exceeds image batch: "
                        f"max_index={maximum_index}, "
                        f"image_batch={img_token.shape[0]}"
                    )

            img_token = img_token.index_select(
                0,
                image_indices,
            )

        text_out = self.text_model(texts)
        text_token = text_out["text_tokens"]
        text_mask = text_out["text_mask"]

        if int(img_token.shape[0]) != int(
            text_token.shape[0]
        ):
            raise ValueError(
                "Image-query batch size mismatch: "
                f"image={img_token.shape[0]}, "
                f"text={text_token.shape[0]}"
            )

        compute_auxiliary = self._resolve_return_aux(
            return_aux
        )
        batch_size = int(img_token.shape[0])

        main_queries = self._expand_object_queries(
            self.main_object_queries,
            batch_size=batch_size,
            device=img_token.device,
            dtype=img_token.dtype,
        )

        aux_queries = None
        if compute_auxiliary:
            if self.aux_object_queries is None:
                raise RuntimeError(
                    "Auxiliary queries are not initialized"
                )

            aux_queries = self._expand_object_queries(
                self.aux_object_queries,
                batch_size=batch_size,
                device=img_token.device,
                dtype=img_token.dtype,
            )

        (
            main_transformer_out,
            aux_transformer_out,
        ) = self.transformer(
            img_token=img_token,
            text_token=text_token,
            text_mask=text_mask,
            main_queries=main_queries,
            aux_queries=aux_queries,
            return_aux=compute_auxiliary,
        )

        # Object queries are appended after all context tokens.
        main_query_out = main_transformer_out[
            :,
            -self.num_object_queries:,
            :,
        ]

        (
            main_bbox,
            main_score_logit,
        ) = self.head(main_query_out)

        aux_bbox = None
        aux_score_logit = None
        aux_query_out = None

        if compute_auxiliary:
            if aux_transformer_out is None:
                raise RuntimeError(
                    "Auxiliary output was requested but "
                    "Transformer returned None"
                )

            aux_query_out = aux_transformer_out[
                :,
                -self.num_object_queries:,
                :,
            ]

            (
                aux_bbox,
                aux_score_logit,
            ) = self.aux_head(aux_query_out)

        return {
            # Backward-compatible Main aliases.
            "bbox": main_bbox,
            "score_logit": main_score_logit,

            # Explicit hybrid predictions.
            "main_bbox": main_bbox,
            "main_score_logit": main_score_logit,
            "aux_bbox": aux_bbox,
            "aux_score_logit": aux_score_logit,
            "aux_computed": bool(compute_auxiliary),

            # Final object-query representations.
            "main_object_query_out": main_query_out,
            "aux_object_query_out": aux_query_out,

            # Initial learned queries expanded to the current batch.
            "main_object_queries": main_queries,
            "aux_object_queries": aux_queries,

            # Full decoder-only sequence outputs for diagnostics.
            "transformer_out": main_transformer_out,
            "main_transformer_out": main_transformer_out,
            "aux_transformer_out": aux_transformer_out,

            "query_groups_isolated": True,
            "query_group_batching": "batch_axis",
            "prediction_source": "object_queries",
            "decoder_only": True,

            "img_token": img_token,
            "text_token": text_token,
            "text_mask": text_mask,
        }


# ------------------------------------------------------------------
# Public LightDet interface
# ------------------------------------------------------------------

def _read_model_defaults() -> Dict[str, Any]:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "model"
        / "cards"
        / "config"
        / "model.yaml"
    )
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return dict(config.get("model", {}))


class VisionTextModel(_BaseVisionTextModel):
    """Two-stage object-query model.

    Stage 1 uses the original Transformer to produce localization queries and
    bbox predictions. Stage 2 receives the localized query identity, optional
    bbox positional conditioning, and the same image/text context, then predicts
    localization quality and text alignment.
    """

    def __init__(
        self,
        *args: Any,
        backbone_config: Optional[Mapping[str, Any]] = None,
        fpn_config: Optional[Mapping[str, Any]] = None,
        image_projector_config: Optional[Mapping[str, Any]] = None,
        staged_query_refinement: Optional[bool] = None,
        score_num_heads: Optional[int] = None,
        score_num_layers: Optional[int] = None,
        score_mlp_ratio: Optional[float] = None,
        score_dropout: Optional[float] = None,
        score_bbox_conditioning: Optional[bool] = None,
        score_bbox_detach: Optional[bool] = None,
        freeze_img_projection: Optional[bool] = None,
        score_fusion: Optional[str] = None,
        score_fusion_eps: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        defaults = _read_model_defaults()

        def resolve(name: str, value: Any, fallback: Any) -> Any:
            if value is not None:
                return value

            return defaults.get(name, fallback)

        if backbone_config is None:
            backbone_config = defaults.get("backbone")

        if fpn_config is None:
            fpn_config = defaults.get("fpn")

        if image_projector_config is None:
            image_projector_config = defaults.get("image_projector")

        if not isinstance(backbone_config, Mapping):
            raise TypeError(
                "backbone_config must be a mapping, "
                f"got {type(backbone_config).__name__}"
            )

        if not isinstance(fpn_config, Mapping):
            raise TypeError(
                "fpn_config must be a mapping, "
                f"got {type(fpn_config).__name__}"
            )

        if not isinstance(image_projector_config, Mapping):
            raise TypeError(
                "image_projector_config must be a mapping, "
                f"got {type(image_projector_config).__name__}"
            )

        resolved_freeze_img_projection = bool(
            resolve(
                "freeze_img_projection",
                freeze_img_projection,
                False,
            )
        )

        super().__init__(
            *args,
            backbone_config=dict(backbone_config),
            fpn_config=dict(fpn_config),
            image_projector_config=dict(image_projector_config),
            freeze_img_projection=resolved_freeze_img_projection,
            **kwargs,
        )

        self.staged_query_refinement = bool(
            resolve(
                "staged_query_refinement",
                staged_query_refinement,
                True,
            )
        )

        self.score_bbox_conditioning = bool(
            resolve(
                "score_bbox_conditioning",
                score_bbox_conditioning,
                True,
            )
        )

        self.score_bbox_detach = bool(
            resolve(
                "score_bbox_detach",
                score_bbox_detach,
                True,
            )
        )

        self.freeze_img_projection = resolved_freeze_img_projection

        resolved_score_fusion = str(
            resolve(
                "score_fusion",
                score_fusion,
                "geometric_mean",
            )
        )

        resolved_score_fusion_eps = float(
            resolve(
                "score_fusion_eps",
                score_fusion_eps,
                1e-6,
            )
        )

        resolved_score_num_heads = int(
            resolve(
                "score_num_heads",
                score_num_heads,
                defaults.get("num_heads", 8),
            )
        )

        resolved_score_num_layers = int(
            resolve(
                "score_num_layers",
                score_num_layers,
                2,
            )
        )

        resolved_score_mlp_ratio = float(
            resolve(
                "score_mlp_ratio",
                score_mlp_ratio,
                3.0,
            )
        )

        resolved_score_dropout = float(
            resolve(
                "score_dropout",
                score_dropout,
                defaults.get("dropout", 0.1),
            )
        )

        if resolved_score_num_heads <= 0:
            raise ValueError("score_num_heads must be > 0")

        if self.hidden_dim % resolved_score_num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by "
                f"score_num_heads={resolved_score_num_heads}"
            )

        if resolved_score_num_layers <= 0:
            raise ValueError("score_num_layers must be > 0")

        if resolved_score_mlp_ratio <= 0:
            raise ValueError("score_mlp_ratio must be > 0")

        if not 0.0 <= resolved_score_dropout < 1.0:
            raise ValueError("score_dropout must be in [0, 1)")

        if resolved_score_fusion_eps <= 0:
            raise ValueError("score_fusion_eps must be > 0")

        for prediction_head in (self.head, self.aux_head):
            if prediction_head is None:
                continue

            prediction_head.fusion_eps = max(
                resolved_score_fusion_eps,
                1e-8,
            )

            prediction_head.set_score_fusion(
                resolved_score_fusion
            )

        if self.head is None:
            raise RuntimeError(
                "VisionTextModel requires a main prediction head"
            )

        self.score_fusion = self.head.score_fusion
        self.score_fusion_eps = self.head.fusion_eps

        self.score_transformer = TransformerBlock(
            hidden_dim=self.hidden_dim,
            fusion_token_num=int(
                defaults.get(
                    "fusion_token_num",
                    self.fusion_token_num,
                )
            ),
            num_heads=resolved_score_num_heads,
            num_layers=resolved_score_num_layers,
            mlp_ratio=resolved_score_mlp_ratio,
            dropout=resolved_score_dropout,
            query_group_init_std=float(
                defaults.get(
                    "query_group_init_std",
                    0.02,
                )
            ),
        )

        self.bbox_query_encoder = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        for parameter in self.img_model.parameters():
            parameter.requires_grad_(
                not self.freeze_img_projection
            )


    def _prepare_legacy_state_dict(self, state_dict):
        prepared = super()._prepare_legacy_state_dict(state_dict)
        current_state = self.state_dict()

        # Old scalar text-alignment parameters do not match the new
        # token-level projection heads.
        prepared = {
            key: value
            for key, value in prepared.items()
            if key in current_state
        }

        # Preserve compatible tensors and initialize new/changed tensors from
        # the current model definition.
        for key, default_value in current_state.items():
            source_value = prepared.get(key)

            if (
                source_value is None
                or not torch.is_tensor(source_value)
                or tuple(source_value.shape)
                != tuple(default_value.shape)
            ):
                prepared[key] = default_value.detach().clone()

        return prepared

    def _condition_score_queries(
        self,
        localization_query: torch.Tensor,
        bbox: torch.Tensor,
    ) -> torch.Tensor:
        if not self.score_bbox_conditioning:
            return localization_query
        bbox_input = bbox.detach() if self.score_bbox_detach else bbox
        return localization_query + self.bbox_query_encoder(
            bbox_input.to(localization_query.dtype)
        )

    @staticmethod
    def _attach_stage_metadata(
        tensor: Optional[torch.Tensor],
        localization_query: Optional[torch.Tensor],
        score_query: Optional[torch.Tensor],
    ) -> None:
        if tensor is None:
            return
        tensor._localization_query_out = localization_query
        tensor._score_query_out = score_query
        tensor._query_stage_separated = True

    def _score_stage(
        self,
        *,
        img_token: torch.Tensor,
        text_token: torch.Tensor,
        text_mask: torch.Tensor,
        localization_query: torch.Tensor,
        bbox: torch.Tensor,
        aux_localization_query: Optional[torch.Tensor],
        aux_bbox: Optional[torch.Tensor],
        compute_auxiliary: bool,
    ):
        main_score_input = self._condition_score_queries(
            localization_query,
            bbox,
        )
        aux_score_input = None
        if compute_auxiliary:
            if aux_localization_query is None or aux_bbox is None:
                raise RuntimeError(
                    "Auxiliary localization output is required for score refinement"
                )
            aux_score_input = self._condition_score_queries(
                aux_localization_query,
                aux_bbox,
            )

        main_score_out, aux_score_out = self.score_transformer(
            img_token=img_token,
            text_token=text_token,
            text_mask=text_mask,
            main_queries=main_score_input,
            aux_queries=aux_score_input,
            return_aux=compute_auxiliary,
        )
        main_score_query = main_score_out[:, -self.num_object_queries:, :]
        aux_score_query = (
            aux_score_out[:, -self.num_object_queries:, :]
            if aux_score_out is not None
            else None
        )
        return main_score_query, aux_score_query


    def forward(
        self,
        img,
        texts,
        image_indices=None,
        return_aux=None,
    ) -> Dict[str, Any]:
        outputs = super().forward(
            img,
            texts,
            image_indices=image_indices,
            return_aux=return_aux,
        )

        # Preserve each BERT token's character interval for ODVG span mapping.
        normalized_texts = self.text_model._normalize_texts(texts)
        tokenized_text = self.text_model.tokenizer(
            normalized_texts,
            padding="max_length",
            truncation=True,
            max_length=self.num_text,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        token_offsets = tokenized_text["offset_mapping"].to(
            device=outputs["text_mask"].device,
            dtype=torch.long,
            non_blocking=True,
        )
        token_attention_mask = tokenized_text["attention_mask"].to(
            device=outputs["text_mask"].device,
            dtype=outputs["text_mask"].dtype,
            non_blocking=True,
        )

        if not torch.equal(
            token_attention_mask,
            outputs["text_mask"],
        ):
            raise RuntimeError(
                "ODVG tokenizer attention mask differs from BERT text mask"
            )

        # [CLS], [SEP] and padding have offset [0, 0].
        alignment_text_mask = (
            outputs["text_mask"].to(dtype=torch.bool)
            & (token_offsets[..., 1] > token_offsets[..., 0])
        )

        outputs["token_offsets"] = token_offsets
        outputs["alignment_text_mask"] = alignment_text_mask

        if not self.staged_query_refinement:
            main_query = outputs["main_object_query_out"]
            main_quality = self.head.predict_quality(main_query)
            main_token_alignment = (
                self.head.predict_token_alignment(
                    main_query,
                    outputs["text_token"],
                    alignment_text_mask,
                )
            )
            main_final = main_quality

            for tensor in (
                main_quality,
                main_token_alignment,
                main_final,
            ):
                self._attach_stage_metadata(
                    tensor,
                    main_query,
                    main_query,
                )

            outputs.update({
                "score_logit": main_final,
                "main_score_logit": main_final,
                "quality_logit": main_quality,
                "text_alignment_logit": main_token_alignment,
                "token_alignment_logits": main_token_alignment,
                "final_score_logit": main_final,
                "main_quality_logit": main_quality,
                "main_text_alignment_logit": main_token_alignment,
                "main_token_alignment_logits": main_token_alignment,
                "main_final_score_logit": main_final,
                "score_decoupled": True,
                "query_stage_separated": False,
                "phrase_score_fusion_deferred": True,
            })
            return outputs

        main_local_query = outputs["main_object_query_out"]
        aux_local_query = outputs.get("aux_object_query_out")
        main_bbox = outputs["main_bbox"]
        aux_bbox = outputs.get("aux_bbox")
        compute_auxiliary = bool(
            outputs.get("aux_computed", False)
        )

        main_score_query, aux_score_query = self._score_stage(
            img_token=outputs["img_token"],
            text_token=outputs["text_token"],
            text_mask=outputs["text_mask"],
            localization_query=main_local_query,
            bbox=main_bbox,
            aux_localization_query=aux_local_query,
            aux_bbox=aux_bbox,
            compute_auxiliary=compute_auxiliary,
        )

        main_quality = self.head.predict_quality(
            main_score_query
        )
        main_token_alignment = (
            self.head.predict_token_alignment(
                main_score_query,
                outputs["text_token"],
                alignment_text_mask,
            )
        )
        main_final = main_quality

        aux_quality = None
        aux_token_alignment = None
        aux_final = None

        if compute_auxiliary and self.aux_head is not None:
            aux_quality = self.aux_head.predict_quality(
                aux_score_query
            )
            aux_token_alignment = (
                self.aux_head.predict_token_alignment(
                    aux_score_query,
                    outputs["text_token"],
                    alignment_text_mask,
                )
            )
            aux_final = aux_quality

        for tensor in (
            main_quality,
            main_token_alignment,
            main_final,
        ):
            self._attach_stage_metadata(
                tensor,
                main_local_query,
                main_score_query,
            )

        for tensor in (
            aux_quality,
            aux_token_alignment,
            aux_final,
        ):
            self._attach_stage_metadata(
                tensor,
                aux_local_query,
                aux_score_query,
            )

        outputs.update({
            # Existing scalar localization-quality interface.
            "score_logit": main_final,
            "main_score_logit": main_final,
            "aux_score_logit": aux_final,

            # ODVG token-level alignment interface.
            "quality_logit": main_quality,
            "text_alignment_logit": main_token_alignment,
            "token_alignment_logits": main_token_alignment,
            "final_score_logit": main_final,

            "main_quality_logit": main_quality,
            "main_text_alignment_logit": main_token_alignment,
            "main_token_alignment_logits": main_token_alignment,
            "main_final_score_logit": main_final,

            "aux_quality_logit": aux_quality,
            "aux_text_alignment_logit": aux_token_alignment,
            "aux_token_alignment_logits": aux_token_alignment,
            "aux_final_score_logit": aux_final,

            "main_localization_query_out": main_local_query,
            "main_score_query_out": main_score_query,
            "aux_localization_query_out": aux_local_query,
            "aux_score_query_out": aux_score_query,
            "score_transformer_out": main_score_query,

            "score_decoupled": True,
            "query_stage_separated": True,
            "score_bbox_conditioning": self.score_bbox_conditioning,
            "score_bbox_detach": self.score_bbox_detach,
            "score_fusion": self.score_fusion,
            "phrase_score_fusion_deferred": True,
        })

        return outputs


def test_img_projector():
    model = BackBone(
        in_channels=3,
        base_channels=(
            64,
            128,
            256,
            512,
            1024,
        ),
        base_depths=(
            2,
            3,
            2,
        ),
        width_multiple=0.75,
        depth_multiple=0.67,
        max_channels=1024,
        channel_divisor=8,
        fpn_channels=256,
        hidden_dim=512,
        projector_layers=2,
        projector_expand_ratio=2.0,
        level_names=(
            "c3",
            "c4",
            "c5",
        ),
        token_grids=(
            (16, 16),
            (8, 8),
            (4, 4),
        ),
    )

    model.eval()

    image = torch.randn(
        1,
        3,
        1024,
        1024,
    )

    with torch.no_grad():
        image_tokens = model(image)

    print(
        "Backbone channels:",
        model.bottle_net.channels,
    )

    print(
        "Backbone depths:",
        model.bottle_net.depths,
    )

    print(
        "FPN input channels:",
        model.bottle_net.output_channels,
    )

    print(
        "tokens per level:",
        model.tokens_per_level,
    )

    print(
        "total tokens:",
        model.num_tokens,
    )

    print(
        "image tokens:",
        tuple(image_tokens.shape),
    )


if __name__ == "__main__":
    test_img_projector()