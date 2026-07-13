from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import yaml

from . import card_legacy as _legacy
from .card_legacy import *  # noqa: F401,F403


class QueryHead(nn.Module):
    """BBox, localization-quality and text-alignment prediction head."""

    SUPPORTED_FUSIONS = {"geometric_mean", "product"}

    def __init__(
        self,
        hidden_dim: int = 512,
        score_fusion: str = "geometric_mean",
        fusion_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.bbox_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        # Keep the legacy parameter name for checkpoint compatibility. This is
        # now the localization-quality branch.
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.text_alignment_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.text_alignment_head.load_state_dict(
            self.score_head.state_dict(),
            strict=True,
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

    def predict_bbox(self, query_tokens: torch.Tensor) -> torch.Tensor:
        return self.decode_xyxy(self.bbox_head(query_tokens))

    def predict_scores(
        self,
        query_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quality_logit = self.score_head(query_tokens)
        text_alignment_logit = self.text_alignment_head(query_tokens)
        final_score_logit = self.fuse_logits(
            quality_logit,
            text_alignment_logit,
        )
        return quality_logit, text_alignment_logit, final_score_logit

    def fuse_logits(
        self,
        quality_logit: torch.Tensor,
        text_alignment_logit: torch.Tensor,
    ) -> torch.Tensor:
        quality = quality_logit.float().sigmoid()
        alignment = text_alignment_logit.float().sigmoid()
        if self.score_fusion == "product":
            probability = quality * alignment
        else:
            probability = torch.sqrt((quality * alignment).clamp_min(0.0))
        probability = probability.clamp(
            min=self.fusion_eps,
            max=1.0 - self.fusion_eps,
        )
        fused_logit = torch.logit(probability).to(quality_logit.dtype)
        fused_logit._quality_logit = quality_logit
        fused_logit._text_alignment_logit = text_alignment_logit
        return fused_logit

    def forward(self, query_tokens: torch.Tensor):
        if query_tokens.ndim != 3:
            raise ValueError(
                "QueryHead input must have shape [B, N_query, C], got "
                f"{tuple(query_tokens.shape)}"
            )
        bbox = self.predict_bbox(query_tokens)
        _, _, final_score_logit = self.predict_scores(query_tokens)
        return bbox, final_score_logit


_legacy.QueryHead = QueryHead


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


class VisionTextModel(_legacy.VisionTextModel):
    """Two-stage object-query model.

    Stage 1 uses the original Transformer to produce localization queries and
    bbox predictions. Stage 2 receives the localized query identity, optional
    bbox positional conditioning, and the same image/text context, then predicts
    localization quality and text alignment.
    """

    def __init__(
        self,
        *args: Any,
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
        super().__init__(*args, **kwargs)

        def resolve(name: str, value: Any, fallback: Any) -> Any:
            return defaults.get(name, fallback) if value is None else value

        self.staged_query_refinement = bool(resolve(
            "staged_query_refinement",
            staged_query_refinement,
            True,
        ))
        self.score_bbox_conditioning = bool(resolve(
            "score_bbox_conditioning",
            score_bbox_conditioning,
            True,
        ))
        self.score_bbox_detach = bool(resolve(
            "score_bbox_detach",
            score_bbox_detach,
            True,
        ))
        self.freeze_img_projection = bool(resolve(
            "freeze_img_projection",
            freeze_img_projection,
            False,
        ))
        score_fusion = str(resolve(
            "score_fusion",
            score_fusion,
            "geometric_mean",
        ))
        score_fusion_eps = float(resolve(
            "score_fusion_eps",
            score_fusion_eps,
            1e-6,
        ))

        for prediction_head in (self.head, self.aux_head):
            if prediction_head is None:
                continue
            prediction_head.fusion_eps = max(score_fusion_eps, 1e-8)
            prediction_head.set_score_fusion(score_fusion)

        self.score_fusion = self.head.score_fusion
        self.score_fusion_eps = self.head.fusion_eps

        self.score_transformer = _legacy.TransformerBlock(
            hidden_dim=self.hidden_dim,
            fusion_token_num=int(defaults.get(
                "fusion_token_num",
                self.fusion_token_num,
            )),
            num_heads=int(resolve(
                "score_num_heads",
                score_num_heads,
                defaults.get("num_heads", 8),
            )),
            num_layers=int(resolve(
                "score_num_layers",
                score_num_layers,
                2,
            )),
            mlp_ratio=float(resolve(
                "score_mlp_ratio",
                score_mlp_ratio,
                3.0,
            )),
            dropout=float(resolve(
                "score_dropout",
                score_dropout,
                defaults.get("dropout", 0.1),
            )),
            query_group_init_std=float(defaults.get(
                "query_group_init_std",
                0.02,
            )),
        )

        self.bbox_query_encoder = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Explicitly keep the image projection trainable unless YAML requests
        # otherwise. The stronger-ranking configuration sets this to false.
        for parameter in self.img_model.parameters():
            parameter.requires_grad_(not self.freeze_img_projection)

    def _prepare_legacy_state_dict(self, state_dict):
        prepared = super()._prepare_legacy_state_dict(state_dict)
        current_state = self.state_dict()

        for key, default_value in current_state.items():
            if ".text_alignment_head." in key and key not in prepared:
                source_key = key.replace(
                    ".text_alignment_head.",
                    ".score_head.",
                )
                source_value = prepared.get(source_key)
                if (
                    torch.is_tensor(source_value)
                    and tuple(source_value.shape) == tuple(default_value.shape)
                ):
                    prepared[key] = source_value.detach().clone()
                    continue
            if (
                key.startswith("score_transformer.")
                or key.startswith("bbox_query_encoder.")
            ) and key not in prepared:
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

        if not self.staged_query_refinement:
            main_score = outputs["main_score_logit"]
            main_quality = getattr(main_score, "_quality_logit", main_score)
            main_alignment = getattr(
                main_score,
                "_text_alignment_logit",
                main_score,
            )
            outputs.update({
                "quality_logit": main_quality,
                "text_alignment_logit": main_alignment,
                "final_score_logit": main_score,
                "score_decoupled": True,
                "query_stage_separated": False,
            })
            return outputs

        main_local_query = outputs["main_object_query_out"]
        aux_local_query = outputs.get("aux_object_query_out")
        main_bbox = outputs["main_bbox"]
        aux_bbox = outputs.get("aux_bbox")
        compute_auxiliary = bool(outputs.get("aux_computed", False))

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

        (
            main_quality,
            main_alignment,
            main_final,
        ) = self.head.predict_scores(main_score_query)

        aux_quality = None
        aux_alignment = None
        aux_final = None
        if compute_auxiliary and self.aux_head is not None:
            (
                aux_quality,
                aux_alignment,
                aux_final,
            ) = self.aux_head.predict_scores(aux_score_query)

        for tensor in (main_quality, main_alignment, main_final):
            self._attach_stage_metadata(
                tensor,
                main_local_query,
                main_score_query,
            )
        for tensor in (aux_quality, aux_alignment, aux_final):
            self._attach_stage_metadata(
                tensor,
                aux_local_query,
                aux_score_query,
            )

        outputs.update({
            # Backward-compatible inference aliases.
            "score_logit": main_final,
            "main_score_logit": main_final,
            "aux_score_logit": aux_final,

            # Explicit stage outputs.
            "quality_logit": main_quality,
            "text_alignment_logit": main_alignment,
            "final_score_logit": main_final,
            "main_quality_logit": main_quality,
            "main_text_alignment_logit": main_alignment,
            "main_final_score_logit": main_final,
            "aux_quality_logit": aux_quality,
            "aux_text_alignment_logit": aux_alignment,
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
        })
        return outputs
