from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from . import card_legacy as _legacy
from .card_legacy import *  # noqa: F401,F403


class QueryHead(nn.Module):
    """Decoupled bbox, localization-quality and text-alignment head.

    ``score_head`` is intentionally retained as the quality branch so old
    checkpoints keep their original parameter names. ``forward`` still
    returns two tensors for compatibility with the legacy VisionTextModel;
    the second tensor is the fused inference logit and carries explicit
    quality/alignment attributes consumed by the decoupled loss.
    """

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

        # Backward-compatible name: this branch now predicts localization
        # quality and is the only scalar score used by Hungarian matching.
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

        # Keep the first fused prediction identical to the legacy score. The
        # alignment branch then separates through its own loss.
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
    def _decode_xyxy(bbox_raw: torch.Tensor) -> torch.Tensor:
        bbox_raw = bbox_raw.sigmoid()
        x1 = torch.minimum(bbox_raw[..., 0], bbox_raw[..., 2])
        y1 = torch.minimum(bbox_raw[..., 1], bbox_raw[..., 3])
        x2 = torch.maximum(bbox_raw[..., 0], bbox_raw[..., 2])
        y2 = torch.maximum(bbox_raw[..., 1], bbox_raw[..., 3])
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _fuse_logits(
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

        # Explicit compatibility bridge for existing train/eval call sites.
        # The output dict also exposes both tensors by stable keys.
        fused_logit._quality_logit = quality_logit
        fused_logit._text_alignment_logit = text_alignment_logit
        return fused_logit

    def forward(
        self,
        query_tokens: torch.Tensor,
    ):
        if query_tokens.ndim != 3:
            raise ValueError(
                "QueryHead input must have shape [B, N_query, C], got "
                f"{tuple(query_tokens.shape)}"
            )

        bbox = self._decode_xyxy(self.bbox_head(query_tokens))
        quality_logit = self.score_head(query_tokens)
        text_alignment_logit = self.text_alignment_head(query_tokens)
        final_score_logit = self._fuse_logits(
            quality_logit,
            text_alignment_logit,
        )
        return bbox, final_score_logit


# The legacy VisionTextModel resolves QueryHead from its defining module.
_legacy.QueryHead = QueryHead


class VisionTextModel(_legacy.VisionTextModel):
    """Backward-compatible VisionTextModel with decoupled scalar scores."""

    def __init__(
        self,
        *args: Any,
        score_fusion: str = "geometric_mean",
        score_fusion_eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        for prediction_head in (self.head, self.aux_head):
            if prediction_head is None:
                continue
            prediction_head.fusion_eps = max(
                float(score_fusion_eps),
                1e-8,
            )
            prediction_head.set_score_fusion(score_fusion)

        self.score_fusion = self.head.score_fusion
        self.score_fusion_eps = self.head.fusion_eps

    def _prepare_legacy_state_dict(self, state_dict):
        prepared = super()._prepare_legacy_state_dict(state_dict)
        current_state = self.state_dict()

        # Old checkpoints contain score_head only. Clone those parameters into
        # text_alignment_head so strict loading and weights-only warm starts work.
        for key, default_value in current_state.items():
            if ".text_alignment_head." not in key or key in prepared:
                continue

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
            else:
                prepared[key] = default_value.detach().clone()

        return prepared

    @staticmethod
    def _score_components(
        score_logit: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if score_logit is None:
            return None, None
        return (
            getattr(score_logit, "_quality_logit", score_logit),
            getattr(score_logit, "_text_alignment_logit", score_logit),
        )

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

        main_score = outputs["main_score_logit"]
        main_quality, main_alignment = self._score_components(main_score)

        aux_score = outputs.get("aux_score_logit")
        aux_quality, aux_alignment = self._score_components(aux_score)

        outputs.update({
            # Main aliases used by existing inference/evaluation paths.
            "score_logit": main_score,
            "final_score_logit": main_score,
            "quality_logit": main_quality,
            "text_alignment_logit": main_alignment,

            # Explicit branch outputs.
            "main_final_score_logit": main_score,
            "main_quality_logit": main_quality,
            "main_text_alignment_logit": main_alignment,
            "aux_final_score_logit": aux_score,
            "aux_quality_logit": aux_quality,
            "aux_text_alignment_logit": aux_alignment,

            "score_decoupled": True,
            "score_fusion": self.score_fusion,
        })
        return outputs
