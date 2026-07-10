from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from . import ranking_loss_legacy as _legacy
from .ranking_loss_legacy import *  # noqa: F401,F403


class RankingGroundingLoss(_legacy.RankingGroundingLoss):
    """Compatibility wrapper that routes optional pairwise ranking to quality."""

    def forward(
        self,
        pred_bbox: torch.Tensor,
        pred_score_logit: torch.Tensor,
        targets: List[dict],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pred_quality_logit = kwargs.pop("pred_quality_logit", None)
        pred_text_alignment_logit = kwargs.pop(
            "pred_text_alignment_logit",
            None,
        )

        quality = pred_quality_logit
        if quality is None:
            quality = getattr(
                pred_score_logit,
                "_quality_logit",
                pred_score_logit,
            )
        alignment = pred_text_alignment_logit
        if alignment is None:
            alignment = getattr(
                pred_score_logit,
                "_text_alignment_logit",
                None,
            )
        if alignment is not None:
            quality._text_alignment_logit = alignment

        aux_score = kwargs.get("aux_pred_score_logit")
        if aux_score is not None:
            aux_quality = kwargs.pop("aux_pred_quality_logit", None)
            aux_alignment = kwargs.pop(
                "aux_pred_text_alignment_logit",
                None,
            )
            if aux_quality is None:
                aux_quality = getattr(
                    aux_score,
                    "_quality_logit",
                    aux_score,
                )
            if aux_alignment is None:
                aux_alignment = getattr(
                    aux_score,
                    "_text_alignment_logit",
                    None,
                )
            if aux_alignment is not None:
                aux_quality._text_alignment_logit = aux_alignment
            kwargs["aux_pred_score_logit"] = aux_quality

        loss, metrics = super().forward(
            pred_bbox,
            quality,
            targets,
            *args,
            **kwargs,
        )

        # Preserve the existing train.py metric names while changing their
        # semantic source from the old coupled score to text alignment.
        if bool(metrics.get("score_decoupled", False)):
            metrics["negative_query_top1_score"] = metrics.get(
                "text_alignment_negative_top1_score",
                pred_bbox.new_zeros(()),
            )
            metrics["positive_query_top1_score"] = metrics.get(
                "text_alignment_positive_top1_score",
                pred_bbox.new_zeros(()),
            )
            metrics["positive_negative_score_margin"] = metrics.get(
                "text_alignment_margin",
                pred_bbox.new_zeros(()),
            )
            metrics["text_negative_count"] = metrics.get(
                "text_negative_query_rows",
                0.0,
            )

        return loss, metrics


def build_grounding_loss_from_config(
    config: Dict[str, Any],
) -> RankingGroundingLoss:
    return RankingGroundingLoss.from_config(config)
