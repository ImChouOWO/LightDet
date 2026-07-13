from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import loss_decoupled as _base
from .loss_decoupled import *  # noqa: F401,F403


class GroundingLoss(_base.GroundingLoss):
    """Decoupled grounding loss with staged-query identity regularization.

    The localization Transformer remains responsible for bbox regression. The
    refinement Transformer is responsible for quality and text alignment. A
    small optional cosine identity term keeps score query ``i`` associated with
    localization query ``i`` without allowing score losses to alter the bbox
    branch when ``detach_localization`` is enabled.
    """

    def __init__(
        self,
        *args: Any,
        query_refinement_identity_weight: float = 0.0,
        query_refinement_start_epoch: int = 1,
        query_refinement_warmup_epoch: int = 5,
        query_refinement_detach_localization: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._set_query_refinement_options(
            identity_weight=query_refinement_identity_weight,
            start_epoch=query_refinement_start_epoch,
            warmup_epoch=query_refinement_warmup_epoch,
            detach_localization=query_refinement_detach_localization,
        )

    def _set_query_refinement_options(
        self,
        *,
        identity_weight: float,
        start_epoch: int,
        warmup_epoch: int,
        detach_localization: bool,
    ) -> None:
        self.query_refinement_identity_weight = max(
            0.0,
            float(identity_weight),
        )
        self.query_refinement_start_epoch = max(1, int(start_epoch))
        self.query_refinement_warmup_epoch = max(1, int(warmup_epoch))
        self.query_refinement_detach_localization = bool(
            detach_localization
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GroundingLoss":
        criterion = _base.GroundingLoss.from_config.__func__(cls, config)
        loss_cfg = config.get("loss", config)
        refinement = dict(loss_cfg.get("query_refinement", {}))
        criterion._set_query_refinement_options(
            identity_weight=refinement.get("identity_weight", 0.0),
            start_epoch=refinement.get("start_epoch", 1),
            warmup_epoch=refinement.get("warmup_epoch", 5),
            detach_localization=refinement.get(
                "detach_localization",
                True,
            ),
        )
        return criterion

    def query_refinement_identity_loss(
        self,
        pred_score_logit: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = pred_score_logit.new_zeros(())
        localization_query = getattr(
            pred_score_logit,
            "_localization_query_out",
            None,
        )
        score_query = getattr(
            pred_score_logit,
            "_score_query_out",
            None,
        )
        if localization_query is None or score_query is None:
            return zero, {
                "cosine_mean": zero,
                "feature_delta_mean": zero,
            }
        if localization_query.shape != score_query.shape:
            raise ValueError(
                "Localization/score query shape mismatch: "
                f"{tuple(localization_query.shape)} != "
                f"{tuple(score_query.shape)}"
            )

        target = (
            localization_query.detach()
            if self.query_refinement_detach_localization
            else localization_query
        )
        target_float = target.float()
        score_float = score_query.float()
        cosine = F.cosine_similarity(
            score_float,
            target_float,
            dim=-1,
        )
        loss = (1.0 - cosine).mean()
        delta = (score_float - target_float).pow(2).mean().sqrt()
        return loss.to(pred_score_logit.dtype), {
            "cosine_mean": cosine.mean().to(pred_score_logit.dtype),
            "feature_delta_mean": delta.to(pred_score_logit.dtype),
        }

    def forward(
        self,
        pred_bbox: torch.Tensor,
        pred_score_logit: torch.Tensor,
        targets: List[dict],
        *args: Any,
        current_epoch=None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        base_loss, metrics = super().forward(
            pred_bbox,
            pred_score_logit,
            targets,
            *args,
            current_epoch=current_epoch,
            **kwargs,
        )

        has_staged_query = bool(getattr(
            pred_score_logit,
            "_query_stage_separated",
            False,
        ))
        alpha = (
            _base.schedule_progress(
                current_epoch,
                self.query_refinement_start_epoch,
                self.query_refinement_warmup_epoch,
                curve="smoothstep",
            )
            if has_staged_query
            and self.query_refinement_identity_weight > 0.0
            else 0.0
        )
        if alpha > 0.0:
            identity_raw, identity_metrics = (
                self.query_refinement_identity_loss(pred_score_logit)
            )
        else:
            identity_raw = pred_score_logit.new_zeros(())
            identity_metrics = {
                "cosine_mean": identity_raw,
                "feature_delta_mean": identity_raw,
            }

        identity_contrib = (
            float(self.query_refinement_identity_weight)
            * float(alpha)
            * identity_raw
        )
        total_loss = base_loss + identity_contrib

        result = dict(metrics)
        result.update({
            "loss": total_loss.detach(),
            "loss_main_total": (
                result.get("loss_main_total", base_loss.detach())
                + identity_contrib.detach()
            ),
            "loss_query_refinement_identity": identity_raw.detach(),
            "loss_query_refinement_identity_contrib": (
                identity_contrib.detach()
            ),
            "query_refinement_enabled": has_staged_query,
            "query_refinement_identity_weight": float(
                self.query_refinement_identity_weight
            ),
            "query_refinement_alpha": float(alpha),
            "query_refinement_detach_localization": bool(
                self.query_refinement_detach_localization
            ),
            "query_refinement_cosine_mean": identity_metrics[
                "cosine_mean"
            ].detach(),
            "query_refinement_feature_delta_mean": identity_metrics[
                "feature_delta_mean"
            ].detach(),
            "localization_loss_source": "localization_query",
            "quality_loss_source": "score_refinement_query",
            "text_alignment_loss_source": "score_refinement_query",
        })
        return total_loss, result
