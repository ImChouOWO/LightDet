from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from . import loss_legacy as _legacy
from .loss_legacy import *  # noqa: F401,F403


class GroundingLoss(_legacy.GroundingLoss):
    """Grounding loss with localization-quality/text-alignment decoupling.

    Existing DETR matching, IoU-aware quality classification, duplicate ranking,
    hard-negative mining and H-DETR auxiliary regression remain unchanged. The
    only behavioral separation is:

    - localization and quality losses consume positive-text rows only;
    - text-negative rows supervise only ``text_alignment_logit``;
    - Hungarian matchers receive ``quality_logit`` rather than the fused score;
    - inference/evaluation still consume the fused score from the model.
    """

    def __init__(
        self,
        *args: Any,
        text_alignment_enabled: bool = True,
        text_alignment_loss_weight: float = 1.0,
        text_alignment_positive_iou_threshold: float = 0.50,
        text_alignment_negative_iou_threshold: float = 0.20,
        text_alignment_negative_weight: float = 0.25,
        text_alignment_negative_text_weight: float = 1.0,
        text_alignment_focal_gamma: float = 2.0,
        text_alignment_start_epoch: int = 1,
        text_alignment_warmup_epoch: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._set_text_alignment_options(
            enabled=text_alignment_enabled,
            loss_weight=text_alignment_loss_weight,
            positive_iou_threshold=(
                text_alignment_positive_iou_threshold
            ),
            negative_iou_threshold=(
                text_alignment_negative_iou_threshold
            ),
            negative_weight=text_alignment_negative_weight,
            negative_text_weight=(
                text_alignment_negative_text_weight
            ),
            focal_gamma=text_alignment_focal_gamma,
            start_epoch=text_alignment_start_epoch,
            warmup_epoch=text_alignment_warmup_epoch,
        )

    def _set_text_alignment_options(
        self,
        *,
        enabled: bool,
        loss_weight: float,
        positive_iou_threshold: float,
        negative_iou_threshold: float,
        negative_weight: float,
        negative_text_weight: float,
        focal_gamma: float,
        start_epoch: int,
        warmup_epoch: int,
    ) -> None:
        positive_iou_threshold = _legacy.clamp01(
            positive_iou_threshold
        )
        negative_iou_threshold = _legacy.clamp01(
            negative_iou_threshold
        )
        if negative_iou_threshold >= positive_iou_threshold:
            raise ValueError(
                "text_alignment.negative_iou_threshold must be lower than "
                "text_alignment.positive_iou_threshold"
            )

        self.text_alignment_enabled = bool(enabled)
        self.text_alignment_loss_weight = max(
            0.0,
            float(loss_weight),
        )
        self.text_alignment_positive_iou_threshold = float(
            positive_iou_threshold
        )
        self.text_alignment_negative_iou_threshold = float(
            negative_iou_threshold
        )
        self.text_alignment_negative_weight = max(
            0.0,
            float(negative_weight),
        )
        self.text_alignment_negative_text_weight = max(
            0.0,
            float(negative_text_weight),
        )
        self.text_alignment_focal_gamma = max(
            0.0,
            float(focal_gamma),
        )
        self.text_alignment_start_epoch = max(1, int(start_epoch))
        self.text_alignment_warmup_epoch = max(1, int(warmup_epoch))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GroundingLoss":
        # Call the legacy implementation with ``cls`` so subclasses such as
        # RankingGroundingLoss are constructed correctly.
        criterion = _legacy.GroundingLoss.from_config.__func__(
            cls,
            config,
        )

        loss_cfg = config.get("loss", config)
        alignment = dict(loss_cfg.get("text_alignment", {}))
        criterion._set_text_alignment_options(
            enabled=alignment.get("enabled", True),
            loss_weight=alignment.get("loss_weight", 1.0),
            positive_iou_threshold=alignment.get(
                "positive_iou_threshold",
                0.50,
            ),
            negative_iou_threshold=alignment.get(
                "negative_iou_threshold",
                0.20,
            ),
            negative_weight=alignment.get("negative_weight", 0.25),
            negative_text_weight=alignment.get(
                "negative_text_weight",
                1.0,
            ),
            focal_gamma=alignment.get("focal_gamma", 2.0),
            start_epoch=alignment.get("start_epoch", 1),
            warmup_epoch=alignment.get("warmup_epoch", 5),
        )
        return criterion

    @staticmethod
    def _extract_score_components(
        pred_score_logit: torch.Tensor,
        pred_quality_logit: Optional[torch.Tensor],
        pred_text_alignment_logit: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        return quality, alignment

    @staticmethod
    def _slice_optional_batch(
        value: Optional[torch.Tensor],
        indices: Sequence[int],
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        index = torch.as_tensor(
            list(indices),
            dtype=torch.long,
            device=value.device,
        )
        return value.index_select(0, index)

    def _alignment_targets(
        self,
        *,
        pred_bbox: torch.Tensor,
        targets: List[dict],
        text_negative_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_queries, _ = pred_bbox.shape
        target = pred_bbox.new_zeros(
            (batch_size, num_queries, 1),
            dtype=torch.float32,
        )
        valid = torch.zeros_like(target, dtype=torch.bool)
        max_iou = pred_bbox.new_zeros(
            (batch_size, num_queries, 1),
            dtype=torch.float32,
        )

        boxes = pred_bbox.detach().float()
        negative_rows = text_negative_mask.detach().cpu().tolist()

        for batch_index, (is_negative, row_target) in enumerate(
            zip(negative_rows, targets)
        ):
            if is_negative:
                valid[batch_index, :, 0] = True
                continue

            gt_boxes = row_target.get("boxes")
            if gt_boxes is None:
                valid[batch_index, :, 0] = True
                continue

            if not torch.is_tensor(gt_boxes):
                gt_boxes = torch.as_tensor(
                    gt_boxes,
                    dtype=torch.float32,
                    device=pred_bbox.device,
                )
            else:
                gt_boxes = gt_boxes.to(
                    device=pred_bbox.device,
                    dtype=torch.float32,
                )
            gt_boxes = gt_boxes.reshape(-1, 4)

            if gt_boxes.numel() == 0 or num_queries == 0:
                valid[batch_index, :, 0] = True
                continue

            row_iou = _legacy.box_iou(
                boxes[batch_index],
                gt_boxes,
            ).max(dim=1).values
            max_iou[batch_index, :, 0] = row_iou

            positive = row_iou >= float(
                self.text_alignment_positive_iou_threshold
            )
            negative = row_iou <= float(
                self.text_alignment_negative_iou_threshold
            )

            # Always retain one semantic positive for a valid positive-text row,
            # even early in training when no box has reached the IoU threshold.
            if not bool(positive.any()):
                best_index = torch.argmax(row_iou)
                positive[best_index] = True
                negative[best_index] = False

            target[batch_index, positive, 0] = 1.0
            valid[batch_index, positive | negative, 0] = True

        return target, valid, max_iou

    def text_alignment_loss(
        self,
        *,
        pred_bbox: torch.Tensor,
        pred_text_alignment_logit: torch.Tensor,
        targets: List[dict],
        text_negative_mask: torch.Tensor,
        query_loss_weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._validate_branch_inputs(
            "text_alignment",
            pred_bbox,
            pred_text_alignment_logit,
            targets,
        )

        logits = pred_text_alignment_logit.float()
        probability = logits.sigmoid()
        target, valid_mask, max_iou = self._alignment_targets(
            pred_bbox=pred_bbox,
            targets=targets,
            text_negative_mask=text_negative_mask,
        )

        zero = logits.new_zeros(())
        if not bool(valid_mask.any()):
            return zero.to(pred_text_alignment_logit.dtype), {
                "positive_score_mean": zero,
                "negative_score_mean": zero,
                "positive_top1_score": zero,
                "negative_text_top1_score": zero,
                "positive_negative_margin": zero,
                "negative_text_loss": zero,
                "positive_count": zero,
                "negative_count": zero,
                "ignored_count": logits.new_tensor(
                    float(valid_mask.numel())
                ),
                "max_iou_mean": zero,
            }

        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        modulation = (target - probability).abs().pow(
            float(self.text_alignment_focal_gamma)
        )
        element_loss = bce * modulation

        positive_mask = valid_mask & (target > 0.5)
        negative_mask = valid_mask & (~positive_mask)
        element_weight = torch.where(
            positive_mask,
            torch.ones_like(element_loss),
            torch.full_like(
                element_loss,
                float(self.text_alignment_negative_weight),
            ),
        )
        if bool(text_negative_mask.any()):
            element_weight[text_negative_mask, :, :] = float(
                self.text_alignment_negative_text_weight
            )

        row_weight = self._prepare_query_loss_weights(
            pred_text_alignment_logit,
            query_loss_weights,
        ).float().expand_as(element_loss)
        combined_weight = element_weight * row_weight
        weighted_loss = element_loss * combined_weight

        denominator = combined_weight[valid_mask].sum().clamp_min(1.0)
        loss = weighted_loss[valid_mask].sum() / denominator

        negative_text_element_mask = (
            text_negative_mask[:, None, None] & valid_mask
        )
        if bool(negative_text_element_mask.any()):
            negative_text_denominator = combined_weight[
                negative_text_element_mask
            ].sum().clamp_min(1.0)
            negative_text_loss = weighted_loss[
                negative_text_element_mask
            ].sum() / negative_text_denominator
            negative_text_top1 = probability[
                text_negative_mask
            ].squeeze(-1).max(dim=1).values.mean()
        else:
            negative_text_loss = zero
            negative_text_top1 = zero

        positive_rows = ~text_negative_mask
        if bool(positive_rows.any()):
            positive_top1 = probability[
                positive_rows
            ].squeeze(-1).max(dim=1).values.mean()
        else:
            positive_top1 = zero

        positive_score_mean = (
            probability[positive_mask].mean()
            if bool(positive_mask.any())
            else zero
        )
        negative_score_mean = (
            probability[negative_mask].mean()
            if bool(negative_mask.any())
            else zero
        )
        max_iou_mean = (
            max_iou[positive_mask].mean()
            if bool(positive_mask.any())
            else zero
        )

        return loss.to(pred_text_alignment_logit.dtype), {
            "positive_score_mean": positive_score_mean.to(
                pred_text_alignment_logit.dtype
            ),
            "negative_score_mean": negative_score_mean.to(
                pred_text_alignment_logit.dtype
            ),
            "positive_top1_score": positive_top1.to(
                pred_text_alignment_logit.dtype
            ),
            "negative_text_top1_score": negative_text_top1.to(
                pred_text_alignment_logit.dtype
            ),
            "positive_negative_margin": (
                positive_top1 - negative_text_top1
            ).to(pred_text_alignment_logit.dtype),
            "negative_text_loss": negative_text_loss.to(
                pred_text_alignment_logit.dtype
            ),
            "positive_count": pred_text_alignment_logit.new_tensor(
                float(positive_mask.sum().item())
            ),
            "negative_count": pred_text_alignment_logit.new_tensor(
                float(negative_mask.sum().item())
            ),
            "ignored_count": pred_text_alignment_logit.new_tensor(
                float((~valid_mask).sum().item())
            ),
            "max_iou_mean": max_iou_mean.to(
                pred_text_alignment_logit.dtype
            ),
        }

    def forward(
        self,
        pred_bbox: torch.Tensor,
        pred_score_logit: torch.Tensor,
        targets: List[dict],
        lambda_bbox: float = 5.0,
        lambda_giou: float = 2.0,
        lambda_score: float = 2.0,
        pos_weight: float = 1.0,
        current_epoch=None,
        total_epochs: Optional[int] = None,
        quality_alpha=None,
        rank_alpha=None,
        quality_warmup_epoch: Optional[int] = None,
        rank_start_epoch: Optional[int] = None,
        rank_warmup_epoch: Optional[int] = None,
        rank_alpha_min: Optional[float] = None,
        lambda_rank: float = 0.0,
        query_loss_weights: Optional[torch.Tensor] = None,
        text_negative_mask: Optional[torch.Tensor] = None,
        aux_pred_bbox: Optional[torch.Tensor] = None,
        aux_pred_score_logit: Optional[torch.Tensor] = None,
        lambda_aux: Optional[float] = None,
        aux_lambda_bbox: Optional[float] = None,
        aux_lambda_giou: Optional[float] = None,
        aux_lambda_score: Optional[float] = None,
        aux_pos_weight: Optional[float] = None,
        pred_quality_logit: Optional[torch.Tensor] = None,
        pred_text_alignment_logit: Optional[torch.Tensor] = None,
        aux_pred_quality_logit: Optional[torch.Tensor] = None,
        aux_pred_text_alignment_logit: Optional[torch.Tensor] = None,
        lambda_text_alignment: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        quality_logit, alignment_logit = self._extract_score_components(
            pred_score_logit,
            pred_quality_logit,
            pred_text_alignment_logit,
        )

        aux_quality_logit = None
        aux_alignment_logit = None
        if aux_pred_score_logit is not None:
            (
                aux_quality_logit,
                aux_alignment_logit,
            ) = self._extract_score_components(
                aux_pred_score_logit,
                aux_pred_quality_logit,
                aux_pred_text_alignment_logit,
            )

        prepared_negative_mask = self._prepare_text_negative_mask(
            pred_score_logit,
            text_negative_mask,
        )
        localization_indices = torch.nonzero(
            ~prepared_negative_mask,
            as_tuple=False,
        ).flatten().detach().cpu().tolist()

        localization_bbox = pred_bbox[localization_indices]
        localization_quality = quality_logit[localization_indices]
        localization_targets = [
            targets[index] for index in localization_indices
        ]
        localization_query_weights = self._slice_optional_batch(
            query_loss_weights,
            localization_indices,
        )
        localization_negative_mask = torch.zeros(
            len(localization_indices),
            dtype=torch.bool,
            device=quality_logit.device,
        )

        localization_aux_bbox = (
            aux_pred_bbox[localization_indices]
            if aux_pred_bbox is not None
            else None
        )
        localization_aux_quality = (
            aux_quality_logit[localization_indices]
            if aux_quality_logit is not None
            else None
        )

        base_loss, metrics = super().forward(
            pred_bbox=localization_bbox,
            pred_score_logit=localization_quality,
            targets=localization_targets,
            lambda_bbox=lambda_bbox,
            lambda_giou=lambda_giou,
            lambda_score=lambda_score,
            pos_weight=pos_weight,
            current_epoch=current_epoch,
            total_epochs=total_epochs,
            quality_alpha=quality_alpha,
            rank_alpha=rank_alpha,
            quality_warmup_epoch=quality_warmup_epoch,
            rank_start_epoch=rank_start_epoch,
            rank_warmup_epoch=rank_warmup_epoch,
            rank_alpha_min=rank_alpha_min,
            lambda_rank=lambda_rank,
            query_loss_weights=localization_query_weights,
            text_negative_mask=localization_negative_mask,
            aux_pred_bbox=localization_aux_bbox,
            aux_pred_score_logit=localization_aux_quality,
            lambda_aux=lambda_aux,
            aux_lambda_bbox=aux_lambda_bbox,
            aux_lambda_giou=aux_lambda_giou,
            aux_lambda_score=aux_lambda_score,
            aux_pos_weight=aux_pos_weight,
        )

        zero = pred_bbox.new_zeros(())
        alignment_active = bool(
            self.text_alignment_enabled
            and alignment_logit is not None
            and self.text_alignment_loss_weight > 0.0
        )
        alignment_alpha = (
            _legacy.schedule_progress(
                current_epoch,
                self.text_alignment_start_epoch,
                self.text_alignment_warmup_epoch,
                curve="smoothstep",
            )
            if alignment_active
            else 0.0
        )

        if alignment_active:
            alignment_loss, alignment_metrics = self.text_alignment_loss(
                pred_bbox=pred_bbox,
                pred_text_alignment_logit=alignment_logit,
                targets=targets,
                text_negative_mask=prepared_negative_mask,
                query_loss_weights=query_loss_weights,
            )
        else:
            alignment_loss = zero
            alignment_metrics = {
                "positive_score_mean": zero,
                "negative_score_mean": zero,
                "positive_top1_score": zero,
                "negative_text_top1_score": zero,
                "positive_negative_margin": zero,
                "negative_text_loss": zero,
                "positive_count": zero,
                "negative_count": zero,
                "ignored_count": zero,
                "max_iou_mean": zero,
            }

        alignment_weight = (
            self.text_alignment_loss_weight
            if lambda_text_alignment is None
            else max(0.0, float(lambda_text_alignment))
        )
        alignment_contrib = (
            float(alignment_weight)
            * float(alignment_alpha)
            * alignment_loss
        )
        total_loss = base_loss + alignment_contrib

        result = dict(metrics)
        result.update({
            "loss": total_loss.detach(),
            "loss_main_total": (
                result.get("loss_main_total", base_loss.detach())
                + alignment_contrib.detach()
            ),
            "loss_quality": result.get(
                "loss_score",
                zero,
            ).detach(),
            "loss_text_alignment": alignment_loss.detach(),
            "loss_text_alignment_contrib": alignment_contrib.detach(),
            "loss_text_negative": alignment_metrics[
                "negative_text_loss"
            ].detach(),
            "loss_text_negative_contrib": (
                float(alignment_weight)
                * float(alignment_alpha)
                * alignment_metrics["negative_text_loss"]
            ).detach(),
            "text_alignment_enabled": bool(
                self.text_alignment_enabled
            ),
            "text_alignment_active": alignment_active,
            "text_alignment_alpha": float(alignment_alpha),
            "lambda_text_alignment": float(alignment_weight),
            "text_alignment_positive_iou_threshold": float(
                self.text_alignment_positive_iou_threshold
            ),
            "text_alignment_negative_iou_threshold": float(
                self.text_alignment_negative_iou_threshold
            ),
            "text_alignment_positive_score_mean": alignment_metrics[
                "positive_score_mean"
            ].detach(),
            "text_alignment_negative_score_mean": alignment_metrics[
                "negative_score_mean"
            ].detach(),
            "text_alignment_positive_top1_score": alignment_metrics[
                "positive_top1_score"
            ].detach(),
            "text_alignment_negative_top1_score": alignment_metrics[
                "negative_text_top1_score"
            ].detach(),
            "text_alignment_margin": alignment_metrics[
                "positive_negative_margin"
            ].detach(),
            "text_alignment_positive_count": alignment_metrics[
                "positive_count"
            ].detach(),
            "text_alignment_negative_count": alignment_metrics[
                "negative_count"
            ].detach(),
            "text_alignment_ignored_count": alignment_metrics[
                "ignored_count"
            ].detach(),
            "text_alignment_positive_iou_mean": alignment_metrics[
                "max_iou_mean"
            ].detach(),
            "localization_query_rows": float(
                len(localization_indices)
            ),
            "text_negative_query_rows": float(
                prepared_negative_mask.sum().item()
            ),
            "score_decoupled": True,
            "matcher_score_source": "quality_logit",
            "final_score_source": "quality_x_text_alignment",
            "negative_text_regression_matches": 0.0,
        })
        return total_loss, result
