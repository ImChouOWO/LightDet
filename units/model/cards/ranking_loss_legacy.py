from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from units.model.cards.loss import (
    GroundingLoss,
    box_iou,
    clamp01,
    interpolate_value,
    schedule_progress,
)


class RankingGroundingLoss(GroundingLoss):
    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> "RankingGroundingLoss":
        criterion = super().from_config(config)

        loss_cfg = config.get("loss", config)
        ranking = dict(loss_cfg.get("ranking", {}))

        criterion.enable_pairwise_ranking = bool(
            ranking.get("enabled", False)
        )
        criterion.lambda_rank_default = max(
            0.0,
            float(ranking.get("lambda_rank", 0.0)),
        )
        criterion.rank_margin = max(
            0.0,
            float(ranking.get("rank_margin", 0.1)),
        )
        criterion.rank_min_quality_gap = max(
            0.0,
            float(
                ranking.get(
                    "rank_min_quality_gap",
                    0.1,
                )
            ),
        )
        criterion.rank_max_pairs = max(
            1,
            int(ranking.get("rank_max_pairs", 512)),
        )
        criterion.rank_start_epoch = max(
            1,
            int(ranking.get("rank_start_epoch", 5)),
        )
        criterion.rank_warmup_epoch = max(
            1,
            int(ranking.get("rank_warmup_epoch", 12)),
        )
        criterion.rank_alpha_min = clamp01(
            ranking.get("rank_alpha_min", 0.0)
        )
        criterion.rank_negative_iou_max = clamp01(
            ranking.get("rank_negative_iou_max", 0.2)
        )

        criterion.legacy_parameters[
            "enable_pairwise_ranking"
        ] = bool(criterion.enable_pairwise_ranking)

        return criterion

    def resolve_pairwise_rank_alpha(
        self,
        current_epoch: Optional[int],
    ) -> float:
        if not self.enable_pairwise_ranking:
            return 0.0

        progress = schedule_progress(
            current_epoch,
            self.rank_start_epoch,
            self.rank_warmup_epoch,
            curve="smoothstep",
        )

        return interpolate_value(
            self.rank_alpha_min,
            1.0,
            progress,
        )

    def _prepare_rank_text_negative_mask(
        self,
        pred_score_logit: torch.Tensor,
        text_negative_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = int(pred_score_logit.shape[0])

        if text_negative_mask is None:
            return torch.zeros(
                batch_size,
                device=pred_score_logit.device,
                dtype=torch.bool,
            )

        if not torch.is_tensor(text_negative_mask):
            text_negative_mask = torch.as_tensor(
                text_negative_mask,
                dtype=torch.bool,
            )

        return text_negative_mask.to(
            device=pred_score_logit.device,
            dtype=torch.bool,
        ).reshape(batch_size)

    def _max_iou_per_query(
        self,
        pred_bbox: torch.Tensor,
        targets: List[dict],
        text_negative_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_queries, _ = pred_bbox.shape
        result = torch.zeros(
            (batch_size, num_queries),
            device=pred_bbox.device,
            dtype=torch.float32,
        )

        pred_boxes = pred_bbox.detach().float()

        for batch_index, target in enumerate(targets):
            if bool(text_negative_mask[batch_index].item()):
                continue

            boxes = target.get("boxes")

            if boxes is None:
                continue

            if not torch.is_tensor(boxes):
                boxes = torch.as_tensor(
                    boxes,
                    dtype=torch.float32,
                    device=pred_bbox.device,
                )
            else:
                boxes = boxes.to(
                    device=pred_bbox.device,
                    dtype=torch.float32,
                )

            boxes = boxes.reshape(-1, 4)

            if boxes.numel() == 0:
                continue

            result[batch_index] = box_iou(
                pred_boxes[batch_index],
                boxes,
            ).max(dim=1).values

        return result

    def quality_pairwise_ranking_loss(
        self,
        *,
        pred_bbox: torch.Tensor,
        pred_score_logit: torch.Tensor,
        targets: List[dict],
        query_loss_weights: Optional[torch.Tensor],
        text_negative_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = pred_score_logit.new_zeros(())
        metrics = {
            "pair_count": zero,
            "violation_fraction": zero,
            "high_score_mean": zero,
            "low_score_mean": zero,
            "quality_gap_mean": zero,
        }

        logits = pred_score_logit.float().squeeze(-1)
        batch_size, num_queries = logits.shape

        if num_queries < 2:
            return zero, metrics

        negative_text_mask = (
            self._prepare_rank_text_negative_mask(
                pred_score_logit,
                text_negative_mask,
            )
        )
        quality = self._max_iou_per_query(
            pred_bbox,
            targets,
            negative_text_mask,
        )

        positive_rows = torch.tensor(
            [
                (
                    target.get("boxes") is not None
                    and len(target.get("boxes")) > 0
                )
                for target in targets
            ],
            device=logits.device,
            dtype=torch.bool,
        )
        valid_rows = positive_rows & (~negative_text_mask)

        high_quality = quality[:, :, None]
        low_quality = quality[:, None, :]
        quality_gap = high_quality - low_quality

        pair_mask = (
            valid_rows[:, None, None]
            & (
                low_quality
                <= float(self.rank_negative_iou_max)
            )
            & (
                quality_gap
                >= float(self.rank_min_quality_gap)
            )
        )

        if not bool(pair_mask.any()):
            return zero, metrics

        high_logit = logits[:, :, None]
        low_logit = logits[:, None, :]
        pair_loss = F.relu(
            float(self.rank_margin)
            + low_logit
            - high_logit
        )

        flat_loss = pair_loss.reshape(batch_size, -1)
        flat_mask = pair_mask.reshape(batch_size, -1)
        max_pairs = min(
            self.rank_max_pairs,
            int(flat_loss.shape[1]),
        )

        masked_loss = flat_loss.masked_fill(
            ~flat_mask,
            -torch.inf,
        )
        top_loss, top_index = torch.topk(
            masked_loss,
            k=max_pairs,
            dim=1,
            largest=True,
            sorted=True,
        )

        valid_count = flat_mask.sum(dim=1).clamp(
            max=max_pairs
        )
        selected = (
            torch.arange(
                max_pairs,
                device=logits.device,
            )[None, :]
            < valid_count[:, None]
        )

        if not bool(selected.any()):
            return zero, metrics

        high_index = torch.div(
            top_index,
            num_queries,
            rounding_mode="floor",
        )
        low_index = top_index.remainder(num_queries)

        selected_high_logit = torch.gather(
            logits,
            1,
            high_index,
        )
        selected_low_logit = torch.gather(
            logits,
            1,
            low_index,
        )
        selected_high_quality = torch.gather(
            quality,
            1,
            high_index,
        )
        selected_low_quality = torch.gather(
            quality,
            1,
            low_index,
        )

        sample_weight = self._prepare_query_loss_weights(
            pred_score_logit,
            query_loss_weights,
        ).reshape(batch_size, 1).float()
        pair_weight = sample_weight.expand_as(top_loss)

        selected_loss = top_loss[selected]
        selected_weight = pair_weight[selected]

        loss = (
            selected_loss * selected_weight
        ).sum() / selected_weight.sum().clamp_min(1.0)

        return loss.to(pred_score_logit.dtype), {
            "pair_count": pred_score_logit.new_tensor(
                float(selected.sum().item())
            ),
            "violation_fraction": (
                selected_loss > 0
            ).float().mean().to(pred_score_logit.dtype),
            "high_score_mean": selected_high_logit[
                selected
            ].sigmoid().mean().to(pred_score_logit.dtype),
            "low_score_mean": selected_low_logit[
                selected
            ].sigmoid().mean().to(pred_score_logit.dtype),
            "quality_gap_mean": (
                selected_high_quality[selected]
                - selected_low_quality[selected]
            ).mean().to(pred_score_logit.dtype),
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
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        base_loss, metrics = super().forward(
            pred_bbox=pred_bbox,
            pred_score_logit=pred_score_logit,
            targets=targets,
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
            lambda_rank=0.0,
            query_loss_weights=query_loss_weights,
            text_negative_mask=text_negative_mask,
            aux_pred_bbox=aux_pred_bbox,
            aux_pred_score_logit=aux_pred_score_logit,
            lambda_aux=lambda_aux,
            aux_lambda_bbox=aux_lambda_bbox,
            aux_lambda_giou=aux_lambda_giou,
            aux_lambda_score=aux_lambda_score,
            aux_pos_weight=aux_pos_weight,
        )

        rank_alpha_value = self.resolve_pairwise_rank_alpha(
            current_epoch
        )
        lambda_rank_value = max(0.0, float(lambda_rank))
        active = bool(
            self.enable_pairwise_ranking
            and lambda_rank_value > 0.0
            and rank_alpha_value > 0.0
        )

        if active:
            rank_raw, rank_metrics = (
                self.quality_pairwise_ranking_loss(
                    pred_bbox=pred_bbox,
                    pred_score_logit=pred_score_logit,
                    targets=targets,
                    query_loss_weights=query_loss_weights,
                    text_negative_mask=text_negative_mask,
                )
            )
        else:
            rank_raw = pred_score_logit.new_zeros(())
            rank_metrics = {
                "pair_count": rank_raw,
                "violation_fraction": rank_raw,
                "high_score_mean": rank_raw,
                "low_score_mean": rank_raw,
                "quality_gap_mean": rank_raw,
            }

        lambda_rank_eff = (
            lambda_rank_value * rank_alpha_value
        )
        rank_contrib = lambda_rank_eff * rank_raw
        total_loss = base_loss + rank_contrib

        result = dict(metrics)
        result.pop("legacy_lambda_rank_ignored", None)

        result.update({
            "loss": total_loss.detach(),
            "loss_main_total": (
                result.get("loss_main_total", base_loss.detach())
                + rank_contrib.detach()
            ),
            "loss_rank": rank_contrib.detach(),
            "loss_rank_raw": rank_raw.detach(),
            "loss_rank_contrib": rank_contrib.detach(),
            "pairwise_ranking_enabled": bool(
                self.enable_pairwise_ranking
            ),
            "pairwise_ranking_active": active,
            "lambda_rank": lambda_rank_value,
            "lambda_rank_eff": lambda_rank_eff,
            "rank_alpha": rank_alpha_value,
            "rank_pair_count": rank_metrics[
                "pair_count"
            ].detach(),
            "rank_violation_fraction": rank_metrics[
                "violation_fraction"
            ].detach(),
            "rank_high_score_mean": rank_metrics[
                "high_score_mean"
            ].detach(),
            "rank_low_score_mean": rank_metrics[
                "low_score_mean"
            ].detach(),
            "rank_quality_gap_mean": rank_metrics[
                "quality_gap_mean"
            ].detach(),
            "rank_negative_iou_max": float(
                self.rank_negative_iou_max
            ),
        })

        return total_loss, result


def build_grounding_loss_from_config(
    config: Dict[str, Any],
) -> RankingGroundingLoss:
    return RankingGroundingLoss.from_config(config)
