import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def clamp01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


def smoothstep(x: float) -> float:
    """Smooth 0 -> 1 warmup curve."""
    x = clamp01(x)
    return x * x * (3.0 - 2.0 * x)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """boxes: [..., 4] in xyxy format."""
    return (
        (boxes[..., 2] - boxes[..., 0]).clamp(min=0)
        * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
    )


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for xyxy boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)

    intersection = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1e-6)


def generalized_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """Pairwise generalized IoU for xyxy boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)

    intersection = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - intersection
    iou = intersection / union.clamp(min=1e-6)

    lt_cover = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_cover = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_cover = (rb_cover - lt_cover).clamp(min=0)
    area_cover = wh_cover[..., 0] * wh_cover[..., 1]

    return iou - (area_cover - union) / area_cover.clamp(min=1e-6)


class OneToManyMatcher:
    """
    One-to-many assignment shared by bbox, GIoU and score supervision.

    A prediction can be assigned to only one GT. Each reachable GT first gets a
    greedy one-to-one match, after which extra low-cost predictions are added up
    to max_positive_per_gt and the global positive budget.
    """

    def __init__(
        self,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cost_score: float = 0.0,
        positive_ratio: float = 0.05,
        max_positive_per_gt: int = 2,
        min_extra_positive_iou: float = 0.0,
    ) -> None:
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)
        self.cost_score = float(cost_score)
        self.positive_ratio = float(positive_ratio)
        self.max_positive_per_gt = max(1, int(max_positive_per_gt))
        self.min_extra_positive_iou = float(min_extra_positive_iou)

    @staticmethod
    @torch.no_grad()
    def _greedy_one_to_one(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_pred, num_gt = cost.shape
        device = cost.device

        if num_pred == 0 or num_gt == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        work = cost.float().clone()
        infinity = torch.finfo(work.dtype).max
        pred_indices: List[torch.Tensor] = []
        gt_indices: List[torch.Tensor] = []

        for _ in range(min(num_pred, num_gt)):
            flat_index = torch.argmin(work.reshape(-1))
            pred_index = torch.div(flat_index, num_gt, rounding_mode="floor")
            gt_index = flat_index % num_gt

            pred_indices.append(pred_index)
            gt_indices.append(gt_index)
            work[pred_index, :] = infinity
            work[:, gt_index] = infinity

        return torch.stack(pred_indices).long(), torch.stack(gt_indices).long()

    @torch.no_grad()
    def __call__(
        self,
        pred_bbox: torch.Tensor,
        pred_score_logit: torch.Tensor,
        targets: List[dict],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, num_pred, _ = pred_bbox.shape
        pred_score = pred_score_logit.sigmoid().squeeze(-1)
        assignments: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for batch_index in range(batch_size):
            gt_bbox = targets[batch_index]["boxes"].to(
                device=pred_bbox.device,
                dtype=pred_bbox.dtype,
                non_blocking=True,
            )
            num_gt = int(gt_bbox.shape[0])

            if num_gt == 0 or num_pred == 0:
                empty = torch.empty(0, dtype=torch.long, device=pred_bbox.device)
                assignments.append((empty, empty))
                continue

            boxes = pred_bbox[batch_index].detach()
            cost_bbox = torch.cdist(boxes, gt_bbox, p=1)
            giou_matrix = generalized_box_iou(boxes, gt_bbox)
            iou_matrix = box_iou(boxes, gt_bbox)

            cost = self.cost_bbox * cost_bbox - self.cost_giou * giou_matrix
            if self.cost_score > 0:
                cost = cost - self.cost_score * pred_score[batch_index, :, None]

            minimum_budget = min(num_pred, num_gt)
            ratio_budget = max(1, int(round(num_pred * self.positive_ratio)))
            maximum_budget = min(num_pred, num_gt * self.max_positive_per_gt)
            positive_budget = min(
                max(minimum_budget, ratio_budget),
                maximum_budget,
            )

            base_pred, base_gt = self._greedy_one_to_one(cost)
            selected_pred: List[int] = []
            selected_gt: List[int] = []
            used_pred = torch.zeros(num_pred, dtype=torch.bool, device=cost.device)
            gt_counts = torch.zeros(num_gt, dtype=torch.long, device=cost.device)

            for pred_index_tensor, gt_index_tensor in zip(base_pred, base_gt):
                pred_index = int(pred_index_tensor.item())
                gt_index = int(gt_index_tensor.item())
                selected_pred.append(pred_index)
                selected_gt.append(gt_index)
                used_pred[pred_index] = True
                gt_counts[gt_index] += 1

            if len(selected_pred) < positive_budget and self.max_positive_per_gt > 1:
                candidate_pred: List[torch.Tensor] = []
                candidate_gt: List[torch.Tensor] = []
                candidate_cost: List[torch.Tensor] = []

                candidate_k = min(num_pred, self.max_positive_per_gt)
                for gt_index in range(num_gt):
                    values, pred_indices = torch.topk(
                        cost[:, gt_index],
                        k=candidate_k,
                        largest=False,
                    )
                    candidate_pred.append(pred_indices)
                    candidate_gt.append(
                        torch.full_like(pred_indices, fill_value=gt_index)
                    )
                    candidate_cost.append(values)

                all_pred = torch.cat(candidate_pred)
                all_gt = torch.cat(candidate_gt)
                all_cost = torch.cat(candidate_cost)

                for order_index_tensor in torch.argsort(all_cost):
                    if len(selected_pred) >= positive_budget:
                        break

                    order_index = int(order_index_tensor.item())
                    pred_index = int(all_pred[order_index].item())
                    gt_index = int(all_gt[order_index].item())

                    if used_pred[pred_index]:
                        continue
                    if int(gt_counts[gt_index].item()) >= self.max_positive_per_gt:
                        continue
                    if (
                        self.min_extra_positive_iou > 0
                        and float(iou_matrix[pred_index, gt_index].item())
                        < self.min_extra_positive_iou
                    ):
                        continue

                    selected_pred.append(pred_index)
                    selected_gt.append(gt_index)
                    used_pred[pred_index] = True
                    gt_counts[gt_index] += 1

            assignments.append(
                (
                    torch.tensor(
                        selected_pred,
                        dtype=torch.long,
                        device=cost.device,
                    ),
                    torch.tensor(
                        selected_gt,
                        dtype=torch.long,
                        device=cost.device,
                    ),
                )
            )

        return assignments


class GroundingLoss(nn.Module):
    """
    Aligned one-to-many detection loss with Quality Focal Loss.

    JSON text-negative queries carry no GT boxes. Their score targets are all
    zero, while query_loss_weights increases their QFL negative contribution.
    This reinforces text-image consistency without an additional model forward.
    """

    def __init__(
        self,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cost_score: float = 0.0,
        hard_negative_ratio: int = 5,
        positive_ratio: float = 0.05,
        max_positive_per_gt: int = 2,
        aux_positive_label: float = 0.7,
        expand_cost_bbox: float = 5.0,
        expand_cost_giou: float = 2.0,
        iou_pos_thr: float = 0.0,
        quality_min: float = 0.0,
        quality_max: float = 1.0,
        qfl_beta: float = 2.0,
        rank_margin: float = 0.1,
        rank_min_quality_gap: float = 0.1,
        rank_max_pairs: int = 512,
        max_query_loss_weight: float = 10.0,
    ) -> None:
        super().__init__()

        # Kept for compatibility with the current train.py/config schema.
        self.hard_negative_ratio = int(hard_negative_ratio)
        self.aux_positive_label = float(aux_positive_label)
        self.quality_min = float(quality_min)
        self.quality_max = float(quality_max)

        self.matcher = OneToManyMatcher(
            cost_bbox=(
                expand_cost_bbox if expand_cost_bbox is not None else cost_bbox
            ),
            cost_giou=(
                expand_cost_giou if expand_cost_giou is not None else cost_giou
            ),
            cost_score=cost_score,
            positive_ratio=positive_ratio,
            max_positive_per_gt=max_positive_per_gt,
            min_extra_positive_iou=iou_pos_thr,
        )

        self.qfl_beta = float(qfl_beta)
        self.rank_margin = float(rank_margin)
        self.rank_min_quality_gap = float(rank_min_quality_gap)
        self.rank_max_pairs = int(rank_max_pairs)
        self.max_query_loss_weight = max(1.0, float(max_query_loss_weight))

    def resolve_epoch_alpha(
        self,
        current_epoch=None,
        quality_alpha=None,
        rank_alpha=None,
        quality_warmup_epoch: int = 10,
        rank_start_epoch: int = 30,
        rank_warmup_epoch: int = 60,
        rank_alpha_min: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Quality target gradually changes from 1 to IoU.

        Ranking starts at rank_start_epoch, reaches the configured lambda_rank at
        rank_start_epoch + rank_warmup_epoch, and then remains fixed.
        """
        if quality_alpha is None:
            if current_epoch is None or quality_warmup_epoch <= 0:
                quality_alpha = 1.0
            else:
                quality_alpha = clamp01(
                    float(current_epoch) / float(quality_warmup_epoch)
                )

        if rank_alpha is None:
            if current_epoch is None:
                rank_alpha = 1.0
            elif float(current_epoch) < float(rank_start_epoch):
                rank_alpha = 0.0
            else:
                progress = (
                    float(current_epoch) - float(rank_start_epoch)
                ) / max(float(rank_warmup_epoch), 1.0)
                rank_alpha = float(rank_alpha_min) + (
                    1.0 - float(rank_alpha_min)
                ) * smoothstep(progress)
                rank_alpha = clamp01(rank_alpha)

        return float(quality_alpha), float(rank_alpha)

    def _prepare_query_loss_weights(
        self,
        pred_score_logit: torch.Tensor,
        query_loss_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = int(pred_score_logit.shape[0])

        if query_loss_weights is None:
            return pred_score_logit.new_ones((batch_size, 1, 1))

        if not torch.is_tensor(query_loss_weights):
            query_loss_weights = torch.as_tensor(
                query_loss_weights,
                dtype=torch.float32,
            )

        query_loss_weights = query_loss_weights.to(
            device=pred_score_logit.device,
            dtype=pred_score_logit.dtype,
            non_blocking=True,
        ).reshape(-1)

        if query_loss_weights.numel() != batch_size:
            raise ValueError(
                "query_loss_weights size mismatch: "
                f"{query_loss_weights.numel()} != {batch_size}"
            )

        return query_loss_weights.clamp(
            min=1.0,
            max=self.max_query_loss_weight,
        ).view(batch_size, 1, 1)

    def _prepare_text_negative_mask(
        self,
        pred_score_logit: torch.Tensor,
        text_negative_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = int(pred_score_logit.shape[0])

        if text_negative_mask is None:
            return torch.zeros(
                (batch_size,),
                dtype=torch.bool,
                device=pred_score_logit.device,
            )

        if not torch.is_tensor(text_negative_mask):
            text_negative_mask = torch.as_tensor(
                text_negative_mask,
                dtype=torch.bool,
            )

        text_negative_mask = text_negative_mask.to(
            device=pred_score_logit.device,
            dtype=torch.bool,
            non_blocking=True,
        ).reshape(-1)

        if text_negative_mask.numel() != batch_size:
            raise ValueError(
                "text_negative_mask size mismatch: "
                f"{text_negative_mask.numel()} != {batch_size}"
            )

        return text_negative_mask

    def score_loss_quality_balanced(
        self,
        pred_score_logit: torch.Tensor,
        score_target: torch.Tensor,
        positive_mask: torch.Tensor,
        pos_weight: float = 1.0,
        qfl_beta=None,
        query_loss_weights: Optional[torch.Tensor] = None,
        text_negative_mask: Optional[torch.Tensor] = None,
    ):
        """
        Quality Focal Loss with per-query negative weighting.

        The weighted negative mean deliberately divides by the number of
        negative elements, not by the sum of weights. Therefore a severe query
        with weight 8 contributes eight times the negative gradient.
        """
        if qfl_beta is None:
            qfl_beta = self.qfl_beta

        pred_prob = pred_score_logit.sigmoid()
        bce = F.binary_cross_entropy_with_logits(
            pred_score_logit,
            score_target,
            reduction="none",
        )
        qfl_weight = (score_target - pred_prob).abs().pow(float(qfl_beta))
        loss_all = bce * qfl_weight

        query_weight = self._prepare_query_loss_weights(
            pred_score_logit,
            query_loss_weights,
        )
        text_negative_mask = self._prepare_text_negative_mask(
            pred_score_logit,
            text_negative_mask,
        )

        negative_mask = ~positive_mask
        has_pos = bool(positive_mask.any())
        has_neg = bool(negative_mask.any())

        loss_pos = (
            loss_all[positive_mask].mean()
            if has_pos
            else pred_score_logit.new_tensor(0.0)
        )

        weighted_loss_all = loss_all * query_weight
        loss_neg = (
            weighted_loss_all[negative_mask].mean()
            if has_neg
            else pred_score_logit.new_tensor(0.0)
        )
        loss_neg_unweighted = (
            loss_all[negative_mask].mean()
            if has_neg
            else pred_score_logit.new_tensor(0.0)
        )

        if bool(text_negative_mask.any()):
            text_negative_loss = weighted_loss_all[text_negative_mask].mean()
            text_negative_count = float(text_negative_mask.sum().item())
            text_negative_weight_mean = query_weight[
                text_negative_mask
            ].mean()
        else:
            text_negative_loss = pred_score_logit.new_tensor(0.0)
            text_negative_count = 0.0
            text_negative_weight_mean = pred_score_logit.new_tensor(1.0)

        if has_pos and has_neg:
            loss_score = (
                float(pos_weight) * loss_pos + loss_neg
            ) / (float(pos_weight) + 1.0)
        elif has_pos:
            loss_score = loss_pos
        elif has_neg:
            loss_score = loss_neg
        else:
            loss_score = pred_score_logit.new_tensor(0.0)

        return (
            loss_score,
            loss_pos,
            loss_neg,
            loss_neg_unweighted,
            text_negative_loss,
            float(positive_mask.sum().item()),
            float(negative_mask.sum().item()),
            text_negative_count,
            text_negative_weight_mean,
        )

    def pairwise_quality_rank_loss(
        self,
        pred_score_logit: torch.Tensor,
        score_target: torch.Tensor,
        positive_mask: torch.Tensor,
        margin=None,
        min_quality_gap=None,
    ) -> torch.Tensor:
        if margin is None:
            margin = self.rank_margin
        if min_quality_gap is None:
            min_quality_gap = self.rank_min_quality_gap

        score = pred_score_logit.sigmoid().reshape(-1)
        quality = score_target.reshape(-1)
        positive_flat = positive_mask.reshape(-1)
        negative_flat = ~positive_flat

        positive_indices = torch.nonzero(
            positive_flat,
            as_tuple=False,
        ).flatten()
        negative_indices = torch.nonzero(
            negative_flat,
            as_tuple=False,
        ).flatten()
        losses = []

        if positive_indices.numel() >= 2:
            positive_quality = quality[positive_indices]
            positive_score = score[positive_indices]
            quality_i = positive_quality[:, None]
            quality_j = positive_quality[None, :]
            rank_mask = quality_i > quality_j + float(min_quality_gap)

            if rank_mask.any():
                score_difference = (
                    positive_score[:, None] - positive_score[None, :]
                )
                loss_positive_pair = F.relu(
                    float(margin) - score_difference
                )[rank_mask]

                if loss_positive_pair.numel() > self.rank_max_pairs:
                    keep = torch.randperm(
                        loss_positive_pair.numel(),
                        device=loss_positive_pair.device,
                    )[: self.rank_max_pairs]
                    loss_positive_pair = loss_positive_pair[keep]
                losses.append(loss_positive_pair.mean())

        if positive_indices.numel() > 0 and negative_indices.numel() > 0:
            positive_score = score[positive_indices]
            negative_score = score[negative_indices]
            hard_negative_k = min(
                int(negative_score.numel()),
                max(
                    int(positive_indices.numel())
                    * max(self.hard_negative_ratio, 1),
                    1,
                ),
            )
            hard_negative_score = torch.topk(
                negative_score,
                k=hard_negative_k,
                largest=True,
            ).values

            loss_positive_negative = F.relu(
                float(margin)
                - (
                    positive_score[:, None]
                    - hard_negative_score[None, :]
                )
            ).reshape(-1)

            if loss_positive_negative.numel() > self.rank_max_pairs:
                keep = torch.randperm(
                    loss_positive_negative.numel(),
                    device=loss_positive_negative.device,
                )[: self.rank_max_pairs]
                loss_positive_negative = loss_positive_negative[keep]
            losses.append(loss_positive_negative.mean())

        if not losses:
            return pred_score_logit.new_tensor(0.0)
        return torch.stack(losses).mean()

    def forward(
        self,
        pred_bbox: torch.Tensor,
        pred_score_logit: torch.Tensor,
        targets: List[dict],
        lambda_bbox: float = 5.0,
        lambda_giou: float = 2.0,
        lambda_score: float = 1.0,
        pos_weight: float = 1.0,
        current_epoch=None,
        quality_alpha=None,
        rank_alpha=None,
        quality_warmup_epoch: int = 10,
        rank_start_epoch: int = 30,
        rank_warmup_epoch: int = 60,
        rank_alpha_min: float = 0.0,
        lambda_rank: float = 0.0,
        query_loss_weights: Optional[torch.Tensor] = None,
        text_negative_mask: Optional[torch.Tensor] = None,
    ):
        quality_alpha, rank_alpha = self.resolve_epoch_alpha(
            current_epoch=current_epoch,
            quality_alpha=quality_alpha,
            rank_alpha=rank_alpha,
            quality_warmup_epoch=quality_warmup_epoch,
            rank_start_epoch=rank_start_epoch,
            rank_warmup_epoch=rank_warmup_epoch,
            rank_alpha_min=rank_alpha_min,
        )

        assignments = self.matcher(
            pred_bbox=pred_bbox,
            pred_score_logit=pred_score_logit,
            targets=targets,
        )

        score_target = torch.zeros_like(pred_score_logit)
        positive_mask = torch.zeros_like(pred_score_logit, dtype=torch.bool)

        loss_bbox_sum = pred_bbox.new_tensor(0.0)
        loss_giou_sum = pred_bbox.new_tensor(0.0)
        matched_iou_sum = pred_bbox.new_tensor(0.0)
        total_positive_pairs = 0

        for batch_index, (pred_index, gt_index) in enumerate(assignments):
            if pred_index.numel() == 0:
                continue

            gt_bbox = targets[batch_index]["boxes"].to(
                device=pred_bbox.device,
                dtype=pred_bbox.dtype,
                non_blocking=True,
            )
            positive_pred_boxes = pred_bbox[batch_index, pred_index]
            positive_gt_boxes = gt_bbox[gt_index]

            loss_bbox_sum = loss_bbox_sum + F.l1_loss(
                positive_pred_boxes,
                positive_gt_boxes,
                reduction="sum",
            )
            giou = torch.diag(
                generalized_box_iou(
                    positive_pred_boxes,
                    positive_gt_boxes,
                )
            )
            loss_giou_sum = loss_giou_sum + (1.0 - giou).sum()

            with torch.no_grad():
                matched_iou = torch.diag(
                    box_iou(
                        positive_pred_boxes.detach(),
                        positive_gt_boxes,
                    )
                ).clamp(0.0, 1.0)
                quality_target = (
                    (1.0 - quality_alpha) * torch.ones_like(matched_iou)
                    + quality_alpha * matched_iou
                )

            score_target[batch_index, pred_index, 0] = quality_target
            positive_mask[batch_index, pred_index, 0] = True
            matched_iou_sum = matched_iou_sum + matched_iou.sum()
            total_positive_pairs += int(pred_index.numel())

        normalizer = max(total_positive_pairs, 1)
        loss_bbox = loss_bbox_sum / normalizer
        loss_giou = loss_giou_sum / normalizer
        matched_iou_mean = matched_iou_sum / normalizer

        (
            loss_score,
            loss_score_pos,
            loss_score_neg,
            loss_score_neg_unweighted,
            loss_text_negative,
            score_pos_count,
            negative_count,
            text_negative_count,
            text_negative_weight_mean,
        ) = self.score_loss_quality_balanced(
            pred_score_logit=pred_score_logit,
            score_target=score_target,
            positive_mask=positive_mask,
            pos_weight=pos_weight,
            qfl_beta=self.qfl_beta,
            query_loss_weights=query_loss_weights,
            text_negative_mask=text_negative_mask,
        )

        lambda_rank_eff = float(lambda_rank) * float(rank_alpha)
        if lambda_rank_eff > 0:
            loss_rank_raw = self.pairwise_quality_rank_loss(
                pred_score_logit=pred_score_logit,
                score_target=score_target,
                positive_mask=positive_mask,
                margin=self.rank_margin,
                min_quality_gap=self.rank_min_quality_gap,
            )
            loss_rank = float(rank_alpha) * loss_rank_raw
            loss_rank_contrib = float(lambda_rank) * loss_rank
        else:
            loss_rank_raw = pred_score_logit.new_tensor(0.0)
            loss_rank = pred_score_logit.new_tensor(0.0)
            loss_rank_contrib = pred_score_logit.new_tensor(0.0)

        loss_main = (
            float(lambda_bbox) * loss_bbox
            + float(lambda_giou) * loss_giou
            + float(lambda_score) * loss_score
        )
        loss = loss_main + loss_rank_contrib

        if positive_mask.any():
            positive_targets = score_target[positive_mask]
            score_target_pos_mean = positive_targets.mean()
            score_target_pos_min = positive_targets.min()
            score_target_pos_max = positive_targets.max()
        else:
            score_target_pos_mean = pred_bbox.new_tensor(0.0)
            score_target_pos_min = pred_bbox.new_tensor(0.0)
            score_target_pos_max = pred_bbox.new_tensor(0.0)

        loss_dict = {
            "loss": loss.detach(),
            "loss_main": loss_main.detach(),
            "loss_bbox": loss_bbox.detach(),
            "loss_giou": loss_giou.detach(),
            "loss_score": loss_score.detach(),
            "loss_score_pos": loss_score_pos.detach(),
            "loss_score_neg": loss_score_neg.detach(),
            "loss_score_neg_unweighted": loss_score_neg_unweighted.detach(),
            "loss_text_negative": loss_text_negative.detach(),
            "loss_rank": loss_rank.detach(),
            "loss_rank_raw": loss_rank_raw.detach(),
            "loss_rank_contrib": loss_rank_contrib.detach(),
            "matched": float(total_positive_pairs),
            "score_pos_count": float(score_pos_count),
            "hard_neg_count": float(negative_count),
            "negative_count": float(negative_count),
            "text_negative_count": float(text_negative_count),
            "text_negative_weight_mean": text_negative_weight_mean.detach(),
            "matched_iou_mean": matched_iou_mean.detach(),
            "score_target_pos_mean": score_target_pos_mean.detach(),
            "score_target_pos_min": score_target_pos_min.detach(),
            "score_target_pos_max": score_target_pos_max.detach(),
            "lambda_bbox": float(lambda_bbox),
            "lambda_giou": float(lambda_giou),
            "lambda_score": float(lambda_score),
            "lambda_rank": float(lambda_rank),
            "lambda_rank_eff": float(lambda_rank_eff),
            "pos_weight": float(pos_weight),
            "quality_alpha": float(quality_alpha),
            "rank_alpha": float(rank_alpha),
        }
        return loss, loss_dict
