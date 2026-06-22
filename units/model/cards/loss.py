import math
from typing import List, Tuple

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
    """
    Pairwise IoU.

    boxes1: [N, 4], normalized or absolute xyxy
    boxes2: [M, 4], normalized or absolute xyxy
    return: [N, M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)

    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def generalized_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise generalized IoU.

    boxes1: [N, 4]
    boxes2: [M, 4]
    return: [N, M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)

    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    lt_c = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1]

    return iou - (area_c - union) / area_c.clamp(min=1e-6)


class OneToManyMatcher:
    """
    One-to-many assignment used by bbox, GIoU and score supervision together.

    Rules:
    1. First create a greedy one-to-one assignment so each reachable GT receives
       at least one prediction.
    2. Then add low-cost predictions until each GT has at most top-k positives.
    3. A prediction can be assigned to only one GT.
    4. Extra positives can optionally be gated by minimum IoU.
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
        inf = torch.finfo(work.dtype).max
        pred_indices: List[torch.Tensor] = []
        gt_indices: List[torch.Tensor] = []

        for _ in range(min(num_pred, num_gt)):
            flat_idx = torch.argmin(work.reshape(-1))
            pred_idx = torch.div(flat_idx, num_gt, rounding_mode="floor")
            gt_idx = flat_idx % num_gt

            pred_indices.append(pred_idx)
            gt_indices.append(gt_idx)

            work[pred_idx, :] = inf
            work[:, gt_idx] = inf

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

        for batch_idx in range(batch_size):
            gt_bbox = targets[batch_idx]["boxes"].to(
                device=pred_bbox.device,
                dtype=pred_bbox.dtype,
                non_blocking=True,
            )
            num_gt = int(gt_bbox.shape[0])

            if num_gt == 0 or num_pred == 0:
                empty = torch.empty(0, dtype=torch.long, device=pred_bbox.device)
                assignments.append((empty, empty))
                continue

            boxes = pred_bbox[batch_idx].detach()
            cost_bbox = torch.cdist(boxes, gt_bbox, p=1)
            giou_matrix = generalized_box_iou(boxes, gt_bbox)
            iou_matrix = box_iou(boxes, gt_bbox)

            cost = self.cost_bbox * cost_bbox - self.cost_giou * giou_matrix
            if self.cost_score > 0:
                cost = cost - self.cost_score * pred_score[batch_idx, :, None]

            # Global positive budget. It is never smaller than one prediction per
            # GT when enough predictions are available.
            minimum_budget = min(num_pred, num_gt)
            ratio_budget = max(1, int(round(num_pred * self.positive_ratio)))
            maximum_budget = min(num_pred, num_gt * self.max_positive_per_gt)
            positive_budget = min(max(minimum_budget, ratio_budget), maximum_budget)

            base_pred, base_gt = self._greedy_one_to_one(cost)

            selected_pred: List[int] = []
            selected_gt: List[int] = []
            used_pred = torch.zeros(num_pred, dtype=torch.bool, device=cost.device)
            gt_counts = torch.zeros(num_gt, dtype=torch.long, device=cost.device)

            for pred_idx_t, gt_idx_t in zip(base_pred, base_gt):
                pred_idx = int(pred_idx_t.item())
                gt_idx = int(gt_idx_t.item())
                selected_pred.append(pred_idx)
                selected_gt.append(gt_idx)
                used_pred[pred_idx] = True
                gt_counts[gt_idx] += 1

            if len(selected_pred) < positive_budget and self.max_positive_per_gt > 1:
                candidate_pred: List[torch.Tensor] = []
                candidate_gt: List[torch.Tensor] = []
                candidate_cost: List[torch.Tensor] = []

                candidate_k = min(num_pred, self.max_positive_per_gt)
                for gt_idx in range(num_gt):
                    values, pred_indices = torch.topk(
                        cost[:, gt_idx],
                        k=candidate_k,
                        largest=False,
                    )
                    candidate_pred.append(pred_indices)
                    candidate_gt.append(
                        torch.full_like(pred_indices, fill_value=gt_idx)
                    )
                    candidate_cost.append(values)

                all_pred = torch.cat(candidate_pred)
                all_gt = torch.cat(candidate_gt)
                all_cost = torch.cat(candidate_cost)
                order = torch.argsort(all_cost)

                for order_idx_t in order:
                    if len(selected_pred) >= positive_budget:
                        break

                    order_idx = int(order_idx_t.item())
                    pred_idx = int(all_pred[order_idx].item())
                    gt_idx = int(all_gt[order_idx].item())

                    if used_pred[pred_idx]:
                        continue
                    if int(gt_counts[gt_idx].item()) >= self.max_positive_per_gt:
                        continue
                    if (
                        self.min_extra_positive_iou > 0
                        and float(iou_matrix[pred_idx, gt_idx].item())
                        < self.min_extra_positive_iou
                    ):
                        continue

                    selected_pred.append(pred_idx)
                    selected_gt.append(gt_idx)
                    used_pred[pred_idx] = True
                    gt_counts[gt_idx] += 1

            assignments.append(
                (
                    torch.tensor(selected_pred, dtype=torch.long, device=cost.device),
                    torch.tensor(selected_gt, dtype=torch.long, device=cost.device),
                )
            )

        return assignments


class GroundingLoss(nn.Module):
    """
    Aligned one-to-many detection loss.

    Every assigned positive pair uses the same prediction/GT indices for:
      - L1 bbox regression
      - generalized IoU loss
      - IoU-aware Quality Focal Loss target

    Every unassigned prediction is a score negative with target 0. No hard
    negative subsampling is used; QFL down-weights easy negatives naturally.
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
    ) -> None:
        super().__init__()

        # These arguments remain in the signature so the current train.py does
        # not need structural changes. hard_negative_ratio, aux_positive_label,
        # quality_min and quality_max are intentionally not used by the aligned
        # formulation.
        self.hard_negative_ratio = int(hard_negative_ratio)
        self.aux_positive_label = float(aux_positive_label)
        self.quality_min = float(quality_min)
        self.quality_max = float(quality_max)

        self.matcher = OneToManyMatcher(
            cost_bbox=expand_cost_bbox if expand_cost_bbox is not None else cost_bbox,
            cost_giou=expand_cost_giou if expand_cost_giou is not None else cost_giou,
            cost_score=cost_score,
            positive_ratio=positive_ratio,
            max_positive_per_gt=max_positive_per_gt,
            min_extra_positive_iou=iou_pos_thr,
        )

        self.qfl_beta = float(qfl_beta)
        self.rank_margin = float(rank_margin)
        self.rank_min_quality_gap = float(rank_min_quality_gap)
        self.rank_max_pairs = int(rank_max_pairs)

    def resolve_epoch_alpha(
        self,
        current_epoch=None,
        quality_alpha=None,
        rank_alpha=None,
        quality_warmup_epoch: int = 10,
        rank_start_epoch: int = 30,
        rank_warmup_epoch: int = 60,
        rank_alpha_min: float = 1e-4,
    ) -> Tuple[float, float]:
        """
        During the short quality warmup:
            target = (1 - alpha) * 1 + alpha * IoU

        After warmup:
            target = IoU

        This keeps initial confidence gradients usable while converging to a
        strictly localization-aware confidence target.
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
                t = (
                    float(current_epoch) - float(rank_start_epoch)
                ) / max(float(rank_warmup_epoch), 1.0)
                rank_alpha = float(rank_alpha_min) + (
                    1.0 - float(rank_alpha_min)
                ) * smoothstep(t)
                rank_alpha = clamp01(rank_alpha)

        return float(quality_alpha), float(rank_alpha)

    def score_loss_quality_balanced(
        self,
        pred_score_logit: torch.Tensor,
        score_target: torch.Tensor,
        positive_mask: torch.Tensor,
        pos_weight: float = 1.0,
        qfl_beta=None,
    ):
        """Quality Focal Loss using all unmatched predictions as negatives."""
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

        negative_mask = ~positive_mask
        has_pos = bool(positive_mask.any())
        has_neg = bool(negative_mask.any())

        loss_pos = (
            loss_all[positive_mask].mean()
            if has_pos
            else pred_score_logit.new_tensor(0.0)
        )
        loss_neg = (
            loss_all[negative_mask].mean()
            if has_neg
            else pred_score_logit.new_tensor(0.0)
        )

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
            float(positive_mask.sum().item()),
            float(negative_mask.sum().item()),
        )

    def pairwise_quality_rank_loss(
        self,
        pred_score_logit: torch.Tensor,
        score_target: torch.Tensor,
        positive_mask: torch.Tensor,
        margin=None,
        min_quality_gap=None,
    ) -> torch.Tensor:
        """Optional ranking loss. Keep lambda_rank=0 for the first baseline."""
        if margin is None:
            margin = self.rank_margin
        if min_quality_gap is None:
            min_quality_gap = self.rank_min_quality_gap

        score = pred_score_logit.sigmoid().reshape(-1)
        quality = score_target.reshape(-1)
        pos_mask = positive_mask.reshape(-1)
        neg_mask = ~pos_mask

        pos_idx = torch.nonzero(pos_mask, as_tuple=False).flatten()
        neg_idx = torch.nonzero(neg_mask, as_tuple=False).flatten()
        losses = []

        if pos_idx.numel() >= 2:
            pos_quality = quality[pos_idx]
            pos_score = score[pos_idx]
            quality_i = pos_quality[:, None]
            quality_j = pos_quality[None, :]
            rank_mask = quality_i > quality_j + float(min_quality_gap)

            if rank_mask.any():
                score_diff = pos_score[:, None] - pos_score[None, :]
                loss_pp = F.relu(float(margin) - score_diff)[rank_mask]
                if loss_pp.numel() > self.rank_max_pairs:
                    keep = torch.randperm(
                        loss_pp.numel(), device=loss_pp.device
                    )[: self.rank_max_pairs]
                    loss_pp = loss_pp[keep]
                losses.append(loss_pp.mean())

        if pos_idx.numel() > 0 and neg_idx.numel() > 0:
            pos_score = score[pos_idx]
            neg_score = score[neg_idx]
            hard_neg_k = min(
                int(neg_score.numel()),
                max(int(pos_idx.numel()) * max(self.hard_negative_ratio, 1), 1),
            )
            hard_neg_score = torch.topk(
                neg_score,
                k=hard_neg_k,
                largest=True,
            ).values
            loss_pn = F.relu(
                float(margin) - (pos_score[:, None] - hard_neg_score[None, :])
            ).reshape(-1)

            if loss_pn.numel() > self.rank_max_pairs:
                keep = torch.randperm(
                    loss_pn.numel(), device=loss_pn.device
                )[: self.rank_max_pairs]
                loss_pn = loss_pn[keep]
            losses.append(loss_pn.mean())

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
        rank_alpha_min: float = 1e-4,
        lambda_rank: float = 0.0,
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

        for batch_idx, (pred_idx, gt_idx) in enumerate(assignments):
            if pred_idx.numel() == 0:
                continue

            gt_bbox = targets[batch_idx]["boxes"].to(
                device=pred_bbox.device,
                dtype=pred_bbox.dtype,
                non_blocking=True,
            )
            positive_pred_boxes = pred_bbox[batch_idx, pred_idx]
            positive_gt_boxes = gt_bbox[gt_idx]

            # The exact same positive pairs supervise bbox and GIoU.
            loss_bbox_sum = loss_bbox_sum + F.l1_loss(
                positive_pred_boxes,
                positive_gt_boxes,
                reduction="sum",
            )
            giou = torch.diag(
                generalized_box_iou(positive_pred_boxes, positive_gt_boxes)
            )
            loss_giou_sum = loss_giou_sum + (1.0 - giou).sum()

            # The exact same pairs supervise confidence. IoU is detached so the
            # score branch cannot alter bbox through its target.
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

            score_target[batch_idx, pred_idx, 0] = quality_target
            positive_mask[batch_idx, pred_idx, 0] = True
            matched_iou_sum = matched_iou_sum + matched_iou.sum()
            total_positive_pairs += int(pred_idx.numel())

        normalizer = max(total_positive_pairs, 1)
        loss_bbox = loss_bbox_sum / normalizer
        loss_giou = loss_giou_sum / normalizer
        matched_iou_mean = matched_iou_sum / normalizer

        (
            loss_score,
            loss_score_pos,
            loss_score_neg,
            score_pos_count,
            negative_count,
        ) = self.score_loss_quality_balanced(
            pred_score_logit=pred_score_logit,
            score_target=score_target,
            positive_mask=positive_mask,
            pos_weight=pos_weight,
            qfl_beta=self.qfl_beta,
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
            "loss_rank": loss_rank.detach(),
            "loss_rank_raw": loss_rank_raw.detach(),
            "loss_rank_contrib": loss_rank_contrib.detach(),
            "matched": float(total_positive_pairs),
            "score_pos_count": float(score_pos_count),
            # Keep this legacy key so current logging code remains compatible.
            "hard_neg_count": float(negative_count),
            "negative_count": float(negative_count),
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
