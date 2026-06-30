from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def clamp01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


def smoothstep(x: float) -> float:
    """Smooth 0 -> 1 warmup curve."""
    x = clamp01(x)
    return x * x * (3.0 - 2.0 * x)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Compute box area for boxes in normalized or pixel xyxy format."""
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




class HungarianOneToOneMatcher:
    """
    Exact one-to-one bipartite matching for the main DETR branch.

    Each prediction can match at most one GT, and each GT can match at most one
    prediction. The assignment is solved with SciPy's linear_sum_assignment.
    """

    def __init__(
        self,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cost_score: float = 0.0,
    ) -> None:
        if (
            float(cost_bbox) == 0.0
            and float(cost_giou) == 0.0
            and float(cost_score) == 0.0
        ):
            raise ValueError(
                "At least one Hungarian matching cost must be non-zero."
            )

        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)
        self.cost_score = float(cost_score)

    @torch.no_grad()
    def __call__(
        self,
        pred_bbox: torch.Tensor,
        pred_score_logit: torch.Tensor,
        targets: List[dict],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if linear_sum_assignment is None:
            raise ImportError(
                "HungarianOneToOneMatcher requires scipy. "
                "Install it with: pip install scipy"
            )

        if pred_bbox.ndim != 3 or pred_bbox.shape[-1] != 4:
            raise ValueError(
                "pred_bbox must have shape [B, N, 4], got "
                f"{tuple(pred_bbox.shape)}"
            )

        if pred_score_logit.ndim == 3 and pred_score_logit.shape[-1] == 1:
            score_logit = pred_score_logit.squeeze(-1)
        elif pred_score_logit.ndim == 2:
            score_logit = pred_score_logit
        else:
            raise ValueError(
                "pred_score_logit must have shape [B, N, 1] or [B, N], got "
                f"{tuple(pred_score_logit.shape)}"
            )

        batch_size, num_pred, _ = pred_bbox.shape

        if score_logit.shape != (batch_size, num_pred):
            raise ValueError(
                "pred_bbox/pred_score_logit shape mismatch: "
                f"bbox={tuple(pred_bbox.shape)}, "
                f"score={tuple(pred_score_logit.shape)}"
            )

        if len(targets) != batch_size:
            raise ValueError(
                "targets batch size mismatch: "
                f"{len(targets)} != {batch_size}"
            )

        pred_score = score_logit.detach().float().sigmoid()
        assignments: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for batch_index in range(batch_size):
            gt_bbox = targets[batch_index]["boxes"].to(
                device=pred_bbox.device,
                dtype=torch.float32,
                non_blocking=True,
            ).reshape(-1, 4)

            num_gt = int(gt_bbox.shape[0])
            device = pred_bbox.device

            if num_gt == 0 or num_pred == 0:
                empty = torch.empty(
                    0,
                    dtype=torch.long,
                    device=device,
                )
                assignments.append((empty, empty))
                continue

            boxes = pred_bbox[batch_index].detach().float()

            cost_bbox = torch.cdist(
                boxes,
                gt_bbox,
                p=1,
            )
            giou_matrix = generalized_box_iou(
                boxes,
                gt_bbox,
            )

            cost = (
                self.cost_bbox * cost_bbox
                - self.cost_giou * giou_matrix
            )

            if self.cost_score != 0.0:
                cost = (
                    cost
                    - self.cost_score
                    * pred_score[batch_index, :, None]
                )

            pred_indices_np, gt_indices_np = linear_sum_assignment(
                cost.detach().cpu().numpy()
            )

            pred_indices = torch.as_tensor(
                pred_indices_np,
                dtype=torch.long,
                device=device,
            )
            gt_indices = torch.as_tensor(
                gt_indices_np,
                dtype=torch.long,
                device=device,
            )

            assignments.append(
                (pred_indices, gt_indices)
            )

        return assignments

class OneToManyMatcher:
    """
    Greedy one-to-many assignment shared by bbox, GIoU and score supervision.

    A prediction can be assigned to only one GT. Each reachable GT first gets a
    greedy one-to-one match. Optional extra low-cost predictions are then added
    up to max_positive_per_gt and the global positive budget.
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
    def _greedy_one_to_one(
        cost: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
                empty = torch.empty(
                    0,
                    dtype=torch.long,
                    device=pred_bbox.device,
                )
                assignments.append((empty, empty))
                continue

            boxes = pred_bbox[batch_index].detach()
            cost_bbox = torch.cdist(boxes, gt_bbox, p=1)
            giou_matrix = generalized_box_iou(boxes, gt_bbox)
            iou_matrix = box_iou(boxes, gt_bbox)

            cost = self.cost_bbox * cost_bbox - self.cost_giou * giou_matrix
            if self.cost_score > 0:
                cost = (
                    cost
                    - self.cost_score * pred_score[batch_index, :, None]
                )

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
            used_pred = torch.zeros(
                num_pred,
                dtype=torch.bool,
                device=cost.device,
            )
            gt_counts = torch.zeros(
                num_gt,
                dtype=torch.long,
                device=cost.device,
            )

            for pred_index_tensor, gt_index_tensor in zip(base_pred, base_gt):
                pred_index = int(pred_index_tensor.item())
                gt_index = int(gt_index_tensor.item())
                selected_pred.append(pred_index)
                selected_gt.append(gt_index)
                used_pred[pred_index] = True
                gt_counts[gt_index] += 1

            if (
                len(selected_pred) < positive_budget
                and self.max_positive_per_gt > 1
            ):
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
    Hybrid detection loss for LightDet.

    Main branch:
      - Hungarian one-to-one assignment.
      - Used for final inference and mAP.
      - Includes bbox, GIoU, QFL score and ranking loss.

    Auxiliary branch:
      - Greedy one-to-many assignment.
      - Used only during training.
      - Includes bbox, GIoU and QFL score loss.
      - Does not contribute predictions to validation or inference.

    Total loss:
        loss = main_loss + lambda_aux * auxiliary_loss
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
        aux_loss_weight: float = 0.5,
        aux_cost_score: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.hard_negative_ratio = max(
            1,
            int(hard_negative_ratio),
        )
        self.aux_positive_label = clamp01(
            aux_positive_label
        )
        self.quality_min = clamp01(quality_min)
        self.quality_max = clamp01(quality_max)

        if self.quality_min > self.quality_max:
            raise ValueError(
                "quality_min must be <= quality_max, got "
                f"{self.quality_min} > {self.quality_max}"
            )

        self.aux_loss_weight = max(
            0.0,
            float(aux_loss_weight),
        )

        # Main branch: exact Hungarian one-to-one assignment.
        self.main_matcher = HungarianOneToOneMatcher(
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
            cost_score=cost_score,
        )

        # Auxiliary branch: one-to-many positive expansion.
        self.aux_matcher = OneToManyMatcher(
            cost_bbox=(
                expand_cost_bbox
                if expand_cost_bbox is not None
                else cost_bbox
            ),
            cost_giou=(
                expand_cost_giou
                if expand_cost_giou is not None
                else cost_giou
            ),
            cost_score=(
                cost_score
                if aux_cost_score is None
                else float(aux_cost_score)
            ),
            positive_ratio=positive_ratio,
            max_positive_per_gt=max_positive_per_gt,
            min_extra_positive_iou=iou_pos_thr,
        )

        # Backward-compatible alias. Existing code that reads criterion.matcher
        # will now see the main one-to-one matcher.
        self.matcher = self.main_matcher

        self.qfl_beta = float(qfl_beta)
        self.rank_margin = float(rank_margin)
        self.rank_min_quality_gap = float(
            rank_min_quality_gap
        )
        self.rank_max_pairs = max(
            1,
            int(rank_max_pairs),
        )
        self.max_query_loss_weight = max(
            1.0,
            float(max_query_loss_weight),
        )

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
        Resolve quality-target and ranking warmup coefficients.

        The positive score target gradually changes from quality_max to the
        clamped matched IoU target. Ranking starts at rank_start_epoch and
        reaches the configured lambda_rank after rank_warmup_epoch.
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

    @staticmethod
    def _safe_mean(
        values: List[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not values:
            return reference.new_tensor(0.0)
        return torch.stack(values).mean()

    def _select_query_hard_negative_indices(
        self,
        query_loss: torch.Tensor,
        query_positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select highest-QFL-loss negatives inside one query.

        For a query with P positives, select at most P * ratio negatives. For a
        query without positives, select ratio negatives. This keeps text-empty
        and text-negative queries focused on their highest-confidence errors.
        """
        flat_loss = query_loss.reshape(-1)
        flat_positive = query_positive_mask.reshape(-1)
        negative_indices = torch.nonzero(
            ~flat_positive,
            as_tuple=False,
        ).flatten()

        if negative_indices.numel() == 0:
            return negative_indices

        positive_count = int(flat_positive.sum().item())
        base_count = max(positive_count, 1)
        hard_negative_k = min(
            int(negative_indices.numel()),
            base_count * self.hard_negative_ratio,
        )

        negative_loss = flat_loss[negative_indices].detach()
        local_indices = torch.topk(
            negative_loss,
            k=hard_negative_k,
            largest=True,
            sorted=False,
        ).indices
        return negative_indices[local_indices]

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
        Quality Focal Loss with per-query hard-negative mining.

        Positive QFL is averaged over matched predictions. Negative QFL is
        mined independently for every query and then averaged per query, so a
        query with many easy negatives cannot dominate a query with only a few
        difficult negatives.
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

        has_pos = bool(positive_mask.any())
        loss_pos = (
            loss_all[positive_mask].mean()
            if has_pos
            else pred_score_logit.new_tensor(0.0)
        )

        negative_mask = ~positive_mask
        total_negative_count = float(negative_mask.sum().item())
        selected_negative_count = 0

        weighted_query_negative_losses: List[torch.Tensor] = []
        unweighted_query_negative_losses: List[torch.Tensor] = []
        text_negative_query_losses: List[torch.Tensor] = []

        batch_size = int(pred_score_logit.shape[0])
        for batch_index in range(batch_size):
            selected_indices = self._select_query_hard_negative_indices(
                query_loss=loss_all[batch_index],
                query_positive_mask=positive_mask[batch_index],
            )
            if selected_indices.numel() == 0:
                continue

            selected_negative_count += int(selected_indices.numel())
            query_loss_flat = loss_all[batch_index].reshape(-1)
            selected_unweighted = query_loss_flat[selected_indices].mean()
            query_weight_scalar = query_weight[batch_index].reshape(())
            selected_weighted = selected_unweighted * query_weight_scalar

            unweighted_query_negative_losses.append(selected_unweighted)
            weighted_query_negative_losses.append(selected_weighted)

            if bool(text_negative_mask[batch_index].item()):
                text_negative_query_losses.append(selected_weighted)

        loss_neg = self._safe_mean(
            weighted_query_negative_losses,
            pred_score_logit,
        )
        loss_neg_unweighted = self._safe_mean(
            unweighted_query_negative_losses,
            pred_score_logit,
        )
        text_negative_loss = self._safe_mean(
            text_negative_query_losses,
            pred_score_logit,
        )

        has_selected_neg = selected_negative_count > 0
        if has_pos and has_selected_neg:
            loss_score = (
                float(pos_weight) * loss_pos + loss_neg
            ) / (float(pos_weight) + 1.0)
        elif has_pos:
            loss_score = loss_pos
        elif has_selected_neg:
            loss_score = loss_neg
        else:
            loss_score = pred_score_logit.new_tensor(0.0)

        text_negative_count = float(text_negative_mask.sum().item())
        if text_negative_count > 0:
            text_negative_weight_mean = query_weight[
                text_negative_mask
            ].mean()
        else:
            text_negative_weight_mean = pred_score_logit.new_tensor(1.0)

        return (
            loss_score,
            loss_pos,
            loss_neg,
            loss_neg_unweighted,
            text_negative_loss,
            float(positive_mask.sum().item()),
            float(selected_negative_count),
            total_negative_count,
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
        """
        Per-query, logit-based pairwise ranking loss.

        Comparisons never cross query boundaries. Positive-positive pairs are
        generated only when their quality difference is meaningful. Positive-
        negative pairs use the highest-logit negatives from the same query.
        """
        if margin is None:
            margin = self.rank_margin
        if min_quality_gap is None:
            min_quality_gap = self.rank_min_quality_gap

        pair_losses: List[torch.Tensor] = []
        batch_size = int(pred_score_logit.shape[0])

        for batch_index in range(batch_size):
            query_logit = pred_score_logit[batch_index].reshape(-1)
            query_quality = score_target[batch_index].reshape(-1)
            query_positive = positive_mask[batch_index].reshape(-1)
            query_negative = ~query_positive

            positive_indices = torch.nonzero(
                query_positive,
                as_tuple=False,
            ).flatten()
            negative_indices = torch.nonzero(
                query_negative,
                as_tuple=False,
            ).flatten()

            if positive_indices.numel() >= 2:
                positive_quality = query_quality[positive_indices]
                positive_logit = query_logit[positive_indices]

                quality_i = positive_quality[:, None]
                quality_j = positive_quality[None, :]
                rank_mask = quality_i > quality_j + float(min_quality_gap)

                if rank_mask.any():
                    logit_difference = (
                        positive_logit[:, None] - positive_logit[None, :]
                    )
                    positive_pair_loss = F.softplus(
                        float(margin) - logit_difference
                    )[rank_mask]
                    pair_losses.append(positive_pair_loss.reshape(-1))

            if (
                positive_indices.numel() > 0
                and negative_indices.numel() > 0
            ):
                positive_logit = query_logit[positive_indices]
                negative_logit = query_logit[negative_indices]
                hard_negative_k = min(
                    int(negative_logit.numel()),
                    max(
                        int(positive_indices.numel())
                        * self.hard_negative_ratio,
                        1,
                    ),
                )
                hard_negative_logit = torch.topk(
                    negative_logit,
                    k=hard_negative_k,
                    largest=True,
                    sorted=False,
                ).values

                logit_difference = (
                    positive_logit[:, None]
                    - hard_negative_logit[None, :]
                )
                positive_negative_loss = F.softplus(
                    float(margin) - logit_difference
                ).reshape(-1)
                pair_losses.append(positive_negative_loss)

        if not pair_losses:
            return pred_score_logit.new_tensor(0.0)

        all_pair_losses = torch.cat(pair_losses)
        if all_pair_losses.numel() > self.rank_max_pairs:
            all_pair_losses = torch.topk(
                all_pair_losses,
                k=self.rank_max_pairs,
                largest=True,
                sorted=False,
            ).values

        return all_pair_losses.mean()


    @staticmethod
    def _validate_branch_inputs(
        branch_name: str,
        pred_bbox: torch.Tensor,
        pred_score_logit: torch.Tensor,
        targets: List[dict],
    ) -> None:
        if pred_bbox.ndim != 3 or pred_bbox.shape[-1] != 4:
            raise ValueError(
                f"{branch_name} pred_bbox must have shape [B, N, 4], "
                f"got {tuple(pred_bbox.shape)}"
            )

        if (
            pred_score_logit.ndim != 3
            or pred_score_logit.shape[-1] != 1
        ):
            raise ValueError(
                f"{branch_name} pred_score_logit must have shape "
                f"[B, N, 1], got {tuple(pred_score_logit.shape)}"
            )

        if pred_bbox.shape[:2] != pred_score_logit.shape[:2]:
            raise ValueError(
                f"{branch_name} bbox/score shape mismatch: "
                f"bbox={tuple(pred_bbox.shape)}, "
                f"score={tuple(pred_score_logit.shape)}"
            )

        if len(targets) != int(pred_bbox.shape[0]):
            raise ValueError(
                f"{branch_name} targets batch size mismatch: "
                f"{len(targets)} != {pred_bbox.shape[0]}"
            )

        if pred_bbox.device != pred_score_logit.device:
            raise ValueError(
                f"{branch_name} bbox/score device mismatch: "
                f"{pred_bbox.device} != {pred_score_logit.device}"
            )

    def _compute_branch_loss(
        self,
        *,
        branch_name: str,
        pred_bbox: torch.Tensor,
        pred_score_logit: torch.Tensor,
        targets: List[dict],
        assignments: List[Tuple[torch.Tensor, torch.Tensor]],
        lambda_bbox: float,
        lambda_giou: float,
        lambda_score: float,
        pos_weight: float,
        quality_alpha: float,
        query_loss_weights: Optional[torch.Tensor],
        text_negative_mask: Optional[torch.Tensor],
        warmup_positive_target: float,
        apply_ranking: bool,
        rank_alpha: float,
        lambda_rank: float,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self._validate_branch_inputs(
            branch_name=branch_name,
            pred_bbox=pred_bbox,
            pred_score_logit=pred_score_logit,
            targets=targets,
        )

        score_target = torch.zeros_like(
            pred_score_logit
        )
        positive_mask = torch.zeros_like(
            pred_score_logit,
            dtype=torch.bool,
        )

        loss_bbox_sum = pred_bbox.new_tensor(0.0)
        loss_giou_sum = pred_bbox.new_tensor(0.0)
        matched_iou_sum = pred_bbox.new_tensor(0.0)
        total_positive_pairs = 0

        warmup_positive_target = clamp01(
            warmup_positive_target
        )

        for batch_index, (
            pred_index,
            gt_index,
        ) in enumerate(assignments):
            if pred_index.numel() == 0:
                continue

            gt_bbox = targets[batch_index]["boxes"].to(
                device=pred_bbox.device,
                dtype=pred_bbox.dtype,
                non_blocking=True,
            ).reshape(-1, 4)

            positive_pred_boxes = pred_bbox[
                batch_index,
                pred_index,
            ]
            positive_gt_boxes = gt_bbox[gt_index]

            loss_bbox_sum = (
                loss_bbox_sum
                + F.l1_loss(
                    positive_pred_boxes,
                    positive_gt_boxes,
                    reduction="sum",
                )
            )

            matched_giou = torch.diag(
                generalized_box_iou(
                    positive_pred_boxes,
                    positive_gt_boxes,
                )
            )
            loss_giou_sum = (
                loss_giou_sum
                + (1.0 - matched_giou).sum()
            )

            with torch.no_grad():
                matched_iou = torch.diag(
                    box_iou(
                        positive_pred_boxes.detach(),
                        positive_gt_boxes,
                    )
                ).clamp(0.0, 1.0)

                final_quality_target = (
                    matched_iou.clamp(
                        min=self.quality_min,
                        max=self.quality_max,
                    )
                )

                warmup_target = torch.full_like(
                    matched_iou,
                    fill_value=warmup_positive_target,
                )

                quality_target = (
                    (1.0 - float(quality_alpha))
                    * warmup_target
                    + float(quality_alpha)
                    * final_quality_target
                ).clamp(0.0, 1.0)

            score_target[
                batch_index,
                pred_index,
                0,
            ] = quality_target

            positive_mask[
                batch_index,
                pred_index,
                0,
            ] = True

            matched_iou_sum = (
                matched_iou_sum
                + matched_iou.sum()
            )
            total_positive_pairs += int(
                pred_index.numel()
            )

        normalizer = max(
            total_positive_pairs,
            1,
        )

        loss_bbox = loss_bbox_sum / normalizer
        loss_giou = loss_giou_sum / normalizer
        matched_iou_mean = (
            matched_iou_sum / normalizer
        )

        (
            loss_score,
            loss_score_pos,
            loss_score_neg,
            loss_score_neg_unweighted,
            loss_text_negative,
            score_pos_count,
            hard_negative_count,
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

        if (
            bool(apply_ranking)
            and float(lambda_rank) > 0.0
            and float(rank_alpha) > 0.0
        ):
            loss_rank_raw = (
                self.pairwise_quality_rank_loss(
                    pred_score_logit=pred_score_logit,
                    score_target=score_target,
                    positive_mask=positive_mask,
                    margin=self.rank_margin,
                    min_quality_gap=(
                        self.rank_min_quality_gap
                    ),
                )
            )
            loss_rank = (
                float(rank_alpha)
                * loss_rank_raw
            )
            loss_rank_contrib = (
                float(lambda_rank)
                * loss_rank
            )
        else:
            loss_rank_raw = (
                pred_score_logit.new_tensor(0.0)
            )
            loss_rank = (
                pred_score_logit.new_tensor(0.0)
            )
            loss_rank_contrib = (
                pred_score_logit.new_tensor(0.0)
            )

        loss_base = (
            float(lambda_bbox) * loss_bbox
            + float(lambda_giou) * loss_giou
            + float(lambda_score) * loss_score
        )
        loss_total = (
            loss_base + loss_rank_contrib
        )

        if positive_mask.any():
            positive_targets = score_target[
                positive_mask
            ]
            score_target_pos_mean = (
                positive_targets.mean()
            )
            score_target_pos_min = (
                positive_targets.min()
            )
            score_target_pos_max = (
                positive_targets.max()
            )
        else:
            zero = pred_bbox.new_tensor(0.0)
            score_target_pos_mean = zero
            score_target_pos_min = zero
            score_target_pos_max = zero

        selected_negative_fraction = (
            float(hard_negative_count)
            / max(float(negative_count), 1.0)
        )

        metrics: Dict[str, Any] = {
            "loss_total": loss_total,
            "loss_base": loss_base,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_score": loss_score,
            "loss_score_pos": loss_score_pos,
            "loss_score_neg": loss_score_neg,
            "loss_score_neg_unweighted": (
                loss_score_neg_unweighted
            ),
            "loss_text_negative": (
                loss_text_negative
            ),
            "loss_rank": loss_rank,
            "loss_rank_raw": loss_rank_raw,
            "loss_rank_contrib": (
                loss_rank_contrib
            ),
            "matched": float(
                total_positive_pairs
            ),
            "score_pos_count": float(
                score_pos_count
            ),
            "hard_neg_count": float(
                hard_negative_count
            ),
            "negative_count": float(
                negative_count
            ),
            "selected_negative_fraction": float(
                selected_negative_fraction
            ),
            "text_negative_count": float(
                text_negative_count
            ),
            "text_negative_weight_mean": (
                text_negative_weight_mean
            ),
            "matched_iou_mean": (
                matched_iou_mean
            ),
            "score_target_pos_mean": (
                score_target_pos_mean
            ),
            "score_target_pos_min": (
                score_target_pos_min
            ),
            "score_target_pos_max": (
                score_target_pos_max
            ),
            "lambda_bbox": float(
                lambda_bbox
            ),
            "lambda_giou": float(
                lambda_giou
            ),
            "lambda_score": float(
                lambda_score
            ),
            "lambda_rank": float(
                lambda_rank
                if apply_ranking
                else 0.0
            ),
            "lambda_rank_eff": float(
                lambda_rank * rank_alpha
                if apply_ranking
                else 0.0
            ),
            "pos_weight": float(
                pos_weight
            ),
            "quality_alpha": float(
                quality_alpha
            ),
            "rank_alpha": float(
                rank_alpha
                if apply_ranking
                else 0.0
            ),
        }

        return loss_total, metrics

    @staticmethod
    def _detach_metric(
        value: Any,
    ) -> Any:
        if torch.is_tensor(value):
            return value.detach()
        return value

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
        aux_pred_bbox: Optional[torch.Tensor] = None,
        aux_pred_score_logit: Optional[torch.Tensor] = None,
        lambda_aux: Optional[float] = None,
        aux_lambda_bbox: Optional[float] = None,
        aux_lambda_giou: Optional[float] = None,
        aux_lambda_score: Optional[float] = None,
        aux_pos_weight: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute hybrid main one-to-one and auxiliary one-to-many losses.

        The existing pred_bbox/pred_score_logit arguments are treated as the
        main branch, preserving compatibility with the current train.py.

        Auxiliary loss is enabled only when both aux_pred_bbox and
        aux_pred_score_logit are supplied.
        """
        quality_alpha, rank_alpha = (
            self.resolve_epoch_alpha(
                current_epoch=current_epoch,
                quality_alpha=quality_alpha,
                rank_alpha=rank_alpha,
                quality_warmup_epoch=(
                    quality_warmup_epoch
                ),
                rank_start_epoch=rank_start_epoch,
                rank_warmup_epoch=rank_warmup_epoch,
                rank_alpha_min=rank_alpha_min,
            )
        )

        main_assignments = self.main_matcher(
            pred_bbox=pred_bbox,
            pred_score_logit=pred_score_logit,
            targets=targets,
        )

        main_loss, main_metrics = (
            self._compute_branch_loss(
                branch_name="main",
                pred_bbox=pred_bbox,
                pred_score_logit=(
                    pred_score_logit
                ),
                targets=targets,
                assignments=main_assignments,
                lambda_bbox=lambda_bbox,
                lambda_giou=lambda_giou,
                lambda_score=lambda_score,
                pos_weight=pos_weight,
                quality_alpha=quality_alpha,
                query_loss_weights=(
                    query_loss_weights
                ),
                text_negative_mask=(
                    text_negative_mask
                ),
                warmup_positive_target=(
                    self.quality_max
                ),
                apply_ranking=True,
                rank_alpha=rank_alpha,
                lambda_rank=lambda_rank,
            )
        )

        has_aux_bbox = aux_pred_bbox is not None
        has_aux_score = (
            aux_pred_score_logit is not None
        )

        if has_aux_bbox != has_aux_score:
            raise ValueError(
                "aux_pred_bbox and aux_pred_score_logit "
                "must either both be provided or both be None."
            )

        aux_enabled = (
            has_aux_bbox and has_aux_score
        )
        lambda_aux_eff = (
            self.aux_loss_weight
            if lambda_aux is None
            else max(0.0, float(lambda_aux))
        )

        if aux_enabled:
            aux_lambda_bbox_eff = (
                float(lambda_bbox)
                if aux_lambda_bbox is None
                else float(aux_lambda_bbox)
            )
            aux_lambda_giou_eff = (
                float(lambda_giou)
                if aux_lambda_giou is None
                else float(aux_lambda_giou)
            )
            aux_lambda_score_eff = (
                float(lambda_score)
                if aux_lambda_score is None
                else float(aux_lambda_score)
            )
            aux_pos_weight_eff = (
                float(pos_weight)
                if aux_pos_weight is None
                else float(aux_pos_weight)
            )

            aux_assignments = self.aux_matcher(
                pred_bbox=aux_pred_bbox,
                pred_score_logit=(
                    aux_pred_score_logit
                ),
                targets=targets,
            )

            aux_loss, aux_metrics = (
                self._compute_branch_loss(
                    branch_name="aux",
                    pred_bbox=aux_pred_bbox,
                    pred_score_logit=(
                        aux_pred_score_logit
                    ),
                    targets=targets,
                    assignments=aux_assignments,
                    lambda_bbox=(
                        aux_lambda_bbox_eff
                    ),
                    lambda_giou=(
                        aux_lambda_giou_eff
                    ),
                    lambda_score=(
                        aux_lambda_score_eff
                    ),
                    pos_weight=(
                        aux_pos_weight_eff
                    ),
                    quality_alpha=(
                        quality_alpha
                    ),
                    query_loss_weights=(
                        query_loss_weights
                    ),
                    text_negative_mask=(
                        text_negative_mask
                    ),
                    warmup_positive_target=(
                        self.aux_positive_label
                    ),
                    apply_ranking=False,
                    rank_alpha=0.0,
                    lambda_rank=0.0,
                )
            )
        else:
            aux_loss = pred_bbox.new_tensor(0.0)
            zero = pred_bbox.new_tensor(0.0)

            aux_metrics = {
                "loss_total": zero,
                "loss_base": zero,
                "loss_bbox": zero,
                "loss_giou": zero,
                "loss_score": zero,
                "loss_score_pos": zero,
                "loss_score_neg": zero,
                "loss_score_neg_unweighted": zero,
                "loss_text_negative": zero,
                "loss_rank": zero,
                "loss_rank_raw": zero,
                "loss_rank_contrib": zero,
                "matched": 0.0,
                "score_pos_count": 0.0,
                "hard_neg_count": 0.0,
                "negative_count": 0.0,
                "selected_negative_fraction": 0.0,
                "text_negative_count": 0.0,
                "text_negative_weight_mean": zero,
                "matched_iou_mean": zero,
                "score_target_pos_mean": zero,
                "score_target_pos_min": zero,
                "score_target_pos_max": zero,
                "lambda_bbox": float(
                    lambda_bbox
                ),
                "lambda_giou": float(
                    lambda_giou
                ),
                "lambda_score": float(
                    lambda_score
                ),
                "lambda_rank": 0.0,
                "lambda_rank_eff": 0.0,
                "pos_weight": float(
                    pos_weight
                ),
                "quality_alpha": float(
                    quality_alpha
                ),
                "rank_alpha": 0.0,
            }

        aux_loss_contrib = (
            float(lambda_aux_eff) * aux_loss
        )
        loss = main_loss + aux_loss_contrib

        loss_dict: Dict[str, Any] = {
            "loss": loss.detach(),
            "loss_main_total": (
                main_loss.detach()
            ),
            "loss_aux_total": (
                aux_loss.detach()
            ),
            "loss_aux_contrib": (
                aux_loss_contrib.detach()
            ),
            "lambda_aux": float(
                lambda_aux_eff
            ),
            "aux_enabled": bool(
                aux_enabled
            ),

            # Backward-compatible main-branch aliases.
            "loss_main": self._detach_metric(
                main_metrics["loss_base"]
            ),
            "loss_bbox": self._detach_metric(
                main_metrics["loss_bbox"]
            ),
            "loss_giou": self._detach_metric(
                main_metrics["loss_giou"]
            ),
            "loss_score": self._detach_metric(
                main_metrics["loss_score"]
            ),
            "loss_score_pos": self._detach_metric(
                main_metrics["loss_score_pos"]
            ),
            "loss_score_neg": self._detach_metric(
                main_metrics["loss_score_neg"]
            ),
            "loss_score_neg_unweighted": (
                self._detach_metric(
                    main_metrics[
                        "loss_score_neg_unweighted"
                    ]
                )
            ),
            "loss_text_negative": (
                self._detach_metric(
                    main_metrics[
                        "loss_text_negative"
                    ]
                )
            ),
            "loss_rank": self._detach_metric(
                main_metrics["loss_rank"]
            ),
            "loss_rank_raw": self._detach_metric(
                main_metrics["loss_rank_raw"]
            ),
            "loss_rank_contrib": (
                self._detach_metric(
                    main_metrics[
                        "loss_rank_contrib"
                    ]
                )
            ),
            "matched": main_metrics["matched"],
            "score_pos_count": (
                main_metrics["score_pos_count"]
            ),
            "hard_neg_count": (
                main_metrics["hard_neg_count"]
            ),
            "negative_count": (
                main_metrics["negative_count"]
            ),
            "selected_negative_fraction": (
                main_metrics[
                    "selected_negative_fraction"
                ]
            ),
            "text_negative_count": (
                main_metrics[
                    "text_negative_count"
                ]
            ),
            "text_negative_weight_mean": (
                self._detach_metric(
                    main_metrics[
                        "text_negative_weight_mean"
                    ]
                )
            ),
            "matched_iou_mean": (
                self._detach_metric(
                    main_metrics[
                        "matched_iou_mean"
                    ]
                )
            ),
            "score_target_pos_mean": (
                self._detach_metric(
                    main_metrics[
                        "score_target_pos_mean"
                    ]
                )
            ),
            "score_target_pos_min": (
                self._detach_metric(
                    main_metrics[
                        "score_target_pos_min"
                    ]
                )
            ),
            "score_target_pos_max": (
                self._detach_metric(
                    main_metrics[
                        "score_target_pos_max"
                    ]
                )
            ),
            "lambda_bbox": (
                main_metrics["lambda_bbox"]
            ),
            "lambda_giou": (
                main_metrics["lambda_giou"]
            ),
            "lambda_score": (
                main_metrics["lambda_score"]
            ),
            "lambda_rank": (
                main_metrics["lambda_rank"]
            ),
            "lambda_rank_eff": (
                main_metrics["lambda_rank_eff"]
            ),
            "pos_weight": (
                main_metrics["pos_weight"]
            ),
            "quality_alpha": (
                main_metrics["quality_alpha"]
            ),
            "rank_alpha": (
                main_metrics["rank_alpha"]
            ),
        }

        for key, value in main_metrics.items():
            loss_dict[f"main_{key}"] = (
                self._detach_metric(value)
            )

        for key, value in aux_metrics.items():
            loss_dict[f"aux_{key}"] = (
                self._detach_metric(value)
            )

        return loss, loss_dict

