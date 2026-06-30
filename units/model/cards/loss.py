from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

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



def matched_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """Elementwise IoU for two equally sized xyxy box tensors [M, 4]."""
    if boxes1.shape != boxes2.shape or boxes1.ndim != 2 or boxes1.shape[-1] != 4:
        raise ValueError(
            "matched_box_iou expects equal [M, 4] tensors, got "
            f"{tuple(boxes1.shape)} and {tuple(boxes2.shape)}"
        )

    if boxes1.numel() == 0:
        return boxes1.new_zeros((0,))

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - intersection
    return intersection / union.clamp(min=1e-6)


def matched_generalized_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """Elementwise generalized IoU for equally sized xyxy boxes [M, 4]."""
    if boxes1.shape != boxes2.shape or boxes1.ndim != 2 or boxes1.shape[-1] != 4:
        raise ValueError(
            "matched_generalized_box_iou expects equal [M, 4] tensors, got "
            f"{tuple(boxes1.shape)} and {tuple(boxes2.shape)}"
        )

    if boxes1.numel() == 0:
        return boxes1.new_zeros((0,))

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - intersection
    iou = intersection / union.clamp(min=1e-6)

    cover_lt = torch.minimum(boxes1[:, :2], boxes2[:, :2])
    cover_rb = torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
    cover_wh = (cover_rb - cover_lt).clamp(min=0)
    cover_area = cover_wh[:, 0] * cover_wh[:, 1]
    return iou - (cover_area - union) / cover_area.clamp(min=1e-6)


def batched_single_target_iou(
    boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> torch.Tensor:
    """IoU for [B, N, 4] predictions against one GT [B, 4] per query."""
    gt = gt_boxes[:, None, :]
    area_pred = box_area(boxes)
    area_gt = box_area(gt_boxes)[:, None]
    lt = torch.maximum(boxes[..., :2], gt[..., :2])
    rb = torch.minimum(boxes[..., 2:], gt[..., 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    union = area_pred + area_gt - intersection
    return intersection / union.clamp(min=1e-6)


def batched_single_target_giou(
    boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> torch.Tensor:
    """GIoU for [B, N, 4] predictions against one GT [B, 4] per query."""
    gt = gt_boxes[:, None, :]
    area_pred = box_area(boxes)
    area_gt = box_area(gt_boxes)[:, None]
    lt = torch.maximum(boxes[..., :2], gt[..., :2])
    rb = torch.minimum(boxes[..., 2:], gt[..., 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    union = area_pred + area_gt - intersection
    iou = intersection / union.clamp(min=1e-6)

    cover_lt = torch.minimum(boxes[..., :2], gt[..., :2])
    cover_rb = torch.maximum(boxes[..., 2:], gt[..., 2:])
    cover_wh = (cover_rb - cover_lt).clamp(min=0)
    cover_area = cover_wh[..., 0] * cover_wh[..., 1]
    return iou - (cover_area - union) / cover_area.clamp(min=1e-6)


@dataclass(frozen=True)
class PackedTargets:
    """All target boxes packed once for both main and auxiliary branches."""

    boxes: torch.Tensor
    offsets: torch.Tensor
    counts: Tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return len(self.counts)

    @classmethod
    def from_targets(
        cls,
        targets: List[dict],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "PackedTargets":
        box_rows: List[torch.Tensor] = []
        counts: List[int] = []
        offsets = [0]

        for target in targets:
            if "boxes" not in target:
                raise KeyError("Each target must contain 'boxes'")

            boxes = target["boxes"]
            if not torch.is_tensor(boxes):
                boxes = torch.as_tensor(boxes, dtype=torch.float32)

            boxes = boxes.to(
                device=device,
                dtype=dtype,
                non_blocking=True,
            ).reshape(-1, 4)

            count = int(boxes.shape[0])
            counts.append(count)
            offsets.append(offsets[-1] + count)

            if count > 0:
                box_rows.append(boxes)

        if box_rows:
            flat_boxes = torch.cat(box_rows, dim=0)
        else:
            flat_boxes = torch.empty((0, 4), device=device, dtype=dtype)

        return cls(
            boxes=flat_boxes,
            offsets=torch.tensor(offsets, device=device, dtype=torch.long),
            counts=tuple(counts),
        )

    def global_gt_indices(
        self,
        batch_indices: torch.Tensor,
        local_gt_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.offsets[batch_indices] + local_gt_indices


@dataclass(frozen=True)
class AssignmentResult:
    """
    Flat batched assignment.

    The tensors contain all matched pairs across the batch. counts stores the
    number of matches per query and keeps the object list-compatible for older
    diagnostic code without using per-query computation in the loss path.
    """

    batch_indices: torch.Tensor
    pred_indices: torch.Tensor
    gt_indices: torch.Tensor
    counts: Tuple[int, ...]
    mode: str = "generic"

    @property
    def num_matches(self) -> int:
        return int(sum(self.counts))

    def __len__(self) -> int:
        return len(self.counts)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        start = 0
        for count in self.counts:
            end = start + count
            yield self.pred_indices[start:end], self.gt_indices[start:end]
            start = end

    @classmethod
    def from_per_batch(
        cls,
        assignments: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        *,
        device: torch.device,
        mode: str,
    ) -> "AssignmentResult":
        counts = tuple(int(pred.numel()) for pred, _ in assignments)
        total = sum(counts)

        if total == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return cls(empty, empty, empty, counts, mode)

        pred_indices = torch.cat(
            [pred.to(device=device, dtype=torch.long) for pred, _ in assignments if pred.numel() > 0],
            dim=0,
        )
        gt_indices = torch.cat(
            [gt.to(device=device, dtype=torch.long) for pred, gt in assignments if pred.numel() > 0],
            dim=0,
        )
        count_tensor = torch.tensor(counts, device=device, dtype=torch.long)
        batch_indices = torch.repeat_interleave(
            torch.arange(len(counts), device=device, dtype=torch.long),
            count_tensor,
        )

        return cls(batch_indices, pred_indices, gt_indices, counts, mode)





class HungarianOneToOneMatcher:
    """Main-branch Hungarian matcher with a batched 0/1-GT fast path."""

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
            raise ValueError("At least one Hungarian matching cost must be non-zero.")

        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)
        self.cost_score = float(cost_score)

    @torch.no_grad()
    def __call__(
        self,
        pred_bbox: torch.Tensor,
        pred_score_logit: torch.Tensor,
        targets: List[dict],
        packed_targets: Optional[PackedTargets] = None,
    ) -> AssignmentResult:
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
            raise ValueError("pred_bbox/pred_score_logit shape mismatch")
        if len(targets) != batch_size:
            raise ValueError(f"targets batch size mismatch: {len(targets)} != {batch_size}")

        if packed_targets is None:
            packed_targets = PackedTargets.from_targets(
                targets,
                device=pred_bbox.device,
                dtype=pred_bbox.dtype,
            )

        device = pred_bbox.device
        empty = torch.empty(0, dtype=torch.long, device=device)
        per_batch: List[Tuple[torch.Tensor, torch.Tensor]] = [
            (empty, empty) for _ in range(batch_size)
        ]
        pred_score = score_logit.detach().float().sigmoid()

        single_rows = [
            index for index, count in enumerate(packed_targets.counts)
            if count == 1
        ]

        if num_pred > 0 and single_rows:
            batch_index = torch.tensor(single_rows, device=device, dtype=torch.long)
            global_gt_index = packed_targets.offsets[batch_index]
            gt_bbox = packed_targets.boxes[global_gt_index].float()
            boxes = pred_bbox.detach().float().index_select(0, batch_index)

            cost_bbox = torch.abs(boxes - gt_bbox[:, None, :]).sum(dim=-1)
            giou = batched_single_target_giou(boxes, gt_bbox)
            cost = self.cost_bbox * cost_bbox - self.cost_giou * giou
            if self.cost_score != 0.0:
                cost = cost - self.cost_score * pred_score.index_select(0, batch_index)

            best_pred = torch.argmin(cost, dim=1)

            if all(count <= 1 for count in packed_targets.counts):
                counts = tuple(1 if count == 1 else 0 for count in packed_targets.counts)
                return AssignmentResult(
                    batch_indices=batch_index,
                    pred_indices=best_pred.to(dtype=torch.long),
                    gt_indices=torch.zeros_like(best_pred, dtype=torch.long),
                    counts=counts,
                    mode="batched_single_gt",
                )

            zero_gt = torch.zeros(1, dtype=torch.long, device=device)
            for row, original_batch_index in enumerate(single_rows):
                per_batch[original_batch_index] = (
                    best_pred[row:row + 1],
                    zero_gt,
                )

        multi_rows = [
            index for index, count in enumerate(packed_targets.counts)
            if count > 1
        ]

        if multi_rows and linear_sum_assignment is None:
            raise ImportError(
                "Multi-GT Hungarian matching requires scipy. "
                "Install it with: pip install scipy"
            )

        for batch_index in multi_rows:
            start = int(packed_targets.offsets[batch_index].item())
            end = int(packed_targets.offsets[batch_index + 1].item())
            gt_bbox = packed_targets.boxes[start:end].float()
            boxes = pred_bbox[batch_index].detach().float()

            cost_bbox = torch.cdist(boxes, gt_bbox, p=1)
            giou = generalized_box_iou(boxes, gt_bbox)
            cost = self.cost_bbox * cost_bbox - self.cost_giou * giou
            if self.cost_score != 0.0:
                cost = cost - self.cost_score * pred_score[batch_index, :, None]

            pred_np, gt_np = linear_sum_assignment(cost.cpu().numpy())
            per_batch[batch_index] = (
                torch.as_tensor(pred_np, device=device, dtype=torch.long),
                torch.as_tensor(gt_np, device=device, dtype=torch.long),
            )

        mode = "batched_single_gt" if not multi_rows else "mixed_fallback"
        return AssignmentResult.from_per_batch(per_batch, device=device, mode=mode)


class OneToManyMatcher:
    """Auxiliary one-to-many matcher with a batched single-GT top-k path."""

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
        packed_targets: Optional[PackedTargets] = None,
    ) -> AssignmentResult:
        batch_size, num_pred, _ = pred_bbox.shape
        if pred_score_logit.ndim == 3:
            pred_score = pred_score_logit.detach().sigmoid().squeeze(-1)
        else:
            pred_score = pred_score_logit.detach().sigmoid()

        if packed_targets is None:
            packed_targets = PackedTargets.from_targets(
                targets,
                device=pred_bbox.device,
                dtype=pred_bbox.dtype,
            )

        device = pred_bbox.device
        empty = torch.empty(0, dtype=torch.long, device=device)
        per_batch: List[Tuple[torch.Tensor, torch.Tensor]] = [
            (empty, empty) for _ in range(batch_size)
        ]

        single_rows = [
            index for index, count in enumerate(packed_targets.counts)
            if count == 1
        ]

        if num_pred > 0 and single_rows:
            batch_index = torch.tensor(single_rows, device=device, dtype=torch.long)
            global_gt_index = packed_targets.offsets[batch_index]
            gt_bbox = packed_targets.boxes[global_gt_index]
            boxes = pred_bbox.detach().index_select(0, batch_index)

            cost_bbox = torch.abs(boxes - gt_bbox[:, None, :]).sum(dim=-1)
            giou = batched_single_target_giou(boxes.float(), gt_bbox.float()).to(boxes.dtype)
            iou = batched_single_target_iou(boxes.float(), gt_bbox.float()).to(boxes.dtype)
            cost = self.cost_bbox * cost_bbox - self.cost_giou * giou
            if self.cost_score != 0.0:
                cost = cost - self.cost_score * pred_score.index_select(0, batch_index)

            ratio_budget = max(1, int(round(num_pred * self.positive_ratio)))
            target_count = min(
                num_pred,
                self.max_positive_per_gt,
                max(1, ratio_budget),
            )

            if target_count == 1:
                selected_pred = torch.argmin(cost, dim=1, keepdim=True)
                keep = torch.ones_like(selected_pred, dtype=torch.bool)
            elif self.min_extra_positive_iou <= 0.0:
                selected_pred = torch.topk(
                    cost,
                    k=target_count,
                    dim=1,
                    largest=False,
                    sorted=True,
                ).indices
                keep = torch.ones_like(selected_pred, dtype=torch.bool)
            else:
                base_pred = torch.argmin(cost, dim=1, keepdim=True)
                candidate_cost = cost.masked_fill(
                    iou < self.min_extra_positive_iou,
                    torch.finfo(cost.dtype).max,
                )
                candidate_cost.scatter_(1, base_pred, torch.finfo(cost.dtype).max)
                extra_count = target_count - 1
                extra_values, extra_pred = torch.topk(
                    candidate_cost,
                    k=extra_count,
                    dim=1,
                    largest=False,
                    sorted=True,
                )
                selected_pred = torch.cat([base_pred, extra_pred], dim=1)
                keep = torch.cat(
                    [
                        torch.ones_like(base_pred, dtype=torch.bool),
                        torch.isfinite(extra_values)
                        & (extra_values < torch.finfo(extra_values.dtype).max),
                    ],
                    dim=1,
                )

            if all(count <= 1 for count in packed_targets.counts):
                selected_batch = batch_index[:, None].expand_as(selected_pred)
                flat_batch = selected_batch[keep]
                flat_pred = selected_pred[keep].to(dtype=torch.long)
                single_counts = keep.sum(dim=1).detach().cpu().tolist()
                counts_list = [0] * batch_size
                for row, original_batch_index in enumerate(single_rows):
                    counts_list[original_batch_index] = int(single_counts[row])
                return AssignmentResult(
                    batch_indices=flat_batch.to(dtype=torch.long),
                    pred_indices=flat_pred,
                    gt_indices=torch.zeros_like(flat_pred, dtype=torch.long),
                    counts=tuple(counts_list),
                    mode="batched_single_gt",
                )

            for row, original_batch_index in enumerate(single_rows):
                pred_row = selected_pred[row][keep[row]]
                per_batch[original_batch_index] = (
                    pred_row.to(dtype=torch.long),
                    torch.zeros_like(pred_row, dtype=torch.long),
                )

        multi_rows = [
            index for index, count in enumerate(packed_targets.counts)
            if count > 1
        ]

        for batch_index in multi_rows:
            start = int(packed_targets.offsets[batch_index].item())
            end = int(packed_targets.offsets[batch_index + 1].item())
            gt_bbox = packed_targets.boxes[start:end]
            num_gt = int(gt_bbox.shape[0])
            boxes = pred_bbox[batch_index].detach()
            cost_bbox = torch.cdist(boxes, gt_bbox, p=1)
            giou_matrix = generalized_box_iou(boxes, gt_bbox)
            iou_matrix = box_iou(boxes, gt_bbox)
            cost = self.cost_bbox * cost_bbox - self.cost_giou * giou_matrix
            if self.cost_score != 0.0:
                cost = cost - self.cost_score * pred_score[batch_index, :, None]

            minimum_budget = min(num_pred, num_gt)
            ratio_budget = max(1, int(round(num_pred * self.positive_ratio)))
            maximum_budget = min(num_pred, num_gt * self.max_positive_per_gt)
            positive_budget = min(max(minimum_budget, ratio_budget), maximum_budget)

            base_pred, base_gt = self._greedy_one_to_one(cost)
            selected_pred = [int(value) for value in base_pred.cpu().tolist()]
            selected_gt = [int(value) for value in base_gt.cpu().tolist()]
            used_pred = torch.zeros(num_pred, dtype=torch.bool, device=device)
            gt_counts = torch.zeros(num_gt, dtype=torch.long, device=device)
            if base_pred.numel() > 0:
                used_pred[base_pred] = True
                gt_counts.scatter_add_(0, base_gt, torch.ones_like(base_gt))

            if len(selected_pred) < positive_budget and self.max_positive_per_gt > 1:
                candidate_k = min(num_pred, self.max_positive_per_gt)
                values, pred_indices = torch.topk(
                    cost.transpose(0, 1),
                    k=candidate_k,
                    dim=1,
                    largest=False,
                    sorted=True,
                )
                gt_indices = torch.arange(num_gt, device=device)[:, None].expand_as(pred_indices)
                order = torch.argsort(values.reshape(-1))
                flat_pred = pred_indices.reshape(-1)
                flat_gt = gt_indices.reshape(-1)

                for order_index in order.cpu().tolist():
                    if len(selected_pred) >= positive_budget:
                        break
                    pred_index = int(flat_pred[order_index].item())
                    gt_index = int(flat_gt[order_index].item())
                    if bool(used_pred[pred_index]):
                        continue
                    if int(gt_counts[gt_index]) >= self.max_positive_per_gt:
                        continue
                    if (
                        self.min_extra_positive_iou > 0
                        and float(iou_matrix[pred_index, gt_index]) < self.min_extra_positive_iou
                    ):
                        continue
                    selected_pred.append(pred_index)
                    selected_gt.append(gt_index)
                    used_pred[pred_index] = True
                    gt_counts[gt_index] += 1

            per_batch[batch_index] = (
                torch.tensor(selected_pred, device=device, dtype=torch.long),
                torch.tensor(selected_gt, device=device, dtype=torch.long),
            )

        mode = "batched_single_gt" if not multi_rows else "mixed_fallback"
        return AssignmentResult.from_per_batch(per_batch, device=device, mode=mode)


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

    def score_loss_quality_balanced(
        self,
        pred_score_logit: torch.Tensor,
        score_target: torch.Tensor,
        positive_mask: torch.Tensor,
        pos_weight: float = 1.0,
        qfl_beta=None,
        query_loss_weights: Optional[torch.Tensor] = None,
        text_negative_mask: Optional[torch.Tensor] = None,
        positive_counts: Optional[Sequence[int]] = None,
    ):
        """Vectorized QFL and per-query hard-negative mining."""
        if qfl_beta is None:
            qfl_beta = self.qfl_beta

        batch_size, num_pred, _ = pred_score_logit.shape
        if positive_counts is None:
            positive_counts = tuple(
                int(value) for value in positive_mask.reshape(batch_size, -1).sum(dim=1).cpu().tolist()
            )
        else:
            positive_counts = tuple(int(value) for value in positive_counts)

        pred_prob = pred_score_logit.sigmoid()
        bce = F.binary_cross_entropy_with_logits(
            pred_score_logit,
            score_target,
            reduction="none",
        )
        qfl_weight = (score_target - pred_prob).abs().pow(float(qfl_beta))
        loss_flat = (bce * qfl_weight).squeeze(-1)
        positive_flat = positive_mask.squeeze(-1)

        query_weight = self._prepare_query_loss_weights(
            pred_score_logit,
            query_loss_weights,
        ).reshape(batch_size)
        text_negative = self._prepare_text_negative_mask(
            pred_score_logit,
            text_negative_mask,
        )

        total_positive_count = sum(positive_counts)
        if total_positive_count > 0:
            loss_pos = loss_flat[positive_flat].mean()
        else:
            loss_pos = pred_score_logit.new_zeros(())

        negative_counts = tuple(num_pred - count for count in positive_counts)
        selected_counts = tuple(
            min(negative_count, max(positive_count, 1) * self.hard_negative_ratio)
            for positive_count, negative_count in zip(positive_counts, negative_counts)
        )
        max_selected = max(selected_counts, default=0)

        if max_selected > 0:
            negative_loss = loss_flat.masked_fill(positive_flat, float("-inf"))
            selected_values = torch.topk(
                negative_loss,
                k=max_selected,
                dim=1,
                largest=True,
                sorted=True,
            ).values
            selected_count_tensor = torch.tensor(
                selected_counts,
                device=loss_flat.device,
                dtype=torch.long,
            )
            selected_mask = (
                torch.arange(max_selected, device=loss_flat.device)[None, :]
                < selected_count_tensor[:, None]
            )
            selected_sum = torch.where(
                selected_mask,
                selected_values,
                torch.zeros_like(selected_values),
            ).sum(dim=1)
            valid_query = selected_count_tensor > 0
            query_negative_mean = selected_sum / selected_count_tensor.clamp(min=1).to(loss_flat.dtype)
            valid_count = valid_query.sum().clamp(min=1).to(loss_flat.dtype)
            loss_neg_unweighted = (
                query_negative_mean * valid_query.to(loss_flat.dtype)
            ).sum() / valid_count
            weighted_query_loss = query_negative_mean * query_weight
            loss_neg = (
                weighted_query_loss * valid_query.to(loss_flat.dtype)
            ).sum() / valid_count

            text_valid = valid_query & text_negative
            text_valid_count = text_valid.sum().clamp(min=1).to(loss_flat.dtype)
            text_negative_loss = (
                weighted_query_loss * text_valid.to(loss_flat.dtype)
            ).sum() / text_valid_count
        else:
            loss_neg = pred_score_logit.new_zeros(())
            loss_neg_unweighted = pred_score_logit.new_zeros(())
            text_negative_loss = pred_score_logit.new_zeros(())

        selected_negative_count = sum(selected_counts)
        if total_positive_count > 0 and selected_negative_count > 0:
            loss_score = (
                float(pos_weight) * loss_pos + loss_neg
            ) / (float(pos_weight) + 1.0)
        elif total_positive_count > 0:
            loss_score = loss_pos
        elif selected_negative_count > 0:
            loss_score = loss_neg
        else:
            loss_score = pred_score_logit.new_zeros(())

        text_negative_count_tensor = text_negative.sum()
        text_negative_weight_mean = torch.where(
            text_negative_count_tensor > 0,
            (query_weight * text_negative.to(query_weight.dtype)).sum()
            / text_negative_count_tensor.clamp(min=1).to(query_weight.dtype),
            query_weight.new_ones(()),
        )

        return (
            loss_score,
            loss_pos,
            loss_neg,
            loss_neg_unweighted,
            text_negative_loss,
            float(total_positive_count),
            float(selected_negative_count),
            float(sum(negative_counts)),
            float(text_negative_count_tensor.detach().cpu().item()),
            text_negative_weight_mean,
        )

    def pairwise_quality_rank_loss(
        self,
        pred_score_logit: torch.Tensor,
        score_target: torch.Tensor,
        positive_mask: torch.Tensor,
        margin=None,
        min_quality_gap=None,
        positive_counts: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Vectorized within-query quality ranking loss."""
        if margin is None:
            margin = self.rank_margin
        if min_quality_gap is None:
            min_quality_gap = self.rank_min_quality_gap

        batch_size, num_pred, _ = pred_score_logit.shape
        logits = pred_score_logit.squeeze(-1)
        quality = score_target.squeeze(-1)
        positive = positive_mask.squeeze(-1)

        if positive_counts is None:
            positive_counts = tuple(
                int(value) for value in positive.sum(dim=1).cpu().tolist()
            )
        else:
            positive_counts = tuple(int(value) for value in positive_counts)

        total_positive = sum(positive_counts)
        max_positive = max(positive_counts, default=0)
        if total_positive == 0 or max_positive == 0:
            return pred_score_logit.new_zeros(())

        positive_count_tensor = torch.tensor(
            positive_counts,
            device=logits.device,
            dtype=torch.long,
        )
        positive_values, positive_indices = torch.topk(
            logits.masked_fill(~positive, float("-inf")),
            k=max_positive,
            dim=1,
            largest=True,
            sorted=True,
        )
        positive_quality = torch.gather(quality, 1, positive_indices)
        positive_valid = (
            torch.arange(max_positive, device=logits.device)[None, :]
            < positive_count_tensor[:, None]
        )

        pair_loss_parts: List[torch.Tensor] = []

        if max_positive >= 2:
            rank_mask = (
                positive_valid[:, :, None]
                & positive_valid[:, None, :]
                & (
                    positive_quality[:, :, None]
                    > positive_quality[:, None, :] + float(min_quality_gap)
                )
            )
            positive_difference = (
                positive_values[:, :, None] - positive_values[:, None, :]
            )
            positive_pair_loss = F.softplus(
                float(margin) - positive_difference
            )
            pair_loss_parts.append(positive_pair_loss[rank_mask])

        negative_counts = tuple(num_pred - count for count in positive_counts)
        selected_negative_counts = tuple(
            min(negative_count, max(positive_count * self.hard_negative_ratio, 1))
            if positive_count > 0
            else 0
            for positive_count, negative_count in zip(positive_counts, negative_counts)
        )
        max_negative = max(selected_negative_counts, default=0)

        if max_negative > 0:
            negative_values = torch.topk(
                logits.masked_fill(positive, float("-inf")),
                k=max_negative,
                dim=1,
                largest=True,
                sorted=True,
            ).values
            selected_negative_count_tensor = torch.tensor(
                selected_negative_counts,
                device=logits.device,
                dtype=torch.long,
            )
            negative_valid = (
                torch.arange(max_negative, device=logits.device)[None, :]
                < selected_negative_count_tensor[:, None]
            )
            positive_negative_mask = (
                positive_valid[:, :, None] & negative_valid[:, None, :]
            )
            logit_difference = (
                positive_values[:, :, None] - negative_values[:, None, :]
            )
            positive_negative_loss = F.softplus(
                float(margin) - logit_difference
            )
            pair_loss_parts.append(
                positive_negative_loss[positive_negative_mask]
            )

        if not pair_loss_parts:
            return pred_score_logit.new_zeros(())

        all_pair_losses = torch.cat(pair_loss_parts)
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
        packed_targets: PackedTargets,
        assignments: AssignmentResult,
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

        score_target = torch.zeros_like(pred_score_logit)
        positive_mask = torch.zeros_like(pred_score_logit, dtype=torch.bool)
        total_positive_pairs = assignments.num_matches
        warmup_positive_target = clamp01(warmup_positive_target)

        if total_positive_pairs > 0:
            batch_index = assignments.batch_indices
            pred_index = assignments.pred_indices
            global_gt_index = packed_targets.global_gt_indices(
                batch_index,
                assignments.gt_indices,
            )
            positive_pred_boxes = pred_bbox[batch_index, pred_index]
            positive_gt_boxes = packed_targets.boxes[global_gt_index].to(
                dtype=pred_bbox.dtype
            )

            loss_bbox = F.l1_loss(
                positive_pred_boxes,
                positive_gt_boxes,
                reduction="sum",
            ) / total_positive_pairs

            matched_giou = matched_generalized_box_iou(
                positive_pred_boxes.float(),
                positive_gt_boxes.float(),
            ).to(dtype=pred_bbox.dtype)
            loss_giou = (1.0 - matched_giou).sum() / total_positive_pairs

            with torch.no_grad():
                matched_iou = matched_box_iou(
                    positive_pred_boxes.detach().float(),
                    positive_gt_boxes.float(),
                ).clamp(0.0, 1.0).to(dtype=pred_score_logit.dtype)
                final_quality_target = matched_iou.clamp(
                    min=self.quality_min,
                    max=self.quality_max,
                )
                quality_target = (
                    (1.0 - float(quality_alpha)) * warmup_positive_target
                    + float(quality_alpha) * final_quality_target
                ).clamp(0.0, 1.0)

            score_target[batch_index, pred_index, 0] = quality_target
            positive_mask[batch_index, pred_index, 0] = True
            matched_iou_mean = matched_iou.mean()
        else:
            zero = pred_bbox.new_zeros(())
            loss_bbox = zero
            loss_giou = zero
            matched_iou_mean = zero

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
            positive_counts=assignments.counts,
        )

        if bool(apply_ranking) and float(lambda_rank) > 0.0 and float(rank_alpha) > 0.0:
            loss_rank_raw = self.pairwise_quality_rank_loss(
                pred_score_logit=pred_score_logit,
                score_target=score_target,
                positive_mask=positive_mask,
                margin=self.rank_margin,
                min_quality_gap=self.rank_min_quality_gap,
                positive_counts=assignments.counts,
            )
            loss_rank = float(rank_alpha) * loss_rank_raw
            loss_rank_contrib = float(lambda_rank) * loss_rank
        else:
            loss_rank_raw = pred_score_logit.new_zeros(())
            loss_rank = pred_score_logit.new_zeros(())
            loss_rank_contrib = pred_score_logit.new_zeros(())

        loss_base = (
            float(lambda_bbox) * loss_bbox
            + float(lambda_giou) * loss_giou
            + float(lambda_score) * loss_score
        )
        loss_total = loss_base + loss_rank_contrib

        if total_positive_pairs > 0:
            positive_targets = score_target[positive_mask]
            score_target_pos_mean = positive_targets.mean()
            score_target_pos_min = positive_targets.min()
            score_target_pos_max = positive_targets.max()
        else:
            zero = pred_bbox.new_zeros(())
            score_target_pos_mean = zero
            score_target_pos_min = zero
            score_target_pos_max = zero

        selected_negative_fraction = (
            float(hard_negative_count) / max(float(negative_count), 1.0)
        )

        metrics: Dict[str, Any] = {
            "loss_total": loss_total,
            "loss_base": loss_base,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_score": loss_score,
            "loss_score_pos": loss_score_pos,
            "loss_score_neg": loss_score_neg,
            "loss_score_neg_unweighted": loss_score_neg_unweighted,
            "loss_text_negative": loss_text_negative,
            "loss_rank": loss_rank,
            "loss_rank_raw": loss_rank_raw,
            "loss_rank_contrib": loss_rank_contrib,
            "matched": float(total_positive_pairs),
            "score_pos_count": float(score_pos_count),
            "hard_neg_count": float(hard_negative_count),
            "negative_count": float(negative_count),
            "selected_negative_fraction": float(selected_negative_fraction),
            "text_negative_count": float(text_negative_count),
            "text_negative_weight_mean": text_negative_weight_mean,
            "matched_iou_mean": matched_iou_mean,
            "score_target_pos_mean": score_target_pos_mean,
            "score_target_pos_min": score_target_pos_min,
            "score_target_pos_max": score_target_pos_max,
            "lambda_bbox": float(lambda_bbox),
            "lambda_giou": float(lambda_giou),
            "lambda_score": float(lambda_score),
            "lambda_rank": float(lambda_rank if apply_ranking else 0.0),
            "lambda_rank_eff": float(lambda_rank * rank_alpha if apply_ranking else 0.0),
            "pos_weight": float(pos_weight),
            "quality_alpha": float(quality_alpha),
            "rank_alpha": float(rank_alpha if apply_ranking else 0.0),
            "assignment_mode": assignments.mode,
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

        packed_targets = PackedTargets.from_targets(
            targets,
            device=pred_bbox.device,
            dtype=pred_bbox.dtype,
        )

        main_assignments = self.main_matcher(
            pred_bbox=pred_bbox,
            pred_score_logit=pred_score_logit,
            targets=targets,
            packed_targets=packed_targets,
        )

        main_loss, main_metrics = (
            self._compute_branch_loss(
                branch_name="main",
                pred_bbox=pred_bbox,
                pred_score_logit=(
                    pred_score_logit
                ),
                targets=targets,
                packed_targets=packed_targets,
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
                packed_targets=packed_targets,
            )

            aux_loss, aux_metrics = (
                self._compute_branch_loss(
                    branch_name="aux",
                    pred_bbox=aux_pred_bbox,
                    pred_score_logit=(
                        aux_pred_score_logit
                    ),
                    targets=targets,
                    packed_targets=packed_targets,
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

