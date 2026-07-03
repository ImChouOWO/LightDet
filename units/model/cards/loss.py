from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import math
import numpy as np
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


def batched_pairwise_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """
    Batched pairwise IoU.

    boxes1: [B, N, 4]
    boxes2: [B, M, 4]
    returns: [B, N, M]
    """
    if boxes1.ndim != 3 or boxes2.ndim != 3:
        raise ValueError(
            "batched_pairwise_iou expects [B, N, 4] and [B, M, 4]"
        )
    if boxes1.shape[0] != boxes2.shape[0]:
        raise ValueError("Batch size mismatch in batched_pairwise_iou")
    if boxes1.shape[-1] != 4 or boxes2.shape[-1] != 4:
        raise ValueError("Last dimension must be 4")

    if boxes1.shape[1] == 0 or boxes2.shape[1] == 0:
        return boxes1.new_zeros(
            (
                boxes1.shape[0],
                boxes1.shape[1],
                boxes2.shape[1],
            )
        )

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.maximum(
        boxes1[:, :, None, :2],
        boxes2[:, None, :, :2],
    )
    rb = torch.minimum(
        boxes1[:, :, None, 2:],
        boxes2[:, None, :, 2:],
    )
    wh = (rb - lt).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    union = (
        area1[:, :, None]
        + area2[:, None, :]
        - intersection
    )
    return intersection / union.clamp(min=1e-6)


def batched_pairwise_giou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """
    Batched pairwise generalized IoU.

    boxes1: [B, N, 4]
    boxes2: [B, M, 4]
    returns: [B, N, M]
    """
    if boxes1.ndim != 3 or boxes2.ndim != 3:
        raise ValueError(
            "batched_pairwise_giou expects [B, N, 4] and [B, M, 4]"
        )
    if boxes1.shape[0] != boxes2.shape[0]:
        raise ValueError("Batch size mismatch in batched_pairwise_giou")
    if boxes1.shape[-1] != 4 or boxes2.shape[-1] != 4:
        raise ValueError("Last dimension must be 4")

    if boxes1.shape[1] == 0 or boxes2.shape[1] == 0:
        return boxes1.new_zeros(
            (
                boxes1.shape[0],
                boxes1.shape[1],
                boxes2.shape[1],
            )
        )

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.maximum(
        boxes1[:, :, None, :2],
        boxes2[:, None, :, :2],
    )
    rb = torch.minimum(
        boxes1[:, :, None, 2:],
        boxes2[:, None, :, 2:],
    )
    wh = (rb - lt).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    union = (
        area1[:, :, None]
        + area2[:, None, :]
        - intersection
    )
    iou = intersection / union.clamp(min=1e-6)

    cover_lt = torch.minimum(
        boxes1[:, :, None, :2],
        boxes2[:, None, :, :2],
    )
    cover_rb = torch.maximum(
        boxes1[:, :, None, 2:],
        boxes2[:, None, :, 2:],
    )
    cover_wh = (cover_rb - cover_lt).clamp(min=0)
    cover_area = cover_wh[..., 0] * cover_wh[..., 1]

    return iou - (
        cover_area - union
    ) / cover_area.clamp(min=1e-6)


def group_rows_by_target_count(
    counts: Sequence[int],
    *,
    minimum_count: int = 2,
) -> Dict[int, List[int]]:
    """Group query rows that contain the same number of GT boxes."""
    groups: Dict[int, List[int]] = {}
    for batch_index, count in enumerate(counts):
        count = int(count)
        if count < int(minimum_count):
            continue
        groups.setdefault(count, []).append(batch_index)
    return groups


def gather_grouped_targets(
    packed_targets: "PackedTargets",
    batch_indices: torch.Tensor,
    target_count: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Gather equal-length target rows as [G, target_count, 4]."""
    local_index = torch.arange(
        int(target_count),
        device=batch_indices.device,
        dtype=torch.long,
    )
    global_index = (
        packed_targets.offsets[batch_indices, None]
        + local_index[None, :]
    )
    return packed_targets.boxes[
        global_index
    ].to(dtype=dtype)


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
    """
    DETR-style one-to-one Hungarian matcher.

    Design:
      - Standard cost = score + L1 + GIoU, following DETR-family practice.
      - Score cost can be smoothly scheduled from 0 -> configured weight.
      - Text-negative rows are converted to empty targets before matching.
      - 0/1-GT rows stay fully vectorized on GPU.
      - Multi-GT rows are grouped by equal target count and copied to CPU once
        per group for SciPy Hungarian assignment.
    """

    def __init__(
        self,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cost_score: float = 2.0,
        score_cost_type: str = "focal",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        if (
            float(cost_bbox) == 0.0
            and float(cost_giou) == 0.0
            and float(cost_score) == 0.0
        ):
            raise ValueError(
                "At least one Hungarian matching cost must be non-zero."
            )

        score_cost_type = str(score_cost_type).strip().lower()
        aliases = {
            "focal": "focal",
            "focal_loss": "focal",
            "prob": "probability",
            "probability": "probability",
            "score": "probability",
        }
        if score_cost_type not in aliases:
            raise ValueError(
                "score_cost_type must be 'focal' or 'probability', got "
                f"{score_cost_type!r}"
            )

        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)
        self.cost_score = float(cost_score)
        self.score_cost_type = aliases[score_cost_type]
        self.focal_alpha = clamp01(focal_alpha)
        self.focal_gamma = max(0.0, float(focal_gamma))

    def _score_cost(self, score_prob: torch.Tensor) -> torch.Tensor:
        """
        Positive-class matching cost.

        For focal mode this follows the DETR focal matcher:
            positive_cost - negative_cost
        so high confidence produces a lower assignment cost.
        """
        score_prob = score_prob.clamp(1e-6, 1.0 - 1e-6)
        if self.score_cost_type == "probability":
            return -score_prob

        alpha = float(self.focal_alpha)
        gamma = float(self.focal_gamma)
        negative_cost = (
            (1.0 - alpha)
            * score_prob.pow(gamma)
            * (-(1.0 - score_prob).log())
        )
        positive_cost = (
            alpha
            * (1.0 - score_prob).pow(gamma)
            * (-score_prob.log())
        )
        return positive_cost - negative_cost

    @torch.no_grad()
    def __call__(
        self,
        pred_bbox: torch.Tensor,
        pred_score_logit: torch.Tensor,
        targets: List[dict],
        packed_targets: Optional[PackedTargets] = None,
        *,
        score_cost_alpha: float = 1.0,
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
            raise ValueError(
                f"targets batch size mismatch: {len(targets)} != {batch_size}"
            )

        if packed_targets is None:
            packed_targets = PackedTargets.from_targets(
                targets,
                device=pred_bbox.device,
                dtype=pred_bbox.dtype,
            )

        device = pred_bbox.device
        score_cost_alpha = clamp01(score_cost_alpha)
        empty = torch.empty(0, dtype=torch.long, device=device)
        per_batch: List[Tuple[torch.Tensor, torch.Tensor]] = [
            (empty, empty) for _ in range(batch_size)
        ]

        pred_score = score_logit.detach().float().sigmoid()
        score_cost_all = self._score_cost(pred_score)

        single_rows = [
            index
            for index, count in enumerate(packed_targets.counts)
            if count == 1
        ]

        if num_pred > 0 and single_rows:
            batch_index = torch.tensor(
                single_rows,
                device=device,
                dtype=torch.long,
            )
            global_gt_index = packed_targets.offsets[batch_index]
            gt_bbox = packed_targets.boxes[global_gt_index].float()
            boxes = pred_bbox.detach().float().index_select(
                0,
                batch_index,
            )

            cost_bbox = torch.abs(
                boxes - gt_bbox[:, None, :]
            ).sum(dim=-1)
            giou = batched_single_target_giou(boxes, gt_bbox)
            cost = (
                self.cost_bbox * cost_bbox
                - self.cost_giou * giou
            )
            if self.cost_score != 0.0 and score_cost_alpha > 0.0:
                cost = (
                    cost
                    + self.cost_score
                    * score_cost_alpha
                    * score_cost_all.index_select(0, batch_index)
                )

            best_pred = torch.argmin(cost, dim=1)

            if all(count <= 1 for count in packed_targets.counts):
                counts = tuple(
                    1 if count == 1 else 0
                    for count in packed_targets.counts
                )
                return AssignmentResult(
                    batch_indices=batch_index,
                    pred_indices=best_pred.long(),
                    gt_indices=torch.zeros_like(
                        best_pred,
                        dtype=torch.long,
                    ),
                    counts=counts,
                    mode=(
                        "detr_batched_single_gt_"
                        f"scorealpha_{score_cost_alpha:.3f}"
                    ),
                )

            zero_gt = torch.zeros(
                1,
                dtype=torch.long,
                device=device,
            )
            for row, original_batch_index in enumerate(single_rows):
                per_batch[original_batch_index] = (
                    best_pred[row:row + 1],
                    zero_gt,
                )

        multi_groups = group_rows_by_target_count(
            packed_targets.counts,
            minimum_count=2,
        )

        if multi_groups and linear_sum_assignment is None:
            raise ImportError(
                "Multi-GT Hungarian matching requires scipy. "
                "Install it with: pip install scipy"
            )

        for num_gt, rows in multi_groups.items():
            row_tensor = torch.tensor(
                rows,
                device=device,
                dtype=torch.long,
            )
            boxes = pred_bbox.detach().float().index_select(
                0,
                row_tensor,
            )
            gt_bbox = gather_grouped_targets(
                packed_targets,
                row_tensor,
                num_gt,
                dtype=torch.float32,
            )

            cost_bbox = torch.abs(
                boxes[:, :, None, :]
                - gt_bbox[:, None, :, :]
            ).sum(dim=-1)
            giou = batched_pairwise_giou(boxes, gt_bbox)
            cost = (
                self.cost_bbox * cost_bbox
                - self.cost_giou * giou
            )
            if self.cost_score != 0.0 and score_cost_alpha > 0.0:
                cost = (
                    cost
                    + self.cost_score
                    * score_cost_alpha
                    * score_cost_all.index_select(
                        0,
                        row_tensor,
                    )[:, :, None]
                )

            cost_cpu = cost.detach().cpu().numpy()
            for group_row, batch_index in enumerate(rows):
                pred_np, gt_np = linear_sum_assignment(
                    cost_cpu[group_row]
                )
                per_batch[batch_index] = (
                    torch.as_tensor(
                        pred_np,
                        device=device,
                        dtype=torch.long,
                    ),
                    torch.as_tensor(
                        gt_np,
                        device=device,
                        dtype=torch.long,
                    ),
                )

        mode = (
            "detr_batched_single_gt"
            if not multi_groups
            else "detr_grouped_multi_gt_scipy"
        )
        mode += f"_scorealpha_{score_cost_alpha:.3f}"
        return AssignmentResult.from_per_batch(
            per_batch,
            device=device,
            mode=mode,
        )


class HDETRRepeatedHungarianMatcher:
    """
    H-DETR-style auxiliary matcher.

    Each GT is repeated K times, then a bipartite matching is performed
    against the augmented target set. Score logits are intentionally excluded
    from the cost so confidence learning cannot change regression assignment.

    Fast paths:
      - 0/1 GT rows are fully vectorized with batched Top-K.
      - Multi-GT rows are grouped by equal GT count. Base costs are computed
        in batched GPU tensors and copied to CPU once per group. SciPy
        Hungarian remains per query because SciPy has no batched API.

    max_positive_per_gt is the H-DETR repeat factor K. positive_ratio is kept
    only for backward compatibility and is not used.
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
        if float(cost_bbox) == 0.0 and float(cost_giou) == 0.0:
            raise ValueError(
                "At least one H-DETR regression matching cost must be non-zero."
            )

        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)
        self.repeat_k = max(1, int(max_positive_per_gt))
        self.min_extra_positive_iou = clamp01(min_extra_positive_iou)

        # Retained for config compatibility. It is intentionally not used.
        self.configured_cost_score = float(cost_score)
        self.configured_positive_ratio = float(positive_ratio)

    @staticmethod
    def _filter_extra_matches(
        pred_indices: torch.Tensor,
        gt_indices: torch.Tensor,
        pair_iou: torch.Tensor,
        *,
        threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pred_indices.numel() == 0 or threshold <= 0.0:
            return pred_indices, gt_indices

        keep = pair_iou >= float(threshold)

        # Preserve at least the best repeated match for every represented GT.
        for gt_index in torch.unique(gt_indices):
            positions = torch.nonzero(
                gt_indices == gt_index,
                as_tuple=False,
            ).squeeze(1)
            if positions.numel() == 0:
                continue
            best_position = positions[
                torch.argmax(pair_iou[positions])
            ]
            keep[best_position] = True

        return pred_indices[keep], gt_indices[keep]

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

        batch_size, num_pred, _ = pred_bbox.shape
        if len(targets) != batch_size:
            raise ValueError(
                f"targets batch size mismatch: {len(targets)} != {batch_size}"
            )

        if packed_targets is None:
            packed_targets = PackedTargets.from_targets(
                targets,
                device=pred_bbox.device,
                dtype=pred_bbox.dtype,
            )

        device = pred_bbox.device
        empty = torch.empty(
            0,
            dtype=torch.long,
            device=device,
        )
        per_batch: List[Tuple[torch.Tensor, torch.Tensor]] = [
            (empty, empty) for _ in range(batch_size)
        ]

        single_rows = [
            index
            for index, count in enumerate(packed_targets.counts)
            if count == 1
        ]

        if num_pred > 0 and single_rows:
            batch_index = torch.tensor(
                single_rows,
                device=device,
                dtype=torch.long,
            )
            global_gt_index = packed_targets.offsets[batch_index]
            gt_bbox = packed_targets.boxes[global_gt_index].float()
            boxes = pred_bbox.detach().float().index_select(
                0,
                batch_index,
            )

            cost_bbox = torch.abs(
                boxes - gt_bbox[:, None, :]
            ).sum(dim=-1)
            giou = batched_single_target_giou(
                boxes,
                gt_bbox,
            )
            iou = batched_single_target_iou(
                boxes,
                gt_bbox,
            )
            cost = (
                self.cost_bbox * cost_bbox
                - self.cost_giou * giou
            )

            k = min(self.repeat_k, num_pred)
            _, selected_pred = torch.topk(
                cost,
                k=k,
                dim=1,
                largest=False,
                sorted=True,
            )
            selected_iou = torch.gather(
                iou,
                1,
                selected_pred,
            )

            if self.min_extra_positive_iou > 0.0 and k > 1:
                keep = (
                    selected_iou
                    >= self.min_extra_positive_iou
                )
                keep[:, 0] = True
            else:
                keep = torch.ones_like(
                    selected_pred,
                    dtype=torch.bool,
                )

            if all(
                count <= 1
                for count in packed_targets.counts
            ):
                selected_batch = (
                    batch_index[:, None]
                    .expand_as(selected_pred)
                )
                flat_batch = selected_batch[keep]
                flat_pred = selected_pred[keep].long()
                single_counts = (
                    keep.sum(dim=1)
                    .detach()
                    .cpu()
                    .tolist()
                )
                counts_list = [0] * batch_size
                for row, original_batch_index in enumerate(single_rows):
                    counts_list[original_batch_index] = int(
                        single_counts[row]
                    )
                return AssignmentResult(
                    batch_indices=flat_batch.long(),
                    pred_indices=flat_pred,
                    gt_indices=torch.zeros_like(
                        flat_pred
                    ),
                    counts=tuple(counts_list),
                    mode="hdetr_repeated_gt_batched",
                )

            for row, original_batch_index in enumerate(single_rows):
                pred_row = (
                    selected_pred[row][keep[row]]
                    .long()
                )
                per_batch[original_batch_index] = (
                    pred_row,
                    torch.zeros_like(pred_row),
                )

        multi_groups = group_rows_by_target_count(
            packed_targets.counts,
            minimum_count=2,
        )

        if multi_groups and linear_sum_assignment is None:
            raise ImportError(
                "Multi-GT H-DETR matching requires scipy. "
                "Install it with: pip install scipy"
            )

        for num_gt, rows in multi_groups.items():
            row_tensor = torch.tensor(
                rows,
                device=device,
                dtype=torch.long,
            )
            boxes = pred_bbox.detach().float().index_select(
                0,
                row_tensor,
            )
            gt_bbox = gather_grouped_targets(
                packed_targets,
                row_tensor,
                num_gt,
                dtype=torch.float32,
            )

            cost_bbox = torch.abs(
                boxes[:, :, None, :]
                - gt_bbox[:, None, :, :]
            ).sum(dim=-1)
            giou = batched_pairwise_giou(
                boxes,
                gt_bbox,
            )
            iou = batched_pairwise_iou(
                boxes,
                gt_bbox,
            )
            base_cost = (
                self.cost_bbox * cost_bbox
                - self.cost_giou * giou
            )

            repeated_gt = torch.arange(
                num_gt,
                device=device,
                dtype=torch.long,
            ).repeat(self.repeat_k)
            expanded_cost = base_cost.index_select(
                2,
                repeated_gt,
            )

            # One GPU->CPU synchronization per equal-GT-count group.
            expanded_cost_cpu = (
                expanded_cost.detach().cpu().numpy()
            )

            for group_row, batch_index in enumerate(rows):
                pred_np, repeated_column_np = (
                    linear_sum_assignment(
                        expanded_cost_cpu[group_row]
                    )
                )
                pred_index = torch.as_tensor(
                    pred_np,
                    device=device,
                    dtype=torch.long,
                )
                repeated_column = torch.as_tensor(
                    repeated_column_np,
                    device=device,
                    dtype=torch.long,
                )
                gt_index = repeated_gt[
                    repeated_column
                ]
                pair_iou = iou[
                    group_row,
                    pred_index,
                    gt_index,
                ]

                pred_index, gt_index = (
                    self._filter_extra_matches(
                        pred_index,
                        gt_index,
                        pair_iou,
                        threshold=(
                            self.min_extra_positive_iou
                        ),
                    )
                )
                per_batch[batch_index] = (
                    pred_index,
                    gt_index,
                )

        mode = (
            "hdetr_repeated_gt_batched"
            if not multi_groups
            else "hdetr_repeated_gt_grouped_scipy"
        )
        return AssignmentResult.from_per_batch(
            per_batch,
            device=device,
            mode=mode,
        )


# Backward-compatible name used by older imports.
OneToManyMatcher = HDETRRepeatedHungarianMatcher




class GroundingLoss(nn.Module):
    """
    DETR-native hybrid grounding loss for LightDet.

    Active training path:
      1. Main branch uses a single Hungarian one-to-one assignment with
         score + L1 + GIoU costs.
      2. Main classification uses one-to-one quality-aware BCE/Focal targets
         derived only from the matched pair.
      3. Unmatched queries are trained as no-object, except duplicate-like
         queries with high IoU to a GT. These are removed from background BCE
         and handled by a duplicate-ranking objective against the matched TP.
      4. Low-IoU, high-score unmatched queries can receive an additional
         hard-negative loss. Negative-text rows remain all-negative.
      5. Auxiliary branch keeps H-DETR repeated-GT one-to-many regression.
      5. Negative-text rows are converted to empty targets before both main and
         auxiliary matching. Therefore they never receive positive box loss.

    Compatibility:
      - The public forward signature and legacy metric keys are retained.
      - rank_start_epoch/rank_warmup_epoch/rank_alpha_min now schedule how
        strongly the text-aware score participates in Hungarian assignment.
      - lambda_rank is accepted but intentionally ignored.
    """

    def __init__(
        self,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cost_score: float = 2.0,
        hard_negative_ratio: int = 5,
        positive_ratio: float = 0.05,
        max_positive_per_gt: int = 2,
        aux_positive_label: float = 0.7,
        expand_cost_bbox: float = 5.0,
        expand_cost_giou: float = 2.0,
        iou_pos_thr: float = 0.0,
        quality_min: float = 0.10,
        quality_max: float = 1.0,
        qfl_beta: float = 2.0,
        rank_margin: float = 0.1,
        rank_min_quality_gap: float = 0.1,
        rank_max_pairs: int = 512,
        max_query_loss_weight: float = 10.0,
        score_negative_iou_ignore_thr: Optional[float] = 0.50,
        score_negative_iou_ignore_start: float = 0.50,
        score_negative_iou_ignore_end: float = 0.45,
        score_negative_iou_ignore_start_epoch: int = 5,
        score_negative_iou_ignore_end_epoch: int = 25,
        score_negative_iou_ignore_schedule: str = "cosine",
        rank_negative_iou_max: float = 0.20,
        duplicate_suppression_enabled: bool = True,
        duplicate_loss_weight: float = 0.10,
        duplicate_margin: float = 0.25,
        duplicate_background_weight: float = 0.05,
        duplicate_max_pairs: int = 128,
        duplicate_start_epoch: int = 5,
        hard_negative_mining_enabled: bool = True,
        hard_negative_loss_weight: float = 0.05,
        hard_negative_topk: int = 10,
        hard_negative_max_iou: float = 0.30,
        hard_negative_start_epoch: int = 10,
        text_negative_loss_weight: float = 0.20,
        text_negative_topk: int = 20,
        text_negative_hard_mix: float = 0.50,
        aux_loss_weight: float = 0.28,
        aux_cost_score: Optional[float] = None,
        score_match_rounds: int = 1,
        score_quality_gamma: float = 1.0,
        score_round_decay: float = 0.25,
        score_min_iou: float = 0.0,
        enable_pairwise_ranking: bool = False,
        aux_score_enabled: bool = False,
        # New DETR-native options.
        classification_type: str = "ia_bce",
        ia_bce_alpha: float = 0.25,
        focal_alpha: float = 0.25,
        focal_gamma: Optional[float] = None,
        matcher_score_cost_type: str = "focal",
        normalize_classification_by_num_gt: bool = True,
        negative_text_as_empty_target: bool = True,
        # Stored defaults used when forward does not receive explicit values.
        quality_warmup_epoch: int = 10,
        matcher_score_start_epoch: int = 5,
        matcher_score_warmup_epoch: int = 12,
        matcher_score_alpha_min: float = 0.0,
    ) -> None:
        super().__init__()

        classification_type = str(classification_type).strip().lower()
        classification_aliases = {
            "ia_bce": "ia_bce",
            "iabce": "ia_bce",
            "align_detr": "ia_bce",
            "normalized_giou": "normalized_giou",
            "giou_aware": "normalized_giou",
            "rank_detr": "normalized_giou",
            "iou": "iou",
        }
        if classification_type not in classification_aliases:
            raise ValueError(
                "classification_type must be one of "
                "['ia_bce', 'normalized_giou', 'iou'], got "
                f"{classification_type!r}"
            )

        self.classification_type = classification_aliases[
            classification_type
        ]
        self.ia_bce_alpha = clamp01(ia_bce_alpha)
        self.focal_alpha = clamp01(focal_alpha)
        self.focal_gamma = max(
            0.0,
            float(qfl_beta if focal_gamma is None else focal_gamma),
        )
        self.quality_min = clamp01(quality_min)
        self.quality_max = clamp01(quality_max)
        if self.quality_min > self.quality_max:
            raise ValueError(
                "quality_min must be <= quality_max, got "
                f"{self.quality_min} > {self.quality_max}"
            )

        self.normalize_classification_by_num_gt = bool(
            normalize_classification_by_num_gt
        )
        self.negative_text_as_empty_target = bool(
            negative_text_as_empty_target
        )
        self.aux_loss_weight = max(0.0, float(aux_loss_weight))
        self.aux_score_enabled = bool(aux_score_enabled)
        self.max_query_loss_weight = max(
            1.0,
            float(max_query_loss_weight),
        )

        # Main-branch unmatched queries whose IoU with any GT is at or
        # above this threshold are ignored by the classification loss.
        # They are not converted into positives and receive no box loss.
        self.score_negative_iou_ignore_thr = (
            0.0
            if score_negative_iou_ignore_thr is None
            else clamp01(score_negative_iou_ignore_thr)
        )

        # Duplicate-like unmatched queries are not background. They are
        # ranked below the Hungarian matched query for the same GT.
        self.duplicate_suppression_enabled = bool(
            duplicate_suppression_enabled
        )
        self.duplicate_loss_weight = max(
            0.0,
            float(duplicate_loss_weight),
        )
        self.duplicate_margin = max(0.0, float(duplicate_margin))
        self.duplicate_background_weight = max(
            0.0,
            float(duplicate_background_weight),
        )
        self.duplicate_max_pairs = max(1, int(duplicate_max_pairs))
        self.duplicate_start_epoch = max(1, int(duplicate_start_epoch))

        # Extra emphasis for high-score unmatched boxes that are genuinely
        # far from every GT. Base IA-BCE still supervises these as target 0.
        self.hard_negative_mining_enabled = bool(
            hard_negative_mining_enabled
        )
        self.hard_negative_ratio = max(1, int(hard_negative_ratio))
        self.hard_negative_loss_weight = max(
            0.0,
            float(hard_negative_loss_weight),
        )
        self.hard_negative_topk = max(1, int(hard_negative_topk))
        self.hard_negative_max_iou = clamp01(hard_negative_max_iou)
        self.hard_negative_start_epoch = max(
            1,
            int(hard_negative_start_epoch),
        )

        self.text_negative_loss_weight = max(
            0.0,
            float(text_negative_loss_weight),
        )
        self.text_negative_topk = max(1, int(text_negative_topk))
        self.text_negative_hard_mix = clamp01(
            text_negative_hard_mix
        )

        self.default_quality_warmup_epoch = max(
            0,
            int(quality_warmup_epoch),
        )
        self.default_matcher_score_start_epoch = max(
            1,
            int(matcher_score_start_epoch),
        )
        self.default_matcher_score_warmup_epoch = max(
            1,
            int(matcher_score_warmup_epoch),
        )
        self.default_matcher_score_alpha_min = clamp01(
            matcher_score_alpha_min
        )

        self.main_matcher = HungarianOneToOneMatcher(
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
            cost_score=cost_score,
            score_cost_type=matcher_score_cost_type,
            focal_alpha=self.focal_alpha,
            focal_gamma=self.focal_gamma,
        )
        self.aux_matcher = HDETRRepeatedHungarianMatcher(
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
            cost_score=0.0,
            positive_ratio=positive_ratio,
            max_positive_per_gt=max_positive_per_gt,
            min_extra_positive_iou=iou_pos_thr,
        )
        self.matcher = self.main_matcher

        # Kept only so existing constructors/config readers do not break.
        self.legacy_parameters: Dict[str, Any] = {
            "hard_negative_ratio": int(hard_negative_ratio),
            "aux_positive_label": float(aux_positive_label),
            "rank_margin": float(rank_margin),
            "rank_min_quality_gap": float(rank_min_quality_gap),
            "rank_max_pairs": int(rank_max_pairs),
            "score_negative_iou_ignore_thr": self.score_negative_iou_ignore_thr,
            "score_negative_iou_ignore_start": float(
                score_negative_iou_ignore_start
            ),
            "score_negative_iou_ignore_end": float(
                score_negative_iou_ignore_end
            ),
            "score_negative_iou_ignore_start_epoch": int(
                score_negative_iou_ignore_start_epoch
            ),
            "score_negative_iou_ignore_end_epoch": int(
                score_negative_iou_ignore_end_epoch
            ),
            "score_negative_iou_ignore_schedule": str(
                score_negative_iou_ignore_schedule
            ),
            "rank_negative_iou_max": float(rank_negative_iou_max),
            "duplicate_suppression_enabled": bool(
                self.duplicate_suppression_enabled
            ),
            "duplicate_loss_weight": float(self.duplicate_loss_weight),
            "duplicate_margin": float(self.duplicate_margin),
            "duplicate_background_weight": float(
                self.duplicate_background_weight
            ),
            "duplicate_max_pairs": int(self.duplicate_max_pairs),
            "duplicate_start_epoch": int(self.duplicate_start_epoch),
            "hard_negative_mining_enabled": bool(
                self.hard_negative_mining_enabled
            ),
            "hard_negative_loss_weight": float(
                self.hard_negative_loss_weight
            ),
            "hard_negative_topk": int(self.hard_negative_topk),
            "hard_negative_max_iou": float(self.hard_negative_max_iou),
            "hard_negative_start_epoch": int(
                self.hard_negative_start_epoch
            ),
            "aux_cost_score": (
                float(cost_score)
                if aux_cost_score is None
                else float(aux_cost_score)
            ),
            "score_match_rounds": int(score_match_rounds),
            "score_quality_gamma": float(score_quality_gamma),
            "score_round_decay": float(score_round_decay),
            "score_min_iou": float(score_min_iou),
            "enable_pairwise_ranking": bool(enable_pairwise_ranking),
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GroundingLoss":
        """
        Build directly from either the complete YAML dictionary or its
        ``loss`` section. This prevents silent omission of new parameters.
        """
        loss_cfg = config.get("loss", config)
        matcher = dict(loss_cfg.get("matcher", {}))
        quality = dict(loss_cfg.get("quality", {}))
        classification = dict(loss_cfg.get("classification", {}))
        hybrid = dict(loss_cfg.get("hybrid", {}))
        sampling = dict(loss_cfg.get("score_sampling", {}))
        text_negative = dict(loss_cfg.get("text_negative", {}))
        duplicate = dict(loss_cfg.get("duplicate_suppression", {}))
        hard_negative = dict(loss_cfg.get("hard_negative", {}))
        ranking = dict(loss_cfg.get("ranking", {}))
        matcher_schedule = dict(
            loss_cfg.get("matcher_schedule", {})
        )

        classification_type = classification.get(
            "type",
            quality.get("classification_type", "ia_bce"),
        )
        focal_gamma = classification.get(
            "focal_gamma",
            quality.get("qfl_beta", 2.0),
        )

        return cls(
            cost_bbox=matcher.get("cost_bbox", 5.0),
            cost_giou=matcher.get("cost_giou", 2.0),
            cost_score=matcher.get("cost_score", 2.0),
            matcher_score_cost_type=matcher.get(
                "score_cost_type",
                "focal",
            ),
            focal_alpha=matcher.get(
                "focal_alpha",
                classification.get("focal_alpha", 0.25),
            ),
            focal_gamma=focal_gamma,
            max_positive_per_gt=sampling.get(
                "max_positive_per_gt",
                2,
            ),
            positive_ratio=sampling.get("positive_ratio", 1.0),
            aux_positive_label=sampling.get(
                "aux_positive_label",
                0.7,
            ),
            expand_cost_bbox=sampling.get(
                "expand_cost_bbox",
                matcher.get("cost_bbox", 5.0),
            ),
            expand_cost_giou=sampling.get(
                "expand_cost_giou",
                matcher.get("cost_giou", 2.0),
            ),
            iou_pos_thr=quality.get("iou_pos_thr", 0.0),
            quality_min=quality.get("quality_min", 0.10),
            quality_max=quality.get("quality_max", 1.0),
            qfl_beta=quality.get("qfl_beta", 2.0),
            classification_type=classification_type,
            ia_bce_alpha=classification.get(
                "ia_bce_alpha",
                0.25,
            ),
            normalize_classification_by_num_gt=classification.get(
                "normalize_by_num_gt",
                True,
            ),
            score_negative_iou_ignore_thr=classification.get(
                "negative_iou_ignore_thr",
                quality.get(
                    "score_negative_iou_ignore_thr",
                    0.50,
                ),
            ),
            duplicate_suppression_enabled=duplicate.get(
                "enabled",
                True,
            ),
            duplicate_loss_weight=duplicate.get(
                "loss_weight",
                duplicate.get("lambda_duplicate", 0.10),
            ),
            duplicate_margin=duplicate.get("margin", 0.25),
            duplicate_background_weight=duplicate.get(
                "background_weight",
                0.05,
            ),
            duplicate_max_pairs=duplicate.get("max_pairs", 128),
            duplicate_start_epoch=duplicate.get("start_epoch", 5),
            hard_negative_mining_enabled=hard_negative.get(
                "enabled",
                sampling.get("hard_negative_mining_enabled", True),
            ),
            hard_negative_loss_weight=hard_negative.get(
                "loss_weight",
                sampling.get("hard_negative_loss_weight", 0.05),
            ),
            hard_negative_topk=hard_negative.get(
                "topk",
                sampling.get("hard_negative_topk", 10),
            ),
            hard_negative_max_iou=hard_negative.get(
                "max_iou",
                sampling.get("hard_negative_max_iou", 0.30),
            ),
            hard_negative_start_epoch=hard_negative.get(
                "start_epoch",
                sampling.get("hard_negative_start_epoch", 10),
            ),
            negative_text_as_empty_target=text_negative.get(
                "as_empty_target",
                True,
            ),
            text_negative_loss_weight=text_negative.get(
                "lambda_text_negative",
                text_negative.get("hard_topk_weight", 0.20),
            ),
            text_negative_topk=text_negative.get(
                "text_negative_topk",
                text_negative.get("hard_topk", 20),
            ),
            text_negative_hard_mix=text_negative.get(
                "text_negative_hard_mix",
                text_negative.get("hard_mix", 0.50),
            ),
            max_query_loss_weight=text_negative.get(
                "max_query_loss_weight",
                4.0,
            ),
            aux_loss_weight=hybrid.get(
                "aux_loss_weight",
                0.28,
            ),
            aux_score_enabled=quality.get(
                "aux_score_enabled",
                False,
            ),
            quality_warmup_epoch=quality.get(
                "quality_warmup_epoch",
                10,
            ),
            matcher_score_start_epoch=matcher_schedule.get(
                "start_epoch",
                ranking.get("rank_start_epoch", 5),
            ),
            matcher_score_warmup_epoch=matcher_schedule.get(
                "warmup_epoch",
                ranking.get("rank_warmup_epoch", 12),
            ),
            matcher_score_alpha_min=matcher_schedule.get(
                "alpha_min",
                ranking.get("rank_alpha_min", 0.0),
            ),
            # Legacy fields are accepted and recorded, not used actively.
            hard_negative_ratio=sampling.get(
                "hard_negative_ratio",
                5,
            ),
            rank_margin=ranking.get("rank_margin", 0.1),
            rank_min_quality_gap=ranking.get(
                "rank_min_quality_gap",
                0.1,
            ),
            rank_max_pairs=ranking.get("rank_max_pairs", 512),
            enable_pairwise_ranking=False,
        )

    def resolve_epoch_alpha(
        self,
        current_epoch=None,
        quality_alpha=None,
        rank_alpha=None,
        quality_warmup_epoch: Optional[int] = None,
        rank_start_epoch: Optional[int] = None,
        rank_warmup_epoch: Optional[int] = None,
        rank_alpha_min: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Resolve two schedules while retaining the previous public API.

        quality_alpha:
          Blends matched positive target from 1.0 toward IoU/GIoU-aware target.

        rank_alpha:
          Legacy name retained for compatibility. It now controls the score
          term in Hungarian matching, not a separate pairwise ranking loss.
        """
        if quality_warmup_epoch is None:
            quality_warmup_epoch = self.default_quality_warmup_epoch
        if rank_start_epoch is None:
            rank_start_epoch = self.default_matcher_score_start_epoch
        if rank_warmup_epoch is None:
            rank_warmup_epoch = self.default_matcher_score_warmup_epoch
        if rank_alpha_min is None:
            rank_alpha_min = self.default_matcher_score_alpha_min

        if quality_alpha is None:
            if current_epoch is None or int(quality_warmup_epoch) <= 0:
                quality_alpha = 1.0
            else:
                quality_alpha = clamp01(
                    float(current_epoch)
                    / float(max(int(quality_warmup_epoch), 1))
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
                batch_size,
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
    def _empty_box_tensor_like(target: dict) -> torch.Tensor:
        boxes = target.get("boxes")
        if torch.is_tensor(boxes):
            return boxes.reshape(-1, 4)[:0]
        return torch.empty((0, 4), dtype=torch.float32)

    def _build_effective_targets(
        self,
        targets: List[dict],
        text_negative_mask: torch.Tensor,
    ) -> List[dict]:
        """
        DETR no-object treatment for negative text.

        Negative text rows receive no positive assignment and no regression
        target. This removes the previous contradiction where a negative
        caption could still train the same positive box location.
        """
        if (
            not self.negative_text_as_empty_target
            or not bool(text_negative_mask.any())
        ):
            return targets

        effective: List[dict] = []
        mask_cpu = text_negative_mask.detach().cpu().tolist()
        for is_negative, target in zip(mask_cpu, targets):
            if not is_negative:
                effective.append(target)
                continue
            row = dict(target)
            row["boxes"] = self._empty_box_tensor_like(target)
            if "labels" in row:
                labels = row["labels"]
                if torch.is_tensor(labels):
                    row["labels"] = labels.reshape(-1)[:0]
                else:
                    row["labels"] = torch.empty(
                        0,
                        dtype=torch.long,
                    )
            effective.append(row)
        return effective

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
                f"{branch_name} bbox/score shape mismatch"
            )
        if len(targets) != int(pred_bbox.shape[0]):
            raise ValueError(
                f"{branch_name} targets batch size mismatch"
            )
        if pred_bbox.device != pred_score_logit.device:
            raise ValueError(
                f"{branch_name} bbox/score device mismatch"
            )

    def _classification_quality_target(
        self,
        *,
        logits: torch.Tensor,
        matched_iou: torch.Tensor,
        matched_giou: torch.Tensor,
    ) -> torch.Tensor:
        if matched_iou.numel() == 0:
            return matched_iou

        if self.classification_type == "normalized_giou":
            target = (matched_giou + 1.0) * 0.5
        elif self.classification_type == "iou":
            target = matched_iou
        else:
            probability = logits.detach().float().sigmoid()
            alpha = float(self.ia_bce_alpha)
            target = (
                probability.pow(alpha)
                * matched_iou.detach().float().clamp(0.0, 1.0).pow(
                    1.0 - alpha
                )
            )

        return target.clamp(
            min=float(self.quality_min),
            max=float(self.quality_max),
        ).detach()

    @torch.no_grad()
    def _build_high_iou_ignore_mask(
        self,
        *,
        pred_bbox: torch.Tensor,
        positive_mask: torch.Tensor,
        packed_targets: PackedTargets,
        threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build duplicate-like masks for unmatched predictions.

        Returns:
          ignore_mask:
            unmatched queries with max IoU >= threshold. They are excluded
            from background classification.
          max_iou:
            maximum IoU against any GT in the same sample, shape [B, N, 1].
          nearest_gt_index:
            local GT index that produced max_iou, shape [B, N, 1]. Empty-GT
            rows contain -1.
        """
        if pred_bbox.ndim != 3 or pred_bbox.shape[-1] != 4:
            raise ValueError(
                "pred_bbox must have shape [B, N, 4], got "
                f"{tuple(pred_bbox.shape)}"
            )
        if positive_mask.shape != pred_bbox.shape[:2] + (1,):
            raise ValueError(
                "positive_mask must have shape [B, N, 1], got "
                f"{tuple(positive_mask.shape)}"
            )

        batch_size, num_queries, _ = pred_bbox.shape
        max_iou = pred_bbox.new_zeros(
            (batch_size, num_queries, 1),
            dtype=torch.float32,
        )
        nearest_gt_index = torch.full(
            (batch_size, num_queries, 1),
            -1,
            dtype=torch.long,
            device=pred_bbox.device,
        )
        ignore_mask = torch.zeros_like(positive_mask, dtype=torch.bool)

        threshold = clamp01(threshold)
        if (
            threshold <= 0.0
            or num_queries == 0
            or packed_targets.boxes.numel() == 0
        ):
            return ignore_mask, max_iou, nearest_gt_index

        boxes = pred_bbox.detach().float()
        device = boxes.device

        single_rows = [
            index
            for index, count in enumerate(packed_targets.counts)
            if int(count) == 1
        ]
        if single_rows:
            row_index = torch.tensor(
                single_rows,
                device=device,
                dtype=torch.long,
            )
            gt_index = packed_targets.offsets[row_index]
            gt_boxes = packed_targets.boxes[gt_index].float()
            row_iou = batched_single_target_iou(
                boxes.index_select(0, row_index),
                gt_boxes,
            )
            max_iou[row_index, :, 0] = row_iou
            nearest_gt_index[row_index, :, 0] = 0

        multi_groups = group_rows_by_target_count(
            packed_targets.counts,
            minimum_count=2,
        )
        for target_count, rows in multi_groups.items():
            row_index = torch.tensor(
                rows,
                device=device,
                dtype=torch.long,
            )
            gt_boxes = gather_grouped_targets(
                packed_targets,
                row_index,
                target_count,
                dtype=torch.float32,
            )
            pairwise_iou = batched_pairwise_iou(
                boxes.index_select(0, row_index),
                gt_boxes,
            )
            row_iou, row_gt_index = pairwise_iou.max(dim=-1)
            max_iou[row_index, :, 0] = row_iou
            nearest_gt_index[row_index, :, 0] = row_gt_index

        ignore_mask = (
            (~positive_mask)
            & (max_iou >= float(threshold))
            & (nearest_gt_index >= 0)
        )
        return ignore_mask, max_iou, nearest_gt_index

    def _quality_aware_classification_loss(
        self,
        *,
        pred_score_logit: torch.Tensor,
        assignments: AssignmentResult,
        packed_targets: PackedTargets,
        pred_bbox: torch.Tensor,
        quality_alpha: float,
        query_loss_weights: Optional[torch.Tensor],
        negative_iou_ignore_thr: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fully vectorized DETR classification over all queries.

        Positive queries are exactly the Hungarian matches. Unmatched
        queries normally use target 0. Duplicate-like unmatched queries whose
        IoU with any GT exceeds ``negative_iou_ignore_thr`` are excluded from
        classification loss, but are not promoted to positives.
        """
        logits = pred_score_logit.float()
        probability = logits.sigmoid()
        target = torch.zeros_like(logits)
        positive_mask = torch.zeros_like(
            logits,
            dtype=torch.bool,
        )

        if assignments.num_matches > 0:
            batch_index = assignments.batch_indices
            pred_index = assignments.pred_indices
            global_gt_index = packed_targets.global_gt_indices(
                batch_index,
                assignments.gt_indices,
            )
            positive_pred_boxes = pred_bbox[
                batch_index,
                pred_index,
            ].float()
            positive_gt_boxes = packed_targets.boxes[
                global_gt_index
            ].float()
            matched_iou = matched_box_iou(
                positive_pred_boxes.detach(),
                positive_gt_boxes,
            ).clamp(0.0, 1.0)
            matched_giou = matched_generalized_box_iou(
                positive_pred_boxes.detach(),
                positive_gt_boxes,
            ).clamp(-1.0, 1.0)

            matched_logits = logits[
                batch_index,
                pred_index,
                0,
            ]
            final_target = self._classification_quality_target(
                logits=matched_logits,
                matched_iou=matched_iou,
                matched_giou=matched_giou,
            )
            warmup_target = torch.full_like(
                final_target,
                float(self.quality_max),
            )
            positive_target = (
                (1.0 - float(quality_alpha)) * warmup_target
                + float(quality_alpha) * final_target
            ).clamp(0.0, 1.0)

            target[
                batch_index,
                pred_index,
                0,
            ] = positive_target.to(target.dtype)
            positive_mask[
                batch_index,
                pred_index,
                0,
            ] = True
        else:
            matched_iou = logits.new_zeros((0,))
            matched_giou = logits.new_zeros((0,))
            positive_target = logits.new_zeros((0,))

        (
            ignore_mask,
            max_iou_to_gt,
            nearest_gt_index,
        ) = self._build_high_iou_ignore_mask(
            pred_bbox=pred_bbox,
            positive_mask=positive_mask,
            packed_targets=packed_targets,
            threshold=negative_iou_ignore_thr,
        )
        valid_classification_mask = ~ignore_mask

        gamma = float(self.focal_gamma)

        # IA-BCE style decomposition:
        # positive confidence follows the quality target;
        # no-object confidence is suppressed with focal weighting.
        positive_term = (
            -target * F.logsigmoid(logits)
            * (target - probability).abs().pow(gamma)
        )
        negative_term = (
            -(1.0 - target) * F.logsigmoid(-logits)
            * probability.pow(gamma)
        )
        element_loss = positive_term + negative_term

        query_weight = self._prepare_query_loss_weights(
            pred_score_logit,
            query_loss_weights,
        ).float()
        weighted_loss = (
            element_loss
            * query_weight
            * valid_classification_mask.float()
        )

        num_matches = max(assignments.num_matches, 1)
        if self.normalize_classification_by_num_gt:
            loss_score = weighted_loss.sum() / float(num_matches)
        else:
            valid_count = valid_classification_mask.sum().clamp_min(1)
            loss_score = weighted_loss.sum() / valid_count.float()

        if bool(positive_mask.any()):
            loss_score_pos = element_loss[
                positive_mask
            ].mean()
        else:
            loss_score_pos = logits.new_zeros(())

        negative_mask = (~positive_mask) & (~ignore_mask)
        if bool(negative_mask.any()):
            loss_score_neg = element_loss[
                negative_mask
            ].mean()
        else:
            loss_score_neg = logits.new_zeros(())

        ignored_score_mean = logits.new_zeros(())
        ignored_iou_mean = logits.new_zeros(())
        if bool(ignore_mask.any()):
            ignored_score_mean = probability[ignore_mask].mean()
            ignored_iou_mean = max_iou_to_gt[ignore_mask].mean()

        return loss_score.to(pred_score_logit.dtype), {
            "target": target.to(pred_score_logit.dtype),
            "positive_mask": positive_mask,
            "ignore_mask": ignore_mask,
            "valid_negative_mask": negative_mask,
            "max_iou_to_gt": max_iou_to_gt.to(
                pred_score_logit.dtype
            ),
            "nearest_gt_index": nearest_gt_index,
            "ignored_score_mean": ignored_score_mean.to(
                pred_score_logit.dtype
            ),
            "ignored_iou_mean": ignored_iou_mean.to(
                pred_score_logit.dtype
            ),
            "loss_pos": loss_score_pos.to(pred_score_logit.dtype),
            "loss_neg": loss_score_neg.to(pred_score_logit.dtype),
            "matched_iou": matched_iou.to(pred_score_logit.dtype),
            "matched_giou": matched_giou.to(pred_score_logit.dtype),
            "positive_target": positive_target.to(
                pred_score_logit.dtype
            ),
        }

    def duplicate_ranking_loss(
        self,
        *,
        pred_score_logit: torch.Tensor,
        assignments: AssignmentResult,
        packed_targets: PackedTargets,
        duplicate_mask: torch.Tensor,
        nearest_gt_index: torch.Tensor,
        query_loss_weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Rank duplicate-like unmatched queries below the matched TP of the
        same GT, while applying only a weak no-object penalty to duplicates.

        The full background BCE is intentionally not used for these queries.
        """
        zero = pred_score_logit.new_zeros(())
        metrics = {
            "pair_count": zero,
            "violation_fraction": zero,
            "tp_score_mean": zero,
            "duplicate_score_mean": zero,
            "score_gap_mean": zero,
            "rank_loss": zero,
            "background_loss": zero,
        }

        if (
            assignments.num_matches == 0
            or not bool(duplicate_mask.any())
            or packed_targets.boxes.numel() == 0
        ):
            return zero, metrics

        logits = pred_score_logit.float().squeeze(-1)
        duplicate_mask_2d = duplicate_mask.squeeze(-1).bool()
        nearest_gt_2d = nearest_gt_index.squeeze(-1).long()

        duplicate_positions = torch.nonzero(
            duplicate_mask_2d,
            as_tuple=False,
        )
        if duplicate_positions.numel() == 0:
            return zero, metrics

        dup_batch = duplicate_positions[:, 0]
        dup_query = duplicate_positions[:, 1]
        dup_local_gt = nearest_gt_2d[dup_batch, dup_query]

        valid = dup_local_gt >= 0
        if not bool(valid.any()):
            return zero, metrics

        dup_batch = dup_batch[valid]
        dup_query = dup_query[valid]
        dup_local_gt = dup_local_gt[valid]

        total_gt = int(packed_targets.boxes.shape[0])
        matched_pred_for_global_gt = torch.full(
            (total_gt,),
            -1,
            dtype=torch.long,
            device=logits.device,
        )
        matched_global_gt = packed_targets.global_gt_indices(
            assignments.batch_indices,
            assignments.gt_indices,
        )
        matched_pred_for_global_gt[matched_global_gt] = (
            assignments.pred_indices
        )

        dup_global_gt = packed_targets.global_gt_indices(
            dup_batch,
            dup_local_gt,
        )
        tp_query = matched_pred_for_global_gt[dup_global_gt]
        valid = (tp_query >= 0) & (tp_query != dup_query)
        if not bool(valid.any()):
            return zero, metrics

        dup_batch = dup_batch[valid]
        dup_query = dup_query[valid]
        tp_query = tp_query[valid]

        dup_logit = logits[dup_batch, dup_query]
        tp_logit = logits[dup_batch, tp_query]

        rank_per_pair = F.relu(
            float(self.duplicate_margin)
            + dup_logit
            - tp_logit
        )
        dup_prob = dup_logit.sigmoid()
        tp_prob = tp_logit.sigmoid()
        background_per_pair = (
            F.softplus(dup_logit)
            * dup_prob.pow(float(self.focal_gamma))
        )

        combined_per_pair = (
            rank_per_pair
            + float(self.duplicate_background_weight)
            * background_per_pair
        )

        if combined_per_pair.numel() > self.duplicate_max_pairs:
            _, keep = torch.topk(
                combined_per_pair.detach(),
                k=self.duplicate_max_pairs,
                largest=True,
                sorted=False,
            )
            dup_batch = dup_batch[keep]
            dup_logit = dup_logit[keep]
            tp_logit = tp_logit[keep]
            dup_prob = dup_prob[keep]
            tp_prob = tp_prob[keep]
            rank_per_pair = rank_per_pair[keep]
            background_per_pair = background_per_pair[keep]
            combined_per_pair = combined_per_pair[keep]

        sample_weight = self._prepare_query_loss_weights(
            pred_score_logit,
            query_loss_weights,
        ).reshape(-1).float()
        pair_weight = sample_weight[dup_batch]
        loss = (combined_per_pair * pair_weight).sum() / (
            pair_weight.sum().clamp_min(1.0)
        )

        return loss.to(pred_score_logit.dtype), {
            "pair_count": pred_score_logit.new_tensor(
                float(combined_per_pair.numel())
            ),
            "violation_fraction": (
                rank_per_pair > 0
            ).float().mean().to(pred_score_logit.dtype),
            "tp_score_mean": tp_prob.mean().to(pred_score_logit.dtype),
            "duplicate_score_mean": dup_prob.mean().to(
                pred_score_logit.dtype
            ),
            "score_gap_mean": (tp_prob - dup_prob).mean().to(
                pred_score_logit.dtype
            ),
            "rank_loss": rank_per_pair.mean().to(
                pred_score_logit.dtype
            ),
            "background_loss": background_per_pair.mean().to(
                pred_score_logit.dtype
            ),
        }

    def unmatched_hard_negative_loss(
        self,
        *,
        pred_score_logit: torch.Tensor,
        valid_negative_mask: torch.Tensor,
        max_iou_to_gt: torch.Tensor,
        packed_targets: PackedTargets,
        query_loss_weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Extra Top-K loss for high-score, low-IoU unmatched queries."""
        zero = pred_score_logit.new_zeros(())
        metrics = {
            "count": zero,
            "score_mean": zero,
            "iou_mean": zero,
            "loss_mean": zero,
        }

        logits = pred_score_logit.float().squeeze(-1)
        negative_mask = valid_negative_mask.squeeze(-1).bool()
        max_iou = max_iou_to_gt.squeeze(-1).float()
        batch_size, num_queries = logits.shape

        if num_queries == 0:
            return zero, metrics

        gt_count = torch.tensor(
            packed_targets.counts,
            device=logits.device,
            dtype=torch.long,
        )
        positive_text_rows = gt_count > 0
        candidate_mask = (
            negative_mask
            & positive_text_rows[:, None]
            & (max_iou <= float(self.hard_negative_max_iou))
        )
        if not bool(candidate_mask.any()):
            return zero, metrics

        probability = logits.sigmoid()
        element_loss = (
            F.softplus(logits)
            * probability.pow(float(self.focal_gamma))
        )
        masked_loss = element_loss.masked_fill(
            ~candidate_mask,
            -torch.inf,
        )

        max_k = min(self.hard_negative_topk, num_queries)
        topk_loss, topk_index = torch.topk(
            masked_loss,
            k=max_k,
            dim=1,
            largest=True,
            sorted=False,
        )
        candidate_count = candidate_mask.sum(dim=1)
        desired_count = (
            gt_count.clamp_min(1) * int(self.hard_negative_ratio)
        )
        k_per_row = torch.minimum(desired_count, candidate_count)
        k_per_row = k_per_row.clamp(max=max_k)

        rank = torch.arange(
            max_k,
            device=logits.device,
        )[None, :]
        selected_mask = (
            rank < k_per_row[:, None]
        ) & torch.isfinite(topk_loss)
        if not bool(selected_mask.any()):
            return zero, metrics

        selected_loss = torch.where(
            selected_mask,
            topk_loss,
            torch.zeros_like(topk_loss),
        )
        row_count = selected_mask.sum(dim=1)
        row_loss = selected_loss.sum(dim=1) / row_count.clamp_min(1)
        valid_rows = row_count > 0

        sample_weight = self._prepare_query_loss_weights(
            pred_score_logit,
            query_loss_weights,
        ).reshape(-1).float()
        loss = (
            row_loss[valid_rows] * sample_weight[valid_rows]
        ).sum() / sample_weight[valid_rows].sum().clamp_min(1.0)

        selected_score = probability.gather(1, topk_index)[selected_mask]
        selected_iou = max_iou.gather(1, topk_index)[selected_mask]

        return loss.to(pred_score_logit.dtype), {
            "count": pred_score_logit.new_tensor(
                float(selected_mask.sum().item())
            ),
            "score_mean": selected_score.mean().to(
                pred_score_logit.dtype
            ),
            "iou_mean": selected_iou.mean().to(pred_score_logit.dtype),
            "loss_mean": selected_loss[selected_mask].mean().to(
                pred_score_logit.dtype
            ),
        }

    def text_negative_suppression_loss(
        self,
        pred_score_logit: torch.Tensor,
        text_negative_mask: torch.Tensor,
        query_loss_weights: Optional[torch.Tensor],
        positive_query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Optional hard Top-K no-object reinforcement for negative captions.

        The main quality-aware classification already supervises every query
        in negative-text rows as no-object. This term only strengthens their
        highest scores and remains fully batched.
        """
        zero = pred_score_logit.new_zeros(())
        mask = self._prepare_text_negative_mask(
            pred_score_logit,
            text_negative_mask,
        )
        if not bool(mask.any()):
            return zero, {
                "all_mean": zero,
                "hard_mean": zero,
                "negative_top1_score": zero,
                "positive_top1_score": zero,
                "positive_negative_margin": zero,
            }

        logits = pred_score_logit.float().squeeze(-1)
        negative_logits = logits[mask]
        negative_prob = negative_logits.sigmoid()
        element_loss = (
            F.softplus(negative_logits)
            * negative_prob.pow(float(self.focal_gamma))
        )

        all_per_query = element_loss.mean(dim=1)
        k = min(
            self.text_negative_topk,
            int(element_loss.shape[1]),
        )
        hard_per_query = torch.topk(
            element_loss,
            k=k,
            dim=1,
            largest=True,
            sorted=False,
        ).values.mean(dim=1)

        mix = float(self.text_negative_hard_mix)
        per_query = (
            (1.0 - mix) * all_per_query
            + mix * hard_per_query
        )
        query_weight = self._prepare_query_loss_weights(
            pred_score_logit,
            query_loss_weights,
        ).reshape(-1)[mask].float()
        loss = (per_query * query_weight).mean()

        negative_top1 = negative_prob.max(dim=1).values.mean()
        positive_top1 = zero.float()
        if positive_query_mask is not None:
            positive_query_mask = positive_query_mask.to(
                device=logits.device,
                dtype=torch.bool,
            ).reshape(-1)
            if bool(positive_query_mask.any()):
                positive_top1 = (
                    logits[positive_query_mask]
                    .sigmoid()
                    .max(dim=1)
                    .values
                    .mean()
                )

        return loss.to(pred_score_logit.dtype), {
            "all_mean": all_per_query.mean().to(
                pred_score_logit.dtype
            ),
            "hard_mean": hard_per_query.mean().to(
                pred_score_logit.dtype
            ),
            "negative_top1_score": negative_top1.to(
                pred_score_logit.dtype
            ),
            "positive_top1_score": positive_top1.to(
                pred_score_logit.dtype
            ),
            "positive_negative_margin": (
                positive_top1 - negative_top1
            ).to(pred_score_logit.dtype),
        }

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
        quality_alpha: float,
        query_loss_weights: Optional[torch.Tensor],
        text_negative_mask: torch.Tensor,
        score_enabled: bool,
        allow_text_negative_hard_loss: bool,
        allow_duplicate_suppression: bool,
        allow_unmatched_hard_negative_loss: bool,
        negative_iou_ignore_thr: float,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self._validate_branch_inputs(
            branch_name,
            pred_bbox,
            pred_score_logit,
            targets,
        )

        total_pairs = assignments.num_matches
        if total_pairs > 0:
            batch_index = assignments.batch_indices
            pred_index = assignments.pred_indices
            global_gt_index = packed_targets.global_gt_indices(
                batch_index,
                assignments.gt_indices,
            )
            positive_pred_boxes = pred_bbox[
                batch_index,
                pred_index,
            ]
            positive_gt_boxes = packed_targets.boxes[
                global_gt_index
            ].to(dtype=pred_bbox.dtype)

            loss_bbox = F.l1_loss(
                positive_pred_boxes,
                positive_gt_boxes,
                reduction="sum",
            ) / float(total_pairs)
            matched_giou = matched_generalized_box_iou(
                positive_pred_boxes.float(),
                positive_gt_boxes.float(),
            ).to(dtype=pred_bbox.dtype)
            loss_giou = (
                1.0 - matched_giou
            ).sum() / float(total_pairs)
            matched_iou_mean = matched_box_iou(
                positive_pred_boxes.detach().float(),
                positive_gt_boxes.float(),
            ).mean().to(pred_bbox.dtype)
        else:
            zero = pred_bbox.new_zeros(())
            loss_bbox = zero
            loss_giou = zero
            matched_iou_mean = zero

        zero = pred_bbox.new_zeros(())
        if score_enabled:
            loss_score, score_info = (
                self._quality_aware_classification_loss(
                    pred_score_logit=pred_score_logit,
                    assignments=assignments,
                    packed_targets=packed_targets,
                    pred_bbox=pred_bbox,
                    quality_alpha=quality_alpha,
                    query_loss_weights=query_loss_weights,
                    negative_iou_ignore_thr=negative_iou_ignore_thr,
                )
            )
            loss_score_pos = score_info["loss_pos"]
            loss_score_neg = score_info["loss_neg"]
            positive_mask = score_info["positive_mask"]
            ignore_mask = score_info["ignore_mask"]
            valid_negative_mask = score_info["valid_negative_mask"]
            max_iou_to_gt = score_info["max_iou_to_gt"]
            nearest_gt_index = score_info["nearest_gt_index"]
            ignored_score_mean = score_info["ignored_score_mean"]
            ignored_iou_mean = score_info["ignored_iou_mean"]
            positive_target = score_info["positive_target"]

            positive_query_mask = torch.tensor(
                [count > 0 for count in packed_targets.counts],
                device=pred_score_logit.device,
                dtype=torch.bool,
            )
            if allow_text_negative_hard_loss:
                (
                    loss_text_negative,
                    text_metrics,
                ) = self.text_negative_suppression_loss(
                    pred_score_logit,
                    text_negative_mask,
                    query_loss_weights,
                    positive_query_mask=positive_query_mask,
                )
            else:
                loss_text_negative = zero
                text_metrics = {
                    "all_mean": zero,
                    "hard_mean": zero,
                    "negative_top1_score": zero,
                    "positive_top1_score": zero,
                    "positive_negative_margin": zero,
                }

            if allow_duplicate_suppression:
                loss_duplicate, duplicate_metrics = (
                    self.duplicate_ranking_loss(
                        pred_score_logit=pred_score_logit,
                        assignments=assignments,
                        packed_targets=packed_targets,
                        duplicate_mask=ignore_mask,
                        nearest_gt_index=nearest_gt_index,
                        query_loss_weights=query_loss_weights,
                    )
                )
            else:
                loss_duplicate = zero
                duplicate_metrics = {
                    "pair_count": zero,
                    "violation_fraction": zero,
                    "tp_score_mean": zero,
                    "duplicate_score_mean": zero,
                    "score_gap_mean": zero,
                    "rank_loss": zero,
                    "background_loss": zero,
                }

            if allow_unmatched_hard_negative_loss:
                loss_hard_negative, hard_negative_metrics = (
                    self.unmatched_hard_negative_loss(
                        pred_score_logit=pred_score_logit,
                        valid_negative_mask=valid_negative_mask,
                        max_iou_to_gt=max_iou_to_gt,
                        packed_targets=packed_targets,
                        query_loss_weights=query_loss_weights,
                    )
                )
            else:
                loss_hard_negative = zero
                hard_negative_metrics = {
                    "count": zero,
                    "score_mean": zero,
                    "iou_mean": zero,
                    "loss_mean": zero,
                }

            if positive_target.numel() > 0:
                target_mean = positive_target.mean()
                target_min = positive_target.min()
                target_max = positive_target.max()
            else:
                target_mean = zero
                target_min = zero
                target_max = zero
        else:
            loss_score = zero
            loss_score_pos = zero
            loss_score_neg = zero
            loss_text_negative = zero
            loss_duplicate = zero
            loss_hard_negative = zero
            positive_mask = torch.zeros_like(
                pred_score_logit,
                dtype=torch.bool,
            )
            ignore_mask = torch.zeros_like(
                pred_score_logit,
                dtype=torch.bool,
            )
            valid_negative_mask = ~positive_mask
            ignored_score_mean = zero
            ignored_iou_mean = zero
            target_mean = zero
            target_min = zero
            target_max = zero
            text_metrics = {
                "all_mean": zero,
                "hard_mean": zero,
                "negative_top1_score": zero,
                "positive_top1_score": zero,
                "positive_negative_margin": zero,
            }
            duplicate_metrics = {
                "pair_count": zero,
                "violation_fraction": zero,
                "tp_score_mean": zero,
                "duplicate_score_mean": zero,
                "score_gap_mean": zero,
                "rank_loss": zero,
                "background_loss": zero,
            }
            hard_negative_metrics = {
                "count": zero,
                "score_mean": zero,
                "iou_mean": zero,
                "loss_mean": zero,
            }

        bbox_contrib = float(lambda_bbox) * loss_bbox
        giou_contrib = float(lambda_giou) * loss_giou
        score_contrib = float(lambda_score) * loss_score
        loss_text_negative_contrib = (
            float(self.text_negative_loss_weight)
            * loss_text_negative
        )
        loss_duplicate_contrib = (
            float(self.duplicate_loss_weight)
            * loss_duplicate
        )
        loss_hard_negative_contrib = (
            float(self.hard_negative_loss_weight)
            * loss_hard_negative
        )

        loss_base = bbox_contrib + giou_contrib + score_contrib
        loss_total = (
            loss_base
            + loss_text_negative_contrib
            + loss_duplicate_contrib
            + loss_hard_negative_contrib
        )

        total_unmatched_count = float(
            positive_mask.numel() - positive_mask.sum().item()
        )
        ignored_negative_count = float(ignore_mask.sum().item())
        negative_count = float(valid_negative_mask.sum().item())
        selected_negative_fraction = (
            negative_count / total_unmatched_count
            if total_unmatched_count > 0.0
            else 1.0
        )
        text_negative_count = float(text_negative_mask.sum().item())

        metrics: Dict[str, Any] = {
            "loss_total": loss_total,
            "loss_base": loss_base,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_score": loss_score,
            "loss_score_pos": loss_score_pos,
            "loss_score_neg": loss_score_neg,
            "loss_score_neg_unweighted": loss_score_neg,
            "loss_bbox_contrib": bbox_contrib,
            "loss_giou_contrib": giou_contrib,
            "loss_score_contrib": score_contrib,
            "loss_text_negative": loss_text_negative,
            "loss_text_negative_contrib": loss_text_negative_contrib,
            "loss_duplicate": loss_duplicate,
            "loss_duplicate_contrib": loss_duplicate_contrib,
            "loss_duplicate_rank": duplicate_metrics["rank_loss"],
            "loss_duplicate_background": duplicate_metrics[
                "background_loss"
            ],
            "duplicate_pair_count": duplicate_metrics["pair_count"],
            "duplicate_violation_fraction": duplicate_metrics[
                "violation_fraction"
            ],
            "duplicate_tp_score_mean": duplicate_metrics[
                "tp_score_mean"
            ],
            "duplicate_score_mean": duplicate_metrics[
                "duplicate_score_mean"
            ],
            "duplicate_score_gap_mean": duplicate_metrics[
                "score_gap_mean"
            ],
            "loss_hard_negative": loss_hard_negative,
            "loss_hard_negative_contrib": loss_hard_negative_contrib,
            "hard_negative_score_mean": hard_negative_metrics[
                "score_mean"
            ],
            "hard_negative_iou_mean": hard_negative_metrics["iou_mean"],
            "hard_negative_loss_mean": hard_negative_metrics["loss_mean"],
            "loss_text_negative_hard_qfl": zero,
            "loss_text_negative_all_mean": text_metrics["all_mean"],
            "loss_text_negative_hard_mean": text_metrics["hard_mean"],
            "negative_query_top1_score": text_metrics[
                "negative_top1_score"
            ],
            "positive_query_top1_score": text_metrics[
                "positive_top1_score"
            ],
            "positive_negative_score_margin": text_metrics[
                "positive_negative_margin"
            ],
            # Kept only for backward-compatible logging.
            "loss_rank": zero,
            "loss_rank_raw": zero,
            "loss_rank_contrib": zero,
            "matched": float(total_pairs),
            "score_pos_count": float(positive_mask.sum().item()),
            "hard_neg_count": hard_negative_metrics["count"],
            "negative_count": negative_count,
            "ignored_negative_count": ignored_negative_count,
            "selected_negative_fraction": selected_negative_fraction,
            "ignored_negative_score_mean": ignored_score_mean,
            "ignored_negative_iou_mean": ignored_iou_mean,
            "text_negative_count": text_negative_count,
            "text_negative_weight_mean": (
                self._prepare_query_loss_weights(
                    pred_score_logit,
                    query_loss_weights,
                )
                .reshape(-1)[text_negative_mask]
                .mean()
                if bool(text_negative_mask.any())
                else pred_bbox.new_ones(())
            ),
            "matched_iou_mean": matched_iou_mean,
            "score_iou_mean": matched_iou_mean,
            "score_target_pos_mean": target_mean,
            "score_target_pos_min": target_min,
            "score_target_pos_max": target_max,
            "lambda_bbox": float(lambda_bbox),
            "lambda_giou": float(lambda_giou),
            "lambda_score": float(lambda_score),
            "lambda_text_negative": float(
                self.text_negative_loss_weight
            ),
            "lambda_duplicate": float(self.duplicate_loss_weight),
            "lambda_hard_negative": float(
                self.hard_negative_loss_weight
            ),
            "score_negative_iou_ignore_thr": float(
                negative_iou_ignore_thr
            ),
            "hard_negative_max_iou": float(self.hard_negative_max_iou),
            "duplicate_margin": float(self.duplicate_margin),
            "rank_negative_iou_max": 0.0,
            "lambda_rank": 0.0,
            "lambda_rank_eff": 0.0,
            "pos_weight": 1.0,
            "quality_alpha": float(quality_alpha),
            "rank_alpha": 0.0,
            "assignment_mode": assignments.mode,
            "score_assignment_mode": "hungarian_one_to_one",
            "score_match_rounds": 1,
            "score_enabled": bool(score_enabled),
            "classification_type": self.classification_type,
        }
        return loss_total, metrics

    @staticmethod
    def _detach_metric(value: Any) -> Any:
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
        lambda_score: float = 2.0,
        pos_weight: float = 1.0,
        current_epoch=None,
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
        """
        The signature remains compatible with the previous training loop.

        ``rank_alpha`` now schedules the score term in Hungarian matching.
        ``lambda_rank`` is intentionally ignored because the explicit dense
        pairwise ranking objective has been removed.
        """
        self._validate_branch_inputs(
            "main",
            pred_bbox,
            pred_score_logit,
            targets,
        )

        quality_alpha, matcher_score_alpha = (
            self.resolve_epoch_alpha(
                current_epoch=current_epoch,
                quality_alpha=quality_alpha,
                rank_alpha=rank_alpha,
                quality_warmup_epoch=quality_warmup_epoch,
                rank_start_epoch=rank_start_epoch,
                rank_warmup_epoch=rank_warmup_epoch,
                rank_alpha_min=rank_alpha_min,
            )
        )

        prepared_text_negative_mask = (
            self._prepare_text_negative_mask(
                pred_score_logit,
                text_negative_mask,
            )
        )
        effective_targets = self._build_effective_targets(
            targets,
            prepared_text_negative_mask,
        )
        packed_targets = PackedTargets.from_targets(
            effective_targets,
            device=pred_bbox.device,
            dtype=pred_bbox.dtype,
        )

        epoch_value = 1 if current_epoch is None else int(current_epoch)
        duplicate_suppression_active = (
            self.duplicate_suppression_enabled
            and epoch_value >= self.duplicate_start_epoch
        )
        hard_negative_mining_active = (
            self.hard_negative_mining_enabled
            and epoch_value >= self.hard_negative_start_epoch
        )

        main_assignments = self.main_matcher(
            pred_bbox=pred_bbox,
            pred_score_logit=pred_score_logit,
            targets=effective_targets,
            packed_targets=packed_targets,
            score_cost_alpha=matcher_score_alpha,
        )
        main_loss, main_metrics = self._compute_branch_loss(
            branch_name="main",
            pred_bbox=pred_bbox,
            pred_score_logit=pred_score_logit,
            targets=effective_targets,
            packed_targets=packed_targets,
            assignments=main_assignments,
            lambda_bbox=lambda_bbox,
            lambda_giou=lambda_giou,
            lambda_score=lambda_score,
            quality_alpha=quality_alpha,
            query_loss_weights=query_loss_weights,
            text_negative_mask=prepared_text_negative_mask,
            score_enabled=True,
            allow_text_negative_hard_loss=True,
            allow_duplicate_suppression=(
                duplicate_suppression_active
            ),
            allow_unmatched_hard_negative_loss=(
                hard_negative_mining_active
            ),
            negative_iou_ignore_thr=(
                self.score_negative_iou_ignore_thr
            ),
        )

        has_aux_bbox = aux_pred_bbox is not None
        has_aux_score = aux_pred_score_logit is not None
        if has_aux_bbox != has_aux_score:
            raise ValueError(
                "aux_pred_bbox and aux_pred_score_logit must either "
                "both be provided or both be None."
            )

        aux_enabled = has_aux_bbox and has_aux_score
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
            if not self.aux_score_enabled:
                aux_lambda_score_eff = 0.0

            aux_assignments = self.aux_matcher(
                pred_bbox=aux_pred_bbox,
                pred_score_logit=aux_pred_score_logit,
                targets=effective_targets,
                packed_targets=packed_targets,
            )
            aux_loss, aux_metrics = self._compute_branch_loss(
                branch_name="aux",
                pred_bbox=aux_pred_bbox,
                pred_score_logit=aux_pred_score_logit,
                targets=effective_targets,
                packed_targets=packed_targets,
                assignments=aux_assignments,
                lambda_bbox=aux_lambda_bbox_eff,
                lambda_giou=aux_lambda_giou_eff,
                lambda_score=aux_lambda_score_eff,
                quality_alpha=quality_alpha,
                query_loss_weights=query_loss_weights,
                text_negative_mask=prepared_text_negative_mask,
                score_enabled=self.aux_score_enabled,
                allow_text_negative_hard_loss=False,
                allow_duplicate_suppression=False,
                allow_unmatched_hard_negative_loss=False,
                negative_iou_ignore_thr=0.0,
            )
        else:
            aux_loss = pred_bbox.new_zeros(())
            zero = pred_bbox.new_zeros(())
            aux_metrics = {
                "loss_total": zero,
                "loss_base": zero,
                "loss_bbox": zero,
                "loss_giou": zero,
                "loss_score": zero,
                "loss_score_pos": zero,
                "loss_score_neg": zero,
                "loss_score_neg_unweighted": zero,
                "loss_bbox_contrib": zero,
                "loss_giou_contrib": zero,
                "loss_score_contrib": zero,
                "loss_text_negative": zero,
                "loss_text_negative_contrib": zero,
                "loss_text_negative_hard_qfl": zero,
                "loss_text_negative_all_mean": zero,
                "loss_text_negative_hard_mean": zero,
                "loss_duplicate": zero,
                "loss_duplicate_contrib": zero,
                "loss_duplicate_rank": zero,
                "loss_duplicate_background": zero,
                "duplicate_pair_count": zero,
                "duplicate_violation_fraction": zero,
                "duplicate_tp_score_mean": zero,
                "duplicate_score_mean": zero,
                "duplicate_score_gap_mean": zero,
                "loss_hard_negative": zero,
                "loss_hard_negative_contrib": zero,
                "hard_negative_score_mean": zero,
                "hard_negative_iou_mean": zero,
                "hard_negative_loss_mean": zero,
                "negative_query_top1_score": zero,
                "positive_query_top1_score": zero,
                "positive_negative_score_margin": zero,
                "loss_rank": zero,
                "loss_rank_raw": zero,
                "loss_rank_contrib": zero,
                "matched": 0.0,
                "score_pos_count": 0.0,
                "hard_neg_count": 0.0,
                "negative_count": 0.0,
                "ignored_negative_count": 0.0,
                "selected_negative_fraction": 1.0,
                "ignored_negative_score_mean": zero,
                "ignored_negative_iou_mean": zero,
                "text_negative_count": 0.0,
                "text_negative_weight_mean": zero,
                "matched_iou_mean": zero,
                "score_iou_mean": zero,
                "score_target_pos_mean": zero,
                "score_target_pos_min": zero,
                "score_target_pos_max": zero,
                "lambda_bbox": float(lambda_bbox),
                "lambda_giou": float(lambda_giou),
                "lambda_score": 0.0,
                "lambda_text_negative": 0.0,
                "lambda_duplicate": 0.0,
                "lambda_hard_negative": 0.0,
                "score_negative_iou_ignore_thr": 0.0,
                "hard_negative_max_iou": 0.0,
                "duplicate_margin": 0.0,
                "rank_negative_iou_max": 0.0,
                "lambda_rank": 0.0,
                "lambda_rank_eff": 0.0,
                "pos_weight": 1.0,
                "quality_alpha": float(quality_alpha),
                "rank_alpha": 0.0,
                "assignment_mode": "disabled",
                "score_assignment_mode": "disabled",
                "score_match_rounds": 0,
                "score_enabled": False,
                "classification_type": self.classification_type,
            }

        aux_loss_contrib = float(lambda_aux_eff) * aux_loss
        loss = main_loss + aux_loss_contrib

        # Negative-text rows must never produce regression assignments.
        negative_regression_matches = 0
        if main_assignments.num_matches > 0:
            negative_regression_matches = int(
                prepared_text_negative_mask[
                    main_assignments.batch_indices
                ].sum().item()
            )
        if negative_regression_matches != 0:
            raise RuntimeError(
                "Negative-text rows received main regression matches. "
                "This violates DETR no-object supervision."
            )

        loss_dict: Dict[str, Any] = {
            "loss": loss.detach(),
            "loss_main_total": main_loss.detach(),
            "loss_aux_total": aux_loss.detach(),
            "loss_aux_contrib": aux_loss_contrib.detach(),
            "lambda_aux": float(lambda_aux_eff),
            "aux_enabled": bool(aux_enabled),
            "aux_score_enabled": bool(self.aux_score_enabled),
            "hdetr_repeat_k": int(self.aux_matcher.repeat_k),
            "score_match_rounds": 1,
            "score_quality_gamma": 1.0,
            "score_round_decay": 1.0,
            "pairwise_ranking_enabled": False,
            "dense_score_assignment_enabled": False,
            "duplicate_suppression_enabled": bool(
                duplicate_suppression_active
            ),
            "hard_negative_mining_enabled": bool(
                hard_negative_mining_active
            ),
            "matcher_score_alpha": float(
                matcher_score_alpha
            ),
            "matcher_cost_score_effective": float(
                self.main_matcher.cost_score
                * matcher_score_alpha
            ),
            "legacy_lambda_rank_ignored": float(lambda_rank),
            "negative_text_as_empty_target": bool(
                self.negative_text_as_empty_target
            ),
            "negative_text_regression_matches": float(
                negative_regression_matches
            ),
            "classification_type": self.classification_type,
            "score_negative_iou_ignore_thr": float(
                self.score_negative_iou_ignore_thr
            ),

            # Backward-compatible main aliases.
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
            "loss_score_neg_unweighted": self._detach_metric(
                main_metrics["loss_score_neg_unweighted"]
            ),
            "loss_bbox_contrib": self._detach_metric(
                main_metrics["loss_bbox_contrib"]
            ),
            "loss_giou_contrib": self._detach_metric(
                main_metrics["loss_giou_contrib"]
            ),
            "loss_score_contrib": self._detach_metric(
                main_metrics["loss_score_contrib"]
            ),
            "loss_text_negative": self._detach_metric(
                main_metrics["loss_text_negative"]
            ),
            "loss_text_negative_contrib": self._detach_metric(
                main_metrics["loss_text_negative_contrib"]
            ),
            "loss_duplicate": self._detach_metric(
                main_metrics["loss_duplicate"]
            ),
            "loss_duplicate_contrib": self._detach_metric(
                main_metrics["loss_duplicate_contrib"]
            ),
            "loss_duplicate_rank": self._detach_metric(
                main_metrics["loss_duplicate_rank"]
            ),
            "loss_duplicate_background": self._detach_metric(
                main_metrics["loss_duplicate_background"]
            ),
            "duplicate_pair_count": self._detach_metric(
                main_metrics["duplicate_pair_count"]
            ),
            "duplicate_violation_fraction": self._detach_metric(
                main_metrics["duplicate_violation_fraction"]
            ),
            "duplicate_tp_score_mean": self._detach_metric(
                main_metrics["duplicate_tp_score_mean"]
            ),
            "duplicate_score_mean": self._detach_metric(
                main_metrics["duplicate_score_mean"]
            ),
            "duplicate_score_gap_mean": self._detach_metric(
                main_metrics["duplicate_score_gap_mean"]
            ),
            "loss_hard_negative": self._detach_metric(
                main_metrics["loss_hard_negative"]
            ),
            "loss_hard_negative_contrib": self._detach_metric(
                main_metrics["loss_hard_negative_contrib"]
            ),
            "hard_negative_score_mean": self._detach_metric(
                main_metrics["hard_negative_score_mean"]
            ),
            "hard_negative_iou_mean": self._detach_metric(
                main_metrics["hard_negative_iou_mean"]
            ),
            "loss_text_negative_hard_qfl": self._detach_metric(
                main_metrics["loss_text_negative_hard_qfl"]
            ),
            "negative_query_top1_score": self._detach_metric(
                main_metrics["negative_query_top1_score"]
            ),
            "positive_query_top1_score": self._detach_metric(
                main_metrics["positive_query_top1_score"]
            ),
            "positive_negative_score_margin": self._detach_metric(
                main_metrics["positive_negative_score_margin"]
            ),
            "loss_rank": pred_bbox.new_zeros(()),
            "loss_rank_raw": pred_bbox.new_zeros(()),
            "loss_rank_contrib": pred_bbox.new_zeros(()),
            "matched": main_metrics["matched"],
            "score_pos_count": main_metrics["score_pos_count"],
            "hard_neg_count": self._detach_metric(
                main_metrics["hard_neg_count"]
            ),
            "negative_count": main_metrics["negative_count"],
            "ignored_negative_count": main_metrics[
                "ignored_negative_count"
            ],
            "selected_negative_fraction": main_metrics[
                "selected_negative_fraction"
            ],
            "ignored_negative_score_mean": self._detach_metric(
                main_metrics["ignored_negative_score_mean"]
            ),
            "ignored_negative_iou_mean": self._detach_metric(
                main_metrics["ignored_negative_iou_mean"]
            ),
            "text_negative_count": main_metrics[
                "text_negative_count"
            ],
            "text_negative_weight_mean": self._detach_metric(
                main_metrics["text_negative_weight_mean"]
            ),
            "matched_iou_mean": self._detach_metric(
                main_metrics["matched_iou_mean"]
            ),
            "score_iou_mean": self._detach_metric(
                main_metrics["score_iou_mean"]
            ),
            "score_target_pos_mean": self._detach_metric(
                main_metrics["score_target_pos_mean"]
            ),
            "score_target_pos_min": self._detach_metric(
                main_metrics["score_target_pos_min"]
            ),
            "score_target_pos_max": self._detach_metric(
                main_metrics["score_target_pos_max"]
            ),
            "lambda_bbox": main_metrics["lambda_bbox"],
            "lambda_giou": main_metrics["lambda_giou"],
            "lambda_score": main_metrics["lambda_score"],
            "lambda_duplicate": main_metrics["lambda_duplicate"],
            "lambda_hard_negative": main_metrics[
                "lambda_hard_negative"
            ],
            "lambda_rank": 0.0,
            "lambda_rank_eff": 0.0,
            "pos_weight": 1.0,
            "quality_alpha": float(quality_alpha),
            # Backward-compatible field; now means matcher score alpha.
            "rank_alpha": float(matcher_score_alpha),
        }

        for key, value in main_metrics.items():
            loss_dict[f"main_{key}"] = self._detach_metric(value)
        for key, value in aux_metrics.items():
            loss_dict[f"aux_{key}"] = self._detach_metric(value)

        return loss, loss_dict


def build_grounding_loss_from_config(
    config: Dict[str, Any],
) -> GroundingLoss:
    """Public config adapter used by train.py."""
    return GroundingLoss.from_config(config)


def grounding_loss_forward_kwargs_from_config(
    config: Dict[str, Any],
    *,
    current_epoch: int,
) -> Dict[str, Any]:
    """
    Build forward kwargs from the same YAML.

    Dynamic bbox/score interpolation remains in train.py when already present;
    this helper only provides deterministic defaults and schedule parameters.
    """
    loss_cfg = config.get("loss", config)
    weights = dict(loss_cfg.get("weight", {}))
    quality = dict(loss_cfg.get("quality", {}))
    ranking = dict(loss_cfg.get("ranking", {}))
    matcher_schedule = dict(
        loss_cfg.get("matcher_schedule", {})
    )
    hybrid = dict(loss_cfg.get("hybrid", {}))

    return {
        "current_epoch": int(current_epoch),
        "quality_warmup_epoch": int(
            quality.get("quality_warmup_epoch", 10)
        ),
        "rank_start_epoch": int(
            matcher_schedule.get(
                "start_epoch",
                ranking.get("rank_start_epoch", 5),
            )
        ),
        "rank_warmup_epoch": int(
            matcher_schedule.get(
                "warmup_epoch",
                ranking.get("rank_warmup_epoch", 12),
            )
        ),
        "rank_alpha_min": float(
            matcher_schedule.get(
                "alpha_min",
                ranking.get("rank_alpha_min", 0.0),
            )
        ),
        "lambda_bbox": float(
            weights.get("bbox_start", 5.0)
        ),
        "lambda_giou": float(weights.get("giou", 2.0)),
        "lambda_score": float(
            weights.get("score_start", 2.0)
        ),
        "lambda_rank": 0.0,
        "lambda_aux": float(
            hybrid.get("aux_loss_weight", 0.28)
        ),
    }

