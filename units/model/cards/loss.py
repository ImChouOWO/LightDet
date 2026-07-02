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






@dataclass(frozen=True)
class ScoreAssignmentResult:
    """
    Score-only assignment that is independent from bbox/GIoU matching.

    quality_target already contains IoU quality shaping and optional
    multi-round decay. round_weight is used to apply the same decay during
    quality warmup. Both tensors have shape [B, N, 1].
    """

    quality_target: torch.Tensor
    round_weight: torch.Tensor
    positive_mask: torch.Tensor
    # Dense localization quality for every prediction. Unlike quality_target,
    # this is not sparse and is not determined by a one-to-one matcher.
    max_iou_per_pred: torch.Tensor
    counts: Tuple[int, ...]
    selected_ious: torch.Tensor
    mode: str
    rounds: int

    @property
    def num_matches(self) -> int:
        return int(sum(self.counts))


class IndependentIoUScoreAssigner:
    """
    Build score targets without using the bbox matcher or score logits.

    Fast paths:
      - 0/1 GT rows are fully vectorized.
      - Multi-GT rows are grouped by equal GT count. IoU matrices are built
        in batched GPU tensors and copied to CPU once per GT-count group.
      - SciPy Hungarian itself remains per query because SciPy has no batched
        API, but the expensive pairwise tensor construction is batched.

    rounds=1:
      IoU-only one-to-one assignment. Each GT supervises at most one score.

    rounds>1:
      Repeats IoU-only one-to-one assignment after removing previously
      selected predictions. This becomes score one-to-many supervision.
      Later rounds are reduced by round_decay ** round_index.
    """

    def __init__(
        self,
        rounds: int = 1,
        quality_gamma: float = 1.0,
        round_decay: float = 0.25,
        min_iou: float = 0.0,
    ) -> None:
        self.rounds = max(1, int(rounds))
        self.quality_gamma = max(0.0, float(quality_gamma))
        self.round_decay = clamp01(round_decay)
        self.min_iou = clamp01(min_iou)

    @torch.no_grad()
    def __call__(
        self,
        pred_bbox: torch.Tensor,
        packed_targets: PackedTargets,
        *,
        output_dtype: torch.dtype,
    ) -> ScoreAssignmentResult:
        if pred_bbox.ndim != 3 or pred_bbox.shape[-1] != 4:
            raise ValueError(
                "pred_bbox must have shape [B, N, 4], got "
                f"{tuple(pred_bbox.shape)}"
            )

        batch_size, num_pred, _ = pred_bbox.shape
        device = pred_bbox.device

        quality_target = torch.zeros(
            (batch_size, num_pred, 1),
            device=device,
            dtype=output_dtype,
        )
        round_weight = torch.zeros_like(quality_target)
        positive_mask = torch.zeros(
            (batch_size, num_pred, 1),
            device=device,
            dtype=torch.bool,
        )
        max_iou_per_pred = torch.zeros(
            (batch_size, num_pred, 1),
            device=device,
            dtype=torch.float32,
        )
        counts = [0] * batch_size
        selected_iou_parts: List[torch.Tensor] = []

        if num_pred == 0 or packed_targets.boxes.numel() == 0:
            return ScoreAssignmentResult(
                quality_target=quality_target,
                round_weight=round_weight,
                positive_mask=positive_mask,
                max_iou_per_pred=max_iou_per_pred,
                counts=tuple(counts),
                selected_ious=torch.empty(
                    0,
                    device=device,
                    dtype=torch.float32,
                ),
                mode="empty",
                rounds=self.rounds,
            )

        single_rows = [
            index
            for index, count in enumerate(packed_targets.counts)
            if count == 1
        ]

        if single_rows:
            batch_index = torch.tensor(
                single_rows,
                device=device,
                dtype=torch.long,
            )
            global_gt_index = packed_targets.offsets[batch_index]
            gt_bbox = packed_targets.boxes[global_gt_index].float()
            boxes = pred_bbox.detach().float().index_select(0, batch_index)
            iou = batched_single_target_iou(boxes, gt_bbox)
            max_iou_per_pred[
                batch_index,
                :,
                0,
            ] = iou.clamp(0.0, 1.0)

            k = min(self.rounds, num_pred)
            selected_iou, selected_pred = torch.topk(
                iou,
                k=k,
                dim=1,
                largest=True,
                sorted=True,
            )
            decay = (
                self.round_decay
                ** torch.arange(
                    k,
                    device=device,
                    dtype=torch.float32,
                )
            )
            shaped_quality = (
                selected_iou.clamp(0.0, 1.0)
                .pow(self.quality_gamma)
                * decay[None, :]
            )
            valid = selected_iou >= self.min_iou

            row_index = batch_index[:, None].expand_as(selected_pred)
            quality_target[
                row_index[valid],
                selected_pred[valid],
                0,
            ] = shaped_quality[valid].to(output_dtype)
            round_weight[
                row_index[valid],
                selected_pred[valid],
                0,
            ] = decay[None, :].expand_as(selected_iou)[valid].to(output_dtype)
            positive_mask[
                row_index[valid],
                selected_pred[valid],
                0,
            ] = True

            valid_counts = valid.sum(dim=1).detach().cpu().tolist()
            for row, original_batch_index in enumerate(single_rows):
                counts[original_batch_index] = int(valid_counts[row])
            if bool(valid.any()):
                selected_iou_parts.append(selected_iou[valid])

        multi_groups = group_rows_by_target_count(
            packed_targets.counts,
            minimum_count=2,
        )

        if multi_groups and linear_sum_assignment is None:
            raise ImportError(
                "Multi-GT score assignment requires scipy. "
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
            iou_group = batched_pairwise_iou(
                boxes,
                gt_bbox,
            )
            max_iou_group = iou_group.max(dim=2).values
            max_iou_per_pred.index_copy_(
                0,
                row_tensor,
                max_iou_group.unsqueeze(-1),
            )
            # One transfer/synchronization per equal-GT-count group.
            iou_group_cpu = iou_group.detach().cpu().numpy()

            for group_row, batch_index in enumerate(rows):
                iou_np = iou_group_cpu[group_row]
                available = np.ones(
                    num_pred,
                    dtype=np.bool_,
                )
                query_count = 0

                for round_index in range(self.rounds):
                    available_pred = np.flatnonzero(available)
                    if available_pred.size == 0:
                        break

                    local_iou = iou_np[available_pred]
                    pred_np, gt_np = linear_sum_assignment(
                        -local_iou
                    )
                    if len(pred_np) == 0:
                        break

                    selected_pred_np = available_pred[pred_np]
                    selected_iou_np = local_iou[pred_np, gt_np]
                    available[selected_pred_np] = False

                    valid_np = selected_iou_np >= self.min_iou
                    if not bool(valid_np.any()):
                        continue

                    selected_pred_np = selected_pred_np[valid_np]
                    selected_iou_np = selected_iou_np[valid_np]
                    decay_value = float(
                        self.round_decay ** round_index
                    )

                    selected_pred = torch.as_tensor(
                        selected_pred_np,
                        device=device,
                        dtype=torch.long,
                    )
                    selected_iou = torch.as_tensor(
                        selected_iou_np,
                        device=device,
                        dtype=torch.float32,
                    )
                    shaped_quality = (
                        selected_iou.clamp(0.0, 1.0)
                        .pow(self.quality_gamma)
                        * decay_value
                    )

                    quality_target[
                        batch_index,
                        selected_pred,
                        0,
                    ] = shaped_quality.to(output_dtype)
                    round_weight[
                        batch_index,
                        selected_pred,
                        0,
                    ] = decay_value
                    positive_mask[
                        batch_index,
                        selected_pred,
                        0,
                    ] = True

                    query_count += int(selected_pred.numel())
                    selected_iou_parts.append(selected_iou)

                counts[batch_index] = query_count

        if selected_iou_parts:
            selected_ious = torch.cat(selected_iou_parts).float()
        else:
            selected_ious = torch.empty(
                0,
                device=device,
                dtype=torch.float32,
            )

        mode = (
            "iou_one_to_one"
            if self.rounds == 1
            else "iou_multi_round"
        )
        if multi_groups:
            mode += "_grouped_multi_gt"

        return ScoreAssignmentResult(
            quality_target=quality_target,
            round_weight=round_weight,
            positive_mask=positive_mask,
            max_iou_per_pred=max_iou_per_pred,
            counts=tuple(counts),
            selected_ious=selected_ious,
            mode=mode,
            rounds=self.rounds,
        )


class HungarianOneToOneMatcher:
    """
    Main-branch Hungarian matcher.

    Fast paths:
      - 0/1 GT rows are fully vectorized.
      - Multi-GT rows are grouped by equal GT count. Pairwise costs are built
        in batched tensors and copied to CPU once per group. SciPy assignment
        remains per query because SciPy has no batched Hungarian API.
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
        empty = torch.empty(0, dtype=torch.long, device=device)
        per_batch: List[Tuple[torch.Tensor, torch.Tensor]] = [
            (empty, empty) for _ in range(batch_size)
        ]
        pred_score = score_logit.detach().float().sigmoid()

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
            cost = (
                self.cost_bbox * cost_bbox
                - self.cost_giou * giou
            )
            if self.cost_score != 0.0:
                cost = (
                    cost
                    - self.cost_score
                    * pred_score.index_select(
                        0,
                        batch_index,
                    )
                )

            best_pred = torch.argmin(
                cost,
                dim=1,
            )

            if all(
                count <= 1
                for count in packed_targets.counts
            ):
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
                    mode="batched_single_gt",
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
            giou = batched_pairwise_giou(
                boxes,
                gt_bbox,
            )
            cost = (
                self.cost_bbox * cost_bbox
                - self.cost_giou * giou
            )
            if self.cost_score != 0.0:
                cost = (
                    cost
                    - self.cost_score
                    * pred_score.index_select(
                        0,
                        row_tensor,
                    )[:, :, None]
                )

            # One GPU->CPU synchronization per equal-GT-count group.
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
            "batched_single_gt"
            if not multi_groups
            else "grouped_multi_gt_scipy"
        )
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
    H-DETR-inspired hybrid loss for LightDet.

    Main regression branch:
      - Score-free Hungarian one-to-one bbox/GIoU assignment.
      - Used for validation and inference.

    Main score branch:
      - Independent IoU-only score assignment.
      - Default is one-to-one (score_match_rounds=1).
      - Does not consume bbox matcher output or score logits.
      - Negative text queries still receive all-zero QFL targets.

    Auxiliary regression branch:
      - H-DETR repeated-GT Hungarian one-to-many assignment.
      - Used only during training.

    Auxiliary score:
      - Disabled by default so score remains a main one-to-one task.
      - Can be enabled; if enabled it uses the same independent score assigner.

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
        # Legacy fixed threshold. When provided, it overrides the dynamic
        # schedule and keeps the threshold constant for backward compatibility.
        score_negative_iou_ignore_thr: Optional[float] = None,
        score_negative_iou_ignore_start: float = 0.50,
        score_negative_iou_ignore_end: float = 0.45,
        score_negative_iou_ignore_start_epoch: int = 5,
        score_negative_iou_ignore_end_epoch: int = 25,
        score_negative_iou_ignore_schedule: str = "cosine",
        rank_negative_iou_max: float = 0.20,
        text_negative_loss_weight: float = 0.50,
        text_negative_topk: int = 20,
        text_negative_hard_mix: float = 0.50,
        aux_loss_weight: float = 0.5,
        aux_cost_score: Optional[float] = None,
        score_match_rounds: int = 1,
        score_quality_gamma: float = 1.0,
        score_round_decay: float = 0.25,
        score_min_iou: float = 0.0,
        enable_pairwise_ranking: bool = False,
        aux_score_enabled: bool = False,
    ) -> None:
        super().__init__()

        self.hard_negative_ratio = max(1, int(hard_negative_ratio))
        self.aux_positive_label = clamp01(aux_positive_label)
        self.quality_min = clamp01(quality_min)
        self.quality_max = clamp01(quality_max)

        if self.quality_min > self.quality_max:
            raise ValueError(
                "quality_min must be <= quality_max, got "
                f"{self.quality_min} > {self.quality_max}"
            )

        self.aux_loss_weight = max(0.0, float(aux_loss_weight))
        self.enable_pairwise_ranking = bool(enable_pairwise_ranking)
        self.aux_score_enabled = bool(aux_score_enabled)

        # Score is deliberately excluded from both regression matchers.
        # cost_score/aux_cost_score are accepted only for old config files.
        self.configured_cost_score = float(cost_score)
        self.configured_aux_cost_score = (
            float(cost_score)
            if aux_cost_score is None
            else float(aux_cost_score)
        )

        self.main_matcher = HungarianOneToOneMatcher(
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
            cost_score=0.0,
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

        self.score_assigner = IndependentIoUScoreAssigner(
            rounds=score_match_rounds,
            quality_gamma=score_quality_gamma,
            round_decay=score_round_decay,
            min_iou=score_min_iou,
        )

        self.matcher = self.main_matcher
        self.qfl_beta = float(qfl_beta)
        self.rank_margin = float(rank_margin)
        self.rank_min_quality_gap = float(rank_min_quality_gap)
        self.rank_max_pairs = max(1, int(rank_max_pairs))
        self.max_query_loss_weight = max(
            1.0,
            float(max_query_loss_weight),
        )
        # Dynamic threshold used to protect high-IoU unmatched candidates
        # from being treated as score negatives. A legacy fixed threshold
        # forces a constant schedule so older callers remain valid.
        if score_negative_iou_ignore_thr is not None:
            fixed_threshold = clamp01(score_negative_iou_ignore_thr)
            score_negative_iou_ignore_start = fixed_threshold
            score_negative_iou_ignore_end = fixed_threshold
            score_negative_iou_ignore_start_epoch = 1
            score_negative_iou_ignore_end_epoch = 1
            score_negative_iou_ignore_schedule = "constant"

        self.score_negative_iou_ignore_start = clamp01(
            score_negative_iou_ignore_start
        )
        self.score_negative_iou_ignore_end = clamp01(
            score_negative_iou_ignore_end
        )
        self.score_negative_iou_ignore_start_epoch = max(
            1,
            int(score_negative_iou_ignore_start_epoch),
        )
        self.score_negative_iou_ignore_end_epoch = max(
            self.score_negative_iou_ignore_start_epoch,
            int(score_negative_iou_ignore_end_epoch),
        )
        self.score_negative_iou_ignore_schedule = str(
            score_negative_iou_ignore_schedule
        ).strip().lower()

        schedule_aliases = {
            "cos": "cosine",
            "cosine": "cosine",
            "linear": "linear",
            "smooth": "smoothstep",
            "smoothstep": "smoothstep",
            "constant": "constant",
        }
        if self.score_negative_iou_ignore_schedule not in schedule_aliases:
            raise ValueError(
                "score_negative_iou_ignore_schedule must be one of "
                "['constant', 'cosine', 'linear', 'smoothstep'], got "
                f"{self.score_negative_iou_ignore_schedule!r}"
            )
        self.score_negative_iou_ignore_schedule = schedule_aliases[
            self.score_negative_iou_ignore_schedule
        ]

        self.rank_negative_iou_max = clamp01(
            rank_negative_iou_max
        )
        self.text_negative_loss_weight = max(
            0.0,
            float(text_negative_loss_weight),
        )
        self.text_negative_topk = max(1, int(text_negative_topk))
        self.text_negative_hard_mix = clamp01(
            text_negative_hard_mix
        )

    def resolve_score_negative_iou_ignore_thr(
        self,
        current_epoch: Optional[int],
    ) -> float:
        """
        Resolve the per-epoch IoU threshold used to ignore unmatched score
        negatives that already overlap a GT well.

        A prediction is ignored from QFL negative mining when:
            not score-positive and max_iou_to_any_gt >= resolved threshold

        Lowering the threshold over training protects more medium/high-IoU
        candidates after localization becomes stable. The value is constant
        before start_epoch and after end_epoch.
        """
        start = float(self.score_negative_iou_ignore_start)
        end = float(self.score_negative_iou_ignore_end)
        start_epoch = int(self.score_negative_iou_ignore_start_epoch)
        end_epoch = int(self.score_negative_iou_ignore_end_epoch)
        schedule = self.score_negative_iou_ignore_schedule

        if current_epoch is None:
            return start

        epoch = int(current_epoch)
        if schedule == "constant" or end_epoch <= start_epoch:
            return start
        if epoch <= start_epoch:
            return start
        if epoch >= end_epoch:
            return end

        progress = clamp01(
            float(epoch - start_epoch)
            / float(end_epoch - start_epoch)
        )

        if schedule == "linear":
            alpha = progress
        elif schedule == "cosine":
            alpha = 0.5 * (1.0 - math.cos(math.pi * progress))
        elif schedule == "smoothstep":
            alpha = smoothstep(progress)
        else:
            alpha = 0.0

        return clamp01(start + (end - start) * alpha)


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

    def text_negative_suppression_loss(
        self,
        pred_score_logit: torch.Tensor,
        text_negative_mask: torch.Tensor,
        query_loss_weights: Optional[torch.Tensor],
        positive_query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Suppress every prediction emitted for a negative text query.

        The existing QFL path performs hard-negative mining and therefore only
        supervises a few predictions when a query has no positive target. This
        dedicated term keeps a mean over all predictions and mixes it with the
        hardest Top-K predictions, preventing a flat high-confidence score
        platform on semantically incorrect queries.
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

        # QFL-style target=0 loss for every prediction.
        element_loss = (
            F.softplus(negative_logits)
            * negative_prob.pow(float(self.qfl_beta))
        )
        all_per_query = element_loss.mean(dim=1)

        k = min(self.text_negative_topk, int(element_loss.shape[1]))
        hard_per_query = torch.topk(
            element_loss,
            k=k,
            dim=1,
            largest=True,
            sorted=False,
        ).values.mean(dim=1)

        hard_mix = float(self.text_negative_hard_mix)
        per_query = (
            (1.0 - hard_mix) * all_per_query
            + hard_mix * hard_per_query
        )

        query_weight = self._prepare_query_loss_weights(
            pred_score_logit,
            query_loss_weights,
        ).reshape(-1)[mask].float()
        loss = (per_query * query_weight).mean()

        negative_top1_score = negative_prob.max(dim=1).values.mean()
        positive_top1_score = zero.float()

        if positive_query_mask is not None:
            positive_query_mask = positive_query_mask.to(
                device=logits.device,
                dtype=torch.bool,
            ).reshape(-1)
            if bool(positive_query_mask.any()):
                positive_top1_score = (
                    logits[positive_query_mask]
                    .sigmoid()
                    .max(dim=1)
                    .values
                    .mean()
                )

        margin = positive_top1_score - negative_top1_score
        return loss.to(dtype=pred_score_logit.dtype), {
            "all_mean": all_per_query.mean().to(pred_score_logit.dtype),
            "hard_mean": hard_per_query.mean().to(pred_score_logit.dtype),
            "negative_top1_score": negative_top1_score.to(
                pred_score_logit.dtype
            ),
            "positive_top1_score": positive_top1_score.to(
                pred_score_logit.dtype
            ),
            "positive_negative_margin": margin.to(
                pred_score_logit.dtype
            ),
        }

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
        negative_ignore_mask: Optional[torch.Tensor] = None,
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

        score_logit_fp32 = pred_score_logit.float()
        score_target_fp32 = score_target.float()
        pred_prob = score_logit_fp32.sigmoid()
        bce = F.binary_cross_entropy_with_logits(
            score_logit_fp32,
            score_target_fp32,
            reduction="none",
        )
        qfl_weight = (score_target_fp32 - pred_prob).abs().pow(float(qfl_beta))
        loss_flat = (bce * qfl_weight).squeeze(-1)
        positive_flat = positive_mask.squeeze(-1)

        if negative_ignore_mask is None:
            ignore_flat = torch.zeros_like(positive_flat)
        else:
            ignore_flat = negative_ignore_mask.to(
                device=positive_flat.device,
                dtype=torch.bool,
            ).squeeze(-1)
            if ignore_flat.shape != positive_flat.shape:
                raise ValueError(
                    "negative_ignore_mask/positive_mask shape mismatch: "
                    f"{tuple(ignore_flat.shape)} != "
                    f"{tuple(positive_flat.shape)}"
                )

        valid_negative_flat = (~positive_flat) & (~ignore_flat)

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

        negative_counts = tuple(
            int(value)
            for value in valid_negative_flat.sum(dim=1).detach().cpu().tolist()
        )
        ignored_negative_count = float(ignore_flat.sum().detach().cpu().item())
        selected_counts = tuple(
            min(negative_count, max(positive_count, 1) * self.hard_negative_ratio)
            for positive_count, negative_count in zip(positive_counts, negative_counts)
        )
        max_selected = max(selected_counts, default=0)

        if max_selected > 0:
            negative_loss = loss_flat.masked_fill(
                ~valid_negative_flat,
                float("-inf"),
            )
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
            float(ignored_negative_count),
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
        dense_quality: Optional[torch.Tensor] = None,
        negative_valid_mask: Optional[torch.Tensor] = None,
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

        if dense_quality is None:
            dense_quality_flat = quality
        else:
            dense_quality_flat = dense_quality.to(
                device=quality.device,
                dtype=quality.dtype,
            ).squeeze(-1)

        if negative_valid_mask is None:
            valid_negative = ~positive
        else:
            valid_negative = negative_valid_mask.to(
                device=positive.device,
                dtype=torch.bool,
            ).squeeze(-1)
            if valid_negative.shape != positive.shape:
                raise ValueError(
                    "negative_valid_mask/positive_mask shape mismatch"
                )

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

        negative_counts = tuple(
            int(value)
            for value in valid_negative.sum(dim=1).detach().cpu().tolist()
        )
        selected_negative_counts = tuple(
            min(
                negative_count,
                max(positive_count * self.hard_negative_ratio, 1),
            )
            if positive_count > 0
            else 0
            for positive_count, negative_count in zip(
                positive_counts,
                negative_counts,
            )
        )
        max_negative = max(selected_negative_counts, default=0)

        if max_negative > 0:
            negative_values, negative_indices = torch.topk(
                logits.masked_fill(~valid_negative, float("-inf")),
                k=max_negative,
                dim=1,
                largest=True,
                sorted=True,
            )
            negative_quality = torch.gather(
                dense_quality_flat,
                1,
                negative_indices,
            )
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
                positive_valid[:, :, None]
                & negative_valid[:, None, :]
                & (
                    positive_quality[:, :, None]
                    > negative_quality[:, None, :]
                    + float(min_quality_gap)
                )
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
        regression_assignments: AssignmentResult,
        score_assignments: Optional[ScoreAssignmentResult],
        lambda_bbox: float,
        lambda_giou: float,
        lambda_score: float,
        pos_weight: float,
        quality_alpha: float,
        query_loss_weights: Optional[torch.Tensor],
        text_negative_mask: Optional[torch.Tensor],
        warmup_positive_target: float,
        score_enabled: bool,
        apply_ranking: bool,
        rank_alpha: float,
        lambda_rank: float,
        score_negative_iou_ignore_thr: float,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self._validate_branch_inputs(
            branch_name=branch_name,
            pred_bbox=pred_bbox,
            pred_score_logit=pred_score_logit,
            targets=targets,
        )

        total_regression_pairs = regression_assignments.num_matches

        if total_regression_pairs > 0:
            batch_index = regression_assignments.batch_indices
            pred_index = regression_assignments.pred_indices
            global_gt_index = packed_targets.global_gt_indices(
                batch_index,
                regression_assignments.gt_indices,
            )
            positive_pred_boxes = pred_bbox[batch_index, pred_index]
            positive_gt_boxes = packed_targets.boxes[
                global_gt_index
            ].to(dtype=pred_bbox.dtype)

            loss_bbox = F.l1_loss(
                positive_pred_boxes,
                positive_gt_boxes,
                reduction="sum",
            ) / total_regression_pairs

            matched_giou = matched_generalized_box_iou(
                positive_pred_boxes.float(),
                positive_gt_boxes.float(),
            ).to(dtype=pred_bbox.dtype)
            loss_giou = (
                1.0 - matched_giou
            ).sum() / total_regression_pairs

            with torch.no_grad():
                regression_iou = matched_box_iou(
                    positive_pred_boxes.detach().float(),
                    positive_gt_boxes.float(),
                ).clamp(0.0, 1.0)
                matched_iou_mean = regression_iou.mean()
        else:
            zero = pred_bbox.new_zeros(())
            loss_bbox = zero
            loss_giou = zero
            matched_iou_mean = zero

        if score_enabled:
            if score_assignments is None:
                raise ValueError(
                    f"{branch_name}: score_assignments is required "
                    "when score_enabled=True"
                )

            positive_mask = score_assignments.positive_mask
            final_quality_target = (
                score_assignments.quality_target.float()
                .clamp(0.0, 1.0)
            )
            warmup_target = (
                clamp01(warmup_positive_target)
                * score_assignments.round_weight.float()
            )
            score_target = (
                (1.0 - float(quality_alpha)) * warmup_target
                + float(quality_alpha) * final_quality_target
            ).clamp(0.0, 1.0).to(
                dtype=pred_score_logit.dtype
            )
            score_target = torch.where(
                positive_mask,
                score_target,
                torch.zeros_like(score_target),
            )
            positive_counts = score_assignments.counts
            max_iou_per_pred = score_assignments.max_iou_per_pred.to(
                device=pred_score_logit.device,
                dtype=torch.float32,
            )

            if float(score_negative_iou_ignore_thr) > 0.0:
                negative_ignore_mask = (
                    (~positive_mask)
                    & (
                        max_iou_per_pred
                        >= float(score_negative_iou_ignore_thr)
                    )
                )
            else:
                negative_ignore_mask = torch.zeros_like(
                    positive_mask,
                    dtype=torch.bool,
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
                ignored_negative_count,
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
                positive_counts=positive_counts,
                negative_ignore_mask=negative_ignore_mask,
            )

            positive_query_mask = torch.tensor(
                [count > 0 for count in packed_targets.counts],
                device=pred_score_logit.device,
                dtype=torch.bool,
            )
            text_negative_mask_prepared = self._prepare_text_negative_mask(
                pred_score_logit,
                text_negative_mask,
            )
            (
                loss_text_negative_all,
                text_negative_metrics,
            ) = self.text_negative_suppression_loss(
                pred_score_logit=pred_score_logit,
                text_negative_mask=text_negative_mask_prepared,
                query_loss_weights=query_loss_weights,
                positive_query_mask=positive_query_mask,
            )
            loss_text_negative_contrib = (
                float(self.text_negative_loss_weight)
                * loss_text_negative_all
            )

            if score_assignments.selected_ious.numel() > 0:
                score_iou_mean = (
                    score_assignments.selected_ious.mean()
                )
            else:
                score_iou_mean = pred_bbox.new_zeros(())

            positive_targets = score_target[positive_mask]
            if positive_targets.numel() > 0:
                score_target_pos_mean = positive_targets.mean()
                score_target_pos_min = positive_targets.min()
                score_target_pos_max = positive_targets.max()
            else:
                zero = pred_bbox.new_zeros(())
                score_target_pos_mean = zero
                score_target_pos_min = zero
                score_target_pos_max = zero

            score_assignment_mode = score_assignments.mode
            score_match_rounds = score_assignments.rounds
        else:
            score_target = torch.zeros_like(pred_score_logit)
            positive_mask = torch.zeros_like(
                pred_score_logit,
                dtype=torch.bool,
            )
            zero = pred_bbox.new_zeros(())
            loss_score = zero
            loss_score_pos = zero
            loss_score_neg = zero
            loss_score_neg_unweighted = zero
            loss_text_negative = zero
            loss_text_negative_all = zero
            loss_text_negative_contrib = zero
            text_negative_metrics = {
                "all_mean": zero,
                "hard_mean": zero,
                "negative_top1_score": zero,
                "positive_top1_score": zero,
                "positive_negative_margin": zero,
            }
            ignored_negative_count = 0.0
            max_iou_per_pred = torch.zeros_like(
                pred_score_logit,
                dtype=torch.float32,
            )
            negative_ignore_mask = torch.zeros_like(
                pred_score_logit,
                dtype=torch.bool,
            )
            score_pos_count = 0.0
            hard_negative_count = 0.0
            negative_count = float(
                pred_score_logit.shape[0]
                * pred_score_logit.shape[1]
            )
            text_negative_count = 0.0
            text_negative_weight_mean = zero
            score_iou_mean = zero
            score_target_pos_mean = zero
            score_target_pos_min = zero
            score_target_pos_max = zero
            positive_counts = tuple(
                0 for _ in range(pred_score_logit.shape[0])
            )
            score_assignment_mode = "disabled"
            score_match_rounds = 0

        if (
            score_enabled
            and bool(apply_ranking)
            and float(lambda_rank) > 0.0
            and float(rank_alpha) > 0.0
        ):
            loss_rank_raw = self.pairwise_quality_rank_loss(
                pred_score_logit=pred_score_logit,
                score_target=score_target,
                positive_mask=positive_mask,
                margin=self.rank_margin,
                min_quality_gap=self.rank_min_quality_gap,
                positive_counts=positive_counts,
                dense_quality=max_iou_per_pred,
                negative_valid_mask=(
                    (~positive_mask)
                    & (
                        max_iou_per_pred
                        <= self.rank_negative_iou_max
                    )
                ),
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
        loss_total = (
            loss_base
            + loss_rank_contrib
            + loss_text_negative_contrib
        )

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
            "loss_score_neg_unweighted": loss_score_neg_unweighted,
            # Dedicated all-prediction negative-text suppression.
            "loss_text_negative": loss_text_negative_all,
            "loss_text_negative_contrib": loss_text_negative_contrib,
            # Legacy hard-mined QFL diagnostic for negative text rows.
            "loss_text_negative_hard_qfl": loss_text_negative,
            "loss_text_negative_all_mean": text_negative_metrics[
                "all_mean"
            ],
            "loss_text_negative_hard_mean": text_negative_metrics[
                "hard_mean"
            ],
            "negative_query_top1_score": text_negative_metrics[
                "negative_top1_score"
            ],
            "positive_query_top1_score": text_negative_metrics[
                "positive_top1_score"
            ],
            "positive_negative_score_margin": text_negative_metrics[
                "positive_negative_margin"
            ],
            "loss_rank": loss_rank,
            "loss_rank_raw": loss_rank_raw,
            "loss_rank_contrib": loss_rank_contrib,
            "matched": float(total_regression_pairs),
            "score_pos_count": float(score_pos_count),
            "hard_neg_count": float(hard_negative_count),
            "negative_count": float(negative_count),
            "ignored_negative_count": float(ignored_negative_count),
            "selected_negative_fraction": float(
                selected_negative_fraction
            ),
            "text_negative_count": float(text_negative_count),
            "text_negative_weight_mean": text_negative_weight_mean,
            "matched_iou_mean": matched_iou_mean,
            "score_iou_mean": score_iou_mean,
            "score_target_pos_mean": score_target_pos_mean,
            "score_target_pos_min": score_target_pos_min,
            "score_target_pos_max": score_target_pos_max,
            "lambda_bbox": float(lambda_bbox),
            "lambda_giou": float(lambda_giou),
            "lambda_score": float(lambda_score),
            "lambda_text_negative": float(
                self.text_negative_loss_weight
            ),
            "score_negative_iou_ignore_thr": float(
                score_negative_iou_ignore_thr
            ),
            "rank_negative_iou_max": float(
                self.rank_negative_iou_max
            ),
            "lambda_rank": float(
                lambda_rank if apply_ranking else 0.0
            ),
            "lambda_rank_eff": float(
                lambda_rank * rank_alpha
                if apply_ranking
                else 0.0
            ),
            "pos_weight": float(pos_weight),
            "quality_alpha": float(quality_alpha),
            "rank_alpha": float(
                rank_alpha if apply_ranking else 0.0
            ),
            "assignment_mode": regression_assignments.mode,
            "score_assignment_mode": score_assignment_mode,
            "score_match_rounds": int(score_match_rounds),
            "score_enabled": bool(score_enabled),
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
        Main:
          - bbox/GIoU: score-free Hungarian one-to-one.
          - score: independent IoU-only one-to-one by default.

        Auxiliary:
          - bbox/GIoU: H-DETR repeated-GT Hungarian one-to-many.
          - score: disabled by default.

        The public signature remains compatible with the existing train.py.
        """
        quality_alpha, rank_alpha = self.resolve_epoch_alpha(
            current_epoch=current_epoch,
            quality_alpha=quality_alpha,
            rank_alpha=rank_alpha,
            quality_warmup_epoch=quality_warmup_epoch,
            rank_start_epoch=rank_start_epoch,
            rank_warmup_epoch=rank_warmup_epoch,
            rank_alpha_min=rank_alpha_min,
        )
        score_negative_iou_ignore_thr = (
            self.resolve_score_negative_iou_ignore_thr(
                current_epoch=current_epoch,
            )
        )

        packed_targets = PackedTargets.from_targets(
            targets,
            device=pred_bbox.device,
            dtype=pred_bbox.dtype,
        )

        main_regression_assignments = self.main_matcher(
            pred_bbox=pred_bbox,
            pred_score_logit=pred_score_logit,
            targets=targets,
            packed_targets=packed_targets,
        )
        main_score_assignments = self.score_assigner(
            pred_bbox=pred_bbox,
            packed_targets=packed_targets,
            output_dtype=pred_score_logit.dtype,
        )

        main_loss, main_metrics = self._compute_branch_loss(
            branch_name="main",
            pred_bbox=pred_bbox,
            pred_score_logit=pred_score_logit,
            targets=targets,
            packed_targets=packed_targets,
            regression_assignments=main_regression_assignments,
            score_assignments=main_score_assignments,
            lambda_bbox=lambda_bbox,
            lambda_giou=lambda_giou,
            lambda_score=lambda_score,
            pos_weight=pos_weight,
            quality_alpha=quality_alpha,
            query_loss_weights=query_loss_weights,
            text_negative_mask=text_negative_mask,
            warmup_positive_target=self.quality_max,
            score_enabled=True,
            apply_ranking=self.enable_pairwise_ranking,
            rank_alpha=rank_alpha,
            lambda_rank=lambda_rank,
            score_negative_iou_ignore_thr=(
                score_negative_iou_ignore_thr
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

            aux_pos_weight_eff = (
                float(pos_weight)
                if aux_pos_weight is None
                else float(aux_pos_weight)
            )

            aux_regression_assignments = self.aux_matcher(
                pred_bbox=aux_pred_bbox,
                pred_score_logit=aux_pred_score_logit,
                targets=targets,
                packed_targets=packed_targets,
            )

            aux_score_assignments = None
            if self.aux_score_enabled:
                aux_score_assignments = self.score_assigner(
                    pred_bbox=aux_pred_bbox,
                    packed_targets=packed_targets,
                    output_dtype=aux_pred_score_logit.dtype,
                )

            aux_loss, aux_metrics = self._compute_branch_loss(
                branch_name="aux",
                pred_bbox=aux_pred_bbox,
                pred_score_logit=aux_pred_score_logit,
                targets=targets,
                packed_targets=packed_targets,
                regression_assignments=aux_regression_assignments,
                score_assignments=aux_score_assignments,
                lambda_bbox=aux_lambda_bbox_eff,
                lambda_giou=aux_lambda_giou_eff,
                lambda_score=aux_lambda_score_eff,
                pos_weight=aux_pos_weight_eff,
                quality_alpha=quality_alpha,
                query_loss_weights=query_loss_weights,
                text_negative_mask=text_negative_mask,
                warmup_positive_target=self.aux_positive_label,
                score_enabled=self.aux_score_enabled,
                apply_ranking=False,
                rank_alpha=0.0,
                lambda_rank=0.0,
                score_negative_iou_ignore_thr=(
                    score_negative_iou_ignore_thr
                ),
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
                "loss_text_negative": zero,
                "loss_text_negative_contrib": zero,
                "loss_text_negative_hard_qfl": zero,
                "loss_text_negative_all_mean": zero,
                "loss_text_negative_hard_mean": zero,
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
                "selected_negative_fraction": 0.0,
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
                "score_negative_iou_ignore_thr": float(
                    score_negative_iou_ignore_thr
                ),
                "rank_negative_iou_max": float(
                    self.rank_negative_iou_max
                ),
                "lambda_rank": 0.0,
                "lambda_rank_eff": 0.0,
                "pos_weight": float(pos_weight),
                "quality_alpha": float(quality_alpha),
                "rank_alpha": 0.0,
                "assignment_mode": "disabled",
                "score_assignment_mode": "disabled",
                "score_match_rounds": 0,
                "score_enabled": False,
            }

        aux_loss_contrib = float(lambda_aux_eff) * aux_loss
        loss = main_loss + aux_loss_contrib

        loss_dict: Dict[str, Any] = {
            "loss": loss.detach(),
            "loss_main_total": main_loss.detach(),
            "loss_aux_total": aux_loss.detach(),
            "loss_aux_contrib": aux_loss_contrib.detach(),
            "lambda_aux": float(lambda_aux_eff),
            "aux_enabled": bool(aux_enabled),
            "aux_score_enabled": bool(self.aux_score_enabled),
            "hdetr_repeat_k": int(self.aux_matcher.repeat_k),
            "score_match_rounds": int(
                self.score_assigner.rounds
            ),
            "score_quality_gamma": float(
                self.score_assigner.quality_gamma
            ),
            "score_round_decay": float(
                self.score_assigner.round_decay
            ),
            "pairwise_ranking_enabled": bool(
                self.enable_pairwise_ranking
            ),
            "score_negative_iou_ignore_thr": float(
                score_negative_iou_ignore_thr
            ),
            "score_negative_iou_ignore_start": float(
                self.score_negative_iou_ignore_start
            ),
            "score_negative_iou_ignore_end": float(
                self.score_negative_iou_ignore_end
            ),
            "score_negative_iou_ignore_start_epoch": int(
                self.score_negative_iou_ignore_start_epoch
            ),
            "score_negative_iou_ignore_end_epoch": int(
                self.score_negative_iou_ignore_end_epoch
            ),
            "score_negative_iou_ignore_schedule": (
                self.score_negative_iou_ignore_schedule
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
            "loss_text_negative": self._detach_metric(
                main_metrics["loss_text_negative"]
            ),
            "loss_text_negative_contrib": self._detach_metric(
                main_metrics["loss_text_negative_contrib"]
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
            "loss_rank": self._detach_metric(
                main_metrics["loss_rank"]
            ),
            "loss_rank_raw": self._detach_metric(
                main_metrics["loss_rank_raw"]
            ),
            "loss_rank_contrib": self._detach_metric(
                main_metrics["loss_rank_contrib"]
            ),
            "matched": main_metrics["matched"],
            "score_pos_count": main_metrics["score_pos_count"],
            "hard_neg_count": main_metrics["hard_neg_count"],
            "negative_count": main_metrics["negative_count"],
            "ignored_negative_count": main_metrics[
                "ignored_negative_count"
            ],
            "selected_negative_fraction": main_metrics[
                "selected_negative_fraction"
            ],
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
            "lambda_rank": main_metrics["lambda_rank"],
            "lambda_rank_eff": main_metrics["lambda_rank_eff"],
            "pos_weight": main_metrics["pos_weight"],
            "quality_alpha": main_metrics["quality_alpha"],
            "rank_alpha": main_metrics["rank_alpha"],
        }

        for key, value in main_metrics.items():
            loss_dict[f"main_{key}"] = self._detach_metric(value)

        for key, value in aux_metrics.items():
            loss_dict[f"aux_{key}"] = self._detach_metric(value)

        return loss, loss_dict
