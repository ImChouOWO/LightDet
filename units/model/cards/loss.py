from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

try:
    from units.model.tool.odvg_token_alignment import build_positive_token_maps
except ImportError:
    build_positive_token_maps = None


def clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def smoothstep(value: float) -> float:
    value = clamp01(value)
    return value * value * (3.0 - 2.0 * value)


def cosine_ramp(value: float) -> float:
    value = clamp01(value)
    return 0.5 - 0.5 * math.cos(math.pi * value)


def schedule_progress(
    current_epoch: Optional[int],
    start_epoch: int,
    duration: int,
    *,
    curve: str = "smoothstep",
) -> float:
    if current_epoch is None:
        return 1.0
    if int(current_epoch) < int(start_epoch):
        return 0.0
    progress = (
        float(current_epoch) - float(start_epoch) + 1.0
    ) / float(max(int(duration), 1))
    curve = str(curve).strip().lower()
    if curve == "linear":
        return clamp01(progress)
    if curve == "cosine":
        return cosine_ramp(progress)
    return smoothstep(progress)


def interpolate_value(start: float, end: float, alpha: float) -> float:
    alpha = clamp01(alpha)
    return float(start) + (float(end) - float(start)) * alpha


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (
        (boxes[..., 2] - boxes[..., 0]).clamp(min=0)
        * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
    )


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
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
    cover_lt = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    cover_rb = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    cover_wh = (cover_rb - cover_lt).clamp(min=0)
    cover_area = cover_wh[..., 0] * cover_wh[..., 1]
    return iou - (cover_area - union) / cover_area.clamp(min=1e-6)


def matched_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    if boxes1.shape != boxes2.shape or boxes1.ndim != 2 or boxes1.shape[-1] != 4:
        raise ValueError(
            "matched_box_iou expects equal [M,4] tensors, got "
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
    if boxes1.shape != boxes2.shape or boxes1.ndim != 2 or boxes1.shape[-1] != 4:
        raise ValueError(
            "matched_generalized_box_iou expects equal [M,4] tensors, got "
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


@dataclass(frozen=True)
class PackedTargets:
    boxes: torch.Tensor
    offsets: torch.Tensor
    counts: Tuple[int, ...]

    @classmethod
    def from_targets(
        cls,
        targets: Sequence[Mapping[str, Any]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "PackedTargets":
        rows: List[torch.Tensor] = []
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
            if count:
                rows.append(boxes)
        flat = (
            torch.cat(rows, dim=0)
            if rows
            else torch.empty((0, 4), device=device, dtype=dtype)
        )
        return cls(
            boxes=flat,
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
    batch_indices: torch.Tensor
    pred_indices: torch.Tensor
    gt_indices: torch.Tensor
    counts: Tuple[int, ...]
    mode: str = "generic"

    @property
    def num_matches(self) -> int:
        return int(self.pred_indices.numel())

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
        if sum(counts) == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return cls(empty, empty, empty, counts, mode)
        pred_indices = torch.cat(
            [pred.to(device=device, dtype=torch.long) for pred, _ in assignments if pred.numel()]
        )
        gt_indices = torch.cat(
            [gt.to(device=device, dtype=torch.long) for pred, gt in assignments if pred.numel()]
        )
        batch_indices = torch.repeat_interleave(
            torch.arange(len(counts), device=device, dtype=torch.long),
            torch.tensor(counts, device=device, dtype=torch.long),
        )
        return cls(batch_indices, pred_indices, gt_indices, counts, mode)


def _balanced_token_cost(
    token_logits: torch.Tensor,
    token_maps: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    negative_weight: float,
) -> torch.Tensor:
    """Return query-to-GT token BCE cost [Q,N]."""
    if token_logits.ndim != 2 or token_maps.ndim != 2 or valid_mask.ndim != 1:
        raise ValueError("token logits/maps/mask must be [Q,L], [N,L], [L]")
    if token_logits.shape[-1] != token_maps.shape[-1]:
        raise ValueError("token logit/map length mismatch")
    valid = valid_mask.to(device=token_logits.device, dtype=torch.bool)
    maps = token_maps.to(device=token_logits.device, dtype=torch.float32)
    maps = maps.clamp(min=0.0) * valid.float().unsqueeze(0)
    if maps.shape[0] == 0:
        return token_logits.new_zeros((token_logits.shape[0], 0))
    sums = maps.sum(dim=-1, keepdim=True)
    if bool((sums <= 0).any()):
        bad = torch.nonzero(sums.squeeze(-1) <= 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"GT token maps contain no valid positive token: {bad}")
    positive_weight = maps / sums.clamp_min(1e-7)
    positive_mask = maps > 0
    negative_mask = valid.unsqueeze(0) & ~positive_mask
    negative_norm = negative_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    negative_token_weight = negative_mask.float() / negative_norm.float()
    logits = token_logits.float()
    positive_nll = -torch.einsum(
        "ql,nl->qn",
        F.logsigmoid(logits),
        positive_weight,
    )
    negative_nll = -torch.einsum(
        "ql,nl->qn",
        F.logsigmoid(-logits),
        negative_token_weight,
    )
    return positive_nll + float(negative_weight) * negative_nll


class HungarianOneToOneMatcher:
    """DETR one-to-one matcher with quality and ODVG token alignment costs."""

    def __init__(
        self,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cost_score: float = 2.0,
        cost_alignment: float = 2.0,
        alignment_negative_weight: float = 0.25,
        score_cost_type: str = "focal",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        if not any(
            float(value) != 0.0
            for value in (cost_bbox, cost_giou, cost_score, cost_alignment)
        ):
            raise ValueError("At least one Hungarian cost must be non-zero")
        aliases = {
            "focal": "focal",
            "focal_loss": "focal",
            "prob": "probability",
            "probability": "probability",
            "score": "probability",
        }
        score_cost_type = str(score_cost_type).strip().lower()
        if score_cost_type not in aliases:
            raise ValueError("score_cost_type must be focal or probability")
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)
        self.cost_score = float(cost_score)
        self.cost_alignment = float(cost_alignment)
        self.alignment_negative_weight = max(0.0, float(alignment_negative_weight))
        self.score_cost_type = aliases[score_cost_type]
        self.focal_alpha = clamp01(focal_alpha)
        self.focal_gamma = max(0.0, float(focal_gamma))

    def _score_cost(self, score_prob: torch.Tensor) -> torch.Tensor:
        score_prob = score_prob.clamp(1e-6, 1.0 - 1e-6)
        if self.score_cost_type == "probability":
            return -score_prob
        negative_cost = (
            (1.0 - self.focal_alpha)
            * score_prob.pow(self.focal_gamma)
            * (-(1.0 - score_prob).log())
        )
        positive_cost = (
            self.focal_alpha
            * (1.0 - score_prob).pow(self.focal_gamma)
            * (-score_prob.log())
        )
        return positive_cost - negative_cost

    @torch.no_grad()
    def __call__(
        self,
        pred_bbox: torch.Tensor,
        pred_score_logit: torch.Tensor,
        targets: Sequence[Mapping[str, Any]],
        packed_targets: Optional[PackedTargets] = None,
        *,
        score_cost_alpha: float = 1.0,
        pred_token_alignment_logit: Optional[torch.Tensor] = None,
        positive_token_maps: Optional[Sequence[torch.Tensor]] = None,
        alignment_text_mask: Optional[torch.Tensor] = None,
        alignment_cost_alpha: float = 1.0,
    ) -> AssignmentResult:
        if pred_bbox.ndim != 3 or pred_bbox.shape[-1] != 4:
            raise ValueError(f"pred_bbox must be [B,Q,4], got {tuple(pred_bbox.shape)}")
        if pred_score_logit.ndim == 3 and pred_score_logit.shape[-1] == 1:
            score_logit = pred_score_logit.squeeze(-1)
        elif pred_score_logit.ndim == 2:
            score_logit = pred_score_logit
        else:
            raise ValueError("pred_score_logit must be [B,Q,1] or [B,Q]")
        batch_size, num_queries, _ = pred_bbox.shape
        if score_logit.shape != (batch_size, num_queries):
            raise ValueError("bbox/score shape mismatch")
        if len(targets) != batch_size:
            raise ValueError("targets batch size mismatch")
        if packed_targets is None:
            packed_targets = PackedTargets.from_targets(
                targets,
                device=pred_bbox.device,
                dtype=pred_bbox.dtype,
            )
        use_alignment = bool(
            self.cost_alignment != 0.0
            and float(alignment_cost_alpha) > 0.0
            and pred_token_alignment_logit is not None
            and positive_token_maps is not None
            and alignment_text_mask is not None
        )
        if use_alignment:
            if pred_token_alignment_logit.ndim != 3:
                raise ValueError("pred_token_alignment_logit must be [B,Q,L]")
            if pred_token_alignment_logit.shape[:2] != (batch_size, num_queries):
                raise ValueError("bbox/token alignment shape mismatch")
            if len(positive_token_maps) != batch_size:
                raise ValueError("positive_token_maps batch mismatch")
            if tuple(alignment_text_mask.shape) != (
                batch_size,
                pred_token_alignment_logit.shape[-1],
            ):
                raise ValueError("alignment_text_mask must be [B,L]")
        score_cost_alpha = clamp01(score_cost_alpha)
        alignment_cost_alpha = clamp01(alignment_cost_alpha)
        score_cost = self._score_cost(score_logit.detach().float().sigmoid())
        empty = torch.empty(0, dtype=torch.long, device=pred_bbox.device)
        rows: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for batch_index, target_count in enumerate(packed_targets.counts):
            target_count = int(target_count)
            if target_count == 0 or num_queries == 0:
                rows.append((empty, empty))
                continue
            start = int(packed_targets.offsets[batch_index].item())
            end = int(packed_targets.offsets[batch_index + 1].item())
            gt_boxes = packed_targets.boxes[start:end].detach().float()
            boxes = pred_bbox[batch_index].detach().float()
            cost = (
                self.cost_bbox * torch.cdist(boxes, gt_boxes, p=1)
                - self.cost_giou * generalized_box_iou(boxes, gt_boxes)
            )
            if self.cost_score != 0.0 and score_cost_alpha > 0.0:
                cost = cost + (
                    self.cost_score
                    * score_cost_alpha
                    * score_cost[batch_index, :, None]
                )
            if use_alignment:
                token_cost = _balanced_token_cost(
                    pred_token_alignment_logit[batch_index],
                    positive_token_maps[batch_index],
                    alignment_text_mask[batch_index],
                    negative_weight=self.alignment_negative_weight,
                )
                if token_cost.shape != cost.shape:
                    raise ValueError(
                        f"alignment cost shape {tuple(token_cost.shape)} does not "
                        f"match box cost {tuple(cost.shape)}"
                    )
                cost = cost + (
                    self.cost_alignment
                    * alignment_cost_alpha
                    * token_cost
                )
            if target_count == 1:
                pred_index = torch.argmin(cost[:, 0]).reshape(1)
                gt_index = torch.zeros(1, dtype=torch.long, device=pred_bbox.device)
            else:
                if linear_sum_assignment is None:
                    raise ImportError("Multi-GT matching requires scipy")
                pred_np, gt_np = linear_sum_assignment(cost.cpu().numpy())
                pred_index = torch.as_tensor(
                    pred_np,
                    device=pred_bbox.device,
                    dtype=torch.long,
                )
                gt_index = torch.as_tensor(
                    gt_np,
                    device=pred_bbox.device,
                    dtype=torch.long,
                )
            rows.append((pred_index, gt_index))
        mode = "odvg_hungarian" if use_alignment else "quality_hungarian"
        return AssignmentResult.from_per_batch(
            rows,
            device=pred_bbox.device,
            mode=mode,
        )


class HDETRRepeatedHungarianMatcher(HungarianOneToOneMatcher):
    """H-DETR auxiliary one-to-many matcher with repeated GT columns."""

    def __init__(
        self,
        *args: Any,
        max_positive_per_gt: int = 5,
        min_extra_positive_iou: float = 0.0,
        positive_ratio: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.repeat_k = max(1, int(max_positive_per_gt))
        self.min_extra_positive_iou = clamp01(min_extra_positive_iou)
        self.configured_positive_ratio = float(positive_ratio)

    @torch.no_grad()
    def __call__(
        self,
        pred_bbox: torch.Tensor,
        pred_score_logit: torch.Tensor,
        targets: Sequence[Mapping[str, Any]],
        packed_targets: Optional[PackedTargets] = None,
        **kwargs: Any,
    ) -> AssignmentResult:
        # First obtain the stable one-to-one assignment using all configured
        # costs, then add nearest unused queries for every GT. The extra matches
        # supervise only the auxiliary branch.
        primary = super().__call__(
            pred_bbox,
            pred_score_logit,
            targets,
            packed_targets,
            **kwargs,
        )
        if self.repeat_k <= 1 or pred_bbox.shape[1] <= 1:
            return primary
        if packed_targets is None:
            packed_targets = PackedTargets.from_targets(
                targets,
                device=pred_bbox.device,
                dtype=pred_bbox.dtype,
            )
        rows: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for batch_index, (primary_pred, primary_gt) in enumerate(primary):
            target_count = int(packed_targets.counts[batch_index])
            if target_count == 0:
                rows.append((primary_pred, primary_gt))
                continue
            start = int(packed_targets.offsets[batch_index].item())
            end = int(packed_targets.offsets[batch_index + 1].item())
            gt_boxes = packed_targets.boxes[start:end].float()
            boxes = pred_bbox[batch_index].detach().float()
            pair_iou = box_iou(boxes, gt_boxes)
            used = torch.zeros(boxes.shape[0], dtype=torch.bool, device=boxes.device)
            if primary_pred.numel():
                used[primary_pred] = True
            pred_rows = [primary_pred]
            gt_rows = [primary_gt]
            for gt_index in range(target_count):
                need = min(self.repeat_k - 1, int((~used).sum().item()))
                if need <= 0:
                    break
                candidate = pair_iou[:, gt_index].masked_fill(used, -1.0)
                values, indices = torch.topk(candidate, k=need, largest=True)
                keep = values >= self.min_extra_positive_iou
                indices = indices[keep]
                if indices.numel():
                    used[indices] = True
                    pred_rows.append(indices)
                    gt_rows.append(torch.full_like(indices, gt_index))
            rows.append((torch.cat(pred_rows), torch.cat(gt_rows)))
        return AssignmentResult.from_per_batch(
            rows,
            device=pred_bbox.device,
            mode="odvg_hdetr_repeated",
        )


OneToManyMatcher = HDETRRepeatedHungarianMatcher


@dataclass(frozen=True)
class HDETRScheduleState:
    quality_alpha: float
    main_matcher_score_alpha: float
    aux_matcher_score_alpha: float
    alignment_alpha: float
    aux_loss_factor: float
    text_negative_alpha: float
    duplicate_alpha: float
    hard_negative_alpha: float
    negative_iou_threshold: float


class GroundingLoss(nn.Module):
    """Unified LightDet ODVG loss and matcher implementation."""

    def __init__(
        self,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cost_score: float = 2.0,
        cost_alignment: float = 2.0,
        matcher_alignment_negative_weight: float = 0.25,
        matcher_score_cost_type: str = "focal",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        classification_type: str = "iou",
        ia_bce_alpha: float = 0.25,
        quality_min: float = 0.10,
        quality_max: float = 1.0,
        qfl_beta: float = 2.0,
        normalize_classification_by_num_gt: bool = True,
        negative_classification_weight: float = 0.25,
        negative_text_classification_weight: float = 1.0,
        score_negative_iou_ignore_thr: Optional[float] = 0.50,
        score_negative_iou_ignore_start: float = 0.55,
        score_negative_iou_ignore_end: float = 0.45,
        score_negative_iou_ignore_start_epoch: int = 5,
        score_negative_iou_ignore_end_epoch: int = 25,
        score_negative_iou_ignore_schedule: str = "cosine",
        duplicate_suppression_enabled: bool = True,
        duplicate_loss_weight: float = 0.10,
        duplicate_margin: float = 0.25,
        duplicate_background_weight: float = 0.05,
        duplicate_classification_weight: float = 0.25,
        duplicate_max_pairs: int = 128,
        duplicate_start_epoch: int = 5,
        duplicate_warmup_epoch: int = 5,
        hard_negative_mining_enabled: bool = True,
        hard_negative_loss_weight: float = 0.05,
        hard_negative_topk: int = 10,
        hard_negative_max_iou: float = 0.30,
        hard_negative_start_epoch: int = 10,
        hard_negative_warmup_epoch: int = 5,
        hard_negative_ratio: int = 5,
        text_negative_loss_weight: float = 0.20,
        text_negative_topk: int = 20,
        text_negative_hard_mix: float = 0.50,
        text_negative_start_epoch: int = 1,
        text_negative_warmup_epoch: int = 5,
        negative_text_as_empty_target: bool = True,
        max_query_loss_weight: float = 4.0,
        text_alignment_enabled: bool = True,
        text_alignment_loss_weight: float = 1.0,
        text_alignment_negative_weight: float = 0.25,
        text_alignment_unmatched_weight: float = 0.10,
        text_alignment_negative_text_weight: float = 1.0,
        text_alignment_focal_gamma: float = 2.0,
        text_alignment_ranking_weight: float = 0.25,
        text_alignment_ranking_margin: float = 0.15,
        text_alignment_start_epoch: int = 1,
        text_alignment_warmup_epoch: int = 5,
        aux_loss_weight: float = 0.50,
        aux_warmup_epoch: int = 3,
        aux_decay_start_ratio: float = 0.75,
        aux_min_factor: float = 0.25,
        aux_cost_score: Optional[float] = None,
        aux_cost_alignment: Optional[float] = None,
        aux_score_enabled: bool = True,
        max_positive_per_gt: int = 5,
        positive_ratio: float = 1.0,
        min_extra_positive_iou: float = 0.0,
        expand_cost_bbox: Optional[float] = None,
        expand_cost_giou: Optional[float] = None,
        quality_warmup_epoch: int = 10,
        matcher_score_start_epoch: int = 5,
        matcher_score_warmup_epoch: int = 12,
        matcher_score_alpha_min: float = 0.0,
        aux_matcher_score_start_epoch: int = 5,
        aux_matcher_score_warmup_epoch: int = 10,
        query_refinement_identity_weight: float = 0.0,
        query_refinement_start_epoch: int = 1,
        query_refinement_warmup_epoch: int = 5,
        query_refinement_detach_localization: bool = True,
        enable_pairwise_ranking: bool = False,
        rank_margin: float = 0.1,
        rank_min_quality_gap: float = 0.1,
        rank_max_pairs: int = 512,
        rank_start_epoch: int = 5,
        rank_warmup_epoch: int = 12,
        rank_alpha_min: float = 0.0,
        rank_negative_iou_max: float = 0.20,
        **legacy_parameters: Any,
    ) -> None:
        super().__init__()
        aliases = {
            "ia_bce": "ia_bce",
            "iabce": "ia_bce",
            "align_detr": "ia_bce",
            "normalized_giou": "normalized_giou",
            "giou_aware": "normalized_giou",
            "rank_detr": "normalized_giou",
            "iou": "iou",
        }
        classification_type = str(classification_type).strip().lower()
        if classification_type not in aliases:
            raise ValueError(f"Unsupported classification_type: {classification_type}")
        self.classification_type = aliases[classification_type]
        self.ia_bce_alpha = clamp01(ia_bce_alpha)
        self.focal_alpha = clamp01(focal_alpha)
        self.focal_gamma = max(0.0, float(focal_gamma if focal_gamma is not None else qfl_beta))
        self.quality_min = clamp01(quality_min)
        self.quality_max = clamp01(quality_max)
        self.normalize_classification_by_num_gt = bool(normalize_classification_by_num_gt)
        self.negative_classification_weight = max(0.0, float(negative_classification_weight))
        self.negative_text_classification_weight = max(0.0, float(negative_text_classification_weight))
        self.negative_text_as_empty_target = bool(negative_text_as_empty_target)
        self.max_query_loss_weight = max(1.0, float(max_query_loss_weight))

        self.score_negative_iou_ignore_thr = clamp01(
            0.0 if score_negative_iou_ignore_thr is None else score_negative_iou_ignore_thr
        )
        self.score_negative_iou_ignore_start = clamp01(score_negative_iou_ignore_start)
        self.score_negative_iou_ignore_end = clamp01(score_negative_iou_ignore_end)
        self.score_negative_iou_ignore_start_epoch = max(1, int(score_negative_iou_ignore_start_epoch))
        self.score_negative_iou_ignore_end_epoch = max(
            self.score_negative_iou_ignore_start_epoch,
            int(score_negative_iou_ignore_end_epoch),
        )
        self.score_negative_iou_ignore_schedule = str(score_negative_iou_ignore_schedule)

        self.duplicate_suppression_enabled = bool(duplicate_suppression_enabled)
        self.duplicate_loss_weight = max(0.0, float(duplicate_loss_weight))
        self.duplicate_margin = max(0.0, float(duplicate_margin))
        self.duplicate_background_weight = max(0.0, float(duplicate_background_weight))
        self.duplicate_classification_weight = clamp01(duplicate_classification_weight)
        self.duplicate_max_pairs = max(1, int(duplicate_max_pairs))
        self.duplicate_start_epoch = max(1, int(duplicate_start_epoch))
        self.duplicate_warmup_epoch = max(1, int(duplicate_warmup_epoch))

        self.hard_negative_mining_enabled = bool(hard_negative_mining_enabled)
        self.hard_negative_loss_weight = max(0.0, float(hard_negative_loss_weight))
        self.hard_negative_topk = max(1, int(hard_negative_topk))
        self.hard_negative_max_iou = clamp01(hard_negative_max_iou)
        self.hard_negative_start_epoch = max(1, int(hard_negative_start_epoch))
        self.hard_negative_warmup_epoch = max(1, int(hard_negative_warmup_epoch))
        self.hard_negative_ratio = max(1, int(hard_negative_ratio))

        self.text_negative_loss_weight = max(0.0, float(text_negative_loss_weight))
        self.text_negative_topk = max(1, int(text_negative_topk))
        self.text_negative_hard_mix = clamp01(text_negative_hard_mix)
        self.text_negative_start_epoch = max(1, int(text_negative_start_epoch))
        self.text_negative_warmup_epoch = max(1, int(text_negative_warmup_epoch))

        self.text_alignment_enabled = bool(text_alignment_enabled)
        self.text_alignment_loss_weight = max(0.0, float(text_alignment_loss_weight))
        self.text_alignment_negative_weight = max(0.0, float(text_alignment_negative_weight))
        self.text_alignment_unmatched_weight = max(0.0, float(text_alignment_unmatched_weight))
        self.text_alignment_negative_text_weight = max(0.0, float(text_alignment_negative_text_weight))
        self.text_alignment_focal_gamma = max(0.0, float(text_alignment_focal_gamma))
        self.text_alignment_ranking_weight = max(0.0, float(text_alignment_ranking_weight))
        self.text_alignment_ranking_margin = max(0.0, float(text_alignment_ranking_margin))
        self.text_alignment_start_epoch = max(1, int(text_alignment_start_epoch))
        self.text_alignment_warmup_epoch = max(1, int(text_alignment_warmup_epoch))

        self.aux_loss_weight = max(0.0, float(aux_loss_weight))
        self.aux_warmup_epoch = max(1, int(aux_warmup_epoch))
        self.aux_decay_start_ratio = clamp01(aux_decay_start_ratio)
        self.aux_min_factor = clamp01(aux_min_factor)
        self.aux_score_enabled = bool(aux_score_enabled)

        self.default_quality_warmup_epoch = max(0, int(quality_warmup_epoch))
        self.default_matcher_score_start_epoch = max(1, int(matcher_score_start_epoch))
        self.default_matcher_score_warmup_epoch = max(1, int(matcher_score_warmup_epoch))
        self.default_matcher_score_alpha_min = clamp01(matcher_score_alpha_min)
        self.default_aux_matcher_score_start_epoch = max(1, int(aux_matcher_score_start_epoch))
        self.default_aux_matcher_score_warmup_epoch = max(1, int(aux_matcher_score_warmup_epoch))

        self.query_refinement_identity_weight = max(0.0, float(query_refinement_identity_weight))
        self.query_refinement_start_epoch = max(1, int(query_refinement_start_epoch))
        self.query_refinement_warmup_epoch = max(1, int(query_refinement_warmup_epoch))
        self.query_refinement_detach_localization = bool(query_refinement_detach_localization)

        self.enable_pairwise_ranking = bool(enable_pairwise_ranking)
        self.lambda_rank_default = 0.0
        self.rank_margin = max(0.0, float(rank_margin))
        self.rank_min_quality_gap = max(0.0, float(rank_min_quality_gap))
        self.rank_max_pairs = max(1, int(rank_max_pairs))
        self.rank_start_epoch = max(1, int(rank_start_epoch))
        self.rank_warmup_epoch = max(1, int(rank_warmup_epoch))
        self.rank_alpha_min = clamp01(rank_alpha_min)
        self.rank_negative_iou_max = clamp01(rank_negative_iou_max)
        self.legacy_parameters = dict(legacy_parameters)

        self.main_matcher = HungarianOneToOneMatcher(
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
            cost_score=cost_score,
            cost_alignment=cost_alignment,
            alignment_negative_weight=matcher_alignment_negative_weight,
            score_cost_type=matcher_score_cost_type,
            focal_alpha=self.focal_alpha,
            focal_gamma=self.focal_gamma,
        )
        self.aux_matcher = HDETRRepeatedHungarianMatcher(
            cost_bbox=cost_bbox if expand_cost_bbox is None else expand_cost_bbox,
            cost_giou=cost_giou if expand_cost_giou is None else expand_cost_giou,
            cost_score=cost_score if aux_cost_score is None else aux_cost_score,
            cost_alignment=(
                cost_alignment if aux_cost_alignment is None else aux_cost_alignment
            ),
            alignment_negative_weight=matcher_alignment_negative_weight,
            score_cost_type=matcher_score_cost_type,
            focal_alpha=self.focal_alpha,
            focal_gamma=self.focal_gamma,
            positive_ratio=positive_ratio,
            max_positive_per_gt=max_positive_per_gt,
            min_extra_positive_iou=min_extra_positive_iou,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GroundingLoss":
        loss_cfg = config.get("loss", config)
        matcher = dict(loss_cfg.get("matcher", {}))
        weight = dict(loss_cfg.get("weight", {}))
        classification = dict(loss_cfg.get("classification", {}))
        quality = dict(loss_cfg.get("quality", {}))
        alignment = dict(loss_cfg.get("text_alignment", {}))
        matcher_schedule = dict(loss_cfg.get("matcher_schedule", {}))
        sampling = dict(loss_cfg.get("score_sampling", {}))
        hybrid = dict(loss_cfg.get("hybrid", {}))
        text_negative = dict(loss_cfg.get("text_negative", {}))
        duplicate = dict(loss_cfg.get("duplicate_suppression", {}))
        hard_negative = dict(loss_cfg.get("hard_negative", {}))
        ranking = dict(loss_cfg.get("ranking", {}))
        refinement = dict(loss_cfg.get("query_refinement", {}))
        focal_gamma = matcher.get(
            "focal_gamma",
            classification.get("focal_gamma", quality.get("qfl_beta", 2.0)),
        )
        return cls(
            cost_bbox=matcher.get("cost_bbox", 5.0),
            cost_giou=matcher.get("cost_giou", 2.0),
            cost_score=matcher.get("cost_score", 2.0),
            cost_alignment=matcher.get("cost_alignment", 2.0),
            matcher_alignment_negative_weight=matcher.get(
                "alignment_negative_weight", 0.25
            ),
            matcher_score_cost_type=matcher.get("score_cost_type", "focal"),
            focal_alpha=matcher.get("focal_alpha", classification.get("focal_alpha", 0.25)),
            focal_gamma=focal_gamma,
            classification_type=classification.get("type", "iou"),
            ia_bce_alpha=classification.get("ia_bce_alpha", 0.25),
            quality_min=quality.get("quality_min", 0.10),
            quality_max=quality.get("quality_max", 1.0),
            qfl_beta=quality.get("qfl_beta", 2.0),
            normalize_classification_by_num_gt=classification.get("normalize_by_num_gt", True),
            negative_classification_weight=classification.get("negative_weight", 0.25),
            negative_text_classification_weight=text_negative.get("classification_weight", 1.0),
            score_negative_iou_ignore_thr=classification.get("negative_iou_ignore_thr", 0.50),
            score_negative_iou_ignore_start=classification.get("negative_iou_ignore_start", 0.55),
            score_negative_iou_ignore_end=classification.get("negative_iou_ignore_end", 0.45),
            score_negative_iou_ignore_start_epoch=classification.get("negative_iou_ignore_start_epoch", 5),
            score_negative_iou_ignore_end_epoch=classification.get("negative_iou_ignore_end_epoch", 25),
            score_negative_iou_ignore_schedule=classification.get("negative_iou_ignore_schedule", "cosine"),
            duplicate_suppression_enabled=duplicate.get("enabled", True),
            duplicate_loss_weight=duplicate.get("loss_weight", 0.10),
            duplicate_margin=duplicate.get("margin", 0.25),
            duplicate_background_weight=duplicate.get("background_weight", 0.05),
            duplicate_classification_weight=duplicate.get("classification_weight", 0.25),
            duplicate_max_pairs=duplicate.get("max_pairs", 128),
            duplicate_start_epoch=duplicate.get("start_epoch", 5),
            duplicate_warmup_epoch=duplicate.get("warmup_epoch", 5),
            hard_negative_mining_enabled=hard_negative.get("enabled", True),
            hard_negative_loss_weight=hard_negative.get("loss_weight", 0.05),
            hard_negative_topk=hard_negative.get("topk", 10),
            hard_negative_max_iou=hard_negative.get("max_iou", 0.30),
            hard_negative_start_epoch=hard_negative.get("start_epoch", 10),
            hard_negative_warmup_epoch=hard_negative.get("warmup_epoch", 5),
            hard_negative_ratio=sampling.get("hard_negative_ratio", 5),
            text_negative_loss_weight=text_negative.get("lambda_text_negative", 0.20),
            text_negative_topk=text_negative.get("text_negative_topk", 20),
            text_negative_hard_mix=text_negative.get("text_negative_hard_mix", 0.50),
            text_negative_start_epoch=text_negative.get("start_epoch", 1),
            text_negative_warmup_epoch=text_negative.get("warmup_epoch", 5),
            negative_text_as_empty_target=text_negative.get("as_empty_target", True),
            max_query_loss_weight=text_negative.get("max_query_loss_weight", 4.0),
            text_alignment_enabled=alignment.get("enabled", True),
            text_alignment_loss_weight=alignment.get("loss_weight", 1.0),
            text_alignment_negative_weight=alignment.get("negative_weight", 0.25),
            text_alignment_unmatched_weight=alignment.get("unmatched_weight", 0.10),
            text_alignment_negative_text_weight=alignment.get("negative_text_weight", 1.0),
            text_alignment_focal_gamma=alignment.get("focal_gamma", 2.0),
            text_alignment_ranking_weight=alignment.get("ranking_weight", 0.25),
            text_alignment_ranking_margin=alignment.get("ranking_margin", 0.15),
            text_alignment_start_epoch=alignment.get("start_epoch", 1),
            text_alignment_warmup_epoch=alignment.get("warmup_epoch", 5),
            aux_loss_weight=hybrid.get("aux_loss_weight", weight.get("aux", 0.50)),
            aux_warmup_epoch=hybrid.get("warmup_epoch", 3),
            aux_decay_start_ratio=hybrid.get("decay_start_ratio", 0.75),
            aux_min_factor=hybrid.get("min_factor", 0.25),
            aux_cost_score=hybrid.get("matcher_cost_score", matcher.get("cost_score", 2.0)),
            aux_cost_alignment=hybrid.get("matcher_cost_alignment", matcher.get("cost_alignment", 2.0)),
            aux_score_enabled=quality.get("aux_score_enabled", True),
            max_positive_per_gt=sampling.get("max_positive_per_gt", 5),
            positive_ratio=sampling.get("positive_ratio", 1.0),
            min_extra_positive_iou=sampling.get("min_extra_positive_iou", 0.0),
            expand_cost_bbox=sampling.get("expand_cost_bbox", matcher.get("cost_bbox", 5.0)),
            expand_cost_giou=sampling.get("expand_cost_giou", matcher.get("cost_giou", 2.0)),
            quality_warmup_epoch=quality.get("quality_warmup_epoch", 10),
            matcher_score_start_epoch=matcher_schedule.get("start_epoch", ranking.get("rank_start_epoch", 5)),
            matcher_score_warmup_epoch=matcher_schedule.get("warmup_epoch", ranking.get("rank_warmup_epoch", 12)),
            matcher_score_alpha_min=matcher_schedule.get("alpha_min", ranking.get("rank_alpha_min", 0.0)),
            aux_matcher_score_start_epoch=hybrid.get("matcher_score_start_epoch", matcher_schedule.get("start_epoch", 5)),
            aux_matcher_score_warmup_epoch=hybrid.get("matcher_score_warmup_epoch", matcher_schedule.get("warmup_epoch", 10)),
            query_refinement_identity_weight=refinement.get("identity_weight", 0.0),
            query_refinement_start_epoch=refinement.get("start_epoch", 1),
            query_refinement_warmup_epoch=refinement.get("warmup_epoch", 5),
            query_refinement_detach_localization=refinement.get("detach_localization", True),
            enable_pairwise_ranking=ranking.get("enabled", False),
            rank_margin=ranking.get("rank_margin", 0.1),
            rank_min_quality_gap=ranking.get("rank_min_quality_gap", 0.1),
            rank_max_pairs=ranking.get("rank_max_pairs", 512),
            rank_start_epoch=ranking.get("rank_start_epoch", 5),
            rank_warmup_epoch=ranking.get("rank_warmup_epoch", 12),
            rank_alpha_min=ranking.get("rank_alpha_min", 0.0),
            rank_negative_iou_max=ranking.get("rank_negative_iou_max", 0.20),
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
        quality_warmup_epoch = (
            self.default_quality_warmup_epoch
            if quality_warmup_epoch is None
            else int(quality_warmup_epoch)
        )
        rank_start_epoch = (
            self.default_matcher_score_start_epoch
            if rank_start_epoch is None
            else int(rank_start_epoch)
        )
        rank_warmup_epoch = (
            self.default_matcher_score_warmup_epoch
            if rank_warmup_epoch is None
            else int(rank_warmup_epoch)
        )
        rank_alpha_min = (
            self.default_matcher_score_alpha_min
            if rank_alpha_min is None
            else clamp01(rank_alpha_min)
        )
        if quality_alpha is None:
            if current_epoch is None or quality_warmup_epoch <= 0:
                quality_alpha = 1.0
            else:
                quality_alpha = clamp01(
                    float(current_epoch) / float(max(quality_warmup_epoch, 1))
                )
        if rank_alpha is None:
            if current_epoch is None:
                rank_alpha = 1.0
            elif int(current_epoch) < rank_start_epoch:
                rank_alpha = 0.0
            else:
                progress = (
                    float(current_epoch) - float(rank_start_epoch)
                ) / float(max(rank_warmup_epoch, 1))
                rank_alpha = rank_alpha_min + (1.0 - rank_alpha_min) * smoothstep(progress)
        return float(clamp01(quality_alpha)), float(clamp01(rank_alpha))

    def resolve_dynamic_schedule(
        self,
        *,
        current_epoch: Optional[int],
        total_epochs: Optional[int],
        quality_alpha: Optional[float],
        matcher_score_alpha: Optional[float],
        quality_warmup_epoch: Optional[int],
        matcher_score_start_epoch: Optional[int],
        matcher_score_warmup_epoch: Optional[int],
        matcher_score_alpha_min: Optional[float],
    ) -> HDETRScheduleState:
        quality_value, matcher_value = self.resolve_epoch_alpha(
            current_epoch=current_epoch,
            quality_alpha=quality_alpha,
            rank_alpha=matcher_score_alpha,
            quality_warmup_epoch=quality_warmup_epoch,
            rank_start_epoch=matcher_score_start_epoch,
            rank_warmup_epoch=matcher_score_warmup_epoch,
            rank_alpha_min=matcher_score_alpha_min,
        )
        aux_matcher_alpha = schedule_progress(
            current_epoch,
            self.default_aux_matcher_score_start_epoch,
            self.default_aux_matcher_score_warmup_epoch,
        )
        alignment_alpha = schedule_progress(
            current_epoch,
            self.text_alignment_start_epoch,
            self.text_alignment_warmup_epoch,
        )
        aux_warmup = schedule_progress(current_epoch, 1, self.aux_warmup_epoch)
        aux_decay = 0.0
        if current_epoch is not None and total_epochs is not None and int(total_epochs) > 1:
            decay_start = max(1, int(round(float(total_epochs) * self.aux_decay_start_ratio)))
            aux_decay = schedule_progress(
                current_epoch,
                decay_start,
                max(1, int(total_epochs) - decay_start + 1),
                curve="cosine",
            )
        threshold_alpha = schedule_progress(
            current_epoch,
            self.score_negative_iou_ignore_start_epoch,
            max(
                1,
                self.score_negative_iou_ignore_end_epoch
                - self.score_negative_iou_ignore_start_epoch
                + 1,
            ),
            curve=self.score_negative_iou_ignore_schedule,
        )
        return HDETRScheduleState(
            quality_alpha=quality_value,
            main_matcher_score_alpha=matcher_value,
            aux_matcher_score_alpha=aux_matcher_alpha,
            alignment_alpha=alignment_alpha,
            aux_loss_factor=aux_warmup * interpolate_value(1.0, self.aux_min_factor, aux_decay),
            text_negative_alpha=schedule_progress(
                current_epoch,
                self.text_negative_start_epoch,
                self.text_negative_warmup_epoch,
            ),
            duplicate_alpha=schedule_progress(
                current_epoch,
                self.duplicate_start_epoch,
                self.duplicate_warmup_epoch,
            ),
            hard_negative_alpha=schedule_progress(
                current_epoch,
                self.hard_negative_start_epoch,
                self.hard_negative_warmup_epoch,
            ),
            negative_iou_threshold=interpolate_value(
                self.score_negative_iou_ignore_start,
                self.score_negative_iou_ignore_end,
                threshold_alpha,
            ),
        )

    @staticmethod
    def _extract_metadata(
        tensor: torch.Tensor,
        explicit: Optional[torch.Tensor],
        name: str,
    ) -> Optional[torch.Tensor]:
        if explicit is not None:
            return explicit
        return getattr(tensor, name, None)

    def _resolve_token_supervision(
        self,
        pred_score_logit: torch.Tensor,
        targets: Sequence[Mapping[str, Any]],
        pred_token_alignment_logit: Optional[torch.Tensor],
        positive_token_maps: Optional[Sequence[torch.Tensor]],
        alignment_text_mask: Optional[torch.Tensor],
        token_offsets: Optional[torch.Tensor],
        captions: Optional[Sequence[str]],
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[torch.Tensor],
    ]:
        token_logits = self._extract_metadata(
            pred_score_logit,
            pred_token_alignment_logit,
            "_token_alignment_logits",
        )
        mask = self._extract_metadata(
            pred_score_logit,
            alignment_text_mask,
            "_alignment_text_mask",
        )
        offsets = self._extract_metadata(
            pred_score_logit,
            token_offsets,
            "_token_offsets",
        )
        if token_logits is None:
            return None, None, None
        if mask is None:
            mask = torch.ones(
                token_logits.shape[0],
                token_logits.shape[-1],
                dtype=torch.bool,
                device=token_logits.device,
            )
        else:
            mask = mask.to(device=token_logits.device, dtype=torch.bool)
        maps: Optional[List[torch.Tensor]] = None
        if positive_token_maps is not None:
            maps = [
                value.to(device=token_logits.device, dtype=torch.float32)
                if torch.is_tensor(value)
                else torch.as_tensor(value, device=token_logits.device, dtype=torch.float32)
                for value in positive_token_maps
            ]
        else:
            target_maps = [target.get("positive_token_maps") for target in targets]
            if all(value is not None for value in target_maps):
                maps = [
                    value.to(device=token_logits.device, dtype=torch.float32)
                    if torch.is_tensor(value)
                    else torch.as_tensor(value, device=token_logits.device, dtype=torch.float32)
                    for value in target_maps
                ]
            elif offsets is not None and build_positive_token_maps is not None:
                char_spans = [target.get("positive_char_spans", []) for target in targets]
                if captions is None:
                    captions = [str(target.get("caption", "")) for target in targets]
                    if not any(captions):
                        captions = None
                built = build_positive_token_maps(
                    token_offsets=offsets,
                    positive_char_spans=char_spans,
                    attention_mask=mask,
                    captions=captions,
                    strict=True,
                    normalize=True,
                )
                maps = built["positive_token_maps"]
        if maps is None:
            return token_logits, None, mask
        if len(maps) != token_logits.shape[0]:
            raise ValueError("positive token map batch mismatch")
        return token_logits, maps, mask

    @staticmethod
    def _prepare_text_negative_mask(
        pred_score_logit: torch.Tensor,
        text_negative_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = int(pred_score_logit.shape[0])
        if text_negative_mask is None:
            return torch.zeros(batch_size, device=pred_score_logit.device, dtype=torch.bool)
        return torch.as_tensor(
            text_negative_mask,
            device=pred_score_logit.device,
            dtype=torch.bool,
        ).reshape(batch_size)

    def _build_effective_targets(
        self,
        targets: Sequence[Mapping[str, Any]],
        text_negative_mask: torch.Tensor,
    ) -> List[dict]:
        effective: List[dict] = []
        for is_negative, target in zip(text_negative_mask.detach().cpu().tolist(), targets):
            row = dict(target)
            if is_negative and self.negative_text_as_empty_target:
                boxes = target.get("boxes")
                if torch.is_tensor(boxes):
                    row["boxes"] = boxes.reshape(-1, 4)[:0]
                else:
                    row["boxes"] = torch.empty((0, 4), dtype=torch.float32)
                row["positive_char_spans"] = []
            effective.append(row)
        return effective

    def _prepare_query_weights(
        self,
        pred_score_logit: torch.Tensor,
        query_loss_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = int(pred_score_logit.shape[0])
        if query_loss_weights is None:
            return pred_score_logit.new_ones((batch_size, 1, 1), dtype=torch.float32)
        weights = torch.as_tensor(
            query_loss_weights,
            device=pred_score_logit.device,
            dtype=torch.float32,
        ).reshape(-1)
        if weights.numel() != batch_size:
            raise ValueError("query_loss_weights batch mismatch")
        return weights.clamp(1.0, self.max_query_loss_weight).reshape(batch_size, 1, 1)

    def _branch_loss(
        self,
        *,
        pred_bbox: torch.Tensor,
        pred_score_logit: torch.Tensor,
        assignments: AssignmentResult,
        packed_targets: PackedTargets,
        lambda_bbox: float,
        lambda_giou: float,
        lambda_score: float,
        quality_alpha: float,
        query_loss_weights: Optional[torch.Tensor],
        text_negative_mask: torch.Tensor,
        duplicate_alpha: float,
        hard_negative_alpha: float,
        negative_iou_ignore_thr: float,
        enable_extra_negative_losses: bool,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        logits = pred_score_logit.float()
        if logits.ndim != 3 or logits.shape[-1] != 1:
            raise ValueError("pred_score_logit must be [B,Q,1]")
        target = torch.zeros_like(logits)
        positive_mask = torch.zeros_like(logits, dtype=torch.bool)
        matched_iou = logits.new_zeros((0,))
        matched_giou = logits.new_zeros((0,))
        if assignments.num_matches:
            batch_idx = assignments.batch_indices
            pred_idx = assignments.pred_indices
            global_gt = packed_targets.global_gt_indices(batch_idx, assignments.gt_indices)
            pred_boxes = pred_bbox[batch_idx, pred_idx].float()
            gt_boxes = packed_targets.boxes[global_gt].float()
            matched_iou = matched_box_iou(pred_boxes.detach(), gt_boxes).clamp(0, 1)
            matched_giou = matched_generalized_box_iou(pred_boxes.detach(), gt_boxes).clamp(-1, 1)
            if self.classification_type == "normalized_giou":
                final_quality = (matched_giou + 1.0) * 0.5
            elif self.classification_type == "ia_bce":
                probability = logits[batch_idx, pred_idx, 0].detach().sigmoid()
                final_quality = (
                    probability.pow(self.ia_bce_alpha)
                    * matched_iou.pow(1.0 - self.ia_bce_alpha)
                )
            else:
                final_quality = matched_iou
            final_quality = final_quality.clamp(self.quality_min, self.quality_max)
            warm = torch.full_like(final_quality, self.quality_max)
            positive_quality = (
                (1.0 - quality_alpha) * warm + quality_alpha * final_quality
            )
            target[batch_idx, pred_idx, 0] = positive_quality
            positive_mask[batch_idx, pred_idx, 0] = True
            loss_bbox = F.l1_loss(pred_boxes, gt_boxes, reduction="mean")
            loss_giou = (1.0 - matched_generalized_box_iou(pred_boxes, gt_boxes)).mean()
        else:
            positive_quality = logits.new_zeros((0,))
            loss_bbox = logits.new_zeros(())
            loss_giou = logits.new_zeros(())

        probability = logits.sigmoid()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        modulation = (target - probability).abs().pow(self.focal_gamma)
        element = bce * modulation
        query_weights = self._prepare_query_weights(pred_score_logit, query_loss_weights)
        negative_weight = torch.full_like(element, self.negative_classification_weight)
        if bool(text_negative_mask.any()):
            negative_weight[text_negative_mask] = self.negative_text_classification_weight
        class_weight = torch.where(positive_mask, torch.ones_like(element), negative_weight)
        weighted = element * query_weights * class_weight
        if self.normalize_classification_by_num_gt:
            positive_loss = (
                (element * query_weights)[positive_mask].mean()
                if bool(positive_mask.any())
                else logits.new_zeros(())
            )
            negative_mask = ~positive_mask
            negative_loss = (
                weighted[negative_mask].sum()
                / (query_weights.expand_as(element) * class_weight)[negative_mask].sum().clamp_min(1.0)
                if bool(negative_mask.any())
                else logits.new_zeros(())
            )
            loss_score = positive_loss + negative_loss
        else:
            loss_score = weighted.sum() / (query_weights.expand_as(element) * class_weight).sum().clamp_min(1.0)

        max_iou = logits.new_zeros(logits.shape)
        nearest_gt = torch.full_like(logits, -1, dtype=torch.long)
        for batch_index, count in enumerate(packed_targets.counts):
            if count == 0:
                continue
            start = int(packed_targets.offsets[batch_index].item())
            end = int(packed_targets.offsets[batch_index + 1].item())
            iou = box_iou(pred_bbox[batch_index].detach().float(), packed_targets.boxes[start:end].float())
            values, indices = iou.max(dim=-1)
            max_iou[batch_index, :, 0] = values
            nearest_gt[batch_index, :, 0] = indices
        duplicate_mask = (
            (~positive_mask)
            & (max_iou >= float(negative_iou_ignore_thr))
            & (nearest_gt >= 0)
        )
        loss_duplicate = logits.new_zeros(())
        duplicate_pairs = logits.new_zeros(())
        if (
            enable_extra_negative_losses
            and self.duplicate_suppression_enabled
            and duplicate_alpha > 0.0
            and bool(duplicate_mask.any())
            and assignments.num_matches
        ):
            pair_losses: List[torch.Tensor] = []
            for batch_index in range(pred_bbox.shape[0]):
                matched_for_row = assignments.pred_indices[
                    assignments.batch_indices == batch_index
                ]
                if matched_for_row.numel() == 0:
                    continue
                positive_logit = logits[batch_index, matched_for_row, 0].max()
                duplicate_logits = logits[batch_index, duplicate_mask[batch_index, :, 0], 0]
                if duplicate_logits.numel():
                    pair_losses.append(
                        F.relu(self.duplicate_margin + duplicate_logits - positive_logit)
                    )
            if pair_losses:
                all_pairs = torch.cat(pair_losses)
                if all_pairs.numel() > self.duplicate_max_pairs:
                    all_pairs = torch.topk(all_pairs, self.duplicate_max_pairs).values
                loss_duplicate = all_pairs.mean()
                duplicate_pairs = logits.new_tensor(float(all_pairs.numel()))

        hard_negative_mask = (
            (~positive_mask)
            & (~duplicate_mask)
            & (max_iou <= self.hard_negative_max_iou)
        )
        loss_hard_negative = logits.new_zeros(())
        hard_count = logits.new_zeros(())
        if (
            enable_extra_negative_losses
            and self.hard_negative_mining_enabled
            and hard_negative_alpha > 0.0
            and bool(hard_negative_mask.any())
        ):
            hard_values = F.softplus(logits[hard_negative_mask])
            k = min(self.hard_negative_topk * max(1, pred_bbox.shape[0]), hard_values.numel())
            hard_values = torch.topk(hard_values, k=k).values
            loss_hard_negative = hard_values.mean()
            hard_count = logits.new_tensor(float(k))

        total = (
            float(lambda_bbox) * loss_bbox
            + float(lambda_giou) * loss_giou
            + float(lambda_score) * loss_score
            + self.duplicate_loss_weight * duplicate_alpha * loss_duplicate
            + self.hard_negative_loss_weight * hard_negative_alpha * loss_hard_negative
        )
        return total, {
            "loss_total": total,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_score": loss_score,
            "loss_duplicate": loss_duplicate,
            "loss_duplicate_contrib": self.duplicate_loss_weight * duplicate_alpha * loss_duplicate,
            "duplicate_pair_count": duplicate_pairs,
            "duplicate_violation_fraction": (loss_duplicate > 0).float(),
            "duplicate_tp_score_mean": probability[positive_mask].mean() if bool(positive_mask.any()) else logits.new_zeros(()),
            "duplicate_score_mean": probability[duplicate_mask].mean() if bool(duplicate_mask.any()) else logits.new_zeros(()),
            "duplicate_score_gap_mean": logits.new_zeros(()),
            "loss_hard_negative": loss_hard_negative,
            "loss_hard_negative_contrib": self.hard_negative_loss_weight * hard_negative_alpha * loss_hard_negative,
            "hard_neg_count": hard_count,
            "hard_negative_score_mean": probability[hard_negative_mask].mean() if bool(hard_negative_mask.any()) else logits.new_zeros(()),
            "hard_negative_iou_mean": max_iou[hard_negative_mask].mean() if bool(hard_negative_mask.any()) else logits.new_zeros(()),
            "hard_negative_loss_mean": loss_hard_negative,
            "matched": float(assignments.num_matches),
            "score_pos_count": float(positive_mask.sum().item()),
            "negative_count": float((~positive_mask).sum().item()),
            "ignored_negative_count": float(duplicate_mask.sum().item()),
            "selected_negative_fraction": 1.0,
            "ignored_negative_score_mean": probability[duplicate_mask].mean() if bool(duplicate_mask.any()) else logits.new_zeros(()),
            "ignored_negative_iou_mean": max_iou[duplicate_mask].mean() if bool(duplicate_mask.any()) else logits.new_zeros(()),
            "matched_iou_mean": matched_iou.mean() if matched_iou.numel() else logits.new_zeros(()),
            "score_iou_mean": matched_iou.mean() if matched_iou.numel() else logits.new_zeros(()),
            "score_target_pos_mean": positive_quality.mean() if positive_quality.numel() else logits.new_zeros(()),
            "score_target_pos_min": positive_quality.min() if positive_quality.numel() else logits.new_zeros(()),
            "score_target_pos_max": positive_quality.max() if positive_quality.numel() else logits.new_zeros(()),
            "assignment_mode": assignments.mode,
            "classification_type": self.classification_type,
        }

    def _token_alignment_loss(
        self,
        *,
        token_logits: torch.Tensor,
        positive_token_maps: Sequence[torch.Tensor],
        valid_token_mask: torch.Tensor,
        assignments: AssignmentResult,
        text_negative_mask: torch.Tensor,
        query_loss_weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_queries, token_count = token_logits.shape
        target = torch.zeros_like(token_logits, dtype=torch.float32)
        weight = torch.zeros_like(token_logits, dtype=torch.float32)
        valid = valid_token_mask.to(device=token_logits.device, dtype=torch.bool)
        unmatched_query = torch.ones(
            (batch_size, num_queries),
            device=token_logits.device,
            dtype=torch.bool,
        )
        for batch_index in range(batch_size):
            row_valid = valid[batch_index]
            weight[batch_index, :, row_valid] = self.text_alignment_unmatched_weight
            if bool(text_negative_mask[batch_index].item()):
                weight[batch_index, :, row_valid] = self.text_alignment_negative_text_weight
        if assignments.num_matches:
            for batch_index, pred_index, gt_index in zip(
                assignments.batch_indices.tolist(),
                assignments.pred_indices.tolist(),
                assignments.gt_indices.tolist(),
            ):
                gt_map = positive_token_maps[batch_index][gt_index].to(
                    device=token_logits.device,
                    dtype=torch.float32,
                )
                if gt_map.numel() != token_count:
                    raise ValueError("positive token map length mismatch")
                positive = (gt_map > 0) & valid[batch_index]
                target[batch_index, pred_index, positive] = 1.0
                weight[batch_index, pred_index, valid[batch_index]] = self.text_alignment_negative_weight
                weight[batch_index, pred_index, positive] = 1.0
                unmatched_query[batch_index, pred_index] = False
        probability = token_logits.float().sigmoid()
        bce = F.binary_cross_entropy_with_logits(
            token_logits.float(),
            target,
            reduction="none",
        )
        modulation = (target - probability).abs().pow(self.text_alignment_focal_gamma)
        query_weight = self._prepare_query_weights(
            token_logits[..., :1],
            query_loss_weights,
        ).expand(-1, num_queries, token_count)
        combined = weight * query_weight
        loss = (bce * modulation * combined).sum() / combined.sum().clamp_min(1.0)

        rank_terms: List[torch.Tensor] = []
        positive_phrase_scores: List[torch.Tensor] = []
        negative_phrase_scores: List[torch.Tensor] = []
        if assignments.num_matches and num_queries > 1:
            for batch_index, pred_index, gt_index in zip(
                assignments.batch_indices.tolist(),
                assignments.pred_indices.tolist(),
                assignments.gt_indices.tolist(),
            ):
                phrase_map = positive_token_maps[batch_index][gt_index].to(
                    device=token_logits.device,
                    dtype=torch.float32,
                )
                phrase_map = phrase_map * valid[batch_index].float()
                phrase_map = phrase_map / phrase_map.sum().clamp_min(1e-7)
                phrase_scores = torch.einsum(
                    "ql,l->q",
                    token_logits[batch_index].float(),
                    phrase_map,
                )
                candidates = unmatched_query[batch_index]
                if not bool(candidates.any()):
                    continue
                positive_score = phrase_scores[pred_index]
                negative_score = phrase_scores[candidates].max()
                rank_terms.append(
                    F.relu(
                        self.text_alignment_ranking_margin
                        - positive_score
                        + negative_score
                    )
                )
                positive_phrase_scores.append(positive_score.sigmoid())
                negative_phrase_scores.append(negative_score.sigmoid())
        rank_loss = (
            torch.stack(rank_terms).mean()
            if rank_terms
            else token_logits.new_zeros(())
        )
        positive_token_mask = (target > 0.5) & (weight > 0)
        negative_token_mask = (target <= 0.5) & (weight > 0)
        zero = token_logits.new_zeros(())
        metrics = {
            "positive_score_mean": probability[positive_token_mask].mean() if bool(positive_token_mask.any()) else zero,
            "negative_score_mean": probability[negative_token_mask].mean() if bool(negative_token_mask.any()) else zero,
            "positive_top1_score": torch.stack(positive_phrase_scores).mean() if positive_phrase_scores else zero,
            "negative_text_top1_score": torch.stack(negative_phrase_scores).mean() if negative_phrase_scores else zero,
            "positive_negative_margin": (
                torch.stack(positive_phrase_scores).mean() - torch.stack(negative_phrase_scores).mean()
                if positive_phrase_scores and negative_phrase_scores
                else zero
            ),
            "positive_count": token_logits.new_tensor(float(positive_token_mask.sum().item())),
            "negative_count": token_logits.new_tensor(float(negative_token_mask.sum().item())),
            "ignored_count": token_logits.new_tensor(float((weight <= 0).sum().item())),
            "rank_pair_count": token_logits.new_tensor(float(len(rank_terms))),
        }
        return loss.to(token_logits.dtype), rank_loss.to(token_logits.dtype), metrics

    def query_refinement_identity_loss(
        self,
        pred_score_logit: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = pred_score_logit.new_zeros(())
        localization_query = getattr(pred_score_logit, "_localization_query_out", None)
        score_query = getattr(pred_score_logit, "_score_query_out", None)
        if localization_query is None or score_query is None:
            return zero, {"cosine_mean": zero, "feature_delta_mean": zero}
        target = (
            localization_query.detach()
            if self.query_refinement_detach_localization
            else localization_query
        )
        cosine = F.cosine_similarity(score_query.float(), target.float(), dim=-1)
        delta = (score_query.float() - target.float()).pow(2).mean().sqrt()
        return (1.0 - cosine).mean().to(pred_score_logit.dtype), {
            "cosine_mean": cosine.mean().to(pred_score_logit.dtype),
            "feature_delta_mean": delta.to(pred_score_logit.dtype),
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
        pred_token_alignment_logit: Optional[torch.Tensor] = None,
        positive_token_maps: Optional[Sequence[torch.Tensor]] = None,
        alignment_text_mask: Optional[torch.Tensor] = None,
        token_offsets: Optional[torch.Tensor] = None,
        captions: Optional[Sequence[str]] = None,
        aux_pred_quality_logit: Optional[torch.Tensor] = None,
        aux_pred_text_alignment_logit: Optional[torch.Tensor] = None,
        aux_pred_token_alignment_logit: Optional[torch.Tensor] = None,
        lambda_text_alignment: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        del pos_weight, aux_pos_weight, kwargs
        quality_logit = pred_quality_logit if pred_quality_logit is not None else pred_score_logit
        if quality_logit.ndim != 3 or quality_logit.shape[-1] != 1:
            raise ValueError("quality logit must be [B,Q,1]")
        token_candidate = pred_token_alignment_logit
        if token_candidate is None and pred_text_alignment_logit is not None:
            if pred_text_alignment_logit.ndim == 3 and pred_text_alignment_logit.shape[-1] != 1:
                token_candidate = pred_text_alignment_logit
        token_logits, token_maps, valid_token_mask = self._resolve_token_supervision(
            quality_logit,
            targets,
            token_candidate,
            positive_token_maps,
            alignment_text_mask,
            token_offsets,
            captions,
        )
        negative_mask = self._prepare_text_negative_mask(quality_logit, text_negative_mask)
        effective_targets = self._build_effective_targets(targets, negative_mask)
        packed = PackedTargets.from_targets(
            effective_targets,
            device=pred_bbox.device,
            dtype=pred_bbox.dtype,
        )
        schedule = self.resolve_dynamic_schedule(
            current_epoch=current_epoch,
            total_epochs=total_epochs,
            quality_alpha=quality_alpha,
            matcher_score_alpha=rank_alpha,
            quality_warmup_epoch=quality_warmup_epoch,
            matcher_score_start_epoch=rank_start_epoch,
            matcher_score_warmup_epoch=rank_warmup_epoch,
            matcher_score_alpha_min=rank_alpha_min,
        )
        use_token_matching = bool(
            self.text_alignment_enabled
            and token_logits is not None
            and token_maps is not None
            and valid_token_mask is not None
            and schedule.alignment_alpha > 0.0
        )
        assignments = self.main_matcher(
            pred_bbox=pred_bbox,
            pred_score_logit=quality_logit,
            targets=effective_targets,
            packed_targets=packed,
            score_cost_alpha=schedule.main_matcher_score_alpha,
            pred_token_alignment_logit=token_logits if use_token_matching else None,
            positive_token_maps=token_maps if use_token_matching else None,
            alignment_text_mask=valid_token_mask if use_token_matching else None,
            alignment_cost_alpha=schedule.alignment_alpha,
        )
        main_loss, main_metrics = self._branch_loss(
            pred_bbox=pred_bbox,
            pred_score_logit=quality_logit,
            assignments=assignments,
            packed_targets=packed,
            lambda_bbox=lambda_bbox,
            lambda_giou=lambda_giou,
            lambda_score=lambda_score,
            quality_alpha=schedule.quality_alpha,
            query_loss_weights=query_loss_weights,
            text_negative_mask=negative_mask,
            duplicate_alpha=schedule.duplicate_alpha,
            hard_negative_alpha=schedule.hard_negative_alpha,
            negative_iou_ignore_thr=schedule.negative_iou_threshold,
            enable_extra_negative_losses=True,
        )

        zero = pred_bbox.new_zeros(())
        alignment_loss = zero
        alignment_rank_loss = zero
        alignment_metrics = {
            "positive_score_mean": zero,
            "negative_score_mean": zero,
            "positive_top1_score": zero,
            "negative_text_top1_score": zero,
            "positive_negative_margin": zero,
            "positive_count": zero,
            "negative_count": zero,
            "ignored_count": zero,
            "rank_pair_count": zero,
        }
        if use_token_matching:
            alignment_loss, alignment_rank_loss, alignment_metrics = self._token_alignment_loss(
                token_logits=token_logits,
                positive_token_maps=token_maps,
                valid_token_mask=valid_token_mask,
                assignments=assignments,
                text_negative_mask=negative_mask,
                query_loss_weights=query_loss_weights,
            )
        alignment_weight = (
            self.text_alignment_loss_weight
            if lambda_text_alignment is None
            else max(0.0, float(lambda_text_alignment))
        )
        alignment_contrib = alignment_weight * schedule.alignment_alpha * alignment_loss
        alignment_rank_contrib = (
            self.text_alignment_ranking_weight
            * schedule.alignment_alpha
            * alignment_rank_loss
        )

        aux_loss = zero
        aux_metrics: Dict[str, Any] = {
            "loss_bbox": zero,
            "loss_giou": zero,
            "loss_score": zero,
            "matched": 0.0,
        }
        aux_token_candidate = aux_pred_token_alignment_logit
        if aux_token_candidate is None and aux_pred_text_alignment_logit is not None:
            if aux_pred_text_alignment_logit.ndim == 3 and aux_pred_text_alignment_logit.shape[-1] != 1:
                aux_token_candidate = aux_pred_text_alignment_logit
        if aux_pred_bbox is not None and aux_pred_score_logit is not None:
            aux_quality = aux_pred_quality_logit if aux_pred_quality_logit is not None else aux_pred_score_logit
            aux_token = self._extract_metadata(
                aux_quality,
                aux_token_candidate,
                "_token_alignment_logits",
            )
            aux_assignments = self.aux_matcher(
                pred_bbox=aux_pred_bbox,
                pred_score_logit=aux_quality,
                targets=effective_targets,
                packed_targets=packed,
                score_cost_alpha=schedule.aux_matcher_score_alpha,
                pred_token_alignment_logit=aux_token if use_token_matching else None,
                positive_token_maps=token_maps if use_token_matching else None,
                alignment_text_mask=valid_token_mask if use_token_matching else None,
                alignment_cost_alpha=schedule.alignment_alpha,
            )
            aux_loss, aux_metrics = self._branch_loss(
                pred_bbox=aux_pred_bbox,
                pred_score_logit=aux_quality,
                assignments=aux_assignments,
                packed_targets=packed,
                lambda_bbox=lambda_bbox if aux_lambda_bbox is None else aux_lambda_bbox,
                lambda_giou=lambda_giou if aux_lambda_giou is None else aux_lambda_giou,
                lambda_score=(lambda_score if aux_lambda_score is None else aux_lambda_score) if self.aux_score_enabled else 0.0,
                quality_alpha=schedule.quality_alpha,
                query_loss_weights=query_loss_weights,
                text_negative_mask=negative_mask,
                duplicate_alpha=0.0,
                hard_negative_alpha=0.0,
                negative_iou_ignore_thr=0.0,
                enable_extra_negative_losses=False,
            )
        elif (aux_pred_bbox is None) != (aux_pred_score_logit is None):
            raise ValueError("aux_pred_bbox and aux_pred_score_logit must be provided together")
        lambda_aux_eff = self.aux_loss_weight if lambda_aux is None else max(0.0, float(lambda_aux))
        aux_contrib = lambda_aux_eff * schedule.aux_loss_factor * aux_loss

        identity_alpha = schedule_progress(
            current_epoch,
            self.query_refinement_start_epoch,
            self.query_refinement_warmup_epoch,
        )
        identity_loss, identity_metrics = self.query_refinement_identity_loss(quality_logit)
        identity_contrib = self.query_refinement_identity_weight * identity_alpha * identity_loss

        total = (
            main_loss
            + aux_contrib
            + alignment_contrib
            + alignment_rank_contrib
            + identity_contrib
        )
        result: Dict[str, Any] = {
            "loss": total.detach(),
            "loss_main_total": (
                main_loss + alignment_contrib + alignment_rank_contrib + identity_contrib
            ).detach(),
            "loss_aux_total": aux_loss.detach(),
            "loss_aux_contrib": aux_contrib.detach(),
            "loss_bbox": main_metrics["loss_bbox"].detach(),
            "loss_giou": main_metrics["loss_giou"].detach(),
            "loss_score": main_metrics["loss_score"].detach(),
            "loss_quality": main_metrics["loss_score"].detach(),
            "loss_text_alignment": alignment_loss.detach(),
            "loss_text_alignment_contrib": alignment_contrib.detach(),
            "loss_text_alignment_rank": alignment_rank_loss.detach(),
            "loss_text_alignment_rank_contrib": alignment_rank_contrib.detach(),
            "loss_text_negative": alignment_loss.detach() if bool(negative_mask.any()) else zero,
            "loss_text_negative_contrib": alignment_contrib.detach() if bool(negative_mask.any()) else zero,
            "loss_rank": alignment_rank_contrib.detach(),
            "loss_rank_raw": alignment_rank_loss.detach(),
            "loss_rank_contrib": alignment_rank_contrib.detach(),
            "lambda_rank": float(lambda_rank),
            "lambda_rank_eff": float(self.text_alignment_ranking_weight * schedule.alignment_alpha),
            "rank_alpha": float(schedule.alignment_alpha),
            "quality_alpha": float(schedule.quality_alpha),
            "matcher_score_alpha": float(schedule.main_matcher_score_alpha),
            "matcher_alignment_alpha": float(schedule.alignment_alpha),
            "matcher_cost_score_effective": float(self.main_matcher.cost_score * schedule.main_matcher_score_alpha),
            "matcher_cost_alignment_effective": float(self.main_matcher.cost_alignment * schedule.alignment_alpha),
            "assignment_mode": assignments.mode,
            "score_assignment_mode": "hungarian_one_to_one",
            "classification_type": self.classification_type,
            "dense_score_assignment_enabled": False,
            "pairwise_ranking_enabled": False,
            "score_decoupled": True,
            "matcher_score_source": "quality_logit",
            "matcher_alignment_source": "token_alignment_logits",
            "final_score_source": "quality_until_phrase_pooling",
            "negative_text_regression_matches": 0.0,
            "localization_query_rows": float((~negative_mask).sum().item()),
            "text_negative_query_rows": float(negative_mask.sum().item()),
            "text_alignment_enabled": self.text_alignment_enabled,
            "text_alignment_active": use_token_matching,
            "text_alignment_alpha": float(schedule.alignment_alpha),
            "lambda_text_alignment": float(alignment_weight),
            "text_alignment_positive_score_mean": alignment_metrics["positive_score_mean"].detach(),
            "text_alignment_negative_score_mean": alignment_metrics["negative_score_mean"].detach(),
            "text_alignment_positive_top1_score": alignment_metrics["positive_top1_score"].detach(),
            "text_alignment_negative_top1_score": alignment_metrics["negative_text_top1_score"].detach(),
            "text_alignment_margin": alignment_metrics["positive_negative_margin"].detach(),
            "text_alignment_positive_count": alignment_metrics["positive_count"].detach(),
            "text_alignment_negative_count": alignment_metrics["negative_count"].detach(),
            "text_alignment_ignored_count": alignment_metrics["ignored_count"].detach(),
            "text_alignment_rank_pair_count": alignment_metrics["rank_pair_count"].detach(),
            "negative_query_top1_score": alignment_metrics["negative_text_top1_score"].detach(),
            "positive_query_top1_score": alignment_metrics["positive_top1_score"].detach(),
            "positive_negative_score_margin": alignment_metrics["positive_negative_margin"].detach(),
            "text_negative_count": float(negative_mask.sum().item()),
            "score_negative_iou_ignore_thr": float(schedule.negative_iou_threshold),
            "duplicate_suppression_enabled": self.duplicate_suppression_enabled,
            "hard_negative_mining_enabled": self.hard_negative_mining_enabled,
            "loss_query_refinement_identity": identity_loss.detach(),
            "loss_query_refinement_identity_contrib": identity_contrib.detach(),
            "query_refinement_enabled": bool(getattr(quality_logit, "_query_stage_separated", False)),
            "query_refinement_identity_weight": self.query_refinement_identity_weight,
            "query_refinement_alpha": float(identity_alpha),
            "query_refinement_cosine_mean": identity_metrics["cosine_mean"].detach(),
            "query_refinement_feature_delta_mean": identity_metrics["feature_delta_mean"].detach(),
        }
        for key, value in main_metrics.items():
            result.setdefault(key, value.detach() if torch.is_tensor(value) else value)
        result.update({
            "loss_duplicate": main_metrics["loss_duplicate"].detach(),
            "loss_duplicate_contrib": main_metrics["loss_duplicate_contrib"].detach(),
            "duplicate_pair_count": main_metrics["duplicate_pair_count"].detach(),
            "duplicate_violation_fraction": main_metrics["duplicate_violation_fraction"].detach(),
            "duplicate_tp_score_mean": main_metrics["duplicate_tp_score_mean"].detach(),
            "duplicate_score_mean": main_metrics["duplicate_score_mean"].detach(),
            "duplicate_score_gap_mean": main_metrics["duplicate_score_gap_mean"].detach(),
            "loss_hard_negative": main_metrics["loss_hard_negative"].detach(),
            "loss_hard_negative_contrib": main_metrics["loss_hard_negative_contrib"].detach(),
            "hard_neg_count": main_metrics["hard_neg_count"].detach(),
            "hard_negative_score_mean": main_metrics["hard_negative_score_mean"].detach(),
            "hard_negative_iou_mean": main_metrics["hard_negative_iou_mean"].detach(),
            "hard_negative_loss_mean": main_metrics["hard_negative_loss_mean"].detach(),
        })
        return total, result


class RankingGroundingLoss(GroundingLoss):
    """Compatibility name; ODVG token ranking is integrated in GroundingLoss."""


def build_grounding_loss_from_config(config: Dict[str, Any]) -> RankingGroundingLoss:
    return RankingGroundingLoss.from_config(config)


def grounding_loss_forward_kwargs_from_config(
    config: Dict[str, Any],
    *,
    current_epoch: int,
    total_epochs: int,
) -> Dict[str, Any]:
    loss_cfg = config.get("loss", config)
    weight = dict(loss_cfg.get("weight", {}))
    dynamic = bool(weight.get("dynamic", False))
    if dynamic:
        start_epoch = int(weight.get("start_epoch", 1))
        end_epoch = int(weight.get("end_epoch", max(start_epoch, total_epochs)))
        progress = clamp01(
            (float(current_epoch) - start_epoch)
            / float(max(end_epoch - start_epoch, 1))
        )
        if str(weight.get("schedule", "cosine")).lower() == "cosine":
            progress = cosine_ramp(progress)
        lambda_bbox = interpolate_value(
            weight.get("bbox_start", weight.get("bbox", 5.0)),
            weight.get("bbox_end", weight.get("bbox", 5.0)),
            progress,
        )
        lambda_giou = interpolate_value(
            weight.get("giou_start", weight.get("giou", 2.0)),
            weight.get("giou_end", weight.get("giou", 2.0)),
            progress,
        )
        lambda_score = interpolate_value(
            weight.get("score_start", weight.get("score", 1.0)),
            weight.get("score_end", weight.get("score", 1.0)),
            progress,
        )
    else:
        lambda_bbox = float(weight.get("bbox", 5.0))
        lambda_giou = float(weight.get("giou", 2.0))
        lambda_score = float(weight.get("score", 1.0))
    hybrid = dict(loss_cfg.get("hybrid", {}))
    ranking = dict(loss_cfg.get("ranking", {}))
    alignment = dict(loss_cfg.get("text_alignment", {}))
    return {
        "lambda_bbox": float(lambda_bbox),
        "lambda_giou": float(lambda_giou),
        "lambda_score": float(lambda_score),
        "lambda_aux": float(hybrid.get("aux_loss_weight", 0.50)),
        "lambda_rank": float(ranking.get("lambda_rank", 0.0)) if ranking.get("enabled", False) else 0.0,
        "lambda_text_alignment": float(alignment.get("loss_weight", 1.0)),
        "current_epoch": int(current_epoch),
        "total_epochs": int(total_epochs),
        "quality_warmup_epoch": int(loss_cfg.get("quality", {}).get("quality_warmup_epoch", 10)),
        "rank_start_epoch": int(loss_cfg.get("matcher_schedule", {}).get("start_epoch", 5)),
        "rank_warmup_epoch": int(loss_cfg.get("matcher_schedule", {}).get("warmup_epoch", 12)),
        "rank_alpha_min": float(loss_cfg.get("matcher_schedule", {}).get("alpha_min", 0.0)),
    }


__all__ = [
    "AssignmentResult",
    "GroundingLoss",
    "HDETRRepeatedHungarianMatcher",
    "HungarianOneToOneMatcher",
    "OneToManyMatcher",
    "PackedTargets",
    "RankingGroundingLoss",
    "box_area",
    "box_iou",
    "build_grounding_loss_from_config",
    "clamp01",
    "generalized_box_iou",
    "grounding_loss_forward_kwargs_from_config",
    "interpolate_value",
    "matched_box_iou",
    "matched_generalized_box_iou",
    "schedule_progress",
]
