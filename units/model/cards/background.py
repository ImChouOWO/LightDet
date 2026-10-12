"""Use annotation evidence for background labels and detached scores for hard mining."""

from dataclasses import dataclass
import math

import torch
from torchvision.ops import box_iou


@dataclass(frozen=True)
class BackgroundPolicy:
    enabled: bool = True
    max_iou: float = 0.2
    max_ioa: float = 0.2
    top_k: int = 10
    negative_ratio: float = 3.0
    empty_top_k: int = 5
    start_epoch: int = 6
    warmup_epochs: int = 5
    quality_weight: float = 0.25
    alignment_weight: float = 0.25
    ranking_weight: float = 1.0

    def __post_init__(self):
        if not 0 <= self.max_iou < 0.5 or not 0 <= self.max_ioa < 1:
            raise ValueError("Background overlap thresholds must stay below foreground thresholds")
        if min(self.top_k, self.empty_top_k, self.start_epoch, self.warmup_epochs) < 1:
            raise ValueError("Background counts and schedule must be positive")
        if self.empty_top_k > self.top_k:
            raise ValueError("empty_top_k must not exceed top_k")
        if not math.isfinite(self.negative_ratio) or self.negative_ratio <= 0:
            raise ValueError("Background negative_ratio must be positive and finite")
        if any(not math.isfinite(w) or w < 0 for w in
               (self.quality_weight, self.alignment_weight, self.ranking_weight)):
            raise ValueError("Background weights must be nonnegative and finite")

    def scale(self, epoch):
        if not self.enabled:
            return 0.0
        if epoch is None:
            return 1.0
        return min(1.0, max(0.0, (epoch - self.start_epoch + 1) / self.warmup_epochs))


@torch.no_grad()
def background_candidates(boxes, gt, complete, policy):
    boxes, gt = boxes.detach().float(), gt.detach().to(boxes).float().reshape(-1, 4)
    valid = torch.isfinite(boxes).all(1) & (boxes[:, 2:] > boxes[:, :2]).all(1)
    if not policy.enabled:
        return torch.zeros_like(valid)
    if not len(gt):
        return valid
    overlap = box_iou(boxes, gt)
    lower = torch.maximum(boxes[:, None, :2], gt[None, :, :2])
    upper = torch.minimum(boxes[:, None, 2:], gt[None, :, 2:])
    intersection = (upper - lower).clamp_min(0).prod(2)
    box_area = (boxes[:, 2:] - boxes[:, :2]).prod(1).clamp_min(1e-8)
    gt_area = (gt[:, 2:] - gt[:, :2]).prod(1).clamp_min(1e-8)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    inside = ((centers[:, None] >= gt[None, :, :2]).all(2)
              & (centers[:, None] <= gt[None, :, 2:]).all(2)).any(1)
    # Protect object fragments and large boxes enclosing a GT despite a small IoU.
    return (valid & ~inside & (overlap.max(1).values <= policy.max_iou)
            & ((intersection / box_area[:, None]).max(1).values <= policy.max_ioa)
            & ((intersection / gt_area[None, :]).max(1).values <= policy.max_ioa))


@torch.no_grad()
def mine_background(candidates, scores, positive_counts, policy):
    squeeze = scores.ndim == 1
    scores = scores.detach().float().reshape(len(candidates), -1)
    counts = torch.as_tensor(positive_counts, device=scores.device).reshape(-1)
    if counts.numel() == 1:
        counts = counts.expand(scores.shape[1])
    if counts.numel() != scores.shape[1]:
        raise ValueError("Background counts must match score columns")
    budget = torch.where(counts > 0, torch.ceil(counts * policy.negative_ratio), policy.empty_top_k)
    budget = budget.long().clamp(max=policy.top_k)
    eligible = candidates[:, None] & torch.isfinite(scores)
    k = min(policy.top_k, scores.shape[0])
    values, indices = scores.masked_fill(~eligible, -torch.inf).topk(k, dim=0)
    selected = torch.isfinite(values) & (torch.arange(k, device=scores.device)[:, None] < budget)
    mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(0, indices, selected)
    return mask[:, 0] if squeeze else mask
