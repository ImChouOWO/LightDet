import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def box_area(box):
    return (
        (box[..., 2] - box[..., 0]).clamp(min=0)
        *
        (box[..., 3] - box[..., 1]).clamp(min=0)
    )


def generalized_box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    lt_c = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1]

    giou = iou - (area_c - union) / area_c.clamp(min=1e-6)

    return giou


class GreedyMatcher:
    def __init__(
        self,
        cost_bbox=5.0,
        cost_giou=2.0,
        cost_score=0.0,
    ):
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)
        self.cost_score = float(cost_score)

    @torch.no_grad()
    def greedy_match(self, cost):
        num_pred, num_gt = cost.shape
        device = cost.device

        if num_pred == 0 or num_gt == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        flat_order = torch.argsort(cost.reshape(-1)).detach().cpu().tolist()

        used_pred = set()
        used_gt = set()

        pred_indices = []
        gt_indices = []

        max_match = min(num_pred, num_gt)

        for flat_idx in flat_order:
            pred_idx = flat_idx // num_gt
            gt_idx = flat_idx % num_gt

            if pred_idx in used_pred:
                continue

            if gt_idx in used_gt:
                continue

            used_pred.add(pred_idx)
            used_gt.add(gt_idx)

            pred_indices.append(pred_idx)
            gt_indices.append(gt_idx)

            if len(pred_indices) >= max_match:
                break

        if len(pred_indices) == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        return (
            torch.tensor(pred_indices, dtype=torch.long, device=device),
            torch.tensor(gt_indices, dtype=torch.long, device=device),
        )

    @torch.no_grad()
    def __call__(
        self,
        pred_bbox,
        pred_score_logit,
        targets,
    ):
        B, N, _ = pred_bbox.shape
        indices = []

        pred_score = pred_score_logit.sigmoid().squeeze(-1)

        for b in range(B):
            tgt_bbox = targets[b]["boxes"].to(
                device=pred_bbox.device,
                dtype=pred_bbox.dtype,
            )

            if tgt_bbox.numel() == 0:
                indices.append((
                    torch.empty(0, dtype=torch.long, device=pred_bbox.device),
                    torch.empty(0, dtype=torch.long, device=pred_bbox.device),
                ))
                continue

            out_bbox = pred_bbox[b]

            cost_bbox = torch.cdist(
                out_bbox,
                tgt_bbox,
                p=1,
            )

            cost_giou = -generalized_box_iou(
                out_bbox,
                tgt_bbox,
            )

            cost = (
                self.cost_bbox * cost_bbox
                +
                self.cost_giou * cost_giou
            )

            if self.cost_score > 0:
                out_score = pred_score[b]
                cost_score = -out_score[:, None]
                cost = cost + self.cost_score * cost_score

            cost = cost.float()

            pred_idx, gt_idx = self.greedy_match(cost)

            indices.append((
                pred_idx,
                gt_idx,
            ))

        return indices


class GroundingLoss(nn.Module):
    def __init__(
        self,
        cost_bbox=5.0,
        cost_giou=2.0,
        cost_score=0.0,
        hard_negative_ratio=5,
        positive_ratio=0.2,
        max_positive_per_gt=5,
        aux_positive_label=0.7,
        expand_cost_bbox=5.0,
        expand_cost_giou=2.0,
    ):
        super().__init__()

        self.matcher = GreedyMatcher(
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
            cost_score=cost_score,
        )

        self.hard_negative_ratio = int(hard_negative_ratio)
        self.positive_ratio = float(positive_ratio)
        self.max_positive_per_gt = int(max_positive_per_gt)
        self.aux_positive_label = float(aux_positive_label)
        self.expand_cost_bbox = float(expand_cost_bbox)
        self.expand_cost_giou = float(expand_cost_giou)

    @torch.no_grad()
    def expand_score_targets(
        self,
        pred_bbox,
        targets,
        score_target,
    ):
        B, N, _ = pred_bbox.shape

        positive_budget = max(
            1,
            int(round(N * self.positive_ratio)),
        )

        positive_budget = min(positive_budget, N)

        for b in range(B):
            tgt_bbox = targets[b]["boxes"].to(
                device=pred_bbox.device,
                dtype=pred_bbox.dtype,
            )

            num_gt = int(tgt_bbox.shape[0])

            if num_gt == 0:
                continue

            current_pos = int((score_target[b, :, 0] > 0.0).sum().item())
            remaining_budget = positive_budget - current_pos

            if remaining_budget <= 0:
                continue

            k_per_gt = max(
                1,
                int(math.ceil(remaining_budget / max(num_gt, 1))),
            )

            k_per_gt = min(
                k_per_gt,
                self.max_positive_per_gt,
                N,
            )

            cost_bbox = torch.cdist(
                pred_bbox[b].detach(),
                tgt_bbox,
                p=1,
            )

            giou = generalized_box_iou(
                pred_bbox[b].detach(),
                tgt_bbox,
            )

            cost = (
                self.expand_cost_bbox * cost_bbox
                -
                self.expand_cost_giou * giou
            )

            candidate_pred = []
            candidate_cost = []

            for gt_i in range(num_gt):
                vals, top_idx = torch.topk(
                    cost[:, gt_i],
                    k=k_per_gt,
                    largest=False,
                )

                candidate_pred.append(top_idx)
                candidate_cost.append(vals)

            if len(candidate_pred) == 0:
                continue

            candidate_pred = torch.cat(candidate_pred, dim=0)
            candidate_cost = torch.cat(candidate_cost, dim=0)

            order = torch.argsort(candidate_cost)

            for idx in order:
                pred_idx = candidate_pred[idx]

                if score_target[b, pred_idx, 0] > 0.0:
                    continue

                score_target[b, pred_idx, 0] = torch.maximum(
                    score_target[b, pred_idx, 0],
                    score_target.new_tensor(self.aux_positive_label),
                )

                current_pos += 1

                if current_pos >= positive_budget:
                    break

        return score_target

    def score_loss_balanced(
        self,
        pred_score_logit,
        score_target,
        pos_weight=1.0,
    ):
        bce = F.binary_cross_entropy_with_logits(
            pred_score_logit,
            score_target,
            reduction="none",
        )

        pos_mask = score_target > 0.0
        neg_mask = score_target <= 0.0

        if pos_mask.any():
            loss_pos = bce[pos_mask].mean()
        else:
            loss_pos = pred_score_logit.new_tensor(0.0)

        neg_loss_all = bce[neg_mask]

        pos_count = int(pos_mask.sum().item())

        if neg_loss_all.numel() > 0:
            hard_neg_k = min(
                neg_loss_all.numel(),
                max(pos_count * self.hard_negative_ratio, 1),
            )

            loss_neg = torch.topk(
                neg_loss_all.reshape(-1),
                k=hard_neg_k,
                largest=True,
            ).values.mean()
        else:
            loss_neg = pred_score_logit.new_tensor(0.0)

        loss_score = (
            float(pos_weight) * loss_pos
            +
            loss_neg
        ) / (float(pos_weight) + 1.0)

        return loss_score, loss_pos, loss_neg, float(pos_count)

    def forward(
        self,
        pred_bbox,
        pred_score_logit,
        targets,
        lambda_bbox=5.0,
        lambda_giou=2.0,
        lambda_score=1.0,
        pos_weight=1.0,
    ):
        indices = self.matcher(
            pred_bbox=pred_bbox,
            pred_score_logit=pred_score_logit,
            targets=targets,
        )

        score_target = torch.zeros_like(pred_score_logit)

        loss_bbox = pred_bbox.new_tensor(0.0)
        loss_giou = pred_bbox.new_tensor(0.0)
        total_matched = 0

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if pred_idx.numel() == 0:
                continue

            tgt_bbox = targets[b]["boxes"].to(
                device=pred_bbox.device,
                dtype=pred_bbox.dtype,
            )

            matched_pred = pred_bbox[b, pred_idx]
            matched_tgt = tgt_bbox[gt_idx]

            score_target[b, pred_idx, 0] = 1.0

            loss_bbox = loss_bbox + F.l1_loss(
                matched_pred,
                matched_tgt,
                reduction="sum",
            )

            giou = generalized_box_iou(
                matched_pred,
                matched_tgt,
            )

            loss_giou = loss_giou + (1.0 - torch.diag(giou)).sum()

            total_matched += int(pred_idx.numel())

        score_target = self.expand_score_targets(
            pred_bbox=pred_bbox,
            targets=targets,
            score_target=score_target,
        )

        total = max(total_matched, 1)

        loss_bbox = loss_bbox / total
        loss_giou = loss_giou / total

        loss_score, loss_score_pos, loss_score_neg, score_pos_count = (
            self.score_loss_balanced(
                pred_score_logit=pred_score_logit,
                score_target=score_target,
                pos_weight=pos_weight,
            )
        )

        loss = (
            float(lambda_bbox) * loss_bbox
            +
            float(lambda_giou) * loss_giou
            +
            float(lambda_score) * loss_score
        )

        loss_dict = {
            "loss": loss.detach(),
            "loss_bbox": loss_bbox.detach(),
            "loss_giou": loss_giou.detach(),
            "loss_score": loss_score.detach(),
            "loss_score_pos": loss_score_pos.detach(),
            "loss_score_neg": loss_score_neg.detach(),
            "matched": float(total_matched),
            "score_pos_count": float(score_pos_count),
            "lambda_bbox": float(lambda_bbox),
            "lambda_giou": float(lambda_giou),
            "lambda_score": float(lambda_score),
            "pos_weight": float(pos_weight),
        }

        return loss, loss_dict