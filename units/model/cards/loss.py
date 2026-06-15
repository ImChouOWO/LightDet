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
        cost_score=1.0,
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

        flat_order = torch.argsort(
            cost.reshape(-1)
        ).detach().cpu().tolist()

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
                    torch.empty(
                        0,
                        dtype=torch.long,
                        device=pred_bbox.device,
                    ),
                    torch.empty(
                        0,
                        dtype=torch.long,
                        device=pred_bbox.device,
                    ),
                ))
                continue

            out_bbox = pred_bbox[b]
            out_score = pred_score[b]

            cost_bbox = torch.cdist(
                out_bbox,
                tgt_bbox,
                p=1,
            )

            cost_giou = -generalized_box_iou(
                out_bbox,
                tgt_bbox,
            )

            cost_score = -out_score[:, None]

            cost = (
                self.cost_bbox * cost_bbox
                +
                self.cost_giou * cost_giou
                +
                self.cost_score * cost_score
            )

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
        cost_score=1.0,
    ):
        super().__init__()

        self.matcher = GreedyMatcher(
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
            cost_score=cost_score,
        )

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

        B, N, _ = pred_bbox.shape

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

        total = max(total_matched, 1)

        loss_bbox = loss_bbox / total
        loss_giou = loss_giou / total

        pos_weight_tensor = pred_score_logit.new_tensor([float(pos_weight)])

        loss_score = F.binary_cross_entropy_with_logits(
            pred_score_logit,
            score_target,
            pos_weight=pos_weight_tensor,
            reduction="mean",
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
            "matched": float(total_matched),
            "lambda_bbox": float(lambda_bbox),
            "lambda_giou": float(lambda_giou),
            "lambda_score": float(lambda_score),
            "pos_weight": float(pos_weight),
        }

        return loss, loss_dict