import torch
import torch.nn as nn
import torch.nn.functional as F


def cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=-1)


def compute_iou(pred_bbox, gt_bbox, eps=1e-7):
    B, N, _ = pred_bbox.shape

    gt_bbox = gt_bbox.unsqueeze(1).expand(-1, N, -1)

    pred_xyxy = cxcywh_to_xyxy(pred_bbox)
    gt_xyxy = cxcywh_to_xyxy(gt_bbox)

    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_xyxy.unbind(-1)

    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    pred_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    gt_area = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)

    union = pred_area + gt_area - inter_area + eps

    return inter_area / union


def compute_iou_multi_single(pred_bbox, gt_boxes, eps=1e-7):
    if gt_boxes.numel() == 0:
        return torch.zeros(
            pred_bbox.shape[0],
            0,
            device=pred_bbox.device,
            dtype=pred_bbox.dtype
        )

    pred_xyxy = cxcywh_to_xyxy(pred_bbox)
    gt_xyxy = cxcywh_to_xyxy(gt_boxes)

    pred_xyxy = pred_xyxy.unsqueeze(1)
    gt_xyxy = gt_xyxy.unsqueeze(0)

    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_xyxy.unbind(-1)

    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    pred_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    gt_area = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)

    union = pred_area + gt_area - inter_area + eps

    return inter_area / union


def siou_loss(pred_bbox, gt_bbox, eps=1e-7):
    px, py, pw, ph = pred_bbox.unbind(-1)
    gx, gy, gw, gh = gt_bbox.unbind(-1)

    pred_xyxy = cxcywh_to_xyxy(pred_bbox)
    gt_xyxy = cxcywh_to_xyxy(gt_bbox)

    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_xyxy.unbind(-1)

    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    pred_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    gt_area = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)

    union = pred_area + gt_area - inter_area + eps
    iou = inter_area / union

    center_distance = (px - gx) ** 2 + (py - gy) ** 2
    shape_cost = (pw - gw) ** 2 + (ph - gh) ** 2

    loss = 1.0 - iou + center_distance + shape_cost

    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        prob = torch.sigmoid(logits)

        ce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        loss = alpha_t * ((1.0 - p_t) ** self.gamma) * ce_loss

        return loss.mean()


def contrastive_loss_multi(text_feat, bbox_feat, matched_indices, temperature=0.07):
    text_feat = F.normalize(text_feat, dim=-1)
    bbox_feat = F.normalize(bbox_feat, dim=-1)

    losses = []

    for b in range(bbox_feat.shape[0]):
        pos_idx = matched_indices[b]

        if pos_idx.numel() == 0:
            continue

        sim = torch.matmul(
            bbox_feat[b],
            text_feat[b].unsqueeze(-1)
        ).squeeze(-1)

        sim = sim / temperature

        target = torch.zeros_like(sim)
        target[pos_idx] = 1.0
        target = target / target.sum().clamp(min=1e-6)

        log_prob = F.log_softmax(sim, dim=0)
        loss = -(target * log_prob).sum()

        losses.append(loss)

    if len(losses) == 0:
        return bbox_feat.sum() * 0.0

    return torch.stack(losses).mean()


def compute_total_loss(
    outputs,
    gt_boxes_per_image,
    text_feat,
    lambda_bbox=1.0,
    lambda_score=1.0,
    lambda_text=0.5,
    positive_ratio=0.05
):
    pred_bbox = outputs["bbox"]
    score_logits = outputs["score"]
    bbox_feat = outputs["att_out"]

    device = pred_bbox.device

    text_feat = text_feat.to(device).float()

    B, N, _ = pred_bbox.shape

    target_score = torch.zeros_like(score_logits)

    bbox_losses = []
    matched_indices_for_text = []
    max_iou_list = []
    best_gt_index_list = []

    for b in range(B):
        gt_boxes = gt_boxes_per_image[b].to(device).float()

        if gt_boxes.numel() == 0:
            matched_indices_for_text.append(
                torch.empty(0, device=device, dtype=torch.long)
            )
            max_iou_list.append(torch.zeros(N, device=device, dtype=pred_bbox.dtype))
            best_gt_index_list.append(torch.zeros(N, device=device, dtype=torch.long))
            continue

        iou_matrix = compute_iou_multi_single(
            pred_bbox[b],
            gt_boxes
        )

        max_iou, best_gt_idx = iou_matrix.max(dim=1)

        max_iou_list.append(max_iou)
        best_gt_index_list.append(best_gt_idx)

        top_k = max(1, int(N * positive_ratio))
        topk_iou, topk_idx = max_iou.detach().topk(k=top_k, dim=0)

        soft_values = topk_iou.pow(2)
        soft_values = soft_values / soft_values.max().clamp(min=1e-6)
        soft_values = soft_values.to(dtype=target_score.dtype)

        target_score[b, topk_idx, 0] = soft_values

        matched_for_text = []

        for gt_idx in range(gt_boxes.shape[0]):
            proposal_for_gt = iou_matrix[:, gt_idx].argmax()
            matched_for_text.append(proposal_for_gt)

            pred_box = pred_bbox[b, proposal_for_gt].unsqueeze(0)
            gt_box = gt_boxes[gt_idx].unsqueeze(0)

            bbox_loss = (
                siou_loss(pred_box, gt_box)
                + F.l1_loss(pred_box, gt_box)
            )

            bbox_losses.append(bbox_loss)

        matched_for_text = torch.stack(matched_for_text).unique()
        matched_indices_for_text.append(matched_for_text)

    if len(bbox_losses) > 0:
        loss_bbox = torch.stack(bbox_losses).mean()
    else:
        loss_bbox = pred_bbox.sum() * 0.0

    loss_score = FocalLoss()(score_logits, target_score)

    loss_text = contrastive_loss_multi(
        text_feat=text_feat,
        bbox_feat=bbox_feat,
        matched_indices=matched_indices_for_text
    )

    total_loss = (
        lambda_bbox * loss_bbox
        + lambda_score * loss_score
        + lambda_text * loss_text
    )

    max_iou_tensor = torch.stack(max_iou_list, dim=0)

    return total_loss, {
        "loss_bbox": loss_bbox.detach(),
        "loss_score": loss_score.detach(),
        "loss_text": loss_text.detach(),
        "iou": max_iou_tensor.detach(),
        "target_score_mean": target_score.detach().mean(),
        "positive_ratio": torch.tensor(positive_ratio, device=device),
    }