import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def clamp01(x):
    return min(max(float(x), 0.0), 1.0)


def smoothstep(x):
    """
    平滑 warmup 曲線。

    x = 0 -> 0
    x = 1 -> 1

    用於 ranking loss，避免 ranking 太早強烈干擾 bbox / score 主目標。
    """
    x = clamp01(x)
    return x * x * (3.0 - 2.0 * x)


def box_area(box):
    return (
        (box[..., 2] - box[..., 0]).clamp(min=0)
        *
        (box[..., 3] - box[..., 1]).clamp(min=0)
    )


def box_iou(boxes1, boxes2):
    """
    boxes1: [N, 4], xyxy normalized or absolute
    boxes2: [M, 4], xyxy normalized or absolute
    return: [N, M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter

    return inter / union.clamp(min=1e-6)


def generalized_box_iou(boxes1, boxes2):
    """
    boxes1: [N, 4]
    boxes2: [M, 4]
    return: [N, M]
    """
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


def bounded_iou_quality(
    iou,
    iou_pos_thr=0.2,
    quality_min=0.3,
    quality_max=1.0,
):
    """
    將 IoU 映射成非二元 score target。

    IoU < iou_pos_thr:
        target = 0.0

    IoU >= iou_pos_thr:
        target 落在 [quality_min, quality_max]
    """
    quality = torch.zeros_like(iou)

    pos_mask = iou >= iou_pos_thr

    quality[pos_mask] = (
        quality_min
        +
        (quality_max - quality_min)
        *
        (iou[pos_mask] - iou_pos_thr)
        /
        max(1.0 - iou_pos_thr, 1e-6)
    )

    return quality.clamp(0.0, 1.0)


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
        """
        GPU greedy matching。

        原本版本：
            torch.argsort(...).detach().cpu().tolist()

        會造成 GPU -> CPU 同步，batch 大時會讓 GPU utilization 掉到很低。

        這版保留 greedy matching 行為，但全程維持在 cost.device。
        """
        num_pred, num_gt = cost.shape
        device = cost.device

        if num_pred == 0 or num_gt == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        max_match = min(num_pred, num_gt)

        cost_work = cost.float().clone()
        inf = torch.finfo(cost_work.dtype).max

        pred_indices = []
        gt_indices = []

        for _ in range(max_match):
            flat_idx = torch.argmin(cost_work.reshape(-1))

            pred_idx = torch.div(flat_idx, num_gt, rounding_mode="floor")
            gt_idx = flat_idx % num_gt

            pred_indices.append(pred_idx)
            gt_indices.append(gt_idx)

            cost_work[pred_idx, :] = inf
            cost_work[:, gt_idx] = inf

        if len(pred_indices) == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        return (
            torch.stack(pred_indices).long(),
            torch.stack(gt_indices).long(),
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
                non_blocking=True,
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

            pred_idx, gt_idx = self.greedy_match(cost.float())

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
        positive_ratio=0.05,
        max_positive_per_gt=2,

        aux_positive_label=0.7,

        expand_cost_bbox=5.0,
        expand_cost_giou=2.0,

        # quality-aware confidence
        iou_pos_thr=0.2,
        quality_min=0.3,
        quality_max=1.0,
        qfl_beta=2.0,

        # ranking loss
        rank_margin=0.1,
        rank_min_quality_gap=0.1,
        rank_max_pairs=512,
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

        self.iou_pos_thr = float(iou_pos_thr)
        self.quality_min = float(quality_min)
        self.quality_max = float(quality_max)
        self.qfl_beta = float(qfl_beta)

        self.rank_margin = float(rank_margin)
        self.rank_min_quality_gap = float(rank_min_quality_gap)
        self.rank_max_pairs = int(rank_max_pairs)

    def resolve_epoch_alpha(
        self,
        current_epoch=None,
        quality_alpha=None,
        rank_alpha=None,
        quality_warmup_epoch=20,
        rank_start_epoch=15,
        rank_warmup_epoch=30,
        rank_alpha_min=1e-4,
    ):
        """
        quality_alpha:
            score target 從 objectness target 緩慢轉成 IoU quality target。

        rank_alpha:
            ranking loss 從 0 緩慢成長到 1。

        注意：
            lambda_rank 是上限。
            實際 ranking 權重是：

                lambda_rank_eff = lambda_rank * rank_alpha
        """
        if quality_alpha is None:
            if current_epoch is None:
                quality_alpha = 1.0
            else:
                quality_alpha = min(
                    1.0,
                    max(
                        0.0,
                        float(current_epoch) / max(float(quality_warmup_epoch), 1.0),
                    ),
                )

        if rank_alpha is None:
            if current_epoch is None:
                rank_alpha = 1.0
            else:
                if float(current_epoch) < float(rank_start_epoch):
                    rank_alpha = 0.0
                else:
                    t = (
                        float(current_epoch) - float(rank_start_epoch)
                    ) / max(float(rank_warmup_epoch), 1.0)

                    curve = smoothstep(t)

                    rank_alpha = (
                        float(rank_alpha_min)
                        +
                        (1.0 - float(rank_alpha_min)) * curve
                    )

                    rank_alpha = min(max(rank_alpha, 0.0), 1.0)

        return float(quality_alpha), float(rank_alpha)

    @torch.no_grad()
    def expand_score_targets(
        self,
        pred_bbox,
        targets,
        score_target,
        quality_alpha=1.0,
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
                non_blocking=True,
            )

            num_gt = int(tgt_bbox.shape[0])

            if num_gt == 0:
                continue

            # 這裡仍有 .item()，但相較原 matcher 的 .cpu().tolist() 成本小很多。
            # 若後續仍瓶頸，再把這段完全向量化。
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

            pred_bbox_b = pred_bbox[b].detach()

            cost_bbox = torch.cdist(
                pred_bbox_b,
                tgt_bbox,
                p=1,
            )

            giou = generalized_box_iou(
                pred_bbox_b,
                tgt_bbox,
            )

            cost = (
                self.expand_cost_bbox * cost_bbox
                -
                self.expand_cost_giou * giou
            )

            iou_mat = box_iou(
                pred_bbox_b,
                tgt_bbox,
            )

            max_iou = iou_mat.max(dim=1).values

            quality_all = bounded_iou_quality(
                max_iou,
                iou_pos_thr=self.iou_pos_thr,
                quality_min=self.quality_min,
                quality_max=self.quality_max,
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

                quality = quality_all[pred_idx]

                if quality <= 0.0:
                    continue

                quality_target = (
                    (1.0 - quality_alpha)
                    * score_target.new_tensor(self.aux_positive_label)
                    +
                    quality_alpha
                    * quality
                )

                score_target[b, pred_idx, 0] = torch.maximum(
                    score_target[b, pred_idx, 0],
                    quality_target,
                )

                current_pos += 1

                if current_pos >= positive_budget:
                    break

        return score_target

    def score_loss_quality_balanced(
        self,
        pred_score_logit,
        score_target,
        pos_weight=1.0,
        qfl_beta=None,
    ):
        """
        Quality Focal Loss + hard negative mining。
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

        pos_mask = score_target > 0.0
        neg_mask = score_target <= 0.0

        if pos_mask.any():
            loss_pos = loss_all[pos_mask].mean()
        else:
            loss_pos = pred_score_logit.new_tensor(0.0)

        neg_loss_all = loss_all[neg_mask]
        pos_count = int(pos_mask.sum().item())

        hard_neg_count = 0

        if neg_loss_all.numel() > 0:
            hard_neg_k = min(
                neg_loss_all.numel(),
                max(pos_count * self.hard_negative_ratio, self.hard_negative_ratio),
            )

            hard_neg_count = int(hard_neg_k)

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

        return (
            loss_score,
            loss_pos,
            loss_neg,
            float(pos_count),
            float(hard_neg_count),
        )

    def pairwise_quality_rank_loss(
        self,
        pred_score_logit,
        score_target,
        margin=None,
        min_quality_gap=None,
    ):
        """
        ranking loss：

        若 quality_i > quality_j，
        則希望 score_i > score_j。

        包含：
        1. positive vs positive ranking
        2. positive vs hard negative ranking
        """
        if margin is None:
            margin = self.rank_margin

        if min_quality_gap is None:
            min_quality_gap = self.rank_min_quality_gap

        score = pred_score_logit.sigmoid().reshape(-1)
        quality = score_target.reshape(-1)

        pos_idx = torch.nonzero(quality > 0.0, as_tuple=False).flatten()
        neg_idx = torch.nonzero(quality <= 0.0, as_tuple=False).flatten()

        losses = []

        # positive vs positive
        if pos_idx.numel() >= 2:
            q = quality[pos_idx]
            s = score[pos_idx]

            qi = q[:, None]
            qj = q[None, :]

            si = s[:, None]
            sj = s[None, :]

            rank_mask = qi > (qj + float(min_quality_gap))

            if rank_mask.any():
                diff = si - sj
                loss_pp = F.relu(float(margin) - diff)[rank_mask]

                if loss_pp.numel() > self.rank_max_pairs:
                    perm = torch.randperm(
                        loss_pp.numel(),
                        device=loss_pp.device,
                    )[:self.rank_max_pairs]
                    loss_pp = loss_pp[perm]

                losses.append(loss_pp.mean())

        # positive vs hard negative
        if pos_idx.numel() > 0 and neg_idx.numel() > 0:
            pos_score = score[pos_idx]
            neg_score = score[neg_idx]

            hard_neg_k = min(
                neg_score.numel(),
                max(pos_idx.numel() * self.hard_negative_ratio, 1),
            )

            hard_neg_score = torch.topk(
                neg_score,
                k=hard_neg_k,
                largest=True,
            ).values

            diff = pos_score[:, None] - hard_neg_score[None, :]
            loss_pn = F.relu(float(margin) - diff).reshape(-1)

            if loss_pn.numel() > self.rank_max_pairs:
                perm = torch.randperm(
                    loss_pn.numel(),
                    device=loss_pn.device,
                )[:self.rank_max_pairs]
                loss_pn = loss_pn[perm]

            losses.append(loss_pn.mean())

        if len(losses) == 0:
            return pred_score_logit.new_tensor(0.0)

        return torch.stack(losses).mean()

    def forward(
        self,
        pred_bbox,
        pred_score_logit,
        targets,
        lambda_bbox=5.0,
        lambda_giou=2.0,
        lambda_score=1.0,
        pos_weight=1.0,

        # schedule
        current_epoch=None,
        quality_alpha=None,
        rank_alpha=None,

        quality_warmup_epoch=20,
        rank_start_epoch=15,
        rank_warmup_epoch=30,
        rank_alpha_min=1e-4,

        # lambda_rank 是 ranking 上限。
        # 實際 ranking 權重是 lambda_rank * rank_alpha。
        lambda_rank=0.15,
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
                non_blocking=True,
            )

            matched_pred = pred_bbox[b, pred_idx]
            matched_tgt = tgt_bbox[gt_idx]

            with torch.no_grad():
                iou = box_iou(
                    matched_pred.detach(),
                    matched_tgt,
                )

                matched_iou = torch.diag(iou).clamp(0.0, 1.0)

                quality = bounded_iou_quality(
                    matched_iou,
                    iou_pos_thr=self.iou_pos_thr,
                    quality_min=self.quality_min,
                    quality_max=self.quality_max,
                )

                quality_target = (
                    (1.0 - quality_alpha) * torch.ones_like(quality)
                    +
                    quality_alpha * quality
                )

            score_target[b, pred_idx, 0] = quality_target

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
            quality_alpha=quality_alpha,
        )

        total = max(total_matched, 1)

        loss_bbox = loss_bbox / total
        loss_giou = loss_giou / total

        (
            loss_score,
            loss_score_pos,
            loss_score_neg,
            score_pos_count,
            hard_neg_count,
        ) = self.score_loss_quality_balanced(
            pred_score_logit=pred_score_logit,
            score_target=score_target,
            pos_weight=pos_weight,
            qfl_beta=self.qfl_beta,
        )

        lambda_rank_eff = float(lambda_rank) * float(rank_alpha)

        if lambda_rank_eff <= 0.0:
            loss_rank_raw = pred_score_logit.new_tensor(0.0)
            loss_rank = pred_score_logit.new_tensor(0.0)
            loss_rank_contrib = pred_score_logit.new_tensor(0.0)
        else:
            loss_rank_raw = self.pairwise_quality_rank_loss(
                pred_score_logit=pred_score_logit,
                score_target=score_target,
                margin=self.rank_margin,
                min_quality_gap=self.rank_min_quality_gap,
            )

            loss_rank = float(rank_alpha) * loss_rank_raw
            loss_rank_contrib = float(lambda_rank) * loss_rank

        loss_main = (
            float(lambda_bbox) * loss_bbox
            +
            float(lambda_giou) * loss_giou
            +
            float(lambda_score) * loss_score
        )

        loss = loss_main + loss_rank_contrib

        pos_mask = score_target > 0.0

        if pos_mask.any():
            score_target_pos_mean = score_target[pos_mask].mean()
            score_target_pos_max = score_target[pos_mask].max()
            score_target_pos_min = score_target[pos_mask].min()
        else:
            score_target_pos_mean = pred_bbox.new_tensor(0.0)
            score_target_pos_max = pred_bbox.new_tensor(0.0)
            score_target_pos_min = pred_bbox.new_tensor(0.0)

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

            "matched": float(total_matched),
            "score_pos_count": float(score_pos_count),
            "hard_neg_count": float(hard_neg_count),

            "score_target_pos_mean": score_target_pos_mean.detach(),
            "score_target_pos_min": score_target_pos_min.detach(),
            "score_target_pos_max": score_target_pos_max.detach(),

            "lambda_bbox": float(lambda_bbox),
            "lambda_giou": float(lambda_giou),
            "lambda_score": float(lambda_score),

            # lambda_rank 是上限
            "lambda_rank": float(lambda_rank),

            # lambda_rank_eff 才是實際 ranking 權重
            "lambda_rank_eff": float(lambda_rank_eff),

            "pos_weight": float(pos_weight),
            "quality_alpha": float(quality_alpha),
            "rank_alpha": float(rank_alpha),
        }

        return loss, loss_dict