from types import SimpleNamespace
import os
import sys
import csv
import json
import time
import math
import copy
import random
import shutil
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UNITS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))

for path in [PROJECT_ROOT, UNITS_DIR, CURRENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from units.tool.card import VisionTextModel, Bert
from units.model.pipeline.data import build_dataloaders


mp.set_sharing_strategy("file_system")



# Basic utils
def set_seed(seed):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def fmt(x):
        if x >= 1e9:
            return f"{x / 1e9:.3f}B"
        if x >= 1e6:
            return f"{x / 1e6:.3f}M"
        if x >= 1e3:
            return f"{x / 1e3:.3f}K"
        return str(x)

    return fmt(total), fmt(trainable)


def get_rng_state_dict():
    rng_state = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python": random.getstate(),
        "numpy": None,
    }

    try:
        rng_state["numpy"] = np.random.get_state()
    except Exception:
        rng_state["numpy"] = None

    return rng_state


def restore_rng_state(rng_state):
    if not isinstance(rng_state, dict):
        return

    if rng_state.get("torch", None) is not None:
        torch.set_rng_state(rng_state["torch"])

    if torch.cuda.is_available() and rng_state.get("cuda", None) is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])

    if rng_state.get("python", None) is not None:
        random.setstate(rng_state["python"])

    if rng_state.get("numpy", None) is not None:
        try:
            np.random.set_state(rng_state["numpy"])
        except Exception:
            pass


def build_scaler(device):
    enabled = device.type == "cuda"

    try:
        return GradScaler(device.type, enabled=enabled)
    except TypeError:
        return GradScaler(enabled=enabled)


def get_amp_enabled(device, use_amp=True):
    return bool(use_amp and device.type == "cuda")



# EMA


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay

        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        ema_state = self.ema.state_dict()

        for k, ema_v in ema_state.items():
            model_v = model_state[k].detach()

            if ema_v.dtype.is_floating_point:
                ema_v.copy_(ema_v * self.decay + model_v * (1.0 - self.decay))
            else:
                ema_v.copy_(model_v)



# Optimizer / Scheduler


def build_optimizer(
    model,
    lr_vision=1e-4,
    lr_text=1e-5,
    lr_transformer=1e-4,
    lr_head=1e-4,
    weight_decay=1e-4,
):
    no_decay = [
        "bias",
        "LayerNorm.weight",
        "norm.weight",
        "bn.weight",
        "BatchNorm",
    ]

    param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "text_model.model" in name:
            lr = lr_text
        elif "text_model.proj" in name:
            lr = lr_text
        elif "bottle_net" in name or "img_model" in name:
            lr = lr_vision
        elif "transformer" in name:
            lr = lr_transformer
        elif "head" in name:
            lr = lr_head
        else:
            lr = lr_head

        decay = 0.0 if any(nd in name for nd in no_decay) else weight_decay

        param_groups.append({
            "params": [param],
            "lr": lr,
            "weight_decay": decay,
            "name": name,
        })

    return AdamW(param_groups)


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.05):
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.step_num = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self):
        self.step_num += 1

        if self.step_num <= self.warmup_steps:
            factor = self.step_num / max(1, self.warmup_steps)
        else:
            progress = (self.step_num - self.warmup_steps) / max(
                1,
                self.total_steps - self.warmup_steps
            )
            progress = min(max(progress, 0.0), 1.0)

            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]



# Dynamic training schedule


def get_loss_weights(epoch, total_epochs, args):
    progress = epoch / max(1, total_epochs)

    if not args.loss_dynamic:
        return (
            float(args.lambda_bbox),
            float(args.lambda_giou),
            float(args.lambda_score),
        )

    bbox_decay_until = max(1e-8, float(args.lambda_bbox_decay_until))
    bbox_ratio = min(progress / bbox_decay_until, 1.0)

    lambda_bbox = (
        float(args.lambda_bbox_start)
        +
        (
            float(args.lambda_bbox_end)
            -
            float(args.lambda_bbox_start)
        )
        * bbox_ratio
    )

    lambda_giou = float(args.lambda_giou)

    score_warm_until = max(1e-8, float(args.lambda_score_warm_until))
    score_ratio = min(progress / score_warm_until, 1.0)

    lambda_score = (
        float(args.lambda_score_start)
        +
        (
            float(args.lambda_score_end)
            -
            float(args.lambda_score_start)
        )
        * score_ratio
    )

    return lambda_bbox, lambda_giou, lambda_score


def get_pos_weight(epoch, total_epochs, args):
    progress = epoch / max(1, total_epochs)

    warm_until = max(1e-8, float(args.pos_weight_warm_until))
    ratio = min(progress / warm_until, 1.0)

    pos_weight = (
        float(args.min_pos_weight)
        +
        (
            float(args.max_pos_weight)
            -
            float(args.min_pos_weight)
        )
        * ratio
    )

    return pos_weight



# Loss: normalized xyxy + Hungarian matcher


def box_area(box):
    return (
        (box[..., 2] - box[..., 0]).clamp(min=0)
        *
        (box[..., 3] - box[..., 1]).clamp(min=0)
    )


def box_iou_xyxy(boxes1, boxes2, eps=1e-7):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter

    return inter / union.clamp(min=eps)


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


class HungarianMatcher:
    def __init__(self, cost_bbox=5.0, cost_giou=2.0, cost_score=1.0):
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_score = cost_score

    @torch.no_grad()
    def __call__(self, pred_bbox, pred_score_logit, targets):
        B, N, _ = pred_bbox.shape
        indices = []

        pred_score = pred_score_logit.sigmoid().squeeze(-1)

        for b in range(B):
            tgt_bbox = targets[b]["boxes"].to(
                device=pred_bbox.device,
                dtype=pred_bbox.dtype
            )

            if tgt_bbox.numel() == 0:
                indices.append((
                    torch.empty(0, dtype=torch.long, device=pred_bbox.device),
                    torch.empty(0, dtype=torch.long, device=pred_bbox.device),
                ))
                continue

            out_bbox = pred_bbox[b]
            out_score = pred_score[b]

            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
            cost_score = -out_score[:, None]

            cost = (
                self.cost_bbox * cost_bbox
                +
                self.cost_giou * cost_giou
                +
                self.cost_score * cost_score
            )

            pred_idx, gt_idx = linear_sum_assignment(cost.detach().cpu().numpy())

            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.long, device=pred_bbox.device),
                torch.as_tensor(gt_idx, dtype=torch.long, device=pred_bbox.device),
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

        self.matcher = HungarianMatcher(
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
        indices = self.matcher(pred_bbox, pred_score_logit, targets)

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
                dtype=pred_bbox.dtype
            )

            matched_pred = pred_bbox[b, pred_idx]
            matched_tgt = tgt_bbox[gt_idx]

            score_target[b, pred_idx, 0] = 1.0

            loss_bbox = loss_bbox + F.l1_loss(
                matched_pred,
                matched_tgt,
                reduction="sum"
            )

            giou = generalized_box_iou(matched_pred, matched_tgt)
            loss_giou = loss_giou + (1.0 - torch.diag(giou)).sum()

            total_matched += int(pred_idx.numel())

        total = max(total_matched, 1)

        loss_bbox = loss_bbox / total
        loss_giou = loss_giou / total

        pos_weight_tensor = pred_score_logit.new_tensor([pos_weight])

        loss_score = F.binary_cross_entropy_with_logits(
            pred_score_logit,
            score_target,
            pos_weight=pos_weight_tensor,
            reduction="mean"
        )

        loss = (
            lambda_bbox * loss_bbox
            +
            lambda_giou * loss_giou
            +
            lambda_score * loss_score
        )

        loss_dict = {
            "loss": loss.detach(),
            "loss_bbox": loss_bbox.detach(),
            "loss_giou": loss_giou.detach(),
            "loss_score": loss_score.detach(),
            "matched": float(total_matched),
        }

        return loss, loss_dict



# Batch helper


def move_targets_to_device(batch, device):
    """
    dataloader:
        batch["targets"] = [{"boxes": ..., ...}, ...]
    """

    if "targets" in batch:
        raw_targets = batch["targets"]
    else:
        raw_targets = []

        boxes_list = batch["target_boxes_per_image"]
        labels_list = batch.get("target_labels_per_image", None)

        for i, boxes in enumerate(boxes_list):
            target = {"boxes": boxes}

            if labels_list is not None:
                target["labels"] = labels_list[i]

            raw_targets.append(target)

    targets = []

    for t in raw_targets:
        nt = {}

        for k, v in t.items():
            if torch.is_tensor(v):
                nt[k] = v.to(device, non_blocking=True)
            else:
                nt[k] = v

        if "boxes" not in nt:
            raise KeyError("target must contain key: boxes")

        targets.append(nt)

    return targets


def get_score_logit(outputs):
    if "score_logit" in outputs:
        return outputs["score_logit"]

    if "score" in outputs:
        return outputs["score"]

    raise KeyError("Model output must contain score_logit")



# Train / Val loss


def train_one_epoch(
    model,
    ema,
    criterion,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    num_epochs,
    use_amp=True,
    grad_clip_norm=1.0,
    lambda_bbox=5.0,
    lambda_giou=2.0,
    lambda_score=1.0,
    pos_weight=1.0,
    log_interval=10,
    step_metrics_path=None,
):
    model.train()

    total_loss_sum = 0.0
    total_bbox_sum = 0.0
    total_giou_sum = 0.0
    total_score_sum = 0.0

    amp_enabled = get_amp_enabled(device, use_amp)

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch}/{num_epochs} [Train]",
        dynamic_ncols=True,
        leave=True
    )

    for step, batch in pbar:
        global_step = (epoch - 1) * len(train_loader) + step + 1

        images = batch["images"].to(device, non_blocking=True)
        query_texts = batch["query_texts"]
        targets = move_targets_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images, query_texts)

            pred_bbox = outputs["bbox"]
            pred_score_logit = get_score_logit(outputs)

            loss, loss_dict = criterion(
                pred_bbox=pred_bbox,
                pred_score_logit=pred_score_logit,
                targets=targets,
                lambda_bbox=lambda_bbox,
                lambda_giou=lambda_giou,
                lambda_score=lambda_score,
                pos_weight=pos_weight,
            )

        if amp_enabled:
            scaler.scale(loss).backward()

            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()

        scheduler.step()

        if ema is not None:
            ema.update(model)

        loss_item = float(loss.item())
        bbox_item = float(loss_dict["loss_bbox"].item())
        giou_item = float(loss_dict["loss_giou"].item())
        score_item = float(loss_dict["loss_score"].item())

        total_loss_sum += loss_item
        total_bbox_sum += bbox_item
        total_giou_sum += giou_item
        total_score_sum += score_item

        avg_loss = total_loss_sum / (step + 1)
        current_lr = scheduler.get_lr()[0]

        pbar.set_postfix({
            "lr": f"{current_lr:.2e}",
            "loss": f"{loss_item:.4f}",
            "avg": f"{avg_loss:.4f}",
            "bbox": f"{bbox_item:.4f}",
            "giou": f"{giou_item:.4f}",
            "score": f"{score_item:.4f}",
            "lb": f"{lambda_bbox:.2f}",
            "lg": f"{lambda_giou:.2f}",
            "ls": f"{lambda_score:.2f}",
            "pw": f"{pos_weight:.2f}",
        })

        if step_metrics_path is not None and (step + 1) % log_interval == 0:
            append_jsonl(step_metrics_path, {
                "type": "step",
                "time": time.time(),
                "epoch": epoch,
                "step": step + 1,
                "global_step": global_step,
                "lr": current_lr,
                "train_loss": loss_item,
                "train_loss_avg": avg_loss,
                "loss_bbox": bbox_item,
                "loss_giou": giou_item,
                "loss_score": score_item,
                "lambda_bbox": lambda_bbox,
                "lambda_giou": lambda_giou,
                "lambda_score": lambda_score,
                "pos_weight": pos_weight,
            })

    n = max(1, len(train_loader))

    return {
        "train_loss": total_loss_sum / n,
        "train_loss_bbox": total_bbox_sum / n,
        "train_loss_giou": total_giou_sum / n,
        "train_loss_score": total_score_sum / n,
    }


@torch.no_grad()
def validate_loss_one_epoch(
    model,
    criterion,
    val_loader,
    device,
    epoch,
    use_amp=True,
    lambda_bbox=5.0,
    lambda_giou=2.0,
    lambda_score=1.0,
    pos_weight=1.0,
):
    model.eval()

    total_loss_sum = 0.0
    total_bbox_sum = 0.0
    total_giou_sum = 0.0
    total_score_sum = 0.0

    amp_enabled = get_amp_enabled(device, use_amp)

    pbar = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        desc=f"Epoch {epoch} [Val Loss]",
        dynamic_ncols=True,
        leave=True
    )

    for step, batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        query_texts = batch["query_texts"]
        targets = move_targets_to_device(batch, device)

        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images, query_texts)

            pred_bbox = outputs["bbox"]
            pred_score_logit = get_score_logit(outputs)

            loss, loss_dict = criterion(
                pred_bbox=pred_bbox,
                pred_score_logit=pred_score_logit,
                targets=targets,
                lambda_bbox=lambda_bbox,
                lambda_giou=lambda_giou,
                lambda_score=lambda_score,
                pos_weight=pos_weight,
            )

        loss_item = float(loss.item())
        bbox_item = float(loss_dict["loss_bbox"].item())
        giou_item = float(loss_dict["loss_giou"].item())
        score_item = float(loss_dict["loss_score"].item())

        total_loss_sum += loss_item
        total_bbox_sum += bbox_item
        total_giou_sum += giou_item
        total_score_sum += score_item

        avg_loss = total_loss_sum / (step + 1)

        pbar.set_postfix({
            "val_loss": f"{loss_item:.4f}",
            "avg": f"{avg_loss:.4f}",
            "bbox": f"{bbox_item:.4f}",
            "giou": f"{giou_item:.4f}",
            "score": f"{score_item:.4f}",
        })

    n = max(1, len(val_loader))

    return {
        "val_loss": total_loss_sum / n,
        "val_loss_bbox": total_bbox_sum / n,
        "val_loss_giou": total_giou_sum / n,
        "val_loss_score": total_score_sum / n,
    }



# Detection metrics: query-conditioned binary detection


@torch.no_grad()
def select_predictions(
    boxes,
    scores,
    score_thr=0.25,
    top_k=20,
    nms_iou_thr=0.5,
    use_topk_fallback=True,
):
    """
    boxes:
        [N, 4], normalized xyxy

    scores:
        [N], sigmoid score
    """

    N = boxes.shape[0]

    if N == 0:
        return boxes.new_zeros((0, 4)), scores.new_zeros((0,))

    keep = scores >= score_thr

    if keep.sum() > 0:
        selected_boxes = boxes[keep]
        selected_scores = scores[keep]
    else:
        if not use_topk_fallback:
            return boxes.new_zeros((0, 4)), scores.new_zeros((0,))

        k = min(top_k, N)
        selected_scores, top_idx = scores.topk(k=k)
        selected_boxes = boxes[top_idx]

    if selected_scores.numel() > top_k:
        selected_scores, top_idx = selected_scores.topk(k=top_k)
        selected_boxes = selected_boxes[top_idx]

    if selected_boxes.numel() == 0:
        return selected_boxes, selected_scores

    keep_idx = nms(
        selected_boxes.float(),
        selected_scores.float(),
        iou_threshold=nms_iou_thr,
    )

    selected_boxes = selected_boxes[keep_idx]
    selected_scores = selected_scores[keep_idx]

    return selected_boxes, selected_scores


def compute_ap_from_pr(precision, recall):
    if precision.numel() == 0 or recall.numel() == 0:
        return 0.0

    mrec = torch.cat([
        torch.tensor([0.0], dtype=recall.dtype),
        recall,
        torch.tensor([1.0], dtype=recall.dtype),
    ])

    mpre = torch.cat([
        torch.tensor([0.0], dtype=precision.dtype),
        precision,
        torch.tensor([0.0], dtype=precision.dtype),
    ])

    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    return float(ap.item())


def evaluate_ap_at_iou(pred_records, gt_by_sample, iou_thr=0.5):
    """
    pred_records:
        list of {
            "sample_id": int,
            "score": float,
            "box": Tensor[4] CPU
        }

    gt_by_sample:
        list of Tensor[M, 4] CPU
    """

    num_gt = sum(int(gt.shape[0]) for gt in gt_by_sample)

    if num_gt == 0:
        return {
            "ap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0,
            "fp": 0,
            "num_gt": 0,
            "num_pred": len(pred_records),
        }

    pred_records = sorted(
        pred_records,
        key=lambda x: x["score"],
        reverse=True
    )

    matched = [
        torch.zeros((gt.shape[0],), dtype=torch.bool)
        for gt in gt_by_sample
    ]

    tp = []
    fp = []

    for rec in pred_records:
        sample_id = rec["sample_id"]
        pred_box = rec["box"].view(1, 4)
        gt_boxes = gt_by_sample[sample_id]

        if gt_boxes.numel() == 0:
            tp.append(0.0)
            fp.append(1.0)
            continue

        ious = box_iou_xyxy(pred_box, gt_boxes)[0]
        best_iou, best_idx = ious.max(dim=0)

        if best_iou.item() >= iou_thr and not matched[sample_id][best_idx]:
            matched[sample_id][best_idx] = True
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)

    if len(tp) == 0:
        return {
            "ap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0,
            "fp": 0,
            "num_gt": num_gt,
            "num_pred": 0,
        }

    tp = torch.tensor(tp, dtype=torch.float32)
    fp = torch.tensor(fp, dtype=torch.float32)

    cum_tp = torch.cumsum(tp, dim=0)
    cum_fp = torch.cumsum(fp, dim=0)

    recall = cum_tp / max(1, num_gt)
    precision = cum_tp / torch.clamp(cum_tp + cum_fp, min=1e-6)

    ap = compute_ap_from_pr(precision, recall)

    final_tp = int(cum_tp[-1].item())
    final_fp = int(cum_fp[-1].item())

    final_precision = final_tp / max(1, final_tp + final_fp)
    final_recall = final_tp / max(1, num_gt)

    return {
        "ap": ap,
        "precision": final_precision,
        "recall": final_recall,
        "tp": final_tp,
        "fp": final_fp,
        "num_gt": num_gt,
        "num_pred": len(pred_records),
    }


@torch.no_grad()
def evaluate_detection_one_epoch(
    model,
    val_loader,
    device,
    epoch,
    use_amp=True,
    score_thr=0.25,
    top_k=20,
    nms_iou_thr=0.5,
    max_val_batches=None,
    use_topk_fallback=True,
):
    model.eval()

    amp_enabled = get_amp_enabled(device, use_amp)

    pred_records = []
    gt_by_sample = []

    sample_id = 0
    skipped_empty_gt = 0
    total_selected = 0

    pbar_total = (
        len(val_loader)
        if max_val_batches is None
        else min(len(val_loader), max_val_batches)
    )

    pbar = tqdm(
        enumerate(val_loader),
        total=pbar_total,
        desc=f"Epoch {epoch} [Eval]",
        dynamic_ncols=True,
        leave=True
    )

    for step, batch in pbar:
        if max_val_batches is not None and step >= max_val_batches:
            break

        images = batch["images"].to(device, non_blocking=True)
        query_texts = batch["query_texts"]
        targets = move_targets_to_device(batch, device)

        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images, query_texts)

        pred_bbox = outputs["bbox"]
        pred_score_logit = get_score_logit(outputs)
        pred_scores = pred_score_logit.sigmoid().squeeze(-1)

        B = pred_bbox.shape[0]

        for b in range(B):
            gt_boxes = targets[b]["boxes"].detach().float().cpu()
            gt_by_sample.append(gt_boxes)

            if gt_boxes.numel() == 0:
                skipped_empty_gt += 1

            boxes_b = pred_bbox[b].detach()
            scores_b = pred_scores[b].detach()

            selected_boxes, selected_scores = select_predictions(
                boxes=boxes_b,
                scores=scores_b,
                score_thr=score_thr,
                top_k=top_k,
                nms_iou_thr=nms_iou_thr,
                use_topk_fallback=use_topk_fallback,
            )

            total_selected += int(selected_boxes.shape[0])

            for box, score in zip(selected_boxes, selected_scores):
                pred_records.append({
                    "sample_id": sample_id,
                    "score": float(score.item()),
                    "box": box.detach().float().cpu(),
                })

            sample_id += 1

        pbar.set_postfix({
            "samples": sample_id,
            "pred": len(pred_records),
            "sel/img": f"{total_selected / max(1, sample_id):.2f}",
            "skip_empty": skipped_empty_gt,
        })

    ap50 = evaluate_ap_at_iou(pred_records, gt_by_sample, iou_thr=0.50)

    ap_list = []
    for thr in np.arange(0.50, 0.96, 0.05):
        ap_t = evaluate_ap_at_iou(pred_records, gt_by_sample, iou_thr=float(thr))
        ap_list.append(ap_t["ap"])

    map50_95 = float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0

    metrics = {
        "map50": ap50["ap"],
        "map50_95": map50_95,
        "precision": ap50["precision"],
        "recall": ap50["recall"],
        "tp": ap50["tp"],
        "fp": ap50["fp"],
        "num_gt": ap50["num_gt"],
        "num_pred": ap50["num_pred"],
        "valid_samples": sample_id,
        "skipped_empty_gt": skipped_empty_gt,
        "avg_selected_per_sample": total_selected / max(1, sample_id),
    }

    tqdm.write(
        f"Eval Epoch [{epoch}] "
        f"mAP50={metrics['map50']:.4f} "
        f"mAP50-95={metrics['map50_95']:.4f} "
        f"P={metrics['precision']:.4f} "
        f"R={metrics['recall']:.4f} "
        f"TP={metrics['tp']} "
        f"FP={metrics['fp']} "
        f"GT={metrics['num_gt']} "
        f"Pred={metrics['num_pred']} "
        f"skip_empty={metrics['skipped_empty_gt']}"
    )

    return metrics



# Metrics logging for external watcher


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_latest_json(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp_path = path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, path)


def append_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    write_header = not os.path.exists(path)

    fieldnames = list(row.keys())

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(row)



# Checkpoint


def build_checkpoint(
    model,
    ema,
    optimizer,
    scaler,
    scheduler,
    epoch,
    best_metric,
    best_metric_name,
    train_metrics,
    val_loss_metrics,
    eval_metrics,
    train_config,
    dynamic_config,
):
    return {
        "epoch": epoch,

        "model": model.state_dict(),
        "ema": ema.ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler_step": scheduler.step_num,

        "best_metric": best_metric,
        "best_metric_name": best_metric_name,

        "train_metrics": train_metrics,
        "val_loss_metrics": val_loss_metrics,
        "eval_metrics": eval_metrics,

        "train_config": train_config,
        "dynamic_config": dynamic_config,
        "scheduler_config": {
            "name": "WarmupCosineScheduler",
            "warmup_steps": scheduler.warmup_steps,
            "total_steps": scheduler.total_steps,
            "min_lr_ratio": scheduler.min_lr_ratio,
            "step_num": scheduler.step_num,
            "base_lrs": scheduler.base_lrs,
        },

        "rng_state": get_rng_state_dict(),
    }


def load_checkpoint(
    resume_path,
    model,
    ema,
    optimizer,
    scaler,
    scheduler,
    device,
):
    ckpt = torch.load(resume_path, map_location=device)

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    if ema is not None and ckpt.get("ema", None) is not None:
        ema.ema.load_state_dict(ckpt["ema"], strict=True)

    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    if "scheduler_step" in ckpt:
        scheduler.step_num = int(ckpt["scheduler_step"])

    if "rng_state" in ckpt:
        restore_rng_state(ckpt["rng_state"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_metric = float(ckpt.get("best_metric", -1.0))

    return start_epoch, best_metric

def collect_query_texts_from_datasets(*datasets):
    texts = set()

    for dataset in datasets:
        if dataset is None:
            continue

        samples = getattr(dataset, "samples", None)

        if samples is None:
            continue

        for sample in samples:
            if not isinstance(sample, dict):
                continue

            text = sample.get("query_text", None)

            if text is None:
                continue

            text = str(text).strip()

            if text:
                texts.add(text)

    return sorted(texts)


def ensure_precomputed_bert_raw_cache(
    cache_path,
    datasets,
    device,
    hidden_dim=512,
    max_length=32,
    batch_size=128,
    enabled=True,
):
    if not enabled:
        print("[BERT Precompute] Skip because freeze_bert=False")
        return None

    if cache_path is None:
        print("[BERT Precompute] Skip because precomputed_bert_path=None")
        return None

    cache_path = os.path.abspath(str(cache_path))

    if os.path.exists(cache_path):
        print(f"[BERT Precompute] Found existing cache, skip: {cache_path}")
        return cache_path

    texts = collect_query_texts_from_datasets(*datasets)

    if len(texts) == 0:
        raise RuntimeError(
            "[BERT Precompute] No query_text found from dataloader datasets."
        )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    print(f"[BERT Precompute] Cache not found. Building: {cache_path}")
    print(f"[BERT Precompute] Unique query texts: {len(texts)}")

    bert = Bert( 
        out_dim=hidden_dim,
        max_length=max_length,
        freeze_bert=True,
        precomputed_bert_path=None,
    ).to(device)

    bert.eval()

    cache = {}

    with torch.no_grad():
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="[BERT Precompute]",
            dynamic_ncols=True,
        ):
            batch_texts = texts[i:i + batch_size]

            encoded = bert.encode_raw(
                batch_texts,
                device=device,
            )

            hidden = encoded["last_hidden_state"].detach().cpu().half()
            mask = encoded["attention_mask"].detach().cpu()

            for j, text in enumerate(batch_texts):
                cache[text] = {
                    "last_hidden_state": hidden[j],
                    "attention_mask": mask[j],
                }

    tmp_path = cache_path + ".tmp"

    torch.save(
        {
            "type": "bert_raw_cache",
            "max_length": max_length,
            "hidden_size": int(hidden.shape[-1]),
            "num_texts": len(cache),
            "cache": cache,
        },
        tmp_path,
    )

    os.replace(tmp_path, cache_path)

    del bert

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[BERT Precompute] Done. Saved {len(cache)} texts to {cache_path}")

    return cache_path

# Main train


def train(args):
    set_seed(args.seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dataset_dir = args.dir

    train_image_dir = os.path.join(dataset_dir, "images", "train")
    train_anno_dir = os.path.join(dataset_dir, "labels", "train")

    val_image_dir = os.path.join(dataset_dir, "images", "val")
    val_anno_dir = os.path.join(dataset_dir, "labels", "val")

    train_loader, val_loader = build_dataloaders(
        train_image_dir=train_image_dir,
        train_anno_dir=train_anno_dir,
        val_image_dir=val_image_dir,
        val_anno_dir=val_anno_dir,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        num_workers=args.num_workers,
        max_text_aug_per_image=args.max_text_aug_per_image,
        random_seed=args.seed,
    )

    args.precomputed_bert_path = ensure_precomputed_bert_raw_cache(
        cache_path=args.precomputed_bert_path,
        datasets=[
            train_loader.dataset,
            val_loader.dataset,
        ],
        device=device,
        hidden_dim=args.hidden_dim,
        max_length=args.text_max_length,
        batch_size=args.batch_size,
        enabled=bool(args.freeze_bert),
    )

    model = VisionTextModel(
        img_in_channels=args.img_in_channels,
        hidden_dim=args.hidden_dim,
        target_size=(args.target_size, args.target_size),
        text_max_length=args.text_max_length,
        fusion_token_num=args.fusion_token_num,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        freeze_bert=args.freeze_bert,
        precomputed_bert_path=args.precomputed_bert_path,
    ).to(device)

    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None

    criterion = GroundingLoss(
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou,
        cost_score=args.cost_score,
    )

    total_params, trainable_params = count_parameters(model)

    print(f"[Info] Device: {device}")
    print(f"[Info] Model parameters: total={total_params}, trainable={trainable_params}")
    print(f"[Info] Train batches: {len(train_loader)}")
    print(f"[Info] Val batches: {len(val_loader)}")

    optimizer = build_optimizer(
        model=model,
        lr_vision=args.lr_vision,
        lr_text=args.lr_text,
        lr_transformer=args.lr_transformer,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(args.max_warmup_steps, args.warmup_epochs * len(train_loader))

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    scaler = build_scaler(device)

    if args.resume_path is not None:
        save_path = os.path.dirname(os.path.abspath(args.resume_path))
    else:
        if args.save_dir is None:
            save_path = os.path.join(
                "checkpoints",
                f"results_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
            )
        else:
            save_path = args.save_dir

    os.makedirs(save_path, exist_ok=True)

    epoch_metrics_path = os.path.join(save_path, "metrics_epoch.jsonl")
    step_metrics_path = os.path.join(save_path, "metrics_step.jsonl")
    latest_metrics_path = os.path.join(save_path, "latest_metrics.json")
    csv_metrics_path = os.path.join(save_path, "metrics_epoch.csv")

    print(f"[Info] Save path: {save_path}")
    print(f"[Info] Watch epoch metrics: {epoch_metrics_path}")
    print(f"[Info] Watch step metrics: {step_metrics_path}")
    print(f"[Info] Watch latest metrics: {latest_metrics_path}")

    start_epoch = 1
    best_metric = -1.0
    best_metric_name = args.best_metric

    if args.resume_path is not None:
        start_epoch, best_metric = load_checkpoint(
            resume_path=args.resume_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            device=device,
        )

        print(f"[Info] Resume from: {args.resume_path}")
        print(f"[Info] Start epoch: {start_epoch}")
        print(f"[Info] Best {best_metric_name}: {best_metric:.4f}")

    train_config = vars(args).copy()
    train_config.update({
        "train_image_dir": train_image_dir,
        "train_anno_dir": train_anno_dir,
        "val_image_dir": val_image_dir,
        "val_anno_dir": val_anno_dir,
        "save_path": save_path,
        "train_loader_len": len(train_loader),
        "val_loader_len": len(val_loader),
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
    })

    for epoch in range(start_epoch, args.epochs + 1):
        lambda_bbox, lambda_giou, lambda_score = get_loss_weights(
            epoch=epoch,
            total_epochs=args.epochs,
            args=args,
        )

        pos_weight = get_pos_weight(
            epoch=epoch,
            total_epochs=args.epochs,
            args=args,
        )

        train_metrics = train_one_epoch(
            model=model,
            ema=ema,
            criterion=criterion,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            num_epochs=args.epochs,
            use_amp=args.use_amp,
            grad_clip_norm=args.grad_clip_norm,
            lambda_bbox=lambda_bbox,
            lambda_giou=lambda_giou,
            lambda_score=lambda_score,
            pos_weight=pos_weight,
            log_interval=args.log_interval,
            step_metrics_path=step_metrics_path if args.emit_step_metrics else None,
        )

        if epoch % args.val_loss_interval == 0:
            val_loss_metrics = validate_loss_one_epoch(
                model=ema.ema if ema is not None else model,
                criterion=criterion,
                val_loader=val_loader,
                device=device,
                epoch=epoch,
                use_amp=args.use_amp,
                lambda_bbox=lambda_bbox,
                lambda_giou=lambda_giou,
                lambda_score=lambda_score,
                pos_weight=pos_weight,
            )
        else:
            val_loss_metrics = {}

        if epoch % args.eval_interval == 0:
            eval_metrics = evaluate_detection_one_epoch(
                model=ema.ema if ema is not None else model,
                val_loader=val_loader,
                device=device,
                epoch=epoch,
                use_amp=args.use_amp,
                score_thr=args.score_thr,
                top_k=args.top_k,
                nms_iou_thr=args.nms_iou_thr,
                max_val_batches=args.max_val_batches,
                use_topk_fallback=args.use_topk_fallback,
            )
        else:
            eval_metrics = {}

        dynamic_config = {
            "lambda_bbox": lambda_bbox,
            "lambda_giou": lambda_giou,
            "lambda_score": lambda_score,
            "pos_weight": pos_weight,
        }

        metric_row = {
            "type": "epoch",
            "time": time.time(),
            "epoch": epoch,
            "lr": scheduler.get_lr()[0],
            **train_metrics,
            **val_loss_metrics,
            **eval_metrics,
            **dynamic_config,
        }

        append_jsonl(epoch_metrics_path, metric_row)
        append_csv(csv_metrics_path, metric_row)
        write_latest_json(latest_metrics_path, metric_row)

        save_metric = float(eval_metrics.get(best_metric_name, -1.0))

        is_best = False

        if save_metric > best_metric:
            best_metric = save_metric
            is_best = True

        ckpt = build_checkpoint(
            model=model,
            ema=ema,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric,
            best_metric_name=best_metric_name,
            train_metrics=train_metrics,
            val_loss_metrics=val_loss_metrics,
            eval_metrics=eval_metrics,
            train_config=train_config,
            dynamic_config=dynamic_config,
        )

        latest_path = os.path.join(save_path, "latest.pt")
        torch.save(ckpt, latest_path)

        if is_best:
            best_path = os.path.join(save_path, f"best_{best_metric_name}.pt")
            torch.save(ckpt, best_path)

            tqdm.write(
                f"Saved best checkpoint: "
                f"epoch={epoch}, {best_metric_name}={best_metric:.4f}"
            )

        if args.save_epoch_interval > 0 and epoch % args.save_epoch_interval == 0:
            epoch_path = os.path.join(save_path, f"epoch_{epoch:03d}.pt")
            torch.save(ckpt, epoch_path)

        tqdm.write(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_metrics.get('train_loss', -1):.4f} "
            f"val_loss={val_loss_metrics.get('val_loss', -1):.4f} "
            f"mAP50={eval_metrics.get('map50', -1):.4f} "
            f"mAP50-95={eval_metrics.get('map50_95', -1):.4f} "
            f"P={eval_metrics.get('precision', -1):.4f} "
            f"R={eval_metrics.get('recall', -1):.4f} "
            f"lb={lambda_bbox:.2f} "
            f"lg={lambda_giou:.2f} "
            f"ls={lambda_score:.2f} "
            f"pw={pos_weight:.2f}"
        )

# ============================================================
# Config
# ============================================================

DEFAULT_MODEL_CFG = {
    "model": {
        "img_in_channels": 1024,
        "hidden_dim": 512,
        "num_heads": 8,
        "num_layers": 3,
        "mlp_ratio": 3.5,
        "image_grid_size": 10,
        "text_max_length": 32,
        "fusion_token_num": 16,
        "dropout": 0.1,
        "freeze_bert": True,
        "precomputed_bert_path": os.path.join(
            CURRENT_DIR,
            "cards",
            "cache",
            "bert_raw_cache.pt",
        ),
    }
}


DEFAULT_TRAIN_CFG = {
    "data": {
        "dataset_dir": os.path.join(PROJECT_ROOT, "datasets"),
        "image_size": 640,
        "max_text_aug_per_image": 1,
    },

    "train": {
        "epochs": 300,
        "batch_size": 12,
        "warmup_epochs": 5,
        "num_workers": 8,
        "device": "cuda:1",
        "seed": 42,
        "deterministic": True,
        "use_amp": True,
        "use_ema": True,
        "ema_decay": 0.999,
        "grad_clip_norm": 1.0,
    },

    "optim": {
        "lr_vision": 1e-4,
        "lr_text": 1e-5,
        "lr_transformer": 1e-4,
        "lr_head": 1e-4,
        "weight_decay": 1e-4,
        "min_lr_ratio": 0.05,
        "max_warmup_steps": 3000,
    },

    "loss": {
        "matcher": {
            "cost_bbox": 5.0,
            "cost_giou": 2.0,
            "cost_score": 1.0,
        },

        "weight": {
            "dynamic": True,

            "bbox": 5.0,
            "giou": 2.0,
            "score": 1.0,

            "bbox_start": 5.0,
            "bbox_end": 3.0,
            "bbox_decay_until": 0.5,

            "score_start": 1.0,
            "score_end": 2.0,
            "score_warm_until": 0.4,
        },

        "pos_weight": {
            "min": 1.0,
            "max": 5.0,
            "warm_until": 0.5,
        },
    },

    "eval": {
        "val_loss_interval": 1,
        "eval_interval": 1,
        "max_val_batches": 320,
        "score_thr": 0.25,
        "top_k": 20,
        "nms_iou_thr": 0.5,
        "best_metric": "map50",
        "use_topk_fallback": True,
    },

    "log": {
        "save_dir": None,
        "resume_path": None,
        "save_epoch_interval": 50,
        "emit_step_metrics": True,
        "log_interval": 10,
    },
}


def deepcopy_cfg(cfg):
    return copy.deepcopy(cfg)


def deep_update(base, override):
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and key in base
            and isinstance(base[key], dict)
        ):
            deep_update(base[key], value)
        else:
            base[key] = value

    return base


def load_yaml(path):
    if path is None:
        return {}

    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return {}

    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config must be a dict: {path}")

    return cfg


def load_model_config(path):
    cfg = deepcopy_cfg(DEFAULT_MODEL_CFG)
    yaml_cfg = load_yaml(path)
    return deep_update(cfg, yaml_cfg)


def load_train_config(path):
    cfg = deepcopy_cfg(DEFAULT_TRAIN_CFG)
    yaml_cfg = load_yaml(path)
    return deep_update(cfg, yaml_cfg)


def normalize_device(device):
    if device is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    if isinstance(device, int):
        return f"cuda:{device}" if torch.cuda.is_available() else "cpu"

    if isinstance(device, str):
        device = device.strip()

        if device.isdigit():
            return f"cuda:{device}" if torch.cuda.is_available() else "cpu"

        return device

    raise TypeError(f"Unsupported device type: {type(device)}")


def set_deterministic(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def cfg_to_args(model_cfg_all, train_cfg_all):
    model_cfg = model_cfg_all["model"]

    data_cfg = train_cfg_all["data"]
    train_cfg = train_cfg_all["train"]
    optim_cfg = train_cfg_all["optim"]
    loss_cfg = train_cfg_all["loss"]
    eval_cfg = train_cfg_all["eval"]
    log_cfg = train_cfg_all["log"]

    matcher_cfg = loss_cfg.get("matcher", {})
    weight_cfg = loss_cfg.get("weight", {})
    pos_weight_cfg = loss_cfg.get("pos_weight", {})

    return SimpleNamespace(
        # data
        dir=data_cfg["dataset_dir"],
        image_size=data_cfg["image_size"],
        max_text_aug_per_image=data_cfg["max_text_aug_per_image"],

        # train
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        warmup_epochs=train_cfg["warmup_epochs"],
        num_workers=train_cfg["num_workers"],
        device=train_cfg["device"],
        seed=train_cfg["seed"],
        deterministic=train_cfg["deterministic"],
        use_amp=train_cfg["use_amp"],
        use_ema=train_cfg["use_ema"],
        ema_decay=train_cfg["ema_decay"],
        grad_clip_norm=train_cfg["grad_clip_norm"],

        # model
        img_in_channels=model_cfg["img_in_channels"],
        hidden_dim=model_cfg["hidden_dim"],
        target_size=model_cfg["image_grid_size"],
        text_max_length=model_cfg["text_max_length"],
        fusion_token_num=model_cfg["fusion_token_num"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        mlp_ratio=model_cfg["mlp_ratio"],
        dropout=model_cfg["dropout"],
        freeze_bert=model_cfg["freeze_bert"],
        precomputed_bert_path=model_cfg.get("precomputed_bert_path", None),

        # optimizer
        lr_vision=optim_cfg["lr_vision"],
        lr_text=optim_cfg["lr_text"],
        lr_transformer=optim_cfg["lr_transformer"],
        lr_head=optim_cfg["lr_head"],
        weight_decay=optim_cfg["weight_decay"],
        min_lr_ratio=optim_cfg["min_lr_ratio"],
        max_warmup_steps=optim_cfg["max_warmup_steps"],

        # matcher cost
        cost_bbox=matcher_cfg.get("cost_bbox", 5.0),
        cost_giou=matcher_cfg.get("cost_giou", 2.0),
        cost_score=matcher_cfg.get("cost_score", 1.0),

        # final loss weight
        loss_dynamic=weight_cfg.get("dynamic", True),

        lambda_bbox=weight_cfg.get("bbox", 5.0),
        lambda_giou=weight_cfg.get("giou", 2.0),
        lambda_score=weight_cfg.get("score", 1.0),

        lambda_bbox_start=weight_cfg.get("bbox_start", 5.0),
        lambda_bbox_end=weight_cfg.get("bbox_end", 3.0),
        lambda_bbox_decay_until=weight_cfg.get("bbox_decay_until", 0.5),

        lambda_score_start=weight_cfg.get("score_start", 1.0),
        lambda_score_end=weight_cfg.get("score_end", 2.0),
        lambda_score_warm_until=weight_cfg.get("score_warm_until", 0.4),

        # BCE positive weight
        min_pos_weight=pos_weight_cfg.get("min", 1.0),
        max_pos_weight=pos_weight_cfg.get("max", 5.0),
        pos_weight_warm_until=pos_weight_cfg.get("warm_until", 0.5),

        # eval
        val_loss_interval=eval_cfg["val_loss_interval"],
        eval_interval=eval_cfg["eval_interval"],
        max_val_batches=eval_cfg["max_val_batches"],
        score_thr=eval_cfg["score_thr"],
        top_k=eval_cfg["top_k"],
        nms_iou_thr=eval_cfg["nms_iou_thr"],
        best_metric=eval_cfg["best_metric"],
        use_topk_fallback=eval_cfg.get("use_topk_fallback", True),

        # log
        save_dir=log_cfg["save_dir"],
        resume_path=log_cfg["resume_path"],
        save_epoch_interval=log_cfg["save_epoch_interval"],
        emit_step_metrics=log_cfg["emit_step_metrics"],
        log_interval=log_cfg["log_interval"],

        # raw config
        model_cfg=model_cfg_all,
        train_cfg=train_cfg_all,
    )


def print_config_summary(model_cfg, train_cfg):
    m = model_cfg["model"]
    d = train_cfg["data"]
    t = train_cfg["train"]
    o = train_cfg["optim"]
    l = train_cfg["loss"]
    e = train_cfg["eval"]
    log = train_cfg["log"]

    matcher = l.get("matcher", {})
    weight = l.get("weight", {})
    pos_weight = l.get("pos_weight", {})

    print("\n[LightDet] Training config")
    print(f"  dataset     : {d['dataset_dir']}")
    print(f"  image_size  : {d['image_size']}")
    print(f"  epochs      : {t['epochs']}")
    print(f"  batch       : {t['batch_size']}")
    print(f"  workers     : {t['num_workers']}")
    print(f"  device      : {t['device']}")
    print(f"  seed        : {t['seed']}")
    print(f"  save_dir    : {log['save_dir']}")

    print(f"  hidden_dim  : {m['hidden_dim']}")
    print(f"  grid        : {m['image_grid_size']}x{m['image_grid_size']}")
    print(f"  num_layers  : {m['num_layers']}")
    print(f"  num_heads   : {m['num_heads']}")
    print(f"  mlp_ratio   : {m['mlp_ratio']}")
    print(f"  bert_cache  : {m.get('precomputed_bert_path', None)}")
    print(f"  lr_vision   : {o['lr_vision']}")
    print(f"  lr_text     : {o['lr_text']}")
    print(f"  lr_trans    : {o['lr_transformer']}")
    print(f"  lr_head     : {o['lr_head']}")

    print(
        f"  matcher     : "
        f"bbox={matcher.get('cost_bbox', 5.0)}, "
        f"giou={matcher.get('cost_giou', 2.0)}, "
        f"score={matcher.get('cost_score', 1.0)}"
    )

    print(
        f"  loss weight : "
        f"dynamic={weight.get('dynamic', True)}, "
        f"bbox={weight.get('bbox_start', weight.get('bbox', 5.0))}"
        f"->{weight.get('bbox_end', weight.get('bbox', 5.0))}, "
        f"giou={weight.get('giou', 2.0)}, "
        f"score={weight.get('score_start', weight.get('score', 1.0))}"
        f"->{weight.get('score_end', weight.get('score', 1.0))}"
    )

    print(
        f"  pos weight  : "
        f"{pos_weight.get('min', 1.0)}->{pos_weight.get('max', 5.0)}"
    )

    print(
        f"  eval        : "
        f"metric={e['best_metric']}, "
        f"score_thr={e['score_thr']}, "
        f"top_k={e['top_k']}"
    )
    print("")


class LightDet:
    def __init__(self, model="cards/config/model.yaml"):
        self.model_cfg_path = model
        self.model_cfg = load_model_config(model)

    def train(
        self,
        cfg="cards/config/train.yaml",
        data=None,
        epochs=None,
        imgsz=None,
        batch=None,
        device=None,
        workers=None,
        seed=None,
        deterministic=None,
        project="runs/train",
        name="exp",
        resume=None,
        lr=None,
        lr_vision=None,
        lr_text=None,
        lr_transformer=None,
        lr_head=None,
        weight_decay=None,
        score_thr=None,
        top_k=None,
        nms_iou_thr=None,
        **kwargs,
    ):
        model_cfg = deepcopy_cfg(self.model_cfg)
        train_cfg = load_train_config(cfg)

        if data is not None:
            train_cfg["data"]["dataset_dir"] = data

        if epochs is not None:
            train_cfg["train"]["epochs"] = epochs

        if imgsz is not None:
            train_cfg["data"]["image_size"] = imgsz

        if batch is not None:
            train_cfg["train"]["batch_size"] = batch

        if device is not None:
            train_cfg["train"]["device"] = normalize_device(device)
        else:
            train_cfg["train"]["device"] = normalize_device(
                train_cfg["train"]["device"]
            )

        if workers is not None:
            train_cfg["train"]["num_workers"] = workers

        if seed is not None:
            train_cfg["train"]["seed"] = seed

        if deterministic is not None:
            train_cfg["train"]["deterministic"] = deterministic

        if resume is not None:
            train_cfg["log"]["resume_path"] = resume

        if project is not None and name is not None:
            train_cfg["log"]["save_dir"] = os.path.join(project, name)

        if lr is not None:
            train_cfg["optim"]["lr_vision"] = lr
            train_cfg["optim"]["lr_transformer"] = lr
            train_cfg["optim"]["lr_head"] = lr

        if lr_vision is not None:
            train_cfg["optim"]["lr_vision"] = lr_vision

        if lr_text is not None:
            train_cfg["optim"]["lr_text"] = lr_text

        if lr_transformer is not None:
            train_cfg["optim"]["lr_transformer"] = lr_transformer

        if lr_head is not None:
            train_cfg["optim"]["lr_head"] = lr_head

        if weight_decay is not None:
            train_cfg["optim"]["weight_decay"] = weight_decay

        if score_thr is not None:
            train_cfg["eval"]["score_thr"] = score_thr

        if top_k is not None:
            train_cfg["eval"]["top_k"] = top_k

        if nms_iou_thr is not None:
            train_cfg["eval"]["nms_iou_thr"] = nms_iou_thr

        if len(kwargs) > 0:
            unknown = ", ".join(kwargs.keys())
            raise TypeError(f"Unsupported train arguments: {unknown}")

        set_deterministic(
            seed=train_cfg["train"]["seed"],
            deterministic=train_cfg["train"]["deterministic"],
        )

        args = cfg_to_args(
            model_cfg_all=model_cfg,
            train_cfg_all=train_cfg,
        )

        print_config_summary(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
        )

        return train(args)


def main():
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    model = LightDet(
        model="/home/soic/Desktop/LightDet/units/model/cards/config/model.yaml"
    )

    model.train(
        cfg="/home/soic/Desktop/LightDet/units/model/cards/config/train.yaml",
        data="/home/soic/Desktop/LightDet/datasets",
        epochs=300,
        imgsz=512,
        batch=24,
        device=1,
        workers=8,
        seed=44,
        deterministic=False,
        project="runs/train",
        name="lightdet_seed44",
    )


if __name__ == "__main__":
    main()
