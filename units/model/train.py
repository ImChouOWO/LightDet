import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UNITS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, UNITS_DIR)

from model.cards.main import Model
from model.cards.loss import compute_total_loss
from model.pipeline.data import build_dataloaders
import shutil
import random
import numpy as np
import math
import copy
import torch
import torch.multiprocessing as mp
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import time
from torchvision.ops import nms

mp.set_sharing_strategy("file_system")


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


def build_optimizer(
    model,
    lr_backbone=1e-5,
    lr_text_encoder=1e-5,
    lr_head=1e-4,
    weight_decay=1e-4
):
    no_decay = ["bias", "LayerNorm.weight", "norm.weight"]
    param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "vis_text_model.backbone" in name:
            lr = lr_backbone
        elif "vis_text_model.text_encoder" in name:
            lr = lr_text_encoder
        else:
            lr = lr_head

        decay = 0.0 if any(nd in name for nd in no_decay) else weight_decay

        param_groups.append({
            "params": [param],
            "lr": lr,
            "weight_decay": decay
        })

    return AdamW(param_groups)


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.05):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.step_num = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self):
        self.step_num += 1

        if self.step_num <= self.warmup_steps:
            factor = self.step_num / max(1, self.warmup_steps)
        else:
            progress = (self.step_num - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


def get_drop_proposal_prob(
    epoch,
    total_epochs,
    max_drop=0.8,
    start_ratio=0.3,
    end_ratio=0.7
):
    progress = epoch / total_epochs

    if progress < start_ratio:
        return 0.0

    if progress >= end_ratio:
        return max_drop

    ratio = (progress - start_ratio) / max(1e-8, end_ratio - start_ratio)
    return max_drop * ratio

def train_one_epoch(
    model,
    ema,
    train_loader,
    optimizer,
    scheduler,
    device,
    epoch,
    num_epochs,
    image_size,
    scaler=None,
    use_amp=True,
    grad_clip_norm=1.0,
    lambda_bbox=1.0,
    lambda_score=1.0,
    lambda_text=0.5,
    positive_ratio=0.05,
    drop_proposal_prob=0.0
):
    model.train()
    total_loss_sum = 0.0
    dropped_batch_count = 0

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch}/{num_epochs} [Train]",
        dynamic_ncols=True,
        leave=True
    )

    for step, batch in pbar:
        images = batch["images"].to(device, non_blocking=True)

        boxes_per_image = [
            b.to(device, non_blocking=True)
            for b in batch["boxes_per_image"]
        ]

        target_boxes_per_image = [
            b.to(device, non_blocking=True)
            for b in batch["target_boxes_per_image"]
        ]

        query_texts = batch["query_texts"]

        proposal_dropped = False

        if drop_proposal_prob > 0.0:
            if torch.rand(1, device=device).item() < drop_proposal_prob:
                boxes_per_image = [
                    torch.empty(
                        0,
                        4,
                        device=device,
                        dtype=b.dtype
                    )
                    for b in boxes_per_image
                ]

                proposal_dropped = True
                dropped_batch_count += 1

        optimizer.zero_grad(set_to_none=True)

        amp_enabled = use_amp and device.type == "cuda"

        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(
                images=images,
                boxes_per_image=boxes_per_image,
                texts=query_texts,
                image_size=image_size
            )

            loss, loss_dict = compute_total_loss(
                outputs=outputs,
                gt_boxes_per_image=target_boxes_per_image,
                text_feat=outputs["text_feat"],
                lambda_bbox=lambda_bbox,
                lambda_score=lambda_score,
                lambda_text=lambda_text,
                positive_ratio=positive_ratio
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

        total_loss_sum += loss.item()
        avg_loss = total_loss_sum / (step + 1)
        current_lr = scheduler.get_lr()[0]
        drop_rate_now = dropped_batch_count / max(1, step + 1)

        pbar.set_postfix({
            "lr": f"{current_lr:.2e}",
            "loss": f"{loss.item():.4f}",
            "avg": f"{avg_loss:.4f}",
            "bbox": f"{loss_dict['loss_bbox'].item():.4f}",
            "score": f"{loss_dict['loss_score'].item():.4f}",
            "text": f"{loss_dict['loss_text'].item():.4f}",
            "lb": f"{lambda_bbox:.2f}",
            "ls": f"{lambda_score:.2f}",
            "lt": f"{lambda_text:.2f}",
            "pos": f"{positive_ratio:.3f}",
            "drop": f"{drop_proposal_prob:.2f}",
            "dr": f"{drop_rate_now:.2f}",
        })

    return total_loss_sum / max(1, len(train_loader))


@torch.no_grad()
def validate_one_epoch(
    model,
    val_loader,
    device,
    epoch,
    image_size,
    use_amp=True,
    lambda_bbox=1.0,
    lambda_score=1.0,
    lambda_text=0.5,
    positive_ratio=0.05
):
    model.eval()
    total_loss_sum = 0.0

    pbar = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        desc=f"Epoch {epoch} [Val]",
        dynamic_ncols=True,
        leave=True
    )

    for step, batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        boxes_per_image = [b.to(device, non_blocking=True) for b in batch["boxes_per_image"]]
        target_boxes_per_image = [
            b.to(device, non_blocking=True)
            for b in batch["target_boxes_per_image"]
        ]
        query_texts = batch["query_texts"]

        amp_enabled = use_amp and device.type == "cuda"

        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(
                images=images,
                boxes_per_image=boxes_per_image,
                texts=query_texts,
                image_size=image_size
            )

            loss, loss_dict = compute_total_loss(
                outputs=outputs,
                gt_boxes_per_image=target_boxes_per_image,
                text_feat=outputs["text_feat"],
                lambda_bbox=lambda_bbox,
                lambda_score=lambda_score,
                lambda_text=lambda_text,
                positive_ratio=positive_ratio
            )

        total_loss_sum += loss.item()
        avg_loss = total_loss_sum / (step + 1)

        pbar.set_postfix({
            "val_loss": f"{loss.item():.4f}",
            "avg": f"{avg_loss:.4f}",
            "bbox": f"{loss_dict['loss_bbox'].item():.4f}",
            "score": f"{loss_dict['loss_score'].item():.4f}",
            "text": f"{loss_dict['loss_text'].item():.4f}",
            "pos": f"{positive_ratio:.3f}",
        })

    return total_loss_sum / max(1, len(val_loader))


def cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=-1)


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


@torch.no_grad()
def inference_validate_one_epoch(
    model,
    val_loader,
    device,
    epoch,
    image_size,
    use_amp=True,
    score_thr=0.25,
    top_k=20,
    nms_iou_thr=0.5,
    max_val_batches=320
):
    model.eval()

    total_oracle_iou = 0.0
    total_multi_best_iou = 0.0
    total_samples = 0

    oracle_recall_hits_05 = 0
    multi_recall_hits_05 = 0

    oracle_recall_hits_075 = 0
    multi_recall_hits_075 = 0

    skipped_empty_gt = 0

    pbar_total = (
        len(val_loader)
        if max_val_batches is None
        else min(len(val_loader), max_val_batches)
    )

    pbar = tqdm(
        enumerate(val_loader),
        total=pbar_total,
        desc=f"Epoch {epoch} [Infer Val]",
        dynamic_ncols=True,
        leave=True
    )

    for step, batch in pbar:
        if max_val_batches is not None and step >= max_val_batches:
            break

        images = batch["images"].to(device, non_blocking=True)

        boxes_per_image = [
            b.to(device, non_blocking=True)
            for b in batch["boxes_per_image"]
        ]

        target_boxes_per_image = [
            b.to(device, non_blocking=True)
            for b in batch["target_boxes_per_image"]
        ]

        query_texts = batch["query_texts"]

        amp_enabled = use_amp and device.type == "cuda"

        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(
                images=images,
                boxes_per_image=boxes_per_image,
                texts=query_texts,
                image_size=image_size
            )

        pred_bbox_set = outputs["bbox"]        # [B, N, 4]
        score_logits = outputs["score"]        # [B, N, 1]

        scores = torch.sigmoid(score_logits).squeeze(-1)

        B, N, _ = pred_bbox_set.shape

        oracle_iou_list = []
        multi_best_ious = []
        valid_B = 0

        for b in range(B):
            gt_boxes = target_boxes_per_image[b]

            # 空 GT 直接略過，不納入 IoU / Recall 統計
            if gt_boxes.numel() == 0:
                skipped_empty_gt += 1
                continue

            valid_B += 1

            
            # Oracle IoU
            # 代表 proposal set 裡面理論上能達到的最佳 IoU
            
            proposal_iou_matrix = compute_iou_multi_single(
                pred_bbox_set[b],
                gt_boxes
            )  # [N, M]

            if proposal_iou_matrix.numel() == 0:
                oracle_iou_b = torch.tensor(0.0, device=device)
            else:
                oracle_iou_b = proposal_iou_matrix.max()

            oracle_iou_list.append(oracle_iou_b)

            
            # Multi Proposal selection
            
            b_boxes = pred_bbox_set[b]
            b_scores = scores[b]

            keep = b_scores >= score_thr

            if keep.sum() > 0:
                selected_boxes = b_boxes[keep]
                selected_scores = b_scores[keep]
            else:
                k = min(top_k, N)

                if k == 0:
                    selected_boxes = b_boxes.new_zeros((0, 4))
                    selected_scores = b_scores.new_zeros((0,))
                else:
                    selected_scores, top_idx = b_scores.topk(k=k)
                    selected_boxes = b_boxes[top_idx]

            # 限制 proposal 數量
            if selected_scores.numel() > top_k:
                selected_scores, top_idx = selected_scores.topk(k=top_k)
                selected_boxes = selected_boxes[top_idx]

            
            # NMS
            
            if selected_boxes.numel() == 0:
                multi_best_iou = torch.tensor(0.0, device=device)
            else:
                selected_boxes_xyxy = cxcywh_to_xyxy(selected_boxes)

                keep_idx = nms(
                    selected_boxes_xyxy.float(),
                    selected_scores.float(),
                    iou_threshold=nms_iou_thr
                )

                selected_boxes = selected_boxes[keep_idx]

                if selected_boxes.numel() == 0:
                    multi_best_iou = torch.tensor(0.0, device=device)
                else:
                    selected_ious_matrix = compute_iou_multi_single(
                        selected_boxes,
                        gt_boxes
                    )

                    if selected_ious_matrix.numel() == 0:
                        multi_best_iou = torch.tensor(0.0, device=device)
                    else:
                        multi_best_iou = selected_ious_matrix.max()

            multi_best_ious.append(multi_best_iou)

        # 這個 batch 如果全部都是 empty GT，就跳過統計
        if valid_B == 0:
            pbar.set_postfix({
                "multi_iou": f"{total_multi_best_iou / max(1, total_samples):.4f}",
                "oracle_iou": f"{total_oracle_iou / max(1, total_samples):.4f}",
                "MR@0.5": f"{multi_recall_hits_05 / max(1, total_samples):.4f}",
                "OR@0.5": f"{oracle_recall_hits_05 / max(1, total_samples):.4f}",
                "skip_empty": skipped_empty_gt,
            })
            continue

        
        # Batch statistics
        
        oracle_iou = torch.stack(oracle_iou_list, dim=0)
        multi_best_ious = torch.stack(multi_best_ious, dim=0)

        total_oracle_iou += oracle_iou.sum().item()
        total_multi_best_iou += multi_best_ious.sum().item()

        total_samples += valid_B

        oracle_recall_hits_05 += (oracle_iou >= 0.5).sum().item()
        multi_recall_hits_05 += (multi_best_ious >= 0.5).sum().item()

        oracle_recall_hits_075 += (oracle_iou >= 0.75).sum().item()
        multi_recall_hits_075 += (multi_best_ious >= 0.75).sum().item()

        mean_oracle_iou = total_oracle_iou / max(1, total_samples)
        mean_multi_iou = total_multi_best_iou / max(1, total_samples)

        oracle_recall_05 = oracle_recall_hits_05 / max(1, total_samples)
        multi_recall_05 = multi_recall_hits_05 / max(1, total_samples)

        oracle_recall_075 = oracle_recall_hits_075 / max(1, total_samples)
        multi_recall_075 = multi_recall_hits_075 / max(1, total_samples)

        pbar.set_postfix({
            "multi_iou": f"{mean_multi_iou:.4f}",
            "oracle_iou": f"{mean_oracle_iou:.4f}",
            "MR@0.5": f"{multi_recall_05:.4f}",
            "OR@0.5": f"{oracle_recall_05:.4f}",
            "MR@0.75": f"{multi_recall_075:.4f}",
            "OR@0.75": f"{oracle_recall_075:.4f}",
            "skip_empty": skipped_empty_gt,
        })

    
    # Final Metrics
    
    metrics = {
        "multi_mean_iou": total_multi_best_iou / max(1, total_samples),
        "oracle_iou": total_oracle_iou / max(1, total_samples),

        "multi_recall@0.5": multi_recall_hits_05 / max(1, total_samples),
        "oracle_recall@0.5": oracle_recall_hits_05 / max(1, total_samples),

        "multi_recall@0.75": multi_recall_hits_075 / max(1, total_samples),
        "oracle_recall@0.75": oracle_recall_hits_075 / max(1, total_samples),

        "valid_samples": total_samples,
        "skipped_empty_gt": skipped_empty_gt,
    }

    tqdm.write(
        f"Inference Val Epoch [{epoch}] "
        f"multi_iou={metrics['multi_mean_iou']:.4f} "
        f"oracle_iou={metrics['oracle_iou']:.4f} "
        f"MR@0.5={metrics['multi_recall@0.5']:.4f} "
        f"OR@0.5={metrics['oracle_recall@0.5']:.4f} "
        f"MR@0.75={metrics['multi_recall@0.75']:.4f} "
        f"OR@0.75={metrics['oracle_recall@0.75']:.4f} "
        f"valid={metrics['valid_samples']} "
        f"skip_empty={metrics['skipped_empty_gt']}"
    )

    return metrics

def state_dict_to_cpu(state_dict):
    """
    將 state_dict 轉到 CPU，避免 deepcopy checkpoint 時額外佔用 GPU VRAM。
    """
    cpu_state = {}

    for k, v in state_dict.items():
        if torch.is_tensor(v):
            cpu_state[k] = v.detach().cpu()
        else:
            cpu_state[k] = v

    return cpu_state

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if total >= 1e9:
        total = f"{total / 1e9:.3f}B"
    elif total >= 1e6:
        total = f"{total / 1e6:.3f}M"

    if trainable >= 1e9:
        trainable = f"{trainable / 1e9:.3f}B"
    elif trainable >= 1e6:
        trainable = f"{trainable / 1e6:.3f}M"

    return [total, trainable]

def get_positive_ratio(epoch, total_epochs):
    progress = epoch / total_epochs

    min_ratio = 0.02
    max_ratio = 0.10

    ratio = min_ratio + (max_ratio - min_ratio) * min(progress / 0.6, 1.0)

    return ratio

def get_lambda_score(progress):
    if progress < 0.2:
        return 1.0 + (3.0 - 1.0) * (progress / 0.2)
    elif progress < 0.6:
        return 3.0 - (3.0 - 1.5) * ((progress - 0.2) / 0.4)
    else:
        return 1.5 - (1.5 - 1.0) * ((progress - 0.6) / 0.4)

def get_loss_weights(epoch, total_epochs):
    progress = epoch / total_epochs

    lambda_bbox = 3.0 - 1.0 * min(progress / 0.5, 1.0)
    lambda_score = get_lambda_score(progress)
    lambda_text = 0.05 + 0.45 * min(progress / 0.4, 1.0)

    return lambda_bbox, lambda_score, lambda_text


def train(
    dir,
    epochs=100,
    warmup_epochs=5,
    batch_size=4,
    image_size=(640, 640),
    ema_decay=0.999,
    lr_backbone=1e-5,
    lr_text_encoder=1e-5,
    lr_head=1e-4,
    weight_decay=1e-4,
    device=None,
    num_workers=16,
    infer_val_interval=1,
    max_val_batches=320,
    score_thr=0.25,
    top_k=20,
    nms_iou_thr=0.5,
    save_best_interval=50,
    resume_path=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = epochs

    train_image_dir = f"{dir}/images/train"
    train_anno_dir = f"{dir}/labels/train"

    val_image_dir = f"{dir}/images/val"
    val_anno_dir = f"{dir}/labels/val"

    train_loader, val_loader = build_dataloaders(
        train_image_dir=train_image_dir,
        train_anno_dir=train_anno_dir,
        val_image_dir=val_image_dir,
        val_anno_dir=val_anno_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers
    )

    model_cfg = None
    config_path = os.path.join(CURRENT_DIR, "cards", "config", "model.yaml")

    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            model_cfg = config.get("model", {})
        except Exception:
            model_cfg = None

    model = Model(num_classes=1).to(device)
    ema = ModelEMA(model, decay=ema_decay)

    parameters = count_parameters(model)
    print(f"[Info]: Model parameters: total={parameters[0]}, trainable={parameters[1]}")

    optimizer = build_optimizer(
        model=model,
        lr_backbone=lr_backbone,
        lr_text_encoder=lr_text_encoder,
        lr_head=lr_head,
        weight_decay=weight_decay
    )

    total_steps = num_epochs * len(train_loader)
    warmup_steps = min(3000, warmup_epochs * len(train_loader))

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=0.05
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_mean_iou = -1.0
    best_window_iou = -1.0
    best_window_epoch = -1

    start_epoch = 1

    os.makedirs("checkpoints", exist_ok=True)

    if resume_path is not None:
        resume_path = os.path.abspath(resume_path)
        ckpt_resume = torch.load(resume_path, map_location=device)

        if "model" in ckpt_resume:
            model.load_state_dict(ckpt_resume["model"], strict=True)
        else:
            model.load_state_dict(ckpt_resume, strict=True)

        if "ema" in ckpt_resume:
            ema.ema.load_state_dict(ckpt_resume["ema"], strict=True)

        if "optimizer" in ckpt_resume:
            optimizer.load_state_dict(ckpt_resume["optimizer"])

        if "scaler" in ckpt_resume:
            scaler.load_state_dict(ckpt_resume["scaler"])

        if "scheduler_step" in ckpt_resume:
            scheduler.step_num = ckpt_resume["scheduler_step"]

        if "rng_state" in ckpt_resume:
            rng_state = ckpt_resume["rng_state"]

            if rng_state.get("torch", None) is not None:
                torch.set_rng_state(rng_state["torch"])

            if (
                torch.cuda.is_available()
                and rng_state.get("cuda", None) is not None
            ):
                torch.cuda.set_rng_state_all(rng_state["cuda"])

            if rng_state.get("python", None) is not None:
                random.setstate(rng_state["python"])

            if rng_state.get("numpy", None) is not None:
                try:
                    np.random.set_state(rng_state["numpy"])
                except Exception:
                    pass

        start_epoch = int(ckpt_resume.get("epoch", 0)) + 1
        best_mean_iou = float(ckpt_resume.get("best_mean_iou", -1.0))

        old_save_path = os.path.dirname(resume_path)
        save_path = old_save_path

        print(f"[Info]: Resume from: {resume_path}")
        print(f"[Info]: Start epoch: {start_epoch}")
        print(f"[Info]: Checkpoints will be saved to: {save_path}")

    else:
        save_path = f"checkpoints/results_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(save_path, exist_ok=True)
        print(f"[Info]: Checkpoints will be saved to: LightDet/units/model/{save_path}")

    window_tmp_path = f"{save_path}/_window_best_tmp.pt"

    last_infer_metrics = {
        "multi_mean_iou": -1.0,
        "oracle_iou": -1.0,
        "multi_recall@0.5": -1.0,
        "oracle_recall@0.5": -1.0,
        "multi_recall@0.75": -1.0,
        "oracle_recall@0.75": -1.0,
        "valid_samples": 0,
        "skipped_empty_gt": 0,
    }

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

    def build_checkpoint(
        epoch,
        train_loss,
        infer_metrics,
        lambda_bbox,
        lambda_score,
        lambda_text,
        positive_ratio,
        best_mean_iou,
        best_window_iou,
        best_window_epoch
    ):
        ckpt = {
            "epoch": epoch,

            "model": model.state_dict(),
            "ema": ema.ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler_step": scheduler.step_num,

            "best_mean_iou": best_mean_iou,
            "best_window_iou": best_window_iou,
            "best_window_epoch": best_window_epoch,
            "best_metric_name": "multi_mean_iou",

            "train_loss": train_loss,
            "infer_metrics": infer_metrics,

            "lambda_bbox": lambda_bbox,
            "lambda_score": lambda_score,
            "lambda_text": lambda_text,
            "positive_ratio": positive_ratio,

            "train_config": {
                "dataset_dir": dir,
                "train_image_dir": train_image_dir,
                "train_anno_dir": train_anno_dir,
                "val_image_dir": val_image_dir,
                "val_anno_dir": val_anno_dir,

                "epochs": num_epochs,
                "start_epoch": start_epoch,
                "warmup_epochs": warmup_epochs,
                "warmup_steps": warmup_steps,
                "total_steps": total_steps,

                "batch_size": batch_size,
                "image_size": image_size,
                "ema_decay": ema_decay,

                "lr_backbone": lr_backbone,
                "lr_text_encoder": lr_text_encoder,
                "lr_head": lr_head,
                "weight_decay": weight_decay,

                "num_workers": num_workers,
                "infer_val_interval": infer_val_interval,

                "max_val_batches": max_val_batches,
                "score_thr": score_thr,
                "top_k": top_k,
                "nms_iou_thr": nms_iou_thr,

                "save_best_interval": save_best_interval,
                "device": str(device),
                "train_loader_len": len(train_loader),
                "val_loader_len": len(val_loader),
            },

            "scheduler_config": {
                "name": "WarmupCosineScheduler",
                "warmup_steps": warmup_steps,
                "total_steps": total_steps,
                "min_lr_ratio": scheduler.min_lr_ratio,
                "step_num": scheduler.step_num,
                "base_lrs": scheduler.base_lrs,
            },

            "loss_schedule_state": {
                "lambda_bbox": lambda_bbox,
                "lambda_score": lambda_score,
                "lambda_text": lambda_text,
                "positive_ratio": positive_ratio,
            },

            "model_cfg": model_cfg,

            "rng_state": get_rng_state_dict(),
        }

        return ckpt

    for epoch in range(start_epoch, num_epochs + 1):
        lambda_bbox, lambda_score, lambda_text = get_loss_weights(
            epoch=epoch,
            total_epochs=num_epochs
        )

        positive_ratio = get_positive_ratio(epoch, num_epochs)
        drop_proposal_prob = get_drop_proposal_prob(
            epoch=epoch,
            total_epochs=num_epochs,
            max_drop=config.get("proposal", {}).get("max_drop", 0.8),
            start_ratio=config.get("proposal", {}).get("start_ratio", 0.3),
            end_ratio=config.get("proposal", {}).get("end_ratio", 0.7)
        )

        train_loss = train_one_epoch(
            model=model,
            ema=ema,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            scaler=scaler,
            use_amp=True,
            grad_clip_norm=1.0,
            num_epochs=num_epochs,
            image_size=image_size,
            lambda_bbox=lambda_bbox,
            lambda_score=lambda_score,
            lambda_text=lambda_text,
            positive_ratio=positive_ratio,
            drop_proposal_prob=drop_proposal_prob
        )

        if epoch % infer_val_interval == 0:
            infer_metrics = inference_validate_one_epoch(
                model=ema.ema,
                val_loader=val_loader,
                device=device,
                epoch=epoch,
                image_size=image_size,
                use_amp=True,
                score_thr=score_thr,
                top_k=top_k,
                nms_iou_thr=nms_iou_thr,
                max_val_batches=max_val_batches
            )
            last_infer_metrics = infer_metrics
        else:
            infer_metrics = last_infer_metrics

        multi_iou = infer_metrics.get("multi_mean_iou", -1.0)
        oracle_iou = infer_metrics.get("oracle_iou", -1.0)

        multi_recall_05 = infer_metrics.get("multi_recall@0.5", -1.0)
        oracle_recall_05 = infer_metrics.get("oracle_recall@0.5", -1.0)

        multi_recall_075 = infer_metrics.get("multi_recall@0.75", -1.0)
        oracle_recall_075 = infer_metrics.get("oracle_recall@0.75", -1.0)

        valid_samples = infer_metrics.get("valid_samples", 0)
        skipped_empty_gt = infer_metrics.get("skipped_empty_gt", 0)

        tqdm.write(
            f"Epoch [{epoch}/{num_epochs}] "
            f"skip_empty={skipped_empty_gt} "
            f"train_loss={train_loss:.4f} "
            f"multi_iou={multi_iou:.4f} "
            f"oracle_iou={oracle_iou:.4f} "
            f"MR@0.5={multi_recall_05:.4f} "
            f"OR@0.5={oracle_recall_05:.4f} "
            f"lb={lambda_bbox:.2f} "
            f"ls={lambda_score:.2f} "
            f"lt={lambda_text:.2f}"
        )

        save_metric = infer_metrics.get("multi_mean_iou", -1.0)

        is_global_best = False
        is_window_best = False

        if save_metric > best_mean_iou:
            best_mean_iou = save_metric
            is_global_best = True

        if save_metric > best_window_iou:
            best_window_iou = save_metric
            best_window_epoch = epoch
            is_window_best = True

        ckpt = build_checkpoint(
            epoch=epoch,
            train_loss=train_loss,
            infer_metrics=infer_metrics,
            lambda_bbox=lambda_bbox,
            lambda_score=lambda_score,
            lambda_text=lambda_text,
            positive_ratio=positive_ratio,
            best_mean_iou=best_mean_iou,
            best_window_iou=best_window_iou,
            best_window_epoch=best_window_epoch
        )

        torch.save(ckpt, f"{save_path}/latest.pt")

        if is_global_best:
            torch.save(ckpt, f"{save_path}/best_iou.pt")
            tqdm.write(
                f"Saved global best checkpoint: "
                f"epoch={epoch}, multi_mean_iou={best_mean_iou:.4f}"
            )

        if is_window_best:
            torch.save(ckpt, window_tmp_path)

        if epoch % save_best_interval == 0:
            start_window_epoch = epoch - save_best_interval + 1
            end_window_epoch = epoch

            if os.path.exists(window_tmp_path):
                window_path = (
                    f"{save_path}/best_iou_epoch_"
                    f"{start_window_epoch:03d}_{end_window_epoch:03d}.pt"
                )

                shutil.copyfile(window_tmp_path, window_path)

                tqdm.write(
                    f"Saved window best IoU checkpoint: "
                    f"epoch {start_window_epoch}-{end_window_epoch}, "
                    f"best_epoch={best_window_epoch}, "
                    f"multi_mean_iou={best_window_iou:.4f}"
                )

                os.remove(window_tmp_path)

            best_window_iou = -1.0
            best_window_epoch = -1


if __name__ == "__main__":
    DIR = "/home/soic/Desktop/LightDet/datasets"

    train(
        dir=DIR,
        epochs=300,
        batch_size=32,
        image_size=(640, 640),
        device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
        num_workers=18,
        infer_val_interval=1
    )