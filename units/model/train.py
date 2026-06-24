from __future__ import annotations

# LightDet optimized train.py v5: EMA + QFL + scheduled ranking + JSON text negatives

from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import copy
import csv
import inspect
import json
import math
import os
import random
import resource
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms
from tqdm import tqdm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UNITS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))

for path in [PROJECT_ROOT, UNITS_DIR, CURRENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from units.tool.card import VisionTextModel, Bert
from units.model.pipeline.data import (
    QueryBudgetBatchSampler,
    build_dataloaders,
    grounding_collate_fn,
)
from units.model.cards.loss import GroundingLoss


# 此資料集的每批資料包含多個 query target。file_descriptor 會為每個
# shared tensor 消耗檔案描述符，容易在 Train/Val worker pools 並存時觸發
# Errno 24。file_system 使用共享記憶體名稱傳遞 storage，較適合此 workload。
try:
    mp.set_sharing_strategy("file_system")
except (RuntimeError, ValueError):
    pass


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


def configure_process_file_limit(target_soft_limit: int = 65536) -> Tuple[int, int]:
    """Raise RLIMIT_NOFILE when the host allows it and return effective limits."""
    try:
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = max(int(soft_limit), int(target_soft_limit))

        if hard_limit != resource.RLIM_INFINITY:
            target = min(target, int(hard_limit))

        if target > soft_limit:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard_limit))

        effective_soft, effective_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        return int(effective_soft), int(effective_hard)
    except (OSError, ValueError, AttributeError):
        return -1, -1


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_deterministic(seed: int = 42, deterministic: bool = True) -> None:
    set_seed(seed)

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

        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def configure_torch_runtime(args: SimpleNamespace) -> None:
    """設定 CUDA / matmul 執行策略，不改變模型輸入輸出介面。"""

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(args.matmul_precision))

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    # reduce-overhead / compile 在 config 中可開啟；預設關閉以保留最大相容性。
    if hasattr(torch, "_dynamo"):
        try:
            torch._dynamo.config.cache_size_limit = 64
        except Exception:
            pass


def count_parameters(model: torch.nn.Module) -> Tuple[str, str]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def fmt(value: int) -> str:
        if value >= 1e9:
            return f"{value / 1e9:.3f}B"
        if value >= 1e6:
            return f"{value / 1e6:.3f}M"
        if value >= 1e3:
            return f"{value / 1e3:.3f}K"
        return str(value)

    return fmt(total), fmt(trainable)


def get_rng_state_dict() -> Dict[str, Any]:
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


def restore_rng_state(rng_state: Any) -> None:
    if not isinstance(rng_state, dict):
        return

    if rng_state.get("torch") is not None:
        torch.set_rng_state(rng_state["torch"])

    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])

    if rng_state.get("python") is not None:
        random.setstate(rng_state["python"])

    if rng_state.get("numpy") is not None:
        try:
            np.random.set_state(rng_state["numpy"])
        except Exception:
            pass


def parse_amp_dtype(value: Any) -> torch.dtype:
    text = str(value).strip().lower()

    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16", "half"}:
        return torch.float16
    if text in {"fp32", "float32", "none"}:
        return torch.float32

    raise ValueError(f"Unsupported AMP dtype: {value}")


def get_amp_enabled(device: torch.device, use_amp: bool = True) -> bool:
    return bool(use_amp and device.type == "cuda")


def resolve_amp_dtype_for_device(
    device: torch.device,
    requested_dtype: torch.dtype,
) -> torch.dtype:
    """依 GPU 能力解析 AMP dtype；Ada 正常使用 BF16。"""
    if device.type != "cuda":
        return torch.float32

    if requested_dtype == torch.bfloat16:
        is_supported = True
        if hasattr(torch.cuda, "is_bf16_supported"):
            is_supported = bool(torch.cuda.is_bf16_supported())

        if not is_supported:
            print(
                "[Warning] CUDA device does not report BF16 support; "
                "fallback to FP16 AMP."
            )
            return torch.float16

    return requested_dtype


def build_scaler(
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> GradScaler:
    # BF16 不需要 loss scaling；FP16 才啟用 GradScaler。
    enabled = bool(
        use_amp
        and device.type == "cuda"
        and amp_dtype == torch.float16
    )

    try:
        return GradScaler(device.type, enabled=enabled)
    except TypeError:
        return GradScaler(enabled=enabled)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def move_images_to_device(
    images: torch.Tensor,
    device: torch.device,
    channels_last: bool,
) -> torch.Tensor:
    """
    將 DataLoader 影像搬到運算裝置並統一為 float32。

    LightDet 的 uint8 image cache 刻意保留 CHW uint8，以降低 CPU RAM 與
    Host-to-Device 傳輸量；因此必須在 GPU transfer 後才轉為 [0, 1] float32。
    非 cache 路徑本來就是 float tensor，則保留其數值尺度，只統一 dtype。
    """
    if not torch.is_tensor(images):
        raise TypeError(f"images must be a Tensor, got {type(images)}")

    if images.ndim != 4:
        raise ValueError(
            f"images must be BCHW [B, C, H, W], got shape={tuple(images.shape)}"
        )

    if images.shape[1] not in (1, 3, 4):
        raise ValueError(
            f"unexpected image channel count: C={images.shape[1]}, "
            f"shape={tuple(images.shape)}"
        )

    source_dtype = images.dtype

    # 先以 uint8 搬到 GPU，維持最小 H2D transfer；再由 GPU 轉 float32。
    images = images.to(device=device, non_blocking=True)

    if source_dtype == torch.uint8:
        images = images.to(dtype=torch.float32).mul_(1.0 / 255.0)
    elif source_dtype.is_floating_point:
        # autocast 會在卷積/矩陣運算時轉為 BF16/FP16；模型輸入維持 FP32
        # 可避免非 autocast op 出現 dtype 不一致。
        if images.dtype != torch.float32:
            images = images.float()
    else:
        raise TypeError(
            "Unsupported image dtype. Expected uint8 or floating point, "
            f"got {source_dtype}."
        )

    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    elif not images.is_contiguous():
        images = images.contiguous()

    return images


def prepare_model_batch(
    batch: Dict[str, Any],
    device: torch.device,
    channels_last: bool,
) -> Tuple[torch.Tensor, List[str], Optional[torch.Tensor]]:
    """
    同時支援兩種 DataLoader schema：

    1. query-level batching
       images:        [Q, C, H, W]
       query_texts:   Q 個文字
       image_indices: None

    2. image-level batching
       unique_images: [U, C, H, W]
       query_texts:   Q 個文字
       image_indices: [Q]，將 U 張影像特徵展開到 Q 個 query

    image-level batching 可避免同一張影像因不同文字 query 重複跑 vision backbone。
    """
    query_texts = batch.get("query_texts")

    if query_texts is None:
        raise KeyError(
            f"Batch missing 'query_texts'. Available keys: {sorted(batch.keys())}"
        )

    if "unique_images" in batch:
        images = move_images_to_device(
            batch["unique_images"],
            device,
            channels_last=channels_last,
        )

        image_indices = batch.get("image_indices")
        if image_indices is None:
            raise KeyError(
                "Image-level batch contains 'unique_images' but is missing "
                "'image_indices'."
            )

        if not torch.is_tensor(image_indices):
            image_indices = torch.as_tensor(image_indices, dtype=torch.long)
        else:
            image_indices = image_indices.to(dtype=torch.long)

        if image_indices.ndim != 1:
            raise ValueError(
                f"image_indices must be 1-D, got shape={tuple(image_indices.shape)}"
            )

        if image_indices.numel() != len(query_texts):
            raise ValueError(
                "image_indices/query_texts size mismatch: "
                f"{image_indices.numel()} != {len(query_texts)}"
            )

        if image_indices.numel() > 0:
            min_index = int(image_indices.min().item())
            max_index = int(image_indices.max().item())
            if min_index < 0 or max_index >= images.shape[0]:
                raise IndexError(
                    "image_indices out of range: "
                    f"min={min_index}, max={max_index}, "
                    f"num_unique_images={images.shape[0]}"
                )

        image_indices = image_indices.to(
            device=device,
            non_blocking=True,
        )

        return images, query_texts, image_indices

    if "images" in batch:
        images = move_images_to_device(
            batch["images"],
            device,
            channels_last=channels_last,
        )

        if images.shape[0] != len(query_texts):
            raise ValueError(
                "images/query_texts batch size mismatch: "
                f"{images.shape[0]} != {len(query_texts)}"
            )

        return images, query_texts, None

    raise KeyError(
        "Batch must contain either 'unique_images' or 'images'. "
        f"Available keys: {sorted(batch.keys())}"
    )


def forward_model_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    query_texts: List[str],
    image_indices: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if image_indices is None:
        return model(images, query_texts)

    return model(
        images,
        query_texts,
        image_indices=image_indices,
    )


def run_startup_smoke_test(
    model: torch.nn.Module,
    criterion: Any,
    train_loader: Any,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    channels_last: bool,
    total_epochs: int,
    args: SimpleNamespace,
) -> None:
    """
    使用真實 DataLoader batch 驗證 cache dtype、image-level mapping、model forward
    與 loss contract。只執行一次，不更新權重或 BatchNorm running statistics。
    """
    print("[Startup Check] Running one real-batch forward/loss smoke test...")

    # Build one batch synchronously from dataset indices. This avoids spawning
    # the persistent Train worker pool before the real epoch begins.
    try:
        first_indices = next(iter(train_loader.batch_sampler))
    except StopIteration as error:
        raise RuntimeError("Training DataLoader is empty") from error

    if isinstance(first_indices, torch.Tensor):
        first_indices = first_indices.tolist()
    elif isinstance(first_indices, int):
        first_indices = [first_indices]
    else:
        first_indices = list(first_indices)

    items = [train_loader.dataset[int(index)] for index in first_indices]
    collate_fn = train_loader.collate_fn or compact_grounding_collate_fn
    batch = collate_fn(items)

    images, query_texts, image_indices = prepare_model_batch(
        batch=batch,
        device=device,
        channels_last=channels_last,
    )
    targets = move_targets_to_device(batch, device)

    if len(query_texts) == 0:
        raise RuntimeError("Startup batch contains no query_texts")
    if len(targets) != len(query_texts):
        raise RuntimeError(
            "Startup targets/query_texts mismatch: "
            f"{len(targets)} != {len(query_texts)}"
        )
    if not images.dtype.is_floating_point:
        raise RuntimeError(
            f"Prepared images must be floating point, got {images.dtype}"
        )

    lambda_bbox, lambda_giou, lambda_score = get_loss_weights(
        epoch=1,
        total_epochs=total_epochs,
        args=args,
    )

    was_training = model.training
    model.eval()
    amp_enabled = get_amp_enabled(device, use_amp)

    try:
        with torch.no_grad():
            with autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=amp_dtype if amp_enabled else None,
            ):
                outputs = forward_model_batch(
                    model=model,
                    images=images,
                    query_texts=query_texts,
                    image_indices=image_indices,
                )

                pred_bbox = outputs["bbox"]
                pred_score_logit = get_score_logit(outputs)

                loss, _ = criterion(
                    pred_bbox=pred_bbox,
                    pred_score_logit=pred_score_logit,
                    targets=targets,
                    lambda_bbox=lambda_bbox,
                    lambda_giou=lambda_giou,
                    lambda_score=lambda_score,
                    lambda_rank=args.lambda_rank,
                    pos_weight=float(args.pos_weight),
                    current_epoch=1,
                    quality_warmup_epoch=args.quality_warmup_epoch,
                    rank_start_epoch=args.rank_start_epoch,
                    rank_warmup_epoch=args.rank_warmup_epoch,
                    rank_alpha_min=args.rank_alpha_min,
                    query_loss_weights=batch.get("query_loss_weights"),
                    text_negative_mask=batch.get("text_negative_mask"),
                )

        expected_queries = len(query_texts)
        if pred_bbox.shape[0] != expected_queries:
            raise RuntimeError(
                "Model bbox batch does not match query count: "
                f"{pred_bbox.shape[0]} != {expected_queries}"
            )
        if pred_score_logit.shape[0] != expected_queries:
            raise RuntimeError(
                "Model score batch does not match query count: "
                f"{pred_score_logit.shape[0]} != {expected_queries}"
            )
        if not bool(torch.isfinite(pred_bbox).all().item()):
            raise FloatingPointError("Startup bbox output contains NaN/Inf")
        if not bool(torch.isfinite(pred_score_logit).all().item()):
            raise FloatingPointError("Startup score output contains NaN/Inf")
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Startup loss is NaN/Inf")

        image_min, image_max = torch.aminmax(images)
        print(
            "[Startup Check] PASS: "
            f"images={tuple(images.shape)} {images.dtype} "
            f"range=[{image_min.item():.6f}, {image_max.item():.6f}], "
            f"queries={expected_queries}, "
            f"bbox={tuple(pred_bbox.shape)}, "
            f"score={tuple(pred_score_logit.shape)}, "
            f"loss={loss.item():.6f}"
        )
    finally:
        model.train(was_training)
        del batch, images, query_texts, targets
        if image_indices is not None:
            del image_indices
        if device.type == "cuda":
            torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# EMA
# -----------------------------------------------------------------------------


class ModelEMA:
    """
    使用 foreach kernel 更新 EMA，避免每 step 建立兩份 state_dict 與逐參數 Python copy。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        update_interval: int = 1,
        buffer_update_interval: int = 1,
    ) -> None:
        model = unwrap_model(model)

        self.ema = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.update_interval = max(1, int(update_interval))
        self.buffer_update_interval = max(1, int(buffer_update_interval))
        self.num_updates = 0

        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

        self._ema_params = list(self.ema.parameters())
        self._ema_buffers = list(self.ema.buffers())

    @staticmethod
    def _group_tensor_pairs(
        destination: Sequence[torch.Tensor],
        source: Sequence[torch.Tensor],
    ) -> Iterable[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        groups: Dict[Tuple[str, Optional[int], torch.dtype], Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}

        for dst, src in zip(destination, source):
            key = (dst.device.type, dst.device.index, dst.dtype)

            if key not in groups:
                groups[key] = ([], [])

            groups[key][0].append(dst)
            groups[key][1].append(src.detach())

        return groups.values()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1

        if self.num_updates % self.update_interval != 0:
            return

        model = unwrap_model(model)
        model_params = list(model.parameters())

        for ema_group, model_group in self._group_tensor_pairs(
            self._ema_params,
            model_params,
        ):
            if len(ema_group) == 0:
                continue

            if ema_group[0].dtype.is_floating_point:
                try:
                    torch._foreach_lerp_(
                        ema_group,
                        model_group,
                        weight=1.0 - self.decay,
                    )
                except (RuntimeError, AttributeError):
                    for ema_value, model_value in zip(ema_group, model_group):
                        ema_value.lerp_(model_value, 1.0 - self.decay)
            else:
                try:
                    torch._foreach_copy_(ema_group, model_group)
                except (RuntimeError, AttributeError):
                    for ema_value, model_value in zip(ema_group, model_group):
                        ema_value.copy_(model_value)

        if self.num_updates % self.buffer_update_interval == 0:
            model_buffers = list(model.buffers())

            for ema_group, model_group in self._group_tensor_pairs(
                self._ema_buffers,
                model_buffers,
            ):
                if len(ema_group) == 0:
                    continue

                try:
                    torch._foreach_copy_(ema_group, model_group)
                except (RuntimeError, AttributeError):
                    for ema_value, model_value in zip(ema_group, model_group):
                        ema_value.copy_(model_value)


# -----------------------------------------------------------------------------
# Optimizer / scheduler
# -----------------------------------------------------------------------------


def _parameter_group_name(name: str) -> str:
    if "text_model.model" in name or "text_model.proj" in name:
        return "text"
    if "bottle_net" in name or "img_model" in name:
        return "vision"
    if "transformer" in name:
        return "transformer"
    return "head"


def _use_weight_decay(name: str, parameter: torch.Tensor) -> bool:
    lower_name = name.lower()

    if parameter.ndim <= 1:
        return False
    if name.endswith(".bias"):
        return False
    if "norm" in lower_name or "batchnorm" in lower_name or ".bn" in lower_name:
        return False

    return True


def build_optimizer(
    model: torch.nn.Module,
    lr_vision: float = 1e-4,
    lr_text: float = 1e-5,
    lr_transformer: float = 1e-4,
    lr_head: float = 1e-4,
    weight_decay: float = 1e-4,
    fused: bool = True,
) -> AdamW:
    """把每個 parameter 一組改成最多 8 組，降低 AdamW 與 scheduler Python overhead。"""

    learning_rates = {
        "vision": float(lr_vision),
        "text": float(lr_text),
        "transformer": float(lr_transformer),
        "head": float(lr_head),
    }

    grouped: Dict[Tuple[str, bool], List[torch.nn.Parameter]] = defaultdict(list)

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        module_group = _parameter_group_name(name)
        apply_decay = _use_weight_decay(name, parameter)
        grouped[(module_group, apply_decay)].append(parameter)

    param_groups = []

    for (module_group, apply_decay), parameters in grouped.items():
        param_groups.append({
            "params": parameters,
            "lr": learning_rates[module_group],
            "weight_decay": float(weight_decay) if apply_decay else 0.0,
            "name": f"{module_group}_{'decay' if apply_decay else 'no_decay'}",
        })

    optimizer_kwargs: Dict[str, Any] = {}

    if fused and torch.cuda.is_available():
        optimizer_kwargs["fused"] = True

    try:
        optimizer = AdamW(param_groups, **optimizer_kwargs)
    except (TypeError, RuntimeError):
        optimizer = AdamW(param_groups)

    return optimizer


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.05,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.step_num = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self) -> None:
        self.step_num += 1

        if self.step_num <= self.warmup_steps:
            factor = self.step_num / max(1, self.warmup_steps)
        else:
            progress = (self.step_num - self.warmup_steps) / max(
                1,
                self.total_steps - self.warmup_steps,
            )
            progress = min(max(progress, 0.0), 1.0)

            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def get_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


# -----------------------------------------------------------------------------
# Dynamic training schedule
# -----------------------------------------------------------------------------


def get_loss_weights(
    epoch: int,
    total_epochs: int,
    args: SimpleNamespace,
) -> Tuple[float, float, float]:
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
        + (
            float(args.lambda_bbox_end)
            - float(args.lambda_bbox_start)
        )
        * bbox_ratio
    )

    lambda_giou = float(args.lambda_giou)

    score_warm_until = max(1e-8, float(args.lambda_score_warm_until))
    score_ratio = min(progress / score_warm_until, 1.0)

    lambda_score = (
        float(args.lambda_score_start)
        + (
            float(args.lambda_score_end)
            - float(args.lambda_score_start)
        )
        * score_ratio
    )

    return lambda_bbox, lambda_giou, lambda_score


# -----------------------------------------------------------------------------
# Box / target helpers
# -----------------------------------------------------------------------------


def box_area(box: torch.Tensor) -> torch.Tensor:
    return (
        (box[..., 2] - box[..., 0]).clamp(min=0)
        * (box[..., 3] - box[..., 1]).clamp(min=0)
    )


def box_iou_xyxy(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
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

    return intersection / union.clamp(min=eps)


def compact_grounding_collate_fn(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate through the project implementation, then pack all GT boxes into one
    contiguous tensor. GroundingLoss and evaluation only consume target["boxes"].

    This changes IPC from dozens/hundreds of small target storages per batch to:
      - one image tensor
      - one image_indices tensor
      - one flat GT box tensor
      - one offsets tensor
    """
    batch = grounding_collate_fn(items)
    targets = batch.pop("targets", [])

    box_tensors: List[torch.Tensor] = []
    offsets = [0]

    for target in targets:
        boxes = target.get("boxes")
        if boxes is None:
            raise KeyError("Each target must contain 'boxes'")
        if not torch.is_tensor(boxes):
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes = boxes.to(dtype=torch.float32).reshape(-1, 4).contiguous()
        box_tensors.append(boxes)
        offsets.append(offsets[-1] + int(boxes.shape[0]))

    if box_tensors and offsets[-1] > 0:
        flat_boxes = torch.cat(box_tensors, dim=0).contiguous()
    else:
        flat_boxes = torch.empty((0, 4), dtype=torch.float32)

    batch["target_boxes_flat"] = flat_boxes
    batch["target_offsets"] = torch.tensor(offsets, dtype=torch.int64)
    batch["num_targets"] = len(targets)

    # Query-level collate contains several duplicate tensor lists. They are not
    # consumed by train.py and substantially increase shared-memory objects.
    for key in (
        "boxes_per_image",
        "boxes_pixel_per_image",
        "labels_per_image",
        "target_boxes_per_image",
        "target_boxes_pixel_per_image",
        "target_labels_per_image",
        "image_sizes",
        "orig_sizes",
        "obj_indices",
    ):
        batch.pop(key, None)

    return batch


def seed_dataloader_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_raw_targets(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "target_boxes_flat" in batch and "target_offsets" in batch:
        flat_boxes = batch["target_boxes_flat"]
        offsets_tensor = batch["target_offsets"]

        if not torch.is_tensor(flat_boxes) or not torch.is_tensor(offsets_tensor):
            raise TypeError("Compact target tensors are invalid")

        offsets = offsets_tensor.tolist()
        return [
            {"boxes": flat_boxes[offsets[index]:offsets[index + 1]]}
            for index in range(len(offsets) - 1)
        ]

    if "targets" in batch:
        return batch["targets"]

    boxes_list = batch.get("target_boxes_per_image")

    if boxes_list is None:
        raise KeyError(
            "Batch must contain compact targets, 'targets', or "
            "'target_boxes_per_image'."
        )

    labels_list = batch.get("target_labels_per_image")
    targets = []

    for index, boxes in enumerate(boxes_list):
        target: Dict[str, Any] = {"boxes": boxes}
        if labels_list is not None:
            target["labels"] = labels_list[index]
        targets.append(target)

    return targets

def move_targets_to_device(
    batch: Dict[str, Any],
    device: torch.device,
) -> List[Dict[str, Any]]:
    if "target_boxes_flat" in batch and "target_offsets" in batch:
        flat_boxes = batch["target_boxes_flat"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        offsets = batch["target_offsets"].tolist()
        return [
            {"boxes": flat_boxes[offsets[index]:offsets[index + 1]]}
            for index in range(len(offsets) - 1)
        ]

    targets = []
    for target in get_raw_targets(batch):
        moved: Dict[str, Any] = {}
        for key, value in target.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device, non_blocking=True)
            else:
                moved[key] = value
        if "boxes" not in moved:
            raise KeyError("target must contain key: boxes")
        targets.append(moved)
    return targets

def get_target_boxes_cpu(batch: Dict[str, Any]) -> List[torch.Tensor]:
    if "target_boxes_flat" in batch and "target_offsets" in batch:
        flat_boxes = batch["target_boxes_flat"]
        if flat_boxes.device.type != "cpu":
            flat_boxes = flat_boxes.cpu()
        flat_boxes = flat_boxes.to(dtype=torch.float32)
        offsets = batch["target_offsets"].tolist()
        return [
            flat_boxes[offsets[index]:offsets[index + 1]].reshape(-1, 4)
            for index in range(len(offsets) - 1)
        ]

    boxes_list = []
    for target in get_raw_targets(batch):
        boxes = target["boxes"]
        if not torch.is_tensor(boxes):
            boxes = torch.as_tensor(boxes)
        boxes = boxes.detach()
        if boxes.device.type != "cpu":
            boxes = boxes.cpu()
        boxes_list.append(boxes.to(dtype=torch.float32).reshape(-1, 4))
    return boxes_list

def get_score_logit(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "score_logit" in outputs:
        return outputs["score_logit"]

    if "score" in outputs:
        return outputs["score"]

    raise KeyError("Model output must contain score_logit or score")


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def make_progress_bar(
    iterable: Iterable[Any],
    *,
    total: int,
    desc: str,
    leave: bool,
    mininterval: float,
) -> tqdm:
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        dynamic_ncols=True,
        leave=leave,
        mininterval=mininterval,
    )


def train_one_epoch(
    model: torch.nn.Module,
    ema_source_model: torch.nn.Module,
    ema: Optional[ModelEMA],
    criterion: torch.nn.Module,
    train_loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    channels_last: bool = False,
    grad_clip_norm: Optional[float] = 1.0,
    lambda_bbox: float = 5.0,
    lambda_giou: float = 2.0,
    lambda_score: float = 1.0,
    lambda_rank: float = 0.10,
    pos_weight: float = 1.0,
    quality_warmup_epoch: int = 20,
    rank_start_epoch: int = 15,
    rank_warmup_epoch: int = 30,
    rank_alpha_min: float = 1e-4,
    log_interval: int = 10,
    step_metrics_path: Optional[str] = None,
    progress_leave: bool = True,
    progress_mininterval: float = 0.5,
) -> Dict[str, float]:
    model.train()

    batch_sampler = getattr(train_loader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)

    total_loss_sum = torch.zeros((), device=device)
    total_bbox_sum = torch.zeros((), device=device)
    total_giou_sum = torch.zeros((), device=device)
    total_score_sum = torch.zeros((), device=device)
    total_rank_contrib_sum = torch.zeros((), device=device)
    total_rank_raw_sum = torch.zeros((), device=device)
    total_text_negative_sum = torch.zeros((), device=device)
    total_text_negative_queries = 0

    amp_enabled = get_amp_enabled(device, use_amp)

    pbar = make_progress_bar(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch}/{num_epochs} [Train]",
        leave=progress_leave,
        mininterval=progress_mininterval,
    )

    for step, batch in pbar:
        global_step = (epoch - 1) * len(train_loader) + step + 1

        first_batch = epoch == 1 and step == 0

        if first_batch:
            raw_images = (
                batch["unique_images"]
                if "unique_images" in batch
                else batch["images"]
            )
            schema = "image-level" if "unique_images" in batch else "query-level"
            tqdm.write(
                f"[Info] Batch schema: {schema} "
                f"raw_shape={tuple(raw_images.shape)}, "
                f"raw_dtype={raw_images.dtype}, "
                f"queries={len(batch['query_texts'])}"
            )

        images, query_texts, image_indices = prepare_model_batch(
            batch=batch,
            device=device,
            channels_last=channels_last,
        )
        targets = move_targets_to_device(batch, device)

        if len(targets) != len(query_texts):
            raise ValueError(
                "targets/query_texts size mismatch: "
                f"{len(targets)} != {len(query_texts)}"
            )

        if first_batch:
            image_min, image_max = torch.aminmax(images.detach())
            tqdm.write(
                "[Info] Prepared images: "
                f"shape={tuple(images.shape)}, dtype={images.dtype}, "
                f"device={images.device}, "
                f"range=[{float(image_min.item()):.6f}, "
                f"{float(image_max.item()):.6f}], "
                f"image_indices="
                f"{None if image_indices is None else tuple(image_indices.shape)}"
            )

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            device_type=device.type,
            enabled=amp_enabled,
            dtype=amp_dtype if amp_enabled else None,
        ):
            outputs = forward_model_batch(
                model=model,
                images=images,
                query_texts=query_texts,
                image_indices=image_indices,
            )

            pred_bbox = outputs["bbox"]
            pred_score_logit = get_score_logit(outputs)

            loss, loss_dict = criterion(
                pred_bbox=pred_bbox,
                pred_score_logit=pred_score_logit,
                targets=targets,
                lambda_bbox=lambda_bbox,
                lambda_giou=lambda_giou,
                lambda_score=lambda_score,
                lambda_rank=lambda_rank,
                pos_weight=pos_weight,
                current_epoch=epoch,
                quality_warmup_epoch=quality_warmup_epoch,
                rank_start_epoch=rank_start_epoch,
                rank_warmup_epoch=rank_warmup_epoch,
                rank_alpha_min=rank_alpha_min,
                query_loss_weights=batch.get("query_loss_weights"),
                text_negative_mask=batch.get("text_negative_mask"),
            )

        if scaler.is_enabled():
            scaler.scale(loss).backward()

            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    ema_source_model.parameters(),
                    grad_clip_norm,
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    ema_source_model.parameters(),
                    grad_clip_norm,
                )

            optimizer.step()

        scheduler.step()

        if ema is not None:
            ema.update(ema_source_model)

        zero = pred_bbox.new_zeros(())

        loss_det = loss.detach()
        bbox_det = loss_dict["loss_bbox"].detach()
        giou_det = loss_dict["loss_giou"].detach()
        score_det = loss_dict["loss_score"].detach()
        rank_raw_det = loss_dict.get("loss_rank_raw", zero).detach()
        rank_det = loss_dict.get("loss_rank", zero).detach()
        rank_contrib_det = loss_dict.get(
            "loss_rank_contrib",
            loss_dict.get("loss_rank", zero),
        ).detach()
        text_negative_det = loss_dict.get(
            "loss_text_negative",
            zero,
        ).detach()
        text_negative_queries = int(
            batch.get(
                "text_negative_mask",
                torch.zeros(0, dtype=torch.bool),
            ).sum().item()
        )

        total_loss_sum.add_(loss_det)
        total_bbox_sum.add_(bbox_det)
        total_giou_sum.add_(giou_det)
        total_score_sum.add_(score_det)
        total_rank_contrib_sum.add_(rank_contrib_det)
        total_rank_raw_sum.add_(rank_raw_det)
        total_text_negative_sum.add_(text_negative_det)
        total_text_negative_queries += text_negative_queries

        should_log = (
            (step + 1) % max(1, int(log_interval)) == 0
            or (step + 1) == len(train_loader)
        )

        if should_log:
            current_lr = scheduler.get_lr()[0]
            loss_item = float(loss_det.item())
            bbox_item = float(bbox_det.item())
            giou_item = float(giou_det.item())
            score_item = float(score_det.item())
            rank_raw_item = float(rank_raw_det.item())
            rank_item = float(rank_det.item())
            rank_contrib_item = float(rank_contrib_det.item())
            text_negative_item = float(text_negative_det.item())
            avg_loss = float((total_loss_sum / (step + 1)).item())

            rank_alpha_item = float(loss_dict.get("rank_alpha", 0.0))
            lambda_rank_eff_item = float(
                loss_dict.get("lambda_rank_eff", 0.0)
            )
            quality_alpha_item = float(loss_dict.get("quality_alpha", 1.0))

            score_target_pos_mean = loss_dict.get(
                "score_target_pos_mean",
                zero,
            )

            if torch.is_tensor(score_target_pos_mean):
                score_target_pos_mean = float(
                    score_target_pos_mean.detach().item()
                )
            else:
                score_target_pos_mean = float(score_target_pos_mean)

            pbar.set_postfix({
                "lr": f"{current_lr:.2e}",
                "loss": f"{loss_item:.4f}",
                "avg": f"{avg_loss:.4f}",
                "bbox": f"{bbox_item:.4f}",
                "giou": f"{giou_item:.4f}",
                "score": f"{score_item:.4f}",
                "rank": f"{rank_contrib_item:.4f}",
                "raw": f"{rank_raw_item:.4f}",
                "ra": f"{rank_alpha_item:.4f}",
                "lrk": f"{lambda_rank_eff_item:.4f}",
                "txtneg": f"{text_negative_item:.4f}",
                "nq": text_negative_queries,
            })

            if step_metrics_path is not None:
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
                    "loss_rank_raw": rank_raw_item,
                    "loss_rank": rank_item,
                    "loss_rank_contrib": rank_contrib_item,
                    "loss_text_negative": text_negative_item,
                    "text_negative_queries": text_negative_queries,
                    "lambda_bbox": lambda_bbox,
                    "lambda_giou": lambda_giou,
                    "lambda_score": lambda_score,
                    "lambda_rank_max": float(lambda_rank),
                    "lambda_rank_eff": lambda_rank_eff_item,
                    "pos_weight": pos_weight,
                    "quality_alpha": quality_alpha_item,
                    "rank_alpha": rank_alpha_item,
                    "rank_alpha_min": float(rank_alpha_min),
                    "score_target_pos_mean": score_target_pos_mean,
                })
    pbar.refresh()
    pbar.close()

    num_batches = max(1, len(train_loader))

    return {
        "train_loss": float((total_loss_sum / num_batches).item()),
        "train_loss_bbox": float((total_bbox_sum / num_batches).item()),
        "train_loss_giou": float((total_giou_sum / num_batches).item()),
        "train_loss_score": float((total_score_sum / num_batches).item()),
        "train_loss_rank_raw": float((total_rank_raw_sum / num_batches).item()),
        "train_loss_rank": float((total_rank_contrib_sum / num_batches).item()),
        "train_loss_text_negative": float(
            (total_text_negative_sum / num_batches).item()
        ),
        "train_text_negative_queries": int(total_text_negative_queries),
    }


# -----------------------------------------------------------------------------
# Fast query-conditioned binary detection metrics
# -----------------------------------------------------------------------------


@torch.no_grad()
def select_predictions_batch(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    score_thr: float = 0.001,
    top_k: int = 20,
    nms_iou_thr: float = 0.5,
    use_topk_fallback: bool = False,
    use_nms: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    在 GPU 上批次完成 top-k、threshold 與 batched NMS；每個 batch 僅集中搬一次 CPU。

    boxes:  [B, N, 4], normalized xyxy
    scores: [B, N], sigmoid scores
    """

    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must be [B, N, 4], got {tuple(boxes.shape)}")

    if scores.ndim != 2:
        raise ValueError(f"scores must be [B, N], got {tuple(scores.shape)}")

    batch_size, num_queries = scores.shape

    if num_queries == 0 or top_k <= 0:
        return [
            (
                torch.empty((0, 4), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32),
            )
            for _ in range(batch_size)
        ]

    k = min(int(top_k), int(num_queries))

    top_scores, top_indices = torch.topk(
        scores,
        k=k,
        dim=1,
        largest=True,
        sorted=True,
    )

    top_boxes = torch.gather(
        boxes,
        dim=1,
        index=top_indices.unsqueeze(-1).expand(-1, -1, 4),
    )

    valid_mask = top_scores >= float(score_thr)

    if use_topk_fallback:
        no_valid = ~valid_mask.any(dim=1)
        valid_mask[no_valid] = True

    batch_ids = torch.arange(
        batch_size,
        device=boxes.device,
        dtype=torch.long,
    ).unsqueeze(1).expand(batch_size, k)

    flat_boxes = top_boxes[valid_mask]
    flat_scores = top_scores[valid_mask]
    flat_batch_ids = batch_ids[valid_mask]

    if flat_boxes.numel() == 0:
        return [
            (
                torch.empty((0, 4), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32),
            )
            for _ in range(batch_size)
        ]

    if use_nms:
        keep = batched_nms(
            flat_boxes.float(),
            flat_scores.float(),
            flat_batch_ids,
            float(nms_iou_thr),
        )

        flat_boxes = flat_boxes[keep]
        flat_scores = flat_scores[keep]
        flat_batch_ids = flat_batch_ids[keep]

    # 非同步 copy 在 pinned destination 上才真正有利；這裡一次性 copy 已避免逐框同步。
    boxes_cpu = flat_boxes.detach().to(dtype=torch.float32, device="cpu")
    scores_cpu = flat_scores.detach().to(dtype=torch.float32, device="cpu")
    batch_ids_cpu = flat_batch_ids.detach().to(device="cpu")

    results: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for batch_index in range(batch_size):
        mask = batch_ids_cpu == batch_index
        boxes_i = boxes_cpu[mask]
        scores_i = scores_cpu[mask]

        # topk 與 batched_nms 的輸出已依 score 由高到低排列；
        # 依 batch mask 取出後仍保留該 sample 的相對順序。
        results.append((boxes_i, scores_i))

    return results


class BinaryDetectionAPAccumulator:
    """
    保留原本 query-conditioned binary detection 的 matching 定義，但：
      1. 每個 sample 僅計算一次 IoU matrix。
      2. 所有 IoU thresholds 同時計算。
      3. 所有 prediction 只做一次全域 score 排序。
      4. 不建立逐框 Python dict。
    """

    def __init__(
        self,
        iou_thresholds: Optional[Sequence[float]] = None,
    ) -> None:
        if iou_thresholds is None:
            iou_thresholds = [
                round(value, 2)
                for value in np.arange(0.50, 0.96, 0.05)
            ]

        self.iou_thresholds = torch.as_tensor(
            iou_thresholds,
            dtype=torch.float32,
        )

        if self.iou_thresholds.numel() == 0:
            raise ValueError("iou_thresholds must not be empty")

        self.score_chunks: List[torch.Tensor] = []
        self.tp_chunks: List[torch.Tensor] = []
        self.num_gt = 0
        self.num_pred = 0

    def update(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> None:
        pred_boxes = pred_boxes.float().reshape(-1, 4)
        pred_scores = pred_scores.float().reshape(-1)
        gt_boxes = gt_boxes.float().reshape(-1, 4)

        num_pred = int(pred_boxes.shape[0])
        num_gt = int(gt_boxes.shape[0])
        num_thresholds = int(self.iou_thresholds.numel())

        self.num_gt += num_gt
        self.num_pred += num_pred

        if num_pred == 0:
            return

        # select_predictions_batch 已保證每個 sample 依 score 降冪。
        tp_matrix = torch.zeros(
            (num_pred, num_thresholds),
            dtype=torch.bool,
        )

        if num_gt > 0:
            iou_matrix = box_iou_xyxy(pred_boxes, gt_boxes)
            best_iou, best_gt_index = iou_matrix.max(dim=1)

            matched = torch.zeros(
                (num_thresholds, num_gt),
                dtype=torch.bool,
            )

            for prediction_index in range(num_pred):
                gt_index = int(best_gt_index[prediction_index])
                valid_iou = (
                    best_iou[prediction_index]
                    >= self.iou_thresholds
                )
                is_true_positive = valid_iou & (~matched[:, gt_index])

                tp_matrix[prediction_index] = is_true_positive
                matched[is_true_positive, gt_index] = True

        self.score_chunks.append(pred_scores.contiguous())
        self.tp_chunks.append(tp_matrix.contiguous())

    def compute(self) -> Dict[str, float]:
        if self.num_gt == 0:
            return {
                "map50": 0.0,
                "map50_95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "tp": 0,
                "fp": int(self.num_pred),
                "num_gt": 0,
                "num_pred": int(self.num_pred),
            }

        if not self.score_chunks:
            return {
                "map50": 0.0,
                "map50_95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "tp": 0,
                "fp": 0,
                "num_gt": int(self.num_gt),
                "num_pred": 0,
            }

        all_scores = torch.cat(self.score_chunks, dim=0)
        all_tp = torch.cat(self.tp_chunks, dim=0).to(torch.float32)

        global_order = torch.argsort(
            all_scores,
            descending=True,
            stable=True,
        )

        all_tp = all_tp[global_order]
        all_fp = 1.0 - all_tp

        cumulative_tp = torch.cumsum(all_tp, dim=0)
        cumulative_fp = torch.cumsum(all_fp, dim=0)

        recall = cumulative_tp / max(1, self.num_gt)
        precision = cumulative_tp / torch.clamp(
            cumulative_tp + cumulative_fp,
            min=1e-7,
        )

        # Precision envelope，等價於原本由後往前逐點 max。
        precision_envelope = torch.flip(
            torch.cummax(
                torch.flip(precision, dims=[0]),
                dim=0,
            ).values,
            dims=[0],
        )

        previous_recall = torch.cat(
            [
                torch.zeros(
                    (1, recall.shape[1]),
                    dtype=recall.dtype,
                ),
                recall[:-1],
            ],
            dim=0,
        )

        recall_delta = recall - previous_recall
        ap_per_threshold = torch.sum(
            recall_delta * precision_envelope,
            dim=0,
        )

        final_tp = cumulative_tp[-1]
        final_fp = cumulative_fp[-1]

        # 找最接近 0.50 的 threshold，避免自訂 threshold 順序造成錯誤。
        threshold_50_index = int(
            torch.argmin(torch.abs(self.iou_thresholds - 0.50)).item()
        )

        tp50 = int(final_tp[threshold_50_index].item())
        fp50 = int(final_fp[threshold_50_index].item())

        return {
            "map50": float(ap_per_threshold[threshold_50_index].item()),
            "map50_95": float(ap_per_threshold.mean().item()),
            "precision": tp50 / max(1, tp50 + fp50),
            "recall": tp50 / max(1, self.num_gt),
            "tp": tp50,
            "fp": fp50,
            "num_gt": int(self.num_gt),
            "num_pred": int(self.num_pred),
        }


# -----------------------------------------------------------------------------
# Unified validation: one forward pass for val loss + metrics
# -----------------------------------------------------------------------------


@torch.inference_mode()
def validate_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    val_loader: Any,
    device: torch.device,
    epoch: int,
    compute_loss: bool,
    compute_metrics: bool,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    channels_last: bool = False,
    lambda_bbox: float = 5.0,
    lambda_giou: float = 2.0,
    lambda_score: float = 1.0,
    lambda_rank: float = 0.10,
    pos_weight: float = 1.0,
    quality_warmup_epoch: int = 20,
    rank_start_epoch: int = 15,
    rank_warmup_epoch: int = 30,
    rank_alpha_min: float = 1e-4,
    score_thr: float = 0.001,
    top_k: int = 20,
    nms_iou_thr: float = 0.5,
    use_topk_fallback: bool = False,
    use_nms: bool = True,
    iou_thresholds: Optional[Sequence[float]] = None,
    max_val_batches: Optional[int] = None,
    log_interval: int = 50,
    progress_leave: bool = True,
    progress_mininterval: float = 0.5,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not compute_loss and not compute_metrics:
        return {}, {}

    model.eval()
    amp_enabled = get_amp_enabled(device, use_amp)

    total_loss_sum = torch.zeros((), device=device)
    total_bbox_sum = torch.zeros((), device=device)
    total_giou_sum = torch.zeros((), device=device)
    total_score_sum = torch.zeros((), device=device)
    total_rank_contrib_sum = torch.zeros((), device=device)
    total_rank_raw_sum = torch.zeros((), device=device)
    total_text_negative_sum = torch.zeros((), device=device)
    total_text_negative_queries = 0

    metric = (
        BinaryDetectionAPAccumulator(iou_thresholds=iou_thresholds)
        if compute_metrics
        else None
    )

    sample_count = 0
    skipped_empty_gt = 0
    total_selected = 0
    processed_batches = 0

    pbar_total = (
        len(val_loader)
        if max_val_batches is None
        else min(len(val_loader), int(max_val_batches))
    )

    labels = []
    if compute_loss:
        labels.append("Loss")
    if compute_metrics:
        labels.append("Eval")

    pbar = make_progress_bar(
        enumerate(val_loader),
        total=pbar_total,
        desc=f"Epoch {epoch} [Val {'+'.join(labels)}]",
        leave=progress_leave,
        mininterval=progress_mininterval,
    )

    validation_start = time.perf_counter()

    for step, batch in pbar:
        if max_val_batches is not None and step >= int(max_val_batches):
            break

        processed_batches += 1

        images, query_texts, image_indices = prepare_model_batch(
            batch=batch,
            device=device,
            channels_last=channels_last,
        )

        targets_device = (
            move_targets_to_device(batch, device)
            if compute_loss
            else None
        )

        gt_boxes_cpu = (
            get_target_boxes_cpu(batch)
            if compute_metrics
            else None
        )

        with autocast(
            device_type=device.type,
            enabled=amp_enabled,
            dtype=amp_dtype if amp_enabled else None,
        ):
            outputs = forward_model_batch(
                model=model,
                images=images,
                query_texts=query_texts,
                image_indices=image_indices,
            )

            pred_bbox = outputs["bbox"]
            pred_score_logit = get_score_logit(outputs)

            if compute_loss:
                loss, loss_dict = criterion(
                    pred_bbox=pred_bbox,
                    pred_score_logit=pred_score_logit,
                    targets=targets_device,
                    lambda_bbox=lambda_bbox,
                    lambda_giou=lambda_giou,
                    lambda_score=lambda_score,
                    lambda_rank=lambda_rank,
                    pos_weight=pos_weight,
                    current_epoch=epoch,
                    quality_warmup_epoch=quality_warmup_epoch,
                    rank_start_epoch=rank_start_epoch,
                    rank_warmup_epoch=rank_warmup_epoch,
                    rank_alpha_min=rank_alpha_min,
                    query_loss_weights=batch.get("query_loss_weights"),
                    text_negative_mask=batch.get("text_negative_mask"),
                )

        if compute_loss:
            zero = pred_bbox.new_zeros(())

            total_loss_sum.add_(loss.detach())
            total_bbox_sum.add_(loss_dict["loss_bbox"].detach())
            total_giou_sum.add_(loss_dict["loss_giou"].detach())
            total_score_sum.add_(loss_dict["loss_score"].detach())
            total_rank_contrib_sum.add_(
                loss_dict.get(
                    "loss_rank_contrib",
                    loss_dict.get("loss_rank", zero),
                ).detach()
            )
            total_rank_raw_sum.add_(
                loss_dict.get("loss_rank_raw", zero).detach()
            )
            total_text_negative_sum.add_(
                loss_dict.get("loss_text_negative", zero).detach()
            )
            total_text_negative_queries += int(
                batch.get(
                    "text_negative_mask",
                    torch.zeros(0, dtype=torch.bool),
                ).sum().item()
            )

        if compute_metrics and metric is not None and gt_boxes_cpu is not None:
            pred_scores = pred_score_logit.sigmoid()

            if pred_scores.ndim == 3:
                pred_scores = pred_scores.squeeze(-1)

            selected_batch = select_predictions_batch(
                boxes=pred_bbox.detach(),
                scores=pred_scores.detach(),
                score_thr=score_thr,
                top_k=top_k,
                nms_iou_thr=nms_iou_thr,
                use_topk_fallback=use_topk_fallback,
                use_nms=use_nms,
            )

            for (selected_boxes, selected_scores), gt_boxes in zip(
                selected_batch,
                gt_boxes_cpu,
            ):
                if gt_boxes.numel() == 0:
                    skipped_empty_gt += 1

                total_selected += int(selected_boxes.shape[0])
                sample_count += 1

                metric.update(
                    pred_boxes=selected_boxes,
                    pred_scores=selected_scores,
                    gt_boxes=gt_boxes,
                )

        should_log = (
            (step + 1) % max(1, int(log_interval)) == 0
            or (step + 1) == pbar_total
        )

        if should_log:
            postfix: Dict[str, Any] = {}

            if compute_loss:
                postfix["loss"] = (
                    f"{float((total_loss_sum / processed_batches).item()):.4f}"
                )

            if compute_metrics and metric is not None:
                postfix.update({
                    "samples": sample_count,
                    "pred": metric.num_pred,
                    "sel/img": f"{total_selected / max(1, sample_count):.2f}",
                    "skip_empty": skipped_empty_gt,
                })

            pbar.set_postfix(postfix)
            
    pbar.refresh()
    pbar.close()
    validation_loop_time = time.perf_counter() - validation_start

    val_loss_metrics: Dict[str, float] = {}
    eval_metrics: Dict[str, float] = {}

    if compute_loss:
        denominator = max(1, processed_batches)
        val_loss_metrics = {
            "val_loss": float((total_loss_sum / denominator).item()),
            "val_loss_bbox": float((total_bbox_sum / denominator).item()),
            "val_loss_giou": float((total_giou_sum / denominator).item()),
            "val_loss_score": float((total_score_sum / denominator).item()),
            "val_loss_rank_raw": float((total_rank_raw_sum / denominator).item()),
            "val_loss_rank": float(
                (total_rank_contrib_sum / denominator).item()
            ),
            "val_loss_text_negative": float(
                (total_text_negative_sum / denominator).item()
            ),
            "val_text_negative_queries": int(total_text_negative_queries),
        }

    if compute_metrics and metric is not None:
        tqdm.write(
            f"[Eval Metric] Start: samples={sample_count}, "
            f"pred={metric.num_pred}, gt={metric.num_gt}"
        )

        metric_start = time.perf_counter()
        eval_metrics = metric.compute()
        metric_time = time.perf_counter() - metric_start

        eval_metrics.update({
            "valid_samples": sample_count,
            "skipped_empty_gt": skipped_empty_gt,
            "avg_selected_per_sample": (
                total_selected / max(1, sample_count)
            ),
            "eval_loop_time": validation_loop_time,
            "eval_metric_time": metric_time,
        })

        tqdm.write(
            f"[Eval Timing] loop={validation_loop_time:.2f}s "
            f"metric={metric_time:.2f}s"
        )
        tqdm.write(
            f"Eval Epoch [{epoch}] "
            f"mAP50={eval_metrics['map50']:.4f} "
            f"mAP50-95={eval_metrics['map50_95']:.4f} "
            f"P={eval_metrics['precision']:.4f} "
            f"R={eval_metrics['recall']:.4f} "
            f"TP={eval_metrics['tp']} "
            f"FP={eval_metrics['fp']} "
            f"GT={eval_metrics['num_gt']} "
            f"Pred={eval_metrics['num_pred']} "
            f"skip_empty={eval_metrics['skipped_empty_gt']}"
        )

    return val_loss_metrics, eval_metrics


# -----------------------------------------------------------------------------
# Metrics logging
# -----------------------------------------------------------------------------


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_latest_json(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = path + ".tmp"

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(row, file, ensure_ascii=False, indent=2)

    os.replace(temporary_path, path)


def append_csv(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    fieldnames = list(row.keys())

    with open(path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


# -----------------------------------------------------------------------------
# Checkpoint
# -----------------------------------------------------------------------------


def build_checkpoint(
    model: torch.nn.Module,
    ema: Optional[ModelEMA],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: WarmupCosineScheduler,
    epoch: int,
    best_metric: float,
    best_metric_name: str,
    train_metrics: Dict[str, Any],
    val_loss_metrics: Dict[str, Any],
    eval_metrics: Dict[str, Any],
    train_config: Dict[str, Any],
    dynamic_config: Dict[str, Any],
) -> Dict[str, Any]:
    model = unwrap_model(model)

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
    resume_path: str,
    model: torch.nn.Module,
    ema: Optional[ModelEMA],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: WarmupCosineScheduler,
    device: torch.device,
) -> Tuple[int, float]:
    try:
        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(resume_path, map_location=device)

    model = unwrap_model(model)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    if ema is not None:
        if checkpoint.get("ema") is not None:
            ema.ema.load_state_dict(checkpoint["ema"], strict=True)
        else:
            # Backward compatibility: older checkpoints may not contain EMA.
            # Initialize EMA from the just-loaded model instead of leaving the
            # pre-checkpoint random initialization in the evaluation branch.
            ema.ema.load_state_dict(model.state_dict(), strict=True)
            print(
                "[Checkpoint] EMA state missing; initialized EMA from model."
            )

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    if "scheduler_step" in checkpoint:
        scheduler.step_num = int(checkpoint["scheduler_step"])

    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_metric = float(checkpoint.get("best_metric", -1.0))

    return start_epoch, best_metric


def save_checkpoint(
    checkpoint: Dict[str, Any],
    path: str,
) -> float:
    start = time.perf_counter()
    temporary_path = path + ".tmp"

    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)

    return time.perf_counter() - start


# -----------------------------------------------------------------------------
# BERT cache
# -----------------------------------------------------------------------------


def collect_query_texts_from_datasets(*datasets: Any) -> List[str]:
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

            text = sample.get("query_text")

            if text is None:
                continue

            text = str(text).strip()

            if text:
                texts.add(text)

    return sorted(texts)


def ensure_precomputed_bert_raw_cache(
    cache_path: Optional[str],
    datasets: Sequence[Any],
    device: torch.device,
    hidden_dim: int = 512,
    max_length: int = 32,
    batch_size: int = 128,
    enabled: bool = True,
) -> Optional[str]:
    """
    Build or incrementally extend the frozen-BERT raw cache.

    Negative-query JSON may introduce new texts after a cache already exists.
    The old implementation skipped every existing cache file, which could leave
    new negative queries missing and cause a lookup failure during training.
    """
    if not enabled:
        print("[BERT Precompute] Skip because freeze_bert=False")
        return None

    if cache_path is None:
        print("[BERT Precompute] Skip because precomputed_bert_path=None")
        return None

    cache_path = os.path.abspath(str(cache_path))
    texts = collect_query_texts_from_datasets(*datasets)

    if not texts:
        raise RuntimeError(
            "[BERT Precompute] No query_text found from dataloader datasets."
        )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    cache: Dict[str, Dict[str, torch.Tensor]] = {}
    cache_metadata: Dict[str, Any] = {}

    if os.path.exists(cache_path):
        try:
            try:
                cached_object = torch.load(
                    cache_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                cached_object = torch.load(cache_path, map_location="cpu")

            if isinstance(cached_object, dict):
                loaded_cache = cached_object.get("cache", {})
                if isinstance(loaded_cache, dict):
                    cache = loaded_cache
                    cache_metadata = cached_object
        except Exception as error:
            print(
                "[BERT Precompute] Existing cache is invalid; rebuild it: "
                f"{error}"
            )
            cache = {}
            cache_metadata = {}

    missing_texts = [text for text in texts if text not in cache]

    if not missing_texts:
        print(
            f"[BERT Precompute] Cache ready: {cache_path}, "
            f"texts={len(cache)}"
        )
        return cache_path

    action = "Extending" if cache else "Building"
    print(f"[BERT Precompute] {action}: {cache_path}")
    print(
        f"[BERT Precompute] required={len(texts)}, "
        f"cached={len(cache)}, missing={len(missing_texts)}"
    )

    bert = Bert(
        out_dim=hidden_dim,
        max_length=max_length,
        freeze_bert=True,
        precomputed_bert_path=None,
    ).to(device)
    bert.eval()

    hidden = None

    with torch.inference_mode():
        for index in tqdm(
            range(0, len(missing_texts), batch_size),
            desc="[BERT Precompute]",
            dynamic_ncols=True,
        ):
            batch_texts = missing_texts[index:index + batch_size]
            encoded = bert.encode_raw(batch_texts, device=device)

            hidden = encoded["last_hidden_state"].detach().cpu().half()
            mask = encoded["attention_mask"].detach().cpu()

            for offset, text_value in enumerate(batch_texts):
                cache[text_value] = {
                    "last_hidden_state": hidden[offset],
                    "attention_mask": mask[offset],
                }

    if hidden is None:
        raise RuntimeError("BERT cache generation produced no output")

    hidden_size = int(hidden.shape[-1])
    if cache_metadata.get("hidden_size") is not None:
        old_hidden_size = int(cache_metadata["hidden_size"])
        if old_hidden_size != hidden_size:
            raise RuntimeError(
                "BERT cache hidden-size mismatch: "
                f"existing={old_hidden_size}, generated={hidden_size}"
            )

    temporary_path = cache_path + ".tmp"
    torch.save(
        {
            "type": "bert_raw_cache",
            "max_length": int(max_length),
            "hidden_size": hidden_size,
            "num_texts": len(cache),
            "cache": cache,
        },
        temporary_path,
    )
    os.replace(temporary_path, cache_path)

    del bert
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[BERT Precompute] Done. Saved {len(cache)} texts to {cache_path}")
    return cache_path


# -----------------------------------------------------------------------------
# DataLoader construction
# -----------------------------------------------------------------------------


def _make_runtime_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    persistent_workers: bool,
    query_budget: bool,
    shuffle: bool,
    drop_last: bool,
    seed: int,
) -> DataLoader:
    num_workers = max(0, int(num_workers))
    prefetch_factor = max(1, int(prefetch_factor))

    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "num_workers": num_workers,
        "collate_fn": compact_grounding_collate_fn,
        "pin_memory": bool(pin_memory),
        "worker_init_fn": seed_dataloader_worker,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = prefetch_factor

    if getattr(dataset, "image_level_batching", False) and query_budget:
        kwargs["batch_sampler"] = QueryBudgetBatchSampler(
            dataset=dataset,
            query_budget=int(batch_size),
            shuffle=bool(shuffle),
            drop_last=bool(drop_last),
            seed=int(seed),
        )
    else:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        kwargs.update({
            "batch_size": int(batch_size),
            "shuffle": bool(shuffle),
            "drop_last": bool(drop_last),
            "generator": generator,
        })

    return DataLoader(**kwargs)


def build_dataloaders_with_supported_options(args: SimpleNamespace, **paths: str):
    """
    Build datasets/cache through the project data module, then replace its
    DataLoaders with compact IPC loaders.

    The project collate returns many small target tensors. With 16 workers and
    prefetch=4 this can create thousands of shared storages. The compact loader
    packs all GT boxes into one tensor and uses separate Train/Val worker plans.
    """
    base_kwargs = {
        **paths,
        "batch_size": args.batch_size,
        "image_size": (args.image_size, args.image_size),
        # Dataset/cache construction does not require multiprocessing. Runtime
        # loaders are created below with explicit safe settings.
        "num_workers": 0,
        "max_text_aug_per_image": args.max_text_aug_per_image,
        "random_seed": args.seed,
        "cache_images": args.cache_images,
        "image_cache_dir": args.image_cache_dir,
        "prebuild_image_cache": args.prebuild_image_cache,
        "prefetch_factor": 1,
        "pin_memory": False,
        "cache_workers": args.cache_workers,
        "query_budget_batching": args.query_budget,
        "negative_query_path": args.negative_query_path,
        "negative_sample_ratio": args.negative_sample_ratio,
        "use_negative_queries_in_val": args.use_negative_queries_in_val,
    }

    signature = inspect.signature(build_dataloaders)
    supported = {
        key: value
        for key, value in base_kwargs.items()
        if key in signature.parameters
    }

    base_train_loader, base_val_loader = build_dataloaders(**supported)

    requested_workers = max(0, int(args.num_workers))
    cpu_count = max(1, int(os.cpu_count() or 1))
    requested_workers = min(requested_workers, cpu_count)

    soft_fd, _ = resource.getrlimit(resource.RLIMIT_NOFILE)

    # Keep the requested Train workers when the process limit is healthy.
    # Validation uses a smaller non-persistent pool because Train persistent
    # workers remain alive during evaluation.
    train_workers = requested_workers
    val_workers = 0 if requested_workers == 0 else min(4, max(1, requested_workers // 4))

    if soft_fd != resource.RLIM_INFINITY and soft_fd < 4096:
        train_workers = min(train_workers, 8)
        val_workers = min(val_workers, 2)

    # prefetch_factor is per worker. 16 x 4 means 64 queued batches and is not
    # useful for cached 512x512 uint8 images. Two queued batches per Train worker
    # normally saturate H2D while keeping shared memory bounded.
    train_prefetch = min(max(1, int(args.prefetch_factor)), 2)
    val_prefetch = 1

    train_persistent = bool(args.persistent_workers and train_workers > 0)
    val_persistent = False

    train_loader = _make_runtime_dataloader(
        base_train_loader.dataset,
        batch_size=args.batch_size,
        num_workers=train_workers,
        prefetch_factor=train_prefetch,
        pin_memory=args.pin_memory,
        persistent_workers=train_persistent,
        query_budget=args.query_budget,
        shuffle=True,
        drop_last=True,
        seed=args.seed,
    )

    val_loader = _make_runtime_dataloader(
        base_val_loader.dataset,
        batch_size=args.batch_size,
        num_workers=val_workers,
        prefetch_factor=val_prefetch,
        pin_memory=args.pin_memory,
        persistent_workers=val_persistent,
        query_budget=args.query_budget,
        shuffle=False,
        drop_last=False,
        seed=args.seed,
    )

    args.effective_train_workers = train_workers
    args.effective_val_workers = val_workers
    args.effective_train_prefetch = train_prefetch
    args.effective_val_prefetch = val_prefetch
    args.effective_train_persistent = train_persistent
    args.effective_val_persistent = val_persistent

    return train_loader, val_loader


# -----------------------------------------------------------------------------
# Main training
# -----------------------------------------------------------------------------


def train(args: SimpleNamespace) -> None:
    set_seed(args.seed)
    fd_soft_limit, fd_hard_limit = configure_process_file_limit()
    configure_torch_runtime(args)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    amp_dtype = resolve_amp_dtype_for_device(
        device=device,
        requested_dtype=parse_amp_dtype(args.amp_dtype),
    )

    if amp_dtype == torch.float32:
        args.use_amp = False
        args.amp_dtype = "fp32"
    elif amp_dtype == torch.float16:
        args.amp_dtype = "fp16"
    elif amp_dtype == torch.bfloat16:
        args.amp_dtype = "bf16"

    dataset_dir = args.dir

    train_image_dir = os.path.join(dataset_dir, "images", "train")
    train_anno_dir = os.path.join(dataset_dir, "labels", "train")
    val_image_dir = os.path.join(dataset_dir, "images", "val")
    val_anno_dir = os.path.join(dataset_dir, "labels", "val")

    train_loader, val_loader = build_dataloaders_with_supported_options(
        args,
        train_image_dir=train_image_dir,
        train_anno_dir=train_anno_dir,
        val_image_dir=val_image_dir,
        val_anno_dir=val_anno_dir,
    )

    args.precomputed_bert_path = ensure_precomputed_bert_raw_cache(
        cache_path=args.precomputed_bert_path,
        datasets=[train_loader.dataset, val_loader.dataset],
        device=device,
        hidden_dim=args.hidden_dim,
        max_length=args.text_max_length,
        batch_size=max(128, args.batch_size),
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

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    ema = (
        ModelEMA(
            model,
            decay=args.ema_decay,
            update_interval=args.ema_update_interval,
            buffer_update_interval=args.ema_buffer_update_interval,
        )
        if args.use_ema
        else None
    )

    criterion = GroundingLoss(
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou,
        cost_score=args.cost_score,
        hard_negative_ratio=args.hard_negative_ratio,
        positive_ratio=args.positive_ratio,
        max_positive_per_gt=args.max_positive_per_gt,
        aux_positive_label=args.aux_positive_label,
        expand_cost_bbox=args.expand_cost_bbox,
        expand_cost_giou=args.expand_cost_giou,
        iou_pos_thr=args.iou_pos_thr,
        quality_min=args.quality_min,
        quality_max=args.quality_max,
        qfl_beta=args.qfl_beta,
        rank_margin=args.rank_margin,
        rank_min_quality_gap=args.rank_min_quality_gap,
        rank_max_pairs=args.rank_max_pairs,
        max_query_loss_weight=args.max_query_loss_weight,
    )

    if args.startup_smoke_test:
        run_startup_smoke_test(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            device=device,
            use_amp=args.use_amp,
            amp_dtype=amp_dtype,
            channels_last=args.channels_last,
            total_epochs=args.epochs,
            args=args,
        )

    optimizer = build_optimizer(
        model=model,
        lr_vision=args.lr_vision,
        lr_text=args.lr_text,
        lr_transformer=args.lr_transformer,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        fused=args.fused_optimizer,
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(
        args.max_warmup_steps,
        args.warmup_epochs * len(train_loader),
    )

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    scaler = build_scaler(
        device=device,
        use_amp=args.use_amp,
        amp_dtype=amp_dtype,
    )

    train_model: torch.nn.Module = model

    if args.compile_model:
        if not hasattr(torch, "compile"):
            print("[Warning] torch.compile is unavailable; continue without compile.")
        else:
            try:
                train_model = torch.compile(
                    model,
                    mode=args.compile_mode,
                    dynamic=False,
                )
                print(
                    f"[Info] torch.compile enabled: mode={args.compile_mode}"
                )
            except Exception as error:
                print(
                    f"[Warning] torch.compile failed, fallback to eager: {error}"
                )
                train_model = model

    total_params, trainable_params = count_parameters(model)

    print(f"[Info] Device: {device}")
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device)
        bf16_supported = (
            torch.cuda.is_bf16_supported()
            if hasattr(torch.cuda, "is_bf16_supported")
            else "unknown"
        )
        print(
            f"[Info] GPU: {torch.cuda.get_device_name(device)}, "
            f"compute_capability={capability[0]}.{capability[1]}, "
            f"bf16_supported={bf16_supported}"
        )
    print(
        f"[Info] AMP: enabled={get_amp_enabled(device, args.use_amp)}, "
        f"dtype={args.amp_dtype}, scaler={scaler.is_enabled()}"
    )
    print(
        f"[Info] TF32: allow={args.allow_tf32}, "
        f"matmul_precision={args.matmul_precision}"
    )
    print(
        f"[Info] DataLoader requested: pin_memory={args.pin_memory}, "
        f"workers={args.num_workers}, prefetch={args.prefetch_factor}, "
        f"persistent_workers={args.persistent_workers}, "
        f"query_budget={args.query_budget}"
    )
    print(
        f"[Info] IPC: sharing_strategy={mp.get_sharing_strategy()}, "
        f"RLIMIT_NOFILE=({fd_soft_limit}, {fd_hard_limit})"
    )
    print(
        f"[Info] DataLoader actual Train: "
        f"pin_memory={train_loader.pin_memory}, "
        f"workers={train_loader.num_workers}, "
        f"prefetch={train_loader.prefetch_factor}, "
        f"persistent_workers={train_loader.persistent_workers}, "
        f"compact_targets=True"
    )
    print(
        f"[Info] DataLoader actual Val: "
        f"pin_memory={val_loader.pin_memory}, "
        f"workers={val_loader.num_workers}, "
        f"prefetch={val_loader.prefetch_factor}, "
        f"persistent_workers={val_loader.persistent_workers}, "
        f"compact_targets=True"
    )
    print(
        f"[Info] Optimizer groups={len(optimizer.param_groups)}, "
        f"fused_requested={args.fused_optimizer}"
    )
    print(f"[Info] Model parameters: total={total_params}, trainable={trainable_params}")
    print(f"[Info] Train batches: {len(train_loader)}")
    print(f"[Info] Val batches: {len(val_loader)}")
    print(
        f"[Info] Text negatives: path={args.negative_query_path}, "
        f"ratio={args.negative_sample_ratio:.4f}, "
        f"val={args.use_negative_queries_in_val}, "
        f"max_loss_weight={args.max_query_loss_weight:.2f}"
    )
    print(
        f"[Info] EMA: enabled={args.use_ema}, decay={args.ema_decay}, "
        f"update_interval={args.ema_update_interval}"
    )

    if args.resume_path is not None:
        save_path = os.path.dirname(os.path.abspath(args.resume_path))
    elif args.save_dir is None:
        save_path = os.path.join(
            "checkpoints",
            f"results_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
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
        epoch_start = time.perf_counter()

        lambda_bbox, lambda_giou, lambda_score = get_loss_weights(
            epoch=epoch,
            total_epochs=args.epochs,
            args=args,
        )

        pos_weight = float(args.pos_weight)

        train_metrics = train_one_epoch(
            model=train_model,
            ema_source_model=model,
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
            amp_dtype=amp_dtype,
            channels_last=args.channels_last,
            grad_clip_norm=args.grad_clip_norm,
            lambda_bbox=lambda_bbox,
            lambda_giou=lambda_giou,
            lambda_score=lambda_score,
            lambda_rank=args.lambda_rank,
            pos_weight=pos_weight,
            quality_warmup_epoch=args.quality_warmup_epoch,
            rank_start_epoch=args.rank_start_epoch,
            rank_warmup_epoch=args.rank_warmup_epoch,
            rank_alpha_min=args.rank_alpha_min,
            log_interval=args.log_interval,
            step_metrics_path=(
                step_metrics_path if args.emit_step_metrics else None
            ),
            progress_leave=args.progress_leave,
            progress_mininterval=args.progress_mininterval,
        )

        run_val_loss = (
            args.val_loss_interval > 0
            and epoch % args.val_loss_interval == 0
        )
        run_eval = (
            args.eval_interval > 0
            and epoch % args.eval_interval == 0
        )

        eval_model = ema.ema if ema is not None else model

        val_loss_metrics, eval_metrics = validate_one_epoch(
            model=eval_model,
            criterion=criterion,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            compute_loss=run_val_loss,
            compute_metrics=run_eval,
            use_amp=args.use_amp,
            amp_dtype=amp_dtype,
            channels_last=args.channels_last,
            lambda_bbox=lambda_bbox,
            lambda_giou=lambda_giou,
            lambda_score=lambda_score,
            lambda_rank=args.lambda_rank,
            pos_weight=pos_weight,
            quality_warmup_epoch=args.quality_warmup_epoch,
            rank_start_epoch=args.rank_start_epoch,
            rank_warmup_epoch=args.rank_warmup_epoch,
            rank_alpha_min=args.rank_alpha_min,
            score_thr=args.score_thr,
            top_k=args.top_k,
            nms_iou_thr=args.nms_iou_thr,
            use_topk_fallback=args.use_topk_fallback,
            use_nms=args.use_nms,
            iou_thresholds=args.iou_thresholds,
            max_val_batches=args.max_val_batches,
            log_interval=args.log_interval,
            progress_leave=args.progress_leave,
            progress_mininterval=args.progress_mininterval,
        )

        dynamic_config = {
            "lambda_bbox": lambda_bbox,
            "lambda_giou": lambda_giou,
            "lambda_score": lambda_score,
            "lambda_rank_max": float(args.lambda_rank),
            "pos_weight": pos_weight,
            "quality_warmup_epoch": int(args.quality_warmup_epoch),
            "rank_start_epoch": int(args.rank_start_epoch),
            "rank_warmup_epoch": int(args.rank_warmup_epoch),
            "rank_alpha_min": float(args.rank_alpha_min),
            "negative_sample_ratio": float(args.negative_sample_ratio),
            "max_query_loss_weight": float(args.max_query_loss_weight),
            "use_negative_queries_in_val": bool(
                args.use_negative_queries_in_val
            ),
        }

        epoch_time = time.perf_counter() - epoch_start

        metric_row = {
            "type": "epoch",
            "time": time.time(),
            "epoch": epoch,
            "epoch_time": epoch_time,
            "lr": scheduler.get_lr()[0],
            **train_metrics,
            **val_loss_metrics,
            **eval_metrics,
            **dynamic_config,
        }

        append_jsonl(epoch_metrics_path, metric_row)
        append_csv(csv_metrics_path, metric_row)
        write_latest_json(latest_metrics_path, metric_row)

        has_eval_metric = best_metric_name in eval_metrics
        save_metric = (
            float(eval_metrics[best_metric_name])
            if has_eval_metric
            else -1.0
        )

        is_best = bool(has_eval_metric and save_metric > best_metric)

        if is_best:
            best_metric = save_metric

        should_save_latest = (
            args.save_latest_interval > 0
            and epoch % args.save_latest_interval == 0
        )
        should_save_epoch = (
            args.save_epoch_interval > 0
            and epoch % args.save_epoch_interval == 0
        )

        if should_save_latest or should_save_epoch or is_best:
            checkpoint = build_checkpoint(
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

            if should_save_latest:
                latest_path = os.path.join(save_path, "latest.pt")
                elapsed = save_checkpoint(checkpoint, latest_path)
                tqdm.write(
                    f"[Checkpoint] latest.pt saved in {elapsed:.2f}s"
                )

            if is_best:
                best_path = os.path.join(
                    save_path,
                    f"best_{best_metric_name}.pt",
                )
                elapsed = save_checkpoint(checkpoint, best_path)
                tqdm.write(
                    f"Saved best checkpoint: epoch={epoch}, "
                    f"{best_metric_name}={best_metric:.4f}, "
                    f"time={elapsed:.2f}s"
                )

            if should_save_epoch:
                epoch_path = os.path.join(
                    save_path,
                    f"epoch_{epoch:03d}.pt",
                )
                elapsed = save_checkpoint(checkpoint, epoch_path)
                tqdm.write(
                    f"[Checkpoint] epoch_{epoch:03d}.pt saved in "
                    f"{elapsed:.2f}s"
                )

        val_loss_text = (
            f"{val_loss_metrics['val_loss']:.4f}"
            if "val_loss" in val_loss_metrics
            else "skip"
        )

        tqdm.write(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_metrics.get('train_loss', -1):.4f} "
            f"val_loss={val_loss_text} "
            f"mAP50={eval_metrics.get('map50', -1):.4f} "
            f"mAP50-95={eval_metrics.get('map50_95', -1):.4f} "
            f"P={eval_metrics.get('precision', -1):.4f} "
            f"R={eval_metrics.get('recall', -1):.4f} "
            f"lb={lambda_bbox:.2f} "
            f"lg={lambda_giou:.2f} "
            f"ls={lambda_score:.2f} "
            f"lrk_max={args.lambda_rank:.2f} "
            f"pw={pos_weight:.2f} "
            f"epoch_time={epoch_time:.2f}s"
        )


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


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
        "cache_images": False,
        "image_cache_dir": None,
        "prebuild_image_cache": False,
        "prefetch_factor": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "query_budget": True,
        "cache_workers": 8,
        "negative_query_path": os.path.join(
            CURRENT_DIR,
            "cards",
            "cache",
            "negative_query_pool.json",
        ),
        "negative_sample_ratio": 0.05,
        "use_negative_queries_in_val": False,
    },
    "train": {
        "epochs": 300,
        "batch_size": 12,
        "warmup_epochs": 5,
        "num_workers": 8,
        "device": "cuda:1",
        "seed": 42,
        "deterministic": False,
        "use_amp": True,
        "amp_dtype": "bf16",
        "use_ema": True,
        "ema_decay": 0.999,
        "ema_update_interval": 1,
        "ema_buffer_update_interval": 1,
        "grad_clip_norm": 1.0,
        "allow_tf32": True,
        "matmul_precision": "high",
        "channels_last": False,
        "compile": False,
        "compile_mode": "reduce-overhead",
        "startup_smoke_test": True,
    },
    "optim": {
        "lr_vision": 1e-4,
        "lr_text": 1e-5,
        "lr_transformer": 1e-4,
        "lr_head": 1e-4,
        "weight_decay": 1e-4,
        "min_lr_ratio": 0.05,
        "max_warmup_steps": 3000,
        "fused": True,
    },
    "loss": {
        "matcher": {
            "cost_bbox": 5.0,
            "cost_giou": 2.0,
            "cost_score": 0.0,
        },
        "score_sampling": {
            "hard_negative_ratio": 5,
            "positive_ratio": 0.05,
            "max_positive_per_gt": 2,
            "aux_positive_label": 0.7,
            "expand_cost_bbox": 5.0,
            "expand_cost_giou": 2.0,
        },
        "quality": {
            "iou_pos_thr": 0.15,
            "quality_min": 0.25,
            "quality_max": 1.0,
            "qfl_beta": 2.0,
            "quality_warmup_epoch": 20,
        },
        "ranking": {
            "lambda_rank": 0.10,
            "rank_margin": 0.1,
            "rank_min_quality_gap": 0.1,
            "rank_max_pairs": 512,
            "rank_start_epoch": 15,
            "rank_warmup_epoch": 30,
            "rank_alpha_min": 0.0,
        },
        "text_negative": {
            "max_query_loss_weight": 10.0,
        },
        "weight": {
            "dynamic": True,
            "bbox": 5.0,
            "giou": 2.0,
            "score": 2.0,
            "bbox_start": 5.0,
            "bbox_end": 3.0,
            "bbox_decay_until": 0.5,
            "score_start": 2.0,
            "score_end": 4.0,
            "score_warm_until": 0.4,
        },
        "pos_weight": {
            "value": 1.0,
        },
    },
    "eval": {
        "val_loss_interval": 5,
        "eval_interval": 1,
        "max_val_batches": None,
        "score_thr": 0.001,
        "top_k": 20,
        "nms_iou_thr": 0.5,
        "use_nms": True,
        "iou_thresholds": None,
        "best_metric": "map50_95",
        "use_topk_fallback": False,
    },
    "log": {
        "save_dir": None,
        "resume_path": None,
        "save_latest_interval": 1,
        "save_epoch_interval": 50,
        "emit_step_metrics": False,
        "log_interval": 50,
        "progress_leave": True,
        "progress_mininterval": 0.5,
    },
}


def deepcopy_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(cfg)


def deep_update(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
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


def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}

    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML config not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(f"YAML config must be a dict: {path}")

    return config


def load_model_config(path: Optional[str]) -> Dict[str, Any]:
    config = deepcopy_cfg(DEFAULT_MODEL_CFG)
    return deep_update(config, load_yaml(path))


def load_train_config(path: Optional[str]) -> Dict[str, Any]:
    config = deepcopy_cfg(DEFAULT_TRAIN_CFG)
    return deep_update(config, load_yaml(path))


def normalize_device(device: Any) -> str:
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


def cfg_to_args(
    model_cfg_all: Dict[str, Any],
    train_cfg_all: Dict[str, Any],
) -> SimpleNamespace:
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
    score_sampling_cfg = loss_cfg.get("score_sampling", {})
    quality_cfg = loss_cfg.get("quality", {})
    ranking_cfg = loss_cfg.get("ranking", {})
    text_negative_cfg = loss_cfg.get("text_negative", {})

    return SimpleNamespace(
        # data
        dir=data_cfg["dataset_dir"],
        image_size=data_cfg["image_size"],
        max_text_aug_per_image=data_cfg["max_text_aug_per_image"],
        cache_images=data_cfg.get("cache_images", False),
        image_cache_dir=data_cfg.get("image_cache_dir"),
        prebuild_image_cache=data_cfg.get("prebuild_image_cache", False),
        prefetch_factor=data_cfg.get("prefetch_factor", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=data_cfg.get("persistent_workers", True),
        query_budget=data_cfg.get("query_budget", True),
        cache_workers=data_cfg.get("cache_workers", 8),
        negative_query_path=data_cfg.get("negative_query_path"),
        negative_sample_ratio=float(
            data_cfg.get("negative_sample_ratio", 0.05)
        ),
        use_negative_queries_in_val=bool(
            data_cfg.get("use_negative_queries_in_val", False)
        ),

        # train
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        warmup_epochs=train_cfg["warmup_epochs"],
        num_workers=train_cfg["num_workers"],
        device=train_cfg["device"],
        seed=train_cfg["seed"],
        deterministic=train_cfg["deterministic"],
        use_amp=train_cfg["use_amp"],
        amp_dtype=train_cfg.get("amp_dtype", "bf16"),
        use_ema=train_cfg["use_ema"],
        ema_decay=train_cfg["ema_decay"],
        ema_update_interval=train_cfg.get("ema_update_interval", 1),
        ema_buffer_update_interval=train_cfg.get(
            "ema_buffer_update_interval",
            1,
        ),
        grad_clip_norm=train_cfg["grad_clip_norm"],
        allow_tf32=train_cfg.get("allow_tf32", True),
        matmul_precision=train_cfg.get("matmul_precision", "high"),
        channels_last=train_cfg.get("channels_last", False),
        compile_model=train_cfg.get("compile", False),
        compile_mode=train_cfg.get("compile_mode", "reduce-overhead"),
        startup_smoke_test=train_cfg.get("startup_smoke_test", True),

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
        precomputed_bert_path=model_cfg.get("precomputed_bert_path"),

        # optimizer
        lr_vision=optim_cfg["lr_vision"],
        lr_text=optim_cfg["lr_text"],
        lr_transformer=optim_cfg["lr_transformer"],
        lr_head=optim_cfg["lr_head"],
        weight_decay=optim_cfg["weight_decay"],
        min_lr_ratio=optim_cfg["min_lr_ratio"],
        max_warmup_steps=optim_cfg["max_warmup_steps"],
        fused_optimizer=optim_cfg.get("fused", True),

        # matcher
        cost_bbox=matcher_cfg.get("cost_bbox", 5.0),
        cost_giou=matcher_cfg.get("cost_giou", 2.0),
        cost_score=matcher_cfg.get("cost_score", 1.0),

        # score sampling
        hard_negative_ratio=score_sampling_cfg.get("hard_negative_ratio", 5),
        positive_ratio=score_sampling_cfg.get("positive_ratio", 0.2),
        max_positive_per_gt=score_sampling_cfg.get("max_positive_per_gt", 5),
        aux_positive_label=score_sampling_cfg.get("aux_positive_label", 0.7),
        expand_cost_bbox=score_sampling_cfg.get("expand_cost_bbox", 5.0),
        expand_cost_giou=score_sampling_cfg.get("expand_cost_giou", 2.0),

        # loss weights
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

        # quality / ranking
        pos_weight=pos_weight_cfg.get("value", 1.0),
        iou_pos_thr=quality_cfg.get("iou_pos_thr", 0.15),
        quality_min=quality_cfg.get("quality_min", 0.25),
        quality_max=quality_cfg.get("quality_max", 1.0),
        qfl_beta=quality_cfg.get("qfl_beta", 2.0),
        quality_warmup_epoch=quality_cfg.get("quality_warmup_epoch", 20),
        lambda_rank=ranking_cfg.get("lambda_rank", 0.10),
        rank_margin=ranking_cfg.get("rank_margin", 0.1),
        rank_min_quality_gap=ranking_cfg.get("rank_min_quality_gap", 0.1),
        rank_max_pairs=ranking_cfg.get("rank_max_pairs", 512),
        rank_start_epoch=ranking_cfg.get("rank_start_epoch", 15),
        rank_warmup_epoch=ranking_cfg.get("rank_warmup_epoch", 30),
        rank_alpha_min=ranking_cfg.get("rank_alpha_min", 0.0),
        max_query_loss_weight=text_negative_cfg.get(
            "max_query_loss_weight",
            10.0,
        ),

        # eval
        val_loss_interval=eval_cfg["val_loss_interval"],
        eval_interval=eval_cfg["eval_interval"],
        max_val_batches=eval_cfg.get("max_val_batches"),
        score_thr=eval_cfg["score_thr"],
        top_k=eval_cfg["top_k"],
        nms_iou_thr=eval_cfg["nms_iou_thr"],
        use_nms=eval_cfg.get("use_nms", True),
        iou_thresholds=eval_cfg.get("iou_thresholds"),
        best_metric=eval_cfg["best_metric"],
        use_topk_fallback=eval_cfg.get("use_topk_fallback", False),

        # log
        save_dir=log_cfg["save_dir"],
        resume_path=log_cfg["resume_path"],
        save_latest_interval=log_cfg.get("save_latest_interval", 1),
        save_epoch_interval=log_cfg["save_epoch_interval"],
        emit_step_metrics=log_cfg["emit_step_metrics"],
        log_interval=log_cfg["log_interval"],
        progress_leave=log_cfg.get("progress_leave", False),
        progress_mininterval=log_cfg.get("progress_mininterval", 0.5),

        model_cfg=model_cfg_all,
        train_cfg=train_cfg_all,
    )


def print_config_summary(
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
) -> None:
    model = model_cfg["model"]
    data = train_cfg["data"]
    training = train_cfg["train"]
    optimizer = train_cfg["optim"]
    loss = train_cfg["loss"]
    evaluation = train_cfg["eval"]
    logging = train_cfg["log"]

    matcher = loss.get("matcher", {})
    weight = loss.get("weight", {})
    pos_weight = loss.get("pos_weight", {})
    quality = loss.get("quality", {})
    ranking = loss.get("ranking", {})
    score_sampling = loss.get("score_sampling", {})
    text_negative = loss.get("text_negative", {})

    print("\n[LightDet] Training config")
    print(f"  dataset      : {data['dataset_dir']}")
    print(f"  image_size   : {data['image_size']}")
    print(
        f"  image cache  : enabled={data.get('cache_images', False)}, "
        f"prebuild={data.get('prebuild_image_cache', False)}, "
        f"prefetch={data.get('prefetch_factor', 4)}, "
        f"dir={data.get('image_cache_dir')}"
    )
    print(
        f"  dataloader   : workers={training['num_workers']}, "
        f"pin_memory={data.get('pin_memory', True)}, "
        f"persistent={data.get('persistent_workers', True)}, "
        f"query_budget={data.get('query_budget', True)}, "
        f"cache_workers={data.get('cache_workers', 8)}"
    )
    print(
        f"  text negative: path={data.get('negative_query_path')}, "
        f"ratio={data.get('negative_sample_ratio', 0.05)}, "
        f"val={data.get('use_negative_queries_in_val', False)}, "
        f"max_weight={text_negative.get('max_query_loss_weight', 10.0)}"
    )
    print(f"  epochs       : {training['epochs']}")
    print(f"  batch        : {training['batch_size']}")
    print(f"  device       : {training['device']}")
    print(f"  seed         : {training['seed']}")
    print(f"  save_dir     : {logging['save_dir']}")
    print(
        f"  AMP          : use={training.get('use_amp', True)}, "
        f"dtype={training.get('amp_dtype', 'bf16')}"
    )
    print(
        f"  TF32         : allow={training.get('allow_tf32', True)}, "
        f"matmul={training.get('matmul_precision', 'high')}"
    )
    print(
        f"  channels_last: {training.get('channels_last', False)}"
    )
    print(
        f"  compile      : {training.get('compile', False)}, "
        f"mode={training.get('compile_mode', 'reduce-overhead')}"
    )
    print(f"  hidden_dim   : {model['hidden_dim']}")
    print(
        f"  grid         : {model['image_grid_size']}x"
        f"{model['image_grid_size']}"
    )
    print(f"  num_layers   : {model['num_layers']}")
    print(f"  num_heads    : {model['num_heads']}")
    print(f"  mlp_ratio    : {model['mlp_ratio']}")
    print(f"  bert_cache   : {model.get('precomputed_bert_path')}")
    print(f"  lr_vision    : {optimizer['lr_vision']}")
    print(f"  lr_text      : {optimizer['lr_text']}")
    print(f"  lr_trans     : {optimizer['lr_transformer']}")
    print(f"  lr_head      : {optimizer['lr_head']}")
    print(f"  fused AdamW  : {optimizer.get('fused', True)}")
    print(
        f"  matcher      : bbox={matcher.get('cost_bbox', 5.0)}, "
        f"giou={matcher.get('cost_giou', 2.0)}, "
        f"score={matcher.get('cost_score', 1.0)}"
    )
    print(
        f"  loss weight  : dynamic={weight.get('dynamic', True)}, "
        f"bbox={weight.get('bbox_start', weight.get('bbox', 5.0))}"
        f"->{weight.get('bbox_end', weight.get('bbox', 5.0))}, "
        f"giou={weight.get('giou', 2.0)}, "
        f"score={weight.get('score_start', weight.get('score', 1.0))}"
        f"->{weight.get('score_end', weight.get('score', 1.0))}"
    )
    print(
        f"  score sample : pos_ratio="
        f"{score_sampling.get('positive_ratio', 0.05)}, "
        f"max_pos/gt={score_sampling.get('max_positive_per_gt', 2)}, "
        f"neg:pos={score_sampling.get('hard_negative_ratio', 5)}:1"
    )
    print(
        f"  quality      : iou_thr={quality.get('iou_pos_thr', 0.15)}, "
        f"q=[{quality.get('quality_min', 0.25)}, "
        f"{quality.get('quality_max', 1.0)}], "
        f"warmup={quality.get('quality_warmup_epoch', 20)}"
    )
    print(
        f"  ranking      : lambda_max={ranking.get('lambda_rank', 0.10)}, "
        f"start={ranking.get('rank_start_epoch', 15)}, "
        f"warmup={ranking.get('rank_warmup_epoch', 30)}, "
        f"alpha_min={ranking.get('rank_alpha_min', 0.0)}"
    )
    print(f"  pos weight   : {pos_weight.get('value', 1.0)}")
    print(
        f"  eval         : metric={evaluation['best_metric']}, "
        f"score_thr={evaluation['score_thr']}, "
        f"top_k={evaluation['top_k']}, "
        f"use_nms={evaluation.get('use_nms', True)}, "
        f"max_batches={evaluation.get('max_val_batches')}"
    )
    print("")


class LightDet:
    def __init__(self, model: str = "cards/config/model.yaml") -> None:
        self.model_cfg_path = model
        self.model_cfg = load_model_config(model)

    def train(
        self,
        cfg: str = "cards/config/train.yaml",
        data: Optional[str] = None,
        epochs: Optional[int] = None,
        imgsz: Optional[int] = None,
        batch: Optional[int] = None,
        device: Optional[Any] = None,
        workers: Optional[int] = None,
        seed: Optional[int] = None,
        deterministic: Optional[bool] = None,
        cache_images: Optional[bool] = None,
        image_cache_dir: Optional[str] = None,
        prebuild_image_cache: Optional[bool] = None,
        prefetch_factor: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        persistent_workers: Optional[bool] = None,
        negative_query_path: Optional[str] = None,
        negative_sample_ratio: Optional[float] = None,
        use_negative_queries_in_val: Optional[bool] = None,
        project: str = "runs/train",
        name: str = "exp",
        resume: Optional[str] = None,
        lr: Optional[float] = None,
        lr_vision: Optional[float] = None,
        lr_text: Optional[float] = None,
        lr_transformer: Optional[float] = None,
        lr_head: Optional[float] = None,
        weight_decay: Optional[float] = None,
        score_thr: Optional[float] = None,
        top_k: Optional[int] = None,
        nms_iou_thr: Optional[float] = None,
        use_nms: Optional[bool] = None,
        use_topk_fallback: Optional[bool] = None,
        amp_dtype: Optional[str] = None,
        compile_model: Optional[bool] = None,
        channels_last: Optional[bool] = None,
        startup_smoke_test: Optional[bool] = None,
        use_ema: Optional[bool] = None,
        ema_decay: Optional[float] = None,
        hard_negative_ratio: Optional[int] = None,
        positive_ratio: Optional[float] = None,
        max_positive_per_gt: Optional[int] = None,
        iou_pos_thr: Optional[float] = None,
        quality_min: Optional[float] = None,
        quality_max: Optional[float] = None,
        qfl_beta: Optional[float] = None,
        quality_warmup_epoch: Optional[int] = None,
        lambda_rank: Optional[float] = None,
        rank_start_epoch: Optional[int] = None,
        rank_warmup_epoch: Optional[int] = None,
        rank_alpha_min: Optional[float] = None,
        max_query_loss_weight: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
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
        if cache_images is not None:
            train_cfg["data"]["cache_images"] = bool(cache_images)
        if image_cache_dir is not None:
            train_cfg["data"]["image_cache_dir"] = image_cache_dir
        if prebuild_image_cache is not None:
            train_cfg["data"]["prebuild_image_cache"] = bool(
                prebuild_image_cache
            )
        if prefetch_factor is not None:
            train_cfg["data"]["prefetch_factor"] = int(prefetch_factor)
        if pin_memory is not None:
            train_cfg["data"]["pin_memory"] = bool(pin_memory)
        if persistent_workers is not None:
            train_cfg["data"]["persistent_workers"] = bool(
                persistent_workers
            )
        if negative_query_path is not None:
            train_cfg["data"]["negative_query_path"] = str(
                negative_query_path
            )
        if negative_sample_ratio is not None:
            ratio = float(negative_sample_ratio)
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(
                    "negative_sample_ratio must be in [0, 1], "
                    f"got {ratio}"
                )
            train_cfg["data"]["negative_sample_ratio"] = ratio
        if use_negative_queries_in_val is not None:
            train_cfg["data"]["use_negative_queries_in_val"] = bool(
                use_negative_queries_in_val
            )
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
        if use_nms is not None:
            train_cfg["eval"]["use_nms"] = bool(use_nms)
        if use_topk_fallback is not None:
            train_cfg["eval"]["use_topk_fallback"] = bool(
                use_topk_fallback
            )

        if amp_dtype is not None:
            train_cfg["train"]["amp_dtype"] = amp_dtype
        if compile_model is not None:
            train_cfg["train"]["compile"] = bool(compile_model)
        if channels_last is not None:
            train_cfg["train"]["channels_last"] = bool(channels_last)

        if startup_smoke_test is not None:
            train_cfg["train"]["startup_smoke_test"] = bool(startup_smoke_test)
        if use_ema is not None:
            train_cfg["train"]["use_ema"] = bool(use_ema)
        if ema_decay is not None:
            decay = float(ema_decay)
            if not 0.0 <= decay < 1.0:
                raise ValueError(f"ema_decay must be in [0, 1), got {decay}")
            train_cfg["train"]["ema_decay"] = decay

        if hard_negative_ratio is not None:
            train_cfg["loss"]["score_sampling"][
                "hard_negative_ratio"
            ] = hard_negative_ratio
        if positive_ratio is not None:
            train_cfg["loss"]["score_sampling"][
                "positive_ratio"
            ] = positive_ratio
        if max_positive_per_gt is not None:
            train_cfg["loss"]["score_sampling"][
                "max_positive_per_gt"
            ] = max_positive_per_gt
        if iou_pos_thr is not None:
            train_cfg["loss"]["quality"]["iou_pos_thr"] = iou_pos_thr
        if quality_min is not None:
            train_cfg["loss"]["quality"]["quality_min"] = quality_min
        if quality_max is not None:
            train_cfg["loss"]["quality"]["quality_max"] = quality_max
        if qfl_beta is not None:
            train_cfg["loss"]["quality"]["qfl_beta"] = qfl_beta
        if quality_warmup_epoch is not None:
            train_cfg["loss"]["quality"][
                "quality_warmup_epoch"
            ] = quality_warmup_epoch
        if lambda_rank is not None:
            train_cfg["loss"]["ranking"]["lambda_rank"] = lambda_rank
        if rank_start_epoch is not None:
            train_cfg["loss"]["ranking"][
                "rank_start_epoch"
            ] = rank_start_epoch
        if rank_warmup_epoch is not None:
            train_cfg["loss"]["ranking"][
                "rank_warmup_epoch"
            ] = rank_warmup_epoch
        if rank_alpha_min is not None:
            train_cfg["loss"]["ranking"]["rank_alpha_min"] = rank_alpha_min
        if max_query_loss_weight is not None:
            train_cfg["loss"]["text_negative"][
                "max_query_loss_weight"
            ] = max(1.0, float(max_query_loss_weight))

        if kwargs:
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

        print_config_summary(model_cfg=model_cfg, train_cfg=train_cfg)
        return train(args)


def main() -> None:
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    model = LightDet(
        model=(
            "/home/soic/Desktop/LightDet/units/model/cards/config/model.yaml"
        )
    )

    model.train(
        cfg=(
            "/home/soic/Desktop/LightDet/units/model/cards/config/train.yaml"
        ),
        data="/home/soic/Desktop/LightDet/datasets",
        epochs=300,
        imgsz=512,
        batch=48,
        device=1,
        workers=16,
        seed=45,
        deterministic=False,
        use_ema=True,
        ema_decay=0.999,
        negative_query_path=(
            "/home/soic/Desktop/LightDet/units/model/cards/cache/"
            "negative_query_pool.json"
        ),
        negative_sample_ratio=0.05,
        use_negative_queries_in_val=False,
        max_query_loss_weight=10.0,
        lambda_rank=0.10,
        rank_start_epoch=15,
        rank_warmup_epoch=30,
        rank_alpha_min=0.0,
        score_thr=0.001,
        top_k=20,
        use_nms=True,
        use_topk_fallback=False,
        project="runs/train",
        name="lightdet_rank_smooth_010",
    )


if __name__ == "__main__":
    main()
