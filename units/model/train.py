from __future__ import annotations

# LightDet optimized train.py v7: DETR-native IA-BCE + scheduled score-aware Hungarian matching

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
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
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
)
from units.model.cards.loss import (
    GroundingLoss,
    build_grounding_loss_from_config,
)

try:
    from units.model.cards.loss import (
        grounding_loss_forward_kwargs_from_config,
    )
except ImportError:
    grounding_loss_forward_kwargs_from_config = None
from units.model.tool.config import (
    DEFAULT_MODEL_CONFIG_PATH,
    DEFAULT_TRAIN_CONFIG_PATH,
    cfg_to_args,
    deepcopy_cfg,
    load_model_config,
    load_train_config,
    normalize_device,
    print_config_summary,
)
from units.model.tool.evaluation import validate_one_epoch
from units.model.tool.runtime import (
    build_text_conditioning_probe,
    compact_grounding_collate_fn,
    forward_model_batch,
    get_score_logit,
    make_progress_bar,
    move_targets_to_device,
    prepare_model_batch,
    seed_dataloader_worker,
)


# 此資料集的每批資料包含多個 query target。file_descriptor 會為每個
# shared tensor 消耗檔案描述符，容易在 Train/Val worker pools 並存時觸發
# Errno 24。file_system 使用共享記憶體名稱傳遞 storage，較適合此 workload。
try:
    mp.set_sharing_strategy("file_system")
except (RuntimeError, ValueError):
    pass



# Basic utilities



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

    loss_forward_kwargs = resolve_loss_forward_kwargs(
        args=args,
        epoch=1,
        total_epochs=total_epochs,
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
                    return_aux=True,
                )

                pred_bbox = outputs["bbox"]
                pred_score_logit = get_score_logit(outputs)
                aux_pred_bbox = outputs.get("aux_bbox")
                aux_pred_score_logit = outputs.get(
                    "aux_score_logit"
                )

                if (
                    aux_pred_bbox is None
                    or aux_pred_score_logit is None
                ):
                    raise RuntimeError(
                        "Startup hybrid check requires auxiliary "
                        "bbox and score outputs."
                    )

                loss, loss_dict = criterion(
                    pred_bbox=pred_bbox,
                    pred_score_logit=pred_score_logit,
                    targets=targets,
                    pos_weight=float(args.pos_weight),
                    query_loss_weights=batch.get("query_loss_weights"),
                    text_negative_mask=batch.get("text_negative_mask"),
                    aux_pred_bbox=aux_pred_bbox,
                    aux_pred_score_logit=aux_pred_score_logit,
                    **loss_forward_kwargs,
                )

                negative_text_matches = int(
                    float(loss_dict.get(
                        "negative_text_regression_matches",
                        0.0,
                    ))
                )
                if negative_text_matches != 0:
                    raise RuntimeError(
                        "Negative-text descriptions received bbox/GIoU "
                        "matches. DETR empty-target supervision is broken."
                    )

                (
                    probe_texts,
                    probe_changed_rows,
                    probe_mode,
                ) = build_text_conditioning_probe(
                    query_texts=query_texts,
                    text_negative_mask=batch.get("text_negative_mask"),
                )

                localization_text_delta = None
                score_text_delta = None
                if probe_texts is not None and probe_changed_rows:
                    probe_outputs = forward_model_batch(
                        model=model,
                        images=images,
                        query_texts=probe_texts,
                        image_indices=image_indices,
                        return_aux=False,
                    )
                    probe_bbox = probe_outputs["bbox"]
                    probe_score_logit = get_score_logit(probe_outputs)
                    changed_index = torch.as_tensor(
                        probe_changed_rows,
                        device=pred_bbox.device,
                        dtype=torch.long,
                    )
                    localization_text_delta = (
                        pred_bbox.index_select(0, changed_index).float()
                        - probe_bbox.index_select(0, changed_index).float()
                    ).abs().mean()
                    score_text_delta = (
                        pred_score_logit.index_select(0, changed_index).float()
                        - probe_score_logit.index_select(0, changed_index).float()
                    ).abs().mean()

                    if not bool(torch.isfinite(localization_text_delta).item()):
                        raise FloatingPointError(
                            "Text-conditioning bbox probe produced NaN/Inf"
                        )
                    if float(localization_text_delta.item()) <= 1e-8:
                        raise RuntimeError(
                            "Changing only the text description did not change "
                            "pred_bbox. The bbox path is not text-conditioned, "
                            "so positive/negative text cannot influence "
                            "localization."
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
        if aux_pred_bbox.shape[0] != expected_queries:
            raise RuntimeError(
                "Aux bbox batch does not match query count: "
                f"{aux_pred_bbox.shape[0]} != {expected_queries}"
            )
        if aux_pred_score_logit.shape[0] != expected_queries:
            raise RuntimeError(
                "Aux score batch does not match query count: "
                f"{aux_pred_score_logit.shape[0]} != {expected_queries}"
            )
        if not bool(torch.isfinite(pred_bbox).all().item()):
            raise FloatingPointError("Startup bbox output contains NaN/Inf")
        if not bool(torch.isfinite(pred_score_logit).all().item()):
            raise FloatingPointError("Startup score output contains NaN/Inf")
        if not bool(torch.isfinite(aux_pred_bbox).all().item()):
            raise FloatingPointError(
                "Startup auxiliary bbox output contains NaN/Inf"
            )
        if not bool(torch.isfinite(aux_pred_score_logit).all().item()):
            raise FloatingPointError(
                "Startup auxiliary score output contains NaN/Inf"
            )
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
            f"aux_bbox={tuple(aux_pred_bbox.shape)}, "
            f"aux_score={tuple(aux_pred_score_logit.shape)}, "
            f"loss={loss.item():.6f}, "
            f"class={loss_dict.get('classification_type', 'unknown')}, "
            f"matcher_alpha={float(loss_dict.get('matcher_score_alpha', 0.0)):.4f}, "
            f"text_probe={probe_mode}, "
            f"bbox_text_delta="
            f"{'skip' if localization_text_delta is None else f'{float(localization_text_delta.item()):.8e}'}"
        )
    finally:
        model.train(was_training)
        del batch, images, query_texts, targets
        if image_indices is not None:
            del image_indices
        if device.type == "cuda":
            torch.cuda.empty_cache()



# EMA



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



# Optimizer / scheduler



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


def resolve_scheduler_steps(
    *,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: float,
    max_warmup_steps: Optional[int],
) -> Tuple[int, int, int]:
    """
    Resolve total and warmup steps from the YAML configuration.

    warmup_epochs determines the requested warmup duration.
    max_warmup_steps is only an upper bound; None or a value <= 0 disables
    the cap. warmup is always kept below total_steps so cosine decay has at
    least one step.
    """
    epochs = int(epochs)
    steps_per_epoch = int(steps_per_epoch)
    warmup_epochs = float(warmup_epochs)

    if epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {epochs}")
    if steps_per_epoch <= 0:
        raise ValueError(
            "steps_per_epoch must be > 0, got "
            f"{steps_per_epoch}"
        )
    if warmup_epochs < 0:
        raise ValueError(
            "warmup_epochs must be >= 0, got "
            f"{warmup_epochs}"
        )

    total_steps = epochs * steps_per_epoch
    requested_warmup_steps = int(
        math.ceil(warmup_epochs * steps_per_epoch)
    )

    if max_warmup_steps is None:
        effective_warmup_steps = requested_warmup_steps
    else:
        max_warmup_steps = int(max_warmup_steps)
        if max_warmup_steps > 0:
            effective_warmup_steps = min(
                requested_warmup_steps,
                max_warmup_steps,
            )
        else:
            effective_warmup_steps = requested_warmup_steps

    effective_warmup_steps = min(
        effective_warmup_steps,
        max(total_steps - 1, 0),
    )

    return (
        int(total_steps),
        int(requested_warmup_steps),
        int(effective_warmup_steps),
    )


class WarmupCosineScheduler:
    """
    Step-based linear warmup followed by cosine decay.

    step() must be called immediately before each optimizer update. This makes
    the first optimizer step use the first warmup LR instead of the full base
    LR. step_num records the number of optimizer updates already scheduled.
    """

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
        self.base_lrs = [
            float(group["lr"])
            for group in optimizer.param_groups
        ]

        if self.total_steps <= 0:
            raise ValueError(
                f"total_steps must be > 0, got {self.total_steps}"
            )
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError(
                "warmup_steps must satisfy 0 <= warmup_steps < "
                f"total_steps, got {self.warmup_steps} and "
                f"{self.total_steps}"
            )
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(
                "min_lr_ratio must be in [0, 1], got "
                f"{self.min_lr_ratio}"
            )
        if not self.base_lrs:
            raise ValueError(
                "optimizer must contain at least one parameter group"
            )

    def _factor(self, step_num: int) -> float:
        step_num = min(
            max(int(step_num), 0),
            self.total_steps,
        )

        if self.warmup_steps > 0 and step_num <= self.warmup_steps:
            return float(step_num) / float(self.warmup_steps)

        decay_steps = self.total_steps - self.warmup_steps
        progress = (step_num - self.warmup_steps) / max(1, decay_steps)
        progress = min(max(float(progress), 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (
            self.min_lr_ratio
            + (1.0 - self.min_lr_ratio) * cosine
        )

    def _apply_step_lr(self, step_num: int) -> None:
        factor = self._factor(step_num)

        if len(self.optimizer.param_groups) != len(self.base_lrs):
            raise RuntimeError(
                "Optimizer parameter-group count changed after scheduler "
                "construction."
            )

        for group, base_lr in zip(
            self.optimizer.param_groups,
            self.base_lrs,
        ):
            group["lr"] = float(base_lr) * factor

    def step(self) -> None:
        if self.step_num < self.total_steps:
            self.step_num += 1

        self._apply_step_lr(self.step_num)

    def get_lr(self) -> List[float]:
        return [
            float(group["lr"])
            for group in self.optimizer.param_groups
        ]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "warmup_steps": int(self.warmup_steps),
            "total_steps": int(self.total_steps),
            "min_lr_ratio": float(self.min_lr_ratio),
            "step_num": int(self.step_num),
            "base_lrs": list(self.base_lrs),
        }

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
    ) -> None:
        if not isinstance(state_dict, dict):
            raise TypeError(
                "scheduler state_dict must be a dict, got "
                f"{type(state_dict)}"
            )

        warmup_steps = int(
            state_dict.get("warmup_steps", self.warmup_steps)
        )
        total_steps = int(
            state_dict.get("total_steps", self.total_steps)
        )
        min_lr_ratio = float(
            state_dict.get("min_lr_ratio", self.min_lr_ratio)
        )
        step_num = int(
            state_dict.get("step_num", 0)
        )
        base_lrs = [
            float(value)
            for value in state_dict.get("base_lrs", self.base_lrs)
        ]

        if total_steps <= 0:
            raise ValueError(
                f"Invalid scheduler total_steps: {total_steps}"
            )
        if not 0 <= warmup_steps < total_steps:
            raise ValueError(
                "Invalid scheduler warmup_steps/total_steps: "
                f"{warmup_steps}/{total_steps}"
            )
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError(
                f"Invalid scheduler min_lr_ratio: {min_lr_ratio}"
            )
        if len(base_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                "Scheduler base_lrs/optimizer group mismatch: "
                f"{len(base_lrs)} != "
                f"{len(self.optimizer.param_groups)}"
            )

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.step_num = min(max(step_num, 0), total_steps)
        self.base_lrs = base_lrs
        self._apply_step_lr(self.step_num)



# Dynamic training schedule



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


def resolve_loss_forward_kwargs(
    *,
    args: SimpleNamespace,
    epoch: int,
    total_epochs: int,
) -> Dict[str, Any]:
    """Resolve all epoch-dependent H-DETR loss parameters from train.yaml."""
    if grounding_loss_forward_kwargs_from_config is not None:
        return grounding_loss_forward_kwargs_from_config(
            args.train_cfg,
            current_epoch=int(epoch),
            total_epochs=int(total_epochs),
        )

    lambda_bbox, lambda_giou, lambda_score = get_loss_weights(
        epoch=epoch,
        total_epochs=total_epochs,
        args=args,
    )
    return {
        "lambda_bbox": float(lambda_bbox),
        "lambda_giou": float(lambda_giou),
        "lambda_score": float(lambda_score),
        "lambda_rank": 0.0,
        "lambda_aux": float(args.aux_loss_weight),
        "current_epoch": int(epoch),
        "total_epochs": int(total_epochs),
        "quality_warmup_epoch": int(args.quality_warmup_epoch),
        "rank_start_epoch": int(args.rank_start_epoch),
        "rank_warmup_epoch": int(args.rank_warmup_epoch),
        "rank_alpha_min": float(args.rank_alpha_min),
    }


# Box / target helpers


# Training


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
    lambda_aux: Optional[float] = None,
    lambda_rank: float = 0.0,
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
    total_main_loss_sum = torch.zeros((), device=device)
    total_aux_loss_sum = torch.zeros((), device=device)
    total_aux_contrib_sum = torch.zeros((), device=device)
    total_bbox_sum = torch.zeros((), device=device)
    total_giou_sum = torch.zeros((), device=device)
    total_score_sum = torch.zeros((), device=device)
    total_rank_contrib_sum = torch.zeros((), device=device)
    total_rank_raw_sum = torch.zeros((), device=device)
    total_text_negative_sum = torch.zeros((), device=device)
    total_text_negative_contrib_sum = torch.zeros((), device=device)
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

        # Schedule the LR before the optimizer update so the very first update
        # uses the first warmup LR instead of the full base learning rate.
        scheduler.step()

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
                return_aux=True,
            )

            pred_bbox = outputs["bbox"]
            pred_score_logit = get_score_logit(outputs)
            aux_pred_bbox = outputs.get("aux_bbox")
            aux_pred_score_logit = outputs.get(
                "aux_score_logit"
            )

            if (
                aux_pred_bbox is None
                or aux_pred_score_logit is None
            ):
                raise RuntimeError(
                    "Hybrid training requires aux_bbox and "
                    "aux_score_logit from VisionTextModel."
                )

            loss, loss_dict = criterion(
                pred_bbox=pred_bbox,
                pred_score_logit=pred_score_logit,
                targets=targets,
                lambda_bbox=lambda_bbox,
                lambda_giou=lambda_giou,
                lambda_score=lambda_score,
                lambda_aux=lambda_aux,
                lambda_rank=0.0,
                pos_weight=pos_weight,
                current_epoch=epoch,
                total_epochs=num_epochs,
                quality_warmup_epoch=quality_warmup_epoch,
                rank_start_epoch=rank_start_epoch,
                rank_warmup_epoch=rank_warmup_epoch,
                rank_alpha_min=rank_alpha_min,
                query_loss_weights=batch.get("query_loss_weights"),
                text_negative_mask=batch.get("text_negative_mask"),
                aux_pred_bbox=aux_pred_bbox,
                aux_pred_score_logit=aux_pred_score_logit,
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

        if ema is not None:
            ema.update(ema_source_model)

        zero = pred_bbox.new_zeros(())

        loss_det = loss.detach()
        main_loss_det = loss_dict.get(
            "loss_main_total",
            loss_det,
        ).detach()
        aux_loss_det = loss_dict.get(
            "loss_aux_total",
            zero,
        ).detach()
        aux_contrib_det = loss_dict.get(
            "loss_aux_contrib",
            zero,
        ).detach()
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
        text_negative_contrib_det = loss_dict.get(
            "loss_text_negative_contrib",
            zero,
        ).detach()
        negative_query_top1_det = loss_dict.get(
            "negative_query_top1_score",
            zero,
        ).detach()
        positive_query_top1_det = loss_dict.get(
            "positive_query_top1_score",
            zero,
        ).detach()
        query_score_margin_det = loss_dict.get(
            "positive_negative_score_margin",
            zero,
        ).detach()
        text_negative_queries = int(
            batch.get(
                "text_negative_mask",
                torch.zeros(0, dtype=torch.bool),
            ).sum().item()
        )

        total_loss_sum.add_(loss_det)
        total_main_loss_sum.add_(main_loss_det)
        total_aux_loss_sum.add_(aux_loss_det)
        total_aux_contrib_sum.add_(aux_contrib_det)
        total_bbox_sum.add_(bbox_det)
        total_giou_sum.add_(giou_det)
        total_score_sum.add_(score_det)
        total_rank_contrib_sum.add_(rank_contrib_det)
        total_rank_raw_sum.add_(rank_raw_det)
        total_text_negative_sum.add_(text_negative_det)
        total_text_negative_contrib_sum.add_(
            text_negative_contrib_det
        )
        total_text_negative_queries += text_negative_queries

        should_log = (
            (step + 1) % max(1, int(log_interval)) == 0
            or (step + 1) == len(train_loader)
        )

        if should_log:
            current_lr = scheduler.get_lr()[0]
            loss_item = float(loss_det.item())
            main_loss_item = float(main_loss_det.item())
            aux_loss_item = float(aux_loss_det.item())
            aux_contrib_item = float(aux_contrib_det.item())
            bbox_item = float(bbox_det.item())
            giou_item = float(giou_det.item())
            score_item = float(score_det.item())
            rank_raw_item = float(rank_raw_det.item())
            rank_item = float(rank_det.item())
            rank_contrib_item = float(rank_contrib_det.item())
            text_negative_item = float(text_negative_det.item())
            text_negative_contrib_item = float(
                text_negative_contrib_det.item()
            )
            negative_query_top1_item = float(
                negative_query_top1_det.item()
            )
            positive_query_top1_item = float(
                positive_query_top1_det.item()
            )
            query_score_margin_item = float(
                query_score_margin_det.item()
            )
            avg_loss = float((total_loss_sum / (step + 1)).item())

            rank_alpha_item = float(loss_dict.get("rank_alpha", 0.0))
            lambda_rank_eff_item = float(
                loss_dict.get("lambda_rank_eff", 0.0)
            )
            quality_alpha_item = float(loss_dict.get("quality_alpha", 1.0))
            score_ignore_thr_item = float(
                loss_dict.get(
                    "score_negative_iou_ignore_thr",
                    criterion.score_negative_iou_ignore_thr,
                )
            )
            matcher_score_alpha_item = float(
                loss_dict.get("matcher_score_alpha", rank_alpha_item)
            )
            matcher_score_cost_item = float(
                loss_dict.get("matcher_cost_score_effective", 0.0)
            )
            negative_text_match_item = int(
                float(loss_dict.get(
                    "negative_text_regression_matches",
                    0.0,
                ))
            )
            if negative_text_match_item != 0:
                raise RuntimeError(
                    "Negative-text sample received a regression match."
                )

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
                "main": f"{main_loss_item:.4f}",
                "aux": f"{aux_contrib_item:.4f}",
                "bbox": f"{bbox_item:.4f}",
                "giou": f"{giou_item:.4f}",
                "score": f"{score_item:.4f}",
                "msa": f"{matcher_score_alpha_item:.4f}",
                "msc": f"{matcher_score_cost_item:.4f}",
                "txtneg": f"{text_negative_item:.4f}",
                "tnc": f"{text_negative_contrib_item:.4f}",
                "pm": f"{query_score_margin_item:.3f}",
                "ntm": negative_text_match_item,
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
                    "loss_main_total": main_loss_item,
                    "loss_aux_total": aux_loss_item,
                    "loss_aux_contrib": aux_contrib_item,
                    "loss_bbox": bbox_item,
                    "loss_giou": giou_item,
                    "loss_score": score_item,
                    "loss_rank_raw": rank_raw_item,
                    "loss_rank": rank_item,
                    "loss_rank_contrib": rank_contrib_item,
                    "loss_text_negative": text_negative_item,
                    "loss_text_negative_contrib": (
                        text_negative_contrib_item
                    ),
                    "negative_query_top1_score": (
                        negative_query_top1_item
                    ),
                    "positive_query_top1_score": (
                        positive_query_top1_item
                    ),
                    "positive_negative_score_margin": (
                        query_score_margin_item
                    ),
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
                    "score_negative_iou_ignore_thr": (
                        score_ignore_thr_item
                    ),
                    "duplicate_suppression_enabled": bool(
                        loss_dict.get(
                            "duplicate_suppression_enabled",
                            False,
                        )
                    ),
                    "loss_duplicate": float(
                        loss_dict.get("loss_duplicate", zero)
                        .detach()
                        .item()
                    ),
                    "loss_duplicate_contrib": float(
                        loss_dict.get(
                            "loss_duplicate_contrib",
                            zero,
                        )
                        .detach()
                        .item()
                    ),
                    "duplicate_pair_count": float(
                        loss_dict.get(
                            "duplicate_pair_count",
                            zero,
                        )
                        .detach()
                        .item()
                    ),
                    "duplicate_violation_fraction": float(
                        loss_dict.get(
                            "duplicate_violation_fraction",
                            zero,
                        )
                        .detach()
                        .item()
                    ),
                    "duplicate_score_gap_mean": float(
                        loss_dict.get(
                            "duplicate_score_gap_mean",
                            zero,
                        )
                        .detach()
                        .item()
                    ),
                    "duplicate_classification_weight": float(
                        criterion.duplicate_classification_weight
                    ),
                    "loss_hard_negative": float(
                        loss_dict.get(
                            "loss_hard_negative",
                            zero,
                        )
                        .detach()
                        .item()
                    ),
                    "loss_hard_negative_contrib": float(
                        loss_dict.get(
                            "loss_hard_negative_contrib",
                            zero,
                        )
                        .detach()
                        .item()
                    ),
                    "hard_neg_count": float(
                        loss_dict.get("hard_neg_count", zero)
                        .detach()
                        .item()
                    ),
                    "hard_negative_score_mean": float(
                        loss_dict.get(
                            "hard_negative_score_mean",
                            zero,
                        )
                        .detach()
                        .item()
                    ),
                    "ignored_negative_count": float(
                        loss_dict.get(
                            "ignored_negative_count",
                            0.0,
                        )
                    ),
                    "selected_negative_fraction": float(
                        loss_dict.get(
                            "selected_negative_fraction",
                            1.0,
                        )
                    ),
                    "matcher_score_alpha": matcher_score_alpha_item,
                    "matcher_cost_score_effective": (
                        matcher_score_cost_item
                    ),
                    "negative_text_regression_matches": (
                        negative_text_match_item
                    ),
                    "classification_type": loss_dict.get(
                        "classification_type",
                        "unknown",
                    ),
                    "dense_score_assignment_enabled": bool(
                        loss_dict.get(
                            "dense_score_assignment_enabled",
                            False,
                        )
                    ),
                    "hard_negative_mining_enabled": bool(
                        loss_dict.get(
                            "hard_negative_mining_enabled",
                            False,
                        )
                    ),
                    "score_target_pos_mean": score_target_pos_mean,
                })
    pbar.refresh()
    pbar.close()

    num_batches = max(1, len(train_loader))

    return {
        "train_loss": float((total_loss_sum / num_batches).item()),
        "train_loss_main": float(
            (total_main_loss_sum / num_batches).item()
        ),
        "train_loss_aux": float(
            (total_aux_loss_sum / num_batches).item()
        ),
        "train_loss_aux_contrib": float(
            (total_aux_contrib_sum / num_batches).item()
        ),
        "train_loss_bbox": float((total_bbox_sum / num_batches).item()),
        "train_loss_giou": float((total_giou_sum / num_batches).item()),
        "train_loss_score": float((total_score_sum / num_batches).item()),
        "train_loss_rank_raw": float((total_rank_raw_sum / num_batches).item()),
        "train_loss_rank": float((total_rank_contrib_sum / num_batches).item()),
        "train_loss_text_negative": float(
            (total_text_negative_sum / num_batches).item()
        ),
        "train_loss_text_negative_contrib": float(
            (total_text_negative_contrib_sum / num_batches).item()
        ),
        "train_text_negative_queries": int(total_text_negative_queries),
        "score_negative_iou_ignore_thr": float(
            criterion.score_negative_iou_ignore_thr
        ),
        "duplicate_suppression_enabled": bool(
            criterion.duplicate_suppression_enabled
        ),
        "duplicate_loss_weight": float(
            criterion.duplicate_loss_weight
        ),
        "duplicate_margin": float(criterion.duplicate_margin),
        "duplicate_background_weight": float(
            criterion.duplicate_background_weight
        ),
        "duplicate_classification_weight": float(
            criterion.duplicate_classification_weight
        ),
        "duplicate_max_pairs": int(
            criterion.duplicate_max_pairs
        ),
        "duplicate_start_epoch": int(
            criterion.duplicate_start_epoch
        ),
        "hard_negative_mining_enabled": bool(
            criterion.hard_negative_mining_enabled
        ),
        "hard_negative_loss_weight": float(
            criterion.hard_negative_loss_weight
        ),
        "hard_negative_topk": int(
            criterion.hard_negative_topk
        ),
        "hard_negative_max_iou": float(
            criterion.hard_negative_max_iou
        ),
        "hard_negative_start_epoch": int(
            criterion.hard_negative_start_epoch
        ),
        "matcher_score_alpha": float(
            criterion.resolve_epoch_alpha(
                current_epoch=epoch,
                quality_warmup_epoch=quality_warmup_epoch,
                rank_start_epoch=rank_start_epoch,
                rank_warmup_epoch=rank_warmup_epoch,
                rank_alpha_min=rank_alpha_min,
            )[1]
        ),
        "matcher_cost_score_effective": float(
            criterion.main_matcher.cost_score
            * criterion.resolve_epoch_alpha(
                current_epoch=epoch,
                quality_warmup_epoch=quality_warmup_epoch,
                rank_start_epoch=rank_start_epoch,
                rank_warmup_epoch=rank_warmup_epoch,
                rank_alpha_min=rank_alpha_min,
            )[1]
        ),
        "train_negative_text_regression_matches": 0,
    }



# Fast query-conditioned binary detection metrics



@torch.no_grad()


# Unified validation: one forward pass for val loss + metrics + Raw Oracle



@torch.inference_mode()



# Metrics logging



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



# Checkpoint



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
        "scheduler": scheduler.state_dict(),
        # Legacy key retained so older tools can still inspect the checkpoint.
        "scheduler_step": scheduler.step_num,
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
        "train_metrics": train_metrics,
        "val_loss_metrics": val_loss_metrics,
        "eval_metrics": eval_metrics,
        "train_config": train_config,
        "dynamic_config": dynamic_config,
        # Legacy alias retained for checkpoints produced by earlier versions.
        "scheduler_config": scheduler.state_dict(),
        "rng_state": get_rng_state_dict(),
    }



def _normalize_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Remove wrapper prefixes that may be introduced by torch.compile or
    DataParallel. Prefixes are removed only when every key uses that prefix.
    """
    normalized = dict(state_dict)

    for prefix in ("_orig_mod.", "module."):
        while normalized and all(
            str(key).startswith(prefix) for key in normalized.keys()
        ):
            normalized = {
                str(key)[len(prefix):]: value
                for key, value in normalized.items()
            }

    return normalized


def _select_checkpoint_state_dict(
    checkpoint: Any,
    *,
    prefer_ema: bool,
) -> Tuple[Dict[str, torch.Tensor], str]:
    """
    Select a model state_dict from a LightDet checkpoint or a raw state_dict.
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a dict or state_dict, "
            f"got {type(checkpoint)}"
        )

    ema_state = checkpoint.get("ema")
    model_state = checkpoint.get("model")
    generic_state = checkpoint.get("state_dict")

    if prefer_ema and isinstance(ema_state, dict) and ema_state:
        return _normalize_state_dict_keys(ema_state), "ema"

    if isinstance(model_state, dict) and model_state:
        return _normalize_state_dict_keys(model_state), "model"

    if isinstance(ema_state, dict) and ema_state:
        return _normalize_state_dict_keys(ema_state), "ema"

    if isinstance(generic_state, dict) and generic_state:
        return _normalize_state_dict_keys(generic_state), "state_dict"

    # Raw PyTorch state_dict: every value should be a tensor.
    if checkpoint and all(
        torch.is_tensor(value) for value in checkpoint.values()
    ):
        return _normalize_state_dict_keys(checkpoint), "raw_state_dict"

    available_keys = sorted(str(key) for key in checkpoint.keys())
    raise KeyError(
        "Checkpoint does not contain model, ema, state_dict, "
        f"or a raw state_dict. Available keys: {available_keys}"
    )


def inspect_checkpoint_file(
    checkpoint_path: str,
) -> Dict[str, Any]:
    """
    Inspect whether a .pt file supports weights-only loading and full resume.
    The checkpoint is loaded on CPU and no model is required.
    """
    checkpoint_path = os.path.abspath(str(checkpoint_path))

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    is_dict = isinstance(checkpoint, dict)
    keys = sorted(str(key) for key in checkpoint.keys()) if is_dict else []

    has_model = bool(
        is_dict
        and isinstance(checkpoint.get("model"), dict)
        and checkpoint.get("model")
    )
    has_ema = bool(
        is_dict
        and isinstance(checkpoint.get("ema"), dict)
        and checkpoint.get("ema")
    )
    has_generic_state = bool(
        is_dict
        and isinstance(checkpoint.get("state_dict"), dict)
        and checkpoint.get("state_dict")
    )
    is_raw_state_dict = bool(
        is_dict
        and checkpoint
        and all(torch.is_tensor(value) for value in checkpoint.values())
    )

    has_optimizer = bool(
        is_dict and isinstance(checkpoint.get("optimizer"), dict)
    )
    has_scaler = bool(
        is_dict and isinstance(checkpoint.get("scaler"), dict)
    )
    has_scheduler = bool(
        is_dict
        and (
            "scheduler_step" in checkpoint
            or isinstance(checkpoint.get("scheduler_config"), dict)
        )
    )
    has_epoch = bool(is_dict and "epoch" in checkpoint)
    has_rng = bool(is_dict and "rng_state" in checkpoint)

    supports_weights_only = bool(
        has_model or has_ema or has_generic_state or is_raw_state_dict
    )
    supports_full_resume = bool(
        has_model
        and has_optimizer
        and has_scheduler
        and has_epoch
    )

    result = {
        "path": checkpoint_path,
        "keys": keys,
        "epoch": (
            int(checkpoint.get("epoch", 0))
            if is_dict
            else 0
        ),
        "best_metric": (
            float(checkpoint.get("best_metric", -1.0))
            if is_dict
            else -1.0
        ),
        "has_model": has_model,
        "has_ema": has_ema,
        "has_optimizer": has_optimizer,
        "has_scaler": has_scaler,
        "has_scheduler": has_scheduler,
        "has_epoch": has_epoch,
        "has_rng_state": has_rng,
        "supports_weights_only": supports_weights_only,
        "supports_full_resume": supports_full_resume,
    }

    print("\n[Checkpoint Inspect]")
    print(f"  path         : {checkpoint_path}")
    print(f"  epoch        : {result['epoch']}")
    print(f"  best_metric  : {result['best_metric']:.6f}")
    print(f"  model        : {has_model}")
    print(f"  ema          : {has_ema}")
    print(f"  optimizer    : {has_optimizer}")
    print(f"  scaler       : {has_scaler}")
    print(f"  scheduler    : {has_scheduler}")
    print(f"  rng_state    : {has_rng}")
    print(f"  weights_only : {supports_weights_only}")
    print(f"  full_resume  : {supports_full_resume}")
    print(f"  keys         : {keys}")

    return result


def load_weights_only(
    weights_path: str,
    model: torch.nn.Module,
    ema: Optional[ModelEMA],
    device: torch.device,
    prefer_ema: bool = True,
) -> Dict[str, Any]:
    """
    Load only network weights.

    This intentionally does not restore:
      - optimizer
      - scheduler
      - GradScaler
      - epoch
      - best metric
      - RNG state

    It is intended for warm-starting a new training experiment after changing
    the dataset, loss function, ranking schedule, or negative-query policy.
    """
    weights_path = os.path.abspath(str(weights_path))

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Weights checkpoint not found: {weights_path}"
        )

    try:
        checkpoint = torch.load(
            weights_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            weights_path,
            map_location=device,
        )

    state_dict, source_name = _select_checkpoint_state_dict(
        checkpoint,
        prefer_ema=bool(prefer_ema),
    )

    target_model = unwrap_model(model)
    load_result = target_model.load_state_dict(
        state_dict,
        strict=True,
    )

    # The new EMA must start from the loaded network, not from the random model
    # snapshot copied before loading the checkpoint.
    if ema is not None:
        ema.ema.load_state_dict(
            target_model.state_dict(),
            strict=True,
        )
        ema.num_updates = 0

    checkpoint_epoch = (
        int(checkpoint.get("epoch", 0))
        if isinstance(checkpoint, dict)
        else 0
    )
    checkpoint_best_metric = (
        float(checkpoint.get("best_metric", -1.0))
        if isinstance(checkpoint, dict)
        else -1.0
    )

    print("\n[Weights] Warm-start checkpoint loaded")
    print(f"  path         : {weights_path}")
    print(f"  source       : {source_name}")
    print(f"  source epoch : {checkpoint_epoch}")
    print(f"  best metric  : {checkpoint_best_metric:.6f}")
    print(f"  prefer EMA   : {bool(prefer_ema)}")
    print(f"  missing keys : {len(load_result.missing_keys)}")
    print(f"  unexpected   : {len(load_result.unexpected_keys)}")
    print("  optimizer    : reset")
    print("  scheduler    : reset")
    print("  scaler       : reset")
    print("  start epoch  : 1")

    return {
        "path": weights_path,
        "source": source_name,
        "source_epoch": checkpoint_epoch,
        "source_best_metric": checkpoint_best_metric,
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

    scheduler_state = checkpoint.get("scheduler")

    if not isinstance(scheduler_state, dict):
        scheduler_state = checkpoint.get("scheduler_config")

    if isinstance(scheduler_state, dict):
        runtime_scheduler = scheduler.state_dict()
        scheduler.load_state_dict(scheduler_state)

        if (
            int(runtime_scheduler["total_steps"])
            != int(scheduler.total_steps)
            or int(runtime_scheduler["warmup_steps"])
            != int(scheduler.warmup_steps)
        ):
            print(
                "[Checkpoint] Restored scheduler geometry from checkpoint: "
                f"warmup={scheduler.warmup_steps}, "
                f"total={scheduler.total_steps}."
            )
    elif "scheduler_step" in checkpoint:
        # Backward compatibility for old checkpoints that stored only a step.
        scheduler.step_num = min(
            max(int(checkpoint["scheduler_step"]), 0),
            scheduler.total_steps,
        )
        scheduler._apply_step_lr(scheduler.step_num)
        print(
            "[Checkpoint] Legacy scheduler_step restored; "
            "runtime warmup/total configuration retained."
        )

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



# BERT cache



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



# DataLoader construction



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



# Main training



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
        cnn_layer= args.cnn_layers,
        hidden_dim=args.hidden_dim,
        target_size=(args.target_size, args.target_size),
        text_max_length=args.text_max_length,
        fusion_token_num=args.fusion_token_num,
        num_object_queries=args.num_object_queries,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        freeze_bert=args.freeze_bert,
        precomputed_bert_path=args.precomputed_bert_path,
        use_auxiliary_head=args.use_auxiliary_head,
        auxiliary_in_eval=args.auxiliary_in_eval,
        initialize_aux_from_main=args.initialize_aux_from_main,
        query_group_init_std=args.query_group_init_std,
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

    weights_info: Optional[Dict[str, Any]] = None

    if getattr(args, "weights_path", None) is not None:
        weights_info = load_weights_only(
            weights_path=args.weights_path,
            model=model,
            ema=ema,
            device=device,
            prefer_ema=bool(
                getattr(args, "prefer_ema", True)
            ),
        )

    # Build once from the complete YAML so every DETR-native parameter is
    # imported. The module remains units.model.cards.loss (loss.py).
    criterion = build_grounding_loss_from_config(
        args.train_cfg
    ).to(device)

    expected_classification = str(args.classification_type).strip().lower()
    if criterion.classification_type != expected_classification:
        raise RuntimeError(
            "GroundingLoss classification configuration mismatch: "
            f"requested={expected_classification}, "
            f"effective={criterion.classification_type}"
        )

    if abs(float(criterion.main_matcher.cost_score) - float(args.cost_score)) > 1e-12:
        raise RuntimeError(
            "GroundingLoss matcher score-cost mismatch: "
            f"requested={args.cost_score}, "
            f"effective={criterion.main_matcher.cost_score}"
        )
    if float(criterion.main_matcher.cost_score) <= 0.0:
        raise ValueError(
            "loss.matcher.cost_score must be > 0 for text-aware DETR "
            "matching. Use the official-style value 2.0."
        )

    if bool(criterion.negative_text_as_empty_target) != bool(
        args.negative_text_as_empty_target
    ):
        raise RuntimeError(
            "GroundingLoss negative-text policy mismatch: "
            f"requested={args.negative_text_as_empty_target}, "
            f"effective={criterion.negative_text_as_empty_target}"
        )
    if not bool(criterion.negative_text_as_empty_target):
        raise ValueError(
            "loss.text_negative.as_empty_target must be true so negative "
            "descriptions cannot receive bbox/GIoU positive supervision."
        )

    parameter_checks = {
        "score_negative_iou_ignore_thr": (
            float(criterion.score_negative_iou_ignore_thr),
            float(args.score_negative_iou_ignore_thr),
        ),
        "duplicate_loss_weight": (
            float(criterion.duplicate_loss_weight),
            float(args.duplicate_loss_weight),
        ),
        "duplicate_margin": (
            float(criterion.duplicate_margin),
            float(args.duplicate_margin),
        ),
        "duplicate_background_weight": (
            float(criterion.duplicate_background_weight),
            float(args.duplicate_background_weight),
        ),
        "duplicate_classification_weight": (
            float(criterion.duplicate_classification_weight),
            float(args.duplicate_classification_weight),
        ),
        "hard_negative_loss_weight": (
            float(criterion.hard_negative_loss_weight),
            float(args.hard_negative_loss_weight),
        ),
        "hard_negative_max_iou": (
            float(criterion.hard_negative_max_iou),
            float(args.hard_negative_max_iou),
        ),
    }
    for parameter_name, (effective_value, requested_value) in (
        parameter_checks.items()
    ):
        if abs(effective_value - requested_value) > 1e-12:
            raise RuntimeError(
                f"GroundingLoss {parameter_name} mismatch: "
                f"requested={requested_value}, "
                f"effective={effective_value}"
            )

    boolean_checks = {
        "duplicate_suppression_enabled": (
            bool(criterion.duplicate_suppression_enabled),
            bool(args.duplicate_suppression_enabled),
        ),
        "hard_negative_mining_enabled": (
            bool(criterion.hard_negative_mining_enabled),
            bool(args.hard_negative_mining_enabled),
        ),
    }
    for parameter_name, (effective_value, requested_value) in (
        boolean_checks.items()
    ):
        if effective_value != requested_value:
            raise RuntimeError(
                f"GroundingLoss {parameter_name} mismatch: "
                f"requested={requested_value}, "
                f"effective={effective_value}"
            )

    if args.ranking_enabled or abs(float(args.lambda_rank)) > 1e-12:
        print(
            "[Warning] Explicit dense pairwise ranking is disabled by the "
            "DETR-native loss. ranking.enabled/lambda_rank are ignored; "
            "the preserved schedule controls score-aware Hungarian matching."
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

    (
        total_steps,
        requested_warmup_steps,
        warmup_steps,
    ) = resolve_scheduler_steps(
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs,
        max_warmup_steps=args.max_warmup_steps,
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
    print(
        "[Info] Scheduler: "
        f"total_steps={total_steps}, "
        f"warmup_requested={requested_warmup_steps}, "
        f"warmup_cap={args.max_warmup_steps}, "
        f"warmup_effective={warmup_steps}, "
        f"min_lr_ratio={args.min_lr_ratio}"
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
    print(
        f"[Info] Hybrid: main=Hungarian one-to-one, "
        f"aux=one-to-many, aux_weight={args.aux_loss_weight:.3f}, "
        f"aux_eval={args.auxiliary_in_eval}"
    )
    print(
        f"[Info] DETR classification: type={args.classification_type}, "
        f"ia_alpha={args.ia_bce_alpha:.3f}, "
        f"focal_alpha={args.classification_focal_alpha:.3f}, "
        f"focal_gamma={args.classification_focal_gamma:.3f}, "
        f"normalize_by_num_gt={args.normalize_classification_by_num_gt}"
    )
    print(
        f"[Info] Hungarian matcher: score:bbox:giou="
        f"{args.cost_score:.2f}:{args.cost_bbox:.2f}:{args.cost_giou:.2f}, "
        f"score_cost={args.matcher_score_cost_type}"
    )
    print(
        f"[Info] Matcher score schedule: start_epoch="
        f"{args.rank_start_epoch}, warmup_epochs={args.rank_warmup_epoch}, "
        f"alpha_min={args.rank_alpha_min:.4f}"
    )
    print(
        "[Info] Text negative: "
        f"as_empty_target={args.negative_text_as_empty_target}, "
        f"lambda={args.lambda_text_negative:.4f}, "
        f"topk={args.text_negative_topk}, "
        f"hard_mix={args.text_negative_hard_mix:.3f}"
    )
    print(
        "[Info] Duplicate suppression: "
        f"enabled={args.duplicate_suppression_enabled}, "
        f"ignore_iou={args.score_negative_iou_ignore_thr:.3f}, "
        f"weight={args.duplicate_loss_weight:.4f}, "
        f"margin={args.duplicate_margin:.3f}, "
        f"background_weight={args.duplicate_background_weight:.4f}, "
        f"classification_weight="
        f"{args.duplicate_classification_weight:.4f}, "
        f"max_pairs={args.duplicate_max_pairs}, "
        f"start_epoch={args.duplicate_start_epoch}"
    )
    print(
        "[Info] Hard-negative mining: "
        f"enabled={args.hard_negative_mining_enabled}, "
        f"weight={args.hard_negative_loss_weight:.4f}, "
        f"topk={args.hard_negative_topk}, "
        f"max_iou={args.hard_negative_max_iou:.3f}, "
        f"ratio={args.hard_negative_ratio}, "
        f"start_epoch={args.hard_negative_start_epoch}"
    )
    print(
        "[Info] Removed dense objectives: "
        "independent_score_assigner=False, pairwise_ranking=False"
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
        "requested_warmup_steps": requested_warmup_steps,
        "warmup_steps": warmup_steps,
        "max_warmup_steps": args.max_warmup_steps,
        "checkpoint_mode": (
            "weights_only"
            if getattr(args, "weights_path", None) is not None
            else (
                "resume"
                if args.resume_path is not None
                else "scratch"
            )
        ),
        "weights_info": weights_info,
    })

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()

        loss_forward_kwargs = resolve_loss_forward_kwargs(
            args=args,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        lambda_bbox = float(loss_forward_kwargs["lambda_bbox"])
        lambda_giou = float(loss_forward_kwargs["lambda_giou"])
        lambda_score = float(loss_forward_kwargs["lambda_score"])
        lambda_aux = float(
            loss_forward_kwargs.get("lambda_aux", args.aux_loss_weight)
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
            lambda_aux=lambda_aux,
            lambda_rank=0.0,
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
            total_epochs=args.epochs,
            compute_metrics=run_eval,
            use_amp=args.use_amp,
            amp_dtype=amp_dtype,
            channels_last=args.channels_last,
            lambda_bbox=lambda_bbox,
            lambda_giou=lambda_giou,
            lambda_score=lambda_score,
            lambda_rank=0.0,
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
            compute_raw_oracle=args.compute_raw_oracle,
            raw_oracle_iou_thresholds=(
                args.raw_oracle_iou_thresholds
            ),
            max_val_batches=args.max_val_batches,
            log_interval=args.log_interval,
            progress_leave=args.progress_leave,
            progress_mininterval=args.progress_mininterval,
        )

        dynamic_config = {
            "lambda_bbox": lambda_bbox,
            "lambda_giou": lambda_giou,
            "lambda_score": lambda_score,
            "classification_type": args.classification_type,
            "ia_bce_alpha": float(args.ia_bce_alpha),
            "classification_focal_alpha": float(
                args.classification_focal_alpha
            ),
            "classification_focal_gamma": float(
                args.classification_focal_gamma
            ),
            "normalize_classification_by_num_gt": bool(
                args.normalize_classification_by_num_gt
            ),
            "dense_score_assignment_enabled": False,
            "pairwise_ranking_enabled": False,
            "duplicate_suppression_enabled": bool(
                args.duplicate_suppression_enabled
            ),
            "duplicate_suppression_active": bool(
                args.duplicate_suppression_enabled
                and epoch >= args.duplicate_start_epoch
            ),
            "duplicate_loss_weight": float(
                args.duplicate_loss_weight
            ),
            "duplicate_margin": float(args.duplicate_margin),
            "duplicate_background_weight": float(
                args.duplicate_background_weight
            ),
            "duplicate_classification_weight": float(
                args.duplicate_classification_weight
            ),
            "duplicate_max_pairs": int(args.duplicate_max_pairs),
            "duplicate_start_epoch": int(
                args.duplicate_start_epoch
            ),
            "hard_negative_mining_enabled": bool(
                args.hard_negative_mining_enabled
            ),
            "hard_negative_mining_active": bool(
                args.hard_negative_mining_enabled
                and epoch >= args.hard_negative_start_epoch
            ),
            "hard_negative_loss_weight": float(
                args.hard_negative_loss_weight
            ),
            "hard_negative_topk": int(args.hard_negative_topk),
            "hard_negative_max_iou": float(
                args.hard_negative_max_iou
            ),
            "hard_negative_start_epoch": int(
                args.hard_negative_start_epoch
            ),
            "lambda_rank_max": 0.0,
            "lambda_aux": float(lambda_aux),
            "pos_weight": 1.0,
            "quality_warmup_epoch": int(args.quality_warmup_epoch),
            "aux_score_enabled": bool(args.aux_score_enabled),
            "matcher_score_start_epoch": int(args.rank_start_epoch),
            "matcher_score_warmup_epoch": int(args.rank_warmup_epoch),
            "matcher_score_alpha_min": float(args.rank_alpha_min),
            "negative_sample_ratio": float(args.negative_sample_ratio),
            "max_query_loss_weight": float(args.max_query_loss_weight),
            "lambda_text_negative": float(args.lambda_text_negative),
            "text_negative_topk": int(args.text_negative_topk),
            "text_negative_hard_mix": float(
                args.text_negative_hard_mix
            ),
            "score_negative_iou_ignore_thr": float(
                args.score_negative_iou_ignore_thr
            ),
            "matcher_score_alpha": float(
                criterion.resolve_epoch_alpha(
                    current_epoch=epoch,
                    quality_warmup_epoch=args.quality_warmup_epoch,
                    rank_start_epoch=args.rank_start_epoch,
                    rank_warmup_epoch=args.rank_warmup_epoch,
                    rank_alpha_min=args.rank_alpha_min,
                )[1]
            ),
            "matcher_cost_score_effective": float(
                criterion.main_matcher.cost_score
                * criterion.resolve_epoch_alpha(
                    current_epoch=epoch,
                    quality_warmup_epoch=args.quality_warmup_epoch,
                    rank_start_epoch=args.rank_start_epoch,
                    rank_warmup_epoch=args.rank_warmup_epoch,
                    rank_alpha_min=args.rank_alpha_min,
                )[1]
            ),
            "negative_text_as_empty_target": bool(
                args.negative_text_as_empty_target
            ),
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
                # tqdm.write(
                #     f"[Checkpoint] latest.pt saved in {elapsed:.2f}s"
                # )

            if is_best:
                best_path = os.path.join(
                    save_path,
                    f"best_{best_metric_name}.pt",
                )
                elapsed = save_checkpoint(checkpoint, best_path)
                GREEN_BG = "\033[42m"
                BLACK_TEXT = "\033[30m"
                RESET = "\033[0m"

                tqdm.write(
                    f"{GREEN_BG}{BLACK_TEXT}"
                    f" Saved best checkpoint: epoch={epoch}, "
                    f"{best_metric_name}={best_metric:.4f}, "
                    f"time={elapsed:.2f}s "
                    f"{RESET}"
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
            f"Oracle50="
            f"{eval_metrics.get('raw_oracle_recall50', -1):.4f} "
            f"Gap50="
            f"{eval_metrics.get('raw_oracle_gap50', -1):.4f} "
            f"lb={lambda_bbox:.2f} "
            f"lg={lambda_giou:.2f} "
            f"ls={lambda_score:.2f} "
            f"matcher_alpha="
            f"{dynamic_config.get('matcher_score_alpha', 0.0):.2f} "
            f"pw={pos_weight:.2f} "
            f"epoch_time={epoch_time:.2f}s"
        )



# Configuration


class LightDet:
    """
    YOLO-style training interface.

    Frequently changed runtime values stay in model.train(...).
    Model, optimizer, matcher, loss, EMA, AMP, cache and evaluation details
    remain in model.yaml / train.yaml as the single source of truth.
    """

    def __init__(
        self,
        model: str = str(DEFAULT_MODEL_CONFIG_PATH),
    ) -> None:
        self.model_cfg_path = model
        self.model_cfg = load_model_config(
            model
        )

    def inspect_checkpoint(
        self,
        checkpoint_path: str,
    ) -> Dict[str, Any]:
        return inspect_checkpoint_file(
            checkpoint_path
        )

    def train(
        self,
        cfg: str = str(DEFAULT_TRAIN_CONFIG_PATH),
        data: Optional[str] = None,
        epochs: Optional[int] = None,
        imgsz: Optional[int] = None,
        batch: Optional[int] = None,
        device: Optional[Any] = None,
        workers: Optional[int] = None,
        seed: Optional[int] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        weights: Optional[str] = None,
        resume: Optional[Any] = None,
        prefer_ema: Optional[bool] = None,
    ) -> None:
        """
        Train LightDet through a compact YOLO-like function call.

        Runtime arguments:
            data:
                Dataset root.
            epochs, imgsz, batch, device, workers, seed:
                Common experiment overrides.
            project, name:
                Output directory: project/name.
            weights:
                Weights-only warm start. Optimizer and epoch restart.
            resume:
                Full-state resume. True resolves project/name/latest.pt.
            prefer_ema:
                Prefer EMA weights when using weights=...

        All advanced settings are read from YAML, including:
            learning rates, AMP, EMA, compile, cache, hybrid loss,
            one-to-many expansion, ranking loss and evaluation policy.
        """
        model_cfg = deepcopy_cfg(
            self.model_cfg
        )
        train_cfg = load_train_config(
            cfg
        )

        if data is not None:
            train_cfg["data"][
                "dataset_dir"
            ] = os.path.abspath(str(data))

        if epochs is not None:
            train_cfg["train"][
                "epochs"
            ] = int(epochs)

        if imgsz is not None:
            train_cfg["data"][
                "image_size"
            ] = int(imgsz)

        if batch is not None:
            train_cfg["train"][
                "batch_size"
            ] = int(batch)

        if workers is not None:
            train_cfg["train"][
                "num_workers"
            ] = int(workers)

        if seed is not None:
            train_cfg["train"][
                "seed"
            ] = int(seed)

        if device is not None:
            train_cfg["train"][
                "device"
            ] = normalize_device(
                device
            )
        else:
            train_cfg["train"][
                "device"
            ] = normalize_device(
                train_cfg["train"][
                    "device"
                ]
            )

        if project is not None or name is not None:
            configured_save_dir = train_cfg["log"].get("save_dir")
            project_value = (
                str(project)
                if project is not None
                else os.path.dirname(str(configured_save_dir))
            )
            name_value = (
                str(name)
                if name is not None
                else os.path.basename(str(configured_save_dir))
            )
            train_cfg["log"]["save_dir"] = os.path.abspath(
                os.path.join(project_value, name_value)
            )

        resolved_weights_path: Optional[str] = None
        resolved_resume_path: Optional[str] = None

        configured_weights = train_cfg["log"].get("weights_path")
        configured_resume = train_cfg["log"].get("resume_path")

        if weights is None:
            weights = configured_weights
        if resume is None:
            resume = configured_resume

        if weights is not None:
            if isinstance(weights, bool):
                raise TypeError(
                    "weights must be a checkpoint "
                    "path or None."
                )

            resolved_weights_path = (
                os.path.abspath(
                    str(weights)
                )
            )

        if isinstance(resume, bool):
            if resume:
                resolved_resume_path = (
                    os.path.abspath(
                        os.path.join(
                            str(train_cfg["log"]["save_dir"]),
                            "latest.pt",
                        )
                    )
                )
        elif resume is not None:
            resolved_resume_path = (
                os.path.abspath(
                    str(resume)
                )
            )

        if (
            resolved_weights_path
            is not None
            and resolved_resume_path
            is not None
        ):
            raise ValueError(
                "weights and resume cannot be "
                "used together."
            )

        if (
            resolved_weights_path
            is not None
            and not os.path.isfile(
                resolved_weights_path
            )
        ):
            raise FileNotFoundError(
                "Weights checkpoint not found: "
                f"{resolved_weights_path}"
            )

        if (
            resolved_resume_path
            is not None
            and not os.path.isfile(
                resolved_resume_path
            )
        ):
            raise FileNotFoundError(
                "Resume checkpoint not found: "
                f"{resolved_resume_path}"
            )

        # Respect eval.use_nms from YAML. Use False for standard DETR AP and
        # True for deployment-oriented duplicate removal.

        # Checkpoint mode is controlled only by this function call.
        train_cfg["log"]["weights_path"] = resolved_weights_path
        train_cfg["log"]["resume_path"] = resolved_resume_path
        if prefer_ema is None:
            prefer_ema = bool(train_cfg["log"].get("prefer_ema", True))
        train_cfg["log"]["prefer_ema"] = bool(prefer_ema)

        set_deterministic(
            seed=train_cfg["train"][
                "seed"
            ],
            deterministic=train_cfg[
                "train"
            ]["deterministic"],
        )

        args = cfg_to_args(
            model_cfg_all=model_cfg,
            train_cfg_all=train_cfg,
        )

        args.weights_path = (
            resolved_weights_path
        )
        args.resume_path = (
            resolved_resume_path
        )
        args.prefer_ema = bool(prefer_ema)

        print_config_summary(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
        )

        if args.weights_path is not None:
            print(
                "\n[Info] Training mode: "
                "weights-only warm start"
            )
            print(
                f"  weights      : "
                f"{args.weights_path}"
            )
            print(
                f"  prefer EMA   : "
                f"{args.prefer_ema}"
            )
            print(
                "  optimizer    : reset"
            )
            print(
                "  scheduler    : reset"
            )
            print(
                "  epoch        : restart from 1"
            )
        elif args.resume_path is not None:
            print(
                "\n[Info] Training mode: "
                "full resume"
            )
            print(
                f"  checkpoint   : "
                f"{args.resume_path}"
            )
        else:
            print(
                "\n[Info] Training mode: "
                "train from scratch"
            )

        return train(args)


def main() -> None:
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    model = LightDet(
        model=str(DEFAULT_MODEL_CONFIG_PATH),
    )
    model.train(
        cfg=str(DEFAULT_TRAIN_CONFIG_PATH),
    )


if __name__ == "__main__":
    main()
