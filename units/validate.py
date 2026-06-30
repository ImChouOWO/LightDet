#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
LightDet 獨立驗證腳本

建議放置位置：
    LightDet/units/validate.py

執行方式：
    cd /home/soic/Desktop/LightDet/units
    python3 validate.py

此腳本不會進行訓練，只會：
    1. 讀取 model.yaml 與 train.yaml。
    2. 建立 validation dataset / dataloader。
    3. 優先載入 checkpoint 內的 EMA 權重。
    4. 使用 train.py 中修正後的 BinaryDetectionAPAccumulator。
    5. 計算 mAP50、mAP50-95、Precision、Recall、TP、FP。
    6. 額外使用全部原始預測框計算 Raw Oracle Recall。
    7. 將結果輸出為 JSON。

Raw Oracle Recall 不套用：
    - SCORE_THRESHOLD
    - TOP_K
    - NMS

其用途是確認模型原始候選框是否曾經覆蓋 GT，
不是正式部署 Precision / Recall，也不是標準 mAP。
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch





PROJECT_ROOT = Path("/home/soic/Desktop/LightDet")

MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "units/model/cards/config/model.yaml"
)

TRAIN_CONFIG_PATH = (
    PROJECT_ROOT
    / "units/model/cards/config/train.yaml"
)

DATASET_DIR = PROJECT_ROOT / "datasets"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "units/model/runs/train/lightdet_Decouple_head_2/best_map50_95.pt"
)


OUTPUT_JSON_PATH = (
    CHECKPOINT_PATH.parent
    / "validation_metrics_fixed.json"
)
DEVICE = "cuda:0"
PREFER_EMA = True

# False：只評估具有正確 GT 的正文字 query，適合比較定位能力。
# True：同時加入負文字 query，負文字上的輸出會計為 FP。
USE_NEGATIVE_QUERIES_IN_VAL = False


SCORE_THRESHOLD = 0.0
TOP_K = 100
NMS_IOU_THRESHOLD = 0.5
USE_NMS = True
USE_TOPK_FALLBACK = False


# 是否額外執行 Raw Oracle Recall。
# True 時會再完整跑一次 validation inference，因此總驗證時間約增加一倍。
ENABLE_RAW_ORACLE = True

# 使用全部 raw prediction，分別計算各 IoU 閾值下的 GT 覆蓋率。
RAW_ORACLE_IOU_THRESHOLDS = (0.25, 0.50, 0.75)


MAX_VAL_BATCHES: Optional[int] = None


COMPUTE_VAL_LOSS = False

NUM_WORKERS = 8
PREFETCH_FACTOR = 2
PIN_MEMORY = True


PREBUILD_IMAGE_CACHE = False


LOG_INTERVAL = 50
PROGRESS_LEAVE = True
PROGRESS_MIN_INTERVAL = 0.5






CURRENT_DIR = Path(__file__).resolve().parent
UNITS_DIR = CURRENT_DIR.parent

for path in (
    PROJECT_ROOT,
    UNITS_DIR,
    CURRENT_DIR,
):
    path_text = str(path)

    if path_text not in sys.path:
        sys.path.insert(0, path_text)


from units.tool.card import VisionTextModel
from units.model.cards.loss import GroundingLoss

from units.model.train import (
    build_dataloaders_with_supported_options,
    cfg_to_args,
    configure_process_file_limit,
    configure_torch_runtime,
    ensure_precomputed_bert_raw_cache,
    forward_model_batch,
    get_amp_enabled,
    get_loss_weights,
    get_target_boxes_cpu,
    load_model_config,
    load_train_config,
    parse_amp_dtype,
    prepare_model_batch,
    resolve_amp_dtype_for_device,
    set_deterministic,
    validate_one_epoch,
)



# Checkpoint helpers


def normalize_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    移除 torch.compile 或 DataParallel 可能加入的前綴。
    只有全部 key 都具有相同前綴時才會移除。
    """
    normalized = dict(state_dict)

    for prefix in ("_orig_mod.", "module."):
        while normalized and all(
            str(key).startswith(prefix)
            for key in normalized.keys()
        ):
            normalized = {
                str(key)[len(prefix):]: value
                for key, value in normalized.items()
            }

    return normalized


def select_checkpoint_state_dict(
    checkpoint: Any,
    prefer_ema: bool = True,
) -> Tuple[Dict[str, torch.Tensor], str]:
    """
    從 LightDet checkpoint 選擇驗證用權重。

    優先順序：
      prefer_ema=True：
        ema -> model -> state_dict -> raw state_dict

      prefer_ema=False：
        model -> ema -> state_dict -> raw state_dict
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint 必須是 dict/state_dict，"
            f"目前型別為 {type(checkpoint)}"
        )

    ema_state = checkpoint.get("ema")
    model_state = checkpoint.get("model")
    generic_state = checkpoint.get("state_dict")

    if (
        prefer_ema
        and isinstance(ema_state, dict)
        and ema_state
    ):
        return normalize_state_dict_keys(ema_state), "ema"

    if isinstance(model_state, dict) and model_state:
        return normalize_state_dict_keys(model_state), "model"

    if isinstance(ema_state, dict) and ema_state:
        return normalize_state_dict_keys(ema_state), "ema"

    if isinstance(generic_state, dict) and generic_state:
        return (
            normalize_state_dict_keys(generic_state),
            "state_dict",
        )

    if checkpoint and all(
        torch.is_tensor(value)
        for value in checkpoint.values()
    ):
        return (
            normalize_state_dict_keys(checkpoint),
            "raw_state_dict",
        )

    raise KeyError(
        "Checkpoint 不含可使用的 ema、model、state_dict "
        "或 raw state_dict。"
    )


def load_checkpoint_for_validation(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    prefer_ema: bool = True,
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"找不到 checkpoint：{checkpoint_path}"
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

    state_dict, weight_source = select_checkpoint_state_dict(
        checkpoint=checkpoint,
        prefer_ema=prefer_ema,
    )

    load_result = model.load_state_dict(
        state_dict,
        strict=True,
    )

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

    checkpoint_best_metric_name = (
        str(checkpoint.get("best_metric_name", "unknown"))
        if isinstance(checkpoint, dict)
        else "unknown"
    )

    print("\n[Checkpoint]")
    print(f"  path         : {checkpoint_path}")
    print(f"  source       : {weight_source}")
    print(f"  epoch        : {checkpoint_epoch}")
    print(
        f"  stored best  : "
        f"{checkpoint_best_metric_name}="
        f"{checkpoint_best_metric:.6f}"
    )
    print(f"  missing keys : {len(load_result.missing_keys)}")
    print(f"  unexpected   : {len(load_result.unexpected_keys)}")

    return {
        "checkpoint": checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "weight_source": weight_source,
        "epoch": checkpoint_epoch,
        "stored_best_metric": checkpoint_best_metric,
        "stored_best_metric_name": checkpoint_best_metric_name,
    }



# Model / loss


def build_validation_model(
    args: Any,
) -> VisionTextModel:
    return VisionTextModel(
        img_in_channels=args.img_in_channels,
        hidden_dim=args.hidden_dim,
        target_size=(
            args.target_size,
            args.target_size,
        ),
        text_max_length=args.text_max_length,
        fusion_token_num=args.fusion_token_num,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        freeze_bert=args.freeze_bert,
        precomputed_bert_path=args.precomputed_bert_path,
    )


def build_validation_criterion(
    args: Any,
) -> GroundingLoss:
    return GroundingLoss(
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



# Raw Oracle Recall


def box_iou_xyxy(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """
    計算兩組 xyxy bbox 的 pairwise IoU。

    boxes1: [N, 4]
    boxes2: [M, 4]
    return: [N, M]
    """
    boxes1 = boxes1.float().reshape(-1, 4)
    boxes2 = boxes2.float().reshape(-1, 4)

    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros(
            (boxes1.shape[0], boxes2.shape[0])
        )

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )
    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )

    left_top = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2],
    )
    right_bottom = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:],
    )

    intersection_wh = (
        right_bottom - left_top
    ).clamp(min=0)

    intersection = (
        intersection_wh[..., 0]
        * intersection_wh[..., 1]
    )

    union = (
        area1[:, None]
        + area2[None, :]
        - intersection
    )

    return intersection / union.clamp(min=1e-6)


class RawOracleRecallAccumulator:
    """
    使用全部 raw prediction 計算每個 GT 的最佳 IoU。

    對每個 GT：
        best_iou = max IoU(raw prediction, GT)

    Raw Oracle Recall@t：
        best_iou >= t 的 GT 數 / 全部 GT 數

    此指標：
      - 不使用 confidence。
      - 不使用 score threshold。
      - 不使用 Top-K。
      - 不使用 NMS。
      - 允許同一 prediction 成為多個 GT 的最佳候選。

    因此它是候選框覆蓋能力的上限診斷，不是正式偵測指標。
    """

    def __init__(
        self,
        iou_thresholds: Tuple[float, ...],
    ) -> None:
        if not iou_thresholds:
            raise ValueError(
                "RAW_ORACLE_IOU_THRESHOLDS 不可為空。"
            )

        self.iou_thresholds = tuple(
            float(value)
            for value in iou_thresholds
        )

        self.best_iou_chunks = []
        self.num_gt = 0
        self.num_samples = 0
        self.num_positive_samples = 0
        self.num_raw_predictions = 0
        self.num_raw_predictions_on_positive = 0

    def update(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> None:
        pred_boxes = (
            pred_boxes.detach()
            .float()
            .cpu()
            .reshape(-1, 4)
        )
        gt_boxes = (
            gt_boxes.detach()
            .float()
            .cpu()
            .reshape(-1, 4)
        )

        num_pred = int(pred_boxes.shape[0])
        num_gt = int(gt_boxes.shape[0])

        self.num_samples += 1
        self.num_raw_predictions += num_pred
        self.num_gt += num_gt

        if num_gt == 0:
            return

        self.num_positive_samples += 1
        self.num_raw_predictions_on_positive += num_pred

        if num_pred == 0:
            best_iou = torch.zeros(
                num_gt,
                dtype=torch.float32,
            )
        else:
            iou_matrix = box_iou_xyxy(
                pred_boxes,
                gt_boxes,
            )
            best_iou = iou_matrix.max(dim=0).values

        self.best_iou_chunks.append(
            best_iou.contiguous()
        )

    def compute(self) -> Dict[str, Any]:
        if self.num_gt == 0:
            result: Dict[str, Any] = {
                "num_gt": 0,
                "num_samples": int(self.num_samples),
                "num_positive_samples": int(
                    self.num_positive_samples
                ),
                "num_raw_predictions": int(
                    self.num_raw_predictions
                ),
                "avg_raw_predictions_per_sample": (
                    self.num_raw_predictions
                    / max(1, self.num_samples)
                ),
                "best_iou_mean": 0.0,
                "best_iou_median": 0.0,
                "best_iou_p25": 0.0,
                "best_iou_p75": 0.0,
            }

            for threshold in self.iou_thresholds:
                key = f"raw_oracle_recall@{threshold:.2f}"
                result[key] = 0.0

            return result

        best_iou = torch.cat(
            self.best_iou_chunks,
            dim=0,
        )

        if int(best_iou.numel()) != int(self.num_gt):
            raise RuntimeError(
                "Raw Oracle GT 數量不一致："
                f"best_iou={best_iou.numel()}, "
                f"num_gt={self.num_gt}"
            )

        result = {
            "num_gt": int(self.num_gt),
            "num_samples": int(self.num_samples),
            "num_positive_samples": int(
                self.num_positive_samples
            ),
            "num_raw_predictions": int(
                self.num_raw_predictions
            ),
            "num_raw_predictions_on_positive": int(
                self.num_raw_predictions_on_positive
            ),
            "avg_raw_predictions_per_sample": (
                self.num_raw_predictions
                / max(1, self.num_samples)
            ),
            "avg_raw_predictions_per_positive_sample": (
                self.num_raw_predictions_on_positive
                / max(1, self.num_positive_samples)
            ),
            "best_iou_mean": float(
                best_iou.mean().item()
            ),
            "best_iou_median": float(
                best_iou.median().item()
            ),
            "best_iou_p25": float(
                torch.quantile(best_iou, 0.25).item()
            ),
            "best_iou_p75": float(
                torch.quantile(best_iou, 0.75).item()
            ),
        }

        for threshold in self.iou_thresholds:
            key = f"raw_oracle_recall@{threshold:.2f}"
            result[key] = float(
                (best_iou >= threshold)
                .float()
                .mean()
                .item()
            )

        return result


@torch.inference_mode()
def evaluate_raw_oracle_recall(
    model: torch.nn.Module,
    val_loader: Any,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    channels_last: bool,
    iou_thresholds: Tuple[float, ...],
    max_val_batches: Optional[int] = None,
    log_interval: int = 50,
) -> Dict[str, Any]:
    """
    額外執行一次 validation inference，使用全部 raw bbox 計算 Oracle Recall。

    此函式不會呼叫任何 prediction filtering，因此 SCORE_THRESHOLD、
    TOP_K、NMS_IOU_THRESHOLD 和 USE_NMS 都不會影響結果。
    """
    model.eval()

    amp_enabled = get_amp_enabled(
        device,
        use_amp,
    )

    accumulator = RawOracleRecallAccumulator(
        iou_thresholds=iou_thresholds,
    )

    total_batches = (
        len(val_loader)
        if max_val_batches is None
        else min(
            len(val_loader),
            int(max_val_batches),
        )
    )

    oracle_start = time.perf_counter()

    for step, batch in enumerate(val_loader):
        if (
            max_val_batches is not None
            and step >= int(max_val_batches)
        ):
            break

        images, query_texts, image_indices = (
            prepare_model_batch(
                batch=batch,
                device=device,
                channels_last=channels_last,
            )
        )

        gt_boxes_cpu = get_target_boxes_cpu(batch)

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
            dtype=(
                amp_dtype
                if amp_enabled
                else None
            ),
        ):
            outputs = forward_model_batch(
                model=model,
                images=images,
                query_texts=query_texts,
                image_indices=image_indices,
            )

            raw_pred_bbox = outputs["bbox"]

        raw_pred_bbox_cpu = (
            raw_pred_bbox.detach()
            .float()
            .cpu()
        )

        if (
            int(raw_pred_bbox_cpu.shape[0])
            != len(gt_boxes_cpu)
        ):
            raise RuntimeError(
                "Raw prediction 與 GT batch size 不一致："
                f"{raw_pred_bbox_cpu.shape[0]} != "
                f"{len(gt_boxes_cpu)}"
            )

        for pred_boxes, gt_boxes in zip(
            raw_pred_bbox_cpu,
            gt_boxes_cpu,
        ):
            accumulator.update(
                pred_boxes=pred_boxes,
                gt_boxes=gt_boxes,
            )

        should_log = (
            (step + 1) % max(1, int(log_interval)) == 0
            or (step + 1) == total_batches
        )

        if should_log:
            print(
                "\r[Raw Oracle] "
                f"batch={step + 1}/{total_batches}, "
                f"samples={accumulator.num_samples}, "
                f"GT={accumulator.num_gt}, "
                f"raw_pred={accumulator.num_raw_predictions}",
                end="",
                flush=True,
            )

    print()

    metrics = accumulator.compute()
    metrics["elapsed_seconds"] = (
        time.perf_counter() - oracle_start
    )

    return metrics



# Validation


def run_validation() -> Dict[str, Any]:
    start_time = time.perf_counter()

    if not MODEL_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"找不到 model config：{MODEL_CONFIG_PATH}"
        )

    if not TRAIN_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"找不到 train config：{TRAIN_CONFIG_PATH}"
        )

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"找不到 dataset：{DATASET_DIR}"
        )

    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"設定 DEVICE={DEVICE}，但 CUDA 不可用。"
        )

    device = torch.device(DEVICE)

    model_cfg = load_model_config(
        str(MODEL_CONFIG_PATH)
    )

    train_cfg = load_train_config(
        str(TRAIN_CONFIG_PATH)
    )

    # 驗證腳本的 runtime override。
    train_cfg["data"]["dataset_dir"] = str(DATASET_DIR)
    train_cfg["data"]["use_negative_queries_in_val"] = bool(
        USE_NEGATIVE_QUERIES_IN_VAL
    )
    train_cfg["data"]["prebuild_image_cache"] = bool(
        PREBUILD_IMAGE_CACHE
    )
    train_cfg["data"]["prefetch_factor"] = int(
        PREFETCH_FACTOR
    )
    train_cfg["data"]["pin_memory"] = bool(
        PIN_MEMORY
    )

    train_cfg["train"]["device"] = DEVICE
    train_cfg["train"]["num_workers"] = int(
        NUM_WORKERS
    )

    train_cfg["eval"]["score_thr"] = float(
        SCORE_THRESHOLD
    )
    train_cfg["eval"]["top_k"] = int(TOP_K)
    train_cfg["eval"]["nms_iou_thr"] = float(
        NMS_IOU_THRESHOLD
    )
    train_cfg["eval"]["use_nms"] = bool(USE_NMS)
    train_cfg["eval"]["use_topk_fallback"] = bool(
        USE_TOPK_FALLBACK
    )
    train_cfg["eval"]["max_val_batches"] = (
        MAX_VAL_BATCHES
    )

    args = cfg_to_args(
        model_cfg_all=model_cfg,
        train_cfg_all=train_cfg,
    )

    set_deterministic(
        seed=args.seed,
        deterministic=args.deterministic,
    )

    fd_soft_limit, fd_hard_limit = (
        configure_process_file_limit()
    )

    configure_torch_runtime(args)

    amp_dtype = resolve_amp_dtype_for_device(
        device=device,
        requested_dtype=parse_amp_dtype(
            args.amp_dtype
        ),
    )

    print("\n[Validation config]")
    print(f"  device       : {device}")
    print(f"  dataset      : {args.dir}")
    print(f"  checkpoint   : {CHECKPOINT_PATH}")
    print(f"  prefer EMA   : {PREFER_EMA}")
    print(
        f"  negatives val: "
        f"{args.use_negative_queries_in_val}"
    )
    print(f"  score thr    : {args.score_thr}")
    print(f"  top-k        : {args.top_k}")
    print(f"  use NMS      : {args.use_nms}")
    print(f"  NMS IoU      : {args.nms_iou_thr}")
    print(f"  max batches  : {args.max_val_batches}")
    print(f"  AMP dtype    : {amp_dtype}")
    print(
        f"  RLIMIT_NOFILE: "
        f"({fd_soft_limit}, {fd_hard_limit})"
    )

    dataset_dir = Path(args.dir)

    train_image_dir = (
        dataset_dir / "images/train"
    )
    train_anno_dir = (
        dataset_dir / "labels/train"
    )
    val_image_dir = (
        dataset_dir / "images/val"
    )
    val_anno_dir = (
        dataset_dir / "labels/val"
    )

    _, val_loader = (
        build_dataloaders_with_supported_options(
            args,
            train_image_dir=str(train_image_dir),
            train_anno_dir=str(train_anno_dir),
            val_image_dir=str(val_image_dir),
            val_anno_dir=str(val_anno_dir),
        )
    )

    # 確保 validation 中會使用的文字都存在於 BERT cache。
    args.precomputed_bert_path = (
        ensure_precomputed_bert_raw_cache(
            cache_path=args.precomputed_bert_path,
            datasets=[val_loader.dataset],
            device=device,
            hidden_dim=args.hidden_dim,
            max_length=args.text_max_length,
            batch_size=max(128, args.batch_size),
            enabled=bool(args.freeze_bert),
        )
    )

    model = build_validation_model(args)

    checkpoint_info = load_checkpoint_for_validation(
        model=model,
        checkpoint_path=CHECKPOINT_PATH,
        prefer_ema=PREFER_EMA,
    )

    model = model.to(device)
    model.eval()

    if args.channels_last:
        model = model.to(
            memory_format=torch.channels_last
        )

    criterion = build_validation_criterion(args)

    checkpoint_epoch = max(
        1,
        int(checkpoint_info["epoch"]),
    )

    lambda_bbox, lambda_giou, lambda_score = (
        get_loss_weights(
            epoch=checkpoint_epoch,
            total_epochs=args.epochs,
            args=args,
        )
    )

    print("\n[Validation runtime]")
    print(f"  val batches  : {len(val_loader)}")
    print(
        f"  loss weights : "
        f"bbox={lambda_bbox:.4f}, "
        f"giou={lambda_giou:.4f}, "
        f"score={lambda_score:.4f}"
    )
    print(
        f"  compute loss : {COMPUTE_VAL_LOSS}"
    )

    val_loss_metrics, eval_metrics = (
        validate_one_epoch(
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            device=device,
            epoch=checkpoint_epoch,
            compute_loss=bool(COMPUTE_VAL_LOSS),
            compute_metrics=True,
            use_amp=args.use_amp,
            amp_dtype=amp_dtype,
            channels_last=args.channels_last,
            lambda_bbox=lambda_bbox,
            lambda_giou=lambda_giou,
            lambda_score=lambda_score,
            lambda_rank=args.lambda_rank,
            pos_weight=float(args.pos_weight),
            quality_warmup_epoch=(
                args.quality_warmup_epoch
            ),
            rank_start_epoch=args.rank_start_epoch,
            rank_warmup_epoch=args.rank_warmup_epoch,
            rank_alpha_min=args.rank_alpha_min,
            score_thr=args.score_thr,
            top_k=args.top_k,
            nms_iou_thr=args.nms_iou_thr,
            use_topk_fallback=(
                args.use_topk_fallback
            ),
            use_nms=args.use_nms,
            iou_thresholds=args.iou_thresholds,
            max_val_batches=args.max_val_batches,
            log_interval=LOG_INTERVAL,
            progress_leave=PROGRESS_LEAVE,
            progress_mininterval=(
                PROGRESS_MIN_INTERVAL
            ),
        )
    )

    raw_oracle_metrics: Dict[str, Any] = {}

    if ENABLE_RAW_ORACLE:
        print("\n[Raw Oracle Recall]")
        print(
            "  filters      : disabled "
            "(raw bbox only)"
        )
        print(
            f"  IoU thresholds: "
            f"{RAW_ORACLE_IOU_THRESHOLDS}"
        )

        raw_oracle_metrics = (
            evaluate_raw_oracle_recall(
                model=model,
                val_loader=val_loader,
                device=device,
                use_amp=args.use_amp,
                amp_dtype=amp_dtype,
                channels_last=args.channels_last,
                iou_thresholds=(
                    RAW_ORACLE_IOU_THRESHOLDS
                ),
                max_val_batches=(
                    args.max_val_batches
                ),
                log_interval=LOG_INTERVAL,
            )
        )

    elapsed = time.perf_counter() - start_time

    result = {
        "checkpoint": checkpoint_info[
            "checkpoint_path"
        ],
        "checkpoint_epoch": checkpoint_info[
            "epoch"
        ],
        "weight_source": checkpoint_info[
            "weight_source"
        ],
        "stored_best_metric_name": checkpoint_info[
            "stored_best_metric_name"
        ],
        "stored_best_metric": checkpoint_info[
            "stored_best_metric"
        ],
        "dataset_dir": str(DATASET_DIR),
        "use_negative_queries_in_val": bool(
            args.use_negative_queries_in_val
        ),
        "score_threshold": float(args.score_thr),
        "top_k": int(args.top_k),
        "use_nms": bool(args.use_nms),
        "nms_iou_threshold": float(
            args.nms_iou_thr
        ),
        "max_val_batches": args.max_val_batches,
        "validation_loss": val_loss_metrics,
        "evaluation": eval_metrics,
        "raw_oracle": raw_oracle_metrics,
        "elapsed_seconds": elapsed,
    }

    OUTPUT_JSON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = Path(
        str(OUTPUT_JSON_PATH) + ".tmp"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(
        temporary_path,
        OUTPUT_JSON_PATH,
    )

    print("\n[Validation result]")
    print(
        f"  mAP50       : "
        f"{eval_metrics.get('map50', 0.0):.6f}"
    )
    print(
        f"  mAP50-95    : "
        f"{eval_metrics.get('map50_95', 0.0):.6f}"
    )
    print(
        f"  Precision   : "
        f"{eval_metrics.get('precision', 0.0):.6f}"
    )
    print(
        f"  Recall      : "
        f"{eval_metrics.get('recall', 0.0):.6f}"
    )
    print(
        f"  TP / FP     : "
        f"{eval_metrics.get('tp', 0)} / "
        f"{eval_metrics.get('fp', 0)}"
    )
    print(
        f"  GT / Pred   : "
        f"{eval_metrics.get('num_gt', 0)} / "
        f"{eval_metrics.get('num_pred', 0)}"
    )
    if raw_oracle_metrics:
        print("\n[Raw Oracle result]")
        for threshold in RAW_ORACLE_IOU_THRESHOLDS:
            key = (
                f"raw_oracle_recall@"
                f"{threshold:.2f}"
            )
            print(
                f"  Recall@{threshold:.2f} : "
                f"{raw_oracle_metrics.get(key, 0.0):.6f}"
            )

        print(
            f"  Best IoU mean   : "
            f"{raw_oracle_metrics.get('best_iou_mean', 0.0):.6f}"
        )
        print(
            f"  Best IoU median : "
            f"{raw_oracle_metrics.get('best_iou_median', 0.0):.6f}"
        )
        print(
            f"  Best IoU P25/P75: "
            f"{raw_oracle_metrics.get('best_iou_p25', 0.0):.6f} / "
            f"{raw_oracle_metrics.get('best_iou_p75', 0.0):.6f}"
        )
        print(
            f"  Raw Pred / GT   : "
            f"{raw_oracle_metrics.get('num_raw_predictions', 0)} / "
            f"{raw_oracle_metrics.get('num_gt', 0)}"
        )
        print(
            f"  Oracle elapsed  : "
            f"{raw_oracle_metrics.get('elapsed_seconds', 0.0):.2f}s"
        )

    print(
        f"  elapsed     : {elapsed:.2f}s"
    )
    print(
        f"  saved JSON  : {OUTPUT_JSON_PATH}"
    )

    return result


def main() -> None:
    os.environ[
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    ] = "1"

    run_validation()


if __name__ == "__main__":
    main()
