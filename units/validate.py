#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
LightDet Hybrid 獨立驗證入口。

建議放置位置：
    LightDet/units/validate.py

呼叫方式：
    model = LightDet(model="/path/to/model.yaml")
    metrics = model.val(
        cfg="/path/to/train.yaml",
        weights="/path/to/best_map50_95.pt",
        data="/path/to/datasets",
        imgsz=512,
        batch=48,
        device=0,
        workers=8,
        project="runs/val",
        name="lightdet_hybrid_best",
    )

驗證規則：
    1. 只執行 Main One-to-One branch。
    2. Auxiliary One-to-Many branch 不參與驗證。
    3. 正式 mAP 不使用 NMS。
    4. Raw Oracle 與正式指標共用同一次 forward。
    5. score、top-k、Raw Oracle 等進階設定由 train.yaml 管理。
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


CURRENT_DIR = Path(__file__).resolve().parent
UNITS_DIR = CURRENT_DIR.parent
PROJECT_ROOT = UNITS_DIR.parent

for path in (
    PROJECT_ROOT,
    UNITS_DIR,
    CURRENT_DIR,
):
    path_text = str(path)

    if path_text not in sys.path:
        sys.path.insert(0, path_text)


from units.model.cards.loss import GroundingLoss
from units.model.train import (
    LightDet as TrainLightDet,
    build_dataloaders_with_supported_options,
    cfg_to_args,
    configure_process_file_limit,
    configure_torch_runtime,
    deepcopy_cfg,
    ensure_precomputed_bert_raw_cache,
    get_amp_enabled,
    get_loss_weights,
    load_train_config,
    normalize_device,
    parse_amp_dtype,
    resolve_amp_dtype_for_device,
    set_deterministic,
    validate_one_epoch,
)
from units.tool.card import VisionTextModel


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def normalize_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    移除 torch.compile 或 DataParallel 可能加入的共同前綴。
    """
    normalized = dict(state_dict)

    for prefix in (
        "_orig_mod.",
        "module.",
    ):
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
    從 LightDet checkpoint 選擇驗證權重。
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
        return (
            normalize_state_dict_keys(ema_state),
            "ema",
        )

    if isinstance(model_state, dict) and model_state:
        return (
            normalize_state_dict_keys(model_state),
            "model",
        )

    if isinstance(ema_state, dict) and ema_state:
        return (
            normalize_state_dict_keys(ema_state),
            "ema",
        )

    if (
        isinstance(generic_state, dict)
        and generic_state
    ):
        return (
            normalize_state_dict_keys(
                generic_state
            ),
            "state_dict",
        )

    if checkpoint and all(
        torch.is_tensor(value)
        for value in checkpoint.values()
    ):
        return (
            normalize_state_dict_keys(
                checkpoint
            ),
            "raw_state_dict",
        )

    available = sorted(
        str(key)
        for key in checkpoint.keys()
    )

    raise KeyError(
        "Checkpoint 不含可使用的 ema、model、state_dict "
        f"或 raw state_dict。Available keys: {available}"
    )


def load_checkpoint_for_validation(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    prefer_ema: bool = True,
) -> Dict[str, Any]:
    """
    載入驗證權重，不恢復 optimizer、scheduler 或 scaler。
    """
    checkpoint_path = Path(
        checkpoint_path
    ).expanduser().resolve()

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

    state_dict, weight_source = (
        select_checkpoint_state_dict(
            checkpoint=checkpoint,
            prefer_ema=prefer_ema,
        )
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

    stored_best_metric = (
        float(
            checkpoint.get(
                "best_metric",
                -1.0,
            )
        )
        if isinstance(checkpoint, dict)
        else -1.0
    )

    stored_best_metric_name = (
        str(
            checkpoint.get(
                "best_metric_name",
                "unknown",
            )
        )
        if isinstance(checkpoint, dict)
        else "unknown"
    )

    print("\n[Checkpoint]")
    print(f"  path         : {checkpoint_path}")
    print(f"  source       : {weight_source}")
    print(f"  epoch        : {checkpoint_epoch}")
    print(
        "  stored best  : "
        f"{stored_best_metric_name}="
        f"{stored_best_metric:.6f}"
    )
    print(
        "  missing keys : "
        f"{len(load_result.missing_keys)}"
    )
    print(
        "  unexpected   : "
        f"{len(load_result.unexpected_keys)}"
    )

    return {
        "checkpoint_path": str(
            checkpoint_path
        ),
        "weight_source": weight_source,
        "epoch": checkpoint_epoch,
        "stored_best_metric": (
            stored_best_metric
        ),
        "stored_best_metric_name": (
            stored_best_metric_name
        ),
    }


# ---------------------------------------------------------------------------
# Build model / loss
# ---------------------------------------------------------------------------


def build_validation_model(
    args: Any,
) -> VisionTextModel:
    """
    建立 Hybrid 模型，但 eval 時固定不執行 auxiliary branch。
    """
    return VisionTextModel(
        img_in_channels=(
            args.img_in_channels
        ),
        hidden_dim=args.hidden_dim,
        target_size=(
            args.target_size,
            args.target_size,
        ),
        text_max_length=(
            args.text_max_length
        ),
        fusion_token_num=(
            args.fusion_token_num
        ),
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        freeze_bert=args.freeze_bert,
        precomputed_bert_path=(
            args.precomputed_bert_path
        ),
        use_auxiliary_head=(
            args.use_auxiliary_head
        ),
        auxiliary_in_eval=False,
        initialize_aux_from_main=(
            args.initialize_aux_from_main
        ),
    )


def build_validation_criterion(
    args: Any,
) -> GroundingLoss:
    """
    建立與訓練一致的 loss。

    validate_one_epoch() 只傳入 main prediction，
    因此 validation loss 只計算 One-to-One 主分支。
    """
    return GroundingLoss(
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou,
        cost_score=args.cost_score,
        hard_negative_ratio=(
            args.hard_negative_ratio
        ),
        positive_ratio=args.positive_ratio,
        max_positive_per_gt=(
            args.max_positive_per_gt
        ),
        aux_positive_label=(
            args.aux_positive_label
        ),
        expand_cost_bbox=(
            args.expand_cost_bbox
        ),
        expand_cost_giou=(
            args.expand_cost_giou
        ),
        iou_pos_thr=args.iou_pos_thr,
        quality_min=args.quality_min,
        quality_max=args.quality_max,
        qfl_beta=args.qfl_beta,
        rank_margin=args.rank_margin,
        rank_min_quality_gap=(
            args.rank_min_quality_gap
        ),
        rank_max_pairs=(
            args.rank_max_pairs
        ),
        max_query_loss_weight=(
            args.max_query_loss_weight
        ),
        aux_loss_weight=(
            args.aux_loss_weight
        ),
        aux_cost_score=(
            args.aux_cost_score
        ),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_json_atomic(
    path: str | Path,
    value: Dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = Path(
        f"{path}.tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(
        temporary_path,
        path,
    )


# ---------------------------------------------------------------------------
# Validation core
# ---------------------------------------------------------------------------


def validate(
    *,
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    prefer_ema: bool = True,
    compute_loss: bool = False,
    save_json: bool = True,
) -> Dict[str, Any]:
    """
    執行完整 validation。

    Runtime 常用參數由 LightDet.val() 覆寫；
    matcher、loss、score、top-k、Raw Oracle 等進階設定由 YAML 管理。
    """
    start_time = time.perf_counter()

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

    device = torch.device(
        normalize_device(
            args.device
        )
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            f"設定 device={device}，但 CUDA 不可用。"
        )

    amp_dtype = (
        resolve_amp_dtype_for_device(
            device=device,
            requested_dtype=parse_amp_dtype(
                args.amp_dtype
            ),
        )
    )

    if amp_dtype == torch.float32:
        args.use_amp = False

    dataset_dir = Path(
        args.dir
    ).expanduser().resolve()

    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"找不到 dataset：{dataset_dir}"
        )

    train_image_dir = (
        dataset_dir
        / "images"
        / "train"
    )
    train_anno_dir = (
        dataset_dir
        / "labels"
        / "train"
    )
    val_image_dir = (
        dataset_dir
        / "images"
        / "val"
    )
    val_anno_dir = (
        dataset_dir
        / "labels"
        / "val"
    )

    _, val_loader = (
        build_dataloaders_with_supported_options(
            args,
            train_image_dir=str(
                train_image_dir
            ),
            train_anno_dir=str(
                train_anno_dir
            ),
            val_image_dir=str(
                val_image_dir
            ),
            val_anno_dir=str(
                val_anno_dir
            ),
        )
    )

    args.precomputed_bert_path = (
        ensure_precomputed_bert_raw_cache(
            cache_path=(
                args.precomputed_bert_path
            ),
            datasets=[
                val_loader.dataset
            ],
            device=device,
            hidden_dim=args.hidden_dim,
            max_length=(
                args.text_max_length
            ),
            batch_size=max(
                128,
                args.batch_size,
            ),
            enabled=bool(
                args.freeze_bert
            ),
        )
    )

    model = build_validation_model(
        args
    )

    checkpoint_info = (
        load_checkpoint_for_validation(
            model=model,
            checkpoint_path=(
                checkpoint_path
            ),
            prefer_ema=prefer_ema,
        )
    )

    model = model.to(device)
    model.eval()

    if args.channels_last:
        model = model.to(
            memory_format=(
                torch.channels_last
            )
        )

    criterion = (
        build_validation_criterion(
            args
        )
    )

    checkpoint_epoch = max(
        1,
        int(
            checkpoint_info["epoch"]
        ),
    )

    (
        lambda_bbox,
        lambda_giou,
        lambda_score,
    ) = get_loss_weights(
        epoch=checkpoint_epoch,
        total_epochs=args.epochs,
        args=args,
    )

    print("\n[LightDet] Validation config")
    print(f"  dataset      : {dataset_dir}")
    print(
        "  checkpoint   : "
        f"{checkpoint_info['checkpoint_path']}"
    )
    print(f"  source       : {checkpoint_info['weight_source']}")
    print(f"  epoch        : {checkpoint_epoch}")
    print(f"  device       : {device}")
    print(
        "  AMP          : "
        f"enabled={get_amp_enabled(device, args.use_amp)}, "
        f"dtype={amp_dtype}"
    )
    print(f"  batch        : {args.batch_size}")
    print(
        "  val workers  : "
        f"{val_loader.num_workers}"
    )
    print(
        "  negative val : "
        f"{args.use_negative_queries_in_val}"
    )
    print(f"  score thr    : {args.score_thr}")
    print(f"  top-k        : {args.top_k}")
    print("  use NMS      : False")
    print(
        "  raw oracle   : "
        f"{args.compute_raw_oracle}"
    )
    print(
        "  max batches  : "
        f"{args.max_val_batches}"
    )
    print(
        "  compute loss : "
        f"{compute_loss}"
    )
    print(
        "  RLIMIT_NOFILE: "
        f"({fd_soft_limit}, {fd_hard_limit})"
    )

    val_loss_metrics, eval_metrics = (
        validate_one_epoch(
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            device=device,
            epoch=checkpoint_epoch,
            compute_loss=bool(
                compute_loss
            ),
            compute_metrics=True,
            use_amp=args.use_amp,
            amp_dtype=amp_dtype,
            channels_last=(
                args.channels_last
            ),
            lambda_bbox=lambda_bbox,
            lambda_giou=lambda_giou,
            lambda_score=lambda_score,
            lambda_rank=args.lambda_rank,
            pos_weight=float(
                args.pos_weight
            ),
            quality_warmup_epoch=(
                args.quality_warmup_epoch
            ),
            rank_start_epoch=(
                args.rank_start_epoch
            ),
            rank_warmup_epoch=(
                args.rank_warmup_epoch
            ),
            rank_alpha_min=(
                args.rank_alpha_min
            ),
            score_thr=args.score_thr,
            top_k=args.top_k,
            nms_iou_thr=(
                args.nms_iou_thr
            ),
            use_topk_fallback=(
                args.use_topk_fallback
            ),
            # Main One-to-One 正式驗證固定關閉 NMS。
            use_nms=False,
            iou_thresholds=(
                args.iou_thresholds
            ),
            compute_raw_oracle=(
                args.compute_raw_oracle
            ),
            raw_oracle_iou_thresholds=(
                args.raw_oracle_iou_thresholds
            ),
            max_val_batches=(
                args.max_val_batches
            ),
            log_interval=(
                args.log_interval
            ),
            progress_leave=(
                args.progress_leave
            ),
            progress_mininterval=(
                args.progress_mininterval
            ),
        )
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    output_dir = Path(
        output_dir
    ).expanduser().resolve()

    result = {
        "checkpoint": (
            checkpoint_info[
                "checkpoint_path"
            ]
        ),
        "checkpoint_epoch": (
            checkpoint_info["epoch"]
        ),
        "weight_source": (
            checkpoint_info[
                "weight_source"
            ]
        ),
        "stored_best_metric_name": (
            checkpoint_info[
                "stored_best_metric_name"
            ]
        ),
        "stored_best_metric": (
            checkpoint_info[
                "stored_best_metric"
            ]
        ),
        "dataset_dir": str(
            dataset_dir
        ),
        "device": str(device),
        "image_size": int(
            args.image_size
        ),
        "batch_size": int(
            args.batch_size
        ),
        "use_negative_queries_in_val": bool(
            args.use_negative_queries_in_val
        ),
        "score_threshold": float(
            args.score_thr
        ),
        "top_k": int(args.top_k),
        "use_nms": False,
        "compute_raw_oracle": bool(
            args.compute_raw_oracle
        ),
        "max_val_batches": (
            args.max_val_batches
        ),
        "validation_loss": (
            val_loss_metrics
        ),
        "evaluation": eval_metrics,
        "elapsed_seconds": (
            elapsed_seconds
        ),
    }

    output_json_path = (
        output_dir
        / "validation_metrics.json"
    )

    if save_json:
        write_json_atomic(
            output_json_path,
            result,
        )

    print("\n[Validation result]")
    print(
        "  mAP50       : "
        f"{eval_metrics.get('map50', 0.0):.6f}"
    )
    print(
        "  mAP50-95    : "
        f"{eval_metrics.get('map50_95', 0.0):.6f}"
    )
    print(
        "  Precision   : "
        f"{eval_metrics.get('precision', 0.0):.6f}"
    )
    print(
        "  Recall      : "
        f"{eval_metrics.get('recall', 0.0):.6f}"
    )
    print(
        "  TP / FP     : "
        f"{eval_metrics.get('tp', 0)} / "
        f"{eval_metrics.get('fp', 0)}"
    )
    print(
        "  GT / Pred   : "
        f"{eval_metrics.get('num_gt', 0)} / "
        f"{eval_metrics.get('num_pred', 0)}"
    )

    if args.compute_raw_oracle:
        print(
            "  Oracle R25  : "
            f"{eval_metrics.get('raw_oracle_recall25', 0.0):.6f}"
        )
        print(
            "  Oracle R50  : "
            f"{eval_metrics.get('raw_oracle_recall50', 0.0):.6f}"
        )
        print(
            "  Oracle R75  : "
            f"{eval_metrics.get('raw_oracle_recall75', 0.0):.6f}"
        )
        print(
            "  BestIoU mean: "
            f"{eval_metrics.get('raw_best_iou_mean', 0.0):.6f}"
        )
        print(
            "  Gap50       : "
            f"{eval_metrics.get('raw_oracle_gap50', 0.0):.6f}"
        )

    print(
        "  elapsed     : "
        f"{elapsed_seconds:.2f}s"
    )

    if save_json:
        print(
            "  saved JSON  : "
            f"{output_json_path}"
        )

    return result


# ---------------------------------------------------------------------------
# YOLO-style interface
# ---------------------------------------------------------------------------


class LightDet(TrainLightDet):
    """
    在 train.py 的 LightDet.train() 基礎上加入 YOLO-style val()。

    Example:
        model = LightDet(model="/path/to/model.yaml")
        metrics = model.val(
            cfg="/path/to/train.yaml",
            weights="/path/to/best.pt",
            data="/path/to/datasets",
            imgsz=512,
            batch=48,
            device=0,
            workers=8,
            project="runs/val",
            name="exp",
        )
    """

    def val(
        self,
        cfg: str = "cards/config/train.yaml",
        weights: Optional[str] = None,
        data: Optional[str] = None,
        imgsz: Optional[int] = None,
        batch: Optional[int] = None,
        device: Optional[Any] = None,
        workers: Optional[int] = None,
        project: str = "runs/val",
        name: str = "exp",
        prefer_ema: bool = True,
        compute_loss: bool = False,
        save_json: bool = True,
    ) -> Dict[str, Any]:
        """
        執行驗證。

        僅保留常用 runtime 參數；其他評估設定由 train.yaml 控制。
        """
        if weights is None:
            raise ValueError(
                "weights 不可為 None。請傳入要驗證的 checkpoint 路徑。"
            )

        checkpoint_path = Path(
            weights
        ).expanduser().resolve()

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"找不到 weights：{checkpoint_path}"
            )

        model_cfg = deepcopy_cfg(
            self.model_cfg
        )
        train_cfg = load_train_config(
            cfg
        )

        if data is not None:
            train_cfg["data"][
                "dataset_dir"
            ] = str(data)

        if imgsz is not None:
            train_cfg["data"][
                "image_size"
            ] = int(imgsz)

        if batch is not None:
            train_cfg["train"][
                "batch_size"
            ] = int(batch)

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

        if workers is not None:
            train_cfg["train"][
                "num_workers"
            ] = int(workers)

        # Hybrid validation contract:
        # - auxiliary branch disabled
        # - official metric does not use NMS
        model_cfg["model"][
            "auxiliary_in_eval"
        ] = False

        train_cfg["eval"][
            "use_nms"
        ] = False

        output_dir = Path(
            project
        ) / str(name)

        return validate(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            checkpoint_path=(
                checkpoint_path
            ),
            output_dir=output_dir,
            prefer_ema=bool(
                prefer_ema
            ),
            compute_loss=bool(
                compute_loss
            ),
            save_json=bool(
                save_json
            ),
        )


def main() -> None:
    os.environ[
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    ] = "1"

    model = LightDet(
        model=(
            "/home/soic/Desktop/LightDet/"
            "units/model/cards/config/model.yaml"
        )
    )

    model.val(
        cfg=(
            "/home/soic/Desktop/LightDet/"
            "units/model/cards/config/train.yaml"
        ),
        weights=(
            "/home/soic/Desktop/LightDet/units/model/runs/train/lightdet_HDETR_transformer_layer_decoupled_v2/epoch_100.pt"
        ),
        data=(
            "/home/soic/Desktop/LightDet/"
            "datasets"
        ),
        imgsz=512,
        batch=48,
        device=0,
        workers=8,
        project="runs/val",
        name="lightdet_hybrid_best",
        prefer_ema=False,
        compute_loss=False,
        save_json=True,
    )


if __name__ == "__main__":
    main()
