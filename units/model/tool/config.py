from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import yaml

MODEL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODEL_DIR.parents[1]
CONFIG_DIR = MODEL_DIR / "cards" / "config"
DEFAULT_MODEL_CONFIG_PATH = CONFIG_DIR / "model.yaml"
DEFAULT_TRAIN_CONFIG_PATH = CONFIG_DIR / "train.yaml"


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
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"YAML config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"YAML config must be a dict: {config_path}")
    return config


def _project_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def _require_sections(config: Dict[str, Any], sections, path: str) -> None:
    missing = [name for name in sections if name not in config]
    if missing:
        raise KeyError(f"Missing config sections in {path}: {missing}")


def _require_mapping(value: Any, name: str, path: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(
            f"{name} must be a mapping in {path}, "
            f"got {type(value).__name__}"
        )
    return value


def _require_keys(
    config: Dict[str, Any],
    keys,
    name: str,
    path: str,
) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError(
            f"Missing keys in {name} from {path}: {missing}"
        )


def _validate_model_config(config: Dict[str, Any], path: str) -> None:
    model = _require_mapping(config["model"], "model", path)
    _require_keys(
        model,
        (
            "hidden_dim",
            "backbone",
            "fpn",
            "image_projector",
            "num_object_queries",
            "query_group_init_std",
            "fusion_token_num",
            "num_heads",
            "num_layers",
            "mlp_ratio",
            "dropout",
            "text_max_length",
            "freeze_bert",
        ),
        "model",
        path,
    )

    backbone = _require_mapping(
        model["backbone"],
        "model.backbone",
        path,
    )
    fpn = _require_mapping(
        model["fpn"],
        "model.fpn",
        path,
    )
    projector = _require_mapping(
        model["image_projector"],
        "model.image_projector",
        path,
    )

    _require_keys(
        backbone,
        (
            "in_channels",
            "base_channels",
            "base_depths",
            "width_multiple",
            "depth_multiple",
            "max_channels",
        ),
        "model.backbone",
        path,
    )
    _require_keys(
        fpn,
        ("out_channels",),
        "model.fpn",
        path,
    )
    _require_keys(
        projector,
        (
            "in_channels",
            "out_channels",
            "layer_num",
            "expand_ratio",
            "level_names",
            "token_grids",
        ),
        "model.image_projector",
        path,
    )

    base_channels = backbone["base_channels"]
    base_depths = backbone["base_depths"]
    level_names = projector["level_names"]
    token_grids = projector["token_grids"]

    if not isinstance(base_channels, (list, tuple)) or len(base_channels) != 5:
        raise ValueError(
            "model.backbone.base_channels must contain 5 values"
        )

    if not isinstance(base_depths, (list, tuple)) or len(base_depths) != 3:
        raise ValueError(
            "model.backbone.base_depths must contain 3 values"
        )

    if not isinstance(level_names, (list, tuple)) or not level_names:
        raise ValueError(
            "model.image_projector.level_names must not be empty"
        )

    if not isinstance(token_grids, (list, tuple)):
        raise TypeError(
            "model.image_projector.token_grids must be a sequence"
        )

    if len(level_names) != len(token_grids):
        raise ValueError(
            "model.image_projector.level_names and token_grids "
            "must have the same length"
        )

    for index, grid in enumerate(token_grids):
        if not isinstance(grid, (list, tuple)) or len(grid) != 2:
            raise ValueError(
                "Each model.image_projector.token_grids entry "
                f"must contain 2 values, got index {index}: {grid}"
            )
        if int(grid[0]) <= 0 or int(grid[1]) <= 0:
            raise ValueError(
                "Each model.image_projector.token_grids value "
                f"must be > 0, got index {index}: {grid}"
            )

    hidden_dim = int(model["hidden_dim"])
    fpn_out_channels = int(fpn["out_channels"])
    projector_in_channels = int(projector["in_channels"])
    projector_out_channels = int(projector["out_channels"])

    if projector_in_channels != fpn_out_channels:
        raise ValueError(
            "model.image_projector.in_channels must equal "
            "model.fpn.out_channels"
        )

    if projector_out_channels != hidden_dim:
        raise ValueError(
            "model.image_projector.out_channels must equal "
            "model.hidden_dim"
        )


def load_model_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_path = str(path or DEFAULT_MODEL_CONFIG_PATH)
    config = load_yaml(config_path)
    _require_sections(config, ("model",), config_path)
    _validate_model_config(config, config_path)
    model = config["model"]
    model["precomputed_bert_path"] = _project_path(
        model.get("precomputed_bert_path")
    )
    return config


def load_train_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_path = str(path or DEFAULT_TRAIN_CONFIG_PATH)
    config = load_yaml(config_path)
    _require_sections(
        config,
        ("data", "train", "optim", "loss", "eval", "log"),
        config_path,
    )
    data = config["data"]
    for key in ("dataset_dir", "image_cache_dir", "negative_query_path"):
        data[key] = _project_path(data.get(key))
    log = config["log"]
    for key in ("save_dir", "weights_path", "resume_path"):
        log[key] = _project_path(log.get(key))
    return config

def normalize_device(device: Any) -> str:
    if device is None:
        if torch.cuda.is_available():
            return "cuda:0"

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return "mps"

        return "cpu"

    if isinstance(device, bool):
        raise TypeError(
            "device cannot be bool"
        )

    if isinstance(device, int):
        device = f"cuda:{device}"

    elif isinstance(device, str):
        device = device.strip().lower()

        if not device:
            raise ValueError(
                "device cannot be empty"
            )

        if device.isdigit():
            device = f"cuda:{device}"

    else:
        raise TypeError(
            "Unsupported device type: "
            f"{type(device).__name__}"
        )

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return "mps"

        return "cpu"

    if device == "cpu":
        return "cpu"

    if device == "mps":
        mps_available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )

        if not mps_available:
            raise RuntimeError(
                "MPS was requested, but it is not available"
            )

        return "mps"

    if device == "cuda":
        device = "cuda:0"

    if device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but CUDA is not available"
            )

        index_text = device.split(":", maxsplit=1)[1]

        if not index_text.isdigit():
            raise ValueError(
                "CUDA device must use format cuda:N, "
                f"got {device!r}"
            )

        device_index = int(index_text)
        device_count = torch.cuda.device_count()

        if device_index < 0 or device_index >= device_count:
            raise ValueError(
                "CUDA device index is out of range: "
                f"requested={device_index}, "
                f"available=0-{device_count - 1}"
            )

        return f"cuda:{device_index}"

    raise ValueError(
        "Unsupported device value: "
        f"{device!r}. "
        "Supported values are auto, cpu, mps, "
        "cuda, cuda:N, or integer N."
    )


def _parse_component_schedule(
    component: str,
    value: Any,
    *,
    legacy_max_lr: float,
    legacy_min_ratio: float,
) -> list:
    if value is None:
        return [
            "cosine",
            float(legacy_max_lr),
            float(legacy_max_lr) * float(legacy_min_ratio),
            0.0,
            1.0,
        ]

    if not isinstance(value, (list, tuple)) or len(value) != 5:
        raise ValueError(
            f"optim.components.{component} must be "
            "[mode, max_lr, min_lr, start_epoch, end_epoch]"
        )

    mode = str(value[0]).strip().lower()

    if mode not in {"constant", "linear", "cosine", "freeze"}:
        raise ValueError(
            f"Unsupported optimizer mode for {component}: {mode}"
        )

    max_lr = float(value[1])
    min_lr = float(value[2])
    start_epoch = float(value[3])
    end_epoch = float(value[4])

    if max_lr < 0.0 or min_lr < 0.0:
        raise ValueError(
            f"{component} learning rates must be >= 0"
        )

    if min_lr > max_lr:
        raise ValueError(
            f"{component} min_lr must be <= max_lr"
        )

    if start_epoch < 0.0 or end_epoch < start_epoch:
        raise ValueError(
            f"Invalid schedule range for {component}: "
            f"{start_epoch} -> {end_epoch}"
        )

    return [
        mode,
        max_lr,
        min_lr,
        start_epoch,
        end_epoch,
    ]


def cfg_to_args(
    model_cfg_all: Dict[str, Any],
    train_cfg_all: Dict[str, Any],
) -> SimpleNamespace:
    model_cfg = model_cfg_all["model"]

    backbone_cfg = model_cfg["backbone"]
    fpn_cfg = model_cfg["fpn"]
    image_projector_cfg = model_cfg[
        "image_projector"
    ]

    data_cfg = train_cfg_all["data"]
    train_cfg = train_cfg_all["train"]
    optim_cfg = train_cfg_all["optim"]
    loss_cfg = train_cfg_all["loss"]
    eval_cfg = train_cfg_all["eval"]
    log_cfg = train_cfg_all["log"]

    hybrid_cfg = loss_cfg.get("hybrid", {})
    matcher_cfg = loss_cfg.get("matcher", {})
    weight_cfg = loss_cfg.get("weight", {})
    pos_weight_cfg = loss_cfg.get("pos_weight", {})
    score_sampling_cfg = loss_cfg.get("score_sampling", {})
    quality_cfg = loss_cfg.get("quality", {})
    ranking_cfg = loss_cfg.get("ranking", {})
    classification_cfg = loss_cfg.get("classification", {})
    matcher_schedule_cfg = loss_cfg.get("matcher_schedule", {})
    text_negative_cfg = loss_cfg.get("text_negative", {})
    duplicate_cfg = loss_cfg.get("duplicate_suppression", {})
    hard_negative_cfg = loss_cfg.get("hard_negative", {})

    legacy_min_lr_ratio = float(
        optim_cfg.get("min_lr_ratio", 0.05)
    )
    component_cfg = optim_cfg.get("components", {})
    component_schedules = {
        "vision": _parse_component_schedule(
            "vision",
            component_cfg.get("vision"),
            legacy_max_lr=float(
                optim_cfg.get("lr_vision", 1e-4)
            ),
            legacy_min_ratio=legacy_min_lr_ratio,
        ),
        "text": _parse_component_schedule(
            "text",
            component_cfg.get("text"),
            legacy_max_lr=float(
                optim_cfg.get("lr_text", 1e-5)
            ),
            legacy_min_ratio=legacy_min_lr_ratio,
        ),
        "transformer": _parse_component_schedule(
            "transformer",
            component_cfg.get("transformer"),
            legacy_max_lr=float(
                optim_cfg.get("lr_transformer", 1e-4)
            ),
            legacy_min_ratio=legacy_min_lr_ratio,
        ),
        "head": _parse_component_schedule(
            "head",
            component_cfg.get("head"),
            legacy_max_lr=float(
                optim_cfg.get("lr_head", 1e-4)
            ),
            legacy_min_ratio=legacy_min_lr_ratio,
        ),
    }

    # Backward compatibility: an old fixed threshold becomes a constant
    # schedule unless any dynamic key is explicitly present.
    legacy_score_ignore = quality_cfg.get(
        "score_negative_iou_ignore_thr"
    )
    has_dynamic_score_ignore = any(
        key in quality_cfg
        for key in (
            "score_negative_iou_ignore_start",
            "score_negative_iou_ignore_end",
            "score_negative_iou_ignore_start_epoch",
            "score_negative_iou_ignore_end_epoch",
            "score_negative_iou_ignore_schedule",
        )
    )
    if legacy_score_ignore is not None and not has_dynamic_score_ignore:
        score_ignore_start = float(legacy_score_ignore)
        score_ignore_end = float(legacy_score_ignore)
        score_ignore_start_epoch = 1
        score_ignore_end_epoch = 1
        score_ignore_schedule = "constant"
    else:
        score_ignore_start = float(
            quality_cfg.get("score_negative_iou_ignore_start", 0.50)
        )
        score_ignore_end = float(
            quality_cfg.get("score_negative_iou_ignore_end", 0.45)
        )
        score_ignore_start_epoch = int(
            quality_cfg.get(
                "score_negative_iou_ignore_start_epoch",
                5,
            )
        )
        score_ignore_end_epoch = int(
            quality_cfg.get(
                "score_negative_iou_ignore_end_epoch",
                25,
            )
        )
        score_ignore_schedule = str(
            quality_cfg.get(
                "score_negative_iou_ignore_schedule",
                "cosine",
            )
        ).strip().lower()

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
        device=normalize_device(train_cfg.get("device")),
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
        backbone_config={
            "in_channels": int(
                backbone_cfg["in_channels"]
            ),
            "base_channels": tuple(
                int(value)
                for value in backbone_cfg[
                    "base_channels"
                ]
            ),
            "base_depths": tuple(
                int(value)
                for value in backbone_cfg[
                    "base_depths"
                ]
            ),
            "width_multiple": float(
                backbone_cfg["width_multiple"]
            ),
            "depth_multiple": float(
                backbone_cfg["depth_multiple"]
            ),
            "max_channels": int(
                backbone_cfg["max_channels"]
            ),
            "channel_divisor": int(
                backbone_cfg.get(
                    "channel_divisor",
                    8,
                )
            ),
        },

        fpn_config={
            "out_channels": int(
                fpn_cfg["out_channels"]
            ),
            "norm_layer": fpn_cfg.get(
                "norm_layer"
            ),
        },

        image_projector_config={
            "in_channels": int(
                image_projector_cfg[
                    "in_channels"
                ]
            ),
            "out_channels": int(
                image_projector_cfg[
                    "out_channels"
                ]
            ),
            "layer_num": int(
                image_projector_cfg[
                    "layer_num"
                ]
            ),
            "expand_ratio": float(
                image_projector_cfg[
                    "expand_ratio"
                ]
            ),
            "level_names": tuple(
                str(value)
                for value in image_projector_cfg[
                    "level_names"
                ]
            ),
            "token_grids": tuple(
                (
                    int(grid[0]),
                    int(grid[1]),
                )
                for grid in image_projector_cfg[
                    "token_grids"
                ]
            ),
        },

        hidden_dim=int(
            model_cfg["hidden_dim"]
        ),

        freeze_img_projection=bool(
            model_cfg.get(
                "freeze_img_projection",
                False,
            )
        ),

        num_object_queries=int(
            model_cfg["num_object_queries"]
        ),

        query_group_init_std=float(
            model_cfg["query_group_init_std"]
        ),

        fusion_token_num=int(
            model_cfg["fusion_token_num"]
        ),

        num_heads=int(
            model_cfg["num_heads"]
        ),

        num_layers=int(
            model_cfg["num_layers"]
        ),

        mlp_ratio=float(
            model_cfg["mlp_ratio"]
        ),

        dropout=float(
            model_cfg["dropout"]
        ),

        staged_query_refinement=bool(
            model_cfg.get(
                "staged_query_refinement",
                True,
            )
        ),

        score_num_heads=int(
            model_cfg.get(
                "score_num_heads",
                model_cfg["num_heads"],
            )
        ),

        score_num_layers=int(
            model_cfg.get(
                "score_num_layers",
                model_cfg["num_layers"],
            )
        ),

        score_mlp_ratio=float(
            model_cfg.get(
                "score_mlp_ratio",
                model_cfg["mlp_ratio"],
            )
        ),

        score_dropout=float(
            model_cfg.get(
                "score_dropout",
                model_cfg["dropout"],
            )
        ),

        score_bbox_conditioning=bool(
            model_cfg.get(
                "score_bbox_conditioning",
                True,
            )
        ),

        score_bbox_detach=bool(
            model_cfg.get(
                "score_bbox_detach",
                True,
            )
        ),

        score_fusion=str(
            model_cfg.get(
                "score_fusion",
                "geometric_mean",
            )
        ),

        score_fusion_eps=float(
            model_cfg.get(
                "score_fusion_eps",
                1e-6,
            )
        ),

        text_max_length=int(
            model_cfg["text_max_length"]
        ),

        freeze_bert=bool(
            model_cfg["freeze_bert"]
        ),

        precomputed_bert_path=model_cfg.get(
            "precomputed_bert_path"
        ),

        use_auxiliary_head=bool(
            model_cfg.get(
                "use_auxiliary_head",
                True,
            )
        ),

        auxiliary_in_eval=bool(
            model_cfg.get(
                "auxiliary_in_eval",
                False,
            )
        ),

        initialize_aux_from_main=bool(
            model_cfg.get(
                "initialize_aux_from_main",
                True,
            )
        ),

        # optimizer
        lr_vision=float(component_schedules["vision"][1]),
        lr_text=float(component_schedules["text"][1]),
        lr_transformer=float(
            component_schedules["transformer"][1]
        ),
        lr_head=float(component_schedules["head"][1]),
        component_schedules=component_schedules,
        weight_decay=optim_cfg["weight_decay"],
        min_lr_ratio=legacy_min_lr_ratio,
        max_warmup_steps=optim_cfg.get("max_warmup_steps", 3000),
        fused_optimizer=optim_cfg.get("fused", True),

        # hybrid loss
        aux_loss_weight=float(
            hybrid_cfg.get("aux_loss_weight", 0.5)
        ),
        aux_cost_score=float(
            hybrid_cfg.get("aux_cost_score", 0.0)
        ),

        # matcher
        cost_bbox=matcher_cfg.get("cost_bbox", 5.0),
        cost_giou=matcher_cfg.get("cost_giou", 2.0),
        cost_score=float(matcher_cfg.get("cost_score", 2.0)),
        matcher_score_cost_type=str(
            matcher_cfg.get("score_cost_type", "focal")
        ),
        matcher_focal_alpha=float(
            matcher_cfg.get("focal_alpha", 0.25)
        ),
        matcher_focal_gamma=float(
            matcher_cfg.get("focal_gamma", 2.0)
        ),

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
        quality_warmup_epoch=quality_cfg.get("quality_warmup_epoch", 10),

        # DETR-native matched-query classification.
        classification_type=str(
            classification_cfg.get("type", "ia_bce")
        ).strip().lower(),
        ia_bce_alpha=float(
            classification_cfg.get("ia_bce_alpha", 0.25)
        ),
        classification_focal_alpha=float(
            classification_cfg.get("focal_alpha", 0.25)
        ),
        classification_focal_gamma=float(
            classification_cfg.get("focal_gamma", 2.0)
        ),
        normalize_classification_by_num_gt=bool(
            classification_cfg.get("normalize_by_num_gt", True)
        ),
        score_negative_iou_ignore_thr=float(
            classification_cfg.get(
                "negative_iou_ignore_thr",
                quality_cfg.get(
                    "score_negative_iou_ignore_thr",
                    0.50,
                ),
            )
        ),

        # Duplicate-like unmatched queries.
        duplicate_suppression_enabled=bool(
            duplicate_cfg.get("enabled", True)
        ),
        duplicate_loss_weight=float(
            duplicate_cfg.get(
                "loss_weight",
                duplicate_cfg.get("lambda_duplicate", 0.10),
            )
        ),
        duplicate_margin=float(
            duplicate_cfg.get("margin", 0.25)
        ),
        duplicate_background_weight=float(
            duplicate_cfg.get("background_weight", 0.05)
        ),
        # Duplicate-like unmatched queries remain target=0 negatives in the
        # main classification loss, but use a reduced weight to avoid
        # suppressing unstable assignment candidates too aggressively.
        duplicate_classification_weight=float(
            duplicate_cfg.get("classification_weight", 0.25)
        ),
        duplicate_max_pairs=int(
            duplicate_cfg.get("max_pairs", 128)
        ),
        duplicate_start_epoch=int(
            duplicate_cfg.get("start_epoch", 5)
        ),

        # High-score, low-IoU unmatched queries.
        hard_negative_mining_enabled=bool(
            hard_negative_cfg.get(
                "enabled",
                score_sampling_cfg.get(
                    "hard_negative_mining_enabled",
                    True,
                ),
            )
        ),
        hard_negative_loss_weight=float(
            hard_negative_cfg.get(
                "loss_weight",
                score_sampling_cfg.get(
                    "hard_negative_loss_weight",
                    0.05,
                ),
            )
        ),
        hard_negative_topk=int(
            hard_negative_cfg.get(
                "topk",
                score_sampling_cfg.get("hard_negative_topk", 10),
            )
        ),
        hard_negative_max_iou=float(
            hard_negative_cfg.get(
                "max_iou",
                score_sampling_cfg.get(
                    "hard_negative_max_iou",
                    0.30,
                ),
            )
        ),
        hard_negative_start_epoch=int(
            hard_negative_cfg.get(
                "start_epoch",
                score_sampling_cfg.get(
                    "hard_negative_start_epoch",
                    10,
                ),
            )
        ),

        negative_text_as_empty_target=bool(
            text_negative_cfg.get("as_empty_target", True)
        ),

        # Legacy dense-score fields are retained only for old YAML parsing.
        score_match_rounds=int(
            quality_cfg.get("score_match_rounds", 1)
        ),
        score_quality_gamma=float(
            quality_cfg.get("score_quality_gamma", 1.0)
        ),
        score_round_decay=float(
            quality_cfg.get("score_round_decay", 0.25)
        ),
        score_min_iou=float(
            quality_cfg.get("score_min_iou", 0.0)
        ),
        score_negative_iou_ignore_start=score_ignore_start,
        score_negative_iou_ignore_end=score_ignore_end,
        score_negative_iou_ignore_start_epoch=(
            score_ignore_start_epoch
        ),
        score_negative_iou_ignore_end_epoch=(
            score_ignore_end_epoch
        ),
        score_negative_iou_ignore_schedule=(
            score_ignore_schedule
        ),
        aux_score_enabled=bool(
            quality_cfg.get("aux_score_enabled", False)
        ),

        # Pairwise score ranking is opt-in.
        ranking_enabled=bool(
            ranking_cfg.get("enabled", False)
        ),
        lambda_rank=ranking_cfg.get("lambda_rank", 0.10),
        rank_margin=ranking_cfg.get("rank_margin", 0.1),
        rank_min_quality_gap=ranking_cfg.get("rank_min_quality_gap", 0.1),
        rank_max_pairs=ranking_cfg.get("rank_max_pairs", 512),
        rank_start_epoch=int(
            matcher_schedule_cfg.get(
                "start_epoch",
                ranking_cfg.get("rank_start_epoch", 5),
            )
        ),
        rank_warmup_epoch=int(
            matcher_schedule_cfg.get(
                "warmup_epoch",
                ranking_cfg.get("rank_warmup_epoch", 12),
            )
        ),
        rank_alpha_min=float(
            matcher_schedule_cfg.get(
                "alpha_min",
                ranking_cfg.get("rank_alpha_min", 0.10),
            )
        ),
        rank_negative_iou_max=float(
            ranking_cfg.get("rank_negative_iou_max", 0.20)
        ),
        max_query_loss_weight=text_negative_cfg.get(
            "max_query_loss_weight",
            10.0,
        ),
        lambda_text_negative=float(
            text_negative_cfg.get("lambda_text_negative", 0.50)
        ),
        text_negative_topk=int(
            text_negative_cfg.get("text_negative_topk", 20)
        ),
        text_negative_hard_mix=float(
            text_negative_cfg.get("text_negative_hard_mix", 0.50)
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
        compute_raw_oracle=bool(
            eval_cfg.get("compute_raw_oracle", True)
        ),
        raw_oracle_iou_thresholds=eval_cfg.get(
            "raw_oracle_iou_thresholds",
            [0.25, 0.50, 0.75],
        ),
        best_metric=eval_cfg["best_metric"],
        use_topk_fallback=eval_cfg.get("use_topk_fallback", False),

        # log
        save_dir=log_cfg["save_dir"],
        weights_path=log_cfg.get("weights_path"),
        resume_path=log_cfg.get("resume_path"),
        prefer_ema=bool(log_cfg.get("prefer_ema", True)),
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

    hybrid = loss.get("hybrid", {})
    matcher = loss.get("matcher", {})
    weight = loss.get("weight", {})
    pos_weight = loss.get("pos_weight", {})
    quality = loss.get("quality", {})
    ranking = loss.get("ranking", {})
    classification = loss.get("classification", {})
    matcher_schedule = loss.get("matcher_schedule", {})
    score_sampling = loss.get("score_sampling", {})
    text_negative = loss.get("text_negative", {})
    duplicate = loss.get("duplicate_suppression", {})
    hard_negative = loss.get("hard_negative", {})

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
        f"max_weight={text_negative.get('max_query_loss_weight', 10.0)}, "
        f"lambda={text_negative.get('lambda_text_negative', 0.50)}, "
        f"topk={text_negative.get('text_negative_topk', 20)}, "
        f"empty_target={text_negative.get('as_empty_target', True)}"
    )
    print(f"  epochs       : {training['epochs']}")
    print(f"  batch        : {training['batch_size']}")
    print(f"  device       : {training['device']}")
    print(f"  seed         : {training['seed']}")
    print(f"  save_dir     : {logging['save_dir']}")
    print(f"  weights      : {logging.get('weights_path')}")
    print(f"  resume       : {logging.get('resume_path')}")
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
    backbone = model["backbone"]
    fpn = model["fpn"]
    projector = model["image_projector"]
    level_names = projector["level_names"]
    token_grids = projector["token_grids"]
    tokens_per_level = {
        str(name): int(grid[0]) * int(grid[1])
        for name, grid in zip(level_names, token_grids)
    }
    total_image_tokens = sum(tokens_per_level.values())

    print(f"  hidden_dim   : {model['hidden_dim']}")
    print(
        f"  backbone     : width={backbone['width_multiple']}, "
        f"depth={backbone['depth_multiple']}"
    )
    print(f"  base channels: {backbone['base_channels']}")
    print(f"  base depths  : {backbone['base_depths']}")
    print(f"  fpn channels : {fpn['out_channels']}")
    print(f"  token grids  : {token_grids}")
    print(f"  token levels : {tokens_per_level}")
    print(f"  image tokens : {total_image_tokens}")
    print(
        f"  projector    : layers={projector['layer_num']}, "
        f"expand={projector['expand_ratio']}"
    )
    print(f"  object query : {model.get('num_object_queries', 100)}")
    print(f"  num_layers   : {model['num_layers']}")
    print(f"  num_heads    : {model['num_heads']}")
    print(f"  mlp_ratio    : {model['mlp_ratio']}")
    print(f"  bert_cache   : {model.get('precomputed_bert_path')}")
    print(
        f"  hybrid head  : enabled="
        f"{model.get('use_auxiliary_head', True)}, "
        f"aux_eval={model.get('auxiliary_in_eval', False)}, "
        f"init_from_main="
        f"{model.get('initialize_aux_from_main', True)}"
    )
    components = optimizer.get("components")

    if isinstance(components, dict):
        for component in (
            "vision",
            "text",
            "transformer",
            "head",
        ):
            print(
                f"  lr_{component:<11}: "
                f"{components.get(component)}"
            )
    else:
        print(f"  lr_vision    : {optimizer['lr_vision']}")
        print(f"  lr_text      : {optimizer['lr_text']}")
        print(f"  lr_trans     : {optimizer['lr_transformer']}")
        print(f"  lr_head      : {optimizer['lr_head']}")
    print(f"  fused AdamW  : {optimizer.get('fused', True)}")
    print(
        f"  hybrid loss  : weight="
        f"{hybrid.get('aux_loss_weight', 0.5)}, "
        f"aux_score_cost={hybrid.get('aux_cost_score', 0.0)}"
    )
    print(
        f"  matcher      : bbox={matcher.get('cost_bbox', 5.0)}, "
        f"giou={matcher.get('cost_giou', 2.0)}, "
        f"score={matcher.get('cost_score', 2.0)}"
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
        f"  aux matching : repeat_k="
        f"{score_sampling.get('max_positive_per_gt', 2)}, "
        f"iou_thr={quality.get('iou_pos_thr', 0.10)}, "
        f"aux_score={quality.get('aux_score_enabled', False)}"
    )
    print(
        f"  class loss   : type={classification.get('type', 'ia_bce')}, "
        f"ia_alpha={classification.get('ia_bce_alpha', 0.25)}, "
        f"focal_alpha={classification.get('focal_alpha', 0.25)}, "
        f"focal_gamma={classification.get('focal_gamma', 2.0)}, "
        f"norm_gt={classification.get('normalize_by_num_gt', True)}, "
        f"ignore_iou={classification.get('negative_iou_ignore_thr', 0.50)}, "
        f"quality_warmup={quality.get('quality_warmup_epoch', 10)}"
    )
    print(
        f"  duplicate    : enabled={duplicate.get('enabled', True)}, "
        f"weight={duplicate.get('loss_weight', 0.10)}, "
        f"margin={duplicate.get('margin', 0.25)}, "
        f"bg_weight={duplicate.get('background_weight', 0.05)}, "
        f"max_pairs={duplicate.get('max_pairs', 128)}, "
        f"start={duplicate.get('start_epoch', 5)}"
    )
    print(
        f"  hard negative: enabled={hard_negative.get('enabled', True)}, "
        f"weight={hard_negative.get('loss_weight', 0.05)}, "
        f"topk={hard_negative.get('topk', 10)}, "
        f"max_iou={hard_negative.get('max_iou', 0.30)}, "
        f"start={hard_negative.get('start_epoch', 10)}"
    )
    print(
        f"  matcher sched: start="
        f"{matcher_schedule.get('start_epoch', ranking.get('rank_start_epoch', 5))}, "
        f"warmup="
        f"{matcher_schedule.get('warmup_epoch', ranking.get('rank_warmup_epoch', 12))}, "
        f"alpha_min="
        f"{matcher_schedule.get('alpha_min', ranking.get('rank_alpha_min', 0.10))}"
    )
    print(
        f"  pairwise rank: enabled={ranking.get('enabled', False)}, "
        f"lambda={ranking.get('lambda_rank', 0.0)}, "
        f"start={ranking.get('rank_start_epoch', 5)}, "
        f"warmup={ranking.get('rank_warmup_epoch', 12)}"
    )
    print(f"  pos weight   : {pos_weight.get('value', 1.0)}")
    print(
        f"  eval         : metric={evaluation['best_metric']}, "
        f"score_thr={evaluation['score_thr']}, "
        f"top_k={evaluation['top_k']}, "
        f"use_nms={evaluation.get('use_nms', True)}, "
        f"max_batches={evaluation.get('max_val_batches')}"
    )
    print(
        f"  raw oracle   : enabled="
        f"{evaluation.get('compute_raw_oracle', True)}, "
        f"iou={evaluation.get('raw_oracle_iou_thresholds', [0.25, 0.5, 0.75])}"
    )
    print("")
