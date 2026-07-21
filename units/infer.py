#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
LightDet YOLO-style inference entry.

Example:
    from units.infer import LightDet

    model = LightDet(model="/path/to/model.yaml")
    results = model.predict(
        weights="/path/to/best_map50_95.pt",
        source="/path/to/image.jpg",
        text=["一台行駛的計程車", "停靠的船"],
        imgsz=512,
        device=0,
        conf=0.001,
        top_k=100,
        use_nms=True,
        nms_iou_threshold=0.6,
        project="runs/predict",
        name="exp",
    )

Hybrid inference contract:
    - only the Main One-to-One branch is executed;
    - auxiliary predictions are disabled;
    - NMS can be enabled or disabled;
    - NMS is executed independently for each text query;
    - one image is encoded only once for multiple text queries.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.ops import nms


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

for path in (PROJECT_ROOT, CURRENT_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


from units.model.train import (  # noqa: E402
    deepcopy_cfg,
    normalize_device,
)
from units.validate import LightDet as ValidateLightDet  # noqa: E402
from units.tool.card import VisionTextModel  # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def normalize_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Remove common wrapper prefixes only when all keys use the prefix."""
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
    """Select EMA/model weights from a LightDet checkpoint or raw state dict."""
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a dict/state_dict, "
            f"got {type(checkpoint)}"
        )

    ema_state = checkpoint.get("ema")
    model_state = checkpoint.get("model")
    legacy_model_state = checkpoint.get("model_state_dict")
    generic_state = checkpoint.get("state_dict")

    if prefer_ema and isinstance(ema_state, dict) and ema_state:
        return normalize_state_dict_keys(ema_state), "ema"

    for state, source in (
        (model_state, "model"),
        (legacy_model_state, "model_state_dict"),
        (ema_state, "ema"),
        (generic_state, "state_dict"),
    ):
        if isinstance(state, dict) and state:
            return normalize_state_dict_keys(state), source

    if checkpoint and all(
        torch.is_tensor(value)
        for value in checkpoint.values()
    ):
        return normalize_state_dict_keys(checkpoint), "raw_state_dict"

    raise KeyError(
        "Checkpoint does not contain ema, model, model_state_dict, "
        "state_dict, or a raw state_dict."
    )


def load_checkpoint_for_inference(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    prefer_ema: bool = True,
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    if not checkpoint_path.is_file():
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

    state_dict, source = select_checkpoint_state_dict(
        checkpoint=checkpoint,
        prefer_ema=prefer_ema,
    )

    # VisionTextModel.load_state_dict() handles legacy single-head checkpoints
    # by copying head.* weights into aux_head.* when needed.
    load_result = model.load_state_dict(
        state_dict,
        strict=True,
    )

    checkpoint_epoch = (
        int(checkpoint.get("epoch", 0))
        if isinstance(checkpoint, dict)
        else 0
    )
    best_metric = (
        float(checkpoint.get("best_metric", -1.0))
        if isinstance(checkpoint, dict)
        else -1.0
    )
    best_metric_name = (
        str(checkpoint.get("best_metric_name", "unknown"))
        if isinstance(checkpoint, dict)
        else "unknown"
    )

    return {
        "path": str(checkpoint_path),
        "source": source,
        "epoch": checkpoint_epoch,
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


# ---------------------------------------------------------------------------
# Model / device
# ---------------------------------------------------------------------------


def build_inference_model(
    model_cfg_all: Dict[str, Any],
) -> VisionTextModel:
    cfg = model_cfg_all["model"]

    return VisionTextModel(
        img_in_channels=cfg["img_in_channels"],
        hidden_dim=cfg["hidden_dim"],
        target_size=(
            int(cfg["image_grid_size"]),
            int(cfg["image_grid_size"]),
        ),
        text_max_length=cfg["text_max_length"],
        fusion_token_num=cfg["fusion_token_num"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        mlp_ratio=cfg["mlp_ratio"],
        dropout=cfg["dropout"],
        freeze_bert=cfg["freeze_bert"],
        precomputed_bert_path=cfg.get("precomputed_bert_path"),
        use_auxiliary_head=bool(
            cfg.get("use_auxiliary_head", True)
        ),
        # Inference always disables auxiliary predictions.
        auxiliary_in_eval=False,
        initialize_aux_from_main=bool(
            cfg.get("initialize_aux_from_main", True)
        ),
    )


def resolve_inference_device(
    device: Optional[Any],
) -> torch.device:
    normalized = normalize_device(device)
    resolved = torch.device(normalized)

    if resolved.type != "cuda":
        return resolved

    if not torch.cuda.is_available():
        print("[Warning] CUDA unavailable; falling back to CPU.")
        return torch.device("cpu")

    index = 0 if resolved.index is None else int(resolved.index)

    if index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested cuda:{index}, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are available."
        )

    return torch.device(f"cuda:{index}")


# ---------------------------------------------------------------------------
# Input / box helpers
# ---------------------------------------------------------------------------


def normalize_queries(
    text: str | Sequence[str],
) -> List[str]:
    if isinstance(text, str):
        candidates = [text]
    else:
        candidates = list(text)

    queries = [
        str(query).strip()
        for query in candidates
        if str(query).strip()
    ]

    if not queries:
        raise ValueError(
            "text must contain at least one non-empty query."
        )

    return queries


def preprocess_image(
    source: str | Path,
    image_size: int,
) -> Tuple[torch.Tensor, np.ndarray, int, int]:
    source = Path(source).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(
            f"Input image not found: {source}"
        )

    try:
        pil_image = Image.open(source).convert("RGB")
    except Exception as error:
        raise RuntimeError(
            f"Unable to read image: {source}"
        ) from error

    original_width, original_height = pil_image.size

    transform = transforms.Compose([
        transforms.Resize(
            (int(image_size), int(image_size)),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
    ])

    image_tensor = transform(pil_image).unsqueeze(0)

    original_bgr = cv2.cvtColor(
        np.asarray(pil_image),
        cv2.COLOR_RGB2BGR,
    )

    return (
        image_tensor,
        original_bgr,
        int(original_width),
        int(original_height),
    )


def sanitize_xyxy_boxes(
    boxes: torch.Tensor,
) -> torch.Tensor:
    boxes = boxes.clamp(0.0, 1.0)

    x1 = torch.minimum(
        boxes[..., 0],
        boxes[..., 2],
    )
    y1 = torch.minimum(
        boxes[..., 1],
        boxes[..., 3],
    )
    x2 = torch.maximum(
        boxes[..., 0],
        boxes[..., 2],
    )
    y2 = torch.maximum(
        boxes[..., 1],
        boxes[..., 3],
    )

    return torch.stack(
        [x1, y1, x2, y2],
        dim=-1,
    )


def normalized_xyxy_to_pixels(
    boxes: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    boxes = sanitize_xyxy_boxes(boxes)

    scale = boxes.new_tensor([
        float(width),
        float(height),
        float(width),
        float(height),
    ])

    pixel_boxes = boxes * scale

    pixel_boxes[..., 0].clamp_(
        0,
        max(width - 1, 0),
    )
    pixel_boxes[..., 1].clamp_(
        0,
        max(height - 1, 0),
    )
    pixel_boxes[..., 2].clamp_(
        0,
        max(width - 1, 0),
    )
    pixel_boxes[..., 3].clamp_(
        0,
        max(height - 1, 0),
    )

    return pixel_boxes


def select_main_predictions(
    boxes: torch.Tensor,
    score_logits: torch.Tensor,
    confidence_threshold: float,
    top_k: int,
    width: int,
    height: int,
    use_nms: bool = True,
    nms_iou_threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    Select Main predictions using:

    1. sigmoid score
    2. confidence threshold
    3. descending score ordering
    4. optional NMS
    5. top-k selection

    NMS is executed independently for each text query because this function
    receives predictions from only one text query at a time.
    """
    boxes = sanitize_xyxy_boxes(
        boxes.float().reshape(-1, 4)
    )
    scores = score_logits.float().reshape(-1).sigmoid()

    if int(boxes.shape[0]) != int(scores.shape[0]):
        raise ValueError(
            "Prediction box/score count mismatch: "
            f"{boxes.shape[0]} != {scores.shape[0]}"
        )

    valid_indices = torch.nonzero(
        scores >= float(confidence_threshold),
        as_tuple=False,
    ).flatten()

    num_after_nms = 0

    if valid_indices.numel() > 0:
        candidate_boxes = boxes.index_select(
            0,
            valid_indices,
        )
        candidate_scores = scores.index_select(
            0,
            valid_indices,
        )
        candidate_indices = valid_indices

        # Sort candidates from highest to lowest confidence.
        order = torch.argsort(
            candidate_scores,
            descending=True,
            stable=True,
        )

        candidate_boxes = candidate_boxes.index_select(
            0,
            order,
        )
        candidate_scores = candidate_scores.index_select(
            0,
            order,
        )
        candidate_indices = candidate_indices.index_select(
            0,
            order,
        )

        # Remove duplicate boxes for this text query.
        if bool(use_nms) and int(candidate_boxes.shape[0]) > 1:
            keep = nms(
                boxes=candidate_boxes,
                scores=candidate_scores,
                iou_threshold=float(nms_iou_threshold),
            )

            candidate_boxes = candidate_boxes.index_select(
                0,
                keep,
            )
            candidate_scores = candidate_scores.index_select(
                0,
                keep,
            )
            candidate_indices = candidate_indices.index_select(
                0,
                keep,
            )

        num_after_nms = int(candidate_scores.numel())

        keep_count = min(
            int(top_k),
            int(candidate_scores.numel()),
        )

        selected_boxes_norm = candidate_boxes[:keep_count]
        selected_scores = candidate_scores[:keep_count]
        selected_indices = candidate_indices[:keep_count]

    else:
        selected_indices = torch.empty(
            (0,),
            device=scores.device,
            dtype=torch.long,
        )
        selected_boxes_norm = boxes.new_empty((0, 4))
        selected_scores = scores.new_empty((0,))

    selected_boxes_pixel = normalized_xyxy_to_pixels(
        selected_boxes_norm,
        width=width,
        height=height,
    )

    return {
        "boxes_norm": selected_boxes_norm.detach().cpu(),
        "boxes_pixel": selected_boxes_pixel.detach().cpu(),
        "scores": selected_scores.detach().cpu(),
        "indices": selected_indices.detach().cpu(),
        "num_raw_predictions": int(scores.numel()),
        "num_above_confidence": int(valid_indices.numel()),
        "num_after_nms": int(num_after_nms),
        "num_selected": int(selected_scores.numel()),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def get_chinese_font(
    font_size: int = 28,
) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansTC-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]

    for font_path in candidates:
        if Path(font_path).is_file():
            return ImageFont.truetype(
                font_path,
                int(font_size),
            )

    return ImageFont.load_default()


def measure_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> Tuple[int, int]:
    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font,
    )

    return (
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    )


def fit_single_line_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    preferred_font_size: int,
    minimum_font_size: int,
    max_width: int,
) -> Tuple[str, ImageFont.ImageFont]:
    preferred_font_size = max(
        1,
        int(preferred_font_size),
    )
    minimum_font_size = max(
        1,
        min(
            int(minimum_font_size),
            preferred_font_size,
        ),
    )
    max_width = max(
        1,
        int(max_width),
    )

    for size in range(
        preferred_font_size,
        minimum_font_size - 1,
        -2,
    ):
        font = get_chinese_font(size)

        if measure_text(
            draw,
            text,
            font,
        )[0] <= max_width:
            return text, font

    font = get_chinese_font(
        minimum_font_size
    )
    ellipsis = "…"
    shortened = text

    while shortened:
        candidate = shortened + ellipsis

        if measure_text(
            draw,
            candidate,
            font,
        )[0] <= max_width:
            return candidate, font

        shortened = shortened[:-1]

    return ellipsis, font


def draw_chinese_text(
    image_bgr: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int = 28,
    color_bgr: Tuple[int, int, int] = (0, 255, 0),
    max_width: Optional[int] = None,
) -> np.ndarray:
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    pil_image = Image.fromarray(
        image_rgb
    )
    draw = ImageDraw.Draw(
        pil_image
    )

    if max_width is None:
        fitted_text = text
        font = get_chinese_font(
            font_size
        )
    else:
        fitted_text, font = fit_single_line_text(
            draw=draw,
            text=text,
            preferred_font_size=font_size,
            minimum_font_size=14,
            max_width=max_width,
        )

    color_rgb = (
        int(color_bgr[2]),
        int(color_bgr[1]),
        int(color_bgr[0]),
    )

    draw.text(
        position,
        fitted_text,
        font=font,
        fill=color_rgb,
    )

    return cv2.cvtColor(
        np.asarray(pil_image),
        cv2.COLOR_RGB2BGR,
    )


def draw_query_panel(
    image_bgr: np.ndarray,
    text_query: str,
) -> np.ndarray:
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    pil_image = Image.fromarray(
        image_rgb
    )
    draw = ImageDraw.Draw(
        pil_image
    )

    label = f"查詢文字：{text_query}"
    image_width, image_height = pil_image.size

    margin = max(
        6,
        min(
            24,
            image_width // 20,
        ),
    )
    padding = max(
        6,
        min(
            20,
            image_width // 30,
        ),
    )

    panel_width = max(
        1,
        image_width - margin * 2,
    )
    text_width = max(
        1,
        panel_width - padding * 2,
    )

    fitted_text, font = fit_single_line_text(
        draw=draw,
        text=label,
        preferred_font_size=max(
            20,
            int(image_width * 0.035),
        ),
        minimum_font_size=14,
        max_width=text_width,
    )

    _, text_height = measure_text(
        draw,
        fitted_text,
        font,
    )

    panel_height = text_height + padding * 2

    x1 = margin
    y1 = max(
        margin,
        image_height - panel_height - margin,
    )
    x2 = image_width - margin
    y2 = min(
        image_height - margin,
        y1 + panel_height,
    )

    draw.rectangle(
        [x1, y1, x2, y2],
        fill=(0, 0, 0),
    )

    draw.text(
        (
            x1 + padding,
            y1 + padding,
        ),
        fitted_text,
        font=font,
        fill=(255, 255, 255),
    )

    return cv2.cvtColor(
        np.asarray(pil_image),
        cv2.COLOR_RGB2BGR,
    )


def draw_predictions(
    image_bgr: np.ndarray,
    boxes_pixel: np.ndarray,
    scores: np.ndarray,
    query: str,
) -> np.ndarray:
    rendered = image_bgr.copy()

    for box, score in zip(
        boxes_pixel,
        scores,
    ):
        x1, y1, x2, y2 = (
            np.asarray(box)
            .astype(np.int32)
            .tolist()
        )

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(
            rendered,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        rendered = draw_chinese_text(
            image_bgr=rendered,
            text=(
                f"{query}: "
                f"{float(score):.3f}"
            ),
            position=(
                x1,
                max(0, y1 - 32),
            ),
            font_size=26,
            color_bgr=(0, 255, 0),
            max_width=max(
                40,
                rendered.shape[1] - x1 - 8,
            ),
        )

    return draw_query_panel(
        image_bgr=rendered,
        text_query=query,
    )


# ---------------------------------------------------------------------------
# Inference core
# ---------------------------------------------------------------------------


@torch.inference_mode()
def predict_image(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    original_bgr: np.ndarray,
    queries: Sequence[str],
    device: torch.device,
    confidence_threshold: float,
    top_k: int,
    use_nms: bool = True,
    nms_iou_threshold: float = 0.6,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    image_tensor = image_tensor.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    query_count = len(queries)

    image_indices = torch.zeros(
        query_count,
        dtype=torch.long,
        device=device,
    )

    # One source image is encoded once, then expanded to every text query.
    outputs = model(
        img=image_tensor,
        texts=list(queries),
        image_indices=image_indices,
        return_aux=False,
    )

    if outputs.get(
        "aux_computed",
        False,
    ):
        raise RuntimeError(
            "Auxiliary branch was unexpectedly computed during inference."
        )

    pred_boxes = outputs["bbox"]
    pred_score_logits = outputs["score_logit"]

    if int(pred_boxes.shape[0]) != query_count:
        raise RuntimeError(
            "Prediction/query batch mismatch: "
            f"{pred_boxes.shape[0]} != {query_count}"
        )

    height, width = original_bgr.shape[:2]

    results: List[Dict[str, Any]] = []
    rendered_images: List[np.ndarray] = []

    for index, query in enumerate(queries):
        # NMS is executed independently for each query.
        selected = select_main_predictions(
            boxes=pred_boxes[index],
            score_logits=pred_score_logits[index],
            confidence_threshold=confidence_threshold,
            top_k=top_k,
            width=width,
            height=height,
            use_nms=use_nms,
            nms_iou_threshold=nms_iou_threshold,
        )

        boxes_norm_np = selected[
            "boxes_norm"
        ].numpy()

        boxes_pixel_np = selected[
            "boxes_pixel"
        ].numpy()

        scores_np = selected[
            "scores"
        ].numpy()

        result = {
            "query": query,
            "boxes_norm": boxes_norm_np.tolist(),
            "boxes_pixel": boxes_pixel_np.tolist(),
            "scores": scores_np.tolist(),
            "indices": selected["indices"].tolist(),
            "num_raw_predictions": selected[
                "num_raw_predictions"
            ],
            "num_above_confidence": selected[
                "num_above_confidence"
            ],
            "num_after_nms": selected[
                "num_after_nms"
            ],
            "num_selected": selected[
                "num_selected"
            ],
        }

        results.append(result)

        rendered_images.append(
            draw_predictions(
                image_bgr=original_bgr,
                boxes_pixel=boxes_pixel_np,
                scores=scores_np,
                query=query,
            )
        )

    return results, rendered_images


# ---------------------------------------------------------------------------
# YOLO-style interface
# ---------------------------------------------------------------------------


class LightDet(ValidateLightDet):
    """
    Adds YOLO-style predict() to the same LightDet object used by train() and
    val().

    Example:
        model = LightDet(model="/path/to/model.yaml")

        results = model.predict(
            weights="/path/to/best.pt",
            source="/path/to/image.jpg",
            text=["ship", "car"],
            imgsz=512,
            device=0,
            conf=0.001,
            top_k=100,
            use_nms=True,
            nms_iou_threshold=0.6,
            project="runs/predict",
            name="exp",
        )
    """

    def predict(
        self,
        weights: str,
        source: str | Path,
        text: str | Sequence[str],
        imgsz: int = 512,
        device: Optional[Any] = None,
        conf: float = 0.001,
        top_k: int = 100,
        use_nms: bool = True,
        nms_iou_threshold: float = 0.6,
        project: str = "runs/predict",
        name: str = "exp",
        prefer_ema: bool = True,
        save: bool = True,
        save_json: bool = True,
    ) -> Dict[str, Any]:
        """
        Run Main One-to-One inference on one image and one or more text queries.

        Frequently changed runtime values remain in predict(). Model structure
        and BERT cache paths are read from model.yaml.
        """
        start_time = time.perf_counter()

        if not 0.0 <= float(conf) <= 1.0:
            raise ValueError(
                f"conf must be within [0, 1], got {conf}"
            )

        if int(top_k) <= 0:
            raise ValueError(
                f"top_k must be > 0, got {top_k}"
            )

        if int(imgsz) <= 0:
            raise ValueError(
                f"imgsz must be > 0, got {imgsz}"
            )

        if not 0.0 <= float(nms_iou_threshold) <= 1.0:
            raise ValueError(
                "nms_iou_threshold must be within [0, 1], "
                f"got {nms_iou_threshold}"
            )

        queries = normalize_queries(text)

        source_path = (
            Path(source)
            .expanduser()
            .resolve()
        )

        checkpoint_path = (
            Path(weights)
            .expanduser()
            .resolve()
        )

        output_dir = (
            Path(project).expanduser()
            / str(name)
        ).resolve()

        resolved_device = resolve_inference_device(
            device
        )

        model_cfg = deepcopy_cfg(
            self.model_cfg
        )
        model_cfg["model"][
            "auxiliary_in_eval"
        ] = False

        inference_model = build_inference_model(
            model_cfg
        )

        checkpoint_info = load_checkpoint_for_inference(
            model=inference_model,
            checkpoint_path=checkpoint_path,
            prefer_ema=bool(prefer_ema),
        )

        inference_model = inference_model.to(
            resolved_device
        )
        inference_model.eval()

        (
            image_tensor,
            original_bgr,
            width,
            height,
        ) = preprocess_image(
            source=source_path,
            image_size=int(imgsz),
        )

        print("\n[LightDet] Prediction config")
        print(f"  source       : {source_path}")
        print(f"  weights      : {checkpoint_info['path']}")
        print(f"  source state : {checkpoint_info['source']}")
        print(f"  epoch        : {checkpoint_info['epoch']}")
        print(f"  device       : {resolved_device}")
        print(f"  image size   : {imgsz}")
        print(f"  original     : {width}x{height}")
        print(f"  queries      : {len(queries)}")
        print(f"  confidence   : {float(conf):.6f}")
        print(f"  top-k        : {int(top_k)}")
        print(f"  use NMS      : {bool(use_nms)}")
        print(
            f"  NMS IoU      : "
            f"{float(nms_iou_threshold):.3f}"
        )
        print("  auxiliary    : disabled")

        results, rendered_images = predict_image(
            model=inference_model,
            image_tensor=image_tensor,
            original_bgr=original_bgr,
            queries=queries,
            device=resolved_device,
            confidence_threshold=float(conf),
            top_k=int(top_k),
            use_nms=bool(use_nms),
            nms_iou_threshold=float(
                nms_iou_threshold
            ),
        )

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        rendered_path: Optional[Path] = None
        json_path: Optional[Path] = None

        if save or save_json:
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        if save:
            if len(rendered_images) == 1:
                combined = rendered_images[0]
            else:
                combined = cv2.hconcat(
                    rendered_images
                )

            rendered_path = (
                output_dir
                / "prediction.jpg"
            )

            if not cv2.imwrite(
                str(rendered_path),
                combined,
            ):
                raise RuntimeError(
                    "Failed to save rendered image: "
                    f"{rendered_path}"
                )

        output: Dict[str, Any] = {
            "source": str(source_path),
            "weights": checkpoint_info["path"],
            "weight_source": checkpoint_info["source"],
            "checkpoint_epoch": checkpoint_info["epoch"],
            "stored_best_metric_name": checkpoint_info[
                "best_metric_name"
            ],
            "stored_best_metric": checkpoint_info[
                "best_metric"
            ],
            "device": str(resolved_device),
            "image_size": int(imgsz),
            "original_size": {
                "width": width,
                "height": height,
            },
            "confidence_threshold": float(conf),
            "top_k": int(top_k),
            "use_nms": bool(use_nms),
            "nms_iou_threshold": float(
                nms_iou_threshold
            ),
            "auxiliary_computed": False,
            "results": results,
            "rendered_path": (
                str(rendered_path)
                if rendered_path is not None
                else None
            ),
            "elapsed_seconds": elapsed_seconds,
        }

        if save_json:
            json_path = (
                output_dir
                / "predictions.json"
            )

            temporary_path = Path(
                str(json_path) + ".tmp"
            )

            with open(
                temporary_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    output,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            os.replace(
                temporary_path,
                json_path,
            )

            output["json_path"] = str(
                json_path
            )
        else:
            output["json_path"] = None

        print("\n[Prediction result]")

        for result in results:
            print(
                f"  query={result['query']!r}, "
                f"raw={result['num_raw_predictions']}, "
                f"above_conf={result['num_above_confidence']}, "
                f"after_nms={result['num_after_nms']}, "
                f"selected={result['num_selected']}"
            )

            for rank, (box, score) in enumerate(
                zip(
                    result["boxes_pixel"],
                    result["scores"],
                ),
                start=1,
            ):
                rounded_box = [
                    int(round(value))
                    for value in box
                ]

                print(
                    f"    rank={rank:02d}, "
                    f"score={float(score):.6f}, "
                    f"box={rounded_box}"
                )

        print(
            f"  elapsed      : "
            f"{elapsed_seconds:.3f}s"
        )

        if rendered_path is not None:
            print(
                f"  saved image  : "
                f"{rendered_path}"
            )

        if json_path is not None:
            print(
                f"  saved JSON   : "
                f"{json_path}"
            )

        return output


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

    model.predict(
        weights=(
            "/home/soic/Desktop/LightDet/units/model/runs/train/lightdet_HDETR_transformer_layer_decoupled_v3/best_map50.pt"
        ),
        source=(
            "/home/soic/Desktop/datasetPreTest15000/dataset/datasetPreTest15000_mixed/images/train/2023_08_08_15_26_49_815814_1.jpg"
        ),
        text=[
            "一輛計程車",
            "一艘紅色的船",
        ],
        imgsz=1024,
        device=0,
        conf=0.4,
        top_k=20,

        # NMS 設定
        use_nms=True,
        nms_iou_threshold=0.3,

        project="runs/predict",
        name="lightdet_hybrid",

        
        prefer_ema=True,

        save=True,
        save_json=True,
    )


if __name__ == "__main__":
    main()