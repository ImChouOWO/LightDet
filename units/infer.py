from __future__ import annotations

"""
LightDet ODVG-style phrase grounding inference.

使用方式：
    from units.infer import LightDet

    model = LightDet(
        model="/home/soic/Desktop/LightDet/units/model/cards/config/model.yaml"
    )

    results = model.predict(
        weights="/path/to/best_map50_95.pt",
        source="/path/to/image.jpg",
        caption="畫面中包含一艘紅色的船與一艘白色的船",
        phrases=[
            "紅色的船",
            "白色的船",
        ],
        imgsz=1024,
        device=0,
        conf=0.30,
        quality_thr=0.50,
        alignment_thr=0.45,
        top_k=20,
        use_nms=True,
        project="runs/predict",
        name="odvg_exp",
    )

ODVG inference contract:
    - 一張影像只執行一次 Vision Backbone；
    - 模型輸入為完整 caption；
    - 每個 phrase 由 caption 內的 character span 映射至 BERT token；
    - 每個 Object Query 同時輸出：
        quality_score
        phrase_alignment_score
        final_score
    - final_score 預設為 sqrt(quality * phrase_alignment)；
    - Auxiliary branch 在推論時停用；
    - NMS 可選，預設關閉以維持 DETR-style 推論。
"""

import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

for path in (PROJECT_ROOT, CURRENT_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


from units.validate import LightDet as ValidateLightDet  # noqa: E402
from units.tool.card import VisionTextModel  # noqa: E402
from units.model.tool.runtime import (  # noqa: E402
    score_queries_for_char_spans,
)


def deepcopy_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(cfg)


def normalize_device(device: Optional[Any]) -> str:
    if device is None:
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if isinstance(device, torch.device):
        return str(device)
    if isinstance(device, int):
        return f"cuda:{device}"

    value = str(device).strip().lower()
    if not value:
        return "cuda:0" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if value.isdigit():
        return f"cuda:{value}"
    if value in {"cuda", "gpu"}:
        return "cuda:0"
    if value.startswith("cuda:") or value in {"cpu", "mps"}:
        return value

    raise ValueError(
        f"Unsupported device value: {device!r}"
    )



# Checkpoint helpers



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



# Model / device



def build_inference_model(
    model_cfg_all: Dict[str, Any],
) -> VisionTextModel:
    cfg = model_cfg_all["model"]

    image_grid_size = int(
        cfg.get("image_grid_size", 10)
    )

    return VisionTextModel(
        img_in_channels=int(
            cfg.get("img_in_channels", 1024)
        ),
        cnn_layer=int(
            cfg.get("cnn_layers", 3)
        ),
        hidden_dim=int(
            cfg.get("hidden_dim", 512)
        ),
        target_size=(
            image_grid_size,
            image_grid_size,
        ),
        text_max_length=int(
            cfg.get("text_max_length", 96)
        ),
        fusion_token_num=int(
            cfg.get("fusion_token_num", 16)
        ),
        num_object_queries=int(
            cfg.get("num_object_queries", 100)
        ),
        num_heads=int(
            cfg.get("num_heads", 8)
        ),
        num_layers=int(
            cfg.get("num_layers", 3)
        ),
        mlp_ratio=float(
            cfg.get("mlp_ratio", 3.5)
        ),
        dropout=float(
            cfg.get("dropout", 0.1)
        ),
        freeze_bert=bool(
            cfg.get("freeze_bert", True)
        ),
        precomputed_bert_path=cfg.get(
            "precomputed_bert_path"
        ),
        use_auxiliary_head=bool(
            cfg.get("use_auxiliary_head", True)
        ),
        auxiliary_in_eval=False,
        initialize_aux_from_main=bool(
            cfg.get("initialize_aux_from_main", True)
        ),
        query_group_init_std=float(
            cfg.get("query_group_init_std", 0.02)
        ),
    )


def resolve_inference_device(
    device: Optional[Any],
) -> torch.device:
    normalized = normalize_device(device)
    resolved = torch.device(normalized)

    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            print("[Warning] CUDA unavailable; falling back to CPU.")
            return torch.device("cpu")

        index = (
            0
            if resolved.index is None
            else int(resolved.index)
        )

        if index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested cuda:{index}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available."
            )

        return torch.device(f"cuda:{index}")

    if resolved.type == "mps":
        if not torch.backends.mps.is_available():
            print("[Warning] MPS unavailable; falling back to CPU.")
            return torch.device("cpu")

    return resolved



# Text / input helpers



def normalize_caption(
    caption: str,
) -> str:
    normalized = str(caption).strip()

    if not normalized:
        raise ValueError(
            "caption must be a non-empty string."
        )

    return normalized


def normalize_phrases(
    phrases: str | Sequence[str],
) -> List[str]:
    if isinstance(phrases, str):
        candidates = [phrases]
    else:
        candidates = list(phrases)

    normalized: List[str] = []
    seen = set()

    for raw_phrase in candidates:
        phrase = str(raw_phrase).strip()

        if not phrase or phrase in seen:
            continue

        normalized.append(phrase)
        seen.add(phrase)

    if not normalized:
        raise ValueError(
            "phrases must contain at least one non-empty phrase."
        )

    return normalized


def find_phrase_char_spans(
    caption: str,
    phrase: str,
    include_all_occurrences: bool = True,
) -> List[List[int]]:
    """
    Locate phrase occurrences in caption.

    The returned intervals follow Python slicing semantics:
        caption[start:end] == phrase
    """
    if not phrase:
        raise ValueError("phrase must not be empty")

    spans: List[List[int]] = []
    start = 0

    while True:
        position = caption.find(
            phrase,
            start,
        )

        if position < 0:
            break

        end = position + len(phrase)
        spans.append([
            int(position),
            int(end),
        ])

        if not include_all_occurrences:
            break

        start = position + max(
            len(phrase),
            1,
        )

    if not spans:
        raise ValueError(
            f"Phrase {phrase!r} is not contained in caption {caption!r}."
        )

    return spans


def preprocess_image(
    source: str | Path,
    image_size: int,
) -> Tuple[torch.Tensor, np.ndarray, int, int]:
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input image not found: {source}")

    try:
        original_pil = Image.open(source).convert("RGB")
    except Exception as error:
        raise RuntimeError(f"Unable to read image: {source}") from error

    original_width, original_height = original_pil.size
    original_bgr = cv2.cvtColor(
        np.asarray(original_pil),
        cv2.COLOR_RGB2BGR,
    )

    resized_pil = original_pil.resize(
        (int(image_size), int(image_size)),
        Image.Resampling.BILINEAR,
    )
    image_array = np.asarray(resized_pil, dtype=np.float32) / 255.0
    image_tensor = (
        torch.from_numpy(image_array)
        .permute(2, 0, 1)
        .contiguous()
        .unsqueeze(0)
    )

    return (
        image_tensor,
        original_bgr,
        int(original_width),
        int(original_height),
    )



# Box / score helpers



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
    boxes = sanitize_xyxy_boxes(
        boxes
    )

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


def select_phrase_predictions(
    boxes: torch.Tensor,
    final_scores: torch.Tensor,
    quality_scores: torch.Tensor,
    alignment_scores: torch.Tensor,
    confidence_threshold: float,
    quality_threshold: float,
    alignment_threshold: float,
    top_k: int,
    width: int,
    height: int,
    use_nms: bool = False,
    nms_iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Select phrase-specific predictions.

    Selection order:
        1. final score threshold
        2. descending final score
        3. optional NMS
        4. top-k
    """
    boxes = sanitize_xyxy_boxes(
        boxes.float().reshape(-1, 4)
    )
    final_scores = (
        final_scores
        .float()
        .reshape(-1)
        .clamp(0.0, 1.0)
    )
    quality_scores = (
        quality_scores
        .float()
        .reshape(-1)
        .clamp(0.0, 1.0)
    )
    alignment_scores = (
        alignment_scores
        .float()
        .reshape(-1)
        .clamp(0.0, 1.0)
    )

    count = int(boxes.shape[0])

    if not (
        final_scores.numel()
        == quality_scores.numel()
        == alignment_scores.numel()
        == count
    ):
        raise ValueError(
            "Prediction box/score count mismatch: "
            f"boxes={count}, "
            f"final={final_scores.numel()}, "
            f"quality={quality_scores.numel()}, "
            f"alignment={alignment_scores.numel()}"
        )

    final_mask = final_scores >= float(confidence_threshold)
    quality_mask = quality_scores >= float(quality_threshold)
    alignment_mask = alignment_scores >= float(alignment_threshold)
    valid_mask = final_mask & quality_mask & alignment_mask

    valid_indices = torch.nonzero(
        valid_mask,
        as_tuple=False,
    ).flatten()

    num_after_nms = 0

    if valid_indices.numel() > 0:
        candidate_boxes = boxes.index_select(
            0,
            valid_indices,
        )
        candidate_final = final_scores.index_select(
            0,
            valid_indices,
        )
        candidate_quality = quality_scores.index_select(
            0,
            valid_indices,
        )
        candidate_alignment = alignment_scores.index_select(
            0,
            valid_indices,
        )
        candidate_indices = valid_indices

        order = torch.argsort(
            candidate_final,
            descending=True,
            stable=True,
        )

        candidate_boxes = candidate_boxes.index_select(
            0,
            order,
        )
        candidate_final = candidate_final.index_select(
            0,
            order,
        )
        candidate_quality = candidate_quality.index_select(
            0,
            order,
        )
        candidate_alignment = candidate_alignment.index_select(
            0,
            order,
        )
        candidate_indices = candidate_indices.index_select(
            0,
            order,
        )

        if (
            bool(use_nms)
            and int(candidate_boxes.shape[0]) > 1
        ):
            keep = nms(
                boxes=candidate_boxes,
                scores=candidate_final,
                iou_threshold=float(
                    nms_iou_threshold
                ),
            )

            candidate_boxes = candidate_boxes.index_select(
                0,
                keep,
            )
            candidate_final = candidate_final.index_select(
                0,
                keep,
            )
            candidate_quality = candidate_quality.index_select(
                0,
                keep,
            )
            candidate_alignment = (
                candidate_alignment.index_select(
                    0,
                    keep,
                )
            )
            candidate_indices = candidate_indices.index_select(
                0,
                keep,
            )

        num_after_nms = int(
            candidate_final.numel()
        )

        keep_count = min(
            int(top_k),
            int(candidate_final.numel()),
        )

        selected_boxes_norm = candidate_boxes[
            :keep_count
        ]
        selected_final = candidate_final[
            :keep_count
        ]
        selected_quality = candidate_quality[
            :keep_count
        ]
        selected_alignment = candidate_alignment[
            :keep_count
        ]
        selected_indices = candidate_indices[
            :keep_count
        ]

    else:
        selected_indices = torch.empty(
            (0,),
            device=final_scores.device,
            dtype=torch.long,
        )
        selected_boxes_norm = boxes.new_empty(
            (0, 4)
        )
        selected_final = final_scores.new_empty(
            (0,)
        )
        selected_quality = quality_scores.new_empty(
            (0,)
        )
        selected_alignment = alignment_scores.new_empty(
            (0,)
        )

    selected_boxes_pixel = normalized_xyxy_to_pixels(
        selected_boxes_norm,
        width=width,
        height=height,
    )

    return {
        "boxes_norm": (
            selected_boxes_norm
            .detach()
            .cpu()
        ),
        "boxes_pixel": (
            selected_boxes_pixel
            .detach()
            .cpu()
        ),
        "scores": (
            selected_final
            .detach()
            .cpu()
        ),
        "quality_scores": (
            selected_quality
            .detach()
            .cpu()
        ),
        "alignment_scores": (
            selected_alignment
            .detach()
            .cpu()
        ),
        "indices": (
            selected_indices
            .detach()
            .cpu()
        ),
        "num_raw_predictions": count,
        "num_above_final": int(final_mask.sum().item()),
        "num_above_quality": int(quality_mask.sum().item()),
        "num_above_alignment": int(alignment_mask.sum().item()),
        "num_after_semantic_gate": int(valid_indices.numel()),
        "num_above_confidence": int(valid_indices.numel()),
        "num_after_nms": int(
            num_after_nms
        ),
        "num_selected": int(
            selected_final.numel()
        ),
    }



# Rendering



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
        font = get_chinese_font(
            size
        )

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


def draw_phrase_panel(
    image_bgr: np.ndarray,
    phrase: str,
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

    label = f"查詢片語：{phrase}"
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
    final_scores: np.ndarray,
    quality_scores: np.ndarray,
    alignment_scores: np.ndarray,
    phrase: str,
) -> np.ndarray:
    rendered = image_bgr.copy()

    for (
        box,
        final_score,
        quality_score,
        alignment_score,
    ) in zip(
        boxes_pixel,
        final_scores,
        quality_scores,
        alignment_scores,
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

        label = (
            f"{phrase} "
            f"F:{float(final_score):.3f} "
            f"Q:{float(quality_score):.3f} "
            f"A:{float(alignment_score):.3f}"
        )

        rendered = draw_chinese_text(
            image_bgr=rendered,
            text=label,
            position=(
                x1,
                max(0, y1 - 32),
            ),
            font_size=24,
            color_bgr=(0, 255, 0),
            max_width=max(
                40,
                rendered.shape[1] - x1 - 8,
            ),
        )

    return draw_phrase_panel(
        image_bgr=rendered,
        phrase=phrase,
    )



# ODVG inference core



def _resolve_main_output(
    outputs: Dict[str, Any],
    keys: Sequence[str],
    name: str,
) -> torch.Tensor:
    for key in keys:
        value = outputs.get(key)

        if torch.is_tensor(value):
            return value

    raise KeyError(
        f"Model output does not contain {name}. "
        f"Tried keys: {list(keys)}"
    )


@torch.inference_mode()
def predict_image(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    original_bgr: np.ndarray,
    caption: str,
    phrases: Sequence[str],
    device: torch.device,
    confidence_threshold: float,
    quality_threshold: float,
    alignment_threshold: float,
    top_k: int,
    use_nms: bool = False,
    nms_iou_threshold: float = 0.5,
    token_reduction: str = "mean",
    score_fusion: str = "geometric_mean",
    include_all_occurrences: bool = True,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    image_tensor = image_tensor.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    # A single image and complete caption are forwarded once.
    outputs = model(
        img=image_tensor,
        texts=[caption],
        image_indices=None,
        return_aux=False,
    )

    if outputs.get(
        "aux_computed",
        False,
    ):
        raise RuntimeError(
            "Auxiliary branch was unexpectedly computed during inference."
        )

    pred_boxes = _resolve_main_output(
        outputs,
        (
            "bbox",
            "main_bbox",
        ),
        "bbox",
    )
    quality_logit = _resolve_main_output(
        outputs,
        (
            "quality_logit",
            "main_quality_logit",
        ),
        "quality_logit",
    )
    token_alignment_logits = _resolve_main_output(
        outputs,
        (
            "token_alignment_logits",
            "main_token_alignment_logits",
            "text_alignment_logit",
        ),
        "token_alignment_logits",
    )
    token_offsets = _resolve_main_output(
        outputs,
        ("token_offsets",),
        "token_offsets",
    )
    alignment_text_mask = _resolve_main_output(
        outputs,
        (
            "alignment_text_mask",
            "text_mask",
        ),
        "alignment_text_mask",
    )

    if pred_boxes.ndim != 3 or pred_boxes.shape[0] != 1:
        raise RuntimeError(
            "ODVG inference expects bbox shape [1,Q,4], got "
            f"{tuple(pred_boxes.shape)}"
        )

    if (
        quality_logit.ndim != 3
        or quality_logit.shape[0] != 1
        or quality_logit.shape[-1] != 1
    ):
        raise RuntimeError(
            "ODVG inference expects quality_logit [1,Q,1], got "
            f"{tuple(quality_logit.shape)}"
        )

    if (
        token_alignment_logits.ndim != 3
        or token_alignment_logits.shape[0] != 1
    ):
        raise RuntimeError(
            "ODVG inference expects token_alignment_logits [1,Q,L], got "
            f"{tuple(token_alignment_logits.shape)}"
        )

    if token_offsets.ndim != 3 or token_offsets.shape[0] != 1:
        raise RuntimeError(
            "ODVG inference expects token_offsets [1,L,2], got "
            f"{tuple(token_offsets.shape)}"
        )

    if (
        alignment_text_mask.ndim != 2
        or alignment_text_mask.shape[0] != 1
    ):
        raise RuntimeError(
            "ODVG inference expects alignment_text_mask [1,L], got "
            f"{tuple(alignment_text_mask.shape)}"
        )

    boxes_row = pred_boxes[0]
    quality_row = quality_logit[0]
    token_logits_row = token_alignment_logits[0]
    offsets_row = token_offsets[0]
    valid_mask_row = alignment_text_mask[0]

    if boxes_row.shape[0] != quality_row.shape[0]:
        raise RuntimeError(
            "bbox/quality query count mismatch"
        )

    if boxes_row.shape[0] != token_logits_row.shape[0]:
        raise RuntimeError(
            "bbox/token alignment query count mismatch"
        )

    if token_logits_row.shape[-1] != offsets_row.shape[0]:
        raise RuntimeError(
            "token alignment/token offset length mismatch"
        )

    height, width = original_bgr.shape[:2]

    results: List[Dict[str, Any]] = []
    rendered_images: List[np.ndarray] = []

    for phrase in phrases:
        char_spans = find_phrase_char_spans(
            caption=caption,
            phrase=phrase,
            include_all_occurrences=(
                include_all_occurrences
            ),
        )

        phrase_scores = score_queries_for_char_spans(
            quality_logit=quality_row,
            token_alignment_logits=token_logits_row,
            token_offsets=offsets_row,
            char_spans=char_spans,
            valid_token_mask=valid_mask_row,
            token_reduction=token_reduction,
            score_fusion=score_fusion,
            strict=True,
        )

        selected = select_phrase_predictions(
            boxes=boxes_row,
            final_scores=phrase_scores[
                "final_score"
            ],
            quality_scores=phrase_scores[
                "quality_score"
            ],
            alignment_scores=phrase_scores[
                "phrase_alignment_score"
            ],
            confidence_threshold=confidence_threshold,
            quality_threshold=quality_threshold,
            alignment_threshold=alignment_threshold,
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
        final_scores_np = selected[
            "scores"
        ].numpy()
        quality_scores_np = selected[
            "quality_scores"
        ].numpy()
        alignment_scores_np = selected[
            "alignment_scores"
        ].numpy()

        positive_token_indices = torch.nonzero(
            phrase_scores["token_mask"],
            as_tuple=False,
        ).flatten().detach().cpu().tolist()

        result = {
            "phrase": phrase,
            "caption": caption,
            "char_spans": char_spans,
            "positive_token_indices": (
                positive_token_indices
            ),
            "boxes_norm": (
                boxes_norm_np.tolist()
            ),
            "boxes_pixel": (
                boxes_pixel_np.tolist()
            ),
            "scores": (
                final_scores_np.tolist()
            ),
            "quality_scores": (
                quality_scores_np.tolist()
            ),
            "alignment_scores": (
                alignment_scores_np.tolist()
            ),
            "indices": (
                selected["indices"].tolist()
            ),
            "num_raw_predictions": selected["num_raw_predictions"],
            "num_above_final": selected["num_above_final"],
            "num_above_quality": selected["num_above_quality"],
            "num_above_alignment": selected["num_above_alignment"],
            "num_after_semantic_gate": selected[
                "num_after_semantic_gate"
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

        results.append(
            result
        )

        rendered_images.append(
            draw_predictions(
                image_bgr=original_bgr,
                boxes_pixel=boxes_pixel_np,
                final_scores=final_scores_np,
                quality_scores=quality_scores_np,
                alignment_scores=alignment_scores_np,
                phrase=phrase,
            )
        )

    return results, rendered_images



# YOLO-style object interface



class LightDet(ValidateLightDet):
    """
    Adds ODVG-style phrase grounding predict() to the same LightDet object used
    by train() and val().

    Example:
        model = LightDet(
            model="/path/to/model.yaml"
        )

        results = model.predict(
            weights="/path/to/best.pt",
            source="/path/to/image.jpg",
            caption="畫面中包含紅色的船與白色的船",
            phrases=[
                "紅色的船",
                "白色的船",
            ],
            imgsz=1024,
            device=0,
            conf=0.05,
            top_k=20,
            use_nms=False,
            project="runs/predict",
            name="odvg_exp",
        )
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._inference_model: Optional[torch.nn.Module] = None
        self._inference_checkpoint_info: Optional[Dict[str, Any]] = None
        self._loaded_weights_path: Optional[str] = None
        self._loaded_device: Optional[str] = None
        self._loaded_prefer_ema: Optional[bool] = None

    def clear_inference_cache(self) -> None:
        self._inference_model = None
        self._inference_checkpoint_info = None
        self._loaded_weights_path = None
        self._loaded_device = None
        self._loaded_prefer_ema = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_inference_model(
        self,
        weights: str | Path,
        device: torch.device,
        prefer_ema: bool,
    ) -> Tuple[torch.nn.Module, Dict[str, Any], bool]:
        checkpoint_path = Path(weights).expanduser().resolve()

        cache_hit = (
            self._inference_model is not None
            and self._inference_checkpoint_info is not None
            and self._loaded_weights_path == str(checkpoint_path)
            and self._loaded_device == str(device)
            and self._loaded_prefer_ema == bool(prefer_ema)
        )

        if cache_hit:
            return (
                self._inference_model,
                self._inference_checkpoint_info,
                True,
            )

        model_cfg = deepcopy_cfg(self.model_cfg)
        model_cfg["model"]["auxiliary_in_eval"] = False

        load_start = time.perf_counter()
        print("\n[LightDet] Loading inference model")
        print(f"  weights : {checkpoint_path}")
        print(f"  device  : {device}")

        inference_model = build_inference_model(model_cfg)
        checkpoint_info = load_checkpoint_for_inference(
            model=inference_model,
            checkpoint_path=checkpoint_path,
            prefer_ema=bool(prefer_ema),
        )
        inference_model = inference_model.to(device)
        inference_model.eval()

        self._inference_model = inference_model
        self._inference_checkpoint_info = checkpoint_info
        self._loaded_weights_path = str(checkpoint_path)
        self._loaded_device = str(device)
        self._loaded_prefer_ema = bool(prefer_ema)

        print(f"  load time: {time.perf_counter() - load_start:.3f}s")
        return inference_model, checkpoint_info, False

    def predict(
        self,
        weights: str,
        source: str | Path,
        caption: str,
        phrases: str | Sequence[str],
        imgsz: int = 1024,
        device: Optional[Any] = None,
        conf: float = 0.05,
        quality_thr: float = 0.50,
        alignment_thr: float = 0.45,
        top_k: int = 20,
        use_nms: bool = False,
        nms_iou_threshold: float = 0.5,
        token_reduction: str = "mean",
        score_fusion: str = "geometric_mean",
        include_all_occurrences: bool = True,
        project: str = "runs/predict",
        name: str = "odvg_exp",
        prefer_ema: bool = True,
        save: bool = True,
        save_json: bool = True,
    ) -> Dict[str, Any]:
        """
        Run Main One-to-One ODVG inference for one image.

        The model receives one complete caption. Every requested phrase is
        scored from the corresponding caption token span without running the
        backbone or Transformer again.
        """
        start_time = time.perf_counter()

        if not 0.0 <= float(conf) <= 1.0:
            raise ValueError(
                f"conf must be within [0, 1], got {conf}"
            )

        if not 0.0 <= float(quality_thr) <= 1.0:
            raise ValueError(
                f"quality_thr must be within [0, 1], got {quality_thr}"
            )

        if not 0.0 <= float(alignment_thr) <= 1.0:
            raise ValueError(
                f"alignment_thr must be within [0, 1], got {alignment_thr}"
            )

        if int(top_k) <= 0:
            raise ValueError(
                f"top_k must be > 0, got {top_k}"
            )

        if int(imgsz) <= 0:
            raise ValueError(
                f"imgsz must be > 0, got {imgsz}"
            )

        if not 0.0 <= float(
            nms_iou_threshold
        ) <= 1.0:
            raise ValueError(
                "nms_iou_threshold must be within [0, 1], "
                f"got {nms_iou_threshold}"
            )

        normalized_caption = normalize_caption(
            caption
        )
        normalized_phrases = normalize_phrases(
            phrases
        )

        # Validate every phrase before loading the model.
        for phrase in normalized_phrases:
            find_phrase_char_spans(
                caption=normalized_caption,
                phrase=phrase,
                include_all_occurrences=(
                    include_all_occurrences
                ),
            )

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

        (
            inference_model,
            checkpoint_info,
            cache_hit,
        ) = self._get_inference_model(
            weights=checkpoint_path,
            device=resolved_device,
            prefer_ema=bool(prefer_ema),
        )

        (
            image_tensor,
            original_bgr,
            width,
            height,
        ) = preprocess_image(
            source=source_path,
            image_size=int(imgsz),
        )

        print("\n[LightDet ODVG] Prediction config")
        print(f"  source          : {source_path}")
        print(f"  weights         : {checkpoint_info['path']}")
        print(f"  source state    : {checkpoint_info['source']}")
        print(f"  epoch           : {checkpoint_info['epoch']}")
        print(f"  device          : {resolved_device}")
        print(f"  image size      : {imgsz}")
        print(f"  original        : {width}x{height}")
        print(f"  caption         : {normalized_caption}")
        print(f"  phrases         : {len(normalized_phrases)}")
        print(f"  confidence      : {float(conf):.6f}")
        print(f"  quality thr     : {float(quality_thr):.6f}")
        print(f"  alignment thr   : {float(alignment_thr):.6f}")
        print(f"  model cache     : {'hit' if cache_hit else 'miss'}")
        print(f"  top-k           : {int(top_k)}")
        print(f"  token reduction : {token_reduction}")
        print(f"  score fusion    : {score_fusion}")
        print(f"  use NMS         : {bool(use_nms)}")
        print(
            f"  NMS IoU         : "
            f"{float(nms_iou_threshold):.3f}"
        )
        print("  auxiliary       : disabled")

        results, rendered_images = predict_image(
            model=inference_model,
            image_tensor=image_tensor,
            original_bgr=original_bgr,
            caption=normalized_caption,
            phrases=normalized_phrases,
            device=resolved_device,
            confidence_threshold=float(conf),
            quality_threshold=float(quality_thr),
            alignment_threshold=float(alignment_thr),
            top_k=int(top_k),
            use_nms=bool(use_nms),
            nms_iou_threshold=float(
                nms_iou_threshold
            ),
            token_reduction=str(
                token_reduction
            ),
            score_fusion=str(
                score_fusion
            ),
            include_all_occurrences=bool(
                include_all_occurrences
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
            "caption": normalized_caption,
            "phrases": normalized_phrases,
            "confidence_threshold": float(conf),
            "quality_threshold": float(quality_thr),
            "alignment_threshold": float(alignment_thr),
            "model_cache_hit": bool(cache_hit),
            "top_k": int(top_k),
            "token_reduction": str(
                token_reduction
            ),
            "score_fusion": str(
                score_fusion
            ),
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
                f"  phrase={result['phrase']!r}, "
                f"spans={result['char_spans']}, "
                f"raw={result['num_raw_predictions']}, "
                f"final_ok={result['num_above_final']}, "
                f"quality_ok={result['num_above_quality']}, "
                f"alignment_ok={result['num_above_alignment']}, "
                f"gated={result['num_after_semantic_gate']}, "
                f"after_nms={result['num_after_nms']}, "
                f"selected={result['num_selected']}"
            )

            for rank, (
                box,
                final_score,
                quality_score,
                alignment_score,
            ) in enumerate(
                zip(
                    result["boxes_pixel"],
                    result["scores"],
                    result["quality_scores"],
                    result["alignment_scores"],
                ),
                start=1,
            ):
                rounded_box = [
                    int(round(value))
                    for value in box
                ]

                print(
                    f"    rank={rank:02d}, "
                    f"final={float(final_score):.6f}, "
                    f"quality={float(quality_score):.6f}, "
                    f"alignment={float(alignment_score):.6f}, "
                    f"box={rounded_box}"
                )

        print(
            f"  elapsed        : "
            f"{elapsed_seconds:.3f}s"
        )

        if rendered_path is not None:
            print(
                f"  saved image    : "
                f"{rendered_path}"
            )

        if json_path is not None:
            print(
                f"  saved JSON     : "
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
            "/home/soic/Desktop/LightDet/units/model/runs/train/lightdet_ODVG_token_alignment/best_map50_95.pt"
        ),
        source=(
            "/home/soic/Desktop/datasetPreTest15000/dataset/sys/rain/2021_11_14_16_43_48_01330.jpg"
        ),

        
        caption=(
            "紅色的船"
        ),

        
        phrases = [
            "紅色的船"
        ],
        imgsz=1024,
        device=0,
        conf=0.5,
        quality_thr=0.5,
        alignment_thr=0.5,
        top_k=20,

        use_nms=True,
        nms_iou_threshold=0.5,

       
        token_reduction="mean",
        score_fusion="geometric_mean",
        include_all_occurrences=True,

        project="runs/predict",
        name="lightdet_odvg",

        prefer_ema=False,
        save=True,
        save_json=True,
    )


if __name__ == "__main__":
    main()