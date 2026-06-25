#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.ops import nms


CURRENT_DIR = Path(__file__).resolve().parent
UNITS_DIR = CURRENT_DIR.parent

if str(UNITS_DIR) not in sys.path:
    sys.path.insert(0, str(UNITS_DIR))


def build_model():
    from units.tool.card import VisionTextModel

    return VisionTextModel(
        img_in_channels=1024,
        hidden_dim=384,
        target_size=(10, 10),
        text_max_length=32,
        fusion_token_num=16,
        num_layers=2,
        num_heads=8,
        mlp_ratio=3.5,
        dropout=0.1,
        freeze_bert=True,
    )


def load_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    try:
        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu",
        )

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value

    load_result = model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    if load_result.missing_keys:
        print("\n[WARNING] Missing keys:")
        for key in load_result.missing_keys:
            print(f"  - {key}")

    if load_result.unexpected_keys:
        print("\n[WARNING] Unexpected keys:")
        for key in load_result.unexpected_keys:
            print(f"  - {key}")

    model = model.to(device)
    model.eval()

    return model


def preprocess_image(
    image_path: str | Path,
    image_size: int = 512,
) -> Tuple[torch.Tensor, int, int]:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"找不到輸入影像: {image_path}")

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    transform = transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0)

    return image_tensor, orig_w, orig_h


def sanitize_xyxy_boxes(boxes: torch.Tensor) -> torch.Tensor:
    boxes = boxes.clamp(0.0, 1.0)

    x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
    x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3])

    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_norm_to_pixel(
    boxes: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    boxes = sanitize_xyxy_boxes(boxes)

    scale = boxes.new_tensor([
        width,
        height,
        width,
        height,
    ])

    boxes = boxes * scale

    boxes[:, 0].clamp_(0, max(width - 1, 0))
    boxes[:, 1].clamp_(0, max(height - 1, 0))
    boxes[:, 2].clamp_(0, max(width - 1, 0))
    boxes[:, 3].clamp_(0, max(height - 1, 0))

    return boxes


def get_chinese_font(font_size: int = 32) -> ImageFont.FreeTypeFont:
    font_candidates = [
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

    for font_path in font_candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, font_size)

    return ImageFont.load_default()


def measure_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
) -> Tuple[int, int]:
    text_bbox = draw.textbbox(
        (0, 0),
        text,
        font=font,
    )

    return (
        text_bbox[2] - text_bbox[0],
        text_bbox[3] - text_bbox[1],
    )


def fit_single_line_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    preferred_font_size: int,
    min_font_size: int,
    max_width: int,
) -> Tuple[str, ImageFont.FreeTypeFont]:
    preferred_font_size = max(
        1,
        int(preferred_font_size),
    )

    min_font_size = max(
        1,
        min(
            int(min_font_size),
            preferred_font_size,
        ),
    )

    max_width = max(
        1,
        int(max_width),
    )

    for current_font_size in range(
        preferred_font_size,
        min_font_size - 1,
        -2,
    ):
        font = get_chinese_font(current_font_size)
        text_w, _ = measure_text(
            draw,
            text,
            font,
        )

        if text_w <= max_width:
            return text, font

    font = get_chinese_font(min_font_size)

    if measure_text(draw, text, font)[0] <= max_width:
        return text, font

    ellipsis = "…"
    fitted_text = text

    while fitted_text:
        candidate = fitted_text + ellipsis

        if measure_text(
            draw,
            candidate,
            font,
        )[0] <= max_width:
            return candidate, font

        fitted_text = fitted_text[:-1]

    return ellipsis, font


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> List[str]:
    max_width = max(
        1,
        int(max_width),
    )

    wrapped_lines: List[str] = []

    source_lines = str(text).splitlines() or [""]

    for source_line in source_lines:
        if source_line == "":
            wrapped_lines.append("")
            continue

        current_line = ""

        for character in source_line:
            candidate = current_line + character

            if (
                current_line
                and measure_text(
                    draw,
                    candidate,
                    font,
                )[0] > max_width
            ):
                wrapped_lines.append(current_line)
                current_line = character
            else:
                current_line = candidate

        if current_line:
            wrapped_lines.append(current_line)

    return wrapped_lines or [""]


def fit_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    preferred_font_size: int,
    min_font_size: int,
    max_width: int,
    max_lines: int,
) -> Tuple[
    ImageFont.FreeTypeFont,
    List[str],
    int,
    int,
]:
    preferred_font_size = max(
        1,
        int(preferred_font_size),
    )

    min_font_size = max(
        1,
        min(
            int(min_font_size),
            preferred_font_size,
        ),
    )

    max_width = max(
        1,
        int(max_width),
    )

    max_lines = max(
        1,
        int(max_lines),
    )

    for current_font_size in range(
        preferred_font_size,
        min_font_size - 1,
        -2,
    ):
        font = get_chinese_font(current_font_size)

        lines = wrap_text_to_width(
            draw=draw,
            text=text,
            font=font,
            max_width=max_width,
        )

        if len(lines) <= max_lines:
            _, line_height = measure_text(
                draw,
                "測試Ag",
                font,
            )

            line_spacing = max(
                4,
                current_font_size // 5,
            )

            return (
                font,
                lines,
                line_height,
                line_spacing,
            )

    font = get_chinese_font(min_font_size)

    lines = wrap_text_to_width(
        draw=draw,
        text=text,
        font=font,
        max_width=max_width,
    )

    if len(lines) > max_lines:
        lines = lines[:max_lines]

        last_line = lines[-1]
        ellipsis = "…"

        while (
            last_line
            and measure_text(
                draw,
                last_line + ellipsis,
                font,
            )[0] > max_width
        ):
            last_line = last_line[:-1]

        lines[-1] = last_line + ellipsis

    _, line_height = measure_text(
        draw,
        "測試Ag",
        font,
    )

    line_spacing = max(
        4,
        min_font_size // 5,
    )

    return (
        font,
        lines,
        line_height,
        line_spacing,
    )


def draw_chinese_text(
    image_bgr: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int = 28,
    color: Tuple[int, int, int] = (0, 255, 0),
    max_width: int | None = None,
    min_font_size: int = 14,
) -> np.ndarray:
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    pil_image = Image.fromarray(image_rgb)

    draw = ImageDraw.Draw(pil_image)

    if max_width is None:
        fitted_text = text
        font = get_chinese_font(font_size)
    else:
        fitted_text, font = fit_single_line_text(
            draw=draw,
            text=text,
            preferred_font_size=font_size,
            min_font_size=min_font_size,
            max_width=max_width,
        )

    rgb_color = (
        color[2],
        color[1],
        color[0],
    )

    draw.text(
        position,
        fitted_text,
        font=font,
        fill=rgb_color,
    )

    return cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR,
    )


def draw_query_panel(
    image_bgr: np.ndarray,
    text_query: str,
    font_size: int = 52,
    padding: int = 20,
    margin: int = 24,
    min_font_size: int = 20,
    max_lines: int = 3,
) -> np.ndarray:
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    pil_image = Image.fromarray(image_rgb)

    draw = ImageDraw.Draw(pil_image)

    label = f"查詢文字：{text_query}"

    img_w, img_h = pil_image.size

    adaptive_margin = min(
        max(4, int(margin)),
        max(4, img_w // 12),
    )

    adaptive_padding = min(
        max(4, int(padding)),
        max(4, img_w // 20),
    )

    panel_width = max(
        1,
        img_w - adaptive_margin * 2,
    )

    text_max_width = max(
        1,
        panel_width - adaptive_padding * 2,
    )

    adaptive_font_size = min(
        max(1, int(font_size)),
        max(
            int(min_font_size),
            int(img_w * 0.045),
        ),
    )

    font, lines, line_height, line_spacing = fit_wrapped_text(
        draw=draw,
        text=label,
        preferred_font_size=adaptive_font_size,
        min_font_size=min_font_size,
        max_width=text_max_width,
        max_lines=max_lines,
    )

    text_block_height = (
        line_height * len(lines)
        + line_spacing * max(
            0,
            len(lines) - 1,
        )
    )

    panel_height = (
        text_block_height
        + adaptive_padding * 2
    )

    x1 = adaptive_margin

    y1 = max(
        adaptive_margin,
        img_h - panel_height - adaptive_margin,
    )

    x2 = img_w - adaptive_margin

    y2 = min(
        img_h - adaptive_margin,
        y1 + panel_height,
    )

    draw.rectangle(
        [x1, y1, x2, y2],
        fill=(0, 0, 0),
    )

    text_y = y1 + adaptive_padding

    for line in lines:
        draw.text(
            (
                x1 + adaptive_padding,
                text_y,
            ),
            line,
            font=font,
            fill=(255, 255, 255),
        )

        text_y += (
            line_height
            + line_spacing
        )

    return cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR,
    )


def draw_boxes_on_image(
    image_bgr: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    text_query: str,
) -> np.ndarray:
    image = image_bgr.copy()

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = (
            box.astype(np.int32)
            .tolist()
        )

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = (
            f"{text_query}: "
            f"{float(score):.3f}"
        )

        image = draw_chinese_text(
            image_bgr=image,
            text=label,
            position=(
                x1,
                max(y1 - 34, 0),
            ),
            font_size=28,
            color=(0, 255, 0),
            max_width=max(
                image.shape[1] - x1 - 8,
                40,
            ),
            min_font_size=14,
        )

    image = draw_query_panel(
        image_bgr=image,
        text_query=text_query,
        font_size=52,
        padding=20,
        margin=24,
    )

    return image


def resolve_device(
    device_name: str,
) -> torch.device:
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[WARNING] CUDA 不可用，改用 CPU")
            return torch.device("cpu")

        if ":" in device_name:
            device_index = int(
                device_name.split(":")[1]
            )

            if device_index >= torch.cuda.device_count():
                raise ValueError(
                    f"指定 {device_name}，但目前只有 "
                    f"{torch.cuda.device_count()} 張 CUDA GPU"
                )

        return torch.device(device_name)

    return torch.device("cpu")


@torch.inference_mode()
def infer_one_query(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    orig_w: int,
    orig_h: int,
    text_query: str,
    score_thr: float,
    iou_thr: float,
    top_k: int,
) -> Dict[str, Any]:
    outputs = model(
        img=image_tensor,
        texts=[text_query],
    )

    pred_boxes = outputs["bbox"][0]

    pred_scores = (
        outputs["score_logit"][0]
        .squeeze(-1)
        .sigmoid()
    )

    num_predictions = pred_scores.numel()

    score_mask = (
        pred_scores >= score_thr
    )

    filtered_boxes = pred_boxes[
        score_mask
    ]

    filtered_scores = pred_scores[
        score_mask
    ]

    if filtered_scores.numel() == 0:
        return {
            "query": text_query,
            "boxes": np.empty(
                (0, 4),
                dtype=np.float32,
            ),
            "scores": np.empty(
                (0,),
                dtype=np.float32,
            ),
            "num_predictions": int(
                num_predictions
            ),
            "after_score_threshold": 0,
            "after_nms": 0,
            "raw_outputs": outputs,
        }

    filtered_pixel_boxes = xyxy_norm_to_pixel(
        boxes=filtered_boxes,
        width=orig_w,
        height=orig_h,
    )

    keep_indices = nms(
        boxes=filtered_pixel_boxes,
        scores=filtered_scores,
        iou_threshold=iou_thr,
    )

    keep_indices = keep_indices[:top_k]

    final_boxes = filtered_pixel_boxes[
        keep_indices
    ]

    final_scores = filtered_scores[
        keep_indices
    ]

    return {
        "query": text_query,
        "boxes": (
            final_boxes
            .detach()
            .cpu()
            .numpy()
        ),
        "scores": (
            final_scores
            .detach()
            .cpu()
            .numpy()
        ),
        "num_predictions": int(
            num_predictions
        ),
        "after_score_threshold": int(
            filtered_scores.numel()
        ),
        "after_nms": int(
            len(keep_indices)
        ),
        "raw_outputs": outputs,
    }


@torch.inference_mode()
def infer_multi_queries(
    ckpt_path: str | Path,
    image_path: str | Path,
    text_queries: List[str],
    save_path: str | Path = "result_side_by_side.jpg",
    image_size: int = 512,
    score_thr: float = 0.20,
    iou_thr: float = 0.50,
    top_k: int = 20,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    if not text_queries:
        raise ValueError(
            "text_queries 不可為空 list"
        )

    text_queries = [
        str(query).strip()
        for query in text_queries
        if str(query).strip()
    ]

    if not text_queries:
        raise ValueError(
            "text_queries 內沒有有效文字"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k 必須大於 0"
        )

    if not 0.0 <= score_thr <= 1.0:
        raise ValueError(
            "score_thr 必須位於 [0, 1]"
        )

    if not 0.0 <= iou_thr <= 1.0:
        raise ValueError(
            "iou_thr 必須位於 [0, 1]"
        )

    device_obj = resolve_device(device)

    print(f"[INFO] Device    : {device_obj}")
    print(f"[INFO] Queries   : {text_queries}")
    print(f"[INFO] Score thr : {score_thr}")
    print(f"[INFO] IoU thr   : {iou_thr}")
    print(f"[INFO] Top-K     : {top_k}")

    model = build_model()

    model = load_checkpoint(
        model=model,
        ckpt_path=ckpt_path,
        device=device_obj,
    )

    image_tensor, orig_w, orig_h = preprocess_image(
        image_path=image_path,
        image_size=image_size,
    )

    image_tensor = image_tensor.to(
        device_obj,
        non_blocking=True,
    )

    original_bgr = cv2.imread(
        str(image_path)
    )

    if original_bgr is None:
        raise RuntimeError(
            f"OpenCV 無法讀取影像: {image_path}"
        )

    rendered_images = []
    results = []

    for text_query in text_queries:
        result = infer_one_query(
            model=model,
            image_tensor=image_tensor,
            orig_w=orig_w,
            orig_h=orig_h,
            text_query=text_query,
            score_thr=score_thr,
            iou_thr=iou_thr,
            top_k=top_k,
        )

        rendered = draw_boxes_on_image(
            image_bgr=original_bgr,
            boxes=result["boxes"],
            scores=result["scores"],
            text_query=text_query,
        )

        rendered_images.append(rendered)
        results.append(result)

        print(
            f"\n[INFO] Query: "
            f"{text_query}"
        )

        print(
            "[INFO] Total candidates       : "
            f"{result['num_predictions']}"
        )

        print(
            "[INFO] After score threshold : "
            f"{result['after_score_threshold']}"
        )

        print(
            "[INFO] After NMS             : "
            f"{result['after_nms']}"
        )

        for rank, (box, score) in enumerate(
            zip(
                result["boxes"],
                result["scores"],
            ),
            start=1,
        ):
            print(
                f"  rank={rank:02d}, "
                f"query={text_query!r}, "
                f"score={float(score):.4f}, "
                f"box={box.astype(int).tolist()}"
            )

    side_by_side_image = cv2.hconcat(
        rendered_images
    )

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(save_path),
        side_by_side_image,
    )

    if not success:
        raise RuntimeError(
            f"結果影像儲存失敗: {save_path}"
        )

    print(
        "\n[INFO] Saved side-by-side result: "
        f"{save_path}"
    )

    return {
        "results": results,
        "save_path": str(save_path),
    }


if __name__ == "__main__":
    result = infer_multi_queries(
        ckpt_path=(
            "/home/soic/Desktop/LightDet/units/model/"
            "runs/train/lightdet_neg_pool/best_map50_95.pt"
        ),
        image_path=(
            # "/home/soic/Desktop/Hualien_1080p/images/val/"
            # "20260407__cam05__cam05_20260407_060013_4_001424_9c29aa8ed6.jpg"
            "/home/soic/Desktop/datasetPreTest15000/dataset/"
            "datasetPreTest15000_mixed/images/train/"
            "2021_10_25_13_27_28_01200.jpg"
        ),
        text_queries=[
            "一台行駛的計程車",
            "停靠的船",
        ],
        save_path=(
            "/home/soic/Desktop/LightDet/datasets/pre/"
            "result_side_by_side.jpg"
        ),
        image_size=512,
        score_thr=0.39,
        iou_thr=0.1,
        top_k=10,
        device="cuda:0",
    )