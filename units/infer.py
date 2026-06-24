#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.ops import nms


CURRENT_DIR = Path(__file__).resolve().parent
UNITS_DIR = CURRENT_DIR.parent

if str(UNITS_DIR) not in sys.path:
    sys.path.insert(0, str(UNITS_DIR))


def build_model():
    from units.tool.card import VisionTextModel

    model = VisionTextModel(
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

    return model


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


def draw_boxes(
    image_path: str | Path,
    boxes: np.ndarray,
    scores: np.ndarray,
    text_query: str,
    save_path: str | Path,
) -> int:
    image_path = Path(image_path)
    save_path = Path(save_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"OpenCV 無法讀取影像: {image_path}")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    draw_count = 0

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(np.int32).tolist()

        if x2 <= x1 or y2 <= y1:
            continue

        draw_count += 1

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = f"{text_query}: {float(score):.3f}"

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    success = cv2.imwrite(str(save_path), image)

    if not success:
        raise RuntimeError(f"結果影像儲存失敗: {save_path}")

    return draw_count


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[WARNING] CUDA 不可用，改用 CPU")
            return torch.device("cpu")

        if ":" in device_name:
            device_index = int(device_name.split(":")[1])

            if device_index >= torch.cuda.device_count():
                raise ValueError(
                    f"指定 {device_name}，"
                    f"但目前只有 {torch.cuda.device_count()} 張 CUDA GPU"
                )

        return torch.device(device_name)

    return torch.device("cpu")


@torch.inference_mode()
def infer(
    ckpt_path: str | Path,
    image_path: str | Path,
    text_query: str,
    save_path: str | Path = "result.jpg",
    image_size: int = 512,
    score_thr: float = 0.20,
    iou_thr: float = 0.50,
    top_k: int = 20,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    if not text_query.strip():
        raise ValueError("text_query 不可為空字串")

    if top_k <= 0:
        raise ValueError("top_k 必須大於 0")

    if not 0.0 <= score_thr <= 1.0:
        raise ValueError("score_thr 必須位於 [0, 1]")

    if not 0.0 <= iou_thr <= 1.0:
        raise ValueError("iou_thr 必須位於 [0, 1]")

    device_obj = resolve_device(device)

    print(f"[INFO] Device    : {device_obj}")
    print(f"[INFO] Query     : {text_query}")
    print(f"[INFO] Score thr : {score_thr}")
    print(f"[INFO] IoU thr   : {iou_thr}")

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

    score_mask = pred_scores >= score_thr

    filtered_boxes = pred_boxes[score_mask]
    filtered_scores = pred_scores[score_mask]

    if filtered_scores.numel() == 0:
        boxes_np = np.empty((0, 4), dtype=np.float32)
        scores_np = np.empty((0,), dtype=np.float32)
        keep_count = 0
    else:
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

        final_boxes = filtered_pixel_boxes[keep_indices]
        final_scores = filtered_scores[keep_indices]

        boxes_np = final_boxes.detach().cpu().numpy()
        scores_np = final_scores.detach().cpu().numpy()
        keep_count = len(keep_indices)

    draw_count = draw_boxes(
        image_path=image_path,
        boxes=boxes_np,
        scores=scores_np,
        text_query=text_query,
        save_path=save_path,
    )

    print("\n[INFO] Model output shapes")
    for key, value in outputs.items():
        if torch.is_tensor(value):
            print(
                f"  {key:16s}: "
                f"shape={tuple(value.shape)}, "
                f"dtype={value.dtype}, "
                f"device={value.device}"
            )

    print("\n[INFO] Detection results")

    for rank, (box, score) in enumerate(
        zip(boxes_np, scores_np),
        start=1,
    ):
        print(
            f"  rank={rank:02d}, "
            f"query={text_query!r}, "
            f"score={float(score):.4f}, "
            f"box={box.astype(int).tolist()}"
        )

    print(f"\n[INFO] Total candidates       : {num_predictions}")
    print(f"[INFO] After score threshold : {filtered_scores.numel()}")
    print(f"[INFO] After NMS             : {keep_count}")
    print(f"[INFO] Drawn boxes           : {draw_count}")
    print(f"[INFO] Saved result          : {save_path}")

    return {
        "boxes": boxes_np,
        "scores": scores_np,
        "num_predictions": num_predictions,
        "after_score_threshold": int(filtered_scores.numel()),
        "after_nms": int(keep_count),
        "draw_count": draw_count,
        "raw_outputs": outputs,
    }


if __name__ == "__main__":
    result = infer(
        ckpt_path=(
            "/home/soic/Desktop/LightDet/units/model/"
            "runs/train/lightdet_rank_smooth_010/"
            "best_map50.pt"
        ),
        image_path=(
            "/home/soic/Desktop/Hualien_1080p/images/val/20260407__cam05__cam05_20260407_060013_4_001424_9c29aa8ed6.jpg"
        ),
        text_query="船",
        save_path="/home/soic/Desktop/LightDet/datasets/pre/result.jpg",
        image_size=512,
        score_thr=0.39,
        iou_thr=0.1,
        top_k=10,
        device="cuda:0",
    )