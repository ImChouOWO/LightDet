#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import torch
from torch.amp import autocast
from torchvision.ops import nms
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UNITS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, UNITS_DIR)

from model.cards.main import Model


def cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=-1)


def load_model(
    checkpoint_path,
    device,
    use_ema=True
):
    model = Model(num_classes=1).to(device)

    ckpt = torch.load(
        checkpoint_path,
        map_location=device
    )

    if use_ema and "ema" in ckpt:
        state_dict = ckpt["ema"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def preprocess_image(
    image_path,
    image_size=(640, 640),
    device="cuda"
):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    target_h, target_w = image_size

    resized = TF.resize(image, [target_h, target_w])
    tensor = TF.to_tensor(resized).unsqueeze(0).to(device)

    return image, tensor, (orig_h, orig_w), (target_h, target_w)


def scale_boxes_to_original(
    boxes_xyxy,
    orig_size,
    resized_size
):
    orig_h, orig_w = orig_size
    resized_h, resized_w = resized_size

    boxes = boxes_xyxy.clone()

    boxes[:, [0, 2]] = boxes[:, [0, 2]] * (orig_w / resized_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * (orig_h / resized_h)

    return boxes


@torch.no_grad()
def infer_single_image(
    model,
    image_path,
    query_text,
    device,
    image_size=(640, 640),
    score_thr=0.25,
    top_k=20,
    nms_iou_thr=0.5,
    use_amp=True
):
    orig_image, image_tensor, orig_size, resized_size = preprocess_image(
        image_path=image_path,
        image_size=image_size,
        device=device
    )

    empty_boxes = torch.empty(
        0,
        4,
        device=device,
        dtype=torch.float32
    )

    boxes_per_image = [empty_boxes]
    query_texts = [query_text]

    amp_enabled = use_amp and device.type == "cuda"

    with autocast(device_type=device.type, enabled=amp_enabled):
        outputs = model(
            images=image_tensor,
            boxes_per_image=boxes_per_image,
            texts=query_texts,
            image_size=image_size
        )

    pred_bbox = outputs["bbox"][0]       # [N, 4], normalized cxcywh
    score_logits = outputs["score"][0]   # [N, 1]

    scores = torch.sigmoid(score_logits).squeeze(-1)

    keep = scores >= score_thr

    if keep.sum() > 0:
        selected_boxes = pred_bbox[keep]
        selected_scores = scores[keep]
    else:
        k = min(top_k, pred_bbox.shape[0])
        selected_scores, top_idx = scores.topk(k=k)
        selected_boxes = pred_bbox[top_idx]

    if selected_scores.numel() > top_k:
        selected_scores, top_idx = selected_scores.topk(k=top_k)
        selected_boxes = selected_boxes[top_idx]

    boxes_xyxy_norm = cxcywh_to_xyxy(selected_boxes)

    boxes_xyxy_resized = boxes_xyxy_norm.clone()
    boxes_xyxy_resized[:, [0, 2]] *= image_size[1]
    boxes_xyxy_resized[:, [1, 3]] *= image_size[0]

    boxes_xyxy_resized[:, [0, 2]] = boxes_xyxy_resized[:, [0, 2]].clamp(0, image_size[1] - 1)
    boxes_xyxy_resized[:, [1, 3]] = boxes_xyxy_resized[:, [1, 3]].clamp(0, image_size[0] - 1)

    keep_idx = nms(
        boxes_xyxy_resized.float(),
        selected_scores.float(),
        iou_threshold=nms_iou_thr
    )

    boxes_xyxy_resized = boxes_xyxy_resized[keep_idx]
    selected_scores = selected_scores[keep_idx]

    boxes_xyxy_orig = scale_boxes_to_original(
        boxes_xyxy=boxes_xyxy_resized,
        orig_size=orig_size,
        resized_size=image_size
    )

    results = []

    for box, score in zip(boxes_xyxy_orig, selected_scores):
        results.append({
            "bbox_xyxy": box.detach().cpu().tolist(),
            "score": float(score.detach().cpu())
        })

    return orig_image, results


def draw_results(
    image,
    results,
    query_text,
    output_path
):
    draw = ImageDraw.Draw(image)

    for item in results:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        score = item["score"]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )

        label = f"{query_text} {score:.3f}"

        draw.text(
            (x1, max(0, y1 - 20)),
            label,
            fill="red"
        )

    image.save(output_path)


def main():
    checkpoint_path = "/home/soic/Desktop/LightDet/units/model/checkpoints/results_2026-05-14_07-52-47/best_iou.pt"
    image_path = "/home/soic/Desktop/LightDet/datasets/images/val/2021_10_20_13_28_41_00000.jpg"
    output_path = "/home/soic/Desktop/LightDet/output/test_result.jpg"

    query_text = "白色的船"

    image_size = (640, 640)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
        use_ema=True
    )

    image, results = infer_single_image(
        model=model,
        image_path=image_path,
        query_text=query_text,
        device=device,
        image_size=image_size,
        score_thr=0.25,
        top_k=20,
        nms_iou_thr=0.5,
        use_amp=True
    )

    for i, item in enumerate(results):
        print(
            f"[{i}] score={item['score']:.4f}, "
            f"bbox={item['bbox_xyxy']}"
        )

    draw_results(
        image=image,
        results=results,
        query_text=query_text,
        output_path=output_path
    )

    print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    main()
