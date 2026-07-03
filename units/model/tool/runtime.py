from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from units.model.pipeline.data import grounding_collate_fn

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
    return_aux: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Unified model forward.

    return_aux:
        None  -> let VisionTextModel decide from train/eval mode
        True  -> force main + auxiliary outputs
        False -> force main-only output
    """
    model_kwargs: Dict[str, Any] = {
        "return_aux": return_aux,
    }

    if image_indices is not None:
        model_kwargs["image_indices"] = image_indices

    return model(
        images,
        query_texts,
        **model_kwargs,
    )

def build_text_conditioning_probe(
    query_texts: Sequence[str],
    text_negative_mask: Optional[torch.Tensor],
) -> Tuple[Optional[List[str]], List[int], str]:
    """
    Build a second text ordering while keeping every image tensor and
    image_indices entry unchanged.

    Priority is given to swapping one positive and one negative description,
    because this directly checks that positive/negative wording can alter the
    predicted localization. If such a pair is unavailable, use the rotation
    that changes the largest number of query strings.
    """
    texts = [str(value) for value in query_texts]
    count = len(texts)
    if count < 2 or len(set(texts)) < 2:
        return None, [], "insufficient_distinct_texts"

    if text_negative_mask is not None:
        if not torch.is_tensor(text_negative_mask):
            mask = torch.as_tensor(text_negative_mask, dtype=torch.bool)
        else:
            mask = text_negative_mask.detach().to(device="cpu", dtype=torch.bool)
        mask = mask.reshape(-1)
        if mask.numel() == count:
            negative_indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
            positive_indices = torch.nonzero(~mask, as_tuple=False).flatten().tolist()
            for positive_index in positive_indices:
                for negative_index in negative_indices:
                    if texts[positive_index] == texts[negative_index]:
                        continue
                    permuted = list(texts)
                    permuted[positive_index], permuted[negative_index] = (
                        permuted[negative_index],
                        permuted[positive_index],
                    )
                    return (
                        permuted,
                        [positive_index, negative_index],
                        "positive_negative_swap",
                    )

    best_permuted: Optional[List[str]] = None
    best_changed: List[int] = []
    for shift in range(1, count):
        candidate = texts[shift:] + texts[:shift]
        changed = [
            index
            for index, (source, target) in enumerate(zip(texts, candidate))
            if source != target
        ]
        if len(changed) > len(best_changed):
            best_permuted = candidate
            best_changed = changed

    if not best_changed:
        return None, [], "no_changed_rows"
    return best_permuted, best_changed, "text_rotation"

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

