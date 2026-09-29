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
    if not torch.is_tensor(images):
        raise TypeError(f"images must be a Tensor, got {type(images)}")
    if images.ndim != 4:
        raise ValueError(
            f"images must be BCHW [B,C,H,W], got {tuple(images.shape)}"
        )
    if images.shape[1] not in (1, 3, 4):
        raise ValueError(f"unexpected image channel count: {images.shape[1]}")

    source_dtype = images.dtype
    images = images.to(device=device, non_blocking=True)
    if source_dtype == torch.uint8:
        images = images.to(dtype=torch.float32).mul_(1.0 / 255.0)
    elif source_dtype.is_floating_point:
        if images.dtype != torch.float32:
            images = images.float()
    else:
        raise TypeError(f"Unsupported image dtype: {source_dtype}")

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
    query_texts = batch.get("query_texts", batch.get("captions"))
    if query_texts is None:
        raise KeyError(
            f"Batch missing query_texts/captions. Keys: {sorted(batch.keys())}"
        )
    query_texts = [str(value) for value in query_texts]

    if "unique_images" in batch:
        images = move_images_to_device(
            batch["unique_images"],
            device,
            channels_last,
        )
        image_indices = batch.get("image_indices")
        if image_indices is None:
            # ODVG image-level batching uses exactly one caption per image.
            if len(query_texts) != int(images.shape[0]):
                raise KeyError(
                    "Image-level batch is missing image_indices and caption/image "
                    "counts differ"
                )
            image_indices = torch.arange(
                images.shape[0],
                dtype=torch.long,
            )
        image_indices = torch.as_tensor(
            image_indices,
            dtype=torch.long,
        ).reshape(-1)
        if image_indices.numel() != len(query_texts):
            raise ValueError("image_indices/query_texts size mismatch")
        if image_indices.numel():
            if int(image_indices.min()) < 0 or int(image_indices.max()) >= images.shape[0]:
                raise IndexError("image_indices out of range")
        return (
            images,
            query_texts,
            image_indices.to(device=device, non_blocking=True),
        )

    if "images" in batch:
        images = move_images_to_device(
            batch["images"],
            device,
            channels_last,
        )
        if images.shape[0] != len(query_texts):
            raise ValueError("images/query_texts batch size mismatch")
        return images, query_texts, None

    raise KeyError("Batch must contain unique_images or images")


def _attach_odvg_metadata(outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve scalar compatibility and expose ODVG tensors to the loss."""
    token_logits = outputs.get(
        "token_alignment_logits",
        outputs.get("main_token_alignment_logits"),
    )
    token_offsets = outputs.get("token_offsets")
    alignment_mask = outputs.get("alignment_text_mask")
    quality = outputs.get("quality_logit", outputs.get("main_quality_logit"))

    # Phrase-specific fusion is intentionally deferred. Until inference has a
    # selected phrase token map, the scalar selection score is localization
    # quality.
    final_score = outputs.get("final_score_logit")
    if (
        final_score is None
        or not torch.is_tensor(final_score)
        or final_score.ndim != 3
        or final_score.shape[-1] != 1
    ):
        if quality is not None:
            outputs["final_score_logit"] = quality

    score = outputs.get("score_logit", outputs.get("main_score_logit"))
    if (
        score is None
        or not torch.is_tensor(score)
        or score.ndim != 3
        or score.shape[-1] != 1
    ):
        score = quality
        if score is not None:
            outputs["score_logit"] = score
            outputs["main_score_logit"] = score

    if score is not None and torch.is_tensor(score):
        score._quality_logit = quality if quality is not None else score
        score._token_alignment_logits = token_logits
        score._token_offsets = token_offsets
        score._alignment_text_mask = alignment_mask
        score._query_stage_separated = bool(
            outputs.get("query_stage_separated", False)
        )
        for metadata_name, output_key in (
            ("_localization_query_out", "main_localization_query_out"),
            ("_score_query_out", "main_score_query_out"),
        ):
            value = outputs.get(output_key)
            if value is not None:
                setattr(score, metadata_name, value)

    aux_score = outputs.get("aux_score_logit")
    aux_token = outputs.get("aux_token_alignment_logits")
    aux_quality = outputs.get("aux_quality_logit")
    if aux_score is not None and torch.is_tensor(aux_score):
        aux_score._quality_logit = (
            aux_quality if aux_quality is not None else aux_score
        )
        aux_score._token_alignment_logits = aux_token
        aux_score._token_offsets = token_offsets
        aux_score._alignment_text_mask = alignment_mask

    # Keep text_alignment_logit token-level. Phrase-specific scalar scores are
    # produced only after selecting a phrase character span.
    return outputs


def forward_model_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    query_texts: List[str],
    image_indices: Optional[torch.Tensor],
    return_aux: Optional[bool] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"return_aux": return_aux}
    if image_indices is not None:
        kwargs["image_indices"] = image_indices
    outputs = model(images, query_texts, **kwargs)
    if not isinstance(outputs, dict):
        raise TypeError(f"VisionTextModel must return dict, got {type(outputs)}")
    return _attach_odvg_metadata(outputs)


def build_text_conditioning_probe(
    query_texts: Sequence[str],
    text_negative_mask: Optional[torch.Tensor],
) -> Tuple[Optional[List[str]], List[int], str]:
    texts = [str(value) for value in query_texts]
    count = len(texts)
    if count < 2 or len(set(texts)) < 2:
        return None, [], "insufficient_distinct_texts"

    if text_negative_mask is not None:
        mask = torch.as_tensor(text_negative_mask, dtype=torch.bool).reshape(-1)
        if mask.numel() == count:
            negatives = torch.nonzero(mask, as_tuple=False).flatten().tolist()
            positives = torch.nonzero(~mask, as_tuple=False).flatten().tolist()
            for positive_index in positives:
                for negative_index in negatives:
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

    best: Optional[List[str]] = None
    changed_best: List[int] = []
    for shift in range(1, count):
        candidate = texts[shift:] + texts[:shift]
        changed = [
            index
            for index, (source, target) in enumerate(zip(texts, candidate))
            if source != target
        ]
        if len(changed) > len(changed_best):
            best = candidate
            changed_best = changed
    if not changed_best:
        return None, [], "no_changed_rows"
    return best, changed_best, "text_rotation"


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
    """Compact images/boxes while preserving ODVG phrase supervision."""
    batch = grounding_collate_fn(items)
    targets = batch.pop("targets", [])

    box_tensors: List[torch.Tensor] = []
    offsets = [0]
    positive_char_spans: List[Any] = []
    target_phrases: List[Any] = []
    target_semantic_keys: List[Any] = []
    target_region_indices: List[Any] = []

    for target in targets:
        boxes = target.get("boxes")
        if boxes is None:
            raise KeyError("Each target must contain 'boxes'")
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4).contiguous()
        box_tensors.append(boxes)
        offsets.append(offsets[-1] + int(boxes.shape[0]))
        positive_char_spans.append(target.get("positive_char_spans", []))
        target_phrases.append(target.get("phrases", []))
        target_semantic_keys.append(target.get("semantic_keys", []))
        target_region_indices.append(target.get("region_indices", []))

    flat_boxes = (
        torch.cat(box_tensors, dim=0).contiguous()
        if box_tensors and offsets[-1] > 0
        else torch.empty((0, 4), dtype=torch.float32)
    )
    batch["target_boxes_flat"] = flat_boxes
    batch["target_offsets"] = torch.tensor(offsets, dtype=torch.int64)
    batch["num_targets"] = len(targets)

    # Preserve the ODVG metadata even if a previous collate version did not
    # expose it at the top level.
    batch.setdefault("positive_char_spans", positive_char_spans)
    batch.setdefault("target_phrases", target_phrases)
    batch.setdefault("target_semantic_keys", target_semantic_keys)
    batch.setdefault("target_region_indices", target_region_indices)

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
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_raw_targets(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "target_boxes_flat" in batch and "target_offsets" in batch:
        flat_boxes = batch["target_boxes_flat"]
        offsets_tensor = batch["target_offsets"]
        if not torch.is_tensor(flat_boxes) or not torch.is_tensor(offsets_tensor):
            raise TypeError("Compact target tensors are invalid")
        offsets = offsets_tensor.tolist()
        spans = batch.get("positive_char_spans", [[] for _ in range(len(offsets) - 1)])
        phrases = batch.get("target_phrases", [[] for _ in range(len(offsets) - 1)])
        semantic_keys = batch.get(
            "target_semantic_keys",
            [[] for _ in range(len(offsets) - 1)],
        )
        region_indices = batch.get(
            "target_region_indices",
            [[] for _ in range(len(offsets) - 1)],
        )
        captions = batch.get("captions", batch.get("query_texts", []))
        result: List[Dict[str, Any]] = []
        for index in range(len(offsets) - 1):
            target: Dict[str, Any] = {
                "boxes": flat_boxes[offsets[index] : offsets[index + 1]],
                "positive_char_spans": spans[index],
                "phrases": phrases[index],
                "semantic_keys": semantic_keys[index],
                "region_indices": region_indices[index],
            }
            if index < len(captions):
                target["caption"] = str(captions[index])
            result.append(target)
        return result

    if "targets" in batch:
        return batch["targets"]

    boxes_list = batch.get("target_boxes_per_image")
    if boxes_list is None:
        raise KeyError("Batch does not contain targets")
    labels_list = batch.get("target_labels_per_image")
    result = []
    for index, boxes in enumerate(boxes_list):
        target: Dict[str, Any] = {"boxes": boxes}
        if labels_list is not None:
            target["labels"] = labels_list[index]
        result.append(target)
    return result


def move_targets_to_device(
    batch: Dict[str, Any],
    device: torch.device,
) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    for target in get_raw_targets(batch):
        moved: Dict[str, Any] = {}
        for key, value in target.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device=device, non_blocking=True)
            else:
                moved[key] = value
        if "boxes" not in moved:
            raise KeyError("target must contain boxes")
        targets.append(moved)
    return targets


def get_target_boxes_cpu(batch: Dict[str, Any]) -> List[torch.Tensor]:
    result: List[torch.Tensor] = []
    for target in get_raw_targets(batch):
        boxes = torch.as_tensor(target["boxes"]).detach()
        if boxes.device.type != "cpu":
            boxes = boxes.cpu()
        result.append(boxes.to(dtype=torch.float32).reshape(-1, 4))
    return result


def get_score_logit(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "score_logit" in outputs:
        return outputs["score_logit"]
    if "score" in outputs:
        return outputs["score"]
    if "quality_logit" in outputs:
        return outputs["quality_logit"]
    raise KeyError("Model output must contain score_logit, score, or quality_logit")


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




def char_spans_to_token_mask(
    token_offsets: torch.Tensor,
    char_spans: Sequence[Sequence[int]],
    valid_token_mask: Optional[torch.Tensor] = None,
    *,
    strict: bool = True,
) -> torch.Tensor:
    """Convert caption character spans into one boolean token mask [L]."""
    offsets = torch.as_tensor(token_offsets)
    if offsets.ndim != 2 or offsets.shape[-1] != 2:
        raise ValueError(
            "token_offsets must have shape [L,2], got "
            f"{tuple(offsets.shape)}"
        )

    mask = torch.zeros(
        offsets.shape[0],
        dtype=torch.bool,
        device=offsets.device,
    )
    token_start = offsets[:, 0]
    token_end = offsets[:, 1]
    real_token = token_end > token_start

    normalized_spans: List[Tuple[int, int]] = []
    for raw_span in char_spans:
        if len(raw_span) != 2:
            raise ValueError(f"Invalid character span: {raw_span!r}")
        start, end = int(raw_span[0]), int(raw_span[1])
        if end <= start:
            raise ValueError(f"Invalid character span: {(start, end)}")
        normalized_spans.append((start, end))
        mask |= real_token & (token_start < end) & (token_end > start)

    if valid_token_mask is not None:
        valid = torch.as_tensor(
            valid_token_mask,
            device=offsets.device,
            dtype=torch.bool,
        ).reshape(-1)
        if valid.numel() != offsets.shape[0]:
            raise ValueError(
                "valid_token_mask length mismatch: "
                f"{valid.numel()} != {offsets.shape[0]}"
            )
        mask &= valid

    if strict and normalized_spans and not bool(mask.any()):
        raise ValueError(
            "Character spans do not overlap any valid BERT token: "
            f"{normalized_spans}"
        )
    return mask


def pool_phrase_alignment_logits(
    token_alignment_logits: torch.Tensor,
    phrase_token_mask: torch.Tensor,
    *,
    reduction: str = "mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Reduce query-to-token logits [Q,L] into one phrase logit per query [Q].

    ``mean`` averages token probabilities. ``geometric_mean`` requires all
    phrase tokens to be confident and is therefore stricter.
    """
    logits = torch.as_tensor(token_alignment_logits)
    if logits.ndim != 2:
        raise ValueError(
            "token_alignment_logits must have shape [Q,L], got "
            f"{tuple(logits.shape)}"
        )

    mask = torch.as_tensor(
        phrase_token_mask,
        device=logits.device,
        dtype=torch.bool,
    ).reshape(-1)
    if mask.numel() != logits.shape[-1]:
        raise ValueError(
            "phrase token mask length mismatch: "
            f"{mask.numel()} != {logits.shape[-1]}"
        )
    if not bool(mask.any()):
        raise ValueError("phrase_token_mask contains no positive token")

    probability = logits.float().sigmoid()[:, mask]
    reduction = str(reduction).strip().lower()
    if reduction in {"mean", "arithmetic_mean", "avg", "average"}:
        pooled_probability = probability.mean(dim=-1)
    elif reduction in {"geometric", "geometric_mean", "gmean"}:
        pooled_probability = torch.exp(
            torch.log(probability.clamp_min(eps)).mean(dim=-1)
        )
    elif reduction == "min":
        pooled_probability = probability.min(dim=-1).values
    else:
        raise ValueError(
            "reduction must be mean, geometric_mean, or min, got "
            f"{reduction!r}"
        )

    pooled_probability = pooled_probability.clamp(
        min=float(eps),
        max=1.0 - float(eps),
    )
    return torch.logit(pooled_probability)


def fuse_quality_and_phrase_logits(
    quality_logit: torch.Tensor,
    phrase_alignment_logit: torch.Tensor,
    *,
    fusion: str = "geometric_mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fuse localization quality and phrase alignment into one scalar logit."""
    quality = torch.as_tensor(quality_logit).float()
    alignment = torch.as_tensor(
        phrase_alignment_logit,
        device=quality.device,
    ).float()

    if quality.ndim == 2 and quality.shape[-1] == 1:
        quality = quality.squeeze(-1)
    if quality.ndim != 1 or alignment.ndim != 1:
        raise ValueError(
            "quality/alignment logits must both reduce to [Q], got "
            f"{tuple(quality.shape)} and {tuple(alignment.shape)}"
        )
    if quality.shape != alignment.shape:
        raise ValueError("quality/alignment query count mismatch")

    quality_probability = quality.sigmoid()
    alignment_probability = alignment.sigmoid()
    fusion = str(fusion).strip().lower()

    if fusion in {"geometric", "geometric_mean", "sqrt_product"}:
        probability = torch.sqrt(
            (quality_probability * alignment_probability).clamp_min(0.0)
        )
    elif fusion in {"product", "multiply"}:
        probability = quality_probability * alignment_probability
    elif fusion in {"arithmetic_mean", "mean", "average"}:
        probability = 0.5 * (
            quality_probability + alignment_probability
        )
    else:
        raise ValueError(
            "fusion must be geometric_mean, product, or arithmetic_mean, got "
            f"{fusion!r}"
        )

    probability = probability.clamp(
        min=float(eps),
        max=1.0 - float(eps),
    )
    return torch.logit(probability)


def score_queries_for_char_spans(
    *,
    quality_logit: torch.Tensor,
    token_alignment_logits: torch.Tensor,
    token_offsets: torch.Tensor,
    char_spans: Sequence[Sequence[int]],
    valid_token_mask: Optional[torch.Tensor] = None,
    token_reduction: str = "mean",
    score_fusion: str = "geometric_mean",
    strict: bool = True,
) -> Dict[str, torch.Tensor]:
    """Return quality, phrase-alignment and fused probabilities for one phrase."""
    phrase_token_mask = char_spans_to_token_mask(
        token_offsets,
        char_spans,
        valid_token_mask,
        strict=strict,
    )
    phrase_alignment_logit = pool_phrase_alignment_logits(
        token_alignment_logits,
        phrase_token_mask,
        reduction=token_reduction,
    )
    final_logit = fuse_quality_and_phrase_logits(
        quality_logit,
        phrase_alignment_logit,
        fusion=score_fusion,
    )

    quality = torch.as_tensor(quality_logit).float()
    if quality.ndim == 2 and quality.shape[-1] == 1:
        quality = quality.squeeze(-1)

    return {
        "token_mask": phrase_token_mask,
        "quality_logit": quality,
        "quality_score": quality.sigmoid(),
        "phrase_alignment_logit": phrase_alignment_logit,
        "phrase_alignment_score": phrase_alignment_logit.sigmoid(),
        "final_logit": final_logit,
        "final_score": final_logit.sigmoid(),
    }



__all__ = [
    "box_iou_xyxy",
    "char_spans_to_token_mask",
    "fuse_quality_and_phrase_logits",
    "pool_phrase_alignment_logits",
    "score_queries_for_char_spans",
    "build_text_conditioning_probe",
    "compact_grounding_collate_fn",
    "forward_model_batch",
    "get_raw_targets",
    "get_score_logit",
    "get_target_boxes_cpu",
    "make_progress_bar",
    "move_images_to_device",
    "move_targets_to_device",
    "prepare_model_batch",
    "seed_dataloader_worker",
]
