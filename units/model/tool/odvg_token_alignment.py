from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch


class TokenAlignmentError(ValueError):
    """ODVG character spans cannot be aligned to tokenizer tokens."""


def _normalize_span(span: Any, *, context: str) -> tuple[int, int]:
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        raise TokenAlignmentError(
            f"{context} must be [start, end], got {span!r}"
        )

    try:
        start = int(span[0])
        end = int(span[1])
    except (TypeError, ValueError) as error:
        raise TokenAlignmentError(
            f"{context} contains a non-integer value: {span!r}"
        ) from error

    if start < 0 or end <= start:
        raise TokenAlignmentError(
            f"{context} must satisfy 0 <= start < end, got {span!r}"
        )

    return start, end


def tokenize_captions_with_offsets(
    tokenizer: Any,
    captions: Sequence[str],
    *,
    max_length: int,
    device: Optional[torch.device | str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize complete ODVG captions and preserve character offsets.

    Returns:
        input_ids:       [B, L]
        attention_mask:  [B, L]
        token_offsets:   [B, L, 2], character range [start, end)
    """
    captions = [str(caption) for caption in captions]
    if not captions:
        raise TokenAlignmentError("captions must not be empty")

    max_length = int(max_length)
    if max_length <= 2:
        raise TokenAlignmentError(
            f"max_length must be greater than 2, got {max_length}"
        )

    encoded = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    result = {
        "input_ids": encoded["input_ids"].to(dtype=torch.long),
        "attention_mask": encoded["attention_mask"].to(dtype=torch.long),
        "token_offsets": encoded["offset_mapping"].to(dtype=torch.long),
    }

    if "token_type_ids" in encoded:
        result["token_type_ids"] = encoded["token_type_ids"].to(
            dtype=torch.long
        )

    if device is not None:
        result = {
            key: value.to(device=device, non_blocking=True)
            for key, value in result.items()
        }

    return result


def char_spans_to_token_mask(
    token_offsets: torch.Tensor,
    char_spans: Sequence[Sequence[int]],
    *,
    attention_mask: Optional[torch.Tensor] = None,
    caption: Optional[str] = None,
    strict: bool = True,
    context: str = "target",
) -> torch.Tensor:
    """Convert one target's ODVG character spans to a token mask."""
    if not torch.is_tensor(token_offsets):
        token_offsets = torch.as_tensor(token_offsets, dtype=torch.long)

    token_offsets = token_offsets.to(dtype=torch.long).reshape(-1, 2)
    token_count = int(token_offsets.shape[0])

    if attention_mask is None:
        valid_attention = torch.ones(
            token_count,
            dtype=torch.bool,
            device=token_offsets.device,
        )
    else:
        if not torch.is_tensor(attention_mask):
            attention_mask = torch.as_tensor(attention_mask)
        valid_attention = attention_mask.to(
            device=token_offsets.device,
            dtype=torch.bool,
        ).reshape(-1)
        if valid_attention.numel() != token_count:
            raise TokenAlignmentError(
                f"{context}: attention/token length mismatch: "
                f"{valid_attention.numel()} != {token_count}"
            )

    token_start = token_offsets[:, 0]
    token_end = token_offsets[:, 1]
    valid_token = valid_attention & (token_end > token_start)
    positive_mask = torch.zeros(
        token_count,
        dtype=torch.bool,
        device=token_offsets.device,
    )

    normalized_spans: List[tuple[int, int]] = []
    for span_index, span in enumerate(char_spans):
        start, end = _normalize_span(
            span,
            context=f"{context}.char_spans[{span_index}]",
        )

        if caption is not None and end > len(caption):
            raise TokenAlignmentError(
                f"{context}: character span [{start}, {end}) exceeds "
                f"caption length {len(caption)}"
            )

        normalized_spans.append((start, end))
        overlaps = (token_start < end) & (token_end > start)
        positive_mask |= valid_token & overlaps

    if strict and normalized_spans and not bool(positive_mask.any().item()):
        preview = ""
        if caption is not None:
            pieces = [
                caption[start:end]
                for start, end in normalized_spans
                if end <= len(caption)
            ]
            preview = f", phrase={pieces!r}"

        raise TokenAlignmentError(
            f"{context}: no tokenizer token matched char spans "
            f"{normalized_spans}{preview}. The caption may have been truncated; "
            "increase model.text_max_length."
        )

    return positive_mask


def build_positive_token_maps(
    *,
    token_offsets: torch.Tensor,
    positive_char_spans: Sequence[
        Sequence[Sequence[Sequence[int]]]
    ],
    attention_mask: Optional[torch.Tensor] = None,
    captions: Optional[Sequence[str]] = None,
    strict: bool = True,
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    Convert an ODVG batch from character spans to token supervision.

    positive_char_spans shape:
        [B][N_target][N_span][2]
    """
    if not torch.is_tensor(token_offsets):
        token_offsets = torch.as_tensor(token_offsets, dtype=torch.long)

    if token_offsets.ndim != 3 or token_offsets.shape[-1] != 2:
        raise TokenAlignmentError(
            "token_offsets must have shape [B, L, 2], got "
            f"{tuple(token_offsets.shape)}"
        )

    batch_size, token_count, _ = token_offsets.shape

    if len(positive_char_spans) != batch_size:
        raise TokenAlignmentError(
            "positive_char_spans/token batch mismatch: "
            f"{len(positive_char_spans)} != {batch_size}"
        )

    if captions is not None and len(captions) != batch_size:
        raise TokenAlignmentError(
            f"captions/token batch mismatch: {len(captions)} != {batch_size}"
        )

    if attention_mask is not None:
        if not torch.is_tensor(attention_mask):
            attention_mask = torch.as_tensor(attention_mask)
        if tuple(attention_mask.shape) != (batch_size, token_count):
            raise TokenAlignmentError(
                "attention_mask must have shape [B, L], got "
                f"{tuple(attention_mask.shape)}, expected "
                f"{(batch_size, token_count)}"
            )
        attention_mask = attention_mask.to(
            device=token_offsets.device,
            dtype=torch.bool,
        )

    masks_per_image: List[torch.Tensor] = []
    maps_per_image: List[torch.Tensor] = []
    counts_per_image: List[torch.Tensor] = []
    target_offsets = [0]

    for image_index in range(batch_size):
        image_target_spans = positive_char_spans[image_index]
        caption = captions[image_index] if captions is not None else None

        target_masks: List[torch.Tensor] = []
        for target_index, target_spans in enumerate(image_target_spans):
            mask = char_spans_to_token_mask(
                token_offsets[image_index],
                target_spans,
                attention_mask=(
                    attention_mask[image_index]
                    if attention_mask is not None
                    else None
                ),
                caption=caption,
                strict=strict,
                context=f"image[{image_index}].target[{target_index}]",
            )
            target_masks.append(mask)

        if target_masks:
            image_mask = torch.stack(target_masks, dim=0)
        else:
            image_mask = torch.zeros(
                (0, token_count),
                dtype=torch.bool,
                device=token_offsets.device,
            )

        image_count = image_mask.sum(dim=1).to(dtype=torch.long)
        image_map = image_mask.to(dtype=torch.float32)

        if normalize and image_map.numel() > 0:
            image_map = image_map / image_map.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0)

        masks_per_image.append(image_mask)
        maps_per_image.append(image_map)
        counts_per_image.append(image_count)
        target_offsets.append(target_offsets[-1] + int(image_mask.shape[0]))

    if target_offsets[-1] > 0:
        flat_mask = torch.cat(masks_per_image, dim=0)
        flat_map = torch.cat(maps_per_image, dim=0)
    else:
        flat_mask = torch.zeros(
            (0, token_count),
            dtype=torch.bool,
            device=token_offsets.device,
        )
        flat_map = torch.zeros(
            (0, token_count),
            dtype=torch.float32,
            device=token_offsets.device,
        )

    return {
        "positive_token_masks": masks_per_image,
        "positive_token_maps": maps_per_image,
        "positive_token_counts": counts_per_image,
        "positive_token_mask_flat": flat_mask,
        "positive_token_map_flat": flat_map,
        "target_offsets": torch.tensor(
            target_offsets,
            dtype=torch.long,
            device=token_offsets.device,
        ),
    }


def attach_positive_token_maps(
    batch: Dict[str, Any],
    model_outputs: Dict[str, Any],
    *,
    strict: bool = True,
    normalize: bool = True,
) -> Dict[str, Any]:
    """Build token maps and attach them to model outputs."""
    captions = batch.get("captions")
    positive_char_spans = batch.get("positive_char_spans")
    token_offsets = model_outputs.get("token_offsets")
    text_mask = model_outputs.get("text_mask")

    missing = []
    if captions is None:
        missing.append("batch.captions")
    if positive_char_spans is None:
        missing.append("batch.positive_char_spans")
    if token_offsets is None:
        missing.append("model_outputs.token_offsets")
    if text_mask is None:
        missing.append("model_outputs.text_mask")
    if missing:
        raise KeyError(
            "Cannot build ODVG positive token maps; missing: "
            + ", ".join(missing)
        )

    alignment = build_positive_token_maps(
        token_offsets=token_offsets,
        positive_char_spans=positive_char_spans,
        attention_mask=text_mask,
        captions=captions,
        strict=strict,
        normalize=normalize,
    )

    model_outputs.update(alignment)
    model_outputs["positive_char_spans"] = positive_char_spans
    model_outputs["captions"] = captions
    return alignment
