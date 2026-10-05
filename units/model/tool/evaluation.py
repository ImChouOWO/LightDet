from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.amp import autocast
from torchvision.ops import batched_nms

from units.model.tool.grounding_metrics import GroundingMetrics

from units.model.tool.runtime import (
    forward_model_batch,
    get_raw_targets,
    make_progress_bar,
    move_targets_to_device,
    prepare_model_batch,
    score_queries_for_char_spans,
)


def get_amp_enabled(
    device: torch.device,
    use_amp: bool = True,
) -> bool:
    return bool(use_amp and device.type == "cuda")


def select_predictions_batch(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    score_thr: float = 0.001,
    top_k: int = 20,
    nms_iou_thr: float = 0.5,
    use_topk_fallback: bool = False,
    use_nms: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Select predictions for each phrase row.

    boxes:  [P,Q,4]
    scores: [P,Q]
    """
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(
            f"boxes must be [P,Q,4], got {tuple(boxes.shape)}"
        )
    if scores.ndim != 2:
        raise ValueError(
            f"scores must be [P,Q], got {tuple(scores.shape)}"
        )
    if boxes.shape[:2] != scores.shape:
        raise ValueError("boxes/scores shape mismatch")

    batch_size, num_queries = scores.shape
    if num_queries == 0 or int(top_k) <= 0:
        return [
            (
                torch.empty((0, 4), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32),
            )
            for _ in range(batch_size)
        ]

    k = min(int(top_k), int(num_queries))
    top_scores, top_indices = torch.topk(
        scores,
        k=k,
        dim=1,
        largest=True,
        sorted=True,
    )
    top_boxes = torch.gather(
        boxes,
        dim=1,
        index=top_indices.unsqueeze(-1).expand(-1, -1, 4),
    )

    valid_mask = top_scores >= float(score_thr)
    if use_topk_fallback:
        no_valid = ~valid_mask.any(dim=1)
        valid_mask[no_valid] = True

    batch_ids = torch.arange(
        batch_size,
        device=boxes.device,
        dtype=torch.long,
    ).unsqueeze(1).expand(batch_size, k)

    flat_boxes = top_boxes[valid_mask]
    flat_scores = top_scores[valid_mask]
    flat_batch_ids = batch_ids[valid_mask]

    if flat_boxes.numel() == 0:
        return [
            (
                torch.empty((0, 4), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32),
            )
            for _ in range(batch_size)
        ]

    if use_nms:
        keep = batched_nms(
            flat_boxes.float(),
            flat_scores.float(),
            flat_batch_ids,
            float(nms_iou_thr),
        )
        flat_boxes = flat_boxes[keep]
        flat_scores = flat_scores[keep]
        flat_batch_ids = flat_batch_ids[keep]

    boxes_cpu = flat_boxes.detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    scores_cpu = flat_scores.detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    batch_ids_cpu = flat_batch_ids.detach().cpu()

    results: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for batch_index in range(batch_size):
        mask = batch_ids_cpu == batch_index
        results.append(
            (
                boxes_cpu[mask],
                scores_cpu[mask],
            )
        )
    return results


@dataclass
class PhraseEvaluationRow:
    image_index: int
    phrase: str
    char_spans: Sequence[Sequence[int]]
    pred_boxes: torch.Tensor
    quality_scores: torch.Tensor
    alignment_scores: torch.Tensor
    final_scores: torch.Tensor
    gt_boxes: torch.Tensor
    ignored_boxes: torch.Tensor


def _normalize_target_indices(
    value: Any,
    *,
    target_count: int,
) -> List[int]:
    if torch.is_tensor(value):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    elif value is None:
        values = []
    else:
        values = [value]

    result: List[int] = []
    seen = set()
    for item in values:
        index = int(item)
        if index < 0 or index >= int(target_count):
            raise IndexError(
                "region_to_target_indices contains out-of-range "
                f"target index {index}/{target_count}"
            )
        if index not in seen:
            seen.add(index)
            result.append(index)
    return result


def build_phrase_evaluation_rows(
    batch: Mapping[str, Any],
    outputs: Mapping[str, Any],
    pred_bbox: torch.Tensor,
    token_reduction: str = "mean",
    score_fusion: str = "geometric_mean",
) -> List[PhraseEvaluationRow]:
    """
    Expand each ODVG image into phrase-conditioned evaluation rows.

    Each region phrase receives its own query scores and its own GT box set.
    """
    quality_logit = outputs.get(
        "quality_logit",
        outputs.get("main_quality_logit"),
    )
    token_logits = outputs.get(
        "token_alignment_logits",
        outputs.get("main_token_alignment_logits"),
    )
    token_offsets = outputs.get("token_offsets")
    alignment_text_mask = outputs.get(
        "alignment_text_mask"
    )

    if not torch.is_tensor(quality_logit):
        raise RuntimeError(
            "Model output does not contain quality_logit"
        )
    if not torch.is_tensor(token_logits):
        raise RuntimeError(
            "Model output does not contain token_alignment_logits"
        )
    if not torch.is_tensor(token_offsets):
        raise RuntimeError(
            "Model output does not contain token_offsets"
        )
    if not torch.is_tensor(alignment_text_mask):
        raise RuntimeError(
            "Model output does not contain alignment_text_mask"
        )

    batch_size = int(pred_bbox.shape[0])
    if quality_logit.shape[:2] != pred_bbox.shape[:2]:
        raise ValueError("quality/bbox shape mismatch")
    if token_logits.shape[:2] != pred_bbox.shape[:2]:
        raise ValueError("token alignment/bbox shape mismatch")
    if token_offsets.shape[:2] != (
        batch_size,
        token_logits.shape[-1],
    ):
        raise ValueError("token offset shape mismatch")
    if alignment_text_mask.shape != (
        batch_size,
        token_logits.shape[-1],
    ):
        raise ValueError("alignment text mask shape mismatch")

    raw_targets = get_raw_targets(dict(batch))
    if len(raw_targets) != batch_size:
        raise ValueError("target/model batch mismatch")

    regions_batch = batch.get("regions")
    mapping_batch = batch.get(
        "region_to_target_indices"
    )
    rows: List[PhraseEvaluationRow] = []

    for image_index in range(batch_size):
        target = raw_targets[image_index]
        target_boxes = torch.as_tensor(
            target["boxes"],
            dtype=torch.float32,
        ).reshape(-1, 4)

        image_regions = (
            regions_batch[image_index]
            if isinstance(regions_batch, Sequence)
            and image_index < len(regions_batch)
            else None
        )
        image_mapping = (
            mapping_batch[image_index]
            if isinstance(mapping_batch, Sequence)
            and image_index < len(mapping_batch)
            else None
        )

        region_rows_created = 0
        if image_regions is not None:
            for region_index, region in enumerate(
                image_regions
            ):
                if not isinstance(region, Mapping):
                    continue
                spans = region.get(
                    "tokens_positive",
                    [],
                )
                if not spans:
                    continue

                if image_mapping is not None:
                    mapping_value = (
                        image_mapping[region_index]
                        if region_index < len(image_mapping)
                        else None
                    )
                    target_indices = _normalize_target_indices(
                        mapping_value,
                        target_count=target_boxes.shape[0],
                    )
                else:
                    target_indices = []

                if image_mapping is None:
                    raise ValueError("Phrase evaluation requires explicit object mapping")

                gt_boxes = target_boxes[
                    target_indices
                ]
                scores = score_queries_for_char_spans(
                    quality_logit=quality_logit[
                        image_index
                    ],
                    token_alignment_logits=token_logits[
                        image_index
                    ],
                    token_offsets=token_offsets[
                        image_index
                    ],
                    char_spans=spans,
                    valid_token_mask=alignment_text_mask[
                        image_index
                    ],
                    token_reduction=token_reduction,
                    score_fusion=score_fusion,
                    strict=True,
                )

                rows.append(
                    PhraseEvaluationRow(
                        image_index=image_index,
                        phrase=str(
                            region.get("phrase", "")
                        ),
                        char_spans=spans,
                        pred_boxes=pred_bbox[image_index],
                        quality_scores=scores[
                            "quality_score"
                        ],
                        alignment_scores=scores[
                            "phrase_alignment_score"
                        ],
                        final_scores=scores[
                            "final_score"
                        ],
                        gt_boxes=gt_boxes,
                        ignored_boxes=target_boxes[
                            torch.as_tensor(target["phrase_relations"])[region_index] < 0
                        ],
                    )
                )
                region_rows_created += 1

        if region_rows_created == 0:
            raise ValueError("Phrase evaluation requires annotated phrase rows")

    return rows


@torch.inference_mode()
def validate_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    val_loader: Any,
    device: torch.device,
    epoch: int,
    compute_loss: bool,
    compute_metrics: bool,
    total_epochs: Optional[int] = None,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    channels_last: bool = False,
    lambda_bbox: float = 5.0,
    lambda_giou: float = 2.0,
    lambda_score: float = 1.0,
    lambda_rank: float = 0.0,
    pos_weight: float = 1.0,
    quality_warmup_epoch: int = 20,
    rank_start_epoch: int = 15,
    rank_warmup_epoch: int = 30,
    rank_alpha_min: float = 1e-4,
    score_thr: float = 0.5,
    nms_iou_thr: float = 0.5,
    use_nms: bool = False,
    max_val_batches: Optional[int] = None,
    log_interval: int = 50,
    progress_leave: bool = True,
    progress_mininterval: float = 0.5,
    phrase_token_reduction: str = "mean",
    phrase_score_fusion: str = "geometric_mean",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not compute_loss and not compute_metrics:
        return {}, {}

    model.eval()
    amp_enabled = get_amp_enabled(
        device,
        use_amp,
    )

    loss_sums: Dict[str, torch.Tensor] = {}
    processed_batches = 0

    metric = GroundingMetrics() if compute_metrics else None

    phrase_count = 0
    image_count = 0
    total_text_negative_queries = 0
    total_selected = 0

    pbar_total = (
        len(val_loader)
        if max_val_batches is None
        else min(
            len(val_loader),
            int(max_val_batches),
        )
    )
    labels: List[str] = []
    if compute_loss:
        labels.append("Loss")
    if compute_metrics:
        labels.append("ODVG Eval")

    pbar = make_progress_bar(
        enumerate(val_loader),
        total=pbar_total,
        desc=f"Epoch {epoch} [Val {'+'.join(labels)}]",
        leave=progress_leave,
        mininterval=progress_mininterval,
    )

    validation_start = time.perf_counter()

    for step, batch in pbar:
        if (
            max_val_batches is not None
            and step >= int(max_val_batches)
        ):
            break

        processed_batches += 1
        images, captions, image_indices = (
            prepare_model_batch(
                batch=batch,
                device=device,
                channels_last=channels_last,
            )
        )
        targets_device = (
            move_targets_to_device(batch, device)
            if compute_loss
            else None
        )
        if compute_loss:
            negative_mask_value = batch.get("text_negative_mask")
            if negative_mask_value is not None:
                total_text_negative_queries += int(
                    torch.as_tensor(
                        negative_mask_value,
                        dtype=torch.bool,
                    ).sum().item()
                )

        with autocast(
            device_type=device.type,
            enabled=amp_enabled,
            dtype=(
                amp_dtype
                if amp_enabled
                else None
            ),
        ):
            outputs = forward_model_batch(
                model=model,
                images=images,
                query_texts=captions,
                image_indices=image_indices,
                return_aux=False,
            )
            pred_bbox = outputs["bbox"]
            quality_logit = outputs.get(
                "quality_logit",
                outputs.get("main_quality_logit"),
            )
            token_logits = outputs.get(
                "token_alignment_logits",
                outputs.get(
                    "main_token_alignment_logits"
                ),
            )

            if not torch.is_tensor(quality_logit):
                raise RuntimeError(
                    "Model output does not contain quality_logit"
                )
            if not torch.is_tensor(token_logits):
                raise RuntimeError(
                    "Model output does not contain "
                    "token_alignment_logits"
                )

            if compute_loss:
                loss, loss_dict = criterion(
                    pred_bbox=pred_bbox,
                    pred_score_logit=quality_logit,
                    pred_quality_logit=quality_logit,
                    pred_text_alignment_logit=token_logits,
                    pred_token_alignment_logit=token_logits,
                    positive_token_maps=None,
                    alignment_text_mask=outputs[
                        "alignment_text_mask"
                    ],
                    token_offsets=outputs[
                        "token_offsets"
                    ],
                    captions=captions,
                    targets=targets_device,
                    lambda_bbox=lambda_bbox,
                    lambda_giou=lambda_giou,
                    lambda_score=lambda_score,
                    lambda_rank=lambda_rank,
                    pos_weight=pos_weight,
                    current_epoch=epoch,
                    total_epochs=total_epochs,
                    quality_warmup_epoch=(
                        quality_warmup_epoch
                    ),
                    rank_start_epoch=rank_start_epoch,
                    rank_warmup_epoch=rank_warmup_epoch,
                    rank_alpha_min=rank_alpha_min,
                    query_loss_weights=batch.get(
                        "query_loss_weights"
                    ),
                    text_negative_mask=batch.get(
                        "text_negative_mask"
                    ),
                )

        if compute_loss:
            values = {
                "loss": loss,
                "loss_bbox": loss_dict.get(
                    "loss_bbox",
                    pred_bbox.new_zeros(()),
                ),
                "loss_giou": loss_dict.get(
                    "loss_giou",
                    pred_bbox.new_zeros(()),
                ),
                "loss_score": loss_dict.get(
                    "loss_score",
                    pred_bbox.new_zeros(()),
                ),
                "loss_rank_raw": loss_dict.get(
                    "loss_rank_raw",
                    pred_bbox.new_zeros(()),
                ),
                "loss_rank": loss_dict.get(
                    "loss_rank_contrib",
                    loss_dict.get(
                        "loss_rank",
                        pred_bbox.new_zeros(()),
                    ),
                ),
                "loss_text_alignment": loss_dict.get(
                    "loss_text_alignment",
                    pred_bbox.new_zeros(()),
                ),
                "loss_text_alignment_contrib": (
                    loss_dict.get(
                        "loss_text_alignment_contrib",
                        pred_bbox.new_zeros(()),
                    )
                ),
                "loss_text_negative": loss_dict.get(
                    "loss_text_negative",
                    pred_bbox.new_zeros(()),
                ),
                "loss_text_negative_contrib": (
                    loss_dict.get(
                        "loss_text_negative_contrib",
                        pred_bbox.new_zeros(()),
                    )
                ),
            }
            for key, value in values.items():
                value = torch.as_tensor(
                    value,
                    device=device,
                ).detach()
                if key not in loss_sums:
                    loss_sums[key] = torch.zeros(
                        (),
                        device=device,
                    )
                loss_sums[key].add_(value)

            last_loss_dict = loss_dict
        else:
            last_loss_dict = {}

        if compute_metrics and metric is not None:
            rows = build_phrase_evaluation_rows(
                batch=batch,
                outputs=outputs,
                pred_bbox=pred_bbox.detach(),
                token_reduction=(
                    phrase_token_reduction
                ),
                score_fusion=(
                    phrase_score_fusion
                ),
            )
            image_count += int(pred_bbox.shape[0])
            phrase_count += len(rows)

            if rows:
                phrase_boxes = torch.stack(
                    [
                        row.pred_boxes
                        for row in rows
                    ],
                    dim=0,
                )
                final_scores = torch.stack(
                    [
                        row.final_scores
                        for row in rows
                    ],
                    dim=0,
                )
                selected_rows = select_predictions_batch(
                    boxes=phrase_boxes,
                    scores=final_scores,
                    score_thr=score_thr,
                    top_k=phrase_boxes.shape[1],
                    nms_iou_thr=nms_iou_thr,
                    use_topk_fallback=False,
                    use_nms=use_nms,
                )

                for row, (
                    selected_boxes,
                    selected_scores,
                ) in zip(rows, selected_rows):
                    total_selected += int(selected_boxes.shape[0])
                    metric.update(
                        pred_boxes=row.pred_boxes,
                        final_scores=row.final_scores,
                        quality_scores=row.quality_scores,
                        alignment_scores=row.alignment_scores,
                        gt_boxes=row.gt_boxes,
                        ignored_boxes=row.ignored_boxes,
                        selected_boxes=selected_boxes,
                    )

        should_log = (
            (step + 1)
            % max(1, int(log_interval))
            == 0
            or (step + 1) == pbar_total
        )
        if should_log:
            postfix: Dict[str, Any] = {}
            if compute_loss and processed_batches > 0:
                postfix["loss"] = (
                    f"{float((loss_sums['loss'] / processed_batches).item()):.4f}"
                )
            if compute_metrics:
                postfix.update({
                    "phrases": phrase_count,
                    "selected": total_selected,
                })
            pbar.set_postfix(postfix)

    pbar.refresh()
    pbar.close()
    validation_loop_time = (
        time.perf_counter()
        - validation_start
    )

    val_loss_metrics: Dict[str, float] = {}
    if compute_loss:
        denominator = max(1, processed_batches)

        def average(key: str) -> float:
            value = loss_sums.get(
                key,
                torch.zeros((), device=device),
            )
            return float(
                (value / denominator).item()
            )

        val_loss_metrics = {
            "val_loss": average("loss"),
            "val_loss_bbox": average("loss_bbox"),
            "val_loss_giou": average("loss_giou"),
            "val_loss_score": average("loss_score"),
            "val_loss_rank_raw": average(
                "loss_rank_raw"
            ),
            "val_loss_rank": average("loss_rank"),
            "val_loss_text_alignment": average(
                "loss_text_alignment"
            ),
            "val_loss_text_alignment_contrib": (
                average(
                    "loss_text_alignment_contrib"
                )
            ),
            "val_loss_text_negative": average(
                "loss_text_negative"
            ),
            "val_loss_text_negative_contrib": (
                average(
                    "loss_text_negative_contrib"
                )
            ),
            "val_text_negative_queries": int(
                total_text_negative_queries
            ),
            "val_score_negative_iou_ignore_thr": float(
                last_loss_dict.get(
                    "score_negative_iou_ignore_thr",
                    getattr(
                        criterion,
                        "score_negative_iou_ignore_thr",
                        0.0,
                    ),
                )
            ),
            "val_duplicate_suppression_enabled": bool(
                last_loss_dict.get(
                    "duplicate_suppression_enabled",
                    getattr(
                        criterion,
                        "duplicate_suppression_enabled",
                        False,
                    ),
                )
            ),
            "val_hard_negative_mining_enabled": bool(
                last_loss_dict.get(
                    "hard_negative_mining_enabled",
                    getattr(
                        criterion,
                        "hard_negative_mining_enabled",
                        False,
                    ),
                )
            ),
            "val_matcher_score_alpha": float(
                last_loss_dict.get(
                    "matcher_score_alpha",
                    0.0,
                )
            ),
            "val_matcher_alignment_alpha": float(
                last_loss_dict.get(
                    "matcher_alignment_alpha",
                    0.0,
                )
            ),
            "val_matcher_cost_score_effective": float(
                last_loss_dict.get(
                    "matcher_cost_score_effective",
                    0.0,
                )
            ),
            "val_matcher_cost_alignment_effective": float(
                last_loss_dict.get(
                    "matcher_cost_alignment_effective",
                    0.0,
                )
            ),
        }

    eval_metrics: Dict[str, float] = {}
    if compute_metrics and metric is not None:
        eval_metrics.update(metric.compute())
        eval_metrics.update({
            "valid_samples": int(phrase_count),
            "valid_images": int(image_count),
            "odvg_phrase_rows": int(phrase_count),
            "set_score_threshold": float(score_thr),
            "set_use_nms": bool(use_nms),
            "set_nms_iou_threshold": float(nms_iou_thr),
            "avg_selected_per_sample": (
                total_selected
                / max(1, phrase_count)
            ),
            "eval_loop_time": float(
                validation_loop_time
            ),
            "phrase_token_reduction": str(
                phrase_token_reduction
            ),
            "phrase_score_fusion": str(
                phrase_score_fusion
            ),
            "evaluation_mode": (
                "odvg_phrase_grounding"
            ),
        })

    return val_loss_metrics, eval_metrics


__all__ = [
    "PhraseEvaluationRow",
    "build_phrase_evaluation_rows",
    "get_amp_enabled",
    "select_predictions_batch",
    "validate_one_epoch",
]
