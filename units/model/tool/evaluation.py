from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.amp import autocast
from torchvision.ops import batched_nms
from tqdm import tqdm

from units.model.tool.runtime import (
    box_iou_xyxy,
    forward_model_batch,
    get_raw_targets,
    get_target_boxes_cpu,
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


class BinaryDetectionAPAccumulator:
    """Global AP accumulator for phrase-conditioned binary detection."""

    def __init__(
        self,
        iou_thresholds: Optional[Sequence[float]] = None,
    ) -> None:
        if iou_thresholds is None:
            iou_thresholds = [
                round(value, 2)
                for value in np.arange(0.50, 0.96, 0.05)
            ]
        self.iou_thresholds = torch.as_tensor(
            list(iou_thresholds),
            dtype=torch.float32,
        )
        if self.iou_thresholds.numel() == 0:
            raise ValueError("iou_thresholds must not be empty")

        self.score_chunks: List[torch.Tensor] = []
        self.tp_chunks: List[torch.Tensor] = []
        self.num_gt = 0
        self.num_pred = 0

    def update(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> None:
        pred_boxes = pred_boxes.float().reshape(-1, 4)
        pred_scores = pred_scores.float().reshape(-1)
        gt_boxes = gt_boxes.float().reshape(-1, 4)

        num_pred = int(pred_boxes.shape[0])
        num_gt = int(gt_boxes.shape[0])
        num_thresholds = int(self.iou_thresholds.numel())

        if pred_scores.numel() != num_pred:
            raise ValueError("pred_boxes/pred_scores size mismatch")

        self.num_gt += num_gt
        self.num_pred += num_pred
        if num_pred == 0:
            return

        order = torch.argsort(
            pred_scores,
            descending=True,
            stable=True,
        )
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]

        tp_matrix = torch.zeros(
            (num_pred, num_thresholds),
            dtype=torch.bool,
            device=pred_boxes.device,
        )

        if num_gt > 0:
            if gt_boxes.device != pred_boxes.device:
                gt_boxes = gt_boxes.to(pred_boxes.device)

            iou_matrix = box_iou_xyxy(
                pred_boxes,
                gt_boxes,
            )
            thresholds = self.iou_thresholds.to(
                device=pred_boxes.device,
                dtype=iou_matrix.dtype,
            )
            matched_gt = torch.zeros(
                (num_thresholds, num_gt),
                dtype=torch.bool,
                device=pred_boxes.device,
            )
            threshold_indices = torch.arange(
                num_thresholds,
                device=pred_boxes.device,
            )

            for prediction_index in range(num_pred):
                candidate = (
                    iou_matrix[prediction_index]
                    .unsqueeze(0)
                    .expand(num_thresholds, -1)
                    .masked_fill(matched_gt, -1.0)
                )
                best_iou, best_gt = candidate.max(dim=1)
                is_tp = best_iou >= thresholds
                tp_matrix[prediction_index] = is_tp
                matched_gt[
                    threshold_indices[is_tp],
                    best_gt[is_tp],
                ] = True

        self.score_chunks.append(
            pred_scores.detach().cpu().contiguous()
        )
        self.tp_chunks.append(
            tp_matrix.detach().cpu().contiguous()
        )

    def compute(self) -> Dict[str, float]:
        if self.num_gt == 0:
            return {
                "map50": 0.0,
                "map50_95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "tp": 0,
                "fp": int(self.num_pred),
                "num_gt": 0,
                "num_pred": int(self.num_pred),
            }

        if not self.score_chunks:
            return {
                "map50": 0.0,
                "map50_95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "tp": 0,
                "fp": 0,
                "num_gt": int(self.num_gt),
                "num_pred": 0,
            }

        all_scores = torch.cat(self.score_chunks, dim=0)
        all_tp = torch.cat(
            self.tp_chunks,
            dim=0,
        ).to(torch.float32)

        order = torch.argsort(
            all_scores,
            descending=True,
            stable=True,
        )
        all_tp = all_tp[order]
        all_fp = 1.0 - all_tp

        cumulative_tp = torch.cumsum(all_tp, dim=0)
        cumulative_fp = torch.cumsum(all_fp, dim=0)
        recall = cumulative_tp / max(1, self.num_gt)
        precision = cumulative_tp / torch.clamp(
            cumulative_tp + cumulative_fp,
            min=1e-7,
        )

        precision_envelope = torch.flip(
            torch.cummax(
                torch.flip(precision, dims=[0]),
                dim=0,
            ).values,
            dims=[0],
        )
        previous_recall = torch.cat(
            [
                torch.zeros(
                    (1, recall.shape[1]),
                    dtype=recall.dtype,
                ),
                recall[:-1],
            ],
            dim=0,
        )
        ap_per_threshold = torch.sum(
            (recall - previous_recall)
            * precision_envelope,
            dim=0,
        )

        threshold_50_index = int(
            torch.argmin(
                torch.abs(self.iou_thresholds - 0.50)
            ).item()
        )
        final_tp = cumulative_tp[-1]
        final_fp = cumulative_fp[-1]
        tp50 = int(final_tp[threshold_50_index].item())
        fp50 = int(final_fp[threshold_50_index].item())

        return {
            "map50": float(
                ap_per_threshold[threshold_50_index].item()
            ),
            "map50_95": float(
                ap_per_threshold.mean().item()
            ),
            "precision": tp50 / max(1, tp50 + fp50),
            "recall": tp50 / max(1, self.num_gt),
            "tp": tp50,
            "fp": fp50,
            "num_gt": int(self.num_gt),
            "num_pred": int(self.num_pred),
        }


class RankedRecallAtKAccumulator:
    """Phrase-conditioned Recall@K at one IoU threshold."""

    def __init__(
        self,
        ks: Sequence[int] = (1, 5, 10),
        iou_threshold: float = 0.50,
    ) -> None:
        self.ks = tuple(
            sorted({max(1, int(value)) for value in ks})
        )
        self.iou_threshold = float(iou_threshold)
        self.num_gt = 0
        self.hits = {value: 0 for value in self.ks}

    def update(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> None:
        pred_boxes = pred_boxes.detach().float().reshape(-1, 4).cpu()
        pred_scores = pred_scores.detach().float().reshape(-1).cpu()
        gt_boxes = gt_boxes.detach().float().reshape(-1, 4).cpu()

        num_gt = int(gt_boxes.shape[0])
        self.num_gt += num_gt
        if num_gt == 0 or pred_boxes.shape[0] == 0:
            return

        order = torch.argsort(
            pred_scores,
            descending=True,
            stable=True,
        )
        iou_matrix = box_iou_xyxy(
            pred_boxes[order],
            gt_boxes,
        )

        for top_k in self.ks:
            matched_gt = torch.zeros(
                (num_gt,),
                dtype=torch.bool,
            )
            hit_count = 0
            for prediction_index in range(
                min(top_k, iou_matrix.shape[0])
            ):
                candidate = iou_matrix[
                    prediction_index
                ].masked_fill(matched_gt, -1.0)
                best_iou, best_gt = candidate.max(dim=0)
                if float(best_iou.item()) >= self.iou_threshold:
                    matched_gt[best_gt] = True
                    hit_count += 1
            self.hits[top_k] += hit_count

    def compute(self, prefix: str) -> Dict[str, float]:
        denominator = max(1, self.num_gt)
        return {
            f"{prefix}_recall_iou": self.iou_threshold,
            **{
                f"{prefix}_recall50_at_{top_k}": (
                    self.hits[top_k] / denominator
                )
                for top_k in self.ks
            },
        }


class RawOracleRecallAccumulator:
    """Raw-box coverage upper bound for each phrase GT set."""

    def __init__(
        self,
        iou_thresholds: Sequence[float] = (
            0.25,
            0.50,
            0.75,
        ),
    ) -> None:
        self.iou_thresholds = torch.as_tensor(
            list(iou_thresholds),
            dtype=torch.float32,
        )
        if self.iou_thresholds.numel() == 0:
            raise ValueError(
                "raw_oracle_iou_thresholds must not be empty"
            )
        self.best_iou_chunks: List[torch.Tensor] = []
        self.num_gt = 0
        self.num_samples = 0
        self.num_positive_samples = 0
        self.num_raw_pred = 0
        self.num_raw_pred_positive = 0

    @staticmethod
    def threshold_key(threshold: float) -> str:
        return (
            "raw_oracle_recall"
            f"{int(round(float(threshold) * 100)):02d}"
        )

    def update(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> None:
        pred_boxes = (
            pred_boxes.detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(-1, 4)
            .contiguous()
        )
        gt_boxes = (
            gt_boxes.detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(-1, 4)
            .contiguous()
        )

        num_pred = int(pred_boxes.shape[0])
        num_gt = int(gt_boxes.shape[0])
        self.num_samples += 1
        self.num_raw_pred += num_pred
        self.num_gt += num_gt

        if num_gt == 0:
            return

        self.num_positive_samples += 1
        self.num_raw_pred_positive += num_pred

        if num_pred == 0:
            best_iou = torch.zeros(
                num_gt,
                dtype=torch.float32,
            )
        else:
            best_iou = box_iou_xyxy(
                pred_boxes,
                gt_boxes,
            ).max(dim=0).values
        self.best_iou_chunks.append(best_iou)

    def compute(self) -> Dict[str, float]:
        result: Dict[str, float] = {
            "raw_oracle_num_gt": int(self.num_gt),
            "raw_oracle_num_samples": int(self.num_samples),
            "raw_oracle_positive_samples": int(
                self.num_positive_samples
            ),
            "raw_oracle_num_pred": int(self.num_raw_pred),
            "raw_oracle_avg_pred_per_sample": (
                self.num_raw_pred / max(1, self.num_samples)
            ),
            "raw_oracle_avg_pred_per_positive_sample": (
                self.num_raw_pred_positive
                / max(1, self.num_positive_samples)
            ),
        }

        if self.num_gt == 0 or not self.best_iou_chunks:
            result.update({
                "raw_best_iou_mean": 0.0,
                "raw_best_iou_median": 0.0,
                "raw_best_iou_p25": 0.0,
                "raw_best_iou_p75": 0.0,
            })
            for threshold in self.iou_thresholds.tolist():
                result[self.threshold_key(threshold)] = 0.0
            return result

        best_iou = torch.cat(
            self.best_iou_chunks,
            dim=0,
        )
        result.update({
            "raw_best_iou_mean": float(best_iou.mean().item()),
            "raw_best_iou_median": float(
                best_iou.median().item()
            ),
            "raw_best_iou_p25": float(
                torch.quantile(best_iou, 0.25).item()
            ),
            "raw_best_iou_p75": float(
                torch.quantile(best_iou, 0.75).item()
            ),
        })
        for threshold in self.iou_thresholds.tolist():
            result[self.threshold_key(threshold)] = float(
                (best_iou >= float(threshold))
                .to(torch.float32)
                .mean()
                .item()
            )
        return result


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


def _flatten_target_spans(
    target: Mapping[str, Any],
) -> List[List[int]]:
    result: List[List[int]] = []
    for target_spans in target.get(
        "positive_char_spans",
        [],
    ):
        for span in target_spans:
            if len(span) == 2:
                result.append(
                    [int(span[0]), int(span[1])]
                )
    return result


def build_phrase_evaluation_rows(
    *,
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

                if not target_indices:
                    # Structural fallback for older ODVG collates.
                    target_indices = list(
                        range(target_boxes.shape[0])
                    )

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
                    )
                )
                region_rows_created += 1

        if region_rows_created > 0:
            continue

        # Fallback: one union-phrase row per image.
        spans = _flatten_target_spans(target)
        if spans:
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
                        target.get("caption", "")
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
                    gt_boxes=target_boxes,
                )
            )

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
    score_thr: float = 0.001,
    top_k: int = 20,
    nms_iou_thr: float = 0.5,
    use_topk_fallback: bool = False,
    use_nms: bool = True,
    iou_thresholds: Optional[Sequence[float]] = None,
    compute_raw_oracle: bool = True,
    raw_oracle_iou_thresholds: Optional[
        Sequence[float]
    ] = (0.25, 0.50, 0.75),
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

    metric = (
        BinaryDetectionAPAccumulator(
            iou_thresholds=iou_thresholds
        )
        if compute_metrics
        else None
    )
    raw_oracle_metric = (
        RawOracleRecallAccumulator(
            iou_thresholds=(
                raw_oracle_iou_thresholds
                if raw_oracle_iou_thresholds is not None
                else (0.25, 0.50, 0.75)
            )
        )
        if compute_metrics and compute_raw_oracle
        else None
    )
    quality_recall_metric = (
        RankedRecallAtKAccumulator()
        if compute_metrics
        else None
    )
    alignment_recall_metric = (
        RankedRecallAtKAccumulator()
        if compute_metrics
        else None
    )
    final_recall_metric = (
        RankedRecallAtKAccumulator()
        if compute_metrics
        else None
    )

    phrase_count = 0
    image_count = 0
    total_text_negative_queries = 0
    skipped_empty_gt = 0
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
                    top_k=top_k,
                    nms_iou_thr=nms_iou_thr,
                    use_topk_fallback=(
                        use_topk_fallback
                    ),
                    use_nms=use_nms,
                )

                for row, (
                    selected_boxes,
                    selected_scores,
                ) in zip(rows, selected_rows):
                    gt_boxes = (
                        row.gt_boxes.detach()
                        .to(
                            device="cpu",
                            dtype=torch.float32,
                        )
                        .reshape(-1, 4)
                    )
                    if gt_boxes.numel() == 0:
                        skipped_empty_gt += 1

                    total_selected += int(
                        selected_boxes.shape[0]
                    )

                    metric.update(
                        pred_boxes=selected_boxes,
                        pred_scores=selected_scores,
                        gt_boxes=gt_boxes,
                    )

                    quality_recall_metric.update(
                        pred_boxes=row.pred_boxes,
                        pred_scores=(
                            row.quality_scores
                        ),
                        gt_boxes=gt_boxes,
                    )
                    alignment_recall_metric.update(
                        pred_boxes=row.pred_boxes,
                        pred_scores=(
                            row.alignment_scores
                        ),
                        gt_boxes=gt_boxes,
                    )
                    final_recall_metric.update(
                        pred_boxes=row.pred_boxes,
                        pred_scores=row.final_scores,
                        gt_boxes=gt_boxes,
                    )

                    if raw_oracle_metric is not None:
                        raw_oracle_metric.update(
                            pred_boxes=row.pred_boxes,
                            gt_boxes=gt_boxes,
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
        eval_metrics.update(
            quality_recall_metric.compute(
                "quality"
            )
        )
        eval_metrics.update(
            alignment_recall_metric.compute(
                "text"
            )
        )
        final_recall = (
            final_recall_metric.compute("final")
        )
        eval_metrics.update(final_recall)

        for top_k in final_recall_metric.ks:
            eval_metrics[f"recall50_at_{top_k}"] = (
                final_recall[
                    f"final_recall50_at_{top_k}"
                ]
            )

        if raw_oracle_metric is not None:
            raw_metrics = raw_oracle_metric.compute()
            eval_metrics.update(raw_metrics)
            raw_recall50 = float(
                raw_metrics.get(
                    "raw_oracle_recall50",
                    0.0,
                )
            )
            final_recall50 = float(
                eval_metrics.get(
                    "recall",
                    0.0,
                )
            )
            eval_metrics["raw_oracle_gap50"] = (
                raw_recall50 - final_recall50
            )
            eval_metrics[
                "raw_oracle_retention50"
            ] = (
                final_recall50
                / max(raw_recall50, 1e-12)
            )

        eval_metrics.update({
            "valid_samples": int(phrase_count),
            "valid_images": int(image_count),
            "odvg_phrase_rows": int(phrase_count),
            "skipped_empty_gt": int(
                skipped_empty_gt
            ),
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
    "BinaryDetectionAPAccumulator",
    "PhraseEvaluationRow",
    "RankedRecallAtKAccumulator",
    "RawOracleRecallAccumulator",
    "build_phrase_evaluation_rows",
    "get_amp_enabled",
    "select_predictions_batch",
    "validate_one_epoch",
]
