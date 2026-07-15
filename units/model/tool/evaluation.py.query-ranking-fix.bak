from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.amp import autocast
from torchvision.ops import batched_nms
from tqdm import tqdm

from units.model.tool.runtime import (
    box_iou_xyxy,
    forward_model_batch,
    get_score_logit,
    get_target_boxes_cpu,
    make_progress_bar,
    move_targets_to_device,
    prepare_model_batch,
)


def get_amp_enabled(device: torch.device, use_amp: bool = True) -> bool:
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
    在 GPU 上批次完成 top-k、threshold 與 batched NMS；每個 batch 僅集中搬一次 CPU。

    boxes:  [B, N, 4], normalized xyxy
    scores: [B, N], sigmoid scores
    """

    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must be [B, N, 4], got {tuple(boxes.shape)}")

    if scores.ndim != 2:
        raise ValueError(f"scores must be [B, N], got {tuple(scores.shape)}")

    batch_size, num_queries = scores.shape

    if num_queries == 0 or top_k <= 0:
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

    # 非同步 copy 在 pinned destination 上才真正有利；這裡一次性 copy 已避免逐框同步。
    boxes_cpu = flat_boxes.detach().to(dtype=torch.float32, device="cpu")
    scores_cpu = flat_scores.detach().to(dtype=torch.float32, device="cpu")
    batch_ids_cpu = flat_batch_ids.detach().to(device="cpu")

    results: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for batch_index in range(batch_size):
        mask = batch_ids_cpu == batch_index
        boxes_i = boxes_cpu[mask]
        scores_i = scores_cpu[mask]

        # topk 與 batched_nms 的輸出已依 score 由高到低排列；
        # 依 batch mask 取出後仍保留該 sample 的相對順序。
        results.append((boxes_i, scores_i))

    return results

class BinaryDetectionAPAccumulator:
    """
    保留原本 query-conditioned binary detection 的 matching 定義，但：
      1. 每個 sample 僅計算一次 IoU matrix。
      2. 所有 IoU thresholds 同時計算。
      3. 所有 prediction 只做一次全域 score 排序。
      4. 不建立逐框 Python dict。
    """

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
            iou_thresholds,
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
            raise ValueError(
                "pred_boxes/pred_scores size mismatch: "
                f"{num_pred} != {pred_scores.numel()}"
            )

        self.num_gt += num_gt
        self.num_pred += num_pred

        if num_pred == 0:
            return

        # AP matching 必須依 confidence 由高到低逐一處理 prediction。
        # 即使上游 select_predictions_batch 已排序，這裡仍再次排序，
        # 避免未來由其他呼叫路徑傳入未排序 prediction 時造成評估偏差。
        score_order = torch.argsort(
            pred_scores,
            descending=True,
            stable=True,
        )
        pred_boxes = pred_boxes[score_order]
        pred_scores = pred_scores[score_order]

        metric_device = pred_boxes.device

        if gt_boxes.device != metric_device:
            gt_boxes = gt_boxes.to(metric_device)

        # 每一列對應一個 prediction，每一欄對應一個 IoU threshold。
        tp_matrix = torch.zeros(
            (num_pred, num_thresholds),
            dtype=torch.bool,
            device=metric_device,
        )

        if num_gt > 0:
            iou_matrix = box_iou_xyxy(
                pred_boxes,
                gt_boxes,
            )

            iou_thresholds = self.iou_thresholds.to(
                device=metric_device,
                dtype=iou_matrix.dtype,
            )

            # 每個 IoU threshold 都必須維護獨立的 GT 配對狀態。
            # shape: [num_thresholds, num_gt]
            matched_gt = torch.zeros(
                (num_thresholds, num_gt),
                dtype=torch.bool,
                device=metric_device,
            )

            threshold_indices = torch.arange(
                num_thresholds,
                device=metric_device,
            )

            # prediction 已依 confidence 由高到低排列。
            for prediction_index in range(num_pred):
                # 同一個 prediction 在每個 IoU threshold 下，都只可從
                # 尚未配對的 GT 中選擇 IoU 最高者。
                candidate_ious = (
                    iou_matrix[prediction_index]
                    .unsqueeze(0)
                    .expand(num_thresholds, -1)
                    .masked_fill(matched_gt, -1.0)
                )

                best_iou, best_gt_index = candidate_ious.max(dim=1)
                is_true_positive = best_iou >= iou_thresholds

                tp_matrix[prediction_index] = is_true_positive

                # 僅更新本次成功配對的 threshold/GT 組合。
                valid_threshold_indices = threshold_indices[
                    is_true_positive
                ]
                valid_gt_indices = best_gt_index[
                    is_true_positive
                ]

                matched_gt[
                    valid_threshold_indices,
                    valid_gt_indices,
                ] = True

        # 累積資料固定搬回 CPU，避免跨 batch 保留 GPU tensor，
        # 並確保 compute() 的全域排序與累積運算裝置一致。
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
        all_tp = torch.cat(self.tp_chunks, dim=0).to(torch.float32)

        global_order = torch.argsort(
            all_scores,
            descending=True,
            stable=True,
        )

        all_tp = all_tp[global_order]
        all_fp = 1.0 - all_tp

        cumulative_tp = torch.cumsum(all_tp, dim=0)
        cumulative_fp = torch.cumsum(all_fp, dim=0)

        recall = cumulative_tp / max(1, self.num_gt)
        precision = cumulative_tp / torch.clamp(
            cumulative_tp + cumulative_fp,
            min=1e-7,
        )

        # Precision envelope，等價於原本由後往前逐點 max。
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

        recall_delta = recall - previous_recall
        ap_per_threshold = torch.sum(
            recall_delta * precision_envelope,
            dim=0,
        )

        final_tp = cumulative_tp[-1]
        final_fp = cumulative_fp[-1]

        # 找最接近 0.50 的 threshold，避免自訂 threshold 順序造成錯誤。
        threshold_50_index = int(
            torch.argmin(torch.abs(self.iou_thresholds - 0.50)).item()
        )

        tp50 = int(final_tp[threshold_50_index].item())
        fp50 = int(final_fp[threshold_50_index].item())

        return {
            "map50": float(ap_per_threshold[threshold_50_index].item()),
            "map50_95": float(ap_per_threshold.mean().item()),
            "precision": tp50 / max(1, tp50 + fp50),
            "recall": tp50 / max(1, self.num_gt),
            "tp": tp50,
            "fp": fp50,
            "num_gt": int(self.num_gt),
            "num_pred": int(self.num_pred),
        }

class RawOracleRecallAccumulator:
    """
    使用全部 raw prediction 計算候選框覆蓋能力。

    對每一個 GT，從該 query 的所有原始預測框中取最大 IoU：
        best_iou(gt) = max IoU(raw_prediction, gt)

    Raw Oracle Recall@t：
        best_iou >= t 的 GT 數 / 全部 GT 數

    此指標不套用 confidence threshold、Top-K 或 NMS，並允許同一個
    prediction 成為多個 GT 的最佳候選。因此它是偏樂觀的候選框覆蓋
    上限，只用於診斷定位能力，不能取代正式 mAP、Precision 或 Recall。
    """

    def __init__(
        self,
        iou_thresholds: Optional[Sequence[float]] = None,
    ) -> None:
        if iou_thresholds is None:
            iou_thresholds = (0.25, 0.50, 0.75)

        self.iou_thresholds = torch.as_tensor(
            list(iou_thresholds),
            dtype=torch.float32,
        )

        if self.iou_thresholds.numel() == 0:
            raise ValueError(
                "raw_oracle_iou_thresholds must not be empty"
            )

        if bool(
            ((self.iou_thresholds < 0.0) | (self.iou_thresholds > 1.0)).any()
        ):
            raise ValueError(
                "raw_oracle_iou_thresholds must be within [0, 1]"
            )

        self.best_iou_chunks: List[torch.Tensor] = []
        self.num_gt = 0
        self.num_samples = 0
        self.num_positive_samples = 0
        self.num_raw_pred = 0
        self.num_raw_pred_positive = 0

    @staticmethod
    def threshold_key(threshold: float) -> str:
        return f"raw_oracle_recall{int(round(float(threshold) * 100)):02d}"

    def update(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> None:
        # Raw Oracle 只需要 bbox，不使用 score。固定搬到 CPU，避免跨 batch
        # 保留 GPU tensor，並讓獨立 validate.py 與 train.py 的結果一致。
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

        # 負文字 query 沒有 GT，不參與 Raw Oracle Recall，但仍保留於
        # 正式 AP/Precision 的 FP 計算。
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
            iou_matrix = box_iou_xyxy(
                pred_boxes,
                gt_boxes,
            )
            # 每一欄是一個 GT，取所有 raw prediction 中的最大 IoU。
            best_iou = iou_matrix.max(dim=0).values

        self.best_iou_chunks.append(
            best_iou.detach().cpu().contiguous()
        )

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

        if int(best_iou.numel()) != int(self.num_gt):
            raise RuntimeError(
                "Raw Oracle GT count mismatch: "
                f"best_iou={best_iou.numel()}, num_gt={self.num_gt}"
            )

        result.update({
            "raw_best_iou_mean": float(best_iou.mean().item()),
            "raw_best_iou_median": float(best_iou.median().item()),
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
    raw_oracle_iou_thresholds: Optional[Sequence[float]] = (
        0.25,
        0.50,
        0.75,
    ),
    max_val_batches: Optional[int] = None,
    log_interval: int = 50,
    progress_leave: bool = True,
    progress_mininterval: float = 0.5,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not compute_loss and not compute_metrics:
        return {}, {}

    model.eval()
    amp_enabled = get_amp_enabled(device, use_amp)

    total_loss_sum = torch.zeros((), device=device)
    total_bbox_sum = torch.zeros((), device=device)
    total_giou_sum = torch.zeros((), device=device)
    total_score_sum = torch.zeros((), device=device)
    total_rank_contrib_sum = torch.zeros((), device=device)
    total_rank_raw_sum = torch.zeros((), device=device)
    total_text_negative_sum = torch.zeros((), device=device)
    total_text_negative_contrib_sum = torch.zeros((), device=device)
    total_text_negative_queries = 0

    metric = (
        BinaryDetectionAPAccumulator(iou_thresholds=iou_thresholds)
        if compute_metrics
        else None
    )

    raw_oracle_metric = (
        RawOracleRecallAccumulator(
            iou_thresholds=raw_oracle_iou_thresholds,
        )
        if compute_metrics and compute_raw_oracle
        else None
    )

    sample_count = 0
    skipped_empty_gt = 0
    total_selected = 0
    processed_batches = 0

    pbar_total = (
        len(val_loader)
        if max_val_batches is None
        else min(len(val_loader), int(max_val_batches))
    )

    labels = []
    if compute_loss:
        labels.append("Loss")
    if compute_metrics:
        labels.append("Eval")

    pbar = make_progress_bar(
        enumerate(val_loader),
        total=pbar_total,
        desc=f"Epoch {epoch} [Val {'+'.join(labels)}]",
        leave=progress_leave,
        mininterval=progress_mininterval,
    )

    validation_start = time.perf_counter()

    for step, batch in pbar:
        if max_val_batches is not None and step >= int(max_val_batches):
            break

        processed_batches += 1

        images, query_texts, image_indices = prepare_model_batch(
            batch=batch,
            device=device,
            channels_last=channels_last,
        )

        targets_device = (
            move_targets_to_device(batch, device)
            if compute_loss
            else None
        )

        gt_boxes_cpu = (
            get_target_boxes_cpu(batch)
            if compute_metrics
            else None
        )

        with autocast(
            device_type=device.type,
            enabled=amp_enabled,
            dtype=amp_dtype if amp_enabled else None,
        ):
            outputs = forward_model_batch(
                model=model,
                images=images,
                query_texts=query_texts,
                image_indices=image_indices,
                return_aux=False,
            )

            pred_bbox = outputs["bbox"]
            pred_score_logit = get_score_logit(outputs)

            if compute_loss:
                loss, loss_dict = criterion(
                    pred_bbox=pred_bbox,
                    pred_score_logit=pred_score_logit,
                    targets=targets_device,
                    lambda_bbox=lambda_bbox,
                    lambda_giou=lambda_giou,
                    lambda_score=lambda_score,
                    lambda_rank=lambda_rank,
                    pos_weight=pos_weight,
                    current_epoch=epoch,
                    total_epochs=total_epochs,
                    quality_warmup_epoch=quality_warmup_epoch,
                    rank_start_epoch=rank_start_epoch,
                    rank_warmup_epoch=rank_warmup_epoch,
                    rank_alpha_min=rank_alpha_min,
                    query_loss_weights=batch.get("query_loss_weights"),
                    text_negative_mask=batch.get("text_negative_mask"),
                )

        if compute_loss:
            zero = pred_bbox.new_zeros(())

            total_loss_sum.add_(loss.detach())
            total_bbox_sum.add_(loss_dict["loss_bbox"].detach())
            total_giou_sum.add_(loss_dict["loss_giou"].detach())
            total_score_sum.add_(loss_dict["loss_score"].detach())
            total_rank_contrib_sum.add_(
                loss_dict.get(
                    "loss_rank_contrib",
                    loss_dict.get("loss_rank", zero),
                ).detach()
            )
            total_rank_raw_sum.add_(
                loss_dict.get("loss_rank_raw", zero).detach()
            )
            total_text_negative_sum.add_(
                loss_dict.get("loss_text_negative", zero).detach()
            )
            total_text_negative_contrib_sum.add_(
                loss_dict.get(
                    "loss_text_negative_contrib",
                    zero,
                ).detach()
            )
            total_text_negative_queries += int(
                batch.get(
                    "text_negative_mask",
                    torch.zeros(0, dtype=torch.bool),
                ).sum().item()
            )

        if compute_metrics and metric is not None and gt_boxes_cpu is not None:
            # Raw Oracle 與正式 filtered metric 共用同一次 model forward。
            # 先使用全部原始 bbox 更新 Oracle，再做 threshold/Top-K/NMS。
            if raw_oracle_metric is not None:
                raw_pred_bbox_cpu = (
                    pred_bbox.detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                )

                if int(raw_pred_bbox_cpu.shape[0]) != len(gt_boxes_cpu):
                    raise RuntimeError(
                        "Raw prediction/GT batch size mismatch: "
                        f"{raw_pred_bbox_cpu.shape[0]} != "
                        f"{len(gt_boxes_cpu)}"
                    )

                for raw_boxes, gt_boxes in zip(
                    raw_pred_bbox_cpu,
                    gt_boxes_cpu,
                ):
                    raw_oracle_metric.update(
                        pred_boxes=raw_boxes,
                        gt_boxes=gt_boxes,
                    )

            pred_scores = pred_score_logit.sigmoid()

            if pred_scores.ndim == 3:
                pred_scores = pred_scores.squeeze(-1)

            selected_batch = select_predictions_batch(
                boxes=pred_bbox.detach(),
                scores=pred_scores.detach(),
                score_thr=score_thr,
                top_k=top_k,
                nms_iou_thr=nms_iou_thr,
                use_topk_fallback=use_topk_fallback,
                use_nms=use_nms,
            )

            for (selected_boxes, selected_scores), gt_boxes in zip(
                selected_batch,
                gt_boxes_cpu,
            ):
                if gt_boxes.numel() == 0:
                    skipped_empty_gt += 1

                total_selected += int(selected_boxes.shape[0])
                sample_count += 1

                metric.update(
                    pred_boxes=selected_boxes,
                    pred_scores=selected_scores,
                    gt_boxes=gt_boxes,
                )

        should_log = (
            (step + 1) % max(1, int(log_interval)) == 0
            or (step + 1) == pbar_total
        )

        if should_log:
            postfix: Dict[str, Any] = {}

            if compute_loss:
                postfix["loss"] = (
                    f"{float((total_loss_sum / processed_batches).item()):.4f}"
                )

            if compute_metrics and metric is not None:
                postfix.update({
                    "samples": sample_count,
                    "pred": metric.num_pred,
                    "sel/img": f"{total_selected / max(1, sample_count):.2f}",
                    "skip_empty": skipped_empty_gt,
                })

            pbar.set_postfix(postfix)
            
    pbar.refresh()
    pbar.close()
    validation_loop_time = time.perf_counter() - validation_start

    val_loss_metrics: Dict[str, float] = {}
    eval_metrics: Dict[str, float] = {}

    if compute_loss:
        denominator = max(1, processed_batches)
        val_loss_metrics = {
            "val_loss": float((total_loss_sum / denominator).item()),
            "val_loss_bbox": float((total_bbox_sum / denominator).item()),
            "val_loss_giou": float((total_giou_sum / denominator).item()),
            "val_loss_score": float((total_score_sum / denominator).item()),
            "val_loss_rank_raw": float((total_rank_raw_sum / denominator).item()),
            "val_loss_rank": float(
                (total_rank_contrib_sum / denominator).item()
            ),
            "val_loss_text_negative": float(
                (total_text_negative_sum / denominator).item()
            ),
            "val_loss_text_negative_contrib": float(
                (total_text_negative_contrib_sum / denominator).item()
            ),
            "val_text_negative_queries": int(total_text_negative_queries),
            "val_score_negative_iou_ignore_thr": float(
                criterion.score_negative_iou_ignore_thr
            ),
            "val_duplicate_suppression_enabled": bool(
                criterion.duplicate_suppression_enabled
            ),
            "val_hard_negative_mining_enabled": bool(
                criterion.hard_negative_mining_enabled
            ),
            "val_matcher_score_alpha": float(
                criterion.resolve_epoch_alpha(
                    current_epoch=epoch,
                    quality_warmup_epoch=quality_warmup_epoch,
                    rank_start_epoch=rank_start_epoch,
                    rank_warmup_epoch=rank_warmup_epoch,
                    rank_alpha_min=rank_alpha_min,
                )[1]
            ),
            "val_matcher_cost_score_effective": float(
                criterion.main_matcher.cost_score
                * criterion.resolve_epoch_alpha(
                    current_epoch=epoch,
                    quality_warmup_epoch=quality_warmup_epoch,
                    rank_start_epoch=rank_start_epoch,
                    rank_warmup_epoch=rank_warmup_epoch,
                    rank_alpha_min=rank_alpha_min,
                )[1]
            ),
        }

    if compute_metrics and metric is not None:
        tqdm.write(
            f"[Eval Metric] Start: samples={sample_count}, "
            f"pred={metric.num_pred}, gt={metric.num_gt}"
        )

        metric_start = time.perf_counter()
        eval_metrics = metric.compute()
        metric_time = time.perf_counter() - metric_start

        raw_oracle_time = 0.0
        if raw_oracle_metric is not None:
            raw_oracle_start = time.perf_counter()
            raw_oracle_metrics = raw_oracle_metric.compute()
            raw_oracle_time = (
                time.perf_counter() - raw_oracle_start
            )
            eval_metrics.update(raw_oracle_metrics)

            raw_oracle_recall50 = float(
                raw_oracle_metrics.get(
                    "raw_oracle_recall50",
                    0.0,
                )
            )
            filtered_recall50 = float(
                eval_metrics.get("recall", 0.0)
            )

            eval_metrics["raw_oracle_gap50"] = (
                raw_oracle_recall50 - filtered_recall50
            )
            eval_metrics["raw_oracle_retention50"] = (
                filtered_recall50
                / max(raw_oracle_recall50, 1e-12)
            )

        eval_metrics.update({
            "valid_samples": sample_count,
            "skipped_empty_gt": skipped_empty_gt,
            "avg_selected_per_sample": (
                total_selected / max(1, sample_count)
            ),
            "eval_loop_time": validation_loop_time,
            "eval_metric_time": metric_time,
            "raw_oracle_metric_time": raw_oracle_time,
        })

        tqdm.write(
            f"[Eval Timing] loop={validation_loop_time:.2f}s "
            f"metric={metric_time:.2f}s "
            f"raw_oracle={raw_oracle_time:.2f}s"
        )
        tqdm.write(
            f"Eval Epoch [{epoch}] "
            f"mAP50={eval_metrics['map50']:.4f} "
            f"mAP50-95={eval_metrics['map50_95']:.4f} "
            f"P={eval_metrics['precision']:.4f} "
            f"R={eval_metrics['recall']:.4f} "
            f"TP={eval_metrics['tp']} "
            f"FP={eval_metrics['fp']} "
            f"GT={eval_metrics['num_gt']} "
            f"Pred={eval_metrics['num_pred']} "
            f"skip_empty={eval_metrics['skipped_empty_gt']}"
        )

        if raw_oracle_metric is not None:
            tqdm.write(
                f"Raw Oracle [{epoch}] "
                f"R25={eval_metrics.get('raw_oracle_recall25', 0.0):.4f} "
                f"R50={eval_metrics.get('raw_oracle_recall50', 0.0):.4f} "
                f"R75={eval_metrics.get('raw_oracle_recall75', 0.0):.4f} "
                f"BestIoUMean="
                f"{eval_metrics.get('raw_best_iou_mean', 0.0):.4f} "
                f"BestIoUMedian="
                f"{eval_metrics.get('raw_best_iou_median', 0.0):.4f} "
                f"Gap50="
                f"{eval_metrics.get('raw_oracle_gap50', 0.0):.4f} "
                f"Retention50="
                f"{eval_metrics.get('raw_oracle_retention50', 0.0):.4f}"
            )

    return val_loss_metrics, eval_metrics

