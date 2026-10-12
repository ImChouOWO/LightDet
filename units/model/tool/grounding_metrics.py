"""GREC reference metrics, single-target REC accuracy, and ranking diagnostics."""

import torch
from torchvision.ops import box_iou, generalized_box_iou


def matched_count(edges):
    """Maximum one-to-one matching; a duplicate cannot recall a GT twice."""
    adjacency = edges.tolist()
    owners = [-1] * edges.shape[1]

    def assign(query, seen):
        for obj, valid in enumerate(adjacency[query]):
            if not valid or obj in seen:
                continue
            seen.add(obj)
            if owners[obj] < 0 or assign(owners[obj], seen):
                owners[obj] = query
                return True
        return False

    return sum(assign(query, set()) for query in range(edges.shape[0]))


def grec_correct_set(boxes, gt):
    """Reproduce gRefCOCO's greedy GIoU matching at threshold 0.5."""
    if len(boxes) != len(gt):
        return False
    if not len(gt):
        return True
    overlaps = generalized_box_iou(boxes, gt)
    for _ in range(len(gt)):
        value, index = overlaps.flatten().max(0)
        if value < 0.5:
            return False
        row, col = int(index) // len(gt), int(index) % len(gt)
        overlaps[row, :] = -1
        overlaps[:, col] = -1
    return True


class GroundingMetrics:
    def __init__(self):
        self.queries = self.positive_queries = self.empty_queries = 0
        self.ignored_queries = self.num_gt = self.selected = 0
        self.hits = {k: 0 for k in (1, 5, 10)}
        self.recalled = {k: 0 for k in (5, 10)}
        self.oracle = 0
        self.quality_hits = self.text_hits = 0
        self.grec_success = self.grec_empty_success = self.grec_target_success = 0
        self.single_queries = self.single_hits = 0

    def update(self, *, pred_boxes, final_scores, quality_scores, alignment_scores,
               gt_boxes, ignored_boxes, selected_boxes):
        def cpu(value):
            return value.detach().to(device="cpu", dtype=torch.float32)

        boxes, gt, ignored, selected = map(cpu, (
            pred_boxes, gt_boxes, ignored_boxes, selected_boxes))
        scores, quality, text = map(cpu, (final_scores, quality_scores, alignment_scores))
        if not all(torch.isfinite(x).all() for x in (boxes, gt, ignored, selected, scores, quality, text)):
            raise ValueError("Non-finite grounding evaluation input")
        # Unknown relations cannot certify an empty query.
        if len(gt) == 0 and len(ignored) > 0:
            self.ignored_queries += 1
            return

        def known_mask(candidates):
            if not len(ignored):
                return torch.ones(len(candidates), dtype=torch.bool)
            overlaps_unknown = (box_iou(candidates, ignored) >= 0.5).any(1)
            overlaps_positive = (box_iou(candidates, gt) >= 0.5).any(1)
            return ~overlaps_unknown | overlaps_positive

        keep = known_mask(boxes)
        boxes, scores, quality, text = boxes[keep], scores[keep], quality[keep], text[keep]
        selected = selected[known_mask(selected)]
        # Benchmark scores use raw candidates, without NMS or top-k fallback.
        order = scores.argsort(descending=True, stable=True)
        benchmark_boxes, benchmark_scores = boxes[order], scores[order]
        benchmark_boxes = benchmark_boxes[benchmark_scores >= 0.7]
        benchmark_success = grec_correct_set(benchmark_boxes, gt)
        self.grec_success += int(benchmark_success)
        self.queries += 1
        self.selected += len(selected)
        self.num_gt += len(gt)
        if not len(gt):
            self.grec_empty_success += int(benchmark_success)
            self.empty_queries += 1
            return

        self.grec_target_success += int(len(benchmark_boxes) > 0)
        self.positive_queries += 1
        edges = box_iou(boxes, gt) >= 0.5
        ranked = edges[scores.argsort(descending=True, stable=True)]
        if len(gt) == 1:
            self.single_queries += 1
            single_iou = box_iou(boxes[scores.argsort(descending=True, stable=True)[:1]], gt)
            self.single_hits += int((single_iou > 0.5).any())
        for k in self.hits:
            self.hits[k] += int(ranked[:k].any())
        for k in self.recalled:
            self.recalled[k] += matched_count(ranked[:k])
        self.oracle += int(edges.any(0).sum())
        self.quality_hits += int(edges[quality.argsort(descending=True, stable=True)[:1]].any())
        self.text_hits += int(edges[text.argsort(descending=True, stable=True)[:1]].any())

    def compute(self):
        def ratio(numerator, denominator):
            return numerator / denominator if denominator else None

        return {
            **{f"hit_at_{k}": ratio(v, self.positive_queries) for k, v in self.hits.items()},
            **{f"object_recall_at_{k}": ratio(v, self.num_gt) for k, v in self.recalled.items()},
            "grec_precision_f1": ratio(self.grec_success, self.queries),
            "grec_n_acc": ratio(self.grec_empty_success, self.empty_queries),
            "grec_t_acc": ratio(self.grec_target_success, self.positive_queries),
            "rec_acc50": ratio(self.single_hits, self.single_queries),
            "single_target_query_count": self.single_queries,
            "oracle50": ratio(self.oracle, self.num_gt),
            "quality_hit_at_1": ratio(self.quality_hits, self.positive_queries),
            "text_hit_at_1": ratio(self.text_hits, self.positive_queries),
            "positive_query_count": self.positive_queries,
            "no_target_query_count": self.empty_queries,
            "ignored_query_count": self.ignored_queries,
            "evaluated_query_count": self.queries,
            "gt_object_phrase_pairs": self.num_gt,
            "avg_selected_per_query": ratio(self.selected, self.queries),
            "metric_schema": "lightdet.grounding.v3",
            "benchmark_protocol": "grefcoco_grec_reference_giou_greedy_score0.7_no_nms",
            "benchmark_score_threshold": 0.7,
            "diagnostic_metric_names": ["hit_at_1", "hit_at_5", "hit_at_10",
                "object_recall_at_5", "object_recall_at_10", "oracle50",
                "quality_hit_at_1", "text_hit_at_1", "avg_selected_per_query"],
        }
