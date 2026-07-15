#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Callable

DATA = Path('units/model/pipeline/data.py')
LOSS = Path('units/model/cards/loss_decoupled.py')
EVAL = Path('units/model/tool/evaluation.py')


class PatchError(RuntimeError):
    pass


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PatchError(f'{label}: expected one match, found {count}')
    return text.replace(old, new, 1)


def regex_once(text: str, pattern: str, replacement: str, label: str) -> str:
    result, count = re.subn(pattern, replacement, text, count=1, flags=re.S | re.M)
    if count != 1:
        raise PatchError(f'{label}: expected one match, found {count}')
    return result


def patch_data(text: str) -> str:
    if 'def build_query_text_groups(self, anns):' in text:
        return text

    text = replace_once(
        text,
        '        use_main_colors: bool = True,\n'
        '        use_text_aug: bool = True,\n',
        '        use_query_text: bool = True,\n'
        '        use_main_colors: bool = False,\n'
        '        use_text_aug: bool = False,\n',
        'dataset flags',
    )
    text = replace_once(
        text,
        '        self.use_main_colors = use_main_colors\n'
        '        self.use_text_aug = use_text_aug\n',
        '        self.use_query_text = bool(use_query_text)\n'
        '        self.use_main_colors = bool(use_main_colors)\n'
        '        self.use_text_aug = bool(use_text_aug)\n',
        'dataset flag assignment',
    )

    sample_block = '''            query_text_groups = (
                self.build_query_text_groups(anns)
                if self.use_query_text
                else {}
            )
            main_color_groups = (
                self.build_main_color_groups(anns)
                if self.use_main_colors
                else {}
            )
            text_aug_groups = (
                self.build_text_aug_groups(
                    anns,
                    max_samples_per_object=self.max_text_aug_per_image,
                )
                if self.use_text_aug
                else {}
            )

            for query_text, obj_indices in query_text_groups.items():
                if obj_indices:
                    self.samples.append({
                        "anno_idx": anno_idx,
                        "query_text": query_text,
                        "obj_indices": obj_indices,
                        "group_source": "query_text",
                    })

            for query_text, obj_indices in main_color_groups.items():
                if obj_indices:
                    self.samples.append({
                        "anno_idx": anno_idx,
                        "query_text": query_text,
                        "obj_indices": obj_indices,
                        "group_source": "main_colors",
                    })

            for query_text, obj_indices in text_aug_groups.items():
                if obj_indices:
                    self.samples.append({
                        "anno_idx": anno_idx,
                        "query_text": query_text,
                        "obj_indices": obj_indices,
                        "group_source": "query_texts_aug",
                    })

'''
    text = regex_once(
        text,
        r'^            main_color_groups = self\.build_main_color_groups\(anns\)\n.*?'
        r'(?=        self\.samples_by_anno =)',
        sample_block,
        'sample construction',
    )

    query_method = '''    def build_query_text_groups(self, anns):
        """Bind every query_text to the object that produced it."""
        query_groups = {}
        for obj_idx, obj in enumerate(anns):
            query_text = str(obj.get("query_text", "")).strip()
            if not self.is_valid_query_text(query_text):
                continue
            query_groups.setdefault(query_text, []).append(obj_idx)

        return {
            query_text: list(dict.fromkeys(obj_indices))
            for query_text, obj_indices in query_groups.items()
            if obj_indices
        }

'''
    text = replace_once(
        text,
        '    def extract_colors_from_text(self, text):\n',
        query_method + '    def extract_colors_from_text(self, text):\n',
        'query_text method insertion',
    )

    aug_method = '''    def build_text_aug_groups(
        self,
        anns,
        max_samples_per_object=1,
    ):
        """Bind each augmented phrase only to its original object."""
        query_groups = {}

        for obj_idx, obj in enumerate(anns):
            base_text = str(obj.get("query_text", "")).strip()
            texts_aug = obj.get("query_texts_aug", [])
            if not isinstance(texts_aug, list):
                continue

            candidates = []
            seen = set()
            for text in texts_aug:
                if not self.is_valid_query_text(text):
                    continue
                text = str(text).strip()
                if text == base_text or text in seen:
                    continue
                seen.add(text)
                candidates.append(text)

            if max_samples_per_object is not None:
                limit = int(max_samples_per_object)
                if limit <= 0:
                    candidates = []
                elif len(candidates) > limit:
                    candidates = self.rng.sample(candidates, k=limit)

            for text in candidates:
                query_groups.setdefault(text, []).append(obj_idx)

        return {
            query_text: list(dict.fromkeys(obj_indices))
            for query_text, obj_indices in query_groups.items()
            if obj_indices
        }

'''
    text = regex_once(
        text,
        r'^    def build_text_aug_groups\(self, anns\):\n.*?'
        r'(?=    def load_boxes_and_labels\(self, anns\):)',
        aug_method,
        'augmented text grouping',
    )

    text = replace_once(
        text,
        '    use_main_colors=True,\n    use_text_aug=True,\n',
        '    use_query_text=True,\n'
        '    use_main_colors=False,\n'
        '    use_text_aug=False,\n',
        'loader defaults',
    )
    marker = '        image_size=image_size,\n        use_main_colors=use_main_colors,\n'
    replacement = (
        '        image_size=image_size,\n'
        '        use_query_text=use_query_text,\n'
        '        use_main_colors=use_main_colors,\n'
    )
    if text.count(marker) != 2:
        raise PatchError(f'dataset construction: expected two matches, found {text.count(marker)}')
    text = text.replace(marker, replacement, 2)
    return text


def patch_loss(text: str) -> str:
    if 'def _unique_iou_assignment(' in text:
        return text

    block = '''    @staticmethod
    def _unique_iou_assignment(
        iou_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a one-to-one IoU assignment for queries and GT boxes."""
        device = iou_matrix.device
        num_queries, num_gt = iou_matrix.shape
        if num_queries == 0 or num_gt == 0:
            empty = torch.empty((0,), dtype=torch.long, device=device)
            return empty, empty

        solver = getattr(_legacy, "linear_sum_assignment", None)
        if solver is not None:
            query_indices, gt_indices = solver(
                -iou_matrix.detach().cpu().numpy()
            )
            return (
                torch.as_tensor(query_indices, dtype=torch.long, device=device),
                torch.as_tensor(gt_indices, dtype=torch.long, device=device),
            )

        work = iou_matrix.detach().clone()
        query_indices = []
        gt_indices = []
        for _ in range(min(num_queries, num_gt)):
            flat_index = torch.argmax(work.reshape(-1))
            if float(work.reshape(-1)[flat_index].item()) < 0.0:
                break
            query_index = int(
                torch.div(flat_index, num_gt, rounding_mode="floor").item()
            )
            gt_index = int((flat_index % num_gt).item())
            query_indices.append(query_index)
            gt_indices.append(gt_index)
            work[query_index, :] = -1.0
            work[:, gt_index] = -1.0

        return (
            torch.tensor(query_indices, dtype=torch.long, device=device),
            torch.tensor(gt_indices, dtype=torch.long, device=device),
        )

    def _alignment_targets(
        self,
        *,
        pred_bbox: torch.Tensor,
        targets: List[dict],
        text_negative_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_queries, _ = pred_bbox.shape
        target = pred_bbox.new_zeros(
            (batch_size, num_queries, 1), dtype=torch.float32
        )
        valid = torch.zeros_like(target, dtype=torch.bool)
        max_iou = pred_bbox.new_zeros(
            (batch_size, num_queries, 1), dtype=torch.float32
        )
        boxes = pred_bbox.detach().float()
        negative_rows = text_negative_mask.detach().cpu().tolist()

        for batch_index, (is_negative, row_target) in enumerate(
            zip(negative_rows, targets)
        ):
            if is_negative:
                valid[batch_index, :, 0] = True
                continue

            gt_boxes = row_target.get("boxes")
            if gt_boxes is None:
                valid[batch_index, :, 0] = True
                continue
            gt_boxes = torch.as_tensor(
                gt_boxes, dtype=torch.float32, device=pred_bbox.device
            ).reshape(-1, 4)
            if gt_boxes.numel() == 0 or num_queries == 0:
                valid[batch_index, :, 0] = True
                continue

            iou_matrix = _legacy.box_iou(boxes[batch_index], gt_boxes)
            row_max_iou = iou_matrix.max(dim=1).values
            max_iou[batch_index, :, 0] = row_max_iou
            matched_query, matched_gt = self._unique_iou_assignment(iou_matrix)

            positive = torch.zeros(
                (num_queries,), dtype=torch.bool, device=pred_bbox.device
            )
            if matched_query.numel() > 0:
                matched_iou = iou_matrix[matched_query, matched_gt]
                accepted = matched_iou >= float(
                    self.text_alignment_positive_iou_threshold
                )
                positive[matched_query[accepted]] = True
            if not bool(positive.any()):
                positive[torch.argmax(row_max_iou)] = True

            low_iou_negative = row_max_iou <= float(
                self.text_alignment_negative_iou_threshold
            )
            duplicate_negative = (
                row_max_iou >= float(
                    self.text_alignment_positive_iou_threshold
                )
            ) & (~positive)
            negative = (low_iou_negative | duplicate_negative) & (~positive)

            target[batch_index, positive, 0] = 1.0
            valid[batch_index, positive | negative, 0] = True

        return target, valid, max_iou

'''
    return regex_once(
        text,
        r'^    def _alignment_targets\(\n.*?'
        r'(?=    def text_alignment_loss\(\n)',
        block,
        'text-alignment targets',
    )


def patch_eval(text: str) -> str:
    if 'class RankedRecallAtKAccumulator:' in text:
        return text

    ranked_class = '''class RankedRecallAtKAccumulator:
    """Score-ranked Recall@K with one-to-one matching at IoU 0.50."""

    def __init__(self, ks=(1, 5, 10), iou_threshold=0.50):
        self.ks = tuple(sorted({max(1, int(value)) for value in ks}))
        self.iou_threshold = float(iou_threshold)
        self.num_gt = 0
        self.hits = {value: 0 for value in self.ks}

    def update(self, pred_boxes, pred_scores, gt_boxes):
        pred_boxes = pred_boxes.detach().float().reshape(-1, 4).cpu()
        pred_scores = pred_scores.detach().float().reshape(-1).cpu()
        gt_boxes = gt_boxes.detach().float().reshape(-1, 4).cpu()
        num_gt = int(gt_boxes.shape[0])
        self.num_gt += num_gt
        if num_gt == 0 or pred_boxes.shape[0] == 0:
            return

        order = torch.argsort(pred_scores, descending=True, stable=True)
        iou_matrix = box_iou_xyxy(pred_boxes[order], gt_boxes)
        for top_k in self.ks:
            matched_gt = torch.zeros((num_gt,), dtype=torch.bool)
            hit_count = 0
            for prediction_index in range(min(top_k, iou_matrix.shape[0])):
                candidate = iou_matrix[prediction_index].masked_fill(
                    matched_gt, -1.0
                )
                best_iou, best_gt = candidate.max(dim=0)
                if float(best_iou.item()) >= self.iou_threshold:
                    matched_gt[best_gt] = True
                    hit_count += 1
            self.hits[top_k] += hit_count

    def compute(self):
        denominator = max(1, self.num_gt)
        result = {"ranking_recall_iou": self.iou_threshold}
        for top_k in self.ks:
            result[f"recall50_at_{top_k}"] = self.hits[top_k] / denominator
        return result


'''
    text = replace_once(
        text,
        'class RawOracleRecallAccumulator:\n',
        ranked_class + 'class RawOracleRecallAccumulator:\n',
        'Recall@K class',
    )
    text = replace_once(
        text,
        '    sample_count = 0\n',
        '    ranked_recall_metric = (\n'
        '        RankedRecallAtKAccumulator() if compute_metrics else None\n'
        '    )\n\n'
        '    sample_count = 0\n',
        'Recall@K initialization',
    )
    text = replace_once(
        text,
        '            selected_batch = select_predictions_batch(\n',
        '            if ranked_recall_metric is not None:\n'
        '                for raw_boxes, raw_scores, gt_boxes in zip(\n'
        '                    pred_bbox.detach(), pred_scores.detach(), gt_boxes_cpu\n'
        '                ):\n'
        '                    ranked_recall_metric.update(\n'
        '                        raw_boxes, raw_scores, gt_boxes\n'
        '                    )\n\n'
        '            selected_batch = select_predictions_batch(\n',
        'Recall@K update',
    )
    text = replace_once(
        text,
        '        metric_time = time.perf_counter() - metric_start\n',
        '        metric_time = time.perf_counter() - metric_start\n'
        '        if ranked_recall_metric is not None:\n'
        '            eval_metrics.update(ranked_recall_metric.compute())\n',
        'Recall@K output',
    )
    text = replace_once(
        text,
        '            f"R={eval_metrics[\'recall\']:.4f} "\n',
        '            f"R={eval_metrics[\'recall\']:.4f} "\n'
        '            f"R@1={eval_metrics.get(\'recall50_at_1\', 0.0):.4f} "\n'
        '            f"R@5={eval_metrics.get(\'recall50_at_5\', 0.0):.4f} "\n'
        '            f"R@10={eval_metrics.get(\'recall50_at_10\', 0.0):.4f} "\n',
        'Recall@K logging',
    )
    return text


PATCHERS: tuple[tuple[Path, Callable[[str], str]], ...] = (
    (DATA, patch_data),
    (LOSS, patch_loss),
    (EVAL, patch_eval),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-backup', action='store_true')
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()

    try:
        prepared = []
        for relative, patcher in PATCHERS:
            path = root / relative
            if not path.is_file():
                raise PatchError(f'missing target file: {path}')
            original = path.read_text(encoding='utf-8')
            modified = patcher(original)
            compile(modified, str(relative), 'exec')
            prepared.append((path, original, modified))

        changed = [item for item in prepared if item[1] != item[2]]
        if not changed:
            print('[ok] patch is already applied')
            return 0
        for path, _, _ in changed:
            print(f'[check] {path.relative_to(root)}')
        if args.dry_run:
            print('[dry-run] no files written')
            return 0

        for path, original, modified in changed:
            if not args.no_backup:
                backup = path.with_name(path.name + '.query-ranking-fix.bak')
                shutil.copy2(path, backup)
                print(f'[backup] {backup.relative_to(root)}')
            path.write_text(modified, encoding='utf-8')
            print(f'[write] {path.relative_to(root)}')

        print('[done] set resume_path: null for the first clean run')
        print('[done] inspect recall50_at_1, recall50_at_5, recall50_at_10')
        return 0
    except PatchError as error:
        print(f'[error] {error}', file=sys.stderr)
        print('[abort] no target files were written', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
