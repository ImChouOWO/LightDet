"""Validate a checkpoint and export phrase-level Top-1 diagnostics."""
import argparse
import gzip
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from units.model.pipeline.data import ShipGroundingDataset, QueryBudgetBatchSampler
from units.model.tool.config import load_model_config, load_train_config
from units.model.tool.evaluation import build_phrase_evaluation_rows, select_predictions_batch
from units.model.tool.grounding_metrics import GroundingMetrics
from units.model.tool.runtime import compact_grounding_collate_fn, get_raw_targets, prepare_model_batch, forward_model_batch
from units.model.train import load_weights_only
from units.tool.card import VisionTextModel


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def collate(items):
    batch = compact_grounding_collate_fn(items)
    batch['diagnostic_files'] = [item['filename'] for item in items]
    return batch


def diagnose(boxes, scores, objects, relations):
    """Match evaluator's ignore filtering; content classes are operational labels."""
    overlaps = box_iou(boxes, objects)
    positive = relations == 1
    unknown = relations < 0
    correct = (overlaps[:, positive] >= .5).any(1)
    keep = ~(overlaps[:, unknown] >= .5).any(1) | correct
    order = torch.where(keep)[0]
    order = order[scores[order].argsort(descending=True, stable=True)]
    if not positive.any():
        return dict(category='ignored' if unknown.any() else 'no_target', candidate_available=None, top1_query=None)
    available = bool(correct[keep].any())
    if not len(order):
        return dict(category='no_candidate', candidate_available=available, top1_query=None)
    top = int(order[0])
    if correct[top]:
        category = 'correct'
    elif (overlaps[top, relations == 0] >= .5).any():
        category = 'wrong_object'
    elif (overlaps[top, positive] >= .1).any():
        category = 'localization_failure'
    else:
        category = 'background_or_unannotated'
    return dict(category=category, candidate_available=available, top1_query=top,
                max_positive_iou=float(overlaps[top, positive].max()),
                max_any_object_iou=float(overlaps[top].max()) if len(objects) else 0.,
                eligible_query_ids=order.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None)
    parser.add_argument('--model-config', default=None)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--device', default=None, help='e.g. cuda:1 or cpu')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-images', type=int, default=None, help='Optional smoke-test subset')
    args = parser.parse_args()
    cfg = load_train_config(args.config)
    data, train, evaluation = cfg['data'], cfg['train'], cfg['eval']
    run = resolve(cfg['log']['save_dir'])
    weights = resolve(args.weights) if args.weights else run / ('best_' + evaluation['best_metric'] + '.pt')
    output = resolve(args.output_dir) if args.output_dir else run.parent / (run.name + '_val_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    device = torch.device(args.device or ('cuda:' + str(train['device']) if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(4)
    torch.manual_seed(train.get('seed', 49))
    config = load_model_config(args.model_config)['model']
    kwargs = {k: v for k, v in config.items() if k not in ('backbone', 'fpn', 'image_projector', 'precomputed_bert_path')}
    model = VisionTextModel(backbone_config=config['backbone'], fpn_config=config['fpn'],
        image_projector_config=config['image_projector'], precomputed_bert_path=None, **kwargs).to(device)
    checkpoint = load_weights_only(str(weights), model, None, device, prefer_ema=cfg['log'].get('prefer_ema', True))
    model.eval()
    dataset_root = resolve(data['dataset_dir'])
    paths = sorted((dataset_root / 'labels/val').glob('*.json'))
    if args.max_images is not None:
        if args.max_images < 1:
            parser.error('--max-images must be positive')
        paths = paths[:args.max_images]
    if not paths:
        raise ValueError('No validation annotations found')
    size = data['image_size']
    dataset = ShipGroundingDataset(str(dataset_root / 'images/val'), [str(p) for p in paths],
        image_size=(size, size), random_seed=train.get('seed', 49))
    dataset.training_sampling = False
    sampler = QueryBudgetBatchSampler(dataset, query_budget=train['batch_size'], shuffle=False, drop_last=False)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers, collate_fn=collate)
    output.mkdir(parents=True, exist_ok=False)
    metrics, counts, errors = GroundingMetrics(), Counter(), Counter()
    amp = train.get('use_amp', True) and device.type == 'cuda'
    dtype = getattr(torch, train.get('amp_dtype', 'bfloat16').replace('bf16', 'bfloat16').replace('fp16', 'float16'))
    with gzip.open(output / 'predictions.jsonl.gz', 'wt', encoding='utf-8') as stream, torch.inference_mode():
        for batch in tqdm(loader, desc='Validation'):
            images, texts, indices = prepare_model_batch(batch, device, False)
            with torch.autocast(device.type, dtype=dtype, enabled=amp):
                outputs = forward_model_batch(model, images, texts, indices, return_aux=False)
            rows = build_phrase_evaluation_rows(batch=batch, outputs=outputs, pred_bbox=outputs['bbox'])
            targets = get_raw_targets(batch)
            phrase_indices = Counter()
            for row in rows:
                i = row.image_index
                p = phrase_indices[i]
                phrase_indices[i] += 1
                boxes, scores, quality, alignment, gt, ignored = [v.detach().float().cpu() for v in (
                    row.pred_boxes, row.final_scores, row.quality_scores, row.alignment_scores, row.gt_boxes, row.ignored_boxes)]
                selected, _ = select_predictions_batch(boxes[None], scores[None], score_thr=evaluation['score_thr'],
                    top_k=len(boxes), nms_iou_thr=evaluation['nms_iou_thr'], use_nms=evaluation['use_nms'], use_topk_fallback=False)[0]
                metrics.update(pred_boxes=boxes, final_scores=scores, quality_scores=quality,
                    alignment_scores=alignment, gt_boxes=gt, ignored_boxes=ignored, selected_boxes=selected)
                target = targets[i]
                objects = torch.as_tensor(target['boxes']).float().cpu()
                relations = torch.as_tensor(target['phrase_relations'])[p].cpu()
                detail = diagnose(boxes, scores, objects, relations)
                counts[detail['category']] += 1
                if detail['category'] not in ('correct', 'ignored', 'no_target'):
                    errors['correct_candidate_exists' if detail['candidate_available'] else 'no_correct_candidate'] += 1
                record = dict(filename=batch['diagnostic_files'][i], phrase=row.phrase, phrase_index=p,
                    object_ids=target['object_ids'], object_boxes=objects.tolist(), relations=relations.tolist(),
                    objects_complete=target.get('objects_complete', False),
                    boxes=boxes.tolist(), quality_scores=quality.tolist(), alignment_scores=alignment.tolist(),
                    final_scores=scores.tolist(), selected_boxes=selected.tolist(), **detail)
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    result = metrics.compute()
    total_errors = sum(errors.values())
    summary = dict(checkpoint=checkpoint, weights=str(weights), images=len(dataset), partial=args.max_images is not None,
        metrics=result, top1_categories=dict(counts), error_candidate_coverage=dict(errors),
        error_category_fractions={k: v / total_errors for k, v in counts.items() if k not in ('correct','ignored','no_target')} if total_errors else {},
        settings=dict(config=cfg, model_config=config, box_format='normalized_xyxy', iou_threshold=.5,
            localization_overlap_threshold=.1, ranking='raw candidates with evaluator ignore filtering',
            category_priority=['correct','wrong_object','localization_failure','background_or_unannotated'],
            note='Localization failure is an overlap heuristic. Unmatched predictions cannot certify background when annotations are incomplete.'))
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), metrics=result, categories=dict(counts), error_candidate_coverage=dict(errors)), indent=2))


if __name__ == '__main__':
    main()
