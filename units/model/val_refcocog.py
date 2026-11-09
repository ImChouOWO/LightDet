"""Evaluate LightDet on the RefCOCOg release used by LocateAnything."""
import argparse
from collections import OrderedDict
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from statistics import mean
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from units.model.pipeline.data import DATA_SOURCES
from units.model.tool.config import load_config, resolve
from units.tool.card import VisionTextModel

IOU_THRESHOLDS = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class RefCOCOgDataset(Dataset):
    """Preserve released expression order and the checkpoint's image preprocessing."""
    def __init__(self, annotation, image_root, image_size, limit=None):
        self.rows = [json.loads(line) for line in annotation.read_text().splitlines()]
        if limit is not None:
            self.rows = self.rows[:limit]
        self.image_root, self.image_size = image_root, image_size
        self.cache = OrderedDict()
        for row in self.rows:
            if row['task_name'] != 'referring_object_detection' or len(row['categories']) != 1:
                raise ValueError('Expected one referring expression per record')
            if set(row['gt']) != set(row['categories']) or sum(map(len, row['gt'].values())) != 1:
                raise ValueError('Expected one target per RefCOCOg expression')
            path = image_root / Path(row['image_path']).name
            if not path.is_file():
                raise FileNotFoundError(path)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        name = Path(row['image_path']).name
        if name not in self.cache:
            with Image.open(self.image_root / name) as source:
                image = source.convert('RGB')
                width, height = image.size
                scale = self.image_size / max(width, height)
                rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
                resized = image.resize((rw, rh), Image.Resampling.BILINEAR)
                canvas = Image.new('RGB', (self.image_size, self.image_size), (124, 116, 104))
                canvas.paste(resized, (0, 0))
                tensor = torch.from_numpy(np.array(canvas)).permute(2, 0, 1)
            self.cache[name] = (tensor, (width, height, rw, rh))
            if len(self.cache) > 16:
                self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        tensor, dimensions = self.cache[name]
        return dict(index=index, name=name, image=tensor, dimensions=dimensions,
                    text=row['categories'][0])


def collate(items):
    positions, images, indices = {}, [], []
    for item in items:
        if item['name'] not in positions:
            positions[item['name']] = len(images)
            images.append(item['image'])
        indices.append(positions[item['name']])
    return dict(images=torch.stack(images), image_indices=torch.tensor(indices),
                texts=[item['text'] for item in items],
                indices=[item['index'] for item in items],
                dimensions=torch.tensor([item['dimensions'] for item in items]))


def load_evaluator(path):
    spec = importlib.util.spec_from_file_location('locateanything_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_evaluator(module):
    box = [0, 0, 10, 10]
    def score(predicted):
        calculator = module.UniversalMetricsCalculator()
        calculator.calculate_metrics_for_sample(dict(task_name='referring_object_detection',
            dataset_name='sanity', gt={'target': [box]}, extracted_predictions={'target': predicted}), .5)
        metric = calculator.results['referring_object_detection_sanity']
        p, r = mean(metric['precisions']), mean(metric['recalls'])
        return 2 * p * r / (p + r) if p + r else 0
    assert score([box]) == 1
    assert score([]) == 0
    assert abs(score([box, box]) - 2 / 3) < 1e-12
    assert score([[20, 20, 30, 30]]) == 0


def summarize(module, records, expected_dataset):
    if not records:
        return None
    calculator = module.UniversalMetricsCalculator()
    with contextlib.redirect_stdout(io.StringIO()):
        values = calculator.calculate_all_metrics(records, IOU_THRESHOLDS)
    key = f'referring_object_detection_{expected_dataset}'
    scores = {}
    for threshold, result in values.items():
        if set(result['basic_metrics']) != {key}:
            raise ValueError('Unexpected dataset in official metric output')
        metric = result['basic_metrics'][key]
        if len(metric['recalls']) != len(records):
            raise ValueError('Official evaluator skipped a record')
        p, r = mean(metric['precisions']), mean(metric['recalls'])
        scores[f'{threshold:.2f}'] = dict(precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0)
    return dict(expressions=len(records), f1_at_05=scores['0.50']['f1'],
                f1_at_095=scores['0.95']['f1'],
                f1_mean=mean(value['f1'] for value in scores.values()), by_iou=scores)


@torch.inference_mode()
def evaluate(model, tokenizer, config, device, dataset, args, output, split, train_images, module):
    texts = [row['categories'][0] for row in dataset.rows]
    lengths = [len(ids) for ids in tokenizer(texts, truncation=False)['input_ids']]
    max_tokens = max(lengths)
    if max_tokens > config['data']['text_max_length']:
        raise ValueError(f'Expression has {max_tokens} tokens; checkpoint text limit exceeded')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=True, collate_fn=collate)
    answers, top1_answers, unseen = [], [], []
    raw_path = output.with_name(f'{output.stem}_{split}_predictions.jsonl')
    if raw_path.exists():
        raise FileExistsError(raw_path)
    rejected, selected, top1_correct = 0, 0, 0
    start = time.monotonic()
    with raw_path.open('x') as stream:
        for batch in tqdm(loader, desc=f'RefCOCOg [{split}]', mininterval=5):
            images = batch['images'].to(device, non_blocking=True).float().div_(255)
            tokens = tokenizer(batch['texts'], padding=True, truncation=False, return_tensors='pt')
            tokens = {key: value.to(device, non_blocking=True) for key, value in tokens.items()
                      if key in ('input_ids', 'attention_mask', 'token_type_ids')}
            dimensions = batch['dimensions']
            valid_sizes = dimensions[:, 2:].to(device) / dataset.image_size
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == 'cuda' and config['train']['amp']):
                prediction = model(images, tokens, valid_sizes=valid_sizes,
                                   image_indices=batch['image_indices'].to(device))
            boxes = prediction['boxes'].float().cpu()
            scores = prediction['scores'].float().cpu()
            if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
                raise ValueError('Non-finite model prediction')
            # Undo the actual rounded letterbox resize independently for x and y.
            scale = dimensions[:, :2] / dimensions[:, 2:] * dataset.image_size
            boxes = boxes * scale.repeat(1, 2)[:, None, :]
            for index, candidate_boxes, candidate_scores in zip(batch['indices'], boxes, scores):
                if index != len(answers):
                    raise ValueError('Evaluation order changed')
                row = dataset.rows[index]
                phrase = row['categories'][0]
                if not (candidate_boxes[:, 2:] >= candidate_boxes[:, :2]).all():
                    raise ValueError('Invalid predicted box')
                order = candidate_scores.argsort(descending=True)
                valid = (candidate_boxes[:, 2:] > candidate_boxes[:, :2]).all(1)
                keep = order[(candidate_scores[order] >= args.score_threshold) & valid[order]]
                record = dict(row, extracted_predictions={phrase: candidate_boxes[keep].tolist()})
                top1 = dict(row, extracted_predictions={phrase: candidate_boxes[order[:1]].tolist()})
                answers.append(record)
                top1_answers.append(top1)
                image_id = int(Path(row['image_path']).stem.rsplit('_', 1)[1])
                if image_id not in train_images:
                    unseen.append(record)
                selected += len(keep)
                rejected += len(keep) == 0
                gt = row['gt'][phrase][0]
                top1_correct += module.calculate_iou(top1['extracted_predictions'][phrase][0], gt) >= .5
                stream.write(json.dumps(dict(record, candidate_boxes=candidate_boxes.tolist(),
                    candidate_scores=candidate_scores.tolist(), selected_indices=keep.tolist())) + '\n')
    metric = summarize(module, answers, f'RefCOCOg_{split}')
    metric.update(images=len({row['image_path'] for row in dataset.rows}), max_tokens=max_tokens,
        score_threshold=args.score_threshold, nms=False, candidates=boxes.shape[1],
        rejected_expressions=rejected, rejection_rate=rejected / len(answers),
        avg_selected=selected / len(answers), seconds=time.monotonic()-start,
        top1_acc_at_05=top1_correct / len(answers),
        top1_diagnostic=summarize(module, top1_answers, f'RefCOCOg_{split}'),
        train_overlap_expressions=len(answers)-len(unseen),
        unseen_image_subset=summarize(module, unseen, f'RefCOCOg_{split}'),
        predictions=str(raw_path), predictions_sha256=file_hash(raw_path))
    return metric


def print_results(result):
    print('\nSplit      Expressions  F1@0.5    F1@0.95   F1@Mean   Top1 Acc@0.5', flush=True)
    for split, metric in result['splits'].items():
        print(f"{split:<10} {metric['expressions']:<12} {metric['f1_at_05']:>8.2%} "
              f"{metric['f1_at_095']:>9.2%} {metric['f1_mean']:>9.2%} "
              f"{metric['top1_acc_at_05']:>13.2%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data-config', help='Explicit training dataset settings; otherwise use checkpoint config')
    parser.add_argument('--annotation-dir', default='datasets/pre')
    parser.add_argument('--image-root', default='datasets/images')
    parser.add_argument('--evaluator', default='datasets/pre/official_other_metric.py')
    parser.add_argument('--splits', nargs='+', choices=['val', 'test'], default=['val', 'test'])
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--score-threshold', type=float, default=.7)
    parser.add_argument('--max-expressions', type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or not 0 <= args.score_threshold <= 1:
        parser.error('Invalid batch size, workers, or score threshold')
    if args.max_expressions is not None and args.max_expressions < 1:
        parser.error('--max-expressions must be positive')
    output, weights = resolve(args.output), resolve(args.weights)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('Use a new output file in an existing directory')
    device = torch.device(args.device)
    torch.set_num_threads(4)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    if checkpoint['schema'] != 'lightdet.grefcoco.queries.v3':
        raise ValueError('Expected text-guided query checkpoint v3')
    settings, config = checkpoint['model_config'], checkpoint['config']
    if args.data_config:
        config['data'] = load_config(args.data_config)['data']
    data = config['data']
    if data.get('dataloader') not in DATA_SOURCES:
        parser.error('Specify the training dataset with --data-config; data.dataloader is required')
    model = VisionTextModel(**dict(settings, backbone_weights=None)).to(device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    tokenizer = AutoTokenizer.from_pretrained(resolve(settings['encoder_path']), local_files_only=True)
    module = load_evaluator(resolve(args.evaluator))
    verify_evaluator(module)
    references_path = resolve(data['dataset_dir']) / 'labels' / DATA_SOURCES[data['dataloader']][0]
    references = json.loads(references_path.read_text())
    train_images = {row['image_id'] for row in references if row['split'] == data['train_split']}
    result = dict(weights=str(weights), weights_sha256=file_hash(weights), epoch=checkpoint['epoch'],
        partial=args.max_expressions is not None, device=str(device),
        parameters=sum(p.numel() for p in model.parameters()),
        training_data=dict(dataloader=data['dataloader'], split=data['train_split'], images=len(train_images),
                           annotations=str(references_path), annotation_sha256=file_hash(references_path)),
        metric_source='https://github.com/NVlabs/Eagle/blob/783f656d127ee498137b5ff52603ce36c292d317/Embodied/evaluation/metrics/other_metric.py',
        metric_sha256=file_hash(resolve(args.evaluator)), script_sha256=file_hash(Path(__file__)),
        protocol=dict(iou_thresholds=IOU_THRESHOLDS, matching='official greedy one-to-one IoU',
            aggregation='F1 of mean per-expression precision and recall; mean over 10 IoU thresholds',
            score_threshold=args.score_threshold, nms=False, candidates=settings['num_object_queries'],
            image_size=config['data']['image_size'], ground_truth_used_for_inference=False,
            top1='diagnostic only: one highest-score box per expression, without score threshold'),
        splits={})
    print(f"\nRefCOCOg | Epoch {result['epoch']} | {device} | Main {settings['num_object_queries']}\n"
          f"Threshold {args.score_threshold} | NMS off | Official LocateAnything F1@IoU\n"
          f"Output: {output}\n", flush=True)
    for split in args.splits:
        annotation = resolve(args.annotation_dir) / f'RefCOCOg_{split}.jsonl'
        dataset = RefCOCOgDataset(annotation, resolve(args.image_root), config['data']['image_size'], args.max_expressions)
        result['splits'][split] = evaluate(model, tokenizer, config, device, dataset, args,
                                          output, split, train_images, module)
        result['splits'][split]['annotation_sha256'] = file_hash(annotation)
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(output)
        print_results(result)
    print(f'Complete: {output}', flush=True)


if __name__ == '__main__':
    main()
