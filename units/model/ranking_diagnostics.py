"""Inspect GT ROI semantics, pair-head contribution, negative coverage and score fusion without updating weights."""
import argparse
import csv
import gzip
import hashlib
import html
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from units.model.cards.phrase_matching import _phrase_masks, phrase_matrices
from units.model.cards.relations import normalize_record
from units.model.pipeline.data import ShipGroundingDataset
from units.model.tool.config import load_model_config, load_train_config
from units.model.tool.runtime import compact_grounding_collate_fn, forward_model_batch, get_raw_targets, prepare_model_batch, fuse_quality_and_phrase_logits
from units.model.train import _select_checkpoint_state_dict
from units.model.val import diagnose
from units.tool.card import VisionTextModel


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def collate(items):
    batch = compact_grounding_collate_fn(items)
    batch['filenames'] = [item['filename'] for item in items]
    return batch


def components(matching, offsets, valid, spans):
    # Reuse production span masks; expose the two terms without changing the trained head.
    masks = _phrase_masks(offsets, valid, spans).to(matching['visual'].device)
    with torch.autocast(matching['visual'].device.type, enabled=False):
        scores = matching['token_weights'].float()[:, None].expand_as(masks)
        scores = scores.masked_fill(~masks, -torch.inf)
        scores = torch.where(masks.any(-1, keepdim=True), scores, torch.zeros_like(scores))
        weights = scores.softmax(-1) * masks
        text = F.normalize(torch.bmm(weights, matching['text'].float()), dim=-1)
        visual = F.normalize(matching['visual'].float(), dim=-1)
        cosine = torch.bmm(visual, text.transpose(1, 2))
        pairs = torch.cat((visual[:, :, None]*text[:, None],
                           (visual[:, :, None]-text[:, None]).abs()), -1)
        delta = .25*matching['pair_head'](pairs).squeeze(-1).tanh()
        full = (cosine+delta)*matching['scale'].float()+matching['bias'].float()
        base = cosine*matching['scale'].float()+matching['bias'].float()
    reference = phrase_matrices(matching, offsets, valid, spans)[0]
    result = []
    for b, groups in enumerate(spans):
        n = len(groups)
        torch.testing.assert_close(full[b, :, :n], reference[b], rtol=2e-5, atol=2e-5)
        result.append(dict(cosine=cosine[b, :, :n], delta=delta[b, :, :n],
                           full=full[b, :, :n], base=base[b, :, :n]))
    return result


def gt_components(model, outputs, targets):
    counts = [len(t['boxes']) for t in targets]
    if not max(counts, default=0):
        return [dict(cosine=torch.empty(0, len(t['phrase_char_spans'])),
                     delta=torch.empty(0, len(t['phrase_char_spans'])),
                     full=torch.empty(0, len(t['phrase_char_spans'])),
                     base=torch.empty(0, len(t['phrase_char_spans']))) for t in targets]
    boxes = outputs['bbox'].new_zeros(len(targets), max(counts), 4)
    for b, target in enumerate(targets):
        boxes[b, :counts[b]] = target['boxes'].to(boxes)
    query = model._condition_score_queries(boxes)
    refined, _ = model.score_transformer._forward_group(outputs['fpn_features'], query,
        boxes, outputs['input_image_size'], model.score_transformer.MAIN_GROUP)
    matching = model.head.predict_phrase_matching(refined, outputs['text_token'], outputs['alignment_text_mask'])
    values = components(matching, outputs['token_offsets'], outputs['alignment_text_mask'],
                        [t['phrase_char_spans'] for t in targets])
    return [{k:v[:counts[b]] for k,v in row.items()} for b,row in enumerate(values)]


def top1(scores, correct, keep):
    ids = torch.where(keep)[0]
    if not len(ids):
        return None, False
    q = int(ids[scores[ids].argmax()])
    return q, bool(correct[q])


def margin(scores, positive, negative):
    return float(scores[positive].max()-scores[negative].max()) if positive.any() and negative.any() else None


def transition(before, after):
    if before is None or after is None:
        return None
    if before != after:
        return 'improved' if after else 'harmed'
    return 'both_correct' if after else 'both_wrong'


class Stats:
    def __init__(self):
        self.counts = Counter()
        self.sums = Counter()
        self.squares = Counter()

    def add(self, name, value):
        if value is None:
            return
        values = torch.as_tensor(value).detach().double().flatten().cpu()
        if not torch.isfinite(values).all():
            raise FloatingPointError(f'Nonfinite diagnostic: {name}')
        self.counts[name] += values.numel()
        self.sums[name] += float(values.sum())
        self.squares[name] += float(values.square().sum())

    def result(self):
        result = {}
        for key, n in self.counts.items():
            mean = self.sums[key]/n if n else None
            variance = max(0., self.squares[key]/n-mean**2) if n else None
            result[key] = dict(count=n, mean=mean, std=variance**.5 if n else None)
        return result


def add_component_stats(stats, prefix, values, keep):
    for name in ('cosine', 'delta'):
        selected = values[name][keep]
        stats.add(prefix+'_'+name, selected)
        stats.add(prefix+'_'+name+'_abs', selected.abs())
    logits = values['full'][keep]
    probability = logits.sigmoid()
    stats.add(prefix+'_alignment_logit', logits)
    stats.add(prefix+'_probability_low', probability <= .01)
    stats.add(prefix+'_probability_high', probability >= .99)
    stats.add(prefix+'_sigmoid_derivative', probability*(1-probability))
    stats.add(prefix+'_delta_near_bound', values['delta'][keep].abs() >= .249)


def semantic_parts(key):
    kind, _, colors = key.partition(':')
    return kind, frozenset(colors.split('|')) if colors else frozenset()


def coverage_record(raw):
    record = normalize_record(raw)
    phrases, objects = record['regions'], record['objects']
    relations = record['phrase_relations']
    counts, types, examples = Counter(), Counter(), []
    counts['objects'] = len(objects)
    counts['phrases'] = len(phrases)
    for p, row in zip(phrases, relations):
        pos = [i for i,r in enumerate(row) if r == 1]
        neg = [i for i,r in enumerate(row) if r == 0]
        counts.update(positive_pairs=len(pos), negative_pairs=len(neg), ignore_pairs=row.count(-1))
        counts['phrases_with_positive'] += bool(pos)
        counts['phrases_without_positive'] += not bool(pos)
        counts['phrases_with_positive_and_negative'] += bool(pos and neg)
        kind, _ = semantic_parts(p['semantic_key'])
        types[kind] += 1
        shared = []
        for i in pos:
            a = set(objects[i].get('attributes', {}).get('colors', []))
            for j in neg:
                b = set(objects[j].get('attributes', {}).get('colors', []))
                if not a or not b:
                    counts['object_pairs_without_color_attributes'] += 1
                elif a & b:
                    counts['same_color_object_pairs'] += 1
                    if not shared:
                        shared = [objects[i]['id'], objects[j]['id']]
                else:
                    counts['different_color_object_pairs'] += 1
        if shared:
            examples.append(dict(kind='same_color_objects', phrase=p['phrase'], object_ids=shared))
    for i, a in enumerate(phrases):
        ka, ca = semantic_parts(a['semantic_key'])
        for j in range(i+1, len(phrases)):
            b = phrases[j]
            kb, cb = semantic_parts(b['semantic_key'])
            if ka == kb and ca == cb:
                counts['same_key_relation_conflicts'] += relations[i] != relations[j]
            if {ka,kb} == {'contains_color','single_color'} and ca == cb:
                name = 'single_vs_contains'
            elif ka == kb and ca != cb:
                name = 'same_predicate_different_colors'
            else:
                continue
            ids = [objects[k]['id'] for k,(ra,rb) in enumerate(zip(relations[i], relations[j]))
                   if {ra,rb} == {0,1}]
            if ids:
                counts[name+'_object_contrasts'] += len(ids)
                examples.append(dict(kind=name, phrases=[a['phrase'],b['phrase']], object_ids=ids))
    return counts, types, examples


def coverage(dataset, limit, seed):
    result = {}
    for split in ('train','val'):
        paths = sorted((dataset/'labels'/split).glob('*.json'))
        if not paths:
            raise ValueError(f'No {split} labels found')
        total_files = len(paths)
        if limit and len(paths) > limit:
            paths = random.Random(seed).sample(paths, limit)
        counts, types, images, samples = Counter(), Counter(), Counter(), defaultdict(list)
        for path in tqdm(paths, desc=f'Coverage {split}'):
            raw = json.loads(path.read_text())
            c, t, examples = coverage_record(raw)
            counts.update(c)
            types.update(t)
            kinds = {e['kind'] for e in examples}
            images.update(kinds)
            for example in examples:
                if len(samples[example['kind']]) < 10:
                    samples[example['kind']].append(dict(filename=raw['filename'], **example))
        result[split] = dict(files=len(paths), total_files=total_files, partial=len(paths)<total_files,
                             counts=dict(counts), phrase_types=dict(types), images_by_contrast=dict(images),
                             examples=dict(samples))
    return result


def load_checkpoint(path, model, prefer_ema):
    before = path.stat()
    blob = torch.load(path, map_location='cpu', weights_only=False)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
        raise RuntimeError('Checkpoint changed while loading; use a fixed checkpoint file')
    state, source = _select_checkpoint_state_dict(blob, prefer_ema=prefer_ema)
    expected = model.state_dict()
    missing = [k for k in expected if k not in state]
    wrong = [k for k in expected if k in state and state[k].shape != expected[k].shape]
    extra = sorted(set(state)-set(expected))
    if missing or wrong or extra:
        raise ValueError(f'Checkpoint/model mismatch; missing={missing}, shape={wrong}, extra={extra}')
    model.load_state_dict(state, strict=True)
    return dict(path=str(path), source=source, source_epoch=blob.get('epoch'),
                source_best_metric=blob.get('best_metric'), sha256=digest.hexdigest())


def gallery(records, dataset, output, coverage_result):
    folder = output/'images'
    folder.mkdir()
    cards = []
    for i, row in enumerate(records):
        with Image.open(dataset/'images/val'/row['filename']) as original:
            image = original.convert('RGB')
        image.thumbnail((1000, 700))
        draw = ImageDraw.Draw(image)
        for box, rel in zip(row['object_boxes'], row['relations']):
            color = 'lime' if rel == 1 else 'orange' if rel == 0 else 'gray'
            draw.rectangle(tuple(v*image.size[k%2] for k,v in enumerate(box)), outline=color, width=2)
        q = row['top1_query']
        if q is not None:
            draw.rectangle(tuple(v*image.size[k%2] for k,v in enumerate(row['boxes'][q])), outline='red', width=4)
        name = f'images/{i:03d}.jpg'
        image.save(output/name, quality=90)
        cards.append(f'<figure><figcaption>{html.escape(row["filename"])} · {html.escape(row["phrase"])} · '
                     f'{html.escape(row["category"])}</figcaption><img src="{name}" loading="lazy"></figure>')
    coverage_cards = []
    for split, values in coverage_result.items():
        for kind, examples in values['examples'].items():
            if not examples:
                continue
            example = examples[0]
            label_path = dataset/'labels'/split/(Path(example['filename']).stem+'.json')
            raw = json.loads(label_path.read_text())
            with Image.open(dataset/'images'/split/example['filename']) as original:
                image = original.convert('RGB')
            image.thumbnail((1000,700))
            draw = ImageDraw.Draw(image)
            for obj in raw['objects']:
                if obj['id'] in example['object_ids']:
                    box = obj['bbox']
                    draw.rectangle(tuple(v/raw['width' if k%2==0 else 'height']*image.size[k%2]
                                         for k,v in enumerate(box)),outline='magenta',width=3)
            name = f'images/coverage_{split}_{kind}.jpg'
            image.save(output/name,quality=90)
            coverage_cards.append('<figure><figcaption>'+html.escape(json.dumps(dict(split=split,**example),ensure_ascii=False))+
                                  f'</figcaption><img src="{name}" loading="lazy"></figure>')
    content = ('<!doctype html><meta charset="utf-8"><title>Ranking diagnostics</title>'
               '<style>body{font-family:system-ui;max-width:1100px;margin:24px auto}img{max-width:100%}pre{white-space:pre-wrap}</style>'
               '<h1>排序檢查</h1><p>綠：GT 正例；橘：GT 負例；灰：ignore；紅：融合 Top-1。'
               '圖片是抽樣案例，不代表全體比例。</p>'
               '<p><a href="summary.json">摘要</a> · <a href="cases.csv">逐描述結果</a> · '
               '<a href="coverage.json">負例覆蓋</a></p><h2>排序案例</h2>'+''.join(cards)+
               '<h2>標記中的區分樣本</h2><p>紫框：此組對照涉及的物件；正負關係請依描述與 coverage.json 判讀。</p>'+''.join(coverage_cards))
    (output/'index.html').write_text(content, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--model-config')
    parser.add_argument('--weights')
    parser.add_argument('--output-dir')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1, help='Images, not query budget')
    parser.add_argument('--max-images', type=int, help='Seeded validation subset')
    parser.add_argument('--coverage-limit', type=int, help='Optional per-split coverage subset')
    parser.add_argument('--examples', type=int, default=12)
    args = parser.parse_args()
    if args.workers < 0 or args.examples < 0 or any(v is not None and v < 1 for v in (args.batch_size,args.max_images,args.coverage_limit)):
        parser.error('Counts must be positive; workers and examples may be zero')
    cfg = load_train_config(args.config)
    config = load_model_config(args.model_config)['model']
    run = resolve(cfg['log']['save_dir'])
    weights = resolve(args.weights) if args.weights else run/'best_hit_at_1.pt'
    output = resolve(args.output_dir) if args.output_dir else run.parent/(run.name+'_ranking_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    dataset_root = resolve(cfg['data']['dataset_dir'])
    seed = cfg['train'].get('seed',49)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    kwargs = {k:v for k,v in config.items() if k not in ('backbone','fpn','image_projector','precomputed_bert_path')}
    model = VisionTextModel(backbone_config=config['backbone'],fpn_config=config['fpn'],
        image_projector_config=config['image_projector'],precomputed_bert_path=None,**kwargs)
    checkpoint = load_checkpoint(weights, model, cfg['log'].get('prefer_ema',True))
    print(json.dumps(checkpoint,ensure_ascii=False),flush=True)
    model.to(device).eval().requires_grad_(False)
    paths = sorted((dataset_root/'labels/val').glob('*.json'))
    if not paths:
        raise ValueError('No validation annotations found')
    total_images = len(paths)
    if args.max_images and len(paths)>args.max_images:
        paths = random.Random(seed).sample(paths,args.max_images)
    size = cfg['data']['image_size']
    size = (size,size) if isinstance(size,int) else tuple(size)
    dataset = ShipGroundingDataset(str(dataset_root/'images/val'),[str(p) for p in paths],image_size=size,random_seed=seed)
    dataset.training_sampling = False
    loader = DataLoader(dataset,batch_size=args.batch_size,num_workers=args.workers,collate_fn=collate)
    amp = cfg['train'].get('use_amp',True) and device.type == 'cuda'
    dtype = getattr(torch,cfg['train'].get('amp_dtype','bf16').replace('bf16','bfloat16').replace('fp16','float16'))
    stats, events, quality_bins = Stats(), Counter(), [Stats() for _ in range(10)]
    samples, seen, rng = [], 0, random.Random(seed)
    with (output/'cases.csv').open('w',encoding='utf-8-sig',newline='') as csv_file, gzip.open(output/'predictions.jsonl.gz','wt',encoding='utf-8') as stream, torch.inference_mode():
        writer = None
        for batch in tqdm(loader,desc='Ranking checks'):
            images,texts,indices = prepare_model_batch(batch,device,False)
            targets = get_raw_targets(batch)
            with torch.autocast(device.type,dtype=dtype,enabled=amp):
                outputs = forward_model_batch(model,images,texts,indices,return_aux=False)
                pred = components(outputs['phrase_matching'],outputs['token_offsets'],outputs['alignment_text_mask'],[t['phrase_char_spans'] for t in targets])
                gt = gt_components(model,outputs,targets)
            pred = [{k:v.cpu() for k,v in row.items()} for row in pred]
            gt = [{k:v.cpu() for k,v in row.items()} for row in gt]
            for b,target in enumerate(targets):
                boxes = outputs['bbox'][b].float().cpu()
                objects = target['boxes'].float().cpu()
                labels = target['phrase_relations'].cpu().T
                quality_logit = outputs['quality_logit'][b].float().cpu().reshape(-1)
                quality = quality_logit.sigmoid()
                overlap = box_iou(boxes,objects)
                actual_iou = overlap.max(1).values if len(objects) else torch.zeros(len(boxes))
                stats.add('quality_iou_absolute_error',(quality-actual_iou).abs())
                for k,bin_stats in enumerate(quality_bins):
                    keep = ((quality*10).long().clamp_max(9)==k)
                    bin_stats.add('predicted_quality',quality[keep])
                    bin_stats.add('actual_iou',actual_iou[keep])
                add_component_stats(stats,'gt_known_pairs',gt[b],labels>=0)
                for p,spans in enumerate(target['phrase_char_spans']):
                    phrase = ''.join(texts[b][start:end] for start,end in spans)
                    rel = labels[:,p]
                    pos,neg = rel==1,rel==0
                    correct = (overlap[:,pos]>=.5).any(1)
                    keep = ~(overlap[:,rel<0]>=.5).any(1) | correct
                    values = {k:v[:,p] for k,v in pred[b].items()}
                    add_component_stats(stats,'pred_eligible_candidates',values,keep)
                    final = fuse_quality_and_phrase_logits(quality_logit,values['full'],fusion=model.score_fusion).sigmoid()
                    cosine_final = fuse_quality_and_phrase_logits(quality_logit,values['base'],fusion=model.score_fusion).sigmoid()
                    full_q,full_hit = top1(values['full'].sigmoid(),correct,keep)
                    cos_q,cos_hit = top1(values['base'].sigmoid(),correct,keep)
                    logit_q,logit_hit = top1(values['full'],correct,keep)
                    _,fused_hit = top1(final,correct,keep)
                    _,cos_fused_hit = top1(cosine_final,correct,keep)
                    _,gt_hit = top1(gt[b]['full'][:,p].sigmoid(),pos,rel>=0)
                    _,gt_cos_hit = top1(gt[b]['base'][:,p].sigmoid(),pos,rel>=0)
                    if not pos.any():
                        full_hit=cos_hit=fused_hit=cos_fused_hit=gt_hit=gt_cos_hit=logit_hit=None
                    detail = diagnose(boxes,final,objects,rel)
                    contrast = bool(pos.any() and neg.any())
                    row = dict(filename=batch['filenames'][b],phrase=phrase,phrase_index=p,
                        semantic_key=batch['regions'][b][p]['semantic_key'],
                        positive_objects=int(pos.sum()),negative_objects=int(neg.sum()),
                        candidate_available=detail['candidate_available'],
                        gt_hit_at_1=gt_hit,gt_cosine_hit_at_1=gt_cos_hit,gt_discriminative=contrast,
                        gt_positive_negative_logit_margin=margin(gt[b]['full'][:,p],pos,neg),
                        text_hit_at_1=full_hit,cosine_text_hit_at_1=cos_hit,
                        logit_text_hit_at_1=logit_hit,sigmoid_top1_changed=logit_q!=full_q if pos.any() else None,
                        hit_at_1=fused_hit,cosine_fused_hit_at_1=cos_fused_hit,
                        text_top1_query=full_q,fused_top1_query=detail['top1_query'],category=detail['category'],
                        mlp_text_transition=transition(cos_hit,full_hit),fusion_transition=transition(full_hit,fused_hit))
                    if writer is None:
                        writer = csv.DictWriter(csv_file,fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    for name in ('gt_hit_at_1','gt_cosine_hit_at_1','text_hit_at_1','cosine_text_hit_at_1','hit_at_1','cosine_fused_hit_at_1','logit_text_hit_at_1','sigmoid_top1_changed'):
                        stats.add(name,row[name])
                    stats.add('gt_discriminative_hit_at_1',gt_hit if contrast else None)
                    stats.add('gt_discriminative_cosine_hit_at_1',gt_cos_hit if contrast else None)
                    stats.add('gt_positive_negative_logit_margin',row['gt_positive_negative_logit_margin'])
                    stats.add('gt_discriminative_fraction',contrast if pos.any() else None)
                    stats.add('candidate_coverage',row['candidate_available'] if pos.any() else None)
                    for name,value in (('mlp_text',transition(cos_hit,full_hit)),('mlp_fused',transition(cos_fused_hit,fused_hit)),('mlp_gt',transition(gt_cos_hit,gt_hit)),('fusion',transition(full_hit,fused_hit))):
                        if value is not None:
                            events[name+'_'+value] += 1
                    stats.add('mlp_text_top1_changed',cos_q!=full_q if pos.any() else None)
                    record = dict(**row,boxes=boxes.tolist(),object_boxes=objects.tolist(),object_ids=target['object_ids'],
                        relations=rel.tolist(),quality_scores=quality.tolist(),alignment_scores=values['full'].sigmoid().tolist(),
                        final_scores=final.tolist(),alignment_logits=values['full'].tolist(),cosine_logits=values['base'].tolist(),
                        cosine=values['cosine'].tolist(),mlp_delta=values['delta'].tolist(),
                        gt_alignment_logits=gt[b]['full'][:,p].tolist(),**{k:v for k,v in detail.items() if k not in row})
                    stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                    eligible = detail['category'] not in ('correct','ignored','no_target') or transition(cos_hit,full_hit) in ('improved','harmed')
                    if eligible and args.examples:
                        seen += 1
                        if len(samples)<args.examples:
                            samples.append(record)
                        else:
                            j = rng.randrange(seen)
                            if j<args.examples:
                                samples[j] = record
            del outputs,pred,gt
    coverage_result = coverage(dataset_root,args.coverage_limit,seed)
    (output/'coverage.json').write_text(json.dumps(coverage_result,ensure_ascii=False,indent=2),encoding='utf-8')
    result = stats.result()
    denominators = {name:sum(events[name+'_'+v] for v in ('improved','harmed','both_correct','both_wrong')) for name in ('mlp_text','mlp_fused','mlp_gt','fusion')}
    summary = dict(checkpoint=checkpoint,images=len(paths),total_images=total_images,partial=len(paths)<total_images,
        gt_roi_semantics={k:v for k,v in result.items() if k.startswith('gt_') or k=='candidate_coverage'},
        pair_mlp_contribution={k:v for k,v in result.items() if 'cosine' in k or 'delta' in k or k=='mlp_text_top1_changed'},
        calibration={k:v for k,v in result.items() if 'probability_' in k or 'sigmoid_derivative' in k or k in ('quality_iou_absolute_error','sigmoid_top1_changed','logit_text_hit_at_1')},
        metrics={k:result.get(k,dict(count=0,mean=None,std=None)) for k in ('hit_at_1','text_hit_at_1','cosine_fused_hit_at_1','cosine_text_hit_at_1')},
        transitions={name:dict(count=denominators[name],events={v:events[name+'_'+v] for v in ('improved','harmed','both_correct','both_wrong')},
                    rates={v:events[name+'_'+v]/denominators[name] if denominators[name] else None for v in ('improved','harmed','both_correct','both_wrong')}) for name in denominators},
        quality_bins=[dict(range=[k/10,(k+1)/10],**s.result()) for k,s in enumerate(quality_bins)],
        hard_negative_coverage=coverage_result,settings=dict(config=cfg,model_config=config,device=str(device),amp=amp,seed=seed),
        notes=['GT ROI changes both crops and candidate set; its score is not directly comparable to predicted-query Hit@1.',
               'GT discriminative metrics include only phrases with a known positive and a known negative object.',
               'Predicted rankings use raw Main candidates, IoU >= 0.5 and the evaluator ignore policy; no NMS.',
               'MLP ablation holds embeddings, scale, bias, boxes and quality fixed; it removes only the residual term, not its training history.',
               'Predicted saturation includes eligible background candidates; GT saturation includes known object-phrase pairs only.',
               'Text metrics rank sigmoid probabilities like the evaluator. Exact arithmetic sigmoid is monotonic, but finite-precision saturation can create ties; raw-logit ranking is reported separately.',
               'Derivative and fusion diagnostics do not prove a training-gradient bottleneck.',
               'Coverage counts annotation structure, not actual sampler exposure or visual difficulty. Same-color attributes are copied from labels.',
               'Quality bins compare predicted quality with maximum IoU to annotated objects, not classification calibration.'])
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    gallery(samples,dataset_root,output,coverage_result)
    print(json.dumps(dict(output=str(output),images=len(paths),partial=summary['partial'],metrics=summary['metrics'],transitions=summary['transitions']),ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
