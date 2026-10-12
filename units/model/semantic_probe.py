"""Inspect phrase attention and ROI contributions with fixed-box semantic interventions."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import gzip
import html
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.ops import box_iou

from units.model.cards.phrase_matching import phrase_matrices
from units.model.pipeline.data import ShipGroundingDataset
from units.model.tool.config import load_model_config, load_train_config
from units.model.tool.runtime import compact_grounding_collate_fn, prepare_model_batch, forward_model_batch, get_raw_targets, char_spans_to_token_mask
from units.model.train import load_weights_only
from units.model.val import diagnose, resolve
from units.tool.card import VisionTextModel


def sample_records(path, count, seed):
    rng, selected, seen = random.Random(seed), [], 0
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if row['category'] != 'wrong_object' or not row['candidate_available']:
                continue
            seen += 1
            if len(selected) < count:
                selected.append(row)
            else:
                index = rng.randrange(seen)
                if index < count:
                    selected[index] = row
    return selected, seen


def normalize_map(values):
    values = values.detach().float().cpu()
    if not torch.isfinite(values).all():
        raise FloatingPointError('Nonfinite attribution')
    maximum = values.max()
    return values / maximum if maximum > 0 else torch.zeros_like(values)


def masked_roi(tokens, position, query, cells):
    result = tokens.detach().clone()
    # Remove projected visual evidence while preserving spatial embeddings.
    result[0, query, cells] = position[0, 0, cells].to(result)
    return result


@contextmanager
def capture_roi(stage):
    saved, attention, original = {}, {'roi': []}, stage._roi_tokens
    methods = []
    def roi(*args, **kwargs):
        result = original(*args, **kwargs)
        tokens, quality = result
        if saved:
            raise RuntimeError('Probe expects one Main ROI extraction without Aux')
        tokens = tokens.detach().requires_grad_(True)
        saved.update(tokens=tokens, quality=quality.detach())
        return tokens, quality
    stage._roi_tokens = roi
    for layer in stage.layers:
        for kind, module in (('roi', layer.roi_attention),):
            method = module.forward
            methods.append((module, method))
            def forward(*args, _method=method, _kind=kind, **kwargs):
                kwargs.update(need_weights=True, average_attn_weights=False)
                output, weights = _method(*args, **kwargs)
                attention[_kind].append(weights.detach())
                return output, weights
            module.forward = forward
    try:
        yield saved, attention
    finally:
        stage._roi_tokens = original
        for module, method in methods:
            module.forward = method


def cached_scores(model, outputs, tokens, text, text_mask, offsets, valid, spans):
    stage, original = model.score_transformer, model.score_transformer._roi_tokens
    stage._roi_tokens = lambda *args, **kwargs: tokens
    try:
        boxes = outputs['bbox'].detach()
        query = model._condition_score_queries(boxes)
        score, _ = stage._forward_group(outputs['fpn_features'], query,
            boxes, outputs['input_image_size'], stage.MAIN_GROUP)
        matching = model.head.predict_phrase_matching(score, text, valid)
        return phrase_matrices(matching, offsets, valid, [spans])[0][0].float()
    finally:
        stage._roi_tokens = original


def color_counterfactual(caption, spans):
    if len(spans) != 1:
        return None
    start, end = spans[0]
    phrase = caption[start:end]
    colors = ('紅色', '綠色', '藍色', '黃色', '白色', '黑色', '灰色', '橙色')
    found = [color for color in colors if color in phrase]
    if len(found) != 1:
        return None
    source = found[0]
    replacement = '藍色' if source != '藍色' else '紅色'
    changed = phrase.replace(source, replacement)
    return caption[:start] + changed + caption[end:], changed, source, replacement


def heat_overlay(crop, values):
    heat = np.asarray(normalize_map(values))
    colors = np.stack((heat, .25 * (1-heat), 1-heat), -1)
    overlay = Image.fromarray((colors*255).astype(np.uint8)).resize(crop.size, Image.Resampling.BILINEAR)
    return Image.blend(crop, overlay, .45)


def font_for(path):
    if path:
        return ImageFont.truetype(path, 18), True
    fonts = list(Path('/usr/share/fonts').rglob('*CJK*.ttc')) + list(Path('/usr/share/fonts').rglob('*cjk*.ttc'))
    if fonts:
        return ImageFont.truetype(str(fonts[0]), 18), True
    return ImageFont.load_default(), False


def render(image, boxes, candidates, phrase, weights, output, font, chinese):
    canvas = Image.new('RGB', (1000, 950), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 10), phrase if chinese else 'Phrase and token details: see case JSON', fill='black', font=font)
    overview = image.resize((460, 460))
    d = ImageDraw.Draw(overview)
    for candidate in candidates:
        box = boxes[candidate['query']]
        d.rectangle(tuple(float(x)*460 for x in box), outline='red' if candidate['role']=='wrong' else 'lime', width=3)
    canvas.paste(overview, (15, 45))
    draw.text((500, 45), 'Phrase pooling weights (shared across ROIs)', fill='black', font=font)
    for i, token in enumerate(weights[:15]):
        y = 78 + i*27
        label = token['text'] if chinese else 'token '+str(token['index'])
        draw.text((500, y), label, fill='black', font=font)
        width = int(260*token['pooling_weight'])
        draw.rectangle((650,y+3,650+width,y+17), fill='steelblue')
        draw.text((920,y), f"{token['pooling_weight']:.2f}", fill='black', font=font)
    for row, candidate in enumerate(candidates):
        q = candidate['query']
        pixel = [float(x)*image.width for x in boxes[q]]
        pixel[2], pixel[3] = max(pixel[0]+1,pixel[2]), max(pixel[1]+1,pixel[3])
        crop = image.crop(tuple(pixel)).resize((190,150))
        y = 535 + row*200
        draw.text((15,y-25), f"{candidate['role']} Q{q}: alignment={candidate['alignment']:.3f}", fill='black', font=font)
        panels = (crop, heat_overlay(crop, torch.tensor(candidate['roi_attention'])),
                  heat_overlay(crop, torch.tensor(candidate['gradient_x_feature_abs'])))
        for column, (panel,label) in enumerate(zip(panels, ('ROI crop','ROI attention','|gradient x feature|'))):
            x = 15 + column*220
            canvas.paste(panel, (x,y))
            draw.text((x,y+151),label,fill='black',font=font)
        draw.text((680,y+10), f"top-cell drop: {candidate['salient_drop']:.3f}", fill='black', font=font)
        draw.text((680,y+40), f"random drop: {candidate['random_drop_mean']:.3f}", fill='black', font=font)
        draw.text((680,y+70), 'Alignment logit, fixed boxes', fill='black', font=font)
    canvas.save(output)


def probe(model, dataset, record, device, fraction, trials, seed):
    batch = compact_grounding_collate_fn([dataset[0]])
    images, captions, indices = prepare_model_batch(batch, device, False)
    target = get_raw_targets(batch)[0]
    p = record['phrase_index']
    spans = target['phrase_char_spans']
    phrase = ''.join(captions[0][start:end] for start,end in spans[p])
    if phrase != record['phrase']:
        raise ValueError('Saved validation phrase differs from current annotation')
    with capture_roi(model.score_transformer) as (saved, attention):
        outputs = forward_model_batch(model, images, captions, indices, return_aux=False)
    logits = phrase_matrices(outputs['phrase_matching'], outputs['token_offsets'],
        outputs['alignment_text_mask'], [spans])[0][0].float()
    alignment = logits[:,p].sigmoid()
    quality = outputs['quality_logit'][0,:,0].float().sigmoid()
    final = (alignment*quality).clamp_min(1e-12).sqrt()
    boxes = outputs['bbox'][0].detach().float()
    objects = target['boxes'].to(boxes)
    relations = target['phrase_relations'][p].to(device)
    detail = diagnose(boxes, final.detach(), objects, relations)
    if detail['category'] != 'wrong_object' or not detail['candidate_available']:
        return dict(filename=record['filename'],phrase=phrase,skipped=True,reason='FP32 replay is not a wrong-object error with a positive candidate',replay=detail), None
    correct = (box_iou(boxes, objects[relations==1])>=.5).any(1)
    ids = torch.where(correct)[0]
    positive = int(ids[final[ids].argmax()])
    wrong = detail['top1_query']
    count, roi_size = boxes.shape[0], model.score_transformer.roi_size
    roi_attention = torch.stack([a.mean(1).squeeze(1) for a in attention['roi']]).mean(0).reshape(count,roi_size,roi_size)
    tokens, position = saved['tokens'], model.score_transformer.roi_position_embeddings
    valid = outputs['alignment_text_mask']
    mask = char_spans_to_token_mask(outputs['token_offsets'][0], spans[p], valid[0])
    pool = outputs['phrase_matching']['token_weights'][0].masked_fill(~mask,-torch.inf).softmax(0)
    offsets = outputs['token_offsets'][0].tolist()
    token_weights = [dict(index=i,text=captions[0][start:end],span=[start,end],pooling_weight=float(pool[i].detach()))
        for i,(start,end) in enumerate(offsets) if mask[i]]
    cases = []
    with torch.no_grad():
        replay = cached_scores(model, outputs, tokens.detach(), outputs['text_token'], outputs['text_mask'], outputs['token_offsets'], valid, spans)
    torch.testing.assert_close(replay, logits.detach(), rtol=2e-4, atol=2e-4)
    for role, q in (('wrong',wrong),('positive',positive)):
        gradient = torch.autograd.grad(logits[q,p],tokens,retain_graph=True)[0][0,q]
        heat = (gradient*(tokens[0,q]-position[0,0].to(tokens))).sum(-1).abs().detach()
        cells = heat.topk(max(1,math.ceil(fraction*heat.numel()))).indices
        with torch.no_grad():
            altered = cached_scores(model,outputs,masked_roi(tokens,position,q,cells),outputs['text_token'],outputs['text_mask'],outputs['token_offsets'],valid,spans)
            other = torch.arange(count,device=device)!=q
            torch.testing.assert_close(altered[other],replay[other],rtol=2e-4,atol=2e-4)
            salient_drop = float(logits[q,p].detach()-altered[q,p])
            generator = torch.Generator(device='cpu').manual_seed(seed+q)
            random_drops = []
            for _ in range(trials):
                random_cells = torch.randperm(heat.numel(),generator=generator)[:len(cells)].to(device)
                altered = cached_scores(model,outputs,masked_roi(tokens,position,q,random_cells),outputs['text_token'],outputs['text_mask'],outputs['token_offsets'],valid,spans)
                random_drops.append(float(logits[q,p].detach()-altered[q,p]))
        cases.append(dict(role=role,query=q,alignment=float(alignment[q].detach()),quality=float(quality[q]),
            final=float(final[q].detach()),alignment_logit=float(logits[q,p].detach()),
            roi_attention=roi_attention[q].cpu().tolist(),gradient_x_feature_abs=heat.reshape(roi_size,roi_size).cpu().tolist(),
            masked_cells=cells.cpu().tolist(),salient_drop=salient_drop,random_drops=random_drops,random_drop_mean=float(np.mean(random_drops))))
    counterfactual = color_counterfactual(captions[0],spans[p])
    color_result = None
    if counterfactual:
        caption, changed, source, replacement = counterfactual
        with torch.no_grad():
            text = model.text_model([caption])
            encoded = model.text_model.tokenizer([caption],padding='max_length',truncation=True,
                max_length=model.num_text,return_offsets_mapping=True,return_tensors='pt')
            changed_offsets = encoded['offset_mapping'].to(device)
            changed_valid = text['text_mask'].bool() & (changed_offsets[...,1]>changed_offsets[...,0])
            changed_logits = cached_scores(model,outputs,tokens.detach(),text['text_tokens'],text['text_mask'],changed_offsets,changed_valid,spans)
        color_result = dict(phrase=changed,source=source,replacement=replacement,
            wrong_logit_change=float(changed_logits[wrong,p]-logits[wrong,p].detach()),
            positive_logit_change=float(changed_logits[positive,p]-logits[positive,p].detach()))
    report = dict(filename=record['filename'],phrase=phrase,phrase_index=p,skipped=False,
        saved_top1=record['top1_query'],replay=detail,tokens=token_weights,candidates=cases,color_counterfactual=color_result)
    image = Image.fromarray((images[0].detach().cpu().permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8))
    return report,(image,boxes.cpu().tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-dir')
    parser.add_argument('--weights')
    parser.add_argument('--config')
    parser.add_argument('--model-config')
    parser.add_argument('--output-dir')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--random-trials', type=int, default=5)
    parser.add_argument('--mask-fraction', type=float, default=.25)
    parser.add_argument('--seed', type=int, default=49)
    parser.add_argument('--font')
    args = parser.parse_args()
    if args.samples<1 or args.random_trials<1 or not 0<args.mask_fraction<=1:
        parser.error('Samples/trials must be positive and mask fraction must be in (0,1]')
    cfg = load_train_config(args.config)
    run = resolve(cfg['log']['save_dir'])
    if args.validation_dir:
        validation = resolve(args.validation_dir)
    else:
        summaries = sorted(run.parent.glob(run.name+'_val_*/summary.json'),key=lambda p:p.stat().st_mtime)
        if not summaries:
            raise FileNotFoundError('No validation summary found; specify --validation-dir')
        validation = summaries[-1].parent
    summary = json.loads((validation/'summary.json').read_text())
    config = load_model_config(args.model_config)['model'] if args.model_config else summary['settings']['model_config']
    weights = resolve(args.weights or summary['weights'])
    device = torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    kwargs = {k:v for k,v in config.items() if k not in ('backbone','fpn','image_projector','precomputed_bert_path')}
    model = VisionTextModel(backbone_config=config['backbone'],fpn_config=config['fpn'],
        image_projector_config=config['image_projector'],precomputed_bert_path=None,**kwargs).to(device).eval()
    checkpoint = load_weights_only(str(weights),model,None,device,prefer_ema=summary['checkpoint']['source']=='ema')
    model.requires_grad_(False)
    selected, eligible_count = sample_records(validation/'predictions.jsonl.gz',args.samples,args.seed)
    output = resolve(args.output_dir) if args.output_dir else validation.parent/(run.name+'_semantic_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=False)
    data_root = resolve(cfg['data']['dataset_dir'])
    size = cfg['data']['image_size']
    font,chinese = font_for(args.font)
    reports=[]
    for i,record in enumerate(selected):
        label = data_root/'labels/val'/Path(record['filename']).with_suffix('.json')
        dataset = ShipGroundingDataset(str(data_root/'images/val'),[str(label)],image_size=(size,size),random_seed=args.seed)
        dataset.training_sampling=False
        report,visual = probe(model,dataset,record,device,args.mask_fraction,args.random_trials,args.seed)
        name=f'case_{i:03d}'
        if visual:
            render(*visual,report['candidates'],report['phrase'],report['tokens'],output/(name+'.png'),font,chinese)
            report['image']=name+'.png'
        (output/(name+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        reports.append(report)
        print(f'{i+1}/{len(selected)} {record["filename"]}: '+('skipped' if report['skipped'] else 'saved'),flush=True)
    active=[r for r in reports if not r['skipped']]
    totals=dict(checkpoint=checkpoint,validation_dir=str(validation),requested=args.samples,eligible_records=eligible_count,
        completed=len(active),skipped=len(reports)-len(active),seed=args.seed,roi_size=model.score_roi_size,
        mask_fraction=args.mask_fraction,random_trials=args.random_trials,dtype='float32',
        image_size=size,weights_override=str(weights)!=summary['weights'],
        notes=['Attention and gradients are diagnostics, not proof of causality.',
               'Heatmaps are normalized separately; raw maps and scores are in JSON.',
               'Ablation removes projected ROI features, not image pixels. Boxes and text are fixed.',
               'Color substitution keeps ROI/boxes fixed and updates text; no counterfactual GT is assumed.',
               'FP32 replay can differ from BF16 validation; changed error categories are skipped.',
               'Phrase pooling weights are shared across candidate ROIs.'])
    for role in ('wrong','positive'):
        candidates=[c for r in active for c in r['candidates'] if c['role']==role]
        totals[role]=dict(mean_salient_drop=float(np.mean([c['salient_drop'] for c in candidates])),
            mean_random_drop=float(np.mean([c['random_drop_mean'] for c in candidates]))) if candidates else None
    (output/'summary.json').write_text(json.dumps(totals,ensure_ascii=False,indent=2),encoding='utf-8')
    panels=''.join('<article><p>'+html.escape(r['filename']+' — '+r['phrase'])+'</p><a href="case_'+f'{i:03d}'+'.json">JSON</a><br><img width="1000" src="'+r['image']+'"></article>' for i,r in enumerate(reports) if not r['skipped'])
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Semantic diagnostics</title><h1>Semantic diagnostics</h1><p>Fixed-box feature interventions; attention is not a causal explanation. See summary.json.</p>'+panels,encoding='utf-8')
    print(json.dumps(dict(output=str(output),completed=len(active),skipped=totals['skipped']),indent=2))


if __name__=='__main__':
    main()
