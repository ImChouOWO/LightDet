"""Export a reproducible visual sample of validation errors without modifying labels."""
import argparse
import gzip
import html
import json
import random
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def pixels(box, size):
    return tuple(round(v * size[i % 2]) for i, v in enumerate(box))


def crop(image, box, path):
    x1, y1, x2, y2 = pixels(box, image.size)
    pad = max(4, round(max(x2-x1, y2-y1) * .15))
    image.crop((max(0, x1-pad), max(0, y1-pad), min(image.width, x2+pad), min(image.height, y2+pad))).save(path, quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--val-dir', help='Defaults to latest validation predictions')
    parser.add_argument('--dataset-dir', help='Defaults to dataset recorded in summary.json')
    parser.add_argument('--output-dir', help='Must be a new directory')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=49)
    parser.add_argument('--category', choices=['wrong_object', 'localization_failure', 'background_or_unannotated', 'all_errors'], default='wrong_object')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    if args.val_dir:
        source = resolve(args.val_dir)
    else:
        sources = list((ROOT / 'units/model/runs/train').glob('*_val_*/predictions.jsonl.gz'))
        if not sources:
            parser.error('No validation predictions found; pass --val-dir')
        source = max(sources, key=lambda p: p.stat().st_mtime).parent
    summary = json.loads((source / 'summary.json').read_text())
    dataset = resolve(args.dataset_dir or summary['settings']['config']['data']['dataset_dir'])
    output = resolve(args.output_dir) if args.output_dir else source / ('audit_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    rng, samples, total = random.Random(args.seed), [], 0
    # Reservoir sampling gives each eligible phrase equal probability with bounded memory.
    with gzip.open(source / 'predictions.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            eligible = row['category'] == args.category if args.category != 'all_errors' else row['category'] not in ('correct', 'ignored', 'no_target')
            if not eligible:
                continue
            total += 1
            if len(samples) < args.count:
                samples.append(row)
            else:
                index = rng.randrange(total)
                if index < args.count:
                    samples[index] = row
    if not samples:
        parser.error('No matching cases found')
    output.mkdir(parents=True, exist_ok=False)
    assets = output / 'images'
    assets.mkdir()
    cards, records = [], []
    esc = lambda value: html.escape(str(value), quote=True)
    for number, row in enumerate(samples, 1):
        label_path = dataset / 'labels/val' / (Path(row['filename']).stem + '.json')
        label = json.loads(label_path.read_text()) if label_path.exists() else {}
        attributes = {obj['id']: obj.get('attributes', {}) for obj in label.get('objects', [])}
        phrase = next((p for p in label.get('grounding', {}).get('phrases', []) if p['phrase'] == row['phrase']), {})
        with Image.open(dataset / 'images/val' / row['filename']) as opened:
            image = opened.convert('RGB')
        canvas = image.copy()
        canvas.thumbnail((1600, 1600))
        draw = ImageDraw.Draw(canvas)
        table, crops = [], []
        for obj_id, box, relation in zip(row['object_ids'], row['object_boxes'], row['relations']):
            color = '#00c853' if relation == 1 else '#ffad32' if relation == 0 else '#aaaaaa'
            coords = pixels(box, canvas.size)
            draw.rectangle(coords, outline=color, width=3)
            draw.text((coords[0], max(0, coords[1]-12)), obj_id, fill=color, stroke_width=1, stroke_fill='black')
            table.append(f'<tr><td>{esc(obj_id)}</td><td>{relation}</td><td>{esc(json.dumps(attributes.get(obj_id, {}), ensure_ascii=False))}</td></tr>')
            if relation == 1:
                name = f'{number:03d}_gt_{len(crops)}.jpg'
                crop(image, box, assets / name)
                crops.append(f'<figure><img loading="lazy" src="images/{name}"><figcaption>GT：{esc(obj_id)}</figcaption></figure>')
        top, scores = row.get('top1_query'), {}
        if top is not None:
            coords = pixels(row['boxes'][top], canvas.size)
            draw.rectangle(coords, outline='#ff3355', width=4)
            draw.text((coords[0], coords[1]), 'TOP-1', fill='white', stroke_width=2, stroke_fill='#a00020')
            name = f'{number:03d}_top1.jpg'
            crop(image, row['boxes'][top], assets / name)
            crops.insert(0, f'<figure><img loading="lazy" src="images/{name}"><figcaption>Top-1</figcaption></figure>')
            scores = {key: round(row[key][top], 6) for key in ['quality_scores', 'alignment_scores', 'final_scores']}
        canvas.save(assets / f'{number:03d}.jpg', quality=90)
        records.append(dict(case_id=number, **row, semantic_key=phrase.get('semantic_key'), attributes=attributes, review='', note=''))
        cards.append(f'''<section><h2>#{number} · {esc(row['phrase'])}</h2>
<p>{esc(row['filename'])} · {esc(row['category'])} · 正確候選存在：{esc(row.get('candidate_available'))}</p>
<p>語意定義：{esc(phrase.get('semantic_key', '未提供'))}<br>Top-1 分數：{esc(json.dumps(scores))}</p>
<a href="images/{number:03d}.jpg" target="_blank"><img class="overview" loading="lazy" src="images/{number:03d}.jpg"></a>
<div class="crops">{''.join(crops)}</div>
<details><summary>物件關係與原始屬性</summary><table><tr><th>物件</th><th>1 正／0 負／-1 忽略</th><th>屬性</th></tr>{''.join(table)}</table></details>
<label>人工判定 <select><option value="">未判定</option><option>標記錯誤</option><option>描述模糊</option><option>影像難辨</option><option>模型判錯</option><option>其他</option></select></label>
<input placeholder="備註" aria-label="備註"></section>''')
    manifest = dict(source=str(source), checkpoint=summary.get('checkpoint'), category=args.category, seed=args.seed,
                    eligible_count=total, sampled_count=len(samples), sampling_unit='phrase; images may repeat', cases=records)
    (output / 'cases.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    payload = json.dumps(manifest, ensure_ascii=False).replace('<', '\\u003c')
    page = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>Grounding 抽查</title>
<style>body{font:16px system-ui;max-width:1200px;margin:24px auto;padding:16px;background:#f3f5f7;color:#18222c}section{background:white;padding:24px;margin:24px 0;border-radius:12px}.overview{width:100%}.crops{display:flex;flex-wrap:wrap;gap:16px}.crops img{max-width:340px;max-height:240px;min-width:120px;object-fit:contain}figure{margin:12px 0}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ddd;text-align:left}select,input,button{padding:10px;margin:12px 8px 0 0}input{width:45%}header{position:sticky;top:0;background:#f3f5f7;padding:12px;z-index:1}</style>
<header><b>Grounding 抽查</b> <button id="export">匯出判定</button><span id="status"></span></header>
<p>綠框：符合描述的 GT；橘框：負例物件；灰框：忽略；紅框：Top-1。點擊全圖可放大。局部圖取自原圖，不代表縮放後模型仍能辨識。</p>
<p>等機率抽取描述，同張圖可能出現多次。判定暫存於目前瀏覽器；完成後請匯出保存。</p>'''
    page += ''.join(cards) + '<script>const data=' + payload + ''';
const key='grounding-audit:'+location.href;
try{const saved=JSON.parse(localStorage.getItem(key)||'null');if(saved)saved.forEach((v,i)=>{data.cases[i].review=v.review;data.cases[i].note=v.note;});}catch(e){}
function update(){document.querySelector('#status').textContent=data.cases.filter(c=>c.review).length+'/'+data.cases.length+' 已判定';try{localStorage.setItem(key,JSON.stringify(data.cases.map(({review,note})=>({review,note}))));}catch(e){document.querySelector('#status').textContent+='（暫存不可用，請匯出）';}}
document.querySelectorAll('section').forEach((section,i)=>{const select=section.querySelector('select'),input=section.querySelector('input');select.value=data.cases[i].review;input.value=data.cases[i].note;select.onchange=input.oninput=()=>{data.cases[i].review=select.value;data.cases[i].note=input.value;update();};});
document.querySelector('#export').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='reviewed_cases.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};update();
</script></html>'''
    (output / 'index.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(output=str(output), sampled=len(samples), eligible=total, report=str(output / 'index.html')), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
