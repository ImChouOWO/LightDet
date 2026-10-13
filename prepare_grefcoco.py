"""Download official gRefCOCO annotations, referenced COCO images, and MiniLM."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.parse import quote
import httpx

CLIENT = httpx.Client(follow_redirects=True,timeout=90,
    limits=httpx.Limits(max_connections=64,max_keepalive_connections=64),
    headers={"User-Agent":"LightDet-gRef","Accept-Encoding":"identity"})

ROOT = Path(__file__).resolve().parent


def download(url, path):
    if path.is_file() and path.stat().st_size:
        return
    for attempt in range(5):
        temporary = path.with_name(path.name + '.part')
        try:
            with CLIENT.stream('GET',url) as source, temporary.open('wb') as target:
                source.raise_for_status()
                expected = source.headers.get('Content-Length')
                for block in source.iter_bytes(1024*1024):
                    target.write(block)
            if expected and temporary.stat().st_size != int(expected):
                raise IOError('Incomplete download')
            temporary.replace(path)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    labels, images, encoder = ROOT/'datasets/labels', ROOT/'datasets/images', ROOT/'units/model/bert'
    for folder in (labels, images, encoder):
        if not folder.is_dir():
            raise FileNotFoundError(f'Expected existing project folder: {folder}')
    for name in ('grefs(unc).json', 'instances.json'):
        download('https://huggingface.co/datasets/FudanCVL/gRefCOCO/resolve/main/'+quote(name), labels/name)
    # Use the model backbone; SentenceTransformer pooling is done by our text branch.
    filenames = ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'sentencepiece.bpe.model')
    marker = encoder/'minilm_source.json'
    if not marker.exists():
        for name in filenames:
            (encoder/name).unlink(missing_ok=True)
    for name in filenames:
        download('https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/'+name, encoder/name)
        print('Encoder:', name, flush=True)
    marker.write_text(json.dumps({'model_id': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'}))
    refs = json.loads((labels/'grefs(unc).json').read_text())
    instances = json.loads((labels/'instances.json').read_text())
    ids = {r['image_id'] for r in refs}
    names = [im['file_name'] for im in instances['images'] if im['id'] in ids]
    def fetch(name):
        download('https://s3.amazonaws.com/images.cocodataset.org/train2014/'+name, images/name)
    with ThreadPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(fetch, names), 1):
            if i % 250 == 0 or i == len(names):
                print(f'COCO images: {i}/{len(names)}', flush=True)
    print(json.dumps(dict(images=len(names), references=len(refs), splits=sorted({r['split'] for r in refs}))), flush=True)


if __name__ == '__main__':
    main()
