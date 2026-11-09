"""Evaluate a checkpoint using the explicitly configured dataset protocol."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from units.tool.card import VisionTextModel
from units.model.pipeline.data import build_dataset,build_dataloader
from units.model.tool.config import load_config,resolve
from units.model.tool.evaluation import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',required=True)
    parser.add_argument('--split',default='val',choices=['val','test','testA','testB'])
    parser.add_argument('--data-config',help='Explicit dataset settings; otherwise use checkpoint data config')
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--max-expressions',type=int,default=None)
    parser.add_argument('--output',default='units/model/runs/train/lightdet_gref_validation.json')
    args=parser.parse_args()
    device=torch.device(args.device);torch.set_num_threads(4)
    checkpoint=torch.load(resolve(args.weights),map_location='cpu',weights_only=False)
    if checkpoint['schema']!='lightdet.grefcoco.queries.v3':raise ValueError('Expected text-guided query checkpoint (v3)')
    model_config=checkpoint['model_config'];config=checkpoint['config']
    if args.data_config:config['data']=load_config(args.data_config)['data']
    if 'dataloader' not in config['data']:
        parser.error('Specify the dataset explicitly with --data-config; data.dataloader is required')
    # The checkpoint contains all visual weights; do not load ImageNet weights again.
    constructor=dict(model_config,backbone_weights=None)
    model=VisionTextModel(**constructor).to(device)
    model.load_state_dict(checkpoint['model'],strict=True)
    tokenizer=AutoTokenizer.from_pretrained(resolve(model_config['encoder_path']),local_files_only=True)
    dataset=build_dataset(config['data'],args.split)
    if args.max_expressions is not None:
        if args.max_expressions<1:parser.error('--max-expressions must be positive')
        dataset.entries=dataset.entries[:args.max_expressions]
        dataset.groups=dataset.groups[:args.max_expressions]
        dataset.image_ids=dataset.image_ids[:args.max_expressions]
    loader=build_dataloader(dataset,config['train']['batch_size'],args.workers)
    result=dict(weights=args.weights,epoch=checkpoint['epoch'],dataloader=config['data']['dataloader'],split=args.split,partial=args.max_expressions is not None,
                metrics=evaluate(model,loader,tokenizer,device,config))
    output=resolve(args.output)
    if not output.parent.is_dir():raise FileNotFoundError('Output parent must already exist')
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
