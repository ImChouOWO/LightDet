"""Evaluate gRefCOCO using the unchanged official MDETR GREC evaluator."""
import argparse
import contextlib
import io
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from units.tool.card import VisionTextModel
from units.model.pipeline.data import GRefCOCODataset,build_dataloader
from units.model.tool.config import resolve
from units.model.tool.runtime import prepare_model_batch
from units.model.tool.official_grec import RefExpEvaluator


class ExpressionAnnotations:
    """Present each sentence as an evaluation image, as in official converted annotations."""
    def __init__(self):
        self.imgs={}
        self.annotations={}

    def add(self,key,annotations):
        if key in self.imgs:raise ValueError('Duplicate expression identifier')
        self.imgs[key]={}
        self.annotations[key]=annotations

    def loadImgs(self,key):
        return [self.imgs[key]]

    def getAnnIds(self,imgIds):
        return [(imgIds,index) for index in range(len(self.annotations[imgIds]))]

    def loadAnns(self,ids):
        return [self.annotations[key][index] for key,index in ids]


@torch.inference_mode()
def evaluate_official(model,tokenizer,config,device,split,batch_size,workers,limit=None):
    dataset=GRefCOCODataset(resolve(config['data']['dataset_dir']),split,config['data']['image_size'])
    if limit:
        dataset.entries=dataset.entries[:limit]
        dataset.groups=dataset.groups[:limit]
        dataset.image_ids=dataset.image_ids[:limit]
    loader=build_dataloader(dataset,batch_size,workers)
    annotations=ExpressionAnnotations();predictions={}
    size=config['data']['image_size']
    for batch in tqdm(loader,desc=f'Official GREC [{split}]'):
        images,tokens,targets=prepare_model_batch(batch,tokenizer,device,config['data']['text_max_length'],move_targets=False)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda' and config['train']['amp']):
            outputs=model(images,tokens,image_indices=batch['image_indices'].to(device),
                valid_sizes=torch.stack([target['valid_size'] for target in targets]).to(device))
        boxes_cpu=outputs['boxes'].float().cpu();scores_cpu=outputs['scores'].float().cpu()
        for boxes,scores,target in zip(boxes_cpu,scores_cpu,targets):
            key=int(target['sent_id'])
            info=dataset.images[target['image_id']]
            rw,rh=(target['valid_size']*size).round().tolist()
            scale=boxes.new_tensor([info['width']/rw,info['height']/rh]*2)
            original_boxes=boxes*size*scale
            if not torch.isfinite(original_boxes).all() or not torch.isfinite(scores).all():
                raise ValueError('Non-finite prediction')
            if not (original_boxes[:,2:]>=original_boxes[:,:2]).all():
                raise ValueError('Invalid prediction coordinates')
            gt=[dict(bbox=dataset.annotations[obj]['bbox'],category_id=1) for obj in target['object_ids']]
            annotations.add(key,gt or [dict(bbox=[0,0,1,1],category_id=-1)])
            # No clipping, NMS, score rescaling, or additional Top-K truncation.
            predictions[key]=dict(boxes=original_boxes,scores=scores)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator=RefExpEvaluator(annotations,('bbox',))
        evaluator.update(predictions)
        values=evaluator.summarize()
    return dict(grec_precision_f1=values['F1_score'],n_acc=values['N_acc'],t_acc=values['T_acc'],
        official_results=values,expression_count=len(predictions),
        positive_count=sum(dataset.groups.count(g) for g in (1,2)),no_target_count=dataset.groups.count(0),
        positive_precision_f1=(values['F1_score']*len(predictions)-values['N_acc']*dataset.groups.count(0))/max(1,sum(dataset.groups.count(g) for g in (1,2))))


def print_results(result):
    columns=('Split','Expressions','GREC Pr@(F1=1)','N-Acc','T-Acc','Positive-Pr')
    widths=(8,13,18,12,12,14)
    def row(values):return ' '.join(str(value).ljust(width) for value,width in zip(values,widths))
    print('\n'+row(columns))
    print('-'*sum(widths))
    for split,metric in result['splits'].items():
        values=[split,metric['expression_count']]+[f"{metric[name]:.2%}" for name in
            ('grec_precision_f1','n_acc','t_acc','positive_precision_f1')]
        print(row(values))
    print(flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',required=True)
    parser.add_argument('--splits',nargs='+',default=['val','testA','testB'],choices=['val','testA','testB'])
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--max-expressions',type=int)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.batch_size<1 or args.workers<0 or (args.max_expressions is not None and args.max_expressions<1):
        parser.error('Invalid batch size, workers, or expression limit')
    output=resolve(args.output)
    if not output.parent.is_dir():raise FileNotFoundError(output.parent)
    if output.exists():raise FileExistsError(output)
    device=torch.device(args.device);torch.set_num_threads(4)
    if device.type=='cuda':torch.cuda.set_device(device)
    weights=resolve(args.weights)
    checkpoint=torch.load(weights,map_location='cpu',weights_only=False)
    if checkpoint['schema']!='lightdet.grefcoco.queries.v3':raise ValueError('Expected text-guided query checkpoint (v3)')
    settings=checkpoint['model_config'];config=checkpoint['config']
    model=VisionTextModel(**dict(settings,backbone_weights=None)).to(device).eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    tokenizer=AutoTokenizer.from_pretrained(resolve(settings['encoder_path']),local_files_only=True)
    result=dict(weights=str(weights),epoch=checkpoint['epoch'],partial=args.max_expressions is not None,
        protocol=dict(source='https://github.com/henghuiding/gRefCOCO/blob/41a8f008006bb1fb0f1df2547f9477bb97e36593/mdetr/datasets/refexp.py',
            evaluator_sha256=hashlib.sha256((ROOT/'units/model/tool/official_grec.py').read_bytes()).hexdigest(),
            score_threshold=.7,overlap='GIoU',overlap_threshold=.5,matching='greedy maximum overlap one-to-one',
            nms=False,coordinates='original image xyxy',candidate_count=settings['num_object_queries']),splits={})
    print(f"\n[Official GREC] Main branch\n  Checkpoint : {weights.name}\n  Epoch      : {result['epoch']}\n  Device     : {device}\n  Protocol   : score >= 0.7 | GIoU >= 0.5 | greedy matching | NMS off\n  Splits     : {', '.join(args.splits)}\n  Output     : {output}\n",flush=True)
    for split in args.splits:
        result['splits'][split]=evaluate_official(model,tokenizer,config,device,split,args.batch_size,args.workers,args.max_expressions)
        temporary=output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,indent=2));temporary.replace(output)
        print_results(result)
    print(f'Saved: {output}',flush=True)


if __name__=='__main__':main()
