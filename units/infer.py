"""Ground one or more Chinese/English expressions in an image."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from collections import OrderedDict
import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer
from units.tool.card import VisionTextModel
from units.model.tool.config import resolve
from units.model.tool.runtime import select_predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',required=True)
    parser.add_argument('--image',required=True)
    parser.add_argument('--text',nargs='+',required=True)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--output',default=None)
    args=parser.parse_args()
    device=torch.device(args.device)
    checkpoint=torch.load(resolve(args.weights),map_location='cpu',weights_only=False)
    if checkpoint['schema']!='lightdet.grefcoco.queries.v3':raise ValueError('Expected text-guided query checkpoint (v3)')
    config,model_config=checkpoint['config'],checkpoint['model_config']
    model=VisionTextModel(**dict(model_config,backbone_weights=None)).to(device).eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    tokenizer=AutoTokenizer.from_pretrained(resolve(model_config['encoder_path']),local_files_only=True)
    size=config['data']['image_size']
    with Image.open(args.image) as source:
        image=source.convert('RGB');width,height=image.size
        scale=size/max(width,height)
        resized=image.resize((max(1,round(width*scale)),max(1,round(height*scale))),Image.Resampling.BILINEAR)
        canvas=Image.new('RGB',(size,size),(124,116,104));canvas.paste(resized,(0,0))
        images=torch.from_numpy(np.array(canvas)).permute(2,0,1)[None].to(device).float()/255
    valid_size=torch.tensor([resized.width/size,resized.height/size],device=device)
    results=[]
    with torch.inference_mode(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda' and config['train']['amp']):
        features=model.vision(images)
        for start in range(0,len(args.text),8):
            texts=args.text[start:start+8]
            tokens=tokenizer(texts,padding=True,truncation=False,return_tensors='pt')
            if tokens['input_ids'].shape[1]>config['data']['text_max_length']:raise ValueError('Expression too long')
            tokens={k:v.to(device) for k,v in tokens.items()}
            shared=OrderedDict((name,value.expand(len(texts),-1,-1,-1)) for name,value in features.items())
            outputs=model(images.expand(len(texts),-1,-1,-1),tokens,features=shared,valid_sizes=valid_size[None].expand(len(texts),-1))
            for text,boxes,scores in zip(texts,outputs['boxes'],outputs['scores']):
                boxes,scores=select_predictions(boxes.float().cpu(),scores.float().cpu(),valid_size.cpu(),**config['eval'])
                boxes=boxes*boxes.new_tensor([width*size/resized.width,height*size/resized.height,width*size/resized.width,height*size/resized.height])
                results.append(dict(text=text,boxes=boxes.cpu().tolist(),scores=scores.cpu().tolist()))
    output=json.dumps(dict(image=args.image,results=results),ensure_ascii=False,indent=2)
    if args.output:
        path=Path(args.output)
        if not path.parent.is_dir():raise FileNotFoundError('Output parent must exist')
        path.write_text(output)
    print(output)


if __name__=='__main__':main()
