"""Native gRefCOCO references; one expression can have zero, one, or many targets."""
from collections import defaultdict, OrderedDict
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler, DataLoader


class GRefCOCODataset(Dataset):
    def __init__(self, dataset_dir, split, image_size=640, training=False, seed=49):
        self.root, self.split = Path(dataset_dir), split
        self._image_cache = OrderedDict()
        if image_size % 32:
            raise ValueError('image_size must be divisible by 32')
        self.image_size, self.training, self.seed, self.epoch = image_size, training, seed, 0
        records = json.loads((self.root/'labels/grefs(unc).json').read_text())
        instances = json.loads((self.root/'labels/instances.json').read_text())
        self.images = {im['id']:im for im in instances['images']}
        annotations = [{k:a[k] for k in ('id','image_id','bbox','iscrowd') if k in a} for a in instances['annotations']]
        self.annotations = {a['id']:a for a in annotations}
        self.image_annotations = defaultdict(list)
        for annotation in annotations:
            if not annotation.get('iscrowd',False):
                self.image_annotations[annotation['image_id']].append(annotation)
        self.references = [r for r in records if r['split']==split]
        if not self.references:
            raise ValueError(f'No references for official split {split}')
        self.entries = [(i,None) for i in range(len(self.references))] if training else [
            (i,j) for i,r in enumerate(self.references) for j in range(len(r['sentences']))]
        self.groups = [min(len(self.target_ids(self.references[i])),2) for i,_ in self.entries]
        self.image_ids = [self.references[i]['image_id'] for i,_ in self.entries]

    @staticmethod
    def target_ids(ref):
        ids = ref['ann_id']
        ids = ids if isinstance(ids,list) else [ids]
        if ref.get('no_target',False):
            if any(i>=0 for i in ids):
                raise ValueError('Conflicting no-target reference')
            return []
        return list(dict.fromkeys(i for i in ids if i>=0))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        ref_index, sentence_index = self.entries[index]
        ref = self.references[ref_index]
        if sentence_index is None:
            sentence_index = (random.Random(self.seed+index).randrange(len(ref['sentences']))+self.epoch)%len(ref['sentences'])
        sentence = ref['sentences'][sentence_index]
        text = sentence.get('sent',sentence.get('raw'))
        if not isinstance(text,str) or not text.strip():
            raise ValueError('Missing referring expression')
        info = self.images[ref['image_id']]
        key = info['file_name']
        if key not in self._image_cache:
            with Image.open(self.root/'images'/key) as source:
                image = source.convert('RGB')
                width,height = image.size
                scale = self.image_size/max(width,height)
                resized = image.resize((max(1,round(width*scale)),max(1,round(height*scale))),Image.Resampling.BILINEAR)
                canvas = Image.new('RGB',(self.image_size,self.image_size),(124,116,104))
                canvas.paste(resized,(0,0))
                tensor = torch.from_numpy(np.array(canvas)).permute(2,0,1)
                self._image_cache[key] = (tensor,width,height,resized.width,resized.height)
            if len(self._image_cache)>16:
                self._image_cache.popitem(last=False)
        self._image_cache.move_to_end(key)
        tensor,width,height,rw,rh = self._image_cache[key]
        sx,sy = rw/width,rh/height
        def boxes(annotations):
            result=[]
            for ann in annotations:
                x,y,w,h=ann['bbox']
                box=[max(x,0)*sx,max(y,0)*sy,min(x+w,width)*sx,min(y+h,height)*sy]
                if box[2]<=box[0] or box[3]<=box[1]:
                    raise ValueError(f'Invalid COCO BBox {ann["id"]}')
                result.append(box)
            return torch.tensor(result,dtype=torch.float32).reshape(-1,4)/self.image_size
        ids = self.target_ids(ref)
        anns = [self.annotations[i] for i in ids]
        if any(a['image_id']!=ref['image_id'] for a in anns):
            raise ValueError('Reference points to another image')
        item = dict(image=tensor,text=text,boxes=boxes(anns),
                    all_boxes=boxes(self.image_annotations[ref['image_id']]),
                    ref_id=ref['ref_id'],sent_id=sentence['sent_id'],image_id=ref['image_id'],
                    object_ids=ids,valid_size=torch.tensor([rw,rh])/self.image_size,
                    group=min(len(ids),2))
        return item


class GRefBatchSampler(Sampler):
    """Group up to four expressions per image, shuffle blocks, and visit every reference once."""
    def __init__(self,dataset,batch_size,shuffle=True):
        self.dataset,self.batch_size,self.shuffle=dataset,batch_size,shuffle

    def __len__(self):
        return (len(self.dataset)+self.batch_size-1)//self.batch_size

    def __iter__(self):
        rng=random.Random(self.dataset.seed+self.dataset.epoch)
        if self.shuffle:
            images=defaultdict(list)
            for index,image_id in enumerate(self.dataset.image_ids):images[image_id].append(index)
            blocks=[]
            for indices in images.values():
                rng.shuffle(indices)
                blocks.extend(indices[start:start+4] for start in range(0,len(indices),4))
            rng.shuffle(blocks)
            order=[index for block in blocks for index in block]
        else:
            order=list(range(len(self.dataset)))
        for start in range(0,len(order),self.batch_size):
            yield order[start:start+self.batch_size]


def grounding_collate_fn(items):
    positions={};images=[];indices=[]
    for item in items:
        image_id=item['image_id']
        if image_id not in positions:
            positions[image_id]=len(images);images.append(item['image'])
        indices.append(positions[image_id])
    targets=[{k:v for k,v in item.items() if k not in ('image','text')} for item in items]
    packed={key:torch.cat([item[key] for item in items],0) for key in ('boxes','all_boxes')}
    packed['valid_size']=torch.stack([item['valid_size'] for item in items])
    offsets={}
    for key in ('boxes','all_boxes'):
        start=0;offsets[key]=[]
        for target in targets:
            end=start+len(target[key]);offsets[key].append((start,end))
            target[key]=packed[key][start:end];start=end
    for i,target in enumerate(targets):
        for key in packed:
            if key not in offsets:target[key]=packed[key][i]
    return dict(images=torch.stack(images),image_indices=torch.tensor(indices),
                texts=[item['text'] for item in items],targets=targets,
                packed_targets=packed,target_offsets=offsets)


def build_dataloader(dataset, batch_size, workers):
    # Workers are restarted each epoch so sentence cycling receives the new epoch.
    return DataLoader(dataset,batch_sampler=GRefBatchSampler(dataset,batch_size,dataset.training),
                      num_workers=workers,pin_memory=True,collate_fn=grounding_collate_fn,
                      persistent_workers=False)
