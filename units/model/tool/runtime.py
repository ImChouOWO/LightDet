"""Batch transfer and shared inference postprocessing."""
import torch
from torchvision.ops import nms


def prepare_model_batch(batch,tokenizer,device,max_length,move_targets=True):
    tokens=tokenizer(batch['texts'],padding=True,truncation=False,return_tensors='pt')
    if tokens['input_ids'].shape[1]>max_length:
        raise ValueError('Expression exceeds text_max_length')
    tokens={k:v.to(device,non_blocking=True) for k,v in tokens.items() if k in ('input_ids','attention_mask','token_type_ids')}
    images=batch['images'].to(device,non_blocking=True).float().div_(255)
    targets=batch['targets']
    if move_targets:
        packed={key:value.to(device,non_blocking=True) for key,value in batch['packed_targets'].items()}
        targets=[dict(item) for item in targets]
        for i,target in enumerate(targets):
            for key,value in packed.items():
                if key in batch['target_offsets']:
                    start,end=batch['target_offsets'][key][i]
                    target[key]=value[start:end]
                else:target[key]=value[i]
    return images,tokens,targets


def select_predictions(boxes,scores,valid_size,score_threshold=.7,use_nms=False,nms_iou=.5,max_detections=100):
    boxes=boxes.float().clamp_min(0).clone()
    boxes[:,[0,2]]=boxes[:,[0,2]].clamp_max(float(valid_size[0]))
    boxes[:,[1,3]]=boxes[:,[1,3]].clamp_max(float(valid_size[1]))
    keep=(scores>=score_threshold)&(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
    boxes,scores=boxes[keep],scores[keep].float()
    ids=(nms(boxes,scores,nms_iou) if use_nms else scores.argsort(descending=True))[:max_detections]
    return boxes[ids],scores[ids]


def select_branch_predictions(outputs,valid_size,branch='main',score_threshold=.7,nms_iou=.5,max_detections=100):
    if branch not in ('main','aux'):raise ValueError('Unknown branch')
    prediction=outputs if branch=='main' else outputs['aux']
    return select_predictions(prediction['boxes'],prediction['scores'],valid_size,
        score_threshold=score_threshold,use_nms=branch=='aux',nms_iou=nms_iou,max_detections=max_detections)
