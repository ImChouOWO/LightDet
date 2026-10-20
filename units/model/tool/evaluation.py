"""Use the same gRefCOCO protocol for training validation and standalone validation."""
import torch
from tqdm import tqdm

from units.model.tool.grounding_metrics import GroundingMetrics
from units.model.tool.runtime import prepare_model_batch


@torch.inference_mode()
def evaluate(model,loader,tokenizer,device,config):
    model.eval()
    metric=GroundingMetrics(**config['eval'])
    for batch in tqdm(loader,desc='Validation'):
        images,tokens,targets=prepare_model_batch(batch,tokenizer,device,config['data']['text_max_length'],move_targets=False)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda' and config['train']['amp']):
            outputs=model(images,tokens,image_indices=batch['image_indices'].to(device),valid_sizes=torch.stack([target['valid_size'] for target in targets]).to(device))
        boxes_cpu=outputs['boxes'].float().cpu()
        scores_cpu=outputs['scores'].float().cpu()
        for boxes,scores,target in zip(boxes_cpu,scores_cpu,targets):
            metric.update(boxes,scores,target)
    return metric.compute()
