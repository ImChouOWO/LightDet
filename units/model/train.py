"""Train the gRefCOCO Transformer Main/Aux model; old LightDet checkpoints are not accepted."""
import argparse
import json
import math
from pathlib import Path
import random
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from units.tool.card import VisionTextModel
from units.model.pipeline.data import GRefCOCODataset,build_dataloader
from units.model.cards.loss import GroundingLoss
from units.model.tool.config import load_config,resolve
from units.model.tool.runtime import prepare_model_batch
from units.model.tool.evaluation import evaluate


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def make_optimizer(model,config):
    groups=[]
    for module,lr in [(model.vision.backbone,config['vision_lr']),
                      (model.vision.fpn,config['fpn_lr']),(model.text,config['text_lr'])]:
        parameters=[p for p in module.parameters() if p.requires_grad]
        groups.append(dict(params=parameters,lr=lr))
    heads=[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith(('vision.','text.'))]
    groups.append(dict(params=heads,lr=config['head_lr']))
    return torch.optim.AdamW(groups,weight_decay=config['weight_decay'],fused=next(model.parameters()).is_cuda)


def build_components(model_path,train_path,device=None,aux_enabled=None):
    model_config=load_config(model_path)['model'];config=load_config(train_path)
    if aux_enabled is not None:config['train']['aux_enabled']=aux_enabled
    selected_device=torch.device(device or config['train']['device'])
    if selected_device.type=='cuda':torch.cuda.set_device(selected_device)
    tokenizer=AutoTokenizer.from_pretrained(resolve(model_config['encoder_path']),local_files_only=True)
    model=VisionTextModel(**model_config,aux_enabled=config['train']['aux_enabled']).to(selected_device)
    return model,tokenizer,config,model_config,selected_device


def gt_roi_factor(epoch,train_config):
    hold,end=train_config['gt_roi_hold_epochs'],train_config['gt_roi_end_epoch']
    if epoch<=hold:return train_config['gt_roi_weight']
    if epoch>=end:return 0.
    return train_config['gt_roi_weight']*.5*(1+math.cos(math.pi*(epoch-hold)/(end-hold)))


def save_checkpoint(path,model,optimizer,scheduler,scaler,epoch,best,config,model_config):
    temporary=path.with_suffix('.tmp')
    torch.save(dict(schema='lightdet.grefcoco.queries.v3',model=model.state_dict(),optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),epoch=epoch,
                    best_metric=best,config=config,model_config=model_config,
                    rng_torch=torch.get_rng_state(),rng_numpy=np.random.get_state(),rng_python=random.getstate(),
                    rng_cuda=torch.cuda.get_rng_state(next(model.parameters()).device) if next(model.parameters()).is_cuda else None),temporary)
    temporary.replace(path)


def format_metric(value):
    return 'N/A' if value is None else f'{value:.4f}'


def print_training_details(model,config,model_config,device,train_data,val_data,warm_steps,start,resume):
    total=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    settings,data,optim=config['train'],config['data'],config['optim']
    hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'
    lines=[
        '[Model] LightDet-gRef',
        f"  Backbone : ResNet-50 -> FPN ({model_config['hidden_dim']} channels)",
        f"  Text     : Multilingual MiniLM, frozen={model_config['freeze_text']}",
        f"  Location : Transformer, layers={model_config['query_layers']}, heads={model_config['query_heads']}, Main queries={model_config['num_object_queries']}",
        f"  Query init: zero + learnable query -> text cross-attention ({model_config['query_heads']} heads) + query + pooled projection",
        f"  Aux      : enabled={settings['aux_enabled']}, queries={model_config['num_aux_queries']}, K={config['loss']['aux_top_k']}, Main weight={config['loss']['main_weight']}, Aux weight={config['loss']['aux_weight']}",
        f"  ROI/Text : ROI {model_config['roi_size']}x{model_config['roi_size']}, refinement layers={model_config['score_layers']}, heads={model_config['score_heads']}",
        f"  Parameters: total={total/1e6:.2f}M, trainable={trainable/1e6:.2f}M",
        '[Training]',
        f"  Device   : {device} ({hardware}), precision={'BF16' if device.type=='cuda' and settings['amp'] else 'FP32'}",
        f"  Data     : train={len(train_data)}, val={len(val_data)}, image={data['image_size']}, batch={settings['batch_size']}, workers={settings['workers']}",
        f"  Capacity : train references above Main capacity={sum(len(train_data.target_ids(ref))>model_config['num_object_queries'] for ref in train_data.references)}; all GT retained",
        f"  Schedule : epochs={settings['epochs']}, start={start}, warmup={warm_steps} steps, mode={'resume' if resume else 'from scratch'}",
        f"  LR       : backbone={optim['vision_lr']:.2g}, FPN={optim['fpn_lr']:.2g}, text projection={optim['text_lr']:.2g}, heads={optim['head_lr']:.2g}",
        f"  GT ROI   : weight={settings['gt_roi_weight']}, hold through epoch {settings['gt_roi_hold_epochs']}, off at epoch {settings['gt_roi_end_epoch']}",
        '  Loss     : '+', '.join(f"{key.removesuffix('_weight')}={value:g}" for key,value in config['loss'].items()),
        f"  Margin   : off through epoch {config['loss']['threshold_margin_start_epoch']-1}, cosine to {config['loss']['threshold_margin_target']} at epoch {config['loss']['threshold_margin_end_epoch']}, weight={config['loss']['threshold_margin_weight']}",
        f"  Eval     : best={config['log']['best_metric']}, threshold={config['eval']['score_threshold']}, Main NMS={config['eval']['use_nms']}, Aux NMS=True",
        f"  Output   : {resolve(config['log']['save_dir'])}/{config['log']['prefix']}"
    ]
    print('\n'.join(lines),flush=True)


def print_epoch_summary(record,epochs,elapsed,is_best,best_path):
    metrics=record['validation']
    line=(f"Epoch [{record['epoch']}/{epochs}] train={record['train']['total']:.4f} "
          f"GREC Pr@(F1=1)={format_metric(metrics['grec_precision_f1'])} "
          f"Acc@0.5={format_metric(metrics['rec_acc50'])} "
          f"N-Acc={format_metric(metrics['n_acc'])} T-Acc={format_metric(metrics['t_acc'])} "
          f"Positive-Pr={format_metric(metrics['positive_precision_f1'])} time={elapsed:.1f}s")
    schedule=record.get('selection_margin')
    if schedule:
        line+=f" margin={schedule['value']:.3f} margin_weight={schedule['weight']:.3f}"
    tqdm.write(line)
    if is_best:
        message=f"Saved best checkpoint: epoch={record['epoch']}, GREC Pr@(F1=1)={format_metric(metrics['grec_precision_f1'])} -> {best_path.name}"
        if sys.stdout.isatty():message=f'\033[1;32m{message}\033[0m'
        tqdm.write(message)
    sys.stdout.flush()


def train(args):
    model,tokenizer,config,model_config,device=build_components(args.model_config,args.train_config,args.device,args.aux)
    settings=config['train'];seed_everything(settings['seed'])
    torch.set_num_threads(4)
    if device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=True
    data=config['data']
    train_data=GRefCOCODataset(resolve(data['dataset_dir']),data['train_split'],data['image_size'],True,settings['seed'])
    val_data=GRefCOCODataset(resolve(data['dataset_dir']),data['val_split'],data['image_size'],False,settings['seed'])
    loader=build_dataloader(train_data,settings['batch_size'],settings['workers'])
    val_loader=build_dataloader(val_data,settings['batch_size'],settings['workers'])
    criterion=GroundingLoss(**config['loss'],selection_threshold=config['eval']['score_threshold']).to(device)
    optimizer=make_optimizer(model,config['optim'])
    total_steps=settings['epochs']*len(loader);warm_steps=min(settings['warmup_epochs']*len(loader),settings['max_warmup_steps'])
    def factor(step):
        if step<warm_steps:return (step+1)/max(warm_steps,1)
        progress=min(1.,(step-warm_steps)/max(total_steps-warm_steps,1))
        minimum=config['optim']['min_lr_ratio']
        return minimum+(1-minimum)*.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    # BF16 does not need a GradScaler, but its empty state is saved consistently.
    scaler=torch.amp.GradScaler(device.type,enabled=False)
    start,best=1,-1.
    if args.resume:
        checkpoint=torch.load(resolve(args.resume),map_location=device,weights_only=False)
        if checkpoint['schema']!='lightdet.grefcoco.queries.v3':raise ValueError('Expected text-guided query checkpoint (v3)')
        if checkpoint['model_config']!=model_config or checkpoint['config']!=config:
            raise ValueError('Resume requires the same architecture and training configuration')
        model.load_state_dict(checkpoint['model'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler']);start=checkpoint['epoch']+1;best=checkpoint['best_metric']
        torch.set_rng_state(checkpoint['rng_torch'].cpu());np.random.set_state(checkpoint['rng_numpy']);random.setstate(checkpoint['rng_python'])
        if checkpoint['rng_cuda'] is not None and device.type=='cuda':torch.cuda.set_rng_state(checkpoint['rng_cuda'].cpu(),device=device)
    save_dir=resolve(config['log']['save_dir'])
    if not save_dir.is_dir():raise FileNotFoundError('save_dir must be an existing project folder')
    prefix=config['log']['prefix']
    metrics_path=save_dir/(prefix+'_metrics.jsonl')
    if not args.resume and metrics_path.exists():raise FileExistsError('Existing run: choose another prefix or resume')
    print_training_details(model,config,model_config,device,train_data,val_data,warm_steps,start,args.resume)
    for epoch in range(start,settings['epochs']+1):
        epoch_started=perf_counter()
        train_data.set_epoch(epoch);model.train()
        totals={};steps=0;teacher_weight=gt_roi_factor(epoch,settings)
        progress=tqdm(loader,desc=f'Epoch {epoch}/{settings["epochs"]} [Train]')
        for batch in progress:
            images,tokens,targets=prepare_model_batch(batch,tokenizer,device,data['text_max_length'])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda' and settings['amp']):
                outputs=model(images,tokens,targets,gt_roi=teacher_weight>0,
                    image_indices=batch['image_indices'].to(device))
                loss,parts=criterion(outputs,targets,epoch=epoch)
                teacher=outputs.get('gt_alignment_logits',[])
                teacher_loss=torch.cat(teacher).float() if teacher else None
                if teacher_loss is not None:
                    teacher_loss=F.binary_cross_entropy_with_logits(teacher_loss,torch.ones_like(teacher_loss))
                    loss=loss+criterion.main_weight*teacher_weight*teacher_loss
                    parts['gt_roi']=teacher_loss
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['grad_clip'],error_if_nonfinite=True)
            optimizer.step();scheduler.step()
            steps+=1
            values=dict(total=loss,**parts)
            measured=torch.stack([value.detach() for value in values.values()]).float().cpu().tolist()
            for name,value in zip(values,measured):totals[name]=totals.get(name,0.)+value
            if steps%20==0 or steps==len(loader):
                current=dict(zip(values,measured))
                progress.set_postfix(lr=f'{optimizer.param_groups[-1]["lr"]:.2e}',
                    loss=f'{current["total"]:.4f}',avg=f'{totals["total"]/steps:.4f}',
                    align=f'{current["main_alignment"]:.4f}',rank=f'{current["main_ranking"]:.4f}')
            if args.smoke_test:
                print(json.dumps(dict(loss=float(loss.detach()),gradient_norm=float(norm),predictions=list(outputs['boxes'].shape),
                    teacher_weight=teacher_weight,losses={k:float(v.detach()) for k,v in parts.items()})))
                return
        metrics=evaluate(model,val_loader,tokenizer,device,config)
        margin,margin_weight=criterion.margin_schedule(epoch)
        record=dict(epoch=epoch,train={name:value/steps for name,value in totals.items()},validation=metrics,
                    selection_margin=dict(value=margin,weight=margin_weight,threshold=criterion.selection_threshold),
                    aux_factor=criterion.aux_factor(epoch) if settings['aux_enabled'] else 0.)
        with metrics_path.open('a') as stream:stream.write(json.dumps(record)+'\n')
        metric=metrics[config['log']['best_metric']]
        is_best=metric is not None and metric>best
        if is_best:best=metric
        save_checkpoint(save_dir/(prefix+'_latest.pt'),model,optimizer,scheduler,scaler,epoch,best,config,model_config)
        if is_best:save_checkpoint(save_dir/(prefix+'_best.pt'),model,optimizer,scheduler,scaler,epoch,best,config,model_config)
        if epoch%10==0:save_checkpoint(save_dir/(prefix+f'_epoch_{epoch:03d}.pt'),model,optimizer,scheduler,scaler,epoch,best,config,model_config)
        print_epoch_summary(record,settings['epochs'],perf_counter()-epoch_started,is_best,save_dir/(prefix+'_best.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-config',default='units/model/cards/config/model.yaml')
    parser.add_argument('--train-config',default='units/model/cards/config/train.yaml')
    parser.add_argument('--device')
    parser.add_argument('--resume')
    parser.add_argument('--aux',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    # Seed before model construction for reproducible initialization.
    seed_everything(load_config(args.train_config)['train']['seed'])
    train(args)


if __name__=='__main__':main()
