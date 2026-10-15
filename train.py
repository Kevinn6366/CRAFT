"""Train full CRAFT with a DINOv3-B or DINOv3-L encoder."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.common import (NoSingletonBatchSampler, inspect_data, provenance,
                                worker_seed, write_json, load_state, device_from_name, file_hash)
from models.factory import build_model, loss_terms
from models.config import VARIANTS
from utils.losses import get_lr_scheduler, set_optimizer_lr
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.set_defaults(global_miou=True, select_miou=True, variant='G3')
    p.add_argument('--data-path', required=True)
    p.add_argument('--num-classes', type=int, required=True,
                   help='LEGACY argument: model has this value + 1 channels; match prior runs')
    p.add_argument('--output', required=True, help='New empty run directory (or existing with --resume)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--epochs', type=int, default=90)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-2)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--init-weights', help='Optional shared initialization; matching tensors only, logged')
    p.add_argument('--height-weight', type=float, default=10.0)
    p.add_argument('--backbone-weights', help='Pinned DINOv3 model.safetensors; strict hash and encoder loading')
    p.add_argument('--resume', action='store_true', help='Resume output/last.pt with identical settings')
    p.add_argument('--train-size', type=int, nargs=2, default=[512, 512], metavar=('H', 'W'),
                   help='Only reduce for synthetic smoke tests; keep identical across real runs')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--backbone', choices=['dinov3_b', 'dinov3_l'], default='dinov3_b')
    return p


def save_checkpoint(path, data):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(data, tmp)
    tmp.replace(path)


def run(args):
    if args.batch_size is None:
        args.batch_size = 24 if args.backbone == 'dinov3_b' else 12
    if args.epochs < 1 or not 1 <= args.num_classes <= 254 or args.batch_size < 2 or args.workers < 0:
        raise ValueError('Invalid epochs/classes/batch-size/workers')
    if min(args.train_size) < 32 or any(n % 32 for n in args.train_size):
        raise ValueError('Training H/W must be positive multiples of 32 (at least 32)')
    if not np.isfinite(args.height_weight) or args.height_weight < 0:
        raise ValueError('Invalid height loss coefficient')
    device = device_from_name(args.device)
    if args.amp and device.type != 'cuda':
        raise ValueError('--amp requires CUDA')
    out = Path(args.output)
    if args.resume:
        if not (out / 'last.pt').is_file():
            raise FileNotFoundError(out / 'last.pt')
    elif out.exists() and any(out.iterdir()):
        raise ValueError(f'Refusing to overwrite nonempty run directory: {out}')
    else:
        out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    splits = ['train.txt', 'val.txt']
    if (Path(args.data_path) / 'VOC2012/ImageSets/Segmentation/test.txt').exists():
        splits.append('test.txt')
    data = inspect_data(args.data_path, splits)
    classes = args.num_classes + 1
    meta = {'variant': args.variant, 'num_classes': classes, 'legacy_num_classes_arg': args.num_classes,
            'seed': args.seed, 'config': vars(args), 'dataset': data, 'code': provenance(),
            'height_target': 'foreground component EDT, exponent 0.2',
            'loss_weights': {'main': 1.0, 'aux': 0.4,
                             'height': args.height_weight if VARIANTS[args.variant].height_supervision else 0.0},
            'validation': 'legacy evaluate: equal mean of batch metrics; shuffled; training size',
            'selection': 'Mean Accuracy (legacy)',
            'last_singleton_policy': 'merge into preceding training batch; keep all images'}
    model = build_model(classes, args.variant, args.seed, getattr(args, 'backbone', 'resnet50'))
    if args.init_weights and args.backbone_weights:
        raise ValueError('Choose one initialization source')
    if args.backbone_weights and not args.resume:
        from models.dinov3 import load as load_backbone
        meta['initialization'] = load_backbone(model, args.backbone_weights)
    if args.init_weights and not args.resume:
        state, _ = load_state(args.init_weights)
        matching = {k: v for k, v in state.items() if k in model.state_dict()
                    and v.shape == model.state_dict()[k].shape}
        if not matching:
            raise ValueError('No compatible initialization tensors')
        model.load_state_dict(matching, strict=False)
        meta['initialization'] = {'path': args.init_weights, 'sha256': file_hash(args.init_weights),
                                  'loaded_keys': sorted(matching),
                                  'skipped_keys': sorted(set(state) - set(matching))}
    model.to(device)
    meta['parameters'] = sum(p.numel() for p in model.parameters())
    params = model.parameters()
    if args.backbone.startswith('dinov3_'):
        encoder = list(model.resnet.encoder.parameters())
        encoder_ids = {id(p) for p in encoder}
        params = [{'params':encoder,'lr_scale':0.1},
                  {'params':[p for p in model.parameters() if id(p) not in encoder_ids],'lr_scale':1.0}]
    if args.global_miou:
        meta['validation'] = 'dataset accumulated confusion matrix, union>0 classes, includes background'
        meta['selection'] = 'global validation mIoU; primary report fixed epoch90'
    if args.select_miou:
        meta['selection'] = 'maximum validation Mean IoU under the recorded evaluation protocol; earliest tie'
    optimizer = torch.optim.AdamW(params, lr=args.lr,
                                 betas=(args.momentum, 0.999), weight_decay=args.weight_decay)
    scheduler = get_lr_scheduler('cos', args.lr, args.lr * 0.01, args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    gen_train = torch.Generator().manual_seed(args.seed)
    gen_val = torch.Generator().manual_seed(args.seed + 1)
    train = UnetDataset(args.data_path, args.train_size, classes, True, 'train.txt')
    val = UnetDataset(args.data_path, args.train_size, classes, False, 'val.txt')
    loader = DataLoader(train, batch_sampler=NoSingletonBatchSampler(len(train), args.batch_size, gen_train),
                        num_workers=args.workers, collate_fn=unet_dataset_collate,
                        worker_init_fn=worker_seed, generator=gen_train, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=True, drop_last=False,
                            num_workers=args.workers, collate_fn=unet_dataset_collate,
                            worker_init_fn=worker_seed, generator=gen_val, pin_memory=device.type == 'cuda')
    start, best, history = 0, -1.0, []
    if args.resume:
        last = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
        old_config = last['metadata']['config']
        for key, value in vars(args).items():
            if key not in ('resume', 'threads', 'device', 'output') and old_config.get(key) != value:
                raise ValueError(f'Resume configuration mismatch: {key}')
        if last['metadata']['dataset'] != data:
            raise ValueError('Dataset splits changed since checkpoint')
        if last['metadata']['code']['source_sha256'] != meta['code']['source_sha256']:
            raise ValueError('Source code changed since checkpoint; start a new run instead')
        model.load_state_dict(last['model'], strict=True)
        optimizer.load_state_dict(last['optimizer'])
        scaler.load_state_dict(last['scaler'])
        start, best, history = last['epoch'] + 1, last['best_miou'], last['history']
        random.setstate(last['rng_python'])
        nr = last['rng_numpy']
        np.random.set_state((nr[0], np.array(nr[1], dtype=np.uint32), nr[2], nr[3], nr[4]))
        torch.set_rng_state(last['rng_torch'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(last['rng_cuda'])
        gen_train.set_state(last['rng_train_loader'])
        gen_val.set_state(last['rng_val_loader'])
        meta = last['metadata']
    write_json(out / 'run.json', meta)
    for epoch in range(start, args.epochs):
        model.train()
        set_optimizer_lr(optimizer, scheduler, epoch)
        for group in optimizer.param_groups:
            group['lr'] *= group.get('lr_scale', 1.0)
        sums = {k: 0.0 for k in ('total', 'main', 'aux', 'height')}
        epoch_start = time.time()
        for step, (images, target, heights) in enumerate(tqdm(loader, desc=f'{args.variant} epoch {epoch+1}/{args.epochs}', mininterval=30), 1):
            images, target, heights = images.to(device), target.to(device), heights.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp):
                terms = loss_terms(model, model(images), target, heights, height_weight=args.height_weight)
            if not torch.isfinite(terms['total']):
                raise FloatingPointError('Non-finite training loss; run stopped')
            scaler.scale(terms['total']).backward()
            scaler.step(optimizer)
            scaler.update()
            for key, value in terms.items():
                sums[key] += float(value.detach())
            if step == 1 or step % 25 == 0:
                write_json(out / 'progress.json', {'variant': args.variant, 'epoch': epoch + 1,
                    'step': step, 'steps': len(loader), 'loss': float(terms['total'].detach()),
                    'elapsed_epoch_seconds': time.time() - epoch_start, 'updated_unix': time.time()})
        if args.global_miou:
            from utils.global_metrics import evaluate_global
            metrics = evaluate_global(model, val_loader, device, classes)
        row = {'epoch': epoch + 1, 'train': {k: v / len(loader) for k, v in sums.items()},
               'epoch_seconds': time.time() - epoch_start, 'validation': metrics, 'lr': optimizer.param_groups[0]['lr']}
        history.append(row)
        print(json.dumps(row), flush=True)
        if epoch + 1 in (50, 90):
            save_checkpoint(out / f'epoch_{epoch + 1:03d}.pt', {
                'model': model.state_dict(), 'metadata': meta,
                'epoch': epoch, 'validation': metrics})
        selection_value = metrics['Mean IoU'] if args.select_miou else metrics['Mean Accuracy']
        if selection_value > best:
            best = selection_value
            save_checkpoint(out / 'best.pt', {'model': model.state_dict(), 'metadata': meta,
                                              'epoch': epoch, 'validation': metrics})
            write_json(out / 'best_validation.json', {
                'epoch': epoch + 1, 'miou': metrics['Mean IoU'],
                'selection_value': best, 'selection': meta['selection'],
                'validation': metrics, 'checkpoint': str(out / 'best.pt'),
                'sha256': file_hash(out / 'best.pt'), 'epoch_indexing': 'one-based'})
        nr = np.random.get_state()
        save_checkpoint(out / 'last.pt', {'model': model.state_dict(), 'metadata': meta,
                         'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                         'epoch': epoch, 'best_miou': best, 'history': history,
                         'rng_python': random.getstate(),
                         'rng_numpy': (nr[0], nr[1].tolist(), nr[2], nr[3], nr[4]),
                         'rng_torch': torch.get_rng_state(),
                         'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                         'rng_train_loader': gen_train.get_state(), 'rng_val_loader': gen_val.get_state()})
        write_json(out / 'history.json', history)
    return out / 'best.pt'


if __name__ == '__main__':
    run(parser().parse_args())
