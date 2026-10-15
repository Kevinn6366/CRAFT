"""Formal test: global mIoU and per-image boundary diagnostics at 480 x 512."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm

from utils.boundary_metrics import BoundaryMetrics
from utils.common import (inspect_data, split_ids, write_json, provenance,
                                load_checkpoint, device_from_name, file_hash)
from models.factory import build_model
from models.config import VARIANTS
from utils.dataloader import UnetDataset
from utils.metrics import mean_iou, pixel_accuracy, mean_accuracy, frequency_weighted_iou
from utils.region_metrics import per_class_regions, region_summary

TEST_SHAPE = (480, 512)  # Fixed to val.py, not the training input size.


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.set_defaults(global_miou=True, variant='G3', predictions=None, allow_unused_craft=False, method_name='CRAFT')
    p.add_argument('--data-path', required=True)
    p.add_argument('--num-classes', type=int, required=True, help='Same legacy argument as val.py: adds 1')
    p.add_argument('--output', required=True, help='New directory for evaluation metrics and prediction masks')
    p.add_argument('--weights', required=True, help='Trained model checkpoint')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=11, help='Only initialization/test order; checkpoint seed is recorded separately')
    p.add_argument('--boundary-ratio', type=float, default=0.02)
    p.add_argument('--tolerance-ratio', type=float, default=0.005)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--backbone', choices=['dinov3_b', 'dinov3_l'], default='dinov3_b')
    return p


def read_mask(path, shape, classes):
    with Image.open(path) as image:
        arr = np.array(image)
    if arr.shape != tuple(shape) or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f'{path}: expected integer class-ID mask with shape {shape}, got {arr.shape}')
    if np.any((arr < 0) | (arr >= classes)):
        raise ValueError(f'{path}: out-of-range predicted class ID; RGB color masks are unsupported')
    return arr.astype(np.int64)


def legacy_mask_miou(mask, target, classes):
    # Reuse the existing implementation, including absent classes / ignore behavior.
    # One-hot scores produce exactly the supplied argmax without reimplementing IoU.
    labels = torch.from_numpy(np.asarray(mask, dtype=np.int64))
    scores = F.one_hot(labels, classes).permute(2, 0, 1).unsqueeze(0).float()
    return mean_iou(scores, target, classes)



def run(args):
    if not 1 <= args.num_classes <= 254:
        raise ValueError('This VOC PNG pipeline supports legacy --num-classes in 1..254')
    if args.weights and not args.variant:
        raise ValueError('--weights requires --variant; old weights cannot identify the ablation by themselves')
    if args.predictions and (args.variant or not args.method_name):
        raise ValueError('External masks require --method-name, without --variant')
    if args.predictions and args.allow_unused_craft:
        raise ValueError('--allow-unused-craft requires model weights')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Refusing to overwrite nonempty output: {out}')
    data = inspect_data(args.data_path, ['test.txt'])
    ids = split_ids(args.data_path, 'test.txt')
    classes = args.num_classes + 1
    torch.set_num_threads(args.threads)
    dataset = UnetDataset(args.data_path, TEST_SHAPE, classes, augmentation=False, txt_name='test.txt')
    from utils.global_metrics import GlobalIoU
    global_metric = GlobalIoU(classes)
    metrics = BoundaryMetrics(classes, args.boundary_ratio, args.tolerance_ratio)
    meta = {'args': vars(args), 'code': provenance(), 'dataset': data, 'num_classes': classes,
            'variant': args.variant, 'method_name': args.method_name or VARIANTS[args.variant].title,
            'miou_protocol': 'legacy mean_iou, batch=1, equal mean over test images; no metric changes',
            'miou_grid': list(TEST_SHAPE), 'boundary_grid': list(TEST_SHAPE),
            'boundary_restoration': 'none: same argmax and GT tensors as legacy mIoU'}
    model, checkpoint_meta = None, {}
    if args.weights:
        device = device_from_name(args.device)
        model = build_model(classes, args.variant, args.seed, getattr(args, 'backbone', 'resnet50'))
        checkpoint_meta = load_checkpoint(model, args.weights, args.variant, classes,
                                           allow_unused_craft=args.allow_unused_craft)
        model.to(device).eval()
        meta['checkpoint'] = {'path': str(Path(args.weights).resolve()),
                              'sha256': file_hash(args.weights), 'metadata': checkpoint_meta}
    out.mkdir(parents=True, exist_ok=True)
    if args.weights:
        (out / 'eval_masks').mkdir()
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(args.seed)).tolist()
    records, prediction_hashes = [], {}
    with torch.no_grad():
        for index in tqdm(order, desc=meta['method_name']):
            name = ids[index]
            image, target, _ = dataset[index]
            target_tensor = torch.from_numpy(target.astype(np.int64)).unsqueeze(0)
            if model is not None:
                logits = model(torch.from_numpy(image).float().unsqueeze(0).to(device))[0].cpu()
                miou = mean_iou(logits, target_tensor, classes)
                prediction = logits.argmax(1)[0].numpy().astype(np.uint8)
                Image.fromarray(prediction).save(out / 'eval_masks' / (name + '.png'))
            else:
                path = Path(args.predictions) / (name + '.png')
                prediction = read_mask(path, TEST_SHAPE, classes)
                prediction_hashes[name] = file_hash(path)
                miou = legacy_mask_miou(prediction, target_tensor, classes)
            if model is None:
                logits = F.one_hot(torch.from_numpy(prediction.astype(np.int64)), classes).permute(2, 0, 1).unsqueeze(0).float()
            other = {'pixel_accuracy': pixel_accuracy(logits, target_tensor, classes),
                     'mean_accuracy': mean_accuracy(logits, target_tensor, classes),
                     'frequency_weighted_iou': frequency_weighted_iou(logits, target_tensor, classes)}
            other['per_class_regions'] = per_class_regions(prediction, target, classes)
            global_metric.update(prediction, target)
            boundary_row = metrics.update(prediction, target)
            records.append({'id': name, 'miou': miou, **other, **boundary_row})
    measured = [r['miou'] for r in records if r['miou'] is not None]
    result = {'metadata': meta, 'miou': sum(measured) / len(measured) if measured else None,
              'miou_images': len(measured), 'boundary': metrics.compute(),
              'per_image': sorted(records, key=lambda r: r['id']), 'units': 'fractions in [0,1]',
              'external_prediction_sha256': prediction_hashes}
    result['global'] = global_metric.compute()
    if args.global_miou:
        result['legacy_miou'] = result['miou']
        result['miou'] = result['global']['miou']
        result['metadata']['miou_protocol'] = result['global']['definition']
    result['region'] = region_summary(records, classes)
    result['additional_metrics'] = {k: float(np.mean([r[k] for r in records]))
        for k in ('pixel_accuracy', 'mean_accuracy', 'frequency_weighted_iou')}
    write_json(out / 'metrics.json', result)
    print({k: result[k] for k in ('miou', 'miou_images')})
    print({k: result['boundary'][k] for k in ('boundary_iou', 'boundary_f1', 'contact_iou')})
    return result


if __name__ == '__main__':
    run(parser().parse_args())
