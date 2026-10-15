"""Evaluate the validation split with the training grid and global mIoU."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from models import build_model
from utils.common import inspect_data, load_checkpoint, write_json, device_from_name, worker_seed, file_hash
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.global_metrics import evaluate_global


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--num-classes', type=int, required=True, help='Legacy class argument: adds one output channel')
    p.add_argument('--backbone', choices=['dinov3_b', 'dinov3_l'], default='dinov3_b')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--output', required=True, help='New JSON file')
    args = p.parse_args()
    if not 1 <= args.num_classes <= 254 or args.workers < 0:
        p.error('Invalid classes/workers')
    args.batch_size = args.batch_size or (24 if args.backbone == 'dinov3_b' else 12)
    if args.batch_size < 1:
        p.error('batch-size must be positive')
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    device = device_from_name(args.device)
    torch.set_num_threads(args.threads)
    splits = inspect_data(args.data_path, ['val.txt'])
    classes = args.num_classes + 1
    model = build_model(classes, seed=args.seed, backbone=args.backbone)
    metadata = load_checkpoint(model, args.weights, 'G3', classes)
    shape = metadata.get('config', {}).get('train_size', [512, 512])
    dataset = UnetDataset(args.data_path, shape, classes, False, 'val.txt')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, collate_fn=unet_dataset_collate,
                        worker_init_fn=worker_seed,
                        generator=torch.Generator().manual_seed(args.seed + 1))
    result = evaluate_global(model.to(device), loader, device, classes)
    result.update(split='val.txt', input_shape=shape, batch_size=args.batch_size,
                  dataset=splits, checkpoint_sha256=file_hash(args.weights),
                  protocol='dataset accumulated confusion matrix; includes background; union > 0 classes')
    write_json(args.output, result)
    print(result)


if __name__ == '__main__':
    main()
