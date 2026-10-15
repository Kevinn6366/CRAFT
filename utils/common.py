"""Run provenance and data checks; no changes to legacy metric definitions."""
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def provenance():
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[1]), *args], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return 'unavailable'
    base = Path(__file__).resolve().parents[1]
    paths = list(base.glob('*.py')) + [p for folder in ('models', 'utils', 'tools') for p in (base / folder).glob('*.py')]
    return {'commit': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
            'git_status': git('status', '--short'), 'torch_version': str(torch.__version__),
            'source_sha256': {str(p.relative_to(base)): file_hash(p) for p in sorted(paths)}}


def split_ids(root, split):
    path = Path(root) / 'VOC2012/ImageSets/Segmentation' / split
    ids = [line.split()[0] for line in path.read_text().splitlines() if line.strip()]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f'{path}: empty or duplicate IDs')
    for name in ids:
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError(f'Expected basename IDs without directories: {name}')
    return ids


def inspect_data(root, splits):
    result, seen = {}, set()
    for split in splits:
        ids = split_ids(root, split)
        overlap = seen.intersection(ids)
        if overlap:
            raise ValueError(f'Split overlap in {split}: {sorted(overlap)[:5]}')
        seen.update(ids)
        for name in ids:
            for folder, suffix in [('JPEGImages', '.jpg'), ('SegmentationClass', '.png')]:
                path = Path(root) / 'VOC2012' / folder / (name + suffix)
                if not path.is_file():
                    raise FileNotFoundError(path)
        result[split] = {'count': len(ids), 'sha256': file_hash(
            Path(root) / 'VOC2012/ImageSets/Segmentation' / split)}
    return result


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


class NoSingletonBatchSampler(Sampler):
    """Keep every image, merging a final singleton to avoid PPM BatchNorm failure."""
    def __init__(self, size, batch_size, generator):
        if size < 2 or batch_size < 2:
            raise ValueError('Training requires at least 2 images and batch-size >= 2')
        self.size, self.batch_size, self.generator = size, batch_size, generator

    def __iter__(self):
        indices = torch.randperm(self.size, generator=self.generator).tolist()
        batches = [indices[i:i+self.batch_size] for i in range(0, self.size, self.batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        yield from batches

    def __len__(self):
        count = (self.size + self.batch_size - 1) // self.batch_size
        return count - int(self.size % self.batch_size == 1 and count > 1)


def load_state(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(data, dict) and 'model' in data:
        return data['model'], data.get('metadata', {})
    if isinstance(data, dict) and 'state_dict' in data:
        return data['state_dict'], data.get('metadata', {})
    return data, {}


def load_checkpoint(model, path, variant, num_classes, allow_unused_craft=False):
    state, meta = load_state(path)
    if meta and (meta.get('variant') != variant or meta.get('num_classes') != num_classes):
        raise ValueError('Checkpoint variant/num_classes does not match requested configuration')
    if allow_unused_craft:
        if hasattr(model, 'craft'):
            raise ValueError('--allow-unused-craft is only for variants without a CRAFT module')
        ignored = sorted(k for k in state if k.startswith('craft.'))
        state = {k: v for k, v in state.items() if k not in ignored}
        meta = {**meta, 'explicitly_ignored_unused_keys': ignored}
    model.load_state_dict(state, strict=True)
    return meta


def device_from_name(name):
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; use --device cpu for smoke checks')
    return device
