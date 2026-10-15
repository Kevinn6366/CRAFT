"""Convert the official manual UEC-FoodPIX Complete release, preserving IDs.

The official masks encode class IDs in red, not grayscale luminance.
Only the 9000 official training images are split 70/30 with seed 11.
"""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(source, output, seed=11):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    groups = {}
    for split, count in [('train', 9000), ('test', 1000)]:
        ids = [x.strip() for x in (source / f'{split}{count}.txt').read_text().splitlines() if x.strip()]
        if len(ids) != count or len(set(ids)) != count or any(not x.isdigit() for x in ids):
            raise ValueError(f'Invalid official {split} list')
        groups[split] = ids
    if set(groups['train']) & set(groups['test']):
        raise ValueError('Official splits overlap')
    root = output / 'VOC2012'
    for folder in ['JPEGImages', 'SegmentationClass', 'ImageSets/Segmentation']:
        (root / folder).mkdir(parents=True)
    mapping, counts = [], np.zeros(103, dtype=np.int64)
    for split, ids in groups.items():
        for name in ids:
            folder = source / 'UECFoodPIXCOMPLETE' / split
            image, mask = folder / 'img' / (name + '.jpg'), folder / 'mask' / (name + '.png')
            with Image.open(image) as im, Image.open(mask) as gt:
                arr = np.array(gt)
                if im.size != gt.size or arr.ndim != 3 or arr.shape[2] not in (3, 4):
                    raise ValueError(f'Unexpected image/mask format: {name}')
                if np.any(arr[:, :, 1:3]) or arr[:, :, 0].max() > 102:
                    raise ValueError(f'Unexpected mask encoding: {name}')
                if arr.shape[2] == 4 and np.any(arr[:, :, 3] != 255):
                    raise ValueError(f'Nonopaque mask alpha: {name}')
                label = arr[:, :, 0]
                counts += np.bincount(label.ravel(), minlength=103)
            shutil.copy2(image, root / 'JPEGImages' / (name + '.jpg'))
            dest = root / 'SegmentationClass' / (name + '.png')
            Image.fromarray(label).save(dest)
            mapping.append({'id': name, 'official_split': split, 'image_sha256': sha(image),
                            'source_mask_sha256': sha(mask), 'label_sha256': sha(dest)})
        print(f'Converted {split}: {len(ids)}', flush=True)
    shuffled = groups['train'].copy()
    random.Random(seed).shuffle(shuffled)
    splits = {'train': sorted(shuffled[:6300]), 'val': sorted(shuffled[6300:]), 'test': groups['test']}
    hashes = {}
    for split, ids in splits.items():
        p = root / 'ImageSets/Segmentation' / (split + '.txt')
        p.write_text('\n'.join(ids) + '\n')
        hashes[split] = {'count': len(ids), 'sha256': sha(p)}
    shutil.copy2(source / 'category.txt', output / 'category.txt')
    meta = {'source_url': 'https://mm.cs.uec.ac.jp/uecfoodpix/UECFOODPIXCOMPLETE.tar',
            'dataset': 'UEC-FoodPIX Complete (manual)', 'seed': seed, 'splits': hashes,
            'policy': '70/30 official training only; official test unchanged',
            'labels': 'red-channel IDs 0..102 preserved; no luminance conversion',
            'num_classes_arg': 102, 'output_channels': 103,
            'label_pixel_counts': counts.tolist(), 'mapping': mapping}
    (output / 'conversion.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(hashes), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=11)
    a = p.parse_args()
    run(a.source, a.output, a.seed)
