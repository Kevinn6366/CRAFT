"""Copy original FoodSeg103 to VOC without relabeling or touching originals."""
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from utils.common import file_hash, inspect_data, write_json


def run(source, output, seed=11):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    root = output / 'VOC2012'
    for folder in ('JPEGImages', 'SegmentationClass', 'ImageSets/Segmentation'):
        (root / folder).mkdir(parents=True)
    mapping, groups, counts = [], {}, {}
    for split in ('train', 'test'):
        lines = (source / 'ImageSets' / (split + '.txt')).read_text().splitlines()
        ids = [Path(line.split()[0]).stem for line in lines if line.strip()]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate source IDs')
        groups[split] = []
        for name in ids:
            image = source / 'Images/img_dir' / split / (name + '.jpg')
            mask = source / 'Images/ann_dir' / split / (name + '.png')
            corrected_image = None
            with Image.open(image) as im, Image.open(mask) as label:
                arr = np.array(label)
                if im.size != label.size and im.getexif().get(274) in (5, 6, 7, 8):
                    corrected_image = ImageOps.exif_transpose(im)
                effective_size = corrected_image.size if corrected_image is not None else im.size
                if effective_size != label.size or arr.ndim != 2:
                    raise ValueError(f'Invalid pair: {image}')
                values, pixels = np.unique(arr, return_counts=True)
                if any(int(v) not in set(range(104)) | {255} for v in values):
                    raise ValueError(f'Unexpected label IDs: {mask}: {values}')
                for v, n in zip(values, pixels):
                    counts[str(int(v))] = counts.get(str(int(v)), 0) + int(n)
            new = split + '_' + name
            if corrected_image is not None:
                corrected_image.save(root / 'JPEGImages' / (new + '.jpg'), quality=100, subsampling=0)
            else:
                shutil.copy2(image, root / 'JPEGImages' / (new + '.jpg'))
            shutil.copy2(mask, root / 'SegmentationClass' / (new + '.png'))
            groups[split].append(new)
            mapping.append({'id': new, 'source_split': split, 'source_id': name,
                            'exif_orientation_applied': corrected_image is not None, 'mask_sha256': file_hash(mask)})
        print(f'Validated/copied {split}: {len(ids)}', flush=True)
    shuffled = groups['train'].copy()
    random.Random(seed).shuffle(shuffled)
    n = int(len(shuffled) * 0.7)
    splits = {'train': sorted(shuffled[:n]), 'val': sorted(shuffled[n:]), 'test': groups['test']}
    for split, ids in splits.items():
        (root / 'ImageSets/Segmentation' / (split + '.txt')).write_text('\n'.join(ids) + '\n')
    shutil.copy2(source / 'category_id.txt', output / 'category_id.txt')
    meta = {'source': str(source), 'seed': seed, 'policy': '70/30 original train; official test untouched; labels unchanged',
            'splits': inspect_data(output, ['train.txt', 'val.txt', 'test.txt']),
            'label_pixel_counts': counts, 'mapping': mapping}
    write_json(output / 'conversion.json', meta)
    print(json.dumps(meta['splits']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=11)
    a = p.parse_args()
    run(a.source, a.output, a.seed)
