"""Download and verify the pinned DINOv3 encoder weights."""
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
from models.dinov3 import SPECS
from utils.common import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone', choices=SPECS, default='dinov3_b')
    p.add_argument('--output', default='pretrained')
    args = p.parse_args()
    spec = SPECS[args.backbone]
    path = hf_hub_download(repo_id='timm/' + spec['model'],
                           filename='model.safetensors', revision=spec['revision'],
                           local_dir=Path(args.output) / args.backbone)
    if file_hash(path) != spec['sha256']:
        raise ValueError('Downloaded encoder SHA256 mismatch')
    print(path)


if __name__ == '__main__':
    main()
