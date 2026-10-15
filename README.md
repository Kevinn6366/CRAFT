# CRAFT: Cross-Modal RGB–Height Fusion for Fine-Grained Food Segmentation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **CRAFT**, using DINOv3-B/L and a project-adapted UPerNet decoder for RGB food image segmentation.

## Abstract

Fine-grained food segmentation is challenging when ingredients have similar appearances and ambiguous boundaries. CRAFT couples semantic features with a learned height representation and exchanges information through spatial rectification, bidirectional cross-attention, and adaptive aggregation. During training, a geometric height target is constructed from the segmentation labels. Inference requires only an RGB image; the height representation is a geometric proxy rather than measured physical depth.

## Overview

The model consists of a DINOv3 encoder, a four-scale feature adapter, a UPerNet decoder with gated skip connections, and three CRAFT components:

1. **Mutual Spatial Rectification (MSR)** — filters the RGB and height streams using cross-stream spatial masks.
2. **Bidirectional Cross-Attention (BCA)** — exchanges global information between the two streams.
3. **Adaptive Gated Aggregation (AGA)** — combines the refined features using learned spatial gates.

Choose DINOv3-B or DINOv3-L with `--backbone dinov3_b` or `--backbone dinov3_l`. Model implementations are in `models/`.

## Requirements

Use Python 3.10 or newer. Install the PyTorch and torchvision builds appropriate for your CUDA environment, then install:

```bash
pip install -r requirements.txt
```

Download the pinned DINOv3 initialization for the selected backbone:

```bash
python -m tools.download_pretrained --backbone dinov3_b --output pretrained
# For DINOv3-L:
python -m tools.download_pretrained --backbone dinov3_l --output pretrained
```

## Dataset Preparation

Organize the dataset using the VOC2012 structure:

```text
dataset_root/
└── VOC2012/
    ├── JPEGImages/          # RGB images (*.jpg)
    ├── SegmentationClass/   # Semantic labels (*.png)
    └── ImageSets/
        └── Segmentation/
            ├── train.txt
            ├── val.txt
            └── test.txt
```

Each split file contains one image ID per line, without the extension. Splits must be disjoint. Labels use integer class IDs; background is 0. RGB-format UEC masks are read from their red channel. Height targets are generated during training from connected foreground regions.

Conversion scripts preserve the official test membership and split the training set with seed 11:

```bash
python -m tools.prepare_foodseg103 --source /path/to/FoodSeg103 --output data/foodseg103
python -m tools.prepare_uecfoodpix --source /path/to/UEC_source --output data/uecfoodpix
```

To retain compatibility with the experiment checkpoints, `--num-classes` follows the original convention and **adds one output channel**. Use `104` for FoodSeg103 (105 output channels) and `102` for UECFoodPixComplete (103 output channels). FoodSeg103 foreground IDs are 1–103; the extra output channel is retained for checkpoint compatibility. Label values outside the output range are ignored.

## Training

DINOv3-B on FoodSeg103:

```bash
python train.py \
    --data-path data/foodseg103 \
    --num-classes 104 \
    --backbone dinov3_b \
    --backbone-weights pretrained/dinov3_b/model.safetensors \
    --batch-size 24 --epochs 90 --lr 1e-4 \
    --height-weight 10 --seed 11 --amp \
    --device cuda --output runs/foodseg103_b
```

For DINOv3-L, change the backbone to `dinov3_l`, use its corresponding pretrained file, and set `--batch-size 12`. For UECFoodPixComplete, change the dataset path and use `--num-classes 102`.

Training uses 512 × 512 inputs. `best.pt` is selected by validation mIoU; `last.pt` stores the full resume state. Fixed checkpoints are also saved at epochs 50 and 90. Resume with the same command and `--resume`; keep all other training settings unchanged.

## Validation

```bash
python validation.py \
    --data-path data/foodseg103 --num-classes 104 \
    --backbone dinov3_b --weights runs/foodseg103_b/best.pt \
    --batch-size 24 --device cuda \
    --output runs/foodseg103_b/validation.json
```

`val.py` is an alias for this validation entry point. Validation uses `val.txt` and the training input size recorded in the checkpoint.

## Test

```bash
python test.py \
    --data-path data/foodseg103 --num-classes 104 \
    --backbone dinov3_b --weights runs/foodseg103_b/best.pt \
    --device cuda --output runs/foodseg103_b/test_best
```

Formal testing uses `test.txt`, a 480 × 512 letterboxed grid, and batch size 1. DINO mIoU is computed from the dataset-accumulated confusion matrix over classes with nonzero union, including background. Boundary IoU, boundary F1, and contact-region IoU use per-image averaging under the same evaluation grid. The output contains metrics, per-image diagnostics, and class-ID prediction masks. Test results do not select checkpoints.

## Project Structure

```text
.
├── models/                 # DINOv3, UPerNet, CRAFT, and model construction
├── utils/                  # Data loading, losses, metrics, and checkpoint utilities
├── tools/                  # Dataset conversion and pretrained weight download
├── train.py                # Training and resume
├── validation.py           # Validation split evaluation
├── val.py                  # Validation alias
├── test.py                 # Formal test and boundary diagnostics
├── README.md
└── requirements.txt
```
