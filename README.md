# CRAFT: Cross-Modal RGB-PD Fusion for Fine-Grained Food Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)


Official implementation of **CRAFT** — a hardware-friendly, end-to-end monocular RGB-PD segmentation framework for fine-grained food image parsing.

## Abstract

Fine-grained food image segmentation is crucial for intelligent food computing, yet blurry instance boundaries caused by complex ingredient layouts remain a fundamental challenge. Existing methods either suffer from prohibitive computational overhead due to heavy Transformer attention, hindering edge deployment, or rely on cumbersome, multi-stage detection-then-segmentation pipelines. To address these issues, we propose **CRAFT**, a hardware-friendly, efficient, end-to-end monocular RGB-PD segmentation framework. Specifically, we generate high-gradient height maps from a single RGB image as explicit geometric constraints, enhancing the network's perception of instance depth and contours. Built upon this, a three-stage cross-modal fusion architecture is designed to significantly improve boundary parsing in highly overlapped scenarios. Extensive experiments on two public datasets demonstrate that CRAFT outperforms state-of-the-art methods in terms of mIoU. Furthermore, with 71.44M parameters, our framework achieves a compelling balance between accuracy and efficiency on the NVIDIA Jetson Orin Nano edge device.

## Overview

CRAFT introduces pseudo-depth (PD) signals derived from monocular RGB inputs to guide cross-modal feature fusion. The architecture consists of three stages:

1. **Cross-modal spatial rectification** — bidirectional mask generation for mutual feature filtering
2. **Transformer-based cross-attention** — global context exchange between RGB and height streams
3. **Spatial-gated aggregation** — pixel-wise learnable fusion weights

The entire pipeline is end-to-end, single-stage, and optimized for edge deployment.

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0
matplotlib>=3.7.0
scipy>=1.10.0
tqdm>=4.65.0
tensorboard>=2.13.0
```

Install with:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Organize your dataset following the VOC2012 structure:

```
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

Each line in `train.txt` / `val.txt` / `test.txt` contains the image filename (without extension).

## Training

```bash
python train.py \
    --data-path /path/to/dataset_root \
    --num-classes 104 \
    --batch-size 18 \
    --epochs 120 \
    --lr 1e-4 \
    --device cuda \
    --log-dir /path/to/logs
```

## Validation

```bash
python val.py \
    --data-path /path/to/dataset_root \
    --weights /path/to/best_model.pth \
    --num-classes 104 \
    --device cuda
```

## Inference

Single image:

```bash
python predict.py \
    --data_path /path/to/image.jpg \
    --weights /path/to/model.pth \
    --num-classes 104 \
    --mix_type
```

Directory batch inference:

```bash
python predict.py \
    --data_path /path/to/image_dir/ \
    --weights /path/to/model.pth \
    --num-classes 104 \
    --mix_type
```

The `--mix_type` flag blends the prediction overlay onto the original image. Omit it for raw mask output.

## Project Structure

```
.
├── model/
│   ├── CRAFT.py              # CRAFT cross-modal fusion module
│   ├── resnet_backbone.py    # ResNet-50 feature extractor
│   ├── unet_resnet.py        # UPerNet decoder
│   └── unet_training.py      # Loss functions and training utilities
├── utils/
│   ├── dataloader.py         # Dataset and data augmentation
│   ├── train_and_eval.py     # Evaluation metrics
│   ├── utils.py              # Image preprocessing utilities
│   ├── create_exp_folder.py  # Experiment directory manager
│   └── plot_results.py       # Training curve plotting
├── train.py                  # Training entry point
├── val.py                    # Validation entry point
├── predict.py                # Inference entry point
├── shell/
│   ├── change.py             
│   └── remove_jpg_suffix.py  # Text file path cleaner
└── requirements.txt
```
