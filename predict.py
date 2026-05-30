import colorsys
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.unet_resnet import Unet
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.create_exp_folder import create_val_exp_folder


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def load_model(model_path, num_classes, device):
    net = Unet(num_classes=num_classes)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    net.to(device)
    return net


def detect_image(file_path, model, num_classes, exp_folder, mix_type=True):
    try:
        image = Image.open(file_path)
    except (FileNotFoundError, IOError) as e:
        print(f"Error opening image: {e}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = cvtColor(image)
    old_img = image.copy()

    input_shape = [480, 480]
    orininal_h, orininal_w = np.array(image).shape[:2]
    image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))

    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    if num_classes <= 21:
        colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                  (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                  (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                  (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 128)]
    else:
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    with torch.no_grad():
        images = torch.from_numpy(image_data).to(device)

        pr = model(images)[0]

        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

        pr = pr[int((input_shape[0] - nh) // 2): int((input_shape[0] - nh) // 2 + nh),
             int((input_shape[1] - nw) // 2): int((input_shape[1] - nw) // 2 + nw)]
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

        pr = pr.argmax(axis=-1)

    seg_img = np.zeros((orininal_h, orininal_w, 3), dtype=np.uint8)
    for c in range(num_classes):
        seg_img[pr == c] = colors[c]

    if mix_type:
        old_img_np = np.array(old_img)
        alpha = 0.7
        blended_img = cv2.addWeighted(old_img_np, 1 - alpha, seg_img, alpha, 0)
        image = Image.fromarray(blended_img)
    else:
        image = Image.fromarray(np.uint8(seg_img))

    img_name = os.path.basename(file_path)
    mask_filename = os.path.splitext(img_name)[0] + "_mask.png"
    save_path = os.path.join(exp_folder, mask_filename)
    image.save(save_path)

    print(f"Mask saved at: {save_path}")


def predict(args):
    exp_folder = create_val_exp_folder()
    num_classes = args.num_classes + 1

    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.weights, num_classes, device)

    if os.path.isdir(args.data_path):
        file_paths = [str(p) for p in Path(args.data_path).rglob("*") if p.suffix in [".jpg", ".png", ".jpeg"]]
    elif os.path.isfile(args.data_path):
        file_paths = [args.data_path]
    else:
        raise ValueError(f"Unsupported input path: {args.data_path}")

    t_start = time_synchronized()

    for file_path in file_paths:
        if file_path.endswith((".jpg", ".png", ".jpeg")):
            detect_image(file_path, model, num_classes, exp_folder, mix_type=args.mix_type)

    t_end = time_synchronized()

    print(f"Inference time: {t_end - t_start:.2f}s")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Semantic segmentation inference")

    parser.add_argument("--data_path", type=str, help="Path to input image or directory")
    parser.add_argument("--weights", type=str, help="Path to model weights")
    parser.add_argument("--num-classes", type=int, help="Number of segmentation classes")
    parser.add_argument("--mix_type", action='store_true', help="Blend prediction overlay on original image")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    predict(args)
