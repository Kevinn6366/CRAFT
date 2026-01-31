import torch
from utils.utils import cvtColor, preprocess_input
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import cv2
from scipy.ndimage import label, distance_transform_edt

class UnetDataset(Dataset):
    def __init__(self, data_path, input_shape, num_classes, augmentation=True, txt_name: str = "train.txt"):
        with open(os.path.join(data_path, "VOC2012/ImageSets/Segmentation", txt_name), "r") as f:
            self.annotation_lines = f.readlines()

        self.length = len(self.annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.data_path = data_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        # 1. 读取图片
        jpg = Image.open(os.path.join(self.data_path, "VOC2012/JPEGImages", name + ".jpg"))
        png = Image.open(os.path.join(self.data_path, "VOC2012/SegmentationClass", name + ".png"))

        # 2. 数据增强
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.augmentation)

        # 3. 预处理图片 [H, W, C] -> [C, H, W]
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        
        # 4. 处理标签 (关键修改：保持为索引，不要转One-Hot)
        png = np.array(png)
        
        # 将大于 num_classes 的值（通常是255作为忽略区域）设为 num_classes
        png[png >= self.num_classes] = self.num_classes
        height_map = self.generate_height_map(png)

        # -------------------------------------------------------
        # 这里不再生成 seg_labels (One-Hot)，直接返回 png
        # -------------------------------------------------------
        return jpg, png, png
    def generate_height_map(self, mask):
        # 1. 制作二值掩码 (Binary Mask)
        # 把所有是食物的地方标记为 1，背景为 0
        foreground_mask = (mask > 0) & (mask < self.num_classes)
        foreground_mask = foreground_mask.astype(np.uint8)

        # 如果全图都是背景，直接返回全0
        if np.sum(foreground_mask) == 0:
            return np.zeros_like(mask, dtype=np.float32)

        # 2. 【核心修改】连通域标记 (Connected Component Labeling)
        # label 函数会把不相连的物体标记成不同的数字 (1, 2, 3...)
        # labeled_array: 形状和 mask 一样，但里面是实例 ID
        # num_features: 找到了多少个独立的物体
        labeled_array, num_features = label(foreground_mask)

        # 初始化一个空的高度图
        final_height_map = np.zeros_like(mask, dtype=np.float32)

        # 3. 循环遍历每一个独立的物体 (Instance)
        for i in range(1, num_features + 1):
            # 3.1 取出当前这一个物体 (也就是 Mask 里等于 i 的部分)
            instance_mask = (labeled_array == i)

            # 3.2 对这单独一个物体算距离变换
            # 此时，dist 的最大值就是这个物体 "半径" (R_I)
            dist = distance_transform_edt(instance_mask)

            # 3.3 【关键】单独归一化
            # 小物体 R_I 小，大物体 R_I 大，但除完之后大家中心都是 1.0
            max_val = dist.max()
            if max_val > 0:
                dist = dist / max_val

            # 3.4 把算好的这块高度贴到总图上
            final_height_map += dist

        return final_height_map

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))

        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data = np.array(image, np.uint8)

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label


def unet_dataset_collate(batch):
    images = []
    pngs = []
    height_maps = [] 

    for img, png, h_map in batch:
        images.append(img)
        pngs.append(png)
        height_maps.append(h_map)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    
    # 高度图是浮点数，且不需要 Long 类型，用 FloatTensor
    height_maps = torch.from_numpy(np.array(height_maps)).type(torch.FloatTensor)

    return images, pngs, height_maps