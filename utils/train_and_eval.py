import os
import torch
import numpy as np
from model.unet_training import CE_Loss, Dice_loss, Focal_Loss, Height_MSE_Loss
from utils.utils import get_lr
from torch.cuda.amp import autocast, GradScaler
import time
import torch.nn.functional as F
import torch.nn.functional as F


class LogColor:
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    RESET = "\033[0m"
    BLUE = "\033[1;34m"


def pixel_accuracy(output, target, num_classes):
    with torch.no_grad():
        if isinstance(output, (tuple, list)):
            output = output[0]

        _, predicted = torch.max(output, 1)

        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        mask = (target >= 0) & (target < num_classes)
        correct = (predicted[mask] == target[mask]).float()

        correct_pixels = correct.sum().item()
        total_pixels = mask.sum().item()

        if total_pixels == 0:
            return 0.0

        return correct_pixels / total_pixels


def mean_accuracy(output, target, num_classes):
    with torch.no_grad():
        if isinstance(output, (tuple, list)):
            output = output[0]

        _, predicted = torch.max(output, dim=1)

        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        accuracies = []
        for i in range(num_classes):
            target_mask = (target == i)
            predicted_mask = (predicted == i)

            total = target_mask.sum().item()
            if total > 0:
                intersection = torch.logical_and(target_mask, predicted_mask).sum().item()
                acc = intersection / total
                accuracies.append(acc)

        if len(accuracies) == 0:
            return 0.0
        return sum(accuracies) / len(accuracies)


def mean_iou(output, target, num_classes):
    with torch.no_grad():
        if isinstance(output, (tuple, list)):
            output = output[0]

        _, predicted = torch.max(output, dim=1)

        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        ious = []
        for i in range(num_classes):
            target_mask = (target == i)
            pred_mask = (predicted == i)

            if target_mask.sum().item() > 0:
                intersection = torch.logical_and(target_mask, pred_mask).sum().item()
                union = torch.logical_or(target_mask, pred_mask).sum().item()
                ious.append(intersection / union if union > 0 else 0.0)

        if len(ious) == 0:
            return 0.0
        return sum(ious) / len(ious)


def frequency_weighted_iou(output, target, num_classes):
    with torch.no_grad():
        if isinstance(output, (tuple, list)):
            output = output[0]

        _, predicted = torch.max(output, 1)

        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        ious = []
        frequencies = []
        for i in range(num_classes):
            target_mask = (target == i)
            pred_mask = (predicted == i)

            if target_mask.sum().item() > 0:
                intersection = torch.logical_and(target_mask, pred_mask).sum().item()
                union = torch.logical_or(target_mask, pred_mask).sum().item()
                freq = target_mask.sum().item()

                frequencies.append(freq)
                ious.append((intersection / union) if union > 0 else 0.0)

        total = sum(frequencies)
        if total == 0:
            return 0.0
        fw_iou = sum(f * iou for f, iou in zip(frequencies, ious)) / total
        return fw_iou


def evaluate(model, val_loader, device, dice_loss, focal_loss, num_classes):

    cls_weights = np.ones([num_classes], np.float32)
    val_loss = 0
    model_eval = model.eval()
    model_eval = model_eval.cuda()

    total_pixel_acc = 0
    total_mean_acc = 0
    total_mean_iou = 0
    total_fw_iou = 0
    num_batches = len(val_loader)

    with torch.no_grad():
        for iteration, batch in enumerate(val_loader):
            imgs, pngs, height_maps = batch

            weights = torch.tensor(cls_weights).to(device)
            imgs = imgs.to(device)
            pngs = pngs.to(device)

            outputs = model_eval(imgs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, pngs)
                loss = loss + main_dice

            pixel_acc = pixel_accuracy(outputs, pngs, num_classes)
            mean_acc = mean_accuracy(outputs, pngs, num_classes)
            mean_iou_value = mean_iou(outputs, pngs, num_classes)
            fw_iou = frequency_weighted_iou(outputs, pngs, num_classes)

            total_pixel_acc += pixel_acc
            total_mean_acc += mean_acc
            total_mean_iou += mean_iou_value
            total_fw_iou += fw_iou
            val_loss += loss.item()

    avg_pixel_acc = total_pixel_acc / num_batches
    avg_mean_acc = total_mean_acc / num_batches
    avg_mean_iou = total_mean_iou / num_batches
    avg_fw_iou = total_fw_iou / num_batches
    avg_loss = val_loss / num_batches

    metrics = {
        'Pixel Accuracy': avg_pixel_acc,
        'Mean Accuracy': avg_mean_acc,
        'Mean IoU': avg_mean_iou,
        'Frequency Weighted IoU': avg_fw_iou,
        'Loss': avg_loss
    }

    return metrics
