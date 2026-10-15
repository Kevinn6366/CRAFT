import torch
import numpy as np

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
