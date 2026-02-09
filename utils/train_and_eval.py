import os
import torch
import numpy as np
from model.unet_training import CE_Loss, Dice_loss, Focal_Loss, Height_MSE_Loss
from utils.utils import get_lr
from torch.cuda.amp import autocast, GradScaler
import time
import torch.nn.functional as F

class LogColor:
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    RESET = "\033[0m"
    BLUE = "\033[1;34m"



def pixel_accuracy(output, target, num_classes):
    """
    计算像素准确率，自动忽略 target >= num_classes 的区域
    """
    with torch.no_grad():
        # 如果 output 是元组，取第一个 (Deep Supervision)
        if isinstance(output, (tuple, list)):
            output = output[0]
            
        _, predicted = torch.max(output, 1) # [N, H, W]
        
        # 维度修正：如果 target 是 [N, 1, H, W]，压成 [N, H, W]
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        # 核心修复：只计算 target 在 [0, num_classes-1] 范围内的像素
        # 这样 105 (背景/忽略) 也就不会被算作错误了
        mask = (target >= 0) & (target < num_classes)
        
        # 应用掩码
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

            # 如果该类别在 GT 中根本不存在，就不应该计入分母
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
            
            # 只在 target 存在的类别计算
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
            
            # 同样只统计存在的类
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


#  Train Loop (基本没变，加了一点稳健性)


def train_one_epoch(model, optimizer, train_loader, device, dice_loss, focal_loss,
                    gpu_used, num_classes, scaler, epoch, train_epoch):

    cls_weights = np.ones([num_classes], np.float32)
    epoch_loss = 0.0
    model_train = model.train()
    model_train = model_train.cuda()

    for iteration, batch in enumerate(train_loader):
        imgs, pngs, labels = batch
        
        weights = torch.tensor(cls_weights).to(device)
        imgs = imgs.to(device)
        pngs = pngs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # --- Forward ---
        if scaler is None:
            outputs = model_train(imgs)
            
            # 这里的 Loss 函数你之前的 unet_training.py 已经修好了，兼容 Tuple
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            loss.backward()
            optimizer.step()
        else:
            with autocast():
                outputs = model_train(imgs)
                
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        epoch_loss += loss.item()

        # --- Logging ---
        # 兼容 Deep Supervision 的输出打印，防止报错
        if isinstance(outputs, (tuple, list)):
             out_tensor = outputs[0]
        else:
             out_tensor = outputs
             
        if iteration == 0:
            print(f"{LogColor.GREEN}Epoch{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}data_num{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}GPU Mem{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}Loss{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}LR{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}Image_size{LogColor.RESET}{' ' * 12}")

        if iteration % 1 == 0:
            a = len(train_loader) if len(train_loader) < 1 else 1
            
            Epoch_len = len("Epoch") + 12 - len(str(f"{epoch + 1}/{train_epoch}"))
            batch_len = len("data_num") + 12 - len(str(f"{iteration + a}/{len(train_loader)}"))
            GPU_len = len("GPU Mem") + 12 - len(str(f"{gpu_used:.2f} MB"))
            Loss_len = len("Loss") + 12 - len(str(f"{loss.item():.8f}"))
            LR_len = len("LR") + 12 - len(str(f"{get_lr(optimizer):.8f}"))

            print(f"\r{epoch + 1}/{train_epoch}{' ' * Epoch_len}"
                  f"{iteration + a}/{len(train_loader)}{' ' * batch_len}"
                  f"{gpu_used:.2f} MB{' ' * GPU_len}"
                  f"{loss.item():.8f}{' ' * Loss_len}"
                  f"{get_lr(optimizer):.8f}{' ' * LR_len}"
                  f"{imgs.shape[2]}", end='', flush=True)

    print(f"{LogColor.GREEN}")
    time.sleep(1)
    return epoch_loss / len(train_loader)




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
            # labels = labels.to(device)

            outputs= model_eval(imgs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # 1. Loss 计算 (CE_Loss 和 Dice_loss 内部已经处理了 Tuple，所以直接传 outputs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs,pngs)
                loss = loss + main_dice
            
            # 在计算指标前，如果 outputs 是元组，必须解包
            # 指标函数现在内部都有 check，但 evaluate 这里也可以显式处理一下
            # 为了安全起见，我们让 metric 函数内部去处理 isinstance，这里就不重复写了
            # 只需要传原始 outputs 进去即可，因为下面的 metric 函数我都加上了 tuple 检查

            # 3. 计算指标
            # 注意：Pixel Accuracy 需要 num_classes 参数来做 mask 掩码
            pixel_acc = pixel_accuracy(outputs, pngs, num_classes) 
            mean_acc = mean_accuracy(outputs, pngs, num_classes)
            mean_iou_value = mean_iou(outputs, pngs, num_classes)
            fw_iou = frequency_weighted_iou(outputs, pngs, num_classes)

            total_pixel_acc += pixel_acc
            total_mean_acc += mean_acc
            total_mean_iou += mean_iou_value
            total_fw_iou += fw_iou
            val_loss += loss.item()

            if iteration == 0:
                epoch_len = len("Epoch") + 12
                data_num_len = len("data_num") - len("data_num") + 12
                Pixelacc_len = len("GPU Mem") - len("Pixelacc") + 12
                Meanacc_len = len("Loss") - len("Meanacc") + 12
                Meaniou_len = len("LR") - len("Meaniou") + 12

                print(f"{' ' * epoch_len}"
                      f"{LogColor.RED}data_num{LogColor.RESET}{' ' * data_num_len}"
                      f"{LogColor.RED}Pixelacc{LogColor.RESET}{' ' * Pixelacc_len}"
                      f"{LogColor.RED}Meanacc{LogColor.RESET}{' ' * Meanacc_len}"
                      f"{LogColor.RED}Meaniou{LogColor.RESET}{' ' * Meaniou_len}"
                      f"{LogColor.RED}Fwiou{LogColor.RESET}")

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

    # 打印部分
    epoch_len = len("Epoch") + 12
    batch_len = data_num_len + len("data_num") - len(str(f"{iteration + 1}/{len(val_loader)}"))
    avg_pixel_acc_len = Pixelacc_len + len("Pixelacc") - len(str(f"{avg_pixel_acc:.2f}"))
    avg_mean_acc_len = Meanacc_len + len("Meanacc") - len(str(f"{avg_mean_acc:.2f}"))
    avg_Mean_iou_len = Meaniou_len + len("Meaniou") - len(str(f"{avg_mean_iou:.2f}"))

    print(f"{' ' * (epoch_len)}"
          f"{iteration + 1}/{len(val_loader)}{' ' * batch_len}"
          f"{avg_pixel_acc:.2f}{' ' * avg_pixel_acc_len}"
          f"{avg_mean_acc:.2f}{' ' * avg_mean_acc_len}"
          f"{avg_mean_iou:.2f}{' ' * avg_Mean_iou_len}"
          f"{avg_fw_iou:.2f}", end='', flush=True)
    print(f"\n{LogColor.GREEN}")
    time.sleep(1)

    return metrics