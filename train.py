# 导入标准库和第三方库
import os
from functools import partial
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import datetime
import subprocess
from torch.utils.tensorboard import SummaryWriter

from model.unet_training import CE_Loss, Dice_loss, Focal_Loss , Height_MSE_Loss

# 导入自定义模块和模型
from model.unet_resnet import Unet
from model.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything, worker_init_fn
from utils.train_and_eval import evaluate
from utils.create_exp_folder import create_exp_folder
from utils.plot_results import plot_training_curves

def get_gpu_usage():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        used, total = map(int, result.strip().split(','))
        return used
    except Exception as e:
        return 0

def create_model(num_classes, weights):
    model = Unet(num_classes=num_classes)
    weights_init(model)

    if weights:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(weights, map_location='cpu')
        load_key, no_load_key, temp_dict = [], [], {}

        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v 
                load_key.append(k) 
            else:
                no_load_key.append(k) 

        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
    
    # 【已删除】：冻结 ResNet 的逻辑已移除，现在全量参数都会参与训练
    
    return model

def get_optimizer_and_lr(model, batch_size, total_epochs, momentum, weight_decay, args_lr):
    Init_lr = args_lr
    Min_lr = Init_lr * 0.01
    lr_decay_type = 'cos'
    nbs = 16
    lr_limit_max = 1e-4
    lr_limit_min = 1e-4

    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    
    # 显式过滤掉不需要梯度的参数
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=Init_lr_fit, 
                            betas=(momentum, 0.999), 
                            weight_decay=weight_decay)
    
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, total_epochs)
    
    return optimizer, lr_scheduler_func

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(args):
    seed_everything(11)
    exp_folder, weights_folder = create_exp_folder()
    num_classes = args.num_classes + 1
    start_epoch = args.start_epoch

    total_epochs = start_epoch + args.epochs
    batch_size = args.batch_size
    num_workers = args.workers

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.log_dir:
        log_dir = args.log_dir
    writer = SummaryWriter(log_dir=log_dir)

    input_shape = [512, 512]

    train_dataset = UnetDataset(args.data_path, input_shape, num_classes, augmentation=True, txt_name="train.txt")
    val_dataset = UnetDataset(args.data_path, input_shape, num_classes, augmentation=False, txt_name="val.txt")

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=True, drop_last=False, collate_fn=unet_dataset_collate,
                              worker_init_fn=partial(worker_init_fn, rank=0, seed=11))

    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, drop_last=False, collate_fn=unet_dataset_collate,
                            worker_init_fn=partial(worker_init_fn, rank=0, seed=11))

    # 创建模型并应用冻结逻辑
    model = create_model(num_classes=num_classes, weights=args.weights)
    model = model.to(device)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 获取优化器
    optimizer, lr_scheduler_func = get_optimizer_and_lr(model, batch_size, total_epochs, args.momentum, args.weight_decay, args.lr)

    start_time = time.time()
    best_acc = 0.0
    best_model_path = os.path.join(weights_folder, f"best_model_{args.num_classes}.pth")
    last_model_path = os.path.join(weights_folder, f"last_model_{args.num_classes}.pth")
    
    train_losses = []
    val_losses = []
    val_metrics_history = []
    
    focal_loss = True
    dice_loss = True

    for epoch in range(start_epoch, total_epochs):
        gpu_used = get_gpu_usage()
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch) 

        loss, loss_main, loss_height, loss_aux = train_one_epoch(
            model, optimizer, train_loader, device, dice_loss, focal_loss,
            gpu_used, num_classes, scaler, epoch, total_epochs, writer
        )

        train_losses.append(loss) 

        writer.add_scalar('Train_Epoch/Total_Loss', loss, epoch)
        writer.add_scalar('Train_Epoch/Loss_Main_Semantic', loss_main, epoch)
        writer.add_scalar('Train_Epoch/Loss_Height_Prior', loss_height, epoch)
        writer.add_scalar('Train_Epoch/Loss_Auxiliary', loss_aux, epoch)

        metrics = evaluate(model, val_loader, device, dice_loss, focal_loss, num_classes)
        
        writer.add_scalar('Val/Loss', metrics["Loss"], epoch)
        writer.add_scalar('Val/Mean_Accuracy', metrics["Mean Accuracy"], epoch)
        writer.add_scalar('Val/Mean_IoU', metrics["Mean IoU"], epoch)
        
        val_losses.append(metrics["Loss"])
        val_metrics_history.append(metrics)

        current_acc = float(metrics["Mean Accuracy"]) 

        if current_acc > best_acc:
            best_acc = current_acc
            torch.save(model.state_dict(), best_model_path)

        torch.save(model.state_dict(), last_model_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

    plot_training_curves(train_losses, val_losses, val_metrics_history, weights_folder)

def train_one_epoch(model, optimizer, data_loader, device, dice_loss, focal_loss, gpu_used, num_classes, scaler, epoch, total_epochs, writer):
    model.train() 
    total_loss = 0.0
    total_loss_main = 0.0
    total_loss_height = 0.0
    total_loss_aux = 0.0
    total_accuracy = 0.0
    
    loss_weights = {
        'main': 1.0,
        'height': 10, 
        'aux': 0.4
    }
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch + 1}/{total_epochs}', mininterval=0.3)
    
    for iteration, batch in enumerate(pbar):
        imgs, pngs, height_maps = batch
        imgs = imgs.to(device)
        pngs = pngs.to(device).long()
        height_maps = height_maps.to(device).float()
        
        current_max = height_maps.max()
        if current_max > 1.0:
            height_maps = height_maps / current_max
            
        optimizer.zero_grad()

        if scaler is not None: 
            with torch.cuda.amp.autocast():
                outputs, aux_outputs, pred_heights = model(imgs) 
                
                l_main = 0
                if focal_loss:
                    l_main += Focal_Loss(outputs, pngs, cls_weights=None, num_classes=num_classes)
                else: 
                    l_main += CE_Loss(outputs, pngs, cls_weights=None, num_classes=num_classes)
                if dice_loss:
                    l_main += Dice_loss(outputs, pngs)
                
                target_small = F.interpolate(pngs.unsqueeze(1).float(), scale_factor=0.25, mode='nearest').squeeze(1).long()
                l_aux = 0
                if focal_loss:
                    l_aux += Focal_Loss(aux_outputs, target_small, cls_weights=None, num_classes=num_classes)
                else:
                    l_aux += CE_Loss(aux_outputs, target_small, cls_weights=None, num_classes=num_classes)
                if dice_loss:
                    l_aux += Dice_loss(aux_outputs, target_small)
                
                l_height = Height_MSE_Loss(pred_heights, height_maps)
                
                loss = (loss_weights['main'] * l_main) + \
                       (loss_weights['height'] * l_height) + \
                       (loss_weights['aux'] * l_aux)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        else: 
            outputs, aux_outputs, pred_heights = model(imgs)    
            
            l_main = 0
            if focal_loss:
                l_main += Focal_Loss(outputs, pngs, cls_weights=None, num_classes=num_classes)
            else:
                l_main += CE_Loss(outputs, pngs, cls_weights=None, num_classes=num_classes)
            if dice_loss:
                l_main += Dice_loss(outputs, pngs)

            target_small = F.interpolate(pngs.unsqueeze(1).float(), scale_factor=0.25, mode='nearest').squeeze(1).long()
            l_aux = 0
            if focal_loss:
                l_aux += Focal_Loss(aux_outputs, target_small, cls_weights=None, num_classes=num_classes)
            else:
                l_aux += CE_Loss(aux_outputs, target_small, cls_weights=None, num_classes=num_classes)
            if dice_loss:
                l_aux += Dice_loss(aux_outputs, target_small)
                
            l_height = Height_MSE_Loss(pred_heights, height_maps)
            
            loss = (loss_weights['main'] * l_main) + \
                   (loss_weights['height'] * l_height) + \
                   (loss_weights['aux'] * l_aux)
                   
            loss.backward()
            optimizer.step()
            
        val_l_main = l_main.item() if isinstance(l_main, torch.Tensor) else l_main
        val_l_height = l_height.item() if isinstance(l_height, torch.Tensor) else l_height
        val_l_aux = l_aux.item() if isinstance(l_aux, torch.Tensor) else l_aux
        val_loss = loss.item()

        total_loss += val_loss
        total_loss_main += val_l_main
        total_loss_height += val_l_height
        total_loss_aux += val_l_aux

        with torch.no_grad():
            _pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            accuracy = torch.mean((_pred == pngs).float()) 
        total_accuracy += accuracy.item()
        
        global_step = epoch * len(data_loader) + iteration
        
        writer.add_scalar('Train_Batch/Total_Loss', val_loss, global_step)
        writer.add_scalar('Train_Batch/Loss_Main_Semantic', val_l_main, global_step)
        writer.add_scalar('Train_Batch/Loss_Height_Prior', val_l_height, global_step)
        writer.add_scalar('Train_Batch/Loss_Auxiliary', val_l_aux, global_step)
        writer.add_scalar('Train_Batch/Accuracy', accuracy.item(), global_step) 

        pbar.set_postfix(**{
            'L_Main': f"{val_l_main:.3f}", 
            'L_Height': f"{val_l_height:.3f}", 
            'L_Aux': f"{val_l_aux:.3f}",
            'acc': f"{total_accuracy / (iteration + 1):.3f}", 
            'lr': get_lr(optimizer)
        })
                            
    num_batches = len(data_loader)
    return total_loss / num_batches, total_loss_main / num_batches, total_loss_height / num_batches, total_loss_aux / num_batches

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--weights", default="",
                        help="Path to the directory containing model weights")
    parser.add_argument("--data-path", default="/home/u241003661121/U-Net/FoodSeg103", help="VOCdevkit root")
    parser.add_argument("--num-classes", default=104, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=60, type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument("--workers", default=0, type=int, metavar="N",
                        help="number of data loading workers")
                        
    parser.add_argument('--lr', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.90, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--amp", default=True, type=bool, help="Use torch.cuda.amp")
    parser.add_argument("--start-epoch", default=0, type=int, help="Start epoch index")
    parser.add_argument("--log-dir", default="/home/u241003661121/U-Net/logs/log1", help="Tensorboard log directory")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)