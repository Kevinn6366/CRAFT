# 导入标准库和第三方库
import os  # 用于操作系统功能，如文件和路径管理
from functools import partial  # 用于部分应用函数
import torch.nn.functional as F
from tqdm import tqdm
# 导入Numpy和PyTorch相关库
import numpy as np  # 用于数值计算
import torch  # 导入PyTorch库
import torch.optim as optim  # 导入PyTorch的优化器模块
from torch.utils.data import DataLoader  # 导入数据加载器模块
import time
import datetime
import subprocess
from torch.utils.tensorboard import SummaryWriter

from model.unet_training import CE_Loss, Dice_loss, Focal_Loss , Height_MSE_Loss# 确保引入这三个函数

# 导入自定义模块和模型
from model.unet_resnet import Unet  # 导入U-Net模型
from model.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init  # 导入与U-Net训练相关的函数（如学习率调度、设置优化器学习率和权重初始化）
from utils.dataloader import UnetDataset, unet_dataset_collate  # 导入U-Net数据集及其合并函数
from utils.utils import seed_everything, worker_init_fn  # 导入一些工具函数（如随机种子设置、展示配置和初始化工作线程）
#from utils.train_and_eval import train_one_epoch, evaluate  # 导入训练一个epoch的函数
from utils.train_and_eval import evaluate
from utils.create_exp_folder import create_exp_folder  # 用于创建实验目录
from utils.plot_results import plot_training_curves  # 绘制模型结果图

# GPU占用计算函数
def get_gpu_usage():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
        encoding='utf-8'
    )
    # 输出类似： '1203, 6144\n'
    used, total = map(int, result.strip().split(','))

    return used
    # print(f"GPU 显存占用: {used} MB / {total} MB")


def create_model(num_classes, weights):
    # 创建一个Unet模型实例，num_classes指定了分类的数量（输出通道数）
    model = Unet(num_classes=num_classes)
    # 使用 weights_init 函数初始化模型权重
    weights_init(model)

    # 检查是否提供了模型路径，即是否需要加载预训练权重
    if weights:
        # 获取当前模型的状态字典（即模型的参数）
        model_dict = model.state_dict()

        # 加载预训练模型的权重字典
        pretrained_dict = torch.load(weights, map_location='cpu')

        # 初始化三个列表和字典，用于存储加载和未加载的参数
        load_key, no_load_key, temp_dict = [], [], {}

        # 遍历预训练模型中的每个参数
        for k, v in pretrained_dict.items():
            # 如果当前模型中有对应的参数，并且形状匹配，则加载该参数
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v  # 将匹配的参数存储到临时字典中
                load_key.append(k)  # 记录成功加载的参数的Key
            else:
                no_load_key.append(k)  # 记录没有匹配上的参数的Key

        # 更新模型的状态字典，将预训练模型的参数加载到当前模型中
        model_dict.update(temp_dict)
        # 加载更新后的状态字典到模型中
        model.load_state_dict(model_dict)
    return model





def get_optimizer_and_lr(model, batch_size, total_epochs, momentum, weight_decay):
    # 初始化学习率（初始学习率为1e-4）
    Init_lr = 1e-4
    # 最小学习率是初始学习率的1%
    Min_lr = Init_lr * 0.01

    # 设置学习率衰减策略，'cos'代表余弦衰减
    lr_decay_type = 'cos'

    # 默认每个小批量（batch）的大小为16
    nbs = 16
    # 设置最大学习率限制
    lr_limit_max = 1e-4
    # 设置最小学习率限制
    lr_limit_min = 1e-4

    # 根据当前batch_size调整学习率，并确保它在最大和最小限制之间
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    # 根据当前batch_size调整最小学习率，并确保它在最大和最小限制之间
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # 使用Adam优化器，设置学习率、动量、权重衰减等参数,但是原论文用的是AdamW
    optimizer = optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999),
                           weight_decay=weight_decay)
    


    
    # 获取学习率调度器函数，根据衰减类型、初始学习率、最小学习率和训练轮次来计算学习率的变化
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, total_epochs)
    # 返回优化器和学习率调度器
    return optimizer, lr_scheduler_func



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(args):
    seed_everything(11)  # 设置种子
    exp_folder,weights_folder = create_exp_folder()
    num_classes = args.num_classes + 1  # 类别加上背景类
    start_epoch = args.start_epoch #继续训练轮次



    total_epochs = start_epoch + args.epochs
    batch_size = args.batch_size  # 设置batch size和类别数
    num_workers = args.workers  # 计算可用的工作线程数，通常取CPU核心数、batch_size和8中的最小值

    # 选择设备（GPU 如果可用，否则使用 CPU）
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 调用函数获取新的exp文件夹和weights文件夹路径
    if args.log_dir:
        log_dir = args.log_dir
    #tensorboard logger 创建
    writer = SummaryWriter(log_dir=log_dir)

    input_shape = [512, 512]  # 一定要是32的整数倍

    # 创建训练数据集对象
    # args.data_path: 数据集的根路径， input_shape: 输入图像的尺，num_classes: 输出类别数，表示分割任务中的类别数
    # augmentation=True: 是否采用数据增强，txt_name="train.txt": 指定用于加载训练数据的文本文件名
    train_dataset = UnetDataset(args.data_path, input_shape, num_classes, augmentation=True, txt_name="train.txt")
    val_dataset = UnetDataset(args.data_path, input_shape, num_classes, augmentation=False, txt_name="val.txt")

    # 加载训练集的DataLoader
    train_loader = DataLoader(train_dataset,
                              shuffle=True,  # 是否打乱数据（训练时一般打乱数据）
                              batch_size=batch_size,  # 每个批次加载的样本数量
                              num_workers=num_workers,  # 加载数据时使用的子进程数量（并行化加载）
                              pin_memory=True,  # 是否将数据复制到CUDA的内存中（如果使用GPU训练，通常设置为True）
                              drop_last=False,  # 如果数据集大小不能被batch_size整除，是否丢弃最后不完整的批次
                              collate_fn=unet_dataset_collate,  # 定义如何将多个样本合并成一个批次，通常是处理不同大小的图像
                              sampler=None,  # 是否使用自定义采样器，默认为None，表示按顺序加载
                              worker_init_fn=partial(worker_init_fn, rank=0, seed=11))  # 初始化worker时的函数，通常用于设置随机种子

    # 加载验证集的DataLoader
    val_loader = DataLoader(val_dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=unet_dataset_collate,
                            sampler=None,
                            worker_init_fn=partial(worker_init_fn, rank=0, seed=11))

    # 创建模型
    model = create_model(num_classes=num_classes, weights=args.weights)
    model = model.to(device)

    # 如果启用混合精度训练（amp），使用GradScaler
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # scaler = torch.amp.GradScaler(device='cuda') if args.amp else None  # 在pytorch2.7.1版本可以运用该代码可以去除警告信息

    # 获取优化器和学习率调度器
    optimizer, lr_scheduler_func = get_optimizer_and_lr(model, batch_size, total_epochs, args.momentum, args.weight_decay)

    # 训练开始
    start_time = time.time()
    #weights_folder
    best_acc = 0.0  # 最优准确率初始化为0
    best_model_path = os.path.join(weights_folder, f"best_model_{args.num_classes}.pth")  # 最优模型保存路径
    last_model_path = os.path.join(weights_folder, f"last_model_{args.num_classes}.pth")  # 最后一轮模型保存路径
    train_losses = []
    val_losses = []
    val_metrics_history = []
    # 是否使用focal loss来防止正负样本不平衡，是否给不同种类赋予不同的损失权值，默认是平衡的。
    focal_loss = True
    #  种类少（几类）时，设置为True
    #  种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #  种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    dice_loss = True

    #   开始模型训练
    for epoch in range(start_epoch, total_epochs):
        gpu_used = get_gpu_usage() # 计算使用GPU内存
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)  # 学习率调度函数

        # 每个epoch进行训练
        loss = train_one_epoch(model, optimizer, train_loader, device, dice_loss, focal_loss,
                               gpu_used, num_classes, scaler, epoch, total_epochs, writer)

        train_losses.append(loss)  # 保存训练过程中的loss值

        
        # #tensorboard 记录 训练损失
        metrics = evaluate(model, val_loader, device, dice_loss, focal_loss, num_classes)
        writer.add_scalar('Train/Epoch_Loss', loss, epoch)
        # 在验证集上评估模型
        writer.add_scalar('Val/Loss', metrics["Loss"], epoch)
        writer.add_scalar('Val/Mean_Accuracy', metrics["Mean Accuracy"], epoch)
        writer.add_scalar('Val/Mean_IoU', metrics["Mean IoU"], epoch)
        val_losses.append(metrics["Loss"])
        val_metrics_history.append(metrics)

        current_acc = float(metrics["Mean Accuracy"])  # 转换为浮动准确率（百分比）

        # 更新最优准确率并保存最优模型
        # 保存最优模型
        if current_acc > best_acc:
            best_acc = current_acc
            torch.save(model.state_dict(), best_model_path)

        # 保存最后一次模型
        torch.save(model.state_dict(), last_model_path)

    # 打印训练的总时长
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

    #  最后画图
    plot_training_curves(train_losses, val_losses, val_metrics_history, weights_folder)

# 训练val1 函数
def train_one_epoch(model, optimizer, data_loader, device, dice_loss, focal_loss, gpu_used, num_classes, scaler, epoch, total_epochs,writer):
    model.train() # 确保模型处于训练模式
    
    total_loss = 0
    total_accuracy = 0
    
    # 进度条
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

        # 【Innovation Point】前向传播：同时获取 主输出 和 辅助输出
        if scaler is not None: # 使用混合精度训练 (AMP)
            with torch.cuda.amp.autocast():
               
                outputs, aux_outputs, pred_heights = model(imgs) 
                
                # 计算主 Loss (原尺寸)
                loss_main = 0
                if focal_loss:
                    loss_main += Focal_Loss(outputs, pngs, cls_weights=None, num_classes=num_classes)
                else: # 使用 CrossEntropy
                    loss_main += CE_Loss(outputs, pngs, cls_weights=None, num_classes=num_classes)
                
                if dice_loss:
                    loss_main += Dice_loss(outputs, pngs)
                
                # 计算辅助 Loss (H/4 尺寸) 
                # 关键步骤：必须把标签 (Ground Truth) 缩小 4 倍以匹配 aux_outputs
                # 使用 nearest 插值，因为标签是整数类别，不能有小数
                target_small = F.interpolate(pngs.unsqueeze(1).float(), scale_factor=0.25, mode='nearest').squeeze(1).long()
                
                loss_aux = 0
                if focal_loss:
                    loss_aux += Focal_Loss(aux_outputs, target_small, cls_weights=None, num_classes=num_classes)
                else:
                    loss_aux += CE_Loss(aux_outputs, target_small, cls_weights=None, num_classes=num_classes)
                
                if dice_loss:
                    loss_aux += Dice_loss(aux_outputs, target_small)
                loss_height = Height_MSE_Loss(pred_heights, height_maps)
                # 总 Loss 加权 (语义一致性监督) 
                # 0.4 是推荐的权重系数   
                loss = loss_main + 0.5 * loss_height
                #+ 0.4 * loss_aux

            # 反向传播 (AMP)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        else: # 不使用混合精度
            outputs, aux_outputs, pred_heights = model(imgs)    # 获取三输出
            # --- 主 Loss ---
            loss_main = 0
            if focal_loss:
                loss_main += Focal_Loss(outputs, pngs, cls_weights=None, num_classes=num_classes)
            else:
                loss_main += CE_Loss(outputs, pngs, cls_weights=None, num_classes=num_classes)
            if dice_loss:
                loss_main += Dice_loss(outputs, pngs)

            #  辅助 Loss 
            target_small = F.interpolate(pngs.unsqueeze(1).float(), scale_factor=0.25, mode='nearest').squeeze(1).long()
            loss_aux = 0
            if focal_loss:
                loss_aux += Focal_Loss(aux_outputs, target_small, cls_weights=None, num_classes=num_classes)
            else:
                loss_aux += CE_Loss(aux_outputs, target_small, cls_weights=None, num_classes=num_classes)
            if dice_loss:
                loss_aux += Dice_loss(aux_outputs, target_small)
            loss_height = Height_MSE_Loss(pred_heights, height_maps)
            # 总 Loss 
            loss = loss_main  + 0.5 * loss_height
            #+ 0.2 * loss_aux
            loss.backward()
            optimizer.step()
        # 计算精度用于显示 (只看主输出)
        with torch.no_grad():
            # [B, C, H, W] -> [B, H, W]
            _pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            accuracy = torch.mean((_pred == pngs).float()) 
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        global_step = epoch * len(data_loader) + iteration
        # #tensorboard 记录 batch级别的loss和auc
        writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
        writer.add_scalar('Train/Batch_Accuracy', accuracy.item(), global_step) # 顺便记录一下Acc
        # 更新进度条
        pbar.set_postfix(**{'loss': total_loss / (iteration + 1), 
                            'acc': total_accuracy / (iteration + 1), 
                            'lr': get_lr(optimizer)})
        
        # pbar.set_postfix(**{
        #     'Main': loss_main.item() if isinstance(loss_main, torch.Tensor) else loss_main,
        #     'Aux': loss_aux.item() if isinstance(loss_aux, torch.Tensor) else loss_aux,
        #     'Height': loss_height.item(), # 看看这个是不是几百几千
        #     'Total': total_loss / (iteration + 1)
        # })
                            
    return total_loss / len(data_loader)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--weights", default="/home/u241003661121/U-Net/run/train/exp157/weights/best_model_104.pth",
                        help="Path to the directory containing model weights")
    parser.add_argument("--data-path", default="/home/u241003661121/U-Net/FoodSeg103", help="VOCdevkit root")
    parser.add_argument("--num-classes", default=104, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=50, type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument("--workers", default=0, type=int, metavar="N",
                        help="number of data loading workers (default: 0, meaning data loading runs in main process)")
    parser.add_argument('--lr', default=1e-5, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.90, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool, help="Use torch.cuda.amp for mixed precision training")
    # 起始轮次
    parser.add_argument("--start-epoch", default=50, type=int, 
                        help="Start epoch index ")
    parser.add_argument("--log-dir", default="/home/u241003661121/U-Net/logs/log1", help="Tensorboard log directory")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)


