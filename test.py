import torch
from torchinfo import summary
from model.unet_resnet import Unet  # 确保路径与你项目一致

# 1. 定义类别数量（FoodSeg103 是 103 类 + 1 背景）
num_classes = 104 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 实例化模型（这就是你报错缺失的 'model'）
model = Unet(num_classes=num_classes)
model.to(device)

# 3. 查看参数摘要
# col_names 可以让你看到更详细的参数量、显存占用和层级结构
summary(model, 
        input_size=(1, 3, 480, 480), 
        device=device,
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=3)