import torch
import torch.nn as nn
from model.resnet_backbone import resnet50
from model.cbam import CBAM
#unet_resnet.py
import torchvision.ops as ops
# 定义一个 U-Net 解码模块（上采样模块）

class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformConv2d, self).__init__()
        
        # 偏移量卷积
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size * kernel_size, 
                                     kernel_size=kernel_size, 
                                     stride=stride, 
                                     padding=padding, 
                                     bias=True)
        
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        self.dcn = ops.DeformConv2d(in_channels, 
                                    out_channels, 
                                    kernel_size=kernel_size, 
                                    stride=stride, 
                                    padding=padding, 
                                    bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        return self.dcn(x, offset)
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_size) #CBAM模块

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        outputs = self.cbam(outputs)#CBAM模块
        return outputs


# 定义 U-Net 主体结构
class Unet(nn.Module):
    def __init__(self, num_classes=21):
        super(Unet, self).__init__()

        # 使用 ResNet50 作为编码器
        self.resnet = resnet50() 
        
        in_filters = [192, 512, 1024, 3072] 
        out_filters = [64, 128, 256, 512] 
        self.encoder_dcn = DeformConv2d(2048, 2048)

        # 定义 4 层上采样模块
        self.up_concat4 = unetUp(in_filters[3], out_filters[3]) 
        self.up_concat3 = unetUp(in_filters[2], out_filters[2]) 
        self.up_concat2 = unetUp(in_filters[1], out_filters[1]) 
        self.up_concat1 = unetUp(in_filters[0], out_filters[0]) 

        # 最后的上采样和卷积
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2), 
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1), 
            nn.ReLU(),
        )

        # 主分类头
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        # 高度图分类头
        self.height_head = nn.Sequential(
            nn.Conv2d(out_filters[0], 1, kernel_size=1, bias=False),
            nn.Sigmoid() 
        )

      
        # 辅助分类头 (Auxiliary Head)
        # 位置：对应 up2 的输出 (Decoder倒数第二层)，特征图大小为 H/4 * W/4
        # 输入通道：out_filters[1] 即 128
   
        self.aux_head = nn.Sequential(
            nn.Conv2d(out_filters[1], 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, inputs):
        # 编码器提取五层特征图
        [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        feat5 = self.encoder_dcn(feat5)

        # 解码过程
        up4 = self.up_concat4(feat4, feat5) 
        up3 = self.up_concat3(feat3, up4) 
        up2 = self.up_concat2(feat2, up3) 
        # 在 H/4 处 (up2) 截取特征进行辅助预测
        # 仅在训练模式 (self.training) 下计算，节省推理时间
        aux_out = None
        if self.training:
            aux_out = self.aux_head(up2)

        up1 = self.up_concat1(feat1, up2) 

        if self.up_conv is not None:
            up1 = self.up_conv(up1)
        # 分支 A: 语义分割 (Segmentation)
        final = self.final(up1) # [B, num_classes, H, W]

        # 分支 B: 高度图预测 (Height Estimation)
        pred_height = self.height_head(up1) # [B, 1, H, W]
        # 训练时返回*三*个值，匹配 train.py 的解包需求
        if self.training:
            return final, aux_out, pred_height
        else:
            return final