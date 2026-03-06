import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入你原有的 ResNet50 Backbone 和 CBAM 模块
from model.resnet_backbone import resnet50
from model.cbam import CBAM

# ==========================================
# 1. 定义 UperNet 的核心组件：金字塔池化模块 (PPM)
# 作用：获取全局和不同尺度的局部上下文信息，解决“只看局部认不出这盘菜”的问题
# ==========================================
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.features = nn.ModuleList()
        for bin_size in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), size=x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

# ==========================================
# 2. 定义 UperNet 解码器
# 作用：融合多尺度特征，代替原来粗暴的 UnetUp Concat
# ==========================================
class UperNetDecoder(nn.Module):
    def __init__(self, in_channels_list, fpn_dim=512):
        super(UperNetDecoder, self).__init__()
        
        # PPM模块，应用于最深层特征 (ResNet 的 feat5)
        self.ppm = PPM(in_channels_list[-1], fpn_dim)
        self.ppm_conv = nn.Sequential(
            nn.Conv2d(in_channels_list[-1] + len((1, 2, 3, 6)) * fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )

        # FPN 侧边连接与融合卷积 (降维统一到 fpn_dim)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_ch in in_channels_list[:-1]:  # 对应 feat2, feat3, feat4
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(in_ch, fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))

        # FPN 最终特征拼接后的融合瓶颈层
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels_list) * fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )
        
        # 【保留你的特色】：CBAM 模块，放在 FPN 融合后，提取最终的增强特征
        self.cbam = CBAM(fpn_dim)

    def forward(self, features):
        # features 包含: [feat2, feat3, feat4, feat5]
        
        # 1. 对最深层 feat5 应用 PPM，提取全局语境
        feat5 = features[-1]
        p5 = self.ppm_conv(self.ppm(feat5))
        
        # 2. FPN 自顶向下路径 (Top-down pathway)
        # laterals 存储 FPN 各层特征，倒序推导，最终顺序为 p5, p4, p3, p2
        laterals = [p5]
        # 倒序遍历 feat4(idx=2), feat3(idx=1), feat2(idx=0)
        for i in range(len(features) - 2, -1, -1):
            lat = self.lateral_convs[i](features[i])
            # 将上一层的 P 特征上采样到当前 lat 的尺寸然后相加
            prev_p_upsampled = F.interpolate(laterals[-1], size=lat.shape[2:], mode='bilinear', align_corners=True)
            p = lat + prev_p_upsampled
            p = self.fpn_convs[i](p)
            laterals.append(p)
        
        # 3. 将所有 FPN 特征上采样到 p2 的尺寸 (即原图的 1/4 分辨率)
        p2_size = laterals[-1].shape[2:]
        outs = [laterals[-1]]  # 先放入 p2
        for i in range(len(laterals) - 1):  # 加入 p5, p4, p3 的上采样结果
            outs.append(F.interpolate(laterals[i], size=p2_size, mode='bilinear', align_corners=True))
        
        # 4. 拼接所有尺度的特征并融合
        out = self.fpn_bottleneck(torch.cat(outs, 1))
        
        # 5. CBAM 注意力机制增强特征
        out = self.cbam(out)
        
        # 返回最终融合特征 out (尺寸为 H/4, W/4)，以及 p4 (对应 feat4，用于 Aux Loss)
        return out, laterals[1] 


# ==========================================
# 3. 组装完整模型 (替代原有的 Unet 类)
# ==========================================
class Unet(nn.Module):
    def __init__(self, num_classes=21):
        super(Unet, self).__init__()

        # 使用 ResNet50 作为编码器
        self.resnet = resnet50() 
        
        # ResNet50 对应 feat2, feat3, feat4, feat5 的输出通道数
        in_channels_list = [256, 512, 1024, 2048]
        fpn_dim = 512  # UperNet 默认内部投影统一通道数
        
        # 实例化 UperNet 解码器 (纯卷积 + PPM + FPN)
        self.decoder = UperNetDecoder(in_channels_list, fpn_dim=fpn_dim)

        # 主分类头 (语义分割)
        self.final = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
        )
        
        # 高度图分类头
        self.height_head = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(fpn_dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 辅助分类头 (Auxiliary Head)
        # 在 UperNet 中，辅助头通常接在 FPN 的 P4（也就是特征图尺寸为 H/16）后面，提供中层监督
        self.aux_head = nn.Sequential(
            nn.Conv2d(fpn_dim, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, inputs):
        input_size = inputs.size()[2:] # 记录输入尺寸 [H, W]
        
        # 编码器提取五层特征图
        features = self.resnet.forward(inputs)
        
        # UperNet 只需要 stage1~stage4 的特征 (对应你代码里的 feat2, feat3, feat4, feat5)
        fpn_features = features[1:] 

        # 解码过程：输出融合主特征 out(H/4, W/4) 和 辅助特征 p4(H/16, W/16)
        out, p4 = self.decoder(fpn_features)

        # --- 分支 A: 语义分割 ---
        final = self.final(out) 
        # 直接在此处将 H/4 的预测图上采样回原图大小 H, W
        final = F.interpolate(final, size=input_size, mode='bilinear', align_corners=True)

        # --- 分支 B: 高度图预测 ---
        pred_height = self.height_head(out) 
        # 同样上采样回原图大小 H, W
        pred_height = F.interpolate(pred_height, size=input_size, mode='bilinear', align_corners=True)

        # --- 辅助分支 (仅训练时计算) ---
        aux_out = None
        if self.training:
            aux_out = self.aux_head(p4)
            # 【细节对齐】：你的 train.py 中 target_small 使用了 scale_factor=0.25 (即 H/4 尺寸)
            # P4 出来的特征是 H/16，我们把它上采样到 H/4 来匹配你的训练逻辑，这样你的 loss 计算代码一行都不用改！
            aux_out = F.interpolate(aux_out, size=(input_size[0]//4, input_size[1]//4), mode='bilinear', align_corners=True)

        # 训练时返回三个值，推理时只返回 final 分割图
        if self.training:
            return final, aux_out, pred_height
        else:
            return final