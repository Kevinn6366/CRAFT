import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入你原有的 ResNet50 Backbone
from model.resnet_backbone import resnet50
# 导入跨模态门控融合模块 (请确保 sa_gate.py 文件存在)
from model.sa_gate import SAGate 

# 1. 定义 UperNet 的核心组件：金字塔池化模块 (PPM)
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


# 2. 定义 UperNet 解码器
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

        # FPN 侧边连接与融合卷积
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        # 【核心新增】：拓扑门控生成器 (Topology Gates)
        # 用来为每一层 Skip Connection 生成 0~1 的空间掩码
        self.topo_gates = nn.ModuleList()
        
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
            
            # 每一层对应一个门控：用 1x1 卷积将 FPN 特征压缩为 1 个通道，再经过 Sigmoid
            self.topo_gates.append(nn.Sequential(
                nn.Conv2d(fpn_dim, 1, kernel_size=1, bias=False),
                nn.Sigmoid()
            ))

        # FPN 最终特征拼接后的融合瓶颈层
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels_list) * fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        feat5 = features[-1]
        p5 = self.ppm_conv(self.ppm(feat5))
        
        laterals = [p5]
        
        # 从深层到浅层遍历 (FPN 的 Top-Down 路径)
        for i in range(len(features) - 2, -1, -1):
            # 1. 获取低层特征 (包含丰富细节，但也含糊了大量的背景和粘连噪声)
            lat = self.lateral_convs[i](features[i]) 
            
            # 2. 将上一层的高层特征上采样 (包含了高级语义和粗略的拓扑信息)
            prev_p_upsampled = F.interpolate(laterals[-1], size=lat.shape[2:], mode='bilinear', align_corners=True)
            
            # ==========================================
            # 【核心修改：拓扑门控 Skip Connection】
            # ==========================================
            # a. 用高层特征生成当前的拓扑 Mask (尺寸 [B, 1, H, W], 值在 0~1 之间)
            topo_mask = self.topo_gates[i](prev_p_upsampled)
            
            # b. 用 Mask 对低层特征进行“提纯” ( Element-wise Multiply )
            # 也就是你说的：拿高度图/拓扑图指导它，只让物体边缘通过，屏蔽背景纹理
            lat_filtered = lat * topo_mask
            
            # c. 完美的融合 (Add)：被提纯的高清细节 + 高层全局语义 (没有任何信息被暴力抛弃)
            p = lat_filtered + prev_p_upsampled
            # ==========================================
            
            p = self.fpn_convs[i](p)
            laterals.append(p)
        
        p2_size = laterals[-1].shape[2:]
        outs = [laterals[-1]]
        for i in range(len(laterals) - 1):
            outs.append(F.interpolate(laterals[i], size=p2_size, mode='bilinear', align_corners=True))
        
        out = self.fpn_bottleneck(torch.cat(outs, 1))
        
        return out, laterals[1]

# 3. 组装完整模型
class Unet(nn.Module):
    def __init__(self, num_classes=21):
        super(Unet, self).__init__()

        self.resnet = resnet50() 
        in_channels_list = [256, 512, 1024, 2048]
        fpn_dim = 512  
        
        self.decoder = UperNetDecoder(in_channels_list, fpn_dim=fpn_dim)

        # 【核心新增】：特征解耦投影层
        self.rgb_proj = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )
        self.height_proj = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )
        
        # 跨模态注意力融合门 (SA-Gate)
        self.sa_gate = SAGate(in_channels=fpn_dim)

        # 主分类头
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

        # 【修复关键点 1】：解开辅助分类头的定义，输入通道是 fpn_dim (512)
        self.aux_head = nn.Sequential(
            nn.Conv2d(fpn_dim, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, inputs):
        input_size = inputs.size()[2:] 
        
        features = self.resnet.forward(inputs)
        fpn_features = features[1:] 

        # 解码
        out, p4 = self.decoder(fpn_features)

        # 特征解耦
        feat_rgb = self.rgb_proj(out)
        feat_height = self.height_proj(out)

        # SA-Gate 融合
        fused_rgb = self.sa_gate(feat_rgb, feat_height)

        # 主任务预测
        final = self.final(fused_rgb) 
        pred_height = self.height_head(feat_height) 

        # 上采样回原图大小
        final = F.interpolate(final, size=input_size, mode='bilinear', align_corners=True)
        pred_height = F.interpolate(pred_height, size=input_size, mode='bilinear', align_corners=True)

        # 【修复关键点 2】：训练模式下计算 aux_out，防止返回 None 导致报错
        aux_out = None
        if self.training:
            # 计算辅助头输出 (使用 FPN 的中层特征 p4)
            aux_out = self.aux_head(p4)
            # 统一上采样到 input_size/4 以对齐你的 train.py 逻辑
            aux_out = F.interpolate(aux_out, size=(input_size[0]//4, input_size[1]//4), 
                                  mode='bilinear', align_corners=True)

        if self.training:
            return final, aux_out, pred_height
        else:
            return final