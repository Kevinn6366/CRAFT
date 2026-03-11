import torch
import torch.nn as nn
import torch.nn.functional as F

class SAGate(nn.Module):
    def __init__(self, in_channels):
        super(SAGate, self).__init__()
        
        # --- 特征分离模块 (Feature Separation, FS) ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # 对应 Eq(2): 生成通道注意力向量 MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels, bias=False),
            nn.Sigmoid()
        )

        # --- 特征聚合模块 (Feature Aggregation, FA) ---
        # 对应 Eq(5) 和 Eq(6): 空间感知门
        self.conv_rgb = nn.Sequential(
            nn.Conv2d(in_channels * 2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_hha = nn.Sequential(
            nn.Conv2d(in_channels * 2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, rgb_in, hha_in):
        """
        rgb_in: 语义主干特征
        hha_in: 高度/几何特征
        """
        b, c, h, w = rgb_in.size()

        # ==========================================
        # 1. 特征分离 (FS) - 通道级清洗
        # ==========================================
        # Eq(1): 全局空间信息融合
        concat_fs = torch.cat([rgb_in, hha_in], dim=1) # [B, 2C, H, W]
        global_ctx = self.global_pool(concat_fs).view(b, -1) # [B, 2C]

        # Eq(2): 学习深度特征的跨模态通道权重
        w_hha = self.mlp(global_ctx).view(b, c, 1, 1) # [B, C, 1, 1]

        # Eq(3): 过滤噪声深度特征
        hha_filtered = hha_in * w_hha

        # Eq(4): 用干净的几何信息重新校准 RGB
        rgb_rec = rgb_in + hha_filtered

        # ==========================================
        # 2. 特征聚合 (FA) - 空间级门控
        # ==========================================
        # Eq(5) & Eq(6): 拼接校准后的特征并生成空间门
        concat_fa = torch.cat([rgb_rec, hha_filtered], dim=1) # [B, 2C, H, W]
        g_rgb = self.conv_rgb(concat_fa) # [B, 1, H, W]
        g_hha = self.conv_hha(concat_fa) # [B, 1, H, W]

        # Eq(8): 最终加权融合 (注意：这里用了论文最严谨的表达，用门控去乘原始特征)
        out_fused = rgb_in * g_rgb + hha_in * g_hha

        return out_fused