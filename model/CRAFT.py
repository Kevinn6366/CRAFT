import torch
import torch.nn as nn
import torch.nn.functional as F

class CRAFT(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(CRAFT, self).__init__()
        self.num_heads = num_heads
        self.dim = in_channels

        self.scale = (in_channels // num_heads) ** -0.5

        self.spatial_rect = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.q_rgb = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.k_rgb = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.v_rgb = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        self.q_hha = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.k_hha = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.v_hha = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        self.proj_rgb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.proj_hha = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.gate_rgb = nn.Sequential(
            nn.Conv2d(in_channels * 2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.gate_hha = nn.Sequential(
            nn.Conv2d(in_channels * 2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, rgb_in, hha_in):
        B, C, H, W = rgb_in.shape
        N = H * W

        concat_feat = torch.cat([rgb_in, hha_in], dim=1)
        spatial_masks = self.spatial_rect(concat_feat)
        mask_rgb = spatial_masks[:, 0:1, :, :]
        mask_hha = spatial_masks[:, 1:2, :, :]

        clean_rgb = rgb_in * mask_hha
        clean_height = hha_in * mask_rgb

        head_dim = C // self.num_heads

        q_r = self.q_rgb(clean_rgb).view(B, self.num_heads, head_dim, N).transpose(2, 3) * self.scale
        k_r = self.k_rgb(clean_rgb).view(B, self.num_heads, head_dim, N).transpose(2, 3)
        v_r = self.v_rgb(clean_rgb).view(B, self.num_heads, head_dim, N).transpose(2, 3)

        q_h = self.q_hha(clean_height).view(B, self.num_heads, head_dim, N).transpose(2, 3) * self.scale
        k_h = self.k_hha(clean_height).view(B, self.num_heads, head_dim, N).transpose(2, 3)
        v_h = self.v_hha(clean_height).view(B, self.num_heads, head_dim, N).transpose(2, 3)

        k_h_soft = F.softmax(k_h, dim=2)
        context_hha = torch.matmul(k_h_soft.transpose(-2, -1), v_h)
        out_rgb = torch.matmul(q_r, context_hha)

        k_r_soft = F.softmax(k_r, dim=2)
        context_rgb = torch.matmul(k_r_soft.transpose(-2, -1), v_r)
        out_hha = torch.matmul(q_h, context_rgb)

        out_rgb = out_rgb.transpose(2, 3).contiguous().view(B, C, H, W)
        out_hha = out_hha.transpose(2, 3).contiguous().view(B, C, H, W)

        fused_rgb = clean_rgb + self.proj_rgb(out_rgb)
        fused_height = clean_height + self.proj_hha(out_hha)

        concat_fused = torch.cat([fused_rgb, fused_height], dim=1)

        g_rgb = self.gate_rgb(concat_fused)
        g_hha = self.gate_hha(concat_fused)

        out_final = fused_rgb * g_rgb + fused_height * g_hha

        return out_final
