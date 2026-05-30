import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet_backbone import resnet50
from model.CRAFT import CRAFT

class PPM(nn.Module):
    """Pyramid Pooling Module."""
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


class UperNetDecoder(nn.Module):
    """FPN-based decoder with topology-gated skip connections."""
    def __init__(self, in_channels_list, fpn_dim=512):
        super(UperNetDecoder, self).__init__()

        self.ppm = PPM(in_channels_list[-1], fpn_dim)
        self.ppm_conv = nn.Sequential(
            nn.Conv2d(in_channels_list[-1] + len((1, 2, 3, 6)) * fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        self.topo_gates = nn.ModuleList()

        for in_ch in in_channels_list[:-1]:
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
            
            self.topo_gates.append(nn.Sequential(
                nn.Conv2d(fpn_dim, 1, kernel_size=1, bias=False),
                nn.Sigmoid()
            ))

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels_list) * fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        feat5 = features[-1]
        p5 = self.ppm_conv(self.ppm(feat5))
        
        laterals = [p5]

        for i in range(len(features) - 2, -1, -1):
            lat = self.lateral_convs[i](features[i])

            prev_p_upsampled = F.interpolate(laterals[-1], size=lat.shape[2:], mode='bilinear', align_corners=True)

            topo_mask = self.topo_gates[i](prev_p_upsampled)
            lat_filtered = lat * topo_mask
            p = lat_filtered + prev_p_upsampled

            p = self.fpn_convs[i](p)
            laterals.append(p)
        
        p2_size = laterals[-1].shape[2:]
        outs = [laterals[-1]]
        for i in range(len(laterals) - 1):
            outs.append(F.interpolate(laterals[i], size=p2_size, mode='bilinear', align_corners=True))
        
        out = self.fpn_bottleneck(torch.cat(outs, 1))
        
        return out, laterals[1]

class Unet(nn.Module):
    def __init__(self, num_classes=21):
        super(Unet, self).__init__()

        self.resnet = resnet50()
        in_channels_list = [256, 512, 1024, 2048]
        fpn_dim = 512

        self.decoder = UperNetDecoder(in_channels_list, fpn_dim=fpn_dim)

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

        self.craft = CRAFT(in_channels=fpn_dim)

        self.final = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
        )

        self.height_head = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(fpn_dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

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

        out, p4 = self.decoder(fpn_features)

        feat_rgb = self.rgb_proj(out)
        feat_height = self.height_proj(out)

        fused_rgb = self.craft(feat_rgb, feat_height)

        final = self.final(fused_rgb)
        pred_height = self.height_head(feat_height)

        final = F.interpolate(final, size=input_size, mode='bilinear', align_corners=True)
        pred_height = F.interpolate(pred_height, size=input_size, mode='bilinear', align_corners=True)

        aux_out = None
        if self.training:
            aux_out = self.aux_head(p4)
            aux_out = F.interpolate(aux_out, size=(input_size[0] // 4, input_size[1] // 4),
                                    mode='bilinear', align_corners=True)

        if self.training:
            return final, aux_out, pred_height
        else:
            return final, pred_height