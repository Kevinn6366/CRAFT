import torch
import torch.nn as nn
import torch.nn.functional as F

from models._initialization import resnet50
from models.craft import CRAFT

from models.upernet import UperNetDecoder

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
