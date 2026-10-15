"""Pinned DINOv3 ViT-B/16 features, with a learned four-scale adapter."""
import torch
from torch import nn
from torch.nn import functional as F
from utils.common import file_hash

MODEL = 'vit_base_patch16_dinov3.lvd1689m'
REVISION = 'c6a5fb7d12bbd3cf3b0079253141c3332aaed7da'
SHA256 = '1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332'

SPECS = {
    'dinov3_b': dict(model=MODEL, revision=REVISION, sha256=SHA256, dim=768, indices=[2,5,8,11]),
    'dinov3_l': dict(model='vit_large_patch16_dinov3.lvd1689m', revision='30c1109559f65dea34316b0d4842d35c5771fe11', sha256='45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941', dim=1024, indices=[5,11,17,23]),
}

class DinoFeatures(nn.Module):
    def __init__(self, backbone='dinov3_b'):
        super().__init__()
        import timm
        self.spec = SPECS[backbone]
        self.encoder = timm.create_model(self.spec['model'], pretrained=False, num_classes=0,
                                         dynamic_img_size=True)
        if len(self.encoder.blocks) != self.spec['indices'][-1] + 1 or self.encoder.num_features != self.spec['dim']:
            raise ValueError('DINO architecture differs from pinned layer/dimension specification')
        self.encoder.set_grad_checkpointing(True)
        self.adapters = nn.ModuleList([nn.Conv2d(self.spec['dim'], 256, 1) for _ in range(4)])
        self.channels = [256] * 4
        self.register_buffer('mean', torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def forward(self, x):
        h,w = x.shape[-2:]
        features = self.encoder.forward_intermediates((x-self.mean)/self.std,
            indices=self.spec['indices'], norm=True, output_fmt='NCHW', intermediates_only=True)
        pyramid = [F.interpolate(proj(f), size=(h//s,w//s), mode='bilinear',
                                  align_corners=False)
                   for proj,f,s in zip(self.adapters,features,[4,8,16,32])]
        return [pyramid[0], *pyramid]


def replace(model, backbone='dinov3_b'):
    from models.upernet import UperNetDecoder
    model.resnet = DinoFeatures(backbone)
    model.decoder = UperNetDecoder([256]*4, fpn_dim=512)
    model.backbone_name = backbone
    return model


def load(model, path):
    from safetensors.torch import load_file
    spec = SPECS[model.backbone_name]
    actual = file_hash(path)
    if actual != spec['sha256']:
        raise ValueError('DINOv3 pretrained SHA256 mismatch')
    state = load_file(str(path), device='cpu')
    model.resnet.encoder.load_state_dict(state, strict=True)
    return {'path':str(path),'sha256':actual,'type':spec['model'],'revision':spec['revision'],
            'pretraining':'Meta DINOv3 LVD-1689M, official weights distributed by timm',
            'strict':True,'loaded_keys':sorted(state),'randomly_initialized':'pyramid adapter, UPerNet, CRAFT and task heads',
            'input':'RGB [0,1] normalized by ImageNet mean/std inside backbone wrapper'}
