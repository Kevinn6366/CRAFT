"""DINOv3 + project UPerNet + full CRAFT, with checkpoint-compatible keys."""
import torch
from torch.nn import functional as F
from models.network import CRAFTSegmentation
from models.dinov3 import SPECS
from utils.losses import Focal_Loss, Dice_loss, Height_MSE_Loss


def build_model(num_classes, variant='G3', seed=11, backbone='dinov3_b'):
    if variant != 'G3' or backbone not in SPECS:
        raise ValueError('This release supports full CRAFT with DINOv3-B/L')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = CRAFTSegmentation(num_classes, backbone)
        model.variant = variant
    return model


def loss_terms(model, outputs, target, heights, height_weight=2.0):
    """Semantic segmentation and height regression losses."""
    logits, aux, pred_height = outputs
    classes = logits.shape[1]
    main = Focal_Loss(logits, target, None, num_classes=classes) + Dice_loss(logits, target)
    small = F.interpolate(target.unsqueeze(1).float(), size=aux.shape[-2:],
                          mode='nearest').squeeze(1).long()
    aux_loss = Focal_Loss(aux, small, None, num_classes=classes) + Dice_loss(aux, small)
    maximum = heights.max()
    if maximum > 1:
        heights = heights / maximum
    height_loss = Height_MSE_Loss(pred_height, heights)
    total = main + 0.4 * aux_loss + height_weight * height_loss
    return {'total': total, 'main': main, 'aux': aux_loss, 'height': height_loss}
