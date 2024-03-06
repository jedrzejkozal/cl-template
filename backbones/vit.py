import torch.nn as nn

from torchvision.models import vit_b_16, vit_h_14, vit_l_16, vit_l_32
from torchvision.models import ViT_B_16_Weights, ViT_H_14_Weights, ViT_L_16_Weights, ViT_L_32_Weights


def vit_base_patch16(n_classes: int, width: int = 1, pretrained: bool = False):
    if width != 1:
        raise ValueError('Width != 1 cannot be used with transformers')
    if pretrained:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(num_classes=1000, weights=weights)
        model.classifier[1] = nn.Linear(model.last_channel, n_classes)
    else:
        model = vit_b_16(num_classes=n_classes, weights=None)

    return model


def vit_huge_patch14(n_classes: int, width: int = 1, pretrained: bool = False):
    if width != 1:
        raise ValueError('Width != 1 cannot be used with transformers')
    if pretrained:
        weights = ViT_H_14_Weights.DEFAULT
        model = vit_h_14(num_classes=1000, weights=weights)
        model.classifier[1] = nn.Linear(model.last_channel, n_classes)
    else:
        model = vit_h_14(num_classes=n_classes, weights=None)

    return model


def vit_large_patch16(n_classes: int, width: int = 1, pretrained: bool = False):
    if width != 1:
        raise ValueError('Width != 1 cannot be used with transformers')
    if pretrained:
        weights = ViT_L_16_Weights.DEFAULT
        model = vit_l_16(num_classes=1000, weights=weights)
        model.classifier[1] = nn.Linear(model.last_channel, n_classes)
    else:
        model = vit_l_16(num_classes=n_classes, weights=None)

    return model


def vit_large_patch32(n_classes: int, width: int = 1, pretrained: bool = False):
    if width != 1:
        raise ValueError('Width != 1 cannot be used with transformers')
    if pretrained:
        weights = ViT_L_32_Weights.DEFAULT
        model = vit_l_32(num_classes=1000, weights=weights)
        model.classifier[1] = nn.Linear(model.last_channel, n_classes)
    else:
        model = vit_l_32(num_classes=n_classes, weights=None)

    return model


def get_all_backbones():
    return ['vit_base_patch16', 'vit_huge_patch14', 'vit_large_patch16', 'vit_large_patch32']
