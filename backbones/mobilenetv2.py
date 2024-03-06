import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def mobilenetv2(n_classes: int, width: int = 1, pretrained: bool = False):
    if pretrained:
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = mobilenet_v2(num_classes=1000, weights=weights, width_mult=width)
        model.classifier[1] = nn.Linear(model.last_channel, n_classes)
    else:
        model = mobilenet_v2(num_classes=n_classes, weights=None, width_mult=width)

    return model


def get_all_backbones():
    return ['mobilenetv2']
