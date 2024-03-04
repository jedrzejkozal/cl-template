# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import importlib

from .mammoth_backbone import *


def get_all_backbone_modules():
    return [model.split('.')[0] for model in os.listdir('backbones')
            if not model.find('__') > -1 and 'py' in model and not 'PNN' in model and not 'mammoth_backbone' in model]


names = {}
for backbone in get_all_backbone_modules():
    backbone_module = importlib.import_module('backbones.' + backbone)
    module_backbones = backbone_module.get_all_backbones()
    for backbone_name in module_backbones:
        names[backbone_name] = getattr(backbone_module, backbone_name)


def get_all_backbones():
    return list(names.keys())


def get_backbone(backbone_name: str, n_classes: int, args):
    return names[backbone_name](n_classes, args.model_width, args.pretrained)
