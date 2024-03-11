# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
import collections

from backbones.resnet import resnet50
from PIL import Image
from torchvision.datasets import ImageFolder

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_benchmark import ContinualBenchmark
from datasets.utils.validation import get_train_val
from utils.conf import imagenet_path

from .seq_imagenet import TrainImageNet, TestImageNet


class SequentialMiniImageNet(ContinualBenchmark):

    NAME = 'seq-miniimagenet'
    SETTING = 'class-il'
    N_CLASSES = 100
    N_TASKS = 10
    N_CLASSES_PER_TASK = N_CLASSES // N_TASKS
    IMG_SIZE = 224

    def get_data_loaders(self):
        train_dataset = TrainImageNet(
            imagenet_path(), split='train', aug_transform=self.train_transform, tensor_transform=self.test_transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, self.test_transform, self.NAME)
        else:
            test_dataset = TestImageNet(imagenet_path(), split='val', transform=self.test_transform)

        # select classes with highest number of learning samples
        train_labels = [label for _, label in train_dataset.samples]
        counter = collections.Counter(train_labels)
        most_common_classes = set([cls for cls, _ in counter.most_common(100)])
        train_dataset.samples = list(filter(lambda s: s[1] in most_common_classes, train_dataset.samples))
        train_dataset.targets = list(filter(lambda t: t in most_common_classes, train_dataset.targets))
        test_dataset.samples = list(filter(lambda s: s[1] in most_common_classes, test_dataset.samples))
        test_dataset.targets = list(filter(lambda t: t in most_common_classes, test_dataset.targets))

        self.permute_tasks(train_dataset, test_dataset)
        train, test = self.store_masked_loaders(train_dataset, test_dataset)

        return train, test

    @property
    def train_transform(self):
        transform_list = [transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          self.get_normalization_transform()]
        if self.image_size != self.IMG_SIZE:
            transform_list = [transforms.Resize(self.image_size), transforms.RandomCrop(self.image_size, padding=4)] + transform_list
        else:
            transform_list = [transforms.RandomCrop(224, padding=4)] + transform_list
        transform = transforms.Compose(transform_list)
        return transform

    @property
    def not_aug_transform(self) -> nn.Module:
        transform_list = [transforms.ToTensor()]
        if self.image_size != self.IMG_SIZE:
            transform_list = [transforms.Resize(self.image_size)] + transform_list
        transform = transforms.Compose(transform_list)
        return transform

    @property
    def test_transform(self):
        transform_list = [transforms.ToTensor(), self.get_normalization_transform()]
        if self.image_size != self.IMG_SIZE:
            transform_list = [transforms.Resize(self.image_size)] + transform_list
        transform = transforms.Compose(transform_list)
        return transform

    def get_transform(self):
        transform = transforms.Compose([transforms.ToPILImage(), self.train_transform])
        return transform

    def get_backbone(self):
        return resnet50(SequentialMiniImageNet.N_CLASSES_PER_TASK * SequentialMiniImageNet.N_TASKS, width=self.args.model_width)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialMiniImageNet.get_batch_size()

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(
        ), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            model.opt, [35, 45], gamma=0.1, verbose=False)
        return scheduler

    def select_subsets(self, train_dataset, test_dataset, n_classes):
        train_dataset.samples = list(filter(lambda s: s[1] >= self.i and s[1] < self.i + n_classes, train_dataset.samples))
        test_dataset.samples = list(filter(lambda s: s[1] >= self.i and s[1] < self.i + n_classes, test_dataset.samples))

        return train_dataset, test_dataset
