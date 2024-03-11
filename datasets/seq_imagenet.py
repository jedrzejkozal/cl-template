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
from backbones.resnet import resnet50
from PIL import Image
from torchvision.datasets import ImageFolder

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_benchmark import ContinualBenchmark
from datasets.utils.validation import get_train_val
from utils.conf import imagenet_path


class TestImageNet(ImageFolder):
    """Workaround to avoid printing the already downloaded messages."""

    def __init__(self, root, split='train', transform=None, target_transform=None) -> None:
        self.root = root = os.path.join(root, split)
        self.split = split
        super().__init__(root, transform=transform, target_transform=target_transform)


class TrainImageNet(ImageFolder):
    """
    Overrides the ImageNet dataset to change the getitem function.
    """

    def __init__(self, root, split='train', transform=None, not_aug_transform=None, target_transform=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root = os.path.join(root, split)
        self.split = split
        self.not_aug_transform = not_aug_transform
        super().__init__(root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        not_aug_img = self.loader(path)

        not_aug_img = self.not_aug_transform(not_aug_img)
        img = self.transform(not_aug_img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, not_aug_img


class SequentialImageNet(ContinualBenchmark):

    NAME = 'seq-imagenet'
    SETTING = 'class-il'
    N_CLASSES = 1000
    N_TASKS = 100
    N_CLASSES_PER_TASK = N_CLASSES // N_TASKS
    IMG_SIZE = 224

    def get_data_loaders(self):
        train_dataset = TrainImageNet(
            imagenet_path(), split='train', transform=self.train_transform, not_aug_transform=self.not_aug_transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, self.test_transform, self.NAME)
        else:
            test_dataset = TestImageNet(imagenet_path(), split='val', transform=self.test_transform)

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
        return resnet50(SequentialImageNet.N_CLASSES_PER_TASK * SequentialImageNet.N_TASKS, width=self.args.model_width)

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
        return SequentialImageNet.get_batch_size()

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
