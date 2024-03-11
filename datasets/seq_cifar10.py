# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbones.resnet import resnet18
from PIL import Image
from torchvision.datasets import CIFAR10

from datasets.seq_tinyimagenet import base_path
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_benchmark import ContinualBenchmark
from datasets.utils.validation import get_train_val


class TestCIFAR10(CIFAR10):
    """Workaround to avoid printing the already downloaded messages."""

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TestCIFAR10, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())


class TrainCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None, not_aug_transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = not_aug_transform
        self.root = root
        super(TrainCIFAR10, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCIFAR10(ContinualBenchmark):

    NAME = 'seq-cifar10'
    SETTING = 'class-il'
    N_CLASSES = 10
    N_TASKS = 5
    N_CLASSES_PER_TASK = N_CLASSES // N_TASKS
    IMG_SIZE = 32

    def get_data_loaders(self):
        train_dataset = TrainCIFAR10(base_path() + 'CIFAR10', train=True,
                                     download=True, transform=self.train_transform, not_aug_transform=self.not_aug_transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, self.test_transform, self.NAME)
        else:
            test_dataset = TestCIFAR10(base_path() + 'CIFAR10', train=False,
                                       download=True, transform=self.test_transform)

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
            transform_list = [transforms.RandomCrop(32, padding=4)] + transform_list
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
        return resnet18(SequentialCIFAR10.N_CLASSES_PER_TASK * SequentialCIFAR10.N_TASKS, width=self.args.model_width)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR10.get_batch_size()
