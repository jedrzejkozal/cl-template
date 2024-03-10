# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch.nn.functional as F
import torchvision.transforms as transforms
from backbones.MNISTMLP import MNISTMLP
from PIL import Image
from torchvision.datasets import MNIST

from datasets.utils.continual_benchmark import ContinualBenchmark
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path


class TrainMNIST(MNIST):
    """
    Overrides the MNIST dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None, not_aug_transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = not_aug_transform
        super(TrainMNIST, self).__init__(root, train,
                                         transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        original_img = self.not_aug_transform(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target, original_img


class SequentialMNIST(ContinualBenchmark):

    NAME = 'seq-mnist'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_CLASSES = 10
    N_TASKS = 5
    IMG_SIZE = 28

    def get_data_loaders(self):
        train_dataset = TrainMNIST(base_path() + 'MNIST',
                                   train=True, download=True, transform=self.train_transform, not_aug_transform=self.train_transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        self.train_transform, self.NAME)
        else:
            test_dataset = MNIST(base_path() + 'MNIST',
                                 train=False, download=True, transform=self.train_transform)

        train, test = self.store_masked_loaders(train_dataset, test_dataset)
        return train, test

    @staticmethod
    def get_backbone():
        return MNISTMLP(28 * 28, SequentialMNIST.N_TASKS
                        * SequentialMNIST.N_CLASSES_PER_TASK)

    @property
    def train_transform(self):
        transform_list = [transforms.ToTensor()]
        if self.image_size != self.IMG_SIZE:
            transform_list = [transforms.Resize(self.image_size)] + transform_list
        transform = transforms.Compose(transform_list)
        return transform

    def get_transform(self):
        transform = transforms.Compose([transforms.ToPILImage(), self.train_transform])
        return transform

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_batch_size():
        return 64

    @staticmethod
    def get_minibatch_size():
        return SequentialMNIST.get_batch_size()
