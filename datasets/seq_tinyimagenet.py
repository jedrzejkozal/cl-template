# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbones.resnet import resnet18
from PIL import Image
from torch.utils.data import Dataset

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_benchmark import ContinualBenchmark
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path
from torchvision.models import mobilenet_v2


class TestTinyImagenet(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """

    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from onedrivedownloader import download

                print('Downloading dataset')
                ln = "https://unimore365-my.sharepoint.com/:u:/g/personal/263133_unimore_it/EVKugslStrtNpyLGbgrhjaABqRHcE3PB_r2OEaV7Jy94oQ?e=9K29aD"
                download(ln, filename=os.path.join(root, 'tiny-imagenet-processed.zip'), unzip=True, unzip_path=root, clean=True)

        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num + 1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num + 1))))
        self.targets = np.concatenate(np.array(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class TrainTinyImagenet(TestTinyImagenet):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """

    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None, not_aug_transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        self.not_aug_transform = not_aug_transform
        super(TrainTinyImagenet, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialTinyImagenet(ContinualBenchmark):

    NAME = 'seq-tinyimg'
    SETTING = 'class-il'
    N_CLASSES = 200
    N_TASKS = 20
    N_CLASSES_PER_TASK = N_CLASSES // N_TASKS
    IMG_SIZE = 64

    def get_data_loaders(self):
        train_dataset = TrainTinyImagenet(base_path() + 'TINYIMG',
                                          train=True, download=True, transform=self.train_transform, not_aug_transform=self.not_aug_transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, self.test_transform, self.NAME)
        else:
            test_dataset = TestTinyImagenet(base_path() + 'TINYIMG',
                                            train=False, download=True, transform=self.test_transform)

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
            transform_list = [transforms.RandomCrop(64, padding=4)] + transform_list
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

    def get_backbone(self):
        return resnet18(SequentialTinyImagenet.N_CLASSES_PER_TASK * SequentialTinyImagenet.N_TASKS, width=self.args.model_width)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose([transforms.ToPILImage(), self.train_transform])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4802, 0.4480, 0.3975),
                                (0.2770, 0.2691, 0.2821))
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
        return SequentialTinyImagenet.get_batch_size()
