# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from backbones.resnet import resnet18
from PIL import Image
from torchvision.datasets import CIFAR100

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_benchmark import ContinualBenchmark
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path
from datasets.utils.additional_augmentations import CIFAR10Policy
from datasets.utils.ops import Cutout


class TestCIFAR100(CIFAR100):
    """Workaround to avoid printing the already downloaded messages."""

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TestCIFAR100, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())


class TrainCIFAR100(CIFAR100):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None, not_aug_transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = not_aug_transform
        self.root = root
        super(TrainCIFAR100, self).__init__(root, train, transform, target_transform, not self._check_integrity())

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


class SequentialCIFAR100(ContinualBenchmark):

    NAME = 'seq-cifar100'
    SETTING = 'class-il'
    N_CLASSES = 100
    N_TASKS = 20
    N_CLASSES_PER_TASK = N_CLASSES // N_TASKS
    IMG_SIZE = 32

    def get_data_loaders(self):
        not_aug_transform = transforms.Compose([transforms.ToTensor()])
        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])
        if self.image_size != self.IMG_SIZE:
            not_aug_transform.transforms.insert(0, transforms.Resize(self.image_size))
            test_transform.transforms.insert(0, transforms.Resize(self.image_size))

        train_dataset = TrainCIFAR100(base_path() + 'CIFAR100', train=True,
                                      download=True, transform=self.train_transform, not_aug_transform=not_aug_transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = TestCIFAR100(base_path() + 'CIFAR100', train=False,
                                        download=True, transform=test_transform)

        self.permute_tasks(train_dataset, test_dataset)
        train, test = self.store_masked_loaders(train_dataset, test_dataset)

        return train, test

    @property
    def train_transform(self):
        if self.args.additional_augmentations:
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63 / 255),
                CIFAR10Policy(),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                self.get_normalization_transform()
            ]
        else:
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.get_normalization_transform()
            ]
        if self.image_size != self.IMG_SIZE:
            transform_list = [transforms.Resize(self.image_size), transforms.RandomCrop(self.image_size, padding=4)] + transform_list
        else:
            transform_list = [transforms.RandomCrop(32, padding=4)] + transform_list
        transform = transforms.Compose(transform_list)
        return transform

    def get_transform(self):
        transform = transforms.Compose([transforms.ToPILImage(), self.train_transform])
        return transform

    def get_backbone(self):
        return resnet18(SequentialCIFAR100.N_CLASSES_PER_TASK * SequentialCIFAR100.N_TASKS, width=self.args.model_width)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR100.get_batch_size()

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
        return scheduler
