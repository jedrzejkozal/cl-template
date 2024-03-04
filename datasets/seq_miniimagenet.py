# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import os
import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
import collections

from backbones.ResNet18 import resnet18
from PIL import Image
from torchvision.datasets import ImageFolder

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.conf import imagenet_path


class TImageNet(ImageFolder):
    """Workaround to avoid printing the already downloaded messages."""

    def __init__(self, root, split='train', transform=None, target_transform=None) -> None:
        self.root = root = os.path.join(root, split)
        self.split = split
        super().__init__(root, transform=transform, target_transform=target_transform)


class MyImageNet(ImageFolder):
    """
    Overrides the ImageNet dataset to change the getitem function.
    """

    def __init__(self, root, split='train', aug_transform=None, tensor_transform=None, target_transform=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root = os.path.join(root, split)
        self.split = split
        self.tensor_transform = tensor_transform
        super().__init__(root, transform=aug_transform, target_transform=target_transform)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        not_aug_img = self.loader(path)

        not_aug_img = self.tensor_transform(not_aug_img)
        img = self.transform(not_aug_img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, not_aug_img


class SequentialImageNet(ContinualDataset):

    NAME = 'seq-miniimagenet'
    SETTING = 'class-il'
    N_CLASSES = 100
    N_TASKS = 10
    N_CLASSES_PER_TASK = N_CLASSES // N_TASKS
    IMG_SIZE = 224
    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    AUG_TRANSFORM = transforms.Compose([
        transforms.RandomHorizontalFlip(),
    ])

    # def get_examples_number(self):
    #     train_dataset = MyImageNet(imagenet_path(), split='train')
    #     return len(train_dataset.data)

    def get_data_loaders(self):
        test_transform = self.TEST_TRANSFORM

        train_dataset = MyImageNet(
            imagenet_path(), split='train', aug_transform=self.AUG_TRANSFORM, tensor_transform=test_transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = TImageNet(
                imagenet_path(), split='val', transform=test_transform)

        # select classes with highest number of learning samples
        train_labels = [label for _, label in train_dataset.samples]
        counter = collections.Counter(train_labels)
        most_common_classes = set([cls for cls, _ in counter.most_common(100)])
        train_dataset.samples = list(filter(lambda s: s[1] in most_common_classes, train_dataset.samples))
        train_dataset.targets = list(filter(lambda t: t in most_common_classes, train_dataset.targets))
        test_dataset.samples = list(filter(lambda s: s[1] in most_common_classes, test_dataset.samples))
        test_dataset.targets = list(filter(lambda t: t in most_common_classes, test_dataset.targets))

        self.permute_tasks(train_dataset, test_dataset)
        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialImageNet.TEST_TRANSFORM, SequentialImageNet.AUG_TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialImageNet.N_CLASSES_PER_TASK
                        * SequentialImageNet.N_TASKS)
        # return vit_base_patch16_224(pretrained=False)

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
