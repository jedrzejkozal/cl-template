# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from typing import Tuple

import numpy as np
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, Dataset


class ContinualBenchmark:
    """
    Continual learning evaluation setting.
    """
    NAME: str
    SETTING: str
    N_CLASSES: int
    N_CLASSES_PER_TASK: int
    N_TASKS: int
    IMG_SIZE: int

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args
        self.image_size = args.img_size

        if not all((self.NAME, self.SETTING, self.N_CLASSES, self.N_CLASSES_PER_TASK, self.N_TASKS)):
            raise NotImplementedError('The dataset must be initialized with all the required fields.')
        if not self.args.half_classes_in_first_task and self.N_CLASSES // self.N_TASKS < 2:
            raise ValueError(f"Each task should have at least 2 classes, got N_CLASSES={self.N_CLASSES}, N_TASKS={self.N_TASKS}")

        if args.n_tasks != None:
            type(self).N_TASKS = args.n_tasks
            type(self).N_CLASSES_PER_TASK = type(self).N_CLASSES // type(self).N_TASKS
        else:
            args.n_tasks = type(self).N_TASKS
        if args.img_size is None:
            args.img_size = self.IMG_SIZE
            self.image_size = self.IMG_SIZE

        if self.args.half_classes_in_first_task:
            type(self).N_CLASSES_PER_TASK = self.N_CLASSES // 2 // (args.n_tasks-1)
            assert self.N_CLASSES_PER_TASK >= 2

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        raise NotImplementedError

    def get_backbone(self) -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        raise NotImplementedError

    @property
    def train_transform(self) -> nn.Module:
        """
        Returns the transform to be used for to the current dataset.
        """
        raise NotImplementedError

    @property
    def not_aug_transform(self) -> nn.Module:
        """
        Returns the transform to be used for images that will be stored in the buffer
        """
        raise NotImplementedError

    @property
    def test_transform(self) -> nn.Module:
        """
        Returns the transform to be used for test set images
        """
        raise NotImplementedError

    def get_transform(self) -> nn.Module:
        """
        Returns the transform for the rehersal buffer
        """
        raise NotImplementedError

    @staticmethod
    def get_loss() -> nn.Module:
        """
        Returns the loss to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_normalization_transform() -> nn.Module:
        """
        Returns the transform used for normalizing the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_denormalization_transform() -> nn.Module:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Returns the scheduler to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_epochs():
        raise NotImplementedError

    @staticmethod
    def get_batch_size():
        raise NotImplementedError

    @staticmethod
    def get_minibatch_size():
        raise NotImplementedError

    def permute_tasks(self, train_dataset, test_dataset) -> None:
        """
        Changes the order of classes in the dataset, so with different seed data in each task is different
        """
        train_labels = train_dataset.targets
        classes = np.unique(train_labels)
        new_classes = np.random.RandomState(seed=self.args.seed).permutation(classes)

        train_dataset.targets = [new_classes[c] for c in train_dataset.targets]
        test_dataset.targets = [new_classes[c] for c in test_dataset.targets]

    def store_masked_loaders(self, train_dataset: Dataset, test_dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
        """
        Divides the dataset into tasks.
        :param train_dataset: train dataset
        :param test_dataset: test dataset
        :param setting: continual learning setting
        :return: train and test loaders
        """
        if self.args.half_classes_in_first_task and self.i == 0:
            n_classes = self.N_CLASSES // 2
        else:
            n_classes = self.N_CLASSES_PER_TASK

        self.select_subsets(train_dataset, test_dataset, n_classes)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        self.test_loaders.append(test_loader)
        self.train_loader = train_loader

        self.i += n_classes
        return train_loader, test_loader

    def select_subsets(self, train_dataset, test_dataset, n_classes):
        """selecting data for each task
            can be overriden in the in the classes that inherit from ContinualBenchmark
        """
        train_mask = np.logical_and(np.array(train_dataset.targets) >= self.i,
                                    np.array(train_dataset.targets) < self.i + n_classes)
        test_mask = np.logical_and(np.array(test_dataset.targets) >= self.i,
                                   np.array(test_dataset.targets) < self.i + n_classes)

        train_dataset.data = train_dataset.data[train_mask]
        test_dataset.data = test_dataset.data[test_mask]

        train_dataset.targets = np.array(train_dataset.targets)[train_mask]
        test_dataset.targets = np.array(test_dataset.targets)[test_mask]

        return train_dataset, test_dataset


def get_previous_train_loader(train_dataset: Dataset, batch_size: int,
                              setting: ContinualBenchmark) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >=
                                setting.i - setting.N_CLASSES_PER_TASK, np.array(train_dataset.targets)
                                < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
