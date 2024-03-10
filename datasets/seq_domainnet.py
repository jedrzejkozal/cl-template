import numpy as np
import shutil
import wget
import pathlib
import os

import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from typing import Tuple

from PIL import Image
from utils.conf import base_path_dataset as base_path
from datasets.utils.continual_benchmark import ContinualBenchmark
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.validation import get_train_val
from backbones.resnet import resnet18
from datasets.utils.additional_augmentations import ImageNetPolicy
from datasets.utils.ops import Cutout


def download_dataset(dataset_path: pathlib.Path):
    links = [
        'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt',
    ]
    dataset_path = pathlib.Path(dataset_path)
    os.makedirs(str(dataset_path), exist_ok=True)
    file_downloaded = False

    for url in links:
        filename = url.split('/')[-1]
        filepath = dataset_path / filename
        if filepath.exists():
            print(f'skip download, file {filename} already exists')
        else:
            print(f'downloading file {filename}')
            wget.download(url, out=str(dataset_path))
            file_downloaded = True

    if file_downloaded:
        print()  # add newline after printing download bar

    for zip_filepath in [path for path in dataset_path.iterdir() if path.suffix == '.zip']:
        domain_name = zip_filepath.stem
        domain_path = dataset_path / domain_name
        if not (domain_path.exists() and domain_path.is_dir()):
            print(f'extracting {domain_name} archive')
            shutil.unpack_archive(zip_filepath, extract_dir=dataset_path)
        else:
            print(f'archive for {domain_name} already exists')


def verify_dataset(dataset_path: pathlib.Path):
    dataset_path = pathlib.Path(dataset_path)

    is_ok = True
    for domain_name in ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']:
        domain_folder = dataset_path / f'{domain_name}'
        if not (domain_folder.exists() and domain_folder.is_dir()):
            is_ok = False
            break

        domain_train_filename = dataset_path / f'{domain_name}_train.txt'
        if not (domain_train_filename.exists() and domain_train_filename.is_file()):
            is_ok = False
            break

        domain_test_filename = dataset_path / f'{domain_name}_test.txt'
        if not (domain_test_filename.exists() and domain_test_filename.is_file()):
            is_ok = False
            break

    return is_ok


class DomainNet:
    def __init__(self, root: str, train=True, transform=None, target_transform=None) -> None:
        if not verify_dataset(root):
            download_dataset(root)
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform

        split = 'train' if train else 'test'
        self.annotations = list()

        for domain_idx, domain in enumerate(['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']):
            annotations_filepath = self.root / f'{domain}_{split}.txt'
            annotations = self.read_annotations(annotations_filepath)
            annotations = [(filepath, label + domain_idx * 345) for filepath, label in annotations]
            self.annotations.extend(annotations)

    def read_annotations(self, annotation_filepath: pathlib.Path):
        annotations = list()
        with open(annotation_filepath, 'r') as f:
            for line in f.readlines():
                filepath, label = line.split()
                annotations.append((filepath, int(label)))
        return annotations

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        filepath, target = self.annotations[index]
        filepath = self.root / filepath
        img = Image.open(filepath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class TestDomainNet(DomainNet):
    def __init__(self, root, transform=None, target_transform=None) -> None:
        super().__init__(root, train=False, transform=transform, target_transform=target_transform)


class TrainDomainNet(DomainNet):
    def __init__(self, root, transform=None, not_aug_transform=None, target_transform=None) -> None:
        super().__init__(root, train=True, transform=transform, target_transform=target_transform)
        self.not_aug_transform = not_aug_transform

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        filepath, target = self.annotations[index]
        filepath = self.root / filepath
        img = Image.open(filepath).convert('RGB')

        original_img = img.copy()
        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialDomainNet(ContinualBenchmark):

    NAME = 'seq-domainnet'
    SETTING = 'class-il'
    N_CLASSES = 345 * 6
    N_TASKS = 12
    N_CLASSES_PER_TASK = N_CLASSES // N_TASKS
    IMG_SIZE = 224

    def get_data_loaders(self):
        train_dataset = TrainDomainNet(base_path() + 'DOMAINNET', transform=self.train_transform, not_aug_transform=self.not_aug_transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, self.test_transform, self.NAME)
        else:
            test_dataset = TestDomainNet(base_path() + 'DOMAINNET', transform=self.test_transform)

        self.permute_tasks(train_dataset, test_dataset)
        train, test = self.store_masked_loaders(train_dataset, test_dataset)

        return train, test

    def select_subsets(self, train_dataset: DomainNet, test_dataset: DomainNet, n_classes):
        train_dataset.annotations = list(filter(lambda s: s[1] >= self.i and s[1] < self.i + n_classes, train_dataset.annotations))
        test_dataset.annotations = list(filter(lambda s: s[1] >= self.i and s[1] < self.i + n_classes, test_dataset.annotations))
        return train_dataset, test_dataset

    @property
    def train_transform(self):
        if self.args.additional_augmentations:
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63 / 255),
                ImageNetPolicy(),
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
            transform_list = [transforms.Resize(self.image_size), transforms.RandomCrop(self.image_size, padding=16)] + transform_list
        else:
            transform_list = [transforms.Resize(224), transforms.RandomCrop(224, padding=16)] + transform_list
        transform = transforms.Compose(transform_list)
        return transform

    @property
    def not_aug_transform(self) -> nn.Module:
        transform_list = [transforms.ToTensor()]
        if self.image_size != self.IMG_SIZE:
            transform_list = [transforms.Resize((self.image_size, self.image_size))] + transform_list
        else:
            transform_list = [transforms.Resize((224, 224))] + transform_list
        transform = transforms.Compose(transform_list)
        return transform

    @property
    def test_transform(self) -> nn.Module:
        transform_list = [transforms.ToTensor(), self.get_normalization_transform()]
        if self.image_size != self.IMG_SIZE:
            transform_list = [transforms.Resize(self.image_size), transforms.CenterCrop(self.image_size)] + transform_list
        else:
            transform_list = [transforms.Resize(224), transforms.CenterCrop(224)] + transform_list
        transform = transforms.Compose(transform_list)
        return transform

    def get_transform(self):
        transform = transforms.Compose([transforms.ToPILImage(), self.train_transform])
        return transform

    def get_backbone(self):
        print(self.N_CLASSES)
        exit()
        return resnet18(self.N_CLASSES, width=self.args.model_width)

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
        return DomainNet.get_batch_size()

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
        return scheduler

    def permute_tasks(self, train_dataset, test_dataset):
        """permute tasks order, but make sure, that two classes form different domains don't end up in the same task"""
        train_dataset = self.permute_dataset(train_dataset)
        test_dataset = self.permute_dataset(test_dataset)
        return train_dataset, test_dataset

    def permute_dataset(self, dataset: DomainNet):
        domain_idxs = list(range(6))
        new_domain_idxs = np.random.RandomState(seed=self.args.seed).permutation(domain_idxs)
        new_classes = np.random.RandomState(seed=self.args.seed).permutation(list(range(345)))

        annotations_grouped = list()
        for i in range(0, 2070, 345):
            domain_annotations = [(filepath, label - i) for filepath, label in dataset.annotations if label >= i and label < i + 345]
            annotations_grouped.append(domain_annotations)

        new_annotations = list()
        for i, domain_idx in enumerate(new_domain_idxs):
            annotations = [(filepath, new_classes[label] + i * 345) for filepath, label in annotations_grouped[domain_idx]]
            new_annotations.extend(annotations)

        dataset.annotations = new_annotations
        return dataset


if __name__ == '__main__':
    download_dataset()
