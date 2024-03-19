# based on PyCIL implementation: https://github.com/G-U-N/PyCIL

import torch
import torch.nn as nn
import copy
import collections
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader
from models.utils.continual_model import ContinualModel

from utils.args import add_management_args, add_experiment_args, ArgumentParser


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Self-Sustaining Representation Expansion, https://arxiv.org/pdf/2203.06359.pdf')
    add_management_args(parser)
    add_experiment_args(parser)

    parser.add_argument('--temp', default=0.1, type=float, help='softmax temperature')
    parser.add_argument('--lambda_fkd', default=10.0, type=float, help='distilation loss weight parameter')
    parser.add_argument('--lambda_proto', default=10, type=float, help='prototype loss weight parameter')
    parser.add_argument('--threshold', default=0.8, type=float, help='threshold for prototype selection')

    return parser


class SSRE(ContinualModel):
    NAME = 'ssre'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self._cur_task = 0
        # self.net = IncrementalNet(args)
        assert args.backbone == 'resnet18_rep'
        self._protos = []
        self._known_classes = 0
        self._total_classes = 0

    def begin_task(self, dataset):
        if self.args.half_classes_in_first_task and self._cur_task == 0:
            task_size = dataset.N_CLASSES // 2
        else:
            task_size = dataset.N_CLASSES_PER_TASK
        self._total_classes = self._known_classes + task_size

        if self._cur_task > 0:
            self.net.eval()

        self._network_expansion()

    def get_scheduler(self):
        self.opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.args.lr, weight_decay=self.args.optim_wd)
        return torch.optim.lr_scheduler.StepLR(self.opt, step_size=45, gamma=0.1)

    def observe(self, inputs: Tensor, labels: Tensor, not_aug_inputs: Tensor) -> float:
        self.opt.zero_grad()
        loss_clf, loss_fkd, loss_proto = self._compute_ssre_loss(inputs, labels)
        loss = loss_clf + loss_fkd + loss_proto
        loss.backward()
        self.opt.step()
        return loss.item()

    def _compute_ssre_loss(self, inputs, targets):
        if self._cur_task == 0:
            logits = self.net(inputs)
            loss_clf = F.cross_entropy(logits/self.args.temp, targets)
            return loss_clf, torch.tensor(0.), torch.tensor(0.)

        features = self.net.forward(inputs, returnt='features')  # N D

        with torch.no_grad():
            features_old = self._old_network.forward(inputs, returnt='features')

        protos = torch.from_numpy(np.array(self._protos)).to(self.device)  # C D
        with torch.no_grad():
            weights = F.normalize(features, p=2, dim=1, eps=1e-12) @ F.normalize(protos, p=2, dim=1, eps=1e-12).T
            weights = torch.max(weights, dim=1)[0]
            # mask = weights > self.args.threshold
            mask = weights
        logits = self.net(inputs)
        loss_clf = F.cross_entropy(logits/self.args.temp, targets, reduction="none")
        # loss_clf = torch.mean(loss_clf * ~mask)
        loss_clf = torch.mean(loss_clf * (1-mask))

        loss_fkd = torch.norm(features - features_old, p=2, dim=1)
        loss_fkd = self.args.lambda_fkd * torch.sum(loss_fkd * mask)

        index = np.random.choice(range(self._known_classes), size=self.args.batch_size, replace=True)

        proto_features = np.array(self._protos)[index]
        proto_targets = index
        proto_features = torch.from_numpy(proto_features).float().to(self.device, non_blocking=True)
        proto_targets = torch.from_numpy(proto_targets).to(self.device, non_blocking=True)

        proto_logits = self.net.fc(proto_features)
        loss_proto = self.args.lambda_proto * F.cross_entropy(proto_logits/self.args.temp, proto_targets)
        return loss_clf, loss_fkd, loss_proto

    def end_task(self, dataset):
        self._build_protos(dataset)
        self._network_compression()

        self._known_classes = self._total_classes
        self._old_network = copy.deepcopy(self.net)
        for param in self._old_network.parameters():
            param.requires_grad = False
        self._old_network.eval()

        if self._cur_task == 0:
            dataset.args.additional_augmentations = False

        self._cur_task += 1

    def _build_protos(self, dataset):
        class_features = self.compute_class_features(dataset)
        for class_idx in range(self._known_classes, self._total_classes):
            class_mean = np.mean(class_features[class_idx], axis=0)
            self._protos.append(class_mean)

    @torch.no_grad()
    def compute_class_features(self, dataset) -> dict:
        train_dataset = copy.deepcopy(dataset.train_loader.dataset)
        test_transforms = dataset.test_loaders[-1].dataset.transform
        train_dataset.transform = test_transforms
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

        class_features = collections.defaultdict(list)
        self.net.eval()

        for images, targets, _ in train_loader:
            images = images.to(self.device)
            features = self.net.forward(images, returnt='features')

            for fv, label in zip(features, targets):
                class_features[label.item()].append(fv)

        class_features = {label: torch.stack(features).cpu().numpy() for label, features in class_features.items()}
        return class_features

    def _network_expansion(self):
        if self._cur_task > 0:
            for p in self.net.parameters():
                p.requires_grad = False
            for p in self.net.fc.parameters():
                p.requires_grad = True
            for k, v in self.net.named_parameters():
                if 'adapter' in k:
                    v.requires_grad = True
        # self.net.re_init_params() # do not use!
        self.net.switch("parallel_adapters")

    def _network_compression(self):
        model_dict = self.net.state_dict()
        for k, v in model_dict.items():
            if 'adapter' in k:
                k_conv3 = k.replace('adapter', 'conv')
                if 'weight' in k:
                    model_dict[k_conv3] = model_dict[k_conv3] + F.pad(v, [1, 1, 1, 1], 'constant', 0)
                    model_dict[k] = torch.zeros_like(v)
                elif 'bias' in k:
                    model_dict[k_conv3] = model_dict[k_conv3] + v
                    model_dict[k] = torch.zeros_like(v)
                else:
                    assert 0
        self.net.load_state_dict(model_dict)
        self.net.switch("normal")
