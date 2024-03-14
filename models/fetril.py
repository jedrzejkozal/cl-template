import copy
import torch
import collections
import numpy as np

from torch import Tensor
from torch.utils.data import DataLoader
from models.utils.continual_model import ContinualModel
from sklearn.metrics import pairwise_distances
from sklearn.svm import LinearSVC

from utils.args import add_management_args, add_experiment_args, ArgumentParser


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Feature Translation for Exemplar-Free Class-Incremental Learning, https://arxiv.org/pdf/2211.13131.pdf')
    add_management_args(parser)
    add_experiment_args(parser)

    parser.add_argument('--svc_tolerance', default=0.0001, type=float, help='SVC tol parameter')
    parser.add_argument('--svc_regularization', default=1.0, type=float, help='SVC C hyperparameter')

    return parser


class FeTrIL(ContinualModel):
    NAME = 'fetril'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.t = 0
        self.old_class_means = dict()
        self.clf = None
        self.classes_so_far = 0

    def begin_task(self, dataset):
        if self.args.half_classes_in_first_task and self.t == 0:
            self.classes_so_far += dataset.N_CLASSES // 2
        else:
            self.classes_so_far += dataset.N_CLASSES_PER_TASK

    def observe(self, inputs: Tensor, labels: Tensor, not_aug_inputs: Tensor) -> float:
        if self.t == 0:
            self.opt.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.opt.step()
            return loss.item()
        return 0

    def end_task(self, dataset):
        if self.t == 0:
            for param in self.net.parameters():
                param.requires_grad = False

        new_class_features = self.compute_features(dataset)
        new_class_means = self.compute_class_means(new_class_features)
        X_train, y_train = self.compute_new_features(new_class_features)
        if self.t > 0:
            X_train_pseudo, y_train_pseudo = self.compute_pseudofeatures(new_class_features, new_class_means)
            X_train = np.concatenate([X_train, X_train_pseudo], axis=0)
            y_train = np.concatenate([y_train, y_train_pseudo], axis=0)
        X_train = X_train / np.linalg.norm(X_train, ord=2, axis=1, keepdims=True)
        self.clf = LinearSVC(penalty='l2', dual=False, tol=self.args.svc_tolerance,
                             C=self.args.svc_regularization, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0)
        self.clf.fit(X_train, y_train)

        self.old_class_means.update(new_class_means)

        self.t += 1

    def compute_features(self, dataset) -> dict:
        train_dataset = copy.deepcopy(dataset.train_loader.dataset)
        test_transforms = dataset.test_loaders[-1].dataset.transform
        train_dataset.transform = test_transforms
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

        class_features = collections.defaultdict(list)
        self.net.eval()

        with torch.no_grad():
            for images, targets, _ in train_loader:
                images = images.to(self.device)
                features = self.net.forward(images, returnt='features')

                for fv, label in zip(features, targets):
                    class_features[label.item()].append(fv)

        class_features = {label: torch.stack(features).cpu().numpy() for label, features in class_features.items()}
        return class_features

    def compute_class_means(self, class_features: dict) -> dict:
        class_means = {label: np.mean(features, axis=0) for label, features in class_features.items()}
        return class_means

    def compute_new_features(self, new_class_features: dict):
        new_classes = sorted(new_class_features.keys())

        features = list()
        labels = list()

        for i in new_classes:
            features.append(new_class_features[i])
            labels.append(np.full(shape=[len(new_class_features[i])], fill_value=i))
        X_train = np.concatenate(features, axis=0)
        y_train = np.concatenate(labels, axis=0)

        assert X_train.shape[0] == y_train.shape[0]
        return X_train, y_train

    def compute_pseudofeatures(self, new_class_features: dict, new_class_means: dict):
        new_classes = sorted(new_class_means.keys())
        new_class_offset = min(new_classes)
        new_class_means_array = np.stack([new_class_means[i] for i in new_classes])

        old_classes = sorted(self.old_class_means.keys())
        old_class_means_array = np.stack([self.old_class_means[i] for i in old_classes])
        distances = pairwise_distances(old_class_means_array, new_class_means_array)

        features = list()
        labels = list()
        for i in old_classes:
            similar_class_idx = np.argmin(distances[i])
            similar_class = similar_class_idx + new_class_offset
            pseudofeatures = new_class_features[similar_class] - np.expand_dims(new_class_means[similar_class], 0) + np.expand_dims(self.old_class_means[i], 0)
            features.append(pseudofeatures)
            class_labels = np.full(shape=[len(new_class_features[similar_class])], fill_value=i)
            labels.append(class_labels)
            assert len(pseudofeatures) == len(class_labels)

        X_train = np.concatenate(features, axis=0)
        y_train = np.concatenate(labels, axis=0)

        assert X_train.shape[0] == y_train.shape[0]
        return X_train, y_train

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net.forward(x, returnt='features').cpu().numpy()
        features = features / np.linalg.norm(features, ord=2, axis=1, keepdims=True)
        pred = self.clf.predict(features)
        pred_out = torch.zeros((len(x), self.classes_so_far))
        for i, idx in enumerate(pred):
            pred_out[i, idx] = 1

        return pred_out.to(self.device)

    def get_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt, T_max=self.args.n_epochs)

    def get_epochs(self):
        if self.t == 0:
            return self.args.n_epochs
        else:
            return 0
