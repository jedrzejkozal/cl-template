# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
from argparse import Namespace
from typing import Tuple

import torch
from datasets import get_dataset
from datasets.utils.continual_benchmark import ContinualBenchmark
from models.utils.continual_model import ContinualModel

from utils.loggers import *
from utils.mlflow_logger import MLFlowLogger
from utils.status import ProgressBar


def mask_classes(outputs: torch.Tensor, dataset: ContinualBenchmark, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
            dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualBenchmark, last=False, debug=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    test_accs, test_accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        acc_class_incr, acc_task_incr = compute_acc(model, dataset, k, test_loader, debug=debug)
        test_accs.append(acc_class_incr)
        test_accs_mask_classes.append(acc_task_incr)

    train_acc, train_acc_mask_classes = compute_acc(model, dataset, k, dataset.train_loader, debug=debug)

    model.net.train(status)
    print('\nevaluation task train acc:')
    print('{:.2f}'.format(train_acc))
    print('\nevaluation task test accs:')
    accs_str = ''
    for a in test_accs:
        accs_str += '{:.2f}, '.format(a)
    accs_str = accs_str[:-2]
    print(accs_str)
    return train_acc, train_acc_mask_classes, test_accs, test_accs_mask_classes


def compute_acc(model, dataset, k, dataloader, debug=False):
    correct, correct_mask_classes, total = 0.0, 0.0, 0.0
    for i, data in enumerate(dataloader):
        with torch.no_grad():
            if debug and i > 3:
                break
            inputs = data[0]
            labels = data[1]
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                outputs = model(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

    acc_class_incr = correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0
    acc_task_incr = correct_mask_classes / total * 100
    return acc_class_incr, acc_task_incr


def train(model: ContinualModel, dataset: ContinualBenchmark,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """

    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log and not args.debug:
        logger = MLFlowLogger(dataset.SETTING, dataset.NAME, model.NAME,
                              experiment_name=args.experiment_name, parent_run_id=args.parent_run_id, run_name=args.run_name)
        logger.log_args(args.__dict__)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    if not args.ignore_other_metrics and not args.debug:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            _, _, random_results_class, random_results_task = evaluate(model, dataset_copy)

    if os.path.exists('old_model.pt'):
        os.remove('old_model.pt')
    if os.path.exists('net.pt'):
        os.remove('net.pt')

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t and not args.ignore_other_metrics and not args.debug:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[2]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[3]

        if hasattr(model, 'get_scheduler'):
            scheduler = model.get_scheduler()
        else:
            scheduler = dataset.get_scheduler(model, args)

        n_epochs = model.get_epochs() if hasattr(model, 'get_epochs') else model.args.n_epochs
        for epoch in range(n_epochs):
            if args.model == 'joint':
                continue
            for i, data in enumerate(train_loader):
                if args.debug and i > 3:
                    break
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs)
                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)

            if scheduler is not None:
                scheduler.step()

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        # if 'clewi' in model.NAME and os.path.exists('old_model.pt') and os.path.exists('net.pt'):
        #     logger.log_artifact('old_model.pt', f'old_model_task_{t}')
        #     logger.log_artifact('net.pt', f'net_model_task_{t}')

        accs = evaluate(model, dataset)
        results.append(accs[2])
        results_mask_classes.append(accs[3])

        mean_acc = np.mean(accs[2:], axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if not args.disable_log and not args.debug:
            logger.log(mean_acc)
            logger.log_fullacc(accs[2:])
            logger.log_train_acc(accs[0], accs[1])

    if not args.disable_log and not args.debug:
        logger.add_forgetting(results, results_mask_classes)
        if not args.ignore_other_metrics:
            logger.add_bwt(results, results_mask_classes)
            if model.NAME != 'icarl' and model.NAME != 'pnn':
                logger.add_fwt(results, random_results_class,
                               results_mask_classes, random_results_task)
