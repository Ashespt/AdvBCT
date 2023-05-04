import argparse
import collections
import json
import os
import random
import warnings
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from parse_config import ConfigParser


def cudalize(model,distributed,device,arch):
    """Select cuda or cpu mode on different machine"""
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if device is not None:
            torch.cuda.set_device(device)
            model.cuda(device)
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[device],
                                                              find_unused_parameters=False)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
    elif device is not None:
        torch.cuda.set_device(device)
        model = model.cuda(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if  arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    return model


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids



class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}:{val' + self.fmt + '}({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def tensor_to_float(y):
    y_value = y if type(y) == float else y.item()
    return y_value


def load_pretrained_model(model, pretrained_model_path, model_key_in_ckpt=None, logger=None):
    if os.path.isfile(pretrained_model_path):

        pretrained_model = torch.load(pretrained_model_path)
        if model_key_in_ckpt:
            pretrained_model = pretrained_model[model_key_in_ckpt]

        if 'checkpoint' in pretrained_model_path:
            unfit_keys = model.load_state_dict(pretrained_model["model"], strict=False)
        else:
            # modify key name in checkpoint
            modified_pretrained_model = OrderedDict()
            for k, v in pretrained_model.items():
                if k.startswith("module."):
                    modified_pretrained_model[k[7:]] = v
                # if not k.startswith("fc_") and not k.startswith("backbone."):
                elif not k.startswith(("fc_", "backbone.")) and '.' in k:
                    modified_pretrained_model[".".join(("backbone", k))] = v
                else:
                    modified_pretrained_model[k] = v
            unfit_keys = model.load_state_dict(modified_pretrained_model, strict=False)
        logger.info("=> loading pretrained_model from '{}'".format(pretrained_model_path))
        logger.info('=> these keys in model are not in state dict: {}'.format(unfit_keys.missing_keys))
        logger.info('=> these keys in state dict are not in model: {}'.format(unfit_keys.unexpected_keys))
        logger.info("=> loading done!")
    else:
        logger.info("=> no pretrained_model found at '{}'".format(pretrained_model_path))


def resume_checkpoint(model, optimizer, grad_scaler, args, logger):
    ckpt_path = args.resume
    if os.path.isfile(ckpt_path):
        logger.info("=> resume checkpoint '{}'".format(ckpt_path))
        if args.device is None:
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.device)
            checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['grad_scaler']:
            grad_scaler.load_state_dict(checkpoint['grad_scaler'])
        logger.info("=> successfully resuming checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        return best_acc1
    else:
        logger.info("=> no checkpoint found at '{}'".format(ckpt_path))
        return 0.


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0.
                    if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
