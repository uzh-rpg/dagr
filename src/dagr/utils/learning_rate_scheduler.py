from functools import partial
import math
from typing import List

import numpy as np


class LRSchedule:
    def __init__(self,
                 warmup_epochs: float,
                 num_iters_per_epoch: int,
                 tot_num_epochs: int,
                 min_lr_ratio: float=0.05,
                 warmup_lr_start: float=0,
                 steps_at_iteration=[50000],
                 reduction_at_step=0.5):

        warmup_total_iters = num_iters_per_epoch * warmup_epochs
        total_iters = tot_num_epochs * num_iters_per_epoch
        no_aug_iters = 0
        self.lr_func = partial(_yolox_warm_cos_lr, min_lr_ratio, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iters, steps_at_iteration, reduction_at_step)

    def __call__(self, *args, **kwargs)->float:
        return self.lr_func(*args, **kwargs)


def _yolox_warm_cos_lr(
    min_lr_ratio: float,
    total_iters: int,
    warmup_total_iters: int,
    warmup_lr_start: float,
    no_aug_iter: int,
    steps_at_iteration: List[int],
    reduction_at_step: float,
    iters: int)->float:
    """Cosine learning rate with warm up."""
    min_lr = min_lr_ratio
    if iters < warmup_total_iters:
        # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
        lr = (1 - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
    else:
        lr = min_lr + 0.5 * (1 - min_lr) * (1.0 + math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter)))

    for step in steps_at_iteration:
        if iters >= step:
            lr *= reduction_at_step

    return lr