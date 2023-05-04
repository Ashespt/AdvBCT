""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler


def create_scheduler(cfg, optimizer):
    num_epochs = cfg.TRAIN.EPOCHS
    # type 1
    # lr_min = 0.01 * cfg.TRAIN.LR_SCHEDULER.BASE_LR
    # warmup_lr_init = 0.001 * cfg.TRAIN.LR_SCHEDULER.BASE_LR
    # type 2
    lr_min = 0.002 * cfg.TRAIN.LR_SCHEDULER.BASE_LR
    warmup_lr_init = 0.01 * cfg.TRAIN.LR_SCHEDULER.BASE_LR
    # type 3
    # lr_min = 0.001 * cfg.TRAIN.LR_SCHEDULER.BASE_LR
    # warmup_lr_init = 0.01 * cfg.TRAIN.LR_SCHEDULER.BASE_LR

    warmup_t = cfg.TRAIN.WARMUP_EPOCHS
    noise_range = None

    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=lr_min,
            t_mul= 1.,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
        )

    return lr_scheduler
