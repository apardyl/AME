import math

import torch
import torch.nn as nn


class WarmUpScheduler(nn.Module):
    """Warm-up and exponential decay chain scheduler. If warm_up_iters > 0 than warm-ups linearly for warm_up_iters iterations.
    Then it decays the learning rate every epoch. It is a good idea to set warm_up_iters as total number of samples in epoch / batch size.
    Good for transformers"""

    def __init__(self, optimizer, warm_up_iters=0, lr_decay=0.97):
        super().__init__()
        self.optimizer = optimizer
        self.total_steps, self.warm_up_iters = 0, warm_up_iters
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1e-6,
                                                                  total_iters=warm_up_iters) if warm_up_iters else None
        self.decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay, last_epoch=-1)

    def step(self):
        self.total_steps += 1
        if self.warmup_scheduler:
            self.warmup_scheduler.step()

    def step_epoch(self):
        if self.total_steps > self.warm_up_iters:
            self.decay_scheduler.step()


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class MaeScheduler(nn.Module):
    def __init__(self, optimizer, lr, warmup_epochs, min_lr, epochs):
        super().__init__()

        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.epochs = epochs
        self.optimizer = optimizer

    def step(self, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs
        else:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * \
                 (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr
