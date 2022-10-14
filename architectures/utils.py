import inspect
import math
import os
import sys
import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class BaseArchitecture(nn.Module, ABC):
    """ Each architecture should be a subclass of this class and implement abstract methods in order
    to work with this template. Architecture should consist of a neural model, criterion, optimizer
    and its scheduler. Use on epoch/iter funcs to update scheduler over the course of training and to
    log stats."""

    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def add_args(parser):
        pass

    @abstractmethod
    def calculate_loss(self, out, target):
        pass

    def on_epoch_end(self):
        pass

    def on_iter_end(self):
        pass


def add_arch_args(parser):
    """ Iterate over all architectures in architectures dir, find the one the user provided in args and add architecture's
    args to the parser"""

    archs = {}
    for module in os.listdir(os.path.dirname(__file__)):
        if module == '__init__.py' or module == 'utils.py' or module[-3:] != '.py':
            continue
        module = f"architectures.{module}" if "architectures." not in module else module  # TODO: refactor this
        __import__(module[:-3], locals(), globals())
        for name, obj in inspect.getmembers(sys.modules[module[:-3]]):
            if inspect.isclass(obj) and hasattr(obj, "add_args") and callable(getattr(obj, "add_args")) and obj.__name__ != "BaseArchitecture":
                archs[name] = obj
    known_args = parser.parse_known_args()[0]
    if known_args.arch not in archs.keys():
        raise RuntimeError(f"Wrong architecture provided. Possible architectures: {archs.keys()}")
    archs[known_args.arch].add_args(parser)
    return parser, archs[known_args.arch]


class WarmUpScheduler(nn.Module):
    """Warm-up and exponential decay chain scheduler. If warm_up_iters > 0 than warm-ups linearly for warm_up_iters iterations.
    Then it decays the learning rate every epoch. It is a good idea to set warm_up_iters as total number of samples in epoch / batch size.
    Good for transformers"""

    def __init__(self, optimizer, warm_up_iters=0, lr_decay=0.97):
        super().__init__()
        self.total_steps, self.warm_up_iters = 0, warm_up_iters
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1e-6, total_iters=warm_up_iters) if warm_up_iters else None
        self.decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay, last_epoch=-1)

    def step_iter(self):
        self.total_steps += 1
        if self.warmup_scheduler:
            self.warmup_scheduler.step()

    def step_epoch(self):
        if self.total_steps > self.warm_up_iters:
            self.decay_scheduler.step()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x += self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AttentionGlobalPooling(nn.Module):
    """ Removes axis using weighted average of values along S axis.
    Weights are softmax probs. """

    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        """ Args:
            x: Tensor, shape BxSxE (batch, sequence, embedding)
        """
        weights = self.linear(x)
        weights = torch.softmax(weights, dim=1)
        x = torch.sum(x * weights, dim=1)
        return x

