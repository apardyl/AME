import torch
import random


def augment_data(x):
    for transform in augmentations:
        x = transform(*x)
    return x


augmentations = []  # TODO
