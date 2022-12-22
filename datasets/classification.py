import abc
import os
import sys
from collections import Counter
from typing import Optional

import torch
from PIL import Image
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from datasets.base import BaseDataModule
from datasets.utils import get_default_aug_img_transform, get_default_img_transform


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, label_list, transform=None):
        self.data = file_list
        self.labels = label_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sample = Image.open(sample).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.labels[index]

    def class_stats(self):
        return [v for k, v in sorted(Counter(self.labels).items())]


class BaseClassificationDataModule(BaseDataModule, abc.ABC):
    num_classes = 0

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        print(f'Train class statistics:', self.train_dataset.class_stats(), file=sys.stderr)
        return super().train_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        print(f'Test class statistics:', self.test_dataset.class_stats(), file=sys.stderr)
        return super().test_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        print(f'Val class statistics:', self.val_dataset.class_stats(), file=sys.stderr)
        return super().val_dataloader()


class Sun360Classification(BaseClassificationDataModule):
    has_test_data = False
    num_classes = 26

    def setup(self, stage: Optional[str] = None) -> None:
        with open(os.path.join(os.path.dirname(__file__), 'meta/sun360-dataset26-random.txt')) as f:
            file_list = f.readlines()
        labels = [str.join('/', p.split('/')[:2]) for p in file_list]
        classes = {name: idx for idx, name in enumerate(sorted(set(labels)))}
        labels = [classes[x] for x in labels]
        file_list = [os.path.join(self.data_dir, p.strip()) for p in file_list]
        val_list = file_list[:len(file_list) // 10]
        val_labels = labels[:len(file_list) // 10]
        train_list = file_list[len(file_list) // 10:]
        train_labels = labels[len(file_list) // 10:]

        if stage == 'fit':
            self.train_dataset = ClassificationDataset(file_list=train_list, label_list=train_labels,
                                                       transform=get_default_aug_img_transform(self.image_size,
                                                                                               scale=False))
            self.val_dataset = ClassificationDataset(file_list=val_list, label_list=val_labels,
                                                     transform=get_default_img_transform(self.image_size))
        else:
            raise NotImplemented()
