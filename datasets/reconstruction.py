import abc
import os.path
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomResizedCrop, InterpolationMode, RandomHorizontalFlip, Resize, \
    ToTensor, Normalize

from datasets.base import BaseDataModule
from datasets.utils import IMAGENET_MEAN, IMAGENET_STD


def get_default_aug_img_transform(img_size):
    return Compose([
        RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
        RandomHorizontalFlip(),
        Resize(img_size),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_default_img_transform(img_size):
    return Compose([
        Resize(img_size),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


class ReconstructionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = os.listdir(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sample = Image.open(os.path.join(self.root_dir, sample)).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, torch.zeros(1)  # empty


class BaseReconstructionDataModule(BaseDataModule, abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_dir, self.val_dir, self.test_dir = self._get_split_dirs(self.data_dir)

    @staticmethod
    @abc.abstractmethod
    def _get_split_dirs(data_dir: str) -> Tuple[str, str, Optional[str]]:
        raise NotImplemented()

    def prepare_data(self) -> None:
        assert self.train_dir and os.path.exists(self.train_dir)
        assert self.val_dir and os.path.exists(self.val_dir)
        assert self.test_dir is None or (self.test_dir and os.path.exists(self.test_dir))

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_dataset = ReconstructionDataset(root_dir=self.train_dir,
                                                       transform=get_default_aug_img_transform(self.image_size))
            print(f'Loaded {len(self.train_dataset)} train samples')
            self.val_dataset = ReconstructionDataset(root_dir=self.val_dir,
                                                     transform=get_default_img_transform(self.image_size))
            print(f'Loaded {len(self.val_dataset)} val samples')
        elif stage == 'test':
            self.test_dataset = ReconstructionDataset(root_dir=self.test_dir,
                                                      transform=get_default_img_transform(self.image_size))
            print(f'Loaded {len(self.test_dataset)} test samples')
        else:
            raise NotImplemented()


class Coco2014Reconstruction(BaseReconstructionDataModule):
    @staticmethod
    def _get_split_dirs(data_dir: str) -> Tuple[str, str, str]:
        train_dir = os.path.join(data_dir, 'train2014')
        val_dir = os.path.join(data_dir, 'val2014')
        test_dir = os.path.join(data_dir, 'test2014')
        return train_dir, val_dir, test_dir
