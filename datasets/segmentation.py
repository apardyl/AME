import abc
import os
from typing import Tuple, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader

from config import IMG_SIZE
from datasets.base import BaseDataModule


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, num_classes, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes
        self.images = os.listdir(os.path.join(root_dir, "images"))
        self.masks = os.listdir(os.path.join(root_dir, "masks"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_id = self.images[index]
        image = Image.open(os.path.join(self.root_dir, "images", image_id))
        image = image.resize(IMG_SIZE, resample=Image.BILINEAR).convert("RGB")

        mask = Image.open(os.path.join(self.root_dir, "masks", image_id.replace(".jpg", ".png")))
        mask = mask.resize(IMG_SIZE, resample=Image.NEAREST)

        image, mask = self.transform(image, mask)

        mask = torch.clip(mask, 0, self.num_classes)
        mask[mask == self.num_classes] = 0

        return image, mask


class BaseSegmentationDataModule(BaseDataModule, abc.ABC):
    def __init__(self, args, num_classes):
        super().__init__(args)
        self.num_classes = num_classes

        self.train_dir, self.val_dir, self.test_dir = self._get_split_dirs(self.data_dir)

    @staticmethod
    @abc.abstractmethod
    def _get_split_dirs(data_dir: str) -> Tuple[str, str, str]:
        raise NotImplemented()

    def prepare_data(self) -> None:
        assert self.train_dir and os.path.exists(self.train_dir)
        assert self.val_dir and os.path.exists(self.val_dir)
        assert self.test_dir and os.path.exists(self.test_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_dataset = SegmentationDataset(root_dir=self.train_dir, num_classes=self.num_classes)
            print(f'Loaded {len(self.train_dataset)} train samples')
            self.val_dataset = SegmentationDataset(root_dir=self.val_dir, num_classes=self.num_classes)
            print(f'Loaded {len(self.val_dataset)} val samples')
        elif stage == 'test':
            self.test_dataset = SegmentationDataset(root_dir=self.test_dir, num_classes=self.num_classes)
            print(f'Loaded {len(self.val_dataset)} test samples')
        else:
            raise NotImplemented()


class ADE20KSegmentation(BaseSegmentationDataModule):
    def __init__(self, args):
        super().__init__(args, num_classes=150)

    @staticmethod
    def _get_split_dirs(data_dir: str) -> Tuple[str, str, str]:
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        test_dir = os.path.join(data_dir, 'test')
        return train_dir, val_dir, test_dir
