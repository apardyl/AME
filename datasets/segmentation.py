import abc
import os
from typing import Tuple, Optional

from datasets.base import BaseDataModule


class BaseSegmentationDataModule(BaseDataModule, abc.ABC):
    def __init__(self, *args, num_classes, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            raise NotImplemented()  # TODO, should return pairs of (image, segmentation)
            self.train_dataset = None  # TODO
            print(f'Loaded {len(self.train_dataset)} train samples')
            self.val_dataset = None  # TODO
            print(f'Loaded {len(self.val_dataset)} val samples')
        elif stage == 'test':
            raise NotImplemented()
            self.test_dataset = None  # TODO
            print(f'Loaded {len(self.test_dataset)} test samples')
        else:
            raise NotImplemented()
