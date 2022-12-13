import abc
import os.path
from typing import Optional, Tuple

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, RandomSampler

from datasets.base import BaseDataModule
from datasets.utils import ReconstructionDataset, DEFAULT_AUG_IMG_TRANSFORM


class BaseReconstructionDataModule(BaseDataModule, abc.ABC):
    def __init__(self, args):
        super().__init__(args)

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
            self.train_dataset = ReconstructionDataset(root_dir=self.train_dir, transform=DEFAULT_AUG_IMG_TRANSFORM)
            print(f'Loaded {len(self.train_dataset)} train samples')
            self.val_dataset = ReconstructionDataset(root_dir=self.val_dir)
            print(f'Loaded {len(self.val_dataset)} val samples')
        elif stage == 'test':
            self.test_dataset = ReconstructionDataset(root_dir=self.test_dir)
            print(f'Loaded {len(self.val_dataset)} test samples')
        else:
            raise NotImplemented()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers,
                          sampler=RandomSampler(self.train_dataset, replacement=True, num_samples=self.num_samples))

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, shuffle=False,
                          num_workers=self.num_workers)


class Coco2014Reconstruction(BaseReconstructionDataModule):
    @staticmethod
    def _get_split_dirs(data_dir: str) -> Tuple[str, str, str]:
        train_dir = os.path.join(data_dir, 'train2014')
        val_dir = os.path.join(data_dir, 'val2014')
        test_dir = os.path.join(data_dir, 'test2014')
        return train_dir, val_dir, test_dir
