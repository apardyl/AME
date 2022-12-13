import abc
from argparse import ArgumentParser

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import RandomSampler, DataLoader


class BaseDataModule(LightningDataModule, abc.ABC):
    def __init__(self, args, has_test_data=True):
        super().__init__()

        self.data_dir = args.data_dir
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.num_workers = args.num_workers
        self.num_samples = args.num_samples

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.has_test_data = has_test_data

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument('--data-dir',
                            help='dataset location',
                            type=str,
                            required=True)
        parser.add_argument('--train-batch-size',
                            help='batch size for training',
                            type=int,
                            default=16)
        parser.add_argument('--eval-batch-size',
                            help='batch size for validation and testing',
                            type=int,
                            default=64)
        parser.add_argument('--num-workers',
                            help='dataloader workers per DDP process',
                            type=int,
                            default=4)
        parser.add_argument('--num-samples',
                            help='number of images to sample in each training epoch',
                            type=int,
                            default=50000)
        return parent_parser

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers,
                          sampler=RandomSampler(self.train_dataset, replacement=True, num_samples=self.num_samples))

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, shuffle=False,
                          num_workers=self.num_workers)
