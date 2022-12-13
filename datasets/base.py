import abc
from argparse import ArgumentParser

from pytorch_lightning import LightningDataModule


class BaseDataModule(LightningDataModule, abc.ABC):
    def __init__(self, args):
        super().__init__()

        self.data_dir = args.data_dir
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.num_workers = args.num_workers
        self.num_samples = args.num_samples

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument('--data_dir',
                            help='dataset location',
                            type=str,
                            required=True)
        parser.add_argument('--train_batch_size',
                            help='batch size for training',
                            type=int,
                            default=16)
        parser.add_argument('--eval_batch_size',
                            help='batch size for validation and testing',
                            type=int,
                            default=64)
        parser.add_argument('--num_workers',
                            help='dataloader workers per DDP process',
                            type=int,
                            default=4)
        parser.add_argument('--num_samples',
                            help='number of images to sample in each training epoch',
                            type=int,
                            default=50000)
        return parent_parser
