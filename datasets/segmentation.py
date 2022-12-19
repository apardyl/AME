import abc

from datasets.base import BaseDataModule


class BaseSegmentationDataModule(BaseDataModule, abc.ABC):
    num_classes = 0
