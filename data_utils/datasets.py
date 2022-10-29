import os
import torch
import numpy as np

from torch.utils.data import Dataset, ConcatDataset
from config import TRAIN_PATH, VALID_PATH, IMG_SIZE
from PIL import Image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).unsqueeze_(1).unsqueeze_(1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).unsqueeze_(1).unsqueeze_(1)


class ReconstructionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = os.listdir(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sample = Image.open(os.path.join(self.root_dir, sample))
        sample = sample.resize(IMG_SIZE)
        sample = torch.tensor(np.array(sample))
        # Deal with grayscale images and permute to CxHxW order
        if len(sample.shape) < 3:
            sample = sample.unsqueeze(2).expand((*sample.shape, 3))
        sample = sample.permute(2, 0, 1)
        # Augment sample
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def custom_collate(data):
        data = torch.stack(data) / 255
        data -= IMAGENET_MEAN
        data /= IMAGENET_STD
        return {"image": data,
                "target": data}


def create_train_val_datasets(dataset_class):
    train_ds = dataset_class(TRAIN_PATH)
    valid_ds = dataset_class(VALID_PATH)
    return train_ds, valid_ds
