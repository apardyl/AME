import os
import PIL.Image
import torch
import numpy as np

from torch.utils.data import Dataset
from config import IMG_SIZE
from PIL import Image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).unsqueeze_(1).unsqueeze_(1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).unsqueeze_(1).unsqueeze_(1)


class ReconstructionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

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


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(os.path.join(root_dir, "images"))
        self.masks = os.listdir(os.path.join(root_dir, "masks"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_id = self.images[index]
        image = Image.open(os.path.join(self.root_dir, "images", image_id))
        image = image.resize(IMG_SIZE)
        image = torch.tensor(np.array(image), dtype=torch.float32)
        # Deal with grayscale images and permute to CxHxW order
        if len(image.shape) < 3:
            image = image.unsqueeze(2).expand((*image.shape, 3))
        image = image.permute(2, 0, 1)

        mask = Image.open(os.path.join(self.root_dir, "masks", image_id.replace(".jpg", ".png")))
        mask = mask.resize(IMG_SIZE, resample=PIL.Image.NEAREST)
        mask = torch.tensor(np.array(mask), dtype=torch.int64)
        sample = (image, mask)

        # Augment sample
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def custom_collate(data):
        images = torch.stack([i[0] for i in data]) / 255
        images -= IMAGENET_MEAN
        images /= IMAGENET_STD
        masks = torch.stack([i[1] for i in data])
        return {"image": images,
                "target": masks}


def create_train_val_datasets(dataset_class, train_path, valid_path):
    train_ds = dataset_class(train_path)
    valid_ds = dataset_class(valid_path)
    return train_ds, valid_ds
