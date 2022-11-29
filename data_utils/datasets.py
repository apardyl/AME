import os
import PIL.Image
import torch
import numpy as np
import torchvision.transforms

from torch.utils.data import Dataset
from config import IMG_SIZE
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMG_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMG_SIZE),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


class ReconstructionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=DEFAULT_IMG_TRANSFORM):
        self.root_dir = root_dir
        self.transform = transform
        self.data = os.listdir(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sample = Image.open(os.path.join(self.root_dir, sample))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, torch.zeros(1)  # empty target for collate


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=DEFAULT_IMG_TRANSFORM, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
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

        # Augment sample
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return image, mask


def create_train_val_datasets(dataset_class, train_path, valid_path, **kwargs):
    train_ds = dataset_class(train_path, **kwargs)
    valid_ds = dataset_class(valid_path, **kwargs)
    return train_ds, valid_ds
