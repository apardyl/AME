import os
import PIL.Image
import torch
import numpy as np
import torchvision.transforms

from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, Resize, ToTensor, Normalize, \
    InterpolationMode

from config import IMG_SIZE
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_AUG_IMG_TRANSFORM = torchvision.transforms.Compose([
    RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
    RandomHorizontalFlip(),
    Resize(IMG_SIZE),
    ToTensor(),
    Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

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
        sample = Image.open(os.path.join(self.root_dir, sample)).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, torch.zeros(1)  # empty


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
        image = Image.open(os.path.join(self.root_dir, "images", image_id)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        mask = Image.open(os.path.join(self.root_dir, "masks", image_id.replace(".jpg", ".png")))
        mask = mask.resize(IMG_SIZE, resample=PIL.Image.NEAREST)
        mask = torch.tensor(np.array(mask), dtype=torch.int64)

        # Augment sample
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return image, mask
