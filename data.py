# References:
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/unaligned_dataset.py

from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
import random

import config


class UnpairedImageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        x_mean,
        x_std,
        y_mean,
        y_std,
        fixed_pairs=False,
        split="train",
    ):
        super().__init__()

        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.fixed_pairs = fixed_pairs
        self.split = split

        self.x_paths = list(Path(data_dir).glob(f"""{split}A/*.jpg"""))
        self.x_len = len(self.x_paths)

        self.y_paths = list(Path(data_dir).glob(f"""{split}B/*.jpg"""))
        self.y_len = len(self.y_paths)

        self.rand_resized_crop = T.RandomResizedCrop(
            size=config.IMG_SIZE, scale=config.SCALE, ratio=(1, 1), antialias=True,
        ) # Not in the paper.

    def transform(self, x, y):
        x = self.rand_resized_crop(x)
        y = self.rand_resized_crop(y)
        if self.split == "train":
            if random.random() > 0.5:
                x = TF.hflip(x)
            if random.random() > 0.5:
                y = TF.hflip(y)

        x = T.ToTensor()(x)
        x = T.Normalize(mean=self.x_mean, std=self.x_std)(x)

        y = T.ToTensor()(y)
        y = T.Normalize(mean=self.y_mean, std=self.y_std)(y)
        return x, y

    def __len__(self):
        return max(self.x_len, self.y_len)

    def __getitem__(self, idx):
        if self.fixed_pairs:
            x_path = self.x_paths[idx % self.x_len]
            y_path = self.y_paths[idx % self.y_len]
        elif self.x_len >= self.y_len:
            x_path = self.x_paths[idx]
            y_path = random.choice(self.y_paths)
        else:
            y_path = self.y_paths[idx]
            x_path = random.choice(self.x_paths)
        x = Image.open(x_path).convert("RGB")
        y = Image.open(y_path).convert("RGB")
        x, y = self.transform(x=x, y=y)
        return x, y


class OneSideImageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        x_or_y,
        mean,
        std,
        split="train",
    ):
        super().__init__()

        self.x_or_y = x_or_y
        self.mean = mean
        self.std = std
        self.split = split

        if x_or_y == "x":
            self.paths = list(Path(data_dir).glob(f"""{split}A/*.jpg"""))
        elif x_or_y == "y":
            self.paths = list(Path(data_dir).glob(f"""{split}B/*.jpg"""))

    def transform(self, image):
        if self.split == "train":
            if random.random() > 0.5:
                image = TF.hflip(image)

        image = T.ToTensor()(image)
        image = T.Normalize(mean=self.mean, std=self.std)(image)
        return image

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        image = self.transform(image)
        return image
