import os
import torch
from torch.utils.data import Dataset, DataLoader

# datamodule
import pandas as pd
import numpy as np

from PIL import Image

BATCH_SIZE = 64
ROOT = "/work/dataset"
IMGSIZE = (64, 48)
LR = 1e-3


class CustomDataset(Dataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        self.path_col = "path"
        self.label_cols = ["vel", "ang"]

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        df = pd.read_csv(os.path.join(root, "label.csv"))

        if split != "all":
            if split == "train":
                split = 0
            elif split == "val":
                split = 1
            elif split == "test":
                split = 2

            df.where(df.loc[:, "split"] == split)
            df.dropna()

        self.df = df.drop(["split"], axis=1)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root, self.df.loc[idx, self.path_col]))
        vel_ang = self.df.loc[idx, self.label_cols].values.astype(np.float32)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            vel_ang = self.target_transform(vel_ang)

        return image, vel_ang

    def __len__(self):
        return len(self.df)


class RamDataset(Dataset):
    def __init__(self, dataset, f, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        if not os.path.isfile(f):
            dataloader = DataLoader(dataset, batch_size=2048)

            data = []
            label = []
            for x, y in dataloader:
                print("hi")
                data.append(x)
                label.append(y)

            data = torch.cat(data, dim=0)
            label = torch.cat(label, dim=0)

            torch.save((data, label), f)

        self.data, self.label = torch.load(f)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(x)

        return x, y

    def __len__(self):
        return len(self.label)
