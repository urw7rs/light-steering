import os

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

import pandas as pd
import numpy as np

from PIL import Image

from tqdm import tqdm


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

            df = df.where(df.loc[:, "split"] == split)
            df = df.dropna()

        self.df = df.drop(["split"], axis=1)

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(
                self.root,
                self.df.loc[self.df.index[idx], self.path_col],
            )
        )
        vel_ang = self.df.loc[self.df.index[idx], self.label_cols].values
        vel_ang = vel_ang.astype(np.float32)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            vel_ang = self.target_transform(vel_ang)

        return image, vel_ang

    def __len__(self):
        return len(self.df)


def cache_dataset(dataset, f):
    if os.path.isfile(f):
        data, label = torch.load(f)
    else:
        print(f"creating cache file {f}")
        # num_workers 0 and batch_size 1 is much faster
        dataloader = DataLoader(dataset, num_workers=0, batch_size=1)

        data = []
        label = []
        for i, (x, y) in enumerate(tqdm(dataloader)):
            data.append(x)
            label.append(y)

        data = torch.cat(data, dim=0)
        label = torch.cat(label, dim=0)

        torch.save((data, label), f)

    return TensorDataset(data, label)
