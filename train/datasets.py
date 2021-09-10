import os

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

import pandas as pd
import numpy as np

from PIL import Image

from tqdm import tqdm

label_file = "label.csv"


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


class SequentialDataset(Dataset):
    def __init__(
        self,
        root,
        window,
        stride=1,
        transform=None,
        target_transform=None,
    ):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        indices = []
        dfs = []
        base = 0
        for clip_path in os.listdir(root):
            full_path = os.path.join(root, clip_path)
            if os.path.isdir(full_path) and clip_path[0] != ".":
                df = pd.read_csv(
                    os.path.join(root, clip_path, label_file),
                )

                df.iloc[:, 0] = df.iloc[:, 0].apply(
                    lambda p: os.path.join(clip_path, p)
                )
                dfs.append(df)

                # split clip
                T = len(df)
                for o in range(0, (T % window), stride):
                    start = np.arange(0, T - window, stride)
                    end = start + window

                    # add base to convert start & end indices
                    indices.append(np.vstack((start, end)) + o + base)

                # update base T - 1 + 1 is the next start
                base += T

        self.indices = np.concatenate(indices, axis=1)
        self.df = pd.concat(dfs, axis=0)

    def __getitem__(self, idx):
        s, e = self.indices[:, idx]

        x = []
        y = []
        for i in range(s, e):
            image = Image.open(
                os.path.join(
                    self.root,
                    self.df.iloc[i, 0],
                )
            )
            label = self.df.iloc[i, 1:].values.astype(np.float32)

            if self.transform is None:
                image = np.array(image)
                image = image.transpose((2, 0, 1))
            else:
                image = self.transform(image)
            if self.target_transform is not None:
                label = self.target_transform(label)

            x.append(image)
            y.append(label)

        return np.stack(x), np.array(y)

    def __len__(self):
        return self.indices.shape[-1]


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
