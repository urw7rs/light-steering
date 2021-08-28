import os
import csv

import pandas as pd

import torch
import numpy as np

from torchvision.io import read_image
import torchvision.transforms as tsfms

from torch.utils.data import Dataset


class RawDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        
        pairs = []
        for clip_path in os.listdir(root):
            df = pd.read_csv(
                os.path.join(root, clip_path, "label.csv"),
                usecols=["path", "vel", "ang"]
            )
            df.iloc[:, 0] = df.iloc[:, 0].apply(lambda p: os.path.join(clip_path, p))
            
            pairs.append(df)

        self.df = pd.concat(pairs, axis=0)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        if isinstance(idx, int):
            image = read_image(os.path.join(self.root, self.df.iloc[idx, 0]))
        else:
            image = torch.stack(
                [read_image(
                    os.path.join(self.root, path)
                ) for path in self.df.iloc[idx, 0]]
            )
        
        image = image / 255.0
        
        vel_ang = torch.from_numpy(
            self.df.iloc[idx, 1:].values.astype(np.float32)
        )

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            vel_ang = self.target_transform(vel_ang)
            
        return image, vel_ang

    def __len__(self):
        return len(self.df)