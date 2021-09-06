import os
from torch.utils.data import Dataset

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
        image = Image.open(
            os.path.join(
                self.root,
                self.df.loc[idx, self.path_col],
            )
        )
        vel_ang = self.df.loc[idx, self.label_cols].values
        vel_ang = vel_ang.astype(np.float32)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            vel_ang = self.target_transform(vel_ang)

        return image, vel_ang

    def __len__(self):
        return len(self.df)
