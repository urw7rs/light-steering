import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

# datamodule
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
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
    def __init__(self, dataset, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        dataloader = DataLoader(dataset, batch_size=20480)

        data = []
        label = []
        for x, y in dataloader:
            data.append(x)
            label.append(y)

        self.data = torch.cat(data, dim=0)
        self.label = torch.cat(label, dim=0)

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


class POCDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, img_size, augmentation=None, fill_memory=True):
        super().__init__()

        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(img_size)]
        )

        self.augmentation = augmentation

        self.memory = fill_memory

        # self.dims = (3, *img_size)

    def prepare_data(self):
        path_col = "path"
        label_cols = ["vel", "ang"]

        pairs = []
        for clip_path in os.listdir(self.data_dir):
            if not os.path.isdir(os.path.join(self.data_dir, clip_path)):
                continue
            elif clip_path[0] == ".":
                continue

            df = pd.read_csv(
                os.path.join(self.data_dir, clip_path, "label.csv"),
                usecols=[path_col, *label_cols],
            )

            df[path_col] = df[path_col].apply(
                lambda path: os.path.join(clip_path, path)
            )

            pairs.append(df)

        # filter bad data
        df = df[df.loc[:, "vel"] > 0]

        # split into train val test 0.96 0.02 0.02
        train, test = train_test_split(df, test_size=0.2)
        test, val = train_test_split(test, test_size=0.5)

        # add split column and concat
        splits = [train, val, test]
        for i, split in enumerate(splits):
            split.insert(len(split.columns), "split", [i] * len(split))
        splits = pd.concat([train, val, test], axis=0)

        # save combined labels to csv file
        splits.to_csv(os.path.join(self.data_dir, "label.csv"))

        # find mean and std for normalization
        norm_path = os.path.join(self.data_dir, "norm.csv")
        if not os.path.isfile(norm_path):
            print("normalization file not found computing mean and std...")

            full_dataset = CustomDataset(
                self.data_dir, split="all", transform=self.transform
            )
            norm_dataloader = DataLoader(
                full_dataset,
                num_workers=len(os.sched_getaffinity(0)),
                batch_size=BATCH_SIZE,
            )

            mean = 0
            mean_squared = 0
            n = 0
            for x, _ in norm_dataloader:
                x = x.to("cuda")
                n += x.shape[1]
                with torch.no_grad():
                    mean += x.mean(dim=(0, 2, 3))
                    mean_squared += (x ** 2).mean(dim=(0, 2, 3))

            with torch.no_grad():
                mean = mean / n
                mean_squared = mean_squared / n
                std = torch.sqrt(mean_squared - mean ** 2)

            mean = mean.cpu().numpy().tolist()
            std = std.cpu().numpy().tolist()

            df = pd.DataFrame(dict(mean=mean, std=std))
            df.to_csv(norm_path)
        else:
            df = pd.read_csv(norm_path)
            mean = df["mean"].tolist()
            std = df["std"].tolist()

        if self.memory:
            self.augmentation = transforms.Compose(
                [*self.augmentation, transforms.Normalize(mean, std)]
            )
        else:
            self.transform.transforms.extend(
                [*self.augmentation, transforms.Normalize(mean, std)]
            )

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = CustomDataset(
                self.data_dir, split="train", transform=self.transform
            )
            self.val = CustomDataset(
                self.data_dir, split="val", transform=self.transform
            )

            if self.memory:
                self.train = RamDataset(self.train, transform=self.augmentation)
                self.val = RamDataset(self.val, transform=self.augmentation)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = CustomDataset(
                self.data_dir, split="test", transform=self.transform
            )

            if self.memory:
                self.test = RamDataset(self.test, transform=self.augmentation)

    def get_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=len(os.sched_getaffinity(0)),
            shuffle=shuffle,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self.get_dataloader(self.train, True)

    def val_dataloader(self):
        return self.get_dataloader(self.val)

    def test_dataloader(self):
        return self.get_dataloader(self.test)


class LitLightSteer(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(100),
            nn.ReLU(),
            nn.LazyLinear(100),
            nn.ReLU(),
            nn.LazyLinear(2),
        )

    def forward(self, x):
        y = self.model(x)
        y[:, 0] = 1.2 * F.sigmoid(y[:, 0])
        y[:, 1] = 0.7 * F.tanh(y[:, 1])
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("learning_rate", self.learning_rate, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": "val_loss",
            },
        }


dm = POCDataModule(
    data_dir=ROOT,
    img_size=IMGSIZE,
    augmentation=[
        transforms.ColorJitter(brightness=0.8, contrast=0.5, saturation=0.5, hue=0.5)
    ],
)
model = LitLightSteer(learning_rate=LR)

checkpoint_callback = ModelCheckpoint(monitor="val_loss")
lr_monitor = LearningRateMonitor(logging_interval="step")

trainer = pl.Trainer(
    gpus=1, precision=16, callbacks=[checkpoint_callback, lr_monitor], max_epochs=200
)

trainer.fit(model, dm)
