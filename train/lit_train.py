import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# datamodule
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from PIL import Image

# logger
from pytorch_lightning.loggers import TensorBoardLogger

import argparse

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


class POCDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, img_size, augmentation=None, memory=True):
        super().__init__()

        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(img_size)]
        )

        self.augmentation = augmentation

        self.memory = memory

        # self.dims = (3, *img_size)

    def prepare_data(self):

        label_path = os.path.join(self.data_dir, "label.csv")
        if not os.path.isfile(label_path):
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

            df = pd.concat(pairs, axis=0)

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

        if self.augmentation is None:
            after_transforms = [transforms.Normalize(mean, std)]
        else:
            after_transforms = [*self.augmentation, transforms.Normalize(mean, std)]

        if self.memory:
            self.augmentation = transforms.Compose(after_transforms)
        else:
            self.transform.transforms.extend(after_transforms)

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
                self.train = RamDataset(
                    dataset=self.train, f="train.pt", transform=self.augmentation
                )
                self.val = RamDataset(
                    dataset=self.val, f="val.pt", transform=self.augmentation
                )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = CustomDataset(
                self.data_dir, split="test", transform=self.transform
            )

            if self.memory:
                self.test = RamDataset(
                    dataset=self.test, f="test.pt", transform=self.augmentation
                )

    def get_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=len(os.sched_getaffinity(0)),
            shuffle=shuffle,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self.get_dataloader(self.train, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.val)

    def test_dataloader(self):
        return self.get_dataloader(self.test)


class LitLightSteer(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2),
            nn.Flatten(),
            nn.LazyLinear(500),
            nn.ReLU(),
            nn.LazyLinear(100),
            nn.ReLU(),
            nn.LazyLinear(2),
        )

    def forward(self, x):
        y = self.model(x)
        y[:, 0] = 1.2 * torch.sigmoid(y[:, 0])
        y[:, 1] = 0.7 * torch.tanh(y[:, 1])
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train model.")
    parser.add_argument(
        "checkpoint",
        help="checkpoint path",
    )

    arg = parser.parse_args()

    dm = POCDataModule(data_dir=ROOT, img_size=IMGSIZE, augmentation=None)
    model = LitLightSteer(learning_rate=LR)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=arg.checkpoint,
        filename="pocmodel-{epoch:02d}-{val_loss:.8f}",
        save_top_k=3,
        mode="min",
    )

    logger = TensorBoardLogger("tb_logs", name="model")
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        callbacks=[checkpoint_callback],
        max_epochs=50,
        default_root_dir=arg.checkpoint,
        logger=logger,
    )

    trainer.fit(model, dm)
