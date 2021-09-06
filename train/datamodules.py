import os
import torch
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# local files
from datasets import CustomDataset, RamDataset


class POCDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir, img_size, batch_size=64, augmentation=None, memory=True
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(img_size)]
        )
        self.augmentation = augmentation
        self.memory = memory

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
            train, test = train_test_split(df, test_size=0.4)
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
                batch_size=self.batch_size,
            )

            mean = []
            squared_mean = []
            weights = []
            N = float(len(full_dataset))
            for x, _ in norm_dataloader:
                with torch.no_grad():
                    mean.append(x.mean(dim=(0, 2, 3)))
                    squared_mean.append((x ** 2).mean(dim=(0, 2, 3)))
                    weights.append(float(x.shape[1]))

            mean = torch.stack(mean, dim=-1)
            squared_mean = torch.stack(squared_mean, dim=-1)

            weights = torch.tensor(weights).unsqueeze(dim=-1)

            with torch.no_grad():
                mean = torch.mm(mean, weights) / N
                squared_mean = torch.mm(squared_mean, weights) / N
                std = torch.sqrt(squared_mean - mean ** 2)

            mean = mean.squeeze().numpy().tolist()
            std = std.squeeze().numpy().tolist()

            df = pd.DataFrame(dict(mean=mean, std=std))
            df.to_csv(norm_path)

            print("saved to csv file")
        else:
            df = pd.read_csv(norm_path)
            mean = df["mean"].tolist()
            std = df["std"].tolist()

        if self.augmentation is None:
            after_transforms = [
                transforms.Normalize(mean, std),
            ]
        else:
            after_transforms = [
                *self.augmentation,
                transforms.Normalize(mean, std),
            ]

        if self.memory:
            self.augmentation = transforms.Compose(after_transforms)
        else:
            self.transform.transforms.extend(after_transforms)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = CustomDataset(
                self.data_dir,
                split="train",
                transform=self.transform,
            )
            self.val = CustomDataset(
                self.data_dir,
                split="val",
                transform=self.transform,
            )

            if self.memory:
                self.train = RamDataset(
                    dataset=self.train,
                    f="train.pt",
                    transform=self.augmentation,
                )
                self.val = RamDataset(
                    dataset=self.val,
                    f="val.pt",
                    transform=self.augmentation,
                )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = CustomDataset(
                self.data_dir,
                split="test",
                transform=self.transform,
            )

            if self.memory:
                self.test = RamDataset(
                    dataset=self.test,
                    f="test.pt",
                    transform=self.augmentation,
                )

    def get_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
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
