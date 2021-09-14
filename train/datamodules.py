import os

import torch
import pytorch_lightning as pl
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# local files
from datasets import CustomDataset, SequentialDataset


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, **kwargs):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

    def get_dataloader(self, dataset, shuffle=False, drop_last=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=len(os.sched_getaffinity(0)),
            shuffle=shuffle,
            pin_memory=True,
            drop_last=drop_last,
        )

    def train_dataloader(self):
        return self.get_dataloader(self.train, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self.get_dataloader(self.val)

    def test_dataloader(self):
        return self.get_dataloader(self.test)


class POCDataModule(BaseDataModule):
    def __init__(
        self,
        img_size,
        train_f="train.pt",
        val_f="val.pt",
        test_f="test.pt",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.train_f = train_f
        self.val_f = val_f
        self.test_f = test_f

        self.transform = transforms.ToTensor()

    def prepare_data(self):
        full_dataset = CustomDataset(self.data_dir, transform=self.transform)
        df = full_dataset.df

        # split into train val test 0.96 0.02 0.02
        train, test = train_test_split(df, test_size=0.2)
        train, val = train_test_split(train, test_size=0.2 / (0.2 + 0.6))

        # find mean and std for normalization
        norm_path = "norm.csv"
        if not os.path.isfile(norm_path):
            from tqdm import tqdm

            print("normalization file not found computing mean and std...")

            full_dataset = CustomDataset(
                self.data_dir,
                transform=self.transform,
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
            for i, (x, _) in enumerate(tqdm(norm_dataloader)):
                with torch.no_grad():
                    mean.append(x.mean(dim=(0, 2, 3)))
                    squared_mean.append((x ** 2).mean(dim=(0, 2, 3)))
                    weights.append(float(x.shape[0]))

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

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def setup(self, stage=None):
        full_dataset = CustomDataset(self.data_dir, transform=self.transform)

        N = len(full_dataset)
        n_train = int(0.6 * N)
        n_val = int(0.2 * N)
        n_test = N - n_train - n_val

        train, val, test = torch.utils.data.random_split(
            dataset=full_dataset,
            lengths=[n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = train
            self.val = val

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = test


class SeqDataModule(BaseDataModule):
    def __init__(self, window, stride, **kwargs):
        super().__init__(**kwargs)

        self.window = window
        self.stride = stride

        self.transform = transforms.ToTensor()

    def prepare_data(self):
        # split dataset into train, val, test sets
        full_dataset = CustomDataset(self.data_dir, transform=self.transform)
        df = full_dataset.df

        train, test = train_test_split(df, test_size=0.2)
        train, val = train_test_split(train, test_size=0.2 / (0.2 + 0.6))

        # read mean and std for normalization from csv file if it exists
        # if not, compute mean and std and save to csv file
        norm_path = "norm.csv"
        if not os.path.isfile(norm_path):
            from tqdm import tqdm

            print("normalization file not found computing mean and std...")

            full_dataset = CustomDataset(
                self.data_dir,
                transform=transforms.ToTensor(),
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
            for i, (x, _) in enumerate(tqdm(norm_dataloader)):
                with torch.no_grad():
                    mean.append(x.mean(dim=(0, 2, 3)))
                    squared_mean.append((x ** 2).mean(dim=(0, 2, 3)))
                    weights.append(float(x.shape[0]))

            mean = torch.stack(mean, dim=-1)
            squared_mean = torch.stack(squared_mean, dim=-1)

            weights = torch.tensor(weights).unsqueeze(dim=-1)

            with torch.no_grad():
                mean = torch.mm(mean, weights) / N
                squared_mean = torch.mm(squared_mean, weights) / N
                std = torch.sqrt(squared_mean - mean ** 2)

            mean = mean.squeeze().numpy().tolist()
            std = std.squeeze().numpy().tolist()

            print("mean: {mean}, std: {std}")

            df = pd.DataFrame(dict(mean=mean, std=std))
            df.to_csv(norm_path)
        else:
            df = pd.read_csv(norm_path)
            mean = df["mean"].tolist()
            std = df["std"].tolist()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def setup(self, stage=None):
        full_dataset = SequentialDataset(
            root=self.data_dir,
            window=self.window,
            stride=self.stride,
            transform=self.transform,
        )

        N = len(full_dataset)
        n_train = int(0.6 * N)
        n_val = int(0.2 * N)
        n_test = N - n_train - n_val

        train, val, test = torch.utils.data.random_split(
            dataset=full_dataset,
            lengths=[n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = train
            self.val = val

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = test
