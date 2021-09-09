import os
import torch
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# local files
from datasets import CustomDataset, cache_dataset


class POCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        img_size,
        batch_size=64,
        train_f="train.pt",
        val_f="val.pt",
        test_f="test.pt",
        **kwargs
    ):
        super().__init__()

        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_f = train_f
        self.val_f = val_f
        self.test_f = test_f

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.img_size),
            ]
        )

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

            # remove velocity data
            # df = df.drop(["vel"], axis=1)

            # split into train val test 0.96 0.02 0.02
            train, test = train_test_split(df, test_size=0.2)
            train, val = train_test_split(train, test_size=0.2 / (0.2 + 0.6))

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
            from tqdm import tqdm

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
            for i, (x, _) in enumerate(tqdm(norm_dataloader)):
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

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.img_size),
                transforms.Normalize(mean, std),
            ]
        )

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = cache_dataset(
                CustomDataset(
                    self.data_dir,
                    split="train",
                    transform=self.transform,
                ),
                f=self.train_f,
            )
            self.val = cache_dataset(
                CustomDataset(
                    self.data_dir,
                    split="val",
                    transform=self.transform,
                ),
                f=self.val_f,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = cache_dataset(
                CustomDataset(
                    self.data_dir,
                    split="test",
                    transform=self.transform,
                ),
                f=self.test_f,
            )

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
