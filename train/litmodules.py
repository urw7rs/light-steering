import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

BATCH_SIZE = 64
ROOT = "/work/dataset"
IMGSIZE = (64, 48)
LR = 1e-3


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
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat[:, 0] = 1.2 * torch.sigmoid(y_hat[:, 0])
        y_hat[:, 1] = 0.7 * torch.tanh(y_hat[:, 1])
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
