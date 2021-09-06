import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models


class LitLightSteer(pl.LightningModule):
    def __init__(self, learning_rate, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        resnet18 = models.resnet18()
        num_ftrs = resnet18.fc.in_features
        resnet18.fc = nn.Linear(num_ftrs, 2)

        self.model = resnet18

    def forward(self, x):
        y = self.model(x)
        return y

    def _predict(self, x):
        y_hat = self.model(x)
        vel = 1.2 * torch.sigmoid(y_hat[:, 0])
        ang = 0.7 * torch.tanh(y_hat[:, 1])
        return vel, ang

    def _compute_loss(self, vel, ang, y):
        vel_loss = F.mse_loss(vel, y[:, 0])
        ang_loss = F.mse_loss(ang, y[:, 1])
        loss = vel_loss + ang_loss
        return loss, vel_loss, ang_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        vel, ang = self._predict(x)
        loss, vel_loss, ang_loss = self._compute_loss(vel, ang, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("vel_loss", vel_loss, prog_bar=True)
        self.log("ang_loss", ang_loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        vel, ang = self._predict(x)
        loss, vel_loss, ang_loss = self._compute_loss(vel, ang, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("vel_loss", vel_loss, prog_bar=True)
        self.log("ang_loss", ang_loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        vel, ang = self._predict(x)
        loss, vel_loss, ang_loss = self._compute_loss(vel, ang, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("vel_loss", vel_loss, prog_bar=True)
        self.log("ang_loss", ang_loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LightSteer")
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parent_parser
