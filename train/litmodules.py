import torch
import torch.nn.functional as F

import pytorch_lightning as pl


class LitLightSteer(pl.LightningModule):
    def __init__(self, model, learning_rate, **kwargs):
        super().__init__()
        self.save_hyperparameters("learning_rate")
        # self.save_hyperparameters()
        self.model = model

    def forward(self, x):
        y = self.model(x)
        return y

    def match_scale(self, y_hat):
        vel = 1.2 * torch.sigmoid(y_hat[:, 0])
        ang = 0.7 * torch.tanh(y_hat[:, 1])
        torq = 1.4 * torch.tanh(y_hat[:, 2])
        return vel, ang, torq

    def _compute_loss(self, vel, ang, torq, y):
        vel_loss = F.mse_loss(vel, y[:, 0])
        ang_loss = F.mse_loss(ang, y[:, 1])
        # delta_vel is stored in y[:, 2]
        torq_loss = F.mse_loss(torq, y[:, 3])
        return vel_loss, ang_loss, torq_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        vel, ang, torq = self.match_scale(y_hat)
        vel_loss, ang_loss, torq_loss = self._compute_loss(vel, ang, torq, y)
        loss = vel_loss + ang_loss + torq_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("vel_loss", vel_loss, prog_bar=True)
        self.log("ang_loss", ang_loss, prog_bar=True)
        self.log("torq_loss", torq_loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        vel, ang, torq = self.match_scale(y_hat)
        vel_loss, ang_loss, torq_loss = self._compute_loss(vel, ang, torq, y)
        loss = vel_loss + ang_loss + torq_loss

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_vel_loss", vel_loss, prog_bar=True)
        self.log("val_ang_loss", ang_loss, prog_bar=True)
        self.log("val_torq_loss", ang_loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        vel, ang, torq = self.match_scale(y_hat)
        vel_loss, ang_loss, torq_loss = self._compute_loss(vel, ang, torq, y)
        loss = vel_loss + ang_loss + torq_loss

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_vel_loss", vel_loss, prog_bar=True)
        self.log("test_ang_loss", ang_loss, prog_bar=True)
        self.log("test_torq_loss", ang_loss, prog_bar=True)

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


class LitSeqLightSteer(pl.LightningModule):
    def __init__(self, model, learning_rate, **kwargs):
        super().__init__()
        self.save_hyperparameters("learning_rate")
        # self.save_hyperparameters()
        self.model = model

        # self.truncated_bptt_steps = 2

    def forward(self, x, hiddens=None):
        y, hiddens = self.model(x, hiddens)
        return y, hiddens

    def match_scale(self, y_hat):
        vel = 1.2 * torch.sigmoid(y_hat[:, :, 0])
        ang = 0.7 * torch.tanh(y_hat[:, :, 1])
        torq = 1.4 * torch.tanh(y_hat[:, :, 2])
        return vel, ang, torq

    def _compute_loss(self, vel, ang, torq, y):
        vel_loss = F.mse_loss(vel, y[:, :, 0])
        ang_loss = F.mse_loss(ang, y[:, :, 1])
        # delta_vel is stored in y[:, 2]
        torq_loss = F.mse_loss(torq, y[:, :, 3])
        return vel_loss, ang_loss, torq_loss

    def training_step(self, batch, batch_idx, hiddens=None):
        x, y = batch
        y_hat, hiddens = self.model(x, hiddens)

        vel, ang, torq = self.match_scale(y_hat)
        vel_loss, ang_loss, torq_loss = self._compute_loss(vel, ang, torq, y)
        loss = vel_loss + ang_loss + torq_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("vel_loss", vel_loss, prog_bar=True)
        self.log("ang_loss", ang_loss, prog_bar=True)
        self.log("torq_loss", torq_loss, prog_bar=True)

        return {"loss": loss, "hiddens": hiddens}

    def validation_step(self, batch, batch_idx, hiddens=None):
        x, y = batch
        y_hat, hiddens = self.model(x, hiddens)

        vel, ang, torq = self.match_scale(y_hat)
        vel_loss, ang_loss, torq_loss = self._compute_loss(vel, ang, torq, y)
        loss = vel_loss + ang_loss + torq_loss

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_vel_loss", vel_loss, prog_bar=True)
        self.log("val_ang_loss", ang_loss, prog_bar=True)
        self.log("val_torq_loss", torq_loss, prog_bar=True)

        return {"loss": loss, "hiddens": hiddens}

    def test_step(self, batch, batch_idx, hiddens=None):
        x, y = batch
        y_hat, hiddens = self.model(x, hiddens)

        vel, ang, torq = self.match_scale(y_hat)
        vel_loss, ang_loss, torq_loss = self._compute_loss(vel, ang, torq, y)
        loss = vel_loss + ang_loss + torq_loss

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_vel_loss", vel_loss, prog_bar=True)
        self.log("test_ang_loss", ang_loss, prog_bar=True)
        self.log("test_torq_loss", torq_loss, prog_bar=True)

        return {"loss": loss, "hiddens": hiddens}

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
