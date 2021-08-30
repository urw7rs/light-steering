import os

import pytorch_lightning as pl

from lit_train import LitLightSteer, POCDataModule

PATH = os.path.join("checkpoint", "pocmodel-epoch=04-val_loss=0.0712.ckpt")

ROOT = "/work/dataset"
IMGSIZE = (64, 48)
LR = 1e-3

pl.seed_everything(42)

dm = POCDataModule(
    data_dir=ROOT,
    img_size=IMGSIZE,
)

model = LitLightSteer(LR)
model.load_from_checkpoint(checkpoint_path=PATH)

trainer = pl.Trainer(gpus=1, precision=16)
trainer.test(model, dm)
