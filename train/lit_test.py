import pytorch_lightning as pl

from litmodules import LitLightSteer
from datamodules import POCDataModule

import argparse

parser = argparse.ArgumentParser(description="test model.")
parser.add_argument(
    "checkpoint",
    help="checkpoint path",
)

arg = parser.parse_args()

PATH = arg.checkpoint
ROOT = "/work/dataset"
IMGSIZE = (64, 48)
LR = 1e-3

pl.seed_everything(42)

dm = POCDataModule(
    data_dir=ROOT,
    img_size=IMGSIZE,
)

model = LitLightSteer.load_from_checkpoint(checkpoint_path=PATH)

trainer = pl.Trainer(gpus=1, precision=16)
trainer.test(model, dm)
