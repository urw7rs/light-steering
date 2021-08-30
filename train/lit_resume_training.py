import argparse

import pytorch_lightning as pl
from lit_train import LitLightSteer, POCDataModule

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

from pytorch_lightning.loggers import TensorBoardLogger

parser = argparse.ArgumentParser(description="test model.")
parser.add_argument(
    "checkpointfile",
    help="checkpoint file path",
)

arg = parser.parse_args()

PATH = arg.checkpointfile
BATCH_SIZE = 64
ROOT = "/work/dataset"
IMGSIZE = (64, 48)
LR = 1e-3

dm = POCDataModule(
    data_dir=ROOT,
    img_size=IMGSIZE,
    augmentation=None  # [
    # transforms.ColorJitter(
    #    brightness=0.8, contrast=0.5, saturation=0.5, hue=0.5
    # )
    # ],
)

model = LitLightSteer.load_from_checkpoint(checkpoint_path=PATH)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoint",
    filename="pocmodel-{epoch:02d}-{val_loss:.8f}",
    save_top_k=3,
    mode="min",
)
lr_monitor = LearningRateMonitor(logging_interval="step")

logger = TensorBoardLogger("tb_logs", name="nvidia-convnet")

trainer = pl.Trainer(
    gpus=1,
    precision=16,
    callbacks=[checkpoint_callback, lr_monitor],
    max_epochs=1000,
    default_root_dir="checkpoint",
    logger=logger,
    resume_from_checkpoint=arg.checkpointfile,
)
trainer.fit(model, dm)
