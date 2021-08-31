import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# logger
from pytorch_lightning.loggers import TensorBoardLogger

import argparse

from datamodules import POCDataModule
from litmodules import LitLightSteer

BATCH_SIZE = 64
ROOT = "/work/dataset"
IMGSIZE = (64, 48)
LR = 1e-3


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
