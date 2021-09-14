import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datamodules import POCDataModule, SeqDataModule
from litmodules import LitLightSteer, LitSeqLightSteer
from torch_models import Model, SeqModel


def main(args):
    dict_args = vars(args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        auto_insert_metric_name=True,
        save_last=True,
    )

    logger = TensorBoardLogger("tb_logs", name=args.name)
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback], logger=logger
    )

    if args.sequential:
        dm = SeqDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            window=args.window,
            stride=args.stride,
        )

        model = SeqModel()

        litmodel = LitSeqLightSteer(model=model, **dict_args)
    else:
        dm = POCDataModule(
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            train_f=args.train_f,
            val_f=args.val_f,
            test_f=args.test_f,
        )

        model = Model()

        litmodel = LitLightSteer(model=model, **dict_args)

    trainer.fit(litmodel, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # add PROGRAM level args
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="best",
        help="checkpointfile path, inside tb_logs/name/version_/checkpoints",
    )
    parser.add_argument(
        "--data_dir", type=str, default="/work/dataset", help="dataset path"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size, learning_rate doesn't change with batch size",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[48, 64],
        help="input image size, .pt files need to be deleted",
    )
    parser.add_argument(
        "--name", type=str, default="default", help="tensorboard logger name"
    )

    parser.add_argument(
        "--sequential",
        default=False,
        action="store_true",
        help="sequential flag",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=25,
        help="sequence length",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="sequence stride",
    )

    parser.add_argument(
        "--train_f",
        type=str,
        default="train.pt",
        help="dataset cache file path",
    )
    parser.add_argument(
        "--val_f",
        type=str,
        default="val.pt",
        help="dataset cache file path",
    )
    parser.add_argument(
        "--test_f",
        type=str,
        default="test.pt",
        help="dataset cache file path",
    )

    # add model specific args
    parser = LitLightSteer.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
