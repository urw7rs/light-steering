import pytorch_lightning as pl

from litmodules import LitLightSteer
from datamodules import POCDataModule
from torch_models import Model

import argparse


def main(args):
    dict_args = vars(args)
    pl.seed_everything(42)

    dm = POCDataModule(**dict_args)

    trainer = pl.Trainer.from_argparse_args(args)
    model = LitLightSteer.load_from_checkpoint(
        checkpoint_path=args.ckpt_path, model=Model()
    )

    trainer.test(model, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test model.")
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
        default=[64, 48],
        help="input image size, .pt files need to be deleted",
    )

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
