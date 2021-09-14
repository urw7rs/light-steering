import pytorch_lightning as pl

from litmodules import LitLightSteer, LitSeqLightSteer
from datamodules import POCDataModule, SeqDataModule
from torch_models import Model, SeqModel

import argparse


def main(args):
    dict_args = vars(args)
    pl.seed_everything(42)

    trainer = pl.Trainer.from_argparse_args(args)

    if args.sequential:
        dm = SeqDataModule(**dict_args)

        model = SeqModel()

        litmodel = LitSeqLightSteer.load_from_checkpoint(
            checkpoint_path=args.ckpt_path, model=model
        )
    else:
        dm = POCDataModule(**dict_args)

        model = Model()

        litmodel = LitLightSteer.load_from_checkpoint(
            checkpoint_path=args.ckpt_path, model=model
        )

    trainer.test(litmodel, dm)


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
        "--sequential",
        default=False,
        action="store_true",
        help="sequential flag",
    )

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
