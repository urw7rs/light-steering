import argparse

import torch
from litmodules import LitLightSteer


def main(args):
    input_sample = torch.randn(1, 3, *args.img_size)

    model = LitLightSteer.load_from_checkpoint(checkpoint_path=args.ckpt_path)
    model.to_onnx(args.onnx_path, input_sample, export_params=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="checkpoint file path",
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="model.onnx",
        help="onnx file path",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[64, 48],
        help="input image size, .pt files need to be deleted",
    )

    args = parser.parse_args()

    main(args)
