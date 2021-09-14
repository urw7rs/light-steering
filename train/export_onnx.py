import argparse

import torch
from litmodules import LitSeqLightSteer
from torch_models import SeqModel


def main(args):
    input = torch.randn(1, 1, 3, *args.img_size)
    h0 = torch.randn(1, 1, 3)
    c0 = torch.randn(1, 1, 256)

    litmodel = LitSeqLightSteer.load_from_checkpoint(
        checkpoint_path=args.ckpt_path, model=SeqModel()
    )

    litmodel.to_onnx(
        args.onnx_path,
        input,
        export_params=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )


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
        default=[48, 64],
        help="input image size, .pt files need to be deleted",
    )

    args = parser.parse_args()

    main(args)
