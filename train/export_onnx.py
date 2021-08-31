import argparse

import torch
from litmodules import LitLightSteer

parser = argparse.ArgumentParser(description="test model.")
parser.add_argument(
    "checkpoint",
    help="checkpoint path",
)

arg = parser.parse_args()

IMGSIZE = (64, 48)
PATH = arg.checkpoint

filepath = "".join(arg.checkpoint.split(".")[:-1])
filepath += ".onnx"
input_sample = torch.randn(1, 3, *IMGSIZE)

model = LitLightSteer.load_from_checkpoint(checkpoint_path=PATH)
model.to_onnx(filepath, input_sample, export_params=True, opset_version=11)
