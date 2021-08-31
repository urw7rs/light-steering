import argparse

import torch
from export_train import LitLightSteer


class ExportModule(LitLightSteer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        return y_hat


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

model = LitLightSteer.load_from_checkpoint(checkpoint_path=PATH)

input_sample = torch.randn(1, 3, 64, 48)
model.to_onnx(
    filepath,
    input_sample,
    export_params=True,
    do_constant_folding=True,
    opset_version=11,
    verbose=True,
)
