from functools import partial
from dataclasses import dataclass

import torch

from pytorch_toolbox.training.callbacks.core import Callback


@dataclass
class FiveCropTTACallback(Callback):
    _order = -20

    aggregate_fns = {
        "MAX": partial(torch.max, dim=1)
    }

    def __init__(self, aggregate_mode="MAX"):
        assert aggregate_mode in self.aggregate_fns.keys()
        self.aggregate_mode = aggregate_mode
        super().__init__()

    def on_batch_begin(self, train, last_input, last_target, **kwargs):
        # B, n_crops=5, C, H, W
        if not train:
            self.last_input_shape = last_input.shape
            *_, c, h, w = self.last_input_shape
            last_input_flattened = last_input.view(-1, c, h, w)
            return last_input_flattened, last_target
        else:
            return last_input, last_target

    def on_loss_begin(self, train, last_output, **kwargs):
        if not train:
            b, n_crops, *_ = self.last_input_shape
            last_output_reshaped = last_output.view(b, n_crops, -1)
            aggregated_last_output, _ = self.aggregate_fns[self.aggregate_mode](last_output_reshaped)
            return aggregated_last_output
        else:
            return last_output
