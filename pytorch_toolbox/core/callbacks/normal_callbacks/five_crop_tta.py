from functools import partial
from dataclasses import dataclass

import torch

from core.utils import Phase
from core.callbacks import Callback


@dataclass
class FiveCropTTACallback(Callback):
    _order = -20

    aggregate_fns = {
        "MAX": partial(torch.max, dim=1),
        "MEAN": partial(torch.mean, dim=1)
    }

    def __init__(self, aggregate_mode="MAX"):
        assert aggregate_mode in self.aggregate_fns.keys()
        self.aggregate_mode = aggregate_mode
        super().__init__()


    def on_batch_begin(self, phase, last_input, last_target, **kwargs):
        if phase is not Phase.TRAIN:
            self.last_input_shape = last_input.shape
            batch_size, n_crops, *other_dims = self.last_input_shape
            last_input_flattened = last_input.view(-1, *other_dims)
            return last_input_flattened, last_target
        else:
            return last_input, last_target

    def on_loss_begin(self, phase, last_output, **kwargs):
        if phase is not Phase.TRAIN:
            batch_size, n_crops, *other_dims = self.last_input_shape
            last_output_reshaped = last_output.view(batch_size, n_crops, -1)
            aggregated_last_output, _ = self.aggregate_fns[self.aggregate_mode](last_output_reshaped)
            return aggregated_last_output
        else:
            return last_output
