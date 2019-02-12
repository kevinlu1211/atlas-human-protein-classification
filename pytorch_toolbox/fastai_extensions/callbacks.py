import sys
from functools import partial
import logging

sys.path.append("../fastai")

import torch

import pytorch_toolbox.fastai.fastai as fastai
from dataclasses import dataclass


@dataclass
class LabelExtractorCallback(fastai.Callback):
    label_key: str = 'label'

    def on_batch_begin(self, last_input, last_target, **kwargs):
        label = last_target.get(self.label_key)
        if label is not None:
            return last_input, last_target[self.label_key]
        else:
            return last_input, last_target



@dataclass
class FiveCropTTAPredictionCallback(fastai.Callback):
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


@dataclass
class MixedPrecision(fastai.callbacks.MixedPrecision):
    """
    Callback that handles mixed-precision training. This class is subclassed because in the fastai implemention the
    conversion from float32 -> float16 is appended to tfms list, which isn't used in our toolbox
    """

    def on_train_begin(self, **kwargs) -> None:
        # Get a copy of the model params in FP32
        self.model_params, self.master_params = fastai.callbacks.fp16.get_master(self.learn.layer_groups,
                                                                                 self.flat_master)
        # Changes the optimizer so that the optimization step is done in FP32.
        opt = self.learn.opt
        mom, wd, beta = opt.mom, opt.wd, opt.beta
        lrs = [lr for lr in self.learn.opt._lr for _ in range(2)]
        opt_params = [{'params': mp, 'lr': lr} for mp, lr in zip(self.master_params, lrs)]
        self.learn.opt.opt = self.learn.opt_func(opt_params)
        opt.mom, opt.wd, opt.beta = mom, wd, beta

    def on_batch_begin(self, last_input, last_target, **kwargs):
        return last_input.half(), last_target

    def on_train_end(self, **kwargs):
        return


callback_lookup = {
    "LabelExtractorCallback": LabelExtractorCallback,
    "FiveCropTTAPredictionCallback": FiveCropTTAPredictionCallback,
}
