import sys
import time
from collections import defaultdict
from functools import partial
from typing import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from pytorch_toolbox.fastai_extensions.basic_train import Phase

sys.path.append("../fastai")
import pytorch_toolbox.fastai.fastai as fastai


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


@dataclass
class TrackerCallback(fastai.LearnerCallback):
    "A `LearnerCallback` that keeps track of the best value in `monitor`."
    monitor: str = 'val_loss'
    mode: str = 'auto'

    def __post_init__(self):
        assert self.mode in ['auto', 'min', 'max'], "Please select a valid model to monitor"
        mode_dict = dict(min=np.less, max=np.greater)
        mode_dict['auto'] = np.less if 'loss' in self.monitor else np.greater
        self.operator = mode_dict[self.mode]

    def on_train_begin(self, **kwargs) -> None:
        self.best = float('inf') if self.operator == np.less else -float('inf')

    def get_monitor_value(self, epoch):
        prev_epoch = epoch - 1
        train_key = (Phase.TRAIN.name, prev_epoch)
        val_key = (Phase.VAL.name, prev_epoch)
        recorder = self.learn.recorder
        values = defaultdict(float)
        for loss_name, loss_values in recorder.loss_history[train_key].items():
            mean_loss = np.mean(loss_values)
            values[f"train_{fastai.camel2snake(loss_name)}"] = mean_loss
            values["train_loss"] += mean_loss
        for loss_name, loss_values in recorder.loss_history[val_key].items():
            mean_loss = np.mean(loss_values)
            values[f"val_{fastai.camel2snake(loss_name)}"] = mean_loss
            values["val_loss"] += mean_loss
        for metric_name, metric_values in recorder.metric_history[val_key].items():
            values[f"val_{fastai.camel2snake(metric_name)}"] = np.mean(metric_values)
        return values.get(self.monitor)


@dataclass
class SaveModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    every: str = 'improvement'
    name: str = 'best_model'

    # need to create a default value to get around the error TypeError: non-default argument 'save_path_creator' follows default argument
    save_path_creator: Callable = None

    def __post_init__(self):
        assert self.save_path_creator is not None
        if self.every not in ['improvement', 'epoch']:
            warn(f'SaveModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'
        self.save_path = self.save_path_creator()
        super().__post_init__()

    def on_epoch_end(self, epoch, **kwargs) -> None:
        if self.every == "epoch":
            self.learn.save(f'{self.name}_{epoch}')
        else:  # every="improvement"
            current = self.get_monitor_value(epoch)
            if current is not None and self.operator(current, self.best):
                self.best = current
                self.learn.save(f'{self.save_path / self.name}')

    def on_train_end(self, epoch, **kwargs):
        current = self.get_monitor_value(epoch)
        if current is None:
            return
        if self.every == "improvement":
            self.learn.load(f'{self.save_path / self.name}')


@dataclass
class ReduceLROnPlateauCallback(TrackerCallback):
    "A `TrackerCallback` that reduces learning rate when a metric has stopped improving."
    patience: int = 0
    factor: float = 0.2
    min_delta: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.operator == np.less:  self.min_delta *= -1

    def on_train_begin(self, **kwargs) -> None:
        self.wait, self.opt = 0, self.learn.opt
        super().on_train_begin(**kwargs)

    def on_epoch_end(self, epoch, **kwargs) -> None:
        current = self.get_monitor_value(epoch)
        if current is None: return
        if self.operator(current - self.min_delta, self.best):
            self.best, self.wait = current, 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.opt.lr *= self.factor
                self.wait = 0
                print(f'Epoch {epoch}: reducing lr to {self.opt.lr}')


@dataclass
class ReduceLROnEpochEndCallback(TrackerCallback):
    wait_duration: int = 10
    save_path_creator: Callable = None

    def __post_init__(self):
        assert self.save_path_creator is not None
        self.lr_history = []
        self.save_path = self.save_path_creator()
        super().__post_init__()

    def _wait_for_user_prompt_to_change_lr(self):
        print(f"Waiting for {self.wait_duration} seconds keyboard interrupt to change LR")
        time.sleep(self.wait_duration)

    def _user_input_prompt_for_new_lr(self):
        new_lr = input(
            "Please type in the new lr, if it is a list of lrs separate them by spaces,"
            " type n or no to continue training without changing lr\n")
        if new_lr.lower() in ["n", "no"]:
            return None
        new_lr = new_lr.split()
        new_lr = [float(lr) for lr in new_lr]
        return new_lr

    def _request_user_input_for_new_lr(self):
        current_lr = self.learn.opt.read_val('lr')
        print(f"The current LR is: {current_lr}")
        while True:
            try:
                new_lr = self._user_input_prompt_for_new_lr()
                if new_lr is None:
                    return
            except Exception as e:
                print(e)

    def on_epoch_end(self, epoch, **kwargs):
        current_lr = self.learn.opt.read_val('lr')
        if epoch == 0:
            self.lr_history.append(dict(epoch=epoch, lr=current_lr))
        try:
            self._wait_for_user_prompt_to_change_lr()
        except KeyboardInterrupt:
            self._request_user_input_for_new_lr()

    def on_train_end(self, **kwargs):
        lr_history_df = pd.DataFrame(self.lr_history)
        lr_history_df.to_csv(self.save_path / "lr_history.csv")


learner_callback_lookup = {
    "SaveModelCallback": SaveModelCallback,
    "ReduceLROnPlateauCallback": ReduceLROnPlateauCallback,
    "ReduceLROnEpochEndCallback": ReduceLROnEpochEndCallback
}
