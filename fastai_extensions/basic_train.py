import sys
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass

sys.path.append("../fastai")

import torch

from fastai import *
import fastai

from pytorch_toolbox.utils import to_numpy


class Phase(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


def determine_phase(train, last_target, label_key="label"):
    if train:
        return Phase.TRAIN
    else:
        label = last_target.get(label_key)
        if label is not None:
            return Phase.VAL
        else:
            return Phase.TEST


class Learner:
    """
    This class serves to create the boundary between the fastai library and the pytorch_toolbox. By creating this class,
    any breaking changes in the fastai Learner class would mean that only this class would need to be changed.
    """
    EXPOSED_ATTRIBUTES_FOR_LEARNER = ["model"]

    @classmethod
    def create(cls, *args, **kwargs):
        learner = PytorchToolboxLearner(*args, **kwargs)
        return cls(learner)

    def __init__(self, learner):
        self.learner = learner

    def __getattr__(self, item):
        if item in self.EXPOSED_ATTRIBUTES_FOR_LEARNER:
            return getattr(self.learner, item)
        else:
            return getattr(self, item)

    def model_gradients(self):
        self.learner.model_gradients()

    def unfreeze(self):
        self.learner.unfreeze()

    def freeze_layer_groups(self, layer_groups_idx):
        self.learner.freeze_layer_groups(layer_groups_idx)

    def unfreeze_layer_groups(self, layer_groups_idx):
        self.learner.unfreeze_layer_groups(layer_groups_idx)

    def predict_on_dl(self, dl, callbacks, callback_fns, metrics):
        self.learner.predict_on_dl(dl, callbacks=callbacks, callback_fns=callback_fns, metrics=metrics)

    def fit(self, *args, **kwargs):
        self.learner.fit(*args, **kwargs)

    def fit_one_cycle(self, *args, **kwargs):
        self.learner.fit_one_cycle(*args, **kwargs)

    def load_from_path(self, path, device=None):
        self.learner.load_from_path(path, device)


@dataclass
class PytorchToolboxLearner(fastai.Learner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_cycle = None
        self._remove_fastai_recorder()
        self._add_custom_recorder()

    def _remove_fastai_recorder(self):
        callback_fns = []
        for cb_fn in self.callback_fns:
            try:
                cb_name = cb_fn.__name__
                if cb_name == "Recorder":
                    pass
                else:
                    callback_fns.append(cb_fn)
            except AttributeError:
                callback_fns.append(cb_fn)

        self.callback_fns = callback_fns

    def _add_custom_recorder(self):
        self.callback_fns = [Recorder] + self.callback_fns

    def model_gradients(self):
        for lg in self.layer_groups:
            for l in lg:
                print(l)
                for p in l.parameters():
                    print(p.shape)
                    print(p.requires_grad)

    def freeze_layer_groups(self, layer_group_idxs):
        if not is_listy(layer_group_idxs): layer_group_idxs = [layer_group_idxs]
        super().unfreeze()
        for i in layer_group_idxs:
            for l in self.layer_groups[i]:
                if not self.train_bn or not isinstance(l, bn_types):
                    requires_grad(l, False)

    def unfreeze_layer_groups(self, layer_group_idxs):
        if not is_listy(layer_group_idxs): layer_group_idxs = [layer_group_idxs]
        layer_group_idxs_to_freeze = list(set(list(range(len(self.layer_groups)))) - set(layer_group_idxs))
        self.freeze_layer_groups(layer_group_idxs_to_freeze)

    def load_from_path(self, path, device=None):
        if device is None: device = self.data.device
        self.model.load_state_dict(torch.load(path, map_location=device))

    def predict_on_dl(self, dl, pbar=None, callbacks=None, callback_fns=None, metrics=None):
        assert dl is not None
        metrics = ifnone(metrics, self.metrics)
        callbacks_fns = [cb(self) for cb in ifnone(callback_fns, [])]
        cb_handler = CallbackHandler(self.callbacks + ifnone(callbacks, []) + callbacks_fns, metrics)
        with torch.no_grad():
            self.model.eval()
            for xb, yb in progress_bar(dl, parent=pbar, leave=(pbar is not None)):
                if cb_handler: xb, yb = cb_handler.on_batch_begin(xb, yb, train=False)
                cb_handler = ifnone(cb_handler, CallbackHandler())
                if not is_listy(xb): xb = [xb]
                out = self.model(*xb)
                _ = cb_handler.on_loss_begin(out)

    def fit(self, *args, **kwargs):
        if self.n_cycle is None:
            self.n_cycle = 0
        else:
            self.n_cycle += 1
        super().fit(*args, **kwargs)


class Recorder(fastai.basic_train.Recorder):
    """
    A extended recorder which has the ability to record the the losses and metric per epoch, this is so that we can
    use the average value of the losses to determine whether a model is good, or if and when to do early
    stopping/reduce LR
    """
    _order = -10

    def __init__(self, learn: Learner):
        super().__init__(learn)
        self.loss_history = defaultdict(lambda: defaultdict(list))
        self.metric_history = defaultdict(lambda: defaultdict(list))
        self.phase = None

    @property
    def history(self):
        return {**self.loss_history, **self.metric_history}

    def on_batch_begin(self, train, epoch, last_target, **kwargs):
        super().on_batch_begin(train, **kwargs)
        self.phase = determine_phase(train, last_target)
        self.key = (self.phase.name, epoch)

    def _create_loss_values_for_batch_for_every_samples(self):
        per_sample_loss_values_for_current_batch = dict()
        for loss in self.learn.loss_func.losses:
            name = loss.__class__.__name__
            per_sample_loss = loss.per_sample_loss
            per_sample_loss_values_for_current_batch[f"{name}"] = per_sample_loss
        return per_sample_loss_values_for_current_batch

    def _update_loss_history(self, loss_for_current_batch):
        for name, loss_value in loss_for_current_batch.items():
            self.loss_history[self.key][name].extend(to_numpy(loss_value))

    def on_batch_end(self, **kwargs):
        super().on_batch_end(**kwargs)
        average_loss_for_current_batch = self._create_loss_values_for_batch_for_every_samples()
        self._update_loss_history(average_loss_for_current_batch)

    def on_epoch_end(self, epoch, num_batch, smooth_loss, last_metrics, **kwargs):
        super().on_epoch_end(epoch, num_batch, smooth_loss, last_metrics, **kwargs)
        if self.phase == Phase.VAL:
            metric_names = self.names[3:]
            for name, metric in zip(metric_names, self.metrics[0]):
                self.metric_history[self.key][name].append(metric.item())


def to_fp16(learn: Learner, loss_scale: float = 512, flat_master: bool = False) -> Learner:
    from .callbacks import MixedPrecision
    learn.model = fastai.model2half(learn.model)
    learn.mp_cb = MixedPrecision(learn, loss_scale=loss_scale, flat_master=flat_master)
    learn.callbacks.append(learn.mp_cb)
    return learn


Learner.to_fp16 = to_fp16
