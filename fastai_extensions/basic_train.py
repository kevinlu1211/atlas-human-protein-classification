import functools
import sys
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass

sys.path.append("../fastai")

import torch

from fastai import *
import fastai

from pytorch_toolbox.utils import to_numpy


class Learner:
    """
    This class serves to create the boundary between the fastai library and the pytorch_toolbox. By creating this class,
    any breaking changes in the fastai Learner class would mean that only this class would need to be changed.
    """

    exposed_attributes = ["data", "model", "model_gradients", "unfreeze", "unfreeze_layer_groups", "recorder",
                          "freeze_layer_groups", "lr_find", "to_fp16", "mixup", "metrics", "callback", "callback_fns"
                          "predict_on_dl", "fit", "fit_one_cycle", "load_from_path"]

    @classmethod
    def create(cls, *args, **kwargs):
        learner = PytorchToolboxLearner(*args, **kwargs)
        return cls(learner)

    def __init__(self, learner):
        self._learner = learner
        self.set_exposed_attributes()
        self.phase = None

    def set_exposed_attributes(self):
        for name in dir(self._learner):
            if len(self.exposed_attributes) == 1 and self.exposed_attributes[0] == "ALL":
                setattr(self, name, getattr(self._learner, name))
            else:
                if name in self.exposed_attributes:
                    setattr(self, name, getattr(self._learner, name))


@dataclass
class PytorchToolboxLearner(fastai.Learner):
    def __init__(self, label_key='label', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_cycle = None
        self.phase = None
        self._remove_fastai_recorder()
        self._add_custom_recorder()
        self._add_phase_determiner(label_key)

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

    def _add_phase_determiner(self, label_key):
        self.callback_fns = [functools.partial(PhaseDeterminer, label_key=label_key)] + self.callback_fns

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


def to_fp16(learn: Learner, loss_scale: float = 512, flat_master: bool = False) -> Learner:
    from .callbacks import MixedPrecision
    learn.model = fastai.model2half(learn.model)
    learn.mp_cb = MixedPrecision(learn, loss_scale=loss_scale, flat_master=flat_master)
    learn.callbacks.append(learn.mp_cb)
    return learn


Learner.to_fp16 = to_fp16


class DeviceDataLoader(fastai.DeviceDataLoader):
    """
    This is subclasses because there are situations where the batch being returns may contain more than tensors e.g.
    maybe we also return the UID of the image. Hence the proc_batch function is overridden to provide this functionality
    """

    def proc_batch(self, b):
        input = fastai.to_device(b[0], self.device)
        output = {}
        for k, v in b[1].items():
            if isinstance(v, torch.Tensor):
                output[k] = fastai.to_device(v, None)
            else:
                output[k] = v
        return input, output

    def __iter__(self):
        for b in self.dl:
            yield self.proc_batch(b)

    @classmethod
    def create(cls, dataset: fastai.Dataset, bs: int = 64, shuffle: bool = False,
               device: torch.device = fastai.defaults.device,
               tfms: Collection[Callable] = None, num_workers: int = fastai.defaults.cpus,
               collate_fn: Callable = fastai.data_collate, **kwargs: Any):
        "Create DeviceDataLoader from `dataset` with `batch_size` and `shuffle`: processs using `num_workers`."
        return cls(DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, **kwargs),
                   device=device, tfms=tfms, collate_fn=collate_fn)

    @classmethod
    def create_with_initialized_dl(cls, dl, device, tfms, collate_fn):
        return cls(dl=dl, device=device, tfms=tfms, collate_fn=collate_fn)


class DataBunch(fastai.DataBunch):
    """
    This purpose of this subclass was to provide more customization on how to set the batch sizes for the val and test
    dataset, instead them being set @ twice the size of the train bs. This is because there are situations where we
    would want to train smaller images (maybe it speed up training, or for better generalization as for smaller images
    the higher features would have to be identitfied), but test on the full sized image.
    """
    "Bind `train_dl`,`valid_dl` and`test_dl` to `device`. tfms are DL tfms (normalize). `path` is for models."

    def __init__(self, train_dl: DataLoader, valid_dl: DataLoader, test_dl: Optional[DataLoader] = None,
                 device: torch.device = None, tfms: Optional[Collection[Callable]] = None, path: PathOrStr = '.',
                 collate_fn: Callable = data_collate):
        "Bind `train_dl`,`valid_dl` and`test_dl` to `device`. tfms are DL tfms (normalize). `path` is for models."
        super().__init__(train_dl, valid_dl, test_dl, device, tfms, path, collate_fn)
        self.train_dl = DeviceDataLoader.create_with_initialized_dl(train_dl, self.device, tfms, collate_fn)
        self.valid_dl = DeviceDataLoader.create_with_initialized_dl(valid_dl, self.device, tfms, collate_fn)
        self.test_dl = DeviceDataLoader.create_with_initialized_dl(test_dl, self.device, tfms,
                                                                   collate_fn) if test_dl else None

    @classmethod
    def create(cls, train_ds: Dataset, valid_ds: Dataset, test_ds: Dataset = None, path: PathOrStr = '.',
               train_bs: int = 64, val_bs: int = None, test_bs: int = None, sampler=None,
               num_workers: int = fastai.defaults.cpus, pin_memory: bool = False,
               tfms: Optional[Collection[Callable]] = None,
               device: torch.device = None,
               collate_fn: Callable = data_collate) -> 'DataBunch':
        "`DataBunch` factory. `bs` batch size, `ds_tfms` for `Dataset`, `tfms` for `DataLoader`."
        if val_bs is None: val_bs = train_bs * 2
        if test_bs is None: test_bs = train_bs * 2

        datasets = [train_ds, valid_ds]
        if test_ds is not None:
            datasets.append(test_ds)
        if sampler is None:
            train_dl = DataLoader(train_ds, train_bs, shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
                                  drop_last=True)
        else:
            train_dl = DataLoader(train_ds, train_bs, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
                                  drop_last=True)
        val_dl = DataLoader(valid_ds, val_bs, shuffle=False, num_workers=num_workers)
        test_dl = DataLoader(test_ds, test_bs, shuffle=False, num_workers=num_workers)
        dls = [train_dl, val_dl, test_dl]
        return cls(*dls, path=path, device=device, tfms=tfms, collate_fn=collate_fn)


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

    def on_batch_begin(self, train, epoch, **kwargs):
        super().on_batch_begin(train, **kwargs)
        self.phase = self.learn.phase
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


class Phase(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


def determine_phase(train, last_target, label_key='label'):
    if train:
        return Phase.TRAIN
    else:
        label = last_target.get(label_key)
        if label is not None:
            return Phase.VAL
        else:
            return Phase.TEST


class PhaseDeterminer(fastai.LearnerCallback):
    _order = -100

    def __init__(self, learn: Learner, label_key: str = 'label'):
        self.learn = learn
        self.label_key = label_key

    def on_batch_begin(self, train, last_target, **kwargs):
        phase = self._determine_phase(train, last_target)
        self.learn.phase = phase

    def _determine_phase(self, train, last_target):
        if train:
            return Phase.TRAIN
        else:
            label = last_target.get(self.label_key)
            if label is not None:
                return Phase.VAL
            else:
                return Phase.TEST
