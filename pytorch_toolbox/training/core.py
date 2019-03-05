import os
import re
from enum import Enum
from types import SimpleNamespace
from typing import Optional, Callable, Any, Union, Collection, Tuple, List, Dict
from numbers import Number
from functools import partial
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

from fastprogress.fastprogress import MasterBar, ProgressBar
from pytorch_toolbox.utils import listify, to_numpy


def num_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


Tensor = torch.Tensor
Tensors = Union[Tensor, Collection['Tensors']]
ModuleList = Collection[nn.Module]
PBar = Union[MasterBar, ProgressBar]
TensorOrNumber = Union[Tensor, Number]
TensorOrNumberList = Collection[TensorOrNumber]

AffineMatrix = Tensor
BoolOrTensor = Union[bool, Tensor]
FloatOrTensor = Union[float, Tensor]
IntOrTensor = Union[int, Tensor]
LambdaFunc = Callable[[Tensor], Tensor]
LayerFunc = Callable[[nn.Module], None]
ModuleList = Collection[nn.Module]
NPArray = np.ndarray
OptOptimizer = Optional[optim.Optimizer]
ParamList = Collection[nn.Parameter]
SplitFunc = Callable[[nn.Module], List[nn.Module]]
SplitFuncOrIdxList = Union[Callable, Collection[ModuleList]]
TensorOrNumber = Union[Tensor, Number]
TensorOrNumberList = Collection[TensorOrNumber]
TensorImage = Tensor
TensorImageSize = Tuple[int, int, int]
Tensors = Union[Tensor, Collection['Tensors']]
Weights = Dict[str, Tensor]

HookFunc = Callable[[nn.Module, Tensors, Tensors], Any]
LogitTensorImage = TensorImage
MetricFunc = Callable[[Tensor, Tensor], TensorOrNumber]
MetricFuncList = Collection[MetricFunc]
MetricsList = Collection[TensorOrNumber]
OptMetrics = Optional[MetricsList]
OptSplitFunc = Optional[SplitFunc]

AdamW = partial(Adam, betas=(0.9, 0.99))
default_collate = torch.utils.data.dataloader.default_collate
default_lr = slice(3e-3)
default_wd = 1e-2
Floats = Union[float, Collection[float]]

_default_cpus = min(16, num_cpus())
_default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
defaults = SimpleNamespace(device=_default_device, cpus=_default_cpus)


def if_none(a: Any, b: Any) -> Any:
    return b if a is None else a


def is_listy(x: Any) -> bool:
    return isinstance(x, (tuple, list))


def is_tuple(x: Any) -> bool:
    return isinstance(x, tuple)


def to_device(t: Tensors, device: torch.device):
    device = if_none(device, defaults.device)
    if is_listy(t):
        return [to_device(o, device) for o in t]
    return t.to(device)


def range_of(x):
    return list(range(len(x)))


def flatten_model(m: nn.Module):
    return sum(map(flatten_model, m.children()), []) if num_children(m) else [m]


def num_children(m: nn.Module) -> int:
    return len(children(m))


def children(m: nn.Module) -> ModuleList:
    return list(m.children())


_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')


def camel2snake(name: str) -> str:
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()


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


class DeviceDataLoader:

    @classmethod
    def create(cls, dataset: Dataset, batch_size: int = 64, shuffle: bool = False,
               device: torch.device = defaults.device,
               num_workers: int = defaults.cpus,
               collate_fn: Callable = default_collate, **kwargs: Any):
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)
        return cls(dl, device=device, collate_fn=collate_fn)

    @property
    def batch_size(self):
        return self.dl.batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self.dl.batch_size = batch_size

    @property
    def num_workers(self):
        return self.dl.num_workers

    @num_workers.setter
    def num_workers(self, num_workers):
        self.dl.num_workers = num_workers

    def __init__(self, dl: DataLoader, device: torch.device,
                 collate_fn: Callable = default_collate):
        self.dl = dl
        self.device = device
        self.dl.collate_fn = collate_fn

    def proc_batch(self, batch):
        input_ = to_device(batch[0], self.device)
        output = {}
        for k, v in batch[1].items():
            if isinstance(v, torch.Tensor):
                output[k] = to_device(v, self.device)
            else:
                output[k] = v
        return input_, output

    def __iter__(self):
        for b in self.dl:
            yield self.proc_batch(b)


class DataBunch:
    def __init__(self, train_dl: DataLoader, valid_dl: DataLoader, test_dl: Optional[DataLoader] = None,
                 device: torch.device = None, collate_fn: Callable = default_collate):
        self.device = defaults.device if device is None else device
        self.train_dl = DeviceDataLoader(train_dl, self.device, collate_fn)
        self.valid_dl = DeviceDataLoader(valid_dl, self.device, collate_fn)
        self.test_dl = DeviceDataLoader(test_dl, self.device, collate_fn) if test_dl else None

    @classmethod
    def create(cls, train_ds: Dataset, valid_ds: Dataset, test_ds: Dataset = None,
               train_bs: int = 64, val_bs: int = None, test_bs: int = None, sampler=None,
               num_workers: int = defaults.cpus, pin_memory: bool = False,
               device: torch.device = None,
               collate_fn: Callable = default_collate) -> 'DataBunch':

        if val_bs is None:
            val_bs = train_bs * 2
        if test_bs is None:
            test_bs = train_bs * 2
        train_dl = DataLoader(train_ds, train_bs, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
                              drop_last=True)
        val_dl = DataLoader(valid_ds, val_bs, shuffle=False, num_workers=num_workers)
        test_dl = DataLoader(test_ds, test_bs, shuffle=False, num_workers=num_workers)
        dls = [train_dl, val_dl, test_dl]
        return cls(*dls, device=device, collate_fn=collate_fn)


class Learner:
    def __init__(self, data: DataBunch, model: nn.Module, loss_func: Callable, opt_func: Callable = AdamW,
                 metrics: Collection[Callable] = None, true_weight_decay: bool = True,
                 batch_norm_weight_decay: bool = True, weight_decay: Floats = 1e-2, train_bn: bool = True,
                 model_dir: str = "model", callback_fns: Collection[Callable] = None,
                 callbacks: Collection[Callable] = [], layer_groups: Collection[nn.Module] = None):
        self.data = data
        self.model = model.to(self.data.device)
        self.loss_func = loss_func
        self.opt_func = opt_func
        self.true_weight_decay = true_weight_decay
        self.batch_norm_weight_decay = batch_norm_weight_decay
        self.weight_decay = weight_decay
        self.train_bn = train_bn
        self.model_dir = model_dir
        self.metrics = listify(metrics)
        self.callbacks = listify(callbacks)
        self.callback_fns = [Recorder] + listify(callback_fns)

        if not layer_groups:
            self.layer_groups = [nn.Sequential(*flatten_model(self.model))]

    def fit(self, learning_rate: Floats, weight_decay: Floats = 0.):
        if weight_decay is not None:
            self.weight_decay = weight_decay

    def create_opt(self, learning_rate: Floats, weight_decay: Floats = 0.) -> None:
        self.optimizer = OptimizerWrapper.create(self.opt_func, lr)


class Callback:
    "Base class for callbacks that want to record values, dynamically change learner params, etc."
    _order = 0

    def on_train_begin(self, **kwargs: Any) -> None:
        "To initialize constants in the callback."
        pass

    def on_epoch_begin(self, **kwargs: Any) -> None:
        "At the beginning of each epoch."
        pass

    def on_batch_begin(self, **kwargs: Any) -> None:
        "Set HP before the step is done. Returns xb, yb (which can allow us to modify the input at that step if needed)."
        pass

    def on_loss_begin(self, **kwargs: Any) -> None:
        "Called after forward pass but before loss has been computed. Returns the output (which can allow us to modify it)."
        pass

    def on_backward_begin(self, **kwargs: Any) -> None:
        """Called after the forward pass and the loss has been computed, but before backprop.
           Returns the loss (which can allow us to modify it, for instance for reg functions)"""
        pass

    def on_backward_end(self, **kwargs: Any) -> None:
        "Called after backprop but before optimizer step. Useful for true weight decay in AdamW."
        pass

    def on_step_end(self, **kwargs: Any) -> None:
        "Called after the step of the optimizer but before the gradients are zeroed."
        pass

    def on_batch_end(self, **kwargs: Any) -> None:
        "Called at the end of the batch."
        pass

    def on_epoch_end(self, **kwargs: Any) -> bool:
        "Called at the end of an epoch."
        return False

    def on_train_end(self, **kwargs: Any) -> None:
        "Useful for cleaning up things and saving files/models."
        pass


class LearnerCallback(Callback):
    "Base class for creating callbacks for a `Learner`."

    def __init__(self, learn):
        self.learn = learn
        if self.cb_name:
            setattr(self.learn, self.cb_name, self)

    @property
    def cb_name(self): return camel2snake(self.__class__.__name__)


class BaseRecorder(LearnerCallback):
    "A `LearnerCallback` that records epoch, loss, opt and metric data during training."
    _order = -10

    def __init__(self, learn: Learner):
        super().__init__(learn)
        self.opt = self.learn.opt
        self.train_dl = self.learn.data.train_dl

    def on_train_begin(self, pbar: PBar, metrics_names: Collection[str], **kwargs: Any) -> None:
        "Initialize recording status at beginning of training."
        self.pbar = pbar
        self.names = ['epoch', 'train_loss', 'valid_loss'] + metrics_names
        if hasattr(self, '_added_met_names'): self.names += self._added_met_names
        self.pbar.write('  '.join(self.names), table=True)
        self.losses, self.val_losses, self.lrs, self.moms, self.metrics, self.nb_batches = [], [], [], [], [], []

    def on_batch_begin(self, train, **kwargs: Any) -> None:
        "Record learning rate and momentum at beginning of batch."
        if train:
            self.lrs.append(self.opt.lr)
            self.moms.append(self.opt.mom)

    def on_backward_begin(self, smooth_loss: Tensor, **kwargs: Any) -> None:
        "Record the loss before any other callback has a chance to modify it."
        self.losses.append(smooth_loss)
        if self.pbar is not None and hasattr(self.pbar, 'child'):
            self.pbar.child.comment = f'{smooth_loss:.4f}'

    def on_epoch_end(self, epoch: int, num_batch: int, smooth_loss: Tensor,
                     last_metrics=MetricsList, **kwargs: Any) -> bool:
        "Save epoch info: num_batch, smooth_loss, metrics."
        self.nb_batches.append(num_batch)
        if last_metrics is not None:
            self.val_losses.append(last_metrics[0])
            if hasattr(self, '_added_mets'): last_metrics += self._added_mets
            if len(last_metrics) > 1: self.metrics.append(last_metrics[1:])
            self.format_stats([epoch, smooth_loss] + last_metrics)
        else:
            self.format_stats([epoch, smooth_loss])
        return False

    def format_stats(self, stats: TensorOrNumberList) -> None:
        "Format stats before printing."
        str_stats = []
        for name, stat in zip(self.names, stats):
            t = str(stat) if isinstance(stat, int) else f'{stat:.6f}'
            t += ' ' * (len(name) - len(t))
            str_stats.append(t)
        self.pbar.write('  '.join(str_stats), table=True)

    def add_metrics(self, metrics):
        self._added_mets = metrics

    def add_metric_names(self, names):
        self._added_met_names = names

    def plot_lr(self, show_moms=False) -> None:
        "Plot learning rate, `show_moms` to include momentum."
        iterations = range_of(self.lrs)
        if show_moms:
            _, axs = plt.subplots(1, 2, figsize=(12, 4))
            axs[0].plot(iterations, self.lrs)
            axs[1].plot(iterations, self.moms)
        else:
            plt.plot(iterations, self.lrs)

    def plot(self, skip_start: int = 10, skip_end: int = 5) -> None:
        "Plot learning rate and losses, trimmed between `skip_start` and `skip_end`."
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]
        _, ax = plt.subplots(1, 1)
        ax.plot(lrs, losses)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate")
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))

    def plot_losses(self) -> None:
        "Plot training and validation losses."
        _, ax = plt.subplots(1, 1)
        iterations = range_of(self.losses)
        ax.plot(iterations, self.losses)
        val_iter = self.nb_batches
        val_iter = np.cumsum(val_iter)
        ax.plot(val_iter, self.val_losses)

    def plot_metrics(self) -> None:
        "Plot metrics collected during training."
        assert len(self.metrics) != 0, "There are no metrics to plot."
        _, axes = plt.subplots(len(self.metrics[0]), 1, figsize=(6, 4 * len(self.metrics[0])))
        val_iter = self.nb_batches
        val_iter = np.cumsum(val_iter)
        axes = axes.flatten() if len(self.metrics[0]) != 1 else [axes]
        for i, ax in enumerate(axes):
            values = [met[i] for met in self.metrics]
            ax.plot(val_iter, values)


class Recorder(BaseRecorder):
    """A extended recorder which has the ability to record the the losses and metric per epoch,
    this is so that we can use the average value of the losses to determine whether a model is good,
     or if and when to do early stopping/reduce LR"""
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


class OptimWrapper:
    "Basic wrapper around an optimizer to simplify HP changes."

    def __init__(self, opt: optim.Optimizer, wd: Floats = 0., true_wd: bool = False, bn_wd: bool = True):
        self.opt, self.true_wd, self.bn_wd = opt, true_wd, bn_wd
        self.opt_keys = list(self.opt.param_groups[0].keys())
        self.opt_keys.remove('params')
        self.read_defaults()
        self.wd = wd

    @classmethod
    def create(cls, opt_func: Union[type, Callable], lr: Union[float, Tuple, List],
               layer_groups: ModuleList, **kwargs: Any) -> optim.Optimizer:
        "Create an optim.Optimizer from `opt_func` with `lr`. Set lr on `layer_groups`."
        split_groups = split_bn_bias(layer_groups)
        opt = opt_func([{'params': trainable_params(l), 'lr': 0} for l in split_groups])
        opt = cls(opt, **kwargs)
        opt.lr = listify(lr, layer_groups)
        return opt

    def __repr__(self) -> str:
        return f'OptimWrapper over {repr(self.opt)}.\nTrue weight decay: {self.true_wd}'

    # Pytorch optimizer methods
    def step(self) -> None:
        "Set weight decay and step optimizer."
        # weight decay outside of optimizer step (AdamW)
        if self.true_wd:
            for lr, wd, pg1, pg2 in zip(self._lr, self._wd, self.opt.param_groups[::2], self.opt.param_groups[1::2]):
                for p in pg1['params']: p.data.mul_(1 - wd * lr)
                if self.bn_wd:
                    for p in pg2['params']: p.data.mul_(1 - wd * lr)
            self.set_val('weight_decay', listify(0, self._wd))
        self.opt.step()

    def zero_grad(self) -> None:
        "Clear optimizer gradients."
        self.opt.zero_grad()

    # Hyperparameters as properties
    @property
    def lr(self) -> float:
        "Get learning rate."
        return self._lr[-1]

    @lr.setter
    def lr(self, val: float) -> None:
        "Set learning rate."
        self._lr = self.set_val('lr', listify(val, self._lr))

    @property
    def mom(self) -> float:
        "Get momentum."
        return self._mom[-1]

    @mom.setter
    def mom(self, val: float) -> None:
        "Set momentum."
        if 'momentum' in self.opt_keys:
            self.set_val('momentum', listify(val, self._mom))
        elif 'betas' in self.opt_keys:
            self.set_val('betas', (listify(val, self._mom), self._beta))
        self._mom = listify(val, self._mom)

    @property
    def beta(self) -> float:
        "Get beta (or alpha as makes sense for given optimizer)."
        return None if self._beta is None else self._beta[-1]

    @beta.setter
    def beta(self, val: float) -> None:
        "Set beta (or alpha as makes sense for given optimizer)."
        if val is None: return
        if 'betas' in self.opt_keys:
            self.set_val('betas', (self._mom, listify(val, self._beta)))
        elif 'alpha' in self.opt_keys:
            self.set_val('alpha', listify(val, self._beta))
        self._beta = listify(val, self._beta)

    @property
    def wd(self) -> float:
        "Get weight decay."
        return self._wd[-1]

    @wd.setter
    def wd(self, val: float) -> None:
        "Set weight decay."
        if not self.true_wd: self.set_val('weight_decay', listify(val, self._wd), bn_groups=self.bn_wd)
        self._wd = listify(val, self._wd)

    # Helper functions
    def read_defaults(self) -> None:
        "Read the values inside the optimizer for the hyper-parameters."
        self._beta = None
        if 'lr' in self.opt_keys: self._lr = self.read_val('lr')
        if 'momentum' in self.opt_keys: self._mom = self.read_val('momentum')
        if 'alpha' in self.opt_keys: self._beta = self.read_val('alpha')
        if 'betas' in self.opt_keys: self._mom, self._beta = self.read_val('betas')
        if 'weight_decay' in self.opt_keys: self._wd = self.read_val('weight_decay')

    def set_val(self, key: str, val: Any, bn_groups: bool = True) -> Any:
        "Set the values inside the optimizer dictionary at the key."
        if is_tuple(val): val = [(v1, v2) for v1, v2 in zip(*val)]
        for v, pg1, pg2 in zip(val, self.opt.param_groups[::2], self.opt.param_groups[1::2]):
            pg1[key] = v
            if bn_groups: pg2[key] = v
        return val

    def read_val(self, key: str) -> Union[List[float], Tuple[List[float], List[float]]]:
        "Read a hyperparameter key in the optimizer dictionary."
        val = [pg[key] for pg in self.opt.param_groups[::2]]
        if is_tuple(val[0]): val = [o[0] for o in val], [o[1] for o in val]
        return val


class Callback():
    "Base class for callbacks that want to record values, dynamically change learner params, etc."
    _order = 0

    def on_train_begin(self, **kwargs: Any) -> None:
        "To initialize constants in the callback."
        pass

    def on_epoch_begin(self, **kwargs: Any) -> None:
        "At the beginning of each epoch."
        pass

    def on_batch_begin(self, **kwargs: Any) -> None:
        "Set HP before the step is done. Returns xb, yb (which can allow us to modify the input at that step if needed)."
        pass

    def on_loss_begin(self, **kwargs: Any) -> None:
        "Called after forward pass but before loss has been computed. Returns the output (which can allow us to modify it)."
        pass

    def on_backward_begin(self, **kwargs: Any) -> None:
        """Called after the forward pass and the loss has been computed, but before backprop.
           Returns the loss (which can allow us to modify it, for instance for reg functions)"""
        pass

    def on_backward_end(self, **kwargs: Any) -> None:
        "Called after backprop but before optimizer step. Useful for true weight decay in AdamW."
        pass

    def on_step_end(self, **kwargs: Any) -> None:
        "Called after the step of the optimizer but before the gradients are zeroed."
        pass

    def on_batch_end(self, **kwargs: Any) -> None:
        "Called at the end of the batch."
        pass

    def on_epoch_end(self, **kwargs: Any) -> bool:
        "Called at the end of an epoch."
        return False

    def on_train_end(self, **kwargs: Any) -> None:
        "Useful for cleaning up things and saving files/models."
        pass


class SmoothenValue():
    "Create a smooth moving average for a value (loss, etc)."

    def __init__(self, beta: float):
        "Create smoother for value, beta should be 0<beta<1."
        self.beta, self.n, self.mov_avg = beta, 0, 0

    def add_value(self, val: float) -> None:
        "Add current value to calculate updated smoothed value."
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)


CallbackList = Collection[Callback]


def _get_init_state():
    return {'epoch': 0, 'iteration': 0, 'num_batch': 0}


@dataclass
class CallbackHandler:
    "Manage all of the registered callback objects, smoothing loss by momentum `beta`."
    callbacks: CallbackList = None
    metrics: CallbackList = None
    beta: float = 0.98

    def __post_init__(self) -> None:
        "Initialize smoother and learning stats."
        self.callbacks = if_none(self.callbacks, [])
        self.metrics = if_none(self.metrics, [])
        self.metrics = [(met if isinstance(met, Callback) else AverageMetric(met)) for met in self.metrics]
        self.callbacks = sorted(self.callbacks, key=lambda o: getattr(o, '_order', 0))
        self.smoothener = SmoothenValue(self.beta)
        self.state_dict: Dict[str, Union[int, float, Tensor]] = _get_init_state()

    def __call__(self, cb_name, call_mets=True, **kwargs) -> None:
        "Call through to all of the `CallbackHandler` functions."
        if call_mets: [getattr(met, f'on_{cb_name}')(**self.state_dict, **kwargs) for met in self.metrics]
        return [getattr(cb, f'on_{cb_name}')(**self.state_dict, **kwargs) for cb in self.callbacks]

    def on_train_begin(self, epochs: int, pbar: PBar, metrics: MetricFuncList) -> None:
        "About to start learning."
        self.state_dict = _get_init_state()
        self.state_dict['n_epochs'], self.state_dict['pbar'], self.state_dict['metrics'] = epochs, pbar, metrics
        names = [(met.name if hasattr(met, 'name') else camel2snake(met.__class__.__name__)) for met in self.metrics]
        self('train_begin', metrics_names=names)

    def on_epoch_begin(self) -> None:
        "Handle new epoch."
        self.state_dict['num_batch'] = 0
        self('epoch_begin')

    def on_batch_begin(self, xb: Tensor, yb: Tensor, train: bool = True) -> None:
        "Handle new batch `xb`,`yb`."
        self.state_dict['last_input'], self.state_dict['last_target'] = xb, yb
        self.state_dict['train'] = train
        for cb in self.callbacks:
            a = cb.on_batch_begin(**self.state_dict)
            if a is not None: self.state_dict['last_input'], self.state_dict['last_target'] = a
        return self.state_dict['last_input'], self.state_dict['last_target']

    def on_loss_begin(self, out: Tensor) -> None:
        "Handle start of loss calculation with model output `out`."
        self.state_dict['last_output'] = out
        for cb in self.callbacks:
            a = cb.on_loss_begin(**self.state_dict)
            if a is not None:
                self.state_dict['last_output'] = a
        return self.state_dict['last_output']

    def on_backward_begin(self, loss: Tensor) -> None:
        "Handle gradient calculation on `loss`."
        self.smoothener.add_value(loss.detach().cpu())
        self.state_dict['last_loss'], self.state_dict['smooth_loss'] = loss, self.smoothener.smooth
        for cb in self.callbacks:
            a = cb.on_backward_begin(**self.state_dict)
            if a is not None:
                self.state_dict['last_loss'] = a
        return self.state_dict['last_loss']

    def on_backward_end(self) -> None:
        "Handle end of gradient calculation."
        self('backward_end', False)

    def on_step_end(self) -> None:
        "Handle end of optimization step."
        self('step_end', False)

    def on_batch_end(self, loss: Tensor) -> None:
        "Handle end of processing one batch with `loss`."
        self.state_dict['last_loss'] = loss
        stop = np.any(self('batch_end', not self.state_dict['train']))
        if self.state_dict['train']:
            self.state_dict['iteration'] += 1
            self.state_dict['num_batch'] += 1
        return stop

    def on_epoch_end(self, val_loss: Tensor) -> bool:
        "Epoch is done, process `val_metrics`."
        self.state_dict['last_metrics'] = [val_loss] if val_loss is not None else None
        self.state_dict['epoch'] += 1
        if not self.state_dict['train']:
            for met in self.metrics:
                met.on_epoch_end(**self.state_dict)
                self.state_dict['last_metrics'].append(met.metric)
        return np.any(self('epoch_end', False))

    def on_train_end(self, exception: Union[bool, Exception]) -> None:
        "Handle end of training, `exception` is an `Exception` or False if no exceptions during training."
        self('train_end', exception=exception)


class AverageMetric(Callback):
    "Wrap a `func` in a callback for metrics computation."

    def __init__(self, func):
        # If it's a partial, use func.func
        name = getattr(func, 'func', func).__name__
        self.func, self.name = func, name

    def on_epoch_begin(self, **kwargs):
        self.val, self.count = 0., 0

    def on_batch_end(self, last_output, last_target, train, **kwargs):
        if not is_listy(last_target): last_target = [last_target]
        self.count += last_target[0].size(0)
        self.val += last_target[0].size(0) * self.func(last_output, *last_target).detach().cpu()

    def on_epoch_end(self, **kwargs):
        self.metric = self.val / self.count


def annealing_no(start: Number, end: Number, pct: float) -> Number:
    "No annealing, always return `start`."
    return start


def annealing_linear(start: Number, end: Number, pct: float) -> Number:
    "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start + pct * (end - start)


def annealing_exp(start: Number, end: Number, pct: float) -> Number:
    "Exponentially anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start * (end / start) ** pct


def annealing_cos(start: Number, end: Number, pct: float) -> Number:
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


def do_annealing_poly(start: Number, end: Number, pct: float, degree: Number) -> Number:
    "Helper function for `anneal_poly`."
    return end + (start - end) * (1 - pct) ** degree


def annealing_poly(degree: Number) -> Number:
    "Anneal polynomically from `start` to `end` as pct goes from 0.0 to 1.0."
    return functools.partial(do_annealing_poly, degree=degree)


AnnealFunc = Callable[[Number, Number, float], Number]
StartOptEnd = Union[float, Tuple[float, float]]


class Stepper:
    "Used to \"step\" from start,end (`vals`) over `n_iter` iterations on a schedule defined by `func`"

    def __init__(self, vals: StartOptEnd, n_iter: int, func: Optional[AnnealFunc] = None):
        self.start, self.end = (vals[0], vals[1]) if is_tuple(vals) else (vals, 0)
        self.n_iter = max(1, n_iter)
        if func is None:
            self.func = annealing_linear if is_tuple(vals) else annealing_no
        else:
            self.func = func
        self.n = 0

    def step(self) -> Number:
        "Return next value along annealed schedule."
        self.n += 1
        return self.func(self.start, self.end, self.n / self.n_iter)

    @property
    def is_done(self) -> bool:
        "Schedule completed."
        return self.n >= self.n_iter
