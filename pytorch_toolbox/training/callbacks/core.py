from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from numbers import Number
from typing import Any, Collection, Dict, Union, Optional

from torch import Tensor
import numpy as np

from pytorch_toolbox.training.defaults import PBar, MetricFuncList, StartOptEnd, AnnealFunc
from pytorch_toolbox.utils import if_none, camel2snake, is_listy, is_tuple, Phase

# __all__ = ["Callback", "CallbackHandler", "LearnerCallback", "CallbackList"]


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


CallbackList = Collection[Callback]


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
        self.state_dict: Dict[str, Union[int, float, Tensor]] = self._get_init_state()

    def __call__(self, cb_name, call_mets=True, **kwargs) -> None:
        "Call through to all of the `CallbackHandler` functions."
        if call_mets: [getattr(met, f'on_{cb_name}')(**self.state_dict, **kwargs) for met in self.metrics]
        return [getattr(cb, f'on_{cb_name}')(**self.state_dict, **kwargs) for cb in self.callbacks]

    @staticmethod
    def _get_init_state():
        return {'epoch': 0, 'iteration': 0, 'num_batch': 0}

    def on_train_begin(self, epochs: int, pbar: PBar, metrics: MetricFuncList) -> None:
        "About to start learning."
        self.state_dict = self._get_init_state()
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


class LearnerCallback(Callback):
    "Base class for creating callbacks for a `Learner`."

    def __init__(self, learn):
        self.learn = learn
        if self.cb_name:
            setattr(self.learn, self.cb_name, self)

    @property
    def cb_name(self): return camel2snake(self.__class__.__name__)


@dataclass
class TrackerCallback(LearnerCallback):
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
            values[f"train_{camel2snake(loss_name)}"] = mean_loss
            values["train_loss"] += mean_loss
        for loss_name, loss_values in recorder.loss_history[val_key].items():
            mean_loss = np.mean(loss_values)
            values[f"val_{camel2snake(loss_name)}"] = mean_loss
            values["val_loss"] += mean_loss
        for metric_name, metric_values in recorder.metric_history[val_key].items():
            values[f"val_{camel2snake(metric_name)}"] = np.mean(metric_values)
        return values.get(self.monitor)


class SmoothenValue:
    "Create a smooth moving average for a value (loss, etc)."

    def __init__(self, beta: float):
        "Create smoother for value, beta should be 0<beta<1."
        self.beta, self.n, self.mov_avg = beta, 0, 0

    def add_value(self, val: float) -> None:
        "Add current value to calculate updated smoothed value."
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)


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
    return partial(do_annealing_poly, degree=degree)


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
