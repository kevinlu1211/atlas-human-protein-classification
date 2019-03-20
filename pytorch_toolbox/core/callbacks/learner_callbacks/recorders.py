from collections import defaultdict
from typing import Collection, Any

import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor

from pytorch_toolbox.core.callbacks import LearnerCallback
from pytorch_toolbox.core.callbacks.learner_callbacks.determine_phase import Phase
from pytorch_toolbox.core.defaults import PBar, MetricsList, TensorOrNumberList
from pytorch_toolbox.utils import range_of, to_numpy


class BaseRecorder(LearnerCallback):
    "A `LearnerCallback` that records epoch, loss, opt and metric data during training."
    _order = -10

    def __init__(self, learn):
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

    def __init__(self, learn):
        super().__init__(learn)
        self.loss_history = defaultdict(lambda: defaultdict(list))
        self.metric_history = defaultdict(lambda: defaultdict(list))
        self.phase = None

    @property
    def history(self):
        return {**self.loss_history, **self.metric_history}

    def on_batch_begin(self, train, epoch, last_target, **kwargs):
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
