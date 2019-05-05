import logging
from collections import defaultdict
from typing import Collection, Any
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor
from tensorboardX import SummaryWriter

from pytorch_toolbox.callbacks import LearnerCallback
from pytorch_toolbox.defaults import PBar, MetricsList, TensorOrNumberList, Callable, Optional
from pytorch_toolbox.utils import range_of, to_numpy, camel2snake, Phase


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

    @property
    def history(self):
        return {**self.loss_history, **self.metric_history}

    def on_batch_begin(self, train, epoch, last_target, phase, **kwargs):
        super().on_batch_begin(train, **kwargs)
        self.key = (phase.name, epoch)

    def on_batch_end(self, phase, **kwargs):
        super().on_batch_end(**kwargs)
        # self.optimizer_hyperparameters_for_current_batch = self._create_optimizer_hyperparameters_for_current_batch()
        self.loss_for_current_batch = self._record_loss_values_for_every_sample_in_batch(phase)
        self._update_loss_history(self.loss_for_current_batch)

    def _record_loss_values_for_every_sample_in_batch(self, phase):
        per_sample_loss_values_for_current_batch = dict()
        all_losses = []
        for loss in self.learn.loss_func.losses:
            name = f"{phase.name}/{camel2snake(loss.__class__.__name__)}"
            per_sample_loss = to_numpy(loss.per_sample_loss)
            per_sample_loss_values_for_current_batch[name] = per_sample_loss
            all_losses.append(per_sample_loss)
        total_loss_name = f"{phase.name}/total_loss"
        per_sample_loss_values_for_current_batch[total_loss_name] = np.sum(all_losses, axis=0)

        return per_sample_loss_values_for_current_batch

    def _update_loss_history(self, loss_for_current_batch):
        for name, loss_value in loss_for_current_batch.items():
            self.loss_history[self.key][name].extend(loss_value)

    def on_epoch_end(self, epoch, num_batch, smooth_loss, last_metrics, phase, **kwargs):
        super().on_epoch_end(epoch, num_batch, smooth_loss, last_metrics, **kwargs)
        if phase == Phase.VAL:
            metric_names = self.names[3:]
            for name, metric in zip(metric_names, self.metrics[epoch - 1]):
                self.metric_history[self.key][camel2snake(name)].append(metric.item())

    def get_losses_and_metrics_for_epoch(self, epoch, with_mean=True):
        losses_and_metrics = {}
        losses_and_metrics.update(**self.get_losses_for_epoch_with_phase(epoch, Phase.TRAIN, with_mean=with_mean))
        losses_and_metrics.update(**self.get_losses_for_epoch_with_phase(epoch, Phase.VAL, with_mean=with_mean))
        losses_and_metrics.update(**self.get_metrics_for_epoch(epoch, with_mean=with_mean))
        return losses_and_metrics

    def get_losses_for_epoch(self, epoch, with_mean=True):
        losses = {}
        losses.update(**self.get_losses_for_epoch_with_phase(epoch, Phase.TRAIN, with_mean=with_mean))
        losses.update(**self.get_losses_for_epoch_with_phase(epoch, Phase.VAL, with_mean=with_mean))
        return losses

    def get_losses_for_epoch_with_phase(self, epoch, phase, with_mean=True):
        losses = {}
        phase_and_epoch_key = (phase.name, epoch)
        for name, values in self.loss_history[phase_and_epoch_key].items():
            if with_mean:
                values = np.mean(values)
            losses[f"{name}"] = values
        return losses

    def get_metrics_for_epoch(self, epoch, with_mean=True):
        metrics = {}
        val_key = (Phase.VAL.name, epoch)
        for name, values in self.metric_history[val_key].items():
            if with_mean:
                values = np.mean(values)
            metrics[f"{name}"] = values
        return metrics


# TODO: allow recording by batch
class TensorBoardRecorder(LearnerCallback):
    _order = 10

    def __init__(self, learn, save_path_creator: Optional[Callable], file_name="tensorboard", per_batch=False):
        super().__init__(learn)
        log_path = Path(
            self.learn.path if save_path_creator is None else save_path_creator()) / f"{file_name}.log"
        logging.info(f"To see tensorboard: tensorboard --purge_orphaned_data false --logdir {log_path}")
        self.tb_writer = SummaryWriter(log_dir=str(log_path))
        self.train_step_idx = 0
        self.val_step_idx = 0
        self.per_batch = per_batch

    def on_batch_end(self, epoch, phase, **kwargs):
        if phase == Phase.TRAIN:
            self._record_losses_for_step(phase, self.train_step_idx)
            self._record_optimizer_hyperparameters(self.train_step_idx)
            self.train_step_idx += 1
        if phase == Phase.VAL:
            self._record_losses_for_step(phase, self.val_step_idx)
            self.val_step_idx += 1

    def _record_optimizer_hyperparameters(self, step_idx):
        optimizer = self.learn.opt
        available_optimizer_hyperparameters = optimizer.available_hyperparameters
        optimizer_hyperparameters_tag_scalar_dict = {}

        for hp_name in available_optimizer_hyperparameters:
            if hp_name == "betas":
                beta_1_for_layers, beta_2_for_layers = optimizer.read_val(hp_name)
                for layer_idx, (beta_1, beta_2) in enumerate(zip(beta_1_for_layers, beta_2_for_layers)):
                    optimizer_hyperparameters_tag_scalar_dict[f"layer_{layer_idx}-beta1"] = beta_1
                    optimizer_hyperparameters_tag_scalar_dict[f"layer_{layer_idx}-beta2"] = beta_2

            else:
                for layer_idx, layer_group_val in enumerate(optimizer.read_val(hp_name)):
                    optimizer_hyperparameters_tag_scalar_dict[f"layer_{layer_idx}-{hp_name}"] = layer_group_val
        self.tb_writer.add_scalars("optimizer_hyperparameter", optimizer_hyperparameters_tag_scalar_dict,
                                   global_step=step_idx)

    def _record_losses_for_step(self, phase, step_idx):
        loss_tag_scalar_dict = {loss_name: np.mean(loss_values) for loss_name, loss_values in
                                self.learn.recorder.loss_for_current_batch.items()}
        self.tb_writer.add_scalars(f"{phase.name.lower()}_losses_per_sample",
                                   tag_scalar_dict=loss_tag_scalar_dict,
                                   global_step=step_idx)

    def on_epoch_end(self, epoch, phase, **kwargs):
        prev_epoch = epoch - 1
        self._record_mean_losses_for_epoch(prev_epoch)
        self._record_metrics_for_epoch(prev_epoch)

    def _record_mean_losses_for_epoch(self, epoch):
        self.tb_writer.add_scalars("mean_losses_for_epoch", self.learn.recorder.get_losses_for_epoch(epoch),
                                   global_step=epoch)

    def _record_metrics_for_epoch(self, epoch):
        self.tb_writer.add_scalars("metrics_for_epoch", self.learn.recorder.get_metrics_for_epoch(epoch),
                                   global_step=epoch)


