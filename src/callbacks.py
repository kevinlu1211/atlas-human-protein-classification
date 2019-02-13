from dataclasses import dataclass
from collections import defaultdict

import torch
import numpy as np
import pandas as pd

from pytorch_toolbox.fastai_extensions.basic_train import Phase, determine_phase
from pytorch_toolbox.utils import to_numpy
import pytorch_toolbox.fastai.fastai as fastai


class ResultRecorder(fastai.Callback):
    _order = -10

    def __init__(self):
        self.names = []
        self.prob_preds = []
        self.targets = []

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        self.phase = determine_phase(train, last_target)
        self.names.extend(last_target['name'])
        if self.phase == Phase.TRAIN or self.phase == Phase.VAL:
            label = to_numpy(last_target['label'])
            self.targets.extend(label)

    def on_loss_begin(self, last_output, **kwargs):
        prob_pred = to_numpy(torch.sigmoid(last_output))
        self.prob_preds.extend(prob_pred)


class OutputRecorder(fastai.LearnerCallback):
    _order = -10

    def __init__(self, learn, save_path, save_img_fn, save_img=False):
        super().__init__(learn)
        self.save_path = save_path
        self.history = defaultdict(list)
        self.phase = None
        self.current_batch = dict()
        self.save_img_fn = save_img_fn
        self.save_img = save_img

    def on_batch_begin(self, last_input, last_target, epoch, train, **kwargs):
        self.phase = determine_phase(train, last_target)
        self.key = (self.phase.name, epoch)
        if self.save_img:
            inputs = self.save_img_fn(last_input)
            self.current_batch['input'] = inputs
        self.current_batch['name'] = last_target['name']
        if self.phase == Phase.TRAIN or self.phase == Phase.VAL:
            label = to_numpy(last_target['label'])
            self.current_batch['label'] = label

    def on_loss_begin(self, last_output, epoch, **kwargs):
        model_output = to_numpy(last_output)
        prediction_probs = 1 / (1 + np.exp(-model_output))
        self.current_batch['prediction_probs'] = prediction_probs
        prediction = prediction_probs.copy()
        prediction[prediction < 0.5] = 0
        prediction[prediction >= 0.5] = 1
        self.current_batch['prediction'] = prediction

    def on_batch_end(self, **kwargs):
        for loss in self.learn.loss_func.losses:
            name = loss.__class__.__name__
            unreduced_loss = to_numpy(loss.loss)
            # reduced_loss = to_numpy(loss.loss.mean())
            self.current_batch[f"{name}"] = unreduced_loss
            # self.current_batch[f"{name}_reduced"] = reduced_loss
        # prediction = self.current_batch['prediction']
        # label = self.current_batch['label']
        # n_classes = label.shape[-1]
        # indices_to_keep = np.where((prediction == label).sum(axis=1) != n_classes)[0]

        """
        self.current_batch is a dictionary with:
        {
            stat1: [stat1_for_sample_1, stat1_for_sample_2, ...] 
            stat2: [stat2_for_sample_1, stat2_for_sample_2, ...]
        }
        """

        # Get the keys, and the array of values associated with each key
        stat_names, stat_values = zip(*self.current_batch.items())

        # zip the array of values so each element in the zip has [stat1_for_sample_1, stat2_for_sample_1, ...]
        stat_values_for_samples = zip(*stat_values)

        for stat_values_for_sample in stat_values_for_samples:
            sample_to_save = dict()
            for stat_name, stat_value in zip(stat_names, stat_values_for_sample):
                if stat_name == 'input': continue
                sample_to_save[stat_name] = stat_value
            self.history[self.key].append(sample_to_save)

    def on_epoch_end(self, epoch, **kwargs):
        prev_epoch = epoch - 1
        n_cycle = self.learn.n_cycle
        history_save_path = self.save_path / 'training_logs' / f"cycle_{n_cycle}_epoch_{prev_epoch}_train.csv"
        history_save_path.parent.mkdir(exist_ok=True, parents=True)
        history = self.history[('TRAIN', prev_epoch)]
        df = pd.DataFrame(history)
        df.to_csv(history_save_path, index=False)

        history_save_path = self.save_path / 'training_logs' / f"cycle_{n_cycle}_epoch_{prev_epoch}_val.csv"
        history_save_path.parent.mkdir(exist_ok=True, parents=True)
        history = self.history[('VAL', prev_epoch)]
        df = pd.DataFrame(history)
        df.to_csv(history_save_path, index=False)

        model_save_path = self.save_path / 'model_checkpoints' / f"cycle_{n_cycle}_epoch_{prev_epoch}"
        model_save_path.parent.mkdir(exist_ok=True, parents=True)
        self.learn.save(model_save_path)
        self.history = defaultdict(list)


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

    def get_monitor_value(self):
        values = {'train_loss': self.learn.recorder.losses[-1:][0].cpu().numpy(),
                  'val_loss': self.learn.recorder.val_losses[-1:][0]}
        for i, name in enumerate(self.learn.recorder.names[3:]):
            values[name] = self.learn.recorder.metrics[-1:][0][i]
        return values.get(self.monitor)


@dataclass
class SaveModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    every: str = 'improvement'
    name: str = 'bestmodel'

    def __post_init__(self):
        if self.every not in ['improvement', 'epoch']:
            warn(f'SaveModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'
        super().__post_init__()

    def on_epoch_end(self, epoch, **kwargs) -> None:
        if self.every == "epoch":
            self.learn.save(f'{self.name}_{epoch}')
        else:  # every="improvement"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                self.best = current
                self.learn.save(f'{self.name}')

    def on_train_end(self, **kwargs):
        if self.every == "improvement": self.learn.load(f'{self.name}')


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
        current = self.get_monitor_value()
        if current is None: return
        if self.operator(current - self.min_delta, self.best):
            self.best, self.wait = current, 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.opt.lr *= self.factor
                self.wait = 0
                print(f'Epoch {epoch}: reducing lr to {self.opt.lr}')
