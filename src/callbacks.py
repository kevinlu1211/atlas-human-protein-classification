from collections import defaultdict

import torch
import numpy as np
import pandas as pd

from pytorch_toolbox.fastai_extensions.basic_train import Phase, determine_phase
from pytorch_toolbox.utils import to_numpy
import pytorch_toolbox.fastai.fastai as fastai


class ResultRecorder(fastai.LearnerCallback):
    _order = -10

    def __init__(self):
        self.names = []
        self.prob_preds = []
        self.targets = []

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        phase = self.learn.phase
        self.names.extend(last_target['name'])
        if phase == Phase.TRAIN or phase == Phase.VAL:
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
        self.key = None
        self.current_batch = dict()
        self.save_img_fn = save_img_fn
        self.save_img = save_img

    def on_batch_begin(self, last_input, last_target, epoch, train, **kwargs):
        phase = self.learn.phase
        self.key = (phase.name, epoch)
        if self.save_img:
            inputs = self.save_img_fn(last_input)
            self.current_batch['input'] = inputs
        self.current_batch['name'] = last_target['name']
        if phase == Phase.TRAIN or phase == Phase.VAL:
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
            unreduced_loss = to_numpy(loss.unreduced_loss)
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
