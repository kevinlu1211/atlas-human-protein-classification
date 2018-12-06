
from collections import defaultdict

import torch
import numpy as np
import pandas as pd

from pytorch_toolbox.utils import to_numpy
from pytorch_toolbox.vision.utils import tensor2img
import pytorch_toolbox.fastai.fastai as fastai

class ResultRecorder(fastai.Callback):
    _order = -10

    def __init__(self):
        self.names = []
        self.prob_preds = []
        self.targets = []

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        if train:
            self.phase = 'TRAIN'
        else:
            label = last_target.get('label')
            if label is not None:
                self.phase = 'VAL'
            else:
                self.phase = 'TEST'
                #         inputs = tensor2img(last_input, denorm_fn=image_net_denormalize)
                #         self.inputs.extend(inputs)
        self.names.extend(last_target['name'])
        if self.phase == 'TRAIN' or self.phase == 'VAL':
            label = to_numpy(last_target['label'])
            self.targets.extend(label)

    def on_loss_begin(self, last_output, **kwargs):
        prob_pred = to_numpy(torch.sigmoid(last_output))
        self.prob_preds.extend(prob_pred)

class OutputRecorder(fastai.LearnerCallback):
    _order = -10

    def __init__(self, learn, save_path, save_img_fn):
        super().__init__(learn)
        self.save_path = save_path
        self.history = defaultdict(list)
        self.phase = None
        self.current_batch = dict()
        self.save_img_fn = save_img_fn

    def on_batch_begin(self, last_input, last_target, epoch, train, **kwargs):
        if train:
            self.phase = 'TRAIN'
        else:
            label = last_target.get('label')
            if label is not None:
                self.phase = 'VAL'
            else:
                self.phase = 'TEST'
        self.key = (self.phase, epoch)
        inputs = self.save_img_fn(last_input)
        self.current_batch['input'] = inputs
        self.current_batch['name'] = last_target['name']
        if self.phase == 'TRAIN' or self.phase == 'VAL':
            label = to_numpy(last_target['label'])
            self.current_batch['label'] = label

    def on_loss_begin(self, last_output, epoch, **kwargs):
        model_output = to_numpy(last_output)
        self.current_batch['prediction_probs'] = model_output
        prediction = model_output.copy()
        prediction[prediction < 0.5] = 0
        prediction[prediction >= 0.5] = 1
        self.current_batch['prediction'] = prediction

    def on_batch_end(self, **kwargs):
        loss = to_numpy(self.learn.loss_func.focal_loss.loss)
        self.current_batch['loss'] = loss
        prediction = self.current_batch['prediction']
        label = self.current_batch['label']
        n_classes = label.shape[-1]
        indices_to_keep = np.where((prediction == label).sum(axis=1) != n_classes)[0]
        if self.phase == "VAL":
            for idx in indices_to_keep:
                sample_to_save = dict()
                for k, v in self.current_batch.items():
                    if k != "input":
                        sample_to_save[k] = v[idx]
                self.history[self.key].append(sample_to_save)

    def on_epoch_end(self, epoch, **kwargs):
        save_path = self.save_path / 'training_logs' / f"epoch_{epoch}.csv"
        save_path.parent.mkdir(exist_ok=True, parents=True)
        history = self.history[self.key]
        df = pd.DataFrame(history)
        df.to_csv(save_path, index=False)
        self.history = defaultdict(list)
