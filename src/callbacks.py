
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
        # print([lr for lr in self.learn.opt.read_val('lr')])
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
        prediction_probs = 1/(1 + np.exp(-model_output))
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
        prediction = self.current_batch['prediction']
        label = self.current_batch['label']
        n_classes = label.shape[-1]
        indices_to_keep = np.where((prediction == label).sum(axis=1) != n_classes)[0]
        if True or self.phase == "VAL":
            for idx in indices_to_keep:
                sample_to_save = dict()
                for k, v in self.current_batch.items():
                    if k != "input":
                        sample_to_save[k] = v[idx]
                self.history[self.key].append(sample_to_save)

    def on_epoch_end(self, epoch, **kwargs):
        history_save_path = self.save_path / 'training_logs' / f"epoch_{epoch}.csv"
        history_save_path.parent.mkdir(exist_ok=True, parents=True)
        history = self.history[self.key]
        df = pd.DataFrame(history)
        df.to_csv(history_save_path, index=False)
        model_save_path = self.save_path / 'model_checkpoints' / f"epoch_{epoch}"
        model_save_path.parent.mkdir(exist_ok=True, parents=True)
        self.learn.save(model_save_path)
        self.history = defaultdict(list)
