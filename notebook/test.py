import sys

sys.path.append("..")

label_to_string = {
    0: 'Nucleoplasm',
    1: 'Nuclear membrane',
    2: 'Nucleoli',
    3: 'Nucleoli fibrillar center',
    4: 'Nuclear speckles',
    5: 'Nuclear bodies',
    6: 'Endoplasmic reticulum',
    7: 'Golgi apparatus',
    8: 'Peroxisomes',
    9: 'Endosomes',
    10: 'Lysosomes',
    11: 'Intermediate filaments',
    12: 'Actin filaments',
    13: 'Focal adhesion sites',
    14: 'Microtubules',
    15: 'Microtubule ends',
    16: 'Cytokinetic bridge',
    17: 'Mitotic spindle',
    18: 'Microtubule organizing center',
    19: 'Centrosome',
    20: 'Lipid droplets',
    21: 'Plasma membrane',
    22: 'Cell junctions',
    23: 'Mitochondria',
    24: 'Aggresome',
    25: 'Cytosol',
    26: 'Cytoplasmic bodies',
    27: 'Rods & rings'
}

import time
import pickle
from pathlib import Path
from functools import partial
import random
from collections import defaultdict

import cv2
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import torch
import torch.utils.data
import torch.nn as nn
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm

from src.data import ProteinClassificationDataset
from src.data import DataPaths, Image, open_rgby
from src.models import (
    cbam_resnet34_four_channel_input, cbam_resnet50_four_channel_input, resnet50_four_channel_input,
    resnet34_four_channel_input, cbam_resnet101_four_channel_input, resnet152_four_channel_input
)

from pytorch_toolbox.fastai import fastai
from pytorch_toolbox.utils.core import to_numpy
from pytorch_toolbox.fastai.fastai import vision
from pytorch_toolbox.vision.transforms import simple_aug
from pytorch_toolbox.vision.utils import normalize, denormalize, tensor2img
from pytorch_toolbox.fastai_extensions.loss import LossWrapper, FocalLoss
from pytorch_toolbox.fastai_extensions.basic_data import DataBunch
from pytorch_toolbox.fastai_extensions.callbacks import NameExtractionTrainer, GradientClipping

config = yaml.load("../configs/iafoss_resnet34.yml")

# Get paths for training
train_paths = [Path(DataPaths.TRAIN_IMAGES, img_id) for img_id in
               np.unique([p.name[:36] for p in DataPaths.TRAIN_IMAGES.glob("*")])]
test_paths = [Path(DataPaths.TEST_IMAGES, img_id) for img_id in
              np.unique([p.name[:36] for p in DataPaths.TEST_IMAGES.glob("*")])]

# Generate training data
labels_df = pd.read_csv(DataPaths.TRAIN_LABELS)
labels_df['Target'] = [[int(i) for i in s.split()] for s in labels_df['Target']]

train_labels_one_hot = []
train_labels = labels_df['Target']
for labels in tqdm_notebook(train_labels):
    one_hot = np.zeros(28)
    for label in labels:
        one_hot[label] = 1
    train_labels_one_hot.append(one_hot.astype(np.float32))

train_paths = np.array(sorted([Path(DataPaths.TRAIN_IMAGES, img_id) for img_id in
                               np.unique([p.name[:36] for p in DataPaths.TRAIN_IMAGES.glob("*")])],
                              key=lambda p: p.name))
labels_df = labels_df.sort_values(["Id"], ascending=[True])
assert np.all(np.array([p.name for p in train_paths]) == labels_df["Id"])
train_labels = labels_df["Target"].values
assert len(train_paths) == len(train_labels)

# Data splitting
shuffle_split_method = partial(ShuffleSplit(n_splits=1, test_size=0.1, random_state=42).split, X=train_paths)


# Transformation / normalization of images
def albumentations_transform_wrapper(image, augment_fn):
    augmentation = augment_fn(image=image.px)
    return augmentation['image']


four_channel_image_net_stats = {
    'mean': [0.485, 0.456, 0.406, 0.485],
    'sd': [0.229, 0.224, 0.224, 0.229]
}
four_channel_image_net_normalize = partial(normalize, **four_channel_image_net_stats)
four_channel_image_net_denormalize = partial(denormalize, **four_channel_image_net_stats)
augment_fn = partial(albumentations_transform_wrapper, augment_fn=simple_aug(p=1))


def create_data_bunch(split_method, augment_fn, normalize_fn, num_workers=fastai.defaults.cpus, train_bs=64,
                      val_bs=None, test_bs=None):
    train_idx, val_idx = next(iter(split_method()))
    if val_bs is None: val_bs = train_bs * 2
    if test_bs is None: test_bs = train_bs * 2


    train_ds = ProteinClassificationDataset(inputs=np.array(train_paths)[train_idx],
                                            image_cached=False,
                                            labels=np.array(train_labels_one_hot)[train_idx],
                                            augment_fn=augment_fn,
                                            normalize_fn=normalize_fn)
    val_ds = ProteinClassificationDataset(inputs=np.array(train_paths)[val_idx],
                                          image_cached=False,
                                          labels=np.array(train_labels_one_hot)[val_idx],
                                          normalize_fn=normalize_fn)
    test_ds = ProteinClassificationDataset(inputs=np.array(test_paths),
                                           image_cached=False,
                                           normalize_fn=normalize_fn)

    data = DataBunch.create(train_ds, val_ds, test_ds,
                            num_workers=num_workers,
                            collate_fn=ProteinClassificationDataset.collate_fn,
                            train_bs=train_bs,
                            val_bs=val_bs,
                            test_bs=test_bs)
    return data


data = create_data_bunch(split_method=shuffle_split_method,
                         augment_fn=augment_fn,
                         normalize_fn=four_channel_image_net_normalize,
                         # normalize_fn=None,
                         train_bs=32,
                         val_bs=32,
                         test_bs=32)

batch = next(iter(data.train_dl))
inp, name, label = batch[0], batch[1]['name'], batch[1]['label']
print("Input shape:")
print(inp.shape)

# Initialize the model
import pytorch_toolbox.fastai.fastai as fastai
from pytorch_toolbox.fastai_extensions.basic_train import Learner

# model = cbam_resnet34_four_channel_input()
# model = resnet34_four_channel_input(pretrained=False)
# model = resnet34_four_channel_input_v1(pretrained=True)
model = resnet50_four_channel_input(pretrained=True)
# model = cbam_resnet50_four_channel_input(pretrained=True)
# model = cbam_resnet101_four_channel_input(pretrained=False)
# model = resnet152_four_channel_input(pretrained=True)
n_starting_layers = len(fastai.flatten_model(model[:6]))
n_middle_layers = len(fastai.flatten_model(model[6:9]))
n_head = len(fastai.flatten_model(model[9:]))
layer_groups = fastai.split_model_idx(model, [n_starting_layers, n_starting_layers + n_middle_layers])
print(model)


class OutputRecorder(fastai.LearnerCallback):
    _order = -10

    def __init__(self, learn):
        super().__init__(learn)
        self.history = defaultdict(list)
        self.phase = None
        self.current_batch = dict()

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
        inputs = tensor2img(last_input, denorm_fn=four_channel_image_net_denormalize)
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
        # indices_to_keep_losses = np.argsort(loss.sum(axis=1))[:int(len(label) * 0.1)]
        # if self.phase == "VAL":
        for idx in indices_to_keep:
            sample_to_save = dict()
            for k, v in self.current_batch.items():
                if k != "input":
                    sample_to_save[k] = v[idx]
            self.history[self.key].append(sample_to_save)

    def on_epoch_end(self, epoch, **kwargs):
        save_path = Path(f"results/{time.strftime('%Y%m%d-%H%M%S')}/epoch_{epoch}.csv")
        save_path.parent.mkdir(exist_ok=True, parents=True)
        history = self.history[self.key]
        df = pd.DataFrame(history)
        df.to_csv(save_path, index=False)
        self.history = defaultdict(list)


def accuracy(preds, targs, th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds == targs).float().mean()


learner = Learner(data,
                  layer_groups=layer_groups,
                  model=model,
                  loss_func=LossWrapper([
                      FocalLoss()
                  ]),
                  callbacks=[NameExtractionTrainer()],
                  callback_fns=[OutputRecorder, GradientClipping],
                  metrics=[accuracy])


def iafoss_training_scheme(learner):
    lr = 2e-2
    lrs = np.array([lr / 10, lr / 3, lr])
    learner.unfreeze_layer_groups(2)
    learner.fit(epochs=1, lr=[0, 0, lr])
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=2, max_lr=lrs / 4)
    learner.fit_one_cycle(cyc_len=2, max_lr=lrs / 4)
    learner.fit_one_cycle(cyc_len=2, max_lr=lrs / 4)
    learner.fit_one_cycle(cyc_len=2, max_lr=lrs / 4)
    learner.fit_one_cycle(cyc_len=4, max_lr=lrs / 4)
    learner.fit_one_cycle(cyc_len=4, max_lr=lrs / 4)
    learner.fit_one_cycle(cyc_len=8, max_lr=lrs / 16)
    # learner.fit_one_cycle(cyc_len=12, max_lr=lrs/4)

def training_scheme_1(learner):
    lr = 2e-2
    lrs = np.array([lr / 10, lr / 3, lr])
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=50, max_lr=lrs / 4)
    # learner.fit_one_cycle(cyc_len=8, max_lr=lrs/16)


def training_scheme_2(learner):
    lr = 2e-2
    lrs = [lr] * 3
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=50, max_lr=lrs)

def training_scheme_3(learner):
    lr = 2e-3
    lrs = np.array([lr / 10, lr / 3, lr])
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=50, max_lr=lrs / 4)


print("training_scheme_3")
training_scheme_3(learner)

# learner.load("20181204-172216_iafoss")
# lr = 2e-2
# lrs=np.array([lr/10,lr/3,lr])
# learner.unfreeze()
# learner.fit_one_cycle(cyc_len=5, max_lr=lrs/100)

save_name = f"{time.strftime('%Y%m%d-%H%M%S')}"
learner.save(save_name)


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


learner.load(save_name)
res_recorder = ResultRecorder()
learner.predict_on_dl(dl=learner.data.valid_dl, callbacks=[res_recorder])

from sklearn.metrics import f1_score
import scipy.optimize as opt


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def F1_soft(preds, targs, th=0.5, d=50.0):
    preds = sigmoid_np(d * (preds - th))
    targs = targs.astype(np.float)
    score = 2.0 * (preds * targs).sum(axis=0) / ((preds + targs).sum(axis=0) + 1e-6)
    return score


def fit_val(x, y):
    params = 0.5 * np.ones(28)
    wd = 1e-5
    error = lambda p: np.concatenate((F1_soft(x, y, p) - 1.0,
                                      wd * (p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)
    return p


pred_probs = np.stack(res_recorder.prob_preds)
targets = np.stack(res_recorder.targets)
print(targets.shape)
print(pred_probs.shape)

th = fit_val(pred_probs, targets)
th[th < 0.1] = 0.1
print('Thresholds: ', th)
print('F1 macro: ', f1_score(targets, pred_probs > th, average='macro'))
print('F1 macro (th = 0.5): ', f1_score(targets, pred_probs > 0.5, average='macro'))
print('F1 micro: ', f1_score(targets, pred_probs > th, average='micro'))

learner.load(save_name)
res_recorder = ResultRecorder()
learner.predict_on_dl(dl=learner.data.test_dl, callbacks=[res_recorder])

names = np.stack(res_recorder.names)
pred_probs = np.stack(res_recorder.prob_preds)
print(names.shape)
print(pred_probs.shape)

predicted = []
for pred in tqdm_notebook(pred_probs):
    classes = [str(c) for c in np.where(pred > th)[0]]
    if len(classes) == 0:
        classes = [str(np.argmax(pred[0]))]
    predicted.append(" ".join(classes))

submission_df = pd.DataFrame({
    "Id": names,
    "Predicted": predicted
})

submission_df.to_csv(f"{save_name}_submission.csv", index=False)
